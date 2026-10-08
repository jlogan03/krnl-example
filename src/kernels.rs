use krnl::{
    anyhow::{Result, bail},
    buffer::{Slice, SliceMut},
    macros::module,
};

/// Vulkan SPIR-V kernels compiled by `krnlc`.
///
/// `krnlc` compiles this `#[module]` in a separate crate without access to its parent scope.
#[module]
mod kernels {
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::macros::kernel;
    use krnl_core::num_traits::Num;

    /// Compute `a*x + b` for a scalar numeric type.
    ///
    /// Define shared helpers inside a `#[module]` or in a `no_std` dependency.
    /// The host can re-export them from the module.
    #[inline]
    pub fn affine_scalar<T: Num>(a: T, b: T, x: T) -> T {
        a * x + b
    }

    /// Compute `a*x + b` for each `f64` input item.
    #[kernel]
    pub fn affine_kernel(#[item] a: f64, #[item] b: f64, #[item] x: f64, #[item] y: &mut f64) {
        *y = affine_scalar(a, b, x);
    }

    // Merge two compensated accumulators and round the sum plus residual to f32.
    // Reassociation can erase compensation. Intermediate overflow is unsupported.
    #[inline]
    pub fn compensated_sum(values: impl IntoIterator<Item = f32>) -> f32 {
        let (sum, residual) = local_sum(values);
        sum + residual
    }

    #[inline]
    fn local_sum(values: impl IntoIterator<Item = f32>) -> (f32, f32) {
        let mut acc = deimos_numerics::twosum::TwoSum::<f32, 2>::new(0.0);
        for value in values {
            acc.add(value);
        }
        acc.finish()
    }

    // Both policies use three passes. Each item owns its output.
    // No pass needs shared memory, barriers, or unsafe indexing.
    macro_rules! parallel_kernels {
        ($first:ident, $middle:ident, $last:ident $(, $policy:ident)?) => {
            #[kernel($($policy)?)]
            pub fn $first(
                #[global] values: Slice<f32>,
                #[item] sum: &mut f32,
                #[item] residual: &mut f32,
            ) {
                // Neighboring threads read neighboring elements so memory requests can coalesce.
                // Distribute the tail so each input is read once.
                let size = values.len() / kernel.items();
                let remainder = values.len() % kernel.items();
                let count = size + usize::from(kernel.item_id() < remainder);
                let chunk = (0..count).map(|i| values[i * kernel.items() + kernel.item_id()]);
                (*sum, *residual) = local_sum(chunk);
            }

            #[kernel($($policy)?)]
            pub fn $middle(
                #[global] sums: Slice<f32>,
                #[global] residuals: Slice<f32>,
                #[item] sum: &mut f32,
                #[item] residual: &mut f32,
            ) {
                // Merge partial pairs and retain both components for the final pass.
                let size = sums.len() / kernel.items();
                let remainder = sums.len() % kernel.items();
                let start = kernel.item_id() * size + kernel.item_id().min(remainder);
                let end = start + size + usize::from(kernel.item_id() < remainder);
                let mut acc = deimos_numerics::twosum::TwoSum::<f32, 2>::new(0.0);
                for i in start..end {
                    acc.add(sums[i]);
                    acc.add(residuals[i]);
                }
                (*sum, *residual) = acc.finish();
            }

            #[kernel($($policy)?)]
            pub fn $last(
                #[global] sums: Slice<f32>,
                #[global] residuals: Slice<f32>,
                #[item] result: &mut f32,
            ) {
                // Merge all partial pairs before rounding to preserve compensation.
                let mut acc = deimos_numerics::twosum::TwoSum::<f32, 2>::new(0.0);
                for i in 0..sums.len() {
                    acc.add(sums[i]);
                    acc.add(residuals[i]);
                }
                let (sum, residual) = acc.finish();
                *result = sum + residual;
            }
        };
    }
    parallel_kernels!(twosum_parallel, twosum_merge, twosum_finish);
    // Fast-math may reorder compensation arithmetic and flush subnormals.
    parallel_kernels!(
        twosum_parallel_incorrect,
        twosum_merge_incorrect,
        twosum_finish_incorrect,
        fast_math
    );

    #[kernel]
    pub fn fma_strict(
        #[item] a: f32,
        #[item] b: f32,
        #[item] c: f32,
        #[item] separate: &mut f32,
        #[item] fused: &mut f32,
    ) {
        use krnl_core::num_traits::Float;
        *separate = a * b + c;
        *fused = a.mul_add(b, c);
    }
}

pub use kernels::{affine_kernel, affine_scalar, compensated_sum};

/// A reduction that owns its scratch buffers and pipelines.
pub type ParallelTwoSum = Box<dyn FnMut(Slice<'_, f32>, SliceMut<'_, f32>) -> Result<()>>;

/// Allocate buffers and pipelines for a reduction with two accumulators per invocation.
/// The caller supplies an output buffer with one element.
pub fn parallel_twosum(
    device: krnl::device::Device,
    len: usize,
    fast_math: bool,
) -> Result<ParallelTwoSum> {
    use krnl::{anyhow::ensure, buffer::Buffer};
    ensure!(len > 0, "parallel reduction requires nonempty input");
    const THREADS: usize = 8192;
    const REDUCTION_THREADS: usize = 256;
    // Use one output item per active thread. krnl calculates the workgroup count.
    let partials = len.min(THREADS);
    let mut sums = Buffer::<f32>::zeros(device.clone(), partials)?;
    let mut residuals = Buffer::<f32>::zeros(device.clone(), partials)?;

    let reduced = partials.min(REDUCTION_THREADS);
    let mut reduced_sums = Buffer::<f32>::zeros(device.clone(), reduced)?;
    let mut reduced_residuals = Buffer::<f32>::zeros(device.clone(), reduced)?;

    // Keep scratch buffers and all three pipelines outside the timed dispatches.
    macro_rules! build_runner {
        ($first:ident, $middle:ident, $last:ident) => {{
            let first = kernels::$first::builder()?.build(device.clone())?;
            let middle = kernels::$middle::builder()?.build(device.clone())?;
            let last = kernels::$last::builder()?
                .with_threads(1)
                .build(device.clone())?;
            Box::new(move |input: Slice<'_, f32>, output: SliceMut<'_, f32>| {
                ensure!(
                    input.len() == len && output.len() == 1,
                    "reduction buffer lengths changed"
                );
                first.dispatch(input, sums.as_slice_mut(), residuals.as_slice_mut())?;
                middle.dispatch(
                    sums.as_slice(),
                    residuals.as_slice(),
                    reduced_sums.as_slice_mut(),
                    reduced_residuals.as_slice_mut(),
                )?;
                last.dispatch(
                    reduced_sums.as_slice(),
                    reduced_residuals.as_slice(),
                    output,
                )
            }) as ParallelTwoSum
        }};
    }
    let run = if fast_math {
        build_runner!(
            twosum_parallel_incorrect,
            twosum_merge_incorrect,
            twosum_finish_incorrect
        )
    } else {
        build_runner!(twosum_parallel, twosum_merge, twosum_finish)
    };
    Ok(run)
}

/// Run `y = a*x + b` for slice inputs on a compute device.
///
/// Keep this host function outside `#[module]` because it requires `std`.
pub fn affine_device(a: Slice<f64>, b: Slice<f64>, x: Slice<f64>, y: SliceMut<f64>) -> Result<()> {
    if a.len() != b.len() || a.len() != x.len() || a.len() != y.len() {
        bail!("a, b, x, and y lengths must match");
    }

    // krnl caches kernels per device.
    kernels::affine_kernel::builder()?
        .build(y.device())?
        .dispatch(a, b, x, y)
}

#[cfg(test)]
mod tests {
    use super::kernels;
    use krnl::{anyhow::Result, buffer::Buffer, device::Device};

    #[test]
    fn parallel_reduction_tails() -> Result<()> {
        let device = Device::builder().build()?;
        for len in [
            1, 7, 127, 128, 129, 1023, 1024, 1025, 8191, 8192, 8193, 32_771, 65_555,
        ] {
            // These integers sum exactly in f64. Lengths cover partial banks and uneven chunks.
            let mut values = vec![1.0_f32; len];
            if len > 1 {
                values[0] = 16_777_216.0;
                values[len - 1] = -16_777_216.0;
            }
            let expected: f64 = values.iter().map(|&x| f64::from(x)).sum();
            let input = Buffer::from(values).into_device(device.clone())?;
            let mut output = Buffer::<f32>::zeros(device.clone(), 1)?;
            let mut run = super::parallel_twosum(device.clone(), len, false)?;
            run(input.as_slice(), output.as_slice_mut())?;
            device.wait()?;
            assert_eq!(f64::from(output.into_vec()?[0]), expected, "len={len}");
        }
        // All three stages must retain subnormals under the strict policy.
        let input = Buffer::from(vec![f32::from_bits(1); 513]).into_device(device.clone())?;
        for fast_math in [false, true] {
            let mut output = Buffer::<f32>::zeros(device.clone(), 1)?;
            let mut run = super::parallel_twosum(device.clone(), 513, fast_math)?;
            run(input.as_slice(), output.as_slice_mut())?;
            device.wait()?;
            assert_eq!(
                output.into_vec()?[0].to_bits(),
                if fast_math { 0 } else { 513 }
            );
        }
        Ok(())
    }

    #[test]
    fn compensated_reduction() -> Result<()> {
        let device = Device::builder().build()?;
        // krnl rejects empty storage buffers, so check empty input on the CPU.
        assert_eq!(kernels::compensated_sum(core::iter::empty()), 0.0);
        // Cover partial and full banks. Cancellation loses all 19 increments in a plain sum.
        let mut cancellation = vec![16_777_216.0_f32; 8];
        cancellation.extend([1.0; 19]);
        cancellation.extend([-16_777_216.0; 8]);
        for (values, expected) in [
            (vec![3.0], 3.0),
            (vec![1.0, 2.0, 3.0], 6.0),
            (cancellation, 19.0),
            (vec![f32::from_bits(1); 2], f32::from_bits(2)),
        ] {
            assert_eq!(kernels::compensated_sum(values.iter().copied()), expected);
            let input = Buffer::from(values).into_device(device.clone())?;
            let mut output = Buffer::<f32>::zeros(device.clone(), 1)?;
            let mut run = super::parallel_twosum(device.clone(), input.len(), false)?;
            run(input.as_slice(), output.as_slice_mut())?;
            device.wait()?;
            assert_eq!(output.into_vec()?[0].to_bits(), expected.to_bits());
        }

        // Fast-math flushes subnormals; other permitted changes depend on the driver.
        let input = Buffer::from(vec![f32::from_bits(1); 2]).into_device(device.clone())?;
        let mut output = Buffer::<f32>::zeros(device.clone(), 1)?;
        let mut run = super::parallel_twosum(device.clone(), input.len(), true)?;
        run(input.as_slice(), output.as_slice_mut())?;
        device.wait()?;
        assert_eq!(output.into_vec()?, vec![0.0]);
        Ok(())
    }

    #[test]
    fn explicit_fma_remains_fused() -> Result<()> {
        let device = Device::builder().build()?;
        let a = 1.0 + f32::EPSILON;
        let b = 1.0 - f32::EPSILON;
        let input = |x| Buffer::from(vec![x]).into_device(device.clone());
        let (input_a, input_b, input_c) = (input(a)?, input(b)?, input(-1.0_f32)?);
        let mut separate = Buffer::<f32>::zeros(device.clone(), 1)?;
        let mut fused = Buffer::<f32>::zeros(device.clone(), 1)?;
        kernels::fma_strict::builder()?
            .build(device.clone())?
            .dispatch(
                input_a.as_slice(),
                input_b.as_slice(),
                input_c.as_slice(),
                separate.as_slice_mut(),
                fused.as_slice_mut(),
            )?;
        device.wait()?;
        assert_eq!(separate.into_vec()?, vec![0.0]);
        assert_eq!(fused.into_vec()?, vec![a.mul_add(b, -1.0)]);
        assert_ne!(a.mul_add(b, -1.0), 0.0);
        Ok(())
    }
}
