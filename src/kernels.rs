use krnl::{
    anyhow::{Result, bail},
    buffer::{Slice, SliceMut},
    macros::module,
};

/// Vulkan SPIRV compute kernels to be compiled with `krnlc`.
///
/// This #[module] scope is extracted to a separate crate to compile,
/// so it doesn't have access to the outer scope in this crate.
#[module]
mod kernels {
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::macros::kernel;
    use krnl_core::num_traits::Num;

    /// Type-generic example scalar kernel.
    ///
    /// Because the krnl `kernels` module can't see its super:: or anything else
    /// not enclosed in a #[module] scope, this generic scalar kernel must be
    /// defined here then exported to super:: instead of the other way around.
    ///
    /// Alternatively, we could have functions like this one stored in another
    /// #[module] scope in this crate, or anywhere in another no_std crate.
    /// They just can't be _both_ inside this crate and outside a #[module] scope.
    #[inline]
    pub fn affine_scalar<T: Num>(a: T, b: T, x: T) -> T {
        a * x + b
    }

    /// Simple example scalar XPU kernel for 64-bit floats.
    #[kernel]
    pub fn affine_kernel(#[item] a: f64, #[item] b: f64, #[item] x: f64, #[item] y: &mut f64) {
        *y = affine_scalar(a, b, x);
    }

    // deimos_numerics stages values across two compensated banks, then merges
    // them. Return one f32 scalar by rounding the final sum plus residual.
    // Reassociation can erase the compensation; intermediate overflow is unsupported.
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

    // Both policies use the same three-pass procedure. Each item owns its output,
    // so no pass needs shared memory, barriers or unsafe indexing.
    macro_rules! parallel_kernels {
        ($first:ident, $middle:ident, $last:ident $(, $policy:ident)?) => {
            #[kernel($($policy)?)]
            pub fn $first(
                #[global] values: Slice<f32>,
                #[item] sum: &mut f32,
                #[item] residual: &mut f32,
            ) {
                // Divide into contiguous chunks, distributing any remainder
                // across the first items so every input is consumed once.
                let size = values.len() / kernel.items();
                let remainder = values.len() % kernel.items();
                let start = kernel.item_id() * size + kernel.item_id().min(remainder);
                let end = start + size + usize::from(kernel.item_id() < remainder);
                let chunk = (start..end).map(|i| values[i]);
                (*sum, *residual) = local_sum(chunk);
            }

            #[kernel($($policy)?)]
            pub fn $middle(
                #[global] sums: Slice<f32>,
                #[global] residuals: Slice<f32>,
                #[item] sum: &mut f32,
                #[item] residual: &mut f32,
            ) {
                // Merge a chunk of partial pairs, preserving both components
                // for the final pass instead of rounding each chunk to a scalar.
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
                // One invocation merges all partial pairs. Rounding each pair
                // to a scalar earlier would discard its compensation.
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

// We can re-export functions from inside a #[module] scope.
pub use kernels::{affine_kernel, affine_scalar, compensated_sum};

/// Reusable dispatch with owned scratch buffers and prebuilt pipelines.
pub type ParallelTwoSum = Box<dyn FnMut(Slice<'_, f32>, SliceMut<'_, f32>) -> Result<()>>;

/// Build a parallel reduction with two banks per invocation. Allocation
/// and pipeline creation happen here; the caller supplies a one-scalar output.
pub fn parallel_twosum(
    device: krnl::device::Device,
    len: usize,
    fast_math: bool,
) -> Result<ParallelTwoSum> {
    use krnl::{anyhow::ensure, buffer::Buffer};
    ensure!(len > 0, "parallel reduction requires nonempty input");
    const THREADS: usize = 8192;
    const REDUCTION_THREADS: usize = 256;
    // One output item per active thread. krnl uses its default workgroup size
    // and calculates enough groups to cover these items.
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
/// This function has to be defined outside the #[module] scope or behind a config flag
/// because it uses stdlib functionality and is not, itself, a #[no_std]-compatible kernel function.
pub fn affine_device(a: Slice<f64>, b: Slice<f64>, x: Slice<f64>, y: SliceMut<f64>) -> Result<()> {
    if a.len() != b.len() || a.len() != x.len() || a.len() != y.len() {
        bail!("a, b, x, and y lengths must match");
    }

    // Kernels are cached per-device internally, so we don't need to wrap this in a LazyCell.
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
            // Integer data makes the f64 reference exact, including after
            // cancellation. Lengths cover bank tails and uneven chunk boundaries.
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
        // Cover a single value, a partial bank, and multiple banks
        // followed by cancellation. The latter loses all 19 units in a plain sum.
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
