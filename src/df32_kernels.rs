//! Df32 reductions with separate f32 high/residual buffers at every pass.
use krnl::{
    anyhow::{Result, ensure},
    buffer::{Buffer, Slice, SliceMut},
    device::Device,
    macros::module,
};

#[module]
mod kernels {
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::macros::kernel;
    use num_synth::Df32;

    /// Two independent pair accumulators, merged once at the end. The host
    /// and GPU share this sequence; input order is chosen by their callers.
    #[inline]
    pub fn sum_df32(len: usize, mut value: impl FnMut(usize) -> Df32) -> Df32 {
        let (mut a, mut b) = (Df32::ZERO, Df32::ZERO);
        // An indexed loop avoids Option<Df32>, whose eight-byte alignment
        // makes rust-gpu emit an otherwise unnecessary 64-bit enum tag.
        for i in 0..len / 2 {
            a += value(2 * i);
            b += value(2 * i + 1);
        }
        if !len.is_multiple_of(2) {
            a += value(len - 1);
        }
        a + b
    }

    macro_rules! reduction {
        ($first:ident, $middle:ident, $last:ident $(, $policy:ident)?) => {
            #[kernel($($policy)?)]
            pub fn $first(
                #[global] values: Slice<f32>,
                #[item] hi: &mut f32,
                #[item] lo: &mut f32,
            ) {
                let size = values.len() / kernel.items();
                let remainder = values.len() % kernel.items();
                let count = size + usize::from(kernel.item_id() < remainder);
                (*hi, *lo) = sum_df32(count, |i| Df32::from_f32(values[i * kernel.items() + kernel.item_id()])).to_parts();
            }

            #[kernel($($policy)?)]
            pub fn $middle(
                #[global] highs: Slice<f32>,
                #[global] lows: Slice<f32>,
                #[item] hi: &mut f32,
                #[item] lo: &mut f32,
            ) {
                let size = highs.len() / kernel.items();
                let remainder = highs.len() % kernel.items();
                let start = kernel.item_id() * size + kernel.item_id().min(remainder);
                let end = start + size + usize::from(kernel.item_id() < remainder);
                (*hi, *lo) = sum_df32(end - start, |i| Df32::from_parts(highs[start + i], lows[start + i])).to_parts();
            }

            #[kernel($($policy)?)]
            pub fn $last(
                #[global] highs: Slice<f32>,
                #[global] lows: Slice<f32>,
                #[item] hi: &mut f32,
                #[item] lo: &mut f32,
            ) {
                (*hi, *lo) = sum_df32(highs.len(), |i| Df32::from_parts(highs[i], lows[i])).to_parts();
            }
        };
    }
    reduction!(df32_first, df32_merge, df32_finish);
    // Diagnostic comparison only: fast math can invalidate pair arithmetic.
    reduction!(
        df32_first_fast,
        df32_merge_fast,
        df32_finish_fast,
        fast_math
    );
}

pub use kernels::sum_df32;
pub type ParallelDf32 =
    Box<dyn FnMut(Slice<'_, f32>, SliceMut<'_, f32>, SliceMut<'_, f32>) -> Result<()>>;

/// Allocate scratch and pipelines for the same 8192 -> 256 -> 1 schedule
/// as parallel_twosum. Both output slices must have one element.
pub fn parallel_df32(device: Device, len: usize, fast_math: bool) -> Result<ParallelDf32> {
    ensure!(len > 0, "parallel reduction requires nonempty input");
    let partials = len.min(8192);
    let reduced = partials.min(256);
    let mut highs = Buffer::<f32>::zeros(device.clone(), partials)?;
    let mut lows = Buffer::<f32>::zeros(device.clone(), partials)?;
    let mut reduced_highs = Buffer::<f32>::zeros(device.clone(), reduced)?;
    let mut reduced_lows = Buffer::<f32>::zeros(device.clone(), reduced)?;
    macro_rules! runner {
        ($first:ident, $middle:ident, $last:ident) => {{
            let first = kernels::$first::builder()?.build(device.clone())?;
            let middle = kernels::$middle::builder()?.build(device.clone())?;
            let last = kernels::$last::builder()?
                .with_threads(1)
                .build(device.clone())?;
            Box::new(
                move |input: Slice<'_, f32>, hi: SliceMut<'_, f32>, lo: SliceMut<'_, f32>| {
                    ensure!(
                        input.len() == len && hi.len() == 1 && lo.len() == 1,
                        "reduction buffer lengths changed"
                    );
                    first.dispatch(input, highs.as_slice_mut(), lows.as_slice_mut())?;
                    middle.dispatch(
                        highs.as_slice(),
                        lows.as_slice(),
                        reduced_highs.as_slice_mut(),
                        reduced_lows.as_slice_mut(),
                    )?;
                    last.dispatch(reduced_highs.as_slice(), reduced_lows.as_slice(), hi, lo)
                },
            ) as ParallelDf32
        }};
    }
    Ok(if fast_math {
        runner!(df32_first_fast, df32_merge_fast, df32_finish_fast)
    } else {
        runner!(df32_first, df32_merge, df32_finish)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_synth::Df32;

    fn reference(values: &[f32]) -> Df32 {
        let n = values.len().min(8192);
        let first: Vec<_> = (0..n)
            .map(|i| {
                sum_df32(values.len() / n + usize::from(i < values.len() % n), |j| {
                    Df32::from_f32(values[j * n + i])
                })
            })
            .collect();
        let m = n.min(256);
        let size = n / m;
        let rem = n % m;
        let middle: Vec<_> = (0..m)
            .map(|i| {
                let start = i * size + i.min(rem);
                let end = start + size + usize::from(i < rem);
                sum_df32(end - start, |j| first[start + j])
            })
            .collect();
        sum_df32(middle.len(), |i| middle[i])
    }

    #[test]
    fn gpu_df32_tails_and_residuals() -> Result<()> {
        let device = Device::builder().build()?;
        for len in [
            1, 7, 127, 128, 129, 1023, 1025, 8191, 8192, 8193, 32771, 65555,
        ] {
            let mut values = vec![1.0f32; len];
            if len > 1 {
                values[0] = 16_777_216.0;
                values[len - 1] = -16_777_216.0;
            }
            let expected = reference(&values);
            let exact: f64 = values.iter().map(|&x| f64::from(x)).sum();
            let input = Buffer::from(values).into_device(device.clone())?;
            let mut hi = Buffer::<f32>::zeros(device.clone(), 1)?;
            let mut lo = Buffer::<f32>::zeros(device.clone(), 1)?;
            parallel_df32(device.clone(), len, false)?(
                input.as_slice(),
                hi.as_slice_mut(),
                lo.as_slice_mut(),
            )?;
            device.wait()?;
            let (h, l) = (hi.into_vec()?[0], lo.into_vec()?[0]);
            assert_eq!(
                (h.to_bits(), l.to_bits()),
                (
                    expected.to_parts().0.to_bits(),
                    expected.to_parts().1.to_bits()
                ),
                "len={len}"
            );
            assert_eq!(f64::from(h) + f64::from(l), exact, "len={len}");
        }
        // An uncanceled result requiring a nonzero residual must survive the
        // final pass too, rather than being collapsed to a single f32.
        let input = Buffer::from(vec![16_777_216.0f32, 1.0]).into_device(device.clone())?;
        let mut hi = Buffer::<f32>::zeros(device.clone(), 1)?;
        let mut lo = Buffer::<f32>::zeros(device.clone(), 1)?;
        parallel_df32(device, 2, false)?(input.as_slice(), hi.as_slice_mut(), lo.as_slice_mut())?;
        assert_eq!(hi.into_vec()?, vec![16_777_216.0]);
        assert_eq!(lo.into_vec()?, vec![1.0]);
        Ok(())
    }

    #[test]
    fn gpu_df32_subnormal_policy() -> Result<()> {
        let device = Device::builder().build()?;
        let input = Buffer::from(vec![f32::from_bits(1); 513]).into_device(device.clone())?;
        for fast in [false, true] {
            let mut hi = Buffer::<f32>::zeros(device.clone(), 1)?;
            let mut lo = Buffer::<f32>::zeros(device.clone(), 1)?;
            parallel_df32(device.clone(), 513, fast)?(
                input.as_slice(),
                hi.as_slice_mut(),
                lo.as_slice_mut(),
            )?;
            device.wait()?;
            assert_eq!(hi.into_vec()?[0].to_bits(), if fast { 0 } else { 513 });
            assert_eq!(lo.into_vec()?[0].to_bits(), 0);
        }
        Ok(())
    }
}
