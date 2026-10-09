//! Half reductions use integer buffers to preserve component bits.
//! The `half-reduction` crate performs arithmetic with Rust's primitive `f16` type.
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

    macro_rules! reduction {
        ($first:ident, $middle:ident, $last:ident, $first_sum:ident, $sum:ident, $round:literal $(, $policy:ident)?) => {
            #[kernel($($policy)?)]
            pub fn $first(#[global] values: Slice<u16>, #[item] partial: &mut u32) {
                let size = values.len() / kernel.items();
                let remainder = values.len() % kernel.items();
                let count = size + usize::from(kernel.item_id() < remainder);
                let values = (0..count).map(|i| values[i * kernel.items() + kernel.item_id()]);
                *partial = half_reduction::$first_sum(values);
            }

            #[kernel($($policy)?)]
            pub fn $middle(#[global] values: Slice<u32>, #[item] partial: &mut u32) {
                let size = values.len() / kernel.items();
                let remainder = values.len() % kernel.items();
                let start = kernel.item_id() * size + kernel.item_id().min(remainder);
                let end = start + size + usize::from(kernel.item_id() < remainder);
                *partial = half_reduction::$sum((start..end).map(|i| values[i]));
            }

            #[kernel($($policy)?)]
            pub fn $last(#[global] values: Slice<u32>, #[item] result: &mut u32) {
                let pair = half_reduction::$sum((0..values.len()).map(|i| values[i]));
                *result = if $round { u32::from(half_reduction::round_pair(pair)) } else { pair };
            }
        };
    }
    reduction!(
        f16_first,
        f16_merge,
        f16_finish,
        sum_twosum_f16,
        merge_twosum_f16,
        true
    );
    reduction!(
        df16_first,
        df16_merge,
        df16_finish,
        sum_df16_scalars,
        sum_df16,
        false
    );
    reduction!(
        f16_first_fast,
        f16_merge_fast,
        f16_finish_fast,
        sum_twosum_f16,
        merge_twosum_f16,
        true,
        fast_math
    );
    reduction!(
        df16_first_fast,
        df16_merge_fast,
        df16_finish_fast,
        sum_df16_scalars,
        sum_df16,
        false,
        fast_math
    );
}

pub type ParallelHalf = Box<dyn FnMut(Slice<'_, u16>, SliceMut<'_, u32>) -> Result<()>>;

/// Reduce with up to 8192, 256, then 1 threads. Retain residuals through every pass.
/// Df16 returns both components; TwoSum rounds the final result to f16.
pub fn parallel_half(
    device: Device,
    len: usize,
    df16: bool,
    fast_math: bool,
) -> Result<ParallelHalf> {
    ensure!(len > 0, "parallel reduction requires nonempty input");
    let partials = len.min(8192);
    let reduced = partials.min(256);
    macro_rules! runner {
        ($first:ident, $middle:ident, $last:ident) => {{
            let mut first_out = Buffer::<u32>::zeros(device.clone(), partials)?;
            let mut middle_out = Buffer::<u32>::zeros(device.clone(), reduced)?;
            let first = kernels::$first::builder()?.build(device.clone())?;
            let middle = kernels::$middle::builder()?.build(device.clone())?;
            let last = kernels::$last::builder()?
                .with_threads(1)
                .build(device.clone())?;
            Box::new(move |input: Slice<'_, u16>, output: SliceMut<'_, u32>| {
                ensure!(
                    input.len() == len && output.len() == 1,
                    "reduction buffer lengths changed"
                );
                first.dispatch(input, first_out.as_slice_mut())?;
                middle.dispatch(first_out.as_slice(), middle_out.as_slice_mut())?;
                last.dispatch(middle_out.as_slice(), output)
            }) as ParallelHalf
        }};
    }
    Ok(match (df16, fast_math) {
        (false, false) => runner!(f16_first, f16_merge, f16_finish),
        (false, true) => runner!(f16_first_fast, f16_merge_fast, f16_finish_fast),
        (true, false) => runner!(df16_first, df16_merge, df16_finish),
        (true, true) => runner!(df16_first_fast, df16_merge_fast, df16_finish_fast),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use half_reduction::{
        decode_pair, encode, merge_twosum_f16, round_pair, sum_df16, sum_df16_scalars,
        sum_twosum_f16,
    };

    // Match the GPU's operation order to distinguish shader errors from rounding differences.
    fn reference(values: &[u16], df16: bool) -> u32 {
        let sum = |values: Vec<u32>| {
            if df16 {
                sum_df16(values.into_iter())
            } else {
                merge_twosum_f16(values.into_iter())
            }
        };
        let n = values.len().min(8192);
        let first: Vec<_> = (0..n)
            .map(|i| {
                let values = values.iter().skip(i).step_by(n).copied();
                if df16 {
                    sum_df16_scalars(values)
                } else {
                    sum_twosum_f16(values)
                }
            })
            .collect();
        let m = n.min(256);
        let size = n / m;
        let rem = n % m;
        let middle: Vec<_> = (0..m)
            .map(|i| {
                let start = i * size + i.min(rem);
                let end = start + size + usize::from(i < rem);
                sum(first[start..end].to_vec())
            })
            .collect();
        let pair = sum(middle);
        if df16 {
            pair
        } else {
            u32::from(round_pair(pair))
        }
    }

    #[test]
    fn gpu_half_reduction_tails_and_residuals() -> Result<()> {
        let device = Device::builder().build()?;
        // Cover partial banks, pass boundaries, and uneven chunks.
        for len in [
            1, 7, 127, 128, 129, 1023, 1025, 8191, 8192, 8193, 32771, 65555,
        ] {
            let mut values = vec![encode(0.25); len];
            // At 2^12, f16 cannot retain a 1/4 increment without compensation.
            if len > 1 {
                values[0] = encode(4096.0);
                values[len - 1] = encode(-4096.0);
            }
            let input = Buffer::from(values.clone()).into_device(device.clone())?;
            for df16 in [false, true] {
                let expected = reference(&values, df16);
                let mut output = Buffer::<u32>::zeros(device.clone(), 1)?;
                parallel_half(device.clone(), len, df16, false)?(
                    input.as_slice(),
                    output.as_slice_mut(),
                )?;
                device.wait()?;
                let got = output.into_vec()?[0];
                assert_eq!(got, expected, "len={len} df16={df16}");
                let exact: f64 = values.iter().map(|&x| half_reduction::decode(x)).sum();
                let expected = if df16 {
                    exact
                } else {
                    half_reduction::decode(encode(exact as f32))
                };
                assert_eq!(decode_pair(got), expected, "len={len}");
            }
        }
        Ok(())
    }

    #[test]
    fn gpu_half_subnormal_policy() -> Result<()> {
        let device = Device::builder().build()?;
        // Each input is the smallest f16 subnormal: 2^-24.
        let input = Buffer::from(vec![1u16; 513]).into_device(device.clone())?;
        for df16 in [false, true] {
            for fast in [false, true] {
                let mut output = Buffer::<u32>::zeros(device.clone(), 1)?;
                parallel_half(device.clone(), 513, df16, fast)?(
                    input.as_slice(),
                    output.as_slice_mut(),
                )?;
                device.wait()?;
                let got = decode_pair(output.into_vec()?[0]);
                assert_eq!(got, if fast { 0.0 } else { 513.0 * 2f64.powi(-24) });
            }
        }
        Ok(())
    }
}
