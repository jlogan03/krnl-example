//! Half inputs use u16 buffers; Df16 partials use packed (hi, lo) u32 buffers.
//! Arithmetic is native f16 in half-reduction, not half::f16's f32 emulation.
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
        ($first:ident, $middle:ident, $last:ident, $partial:ty, $sum:ident $(, $policy:ident)?) => {
            #[kernel($($policy)?)]
            pub fn $first(#[global] values: Slice<u16>, #[item] partial: &mut $partial) {
                let size = values.len() / kernel.items();
                let remainder = values.len() % kernel.items();
                let count = size + usize::from(kernel.item_id() < remainder);
                let values = (0..count).map(|i| values[i * kernel.items() + kernel.item_id()] as $partial);
                *partial = half_reduction::$sum(values);
            }

            #[kernel($($policy)?)]
            pub fn $middle(#[global] values: Slice<$partial>, #[item] partial: &mut $partial) {
                let size = values.len() / kernel.items();
                let remainder = values.len() % kernel.items();
                let start = kernel.item_id() * size + kernel.item_id().min(remainder);
                let end = start + size + usize::from(kernel.item_id() < remainder);
                *partial = half_reduction::$sum((start..end).map(|i| values[i]));
            }

            #[kernel($($policy)?)]
            pub fn $last(#[global] values: Slice<$partial>, #[item] result: &mut u32) {
                *result = half_reduction::$sum((0..values.len()).map(|i| values[i])) as u32;
            }
        };
    }
    reduction!(f16_first, f16_merge, f16_finish, u16, sum_f16);
    reduction!(df16_first, df16_merge, df16_finish, u32, sum_df16);
    reduction!(
        f16_first_fast,
        f16_merge_fast,
        f16_finish_fast,
        u16,
        sum_f16,
        fast_math
    );
    reduction!(
        df16_first_fast,
        df16_merge_fast,
        df16_finish_fast,
        u32,
        sum_df16,
        fast_math
    );
}

pub type ParallelHalf = Box<dyn FnMut(Slice<'_, u16>, SliceMut<'_, u32>) -> Result<()>>;

/// The same 8192 -> 256 -> 1 schedule as parallel_twosum. Each invocation
/// accumulates two banks; packed Df16 partials retain both components.
pub fn parallel_half(
    device: Device,
    len: usize,
    paired: bool,
    fast_math: bool,
) -> Result<ParallelHalf> {
    ensure!(len > 0, "parallel reduction requires nonempty input");
    let partials = len.min(8192);
    let reduced = partials.min(256);
    macro_rules! runner {
        ($ty:ty, $first:ident, $middle:ident, $last:ident) => {{
            let mut first_out = Buffer::<$ty>::zeros(device.clone(), partials)?;
            let mut middle_out = Buffer::<$ty>::zeros(device.clone(), reduced)?;
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
    Ok(match (paired, fast_math) {
        (false, false) => runner!(u16, f16_first, f16_merge, f16_finish),
        (false, true) => runner!(u16, f16_first_fast, f16_merge_fast, f16_finish_fast),
        (true, false) => runner!(u32, df16_first, df16_merge, df16_finish),
        (true, true) => runner!(u32, df16_first_fast, df16_merge_fast, df16_finish_fast),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use half_reduction::{decode_pair, encode, sum_df16, sum_f16};

    // Match the GPU's three passes and operation order, independently of its
    // dispatch implementation. Comparing to a differently ordered CPU sum
    // would mistake legitimate rounding differences for a shader defect.
    fn reference(values: &[u16], paired: bool) -> u32 {
        let sum = |values: Vec<u32>| {
            if paired {
                sum_df16(values.into_iter())
            } else {
                u32::from(sum_f16(values.into_iter().map(|x| x as u16)))
            }
        };
        let n = values.len().min(8192);
        let first: Vec<_> = (0..n)
            .map(|i| {
                sum(values
                    .iter()
                    .skip(i)
                    .step_by(n)
                    .map(|&x| u32::from(x))
                    .collect())
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
        sum(middle)
    }

    #[test]
    fn gpu_half_reduction_tails_and_residuals() -> Result<()> {
        let device = Device::builder().build()?;
        for len in [
            1, 7, 127, 128, 129, 1023, 1025, 8191, 8192, 8193, 32771, 65555,
        ] {
            let mut values = vec![encode(0.25); len];
            if len > 1 {
                values[0] = encode(4096.0);
                values[len - 1] = encode(-4096.0);
            }
            let input = Buffer::from(values.clone()).into_device(device.clone())?;
            for paired in [false, true] {
                let expected = reference(&values, paired);
                let mut output = Buffer::<u32>::zeros(device.clone(), 1)?;
                parallel_half(device.clone(), len, paired, false)?(
                    input.as_slice(),
                    output.as_slice_mut(),
                )?;
                device.wait()?;
                let got = output.into_vec()?[0];
                assert_eq!(got, expected, "len={len} paired={paired}");
                if paired {
                    let exact: f64 = values.iter().map(|&x| half_reduction::decode(x)).sum();
                    assert_eq!(decode_pair(got), exact, "len={len}");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn gpu_half_subnormal_policy() -> Result<()> {
        let device = Device::builder().build()?;
        let input = Buffer::from(vec![1u16; 513]).into_device(device.clone())?;
        for paired in [false, true] {
            for fast in [false, true] {
                let mut output = Buffer::<u32>::zeros(device.clone(), 1)?;
                parallel_half(device.clone(), 513, paired, fast)?(
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
