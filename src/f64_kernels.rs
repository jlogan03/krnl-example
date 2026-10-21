//! Sequential f64 sums within a three-pass reduction.
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

    #[kernel]
    pub fn f64_first(#[global] values: Slice<f64>, #[item] partial: &mut f64) {
        let size = values.len() / kernel.items();
        let remainder = values.len() % kernel.items();
        let count = size + usize::from(kernel.item_id() < remainder);
        let mut sum = 0.0;
        for i in 0..count {
            sum += values[i * kernel.items() + kernel.item_id()];
        }
        *partial = sum;
    }

    #[kernel]
    pub fn f64_merge(#[global] values: Slice<f64>, #[item] partial: &mut f64) {
        let size = values.len() / kernel.items();
        let remainder = values.len() % kernel.items();
        let start = kernel.item_id() * size + kernel.item_id().min(remainder);
        let end = start + size + usize::from(kernel.item_id() < remainder);
        let mut sum = 0.0;
        for i in start..end {
            sum += values[i];
        }
        *partial = sum;
    }

    #[kernel]
    pub fn f64_finish(#[global] values: Slice<f64>, #[item] result: &mut f64) {
        let mut sum = 0.0;
        for i in 0..values.len() {
            sum += values[i];
        }
        *result = sum;
    }
}

pub type ParallelF64 = Box<dyn FnMut(Slice<'_, f64>, SliceMut<'_, f64>) -> Result<()>>;

/// Allocate buffers and pipelines for up to 8192, 256, then 1 reduction threads.
pub fn parallel_f64(device: Device, len: usize) -> Result<ParallelF64> {
    ensure!(len > 0, "parallel reduction requires nonempty input");
    let partials = len.min(8192);
    let reduced = partials.min(256);
    let mut first_out = Buffer::<f64>::zeros(device.clone(), partials)?;
    let mut middle_out = Buffer::<f64>::zeros(device.clone(), reduced)?;

    let first = kernels::f64_first::builder()?.build(device.clone())?;
    let middle = kernels::f64_merge::builder()?.build(device.clone())?;
    let last = kernels::f64_finish::builder()?
        .with_threads(1)
        .build(device.clone())?;
    Ok(Box::new(
        move |input: Slice<'_, f64>, output: SliceMut<'_, f64>| {
            ensure!(
                input.len() == len && output.len() == 1,
                "reduction buffer lengths changed"
            );
            first.dispatch(input, first_out.as_slice_mut())?;
            middle.dispatch(first_out.as_slice(), middle_out.as_slice_mut())?;
            last.dispatch(middle_out.as_slice(), output)
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_f64_tails_and_cancellation() -> Result<()> {
        let device = Device::builder().build()?;
        // Cover both pass boundaries and uneven chunks. f32 would lose these increments.
        for len in [1, 7, 255, 256, 257, 8191, 8192, 8193, 32_771] {
            let mut values = vec![1.0_f64; len];
            if len > 1 {
                values[0] = f64::from(1u32 << 24);
                values[len - 1] = -values[0];
            }
            let expected: f64 = values.iter().sum();
            let input = Buffer::from(values).into_device(device.clone())?;

            let mut output = Buffer::<f64>::zeros(device.clone(), 1)?;
            parallel_f64(device.clone(), len)?(input.as_slice(), output.as_slice_mut())?;
            device.wait()?;
            assert_eq!(output.into_vec()?, vec![expected], "len={len}");
        }
        Ok(())
    }
}
