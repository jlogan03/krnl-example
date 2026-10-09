use super::{CaseSummary, cpu_threads, time_gpu, time_runs};
use krnl::{anyhow::Result, buffer::Buffer, device::Device};
use krnl_example::df32_kernels::{parallel_df32, sum_df32};
use num_synth::Df32;
use rayon::prelude::*;
use std::{hint::black_box, time::Duration};

fn cpu_sum(values: &[f32], threads: usize) -> Df32 {
    let chunk_size = values.len().div_ceil(threads.max(1)).max(1);
    let partials: Vec<_> = values
        .par_chunks(chunk_size)
        .map(|chunk| sum_df32(chunk.len(), |i| Df32::from_f32(chunk[i])))
        .collect();
    sum_df32(partials.len(), |i| partials[i])
}

pub fn benchmark(
    values: &[f32],
    input: &Buffer<f32>,
    device: &Device,
    upload: Duration,
    summary: &mut CaseSummary,
) -> Result<()> {
    let threads = cpu_threads();
    let cpu = time_runs(|| Ok(cpu_sum(black_box(values), threads).to_f64()))?;
    for fast in [false, true] {
        let mut run = parallel_df32(device.clone(), values.len(), fast)?;
        let mut hi = Buffer::<f32>::zeros(device.clone(), 1)?;
        let mut lo = Buffer::<f32>::zeros(device.clone(), 1)?;
        let compute = time_gpu(device, || {
            run(input.as_slice(), hi.as_slice_mut(), lo.as_slice_mut())
        })?;
        // Read both components without normalizing the pair, which could hide shader errors.
        let (result, download) =
            time_runs(|| Ok(f64::from(hi.to_vec()?[0]) + f64::from(lo.to_vec()?[0])))?;
        summary.record(
            "Sequential Df32",
            fast,
            cpu,
            (result, compute),
            upload,
            download,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_df32_chunk_reduction_preserves_residuals() {
        for threads in [1, 2, 3, 8] {
            for len in [0, 1, 7, 17, 1025] {
                let mut values = vec![1.0f32; len];
                if len > 1 {
                    values[0] = 16_777_216.0;
                    values[len - 1] = -16_777_216.0;
                }
                let exact: f64 = values.iter().map(|&x| f64::from(x)).sum();
                assert_eq!(cpu_sum(&values, threads).to_f64(), exact);
            }
            assert_eq!(
                cpu_sum(&[16_777_216.0, 1.0], threads).to_parts(),
                (16_777_216.0, 1.0)
            );
            assert_eq!(
                cpu_sum(&[f32::from_bits(1); 17], threads)
                    .to_f32()
                    .to_bits(),
                17
            );
        }
    }
}
