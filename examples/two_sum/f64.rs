use super::{CaseSummary, cpu_threads, time_gpu, time_runs, upload};
use krnl::{anyhow::Result, buffer::Buffer, device::Device};
use krnl_example::f64_kernels::parallel_f64;
use rayon::prelude::*;
use std::hint::black_box;

fn cpu_sum(values: &[f64], threads: usize) -> f64 {
    let size = values.len().div_ceil(threads.max(1)).max(1);
    let partials: Vec<f64> = values
        .par_chunks(size)
        .map(|chunk| chunk.iter().sum())
        .collect();
    partials.into_iter().sum()
}

pub fn benchmark(values: &[f32], device: &Device, summary: &mut CaseSummary) -> Result<()> {
    let values: Vec<f64> = values.iter().map(|&x| f64::from(x)).collect();
    let threads = cpu_threads();
    let cpu = time_runs(|| Ok(cpu_sum(black_box(&values), threads)))?;
    let (input, upload) = upload(&values, device)?;
    for fast in [false, true] {
        let mut run = parallel_f64(device.clone(), values.len(), fast)?;
        let mut output = Buffer::<f64>::zeros(device.clone(), 1)?;
        let compute = time_gpu(device, || run(input.as_slice(), output.as_slice_mut()))?;
        let (result, download) = time_runs(|| Ok(output.to_vec()?[0]))?;
        summary.record(
            "Sequential f64",
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
    use super::cpu_sum;

    #[test]
    fn cpu_f64_tails_and_cancellation() {
        for threads in [1, 2, 3, 8] {
            // Cover empty input and uneven chunks; 2^24 hides unit increments in f32.
            for len in [0, 1, 7, 17, 1025] {
                let mut values = vec![1.0_f64; len];
                if len > 1 {
                    values[0] = f64::from(1u32 << 24);
                    values[len - 1] = -values[0];
                }
                assert_eq!(cpu_sum(&values, threads), values.iter().sum::<f64>());
            }
        }
    }
}
