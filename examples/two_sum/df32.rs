use super::{GPU_WARMUP, cpu_threads, time_runs};
use krnl::{
    anyhow::{Result, ensure},
    buffer::Buffer,
    device::Device,
};
use krnl_example::df32_kernels::{parallel_df32, sum_df32};
use num_synth::Df32;
use rayon::prelude::*;
use std::{hint::black_box, time::Instant};

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
    reference: f64,
    upload: std::time::Duration,
) -> Result<()> {
    println!("Df32: same f32 input, two pair banks per invocation, pair output retained.");
    let threads = cpu_threads();
    let mut cpu_result = Df32::ZERO;
    let cpu_time = time_runs(|| {
        cpu_result = black_box(cpu_sum(black_box(values), threads));
        Ok(())
    })?;
    ensure!(cpu_result.is_finite(), "CPU Df32 overflowed");
    println!(
        "  CPU Df32, {threads} Rayon chunks: result={:.12e}, absolute error={:.9e}, mean={cpu_time:.3?}",
        cpu_result.to_f64(),
        (cpu_result.to_f64() - reference).abs()
    );
    let mut strict_bits = (0, 0);
    for fast in [false, true] {
        let policy = if fast { "fast" } else { "strict" };
        let mut run = parallel_df32(device.clone(), values.len(), fast)?;
        let mut hi = Buffer::<f32>::zeros(device.clone(), 1)?;
        let mut lo = Buffer::<f32>::zeros(device.clone(), 1)?;
        device.wait()?;
        let warmup = Instant::now();
        while warmup.elapsed() < GPU_WARMUP {
            run(input.as_slice(), hi.as_slice_mut(), lo.as_slice_mut())?;
            device.wait()?;
        }
        let gpu_time = time_runs(|| {
            run(input.as_slice(), hi.as_slice_mut(), lo.as_slice_mut())?;
            device.wait()?;
            Ok(())
        })?;
        let start = Instant::now();
        let (hi, lo) = (hi.into_vec()?[0], lo.into_vec()?[0]);
        device.wait()?;
        let download = start.elapsed();
        // Report the raw returned sum even for fast math. Re-normalizing on
        // the CPU could hide a shader's failure to preserve pair invariants.
        let result = f64::from(hi) + f64::from(lo);
        ensure!(result.is_finite(), "GPU Df32/{policy} overflowed");
        let bits = (hi.to_bits(), lo.to_bits());
        if !fast {
            strict_bits = bits;
        }
        let total = upload + gpu_time + download;
        println!(
            "  GPU Df32/{policy}: result={result:.12e}, absolute error={:.9e}",
            (result - reference).abs()
        );
        println!(
            "    dispatch + wait={gpu_time:.3?}, pair download={download:.3?}, with transfers~{total:.3?}, CPU/GPU={:.2}x / {:.2}x",
            cpu_time.as_secs_f64() / gpu_time.as_secs_f64(),
            cpu_time.as_secs_f64() / total.as_secs_f64()
        );
        if fast {
            println!("    differs from strict: {}", bits != strict_bits);
        }
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
