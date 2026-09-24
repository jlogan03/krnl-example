use super::{CaseSummary, GPU_WARMUP, TIMING_RUNS, cpu_threads, run_case, time_runs};
use half_reduction::{decode, decode_pair, encode, sum_df16, sum_f16};
use krnl::{
    anyhow::{Result, ensure},
    buffer::{Buffer, Slice},
    device::Device,
};
use krnl_example::half_kernels::parallel_half;
use rand::Rng;
use rayon::prelude::*;
use std::{hint::black_box, time::Instant};

fn cpu_sum(values: &[u16], threads: usize, paired: bool) -> u32 {
    let size = values.len().div_ceil(threads.max(1)).max(1);
    if paired {
        let partials: Vec<_> = values
            .par_chunks(size)
            .map(|chunk| sum_df16(chunk.iter().map(|&x| u32::from(x))))
            .collect();
        sum_df16(partials.into_iter())
    } else {
        let partials: Vec<_> = values
            .par_chunks(size)
            .map(|chunk| sum_f16(chunk.iter().copied()))
            .collect();
        u32::from(sum_f16(partials.into_iter()))
    }
}

fn run(name: &str, source: &[f32], device: &Device) -> Result<CaseSummary> {
    // Round once before any timing. All CPU/GPU types see the same quantized
    // mathematical input, including the existing f32 TwoSum comparison.
    let values: Vec<_> = source.iter().map(|&x| encode(x)).collect();
    let reference: f64 = values.iter().map(|&x| decode(x)).sum();
    let singles: Vec<_> = values.iter().map(|&x| decode(x) as f32).collect();
    let (_, mut summary) = run_case(name, &singles, device)?;
    println!(
        "Half variants: {} quantized f16 inputs; reference={reference:.12e}",
        values.len()
    );
    println!(
        "  Plain f16 and Df16 use two banks per invocation; Df16 retains both output components."
    );
    println!("  CPU chunking and GPU interleaving differ; equal result bits are not assumed.");
    let start = Instant::now();
    let mut input = Buffer::<u16>::zeros(device.clone(), values.len())?;
    device.wait()?;
    let allocation = start.elapsed();
    let upload = time_runs(|| {
        input.copy_from_slice(&Slice::from(values.as_slice()))?;
        device.wait()?;
        Ok(())
    })?;
    println!("  Half input allocation={allocation:.3?}; mean upload={upload:.3?}");

    for paired in [false, true] {
        let label = if paired { "Df16" } else { "f16" };
        let threads = cpu_threads();
        let mut cpu_bits = 0;
        let cpu_time = time_runs(|| {
            cpu_bits = black_box(cpu_sum(black_box(&values), threads, paired));
            Ok(())
        })?;
        let cpu_result = decode_pair(cpu_bits);
        ensure!(cpu_result.is_finite(), "CPU {label} overflowed");
        summary.record(label, format!("CPU {threads}"), cpu_result, cpu_time, None);
        println!(
            "  CPU {label}, {threads} Rayon chunks: result={cpu_result:.12e}, absolute error={:.9e}, mean={cpu_time:.3?}",
            (cpu_result - reference).abs()
        );
        let mut strict_bits = 0;
        for fast_math in [false, true] {
            let policy = if fast_math { "fast" } else { "strict" };
            let mut run = parallel_half(device.clone(), values.len(), paired, fast_math)?;
            let mut output = Buffer::<u32>::zeros(device.clone(), 1)?;
            device.wait()?;
            let start = Instant::now();
            while start.elapsed() < GPU_WARMUP {
                run(input.as_slice(), output.as_slice_mut())?;
                device.wait()?;
            }
            let gpu_time = time_runs(|| {
                run(input.as_slice(), output.as_slice_mut())?;
                device.wait()?;
                Ok(())
            })?;
            let start = Instant::now();
            let bits = output.into_vec()?[0];
            device.wait()?;
            let download = start.elapsed();
            let result = decode_pair(bits);
            ensure!(result.is_finite(), "GPU {label}/{policy} overflowed");
            if !fast_math {
                strict_bits = bits;
            }
            let total = upload + gpu_time + download;
            summary.record(
                label,
                format!("GPU {policy}"),
                result,
                gpu_time,
                Some(total),
            );
            println!(
                "  GPU {label}/{policy}: result={result:.12e}, absolute error={:.9e}",
                (result - reference).abs()
            );
            println!(
                "    dispatch + wait={gpu_time:.3?}, download={download:.3?}, with transfers~{total:.3?}, CPU/GPU={:.2}x / {:.2}x",
                cpu_time.as_secs_f64() / gpu_time.as_secs_f64(),
                cpu_time.as_secs_f64() / total.as_secs_f64()
            );
            if fast_math {
                println!("    differs from strict: {}", bits != strict_bits);
            }
        }
    }
    Ok(summary)
}

pub fn benchmark(
    inputs: usize,
    rng: &mut impl Rng,
    device: &Device,
    summaries: &mut Vec<CaseSummary>,
) -> Result<()> {
    println!(
        "\nHalf-range datasets; timings average {TIMING_RUNS} runs, warm-up {GPU_WARMUP:?}/policy."
    );
    let random: Vec<f32> = (0..inputs)
        .map(|_| {
            let x = rng.random_range(0.0625f32..1.0);
            if rng.random() { x } else { -x }
        })
        .collect();
    summaries.push(run(
        "Quantized half random signed magnitudes [1/16, 1)",
        &random,
        device,
    )?);

    // Normal half inputs and bounded totals. Large-value cancellation exposes
    // lost increments without giving the half variants overflowing inputs.
    let mut cancellation = vec![128.0f32; 8];
    cancellation.extend(std::iter::repeat_n(2f32.powi(-14), inputs - 16));
    cancellation.extend([-128.0f32; 8]);
    summaries.push(run(
        "Quantized half small increments followed by cancellation",
        &cancellation,
        device,
    )?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_half_reduction_tails_and_cancellation() {
        for threads in [1, 2, 3, 8] {
            for len in [0, 1, 7, 17, 1025] {
                let mut values = vec![encode(1.0); len];
                if len > 1 {
                    values[0] = encode(4096.0);
                    values[len - 1] = encode(-4096.0);
                }
                let exact: f64 = values.iter().map(|&x| decode(x)).sum();
                assert_eq!(decode_pair(cpu_sum(&values, threads, true)), exact);
            }
        }
        assert_eq!(decode_pair(cpu_sum(&[encode(1.0); 17], 3, false)), 17.0);
        let tiny = [1u16; 17];
        assert_eq!(decode_pair(cpu_sum(&tiny, 3, true)), 17.0 * 2f64.powi(-24));
    }
}
