#[path = "two_sum/df32.rs"]
mod df32_bench;

#[path = "two_sum/summary.rs"]
mod summary;
use summary::CaseSummary;

#[cfg(feature = "half")]
#[path = "two_sum/half.rs"]
mod half_bench;

use deimos_numerics::twosum::TwoSum;
use krnl::{
    anyhow::{Context, Result, ensure},
    buffer::{Buffer, Slice},
    device::Device,
};
use krnl_example::kernels::parallel_twosum;
use rand::{Rng, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use std::{
    hint::black_box,
    sync::LazyLock,
    time::{Duration, Instant},
};

const TIMING_RUNS: u32 = 20;
const GPU_WARMUP: Duration = Duration::from_secs(3);

// Match interpn: cache physical-core discovery, then cap it by the Rayon pool.
static PHYSICAL_CORES: LazyLock<usize> = LazyLock::new(num_cpus::get_physical);

fn cpu_threads() -> usize {
    (*PHYSICAL_CORES).min(rayon::current_num_threads()).max(1)
}

fn cpu_compensated_sum(values: &[f32], threads: usize) -> f32 {
    let chunk_size = values.len().div_ceil(threads.max(1)).max(1);
    let partials: Vec<_> = values
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut acc = TwoSum::<f32, 2>::new(0.0);
            for &value in chunk {
                acc.add(value);
            }
            acc.finish()
        })
        .collect();
    // Merge in chunk order, retaining each residual until the final rounding.
    let mut acc = TwoSum::<f32, 2>::new(0.0);
    for (sum, residual) in partials {
        acc.add(sum);
        acc.add(residual);
    }
    let (sum, residual) = acc.finish();
    sum + residual
}

// Warm up once, then average complete runs. GPU callers wait inside the timer.
fn time_runs(mut run: impl FnMut() -> Result<()>) -> Result<Duration> {
    run()?;
    let start = Instant::now();
    for _ in 0..TIMING_RUNS {
        run()?;
    }
    Ok(start.elapsed() / TIMING_RUNS)
}

fn benchmark_parallel(
    input: &Buffer<f32>,
    device: &Device,
    reference: f64,
    cpu_time: Duration,
    upload_time: Duration,
    summary: &mut CaseSummary,
) -> Result<f32> {
    println!(
        "Parallel reduction: 8192 chunk threads -> 256 reduction threads -> one scalar, two banks/thread."
    );
    println!("GPU warm-up per policy: {GPU_WARMUP:?} (excluded from timing).");
    let mut strict_result = 0.0_f32;
    for fast_math in [false, true] {
        let mut run = parallel_twosum(device.clone(), input.len(), fast_math)?;
        let mut output = Buffer::<f32>::zeros(device.clone(), 1)?;
        device.wait()?;
        // A single dispatch does not warm the GPU enough for steady-state
        // timings. Exercise this same procedure before starting the timer.
        let warmup = Instant::now();
        while warmup.elapsed() < GPU_WARMUP {
            run(input.as_slice(), output.as_slice_mut())?;
            device.wait()?;
        }
        let time = time_runs(|| {
            run(input.as_slice(), output.as_slice_mut())?;
            device.wait()?;
            Ok(())
        })?;
        let policy = if fast_math { "fast" } else { "strict" };
        let start = Instant::now();
        let result = output.into_vec()?[0];
        device.wait()?;
        let download = start.elapsed();
        let error = (f64::from(result) - reference).abs();
        ensure!(result.is_finite(), "parallel {policy}: nonfinite result");
        if !fast_math {
            strict_result = result;
            // These datasets should round to the f64 reference. This is a
            // regression check, not an exactness guarantee for all inputs.
            ensure!(
                result == reference as f32,
                "parallel strict: result differs from rounded reference"
            );
        }
        if fast_math {
            println!(
                "  Fast-math differs from strict: {}",
                result.to_bits() != strict_result.to_bits()
            );
        }
        // Combine mean upload/dispatch times with a single scalar readback.
        let total = upload_time + time + download;
        summary.record(
            "TwoSum",
            format!("GPU {policy}"),
            f64::from(result),
            time,
            Some(total),
        );
        println!("  {policy}: result={result:.9e}, absolute error={error:.9e}");
        println!("    scalar download={download:.3?}");
        println!(
            "    dispatch + wait={time:.3?}, with transfers~{total:.3?}, CPU/GPU={:.2}x / {:.2}x",
            cpu_time.as_secs_f64() / time.as_secs_f64(),
            cpu_time.as_secs_f64() / total.as_secs_f64()
        );
    }
    Ok(strict_result)
}

fn run_case(name: &str, values: &[f32], device: &Device) -> Result<(f32, CaseSummary)> {
    println!("\n{name}: {} f32 values -> one f32 scalar", values.len());

    // The CPU and GPU call the same deimos_numerics two-bank procedure.
    // The f64 sum is a reference, not a claim that compensated f32 is error-free.
    let reference: f64 = values.iter().map(|&x| f64::from(x)).sum();
    let mut summary = CaseSummary::new(name, values.len(), reference);
    let threads = cpu_threads();
    let mut cpu_result = 0.0;
    let cpu_time = time_runs(|| {
        cpu_result = black_box(cpu_compensated_sum(black_box(values), threads));
        Ok(())
    })?;
    ensure!(
        cpu_result == reference as f32,
        "CPU result differs from rounded reference"
    );
    let mut naive_result = 0.0;
    let naive_time = time_runs(|| {
        naive_result = black_box(black_box(values).iter().copied().sum::<f32>());
        Ok(())
    })?;
    summary.record(
        "f32",
        "CPU serial".into(),
        f64::from(naive_result),
        naive_time,
        None,
    );
    summary.record(
        "TwoSum",
        format!("CPU {threads}"),
        f64::from(cpu_result),
        cpu_time,
        None,
    );

    // Allocate once and measure uploads into the existing input buffer.
    let start = Instant::now();
    let mut input = Buffer::<f32>::zeros(device.clone(), values.len())?;
    device.wait()?;
    let input_allocation_time = start.elapsed();
    let upload_time = time_runs(|| {
        input.copy_from_slice(&Slice::from(values))?;
        device.wait()?;
        Ok(())
    })?;
    println!("  f64 reference: {reference:.12e}");
    println!("  Reference rounded to f32: {:.9e}", reference as f32);
    for (label, result) in [
        ("CPU plain f32", naive_result),
        ("CPU compensated f32", cpu_result),
    ] {
        let error = (f64::from(result) - reference).abs();
        println!("  {label}: {result:.9e}, absolute error = {error:.9e}");
    }
    println!("Timing (mean of {TIMING_RUNS} runs after warm-up):");
    println!("  CPU plain f32, single thread:       {naive_time:.3?}");
    println!("  CPU compensated f32, {threads} Rayon chunks: {cpu_time:.3?}");
    println!("  Input upload, reused buffer:        {upload_time:.3?}");
    println!("One-time setup:");
    println!("  Input allocation + initialization: {input_allocation_time:.3?}");
    println!("GPU totals with transfers exclude allocation; CPU/GPU ratios >1 mean GPU faster.");
    let strict = benchmark_parallel(
        &input,
        device,
        reference,
        cpu_time,
        upload_time,
        &mut summary,
    )?;
    df32_bench::benchmark(values, &input, device, reference, upload_time, &mut summary)?;
    Ok((strict, summary))
}

fn main() -> Result<()> {
    let device = Device::builder()
        .build()
        .context("No Vulkan device found")?;
    ensure!(device.is_device(), "Expected a Vulkan device");
    println!("Using device: {device:?}");
    println!("TwoSum and Df32: two-bank CPU and three-pass GPU reductions.");
    println!(
        "GPU timings use host wall time; dispatch excludes kernel creation and buffer allocation."
    );
    if cfg!(debug_assertions) {
        println!("Use cargo run --release --example two_sum for performance comparisons.");
    }

    const INPUTS: usize = 10_000_000;
    const SEED: u64 = 42;
    let mut rng = StdRng::seed_from_u64(SEED);
    // Normal values only: differences here cannot be explained by subnormals.
    let random: Vec<f32> = (0..INPUTS)
        .map(|_| {
            let magnitude = rng.random_range(1.0_f32..1_000_000.0);
            if rng.random() { magnitude } else { -magnitude }
        })
        .collect();
    println!("Random seed: {SEED}");
    let (_, random_summary) = run_case(
        "Random signed values, magnitudes [1, 1e6)",
        &random,
        &device,
    )?;

    // Large initial values hide small integer increments in plain f32 arithmetic.
    // Cancel the large values at the end to expose the accumulated lost increments.
    // The integer sum is exact in f64; the final f32 result may need rounding.
    let mut cancellation = vec![16_777_216.0_f32; 8];
    cancellation.extend((0..INPUTS - 16).map(|_| rng.random_range(1..4) as f32));
    cancellation.extend([-16_777_216.0; 8]);
    let expected: f64 = cancellation.iter().map(|&x| f64::from(x)).sum();
    let (result, cancellation_summary) = run_case(
        "Small increments followed by large cancellation",
        &cancellation,
        &device,
    )?;
    assert_eq!(
        result, expected as f32,
        "compensation must recover the small increments up to final f32 rounding"
    );
    let summaries = vec![random_summary, cancellation_summary];
    #[cfg(feature = "half")]
    let summaries = {
        let mut summaries = summaries;
        half_bench::benchmark(INPUTS, &mut rng, &device, &mut summaries)?;
        summaries
    };
    summary::print(&summaries);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::cpu_compensated_sum;

    #[test]
    fn cpu_chunk_reduction_preserves_residuals() {
        for threads in [1, 2, 3, 8] {
            for len in [0, 1, 7, 17, 1025] {
                let mut values = vec![1.0_f32; len];
                if len > 1 {
                    values[0] = 16_777_216.0;
                    values[len - 1] = -16_777_216.0;
                }
                let expected: f64 = values.iter().map(|&x| f64::from(x)).sum();
                assert_eq!(f64::from(cpu_compensated_sum(&values, threads)), expected);
            }
            assert_eq!(
                cpu_compensated_sum(&[f32::from_bits(1); 17], threads).to_bits(),
                17
            );
        }
    }
}
