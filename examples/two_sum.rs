use krnl::{
    anyhow::{Context, Result, ensure},
    buffer::{Buffer, Slice},
    device::Device,
};
use krnl_example::kernels::{compensated_sum, parallel_twosum};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::{
    hint::black_box,
    time::{Duration, Instant},
};

const TIMING_RUNS: u32 = 20;
const GPU_WARMUP: Duration = Duration::from_secs(3);

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

fn run_case(name: &str, values: &[f32], device: &Device) -> Result<f32> {
    println!("\n{name}: {} f32 values -> one f32 scalar", values.len());

    // The CPU and GPU call the same deimos_numerics two-bank procedure.
    // The f64 sum is a reference, not a claim that compensated f32 is error-free.
    let reference: f64 = values.iter().map(|&x| f64::from(x)).sum();
    let mut cpu_result = 0.0;
    let cpu_time = time_runs(|| {
        cpu_result = black_box(compensated_sum(black_box(values).iter().copied()));
        Ok(())
    })?;
    let mut naive_result = 0.0;
    let naive_time = time_runs(|| {
        naive_result = black_box(black_box(values).iter().copied().sum::<f32>());
        Ok(())
    })?;

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
    println!("  CPU compensated f32, single thread: {cpu_time:.3?}");
    println!("  Input upload, reused buffer:        {upload_time:.3?}");
    println!("One-time setup:");
    println!("  Input allocation + initialization: {input_allocation_time:.3?}");
    println!("GPU totals with transfers exclude allocation; CPU/GPU ratios >1 mean GPU faster.");
    benchmark_parallel(&input, device, reference, cpu_time, upload_time)
}

fn main() -> Result<()> {
    let device = Device::builder()
        .build()
        .context("No Vulkan device found")?;
    ensure!(device.is_device(), "Expected a Vulkan device");
    println!("Using device: {device:?}");
    println!("deimos_numerics TwoSum: two-bank CPU and three-pass GPU reductions.");
    println!(
        "GPU timings use host wall time; dispatch excludes kernel creation and buffer allocation."
    );
    if cfg!(debug_assertions) {
        println!("Use cargo run --release --example two_sum for performance comparisons.");
    }

    const INPUTS: usize = 1_000_000;
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
    run_case(
        "Random signed values, magnitudes [1, 1e6)",
        &random,
        &device,
    )?;

    // Large initial values hide small integer increments in plain f32 arithmetic.
    // Cancel the large values at the end to expose the accumulated lost increments.
    // Every input and the exact final integer sum fit in f64 (and the final sum in f32).
    let mut cancellation = vec![16_777_216.0_f32; 8];
    cancellation.extend((0..INPUTS - 16).map(|_| rng.random_range(1..4) as f32));
    cancellation.extend([-16_777_216.0; 8]);
    let expected: f64 = cancellation.iter().map(|&x| f64::from(x)).sum();
    let result = run_case(
        "Small increments followed by large cancellation",
        &cancellation,
        &device,
    )?;
    assert_eq!(
        f64::from(result),
        expected,
        "compensation must recover the small increments"
    );
    Ok(())
}
