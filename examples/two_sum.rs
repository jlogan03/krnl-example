#[path = "two_sum/df32.rs"]
mod df32_bench;

#[path = "two_sum/f64.rs"]
mod f64_bench;

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
    scalar::Scalar,
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

// Cache the physical-core count to exclude discovery from timings.
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
    // Merge in chunk order and retain residuals until the final rounding.
    let mut acc = TwoSum::<f32, 2>::new(0.0);
    for (sum, residual) in partials {
        acc.add(sum);
        acc.add(residual);
    }
    let (sum, residual) = acc.finish();
    sum + residual
}

// Average runs after one warm-up. GPU callers include completion waits.
fn time_runs<T>(mut run: impl FnMut() -> Result<T>) -> Result<(T, Duration)> {
    let mut result = run()?;
    let start = Instant::now();
    for _ in 0..TIMING_RUNS {
        result = black_box(run()?);
    }
    Ok((result, start.elapsed() / TIMING_RUNS))
}

fn time_gpu(device: &Device, mut run: impl FnMut() -> Result<()>) -> Result<Duration> {
    device.wait()?;
    let warmup = Instant::now();
    while warmup.elapsed() < GPU_WARMUP {
        run()?;
        device.wait()?;
    }
    let (_, time) = time_runs(|| {
        run()?;
        device.wait()?;
        Ok(())
    })?;
    Ok(time)
}

// Allocate before timing and reuse the input buffer for each upload.
fn upload<T: Scalar>(values: &[T], device: &Device) -> Result<(Buffer<T>, Duration)> {
    let mut input = Buffer::<T>::zeros(device.clone(), values.len())?;
    device.wait()?;
    let (_, time) = time_runs(|| {
        input.copy_from_slice(&Slice::from(values))?;
        device.wait()?;
        Ok(())
    })?;
    Ok((input, time))
}

fn run_case(name: &str, values: &[f32], device: &Device) -> Result<CaseSummary> {
    let reference: f64 = values.iter().map(|&x| f64::from(x)).sum();
    let mut summary = CaseSummary::new(name, values.len(), reference);
    let threads = cpu_threads();
    let cpu = time_runs(|| Ok(f64::from(cpu_compensated_sum(black_box(values), threads))))?;
    ensure!(
        cpu.0 == f64::from(reference as f32),
        "CPU TwoSum differs from rounded reference"
    );
    let (input, upload) = upload(values, device)?;
    for fast in [false, true] {
        let mut run = parallel_twosum(device.clone(), values.len(), fast)?;
        let mut output = Buffer::<f32>::zeros(device.clone(), 1)?;
        let compute = time_gpu(device, || run(input.as_slice(), output.as_slice_mut()))?;
        let (result, download) = time_runs(|| Ok(f64::from(output.to_vec()?[0])))?;
        if !fast {
            // These datasets should match the rounded reference. Other inputs may differ.
            ensure!(
                result == f64::from(reference as f32),
                "GPU TwoSum differs from rounded reference"
            );
        }
        summary.record("TwoSum f32", fast, cpu, (result, compute), upload, download)?;
    }
    f64_bench::benchmark(values, device, &mut summary)?;
    df32_bench::benchmark(values, &input, device, upload, &mut summary)?;
    Ok(summary)
}

fn main() -> Result<()> {
    let device = Device::builder()
        .build()
        .context("No Vulkan device found")?;
    ensure!(device.is_device(), "Expected a Vulkan device");
    const INPUTS: usize = 10_000_000;
    const SEED: u64 = 42;
    let mut rng = StdRng::seed_from_u64(SEED);
    // Use normal values to isolate rounding loss from subnormal behavior.
    let random: Vec<f32> = (0..INPUTS)
        .map(|_| {
            let magnitude = rng.random_range(1.0_f32..1_000_000.0);
            if rng.random() { magnitude } else { -magnitude }
        })
        .collect();
    run_case(
        "Random signed values, magnitudes [1, 1e6)",
        &random,
        &device,
    )?
    .print();

    // Large values hide small increments in f32. Cancel them to expose the lost increments.
    // The integer sum is exact in f64 but may need rounding to f32.
    let mut cancellation = vec![16_777_216.0_f32; 8];
    cancellation.extend((0..INPUTS - 16).map(|_| rng.random_range(1..4) as f32));
    cancellation.extend([-16_777_216.0; 8]);
    run_case(
        "Small increments followed by large cancellation",
        &cancellation,
        &device,
    )?
    .print();
    #[cfg(feature = "half")]
    half_bench::benchmark(INPUTS, &mut rng, &device)?;
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
