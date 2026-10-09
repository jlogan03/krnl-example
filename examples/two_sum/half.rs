use super::{CaseSummary, cpu_threads, run_case, time_gpu, time_runs, upload};
use krnl::{anyhow::Result, buffer::Buffer, device::Device};
use krnl_example::half_kernels::{
    decode, decode_pair, encode, merge_twosum_f16, parallel_half, round_pair, sum_df16,
    sum_df16_scalars, sum_twosum_f16,
};
use rand::Rng;
use rayon::prelude::*;
use std::hint::black_box;

fn cpu_sum(values: &[u16], threads: usize, df16: bool) -> u32 {
    let size = values.len().div_ceil(threads.max(1)).max(1);
    let partials: Vec<_> = values
        .par_chunks(size)
        .map(|chunk| {
            if df16 {
                sum_df16_scalars(chunk.iter().copied())
            } else {
                sum_twosum_f16(chunk.iter().copied())
            }
        })
        .collect();
    if df16 {
        sum_df16(partials.into_iter())
    } else {
        u32::from(round_pair(merge_twosum_f16(partials.into_iter())))
    }
}

fn run(name: &str, source: &[f32], device: &Device) -> Result<CaseSummary> {
    // Quantize before timing so every variant sums the same values.
    let values: Vec<_> = source.iter().map(|&x| encode(x)).collect();
    let singles: Vec<_> = values.iter().map(|&x| decode(x) as f32).collect();
    let mut summary = run_case(name, &singles, device)?;
    let (input, upload) = upload(&values, device)?;
    for df16 in [false, true] {
        let label = if df16 {
            "Sequential Df16"
        } else {
            "TwoSum f16"
        };
        let threads = cpu_threads();
        let cpu = time_runs(|| Ok(decode_pair(cpu_sum(black_box(&values), threads, df16))))?;
        for fast in [false, true] {
            let mut run = parallel_half(device.clone(), values.len(), df16, fast)?;
            let mut output = Buffer::<u32>::zeros(device.clone(), 1)?;
            let compute = time_gpu(device, || run(input.as_slice(), output.as_slice_mut()))?;
            let (result, download) = time_runs(|| Ok(decode_pair(output.to_vec()?[0])))?;
            summary.record(label, fast, cpu, (result, compute), upload, download)?;
        }
    }
    Ok(summary)
}

pub fn benchmark(inputs: usize, rng: &mut impl Rng, device: &Device) -> Result<()> {
    let random: Vec<f32> = (0..inputs)
        .map(|_| {
            let x = rng.random_range(0.0625f32..1.0);
            if rng.random() { x } else { -x }
        })
        .collect();
    run(
        "Quantized half random signed magnitudes [1/16, 1)",
        &random,
        device,
    )?
    .print();

    // Use normal half values and bounded totals to expose rounding loss without overflow.
    let mut cancellation = vec![128.0f32; 8];
    cancellation.extend(std::iter::repeat_n(2f32.powi(-14), inputs - 16));
    cancellation.extend([-128.0f32; 8]);
    run(
        "Quantized half small increments followed by cancellation",
        &cancellation,
        device,
    )?
    .print();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_half_reduction_tails_and_cancellation() {
        for threads in [1, 2, 3, 8] {
            // Cover empty input, partial banks, and uneven chunks.
            for len in [0, 1, 7, 17, 1025] {
                let mut values = vec![encode(1.0); len];
                // At 2^12, f16 loses unit increments without compensation.
                if len > 1 {
                    values[0] = encode(4096.0);
                    values[len - 1] = encode(-4096.0);
                }
                let exact: f64 = values.iter().map(|&x| decode(x)).sum();
                for df16 in [false, true] {
                    assert_eq!(decode_pair(cpu_sum(&values, threads, df16)), exact);
                }
            }
        }
        assert_eq!(decode_pair(cpu_sum(&[encode(1.0); 17], 3, false)), 17.0);
        // Retain the smallest f16 subnormal (2^-24) under strict math.
        let tiny = [1u16; 17];
        for df16 in [false, true] {
            assert_eq!(decode_pair(cpu_sum(&tiny, 3, df16)), 17.0 * 2f64.powi(-24));
        }
        // Keep the unit residual in partials before TwoSum rounds its final output.
        let pair = sum_twosum_f16([encode(4096.0), encode(1.0)].into_iter());
        assert_eq!(decode_pair(pair), 4097.0);
        assert_eq!(decode(round_pair(pair)), 4096.0);
    }
}
