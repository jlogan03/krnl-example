# krnl-example

This example uses the sibling `../krnl` checkout for both the library and kernel
compiler. Its rust-gpu support crates come from `jlogan03/rust-gpu`'s
`jlogan/float-controls-default` Git branch, pinned by the Cargo lockfiles.
`../krnl/krnlc` patches its codegen backend to the sibling `../rust-gpu`
checkout, which includes the `fmaf16` intrinsic lowering needed when compiling
`num-synth::Df16`. The compiler manifest and lockfile record this override.
The default host application requires Rust 1.95 or newer; half benchmarks
require nightly Rust.

Install the version of Rust nightly and components that `krnlc` needs:

```bash
rustup toolchain install nightly-2026-05-22 --profile minimal \
  --component rust-src,rustc-dev,llvm-tools
```

Build the local compiler and regenerate the kernel cache for this crate:

```bash
bash compile_kernels.sh
```

The script uses `../krnl/krnlc/rust-toolchain.toml` and its locked dependencies.
The first build compiles SPIRV-Tools from source and requires a C++ compiler.
Re-run it after changing kernels or updating `../krnl`.

Run the example with a Vulkan 1.2-capable device and driver supporting `f64`,
`shaderFloatControls2`, f64 subnormal preservation and round-to-nearest/ties-to-even:

```bash
cargo run
```

krnl explicitly selects rust-gpu's `rust_math` mode by default.

Run the TwoSum correctness and timing comparison with an optimized host build:

```bash
cargo run --release --example two_sum
```

It uses `TwoSum` from the sibling `../deimos/software/deimos_numerics` crate,
with its allocation features disabled. Each 10-million-element input reduces to one
`f32` scalar. All accumulators use two banks, including the parallel CPU baseline.
The CPU uses Rayon to reduce contiguous chunks, capped at the smaller of the
physical-core count and the Rayon worker count, following `interpn`. It merges
the resulting sum/residual pairs before rounding. The plain `f32` comparison
remains single-threaded; CPU/GPU ratios use the parallel compensated CPU time.

The parallel GPU procedure uses three passes: up to 8,192 threads reduce
interleaved input elements, up to 256 threads merge chunks of those partial pairs,
and one thread merges the remaining pairs into the final scalar. Set `THREADS`
and `REDUCTION_THREADS` in `parallel_twosum` to adjust the two parallel thread
counts; krnl handles dispatch sizing automatically. Both components of each
`(sum, residual)` pair survive until the final rounding. All kernels use safe
item outputs, with no shared memory or barriers. Each parallel pass caps its
active thread count at its input length.

The example compares compensated CPU, strict GPU and fast-math GPU results with
a plain `f32` sum and an `f64` reference. Seeded random inputs and small increments
followed by large cancellation demonstrate rounding loss without subnormals.
Compensation improves accuracy but does not promise exact, order-independent
sums for arbitrary inputs. Compensated CPU and strict GPU results are checked
against the rounded `f64` reference for these benchmark datasets.

CPU, GPU dispatch and input upload timings average 20 runs after warm-up; GPU
timings include waiting for completion. Each GPU policy runs the parallel
procedure for three seconds before measurement to warm up the device. Reported
GPU times describe warmed performance and exclude that warm-up. Uploads reuse
preallocated buffers.
One-time input allocation/initialization and scalar downloads are timed separately,
and kernel creation is excluded from dispatch timings. Parallel timings include
all three dispatches and waiting for the final result, with scratch buffers allocated
before timing.
Transfer-inclusive GPU totals sum the mean upload and dispatch times with the
one-time output download; they exclude device-buffer allocation.

## Df32 reduction comparison

The default TwoSum benchmark also includes `num-synth::Df32`, using the
sibling `../num-synth` checkout. No feature flag or nightly host compiler is
needed for Df32:

```bash
cargo run --release --example two_sum
```

Each dataset compares the existing plain f32/TwoSum results with a two-bank
Df32 CPU reduction and strict/fast GPU reductions. Df32 reuses the same f32
input buffer and measured upload time; enabling `half` also runs Df32 on
both shared quantized half-range datasets.

CPU Df32 uses the same Rayon chunk count as TwoSum. GPU Df32 uses the same
`8192 -> 256 -> 1` schedule. Every accumulator adds Df32 values, and separate
f32 buffers preserve high and residual components through every pass and
through the final output. Results are evaluated in f64 on the host for error
reporting. This differs from the TwoSum baseline, whose final output is one
rounded f32 scalar; neither is an exact, order-independent summation contract.
Strict CPU/GPU results need not match when their addition order differs.

Timings follow the existing protocol: three seconds of GPU warm-up per
policy, means of 20 runs, prebuilt pipelines/scratch, and dispatch plus wait
reported separately from transfers. Df32 readback measures two four-byte
scalar downloads together. CPU timing includes allocating and merging the
per-thread partials. The host prints numerical error and strict/fast result
changes; fast-math remains a diagnostic rather than a valid compensation mode.

The indexed two-bank loop avoids an incidental 64-bit enum tag from
`Option<Df32>`. Generated reduction shaders use only 32-bit floating-point
and integer types, with no f64 or integer-64 capability requirement. Strict
kernels retain f32 subnormals and nearest-even controls. GPU tests cover tails,
cancellation, nonzero final residuals, and strict/fast subnormal behavior;
CPU tests cover empty input and uneven Rayon chunks.

```bash
cargo test --lib --example two_sum
```

## Half-precision reduction comparisons

Enable the additional variants in the same TwoSum benchmark:

```bash
bash compile_kernels.sh
cargo +nightly-2026-05-22 run --release --features half --example two_sum
```

The cache includes both baseline and half kernels. Default builds still work
on stable Rust. `half` enables the small no-std `half-reduction` adapter and
`../num-synth`'s Df16 implementation; Rust's primitive f16 arithmetic executes
inside that adapter. Integer buffers carry exact component bits because
krnl's public f16 buffer type belongs to the separate `half` crate. The
adapter does not substitute that crate's f32-based arithmetic.

The original f32 datasets remain. Two additional 10-million-element datasets
compare f32 TwoSum, Df32, plain f16, and Df16 on identical quantized inputs:

- Random signed half values with magnitudes in `[1/16, 1)`.
- Eight `128` values, normal half increments of `2^-14`, then eight `-128`
  values to expose rounding loss through cancellation.

The original magnitudes near `1e6` overflow f16, so they are not reused for
half arithmetic. Quantization happens before timing; the f64 reference sums
the actual stored half values, not the original f32 samples. f16/Df16 errors
are reported without imposing the f32 benchmark's exact-result criterion.
CPU contiguous chunks and GPU interleaving can round differently even under
the strict policy.

Both new types use two independent accumulation banks, on CPU and in every
GPU pass. Plain f16 is an uncompensated baseline. Df16 uses pair addition from
num-synth. Each follows the same `8192 -> 256 -> 1` parallel schedule. f16
partials occupy two bytes; Df16 partials occupy four, packing high in the low
16 bits and residual in the upper 16 bits. Df16 preserves both components
through all passes and returns the pair, decoded to f64 for reporting. Plain
f16 returns one scalar in the low half of the same four-byte result buffer.
There is no conversion to f32 between Df16 passes.

CPU timings use Rayon with the existing core-count cap. GPU strict and
fast-math policies each receive the same three-second warm-up and 20 timed
runs as f32. Input uploads reuse a two-byte-per-element buffer. Kernel creation,
scratch allocation, input quantization, and warm-up are outside dispatch
measurements. Transfer-inclusive times include mean upload, dispatch/wait,
and a one-time four-byte download. CPU timings include allocating/merging
per-thread partials, as in the existing f32 CPU benchmark.

The half kernels require shader f16 arithmetic and the applicable strict
float controls; kernel construction reports unsupported device features.
The dumped SPIR-V uses `OpTypeFloat 16`, half `OpFAdd`, and strict half
rounding/subnormal controls. Fast-math can invalidate compensation and is
reported separately. Tests compare strict results against the same reduction
order on the CPU, cover tails across both pass boundaries, and check strict
versus fast subnormal behavior.

CPU builds use the compiler's default target unless overridden. For timings
using all instructions available on the local CPU, run the host benchmark
with `RUSTFLAGS="-C target-cpu=native"`. This does not imply native half
arithmetic exists on that CPU, and must not be applied to the SPIR-V compiler
command. GPU timings include dispatch/wait overhead and do not measure a
single isolated arithmetic instruction.

Run the CPU/GPU regression tests with:

```bash
cargo +nightly-2026-05-22 test --features half --lib --example two_sum
```

The sibling num-synth manifest accepts libm 0.2.8 or newer so this example can
retain krnl's required 0.2.8 pin. Its normal standalone lockfile still selects
0.2.16. The half reduction itself only uses addition and normalization.

## License

Licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
