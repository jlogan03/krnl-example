# krnl-example

This example uses the sibling `../krnl` checkout for both the library and kernel
compiler. Its rust-gpu dependencies come from `jlogan03/rust-gpu`'s
`jlogan/float-controls-default` Git branch, pinned by the Cargo lockfiles.
The host application requires Rust 1.95 or newer.

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
with its allocation features disabled. Each million-element input reduces to one
`f32` scalar. All accumulators use two banks, including the sequential CPU baseline.

The parallel GPU procedure uses three passes: up to 8,192 threads each reduce a
contiguous input chunk, up to 256 threads merge chunks of those partial pairs,
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
sums for arbitrary inputs. Strict GPU results are checked against the rounded `f64` reference for these
benchmark datasets.

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

## License

Licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
