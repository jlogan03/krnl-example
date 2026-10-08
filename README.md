# krnl-example

Vulkan compute examples using [krnl](https://github.com/jlogan03/krnl/tree/jlogan/update-deps):
a basic `y = a*x + b` kernel and CPU/GPU reduction benchmarks.

## Run

Requires Rust 1.95+ and a Vulkan 1.2 device with `shaderFloatControls2`.
Strict kernels also require subnormal preservation and round-to-nearest/ties-to-even
for their float types. The basic example uses `f64`.

```bash
cargo run
cargo run --release --example two_sum
```

The benchmark compares plain `f32`, [TwoSum](https://github.com/deimoscontrols/deimos),
and [Df32](https://github.com/jlogan03/num-synth). To include `f16` and `Df16`,
use nightly Rust and a device with shader `f16` support:

```bash
rustup toolchain install nightly-2026-05-22 --profile minimal
cargo +nightly-2026-05-22 run --release --features half --example two_sum
```

## Read the results

Each dataset contains 10 million values. Seeded random values and cancellation
cases expose rounding loss. Half comparisons use the same quantized inputs for
all variants, including the `f64` reference.

- CPU reductions use Rayon, capped by the physical-core and worker counts.
  Plain `f32` is a serial baseline.
- GPU reductions use three passes with up to `8192 -> 256 -> 1` threads.
  Compensated reductions retain residuals through every pass. TwoSum rounds the
  final result to `f32`; Df32 and Df16 return both components.
- Strict kernels use `rust_math`. Fast-math results show how optimizations can
  break compensation. Different addition orders can produce different results
  even under strict math.

Compensation improves accuracy but does not guarantee exact sums.
Tables report results, absolute errors against the `f64` reference, and timings.

Compute and upload times average 20 runs. Each GPU variant warms up for three
seconds per policy. GPU compute includes all dispatches and waiting for completion.
GPU totals add the mean upload time and one output download. They exclude device
allocation, pipeline creation, input preparation, and warm-up. CPU timings include
allocating and merging partial results.

Set `RUSTFLAGS="-C target-cpu=native"` for host timings that use local CPU features.
Apply it only to the host build, not the kernel compiler. Native CPU `f16`
instructions are not guaranteed.

## Rebuild kernels

The checked-in cache includes all variants. Rebuild it after changing kernels or
dependencies. The krnl fork uses rust-gpu's
[`jlogan/float-controls-intrinsics`](https://github.com/jlogan03/rust-gpu/tree/jlogan/float-controls-intrinsics)
branch, with revisions pinned in the Cargo lockfiles.

Install the matching compiler and its toolchain components. Building SPIRV-Tools
requires a C++ compiler.

```bash
rustup toolchain install nightly-2026-05-22 --profile minimal \
  --component rust-src,rustc-dev,llvm-tools
cargo +nightly-2026-05-22 install --git https://github.com/jlogan03/krnl \
  --branch jlogan/update-deps --locked krnlc
bash compile_kernels.sh
```

Reinstall `krnlc` after updating krnl. The script uses the installed compiler,
which selects its pinned Rust toolchain.

## Test

These tests include GPU checks and require a supported device:

```bash
cargo test --lib --example two_sum
cargo +nightly-2026-05-22 test --features half --lib --example two_sum
```

## License

Licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
