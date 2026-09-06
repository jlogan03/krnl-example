# krnl-example

This example uses the sibling `../krnl` checkout for both the library and kernel
compiler. Its rust-gpu dependencies are pinned to upstream commit `7fa56ad6e8`.
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

Run the example with a Vulkan 1.2-capable device and driver supporting `f64`:

```bash
cargo run
```

## License

Licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
