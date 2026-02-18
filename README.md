# krnl-example

Install the version of rustc nightly and rustc extras that `krnlc` needs:

```bash
rustup toolchain install nightly-2023-05-27
rustup component add --toolchain nightly-2023-05-27 rust-src rustc-dev llvm-tools-preview
```

Install `krnlc` using its required version of rustc:

```bash
cargo +nightly-2023-05-27 install krnlc --locked
```

Build the kernel cache for this crate:

```bash
sh krnlc.sh
```

Run the example:

```bash
cargo run
```

## License

Licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
