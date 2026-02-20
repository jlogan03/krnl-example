# krnl-example

Install the version of rustc nightly and rustc extras that `krnlc` needs:

```bash
rustup toolchain install nightly-2025-06-30
rustup component add --toolchain nightly-2025-06-30 rust-src rustc-dev llvm-tools
```

Install `krnlc` from the same git branch as `krnl`:

```bash
cargo +nightly-2025-06-30 install --git git@github.com:jlogan03/krnl.git \
  --branch jlogan/update-deps --locked --force krnlc
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
