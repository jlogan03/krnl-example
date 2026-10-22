# Iterator reproduction

This standalone crate reproduces a rust-gpu failure with a mapped iterator whose
consumer updates two accumulators. It needs no GPU, MoltenVK, or Metal compiler.

Install the compiler revision used to reproduce the failure:

```bash
rustup toolchain install nightly-2026-08-15 --profile minimal \
  --component rust-src,rustc-dev,llvm-tools
cargo +nightly-2026-08-15 install --git https://github.com/jlogan03/krnl \
  --rev 2cc7a4f8c9f8371b81cd8041537690c405c159b2 --locked krnlc
```

From this directory, run:

```bash
krnlc --manifest-path Cargo.toml
```

Compilation fails during SPIR-V validation:

```text
error: OpLoad Pointer <id> '…' is not a logical pointer.
```

The diagnostic names the emitted `.spv` file. The iterator and closure captures
leave pointers in composites; the generated loads do not satisfy SPIR-V's
logical-pointer rules. The failure occurs with rust-gpu revision `18e7d247`.

As a control, replace the body of `iterator_sum` with an index loop:

```rust
let mut sum = (0.0, 0.0);
for i in 0..values.len() {
    let value = values[i];
    sum.0 += value;
    sum.1 += value * value;
}
*output = sum.0 + sum.1;
```

The control compiles. Both forms compute the sum plus the sum of squares.
