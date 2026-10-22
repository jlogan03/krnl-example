# Iterator reproduction

This standalone workspace reproduces a rust-gpu failure with a mapped iterator
whose consumer updates two accumulators. It uses `spirv-std` and `SpirvBuilder`
directly, with no krnl, GPU, MoltenVK, or Metal compiler.

From this directory, run:

```bash
cargo run --locked
```

Rustup selects the pinned nightly toolchain. Cargo builds rust-gpu and its
SPIRV-Tools dependency, which requires a C++ compiler. The launcher
then compiles [shader/src/lib.rs](shader/src/lib.rs) and fails during SPIR-V
validation:

```text
error: OpLoad Pointer <id> '…' is not a logical pointer.
```

The diagnostic names the emitted `.spv` file. The iterator and closure captures
leave pointers in composites; the generated loads do not satisfy SPIR-V's
logical-pointer rules. The reproduction pins upstream rust-gpu `main` at
[`57d07949`](https://github.com/Rust-GPU/rust-gpu/commit/57d07949423593dd701a77d9a78acb4ff7ed943e).

As a control, replace the body of `iterator_sum` in `shader/src/lib.rs` with an
index loop and rerun the command:

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
