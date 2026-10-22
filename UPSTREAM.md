# Upstream work

## 1. Remove pointer-containing iterator state from rust-gpu output

Mapped iterators can leave captured pointers in invalid SPIR-V composites,
causing a compilation failure.

The standalone [reproduction](examples/iterator_repro/README.md) fails with
`OpLoad … is not a logical pointer` on upstream `main` at
[`57d07949`](https://github.com/Rust-GPU/rust-gpu/commit/57d07949423593dd701a77d9a78acb4ff7ed943e).
Its index-loop control compiles.
