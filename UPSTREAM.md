# Upstream work

## 1. Remove pointer-containing iterator state from rust-gpu output

Observed with rust-gpu revision
[`18e7d247`](https://github.com/jlogan03/rust-gpu/tree/18e7d247363b7e5a8642b476ecfdca797d899291)
and `nightly-2026-08-15`. Not yet reported upstream.

Passing a mapped iterator to a reduction helper can leave captured pointers in
SPIR-V composites that carry iterator state through the loop. The affected
kernels use logical addressing, which does not permit these pointer operands
in `OpCompositeConstruct`. This occurs in the f32 TwoSum and f16 reductions.
SPIRV-Cross translates the output into Metal code with incompatible pointer
types. Metal then rejects the shader.

The current workaround uses an explicit index loop for f32. The f16 helpers take
an index iterator and a separate reader closure, so the captured pointers stay
out of the iterator state. Adding `move` to the original closure was not enough:
it removed references to local variables but left captured buffer pointers.

### Reproduce without MoltenVK

The standalone [iterator_repro](examples/iterator_repro/README.md) crate reduces
the failure to a mapped iterator and two scalar accumulators. Its `.for_each`
closure triggers an `OpLoad` validation error about a non-logical pointer.
The documented index-loop control compiles. Both can be checked on Linux without
a GPU, MoltenVK, or Metal compiler.

The original `for`-loop consumers passed SPIRV-Tools validation despite retaining
invalid pointer composites. An upstream regression should cover both forms:
compile the reduced example, and inspect the original form for pointer operands
in `OpCompositeConstruct`. A validator regression should cover the form that
SPIRV-Tools accepted.
