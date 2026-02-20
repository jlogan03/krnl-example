use krnl::{
    anyhow::{Result, bail},
    buffer::{Slice, SliceMut},
    macros::module,
};

/// Compute kernels to be compiled with `krnlc`
#[module]
mod kernels {
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::macros::kernel;

    /// Simple example scalar kernel.
    #[kernel]
    pub fn affine(#[item] a: f64, #[item] b: f64, #[item] x: f64, #[item] y: &mut f64) {
        *y = a * x + b;
    }
}

/// Run `y = a*x + b` on slice inputs. 
pub fn affine(a: Slice<f64>, b: Slice<f64>, x: Slice<f64>, y: SliceMut<f64>) -> Result<()> {
    if a.len() != b.len() || a.len() != x.len() || a.len() != y.len() {
        bail!("a, b, x, and y lengths must match");
    }

    // Kernels are cached per-device internally, so we don't need to wrap this in a LazyCell.
    kernels::affine::builder()?
        .build(y.device())?
        .dispatch(a, b, x, y)
}