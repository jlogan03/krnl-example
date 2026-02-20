use krnl::{
    anyhow::{Result, bail}, buffer::{Slice, SliceMut},  macros::module
};

/// Vulkan SPIRV compute kernels to be compiled with `krnlc`.
#[module]
mod kernels {
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::macros::kernel;
    use krnl_core::num_traits::Num;

    /// Type-generic example scalar kernel.
    /// 
    /// Because the krnl `kernels` module can't see its super:: or anything else 
    /// not enclosed in a #[module] scope, this generic scalar kernel must be
    /// defined here then exported to super:: instead of the other way around.
    /// 
    /// Alternatively, we could have functions like this one stored in another
    /// #[module] scope in this crate, or anywhere in another no_std crate.
    /// They just can't be _both_ inside this crate and outside a #[module] scope.
    #[inline]
    pub fn affine_scalar<T: Num> (a: T, b: T, x: T) -> T {
        a * x + b
    }

    /// Simple example scalar XPU kernel for 64-bit floats.
    #[kernel]
    pub fn affine(#[item] a: f64, #[item] b: f64, #[item] x: f64, #[item] y: &mut f64) {
        *y = affine_scalar(a, b, x);
    }
}

// We can re-export generic functions from inside a #[module] scope.
pub use kernels::affine_scalar;

/// Run `y = a*x + b` for slice inputs on a compute device. 
pub fn affine_device(a: Slice<f64>, b: Slice<f64>, x: Slice<f64>, y: SliceMut<f64>) -> Result<()> {
    if a.len() != b.len() || a.len() != x.len() || a.len() != y.len() {
        bail!("a, b, x, and y lengths must match");
    }

    // Kernels are cached per-device internally, so we don't need to wrap this in a LazyCell.
    kernels::affine::builder()?
        .build(y.device())?
        .dispatch(a, b, x, y)
}