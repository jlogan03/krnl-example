use krnl::{
    anyhow::{bail, Context, Result},
    buffer::{Buffer, Slice, SliceMut},
    device::Device,
    macros::module,
};

#[module]
mod kernels {
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::macros::kernel;

    #[kernel]
    pub fn affine(#[item] x: f32, a: f32, b: f32, #[item] y: &mut f32) {
        *y = a * x + b;
    }
}

fn affine(a: f32, b: f32, x: Slice<f32>, mut y: SliceMut<f32>) -> Result<()> {
    if x.len() != y.len() {
        bail!("x and y lengths must match");
    }

    if let Some((x, y)) = x.as_host_slice().zip(y.as_host_slice_mut()) {
        for (x, y) in x.iter().copied().zip(y.iter_mut()) {
            *y = a * x + b;
        }
        return Ok(());
    }

    kernels::affine::builder()?.build(y.device())?.dispatch(x, a, b, y)
}

fn main() -> Result<()> {
    let a = 2.0f32;
    let b = 1.0f32;
    let x = vec![0.0f32, 1.0, 2.0, 3.5];

    let device = Device::builder()
        .build()
        .context("No Vulkan device found. Install Vulkan and check `vulkaninfo --summary`.")?;
    if !device.is_device() {
        bail!("Expected a device-backed runtime, got host");
    }

    let x = Buffer::from(x).into_device(device.clone())?;
    let mut y = Buffer::<f32>::zeros(device.clone(), x.len())?;

    affine(a, b, x.as_slice(), y.as_slice_mut())?;
    device.wait()?;

    let y = y.into_vec()?;
    assert_eq!(y, vec![1.0, 3.0, 5.0, 8.0]);
    println!("y = {y:?}");
    Ok(())
}
