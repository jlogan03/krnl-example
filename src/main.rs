use std::sync::LazyLock;

use krnl::{
    anyhow::{Context, Result, bail},
    buffer::Buffer,
    device::Device,
};

pub(crate) mod helpers;
use helpers::{print_available_devices, print_device_capabilities};

pub(crate) mod kernels;
use kernels::affine_device;

/// Device handle to be initialized once then never again.
static DEVICE: LazyLock<Result<Device>> = LazyLock::new(|| {
    let device = Device::builder()
        .build()
        .context("No Vulkan device found. Install Vulkan and check `vulkaninfo --summary`.")?;
    if !device.is_device() {
        bail!("Expected a device-backed runtime, got host");
    }
    Ok(device)
});

fn main() -> Result<()> {
    let a = vec![2.0f64, 2.0, 2.0, 2.0];
    let b = vec![1.0f64, 1.0, 1.0, 1.0];
    let x = vec![0.0f64, 1.0, 2.0, 3.5];

    print_available_devices();

    let device = (&*DEVICE)
        .as_ref()
        .map_err(|err| krnl::anyhow::anyhow!("{err:#}"))?;
    println!("");
    println!("Using device:");
    print_device_capabilities(&device);

    let a = Buffer::from(a).into_device(device.clone())?;
    let b = Buffer::from(b).into_device(device.clone())?;
    let x = Buffer::from(x).into_device(device.clone())?;
    let mut y = Buffer::<f64>::zeros(device.clone(), x.len())?;

    affine_device(a.as_slice(), b.as_slice(), x.as_slice(), y.as_slice_mut())?;
    device.wait()?;

    let y = y.into_vec()?;
    assert_eq!(y, vec![1.0, 3.0, 5.0, 8.0]);
    println!("y = {y:?}");
    Ok(())
}
