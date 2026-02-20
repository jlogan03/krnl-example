use std::sync::LazyLock;

use krnl::{
    anyhow::{Context, Result, bail},
    buffer::{Buffer, Slice, SliceMut},
    device::{Device, error::DeviceIndexOutOfRange},
    macros::module,
};

/// Compute kernels to be compiled with `krnlc`
#[module]
mod kernels {
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::macros::kernel;

    #[kernel]
    pub fn affine(#[item] a: f64, #[item] b: f64, #[item] x: f64, #[item] y: &mut f64) {
        *y = a * x + b;
    }
}

fn affine(a: Slice<f64>, b: Slice<f64>, x: Slice<f64>, y: SliceMut<f64>) -> Result<()> {
    if a.len() != b.len() || a.len() != x.len() || a.len() != y.len() {
        bail!("a, b, x, and y lengths must match");
    }

    kernels::affine::builder()?
        .build(y.device())?
        .dispatch(a, b, x, y)
}

/// Pretty-print device info including threads and numeric features.
fn print_device_capabilities(device: &Device) {
    if let Some(info) = device.info() {
        println!("{:#?}", info); 
    } else {
        println!("device capabilities: host backend");
    }
}

/// Get a list of available devices.
fn available_devices() -> Vec<Device> {
    let mut devices = Vec::new();
    for index in 0usize.. {
        match Device::builder().index(index).build() {
            Ok(device) => {
                devices.push(device);
            }
            Err(err) => {
                if err.downcast_ref::<DeviceIndexOutOfRange>().is_some() {
                    break;
                }
                eprintln!("warning: failed to create device {index}: {err:#}");
                break;
            }
        }
    }
    devices
}

/// Pretty-print info for each available device.
fn print_available_devices() {
    println!("Available devices:");
    let devices = available_devices();
    if devices.is_empty() {
        println!("  (none)");
        return;
    }
    for (index, device) in devices.into_iter().enumerate() {
        println!("    [{index}] {device:?}");
        print_device_capabilities(&device);
    }
}

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

    affine(a.as_slice(), b.as_slice(), x.as_slice(), y.as_slice_mut())?;
    device.wait()?;

    let y = y.into_vec()?;
    assert_eq!(y, vec![1.0, 3.0, 5.0, 8.0]);
    println!("y = {y:?}");
    Ok(())
}
