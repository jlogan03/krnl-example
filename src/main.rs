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
    pub fn affine(#[item] x: f64, a: f64, b: f64, #[item] y: &mut f64) {
        *y = a * x + b;
    }
}

fn affine(a: f64, b: f64, x: Slice<f64>, mut y: SliceMut<f64>) -> Result<()> {
    if x.len() != y.len() {
        bail!("x and y lengths must match");
    }

    kernels::affine::builder()?.build(y.device())?.dispatch(x, a, b, y)
}

fn print_device_capabilities(device: &Device) {
    if let Some(info) = device.info() {
        println!("device info:");
        println!("  is_device: {}", device.is_device());
        println!("  is_host: {}", device.is_host());
        println!("  max_groups: {}", info.max_groups());
        println!("  max_threads_per_group: {}", info.max_threads());
        println!(
            "  subgroup_threads: {}..={}",
            info.min_subgroup_threads(),
            info.max_subgroup_threads()
        );
        println!("  features: {:#?}", info.features());
    } else {
        println!("device capabilities: host backend");
    }
}

fn main() -> Result<()> {
    let a = 2.0f64;
    let b = 1.0f64;
    let x = vec![0.0f64, 1.0, 2.0, 3.5];

    let device = Device::builder()
        .build()
        .context("No Vulkan device found. Install Vulkan and check `vulkaninfo --summary`.")?;
    if !device.is_device() {
        bail!("Expected a device-backed runtime, got host");
    }
    print_device_capabilities(&device);

    let x = Buffer::from(x).into_device(device.clone())?;
    let mut y = Buffer::<f64>::zeros(device.clone(), x.len())?;

    affine(a, b, x.as_slice(), y.as_slice_mut())?;
    device.wait()?;

    let y = y.into_vec()?;
    assert_eq!(y, vec![1.0, 3.0, 5.0, 8.0]);
    println!("y = {y:?}");
    Ok(())
}
