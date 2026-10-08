use krnl::{
    device::{Device, error::DeviceIndexOutOfRange},
};


/// Print device limits and numeric features.
pub fn print_device_capabilities(device: &Device) {
    if let Some(info) = device.info() {
        println!("{:#?}", info); 
    } else {
        println!("device capabilities: host backend");
    }
}

/// List available devices.
pub fn available_devices() -> Vec<Device> {
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

/// Print each device's limits and numeric features.
pub fn print_available_devices() {
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
