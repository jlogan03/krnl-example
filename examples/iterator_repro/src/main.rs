use spirv_builder::SpirvBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    SpirvBuilder::new(
        concat!(env!("CARGO_MANIFEST_DIR"), "/shader"),
        "spirv-unknown-vulkan1.2",
    )
    .build()?;
    Ok(())
}
