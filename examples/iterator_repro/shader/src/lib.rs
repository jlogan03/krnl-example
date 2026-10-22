#![no_std]
use spirv_std::spirv;

fn sum(values: impl Iterator<Item = f32>) -> f32 {
    let mut sum = (0.0, 0.0);
    values.for_each(|value| {
        sum.0 += value;
        sum.1 += value * value;
    });
    sum.0 + sum.1
}

#[spirv(compute(threads(1)))]
pub fn iterator_sum(
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] values: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut f32,
) {
    *output = sum((0..values.len()).map(|i| values[i]));
}
