#[krnl::macros::module]
mod kernels {
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::macros::kernel;

    fn sum(values: impl Iterator<Item = f32>) -> f32 {
        let mut sum = (0.0, 0.0);
        values.for_each(|value| {
            sum.0 += value;
            sum.1 += value * value;
        });
        sum.0 + sum.1
    }

    #[kernel]
    pub fn iterator_sum(#[global] values: Slice<f32>, #[item] output: &mut f32) {
        let count = values.len();
        *output = sum((0..count).map(|i| values[i]));
    }
}
