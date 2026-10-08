//! Rust `f16` arithmetic with integer buffers for krnl.
#![no_std]
#![feature(f16)]

use num_synth::Df16;

#[inline]
pub fn encode(value: f32) -> u16 {
    (value as f16).to_bits()
}

#[inline]
pub fn decode(value: u16) -> f64 {
    f16::from_bits(value) as f64
}

#[inline]
pub fn decode_pair(value: u32) -> f64 {
    decode(value as u16) + decode((value >> 16) as u16)
}

/// Sum with two f16 accumulators and merge them at the end.
#[inline]
pub fn sum_f16(mut values: impl Iterator<Item = u16>) -> u16 {
    let (mut a, mut b) = (0.0f16, 0.0f16);
    while let Some(value) = values.next() {
        a += f16::from_bits(value);
        if let Some(value) = values.next() {
            b += f16::from_bits(value);
        }
    }
    (a + b).to_bits()
}

/// Sum with two Df16 accumulators. The low 16 bits hold the high component.
/// The upper 16 bits hold the residual. Zero-extend scalar inputs to `u32` for a zero residual.
#[inline]
pub fn sum_df16(mut values: impl Iterator<Item = u32>) -> u32 {
    let unpack = |bits: u32| {
        Df16::from_parts(
            f16::from_bits(bits as u16),
            f16::from_bits((bits >> 16) as u16),
        )
    };
    let (mut a, mut b) = (Df16::ZERO, Df16::ZERO);
    while let Some(value) = values.next() {
        a += unpack(value);
        if let Some(value) = values.next() {
            b += unpack(value);
        }
    }
    let (hi, lo) = (a + b).to_parts();
    u32::from(hi.to_bits()) | (u32::from(lo.to_bits()) << 16)
}
