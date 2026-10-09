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

// The low 16 bits hold the high component; the upper 16 bits hold the residual.
#[inline]
fn pack(hi: f16, lo: f16) -> u32 {
    u32::from(hi.to_bits()) | (u32::from(lo.to_bits()) << 16)
}

#[inline]
pub fn round_pair(value: u32) -> u16 {
    (f16::from_bits(value as u16) + f16::from_bits((value >> 16) as u16)).to_bits()
}

// Match deimos TwoSum's scalar update. Its num_traits::Float bound excludes primitive f16.
#[inline]
fn add(sum: &mut f16, residual: &mut f16, value: f16) {
    let next = *sum + value;
    let rounded = next - *sum;
    *residual += (*sum - (next - rounded)) + (value - rounded);
    *sum = next;
}

#[inline]
fn finish_twosum(mut a: f16, mut ar: f16, b: f16, br: f16) -> u32 {
    add(&mut a, &mut ar, b);
    add(&mut a, &mut ar, br);
    pack(a, ar)
}

/// Sum with two compensated f16 accumulators and retain the residual.
#[inline]
pub fn sum_twosum_f16(mut values: impl Iterator<Item = u16>) -> u32 {
    let (mut a, mut ar, mut b, mut br) = (0.0f16, 0.0f16, 0.0f16, 0.0f16);
    while let Some(value) = values.next() {
        add(&mut a, &mut ar, f16::from_bits(value));
        if let Some(value) = values.next() {
            add(&mut b, &mut br, f16::from_bits(value));
        }
    }
    finish_twosum(a, ar, b, br)
}

#[inline]
pub fn merge_twosum_f16(values: impl Iterator<Item = u32>) -> u32 {
    let (mut a, mut ar, mut b, mut br) = (0.0f16, 0.0f16, 0.0f16, 0.0f16);
    for bits in values {
        add(&mut a, &mut ar, f16::from_bits(bits as u16));
        add(&mut b, &mut br, f16::from_bits((bits >> 16) as u16));
    }
    finish_twosum(a, ar, b, br)
}

#[inline]
pub fn sum_df16_scalars(values: impl Iterator<Item = u16>) -> u32 {
    let mut sum = Df16::ZERO;
    for bits in values {
        sum += Df16::from_parts(f16::from_bits(bits), 0.0f16);
    }
    let (hi, lo) = sum.to_parts();
    pack(hi, lo)
}

/// Add each packed pair to one Df16 accumulator in input order.
#[inline]
pub fn sum_df16(values: impl Iterator<Item = u32>) -> u32 {
    let mut sum = Df16::ZERO;
    for bits in values {
        sum += Df16::from_parts(
            f16::from_bits(bits as u16),
            f16::from_bits((bits >> 16) as u16),
        );
    }
    let (hi, lo) = sum.to_parts();
    pack(hi, lo)
}
