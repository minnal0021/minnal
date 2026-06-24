/// Sum a slice of u16 values, returning a usize.
///
/// Dispatch order:
///   1. AVX-512F — zero-extends 16 u16s to u32 via `_mm512_cvtepu16_epi32`, accumulates
///      into a 16-lane u32 ZMM register, then reduces with `_mm512_reduce_add_epi32`.
///      Requires only avx512f (base AVX-512 instruction set).
///   2. AVX2 — zero-extends 8 u16s to u32 via `_mm256_cvtepu16_epi32`, accumulates into
///      an 8-lane u32 YMM register, then reduces with a 128-bit fold + scalar extract.
///      Requires avx2.
///   3. Scalar fallback — simple iterator sum.
///
/// Used by `RunContainer::cardinality()` to sum run lengths across many runs.
pub fn sum_u16_slice(data: &[u16]) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: feature detection above guarantees AVX-512F is available.
            return unsafe { sum_u16_avx512(data) };
        }
        if is_x86_feature_detected!("avx2") {
            // SAFETY: feature detection above guarantees AVX2 is available.
            return unsafe { sum_u16_avx2(data) };
        }
    }

    sum_u16_scalar(data)
}

#[inline]
fn sum_u16_scalar(data: &[u16]) -> usize {
    data.iter().map(|&v| v as usize).sum()
}

/// AVX2 implementation.
///
/// Processes 8 u16s per iteration:
///   - Load 8 u16s (128 bits) into an XMM register.
///   - Zero-extend to 8 × u32 (256 bits) in a YMM register via `_mm256_cvtepu16_epi32`.
///   - Accumulate into a running u32 YMM accumulator.
///
/// Reduction: fold the 8-lane YMM into a 4-lane XMM by adding the upper and lower
/// 128-bit halves, then extract all 4 lanes and sum them.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_u16_avx2(data: &[u16]) -> usize {
    use std::arch::x86_64::*;

    let mut acc = _mm256_setzero_si256(); // 8 × u32 accumulator
    let chunks = data.len() / 8;

    for i in 0..chunks {
        unsafe {
            // Load 8 u16s (128 bits) unaligned
            let v16 = _mm_loadu_si128(data.as_ptr().add(i * 8) as *const __m128i);
            // Zero-extend each u16 to u32 (VPMOVZXWD): 8 × u16 → 8 × u32
            let v32 = _mm256_cvtepu16_epi32(v16);
            acc = _mm256_add_epi32(acc, v32);
        }
    }

    // Fold YMM (8 × u32) into XMM (4 × u32) by adding the two 128-bit halves
    let lo128 = _mm256_castsi256_si128(acc);
    let hi128 = _mm256_extracti128_si256(acc, 1);
    let sum128 = _mm_add_epi32(lo128, hi128);

    // Extract and sum the 4 u32 lanes
    let mut total = (_mm_extract_epi32(sum128, 0) as u32
        + _mm_extract_epi32(sum128, 1) as u32
        + _mm_extract_epi32(sum128, 2) as u32
        + _mm_extract_epi32(sum128, 3) as u32) as usize;

    // Scalar tail for the remaining 0–7 values
    for &v in &data[chunks * 8..] {
        total += v as usize;
    }

    total
}

/// AVX-512F implementation.
///
/// Processes 16 u16s per iteration:
///   - Load 16 u16s (256 bits) into a YMM register.
///   - Zero-extend to 16 × u32 (512 bits) in a ZMM register via VPMOVZXWD.
///   - Accumulate into a running u32 ZMM accumulator.
///
/// Max safe accumulation without overflow: each u16 ≤ 65535, 16 lanes,
/// worst-case total per container ≤ 65536, so u32 lanes are fine.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sum_u16_avx512(data: &[u16]) -> usize {
    use std::arch::x86_64::*;

    let mut acc = _mm512_setzero_si512(); // 16 × u32 accumulator
    let chunks = data.len() / 16; // 16 u16s fit in a 256-bit YMM

    for i in 0..chunks {
        unsafe {
            // Load 16 u16s (256 bits) unaligned
            let v16 = _mm256_loadu_si256(data.as_ptr().add(i * 16) as *const __m256i);
            // Zero-extend each u16 to u32 (VPMOVZXWD): 16 × u16 → 16 × u32
            let v32 = _mm512_cvtepu16_epi32(v16);
            // Accumulate
            acc = _mm512_add_epi32(acc, v32);
        }
    }

    // Horizontal sum of the 16 u32 lanes
    let mut total = _mm512_reduce_add_epi32(acc) as usize;

    // Handle the tail (0–15 values) with scalar
    for &v in &data[chunks * 16..] {
        total += v as usize;
    }

    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_zeros() {
        assert_eq!(sum_u16_scalar(&[0u16; 100]), 0);
    }

    #[test]
    fn scalar_known() {
        assert_eq!(sum_u16_scalar(&[1, 2, 3, 4, 5]), 15);
        assert_eq!(sum_u16_scalar(&[u16::MAX; 1]), 65535);
    }

    #[test]
    fn scalar_handles_tail() {
        // 17 values — 1 full chunk of 16 + 1 remainder
        let data = vec![1u16; 17];
        assert_eq!(sum_u16_scalar(&data), 17);
    }

    #[test]
    fn dispatch_matches_scalar() {
        let data: Vec<u16> = (0..500).map(|i| i as u16).collect();
        let expected = sum_u16_scalar(&data);
        assert_eq!(sum_u16_slice(&data), expected);
    }

    #[test]
    fn dispatch_non_multiple_of_16() {
        let data: Vec<u16> = (0..33).collect();
        assert_eq!(sum_u16_slice(&data), sum_u16_scalar(&data));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar_when_available() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let data: Vec<u16> = (0..500).map(|i| i as u16 % 200).collect();
        let scalar = sum_u16_scalar(&data);
        let simd = unsafe { sum_u16_avx2(&data) };
        assert_eq!(simd, scalar);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_handles_tail() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        // 13 values — not a multiple of 8, exercises the scalar tail
        let data: Vec<u16> = (0..13).collect();
        let scalar = sum_u16_scalar(&data);
        let simd = unsafe { sum_u16_avx2(&data) };
        assert_eq!(simd, scalar);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_matches_scalar_when_available() {
        if !is_x86_feature_detected!("avx512f") {
            return; // skip if AVX-512F not available
        }
        let data: Vec<u16> = (0..500).map(|i| i as u16 % 200).collect();
        let scalar = sum_u16_scalar(&data);
        let simd = unsafe { sum_u16_avx512(&data) };
        assert_eq!(simd, scalar);
    }
}
