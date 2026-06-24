/// Count the total number of set bits across a slice of u64 words.
///
/// Dispatch order:
///   1. AVX-512 VPOPCNTDQ — VPOPCNTQ on 8 u64s per ZMM register (512-bit),
///      requires `avx512vpopcntdq` + `avx512f`.
///   2. AVX2 — Wilkes-Wheeler-Gill nibble lookup via `_mm256_shuffle_epi8`
///      (VPSHUFB), processing 4 u64s (256 bits) per iteration. Requires `avx2`.
///   3. Scalar fallback — `u64::count_ones()` on each word (compiles to
///      the hardware POPCNT instruction on x86-64).
///
/// The AVX-512 and AVX2 paths are selected at runtime via `is_x86_feature_detected!`
/// so the binary runs correctly on CPUs without those extensions.
pub fn popcount_u64_slice(data: &[u64]) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vpopcntdq") && is_x86_feature_detected!("avx512f") {
            // SAFETY: feature detection above guarantees AVX-512 VPOPCNTDQ + F are available.
            return unsafe { popcount_avx512vpopcntdq(data) };
        }
        if is_x86_feature_detected!("avx2") {
            // SAFETY: feature detection above guarantees AVX2 is available.
            return unsafe { popcount_avx2(data) };
        }
    }

    popcount_scalar(data)
}

/// Scalar fallback: uses the hardware `POPCNT` instruction via `count_ones()`.
#[inline]
fn popcount_scalar(data: &[u64]) -> usize {
    data.iter().map(|w| w.count_ones() as usize).sum()
}

/// AVX2 implementation using the Wilkes-Wheeler-Gill nibble lookup (VPSHUFB).
///
/// Each byte is split into its low and high 4-bit nibbles. A 16-entry lookup
/// table (broadcast across both 128-bit lanes) maps each nibble to its popcount.
/// The two per-byte nibble popcounts are summed, then `_mm256_sad_epu8` (sum of
/// absolute differences against zero) accumulates byte sums into 64-bit lanes.
/// A final horizontal reduction collapses the 4 lanes to a scalar.
///
/// Processes 4 u64s (256 bits) per iteration.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn popcount_avx2(data: &[u64]) -> usize {
    use std::arch::x86_64::*;

    // Nibble popcount lookup table, duplicated across both 128-bit lanes.
    let lookup = _mm256_set_epi8(
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0, 4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
    );
    let low_mask = _mm256_set1_epi8(0x0f_u8 as i8); // mask for low 4 bits

    let mut acc = _mm256_setzero_si256(); // 4 × u64 accumulator
    let chunks = data.len() / 4; // 4 u64s = 256 bits per iteration

    for i in 0..chunks {
        unsafe {
            let v = _mm256_loadu_si256(data.as_ptr().add(i * 4) as *const __m256i);
            // Low nibbles: v & 0x0F
            let lo = _mm256_and_si256(v, low_mask);
            // High nibbles: (v >> 4) & 0x0F  (logical shift, not arithmetic)
            let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
            // Lookup popcount for each nibble
            let cnt_lo = _mm256_shuffle_epi8(lookup, lo);
            let cnt_hi = _mm256_shuffle_epi8(lookup, hi);
            // Sum nibble counts per byte
            let cnt = _mm256_add_epi8(cnt_lo, cnt_hi);
            // _mm256_sad_epu8 against zero: sums 8 consecutive bytes into a u64 lane
            acc = _mm256_add_epi64(acc, _mm256_sad_epu8(cnt, _mm256_setzero_si256()));
        }
    }

    // Horizontal reduction: sum 4 × u64 lanes
    let lo128 = _mm256_castsi256_si128(acc);
    let hi128 = _mm256_extracti128_si256(acc, 1);
    let sum128 = _mm_add_epi64(lo128, hi128);
    let mut total = (_mm_extract_epi64(sum128, 0) + _mm_extract_epi64(sum128, 1)) as usize;

    // Scalar tail for the remaining 0–3 words
    for &word in &data[chunks * 4..] {
        total += word.count_ones() as usize;
    }

    total
}

/// AVX-512 VPOPCNTDQ implementation.
///
/// Loads 8 u64s (512 bits) per iteration into a ZMM register and uses
/// VPOPCNTQ (`_mm512_popcnt_epi64`) to count bits in each 64-bit lane,
/// then accumulates with `_mm512_add_epi64`. A final `_mm512_reduce_add_epi64`
/// collapses the 8-lane sum.
///
/// Each lane holds at most 64, so the per-lane u64 accumulator cannot overflow
/// even for arbitrarily long slices.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
unsafe fn popcount_avx512vpopcntdq(data: &[u64]) -> usize {
    use std::arch::x86_64::*;

    let mut acc = _mm512_setzero_si512();
    let chunks = data.len() / 8; // 8 u64s = 512 bits = one ZMM register

    for i in 0..chunks {
        unsafe {
            // Load 8 u64s unaligned — cast to *const __m512i as required by the intrinsic
            let v = _mm512_loadu_si512(data.as_ptr().add(i * 8) as *const __m512i);
            // VPOPCNTQ: count set bits in each of the 8 × 64-bit lanes
            let popcnt = _mm512_popcnt_epi64(v);
            // Accumulate into the running per-lane sum
            acc = _mm512_add_epi64(acc, popcnt);
        }
    }

    // Horizontal sum of the 8 u64 lanes
    let mut total = _mm512_reduce_add_epi64(acc) as usize;

    // Scalar tail for the remaining 0–7 words
    for &word in &data[chunks * 8..] {
        total += word.count_ones() as usize;
    }

    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_all_zeros() {
        assert_eq!(popcount_scalar(&[0u64; 1024]), 0);
    }

    #[test]
    fn scalar_all_ones() {
        // 1024 words × 64 bits = 65536
        assert_eq!(popcount_scalar(&[u64::MAX; 1024]), 65536);
    }

    #[test]
    fn scalar_known_values() {
        assert_eq!(popcount_scalar(&[0b1010_1010]), 4);
        assert_eq!(popcount_scalar(&[1, 3, 7, 15]), 1 + 2 + 3 + 4);
    }

    #[test]
    fn scalar_handles_tail() {
        // 9 words — 1 full chunk of 8, 1 remainder
        let data = vec![u64::MAX; 9];
        assert_eq!(popcount_scalar(&data), 9 * 64);
    }

    #[test]
    fn dispatch_matches_scalar() {
        // Whatever path is taken, the result must match scalar
        let data: Vec<u64> = (0..1024).map(|i| (i as u64).wrapping_mul(0x0101_0101_0101_0101)).collect();
        let expected = popcount_scalar(&data);
        assert_eq!(popcount_u64_slice(&data), expected);
    }

    #[test]
    fn dispatch_handles_non_multiple_of_8() {
        let data: Vec<u64> = (0..13).map(|i| (i as u64).wrapping_mul(123456789)).collect();
        assert_eq!(popcount_u64_slice(&data), popcount_scalar(&data));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar_when_available() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let data: Vec<u64> = (0..1024).map(|i| i as u64).collect();
        let scalar = popcount_scalar(&data);
        let simd = unsafe { popcount_avx2(&data) };
        assert_eq!(simd, scalar);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_handles_tail() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        // 13 words — not a multiple of 4, exercises the scalar tail
        let data: Vec<u64> = (0..13).map(|i| i as u64).collect();
        let scalar = popcount_scalar(&data);
        let simd = unsafe { popcount_avx2(&data) };
        assert_eq!(simd, scalar);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_matches_scalar_when_available() {
        if !is_x86_feature_detected!("avx512vpopcntdq") || !is_x86_feature_detected!("avx512f") {
            return; // skip if AVX-512 VPOPCNTDQ not available on this CPU
        }
        let data: Vec<u64> = (0..1024).map(|i| i as u64).collect();
        let scalar = popcount_scalar(&data);
        let simd = unsafe { popcount_avx512vpopcntdq(&data) };
        assert_eq!(simd, scalar);
    }
}
