use crate::index::container::bitset::BITSET_WORDS;

/// Cardinality threshold above which we use the AVX-512 + lookup table path.
/// At 1024 elements the average bits-per-word is 1, making the lookup approach
/// equally cheap to scalar; above this it starts to win.
pub const DENSE_EXTRACT_THRESHOLD: usize = 1024;

// ── Compile-time lookup tables ───────────────────────────────────────────────

/// For each byte value `b` (0–255), `BIT_POSITIONS[b]` holds the bit positions
/// (0–7) of every set bit in `b`, in ascending order. Unused trailing entries are 0.
static BIT_POSITIONS: [[u8; 8]; 256] = {
    let mut table = [[0u8; 8]; 256];
    let mut byte = 0usize;
    while byte < 256 {
        let mut count = 0usize;
        let mut b = byte;
        while b != 0 {
            let pos = b.trailing_zeros() as u8;
            table[byte][count] = pos;
            count += 1;
            b &= b - 1; // clear lowest set bit
        }
        byte += 1;
    }
    table
};

/// `BIT_COUNTS[b]` — number of set bits in byte value `b` (0–8).
static BIT_COUNTS: [u8; 256] = {
    let mut table = [0u8; 256];
    let mut byte = 0usize;
    while byte < 256 {
        table[byte] = byte.count_ones() as u8;
        byte += 1;
    }
    table
};

// ── Public entry point ───────────────────────────────────────────────────────

/// Extract the positions of all set bits from a bitset into a sorted `Vec<u16>`.
///
/// Dispatch:
///  - `cardinality >= DENSE_EXTRACT_THRESHOLD` **and** AVX-512F available →
///    AVX-512 word scan (8 u64s per ZMM) + byte-level lookup table extraction.
///  - Otherwise → scalar `trailing_zeros` + `blsr` loop (unchanged behaviour).
#[allow(unreachable_code)]
pub fn extract_bitset_to_values(bits: &[u64; BITSET_WORDS], cardinality: usize) -> Vec<u16> {
    if cardinality >= DENSE_EXTRACT_THRESHOLD {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                // SAFETY: avx512f detected above.
                return unsafe { extract_dense_avx512(bits, cardinality) };
            }
            if is_x86_feature_detected!("avx2") {
                // SAFETY: avx2 detected above.
                return unsafe { extract_dense_avx2(bits, cardinality) };
            }
        }
        // SAFETY: NEON is a baseline feature on all aarch64 targets.
        #[cfg(target_arch = "aarch64")]
        return unsafe { extract_dense_neon(bits, cardinality) };
        // Neither AVX-512 nor AVX2: use lookup table without SIMD word scan.
        return extract_dense_lookup(bits, cardinality);
    }

    extract_scalar(bits, cardinality)
}

// ── Scalar path (existing logic, baseline) ───────────────────────────────────

/// Original scalar extraction — one bit at a time via `trailing_zeros` + `blsr`.
/// Best for sparse bitsets (few bits set per word).
fn extract_scalar(bits: &[u64; BITSET_WORDS], cardinality: usize) -> Vec<u16> {
    let mut result = Vec::with_capacity(cardinality);
    for (i, &word) in bits.iter().enumerate() {
        let mut w = word;
        while w != 0 {
            let bit_pos = w.trailing_zeros() as u16;
            result.push((i as u16) * 64 + bit_pos);
            w &= w - 1;
        }
    }
    result
}

// ── Byte lookup table path (no intrinsics, cache-friendly) ───────────────────

/// Lookup-table extraction without SIMD intrinsics.
/// Reduces the inner loop from ≤64 iterations (bit-by-bit) to exactly 8
/// iterations (byte-by-byte) per word, regardless of density.
fn extract_dense_lookup(bits: &[u64; BITSET_WORDS], cardinality: usize) -> Vec<u16> {
    let mut result = Vec::with_capacity(cardinality);
    for (word_idx, &word) in bits.iter().enumerate() {
        if word == 0 {
            continue;
        }
        push_word_positions(word, (word_idx * 64) as u16, &mut result);
    }
    result
}

// ── NEON path (dense bitsets, aarch64) ──────────────────────────────────────
//
// Strategy mirrors the AVX2/AVX-512 paths at 128-bit width:
//   1. Load 2 u64 words (128 bits) per iteration into a Q register.
//   2. Reinterpret as u32×4 and reduce with `vmaxvq_u32`; a zero result means
//      both words are zero, so the whole group is skipped with one branch.
//   3. For each non-zero word, use the byte lookup table to push positions.

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn extract_dense_neon(bits: &[u64; BITSET_WORDS], cardinality: usize) -> Vec<u16> {
    use std::arch::aarch64::*;

    let mut result = Vec::with_capacity(cardinality);
    let chunks = BITSET_WORDS / 2; // 1024 / 2 = 512 chunks

    unsafe {
        for chunk in 0..chunks {
            let base_word = chunk * 2;
            let v = vld1q_u64(bits.as_ptr().add(base_word));

            // Skip the whole 2-word group when every bit is zero.
            if vmaxvq_u32(vreinterpretq_u32_u64(v)) == 0 {
                continue;
            }

            let w0 = vgetq_lane_u64(v, 0);
            if w0 != 0 {
                push_word_positions(w0, (base_word * 64) as u16, &mut result);
            }
            let w1 = vgetq_lane_u64(v, 1);
            if w1 != 0 {
                push_word_positions(w1, ((base_word + 1) * 64) as u16, &mut result);
            }
        }
    }

    result
}

// ── AVX2 path (dense bitsets, AVX2 required) ────────────────────────────────
//
// Strategy (mirrors the AVX-512 path at half the register width):
//   1. Load 4 u64 words (256 bits) per iteration into a YMM register.
//   2. Compare each 64-bit lane to zero with `_mm256_cmpeq_epi64` → a YMM where
//      matching (zero) lanes are all-1s. Reinterpret as `__m256d` and use
//      `_mm256_movemask_pd` to collapse the sign-bit of each 64-bit lane into
//      a 4-bit integer; invert to get the *non-zero* lane mask.
//   3. Skip the entire group when the mask is 0 (all 4 words are zero).
//   4. For each non-zero lane, use the byte lookup table to push positions.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn extract_dense_avx2(bits: &[u64; BITSET_WORDS], cardinality: usize) -> Vec<u16> {
    use std::arch::x86_64::*;

    let mut result = Vec::with_capacity(cardinality);
    let zero = _mm256_setzero_si256();
    let chunks = BITSET_WORDS / 4; // 1024 / 4 = 256 chunks

    for chunk in 0..chunks {
        let base_word = chunk * 4;

        let v = unsafe { _mm256_loadu_si256(bits.as_ptr().add(base_word) as *const __m256i) };

        // Compare each 64-bit lane to zero; result lanes are 0xFFFF…FFFF where zero.
        let eq_zero = _mm256_cmpeq_epi64(v, zero);
        // movemask_pd takes the top bit (sign bit) of each 64-bit lane.
        // Lanes that were zero have all bits set → sign bit = 1 → mask bit = 1.
        let zero_mask = _mm256_movemask_pd(_mm256_castsi256_pd(eq_zero)) as u8;
        let nonzero_mask: u8 = (!zero_mask) & 0x0F; // 4-bit mask of non-zero lanes

        if nonzero_mask == 0 {
            continue;
        }

        let mut mask = nonzero_mask;
        while mask != 0 {
            let lane = mask.trailing_zeros() as usize;
            mask &= mask - 1;

            let word_idx = base_word + lane;
            let word = bits[word_idx];
            push_word_positions(word, (word_idx * 64) as u16, &mut result);
        }
    }

    result
}

// ── AVX-512 path (dense bitsets, AVX-512F required) ─────────────────────────

/// AVX-512F-accelerated extraction for dense bitsets.
///
/// Strategy:
///  1. Load 8 u64 words (512 bits) per loop iteration into a ZMM register.
///  2. Compare each 64-bit lane to zero with `_mm512_cmpeq_epi64_mask` →
///     an 8-bit mask; invert to get the *non-zero* lanes mask.
///  3. Skip the entire group when the mask is 0 (all 8 words are zero).
///  4. For each non-zero lane, use the byte lookup table to push positions.
///
/// Step 3 eliminates a costly inner loop for empty word groups, which is
/// common even in dense bitsets when specific ranges of values are absent.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn extract_dense_avx512(bits: &[u64; BITSET_WORDS], cardinality: usize) -> Vec<u16> {
    use std::arch::x86_64::*;

    let mut result = Vec::with_capacity(cardinality);
    let zero = _mm512_setzero_si512();
    let chunks = BITSET_WORDS / 8; // 1024 / 8 = 128 chunks

    for chunk in 0..chunks {
        let base_word = chunk * 8;

        // Load 8 u64 words unaligned into a ZMM register
        let v = unsafe { _mm512_loadu_si512(bits.as_ptr().add(base_word) as *const __m512i) };

        // `_mm512_cmpeq_epi64_mask` → 1 bit per lane where lane == 0.
        // Invert (!): 1 bit per lane where lane is *non-zero*.
        let zero_mask: __mmask8 = _mm512_cmpeq_epi64_mask(v, zero);
        let nonzero_mask: u8 = !zero_mask;

        if nonzero_mask == 0 {
            // All 8 words are zero — skip with a single branch
            continue;
        }

        // Iterate over the non-zero lane indices using the bitmask
        let mut mask = nonzero_mask;
        while mask != 0 {
            let lane = mask.trailing_zeros() as usize; // index within the chunk
            mask &= mask - 1; // clear lowest set bit

            let word_idx = base_word + lane;
            let word = bits[word_idx];
            push_word_positions(word, (word_idx * 64) as u16, &mut result);
        }
    }

    result
}

// ── Shared helper ────────────────────────────────────────────────────────────

/// Expand all set bits of a single u64 `word` into `out`, adding `base` to each
/// position. Uses the byte lookup table — 8 iterations per word instead of ≤64.
#[inline(always)]
fn push_word_positions(word: u64, base: u16, out: &mut Vec<u16>) {
    for byte_idx in 0..8u16 {
        let byte = ((word >> (byte_idx * 8)) & 0xFF) as usize;
        if byte == 0 {
            continue;
        }
        let byte_base = base + byte_idx * 8;
        let n = BIT_COUNTS[byte] as usize;
        for j in 0..n {
            // SAFETY: j < n = popcount(byte) ≤ 8, and BIT_POSITIONS rows are [u8; 8]
            out.push(byte_base + unsafe { *BIT_POSITIONS.get_unchecked(byte).get_unchecked(j) } as u16);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::container::bitset::BitsetContainer;

    fn make_bits(values: &[u16]) -> Box<[u64; BITSET_WORDS]> {
        let mut bits = Box::new([0u64; BITSET_WORDS]);
        for &v in values {
            let word = (v >> 6) as usize;
            let bit = 1u64 << (v & 63);
            bits[word] |= bit;
        }
        bits
    }

    #[test]
    fn scalar_path_sparse() {
        let values: Vec<u16> = (0..10).map(|i| i * 7).collect();
        let bits = make_bits(&values);
        let mut expected = values.clone();
        expected.sort_unstable();
        let result = extract_scalar(&bits, values.len());
        assert_eq!(result, expected);
    }

    #[test]
    fn lookup_path_matches_scalar() {
        let values: Vec<u16> = (0..2000u16).step_by(3).collect();
        let bits = make_bits(&values);
        let scalar = extract_scalar(&bits, values.len());
        let lookup = extract_dense_lookup(&bits, values.len());
        assert_eq!(lookup, scalar);
    }

    #[test]
    fn dispatch_sparse_uses_scalar_path() {
        // cardinality < threshold → scalar path
        let values: Vec<u16> = (0..100).collect();
        let bits = make_bits(&values);
        let result = extract_bitset_to_values(&bits, values.len());
        assert_eq!(result, values);
    }

    #[test]
    fn dispatch_dense_matches_scalar() {
        // cardinality >= threshold → SIMD/lookup path, result must match scalar
        let values: Vec<u16> = (0..4096).collect();
        let bits = make_bits(&values);
        let scalar = extract_scalar(&bits, values.len());
        let dense = extract_bitset_to_values(&bits, values.len());
        assert_eq!(dense, scalar);
    }

    #[test]
    fn dispatch_full_bitset() {
        // All 65536 bits set
        let values: Vec<u16> = (0..=65535).collect();
        let bits = make_bits(&values);
        let scalar = extract_scalar(&bits, 65536);
        let dense = extract_bitset_to_values(&bits, 65536);
        assert_eq!(dense, scalar);
    }

    #[test]
    fn dispatch_empty_bitset() {
        let bits = Box::new([0u64; BITSET_WORDS]);
        let result = extract_bitset_to_values(&bits, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn dispatch_large_values() {
        // Values in upper range (> 32768)
        let values: Vec<u16> = (32768u16..=36864).collect();
        let bits = make_bits(&values);
        let scalar = extract_scalar(&bits, values.len());
        let dense = extract_bitset_to_values(&bits, values.len());
        assert_eq!(dense, scalar);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_path_matches_scalar() {
        let values: Vec<u16> = (0..4096).step_by(2).collect();
        let bits = make_bits(&values);
        let scalar = extract_scalar(&bits, values.len());
        let simd = unsafe { extract_dense_neon(&bits, values.len()) };
        assert_eq!(simd, scalar);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_path_full_bitset() {
        let values: Vec<u16> = (0..=65535).collect();
        let bits = make_bits(&values);
        let simd = unsafe { extract_dense_neon(&bits, 65536) };
        assert_eq!(simd, values);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_path_matches_scalar_when_available() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let values: Vec<u16> = (0..4096).step_by(2).collect();
        let bits = make_bits(&values);
        let scalar = extract_scalar(&bits, values.len());
        let simd = unsafe { extract_dense_avx2(&bits, values.len()) };
        assert_eq!(simd, scalar);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_path_matches_scalar_when_available() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let values: Vec<u16> = (0..4096).step_by(2).collect();
        let bits = make_bits(&values);
        let scalar = extract_scalar(&bits, values.len());
        let simd = unsafe { extract_dense_avx512(&bits, values.len()) };
        assert_eq!(simd, scalar);
    }

    #[test]
    fn bitset_container_to_values_consistent() {
        let values: Vec<u16> = (0..4096).collect();
        let c = BitsetContainer::from_values(&values);
        let result = c.to_values();
        assert_eq!(result, values);
    }
}
