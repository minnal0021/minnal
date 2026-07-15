//! SIMD-accelerated lexicographical byte array comparison.
//!
//! This module provides optimized comparison for byte arrays using:
//! - AVX512 for x86_64 targets with avx512f support
//! - AVX2 for x86_64 targets with avx2 but without avx512f
//! - NEON for aarch64 targets (Advanced SIMD is baseline-mandatory on AArch64)
//! - Fallback to scalar comparison for other targets
//!
//! The comparison maintains strict lexicographic byte order across all implementations.

#[allow(clippy::module_inception)]
pub mod simd_support {
    use core::cmp::Ordering;

    /// Compare two byte arrays lexicographically, using SIMD acceleration when available.
    ///
    /// This function maintains the same semantics as `a.cmp(b)` but may use vector instructions
    /// for faster comparison on supported CPUs.
    #[allow(dead_code)]
    #[inline]
    pub fn compare_bytes_simd(a: &[u8], b: &[u8]) -> Ordering {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        {
            compare_bytes_avx512(a, b)
        }

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "avx512f")))]
        {
            compare_bytes_avx2(a, b)
        }

        // SAFETY: NEON (Advanced SIMD) is a baseline feature on all aarch64 targets,
        // so the `target_feature(enable = "neon")` fn is always safe to call here.
        #[cfg(target_arch = "aarch64")]
        {
            unsafe { compare_bytes_neon(a, b) }
        }

        #[cfg(not(any(
            all(target_arch = "x86_64", target_feature = "avx512f"),
            all(target_arch = "x86_64", target_feature = "avx2"),
            target_arch = "aarch64",
        )))]
        {
            a.cmp(b)
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    fn compare_bytes_avx512(a: &[u8], b: &[u8]) -> Ordering {
        use std::arch::x86_64::*;

        let mut a_pos = 0;
        let mut b_pos = 0;
        let a_len = a.len();
        let b_len = b.len();

        // Compare 64-byte chunks using AVX512
        while a_pos + 64 <= a_len && b_pos + 64 <= b_len {
            unsafe {
                let a_vec = _mm512_loadu_epi8(a.as_ptr().add(a_pos) as *const i8);
                let b_vec = _mm512_loadu_epi8(b.as_ptr().add(b_pos) as *const i8);

                // Find first differing byte
                let eq_mask = _mm512_cmpeq_epi8_mask(a_vec, b_vec);

                // If not all equal, find the first differing byte
                if eq_mask != u64::MAX {
                    // Find the position of the first unequal byte
                    let diff_pos = eq_mask.trailing_ones() as usize;
                    let a_byte = a[a_pos + diff_pos];
                    let b_byte = b[b_pos + diff_pos];

                    return a_byte.cmp(&b_byte);
                }
            }

            a_pos += 64;
            b_pos += 64;
        }

        // Handle remaining bytes with 32-byte chunks if both are available
        if a_pos + 32 <= a_len && b_pos + 32 <= b_len {
            unsafe {
                // Use AVX512 with 32-byte loads (cast to appropriate type)
                let a_vec = _mm256_loadu_si256(a.as_ptr().add(a_pos) as *const __m256i);
                let b_vec = _mm256_loadu_si256(b.as_ptr().add(b_pos) as *const __m256i);

                let eq_mask = _mm256_cmpeq_epi8(a_vec, b_vec);
                let cmp_result = _mm256_cmpeq_epi8(eq_mask, _mm256_set1_epi8(-1));

                // If any bytes differ
                if _mm256_movemask_epi8(cmp_result) != -1 {
                    // Fall back to scalar comparison for this chunk
                    for i in 0..32 {
                        if a_pos + i < a_len && b_pos + i < b_len {
                            let cmp = a[a_pos + i].cmp(&b[b_pos + i]);
                            if cmp != Ordering::Equal {
                                return cmp;
                            }
                        } else {
                            break;
                        }
                    }
                }
            }
            a_pos += 32;
            b_pos += 32;
        }

        // Handle remaining bytes with scalar comparison
        let min_remaining = (a_len - a_pos).min(b_len - b_pos);
        for i in 0..min_remaining {
            let cmp = a[a_pos + i].cmp(&b[b_pos + i]);
            if cmp != Ordering::Equal {
                return cmp;
            }
        }

        // Compare lengths if all compared bytes are equal
        a_len.cmp(&b_len)
    }

    /// AVX2 path: compares 32 bytes per iteration using 256-bit YMM registers.
    ///
    /// `_mm256_cmpeq_epi8` produces a mask with all bits set in each byte lane where
    /// bytes are equal. `_mm256_movemask_epi8` collapses this to a 32-bit integer
    /// (1 bit per byte), so all-equal gives `0xFFFF_FFFF`. The position of the first
    /// differing byte is found via `trailing_ones()` on the equality mask.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "avx512f")))]
    fn compare_bytes_avx2(a: &[u8], b: &[u8]) -> Ordering {
        use std::arch::x86_64::*;

        let min_len = a.len().min(b.len());
        let mut pos = 0;

        while pos + 32 <= min_len {
            unsafe {
                let a_vec = _mm256_loadu_si256(a.as_ptr().add(pos) as *const __m256i);
                let b_vec = _mm256_loadu_si256(b.as_ptr().add(pos) as *const __m256i);
                let eq = _mm256_cmpeq_epi8(a_vec, b_vec);
                // movemask_epi8 returns 1 per byte where top bit is 1.
                // cmpeq sets all 8 bits of matching lanes → mask bit = 1 means equal.
                let mask = _mm256_movemask_epi8(eq) as u32;
                if mask != u32::MAX {
                    // trailing_ones() = index of first 0 bit = first differing byte
                    let diff_pos = mask.trailing_ones() as usize;
                    return a[pos + diff_pos].cmp(&b[pos + diff_pos]);
                }
            }
            pos += 32;
        }

        // Scalar tail
        for i in pos..min_len {
            match a[i].cmp(&b[i]) {
                Ordering::Equal => {}
                other => return other,
            }
        }

        a.len().cmp(&b.len())
    }

    /// NEON path: compares 16 bytes per iteration using 128-bit Q registers.
    ///
    /// AArch64 has no direct `PMOVMSKB` equivalent, so the per-lane equality
    /// result from `vceqq_u8` (0xFF in each lane where the bytes match) is
    /// collapsed to a 16-bit mask — 1 bit per byte, set where equal — by ANDing
    /// with per-lane bit weights and horizontally summing each 8-byte half
    /// (`vaddv_u8`). All-equal therefore yields `0xFFFF`; the first differing
    /// byte is `trailing_ones()` of that mask, mirroring the AVX2 path exactly.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn compare_bytes_neon(a: &[u8], b: &[u8]) -> Ordering {
        use std::arch::aarch64::*;

        unsafe {
            let min_len = a.len().min(b.len());
            let mut pos = 0;

            // Per-byte bit weights for the movemask emulation (lane i → bit i).
            let bit_weights =
                vld1q_u8([1u8, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128].as_ptr());

            while pos + 16 <= min_len {
                let a_vec = vld1q_u8(a.as_ptr().add(pos));
                let b_vec = vld1q_u8(b.as_ptr().add(pos));
                // 0xFF per lane where equal, 0x00 where different.
                let eq = vceqq_u8(a_vec, b_vec);
                let masked = vandq_u8(eq, bit_weights);
                // Sum each 8-byte half into one bit-packed byte, combine to a u16.
                let lo = vaddv_u8(vget_low_u8(masked)) as u16;
                let hi = vaddv_u8(vget_high_u8(masked)) as u16;
                let mask = lo | (hi << 8);
                if mask != u16::MAX {
                    // trailing_ones() = index of first 0 bit = first differing byte.
                    let diff_pos = mask.trailing_ones() as usize;
                    return a[pos + diff_pos].cmp(&b[pos + diff_pos]);
                }
                pos += 16;
            }

            // Scalar tail.
            for i in pos..min_len {
                match a[i].cmp(&b[i]) {
                    Ordering::Equal => {}
                    other => return other,
                }
            }

            a.len().cmp(&b.len())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_compare_equal_small() {
            assert_eq!(compare_bytes_simd(b"hello", b"hello"), Ordering::Equal);
        }

        #[test]
        fn test_compare_less() {
            assert_eq!(compare_bytes_simd(b"abc", b"def"), Ordering::Less);
        }

        #[test]
        fn test_compare_greater() {
            assert_eq!(compare_bytes_simd(b"xyz", b"abc"), Ordering::Greater);
        }

        #[test]
        fn test_compare_prefix() {
            assert_eq!(compare_bytes_simd(b"hello", b"hello world"), Ordering::Less);
            assert_eq!(compare_bytes_simd(b"hello world", b"hello"), Ordering::Greater);
        }

        #[test]
        fn test_compare_64byte_chunks() {
            // Test with keys that span multiple 64-byte chunks
            let key_a: Vec<u8> = (0..200).map(|i| (i % 256) as u8).collect();
            let mut key_b = key_a.clone();
            key_b[80] = key_b[80].wrapping_add(1);

            assert_eq!(compare_bytes_simd(&key_a, &key_b), Ordering::Less);
            assert_eq!(compare_bytes_simd(&key_b, &key_a), Ordering::Greater);
        }

        #[test]
        fn test_compare_32byte_chunks() {
            let key_a: Vec<u8> = (0..80).map(|i| (i % 256) as u8).collect();
            let mut key_b = key_a.clone();
            key_b[40] = key_b[40].wrapping_add(1);

            assert_eq!(compare_bytes_simd(&key_a, &key_b), Ordering::Less);
            assert_eq!(compare_bytes_simd(&key_b, &key_a), Ordering::Greater);
        }

        #[test]
        fn test_compare_scalar_tail() {
            let key_a = vec![0u8; 75];
            let mut key_b = key_a.clone();
            key_b[70] = 1;

            assert_eq!(compare_bytes_simd(&key_a, &key_b), Ordering::Less);
        }

        #[test]
        fn test_compare_length_difference() {
            let key_a = b"short";
            let key_b = b"shorter";

            assert_eq!(compare_bytes_simd(key_a, key_b), Ordering::Less);
            assert_eq!(compare_bytes_simd(key_b, key_a), Ordering::Greater);
        }

        #[test]
        fn test_compare_empty() {
            assert_eq!(compare_bytes_simd(b"", b""), Ordering::Equal);
            assert_eq!(compare_bytes_simd(b"", b"a"), Ordering::Less);
            assert_eq!(compare_bytes_simd(b"a", b""), Ordering::Greater);
        }

        #[test]
        fn test_consistency_with_slice_cmp() {
            // Verify that SIMD results match standard slice comparison
            let test_cases = vec![
                (b"".to_vec(), b"".to_vec()),
                (b"a".to_vec(), b"a".to_vec()),
                (b"a".to_vec(), b"b".to_vec()),
                (b"abc".to_vec(), b"abc".to_vec()),
                (b"abc".to_vec(), b"def".to_vec()),
                (b"abc".to_vec(), b"ab".to_vec()),
                (vec![0u8; 100], vec![0u8; 100]),
                (vec![0u8; 100], {
                    let mut v = vec![0u8; 100];
                    v[99] = 1;
                    v
                }),
                (vec![1u8; 200], vec![1u8; 200]),
                (
                    (0..200).map(|i| (i % 256) as u8).collect::<Vec<_>>(),
                    (0..200).map(|i| (i % 256) as u8).collect::<Vec<_>>(),
                ),
            ];

            for (a, b) in test_cases {
                let simd_result = compare_bytes_simd(&a, &b);
                let slice_result = a.cmp(&b);
                assert_eq!(
                    simd_result,
                    slice_result,
                    "SIMD and slice comparison mismatch for {:?} vs {:?}",
                    String::from_utf8_lossy(&a),
                    String::from_utf8_lossy(&b)
                );
            }
        }
    }
}
