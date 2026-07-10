use crate::index::container::bitset::BITSET_WORDS;
use crate::index::container::run::Run;

/// Build a `[u64; BITSET_WORDS]` bitmask with every bit in every run set.
///
/// Dispatch:
///  - AVX-512F → fill whole-word ranges with `_mm512_storeu_si512` (8 u64s = 512 bits
///    per store), then handle the partial head/tail words with scalar masking.
///  - Scalar fallback → word-range fill loop.
///
/// Used by the Bitset × Run cross-container ops in `container::ops`: building the
/// run bitmask once and then invoking the existing `simd_support::bitwise` kernel
/// gives a single-pass fused bitwise+popcount when AVX-512 VPOPCNTDQ is present.
#[allow(unreachable_code)]
pub fn runs_to_bitmask(runs: &[Run]) -> Box<[u64; BITSET_WORDS]> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { runs_to_bitmask_avx512(runs) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { runs_to_bitmask_avx2(runs) };
        }
    }
    // SAFETY: NEON is a baseline feature on all aarch64 targets.
    #[cfg(target_arch = "aarch64")]
    return unsafe { runs_to_bitmask_neon(runs) };
    runs_to_bitmask_scalar(runs)
}

/// Set every bit position in `values` into `bits`.
///
/// More efficient than calling `BitsetContainer::insert()` per element because it
/// skips the exists-check branch: if a bit is already set the OR is a no-op.
/// Dispatch mirrors `runs_to_bitmask`.
pub fn set_bits_from_array(bits: &mut [u64; BITSET_WORDS], values: &[u16]) {
    for &v in values {
        let word = (v >> 6) as usize;
        let bit = 1u64 << (v & 63);
        bits[word] |= bit;
    }
}

/// Clear every bit position in `values` from `bits`.
///
/// Skips the exists-check branch in `BitsetContainer::remove()`.
pub fn clear_bits_from_array(bits: &mut [u64; BITSET_WORDS], values: &[u16]) {
    for &v in values {
        let word = (v >> 6) as usize;
        let bit = 1u64 << (v & 63);
        bits[word] &= !bit;
    }
}

// ── Scalar implementation ────────────────────────────────────────────────────

fn runs_to_bitmask_scalar(runs: &[Run]) -> Box<[u64; BITSET_WORDS]> {
    let mut bits = Box::new([0u64; BITSET_WORDS]);
    set_runs_into(&mut bits, runs);
    bits
}

/// Core run-fill logic shared by scalar and AVX-512 tail handling.
pub fn set_runs_into(bits: &mut [u64; BITSET_WORDS], runs: &[Run]) {
    for run in runs {
        let first = run.start as usize;
        let last = run.end() as usize; // inclusive

        let first_word = first >> 6; // first / 64
        let last_word = last >> 6;

        if first_word == last_word {
            // Run fits in a single word
            let mask = word_mask(first & 63, last & 63);
            bits[first_word] |= mask;
        } else {
            // Partial head word
            bits[first_word] |= !0u64 << (first & 63);
            // Full middle words
            bits[(first_word + 1)..last_word].fill(!0u64);
            // Partial tail word
            bits[last_word] |= word_mask(0, last & 63);
        }
    }
}

/// Build a u64 bitmask with bits [lo..=hi] set (both inclusive, within a single word).
#[inline]
fn word_mask(lo: usize, hi: usize) -> u64 {
    debug_assert!(lo <= hi && hi < 64);
    let len = hi - lo + 1;
    if len == 64 { !0u64 } else { ((1u64 << len) - 1) << lo }
}

// ── NEON implementation ───────────────────────────────────────────────────────
//
// Strategy mirrors the AVX2 path at 128-bit width: fill complete 2-word chunks
// with ALL_ONES using a Q-register store; partial head/tail words and the small
// scalar remainder (0–1 words) are handled identically to the scalar path.

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn runs_to_bitmask_neon(runs: &[Run]) -> Box<[u64; BITSET_WORDS]> {
    use std::arch::aarch64::*;

    let mut bits = Box::new([0u64; BITSET_WORDS]);
    let all_ones = vdupq_n_u64(!0u64);

    unsafe {
        for run in runs {
            let first = run.start as usize;
            let last = run.end() as usize;
            let first_word = first >> 6;
            let last_word = last >> 6;

            if first_word == last_word {
                bits[first_word] |= word_mask(first & 63, last & 63);
                continue;
            }

            // Partial head
            bits[first_word] |= !0u64 << (first & 63);

            let fill_start = first_word + 1;
            let fill_end = last_word; // exclusive

            // NEON fill: 2 words (128 bits) per store
            let chunks = (fill_end - fill_start) / 2;
            let neon_end = fill_start + chunks * 2;
            for chunk in 0..chunks {
                vst1q_u64(bits.as_mut_ptr().add(fill_start + chunk * 2), all_ones);
            }

            // Scalar fill of remaining full word (0–1)
            for w in neon_end..fill_end {
                bits[w] = !0u64;
            }

            // Partial tail
            bits[last_word] |= word_mask(0, last & 63);
        }
    }

    bits
}

// ── AVX2 implementation ───────────────────────────────────────────────────────
//
// Strategy mirrors the AVX-512 path at half the register width:
//   - AVX2 fills complete 4-word (256-bit) chunks with ALL_ONES using a YMM store.
//   - Partial head/tail words and the small scalar remainder (0–3 words) are
//     handled identically to the scalar path.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn runs_to_bitmask_avx2(runs: &[Run]) -> Box<[u64; BITSET_WORDS]> {
    use std::arch::x86_64::*;

    let mut bits = Box::new([0u64; BITSET_WORDS]);
    let all_ones = _mm256_set1_epi64x(-1i64);

    for run in runs {
        let first = run.start as usize;
        let last = run.end() as usize;
        let first_word = first >> 6;
        let last_word = last >> 6;

        if first_word == last_word {
            bits[first_word] |= word_mask(first & 63, last & 63);
            continue;
        }

        // Partial head
        bits[first_word] |= !0u64 << (first & 63);

        let fill_start = first_word + 1;
        let fill_end = last_word; // exclusive

        // AVX2 fill: 4 words (256 bits) per store
        let chunks = (fill_end - fill_start) / 4;
        let avx_end = fill_start + chunks * 4;
        for chunk in 0..chunks {
            unsafe {
                _mm256_storeu_si256(bits.as_mut_ptr().add(fill_start + chunk * 4) as *mut __m256i, all_ones);
            }
        }

        // Scalar fill of remaining full words (0–3)
        for w in avx_end..fill_end {
            bits[w] = !0u64;
        }

        // Partial tail
        bits[last_word] |= word_mask(0, last & 63);
    }

    bits
}

// ── AVX-512F implementation ───────────────────────────────────────────────────
//
// Strategy: iterate over runs and fill whole-word ranges using 512-bit stores.
//
// For a run spanning words [first_word, last_word]:
//   - scalar-fill the partial head word (0–63 bits)
//   - AVX-512 fill complete 8-word (512-bit) chunks with ALL_ONES ZMM
//   - scalar-fill any remaining full words (0–7) before the partial tail word
//   - scalar-fill the partial tail word
//
// The gain is on large runs (e.g. a million consecutive keys → 15,625 words to fill).

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn runs_to_bitmask_avx512(runs: &[Run]) -> Box<[u64; BITSET_WORDS]> {
    use std::arch::x86_64::*;

    let mut bits = Box::new([0u64; BITSET_WORDS]);

    // Pre-build the all-ones ZMM register
    let all_ones = _mm512_set1_epi64(-1i64);

    for run in runs {
        let first = run.start as usize;
        let last = run.end() as usize;
        let first_word = first >> 6;
        let last_word = last >> 6;

        if first_word == last_word {
            bits[first_word] |= word_mask(first & 63, last & 63);
            continue;
        }

        // Partial head
        bits[first_word] |= !0u64 << (first & 63);

        let fill_start = first_word + 1;
        let fill_end = last_word; // exclusive

        // AVX-512 fill: 8 words (512 bits) per store
        let chunks = (fill_end - fill_start) / 8;
        let avx_start = fill_start;
        let avx_end = fill_start + chunks * 8;
        for chunk in 0..chunks {
            unsafe {
                _mm512_storeu_si512(bits.as_mut_ptr().add(avx_start + chunk * 8) as *mut __m512i, all_ones);
            }
        }

        // Scalar fill of remaining full words (0–7)
        for w in avx_end..fill_end {
            bits[w] = !0u64;
        }

        // Partial tail
        bits[last_word] |= word_mask(0, last & 63);
    }

    bits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::container::run::RunContainer;

    fn bits_to_values(bits: &[u64; BITSET_WORDS]) -> Vec<u16> {
        let mut result = Vec::new();
        for (i, &w) in bits.iter().enumerate() {
            let mut word = w;
            while word != 0 {
                let pos = word.trailing_zeros() as u16;
                result.push(i as u16 * 64 + pos);
                word &= word - 1;
            }
        }
        result
    }

    #[test]
    fn single_run_within_one_word() {
        let runs = vec![Run::new(3, 4)]; // [3,7]
        let bits = runs_to_bitmask(&runs);
        let vals = bits_to_values(&bits);
        assert_eq!(vals, vec![3, 4, 5, 6, 7]);
    }

    #[test]
    fn run_spanning_multiple_words() {
        let runs = vec![Run::new(60, 7)]; // [60, 67]
        let bits = runs_to_bitmask(&runs);
        let vals = bits_to_values(&bits);
        assert_eq!(vals, (60u16..=67).collect::<Vec<_>>());
    }

    #[test]
    fn multiple_runs() {
        let runs = vec![Run::new(0, 4), Run::new(100, 9)]; // [0,4] and [100,109]
        let bits = runs_to_bitmask(&runs);
        let vals = bits_to_values(&bits);
        let expected: Vec<u16> = (0u16..=4).chain(100u16..=109).collect();
        assert_eq!(vals, expected);
    }

    #[test]
    fn large_run_matches_scalar() {
        let run = Run::new(0, 999); // [0,999] spans 16 full words
        let scalar = runs_to_bitmask_scalar(&[run]);
        let dispatch = runs_to_bitmask(&[run]);
        assert_eq!(scalar[..], dispatch[..]);
    }

    #[test]
    fn dispatch_matches_scalar_random() {
        let rc = RunContainer::from_sorted_values(&(0u16..1000).step_by(3).collect::<Vec<_>>());
        let scalar = runs_to_bitmask_scalar(rc.runs());
        let dispatch = runs_to_bitmask(rc.runs());
        assert_eq!(scalar[..], dispatch[..]);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar_when_available() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let run = Run::new(128, 511); // [128, 639] — 9 full words
        let scalar = runs_to_bitmask_scalar(&[run]);
        let simd = unsafe { runs_to_bitmask_avx2(&[run]) };
        assert_eq!(scalar[..], simd[..]);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar() {
        let run = Run::new(128, 511); // [128, 639] — 9 full words
        let scalar = runs_to_bitmask_scalar(&[run]);
        let simd = unsafe { runs_to_bitmask_neon(&[run]) };
        assert_eq!(scalar[..], simd[..]);
        // Also exercise a multi-run / odd-remainder case against scalar.
        let rc = RunContainer::from_sorted_values(&(0u16..1000).step_by(3).collect::<Vec<_>>());
        assert_eq!(runs_to_bitmask_scalar(rc.runs())[..], unsafe { runs_to_bitmask_neon(rc.runs()) }[..]);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_matches_scalar_when_available() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let run = Run::new(128, 511); // [128, 639] — 9 full words
        let scalar = runs_to_bitmask_scalar(&[run]);
        let simd = unsafe { runs_to_bitmask_avx512(&[run]) };
        assert_eq!(scalar[..], simd[..]);
    }

    #[test]
    fn set_bits_from_array_basic() {
        let mut bits = [0u64; BITSET_WORDS];
        set_bits_from_array(&mut bits, &[0, 1, 63, 64, 65535]);
        assert!(bits[0] & 1 != 0);
        assert!(bits[0] & 2 != 0);
        assert!(bits[0] >> 63 != 0);
        assert!(bits[1] & 1 != 0);
        assert!(bits[1023] >> 63 != 0);
    }

    #[test]
    fn clear_bits_from_array_basic() {
        let mut bits = [!0u64; BITSET_WORDS];
        clear_bits_from_array(&mut bits, &[0, 64, 128]);
        assert!(bits[0] & 1 == 0);
        assert!(bits[1] & 1 == 0);
        assert!(bits[2] & 1 == 0);
        // Other bits still set
        assert!(bits[0] & 2 != 0);
    }
}
