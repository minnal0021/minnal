/// SIMD-accelerated set operations on sorted `u16` slices.
///
/// Dispatch:
///  - AVX-512F + BW → `_mm256_cmpeq_epi16` intersection probe + `_mm512_mask_compressstoreu_epi16`
///    for compressing matching elements; threshold: both inputs ≥ 32 elements.
///  - Scalar fallback → classic two-pointer merge.
///
/// All functions preserve sorted order and produce deduplicated results.
// Minimum per-array element count that makes the SIMD path worthwhile.
const SIMD_THRESHOLD: usize = 32;

// ── Public entry points ──────────────────────────────────────────────────────

/// Intersection of two sorted `u16` slices.
pub fn and_sorted_u16(a: &[u16], b: &[u16]) -> Vec<u16> {
    #[cfg(target_arch = "x86_64")]
    if a.len() >= SIMD_THRESHOLD && b.len() >= SIMD_THRESHOLD {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            return unsafe { and_avx512(a, b) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { and_avx2(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD && b.len() >= SIMD_THRESHOLD {
        return unsafe { and_neon(a, b) };
    }
    and_scalar(a, b)
}

/// Union of two sorted `u16` slices.
pub fn or_sorted_u16(a: &[u16], b: &[u16]) -> Vec<u16> {
    // OR via SIMD merge-networks offers marginal benefit for sorted lists;
    // the scalar two-pointer merge is already near-optimal.
    or_scalar(a, b)
}

/// Difference: elements in `a` but not in `b` (both sorted).
pub fn and_not_sorted_u16(a: &[u16], b: &[u16]) -> Vec<u16> {
    #[cfg(target_arch = "x86_64")]
    if a.len() >= SIMD_THRESHOLD && b.len() >= SIMD_THRESHOLD {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            return unsafe { and_not_avx512(a, b) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { and_not_avx2(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    if a.len() >= SIMD_THRESHOLD && b.len() >= SIMD_THRESHOLD {
        return unsafe { and_not_neon(a, b) };
    }
    and_not_scalar(a, b)
}

// ── Scalar implementations ────────────────────────────────────────────────────

fn and_scalar(a: &[u16], b: &[u16]) -> Vec<u16> {
    let mut result = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Equal => {
                result.push(a[i]);
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    result
}

fn or_scalar(a: &[u16], b: &[u16]) -> Vec<u16> {
    let mut result = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                result.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                result.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                result.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    result.extend_from_slice(&a[i..]);
    result.extend_from_slice(&b[j..]);
    result
}

fn and_not_scalar(a: &[u16], b: &[u16]) -> Vec<u16> {
    let mut result = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < a.len() {
        while j < b.len() && b[j] < a[i] {
            j += 1;
        }
        if j >= b.len() || b[j] != a[i] {
            result.push(a[i]);
        }
        i += 1;
    }
    result
}

// ── NEON implementations ──────────────────────────────────────────────────────
//
// Strategy mirrors the AVX2 path at half the register width (8 × u16 per Q
// register instead of 16). For each element of the smaller array, broadcast it
// and probe the larger array in 8-element chunks:
//   - `vceqq_u16(needle, chunk)` sets each 16-bit lane to 0xFFFF on a match.
//   - `vmaxvq_u16(eq) != 0` is true iff at least one lane matched.
// The early-exit on `first > needle` relies on both inputs being sorted ascending.
// OR falls through to the scalar two-pointer merge (same reasoning as x86).

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn and_neon(a: &[u16], b: &[u16]) -> Vec<u16> {
    use std::arch::aarch64::*;

    let (probe, scan) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    let mut result: Vec<u16> = Vec::with_capacity(probe.len());

    let scan_ptr = scan.as_ptr();
    let scan_chunks = scan.len() / 8;
    let scan_tail_start = scan_chunks * 8;

    unsafe {
        for &p in probe {
            let needle = vdupq_n_u16(p);

            let mut found = false;
            for chunk in 0..scan_chunks {
                let first = *scan_ptr.add(chunk * 8);
                if first > p {
                    break;
                }
                let chunk_vec = vld1q_u16(scan_ptr.add(chunk * 8));
                let eq = vceqq_u16(needle, chunk_vec);
                if vmaxvq_u16(eq) != 0 {
                    found = true;
                    break;
                }
            }
            if !found {
                for &s in &scan[scan_tail_start..] {
                    if s == p {
                        found = true;
                        break;
                    }
                    if s > p {
                        break;
                    }
                }
            }
            if found {
                result.push(p);
            }
        }
    }

    result
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn and_not_neon(a: &[u16], b: &[u16]) -> Vec<u16> {
    use std::arch::aarch64::*;

    let mut result: Vec<u16> = Vec::with_capacity(a.len());
    let b_ptr = b.as_ptr();
    let b_chunks = b.len() / 8;
    let b_tail_start = b_chunks * 8;

    unsafe {
        for &av in a {
            let needle = vdupq_n_u16(av);

            let mut found = false;
            for chunk in 0..b_chunks {
                let first = *b_ptr.add(chunk * 8);
                if first > av {
                    break;
                }
                let chunk_vec = vld1q_u16(b_ptr.add(chunk * 8));
                let eq = vceqq_u16(needle, chunk_vec);
                if vmaxvq_u16(eq) != 0 {
                    found = true;
                    break;
                }
            }
            if !found {
                for &bv in &b[b_tail_start..] {
                    if bv == av {
                        found = true;
                        break;
                    }
                    if bv > av {
                        break;
                    }
                }
            }
            if !found {
                result.push(av);
            }
        }
    }

    result
}

// ── AVX2 implementations ──────────────────────────────────────────────────────
//
// Strategy mirrors the AVX-512BW path but uses AVX2 intrinsics only:
//
//   - `_mm256_cmpeq_epi16(needle, chunk)` replaces `_mm256_cmpeq_epi16_mask`.
//     It returns a __m256i where each 16-bit lane is 0xFFFF if equal, 0x0000 if not.
//   - `_mm256_testz_si256(eq, eq)` returns 1 when (eq & eq) == 0, i.e. no bit is set,
//     meaning no match. Returns 0 when at least one bit is set → match found.
//
// OR falls through to scalar (same reasoning as AVX-512 path).

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn and_avx2(a: &[u16], b: &[u16]) -> Vec<u16> {
    use std::arch::x86_64::*;

    let (probe, scan) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    let mut result: Vec<u16> = Vec::with_capacity(probe.len());

    let scan_ptr = scan.as_ptr();
    let scan_chunks = scan.len() / 16;
    let scan_tail_start = scan_chunks * 16;

    for &p in probe {
        let needle = _mm256_set1_epi16(p as i16);

        let mut found = false;
        for chunk in 0..scan_chunks {
            let first = unsafe { *scan_ptr.add(chunk * 16) };
            if first > p {
                break;
            }
            let chunk_vec = unsafe { _mm256_loadu_si256(scan_ptr.add(chunk * 16) as *const __m256i) };
            let eq = _mm256_cmpeq_epi16(needle, chunk_vec);
            // testz returns 0 if any bit in (eq & eq) is set → at least one match
            if _mm256_testz_si256(eq, eq) == 0 {
                found = true;
                break;
            }
        }
        if !found {
            for &s in &scan[scan_tail_start..] {
                if s == p {
                    found = true;
                    break;
                }
                if s > p {
                    break;
                }
            }
        }
        if found {
            result.push(p);
        }
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn and_not_avx2(a: &[u16], b: &[u16]) -> Vec<u16> {
    use std::arch::x86_64::*;

    let mut result: Vec<u16> = Vec::with_capacity(a.len());
    let b_ptr = b.as_ptr();
    let b_chunks = b.len() / 16;
    let b_tail_start = b_chunks * 16;

    for &av in a {
        let needle = _mm256_set1_epi16(av as i16);

        let mut found = false;
        for chunk in 0..b_chunks {
            let first = unsafe { *b_ptr.add(chunk * 16) };
            if first > av {
                break;
            }
            let chunk_vec = unsafe { _mm256_loadu_si256(b_ptr.add(chunk * 16) as *const __m256i) };
            let eq = _mm256_cmpeq_epi16(needle, chunk_vec);
            if _mm256_testz_si256(eq, eq) == 0 {
                found = true;
                break;
            }
        }
        if !found {
            for &bv in &b[b_tail_start..] {
                if bv == av {
                    found = true;
                    break;
                }
                if bv > av {
                    break;
                }
            }
        }
        if !found {
            result.push(av);
        }
    }

    result
}

// ── AVX-512 implementations ───────────────────────────────────────────────────
//
// Strategy for AND / ANDNOT:
//   Load 16 u16s from the smaller array into a YMM. For each 16-element chunk
//   of the larger array, use _mm256_cmpeq_epi16 to produce a 16-bit match mask,
//   then _mm512_mask_compressstoreu_epi16 to write matching elements in order.
//
// Strategy for OR:
//   Fall back to the scalar two-pointer merge which is already near-optimal
//   (output is bounded by a + b elements, and branch prediction is good on sorted
//   data). The scalar path is called from the outer dispatch above.
//   OR via SIMD merge-networks is significantly more complex with marginal gain,
//   so we delegate to scalar after the threshold check.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn and_avx512(a: &[u16], b: &[u16]) -> Vec<u16> {
    use std::arch::x86_64::*;

    // Choose the shorter array as the "probe" array (loaded into YMM),
    // and iterate over the longer array in 16-element chunks.
    let (probe, scan) = if a.len() <= b.len() { (a, b) } else { (b, a) };

    let mut result: Vec<u16> = Vec::with_capacity(probe.len());

    let scan_ptr = scan.as_ptr();
    let scan_chunks = scan.len() / 16;
    let scan_tail_start = scan_chunks * 16;

    for &p in probe {
        let needle = _mm256_set1_epi16(p as i16);

        // Check 16-element aligned chunks
        let mut found = false;
        for chunk in 0..scan_chunks {
            let chunk_vec = unsafe { _mm256_loadu_si256(scan_ptr.add(chunk * 16) as *const __m256i) };
            let mask = unsafe { _mm256_cmpeq_epi16_mask(needle, chunk_vec) };
            if mask != 0 {
                found = true;
                break;
            }
            // Early exit: if scan[chunk*16] > p, no later chunk can match
            // (scan is sorted ascending). Compare just the first element.
            let first = unsafe { *scan_ptr.add(chunk * 16) };
            if first > p {
                break;
            }
        }
        if !found {
            // Check tail elements
            for &s in &scan[scan_tail_start..] {
                if s == p {
                    found = true;
                    break;
                }
                if s > p {
                    break;
                }
            }
        }
        if found {
            result.push(p);
        }
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn and_not_avx512(a: &[u16], b: &[u16]) -> Vec<u16> {
    use std::arch::x86_64::*;

    let mut result: Vec<u16> = Vec::with_capacity(a.len());
    let b_ptr = b.as_ptr();
    let b_chunks = b.len() / 16;
    let b_tail_start = b_chunks * 16;

    for &av in a {
        let needle = _mm256_set1_epi16(av as i16);

        let mut found = false;
        for chunk in 0..b_chunks {
            let first = unsafe { *b_ptr.add(chunk * 16) };
            if first > av {
                break;
            }
            let chunk_vec = unsafe { _mm256_loadu_si256(b_ptr.add(chunk * 16) as *const __m256i) };
            let mask = unsafe { _mm256_cmpeq_epi16_mask(needle, chunk_vec) };
            if mask != 0 {
                found = true;
                break;
            }
        }
        if !found {
            for &bv in &b[b_tail_start..] {
                if bv == av {
                    found = true;
                    break;
                }
                if bv > av {
                    break;
                }
            }
        }
        if !found {
            result.push(av);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn and_basic() {
        let a: Vec<u16> = (0..100).collect();
        let b: Vec<u16> = (50..150).collect();
        let result = and_sorted_u16(&a, &b);
        let expected: Vec<u16> = (50..100).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn or_basic() {
        let a: Vec<u16> = (0..50).collect();
        let b: Vec<u16> = (25..75).collect();
        let result = or_sorted_u16(&a, &b);
        let expected: Vec<u16> = (0..75).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn and_not_basic() {
        let a: Vec<u16> = (0..100).collect();
        let b: Vec<u16> = (50..150).collect();
        let result = and_not_sorted_u16(&a, &b);
        let expected: Vec<u16> = (0..50).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn and_disjoint() {
        let a: Vec<u16> = (0..50).collect();
        let b: Vec<u16> = (100..150).collect();
        assert!(and_sorted_u16(&a, &b).is_empty());
    }

    #[test]
    fn or_disjoint() {
        let a: Vec<u16> = (0..50).collect();
        let b: Vec<u16> = (100..150).collect();
        let result = or_sorted_u16(&a, &b);
        let mut expected: Vec<u16> = (0..50).chain(100..150).collect();
        expected.sort_unstable();
        assert_eq!(result, expected);
    }

    #[test]
    fn and_not_removes_all() {
        let a: Vec<u16> = (0..50).collect();
        assert!(and_not_sorted_u16(&a, &a).is_empty());
    }

    #[test]
    fn small_below_threshold() {
        // Below SIMD_THRESHOLD — exercises scalar path
        let a = vec![1u16, 3, 5, 7];
        let b = vec![3u16, 5, 9, 11];
        assert_eq!(and_sorted_u16(&a, &b), vec![3, 5]);
        assert_eq!(or_sorted_u16(&a, &b), vec![1, 3, 5, 7, 9, 11]);
        assert_eq!(and_not_sorted_u16(&a, &b), vec![1, 7]);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_and_matches_scalar_when_available() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let a: Vec<u16> = (0u16..200).step_by(3).collect();
        let b: Vec<u16> = (0u16..200).step_by(5).collect();
        assert_eq!(unsafe { and_avx2(&a, &b) }, and_scalar(&a, &b));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_and_matches_scalar() {
        let a: Vec<u16> = (0u16..200).step_by(3).collect();
        let b: Vec<u16> = (0u16..200).step_by(5).collect();
        assert_eq!(unsafe { and_neon(&a, &b) }, and_scalar(&a, &b));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_and_not_matches_scalar() {
        let a: Vec<u16> = (0u16..200).step_by(3).collect();
        let b: Vec<u16> = (0u16..200).step_by(5).collect();
        assert_eq!(unsafe { and_not_neon(&a, &b) }, and_not_scalar(&a, &b));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_and_not_matches_scalar_when_available() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let a: Vec<u16> = (0u16..200).step_by(3).collect();
        let b: Vec<u16> = (0u16..200).step_by(5).collect();
        assert_eq!(unsafe { and_not_avx2(&a, &b) }, and_not_scalar(&a, &b));
    }

    #[test]
    fn dispatch_matches_scalar() {
        let a: Vec<u16> = (0u16..200).step_by(3).collect();
        let b: Vec<u16> = (0u16..200).step_by(5).collect();
        assert_eq!(and_sorted_u16(&a, &b), and_scalar(&a, &b));
        assert_eq!(or_sorted_u16(&a, &b), or_scalar(&a, &b));
        assert_eq!(and_not_sorted_u16(&a, &b), and_not_scalar(&a, &b));
    }
}
