//! SIMD-accelerated packed inner-product kernel.
//!
//! Computes `Σ q[i]` for all `i` where bit `i` is set in a packed `&[u64]`
//! binary vector.  This is the hot path for single-bit RaBitQ estimation.
//!
//! Runtime dispatch selects the best available backend:
//! - **AVX-512F** (x86-64): 64 dims / iteration via `__mmask16` masked loads.
//! - **AVX2** (x86-64): 32 dims / iteration via sign-bit mask trick.
//! - **NEON** (aarch64): 4 dims / nibble via `vtstq_u32`.
//! - **Scalar** (fallback): set-bit iteration, branch-free inner loop.

// ── Scalar fallback ───────────────────────────────────────────────────────────

/// Scalar fallback: iterate over set bits and accumulate matching query values.
#[inline]
fn packed_ip_scalar(packed: &[u64], q: &[f32], dim: usize) -> f32 {
    let mut sum = 0.0f32;
    for (word_idx, &word) in packed.iter().enumerate() {
        let base = word_idx * 64;
        if base >= dim {
            break;
        }
        let mut w = word;
        while w != 0 {
            let bit = w.trailing_zeros() as usize;
            let idx = base + bit;
            if idx < dim {
                sum += q[idx];
            }
            w &= w - 1;
        }
    }
    sum
}

// ── aarch64 / NEON ────────────────────────────────────────────────────────────
// Processes 4 dims per nibble using a 4-lane f32 mask via `vtstq_u32`.

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn packed_ip_neon(packed: &[u64], q: &[f32], dim: usize) -> f32 {
    use std::arch::aarch64::*;
    unsafe {
        let q_ptr = q.as_ptr();
        let mut acc = vdupq_n_f32(0.0f32);
        let sel = vld1q_u32([1u32, 2u32, 4u32, 8u32].as_ptr());

        for (word_idx, &word) in packed.iter().enumerate() {
            let word_base = word_idx * 64;
            if word_base >= dim {
                break;
            }
            for nibble in 0..16usize {
                let dim_base = word_base + nibble * 4;
                if dim_base >= dim {
                    break;
                }
                let bits4 = ((word >> (nibble * 4)) & 0xF) as u32;
                let mask = vtstq_u32(vdupq_n_u32(bits4), sel);
                let q_vec = if dim_base + 4 <= dim {
                    vld1q_f32(q_ptr.add(dim_base))
                } else {
                    let mut tmp = [0.0f32; 4];
                    for k in 0..(dim - dim_base) {
                        tmp[k] = *q_ptr.add(dim_base + k);
                    }
                    vld1q_f32(tmp.as_ptr())
                };
                let masked = vreinterpretq_f32_u32(vandq_u32(mask, vreinterpretq_u32_f32(q_vec)));
                acc = vaddq_f32(acc, masked);
            }
        }
        let sum2 = vpadd_f32(vget_low_f32(acc), vget_high_f32(acc));
        let sum1 = vpadd_f32(sum2, sum2);
        vget_lane_f32(sum1, 0)
    }
}

// ── x86-64 / AVX2 ────────────────────────────────────────────────────────────
// Processes 32 dims (4 bytes) per outer iteration using 4 independent accumulators
// to break the FP-add dependency chain (latency 4 cycles, throughput 0.5).
// Mask generation: sign-bit trick — sllv shifts desired bit to bit-31, srai-31
// broadcasts it to all 32 bits — 2 ops vs and+cmpeq+xor (3 ops).

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn packed_ip_avx2(packed: &[u64], q: &[f32], dim: usize) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        let q_ptr = q.as_ptr();
        // Shifts to move bit k to position 31: lane k shifts left by (31 - k).
        // _mm256_set_epi32: arg0 = lane 7, arg7 = lane 0.
        let shifts = _mm256_set_epi32(24, 25, 26, 27, 28, 29, 30, 31);
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        // Process 4 bytes (32 dims) per outer iteration.
        let total_bytes = dim.div_ceil(8);
        let full_groups = (total_bytes / 4).min(packed.len() * 2);

        for g in 0..full_groups {
            let base = g * 32;
            let word = packed[g / 2];
            let byte_base = (g % 2) * 4;

            macro_rules! lane {
                ($off:expr, $acc:expr) => {{
                    let byte = ((word >> ((byte_base + $off) * 8)) & 0xFF) as i32;
                    let b = _mm256_set1_epi32(byte);
                    let mask = _mm256_castsi256_ps(_mm256_srai_epi32(_mm256_sllv_epi32(b, shifts), 31));
                    let q_v = _mm256_loadu_ps(q_ptr.add(base + $off * 8));
                    $acc = _mm256_add_ps($acc, _mm256_and_ps(mask, q_v));
                }};
            }
            lane!(0, acc0);
            lane!(1, acc1);
            lane!(2, acc2);
            lane!(3, acc3);
        }

        // Scalar tail for remaining bytes (at most 3, only for dims not divisible by 32).
        let tail_start = full_groups * 32;
        if tail_start < dim {
            let zero = _mm256_setzero_si256();
            let ones = _mm256_set1_epi32(-1i32);
            let sel = _mm256_set_epi32(0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01);
            for byte_idx in (full_groups * 4)..total_bytes {
                let dim_base = byte_idx * 8;
                if dim_base >= dim {
                    break;
                }
                let byte = {
                    let word = packed[byte_idx / 8];
                    ((word >> ((byte_idx % 8) * 8)) & 0xFF) as i32
                };
                let isolated = _mm256_and_si256(_mm256_set1_epi32(byte), sel);
                let mask = _mm256_castsi256_ps(_mm256_xor_si256(_mm256_cmpeq_epi32(isolated, zero), ones));
                let q_vec = if dim_base + 8 <= dim {
                    _mm256_loadu_ps(q_ptr.add(dim_base))
                } else {
                    let mut tmp = [0.0f32; 8];
                    for (k, val) in tmp.iter_mut().enumerate().take(dim - dim_base) {
                        *val = *q_ptr.add(dim_base + k);
                    }
                    _mm256_loadu_ps(tmp.as_ptr())
                };
                acc0 = _mm256_add_ps(acc0, _mm256_and_ps(mask, q_vec));
            }
        }

        // Reduce 4 accumulators → scalar
        let acc01 = _mm256_add_ps(acc0, acc1);
        let acc23 = _mm256_add_ps(acc2, acc3);
        let acc = _mm256_add_ps(acc01, acc23);
        let hi = _mm256_extractf128_ps(acc, 1);
        let lo = _mm256_castps256_ps128(acc);
        let s4 = _mm_add_ps(lo, hi);
        let sh = _mm_movehdup_ps(s4);
        let s2 = _mm_add_ps(s4, sh);
        let sh2 = _mm_movehl_ps(s2, s2);
        _mm_cvtss_f32(_mm_add_ss(s2, sh2))
    }
}

// ── x86-64 / AVX-512F ────────────────────────────────────────────────────────
// Processes 64 dims (one full u64 word = 4 × 16-bit groups) per outer iteration
// using 4 independent accumulators to break the 4-cycle FP-add dependency chain.
// maskz_loadu_ps zeros lanes where bit == 0 — no explicit multiply needed.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn packed_ip_avx512(packed: &[u64], q: &[f32], dim: usize) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        let q_ptr = q.as_ptr();
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();

        // Full u64 words: each word covers 64 dims across 4 × 16-bit groups.
        let full_words = (dim / 64).min(packed.len());
        for (word_idx, &word) in packed[..full_words].iter().enumerate() {
            let base = word_idx * 64;

            macro_rules! lane {
                ($g:expr, $acc:expr) => {{
                    let m: __mmask16 = ((word >> ($g * 16)) & 0xFFFF) as u16;
                    let v = _mm512_maskz_loadu_ps(m, q_ptr.add(base + $g * 16));
                    $acc = _mm512_add_ps($acc, v);
                }};
            }
            lane!(0, acc0);
            lane!(1, acc1);
            lane!(2, acc2);
            lane!(3, acc3);
        }

        // Tail: remaining dims not covered by a full u64 word.
        let tail_base = full_words * 64;
        if tail_base < dim && full_words < packed.len() {
            let word = packed[full_words];
            for group in 0..4usize {
                let dim_base = tail_base + group * 16;
                if dim_base >= dim {
                    break;
                }
                let bits16 = ((word >> (group * 16)) & 0xFFFF) as u16;
                let mask: __mmask16 = bits16;
                let q_vec = if dim_base + 16 <= dim {
                    _mm512_maskz_loadu_ps(mask, q_ptr.add(dim_base))
                } else {
                    let rem = dim - dim_base;
                    let partial = mask & ((1u16 << rem) - 1);
                    let mut tmp = [0.0f32; 16];
                    for (k, val) in tmp.iter_mut().enumerate().take(rem) {
                        *val = *q_ptr.add(dim_base + k);
                    }
                    _mm512_maskz_loadu_ps(partial, tmp.as_ptr())
                };
                acc0 = _mm512_add_ps(acc0, q_vec);
            }
        }

        // Reduce 4 accumulators → scalar
        let acc01 = _mm512_add_ps(acc0, acc1);
        let acc23 = _mm512_add_ps(acc2, acc3);
        let acc = _mm512_add_ps(acc01, acc23);
        let a = _mm512_extractf32x4_ps(acc, 0);
        let b = _mm512_extractf32x4_ps(acc, 1);
        let c = _mm512_extractf32x4_ps(acc, 2);
        let d = _mm512_extractf32x4_ps(acc, 3);
        let s4 = _mm_add_ps(_mm_add_ps(a, b), _mm_add_ps(c, d));
        let sh = _mm_movehdup_ps(s4);
        let s2 = _mm_add_ps(s4, sh);
        let sh2 = _mm_movehl_ps(s2, s2);
        _mm_cvtss_f32(_mm_add_ss(s2, sh2))
    }
}

// ── Runtime dispatch ──────────────────────────────────────────────────────────

/// Compute `Σ q[i]` for all `i` where bit `i` is set in `packed`.
///
/// Selects the best SIMD backend at runtime: AVX-512F → AVX2 → NEON → scalar.
#[allow(unreachable_code)]
pub(crate) fn packed_ip_best(packed: &[u64], q: &[f32], dim: usize) -> f32 {
    if packed.is_empty() || dim == 0 {
        return 0.0;
    }

    #[cfg(target_arch = "aarch64")]
    return unsafe { packed_ip_neon(packed, q, dim) };

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { packed_ip_avx512(packed, q, dim) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { packed_ip_avx2(packed, q, dim) };
        }
    }

    packed_ip_scalar(packed, q, dim)
}

// ── Multi-bit (u8 × f32) dot product ─────────────────────────────────────────
//
// Each packed u64 word stores 8 quantised u8 values (one per dimension, LE).
// Computes Σ bytes[i] * q[i] for i in 0..dim.
//
// The SIMD path for each backend:
//   AVX-512F : load 16 bytes → cvtepu8_epi32 → cvtepi32_ps → fmadd_ps  (16 d/iter)
//   AVX2     : load  8 bytes → cvtepu8_epi32 → cvtepi32_ps → mul+add   ( 8 d/iter)
//   NEON     : load  8 bytes → movl_u8 → movl_u16 (×2) → cvtq_f32 → vmla ( 8 d/iter)
//   Scalar   : sequential byte iteration (fallback)
//
// 4 independent accumulators break the 4-cycle FP-add dependency chain on all paths.

#[inline]
fn multi_bit_dot_scalar(packed: &[u64], q: &[f32], dim: usize) -> f32 {
    let mut dot = 0.0f32;
    let mut i = 0usize;
    'outer: for &word in packed {
        for b in word.to_le_bytes() {
            if i >= dim {
                break 'outer;
            }
            dot += b as f32 * q[i];
            i += 1;
        }
    }
    dot
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn multi_bit_dot_neon(packed: &[u64], q: &[f32], dim: usize) -> f32 {
    use std::arch::aarch64::*;
    unsafe {
        // SAFETY: u64 is at least u8-aligned; slice length is clamped to packed.len()*8.
        let bytes = std::slice::from_raw_parts(packed.as_ptr() as *const u8, packed.len() * 8);
        let q_ptr = q.as_ptr();
        let full_chunks = (dim / 8).min(bytes.len() / 8);

        let mut acc0 = vdupq_n_f32(0.0f32);
        let mut acc1 = vdupq_n_f32(0.0f32);

        for i in 0..full_chunks {
            let base = i * 8;
            let b8 = vld1_u8(bytes.as_ptr().add(base));
            let b16 = vmovl_u8(b8);
            let lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(b16)));
            let hi = vcvtq_f32_u32(vmovl_high_u16(b16));
            let q0 = vld1q_f32(q_ptr.add(base));
            let q1 = vld1q_f32(q_ptr.add(base + 4));
            acc0 = vmlaq_f32(acc0, lo, q0);
            acc1 = vmlaq_f32(acc1, hi, q1);
        }

        let acc = vaddq_f32(acc0, acc1);
        let sum2 = vpadd_f32(vget_low_f32(acc), vget_high_f32(acc));
        let sum1 = vpadd_f32(sum2, sum2);
        let mut sum = vget_lane_f32(sum1, 0);

        for i in (full_chunks * 8)..dim.min(bytes.len()) {
            sum += bytes[i] as f32 * q[i];
        }
        sum
    }
}

// Processes 8 dims per SIMD iteration; 4 independent accumulators avoid the
// 4-cycle FP-add dependency chain (latency 4, throughput 0.5 on Skylake+).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn multi_bit_dot_avx2(packed: &[u64], q: &[f32], dim: usize) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        // SAFETY: see multi_bit_dot_neon.
        let bytes = std::slice::from_raw_parts(packed.as_ptr() as *const u8, packed.len() * 8);
        let q_ptr = q.as_ptr();
        let full_chunks = (dim / 8).min(bytes.len() / 8);
        let full_quads = full_chunks / 4;

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        for i in 0..full_quads {
            let base = i * 32;
            macro_rules! chunk {
                ($off:expr, $acc:expr) => {{
                    let bp = bytes.as_ptr().add(base + $off * 8) as *const __m128i;
                    let b_f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(bp)));
                    let qv = _mm256_loadu_ps(q_ptr.add(base + $off * 8));
                    $acc = _mm256_add_ps($acc, _mm256_mul_ps(b_f32, qv));
                }};
            }
            chunk!(0, acc0);
            chunk!(1, acc1);
            chunk!(2, acc2);
            chunk!(3, acc3);
        }

        for i in (full_quads * 4)..full_chunks {
            let base = i * 8;
            let bp = bytes.as_ptr().add(base) as *const __m128i;
            let b_f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(bp)));
            let qv = _mm256_loadu_ps(q_ptr.add(base));
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(b_f32, qv));
        }

        let total = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        let hi = _mm256_extractf128_ps(total, 1);
        let lo = _mm256_castps256_ps128(total);
        let s4 = _mm_add_ps(lo, hi);
        let sh = _mm_movehdup_ps(s4);
        let s2 = _mm_add_ps(s4, sh);
        let sh2 = _mm_movehl_ps(s2, s2);
        let mut sum = _mm_cvtss_f32(_mm_add_ss(s2, sh2));

        for i in (full_chunks * 8)..dim.min(bytes.len()) {
            sum += bytes[i] as f32 * q[i];
        }
        sum
    }
}

// Processes 16 dims per SIMD iteration using AVX-512F fmadd.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn multi_bit_dot_avx512(packed: &[u64], q: &[f32], dim: usize) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        // SAFETY: see multi_bit_dot_neon.
        let bytes = std::slice::from_raw_parts(packed.as_ptr() as *const u8, packed.len() * 8);
        let q_ptr = q.as_ptr();
        let full_chunks = (dim / 16).min(bytes.len() / 16);
        let full_quads = full_chunks / 4;

        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();

        for i in 0..full_quads {
            let base = i * 64;
            macro_rules! chunk {
                ($off:expr, $acc:expr) => {{
                    let bp = bytes.as_ptr().add(base + $off * 16) as *const __m128i;
                    let b_f32 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(bp)));
                    let qv = _mm512_loadu_ps(q_ptr.add(base + $off * 16));
                    $acc = _mm512_fmadd_ps(b_f32, qv, $acc);
                }};
            }
            chunk!(0, acc0);
            chunk!(1, acc1);
            chunk!(2, acc2);
            chunk!(3, acc3);
        }

        for i in (full_quads * 4)..full_chunks {
            let base = i * 16;
            let bp = bytes.as_ptr().add(base) as *const __m128i;
            let b_f32 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(bp)));
            let qv = _mm512_loadu_ps(q_ptr.add(base));
            acc0 = _mm512_fmadd_ps(b_f32, qv, acc0);
        }

        let total = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
        let a = _mm512_extractf32x4_ps(total, 0);
        let b = _mm512_extractf32x4_ps(total, 1);
        let c = _mm512_extractf32x4_ps(total, 2);
        let d = _mm512_extractf32x4_ps(total, 3);
        let s4 = _mm_add_ps(_mm_add_ps(a, b), _mm_add_ps(c, d));
        let sh = _mm_movehdup_ps(s4);
        let s2 = _mm_add_ps(s4, sh);
        let sh2 = _mm_movehl_ps(s2, s2);
        let mut sum = _mm_cvtss_f32(_mm_add_ss(s2, sh2));

        for i in (full_chunks * 16)..dim.min(bytes.len()) {
            sum += bytes[i] as f32 * q[i];
        }
        sum
    }
}

/// Compute `Σ bytes[i] * q[i]` for a multi-bit quantised vector.
///
/// Each `u64` in `packed` stores 8 quantised u8 values (one per dimension,
/// little-endian).  Selects the best SIMD backend at runtime:
/// AVX-512F → AVX2 → NEON → scalar.
#[allow(unreachable_code)]
pub(crate) fn multi_bit_dot_best(packed: &[u64], q: &[f32], dim: usize) -> f32 {
    if packed.is_empty() || dim == 0 {
        return 0.0;
    }

    #[cfg(target_arch = "aarch64")]
    return unsafe { multi_bit_dot_neon(packed, q, dim) };

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { multi_bit_dot_avx512(packed, q, dim) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { multi_bit_dot_avx2(packed, q, dim) };
        }
    }

    multi_bit_dot_scalar(packed, q, dim)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_packed_and_query(dim: usize, seed: u64) -> (Vec<u64>, Vec<f32>) {
        let mut state = seed ^ 0x853c49e6748fea9b;
        let query: Vec<f32> = (0..dim)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state as i64 as f32) / (i64::MAX as f32)
            })
            .collect();
        // Pack random bits
        let mut state2 = seed ^ 0xdeadbeef;
        let bits: Vec<u64> = (0..dim.div_ceil(64))
            .map(|_| {
                state2 ^= state2 << 13;
                state2 ^= state2 >> 7;
                state2 ^= state2 << 17;
                state2
            })
            .collect();
        (bits, query)
    }

    fn reference_packed_ip(packed: &[u64], q: &[f32], dim: usize) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..dim {
            if (packed[i / 64] >> (i % 64)) & 1 == 1 {
                sum += q[i];
            }
        }
        sum
    }

    #[test]
    fn scalar_matches_reference() {
        for dim in [1, 7, 64, 65, 128, 256, 768] {
            let (packed, query) = make_packed_and_query(dim, dim as u64 * 31 + 7);
            let got = packed_ip_scalar(&packed, &query, dim);
            let expected = reference_packed_ip(&packed, &query, dim);
            assert!((got - expected).abs() < 1e-4, "dim={dim}: scalar={got} ref={expected}");
        }
    }

    #[test]
    fn best_matches_reference() {
        for dim in [1, 7, 64, 65, 128, 256, 768] {
            let (packed, query) = make_packed_and_query(dim, dim as u64 * 17 + 3);
            let got = packed_ip_best(&packed, &query, dim);
            let expected = reference_packed_ip(&packed, &query, dim);
            assert!((got - expected).abs() < 1e-4, "dim={dim}: best={got} ref={expected}");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_reference() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        for dim in [8, 32, 64, 128, 256, 768] {
            let (packed, query) = make_packed_and_query(dim, dim as u64 * 13 + 5);
            let got = unsafe { packed_ip_avx2(&packed, &query, dim) };
            let expected = reference_packed_ip(&packed, &query, dim);
            assert!((got - expected).abs() < 1e-4, "dim={dim}: avx2={got} ref={expected}");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_matches_reference() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        for dim in [16, 64, 128, 256, 768] {
            let (packed, query) = make_packed_and_query(dim, dim as u64 * 19 + 11);
            let got = unsafe { packed_ip_avx512(&packed, &query, dim) };
            let expected = reference_packed_ip(&packed, &query, dim);
            assert!((got - expected).abs() < 1e-4, "dim={dim}: avx512={got} ref={expected}");
        }
    }
}

#[cfg(test)]
mod multi_bit_tests {
    use super::*;

    fn make_multi_bit_packed_and_query(dim: usize, seed: u64) -> (Vec<u64>, Vec<f32>) {
        let mut state = seed ^ 0xdeadcafe_u64;
        let mut next = || -> u64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let bytes: Vec<u8> = (0..dim).map(|_| (next() & 0xFF) as u8).collect();
        let query: Vec<f32> = (0..dim).map(|_| (next() as i64 as f32) / (i64::MAX as f32)).collect();
        let packed: Vec<u64> = bytes
            .chunks(8)
            .map(|chunk| {
                let mut arr = [0u8; 8];
                arr[..chunk.len()].copy_from_slice(chunk);
                u64::from_le_bytes(arr)
            })
            .collect();
        (packed, query)
    }

    fn reference_multi_bit_dot(packed: &[u64], q: &[f32], dim: usize) -> f32 {
        multi_bit_dot_scalar(packed, q, dim)
    }

    #[test]
    fn scalar_matches_reference_multi_bit() {
        for dim in [1, 7, 8, 16, 64, 128, 768] {
            let (packed, query) = make_multi_bit_packed_and_query(dim, dim as u64 * 31 + 7);
            let got = multi_bit_dot_scalar(&packed, &query, dim);
            let expected = reference_multi_bit_dot(&packed, &query, dim);
            let tol = expected.abs().max(1.0) * 1e-5;
            assert!((got - expected).abs() < tol, "dim={dim}: scalar={got} ref={expected}");
        }
    }

    #[test]
    fn best_matches_reference_multi_bit() {
        for dim in [1, 7, 8, 16, 64, 128, 768] {
            let (packed, query) = make_multi_bit_packed_and_query(dim, dim as u64 * 17 + 3);
            let got = multi_bit_dot_best(&packed, &query, dim);
            let expected = reference_multi_bit_dot(&packed, &query, dim);
            // Tolerance covers FP summation-order differences across SIMD widths.
            let tol = expected.abs().max(1.0) * 1e-3;
            assert!((got - expected).abs() < tol, "dim={dim}: best={got} ref={expected}");
        }
    }

    #[test]
    fn empty_packed_returns_zero() {
        assert_eq!(multi_bit_dot_best(&[], &[1.0f32, 2.0], 2), 0.0);
        assert_eq!(multi_bit_dot_best(&[0u64], &[], 0), 0.0);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_reference_multi_bit() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        for dim in [8, 16, 32, 64, 128, 768] {
            let (packed, query) = make_multi_bit_packed_and_query(dim, dim as u64 * 13 + 5);
            let got = unsafe { multi_bit_dot_avx2(&packed, &query, dim) };
            let expected = reference_multi_bit_dot(&packed, &query, dim);
            let tol = expected.abs().max(1.0) * 1e-3;
            assert!((got - expected).abs() < tol, "dim={dim}: avx2={got} ref={expected}");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_matches_reference_multi_bit() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        for dim in [16, 32, 64, 128, 768] {
            let (packed, query) = make_multi_bit_packed_and_query(dim, dim as u64 * 19 + 11);
            let got = unsafe { multi_bit_dot_avx512(&packed, &query, dim) };
            let expected = reference_multi_bit_dot(&packed, &query, dim);
            let tol = expected.abs().max(1.0) * 1e-3;
            assert!((got - expected).abs() < tol, "dim={dim}: avx512={got} ref={expected}");
        }
    }
}
