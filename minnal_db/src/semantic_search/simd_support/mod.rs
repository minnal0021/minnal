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

        // Process 4 bytes (32 dims) per outer iteration. A group's last lane
        // (`off = 3`) does a 256-bit load of q[base+24 .. base+32), so a group is
        // only in-bounds when the full 32 dims exist: `(g+1)*32 <= dim`. Counting
        // groups as `dim / 32` (floor) — NOT `ceil(dim/8) / 4`, which rounds a
        // final partial 32-dim group up to "full" and reads past `q` for dims like
        // 25..31, 57..63, 89..95 (e.g. dim=31 would load q[24..32) over a 31-len q).
        // The scalar tail below handles every remaining sub-32 dim safely.
        let total_bytes = dim.div_ceil(8);
        let full_groups = (dim / 32).min(packed.len() * 2);

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

// ── Bit-plane (2- or 4-bit chunk code × f32) dot product ─────────────────────
//
// A bit-plane code stores `bits` planes of `dim.div_ceil(64)` words each, plane
// `j` holding bit `j` of every dimension's code. `Σ code[i] · q[i]` is
// `Σ_j 2^j · Σ_{i: bit j of code[i]} q[i]`.

/// `Σ_j 2^j · packed_ip(plane_j, q)` for a code of `bits` planes, plane by
/// plane: the reference every faster form must agree with.
fn bit_planes_ip_per_plane(packed: &[u64], q: &[f32], dim: usize, bits: usize) -> f32 {
    let words = dim.div_ceil(64);
    packed
        .chunks_exact(words)
        .take(bits)
        .enumerate()
        .map(|(j, plane)| (1u32 << j) as f32 * packed_ip_best(plane, q, dim))
        .sum()
}

/// One pass for every plane: each 16 dimensions of `q` are loaded once and added,
/// under each plane's 16-bit mask, into that plane's accumulator. Requires `dim`
/// to be a multiple of 64 and `1 <= bits <= 4`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn bit_planes_ip_avx512(packed: &[u64], q: &[f32], dim: usize, bits: usize) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        let words = dim / 64;
        let q_ptr = q.as_ptr();
        let mut acc = [_mm512_setzero_ps(); 4];
        for w in 0..words {
            for g in 0..4 {
                let v = _mm512_loadu_ps(q_ptr.add(w * 64 + g * 16));
                for (j, a) in acc.iter_mut().enumerate().take(bits) {
                    let m: __mmask16 = ((packed[j * words + w] >> (g * 16)) & 0xFFFF) as u16;
                    *a = _mm512_mask_add_ps(*a, m, *a, v);
                }
            }
        }
        let mut total = _mm512_setzero_ps();
        for (j, a) in acc.iter().enumerate().take(bits) {
            total = _mm512_fmadd_ps(*a, _mm512_set1_ps((1u32 << j) as f32), total);
        }
        _mm512_reduce_add_ps(total)
    }
}

/// `Σ code[i] · q[i]` for a bit-plane code `bits` wide (see
/// [`QuantisationStyle::BitPlanes`](crate::semantic_search::QuantisationStyle::BitPlanes)).
/// `packed` must hold `bits` planes of `dim.div_ceil(64)` words. Selects the
/// fused AVX-512F kernel when the CPU has it and the shape fits (768 dimensions
/// do), and otherwise sums [`packed_ip_best`] over the planes.
#[allow(unreachable_code)]
pub(crate) fn bit_planes_ip_best(packed: &[u64], q: &[f32], dim: usize, bits: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if dim.is_multiple_of(64)
            && (1..=4).contains(&bits)
            && packed.len() >= bits * (dim / 64)
            && q.len() >= dim
            && is_x86_feature_detected!("avx512f")
        {
            return unsafe { bit_planes_ip_avx512(packed, q, dim, bits) };
        }
    }
    bit_planes_ip_per_plane(packed, q, dim, bits)
}

// ── Reconstructing a code ────────────────────────────────────────────────────
//
// A 1-bit code reconstructs to `c + s` where its bit is set and `c - s` elsewhere.
// A bit-plane code reconstructs to `c - s * (u + shift)`, with `u` the dimension's
// code and `shift = -(2^(bits-1) - 0.5)`. Every backend uses separate multiply and
// subtract (no FMA), so all give bit-for-bit the scalar result, and a k-means run
// does not depend on the CPU.
//
//   AVX-512F : 16 dims / iter, blend or masked add under the code's 16-bit mask
//   AVX2     :  8 dims / iter, a lane mask per bit
//   NEON     :  4 dims / iter, a lane mask per bit via `vtstq_u32`
//   Scalar   : the tail of every backend, and the fallback

/// Scalar 1-bit reconstruction of dimensions `from..`.
///
/// # Safety
/// `out` must be valid for writes of `centre.len()` values.
unsafe fn single_bit_scalar(packed: &[u64], centre: &[f32], s: f32, from: usize, out: *mut f32) {
    for (i, &c) in centre.iter().enumerate().skip(from) {
        let v = if (packed[i / 64] >> (i % 64)) & 1 == 1 { c + s } else { c - s };
        // SAFETY: `i < centre.len()`, and the caller gives room for that many.
        unsafe { out.add(i).write(v) };
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn single_bit_neon(packed: &[u64], centre: &[f32], s: f32, out: *mut f32) {
    use std::arch::aarch64::*;
    unsafe {
        let n = centre.len() / 4 * 4;
        let sv = vdupq_n_f32(s);
        let sel = vld1q_u32([1u32, 2, 4, 8].as_ptr());
        for i in (0..n).step_by(4) {
            let bits = ((packed[i / 64] >> (i % 64)) & 0xF) as u32;
            let mask = vtstq_u32(vdupq_n_u32(bits), sel);
            let c = vld1q_f32(centre.as_ptr().add(i));
            vst1q_f32(out.add(i), vbslq_f32(mask, vaddq_f32(c, sv), vsubq_f32(c, sv)));
        }
        single_bit_scalar(packed, centre, s, n, out);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn single_bit_avx2(packed: &[u64], centre: &[f32], s: f32, out: *mut f32) {
    use std::arch::x86_64::*;
    unsafe {
        let n = centre.len() / 8 * 8;
        let sv = _mm256_set1_ps(s);
        let sel = _mm256_set_epi32(0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01);
        for i in (0..n).step_by(8) {
            let byte = ((packed[i / 64] >> (i % 64)) & 0xFF) as i32;
            let mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(_mm256_and_si256(_mm256_set1_epi32(byte), sel), sel));
            let c = _mm256_loadu_ps(centre.as_ptr().add(i));
            _mm256_storeu_ps(out.add(i), _mm256_blendv_ps(_mm256_sub_ps(c, sv), _mm256_add_ps(c, sv), mask));
        }
        single_bit_scalar(packed, centre, s, n, out);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn single_bit_avx512(packed: &[u64], centre: &[f32], s: f32, out: *mut f32) {
    use std::arch::x86_64::*;
    unsafe {
        let n = centre.len() / 16 * 16;
        let sv = _mm512_set1_ps(s);
        for i in (0..n).step_by(16) {
            let mask: __mmask16 = ((packed[i / 64] >> (i % 64)) & 0xFFFF) as u16;
            let c = _mm512_loadu_ps(centre.as_ptr().add(i));
            _mm512_storeu_ps(out.add(i), _mm512_mask_blend_ps(mask, _mm512_sub_ps(c, sv), _mm512_add_ps(c, sv)));
        }
        single_bit_scalar(packed, centre, s, n, out);
    }
}

/// Write `centre[i] + s` where bit `i` of `packed` is set and `centre[i] - s`
/// elsewhere into `out` (replacing its contents). `packed` must hold
/// `centre.len().div_ceil(64)` words.
#[allow(unreachable_code)]
pub(crate) fn reconstruct_single_bit_best(packed: &[u64], centre: &[f32], s: f32, out: &mut Vec<f32>) {
    assert!(packed.len() >= centre.len().div_ceil(64), "code shorter than its dimension");
    out.clear();
    out.reserve(centre.len());
    let ptr = out.as_mut_ptr();
    // SAFETY: `ptr` has room for `centre.len()` values, each kernel writes all of
    // them, and each feature is checked (NEON is baseline on aarch64).
    unsafe {
        single_bit_dispatch(packed, centre, s, ptr);
        out.set_len(centre.len());
    }
}

#[allow(unreachable_code)]
unsafe fn single_bit_dispatch(packed: &[u64], centre: &[f32], s: f32, ptr: *mut f32) {
    unsafe {
        #[cfg(target_arch = "aarch64")]
        return single_bit_neon(packed, centre, s, ptr);
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return single_bit_avx512(packed, centre, s, ptr);
            }
            if is_x86_feature_detected!("avx2") {
                return single_bit_avx2(packed, centre, s, ptr);
            }
        }
        single_bit_scalar(packed, centre, s, 0, ptr);
    }
}

/// Scalar bit-plane reconstruction of dimensions `from..`.
///
/// # Safety
/// `out` must be valid for writes of `centre.len()` values.
unsafe fn bit_planes_scalar(packed: &[u64], centre: &[f32], s: f32, bits: usize, from: usize, out: *mut f32) {
    let words = centre.len().div_ceil(64);
    let shift = -(((1u32 << (bits - 1)) as f32) - 0.5);
    for (i, &c) in centre.iter().enumerate().skip(from) {
        let code = (0..bits).fold(0u32, |code, j| code | (((packed[j * words + i / 64] >> (i % 64)) & 1) as u32) << j);
        // SAFETY: `i < centre.len()`, and the caller gives room for that many.
        unsafe { out.add(i).write(c - s * (code as f32 + shift)) };
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn bit_planes_neon(packed: &[u64], centre: &[f32], s: f32, bits: usize, out: *mut f32) {
    use std::arch::aarch64::*;
    unsafe {
        let n = centre.len() / 4 * 4;
        let words = centre.len().div_ceil(64);
        let (sv, shift) = (vdupq_n_f32(s), vdupq_n_f32(-(((1u32 << (bits - 1)) as f32) - 0.5)));
        let sel = vld1q_u32([1u32, 2, 4, 8].as_ptr());
        for i in (0..n).step_by(4) {
            let mut u = vdupq_n_u32(0);
            for j in 0..bits {
                let plane = ((packed[j * words + i / 64] >> (i % 64)) & 0xF) as u32;
                u = vorrq_u32(u, vandq_u32(vtstq_u32(vdupq_n_u32(plane), sel), vdupq_n_u32(1 << j)));
            }
            let code = vaddq_f32(vcvtq_f32_u32(u), shift);
            let c = vld1q_f32(centre.as_ptr().add(i));
            vst1q_f32(out.add(i), vsubq_f32(c, vmulq_f32(sv, code)));
        }
        bit_planes_scalar(packed, centre, s, bits, n, out);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn bit_planes_avx2(packed: &[u64], centre: &[f32], s: f32, bits: usize, out: *mut f32) {
    use std::arch::x86_64::*;
    unsafe {
        let n = centre.len() / 8 * 8;
        let words = centre.len().div_ceil(64);
        let (sv, shift) = (_mm256_set1_ps(s), _mm256_set1_ps(-(((1u32 << (bits - 1)) as f32) - 0.5)));
        let sel = _mm256_set_epi32(0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01);
        for i in (0..n).step_by(8) {
            let mut u = _mm256_setzero_si256();
            for j in 0..bits {
                let byte = ((packed[j * words + i / 64] >> (i % 64)) & 0xFF) as i32;
                let set = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_set1_epi32(byte), sel), sel);
                u = _mm256_or_si256(u, _mm256_and_si256(set, _mm256_set1_epi32(1 << j)));
            }
            let code = _mm256_add_ps(_mm256_cvtepi32_ps(u), shift);
            let c = _mm256_loadu_ps(centre.as_ptr().add(i));
            _mm256_storeu_ps(out.add(i), _mm256_sub_ps(c, _mm256_mul_ps(sv, code)));
        }
        bit_planes_scalar(packed, centre, s, bits, n, out);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn bit_planes_avx512(packed: &[u64], centre: &[f32], s: f32, bits: usize, out: *mut f32) {
    use std::arch::x86_64::*;
    unsafe {
        let n = centre.len() / 16 * 16;
        let words = centre.len().div_ceil(64);
        let (sv, shift) = (_mm512_set1_ps(s), _mm512_set1_ps(-(((1u32 << (bits - 1)) as f32) - 0.5)));
        for i in (0..n).step_by(16) {
            let mut u = _mm512_setzero_si512();
            for j in 0..bits {
                let mask: __mmask16 = ((packed[j * words + i / 64] >> (i % 64)) & 0xFFFF) as u16;
                u = _mm512_mask_or_epi32(u, mask, u, _mm512_set1_epi32(1 << j));
            }
            let code = _mm512_add_ps(_mm512_cvtepi32_ps(u), shift);
            let c = _mm512_loadu_ps(centre.as_ptr().add(i));
            _mm512_storeu_ps(out.add(i), _mm512_sub_ps(c, _mm512_mul_ps(sv, code)));
        }
        bit_planes_scalar(packed, centre, s, bits, n, out);
    }
}

/// Write `centre[i] - s * (u[i] + shift)` into `out` (replacing its contents), for
/// a bit-plane code `bits` wide (`1..=8`). `packed` must hold `bits` planes of
/// `centre.len().div_ceil(64)` words.
pub(crate) fn reconstruct_bit_planes_best(packed: &[u64], centre: &[f32], s: f32, bits: usize, out: &mut Vec<f32>) {
    assert!((1..=8).contains(&bits), "bit-plane width out of range");
    assert!(packed.len() >= centre.len().div_ceil(64) * bits, "code shorter than its dimension");
    out.clear();
    out.reserve(centre.len());
    let ptr = out.as_mut_ptr();
    // SAFETY: as in `reconstruct_single_bit_best`.
    unsafe {
        bit_planes_dispatch(packed, centre, s, bits, ptr);
        out.set_len(centre.len());
    }
}

#[allow(unreachable_code)]
unsafe fn bit_planes_dispatch(packed: &[u64], centre: &[f32], s: f32, bits: usize, ptr: *mut f32) {
    unsafe {
        #[cfg(target_arch = "aarch64")]
        return bit_planes_neon(packed, centre, s, bits, ptr);
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return bit_planes_avx512(packed, centre, s, bits, ptr);
            }
            if is_x86_feature_detected!("avx2") {
                return bit_planes_avx2(packed, centre, s, bits, ptr);
            }
        }
        bit_planes_scalar(packed, centre, s, bits, 0, ptr);
    }
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

    /// Random codes and a centre for the reconstruction kernels.
    fn reconstruction_case(dim: usize, bits: usize, seed: u64) -> (Vec<u64>, Vec<f32>) {
        let mut state = seed;
        let mut next = move || {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let z = (state ^ (state >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z ^ (z >> 27)
        };
        let packed = (0..dim.div_ceil(64) * bits).map(|_| next()).collect();
        let centre = (0..dim).map(|_| (next() % 2000) as f32 / 1000.0 - 1.0).collect();
        (packed, centre)
    }

    const RECONSTRUCTION_DIMS: [usize; 13] = [0, 1, 3, 4, 7, 8, 15, 16, 17, 63, 64, 65, 768];

    /// The written form of each reconstruction, as a reference for the kernels.
    fn reference_single_bit(packed: &[u64], centre: &[f32], s: f32) -> Vec<f32> {
        (0..centre.len())
            .map(|i| {
                if (packed[i / 64] >> (i % 64)) & 1 == 1 {
                    centre[i] + s
                } else {
                    centre[i] - s
                }
            })
            .collect()
    }

    fn reference_bit_planes(packed: &[u64], centre: &[f32], s: f32, bits: usize) -> Vec<f32> {
        use crate::semantic_search::quantisation::rabitq::quantisation_support::bit_plane_code;
        let shift = -(((1u32 << (bits - 1)) as f32) - 0.5);
        (0..centre.len())
            .map(|i| centre[i] - s * (bit_plane_code(packed, centre.len(), bits, i) as f32 + shift))
            .collect()
    }

    /// Bit-for-bit, so that a clustering does not depend on the CPU.
    fn assert_same_bits(got: &[f32], want: &[f32], what: &str) {
        let (got, want): (Vec<u32>, Vec<u32>) = (got.iter().map(|x| x.to_bits()).collect(), want.iter().map(|x| x.to_bits()).collect());
        assert_eq!(got, want, "{what}");
    }

    #[test]
    fn best_reconstruction_matches_reference() {
        for dim in RECONSTRUCTION_DIMS {
            let (packed, centre) = reconstruction_case(dim, 1, dim as u64);
            let mut out = vec![9.0; 3];
            reconstruct_single_bit_best(&packed, &centre, 0.037, &mut out);
            assert_same_bits(&out, &reference_single_bit(&packed, &centre, 0.037), &format!("1-bit dim {dim}"));
            for bits in 1..=8 {
                let (packed, centre) = reconstruction_case(dim, bits, dim as u64 + 100);
                reconstruct_bit_planes_best(&packed, &centre, -0.037, bits, &mut out);
                assert_same_bits(
                    &out,
                    &reference_bit_planes(&packed, &centre, -0.037, bits),
                    &format!("{bits}-bit dim {dim}"),
                );
            }
        }
    }

    /// Each kernel this CPU can run (`best` only reaches the widest), and the
    /// scalar fallback, against the reference.
    #[test]
    fn every_reconstruction_kernel_matches_reference() {
        type SingleBit = unsafe fn(&[u64], &[f32], f32, *mut f32);
        type BitPlanes = unsafe fn(&[u64], &[f32], f32, usize, *mut f32);
        unsafe fn single_scalar(p: &[u64], c: &[f32], s: f32, out: *mut f32) {
            unsafe { single_bit_scalar(p, c, s, 0, out) }
        }
        unsafe fn planes_scalar(p: &[u64], c: &[f32], s: f32, b: usize, out: *mut f32) {
            unsafe { bit_planes_scalar(p, c, s, b, 0, out) }
        }
        let mut kernels: Vec<(&str, SingleBit, BitPlanes)> = vec![("scalar", single_scalar, planes_scalar)];
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                kernels.push(("avx2", single_bit_avx2, bit_planes_avx2));
            }
            if is_x86_feature_detected!("avx512f") {
                kernels.push(("avx512", single_bit_avx512, bit_planes_avx512));
            }
        }
        #[cfg(target_arch = "aarch64")]
        kernels.push(("neon", single_bit_neon, bit_planes_neon));
        for (name, single, planes) in kernels {
            for dim in RECONSTRUCTION_DIMS {
                let (packed, centre) = reconstruction_case(dim, 1, dim as u64);
                let mut out = vec![0.0f32; dim];
                unsafe { single(&packed, &centre, 0.037, out.as_mut_ptr()) };
                assert_same_bits(&out, &reference_single_bit(&packed, &centre, 0.037), &format!("{name} 1-bit dim {dim}"));
                for bits in 1..=8 {
                    let (packed, centre) = reconstruction_case(dim, bits, dim as u64 + 100);
                    unsafe { planes(&packed, &centre, 0.037, bits, out.as_mut_ptr()) };
                    let want = reference_bit_planes(&packed, &centre, 0.037, bits);
                    assert_same_bits(&out, &want, &format!("{name} {bits}-bit dim {dim}"));
                }
            }
        }
    }

    #[test]
    fn bit_planes_kernel_matches_plane_by_plane() {
        for (dim, seed) in [(64usize, 1u64), (768, 2), (1024, 3), (96, 4), (100, 5)] {
            for bits in [2usize, 4] {
                let (_, q) = make_packed_and_query(dim, seed);
                let words = dim.div_ceil(64);
                let mut state = seed ^ 0x5151;
                let packed: Vec<u64> = (0..words * bits)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        state
                    })
                    .collect();
                let fast = bit_planes_ip_best(&packed, &q, dim, bits);
                let reference: f32 = (0..bits)
                    .map(|j| (1u32 << j) as f32 * reference_packed_ip(&packed[j * words..(j + 1) * words], &q, dim))
                    .sum();
                assert!(
                    (fast - reference).abs() <= 1e-3 * (1.0 + reference.abs()),
                    "dim {dim} bits {bits}: {fast} vs {reference}"
                );
            }
        }
    }

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

    // Dims that exercise a partial final 32-/64-dim group — the AVX2 single-bit
    // OOB-read regression (groups counted as ceil(dim/8)/4 instead of dim/32).
    // 25,57,89 sit in [n*32+1, n*32+8]; 31,63,95 are one short of a full group;
    // 33 is one past. With q.len() == dim, the old code read q[dim] here.
    const PARTIAL_GROUP_DIMS: [usize; 7] = [25, 31, 33, 57, 63, 89, 95];

    #[test]
    fn scalar_matches_reference() {
        for dim in [1, 7, 64, 65, 128, 256, 768].into_iter().chain(PARTIAL_GROUP_DIMS) {
            let (packed, query) = make_packed_and_query(dim, dim as u64 * 31 + 7);
            let got = packed_ip_scalar(&packed, &query, dim);
            let expected = reference_packed_ip(&packed, &query, dim);
            assert!((got - expected).abs() < 1e-4, "dim={dim}: scalar={got} ref={expected}");
        }
    }

    #[test]
    fn best_matches_reference() {
        for dim in [1, 7, 64, 65, 128, 256, 768].into_iter().chain(PARTIAL_GROUP_DIMS) {
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
        for dim in [8, 32, 64, 128, 256, 768].into_iter().chain(PARTIAL_GROUP_DIMS) {
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
        for dim in [16, 64, 128, 256, 768].into_iter().chain(PARTIAL_GROUP_DIMS) {
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
