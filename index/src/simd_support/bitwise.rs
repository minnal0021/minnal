use crate::container::bitset::BITSET_WORDS;

/// Result of a SIMD bitwise operation: the output words and their total popcount.
pub struct BitwiseResult {
    pub bits: Box<[u64; BITSET_WORDS]>,
    pub cardinality: usize,
}

// ── Public entry points ──────────────────────────────────────────────────────

/// AND: `a & b` — intersection.
#[allow(unreachable_code)]
pub fn and(a: &[u64; BITSET_WORDS], b: &[u64; BITSET_WORDS]) -> BitwiseResult {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vpopcntdq") && is_x86_feature_detected!("avx512f") {
            return unsafe { and_avx512_popcnt(a, b) };
        }
        if is_x86_feature_detected!("avx512f") {
            return unsafe { and_avx512(a, b) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { and_avx2(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    return unsafe { and_neon(a, b) };
    and_scalar(a, b)
}

/// OR: `a | b` — union.
#[allow(unreachable_code)]
pub fn or(a: &[u64; BITSET_WORDS], b: &[u64; BITSET_WORDS]) -> BitwiseResult {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vpopcntdq") && is_x86_feature_detected!("avx512f") {
            return unsafe { or_avx512_popcnt(a, b) };
        }
        if is_x86_feature_detected!("avx512f") {
            return unsafe { or_avx512(a, b) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { or_avx2(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    return unsafe { or_neon(a, b) };
    or_scalar(a, b)
}

/// AND NOT: `a & !b` — difference.
#[allow(unreachable_code)]
pub fn and_not(a: &[u64; BITSET_WORDS], b: &[u64; BITSET_WORDS]) -> BitwiseResult {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vpopcntdq") && is_x86_feature_detected!("avx512f") {
            return unsafe { and_not_avx512_popcnt(a, b) };
        }
        if is_x86_feature_detected!("avx512f") {
            return unsafe { and_not_avx512(a, b) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { and_not_avx2(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    return unsafe { and_not_neon(a, b) };
    and_not_scalar(a, b)
}

/// In-place AND: `dst &= src`. Returns the new cardinality.
#[allow(unreachable_code)]
pub fn and_inplace(dst: &mut [u64; BITSET_WORDS], src: &[u64; BITSET_WORDS]) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vpopcntdq") && is_x86_feature_detected!("avx512f") {
            return unsafe { and_inplace_avx512_popcnt(dst, src) };
        }
        if is_x86_feature_detected!("avx512f") {
            return unsafe { and_inplace_avx512(dst, src) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { and_inplace_avx2(dst, src) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    return unsafe { and_inplace_neon(dst, src) };
    and_inplace_scalar(dst, src)
}

/// In-place OR: `dst |= src`. Returns the new cardinality.
#[allow(unreachable_code)]
pub fn or_inplace(dst: &mut [u64; BITSET_WORDS], src: &[u64; BITSET_WORDS]) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vpopcntdq") && is_x86_feature_detected!("avx512f") {
            return unsafe { or_inplace_avx512_popcnt(dst, src) };
        }
        if is_x86_feature_detected!("avx512f") {
            return unsafe { or_inplace_avx512(dst, src) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { or_inplace_avx2(dst, src) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    return unsafe { or_inplace_neon(dst, src) };
    or_inplace_scalar(dst, src)
}

/// In-place AND NOT: `dst &= !src`. Returns the new cardinality.
#[allow(unreachable_code)]
pub fn and_not_inplace(dst: &mut [u64; BITSET_WORDS], src: &[u64; BITSET_WORDS]) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vpopcntdq") && is_x86_feature_detected!("avx512f") {
            return unsafe { and_not_inplace_avx512_popcnt(dst, src) };
        }
        if is_x86_feature_detected!("avx512f") {
            return unsafe { and_not_inplace_avx512(dst, src) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { and_not_inplace_avx2(dst, src) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    return unsafe { and_not_inplace_neon(dst, src) };
    and_not_inplace_scalar(dst, src)
}

// ── Scalar paths ─────────────────────────────────────────────────────────────

fn and_scalar(a: &[u64; BITSET_WORDS], b: &[u64; BITSET_WORDS]) -> BitwiseResult {
    let mut bits = Box::new([0u64; BITSET_WORDS]);
    let mut cardinality = 0;
    for i in 0..BITSET_WORDS {
        bits[i] = a[i] & b[i];
        cardinality += bits[i].count_ones() as usize;
    }
    BitwiseResult { bits, cardinality }
}

fn or_scalar(a: &[u64; BITSET_WORDS], b: &[u64; BITSET_WORDS]) -> BitwiseResult {
    let mut bits = Box::new([0u64; BITSET_WORDS]);
    let mut cardinality = 0;
    for i in 0..BITSET_WORDS {
        bits[i] = a[i] | b[i];
        cardinality += bits[i].count_ones() as usize;
    }
    BitwiseResult { bits, cardinality }
}

fn and_not_scalar(a: &[u64; BITSET_WORDS], b: &[u64; BITSET_WORDS]) -> BitwiseResult {
    let mut bits = Box::new([0u64; BITSET_WORDS]);
    let mut cardinality = 0;
    for i in 0..BITSET_WORDS {
        bits[i] = a[i] & !b[i];
        cardinality += bits[i].count_ones() as usize;
    }
    BitwiseResult { bits, cardinality }
}

fn and_inplace_scalar(dst: &mut [u64; BITSET_WORDS], src: &[u64; BITSET_WORDS]) -> usize {
    let mut cardinality = 0;
    for i in 0..BITSET_WORDS {
        dst[i] &= src[i];
        cardinality += dst[i].count_ones() as usize;
    }
    cardinality
}

fn or_inplace_scalar(dst: &mut [u64; BITSET_WORDS], src: &[u64; BITSET_WORDS]) -> usize {
    let mut cardinality = 0;
    for i in 0..BITSET_WORDS {
        dst[i] |= src[i];
        cardinality += dst[i].count_ones() as usize;
    }
    cardinality
}

fn and_not_inplace_scalar(dst: &mut [u64; BITSET_WORDS], src: &[u64; BITSET_WORDS]) -> usize {
    let mut cardinality = 0;
    for i in 0..BITSET_WORDS {
        dst[i] &= !src[i];
        cardinality += dst[i].count_ones() as usize;
    }
    cardinality
}

// ── NEON paths (bitwise only, cardinality via second pass) ───────────────────
//
// Uses 128-bit Q registers: 2 × u64 per register = 512 chunks × 2 = 1024 words
// (BITSET_WORDS is a multiple of 2, so there is no tail). After the bitwise
// operation a separate popcount pass computes cardinality (which itself uses the
// NEON popcount kernel).
//
// Available operations:
//   AND      : vandq_u64
//   OR       : vorrq_u64
//   AND NOT  : vbicq_u64(a, b)  →  a & ~b   (note operand order: ~second & first)

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn and_neon(a: &[u64; BITSET_WORDS], b: &[u64; BITSET_WORDS]) -> BitwiseResult {
    use std::arch::aarch64::*;
    let mut bits = Box::new([0u64; BITSET_WORDS]);
    let chunks = BITSET_WORDS / 2;
    unsafe {
        for i in 0..chunks {
            let va = vld1q_u64(a.as_ptr().add(i * 2));
            let vb = vld1q_u64(b.as_ptr().add(i * 2));
            vst1q_u64(bits.as_mut_ptr().add(i * 2), vandq_u64(va, vb));
        }
    }
    let cardinality = super::popcount::popcount_u64_slice(bits.as_ref());
    BitwiseResult { bits, cardinality }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn or_neon(a: &[u64; BITSET_WORDS], b: &[u64; BITSET_WORDS]) -> BitwiseResult {
    use std::arch::aarch64::*;
    let mut bits = Box::new([0u64; BITSET_WORDS]);
    let chunks = BITSET_WORDS / 2;
    unsafe {
        for i in 0..chunks {
            let va = vld1q_u64(a.as_ptr().add(i * 2));
            let vb = vld1q_u64(b.as_ptr().add(i * 2));
            vst1q_u64(bits.as_mut_ptr().add(i * 2), vorrq_u64(va, vb));
        }
    }
    let cardinality = super::popcount::popcount_u64_slice(bits.as_ref());
    BitwiseResult { bits, cardinality }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn and_not_neon(a: &[u64; BITSET_WORDS], b: &[u64; BITSET_WORDS]) -> BitwiseResult {
    use std::arch::aarch64::*;
    let mut bits = Box::new([0u64; BITSET_WORDS]);
    let chunks = BITSET_WORDS / 2;
    unsafe {
        for i in 0..chunks {
            let va = vld1q_u64(a.as_ptr().add(i * 2));
            let vb = vld1q_u64(b.as_ptr().add(i * 2));
            // vbicq_u64(x, y) computes x & ~y → pass (a, b) to get a & ~b.
            vst1q_u64(bits.as_mut_ptr().add(i * 2), vbicq_u64(va, vb));
        }
    }
    let cardinality = super::popcount::popcount_u64_slice(bits.as_ref());
    BitwiseResult { bits, cardinality }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn and_inplace_neon(dst: &mut [u64; BITSET_WORDS], src: &[u64; BITSET_WORDS]) -> usize {
    use std::arch::aarch64::*;
    let chunks = BITSET_WORDS / 2;
    unsafe {
        for i in 0..chunks {
            let vd = vld1q_u64(dst.as_ptr().add(i * 2));
            let vs = vld1q_u64(src.as_ptr().add(i * 2));
            vst1q_u64(dst.as_mut_ptr().add(i * 2), vandq_u64(vd, vs));
        }
    }
    super::popcount::popcount_u64_slice(dst.as_ref())
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn or_inplace_neon(dst: &mut [u64; BITSET_WORDS], src: &[u64; BITSET_WORDS]) -> usize {
    use std::arch::aarch64::*;
    let chunks = BITSET_WORDS / 2;
    unsafe {
        for i in 0..chunks {
            let vd = vld1q_u64(dst.as_ptr().add(i * 2));
            let vs = vld1q_u64(src.as_ptr().add(i * 2));
            vst1q_u64(dst.as_mut_ptr().add(i * 2), vorrq_u64(vd, vs));
        }
    }
    super::popcount::popcount_u64_slice(dst.as_ref())
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn and_not_inplace_neon(dst: &mut [u64; BITSET_WORDS], src: &[u64; BITSET_WORDS]) -> usize {
    use std::arch::aarch64::*;
    let chunks = BITSET_WORDS / 2;
    unsafe {
        for i in 0..chunks {
            let vd = vld1q_u64(dst.as_ptr().add(i * 2));
            let vs = vld1q_u64(src.as_ptr().add(i * 2));
            vst1q_u64(dst.as_mut_ptr().add(i * 2), vbicq_u64(vd, vs)); // dst & ~src
        }
    }
    super::popcount::popcount_u64_slice(dst.as_ref())
}

// ── AVX2 paths (bitwise only, cardinality via second pass) ───────────────────
//
// Uses 256-bit YMM registers: 4 × u64 per register = 256 chunks × 4 = 1024 words.
// After the bitwise operation a separate popcount pass computes cardinality.
//
// Available operations:
//   AND      : _mm256_and_si256
//   OR       : _mm256_or_si256
//   AND NOT  : _mm256_andnot_si256(b, a)  →  ~b & a

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn and_avx2(a: &[u64; BITSET_WORDS], b: &[u64; BITSET_WORDS]) -> BitwiseResult {
    use std::arch::x86_64::*;
    let mut bits = Box::new([0u64; BITSET_WORDS]);
    let chunks = BITSET_WORDS / 4;
    for i in 0..chunks {
        unsafe {
            let va = _mm256_loadu_si256(a.as_ptr().add(i * 4) as *const __m256i);
            let vb = _mm256_loadu_si256(b.as_ptr().add(i * 4) as *const __m256i);
            let r = _mm256_and_si256(va, vb);
            _mm256_storeu_si256(bits.as_mut_ptr().add(i * 4) as *mut __m256i, r);
        }
    }
    let cardinality = super::popcount::popcount_u64_slice(bits.as_ref());
    BitwiseResult { bits, cardinality }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn or_avx2(a: &[u64; BITSET_WORDS], b: &[u64; BITSET_WORDS]) -> BitwiseResult {
    use std::arch::x86_64::*;
    let mut bits = Box::new([0u64; BITSET_WORDS]);
    let chunks = BITSET_WORDS / 4;
    for i in 0..chunks {
        unsafe {
            let va = _mm256_loadu_si256(a.as_ptr().add(i * 4) as *const __m256i);
            let vb = _mm256_loadu_si256(b.as_ptr().add(i * 4) as *const __m256i);
            let r = _mm256_or_si256(va, vb);
            _mm256_storeu_si256(bits.as_mut_ptr().add(i * 4) as *mut __m256i, r);
        }
    }
    let cardinality = super::popcount::popcount_u64_slice(bits.as_ref());
    BitwiseResult { bits, cardinality }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn and_not_avx2(a: &[u64; BITSET_WORDS], b: &[u64; BITSET_WORDS]) -> BitwiseResult {
    use std::arch::x86_64::*;
    let mut bits = Box::new([0u64; BITSET_WORDS]);
    let chunks = BITSET_WORDS / 4;
    for i in 0..chunks {
        unsafe {
            let va = _mm256_loadu_si256(a.as_ptr().add(i * 4) as *const __m256i);
            let vb = _mm256_loadu_si256(b.as_ptr().add(i * 4) as *const __m256i);
            // _mm256_andnot_si256(x, y) computes ~x & y, so pass b first to get ~b & a
            let r = _mm256_andnot_si256(vb, va);
            _mm256_storeu_si256(bits.as_mut_ptr().add(i * 4) as *mut __m256i, r);
        }
    }
    let cardinality = super::popcount::popcount_u64_slice(bits.as_ref());
    BitwiseResult { bits, cardinality }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn and_inplace_avx2(dst: &mut [u64; BITSET_WORDS], src: &[u64; BITSET_WORDS]) -> usize {
    use std::arch::x86_64::*;
    let chunks = BITSET_WORDS / 4;
    for i in 0..chunks {
        unsafe {
            let vd = _mm256_loadu_si256(dst.as_ptr().add(i * 4) as *const __m256i);
            let vs = _mm256_loadu_si256(src.as_ptr().add(i * 4) as *const __m256i);
            let r = _mm256_and_si256(vd, vs);
            _mm256_storeu_si256(dst.as_mut_ptr().add(i * 4) as *mut __m256i, r);
        }
    }
    super::popcount::popcount_u64_slice(dst.as_ref())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn or_inplace_avx2(dst: &mut [u64; BITSET_WORDS], src: &[u64; BITSET_WORDS]) -> usize {
    use std::arch::x86_64::*;
    let chunks = BITSET_WORDS / 4;
    for i in 0..chunks {
        unsafe {
            let vd = _mm256_loadu_si256(dst.as_ptr().add(i * 4) as *const __m256i);
            let vs = _mm256_loadu_si256(src.as_ptr().add(i * 4) as *const __m256i);
            let r = _mm256_or_si256(vd, vs);
            _mm256_storeu_si256(dst.as_mut_ptr().add(i * 4) as *mut __m256i, r);
        }
    }
    super::popcount::popcount_u64_slice(dst.as_ref())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn and_not_inplace_avx2(dst: &mut [u64; BITSET_WORDS], src: &[u64; BITSET_WORDS]) -> usize {
    use std::arch::x86_64::*;
    let chunks = BITSET_WORDS / 4;
    for i in 0..chunks {
        unsafe {
            let vd = _mm256_loadu_si256(dst.as_ptr().add(i * 4) as *const __m256i);
            let vs = _mm256_loadu_si256(src.as_ptr().add(i * 4) as *const __m256i);
            let r = _mm256_andnot_si256(vs, vd); // ~src & dst
            _mm256_storeu_si256(dst.as_mut_ptr().add(i * 4) as *mut __m256i, r);
        }
    }
    super::popcount::popcount_u64_slice(dst.as_ref())
}

// ── AVX-512F paths (bitwise only, cardinality via second pass) ────────────────
//
// Uses 512-bit ZMM registers: 8 × u64 per register = 128 chunks × 8 = 1024 words.
// VPTERNLOGQ (imm8 truth table) covers AND/OR/ANDNOT in a single fused instruction.
//
// VPTERNLOGQ truth table immediates:
//   AND      (A & B)   : imm8 = 0x80  (1000_0000)
//   OR       (A | B)   : imm8 = 0xFE  (1111_1110)
//   AND NOT  (A & ~B)  : imm8 = 0x44  (0100_0100)  i.e. A=src1, B=src2, C unused
//     Note: _mm512_ternarylogic_epi64(a, b, c, imm) computes on 3 operands;
//     for two-operand ops the third can be anything — we reuse `a`.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn and_avx512(a: &[u64; BITSET_WORDS], b: &[u64; BITSET_WORDS]) -> BitwiseResult {
    use std::arch::x86_64::*;
    let mut bits = Box::new([0u64; BITSET_WORDS]);
    let chunks = BITSET_WORDS / 8;
    for i in 0..chunks {
        unsafe {
            let va = _mm512_loadu_si512(a.as_ptr().add(i * 8) as *const __m512i);
            let vb = _mm512_loadu_si512(b.as_ptr().add(i * 8) as *const __m512i);
            let r = _mm512_and_epi64(va, vb);
            _mm512_storeu_si512(bits.as_mut_ptr().add(i * 8) as *mut __m512i, r);
        }
    }
    let cardinality = super::popcount::popcount_u64_slice(bits.as_ref());
    BitwiseResult { bits, cardinality }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn or_avx512(a: &[u64; BITSET_WORDS], b: &[u64; BITSET_WORDS]) -> BitwiseResult {
    use std::arch::x86_64::*;
    let mut bits = Box::new([0u64; BITSET_WORDS]);
    let chunks = BITSET_WORDS / 8;
    for i in 0..chunks {
        unsafe {
            let va = _mm512_loadu_si512(a.as_ptr().add(i * 8) as *const __m512i);
            let vb = _mm512_loadu_si512(b.as_ptr().add(i * 8) as *const __m512i);
            let r = _mm512_or_epi64(va, vb);
            _mm512_storeu_si512(bits.as_mut_ptr().add(i * 8) as *mut __m512i, r);
        }
    }
    let cardinality = super::popcount::popcount_u64_slice(bits.as_ref());
    BitwiseResult { bits, cardinality }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn and_not_avx512(a: &[u64; BITSET_WORDS], b: &[u64; BITSET_WORDS]) -> BitwiseResult {
    use std::arch::x86_64::*;
    let mut bits = Box::new([0u64; BITSET_WORDS]);
    let chunks = BITSET_WORDS / 8;
    for i in 0..chunks {
        unsafe {
            let va = _mm512_loadu_si512(a.as_ptr().add(i * 8) as *const __m512i);
            let vb = _mm512_loadu_si512(b.as_ptr().add(i * 8) as *const __m512i);
            // VPANDNQ: computes ~b & a  (note: operand order is ~src1 & src2)
            let r = _mm512_andnot_epi64(vb, va);
            _mm512_storeu_si512(bits.as_mut_ptr().add(i * 8) as *mut __m512i, r);
        }
    }
    let cardinality = super::popcount::popcount_u64_slice(bits.as_ref());
    BitwiseResult { bits, cardinality }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn and_inplace_avx512(dst: &mut [u64; BITSET_WORDS], src: &[u64; BITSET_WORDS]) -> usize {
    use std::arch::x86_64::*;
    let chunks = BITSET_WORDS / 8;
    for i in 0..chunks {
        unsafe {
            let vd = _mm512_loadu_si512(dst.as_ptr().add(i * 8) as *const __m512i);
            let vs = _mm512_loadu_si512(src.as_ptr().add(i * 8) as *const __m512i);
            let r = _mm512_and_epi64(vd, vs);
            _mm512_storeu_si512(dst.as_mut_ptr().add(i * 8) as *mut __m512i, r);
        }
    }
    super::popcount::popcount_u64_slice(dst.as_ref())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn or_inplace_avx512(dst: &mut [u64; BITSET_WORDS], src: &[u64; BITSET_WORDS]) -> usize {
    use std::arch::x86_64::*;
    let chunks = BITSET_WORDS / 8;
    for i in 0..chunks {
        unsafe {
            let vd = _mm512_loadu_si512(dst.as_ptr().add(i * 8) as *const __m512i);
            let vs = _mm512_loadu_si512(src.as_ptr().add(i * 8) as *const __m512i);
            let r = _mm512_or_epi64(vd, vs);
            _mm512_storeu_si512(dst.as_mut_ptr().add(i * 8) as *mut __m512i, r);
        }
    }
    super::popcount::popcount_u64_slice(dst.as_ref())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn and_not_inplace_avx512(dst: &mut [u64; BITSET_WORDS], src: &[u64; BITSET_WORDS]) -> usize {
    use std::arch::x86_64::*;
    let chunks = BITSET_WORDS / 8;
    for i in 0..chunks {
        unsafe {
            let vd = _mm512_loadu_si512(dst.as_ptr().add(i * 8) as *const __m512i);
            let vs = _mm512_loadu_si512(src.as_ptr().add(i * 8) as *const __m512i);
            let r = _mm512_andnot_epi64(vs, vd);
            _mm512_storeu_si512(dst.as_mut_ptr().add(i * 8) as *mut __m512i, r);
        }
    }
    super::popcount::popcount_u64_slice(dst.as_ref())
}

// ── AVX-512F + VPOPCNTDQ paths (bitwise + fused popcount in one pass) ─────────
//
// When avx512vpopcntdq is available we can compute the result AND its popcount
// in a single pass, avoiding a separate `recompute_cardinality` scan.
//
// Each chunk:
//   1. Load 8 words into va, vb.
//   2. Compute result r = op(va, vb).
//   3. popcnt = _mm512_popcnt_epi64(r)  — per-lane u64 bit counts.
//   4. acc    = _mm512_add_epi64(acc, popcnt) — accumulate lane sums.
//   5. Store r to output.
// After all chunks: total = _mm512_reduce_add_epi64(acc).

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
unsafe fn and_avx512_popcnt(a: &[u64; BITSET_WORDS], b: &[u64; BITSET_WORDS]) -> BitwiseResult {
    use std::arch::x86_64::*;
    let mut bits = Box::new([0u64; BITSET_WORDS]);
    let mut acc = _mm512_setzero_si512();
    let chunks = BITSET_WORDS / 8;
    for i in 0..chunks {
        unsafe {
            let va = _mm512_loadu_si512(a.as_ptr().add(i * 8) as *const __m512i);
            let vb = _mm512_loadu_si512(b.as_ptr().add(i * 8) as *const __m512i);
            let r = _mm512_and_epi64(va, vb);
            acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(r));
            _mm512_storeu_si512(bits.as_mut_ptr().add(i * 8) as *mut __m512i, r);
        }
    }
    let cardinality = _mm512_reduce_add_epi64(acc) as usize;
    BitwiseResult { bits, cardinality }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
unsafe fn or_avx512_popcnt(a: &[u64; BITSET_WORDS], b: &[u64; BITSET_WORDS]) -> BitwiseResult {
    use std::arch::x86_64::*;
    let mut bits = Box::new([0u64; BITSET_WORDS]);
    let mut acc = _mm512_setzero_si512();
    let chunks = BITSET_WORDS / 8;
    for i in 0..chunks {
        unsafe {
            let va = _mm512_loadu_si512(a.as_ptr().add(i * 8) as *const __m512i);
            let vb = _mm512_loadu_si512(b.as_ptr().add(i * 8) as *const __m512i);
            let r = _mm512_or_epi64(va, vb);
            acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(r));
            _mm512_storeu_si512(bits.as_mut_ptr().add(i * 8) as *mut __m512i, r);
        }
    }
    let cardinality = _mm512_reduce_add_epi64(acc) as usize;
    BitwiseResult { bits, cardinality }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
unsafe fn and_not_avx512_popcnt(a: &[u64; BITSET_WORDS], b: &[u64; BITSET_WORDS]) -> BitwiseResult {
    use std::arch::x86_64::*;
    let mut bits = Box::new([0u64; BITSET_WORDS]);
    let mut acc = _mm512_setzero_si512();
    let chunks = BITSET_WORDS / 8;
    for i in 0..chunks {
        unsafe {
            let va = _mm512_loadu_si512(a.as_ptr().add(i * 8) as *const __m512i);
            let vb = _mm512_loadu_si512(b.as_ptr().add(i * 8) as *const __m512i);
            let r = _mm512_andnot_epi64(vb, va); // ~b & a
            acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(r));
            _mm512_storeu_si512(bits.as_mut_ptr().add(i * 8) as *mut __m512i, r);
        }
    }
    let cardinality = _mm512_reduce_add_epi64(acc) as usize;
    BitwiseResult { bits, cardinality }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
unsafe fn and_inplace_avx512_popcnt(dst: &mut [u64; BITSET_WORDS], src: &[u64; BITSET_WORDS]) -> usize {
    use std::arch::x86_64::*;
    let mut acc = _mm512_setzero_si512();
    let chunks = BITSET_WORDS / 8;
    for i in 0..chunks {
        unsafe {
            let vd = _mm512_loadu_si512(dst.as_ptr().add(i * 8) as *const __m512i);
            let vs = _mm512_loadu_si512(src.as_ptr().add(i * 8) as *const __m512i);
            let r = _mm512_and_epi64(vd, vs);
            acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(r));
            _mm512_storeu_si512(dst.as_mut_ptr().add(i * 8) as *mut __m512i, r);
        }
    }
    _mm512_reduce_add_epi64(acc) as usize
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
unsafe fn or_inplace_avx512_popcnt(dst: &mut [u64; BITSET_WORDS], src: &[u64; BITSET_WORDS]) -> usize {
    use std::arch::x86_64::*;
    let mut acc = _mm512_setzero_si512();
    let chunks = BITSET_WORDS / 8;
    for i in 0..chunks {
        unsafe {
            let vd = _mm512_loadu_si512(dst.as_ptr().add(i * 8) as *const __m512i);
            let vs = _mm512_loadu_si512(src.as_ptr().add(i * 8) as *const __m512i);
            let r = _mm512_or_epi64(vd, vs);
            acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(r));
            _mm512_storeu_si512(dst.as_mut_ptr().add(i * 8) as *mut __m512i, r);
        }
    }
    _mm512_reduce_add_epi64(acc) as usize
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
unsafe fn and_not_inplace_avx512_popcnt(dst: &mut [u64; BITSET_WORDS], src: &[u64; BITSET_WORDS]) -> usize {
    use std::arch::x86_64::*;
    let mut acc = _mm512_setzero_si512();
    let chunks = BITSET_WORDS / 8;
    for i in 0..chunks {
        unsafe {
            let vd = _mm512_loadu_si512(dst.as_ptr().add(i * 8) as *const __m512i);
            let vs = _mm512_loadu_si512(src.as_ptr().add(i * 8) as *const __m512i);
            let r = _mm512_andnot_epi64(vs, vd); // ~src & dst
            acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(r));
            _mm512_storeu_si512(dst.as_mut_ptr().add(i * 8) as *mut __m512i, r);
        }
    }
    _mm512_reduce_add_epi64(acc) as usize
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::bitset::BitsetContainer;

    fn make_bits(values: &[u16]) -> Box<[u64; BITSET_WORDS]> {
        let mut bits = Box::new([0u64; BITSET_WORDS]);
        for &v in values {
            bits[(v >> 6) as usize] |= 1u64 << (v & 63);
        }
        bits
    }

    fn to_values(bits: &[u64; BITSET_WORDS]) -> Vec<u16> {
        let c = BitsetContainer::from_values(
            &(0..=65535u16)
                .filter(|&v| bits[(v >> 6) as usize] & (1u64 << (v & 63)) != 0)
                .collect::<Vec<_>>(),
        );
        c.to_values()
    }

    // ── scalar correctness ──────────────────────────────────────────

    #[test]
    fn scalar_and() {
        let a = make_bits(&[1, 3, 5, 7]);
        let b = make_bits(&[3, 5, 9]);
        let r = and_scalar(&a, &b);
        assert_eq!(to_values(&r.bits), vec![3, 5]);
        assert_eq!(r.cardinality, 2);
    }

    #[test]
    fn scalar_or() {
        let a = make_bits(&[1, 3]);
        let b = make_bits(&[3, 5]);
        let r = or_scalar(&a, &b);
        assert_eq!(to_values(&r.bits), vec![1, 3, 5]);
        assert_eq!(r.cardinality, 3);
    }

    #[test]
    fn scalar_and_not() {
        let a = make_bits(&[1, 3, 5, 7]);
        let b = make_bits(&[3, 7]);
        let r = and_not_scalar(&a, &b);
        assert_eq!(to_values(&r.bits), vec![1, 5]);
        assert_eq!(r.cardinality, 2);
    }

    #[test]
    fn scalar_inplace_and() {
        let mut a = make_bits(&[1, 3, 5, 7]);
        let b = make_bits(&[3, 5, 9]);
        let card = and_inplace_scalar(&mut a, &b);
        assert_eq!(to_values(&a), vec![3, 5]);
        assert_eq!(card, 2);
    }

    #[test]
    fn scalar_inplace_or() {
        let mut a = make_bits(&[1, 3]);
        let b = make_bits(&[3, 5]);
        let card = or_inplace_scalar(&mut a, &b);
        assert_eq!(to_values(&a), vec![1, 3, 5]);
        assert_eq!(card, 3);
    }

    #[test]
    fn scalar_inplace_and_not() {
        let mut a = make_bits(&[1, 3, 5, 7]);
        let b = make_bits(&[3, 7]);
        let card = and_not_inplace_scalar(&mut a, &b);
        assert_eq!(to_values(&a), vec![1, 5]);
        assert_eq!(card, 2);
    }

    // ── dispatch correctness (works on all CPUs) ────────────────────

    #[test]
    fn dispatch_and_matches_scalar() {
        let a = make_bits(&(0u16..500).step_by(2).collect::<Vec<_>>());
        let b = make_bits(&(0u16..500).step_by(3).collect::<Vec<_>>());
        let scalar = and_scalar(&a, &b);
        let dispatched = and(&a, &b);
        assert_eq!(dispatched.bits[..], scalar.bits[..]);
        assert_eq!(dispatched.cardinality, scalar.cardinality);
    }

    #[test]
    fn dispatch_or_matches_scalar() {
        let a = make_bits(&(0u16..500).step_by(2).collect::<Vec<_>>());
        let b = make_bits(&(0u16..500).step_by(3).collect::<Vec<_>>());
        let scalar = or_scalar(&a, &b);
        let dispatched = or(&a, &b);
        assert_eq!(dispatched.bits[..], scalar.bits[..]);
        assert_eq!(dispatched.cardinality, scalar.cardinality);
    }

    #[test]
    fn dispatch_and_not_matches_scalar() {
        let a = make_bits(&(0u16..500).step_by(2).collect::<Vec<_>>());
        let b = make_bits(&(0u16..500).step_by(3).collect::<Vec<_>>());
        let scalar = and_not_scalar(&a, &b);
        let dispatched = and_not(&a, &b);
        assert_eq!(dispatched.bits[..], scalar.bits[..]);
        assert_eq!(dispatched.cardinality, scalar.cardinality);
    }

    #[test]
    fn dispatch_inplace_and_matches_scalar() {
        let a = make_bits(&(0u16..500).step_by(2).collect::<Vec<_>>());
        let b = make_bits(&(0u16..500).step_by(3).collect::<Vec<_>>());
        let mut scalar_dst = a.clone();
        let scalar_card = and_inplace_scalar(&mut scalar_dst, &b);
        let mut dispatch_dst = a.clone();
        let dispatch_card = and_inplace(&mut dispatch_dst, &b);
        assert_eq!(dispatch_dst[..], scalar_dst[..]);
        assert_eq!(dispatch_card, scalar_card);
    }

    #[test]
    fn dispatch_inplace_or_matches_scalar() {
        let a = make_bits(&(0u16..500).step_by(2).collect::<Vec<_>>());
        let b = make_bits(&(0u16..500).step_by(3).collect::<Vec<_>>());
        let mut scalar_dst = a.clone();
        let scalar_card = or_inplace_scalar(&mut scalar_dst, &b);
        let mut dispatch_dst = a.clone();
        let dispatch_card = or_inplace(&mut dispatch_dst, &b);
        assert_eq!(dispatch_dst[..], scalar_dst[..]);
        assert_eq!(dispatch_card, scalar_card);
    }

    #[test]
    fn dispatch_inplace_and_not_matches_scalar() {
        let a = make_bits(&(0u16..500).step_by(2).collect::<Vec<_>>());
        let b = make_bits(&(0u16..500).step_by(3).collect::<Vec<_>>());
        let mut scalar_dst = a.clone();
        let scalar_card = and_not_inplace_scalar(&mut scalar_dst, &b);
        let mut dispatch_dst = a.clone();
        let dispatch_card = and_not_inplace(&mut dispatch_dst, &b);
        assert_eq!(dispatch_dst[..], scalar_dst[..]);
        assert_eq!(dispatch_card, scalar_card);
    }

    // ── NEON specific tests (aarch64; NEON is baseline so always runs) ──

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_ops_match_scalar() {
        let a = make_bits(&(0u16..1000).step_by(2).collect::<Vec<_>>());
        let b = make_bits(&(0u16..1000).step_by(3).collect::<Vec<_>>());

        let s_and = and_scalar(&a, &b);
        let n_and = unsafe { and_neon(&a, &b) };
        assert_eq!(n_and.bits[..], s_and.bits[..]);
        assert_eq!(n_and.cardinality, s_and.cardinality);

        let s_or = or_scalar(&a, &b);
        let n_or = unsafe { or_neon(&a, &b) };
        assert_eq!(n_or.bits[..], s_or.bits[..]);
        assert_eq!(n_or.cardinality, s_or.cardinality);

        let s_andnot = and_not_scalar(&a, &b);
        let n_andnot = unsafe { and_not_neon(&a, &b) };
        assert_eq!(n_andnot.bits[..], s_andnot.bits[..]);
        assert_eq!(n_andnot.cardinality, s_andnot.cardinality);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_inplace_ops_match_scalar() {
        let a = make_bits(&(0u16..1000).step_by(2).collect::<Vec<_>>());
        let b = make_bits(&(0u16..1000).step_by(3).collect::<Vec<_>>());

        let mut s = a.clone();
        let s_card = and_inplace_scalar(&mut s, &b);
        let mut n = a.clone();
        let n_card = unsafe { and_inplace_neon(&mut n, &b) };
        assert_eq!(n[..], s[..]);
        assert_eq!(n_card, s_card);

        let mut s = a.clone();
        let s_card = or_inplace_scalar(&mut s, &b);
        let mut n = a.clone();
        let n_card = unsafe { or_inplace_neon(&mut n, &b) };
        assert_eq!(n[..], s[..]);
        assert_eq!(n_card, s_card);

        let mut s = a.clone();
        let s_card = and_not_inplace_scalar(&mut s, &b);
        let mut n = a.clone();
        let n_card = unsafe { and_not_inplace_neon(&mut n, &b) };
        assert_eq!(n[..], s[..]);
        assert_eq!(n_card, s_card);
    }

    // ── AVX2 specific tests (skipped if not available) ──────────────

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_and_matches_scalar_when_available() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let a = make_bits(&(0u16..1000).step_by(2).collect::<Vec<_>>());
        let b = make_bits(&(0u16..1000).step_by(3).collect::<Vec<_>>());
        let scalar = and_scalar(&a, &b);
        let simd = unsafe { and_avx2(&a, &b) };
        assert_eq!(simd.bits[..], scalar.bits[..]);
        assert_eq!(simd.cardinality, scalar.cardinality);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_or_matches_scalar_when_available() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let a = make_bits(&(0u16..1000).step_by(2).collect::<Vec<_>>());
        let b = make_bits(&(0u16..1000).step_by(3).collect::<Vec<_>>());
        let scalar = or_scalar(&a, &b);
        let simd = unsafe { or_avx2(&a, &b) };
        assert_eq!(simd.bits[..], scalar.bits[..]);
        assert_eq!(simd.cardinality, scalar.cardinality);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_and_not_matches_scalar_when_available() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let a = make_bits(&(0u16..1000).step_by(2).collect::<Vec<_>>());
        let b = make_bits(&(0u16..1000).step_by(3).collect::<Vec<_>>());
        let scalar = and_not_scalar(&a, &b);
        let simd = unsafe { and_not_avx2(&a, &b) };
        assert_eq!(simd.bits[..], scalar.bits[..]);
        assert_eq!(simd.cardinality, scalar.cardinality);
    }

    // ── AVX-512 specific tests (skipped if not available) ───────────

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_and_matches_scalar_when_available() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let a = make_bits(&(0u16..1000).step_by(2).collect::<Vec<_>>());
        let b = make_bits(&(0u16..1000).step_by(3).collect::<Vec<_>>());
        let scalar = and_scalar(&a, &b);
        let simd = unsafe { and_avx512(&a, &b) };
        assert_eq!(simd.bits[..], scalar.bits[..]);
        assert_eq!(simd.cardinality, scalar.cardinality);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_or_matches_scalar_when_available() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let a = make_bits(&(0u16..1000).step_by(2).collect::<Vec<_>>());
        let b = make_bits(&(0u16..1000).step_by(3).collect::<Vec<_>>());
        let scalar = or_scalar(&a, &b);
        let simd = unsafe { or_avx512(&a, &b) };
        assert_eq!(simd.bits[..], scalar.bits[..]);
        assert_eq!(simd.cardinality, scalar.cardinality);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_and_not_matches_scalar_when_available() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let a = make_bits(&(0u16..1000).step_by(2).collect::<Vec<_>>());
        let b = make_bits(&(0u16..1000).step_by(3).collect::<Vec<_>>());
        let scalar = and_not_scalar(&a, &b);
        let simd = unsafe { and_not_avx512(&a, &b) };
        assert_eq!(simd.bits[..], scalar.bits[..]);
        assert_eq!(simd.cardinality, scalar.cardinality);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_fused_popcnt_matches_separate_when_available() {
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512vpopcntdq") {
            return;
        }
        let a = make_bits(&(0u16..2000).step_by(2).collect::<Vec<_>>());
        let b = make_bits(&(0u16..2000).step_by(3).collect::<Vec<_>>());
        let fused = unsafe { and_avx512_popcnt(&a, &b) };
        let separate = unsafe { and_avx512(&a, &b) };
        assert_eq!(fused.bits[..], separate.bits[..]);
        assert_eq!(fused.cardinality, separate.cardinality);
    }
}
