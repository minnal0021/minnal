//! Seeded random orthogonal rotation for RaBitQ codes ("FhtKac").
//!
//! Four rounds of: random sign flips → fast Walsh–Hadamard transform on a
//! power-of-two block → one Kac's-walk butterfly between the two halves.
//! The Hadamard block alternates between the front and the back of the vector,
//! so a non-power-of-two dimension such as 768 is fully mixed without padding
//! (blocks [0, 512) and [256, 768)).
//!
//! Algorithm after RaBitQ-Library's `FhtKacRotator` (Apache-2.0,
//! <https://github.com/VectorDB-NTU/RaBitQ-Library>); this is an independent
//! Rust implementation.
//!
//! In minnal, codes are computed on rotated vectors and each query is rotated
//! once per search (see [`ClusterIndex`](crate::semantic_search::ClusterIndex)).
//! The output is part of the stored index format, pinned by
//! `output_is_pinned_for_a_fixed_seed`.
//!
//! Cost is O(D log D) adds per vector and the state is 4·D sign bits, so
//! persisting the seed (or the 384 bytes of signs for D = 768) is enough to
//! reproduce it. Every step is orthogonal, so inner products and L2 norms are
//! preserved exactly up to float rounding.

/// Number of flip → Hadamard → Kac rounds. Four is what RaBitQ-Library and
/// VectorChord ship.
const ROUNDS: usize = 4;

/// A seeded random rotation of `dim`-dimensional f32 vectors.
#[derive(Clone, Debug)]
pub struct FhtKacRotator {
    dim: usize,
    /// Largest power of two ≤ dim: the Hadamard block length.
    trunc: usize,
    /// 1/√trunc, which makes each Hadamard block orthonormal.
    fac: f32,
    /// Per round, per coordinate: 0 or 0x8000_0000, XOR-ed into the f32 bits.
    masks: Vec<u32>,
    seed: u64,
}

impl FhtKacRotator {
    /// Build the rotation for `dim` from a seed. `dim` must be even and ≥ 8
    /// (768 qualifies; pad odd dimensions with a zero).
    pub fn new(dim: usize, seed: u64) -> Self {
        assert!(dim >= 8 && dim.is_multiple_of(2), "FhtKacRotator needs an even dim ≥ 8, got {dim}");
        let trunc = 1usize << dim.ilog2();
        let mut state = seed;
        let mut masks = Vec::with_capacity(ROUNDS * dim);
        let mut word = 0u64;
        for i in 0..ROUNDS * dim {
            if i.is_multiple_of(64) {
                word = splitmix64(&mut state);
            }
            masks.push((((word >> (i % 64)) & 1) as u32) << 31);
        }
        Self {
            dim,
            trunc,
            fac: (trunc as f32).sqrt().recip(),
            masks,
            seed,
        }
    }

    /// The dimension this rotation acts on.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The seed that reproduces this rotation.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Rotate `x` in place: `x ← Pᵀx`.
    pub fn rotate_inplace(&self, x: &mut [f32]) {
        assert_eq!(x.len(), self.dim, "vector length must equal the rotation dim");
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx512f") {
                // SAFETY: the feature was detected at runtime.
                return unsafe { self.rotate_avx512(x) };
            }
            if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
                // SAFETY: the features were detected at runtime.
                return unsafe { self.rotate_avx2(x) };
            }
        }
        self.rotate_generic(x);
    }

    /// Undo [`rotate_inplace`](Self::rotate_inplace): `x ← Px`.
    pub fn unrotate_inplace(&self, x: &mut [f32]) {
        assert_eq!(x.len(), self.dim, "vector length must equal the rotation dim");
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx512f") {
                // SAFETY: the feature was detected at runtime.
                return unsafe { self.unrotate_avx512(x) };
            }
            if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
                // SAFETY: the features were detected at runtime.
                return unsafe { self.unrotate_avx2(x) };
            }
        }
        self.unrotate_generic(x);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    fn rotate_avx512(&self, x: &mut [f32]) {
        self.rotate_with(x, |b| x86::fwht_avx512(b))
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    fn rotate_avx2(&self, x: &mut [f32]) {
        self.rotate_with(x, |b| x86::fwht_avx2(b))
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    fn unrotate_avx512(&self, x: &mut [f32]) {
        self.unrotate_with(x, |b| x86::fwht_avx512(b))
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    fn unrotate_avx2(&self, x: &mut [f32]) {
        self.unrotate_with(x, |b| x86::fwht_avx2(b))
    }

    /// Portable rotation (the reference the SIMD paths are tested against).
    pub fn rotate_generic(&self, x: &mut [f32]) {
        self.rotate_with(x, fwht)
    }

    /// Portable inverse.
    pub fn unrotate_generic(&self, x: &mut [f32]) {
        self.unrotate_with(x, fwht)
    }

    /// Rotation body, generic over the FWHT kernel. `#[inline(always)]` so each
    /// `target_feature` wrapper above gets its own vectorised copy.
    #[inline(always)]
    fn rotate_with(&self, x: &mut [f32], fwht: impl Fn(&mut [f32])) {
        let pow2 = self.trunc == self.dim;
        for round in 0..ROUNDS {
            flip(x, &self.masks[round * self.dim..][..self.dim]);
            let block = self.block(round, x);
            fwht(block);
            scale(block, self.fac);
            if !pow2 {
                kacs_walk(x);
            }
        }
        if !pow2 {
            // Each unnormalised Kac butterfly scales by √2: (√2)⁴ = 4.
            scale(x, 0.25);
        }
    }

    /// Inverse: the same steps in reverse order (each step is its own
    /// inverse up to the scale folded into the final 0.25).
    #[inline(always)]
    fn unrotate_with(&self, x: &mut [f32], fwht: impl Fn(&mut [f32])) {
        let pow2 = self.trunc == self.dim;
        for round in (0..ROUNDS).rev() {
            if !pow2 {
                kacs_walk(x);
            }
            let block = self.block(round, x);
            fwht(block);
            scale(block, self.fac);
            flip(x, &self.masks[round * self.dim..][..self.dim]);
        }
        if !pow2 {
            scale(x, 0.25);
        }
    }

    #[inline(always)]
    fn block<'a>(&self, round: usize, x: &'a mut [f32]) -> &'a mut [f32] {
        if round.is_multiple_of(2) {
            &mut x[..self.trunc]
        } else {
            &mut x[self.dim - self.trunc..]
        }
    }
}

#[inline(always)]
fn flip(x: &mut [f32], masks: &[u32]) {
    for (v, &m) in x.iter_mut().zip(masks) {
        *v = f32::from_bits(v.to_bits() ^ m);
    }
}

#[inline(always)]
fn scale(x: &mut [f32], s: f32) {
    for v in x.iter_mut() {
        *v *= s;
    }
}

/// One Kac's-walk step: butterfly every i in the first half with i + n/2.
#[inline(always)]
fn kacs_walk(x: &mut [f32]) {
    let half = x.len() / 2;
    let (a, b) = x.split_at_mut(half);
    for (u, v) in a.iter_mut().zip(b.iter_mut()) {
        let (p, q) = (*u, *v);
        *u = p + q;
        *v = p - q;
    }
}

/// Unnormalised in-place fast Walsh–Hadamard transform; `x.len()` is a power
/// of two. Strides below 8 run inside 8-lane chunks with a fixed shuffle
/// pattern; strides ≥ 8 are contiguous butterflies that vectorise directly.
#[inline(always)]
fn fwht(x: &mut [f32]) {
    let n = x.len();
    debug_assert!(n.is_power_of_two());
    if n >= 8 {
        for c in x.chunks_exact_mut(8) {
            let c: &mut [f32; 8] = c.try_into().unwrap();
            fwht8(c);
        }
    } else {
        let mut h = 1;
        while h < n {
            butterflies(x, h);
            h *= 2;
        }
        return;
    }
    let mut h = 8;
    while h < n {
        butterflies(x, h);
        h *= 2;
    }
}

#[inline(always)]
fn butterflies(x: &mut [f32], h: usize) {
    for pair in x.chunks_exact_mut(2 * h) {
        let (a, b) = pair.split_at_mut(h);
        for (u, v) in a.iter_mut().zip(b.iter_mut()) {
            let (p, q) = (*u, *v);
            *u = p + q;
            *v = p - q;
        }
    }
}

/// The first three FWHT stages on 8 lanes, fully unrolled.
#[inline(always)]
fn fwht8(c: &mut [f32; 8]) {
    let [a0, a1, a2, a3, a4, a5, a6, a7] = *c;
    let (b0, b1, b2, b3, b4, b5, b6, b7) = (a0 + a1, a0 - a1, a2 + a3, a2 - a3, a4 + a5, a4 - a5, a6 + a7, a6 - a7);
    let (d0, d1, d2, d3, d4, d5, d6, d7) = (b0 + b2, b1 + b3, b0 - b2, b1 - b3, b4 + b6, b5 + b7, b4 - b6, b5 - b7);
    *c = [d0 + d4, d1 + d5, d2 + d6, d3 + d7, d0 - d4, d1 - d5, d2 - d6, d3 - d7];
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{butterflies, fwht};
    use core::arch::x86_64::*;

    /// FWHT with AVX-512. Strides 1, 2, 4, 8 run inside each 16-lane register
    /// (permute + add + masked sub); strides 16, 32, 64 run between the eight
    /// registers of a 128-float tile without touching memory; strides ≥ 128 are
    /// contiguous butterflies.
    #[target_feature(enable = "avx512f")]
    pub fn fwht_avx512(x: &mut [f32]) {
        let n = x.len();
        if n < 128 {
            return fwht(x);
        }
        for tile in x.chunks_exact_mut(128) {
            // SAFETY: `tile` is exactly 128 f32s = 8 registers; unaligned loads/stores.
            unsafe {
                let mut r = [_mm512_setzero_ps(); 8];
                for (k, reg) in r.iter_mut().enumerate() {
                    let mut v = _mm512_loadu_ps(tile.as_ptr().add(16 * k));
                    let p = _mm512_permute_ps::<0xB1>(v); // partner at i ^ 1
                    v = _mm512_mask_sub_ps(_mm512_add_ps(v, p), 0xAAAA, p, v);
                    let p = _mm512_permute_ps::<0x4E>(v); // i ^ 2
                    v = _mm512_mask_sub_ps(_mm512_add_ps(v, p), 0xCCCC, p, v);
                    let p = _mm512_shuffle_f32x4::<0xB1>(v, v); // i ^ 4
                    v = _mm512_mask_sub_ps(_mm512_add_ps(v, p), 0xF0F0, p, v);
                    let p = _mm512_shuffle_f32x4::<0x4E>(v, v); // i ^ 8
                    *reg = _mm512_mask_sub_ps(_mm512_add_ps(v, p), 0xFF00, p, v);
                }
                for span in [1usize, 2, 4] {
                    // strides 16, 32, 64 floats = 1, 2, 4 registers
                    for base in (0..8).step_by(2 * span) {
                        for j in base..base + span {
                            let (a, b) = (r[j], r[j + span]);
                            r[j] = _mm512_add_ps(a, b);
                            r[j + span] = _mm512_sub_ps(a, b);
                        }
                    }
                }
                for (k, reg) in r.iter().enumerate() {
                    _mm512_storeu_ps(tile.as_mut_ptr().add(16 * k), *reg);
                }
            }
        }
        let mut h = 128;
        while h < n {
            butterflies(x, h);
            h *= 2;
        }
    }

    /// FWHT with AVX2: strides 1, 2, 4 inside each 8-lane register
    /// (permute + add + sub + blend), then strides ≥ 8 as contiguous butterflies.
    #[target_feature(enable = "avx2,fma")]
    pub fn fwht_avx2(x: &mut [f32]) {
        let n = x.len();
        if n < 8 {
            return fwht(x);
        }
        for c in x.chunks_exact_mut(8) {
            // SAFETY: `c` is exactly 8 f32s; unaligned load/store.
            unsafe {
                let mut v = _mm256_loadu_ps(c.as_ptr());
                let p = _mm256_permute_ps::<0xB1>(v); // i ^ 1
                v = _mm256_blend_ps::<0xAA>(_mm256_add_ps(v, p), _mm256_sub_ps(p, v));
                let p = _mm256_permute_ps::<0x4E>(v); // i ^ 2
                v = _mm256_blend_ps::<0xCC>(_mm256_add_ps(v, p), _mm256_sub_ps(p, v));
                let p = _mm256_permute2f128_ps::<0x01>(v, v); // i ^ 4
                v = _mm256_blend_ps::<0xF0>(_mm256_add_ps(v, p), _mm256_sub_ps(p, v));
                _mm256_storeu_ps(c.as_mut_ptr(), v);
            }
        }
        let mut h = 8;
        while h < n {
            butterflies(x, h);
            h *= 2;
        }
    }
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n).map(|_| (splitmix64(&mut s) >> 40) as f32 / (1u64 << 24) as f32 - 0.5).collect()
    }

    fn dot(a: &[f32], b: &[f32]) -> f64 {
        a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
    }

    #[test]
    fn preserves_inner_products_and_inverts() {
        for dim in [8, 64, 96, 256, 384, 768, 1024, 1536] {
            let r = FhtKacRotator::new(dim, 42);
            let (a, b) = (lcg_vec(dim, 1), lcg_vec(dim, 2));
            let (mut ra, mut rb) = (a.clone(), b.clone());
            r.rotate_inplace(&mut ra);
            r.rotate_inplace(&mut rb);
            assert!((dot(&a, &b) - dot(&ra, &rb)).abs() < 1e-4, "dim {dim}: inner product changed");
            assert!((dot(&a, &a) - dot(&ra, &ra)).abs() < 1e-4, "dim {dim}: norm changed");
            r.unrotate_inplace(&mut ra);
            let err = a.iter().zip(&ra).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
            assert!(err < 1e-5, "dim {dim}: inverse error {err}");
        }
    }

    #[test]
    fn simd_paths_match_portable_path() {
        let r = FhtKacRotator::new(768, 7);
        let x = lcg_vec(768, 3);
        let (mut fast, mut slow) = (x.clone(), x.clone());
        r.rotate_inplace(&mut fast);
        r.rotate_generic(&mut slow);
        let err = fast.iter().zip(&slow).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(err < 1e-6, "dispatch and portable paths differ by {err}");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn simd_fwht_kernels_match_portable() {
        for n in [16usize, 32, 128, 256, 512, 1024, 2048] {
            let x = lcg_vec(n, n as u64);
            let mut want = x.clone();
            fwht(&mut want);
            if std::arch::is_x86_feature_detected!("avx512f") {
                let mut got = x.clone();
                unsafe { x86::fwht_avx512(&mut got) };
                assert!(want.iter().zip(&got).all(|(a, b)| (a - b).abs() < 1e-4), "avx512 fwht n={n}");
            }
            if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
                let mut got = x.clone();
                unsafe { x86::fwht_avx2(&mut got) };
                assert!(want.iter().zip(&got).all(|(a, b)| (a - b).abs() < 1e-4), "avx2 fwht n={n}");
            }
        }
    }

    #[test]
    fn matrix_is_orthogonal() {
        // Build the 768×768 matrix column by column and check PᵀP = I.
        let dim = 768;
        let r = FhtKacRotator::new(dim, 9);
        let cols: Vec<Vec<f32>> = (0..dim)
            .map(|j| {
                let mut e = vec![0.0f32; dim];
                e[j] = 1.0;
                r.rotate_inplace(&mut e);
                e
            })
            .collect();
        let mut worst = 0.0f64;
        for i in (0..dim).step_by(37) {
            for j in 0..dim {
                let want = if i == j { 1.0 } else { 0.0 };
                worst = worst.max((dot(&cols[i], &cols[j]) - want).abs());
            }
        }
        assert!(worst < 1e-5, "PᵀP deviates from I by {worst}");
    }

    /// Every implementation (portable, AVX2, AVX-512), both directions, agrees with
    /// the portable path. Calls the `#[target_feature]` methods directly, so the
    /// AVX2 path is tested even on an AVX-512 host, where dispatch never picks it.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn every_path_matches_portable_both_directions() {
        fn max_diff(a: &[f32], b: &[f32]) -> f32 {
            a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
        }
        for dim in [8usize, 96, 384, 768, 1024, 1536] {
            let r = FhtKacRotator::new(dim, 11);
            let x = lcg_vec(dim, 4);
            let (mut fwd, mut inv) = (x.clone(), x.clone());
            r.rotate_generic(&mut fwd);
            r.unrotate_generic(&mut inv);
            if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
                let (mut a, mut b) = (x.clone(), x.clone());
                unsafe { r.rotate_avx2(&mut a) };
                unsafe { r.unrotate_avx2(&mut b) };
                assert!(max_diff(&a, &fwd) < 1e-6 && max_diff(&b, &inv) < 1e-6, "avx2, dim {dim}");
            }
            if std::arch::is_x86_feature_detected!("avx512f") {
                let (mut a, mut b) = (x.clone(), x.clone());
                unsafe { r.rotate_avx512(&mut a) };
                unsafe { r.unrotate_avx512(&mut b) };
                assert!(max_diff(&a, &fwd) < 1e-6 && max_diff(&b, &inv) < 1e-6, "avx512, dim {dim}");
            }
        }
    }

    /// 1-bit codes are sign patterns, so a coordinate whose sign differs between
    /// the dispatched and portable paths would change a code. Over 2,000 unit
    /// vectors none does.
    #[test]
    fn dispatched_and_portable_paths_agree_on_every_sign() {
        let r = FhtKacRotator::new(768, 99);
        let mut flips = 0;
        for s in 0..2000 {
            let mut x = lcg_vec(768, s);
            let n = x.iter().map(|a| a * a).sum::<f32>().sqrt();
            x.iter_mut().for_each(|a| *a /= n);
            let (mut g, mut d) = (x.clone(), x);
            r.rotate_generic(&mut g);
            r.rotate_inplace(&mut d);
            flips += g.iter().zip(&d).filter(|(p, q)| p.is_sign_negative() != q.is_sign_negative()).count();
        }
        assert_eq!(flips, 0);
    }

    /// The rotation is part of the stored index format: a change to `splitmix64`,
    /// `ROUNDS`, the mask layout or the block order would change every stored
    /// code. Pin its output for one seed and input against a value recorded in
    /// `test_data/rotation_golden_768.json`.
    #[test]
    fn output_is_pinned_for_a_fixed_seed() {
        let r = FhtKacRotator::new(768, 42);
        let mut x: Vec<f32> = (0..768).map(|i| ((i * 37 % 101) as f32 - 50.0) / 83.0).collect();
        r.rotate_generic(&mut x);
        let golden: Vec<f32> = serde_json::from_str(include_str!("test_data/rotation_golden_768.json")).unwrap();
        let worst = x.iter().zip(&golden).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(
            worst < 1e-5,
            "rotation output changed (max diff {worst}): stored codes would no longer match"
        );
    }

    #[test]
    fn same_seed_same_rotation() {
        let (r1, r2) = (FhtKacRotator::new(768, 123), FhtKacRotator::new(768, 123));
        let (mut a, mut b) = (lcg_vec(768, 5), lcg_vec(768, 5));
        r1.rotate_inplace(&mut a);
        r2.rotate_inplace(&mut b);
        assert_eq!(a, b);
    }
}
