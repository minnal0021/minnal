//! Vector maths shared by the quantiser, k-means and partition maintenance:
//! squared distances, summing points into a mean, and the sign code of a
//! residual.

/// Squared Euclidean distance through [`simsimd`]'s runtime-selected kernel.
/// Clustering and partition maintenance compare millions of these per
/// namespace, so it is the cost that matters there.
///
/// # Panics
/// If the slices differ in length: all vectors of a namespace share its dimension.
pub(crate) fn sq_dist(a: &[f32], b: &[f32]) -> f64 {
    use simsimd::SpatialSimilarity;
    f32::sqeuclidean(a, b).expect("equal dimensions")
}

/// `sum[i] += x[i]`, widening to `f64`, for summing many points into a mean. The
/// default x86-64 target only vectorises this 2 lanes wide, so the same loop is
/// also compiled for AVX-512 and AVX2 and chosen at runtime. Each lane is added
/// on its own, so every path gives exactly the same result.
pub(crate) fn add_to_sum(sum: &mut [f64], x: &[f32]) {
    #[inline(always)]
    fn add(sum: &mut [f64], x: &[f32]) {
        for (acc, &v) in sum.iter_mut().zip(x) {
            *acc += f64::from(v);
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        #[target_feature(enable = "avx512f")]
        fn add_avx512(sum: &mut [f64], x: &[f32]) {
            add(sum, x);
        }
        #[target_feature(enable = "avx2")]
        fn add_avx2(sum: &mut [f64], x: &[f32]) {
            add(sum, x);
        }
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: the CPU has AVX-512F, checked just above.
            return unsafe { add_avx512(sum, x) };
        }
        if is_x86_feature_detected!("avx2") {
            // SAFETY: the CPU has AVX2, checked just above.
            return unsafe { add_avx2(sum, x) };
        }
    }
    add(sum, x);
}

/// One bit per dimension: 1 where the residual is non-negative, 0 where it
/// is negative. The sign code a single-bit RaBitQ code is packed from.
pub fn binary_quantize(residual_vector: &[f32]) -> Vec<u8> {
    residual_vector.iter().map(|&x| if x >= 0.0 { 1u8 } else { 0u8 }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_to_sum_adds_each_lane() {
        for dim in [1usize, 3, 7, 8, 9, 16, 17, 768] {
            let x: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1 - 3.0).collect();
            let mut sum: Vec<f64> = (0..dim).map(|i| 1e9 + i as f64).collect();
            let want: Vec<f64> = sum.iter().zip(&x).map(|(s, &v)| s + f64::from(v)).collect();
            add_to_sum(&mut sum, &x);
            assert_eq!(sum, want, "dim {dim}");
        }
    }

    #[test]
    fn test_binary_quantize() {
        let embeddings = vec![1.0, -2.0, 3.0];
        let binary_quantized_embeddings = binary_quantize(&embeddings);
        assert_eq!(binary_quantized_embeddings, vec![1, 0, 1]);
    }
}
