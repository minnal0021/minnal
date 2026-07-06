use crate::index::vector_index::VectorIndex;

/// Compute an estimated dot-product distance between a query embedding and a stored
/// [`VectorIndex`] entry.  Both multi-bit and single-bit quantised vectors implement this.
pub trait DistanceEstimator {
    fn estimate_distance(&self, query_embedding: &[f32], vector_index: &VectorIndex) -> f32;
}

/// Pre-computed query-side state shared by both multi-bit and single-bit estimators.
///
/// Holds the cluster assignment and the two scalars that are constant across all
/// candidates in a single cluster scan: the query-to-centroid dot product and the
/// scaled query sum (whose formula depends on quantisation bit-width).
#[derive(Clone, Debug, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct DotProductEstimatorState {
    pub cluster_id: u32,
    pub query_to_centroid_dot_product: f32,
    pub scaled_query_sum: f32,
}

#[derive(Clone, Debug, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct MultiBitQuanDotProductEstimator(pub DotProductEstimatorState);

#[derive(Clone, Debug, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SingleBitQuanDotProductEstimator(pub DotProductEstimatorState);

impl MultiBitQuanDotProductEstimator {
    /// Pre-compute the query-side scalar that is constant across all clusters for a given search.
    /// Pass the result into [`MultiBitQuanDotProductEstimator::with_scaled_query_sum`] once per cluster.
    pub fn scaled_query_sum(query_embedding: &[f32], number_of_bits_for_quantisation: usize) -> f32 {
        let query_sum: f32 = query_embedding.iter().sum();
        let cb: f32 = -((1i32 << (number_of_bits_for_quantisation - 1)) as f32 - 0.5);
        cb * query_sum
    }

    /// Build an estimator for one cluster, reusing the pre-computed `scaled_query_sum`.
    ///
    /// `query_embedding` and `centroid` must have the same dimension; the dot product
    /// below would otherwise fail. `search()` validates the query dimension against
    /// `cluster_index.dim()` up front, so callers on that path never hit it.
    pub fn with_scaled_query_sum(cluster_id: u32, query_embedding: &[f32], centroid: &[f32], scaled_query_sum: f32) -> Self {
        let query_to_centroid_dot_product = simsimd::SpatialSimilarity::dot(query_embedding, centroid)
            .expect("MultiBit estimator: query/centroid dimension mismatch (search() validates query dim against cluster_index.dim())")
            as f32;
        MultiBitQuanDotProductEstimator(DotProductEstimatorState {
            cluster_id,
            query_to_centroid_dot_product,
            scaled_query_sum,
        })
    }

    pub fn new(cluster_id: u32, query_embedding: &[f32], centroid: &[f32], number_of_bits_for_quantisation: usize) -> Self {
        let scaled_query_sum = Self::scaled_query_sum(query_embedding, number_of_bits_for_quantisation);
        Self::with_scaled_query_sum(cluster_id, query_embedding, centroid, scaled_query_sum)
    }
}

impl SingleBitQuanDotProductEstimator {
    /// `query_embedding` and `centroid` must share a dimension (see the MultiBit
    /// constructor); `search()` guarantees this on the search path.
    pub fn new(cluster_id: u32, query_embedding: &[f32], centroid: &[f32]) -> Self {
        let query_to_centroid_dot_product = simsimd::SpatialSimilarity::dot(query_embedding, centroid)
            .expect("SingleBit estimator: query/centroid dimension mismatch (search() validates query dim against cluster_index.dim())")
            as f32;
        let sum_q: f32 = query_embedding.iter().sum();

        SingleBitQuanDotProductEstimator(DotProductEstimatorState {
            cluster_id,
            query_to_centroid_dot_product,
            scaled_query_sum: sum_q,
        })
    }
}

impl MultiBitQuanDotProductEstimator {
    /// Estimate `⟨x, q⟩` from a candidate's raw parts rather than a materialised
    /// [`VectorIndex`].
    ///
    /// The search hot path reads these three fields straight out of the rkyv archive
    /// (packed words copied into a reused buffer, the two scalars via `to_native`),
    /// so it never allocates an owned `VectorIndex` per candidate. `estimate_distance`
    /// is the same computation over an owned entry.
    pub fn estimate_from_parts(&self, query_embedding: &[f32], packed_vector: &[u64], addition_factor: f32, scaling_factor: f32) -> f32 {
        let dot_product = crate::simd::multi_bit_dot_best(packed_vector, query_embedding, query_embedding.len());
        let query_addition_factor = dot_product + self.0.scaled_query_sum;
        let estimated_distance = -self.0.query_to_centroid_dot_product + addition_factor + (scaling_factor * query_addition_factor);
        1.0 - estimated_distance
    }
}

impl DistanceEstimator for MultiBitQuanDotProductEstimator {
    fn estimate_distance(&self, query_embedding: &[f32], vector_index: &VectorIndex) -> f32 {
        self.estimate_from_parts(
            query_embedding,
            &vector_index.packed_vector,
            vector_index.addition_factor,
            vector_index.scaling_factor,
        )
    }
}

impl SingleBitQuanDotProductEstimator {
    /// Estimate `⟨x, q⟩` from a candidate's raw parts rather than a materialised
    /// [`VectorIndex`] — the SingleBit formula needs only the packed bits and the
    /// `scaling_factor`. See [`MultiBitQuanDotProductEstimator::estimate_from_parts`]
    /// for why the hot path avoids materialising an owned entry.
    pub fn estimate_from_parts(&self, query_embedding: &[f32], packed_vector: &[u64], scaling_factor: f32) -> f32 {
        let ip = crate::simd::packed_ip_best(packed_vector, query_embedding, query_embedding.len());
        let est_residual_ip = scaling_factor * (2.0 * ip - self.0.scaled_query_sum);
        self.0.query_to_centroid_dot_product + est_residual_ip
    }
}

impl DistanceEstimator for SingleBitQuanDotProductEstimator {
    /// Estimate `⟨x, q⟩` using the 1-bit RaBitQ formula.
    ///
    /// `<x,q> = <c,q> + <r,q>` where `r = x − c`.
    /// With `scaling_factor = f_rescale = ‖r‖ / (⟨ō,o⟩ · √D)` stored at index time:
    ///   `est_ip = f_rescale · (2·ip − sum_q)`  where `ip = Σ_{b_i=1} q_i`
    ///
    /// Dispatches to the best available SIMD backend (AVX-512F → AVX2 → NEON → scalar)
    /// via `crate::simd::packed_ip_best`.
    fn estimate_distance(&self, query_embedding: &[f32], vector_index: &VectorIndex) -> f32 {
        self.estimate_from_parts(query_embedding, &vector_index.packed_vector, vector_index.scaling_factor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::vector_index::{QuantisationStyle, VectorIndex};
    use crate::quantisation::rabitq::quantisation_support::pack_bits;
    use approx::assert_relative_eq;

    /// Replicates the pre-SIMD pmone_dot path that `estimate_distance` replaced.
    /// Used as the reference in the regression test below.
    fn reference_estimate(query: &[f32], packed: &[u64], scaling_factor: f32, sum_q: f32, centroid_dot: f32) -> f32 {
        let dim = query.len();
        let mut pmone_dot = 0.0f32;
        for (word_idx, &word) in packed.iter().enumerate() {
            let base = word_idx * 64;
            if base >= dim {
                break;
            }
            for b in 0..(dim - base).min(64) {
                pmone_dot += (2 * ((word >> b) & 1) as i32 - 1) as f32 * query[base + b];
            }
        }
        let ip = (pmone_dot + sum_q) / 2.0;
        centroid_dot + scaling_factor * (2.0 * ip - sum_q)
    }

    fn make_estimator(centroid_dot: f32, sum_q: f32) -> SingleBitQuanDotProductEstimator {
        SingleBitQuanDotProductEstimator(DotProductEstimatorState {
            cluster_id: 1,
            query_to_centroid_dot_product: centroid_dot,
            scaled_query_sum: sum_q,
        })
    }

    fn make_vi(scaling_factor: f32, packed: Vec<u64>) -> VectorIndex {
        VectorIndex::new(1, QuantisationStyle::SingleBit, 0.0, scaling_factor, 0.0, packed)
    }

    #[test]
    fn matches_scalar_reference_at_768_dims() {
        let dim = 768;
        let mut state = 0xdeadbeef_u64;
        let mut next = || -> f32 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as i64 as f32) / (i64::MAX as f32)
        };

        let query: Vec<f32> = (0..dim).map(|_| next()).collect();
        let bits: Vec<u8> = (0..dim).map(|_| if next() >= 0.0 { 1u8 } else { 0u8 }).collect();
        let packed = pack_bits(&bits);
        let sum_q: f32 = query.iter().sum();
        let scaling_factor = 0.031_f32;
        let centroid_dot = 0.35_f32;

        let got = make_estimator(centroid_dot, sum_q).estimate_distance(&query, &make_vi(scaling_factor, packed.clone()));
        let expected = reference_estimate(&query, &packed, scaling_factor, sum_q, centroid_dot);

        assert_relative_eq!(got, expected, epsilon = 1e-4);
    }

    #[test]
    fn empty_packed_vector_gives_no_set_bits() {
        // packed_ip returns 0 for an empty packed vector, so ip = 0 and the residual
        // term becomes scaling_factor * (0 - sum_q) = -scaling_factor * sum_q.
        let query = vec![1.0_f32, 2.0, 3.0];
        let sum_q: f32 = query.iter().sum();
        let scaling_factor = 1.5_f32;
        let centroid_dot = 0.42_f32;

        let got = make_estimator(centroid_dot, sum_q).estimate_distance(&query, &make_vi(scaling_factor, vec![]));
        assert_relative_eq!(got, centroid_dot - scaling_factor * sum_q, epsilon = 1e-6);
    }

    #[test]
    fn all_ones_bits_est_equals_centroid_dot_plus_scaling_times_sum_q() {
        // When all bits are 1: ip = sum_q, so 2·ip − sum_q = sum_q.
        let query = vec![0.1_f32, 0.2, 0.3, 0.4];
        let sum_q: f32 = query.iter().sum();
        let scaling_factor = 2.0_f32;
        let centroid_dot = 0.5_f32;
        let packed = pack_bits(&[1u8, 1, 1, 1]);

        let got = make_estimator(centroid_dot, sum_q).estimate_distance(&query, &make_vi(scaling_factor, packed));
        assert_relative_eq!(got, centroid_dot + scaling_factor * sum_q, epsilon = 1e-5);
    }

    #[test]
    fn all_zeros_bits_est_equals_centroid_dot_minus_scaling_times_sum_q() {
        // When all bits are 0: ip = 0, so 2·ip − sum_q = −sum_q.
        let query = vec![0.1_f32, 0.2, 0.3, 0.4];
        let sum_q: f32 = query.iter().sum();
        let scaling_factor = 2.0_f32;
        let centroid_dot = 0.5_f32;
        let packed = pack_bits(&[0u8, 0, 0, 0]);

        let got = make_estimator(centroid_dot, sum_q).estimate_distance(&query, &make_vi(scaling_factor, packed));
        assert_relative_eq!(got, centroid_dot - scaling_factor * sum_q, epsilon = 1e-5);
    }
}
