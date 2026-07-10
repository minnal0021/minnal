use crate::semantic_search::index::distance_estimator::DistanceEstimator;
use rkyv::rancor::Error as RkyvError;
use rkyv::vec::ArchivedVec;
use serde::{Deserialize, Serialize};

/// Map from cluster ID to `(doc_id_bytes, raw_rkyv_bytes)` pairs.
///
/// Returned by [`VectorKvStore::scan_sparse_clusters_batch`].
pub type ClusterBatchResult = std::collections::HashMap<u32, Vec<(Vec<u8>, Vec<u8>)>>;

// ── QuantisationStyle ─────────────────────────────────────────────────────────

/// Selects the RaBitQ quantisation strategy used when indexing a document embedding.
///
/// Stored as a field inside [`VectorIndex`] so the index is self-describing.
///
/// - [`SingleBit`][QuantisationStyle::SingleBit]: 1-bit quantisation — each dimension
///   is the sign of the residual.  Packed via `pack_bits` (64 dims per `u64`).
/// - [`MultiBit`][QuantisationStyle::MultiBit]: multi-bit quantisation — each dimension
///   encoded with `number_of_bits` bits (> 1; typical values 4–8).
///   Packed via `pack_bytes` (8 bytes per `u64`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub enum QuantisationStyle {
    /// 1-bit RaBitQ — each dimension encoded as its sign bit.
    SingleBit,
    /// Multi-bit RaBitQ — each dimension encoded with `number_of_bits` bits (> 1).
    MultiBit {
        /// Bits per dimension (> 1; typical values 4–8).
        number_of_bits: usize,
    },
}

impl QuantisationStyle {
    /// Returns the raw bit count used by the underlying quantisation algorithm.
    pub fn number_of_bits(&self) -> usize {
        match self {
            Self::SingleBit => 1,
            Self::MultiBit { number_of_bits } => *number_of_bits,
        }
    }
}

impl Default for QuantisationStyle {
    fn default() -> Self {
        Self::MultiBit { number_of_bits: 8 }
    }
}

/// Async KV store view over the two vector index namespaces.
///
/// The sparse namespace holds SingleBit entries keyed by `[cluster_id (4B) ‖ doc_id]`
/// and is scanned cluster-by-cluster in the first pass.
/// The dense namespace holds MultiBit entries keyed by `doc_id` and is queried
/// directly by document ID in the re-ranking pass.
pub trait VectorKvStore {
    /// Return all SingleBit entries for the given `cluster_id` from the sparse namespace.
    ///
    /// Each item is `(document_id_bytes, raw_rkyv_bytes)` where
    /// `raw_rkyv_bytes` is a serialized `Vec<VectorIndex>` (one entry per chunk
    /// of the document that was assigned to this cluster).  Pass it to
    /// [`score_rkyv_bytes`] to obtain the per-query-token max, or deserialize
    /// with [`VectorIndex::list_from_bytes`] to score each chunk directly.
    fn scan_sparse_cluster(&self, cluster_id: u32) -> impl std::future::Future<Output = Vec<(Vec<u8>, Vec<u8>)>> + Send;

    /// Fetch the raw rkyv bytes for a document's MultiBit entry from the dense namespace.
    ///
    /// Returns `None` when the entry does not exist (document was never indexed).
    fn get_dense_entry(&self, doc_id_bytes: &[u8]) -> impl std::future::Future<Output = Option<Vec<u8>>> + Send;

    /// Fetch dense entries for multiple documents in a single operation.
    ///
    /// Returns one `Option<Vec<u8>>` per input doc ID in the same order.
    /// Prefer this over repeated `get_dense_entry` calls when fetching many candidates;
    /// implementations backed by a real KV store resolve all keys in one blocking task
    /// instead of one task per document.
    fn get_dense_entries_batch(&self, doc_ids: &[Vec<u8>]) -> impl std::future::Future<Output = Vec<Option<Vec<u8>>>> + Send;

    /// Scan multiple IVF cluster prefixes in a single operation.
    ///
    /// Returns a map from `cluster_id` to `(doc_id_bytes, raw_rkyv_bytes)` pairs.
    /// Prefer this over repeated `scan_sparse_cluster` calls when probing many clusters;
    /// implementations backed by a real KV store do one blocking task with `num_buckets`
    /// value-log reader threads instead of one blocking task × `num_buckets` threads
    /// per cluster.
    fn scan_sparse_clusters_batch(&self, cluster_ids: &[u32]) -> impl std::future::Future<Output = ClusterBatchResult> + Send;
}

/// Deserialize a list of `VectorIndex` entries from raw rkyv bytes and return
/// `max_j ⟨q, d_j⟩` — the best estimated dot-product score across all stored
/// chunks for a **single** query embedding `query_embedding`.
///
/// This is the inner "max over chunks" term used when building a ColBERT MaxSim
/// score.  The caller is responsible for summing the results across all query
/// tokens to form the full MaxSim aggregate `S(q, d) = Σ_i max_j ⟨q_i, d_j⟩`.
///
/// For single-chunk documents the list has one element and the result equals
/// plain scoring.  For multi-chunk documents the chunk whose estimated dot
/// product with `query_embedding` is highest wins.
///
/// Returns `None` when the bytes cannot be deserialized or the list is empty.
pub fn score_rkyv_bytes<E: DistanceEstimator>(bytes: &[u8], query_embedding: &[f32], estimator: &E) -> Option<(f32, f32)> {
    let list = match VectorIndex::list_from_bytes(bytes) {
        Ok(list) => list,
        Err(e) => {
            // Corruption, not "no match": surface it instead of silently dropping
            // the entry, so a degraded index is distinguishable from an empty one.
            log::warn!("vector index entry failed to deserialize ({} bytes): {e}; skipping", bytes.len());
            return None;
        }
    };
    list.iter()
        .map(|vi| (vi.estimated_distance(query_embedding, estimator), vi.error_bound))
        .max_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
}

/// A single ranked candidate returned by an ANN search.
#[derive(Clone, Debug)]
pub struct QueryResult {
    /// Raw bytes of the document identifier, taken from the KV store key
    /// suffix after the 4-byte cluster prefix.  Decode into the concrete
    /// type with e.g. `u64::from_be_bytes(...)` or `Uuid::from_slice(...)`.
    pub document_id: Vec<u8>,
    /// Estimated dot-product similarity to the query embedding.
    pub dot_product: f32,
    /// Per-document error bound from the quantised vector index.
    pub error_bound: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct VectorIndex {
    pub cluster_id: u32,
    /// Quantisation strategy used to produce this index entry.
    pub quantisation_style: QuantisationStyle,
    pub addition_factor: f32,
    pub scaling_factor: f32,
    pub error_bound: f32,
    /// Quantised embedding packed into `u64` words for compact storage.
    /// Single-bit: 64 dimensions per word (bit-packed). Multi-bit: 8 bytes per word.
    pub packed_vector: Vec<u64>,
}

impl VectorIndex {
    /// Serialize this `VectorIndex` to bytes for storage in the vector KV store.
    ///
    /// Uses rkyv for compact, allocation-free encoding.
    pub fn to_bytes(&self) -> Vec<u8> {
        rkyv::to_bytes::<RkyvError>(self)
            .expect("VectorIndex rkyv serialization is infallible")
            .to_vec()
    }

    /// Deserialize a `VectorIndex` from bytes produced by [`to_bytes`].
    ///
    /// [`to_bytes`]: VectorIndex::to_bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, RkyvError> {
        rkyv::from_bytes::<Self, RkyvError>(bytes)
    }

    pub fn new(
        cluster_id: u32,
        quantisation_style: QuantisationStyle,
        addition_factor: f32,
        scaling_factor: f32,
        error_bound: f32,
        packed_vector: Vec<u64>,
    ) -> VectorIndex {
        VectorIndex {
            cluster_id,
            quantisation_style,
            addition_factor,
            scaling_factor,
            error_bound,
            packed_vector,
        }
    }

    pub fn estimated_distance<E: DistanceEstimator>(&self, query_embedding: &[f32], estimator: &E) -> f32 {
        estimator.estimate_distance(query_embedding, self)
    }

    /// Serialize a slice of `VectorIndex` entries to bytes for storage.
    ///
    /// Use [`list_from_bytes`] to round-trip.  The format is an rkyv-encoded
    /// `Vec<VectorIndex>`; a single-element list is the multi-bit case, a
    /// multi-element list is the single-bit (multi-chunk) case.
    ///
    /// [`list_from_bytes`]: VectorIndex::list_from_bytes
    pub fn list_to_bytes(list: &[Self]) -> Vec<u8> {
        rkyv::to_bytes::<RkyvError>(&list.to_vec())
            .expect("VectorIndex list rkyv serialization is infallible")
            .to_vec()
    }

    /// Deserialize a list of `VectorIndex` entries from bytes produced by [`list_to_bytes`].
    ///
    /// [`list_to_bytes`]: VectorIndex::list_to_bytes
    pub fn list_from_bytes(bytes: &[u8]) -> Result<Vec<Self>, RkyvError> {
        rkyv::from_bytes::<Vec<Self>, RkyvError>(bytes)
    }

    /// Validated **zero-copy** access to a stored `VectorIndex` list.
    ///
    /// Unlike [`list_from_bytes`](Self::list_from_bytes), this validates the archive
    /// (so corrupt bytes still return `Err` rather than causing UB) but does **not**
    /// deserialize into an owned `Vec<VectorIndex>` — it returns a view into `bytes`. The
    /// search scoring path uses this to avoid a per-candidate heap allocation: it reads
    /// each entry's scalars via the [`ArchivedVectorIndex`] accessors and copies the
    /// packed words into a reused buffer with
    /// [`copy_packed_into`](ArchivedVectorIndex::copy_packed_into).
    ///
    /// The archived `u64` words are stored packed (1-byte aligned), so they cannot be
    /// reinterpreted as a native `&[u64]` slice without a copy — hence the buffer.
    pub fn access_list(bytes: &[u8]) -> Result<&ArchivedVec<ArchivedVectorIndex>, RkyvError> {
        rkyv::access::<ArchivedVec<ArchivedVectorIndex>, RkyvError>(bytes)
    }
}

impl ArchivedVectorIndex {
    /// The cluster this entry was quantised against (native value).
    pub fn cluster_id(&self) -> u32 {
        self.cluster_id.to_native()
    }

    /// The quantisation style, converted to the owned enum (no allocation).
    pub fn style(&self) -> QuantisationStyle {
        match self.quantisation_style {
            ArchivedQuantisationStyle::SingleBit => QuantisationStyle::SingleBit,
            ArchivedQuantisationStyle::MultiBit { number_of_bits } => QuantisationStyle::MultiBit {
                number_of_bits: number_of_bits.to_native() as usize,
            },
        }
    }

    /// RaBitQ addition-factor scalar (native value).
    pub fn addition_factor(&self) -> f32 {
        self.addition_factor.to_native()
    }

    /// RaBitQ scaling-factor scalar (native value).
    pub fn scaling_factor(&self) -> f32 {
        self.scaling_factor.to_native()
    }

    /// Per-document error bound (native value).
    pub fn error_bound(&self) -> f32 {
        self.error_bound.to_native()
    }

    /// Copy the packed quantised words into `buf` (cleared first) as native `u64`.
    ///
    /// The archived words are 1-byte-aligned little-endian, so scoring needs them
    /// widened into an aligned native slice; reusing one `buf` across a cluster's
    /// entries keeps this to a memcpy per entry with no per-entry heap allocation.
    pub fn copy_packed_into(&self, buf: &mut Vec<u64>) {
        buf.clear();
        buf.extend(self.packed_vector.iter().map(|w| w.to_native()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic_search::index::distance_estimator::{
        DotProductEstimatorState, MultiBitQuanDotProductEstimator, SingleBitQuanDotProductEstimator,
    };
    use crate::semantic_search::quantisation::rabitq::quantisation_support::{pack_bits, pack_bytes};
    use approx::assert_relative_eq;
    use simsimd::SpatialSimilarity;

    fn sample_index() -> VectorIndex {
        VectorIndex::new(
            3,
            QuantisationStyle::MultiBit { number_of_bits: 8 },
            0.5,
            1.2,
            0.01,
            pack_bytes(&[10u8, 12u8]),
        )
    }

    /// Parse a JSON array of `f32` from a `test_data` fixture file.
    fn load_vec_f32(json: &str) -> Vec<f32> {
        serde_json::from_str(json).expect("fixture must be a JSON array of f32")
    }

    /// Parse a JSON array of `u8` from a `test_data` fixture file.
    fn load_vec_u8(json: &str) -> Vec<u8> {
        serde_json::from_str(json).expect("fixture must be a JSON array of u8")
    }

    #[test]
    fn test_vector_index_create() {
        let idx = sample_index();
        assert_eq!(idx.cluster_id, 3);
        assert_eq!(idx.addition_factor, 0.5);
        assert_eq!(idx.scaling_factor, 1.2);
        assert_eq!(idx.error_bound, 0.01);
        assert_eq!(idx.packed_vector, pack_bytes(&[10u8, 12u8]));
    }

    fn centroid_data() -> Vec<f32> {
        load_vec_f32(include_str!("../../test_data/est_dot_product_centroid.json"))
    }

    fn embedding_data() -> Vec<f32> {
        load_vec_f32(include_str!("../../test_data/real_embedding.json"))
    }

    fn query_embedding_data() -> Vec<f32> {
        load_vec_f32(include_str!("../../test_data/est_dot_product_query.json"))
    }

    #[test]
    fn test_estimated_dot_product() {
        let centroid = centroid_data();
        let embedding = embedding_data();
        let query_embedding = query_embedding_data();

        assert_estimated_dot_product_using_multibit_quantisation(&centroid, &embedding, &query_embedding);

        assert_estimated_dot_product_using_single_bit_quantisation(&centroid, &embedding, &query_embedding);
    }

    fn assert_estimated_dot_product_using_multibit_quantisation(centroid: &[f32], embedding: &[f32], query_embedding: &[f32]) {
        let number_of_bits = 8;

        assert_eq!(query_embedding.len(), embedding.len());

        let quantisation = crate::semantic_search::quantisation::rabitq::quantise(embedding, centroid, number_of_bits);
        assert_eq!(
            quantisation.quantised_embedding,
            load_vec_u8(include_str!("../../test_data/est_dot_product_multibit_codes.json"))
        );

        assert_relative_eq!(quantisation.addition_factor, 0.99986607, epsilon = 0.0001);
        assert_relative_eq!(quantisation.scaling_factor, -0.0006141426, epsilon = 0.0001);
        assert_relative_eq!(quantisation.error_bound, 0.00033426087, epsilon = 0.0001);

        let estimator = MultiBitQuanDotProductEstimator::new(1, query_embedding, centroid, number_of_bits);

        let vector_index = VectorIndex::new(
            1,
            QuantisationStyle::MultiBit { number_of_bits },
            quantisation.addition_factor,
            quantisation.scaling_factor,
            quantisation.error_bound,
            pack_bytes(&quantisation.quantised_embedding),
        );

        let estimated_dot_product = vector_index.estimated_distance(query_embedding, &estimator);
        let actual_dot_product = SpatialSimilarity::dot(embedding, query_embedding).unwrap() as f32;
        assert_relative_eq!(estimated_dot_product, 0.34942555, epsilon = 0.0001);
        assert_relative_eq!(actual_dot_product, 0.34923643, epsilon = 0.0001);
        assert_relative_eq!(actual_dot_product, estimated_dot_product, epsilon = 0.0009);
    }

    fn assert_estimated_dot_product_using_single_bit_quantisation(centroid: &[f32], embedding: &[f32], query_embedding: &[f32]) {
        let number_of_bits = 1;

        assert_eq!(query_embedding.len(), embedding.len());

        let quantisation = crate::semantic_search::quantisation::rabitq::quantise(embedding, centroid, number_of_bits);
        assert_eq!(
            quantisation.quantised_embedding,
            load_vec_u8(include_str!("../../test_data/est_dot_product_singlebit_codes.json"))
        );

        assert_relative_eq!(quantisation.addition_factor, 0.0, epsilon = 0.0001);
        assert_relative_eq!(quantisation.scaling_factor, 0.030980652, epsilon = 0.001);
        assert_relative_eq!(quantisation.error_bound, 0.038_128_49, epsilon = 0.001);

        let estimator = SingleBitQuanDotProductEstimator::new(1, query_embedding, centroid);

        let vector_index = VectorIndex::new(
            1,
            QuantisationStyle::SingleBit,
            quantisation.addition_factor,
            quantisation.scaling_factor,
            quantisation.error_bound,
            pack_bits(&quantisation.quantised_embedding),
        );

        let estimated_dot_product = vector_index.estimated_distance(query_embedding, &estimator);
        let actual_dot_product = SpatialSimilarity::dot(embedding, query_embedding).unwrap() as f32;
        assert_relative_eq!(estimated_dot_product, 0.36748818, epsilon = 0.0001);
        assert_relative_eq!(actual_dot_product, 0.34923643, epsilon = 0.0001);
        assert_relative_eq!(actual_dot_product, estimated_dot_product, epsilon = 0.05);
    }

    // ── Vec<VectorIndex> storage: list_to_bytes / list_from_bytes / score_rkyv_bytes ──

    // Builds a MultiBitQuanDotProductEstimator whose per-cluster scalars are all
    // zero.  With an empty query and empty packed_vector, estimate_distance returns
    // `1.0 - addition_factor`, giving full control over scores in the tests below.
    fn zero_estimator() -> MultiBitQuanDotProductEstimator {
        MultiBitQuanDotProductEstimator(DotProductEstimatorState {
            cluster_id: 1,
            query_to_centroid_dot_product: 0.0,
            scaled_query_sum: 0.0,
        })
    }

    // ── Zero-copy archived access matches owned deserialize ───────────────────

    /// The archived accessors (`access_list` + `ArchivedVectorIndex::*`) must read back
    /// exactly what `list_from_bytes` deserializes — same scalars, same packed words,
    /// same style. This underpins the search hot path scoring over archived bytes.
    #[test]
    fn access_list_matches_list_from_bytes() {
        let list = vec![
            VectorIndex::new(5, QuantisationStyle::SingleBit, 0.0, 0.031, 0.038, pack_bits(&[1u8, 0, 1, 1, 0, 0, 1, 0])),
            VectorIndex::new(
                9,
                QuantisationStyle::MultiBit { number_of_bits: 8 },
                0.99,
                -0.0006,
                0.0003,
                pack_bytes(&[10u8, 200, 30, 40]),
            ),
        ];
        let bytes = VectorIndex::list_to_bytes(&list);

        let owned = VectorIndex::list_from_bytes(&bytes).unwrap();
        let archived = VectorIndex::access_list(&bytes).unwrap();
        assert_eq!(archived.len(), owned.len());

        let mut buf = Vec::new();
        for (o, a) in owned.iter().zip(archived.iter()) {
            assert_eq!(a.cluster_id(), o.cluster_id);
            assert_eq!(a.style(), o.quantisation_style);
            assert_eq!(a.addition_factor(), o.addition_factor);
            assert_eq!(a.scaling_factor(), o.scaling_factor);
            assert_eq!(a.error_bound(), o.error_bound);
            a.copy_packed_into(&mut buf);
            assert_eq!(buf, o.packed_vector, "packed words must widen back to the native slice");
        }
    }

    /// Scoring a real (non-empty packed) entry through the archived path
    /// (`copy_packed_into` + `estimate_from_parts`) must be bit-identical to scoring the
    /// owned entry through `estimate_distance`, for both quantisation styles.
    #[test]
    fn archived_scoring_matches_owned_scoring() {
        use crate::semantic_search::index::distance_estimator::{MultiBitQuanDotProductEstimator, SingleBitQuanDotProductEstimator};

        let mut state = 0x51ce_9a7e_1234_5678_u64;
        let mut next = || -> f32 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as i64 as f32) / (i64::MAX as f32)
        };
        let dim = 768;
        let query: Vec<f32> = (0..dim).map(|_| next()).collect();
        let centroid: Vec<f32> = (0..dim).map(|_| next()).collect();

        // SingleBit
        let sb_bits: Vec<u8> = (0..dim).map(|_| if next() >= 0.0 { 1 } else { 0 }).collect();
        let sb = VectorIndex::new(1, QuantisationStyle::SingleBit, 0.0, 0.031, 0.02, pack_bits(&sb_bits));
        let sb_bytes = VectorIndex::list_to_bytes(std::slice::from_ref(&sb));
        let sb_est = SingleBitQuanDotProductEstimator::new(1, &query, &centroid);
        let mut buf = Vec::new();
        VectorIndex::access_list(&sb_bytes).unwrap()[0].copy_packed_into(&mut buf);
        let archived_sb = sb_est.estimate_from_parts(&query, &buf, sb.scaling_factor);
        let owned_sb = sb.estimated_distance(&query, &sb_est);
        assert_eq!(archived_sb, owned_sb, "SingleBit archived vs owned scoring must match exactly");

        // MultiBit
        let mb_codes: Vec<u8> = (0..dim).map(|_| (next().abs() * 255.0) as u8).collect();
        let mb = VectorIndex::new(
            2,
            QuantisationStyle::MultiBit { number_of_bits: 8 },
            0.99,
            -0.0006,
            0.0003,
            pack_bytes(&mb_codes),
        );
        let mb_bytes = VectorIndex::list_to_bytes(std::slice::from_ref(&mb));
        let mb_est = MultiBitQuanDotProductEstimator::new(2, &query, &centroid, 8);
        VectorIndex::access_list(&mb_bytes).unwrap()[0].copy_packed_into(&mut buf);
        let archived_mb = mb_est.estimate_from_parts(&query, &buf, mb.addition_factor, mb.scaling_factor);
        let owned_mb = mb.estimated_distance(&query, &mb_est);
        assert_eq!(archived_mb, owned_mb, "MultiBit archived vs owned scoring must match exactly");
    }

    #[test]
    fn access_list_rejects_corrupt_bytes() {
        assert!(VectorIndex::access_list(b"not valid rkyv bytes at all").is_err());
    }

    #[test]
    fn test_list_round_trip() {
        let a = VectorIndex::new(
            1,
            QuantisationStyle::MultiBit { number_of_bits: 8 },
            0.1,
            0.2,
            0.01,
            pack_bytes(&[10u8, 20u8]),
        );
        let b = VectorIndex::new(
            2,
            QuantisationStyle::MultiBit { number_of_bits: 8 },
            0.3,
            0.4,
            0.02,
            pack_bytes(&[30u8, 40u8]),
        );
        let list = vec![a, b];

        let bytes = VectorIndex::list_to_bytes(&list);
        let recovered = VectorIndex::list_from_bytes(&bytes).unwrap();

        assert_eq!(recovered.len(), 2);
        assert_eq!(recovered[0].cluster_id, 1);
        assert_relative_eq!(recovered[0].addition_factor, 0.1);
        assert_eq!(recovered[0].packed_vector, pack_bytes(&[10u8, 20u8]));
        assert_eq!(recovered[1].cluster_id, 2);
        assert_relative_eq!(recovered[1].addition_factor, 0.3);
        assert_eq!(recovered[1].packed_vector, pack_bytes(&[30u8, 40u8]));
    }

    #[test]
    fn test_score_rkyv_bytes_single_entry() {
        // estimated_distance = 1.0 - addition_factor when all other terms are zero.
        let vi = VectorIndex::new(1, QuantisationStyle::MultiBit { number_of_bits: 8 }, 0.3, 0.0, 0.05, vec![]);
        let bytes = VectorIndex::list_to_bytes(&[vi]);

        let (score, error_bound) = score_rkyv_bytes(&bytes, &[], &zero_estimator()).unwrap();
        assert_relative_eq!(score, 0.7, epsilon = 1e-6);
        assert_relative_eq!(error_bound, 0.05, epsilon = 1e-6);
    }

    #[test]
    fn test_score_rkyv_bytes_maxsim_picks_highest_chunk_score() {
        // Three chunks; the one with addition_factor=0.1 scores highest (1.0 - 0.1 = 0.9).
        let e1 = VectorIndex::new(1, QuantisationStyle::MultiBit { number_of_bits: 8 }, 0.3, 0.0, 0.01, vec![]); // score 0.7
        let e2 = VectorIndex::new(1, QuantisationStyle::MultiBit { number_of_bits: 8 }, 0.1, 0.0, 0.05, vec![]); // score 0.9 ← winner
        let e3 = VectorIndex::new(1, QuantisationStyle::MultiBit { number_of_bits: 8 }, 0.5, 0.0, 0.02, vec![]); // score 0.5
        let bytes = VectorIndex::list_to_bytes(&[e1, e2, e3]);

        let (score, error_bound) = score_rkyv_bytes(&bytes, &[], &zero_estimator()).unwrap();
        assert_relative_eq!(score, 0.9, epsilon = 1e-6);
        assert_relative_eq!(error_bound, 0.05, epsilon = 1e-6);
    }

    #[test]
    fn test_score_rkyv_bytes_empty_list_returns_none() {
        let bytes = VectorIndex::list_to_bytes(&[]);
        assert!(score_rkyv_bytes(&bytes, &[], &zero_estimator()).is_none());
    }

    #[test]
    fn test_score_rkyv_bytes_corrupt_bytes_returns_none() {
        assert!(score_rkyv_bytes(b"not valid rkyv bytes", &[], &zero_estimator()).is_none());
    }
}
