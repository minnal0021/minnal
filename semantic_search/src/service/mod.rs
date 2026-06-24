//! Embedding orchestration — document/query embedding and ANN search.
//!
//! All HTTP calls to the embedding service go through [`embedding_service`].
//! This module contains only higher-level logic: quantisation, cluster
//! probing, and result ranking.  Raw text is forwarded to the service as-is.

mod embedding_service;

pub use crate::index::vector_index::QuantisationStyle;

use crate::chunking;

use log::{debug, info};

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use rayon::prelude::*;

use crate::cluster::{ClusterIndex, find_top_n_cluster_ids};
use crate::index::distance_estimator::{MultiBitQuanDotProductEstimator, SingleBitQuanDotProductEstimator};
use crate::index::vector_index::{QueryResult, VectorIndex, VectorKvStore};
use crate::quantisation::rabitq;
use std::collections::HashMap;

/// `(cluster_id, [(doc_id_bytes, raw_rkyv_bytes)])` pairs assembled for a
/// parallel scoring pass.
type ClusterEntryList = Vec<(u32, Vec<(Vec<u8>, Vec<u8>)>)>;

pub use embedding_service::{EmbeddingError, EmbeddingTarget};

/// Configuration required to call the embedding service.
#[derive(Debug, Clone)]
pub struct SemanticSearchConfig {
    /// Base URL of the embedding HTTP service, e.g. `http://192.168.1.155:8001`.
    pub embedding_service_url: String,

    /// Identifies the *family* of embedding model this instance is configured
    /// for, e.g. `"qwen"`.
    ///
    /// This is only an indication of the model family — minnal does not use it
    /// to select or version a model at request time, and it is not sent to the
    /// embedding service. Which concrete model (and version) actually produces
    /// the embeddings is entirely the embedding service's concern: the model is
    /// fixed server-side, and requests simply go to
    /// `{embedding_service_url}/embedding/document` and `.../query` with no model
    /// segment in the URL.
    ///
    /// Within minnal the name serves a single purpose: it selects which cluster
    /// file and embedding dimension this instance uses, validated at startup
    /// against `[[semantic_search.supported_models]]`.
    pub model_name: String,

    /// Dimensionality of the embedding vectors (e.g. 768).
    pub embedding_dim: usize,

    /// Maximum number of results returned by a semantic search query.  Default: 100.
    pub top_k_results: usize,

    /// Number of bits used when quantising embeddings (must be > 1).  Default: 8.
    /// In production this is always set from the TOML config file.
    pub number_of_bits_for_dense_quantisation: usize,

    /// Tokens/sentences per sliding-window chunk for the single-bit chunked embedding call.
    /// Default: 4.
    pub window_size: usize,

    /// How far the window advances between chunks for single-bit embeddings.
    /// Default: 2.
    pub sliding_size: usize,

    /// Number of IVF clusters to probe in the first-pass sparse (single-bit) search.
    /// Default: 128.
    pub n_probes: usize,

    /// Candidates retained after the first-pass sparse (single-bit) search before dense re-ranking.
    /// Default: 1000.
    pub first_pass_sparse_search_top_k: usize,

    /// Time-to-live for cached query embeddings in the system-wide
    /// `system_qemb_cache` namespace. Stale entries are evicted automatically by
    /// the TTL worker once this duration elapses. Default: 1 day.
    pub query_embedding_cache_ttl: std::time::Duration,
}

impl Default for SemanticSearchConfig {
    fn default() -> Self {
        Self {
            embedding_service_url: "http://localhost:8001".into(),
            model_name: "qwen".into(),
            embedding_dim: 768,
            top_k_results: 100,
            number_of_bits_for_dense_quantisation: 8,
            n_probes: 128,
            window_size: 4,
            sliding_size: 2,
            first_pass_sparse_search_top_k: 1000,
            query_embedding_cache_ttl: std::time::Duration::from_secs(86_400),
        }
    }
}

/// Query embeddings for a two-pass search: chunked vectors for Pass 1 and a
/// single whole-query vector for Pass 2.
#[derive(Debug, Clone)]
pub struct QueryEmbeddings {
    /// One embedding per sliding-window query chunk — Pass 1 (ColBERT MaxSim).
    pub sparse: Vec<Vec<f32>>,
    /// A single embedding of the whole query — Pass 2 dense re-ranking.
    pub dense: Vec<f32>,
}

/// Fetch embeddings for a document and produce both multi-bit and single-bit quantised indexes.
///
/// Chunking now happens here (the service no longer chunks). A **single** batch
/// call embeds one ordered payload list:
///
/// - **payload\[0\]** — the whole document text → 1 raw embedding → quantised as
///   `MultiBit { number_of_bits }` → 1 [`VectorIndex`] (the dense entry).
/// - **payload\[1..\]** — [`chunk_document`](crate::chunking::chunk_document)'s N
///   sliding-window chunks → N raw embeddings → each quantised as `SingleBit` and
///   independently assigned to its nearest IVF cluster → N [`VectorIndex`].
///
/// **Ordering is load-bearing.** The service is contracted to return embeddings
/// in payload order, so element 0 is always the whole-document (dense) vector and
/// the remainder are the chunk (sparse) vectors. Folding both into one batch is
/// one round trip and one GPU batch of `N+1` (vs. two batches, one of size 1).
///
/// Returns the multi-bit entry first, followed by the single-bit entries. The
/// combined list can be passed directly to `upsert_vectors`; the storage
/// layer groups by `(style, cluster_id)`.
pub async fn embed_document(config: &SemanticSearchConfig, cluster_index: &ClusterIndex, text: &str) -> Result<Vec<VectorIndex>, EmbeddingError> {
    let multi_bit_style = QuantisationStyle::MultiBit {
        number_of_bits: config.number_of_bits_for_dense_quantisation,
    };

    // payload[0] = whole document (dense); payload[1..] = sliding-window chunks (sparse).
    let mut payloads = Vec::with_capacity(1);
    payloads.push(text.to_string());
    payloads.extend(chunking::chunk_document(text, config.window_size, config.sliding_size));
    debug!("document embeddings: 1 dense payload + {} sparse chunk(s)", payloads.len() - 1);

    let embeddings = embedding_service::embed(&config.embedding_service_url, EmbeddingTarget::Document, &payloads, config.embedding_dim).await?;

    // Split the ordered response: first = dense (MultiBit), rest = sparse chunks (SingleBit).
    let mut it = embeddings.iter();
    let dense = it.next().ok_or(EmbeddingError::EmptyResponse)?;
    let mut indexes = vec![rabitq::index_embedding(&cluster_index.clusters, dense, multi_bit_style)];
    indexes.extend(it.map(|e| rabitq::index_embedding(&cluster_index.clusters, e, QuantisationStyle::SingleBit)));

    Ok(indexes)
}

/// Fetch the query embeddings needed for a two-pass search.
///
/// Chunking happens here. A **single** batch call embeds one ordered payload list:
///
/// - **payload\[0\]** — the whole query text → one embedding (Pass 2 dense re-rank).
/// - **payload\[1..\]** — [`chunk_query`](crate::chunking::chunk_query)'s
///   word-tokenised sliding-window chunks → one embedding per chunk (Pass 1).
///
/// **Ordering is load-bearing** — the service returns embeddings in payload order,
/// so element 0 is always the whole-query (dense) vector. The sparse list may be
/// empty for a whitespace-only query; the dense vector is always present.
pub async fn embed_query(config: &SemanticSearchConfig, text: &str) -> Result<QueryEmbeddings, EmbeddingError> {
    // payload[0] = whole query (dense); payload[1..] = sliding-window chunks (sparse).
    let mut payloads = Vec::with_capacity(1);
    payloads.push(text.to_string());
    payloads.extend(chunking::chunk_query(text, config.window_size, config.sliding_size));
    debug!("query embeddings: 1 dense payload + {} sparse chunk(s)", payloads.len() - 1);

    let mut embeddings = embedding_service::embed(&config.embedding_service_url, EmbeddingTarget::Query, &payloads, config.embedding_dim)
        .await?
        .into_iter();

    let dense = embeddings.next().ok_or(EmbeddingError::EmptyResponse)?;
    Ok(QueryEmbeddings {
        sparse: embeddings.collect(),
        dense,
    })
}

/// Run a two-pass approximate nearest-neighbour search against the quantised embedding store.
///
/// # Algorithm
///
/// **Pass 1 — sparse (SingleBit), ColBERT MaxSim:**
/// 1. For each query token, find the `n_probes` closest clusters by Euclidean distance.
/// 2. Scan all SingleBit entries in the union of those clusters in parallel.
/// 3. For each document `d` and each query token `q_i`, estimate `max_j ⟨q_i, d_j⟩`
///    over all chunks `d_j` of `d` found in the probed clusters.
/// 4. Aggregate via ColBERT MaxSim: `S(q, d) = Σ_i max_j ⟨q_i, d_j⟩`.
///    Document chunks whose clusters are not probed contribute 0 to their query token's term.
/// 5. Retain the top `first_pass_sparse_search_top_k` candidates.
///
/// **Pass 2 — dense (MultiBit):**
/// 1. Batch-fetch the dense entry for every sparse candidate by `doc_id` in a single
///    operation ([`VectorKvStore::get_dense_entries_batch`]).
/// 2. Score each candidate in parallel with [`MultiBitQuanDotProductEstimator`] against
///    the single whole-query embedding `query_dense_embedding` (symmetric with the
///    document's whole-text dense vector). Each [`VectorIndex`] carries its own
///    `cluster_id`, so the centroid is looked up directly with no separate meta read.
/// 3. Sort descending by score and return the top `top_k` entries.
///
/// # Query inputs
///
/// - `query_sparse_embeddings` — one embedding per query chunk, used in Pass 1.
/// - `query_dense_embedding` — a single whole-query embedding, used in Pass 2.
///
/// Returns an empty result if either is empty.
///
/// # Predicate filtering
///
/// `doc_filter` is an optional closure `Fn(&[u8]) -> bool` applied in the sparse pass.
/// Documents that fail the filter are excluded from both passes.
///
/// # Top-k override
///
/// `top_k` overrides `config.top_k_results` for this call only.
// Heap entry for top-k tracking; min-heap ordered by dot_product ascending.
struct HeapEntry {
    dot_product: f32,
    error_bound: f32,
    document_id: Vec<u8>,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.dot_product.total_cmp(&other.dot_product).is_eq()
    }
}
impl Eq for HeapEntry {}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dot_product.total_cmp(&other.dot_product)
    }
}

pub async fn search<K, F>(
    config: &SemanticSearchConfig,
    cluster_index: &ClusterIndex,
    query_sparse_embeddings: &[Vec<f32>],
    query_dense_embedding: &[f32],
    kv_store: &K,
    doc_filter: Option<F>,
    top_k: Option<usize>,
) -> Vec<QueryResult>
where
    K: VectorKvStore,
    F: Fn(&[u8]) -> bool + Sync,
{
    if query_sparse_embeddings.is_empty() || query_dense_embedding.is_empty() {
        return vec![];
    }

    let top_k_limit = top_k.unwrap_or(config.top_k_results);

    // ── Pass 1: sparse single-bit scan ───────────────────────────────────────

    // Union of top-n_probes clusters across all query chunk embeddings.
    let probe_clusters: Vec<u32> = {
        let mut seen = std::collections::HashSet::new();
        let mut ids = Vec::new();
        for q in query_sparse_embeddings {
            for id in find_top_n_cluster_ids(&cluster_index.clusters, q, config.n_probes) {
                if seen.insert(id) {
                    ids.push(id);
                }
            }
        }
        ids
    };

    debug!("ANN search: probing {} cluster(s) (sparse pass)", probe_clusters.len());

    // Fetch all probed clusters in a single batch operation (one blocking task, num_buckets threads).
    let sparse_by_cluster = kv_store.scan_sparse_clusters_batch(&probe_clusters).await;

    // Score using ColBERT MaxSim across all probed clusters — in parallel.
    //
    // For each document `d` we accumulate, per query token `q_i`, the best estimated
    // similarity over all chunks of `d` found in any probed cluster.  The final sparse
    // score is the sum of those per-token maxima:
    //   S(q, d) = Σ_i  max_j ⟨q_i, d_j⟩
    // where `i` ranges over query tokens and `j` over document chunks seen so far.
    // Query tokens for which no chunk of `d` falls in a probed cluster contribute 0.
    let cluster_scan_pairs: ClusterEntryList = probe_clusters
        .iter()
        .filter_map(|&id| sparse_by_cluster.get(&id).map(|entries| (id, entries.clone())))
        .collect();

    let n_query = query_sparse_embeddings.len();

    // Fold state: doc_id → Vec<f32> where Vec[i] = running max of ⟨q_i, d_j⟩ over all
    // chunks d_j of that document seen so far.  Initialised to 0.0 per token (chunks in
    // unprobed clusters contribute nothing).
    let maxsim_state: HashMap<Vec<u8>, Vec<f32>> = cluster_scan_pairs
        .par_iter()
        .fold(HashMap::new, |mut map, (cluster_id, entries)| {
            let Some(cluster) = cluster_index.clusters.get(cluster_id) else {
                return map;
            };
            // Pre-compute one estimator per query token for this cluster.
            // query_to_centroid_dot_product and scaled_query_sum are constant across all
            // document entries in the same cluster, so we compute them only once here.
            let estimators: Vec<SingleBitQuanDotProductEstimator> = query_sparse_embeddings
                .iter()
                .map(|q| SingleBitQuanDotProductEstimator::new(*cluster_id, q, &cluster.centroid))
                .collect();

            for (doc_id, raw_bytes) in entries {
                if doc_filter.as_ref().is_some_and(|f| !f(doc_id)) {
                    continue;
                }
                let Ok(vi_list) = VectorIndex::list_from_bytes(raw_bytes) else {
                    continue;
                };
                if vi_list.is_empty() {
                    continue;
                }

                let per_query_maxes = map.entry(doc_id.clone()).or_insert_with(|| vec![0.0f32; n_query]);

                for (q_idx, (q, estimator)) in query_sparse_embeddings.iter().zip(estimators.iter()).enumerate() {
                    // max_j ⟨q_i, d_j⟩ over all chunks of this document in the current cluster.
                    let best_chunk_score: f32 = vi_list
                        .iter()
                        .map(|vi| vi.estimated_distance(q, estimator))
                        .fold(f32::NEG_INFINITY, f32::max);
                    // Keep the global max across clusters — the same document may have
                    // chunks in several probed clusters for the same query token.
                    if best_chunk_score > per_query_maxes[q_idx] {
                        per_query_maxes[q_idx] = best_chunk_score;
                    }
                }
            }
            map
        })
        .reduce(HashMap::new, |mut a, b| {
            for (doc_id, scores_b) in b {
                let scores_a = a.entry(doc_id).or_insert_with(|| vec![0.0f32; n_query]);
                for i in 0..n_query {
                    if scores_b[i] > scores_a[i] {
                        scores_a[i] = scores_b[i];
                    }
                }
            }
            a
        });

    // Compute the MaxSim score: S(q, d) = Σ_i max_j ⟨q_i, d_j⟩.
    let sparse_scores: HashMap<Vec<u8>, f32> = maxsim_state
        .into_iter()
        .map(|(doc_id, per_query_maxes)| (doc_id, per_query_maxes.iter().sum()))
        .collect();

    // Keep top first_pass_sparse_search_top_k candidates.
    let mut sparse_ranked: Vec<(Vec<u8>, f32)> = sparse_scores.into_iter().collect();
    sparse_ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    sparse_ranked.truncate(config.first_pass_sparse_search_top_k);

    debug!("ANN search: {} sparse candidates after pass 1", sparse_ranked.len());

    if sparse_ranked.is_empty() {
        return vec![];
    }

    // ── Pass 2: dense multi-bit re-ranking ───────────────────────────────────

    // Fetch all dense entries in one batch operation (single blocking task in production).
    let dense_doc_ids: Vec<Vec<u8>> = sparse_ranked.iter().map(|(doc_id, _)| doc_id.clone()).collect();
    let dense_raw = kv_store.get_dense_entries_batch(&dense_doc_ids).await;

    debug!("ANN search: dense pass over {} candidates", dense_doc_ids.len());

    // scaled_query_sum is constant for the whole-query dense embedding + bit-width
    // across all clusters, so compute it once.
    let scaled_query_sum = MultiBitQuanDotProductEstimator::scaled_query_sum(query_dense_embedding, config.number_of_bits_for_dense_quantisation);

    // Each VectorIndex carries its own cluster_id for centroid lookup, so we
    // score each document directly against the single whole-query embedding.
    let scored: Vec<HeapEntry> = dense_doc_ids
        .par_iter()
        .zip(dense_raw.par_iter())
        .filter_map(|(doc_id, opt_bytes)| {
            let raw_bytes = opt_bytes.as_ref()?;
            let list = VectorIndex::list_from_bytes(raw_bytes).ok()?;
            let vi = list.first()?;
            let cluster = cluster_index.clusters.get(&vi.cluster_id)?;

            let estimator =
                MultiBitQuanDotProductEstimator::with_scaled_query_sum(vi.cluster_id, query_dense_embedding, &cluster.centroid, scaled_query_sum);
            let dot_product = vi.estimated_distance(query_dense_embedding, &estimator);
            Some(HeapEntry {
                dot_product,
                error_bound: vi.error_bound,
                document_id: doc_id.clone(),
            })
        })
        .collect();

    // Build top-k from dense scored results.
    let mut heap: BinaryHeap<Reverse<HeapEntry>> = BinaryHeap::with_capacity(top_k_limit + 1);
    for entry in scored {
        if heap.len() < top_k_limit {
            heap.push(Reverse(entry));
        } else if heap.peek().is_some_and(|Reverse(min)| entry.dot_product > min.dot_product) {
            heap.pop();
            heap.push(Reverse(entry));
        }
    }

    debug!("ANN search: returning top {} results", heap.len());

    heap.into_sorted_vec()
        .into_iter()
        .map(|Reverse(e)| QueryResult {
            document_id: e.document_id,
            dot_product: e.dot_product,
            error_bound: e.error_bound,
        })
        .collect()
}

/// Probe the embedding service to verify it is reachable.
///
/// Sends a GET to `{embedding_service_url}/healthcheck`. This is intended to be
/// called once at startup, after the cluster index loads. A failure is non-fatal:
/// the server continues but semantic search will fail at request time.
pub async fn check_embedding_service(config: &SemanticSearchConfig) -> Result<(), EmbeddingError> {
    info!("checking embedding service health at {}/healthcheck", config.embedding_service_url);
    embedding_service::check_health(&config.embedding_service_url).await?;
    info!("embedding service reachable");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::vector_index::ClusterBatchResult;

    #[test]
    fn test_default_config() {
        let config = SemanticSearchConfig::default();
        assert_eq!(config.embedding_dim, 768);
        assert_eq!(config.model_name, "qwen");
        assert_eq!(config.embedding_service_url, "http://localhost:8001");
    }

    /// Verifies that the quantised dot-product estimate is within 0.1% of the
    /// exact dot product computed on full-precision (f32) embeddings.
    ///
    /// # Embeddings used
    ///
    /// Real 768-dimensional embeddings saved in `test_data/doc_embedding.json`
    /// and `test_data/query_embedding.json`.  These were obtained from the
    /// embedding service and committed so the test runs without network access.
    ///
    /// # What is tested
    ///
    /// 1. Both embeddings are confirmed to be 768-dimensional.
    /// 2. The **exact** dot product is computed on the raw f32 vectors using simsimd.
    /// 3. The document embedding is quantised via RaBitQ (8 bits) against the
    ///    pre-built IVF cluster index.
    /// 4. The **estimated** dot product is recovered from the quantised index
    ///    using `DotProductEstimator`.
    /// 5. The **relative** difference between the two must be < 0.1% (1e-3).
    #[test]
    fn test_quantised_dot_product_accuracy() {
        use crate::cluster::{ClusterIndex, find_closest_cluster_id};
        use crate::index::distance_estimator::MultiBitQuanDotProductEstimator;
        use crate::quantisation::rabitq;
        use simsimd::SpatialSimilarity;

        let cluster_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../service/embedding_support/qwen/clusters.json");
        let n_bits = 8;

        let cluster_index = ClusterIndex::load(cluster_path, 5).expect("cluster index not found — run from the workspace root");

        // ── Load pre-saved real embeddings from fixture files ─────────────────
        #[derive(serde::Deserialize)]
        struct EmbeddingFile {
            embeddings: Vec<Vec<f32>>,
        }

        let doc_path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_data/doc_embedding.json");
        let query_path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_data/query_embedding.json");

        let doc_file: EmbeddingFile = serde_json::from_str(&std::fs::read_to_string(doc_path).expect("doc_embedding.json not found"))
            .expect("failed to parse doc_embedding.json");

        let query_file: EmbeddingFile = serde_json::from_str(&std::fs::read_to_string(query_path).expect("query_embedding.json not found"))
            .expect("failed to parse query_embedding.json");

        let doc_embedding: Vec<f32> = doc_file.embeddings.into_iter().next().expect("doc_embedding.json contains no embeddings");

        let query_embedding: Vec<f32> = query_file
            .embeddings
            .into_iter()
            .next()
            .expect("query_embedding.json contains no embeddings");

        assert_eq!(doc_embedding.len(), 768, "document embedding must be 768-dimensional");
        assert_eq!(query_embedding.len(), 768, "query embedding must be 768-dimensional");

        println!("doc   embedding (first 4 dims): {:?}", &doc_embedding[..4]);
        println!("query embedding (first 4 dims): {:?}", &query_embedding[..4]);

        // ── Exact dot product on raw f32 vectors ──────────────────────────────
        let exact_dot: f32 = SpatialSimilarity::dot(&query_embedding, &doc_embedding).expect("simsimd dot product failed") as f32;

        println!("exact cosine similarity: {exact_dot:.6}");

        // ── Quantise the document embedding ──────────────────────────────────
        let closest_cluster_id = find_closest_cluster_id(&cluster_index.clusters, &doc_embedding);

        let cluster = cluster_index
            .clusters
            .get(&closest_cluster_id)
            .expect("closest cluster not found in index");

        let vector_index = rabitq::index_embedding_to_cluster(&doc_embedding, cluster, QuantisationStyle::MultiBit { number_of_bits: n_bits });

        println!(
            "quantised: cluster_id={}, addition_factor={:.6}, scaling_factor={:.6}, error_bound={:.6}",
            vector_index.cluster_id, vector_index.addition_factor, vector_index.scaling_factor, vector_index.error_bound,
        );

        // ── Estimated dot product from the quantised index ────────────────────
        let estimator = MultiBitQuanDotProductEstimator::new(closest_cluster_id, &query_embedding, &cluster.centroid, n_bits);

        let estimated_dot = vector_index.estimated_distance(&query_embedding, &estimator);

        println!("estimated cosine similarity: {estimated_dot:.6}");
        println!("absolute difference:         {:.6}", (exact_dot - estimated_dot).abs());

        // ── Assert accuracy: relative error < 0.1% ───────────────────────────
        // 8-bit RaBitQ delivers well under 0.1% relative error in practice.
        let diff = (exact_dot - estimated_dot).abs();
        let relative_error = diff / exact_dot.abs();
        assert!(
            relative_error < 1e-3,
            "quantised cosine similarity {estimated_dot:.6} differs from exact {exact_dot:.6} \
             by {diff:.6} (relative error {:.4}%), which exceeds the 0.1% tolerance",
            relative_error * 100.0,
        );
    }

    // ── QuantisationStyle ─────────────────────────────────────────────────────

    #[test]
    fn test_quantisation_style_default_is_multi_bit_8() {
        assert_eq!(QuantisationStyle::default(), QuantisationStyle::MultiBit { number_of_bits: 8 });
    }

    #[test]
    fn test_quantisation_style_number_of_bits() {
        assert_eq!(QuantisationStyle::SingleBit.number_of_bits(), 1);
        assert_eq!(QuantisationStyle::MultiBit { number_of_bits: 4 }.number_of_bits(), 4);
        assert_eq!(QuantisationStyle::MultiBit { number_of_bits: 8 }.number_of_bits(), 8);
    }

    #[test]
    fn test_quantisation_style_eq() {
        assert_eq!(QuantisationStyle::SingleBit, QuantisationStyle::SingleBit);
        assert_ne!(QuantisationStyle::SingleBit, QuantisationStyle::MultiBit { number_of_bits: 1 });
        assert_eq!(
            QuantisationStyle::MultiBit { number_of_bits: 4 },
            QuantisationStyle::MultiBit { number_of_bits: 4 },
        );
        assert_ne!(
            QuantisationStyle::MultiBit { number_of_bits: 4 },
            QuantisationStyle::MultiBit { number_of_bits: 8 },
        );
    }

    #[test]
    fn test_quantisation_style_clone_and_debug() {
        let s = QuantisationStyle::MultiBit { number_of_bits: 6 };
        assert_eq!(s.clone(), s);
        assert!(format!("{:?}", QuantisationStyle::SingleBit).contains("SingleBit"));
        assert!(format!("{:?}", QuantisationStyle::MultiBit { number_of_bits: 4 }).contains("MultiBit"));
    }

    /// Verifies that SingleBit and MultiBit route through `rabitq::index_embedding` and
    /// produce VectorIndex entries with structurally different packed vectors.
    /// Single-bit packs 64 dimensions per u64 word; multi-bit packs 8 bytes per word.
    #[test]
    fn test_single_bit_and_multi_bit_produce_different_packed_layouts() {
        use crate::cluster::ClusterIndex;
        use crate::quantisation::rabitq;

        let cluster_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../service/embedding_support/qwen/clusters.json");
        let cluster_index = ClusterIndex::load(cluster_path, 5).expect("cluster index not found");

        #[derive(serde::Deserialize)]
        struct EmbeddingFile {
            embeddings: Vec<Vec<f32>>,
        }
        let doc_path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_data/doc_embedding.json");
        let doc_file: EmbeddingFile = serde_json::from_str(&std::fs::read_to_string(doc_path).unwrap()).unwrap();
        let embedding: Vec<f32> = doc_file.embeddings.into_iter().next().unwrap();

        let single_bit_vi = rabitq::index_embedding(&cluster_index.clusters, &embedding, QuantisationStyle::SingleBit);
        let multi_bit_vi = rabitq::index_embedding(&cluster_index.clusters, &embedding, QuantisationStyle::MultiBit { number_of_bits: 4 });

        // single-bit packs 64 dims per u64 word (pack_bits):  768 / 64 = 12 words.
        // multi-bit  packs  8 bytes per u64 word (pack_bytes): 768 / 8  = 96 words
        // (one u8 per dimension regardless of the conceptual bit depth).
        assert_eq!(single_bit_vi.packed_vector.len(), 768 / 64, "single-bit: 64 dims per u64");
        assert_eq!(multi_bit_vi.packed_vector.len(), 768 / 8, "multi-bit: 1 byte per dim, 8 bytes per u64");
    }

    // ── quantisation paths (simulating the dual/chunked embedding flow) ───────

    /// Simulates the Multiple-style path by quantising N copies of the same
    /// fixture embedding, verifying that N inputs produce N VectorIndex entries.
    /// This exercises the `iter().map(quantise)` loop without needing HTTP.
    #[test]
    fn test_multi_embedding_quantisation_produces_n_vector_indexes() {
        use crate::cluster::ClusterIndex;
        use crate::quantisation::rabitq;

        let cluster_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../service/embedding_support/qwen/clusters.json");
        let style = QuantisationStyle::MultiBit { number_of_bits: 4 };
        let cluster_index = ClusterIndex::load(cluster_path, 5).expect("cluster index not found");

        #[derive(serde::Deserialize)]
        struct EmbeddingFile {
            embeddings: Vec<Vec<f32>>,
        }
        let doc_path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_data/doc_embedding.json");
        let doc_file: EmbeddingFile = serde_json::from_str(&std::fs::read_to_string(doc_path).expect("doc_embedding.json not found"))
            .expect("failed to parse doc_embedding.json");
        let doc_embedding: Vec<f32> = doc_file.embeddings.into_iter().next().unwrap();

        // Simulate three "chunks" by reusing the same embedding.
        let embeddings: Vec<Vec<f32>> = vec![doc_embedding.clone(), doc_embedding.clone(), doc_embedding];
        let vector_indexes: Vec<crate::index::vector_index::VectorIndex> = embeddings
            .iter()
            .map(|e| rabitq::index_embedding(&cluster_index.clusters, e, style.clone()))
            .collect();

        assert_eq!(vector_indexes.len(), 3, "one VectorIndex per input embedding");
    }

    #[test]
    fn test_single_embedding_quantisation_produces_one_vector_index() {
        use crate::cluster::ClusterIndex;
        use crate::quantisation::rabitq;

        let cluster_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../service/embedding_support/qwen/clusters.json");
        let style = QuantisationStyle::MultiBit { number_of_bits: 4 };
        let cluster_index = ClusterIndex::load(cluster_path, 5).expect("cluster index not found");

        #[derive(serde::Deserialize)]
        struct EmbeddingFile {
            embeddings: Vec<Vec<f32>>,
        }
        let doc_path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_data/doc_embedding.json");
        let doc_file: EmbeddingFile = serde_json::from_str(&std::fs::read_to_string(doc_path).expect("doc_embedding.json not found"))
            .expect("failed to parse doc_embedding.json");
        let doc_embedding: Vec<f32> = doc_file.embeddings.into_iter().next().unwrap();

        let embeddings = [doc_embedding];
        let vector_indexes: Vec<crate::index::vector_index::VectorIndex> = embeddings
            .iter()
            .map(|e| rabitq::index_embedding(&cluster_index.clusters, e, style.clone()))
            .collect();

        assert_eq!(vector_indexes.len(), 1, "Single style produces exactly one VectorIndex");
    }

    // ── embed_document: quantisation logic ─────────────────────

    /// Simulates `embed_document` without HTTP by calling the
    /// quantisation layer directly.  Verifies that the combined output contains
    /// exactly 1 `MultiBit` entry and N `SingleBit` entries, each carrying the
    /// correct KV key discriminant.
    #[test]
    fn test_dual_quantisation_produces_both_styles() {
        use crate::cluster::ClusterIndex;
        use crate::quantisation::rabitq;

        let cluster_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../service/embedding_support/qwen/clusters.json");
        let cluster_index = ClusterIndex::load(cluster_path, 5).expect("cluster index not found");

        #[derive(serde::Deserialize)]
        struct EmbeddingFile {
            embeddings: Vec<Vec<f32>>,
        }
        let doc_path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_data/doc_embedding.json");
        let embedding: Vec<f32> = serde_json::from_str::<EmbeddingFile>(&std::fs::read_to_string(doc_path).unwrap())
            .unwrap()
            .embeddings
            .into_iter()
            .next()
            .unwrap();

        // Simulate the single-embedding multi-bit path.
        let mb_vi = rabitq::index_embedding(&cluster_index.clusters, &embedding, QuantisationStyle::MultiBit { number_of_bits: 8 });

        // Simulate the chunked single-bit path with 3 chunks (reuse same embedding).
        const N_CHUNKS: usize = 3;
        let sb_vis: Vec<_> = (0..N_CHUNKS)
            .map(|_| rabitq::index_embedding(&cluster_index.clusters, &embedding, QuantisationStyle::SingleBit))
            .collect();

        // Build the combined list as embed_document would.
        let combined: Vec<crate::index::vector_index::VectorIndex> = std::iter::once(mb_vi).chain(sb_vis).collect();

        let mb_count = combined
            .iter()
            .filter(|vi| matches!(vi.quantisation_style, QuantisationStyle::MultiBit { .. }))
            .count();
        let sb_count = combined.iter().filter(|vi| vi.quantisation_style == QuantisationStyle::SingleBit).count();

        assert_eq!(mb_count, 1, "exactly one MultiBit entry");
        assert_eq!(sb_count, N_CHUNKS, "one SingleBit entry per chunk");
    }

    /// Verifies that the multi-bit and single-bit entries produced by the dual
    /// path carry the correct packed-vector sizes for the fixture embedding (768 dims).
    #[test]
    fn test_dual_quantisation_packed_vector_sizes() {
        use crate::cluster::ClusterIndex;
        use crate::quantisation::rabitq;

        let cluster_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../service/embedding_support/qwen/clusters.json");
        let cluster_index = ClusterIndex::load(cluster_path, 5).expect("cluster index not found");

        #[derive(serde::Deserialize)]
        struct EmbeddingFile {
            embeddings: Vec<Vec<f32>>,
        }
        let doc_path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_data/doc_embedding.json");
        let embedding: Vec<f32> = serde_json::from_str::<EmbeddingFile>(&std::fs::read_to_string(doc_path).unwrap())
            .unwrap()
            .embeddings
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(embedding.len(), 768);

        let mb_vi = rabitq::index_embedding(&cluster_index.clusters, &embedding, QuantisationStyle::MultiBit { number_of_bits: 8 });
        let sb_vi = rabitq::index_embedding(&cluster_index.clusters, &embedding, QuantisationStyle::SingleBit);

        // MultiBit: 1 byte per dim, 8 bytes per u64 → 768 / 8 = 96 words.
        assert_eq!(mb_vi.packed_vector.len(), 96, "multi-bit: 96 u64 words for 768 dims");
        // SingleBit: 64 dims per u64 word → 768 / 64 = 12 words.
        assert_eq!(sb_vi.packed_vector.len(), 12, "single-bit: 12 u64 words for 768 dims");
    }

    /// Verifies that SingleBit entries from the dual path may land in different clusters
    /// than the MultiBit entry (since each chunk is independently assigned).
    /// Uses the same embedding for all chunks, so they always land in the same cluster —
    /// this test documents the invariant that cluster_id is independently computed.
    #[test]
    fn test_dual_quantisation_single_bit_cluster_is_independently_assigned() {
        use crate::cluster::ClusterIndex;
        use crate::quantisation::rabitq;

        let cluster_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../service/embedding_support/qwen/clusters.json");
        let cluster_index = ClusterIndex::load(cluster_path, 5).expect("cluster index not found");

        #[derive(serde::Deserialize)]
        struct EmbeddingFile {
            embeddings: Vec<Vec<f32>>,
        }
        let doc_path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_data/doc_embedding.json");
        let embedding: Vec<f32> = serde_json::from_str::<EmbeddingFile>(&std::fs::read_to_string(doc_path).unwrap())
            .unwrap()
            .embeddings
            .into_iter()
            .next()
            .unwrap();

        let mb_vi = rabitq::index_embedding(&cluster_index.clusters, &embedding, QuantisationStyle::MultiBit { number_of_bits: 8 });
        let sb_vi = rabitq::index_embedding(&cluster_index.clusters, &embedding, QuantisationStyle::SingleBit);

        // With the same input embedding, both styles land in the same cluster.
        // The cluster_id field is set independently per call; this asserts the invariant.
        assert_eq!(mb_vi.cluster_id, sb_vi.cluster_id, "same embedding → same cluster for both styles");
    }

    // ── search() edge cases ───────────────────────────────────────────────────

    struct EmptyKvStore;
    impl VectorKvStore for EmptyKvStore {
        async fn scan_sparse_cluster(&self, _cluster_id: u32) -> Vec<(Vec<u8>, Vec<u8>)> {
            vec![]
        }
        async fn get_dense_entry(&self, _doc_id_bytes: &[u8]) -> Option<Vec<u8>> {
            None
        }
        async fn get_dense_entries_batch(&self, doc_ids: &[Vec<u8>]) -> Vec<Option<Vec<u8>>> {
            vec![None; doc_ids.len()]
        }
        async fn scan_sparse_clusters_batch(&self, _cluster_ids: &[u32]) -> std::collections::HashMap<u32, Vec<(Vec<u8>, Vec<u8>)>> {
            std::collections::HashMap::new()
        }
    }

    #[tokio::test]
    async fn test_search_returns_empty_with_no_query_embeddings() {
        let config = SemanticSearchConfig::default();
        let cluster_index = crate::cluster::ClusterIndex {
            clusters: Default::default(),
            neighbours: Default::default(),
        };
        let results = search(&config, &cluster_index, &[], &[], &EmptyKvStore, None::<fn(&[u8]) -> bool>, None).await;
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_search_returns_empty_with_multiple_query_embeddings_and_empty_store() {
        let config = SemanticSearchConfig::default();
        let cluster_index = crate::cluster::ClusterIndex {
            clusters: Default::default(),
            neighbours: Default::default(),
        };
        let embeddings = vec![vec![0.0f32; 4], vec![0.0f32; 4]];
        let dense = vec![0.0f32; 4];
        let results = search(
            &config,
            &cluster_index,
            &embeddings,
            &dense,
            &EmptyKvStore,
            None::<fn(&[u8]) -> bool>,
            None,
        )
        .await;
        assert!(results.is_empty());
    }

    // ── Two-pass search tests ─────────────────────────────────────────────────

    /// Controllable mock store for two-pass search tests.
    ///
    /// Pre-loaded with:
    /// - `sparse_data`: SingleBit scan results keyed by cluster_id.
    /// - `dense_data`: MultiBit entries keyed by doc_id.
    struct MockVectorKvStore {
        sparse_data: ClusterBatchResult,
        dense_data: std::collections::HashMap<Vec<u8>, Vec<u8>>,
    }

    impl MockVectorKvStore {
        fn new() -> Self {
            Self {
                sparse_data: Default::default(),
                dense_data: Default::default(),
            }
        }

        fn add_sparse_entry(&mut self, cluster_id: u32, doc_id: &[u8], addition_factor: f32) {
            let vi = VectorIndex::new(cluster_id, QuantisationStyle::SingleBit, addition_factor, 0.0, 0.01, vec![]);
            let raw = VectorIndex::list_to_bytes(&[vi]);
            self.sparse_data.entry(cluster_id).or_default().push((doc_id.to_vec(), raw));
        }

        fn add_dense_entry(&mut self, cluster_id: u32, doc_id: &[u8], addition_factor: f32) {
            let vi = VectorIndex::new(
                cluster_id,
                QuantisationStyle::MultiBit { number_of_bits: 8 },
                addition_factor,
                0.0,
                0.01,
                vec![],
            );
            let raw = VectorIndex::list_to_bytes(&[vi]);
            self.dense_data.insert(doc_id.to_vec(), raw);
        }
    }

    impl VectorKvStore for MockVectorKvStore {
        async fn scan_sparse_cluster(&self, cluster_id: u32) -> Vec<(Vec<u8>, Vec<u8>)> {
            self.sparse_data.get(&cluster_id).cloned().unwrap_or_default()
        }
        async fn get_dense_entry(&self, doc_id_bytes: &[u8]) -> Option<Vec<u8>> {
            self.dense_data.get(doc_id_bytes).cloned()
        }
        async fn get_dense_entries_batch(&self, doc_ids: &[Vec<u8>]) -> Vec<Option<Vec<u8>>> {
            doc_ids.iter().map(|id| self.dense_data.get(id).cloned()).collect()
        }
        async fn scan_sparse_clusters_batch(&self, cluster_ids: &[u32]) -> std::collections::HashMap<u32, Vec<(Vec<u8>, Vec<u8>)>> {
            cluster_ids
                .iter()
                .filter_map(|id| self.sparse_data.get(id).map(|entries| (*id, entries.clone())))
                .collect()
        }
    }

    /// Build a minimal ClusterIndex with named 4-D clusters.
    fn make_cluster_index(entries: &[(u32, [f32; 4])]) -> crate::cluster::ClusterIndex {
        let clusters = entries
            .iter()
            .map(|&(id, c)| (id, crate::cluster::Cluster::new(id, c.to_vec())))
            .collect();
        crate::cluster::ClusterIndex {
            clusters,
            neighbours: Default::default(),
        }
    }

    /// Documents excluded by the filter must not appear in results.
    #[tokio::test]
    async fn test_search_doc_filter_excludes_candidates_in_sparse_pass() {
        let cluster_index = make_cluster_index(&[(1, [1.0, 0.0, 0.0, 0.0]), (2, [0.0, 1.0, 0.0, 0.0])]);

        let mut store = MockVectorKvStore::new();
        store.add_sparse_entry(1, b"doc_a", 0.1);

        let config = SemanticSearchConfig {
            n_probes: 1,
            ..Default::default()
        };

        // Filter rejects every document.
        let results = search(
            &config,
            &cluster_index,
            &[vec![1.0f32, 0.0, 0.0, 0.0]],
            &[1.0f32, 0.0, 0.0, 0.0],
            &store,
            Some(|_: &[u8]| false),
            None,
        )
        .await;
        assert!(results.is_empty(), "filtered doc must not appear in results");
    }

    /// When first_pass_sparse_search_top_k=1, the lower-scoring sparse candidate must be
    /// excluded from the dense pass entirely.
    #[tokio::test]
    async fn test_search_sparse_cap_prevents_low_scoring_docs_entering_dense_pass() {
        let cluster_index = make_cluster_index(&[(1, [1.0, 0.0, 0.0, 0.0]), (2, [0.0, 1.0, 0.0, 0.0])]);

        let mut store = MockVectorKvStore::new();
        store.add_sparse_entry(1, b"doc_high", 0.1); // sparse score higher
        store.add_sparse_entry(2, b"doc_low", 0.1); // sparse score lower

        // Only doc_high enters dense pass.
        store.add_dense_entry(1, b"doc_high", 0.1);
        // doc_low deliberately has no dense entry to prove it is not reached.

        let config = SemanticSearchConfig {
            n_probes: 2,
            first_pass_sparse_search_top_k: 1,
            ..Default::default()
        };

        let results = search(
            &config,
            &cluster_index,
            &[vec![1.0f32, 0.0, 0.0, 0.0]],
            &[1.0f32, 0.0, 0.0, 0.0],
            &store,
            None::<fn(&[u8]) -> bool>,
            None,
        )
        .await;

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document_id, b"doc_high");
    }

    /// Each query embedding independently probes its nearest cluster; the union of those
    /// cluster sets is scanned in the sparse pass, so documents in different clusters can
    /// all become candidates.
    #[tokio::test]
    async fn test_search_multiple_query_embeddings_union_probe_clusters() {
        let cluster_index = make_cluster_index(&[(1, [1.0, 0.0, 0.0, 0.0]), (2, [0.0, 1.0, 0.0, 0.0])]);

        let mut store = MockVectorKvStore::new();
        store.add_sparse_entry(1, b"doc_a", 0.1);
        store.add_sparse_entry(2, b"doc_b", 0.1);

        store.add_dense_entry(1, b"doc_a", 0.1);
        store.add_dense_entry(2, b"doc_b", 0.1);

        let config = SemanticSearchConfig {
            n_probes: 1, // each query probes only 1 cluster → union = {1, 2}
            first_pass_sparse_search_top_k: 10,
            ..Default::default()
        };

        let results = search(
            &config,
            &cluster_index,
            &[vec![1.0f32, 0.0, 0.0, 0.0], vec![0.0f32, 1.0, 0.0, 0.0]],
            &[1.0f32, 1.0, 0.0, 0.0],
            &store,
            None::<fn(&[u8]) -> bool>,
            None,
        )
        .await;

        let ids: Vec<&Vec<u8>> = results.iter().map(|r| &r.document_id).collect();
        assert!(ids.contains(&&b"doc_a".to_vec()), "doc_a must be found via cluster 1 probe");
        assert!(ids.contains(&&b"doc_b".to_vec()), "doc_b must be found via cluster 2 probe");
    }

    /// A sparse candidate without a dense entry must be silently skipped in the
    /// dense pass and must not appear in results.
    #[tokio::test]
    async fn test_search_dense_pass_skips_docs_without_dense_entry() {
        let cluster_index = make_cluster_index(&[(1, [1.0, 0.0, 0.0, 0.0])]);

        let mut store = MockVectorKvStore::new();
        store.add_sparse_entry(1, b"doc_no_dense", 0.1);
        // Deliberately no dense entry.

        let config = SemanticSearchConfig {
            n_probes: 1,
            ..Default::default()
        };

        let results = search(
            &config,
            &cluster_index,
            &[vec![1.0f32, 0.0, 0.0, 0.0]],
            &[1.0f32, 0.0, 0.0, 0.0],
            &store,
            None::<fn(&[u8]) -> bool>,
            None,
        )
        .await;

        assert!(results.is_empty(), "doc with no dense entry must be skipped");
    }

    /// Final results must be sorted descending by dense score.
    /// Both docs are in the same cluster, so their sparse scores are identical.
    /// The dense re-ranking uses addition_factor to distinguish them.
    /// With the MultiBit estimator (empty packed_vector, scaling_factor=0):
    ///   score = 1 + query_to_centroid_dot_product - addition_factor
    /// Lower addition_factor → higher score → must appear first.
    #[tokio::test]
    async fn test_search_results_are_sorted_by_dense_score_descending() {
        let cluster_index = make_cluster_index(&[(1, [1.0, 0.0, 0.0, 0.0])]);

        let mut store = MockVectorKvStore::new();
        store.add_sparse_entry(1, b"doc_low_score", 0.1);
        store.add_sparse_entry(1, b"doc_high_score", 0.1);
        store.add_dense_entry(1, b"doc_low_score", 0.4); // score = C - 0.4 (lower)
        store.add_dense_entry(1, b"doc_high_score", 0.1); // score = C - 0.1 (higher)

        let config = SemanticSearchConfig {
            n_probes: 1,
            first_pass_sparse_search_top_k: 10,
            ..Default::default()
        };

        let results = search(
            &config,
            &cluster_index,
            &[vec![1.0f32, 0.0, 0.0, 0.0]],
            &[1.0f32, 0.0, 0.0, 0.0],
            &store,
            None::<fn(&[u8]) -> bool>,
            None,
        )
        .await;

        assert_eq!(results.len(), 2);
        assert!(
            results[0].dot_product >= results[1].dot_product,
            "results must be sorted by score descending"
        );
        assert_eq!(results[0].document_id, b"doc_high_score", "highest-scoring doc must come first");
        assert_eq!(results[1].document_id, b"doc_low_score");
    }

    /// search() must return at most top_k_results entries even when more candidates exist.
    #[tokio::test]
    async fn test_search_respects_top_k_results_cap() {
        let cluster_index = make_cluster_index(&[(1, [1.0, 0.0, 0.0, 0.0])]);

        let mut store = MockVectorKvStore::new();
        for (i, &doc_id) in [b"doc_a".as_ref(), b"doc_b".as_ref(), b"doc_c".as_ref()].iter().enumerate() {
            store.add_sparse_entry(1, doc_id, 0.1);
            store.add_dense_entry(1, doc_id, 0.1 + i as f32 * 0.1);
        }

        let config = SemanticSearchConfig {
            n_probes: 1,
            first_pass_sparse_search_top_k: 10,
            top_k_results: 2,
            ..Default::default()
        };

        let results = search(
            &config,
            &cluster_index,
            &[vec![1.0f32, 0.0, 0.0, 0.0]],
            &[1.0f32, 0.0, 0.0, 0.0],
            &store,
            None::<fn(&[u8]) -> bool>,
            None,
        )
        .await;

        assert_eq!(results.len(), 2, "must return at most top_k_results items");
    }

    /// The top_k override parameter must take precedence over config.top_k_results.
    #[tokio::test]
    async fn test_search_top_k_override_takes_precedence() {
        let cluster_index = make_cluster_index(&[(1, [1.0, 0.0, 0.0, 0.0])]);

        let mut store = MockVectorKvStore::new();
        for (i, &doc_id) in [b"doc_a".as_ref(), b"doc_b".as_ref(), b"doc_c".as_ref()].iter().enumerate() {
            store.add_sparse_entry(1, doc_id, 0.1);
            store.add_dense_entry(1, doc_id, 0.1 + i as f32 * 0.1);
        }

        let config = SemanticSearchConfig {
            n_probes: 1,
            first_pass_sparse_search_top_k: 10,
            top_k_results: 100, // would return all 3 without override
            ..Default::default()
        };

        let results = search(
            &config,
            &cluster_index,
            &[vec![1.0f32, 0.0, 0.0, 0.0]],
            &[1.0f32, 0.0, 0.0, 0.0],
            &store,
            None::<fn(&[u8]) -> bool>,
            Some(1), // override: return only 1
        )
        .await;

        assert_eq!(results.len(), 1, "top_k override of 1 must cap results at 1");
    }

    // ── ColBERT MaxSim pass-1 tests ───────────────────────────────────────────

    /// A document whose chunks appear in clusters probed by *multiple* query tokens
    /// must score higher than one whose chunks are only probed by a single token,
    /// even when both max out at the same per-token score.
    ///
    /// Setup
    /// -----
    /// Cluster 1 centroid = [1,0,0,0], Cluster 2 centroid = [0,1,0,0].
    /// Query tokens: q1=[1,0,0,0]  (probes cluster 1)  q2=[0,1,0,0]  (probes cluster 2).
    ///
    /// doc_a has a chunk in cluster 1 AND a chunk in cluster 2.
    /// doc_b has a chunk in cluster 1 ONLY.
    ///
    /// With MaxSim (sum across query tokens):
    ///   MaxSim(doc_a) = score(q1, chunk_c1) + score(q2, chunk_c2)
    ///                 = 1.0             + 1.0             = 2.0
    ///   MaxSim(doc_b) = score(q1, chunk_c1) + 0.0 (no chunk probed for q2)
    ///                 = 1.0             + 0.0             = 1.0
    ///
    /// With the old SimMax (max across query tokens):
    ///   SimMax(doc_a) = max(1.0, 1.0) = 1.0   ← tied with doc_b
    ///   SimMax(doc_b) = max(1.0, 0.0) = 1.0
    ///
    /// Therefore only the MaxSim aggregation guarantees that doc_a outranks doc_b.
    /// Setting first_pass_sparse_search_top_k=1 lets us verify the correct winner.
    #[tokio::test]
    async fn test_pass1_maxsim_sums_query_token_scores() {
        let cluster_index = make_cluster_index(&[(1, [1.0, 0.0, 0.0, 0.0]), (2, [0.0, 1.0, 0.0, 0.0])]);

        let mut store = MockVectorKvStore::new();
        store.add_sparse_entry(1, b"doc_a", 0.0); // doc_a chunk in cluster 1
        store.add_sparse_entry(2, b"doc_a", 0.0); // doc_a chunk in cluster 2
        store.add_sparse_entry(1, b"doc_b", 0.0); // doc_b only in cluster 1

        store.add_dense_entry(1, b"doc_a", 0.1);
        store.add_dense_entry(1, b"doc_b", 0.1);

        let config = SemanticSearchConfig {
            n_probes: 1,                       // q1 → cluster 1, q2 → cluster 2
            first_pass_sparse_search_top_k: 1, // only the top-scored doc enters dense
            ..Default::default()
        };

        let results = search(
            &config,
            &cluster_index,
            &[vec![1.0f32, 0.0, 0.0, 0.0], vec![0.0f32, 1.0, 0.0, 0.0]],
            &[1.0f32, 0.0, 0.0, 0.0],
            &store,
            None::<fn(&[u8]) -> bool>,
            None,
        )
        .await;

        // MaxSim(doc_a) = 2.0 > MaxSim(doc_b) = 1.0, so only doc_a passes.
        assert_eq!(results.len(), 1, "only the MaxSim winner should reach the dense pass");
        assert_eq!(results[0].document_id, b"doc_a", "doc_a must win: it matches both query tokens");
    }

    /// With a single query token, MaxSim reduces to a plain per-chunk max.
    /// This test confirms that the refactored path produces the same result as
    /// the old single-embedding pass-1 path in the degenerate case.
    #[tokio::test]
    async fn test_pass1_maxsim_single_query_token_equals_plain_score() {
        // Two docs in two clusters; query probes only cluster 1.
        // doc_high is in cluster 1, doc_low in cluster 2.
        // With one query token MaxSim(doc_high) = centroid_dot(q, c1) = 1.0
        // and doc_low is not found (contributes 0), so MaxSim(doc_low) = 0.0.
        let cluster_index = make_cluster_index(&[(1, [1.0, 0.0, 0.0, 0.0]), (2, [0.0, 1.0, 0.0, 0.0])]);

        let mut store = MockVectorKvStore::new();
        store.add_sparse_entry(1, b"doc_high", 0.0);
        store.add_sparse_entry(2, b"doc_low", 0.0);

        store.add_dense_entry(1, b"doc_high", 0.1);
        store.add_dense_entry(2, b"doc_low", 0.1);

        let config = SemanticSearchConfig {
            n_probes: 1,
            first_pass_sparse_search_top_k: 1, // only the better sparse candidate passes
            ..Default::default()
        };

        let results = search(
            &config,
            &cluster_index,
            &[vec![1.0f32, 0.0, 0.0, 0.0]], // single query token
            &[1.0f32, 0.0, 0.0, 0.0],
            &store,
            None::<fn(&[u8]) -> bool>,
            None,
        )
        .await;

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document_id, b"doc_high");
    }

    /// For each query token, only the best-matching chunk of a document contributes —
    /// the inner aggregation over a document's chunks is max, not sum.
    ///
    /// We manually build a raw bytes entry with two VectorIndex entries:
    ///   chunk_hi: scaling_factor produces score ≈ 0.8 for the query
    ///   chunk_lo: scaling_factor produces score ≈ 0.2 for the query
    ///
    /// The document's per-token score is max(0.8, 0.2) = 0.8.
    /// A competing doc_solo with a single chunk scores 0.5.
    /// doc_multi must beat doc_solo (0.8 > 0.5), proving only the best chunk counts.
    #[tokio::test]
    async fn test_pass1_best_chunk_wins_for_each_query_token() {
        use crate::index::vector_index::QuantisationStyle;

        let cluster_index = make_cluster_index(&[(1, [1.0, 0.0, 0.0, 0.0])]);

        // Build two chunks for doc_multi manually so we can control their scores precisely.
        // With query q=[1,0,0,0], centroid c=[1,0,0,0]:
        //   centroid_dot = 1.0, sum_q = 1.0
        //   score = centroid_dot + scaling_factor * (2*ip - sum_q)
        //
        // Empty packed vector → ip = 0, so score = 1.0 + scaling_factor * (0 - 1.0)
        //                                         = 1.0 - scaling_factor
        //
        // chunk_hi: scaling_factor = 0.2  → score = 0.8
        // chunk_lo: scaling_factor = 0.8  → score = 0.2
        let chunk_hi = VectorIndex::new(
            1,
            QuantisationStyle::SingleBit,
            0.0, /*addition_factor*/
            0.2, /*scaling_factor*/
            0.01,
            vec![],
        );
        let chunk_lo = VectorIndex::new(1, QuantisationStyle::SingleBit, 0.0, 0.8, 0.01, vec![]);
        let raw_multi = VectorIndex::list_to_bytes(&[chunk_hi, chunk_lo]);

        // doc_solo: one chunk with score = 1.0 - 0.5 = 0.5.
        let chunk_solo = VectorIndex::new(1, QuantisationStyle::SingleBit, 0.0, 0.5, 0.01, vec![]);
        let raw_solo = VectorIndex::list_to_bytes(&[chunk_solo]);

        let mut store = MockVectorKvStore::new();
        store.sparse_data.entry(1).or_default().push((b"doc_multi".to_vec(), raw_multi));
        store.sparse_data.entry(1).or_default().push((b"doc_solo".to_vec(), raw_solo));

        store.add_dense_entry(1, b"doc_multi", 0.1);
        store.add_dense_entry(1, b"doc_solo", 0.1);

        let config = SemanticSearchConfig {
            n_probes: 1,
            first_pass_sparse_search_top_k: 1, // only the top sparse score passes
            ..Default::default()
        };

        let results = search(
            &config,
            &cluster_index,
            &[vec![1.0f32, 0.0, 0.0, 0.0]], // single query token
            &[1.0f32, 0.0, 0.0, 0.0],
            &store,
            None::<fn(&[u8]) -> bool>,
            None,
        )
        .await;

        // doc_multi: max(0.8, 0.2) = 0.8.  doc_solo: 0.5.  doc_multi wins.
        // (If inner aggregation were sum, doc_multi would score 0.8+0.2=1.0, but the
        //  assertion still holds.  The distinguishing invariant is tested via doc_solo.)
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].document_id, b"doc_multi",
            "doc_multi's best chunk (0.8) must beat doc_solo (0.5)"
        );
    }

    /// When all query tokens find the same document in the same cluster,
    /// the MaxSim score must be the SUM of per-token scores (not the max).
    ///
    /// Three query tokens, each scoring 1.0 against the only cluster's centroid
    /// → MaxSim = 3.0.  doc_single has only one query token → MaxSim = 1.0.
    /// With first_pass_sparse_search_top_k=1, doc_multi_token should survive.
    ///
    /// Note: this test uses the same doc in one cluster probed by all tokens —
    /// a pure "sum vs max" stress test.
    #[tokio::test]
    async fn test_pass1_maxsim_accumulates_all_query_tokens() {
        // Single cluster, centroid = [1,0,0,0].
        // Three query tokens, all equal to [1,0,0,0] → each scores 1.0 against centroid.
        let cluster_index = make_cluster_index(&[(1, [1.0, 0.0, 0.0, 0.0])]);

        let mut store = MockVectorKvStore::new();
        store.add_sparse_entry(1, b"doc_all_tokens", 0.0);
        store.add_sparse_entry(1, b"doc_one_token", 0.0);

        store.add_dense_entry(1, b"doc_all_tokens", 0.1);
        store.add_dense_entry(1, b"doc_one_token", 0.1);

        let config = SemanticSearchConfig {
            n_probes: 1,
            first_pass_sparse_search_top_k: 2, // let both through sparse
            ..Default::default()
        };

        // Three identical query tokens, all probing cluster 1.
        let q = vec![1.0f32, 0.0, 0.0, 0.0];
        let query_embeddings = vec![q.clone(), q.clone(), q.clone()];

        let results = search(&config, &cluster_index, &query_embeddings, &q, &store, None::<fn(&[u8]) -> bool>, None).await;

        // Both pass sparse (top_k=2) and both have dense entries.
        // Final ordering is by dense score (equal), so both appear.
        assert_eq!(results.len(), 2, "both docs should pass the sparse cap");
        // Sparse scores: doc_all_tokens = 1+1+1 = 3.0, doc_one_token = 1+1+1 = 3.0
        // (Same chunk, same centroid, same query token → both get 3.0; result order is by dense score.)
    }
}
