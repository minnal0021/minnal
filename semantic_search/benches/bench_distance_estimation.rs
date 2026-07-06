//! Benchmarks for first-pass (SingleBit) and second-pass (MultiBit) distance estimation.
//!
//!   cargo bench -p semantic_search --bench bench_distance_estimation
//!
//! On first run the bench generates 1 000 synthetic 768-d embeddings via a seeded PRNG,
//! quantises them against the real cluster centroids, and saves the raw float vectors to
//! `benches/data/bench_embeddings.json` for reuse.
//!
//! To use embeddings from the live embedding service instead, populate that file yourself
//! using the service's HTTP API (POST to /embedding/document) before running the
//! bench.  The expected JSON format is:
//!   { "query": [f32; 768], "docs": [[f32; 768]; N] }

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use semantic_search::{
    ClusterIndex,
    cluster::{Cluster, find_closest_cluster_id, find_top_n_cluster_ids, read_clusters_from_file},
    index::distance_estimator::{DistanceEstimator, MultiBitQuanDotProductEstimator, SingleBitQuanDotProductEstimator},
    index::vector_index::{ClusterBatchResult, QuantisationStyle, VectorIndex, VectorKvStore},
    index_embedding_to_cluster,
    service::{SemanticSearchConfig, search},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hint::black_box;
use std::time::Duration;

const DIM: usize = 768;
const N_BITS_MULTI: usize = 8;
const N_DOCS: usize = 1_000;

// Relative to the crate root (semantic_search/).
const CLUSTER_PATH: &str = "../service/embedding_support/qwen/clusters.json";
const BENCH_DATA_PATH: &str = "benches/data/bench_embeddings.json";

// ── Embedding data ─────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct BenchEmbeddings {
    query: Vec<f32>,
    docs: Vec<Vec<f32>>,
}

/// Load embeddings from `BENCH_DATA_PATH` if present; otherwise generate
/// `N_DOCS` synthetic embeddings via a deterministic XorShift64 PRNG, save
/// them to disk, and return them.
fn load_or_generate_embeddings() -> BenchEmbeddings {
    let path = std::path::Path::new(BENCH_DATA_PATH);
    if path.exists() {
        let json = std::fs::read_to_string(path).expect("bench embeddings: read failed");
        return serde_json::from_str(&json).expect("bench embeddings: parse failed");
    }

    // XorShift64 — fast, reproducible, no extra dep.
    let mut s = 0xfeed_face_cafe_dead_u64;
    let mut next_f32 = move || -> f32 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s as i64 as f32) / (i64::MAX as f32)
    };

    let query: Vec<f32> = (0..DIM).map(|_| next_f32()).collect();
    let docs: Vec<Vec<f32>> = (0..N_DOCS).map(|_| (0..DIM).map(|_| next_f32()).collect()).collect();

    let data = BenchEmbeddings { query, docs };

    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string(&data) {
        let _ = std::fs::write(path, json);
    }

    data
}

// ── Setup ──────────────────────────────────────────────────────────────────────

struct BenchSetup {
    query: Vec<f32>,
    single_bit_entries: Vec<VectorIndex>,
    multi_bit_entries: Vec<VectorIndex>,
    single_bit_estimator: SingleBitQuanDotProductEstimator,
    multi_bit_estimator: MultiBitQuanDotProductEstimator,
}

fn setup() -> BenchSetup {
    let raw = read_clusters_from_file(CLUSTER_PATH)
        .expect("bench setup: failed to load clusters — is service/embedding_support/qwen/clusters.json present?");
    let cluster_map: HashMap<u32, Cluster> = raw.into_iter().map(|(id, c)| (id, Cluster::new(id, c))).collect();

    let embeddings = load_or_generate_embeddings();
    let query = embeddings.query.clone();

    // Build per-query estimator state once — both estimators share the same
    // cluster id (the one closest to the query).
    let qcid = find_closest_cluster_id(&cluster_map, &query);
    let centroid = &cluster_map[&qcid].centroid;

    let single_bit_estimator = SingleBitQuanDotProductEstimator::new(qcid, &query, centroid);
    let scaled_sum = MultiBitQuanDotProductEstimator::scaled_query_sum(&query, N_BITS_MULTI);
    let multi_bit_estimator = MultiBitQuanDotProductEstimator::with_scaled_query_sum(qcid, &query, centroid, scaled_sum);

    // Quantise each doc embedding and assign it to its closest cluster.
    let single_bit_entries: Vec<VectorIndex> = embeddings
        .docs
        .iter()
        .map(|emb| {
            let cid = find_closest_cluster_id(&cluster_map, emb);
            index_embedding_to_cluster(emb, &cluster_map[&cid], QuantisationStyle::SingleBit)
        })
        .collect();

    let multi_bit_entries: Vec<VectorIndex> = embeddings
        .docs
        .iter()
        .map(|emb| {
            let cid = find_closest_cluster_id(&cluster_map, emb);
            index_embedding_to_cluster(
                emb,
                &cluster_map[&cid],
                QuantisationStyle::MultiBit {
                    number_of_bits: N_BITS_MULTI,
                },
            )
        })
        .collect();

    BenchSetup {
        query,
        single_bit_entries,
        multi_bit_entries,
        single_bit_estimator,
        multi_bit_estimator,
    }
}

// ── First pass: SingleBit estimation ──────────────────────────────────────────

/// Simulates the scoring loop that runs over all SingleBit entries in a scanned
/// IVF cluster during Pass 1.  Each entry is a 1-bit quantised doc embedding;
/// the estimator accumulates the dot-product estimate from a packed inner product
/// via SIMD (`packed_ip_best`).
fn bench_first_pass(c: &mut Criterion) {
    let s = setup();
    let mut group = c.benchmark_group("first_pass/single_bit");
    group.measurement_time(Duration::from_secs(10));

    for &n in &[100usize, 500, 1_000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let entries = &s.single_bit_entries[..n];
            b.iter(|| {
                let mut score = 0.0f32;
                for vi in entries {
                    score += s.single_bit_estimator.estimate_distance(black_box(&s.query), black_box(vi));
                }
                black_box(score)
            });
        });
    }

    group.finish();
}

// ── Second pass: MultiBit estimation ──────────────────────────────────────────

/// Simulates the re-ranking loop in Pass 2 over the top-k candidates returned
/// by Pass 1.  Each entry is a multi-bit (8 b/dim) quantised dense embedding;
/// the estimator unpacks 8 bytes per u64 word and computes a weighted dot
/// product using the quantised values directly.
fn bench_second_pass(c: &mut Criterion) {
    let s = setup();
    let mut group = c.benchmark_group("second_pass/multi_bit");
    group.measurement_time(Duration::from_secs(10));

    for &n in &[100usize, 500, 1_000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let entries = &s.multi_bit_entries[..n];
            b.iter(|| {
                let mut score = 0.0f32;
                for vi in entries {
                    score += s.multi_bit_estimator.estimate_distance(black_box(&s.query), black_box(vi));
                }
                black_box(score)
            });
        });
    }

    group.finish();
}

// ── Top-n cluster selection: select_nth vs full sort ──────────────────────────

/// The old O(C log C) implementation: sort every cluster, then truncate to n.
/// Kept here as the baseline to compare against the current select_nth-based
/// `find_top_n_cluster_ids`.
fn full_sort_top_n(cluster_map: &HashMap<u32, Cluster>, embedding: &[f32], n: usize) -> Vec<u32> {
    let mut distances: Vec<(u32, f32)> = cluster_map.values().map(|c| (c.cluster_id, c.euclidean_distance(embedding))).collect();
    distances.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    distances.truncate(n);
    distances.into_iter().map(|(id, _)| id).collect()
}

/// Compares the current selection-based `find_top_n_cluster_ids` against the old
/// full-sort baseline over the real cluster file, at a few `n_probes` values. This
/// is the Pass-1 cluster-probing cost, paid once per query chunk.
fn bench_top_n_cluster_selection(c: &mut Criterion) {
    let raw = read_clusters_from_file(CLUSTER_PATH).expect("bench setup: failed to load clusters");
    let cluster_map: HashMap<u32, Cluster> = raw.into_iter().map(|(id, c)| (id, Cluster::new(id, c))).collect();
    let query = load_or_generate_embeddings().query;
    let count = cluster_map.len();

    let mut group = c.benchmark_group("top_n_cluster_selection");
    for &n in &[8usize, 32, 128] {
        group.bench_with_input(BenchmarkId::new("select_nth", n), &n, |b, &n| {
            b.iter(|| black_box(find_top_n_cluster_ids(black_box(&cluster_map), black_box(&query), n)));
        });
        group.bench_with_input(BenchmarkId::new("full_sort", n), &n, |b, &n| {
            b.iter(|| black_box(full_sort_top_n(black_box(&cluster_map), black_box(&query), n)));
        });
    }
    group.finish();
    eprintln!("top_n_cluster_selection: benchmarked over {count} clusters");
}

// ── Coarse assignment: serial-HashMap (before) vs batched-matrix (after) ───────

/// Generate `t` synthetic query chunks of `DIM` via a seeded XorShift64 PRNG.
fn synthetic_query_chunks(t: usize) -> Vec<Vec<f32>> {
    let mut s = 0x00c0_ffee_1234_5678_u64;
    let mut next_f32 = move || -> f32 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s as i64 as f32) / (i64::MAX as f32)
    };
    (0..t).map(|_| (0..DIM).map(|_| next_f32()).collect()).collect()
}

/// Union the top-`n` clusters across query chunks, the way `service::search` does —
/// so the bench measures the whole Pass-1 coarse-assignment step, not just one scan.
fn union_dedup(per_query: Vec<Vec<u32>>) -> Vec<u32> {
    let mut seen = std::collections::HashSet::new();
    let mut ids = Vec::new();
    for chunk in per_query {
        for id in chunk {
            if seen.insert(id) {
                ids.push(id);
            }
        }
    }
    ids
}

/// The full Pass-1 coarse-assignment cost (`T·C·D` + union), paid once per query.
///
/// `serial_hashmap` is the historical path — a serial loop of `find_top_n_cluster_ids`
/// over the scattered `HashMap<u32, Cluster>`. `batched_matrix` is the replacement —
/// `ClusterIndex::find_top_n_cluster_ids_batch`, parallel over chunks over a contiguous
/// centroid matrix. Both produce the same probe set; this measures the speed-up as the
/// query grows (the `T` factor the old serial loop scaled linearly with).
fn bench_coarse_assignment(c: &mut Criterion) {
    let raw = read_clusters_from_file(CLUSTER_PATH).expect("bench setup: failed to load clusters");
    let cluster_map: HashMap<u32, Cluster> = raw.into_iter().map(|(id, c)| (id, Cluster::new(id, c))).collect();
    let count = cluster_map.len();
    let index = ClusterIndex::from_clusters(cluster_map.clone());
    const N_PROBES: usize = 128; // production default

    let mut group = c.benchmark_group("coarse_assignment");
    group.measurement_time(Duration::from_secs(10));

    for &t in &[4usize, 40, 100] {
        let queries = synthetic_query_chunks(t);
        group.throughput(Throughput::Elements(t as u64));

        group.bench_with_input(BenchmarkId::new("serial_hashmap", t), &t, |b, _| {
            b.iter(|| {
                let per_query: Vec<Vec<u32>> = queries
                    .iter()
                    .map(|q| find_top_n_cluster_ids(black_box(&cluster_map), black_box(q), N_PROBES))
                    .collect();
                black_box(union_dedup(per_query))
            });
        });

        group.bench_with_input(BenchmarkId::new("batched_matrix", t), &t, |b, _| {
            b.iter(|| {
                let per_query = index.find_top_n_cluster_ids_batch(black_box(&queries), N_PROBES);
                black_box(union_dedup(per_query))
            });
        });
    }
    group.finish();
    eprintln!("coarse_assignment: {count} clusters, dim {DIM}, n_probes {N_PROBES}");
}

// ── End-to-end search(): where does the coarse-assignment win land? ────────────

/// In-memory `VectorKvStore` holding real quantised vectors, so an end-to-end
/// `search()` bench pays realistic Pass-1 (sparse scan + MaxSim) and Pass-2 (dense
/// re-rank) cost — not the empty-payload cost of the unit-test mock.
struct BenchKvStore {
    sparse_data: ClusterBatchResult,
    dense_data: HashMap<Vec<u8>, Vec<u8>>,
}

impl VectorKvStore for BenchKvStore {
    async fn scan_sparse_cluster(&self, cluster_id: u32) -> Vec<(Vec<u8>, Vec<u8>)> {
        self.sparse_data.get(&cluster_id).cloned().unwrap_or_default()
    }
    async fn get_dense_entry(&self, doc_id_bytes: &[u8]) -> Option<Vec<u8>> {
        self.dense_data.get(doc_id_bytes).cloned()
    }
    async fn get_dense_entries_batch(&self, doc_ids: &[Vec<u8>]) -> Vec<Option<Vec<u8>>> {
        doc_ids.iter().map(|id| self.dense_data.get(id).cloned()).collect()
    }
    async fn scan_sparse_clusters_batch(&self, cluster_ids: &[u32]) -> ClusterBatchResult {
        cluster_ids
            .iter()
            .filter_map(|id| self.sparse_data.get(id).map(|e| (*id, e.clone())))
            .collect()
    }
}

/// Build a store of `n_docs` documents: `chunks_per_doc` SingleBit sparse chunks
/// (each assigned to its nearest cluster) plus one MultiBit dense whole-doc entry,
/// quantised against the real centroids — the same shape `index`/`search` see in
/// production. A doc's chunks are grouped by cluster, so a doc appears once per
/// cluster its chunks landed in, with the rkyv list holding just that cluster's
/// chunks (the real `{ns}_sparse_vector` layout). `chunks_per_doc = 1` reproduces
/// the original single-chunk store.
fn build_bench_store(cluster_map: &HashMap<u32, Cluster>, n_docs: usize, chunks_per_doc: usize) -> BenchKvStore {
    // Deterministic doc embeddings, independent of the query PRNG.
    let mut s = 0xdead_beef_0bad_f00d_u64;
    let mut next_f32 = move || -> f32 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s as i64 as f32) / (i64::MAX as f32)
    };

    let mut sparse_data: ClusterBatchResult = HashMap::new();
    let mut dense_data: HashMap<Vec<u8>, Vec<u8>> = HashMap::new();
    for d in 0..n_docs {
        let doc_id = (d as u64).to_be_bytes().to_vec();

        // Whole-doc dense vector (payload[0] in real indexing), assigned to its cluster.
        let dense_emb: Vec<f32> = (0..DIM).map(|_| next_f32()).collect();
        let dcid = find_closest_cluster_id(cluster_map, &dense_emb);
        let dense = index_embedding_to_cluster(
            &dense_emb,
            &cluster_map[&dcid],
            QuantisationStyle::MultiBit {
                number_of_bits: N_BITS_MULTI,
            },
        );
        dense_data.insert(doc_id.clone(), VectorIndex::list_to_bytes(&[dense]));

        // Sparse chunks grouped by their assigned cluster → one entry per (doc, cluster).
        let mut by_cluster: HashMap<u32, Vec<VectorIndex>> = HashMap::new();
        for _ in 0..chunks_per_doc {
            let emb: Vec<f32> = (0..DIM).map(|_| next_f32()).collect();
            let cid = find_closest_cluster_id(cluster_map, &emb);
            let sparse = index_embedding_to_cluster(&emb, &cluster_map[&cid], QuantisationStyle::SingleBit);
            by_cluster.entry(cid).or_default().push(sparse);
        }
        for (cid, chunks) in by_cluster {
            sparse_data.entry(cid).or_default().push((doc_id.clone(), VectorIndex::list_to_bytes(&chunks)));
        }
    }
    BenchKvStore { sparse_data, dense_data }
}

/// Full two-pass `search()` over a realistic in-memory store, to see whether the
/// coarse-assignment change (a Pass-1 sub-step) moves total search latency.
///
/// `search()` here excludes the external embedding-service round trip (it takes
/// pre-computed query embeddings) — that network+GPU call dominates a real query and
/// dwarfs everything measured below. Compare the `coarse_assignment` group at the same
/// T to read off the coarse-assignment fraction of this total.
fn bench_end_to_end_search(c: &mut Criterion) {
    const N_DOCS: usize = 5_000;
    let raw = read_clusters_from_file(CLUSTER_PATH).expect("bench setup: failed to load clusters");
    let cluster_map: HashMap<u32, Cluster> = raw.into_iter().map(|(id, c)| (id, Cluster::new(id, c))).collect();
    let index = ClusterIndex::from_clusters(cluster_map.clone());
    let store = build_bench_store(&cluster_map, N_DOCS, 1);
    let dense_query = load_or_generate_embeddings().query;
    let config = SemanticSearchConfig::default(); // n_probes 128, top_k 1000/100

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("bench: tokio runtime");
    let no_filter: Option<fn(&[u8]) -> bool> = None;

    let mut group = c.benchmark_group("end_to_end_search");
    group.measurement_time(Duration::from_secs(10));

    for &t in &[4usize, 40] {
        let sparse_query = synthetic_query_chunks(t);
        group.throughput(Throughput::Elements(t as u64));
        group.bench_with_input(BenchmarkId::from_parameter(t), &t, |b, _| {
            b.iter(|| {
                let results = rt.block_on(search(
                    black_box(&config),
                    "bench_ns",
                    black_box(&index),
                    black_box(&sparse_query),
                    black_box(&dense_query),
                    black_box(&store),
                    no_filter,
                    None,
                ));
                black_box(results)
            });
        });
    }
    group.finish();
    eprintln!(
        "end_to_end_search: {N_DOCS} docs, {} clusters, n_probes {}",
        cluster_map.len(),
        config.n_probes
    );
}

// ── End-to-end vs chunks-per-doc: how much of search() is really Pass-1? ────────
//
// The single-chunk `end_to_end_search` under-represents Pass-1: with one sparse chunk
// per doc, Pass-1's absolute scoring cost is tiny and Pass-2 dense re-rank dominates.
// Real docs have several sliding-window chunks. This sweep holds T and n_docs fixed and
// varies `chunks_per_doc`, so Pass-2 stays ~constant (it re-ranks a top_k-capped
// candidate set, independent of chunk count) while Pass-1 scales with total chunks.
// Reading `end_to_end(cpd) − end_to_end(1)` therefore backs out Pass-1's real cost /
// share — the number that decides whether pruning Pass-1 (lever #2) is worth pursuing.
fn bench_end_to_end_multichunk(c: &mut Criterion) {
    const N_DOCS: usize = 5_000;
    const T: usize = 4; // representative query length
    let raw = read_clusters_from_file(CLUSTER_PATH).expect("bench setup: failed to load clusters");
    let cluster_map: HashMap<u32, Cluster> = raw.into_iter().map(|(id, c)| (id, Cluster::new(id, c))).collect();
    let index = ClusterIndex::from_clusters(cluster_map.clone());
    let dense_query = load_or_generate_embeddings().query;
    let sparse_query = synthetic_query_chunks(T);
    let config = SemanticSearchConfig::default();

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("bench: tokio runtime");
    let no_filter: Option<fn(&[u8]) -> bool> = None;

    let mut group = c.benchmark_group("end_to_end_multichunk");
    group.measurement_time(Duration::from_secs(10));

    for &cpd in &[1usize, 4, 8] {
        let store = build_bench_store(&cluster_map, N_DOCS, cpd);
        group.throughput(Throughput::Elements((N_DOCS * cpd) as u64)); // total sparse chunks
        group.bench_with_input(BenchmarkId::new("chunks_per_doc", cpd), &cpd, |b, _| {
            b.iter(|| {
                let results = rt.block_on(search(
                    black_box(&config),
                    "bench_ns",
                    black_box(&index),
                    black_box(&sparse_query),
                    black_box(&dense_query),
                    black_box(&store),
                    no_filter,
                    None,
                ));
                black_box(results)
            });
        });
    }
    group.finish();
    eprintln!("end_to_end_multichunk: {N_DOCS} docs, T={T}, {} clusters, n_probes {}", cluster_map.len(), config.n_probes);
}

// ── Pass-1 scoring isolation: dot arithmetic vs deserialize vs hashmap ─────────
//
// Decomposes the Pass-1 MaxSim inner fold (`service::search`, the loop over a probed
// cluster's entries at `service/mod.rs`) into three CUMULATIVE layers over ONE fixed
// candidate set, so the marginal cost of each stage is directly readable:
//
//   dot_arithmetic  pure `estimate_from_parts` + running-max, over pre-widened native
//                   u64 buffers — no rkyv, no map. This is the SIMD masked-sum the
//                   "can we short-circuit the dot product" question targets.
//   plus_archived   + `VectorIndex::access_list` (validated zero-copy) and
//                   `copy_packed_into` per chunk — the deserialize+widen the real
//                   path actually pays (it does NOT own-deserialize).
//   plus_hashmap    + accumulate into the `HashMap<doc_id, Vec<f32>>` MaxSim state,
//                   exactly like the real fold. This layer == real Pass-1 scoring.
//
// Read the attribution as:
//   dot_arithmetic                 → the dot products themselves
//   plus_archived − dot_arithmetic → access_list + copy_packed_into (deserialize/widen)
//   plus_hashmap  − plus_archived  → the MaxSim state map
//
// Throughput is set to the number of dot products (docs × chunks × tokens), so criterion
// reports per-dot-product time — compare it against the deserialize/map deltas to decide
// whether optimising the dot product can move Pass-1 at all.

const CHUNKS_PER_DOC: usize = 4; // realistic sliding-window chunk count per doc
const SCORING_N_DOCS: usize = 2_000;

/// One pre-built candidate set, materialised in every representation the three layers
/// need so each layer times only its own added work.
struct ScoringInput {
    /// L1 input: per doc, a list of `(native u64 words, scaling_factor)` chunks —
    /// already widened, so the dot layer touches no rkyv.
    native: Vec<Vec<(Vec<u64>, f32)>>,
    /// L2/L3 input: per doc, `(doc_id, rkyv list blob)` — the on-disk shape the real
    /// path scans and `access_list`es.
    blobs: Vec<(Vec<u8>, Vec<u8>)>,
    /// One estimator per query token (against that token's own closest centroid) —
    /// reused across all docs, as the real per-cluster fold reuses its estimator set.
    estimators: Vec<SingleBitQuanDotProductEstimator>,
    /// The `t` sparse query-token embeddings.
    query: Vec<Vec<f32>>,
}

fn build_scoring_input(t: usize, n_docs: usize) -> ScoringInput {
    let raw = read_clusters_from_file(CLUSTER_PATH).expect("scoring bench: failed to load clusters");
    let cluster_map: HashMap<u32, Cluster> = raw.into_iter().map(|(id, c)| (id, Cluster::new(id, c))).collect();

    let query = synthetic_query_chunks(t);
    let estimators: Vec<SingleBitQuanDotProductEstimator> = query
        .iter()
        .map(|q| {
            let cid = find_closest_cluster_id(&cluster_map, q);
            SingleBitQuanDotProductEstimator::new(cid, q, &cluster_map[&cid].centroid)
        })
        .collect();

    // Doc embeddings from an independent PRNG stream (distinct seed from the query's).
    let mut s = 0x51ce_d0c5_1234_9abc_u64;
    let mut next_f32 = move || -> f32 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s as i64 as f32) / (i64::MAX as f32)
    };

    let mut native = Vec::with_capacity(n_docs);
    let mut blobs = Vec::with_capacity(n_docs);
    for d in 0..n_docs {
        let mut chunks = Vec::with_capacity(CHUNKS_PER_DOC);
        let mut nat = Vec::with_capacity(CHUNKS_PER_DOC);
        for _ in 0..CHUNKS_PER_DOC {
            let emb: Vec<f32> = (0..DIM).map(|_| next_f32()).collect();
            let cid = find_closest_cluster_id(&cluster_map, &emb);
            let vi = index_embedding_to_cluster(&emb, &cluster_map[&cid], QuantisationStyle::SingleBit);
            nat.push((vi.packed_vector.clone(), vi.scaling_factor));
            chunks.push(vi);
        }
        native.push(nat);
        blobs.push(((d as u64).to_be_bytes().to_vec(), VectorIndex::list_to_bytes(&chunks)));
    }

    ScoringInput {
        native,
        blobs,
        estimators,
        query,
    }
}

/// Collapse a doc's per-token running maxes into its MaxSim score, NEG_INFINITY → 0
/// (a token that matched no chunk contributes 0). Mirrors `service::maxsim_score`.
#[inline]
fn collapse(maxes: &[f32]) -> f32 {
    maxes.iter().map(|&m| if m == f32::NEG_INFINITY { 0.0 } else { m }).sum()
}

fn bench_pass1_scoring_isolation(c: &mut Criterion) {
    let mut group = c.benchmark_group("pass1_scoring");
    group.measurement_time(Duration::from_secs(10));

    for &t in &[4usize, 40] {
        let inp = build_scoring_input(t, SCORING_N_DOCS);
        let n_dots = (SCORING_N_DOCS * CHUNKS_PER_DOC * t) as u64;
        group.throughput(Throughput::Elements(n_dots));

        // L1 — pure dot + running-max over pre-widened native buffers.
        group.bench_with_input(BenchmarkId::new("dot_arithmetic", t), &t, |b, _| {
            let mut maxes = vec![f32::NEG_INFINITY; t];
            b.iter(|| {
                let mut acc = 0.0f32;
                for doc in &inp.native {
                    maxes.iter_mut().for_each(|m| *m = f32::NEG_INFINITY);
                    for (words, scaling) in doc {
                        for (i, (q, est)) in inp.query.iter().zip(&inp.estimators).enumerate() {
                            let sc = est.estimate_from_parts(q, words, *scaling);
                            if sc > maxes[i] {
                                maxes[i] = sc;
                            }
                        }
                    }
                    acc += collapse(&maxes);
                }
                black_box(acc)
            });
        });

        // L2 — + access_list (zero-copy deserialize) + copy_packed_into (widen) per chunk.
        group.bench_with_input(BenchmarkId::new("plus_archived", t), &t, |b, _| {
            let mut words_buf: Vec<u64> = Vec::new();
            let mut maxes = vec![f32::NEG_INFINITY; t];
            b.iter(|| {
                let mut acc = 0.0f32;
                for (_doc_id, blob) in &inp.blobs {
                    let list = VectorIndex::access_list(blob).expect("scoring bench: archive access");
                    maxes.iter_mut().for_each(|m| *m = f32::NEG_INFINITY);
                    for vi in list.iter() {
                        vi.copy_packed_into(&mut words_buf);
                        let scaling = vi.scaling_factor();
                        for (i, (q, est)) in inp.query.iter().zip(&inp.estimators).enumerate() {
                            let sc = est.estimate_from_parts(q, &words_buf, scaling);
                            if sc > maxes[i] {
                                maxes[i] = sc;
                            }
                        }
                    }
                    acc += collapse(&maxes);
                }
                black_box(acc)
            });
        });

        // L3 — + the HashMap<doc_id, Vec<f32>> MaxSim state map. == real Pass-1 scoring.
        group.bench_with_input(BenchmarkId::new("plus_hashmap", t), &t, |b, _| {
            b.iter(|| {
                let mut map: HashMap<&[u8], Vec<f32>> = HashMap::new();
                let mut words_buf: Vec<u64> = Vec::new();
                for (doc_id, blob) in &inp.blobs {
                    let list = VectorIndex::access_list(blob).expect("scoring bench: archive access");
                    let maxes = map.entry(doc_id.as_slice()).or_insert_with(|| vec![f32::NEG_INFINITY; t]);
                    for vi in list.iter() {
                        vi.copy_packed_into(&mut words_buf);
                        let scaling = vi.scaling_factor();
                        for (i, (q, est)) in inp.query.iter().zip(&inp.estimators).enumerate() {
                            let sc = est.estimate_from_parts(q, &words_buf, scaling);
                            if sc > maxes[i] {
                                maxes[i] = sc;
                            }
                        }
                    }
                }
                let acc: f32 = map.values().map(|v| collapse(v)).sum();
                black_box(acc)
            });
        });
    }
    group.finish();
    eprintln!("pass1_scoring: {SCORING_N_DOCS} docs × {CHUNKS_PER_DOC} chunks, dim {DIM}");
}

// ── Criterion wiring ──────────────────────────────────────────────────────────

criterion_group!(
    distance_estimation,
    bench_first_pass,
    bench_second_pass,
    bench_top_n_cluster_selection,
    bench_coarse_assignment,
    bench_end_to_end_search,
    bench_end_to_end_multichunk,
    bench_pass1_scoring_isolation
);
criterion_main!(distance_estimation);
