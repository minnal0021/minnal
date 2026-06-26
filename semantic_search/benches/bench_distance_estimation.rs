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
    cluster::{Cluster, find_closest_cluster_id, find_top_n_cluster_ids, read_clusters_from_file},
    index::distance_estimator::{DistanceEstimator, MultiBitQuanDotProductEstimator, SingleBitQuanDotProductEstimator},
    index::vector_index::{QuantisationStyle, VectorIndex},
    index_embedding_to_cluster,
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

// ── Criterion wiring ──────────────────────────────────────────────────────────

criterion_group!(distance_estimation, bench_first_pass, bench_second_pass, bench_top_n_cluster_selection);
criterion_main!(distance_estimation);
