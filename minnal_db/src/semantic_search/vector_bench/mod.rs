//! Vector-index benchmark on frozen embeddings (design doc M0).
//!
//! Indexes a BEIR dataset's saved embeddings through the production write path
//! ([`index_embeddings`] + [`upsert_vectors`]), runs every judged query through
//! the production [`search`], and measures relevance, recall against the exact
//! (unquantised, unpartitioned) ranking, estimator error, partition shape,
//! footprint, indexing throughput and latency over a sweep of fixed `n_probes`
//! and of entry budgets (design doc M2d).
//! Results go to `{bench_root}/results/{label}/{dataset}-{order}.{json,md}`;
//! `vector_bench_compare` compares two result files with paired per-query deltas.
//!
//! The first run for a dataset dumps its embeddings through the embedding service
//! (see [`frozen`]); later runs never call it. Run from the crate root
//! (`minnal_db/`), on an idle host:
//!
//! ```sh
//! MINNAL_BENCH_DATASET=scifact MINNAL_BENCH_LABEL=m0-baseline \
//!   cargo test -p minnal_db --all-features --release --lib vector_bench -- --ignored --nocapture --test-threads=1
//! MINNAL_BENCH_BASE=../work/bench/results/m0-baseline/scifact-corpus.json \
//! MINNAL_BENCH_NEW=../work/bench/results/m1/scifact-corpus.json \
//!   cargo test -p minnal_db --all-features --release --lib vector_bench_compare -- --ignored --nocapture
//! ```
//!
//! Env knobs (all optional): `MINNAL_BENCH_DATASET` (`scifact`),
//! `MINNAL_BENCH_LABEL` (`run`), `MINNAL_BENCH_ORDER` (`corpus` | `shuffled` |
//! `drifting`), `MINNAL_BENCH_REPEATS` (latency passes per setting, 3),
//! `MINNAL_BENCH_ROOT` (`../work/bench`), `MINNAL_BEIR_ROOT` (`../work/beir`),
//! `MINNAL_BEIR_SPLIT` (`test`), `MINNAL_BENCH_MODEL` (`gemma`),
//! `MINNAL_EMBED_URL` (only for the dump).

mod code_width;
mod exact;
mod frozen;
mod study;

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;

use super::metrics::beir_eval::{env_or, ndcg_at, read_qrels, score};
use crate::semantic_search::ClusterIndex;
use crate::semantic_search::cluster::{Cluster, find_closest_cluster_id, read_clusters_from_file};
use crate::semantic_search::index::distance_estimator::{DistanceEstimator, MultiBitQuanDotProductEstimator, SingleBitQuanDotProductEstimator};
use crate::semantic_search::index::vector_index::{QuantisationStyle, VectorIndex};
use crate::semantic_search::quantisation::rabitq::index_embedding_rotated;
use crate::semantic_search::service::{DEFAULT_PROBE, SemanticSearchConfig, index_embeddings, search};
use crate::semantic_search::{IvfLayout, ProbeSettings, select_probes};
use crate::vector_kv::{DbVectorStore, dense_vectors_ns, sparse_vectors_meta_ns, sparse_vectors_ns, upsert_vectors};
use crate::{AsyncDb, DbConfig};
use exact::{FINAL_K, GroundTruth, dot};
use frozen::Frozen;

const NS: &str = "bench";
/// Documents indexed concurrently (quantisation + upsert).
const INDEX_CONCURRENCY: usize = 8;
/// `n_probes` values swept; values above the cluster count are skipped.
const NPROBES: [usize; 7] = [4, 8, 16, 32, 64, 128, 256];
/// The fixed probe count gates compare against (the production setting before M2d).
const PRODUCTION_NPROBES: usize = 64;
/// Entry budgets swept, as shares of the namespace's entries (plus the default
/// budget, [`DEFAULT_PROBE`]), so a small corpus gets a curve too.
const BUDGET_SHARES: [f64; 9] = [0.025, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0];
/// Candidates kept after Pass 1 (production default), and returned per query so
/// the whole candidate list can be scored.
const FIRST_PASS: usize = 1000;
/// Queries sampled for the estimator-error measurement.
const ESTIMATOR_QUERIES: usize = 100;
/// Per sampled query: the exact top this many, plus this many random picks.
const ESTIMATOR_TOP: usize = 25;
const ESTIMATOR_RANDOM: usize = 25;
const SEED: u64 = 42;
/// nDCG cutoffs reported beside nDCG@10 (and gated on by `vector_bench_compare`).
const NDCG_CUTOFFS: [usize; 6] = [10, 20, 30, 40, 50, 100];

// ── Result types (serialised; `vector_bench_compare` reads them back) ──────────

#[derive(Serialize, Deserialize, Default, Clone, Copy)]
struct ErrorStats {
    pairs: usize,
    rmse: f64,
    bias: f64,
    /// Share of pairs whose error is within the entry's `error_bound`.
    within_bound: f64,
}

#[derive(Serialize, Deserialize)]
struct Partition {
    clusters_used: usize,
    entries: usize,
    chunks: usize,
    largest_share_chunks: f64,
    p99_cluster_chunks: usize,
    /// Coefficient of variation of chunks per used cluster.
    cv_chunks: f64,
}

#[derive(Serialize, Deserialize)]
struct Footprint {
    sparse_bytes: u64,
    sparse_meta_bytes: u64,
    dense_bytes: u64,
    sparse_keys: usize,
}

#[derive(Serialize, Deserialize, Clone, Copy, Default)]
struct QueryResultRow {
    ndcg10: f64,
    mrr10: f64,
    recall100: f64,
    cand_recall: f64,
    /// Share of the exact final top 10 in the index's top 10.
    ann_r10: f64,
    /// Share of the exact final top 100 in the index's top 100.
    ann_r100: f64,
    /// Share of the exact Pass-1 candidates among the index's candidates.
    pass1_recall: f64,
    entries_scanned: usize,
    chunks_scanned: usize,
    latency_ms: f64,
    /// nDCG at each of [`NDCG_CUTOFFS`] (absent from results written before it existed).
    #[serde(default)]
    ndcg_at: [f64; NDCG_CUTOFFS.len()],
    /// Postings probed (absent from results written before M2d).
    #[serde(default)]
    probes: f64,
}

#[derive(Serialize, Deserialize)]
struct Setting {
    /// Fixed probe count, or 0 for a budget row.
    n_probes: usize,
    /// Entry budget (min 1, max [`DEFAULT_PROBE`]'s), or `None` for a fixed row.
    #[serde(default)]
    budget_entries: Option<u64>,
    mean: QueryResultRow,
    latency_p50_ms: f64,
    latency_p95_ms: f64,
    latency_p99_ms: f64,
    /// p50 of each repeat, to show run-to-run noise.
    repeat_p50_ms: Vec<f64>,
    /// Queries whose result list differed between repeats.
    nondeterministic_queries: usize,
    per_query: Vec<QueryResultRow>,
}

#[derive(Serialize, Deserialize)]
struct BenchResult {
    label: String,
    git: String,
    dataset: String,
    model: String,
    order: String,
    docs: usize,
    chunks: usize,
    queries: Vec<String>,
    index_seconds: f64,
    docs_per_second: f64,
    footprint: Footprint,
    partition: Partition,
    estimator_pass1: ErrorStats,
    estimator_pass2: ErrorStats,
    settings: Vec<Setting>,
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let i = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[i]
}

fn git_describe() -> String {
    let run = |args: &[&str]| {
        std::process::Command::new("git")
            .args(args)
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default()
    };
    let head = run(&["rev-parse", "--short", "HEAD"]);
    let dirty = !run(&["status", "--porcelain", "--untracked-files=no"]).is_empty();
    if dirty { format!("{head}-dirty") } else { head }
}

/// Insertion order of doc rows.
fn insertion_order(order: &str, frozen: &Frozen, index: &ClusterIndex) -> Vec<usize> {
    let mut rows: Vec<usize> = (0..frozen.n_docs()).collect();
    match order {
        "corpus" => {}
        "shuffled" => {
            let mut s = SEED;
            for i in (1..rows.len()).rev() {
                let j = (splitmix(&mut s) % (i as u64 + 1)) as usize;
                rows.swap(i, j);
            }
        }
        // Topics arrive one at a time: docs grouped by their nearest bundled centroid.
        "drifting" => {
            let key: Vec<u32> = rows.iter().map(|&d| find_closest_cluster_id(&index.clusters, frozen.dense(d))).collect();
            rows.sort_by_key(|&d| (key[d], d));
        }
        other => panic!("unknown MINNAL_BENCH_ORDER '{other}' (corpus | shuffled | drifting)"),
    }
    rows
}

fn error_stats(errors: &[(f32, f32)]) -> ErrorStats {
    let n = errors.len().max(1) as f64;
    ErrorStats {
        pairs: errors.len(),
        rmse: (errors.iter().map(|(e, _)| (*e as f64).powi(2)).sum::<f64>() / n).sqrt(),
        bias: errors.iter().map(|(e, _)| *e as f64).sum::<f64>() / n,
        within_bound: errors.iter().filter(|(e, b)| e.abs() <= *b).count() as f64 / n,
    }
}

/// Estimator error on sampled (query, chunk) and (query, doc) pairs: each query's
/// exact top [`ESTIMATOR_TOP`] plus [`ESTIMATOR_RANDOM`] random picks.
///
/// Builds codes and estimators the way indexing and `search()` do (rotated codes and
/// query terms, ⟨q, c⟩ unrotated); a milestone that changes that wiring (centres)
/// must change it here too.
fn estimator_errors(frozen: &Frozen, gt: &GroundTruth, index: &ClusterIndex, dense_bits: usize) -> (ErrorStats, ErrorStats) {
    let nq = frozen.n_queries();
    let step = (nq / ESTIMATOR_QUERIES).max(1);
    let mut rng = SEED;
    let (mut sparse_err, mut dense_err) = (Vec::new(), Vec::new());
    for qi in (0..nq).step_by(step).take(ESTIMATOR_QUERIES) {
        let q = frozen.query(qi);
        let mut chunks: Vec<usize> = gt.pass1[qi]
            .iter()
            .take(ESTIMATOR_TOP)
            .filter_map(|&d| {
                let rows = frozen.chunk_rows(d as usize);
                rows.max_by(|&a, &b| dot(q, frozen.chunk(a)).total_cmp(&dot(q, frozen.chunk(b))))
            })
            .collect();
        chunks.extend((0..ESTIMATOR_RANDOM).map(|_| (splitmix(&mut rng) % frozen.n_chunks() as u64) as usize));
        // Wired as indexing and `search()` are: codes and the query's code-facing
        // terms in rotated space, ⟨q, c⟩ in the original space.
        let rq = index.rotate(q);
        for c in chunks {
            let x = frozen.chunk(c);
            let vi = index_embedding_rotated(index, x, QuantisationStyle::SingleBit).unwrap();
            let centroid = &index.clusters[&vi.cluster_id].centroid;
            let sum = SingleBitQuanDotProductEstimator::query_sum(&rq);
            let est = SingleBitQuanDotProductEstimator::with_query_sum(vi.cluster_id, q, centroid, sum).estimate_distance(&rq, &vi);
            sparse_err.push((est - dot(q, x), vi.error_bound));
        }
        let mut docs: Vec<usize> = gt.final_ranking[qi].iter().take(ESTIMATOR_TOP).map(|&d| d as usize).collect();
        docs.extend((0..ESTIMATOR_RANDOM).map(|_| (splitmix(&mut rng) % frozen.n_docs() as u64) as usize));
        for d in docs {
            let x = frozen.dense(d);
            let style = QuantisationStyle::MultiBit { number_of_bits: dense_bits };
            // Production encodes whole-document codes against the zero centre.
            let vi = crate::semantic_search::index_embedding_zero_centred(index, x, style);
            let origin = vec![0.0f32; x.len()];
            let sum = MultiBitQuanDotProductEstimator::scaled_query_sum(&rq, dense_bits);
            let est = MultiBitQuanDotProductEstimator::with_scaled_query_sum(vi.centre_id, q, &origin, sum).estimate_distance(&rq, &vi);
            dense_err.push((est - dot(q, x), vi.error_bound));
        }
    }
    (error_stats(&sparse_err), error_stats(&dense_err))
}

async fn namespace_bytes(db: &AsyncDb, name: String) -> (u64, Vec<(Vec<u8>, Vec<u8>)>) {
    let rows = db.namespace(name).await.unwrap().scan_prefix(Vec::new()).await.unwrap();
    let bytes = rows.iter().map(|(k, v)| (k.len() + v.len()) as u64).sum();
    (bytes, rows)
}

/// Overlap of the first `k` of `got` with the first `k` of `want`, over `min(k, |want|)`.
fn overlap_at(got: &[u32], want: &[u32], k: usize) -> f64 {
    let want: HashSet<u32> = want.iter().take(k).copied().collect();
    if want.is_empty() {
        return 1.0;
    }
    got.iter().take(k).filter(|d| want.contains(d)).count() as f64 / want.len() as f64
}

fn mean_row(rows: &[QueryResultRow]) -> QueryResultRow {
    let n = rows.len().max(1) as f64;
    let f = |g: fn(&QueryResultRow) -> f64| rows.iter().map(g).sum::<f64>() / n;
    QueryResultRow {
        ndcg10: f(|r| r.ndcg10),
        mrr10: f(|r| r.mrr10),
        recall100: f(|r| r.recall100),
        cand_recall: f(|r| r.cand_recall),
        ann_r10: f(|r| r.ann_r10),
        ann_r100: f(|r| r.ann_r100),
        pass1_recall: f(|r| r.pass1_recall),
        entries_scanned: (rows.iter().map(|r| r.entries_scanned).sum::<usize>() as f64 / n).round() as usize,
        chunks_scanned: (rows.iter().map(|r| r.chunks_scanned).sum::<usize>() as f64 / n).round() as usize,
        latency_ms: f(|r| r.latency_ms),
        ndcg_at: std::array::from_fn(|i| rows.iter().map(|r| r.ndcg_at[i]).sum::<f64>() / n),
        probes: f(|r| r.probes),
    }
}

impl Setting {
    fn probe(&self) -> ProbeSettings {
        match self.budget_entries {
            Some(budget_entries) => ProbeSettings {
                budget_entries,
                ..DEFAULT_PROBE
            },
            None => ProbeSettings::fixed(self.n_probes),
        }
    }

    fn name(&self) -> String {
        match self.budget_entries {
            Some(b) => format!("budget {b}"),
            None => format!("n_probes {}", self.n_probes),
        }
    }

    fn same_as(&self, other: &Setting) -> bool {
        (self.n_probes, self.budget_entries) == (other.n_probes, other.budget_entries)
    }
}

// ── The benchmark ────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn vector_bench() {
    let dataset = env_or("MINNAL_BENCH_DATASET", "scifact");
    let label = env_or("MINNAL_BENCH_LABEL", "run");
    let order = env_or("MINNAL_BENCH_ORDER", "corpus");
    let repeats: usize = env_or("MINNAL_BENCH_REPEATS", "3").parse().unwrap();
    let bench_root = PathBuf::from(env_or("MINNAL_BENCH_ROOT", "../work/bench"));
    let beir_root = PathBuf::from(env_or("MINNAL_BEIR_ROOT", "../work/beir"));
    let split = env_or("MINNAL_BEIR_SPLIT", "test");
    let model = env_or("MINNAL_BENCH_MODEL", "gemma");
    let config = Arc::new(SemanticSearchConfig {
        model_name: model.clone(),
        embedding_service_url: std::env::var("MINNAL_EMBED_URL").unwrap_or_else(|_| SemanticSearchConfig::default().embedding_service_url),
        ..SemanticSearchConfig::default()
    });
    let dataset_dir = beir_root.join(&dataset);
    if !dataset_dir.join("corpus.jsonl").exists() {
        eprintln!("SKIP vector_bench: {} missing", dataset_dir.join("corpus.jsonl").display());
        return;
    }

    // ── Frozen embeddings and exact ground truth ──
    let frozen_dir = bench_root.join(&dataset).join(&model);
    let frozen = Arc::new(
        frozen::load_or_dump(
            &frozen_dir,
            &frozen::DumpSpec {
                dataset: &dataset,
                dataset_dir: &dataset_dir,
                split: &split,
                model: &model,
                dim: config.embedding_dim,
                window_size: config.window_size,
                sliding_size: config.sliding_size,
                embed_url: &config.embedding_service_url,
            },
        )
        .await,
    );
    let gt = exact::load_or_compute(&frozen_dir, &frozen);
    let qrels = read_qrels(&dataset_dir.join("qrels").join(format!("{split}.tsv")));
    eprintln!(
        "\n=== vector bench: {dataset} ({split}), model {model}, order {order}, label {label} ===\n  {} docs, {} chunks, {} queries",
        frozen.n_docs(),
        frozen.n_chunks(),
        frozen.n_queries()
    );

    let cluster_path = format!("../service/embedding_support/{model}/clusters.json");
    let raw = read_clusters_from_file(&cluster_path).unwrap_or_else(|e| panic!("load {cluster_path}: {e}"));
    let index = Arc::new(ClusterIndex::from_clusters(
        raw.iter().map(|(&id, c)| (id, Cluster::new(id, c.clone()))).collect(),
    ));

    // ── Index through the production write path ──
    let tmp = tempfile::TempDir::new().unwrap();
    let db = Arc::new(AsyncDb::open_with_config(tmp.path().to_owned(), DbConfig::default()).await.unwrap());
    db.namespace(NS.to_string()).await.unwrap();
    // The namespace's own partition, seeded and loaded back as a store does it:
    // indexing and search run against it, not the file.
    crate::vector_kv::seed_ivf(&db, NS, &raw).await.unwrap();
    let ivf = Arc::new(
        crate::vector_kv::load_ivf(&db, NS, crate::semantic_search::cluster::DEFAULT_ROTATION_SEED)
            .await
            .unwrap()
            .expect("just seeded"),
    );
    let rows = insertion_order(&order, &frozen, &index);
    let t = Instant::now();
    {
        let sem = Arc::new(Semaphore::new(INDEX_CONCURRENCY));
        let mut set = tokio::task::JoinSet::new();
        for d in rows {
            let permit = sem.clone().acquire_owned().await.unwrap();
            let (frozen, config, ivf, db) = (frozen.clone(), config.clone(), ivf.clone(), db.clone());
            set.spawn(async move {
                let _permit = permit;
                let vis = index_embeddings(&config, &ivf, frozen.dense(d), &frozen.doc_chunks(d)).unwrap();
                upsert_vectors(&db, NS, &(d as u64).to_be_bytes(), "", &vis).await.unwrap();
            });
        }
        while let Some(r) = set.join_next().await {
            r.unwrap();
        }
    }
    let index_seconds = t.elapsed().as_secs_f64();
    eprintln!("  indexed in {index_seconds:.1}s ({:.0} docs/s)", frozen.n_docs() as f64 / index_seconds);
    // Settle the LSM so every setting reads the same on-disk shape.
    db.shutdown().await.unwrap();
    drop(db);
    let db = Arc::new(AsyncDb::open_with_config(tmp.path().to_owned(), DbConfig::default()).await.unwrap());
    db.compact().await.unwrap();
    // Loaded again as a restarted store loads it: its entry counts come from the
    // stored keys.
    let ivf = Arc::new(
        crate::vector_kv::load_ivf(&db, NS, crate::semantic_search::cluster::DEFAULT_ROTATION_SEED)
            .await
            .unwrap()
            .expect("seeded"),
    );

    // ── Footprint and partition shape ──
    let (sparse_bytes, sparse_rows) = namespace_bytes(&db, sparse_vectors_ns(NS)).await;
    let (sparse_meta_bytes, _) = namespace_bytes(&db, sparse_vectors_meta_ns(NS)).await;
    let (dense_bytes, _) = namespace_bytes(&db, dense_vectors_ns(NS)).await;
    let mut cluster_entries: HashMap<u32, usize> = HashMap::new();
    let mut cluster_chunks: HashMap<u32, usize> = HashMap::new();
    for (k, v) in &sparse_rows {
        let cluster = u32::from_be_bytes(k[..4].try_into().unwrap());
        *cluster_entries.entry(cluster).or_default() += 1;
        *cluster_chunks.entry(cluster).or_default() += VectorIndex::access_list(v).unwrap().len();
    }
    let total_chunks: usize = cluster_chunks.values().sum();
    let mut sizes: Vec<usize> = cluster_chunks.values().copied().collect();
    sizes.sort_unstable();
    let mean_size = total_chunks as f64 / sizes.len().max(1) as f64;
    let sd = (sizes.iter().map(|&s| (s as f64 - mean_size).powi(2)).sum::<f64>() / sizes.len().max(1) as f64).sqrt();
    let partition = Partition {
        clusters_used: sizes.len(),
        entries: sparse_rows.len(),
        chunks: total_chunks,
        largest_share_chunks: *sizes.last().unwrap_or(&0) as f64 / total_chunks.max(1) as f64,
        p99_cluster_chunks: sizes[((sizes.len().max(1) - 1) as f64 * 0.99).round() as usize],
        cv_chunks: sd / mean_size,
    };
    let footprint = Footprint {
        sparse_bytes,
        sparse_meta_bytes,
        dense_bytes,
        sparse_keys: sparse_rows.len(),
    };
    drop(sparse_rows);
    for (&posting, &n) in &cluster_entries {
        assert_eq!(ivf.posting_entries(posting), Some(n as u64), "entry count of posting {posting}");
    }

    let (estimator_pass1, estimator_pass2) = estimator_errors(&frozen, &gt, &index, config.number_of_bits_for_dense_quantisation);

    // ── Search sweep ──
    let store = DbVectorStore::new(&db, NS).await.unwrap();
    let doc_ids = &frozen.manifest.doc_ids;
    let run_query = |probe: ProbeSettings, qi: usize| {
        let cfg = SemanticSearchConfig {
            probe,
            first_pass_sparse_search_top_k: FIRST_PASS,
            ..(*config).clone()
        };
        let (store, ivf, frozen) = (&store, &ivf, &frozen);
        async move {
            let q = frozen.query(qi).to_vec();
            let t = Instant::now();
            let results = search(
                &cfg,
                NS,
                ivf,
                std::slice::from_ref(&q),
                &q,
                store,
                None::<fn(&[u8]) -> bool>,
                Some(FIRST_PASS),
            )
            .await
            .expect("search");
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let ranked: Vec<u32> = results
                .iter()
                .map(|r| u64::from_be_bytes(r.document_id[..].try_into().unwrap()) as u32)
                .collect();
            (ranked, ms)
        }
    };
    for qi in 0..frozen.n_queries() {
        run_query(ProbeSettings::fixed(PRODUCTION_NPROBES), qi).await; // warm-up
    }
    let total_entries = partition.entries as u64;
    let mut budgets: Vec<u64> = BUDGET_SHARES.iter().map(|s| (s * total_entries as f64).round() as u64).collect();
    budgets.push(DEFAULT_PROBE.budget_entries);
    budgets.sort_unstable();
    budgets.dedup();
    let sweep = NPROBES
        .iter()
        .filter(|&&n| n <= index.len())
        .map(|&n| (n, None))
        .chain(budgets.into_iter().map(|b| (0, Some(b))));
    let mut settings = Vec::new();
    for (n_probes, budget_entries) in sweep {
        let mut s = Setting {
            n_probes,
            budget_entries,
            mean: QueryResultRow::default(),
            latency_p50_ms: 0.0,
            latency_p95_ms: 0.0,
            latency_p99_ms: 0.0,
            repeat_p50_ms: Vec::new(),
            nondeterministic_queries: 0,
            per_query: Vec::new(),
        };
        let probe = s.probe();
        let mut rows: Vec<QueryResultRow> = Vec::with_capacity(frozen.n_queries());
        let mut first: Vec<Vec<u32>> = Vec::new();
        let mut latencies: Vec<f64> = Vec::new();
        let mut repeat_p50 = Vec::new();
        let mut nondeterministic: HashSet<usize> = HashSet::new();
        for rep in 0..repeats.max(1) {
            let mut lat = Vec::with_capacity(frozen.n_queries());
            for qi in 0..frozen.n_queries() {
                let (ranked, ms) = run_query(probe, qi).await;
                lat.push(ms);
                if rep == 0 {
                    let q = frozen.query(qi);
                    let (probed, _) = select_probes(&*ivf, &[q.to_vec()], &probe);
                    let ids: Vec<&str> = ranked.iter().map(|&d| doc_ids[d as usize].as_str()).collect();
                    let rels = &qrels[&frozen.manifest.query_ids[qi]];
                    let m = score(&ids, rels);
                    let pass1: HashSet<u32> = gt.pass1[qi].iter().copied().collect();
                    rows.push(QueryResultRow {
                        ndcg_at: NDCG_CUTOFFS.map(|k| ndcg_at(&ids, rels, k)),
                        ndcg10: m.ndcg,
                        mrr10: m.mrr,
                        recall100: m.recall,
                        cand_recall: m.cand_recall,
                        ann_r10: overlap_at(&ranked, &gt.final_ranking[qi], 10),
                        ann_r100: overlap_at(&ranked, &gt.final_ranking[qi], FINAL_K),
                        pass1_recall: ranked.iter().filter(|d| pass1.contains(d)).count() as f64 / pass1.len().max(1) as f64,
                        entries_scanned: probed.iter().map(|c| cluster_entries.get(c).copied().unwrap_or(0)).sum(),
                        chunks_scanned: probed.iter().map(|c| cluster_chunks.get(c).copied().unwrap_or(0)).sum(),
                        latency_ms: 0.0,
                        probes: probed.len() as f64,
                    });
                    first.push(ranked);
                } else if ranked != first[qi] {
                    nondeterministic.insert(qi);
                }
            }
            for (qi, ms) in lat.iter().enumerate() {
                if rep == 0 {
                    rows[qi].latency_ms = *ms;
                }
            }
            let mut sorted = lat.clone();
            sorted.sort_by(f64::total_cmp);
            repeat_p50.push(percentile(&sorted, 0.5));
            latencies.extend(lat);
        }
        latencies.sort_by(f64::total_cmp);
        s.mean = mean_row(&rows);
        s.latency_p50_ms = percentile(&latencies, 0.5);
        s.latency_p95_ms = percentile(&latencies, 0.95);
        s.latency_p99_ms = percentile(&latencies, 0.99);
        s.repeat_p50_ms = repeat_p50;
        s.nondeterministic_queries = nondeterministic.len();
        s.per_query = rows;
        eprintln!(
            "  {:>16}: nDCG@10 {:.4}  ANN R@10 {:.4}  Pass-1 recall {:.4}  entries {:>6}  probes {:>6.1}  p50 {:.2} ms  p95 {:.2} ms",
            s.name(),
            s.mean.ndcg10,
            s.mean.ann_r10,
            s.mean.pass1_recall,
            s.mean.entries_scanned,
            s.mean.probes,
            s.latency_p50_ms,
            s.latency_p95_ms
        );
        settings.push(s);
    }

    let result = BenchResult {
        label: label.clone(),
        git: git_describe(),
        dataset: dataset.clone(),
        model,
        order: order.clone(),
        docs: frozen.n_docs(),
        chunks: frozen.n_chunks(),
        queries: frozen.manifest.query_ids.clone(),
        index_seconds,
        docs_per_second: frozen.n_docs() as f64 / index_seconds,
        footprint,
        partition,
        estimator_pass1,
        estimator_pass2,
        settings,
    };
    let out_dir = bench_root.join("results").join(&label);
    std::fs::create_dir_all(&out_dir).unwrap();
    let stem = format!("{dataset}-{order}");
    std::fs::write(out_dir.join(format!("{stem}.json")), serde_json::to_vec(&result).unwrap()).unwrap();
    let md = render(&result);
    std::fs::write(out_dir.join(format!("{stem}.md")), &md).unwrap();
    eprintln!("\n{md}\n  written to {}", out_dir.join(format!("{stem}.{{json,md}}")).display());
    db.shutdown().await.unwrap();
}

fn render(r: &BenchResult) -> String {
    let mut md = String::new();
    let _ = writeln!(
        md,
        "# Vector bench — {} ({}), {} order, label `{}`\n",
        r.dataset, r.model, r.order, r.label
    );
    let _ = writeln!(
        md,
        "Commit `{}`. {} docs, {} chunks, {} queries. Indexed in {:.1} s ({:.0} docs/s, embedding excluded).\n",
        r.git,
        r.docs,
        r.chunks,
        r.queries.len(),
        r.index_seconds,
        r.docs_per_second
    );
    let p = &r.partition;
    let _ = writeln!(
        md,
        "Partition: {} clusters used, {} entries, {} chunks; largest cluster {:.1}% of chunks, p99 {} chunks, CV {:.2}.",
        p.clusters_used,
        p.entries,
        p.chunks,
        p.largest_share_chunks * 100.0,
        p.p99_cluster_chunks,
        p.cv_chunks
    );
    let f = &r.footprint;
    let _ = writeln!(
        md,
        "Footprint: sparse {:.1} MiB ({} keys), sparse meta {:.1} MiB, dense {:.1} MiB.\n",
        f.sparse_bytes as f64 / 1048576.0,
        f.sparse_keys,
        f.sparse_meta_bytes as f64 / 1048576.0,
        f.dense_bytes as f64 / 1048576.0
    );
    let _ = writeln!(md, "| Estimator | Pairs | RMSE | Bias | Within error bound |\n|---|---:|---:|---:|---:|");
    for (name, e) in [("Pass 1 (1-bit chunks)", &r.estimator_pass1), ("Pass 2 (dense)", &r.estimator_pass2)] {
        let _ = writeln!(
            md,
            "| {name} | {} | {:.5} | {:+.5} | {:.1}% |",
            e.pairs,
            e.rmse,
            e.bias,
            e.within_bound * 100.0
        );
    }
    let _ = writeln!(
        md,
        "\n| Probing | nDCG@10 | MRR@10 | R@100 | Cand. recall | ANN R@10 | ANN R@100 | Pass-1 recall | Entries scanned | Postings probed | p50 ms | p95 ms | p99 ms | Nondet. |"
    );
    let _ = writeln!(md, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    for s in &r.settings {
        let m = &s.mean;
        let _ = writeln!(
            md,
            "| {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {} | {:.1} | {:.2} | {:.2} | {:.2} | {} |",
            s.name(),
            m.ndcg10,
            m.mrr10,
            m.recall100,
            m.cand_recall,
            m.ann_r10,
            m.ann_r100,
            m.pass1_recall,
            m.entries_scanned,
            m.probes,
            s.latency_p50_ms,
            s.latency_p95_ms,
            s.latency_p99_ms,
            s.nondeterministic_queries
        );
    }
    let _ = writeln!(
        md,
        "\n| Probing | {} |\n|---|{}",
        NDCG_CUTOFFS.map(|k| format!("nDCG@{k}")).join(" | "),
        "---:|".repeat(NDCG_CUTOFFS.len())
    );
    for s in &r.settings {
        let _ = writeln!(md, "| {} | {} |", s.name(), s.mean.ndcg_at.map(|v| format!("{v:.4}")).join(" | "));
    }
    md
}

// ── Worker completion cost (design doc M0-1) ────────────────────────────────

/// Index one frozen dataset through the vector worker's write-then-complete path,
/// twice, and report documents per second (embedding excluded) for each:
///
/// - **complete at once:** each entry completed right after its vectors are
///   written, as the worker did before M0-1;
/// - **flush then complete:** entries completed in batches of
///   `COMPLETION_BATCH` after one flush of the vector namespaces, as it does now.
///
/// Sequential, one entry at a time, so the difference is the flushes alone.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn vector_bench_worker_completion() {
    use crate::vector_kv::{complete_embed, enqueue_embed, get_queue_entry, make_vector_writes_durable};
    const BATCH: usize = 256; // the worker's COMPLETION_BATCH
    let dataset = env_or("MINNAL_BENCH_DATASET", "scifact");
    let bench_root = PathBuf::from(env_or("MINNAL_BENCH_ROOT", "../work/bench"));
    let model = env_or("MINNAL_BENCH_MODEL", "gemma");
    let frozen_dir = bench_root.join(&dataset).join(&model);
    assert!(
        frozen_dir.join("manifest.json").exists(),
        "run vector_bench for {dataset} first to dump its embeddings"
    );
    let config = SemanticSearchConfig {
        model_name: model.clone(),
        ..SemanticSearchConfig::default()
    };
    let dataset_dir = PathBuf::from(env_or("MINNAL_BEIR_ROOT", "../work/beir")).join(&dataset);
    let frozen = frozen::load_or_dump(
        &frozen_dir,
        &frozen::DumpSpec {
            dataset: &dataset,
            dataset_dir: &dataset_dir,
            split: &env_or("MINNAL_BEIR_SPLIT", "test"),
            model: &model,
            dim: config.embedding_dim,
            window_size: config.window_size,
            sliding_size: config.sliding_size,
            embed_url: &config.embedding_service_url,
        },
    )
    .await;
    let raw = read_clusters_from_file(&format!("../service/embedding_support/{model}/clusters.json")).unwrap();
    let index = ClusterIndex::from_clusters(raw.into_iter().map(|(id, c)| (id, Cluster::new(id, c))).collect());
    let ns = NS.to_string();

    let mut report = String::new();
    for flush_first in [false, true] {
        let tmp = tempfile::TempDir::new().unwrap();
        let db = AsyncDb::open_with_config(tmp.path().to_owned(), DbConfig::default()).await.unwrap();
        let parent = db.namespace(ns.clone()).await.unwrap();
        for d in 0..frozen.n_docs() {
            let id = (d as u64).to_be_bytes();
            parent.put(id.to_vec(), b"{}".to_vec()).await.unwrap();
            enqueue_embed(&db, NS, &id, "text").await.unwrap();
        }
        let t = Instant::now();
        let mut pending = Vec::new();
        for d in 0..frozen.n_docs() {
            let id = (d as u64).to_be_bytes();
            let entry = get_queue_entry(&db, NS, &id).await.unwrap().unwrap();
            let vis = index_embeddings(&config, &index, frozen.dense(d), &frozen.doc_chunks(d)).unwrap();
            upsert_vectors(&db, NS, &id, "text", &vis).await.unwrap();
            if flush_first {
                pending.push(entry);
                if pending.len() >= BATCH || d + 1 == frozen.n_docs() {
                    make_vector_writes_durable(&db, std::slice::from_ref(&ns)).await.unwrap();
                    for e in pending.drain(..) {
                        complete_embed(&db, &e).await.unwrap();
                    }
                }
            } else {
                complete_embed(&db, &entry).await.unwrap();
            }
        }
        let secs = t.elapsed().as_secs_f64();
        let name = if flush_first { "flush then complete" } else { "complete at once" };
        let _ = writeln!(report, "  {name:>20}: {:.0} docs/s ({secs:.1} s)", frozen.n_docs() as f64 / secs);
        db.shutdown().await.unwrap();
    }
    eprintln!("\n=== worker completion cost: {dataset}, {} docs ===\n{report}", frozen.n_docs());
}

// ── How much rotation can help (design doc M1 follow-up) ─────────────────────

/// Whether a dataset's residuals (chunk − nearest centroid) are evenly spread over
/// dimensions, measured before and after rotation on a sample of chunks:
///
/// - share of total variance in the top 10% of dimensions (10% = perfectly even),
///   and the largest dimension's variance against the mean;
/// - mean `⟨ō, o⟩` of the 1-bit code (`ō = sign(o)/√D`): how much of each residual
///   one bit per dimension captures (≈0.80 for evenly spread vectors; RaBitQ's
///   error grows as it falls);
/// - mean |correlation| between the sign bits of random dimension pairs (0 =
///   every bit carries new information).
///
/// Read-only over the frozen embeddings; `MINNAL_BENCH_DATASET` picks the dataset.
#[test]
#[ignore]
fn vector_bench_residual_spread() {
    const SAMPLE: usize = 5000;
    const PAIRS: usize = 2000;
    let dataset = env_or("MINNAL_BENCH_DATASET", "scifact");
    let model = env_or("MINNAL_BENCH_MODEL", "gemma");
    let bench_root = PathBuf::from(env_or("MINNAL_BENCH_ROOT", "../work/bench"));
    let frozen_dir = bench_root.join(&dataset).join(&model);
    assert!(frozen_dir.join("manifest.json").exists(), "run vector_bench for {dataset} first");
    let config = SemanticSearchConfig::default();
    let dataset_dir = PathBuf::from(env_or("MINNAL_BEIR_ROOT", "../work/beir")).join(&dataset);
    let frozen = tokio::runtime::Runtime::new().unwrap().block_on(frozen::load_or_dump(
        &frozen_dir,
        &frozen::DumpSpec {
            dataset: &dataset,
            dataset_dir: &dataset_dir,
            split: &env_or("MINNAL_BEIR_SPLIT", "test"),
            model: &model,
            dim: config.embedding_dim,
            window_size: config.window_size,
            sliding_size: config.sliding_size,
            embed_url: &config.embedding_service_url,
        },
    ));
    let raw = read_clusters_from_file(&format!("../service/embedding_support/{model}/clusters.json")).unwrap();
    let index = ClusterIndex::from_clusters(raw.into_iter().map(|(id, c)| (id, Cluster::new(id, c))).collect());
    let dim = frozen.dim();
    let mut rng = SEED;
    let rows: Vec<usize> = (0..SAMPLE).map(|_| (splitmix(&mut rng) % frozen.n_chunks() as u64) as usize).collect();
    let pairs: Vec<(usize, usize)> = (0..PAIRS)
        .map(|_| {
            let a = (splitmix(&mut rng) % dim as u64) as usize;
            let b = (a + 1 + (splitmix(&mut rng) % (dim as u64 - 1)) as usize) % dim;
            (a, b)
        })
        .collect();

    let mut report = String::new();
    for rotated in [false, true] {
        // Unit residuals o, one row per sampled chunk.
        let units: Vec<Vec<f32>> = rows
            .iter()
            .map(|&c| {
                let x = frozen.chunk(c);
                let id = find_closest_cluster_id(&index.clusters, x);
                let mut r: Vec<f32> = x.iter().zip(&index.clusters[&id].centroid).map(|(a, b)| a - b).collect();
                if rotated {
                    r = index.rotate(&r);
                }
                let n = r.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
                r.iter().map(|v| v / n).collect()
            })
            .collect();
        let mut var = vec![0f64; dim];
        for o in &units {
            for (v, &x) in var.iter_mut().zip(o) {
                *v += (x as f64).powi(2);
            }
        }
        let total: f64 = var.iter().sum();
        let mean = total / dim as f64;
        let mut sorted = var.clone();
        sorted.sort_by(|a, b| b.total_cmp(a));
        let top10 = sorted.iter().take(dim / 10).sum::<f64>() / total;
        let max_over_mean = sorted[0] / mean;
        let alignment = units
            .iter()
            .map(|o| o.iter().map(|v| v.abs() as f64).sum::<f64>() / (dim as f64).sqrt())
            .sum::<f64>()
            / units.len() as f64;
        // Sign-bit correlation over random dimension pairs (bits as ±1).
        let corr = pairs
            .iter()
            .map(|&(a, b)| {
                let n = units.len() as f64;
                let (mut sa, mut sb, mut sab) = (0f64, 0f64, 0f64);
                for o in &units {
                    let (x, y) = (o[a].signum() as f64, o[b].signum() as f64);
                    sa += x;
                    sb += y;
                    sab += x * y;
                }
                let (ma, mb) = (sa / n, sb / n);
                let cov = sab / n - ma * mb;
                let denom = ((1.0 - ma * ma) * (1.0 - mb * mb)).sqrt();
                if denom > 0.0 { (cov / denom).abs() } else { 0.0 }
            })
            .sum::<f64>()
            / pairs.len() as f64;
        let _ = writeln!(
            report,
            "  {:>10}: top 10% of dims hold {:.1}% of variance; max dim {:.1}x the mean; mean <ō,o> {:.4}; mean |sign-bit corr| {:.4}",
            if rotated { "rotated" } else { "unrotated" },
            top10 * 100.0,
            max_over_mean,
            alignment,
            corr
        );
    }
    // Reference: what perfectly evenly spread (random) unit vectors give.
    let _ = writeln!(
        report,
        "  reference (random unit vectors): 10.0% (plus sampling noise); <ō,o> ≈ {:.4}; |corr| ≈ {:.4} at {SAMPLE} samples",
        (2.0f64 / std::f64::consts::PI).sqrt(),
        (2.0 / (std::f64::consts::PI * SAMPLE as f64)).sqrt()
    );
    eprintln!("\n=== residual spread: {dataset} ({model}), {SAMPLE} chunks, {PAIRS} dimension pairs ===\n{report}");
}

// ── Comparison ───────────────────────────────────────────────────────────────

/// Mean of `new − base` over queries, with a paired bootstrap 95% interval.
fn paired_delta(base: &[f64], new: &[f64]) -> (f64, f64, f64) {
    let d: Vec<f64> = base.iter().zip(new).map(|(b, n)| n - b).collect();
    let n = d.len();
    let mean = d.iter().sum::<f64>() / n as f64;
    let mut rng = SEED;
    let mut means: Vec<f64> = (0..10_000)
        .map(|_| (0..n).map(|_| d[(splitmix(&mut rng) % n as u64) as usize]).sum::<f64>() / n as f64)
        .collect();
    means.sort_by(f64::total_cmp);
    (mean, percentile(&means, 0.025), percentile(&means, 0.975))
}

/// Paired per-query deltas of `n` against `b`: nDCG@10 and ANN R@10 as
/// (mean, CI low, CI high), and nDCG@10 wins / losses / ties.
struct Paired {
    ndcg: (f64, f64, f64),
    ann: (f64, f64, f64),
    wlt: (usize, usize, usize),
}

fn paired(b: &Setting, n: &Setting) -> Paired {
    let col = |s: &Setting, f: fn(&QueryResultRow) -> f64| s.per_query.iter().map(f).collect::<Vec<f64>>();
    let (mut w, mut l, mut t) = (0, 0, 0);
    for (x, y) in b.per_query.iter().zip(&n.per_query) {
        match y.ndcg10 - x.ndcg10 {
            d if d > 1e-9 => w += 1,
            d if d < -1e-9 => l += 1,
            _ => t += 1,
        }
    }
    Paired {
        ndcg: paired_delta(&col(b, |r| r.ndcg10), &col(n, |r| r.ndcg10)),
        ann: paired_delta(&col(b, |r| r.ann_r10), &col(n, |r| r.ann_r10)),
        wlt: (w, l, t),
    }
}

/// `f` of the curve (sorted by mean entries scanned) at `x` entries.
fn interpolate(curve: &[&Setting], x: f64, f: fn(&Setting) -> f64) -> f64 {
    let e = |s: &Setting| s.mean.entries_scanned as f64;
    if x <= e(curve[0]) {
        return f(curve[0]);
    }
    for w in curve.windows(2) {
        let (a, b) = (w[0], w[1]);
        if x <= e(b) {
            let span = e(b) - e(a);
            let t = if span > 0.0 { (x - e(a)) / span } else { 1.0 };
            return f(a) + t * (f(b) - f(a));
        }
    }
    f(curve[curve.len() - 1])
}

/// Gate `n` against the base row `b`, paired per query: write the nDCG table by
/// cutoff and push the gate checks, named after `n`. A budget at least the
/// namespace's `entries` scans it in full, buying recall with time on purpose,
/// so its latency is reported but not gated.
fn gate_pair(md: &mut String, gate: &mut Vec<(String, bool)>, b: &Setting, n: &Setting, entries: usize) {
    let who = n.name();
    let p = paired(b, n);
    let within = |old: f64, new: f64| new <= (old * 1.05).max(old + 0.3);
    let _ = writeln!(md, "\nnDCG by cutoff, {who} against base {}, paired per query:\n", b.name());
    let _ = writeln!(md, "| Cutoff | Base | New | Δ [95% CI] | W / L / T |\n|---:|---:|---:|---|---|");
    let mut worst = f64::INFINITY;
    for (i, k) in NDCG_CUTOFFS.iter().enumerate() {
        let (x, y): (Vec<f64>, Vec<f64>) = b.per_query.iter().zip(&n.per_query).map(|(x, y)| (x.ndcg_at[i], y.ndcg_at[i])).unzip();
        let (d, lo, hi) = paired_delta(&x, &y);
        let (w, l) = x.iter().zip(&y).fold((0, 0), |(w, l), (a, b)| match b - a {
            v if v > 1e-9 => (w + 1, l),
            v if v < -1e-9 => (w, l + 1),
            _ => (w, l),
        });
        let _ = writeln!(
            md,
            "| @{k} | {:.4} | {:.4} | {d:+.4} [{lo:+.4}, {hi:+.4}] | {w} / {l} / {} |",
            b.mean.ndcg_at[i],
            n.mean.ndcg_at[i],
            x.len() - w - l
        );
        worst = worst.min(d);
    }
    let _ = writeln!(
        md,
        "\nEntries {} → {}, postings {:.1} → {:.1}, ANN R@10 {:.4} → {:.4}, Pass-1 recall {:.4} → {:.4}, p50 {:.2} → {:.2} ms, p95 {:.2} → {:.2} ms.",
        b.mean.entries_scanned,
        n.mean.entries_scanned,
        b.mean.probes,
        n.mean.probes,
        b.mean.ann_r10,
        n.mean.ann_r10,
        b.mean.pass1_recall,
        n.mean.pass1_recall,
        b.latency_p50_ms,
        n.latency_p50_ms,
        b.latency_p95_ms,
        n.latency_p95_ms
    );
    gate.push((
        format!("{who}: nDCG@10 Δ ≥ −0.005 and CI upper bound ≥ 0"),
        p.ndcg.0 >= -0.005 && p.ndcg.2 >= 0.0,
    ));
    gate.push((format!("{who}: ANN recall@10 Δ ≥ −0.005"), p.ann.0 >= -0.005));
    gate.push((format!("{who}: nDCG@{{10..100}} Δ ≥ −0.002 at every cutoff"), worst >= -0.002));
    if n.budget_entries.is_some_and(|budget| budget >= entries as u64) {
        let _ = writeln!(md, "\nThe budget covers all {entries} entries: a full scan, so latency is not gated.");
        gate.push((format!("{who}: ANN recall@10 not below the base (full scan)"), p.ann.0 >= 0.0));
        return;
    }
    gate.push((
        format!("{who}: p50 latency within +5% (or +0.3 ms)"),
        within(b.latency_p50_ms, n.latency_p50_ms),
    ));
    gate.push((
        format!("{who}: p95 latency within +5% (or +0.3 ms)"),
        within(b.latency_p95_ms, n.latency_p95_ms),
    ));
}

#[test]
#[ignore]
fn vector_bench_compare() {
    let load = |var: &str| -> BenchResult {
        let path = std::env::var(var).unwrap_or_else(|_| panic!("set {var} to a vector_bench result .json"));
        serde_json::from_slice(&std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))).unwrap()
    };
    let (base, new) = (load("MINNAL_BENCH_BASE"), load("MINNAL_BENCH_NEW"));
    assert_eq!(base.queries, new.queries, "the two runs must use the same queries");
    let mut md = String::new();
    let _ = writeln!(
        md,
        "# `{}` ({}) vs `{}` ({}) — {} {}\n",
        new.label, new.git, base.label, base.git, new.dataset, new.order
    );
    let _ = writeln!(
        md,
        "| Probing | Δ nDCG@10 [95% CI] | W / L / T | Δ ANN R@10 [95% CI] | Δ Pass-1 recall | Entries scanned | p50 ms | p95 ms |"
    );
    let _ = writeln!(md, "|---|---|---|---|---:|---|---|---|");
    for b in &base.settings {
        let Some(n) = new.settings.iter().find(|s| s.same_as(b)) else {
            continue;
        };
        let p = paired(b, n);
        let _ = writeln!(
            md,
            "| {} | {:+.4} [{:+.4}, {:+.4}] | {} / {} / {} | {:+.4} [{:+.4}, {:+.4}] | {:+.4} | {} → {} | {:.2} → {:.2} | {:.2} → {:.2} |",
            b.name(),
            p.ndcg.0,
            p.ndcg.1,
            p.ndcg.2,
            p.wlt.0,
            p.wlt.1,
            p.wlt.2,
            p.ann.0,
            p.ann.1,
            p.ann.2,
            n.mean.pass1_recall - b.mean.pass1_recall,
            b.mean.entries_scanned,
            n.mean.entries_scanned,
            b.latency_p50_ms,
            n.latency_p50_ms,
            b.latency_p95_ms,
            n.latency_p95_ms
        );
    }

    // Budget rows against the base's fixed-probe curve, interpolated at the
    // same mean entries scanned (design doc M2d: the curves should overlay).
    let curve: Vec<&Setting> = {
        let mut c: Vec<&Setting> = base.settings.iter().filter(|s| s.budget_entries.is_none()).collect();
        c.sort_by_key(|s| s.mean.entries_scanned);
        c
    };
    let budget_rows: Vec<&Setting> = new.settings.iter().filter(|s| s.budget_entries.is_some()).collect();
    if !budget_rows.is_empty() && curve.len() >= 2 {
        let _ = writeln!(
            md,
            "\nBudget rows against the base's fixed-probe curve at equal mean entries (linear interpolation; outside the curve's range, its nearest end):\n"
        );
        let _ = writeln!(
            md,
            "| Budget | Entries scanned | Postings probed (mean) | nDCG@10 Δ | ANN R@10 Δ | Pass-1 recall Δ | p50 ms Δ | p95 ms Δ |\n|---:|---:|---:|---:|---:|---:|---:|---:|"
        );
        for n in &budget_rows {
            let x = n.mean.entries_scanned as f64;
            let at = |f: fn(&Setting) -> f64| interpolate(&curve, x, f);
            let _ = writeln!(
                md,
                "| {} | {} | {:.1} | {:+.4} | {:+.4} | {:+.4} | {:+.2} | {:+.2} |",
                n.budget_entries.unwrap(),
                n.mean.entries_scanned,
                n.mean.probes,
                n.mean.ndcg10 - at(|s| s.mean.ndcg10),
                n.mean.ann_r10 - at(|s| s.mean.ann_r10),
                n.mean.pass1_recall - at(|s| s.mean.pass1_recall),
                n.latency_p50_ms - at(|s| s.latency_p50_ms),
                n.latency_p95_ms - at(|s| s.latency_p95_ms)
            );
        }
    }

    // Gates against the base at the fixed production probe count: the new run
    // at the same count, and (when it has one) at the default budget.
    let mut gate = Vec::new();
    if let Some(b) = base
        .settings
        .iter()
        .find(|s| s.budget_entries.is_none() && s.n_probes == PRODUCTION_NPROBES)
    {
        let candidates = [
            new.settings.iter().find(|s| s.same_as(b)),
            new.settings.iter().find(|s| s.budget_entries == Some(DEFAULT_PROBE.budget_entries)),
        ];
        for n in candidates.into_iter().flatten() {
            gate_pair(&mut md, &mut gate, b, n, new.partition.entries);
        }
    }
    let footprint = |r: &BenchResult| (r.footprint.sparse_bytes + r.footprint.sparse_meta_bytes + r.footprint.dense_bytes) as f64;
    gate.push(("footprint within +5%".to_string(), footprint(&new) <= footprint(&base) * 1.05));
    let _ = writeln!(
        md,
        "\nIndexing: {:.0} → {:.0} docs/s. Estimator RMSE: Pass 1 {:.5} → {:.5}, Pass 2 {:.5} → {:.5}.\n",
        base.docs_per_second,
        new.docs_per_second,
        base.estimator_pass1.rmse,
        new.estimator_pass1.rmse,
        base.estimator_pass2.rmse,
        new.estimator_pass2.rmse
    );
    let _ = writeln!(md, "Gate against the base at n_probes {PRODUCTION_NPROBES}:\n");
    for (name, ok) in &gate {
        let _ = writeln!(md, "- {} {name}", if *ok { "PASS" } else { "FAIL" });
    }
    eprintln!("\n{md}");
    if let Ok(dir) = std::env::var("MINNAL_BENCH_NEW").map(PathBuf::from)
        && let Some(parent) = dir.parent()
    {
        let out = parent.join(format!("compare_{}_vs_{}-{}.md", new.dataset, base.label, new.order));
        std::fs::write(&out, &md).unwrap();
        eprintln!("  written to {}", out.display());
    }
}
