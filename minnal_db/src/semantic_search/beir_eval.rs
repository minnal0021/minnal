//! BEIR relevance eval for the final-ranking fusion modes.
//!
//! `real_recall` (in `vector_kv.rs`) measures ANN *approximation* — overlap with the
//! pipeline's own exhaustive-probe ranking — so it cannot say whether one ranking is
//! more *relevant* than another: any change of order reads as a recall loss there.
//! This harness answers the relevance question with human judgements: it indexes a
//! [BEIR](https://github.com/beir-cellar/beir) corpus through the real embedding
//! service and scores every [`RankFusion`] mode against the dataset's qrels with
//! nDCG@10, MRR@10, and Recall@100.
//!
//! Each query runs **one** live `search` in `Dense` mode with
//! `top_k = first_pass_sparse_search_top_k`, which returns every candidate with both
//! pass scores. Every variant is then derived offline through the production
//! [`fuse`](super::service::fuse), so the numbers come from the shipped code — a
//! sanity check asserts a live `Rrf` search matches the offline fusion. Variants:
//!
//! - `dense` (the pre-fusion behaviour) and `sparse` (MaxSim only — if it beats
//!   `dense`, ColBERT's signal was being thrown away);
//! - `rrf` over the full candidate set for `k ∈ {1 … 200}` × `w ∈ {0.1 … 0.9}`;
//! - `zscore` for `w ∈ {0.1 … 0.9}` (keeps score gaps that RRF discards);
//! - `rrf-dense100` — RRF re-order restricted to the dense top-100 (can only
//!   reorder, never promote a doc the dense pass ranked lower).
//!
//! `p` is a two-sided paired randomization test of per-query nDCG@10 vs `dense`.
//! Tune on a training split (`MINNAL_BEIR_SPLIT=train`/`dev`) and confirm on `test`
//! — picking the best grid cell on `test` alone overstates its gain.
//!
//! Setup (from the repo root), then run from anywhere:
//!
//! ```sh
//! service/scripts/fetch_beir.sh scifact
//! MINNAL_EMBED_URL=http://localhost:8001 \
//!   cargo test -p minnal_db --all-features --release --lib beir_rank_fusion_eval -- --ignored --nocapture
//! ```
//!
//! Env knobs (all optional): `MINNAL_BEIR_DATASET` (default `scifact`; `nfcorpus`,
//! `fiqa`, … — anything with `corpus.jsonl`/`queries.jsonl`/`qrels/<split>.tsv`),
//! `MINNAL_BEIR_ROOT` (default `../work/beir`), `MINNAL_BEIR_SPLIT` (default `test`),
//! `MINNAL_BEIR_MODEL` (cluster set, default `gemma` — must match the model the
//! service serves), `MINNAL_EMBED_URL`, `MINNAL_BEIR_MAX_QUERIES`,
//! `MINNAL_BEIR_QUERY_CHUNKING` (`window` — production 4-word query chunks, the
//! default; `whole` — one sparse chunk = the whole-query embedding; `both`), and
//! `MINNAL_BEIR_DB` — a directory to keep the built index in so later runs skip
//! re-embedding the corpus. A results table is also written to
//! `{root}/{dataset}/fusion_eval_{model}_{split}[_q<chunking>].{md,tsv}`.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use crate::AsyncDb;
use crate::semantic_search::ClusterIndex;
use crate::semantic_search::cluster::{Cluster, read_clusters_from_file};
use crate::semantic_search::service::{
    Candidate, QueryEmbeddings, RankFusion, RankingParams, SearchOptions, SemanticSearchConfig, embed_query, fuse, search,
};
use crate::vector_kv::DbVectorStore;
use crate::vector_kv::eval_indexing::index_texts;

const NS: &str = "beir";
const INDEX_CONCURRENCY: usize = 8;
/// Default production n_probes, plus exhaustive probing (appended at runtime).
const NPROBES: usize = 32;
const NDCG_K: usize = 10;
const RECALL_K: usize = 100;
/// Number of queries the live-vs-offline fusion sanity check compares.
const SANITY_QUERIES: usize = 20;

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn read_jsonl(path: &Path) -> Vec<serde_json::Value> {
    let f = std::fs::File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    std::io::BufReader::new(f)
        .lines()
        .map_while(Result::ok)
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(&l).unwrap_or_else(|e| panic!("bad JSON line in {}: {e}", path.display())))
        .collect()
}

fn str_field<'a>(v: &'a serde_json::Value, key: &str) -> &'a str {
    v.get(key).and_then(|x| x.as_str()).unwrap_or("").trim()
}

/// `query-id → (corpus-id → graded relevance)`, keeping only positive judgements.
fn read_qrels(path: &Path) -> HashMap<String, HashMap<String, u32>> {
    let f = std::fs::File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let mut qrels: HashMap<String, HashMap<String, u32>> = HashMap::new();
    for line in std::io::BufReader::new(f).lines().map_while(Result::ok).skip(1) {
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 3 {
            continue;
        }
        let score: i64 = cols[2].trim().parse().unwrap_or(0);
        if score > 0 {
            qrels.entry(cols[0].to_string()).or_default().insert(cols[1].to_string(), score as u32);
        }
    }
    qrels
}

/// Per-query relevance metrics for one ranked list of corpus ids.
#[derive(Default, Clone, Copy)]
struct Metrics {
    ndcg: f64,
    mrr: f64,
    recall: f64,
}

fn score_ranking(ranked: &[&str], rels: &HashMap<String, u32>) -> Metrics {
    let dcg: f64 = ranked
        .iter()
        .take(NDCG_K)
        .enumerate()
        .map(|(i, id)| rels.get(*id).copied().unwrap_or(0) as f64 / ((i + 2) as f64).log2())
        .sum();
    let mut ideal: Vec<u32> = rels.values().copied().collect();
    ideal.sort_unstable_by(|a, b| b.cmp(a));
    let idcg: f64 = ideal
        .iter()
        .take(NDCG_K)
        .enumerate()
        .map(|(i, &r)| r as f64 / ((i + 2) as f64).log2())
        .sum();
    let mrr = ranked
        .iter()
        .take(NDCG_K)
        .position(|id| rels.contains_key(*id))
        .map_or(0.0, |p| 1.0 / (p + 1) as f64);
    let hits = ranked.iter().take(RECALL_K).filter(|id| rels.contains_key(**id)).count();
    Metrics {
        ndcg: if idcg > 0.0 { dcg / idcg } else { 0.0 },
        mrr,
        recall: hits as f64 / rels.len() as f64,
    }
}

/// A ranking to evaluate: fusion over the full candidate set, or RRF re-ordering
/// only the dense top-`depth`.
enum Variant {
    Full(RankingParams),
    DenseTopRrf { depth: usize, params: RankingParams },
}

impl Variant {
    fn name(&self) -> String {
        match self {
            Variant::Full(p) => match p.mode {
                RankFusion::Dense => "dense".into(),
                RankFusion::Sparse => "sparse".into(),
                RankFusion::Rrf => format!("rrf k={} w={}", p.rrf_k, p.sparse_weight),
                RankFusion::Zscore => format!("zscore w={}", p.sparse_weight),
            },
            Variant::DenseTopRrf { depth, params } => format!("rrf-dense{depth} k={} w={}", params.rrf_k, params.sparse_weight),
        }
    }

    fn rank(&self, cands: &[Candidate]) -> Vec<u64> {
        let ranked = match self {
            Variant::Full(p) => fuse(cands.to_vec(), p, RECALL_K),
            Variant::DenseTopRrf { depth, params } => {
                let dense = fuse(cands.to_vec(), &RankingParams::default(), *depth);
                fuse(dense.into_iter().map(to_candidate).collect(), params, RECALL_K)
            }
        };
        ranked.iter().map(|r| doc_u64(&r.document_id)).collect()
    }
}

fn params(mode: RankFusion, rrf_k: f32, sparse_weight: f32) -> RankingParams {
    RankingParams { mode, rrf_k, sparse_weight }
}

/// RRF `k` grid and the `sparse_weight` grid shared by `rrf` and `zscore`.
const RRF_KS: [f32; 8] = [1.0, 5.0, 10.0, 20.0, 30.0, 60.0, 100.0, 200.0];
const WEIGHTS: [f32; 9] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
/// Sign-flip permutations for the paired significance test vs `dense`.
const PERMUTATIONS: usize = 10_000;

/// Two-sided paired randomization (sign-flip) test on per-query differences:
/// the fraction of random sign assignments whose |mean| is ≥ the observed |mean|.
/// Deterministic (fixed-seed xorshift) so reruns report the same p.
fn paired_p_value(diffs: &[f64]) -> f64 {
    let n = diffs.len() as f64;
    let observed = (diffs.iter().sum::<f64>() / n).abs();
    if observed == 0.0 {
        return 1.0;
    }
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut at_least = 0usize;
    for _ in 0..PERMUTATIONS {
        let mut sum = 0.0;
        for &d in diffs {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            sum += if state & 1 == 0 { d } else { -d };
        }
        if (sum / n).abs() >= observed - 1e-12 {
            at_least += 1;
        }
    }
    (at_least + 1) as f64 / (PERMUTATIONS + 1) as f64
}

fn variants() -> Vec<Variant> {
    let mut v = vec![
        Variant::Full(params(RankFusion::Dense, 60.0, 0.5)),
        Variant::Full(params(RankFusion::Sparse, 60.0, 0.5)),
    ];
    for k in RRF_KS {
        for w in WEIGHTS {
            v.push(Variant::Full(params(RankFusion::Rrf, k, w)));
        }
    }
    for w in WEIGHTS {
        v.push(Variant::Full(params(RankFusion::Zscore, 60.0, w)));
    }
    for w in [0.3, 0.5, 0.7] {
        v.push(Variant::DenseTopRrf {
            depth: RECALL_K,
            params: params(RankFusion::Rrf, 60.0, w),
        });
    }
    v
}

/// How the Pass-1 (sparse) query chunks are formed — an eval-only experiment knob
/// (`MINNAL_BEIR_QUERY_CHUNKING`). Production `embed_query` always uses `Window`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum QueryChunking {
    /// Production: 4-word sliding windows (`chunk_query`).
    Window,
    /// A single sparse chunk = the whole-query embedding (the Pass-2 dense vector,
    /// which is the same `/embedding/query` embedding of the full query text).
    Whole,
    /// The whole-query embedding plus the window chunks.
    Both,
}

impl QueryChunking {
    fn from_env() -> Self {
        match std::env::var("MINNAL_BEIR_QUERY_CHUNKING").as_deref() {
            Err(_) | Ok("window") => Self::Window,
            Ok("whole") => Self::Whole,
            Ok("both") => Self::Both,
            Ok(other) => panic!("MINNAL_BEIR_QUERY_CHUNKING must be window | whole | both, got {other:?}"),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Window => "window",
            Self::Whole => "whole",
            Self::Both => "both",
        }
    }

    /// Output-file suffix; empty for the production default so existing paths are unchanged.
    fn file_suffix(self) -> String {
        match self {
            Self::Window => String::new(),
            other => format!("_q{}", other.name()),
        }
    }

    fn apply(self, mut qe: QueryEmbeddings) -> QueryEmbeddings {
        match self {
            Self::Window => {}
            Self::Whole => qe.sparse = vec![qe.dense.clone()],
            Self::Both => qe.sparse.insert(0, qe.dense.clone()),
        }
        qe
    }
}

fn to_candidate(r: crate::semantic_search::index::vector_index::QueryResult) -> Candidate {
    Candidate {
        document_id: r.document_id,
        sparse_score: r.sparse_score,
        dense_score: r.dense_score,
        error_bound: r.error_bound,
    }
}

fn doc_u64(id: &[u8]) -> u64 {
    u64::from_be_bytes(id.try_into().expect("beir doc ids are u64 BE"))
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn beir_rank_fusion_eval() {
    let dataset = env_or("MINNAL_BEIR_DATASET", "scifact");
    let root = PathBuf::from(env_or("MINNAL_BEIR_ROOT", "../work/beir"));
    let split = env_or("MINNAL_BEIR_SPLIT", "test");
    let model = env_or("MINNAL_BEIR_MODEL", "gemma");
    let embed_url = std::env::var("MINNAL_EMBED_URL").unwrap_or_else(|_| SemanticSearchConfig::default().embedding_service_url);
    let max_queries: Option<usize> = std::env::var("MINNAL_BEIR_MAX_QUERIES").ok().and_then(|v| v.parse().ok());
    let query_chunking = QueryChunking::from_env();
    let dir = root.join(&dataset);
    if !dir.join("corpus.jsonl").exists() {
        eprintln!(
            "SKIP beir_rank_fusion_eval: {} missing — run service/scripts/fetch_beir.sh {dataset}",
            dir.join("corpus.jsonl").display()
        );
        return;
    }

    // ── Load corpus, queries, qrels ──
    let corpus = read_jsonl(&dir.join("corpus.jsonl"));
    let corpus_ids: Vec<String> = corpus.iter().map(|d| str_field(d, "_id").to_string()).collect();
    let qrels = read_qrels(&dir.join("qrels").join(format!("{split}.tsv")));
    let mut queries: Vec<(String, String)> = read_jsonl(&dir.join("queries.jsonl"))
        .iter()
        .map(|q| (str_field(q, "_id").to_string(), str_field(q, "text").to_string()))
        .filter(|(id, text)| qrels.contains_key(id) && !text.is_empty())
        .collect();
    queries.sort_by(|a, b| a.0.cmp(&b.0));
    if let Some(m) = max_queries {
        queries.truncate(m);
    }

    let cluster_path = format!("../service/embedding_support/{model}/clusters.json");
    let raw = read_clusters_from_file(&cluster_path).unwrap_or_else(|e| panic!("load {cluster_path}: {e}"));
    let n_clusters = raw.len();
    let index = Arc::new(ClusterIndex::from_clusters(
        raw.into_iter().map(|(id, c)| (id, Cluster::new(id, c))).collect(),
    ));
    let base_config = Arc::new(SemanticSearchConfig {
        embedding_service_url: embed_url.clone(),
        model_name: model.clone(),
        ..SemanticSearchConfig::default()
    });

    eprintln!("\n=== BEIR rank-fusion eval: {dataset} ({split}) ===");
    eprintln!(
        "  service {embed_url}  |  model {model}  |  {} docs, {} judged queries, {n_clusters} clusters",
        corpus.len(),
        queries.len()
    );

    // ── Index (or reuse a persisted index) ──
    let _tmp;
    let db_dir = match std::env::var("MINNAL_BEIR_DB") {
        Ok(d) => PathBuf::from(d),
        Err(_) => {
            _tmp = tempfile::TempDir::new().unwrap();
            _tmp.path().to_owned()
        }
    };
    // The marker pins everything the stored vectors depend on.
    let marker = db_dir.join(format!(
        "indexed_{dataset}_{model}_w{}_s{}_b{}.done",
        base_config.window_size, base_config.sliding_size, base_config.number_of_bits_for_dense_quantisation
    ));
    std::fs::create_dir_all(&db_dir).unwrap();
    let db = Arc::new(AsyncDb::open_with_config(db_dir.clone(), crate::support::test_db_config()).await.unwrap());
    if marker.exists() {
        eprintln!("  reusing index in {}", db_dir.display());
    } else {
        let docs: Vec<(u64, String)> = corpus
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let (title, text) = (str_field(d, "title"), str_field(d, "text"));
                let body = if title.is_empty() {
                    text.to_string()
                } else {
                    format!("{title}. {text}")
                };
                (i as u64, body)
            })
            .collect();
        let t = Instant::now();
        let (ok, fail, first_err) = index_texts(&db, NS, &base_config, &index, docs, INDEX_CONCURRENCY).await;
        eprintln!("  indexed {ok} docs ({fail} failed) in {:.1}s", t.elapsed().as_secs_f64());
        if ok == 0 {
            eprintln!("SKIP beir_rank_fusion_eval: every embed failed — is the service up at {embed_url}? first error: {first_err:?}");
            db.shutdown().await.unwrap();
            return;
        }
        if fail == 0 {
            std::fs::write(&marker, format!("{ok}\n")).unwrap();
        }
    }
    let store = DbVectorStore::new(&db, NS).await.unwrap();

    // ── Embed queries ──
    let mut q_embs: Vec<(String, QueryEmbeddings)> = Vec::with_capacity(queries.len());
    for (qid, text) in &queries {
        match embed_query(&base_config, text).await {
            Ok(qe) if !qe.sparse.is_empty() && !qe.dense.is_empty() => q_embs.push((qid.clone(), query_chunking.apply(qe))),
            Ok(_) => {}
            Err(e) => eprintln!("  query {qid} embed failed (skipping): {e}"),
        }
    }
    assert!(!q_embs.is_empty(), "no queries could be embedded");

    let variants = variants();
    let mut tsv = String::from("n_probes\tvariant\tmode\trrf_k\tsparse_weight\tndcg10\tmrr10\trecall100\twins\tlosses\tp_vs_dense\n");
    let mut report = String::new();
    let _ = writeln!(
        report,
        "# BEIR rank-fusion eval — {dataset} ({split}), model {model}, query chunking `{}`\n",
        query_chunking.name()
    );
    let _ = writeln!(
        report,
        "{} docs, {} queries. nDCG@{NDCG_K} / MRR@{NDCG_K} / R@{RECALL_K}; W/L = queries with higher/lower nDCG@{NDCG_K} than `dense`.\n",
        corpus.len(),
        q_embs.len()
    );

    let mut probe_set = vec![NPROBES.min(n_clusters)];
    if n_clusters != probe_set[0] {
        probe_set.push(n_clusters);
    }
    for np in probe_set {
        let config = SemanticSearchConfig {
            n_probes: np,
            ..(*base_config).clone()
        };
        let all_candidates = SearchOptions {
            top_k: Some(config.first_pass_sparse_search_top_k),
            ranking: Some(RankingParams::default()),
        };

        // One live search per query → the full candidate set with both scores.
        let t = Instant::now();
        let mut per_query: Vec<(&str, Vec<Candidate>)> = Vec::with_capacity(q_embs.len());
        for (qid, qe) in &q_embs {
            let r = search(
                &config,
                NS,
                &index,
                &qe.sparse,
                &qe.dense,
                &store,
                None::<fn(&[u8]) -> bool>,
                all_candidates,
            )
            .await;
            per_query.push((qid.as_str(), r.into_iter().map(to_candidate).collect()));
        }
        let search_s = t.elapsed().as_secs_f64();
        assert!(
            per_query.iter().any(|(_, c)| !c.is_empty()),
            "every query returned zero candidates — the vector index is empty (delete {} and re-run to rebuild it)",
            db_dir.display()
        );

        // Sanity: the live Rrf path must agree with the offline fusion used below.
        let rrf = params(RankFusion::Rrf, 60.0, 0.5);
        for ((_, qe), (qid, cands)) in q_embs.iter().zip(&per_query).take(SANITY_QUERIES) {
            let opts = SearchOptions {
                top_k: Some(NDCG_K),
                ranking: Some(rrf),
            };
            let live: Vec<u64> = search(&config, NS, &index, &qe.sparse, &qe.dense, &store, None::<fn(&[u8]) -> bool>, opts)
                .await
                .iter()
                .map(|r| doc_u64(&r.document_id))
                .collect();
            let offline: Vec<u64> = fuse(cands.clone(), &rrf, NDCG_K).iter().map(|r| doc_u64(&r.document_id)).collect();
            assert_eq!(live, offline, "live Rrf search disagrees with offline fusion for query {qid}");
        }

        // Candidate-set recall: the ceiling any re-ranking of this set can reach.
        let cand_recall: f64 = per_query
            .iter()
            .map(|(qid, cands)| {
                let rels = &qrels[*qid];
                let ids: HashSet<&str> = cands.iter().map(|c| corpus_ids[doc_u64(&c.document_id) as usize].as_str()).collect();
                rels.keys().filter(|id| ids.contains(id.as_str())).count() as f64 / rels.len() as f64
            })
            .sum::<f64>()
            / per_query.len() as f64;
        let mean_cands = per_query.iter().map(|(_, c)| c.len()).sum::<usize>() as f64 / per_query.len() as f64;

        let per_variant: Vec<Vec<Metrics>> = variants
            .iter()
            .map(|v| {
                per_query
                    .iter()
                    .map(|(qid, cands)| {
                        let ranked = v.rank(cands);
                        let ids: Vec<&str> = ranked.iter().map(|&d| corpus_ids[d as usize].as_str()).collect();
                        score_ranking(&ids, &qrels[*qid])
                    })
                    .collect()
            })
            .collect();

        let nq = per_query.len() as f64;
        let mean = |ms: &[Metrics], f: fn(&Metrics) -> f64| ms.iter().map(f).sum::<f64>() / nq;
        let dense_ndcg = mean(&per_variant[0], |m| m.ndcg);
        let _ = writeln!(
            report,
            "## n_probes = {np}{}\n\nmean candidates {mean_cands:.0}, candidate-set recall {cand_recall:.3} (ceiling for any re-ranking), search {search_s:.1}s total\n",
            if np == n_clusters { " (exhaustive)" } else { "" }
        );
        let _ = writeln!(
            report,
            "| variant | nDCG@{NDCG_K} | Δ vs dense | MRR@{NDCG_K} | R@{RECALL_K} | W / L | p |"
        );
        let _ = writeln!(report, "|---|---:|---:|---:|---:|---:|---:|");
        for (v, ms) in variants.iter().zip(&per_variant) {
            let ndcg = mean(ms, |m| m.ndcg);
            let (w, l) = ms.iter().zip(&per_variant[0]).fold((0, 0), |(w, l), (a, d)| {
                if a.ndcg > d.ndcg + 1e-9 {
                    (w + 1, l)
                } else if a.ndcg + 1e-9 < d.ndcg {
                    (w, l + 1)
                } else {
                    (w, l)
                }
            });
            let diffs: Vec<f64> = ms.iter().zip(&per_variant[0]).map(|(a, d)| a.ndcg - d.ndcg).collect();
            let p = paired_p_value(&diffs);
            let (mrr, recall) = (mean(ms, |m| m.mrr), mean(ms, |m| m.recall));
            let _ = writeln!(
                report,
                "| {} | {ndcg:.4} | {:+.4} | {mrr:.4} | {recall:.4} | {w} / {l} | {p:.4} |",
                v.name(),
                ndcg - dense_ndcg,
            );
            let (mode, p_params) = match v {
                Variant::Full(p) => (format!("{:?}", p.mode).to_lowercase(), p),
                Variant::DenseTopRrf { params, .. } => ("rrf-dense100".to_string(), params),
            };
            let _ = writeln!(
                tsv,
                "{np}\t{}\t{mode}\t{}\t{}\t{ndcg:.6}\t{mrr:.6}\t{recall:.6}\t{w}\t{l}\t{p:.6}",
                v.name(),
                p_params.rrf_k,
                p_params.sparse_weight
            );
        }
        let _ = writeln!(report);
    }

    eprintln!("\n{report}");
    for (ext, body) in [("md", &report), ("tsv", &tsv)] {
        let out = dir.join(format!("fusion_eval_{model}_{split}{}.{ext}", query_chunking.file_suffix()));
        match std::fs::write(&out, body) {
            Ok(()) => eprintln!("  results written to {}", out.display()),
            Err(e) => eprintln!("  could not write results to {}: {e}", out.display()),
        }
    }
    db.shutdown().await.unwrap();
}

#[cfg(test)]
mod metric_tests {
    use super::*;

    fn rels(pairs: &[(&str, u32)]) -> HashMap<String, u32> {
        pairs.iter().map(|&(k, v)| (k.to_string(), v)).collect()
    }

    #[test]
    fn perfect_ranking_scores_one() {
        let m = score_ranking(&["a", "b", "x"], &rels(&[("a", 2), ("b", 1)]));
        assert!((m.ndcg - 1.0).abs() < 1e-12);
        assert_eq!(m.mrr, 1.0);
        assert_eq!(m.recall, 1.0);
    }

    #[test]
    fn misses_and_late_hits_are_discounted() {
        let m = score_ranking(&["x", "a"], &rels(&[("a", 1), ("b", 1)]));
        let idcg = 1.0 + 1.0 / 3f64.log2();
        assert!((m.ndcg - (1.0 / 3f64.log2()) / idcg).abs() < 1e-12);
        assert_eq!(m.mrr, 0.5);
        assert_eq!(m.recall, 0.5);
        let none = score_ranking(&["x", "y"], &rels(&[("a", 1)]));
        assert_eq!((none.ndcg, none.mrr, none.recall), (0.0, 0.0, 0.0));
    }

    #[test]
    fn query_chunking_modes() {
        let qe = || QueryEmbeddings {
            sparse: vec![vec![1.0], vec![2.0]],
            dense: vec![9.0],
        };
        assert_eq!(QueryChunking::Window.apply(qe()).sparse, vec![vec![1.0], vec![2.0]]);
        assert_eq!(QueryChunking::Whole.apply(qe()).sparse, vec![vec![9.0]]);
        assert_eq!(QueryChunking::Both.apply(qe()).sparse, vec![vec![9.0], vec![1.0], vec![2.0]]);
        assert_eq!(QueryChunking::Whole.apply(qe()).dense, vec![9.0]);
        assert_eq!(QueryChunking::Window.file_suffix(), "");
        assert_eq!(QueryChunking::Both.file_suffix(), "_qboth");
    }

    #[test]
    fn paired_p_value_separates_signal_from_noise() {
        assert_eq!(paired_p_value(&[0.0; 50]), 1.0);
        assert!(paired_p_value(&[0.1; 50]) < 0.001, "a consistent gain is significant");
        let noise: Vec<f64> = (0..50).map(|i| if i % 2 == 0 { 0.1 } else { -0.1 }).collect();
        assert!(paired_p_value(&noise) > 0.5, "balanced wins/losses are not");
    }

    #[test]
    fn dense_top_rrf_only_reorders_dense_top() {
        // Doc 3 is 1st by sparse but outside the dense top-2, so rrf-dense2 cannot surface it.
        let c = |id: u64, s: f32, d: f32| Candidate {
            document_id: id.to_be_bytes().to_vec(),
            sparse_score: s,
            dense_score: d,
            error_bound: 0.0,
        };
        let cands = vec![c(1, 1.0, 0.9), c(2, 2.0, 0.8), c(3, 9.0, 0.1)];
        let v = Variant::DenseTopRrf {
            depth: 2,
            params: params(RankFusion::Rrf, 60.0, 0.5),
        };
        let r = v.rank(&cands);
        assert_eq!(r.len(), 2);
        assert!(!r.contains(&3));
        let full = Variant::Full(params(RankFusion::Rrf, 60.0, 0.5)).rank(&cands);
        assert_eq!(full.len(), 3);
    }
}
