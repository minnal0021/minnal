//! BEIR relevance eval of the production semantic-search pipeline.
//!
//! Indexes a [BEIR](https://github.com/beir-cellar/beir) corpus through the real
//! embedding service and runs every judged query through `search()`, measuring:
//!
//! - **End to end:** nDCG@10 (primary), MRR@10, Recall@100 of the final
//!   (dense-ranked) list.
//! - **Pass 1:** candidate recall, the share of relevant docs among the
//!   `first_pass_sparse_search_top_k` candidates. The run is repeated at
//!   first-pass 100 as well as the production 1000, because a tight cut exposes
//!   Pass 1's own (MaxSim) ranking quality.
//! - **Cost:** clusters probed, query vectors, embedding time per query, and
//!   median search latency, at `n_probes` 32 (production) and 256 (exhaustive).
//!
//! It was used to choose whole-query Pass-1 embedding over chunked queries
//! (`semantic_search/query-embedding-report.md`); rerun it to evaluate any
//! change to indexing, centroids or search.
//!
//! Setup, then run from the crate root (`minnal_db/`):
//!
//! ```sh
//! service/scripts/fetch_beir.sh scifact nfcorpus arguana
//! MINNAL_EMBED_URL=http://localhost:8001 MINNAL_BEIR_DATASET=scifact MINNAL_BEIR_DB=$PWD/../work/beir/db_scifact \
//!   cargo test -p minnal_db --all-features --release --lib beir_eval -- --ignored --nocapture
//! ```
//!
//! Env knobs (all optional): `MINNAL_BEIR_DATASET` (default `scifact`),
//! `MINNAL_BEIR_ROOT` (default `../work/beir`), `MINNAL_BEIR_SPLIT` (default
//! `test`), `MINNAL_BEIR_MODEL` (cluster set, default `gemma`),
//! `MINNAL_EMBED_URL`, `MINNAL_BEIR_MAX_QUERIES`, and `MINNAL_BEIR_DB`, a
//! directory to keep the built index in so later runs skip re-embedding the
//! corpus. Results are also written to `{root}/{dataset}/beir_eval_{model}_{split}.{md,tsv}`.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use crate::AsyncDb;
use crate::semantic_search::ClusterIndex;
use crate::semantic_search::cluster::{Cluster, read_clusters_from_file};
use crate::semantic_search::service::{QueryEmbeddings, SemanticSearchConfig, embed_query, search};
use crate::vector_kv::DbVectorStore;
use crate::vector_kv::eval_indexing::index_texts;

const NS: &str = "beir";
const INDEX_CONCURRENCY: usize = 32;
const NDCG_K: usize = 10;
const RECALL_K: usize = 100;
const NPROBES: [usize; 2] = [32, 256];
const FIRST_PASS: [usize; 2] = [1000, 100];

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

#[derive(Default, Clone, Copy)]
struct Metrics {
    ndcg: f64,
    mrr: f64,
    recall: f64,
    cand_recall: f64,
}

/// Score one query: `ranked` is the dense-ordered candidate list (corpus ids).
fn score(ranked: &[&str], rels: &HashMap<String, u32>) -> Metrics {
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
    let hits = |k: usize| ranked.iter().take(k).filter(|id| rels.contains_key(**id)).count() as f64 / rels.len() as f64;
    Metrics {
        ndcg: if idcg > 0.0 { dcg / idcg } else { 0.0 },
        mrr: ranked
            .iter()
            .take(NDCG_K)
            .position(|id| rels.contains_key(*id))
            .map_or(0.0, |p| 1.0 / (p + 1) as f64),
        recall: hits(RECALL_K),
        cand_recall: hits(ranked.len()),
    }
}

fn doc_u64(id: &[u8]) -> u64 {
    u64::from_be_bytes(id.try_into().expect("beir doc ids are u64 BE"))
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn beir_eval() {
    let dataset = env_or("MINNAL_BEIR_DATASET", "scifact");
    let root = PathBuf::from(env_or("MINNAL_BEIR_ROOT", "../work/beir"));
    let split = env_or("MINNAL_BEIR_SPLIT", "test");
    let model = env_or("MINNAL_BEIR_MODEL", "gemma");
    let embed_url = std::env::var("MINNAL_EMBED_URL").unwrap_or_else(|_| SemanticSearchConfig::default().embedding_service_url);
    let max_queries: Option<usize> = std::env::var("MINNAL_BEIR_MAX_QUERIES").ok().and_then(|v| v.parse().ok());
    let dir = root.join(&dataset);
    if !dir.join("corpus.jsonl").exists() {
        eprintln!(
            "SKIP beir_eval: {} missing — run service/scripts/fetch_beir.sh {dataset}",
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
    let query_words: Vec<usize> = queries.iter().map(|(_, t)| t.split_whitespace().count()).collect();
    let mut sorted_words = query_words.clone();
    sorted_words.sort_unstable();

    let cluster_path = format!("../service/embedding_support/{model}/clusters.json");
    let raw = read_clusters_from_file(&cluster_path).unwrap_or_else(|e| panic!("load {cluster_path}: {e}"));
    let index = Arc::new(ClusterIndex::from_clusters(
        raw.into_iter().map(|(id, c)| (id, Cluster::new(id, c))).collect(),
    ));
    let base_config = Arc::new(SemanticSearchConfig {
        embedding_service_url: embed_url.clone(),
        model_name: model.clone(),
        ..SemanticSearchConfig::default()
    });

    eprintln!("\n=== BEIR eval: {dataset} ({split}) ===");
    eprintln!(
        "  service {embed_url} | model {model} | {} docs, {} judged queries (words: median {}, p90 {})",
        corpus.len(),
        queries.len(),
        sorted_words[sorted_words.len() / 2],
        sorted_words[sorted_words.len() * 9 / 10]
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
        assert!(
            ok > 0,
            "every embed failed — is the service up at {embed_url}? first error: {first_err:?}"
        );
        if fail == 0 {
            std::fs::write(&marker, format!("{ok}\n")).unwrap();
        }
    }
    let store = DbVectorStore::new(&db, NS).await.unwrap();

    // ── Embed queries once, then every n_probes × first-pass setting ──
    struct Row {
        n_probes: usize,
        first_pass: usize,
        per_query: Vec<Metrics>,
        probed: f64,
        latency_ms: f64,
    }
    let t = Instant::now();
    let mut q_embs: Vec<QueryEmbeddings> = Vec::with_capacity(queries.len());
    for (qid, text) in &queries {
        q_embs.push(embed_query(&base_config, text).await.unwrap_or_else(|e| panic!("embed query {qid}: {e}")));
    }
    let embed_ms = t.elapsed().as_secs_f64() * 1e3 / queries.len() as f64;
    let vectors = q_embs.iter().map(|q| q.sparse.len()).sum::<usize>() as f64 / q_embs.len() as f64;

    let mut rows: Vec<Row> = Vec::new();
    for &n_probes in &NPROBES {
        for &first_pass in &FIRST_PASS {
            let run_config = SemanticSearchConfig {
                n_probes,
                first_pass_sparse_search_top_k: first_pass,
                ..(*base_config).clone()
            };
            let mut per_query = Vec::with_capacity(queries.len());
            let mut latencies = Vec::with_capacity(queries.len());
            let mut probed_total = 0usize;
            for ((qid, _), qe) in queries.iter().zip(&q_embs) {
                probed_total += index
                    .find_top_n_cluster_ids_batch(&qe.sparse, n_probes)
                    .into_iter()
                    .flatten()
                    .collect::<HashSet<u32>>()
                    .len();
                let t = Instant::now();
                // top_k = first_pass returns every candidate, in final (dense) order.
                let results = search(
                    &run_config,
                    NS,
                    &index,
                    &qe.sparse,
                    &qe.dense,
                    &store,
                    None::<fn(&[u8]) -> bool>,
                    Some(first_pass),
                )
                .await;
                latencies.push(t.elapsed().as_secs_f64() * 1e3);
                let ranked: Vec<&str> = results.iter().map(|r| corpus_ids[doc_u64(&r.document_id) as usize].as_str()).collect();
                per_query.push(score(&ranked, &qrels[qid]));
            }
            latencies.sort_by(|a, b| a.total_cmp(b));
            rows.push(Row {
                n_probes,
                first_pass,
                per_query,
                probed: probed_total as f64 / queries.len() as f64,
                latency_ms: latencies[latencies.len() / 2],
            });
        }
    }

    // ── Report ──
    let nq = queries.len() as f64;
    let mean = |ms: &[Metrics], f: fn(&Metrics) -> f64| ms.iter().map(f).sum::<f64>() / nq;
    let mut md = String::new();
    let mut tsv = String::from("n_probes\tfirst_pass\tndcg10\tmrr10\trecall100\tcand_recall\tprobed\tsearch_ms\n");
    let _ = writeln!(md, "# BEIR eval — {dataset} ({split}), model {model}\n");
    let _ = writeln!(
        md,
        "{} docs, {} queries (words: median {}, p90 {}). Query vectors per query: {vectors:.1}; query embedding {embed_ms:.1} ms/q.\n",
        corpus.len(),
        queries.len(),
        sorted_words[sorted_words.len() / 2],
        sorted_words[sorted_words.len() * 9 / 10]
    );
    let _ = writeln!(
        md,
        "| n_probes | first-pass top-k | nDCG@10 | MRR@10 | R@100 | cand. recall | clusters probed | search ms (p50) |"
    );
    let _ = writeln!(md, "|---:|---:|---:|---:|---:|---:|---:|---:|");
    for r in &rows {
        let (ndcg, mrr, rec, cand) = (
            mean(&r.per_query, |m| m.ndcg),
            mean(&r.per_query, |m| m.mrr),
            mean(&r.per_query, |m| m.recall),
            mean(&r.per_query, |m| m.cand_recall),
        );
        let _ = writeln!(
            md,
            "| {} | {} | {ndcg:.4} | {mrr:.4} | {rec:.4} | {cand:.4} | {:.0} | {:.1} |",
            r.n_probes, r.first_pass, r.probed, r.latency_ms
        );
        let _ = writeln!(
            tsv,
            "{}\t{}\t{ndcg:.6}\t{mrr:.6}\t{rec:.6}\t{cand:.6}\t{:.1}\t{:.2}",
            r.n_probes, r.first_pass, r.probed, r.latency_ms
        );
    }
    eprintln!("\n{md}");
    for (ext, body) in [("md", &md), ("tsv", &tsv)] {
        let out = dir.join(format!("beir_eval_{model}_{split}.{ext}"));
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
        let m = score(&["a", "b", "x"], &rels(&[("a", 2), ("b", 1)]));
        assert!((m.ndcg - 1.0).abs() < 1e-12);
        assert_eq!((m.mrr, m.recall, m.cand_recall), (1.0, 1.0, 1.0));
    }

    #[test]
    fn candidate_recall_uses_the_whole_candidate_list() {
        let ranked: Vec<String> = (0..150).map(|i| format!("d{i}")).collect();
        let ranked: Vec<&str> = ranked.iter().map(String::as_str).collect();
        let m = score(&ranked, &rels(&[("d5", 1), ("d120", 1)]));
        assert_eq!(m.recall, 0.5, "d120 is outside the top 100");
        assert_eq!(m.cand_recall, 1.0, "but it is among the candidates");
    }
}
