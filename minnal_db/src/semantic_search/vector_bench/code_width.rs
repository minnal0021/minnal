//! Code width against centre choice (design doc M2c-pre).
//!
//! On the frozen embeddings, with production Pass 1 (1-bit chunk codes, 64
//! probes, cut 1,000) supplying the candidates:
//!
//! - **Pass 2**: whole-document codes at {2, 4, 8} bits, encoded against the
//!   nearest centroid (production) or a zero centre. Estimator RMSE and bias
//!   over every candidate, nDCG@10, and top-10 agreement with an exact dense
//!   rerank of the same candidates.
//! - **Pass 1**: chunk codes at {1, 2} bits against the nearest centroid.
//!   Estimator RMSE, Pass-1 recall (overlap with the exact top 1,000), exact
//!   top-100 coverage, and nDCG@10 after an exact dense rerank (so only Pass 1
//!   differs).
//!
//! Every code is built with the production quantiser (`index_embedding_rotated`
//! / `index_embedding_to_cluster`) and scored with the production estimators.
//!
//! ```sh
//! MINNAL_BENCH_DATASET=fiqa MINNAL_BENCH_MODEL=qwen cargo test -p minnal_db --all-features \
//!   --release --lib vector_bench_code_width -- --ignored --nocapture
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::path::PathBuf;

use rayon::prelude::*;

use super::super::metrics::beir_eval::{env_or, ndcg_at, read_qrels, score};
use super::exact::{self, dot};
use super::frozen;
use super::paired_delta;
use crate::semantic_search::ClusterIndex;
use crate::semantic_search::cluster::{Cluster, read_clusters_from_file};
use crate::semantic_search::index::distance_estimator::{MultiBitQuanDotProductEstimator, SingleBitQuanDotProductEstimator};
use crate::semantic_search::index::vector_index::{QuantisationStyle, VectorIndex};
use crate::semantic_search::quantisation::rabitq::{index_embedding_rotated, index_embedding_to_cluster};
use crate::semantic_search::service::SemanticSearchConfig;

const N_PROBES: usize = 64;
const CUT: usize = 1000;
const PASS2_BITS: [usize; 3] = [2, 4, 8];
const PASS1_BITS: [usize; 2] = [1, 2];
/// nDCG cutoffs Pass 2 is judged at.
const CUTOFFS: [usize; 6] = [10, 20, 30, 40, 50, 100];
/// Every this-many queries contribute every probed chunk to Pass-1 error.
const ERROR_QUERY_STEP: usize = 8;

#[derive(Default, Clone, Copy)]
struct Err {
    n: usize,
    sum: f64,
    sum_sq: f64,
}

impl Err {
    fn add(&mut self, e: f32) {
        self.n += 1;
        self.sum += e as f64;
        self.sum_sq += (e as f64).powi(2);
    }
    fn merge(&mut self, o: &Err) {
        self.n += o.n;
        self.sum += o.sum;
        self.sum_sq += o.sum_sq;
    }
    fn rmse(&self) -> f64 {
        (self.sum_sq / self.n.max(1) as f64).sqrt()
    }
    fn bias(&self) -> f64 {
        self.sum / self.n.max(1) as f64
    }
}

/// Per query, per Pass-2 configuration.
#[derive(Default, Clone, Copy)]
struct Pass2Row {
    err: Err,
    ndcg: f64,
    /// nDCG at each of [`CUTOFFS`].
    ndcg_k: [f64; CUTOFFS.len()],
    top10_overlap: f64,
    top10_identical: bool,
}

/// Per query, per Pass-1 width.
#[derive(Default, Clone, Copy)]
struct Pass1Row {
    err: Err,
    recall1000: f64,
    recall100: f64,
    ndcg: f64,
}

/// Score a multi-bit code (any width) as Pass 2 does.
fn multi_bit_score(vi: &VectorIndex, q: &[f32], rq: &[f32], centroid: &[f32], bits: usize) -> f32 {
    let sum = MultiBitQuanDotProductEstimator::scaled_query_sum(rq, bits);
    MultiBitQuanDotProductEstimator::with_scaled_query_sum(vi.cluster_id, q, centroid, sum).estimate_from_parts(
        rq,
        &vi.packed_vector,
        vi.addition_factor,
        vi.scaling_factor,
    )
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn vector_bench_code_width() {
    let dataset = env_or("MINNAL_BENCH_DATASET", "scifact");
    let model = env_or("MINNAL_BENCH_MODEL", "gemma");
    let label = env_or("MINNAL_BENCH_LABEL", "code-width");
    let bench_root = PathBuf::from(env_or("MINNAL_BENCH_ROOT", "../work/bench"));
    let split = env_or("MINNAL_BEIR_SPLIT", "test");
    let dataset_dir = PathBuf::from(env_or("MINNAL_BEIR_ROOT", "../work/beir")).join(&dataset);
    let frozen_dir = bench_root.join(&dataset).join(&model);
    assert!(frozen_dir.join("manifest.json").exists(), "run vector_bench for {dataset}/{model} first");
    let config = SemanticSearchConfig {
        model_name: model.clone(),
        ..SemanticSearchConfig::default()
    };
    let frozen = frozen::load_or_dump(
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
    .await;
    let gt = exact::load_or_compute(&frozen_dir, &frozen);
    let qrels = read_qrels(&dataset_dir.join("qrels").join(format!("{split}.tsv")));
    let raw = read_clusters_from_file(&format!("../service/embedding_support/{model}/clusters.json")).unwrap();
    let index = ClusterIndex::from_clusters(raw.into_iter().map(|(id, c)| (id, Cluster::new(id, c))).collect());
    let (nd, nq, dim) = (frozen.n_docs(), frozen.n_queries(), frozen.dim());
    eprintln!(
        "\n=== code width: {dataset} ({model}), {nd} docs, {} chunks, {nq} queries ===",
        frozen.n_chunks()
    );

    // ── Codes ──
    let chunk_codes: Vec<Vec<VectorIndex>> = PASS1_BITS
        .iter()
        .map(|&b| {
            let style = if b == 1 {
                QuantisationStyle::SingleBit
            } else {
                QuantisationStyle::MultiBit { number_of_bits: b }
            };
            (0..frozen.n_chunks())
                .into_par_iter()
                .map(|c| index_embedding_rotated(&index, frozen.chunk(c), style.clone()).unwrap())
                .collect()
        })
        .collect();
    let zero = Cluster::new(u32::MAX, vec![0.0; dim]);
    // (bits, zero_centre) → one code per doc.
    let mut dense_codes: Vec<((usize, bool), Vec<VectorIndex>)> = Vec::new();
    for &b in &PASS2_BITS {
        let style = QuantisationStyle::MultiBit { number_of_bits: b };
        let nearest = (0..nd)
            .into_par_iter()
            .map(|d| index_embedding_rotated(&index, frozen.dense(d), style.clone()).unwrap())
            .collect();
        let zeroed = (0..nd)
            .into_par_iter()
            .map(|d| index_embedding_to_cluster(&index.rotate(frozen.dense(d)), &zero, style.clone()))
            .collect();
        dense_codes.push(((b, false), nearest));
        dense_codes.push(((b, true), zeroed));
    }
    let mut cluster_chunks: HashMap<u32, Vec<usize>> = HashMap::new();
    for (c, vi) in chunk_codes[0].iter().enumerate() {
        cluster_chunks.entry(vi.cluster_id).or_default().push(c);
    }
    let zero_vec = vec![0.0f32; dim];

    // ── Per query ──
    let per_query: Vec<(Vec<Pass1Row>, Vec<Pass2Row>)> = (0..nq)
        .into_par_iter()
        .map(|qi| {
            let q = frozen.query(qi);
            let rq = index.rotate(q);
            let rels = &qrels[&frozen.manifest.query_ids[qi]];
            let ids = |list: &[u32]| -> Vec<&str> { list.iter().map(|&d| frozen.manifest.doc_ids[d as usize].as_str()).collect() };
            let exact_top: HashSet<u32> = gt.pass1[qi].iter().copied().collect();
            let exact_top100: Vec<u32> = gt.pass1[qi].iter().take(100).copied().collect();
            let dense_exact: Vec<f32> = (0..nd).map(|d| dot(q, frozen.dense(d))).collect();
            let rerank_exact = |cands: &[u32]| -> Vec<u32> {
                let mut s = vec![f32::NEG_INFINITY; nd];
                for &d in cands {
                    s[d as usize] = dense_exact[d as usize];
                }
                exact::top_k(&s, 10)
            };
            let probes = index.find_top_n_cluster_ids_batch(&[q.to_vec()], N_PROBES).remove(0);
            let track_err = qi % ERROR_QUERY_STEP == 0;

            // Pass 1 at each width.
            let mut pass1 = Vec::new();
            let mut production_cut: Vec<u32> = Vec::new();
            for (wi, &bits) in PASS1_BITS.iter().enumerate() {
                let mut row = Pass1Row::default();
                let mut doc = vec![f32::NEG_INFINITY; nd];
                let sum1 = SingleBitQuanDotProductEstimator::query_sum(&rq);
                for &cl in &probes {
                    let centroid = &index.clusters[&cl].centroid;
                    let single = SingleBitQuanDotProductEstimator::with_query_sum(cl, q, centroid, sum1);
                    for &c in cluster_chunks.get(&cl).map(Vec::as_slice).unwrap_or(&[]) {
                        let vi = &chunk_codes[wi][c];
                        let est = if bits == 1 {
                            single.estimate_from_parts(&rq, &vi.packed_vector, vi.scaling_factor)
                        } else {
                            multi_bit_score(vi, q, &rq, centroid, bits)
                        };
                        if track_err {
                            row.err.add(est - dot(q, frozen.chunk(c)));
                        }
                        let d = frozen.chunk_doc[c] as usize;
                        if est > doc[d] {
                            doc[d] = est;
                        }
                    }
                }
                let cut = exact::top_k(&doc, CUT);
                let cut_set: HashSet<u32> = cut.iter().copied().collect();
                row.recall1000 = cut.iter().filter(|d| exact_top.contains(d)).count() as f64 / exact_top.len().max(1) as f64;
                row.recall100 = exact_top100.iter().filter(|d| cut_set.contains(d)).count() as f64 / exact_top100.len().max(1) as f64;
                row.ndcg = score(&ids(&rerank_exact(&cut)), rels).ndcg;
                if bits == 1 {
                    production_cut = cut;
                }
                pass1.push(row);
            }

            // Pass 2 at each width and centre, on the production candidates.
            let exact10 = rerank_exact(&production_cut);
            let exact10_set: HashSet<u32> = exact10.iter().copied().collect();
            let pass2 = dense_codes
                .iter()
                .map(|&((bits, zeroed), ref codes)| {
                    let mut row = Pass2Row::default();
                    let mut s = vec![f32::NEG_INFINITY; nd];
                    for &d in &production_cut {
                        let vi = &codes[d as usize];
                        let centroid: &[f32] = if zeroed { &zero_vec } else { &index.clusters[&vi.cluster_id].centroid };
                        let est = multi_bit_score(vi, q, &rq, centroid, bits);
                        row.err.add(est - dense_exact[d as usize]);
                        s[d as usize] = est;
                    }
                    let top = exact::top_k(&s, 100);
                    let top10: Vec<u32> = top.iter().take(10).copied().collect();
                    row.ndcg = score(&ids(&top10), rels).ndcg;
                    let ranked = ids(&top);
                    for (slot, &k) in row.ndcg_k.iter_mut().zip(&CUTOFFS) {
                        *slot = ndcg_at(&ranked, rels, k);
                    }
                    row.top10_overlap = top10.iter().filter(|d| exact10_set.contains(d)).count() as f64 / exact10_set.len().max(1) as f64;
                    row.top10_identical = top10 == exact10;
                    row
                })
                .collect();
            (pass1, pass2)
        })
        .collect();

    // ── Report ──
    let n = nq as f64;
    let exact_ndcg = (0..nq)
        .map(|qi| {
            let q = frozen.query(qi);
            let mut s = vec![f32::NEG_INFINITY; nd];
            for &d in &gt.pass1[qi] {
                s[d as usize] = dot(q, frozen.dense(d as usize));
            }
            let top: Vec<&str> = exact::top_k(&s, 10)
                .iter()
                .map(|&d| frozen.manifest.doc_ids[d as usize].as_str())
                .collect();
            score(&top, &qrels[&frozen.manifest.query_ids[qi]]).ndcg
        })
        .sum::<f64>()
        / n;
    let mut md = String::new();
    let _ = writeln!(md, "# Code width vs centre — {dataset} ({model})\n");
    let _ = writeln!(
        md,
        "{nd} docs, {} chunks, {nq} queries; Pass 1 at {N_PROBES} probes, cut {CUT}. Exact two-pass nDCG@10 {exact_ndcg:.4}.\n",
        frozen.n_chunks()
    );
    let _ = writeln!(md, "## Pass 2 (whole-document codes) on production Pass-1 candidates\n");
    let _ = writeln!(
        md,
        "| Bits | Centre | RMSE | Bias | nDCG@10 | Top-10 overlap with exact rerank | Identical top 10 |"
    );
    let _ = writeln!(md, "|---:|---|---:|---:|---:|---:|---:|");
    for (i, ((bits, zeroed), _)) in dense_codes.iter().enumerate() {
        let mut e = Err::default();
        for (_, p2) in &per_query {
            e.merge(&p2[i].err);
        }
        let _ = writeln!(
            md,
            "| {bits} | {} | {:.5} | {:+.5} | {:.4} | {:.4} | {}/{nq} |",
            if *zeroed { "zero" } else { "nearest" },
            e.rmse(),
            e.bias(),
            per_query.iter().map(|(_, p2)| p2[i].ndcg).sum::<f64>() / n,
            per_query.iter().map(|(_, p2)| p2[i].top10_overlap).sum::<f64>() / n,
            per_query.iter().filter(|(_, p2)| p2[i].top10_identical).count(),
        );
    }
    let _ = writeln!(md, "\n### Pass 2 nDCG by cutoff\n");
    let _ = writeln!(
        md,
        "| Bits | Centre | {} |",
        CUTOFFS.iter().map(|k| format!("@{k}")).collect::<Vec<_>>().join(" | ")
    );
    let _ = writeln!(md, "|---:|---|{}", "---:|".repeat(CUTOFFS.len()));
    for (i, ((bits, zeroed), _)) in dense_codes.iter().enumerate() {
        let means: Vec<String> = (0..CUTOFFS.len())
            .map(|c| format!("{:.4}", per_query.iter().map(|(_, p2)| p2[i].ndcg_k[c]).sum::<f64>() / n))
            .collect();
        let _ = writeln!(md, "| {bits} | {} | {} |", if *zeroed { "zero" } else { "nearest" }, means.join(" | "));
    }
    let _ = writeln!(
        md,
        "\n### Zero minus nearest centre, paired per query (mean [95% interval], better / worse / same)\n"
    );
    let _ = writeln!(md, "| Bits | Cutoff | Δ nDCG | Better / worse / same |\n|---:|---:|---|---|");
    for &bits in &PASS2_BITS {
        let idx = |zeroed: bool| dense_codes.iter().position(|((b, z), _)| *b == bits && *z == zeroed).unwrap();
        let (near, zero) = (idx(false), idx(true));
        for (c, k) in CUTOFFS.iter().enumerate() {
            let base: Vec<f64> = per_query.iter().map(|(_, p2)| p2[near].ndcg_k[c]).collect();
            let new: Vec<f64> = per_query.iter().map(|(_, p2)| p2[zero].ndcg_k[c]).collect();
            let (mean, lo, hi) = paired_delta(&base, &new);
            let (better, worse) = base.iter().zip(&new).fold((0, 0), |(b, w), (x, y)| {
                if y > x {
                    (b + 1, w)
                } else if y < x {
                    (b, w + 1)
                } else {
                    (b, w)
                }
            });
            let _ = writeln!(
                md,
                "| {bits} | @{k} | {mean:+.4} [{lo:+.4}, {hi:+.4}] | {better} / {worse} / {} |",
                nq - better - worse
            );
        }
    }
    let _ = writeln!(md, "\n## Pass 1 (chunk codes, nearest centre), exact dense rerank\n");
    let _ = writeln!(md, "| Bits | RMSE | Bias | Pass-1 recall | Exact top 100 kept | nDCG@10 |");
    let _ = writeln!(md, "|---:|---:|---:|---:|---:|---:|");
    for (i, bits) in PASS1_BITS.iter().enumerate() {
        let mut e = Err::default();
        for (p1, _) in &per_query {
            e.merge(&p1[i].err);
        }
        let _ = writeln!(
            md,
            "| {bits} | {:.5} | {:+.5} | {:.4} | {:.4} | {:.4} |",
            e.rmse(),
            e.bias(),
            per_query.iter().map(|(p1, _)| p1[i].recall1000).sum::<f64>() / n,
            per_query.iter().map(|(p1, _)| p1[i].recall100).sum::<f64>() / n,
            per_query.iter().map(|(p1, _)| p1[i].ndcg).sum::<f64>() / n,
        );
    }
    eprintln!("{md}");
    let out = bench_root.join("results").join(&label);
    std::fs::create_dir_all(&out).unwrap();
    std::fs::write(out.join(format!("{dataset}-{model}-code-width.md")), &md).unwrap();
}
