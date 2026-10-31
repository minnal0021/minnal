//! Pass-1 code width against probe count and first-pass cut (Pass-1 design doc,
//! Part B, B5).
//!
//! A wider chunk code costs more bytes per entry read, so it can only pay for
//! itself by letting a search read fewer entries for the same result: fewer
//! probes, or a smaller first-pass cut. This sweeps chunk codes of 1, 2 and 4
//! bits over probe counts and two cuts, with production Pass 2 (8-bit
//! whole-document codes against the zero centre) after each, and reports:
//!
//! - entries and code bytes read per query (bytes at the tight packing: 1, 2 or
//!   4 bits per dimension, plus the 12 bytes of scalar factors);
//! - Pass-1 recall (overlap of the cut with the exact top 1,000);
//! - top-10 overlap with the production configuration's result (1 bit, every
//!   posting, cut 1,000), the "same result as a full read" question;
//! - nDCG at 10, 20, 30, 40, 50 and 100, and the paired change against that
//!   reference.
//!
//! The partition is the study's static k-means (256 centres), the same for every
//! width, so this measures the estimate only, not the better placement wider
//! codes would give maintenance (B4).
//!
//! ```sh
//! MINNAL_BENCH_DATASET=fiqa MINNAL_BENCH_MODEL=qwen cargo test -p minnal_db --all-features \
//!   --release --lib vector_bench_pass1_width -- --ignored --nocapture
//! ```

use std::collections::HashSet;
use std::fmt::Write as _;
use std::path::PathBuf;

use rayon::prelude::*;

use super::super::metrics::beir_eval::{env_or, ndcg_at, read_qrels};
use super::exact;
use super::frozen;
use super::paired_delta;
use crate::semantic_search::cluster::Cluster;
use crate::semantic_search::index::distance_estimator::{MultiBitQuanDotProductEstimator, SingleBitQuanDotProductEstimator};
use crate::semantic_search::index::vector_index::{QuantisationStyle, VectorIndex};
use crate::semantic_search::quantisation::rabitq::{index_embedding_rotated, index_embedding_to_cluster};
use crate::semantic_search::service::SemanticSearchConfig;

const WIDTHS: [usize; 3] = [1, 2, 4];
const PROBES: [usize; 7] = [4, 8, 16, 32, 64, 128, 256];
const CUTS: [usize; 2] = [1000, 300];
const PASS2_BITS: usize = 8;
const CUTOFFS: [usize; 6] = [10, 20, 30, 40, 50, 100];
/// Scalar factors stored beside each code (addition, scaling, error bound).
const FACTOR_BYTES: usize = 12;
/// "Same quality" for the summary: within this much nDCG of the reference at
/// every cutoff.
const TOLERANCE: f64 = 0.001;

/// One query's outcome for one (width, probes, cut).
#[derive(Default, Clone, Copy)]
struct Cell {
    entries: usize,
    recall: f64,
    overlap10: f64,
    ndcg: [f64; CUTOFFS.len()],
}

/// A Pass-1 estimator of either kind, built once per (query, posting).
enum Est {
    Single(SingleBitQuanDotProductEstimator),
    Multi(MultiBitQuanDotProductEstimator),
}

impl Est {
    fn score(&self, rq: &[f32], vi: &VectorIndex) -> f32 {
        match self {
            Est::Single(e) => e.estimate_from_parts(rq, &vi.packed_vector, vi.scaling_factor),
            Est::Multi(e) => e.estimate_from_parts(rq, &vi.packed_vector, vi.addition_factor, vi.scaling_factor),
        }
    }
}

fn style(bits: usize) -> QuantisationStyle {
    if bits == 1 {
        QuantisationStyle::SingleBit
    } else {
        QuantisationStyle::MultiBit { number_of_bits: bits }
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn vector_bench_pass1_width() {
    let dataset = env_or("MINNAL_BENCH_DATASET", "scifact");
    let model = env_or("MINNAL_BENCH_MODEL", "gemma");
    let label = env_or("MINNAL_BENCH_LABEL", "pass1-width");
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
    let index = super::study_centroids(&frozen);
    let (nd, nq, dim) = (frozen.n_docs(), frozen.n_queries(), frozen.dim());
    let n_centres = index.clusters.len();
    eprintln!(
        "\n=== Pass-1 width: {dataset} ({model}), {nd} docs, {} chunks, {nq} queries, {n_centres} centres ===",
        frozen.n_chunks()
    );

    // ── Codes ──
    let chunk_codes: Vec<Vec<VectorIndex>> = WIDTHS
        .iter()
        .map(|&b| {
            (0..frozen.n_chunks())
                .into_par_iter()
                .map(|c| index_embedding_rotated(&index, frozen.chunk(c), style(b)).unwrap())
                .collect()
        })
        .collect();
    let zero = Cluster::new(u32::MAX, vec![0.0; dim]);
    let dense_codes: Vec<VectorIndex> = (0..nd)
        .into_par_iter()
        .map(|d| index_embedding_to_cluster(&index.rotate(frozen.dense(d)), &zero, style(PASS2_BITS)))
        .collect();
    // Postings by id; every width routes a chunk the same way (nearest centre).
    let max_id = index.clusters.keys().copied().max().unwrap_or(0) as usize;
    let mut posting_chunks: Vec<Vec<usize>> = vec![Vec::new(); max_id + 1];
    for (c, vi) in chunk_codes[0].iter().enumerate() {
        posting_chunks[vi.cluster_id as usize].push(c);
    }
    let zero_vec = vec![0.0f32; dim];
    let probe_counts: Vec<usize> = PROBES.iter().copied().filter(|&p| p < n_centres).chain([n_centres]).collect();

    // ── Per query: cells[width][probe][cut] ──
    let per_query: Vec<Vec<Vec<Vec<Cell>>>> = (0..nq)
        .into_par_iter()
        .map(|qi| {
            let q = frozen.query(qi);
            let rq = index.rotate(q);
            let rels = &qrels[&frozen.manifest.query_ids[qi]];
            let exact_top: HashSet<u32> = gt.pass1[qi].iter().copied().collect();
            // Postings nearest first, by Euclidean distance (production probing).
            let mut order: Vec<(f32, u32)> = index
                .clusters
                .iter()
                .map(|(&id, c)| (c.centroid.iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum::<f32>(), id))
                .collect();
            order.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let sum1 = SingleBitQuanDotProductEstimator::query_sum(&rq);
            let dense_est = MultiBitQuanDotProductEstimator::new(u32::MAX, q, &zero_vec, PASS2_BITS);
            let pass2 = |cut: &[u32]| -> Vec<u32> {
                let mut s = vec![f32::NEG_INFINITY; nd];
                for &d in cut {
                    let vi = &dense_codes[d as usize];
                    s[d as usize] = dense_est.estimate_from_parts(&rq, &vi.packed_vector, vi.addition_factor, vi.scaling_factor);
                }
                exact::top_k(&s, 100)
            };
            WIDTHS
                .iter()
                .enumerate()
                .map(|(wi, &bits)| {
                    let mut doc = vec![f32::NEG_INFINITY; nd];
                    let mut read = 0usize;
                    let mut next = 0usize;
                    let mut rows = Vec::new();
                    for &p in &probe_counts {
                        while next < p {
                            let id = order[next].1;
                            let centroid = &index.clusters[&id].centroid;
                            let est = if bits == 1 {
                                Est::Single(SingleBitQuanDotProductEstimator::with_query_sum(id, q, centroid, sum1))
                            } else {
                                Est::Multi(MultiBitQuanDotProductEstimator::new(id, q, centroid, bits))
                            };
                            for &c in &posting_chunks[id as usize] {
                                let s = est.score(&rq, &chunk_codes[wi][c]);
                                let d = frozen.chunk_doc[c] as usize;
                                if s > doc[d] {
                                    doc[d] = s;
                                }
                            }
                            read += posting_chunks[id as usize].len();
                            next += 1;
                        }
                        let row: Vec<(Cell, Vec<u32>)> = CUTS
                            .iter()
                            .map(|&cut| {
                                let kept = exact::top_k(&doc, cut);
                                let top = pass2(&kept);
                                let ranked: Vec<&str> = top.iter().map(|&d| frozen.manifest.doc_ids[d as usize].as_str()).collect();
                                let mut ndcg = [0.0; CUTOFFS.len()];
                                for (slot, &k) in ndcg.iter_mut().zip(&CUTOFFS) {
                                    *slot = ndcg_at(&ranked, rels, k);
                                }
                                Cell {
                                    entries: read,
                                    recall: kept.iter().filter(|d| exact_top.contains(d)).count() as f64 / exact_top.len().max(1) as f64,
                                    // Filled in against the reference below; keep the top 10 here.
                                    overlap10: 0.0,
                                    ndcg,
                                }
                                .with_top10(&top)
                            })
                            .collect();
                        rows.push(row);
                    }
                    rows
                })
                .collect()
        })
        .collect::<Vec<Vec<Vec<Vec<(Cell, Vec<u32>)>>>>>()
        .into_iter()
        .map(|q| {
            // Reference: 1 bit, every posting, cut 1,000.
            let reference: HashSet<u32> = q[0][probe_counts.len() - 1][0].1.iter().copied().collect();
            q.into_iter()
                .map(|w| {
                    w.into_iter()
                        .map(|p| {
                            p.into_iter()
                                .map(|(mut cell, top10)| {
                                    cell.overlap10 = top10.iter().filter(|d| reference.contains(d)).count() as f64 / reference.len().max(1) as f64;
                                    cell
                                })
                                .collect()
                        })
                        .collect()
                })
                .collect()
        })
        .collect();

    // ── Report ──
    let n = nq as f64;
    let mean = |f: &dyn Fn(&Cell) -> f64, w: usize, p: usize, c: usize| per_query.iter().map(|q| f(&q[w][p][c])).sum::<f64>() / n;
    let reference_ndcg = |k: usize| -> Vec<f64> { per_query.iter().map(|q| q[0][probe_counts.len() - 1][0].ndcg[k]).collect() };
    let mut md = String::new();
    let _ = writeln!(md, "# Pass-1 code width vs probes and cut — {dataset} ({model})\n");
    let _ = writeln!(
        md,
        "{nd} docs, {} chunks, {nq} queries, {n_centres} static k-means centres (same partition for every width). \
         Pass 2: production ({PASS2_BITS}-bit whole-document codes, zero centre). Reference: 1 bit, every posting, cut 1,000. \
         Bytes are code bytes at the tight packing plus {FACTOR_BYTES} bytes of factors per entry.\n",
        frozen.n_chunks()
    );
    let _ = writeln!(
        md,
        "| Bits | Probes | Cut | Entries read | Code MB read | Pass-1 recall | Top-10 overlap with reference | {} |",
        CUTOFFS.iter().map(|k| format!("nDCG@{k}")).collect::<Vec<_>>().join(" | ")
    );
    let _ = writeln!(md, "|---:|---:|---:|---:|---:|---:|---:|{}", "---:|".repeat(CUTOFFS.len()));
    for (wi, &bits) in WIDTHS.iter().enumerate() {
        let bytes_per = dim * bits / 8 + FACTOR_BYTES;
        for (pi, &p) in probe_counts.iter().enumerate() {
            for (ci, &cut) in CUTS.iter().enumerate() {
                let entries = mean(&|c| c.entries as f64, wi, pi, ci);
                let ndcg: Vec<String> = (0..CUTOFFS.len()).map(|k| format!("{:.4}", mean(&|c| c.ndcg[k], wi, pi, ci))).collect();
                let _ = writeln!(
                    md,
                    "| {bits} | {p} | {cut} | {entries:.0} | {:.2} | {:.4} | {:.4} | {} |",
                    entries * bytes_per as f64 / 1e6,
                    mean(&|c| c.recall, wi, pi, ci),
                    mean(&|c| c.overlap10, wi, pi, ci),
                    ndcg.join(" | ")
                );
            }
        }
    }

    let _ = writeln!(
        md,
        "\n## Smallest read within {TOLERANCE} nDCG of the reference at every cutoff\n\n\
         Paired per query against the reference; the worst cutoff's mean change and its 95% interval.\n"
    );
    let _ = writeln!(md, "| Bits | Cut | Probes | Code MB read | Worst Δ nDCG [95% interval] | At cutoff |");
    let _ = writeln!(md, "|---:|---:|---:|---:|---|---:|");
    for (wi, &bits) in WIDTHS.iter().enumerate() {
        let bytes_per = dim * bits / 8 + FACTOR_BYTES;
        for (ci, &cut) in CUTS.iter().enumerate() {
            let found = probe_counts.iter().enumerate().find_map(|(pi, &p)| {
                let worst = (0..CUTOFFS.len())
                    .map(|k| {
                        let new: Vec<f64> = per_query.iter().map(|q| q[wi][pi][ci].ndcg[k]).collect();
                        (paired_delta(&reference_ndcg(k), &new), k)
                    })
                    .min_by(|a, b| a.0.0.total_cmp(&b.0.0))
                    .unwrap();
                (worst.0.0 >= -TOLERANCE).then_some((pi, p, worst))
            });
            match found {
                Some((pi, p, ((m, lo, hi), k))) => {
                    let _ = writeln!(
                        md,
                        "| {bits} | {cut} | {p} | {:.2} | {m:+.4} [{lo:+.4}, {hi:+.4}] | @{} |",
                        mean(&|c| c.entries as f64, wi, pi, ci) * bytes_per as f64 / 1e6,
                        CUTOFFS[k]
                    );
                }
                None => {
                    let _ = writeln!(md, "| {bits} | {cut} | none | | | |");
                }
            }
        }
    }
    eprintln!("{md}");
    let out = bench_root.join("results").join(&label);
    std::fs::create_dir_all(&out).unwrap();
    std::fs::write(out.join(format!("{dataset}-{model}-pass1-width.md")), &md).unwrap();
}

impl Cell {
    fn with_top10(self, top: &[u32]) -> (Cell, Vec<u32>) {
        (self, top.iter().take(10).copied().collect())
    }
}
