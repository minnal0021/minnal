//! Pass-1 / Pass-2 study: where the exact Pass-1 candidates that the index misses
//! sit, why they are missed, what that costs, and what the rotation and the
//! estimator's form contribute (`pass1-recall-study.md`, `rabitq-rotation-audit.md`).
//!
//! Works on the frozen embeddings only. Every 1-bit code is rebuilt in memory with
//! the production quantiser (`quantise`), and Pass 1 is scored with the production
//! estimator (`SingleBitQuanDotProductEstimator::estimate_from_parts`), so the
//! simulated candidate list is the one `search()` produces. The last section checks
//! that against `search()` itself over a real index.
//!
//! ```sh
//! MINNAL_BENCH_DATASET=fiqa MINNAL_BENCH_LABEL=study \
//!   cargo test -p minnal_db --all-features --release --lib vector_bench_pass1_study -- --ignored --nocapture
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Arc;

use rayon::prelude::*;

use super::super::beir_eval::{env_or, read_qrels, score};
use super::exact::{self, dot};
use super::frozen::{self, Frozen};
use super::splitmix;
use crate::semantic_search::cluster::{Cluster, find_closest_cluster_id, read_clusters_from_file};
use crate::semantic_search::index::distance_estimator::{MultiBitQuanDotProductEstimator, SingleBitQuanDotProductEstimator};
use crate::semantic_search::index::vector_index::QuantisationStyle;
use crate::semantic_search::quantisation::rabitq::index_embedding_rotated;
use crate::semantic_search::quantisation::rabitq::quantisation_support::pack_bits;
use crate::semantic_search::quantisation::rabitq::quantise;
use crate::semantic_search::service::{SemanticSearchConfig, index_embeddings, search};
use crate::semantic_search::{ClusterIndex, simd};
use crate::vector_kv::{DbVectorStore, upsert_vectors};
use crate::{AsyncDb, DbConfig};

/// Production Pass-1 cut.
const CUT: usize = 1000;
/// Candidate-list depths for the depth curve.
const DEPTHS: [usize; 8] = [250, 500, 1000, 1500, 2000, 3000, 4000, usize::MAX];
/// Exact top-k sizes whose coverage is tracked.
const TOPS: [usize; 5] = [10, 50, 100, 500, 1000];
/// Exact-rank buckets for the missed documents.
const BUCKETS: [(usize, usize); 4] = [(1, 10), (11, 100), (101, 500), (501, 1000)];
/// Every this-many queries contribute every probed chunk to the error statistics.
const ERROR_QUERY_STEP: usize = 8;

/// The rotation codes are computed in.
#[derive(Clone, Copy, PartialEq)]
enum Rot {
    None,
    FhtKac,
    Haar,
}

impl Rot {
    fn name(self) -> &'static str {
        match self {
            Rot::None => "none",
            Rot::FhtKac => "FhtKac (production)",
            Rot::Haar => "dense Haar",
        }
    }
}

/// How Pass 1 scores a chunk.
#[derive(Clone, Copy, PartialEq)]
enum Form {
    /// Production: `⟨q,c⟩ + est⟨r, q⟩` against the raw query.
    Raw,
    /// The paper's form: `⟨q,c⟩ + ⟨r,c⟩ + est⟨r, q − c⟩`, with `⟨r,c⟩` exact.
    Centred,
    /// Production estimate plus the chunk's `error_bound` (optimistic score).
    Optimistic,
}

impl Form {
    fn name(self) -> &'static str {
        match self {
            Form::Raw => "raw q (production)",
            Form::Centred => "q − c (paper)",
            Form::Optimistic => "est + error_bound",
        }
    }
}

/// Codes built under one rotation, with that rotation's centroids.
type RotCodes = (Rot, Vec<Code>, HashMap<u32, Vec<f32>>);

/// One chunk's 1-bit code under one rotation.
struct Code {
    cluster: u32,
    packed: Vec<u64>,
    scaling: f32,
    bound: f32,
    /// `⟨x − c, c⟩`, exact (original space).
    rc: f32,
}

/// A dense Haar-random orthogonal matrix (QR of a Gaussian), rows of `Pᵀ`.
fn haar(dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut s = seed;
    let mut gauss = || {
        let u1 = ((splitmix(&mut s) >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
        let u2 = (splitmix(&mut s) >> 11) as f64 / (1u64 << 53) as f64;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };
    let mut rows: Vec<Vec<f64>> = (0..dim).map(|_| (0..dim).map(|_| gauss()).collect()).collect();
    // Modified Gram–Schmidt, twice for numerical orthogonality.
    for _ in 0..2 {
        for i in 0..dim {
            let (done, rest) = rows.split_at_mut(i);
            let r = &mut rest[0];
            for q in done.iter() {
                let p: f64 = r.iter().zip(q).map(|(a, b)| a * b).sum();
                r.iter_mut().zip(q).for_each(|(a, b)| *a -= p * b);
            }
            let n = r.iter().map(|a| a * a).sum::<f64>().sqrt();
            r.iter_mut().for_each(|a| *a /= n);
        }
    }
    rows.into_iter().map(|r| r.into_iter().map(|v| v as f32).collect()).collect()
}

struct Rotator<'a> {
    rot: Rot,
    index: &'a ClusterIndex,
    haar: Option<&'a [Vec<f32>]>,
}

impl Rotator<'_> {
    fn apply(&self, v: &[f32]) -> Vec<f32> {
        match self.rot {
            Rot::None => v.to_vec(),
            Rot::FhtKac => self.index.rotate(v),
            Rot::Haar => self.haar.unwrap().iter().map(|row| dot(row, v)).collect(),
        }
    }
}

fn build_codes(frozen: &Frozen, index: &ClusterIndex, rotator: &Rotator, rotated_centroids: &HashMap<u32, Vec<f32>>) -> Vec<Code> {
    (0..frozen.n_chunks())
        .into_par_iter()
        .map(|c| {
            let x = frozen.chunk(c);
            let cluster = find_closest_cluster_id(&index.clusters, x);
            let centroid = &index.clusters[&cluster].centroid;
            let q = quantise(&rotator.apply(x), &rotated_centroids[&cluster], 1);
            let rc = x.iter().zip(centroid).map(|(a, b)| (a - b) * b).sum();
            Code {
                cluster,
                packed: pack_bits(&q.quantised_embedding),
                scaling: q.scaling_factor,
                bound: q.error_bound,
                rc,
            }
        })
        .collect()
}

/// Rank position (0-based) of every doc under `scores` (higher first, ties by
/// row); `usize::MAX` for a doc with no score.
fn ranks(scores: &[f32]) -> (Vec<u32>, Vec<usize>) {
    let order = exact::top_k(scores, scores.len());
    let mut rank = vec![usize::MAX; scores.len()];
    for (i, &d) in order.iter().enumerate() {
        rank[d as usize] = i;
    }
    (order, rank)
}

/// Accumulated per-variant numbers (sums over queries).
#[derive(Default, Clone)]
struct Acc {
    queries: usize,
    /// Share of the exact top-k inside the top `CUT` (summed).
    top_recall: [f64; TOPS.len()],
    /// Exact top-1000 recall at each depth.
    depth_recall: [f64; DEPTHS.len()],
    /// Exact top-10 / top-100 recall at each depth.
    depth_recall10: [f64; DEPTHS.len()],
    depth_recall100: [f64; DEPTHS.len()],
    /// Missed exact-top-1000 docs by exact rank bucket.
    miss_bucket: [usize; BUCKETS.len()],
    /// Miss causes: not probed, best chunk not probed, near tie, estimator.
    causes: [usize; 4],
    misses: usize,
    /// Approximate rank of the missed docs (for the ones with a score).
    miss_approx_rank: Vec<usize>,
    /// qrels recall inside the top `CUT`.
    cand_recall: f64,
    /// nDCG@10 after exact dense rerank of the top `CUT`, and of the top 4000.
    ndcg_exact_rerank: f64,
    ndcg_exact_rerank_4000: f64,
}

#[derive(Default, Clone)]
struct ErrAcc {
    n: usize,
    sum: f64,
    sum_sq: f64,
    within: usize,
    within_scaled: usize,
}

impl ErrAcc {
    fn add(&mut self, err: f32, bound: f32, scaled_bound: f32) {
        self.n += 1;
        self.sum += err as f64;
        self.sum_sq += (err as f64).powi(2);
        self.within += (err.abs() <= bound) as usize;
        self.within_scaled += (err.abs() <= scaled_bound) as usize;
    }
    fn merge(&mut self, o: &ErrAcc) {
        self.n += o.n;
        self.sum += o.sum;
        self.sum_sq += o.sum_sq;
        self.within += o.within;
        self.within_scaled += o.within_scaled;
    }
}

struct QueryOut {
    accs: Vec<Acc>,
    err_raw: ErrAcc,
    err_centred: ErrAcc,
    /// (Σ‖q − c‖ over probed clusters, count), and ‖q‖.
    qc_norm: (f64, usize),
    q_norm: f64,
    ndcg: [f64; 5],
    cand_recall_exact: f64,
    rel_exact_rank: Vec<usize>,
    pass2_top10_overlap: f64,
    pass2_top10_identical: bool,
    /// The production variant's top-`CUT` doc set, for the `search()` check.
    prod_cut: Vec<u32>,
    prod_final10: Vec<u32>,
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn vector_bench_pass1_study() {
    let dataset = env_or("MINNAL_BENCH_DATASET", "scifact");
    let model = env_or("MINNAL_BENCH_MODEL", "gemma");
    let label = env_or("MINNAL_BENCH_LABEL", "study");
    let bench_root = PathBuf::from(env_or("MINNAL_BENCH_ROOT", "../work/bench"));
    let beir_root = PathBuf::from(env_or("MINNAL_BEIR_ROOT", "../work/beir"));
    let split = env_or("MINNAL_BEIR_SPLIT", "test");
    let config = SemanticSearchConfig {
        model_name: model.clone(),
        ..SemanticSearchConfig::default()
    };
    let dataset_dir = beir_root.join(&dataset);
    let frozen_dir = bench_root.join(&dataset).join(&model);
    assert!(frozen_dir.join("manifest.json").exists(), "run vector_bench for {dataset}/{model} first");
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
    let raw = read_clusters_from_file(&format!("../service/embedding_support/{model}/clusters.json")).unwrap();
    let index = Arc::new(ClusterIndex::from_clusters(
        raw.into_iter().map(|(id, c)| (id, Cluster::new(id, c))).collect(),
    ));
    let (nd, nq, dim) = (frozen.n_docs(), frozen.n_queries(), frozen.dim());
    eprintln!(
        "\n=== pass-1 study: {dataset} ({model}), {nd} docs, {} chunks, {nq} queries ===",
        frozen.n_chunks()
    );

    // ── Codes under each rotation ──
    let haar_p = haar(dim, 0x5eed_4a11);
    let mut codes: Vec<RotCodes> = Vec::new();
    for rot in [Rot::FhtKac, Rot::None, Rot::Haar] {
        let rotator = Rotator {
            rot,
            index: &index,
            haar: Some(&haar_p),
        };
        let rc: HashMap<u32, Vec<f32>> = index.clusters.iter().map(|(&id, c)| (id, rotator.apply(&c.centroid))).collect();
        let t = std::time::Instant::now();
        let built = build_codes(&frozen, &index, &rotator, &rc);
        eprintln!("  codes ({}) in {:.1}s", rot.name(), t.elapsed().as_secs_f64());
        codes.push((rot, built, rc));
    }
    // FhtKac codes must be exactly what production writes.
    for c in (0..frozen.n_chunks()).step_by(997) {
        let vi = index_embedding_rotated(&index, frozen.chunk(c), QuantisationStyle::SingleBit).unwrap();
        let mine = &codes[0].1[c];
        assert_eq!(
            (vi.cluster_id, &vi.packed_vector, vi.scaling_factor),
            (mine.cluster, &mine.packed, mine.scaling)
        );
    }
    // Production dense (Pass 2) codes.
    let dense_bits = config.number_of_bits_for_dense_quantisation;
    let dense_codes: Vec<_> = (0..nd)
        .into_par_iter()
        .map(|d| index_embedding_rotated(&index, frozen.dense(d), QuantisationStyle::MultiBit { number_of_bits: dense_bits }).unwrap())
        .collect();
    let mut cluster_chunks: HashMap<u32, Vec<usize>> = HashMap::new();
    for (c, code) in codes[0].1.iter().enumerate() {
        cluster_chunks.entry(code.cluster).or_default().push(c);
    }

    // Variants: (rotation index into `codes`, form, n_probes).
    let mut variants: Vec<(usize, Form, usize)> = vec![(0, Form::Raw, 16), (0, Form::Raw, 64), (0, Form::Raw, 256)];
    variants.extend([(1, Form::Raw, 64), (2, Form::Raw, 64), (0, Form::Centred, 64), (0, Form::Optimistic, 64)]);
    let prod_variant = 1;

    let per_query: Vec<QueryOut> = (0..nq)
        .into_par_iter()
        .map(|qi| {
            let q = frozen.query(qi);
            let rels = &qrels[&frozen.manifest.query_ids[qi]];
            let ids = |list: &[u32]| -> Vec<&str> { list.iter().map(|&d| frozen.manifest.doc_ids[d as usize].as_str()).collect() };
            // Exact chunk scores, exact MaxSim, exact best chunk per doc.
            let chunk_exact: Vec<f32> = (0..frozen.n_chunks()).map(|c| dot(q, frozen.chunk(c))).collect();
            let mut doc_exact = vec![f32::NEG_INFINITY; nd];
            let mut best_chunk = vec![usize::MAX; nd];
            for (c, &s) in chunk_exact.iter().enumerate() {
                let d = frozen.chunk_doc[c] as usize;
                if s > doc_exact[d] {
                    doc_exact[d] = s;
                    best_chunk[d] = c;
                }
            }
            let (exact_order, exact_rank) = ranks(&doc_exact);
            let exact_top: &[u32] = &gt.pass1[qi];
            debug_assert_eq!(&exact_order[..exact_top.len()], exact_top);
            let s_cut = doc_exact[exact_top[exact_top.len() - 1] as usize];
            let dense_exact: Vec<f32> = (0..nd).map(|d| dot(q, frozen.dense(d))).collect();
            let rerank_exact = |cands: &[u32], k: usize| -> Vec<u32> {
                let mut s = vec![f32::NEG_INFINITY; nd];
                for &d in cands {
                    s[d as usize] = dense_exact[d as usize];
                }
                exact::top_k(&s, k)
            };

            let mut accs = vec![Acc::default(); variants.len()];
            let (mut err_raw, mut err_centred) = (ErrAcc::default(), ErrAcc::default());
            let mut qc_norm = (0f64, 0usize);
            let mut prod_cut = Vec::new();
            let mut prod_final10 = Vec::new();
            let mut pass2_top10_overlap = 0.0;
            let mut pass2_top10_identical = false;
            let probes: HashMap<usize, Vec<u32>> = [16usize, 64, 256]
                .iter()
                .map(|&n| (n, index.find_top_n_cluster_ids_batch(&[q.to_vec()], n.min(index.len())).remove(0)))
                .collect();

            for (vi, &(ri, form, n_probes)) in variants.iter().enumerate() {
                let (rot, rcodes, rcentroids) = (&codes[ri].0, &codes[ri].1, &codes[ri].2);
                let rotator = Rotator {
                    rot: *rot,
                    index: &index,
                    haar: Some(&haar_p),
                };
                let rq = rotator.apply(q);
                let sum = SingleBitQuanDotProductEstimator::query_sum(&rq);
                let probed = &probes[&n_probes];
                let probed_set: HashSet<u32> = probed.iter().copied().collect();
                let mut doc_approx = vec![f32::NEG_INFINITY; nd];
                let track_err = ri == 0 && n_probes == 64 && form != Form::Optimistic && qi % ERROR_QUERY_STEP == 0;
                for &cl in probed {
                    let centroid = &index.clusters[&cl].centroid;
                    let est = SingleBitQuanDotProductEstimator::with_query_sum(cl, q, centroid, sum);
                    // Centred form: the query residual in the code's space.
                    let (rqc, sum_c, qc_n) = if form == Form::Centred {
                        let v: Vec<f32> = rq.iter().zip(&rcentroids[&cl]).map(|(a, b)| a - b).collect();
                        let s: f32 = v.iter().sum();
                        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                        (v, s, n)
                    } else {
                        (
                            Vec::new(),
                            0.0,
                            q.iter().zip(centroid).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt(),
                        )
                    };
                    if vi == prod_variant {
                        qc_norm.0 += qc_n as f64;
                        qc_norm.1 += 1;
                    }
                    for &c in cluster_chunks.get(&cl).map(Vec::as_slice).unwrap_or(&[]) {
                        let code = &rcodes[c];
                        let score = match form {
                            Form::Raw => est.estimate_from_parts(&rq, &code.packed, code.scaling),
                            Form::Optimistic => est.estimate_from_parts(&rq, &code.packed, code.scaling) + code.bound,
                            Form::Centred => {
                                let ip = simd::packed_ip_best(&code.packed, &rqc, dim);
                                est.0.query_to_centroid_dot_product + code.rc + code.scaling * (2.0 * ip - sum_c)
                            }
                        };
                        if track_err {
                            let err = score - chunk_exact[c];
                            match form {
                                Form::Raw => err_raw.add(err, code.bound, code.bound * q.iter().map(|x| x * x).sum::<f32>().sqrt()),
                                Form::Centred => err_centred.add(err, code.bound, code.bound * qc_n),
                                Form::Optimistic => {}
                            }
                        }
                        let d = frozen.chunk_doc[c] as usize;
                        if score > doc_approx[d] {
                            doc_approx[d] = score;
                        }
                    }
                }
                let (approx_order, approx_rank) = ranks(&doc_approx);
                let cut: Vec<u32> = approx_order.iter().take(CUT).copied().collect();
                let cut_set: HashSet<u32> = cut.iter().copied().collect();
                let acc = &mut accs[vi];
                acc.queries = 1;
                for (i, &k) in TOPS.iter().enumerate() {
                    let k = k.min(exact_top.len());
                    acc.top_recall[i] = exact_top[..k].iter().filter(|d| cut_set.contains(d)).count() as f64 / k.max(1) as f64;
                }
                for (i, &depth) in DEPTHS.iter().enumerate() {
                    let within = |k: usize| {
                        let k = k.min(exact_top.len());
                        exact_top[..k].iter().filter(|&&d| approx_rank[d as usize] < depth).count() as f64 / k.max(1) as f64
                    };
                    acc.depth_recall[i] = within(1000);
                    acc.depth_recall10[i] = within(10);
                    acc.depth_recall100[i] = within(100);
                }
                for &d in exact_top.iter().filter(|d| !cut_set.contains(d)) {
                    let du = d as usize;
                    acc.misses += 1;
                    let r = exact_rank[du] + 1;
                    if let Some(b) = BUCKETS.iter().position(|&(lo, hi)| r >= lo && r <= hi) {
                        acc.miss_bucket[b] += 1;
                    }
                    let chunks = frozen.chunk_rows(du);
                    let any_probed = chunks.clone().any(|c| probed_set.contains(&rcodes[c].cluster));
                    let best_probed = probed_set.contains(&rcodes[best_chunk[du]].cluster);
                    let cause = if !any_probed {
                        0
                    } else if !best_probed {
                        1
                    } else if doc_exact[du] - s_cut <= rcodes[best_chunk[du]].bound {
                        2
                    } else {
                        3
                    };
                    acc.causes[cause] += 1;
                    if approx_rank[du] != usize::MAX {
                        acc.miss_approx_rank.push(approx_rank[du] + 1);
                    }
                }
                acc.cand_recall = score(&ids(&cut), rels).cand_recall;
                acc.ndcg_exact_rerank = score(&ids(&rerank_exact(&cut, 10)), rels).ndcg;
                let deep: Vec<u32> = approx_order.iter().take(4000).copied().collect();
                acc.ndcg_exact_rerank_4000 = score(&ids(&rerank_exact(&deep, 10)), rels).ndcg;

                if vi == prod_variant {
                    // Pass 2 as production runs it: 8-bit codes, rotated query.
                    let dsum = MultiBitQuanDotProductEstimator::scaled_query_sum(&rq, dense_bits);
                    let mut s = vec![f32::NEG_INFINITY; nd];
                    for &d in &cut {
                        let dc = &dense_codes[d as usize];
                        let centroid = &index.clusters[&dc.cluster_id].centroid;
                        let e = MultiBitQuanDotProductEstimator::with_scaled_query_sum(dc.cluster_id, q, centroid, dsum);
                        s[d as usize] = e.estimate_from_parts(&rq, &dc.packed_vector, dc.addition_factor, dc.scaling_factor);
                    }
                    let quant10 = exact::top_k(&s, 10);
                    let exact10 = rerank_exact(&cut, 10);
                    let ex: HashSet<u32> = exact10.iter().copied().collect();
                    pass2_top10_overlap = quant10.iter().filter(|d| ex.contains(d)).count() as f64 / ex.len().max(1) as f64;
                    pass2_top10_identical = quant10 == exact10;
                    prod_cut = cut.clone();
                    prod_final10 = quant10;
                }
            }

            // Pipelines on exact scores.
            let all_docs: Vec<u32> = (0..nd as u32).collect();
            let ndcg = [
                score(&ids(&prod_final10), rels).ndcg,
                score(&ids(&gt.final_ranking[qi][..10.min(gt.final_ranking[qi].len())]), rels).ndcg,
                score(&ids(&rerank_exact(&all_docs, 10)), rels).ndcg,
                score(&ids(&exact_order[..10]), rels).ndcg,
                score(&ids(&rerank_exact(&exact_order[..4000.min(exact_order.len())], 10)), rels).ndcg,
            ];
            let cand_recall_exact = score(&ids(exact_top), rels).cand_recall;
            let rel_exact_rank = rels
                .keys()
                .filter_map(|id| frozen.manifest.doc_ids.iter().position(|x| x == id))
                .map(|d| exact_rank[d].saturating_add(1))
                .collect();
            QueryOut {
                accs,
                err_raw,
                err_centred,
                qc_norm,
                q_norm: (q.iter().map(|x| x * x).sum::<f32>() as f64).sqrt(),
                ndcg,
                cand_recall_exact,
                rel_exact_rank,
                pass2_top10_overlap,
                pass2_top10_identical,
                prod_cut,
                prod_final10,
            }
        })
        .collect();

    // ── Check the simulation against search() over a real index ──
    let tmp = tempfile::TempDir::new().unwrap();
    let db = Arc::new(AsyncDb::open_with_config(tmp.path().to_owned(), DbConfig::default()).await.unwrap());
    db.namespace("study".to_string()).await.unwrap();
    for d in 0..nd {
        let vis = index_embeddings(&config, &index, frozen.dense(d), &frozen.doc_chunks(d)).unwrap();
        upsert_vectors(&db, "study", &(d as u64).to_be_bytes(), "", &vis).await.unwrap();
    }
    let store = DbVectorStore::new(&db, "study").await.unwrap();
    let cfg = SemanticSearchConfig {
        n_probes: 64,
        first_pass_sparse_search_top_k: CUT,
        ..config.clone()
    };
    let (mut same_set, mut same_top10, mut set_overlap) = (0usize, 0usize, 0f64);
    for (qi, out) in per_query.iter().enumerate() {
        let q = frozen.query(qi).to_vec();
        let res = search(
            &cfg,
            "study",
            &index,
            std::slice::from_ref(&q),
            &q,
            &store,
            None::<fn(&[u8]) -> bool>,
            Some(CUT),
        )
        .await
        .unwrap();
        let got: Vec<u32> = res
            .iter()
            .map(|r| u64::from_be_bytes(r.document_id[..].try_into().unwrap()) as u32)
            .collect();
        let a: HashSet<u32> = got.iter().copied().collect();
        let b: HashSet<u32> = out.prod_cut.iter().copied().collect();
        same_set += (a == b) as usize;
        set_overlap += a.intersection(&b).count() as f64 / b.len().max(1) as f64;
        same_top10 += (got.iter().take(10).copied().collect::<Vec<_>>() == out.prod_final10) as usize;
    }

    // ── Report ──
    let n = nq as f64;
    let mut md = String::new();
    let _ = writeln!(md, "# Pass-1 study — {dataset} ({model})\n");
    let _ = writeln!(
        md,
        "{nd} docs, {} chunks, {nq} queries. Pass-1 cut {CUT}. Commit `{}`.\n",
        frozen.n_chunks(),
        super::git_describe()
    );
    let _ = writeln!(
        md,
        "Simulation vs `search()` (n_probes 64): identical candidate set {same_set}/{nq}, mean set overlap {:.4}, identical final top 10 {same_top10}/{nq}.\n",
        set_overlap / n
    );
    let vname = |&(ri, form, np): &(usize, Form, usize)| format!("{} / {} / {np}", codes[ri].0.name(), form.name());
    let merged: Vec<Acc> = (0..variants.len())
        .map(|vi| {
            let mut a = Acc::default();
            for o in &per_query {
                let x = &o.accs[vi];
                a.queries += x.queries;
                for i in 0..TOPS.len() {
                    a.top_recall[i] += x.top_recall[i];
                }
                for i in 0..DEPTHS.len() {
                    a.depth_recall[i] += x.depth_recall[i];
                    a.depth_recall10[i] += x.depth_recall10[i];
                    a.depth_recall100[i] += x.depth_recall100[i];
                }
                for i in 0..BUCKETS.len() {
                    a.miss_bucket[i] += x.miss_bucket[i];
                }
                for i in 0..4 {
                    a.causes[i] += x.causes[i];
                }
                a.misses += x.misses;
                a.miss_approx_rank.extend(&x.miss_approx_rank);
                a.cand_recall += x.cand_recall;
                a.ndcg_exact_rerank += x.ndcg_exact_rerank;
                a.ndcg_exact_rerank_4000 += x.ndcg_exact_rerank_4000;
            }
            a
        })
        .collect();

    let _ = writeln!(md, "## Exact top-k kept in the top {CUT} (rotation / form / n_probes)\n");
    let _ = writeln!(
        md,
        "| Variant | {} |",
        TOPS.iter().map(|k| format!("top {k}")).collect::<Vec<_>>().join(" | ")
    );
    let _ = writeln!(md, "|---|{}", "---:|".repeat(TOPS.len()));
    for (v, a) in variants.iter().zip(&merged) {
        let _ = writeln!(
            md,
            "| {} | {} |",
            vname(v),
            a.top_recall.iter().map(|x| format!("{:.4}", x / n)).collect::<Vec<_>>().join(" | ")
        );
    }
    let _ = writeln!(md, "\n## Missed exact top-{CUT} docs: exact rank and cause\n");
    let _ = writeln!(
        md,
        "| Variant | Misses/query | rank 1–10 | 11–100 | 101–500 | 501–1000 | not probed | best chunk not probed | near tie | estimator | approx rank p50 / p90 / p99 |"
    );
    let _ = writeln!(md, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|");
    for (v, a) in variants.iter().zip(&merged) {
        let m = a.misses.max(1) as f64;
        let mut r = a.miss_approx_rank.clone();
        r.sort_unstable();
        let p = |f: f64| r.get(((r.len().max(1) - 1) as f64 * f) as usize).copied().unwrap_or(0);
        let _ = writeln!(
            md,
            "| {} | {:.1} | {:.1}% | {:.1}% | {:.1}% | {:.1}% | {:.1}% | {:.1}% | {:.1}% | {:.1}% | {} / {} / {} |",
            vname(v),
            a.misses as f64 / n,
            a.miss_bucket[0] as f64 / m * 100.0,
            a.miss_bucket[1] as f64 / m * 100.0,
            a.miss_bucket[2] as f64 / m * 100.0,
            a.miss_bucket[3] as f64 / m * 100.0,
            a.causes[0] as f64 / m * 100.0,
            a.causes[1] as f64 / m * 100.0,
            a.causes[2] as f64 / m * 100.0,
            a.causes[3] as f64 / m * 100.0,
            p(0.5),
            p(0.9),
            p(0.99)
        );
    }
    let _ = writeln!(md, "\n## Depth: exact top-k recall at Pass-1 depth (∞ = every probed doc)\n");
    let dh = DEPTHS
        .iter()
        .map(|&d| if d == usize::MAX { "∞".to_string() } else { d.to_string() })
        .collect::<Vec<_>>();
    let _ = writeln!(md, "| Variant | exact top | {} |", dh.join(" | "));
    let _ = writeln!(md, "|---|---:|{}", "---:|".repeat(DEPTHS.len()));
    for (v, a) in variants.iter().zip(&merged) {
        for (k, arr) in [(10, &a.depth_recall10), (100, &a.depth_recall100), (1000, &a.depth_recall)] {
            let _ = writeln!(
                md,
                "| {} | {k} | {} |",
                vname(v),
                arr.iter().map(|x| format!("{:.4}", x / n)).collect::<Vec<_>>().join(" | ")
            );
        }
    }
    let _ = writeln!(md, "\n## Relevance\n");
    let _ = writeln!(
        md,
        "| Variant | qrels recall in top {CUT} | nDCG@10, exact rerank of top {CUT} | of top 4000 |"
    );
    let _ = writeln!(md, "|---|---:|---:|---:|");
    let _ = writeln!(
        md,
        "| exact Pass 1 | {:.4} | {:.4} | {:.4} |",
        per_query.iter().map(|o| o.cand_recall_exact).sum::<f64>() / n,
        per_query.iter().map(|o| o.ndcg[1]).sum::<f64>() / n,
        per_query.iter().map(|o| o.ndcg[4]).sum::<f64>() / n
    );
    for (v, a) in variants.iter().zip(&merged) {
        let _ = writeln!(
            md,
            "| {} | {:.4} | {:.4} | {:.4} |",
            vname(v),
            a.cand_recall / n,
            a.ndcg_exact_rerank / n,
            a.ndcg_exact_rerank_4000 / n
        );
    }
    let mean = |i: usize| per_query.iter().map(|o| o.ndcg[i]).sum::<f64>() / n;
    let _ = writeln!(md, "\n| Pipeline | nDCG@10 |\n|---|---:|");
    for (name, i) in [
        ("production simulation (64 probes, 1-bit cut 1000, 8-bit rerank)", 0),
        ("exact two-pass (exact MaxSim top 1000, exact dense rerank)", 1),
        ("exact two-pass, cut 4000", 4),
        ("exact dense over every doc (no Pass 1)", 2),
        ("exact MaxSim only (no Pass 2)", 3),
    ] {
        let _ = writeln!(md, "| {name} | {:.4} |", mean(i));
    }
    let mut rel_ranks: Vec<usize> = per_query.iter().flat_map(|o| o.rel_exact_rank.iter().copied()).collect();
    rel_ranks.sort_unstable();
    let share = |lo: usize, hi: usize| rel_ranks.iter().filter(|&&r| r >= lo && r <= hi).count() as f64 / rel_ranks.len().max(1) as f64 * 100.0;
    let _ = writeln!(
        md,
        "\nRelevant docs by exact MaxSim rank ({} judged): 1–10 {:.1}%, 11–100 {:.1}%, 101–1000 {:.1}%, 1001–4000 {:.1}%, beyond {:.1}%.",
        rel_ranks.len(),
        share(1, 10),
        share(11, 100),
        share(101, 1000),
        share(1001, 4000),
        share(4001, usize::MAX)
    );
    let _ = writeln!(
        md,
        "\nPass 2 alone (same candidates, 8-bit vs exact dense): top-10 overlap {:.4}, identical top 10 {}/{nq}.",
        per_query.iter().map(|o| o.pass2_top10_overlap).sum::<f64>() / n,
        per_query.iter().filter(|o| o.pass2_top10_identical).count()
    );
    let (mut er, mut ec) = (ErrAcc::default(), ErrAcc::default());
    for o in &per_query {
        er.merge(&o.err_raw);
        ec.merge(&o.err_centred);
    }
    let qc: (f64, usize) = per_query.iter().fold((0.0, 0), |a, o| (a.0 + o.qc_norm.0, a.1 + o.qc_norm.1));
    let _ = writeln!(
        md,
        "\n## Pass-1 estimator over every probed chunk (FhtKac, 64 probes, every {ERROR_QUERY_STEP}th query)\n"
    );
    let _ = writeln!(
        md,
        "Mean ‖q‖ {:.4}; mean ‖q − c‖ over probed clusters {:.4}.\n",
        per_query.iter().map(|o| o.q_norm).sum::<f64>() / n,
        qc.0 / qc.1.max(1) as f64
    );
    let _ = writeln!(
        md,
        "| Form | Pairs | RMSE | Bias | Within error_bound | Within error_bound × query norm |\n|---|---:|---:|---:|---:|---:|"
    );
    for (name, e) in [("raw q (production)", &er), ("q − c (paper)", &ec)] {
        let m = e.n.max(1) as f64;
        let _ = writeln!(
            md,
            "| {name} | {} | {:.5} | {:+.5} | {:.1}% | {:.1}% |",
            e.n,
            (e.sum_sq / m).sqrt(),
            e.sum / m,
            e.within as f64 / m * 100.0,
            e.within_scaled as f64 / m * 100.0
        );
    }
    eprintln!("{md}");
    let out = bench_root.join("results").join(&label);
    std::fs::create_dir_all(&out).unwrap();
    std::fs::write(out.join(format!("{dataset}-{model}-study.md")), &md).unwrap();
}

/// Design-doc audit E1: how close FhtKac's output is to a Haar rotation's. A Haar
/// rotation sends every unit vector to a uniform point on the sphere, whose
/// coordinates (×√D) are close to N(0, 1): excess kurtosis ≈ 0 and a maximum of a
/// few units. A single Hadamard transform of a basis vector is the worst case
/// (every coordinate ±1, excess kurtosis −2). Measured on basis vectors and on
/// real residuals, for FhtKac (4 rounds), two FhtKacs composed (8 rounds) and a
/// dense Haar matrix.
#[test]
#[ignore]
fn vector_bench_rotation_spread() {
    use crate::semantic_search::quantisation::rotation::FhtKacRotator;
    let dim = 768;
    let one = FhtKacRotator::new(dim, crate::semantic_search::cluster::ROTATION_SEED);
    let two = FhtKacRotator::new(dim, 0x5eed_0002);
    let haar_p = haar(dim, 0x5eed_4a11);
    let stats = |vs: &[Vec<f32>]| {
        let (mut m2, mut m4, mut mx, mut n) = (0f64, 0f64, 0f64, 0usize);
        for v in vs {
            for &x in v {
                let z = x as f64 * (dim as f64).sqrt();
                m2 += z * z;
                m4 += z.powi(4);
                mx = mx.max(z.abs());
                n += 1;
            }
        }
        let (m2, m4) = (m2 / n as f64, m4 / n as f64);
        (m4 / (m2 * m2) - 3.0, mx)
    };
    let basis: Vec<Vec<f32>> = (0..dim)
        .map(|i| {
            let mut e = vec![0f32; dim];
            e[i] = 1.0;
            e
        })
        .collect();
    let mut md = String::from("| Input | Rotation | excess kurtosis (0 = Gaussian) | max |coord|·√D |\n|---|---|---:|---:|\n");
    for (name, inputs) in [("basis vectors", &basis)] {
        type RotFn<'a> = Box<dyn Fn(&[f32]) -> Vec<f32> + 'a>;
        let runs: [(&str, RotFn); 4] = [
            ("none", Box::new(|v: &[f32]| v.to_vec())),
            (
                "FhtKac, 4 rounds",
                Box::new(|v: &[f32]| {
                    let mut x = v.to_vec();
                    one.rotate_inplace(&mut x);
                    x
                }),
            ),
            (
                "FhtKac twice, 8 rounds",
                Box::new(|v: &[f32]| {
                    let mut x = v.to_vec();
                    one.rotate_inplace(&mut x);
                    two.rotate_inplace(&mut x);
                    x
                }),
            ),
            ("dense Haar", Box::new(|v: &[f32]| haar_p.iter().map(|row| dot(row, v)).collect())),
        ];
        for (rname, f) in runs.iter() {
            let out: Vec<Vec<f32>> = inputs.iter().map(|v| f(v)).collect();
            let (k, mx) = stats(&out);
            let _ = writeln!(md, "| {name} | {rname} | {k:+.3} | {mx:.2} |");
        }
    }
    eprintln!("\n{md}");
}
