//! Exact ground truth: the two-pass ranking computed on full-precision vectors with
//! no partitioning and no quantisation.
//!
//! - **Pass 1:** each document's score is its best chunk, `max_j ⟨q, d_j⟩` (the
//!   production MaxSim with one whole-query vector); keep the top
//!   [`PASS1_K`].
//! - **Final:** rerank those by the exact whole-document `⟨q, d⟩`; keep the top
//!   [`FINAL_K`].
//!
//! ANN recall compares the index's results with these lists, which separates what
//! the index loses from what the embedding model gets wrong.

use std::path::Path;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use simsimd::SpatialSimilarity;

use super::frozen::Frozen;

/// Candidates kept by the exact Pass 1 (production `first_pass_sparse_search_top_k`).
pub(super) const PASS1_K: usize = 1000;
/// Results kept by the exact final ranking.
pub(super) const FINAL_K: usize = 100;

/// Exact lists per query, as doc rows, best first.
#[derive(Serialize, Deserialize)]
pub(super) struct GroundTruth {
    pub pass1: Vec<Vec<u32>>,
    pub final_ranking: Vec<Vec<u32>>,
}

pub(super) fn dot(a: &[f32], b: &[f32]) -> f32 {
    f32::dot(a, b).expect("equal dimensions") as f32
}

/// Exact MaxSim score of every doc for query `q` (`-inf` for a doc with no chunks).
pub(super) fn doc_maxsim(frozen: &Frozen, q: &[f32]) -> Vec<f32> {
    let mut best = vec![f32::NEG_INFINITY; frozen.n_docs()];
    for c in 0..frozen.n_chunks() {
        let d = frozen.chunk_doc[c] as usize;
        let s = dot(q, frozen.chunk(c));
        if s > best[d] {
            best[d] = s;
        }
    }
    best
}

/// Indices of the `k` largest finite scores, best first; ties go to the lower row.
pub(super) fn top_k(scores: &[f32], k: usize) -> Vec<u32> {
    let mut rows: Vec<u32> = (0..scores.len() as u32).filter(|&r| scores[r as usize].is_finite()).collect();
    let cmp = |a: &u32, b: &u32| scores[*b as usize].total_cmp(&scores[*a as usize]).then(a.cmp(b));
    if rows.len() > k {
        rows.select_nth_unstable_by(k, cmp);
        rows.truncate(k);
    }
    rows.sort_by(cmp);
    rows
}

fn compute(frozen: &Frozen) -> GroundTruth {
    let per_query: Vec<(Vec<u32>, Vec<u32>)> = (0..frozen.n_queries())
        .into_par_iter()
        .map(|qi| {
            let q = frozen.query(qi);
            let pass1 = top_k(&doc_maxsim(frozen, q), PASS1_K);
            let mut dense: Vec<f32> = vec![f32::NEG_INFINITY; frozen.n_docs()];
            for &d in &pass1 {
                dense[d as usize] = dot(q, frozen.dense(d as usize));
            }
            let final_ranking = top_k(&dense, FINAL_K);
            (pass1, final_ranking)
        })
        .collect();
    let (pass1, final_ranking) = per_query.into_iter().unzip();
    GroundTruth { pass1, final_ranking }
}

/// Load the cached ground truth from `dir`, computing and caching it first if absent.
pub(super) fn load_or_compute(dir: &Path, frozen: &Frozen) -> GroundTruth {
    let path = dir.join(format!("ground_truth_{}.json", frozen.manifest.split));
    if let Ok(bytes) = std::fs::read(&path) {
        let gt: GroundTruth = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(gt.pass1.len(), frozen.n_queries(), "stale {}", path.display());
        return gt;
    }
    let t = std::time::Instant::now();
    let gt = compute(frozen);
    std::fs::write(&path, serde_json::to_vec(&gt).unwrap()).unwrap();
    eprintln!("  exact ground truth computed in {:.1}s", t.elapsed().as_secs_f64());
    gt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_orders_best_first_and_breaks_ties_by_row() {
        let s = [0.5, f32::NEG_INFINITY, 0.9, 0.5, 0.1];
        assert_eq!(top_k(&s, 3), vec![2, 0, 3]);
        assert_eq!(top_k(&s, 10), vec![2, 0, 3, 4], "infinite scores are never returned");
    }
}
