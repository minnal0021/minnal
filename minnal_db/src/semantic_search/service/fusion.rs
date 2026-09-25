//! Final-ranking fusion of the two search passes.
//!
//! Pass 1 scores every candidate with ColBERT MaxSim over 1-bit chunk embeddings
//! (passage-level evidence); Pass 2 scores the same candidates with the 8-bit
//! whole-document dense embedding (document-level evidence). [`fuse`] combines the
//! two into the final order according to [`RankingParams`]:
//!
//! | [`RankFusion`] | Final score |
//! |---|---|
//! | `Dense`  | dense dot product only (Pass 1 only selects candidates) |
//! | `Sparse` | MaxSim only (ablation) |
//! | `Rrf`    | `w/(k + rank_sparse) + (1−w)/(k + rank_dense)` — reciprocal rank fusion |
//! | `Zscore` | `w·z(sparse) + (1−w)·z(dense)` — score fusion that keeps score gaps |
//!
//! Ranks and z-scores are computed over the **whole candidate set** (every doc that
//! survived Pass 1 and has a valid dense entry), not just the dense top-k, so a
//! document ColBERT ranks highly can overtake one the dense pass alone preferred.
//! `w` is [`RankingParams::sparse_weight`]; it exists because the two signals are
//! not equally reliable — the 1-bit MaxSim is noisier than the 8-bit dense score,
//! and chunks in unprobed clusters contribute 0 to it.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::semantic_search::index::vector_index::QueryResult;

/// How the final result order is derived from the two search passes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RankFusion {
    /// Rank by the Pass-2 dense score only; Pass 1 only selects candidates.
    #[default]
    Dense,
    /// Rank by the Pass-1 MaxSim score only.
    Sparse,
    /// Reciprocal rank fusion of the sparse and dense rankings.
    Rrf,
    /// Weighted sum of the per-candidate-set z-scores of the two scores.
    Zscore,
}

/// Default RRF smoothing constant — the BEIR-tuned value (SciFact train + NFCorpus
/// dev, see `semantic_search/report.md`). The textbook `k = 60` with `w = 0.5`
/// measured 2–4.5 nDCG@10 points *worse* than `Dense` here.
pub const DEFAULT_RRF_K: f32 = 1.0;
/// Default weight of the sparse signal in `Rrf` / `Zscore` fusion — BEIR-tuned
/// (see [`DEFAULT_RRF_K`]). The 1-bit MaxSim signal is much weaker than the dense
/// score on the evaluated corpora, so it only breaks near-ties.
pub const DEFAULT_SPARSE_WEIGHT: f32 = 0.1;

/// Parameters for [`fuse`]. Validate with [`RankingParams::validate`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RankingParams {
    /// Fusion mode.
    pub mode: RankFusion,
    /// RRF smoothing constant `k` (> 0). Larger values flatten the gap between
    /// top and lower ranks. Used only by `Rrf`.
    pub rrf_k: f32,
    /// Weight `w ∈ [0, 1]` of the sparse (MaxSim) signal; the dense signal gets
    /// `1 − w`. Used by `Rrf` and `Zscore`.
    pub sparse_weight: f32,
}

impl Default for RankingParams {
    fn default() -> Self {
        Self {
            mode: RankFusion::default(),
            rrf_k: DEFAULT_RRF_K,
            sparse_weight: DEFAULT_SPARSE_WEIGHT,
        }
    }
}

/// An invalid [`RankingParams`] value.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum RankingError {
    /// `rrf_k` was not a finite number greater than 0.
    #[error("rrf_k must be a finite number > 0, got {0}")]
    InvalidRrfK(f32),
    /// `sparse_weight` was outside `[0, 1]`.
    #[error("sparse_weight must be within [0, 1], got {0}")]
    InvalidSparseWeight(f32),
}

impl RankingParams {
    /// Check that `rrf_k` is finite and positive and `sparse_weight` is in `[0, 1]`.
    pub fn validate(&self) -> Result<(), RankingError> {
        if !(self.rrf_k.is_finite() && self.rrf_k > 0.0) {
            return Err(RankingError::InvalidRrfK(self.rrf_k));
        }
        if !(0.0..=1.0).contains(&self.sparse_weight) {
            return Err(RankingError::InvalidSparseWeight(self.sparse_weight));
        }
        Ok(())
    }

    /// Apply a per-request override on top of these (server-configured) params and
    /// validate the result.
    pub fn with_override(&self, o: &RankingOverride) -> Result<Self, RankingError> {
        let p = Self {
            mode: o.mode.unwrap_or(self.mode),
            rrf_k: o.rrf_k.unwrap_or(self.rrf_k),
            sparse_weight: o.sparse_weight.unwrap_or(self.sparse_weight),
        };
        p.validate()?;
        Ok(p)
    }
}

/// A per-request, partial override of the configured [`RankingParams`]. Every
/// field is optional; an absent field keeps the configured value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RankingOverride {
    /// Fusion mode override.
    pub mode: Option<RankFusion>,
    /// `rrf_k` override.
    pub rrf_k: Option<f32>,
    /// `sparse_weight` override.
    pub sparse_weight: Option<f32>,
}

/// A search candidate carrying both pass scores, ready for fusion.
#[derive(Clone, Debug)]
pub(crate) struct Candidate {
    pub document_id: Vec<u8>,
    /// Pass-1 ColBERT MaxSim score.
    pub sparse_score: f32,
    /// Pass-2 dense dot-product estimate.
    pub dense_score: f32,
    /// Error bound of the dense estimate.
    pub error_bound: f32,
}

/// 1-based ranks of `scores` (descending), ties broken by `ids` ascending.
fn ranks_desc(scores: &[f32], ids: &[&[u8]]) -> Vec<u32> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]).then_with(|| ids[a].cmp(ids[b])));
    let mut ranks = vec![0u32; scores.len()];
    for (pos, &i) in order.iter().enumerate() {
        ranks[i] = pos as u32 + 1;
    }
    ranks
}

/// z-scores of `scores` over the candidate set (population std); all 0 when the
/// set has no variance, so a flat signal contributes nothing to the fused score.
fn z_scores(scores: &[f32]) -> Vec<f32> {
    let n = scores.len() as f64;
    let mean = scores.iter().map(|&s| s as f64).sum::<f64>() / n;
    let var = scores.iter().map(|&s| (s as f64 - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    if !std.is_finite() || std <= f64::EPSILON {
        return vec![0.0; scores.len()];
    }
    scores.iter().map(|&s| ((s as f64 - mean) / std) as f32).collect()
}

/// Rank `candidates` by `params` and return the top `top_k`, best first.
///
/// Every returned [`QueryResult`] carries both raw scores, the fused score, and
/// its rank under each signal alone — `dense_rank` is where the `Dense` mode
/// would have placed it. Ties on the fused score are broken by `dense_rank`
/// (itself tie-broken by doc id), so the order is deterministic.
///
/// `params` must be valid ([`RankingParams::validate`]).
pub(crate) fn fuse(candidates: Vec<Candidate>, params: &RankingParams, top_k: usize) -> Vec<QueryResult> {
    if candidates.is_empty() || top_k == 0 {
        return vec![];
    }
    let ids: Vec<&[u8]> = candidates.iter().map(|c| c.document_id.as_slice()).collect();
    let sparse: Vec<f32> = candidates.iter().map(|c| c.sparse_score).collect();
    let dense: Vec<f32> = candidates.iter().map(|c| c.dense_score).collect();
    let sparse_rank = ranks_desc(&sparse, &ids);
    let dense_rank = ranks_desc(&dense, &ids);
    drop(ids);

    let w = params.sparse_weight;
    let fused: Vec<f32> = match params.mode {
        RankFusion::Dense => dense.clone(),
        RankFusion::Sparse => sparse.clone(),
        RankFusion::Rrf => {
            let k = params.rrf_k;
            sparse_rank
                .iter()
                .zip(&dense_rank)
                .map(|(&rs, &rd)| w / (k + rs as f32) + (1.0 - w) / (k + rd as f32))
                .collect()
        }
        RankFusion::Zscore => {
            let (zs, zd) = (z_scores(&sparse), z_scores(&dense));
            zs.iter().zip(&zd).map(|(&s, &d)| w * s + (1.0 - w) * d).collect()
        }
    };

    let mut results: Vec<QueryResult> = candidates
        .into_iter()
        .enumerate()
        .map(|(i, c)| QueryResult {
            document_id: c.document_id,
            dense_score: c.dense_score,
            sparse_score: c.sparse_score,
            fused_score: fused[i],
            dense_rank: dense_rank[i],
            sparse_rank: sparse_rank[i],
            error_bound: c.error_bound,
        })
        .collect();

    let by_rank = |a: &QueryResult, b: &QueryResult| b.fused_score.total_cmp(&a.fused_score).then(a.dense_rank.cmp(&b.dense_rank));
    if top_k < results.len() {
        results.select_nth_unstable_by(top_k - 1, by_rank);
        results.truncate(top_k);
    }
    results.sort_unstable_by(by_rank);
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(id: &str, sparse: f32, dense: f32) -> Candidate {
        Candidate {
            document_id: id.as_bytes().to_vec(),
            sparse_score: sparse,
            dense_score: dense,
            error_bound: 0.0,
        }
    }

    fn ids(r: &[QueryResult]) -> Vec<&str> {
        r.iter().map(|q| std::str::from_utf8(&q.document_id).unwrap()).collect()
    }

    fn params(mode: RankFusion, w: f32) -> RankingParams {
        RankingParams {
            mode,
            sparse_weight: w,
            ..Default::default()
        }
    }

    /// a: dense best, sparse worst; c: sparse best, dense 2nd — a doc strong on both.
    fn sample() -> Vec<Candidate> {
        vec![cand("a", 1.0, 0.90), cand("b", 2.0, 0.50), cand("c", 4.0, 0.85), cand("d", 3.0, 0.10)]
    }

    #[test]
    fn dense_mode_orders_by_dense_score() {
        let r = fuse(sample(), &params(RankFusion::Dense, 0.5), 10);
        assert_eq!(ids(&r), ["a", "c", "b", "d"]);
        for q in &r {
            assert_eq!(q.fused_score, q.dense_score);
        }
        assert_eq!(r.iter().map(|q| q.dense_rank).collect::<Vec<_>>(), [1, 2, 3, 4]);
    }

    #[test]
    fn sparse_mode_orders_by_sparse_score() {
        let r = fuse(sample(), &params(RankFusion::Sparse, 0.5), 10);
        assert_eq!(ids(&r), ["c", "d", "b", "a"]);
        assert_eq!(r.iter().map(|q| q.sparse_rank).collect::<Vec<_>>(), [1, 2, 3, 4]);
    }

    #[test]
    fn rrf_promotes_doc_ranked_well_by_both() {
        let r = fuse(sample(), &params(RankFusion::Rrf, 0.5), 10);
        assert_eq!(r[0].document_id, b"c", "c is 1st sparse + 2nd dense, so it beats dense-only winner a");
        assert_eq!(r[0].dense_rank, 2);
        assert_eq!(r[0].sparse_rank, 1);
        assert_eq!(r[0].dense_score, 0.85, "dense score is reported unchanged");
    }

    #[test]
    fn weight_extremes_reduce_to_single_list_order() {
        for mode in [RankFusion::Rrf, RankFusion::Zscore] {
            assert_eq!(ids(&fuse(sample(), &params(mode, 0.0), 10)), ["a", "c", "b", "d"], "{mode:?} w=0");
            assert_eq!(ids(&fuse(sample(), &params(mode, 1.0), 10)), ["c", "d", "b", "a"], "{mode:?} w=1");
        }
    }

    #[test]
    fn zscore_with_flat_signal_falls_back_to_other_signal() {
        let flat = vec![cand("a", 1.0, 0.2), cand("b", 1.0, 0.9), cand("c", 1.0, 0.5)];
        let r = fuse(flat, &params(RankFusion::Zscore, 0.5), 10);
        assert_eq!(ids(&r), ["b", "c", "a"]);
        assert!(r.iter().all(|q| q.fused_score.is_finite()));
    }

    #[test]
    fn ties_are_deterministic() {
        // Symmetric ranks → equal RRF scores for a and b; tie broken by dense rank.
        let c = vec![cand("a", 2.0, 0.1), cand("b", 1.0, 0.2)];
        for _ in 0..5 {
            assert_eq!(ids(&fuse(c.clone(), &params(RankFusion::Rrf, 0.5), 10)), ["b", "a"]);
        }
        // Identical scores → ranks tie-broken by doc id.
        let same = vec![cand("z", 1.0, 1.0), cand("y", 1.0, 1.0)];
        let r = fuse(same, &params(RankFusion::Dense, 0.5), 10);
        assert_eq!(ids(&r), ["y", "z"]);
    }

    #[test]
    fn top_k_truncates_and_ranks_span_full_set() {
        let r = fuse(sample(), &params(RankFusion::Dense, 0.5), 2);
        assert_eq!(ids(&r), ["a", "c"]);
        // sparse ranks are over all 4 candidates, not just the returned 2.
        assert_eq!(r[0].sparse_rank, 4);
        assert!(fuse(sample(), &params(RankFusion::Dense, 0.5), 0).is_empty());
        assert!(fuse(vec![], &params(RankFusion::Rrf, 0.5), 10).is_empty());
    }

    #[test]
    fn validate_and_override() {
        let base = RankingParams::default();
        assert!(base.validate().is_ok());
        let o = RankingOverride {
            mode: Some(RankFusion::Rrf),
            ..Default::default()
        };
        let p = base.with_override(&o).unwrap();
        assert_eq!(p.mode, RankFusion::Rrf);
        assert_eq!(p.rrf_k, DEFAULT_RRF_K);
        let bad_k = RankingOverride {
            rrf_k: Some(0.0),
            ..Default::default()
        };
        assert_eq!(base.with_override(&bad_k), Err(RankingError::InvalidRrfK(0.0)));
        let bad_w = RankingOverride {
            sparse_weight: Some(1.5),
            ..Default::default()
        };
        assert_eq!(base.with_override(&bad_w), Err(RankingError::InvalidSparseWeight(1.5)));
        assert!(RankingParams { rrf_k: f32::NAN, ..base }.validate().is_err());
    }

    #[test]
    fn serde_round_trip() {
        let o: RankingOverride = serde_json::from_str(r#"{"mode":"rrf","sparse_weight":0.3}"#).unwrap();
        assert_eq!(o.mode, Some(RankFusion::Rrf));
        assert_eq!(o.sparse_weight, Some(0.3));
        assert!(serde_json::from_str::<RankingOverride>(r#"{"mode":"bogus"}"#).is_err());
        assert!(serde_json::from_str::<RankingOverride>(r#"{"typo":1}"#).is_err());
        let p: RankingParams = serde_json::from_str(r#"{"mode":"zscore"}"#).unwrap();
        assert_eq!(p.rrf_k, DEFAULT_RRF_K);
    }
}
