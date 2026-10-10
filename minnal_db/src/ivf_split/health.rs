//! What a namespace's partition looks like: posting sizes, lineage, and how far
//! moved codes have drifted from their posting's centre (design doc M3a,
//! *Partition health*).
//!
//! Sizes, states and lineage come from memory (the partition snapshot and its
//! exact counts) and are cheap to poll. The code metrics read every chunk code
//! of the namespace, so they are computed only when asked for (`codes`).
//!
//! **Centre drift** of a code is the distance from the centre it was encoded
//! against, `c`, to its current posting's centre `c'`, relative to its residual
//! norm: `‖c − c'‖ / ‖r‖`. 0 for a code at home; it grows as splits move the
//! code's key away from where it was encoded, and a re-encode against `c'`
//! would reset it. A code's **error band** is its stored `error_bound`. `‖r‖`
//! comes from what each code stores: `scaling_factor = ‖r‖ / (f·√D)` and
//! `error_bound = ‖r‖·√((1 − f²)/f²)·ε₀/√(D − 1)` give `‖r‖` and `f = ⟨ō, o⟩`.
//!
//! The design first asked for *residual inflation*, `‖r‖ / ‖x − c'‖`, with
//! `‖x − c'‖² = ‖r‖² + ‖c − c'‖² + 2⟨r, c − c'⟩` estimated from the code. That
//! estimate is biased and cannot be used: a posting's centre is the mean of its
//! members' *reconstructions*, and with 1-bit codes 2-means groups chunks partly
//! by their code noise, so `⟨r̂, c − c'⟩` correlates with the noise and the
//! estimate of `‖x − c'‖²` came out negative on real data (an end-to-end SciFact
//! run read means above 10,000). Drift needs no estimate.

use std::collections::HashMap;

use serde::Serialize;

use super::*;

/// RaBitQ's confidence multiplier in `error_bound` (`rabitq::quantise_using_single_bit`).
const EPSILON_0: f64 = 1.9;

/// One posting.
#[derive(Debug, Clone, Serialize)]
pub struct PostingHealth {
    /// Its key prefix.
    pub posting_id: u32,
    /// `active`, `draining` or `retired`.
    pub state: &'static str,
    /// The centre new codes in it are encoded against.
    pub centre_id: u32,
    /// The posting it was split from.
    pub parent_id: Option<u32>,
    /// `root`, `seed` or `split`.
    pub created_by: &'static str,
    /// Entries (one per document with chunks in it).
    pub entries: u64,
    /// Chunks.
    pub chunks: u64,
    /// Code metrics, when asked for.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub codes: Option<CodeHealth>,
}

/// How a posting's codes relate to its centre.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct CodeHealth {
    /// Share of codes encoded against another centre than the posting's.
    pub foreign_share: f64,
    /// Share of codes encoded against a zero centre (written while the
    /// namespace was one root posting).
    pub zero_centre_share: f64,
    /// Mean centre drift, `‖c − c'‖ / ‖r‖`.
    pub drift_mean: f64,
    /// 90th-percentile centre drift.
    pub drift_p90: f64,
    /// Mean error band (`error_bound`), in score units.
    pub error_band_mean: f64,
}

/// A namespace's partition at a glance.
#[derive(Debug, Clone, Serialize)]
pub struct PartitionHealth {
    /// Postings that are not retired (K).
    pub postings: usize,
    /// Retired postings kept as records.
    pub retired: usize,
    /// Centres stored.
    pub centres: usize,
    /// Entries across live postings.
    pub entries: u64,
    /// Chunks across live postings.
    pub chunks: u64,
    /// Splits made so far (postings created by a split, halved).
    pub splits: usize,
    /// Splits recorded in the journal and not finished.
    pub unfinished_splits: usize,
    /// Chunk count of the live postings: 10th, 50th, 90th percentile and max.
    pub size_p10_p50_p90_max: [u64; 4],
    /// Entry-weighted means of the posting code metrics, when asked for.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub codes: Option<CodeHealth>,
    /// The postings with the highest mean centre drift (at most 20): candidates
    /// for a re-encode. Present when code metrics were asked for.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reencode_candidates: Option<Vec<u32>>,
}

fn state_name(s: PostingState) -> &'static str {
    match s {
        PostingState::Active => "active",
        PostingState::Draining => "draining",
        PostingState::Retired => "retired",
    }
}

fn origin_name(o: PostingOrigin) -> &'static str {
    match o {
        PostingOrigin::Root => "root",
        PostingOrigin::Seed => "seed",
        PostingOrigin::Split => "split",
    }
}

/// Centre drift and error band of one code filed under a posting whose rotated
/// centre is `posting_centre`.
fn code_metrics(ivf: &NamespaceIvf, code: &VectorIndex, posting_centre: &[f32]) -> Option<(f64, f64)> {
    let own = ivf.rotated_centre(code.centre_id)?;
    let d = own.len() as f64;
    let sf = f64::from(code.scaling_factor);
    let eb = f64::from(code.error_bound);
    let shift = own
        .iter()
        .zip(posting_centre)
        .map(|(&c, &cp)| (f64::from(c) - f64::from(cp)).powi(2))
        .sum::<f64>()
        .sqrt();
    if sf <= 0.0 || d < 2.0 {
        return Some((if shift > 0.0 { f64::INFINITY } else { 0.0 }, eb));
    }
    // √(1 − f²) from the two stored factors, then f and ‖r‖.
    let t = (eb / (sf * EPSILON_0 * (d / (d - 1.0)).sqrt())).clamp(0.0, 1.0);
    let f = (1.0 - t * t).sqrt().max(1e-6);
    let r_norm = sf * f * d.sqrt();
    Some((shift / r_norm.max(1e-12), eb))
}

/// A namespace's partition health; with `codes`, also the code metrics (reads
/// every chunk code). Returns the summary and every posting, retired ones
/// included.
pub async fn partition_health(
    db: &AsyncDb,
    namespace: &str,
    handle: &PartitionHandle,
    codes: bool,
) -> Result<(PartitionHealth, Vec<PostingHealth>), crate::KVError> {
    let ivf = handle.snapshot();
    let mut postings: Vec<PostingHealth> = ivf
        .posting_infos()
        .iter()
        .map(|(&id, info)| {
            let (entries, chunks) = ivf.counts().get(id);
            PostingHealth {
                posting_id: id,
                state: state_name(info.state),
                centre_id: info.centre_id,
                parent_id: info.parent_id,
                created_by: origin_name(info.origin),
                entries,
                chunks,
                codes: None,
            }
        })
        .collect();
    postings.sort_by_key(|p| p.posting_id);

    let mut summary_codes = None;
    let mut candidates = None;
    if codes {
        let zero: HashMap<u32, bool> = ivf
            .centre_ids()
            .into_iter()
            .map(|id| (id, ivf.centre(id).is_some_and(|c| c.iter().all(|&x| x == 0.0))))
            .collect();
        let (mut w_sum, mut acc) = (0.0f64, CodeHealth::default());
        for p in postings.iter_mut().filter(|p| p.state != "retired") {
            let Some(posting_centre) = ivf.rotated_centre(p.centre_id).map(<[f32]>::to_vec) else {
                continue;
            };
            let (mut n, mut foreign, mut zeroed, mut band) = (0usize, 0usize, 0usize, 0.0f64);
            let mut drift = Vec::new();
            for (_, list) in posting_entries(db, namespace, p.posting_id).await? {
                for code in &list {
                    n += 1;
                    foreign += usize::from(code.centre_id != p.centre_id);
                    zeroed += usize::from(zero.get(&code.centre_id).copied().unwrap_or(false));
                    if let Some((dr, eb)) = code_metrics(&ivf, code, &posting_centre) {
                        drift.push(dr);
                        band += eb;
                    }
                }
            }
            if n == 0 {
                continue;
            }
            drift.sort_by(f64::total_cmp);
            let m = drift.len().max(1) as f64;
            let h = CodeHealth {
                foreign_share: foreign as f64 / n as f64,
                zero_centre_share: zeroed as f64 / n as f64,
                drift_mean: drift.iter().sum::<f64>() / m,
                drift_p90: drift
                    .get(((drift.len() as f64 * 0.9) as usize).min(drift.len().saturating_sub(1)))
                    .copied()
                    .unwrap_or(1.0),
                error_band_mean: band / m,
            };
            let w = p.entries as f64;
            w_sum += w;
            acc.foreign_share += w * h.foreign_share;
            acc.zero_centre_share += w * h.zero_centre_share;
            acc.drift_mean += w * h.drift_mean;
            acc.drift_p90 += w * h.drift_p90;
            acc.error_band_mean += w * h.error_band_mean;
            p.codes = Some(h);
        }
        if w_sum > 0.0 {
            summary_codes = Some(CodeHealth {
                foreign_share: acc.foreign_share / w_sum,
                zero_centre_share: acc.zero_centre_share / w_sum,
                drift_mean: acc.drift_mean / w_sum,
                drift_p90: acc.drift_p90 / w_sum,
                error_band_mean: acc.error_band_mean / w_sum,
            });
        }
        let mut ranked: Vec<(f64, u32)> = postings.iter().filter_map(|p| p.codes.map(|c| (c.drift_mean, p.posting_id))).collect();
        ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
        candidates = Some(ranked.into_iter().take(20).map(|(_, id)| id).collect());
    }

    let live: Vec<&PostingHealth> = postings.iter().filter(|p| p.state != "retired").collect();
    let mut sizes: Vec<u64> = live.iter().map(|p| p.chunks).collect();
    sizes.sort_unstable();
    let pct = |q: f64| {
        sizes
            .get(((sizes.len() as f64 * q) as usize).min(sizes.len().saturating_sub(1)))
            .copied()
            .unwrap_or(0)
    };
    let summary = PartitionHealth {
        postings: live.len(),
        retired: postings.len() - live.len(),
        centres: ivf.centres(),
        entries: live.iter().map(|p| p.entries).sum(),
        chunks: live.iter().map(|p| p.chunks).sum(),
        splits: postings.iter().filter(|p| p.created_by == "split").count() / 2,
        unfinished_splits: records(db, namespace).await?.len(),
        size_p10_p50_p90_max: [pct(0.1), pct(0.5), pct(0.9), sizes.last().copied().unwrap_or(0)],
        codes: summary_codes,
        reencode_candidates: candidates,
    };
    Ok((summary, postings))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Codes encoded against their own posting's centre have drift 0; after a
    /// split, moved codes drift and are foreign to their new posting.
    #[tokio::test]
    async fn drift_is_zero_at_home_and_grows_after_a_move() {
        let dir = tempfile::TempDir::new().unwrap();
        let db = super::super::tests::setup(&dir, 60).await;
        let h = super::super::tests::handle(&db).await;
        let (before, _) = partition_health(&db, super::super::tests::NS, &h, true).await.unwrap();
        let c = before.codes.unwrap();
        assert!(c.drift_mean.abs() < 1e-9, "{c:?}");
        assert_eq!((c.foreign_share, c.zero_centre_share), (0.0, 1.0));
        split_oversized(&db, super::super::tests::NS, &h, &super::super::tests::settings(40))
            .await
            .unwrap();
        let (after, postings) = partition_health(&db, super::super::tests::NS, &h, true).await.unwrap();
        let c = after.codes.unwrap();
        assert!(c.foreign_share > 0.9, "every code moved off the root: {c:?}");
        assert!(
            c.drift_mean > 0.1 && c.drift_mean.is_finite(),
            "moved codes drift from their encoding centre: {c:?}"
        );
        assert!(after.splits >= 1 && after.retired >= 1 && after.unfinished_splits == 0);
        assert_eq!(after.chunks, 180);
        assert!(after.size_p10_p50_p90_max[3] <= 40);
        assert_eq!(after.reencode_candidates.unwrap().len(), after.postings.min(20));
        assert_eq!(postings.iter().filter(|p| p.state == "retired").count(), after.retired);
        // Without code metrics nothing is read but memory.
        let (cheap, _) = partition_health(&db, super::super::tests::NS, &h, false).await.unwrap();
        assert!(cheap.codes.is_none() && cheap.reencode_candidates.is_none());
        db.shutdown().await.unwrap();
    }
}
