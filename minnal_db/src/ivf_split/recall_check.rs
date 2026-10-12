//! Does a namespace's search miss what a full read finds? The check behind
//! "should I rebuild" (FR-007).
//!
//! A rebuild changes one thing: which postings hold which chunks, and so which
//! documents a search reaches within its probe budget. When a search already
//! reads every posting, no partition can change its results. When it reads a
//! small share, a better partition finds more of what a full read would.
//!
//! So the check measures that directly, on queries it needs no embedding call
//! for. It prefers **real queries**: the caller passes cached query embeddings
//! (the doc store reads them from the query-embedding cache), and the check
//! samples them. With fewer than [`MIN_GIVEN_QUERIES`] it falls back to **stored
//! chunks**, sampled uniformly over chunks, each turned into a query: its 1-bit
//! reconstruction, back in the original space and normalised. Each query is
//! searched twice through the production [`search`]:
//!
//! - with the namespace's own settings;
//! - with every posting probed (a full read), all else equal. Above
//!   [`MAX_REFERENCE_ENTRIES`] the reference reads the nearest postings up to
//!   that many entries instead, which bounds its memory.
//!
//! A chunk query's source document is dropped from both lists (a chunk finds its
//! own document first). **Overlap@k** is the share of the full read's top k that
//! the normal search also returns. It is the recall the probe budget costs, so it
//! is low when the budget is too small for the namespace *or* when the partition
//! puts neighbours in postings the search does not reach. A rebuild fixes only
//! the second, so the check is read twice: before a rebuild and after it, with
//! the same seed (the same queries). A rise is what the rebuild gained.
//!
//! Chunk queries are not real queries: they are noisy reconstructions of
//! document text, and they do not see everything a rebuild does for real ones
//! (FR-007, *Calibration*). Use them when no queries are cached.

use std::collections::HashSet;
use std::time::Instant;

use serde::Serialize;

use super::*;
use crate::semantic_search::service::{SemanticSearchConfig, search};
use crate::semantic_search::{ProbeSettings, select_probes};
use crate::vector_kv::DbVectorStore;

/// The cutoffs overlap is reported at.
pub const CUTOFFS: [usize; 6] = [10, 20, 30, 40, 50, 100];

/// The overlap@10 below which the verdict is `below_target`, by default.
pub const DEFAULT_MIN_OVERLAP: f64 = 0.95;

/// The most queries one check may sample.
pub const MAX_SAMPLE: usize = 1_000;

/// Fewer given (cached) queries than this and the check samples stored chunks.
pub const MIN_GIVEN_QUERIES: usize = 20;

/// The most entries the reference search reads. A search holds every entry it
/// scans in memory until it returns (a few hundred bytes each), so on a larger
/// namespace the reference is the nearest postings up to this many entries, not
/// a full read: about 150 MB and 0.1 s per query at the cap.
pub const MAX_REFERENCE_ENTRIES: u64 = 500_000;

/// How a check runs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecallCheckParams {
    /// Queries to sample (1 to [`MAX_SAMPLE`]).
    pub sample: usize,
    /// Seed of the sample, so a check before and after a rebuild can use the
    /// same chunks (the chunks are the same; the postings they sit in move).
    pub seed: u64,
    /// The overlap@10 below which the verdict is `below_target`.
    pub min_overlap: f64,
}

impl Default for RecallCheckParams {
    fn default() -> Self {
        Self {
            sample: 100,
            seed: 0x5eed,
            min_overlap: DEFAULT_MIN_OVERLAP,
        }
    }
}

/// What the check suggests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RecallVerdict {
    /// Every sampled search already read every posting: no partition can change
    /// its results, so a rebuild cannot help.
    ReadInFull,
    /// Overlap@10 is at or above the threshold.
    Ok,
    /// Overlap@10 is below it: searches miss part of what a full read finds.
    /// Rebuild and check again with the same seed; keep the rebuild if overlap
    /// rose. If it did not, the budget is too small for the namespace, and only a
    /// larger budget recovers the rest.
    BelowTarget,
    /// Nothing to sample.
    Empty,
}

/// Where a check's queries came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QuerySource {
    /// Real queries' embeddings (the query-embedding cache).
    CachedQueries,
    /// Stored chunks' 1-bit reconstructions.
    StoredChunks,
}

/// Overlap at one cutoff.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Overlap {
    /// The cutoff.
    pub k: usize,
    /// Mean over the sampled queries.
    pub mean: f64,
}

/// A recall check's result.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RecallCheck {
    /// The verdict at `min_overlap`.
    pub verdict: RecallVerdict,
    /// The threshold used.
    pub min_overlap: f64,
    /// Where the queries came from.
    pub query_source: QuerySource,
    /// Queries searched.
    pub sampled: usize,
    /// Entries in the namespace's live postings.
    pub entries: u64,
    /// Live postings.
    pub postings: usize,
    /// The probe budget the namespace's searches use (entries).
    pub budget_entries: u64,
    /// Mean share of the entries a normal search read.
    pub read_share: f64,
    /// Mean postings a normal search probed.
    pub probes_mean: f64,
    /// Whether the reference search read every posting (false above
    /// [`MAX_REFERENCE_ENTRIES`]).
    pub reference_is_full_read: bool,
    /// Mean share of the entries the reference search read.
    pub reference_share: f64,
    /// Overlap with the full read at each of [`CUTOFFS`].
    pub overlap: Vec<Overlap>,
    /// 10th percentile of overlap@10 over the queries (the bad tail).
    pub overlap_at_10_p10: f64,
    /// Mean time of a normal search, in ms.
    pub search_ms: f64,
    /// Mean time of a full-read search, in ms.
    pub full_read_ms: f64,
    /// Wall time of the whole check, sampling included, in ms.
    pub elapsed_ms: f64,
}

/// Run the check on `namespace`, searching with `config` (the namespace's
/// settings, or a request's overrides of them), on a sample of `given` query
/// embeddings or, with fewer than [`MIN_GIVEN_QUERIES`] of them, of stored
/// chunks. Searches run one at a time and take the scoring gate like any search,
/// so a check adds the load of one client.
pub async fn recall_check(
    db: &AsyncDb,
    namespace: &str,
    handle: &PartitionHandle,
    config: &SemanticSearchConfig,
    params: &RecallCheckParams,
    given: Vec<Vec<f32>>,
) -> Result<RecallCheck, crate::KVError> {
    let started = Instant::now();
    let dim = handle.snapshot().dim();
    let given: Vec<Vec<f32>> = given.into_iter().filter(|q| q.len() == dim).filter_map(normalised).collect();
    let (query_source, queries) = if given.len() >= MIN_GIVEN_QUERIES {
        (QuerySource::CachedQueries, sample_given(given, params))
    } else {
        let _epoch = handle.read_epoch().await;
        (
            QuerySource::StoredChunks,
            sample_queries(db, namespace, &handle.snapshot(), params).await?,
        )
    };
    let top = CUTOFFS[CUTOFFS.len() - 1] + 1;
    let store = DbVectorStore::new(db, namespace).await?;
    let (mut sums, mut at10) = (vec![0.0f64; CUTOFFS.len()], Vec::with_capacity(queries.len()));
    let (mut read_share, mut probes, mut normal_s, mut full_s) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let (mut read_in_full, mut reference_full, mut reference_share) = (true, true, 0.0f64);
    let (mut entries, mut postings) = (0u64, 0usize);
    for (doc, q) in &queries {
        let doc = doc.as_deref();
        let _epoch = handle.read_epoch().await;
        let ivf = handle.snapshot();
        entries = ivf.total_entries();
        postings = ivf.postings();
        let (chosen, planned) = select_probes(&*ivf, std::slice::from_ref(q), &config.probe);
        read_in_full &= chosen.len() >= postings;
        read_share += planned as f64 / entries.max(1) as f64;
        probes += chosen.len() as f64;
        let mut full = config.clone();
        full.probe = if entries <= MAX_REFERENCE_ENTRIES {
            ProbeSettings {
                budget_entries: u64::MAX,
                min_probes: postings,
                max_probes: postings,
            }
        } else {
            ProbeSettings {
                budget_entries: MAX_REFERENCE_ENTRIES,
                min_probes: 1,
                max_probes: postings,
            }
        };
        let (reference, reference_planned) = select_probes(&*ivf, std::slice::from_ref(q), &full.probe);
        reference_full &= reference.len() >= postings;
        reference_share += reference_planned as f64 / entries.max(1) as f64;
        let t = Instant::now();
        let normal = search(
            config,
            namespace,
            &*ivf,
            std::slice::from_ref(q),
            q,
            &store,
            None::<fn(&[u8]) -> bool>,
            Some(top),
        )
        .await?;
        normal_s += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let truth = search(
            &full,
            namespace,
            &*ivf,
            std::slice::from_ref(q),
            q,
            &store,
            None::<fn(&[u8]) -> bool>,
            Some(top),
        )
        .await?;
        full_s += t.elapsed().as_secs_f64();
        let normal: Vec<&[u8]> = normal.iter().map(|r| r.document_id.as_slice()).filter(|d| Some(*d) != doc).collect();
        let truth: Vec<&[u8]> = truth.iter().map(|r| r.document_id.as_slice()).filter(|d| Some(*d) != doc).collect();
        for (i, &k) in CUTOFFS.iter().enumerate() {
            let o = overlap_at(&normal, &truth, k);
            sums[i] += o;
            if k == 10 {
                at10.push(o);
            }
        }
    }
    let n = queries.len();
    let m = n.max(1) as f64;
    at10.sort_by(f64::total_cmp);
    let overlap: Vec<Overlap> = CUTOFFS.iter().zip(&sums).map(|(&k, &s)| Overlap { k, mean: s / m }).collect();
    let verdict = if n == 0 {
        RecallVerdict::Empty
    } else if read_in_full {
        RecallVerdict::ReadInFull
    } else if overlap[0].mean >= params.min_overlap {
        RecallVerdict::Ok
    } else {
        RecallVerdict::BelowTarget
    };
    Ok(RecallCheck {
        verdict,
        min_overlap: params.min_overlap,
        query_source,
        sampled: n,
        entries,
        postings,
        budget_entries: config.probe.budget_entries,
        read_share: read_share / m,
        probes_mean: probes / m,
        reference_is_full_read: reference_full,
        reference_share: reference_share / m,
        overlap,
        overlap_at_10_p10: at10.get(n / 10).copied().unwrap_or(0.0),
        search_ms: normal_s * 1e3 / m,
        full_read_ms: full_s * 1e3 / m,
        elapsed_ms: started.elapsed().as_secs_f64() * 1e3,
    })
}

/// Of the full read's top `k`, the share the normal search also has in its top
/// `k` (1 when the full read finds nothing).
fn overlap_at(normal: &[&[u8]], truth: &[&[u8]], k: usize) -> f64 {
    let truth = &truth[..truth.len().min(k)];
    if truth.is_empty() {
        return 1.0;
    }
    let got: HashSet<&[u8]> = normal.iter().take(k).copied().collect();
    truth.iter().filter(|d| got.contains(*d)).count() as f64 / truth.len() as f64
}

/// `params.sample` of `given`, drawn without replacement by `params.seed`, with
/// no source document.
fn sample_given(mut given: Vec<Vec<f32>>, params: &RecallCheckParams) -> Vec<(Option<Vec<u8>>, Vec<f32>)> {
    let mut rng = crate::semantic_search::cluster::split::SplitRng::new(params.seed);
    let n = params.sample.clamp(1, MAX_SAMPLE).min(given.len());
    // A partial Fisher-Yates shuffle: the first `n` places are the sample.
    for i in 0..n {
        let j = i + (rng.next_u64() % (given.len() - i) as u64) as usize;
        given.swap(i, j);
    }
    given.truncate(n);
    given.into_iter().map(|q| (None, q)).collect()
}

/// `params.sample` chunks drawn uniformly over the live postings' chunks, each
/// as `(source document, query)`.
async fn sample_queries(
    db: &AsyncDb,
    namespace: &str,
    ivf: &NamespaceIvf,
    params: &RecallCheckParams,
) -> Result<Vec<(Option<Vec<u8>>, Vec<f32>)>, crate::KVError> {
    let mut live: Vec<(u32, u64)> = ivf
        .posting_infos()
        .iter()
        .filter(|(_, info)| info.state != PostingState::Retired)
        .map(|(&id, _)| (id, ivf.posting_chunks(id)))
        .filter(|&(_, chunks)| chunks > 0)
        .collect();
    live.sort_unstable_by_key(|&(id, _)| id);
    let total: u64 = live.iter().map(|&(_, c)| c).sum();
    if total == 0 {
        return Ok(Vec::new());
    }
    let mut rng = crate::semantic_search::cluster::split::SplitRng::new(params.seed);
    // (posting, rank within it), in draw order.
    let picks: Vec<(u32, u64)> = (0..params.sample.clamp(1, MAX_SAMPLE))
        .map(|_| {
            let mut r = rng.next_u64() % total;
            for &(id, chunks) in &live {
                if r < chunks {
                    return (id, r);
                }
                r -= chunks;
            }
            unreachable!("rank below the total")
        })
        .collect();
    let mut wanted: Vec<u32> = picks.iter().map(|&(p, _)| p).collect();
    wanted.sort_unstable();
    wanted.dedup();
    let rows = postings_entries(db, namespace, &wanted).await?;
    let mut out = Vec::with_capacity(picks.len());
    for (posting, rank) in picks {
        let Some(entries) = rows.get(&posting) else { continue };
        let held: u64 = entries.iter().map(|(_, codes)| codes.len() as u64).sum();
        if held == 0 {
            continue;
        }
        // Counts can trail the keys by a write; wrap rather than miss.
        let mut r = rank % held;
        for (doc, codes) in entries {
            if r < codes.len() as u64 {
                let code = &codes[r as usize];
                if let Some(q) = ivf
                    .rotated_centre(code.centre_id)
                    .and_then(|c| reconstruct_single_bit(code, c))
                    .map(|x| ivf.unrotate(&x))
                    .and_then(normalised)
                {
                    out.push((Some(doc.clone()), q));
                }
                break;
            }
            r -= codes.len() as u64;
        }
    }
    Ok(out)
}

fn normalised(mut v: Vec<f32>) -> Option<Vec<f32>> {
    let norm = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>().sqrt();
    if norm <= 0.0 || !norm.is_finite() {
        return None;
    }
    v.iter_mut().for_each(|x| *x = (f64::from(*x) / norm) as f32);
    Some(v)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{NS, handle, settings, setup};
    use super::*;

    fn config(probe: ProbeSettings) -> SemanticSearchConfig {
        SemanticSearchConfig {
            probe,
            ..SemanticSearchConfig::default()
        }
    }

    fn params(sample: usize) -> RecallCheckParams {
        RecallCheckParams {
            sample,
            ..RecallCheckParams::default()
        }
    }

    /// One posting is read in full by any search: the full read finds nothing
    /// more, and the verdict says a rebuild cannot help.
    #[tokio::test]
    async fn a_namespace_read_in_full_needs_no_rebuild() {
        let dir = tempfile::TempDir::new().unwrap();
        let db = setup(&dir, 40).await;
        let h = handle(&db).await;
        let r = recall_check(&db, NS, &h, &config(ProbeSettings::fixed(1)), &params(20), Vec::new())
            .await
            .unwrap();
        assert_eq!(r.verdict, RecallVerdict::ReadInFull, "{r:?}");
        assert_eq!((r.sampled, r.postings, r.entries), (20, 1, 40));
        assert!(r.overlap.iter().all(|o| o.mean == 1.0), "{r:?}");
        assert!((r.read_share - 1.0).abs() < 1e-9);
        assert!(r.reference_is_full_read && (r.reference_share - 1.0).abs() < 1e-9, "{r:?}");
        db.shutdown().await.unwrap();
    }

    /// After splits, a one-posting search reads part of the namespace and can
    /// miss what the full read finds; probing every posting cannot.
    #[tokio::test]
    async fn a_partial_read_is_measured_against_the_full_read() {
        let dir = tempfile::TempDir::new().unwrap();
        let db = setup(&dir, 60).await;
        let h = handle(&db).await;
        maintain_partition(&db, NS, &h, &settings(40)).await.unwrap();
        let postings = h.snapshot().postings();
        assert!(postings > 2, "split into {postings}");
        let narrow = recall_check(&db, NS, &h, &config(ProbeSettings::fixed(1)), &params(30), Vec::new())
            .await
            .unwrap();
        assert_ne!(narrow.verdict, RecallVerdict::ReadInFull, "{narrow:?}");
        assert!(narrow.read_share < 1.0 && (narrow.probes_mean - 1.0).abs() < 1e-9, "{narrow:?}");
        assert!(narrow.overlap[0].mean < 1.0, "one posting of {postings} misses neighbours: {narrow:?}");
        let wide = recall_check(&db, NS, &h, &config(ProbeSettings::fixed(postings)), &params(30), Vec::new())
            .await
            .unwrap();
        assert_eq!(wide.verdict, RecallVerdict::ReadInFull, "{wide:?}");
        assert!(wide.overlap.iter().all(|o| o.mean == 1.0), "{wide:?}");
        // The same seed samples the same chunks.
        let again = recall_check(&db, NS, &h, &config(ProbeSettings::fixed(1)), &params(30), Vec::new())
            .await
            .unwrap();
        assert_eq!(again.overlap, narrow.overlap);
        db.shutdown().await.unwrap();
    }

    /// Given enough queries, the check samples them instead of chunks; given
    /// too few, it falls back to chunks.
    #[tokio::test]
    async fn given_queries_are_preferred_to_chunks() {
        let dir = tempfile::TempDir::new().unwrap();
        let db = setup(&dir, 60).await;
        let h = handle(&db).await;
        maintain_partition(&db, NS, &h, &settings(40)).await.unwrap();
        let dim = h.snapshot().dim();
        let queries: Vec<Vec<f32>> = (0..25)
            .map(|i| (0..dim).map(|j| if j % 4 == i % 4 { 1.0 } else { 0.1 * (i + j) as f32 / 50.0 }).collect())
            .collect();
        let r = recall_check(&db, NS, &h, &config(ProbeSettings::fixed(1)), &params(10), queries.clone())
            .await
            .unwrap();
        assert_eq!((r.query_source, r.sampled), (QuerySource::CachedQueries, 10), "{r:?}");
        let r = recall_check(&db, NS, &h, &config(ProbeSettings::fixed(1)), &params(100), queries.clone())
            .await
            .unwrap();
        assert_eq!(r.sampled, 25, "no more than were given");
        let few = queries[..MIN_GIVEN_QUERIES - 1].to_vec();
        let r = recall_check(&db, NS, &h, &config(ProbeSettings::fixed(1)), &params(10), few)
            .await
            .unwrap();
        assert_eq!((r.query_source, r.sampled), (QuerySource::StoredChunks, 10), "{r:?}");
        db.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn an_empty_namespace_has_nothing_to_check() {
        let dir = tempfile::TempDir::new().unwrap();
        let db = setup(&dir, 0).await;
        let h = handle(&db).await;
        let r = recall_check(&db, NS, &h, &config(ProbeSettings::fixed(1)), &params(10), Vec::new())
            .await
            .unwrap();
        assert_eq!((r.verdict, r.sampled), (RecallVerdict::Empty, 0));
        db.shutdown().await.unwrap();
    }

    #[test]
    fn overlap_counts_the_full_reads_top_k_found() {
        let (a, b, c, d): (&[u8], &[u8], &[u8], &[u8]) = (b"a", b"b", b"c", b"d");
        assert_eq!(overlap_at(&[a, b], &[a, b], 10), 1.0);
        assert_eq!(overlap_at(&[a, c], &[a, b], 10), 0.5);
        // Only the top k of each list counts.
        assert_eq!(overlap_at(&[c, a], &[a, b, d], 1), 0.0);
        assert_eq!(overlap_at(&[], &[], 10), 1.0);
    }
}
