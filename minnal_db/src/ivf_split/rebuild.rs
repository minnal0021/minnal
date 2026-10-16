//! Rebuilding a namespace's partition from its codes (design doc M4).
//!
//! K-means ([`crate::semantic_search::cluster::kmeans`]) runs over the
//! reconstructions of every chunk in an active posting, and every active
//! posting is then drained into the fitted ones, each chunk to the nearest new
//! centre. No embedding is needed, and no code is re-encoded: a moved code keeps
//! its `centre_id`, as in every other move. Afterwards the ordinary
//! maintenance loop splits or merges whatever k-means left outside the limits.
//!
//! It goes through the journal like a split, with many postings drained at
//! once:
//!
//! | Step | Writes | Record after |
//! |---|---|---|
//! | 1. Plan | read every active posting's codes, k-means; write the record | `Planned` |
//! | 2. Prepare | append the centres; write the new postings `Active`; mark the old ones `Draining` | `Planned` |
//! | 3. Publish | under the epoch write guard: record `Published`, then the snapshot | `Published` |
//! | 4. Copy | each old posting's chunks to the nearest new posting | `Published` |
//! | 5. Barrier | flush the sparse and meta namespaces | `Copied` |
//! | 6. Delete | after an epoch barrier, each old posting's keys | `Copied` |
//! | 7. Retire | the old postings `Retired`; publish; delete the record | (gone) |
//!
//! Writes go on throughout: before the publish they land in old postings and
//! are moved with the rest; after it they are routed to new ones. Recovery
//! abandons a `Planned` rebuild (the new postings retired empty, the old ones
//! active again) and redoes a `Published` or `Copied` one.
//!
//! K-means runs on a pool of its own (at most [`MAX_REBUILD_THREADS`] threads),
//! inside the blocking pool: never on the rayon pool searches share, where a
//! long job would be stolen by a search waiting in a `join` (the scoring-gate
//! rule), and never on the async threads.

use std::sync::Arc;

use super::*;
use crate::semantic_search::cluster::kmeans::{KmeansParams, kmeans};

/// The most threads a rebuild's k-means may use.
pub const MAX_REBUILD_THREADS: usize = 2;

/// How a rebuild runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RebuildParams {
    /// Postings to fit; by default as many as are active now, so a search
    /// probes about as many as before.
    pub postings: Option<usize>,
    /// K-means threads (1 to [`MAX_REBUILD_THREADS`]).
    pub threads: usize,
}

impl Default for RebuildParams {
    fn default() -> Self {
        Self {
            postings: None,
            threads: MAX_REBUILD_THREADS,
        }
    }
}

/// What a rebuild did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct RebuildReport {
    /// Active postings before.
    pub postings_before: usize,
    /// Postings k-means fitted.
    pub postings_fitted: usize,
    /// Active postings after the maintenance that followed.
    pub postings_after: usize,
    /// Chunks clustered.
    pub chunks: u64,
    /// K-means iterations run.
    pub iterations: usize,
    /// Chunks copied to a new posting.
    pub chunks_moved: u64,
}

/// Where a namespace's last rebuild stands (`GET …/vector/partition`).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum RebuildState {
    /// Running now.
    Running,
    /// Finished; `report` says what it did.
    Done,
    /// Stopped by an error (a journalled rebuild is finished by recovery at
    /// the next start).
    Failed,
}

/// A namespace's last rebuild since the server started.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct RebuildStatus {
    /// Running, done or failed.
    pub state: RebuildState,
    /// Postings asked for (`None`: as many as were active).
    pub postings_requested: Option<usize>,
    /// When it started (Unix ms).
    pub started_at_ms: u64,
    /// When it ended (Unix ms).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finished_at_ms: Option<u64>,
    /// What it did, once done.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub report: Option<RebuildReport>,
    /// Why it failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Rebuild `namespace`'s partition, then split and merge what it left outside
/// the limits. Takes the maintenance lock.
pub async fn rebuild_partition(
    db: &AsyncDb,
    namespace: &str,
    handle: &PartitionHandle,
    ops: &MaintenanceOps,
    params: &RebuildParams,
) -> Result<RebuildReport, crate::KVError> {
    let mut report = {
        let _maintenance = handle.lock_maintenance().await;
        rebuild_locked(db, namespace, handle, ops, params).await?
    };
    maintain_partition(db, namespace, handle, ops).await?;
    let ivf = handle.snapshot();
    report.postings_after = ivf.posting_infos().values().filter(|i| i.state == PostingState::Active).count();
    Ok(report)
}

async fn rebuild_locked(
    db: &AsyncDb,
    namespace: &str,
    handle: &PartitionHandle,
    ops: &MaintenanceOps,
    params: &RebuildParams,
) -> Result<RebuildReport, crate::KVError> {
    let ivf = handle.snapshot();
    let mut active: Vec<u32> = ivf
        .posting_infos()
        .iter()
        .filter(|(_, info)| info.state == PostingState::Active)
        .map(|(&id, _)| id)
        .collect();
    active.sort_unstable();
    let mut report = RebuildReport {
        postings_before: active.len(),
        ..RebuildReport::default()
    };
    if active.is_empty() {
        return Ok(report);
    }

    // 1. Plan: every code that can be reconstructed, clustered.
    let mut entries = postings_entries(db, namespace, &active).await?;
    let mut buf = Vec::new();
    let codes: Vec<VectorIndex> = active
        .iter()
        .filter_map(|p| entries.remove(p))
        .flatten()
        .flat_map(|(_, codes)| codes)
        .filter(|c| reconstruct_into(&ivf, c, &mut buf))
        .collect();
    report.chunks = codes.len() as u64;
    if codes.is_empty() {
        return Ok(report);
    }
    let op = next_op_id(db, namespace).await?;
    let k = params.postings.unwrap_or(active.len()).max(1);
    let threads = params.threads.clamp(1, MAX_REBUILD_THREADS);
    let seed = op ^ (codes.len() as u64).rotate_left(32);
    let dim = IvfLayout::dim(ivf.as_ref());
    let fit = {
        let ivf = Arc::clone(&ivf);
        tokio::task::spawn_blocking(move || {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|i| format!("minnal-rebuild-{i}"))
                .build()
                .map_err(|e| crate::KVError::Io(std::io::Error::other(e)))?;
            Ok::<_, crate::KVError>(pool.install(|| {
                kmeans(
                    codes.len(),
                    dim,
                    |i, buf| {
                        reconstruct_into(&ivf, &codes[i], buf);
                    },
                    &KmeansParams::new(k),
                    &mut SplitRng::new(seed),
                )
            }))
        })
        .await
        .map_err(|e| crate::KVError::Io(std::io::Error::other(e)))??
    };
    let Some(fit) = fit else { return Ok(report) };
    report.iterations = fit.iterations;
    // A centre no chunk is nearest would be an empty posting.
    let centres: Vec<Vec<f32>> = fit
        .centres
        .iter()
        .zip(&fit.sizes)
        .filter(|(_, n)| **n > 0)
        .map(|(c, _)| ivf.unrotate(c))
        .collect();
    report.postings_fitted = centres.len();
    let (max_posting, max_centre) = ivf.max_ids();
    let record = Record {
        state: JournalState::Planned,
        op: Op::Rebuild {
            first_centre: max_centre + 1,
            first_posting: max_posting + 1,
            postings: centres.len() as u32,
            drained: active.clone(),
        },
    };
    write_record(db, namespace, op, &record).await?;

    // 2. Prepare: centres and new postings (not yet routable), old ones draining.
    let centres_ns = db.namespace(ivf_centres_ns(namespace)).await?;
    let mut new_centres = Vec::with_capacity(centres.len());
    let mut changed = Vec::with_capacity(centres.len() + active.len());
    for (i, c) in centres.into_iter().enumerate() {
        let (centre_id, posting_id) = (max_centre + 1 + i as u32, max_posting + 1 + i as u32);
        crash_point!();
        centres_ns.put(centre_id.to_be_bytes().to_vec(), f32s_to_bytes(&c)).await?;
        let posting = Posting {
            posting_id,
            routing_centroid: c.clone(),
            centre_id,
            state: PostingState::Active,
            parent_id: None,
            origin: PostingOrigin::Rebuild,
        };
        write_posting(db, namespace, &posting).await?;
        new_centres.push((centre_id, c));
        changed.push(posting);
    }
    for &id in &active {
        let mut old = posting_of(&ivf, id).ok_or_else(|| corrupt(namespace, format!("posting {id} vanished")))?;
        old.state = PostingState::Draining;
        write_posting(db, namespace, &old).await?;
        changed.push(old);
    }

    // 3. Publish, then drain.
    publish_drain(db, namespace, handle, op, &record, &new_centres, &changed).await?;
    let before = handle.counters().read().rebuild_moves;
    finish(db, namespace, handle, ops, op, record.with_state(JournalState::Published)).await?;
    report.chunks_moved = handle.counters().read().rebuild_moves - before;
    Ok(report)
}

/// Steps 4–7 of a published rebuild (also what recovery runs).
pub(super) async fn finish(
    db: &AsyncDb,
    namespace: &str,
    handle: &PartitionHandle,
    ops: &MaintenanceOps,
    op: u64,
    record: Record,
) -> Result<(), crate::KVError> {
    let Op::Rebuild { drained, .. } = &record.op else {
        unreachable!("not a rebuild")
    };
    if record.state == JournalState::Published {
        let ivf = handle.snapshot();
        let rule = drain_rule(namespace, &ivf, ops, &record.op)?;
        let mut copied = 0;
        for &posting in drained {
            copied += drain_copy(db, namespace, handle, &ivf, &rule, posting).await?;
        }
        MaintenanceCounters::add(&handle.counters().rebuild_moves, copied);
        // 5. Barrier: the copies are durable before any delete (one fsync of
        // the vector index WAL, which holds them; the metas are WAL-backed).
        crash_point!();
        db.sync_vector_index_wal().await?;
        write_record(db, namespace, op, &record.with_state(JournalState::Copied)).await?;
    }
    handle.epoch_barrier().await;
    for &posting in drained {
        drain_delete(db, namespace, handle, posting).await?;
    }
    // 7. Retire.
    let mut retired = Vec::with_capacity(drained.len());
    for &id in drained {
        let mut posting = stored_posting(db, namespace, id)
            .await?
            .ok_or_else(|| corrupt(namespace, format!("drained posting {id} is missing")))?;
        if posting.state != PostingState::Retired {
            posting.state = PostingState::Retired;
            write_posting(db, namespace, &posting).await?;
        }
        retired.push(posting);
    }
    {
        let guard = handle.write_epoch().await;
        let next = handle.snapshot().evolve(&[], &retired).map_err(|e| corrupt(namespace, e))?;
        handle.publish(&guard, next);
    }
    MaintenanceCounters::add(&handle.counters().rebuilds, 1);
    delete_record(db, namespace, op).await
}
