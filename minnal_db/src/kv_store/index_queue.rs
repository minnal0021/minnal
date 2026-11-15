//! A field index's queue of row changes: how a write leaves its field-index
//! change behind in version order and returns, for the field-index worker to
//! apply.
//!
//! Each change is relative to the key's previous value (out of the old value's
//! bitmap, into the new one's), so a key's changes must land in the order its
//! versions do. That order is fixed under the key's value-log bucket lock, which
//! every write of the key holds (TTL expiry, which takes no key stripe, too).
//! The field's own lock cannot be taken there: a query or the index checkpoint
//! can hold it for milliseconds (a range merging hundreds of bitmaps, an
//! `msync`), and every write to the bucket would wait behind it.
//!
//! So a write only **queues** its change under the bucket lock (one short
//! mutex, memory only) and returns. The field-index worker
//! (`db::workers::index_apply`) **applies** queued changes under the field's
//! write lock ([`ChangeQueue::apply_queued`]), at most [`APPLY_BATCH`] per hold
//! so a query waits for one batch, not a backlog. Changes are taken off the
//! queue only by a holder of that write lock, so they apply in queue order
//! whoever applies them. Besides the worker, the index checkpoint applies
//! everything queued before it records its WAL offset (crash replay starts
//! there, so nothing queued may be left behind it), a writer applies when its
//! field's queue is past [`QUEUE_CAP`] or no worker runs (recovery at open, an
//! engine opened without workers), and `sync_field_indexes` applies on request.
//! Queries never do: they read what has been applied, and may lag the writes
//! that have returned, as vector search does.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use log::warn;
use parking_lot::{Mutex, RwLock};

use crate::field_index::{DynFieldIndex, IndexValue, RowChange};

/// Changes applied per hold of a field's write lock: a query waiting for the
/// lock waits for one batch at most. A few µs a change, so about a millisecond.
pub(crate) const APPLY_BATCH: usize = 1_024;

/// A field's queue length past which a writer applies the queue itself, after
/// its bucket lock: the worker has fallen behind, and an unbounded queue would
/// grow memory and lag without limit. Reached only under sustained overload.
pub(crate) const QUEUE_CAP: usize = 65_536;

/// One row's change to a field index, owned so that it can wait in the queue
/// after the write that made it has released its locks. The variants are
/// [`RowChange`]'s.
pub(crate) enum QueuedChange {
    /// Move the row from `old`'s bucket to `new`'s.
    Update { old: Option<IndexValue>, new: Option<IndexValue> },
    /// Make the value the row's only one.
    Set(IndexValue),
    /// Add the row under the value, for a row in no bucket.
    Insert(IndexValue),
    /// Take the row out of the value's bucket.
    Remove(IndexValue),
    /// Take the row out of every bucket.
    RemoveAll,
}

impl QueuedChange {
    fn as_row_change(&self) -> RowChange<'_> {
        match self {
            Self::Update { old, new } => RowChange::Update {
                old: old.as_ref(),
                new: new.as_ref(),
            },
            Self::Set(v) => RowChange::Set(v),
            Self::Insert(v) => RowChange::Insert(v),
            Self::Remove(v) => RowChange::Remove(v),
            Self::RemoveAll => RowChange::RemoveAll,
        }
    }
}

/// Where a caller that must report its own change's rejection (the field
/// reindex) finds it once the change is applied. Without one, a rejection is
/// reported as an index gap.
pub(crate) type ChangeOutcome = Arc<Mutex<Option<String>>>;

/// Who applied a batch of queued changes, for the progress counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Applier {
    /// The field-index worker.
    Worker,
    /// A writer: its field's queue was past [`QUEUE_CAP`], no worker was
    /// running, or it is a field reindex waiting for its own outcome.
    Writer,
    /// The index checkpoint, before it records its WAL offset, or an explicit
    /// `sync_field_indexes`.
    Barrier,
}

struct Queued {
    row_id: u128,
    change: QueuedChange,
    /// The key the row belongs to, for the gap record a rejection leaves.
    key: Vec<u8>,
    outcome: Option<ChangeOutcome>,
    /// The user seq of the write that made the change (0 for a reindex).
    seq: u64,
    queued_at: Instant,
}

/// What applying the queue left for the caller to do.
pub(crate) struct Applied {
    /// Changes applied.
    pub(crate) count: usize,
    /// Keys whose change the field rejected, to record as index gaps.
    pub(crate) rejected: Vec<Vec<u8>>,
    /// The field's reclaimable dead blob bytes, for the checkpoint trigger.
    pub(crate) dead_bytes: u64,
    /// Whether the write buffers are past their soft limit.
    pub(crate) overlay_full: bool,
}

/// A field's queue at a glance: what the worker has to do and how far it has
/// come. Reported per field by the index health report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct FieldQueueProgress {
    /// Changes waiting to be applied.
    pub pending: u64,
    /// How long the oldest of them has waited, in milliseconds (0 when none).
    pub oldest_pending_ms: u64,
    /// The highest write seq whose change is applied.
    pub applied_up_to_seq: u64,
    /// Changes queued since the field was activated.
    pub queued: u64,
    /// Changes the worker applied.
    pub applied_by_worker: u64,
    /// Changes writers applied themselves (queue over its cap, no worker
    /// running, or a field reindex).
    pub applied_by_writers: u64,
    /// Changes the index checkpoint or `sync_field_indexes` applied.
    pub applied_by_barriers: u64,
    /// Locked batches the worker applied.
    pub worker_batches: u64,
}

/// One field index's changes in the order writes queued them, waiting to be
/// applied (see the module comment).
#[derive(Default)]
pub(crate) struct ChangeQueue {
    items: Mutex<VecDeque<Queued>>,
    queued: AtomicU64,
    applied_up_to_seq: AtomicU64,
    applied_by_worker: AtomicU64,
    applied_by_writers: AtomicU64,
    applied_by_barriers: AtomicU64,
    worker_batches: AtomicU64,
}

impl ChangeQueue {
    /// Queue `change` to `key`'s row, made by the write at user seq `seq`.
    /// Called under the key's bucket lock, so a key's changes queue in the order
    /// its versions apply; memory only. Returns the queue's length after it.
    pub(crate) fn push(&self, key: &[u8], row_id: u128, change: QueuedChange, outcome: Option<ChangeOutcome>, seq: u64) -> usize {
        self.queued.fetch_add(1, Ordering::Relaxed);
        let mut items = self.items.lock();
        items.push_back(Queued {
            row_id,
            change,
            key: key.to_vec(),
            outcome,
            seq,
            queued_at: Instant::now(),
        });
        items.len()
    }

    /// How many changes wait to be applied.
    pub(crate) fn len(&self) -> usize {
        self.items.lock().len()
    }

    /// Apply at most `limit` of the changes queued so far (all of them for
    /// `None`) to `index`, in order, under its write lock, then spill the field
    /// if that took its write buffers past the hard limit (I/O, under the
    /// field's lock only). Called with no other lock held.
    pub(crate) fn apply_queued(&self, index: &RwLock<DynFieldIndex>, applier: Applier, limit: Option<usize>) -> Applied {
        let mut idx = index.write();
        let mut applied = self.apply_locked(&mut idx, applier, limit);
        if idx.over_hard_limit() {
            idx.spill_if_over_budget();
        }
        applied.dead_bytes = idx.reclaimable_dead_bytes();
        applied.overlay_full = idx.overlay_budget().over_soft();
        applied
    }

    /// As [`apply_queued`](Self::apply_queued), for a caller already holding
    /// the field's write lock (`idx`), and without the spill: the index
    /// checkpoint writes the buffers out itself.
    pub(crate) fn apply_locked(&self, idx: &mut DynFieldIndex, applier: Applier, limit: Option<usize>) -> Applied {
        // Taken under the field's write lock: whoever took changes before us
        // applied them before we got the lock, so they land in queue order.
        let batch: Vec<Queued> = {
            let mut items = self.items.lock();
            let n = limit.map_or(items.len(), |l| l.min(items.len()));
            items.drain(..n).collect()
        };
        let mut rejected = Vec::new();
        let mut max_seq = 0;
        for q in &batch {
            max_seq = max_seq.max(q.seq);
            if let Err(e) = idx.apply_without_spill(q.row_id, q.change.as_row_change()) {
                match &q.outcome {
                    Some(outcome) => *outcome.lock() = Some(e),
                    None => {
                        warn!("Index update rejected: {e}");
                        rejected.push(q.key.clone());
                    }
                }
            }
        }
        let count = batch.len();
        if count > 0 {
            self.applied_up_to_seq.fetch_max(max_seq, Ordering::Relaxed);
            let counter = match applier {
                Applier::Worker => {
                    self.worker_batches.fetch_add(1, Ordering::Relaxed);
                    &self.applied_by_worker
                }
                Applier::Writer => &self.applied_by_writers,
                Applier::Barrier => &self.applied_by_barriers,
            };
            counter.fetch_add(count as u64, Ordering::Relaxed);
        }
        Applied {
            count,
            rejected,
            dead_bytes: 0,
            overlay_full: false,
        }
    }

    /// The queue's progress counters and its oldest change's age.
    pub(crate) fn progress(&self) -> FieldQueueProgress {
        let (pending, oldest_pending_ms) = {
            let items = self.items.lock();
            let age = items.front().map_or(0, |q| q.queued_at.elapsed().as_millis() as u64);
            (items.len() as u64, age)
        };
        FieldQueueProgress {
            pending,
            oldest_pending_ms,
            applied_up_to_seq: self.applied_up_to_seq.load(Ordering::Relaxed),
            queued: self.queued.load(Ordering::Relaxed),
            applied_by_worker: self.applied_by_worker.load(Ordering::Relaxed),
            applied_by_writers: self.applied_by_writers.load(Ordering::Relaxed),
            applied_by_barriers: self.applied_by_barriers.load(Ordering::Relaxed),
            worker_batches: self.worker_batches.load(Ordering::Relaxed),
        }
    }
}
