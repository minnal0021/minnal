//! A field index's queue of row changes: how a write changes the index in
//! version order without waiting for the index under its bucket lock.
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
//! mutex, memory only), and after releasing it **applies** every change queued
//! so far under the field's write lock ([`ChangeQueue::apply_queued`]). Changes
//! are taken off the queue only by a holder of that write lock, so they apply in
//! queue order, and a writer that has taken the lock finds its own change
//! applied, by itself or by an earlier writer. A write therefore returns only
//! once its change is in the index, as the index checkpoint's in-flight
//! accounting requires.

use std::sync::Arc;

use log::warn;
use parking_lot::{Mutex, RwLock};

use crate::field_index::{DynFieldIndex, IndexValue, RowChange};

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

struct Queued {
    row_id: u128,
    change: QueuedChange,
    /// The key the row belongs to, for the gap record a rejection leaves.
    key: Vec<u8>,
    outcome: Option<ChangeOutcome>,
}

/// What applying the queue left for the caller to do.
pub(crate) struct Applied {
    /// Keys whose change the field rejected, to record as index gaps.
    pub(crate) rejected: Vec<Vec<u8>>,
    /// The field's reclaimable dead blob bytes, for the checkpoint trigger.
    pub(crate) dead_bytes: u64,
    /// Whether the write buffers are past their soft limit.
    pub(crate) overlay_full: bool,
}

/// One field index's changes in the order writes queued them, waiting to be
/// applied (see the module comment).
#[derive(Default)]
pub(crate) struct ChangeQueue {
    items: Mutex<Vec<Queued>>,
}

impl ChangeQueue {
    /// Queue `change` to `key`'s row. Called under the key's bucket lock, so a
    /// key's changes queue in the order its versions apply; memory only.
    pub(crate) fn push(&self, key: &[u8], row_id: u128, change: QueuedChange, outcome: Option<ChangeOutcome>) {
        self.items.lock().push(Queued {
            row_id,
            change,
            key: key.to_vec(),
            outcome,
        });
    }

    /// How many changes wait to be applied.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.items.lock().len()
    }

    /// Apply every change queued so far to `index`, in order, under its write
    /// lock, then spill the field if that took its write buffers past the hard
    /// limit (I/O, under the field's lock only). Called with no other lock held.
    pub(crate) fn apply_queued(&self, index: &RwLock<DynFieldIndex>) -> Applied {
        let mut idx = index.write();
        // Taken under the write lock: whoever took changes before us applied
        // them before we got the lock, so ours is applied when this returns.
        let items = std::mem::take(&mut *self.items.lock());
        let mut rejected = Vec::new();
        for q in items {
            if let Err(e) = idx.apply_without_spill(q.row_id, q.change.as_row_change()) {
                match q.outcome {
                    Some(outcome) => *outcome.lock() = Some(e),
                    None => {
                        warn!("Index update rejected: {e}");
                        rejected.push(q.key);
                    }
                }
            }
        }
        if idx.over_hard_limit() {
            idx.spill_if_over_budget();
        }
        Applied {
            rejected,
            dead_bytes: idx.reclaimable_dead_bytes(),
            overlay_full: idx.overlay_budget().over_soft(),
        }
    }
}
