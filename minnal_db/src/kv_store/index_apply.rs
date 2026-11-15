//! Applying a store's queued field-index changes: what a write does with its
//! changes once its bucket lock is released, and the passes the field-index
//! worker and the barriers run (see `kv_store/index_queue.rs`).

use super::*;
use crate::field_index::{DynFieldIndex, FieldId};
use crate::kv_store::index_queue::{APPLY_BATCH, Applier, ChangeQueue, FieldQueueProgress, QUEUE_CAP};
use crate::kv_store::writes::{FieldQueue, IndexFollowUp};

impl KVStore {
    /// Leave a write's queued field-index changes to the field-index worker,
    /// and return: the write path does no index work (root `CLAUDE.md`, *Keep
    /// the write path to what durability and order need*). The write applies
    /// them itself only when nothing else will soon: no worker runs (recovery
    /// at open, an engine opened without workers), or a queue it pushed to is
    /// past [`QUEUE_CAP`] because the worker has fallen behind, when it applies
    /// batches until that queue is back under the cap.
    pub(super) fn hand_off_index_follow_up(&self, follow_up: IndexFollowUp) {
        if follow_up.fields.is_empty() {
            return;
        }
        if let Some(m) = self.metrics() {
            Metrics::add(&m.index_changes_queued, follow_up.fields.len() as u64);
        }
        let wake = self.index_apply_wake.read().clone();
        let Some(wake) = wake else {
            self.finish_index_follow_up(follow_up);
            return;
        };
        wake.wake();
        if follow_up.over_cap {
            for (field_id, index, queue) in &follow_up.fields {
                while queue.len() > QUEUE_CAP {
                    if self.apply_field_queue(*field_id, index, queue, Applier::Writer, Some(APPLY_BATCH)) == 0 {
                        break;
                    }
                }
            }
        }
    }

    /// Apply every change queued on the fields of `follow_up` now, the
    /// caller's included: for a write with no worker to hand its changes to,
    /// and for a field reindex, which reports its own change's outcome.
    pub(super) fn finish_index_follow_up(&self, follow_up: IndexFollowUp) {
        for (field_id, index, queue) in &follow_up.fields {
            self.apply_field_queue(*field_id, index, queue, Applier::Writer, None);
        }
    }

    /// The field-index worker's pass over this store: apply every field's
    /// queue, at most [`APPLY_BATCH`] changes per hold of a field's write lock,
    /// round-robin over the fields so a busy one does not hold up the rest.
    /// Returns the changes applied.
    pub(crate) fn apply_index_queues(&self) -> usize {
        self.drain_index_queues(Applier::Worker)
    }

    /// Apply every change queued on this store's fields so far, as a barrier:
    /// once it returns, every write that returned before the call is in the
    /// field indexes (`AsyncDb::sync_field_indexes`). Returns the changes
    /// applied.
    pub(crate) fn sync_index_queues(&self) -> usize {
        self.drain_index_queues(Applier::Barrier)
    }

    /// Apply, batch by batch and field by field in turn, what each field had
    /// queued when the call began, so a stream of new writes cannot keep it
    /// going. The worker is woken again by the writes that came meanwhile.
    fn drain_index_queues(&self, applier: Applier) -> usize {
        let mut owed: Vec<(FieldQueue, usize)> = self
            .namespace_index
            .read()
            .iter()
            .map(|e| ((e.field_id, Arc::clone(&e.index), Arc::clone(&e.queue)), e.queue.len()))
            .filter(|(_, len)| *len > 0)
            .collect();
        let mut total = 0;
        while !owed.is_empty() {
            for ((field_id, index, queue), left) in &mut owed {
                let n = self.apply_field_queue(*field_id, index, queue, applier, Some(APPLY_BATCH.min(*left)));
                // A short batch means someone else (the checkpoint, a capped
                // writer) applied the rest.
                *left = if n < APPLY_BATCH.min(*left) { 0 } else { *left - n };
                total += n;
            }
            owed.retain(|(_, left)| *left > 0);
        }
        total
    }

    /// Apply up to `limit` of a field's queued changes under its write lock,
    /// then report what that left to do: rejected updates become gap records,
    /// and a field's dead space or full write buffers ask for an early
    /// checkpoint. Returns the changes applied.
    pub(crate) fn apply_field_queue(
        &self,
        field_id: FieldId,
        index: &parking_lot::RwLock<DynFieldIndex>,
        queue: &ChangeQueue,
        applier: Applier,
        limit: Option<usize>,
    ) -> usize {
        let applied = queue.apply_queued(index, applier, limit);
        self.note_applied_index_changes(field_id, applier, applied.count, applied.rejected);
        // Disk: if the field's blob store has piled up enough dead space (old
        // bitmap copies left by spills), ask the checkpoint worker to compact
        // early instead of waiting for the next tick (debounced inside the
        // trigger).
        self.request_checkpoint_if_over_cap(applied.dead_bytes);
        // Memory: past the write buffers' soft limit, ask for a checkpoint to
        // write them out.
        if applied.overlay_full {
            self.request_checkpoint_now();
        }
        applied.count
    }

    /// Count `count` changes applied to `field_id` by `applier`, and report the
    /// keys whose change the field rejected: a rejected update leaves the row
    /// absent from the field for good, so it becomes a durable gap record,
    /// which index health reports and a repair fixes, rather than a log line.
    pub(crate) fn note_applied_index_changes(&self, field_id: FieldId, applier: Applier, count: usize, rejected: Vec<Vec<u8>>) {
        if count > 0
            && let Some(m) = self.metrics()
        {
            let counter = match applier {
                Applier::Worker => {
                    Metrics::bump(&m.index_apply_batches);
                    &m.index_changes_applied_by_worker
                }
                Applier::Writer => &m.index_changes_applied_by_writers,
                Applier::Barrier => &m.index_changes_applied_by_barriers,
            };
            Metrics::add(counter, count as u64);
        }
        if rejected.is_empty() {
            return;
        }
        if let Some(sink) = self.index_gap_sink.read().as_ref() {
            for key in rejected {
                sink.note_rejected_update(self.namespace_id, field_id, &key);
            }
        }
    }

    /// Each active field's queue progress, by field id.
    pub(crate) fn index_queue_progress(&self) -> Vec<(FieldId, FieldQueueProgress)> {
        self.namespace_index.read().iter().map(|e| (e.field_id, e.queue.progress())).collect()
    }

    /// Request an index checkpoint now, if the checkpoint worker is running.
    /// Debounced inside the trigger. O(1) and non-blocking.
    fn request_checkpoint_now(&self) {
        if let Some(trigger) = self.index_checkpoint_trigger.read().as_ref() {
            trigger.request();
        }
    }

    /// Signal the index-checkpoint backpressure valve, if wired, with a field's
    /// current reclaimable dead blob bytes. O(1) and non-blocking.
    fn request_checkpoint_if_over_cap(&self, dead_bytes: u64) {
        if let Some(trigger) = self.index_checkpoint_trigger.read().as_ref() {
            trigger.request_if_over_cap(dead_bytes);
        }
    }
}
