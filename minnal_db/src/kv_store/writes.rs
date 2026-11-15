//! Writes: put and delete, keeping the live field indexes current.

use super::*;
use crate::field_index::{DynFieldIndex, FieldId};
use crate::kv_store::index_queue::{ChangeOutcome, ChangeQueue, QueuedChange};
use crate::kv_store::index_set::IndexEntry;

/// The fields a write queued index changes on under its bucket lock, for
/// [`KVStore::finish_index_follow_up`] to apply once the lock is released
/// (see `kv_store/index_queue.rs`).
#[derive(Default)]
pub(super) struct IndexFollowUp {
    fields: Vec<(FieldId, Arc<parking_lot::RwLock<DynFieldIndex>>, Arc<ChangeQueue>)>,
}

impl IndexFollowUp {
    /// Queue `change` to `key`'s row in `entry`'s field. Called under the key's
    /// bucket lock; memory only. A rejection is recorded as an index gap, or
    /// left in `outcome` for a caller that reports it itself.
    fn queue(&mut self, entry: &IndexEntry, key: &[u8], row_id: u128, change: QueuedChange, outcome: Option<ChangeOutcome>) {
        entry.queue.push(key, row_id, change, outcome);
        self.fields.push((entry.field_id, Arc::clone(&entry.index), Arc::clone(&entry.queue)));
    }
}

impl KVStore {
    /// Put a key-value pair into storage (LSM + value log).
    /// WAL writing is handled by the caller (Database coordinator).
    ///
    /// Allocates the write sequence from the store's counter: for standalone
    /// tests. Every database write path uses
    /// [`put_to_storage_seq`](Self::put_to_storage_seq) with the seq it drew
    /// under the key's stripe, so live order equals recovery's.
    pub fn put_to_storage(&self, key: &[u8], value: &[u8]) -> Result<()> {
        self.put_to_storage_inner(key, value, self.alloc_seq(), false)
    }

    /// Put a key-value pair carrying an explicit global write sequence, so
    /// concurrent writes to the same key resolve highest-sequence-wins
    /// (live == recovery). `seq` is the full global `u64` sequence, stored at full
    /// width in both the LSM and the value record.
    pub fn put_to_storage_seq(&self, key: &[u8], value: &[u8], seq: u64) -> Result<()> {
        self.put_to_storage_inner(key, value, seq, false)
    }

    /// A put replayed from a log at its original `seq` (see
    /// [`replay_upsert`](Self::replay_upsert)): dropped unless it is newer than
    /// every stored copy of the key.
    pub(super) fn put_replayed(&self, key: &[u8], value: &[u8], seq: u64) -> Result<()> {
        self.put_to_storage_inner(key, value, seq, true)
    }

    fn put_to_storage_inner(&self, key: &[u8], value: &[u8], seq: u64, replayed: bool) -> Result<()> {
        let epoch = current_epoch_millis();
        // Only field indexes need to know what this write replaces: its previous
        // value (for an `O(1)` targeted update instead of a scan of every value)
        // and whether this write is the key's newest at all. A namespace without
        // them writes blind: no LSM lookup, no value-log read.
        let indexed = !self.namespace_index.read().is_empty();
        // The row id is the key's for good (ids are never freed), so it is
        // resolved before the bucket lock: allocating one can wait for a row-map
        // flush, and nothing may wait for I/O under that lock.
        let row_id = indexed.then(|| self.resolve_row_id_alloc(key));

        // Hold the bucket lock across the value-log append, the LSM insert and
        // queueing the field-index change. GC's and TTL's compare-and-set runs
        // under it, so no write applies between their check and their insert;
        // and every change to a key's index rows is queued under it, so they
        // apply in the order the key's versions do — a TTL tombstone and a put
        // racing it (TTL takes no stripe) cannot change the index the other way
        // round. The change is applied after the lock (`finish_index_follow_up`):
        // the field's own lock can be held for milliseconds by a query or the
        // checkpoint, and no write to the bucket may wait on it. Same-key user
        // writes are serialised by the key stripe as well.
        let bucket = self.value_log.bucket_for_key(key);
        let bucket_guard = self.value_log.lock_bucket_for_write(bucket)?;

        if replayed && !self.replay_wins(key, seq)? {
            return Ok(());
        }

        // A losing write (older than the key's current copy) leaves the index
        // alone, so it reads nothing. Live writes never lose (the stripe), and a
        // losing replay was dropped above; the check stays as the index's guard. Which write is
        // newest is decided by `seq` alone, never by anything in the record.
        let (wins, key_existed, old_value) = if indexed {
            let (existing_u128, wins) = self.lsm.current_pointer_and_wins(key, seq)?;
            let displaced = if wins { existing_u128.and_then(decode_sharded_pointer) } else { None };
            let old_value = displaced.and_then(|old| self.value_log.read_value(old).ok());
            (wins, displaced.is_some(), old_value)
        } else {
            // Unknown, so assume the key existed: should a field index appear
            // mid-write, the update below then clears the row's old value instead
            // of leaving it beside the new one.
            (true, true, None)
        };

        let record_meta = ValueRecordMeta { epoch, seq };
        let pointer = self.value_log.append_to_locked_bucket(bucket, key, value, record_meta, false)?;
        let applied = match self.lsm.insert_with_seq(key, pointer.to_u128(), seq) {
            Ok(applied) => applied,
            Err(e) => {
                // A failed insert inserts nothing, so nothing points at the record
                // just appended: count it as garbage now. The apply is retried, each
                // attempt appending afresh, and bytes left counted as live would make
                // their segment look cleaner than it is to GC's selection.
                self.value_log.note_displaced(pointer, key.len());
                return Err(e.into());
            }
        };

        // Garbage accounting, no I/O: the pointer this write replaced in the
        // memtable, or its own record if it was stale there. A copy in a lower
        // layer is counted when the L0→L1 merge drops its entry (`MergeGarbage`).
        self.note_applied(applied, Some(pointer), key.len());

        // A lost write is not the key's current value, so it must not touch the
        // indices. (With no index this returns at once.)
        let follow_up = if wins {
            self.update_indices_on_put(key, row_id, value, old_value.as_deref(), key_existed)
        } else {
            IndexFollowUp::default()
        };
        drop(bucket_guard);
        // A user write supersedes any earlier one to the key that never applied.
        self.take_unapplied_write(key);
        self.finish_index_follow_up(follow_up);

        Ok(())
    }

    /// Queue the field indices' changes for a put. Called under the key's
    /// bucket lock, memory only; [`finish_index_follow_up`](Self::finish_index_follow_up)
    /// applies them once the lock is released.
    ///
    /// `old_value` is the document's *prior* stored bytes when this put replaced
    /// an existing key (`None` for a fresh insert, or when the prior bytes could
    /// not be read). Supplying it lets each field do an `O(1)` targeted update
    /// (move the row from its old value's bucket to the new one) instead of the
    /// `O(distinct values)` scan `remove_all_for_row` performs — a big win for
    /// high-cardinality fields. When the prior bytes are unavailable on a replace
    /// we fall back to the scan so the row never lingers in a stale bucket.
    /// `row_id` is the key's, resolved before the lock (`None` if no index
    /// existed then).
    fn update_indices_on_put(&self, key: &[u8], row_id: Option<u128>, value: &[u8], old_value: Option<&[u8]>, key_existed: bool) -> IndexFollowUp {
        let ns_index = self.namespace_index.read();
        let mut follow_up = IndexFollowUp::default();
        if ns_index.is_empty() {
            return follow_up;
        }
        // An index activated since the caller looked: resolve the id now.
        let row_id = row_id.unwrap_or_else(|| self.resolve_row_id_alloc(key));
        for entry in ns_index.iter() {
            let new_val = (entry.extractor)(value);
            let change = match (key_existed, old_value) {
                // Replace with the prior bytes in hand → O(1) targeted update.
                (true, Some(ov)) => {
                    let old_val = (entry.extractor)(ov);
                    if old_val == new_val {
                        continue;
                    }
                    QueuedChange::Update { old: old_val, new: new_val }
                }
                // Fresh insert: the row is in no bucket yet, so skip the scan.
                (false, _) => match new_val {
                    Some(v) => QueuedChange::Insert(v),
                    None => continue,
                },
                // Replace but prior bytes unreadable: scan to clear the old bucket,
                // then insert the new value, preserving one-value-per-row.
                (true, None) => match new_val {
                    Some(v) => QueuedChange::Set(v),
                    None => QueuedChange::RemoveAll,
                },
            };
            follow_up.queue(entry, key, row_id, change, None);
        }
        follow_up
    }

    /// Apply the index changes a write queued under its bucket lock, once that
    /// lock is released: under each field's write lock, with whatever other
    /// writes queued before it (`ChangeQueue::apply_queued`, which also spills a
    /// field past the overlay budget's hard limit). Then report rejected
    /// updates, and ask for an early checkpoint when a field's write buffers or
    /// dead space call for one. The write returns only after this, so its
    /// change is in the index when it does.
    fn finish_index_follow_up(&self, follow_up: IndexFollowUp) {
        for (field_id, index, queue) in follow_up.fields {
            let applied = queue.apply_queued(&index);
            for key in applied.rejected {
                // A rejected update leaves this row absent from the field for
                // good. Report it so it becomes a durable gap record, which index
                // health reports and a repair fixes, rather than only a log line.
                if let Some(sink) = self.index_gap_sink.read().as_ref() {
                    sink.note_rejected_update(self.namespace_id, field_id, &key);
                }
            }
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
        }
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

    /// Reindex a single field for a single key: re-derive the field value from
    /// the key's *current* stored bytes and rewrite its entry in that field's
    /// index, using the same extractor + `DynFieldIndex` ops as the put path
    /// (clear the row's old buckets, then insert the freshly-extracted value).
    ///
    /// This touches only the named field — it does not rewrite the stored value,
    /// re-run other fields' extractors, or trigger any vector re-embedding. Use
    /// it to repair a single document's entry in one field index.
    ///
    /// Returns [`FieldReindexOutcome`](crate::FieldReindexOutcome): `Reindexed` on success, `KeyNotFound`
    /// when the key has no value, `FieldNotActive` when `field_id` has no live
    /// index in this namespace.
    ///
    /// **An absent key still clears the row.** A key can be missing because its
    /// *delete* is what was lost — the index then holds a stale row in the old
    /// value's bucket, and a query returns a key that no longer exists. That is
    /// as wrong as a missing row, so the row is removed before reporting
    /// `KeyNotFound`; the outcome describes what was found, not whether work was
    /// done. (Removing a row that is already absent is a no-op, so this is safe
    /// for a key that simply never existed.)
    ///
    /// **Compare-and-set, like GC and TTL.** The value is read with no lock;
    /// the row is set under the key's bucket lock, and only if the key still
    /// points where it was read (pointers are unique, so that means the same
    /// write). Every write changes the key's rows under that lock, so a write
    /// that lands between the read and the set is caught and the reindex reads
    /// again. Set unchecked, a reindex that read v1 could undo a put's move of
    /// the row to v2. After `MAX_READ_ATTEMPTS` lost races the read is made
    /// under the lock, which nothing can overtake.
    pub fn reindex_field(&self, field_id: crate::field_index::FieldId, key: &[u8]) -> Result<crate::field_index::FieldReindexOutcome> {
        use crate::field_index::FieldReindexOutcome;

        if self.namespace_index.read().get(field_id).is_none() {
            return Ok(FieldReindexOutcome::FieldNotActive);
        }
        let bucket = self.value_log.bucket_for_key(key);
        for attempt in 1..=Self::MAX_READ_ATTEMPTS {
            let last = attempt == Self::MAX_READ_ATTEMPTS;
            let held = if last {
                Some(self.value_log.lock_bucket_for_write(bucket)?)
            } else {
                None
            };
            let pointer = self.lsm.get(key)?.and_then(decode_sharded_pointer);
            let value = match pointer {
                None => None,
                Some(p) => match self.value_log.read_value(p) {
                    Ok(value) => Some(value),
                    // GC moved it: read it where it is now.
                    Err(e) if e.is_segment_missing() && !last => continue,
                    Err(e) => return Err(e.into()),
                },
            };
            // `resolve_row_id_get` never allocates: a key with no row id was
            // never indexed, so there is nothing to clear.
            let row_id = match value {
                Some(_) => Some(self.resolve_row_id_alloc(key)),
                None => self.resolve_row_id_get(key),
            };
            let bucket_guard = match held {
                Some(guard) => guard,
                None => {
                    let guard = self.value_log.lock_bucket_for_write(bucket)?;
                    if self.lsm.get(key)?.and_then(decode_sharded_pointer) != pointer {
                        continue;
                    }
                    guard
                }
            };

            let ns_index = self.namespace_index.read();
            let Some(entry) = ns_index.get(field_id) else {
                return Ok(FieldReindexOutcome::FieldNotActive);
            };
            let mut follow_up = IndexFollowUp::default();
            let rejection: ChangeOutcome = Arc::default();
            let outcome = match (&value, row_id) {
                (None, None) => FieldReindexOutcome::KeyNotFound,
                (None, Some(row_id)) => {
                    follow_up.queue(entry, key, row_id, QueuedChange::RemoveAll, None);
                    FieldReindexOutcome::KeyNotFound
                }
                // Same per-field ops as `update_indices_on_put`'s
                // replace-without-prior-bytes branch: `Set` clears the row's
                // existing buckets (one value per row) and inserts the new one.
                (Some(value), row_id) => {
                    let row_id = row_id.expect("allocated above for a present key");
                    let change = match (entry.extractor)(value) {
                        Some(v) => QueuedChange::Set(v),
                        None => QueuedChange::RemoveAll,
                    };
                    follow_up.queue(entry, key, row_id, change, Some(Arc::clone(&rejection)));
                    FieldReindexOutcome::Reindexed
                }
            };
            drop(ns_index);
            drop(bucket_guard);
            self.finish_index_follow_up(follow_up);
            if let Some(e) = rejection.lock().take() {
                return Err(KVError::Io(std::io::Error::other(e)));
            }
            return Ok(outcome);
        }
        unreachable!("the last attempt holds the bucket lock and always returns")
    }

    /// Delete a key from storage (creates tombstone in LSM).
    /// WAL writing is handled by the caller. Allocates the sequence from the
    /// store's counter: for standalone tests. Every database delete path uses
    /// [`delete_from_storage_seq`](Self::delete_from_storage_seq), and TTL
    /// expiry deletes at a system version (`delete_if_current`).
    pub fn delete_from_storage(&self, key: &[u8]) -> Result<()> {
        self.delete_from_storage_inner(key, self.alloc_seq(), false)
    }

    /// Delete carrying an explicit global write sequence, so a delete racing a
    /// write to the same key resolves highest-sequence-wins (live == recovery).
    pub fn delete_from_storage_seq(&self, key: &[u8], seq: u64) -> Result<()> {
        self.delete_from_storage_inner(key, seq, false)
    }

    /// A delete replayed from a log at its original `seq` (see
    /// [`replay_delete`](Self::replay_delete)): dropped unless it is newer than
    /// every stored copy of the key.
    pub(super) fn delete_replayed(&self, key: &[u8], seq: u64) -> Result<()> {
        self.delete_from_storage_inner(key, seq, true)
    }

    fn delete_from_storage_inner(&self, key: &[u8], seq: u64, replayed: bool) -> Result<()> {
        // Hold the bucket lock across the LSM delete (and an indexed namespace's
        // pointer read), for the reason `put_to_storage_inner` gives: GC's
        // batched compare-and-set runs under it. GC's liveness scan (phase 1)
        // takes no lock; a delete landing after it is caught by the
        // compare-and-set, which re-reads the key under this lock and skips a
        // key that is gone.
        let row_id = self.row_id_for_delete(key);
        let bucket = self.value_log.bucket_for_key(key);
        let bucket_guard = self.value_log.lock_bucket_for_write(bucket)?;
        if replayed && !self.replay_wins(key, seq)? {
            return Ok(());
        }
        self.delete_with_bucket_locked(key, seq, row_id, bucket_guard)?;
        // A user write supersedes any earlier one to the key that never applied.
        self.take_unapplied_write(key);
        Ok(())
    }

    /// Whether a write replayed at `seq` is newer than every stored copy of
    /// `key`, in any layer, tombstones included. Called with the key's bucket
    /// lock held.
    ///
    /// Replay is the only path that applies a write at an old seq. Live writers
    /// draw their seq under the key's stripe, so theirs is always the newest;
    /// a replayed record can be older than a copy already stored: a namespace's
    /// safe offset is only a lower bound (records past it may be in L0 files
    /// already, at the same seq), the other log may hold a newer write of the
    /// key that is flushed and not replayed, and a no-WAL write (TTL expiry,
    /// `put_no_wal`, `merge_no_wal`) is in no log at all. Applied anyway, such a
    /// record would land in the memtable, a newer layer, holding an older
    /// version. An equal seq is the same write, already stored. So both are
    /// dropped: nothing is appended to the value log and nothing changes.
    fn replay_wins(&self, key: &[u8], seq: u64) -> Result<bool> {
        let (_pointer, wins) = self.lsm.current_pointer_and_wins(key, seq)?;
        Ok(wins)
    }

    /// `key`'s row id for a delete's index change, looked up **before** the
    /// bucket lock is taken: the row map's lock can be waiting on its flush.
    /// `None` when the namespace has no field index (nothing looked up);
    /// `Some(None)` for a key that was never indexed.
    pub(super) fn row_id_for_delete(&self, key: &[u8]) -> Option<Option<u128>> {
        (!self.namespace_index.read().is_empty()).then(|| self.resolve_row_id_get(key))
    }

    /// The body of a delete, entered with `key`'s value-log bucket write lock
    /// already held (`bucket_guard`), and its row id from
    /// [`row_id_for_delete`](Self::row_id_for_delete). The field indices'
    /// changes are queued before it is released and applied after, for the
    /// reason `put_to_storage_inner` gives.
    ///
    /// In a namespace with field indexes it asks whether this delete actually
    /// **wins**: a delete older than the key's current copy is dropped by the LSM,
    /// so it must not strip that live document's rows from the field indices.
    /// Doing so regardless would leave a live document missing from every query
    /// on the field whenever a delete loses its race. A namespace without them
    /// deletes blind, as it writes.
    pub(super) fn delete_with_bucket_locked(
        &self,
        key: &[u8],
        seq: impl Into<Version>,
        row_id: Option<Option<u128>>,
        bucket_guard: parking_lot::MutexGuard<'_, ()>,
    ) -> Result<()> {
        let seq = seq.into();
        // Read the prior value (under the bucket lock, so GC can't relocate it)
        // so index removal can target the old value's bucket instead of scanning
        // every bucket.
        let indexed = !self.namespace_index.read().is_empty();
        let (displaced, old_value) = if indexed {
            let (existing_u128, wins) = self.lsm.current_pointer_and_wins(key, seq)?;
            let displaced = if wins { existing_u128.and_then(decode_sharded_pointer) } else { None };
            (displaced, displaced.and_then(|old| self.value_log.read_value(old).ok()))
        } else {
            (None, None)
        };
        let applied = self.lsm.delete_with_seq(key, seq)?;

        // The deleted key's record is garbage now. Accounting only: the record
        // itself is never touched (its segment may be sealed and immutable). Counted
        // here if the memtable held it, else when the merge drops its entry.
        self.note_applied(applied, None, key.len());

        // Remove the row from the field indices only if this delete removed a value
        // (`displaced` is set only when it wins). A key with no current value has
        // nothing indexed, as the put path assumes for a new key, so skip the
        // `O(distinct values)` scan `update_indices_on_delete` falls back to
        // without old bytes; a row a lost update left behind is for gap records
        // and repair, not for every delete of an absent key. Without indexes this
        // is a no-op, unless one appeared mid-delete: then it clears the row.
        let follow_up = if displaced.is_some() || !indexed {
            self.update_indices_on_delete(key, row_id, old_value.as_deref())
        } else {
            IndexFollowUp::default()
        };
        drop(bucket_guard);
        self.finish_index_follow_up(follow_up);

        Ok(())
    }

    /// Queue the removal of a deleted key's row from the field indices (see
    /// [`update_indices_on_put`](Self::update_indices_on_put)).
    ///
    /// With the deleted document's prior bytes (`old_value`) in hand, each field
    /// removes the row from just its old value's bucket — `O(1)` instead of the
    /// `O(distinct values)` scan. Falls back to the scan only when those bytes
    /// could not be read, so a row never lingers in a stale bucket.
    fn update_indices_on_delete(&self, key: &[u8], row_id: Option<Option<u128>>, old_value: Option<&[u8]>) -> IndexFollowUp {
        let ns_index = self.namespace_index.read();
        let mut follow_up = IndexFollowUp::default();
        if ns_index.is_empty() {
            return follow_up;
        }
        // An index activated since the caller looked: look the id up now. No
        // allocation: a key the dense map has never seen has nothing indexed.
        let Some(row_id) = row_id.unwrap_or_else(|| self.resolve_row_id_get(key)) else {
            return follow_up;
        };
        for entry in ns_index.iter() {
            let change = match old_value.map(|ov| (entry.extractor)(ov)) {
                // Targeted: the row is under the old value's bucket.
                Some(Some(v)) => QueuedChange::Remove(v),
                // Field absent in the old doc → row was never indexed for it.
                Some(None) => continue,
                // Prior bytes unavailable → scan to be safe.
                None => QueuedChange::RemoveAll,
            };
            follow_up.queue(entry, key, row_id, change, None);
        }
        follow_up
    }

    /// Increment write count and return true if a sync should be triggered
    pub fn should_sync(&self) -> bool {
        let n = self.sync_config.records_per_sync;
        if n == 0 {
            return false;
        }
        let count = self.write_count.fetch_add(1, Ordering::Relaxed) + 1;
        count.is_multiple_of(n as u64)
    }

    pub fn sync_value_log(&self) -> Result<()> {
        self.value_log.sync_all().map_err(|e| {
            warn!("[KVStore:{}] Value log sync failed: {:?}", self.name, e);
            KVError::from(e)
        })
    }
}
