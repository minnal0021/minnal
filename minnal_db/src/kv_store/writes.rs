//! Writes: put and delete, keeping the live field indexes current.

use super::*;

impl KVStore {
    /// Put a key-value pair into storage (LSM + value log).
    /// WAL writing is handled by the caller (Database coordinator).
    ///
    /// Allocates the write sequence from the store's counter. Used by non-WAL
    /// writes (TTL expiry, bulk load) and standalone tests. The WAL-backed path
    /// uses [`put_to_storage_seq`](Self::put_to_storage_seq) so the conflict-
    /// resolution sequence equals the WAL append order (live == recovery).
    pub fn put_to_storage(&self, key: &[u8], value: &[u8]) -> Result<()> {
        self.put_to_storage_inner(key, value, self.alloc_seq())
    }

    /// Put a key-value pair carrying an explicit global write sequence, so
    /// concurrent writes to the same key resolve highest-sequence-wins
    /// (live == recovery). `seq` is the full global `u64` sequence, stored at full
    /// width in both the LSM and the value record.
    pub fn put_to_storage_seq(&self, key: &[u8], value: &[u8], seq: u64) -> Result<()> {
        self.put_to_storage_inner(key, value, seq)
    }

    fn put_to_storage_inner(&self, key: &[u8], value: &[u8], seq: u64) -> Result<()> {
        let epoch = current_epoch_millis();
        let mut next_version = 1u32;
        // Capture the prior value's bytes so field-index updates can be O(1)
        // (targeted at the old value's bucket) instead of scanning every bucket.
        // Only worth a value-log read when the namespace actually has indexes.
        let want_old_for_index = !self.namespace_index.read().is_empty();

        // Hold the bucket lock across the existing-pointer read, the value-log append,
        // AND the LSM insert. Reading the displaced pointer *inside* the lock is
        // load-bearing for garbage accounting: two racing same-key writers must not both
        // observe the same old pointer and each mark it displaced (double-counting it and
        // leaking the intermediate record). GC re-point is excluded for the same reason.
        let bucket = self.value_log.bucket_for_key(key);
        let _bucket_guard = self.value_log.lock_bucket_for_write(bucket)?;

        let (existing_u128, wins) = self.lsm.current_pointer_and_wins(key, seq)?;
        let displaced = existing_u128.and_then(decode_sharded_pointer);
        let mut old_value: Option<Vec<u8>> = None;
        if let Some(existing_ptr) = displaced {
            if let Ok(meta) = self.value_log.read_record_meta(existing_ptr) {
                next_version = meta.version.saturating_add(1);
            }
            if want_old_for_index {
                old_value = self.value_log.read_value(existing_ptr).ok();
            }
        }

        let record_meta = ValueRecordMeta {
            version: next_version,
            epoch,
            seq,
        };
        let pointer = self.value_log.append_to_locked_bucket(bucket, key, value, record_meta, false)?;
        self.lsm.insert_with_seq(key, pointer.to_u128(), seq)?;

        // Account the record that is now garbage — accounting only, no I/O and no
        // in-place edit (the old record's segment may be sealed and immutable). If this
        // write won, that is the record it displaced; if it *lost* to a concurrent
        // higher-sequence write (so the LSM dropped our insert), our own just-appended
        // record is the dead one instead.
        if wins {
            if let Some(old) = displaced {
                self.value_log.note_displaced(old, key.len());
            }
        } else {
            self.value_log.note_displaced(pointer, key.len());
        }
        drop(_bucket_guard);

        // Update in-memory field indices. A lost write is not the key's current value,
        // so it must not touch the indices.
        if wins {
            self.update_indices_on_put(key, value, old_value.as_deref(), displaced.is_some());
        }

        Ok(())
    }

    /// Update field indices for a put.
    ///
    /// `old_value` is the document's *prior* stored bytes when this put replaced
    /// an existing key (`None` for a fresh insert, or when the prior bytes could
    /// not be read). Supplying it lets each field do an `O(1)` targeted update
    /// (move the row from its old value's bucket to the new one) instead of the
    /// `O(distinct values)` scan `remove_all_for_row` performs — a big win for
    /// high-cardinality fields. When the prior bytes are unavailable on a replace
    /// we fall back to the scan so the row never lingers in a stale bucket.
    pub(super) fn update_indices_on_put(&self, key: &[u8], value: &[u8], old_value: Option<&[u8]>, key_existed: bool) {
        let ns_index = self.namespace_index.read();
        if ns_index.is_empty() {
            return;
        }
        // Resolve only after confirming there are indexed fields — an unindexed
        // namespace has no row map loaded and nothing to resolve for.
        let row_id = self.resolve_row_id_alloc(key);
        for entry in ns_index.iter() {
            let new_val = (entry.extractor)(value);
            let mut idx = entry.index.write();
            let result = match (key_existed, old_value) {
                // Replace with the prior bytes in hand → O(1) targeted update.
                (true, Some(ov)) => {
                    let old_val = (entry.extractor)(ov);
                    idx.update(old_val.as_ref(), new_val.as_ref(), row_id)
                }
                // Fresh insert: the row is in no bucket yet, so skip the scan.
                (false, _) => match &new_val {
                    Some(v) => idx.insert(v, row_id),
                    None => Ok(()),
                },
                // Replace but prior bytes unreadable: scan to clear the old bucket,
                // then insert the new value, preserving one-value-per-row.
                (true, None) => match &new_val {
                    Some(v) => idx.set(v, row_id),
                    None => {
                        idx.remove_all_for_row(row_id);
                        Ok(())
                    }
                },
            };
            let dead_bytes = idx.reclaimable_dead_bytes();
            let overlay_full = idx.overlay_budget().over_soft();
            drop(idx);
            if let Err(e) = result {
                // A rejected update leaves this row absent from the field for
                // good. Report it so it becomes a durable gap record, which index
                // health reports and a repair fixes, rather than only a log line.
                warn!("[KVStore '{}'] Index update rejected for field {}: {}", self.name, entry.field_id, e);
                if let Some(sink) = self.index_gap_sink.read().as_ref() {
                    sink.note_rejected_update(self.namespace_id, entry.field_id, key);
                }
            }
            // Disk: if this field's blob store has piled up enough dead space
            // (old bitmap copies left by spills), ask the checkpoint worker to
            // compact early instead of waiting for the next tick (debounced
            // inside the trigger).
            self.request_checkpoint_if_over_cap(dead_bytes);
            // Memory: past the write buffers' soft limit, ask for a checkpoint
            // to write them out. (The hard limit is enforced by the index itself.)
            if overlay_full {
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
    pub fn reindex_field(&self, field_id: crate::field_index::FieldId, key: &[u8]) -> Result<crate::field_index::FieldReindexOutcome> {
        use crate::field_index::FieldReindexOutcome;

        let value = self.get(key)?;

        let ns_index = self.namespace_index.read();
        let Some(entry) = ns_index.get(field_id) else {
            return Ok(FieldReindexOutcome::FieldNotActive);
        };

        let Some(value) = value else {
            // `resolve_row_id_get` never allocates: a key with no row id was
            // never indexed, so there is nothing to clear.
            if let Some(row_id) = self.resolve_row_id_get(key) {
                entry.index.write().remove_all_for_row(row_id);
            }
            return Ok(FieldReindexOutcome::KeyNotFound);
        };

        // Same row-ID resolution and per-field ops as `update_indices_on_put`'s
        // replace-without-prior-bytes branch: `set` clears the row's existing
        // buckets (preserving one-value-per-row) and inserts the new value.
        let row_id = self.resolve_row_id_alloc(key);
        let new_val = (entry.extractor)(&value);
        let mut idx = entry.index.write();
        match &new_val {
            Some(v) => idx.set(v, row_id).map_err(|e| KVError::Io(std::io::Error::other(e)))?,
            None => idx.remove_all_for_row(row_id),
        }
        Ok(FieldReindexOutcome::Reindexed)
    }

    /// Delete a key from storage (creates tombstone in LSM).
    /// WAL writing is handled by the caller. Allocates the sequence from the
    /// store's counter (used by TTL expiry / standalone tests); the WAL-backed
    /// path uses [`delete_from_storage_seq`](Self::delete_from_storage_seq).
    pub fn delete_from_storage(&self, key: &[u8]) -> Result<()> {
        self.delete_from_storage_inner(key, self.alloc_seq())
    }

    /// Delete carrying an explicit global write sequence, so a delete racing a
    /// write to the same key resolves highest-sequence-wins (live == recovery).
    pub fn delete_from_storage_seq(&self, key: &[u8], seq: u64) -> Result<()> {
        self.delete_from_storage_inner(key, seq)
    }

    fn delete_from_storage_inner(&self, key: &[u8], seq: u64) -> Result<()> {
        // Hold the bucket lock across the garbage accounting AND the LSM delete,
        // for the same reason `put_to_storage` does (see its comment): GC scans
        // the LSM for live keys under this lock and then re-inserts the pointers
        // of the records it copied. Without the lock, a delete landing in GC's
        // scan→reinsert window would be undone — GC would copy the about-to-be
        // deleted record and re-insert its pointer *after* our `lsm.delete`,
        // resurrecting the key. Taking the lock serialises us with GC so the
        // delete is either fully visible to GC's scan or applied after GC ends.
        let bucket = self.value_log.bucket_for_key(key);
        let bucket_guard = self.value_log.lock_bucket_for_write(bucket)?;
        self.delete_with_bucket_locked(key, seq, bucket_guard)
    }

    /// The body of a delete, entered with `key`'s value-log bucket write lock
    /// already held (`bucket_guard`, released before the field indices are
    /// touched).
    ///
    /// Like the put path, it asks whether this delete actually **wins**: a delete
    /// older than the key's current copy is dropped by the LSM, so it must not
    /// strip that live document's rows from the field indices or charge its record
    /// as garbage. Doing both regardless would leave a live document missing from
    /// every query on the field whenever a delete loses its race.
    pub(super) fn delete_with_bucket_locked(&self, key: &[u8], seq: u64, bucket_guard: parking_lot::MutexGuard<'_, ()>) -> Result<()> {
        // Read the prior value (under the bucket lock, so GC can't relocate it)
        // so index removal can target the old value's bucket instead of scanning
        // every bucket. Only when the namespace has indexes to update.
        let want_old_for_index = !self.namespace_index.read().is_empty();
        let (existing_u128, wins) = self.lsm.current_pointer_and_wins(key, seq)?;
        let displaced = if wins { existing_u128.and_then(decode_sharded_pointer) } else { None };
        let old_value = match displaced {
            Some(existing_ptr) if want_old_for_index => self.value_log.read_value(existing_ptr).ok(),
            _ => None,
        };
        self.lsm.delete_with_seq(key, seq)?;

        // The deleted key's record is garbage now. Accounting only — the record itself
        // is never touched (its segment may be sealed and immutable), and liveness was
        // never read from the record anyway: the LSM is the authority.
        if let Some(old) = displaced {
            self.value_log.note_displaced(old, key.len());
        }
        drop(bucket_guard);

        // Remove row from all field indices — only if the key is now deleted.
        if wins {
            self.update_indices_on_delete(key, old_value.as_deref());
        }

        Ok(())
    }

    /// Remove a deleted key's row from the field indices.
    ///
    /// With the deleted document's prior bytes (`old_value`) in hand, each field
    /// removes the row from just its old value's bucket — `O(1)` instead of the
    /// `O(distinct values)` scan. Falls back to the scan only when those bytes
    /// could not be read, so a row never lingers in a stale bucket.
    pub(super) fn update_indices_on_delete(&self, key: &[u8], old_value: Option<&[u8]>) {
        let ns_index = self.namespace_index.read();
        if ns_index.is_empty() {
            return;
        }
        // No allocation: a key the dense map has never seen has nothing indexed.
        let Some(row_id) = self.resolve_row_id_get(key) else {
            return;
        };
        for entry in ns_index.iter() {
            let mut idx = entry.index.write();
            match old_value {
                // Targeted: the row is under the old value's bucket (if any).
                Some(ov) => {
                    if let Some(old_val) = (entry.extractor)(ov) {
                        idx.remove(&old_val, row_id);
                    }
                    // Field absent in the old doc → row was never indexed for it.
                }
                // Prior bytes unavailable → scan to be safe.
                None => idx.remove_all_for_row(row_id),
            }
            // A removal changes the value's bitmap like a put does: signal both
            // valves the same way.
            let dead_bytes = idx.reclaimable_dead_bytes();
            let overlay_full = idx.overlay_budget().over_soft();
            drop(idx);
            self.request_checkpoint_if_over_cap(dead_bytes);
            if overlay_full {
                self.request_checkpoint_now();
            }
        }
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
