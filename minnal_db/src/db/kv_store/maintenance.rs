//! Flush, compaction, TTL expiry, WAL replay and stats.

use super::*;

impl KVStore {
    /// Delete records whose creation epoch is older than `ttl`, for a store with
    /// no concurrent same-key writers to exclude (tests and standalone use). The
    /// database's TTL worker uses [`expire_records_locked`](Self::expire_records_locked)
    /// with its key stripe.
    #[cfg(test)]
    pub(crate) fn expire_records(&self, ttl: Duration, max_deletes_per_run: usize) -> Result<usize> {
        self.expire_records_locked(ttl, max_deletes_per_run, |_| ())
    }

    /// Delete records whose creation epoch is older than `ttl`.
    ///
    /// Scans live keys and reads each record's `epoch` straight from the value log
    /// (a 36-byte header read — no value bytes). At most `max_deletes_per_run`
    /// records are removed per call so a single pass can't stall on a huge backlog;
    /// value-log GC reclaims the physical space afterwards. Returns the number
    /// deleted.
    ///
    /// Each expired key is deleted by **compare-and-set**: only if it still points
    /// at the exact record judged expired ([`delete_if_current`](Self::delete_if_current)).
    /// A key rewritten after the check — its fresh value just acknowledged to a
    /// writer — is left alone; the old code deleted whatever was current. And the
    /// delete runs under `lock_key(key)`, which the database supplies as the key
    /// stripe: a WAL-backed write holds that stripe from allocating its seq to
    /// applying it, so no same-key write can sit between the two while TTL
    /// allocates its own seq. Without it TTL deletes carried seqs out of order
    /// with concurrent puts, the one source of "lower-seq copy in a newer layer"
    /// that the read paths (and the L1 tombstone-drop contract) had to tolerate.
    pub(crate) fn expire_records_locked<G>(&self, ttl: Duration, max_deletes_per_run: usize, lock_key: impl Fn(&[u8]) -> G) -> Result<usize> {
        let now_millis = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64;
        let ttl_millis = ttl.as_millis() as u64;

        let mut deleted = 0usize;
        for key in self.keys()? {
            if deleted >= max_deletes_per_run {
                break;
            }
            // Resolved per key as the pass reaches it, not all up front: the
            // compare-and-set below only holds off writes after this point.
            let Some(pointer) = self.lsm.get(&key)?.and_then(decode_sharded_pointer) else {
                continue; // deleted since the key list was taken
            };
            let Ok(meta) = self.value_log.read_record_meta(pointer) else {
                continue;
            };
            if now_millis.saturating_sub(meta.epoch) >= ttl_millis {
                let _key_guard = lock_key(&key);
                if self.delete_if_current(&key, pointer)? {
                    deleted += 1;
                }
            }
        }

        Ok(deleted)
    }

    /// Delete `key` only if it still points at `expected` — the record a TTL pass
    /// judged expired. Returns whether it deleted. A key rewritten (or relocated by
    /// GC) since is left for the next pass to judge afresh; pointers are unique for
    /// the life of the database, so pointer equality means "the same write".
    ///
    /// The seq is allocated only after the check, under the bucket lock and the
    /// caller's key lock, so the delete is newer than anything already applied to
    /// the key and — with the key stripe held — than anything about to be.
    fn delete_if_current(&self, key: &[u8], expected: ShardedValuePointer) -> Result<bool> {
        let bucket = self.value_log.bucket_for_key(key);
        let bucket_guard = self.value_log.lock_bucket_for_write(bucket)?;
        if self.lsm.get(key)?.and_then(decode_sharded_pointer) != Some(expected) {
            return Ok(false);
        }
        let seq = self.alloc_seq();
        self.delete_with_bucket_locked(key, seq, bucket_guard)?;
        Ok(true)
    }

    /// Remove stale LSM files left by an interrupted compaction.
    ///
    /// The value log needs no startup cleanup any more: GC never writes a shadow file
    /// and never swaps one in, so there is nothing half-finished to roll forward or
    /// back. Its only mutation of existing state is unlinking a segment, which only
    /// happens after the re-point is durable.
    pub fn cleanup_old_files_on_startup(&self) -> Result<()> {
        self.lsm
            .cleanup_old_files_on_startup()
            .map_err(|e| KVError::Io(std::io::Error::other(format!("Failed to cleanup LSM old files: {:?}", e))))
    }

    /// Replay a single WAL entry into this namespace's storage, stamping the
    /// memtable with the entry's original WAL sequence so recovery reproduces the
    /// same per-key winner that was live before the crash (highest-sequence-wins).
    pub fn replay_upsert(&self, key: &[u8], value: &[u8], seq: u64) -> Result<()> {
        let epoch = current_epoch_millis();
        let mut next_version = 1u32;

        // Read the displaced pointer and decide the winner inside the bucket lock, as
        // the live put path does — replay reuses that path's accounting exactly.
        let bucket = self.value_log.bucket_for_key(key);
        let _bucket_guard = self.value_log.lock_bucket_for_write(bucket)?;

        let (existing_u128, wins) = self.lsm.current_pointer_and_wins(key, seq)?;
        let displaced = existing_u128.and_then(decode_sharded_pointer);
        if let Some(existing_ptr) = displaced
            && let Ok(meta) = self.value_log.read_record_meta(existing_ptr)
        {
            next_version = meta.version.saturating_add(1);
        }

        let record_meta = ValueRecordMeta {
            version: next_version,
            epoch,
            seq,
        };
        let pointer = self.value_log.append_to_locked_bucket(bucket, key, value, record_meta, false)?;
        self.lsm.insert_with_seq(key, pointer.to_u128(), seq)?;
        if wins {
            if let Some(old) = displaced {
                self.value_log.note_displaced(old, key.len());
            }
        } else {
            self.value_log.note_displaced(pointer, key.len());
        }
        drop(_bucket_guard);

        if wins {
            self.update_indices_on_put(key, value, None, displaced.is_some());
        }
        Ok(())
    }

    /// Replay a WAL delete into this namespace's storage, under the entry's original
    /// sequence (see [`replay_upsert`](Self::replay_upsert)).
    pub fn replay_delete(&self, key: &[u8], seq: u64) -> Result<()> {
        let bucket = self.value_log.bucket_for_key(key);
        let _bucket_guard = self.value_log.lock_bucket_for_write(bucket)?;
        let displaced = self.lsm.get(key).ok().flatten().and_then(decode_sharded_pointer);
        self.lsm.delete_with_seq(key, seq)?;
        if let Some(old) = displaced {
            self.value_log.note_displaced(old, key.len());
        }
        drop(_bucket_guard);

        self.update_indices_on_delete(key, None);
        Ok(())
    }

    pub fn flush_and_compact_all(&self) -> Result<()> {
        self.lsm.flush_and_compact_all().map_err(KVError::from)
    }

    /// Seal and flush the active memtable to level 0 without compacting.
    ///
    /// Used to unpin the WAL persisted watermark: a namespace holding
    /// un-flushed WAL-backed writes blocks reclamation for every namespace
    /// (the WAL is shared), and flushing is what makes those entries durable
    /// on disk and therefore skippable by recovery.
    pub fn flush_memtable_to_level0(&self) -> Result<()> {
        // Cleared before the flush, not after: a write landing mid-flush may or
        // may not be included, so leaving the flag set costs one redundant flush
        // next tick, whereas clearing afterwards could drop that write's only
        // record of being unflushed.
        let was_set = self.unflushed_no_wal_writes.swap(false, Ordering::AcqRel);
        let result = self.lsm.flush_memtable_to_level0().map_err(KVError::from);
        if result.is_err() && was_set {
            // The flush did not happen, so those no-WAL writes are still
            // memory-only — and they have no WAL entry to replay, so nothing
            // else will ever save them. Restoring the flag makes the next tick
            // retry; leaving it clear would silently drop this namespace out of
            // `flush_no_wal_memtables` for good and hand a crash the whole
            // memtable. Only restore what we cleared: a failed flush on a
            // WAL-backed namespace must not start claiming no-WAL writes.
            self.unflushed_no_wal_writes.store(true, Ordering::Release);
        }
        result
    }

    /// As [`Self::flush_memtable_to_level0`], but seals the memtable even when
    /// it is empty, so the flush observers always see a seal and a flush. The
    /// vector index WAL's GC pass uses it: a namespace holding that log's
    /// retired cut is released only by a seal, and its memtable can be empty.
    pub(crate) fn seal_and_flush_to_level0(&self) -> Result<()> {
        // Same flag handling as `flush_memtable_to_level0`.
        let was_set = self.unflushed_no_wal_writes.swap(false, Ordering::AcqRel);
        let result = self.lsm.seal_and_flush_to_level0().map_err(KVError::from);
        if result.is_err() && was_set {
            self.unflushed_no_wal_writes.store(true, Ordering::Release);
        }
        result
    }

    /// Record that a no-WAL write landed in this namespace's memtable.
    pub(crate) fn note_no_wal_write(&self) {
        self.unflushed_no_wal_writes.store(true, Ordering::Release);
    }

    /// Whether this namespace holds no-WAL writes that exist nowhere but memory.
    pub(crate) fn has_unflushed_no_wal_writes(&self) -> bool {
        self.unflushed_no_wal_writes.load(Ordering::Acquire)
    }

    #[allow(dead_code)]
    pub fn stats(&self) -> Stats {
        let mut total_gc_runs = 0u64;
        let mut total_bytes_reclaimed = 0u64;
        let mut live_bytes = 0u64;
        let mut garbage_bytes = 0u64;
        let mut segment_count = 0u64;
        for (_bucket, metadata) in self.value_log.all_bucket_metadata() {
            total_gc_runs = total_gc_runs.saturating_add(metadata.total_gc_runs);
            total_bytes_reclaimed = total_bytes_reclaimed.saturating_add(metadata.total_bytes_reclaimed);
            live_bytes = live_bytes.saturating_add(metadata.live_bytes());
            garbage_bytes = garbage_bytes.saturating_add(metadata.garbage_bytes());
            segment_count = segment_count.saturating_add(metadata.segments.len() as u64);
        }

        let written = live_bytes.saturating_add(garbage_bytes);
        let waste_ratio = if written > 0 {
            (garbage_bytes as f64 / written as f64) * 100.0
        } else {
            0.0
        };
        let disk_bytes: u64 = self.value_log.physical_stats().iter().map(|s| s.physical_bytes).sum();

        Stats {
            segment_count,
            disk_bytes,
            garbage_size: garbage_bytes,
            waste_ratio,
            total_gc_runs,
            total_bytes_reclaimed,
            live_bytes,
        }
    }

    pub fn compact_lsm(&self) -> Result<()> {
        if !self.lsm.has_compaction_work() {
            return Ok(());
        }
        self.lsm.compact_all().map_err(KVError::from)
    }

    pub fn has_lsm_compaction_work(&self) -> bool {
        self.lsm.has_compaction_work()
    }

    /// The background worker's compaction: flush read-only memtables and merge
    /// only the buckets whose merge is due (`LSMTree::compaction_due`).
    pub(crate) fn compact_lsm_due(&self) -> Result<()> {
        self.lsm.compact_due().map_err(KVError::from)
    }

    /// Whether [`compact_lsm_due`](Self::compact_lsm_due) has anything to do.
    pub(crate) fn has_due_lsm_compaction_work(&self) -> bool {
        self.lsm.has_due_compaction_work()
    }
}
