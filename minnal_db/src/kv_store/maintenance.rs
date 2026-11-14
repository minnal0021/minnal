//! Flush, compaction, TTL expiry, WAL replay and stats.

use super::*;
use std::ops::ControlFlow;

impl KVStore {
    /// Delete records whose creation epoch is older than `ttl`; see
    /// [`expire_records_with`](Self::expire_records_with).
    pub(crate) fn expire_records(&self, ttl: Duration, max_deletes_per_run: usize) -> Result<usize> {
        self.expire_records_with(ttl, max_deletes_per_run, |_| ())
    }

    /// Delete records whose creation epoch is older than `ttl`.
    ///
    /// Walks the live keys a page at a time ([`for_each_key_page`](Self::for_each_key_page)),
    /// so a pass holds one page of keys, and reads each record's `epoch` straight
    /// from the value log (a 32-byte header read, no value bytes). At most
    /// `max_deletes_per_run` records are removed per call so a single pass cannot
    /// stall on a huge backlog; value-log GC reclaims the physical space
    /// afterwards. Returns the number deleted.
    ///
    /// Each expired key is deleted by **compare-and-set**: only if it still points
    /// at the exact record judged expired ([`delete_if_current`](Self::delete_if_current)).
    /// A key rewritten after the page was read, its fresh value just acknowledged
    /// to a writer, is left alone. A pointer the page holds that GC has since
    /// moved fails the compare too, and the next pass judges the key afresh.
    ///
    /// Expiry is a system action, not a user write: its tombstone keeps the
    /// expired copy's user seq and takes the next system seq
    /// ([`Version::next_system`]), so it needs no key stripe. A user write that
    /// drew its seq before the delete and has not applied yet is newer, and wins
    /// whenever it lands, after a crash too. `before_delete` runs between the
    /// verdict and the compare-and-set: a test hook for landing a write there.
    pub(crate) fn expire_records_with(&self, ttl: Duration, max_deletes_per_run: usize, before_delete: impl Fn(&[u8])) -> Result<usize> {
        let now_millis = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64;
        let ttl_millis = ttl.as_millis() as u64;

        let mut deleted = 0usize;
        self.for_each_key_page(&[], None, |page| {
            for (key, pointer, _seq) in page {
                if deleted >= max_deletes_per_run {
                    return Ok(ControlFlow::Break(()));
                }
                let Some(pointer) = decode_sharded_pointer(pointer) else {
                    continue;
                };
                // A segment GC reclaimed since the page was read: the key has
                // moved, and the next pass reads it where it is now.
                let Ok(meta) = self.value_log.read_record_meta(pointer) else {
                    continue;
                };
                if now_millis.saturating_sub(meta.epoch) >= ttl_millis {
                    before_delete(&key);
                    if self.delete_if_current(&key, pointer)? {
                        deleted += 1;
                    }
                }
            }
            Ok(ControlFlow::Continue(()))
        })?;

        Ok(deleted)
    }

    /// Delete `key` only if it still points at `expected` — the record a TTL pass
    /// judged expired. Returns whether it deleted. A key rewritten (or relocated by
    /// GC) since is left for the next pass to judge afresh; pointers are unique for
    /// the life of the database, so pointer equality means "the same write".
    ///
    /// The tombstone's version is the expired copy's next system version, decided
    /// under the bucket lock with the compare. A copy whose system seq is used up
    /// is deleted as a user write instead: stripe, then bucket, a fresh user seq.
    fn delete_if_current(&self, key: &[u8], expected: ShardedValuePointer) -> Result<bool> {
        let bucket = self.value_log.bucket_for_key(key);
        {
            let bucket_guard = self.value_log.lock_bucket_for_write(bucket)?;
            let Some((pointer, version)) = self.lsm.get_with_seq(key)? else {
                return Ok(false);
            };
            if decode_sharded_pointer(pointer) != Some(expected) {
                return Ok(false);
            }
            if let Some(expiry) = version.next_system() {
                self.delete_with_bucket_locked(key, expiry, bucket_guard)?;
                return Ok(true);
            }
        }
        let _stripe = self.lock_key(key);
        let bucket_guard = self.value_log.lock_bucket_for_write(bucket)?;
        if self.lsm.get(key)?.and_then(decode_sharded_pointer) != Some(expected) {
            return Ok(false);
        }
        let seq = self.alloc_seq();
        self.delete_with_bucket_locked(key, Version::user(seq), bucket_guard)?;
        Ok(true)
    }

    /// Replay a WAL upsert into this namespace's storage under the entry's
    /// original sequence, so recovery reproduces the per-key winner that was live
    /// before the crash (highest-sequence-wins).
    ///
    /// Unlike a live write, a replayed one is applied only if it is newer than
    /// every stored copy of the key (see `replay_wins`): otherwise the memtable,
    /// the newest layer, would hold an older version than a layer below it. A
    /// dropped record touches nothing: no value-log append, no garbage count, no
    /// field-index change. Past that check it is the live put path, not a copy.
    pub fn replay_upsert(&self, key: &[u8], value: &[u8], seq: u64) -> Result<()> {
        self.put_replayed(key, value, seq)
    }

    /// Replay a WAL delete into this namespace's storage under the entry's
    /// original sequence, through the live delete path (see
    /// [`replay_upsert`](Self::replay_upsert) for why not a copy).
    pub fn replay_delete(&self, key: &[u8], seq: u64) -> Result<()> {
        self.delete_replayed(key, seq)
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
