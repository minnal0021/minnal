//! Size checks, flushes that move the WAL's cut, and vector index WAL upkeep.

use super::*;
use crate::namespace::NamespaceId;

impl Database {
    pub(in crate::db) fn check_closed(&self) -> Result<()> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(KVError::DatabaseClosed);
        }
        Ok(())
    }

    /// Reject a write whose key/value cannot be represented by the on-disk
    /// formats, with a clean [`KVError::WriteTooLarge`] instead of silently
    /// truncating a length deep in the store.
    ///
    /// **Two** bounds apply, and both must be checked here.
    ///
    /// 1. **Field widths.** Several persisted lengths are `u32`: the value
    ///    record's `value_len`, the row-ID map's `key_len`, and the whole write
    ///    is framed as one `u32`-sized WAL entry (key + value + op-name + rkyv
    ///    overhead). A single combined check — `key + value + headroom ≤
    ///    u32::MAX` — covers all three (key alone and value alone are subsets of
    ///    the entry). The headroom bounds the per-entry framing.
    /// 2. **Segment capacity.** A value record must fit inside one value-log
    ///    segment, whose size is `DbConfig::segment_size_bytes` (default 256
    ///    MiB, so *far* tighter than the ~4 GiB field-width bound).
    ///
    /// Checking only the first left a gap that produced an acknowledged write
    /// nobody could read: a value between the segment size and ~4 GiB passed
    /// here, got a fsynced WAL entry — so `put` returned `Ok` — and only then hit
    /// `ValueLogError::ValueTooLarge` inside `ValueLog::append`, where the write
    /// path is past its durability barrier and treats failures as best-effort.
    /// All three apply attempts failed identically (the value's size does not
    /// change between retries), `get` kept returning the previous value, and WAL
    /// replay hit the same error on the next open and wrote a fail log. **Both
    /// bounds must be enforced before the WAL append**, which is the only point
    /// at which a rejection can still mean "nothing happened".
    pub(super) fn check_write_size(&self, key: &[u8], value: &[u8]) -> Result<()> {
        // Generous headroom for WAL-entry framing (short op-name + rkyv) and the
        // value-record header — far larger than any of those.
        const HEADROOM: u64 = 64 * 1024;
        const FIELD_WIDTH_LIMIT: u64 = u32::MAX as u64 - HEADROOM;

        Self::check_total_len(key.len(), value.len(), FIELD_WIDTH_LIMIT, "lengths are stored as u32")?;

        let segment_size = self.config.segment_size_bytes;
        Self::check_total_len(
            key.len(),
            value.len(),
            crate::storage::value_log::max_record_payload(segment_size),
            &format!("a record must fit one value-log segment; value_log.segment_size_bytes = {segment_size}"),
        )
    }

    /// Inner of [`check_write_size`] with the limit as a parameter so both
    /// branches are testable without allocating a multi-GiB key/value.
    /// `reason` names which bound was hit, since the two have different fixes.
    pub(super) fn check_total_len(key_len: usize, value_len: usize, limit: u64, reason: &str) -> Result<()> {
        let total = key_len as u64 + value_len as u64;
        if total > limit {
            return Err(KVError::WriteTooLarge(format!(
                "key {key_len} + value {value_len} = {total} bytes exceeds the {limit}-byte limit ({reason})"
            )));
        }
        Ok(())
    }

    /// Flush every namespace holding no-WAL writes that exist only in memory.
    ///
    /// WAL-backed writes survive a crash because recovery replays them; no-WAL
    /// writes have no such copy, so whatever sits in the memtable when the
    /// process dies is gone. That is the intended trade for the vector index and
    /// the query-embedding cache — both are bulky and re-derivable — but a
    /// memtable is only flushed at capacity, so in practice a crash discarded
    /// *everything* those namespaces had accumulated. Observed: a `SIGKILL` left
    /// `stress_docs_sparse_vector` with 4 entries on disk against 6020 in
    /// memory, and reconciliation re-enqueued ~18000 documents whose embeddings
    /// had already been computed — hours of work against the embedding service.
    ///
    /// Running this on the compaction worker's tick bounds that loss to one tick
    /// of writes instead of a whole memtable. Only namespaces with no-WAL writes
    /// are flushed: WAL-backed ones are recoverable, are already flushed by
    /// [`flush_namespaces_pinning_wal`], and forcing extra flushes on them would
    /// buy no durability while adding L0 files for compaction to merge.
    ///
    /// Returns how many namespaces were flushed.
    ///
    /// [`flush_namespaces_pinning_wal`]: Self::flush_namespaces_pinning_wal
    pub(crate) fn flush_no_wal_memtables(&self) -> usize {
        let stores = self.stores.read();
        let mut flushed = 0;
        for (ns_id, kv_store) in stores.iter() {
            if !kv_store.has_unflushed_no_wal_writes() {
                continue;
            }
            match kv_store.flush_memtable_to_level0() {
                Ok(()) => flushed += 1,
                Err(e) => warn!("[LSM] Failed to flush no-WAL memtable for ns={}: {:?}", ns_id, e),
            }
        }
        flushed
    }

    /// Make every write already applied to the named namespaces durable, no-WAL
    /// writes included, by flushing each one's memtables to level 0. Names that
    /// are not registered are skipped (a dropped namespace must not be recreated).
    ///
    /// A store drop uses it to make its no-WAL deletes of the embed queue
    /// durable in one flush; the crash audits use it to choose which namespaces
    /// reach disk before a simulated crash. Unlike [`flush_no_wal_memtables`] it
    /// flushes whether or not the no-WAL flag is set, and a read-only memtable
    /// another thread is already flushing is waited for, not skipped.
    ///
    /// [`flush_no_wal_memtables`]: Self::flush_no_wal_memtables
    #[cfg(any(test, feature = "semantic-search"))]
    pub(crate) fn flush_namespaces(&self, names: &[String]) -> Result<()> {
        for name in names {
            let Some(ns_id) = self.registry.read().get_id(name) else {
                continue;
            };
            // Dropped between the lookup and here: nothing left to make durable.
            let Ok(store) = self.get_store(ns_id) else {
                continue;
            };
            store.flush_memtable_to_level0()?;
        }
        Ok(())
    }

    /// Flush every namespace that is holding the WAL persisted watermark back.
    ///
    /// The watermark can only advance to the slowest namespace's flushed offset,
    /// so a namespace that writes a few records and then goes idle would pin the
    /// whole WAL forever. Flushing those memtables to level 0 is what lets the
    /// watermark — and WAL GC behind it — move again. Returns how many
    /// namespaces were flushed.
    pub(crate) fn flush_namespaces_pinning_wal(&self) -> usize {
        let pinning = self.wal_flush_observer.namespaces_with_unflushed();
        if pinning.is_empty() {
            return 0;
        }
        let stores = self.stores.read();
        let mut flushed = 0;
        for ns_id in pinning {
            let Some(kv_store) = stores.get(&ns_id) else {
                continue;
            };
            match kv_store.flush_memtable_to_level0() {
                Ok(()) => flushed += 1,
                Err(e) => warn!("[WAL] Failed to flush ns={} while unpinning the WAL watermark: {:?}", ns_id, e),
            }
        }
        flushed
    }

    /// Flush every namespace holding the vector index WAL's retired cut back,
    /// as [`Self::flush_namespaces_pinning_wal`] does for the main WAL. A vector
    /// namespace that writes a little and goes idle would otherwise keep every
    /// later segment on disk. Returns how many were flushed.
    ///
    /// Seals the memtable even when it is empty: a namespace is released only
    /// by a seal and its flush, and one can hold the cut with an empty
    /// memtable (its last record was logged but never applied, for one).
    pub(crate) fn flush_namespaces_pinning_vector_wal(&self) -> usize {
        let holding = self.vector_index_wal.namespaces_holding_cut();
        if holding.is_empty() {
            return 0;
        }
        // Cloned so no flush (and its fsyncs) holds the store map, which
        // creating or dropping a namespace needs.
        let stores: Vec<(NamespaceId, Arc<KVStore>)> = {
            let stores = self.stores.read();
            holding.iter().filter_map(|ns| stores.get(ns).map(|s| (*ns, Arc::clone(s)))).collect()
        };
        let mut flushed = 0;
        for (ns_id, kv_store) in stores {
            match kv_store.seal_and_flush_to_level0() {
                Ok(()) => {
                    flushed += 1;
                    crate::support::metrics::Metrics::bump(&self.metrics.vector_wal_pin_flushes);
                }
                Err(e) => warn!("[vector index WAL] failed to flush ns={ns_id} holding the retired cut: {e:?}"),
            }
        }
        flushed
    }

    /// Whether any vector index WAL segment lies wholly below the retired cut.
    pub(crate) fn has_deletable_vector_wal_segments(&self) -> bool {
        self.vector_index_wal.has_deletable_segments().unwrap_or_else(|e| {
            warn!("[vector index WAL] could not list segments: {e}");
            false
        })
    }

    /// Delete vector index WAL segments wholly below the retired cut. Returns
    /// `(segments, bytes)` deleted.
    pub(crate) fn garbage_collect_vector_wal(&self) -> Result<(u64, u64)> {
        Ok(self.vector_index_wal.garbage_collect()?)
    }

    /// The vector index WAL's state: segments, tail, retired cut, and the
    /// namespaces holding the cut back.
    pub fn vector_index_wal_stats(&self) -> Result<VectorIndexWalStats> {
        // The registry is read per name, not held across the log's file I/O.
        Ok(self
            .vector_index_wal
            .stats(|ns| self.registry.read().get_name(ns).map_or_else(|| format!("#{ns}"), str::to_owned))?)
    }

    /// Save the WAL metadata, serialised with every progress save (see
    /// `WalPersistObserver::persist_lock`).
    pub(crate) fn flush_wal_metadata_internal(&self) -> Result<()> {
        Ok(self.wal_flush_observer.save_metadata()?)
    }
}
