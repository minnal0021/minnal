//! The memtables: writes into the active skip list, and the sealed read-only copies
//! kept as sorted record lists until they are flushed.

use super::*;

impl MemTable {
    pub(super) fn with_capacity(skip_list_capacity: usize) -> Self {
        Self {
            skip_list: SkipList::with_capacity(skip_list_capacity),
        }
    }

    /// Check if memtable should be flushed (approaching capacity)
    /// Flushes if ANY condition is true:
    /// 1. Capacity usage percentage >= threshold_percent, OR
    /// 2. Cannot insert a new key into in memory memtable, OR
    /// 3. Any internal arena (nodes, links, keys) has reached half of its u32 limit.
    pub(super) fn should_flush(&self, threshold_percent: usize) -> bool {
        // Get max capacity from SkipList
        let max_capacity = self.skip_list.max_capacity();

        // Calculate current usage (live nodes + tombstones)
        let live_nodes = self.skip_list.number_of_live_nodes();
        let tombstone_nodes = self.skip_list.number_of_tombstone_nodes();
        let total_nodes = live_nodes + tombstone_nodes;

        // Calculate usage percentage
        let usage_percent = (total_nodes * 100) / max_capacity;

        // Check if threshold percentage is exceeded
        let threshold_exceeded = usage_percent >= threshold_percent;

        // Check if we can still insert a reasonable sized key
        let cannot_insert = !self.skip_list.can_insert_key(&[0u8; 256]);

        // Check if any arena is approaching its u32 address-space limit
        let arenas_pressure = self.skip_list.arenas_half_full();

        // Flush if ANY condition is true
        threshold_exceeded || cannot_insert || arenas_pressure
    }

    /// Collect all entries for compaction
    pub(super) fn collect_records(&self) -> Vec<KeyValueRecord> {
        self.skip_list.collect_key_value_records()
    }
}

impl ReadOnlyMemTable {
    pub(super) fn new(records: Vec<KeyValueRecord>, version: u64) -> Self {
        // `find` binary-searches; an unsorted or duplicated dump would make
        // lookups miss silently rather than fail.
        debug_assert!(
            records.windows(2).all(|w| w[0].key < w[1].key),
            "read-only memtable records must be strictly ascending by key"
        );
        Self {
            records: Arc::new(records),
            version,
            flushed_to_level0: Arc::new(AtomicBool::new(false)),
            flush_lock: Arc::new(parking_lot::Mutex::new(())),
        }
    }

    pub(super) fn records(&self) -> Arc<Vec<KeyValueRecord>> {
        Arc::clone(&self.records)
    }

    /// The record for `key`, if this memtable holds one. Binary search: the
    /// records are the skip list's dump, which is in ascending byte order (its
    /// zero-padded big-endian prefix compare agrees with `[u8]`'s `Ord`) with one
    /// record per key — `new` checks both in debug builds.
    pub(super) fn find(&self, key: &[u8]) -> Option<&KeyValueRecord> {
        self.records
            .binary_search_by(|record| record.key.as_slice().cmp(key))
            .ok()
            .map(|i| &self.records[i])
    }

    pub(super) fn version(&self) -> u64 {
        self.version
    }

    pub(super) fn is_flushed_to_level0(&self) -> bool {
        self.flushed_to_level0.load(Ordering::SeqCst)
    }

    pub(super) fn mark_flushed_to_level0(&self) {
        self.flushed_to_level0.store(true, Ordering::SeqCst);
    }
}

impl std::fmt::Debug for ReadOnlyMemTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReadOnlyMemTable").field("version", &self.version).finish()
    }
}

impl LSMTree {
    /// Insert/update a value carrying an explicit global write sequence, with
    /// highest-sequence-wins conflict resolution (see
    /// [`SkipList::try_insert_with_seq`]). This is the production write path; it
    /// makes the in-memory winner for racing same-key writes match the winner
    /// recovery would pick (recovery replays in sequence order).
    ///
    /// The result says what the write did to the active memtable: the live
    /// pointer it replaced there, or that it was stale. A copy of the key in a
    /// lower layer is not consulted; the L0→L1 merge reports it when it drops it.
    pub(crate) fn insert_with_seq(&self, key: &[u8], value: u128, seq: u64) -> Result<Applied> {
        #[cfg(test)]
        if self.fail_next_insert.swap(false, Ordering::SeqCst) {
            return Err(LSMError::CapacityExceeded);
        }
        {
            let memtable = self.memtable.read();
            if memtable.should_flush(self.config.compaction_threshold_percent) {
                drop(memtable);
                self.flush_memtable()?;
            }
        }

        let mut memtable = self.memtable.write();
        match memtable.skip_list.try_insert_with_seq(key, value, seq) {
            Ok(applied) => Ok(applied),
            Err(crate::storage::lsm::skip_list::skip_list::InsertError::CapacityExceeded) => Err(LSMError::CapacityExceeded),
        }
    }

    /// Tombstone a key carrying an explicit global write sequence, with
    /// highest-sequence-wins resolution: a delete older than a concurrent write
    /// to the same key is dropped, matching recovery's sequence-ordered replay.
    /// The result is as for [`insert_with_seq`](Self::insert_with_seq).
    pub(crate) fn delete_with_seq(&self, key: &[u8], seq: u64) -> Result<Applied> {
        {
            let memtable = self.memtable.read();
            if memtable.should_flush(self.config.compaction_threshold_percent) {
                drop(memtable);
                self.flush_memtable()?;
            }
        }

        let mut memtable = self.memtable.write();
        // Record the tombstone directly — creating the node if the key is not
        // currently in the memtable, so the delete still shadows lower layers.
        // Honours the sequence guard: a newer write to this key is left untouched.
        match memtable.skip_list.insert_tombstone_with_seq(key, seq) {
            Ok(applied) => Ok(applied),
            Err(crate::storage::lsm::skip_list::skip_list::InsertError::CapacityExceeded) => Err(LSMError::CapacityExceeded),
        }
    }

    /// Raise the lower-layer bound to `seq` if it is newer, whenever
    /// entries move from the active memtable into a lower layer.
    pub(super) fn note_lower_seq(&self, seq: u64) {
        let mut cur = self.max_lower_seq.load(Ordering::Relaxed);
        while seq_newer(seq, cur) {
            match self.max_lower_seq.compare_exchange_weak(cur, seq, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(observed) => cur = observed,
            }
        }
    }
}
