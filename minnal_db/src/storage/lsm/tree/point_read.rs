//! Point reads: a key's current copy is the first one found, newest layer first —
//! the active memtable, the read-only memtables (newest first), the bucket's
//! Level-0 files (newest first), then Level 1. Every writer keeps newer versions
//! in newer layers (`docs/Layer-Order-Design.md`), so the first copy found is the
//! current one, and a tombstone found first shadows everything below it.

use super::*;

impl LSMTree {
    pub(crate) fn get(&self, key: &[u8]) -> Result<Option<u128>> {
        Ok(self.get_with_seq(key)?.map(|(ptr, _)| ptr))
    }

    /// Like [`get`](Self::get) but also returns the live value's version.
    pub(crate) fn get_with_seq(&self, key: &[u8]) -> Result<Option<(u128, Version)>> {
        Ok(self.current_copy(key)?.and_then(|(value, version)| value.map(|ptr| (ptr, version))))
    }

    /// The current copy of `key` — `Some((Some(pointer), version))` for a live
    /// value, `Some((None, version))` for a **tombstone**, `None` when no layer
    /// holds the key. [`get_with_seq`](Self::get_with_seq) is this with
    /// tombstones read as absent; the write path needs the tombstone's version
    /// to tell whether a write is newer than the delete.
    pub(super) fn current_copy(&self, key: &[u8]) -> Result<Candidate> {
        if let Some(m) = self.metrics() {
            Metrics::bump(&m.lookups);
        }

        if let Some((value, version, tombstone)) = self.memtable.read().skip_list.entry(key) {
            if let Some(m) = self.metrics() {
                Metrics::bump(&m.fast_path_hits);
            }
            return Ok(Some(((!tombstone).then_some(value), version)));
        }

        {
            let ro_memtables = self.read_only_memtables.read();
            for ro in ro_memtables.iter().rev() {
                if let Some(record) = ro.find(key) {
                    return Ok(Some(((!record.tombstone).then_some(record.value), record.version)));
                }
            }
        }

        let bucket = get_bucket_for_key(key, self.config.num_buckets);
        Ok(self.lookup_sstable(bucket, key)?.candidate())
    }

    /// The pointer `key` currently refers to, and whether a write at `version`
    /// is newer than the key's current copy.
    ///
    /// An indexed namespace's write path needs both, under the bucket write lock:
    /// the displaced pointer to read the old value for a targeted index update,
    /// and whether the write wins, since a losing write must leave the index
    /// alone. Replay asks too, and drops a write that would lose.
    ///
    /// A current **tombstone** counts: the pointer is `None` (nothing to
    /// displace), but a write older than the delete still loses to it. `true`
    /// only when no layer holds the key at all, or the write is strictly newer
    /// than the current copy: versions are unique, so an equal one is this same
    /// write, already applied (a replayed record whose memtable was flushed).
    pub(crate) fn current_pointer_and_wins(&self, key: &[u8], version: impl Into<Version>) -> Result<(Option<u128>, bool)> {
        let version = version.into();
        match self.current_copy(key)? {
            Some((ptr, current)) => Ok((ptr, version > current)),
            None => Ok((None, true)),
        }
    }

    /// The key's copy in this bucket's SSTables: the newest Level-0 file that
    /// holds it, else Level 1.
    pub(super) fn lookup_sstable(&self, bucket: u32, key: &[u8]) -> Result<SsLookup> {
        match self.search_level0_files(bucket, key)? {
            SsLookup::Missing => self.lookup_level1(bucket, key),
            hit => Ok(hit),
        }
    }

    /// Look the key up in this bucket's Level-1 file, using the min/max,
    /// bloom-filter and sparse-index fast-rejects before any linear scan.
    ///
    /// A merge publishes the new file, then its metadata, bloom and index, and
    /// only then marks the merged L0 files obsolete. A reader that found those
    /// files obsolete (and so came here) therefore reads the new L1's metadata;
    /// one that read them while live found the key there if they held it.
    pub(super) fn lookup_level1(&self, bucket: u32, key: &[u8]) -> Result<SsLookup> {
        // Empty metadata (no L1 file yet, or it failed to load) skips the check.
        {
            let metadata = self.sstable_metadata[bucket as usize].read();
            if let Some(meta) = metadata.first()
                && (key < meta.min_key.as_slice() || key > meta.max_key.as_slice())
            {
                return Ok(SsLookup::Missing);
            }
        }
        // Bloom early-out: a negative is exact, so skip the linear scan when the
        // filter says the key is definitely absent. A positive may be a false
        // positive, so fall through and scan.
        {
            let bloom = self.sstable_blooms[bucket as usize].read();
            if let Some(bloom) = bloom.as_ref()
                && !bloom.contains(key)
            {
                if let Some(m) = self.metrics() {
                    Metrics::bump(&m.bloom_rejects);
                }
                return Ok(SsLookup::Missing);
            }
        }
        // Sparse-index hint: binary-search to a near-by start offset so the scan
        // covers ~SAMPLE_INTERVAL entries instead of the whole file. The hint is
        // validated against the file inside `lookup_in_sstable_file_from`.
        let start_offset = {
            let index = self.sstable_indexes[bucket as usize].read();
            index.as_ref().map(|i| i.block_start(key)).unwrap_or(0)
        };
        let file = self.sstable_files[bucket as usize].read().clone();
        if let Some(m) = self.metrics() {
            Metrics::bump(&m.l1_probes);
        }
        self.lookup_in_sstable_file_from(&file, key, start_offset)
    }

    /// The key's copy in the newest Level-0 file that holds it (a value or a
    /// tombstone), or [`SsLookup::Missing`] if no L0 file mentions it. A file is
    /// skipped unread when its summary rules the key out (out of range, or
    /// bloom-negative).
    pub(super) fn search_level0_files(&self, bucket: u32, key: &[u8]) -> Result<SsLookup> {
        let mut entries = self.level0_entries_snapshot(bucket);
        if entries.is_empty() {
            return Ok(SsLookup::Missing);
        }
        entries.sort_by_key(|b| std::cmp::Reverse(b.created_at_ms));

        for entry in entries {
            if entry.is_obsolete() {
                continue;
            }
            if !entry.may_contain(key) {
                if let Some(m) = self.metrics() {
                    Metrics::bump(&m.l0_bloom_rejects);
                }
                continue;
            }
            if let Some(m) = self.metrics() {
                Metrics::bump(&m.l0_probes);
            }
            let _guard = L0ReadGuard::new(Arc::clone(&entry));
            match self.lookup_in_sstable_file_path(entry.path.clone(), key, entry.scan_start(key))? {
                SsLookup::Missing => {}
                hit => return Ok(hit),
            }
        }
        Ok(SsLookup::Missing)
    }
}
