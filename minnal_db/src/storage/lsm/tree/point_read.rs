//! Point reads: the newest copy of one key across every layer, with the shortcuts
//! that skip layers which cannot hold a newer copy.

use super::*;

impl LSMTree {
    pub(crate) fn get(&self, key: &[u8]) -> Result<Option<u128>> {
        Ok(self.get_with_seq(key)?.map(|(ptr, _)| ptr))
    }

    /// Fold a candidate `(value-or-tombstone, seq)` into the running best,
    /// keeping the newest sequence. Callers feed candidates newest-layer-first so
    /// an exact sequence tie stays with the newer copy.
    pub(super) fn merge_candidate(best: &mut Candidate, value: Option<u128>, seq: u64) {
        if best.is_none_or(|(_, b)| seq_newer(seq, b)) {
            *best = Some((value, seq));
        }
    }

    /// Like [`get`](Self::get) but also returns the stored sequence of the live
    /// value. Used by GC / journal replay to relocate a value's pointer while
    /// preserving its sequence, so the relocation neither loses to nor blocks a
    /// real write under highest-sequence-wins resolution.
    pub(crate) fn get_with_seq(&self, key: &[u8]) -> Result<Option<(u128, u64)>> {
        Ok(self.newest_copy(key)?.and_then(|(value, seq)| value.map(|ptr| (ptr, seq))))
    }

    /// The winning copy of `key` across every layer — `Some((Some(pointer), seq))`
    /// for a live value, `Some((None, seq))` when a **tombstone** wins, `None`
    /// when no layer holds the key at all. [`get_with_seq`](Self::get_with_seq)
    /// is this with tombstones read as absent; the write path needs the
    /// tombstone's seq to tell whether a write is newer than the delete.
    pub(super) fn newest_copy(&self, key: &[u8]) -> Result<Candidate> {
        // Resolve by highest write sequence across ALL layers. The writers keep
        // newer versions in newer layers (`docs/Layer-Order-Design.md`), but this
        // read does not rely on it: past the fast path below it inspects every
        // layer and keep the newest-sequence entry; a tombstone
        // winning means the key is deleted. Layers are visited newest-first so an
        // exact sequence tie resolves to the newer copy.
        let mut best: Candidate = None;

        if let Some(m) = self.metrics() {
            Metrics::bump(&m.lookups);
        }

        {
            let memtable = self.memtable.read();
            if let Some((value, seq, tombstone)) = memtable.skip_list.entry(key) {
                // Fast path: if this active-memtable entry is at least as new as
                // everything in the lower layers, it is authoritative — no buried
                // tombstone can be newer — so return without scanning below.
                if seq_newer_or_eq(seq, self.max_lower_seq.load(Ordering::Relaxed)) {
                    if let Some(m) = self.metrics() {
                        Metrics::bump(&m.fast_path_hits);
                    }
                    return Ok(Some(((!tombstone).then_some(value), seq)));
                }
                Self::merge_candidate(&mut best, (!tombstone).then_some(value), seq);
            }
        }

        {
            let ro_memtables = self.read_only_memtables.read();
            for ro in ro_memtables.iter().rev() {
                let hit = ro.find(key).map(|r| (r.value, r.seq, r.tombstone));
                if let Some((value, seq, tombstone)) = hit {
                    Self::merge_candidate(&mut best, (!tombstone).then_some(value), seq);
                }
            }
        }

        let bucket = get_bucket_for_key(key, self.config.num_buckets);
        match self.lookup_sstable(bucket, key, best.map(|(_, seq)| seq))? {
            SsLookup::Found(ptr, seq) => Self::merge_candidate(&mut best, Some(ptr), seq),
            SsLookup::Deleted(seq) => Self::merge_candidate(&mut best, None, seq),
            SsLookup::Missing => {}
        }

        Ok(best)
    }

    /// The pointer currently referenced for `key` (highest sequence across all
    /// layers), and whether a write carrying `new_seq` would become that key's
    /// winner under highest-sequence-wins resolution.
    ///
    /// An indexed namespace's write path needs both, under the bucket write lock:
    /// the displaced pointer to read the old value for a targeted index update,
    /// and whether the write wins, since a losing write must leave the index
    /// alone. Replay asks too, and drops a write that would lose.
    ///
    /// A winning **tombstone** counts: the pointer is `None` (nothing to displace),
    /// but a write older than the delete still loses to it. Reading that case as
    /// "absent, so any write wins" would make a lost put add a field-index row for
    /// a key that reads as deleted. `true` only when no layer holds the key at all,
    /// or the write is strictly newer than whatever does: seqs are unique, so a
    /// stored copy at `new_seq` is this same write, already applied (a replayed
    /// record whose memtable was flushed).
    pub(crate) fn current_pointer_and_wins(&self, key: &[u8], new_seq: u64) -> Result<(Option<u128>, bool)> {
        match self.newest_copy(key)? {
            Some((ptr, existing_seq)) => Ok((ptr, seq_newer(new_seq, existing_seq))),
            None => Ok((None, true)),
        }
    }

    /// Tri-state lookup of a key in the SSTable hierarchy (Level 0 then Level 1),
    /// resolved by highest sequence.
    ///
    /// `floor` is the sequence of the best candidate the caller already holds from
    /// a newer layer (the memtables), if any. A file whose newest entry is `<=` the
    /// best seen so far cannot win — an exact tie stays with the newer layer — so
    /// it is skipped unread. The result is still only this layer's best hit; the
    /// caller merges it with its own candidate.
    pub(super) fn lookup_sstable(&self, bucket: u32, key: &[u8], floor: Option<u64>) -> Result<SsLookup> {
        // L0 is newer than L1, so an exact sequence tie resolves to L0 (take L1
        // only if strictly newer). Layer order is not assumed to imply recency.
        let l0 = self.search_level0_files(bucket, key, floor)?;
        let floor = match (floor, l0.seq()) {
            (Some(f), Some(s)) => Some(if seq_newer(s, f) { s } else { f }),
            (f, s) => f.or(s),
        };
        let l1 = self.lookup_level1(bucket, key, floor)?;
        Ok(match (l0.seq(), l1.seq()) {
            (Some(s0), Some(s1)) if seq_newer(s1, s0) => l1,
            (Some(_), _) => l0,
            (None, _) => l1,
        })
    }

    /// Look the key up in this bucket's Level-1 file, using the max-seq, min/max,
    /// bloom-filter and sparse-index fast-rejects before any linear scan.
    pub(super) fn lookup_level1(&self, bucket: u32, key: &[u8], floor: Option<u64>) -> Result<SsLookup> {
        // L1 early-outs from its metadata: the key falls outside the file's
        // [min_key, max_key], or a newer layer already holds a copy at least as new
        // as anything in the file. The metadata is updated after the file swap and
        // before the merged L0 files are marked obsolete, so a reader that sees stale
        // metadata still sees those L0 files (and has folded their seqs into
        // `floor`) — this can never turn a live key into a false `Missing`. Empty
        // metadata (no L1 file yet, or it failed to load) skips the optimisation.
        {
            let metadata = self.sstable_metadata[bucket as usize].read();
            if let Some(meta) = metadata.first() {
                if key < meta.min_key.as_slice() || key > meta.max_key.as_slice() {
                    return Ok(SsLookup::Missing);
                }
                if let (Some(floor), Some(max_seq)) = (floor, meta.max_seq)
                    && seq_newer_or_eq(floor, max_seq)
                {
                    if let Some(m) = self.metrics() {
                        Metrics::bump(&m.seq_prunes);
                    }
                    return Ok(SsLookup::Missing);
                }
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

    /// Search the Level 0 files for a key and return the hit with the newest
    /// sequence (a value or a tombstone), or [`SsLookup::Missing`] if no L0 file
    /// mentions it.
    ///
    /// Resolution is by sequence, not file recency, so no hit settles the key on
    /// its own. Files are visited newest-first only so an exact sequence tie
    /// resolves to the newer file.
    ///
    /// A file is skipped unread when its summary proves it cannot matter: the key
    /// is outside its range or bloom-negative, or its newest entry is `<=` the best
    /// sequence seen so far (`floor` from the memtables, or an earlier file here).
    pub(super) fn search_level0_files(&self, bucket: u32, key: &[u8], floor: Option<u64>) -> Result<SsLookup> {
        let mut entries = self.level0_entries_snapshot(bucket);
        if entries.is_empty() {
            return Ok(SsLookup::Missing);
        }

        entries.sort_by_key(|b| std::cmp::Reverse(b.created_at_ms));

        let mut best = SsLookup::Missing;
        let mut probed = false;
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
            let best_seq = match (floor, best.seq()) {
                (Some(f), Some(b)) => Some(if seq_newer(b, f) { b } else { f }),
                (f, b) => f.or(b),
            };
            if best_seq.is_some_and(|s| entry.cannot_beat(s)) {
                if let Some(m) = self.metrics() {
                    Metrics::bump(&m.seq_prunes);
                }
                continue;
            }
            if !probed {
                probed = true;
                if let Some(m) = self.metrics() {
                    Metrics::bump(&m.l0_probes);
                }
            }
            let _guard = L0ReadGuard::new(Arc::clone(&entry));
            let hit = self.lookup_in_sstable_file_path(entry.path.clone(), key, entry.scan_start(key))?;
            if let Some(seq) = hit.seq()
                && best.seq().is_none_or(|b| seq_newer(seq, b))
            {
                best = hit;
            }
        }

        Ok(best)
    }
}
