//! Level-0 files: their summaries (key range, Bloom filter, max seq, sparse index),
//! read guards, registration in flush order, and range reads of one file.

use super::*;

/// At most `limit` entries of one registered Level-0 file in `[start, end)`,
/// in key order. A file whose key range misses `[start, end)` is not opened;
/// one with a sparse index is read from the block holding `start` (the hint is
/// checked against the file first), and the read stops at the first key at or
/// past `end`.
pub(super) fn read_l0_range(entry: &L0FileEntry, start: &[u8], end: Option<&[u8]>, limit: usize) -> Result<Vec<ScanEntry>> {
    if let Some(summary) = &entry.summary
        && (summary.max_key.as_slice() < start || end.is_some_and(|e| summary.min_key.as_slice() >= e))
    {
        return Ok(Vec::new());
    }
    let file = open_registered_l0(&entry.path)?;
    let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
    let hint = entry.scan_start(start);
    let offset = if hint != 0 && LSMTree::valid_scan_start(&file, hint, file_len, start) {
        hint
    } else {
        SSTABLE_DATA_START
    };
    let mut frames = FrameReader::new(&file, offset);
    let mut entries = Vec::new();
    while entries.len() < limit
        && let Some(body) = frames.next_frame(file_len)?
    {
        #[cfg(test)]
        entry.frames_read.fetch_add(1, Ordering::Relaxed);
        // SAFETY: `verify_sstable_payload` checked the frame's CRC, and a
        // registered L0 file's header was validated when it entered the registry.
        let archived = unsafe { rkyv::access_unchecked::<ArchivedSStableEntry>(verify_sstable_payload(body)?) };
        let key = archived.key.as_slice();
        if end.is_some_and(|e| key >= e) {
            break;
        }
        if key < start {
            continue;
        }
        entries.push((
            key.to_vec(),
            (!archived.tombstone).then(|| archived.value.to_native()),
            archived.seq.to_native(),
        ));
    }
    Ok(entries)
}

impl L0Summary {
    /// Summarise a file's records, or `None` for an empty file. `records` must be
    /// key-sorted, as every L0 file is.
    pub(super) fn of_sorted<'a>(records: impl ExactSizeIterator<Item = (&'a [u8], u64)> + Clone) -> Option<Self> {
        let len = records.len();
        let mut iter = records.clone();
        let (min_key, first_seq) = iter.next()?;
        let (max_key, max_seq) = iter.fold((min_key, first_seq), |(_, max_seq), (key, seq)| {
            (key, if seq_newer(seq, max_seq) { seq } else { max_seq })
        });
        Some(Self {
            min_key: min_key.to_vec(),
            max_key: max_key.to_vec(),
            max_seq,
            entries: len as u64,
            bloom: BloomFilter::build(records.map(|(key, _)| key), len),
            index: None,
        })
    }
}

impl L0FileEntry {
    pub(super) fn new(path: PathBuf, created_at_ms: u128, summary: Option<L0Summary>) -> Self {
        Self {
            path,
            created_at_ms,
            summary,
            reader_count: AtomicU64::new(0),
            obsolete: AtomicBool::new(false),
            #[cfg(test)]
            frames_read: AtomicU64::new(0),
        }
    }

    /// Whether `key` could be in this file. `false` is exact (outside the key
    /// range, or a Bloom negative); `true` means read the file to find out.
    pub(super) fn may_contain(&self, key: &[u8]) -> bool {
        self.summary
            .as_ref()
            .is_none_or(|s| key >= s.min_key.as_slice() && key <= s.max_key.as_slice() && s.bloom.contains(key))
    }

    /// Whether every entry in this file is at most `seq` — so, visiting layers
    /// newest first with a candidate at `seq` already in hand, nothing here can
    /// win (an exact tie stays with the newer layer already visited).
    pub(super) fn cannot_beat(&self, seq: u64) -> bool {
        self.summary.as_ref().is_some_and(|s| seq_newer_or_eq(seq, s.max_seq))
    }

    /// Where to start scanning this file for `key`: a sparse-index hint, or the
    /// head of the file when there is no index.
    pub(super) fn scan_start(&self, key: &[u8]) -> u64 {
        self.summary
            .as_ref()
            .and_then(|s| s.index.as_ref())
            .map_or(0, |index| index.block_start(key))
    }

    pub(super) fn is_obsolete(&self) -> bool {
        self.obsolete.load(Ordering::SeqCst)
    }

    pub(super) fn mark_obsolete(&self) {
        self.obsolete.store(true, Ordering::SeqCst);
    }

    pub(super) fn increment_readers(&self) {
        self.reader_count.fetch_add(1, Ordering::SeqCst);
    }

    pub(super) fn decrement_readers(&self) {
        self.reader_count.fetch_sub(1, Ordering::SeqCst);
    }

    pub(super) fn readers(&self) -> u64 {
        self.reader_count.load(Ordering::SeqCst)
    }
}

impl L0ReadGuard {
    pub(super) fn new(entry: Arc<L0FileEntry>) -> Self {
        entry.increment_readers();
        Self { entry }
    }
}

impl Drop for L0ReadGuard {
    fn drop(&mut self) {
        self.entry.decrement_readers();
    }
}

impl LSMTree {
    /// The stamp for the next L0 flush: `candidate` (wall-clock based, so file names
    /// stay readable), but never at or below any stamp already issued or on disk.
    ///
    /// L0 file order breaks **exact seq ties**, and GC's relocation makes them: it
    /// re-inserts a key at its existing seq with a new pointer, so the old and the
    /// relocated copy can sit in two L0 files with the same seq, and only file
    /// order says the relocation is newer. Ordered by raw wall time, a clock that
    /// steps backwards (NTP, a VM or WSL resume) between the two flushes would put
    /// the relocation *first*: every read would return the pre-relocation pointer,
    /// the L0→L1 merge would keep it in L1, and once GC unlinked its segment the key
    /// would read `SegmentMissing` for good (and the next GC pass would judge the
    /// real record dead). Stamps are taken after the per-memtable flush lock, and
    /// memtables flush in version order, so a later flush always gets a larger stamp.
    pub(super) fn next_l0_stamp(&self, candidate: u128) -> u128 {
        let mut last = self.last_l0_stamp.lock();
        let stamp = candidate.max(last.saturating_add(1));
        *last = stamp;
        stamp
    }

    /// Add a Level-0 file to the in-memory registry.
    ///
    /// One of the two points that establish "a registered L0 file has a valid
    /// header" — the invariant [`open_registered_l0`] relies on to skip
    /// re-validation on every read. Here it holds by construction: the only caller
    /// is `flush_ro_memtable_to_level0`, whose `write_level0_file_for_records` just wrote the header and
    /// fsynced. The other point is `load_existing_level0_files` at open, which
    /// validates what it finds on disk. **Do not register a file from anywhere
    /// else without validating its header first.**
    pub(super) fn register_level0_file(&self, bucket: u32, path: PathBuf, created_at_ms: u128, summary: Option<L0Summary>) {
        let mut guard = self.level0_files[bucket as usize].write();
        if guard.iter().any(|entry| entry.path == path) {
            return;
        }
        guard.push(Arc::new(L0FileEntry::new(path, created_at_ms, summary)));
    }

    /// Capture frozen SSTable read handles for a scan, **newest layer first**:
    /// per-bucket Level-0 read guards (skipping already-obsolete files, ordered
    /// oldest→newest within a bucket) captured *before* the per-bucket Level-1
    /// `Arc<File>`s.
    ///
    /// Capturing in the same direction that compaction relocates keys (L0→L1) is
    /// what makes a scan complete under a concurrent merge: a key the merge moves
    /// into L1 after we captured L0 is still found when we capture L1 immediately
    /// after, so it is never absent from both. The returned guards pin their files
    /// (`cleanup_obsolete_level0_files` only deletes when `readers() == 0`), so a
    /// merge that obsoletes them mid-scan cannot delete them before the caller has
    /// read them. The caller still merges oldest→newest (L1 then L0) so precedence
    /// is unchanged.
    pub(super) fn capture_sstable_layers(&self) -> (Vec<Vec<L0ReadGuard>>, Vec<Arc<File>>) {
        let num_buckets = self.config.num_buckets;
        let l0_guards: Vec<Vec<L0ReadGuard>> = (0..num_buckets)
            .map(|bucket| {
                let entries = self.level0_files[bucket].read();
                let mut guards: Vec<L0ReadGuard> = entries
                    .iter()
                    .filter(|e| !e.is_obsolete())
                    .map(|e| L0ReadGuard::new(Arc::clone(e)))
                    .collect();
                guards.sort_by_key(|a| a.entry.created_at_ms);
                guards
            })
            .collect();
        let l1_files: Vec<Arc<File>> = (0..num_buckets).map(|bucket| self.sstable_files[bucket].read().clone()).collect();
        (l0_guards, l1_files)
    }

    pub(super) fn level0_entries_snapshot(&self, bucket: u32) -> Vec<Arc<L0FileEntry>> {
        self.level0_files[bucket as usize].read().clone()
    }

    pub(super) fn cleanup_obsolete_level0_files(&self) {
        for bucket in 0u32..self.config.num_buckets as u32 {
            let mut guard = self.level0_files[bucket as usize].write();
            guard.retain(|entry| {
                if entry.is_obsolete() && entry.readers() == 0 {
                    let _ = std::fs::remove_file(&entry.path);
                    false
                } else {
                    true
                }
            });
        }
    }
}
