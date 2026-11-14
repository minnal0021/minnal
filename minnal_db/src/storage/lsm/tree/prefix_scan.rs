//! Prefix scans (`scan_prefix`): every live key under one prefix, each layer read
//! from the prefix's first key only.

use super::*;

impl LSMTree {
    /// Every live `(key, pointer, seq)` whose key starts with `prefix`, across
    /// every layer, in key order.
    ///
    /// A returned pointer is what the LSM held when its layer was read. Value-log GC
    /// may relocate the value and delete its segment before the caller reads it; that
    /// read then fails with `SegmentMissing` and the caller re-resolves the key through
    /// the LSM (`KVStore::refetch_dropped`). Segment ids are never reused, so a stale
    /// pointer can fail but never return another key's value.
    pub(crate) fn scan_prefix(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, u128, u64)>> {
        // Layers are folded oldest→newest by `fold_newest`: each key resolves to its
        // highest-seq copy, exactly as `get_with_seq` resolves it, so a tombstone
        // shadows only copies older than itself.
        //
        // What changes for concurrency is the *capture* order, which is the
        // reverse — newest layer first. Flush/compaction relocate a key
        // active→RO→L0→L1; capturing in that same direction guarantees the key is
        // seen in the newer layer before it can move into an older layer we have
        // not captured yet, so it is never absent from every captured layer at
        // once. We capture cheap, frozen handles (active+RO snapshot, L0 read
        // guards, L1 `Arc<File>`) up front and do the I/O afterwards, then merge.
        let mut all_entries = ScanMap::new();

        // Capture the memtable layers (newest) first: snapshot the active
        // memtable's matching records and clone the read-only list under a single
        // `memtable.read()` guard. Holding that guard excludes `flush_memtable`
        // (which needs the write lock), so a key cannot slip from active into RO
        // between the two captures. The read-only records are `Arc`s, frozen even
        // if the list is later mutated.
        let (active_snapshot, ro_snapshot) = {
            let memtable = self.memtable.read();
            let mut active = Vec::new();
            for (key, value, seq, tombstone) in memtable.skip_list.iter_raw_from(prefix) {
                if !key.starts_with(prefix) {
                    break;
                }
                active.push((key.to_vec(), (!tombstone).then_some(value), seq));
            }
            let ro = self.read_only_memtables.read().clone();
            (active, ro)
        };

        // SSTables (oldest layers), buckets scanned in parallel. Each bucket
        // captures L0 before L1 internally and returns entries ordered L1 (oldest)
        // → L0 (oldest→newest) with tombstones carried as `None`.
        //
        // Only buckets that could hold a matching key spawn a worker: one whose L1
        // does not overlap `[prefix, upper)` *and* has no non-obsolete L0 file has
        // nothing to contribute. A key that races into such a bucket after this
        // check (memtable→L0 flush, or L0→L1 merge) was already in the memtable
        // snapshot captured above, so skipping the spawn cannot drop it.
        let upper_owned = prefix_upper_bound(prefix);
        let upper = upper_owned.as_deref();
        {
            let mut first_err: Option<LSMError> = None;
            std::thread::scope(|s| {
                let handles: Vec<_> = (0u32..self.config.num_buckets as u32)
                    .filter(|&bucket| self.bucket_has_nonobsolete_l0(bucket as usize) || self.l1_overlaps(bucket as usize, prefix, upper))
                    .map(|bucket| s.spawn(move || self.scan_prefix_in_bucket(bucket, prefix, upper)))
                    .collect();
                for handle in handles {
                    match handle
                        .join()
                        .unwrap_or_else(|_| Err(LSMError::Io(std::io::Error::other("thread panicked"))))
                    {
                        Ok(entries) => {
                            for (key, value, seq) in entries {
                                fold_newest(&mut all_entries, key, value, seq);
                            }
                        }
                        Err(e) => {
                            if first_err.is_none() {
                                first_err = Some(e);
                            }
                        }
                    }
                }
            });
            if let Some(e) = first_err {
                return Err(e);
            }
        }

        // Read-only memtables oldest-first.
        for ro_memtable in ro_snapshot.iter() {
            let records = ro_memtable.records();
            for record in records.iter() {
                if record.key.starts_with(prefix) {
                    fold_newest(
                        &mut all_entries,
                        record.key.clone(),
                        (!record.tombstone).then_some(record.value),
                        record.seq,
                    );
                }
            }
        }

        // Active memtable (newest layer).
        for (key, value, seq) in active_snapshot {
            fold_newest(&mut all_entries, key, value, seq);
        }

        Ok(all_entries
            .into_iter()
            .filter_map(|(key, (val, seq))| val.map(|v| (key, v, seq)))
            .collect())
    }

    /// Whether this bucket's L1 file holds any key in `[start, end)`, decided from
    /// the in-memory min/max metadata alone (no file I/O). A `false` provably means
    /// the L1 file has no matching key, so a scan can skip reading it entirely; an
    /// empty/absent L1 (no metadata, or `entry_count == 0`) is also `false`.
    ///
    /// Safe under concurrent compaction *provided the caller has already captured
    /// this bucket's L0 read guards*: a merge marks the old L0 files obsolete only
    /// after installing the new L1 file and updating this metadata (see
    /// `merge_level0_to_level1`). So whatever version of the metadata a scan reads,
    /// any key it would cause the scan to skip is either already reflected here or
    /// still present in a captured (not-yet-obsolete) L0 file — never invisible.
    pub(super) fn l1_overlaps(&self, bucket: usize, start: &[u8], end: Option<&[u8]>) -> bool {
        let metadata = self.sstable_metadata[bucket].read();
        match metadata.first() {
            Some(meta) if meta.entry_count > 0 => meta.max_key.as_slice() >= start && end.is_none_or(|e| meta.min_key.as_slice() < e),
            _ => false,
        }
    }

    /// Whether this bucket has any non-obsolete Level-0 file. Used only to decide
    /// whether a scan needs to spawn a worker for the bucket at all.
    pub(super) fn bucket_has_nonobsolete_l0(&self, bucket: usize) -> bool {
        self.level0_files[bucket].read().iter().any(|e| !e.is_obsolete())
    }

    /// Read this bucket's L1 file for keys in `[start, end)`, appending each as a
    /// [`ScanEntry`] (tombstone carried as `None`). Entries are key-sorted, so the
    /// scan seeks near `start` via the sparse index and stops as soon as it reaches
    /// `end`.
    ///
    /// The sparse-index offset is validated against the file with `valid_scan_start`
    /// (a concurrent compaction can swap the file under the in-memory index): a
    /// stale hint just falls back to a full scan from 0, so it is a performance
    /// hint, never a correctness input.
    pub(super) fn scan_l1_range_into(
        &self,
        bucket: usize,
        file: &File,
        start: &[u8],
        end: Option<&[u8]>,
        max_live: Option<usize>,
        out: &mut Vec<ScanEntry>,
    ) -> Result<()> {
        self.for_each_l1_entry_in_range(bucket, file, start, end, max_live, |archived| {
            let val = (!archived.tombstone).then(|| archived.value.to_native());
            out.push((archived.key.as_slice().to_vec(), val, archived.seq.to_native()));
        })
    }

    /// Visit every L1 entry whose key is in `[start, end)`, in ascending key order,
    /// seeking near `start` via the sparse index and stopping at `end`.
    ///
    /// The shared L1 range primitive behind `scan_l1_range_into`, `scan_prefixes`
    /// (once per requested prefix), and `range_pointers_bounded`. `visit` receives each
    /// archived entry (borrowed only for the call) so callers extract whatever value
    /// shape they need — pointer+seq, pointer only, or liveness.
    ///
    /// The sparse-index offset is validated against the file (`valid_scan_start`): a
    /// stale hint after a concurrent compaction falls back to a full scan, so it is a
    /// performance hint, never a correctness input.
    ///
    /// `max_live`, when set, stops the scan after that many **non-tombstone** entries
    /// have been visited (tombstones are still visited but do not count). This is the
    /// per-bucket early-stop that bounds a paginated scan: since keys are hash-sharded
    /// (each key lives in exactly one bucket) and this file is key-sorted, a bucket's
    /// smallest `k` live keys are enough for the caller to assemble the global smallest
    /// `k`. Callers that need the whole range (unbounded scans) pass `None`.
    pub(super) fn for_each_l1_entry_in_range(
        &self,
        bucket: usize,
        file: &File,
        start: &[u8],
        end: Option<&[u8]>,
        max_live: Option<usize>,
        mut visit: impl FnMut(&ArchivedSStableEntry),
    ) -> Result<()> {
        let start_offset = {
            let index = self.sstable_indexes[bucket].read();
            index.as_ref().map(|i| i.block_start(start)).unwrap_or(0)
        };
        let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
        let offset = if start_offset != 0 && Self::valid_scan_start(file, start_offset, file_len, start) {
            start_offset
        } else {
            SSTABLE_DATA_START
        };

        let mut live_seen = 0usize;
        let mut frames = FrameReader::new(file, offset);
        while let Some(body) = frames.next_frame(file_len)? {
            let archived = unsafe { rkyv::access_unchecked::<ArchivedSStableEntry>(verify_sstable_payload(body)?) };
            let key = archived.key.as_slice();
            // Sorted order: once we reach `end`, no later key can match.
            if end.is_some_and(|e| key >= e) {
                break;
            }
            // The seeked block may begin below `start`; skip those pre-window keys.
            if key >= start {
                visit(archived);
                if !archived.tombstone {
                    live_seen += 1;
                    if max_live.is_some_and(|m| live_seen >= m) {
                        break;
                    }
                }
            }
        }
        Ok(())
    }

    /// Scan prefix within a specific SSTable bucket using key_prefix for efficiency.
    ///
    /// Returns matching entries as `(key, Option<pointer>)` ordered **L1 (oldest)
    /// first, then L0 oldest-to-newest** — the same oldest→newest layer order the
    /// other scans use. A tombstone is carried as `None` (not dropped) so the
    /// caller's merge map can let a newer L0 tombstone shadow a live L1 value,
    /// and a newer L0 value shadow a stale L1 value. Dropping tombstones here would
    /// resurrect deleted keys: L0 holds entries, tombstones included, not yet merged
    /// into L1.
    ///
    /// `upper` is the prefix's upper bound (`prefix_upper_bound(prefix)`): the L1
    /// read skips a non-overlapping file via min/max and seeks via the sparse index
    /// over `[prefix, upper)`, and each L0 file the same way (`read_l0_range`):
    /// skipped when its key range misses, seeked through its sparse index
    /// otherwise.
    pub(super) fn scan_prefix_in_bucket(&self, bucket: u32, prefix: &[u8], upper: Option<&[u8]>) -> Result<Vec<ScanEntry>> {
        let mut results = Vec::new();

        // Capture the SSTable handles newest-first — L0 *before* L1 — so a
        // concurrent L0→L1 merge cannot drop a key in the gap between the two
        // reads. The read guards pin every captured L0 file, so a merge that
        // marks them obsolete (and a cleanup that would delete them) cannot run
        // until this scan releases them; any key the merge moves into L1 after we
        // captured L0 is still found in L1, which we capture afterwards. The merge
        // order itself is unchanged: L1 (oldest) then L0 oldest→newest, so the
        // caller's overwrite lets newer L0 entries/tombstones shadow older ones.
        let mut l0_guards: Vec<L0ReadGuard> = {
            let entries = self.level0_files[bucket as usize].read();
            entries
                .iter()
                .filter(|e| !e.is_obsolete())
                .map(|e| L0ReadGuard::new(Arc::clone(e)))
                .collect()
        };
        l0_guards.sort_by_key(|a| a.entry.created_at_ms);

        // Search the Level 1 file (oldest layer) first — but only if its min/max
        // says it overlaps `[prefix, upper)`, and then seeking near `prefix` via the
        // sparse index rather than reading from offset 0. The L0 guards are captured
        // above, before this metadata read, so the skip is race-safe (see
        // `l1_overlaps`). `[prefix, upper)` is exactly the keys starting with `prefix`.
        if self.l1_overlaps(bucket as usize, prefix, upper) {
            let level1_file = self.sstable_files[bucket as usize].read().clone();
            self.scan_l1_range_into(bucket as usize, &level1_file, prefix, upper, None, &mut results)?;
        }

        // Then the captured Level 0 files oldest-first. Each is sorted by key,
        // so the keys with the prefix are its range `[prefix, upper)`: a file
        // whose key range misses it is skipped, and the rest are read from the
        // block holding `prefix` to the first key past it.
        for guard in &l0_guards {
            results.extend(read_l0_range(&guard.entry, prefix, upper, usize::MAX)?);
        }

        Ok(results)
    }
}
