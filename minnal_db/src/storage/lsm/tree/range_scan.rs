//! The bounded range scan: a page of live `(key, pointer, seq)` entries from a
//! start key, resolved across every layer of the tree.

use super::*;

impl LSMTree {
    /// Returns the first `limit` live `(key, pointer, seq)` entries in `[start, end)`, in sorted order.
    ///
    /// Avoids a redundant per-key `lsm.get()` call in callers that immediately need to
    /// decode the pointer anyway (e.g. `resolve_entries_from_pointers`).
    ///
    /// Uses the same oldest-first, last-writer-wins `BTreeMap<key, Option<pointer>>`
    /// pattern as `key_pointer_pairs` so active-memtable tombstones always shadow stale
    /// SSTable entries — the same guarantee `lsm.keys()` provides.
    ///
    /// Same pointer-resolution invariant as [`scan_prefix`](Self::scan_prefix):
    /// returned pointers are LSM-complete but must be resolved against the value log
    /// only inside the value-log generation bracket / GC bucket lock.
    pub(crate) fn range_pointers_bounded(&self, start: &[u8], end: Option<&[u8]>, limit: usize) -> Result<Vec<(Vec<u8>, u128, u64)>> {
        // Layers resolve by highest seq (`fold_newest`, oldest→newest: L1 → L0 → RO →
        // active); only the *capture* order is newest-first so a concurrent
        // flush/compaction cannot drop a live key. See `scan_prefix`. `take(limit)`
        // runs on the fully merged, sorted map.
        //
        // `end` (exclusive) bounds the scan to `[start, end)`; entries `>= end` are
        // never inserted so the merge map and the resulting page stay within the
        // requested key window.
        let mut all_entries = ScanMap::new();

        // Memtable layers (newest) captured first, under one `memtable.read()` guard
        // so a flush cannot split a key between active and RO.
        let (active_snapshot, ro_snapshot) = {
            let memtable = self.memtable.read();
            let active: Vec<ScanEntry> = memtable
                .skip_list
                .iter_raw_from(start)
                .take_while(|(key, _value, _seq, _tombstone)| end.is_none_or(|e| *key < e))
                .map(|(key, value, seq, tombstone)| (key.to_vec(), (!tombstone).then_some(value), seq))
                .collect();
            let ro = self.read_only_memtables.read().clone();
            (active, ro)
        };

        // SSTable handles captured newest-first (L0 before L1); read in merge order.
        let (l0_guards, l1_files) = self.capture_sstable_layers();

        // Pages ≤ this bound use the limit-bounded serial L1 path; larger scans keep the
        // parallel read-everything path (see the L1 step below).
        const BOUNDED_SCAN_LIMIT: usize = 10_000;

        // Collect the layers NEWER than L1 — L0 (oldest-first per bucket), then read-only
        // memtables (oldest-first), then the active memtable (newest) — into one list in
        // that oldest→newest order. Two uses: (a) bound the L1 scan (a page needs only each
        // bucket's smallest `limit` live keys, but a newer-layer tombstone can delete one,
        // so we over-read by the count of newer in-range entries — a safe upper bound on
        // possible deletes, since keys are bucket-partitioned so only a same-bucket newer
        // entry can shadow a bucket's L1 key); (b) replay over L1 for identical
        // tombstone-suppression semantics.
        let mut newer: Vec<ScanEntry> = Vec::new();
        let bounded = limit <= BOUNDED_SCAN_LIMIT;

        // Reads one bucket's L0 files (oldest-first, so an exact seq tie goes to the
        // newer file), range-filtered. L0 has no min/max or sparse index, so files are
        // scanned in full.
        let read_l0_bucket = |bucket: usize| -> Result<Vec<ScanEntry>> {
            let mut bucket_entries = Vec::new();
            for guard in &l0_guards[bucket] {
                let mut file = open_registered_l0_for_scan(&guard.entry.path)?;
                // Bound for `check_frame_len`: a frame can never be longer than its file.
                let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
                let mut entry_bytes: Vec<u8> = Vec::new();
                loop {
                    let mut size_buf = [0u8; 4];
                    match file.read_exact(&mut size_buf) {
                        Ok(_) => {}
                        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                        Err(e) => return Err(LSMError::Io(e)),
                    }
                    let size = u32::from_le_bytes(size_buf) as usize;
                    check_frame_len(size as u64, file_len)?;
                    entry_bytes.resize(size, 0);
                    file.read_exact(&mut entry_bytes[..size])?;
                    let archived = unsafe { rkyv::access_unchecked::<ArchivedSStableEntry>(verify_sstable_payload(&entry_bytes[..size])?) };
                    if archived.key.as_slice() >= start && end.is_none_or(|e| archived.key.as_slice() < e) {
                        let val = (!archived.tombstone).then(|| archived.value.to_native());
                        bucket_entries.push((archived.key.as_slice().to_vec(), val, archived.seq.to_native()));
                    }
                }
            }
            Ok(bucket_entries)
        };

        // L0 (newer than L1). Bounded page → read busy buckets SERIALLY: L0 is small in the
        // steady state, so spawning a thread per bucket costs more than it saves (same
        // reasoning as the bounded L1 path below). Full/large scan → parallel read-everything.
        let busy_l0: Vec<usize> = (0..self.config.num_buckets).filter(|b| !l0_guards[*b].is_empty()).collect();
        if bounded {
            for bucket in busy_l0 {
                newer.extend(read_l0_bucket(bucket)?);
            }
        } else {
            let mut first_err: Option<LSMError> = None;
            std::thread::scope(|s| {
                let handles: Vec<_> = busy_l0.into_iter().map(|bucket| s.spawn(move || read_l0_bucket(bucket))).collect();
                for handle in handles {
                    match handle
                        .join()
                        .unwrap_or_else(|_| Err(LSMError::Io(std::io::Error::other("thread panicked"))))
                    {
                        Ok(entries) => newer.extend(entries),
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
        for ro_memtable in ro_snapshot.iter() {
            for record in ro_memtable.records().iter() {
                if record.key.as_slice() >= start && end.is_none_or(|e| record.key.as_slice() < e) {
                    newer.push((record.key.clone(), (!record.tombstone).then_some(record.value), record.seq));
                }
            }
        }
        newer.extend(active_snapshot);

        // L1 SSTables (oldest layer) — only buckets whose min/max overlaps `[start, end)`;
        // each seeks near `start` via the sparse index and stops at `end`. The L0 guards
        // captured above make the metadata read race-safe (see `l1_overlaps`). Keys are
        // hash-sharded so bucket key-sets are disjoint.
        //
        // Bounded page: read at most `limit + 1 + |newer in range in THIS bucket|` LIVE keys
        // per bucket and scan buckets SERIALLY — per-bucket work is tiny, so a thread per
        // bucket costs more than it saves. Full/large scan: keep the parallel, read-everything
        // path. Only a same-bucket newer entry can shadow a bucket's L1 key (keys are
        // hash-sharded), so the slack is sharded per bucket — a global `|newer|` slack would
        // over-read every bucket by the total newer count. See `get_bucket_for_key`.
        match bounded {
            true => {
                let mut newer_per_bucket = vec![0usize; self.config.num_buckets];
                for (key, _, _) in newer.iter() {
                    newer_per_bucket[get_bucket_for_key(key, self.config.num_buckets) as usize] += 1;
                }
                for (bucket, file) in l1_files.iter().enumerate() {
                    if !self.l1_overlaps(bucket, start, end) {
                        continue;
                    }
                    let cap = limit.saturating_add(1).saturating_add(newer_per_bucket[bucket]);
                    let mut entries = Vec::new();
                    self.scan_l1_range_into(bucket, file, start, end, Some(cap), &mut entries)?;
                    for (k, v, seq) in entries {
                        fold_newest(&mut all_entries, k, v, seq);
                    }
                }
            }
            false => {
                let mut first_err: Option<LSMError> = None;
                std::thread::scope(|s| {
                    let handles: Vec<_> = l1_files
                        .iter()
                        .enumerate()
                        .filter(|(bucket, _)| self.l1_overlaps(*bucket, start, end))
                        .map(|(bucket, file)| {
                            s.spawn(move || -> Result<Vec<ScanEntry>> {
                                let mut entries = Vec::new();
                                self.scan_l1_range_into(bucket, file, start, end, None, &mut entries)?;
                                Ok(entries)
                            })
                        })
                        .collect();
                    for handle in handles {
                        match handle
                            .join()
                            .unwrap_or_else(|_| Err(LSMError::Io(std::io::Error::other("thread panicked"))))
                        {
                            Ok(entries) => {
                                for (k, v, seq) in entries {
                                    fold_newest(&mut all_entries, k, v, seq);
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
        }

        // Fold the newer layers over L1 (oldest→newest), resolving by seq.
        for (key, value, seq) in newer {
            fold_newest(&mut all_entries, key, value, seq);
        }

        Ok(all_entries
            .into_iter()
            .filter_map(|(key, (value, seq))| value.map(|v| (key, v, seq)))
            .take(limit)
            .collect())
    }
}
