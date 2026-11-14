//! Batch reads (`get_multiple`): many keys resolved at once, each file read once
//! with a forward cursor rather than one search per key.

use super::*;

impl LSMTree {
    /// Fetch multiple keys in a **single pass per bucket**.
    ///
    /// Where `get` × N keys would scan every Level-0 file N times, this groups the
    /// keys by hash bucket and makes one forward pass over each of the bucket's
    /// L0 files and its L1 file, resolving all of that bucket's keys together.
    /// The pass reads only near the keys: it jumps between them with the
    /// sparse indexes (`scan_sorted_candidates`), so a query page of a few keys
    /// reads a few KiB per file and a large batch one sequential read, never more
    /// than the file.
    ///
    /// Resolution is identical to [`get_with_seq`](Self::get_with_seq): the
    /// highest-`seq` copy across **every** layer wins (a winning tombstone ⇒
    /// `None`), with the same active-memtable fast path. It must never become
    /// first-hit-by-layer: value-log GC decides liveness with this method, so a stale
    /// lower-seq tombstone in a newer layer answering "deleted" for a live key would
    /// make GC drop that key's only record (see
    /// `test_get_multiple_resolves_by_seq_not_layer_order`).
    ///
    /// Returns one `Option<(pointer, seq)>` per input key, in input order; a key
    /// given more than once resolves at every position. `None` means the key does
    /// not exist or was deleted.
    pub(crate) fn get_multiple(&self, keys: &[Vec<u8>]) -> Result<BatchLookup> {
        Ok(self.get_multiple_counted(keys)?.0)
    }

    /// [`get_multiple`](Self::get_multiple), also returning the SSTable bytes
    /// read (for tests).
    pub(super) fn get_multiple_counted(&self, keys: &[Vec<u8>]) -> Result<(BatchLookup, u64)> {
        let n = keys.len();
        let mut sstable_bytes = 0u64;
        // Running winner per input position, folded newest layer first through
        // `merge_candidate` so an exact sequence tie stays with the newer layer.
        let mut best: Vec<Candidate> = vec![None; n];
        // Positions the active-memtable fast path settled: nothing below can beat them.
        let mut settled: Vec<bool> = vec![false; n];

        // ── 1. Active memtable ────────────────────────────────────────────────
        {
            let memtable = self.memtable.read();
            // Loaded under the memtable lock, as in `get_with_seq`: a flush raises
            // the bound while holding the write lock, so a bound read before taking
            // the lock could be too low for the memtable we then read.
            let max_lower_seq = self.max_lower_seq.load(Ordering::Relaxed);
            for i in 0..n {
                if let Some((value, seq, tombstone)) = memtable.skip_list.entry(&keys[i]) {
                    Self::merge_candidate(&mut best[i], (!tombstone).then_some(value), seq);
                    settled[i] = seq_newer_or_eq(seq, max_lower_seq);
                }
            }
        }

        // ── 2. Read-only memtables, newest first ─────────────────────────────
        {
            let ro_memtables = self.read_only_memtables.read();
            for ro in ro_memtables.iter().rev() {
                for i in 0..n {
                    if settled[i] {
                        continue;
                    }
                    if let Some(record) = ro.find(&keys[i]) {
                        Self::merge_candidate(&mut best[i], (!record.tombstone).then_some(record.value), record.seq);
                    }
                }
            }
        }

        // ── 3. Group unsettled keys by SSTable bucket, one slot per distinct key ─
        let mut pending_by_bucket: Vec<BatchSlots> = (0..self.config.num_buckets).map(|_| std::collections::HashMap::new()).collect();
        for i in 0..n {
            if !settled[i] {
                let bucket = get_bucket_for_key(&keys[i], self.config.num_buckets) as usize;
                // Every position of a duplicated key carries the same memtable winner.
                pending_by_bucket[bucket].entry(keys[i].as_slice()).or_insert(best[i]);
            }
        }

        // ── 4. One L0+L1 pass per bucket in parallel. Buckets are hash-disjoint,
        //    so each thread owns its bucket's slots exclusively.
        {
            let mut first_err: Option<LSMError> = None;
            std::thread::scope(|s| {
                let handles: Vec<_> = pending_by_bucket
                    .iter_mut()
                    .enumerate()
                    .filter(|(_, pending)| !pending.is_empty())
                    .map(|(bucket, pending)| s.spawn(move || self.resolve_bucket_batch(bucket, pending)))
                    .collect();
                for handle in handles {
                    let outcome = handle
                        .join()
                        .unwrap_or_else(|_| Err(LSMError::Io(std::io::Error::other("thread panicked"))));
                    match outcome {
                        Ok(read) => sstable_bytes += read,
                        Err(e) if first_err.is_none() => first_err = Some(e),
                        Err(_) => {}
                    }
                }
            });
            if let Some(e) = first_err {
                return Err(e);
            }
        }

        let results = (0..n)
            .map(|i| {
                let winner = if settled[i] {
                    best[i]
                } else {
                    let bucket = get_bucket_for_key(&keys[i], self.config.num_buckets) as usize;
                    pending_by_bucket[bucket].get(keys[i].as_slice()).copied().flatten()
                };
                winner.and_then(|(value, seq)| value.map(|ptr| (ptr, seq)))
            })
            .collect();
        Ok((results, sstable_bytes))
    }

    /// The SSTable half of [`get_multiple`](Self::get_multiple) for one bucket: fold
    /// every L0 file's and then the L1 file's copy of each pending key into its slot,
    /// highest `seq` winning. Each slot arrives holding the memtable winner (if any).
    pub(super) fn resolve_bucket_batch(&self, bucket: usize, pending: &mut BatchSlots<'_>) -> Result<u64> {
        let mut bytes_read = 0u64;
        let fold = |pending: &mut BatchSlots<'_>, archived: &ArchivedSStableEntry| {
            if let Some(slot) = pending.get_mut(archived.key.as_slice()) {
                let value = (!archived.tombstone).then(|| archived.value.to_native());
                Self::merge_candidate(slot, value, archived.seq.to_native());
            }
        };

        // Level 0: EVERY captured file, newest first. A hit in one file does not
        // settle the key — an older file (or L1) can hold a higher seq, because
        // flush order is not seq order. The guards are held through the L1 read
        // below, so a key a concurrent merge moves into L1 is still in a captured,
        // already-scanned L0 file.
        let mut l0_guards: Vec<L0ReadGuard> = self
            .level0_entries_snapshot(bucket as u32)
            .iter()
            .filter(|e| !e.is_obsolete())
            .map(|e| L0ReadGuard::new(Arc::clone(e)))
            .collect();
        l0_guards.sort_by_key(|b| std::cmp::Reverse(b.entry.created_at_ms));
        for guard in &l0_guards {
            // Read the file only if some pending key may be in it AND could still
            // be beaten by it — the same exact rejects as `search_level0_files`.
            let entry = &guard.entry;
            let mut wanted: Vec<&[u8]> = pending
                .iter()
                .filter(|(key, slot)| entry.may_contain(key) && !slot.is_some_and(|(_, seq)| entry.cannot_beat(seq)))
                .map(|(key, _)| *key)
                .collect();
            if wanted.is_empty() {
                continue;
            }
            wanted.sort_unstable();
            let wanted: Vec<(&[u8], u64)> = wanted.into_iter().map(|k| (k, entry.scan_start(k))).collect();
            let file = open_registered_l0_for_scan(&entry.path)?;
            let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
            let (read, rejected) = self.scan_sorted_candidates(&file, file_len, &wanted, |a| fold(pending, a))?;
            // An L0 file is never rewritten, so its hints cannot go stale.
            debug_assert!(
                !rejected,
                "an L0 sparse-index hint was rejected: the index offsets disagree with the file"
            );
            bytes_read += read;
        }

        // L1 pre-filter: skip the file when no pending key can be in it.
        // Same exact fast-rejects as `lookup_level1` (min/max is exact, a bloom
        // negative is exact), and only applied when the metadata/bloom are present —
        // absent metadata reads the file rather than risk skipping a live key.
        //
        // Race-safety mirrors the scan paths: the L0 guards above are still held, and
        // `merge_level0_to_level1` installs the new L1 file + metadata/bloom BEFORE
        // marking the old L0 files obsolete, so a key stale L1 metadata would reject
        // is still in a captured L0 file that we have already folded in.
        //
        // A key whose slot already holds a copy at least as new as the file's
        // newest entry is dropped too (it cannot be beaten there), by the same
        // argument: stale metadata implies the merged L0 files are still captured
        // and their seqs already folded into the slots.
        let may_be_in_l1: Vec<&[u8]> = {
            let (min_max, max_seq) = {
                let meta = self.sstable_metadata[bucket].read();
                let meta = meta.first().filter(|m| m.entry_count > 0);
                (meta.map(|m| (m.min_key.clone(), m.max_key.clone())), meta.and_then(|m| m.max_seq))
            };
            let bloom = self.sstable_blooms[bucket].read();
            pending
                .iter()
                .filter(|(_, slot)| !max_seq.is_some_and(|max| slot.is_some_and(|(_, seq)| seq_newer_or_eq(seq, max))))
                .map(|(k, _)| *k)
                .filter(|k| min_max.as_ref().is_none_or(|(min, max)| *k >= min.as_slice() && *k <= max.as_slice()))
                .filter(|k| bloom.as_ref().is_none_or(|b| b.contains(k)))
                .collect()
        };
        let mut may_be_in_l1 = may_be_in_l1;
        if may_be_in_l1.is_empty() {
            return Ok(bytes_read);
        }
        may_be_in_l1.sort_unstable();

        // Level 1: hop between the candidates with the sparse index, reading only
        // near them. The index may belong to a newer file than the handle (a
        // compaction swapping it), so each hint is validated before use.
        let level1_file = self.sstable_files[bucket].read().clone();
        let file_size = level1_file.metadata().map(|m| m.len()).unwrap_or(0);
        if file_size == 0 {
            return Ok(bytes_read);
        }
        let wanted: Vec<(&[u8], u64)> = {
            let index = self.sstable_indexes[bucket].read();
            may_be_in_l1
                .iter()
                .map(|&k| (k, index.as_ref().map_or(0, |i| i.block_start(k))))
                .collect()
        };
        let (read, rejected) = self.scan_sorted_candidates(&level1_file, file_size, &wanted, |a| fold(pending, a))?;
        if rejected && let Some(m) = self.metrics() {
            Metrics::bump(&m.sparse_hint_rejects);
        }
        Ok(bytes_read + read)
    }

    /// Visit the entries of one sorted SSTable file near each of `candidates`
    /// (`(key, sparse-index hint)`, sorted by key), passing every entry read to
    /// `visit`.
    ///
    /// One forward cursor: for each candidate, jump to its hint when that lies
    /// past the cursor, then read forward until the
    /// cursor passes the candidate. A batch of a few keys therefore reads about
    /// `SAMPLE_INTERVAL` entries per key, while a dense batch turns into one
    /// sequential read, never more than the file. Entries are key-sorted, so any
    /// valid frame boundary at or below a candidate finds it: a hint is used
    /// only after [`valid_scan_start`](Self::valid_scan_start) accepts it, and a
    /// rejected one just keeps scanning from the cursor, which is always below
    /// the current candidate.
    ///
    /// Returns the bytes read and whether any hint was rejected.
    pub(super) fn scan_sorted_candidates(
        &self,
        file: &File,
        file_len: u64,
        candidates: &[(&[u8], u64)],
        mut visit: impl FnMut(&ArchivedSStableEntry),
    ) -> Result<(u64, bool)> {
        debug_assert!(candidates.windows(2).all(|w| w[0].0 <= w[1].0), "candidates must be sorted");
        let mut frames = FrameReader::new(file, SSTABLE_DATA_START);
        let mut rejected = false;
        let mut next = 0usize;
        'candidates: while next < candidates.len() {
            let (wanted, start) = candidates[next];
            if start > frames.offset() {
                if Self::valid_scan_start(file, start, file_len, wanted) {
                    frames.seek(start);
                } else {
                    rejected = true;
                }
            }
            while let Some(body) = frames.next_frame(file_len)? {
                let archived = unsafe { rkyv::access_unchecked::<ArchivedSStableEntry>(verify_sstable_payload(body)?) };
                visit(archived);
                let key = archived.key.as_slice();
                if key >= wanted {
                    while next < candidates.len() && candidates[next].0 <= key {
                        next += 1;
                    }
                    continue 'candidates;
                }
            }
            break;
        }
        Ok((frames.bytes_read, rejected))
    }
}
