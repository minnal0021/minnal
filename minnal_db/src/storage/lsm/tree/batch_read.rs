//! Batch reads (`get_multiple`): many keys resolved at once, each file read once
//! with a forward cursor rather than one search per key, each key's copy taken
//! from the newest layer that holds it.

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
    /// Resolution is identical to [`get_with_seq`](Self::get_with_seq): each
    /// key's copy in the newest layer that holds it (a tombstone there ⇒ `None`).
    /// Value-log GC decides liveness with this method.
    ///
    /// Returns one `Option<(pointer, version)>` per input key, in input order; a
    /// key given more than once resolves at every position. `None` means the key
    /// does not exist or was deleted.
    pub(crate) fn get_multiple(&self, keys: &[Vec<u8>]) -> Result<BatchLookup> {
        Ok(self.get_multiple_counted(keys)?.0)
    }

    /// [`get_multiple`](Self::get_multiple), also returning the SSTable bytes
    /// read (for tests).
    pub(super) fn get_multiple_counted(&self, keys: &[Vec<u8>]) -> Result<(BatchLookup, u64)> {
        let n = keys.len();
        let mut sstable_bytes = 0u64;
        // Each position's copy, from the newest layer holding the key; `None`
        // until a layer does.
        let mut found: Vec<Candidate> = vec![None; n];

        // ── 1. Active memtable ────────────────────────────────────────────────
        {
            let memtable = self.memtable.read();
            for i in 0..n {
                if let Some((value, version, tombstone)) = memtable.skip_list.entry(&keys[i]) {
                    found[i] = Some(((!tombstone).then_some(value), version));
                }
            }
        }

        // ── 2. Read-only memtables, newest first ─────────────────────────────
        {
            let ro_memtables = self.read_only_memtables.read();
            for ro in ro_memtables.iter().rev() {
                for i in 0..n {
                    if found[i].is_some() {
                        continue;
                    }
                    if let Some(record) = ro.find(&keys[i]) {
                        found[i] = Some(((!record.tombstone).then_some(record.value), record.version));
                    }
                }
            }
        }

        // ── 3. Group the keys still unfound by SSTable bucket, one slot per
        //    distinct key ──────────────────────────────────────────────────────
        let mut pending_by_bucket: Vec<BatchSlots> = (0..self.config.num_buckets).map(|_| std::collections::HashMap::new()).collect();
        for i in 0..n {
            if found[i].is_none() {
                let bucket = get_bucket_for_key(&keys[i], self.config.num_buckets) as usize;
                pending_by_bucket[bucket].entry(keys[i].as_slice()).or_insert(None);
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
                let copy = found[i].or_else(|| {
                    let bucket = get_bucket_for_key(&keys[i], self.config.num_buckets) as usize;
                    pending_by_bucket[bucket].get(keys[i].as_slice()).copied().flatten()
                });
                copy.and_then(|(value, version)| value.map(|ptr| (ptr, version)))
            })
            .collect();
        Ok((results, sstable_bytes))
    }

    /// The SSTable half of [`get_multiple`](Self::get_multiple) for one bucket:
    /// fill each pending key's slot from the newest L0 file holding it, else from
    /// the L1 file.
    pub(super) fn resolve_bucket_batch(&self, bucket: usize, pending: &mut BatchSlots<'_>) -> Result<u64> {
        let mut bytes_read = 0u64;
        // A file is read newest first, so the first copy to fill a slot is the
        // key's current one; a later (older) file leaves it alone.
        let fold = |pending: &mut BatchSlots<'_>, archived: &ArchivedSStableEntry| {
            if let Some(slot) = pending.get_mut(archived.key.as_slice())
                && slot.is_none()
            {
                let value = (!archived.tombstone).then(|| archived.value.to_native());
                *slot = Some((value, SStableEntry::archived_version(archived)));
            }
        };

        // Level 0, newest first. The guards are held through the L1 read below,
        // so a key a concurrent merge moves into L1 is still in a captured L0 file.
        let mut l0_guards: Vec<L0ReadGuard> = self
            .level0_entries_snapshot(bucket as u32)
            .iter()
            .filter(|e| !e.is_obsolete())
            .map(|e| L0ReadGuard::new(Arc::clone(e)))
            .collect();
        l0_guards.sort_by_key(|b| std::cmp::Reverse(b.entry.created_at_ms));
        for guard in &l0_guards {
            // Read the file only for keys still unfound that it may hold — the
            // same exact rejects as `search_level0_files`.
            let entry = &guard.entry;
            let mut wanted: Vec<&[u8]> = pending
                .iter()
                .filter(|(key, slot)| slot.is_none() && entry.may_contain(key))
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

        // L1 pre-filter: the keys still unfound that may be in the file. Same exact
        // fast-rejects as `lookup_level1` (min/max is exact, a bloom negative is
        // exact), applied only when the metadata/bloom are present — absent
        // metadata reads the file rather than risk skipping a live key.
        //
        // Race-safety mirrors the scan paths: the L0 guards above are still held, and
        // `merge_level0_to_level1` installs the new L1 file + metadata/bloom BEFORE
        // marking the old L0 files obsolete, so a key stale L1 metadata would reject
        // is still in a captured L0 file that we have already read.
        let mut may_be_in_l1: Vec<&[u8]> = {
            let min_max = {
                let meta = self.sstable_metadata[bucket].read();
                meta.first().filter(|m| m.entry_count > 0).map(|m| (m.min_key.clone(), m.max_key.clone()))
            };
            let bloom = self.sstable_blooms[bucket].read();
            pending
                .iter()
                .filter(|(_, slot)| slot.is_none())
                .map(|(k, _)| *k)
                .filter(|k| min_max.as_ref().is_none_or(|(min, max)| *k >= min.as_slice() && *k <= max.as_slice()))
                .filter(|k| bloom.as_ref().is_none_or(|b| b.contains(k)))
                .collect()
        };
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
