//! Whole-tree scans: every live key with its pointer (`key_pointer_pairs`), and
//! every pointer any layer holds (`referenced_pointers`).

use super::*;

impl LSMTree {
    /// Every distinct value-log pointer held by any entry in any layer, newest
    /// or not, with its key's length, for the buckets `qualifying_buckets` marks
    /// (all when `None`). This is what the value-log counters call live (see
    /// `tree/displaced.rs`), so rebuilding them from it agrees with the running
    /// count. Reads every layer of those buckets in full; meant for open, with
    /// no writes running.
    pub(crate) fn referenced_pointers(&self, qualifying_buckets: Option<&[bool]>) -> Result<Vec<Displaced>> {
        let wanted = |key: &[u8]| qualifying_buckets.is_none_or(|qb| qb[get_bucket_for_key(key, self.config.num_buckets) as usize]);
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        let mut add = |key: &[u8], pointer: u128| {
            if seen.insert(pointer) {
                out.push(Displaced { key_len: key.len(), pointer });
            }
        };
        {
            let memtable = self.memtable.read();
            for (key, value, _seq, tombstone) in memtable.skip_list.iter_raw() {
                if !tombstone && wanted(key) {
                    add(key, value);
                }
            }
        }
        for ro in self.read_only_memtables.read().iter() {
            for r in ro.records().iter() {
                if !r.tombstone && wanted(&r.key) {
                    add(&r.key, r.value);
                }
            }
        }
        for bucket in 0..self.config.num_buckets as u32 {
            if qualifying_buckets.is_some_and(|qb| !qb[bucket as usize]) {
                continue;
            }
            for l0 in self.level0_entries_snapshot(bucket) {
                if l0.is_obsolete() {
                    continue;
                }
                let _guard = L0ReadGuard::new(Arc::clone(&l0));
                for e in self.read_sstable_entries_from_path(&l0.path)? {
                    if let Some(pointer) = e.live_pointer() {
                        add(&e.key, pointer);
                    }
                }
            }
            for e in self.read_sstable_entries(bucket)? {
                if let Some(pointer) = e.live_pointer() {
                    add(&e.key, pointer);
                }
            }
        }
        Ok(out)
    }

    /// Every live key, in key order, as one list in memory.
    pub(crate) fn keys(&self) -> Result<Vec<Vec<u8>>> {
        // `key_pointer_pairs` resolves every layer by seq, so a newer tombstone
        // hides an older live copy.
        Ok(self.key_pointer_pairs(None)?.into_iter().map(|(k, _)| k).collect())
    }

    /// Every live `(key, pointer)`, optionally only for the keys of the buckets
    /// `qualifying_buckets` marks (their SSTables alone are read).
    ///
    /// Its pointers can go stale as [`scan_prefix`](Self::scan_prefix)'s can.
    pub(crate) fn key_pointer_pairs(&self, qualifying_buckets: Option<&[bool]>) -> Result<Vec<(Vec<u8>, u128)>> {
        // Layers are folded oldest→newest (L1 → L0 → read-only → active) into a map
        // by `fold_newest`. The *capture* order is newest-first so a concurrent
        // flush or merge cannot drop a live key, which matters most here: this scan
        // rebuilds the value log's live accounting, where a missed live key would be
        // counted as garbage. See `scan_prefix`.
        //
        // Liveness is resolved by highest write `seq` (globally ordered), matching
        // the seq-resolved point reads (`get_with_seq`) and the seq-aware L0→L1
        // merge. Layer/file recency is NOT used to pick the winner: an out-of-order
        // ro→L0 flush (file `created_at_ms` is set at flush time, which can run out
        // of seal/seq order) or a GC re-point can leave a lower-seq value in a
        // "newer" file above a higher-seq tombstone. Resolving by recency would read
        // a higher-seq live value as dead (a newer-file lower-seq tombstone
        // shadowing it). The
        // map carries each key's best `(value-or-tombstone, seq)`; a later insert
        // wins only if its seq is newer-or-equal. The *capture* order stays
        // newest-first (memtables before SSTables) so a concurrent flush/compaction
        // cannot drop a live key; only the conflict *resolution* is seq-based.
        let mut all_entries = ScanMap::new();

        let buckets_to_read: Vec<u32> = (0u32..self.config.num_buckets as u32)
            .filter(|&b| qualifying_buckets.is_none_or(|qb| qb[b as usize]))
            .collect();

        // Memtable layers (newest) captured first, under one `memtable.read()` guard
        // so a flush cannot split a key between active and RO.
        let (active_snapshot, ro_snapshot) = {
            let memtable = self.memtable.read();
            let mut active = Vec::new();
            for (key, value, seq, tombstone) in memtable.skip_list.iter_raw() {
                if let Some(qb) = qualifying_buckets
                    && !qb[get_bucket_for_key(key, self.config.num_buckets) as usize]
                {
                    continue;
                }
                active.push((key.to_vec(), if tombstone { None } else { Some(value) }, seq));
            }
            let ro = self.read_only_memtables.read().clone();
            (active, ro)
        };

        // SSTable handles captured newest-first (L0 before L1); read in merge order.
        // We capture all buckets but only read the qualifying ones.
        let (l0_guards, l1_files) = self.capture_sstable_layers();

        // 1. Level-1 SSTables (oldest layer) — qualifying buckets read in parallel.
        //    scan_bucket_key_pointers returns only live entries (no tombstones at L1),
        //    so there is no cross-bucket tombstone ordering concern.
        {
            let l1_ref = &l1_files;
            let mut first_err: Option<LSMError> = None;
            std::thread::scope(|s| {
                let handles: Vec<_> = buckets_to_read
                    .iter()
                    .map(|&bucket| s.spawn(move || self.scan_bucket_key_pointers(&l1_ref[bucket as usize])))
                    .collect();
                for handle in handles {
                    match handle
                        .join()
                        .unwrap_or_else(|_| Err(LSMError::Io(std::io::Error::other("thread panicked"))))
                    {
                        Ok(kvs) => {
                            for (key, value, seq) in kvs {
                                fold_newest(&mut all_entries, key, Some(value), seq);
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

        // 2. Level-0 SSTable files (newer than L1, older than the memtables). This
        //    step is essential for correctness: L0 holds data flushed off the
        //    memtable/read-only layer but not yet merged into L1, tombstones
        //    included. Without L0 a flushed tombstone would be invisible, so a key
        //    deleted but still live in L1 would count as live, and a live key that
        //    exists only in L0 would be missed. Captured guards are read oldest-file-first per
        //    bucket so newer L0 entries (and tombstones) shadow older ones.
        {
            let l0_ref = &l0_guards;
            let mut first_err: Option<LSMError> = None;
            std::thread::scope(|s| {
                let handles: Vec<_> = buckets_to_read
                    .iter()
                    .map(|&bucket| {
                        s.spawn(move || -> Result<Vec<ScanEntry>> {
                            let mut out = Vec::new();
                            for guard in &l0_ref[bucket as usize] {
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
                                    let archived =
                                        unsafe { rkyv::access_unchecked::<ArchivedSStableEntry>(verify_sstable_payload(&entry_bytes[..size])?) };
                                    let val = if archived.tombstone { None } else { Some(archived.value.to_native()) };
                                    out.push((archived.key.as_slice().to_vec(), val, archived.seq.to_native()));
                                }
                            }
                            Ok(out)
                        })
                    })
                    .collect();
                for handle in handles {
                    match handle
                        .join()
                        .unwrap_or_else(|_| Err(LSMError::Io(std::io::Error::other("thread panicked"))))
                    {
                        Ok(kvs) => {
                            for (k, v, seq) in kvs {
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

        // 3. Read-only memtables — resolved by seq (oldest-first only tie-breaks).
        for ro_memtable in ro_snapshot.iter() {
            for record in ro_memtable.records().iter() {
                if let Some(qb) = qualifying_buckets
                    && !qb[get_bucket_for_key(&record.key, self.config.num_buckets) as usize]
                {
                    continue;
                }
                fold_newest(
                    &mut all_entries,
                    record.key.clone(),
                    (!record.tombstone).then_some(record.value),
                    record.seq,
                );
            }
        }

        // 4. Active memtable (captured last, but resolved by seq like the rest).
        for (key, value, seq) in active_snapshot {
            fold_newest(&mut all_entries, key, value, seq);
        }

        // Filter out tombstoned entries and collect results
        Ok(all_entries
            .into_iter()
            .filter_map(|(key, (value, _seq))| value.map(|v| (key, v)))
            .collect())
    }

    /// Scan all live key-value pairs (with their write `seq`) from a bucket's
    /// (pre-captured) Level-1 file. Tombstones are skipped: a seq-aware merge
    /// (`merge_level0_to_level1`) never writes a tombstone into L1, so an L1 entry
    /// is always live; a newer layer carrying the key shadows it by seq.
    pub(super) fn scan_bucket_key_pointers(&self, level1_file: &File) -> Result<Vec<(Vec<u8>, u128, u64)>> {
        let mut kvs = Vec::new();
        // Bound for `check_frame_len`: a frame can never be longer than its file.
        let file_len = level1_file.metadata().map(|m| m.len()).unwrap_or(0);
        let mut offset = SSTABLE_DATA_START;
        let mut entry_bytes = Vec::new();

        loop {
            let mut size_buf = [0u8; 4];
            if !Self::read_exact_at(level1_file, &mut size_buf, &mut offset)? {
                break;
            }

            let size = u32::from_le_bytes(size_buf) as usize;
            check_frame_len(size as u64, file_len)?;
            entry_bytes.resize(size, 0);
            Self::read_exact_at(level1_file, &mut entry_bytes[..size], &mut offset)?;

            let archived = unsafe { rkyv::access_unchecked::<ArchivedSStableEntry>(verify_sstable_payload(&entry_bytes[..size])?) };

            if !archived.tombstone {
                kvs.push((archived.key.to_vec(), archived.value.to_native(), archived.seq.to_native()));
            }
        }

        Ok(kvs)
    }
}
