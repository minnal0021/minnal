//! Multi-prefix scans (`scan_prefixes`): every live key under any of a set of
//! 4-byte prefixes, each layer seeked once per prefix.

use super::*;

impl LSMTree {
    /// Visit every entry of `bucket`'s L1 file whose key starts with one of
    /// `sorted_ids` (4-byte big-endian prefixes, ascending), in key order, with one
    /// forward cursor over the file.
    ///
    /// Each prefix's seek point comes from the sparse index, all read under one
    /// lock before any I/O, and the cursor jumps only when that point lies past it.
    /// When the prefixes are close together, as when a search probes most
    /// postings, the jumps vanish and this is one sequential read. Calling
    /// [`for_each_l1_entry_in_range`](Self::for_each_l1_entry_in_range) per prefix
    /// instead would start a fresh reader (and an `fstat` and a seek check) for
    /// every prefix, the main cost of a search probing hundreds of postings.
    ///
    /// Keys shorter than 4 bytes have no prefix id and are skipped, as the L0 and
    /// memtable passes of [`scan_prefixes`](Self::scan_prefixes) skip them.
    pub(super) fn for_each_l1_entry_in_prefixes(
        &self,
        bucket: usize,
        file: &File,
        sorted_ids: &[u32],
        visit: impl FnMut(&ArchivedSStableEntry),
    ) -> Result<()> {
        let starts: Vec<u64> = {
            let index = self.sstable_indexes[bucket].read();
            sorted_ids
                .iter()
                .map(|id| index.as_ref().map(|i| i.block_start(&id.to_be_bytes())).unwrap_or(0))
                .collect()
        };
        Self::for_each_entry_in_prefixes(file, &starts, sorted_ids, visit).map(|_| ())
    }

    /// The cursor behind [`for_each_l1_entry_in_prefixes`](Self::for_each_l1_entry_in_prefixes),
    /// for any key-sorted SSTable: `starts[i]` is a sparse-index hint for
    /// `sorted_ids[i]` (0 for none), used only after
    /// [`valid_scan_start`](Self::valid_scan_start) accepts it. A file without an
    /// index is read from its head to its last requested prefix. Returns the bytes
    /// read (tests use it to check that the cursor seeks).
    pub(super) fn for_each_entry_in_prefixes(
        file: &File,
        starts: &[u64],
        sorted_ids: &[u32],
        mut visit: impl FnMut(&ArchivedSStableEntry),
    ) -> Result<u64> {
        debug_assert!(sorted_ids.windows(2).all(|w| w[0] < w[1]), "prefixes must be sorted and distinct");
        debug_assert_eq!(starts.len(), sorted_ids.len());
        let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
        let mut frames = FrameReader::new(file, SSTABLE_DATA_START);
        let mut next = 0usize;
        'prefixes: while next < sorted_ids.len() {
            let wanted = sorted_ids[next];
            if starts[next] > frames.offset() && Self::valid_scan_start(file, starts[next], file_len, &wanted.to_be_bytes()) {
                frames.seek(starts[next]);
            }
            while let Some(body) = frames.next_frame(file_len)? {
                let archived = unsafe { rkyv::access_unchecked::<ArchivedSStableEntry>(verify_sstable_payload(body)?) };
                let Some(prefix) = archived.key.as_slice().get(..4) else {
                    continue;
                };
                let id = u32::from_be_bytes(prefix.try_into().unwrap());
                if id < wanted {
                    continue;
                }
                while next < sorted_ids.len() && sorted_ids[next] < id {
                    next += 1;
                }
                if next == sorted_ids.len() {
                    break 'prefixes;
                }
                if sorted_ids[next] == id {
                    visit(archived);
                    continue;
                }
                // `id` lies between two requested prefixes: jump to the next one if
                // its block starts past the cursor, else keep reading.
                continue 'prefixes;
            }
            break;
        }
        Ok(frames.bytes_read)
    }

    /// Scan multiple 4-byte BE u32 prefixes in one call, returning live
    /// `(key, pointer)` pairs in ascending key order.
    ///
    /// Each layer is read once for the whole prefix set rather than once per prefix:
    ///
    /// * **L1** — the sorted ids are split into contiguous groups, one scoped thread
    ///   per group (as many as the buckets whose key span overlaps them). Each
    ///   thread seeks to every id of its group through each bucket's sparse index
    ///   ([`for_each_l1_entry_in_prefixes`](Self::for_each_l1_entry_in_prefixes))
    ///   and merges the buckets' key-sorted runs itself, so the groups' outputs
    ///   follow one another in key order.
    /// * **L0** — one scoped thread per bucket holding L0 files; a file whose key
    ///   range misses every id is skipped, the rest are seeked per id through
    ///   their sparse index and read only within the requested ranges.
    /// * **Memtables** — the active one is range-iterated per prefix; read-only ones
    ///   are filtered.
    ///
    /// The layers are resolved like [`merge_scan_layers`]: L1's key-sorted runs are
    /// merged in bulk, then a sorted merge lets the newer layers shadow them. The
    /// newer layers go through a `BTreeMap` first, which is cheap because they are
    /// usually small; L1 does not, since a map insert per entry would be most of the
    /// cost of a large scan.
    ///
    /// The caller maps entries back to their originating prefix via the first 4
    /// bytes of each key.
    ///
    /// Its pointers can go stale as [`scan_prefix`](Self::scan_prefix)'s can.
    pub(crate) fn scan_prefixes(&self, prefix_ids: &std::collections::HashSet<u32>) -> Result<PrefixScanKeys> {
        if prefix_ids.is_empty() {
            return Ok(PrefixScanKeys::default());
        }

        // Layers resolve by highest seq (`fold_newest`, oldest→newest: L1 → L0 →
        // ro_memtables → active). The *capture* order is newest-first so a
        // concurrent flush/compaction cannot drop a live key — see `scan_prefix`.
        //
        // L1 — the bulk of a large scan — is gathered into a Vec and resolved by
        // `resolve_scan_layers`, not inserted into the map: a BTreeMap insert of an
        // owned key per entry measured ~8 ms of a ~10 ms scan over 87k entries,
        // against ~1 ms for the parallel L1 reads. The newer layers (usually small) still
        // merge through `newer` by `fold_newest`. Each worker packs its L1 keys into
        // one buffer (`l1_keys`) instead of allocating one per entry.
        let mut l1_keys: Vec<Vec<u8>> = Vec::new();
        let mut l1_entries: Vec<PackedScanEntry> = Vec::new();
        let mut newer = ScanMap::new();

        // Memtable layers (newest) captured first, under one `memtable.read()`
        // guard so a flush cannot split a key between active and RO.
        let (active_snapshot, ro_snapshot) = {
            let memtable = self.memtable.read();
            let mut active = Vec::new();
            for prefix_id in prefix_ids {
                let prefix = prefix_id.to_be_bytes();
                for (key, value, seq, tombstone) in memtable.skip_list.iter_raw_from(&prefix) {
                    if key.len() < 4 || u32::from_be_bytes(key[..4].try_into().unwrap()) != *prefix_id {
                        break;
                    }
                    active.push((key.to_vec(), (!tombstone).then_some(value), seq));
                }
            }
            let ro = self.read_only_memtables.read().clone();
            (active, ro)
        };

        // SSTable handles captured newest-first (L0 before L1); read in merge
        // order (L1 then L0). Guards in `l0_guards` pin the files until we finish.
        let (l0_guards, l1_files) = self.capture_sstable_layers();

        // Each requested prefix is a 4-byte big-endian cluster id; a prefix scan
        // over id `P` is a range over `[P, P+1)`. Sorted so per-bucket seeks are
        // monotonic. The span `[min_id, max_id+1)` gives a cheap per-bucket overlap
        // test (keys are hash-sharded, so a bucket's L1 spans many ids), and each
        // bucket then seeks per requested id via the sparse index rather than
        // reading the whole file — the win for probing a few clusters out of a
        // large L1.
        let mut sorted_ids: Vec<u32> = prefix_ids.iter().copied().collect();
        sorted_ids.sort_unstable();
        let sorted_ids = &sorted_ids;
        let span_start = sorted_ids[0].to_be_bytes();
        let span_end: Option<[u8; 4]> = sorted_ids[sorted_ids.len() - 1].checked_add(1).map(u32::to_be_bytes);
        let span_end = span_end.as_ref().map(|e| e.as_slice());

        // ── 1. L1 SSTables (oldest layer) — postings split into groups, read in
        //    parallel. Each worker reads its group's postings from every overlapping
        //    bucket and merges those runs itself; the groups are ascending, disjoint
        //    ranges of prefix ids, so their merged outputs simply follow one another
        //    in key order. One worker per bucket, with the buckets' runs merged
        //    afterwards on one thread, measured about half a 1,024-posting scan's time.
        let l1_buckets: Vec<(usize, &Arc<File>)> = l1_files
            .iter()
            .enumerate()
            .filter(|(bucket, _)| self.l1_overlaps(*bucket, &span_start, span_end))
            .collect();
        let l1_buckets = &l1_buckets;
        if !l1_buckets.is_empty() {
            let group_len = sorted_ids.len().div_ceil(l1_buckets.len());
            let mut first_err: Option<LSMError> = None;
            std::thread::scope(|s| {
                let handles: Vec<_> = sorted_ids
                    .chunks(group_len)
                    .map(|group| {
                        s.spawn(move || -> Result<(Vec<u8>, Vec<PackedScanEntry>)> {
                            let mut keys = Vec::new();
                            let mut entries = Vec::new();
                            for &(bucket, file) in l1_buckets {
                                self.for_each_l1_entry_in_prefixes(bucket, file, group, |archived| {
                                    let val = (!archived.tombstone).then(|| archived.value.to_native());
                                    let key_start = keys.len();
                                    keys.extend_from_slice(archived.key.as_slice());
                                    entries.push((0, key_start, keys.len(), val, SStableEntry::archived_version(archived)));
                                })?;
                            }
                            let merged = merge_l1_runs(&entries, |e: &PackedScanEntry| &keys[e.1..e.2]);
                            Ok((keys, merged))
                        })
                    })
                    .collect();
                for handle in handles {
                    match handle
                        .join()
                        .unwrap_or_else(|_| Err(LSMError::Io(std::io::Error::other("thread panicked"))))
                    {
                        Ok((keys, entries)) => {
                            let buffer = l1_keys.len();
                            l1_keys.push(keys);
                            l1_entries.extend(entries.into_iter().map(|(_, start, end, val, seq)| (buffer, start, end, val, seq)));
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

        // ── 2. L0 SSTables (newer than L1) — captured guards, oldest-first per
        //    bucket so an exact seq tie goes to the newer file. Only buckets that hold
        //    a captured L0 file spawn a worker. A file whose key range misses every
        //    probed prefix is skipped, and the rest are read with one cursor that
        //    seeks per prefix through the file's sparse index, as L1 is: L0 can hold
        //    a tenth of L1's entries before it is merged, and reading every file
        //    whole would cost every search that share of L1 again.
        {
            let mut first_err: Option<LSMError> = None;
            std::thread::scope(|s| {
                let handles: Vec<_> = l0_guards
                    .iter()
                    .filter(|guards| !guards.is_empty())
                    .map(|guards| {
                        s.spawn(move || -> Result<Vec<ScanEntry>> {
                            let mut bucket_entries = Vec::new();
                            for guard in guards {
                                let summary = guard.entry.summary.as_ref();
                                if summary.is_some_and(|s| {
                                    s.max_key.as_slice() < span_start.as_slice() || span_end.is_some_and(|e| s.min_key.as_slice() >= e)
                                }) {
                                    continue;
                                }
                                let file = open_registered_l0(&guard.entry.path)?;
                                let starts: Vec<u64> = sorted_ids.iter().map(|id| guard.entry.scan_start(&id.to_be_bytes())).collect();
                                Self::for_each_entry_in_prefixes(&file, &starts, sorted_ids, |archived| {
                                    let val = (!archived.tombstone).then(|| archived.value.to_native());
                                    bucket_entries.push((archived.key.as_slice().to_vec(), val, SStableEntry::archived_version(archived)));
                                })?;
                            }
                            Ok(bucket_entries)
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
                                fold_newest(&mut newer, k, v, seq);
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

        // ── 3. Read-only memtables oldest-first ───────────────────────────────
        for ro_memtable in ro_snapshot.iter() {
            for record in ro_memtable.records().iter() {
                if record.key.len() >= 4 {
                    let prefix_id = u32::from_be_bytes(record.key[..4].try_into().unwrap());
                    if prefix_ids.contains(&prefix_id) {
                        fold_newest(
                            &mut newer,
                            record.key.clone(),
                            (!record.tombstone).then_some(record.value),
                            record.version,
                        );
                    }
                }
            }
        }

        // ── 4. Active memtable (newest layer) ─────────────────────────────────
        for (key, value, seq) in active_snapshot {
            fold_newest(&mut newer, key, value, seq);
        }

        Ok(resolve_scan_layers(&l1_keys, l1_entries, newer))
    }
}
