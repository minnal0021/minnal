//! Merging a bucket's Level-0 files into its Level-1 file, the newer layer's
//! copy of each key winning, and
//! deciding when a bucket's merge is due.

use super::*;

impl LSMTree {
    /// Perform 2-way merge of existing entries and new records
    pub(super) fn two_way_merge(
        &self,
        existing: Vec<SStableEntry>,
        new_records: Vec<KeyValueRecord>,
        displaced: &mut DisplacedCollector,
    ) -> Vec<SStableEntry> {
        let mut result = Vec::new();
        let mut i = 0;
        let mut j = 0;

        while i < existing.len() && j < new_records.len() {
            let existing_entry = &existing[i];
            let new_record = &new_records[j];

            match existing_entry.key.as_slice().cmp(new_record.key.as_slice()) {
                std::cmp::Ordering::Less => {
                    if !existing_entry.tombstone {
                        result.push(existing_entry.clone());
                    }
                    i += 1;
                }
                std::cmp::Ordering::Equal => {
                    // The L0 record is from the newer layer, so it is the key's
                    // current copy (`docs/Layer-Order-Design.md`). A tombstone drops
                    // the key: L1 is the bottom level, so there is nothing below
                    // for it to shadow.
                    let new_pointer = (!new_record.tombstone).then_some(new_record.value);
                    displaced.lost(existing_entry.key.len(), existing_entry.live_pointer(), new_pointer);
                    if !new_record.tombstone {
                        result.push(SStableEntry::of_record(new_record));
                    }
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Greater => {
                    if !new_record.tombstone {
                        result.push(SStableEntry::of_record(new_record));
                    }
                    j += 1;
                }
            }
        }

        while i < existing.len() {
            if !existing[i].tombstone {
                result.push(existing[i].clone());
            }
            i += 1;
        }

        while j < new_records.len() {
            if !new_records[j].tombstone {
                result.push(SStableEntry::of_record(&new_records[j]));
            }
            j += 1;
        }

        result
    }

    /// Write merged entries to a temporary SSTable file, building the sparse
    /// index from the byte offsets as it writes (no extra pass).
    pub(super) fn write_merged_sstable(&self, path: &Path, entries: &[SStableEntry], merged_through: u64) -> Result<MergedSstableInfo> {
        if entries.is_empty() {
            return Err(LSMError::Serialization("Cannot write empty SSTable".to_string()));
        }

        let mut file = OpenOptions::new().create(true).write(true).truncate(true).open(path)?;
        file.write_all(&sstable_header_bytes(merged_through))?;

        // Counts entry bytes only — it is returned as the merge's `bytes_merged`
        // metric, not as a file position. The sparse index adds `SSTABLE_DATA_START`
        // so its offsets stay absolute.
        let mut total_size = 0u64;
        let min_key = entries[0].key.clone();
        let max_key = entries[entries.len() - 1].key.clone();
        let mut index = SparseIndex::new();

        for (i, entry) in entries.iter().enumerate() {
            if (i as u64).is_multiple_of(SparseIndex::SAMPLE_INTERVAL) {
                index.push(entry.key.clone(), SSTABLE_DATA_START + total_size);
            }
            let payload = encode_sstable_entry(entry)?;
            let size = payload.len() as u32;
            file.write_all(&size.to_le_bytes())?;
            file.write_all(&payload)?;
            total_size += 4 + size as u64;
        }

        file.flush()?;
        file.sync_all()?;

        Ok((min_key, max_key, entries.len() as u64, total_size, index))
    }

    /// Merge all Level 0 files into Level 1 file for a bucket
    pub(super) fn merge_level0_to_level1(&self, bucket: u32) -> Result<()> {
        let mut entries = self.level0_entries_snapshot(bucket);
        if entries.is_empty() {
            return Ok(());
        }
        let compaction_started = std::time::Instant::now();
        let mut bytes_merged = 0u64;

        // Collapse same-key copies across L0 files: files are read oldest first,
        // so a newer file's copy replaces an older one's — the newer layer holds
        // the newer version (`docs/Layer-Order-Design.md`).
        entries.sort_by_key(|a| a.created_at_ms);

        let mut by_key: std::collections::BTreeMap<Vec<u8>, SStableEntry> = std::collections::BTreeMap::new();
        let mut displaced = DisplacedCollector::default();
        let mut any_read = false;
        // The newest stamp this merge takes in, recorded in the new L1 file so an
        // open after a crash deletes the merged files instead of reading them
        // again. A stamp past `u64` (centuries away) is not recorded: the merged
        // files are then only removed by cleanup, which is safe, just slower.
        let mut merged_through = self.level1_merged_through[bucket as usize].load(Ordering::SeqCst);
        for entry in &entries {
            if entry.is_obsolete() {
                continue;
            }
            any_read = true;
            if let Ok(stamp) = u64::try_from(entry.created_at_ms) {
                merged_through = merged_through.max(stamp);
            }
            let _guard = L0ReadGuard::new(Arc::clone(entry));
            // Within one file each key appears once; across files the newer
            // file's copy replaces the older one's.
            for record in self.read_sstable_entries_from_path(&entry.path)? {
                let key = record.key.clone();
                if let Some(old) = by_key.insert(key, record) {
                    displaced.lost(old.key.len(), old.live_pointer(), by_key[&old.key].live_pointer());
                }
            }
        }

        if !any_read || by_key.is_empty() {
            return Ok(());
        }

        let level0_records: Vec<KeyValueRecord> = by_key
            .into_values()
            .map(|e| KeyValueRecord {
                version: e.version(),
                key: e.key,
                value: e.value,
                tombstone: e.tombstone,
            })
            .collect();

        let level1_path = Self::level1_path_from(&self.base_path, bucket);
        let level1_entries = self.read_sstable_entries(bucket)?;

        let merged_entries = self.two_way_merge(level1_entries, level0_records, &mut displaced);

        if merged_entries.is_empty() {
            let bucket_idx = bucket as usize;
            // A header-only file: no entries, but it keeps `merged_through`, so the
            // files this merge emptied are still recognised as merged at open.
            // Published like a full one: written, fsynced, renamed, directory
            // fsynced.
            let temp_path = Self::level1_dir_from(&self.base_path).join(format!("level1_{}.dat.tmp", bucket));
            {
                let mut file = OpenOptions::new().create(true).write(true).truncate(true).open(&temp_path)?;
                file.write_all(&sstable_header_bytes(merged_through))?;
                file.sync_all()?;
            }
            let null_file = Arc::new(Self::create_null_file()?);
            {
                let mut file_guard = self.sstable_files[bucket_idx].write();
                std::fs::rename(&temp_path, &level1_path)?;
                File::open(Self::level1_dir_from(&self.base_path))?.sync_all()?;
                *file_guard = null_file;
            }
            let mut metadata_guard = self.sstable_metadata[bucket_idx].write();
            metadata_guard.clear();
            // Empty filter, as at open: every lookup in this bucket's L1 is a miss.
            *self.sstable_blooms[bucket_idx].write() = Some(Arc::new(BloomFilter::build([], 0)));
            *self.sstable_indexes[bucket_idx].write() = None;
        } else {
            let temp_path = Self::level1_dir_from(&self.base_path).join(format!("level1_{}.dat.tmp", bucket));
            let (min_key, max_key, entry_count, data_size, index) = self.write_merged_sstable(&temp_path, &merged_entries, merged_through)?;
            bytes_merged = data_size;

            let bucket_idx = bucket as usize;
            {
                let mut file_guard = self.sstable_files[bucket_idx].write();

                // Publish with a SINGLE atomic rename. `rename(2)` replaces the
                // destination atomically, so the bucket's L1 path always names a
                // complete SSTable — the old one before this call, the new one after,
                // never nothing.
                //
                // Never move the old file aside first (`rename(level1 -> .old)`,
                // then `rename(.tmp -> level1)`): between the two renames the L1
                // path would not exist, `LSMTree::open` after a crash there would
                // create an empty file and read the bucket as empty, and every key
                // that lived only in L1 would be lost. Test:
                // `test_crash_between_l1_publish_steps_keeps_l1`.
                //
                // Readers holding the previous `Arc<File>` keep reading the old
                // inode until they drop it.
                std::fs::rename(&temp_path, &level1_path)?;

                // Make the rename itself durable: without an fsync of the containing
                // directory the swap can be lost on power failure, silently reverting
                // to the pre-merge L1. That is consistent (the L0 inputs are still
                // there, so it only costs the merge work) but it is not what the code
                // above just reported as done.
                if let Ok(dir) = File::open(Self::level1_dir_from(&self.base_path)) {
                    let _ = dir.sync_all();
                }

                let new_file = OpenOptions::new().read(true).write(true).open(&level1_path)?;
                *file_guard = Arc::new(new_file);
            }

            let mut metadata_guard = self.sstable_metadata[bucket_idx].write();
            metadata_guard.clear();
            metadata_guard.push(SStableMetadata {
                min_key,
                max_key,
                entry_count,
            });

            // Rebuild the L1 bloom from the merged entries (in memory, no extra
            // I/O). Replaced after the file swap so it always matches the file
            // now installed.
            let bloom = BloomFilter::build(merged_entries.iter().map(|e| e.key.as_slice()), merged_entries.len());
            *self.sstable_blooms[bucket_idx].write() = Some(Arc::new(bloom));
            // The sparse index was built from the byte offsets as the file was
            // written, so it matches the file just installed.
            *self.sstable_indexes[bucket_idx].write() = (!index.is_empty()).then(|| Arc::new(index));
        }

        // Published with the file, before its inputs are marked obsolete.
        self.level1_merged_through[bucket as usize].store(merged_through, Ordering::SeqCst);
        for entry in entries {
            entry.mark_obsolete();
        }

        // Only now, with the merged file installed, are the dropped entries gone.
        if let Some(sink) = self.displaced_sink.get() {
            let records = displaced.into_records();
            if !records.is_empty() {
                sink.on_displaced(&records);
            }
        }

        if let Some(m) = self.metrics() {
            Metrics::bump(&m.l0_l1_compactions);
            Metrics::add(&m.compaction_bytes_merged, bytes_merged);
            Metrics::add(&m.compaction_duration_ms, compaction_started.elapsed().as_millis() as u64);
        }

        Ok(())
    }

    /// Compact a specific bucket's SSTable asynchronously
    pub(crate) fn compact_bucket(&self, bucket: u32) -> Result<()> {
        let bucket_idx = bucket as usize;

        if self.compaction_in_progress[bucket_idx].swap(true, Ordering::SeqCst) {
            return Ok(());
        }

        // Clear the in-progress flag on every exit path — including the `?` error
        // return below and any panic. Resetting only on the success path would
        // let a failed merge wedge the flag at `true`, making every later
        // compact_bucket for this bucket a silent no-op until restart.
        struct InProgressGuard<'a>(&'a AtomicBool);
        impl Drop for InProgressGuard<'_> {
            fn drop(&mut self) {
                self.0.store(false, Ordering::SeqCst);
            }
        }
        let _guard = InProgressGuard(&self.compaction_in_progress[bucket_idx]);

        self.merge_level0_to_level1(bucket)?;
        Ok(())
    }

    /// Compact all buckets in parallel
    pub(crate) fn compact_all(&self) -> Result<()> {
        self.cleanup_obsolete_level0_files();

        if self.read_only_memtables.read().is_empty() && !self.has_level0_files() {
            return Ok(());
        }
        self.flush_read_only_memtables()?;

        for bucket in 0u32..self.config.num_buckets as u32 {
            if let Err(err) = self.compact_bucket(bucket) {
                if let LSMError::Io(io_err) = &err
                    && io_err.kind() == std::io::ErrorKind::NotFound
                {
                    continue;
                }
                return Err(err);
            }
        }

        self.release_flushed_memtables();
        Ok(())
    }

    pub(crate) fn has_level0_files(&self) -> bool {
        for bucket in 0u32..self.config.num_buckets as u32 {
            if !self.level0_files[bucket as usize].read().is_empty() {
                return true;
            }
        }
        false
    }

    pub(crate) fn has_compaction_work(&self) -> bool {
        !self.read_only_memtables.read().is_empty() || self.has_level0_files()
    }

    /// Whether `bucket`'s Level-0 files should be merged into its Level 1 now.
    ///
    /// A merge rewrites the bucket's whole Level-1 file, so merging whatever L0
    /// holds at every opportunity costs the size of Level 1 per flush. Callers
    /// that flush small memtables often (the vector worker's durability barrier,
    /// one flush per batch) would then rewrite megabytes to add a dozen entries:
    /// merging at every flush, the system benchmark at 50 requests/s measured 10
    /// flushes and 16 merges a second, about 2 MB rewritten per document and 93%
    /// of all disk writes.
    ///
    /// So a bucket waits until its L0 files hold [`L0_COMPACTION_RATIO_PERCENT`]
    /// of Level 1's entries, which bounds the rewrite to about 100 / that times
    /// what was flushed, or until there are [`L0_COMPACTION_MAX_FILES`] of them,
    /// which bounds what a point read may have to open. An L0 file without a
    /// summary counts as due: its size is unknown.
    pub(crate) fn compaction_due(&self, bucket: u32) -> bool {
        let files = self.level0_files[bucket as usize].read();
        let live: Vec<&Arc<L0FileEntry>> = files.iter().filter(|f| !f.is_obsolete()).collect();
        if live.is_empty() {
            return false;
        }
        if live.len() >= L0_COMPACTION_MAX_FILES {
            return true;
        }
        let mut l0_entries = 0u64;
        for f in &live {
            match &f.summary {
                Some(s) => l0_entries += s.entries,
                None => return true,
            }
        }
        let l1_entries: u64 = self.sstable_metadata[bucket as usize].read().iter().map(|m| m.entry_count).sum();
        l0_entries * 100 >= l1_entries * L0_COMPACTION_RATIO_PERCENT
    }

    /// Whether [`compact_due`](Self::compact_due) has anything to do: a read-only
    /// memtable to flush or release (a flushed one is held until a tick drops
    /// it), or a bucket whose merge is due.
    pub(crate) fn has_due_compaction_work(&self) -> bool {
        !self.read_only_memtables.read().is_empty() || (0..self.config.num_buckets as u32).any(|b| self.compaction_due(b))
    }

    /// The background worker's compaction: flush every read-only memtable to
    /// Level 0 (they pin the WAL), then merge only the buckets whose merge is
    /// [due](Self::compaction_due). [`compact_all`](Self::compact_all) merges
    /// every bucket with any L0 file, for shutdown and explicit requests.
    pub(crate) fn compact_due(&self) -> Result<()> {
        self.cleanup_obsolete_level0_files();

        self.flush_read_only_memtables()?;

        for bucket in 0u32..self.config.num_buckets as u32 {
            if !self.compaction_due(bucket) {
                continue;
            }
            if let Err(err) = self.compact_bucket(bucket) {
                if let LSMError::Io(io_err) = &err
                    && io_err.kind() == std::io::ErrorKind::NotFound
                {
                    continue;
                }
                return Err(err);
            }
        }

        self.release_flushed_memtables();
        Ok(())
    }
}
