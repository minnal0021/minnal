//! Sealing the active memtable and flushing read-only memtables to Level-0 files,
//! with the flush observer's events around each flush.

use super::*;

impl LSMTree {
    /// Register or clear a flush observer.
    pub(crate) fn set_flush_observer(&self, observer: Option<Arc<dyn LsmFlushObserver>>) {
        *self.flush_observer.write() = observer;
    }

    pub(super) fn notify_memtable_sealed(&self, version: u64) {
        if let Some(observer) = self.flush_observer.read().as_ref() {
            observer.on_memtable_sealed(version);
        }
    }

    pub(super) fn notify_ro_memtable_flushing(&self, version: u64) -> std::io::Result<()> {
        match self.flush_observer.read().as_ref() {
            Some(observer) => observer.on_ro_memtable_flushing(version),
            None => Ok(()),
        }
    }

    pub(super) fn notify_ro_memtable_written(&self, version: u64) -> std::io::Result<()> {
        match self.flush_observer.read().as_ref() {
            Some(observer) => observer.on_ro_memtable_written(version),
            None => Ok(()),
        }
    }

    pub(super) fn notify_ro_memtable_flushed_to_level0(&self, version: u64) {
        if let Some(observer) = self.flush_observer.read().as_ref() {
            observer.on_ro_memtable_flushed_to_level0(version);
        }
    }

    pub(super) fn flush_ro_memtable_to_level0(&self, ro_memtable: &ReadOnlyMemTable) -> Result<()> {
        // The flushed check and the flush must be one step. Several paths flush
        // (the compaction worker, a GC pass, the no-WAL unpin) and nothing else
        // serialises them, so an unlocked check would let two write the same
        // memtable: a duplicate L0 file, or (in the same millisecond) the same file
        // name, with the second `File::create` truncating the first mid-write. A second
        // flusher WAITS rather than skipping, because callers rely on the flush
        // being durable when this returns: GC unlinks value-log segments right
        // after, and returning early would let it unlink before the fsync.
        let _flushing = ro_memtable.flush_lock.lock();
        if ro_memtable.is_flushed_to_level0() {
            return Ok(());
        }

        // Level 0 takes memtables in the order they were frozen: each memtable's
        // files must be registered before a newer one's, so that every layer
        // holds newer versions of a key than the layers below it (and the stamp,
        // which orders L0 files, is drawn after this check). Every caller goes
        // through `flush_read_only_memtables`, which flushes oldest first and
        // stops at a failure, so this never fires; it is the guard should a new
        // caller pick one memtable out of order.
        let version = ro_memtable.version();
        let older = self
            .read_only_memtables
            .read()
            .iter()
            .find(|ro| ro.version() < version && !ro.is_flushed_to_level0())
            .map(ReadOnlyMemTable::version);
        if let Some(older) = older {
            return Err(LSMError::FlushOutOfOrder { version, older });
        }

        // Nothing is on disk yet, so a veto leaves nothing to undo: the
        // memtable stays read-only and unflushed, and a later flush retries it.
        self.notify_ro_memtable_flushing(version).map_err(LSMError::Io)?;

        let records = ro_memtable.records();
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let timestamp = self.next_l0_stamp(now_ms.saturating_mul(1_000_000).saturating_add(version as u128));

        // Write and fsync every bucket's file first, register them after. Between
        // the two, observers learn the files are durable while nothing can yet
        // read or merge them (`on_ro_memtable_written`): the vector index WAL
        // relies on that to make its progress durable before a merge could drop
        // a tombstone these files hold. Reads stay correct meanwhile, since the
        // read-only memtable is searched until it is marked flushed below.
        let mut written: Vec<(u32, WrittenLevel0File)> = Vec::new();
        for bucket in 0u32..self.config.num_buckets as u32 {
            match self.write_level0_file_for_records(bucket, &records, timestamp) {
                Ok(Some(file)) => written.push((bucket, file)),
                Ok(None) => {}
                Err(e) => {
                    // Unregistered files are not live; remove them so a retry
                    // (with a new stamp) does not leave duplicates on disk.
                    for (_, file) in &written {
                        let _ = std::fs::remove_file(&file.path);
                    }
                    return Err(e);
                }
            }
        }

        if let Err(e) = self.notify_ro_memtable_written(version) {
            // An observer could not record the flush (the vector index WAL's
            // progress marker, or the value-log fsync before it). Registering
            // anyway would make the files mergeable while that progress is not
            // durable, so the flush is undone and retried later.
            for (_, file) in &written {
                let _ = std::fs::remove_file(&file.path);
            }
            return Err(LSMError::Io(e));
        }
        for (bucket, file) in written {
            self.register_level0_file(bucket, file.path, file.created_at_ms, file.summary);
        }

        ro_memtable.mark_flushed_to_level0();
        self.notify_ro_memtable_flushed_to_level0(version);
        Ok(())
    }

    /// Writes and fsyncs (file and directory) one bucket's L0 file for
    /// `records`, without registering it. `None` when the bucket has no records.
    pub(super) fn write_level0_file_for_records(
        &self,
        bucket: u32,
        records: &[KeyValueRecord],
        timestamp_ms: u128,
    ) -> Result<Option<WrittenLevel0File>> {
        let mut bucket_records = Vec::new();
        for record in records {
            let record_bucket = get_bucket_for_key(&record.key, self.config.num_buckets);
            if record_bucket == bucket {
                bucket_records.push(record.clone());
            }
        }

        if bucket_records.is_empty() {
            return Ok(None);
        }

        bucket_records.sort_by(|a, b| a.key.cmp(&b.key));
        let mut summary = L0Summary::of_sorted(bucket_records.iter().map(|r| (r.key.as_slice(), r.seq)));
        // Absolute frame offsets, sampled exactly as `write_merged_sstable` and
        // `load_metadata_from_file` do, so all three agree.
        let mut index = SparseIndex::new();
        let mut offset = SSTABLE_DATA_START;

        let level0_dir = Self::level0_bucket_dir_from(&self.base_path, bucket);
        let dir_is_new = !level0_dir.exists();
        std::fs::create_dir_all(&level0_dir)?;

        // Level 0 files are named by timestamp/version, not by a global sequence.
        let level0_path = level0_dir.join(format!("{}.dat", timestamp_ms));
        // Written under a temporary name and renamed once complete and fsynced, as
        // L1 is, so a `.dat` file is always whole. Written in place, a process
        // killed mid-flush would leave a `.dat` cut anywhere, even inside a frame,
        // and the next open would fail on it (`UnexpectedEof`, or a frame longer
        // than its file) until the file was removed by hand.
        // Open deletes leftover temporaries (`load_existing_level0_files`).
        let tmp_path = level0_dir.join(format!("{timestamp_ms}{LEVEL0_TMP_SUFFIX}"));

        let write = || -> Result<()> {
            let file = std::fs::File::create(&tmp_path)?;
            let mut out = std::io::BufWriter::new(&file);
            out.write_all(&sstable_header_bytes())?;
            for (i, entry) in bucket_records.into_iter().enumerate() {
                if (i as u64).is_multiple_of(SparseIndex::SAMPLE_INTERVAL) {
                    index.push(entry.key.clone(), offset);
                }
                let key_prefix = key_prefix_of(&entry.key);
                let entry_obj = SStableEntry {
                    key: entry.key,
                    key_prefix,
                    value: entry.value,
                    tombstone: entry.tombstone,
                    seq: entry.seq,
                };
                let payload = encode_sstable_entry(&entry_obj)?;
                out.write_all(&(payload.len() as u32).to_le_bytes())?;
                out.write_all(&payload)?;
                offset += 4 + payload.len() as u64;
            }
            out.flush()?;
            drop(out);
            file.sync_all()?;
            std::fs::rename(&tmp_path, &level0_path)?;
            Ok(())
        };
        let written = write();
        if written.is_err() {
            let _ = std::fs::remove_file(&tmp_path);
        }
        written?;
        // Make the new file's directory entry durable too (the rename's, here). `sync_all` on the file
        // covers its contents, not its name: until the directory is fsynced a power
        // loss can drop the file entirely — after the flush observer has already
        // let its WAL entries be marked persisted, leaving nothing to replay. A
        // directory this call created needs its own entry made durable as well.
        File::open(&level0_dir)?.sync_all()?;
        if dir_is_new {
            let level0_root = Self::level0_dir_from(&self.base_path);
            File::open(&level0_root)?.sync_all()?;
            File::open(&self.base_path)?.sync_all()?;
        }
        if let Some(summary) = summary.as_mut() {
            summary.index = (!index.is_empty()).then_some(index);
        }

        Ok(Some(WrittenLevel0File {
            path: level0_path,
            created_at_ms: timestamp_ms,
            summary,
        }))
    }

    /// Flush current memtable to read-only layer
    pub(super) fn flush_memtable(&self) -> Result<()> {
        if let Some(m) = self.metrics() {
            Metrics::bump(&m.memtable_flushes);
        }
        let mut memtable = self.memtable.write();
        let records = memtable.collect_records();
        // These records are leaving the active memtable for a lower layer, so
        // advance the lower-layer sequence bound to cover them.
        if let Some(max) = records.iter().map(|r| r.seq).reduce(|a, b| if seq_newer(b, a) { b } else { a }) {
            self.note_lower_seq(max);
        }
        let version = self.ro_memtable_version.fetch_add(1, Ordering::SeqCst);
        let ro_memtable = ReadOnlyMemTable::new(records, version);

        {
            let mut ro_memtables = self.read_only_memtables.write();
            ro_memtables.push(ro_memtable);
        }

        *memtable = MemTable::with_capacity(self.config.skip_list_capacity);
        self.notify_memtable_sealed(version);
        Ok(())
    }

    /// Flush the current memtable (if non-empty) and all read-only memtables to
    /// level-0 SSTable files on disk.  Unlike `flush_and_compact_all` this does
    /// not run a full compaction, so it is cheaper and suitable for use in the
    /// GC path, which must make its in-memory pointer updates durable before it
    /// unlinks the value-log segments they moved out of.
    pub(crate) fn flush_memtable_to_level0(&self) -> Result<()> {
        self.flush_to_level0(false)
    }

    /// As [`Self::flush_memtable_to_level0`], but seals the active memtable
    /// even when it is empty, so the observers see a seal and a flush. The
    /// vector index WAL needs that to retire a namespace whose records are all
    /// on disk already (or were never applied) while its memtable is empty.
    pub(crate) fn seal_and_flush_to_level0(&self) -> Result<()> {
        self.flush_to_level0(true)
    }

    pub(super) fn flush_to_level0(&self, seal_empty: bool) -> Result<()> {
        {
            let memtable = self.memtable.read();
            let has_data = memtable.skip_list.number_of_live_nodes() > 0 || memtable.skip_list.number_of_tombstone_nodes() > 0;
            drop(memtable);
            if has_data || seal_empty {
                self.flush_memtable()?;
            }
        }
        self.flush_read_only_memtables()
    }

    /// Flush every read-only memtable to Level 0, oldest first, stopping at the
    /// first failure — the one way memtables reach L0, so they arrive in the
    /// order they were frozen. A memtable another thread is flushing is waited
    /// for (its flush lock), so on return every memtable in the snapshot is on
    /// disk.
    pub(super) fn flush_read_only_memtables(&self) -> Result<()> {
        let snapshot = self.read_only_memtables.read().clone();
        for ro in snapshot.iter() {
            self.flush_ro_memtable_to_level0(ro)?;
        }
        Ok(())
    }

    /// Force flush all data to persistent SSTables (for database close)
    pub(crate) fn flush_and_compact_all(&self) -> Result<()> {
        {
            let memtable = self.memtable.read();
            if memtable.skip_list.number_of_live_nodes() > 0 || memtable.skip_list.number_of_tombstone_nodes() > 0 {
                drop(memtable);
                self.flush_memtable()?;
            }
        }

        self.compact_all()?;
        Ok(())
    }

    /// Take the flushed read-only memtables older than the current version off
    /// the read path; their records are in L0 files now. A memtable not yet
    /// flushed stays. Nothing waits for readers: a point read holds the list's
    /// lock while it reads, and a scan holds its own clone of the records
    /// (`Arc`), which outlives the removal.
    pub(super) fn release_flushed_memtables(&self) {
        let version = self.ro_memtable_version.load(Ordering::SeqCst);
        self.read_only_memtables
            .write()
            .retain(|ro| ro.version() >= version || !ro.is_flushed_to_level0());
    }
}
