//! Opening a tree: the directory layout, loading each bucket's Level-1 file and
//! the Level-0 files left on disk, and the sequence bounds read from them.

use super::*;

impl LSMTree {
    /// Create or open an LSM tree
    pub(crate) fn open<P: AsRef<Path>>(base_path: P, config: LSMConfig) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        let level1_dir = Self::level1_dir_from(&base_path);
        let level0_dir = Self::level0_dir_from(&base_path);
        std::fs::create_dir_all(&level1_dir)?;
        std::fs::create_dir_all(&level0_dir)?;

        let num_buckets = config.num_buckets;

        // Each bucket's Level-1 file sits behind an `Arc<File>` that a merge swaps.
        let mut sstable_files_vec: Vec<Arc<RwLock<Arc<File>>>> = Vec::with_capacity(num_buckets);
        let mut sstable_metadata_vec: Vec<Arc<RwLock<Vec<SStableMetadata>>>> = Vec::with_capacity(num_buckets);
        let mut sstable_blooms_vec: Vec<Arc<RwLock<Option<Arc<BloomFilter>>>>> = Vec::with_capacity(num_buckets);
        let mut sstable_indexes_vec: Vec<Arc<RwLock<Option<Arc<SparseIndex>>>>> = Vec::with_capacity(num_buckets);
        // Newest sequence across all loaded L1 files — the initial lower-layer
        // bound for the read fast path.
        let mut max_lower_seq_init: Option<u64> = None;

        // Open or create each bucket's Level-1 file.
        for bucket in 0u32..num_buckets as u32 {
            let level1_path = Self::level1_path_from(&base_path, bucket);
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(&level1_path)?;

            let file_len = std::fs::metadata(&level1_path)?.len();
            // Reject an unreadable on-disk format loudly, before anything tries to
            // interpret the file. The metadata load below is deliberately lenient
            // (a partially-written tail leaves the bucket without a sparse index
            // rather than failing open), but that leniency must not extend to a
            // file this build cannot parse at all — scanning it would yield
            // garbage entries instead of an error.
            if file_len > 0 {
                validate_sstable_header(&file, &level1_path)?;
            }
            sstable_files_vec.push(Arc::new(RwLock::new(Arc::new(file))));

            let mut metadata = Vec::new();
            // An empty L1 file gets an empty filter, which rejects every key: a
            // miss then stops here instead of reading the file's size and trying
            // to read it (two system calls per miss on a store with nothing on
            // disk). A file whose metadata fails to load keeps no filter and is
            // scanned.
            let mut bloom = (file_len == 0).then(|| Arc::new(BloomFilter::build([], 0)));
            let mut index = None;
            if file_len > 0
                && let Ok((meta, built_bloom, built_index, file_max_seq)) = Self::load_metadata_from_file(&level1_path)
            {
                metadata.push(meta);
                bloom = built_bloom.map(Arc::new);
                index = built_index.map(Arc::new);
                if let Some(seq) = file_max_seq
                    && max_lower_seq_init.is_none_or(|m| seq_newer(seq, m))
                {
                    max_lower_seq_init = Some(seq);
                }
            }
            sstable_metadata_vec.push(Arc::new(RwLock::new(metadata)));
            sstable_blooms_vec.push(Arc::new(RwLock::new(bloom)));
            sstable_indexes_vec.push(Arc::new(RwLock::new(index)));
        }

        let sstable_files: Vec<Arc<RwLock<Arc<File>>>> = sstable_files_vec;
        let sstable_metadata: Vec<Arc<RwLock<Vec<SStableMetadata>>>> = sstable_metadata_vec;
        let sstable_blooms: Vec<Arc<RwLock<Option<Arc<BloomFilter>>>>> = sstable_blooms_vec;
        let sstable_indexes: Vec<Arc<RwLock<Option<Arc<SparseIndex>>>>> = sstable_indexes_vec;

        let level0_files: Vec<Arc<RwLock<Vec<Arc<L0FileEntry>>>>> = (0..num_buckets).map(|_| Arc::new(RwLock::new(Vec::new()))).collect();

        // L0 is newer than L1, so its sequences must be folded into the read
        // fast-path bound too — otherwise the bound (L1-only above) is too low
        // after a restart and a low-seq active entry could wrongly short-circuit.
        if let Some(seq) = Self::load_existing_level0_files(&base_path, &level0_files)?
            && max_lower_seq_init.is_none_or(|m| seq_newer(seq, m))
        {
            max_lower_seq_init = Some(seq);
        }

        // Seed the L0 stamp floor from the newest file on disk, so the first flush
        // after a restart sorts after every file already there even if the clock
        // is now behind the one that named them.
        let last_l0_stamp = level0_files
            .iter()
            .flat_map(|bucket| bucket.read().iter().map(|e| e.created_at_ms).collect::<Vec<_>>())
            .max()
            .unwrap_or(0);

        // Per-bucket compaction flags
        let compaction_in_progress: Vec<Arc<AtomicBool>> = (0..num_buckets).map(|_| Arc::new(AtomicBool::new(false))).collect();

        Ok(Self {
            memtable: Arc::new(RwLock::new(MemTable::with_capacity(config.skip_list_capacity))),
            read_only_memtables: Arc::new(RwLock::new(Vec::new())),
            sstable_files,
            sstable_metadata,
            sstable_blooms,
            sstable_indexes,
            max_lower_seq: AtomicU64::new(max_lower_seq_init.unwrap_or(0)),
            metrics: OnceLock::new(),
            displaced_sink: OnceLock::new(),
            config,
            compaction_in_progress,
            ro_memtable_version: Arc::new(AtomicU64::new(0)),
            level0_files,
            last_l0_stamp: parking_lot::Mutex::new(last_l0_stamp),
            level0_cleanup_lock: parking_lot::Mutex::new(()),
            flush_observer: Arc::new(RwLock::new(None)),
            base_path,
            #[cfg(test)]
            fail_next_insert: AtomicBool::new(false),
        })
    }

    /// A handle to `/dev/null`, which reads as empty: the Level-1 handle of a
    /// bucket whose merge left no entries.
    pub(super) fn create_null_file() -> Result<File> {
        OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open("/dev/null")
            .map_err(LSMError::Io)
    }

    /// Whether the tree at `base_path` holds any L1 data, without opening it:
    /// a non-empty `level1_{bucket}.dat`. L1 is where compaction drops
    /// tombstones, so it is what a full WAL replay could contradict.
    pub(crate) fn has_level1_data(base_path: &Path) -> std::io::Result<bool> {
        let dir = Self::level1_dir_from(base_path);
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(e) => return Err(e),
        };
        for entry in entries {
            let entry = entry?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("level1_") && name.ends_with(".dat") && entry.metadata()?.len() > 0 {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Level 1 directory path under base path
    pub(super) fn level1_dir_from(base_path: &Path) -> PathBuf {
        base_path.join("level1")
    }

    /// A bucket's Level-1 file: one fixed file per bucket, `level1_{bucket}.dat`.
    /// (Level-0 files are named by a per-flush stamp instead; see
    /// `write_level0_file_for_records`.)
    pub(super) fn level1_path_from(base_path: &Path, bucket: u32) -> PathBuf {
        Self::level1_dir_from(base_path).join(format!("level1_{}.dat", bucket))
    }

    /// Level 0 directory path under base path
    pub(super) fn level0_dir_from(base_path: &Path) -> PathBuf {
        base_path.join("level0")
    }

    /// Level 0 directory for a specific bucket
    pub(super) fn level0_bucket_dir_from(base_path: &Path, bucket: u32) -> PathBuf {
        Self::level0_dir_from(base_path).join(format!("level0_{}", bucket))
    }

    pub(super) fn level0_created_at_from_path(path: &Path) -> u128 {
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("0");
        let first = stem.split('_').next().unwrap_or("0");
        if let Ok(ts) = first.parse::<u128>() {
            return ts;
        }
        path.metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_millis())
            .unwrap_or(0)
    }

    /// Scan an on-disk Level-0 file at open (validating its header) and summarise
    /// it for the read path. `Ok(None)` when the file is empty or cannot be opened;
    /// a frame that fails its length or CRC check is an error, as in every reader.
    pub(super) fn level0_file_summary(path: &Path) -> Result<Option<L0Summary>> {
        let (meta, bloom, index, max_seq) = match Self::load_metadata_from_file(path) {
            Ok(loaded) => loaded,
            Err(LSMError::Io(_)) => return Ok(None),
            Err(e) => return Err(e),
        };
        Ok(bloom.zip(max_seq).map(|(bloom, max_seq)| L0Summary {
            min_key: meta.min_key,
            max_key: meta.max_key,
            max_seq,
            entries: meta.entry_count,
            bloom,
            index,
        }))
    }

    /// Load the on-disk Level-0 files into the in-memory registry and return the
    /// newest write sequence across all of them. The returned seq is folded into
    /// `max_lower_seq` so the read fast path's bound covers L0 too — L0 is newer
    /// than L1, so omitting it would leave the bound too low after a restart (a
    /// low-seq active entry could then wrongly short-circuit above a higher-seq
    /// L0 tombstone).
    pub(super) fn load_existing_level0_files(base_path: &Path, level0_files: &[Arc<RwLock<Vec<Arc<L0FileEntry>>>>]) -> Result<Option<u64>> {
        let mut max_seq: Option<u64> = None;
        for bucket in 0u32..level0_files.len() as u32 {
            let dir = Self::level0_bucket_dir_from(base_path, bucket);
            if !dir.exists() {
                continue;
            }
            let mut guard = level0_files[bucket as usize].write();
            for entry in std::fs::read_dir(&dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.file_name().is_some_and(|n| n.to_string_lossy().ends_with(LEVEL0_TMP_SUFFIX)) {
                    // A flush killed before its rename. It never reported itself
                    // written, so its WAL entries were never retired: replay
                    // restores its records.
                    warn!(
                        "removing unfinished Level-0 file {} (a flush interrupted before its rename)",
                        path.display()
                    );
                    std::fs::remove_file(&path)?;
                    continue;
                }
                if path.extension().map(|ext| ext == "dat").unwrap_or(false) {
                    // A `.dat` is renamed into place only once complete (above), so
                    // one too short to hold a header is damage, not a crash's
                    // leftover: fail the open rather than register it.
                    if std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0) < SSTABLE_DATA_START {
                        return Err(LSMError::Corruption(format!(
                            "Level-0 file {} is shorter than an SSTable header",
                            path.display()
                        )));
                    }
                    let summary = Self::level0_file_summary(&path)?;
                    if let Some(seq) = summary.as_ref().map(|s| s.max_seq)
                        && max_seq.is_none_or(|m| seq_newer(seq, m))
                    {
                        max_seq = Some(seq);
                    }
                    let created_at = Self::level0_created_at_from_path(&path);
                    guard.push(Arc::new(L0FileEntry::new(path, created_at, summary)));
                }
            }
        }
        Ok(max_seq)
    }
}
