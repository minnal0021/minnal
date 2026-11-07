//! Opening a database: the main WAL, the vector index WAL, and wiring the flush observers.

use super::*;

/// Open the main WAL in `{db_path}/wal` from its saved metadata, and work out
/// where the sequence counter starts.
///
/// The metadata says where reading starts (its cut) and the furthest position
/// its progress names; the log validates everything from the cut, drops a
/// torn tail, and starts a new segment if the progress names positions past
/// what survived (`SegmentedLog::open`). The head and the tail are taken from
/// the log, not the metadata.
///
/// **Missing or unreadable metadata.** Without it the cut is unknown, and the
/// whole log is replayed. That is safe only while no tombstone can have been
/// dropped, and compaction drops tombstones only into L1: so a database with
/// L1 data and WAL segments to replay fails the open instead (a replayed put
/// could otherwise bring back a key deleted since). With no L1 data — a fresh
/// database, or a crash during its first flush — the log is replayed whole.
/// Unreadable metadata is kept as `metadata.corrupt`.
///
/// The counter starts above the saved high-water mark and every entry at or
/// above the cut; entries below the cut are in SSTables, which raise it too
/// (`KVStore::set_seq_counter`).
pub(super) fn open_main_wal(db_path: &Path, segment_size: u32) -> Result<(Arc<Wal>, WalMetadata, PathBuf, u64)> {
    let dir = crate::db::layout::main_wal_dir(db_path);
    let metadata_path = dir.join(crate::db::wal::METADATA_FILE);
    let (saved, unreadable) = match std::fs::read(&metadata_path) {
        Ok(data) => match WalMetadata::from_file_bytes(&data) {
            Ok(m) => (Some(m), None),
            Err(e) => (None, Some(e)),
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => (None, None),
        Err(e) => return Err(e.into()),
    };
    if saved.is_none() && crate::db::wal::segmented_log::has_segments(&dir)? && any_level1_data(db_path)? {
        let state = match &unreadable {
            Some(e) => format!("unreadable ({e})"),
            None => "missing".to_owned(),
        };
        return Err(WalError::Log(LogError::Corruption(format!(
            "the WAL metadata ({}) is {state}, but the database holds L1 data: replaying the whole log could bring back \
             deleted keys. Restore the metadata file, or remove the `wal` directory to start the WAL empty (writes not \
             yet flushed to an SSTable are then lost)",
            metadata_path.display()
        )))
        .into());
    }
    let (wal, report) = Wal::open(
        &dir,
        segment_size,
        crate::db::wal::segmented_log::Progress::open_from(saved.as_ref().map(|m| m.progress()).as_ref()),
    )?;
    if let Some(e) = unreadable {
        warn!("[WAL] the WAL metadata was unreadable ({e}); the log was read from its start (no L1 data yet)");
        let _ = std::fs::rename(&metadata_path, metadata_path.with_extension("corrupt"));
    }
    let mut metadata = saved.unwrap_or_default();
    metadata.tail = report.tail;
    metadata.head = wal.head()?;
    let next_seq = metadata.last_sequence.max(report.max_sequence).saturating_add(1);
    Ok((Arc::new(wal), metadata, metadata_path, next_seq))
}

/// Whether any namespace directory under `db_path` holds L1 data.
fn any_level1_data(db_path: &Path) -> std::io::Result<bool> {
    for dir in crate::db::layout::namespace_data_dirs(db_path)? {
        if KVStore::has_level1_data(&dir)? {
            return Ok(true);
        }
    }
    Ok(false)
}

impl Database {
    /// Open a multi-namespace database at the given path
    pub fn open(db_path: &Path, mut config: DbConfig) -> Result<Self> {
        std::fs::create_dir_all(db_path)?;

        // Detect existing bucket count from the default namespace's value logs
        let default_vlog_dir = db_path.join("ns_default").join("value_logs");
        if default_vlog_dir.exists() {
            let existing_count = Self::detect_bucket_count(&default_vlog_dir);
            if existing_count > 0 && existing_count != config.num_buckets {
                warn!(
                    "[WARNING] Configured num_buckets={} but existing database at '{}' uses {} buckets. \
                     The configured value will NOT be applied. To change bucket count, \
                     database migration must be done externally.",
                    config.num_buckets,
                    db_path.display(),
                    existing_count,
                );
                config.num_buckets = existing_count;
            }
        }

        // Propagate top-level config values into lsm_config
        config.lsm_config.num_buckets = config.num_buckets;
        config.lsm_config.skip_list_capacity = config.skip_list_capacity;

        let (wal, wal_metadata, wal_metadata_path, next_seq_start) = open_main_wal(db_path, Wal::validate_segment_size(config.wal_segment_size)?)?;
        let next_seq = Arc::new(AtomicU64::new(next_seq_start));

        let wal_metadata = Arc::new(RwLock::new(wal_metadata));
        let pending_wal_flushes = Arc::new(RwLock::new(BTreeMap::new()));
        let wal_flush_observer = Arc::new(WalPersistObserver::new(
            Arc::clone(&wal),
            Arc::clone(&wal_metadata),
            wal_metadata_path.clone(),
            Arc::clone(&pending_wal_flushes),
            Arc::clone(&next_seq),
        ));

        // Open namespace registry
        let registry = NamespaceRegistry::open(db_path)?;

        // Open index manager (creates {db_path}/field-index/ if needed)
        let index_manager = IndexManager::open(db_path)?;

        // Open KVStores for all registered namespaces, restoring each one's
        // persisted TTL (if any) so `store.ttl` reflects the durable config.
        let mut stores = HashMap::new();
        for (name, ns_id) in registry.list() {
            let ns_path = crate::db::layout::namespace_data_dir(db_path, name);
            let ttl = registry.ttl_config(ns_id).map(|(ttl, _)| ttl);
            let kv_store = KVStore::open_with_ttl(
                ns_id,
                name,
                &ns_path,
                config.lsm_config.clone(),
                config.sync_config,
                config.segment_size_bytes,
                ttl,
            )?;
            kv_store.set_verify_checksums_on_read(config.verify_checksums_on_read);
            kv_store.cleanup_old_files_on_startup()?;
            stores.insert(ns_id, Arc::new(kv_store));
        }

        // Cleanup old WAL file
        let old_wal = db_path.join("wal.log.old");
        if old_wal.exists() {
            match std::fs::remove_file(&old_wal) {
                Ok(_) => info!("[STARTUP] Cleaned up old WAL file"),
                Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                    warn!("[STARTUP] Failed to cleanup old WAL file: {:?}", e);
                }
                _ => {}
            }
        }

        let fail_log_dir = config.fail_log_dir.clone().unwrap_or_else(|| db_path.join("fail_logs"));

        // The engine-wide counters, shared with the vector index WAL.
        let metrics = Arc::new(crate::support::metrics::Metrics::default());
        let vector_index_wal = Arc::new(Self::open_vector_index_wal(db_path, &registry, &metrics)?);

        // One write-buffer budget for every field index of this database.
        let index_overlay_budget = Arc::new(crate::field_index::IndexOverlayBudget::new(
            config.threshold_config.index_overlay_soft_bytes,
            config.threshold_config.index_overlay_hard_bytes,
        ));
        let db = Self {
            db_path: db_path.to_path_buf(),
            config,
            wal,
            wal_metadata,
            wal_flush_observer,
            pending_wal_flushes,
            wal_gc_in_progress: Arc::new(AtomicBool::new(false)),
            vector_index_wal,
            registry: RwLock::new(registry),
            namespace_create_lock: Mutex::new(()),
            index_activate_lock: Mutex::new(()),
            stores: RwLock::new(stores),
            closed: Arc::new(AtomicBool::new(false)),
            wal_gc_worker: Arc::new(tokio::sync::RwLock::new(None)),
            lsm_compaction_worker: Arc::new(tokio::sync::RwLock::new(None)),
            value_log_gc_worker: Arc::new(tokio::sync::RwLock::new(None)),
            lsm_compaction_sender: Arc::new(parking_lot::RwLock::new(None)),
            index_checkpoint_trigger: Arc::new(parking_lot::RwLock::new(None)),
            index_overlay_budget,
            rejected_index_updates: Arc::new(crate::db::indexing::manager::RejectedUpdateBuffer::default()),
            no_wal_pending: parking_lot::Mutex::new(std::collections::HashSet::new()),
            ttl_worker: Arc::new(tokio::sync::RwLock::new(None)),
            index_manager,
            index_checkpoint_worker: Arc::new(tokio::sync::RwLock::new(None)),
            next_seq,
            key_locks: KeyLocks::new(),
            fail_log_dir,
            metrics,
        };

        // Share the global WAL sequence counter with every store so that all
        // writes — WAL-backed, recovery replay, TTL expiry and bulk — draw from
        // one monotonic sequence space (required for highest-sequence-wins
        // conflict resolution in the memtable). Must happen before recovery.
        // Share the operational counters at the same time.
        {
            let stores = db.stores.read();
            for kv_store in stores.values() {
                kv_store.set_seq_counter(db.next_seq.clone());
                kv_store.set_metrics(std::sync::Arc::new(crate::support::metrics::Metrics::default()));
                kv_store.set_index_gap_sink(Some(
                    Arc::clone(&db.rejected_index_updates) as Arc<dyn crate::kv_store::index_set::IndexGapSink>
                ));
            }
        }

        // Finish any field-index drop a crash interrupted, before recovery can
        // replay into a directory that is meant to be gone.
        db.complete_interrupted_field_drops();

        // Report field indices left incomplete by no-WAL writes that an unclean
        // shutdown caught before any checkpoint covered them.
        db.record_no_wal_gaps_after_unclean_shutdown();

        // Recover from WAL
        db.recover_from_vector_index_wal()?;
        db.recover_from_wal()?;

        db.wire_up_flush_observers();

        Ok(db)
    }

    /// Open the vector index WAL in its directory, sharing the engine-wide
    /// counters. Records of namespaces no longer in the registry are released
    /// at once: replay skips them, so they must not hold the retired cut.
    pub(super) fn open_vector_index_wal(
        db_path: &Path,
        registry: &NamespaceRegistry,
        metrics: &Arc<crate::support::metrics::Metrics>,
    ) -> Result<VectorIndexWal> {
        let (wal, report) = VectorIndexWal::open(
            &crate::db::layout::vector_index_wal_dir(db_path),
            crate::db::wal::vector_index_wal::DEFAULT_SEGMENT_SIZE,
            Arc::clone(metrics),
        )?;
        if report.torn_bytes_dropped > 0 {
            warn!("[STARTUP] vector index WAL: dropped a torn tail of {} bytes", report.torn_bytes_dropped);
        }
        for ns in wal.tracked_namespaces() {
            if registry.get_name(ns).is_none() {
                wal.forget_namespace(ns);
            }
        }
        Ok(wal)
    }

    /// Detect an existing database's bucket count from its value-log filenames.
    ///
    /// Both the segment files (`value_log_{bucket}.seg000123`) and the per-bucket
    /// metadata (`value_log_{bucket}.metadata`) name their bucket right after the
    /// `value_log_` prefix. A bucket now owns **many** segment files, so a plain file
    /// count over-reports; instead take the highest bucket id seen and add one. Buckets
    /// are numbered `0..num_buckets`, so that is the original count — and it stays
    /// correct even if one bucket's files are momentarily absent.
    pub(super) fn detect_bucket_count(vlog_dir: &Path) -> usize {
        let Ok(entries) = std::fs::read_dir(vlog_dir) else {
            return 0;
        };
        let highest = entries
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name();
                let name = name.to_string_lossy();
                let rest = name.strip_prefix("value_log_")?;
                rest.split('.').next()?.parse::<u32>().ok()
            })
            .max();
        highest.map_or(0, |b| b as usize + 1)
    }

    /// Wire up LSM flush observers for all KVStores
    pub(super) fn wire_up_flush_observers(&self) {
        let stores = self.stores.read();
        for (ns_id, kv_store) in stores.iter() {
            let observer: Arc<dyn LsmFlushObserver> = Arc::new(LsmFlushObserverHub::new(
                *ns_id,
                Arc::clone(&self.wal_flush_observer),
                Arc::clone(&self.vector_index_wal),
                Arc::clone(&kv_store.lsm_compaction_trigger),
            ));
            kv_store.set_flush_observer(Some(observer));
        }
    }

    /// Build the index-checkpoint backpressure valve for `worker` (using the
    /// configured `index_blob_backpressure_bytes` cap), wire it into every
    /// existing store, and record it centrally so namespaces opened later inherit
    /// it. Called when the checkpoint worker is enabled.
    pub(crate) fn wire_index_checkpoint_trigger(&self, worker: &IndexCheckpointWorker) {
        let cap = self.config.threshold_config.index_blob_backpressure_bytes;
        let trigger = worker.backpressure_trigger(cap);
        {
            let stores = self.stores.read();
            for kv_store in stores.values() {
                kv_store.set_index_checkpoint_trigger(Some(Arc::clone(&trigger) as Arc<dyn crate::kv_store::index_set::CheckpointRequests>));
            }
        }
        *self.index_checkpoint_trigger.write() = Some(trigger);
    }
}
