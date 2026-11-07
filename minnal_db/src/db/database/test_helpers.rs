//! Test-only: simulated crashes and power loss.

use super::open::open_main_wal;
use super::*;

impl Database {
    /// Test-only: mark the database closed without flushing anything, so its
    /// `Drop` skips the flushes a clean close runs. See `AsyncDb::crash`.
    #[cfg(test)]
    pub(crate) fn mark_crashed(&self) {
        self.closed.store(true, std::sync::atomic::Ordering::SeqCst);
        // A crash test must see a missing `sync`: drop what a power loss would.
        self.vector_index_wal.lose_unsynced();
        // Each store flushes itself when dropped; a crash must not.
        for store in self.stores.read().values() {
            store.mark_crashed();
        }
    }

    /// Test-only: simulate a crash. Drops the database without the flushes a clean
    /// close runs, and closes every file. Use this, not `std::mem::forget`, which
    /// leaks each file until the test process exits. See `AsyncDb::crash`.
    #[cfg(test)]
    pub(crate) fn crash(self) {
        self.mark_crashed();
    }

    #[cfg(test)]
    pub(crate) fn open_with_wal_segment_size(db_path: &Path, mut config: DbConfig, wal_segment_size: u64) -> Result<Self> {
        std::fs::create_dir_all(db_path)?;

        let default_vlog_dir = db_path.join("ns_default").join("value_logs");
        if default_vlog_dir.exists() {
            let existing_count = Self::detect_bucket_count(&default_vlog_dir);
            if existing_count > 0 && existing_count != config.num_buckets {
                config.num_buckets = existing_count;
            }
        }

        config.lsm_config.num_buckets = config.num_buckets;
        config.lsm_config.skip_list_capacity = config.skip_list_capacity;

        let (wal, wal_metadata, wal_metadata_path, next_seq_start) =
            open_main_wal(db_path, u32::try_from(wal_segment_size).expect("a test segment size fits u32"))?;
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

        let registry = NamespaceRegistry::open(db_path)?;
        let index_manager = IndexManager::open(db_path)?;

        let mut stores = HashMap::new();
        for (name, ns_id) in registry.list() {
            let ns_path = crate::db::layout::namespace_data_dir(db_path, name);
            let kv_store = KVStore::open(
                ns_id,
                name,
                &ns_path,
                config.lsm_config.clone(),
                config.sync_config,
                config.segment_size_bytes,
            )?;
            kv_store.set_verify_checksums_on_read(config.verify_checksums_on_read);
            kv_store.cleanup_old_files_on_startup()?;
            stores.insert(ns_id, Arc::new(kv_store));
        }

        let old_wal = db_path.join("wal.log.old");
        if old_wal.exists() {
            let _ = std::fs::remove_file(&old_wal);
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

        // Share the global WAL sequence counter with every store (see the other
        // open path); must happen before recovery replays entries. Share the
        // operational counters too.
        {
            let stores = db.stores.read();
            for kv_store in stores.values() {
                kv_store.set_seq_counter(db.next_seq.clone());
                kv_store.set_metrics(std::sync::Arc::new(crate::support::metrics::Metrics::default()));
                kv_store.set_index_gap_sink(Some(
                    Arc::clone(&db.rejected_index_updates) as Arc<dyn crate::db::kv_store::index_set::IndexGapSink>
                ));
            }
        }

        db.complete_interrupted_field_drops();

        // Report field indices left incomplete by no-WAL writes that an unclean
        // shutdown caught before any checkpoint covered them.
        db.record_no_wal_gaps_after_unclean_shutdown();
        db.recover_from_vector_index_wal()?;
        db.recover_from_wal()?;
        db.wire_up_flush_observers();

        Ok(db)
    }

    /// Test hook: the key-stripe lock `key` takes in `namespace_id`.
    #[cfg(test)]
    pub(crate) fn stripe_of(&self, namespace_id: u32, key: &[u8]) -> usize {
        self.key_locks.stripe_of(namespace_id, key)
    }

    /// Test hook: copy this database's directory to `to` as a power loss
    /// would leave it — every log cut back to what was fsynced: the main WAL
    /// to its durable tail, each value-log bucket to its synced offset, the
    /// vector index WAL to its durable tail. Other files are written durably
    /// before anything depends on them (L0 files, metadata, markers), so they
    /// are copied whole.
    ///
    /// The caller must have stopped every writer and worker first
    /// (`Wal::freeze_commits` holds writers at the point a power loss would
    /// catch them), or the copy is not one moment.
    #[cfg(test)]
    pub(crate) fn power_loss_image(&self, to: &Path) -> std::io::Result<()> {
        let mut extents = self.wal.power_loss_extents();
        extents.extend(self.vector_index_wal.power_loss_extents());
        for store in self.stores.read().values() {
            for log in store.value_log.bucket_logs() {
                extents.push(log.power_loss_extent());
            }
        }
        fn copy_dir(from: &Path, to: &Path) -> std::io::Result<()> {
            std::fs::create_dir_all(to)?;
            for entry in std::fs::read_dir(from)? {
                let entry = entry?;
                let target = to.join(entry.file_name());
                if entry.file_type()?.is_dir() {
                    copy_dir(&entry.path(), &target)?;
                } else {
                    std::fs::copy(entry.path(), &target)?;
                }
            }
            Ok(())
        }
        copy_dir(&self.db_path, to)?;
        for (file, durable) in extents {
            let Ok(relative) = file.strip_prefix(&self.db_path) else { continue };
            let copy = to.join(relative);
            // Lose everything past what was fsynced, keeping the file's size:
            // a preallocated WAL segment then reads as zeros there, as it would
            // after the kernel lost those pages.
            if let Ok(f) = std::fs::OpenOptions::new().write(true).open(&copy) {
                let len = f.metadata()?.len();
                if len > durable {
                    f.set_len(durable)?;
                    f.set_len(len)?;
                }
            }
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn simulate_crash_with_wal_entries(&self, entries: Vec<WalEntry>) -> Result<()> {
        // The entries stand for acknowledged writes, so they are made durable as
        // a commit round would, then left unapplied, as a crash would leave them.
        let mut wal_metadata = self.wal_metadata.write();
        for entry in entries {
            let _ = self.wal.append_entry(&entry, &mut wal_metadata.tail)?;
        }
        let tail = wal_metadata.tail;
        drop(wal_metadata);
        self.wal.commit(tail, Duration::ZERO)?;
        self.flush_wal_metadata_internal()?;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn flush_all_namespaces(&self) -> Result<()> {
        let stores = self.stores.read();
        for kv_store in stores.values() {
            kv_store.flush_and_compact_all()?;
        }
        Ok(())
    }
}
