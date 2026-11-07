//! The async entry point, [`AsyncDb`], and its background workers.

use super::*;

// ── AsyncDb ────────────────────────────────────────────────────────────

/// Async wrapper around [`Db`].
///
/// All blocking operations are offloaded via `tokio::task::spawn_blocking`.
#[derive(Clone)]
pub struct AsyncDb {
    pub(super) inner: Arc<Db>,
}

#[cfg(all(test, feature = "semantic-search"))]
impl AsyncDb {
    /// Test-only access to the internal coordinator. See
    /// [`Db::coordinator_for_test`].
    pub(crate) fn coordinator_for_test(&self) -> &Database {
        self.inner.coordinator_for_test()
    }
}

impl AsyncDb {
    /// Hold the doc store's directory lock until this database is dropped, not
    /// just the handle that opened it (see `Db::dir_lock`).
    #[cfg(feature = "doc-store")]
    pub(crate) fn hold_dir_lock(&self, lock: crate::support::file_lock::DirLock) {
        *self.inner.dir_lock.lock() = Some(lock);
    }

    // ── Open / Close ──────────────────────────────────────────────────

    /// Open a database with default config.
    pub async fn open<P: AsRef<Path> + Send + 'static>(path: P) -> Result<Self> {
        Self::open_with_config(path, DbConfig::default()).await
    }

    /// Open a database using a `minnal.toml` configuration file.
    pub async fn open_with_config_file<P: AsRef<Path> + Send + 'static>(config_path: P) -> Result<Self> {
        let config_path_buf = config_path.as_ref().to_path_buf();
        let db = tokio::task::spawn_blocking(move || Db::open_with_config_file(config_path_buf))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))??;
        Ok(Self { inner: Arc::new(db) })
    }

    /// Open a database with the given config.
    pub async fn open_with_config<P: AsRef<Path> + Send + 'static>(path: P, config: DbConfig) -> Result<Self> {
        let path_buf = path.as_ref().to_path_buf();
        let db = tokio::task::spawn_blocking(move || Db::open_with_config(path_buf, config))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))??;

        Ok(Self { inner: Arc::new(db) })
    }

    /// Close the database, flushing all data to disk.
    pub async fn shutdown(&self) -> Result<()> {
        self.stop_workers().await;

        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            // Flush index state on clean shutdown, now that no worker can race
            // with it. This is also what clears the no-WAL markers, so a clean
            // shutdown reports no gap on the next open (FR-001).
            if let Err(e) = db.inner.run_index_checkpoint() {
                log::warn!("[AsyncDb] Final index checkpoint failed: {:?}", e);
            }
            db.shutdown()
        })
        .await
        .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// Test-only: simulate a crash. Stops the background workers, then drops
    /// the database **without** the flushes `shutdown` and `Drop` would run,
    /// so unflushed memtables and no-WAL writes are lost exactly as in a
    /// process kill, while every file it had open is closed.
    ///
    /// Use this instead of `std::mem::forget(db)`, which also skips the flushes
    /// but leaks every file descriptor and leaves the workers running against a
    /// directory the test then reopens: a loop of forgotten databases runs a
    /// test process out of descriptors (the vector crash audit hit 1,024).
    ///
    /// `self` must be the last handle to the database; a clone kept elsewhere
    /// keeps it, and its files, alive.
    #[cfg(test)]
    pub(crate) async fn crash(self) {
        self.stop_workers().await;
        self.inner.inner.mark_crashed();
        debug_assert_eq!(Arc::strong_count(&self.inner), 1, "crash(): another handle still holds the database");
    }

    /// Stop every background worker (TTL, WAL GC, LSM compaction, value-log GC,
    /// index checkpoint), waiting for each to finish its current pass.
    async fn stop_workers(&self) {
        // Shutdown the global TTL worker first. The TTL config stays persisted in
        // the registry so it is restored on the next open.
        if let Some(worker) = self.inner.inner.ttl_worker.write().await.take() {
            info!("[AsyncDb] Shutting down TTL worker...");
            worker.shutdown().await;
        }

        // Shutdown WAL GC worker
        if self.inner.inner.wal_gc_worker.read().await.is_some() {
            info!("[AsyncDb] Shutting down WAL GC worker...");
            if let Some(w) = self.inner.inner.wal_gc_worker.write().await.take() {
                w.shutdown().await;
            }
        }

        // Shutdown LSM compaction worker
        if self.inner.inner.lsm_compaction_worker.read().await.is_some() {
            info!("[AsyncDb] Shutting down LSM compaction worker...");
            if let Some(w) = self.inner.inner.lsm_compaction_worker.write().await.take() {
                w.shutdown().await;
            }
            *self.inner.inner.lsm_compaction_sender.write() = None;
        }

        // Shutdown value-log GC worker
        if self.inner.inner.value_log_gc_worker.read().await.is_some() {
            info!("[AsyncDb] Shutting down value-log GC worker...");
            if let Some(w) = self.inner.inner.value_log_gc_worker.write().await.take() {
                w.shutdown().await;
            }
        }

        // Shutdown the index checkpoint worker. Stopped *before* the final
        // checkpoint below so the two cannot run concurrently, and so a tick
        // cannot land on an already-closed database.
        if self.inner.inner.index_checkpoint_worker.read().await.is_some() {
            info!("[AsyncDb] Shutting down index checkpoint worker...");
            if let Some(w) = self.inner.inner.index_checkpoint_worker.write().await.take() {
                w.shutdown().await;
            }
        }
    }

    /// Test-only: whether the index checkpoint worker is currently installed.
    ///
    /// Exists so a test can assert that [`shutdown`](Self::shutdown) actually
    /// stops it — a gap that went unnoticed for as long as a second, unused
    /// async wrapper was the only thing stopping the worker.
    #[cfg(test)]
    pub(crate) async fn index_checkpoint_worker_active(&self) -> bool {
        self.inner.inner.index_checkpoint_worker.read().await.is_some()
    }

    // ── Background worker management ──────────────────────────────────

    /// Start the WAL GC background worker.
    pub async fn enable_wal_gc_worker(&self, interval: Duration) -> Result<()> {
        let worker = WalGcWorker::new(Arc::clone(&self.inner), interval);
        *self.inner.inner.wal_gc_worker.write().await = Some(Arc::new(worker));
        info!("[AsyncDb] WAL GC worker enabled with {}ms interval", interval.as_millis());
        Ok(())
    }

    /// Start the index checkpoint background worker.
    pub async fn enable_index_checkpoint_worker(&self, interval: Duration) -> Result<()> {
        let worker = IndexCheckpointWorker::new(Arc::clone(&self.inner), interval);
        // Wire the write-path backpressure valve to this worker before publishing it.
        self.inner.inner.wire_index_checkpoint_trigger(&worker);
        *self.inner.inner.index_checkpoint_worker.write().await = Some(Arc::new(worker));
        // Milliseconds, not seconds: the default is 1750 ms, which `as_secs()`
        // truncated to a misleading "1s".
        info!("[AsyncDb] Index checkpoint worker enabled with {}ms interval", interval.as_millis());
        Ok(())
    }

    /// Start the LSM compaction background worker.
    ///
    /// Wires all existing KVStores' flush triggers to the new worker so that
    /// memtable flushes immediately schedule a compaction check.  Namespaces
    /// opened after this call are wired up at creation time.
    pub async fn enable_lsm_compaction_worker(&self, interval: Duration) -> Result<()> {
        let worker = LsmCompactionWorker::new(Arc::clone(&self.inner), interval);
        let sender = worker.sender();
        {
            let stores = self.inner.inner.stores.read();
            for kv_store in stores.values() {
                kv_store.set_compaction_trigger(sender.clone());
            }
        }
        *self.inner.inner.lsm_compaction_sender.write() = Some(sender);
        *self.inner.inner.lsm_compaction_worker.write().await = Some(Arc::new(worker));
        info!("[AsyncDb] LSM compaction worker enabled with {}ms interval", interval.as_millis());
        Ok(())
    }

    /// Start the value-log GC background worker.
    ///
    /// Runs GC on every namespace whose waste ratio exceeds `waste_threshold` on
    /// each `interval` tick.
    pub async fn enable_value_log_gc_worker(&self, interval: Duration, waste_threshold: f64) -> Result<()> {
        let worker = GCWorker::new(Arc::clone(&self.inner), interval, waste_threshold);
        *self.inner.inner.value_log_gc_worker.write().await = Some(Arc::new(worker));
        info!(
            "[AsyncDb] Value-log GC worker enabled with {}s interval, threshold {:.1}%",
            interval.as_secs(),
            waste_threshold
        );
        Ok(())
    }

    /// Ensure the single global TTL worker is running. Idempotent — a no-op if
    /// it has already been started. The worker scans every TTL-registered
    /// namespace on each `ttl_cleanup_interval` tick.
    pub(crate) async fn ensure_ttl_worker(&self) {
        let mut slot = self.inner.inner.ttl_worker.write().await;
        if slot.is_none() {
            let interval = self.inner.inner.config.scheduled_task_config.ttl_cleanup_interval;
            let worker = TtlWorker::new(Arc::clone(&self.inner), interval);
            *slot = Some(Arc::new(worker));
            info!("[AsyncDb] Global TTL worker enabled (interval={}s)", interval.as_secs());
        }
    }

    /// Start all background maintenance workers using intervals and thresholds
    /// from the given [`DbConfig`].
    ///
    /// This is the convenience call used by `DocStore::open_with_config` to
    /// ensure every long-running process has all workers running.
    pub async fn enable_all_workers(&self, config: &DbConfig) -> Result<()> {
        let st = config.scheduled_task_config;
        let threshold = config.threshold_config.value_log_waste_threshold;
        self.enable_wal_gc_worker(st.wal_gc_interval).await?;
        self.enable_lsm_compaction_worker(st.lsm_compaction_interval).await?;
        self.enable_value_log_gc_worker(st.value_log_gc_interval, threshold).await?;
        self.enable_index_checkpoint_worker(config.scheduled_task_config.index_checkpoint_interval())
            .await?;
        // Restore TTL: if any namespace has a persisted TTL config, start the
        // single global TTL worker so expiry resumes across restarts.
        if !self.inner.inner.registry.read().ttl_configs().is_empty() {
            self.ensure_ttl_worker().await;
        }
        Ok(())
    }

    /// Returns `true` if the database has been closed.
    pub fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }

    // ── CRUD ──────────────────────────────────────────────────────────

    /// Insert or update a key-value pair.
    ///
    /// Returns `Ok(())` once the write is durable. See [*Write durability*](Db#write-durability)
    /// for what that does and does not guarantee about immediate readability.
    pub async fn put(&self, key: Vec<u8>, value: Vec<u8>) -> Result<()> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || db.put(&key, &value))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    pub async fn get(&self, key: Vec<u8>) -> Result<Option<Vec<u8>>> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || db.get(&key))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// Delete a key.
    ///
    /// Returns `Ok(())` once the write is durable. See [*Write durability*](Db#write-durability)
    /// for what that does and does not guarantee about immediate readability.
    pub async fn delete(&self, key: Vec<u8>) -> Result<()> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || db.delete(&key))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// Atomically read-modify-write a key.
    ///
    /// See [`Db::merge`] for the full contract. `merge_fn` must be `Send + 'static`
    /// because it runs on a blocking worker thread, exactly as the rest of this
    /// surface takes owned buffers for the same reason.
    pub async fn merge<F>(&self, key: Vec<u8>, value: Vec<u8>, merge_fn: F) -> Result<Option<Vec<u8>>>
    where
        F: FnOnce(Option<&[u8]>, &[u8]) -> Result<Option<Vec<u8>>> + Send + 'static,
    {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || db.merge(&key, &value, merge_fn))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    // ── Iteration ─────────────────────────────────────────────────────

    pub async fn iter(&self) -> Result<Vec<KeyValue>> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || db.iter())
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    pub async fn keys(&self) -> Result<Vec<Vec<u8>>> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || db.keys())
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    pub async fn range(&self, start: Vec<u8>, end: Option<Vec<u8>>) -> Result<Vec<KeyValue>> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || db.range(&start, end.as_deref()))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    pub async fn scan_prefix(&self, prefix: Vec<u8>) -> Result<Vec<KeyValue>> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || db.scan_prefix(&prefix))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// Paginated cursor scan. See [`Db::scan`] for details.
    pub async fn scan(&self, cursor: Option<Vec<u8>>, end: Option<Vec<u8>>, limit: usize) -> Result<ScanPage> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || db.scan(cursor.as_deref(), end.as_deref(), limit))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    // ── Namespaces ────────────────────────────────────────────────────

    /// Get or create a namespace and return an async scoped handle.
    pub async fn namespace(&self, name: String) -> Result<AsyncNamespace> {
        let db = self.inner.clone();
        let name_clone = name.clone();
        let (ns_id, store) = tokio::task::spawn_blocking(move || -> Result<(u32, Arc<KVStore>)> {
            let ns_id = match db.inner.get_namespace_id(&name_clone) {
                Some(id) => id,
                None => db.inner.create_namespace(&name_clone)?,
            };
            let store = db.inner.get_store_by_name(&name_clone)?;
            Ok((ns_id, store))
        })
        .await
        .map_err(|e| KVError::Io(std::io::Error::other(e)))??;

        Ok(AsyncNamespace {
            ns_id,
            store,
            db: self.inner.clone(),
        })
    }

    /// Async [`Db::existing_namespace`].
    ///
    /// Answered on the calling task: the lookup is two in-memory reads. A trip to
    /// the blocking pool and back cost more than the read that followed it (half
    /// of a slow document `get` under load). Only a namespace whose store is not
    /// in the store map yet takes the blocking path, where
    /// [`Db::existing_namespace`] waits on the namespace-creation lock.
    pub async fn existing_namespace(&self, name: String) -> Result<Option<AsyncNamespace>> {
        let found = match self.inner.inner.lookup_open_store(&name) {
            OpenStoreLookup::Open(ns_id, store) => Some((ns_id, store)),
            OpenStoreLookup::Absent => None,
            OpenStoreLookup::NotReady => {
                let db = self.inner.clone();
                tokio::task::spawn_blocking(move || db.existing_namespace(&name).map(|ns| (ns.ns_id, Arc::clone(&ns.store))))
                    .await
                    .map_err(|e| KVError::Io(std::io::Error::other(e)))?
            }
        };
        Ok(found.map(|(ns_id, store)| AsyncNamespace {
            ns_id,
            store,
            db: self.inner.clone(),
        }))
    }

    /// Get or create a namespace with a TTL and return an async scoped handle.
    ///
    /// Records in this namespace will be automatically expired after `ttl` by the
    /// single global TTL worker, capped at `max_deletes_per_run` deletions per
    /// pass. The first TTL namespace starts that worker; subsequent ones just
    /// register with it.
    pub async fn namespace_with_ttl(&self, name: String, ttl: Duration, max_deletes_per_run: usize) -> Result<AsyncNamespace> {
        let db = self.inner.clone();
        let name_clone = name.clone();
        let (ns_id, store) = tokio::task::spawn_blocking(move || -> Result<(u32, Arc<KVStore>)> {
            let ns_id = match db.inner.get_namespace_id(&name_clone) {
                Some(id) => id,
                None => db.inner.create_namespace_with_ttl(&name_clone, Some(ttl))?,
            };
            let store = db.inner.get_store_by_name(&name_clone)?;
            Ok((ns_id, store))
        })
        .await
        .map_err(|e| KVError::Io(std::io::Error::other(e)))??;

        // `namespace_with_ttl` is get-or-create, so re-persist the config, (re)start
        // the worker, and log only when the config is actually new or changed.
        // Otherwise a plain "get" of an already-registered TTL namespace — which the
        // admin REST endpoints do on every request — would rewrite the registry file
        // and emit a spurious "TTL registered" line on each access. The persisted
        // config (restored on the next open) is the source of truth the worker reads.
        let already_registered = self.inner.inner.registry.read().ttl_config(ns_id) == Some((ttl, max_deletes_per_run));
        if !already_registered {
            self.inner.inner.registry.write().set_ttl_config(ns_id, ttl, max_deletes_per_run)?;
            self.ensure_ttl_worker().await;
            info!(
                "[AsyncDb] TTL registered for namespace '{}' (ttl={}s, max_deletes={})",
                name,
                ttl.as_secs(),
                max_deletes_per_run
            );
        }

        Ok(AsyncNamespace {
            ns_id,
            store,
            db: self.inner.clone(),
        })
    }

    pub fn list_namespaces(&self) -> Vec<(String, u32)> {
        self.inner.list_namespaces()
    }

    /// Return the `(ttl_secs, max_deletes_per_run)` config for a namespace, or
    /// `None` if no TTL is registered for it.
    pub fn ttl_config_for_ns(&self, ns_id: u32) -> Option<(u64, usize)> {
        self.inner
            .inner
            .registry
            .read()
            .ttl_config(ns_id)
            .map(|(ttl, max_del)| (ttl.as_secs(), max_del))
    }

    pub async fn remove_namespace(&self, name: String) -> Result<u32> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || db.remove_namespace(&name))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// Async [`Db::remove_unregistered_namespace_dir`].
    pub async fn remove_unregistered_namespace_dir(&self, name: String) -> Result<()> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || db.remove_unregistered_namespace_dir(&name))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    // ── Maintenance ───────────────────────────────────────────────────

    pub async fn garbage_collect(&self) -> Result<GCStats> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || db.garbage_collect())
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// Run value-log GC on every namespace and return per-namespace results.
    pub async fn garbage_collect_all(&self) -> Vec<(String, GCStats)> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || db.garbage_collect_all()).await.unwrap_or_default()
    }

    /// Run WAL garbage collection (reclaims segments wholly below the cut).
    ///
    /// Returns `(bytes_reclaimed, pending_bytes)`: the bytes of segment files
    /// deleted by this pass, and the bytes of WAL recovery would replay
    /// afterwards (from the cut to the tail). If another pass is already
    /// running, nothing is reclaimed and the result is `(0, pending_bytes)`.
    pub async fn garbage_collect_wal(&self) -> Result<(u64, u64)> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || db.garbage_collect_wal())
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    pub async fn compact(&self) -> Result<()> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || db.compact())
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// Async [`Db::flush_namespaces`].
    #[cfg(any(test, feature = "semantic-search"))]
    pub(crate) async fn flush_namespaces(&self, names: Vec<String>) -> Result<()> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || db.flush_namespaces(&names))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// Async [`Db::sync_vector_index_wal`].
    #[cfg(feature = "semantic-search")]
    pub(crate) async fn sync_vector_index_wal(&self) -> Result<()> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || db.sync_vector_index_wal())
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// Force an index checkpoint across all namespaces: flush each namespace's
    /// dense row map and all active field indexes to disk, compacting any
    /// field-index bitmap store whose waste exceeds the configured
    /// `index_blob_waste_threshold`. Returns the number of active field indexes
    /// checkpointed.
    ///
    /// This runs the same pass as the periodic `IndexCheckpointWorker` and the
    /// one performed on clean shutdown; exposed for callers that want to force a
    /// flush + compaction on demand. Runs the blocking checkpoint on the
    /// blocking thread pool.
    pub async fn checkpoint_index(&self) -> Result<usize> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || db.checkpoint_index())
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    pub fn stats(&self) -> Stats {
        self.inner.stats()
    }

    /// Snapshot of engine-wide operational metrics (runtime counters).
    pub fn ops_metrics(&self) -> crate::support::metrics::MetricsSnapshot {
        self.inner.ops_metrics()
    }

    /// Field-index write buffers: bytes held now and at peak, the soft and
    /// hard limits, and how often each fired.
    pub fn index_overlay_stats(&self) -> crate::IndexOverlayStats {
        self.inner.index_overlay_stats()
    }

    /// Operational metrics for a single namespace, by name.
    pub fn ops_metrics_for(&self, namespace: &str) -> Result<crate::support::metrics::MetricsSnapshot> {
        self.inner.ops_metrics_for(namespace)
    }

    /// Per-namespace operational metrics for every live namespace, keyed by name.
    pub fn ops_metrics_by_namespace(&self) -> Vec<(String, crate::support::metrics::MetricsSnapshot)> {
        self.inner.ops_metrics_by_namespace()
    }

    /// Returns a snapshot of the current WAL metadata.
    pub fn wal_metadata(&self) -> crate::db::wal::WalMetadata {
        self.inner.wal_metadata()
    }

    /// Returns the main WAL's state: bounds, cut, per-namespace progress and
    /// each segment from the head to the tail.
    pub fn wal_stats(&self) -> crate::WalStats {
        self.inner.wal_stats()
    }

    /// Returns the vector index WAL's state: segments, tail, retired cut and
    /// the namespaces holding the cut back.
    pub fn vector_index_wal_stats(&self) -> Result<crate::VectorIndexWalStats> {
        self.inner.vector_index_wal_stats()
    }

    /// Returns a live LSM manifest snapshot for every active namespace.
    pub fn lsm_manifests(&self) -> Vec<(String, crate::storage::lsm::manifest::LsmManifest)> {
        self.inner.lsm_manifests()
    }

    /// Returns the in-memory (non-SSTable) LSM stats for every active namespace.
    pub fn lsm_runtime_stats(&self) -> Vec<(String, crate::storage::lsm::tree::LSMStats)> {
        self.inner.lsm_runtime_stats()
    }

    /// Returns per-bucket value-log metadata for every active namespace.
    pub fn value_log_shard_stats(&self) -> Vec<(String, Vec<(u32, crate::storage::value_log::ValueLogMetadata)>)> {
        self.inner.value_log_shard_stats()
    }

    /// Physical (on-disk) vs logical value-log footprint per shard, per namespace.
    pub fn value_log_physical_stats(&self) -> Vec<(String, Vec<crate::storage::value_log::sharded::ShardPhysicalStats>)> {
        self.inner.value_log_physical_stats()
    }

    /// Per-page value-log garbage breakdown for one namespace (by name).
    pub fn value_log_segment_stats(&self, namespace: &str) -> Result<Vec<(u32, Vec<crate::storage::value_log::SegmentStats>)>> {
        self.inner.value_log_segment_stats(namespace)
    }

    pub fn waste_ratio(&self) -> f64 {
        self.inner.waste_ratio()
    }

    // ── Typed CRUD (rkyv ser/de) ──────────────────────────────────────

    /// Insert a typed key-value pair, serialized via rkyv.
    ///
    /// Returns `Ok(())` once the write is durable. See [*Write durability*](Db#write-durability)
    /// for what that does and does not guarantee about immediate readability.
    pub async fn put_typed<K, V>(&self, key: &K, value: &V) -> Result<()>
    where
        K: for<'a> rkyv::Serialize<HighSerializer<rkyv::util::AlignedVec, ArenaHandle<'a>, RkyvError>>,
        V: for<'a> rkyv::Serialize<HighSerializer<rkyv::util::AlignedVec, ArenaHandle<'a>, RkyvError>>,
    {
        let kb = rkyv_serialize(key)?;
        let vb = rkyv_serialize(value)?;
        self.put(kb, vb).await
    }

    /// Look up a typed value by typed key.
    pub async fn get_typed<K, V>(&self, key: &K) -> Result<Option<V>>
    where
        K: for<'a> rkyv::Serialize<HighSerializer<rkyv::util::AlignedVec, ArenaHandle<'a>, RkyvError>>,
        V: rkyv::Archive,
        V::Archived: for<'a> bytecheck::CheckBytes<HighValidator<'a, RkyvError>> + rkyv::Deserialize<V, Strategy<rkyv::de::Pool, RkyvError>>,
    {
        let kb = rkyv_serialize(key)?;
        match self.get(kb).await? {
            Some(vb) => Ok(Some(rkyv_deserialize::<V>(&vb)?)),
            None => Ok(None),
        }
    }

    /// Delete by typed key.
    ///
    /// Returns `Ok(())` once the write is durable. See [*Write durability*](Db#write-durability)
    /// for what that does and does not guarantee about immediate readability.
    pub async fn delete_typed<K>(&self, key: &K) -> Result<()>
    where
        K: for<'a> rkyv::Serialize<HighSerializer<rkyv::util::AlignedVec, ArenaHandle<'a>, RkyvError>>,
    {
        let kb = rkyv_serialize(key)?;
        self.delete(kb).await
    }

    /// Atomically merge a typed value, serialized via rkyv.
    ///
    /// The typed twin of [`Db::merge`]; see [`Db::merge_typed`] for an example.
    pub async fn merge_typed<K, V, F>(&self, key: &K, value: &V, merge_fn: F) -> Result<Option<V>>
    where
        K: for<'a> rkyv::Serialize<HighSerializer<rkyv::util::AlignedVec, ArenaHandle<'a>, RkyvError>>,
        V: rkyv::Archive + for<'a> rkyv::Serialize<HighSerializer<rkyv::util::AlignedVec, ArenaHandle<'a>, RkyvError>> + Send + 'static,
        V::Archived: for<'a> bytecheck::CheckBytes<HighValidator<'a, RkyvError>> + rkyv::Deserialize<V, Strategy<rkyv::de::Pool, RkyvError>>,
        F: FnOnce(Option<V>, V) -> Result<Option<V>> + Send + 'static,
    {
        let kb = rkyv_serialize(key)?;
        let vb = rkyv_serialize(value)?;
        // Unlike the sync twin, the new `V` cannot be carried out of the closure
        // in a borrowed slot: the closure crosses a `spawn_blocking` boundary and
        // so must be `'static`. Deserializing the returned bytes is the price of
        // that, and it is paid only when the merge actually wrote.
        let written = self
            .merge(kb, vb, move |existing, operand| {
                let current = match existing {
                    Some(bytes) => Some(rkyv_deserialize::<V>(bytes)?),
                    None => None,
                };
                let operand = rkyv_deserialize::<V>(operand)?;
                match merge_fn(current, operand)? {
                    Some(new) => Ok(Some(rkyv_serialize(&new)?)),
                    None => Ok(None),
                }
            })
            .await?;
        match written {
            Some(bytes) => Ok(Some(rkyv_deserialize::<V>(&bytes)?)),
            None => Ok(None),
        }
    }

    // ── Typed Iteration (rkyv ser/de) ─────────────────────────────────

    /// Iterate over all typed key-value pairs.
    pub async fn iter_typed<K, V>(&self) -> Result<Vec<(K, V)>>
    where
        K: rkyv::Archive,
        K::Archived: for<'a> bytecheck::CheckBytes<HighValidator<'a, RkyvError>> + rkyv::Deserialize<K, Strategy<rkyv::de::Pool, RkyvError>>,
        V: rkyv::Archive,
        V::Archived: for<'a> bytecheck::CheckBytes<HighValidator<'a, RkyvError>> + rkyv::Deserialize<V, Strategy<rkyv::de::Pool, RkyvError>>,
    {
        deserialize_pairs(self.iter().await?)
    }

    /// Iterate over all typed keys.
    pub async fn keys_typed<K>(&self) -> Result<Vec<K>>
    where
        K: rkyv::Archive,
        K::Archived: for<'a> bytecheck::CheckBytes<HighValidator<'a, RkyvError>> + rkyv::Deserialize<K, Strategy<rkyv::de::Pool, RkyvError>>,
    {
        deserialize_keys(self.keys().await?)
    }

    /// Iterate over typed key-value pairs in `[start, end)`.
    pub async fn range_typed<K, V>(&self, start: &K, end: Option<&K>) -> Result<Vec<(K, V)>>
    where
        K: for<'a> rkyv::Serialize<HighSerializer<rkyv::util::AlignedVec, ArenaHandle<'a>, RkyvError>> + rkyv::Archive,
        K::Archived: for<'a> bytecheck::CheckBytes<HighValidator<'a, RkyvError>> + rkyv::Deserialize<K, Strategy<rkyv::de::Pool, RkyvError>>,
        V: rkyv::Archive,
        V::Archived: for<'a> bytecheck::CheckBytes<HighValidator<'a, RkyvError>> + rkyv::Deserialize<V, Strategy<rkyv::de::Pool, RkyvError>>,
    {
        let start_bytes = rkyv_serialize(start)?;
        let end_bytes = end.map(|e| rkyv_serialize(e)).transpose()?;
        let raw = self.range(start_bytes, end_bytes).await?;
        deserialize_pairs(raw)
    }

    /// Iterate over typed key-value pairs whose keys start with `prefix`.
    pub async fn scan_prefix_typed<K, V>(&self, prefix: Vec<u8>) -> Result<Vec<(K, V)>>
    where
        K: rkyv::Archive,
        K::Archived: for<'a> bytecheck::CheckBytes<HighValidator<'a, RkyvError>> + rkyv::Deserialize<K, Strategy<rkyv::de::Pool, RkyvError>>,
        V: rkyv::Archive,
        V::Archived: for<'a> bytecheck::CheckBytes<HighValidator<'a, RkyvError>> + rkyv::Deserialize<V, Strategy<rkyv::de::Pool, RkyvError>>,
    {
        deserialize_pairs(self.scan_prefix(prefix).await?)
    }

    // ── Index ──────────────────────────────────────────────────────────

    /// Register an indexed field for a namespace and return its `FieldId`.
    pub fn register_index_field(&self, namespace_id: u32, field_name: &str, value_type: IndexValueType) -> Result<FieldId> {
        self.inner.inner.register_index_field(namespace_id, field_name, value_type)
    }

    /// Activate a registered field index, loading its snapshot and replaying
    /// any WAL entries since the last checkpoint.
    ///
    /// After this call every `put` / `delete` on the namespace automatically
    /// keeps the index current.
    pub async fn activate_field_index(&self, namespace_id: u32, field_id: FieldId, value_type: IndexValueType, extractor: ExtractorFn) -> Result<()> {
        let db = Arc::clone(&self.inner);
        tokio::task::spawn_blocking(move || db.inner.activate_field_index(namespace_id, field_id, value_type, extractor))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// [`activate_field_index`](Self::activate_field_index) for a new index that
    /// a build is about to fill: it starts empty, with no WAL replay, and is safe
    /// on a store taking writes as long as the build then indexes every key.
    pub async fn activate_field_index_for_build(
        &self,
        namespace_id: u32,
        field_id: FieldId,
        value_type: IndexValueType,
        extractor: ExtractorFn,
    ) -> Result<()> {
        let db = Arc::clone(&self.inner);
        tokio::task::spawn_blocking(move || db.inner.activate_field_index_for_build(namespace_id, field_id, value_type, extractor))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// Remove a field index from the in-memory registry.
    ///
    /// After this call predicate queries that reference the field return an
    /// `InactiveField` error.  The on-disk checkpoint files are not touched.
    pub fn deactivate_field_index(&self, namespace_id: u32, field_id: FieldId) -> Result<()> {
        self.inner.inner.deactivate_field_index(namespace_id, field_id)
    }

    /// Permanently drop a field index: persist the drop, deregister it, and
    /// delete its on-disk directory.
    ///
    /// See [`Db::drop_field_index`] for the semantics and the ordering
    /// guarantee. Synchronous — the work is a schema write plus a directory
    /// removal, not a scan.
    pub fn drop_field_index(&self, namespace_id: u32, field_id: FieldId) -> Result<()> {
        self.inner.inner.drop_field_index(namespace_id, field_id)
    }

    /// See [`Db::reset_rowmap`]. Synchronous — a marker removal, a directory
    /// fsync and three small file truncations.
    pub fn reset_rowmap(&self, namespace_id: u32) -> Result<bool> {
        self.inner.inner.reset_rowmap(namespace_id)
    }

    /// See [`Db::rowmap_ids_allocated`].
    pub fn rowmap_ids_allocated(&self, namespace_id: u32) -> Result<Option<u64>> {
        self.inner.inner.rowmap_ids_allocated(namespace_id)
    }

    /// Register a custom row-ID function (and optionally its inverse) for a namespace.
    ///
    /// Call this before `activate_field_index` so WAL replay uses consistent row IDs.
    /// Providing `row_to_key_fn` enables O(|hits|) query resolution with zero memory overhead.
    ///
    /// See [`Database::set_row_id_fn`][crate::db::database::Database::set_row_id_fn]
    /// for full documentation.
    pub async fn set_row_id_fn(
        &self,
        namespace_id: u32,
        row_id_fn: crate::kv_store::index_set::RowIdFn,
        row_to_key_fn: Option<crate::kv_store::index_set::RowToKeyFn>,
    ) -> Result<()> {
        let db = Arc::clone(&self.inner);
        tokio::task::spawn_blocking(move || db.inner.set_row_id_fn(namespace_id, row_id_fn, row_to_key_fn))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// Return all indexed fields registered for a namespace, sorted by [`FieldId`].
    ///
    /// On a fresh open the list is populated from `config.json` automatically,
    /// so callers do not need to re-call `register_index_field` after a restart.
    pub fn list_index_fields(&self, namespace_id: u32) -> Vec<crate::field_index::FieldMeta> {
        self.inner.inner.list_index_fields(namespace_id)
    }

    /// Return the number of distinct indexed values for a field, or `None` if
    /// the field is not currently active.
    pub fn field_index_distinct_count(&self, namespace_id: u32, field_id: FieldId) -> Option<usize> {
        self.inner.inner.field_index_distinct_count(namespace_id, field_id)
    }

    /// Reclaimable dead-space ratios `(bitmap_waste, keymap_waste)` for a field,
    /// or `None` if the field is not currently active.
    pub fn field_index_waste(&self, namespace_id: u32, field_id: FieldId) -> Option<(f64, f64)> {
        self.inner.inner.field_index_waste(namespace_id, field_id)
    }

    /// On-disk blob growth/waste metrics for a field — bitmap and keymap store
    /// sizes (logical vs. live bytes) and waste ratios — or `None` if the field
    /// is not currently active. Use it to monitor the append-only write
    /// amplification that low-cardinality fields suffer.
    pub fn field_index_blob_stats(&self, namespace_id: u32, field_id: FieldId) -> Option<crate::field_index::IndexBlobStats> {
        self.inner.inner.field_index_blob_stats(namespace_id, field_id)
    }

    /// Reindex a single field for a single key, re-deriving its value from the
    /// key's current stored bytes using the same logic as the put path. Touches
    /// only the named field. See [`crate::FieldReindexOutcome`].
    /// Report the health of every registered field index in a namespace.
    pub async fn index_health(&self, namespace_id: u32) -> Result<Vec<FieldIndexHealth>> {
        let db = Arc::clone(&self.inner);
        tokio::task::spawn_blocking(move || db.inner.index_health(namespace_id))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// Repair a degraded field index and clear its gap record.
    ///
    /// Offloaded to a blocking task: a full rebuild scans the namespace, so this
    /// must not run on the async runtime's worker threads.
    pub async fn repair_field_index(&self, namespace_id: u32, field_id: FieldId) -> Result<FieldRepairOutcome> {
        let db = Arc::clone(&self.inner);
        tokio::task::spawn_blocking(move || db.inner.repair_field_index(namespace_id, field_id))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    pub async fn reindex_field(&self, namespace_id: u32, field_id: FieldId, key: Vec<u8>) -> Result<FieldReindexOutcome> {
        let db = Arc::clone(&self.inner);
        tokio::task::spawn_blocking(move || db.inner.reindex_field(namespace_id, field_id, &key))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// The configured field-index compaction threshold as a fraction (`0.0..1.0`).
    pub fn index_blob_waste_threshold(&self) -> f64 {
        self.inner.inner.index_blob_waste_threshold()
    }

    /// Evaluate a query string against the active field indices of a namespace
    /// and return the raw keys of all matching documents.
    pub async fn query_index(&self, namespace_id: u32, query: impl Into<String> + Send + 'static) -> Result<QueryOutcome> {
        let db = Arc::clone(&self.inner);
        let q = query.into();
        tokio::task::spawn_blocking(move || db.inner.query_keys(namespace_id, &q))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// Like [`query_index`] but returns only the `[offset, offset+limit)` window
    /// of matching keys together with the full match count.
    ///
    /// [`query_index`]: AsyncDb::query_index
    pub async fn query_index_paginated(
        &self,
        namespace_id: u32,
        query: impl Into<String> + Send + 'static,
        offset: usize,
        limit: usize,
    ) -> Result<QueryOutcome> {
        let db = Arc::clone(&self.inner);
        let q = query.into();
        tokio::task::spawn_blocking(move || db.inner.query_keys_paginated(namespace_id, &q, offset, limit))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }
}
