//! The async entry point, [`AsyncDb`], and its background workers.

use super::*;
use crate::namespace::NamespaceId;

// ── AsyncDb ────────────────────────────────────────────────────────────

/// The entry point for MinnalDB: an open database, with its background
/// workers running.
///
/// ```rust,no_run
/// # async fn run() -> minnal_db::error::Result<()> {
/// use minnal_db::AsyncDb;
///
/// let db = AsyncDb::open("/tmp/my_db").await?;
/// db.put(b"hello".to_vec(), b"world".to_vec()).await?;
/// assert_eq!(db.get(b"hello".to_vec()).await?, Some(b"world".to_vec()));
/// db.shutdown().await?;
/// # Ok(())
/// # }
/// ```
///
/// Calls that touch the disk run on tokio's blocking pool
/// (`tokio::task::spawn_blocking`), so they never hold up the runtime's
/// threads; that is why they take owned buffers.
///
/// # Write durability
///
/// This contract applies to every write entry point on `AsyncDb` and
/// [`AsyncNamespace`] — `put`, `delete`, [`merge`](AsyncDb::merge) and their
/// `_typed` variants.
///
/// A write returns `Ok(())` once it is **durable**: its write-ahead-log entry
/// has been fsynced to stable storage, so it survives power loss from that
/// moment on.
///
/// `Ok(())` does **not** guarantee the write is immediately readable. After the
/// WAL fsync the engine applies the operation to its in-memory structures, and
/// that step is best-effort: it retries a bounded number of times, and if it
/// still fails — a full disk, for instance — it logs at `ERROR`, increments the
/// `apply_failures` metric, and returns `Ok(())` regardless. The data is
/// already committed, so reporting an error would wrongly suggest it was lost.
///
/// In that rare window [`AsyncDb::get`] returns `None` for a key you successfully
/// wrote. It is self-healing but not immediately so: the entry is replayed from
/// the WAL on the next open, so state is correct again after a restart, and
/// nothing repairs it while the process keeps running. `apply_failures` on
/// [`AsyncDb::ops_metrics`] is how to detect it.
///
/// [`merge`](AsyncDb::merge) is the **one exception** to the paragraph above: it
/// returns [`KVError::MergeNotApplied`] rather than `Ok` when its own write does
/// not become readable, because a merge whose result is invisible poisons the
/// next merge on that key. It cannot detect the converse — a merge reading a key
/// whose *last write* is stuck in that window computes from the stale value and
/// has no way to know.
///
/// Every write is its own transaction, and returns only once a WAL fsync covers
/// it. Concurrent writes share fsyncs (group commit), so one writer's throughput
/// is bounded by the storage's fsync rate and concurrent writers go further.
/// There is no batch or multi-key transaction primitive. [`merge`](AsyncDb::merge) is a *single-key*
/// exception and not a transaction: it makes one key's read-modify-write
/// indivisible, and says nothing about any other key. The `_no_wal` variants
/// (e.g. [`AsyncNamespace::put_no_wal`]) skip the WAL entirely and give up this
/// durability guarantee in exchange for speed.
#[derive(Clone)]
pub struct AsyncDb {
    pub(super) inner: Arc<Db>,
}

#[cfg(test)]
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

    /// Open a database with default config, with every background worker
    /// started (see [`open_with_config`](Self::open_with_config)).
    pub async fn open<P: AsRef<Path> + Send + 'static>(path: P) -> Result<Self> {
        Self::open_with_config(path, DbConfig::default()).await
    }

    /// Open the database a config file describes (see
    /// [`ConfigFile`](crate::ConfigFile)), with every background worker
    /// started (see [`open_with_config`](Self::open_with_config)). The API
    /// server's own tables (`[api]`, `[logging]`) are allowed and ignored, so
    /// the server's file opens here too.
    pub async fn open_with_config_file<P: AsRef<Path> + Send + 'static>(config_path: P) -> Result<Self> {
        let (config, _) = crate::ConfigFile::from_file(config_path.as_ref(), crate::config_file::API_SERVER_TABLES)
            .map_err(|e| KVError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())))?;
        Self::open_with_config(config.db_path(), config.db_config()).await
    }

    /// Open a database with the given config, with every background worker
    /// started: WAL GC (the main WAL and the vector index WAL), value-log GC,
    /// LSM compaction, index checkpoints, and TTL expiry when a namespace has
    /// a TTL. Their intervals and thresholds come from `config`. Call
    /// [`shutdown`](Self::shutdown) to stop them and flush.
    pub async fn open_with_config<P: AsRef<Path> + Send + 'static>(path: P, config: DbConfig) -> Result<Self> {
        let db = Self::open_without_workers(path, config.clone()).await?;
        db.enable_all_workers(&config).await?;
        Ok(db)
    }

    /// Open a database and start no worker: for an owner that starts them
    /// itself once it is ready (the doc store activates its field indexes
    /// first, so the first index checkpoint sees them), and for tests that
    /// drive each step by hand.
    pub(crate) async fn open_without_workers<P: AsRef<Path> + Send + 'static>(path: P, config: DbConfig) -> Result<Self> {
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
            // shutdown reports no gap on the next open.
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

    /// Test-only: the background workers currently installed, by name.
    #[cfg(test)]
    pub(crate) async fn running_workers(&self) -> Vec<&'static str> {
        let db = &self.inner.inner;
        let mut running = Vec::new();
        if db.wal_gc_worker.read().await.is_some() {
            running.push("wal_gc");
        }
        if db.lsm_compaction_worker.read().await.is_some() {
            running.push("lsm_compaction");
        }
        if db.value_log_gc_worker.read().await.is_some() {
            running.push("value_log_gc");
        }
        if db.index_checkpoint_worker.read().await.is_some() {
            running.push("index_checkpoint");
        }
        running
    }

    // ── Background worker management ──────────────────────────────────

    /// Start the WAL GC background worker.
    pub(crate) async fn enable_wal_gc_worker(&self, interval: Duration) -> Result<()> {
        let worker = WalGcWorker::new(Arc::clone(&self.inner), interval);
        *self.inner.inner.wal_gc_worker.write().await = Some(Arc::new(worker));
        info!("[AsyncDb] WAL GC worker enabled with {}ms interval", interval.as_millis());
        Ok(())
    }

    /// Start the index checkpoint background worker.
    pub(crate) async fn enable_index_checkpoint_worker(&self, interval: Duration) -> Result<()> {
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
    pub(crate) async fn enable_lsm_compaction_worker(&self, interval: Duration) -> Result<()> {
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
    pub(crate) async fn enable_value_log_gc_worker(&self, interval: Duration, waste_threshold: f64) -> Result<()> {
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
    /// [`open_with_config`](Self::open_with_config) calls it; the doc store
    /// calls it once its field indexes are active. Call it once: each
    /// `enable_*` installs a new worker.
    pub(crate) async fn enable_all_workers(&self, config: &DbConfig) -> Result<()> {
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
    /// Returns `Ok(())` once the write is durable. See [*Write durability*](AsyncDb#write-durability)
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
    /// Returns `Ok(())` once the write is durable. See [*Write durability*](AsyncDb#write-durability)
    /// for what that does and does not guarantee about immediate readability.
    pub async fn delete(&self, key: Vec<u8>) -> Result<()> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || db.delete(&key))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// Read `key`, hand the stored value and `value` to `merge_fn`, and write
    /// the result back — atomically with respect to every other write to that key.
    ///
    /// This is the read-modify-write that `get` + `put` cannot give you: two
    /// callers that each `get`, compute, and `put` will silently lose one
    /// update, because nothing stops the second read from happening before the
    /// first write lands. `merge` closes that window.
    ///
    /// `merge_fn` receives `(existing, operand)` — `existing` is `None` when the
    /// key is absent, `operand` is the `value` passed in — and returns:
    ///
    /// | Returns | Effect |
    /// |---|---|
    /// | `Ok(Some(new))` | write `new` |
    /// | `Ok(None)` | delete the key (nothing is written if it was already absent) |
    /// | `Err(e)` | abort — nothing is written, and `e` is returned to the caller |
    ///
    /// The call returns the value that was written, or `None` for the delete and
    /// no-op cases. It can also fail *after* the closure ran — see *Durability*
    /// below for [`KVError::MergeNotApplied`].
    ///
    /// ```rust,no_run
    /// # async fn run() -> Result<(), minnal_db::KVError> {
    /// # let db = minnal_db::AsyncDb::open("/tmp/merge_doc").await?;
    /// // A counter that increments by the operand, starting from zero.
    /// let total = db
    ///     .merge(b"visits".to_vec(), 1u64.to_le_bytes().to_vec(), |existing, operand| {
    ///         let current = existing.map_or(0, |b| u64::from_le_bytes(b.try_into().unwrap()));
    ///         let step = u64::from_le_bytes(operand.try_into().unwrap());
    ///         Ok(Some((current + step).to_le_bytes().to_vec()))
    ///     })
    ///     .await?;
    /// # let _ = total;
    /// # db.shutdown().await?;
    /// # Ok(()) }
    /// ```
    ///
    /// # Durability
    ///
    /// A merge is one ordinary WAL-backed write of the *result*, so it carries
    /// the guarantee described in [*Write durability*](AsyncDb#write-durability) — with
    /// one deliberate difference. Where `put` returns `Ok` for a write that is
    /// durable but whose in-memory apply failed, `merge` returns
    /// [`KVError::MergeNotApplied`]: its result never became readable, so the
    /// next merge on this key would accumulate onto a stale base. The write is
    /// **not lost** — the WAL entry replays on the next open — so treat that
    /// error as "not readable yet", not "nothing happened".
    ///
    /// The converse is not detectable: if some earlier `put` to this key is
    /// itself stuck in that window, the closure is handed the stale value and
    /// nothing here can tell. `apply_failures` on [`AsyncDb::ops_metrics`] is the
    /// signal that it has happened at all.
    ///
    /// Nothing re-runs `merge_fn` during recovery: the WAL records the value the
    /// closure produced, not the closure.
    ///
    /// # `merge_fn` runs while a lock is held
    ///
    /// It must be quick, must not block, and must **not write back** into the
    /// database — a re-entrant write to this key, or to any key that happens to
    /// share its lock stripe, self-deadlocks. Reads are safe but pointless: a
    /// read of this key returns the same pre-merge value the closure was handed.
    ///
    /// `merge_fn` must be `Send + 'static` because it runs on a blocking
    /// worker thread, for the same reason the rest of this surface takes owned
    /// buffers.
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

    /// Paginated cursor scan over `[cursor, end)` on the default namespace.
    ///
    /// Returns up to `limit` key-value pairs starting from `cursor`
    /// (or the beginning if `None`) and ending before `end` (or the last key if
    /// `None`), plus an optional next cursor. Pass the returned cursor to the next
    /// call to fetch the next page.
    ///
    /// Each page fetch resolves only the page's values, not the whole result set.
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
        let (ns_id, store) = tokio::task::spawn_blocking(move || -> Result<(NamespaceId, Arc<KVStore>)> {
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

    /// The namespace named `name`, if it exists. Unlike
    /// [`namespace`](Self::namespace) it never creates one, so a caller racing
    /// the namespace's removal cannot bring it back.
    ///
    /// Answered on the calling task: the lookup is two in-memory reads. A trip to
    /// the blocking pool and back cost more than the read that followed it (half
    /// of a slow document `get` under load). Only a namespace whose store is not
    /// in the store map yet takes the blocking path, where
    /// the engine waits on the namespace-creation lock.
    pub async fn existing_namespace(&self, name: String) -> Result<Option<AsyncNamespace>> {
        let found = match self.inner.inner.lookup_open_store(&name) {
            OpenStoreLookup::Open(ns_id, store) => Some((ns_id, store)),
            OpenStoreLookup::Absent => None,
            OpenStoreLookup::NotReady => {
                let db = self.inner.clone();
                tokio::task::spawn_blocking(move || db.existing_namespace(&name))
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
        let (ns_id, store) = tokio::task::spawn_blocking(move || -> Result<(NamespaceId, Arc<KVStore>)> {
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
        // admin REST endpoints do on every request — would rewrite the catalog record
        // and emit a spurious "TTL registered" line on each access. The persisted
        // config (restored on the next open) is the source of truth the worker reads.
        let already_registered = self.inner.inner.registry.read().ttl_config(ns_id) == Some((ttl, max_deletes_per_run));
        if !already_registered {
            self.inner.inner.update_registry(|r| r.set_ttl_config(ns_id, ttl, max_deletes_per_run))?;
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

    pub fn list_namespaces(&self) -> Vec<(String, NamespaceId)> {
        self.inner.list_namespaces()
    }

    /// Return the `(ttl_secs, max_deletes_per_run)` config for a namespace, or
    /// `None` if no TTL is registered for it.
    pub fn ttl_config_for_ns(&self, ns_id: NamespaceId) -> Option<(u64, usize)> {
        self.inner
            .inner
            .registry
            .read()
            .ttl_config(ns_id)
            .map(|(ttl, max_del)| (ttl.as_secs(), max_del))
    }

    pub async fn remove_namespace(&self, name: String) -> Result<NamespaceId> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || db.remove_namespace(&name))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// Delete the data directory an interrupted drop left for `name`. Does nothing
    /// if `name` is a namespace: its directory holds live files.
    pub async fn remove_unregistered_namespace_dir(&self, name: String) -> Result<()> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || db.remove_unregistered_namespace_dir(&name))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }
}
