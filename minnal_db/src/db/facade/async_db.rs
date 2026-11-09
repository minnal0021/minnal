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

    /// Open a database with default config, with every background worker
    /// started (see [`open_with_config`](Self::open_with_config)).
    pub async fn open<P: AsRef<Path> + Send + 'static>(path: P) -> Result<Self> {
        Self::open_with_config(path, DbConfig::default()).await
    }

    /// Open a database using a `minnal.toml` configuration file, with every
    /// background worker started (see [`open_with_config`](Self::open_with_config)).
    pub async fn open_with_config_file<P: AsRef<Path> + Send + 'static>(config_path: P) -> Result<Self> {
        let toml_config = crate::db::config::toml::MinnalTomlConfig::from_file(config_path.as_ref())?;
        Self::open_with_config(toml_config.db_path(), toml_config.to_db_config()).await
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
    /// How many index checkpoints have started so far; see
    /// [`index_checkpoint_covers`](Self::index_checkpoint_covers).
    pub fn index_checkpoints_started(&self) -> u64 {
        self.inner.inner.index_checkpoints_started()
    }

    /// Whether an index checkpoint that started after `started` checkpoints had
    /// has finished, so index changes made before `started` was read are on disk.
    /// A field build's changes are not in the WAL: only this makes them durable.
    pub fn index_checkpoint_covers(&self, started: u64) -> bool {
        self.inner.inner.index_checkpoint_covers(started)
    }

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
    //
    // On the default namespace: each forwards to its twin on that namespace's
    // handle, which holds the one implementation.

    async fn default_namespace(&self) -> Result<AsyncNamespace> {
        self.namespace(crate::namespace::DEFAULT_NAMESPACE.to_owned()).await
    }

    /// Insert a typed key-value pair, serialized via rkyv.
    ///
    /// Returns `Ok(())` once the write is durable. See [*Write durability*](AsyncDb#write-durability)
    /// for what that does and does not guarantee about immediate readability.
    pub async fn put_typed<K, V>(&self, key: &K, value: &V) -> Result<()>
    where
        K: RkyvEncode,
        V: RkyvEncode,
    {
        self.default_namespace().await?.put_typed(key, value).await
    }

    /// Look up a typed value by typed key.
    pub async fn get_typed<K, V>(&self, key: &K) -> Result<Option<V>>
    where
        K: RkyvEncode,
        V: RkyvDecode,
    {
        self.default_namespace().await?.get_typed(key).await
    }

    /// Delete by typed key.
    ///
    /// Returns `Ok(())` once the write is durable. See [*Write durability*](AsyncDb#write-durability)
    /// for what that does and does not guarantee about immediate readability.
    pub async fn delete_typed<K>(&self, key: &K) -> Result<()>
    where
        K: RkyvEncode,
    {
        self.default_namespace().await?.delete_typed(key).await
    }

    /// Atomically merge a typed value, serialized via rkyv.
    ///
    /// The typed twin of [`merge`](Self::merge): `merge_fn` sees the stored
    /// value and the operand already deserialized, and returns the new value
    /// (or `None` to delete the key). Every caveat on [`merge`](Self::merge)
    /// applies — in particular the closure runs under a lock and must not call
    /// back into the database.
    ///
    /// ```rust,no_run
    /// # async fn run() -> Result<(), minnal_db::KVError> {
    /// # let db = minnal_db::AsyncDb::open("/tmp/merge_typed_doc").await?;
    /// let total: Option<u64> = db
    ///     .merge_typed(&"visits".to_string(), &1u64, |current, step| Ok(Some(current.unwrap_or(0) + step)))
    ///     .await?;
    /// # let _ = total;
    /// # db.shutdown().await?;
    /// # Ok(()) }
    /// ```
    pub async fn merge_typed<K, V, F>(&self, key: &K, value: &V, merge_fn: F) -> Result<Option<V>>
    where
        K: RkyvEncode,
        V: RkyvEncode + RkyvDecode + Send + 'static,
        F: FnOnce(Option<V>, V) -> Result<Option<V>> + Send + 'static,
    {
        self.default_namespace().await?.merge_typed(key, value, merge_fn).await
    }

    // ── Typed Iteration (rkyv ser/de) ─────────────────────────────────

    /// Iterate over all typed key-value pairs.
    pub async fn iter_typed<K, V>(&self) -> Result<Vec<(K, V)>>
    where
        K: RkyvDecode,
        V: RkyvDecode,
    {
        self.default_namespace().await?.iter_typed().await
    }

    /// Iterate over all typed keys.
    pub async fn keys_typed<K>(&self) -> Result<Vec<K>>
    where
        K: RkyvDecode,
    {
        self.default_namespace().await?.keys_typed().await
    }

    /// Iterate over typed key-value pairs in `[start, end)`.
    pub async fn range_typed<K, V>(&self, start: &K, end: Option<&K>) -> Result<Vec<(K, V)>>
    where
        K: RkyvEncode + RkyvDecode,
        V: RkyvDecode,
    {
        self.default_namespace().await?.range_typed(start, end).await
    }

    /// Iterate over typed key-value pairs whose keys start with `prefix`.
    pub async fn scan_prefix_typed<K, V>(&self, prefix: Vec<u8>) -> Result<Vec<(K, V)>>
    where
        K: RkyvDecode,
        V: RkyvDecode,
    {
        self.default_namespace().await?.scan_prefix_typed(prefix).await
    }

    // ── Index ──────────────────────────────────────────────────────────

    /// Register an indexed field for a namespace and return its `FieldId`.
    pub fn register_index_field(&self, namespace_id: NamespaceId, field_name: &str, value_type: IndexValueType) -> Result<FieldId> {
        self.inner.inner.register_index_field(namespace_id, field_name, value_type)
    }

    /// Activate a registered field index, loading its snapshot and replaying
    /// any WAL entries since the last checkpoint.
    ///
    /// After this call every `put` / `delete` on the namespace automatically
    /// keeps the index current.
    pub async fn activate_field_index(
        &self,
        namespace_id: NamespaceId,
        field_id: FieldId,
        value_type: IndexValueType,
        extractor: ExtractorFn,
    ) -> Result<()> {
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
        namespace_id: NamespaceId,
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
    pub fn deactivate_field_index(&self, namespace_id: NamespaceId, field_id: FieldId) -> Result<()> {
        self.inner.inner.deactivate_field_index(namespace_id, field_id)
    }

    /// Permanently drop a field index: persist the drop, deregister it, and
    /// delete its on-disk directory.
    ///
    /// Use this rather than [`deactivate_field_index`](Self::deactivate_field_index)
    /// when the index is going away for good — the latter is an in-memory
    /// deregister that leaves every file in place. The field keeps its
    /// [`FieldId`] for reuse, but must be re-registered before it can be
    /// activated again. Synchronous — the work is a schema write plus a
    /// directory removal, not a scan.
    pub fn drop_field_index(&self, namespace_id: NamespaceId, field_id: FieldId) -> Result<()> {
        self.inner.inner.drop_field_index(namespace_id, field_id)
    }

    /// Free every row ID of a namespace by replacing its dense row map with an
    /// empty one; the next field builds allocate fresh dense IDs for the live
    /// keys only. Every field index of the namespace must already be dropped
    /// ([`drop_field_index`](Self::drop_field_index)) — otherwise this returns
    /// an error and changes nothing. Returns `false` for a namespace whose row
    /// IDs come from a key-derived `RowIdFn` (nothing to reset). Synchronous —
    /// a marker removal, a directory fsync and three small file truncations.
    pub fn reset_rowmap(&self, namespace_id: NamespaceId) -> Result<bool> {
        self.inner.inner.reset_rowmap(namespace_id)
    }

    /// Row IDs a namespace's dense row map has allocated — every key ever
    /// indexed, live or deleted — or `None` when the namespace uses a
    /// key-derived `RowIdFn` or no field index was ever activated.
    pub fn rowmap_ids_allocated(&self, namespace_id: NamespaceId) -> Result<Option<u64>> {
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
        namespace_id: NamespaceId,
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
    pub fn list_index_fields(&self, namespace_id: NamespaceId) -> Vec<crate::field_index::FieldMeta> {
        self.inner.inner.list_index_fields(namespace_id)
    }

    /// Return the number of distinct indexed values for a field, or `None` if
    /// the field is not currently active.
    pub fn field_index_distinct_count(&self, namespace_id: NamespaceId, field_id: FieldId) -> Option<usize> {
        self.inner.inner.field_index_distinct_count(namespace_id, field_id)
    }

    /// Reclaimable dead-space ratios `(bitmap_waste, keymap_waste)` for a field,
    /// or `None` if the field is not currently active.
    pub fn field_index_waste(&self, namespace_id: NamespaceId, field_id: FieldId) -> Option<(f64, f64)> {
        self.inner.inner.field_index_waste(namespace_id, field_id)
    }

    /// On-disk blob growth/waste metrics for a field — bitmap and keymap store
    /// sizes (logical vs. live bytes) and waste ratios — or `None` if the field
    /// is not currently active. Use it to monitor the append-only write
    /// amplification that low-cardinality fields suffer.
    pub fn field_index_blob_stats(&self, namespace_id: NamespaceId, field_id: FieldId) -> Option<crate::field_index::IndexBlobStats> {
        self.inner.inner.field_index_blob_stats(namespace_id, field_id)
    }

    /// Reindex a single field for a single key, re-deriving its value from the
    /// key's current stored bytes using the same logic as the put path. Touches
    /// only the named field. See [`crate::FieldReindexOutcome`].
    /// Report the health of every registered field index in a namespace.
    pub async fn index_health(&self, namespace_id: NamespaceId) -> Result<Vec<FieldIndexHealth>> {
        let db = Arc::clone(&self.inner);
        tokio::task::spawn_blocking(move || db.inner.index_health(namespace_id))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// Repair a degraded field index and clear its gap record.
    ///
    /// Offloaded to a blocking task: a full rebuild scans the namespace, so this
    /// must not run on the async runtime's worker threads.
    pub async fn repair_field_index(&self, namespace_id: NamespaceId, field_id: FieldId) -> Result<FieldRepairOutcome> {
        let db = Arc::clone(&self.inner);
        tokio::task::spawn_blocking(move || db.inner.repair_field_index(namespace_id, field_id))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    pub async fn reindex_field(&self, namespace_id: NamespaceId, field_id: FieldId, key: Vec<u8>) -> Result<FieldReindexOutcome> {
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
    pub async fn query_index(&self, namespace_id: NamespaceId, query: impl Into<String> + Send + 'static) -> Result<QueryOutcome> {
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
        namespace_id: NamespaceId,
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
