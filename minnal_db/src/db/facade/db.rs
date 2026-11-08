//! The synchronous entry point, [`Db`].

use super::*;
use crate::namespace::NamespaceId;

// ── Db (sync facade) ──────────────────────────────────────────────────

/// The primary entry point for MinnalDB.
///
/// `Db` delegates to the internal multi-namespace `Database` engine but
/// presents a minimal, ergonomic surface:
///
/// ```rust,no_run
/// use minnal_db::Db;
///
/// let db = Db::open("/tmp/my_db").unwrap();
/// db.put(b"hello", b"world").unwrap();
/// assert_eq!(db.get(b"hello").unwrap(), Some(b"world".to_vec()));
/// db.shutdown().unwrap();
/// ```
///
/// # Write durability
///
/// This contract applies to every write entry point on `Db`, [`Namespace`],
/// [`AsyncDb`] and [`AsyncNamespace`] — `put`, `delete`, [`merge`](Db::merge)
/// and their `_typed` variants.
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
/// In that rare window [`Db::get`] returns `None` for a key you successfully
/// wrote. It is self-healing but not immediately so: the entry is replayed from
/// the WAL on the next open, so state is correct again after a restart, and
/// nothing repairs it while the process keeps running. `apply_failures` on
/// [`Db::ops_metrics`] is how to detect it.
///
/// [`merge`](Db::merge) is the **one exception** to the paragraph above: it
/// returns [`KVError::MergeNotApplied`] rather than `Ok` when its own write does
/// not become readable, because a merge whose result is invisible poisons the
/// next merge on that key. It cannot detect the converse — a merge reading a key
/// whose *last write* is stuck in that window computes from the stale value and
/// has no way to know.
///
/// Every write is its own transaction, and returns only once a WAL fsync covers
/// it. Concurrent writes share fsyncs (group commit), so one writer's throughput
/// is bounded by the storage's fsync rate and concurrent writers go further.
/// There is no batch or multi-key transaction primitive. [`merge`](Db::merge) is a *single-key*
/// exception and not a transaction: it makes one key's read-modify-write
/// indivisible, and says nothing about any other key. The `_no_wal` variants
/// (e.g. [`AsyncNamespace::put_no_wal`]) skip the WAL entirely and give up this
/// durability guarantee in exchange for speed.
pub struct Db {
    pub(super) inner: Database,
    /// The doc store's directory lock. Fields drop in declaration order, so it
    /// is released only after `inner` has finished dropping (its final flush
    /// and compaction), however long a worker's pass keeps this `Db` alive past
    /// the last user handle. Keep it last.
    #[cfg(feature = "doc-store")]
    pub(super) dir_lock: parking_lot::Mutex<Option<crate::support::file_lock::DirLock>>,
}

// Gated on `semantic-search` as well as `test` because that is the only feature
// combination with a test that needs it; an unconditional `cfg(test)` is dead
// code under the other combinations, which clippy rejects at `-D warnings`.
#[cfg(all(test, feature = "semantic-search"))]
impl Db {
    /// Test-only access to the internal coordinator.
    ///
    /// Some durability behaviour is only observable as per-namespace engine
    /// state — the no-WAL flush flag, for one — which the facade deliberately
    /// does not expose. Tests in other modules of this crate need a way in
    /// without that becoming public API.
    pub(crate) fn coordinator_for_test(&self) -> &Database {
        &self.inner
    }
}

impl Db {
    // ── Open / Close ──────────────────────────────────────────────────

    /// Open a database at `path` with default configuration.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with_config(path, DbConfig::default())
    }

    /// Open a database using a `minnal.toml` configuration file.
    ///
    /// The config file specifies the database path and all tunables.
    /// ```rust,no_run
    /// use minnal_db::Db;
    /// let db = Db::open_with_config_file("/etc/minnal/minnal.toml").unwrap();
    /// ```
    pub fn open_with_config_file<P: AsRef<Path>>(config_path: P) -> Result<Self> {
        let toml_config = MinnalTomlConfig::from_file(config_path.as_ref())?;
        let db_path = toml_config.db_path();
        let config = toml_config.to_db_config();
        Self::open_with_config(db_path, config)
    }

    /// Open a database at `path` with the given configuration.
    pub fn open_with_config<P: AsRef<Path>>(path: P, config: DbConfig) -> Result<Self> {
        let db = Database::open(path.as_ref(), config)?;
        Ok(Self {
            inner: db,
            #[cfg(feature = "doc-store")]
            dir_lock: parking_lot::Mutex::new(None),
        })
    }

    /// Shut down the database, flushing all data to disk.
    pub fn shutdown(&self) -> Result<()> {
        self.inner.shutdown()
    }

    /// Run an index checkpoint now: flush every active field index to disk and
    /// compact any whose append-only bitmap value region has crossed the
    /// configured waste threshold (`ThresholdConfig::index_blob_waste_threshold`).
    ///
    /// This is the same pass the background `IndexCheckpointWorker` runs
    /// periodically and that `shutdown` runs once on close; exposed for callers
    /// that want to force a flush/compaction on demand. Returns the number of
    /// active field indices checkpointed.
    pub fn checkpoint_index(&self) -> Result<usize> {
        self.inner.run_index_checkpoint()
    }

    /// Returns `true` if the database has been closed.
    pub fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }

    // ── CRUD (default namespace) ──────────────────────────────────────

    /// Insert or update a key-value pair.
    ///
    /// Returns `Ok(())` once the write is durable. See [*Write durability*](Db#write-durability)
    /// for what that does and does not guarantee about immediate readability.
    pub fn put(&self, key: &[u8], value: &[u8]) -> Result<()> {
        self.inner.put(key, value)
    }

    /// Look up a value by key. Returns `None` if the key does not exist.
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.inner.get(key)
    }

    /// Delete a key.
    ///
    /// Returns `Ok(())` once the write is durable. See [*Write durability*](Db#write-durability)
    /// for what that does and does not guarantee about immediate readability.
    pub fn delete(&self, key: &[u8]) -> Result<()> {
        self.inner.delete(key)
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
    /// # use minnal_db::Db;
    /// # fn main() -> Result<(), minnal_db::KVError> {
    /// # let db = Db::open("/tmp/merge_doc")?;
    /// // A counter that increments by the operand, starting from zero.
    /// let total = db.merge(b"visits", &1u64.to_le_bytes(), |existing, operand| {
    ///     let current = existing.map_or(0, |b| u64::from_le_bytes(b.try_into().unwrap()));
    ///     let step = u64::from_le_bytes(operand.try_into().unwrap());
    ///     Ok(Some((current + step).to_le_bytes().to_vec()))
    /// })?;
    /// # let _ = total;
    /// # db.shutdown()?;
    /// # Ok(()) }
    /// ```
    ///
    /// # Durability
    ///
    /// A merge is one ordinary WAL-backed write of the *result*, so it carries
    /// the guarantee described in [*Write durability*](Db#write-durability) — with
    /// one deliberate difference. Where `put` returns `Ok` for a write that is
    /// durable but whose in-memory apply failed, `merge` returns
    /// [`KVError::MergeNotApplied`]: its result never became readable, so the
    /// next merge on this key would accumulate onto a stale base. The write is
    /// **not lost** — the WAL entry replays on the next open — so treat that
    /// error as "not readable yet", not "nothing happened".
    ///
    /// The converse is not detectable: if some earlier `put` to this key is
    /// itself stuck in that window, the closure is handed the stale value and
    /// nothing here can tell. `apply_failures` on [`Db::ops_metrics`] is the
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
    pub fn merge<F>(&self, key: &[u8], value: &[u8], merge_fn: F) -> Result<Option<Vec<u8>>>
    where
        F: FnOnce(Option<&[u8]>, &[u8]) -> Result<Option<Vec<u8>>>,
    {
        self.inner.merge(key, value, merge_fn)
    }

    // ── Iteration (default namespace) ─────────────────────────────────

    /// Iterate over all key-value pairs in key order.
    pub fn iter(&self) -> Result<Vec<KeyValue>> {
        let store = self.inner.default_store()?;
        store.scan_range_batch(&[], None)
    }

    /// Iterate over all keys (does not touch the value log).
    pub fn keys(&self) -> Result<Vec<Vec<u8>>> {
        let store = self.inner.default_store()?;
        store.keys()
    }

    /// Iterate over key-value pairs in `[start, end)`.
    /// Pass `None` for `end` to scan to the last key.
    pub fn range(&self, start: &[u8], end: Option<&[u8]>) -> Result<Vec<KeyValue>> {
        let store = self.inner.default_store()?;
        store.scan_range_batch(start, end)
    }

    /// Iterate over key-value pairs whose keys start with `prefix`.
    pub fn scan_prefix(&self, prefix: &[u8]) -> Result<Vec<KeyValue>> {
        let store = self.inner.default_store()?;
        store.scan_prefix_batch(prefix)
    }

    /// Paginated cursor scan over `[cursor, end)` on the default namespace.
    ///
    /// Returns up to `limit` key-value pairs starting from `cursor`
    /// (or the beginning if `None`) and ending before `end` (or the last key if
    /// `None`), plus an optional next cursor. Pass the returned cursor to the next
    /// call to fetch the next page.
    ///
    /// Each page fetch resolves only the page's values, not the whole result set.
    pub fn scan(&self, cursor: Option<&[u8]>, end: Option<&[u8]>, limit: usize) -> Result<ScanPage> {
        let store = self.inner.default_store()?;
        store.scan_page_batch(cursor, end, limit)
    }

    // ── Namespaces ────────────────────────────────────────────────────

    /// Get or create a namespace and return a scoped handle.
    ///
    /// ```rust,no_run
    /// # use minnal_db::Db;
    /// let db = Db::open("/tmp/ns_db").unwrap();
    /// let users = db.namespace("users").unwrap();
    /// users.put(b"user:1", b"alice").unwrap();
    /// ```
    pub fn namespace<'a>(&'a self, name: &str) -> Result<Namespace<'a>> {
        let ns_id = match self.inner.get_namespace_id(name) {
            Some(id) => id,
            None => self.inner.create_namespace(name)?,
        };
        let store = self.inner.get_store_by_name(name)?;
        Ok(Namespace {
            ns_id,
            store,
            db: &self.inner,
        })
    }

    /// The namespace named `name`, if it exists. Unlike [`namespace`](Self::namespace)
    /// it never creates one, so a caller racing the namespace's removal cannot
    /// bring it back.
    pub fn existing_namespace<'a>(&'a self, name: &str) -> Option<Namespace<'a>> {
        let ns_id = self.inner.get_namespace_id(name)?;
        // Removed between the two lookups.
        let store = self.inner.get_store(ns_id).ok()?;
        Some(Namespace {
            ns_id,
            store,
            db: &self.inner,
        })
    }

    /// Get or create a namespace with a TTL and return a scoped handle.
    ///
    /// Records in this namespace will be automatically expired after `ttl`.
    /// Note: the TTL worker is only active when using [`AsyncDb`].
    /// In sync mode, the KVStore stores the TTL but no background worker runs.
    ///
    /// ```rust,no_run
    /// # use minnal_db::Db;
    /// # use std::time::Duration;
    /// let db = Db::open("/tmp/ttl_db").unwrap();
    /// let cache = db.namespace_with_ttl("cache", Duration::from_secs(3600)).unwrap();
    /// cache.put(b"session:1", b"data").unwrap();
    /// ```
    pub fn namespace_with_ttl<'a>(&'a self, name: &str, ttl: Duration) -> Result<Namespace<'a>> {
        let ns_id = match self.inner.get_namespace_id(name) {
            Some(id) => id,
            None => self.inner.create_namespace_with_ttl(name, Some(ttl))?,
        };
        let store = self.inner.get_store_by_name(name)?;
        Ok(Namespace {
            ns_id,
            store,
            db: &self.inner,
        })
    }

    /// List all namespaces as `(name, id)` pairs.
    pub fn list_namespaces(&self) -> Vec<(String, NamespaceId)> {
        self.inner.list_namespaces()
    }

    /// Return `(ttl_secs, max_deletes_per_run)` for a namespace, or `None` if
    /// no TTL is registered for it.
    pub fn ttl_config_for_ns(&self, ns_id: NamespaceId) -> Option<(u64, usize)> {
        self.inner
            .registry
            .read()
            .ttl_config(ns_id)
            .map(|(ttl, max_del)| (ttl.as_secs(), max_del))
    }

    /// Remove a namespace and reclaim its on-disk storage (data directory and
    /// index files). The shared WAL is left untouched, so crash recovery for the
    /// remaining namespaces is unaffected.
    pub fn remove_namespace(&self, name: &str) -> Result<NamespaceId> {
        self.inner.remove_namespace(name)
    }

    /// Delete the data directory an interrupted drop left for `name`. Does nothing
    /// if `name` is a namespace: its directory holds live files.
    pub fn remove_unregistered_namespace_dir(&self, name: &str) -> Result<()> {
        self.inner.remove_unregistered_namespace_dir(name)
    }

    // ── Maintenance ───────────────────────────────────────────────────

    /// Returns snapshot statistics for the default namespace.
    pub fn stats(&self) -> Stats {
        self.inner.stats()
    }

    /// Snapshot of engine-wide operational metrics (runtime counters).
    pub fn ops_metrics(&self) -> crate::support::metrics::MetricsSnapshot {
        self.inner.metrics_snapshot()
    }

    /// Field-index write buffers: bytes held now and at peak, the soft and
    /// hard limits, and how often each fired.
    pub fn index_overlay_stats(&self) -> crate::IndexOverlayStats {
        self.inner.index_overlay_budget.stats()
    }

    /// Operational metrics for a single namespace, by name.
    pub fn ops_metrics_for(&self, namespace: &str) -> Result<crate::support::metrics::MetricsSnapshot> {
        self.inner.metrics_snapshot_for(namespace)
    }

    /// Per-namespace operational metrics for every live namespace, keyed by name.
    pub fn ops_metrics_by_namespace(&self) -> Vec<(String, crate::support::metrics::MetricsSnapshot)> {
        self.inner.metrics_snapshot_by_namespace()
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

    /// Run value-log garbage collection on the default namespace.
    pub fn garbage_collect(&self) -> Result<GCStats> {
        self.inner.garbage_collect()
    }

    /// Run value-log GC on every namespace and return per-namespace results.
    pub fn garbage_collect_all(&self) -> Vec<(String, GCStats)> {
        self.inner.garbage_collect_all()
    }

    /// Run WAL garbage collection (reclaims segments wholly below the cut).
    ///
    /// Returns `(bytes_reclaimed, pending_bytes)`: the bytes of segment files
    /// deleted by this pass, and the bytes of WAL recovery would replay
    /// afterwards (from the cut to the tail). If another pass is already
    /// running, nothing is reclaimed and the result is `(0, pending_bytes)`.
    pub fn garbage_collect_wal(&self) -> Result<(u64, u64)> {
        self.inner.garbage_collect_wal()
    }

    /// Trigger LSM compaction across all namespaces.
    pub fn compact(&self) -> Result<()> {
        self.inner.compact_lsm()
    }

    /// Flush the named namespaces' memtables to level 0, making every write
    /// already applied to them durable. See `Database::flush_namespaces`.
    #[cfg(any(test, feature = "semantic-search"))]
    pub(crate) fn flush_namespaces(&self, names: &[String]) -> Result<()> {
        self.inner.flush_namespaces(names)
    }

    /// Make every logged write (`put_logged`) already returned durable: one
    /// fsync of the vector index WAL. See `Database::sync_vector_index_wal`.
    #[cfg(feature = "semantic-search")]
    pub(crate) fn sync_vector_index_wal(&self) -> Result<()> {
        self.inner.sync_vector_index_wal()
    }

    /// The current value-log waste ratio for the default namespace.
    pub fn waste_ratio(&self) -> f64 {
        self.inner.get_waste_ratio()
    }

    // ── Typed CRUD (rkyv ser/de) ──────────────────────────────────────
    //
    // On the default namespace: each forwards to its twin on that namespace's
    // handle, which holds the one implementation.

    fn default_namespace(&self) -> Result<Namespace<'_>> {
        self.namespace(crate::namespace::DEFAULT_NAMESPACE)
    }

    /// Insert a typed key-value pair, serialized via rkyv.
    ///
    /// Returns `Ok(())` once the write is durable. See [*Write durability*](Db#write-durability)
    /// for what that does and does not guarantee about immediate readability.
    pub fn put_typed<K, V>(&self, key: &K, value: &V) -> Result<()>
    where
        K: RkyvEncode,
        V: RkyvEncode,
    {
        self.default_namespace()?.put_typed(key, value)
    }

    /// Look up a typed value by typed key.
    pub fn get_typed<K, V>(&self, key: &K) -> Result<Option<V>>
    where
        K: RkyvEncode,
        V: RkyvDecode,
    {
        self.default_namespace()?.get_typed(key)
    }

    /// Delete by typed key.
    ///
    /// Returns `Ok(())` once the write is durable. See [*Write durability*](Db#write-durability)
    /// for what that does and does not guarantee about immediate readability.
    pub fn delete_typed<K>(&self, key: &K) -> Result<()>
    where
        K: RkyvEncode,
    {
        self.default_namespace()?.delete_typed(key)
    }

    /// Atomically merge a typed value, serialized via rkyv.
    ///
    /// The typed twin of [`Db::merge`]: `merge_fn` sees the stored value and the
    /// operand already deserialized, and returns the new value (or `None` to
    /// delete the key). Every caveat on [`Db::merge`] applies — in particular the
    /// closure runs under a lock and must not call back into the database.
    ///
    /// ```rust,no_run
    /// # use minnal_db::Db;
    /// # fn main() -> Result<(), minnal_db::KVError> {
    /// # let db = Db::open("/tmp/merge_typed_doc")?;
    /// let total: Option<u64> = db.merge_typed(&"visits".to_string(), &1u64, |current, step| {
    ///     Ok(Some(current.unwrap_or(0) + step))
    /// })?;
    /// # let _ = total;
    /// # db.shutdown()?;
    /// # Ok(()) }
    /// ```
    pub fn merge_typed<K, V, F>(&self, key: &K, value: &V, merge_fn: F) -> Result<Option<V>>
    where
        K: RkyvEncode,
        V: RkyvEncode + RkyvDecode,
        F: FnOnce(Option<V>, V) -> Result<Option<V>>,
    {
        self.default_namespace()?.merge_typed(key, value, merge_fn)
    }

    // ── Typed Iteration (rkyv ser/de) ─────────────────────────────────

    /// Iterate over all key-value pairs, deserializing into typed pairs.
    pub fn iter_typed<K, V>(&self) -> Result<Vec<(K, V)>>
    where
        K: RkyvDecode,
        V: RkyvDecode,
    {
        self.default_namespace()?.iter_typed()
    }

    /// Iterate over all keys, deserializing into typed keys.
    pub fn keys_typed<K>(&self) -> Result<Vec<K>>
    where
        K: RkyvDecode,
    {
        self.default_namespace()?.keys_typed()
    }

    /// Iterate over typed key-value pairs in `[start, end)`.
    pub fn range_typed<K, V>(&self, start: &K, end: Option<&K>) -> Result<Vec<(K, V)>>
    where
        K: RkyvEncode + RkyvDecode,
        V: RkyvDecode,
    {
        self.default_namespace()?.range_typed(start, end)
    }

    /// Iterate over typed key-value pairs whose keys start with `prefix`.
    pub fn scan_prefix_typed<K, V>(&self, prefix: &[u8]) -> Result<Vec<(K, V)>>
    where
        K: RkyvDecode,
        V: RkyvDecode,
    {
        self.default_namespace()?.scan_prefix_typed(prefix)
    }

    // ── Index ─────────────────────────────────────────────────────────

    /// Register an indexed field for a namespace and return its [`FieldId`].
    pub fn register_index_field(&self, namespace_id: NamespaceId, field_name: &str, value_type: IndexValueType) -> Result<FieldId> {
        self.inner.register_index_field(namespace_id, field_name, value_type)
    }

    /// Register a custom row-ID function (and optionally its inverse) for a namespace.
    ///
    /// Call this before `activate_field_index` so WAL replay uses consistent row IDs.
    /// Providing `row_to_key_fn` enables O(|hits|) query resolution with zero memory overhead.
    ///
    /// See [`Database::set_row_id_fn`][crate::db::database::Database::set_row_id_fn]
    /// for full documentation.
    pub fn set_row_id_fn(
        &self,
        namespace_id: NamespaceId,
        row_id_fn: crate::kv_store::index_set::RowIdFn,
        row_to_key_fn: Option<crate::kv_store::index_set::RowToKeyFn>,
    ) -> Result<()> {
        self.inner.set_row_id_fn(namespace_id, row_id_fn, row_to_key_fn)
    }

    /// Activate a registered field index.
    pub fn activate_field_index(
        &self,
        namespace_id: NamespaceId,
        field_id: FieldId,
        value_type: IndexValueType,
        extractor: ExtractorFn,
    ) -> Result<()> {
        self.inner.activate_field_index(namespace_id, field_id, value_type, extractor)
    }

    /// Remove a field index from the in-memory registry.
    ///
    /// After this call predicate queries that reference the field return an
    /// `InactiveField` error.  The on-disk checkpoint files are not touched.
    pub fn deactivate_field_index(&self, namespace_id: NamespaceId, field_id: FieldId) -> Result<()> {
        self.inner.deactivate_field_index(namespace_id, field_id)
    }

    /// Permanently drop a field index: persist the drop, deregister it, and
    /// delete its on-disk directory.
    ///
    /// Use this rather than [`deactivate_field_index`](Self::deactivate_field_index)
    /// when the index is going away for good — the latter is an in-memory
    /// deregister that leaves every file in place. The field keeps its
    /// [`FieldId`] for reuse, but must be re-registered before it can be
    /// activated again.
    pub fn drop_field_index(&self, namespace_id: NamespaceId, field_id: FieldId) -> Result<()> {
        self.inner.drop_field_index(namespace_id, field_id)
    }

    /// Free every row ID of a namespace by replacing its dense row map with an
    /// empty one; the next field builds allocate fresh dense IDs for the live
    /// keys only. Every field index of the namespace must already be dropped
    /// ([`drop_field_index`](Self::drop_field_index)) — otherwise this returns
    /// an error and changes nothing. Returns `false` for a namespace whose row
    /// IDs come from a key-derived `RowIdFn` (nothing to reset).
    pub fn reset_rowmap(&self, namespace_id: NamespaceId) -> Result<bool> {
        self.inner.reset_rowmap(namespace_id)
    }

    /// Row IDs a namespace's dense row map has allocated — every key ever
    /// indexed, live or deleted — or `None` when the namespace uses a
    /// key-derived `RowIdFn` or no field index was ever activated.
    pub fn rowmap_ids_allocated(&self, namespace_id: NamespaceId) -> Result<Option<u64>> {
        self.inner.rowmap_ids_allocated(namespace_id)
    }

    /// Return all indexed fields registered for a namespace, sorted by [`FieldId`].
    ///
    /// On a fresh open the list is populated from `config.json` automatically,
    /// so callers do not need to re-call `register_index_field` after a restart.
    pub fn list_index_fields(&self, namespace_id: NamespaceId) -> Vec<crate::field_index::FieldMeta> {
        self.inner.list_index_fields(namespace_id)
    }

    /// Return the number of distinct indexed values for a field, or `None` if
    /// the field is not currently active.
    pub fn field_index_distinct_count(&self, namespace_id: NamespaceId, field_id: FieldId) -> Option<usize> {
        self.inner.field_index_distinct_count(namespace_id, field_id)
    }

    /// Reclaimable dead-space ratios `(bitmap_waste, keymap_waste)` for a field,
    /// or `None` if the field is not currently active.
    pub fn field_index_waste(&self, namespace_id: NamespaceId, field_id: FieldId) -> Option<(f64, f64)> {
        self.inner.field_index_waste(namespace_id, field_id)
    }

    /// On-disk blob growth/waste metrics for a field — bitmap and keymap store
    /// sizes (logical vs. live bytes) and waste ratios — or `None` if the field
    /// is not currently active. Use it to monitor the append-only write
    /// amplification that low-cardinality fields suffer.
    pub fn field_index_blob_stats(&self, namespace_id: NamespaceId, field_id: FieldId) -> Option<crate::field_index::IndexBlobStats> {
        self.inner.field_index_blob_stats(namespace_id, field_id)
    }

    /// Reindex a single field for a single key, re-deriving its value from the
    /// key's current stored bytes using the same logic as the put path. Touches
    /// only the named field. See [`crate::FieldReindexOutcome`].
    pub fn reindex_field(&self, namespace_id: NamespaceId, field_id: FieldId, key: &[u8]) -> Result<FieldReindexOutcome> {
        self.inner.reindex_field(namespace_id, field_id, key)
    }

    /// The configured field-index compaction threshold as a fraction (`0.0..1.0`).
    pub fn index_blob_waste_threshold(&self) -> f64 {
        self.inner.index_blob_waste_threshold()
    }

    /// Report the health of every registered field index in a namespace,
    /// including any outstanding gap and what it would take to repair.
    pub fn index_health(&self, namespace_id: NamespaceId) -> Result<Vec<FieldIndexHealth>> {
        self.inner.index_health(namespace_id)
    }

    /// Repair a degraded field index and clear its gap record.
    ///
    /// Row-scoped when the gap named the affected keys, a full rebuild
    /// otherwise. Does not re-put documents. See
    /// [`FieldRepairOutcome`].
    pub fn repair_field_index(&self, namespace_id: NamespaceId, field_id: FieldId) -> Result<FieldRepairOutcome> {
        self.inner.repair_field_index(namespace_id, field_id)
    }

    /// Evaluate a query string against the active field indices of a namespace
    /// and return the matching keys.
    ///
    /// Returns a [`QueryOutcome`], not a bare `Vec`: a field index can be
    /// *degraded* (missing updates it can no longer recover) while still being
    /// queryable, and serving results that look complete over one is the very
    /// failure FR-001 exists to remove. Check
    /// [`is_degraded`](QueryOutcome::is_degraded) — or `degraded_fields` for
    /// which ones — before treating the result as exhaustive.
    pub fn query_index(&self, namespace_id: NamespaceId, query_str: &str) -> Result<QueryOutcome> {
        self.inner.query_keys(namespace_id, query_str)
    }

    /// Like [`query_index`] but resolves only the `[offset, offset+limit)` window
    /// of matching keys; [`QueryOutcome::total`] still carries the full match
    /// count.
    ///
    /// Prefer this over `query_index` when serving a paginated API — with a
    /// registered `RowToKeyFn` only `offset + limit` keys need to be resolved.
    ///
    /// [`query_index`]: Db::query_index
    pub fn query_index_paginated(&self, namespace_id: NamespaceId, query_str: &str, offset: usize, limit: usize) -> Result<QueryOutcome> {
        self.inner.query_keys_paginated(namespace_id, query_str, offset, limit)
    }
}

// ── Worker trait impls for Db ─────────────────────────────────────────
//
// These allow AsyncDb to use Arc<Db> as the target for background workers,
// avoiding the need to extract Arc<Database> from the non-Arc Db.inner field.

use crate::db::workers::wal_gc::WalGcTarget;

impl WalGcTarget for Db {
    fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }
    fn flush_namespaces_pinning_wal(&self) -> usize {
        self.inner.flush_namespaces_pinning_wal()
    }
    fn wal_pending_bytes(&self) -> u64 {
        self.inner.wal_pending_bytes()
    }
    fn has_deletable_wal_segments(&self) -> bool {
        self.inner.has_deletable_wal_segments()
    }
    fn garbage_collect_wal(&self) -> Result<(u64, u64)> {
        self.inner.garbage_collect_wal()
    }
    fn flush_namespaces_pinning_vector_wal(&self) -> usize {
        self.inner.flush_namespaces_pinning_vector_wal()
    }
    fn has_deletable_vector_wal_segments(&self) -> bool {
        self.inner.has_deletable_vector_wal_segments()
    }
    fn garbage_collect_vector_wal(&self) -> Result<(u64, u64)> {
        self.inner.garbage_collect_vector_wal()
    }
}

impl LsmCompactionTarget for Db {
    fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }
    fn flush_no_wal_memtables(&self) -> usize {
        self.inner.flush_no_wal_memtables()
    }
    fn has_due_compaction_work(&self) -> bool {
        self.inner.has_due_lsm_compaction_work()
    }
    fn compact_due(&self) -> Result<()> {
        self.inner.compact_lsm_due()
    }
}

impl ValueLogGcTarget for Db {
    fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }
    fn run_gc_if_needed(&self, threshold: f64) {
        self.inner.run_gc_if_needed(threshold)
    }
}

impl IndexCheckpointTarget for Db {
    fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }
    fn run_index_checkpoint(&self) -> Result<usize> {
        self.inner.run_index_checkpoint()
    }
}

impl TtlTarget for Db {
    fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }
    fn run_ttl_pass(&self) {
        self.inner.run_ttl_pass()
    }
}
