//! [`Db`], the synchronous engine handle that [`AsyncDb`] wraps (crate-internal).

use super::*;
use crate::namespace::NamespaceId;

// ── Db (the engine behind AsyncDb) ───────────────────────────────────

/// The synchronous engine behind [`AsyncDb`], crate-internal: `AsyncDb` runs
/// each call on the blocking pool. The public contract (write durability,
/// `merge`, scans) is documented on [`AsyncDb`].
pub struct Db {
    pub(super) inner: Database,
    /// The doc store's directory lock. Fields drop in declaration order, so it
    /// is released only after `inner` has finished dropping (its final flush
    /// and compaction), however long a worker's pass keeps this `Db` alive past
    /// the last user handle. Keep it last.
    #[cfg(feature = "doc-store")]
    pub(super) dir_lock: parking_lot::Mutex<Option<crate::support::file_lock::DirLock>>,
}

#[cfg(test)]
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

    /// Open a database at `path` with the given configuration; starts no
    /// worker (they run under [`AsyncDb`]).
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
    /// Returns `Ok(())` once the write is durable. See [*Write durability*](AsyncDb#write-durability)
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
    /// Returns `Ok(())` once the write is durable. See [*Write durability*](AsyncDb#write-durability)
    /// for what that does and does not guarantee about immediate readability.
    pub fn delete(&self, key: &[u8]) -> Result<()> {
        self.inner.delete(key)
    }

    /// The engine's [`AsyncDb::merge`]: see there for the contract.
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

    /// The namespace named `name` (its id and store), if it exists. It never
    /// creates one, so a caller racing the namespace's removal cannot bring it
    /// back.
    pub fn existing_namespace(&self, name: &str) -> Option<(NamespaceId, Arc<KVStore>)> {
        let ns_id = self.inner.get_namespace_id(name)?;
        // Removed between the two lookups.
        let store = self.inner.get_store(ns_id).ok()?;
        Some((ns_id, store))
    }

    /// List all namespaces as `(name, id)` pairs.
    pub fn list_namespaces(&self) -> Vec<(String, NamespaceId)> {
        self.inner.list_namespaces()
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
