//! Db — single entry-point facade for MinnalDB.
//!
//! Wraps the internal `Database` coordinator and exposes a clean,
//! user-facing API for CRUD, iteration, namespaces, and maintenance.

use log::info;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use crate::db::config::DbConfig;
use crate::db::database::Database;
use crate::db::error::{KVError, Result};
use crate::db::index_checkpoint_worker::{DEFAULT_CHECKPOINT_INTERVAL, IndexCheckpointTarget, IndexCheckpointWorker};
use crate::db::kv_store::{KVStore, KeyValue, ScanPage};
use crate::db::namespace::{FieldId, FieldReindexOutcome};
use crate::db::namespace_index::ExtractorFn;
use crate::db::stats::{GCStats, Stats};
use crate::db::toml_config::MinnalTomlConfig;
use crate::db::ttl_worker::{TtlTarget, TtlWorker};
use crate::db::wal_worker::WalGcWorker;
use crate::index::IndexValueType;
use crate::store::gc_value_log_worker::{GCWorker, ValueLogGcTarget};
use crate::store::lsm_worker::{LsmCompactionTarget, LsmCompactionWorker};

// ── rkyv trait alias ───────────────────────────────────────────────────
//
// Bounds required for a type to be used with the `_typed` methods.
// Users derive these via `#[derive(Archive, RkyvSerialize, RkyvDeserialize)]`.

use rkyv::api::high::{HighSerializer, HighValidator};
use rkyv::rancor::Error as RkyvError;
use rkyv::rancor::Strategy;
use rkyv::ser::allocator::ArenaHandle;

/// Serialize a value to bytes via rkyv.
fn rkyv_serialize<T>(value: &T) -> Result<Vec<u8>>
where
    T: for<'a> rkyv::Serialize<HighSerializer<rkyv::util::AlignedVec, ArenaHandle<'a>, RkyvError>>,
{
    rkyv::to_bytes::<RkyvError>(value)
        .map(|buf| buf.to_vec())
        .map_err(|e| KVError::Serialization(format!("rkyv serialize: {}", e)))
}

/// Deserialize bytes into a value via rkyv.
///
/// Bytes here always originate from our own `rkyv_serialize`, so bytecheck
/// validation is skipped — `access_unchecked` is safe for trusted internal storage.
fn rkyv_deserialize<T>(bytes: &[u8]) -> Result<T>
where
    T: rkyv::Archive,
    T::Archived: rkyv::Deserialize<T, rkyv::api::high::HighDeserializer<RkyvError>>,
{
    // SAFETY: bytes were written by rkyv_serialize from the same type T,
    // so the archived layout is valid and correctly aligned.
    let archived = unsafe { rkyv::access_unchecked::<T::Archived>(bytes) };
    rkyv::deserialize::<T, RkyvError>(archived).map_err(|e| KVError::Serialization(format!("rkyv deserialize: {:?}", e)))
}

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
pub struct Db {
    inner: Database,
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
        Ok(Self { inner: db })
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
    pub fn put(&self, key: &[u8], value: &[u8]) -> Result<()> {
        self.inner.put(key, value)
    }

    /// Look up a value by key. Returns `None` if the key does not exist.
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.inner.get(key)
    }

    /// Delete a key.
    pub fn delete(&self, key: &[u8]) -> Result<()> {
        self.inner.delete(key)
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

    /// Get or create a namespace with a TTL and return a scoped handle.
    ///
    /// Records in this namespace will be automatically expired after `ttl`.
    /// Note: the TTL worker is only active when using `AsyncDatabase`.
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
    pub fn list_namespaces(&self) -> Vec<(String, u32)> {
        self.inner.list_namespaces()
    }

    /// Return `(ttl_secs, max_deletes_per_run)` for a namespace, or `None` if
    /// no TTL is registered for it.
    pub fn ttl_config_for_ns(&self, ns_id: u32) -> Option<(u64, usize)> {
        self.inner
            .registry
            .read()
            .ttl_config(ns_id)
            .map(|(ttl, max_del)| (ttl.as_secs(), max_del))
    }

    /// Remove a namespace and reclaim its on-disk storage (data directory and
    /// index files). The shared WAL is left untouched, so crash recovery for the
    /// remaining namespaces is unaffected.
    pub fn remove_namespace(&self, name: &str) -> Result<u32> {
        self.inner.remove_namespace(name)
    }

    // ── Maintenance ───────────────────────────────────────────────────

    /// Returns snapshot statistics for the default namespace.
    pub fn stats(&self) -> Stats {
        self.inner.stats()
    }

    /// Snapshot of engine-wide operational metrics (runtime counters).
    pub fn ops_metrics(&self) -> crate::db::metrics::MetricsSnapshot {
        self.inner.metrics_snapshot()
    }

    /// Operational metrics for a single namespace, by name.
    pub fn ops_metrics_for(&self, namespace: &str) -> Result<crate::db::metrics::MetricsSnapshot> {
        self.inner.metrics_snapshot_for(namespace)
    }

    /// Per-namespace operational metrics for every live namespace, keyed by name.
    pub fn ops_metrics_by_namespace(&self) -> Vec<(String, crate::db::metrics::MetricsSnapshot)> {
        self.inner.metrics_snapshot_by_namespace()
    }

    /// Returns a snapshot of the current WAL metadata.
    pub fn wal_metadata(&self) -> crate::db::wal::WalMetadata {
        self.inner.wal_metadata()
    }

    /// Returns a live LSM manifest snapshot for every active namespace.
    pub fn lsm_manifests(&self) -> Vec<(String, crate::store::lsm::lsm_manifest::LsmManifest)> {
        self.inner.lsm_manifests()
    }

    /// Returns the in-memory (non-SSTable) LSM stats for every active namespace.
    pub fn lsm_runtime_stats(&self) -> Vec<(String, crate::store::lsm::lsm_tree::LSMStats)> {
        self.inner.lsm_runtime_stats()
    }

    /// Returns per-bucket value-log metadata for every active namespace.
    pub fn value_log_shard_stats(&self) -> Vec<(String, Vec<(u32, crate::store::value_log::ValueLogMetadata)>)> {
        self.inner.value_log_shard_stats()
    }

    /// Physical (on-disk) vs logical value-log footprint per shard, per namespace.
    pub fn value_log_physical_stats(&self) -> Vec<(String, Vec<crate::store::value_log::sharded::ShardPhysicalStats>)> {
        self.inner.value_log_physical_stats()
    }

    /// Per-page value-log garbage breakdown for one namespace (by name).
    pub fn value_log_segment_stats(&self, namespace: &str) -> Result<Vec<(u32, Vec<crate::store::value_log::SegmentStats>)>> {
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

    /// Run WAL garbage collection (reclaims fully-persisted segments).
    pub fn garbage_collect_wal(&self) -> Result<(u64, u64)> {
        self.inner.garbage_collect_wal()
    }

    /// Trigger LSM compaction across all namespaces.
    pub fn compact(&self) -> Result<()> {
        self.inner.compact_lsm()
    }

    /// The current value-log waste ratio for the default namespace.
    pub fn waste_ratio(&self) -> f64 {
        self.inner.get_waste_ratio()
    }

    // ── Typed CRUD (rkyv ser/de) ──────────────────────────────────────

    /// Insert a typed key-value pair, serialized via rkyv.
    pub fn put_typed<K, V>(&self, key: &K, value: &V) -> Result<()>
    where
        K: for<'a> rkyv::Serialize<HighSerializer<rkyv::util::AlignedVec, ArenaHandle<'a>, RkyvError>>,
        V: for<'a> rkyv::Serialize<HighSerializer<rkyv::util::AlignedVec, ArenaHandle<'a>, RkyvError>>,
    {
        let kb = rkyv_serialize(key)?;
        let vb = rkyv_serialize(value)?;
        self.put(&kb, &vb)
    }

    /// Look up a typed value by typed key.
    pub fn get_typed<K, V>(&self, key: &K) -> Result<Option<V>>
    where
        K: for<'a> rkyv::Serialize<HighSerializer<rkyv::util::AlignedVec, ArenaHandle<'a>, RkyvError>>,
        V: rkyv::Archive,
        V::Archived: for<'a> bytecheck::CheckBytes<HighValidator<'a, RkyvError>> + rkyv::Deserialize<V, Strategy<rkyv::de::Pool, RkyvError>>,
    {
        let kb = rkyv_serialize(key)?;
        match self.get(&kb)? {
            Some(vb) => Ok(Some(rkyv_deserialize::<V>(&vb)?)),
            None => Ok(None),
        }
    }

    /// Delete by typed key.
    pub fn delete_typed<K>(&self, key: &K) -> Result<()>
    where
        K: for<'a> rkyv::Serialize<HighSerializer<rkyv::util::AlignedVec, ArenaHandle<'a>, RkyvError>>,
    {
        let kb = rkyv_serialize(key)?;
        self.delete(&kb)
    }

    // ── Typed Iteration (rkyv ser/de) ─────────────────────────────────

    /// Iterate over all key-value pairs, deserializing into typed pairs.
    pub fn iter_typed<K, V>(&self) -> Result<Vec<(K, V)>>
    where
        K: rkyv::Archive,
        K::Archived: for<'a> bytecheck::CheckBytes<HighValidator<'a, RkyvError>> + rkyv::Deserialize<K, Strategy<rkyv::de::Pool, RkyvError>>,
        V: rkyv::Archive,
        V::Archived: for<'a> bytecheck::CheckBytes<HighValidator<'a, RkyvError>> + rkyv::Deserialize<V, Strategy<rkyv::de::Pool, RkyvError>>,
    {
        deserialize_pairs(self.iter()?)
    }

    /// Iterate over all keys, deserializing into typed keys.
    pub fn keys_typed<K>(&self) -> Result<Vec<K>>
    where
        K: rkyv::Archive,
        K::Archived: for<'a> bytecheck::CheckBytes<HighValidator<'a, RkyvError>> + rkyv::Deserialize<K, Strategy<rkyv::de::Pool, RkyvError>>,
    {
        deserialize_keys(self.keys()?)
    }

    /// Iterate over typed key-value pairs in `[start, end)`.
    pub fn range_typed<K, V>(&self, start: &K, end: Option<&K>) -> Result<Vec<(K, V)>>
    where
        K: for<'a> rkyv::Serialize<HighSerializer<rkyv::util::AlignedVec, ArenaHandle<'a>, RkyvError>> + rkyv::Archive,
        K::Archived: for<'a> bytecheck::CheckBytes<HighValidator<'a, RkyvError>> + rkyv::Deserialize<K, Strategy<rkyv::de::Pool, RkyvError>>,
        V: rkyv::Archive,
        V::Archived: for<'a> bytecheck::CheckBytes<HighValidator<'a, RkyvError>> + rkyv::Deserialize<V, Strategy<rkyv::de::Pool, RkyvError>>,
    {
        let start_bytes = rkyv_serialize(start)?;
        let end_bytes = end.map(|e| rkyv_serialize(e)).transpose()?;
        deserialize_pairs(self.range(&start_bytes, end_bytes.as_deref())?)
    }

    /// Iterate over typed key-value pairs whose keys start with `prefix`.
    pub fn scan_prefix_typed<K, V>(&self, prefix: &[u8]) -> Result<Vec<(K, V)>>
    where
        K: rkyv::Archive,
        K::Archived: for<'a> bytecheck::CheckBytes<HighValidator<'a, RkyvError>> + rkyv::Deserialize<K, Strategy<rkyv::de::Pool, RkyvError>>,
        V: rkyv::Archive,
        V::Archived: for<'a> bytecheck::CheckBytes<HighValidator<'a, RkyvError>> + rkyv::Deserialize<V, Strategy<rkyv::de::Pool, RkyvError>>,
    {
        deserialize_pairs(self.scan_prefix(prefix)?)
    }

    // ── Index ─────────────────────────────────────────────────────────

    /// Register an indexed field for a namespace and return its [`FieldId`].
    pub fn register_index_field(&self, namespace_id: u32, field_name: &str, value_type: IndexValueType) -> Result<FieldId> {
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
        namespace_id: u32,
        row_id_fn: crate::db::namespace_index::RowIdFn,
        row_to_key_fn: Option<crate::db::namespace_index::RowToKeyFn>,
    ) -> Result<()> {
        self.inner.set_row_id_fn(namespace_id, row_id_fn, row_to_key_fn)
    }

    /// Activate a registered field index.
    pub fn activate_field_index(&self, namespace_id: u32, field_id: FieldId, value_type: IndexValueType, extractor: ExtractorFn) -> Result<()> {
        self.inner.activate_field_index(namespace_id, field_id, value_type, extractor)
    }

    /// Remove a field index from the in-memory registry.
    ///
    /// After this call predicate queries that reference the field return an
    /// `InactiveField` error.  The on-disk checkpoint files are not touched.
    pub fn deactivate_field_index(&self, namespace_id: u32, field_id: FieldId) -> Result<()> {
        self.inner.deactivate_field_index(namespace_id, field_id)
    }

    /// Return all indexed fields registered for a namespace, sorted by [`FieldId`].
    ///
    /// On a fresh open the list is populated from `config.json` automatically,
    /// so callers do not need to re-call `register_index_field` after a restart.
    pub fn list_index_fields(&self, namespace_id: u32) -> Vec<crate::db::namespace::FieldMeta> {
        self.inner.list_index_fields(namespace_id)
    }

    /// Return the number of distinct indexed values for a field, or `None` if
    /// the field is not currently active.
    pub fn field_index_distinct_count(&self, namespace_id: u32, field_id: FieldId) -> Option<usize> {
        self.inner.field_index_distinct_count(namespace_id, field_id)
    }

    /// Reclaimable dead-space ratios `(bitmap_waste, keymap_waste)` for a field,
    /// or `None` if the field is not currently active.
    pub fn field_index_waste(&self, namespace_id: u32, field_id: FieldId) -> Option<(f64, f64)> {
        self.inner.field_index_waste(namespace_id, field_id)
    }

    /// On-disk blob growth/waste metrics for a field — bitmap and keymap store
    /// sizes (logical vs. live bytes) and waste ratios — or `None` if the field
    /// is not currently active. Use it to monitor the append-only write
    /// amplification that low-cardinality fields suffer.
    pub fn field_index_blob_stats(&self, namespace_id: u32, field_id: FieldId) -> Option<crate::index::IndexBlobStats> {
        self.inner.field_index_blob_stats(namespace_id, field_id)
    }

    /// Reindex a single field for a single key, re-deriving its value from the
    /// key's current stored bytes using the same logic as the put path. Touches
    /// only the named field. See [`crate::FieldReindexOutcome`].
    pub fn reindex_field(&self, namespace_id: u32, field_id: FieldId, key: &[u8]) -> Result<FieldReindexOutcome> {
        self.inner.reindex_field(namespace_id, field_id, key)
    }

    /// The configured field-index compaction threshold as a fraction (`0.0..1.0`).
    pub fn index_blob_waste_threshold(&self) -> f64 {
        self.inner.index_blob_waste_threshold()
    }

    /// Evaluate a query string against the active field indices of a namespace
    /// and return the raw keys of all matching documents.
    pub fn query_index(&self, namespace_id: u32, query_str: &str) -> Result<Vec<Vec<u8>>> {
        self.inner.query_keys(namespace_id, query_str)
    }

    /// Like [`query_index`] but returns only the `[offset, offset+limit)` window of
    /// matching keys together with the full match count.
    ///
    /// Prefer this over `query_index` when serving a paginated API — with a
    /// registered `RowToKeyFn` only `offset + limit` keys need to be resolved.
    ///
    /// [`query_index`]: Db::query_index
    pub fn query_index_paginated(&self, namespace_id: u32, query_str: &str, offset: usize, limit: usize) -> Result<(Vec<Vec<u8>>, usize)> {
        self.inner.query_keys_paginated(namespace_id, query_str, offset, limit)
    }
}

// ── Worker trait impls for Db ─────────────────────────────────────────
//
// These allow AsyncDb to use Arc<Db> as the target for background workers,
// avoiding the need to extract Arc<Database> from the non-Arc Db.inner field.

use crate::db::wal_worker::WalGcTarget;

impl WalGcTarget for Db {
    fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }
    fn get_wal_gc_stats(&self) -> (u64, u64) {
        self.inner.get_wal_gc_stats()
    }
    fn has_deletable_wal_segments(&self) -> bool {
        self.inner.has_deletable_wal_segments()
    }
    fn garbage_collect_wal(&self) -> Result<(u64, u64)> {
        self.inner.garbage_collect_wal()
    }
}

impl LsmCompactionTarget for Db {
    fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }
    fn has_lsm_compaction_work(&self) -> bool {
        self.inner.has_lsm_compaction_work()
    }
    fn compact_lsm(&self) -> Result<()> {
        self.inner.compact_lsm()
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

// ── Namespace (scoped handle) ──────────────────────────────────────────

/// A scoped handle to a single namespace within the database.
///
/// Provides the same CRUD and iteration methods as `Db`, but all
/// operations are pinned to the namespace this handle was created for.
pub struct Namespace<'db> {
    ns_id: u32,
    store: Arc<KVStore>,
    db: &'db Database,
}

impl<'db> Namespace<'db> {
    /// The namespace ID.
    pub fn id(&self) -> u32 {
        self.ns_id
    }

    /// The TTL for this namespace, if configured.
    pub fn ttl(&self) -> Option<Duration> {
        self.store.ttl
    }

    // ── CRUD ──────────────────────────────────────────────────────────

    /// Insert or update a key-value pair in this namespace.
    pub fn put(&self, key: &[u8], value: &[u8]) -> Result<()> {
        self.db.put_ns(self.ns_id, key, value)
    }

    /// Look up a value by key. Returns `None` if the key does not exist.
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.db.get_ns(self.ns_id, key)
    }

    /// Delete a key from this namespace.
    pub fn delete(&self, key: &[u8]) -> Result<()> {
        self.db.delete_ns(self.ns_id, key)
    }

    // ── Iteration ─────────────────────────────────────────────────────

    /// Iterate over all key-value pairs in key order.
    pub fn iter(&self) -> Result<Vec<KeyValue>> {
        self.store.scan_range_batch(&[], None)
    }

    /// Iterate over all keys.
    pub fn keys(&self) -> Result<Vec<Vec<u8>>> {
        self.store.keys()
    }

    /// Iterate over key-value pairs in `[start, end)`.
    pub fn range(&self, start: &[u8], end: Option<&[u8]>) -> Result<Vec<KeyValue>> {
        self.store.scan_range_batch(start, end)
    }

    /// Iterate over key-value pairs whose keys start with `prefix`.
    pub fn scan_prefix(&self, prefix: &[u8]) -> Result<Vec<KeyValue>> {
        self.store.scan_prefix_batch(prefix)
    }

    /// Paginated cursor scan over `[cursor, end)` on this namespace.
    ///
    /// Returns up to `limit` key-value pairs starting from `cursor`
    /// (or the beginning if `None`) and ending before `end` (or the last key if
    /// `None`), plus an optional next cursor.
    pub fn scan(&self, cursor: Option<&[u8]>, end: Option<&[u8]>, limit: usize) -> Result<ScanPage> {
        self.store.scan_page_batch(cursor, end, limit)
    }

    // ── Maintenance ───────────────────────────────────────────────────

    /// Returns snapshot statistics for this namespace.
    pub fn stats(&self) -> Stats {
        self.store.stats()
    }

    /// Run value-log garbage collection on this namespace.
    pub fn garbage_collect(&self) -> Result<GCStats> {
        self.db.garbage_collect_namespace(self.ns_id)
    }

    // ── Typed CRUD (rkyv ser/de) ──────────────────────────────────────

    /// Insert a typed key-value pair, serialized via rkyv.
    pub fn put_typed<K, V>(&self, key: &K, value: &V) -> Result<()>
    where
        K: for<'a> rkyv::Serialize<HighSerializer<rkyv::util::AlignedVec, ArenaHandle<'a>, RkyvError>>,
        V: for<'a> rkyv::Serialize<HighSerializer<rkyv::util::AlignedVec, ArenaHandle<'a>, RkyvError>>,
    {
        let kb = rkyv_serialize(key)?;
        let vb = rkyv_serialize(value)?;
        self.put(&kb, &vb)
    }

    /// Look up a typed value by typed key.
    pub fn get_typed<K, V>(&self, key: &K) -> Result<Option<V>>
    where
        K: for<'a> rkyv::Serialize<HighSerializer<rkyv::util::AlignedVec, ArenaHandle<'a>, RkyvError>>,
        V: rkyv::Archive,
        V::Archived: for<'a> bytecheck::CheckBytes<HighValidator<'a, RkyvError>> + rkyv::Deserialize<V, Strategy<rkyv::de::Pool, RkyvError>>,
    {
        let kb = rkyv_serialize(key)?;
        match self.db.get_ns(self.ns_id, &kb)? {
            Some(vb) => Ok(Some(rkyv_deserialize::<V>(&vb)?)),
            None => Ok(None),
        }
    }

    /// Delete by typed key.
    pub fn delete_typed<K>(&self, key: &K) -> Result<()>
    where
        K: for<'a> rkyv::Serialize<HighSerializer<rkyv::util::AlignedVec, ArenaHandle<'a>, RkyvError>>,
    {
        let kb = rkyv_serialize(key)?;
        self.delete(&kb)
    }

    // ── Typed Iteration (rkyv ser/de) ─────────────────────────────────

    /// Iterate over all typed key-value pairs in key order.
    pub fn iter_typed<K, V>(&self) -> Result<Vec<(K, V)>>
    where
        K: rkyv::Archive,
        K::Archived: for<'a> bytecheck::CheckBytes<HighValidator<'a, RkyvError>> + rkyv::Deserialize<K, Strategy<rkyv::de::Pool, RkyvError>>,
        V: rkyv::Archive,
        V::Archived: for<'a> bytecheck::CheckBytes<HighValidator<'a, RkyvError>> + rkyv::Deserialize<V, Strategy<rkyv::de::Pool, RkyvError>>,
    {
        deserialize_pairs(self.iter()?)
    }

    /// Iterate over all typed keys.
    pub fn keys_typed<K>(&self) -> Result<Vec<K>>
    where
        K: rkyv::Archive,
        K::Archived: for<'a> bytecheck::CheckBytes<HighValidator<'a, RkyvError>> + rkyv::Deserialize<K, Strategy<rkyv::de::Pool, RkyvError>>,
    {
        deserialize_keys(self.keys()?)
    }

    /// Iterate over typed key-value pairs in `[start, end)`.
    pub fn range_typed<K, V>(&self, start: &K, end: Option<&K>) -> Result<Vec<(K, V)>>
    where
        K: for<'a> rkyv::Serialize<HighSerializer<rkyv::util::AlignedVec, ArenaHandle<'a>, RkyvError>> + rkyv::Archive,
        K::Archived: for<'a> bytecheck::CheckBytes<HighValidator<'a, RkyvError>> + rkyv::Deserialize<K, Strategy<rkyv::de::Pool, RkyvError>>,
        V: rkyv::Archive,
        V::Archived: for<'a> bytecheck::CheckBytes<HighValidator<'a, RkyvError>> + rkyv::Deserialize<V, Strategy<rkyv::de::Pool, RkyvError>>,
    {
        let start_bytes = rkyv_serialize(start)?;
        let end_bytes = end.map(|e| rkyv_serialize(e)).transpose()?;
        deserialize_pairs(self.range(&start_bytes, end_bytes.as_deref())?)
    }

    /// Iterate over typed key-value pairs whose keys start with `prefix`.
    pub fn scan_prefix_typed<K, V>(&self, prefix: &[u8]) -> Result<Vec<(K, V)>>
    where
        K: rkyv::Archive,
        K::Archived: for<'a> bytecheck::CheckBytes<HighValidator<'a, RkyvError>> + rkyv::Deserialize<K, Strategy<rkyv::de::Pool, RkyvError>>,
        V: rkyv::Archive,
        V::Archived: for<'a> bytecheck::CheckBytes<HighValidator<'a, RkyvError>> + rkyv::Deserialize<V, Strategy<rkyv::de::Pool, RkyvError>>,
    {
        deserialize_pairs(self.scan_prefix(prefix)?)
    }
}

// ── Shared helpers ─────────────────────────────────────────────────────

/// Deserialize a list of raw key-value byte pairs into typed pairs.
fn deserialize_pairs<K, V>(pairs: Vec<(Vec<u8>, Vec<u8>)>) -> Result<Vec<(K, V)>>
where
    K: rkyv::Archive,
    K::Archived: for<'a> bytecheck::CheckBytes<HighValidator<'a, RkyvError>> + rkyv::Deserialize<K, Strategy<rkyv::de::Pool, RkyvError>>,
    V: rkyv::Archive,
    V::Archived: for<'a> bytecheck::CheckBytes<HighValidator<'a, RkyvError>> + rkyv::Deserialize<V, Strategy<rkyv::de::Pool, RkyvError>>,
{
    pairs
        .into_iter()
        .map(|(kb, vb)| {
            let key = rkyv_deserialize::<K>(&kb)?;
            let val = rkyv_deserialize::<V>(&vb)?;
            Ok((key, val))
        })
        .collect()
}

/// Deserialize a list of raw key bytes into typed keys.
fn deserialize_keys<K>(keys: Vec<Vec<u8>>) -> Result<Vec<K>>
where
    K: rkyv::Archive,
    K::Archived: for<'a> bytecheck::CheckBytes<HighValidator<'a, RkyvError>> + rkyv::Deserialize<K, Strategy<rkyv::de::Pool, RkyvError>>,
{
    keys.into_iter().map(|kb| rkyv_deserialize::<K>(&kb)).collect()
}

// ── AsyncDb ────────────────────────────────────────────────────────────

/// Async wrapper around [`Db`].
///
/// All blocking operations are offloaded via `tokio::task::spawn_blocking`.
#[derive(Clone)]
pub struct AsyncDb {
    inner: Arc<Db>,
}

impl AsyncDb {
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

        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            // Flush index state on clean shutdown before stopping the checkpoint worker.
            if let Err(e) = db.inner.run_index_checkpoint() {
                log::warn!("[AsyncDb] Final index checkpoint failed: {:?}", e);
            }
            db.shutdown()
        })
        .await
        .map_err(|e| KVError::Io(std::io::Error::other(e)))?
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
        info!("[AsyncDb] Index checkpoint worker enabled with {}s interval", interval.as_secs());
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
        self.enable_index_checkpoint_worker(DEFAULT_CHECKPOINT_INTERVAL).await?;
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

    pub async fn delete(&self, key: Vec<u8>) -> Result<()> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || db.delete(&key))
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

    /// Run WAL garbage collection (reclaims fully-persisted segments).
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
    pub fn ops_metrics(&self) -> crate::db::metrics::MetricsSnapshot {
        self.inner.ops_metrics()
    }

    /// Operational metrics for a single namespace, by name.
    pub fn ops_metrics_for(&self, namespace: &str) -> Result<crate::db::metrics::MetricsSnapshot> {
        self.inner.ops_metrics_for(namespace)
    }

    /// Per-namespace operational metrics for every live namespace, keyed by name.
    pub fn ops_metrics_by_namespace(&self) -> Vec<(String, crate::db::metrics::MetricsSnapshot)> {
        self.inner.ops_metrics_by_namespace()
    }

    /// Returns a snapshot of the current WAL metadata.
    pub fn wal_metadata(&self) -> crate::db::wal::WalMetadata {
        self.inner.wal_metadata()
    }

    /// Returns a live LSM manifest snapshot for every active namespace.
    pub fn lsm_manifests(&self) -> Vec<(String, crate::store::lsm::lsm_manifest::LsmManifest)> {
        self.inner.lsm_manifests()
    }

    /// Returns the in-memory (non-SSTable) LSM stats for every active namespace.
    pub fn lsm_runtime_stats(&self) -> Vec<(String, crate::store::lsm::lsm_tree::LSMStats)> {
        self.inner.lsm_runtime_stats()
    }

    /// Returns per-bucket value-log metadata for every active namespace.
    pub fn value_log_shard_stats(&self) -> Vec<(String, Vec<(u32, crate::store::value_log::ValueLogMetadata)>)> {
        self.inner.value_log_shard_stats()
    }

    /// Physical (on-disk) vs logical value-log footprint per shard, per namespace.
    pub fn value_log_physical_stats(&self) -> Vec<(String, Vec<crate::store::value_log::sharded::ShardPhysicalStats>)> {
        self.inner.value_log_physical_stats()
    }

    /// Per-page value-log garbage breakdown for one namespace (by name).
    pub fn value_log_segment_stats(&self, namespace: &str) -> Result<Vec<(u32, Vec<crate::store::value_log::SegmentStats>)>> {
        self.inner.value_log_segment_stats(namespace)
    }

    pub fn waste_ratio(&self) -> f64 {
        self.inner.waste_ratio()
    }

    // ── Typed CRUD (rkyv ser/de) ──────────────────────────────────────

    /// Insert a typed key-value pair, serialized via rkyv.
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
    pub async fn delete_typed<K>(&self, key: &K) -> Result<()>
    where
        K: for<'a> rkyv::Serialize<HighSerializer<rkyv::util::AlignedVec, ArenaHandle<'a>, RkyvError>>,
    {
        let kb = rkyv_serialize(key)?;
        self.delete(kb).await
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

    /// Remove a field index from the in-memory registry.
    ///
    /// After this call predicate queries that reference the field return an
    /// `InactiveField` error.  The on-disk checkpoint files are not touched.
    pub fn deactivate_field_index(&self, namespace_id: u32, field_id: FieldId) -> Result<()> {
        self.inner.inner.deactivate_field_index(namespace_id, field_id)
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
        row_id_fn: crate::db::namespace_index::RowIdFn,
        row_to_key_fn: Option<crate::db::namespace_index::RowToKeyFn>,
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
    pub fn list_index_fields(&self, namespace_id: u32) -> Vec<crate::db::namespace::FieldMeta> {
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
    pub fn field_index_blob_stats(&self, namespace_id: u32, field_id: FieldId) -> Option<crate::index::IndexBlobStats> {
        self.inner.inner.field_index_blob_stats(namespace_id, field_id)
    }

    /// Reindex a single field for a single key, re-deriving its value from the
    /// key's current stored bytes using the same logic as the put path. Touches
    /// only the named field. See [`crate::FieldReindexOutcome`].
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
    pub async fn query_index(&self, namespace_id: u32, query: impl Into<String> + Send + 'static) -> Result<Vec<Vec<u8>>> {
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
    ) -> Result<(Vec<Vec<u8>>, usize)> {
        let db = Arc::clone(&self.inner);
        let q = query.into();
        tokio::task::spawn_blocking(move || db.inner.query_keys_paginated(namespace_id, &q, offset, limit))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }
}

// ── AsyncNamespace ─────────────────────────────────────────────────────

/// Async scoped handle to a single namespace.
#[derive(Clone)]
pub struct AsyncNamespace {
    ns_id: u32,
    store: Arc<KVStore>,
    db: Arc<Db>,
}

impl AsyncNamespace {
    pub fn id(&self) -> u32 {
        self.ns_id
    }

    /// The TTL for this namespace, if configured.
    pub fn ttl(&self) -> Option<Duration> {
        self.store.ttl
    }

    pub async fn put(&self, key: Vec<u8>, value: Vec<u8>) -> Result<()> {
        let db = self.db.clone();
        let ns_id = self.ns_id;
        tokio::task::spawn_blocking(move || db.inner.put_ns(ns_id, &key, &value))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// Write a key-value pair without WAL — see [`Database::put_ns_no_wal`] for
    /// the crash-safety trade-off.
    pub async fn put_no_wal(&self, key: Vec<u8>, value: Vec<u8>) -> Result<()> {
        let db = self.db.clone();
        let ns_id = self.ns_id;
        tokio::task::spawn_blocking(move || db.inner.put_ns_no_wal(ns_id, &key, &value))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    pub async fn get(&self, key: Vec<u8>) -> Result<Option<Vec<u8>>> {
        let db = self.db.clone();
        let ns_id = self.ns_id;
        tokio::task::spawn_blocking(move || db.inner.get_ns(ns_id, &key))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// Scan multiple 4-byte BE u32 prefixes in a single blocking task.
    ///
    /// Internally performs one sequential LSM scan across all prefix IDs, then reads
    /// all value-log entries in exactly `num_buckets` threads —
    /// dramatically cheaper than one `spawn_blocking` per prefix.
    ///
    /// Returns a map from `prefix_id` to `(raw_key_bytes, value_bytes)` pairs.
    pub async fn scan_prefixes_batch(&self, prefix_ids: Vec<u32>) -> std::collections::HashMap<u32, Vec<(Vec<u8>, Vec<u8>)>> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || store.scan_prefixes_batch(&prefix_ids))
            .await
            .ok()
            .and_then(|r| r.ok())
            .unwrap_or_default()
    }

    /// Fetch multiple keys in a single blocking task.
    ///
    /// Returns one `Option<Vec<u8>>` per input key in the same order.
    /// Missing keys produce `None`; storage errors are silently mapped to `None`.
    pub async fn get_multiple(&self, keys: Vec<Vec<u8>>) -> Vec<Option<Vec<u8>>> {
        let store = self.store.clone();
        let n = keys.len();
        tokio::task::spawn_blocking(move || store.get_multiple(&keys))
            .await
            .unwrap_or_else(|_| vec![None; n])
    }

    pub async fn delete(&self, key: Vec<u8>) -> Result<()> {
        let db = self.db.clone();
        let ns_id = self.ns_id;
        tokio::task::spawn_blocking(move || db.inner.delete_ns(ns_id, &key))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// Delete a key without WAL — see [`Database::delete_ns_no_wal`] for the
    /// crash-safety trade-off. The no-WAL counterpart of [`Self::put_no_wal`],
    /// intended for derived/regenerable data such as TTL caches.
    pub async fn delete_no_wal(&self, key: Vec<u8>) -> Result<()> {
        let db = self.db.clone();
        let ns_id = self.ns_id;
        tokio::task::spawn_blocking(move || db.inner.delete_ns_no_wal(ns_id, &key))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    pub async fn iter(&self) -> Result<Vec<KeyValue>> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || store.scan_range_batch(&[], None))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    pub async fn keys(&self) -> Result<Vec<Vec<u8>>> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || store.keys())
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    pub async fn range(&self, start: Vec<u8>, end: Option<Vec<u8>>) -> Result<Vec<KeyValue>> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || store.scan_range_batch(&start, end.as_deref()))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    pub async fn scan_prefix(&self, prefix: Vec<u8>) -> Result<Vec<KeyValue>> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || store.scan_prefix_batch(&prefix))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// Paginated cursor scan. See [`Db::scan`] for details.
    pub async fn scan(&self, cursor: Option<Vec<u8>>, end: Option<Vec<u8>>, limit: usize) -> Result<ScanPage> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || store.scan_page_batch(cursor.as_deref(), end.as_deref(), limit))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    pub fn stats(&self) -> Stats {
        self.store.stats()
    }

    pub async fn garbage_collect(&self) -> Result<GCStats> {
        let db = self.db.clone();
        let ns_id = self.ns_id;
        tokio::task::spawn_blocking(move || db.inner.garbage_collect_namespace(ns_id))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    // ── Typed CRUD (rkyv ser/de) ──────────────────────────────────────

    /// Insert a typed key-value pair, serialized via rkyv.
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
    pub async fn delete_typed<K>(&self, key: &K) -> Result<()>
    where
        K: for<'a> rkyv::Serialize<HighSerializer<rkyv::util::AlignedVec, ArenaHandle<'a>, RkyvError>>,
    {
        let kb = rkyv_serialize(key)?;
        self.delete(kb).await
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
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
    use tempfile::TempDir;

    /// Config for facade tests: keeps the eager per-namespace fd footprint small
    /// so the suite survives high `cargo test` parallelism. See
    /// [`crate::support::TEST_NUM_BUCKETS`].
    fn test_config() -> DbConfig {
        DbConfig {
            num_buckets: crate::support::TEST_NUM_BUCKETS,
            ..DbConfig::default()
        }
    }

    #[test]
    fn test_db_open_and_crud() {
        let dir = TempDir::new().unwrap();
        let db = Db::open_with_config(dir.path(), test_config()).unwrap();

        db.put(b"k1", b"v1").unwrap();
        db.put(b"k2", b"v2").unwrap();

        assert_eq!(db.get(b"k1").unwrap(), Some(b"v1".to_vec()));
        assert_eq!(db.get(b"k2").unwrap(), Some(b"v2".to_vec()));
        assert_eq!(db.get(b"k3").unwrap(), None);

        db.delete(b"k1").unwrap();
        assert_eq!(db.get(b"k1").unwrap(), None);

        db.shutdown().unwrap();
    }

    #[test]
    fn test_db_iteration() {
        let dir = TempDir::new().unwrap();
        let db = Db::open_with_config(dir.path(), test_config()).unwrap();

        db.put(b"a", b"1").unwrap();
        db.put(b"b", b"2").unwrap();
        db.put(b"c", b"3").unwrap();

        let all = db.iter().unwrap();
        assert_eq!(all.len(), 3);
        assert_eq!(all[0], (b"a".to_vec(), b"1".to_vec()));
        assert_eq!(all[1], (b"b".to_vec(), b"2".to_vec()));
        assert_eq!(all[2], (b"c".to_vec(), b"3".to_vec()));

        let keys = db.keys().unwrap();
        assert_eq!(keys, vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()]);

        db.shutdown().unwrap();
    }

    #[test]
    fn test_db_range() {
        let dir = TempDir::new().unwrap();
        let db = Db::open_with_config(dir.path(), test_config()).unwrap();

        db.put(b"a", b"1").unwrap();
        db.put(b"b", b"2").unwrap();
        db.put(b"c", b"3").unwrap();
        db.put(b"d", b"4").unwrap();

        let range = db.range(b"b", Some(b"d")).unwrap();
        assert_eq!(range.len(), 2);
        assert_eq!(range[0].0, b"b".to_vec());
        assert_eq!(range[1].0, b"c".to_vec());

        db.shutdown().unwrap();
    }

    #[test]
    fn test_db_scan_prefix() {
        let dir = TempDir::new().unwrap();
        let db = Db::open_with_config(dir.path(), test_config()).unwrap();

        db.put(b"user:1", b"alice").unwrap();
        db.put(b"user:2", b"bob").unwrap();
        db.put(b"order:1", b"item").unwrap();

        let users = db.scan_prefix(b"user:").unwrap();
        assert_eq!(users.len(), 2);
        assert_eq!(users[0].0, b"user:1".to_vec());

        db.shutdown().unwrap();
    }

    #[test]
    fn test_db_namespaces() {
        let dir = TempDir::new().unwrap();
        let db = Db::open_with_config(dir.path(), test_config()).unwrap();

        // Default namespace
        db.put(b"global", b"value").unwrap();

        // Create and use a namespace
        let users = db.namespace("users").unwrap();
        users.put(b"u1", b"alice").unwrap();
        users.put(b"u2", b"bob").unwrap();

        // Isolation: default doesn't see namespace data
        assert_eq!(db.get(b"u1").unwrap(), None);
        assert_eq!(users.get(b"global").unwrap(), None);

        // Namespace data
        assert_eq!(users.get(b"u1").unwrap(), Some(b"alice".to_vec()));

        // Namespace iteration
        let keys = users.keys().unwrap();
        assert_eq!(keys.len(), 2);

        // List namespaces
        let ns_list = db.list_namespaces();
        assert!(ns_list.len() >= 2); // default + users

        db.shutdown().unwrap();
    }

    #[test]
    fn test_per_namespace_ops_metrics() {
        let dir = TempDir::new().unwrap();
        let db = Db::open_with_config(dir.path(), test_config()).unwrap();

        let a = db.namespace("a").unwrap();
        a.put(b"k1", b"v").unwrap();
        a.put(b"k2", b"v").unwrap();
        let b = db.namespace("b").unwrap();
        b.put(b"k1", b"v").unwrap();

        // Counters are isolated per namespace.
        assert_eq!(db.ops_metrics_for("a").unwrap().puts, 2);
        assert_eq!(db.ops_metrics_for("b").unwrap().puts, 1);

        // Engine-wide view is the sum across namespaces.
        assert_eq!(db.ops_metrics().puts, 3);

        // by-namespace map carries each namespace's own counters.
        let by_ns: std::collections::HashMap<String, _> = db.ops_metrics_by_namespace().into_iter().collect();
        assert_eq!(by_ns["a"].puts, 2);
        assert_eq!(by_ns["b"].puts, 1);

        // Unknown namespace is an error, not a zero snapshot.
        assert!(db.ops_metrics_for("missing").is_err());

        // Dropping a namespace folds its totals into the global accumulator, so
        // the engine aggregate stays monotonic, but the namespace is no longer
        // individually queryable.
        db.remove_namespace("a").unwrap();
        assert_eq!(db.ops_metrics().puts, 3, "engine total stays monotonic after drop");
        assert!(db.ops_metrics_for("a").is_err(), "dropped ns is no longer queryable");
        assert!(!db.ops_metrics_by_namespace().iter().any(|(n, _)| n == "a"));

        db.shutdown().unwrap();
    }

    #[test]
    fn test_remove_namespace_deletes_storage_and_survives_reopen() {
        let dir = TempDir::new().unwrap();
        let doomed_dir = dir.path().join("ns_doomed");
        let keeper_dir = dir.path().join("ns_keeper");

        {
            let db = Db::open_with_config(dir.path(), test_config()).unwrap();

            // Two namespaces so we can confirm the survivor is untouched.
            let doomed = db.namespace("doomed").unwrap();
            doomed.put(b"k1", b"v1").unwrap();
            doomed.put(b"k2", b"v2").unwrap();
            let keeper = db.namespace("keeper").unwrap();
            keeper.put(b"alive", b"yes").unwrap();

            assert!(doomed_dir.exists(), "data dir should exist before removal");

            db.remove_namespace("doomed").unwrap();

            // On-disk storage for the removed namespace is gone; the survivor remains.
            assert!(!doomed_dir.exists(), "data dir should be deleted on removal");
            assert!(keeper_dir.exists(), "survivor data dir must remain");
            assert!(!db.list_namespaces().iter().any(|(n, _)| n == "doomed"));

            db.shutdown().unwrap();
        }

        // Reopen: recovery must not resurrect the removed namespace, and the
        // surviving namespace's data must still be intact.
        let db = Db::open_with_config(dir.path(), test_config()).unwrap();
        assert!(!db.list_namespaces().iter().any(|(n, _)| n == "doomed"));
        let keeper = db.namespace("keeper").unwrap();
        assert_eq!(keeper.get(b"alive").unwrap(), Some(b"yes".to_vec()));
        db.shutdown().unwrap();
    }

    #[test]
    fn test_db_namespace_gc() {
        let dir = TempDir::new().unwrap();
        let db = Db::open_with_config(dir.path(), test_config()).unwrap();

        let ns = db.namespace("test_ns").unwrap();
        ns.put(b"k", b"v").unwrap();
        let gc_stats = ns.garbage_collect().unwrap();
        assert_eq!(gc_stats.bytes_reclaimed, 0);

        db.shutdown().unwrap();
    }

    #[test]
    fn test_db_maintenance() {
        let dir = TempDir::new().unwrap();
        let db = Db::open_with_config(dir.path(), test_config()).unwrap();

        db.put(b"k", b"v").unwrap();

        let _stats = db.stats();
        let _ratio = db.waste_ratio();
        let _gc = db.garbage_collect().unwrap();
        let _compact = db.compact();

        db.shutdown().unwrap();
    }

    #[tokio::test]
    async fn test_async_db_crud() {
        let dir = TempDir::new().unwrap();
        let db = AsyncDb::open_with_config(dir.path().to_path_buf(), test_config()).await.unwrap();

        db.put(b"k1".to_vec(), b"v1".to_vec()).await.unwrap();
        assert_eq!(db.get(b"k1".to_vec()).await.unwrap(), Some(b"v1".to_vec()));

        db.delete(b"k1".to_vec()).await.unwrap();
        assert_eq!(db.get(b"k1".to_vec()).await.unwrap(), None);

        db.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_async_db_namespace() {
        let dir = TempDir::new().unwrap();
        let db = AsyncDb::open_with_config(dir.path().to_path_buf(), test_config()).await.unwrap();

        let ns = db.namespace("events".to_string()).await.unwrap();
        ns.put(b"e1".to_vec(), b"click".to_vec()).await.unwrap();
        assert_eq!(ns.get(b"e1".to_vec()).await.unwrap(), Some(b"click".to_vec()));

        // Default namespace should not see it
        assert_eq!(db.get(b"e1".to_vec()).await.unwrap(), None);

        db.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_async_namespace_no_wal_crud() {
        let dir = TempDir::new().unwrap();
        let db = AsyncDb::open_with_config(dir.path().to_path_buf(), test_config()).await.unwrap();
        let ns = db.namespace("cache".to_string()).await.unwrap();

        // No-WAL write is readable like any other write.
        ns.put_no_wal(b"q1".to_vec(), b"emb1".to_vec()).await.unwrap();
        assert_eq!(ns.get(b"q1".to_vec()).await.unwrap(), Some(b"emb1".to_vec()));

        // No-WAL delete removes it.
        ns.delete_no_wal(b"q1".to_vec()).await.unwrap();
        assert_eq!(ns.get(b"q1".to_vec()).await.unwrap(), None);

        // No-WAL delete tombstones a key written *with* the WAL too.
        ns.put(b"q2".to_vec(), b"emb2".to_vec()).await.unwrap();
        ns.delete_no_wal(b"q2".to_vec()).await.unwrap();
        assert_eq!(ns.get(b"q2".to_vec()).await.unwrap(), None);

        // The no-WAL paths bump their own counters and skip the WAL fsync path:
        // one WAL-backed put (q2) accounts for the only wal_fsync from these ops.
        let m = db.ops_metrics();
        assert_eq!(m.no_wal_puts, 1, "one put_no_wal");
        assert_eq!(m.no_wal_deletes, 2, "two delete_no_wal calls");

        db.shutdown().await.unwrap();
    }

    // ── Typed API tests ───────────────────────────────────────────────

    #[derive(Debug, Clone, PartialEq, Archive, RkyvSerialize, RkyvDeserialize)]
    struct UserId(u64);

    #[derive(Debug, Clone, PartialEq, Archive, RkyvSerialize, RkyvDeserialize)]
    struct UserProfile {
        name: String,
        age: u32,
    }

    #[test]
    fn test_db_typed_crud() {
        let dir = TempDir::new().unwrap();
        let db = Db::open_with_config(dir.path(), test_config()).unwrap();

        let key = UserId(42);
        let value = UserProfile {
            name: "Alice".to_string(),
            age: 30,
        };

        db.put_typed(&key, &value).unwrap();

        let got: Option<UserProfile> = db.get_typed::<UserId, UserProfile>(&key).unwrap();
        assert_eq!(got, Some(value.clone()));

        // Update
        let updated = UserProfile {
            name: "Alice".to_string(),
            age: 31,
        };
        db.put_typed(&key, &updated).unwrap();
        let got: Option<UserProfile> = db.get_typed(&key).unwrap();
        assert_eq!(got, Some(updated));

        // Delete
        db.delete_typed(&key).unwrap();
        let got: Option<UserProfile> = db.get_typed(&key).unwrap();
        assert_eq!(got, None);

        db.shutdown().unwrap();
    }

    #[test]
    fn test_namespace_typed_crud() {
        let dir = TempDir::new().unwrap();
        let db = Db::open_with_config(dir.path(), test_config()).unwrap();

        let ns = db.namespace("users").unwrap();
        let key = UserId(1);
        let value = UserProfile {
            name: "Bob".to_string(),
            age: 25,
        };

        ns.put_typed(&key, &value).unwrap();
        let got: Option<UserProfile> = ns.get_typed(&key).unwrap();
        assert_eq!(got, Some(value));

        // Isolation: default namespace shouldn't have it
        let from_default: Option<UserProfile> = db.get_typed(&key).unwrap();
        assert_eq!(from_default, None);

        db.shutdown().unwrap();
    }

    #[tokio::test]
    async fn test_async_db_typed_crud() {
        let dir = TempDir::new().unwrap();
        let db = AsyncDb::open_with_config(dir.path().to_path_buf(), test_config()).await.unwrap();

        let key = UserId(99);
        let value = UserProfile {
            name: "Charlie".to_string(),
            age: 40,
        };

        db.put_typed(&key, &value).await.unwrap();
        let got: Option<UserProfile> = db.get_typed(&key).await.unwrap();
        assert_eq!(got, Some(value));

        db.delete_typed(&key).await.unwrap();
        let got: Option<UserProfile> = db.get_typed(&key).await.unwrap();
        assert_eq!(got, None);

        db.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_async_namespace_typed_crud() {
        let dir = TempDir::new().unwrap();
        let db = AsyncDb::open_with_config(dir.path().to_path_buf(), test_config()).await.unwrap();

        let ns = db.namespace("products".to_string()).await.unwrap();
        let key = 42u64;
        let value = "Widget".to_string();

        ns.put_typed(&key, &value).await.unwrap();
        let got: Option<String> = ns.get_typed(&key).await.unwrap();
        assert_eq!(got, Some("Widget".to_string()));

        db.shutdown().await.unwrap();
    }

    // ── Typed Iteration tests ─────────────────────────────────────────

    #[test]
    fn test_db_typed_iter() {
        let dir = TempDir::new().unwrap();
        let db = Db::open_with_config(dir.path(), test_config()).unwrap();

        db.put_typed(
            &UserId(1),
            &UserProfile {
                name: "Alice".into(),
                age: 30,
            },
        )
        .unwrap();
        db.put_typed(&UserId(2), &UserProfile { name: "Bob".into(), age: 25 }).unwrap();
        db.put_typed(
            &UserId(3),
            &UserProfile {
                name: "Charlie".into(),
                age: 35,
            },
        )
        .unwrap();

        // iter_typed
        let all: Vec<(UserId, UserProfile)> = db.iter_typed().unwrap();
        assert_eq!(all.len(), 3);
        assert_eq!(all[0].1.name, "Alice");

        // keys_typed
        let keys: Vec<UserId> = db.keys_typed().unwrap();
        assert_eq!(keys.len(), 3);

        db.shutdown().unwrap();
    }

    #[test]
    fn test_namespace_typed_iter() {
        let dir = TempDir::new().unwrap();
        let db = Db::open_with_config(dir.path(), test_config()).unwrap();

        let ns = db.namespace("users").unwrap();
        ns.put_typed(&UserId(10), &UserProfile { name: "Dan".into(), age: 20 }).unwrap();
        ns.put_typed(&UserId(20), &UserProfile { name: "Eve".into(), age: 28 }).unwrap();

        let all: Vec<(UserId, UserProfile)> = ns.iter_typed().unwrap();
        assert_eq!(all.len(), 2);

        let keys: Vec<UserId> = ns.keys_typed().unwrap();
        assert_eq!(keys.len(), 2);

        db.shutdown().unwrap();
    }

    #[tokio::test]
    async fn test_async_db_typed_iter() {
        let dir = TempDir::new().unwrap();
        let db = AsyncDb::open_with_config(dir.path().to_path_buf(), test_config()).await.unwrap();

        db.put_typed(
            &UserId(1),
            &UserProfile {
                name: "Alice".into(),
                age: 30,
            },
        )
        .await
        .unwrap();
        db.put_typed(&UserId(2), &UserProfile { name: "Bob".into(), age: 25 }).await.unwrap();

        let all: Vec<(UserId, UserProfile)> = db.iter_typed().await.unwrap();
        assert_eq!(all.len(), 2);

        let keys: Vec<UserId> = db.keys_typed().await.unwrap();
        assert_eq!(keys.len(), 2);

        db.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_async_namespace_typed_iter() {
        let dir = TempDir::new().unwrap();
        let db = AsyncDb::open_with_config(dir.path().to_path_buf(), test_config()).await.unwrap();

        let ns = db.namespace("items".to_string()).await.unwrap();
        ns.put_typed(&1u64, &"first".to_string()).await.unwrap();
        ns.put_typed(&2u64, &"second".to_string()).await.unwrap();

        let all: Vec<(u64, String)> = ns.iter_typed().await.unwrap();
        assert_eq!(all.len(), 2);

        let keys: Vec<u64> = ns.keys_typed().await.unwrap();
        assert_eq!(keys.len(), 2);

        db.shutdown().await.unwrap();
    }
}
