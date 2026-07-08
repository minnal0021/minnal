//! DocStore — document store built on top of minnal_db.
//!
//! A `DocStore` manages a collection of named document stores, each backed by
//! one minnal_db namespace.  Documents are stored as UTF-8 JSON values and
//! retrieved by a typed ID (`u64`, `u128`, or UUID as `u128`).
//!
//! # Lifecycle
//!
//! ```text
//! DocStore::open(db_path, schema_dir)   ← opens existing stores
//!   .create(schema)                     ← define a new store
//!   .put("users", id, doc)              ← write a document
//!   .get("users", id)            ← read by primary key
//!   .query("users", "status = \"active\"") ← index query
//!   .add_index("users", spec)           ← add an index (background build)
//!   .drop_index("users", "status")      ← remove an index
//!   .amend("users", amendment)          ← add/remove non-index attributes
//!   .remove("users")                ← destroy everything
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use log::{debug, error, info, warn};
use minnal_db::{AsyncDb, DbConfig, ExtractorFn, FieldId, IndexValue, IndexValueType, RowIdFn, RowToKeyFn};
use semantic_search::ClusterIndex;
use semantic_search::service::SemanticSearchConfig;

use crate::error::{DocStoreError, SchemaError};
use crate::hex::hex_to_bytes;
use crate::index_observer::{ChainedObserver, DiskProgress, InMemoryProgress, IndexProgressObserver};
use crate::index_progress::{BuildStatus, now_ms};
use crate::pagination::{CursorPage, Page, Pagination, prefix_upper_bound};
use crate::schema::{DocStoreSchema, IndexSpec, IndexType, KeyType, KvStoreSchema, SchemaAmendment, StoreType};
use crate::vec_index_worker::{VecIndexWorker, VecIndexWorkerHandle, VectorIndexConfig};
use crate::vector_kv;

// ── ID type ───────────────────────────────────────────────────────────────────

/// A document identifier, typed to match the [`KeyType`] of the store.
///
/// Keys are stored in big-endian byte order so that lexicographic range
/// scans correspond to numeric ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DocId {
    /// 128-bit UUID represented as a `u128`.
    Uuid(u128),
    /// Unsigned 64-bit integer.
    U64(u64),
    /// Unsigned 128-bit integer.
    U128(u128),
}

impl DocId {
    /// Serialize this ID to big-endian bytes (the raw database key).
    pub fn to_bytes(self) -> Vec<u8> {
        match self {
            DocId::Uuid(v) | DocId::U128(v) => v.to_be_bytes().to_vec(),
            DocId::U64(v) => v.to_be_bytes().to_vec(),
        }
    }

    /// Deserialize bytes back to a `DocId` given the store's [`KeyType`].
    pub fn from_bytes(bytes: &[u8], key_type: KeyType) -> Result<Self, DocStoreError> {
        match key_type {
            KeyType::Uuid => {
                let arr: [u8; 16] = bytes
                    .try_into()
                    .map_err(|_| DocStoreError::InvalidId(format!("expected 16 bytes for UUID key, got {}", bytes.len())))?;
                Ok(DocId::Uuid(u128::from_be_bytes(arr)))
            }
            KeyType::U64 => {
                let arr: [u8; 8] = bytes
                    .try_into()
                    .map_err(|_| DocStoreError::InvalidId(format!("expected 8 bytes for u64 key, got {}", bytes.len())))?;
                Ok(DocId::U64(u64::from_be_bytes(arr)))
            }
            KeyType::U128 => {
                let arr: [u8; 16] = bytes
                    .try_into()
                    .map_err(|_| DocStoreError::InvalidId(format!("expected 16 bytes for u128 key, got {}", bytes.len())))?;
                Ok(DocId::U128(u128::from_be_bytes(arr)))
            }
        }
    }
}

// ── Index build progress ──────────────────────────────────────────────────────

/// Persistent state for a background index build, written to
/// `{db_path}/index/{ns_id}/{field_id}/build_progress.json`.
///
/// Survives server restarts — on startup the store uses this to resume
/// interrupted builds instead of restarting from scratch.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DiskBuildProgress {
    /// `"in_progress"`, `"complete"`, or `"failed"`.
    pub status: String,
    /// Total document count (0 until the initial scan finishes).
    pub total: u64,
    /// Documents processed so far.
    pub indexed: u64,
    /// Hex-encoded bytes of the last successfully processed key.
    /// `None` if no key has been processed yet.
    pub last_key_hex: Option<String>,
    /// Error message if `status == "failed"`.
    pub error: Option<String>,
}

/// Path to the build-progress file for `(ns_id, field_id)`.
fn build_progress_path(db_path: &Path, ns_id: u32, field_id: FieldId) -> PathBuf {
    db_path
        .join("index")
        .join(ns_id.to_string())
        .join(field_id.to_string())
        .join("build_progress.json")
}

/// Read the persisted build progress for a field.  Returns `None` if the file
/// does not exist or cannot be parsed.
fn read_disk_progress(path: &Path) -> Option<DiskBuildProgress> {
    let bytes = std::fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

// ── Vector-index reindex ─────────────────────────────────────────────────────

/// Persistent record for one `index_all` reindex, written to
/// `{db_path}/index/{ns_id}/vector_reindex.json`.
///
/// A reindex is the unit of work created by a single `index_all` call: it
/// tracks the total documents enqueued, how many have been indexed, and the
/// lifecycle status.  The record survives restarts so that the progress API can
/// show historical reindexs even after the server restarts.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VecReindexProgress {
    /// `"running"`, `"complete"`, or `"failed"`.
    pub status: String,
    /// Unix milliseconds when `index_all` was called.
    pub started_at_ms: u64,
    /// Unix milliseconds when the reindex finished (terminal states only).
    pub completed_at_ms: Option<u64>,
    /// Number of documents enqueued by this reindex.
    pub total_enqueued: usize,
    /// Number of exhausted entries cleared before enqueueing.
    pub exhausted_cleared: usize,
    /// Error message when `status == "failed"`.
    pub error: Option<String>,
}

fn vec_reindex_path(db_path: &Path, ns_id: u32) -> PathBuf {
    db_path.join("index").join(ns_id.to_string()).join("vector_reindex.json")
}

fn read_vec_reindex(path: &Path) -> Option<VecReindexProgress> {
    let bytes = std::fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn write_vec_reindex(path: &Path, reindex: &VecReindexProgress) {
    let tmp = path.with_extension("tmp");
    if let Ok(bytes) = serde_json::to_vec(reindex) {
        let _ = std::fs::write(&tmp, &bytes);
        let _ = std::fs::rename(&tmp, path);
    }
}

/// A snapshot of the progress of a background index build.
#[derive(Debug, Clone)]
pub struct IndexBuildProgress {
    /// Total number of documents to be indexed (0 until the scan begins).
    pub total: u64,
    /// Number of documents processed so far.
    pub indexed: u64,
    /// `true` once the build has finished (successfully or with an error).
    pub done: bool,
    /// Non-`None` if the build failed.
    pub error: Option<String>,
}

/// Handle returned by [`DocStore::add_index`] or [`DocStore::resume_pending_builds`].
///
/// Use [`progress`] to poll status or [`wait`] to block until complete.
///
/// [`progress`]: IndexBuildHandle::progress
/// [`wait`]: IndexBuildHandle::wait
pub struct IndexBuildHandle {
    /// The namespace this build belongs to.
    pub namespace: String,
    /// The field being indexed.
    pub field: String,
    /// Live progress counters shared with the observer inside the build task.
    pub mem: Arc<InMemoryProgress>,
    task: tokio::task::JoinHandle<Result<(), DocStoreError>>,
}

impl IndexBuildHandle {
    /// Return a snapshot of the current build progress.
    pub fn progress(&self) -> IndexBuildProgress {
        IndexBuildProgress {
            total: self.mem.total.load(Ordering::Relaxed),
            indexed: self.mem.indexed.load(Ordering::Relaxed),
            done: self.mem.done.load(Ordering::Relaxed),
            error: self.mem.error.lock().unwrap().clone(),
        }
    }

    /// Await the build task and return its result.
    pub async fn wait(self) -> Result<(), DocStoreError> {
        self.task.await.map_err(|e| DocStoreError::BuildFailed(e.to_string()))?
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Convert an [`IndexType`] to the equivalent [`AttributeType`].
fn index_type_to_attr_type(t: IndexType) -> crate::schema::AttributeType {
    match t {
        IndexType::Bool => crate::schema::AttributeType::Bool,
        IndexType::Int => crate::schema::AttributeType::Int,
        IndexType::Str => crate::schema::AttributeType::Str,
    }
}

/// Convert an [`IndexType`] to the minnal_db [`IndexValueType`].
fn to_ivt(t: IndexType) -> IndexValueType {
    match t {
        IndexType::Bool => IndexValueType::Bool,
        IndexType::Int => IndexValueType::Int,
        IndexType::Str => IndexValueType::Str,
    }
}

/// Build a JSON-field extractor closure for a given field name and type.
fn json_extractor(field: String, index_type: IndexType) -> ExtractorFn {
    Arc::new(move |bytes: &[u8]| {
        let v: serde_json::Value = serde_json::from_slice(bytes).ok()?;
        match index_type {
            IndexType::Bool => v.get(&field)?.as_bool().map(IndexValue::Bool),
            IndexType::Int => v.get(&field)?.as_i64().map(IndexValue::Int),
            IndexType::Str => v.get(&field)?.as_str().map(|s| IndexValue::Str(s.to_string())),
        }
    })
}

/// Concatenate the specified embedding fields from a document into a single
/// text string for embedding.  Each field contributes `"field_name: value\n"`.
/// Fields that are absent or not strings are silently skipped.
fn build_embedding_text(doc: &serde_json::Value, fields: &[String]) -> String {
    fields
        .iter()
        .filter_map(|f| doc.get(f)?.as_str().map(|v| format!("{f}: {v}")))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Activate all field indices defined in `schema` for namespace `ns_id`.
///
/// Also registers a [`RowIdFn`] + [`RowToKeyFn`] pair for numeric key types
/// (`U64`, `U128`, `Uuid`) so that index queries resolve matching keys in
/// O(|hits|) rather than the O(n_keys) fallback scan.
///
/// Called once on open and once after creating a new store.
async fn activate_indices(db: &AsyncDb, ns_id: u32, schema: &DocStoreSchema) -> Result<(), DocStoreError> {
    // Register the row-ID functions FIRST — before activating any field index.
    //
    // `activate_field_index` replays the WAL tail and resolves each affected
    // key's row ID through the store's *current* resolver. The resolver
    // precedence is RowIdFn > dense RowMap > legacy hash, so if the custom
    // key-derived RowIdFn is not yet installed, replay would fall back to the
    // RowMap and index those WAL-tail keys under dense IDs that disagree with
    // the key-derived IDs used by prior persisted entries and all future writes
    // — mixing two row-ID schemes in one field index (stale hits, wrong key
    // resolution). The engine documents this ordering requirement on
    // `Database::set_row_id_fn`; honour it here.
    //
    // The functions exist for key types whose raw bytes are injective into u128,
    // which also gives O(|hits|) key resolution in index queries.
    let (row_id_fn, row_to_key_fn): (RowIdFn, RowToKeyFn) = match schema.key_type {
        KeyType::U64 => (
            std::sync::Arc::new(|k: &[u8]| {
                let arr: [u8; 8] = k[..8].try_into().unwrap_or_default();
                u64::from_be_bytes(arr) as u128
            }),
            std::sync::Arc::new(|id: u128| (id as u64).to_be_bytes().to_vec()),
        ),
        KeyType::U128 | KeyType::Uuid => (
            std::sync::Arc::new(|k: &[u8]| {
                let arr: [u8; 16] = k[..16].try_into().unwrap_or_default();
                u128::from_be_bytes(arr)
            }),
            std::sync::Arc::new(|id: u128| id.to_be_bytes().to_vec()),
        ),
    };
    db.set_row_id_fn(ns_id, row_id_fn, Some(row_to_key_fn)).await.map_err(DocStoreError::Db)?;

    for spec in &schema.indices {
        let ivt = to_ivt(spec.index_type);
        // register_index_field is idempotent — returns existing field_id on restart
        let field_id = db.register_index_field(ns_id, &spec.field, ivt)?;
        let extractor = json_extractor(spec.field.clone(), spec.index_type);
        db.activate_field_index(ns_id, field_id, ivt, extractor).await?;
    }

    Ok(())
}

// ── SemanticSearchContext ─────────────────────────────────────────────────────

/// Semantic-search configuration attached to a [`DocStore`].
///
/// When present, writes to namespaces with `semantic_search_enabled = true`
/// enqueue a pending embedding job instead of calling the embedding service
/// inline.  A background `VecIndexWorker` processes the queue
/// asynchronously, making the write path independent of embedding service
/// availability.
///
/// Construct once at startup and attach via [`DocStore::with_semantic_search`].
pub struct SemanticSearchContext {
    /// Embedding service configuration (URL, model, dimensions).
    pub config: SemanticSearchConfig,
    /// IVF cluster centroids, probed by exact nearest-centroid distance.
    pub cluster_index: Arc<ClusterIndex>,
}

// ── ReindexStats ─────────────────────────────────────────────────────────────

/// Outcome of a single-document vector reindex
/// ([`DocStore::reindex_doc_vector`] / [`DocStore::kv_reindex_doc_vector`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorReindexOutcome {
    /// The document was (re-)enqueued for embedding.
    Enqueued,
    /// No document/value exists for the given id.
    NotFound,
    /// The document produced no embedding text, so nothing was enqueued.
    SkippedEmptyText,
}

/// Result of a [`DocStore::index_all`] call.
#[derive(Debug, Clone, Copy)]
pub struct ReindexStats {
    /// Number of exhausted queue entries (retry_count ≥ max_retries) that were
    /// removed before re-enqueueing.
    pub exhausted_cleared: usize,
    /// Number of documents enqueued for re-embedding.
    pub enqueued: usize,
}

// ── DocStore ──────────────────────────────────────────────────────────────────

// The two kinds of secondary index maintained by a document store.

/// A document store manager backed by a `minnal_db` database.
///
/// Each logical document store is one minnal_db namespace with a JSON schema
/// persisted in `schema_dir`.  `DocStore` owns the `AsyncDb` instance.
pub struct DocStore {
    db: Arc<AsyncDb>,
    db_path: PathBuf,
    schema_dir: PathBuf,
    lock_path: PathBuf,
    /// Semantic-search context used by query paths (embedding + cluster index).
    /// `None` when semantic search is not configured.
    semantic_ctx: Option<Arc<SemanticSearchContext>>,
    /// Wake signal sent to the [`VecIndexWorker`] after each write that
    /// enqueues a pending embedding.  `None` when semantic search is not
    /// configured.
    notify: Option<Arc<tokio::sync::Notify>>,
    /// Handle to the background vector-index worker.  `None` when semantic
    /// search is not configured.  Wrapped in a `Mutex` so the handle can be
    /// taken out for an async graceful shutdown while `DocStore` is behind
    /// an `Arc`.
    worker_handle: std::sync::Mutex<Option<VecIndexWorkerHandle>>,
    /// Configuration for the vector-index background worker.
    /// Set via [`DocStore::with_vector_index_config`] before calling
    /// [`DocStore::with_semantic_search`]; defaults are used otherwise.
    vector_index_config: VectorIndexConfig,
}

impl DocStore {
    // ── Construction ──────────────────────────────────────────────────────

    /// Open (or create) a `DocStore` at `db_path` with a custom [`DbConfig`].
    ///
    /// Prefer this over [`open`] when you need to tune the underlying engine
    /// (e.g. sync policy, GC intervals, WAL segment size).  For every schema
    /// file found, the corresponding namespace is opened and all field indices
    /// are activated automatically.
    ///
    /// [`open`]: DocStore::open
    pub async fn open_with_config(db_path: impl AsRef<Path>, schema_dir: impl AsRef<Path>, config: DbConfig) -> Result<Self, DocStoreError> {
        let db_path = db_path.as_ref().to_path_buf();
        let schema_dir = schema_dir.as_ref().to_path_buf();
        info!("opening doc store at {}", db_path.display());
        std::fs::create_dir_all(&db_path)?;
        std::fs::create_dir_all(&schema_dir)?;

        let lock_path = db_path.join(".lock");
        if lock_path.exists() {
            return Err(DocStoreError::StoreLocked { path: db_path });
        }
        std::fs::write(&lock_path, "")?;

        let db = Arc::new(AsyncDb::open_with_config(db_path.clone(), config.clone()).await?);
        let store = Self {
            db,
            db_path,
            schema_dir,
            lock_path,
            semantic_ctx: None,
            notify: None,
            worker_handle: std::sync::Mutex::new(None),
            vector_index_config: VectorIndexConfig::default(),
        };

        let schemas = store.load_all_schemas()?;
        info!("loaded {} schema(s)", schemas.len());
        for schema in schemas {
            if let Some(ns_id) = schema.ns_id {
                debug!("activating indices for namespace '{}' (ns_id={})", schema.namespace, ns_id);
                activate_indices(&store.db, ns_id, &schema).await?;
            }
        }

        let kv_schemas = store.load_all_kv_schemas()?;
        info!("loaded {} KV schema(s)", kv_schemas.len());
        for schema in kv_schemas {
            if schema.ns_id.is_some() {
                debug!("opening KV namespace '{}'", schema.namespace);
                store.db.namespace(schema.namespace.clone()).await?;
            }
        }

        // Start background workers after all indices are activated so the
        // index checkpoint worker's first immediate tick captures a complete
        // (not empty) index state.
        store.db.enable_all_workers(&config).await.map_err(DocStoreError::Db)?;

        info!("doc store ready");
        Ok(store)
    }

    /// Open (or create) a `DocStore` at `db_path` with default [`DbConfig`].
    ///
    /// For every schema file found, the corresponding namespace is opened and
    /// all field indices are activated automatically — no call to
    /// `register_index_field` is needed after the first `create`.
    pub async fn open(db_path: impl AsRef<Path>, schema_dir: impl AsRef<Path>) -> Result<Self, DocStoreError> {
        Self::open_with_config(db_path, schema_dir, DbConfig::default()).await
    }

    /// Set the tuning parameters for the background vector-index worker.
    ///
    /// Must be called **before** [`with_semantic_search`]; configuration set
    /// after the worker has already started has no effect.
    ///
    /// [`with_semantic_search`]: DocStore::with_semantic_search
    pub fn with_vector_index_config(mut self, config: VectorIndexConfig) -> Self {
        self.vector_index_config = config;
        self
    }

    /// Attach a [`SemanticSearchContext`] and start the background vector-index
    /// worker.
    ///
    /// After this call:
    /// - `put` / `kv_put` on semantic-search-enabled namespaces enqueue a
    ///   pending embedding job (atomic, WAL-backed) instead of calling the
    ///   embedding service inline.
    /// - A `VecIndexWorker` drains the queue in the background using the
    ///   parameters from [`with_vector_index_config`] (or built-in defaults).
    /// - `search_semantic` / `kv_search_semantic` still embed query text
    ///   synchronously (low count, TTL-cached).
    /// - A one-shot **vector-index reconciliation** is spawned as a background
    ///   task: it re-enqueues any document missing both a committed vector index
    ///   entry and a pending queue entry — closing the `put` / `kv_put` crash
    ///   window and the `put_no_wal` vector-write window (a crash before the
    ///   memtable flush drops a just-indexed vector). It runs asynchronously so
    ///   startup is not blocked; if it fails it logs an error and the operator
    ///   can re-run it via `POST /admin/indices/vector/reconcile`. A count
    ///   short-circuit makes a clean boot cheap.
    ///
    /// Call [`shutdown_vec_index_worker`] for a clean stop before dropping the
    /// store.
    ///
    /// [`with_vector_index_config`]: DocStore::with_vector_index_config
    /// [`shutdown_vec_index_worker`]: DocStore::shutdown_vec_index_worker
    pub fn with_semantic_search(mut self, ctx: SemanticSearchContext) -> Self {
        let ctx = Arc::new(ctx);
        let notify = Arc::new(tokio::sync::Notify::new());
        let handle = VecIndexWorker::start(
            Arc::clone(&self.db),
            Arc::clone(&ctx),
            Arc::clone(&notify),
            self.vector_index_config.clone(),
        );

        // Spawn a one-shot startup reconciliation: re-enqueue any document whose
        // vector index was never committed — including the `put_no_wal` window
        // where a crash before the memtable flush drops a just-indexed vector.
        // Runs in the background so it never blocks startup; on failure it logs
        // an error and leaves recovery to the manual admin endpoint.
        {
            let db = Arc::clone(&self.db);
            let schema_dir = self.schema_dir.clone();
            let notify = Arc::clone(&notify);
            tokio::spawn(async move {
                info!("startup vector-index reconciliation: scanning for documents missing a vector index");
                let outcome = reconcile_all_vector_indexes(&db, &schema_dir, false).await;
                if outcome.failed > 0 {
                    error!(
                        "startup vector-index reconciliation did not fully complete ({} namespace(s) failed, {} doc(s) re-enqueued); \
                         re-run it manually via POST /admin/indices/vector/reconcile",
                        outcome.failed, outcome.reenqueued
                    );
                } else {
                    info!("startup vector-index reconciliation complete: {} doc(s) re-enqueued", outcome.reenqueued);
                }
                if outcome.reenqueued > 0 {
                    notify.notify_one();
                }
            });
        }

        self.semantic_ctx = Some(ctx);
        self.notify = Some(notify);
        *self.worker_handle.lock().unwrap() = Some(handle);
        self
    }

    /// The configured maximum number of embedding attempts per queue entry.
    ///
    /// Entries whose `retry_count` reaches this value are skipped by the worker
    /// and must be removed manually via [`delete_queue_entry`].
    ///
    /// [`delete_queue_entry`]: DocStore::delete_queue_entry
    pub fn vector_index_max_retries(&self) -> u32 {
        self.vector_index_config.max_retries
    }

    /// Signal the background vector-index worker to stop and await its exit.
    ///
    /// Call once during graceful server shutdown, before dropping the store.
    /// Any pending queue entries are preserved in the durable queue and will
    /// be processed on the next startup.
    pub async fn shutdown_vec_index_worker(&self) {
        let handle = self.worker_handle.lock().unwrap().take();
        if let Some(h) = handle {
            h.shutdown().await;
        }
    }

    /// Gracefully shut down the store.
    ///
    /// Stops the vector-index worker, then signals all background DB workers
    /// (GC, WAL GC, LSM compaction, index checkpoint, TTL) to stop and waits
    /// for each to exit cleanly.  Finally flushes all in-memory state to disk.
    ///
    /// Must be called before dropping the store to ensure a clean exit and
    /// avoid losing buffered writes or leaving the tokio runtime hanging on
    /// background tasks.
    pub async fn shutdown(&self) -> Result<(), DocStoreError> {
        self.shutdown_vec_index_worker().await;
        self.db.shutdown().await.map_err(DocStoreError::Db)
    }

    /// Number of entries currently waiting in the async vector-index queue.
    ///
    /// This is the count of documents that have been written but whose
    /// vector index entry has not yet been committed — either because the
    /// worker has not processed them yet or the embedding service is
    /// temporarily unavailable.
    pub async fn pending_vector_index_count(&self) -> usize {
        vector_kv::list_queue_entries(&self.db).await.map(|e| e.len()).unwrap_or(0)
    }

    /// Reconcile vector indexes across all semantic-search-enabled namespaces.
    ///
    /// Enqueues any document that has neither a committed vector index entry nor
    /// a pending queue entry — the `put` / `kv_put` crash window, plus the
    /// `put_no_wal` vector-write window where a crash before the memtable flush
    /// drops a just-indexed vector. A cheap count short-circuit skips namespaces
    /// that are already fully covered, so a clean run is inexpensive. Returns the
    /// number of documents re-enqueued and notifies the worker if any were.
    ///
    /// This same pass runs automatically as a background task on startup (see
    /// [`with_semantic_search`]); it is also exposed on demand over the admin
    /// REST API (`POST /admin/indices/vector/reconcile`) so an operator can
    /// re-run it — e.g. if the startup pass logged a failure.
    ///
    /// [`with_semantic_search`]: DocStore::with_semantic_search
    pub async fn reconcile_vector_indexes(&self) -> usize {
        let outcome = reconcile_all_vector_indexes(&self.db, &self.schema_dir, false).await;
        if outcome.reenqueued > 0
            && let Some(notify) = &self.notify
        {
            notify.notify_one();
        }
        outcome.reenqueued
    }

    /// Like [`reconcile_vector_indexes`](Self::reconcile_vector_indexes), but also
    /// re-enqueues documents whose committed vector bytes are **present yet corrupt**
    /// (fail to deserialize) — not just those missing a half.
    ///
    /// This deserializes every entry and skips the count short-circuit, so it is a
    /// full value-reading scan of every semantic-search namespace — run it in the
    /// background, not on a latency-sensitive request path. Per-namespace failures
    /// are logged here (`warn!` per namespace inside the pass, plus an `error!`
    /// summary when any failed). Returns the number of documents re-enqueued and
    /// notifies the worker if any were.
    pub async fn validate_and_reconcile_vector_indexes(&self) -> usize {
        let outcome = reconcile_all_vector_indexes(&self.db, &self.schema_dir, true).await;
        if outcome.failed > 0 {
            error!(
                "validating vector-index reconcile did not fully complete: {} namespace(s) failed, {} doc(s) re-enqueued",
                outcome.failed, outcome.reenqueued
            );
        }
        if outcome.reenqueued > 0
            && let Some(notify) = &self.notify
        {
            notify.notify_one();
        }
        outcome.reenqueued
    }

    /// Return the live count of documents that have a committed vector index
    /// entry in `{namespace}_sparse_vector_meta`.
    ///
    /// Unlike the LSM manifest entry counts, this reads through the active
    /// memtable so it reflects recently indexed documents immediately.  It also
    /// excludes tombstones (the engine merges deletes before returning results),
    /// so the count decreases as soon as a document's vector is removed — no
    /// need to wait for LSM compaction.
    ///
    /// Returns 0 when the companion namespace has not been created yet (i.e. no
    /// document has been indexed in this namespace).
    pub async fn count_indexed_docs(&self, namespace: &str) -> u64 {
        let meta_ns = vector_kv::sparse_vectors_meta_ns(namespace);
        match self.db.namespace(meta_ns).await {
            Ok(ns) => ns.iter().await.map(|v| v.len() as u64).unwrap_or(0),
            Err(_) => 0,
        }
    }

    /// Return all entries currently in the async vector-index queue.
    ///
    /// Includes entries that are still pending, currently being retried, and
    /// those that have exceeded the configured `max_retries` threshold and are
    /// waiting for manual removal via [`delete_queue_entry`].
    ///
    /// [`delete_queue_entry`]: DocStore::delete_queue_entry
    pub async fn list_queue_entries(&self) -> Vec<vector_kv::QueueEntry> {
        vector_kv::list_queue_entries(&self.db).await.unwrap_or_default()
    }

    /// Look up a single pending embedding queue entry by namespace and document ID.
    ///
    /// Returns `None` when no entry exists for the pair.  This is an O(1) key
    /// lookup — it does not scan the whole queue.
    pub async fn get_queue_entry(&self, namespace: &str, doc_id_bytes: &[u8]) -> Option<vector_kv::QueueEntry> {
        vector_kv::get_queue_entry(&self.db, namespace, doc_id_bytes).await.unwrap_or(None)
    }

    /// Remove a specific entry from the async vector-index queue.
    ///
    /// This is an admin operation intended for entries that have exceeded
    /// `max_retries` or that need to be manually cleared.  Returns `Ok` even
    /// if no matching entry exists.
    pub async fn delete_queue_entry(&self, namespace: &str, doc_id_bytes: &[u8]) -> Result<(), DocStoreError> {
        vector_kv::remove_queue_entry(&self.db, namespace, doc_id_bytes).await?;
        Ok(())
    }

    /// Reset the retry count to zero for a specific queue entry, allowing the
    /// worker to attempt embedding it again on its next pass.
    ///
    /// Returns the entry as it was before the reset (so the caller can inspect
    /// its previous `retry_count`), or `None` when no matching entry exists.
    /// The worker is notified immediately after the reset so processing starts
    /// without waiting for the next scheduled wake-up.
    pub async fn retry_queue_entry(&self, namespace: &str, doc_id_bytes: &[u8]) -> Result<Option<vector_kv::QueueEntry>, DocStoreError> {
        let entry = match vector_kv::get_queue_entry(&self.db, namespace, doc_id_bytes).await? {
            None => return Ok(None),
            Some(e) => e,
        };
        vector_kv::enqueue_embed(&self.db, namespace, doc_id_bytes, &entry.text).await?;
        if let Some(notify) = &self.notify {
            notify.notify_one();
        }
        Ok(Some(entry))
    }

    /// Reset the retry count to zero for every exhausted queue entry in
    /// `namespace` (those whose `retry_count` has reached `max_retries`), so
    /// they are picked up by the worker on its next pass.
    ///
    /// Returns the number of entries that were reset.  The worker is notified
    /// immediately when at least one entry was reset.
    pub async fn retry_all_failed_queue_entries(&self, namespace: &str) -> Result<usize, DocStoreError> {
        let max_retries = self.vector_index_config.max_retries;
        let entries = vector_kv::list_queue_entries(&self.db).await?;
        let exhausted: Vec<_> = entries
            .into_iter()
            .filter(|e| e.namespace == namespace && e.retry_count >= max_retries)
            .collect();
        let count = exhausted.len();
        if count == 0 {
            return Ok(0);
        }
        for entry in &exhausted {
            vector_kv::enqueue_embed(&self.db, &entry.namespace, &entry.doc_id_bytes, &entry.text).await?;
        }
        if let Some(notify) = &self.notify {
            notify.notify_one();
        }
        Ok(count)
    }

    /// Delete every entry in the async vector-index queue that belongs to
    /// `namespace`, regardless of retry count.
    ///
    /// Returns the number of entries removed.  Use this to completely drain the
    /// queue for a namespace before dropping it or when forcing a clean slate.
    pub async fn delete_all_queue_entries(&self, namespace: &str) -> Result<usize, DocStoreError> {
        let entries = vector_kv::list_queue_entries(&self.db).await?;
        let to_delete: Vec<_> = entries.into_iter().filter(|e| e.namespace == namespace).collect();
        let count = to_delete.len();
        if count == 0 {
            return Ok(0);
        }
        for entry in &to_delete {
            vector_kv::remove_queue_entry(&self.db, &entry.namespace, &entry.doc_id_bytes).await?;
        }
        Ok(count)
    }

    /// Validate that `namespace` exists and has semantic search enabled.
    ///
    /// Returns `Ok(())` when the namespace is ready for `index_all`.
    /// Returns `Err(NotFound)` or `Err(SemanticSearchNotEnabled)` otherwise.
    /// This is a synchronous check (no I/O) suitable for upfront validation
    /// before spawning a background task.
    pub fn check_index_all_preconditions(&self, namespace: &str) -> Result<(), DocStoreError> {
        let schema = self.load_schema(namespace)?;
        if !schema.is_semantic_search_enabled() {
            return Err(DocStoreError::SemanticSearchNotEnabled {
                namespace: namespace.to_owned(),
            });
        }
        // Also reject if a reindex is currently running.
        if let Some(ns_id) = schema.ns_id {
            let path = vec_reindex_path(&self.db_path, ns_id);
            if let Some(c) = read_vec_reindex(&path)
                && c.status == "running"
            {
                return Err(DocStoreError::VecReindexInProgress {
                    namespace: namespace.to_owned(),
                });
            }
        }
        Ok(())
    }

    /// Return the persisted vector-index reindex record for `namespace`, or
    /// `None` when no reindex has been started for that namespace.
    pub fn vec_reindex_progress(&self, namespace: &str) -> Option<VecReindexProgress> {
        let schema = self.load_schema(namespace).ok()?;
        let ns_id = schema.ns_id?;
        read_vec_reindex(&vec_reindex_path(&self.db_path, ns_id))
    }

    /// Re-enqueue every document in `namespace` for vector indexing.
    ///
    /// The operation is a point-in-time snapshot of the doc store:
    ///
    /// 1. Rejects with [`DocStoreError::VecReindexInProgress`] when a
    ///    previous `index_all` reindex is still running.
    /// 2. All **exhausted** queue entries for the namespace (those whose
    ///    `retry_count` has reached `max_retries`) are removed first so they
    ///    don't block processing.
    /// 3. Every document whose embedding fields produce non-empty text is
    ///    enqueued with `retry_count = 0`.  Documents already in the queue
    ///    with a lower retry count are overwritten (natural deduplication
    ///    — only the latest text is kept).
    /// 4. A reindex record is written to
    ///    `{db_path}/index/{ns_id}/vector_reindex.json` so the progress API
    ///    can track this reindex across server restarts.
    /// 5. The vector-index worker is notified to start processing immediately.
    ///
    /// Returns [`DocStoreError::SemanticSearchNotEnabled`] when the namespace
    /// does not have semantic search configured.
    pub async fn index_all(&self, namespace: &str) -> Result<ReindexStats, DocStoreError> {
        let schema = self.load_schema(namespace)?;
        if !schema.is_semantic_search_enabled() {
            return Err(DocStoreError::SemanticSearchNotEnabled {
                namespace: namespace.to_owned(),
            });
        }

        let ns_id = schema.ns_id.ok_or_else(|| DocStoreError::MissingNsId {
            namespace: namespace.to_owned(),
        })?;

        // 409 guard: reject concurrent reindexs.
        let reindex_path = vec_reindex_path(&self.db_path, ns_id);
        if let Some(c) = read_vec_reindex(&reindex_path)
            && c.status == "running"
        {
            return Err(DocStoreError::VecReindexInProgress {
                namespace: namespace.to_owned(),
            });
        }

        // Write "running" reindex record immediately so the guard works even
        // if the process crashes before we finish enqueueing.
        let started_at_ms = now_ms();
        std::fs::create_dir_all(reindex_path.parent().unwrap())?;
        write_vec_reindex(
            &reindex_path,
            &VecReindexProgress {
                status: "running".to_owned(),
                started_at_ms,
                completed_at_ms: None,
                total_enqueued: 0,
                exhausted_cleared: 0,
                error: None,
            },
        );

        let max_retries = self.vector_index_config.max_retries;

        info!(
            "index_all: namespace='{}' scanning pending queue for exhausted entries (max_retries={})",
            namespace, max_retries
        );

        // Collect exhausted entries to clear.
        let all_queue = vector_kv::list_queue_entries(&self.db).await?;
        let total_queue = all_queue.len();
        let exhausted: Vec<_> = all_queue
            .into_iter()
            .filter(|e| e.namespace == namespace && e.retry_count >= max_retries)
            .collect();
        let exhausted_cleared = exhausted.len();
        info!(
            "index_all: namespace='{}' queue scan done — total_queue={} exhausted_to_clear={}",
            namespace, total_queue, exhausted_cleared
        );

        // Scan all documents in the namespace.
        info!("index_all: namespace='{}' scanning all documents", namespace);
        let ns = self.db.namespace(namespace.to_owned()).await?;
        let all_docs = ns.iter().await?;
        let total_docs = all_docs.len();
        info!("index_all: namespace='{}' document scan done — total_docs={}", namespace, total_docs);

        // Clear exhausted entries (durable single-op deletes), then enqueue every
        // document whose embedding fields produce non-empty text. Enqueues are
        // WAL-backed: with the vector index itself written no-WAL, the queue is the
        // durable source of truth for what still needs (re-)indexing, so every
        // enqueue must survive a crash.
        let mut enqueued = 0usize;
        let mut skipped_empty_text = 0usize;
        let commit_result: Result<(), DocStoreError> = async {
            for entry in &exhausted {
                vector_kv::remove_queue_entry(&self.db, &entry.namespace, &entry.doc_id_bytes).await?;
            }
            for (key, value) in &all_docs {
                let doc = match serde_json::from_slice::<serde_json::Value>(value) {
                    Ok(d) => d,
                    Err(_) => continue,
                };
                let text = build_embedding_text(&doc, &schema.embedding_fields);
                if !text.is_empty() {
                    vector_kv::enqueue_embed(&self.db, namespace, key, &text).await?;
                    enqueued += 1;
                } else {
                    skipped_empty_text += 1;
                }
            }
            Ok(())
        }
        .await;

        info!(
            "index_all: namespace='{}' enqueue done — enqueued={} skipped_empty_text={}",
            namespace, enqueued, skipped_empty_text
        );

        if let Err(e) = commit_result {
            write_vec_reindex(
                &reindex_path,
                &VecReindexProgress {
                    status: "failed".to_owned(),
                    started_at_ms,
                    completed_at_ms: Some(now_ms()),
                    total_enqueued: 0,
                    exhausted_cleared,
                    error: Some(e.to_string()),
                },
            );
            return Err(e);
        }

        // Update reindex record with the final enqueued count; status stays
        // "running" until the worker finishes (tracked via queue depth).
        write_vec_reindex(
            &reindex_path,
            &VecReindexProgress {
                status: "running".to_owned(),
                started_at_ms,
                completed_at_ms: None,
                total_enqueued: enqueued,
                exhausted_cleared,
                error: None,
            },
        );

        info!("index_all: namespace='{}' batch committed", namespace);

        if let Some(notify) = &self.notify {
            notify.notify_one();
            info!("index_all: namespace='{}' worker notified", namespace);
        }

        info!(
            "index_all: namespace='{}' enqueued={} exhausted_cleared={}",
            namespace, enqueued, exhausted_cleared,
        );

        Ok(ReindexStats { exhausted_cleared, enqueued })
    }

    /// Like [`check_index_all_preconditions`] but for KV store namespaces.
    ///
    /// [`check_index_all_preconditions`]: DocStore::check_index_all_preconditions
    pub fn check_kv_index_all_preconditions(&self, namespace: &str) -> Result<(), DocStoreError> {
        let schema = self.load_kv_schema(namespace)?;
        if !schema.is_semantic_search_enabled() {
            return Err(DocStoreError::SemanticSearchNotEnabled {
                namespace: namespace.to_owned(),
            });
        }
        if let Some(ns_id) = schema.ns_id {
            let path = vec_reindex_path(&self.db_path, ns_id);
            if let Some(c) = read_vec_reindex(&path)
                && c.status == "running"
            {
                return Err(DocStoreError::VecReindexInProgress {
                    namespace: namespace.to_owned(),
                });
            }
        }
        Ok(())
    }

    /// Like [`index_all`] but for KV store namespaces.
    ///
    /// Re-enqueues every KV entry whose value is a non-empty UTF-8 string.
    /// Only valid when `semantic_search_enabled = true` and `value_type = str`.
    ///
    /// [`index_all`]: DocStore::index_all
    pub async fn kv_index_all(&self, namespace: &str) -> Result<ReindexStats, DocStoreError> {
        let schema = self.load_kv_schema(namespace)?;
        if !schema.is_semantic_search_enabled() {
            return Err(DocStoreError::SemanticSearchNotEnabled {
                namespace: namespace.to_owned(),
            });
        }

        let ns_id = schema.ns_id.ok_or_else(|| DocStoreError::MissingNsId {
            namespace: namespace.to_owned(),
        })?;

        let reindex_path = vec_reindex_path(&self.db_path, ns_id);
        if let Some(c) = read_vec_reindex(&reindex_path)
            && c.status == "running"
        {
            return Err(DocStoreError::VecReindexInProgress {
                namespace: namespace.to_owned(),
            });
        }

        let started_at_ms = now_ms();
        std::fs::create_dir_all(reindex_path.parent().unwrap())?;
        write_vec_reindex(
            &reindex_path,
            &VecReindexProgress {
                status: "running".to_owned(),
                started_at_ms,
                completed_at_ms: None,
                total_enqueued: 0,
                exhausted_cleared: 0,
                error: None,
            },
        );

        let max_retries = self.vector_index_config.max_retries;

        info!(
            "kv_index_all: namespace='{}' scanning pending queue for exhausted entries (max_retries={})",
            namespace, max_retries
        );
        let all_queue = vector_kv::list_queue_entries(&self.db).await?;
        let exhausted: Vec<_> = all_queue
            .into_iter()
            .filter(|e| e.namespace == namespace && e.retry_count >= max_retries)
            .collect();
        let exhausted_cleared = exhausted.len();
        info!(
            "kv_index_all: namespace='{}' queue scan done — exhausted_to_clear={}",
            namespace, exhausted_cleared
        );

        info!("kv_index_all: namespace='{}' scanning all KV entries", namespace);
        let ns = self.db.namespace(namespace.to_owned()).await?;
        let all_entries = ns.iter().await?;
        info!("kv_index_all: namespace='{}' scan done — total_entries={}", namespace, all_entries.len());

        // Clear exhausted entries (durable single-op deletes), then enqueue every
        // entry with non-empty text. Enqueues are WAL-backed: the queue is the
        // durable source of truth for the no-WAL vector index — see index_all for
        // the rationale.
        let mut enqueued = 0usize;
        let mut skipped_empty_text = 0usize;
        let commit_result: Result<(), DocStoreError> = async {
            for entry in &exhausted {
                vector_kv::remove_queue_entry(&self.db, &entry.namespace, &entry.doc_id_bytes).await?;
            }
            for (key, value_bytes) in &all_entries {
                match std::str::from_utf8(value_bytes) {
                    Ok(text) if !text.is_empty() => {
                        vector_kv::enqueue_embed(&self.db, namespace, key, text).await?;
                        enqueued += 1;
                    }
                    _ => {
                        skipped_empty_text += 1;
                    }
                }
            }
            Ok(())
        }
        .await;

        info!(
            "kv_index_all: namespace='{}' enqueue done — enqueued={} skipped_empty_text={}",
            namespace, enqueued, skipped_empty_text
        );

        if let Err(e) = commit_result {
            write_vec_reindex(
                &reindex_path,
                &VecReindexProgress {
                    status: "failed".to_owned(),
                    started_at_ms,
                    completed_at_ms: Some(now_ms()),
                    total_enqueued: 0,
                    exhausted_cleared,
                    error: Some(e.to_string()),
                },
            );
            return Err(e);
        }

        write_vec_reindex(
            &reindex_path,
            &VecReindexProgress {
                status: "running".to_owned(),
                started_at_ms,
                completed_at_ms: None,
                total_enqueued: enqueued,
                exhausted_cleared,
                error: None,
            },
        );

        if let Some(notify) = &self.notify {
            notify.notify_one();
            info!("kv_index_all: namespace='{}' worker notified", namespace);
        }

        info!(
            "kv_index_all: namespace='{}' enqueued={} exhausted_cleared={}",
            namespace, enqueued, exhausted_cleared
        );

        Ok(ReindexStats { exhausted_cleared, enqueued })
    }

    // ── Admin: create / list / drop ───────────────────────────────────────

    /// Create a new document store from the given schema.
    ///
    /// - Validates the schema.
    /// - Creates the namespace in minnal_db and records the `ns_id`.
    /// - Registers and activates all declared field indices.
    /// - Persists the schema (with `ns_id`) to `schema_dir`.
    ///
    /// Returns [`DocStoreError::AlreadyExists`] if a store with that namespace
    /// is already registered.
    pub async fn create(&self, mut schema: DocStoreSchema) -> Result<(), DocStoreError> {
        schema.validate()?;

        let ns_name = schema.namespace.clone();
        if self.schema_path(&ns_name).exists() {
            return Err(DocStoreError::AlreadyExists { namespace: ns_name });
        }

        info!(
            "creating namespace '{}' (key_type={:?}, indices={}, semantic_search={})",
            ns_name,
            schema.key_type,
            schema.indices.len(),
            schema.semantic_search_enabled
        );

        // Create namespace — `namespace()` creates it if absent
        let ns_handle = self.db.namespace(ns_name.clone()).await?;
        let ns_id = ns_handle.id();
        schema.ns_id = Some(ns_id);

        // Register + activate field indices
        activate_indices(&self.db, ns_id, &schema).await?;

        // Persist schema
        schema.save(&self.schema_dir)?;
        info!("namespace '{}' created (ns_id={})", ns_name, ns_id);
        Ok(())
    }

    /// List all known document stores.
    ///
    /// Returns each store's full schema as a [`serde_json::Value`] so callers
    /// can forward it to an HTTP response or log it without an intermediate
    /// struct conversion.
    pub fn list(&self) -> Result<Vec<serde_json::Value>, DocStoreError> {
        self.load_all_schemas()?
            .into_iter()
            .map(|s| serde_json::to_value(&s).map_err(DocStoreError::from))
            .collect()
    }

    /// Destroy a document store completely.
    ///
    /// Removes the minnal_db namespace (in-memory), deletes all on-disk
    /// storage (`ns_{name}/`, `index/{ns_id}/`), and removes the schema file.
    ///
    /// This is irreversible.
    pub async fn remove(&self, namespace: &str) -> Result<(), DocStoreError> {
        info!("dropping namespace '{}'", namespace);
        let schema = self.load_schema(namespace)?;
        let ns_id = schema.ns_id.ok_or_else(|| DocStoreError::MissingNsId {
            namespace: namespace.to_owned(),
        })?;
        let schema_path = self.schema_path(namespace);
        cleanup_store_namespaces(&self.db, &self.db_path, namespace, ns_id, &schema_path).await?;
        info!("namespace '{}' dropped", namespace);
        Ok(())
    }

    // ── KV store lifecycle ─────────────────────────────────────────────────

    /// Create a new KV store namespace from the given schema.
    ///
    /// Unlike [`create`], no field indices are registered.  The namespace
    /// stores raw bytes keyed by the declared `key_type`.
    ///
    /// [`create`]: DocStore::create
    pub async fn create_kv(&self, mut schema: KvStoreSchema) -> Result<(), DocStoreError> {
        schema.validate()?;

        let ns_name = schema.namespace.clone();
        if self.schema_path(&ns_name).exists() {
            return Err(DocStoreError::AlreadyExists { namespace: ns_name });
        }

        info!(
            "creating KV namespace '{}' (key_type={:?}, value_type={:?}, semantic_search={})",
            ns_name, schema.key_type, schema.value_type, schema.semantic_search_enabled
        );

        let ns_handle = self.db.namespace(ns_name.clone()).await?;
        schema.ns_id = Some(ns_handle.id());
        schema.save(&self.schema_dir)?;
        info!("KV namespace '{}' created (ns_id={})", ns_name, schema.ns_id.unwrap());
        Ok(())
    }

    /// List all known KV stores.
    pub fn list_kv(&self) -> Result<Vec<serde_json::Value>, DocStoreError> {
        self.load_all_kv_schemas()?
            .into_iter()
            .map(|s| serde_json::to_value(&s).map_err(DocStoreError::from))
            .collect()
    }

    /// Destroy a KV store completely (irreversible).
    pub async fn remove_kv(&self, namespace: &str) -> Result<(), DocStoreError> {
        info!("dropping KV namespace '{}'", namespace);
        let schema = self.load_kv_schema(namespace)?;
        let ns_id = schema.ns_id.ok_or_else(|| DocStoreError::MissingNsId {
            namespace: namespace.to_owned(),
        })?;
        let schema_path = self.schema_path(namespace);
        cleanup_store_namespaces(&self.db, &self.db_path, namespace, ns_id, &schema_path).await?;
        info!("KV namespace '{}' dropped", namespace);
        Ok(())
    }

    // ── KV CRUD ────────────────────────────────────────────────────────────

    /// Insert or replace a value in a KV namespace.
    ///
    /// `key` and `value` are JSON values typed according to the namespace schema.
    /// When semantic search is configured and the namespace has
    /// `semantic_search_enabled = true`, the KV value is written first, then a
    /// pending embedding queue entry is enqueued.  The background
    /// `VecIndexWorker` processes the queue asynchronously — the vector index
    /// is eventually consistent with the KV store.  A crash between the two
    /// writes leaves the value un-indexed until reconciliation re-enqueues it.
    pub async fn kv_put(&self, namespace: &str, key: &serde_json::Value, value: &serde_json::Value) -> Result<(), DocStoreError> {
        self.kv_put_inner(namespace, key, value, false).await
    }

    /// Insert or replace a value in a KV namespace, **bypassing the WAL**.
    ///
    /// Identical to [`kv_put`](Self::kv_put) except the value write skips the WAL
    /// for maximum throughput.  Data written this way is unrecoverable on a crash
    /// — only use during bulk loading where re-running the load is acceptable.
    /// The embed marker (when semantic search is enabled) is still enqueued
    /// through the WAL, matching the document-store `put_no_wal` behaviour.
    pub async fn kv_put_no_wal(&self, namespace: &str, key: &serde_json::Value, value: &serde_json::Value) -> Result<(), DocStoreError> {
        self.kv_put_inner(namespace, key, value, true).await
    }

    /// Shared body for [`kv_put`](Self::kv_put) and
    /// [`kv_put_no_wal`](Self::kv_put_no_wal); `skip_wal` selects the value-write
    /// durability path.
    async fn kv_put_inner(&self, namespace: &str, key: &serde_json::Value, value: &serde_json::Value, skip_wal: bool) -> Result<(), DocStoreError> {
        let schema = self.load_kv_schema(namespace)?;
        let key_bytes = schema.key_type.serialize_key(key)?;
        let value_bytes = schema.value_type.serialize_value(value)?;
        let ns = self.db.namespace(namespace.to_owned()).await?;

        if skip_wal {
            ns.put_no_wal(key_bytes.clone(), value_bytes).await?;
        } else {
            ns.put(key_bytes.clone(), value_bytes).await?;
        }

        if let Some(notify) = &self.notify
            && schema.is_semantic_search_enabled()
            && let Some(text) = value.as_str()
            && !text.is_empty()
        {
            // Enqueue the embed marker as a separate single op (no cross-namespace
            // atomicity needed — see kv_put docs).
            vector_kv::enqueue_embed(&self.db, namespace, &key_bytes, text).await?;
            notify.notify_one();
        }

        Ok(())
    }

    /// Retrieve a value by key from a KV namespace.
    ///
    /// Returns `None` when the key does not exist.
    pub async fn kv_get(&self, namespace: &str, key: &serde_json::Value) -> Result<Option<serde_json::Value>, DocStoreError> {
        let schema = self.load_kv_schema(namespace)?;
        let key_bytes = schema.key_type.serialize_key(key)?;
        let ns = self.db.namespace(namespace.to_owned()).await?;
        match ns.get(key_bytes).await? {
            Some(bytes) => Ok(Some(schema.value_type.deserialize_value(&bytes)?)),
            None => Ok(None),
        }
    }

    /// Retrieve a value by its raw URL-path key string from a KV namespace.
    ///
    /// Convenience wrapper around [`kv_get`] for HTTP handlers that receive
    /// the key as a URL path segment.
    ///
    /// [`kv_get`]: DocStore::kv_get
    pub async fn kv_get_by_str(&self, namespace: &str, raw_key: &str) -> Result<Option<serde_json::Value>, DocStoreError> {
        let schema = self.load_kv_schema(namespace)?;
        let key_bytes = schema.key_type.serialize_key_from_str(raw_key)?;
        let ns = self.db.namespace(namespace.to_owned()).await?;
        match ns.get(key_bytes).await? {
            Some(bytes) => Ok(Some(schema.value_type.deserialize_value(&bytes)?)),
            None => Ok(None),
        }
    }

    /// Delete a key from a KV namespace.  No-op when the key does not exist.
    ///
    /// When semantic search is configured and the namespace has
    /// `semantic_search_enabled = true`, the pending queue entry and the vector
    /// index are removed first, then the KV value is deleted.  Each is a separate
    /// single-op write; ordering derived data before the value means a crash
    /// between them leaves an un-indexed value (reconciliation cleans it up),
    /// never an orphaned vector.
    pub async fn kv_delete(&self, namespace: &str, raw_key: &str) -> Result<(), DocStoreError> {
        let schema = self.load_kv_schema(namespace)?;
        let key_bytes = schema.key_type.serialize_key_from_str(raw_key)?;

        if schema.is_semantic_search_enabled() {
            vector_kv::remove_queue_entry(&self.db, namespace, &key_bytes).await?;
            vector_kv::delete_vector(&self.db, namespace, &key_bytes).await?;
            let ns = self.db.namespace(namespace.to_owned()).await?;
            ns.delete(key_bytes).await?;
            return Ok(());
        }

        let ns = self.db.namespace(namespace.to_owned()).await?;
        ns.delete(key_bytes).await?;
        Ok(())
    }

    /// Return entries in `[start, end)` from a KV namespace, one cursor page at a time.
    ///
    /// Pass `end = None` for an open-ended scan to the last key, and `cursor = None`
    /// for the first page; thereafter pass back the previous page's `next_cursor`.
    /// Only the page's values are resolved from the value log — memory stays O(limit),
    /// not O(total matches). Results are ordered by key ascending.
    pub async fn kv_scan_range(
        &self,
        namespace: &str,
        start: &str,
        end: Option<&str>,
        cursor: Option<Vec<u8>>,
        limit: usize,
    ) -> Result<CursorPage<(serde_json::Value, serde_json::Value)>, DocStoreError> {
        let schema = self.load_kv_schema(namespace)?;
        let start_bytes = schema.key_type.serialize_key_from_str(start)?;
        let end_bytes = end.map(|e| schema.key_type.serialize_key_from_str(e)).transpose()?;
        let scan_start = cursor.unwrap_or(start_bytes);
        let ns = self.db.namespace(namespace.to_owned()).await?;
        let (pairs, next_cursor) = ns.scan(Some(scan_start), end_bytes, limit).await?;

        let results = pairs
            .into_iter()
            .map(|(k, v)| -> Result<_, DocStoreError> {
                let key = schema.key_type.deserialize_key(&k)?;
                let value = schema.value_type.deserialize_value(&v)?;
                Ok((key, value))
            })
            .collect::<Result<_, _>>()?;

        Ok(CursorPage::new(results, next_cursor))
    }

    /// Return entries whose key starts with `prefix` from a KV namespace, one cursor
    /// page at a time.
    ///
    /// For `key_type = str` the prefix is a plain string matched against the
    /// UTF-8 key bytes.  For `key_type = int` the prefix is a decimal integer
    /// serialised as big-endian bytes (i.e. an exact-key prefix scan). The prefix
    /// is scanned as the range `[prefix, prefix⁺)`, so only the page's keys (not the
    /// whole keyspace tail) are resolved. Pass back `next_cursor` for the next page.
    pub async fn kv_scan_prefix(
        &self,
        namespace: &str,
        prefix: &str,
        cursor: Option<Vec<u8>>,
        limit: usize,
    ) -> Result<CursorPage<(serde_json::Value, serde_json::Value)>, DocStoreError> {
        let schema = self.load_kv_schema(namespace)?;
        let prefix_bytes = schema.key_type.serialize_key_from_str(prefix)?;
        let end_bytes = prefix_upper_bound(&prefix_bytes);
        let scan_start = cursor.unwrap_or_else(|| prefix_bytes.clone());
        let ns = self.db.namespace(namespace.to_owned()).await?;
        let (pairs, next_cursor) = ns.scan(Some(scan_start), end_bytes, limit).await?;

        let results = pairs
            .into_iter()
            .map(|(k, v)| -> Result<_, DocStoreError> {
                let key = schema.key_type.deserialize_key(&k)?;
                let value = schema.value_type.deserialize_value(&v)?;
                Ok((key, value))
            })
            .collect::<Result<_, _>>()?;

        Ok(CursorPage::new(results, next_cursor))
    }

    /// Run an ANN semantic search against a KV namespace with `value_type = str`.
    ///
    /// Returns [`DocStoreError::EmbeddingFailed`] when no [`SemanticSearchContext`]
    /// is configured, when the namespace does not have `semantic_search_enabled`,
    /// or when the embedding service call fails.
    pub async fn kv_search_semantic(
        &self,
        namespace: &str,
        query_text: &str,
        top_k: Option<usize>,
        pagination: crate::pagination::Pagination,
    ) -> Result<crate::pagination::Page<semantic_search::index::vector_index::QueryResult>, DocStoreError> {
        let ctx = self
            .semantic_ctx
            .as_ref()
            .ok_or_else(|| DocStoreError::EmbeddingFailed("semantic search not configured on this store".into()))?;

        let schema = self.load_kv_schema(namespace)?;
        if !schema.is_semantic_search_enabled() {
            return Err(DocStoreError::EmbeddingFailed(format!(
                "KV namespace '{namespace}' does not have semantic_search_enabled"
            )));
        }

        let (query_dense, query_sparse) = self.cached_query_embeddings(ctx, query_text).await?;

        let db_store = vector_kv::DbVectorStore::new(&self.db, namespace)
            .await
            .map_err(|e| DocStoreError::EmbeddingFailed(e.to_string()))?;
        let all = semantic_search::service::search(
            &ctx.config,
            namespace,
            &ctx.cluster_index,
            &query_sparse,
            &query_dense,
            &db_store,
            None::<fn(&[u8]) -> bool>,
            top_k,
        )
        .await;

        Ok(crate::pagination::Page::from_vec(all, pagination))
    }

    // ── Admin: schema amendment ────────────────────────────────────────────

    /// Amend the non-indexed attribute declarations of an existing store.
    ///
    /// Only [`SchemaAmendment::AddAttribute`], [`SchemaAmendment::RemoveAttribute`],
    /// and [`SchemaAmendment::UpdateAttribute`] are supported.  Attempting to
    /// remove or update an attribute that is used by an active index returns
    /// [`DocStoreError::AttributeIsIndexed`] — drop the index first.
    pub fn amend(&self, namespace: &str, amendment: SchemaAmendment) -> Result<(), DocStoreError> {
        let mut schema = self.load_schema(namespace)?;

        // Re-map SchemaError::AttributeIsIndexed to DocStoreError with namespace context
        schema.apply_amendment(amendment).map_err(|e| match e {
            SchemaError::AttributeIsIndexed { name } => DocStoreError::AttributeIsIndexed {
                namespace: namespace.to_owned(),
                field: name,
            },
            other => DocStoreError::Schema(other),
        })?;

        schema.save(&self.schema_dir)?;
        Ok(())
    }

    /// Return the schema for a namespace without the JSON round-trip overhead of [`list`].
    ///
    /// [`list`]: DocStore::list
    pub fn get_schema(&self, namespace: &str) -> Result<DocStoreSchema, DocStoreError> {
        self.load_schema(namespace)
    }

    /// Return the KV schema for a namespace without the JSON round-trip overhead of [`list_kv`].
    ///
    /// [`list_kv`]: DocStore::list_kv
    pub fn get_kv_schema(&self, namespace: &str) -> Result<KvStoreSchema, DocStoreError> {
        self.load_kv_schema(namespace)
    }

    /// Remove an attribute from the schema, and, if it is an embedding field,
    /// also remove it from `embedding_fields`.  When the removal empties
    /// `embedding_fields`, `semantic_search_enabled` is set to `false` and the
    /// updated schema is persisted in one atomic write.
    ///
    /// Returns `true` when the operation disabled semantic search (all embedding
    /// fields are now gone), signalling that the caller should trigger a
    /// background vector-index cleanup.
    ///
    /// Returns `Err(AttributeIsIndexed)` if the attribute is used by a field
    /// index — drop the index first.
    pub fn remove_attribute(&self, namespace: &str, field_name: &str) -> Result<bool, DocStoreError> {
        let mut schema = self.load_schema(namespace)?;

        let was_embedding_field = schema.embedding_fields.contains(&field_name.to_owned());
        let will_disable_ss = was_embedding_field && schema.embedding_fields.len() == 1;

        schema
            .apply_amendment(SchemaAmendment::RemoveAttribute { name: field_name.to_owned() })
            .map_err(|e| match e {
                SchemaError::AttributeIsIndexed { name } => DocStoreError::AttributeIsIndexed {
                    namespace: namespace.to_owned(),
                    field: name,
                },
                other => DocStoreError::Schema(other),
            })?;

        if was_embedding_field {
            schema.embedding_fields.retain(|f| f != field_name);
            if schema.embedding_fields.is_empty() {
                schema.semantic_search_enabled = false;
            }
        }

        schema.save(&self.schema_dir)?;
        Ok(will_disable_ss)
    }

    /// Disable semantic search for a namespace by clearing `embedding_fields`
    /// and setting `semantic_search_enabled = false`.
    ///
    /// Returns `Err(SemanticSearchNotEnabled)` when the namespace does not have
    /// semantic search configured, so callers can surface a proper 422.
    pub fn disable_semantic_search(&self, namespace: &str) -> Result<(), DocStoreError> {
        let mut schema = self.load_schema(namespace)?;
        if !schema.semantic_search_enabled && schema.embedding_fields.is_empty() {
            return Err(DocStoreError::SemanticSearchNotEnabled {
                namespace: namespace.to_owned(),
            });
        }
        schema.semantic_search_enabled = false;
        schema.embedding_fields.clear();
        schema.save(&self.schema_dir)?;
        Ok(())
    }

    /// Drop every field index for a namespace and return their specs.
    ///
    /// Each field is demoted to a plain attribute (its data stays in stored
    /// documents) and its on-disk index files are deleted.  The caller can pass
    /// the returned specs back to [`add_index`] to rebuild them.
    ///
    /// [`add_index`]: DocStore::add_index
    pub fn drop_all_attribute_indices(&self, namespace: &str) -> Result<Vec<IndexSpec>, DocStoreError> {
        let schema = self.load_schema(namespace)?;
        let specs = schema.indices.clone();
        for spec in &specs {
            self.drop_index(namespace, &spec.field)?;
        }
        Ok(specs)
    }

    /// Delete all vector-index backing data for a namespace.
    ///
    /// Clears:
    /// - Every entry in the global embedding queue that belongs to `namespace`.
    /// - All entries in the `{namespace}_sparse_vector_meta` companion namespace.
    /// - All entries in the `{namespace}_sparse_vector` companion namespace.
    /// - All entries in the `{namespace}_dense_vector` companion namespace.
    /// - The `vector_reindex.json` progress file (if present).
    ///
    /// This does **not** update the schema — the caller must call
    /// [`disable_semantic_search`] (or equivalent) before spawning this as a
    /// background task, so that new writes do not re-enqueue embeddings during
    /// cleanup.
    ///
    /// [`disable_semantic_search`]: DocStore::disable_semantic_search
    pub async fn drop_vector_index_data(&self, namespace: &str) -> Result<(), DocStoreError> {
        info!("drop_vector_index_data: clearing embedding queue for namespace='{namespace}'");
        self.delete_all_queue_entries(namespace).await?;

        // Remove the companion vector namespaces outright (reclaiming their
        // storage) rather than just emptying their entries — otherwise the empty
        // namespaces linger in /admin/storage/kv-namespaces as orphaned
        // "companion" stores after the index is dropped.
        for companion in [
            vector_kv::sparse_vectors_meta_ns(namespace),
            vector_kv::sparse_vectors_ns(namespace),
            vector_kv::dense_vectors_ns(namespace),
        ] {
            if let Err(e) = self.db.remove_namespace(companion.clone()).await {
                // Best-effort: a missing companion is fine (nothing was indexed).
                debug!("drop_vector_index_data: removing companion '{companion}' for '{namespace}': {e}");
            }
        }

        // Clear the in-memory corruption counters so a dropped index stops
        // showing up in /admin/indices/vector/corruption-metrics.
        semantic_search::metrics::reset(namespace);

        if let Ok(schema) = self.load_schema(namespace)
            && let Some(ns_id) = schema.ns_id
        {
            let _ = std::fs::remove_file(vec_reindex_path(&self.db_path, ns_id));
        }

        info!("drop_vector_index_data: namespace='{namespace}' cleanup complete");
        Ok(())
    }

    // ── Admin: index management ────────────────────────────────────────────

    /// Drop an index from an existing store.
    ///
    /// Removes the `IndexSpec` from `indices` and **demotes the field to a
    /// non-indexed `AttributeDef`** in `attributes`, preserving the type
    /// declaration.  The field's data remains in every stored document; only
    /// the live index files (`index/{ns_id}/{field_id}/`) are deleted.
    ///
    /// The field registration in `config.json` is left in place so that a
    /// later [`add_index`] call for the same field reuses the same `field_id`.
    ///
    /// [`add_index`]: DocStore::add_index
    pub fn drop_index(&self, namespace: &str, field: &str) -> Result<(), DocStoreError> {
        info!("dropping index '{}' from namespace '{}'", field, namespace);
        let mut schema = self.load_schema(namespace)?;
        let ns_id = schema.ns_id.ok_or_else(|| DocStoreError::MissingNsId {
            namespace: namespace.to_owned(),
        })?;

        // Confirm the index exists in the schema
        let idx_pos = schema
            .indices
            .iter()
            .position(|s| s.field == field)
            .ok_or_else(|| DocStoreError::IndexNotFound {
                namespace: namespace.to_owned(),
                field: field.to_owned(),
            })?;

        let spec = schema.indices.remove(idx_pos);

        // Demote to a non-indexed attribute so the field is still documented
        // in the schema. Skip if an AttributeDef with this name already exists
        // (shouldn't happen, but be safe).
        if !schema.attributes.iter().any(|a| a.name == spec.field) {
            schema.attributes.push(crate::schema::AttributeDef {
                name: spec.field.clone(),
                attr_type: index_type_to_attr_type(spec.index_type),
                description: Some("previously indexed; field data is still present in stored documents".to_owned()),
            });
        }

        // Deactivate the in-memory bitmap and delete on-disk checkpoint files.
        let fields = self.db.list_index_fields(ns_id);
        if let Some(meta) = fields.iter().find(|f| f.field_name == field) {
            self.db.deactivate_field_index(ns_id, meta.field_id)?;
            let index_dir = self.db_path.join("index").join(ns_id.to_string()).join(meta.field_id.to_string());
            if index_dir.exists() {
                std::fs::remove_dir_all(&index_dir)?;
            }
        }

        schema.save(&self.schema_dir)?;
        info!("index '{}' dropped from namespace '{}'", field, namespace);
        Ok(())
    }

    /// Add a new index to an existing store and build it in the background.
    ///
    /// Returns an [`IndexBuildHandle`] immediately.  The index is activated
    /// before the handle is returned, so new writes are indexed straight away.
    /// The background task then scans all existing documents and re-puts each
    /// one so the extractor is called for historical data too.
    ///
    /// Returns [`DocStoreError::IndexAlreadyExists`] if the field is already
    /// indexed.
    pub async fn add_index(&self, namespace: &str, spec: IndexSpec) -> Result<IndexBuildHandle, DocStoreError> {
        let mut schema = self.load_schema(namespace)?;
        let ns_id = schema.ns_id.ok_or_else(|| DocStoreError::MissingNsId {
            namespace: namespace.to_owned(),
        })?;

        if schema.indices.iter().any(|s| s.field == spec.field) {
            return Err(DocStoreError::IndexAlreadyExists {
                namespace: namespace.to_owned(),
                field: spec.field.clone(),
            });
        }

        // Enforce the per-namespace index cap here too: `schema.save()` below does
        // not validate (only `validate_and_save` does), so without this check
        // `add_index` could push the namespace past MAX_INDICES one field at a
        // time even though create/import reject it.
        if schema.indices.len() >= crate::schema::MAX_INDICES {
            return Err(DocStoreError::Schema(SchemaError::TooManyIndices {
                count: schema.indices.len() + 1,
                max: crate::schema::MAX_INDICES,
            }));
        }

        // Guard against a second caller racing in while a background build is
        // still running (e.g. after a drop_index + immediate re-add).
        // We detect this via the on-disk progress file rather than in-process
        // state so the check also works when called directly (not via the API).
        {
            let ivt = to_ivt(spec.index_type);
            if let Ok(fid) = self.db.register_index_field(ns_id, &spec.field, ivt) {
                let progress_path = build_progress_path(&self.db_path, ns_id, fid);
                if read_disk_progress(&progress_path).map(|p| p.status == "in_progress").unwrap_or(false) {
                    return Err(DocStoreError::IndexBuildInProgress {
                        namespace: namespace.to_owned(),
                        field: spec.field.clone(),
                    });
                }
            }
        }

        let ivt = to_ivt(spec.index_type);
        let field_id = self.db.register_index_field(ns_id, &spec.field, ivt)?;
        info!(
            "adding index '{}' (type={:?}, field_id={}) to namespace '{}' (ns_id={})",
            spec.field, spec.index_type, field_id, namespace, ns_id
        );
        let extractor = json_extractor(spec.field.clone(), spec.index_type);
        self.db.activate_field_index(ns_id, field_id, ivt, extractor).await?;

        // Persist updated schema — if the field was previously declared as a
        // non-indexed attribute, move it to indices (remove from attributes).
        schema.attributes.retain(|a| a.name != spec.field);
        schema.indices.push(spec.clone());
        schema.save(&self.schema_dir)?;

        // ── Background rebuild ────────────────────────────────────────────
        // Check for a previously interrupted build and resume from where it left off.
        let progress_path = build_progress_path(&self.db_path, ns_id, field_id);
        let resume_after: Option<Vec<u8>> = read_disk_progress(&progress_path)
            .filter(|p| p.status == "in_progress")
            .and_then(|p| p.last_key_hex)
            .and_then(|h| hex_to_bytes(&h));
        if resume_after.is_some() {
            info!("resuming interrupted index build for namespace '{}' field '{}'", namespace, spec.field);
        } else {
            info!("starting background index build for namespace '{}' field '{}'", namespace, spec.field);
        }

        let mem = Arc::new(InMemoryProgress::new());
        let disk = Arc::new(DiskProgress::new(&progress_path, 1_000));
        let observer: Arc<dyn IndexProgressObserver> = Arc::new(ChainedObserver(vec![
            Arc::clone(&mem) as Arc<dyn IndexProgressObserver>,
            Arc::clone(&disk) as Arc<dyn IndexProgressObserver>,
        ]));

        let observer_clone = Arc::clone(&observer);
        let fail_observer = Arc::clone(&observer);
        let db_clone = Arc::clone(&self.db);
        let ns_name = namespace.to_owned();
        let field_name = spec.field.clone();
        let key_type = schema.key_type;

        let task = tokio::spawn(async move {
            let result = rebuild_index_for_namespace(db_clone, ns_name, key_type, field_id, resume_after, observer_clone).await;
            if let Err(ref e) = result {
                // Notify the whole chain (in-memory *and* disk), so the failure
                // is persisted, not just visible to live pollers.
                fail_observer.on_status(BuildStatus::Failed, Some(&e.to_string()));
            }
            result
        });

        Ok(IndexBuildHandle {
            namespace: namespace.to_owned(),
            field: field_name,
            mem,
            task,
        })
    }

    /// Resume any index builds that were interrupted by a previous shutdown.
    ///
    /// Scans all persisted schemas.  For each index whose
    /// `build_progress.json` has `status == "in_progress"`, a new background
    /// task is spawned that continues from the last checkpointed key rather
    /// than rescanning every document.
    ///
    /// Call this once at startup, right after [`open`] / [`open_with_config`],
    /// and store the returned handles the same way you would handles returned
    /// by [`add_index`].
    ///
    /// [`open`]: DocStore::open
    /// [`open_with_config`]: DocStore::open_with_config
    /// [`add_index`]: DocStore::add_index
    pub async fn resume_pending_builds(&self) -> Result<Vec<IndexBuildHandle>, DocStoreError> {
        let mut handles = Vec::new();

        for schema in self.load_all_schemas()? {
            let ns_id = match schema.ns_id {
                Some(id) => id,
                None => continue,
            };

            for spec in &schema.indices {
                let ivt = to_ivt(spec.index_type);
                let field_id = self.db.register_index_field(ns_id, &spec.field, ivt)?;
                let progress_path = build_progress_path(&self.db_path, ns_id, field_id);

                let Some(disk) = read_disk_progress(&progress_path) else { continue };
                if disk.status != "in_progress" {
                    continue;
                }

                info!(
                    "resuming interrupted index build: namespace='{}' field='{}' progress={}/{}",
                    schema.namespace, spec.field, disk.indexed, disk.total
                );
                let resume_after = disk.last_key_hex.as_deref().and_then(hex_to_bytes);

                let mem = Arc::new(InMemoryProgress::with_initial(disk.total, disk.indexed));
                let disk_obs = Arc::new(DiskProgress::new(&progress_path, 1_000));
                let observer: Arc<dyn IndexProgressObserver> = Arc::new(ChainedObserver(vec![
                    Arc::clone(&mem) as Arc<dyn IndexProgressObserver>,
                    Arc::clone(&disk_obs) as Arc<dyn IndexProgressObserver>,
                ]));

                let observer_clone = Arc::clone(&observer);
                let fail_observer = Arc::clone(&observer);
                let db_clone = Arc::clone(&self.db);
                let ns_name = schema.namespace.clone();
                let field_name = spec.field.clone();
                let key_type = schema.key_type;

                let task = tokio::spawn(async move {
                    let result = rebuild_index_for_namespace(db_clone, ns_name, key_type, field_id, resume_after, observer_clone).await;
                    if let Err(ref e) = result {
                        // Notify the whole chain (in-memory *and* disk), so the
                        // failure is persisted, not just visible to live pollers.
                        fail_observer.on_status(BuildStatus::Failed, Some(&e.to_string()));
                    }
                    result
                });

                handles.push(IndexBuildHandle {
                    namespace: schema.namespace.clone(),
                    field: field_name,
                    mem,
                    task,
                });
            }
        }

        Ok(handles)
    }

    /// Return the persisted build progress for a specific index, or `None` if
    /// no `build_progress.json` exists for that field.
    ///
    /// This is the fallback used by the progress API when no active in-memory
    /// handle exists (e.g. after the build completed and was drained on shutdown).
    pub fn index_build_disk_progress(&self, namespace: &str, field: &str) -> Option<DiskBuildProgress> {
        let schema = self.load_schema(namespace).ok()?;
        let ns_id = schema.ns_id?;
        let fields = self.db.list_index_fields(ns_id);
        let meta = fields.iter().find(|f| f.field_name == field)?;
        let path = build_progress_path(&self.db_path, ns_id, meta.field_id);
        read_disk_progress(&path)
    }

    // ── Admin / diagnostics ───────────────────────────────────────────────

    /// Return the number of documents stored in `namespace`.
    pub async fn count_docs(&self, namespace: &str) -> Result<usize, DocStoreError> {
        self.load_schema(namespace)?;
        let ns = self.db.namespace(namespace.to_owned()).await?;
        let keys = ns.keys().await?;
        Ok(keys.len())
    }

    /// Returns engine-wide value-log statistics.
    pub fn db_stats(&self) -> minnal_db::Stats {
        self.db.stats()
    }

    /// Returns a snapshot of engine-wide operational metrics (runtime counters).
    pub fn ops_metrics(&self) -> minnal_db::MetricsSnapshot {
        self.db.ops_metrics()
    }

    /// Operational metrics for a single namespace, by name.
    ///
    /// The only failure mode is an unknown namespace, so any underlying error is
    /// surfaced as [`DocStoreError::NotFound`] (→ 404).
    pub fn ops_metrics_for(&self, namespace: &str) -> Result<minnal_db::MetricsSnapshot, DocStoreError> {
        self.db.ops_metrics_for(namespace).map_err(|_| DocStoreError::NotFound {
            namespace: namespace.to_owned(),
        })
    }

    /// Per-namespace operational metrics for every live namespace, keyed by name.
    pub fn ops_metrics_by_namespace(&self) -> Vec<(String, minnal_db::MetricsSnapshot)> {
        self.db.ops_metrics_by_namespace()
    }

    /// Returns a snapshot of the shared WAL metadata.
    pub fn wal_metadata(&self) -> minnal_db::WalMetadata {
        self.db.wal_metadata()
    }

    /// Returns a live LSM manifest snapshot for every active namespace.
    pub fn lsm_manifests(&self) -> Vec<(String, minnal_db::LsmManifest)> {
        self.db.lsm_manifests()
    }

    /// Returns the in-memory (non-SSTable) LSM stats for every active namespace.
    pub fn lsm_runtime_stats(&self) -> Vec<(String, minnal_db::LSMStats)> {
        self.db.lsm_runtime_stats()
    }

    /// Returns per-bucket value-log metadata for every active namespace.
    pub fn value_log_shard_stats(&self) -> Vec<(String, Vec<(u32, minnal_db::ValueLogMetadata)>)> {
        self.db.value_log_shard_stats()
    }

    /// Physical (on-disk) vs logical value-log footprint per shard, per namespace.
    pub fn value_log_physical_stats(&self) -> Vec<(String, Vec<minnal_db::ShardPhysicalStats>)> {
        self.db.value_log_physical_stats()
    }

    /// Per-page value-log garbage breakdown for one namespace (by name).
    pub fn value_log_page_stats(&self, namespace: &str) -> Result<Vec<(u32, Vec<minnal_db::PageGarbageStats>)>, DocStoreError> {
        self.db.value_log_page_stats(namespace).map_err(DocStoreError::from)
    }

    /// Run value-log GC on every namespace and return per-namespace results.
    pub async fn garbage_collect_all(&self) -> Vec<(String, minnal_db::GCStats)> {
        self.db.garbage_collect_all().await
    }

    /// Run WAL garbage collection (reclaims fully-persisted WAL segments).
    pub async fn garbage_collect_wal(&self) -> Result<(u64, u64), minnal_db::KVError> {
        self.db.garbage_collect_wal().await
    }

    /// Trigger LSM compaction across all namespaces.
    pub async fn compact(&self) -> Result<(), minnal_db::KVError> {
        self.db.compact().await
    }

    /// Force an index checkpoint across all namespaces: flush each namespace's
    /// dense row map and all active field indexes to disk, compacting any
    /// field-index bitmap store over the configured waste threshold. Returns the
    /// number of active field indexes checkpointed.
    ///
    /// This is the same pass the periodic index-checkpoint worker runs and that
    /// shutdown runs once on close — exposed for on-demand flush + compaction.
    pub async fn checkpoint_index(&self) -> Result<usize, minnal_db::KVError> {
        self.db.checkpoint_index().await
    }

    /// Returns all indexed fields registered for a namespace.
    pub fn list_index_fields(&self, namespace_id: u32) -> Vec<minnal_db::FieldMeta> {
        self.db.list_index_fields(namespace_id)
    }

    /// Return the number of distinct indexed values for a field, or `None` if the
    /// field is not currently active (e.g. still building).
    pub fn field_index_distinct_count(&self, namespace: &str, field: &str) -> Option<usize> {
        let schema = self.load_schema(namespace).ok()?;
        let ns_id = schema.ns_id?;
        let fields = self.db.list_index_fields(ns_id);
        let fm = fields.iter().find(|f| f.field_name == field)?;
        self.db.field_index_distinct_count(ns_id, fm.field_id)
    }

    /// Reclaimable dead-space ratios `(bitmap_waste, keymap_waste)` for a field's
    /// append-only index stores, or `None` if the field is not currently active.
    /// Useful for monitoring how close a field is to triggering compaction.
    pub fn field_index_waste(&self, namespace: &str, field: &str) -> Option<(f64, f64)> {
        let schema = self.load_schema(namespace).ok()?;
        let ns_id = schema.ns_id?;
        let fields = self.db.list_index_fields(ns_id);
        let fm = fields.iter().find(|f| f.field_name == field)?;
        self.db.field_index_waste(ns_id, fm.field_id)
    }

    /// On-disk blob growth/waste metrics for a field's append-only index stores
    /// (bitmap + keymap logical vs. live bytes, waste ratios, distinct-value
    /// count), or `None` if the field is not currently active. Surfaces the
    /// absolute blob *growth* between compactions that the waste *ratio* alone
    /// hides — worst for low-cardinality, high-churn fields.
    pub fn field_index_blob_stats(&self, namespace: &str, field: &str) -> Option<minnal_db::IndexBlobStats> {
        let schema = self.load_schema(namespace).ok()?;
        let ns_id = schema.ns_id?;
        let fields = self.db.list_index_fields(ns_id);
        let fm = fields.iter().find(|f| f.field_name == field)?;
        self.db.field_index_blob_stats(ns_id, fm.field_id)
    }

    /// Reindex a single document's entry in one field index, re-deriving the
    /// field value from the document's current stored bytes using the same logic
    /// as the write path (clear the row's old buckets, re-extract, insert). Only
    /// the named field is touched — the document is not rewritten and no other
    /// field or vector index is affected.
    ///
    /// Returns the [`minnal_db::FieldReindexOutcome`]. Errors with
    /// [`DocStoreError::IndexNotFound`] when `field` is not an indexed field of
    /// the namespace.
    pub async fn reindex_doc_field(&self, namespace: &str, id: DocId, field: &str) -> Result<minnal_db::FieldReindexOutcome, DocStoreError> {
        let schema = self.load_schema(namespace)?;
        let ns_id = schema.ns_id.ok_or_else(|| DocStoreError::MissingNsId {
            namespace: namespace.to_owned(),
        })?;
        let fields = self.db.list_index_fields(ns_id);
        let fm = fields
            .iter()
            .find(|f| f.field_name == field)
            .ok_or_else(|| DocStoreError::IndexNotFound {
                namespace: namespace.to_owned(),
                field: field.to_owned(),
            })?;
        Ok(self.db.reindex_field(ns_id, fm.field_id, id.to_bytes()).await?)
    }

    /// Re-enqueue a single document for vector (re-)embedding — the same enqueue
    /// the write path and [`index_all`](DocStore::index_all) use, scoped to one
    /// document. The async worker picks it up on its next pass.
    ///
    /// Errors with [`DocStoreError::SemanticSearchNotEnabled`] when the namespace
    /// is not semantic-search-enabled.
    pub async fn reindex_doc_vector(&self, namespace: &str, id: DocId) -> Result<VectorReindexOutcome, DocStoreError> {
        let schema = self.load_schema(namespace)?;
        if !schema.is_semantic_search_enabled() {
            return Err(DocStoreError::SemanticSearchNotEnabled {
                namespace: namespace.to_owned(),
            });
        }
        let key = id.to_bytes();
        let ns = self.db.namespace(namespace.to_owned()).await?;
        let value = match ns.get(key.clone()).await? {
            Some(v) => v,
            None => return Ok(VectorReindexOutcome::NotFound),
        };
        let doc: serde_json::Value = serde_json::from_slice(&value).map_err(|e| DocStoreError::InvalidId(e.to_string()))?;
        let text = build_embedding_text(&doc, &schema.embedding_fields);
        if text.is_empty() {
            return Ok(VectorReindexOutcome::SkippedEmptyText);
        }
        vector_kv::enqueue_embed(&self.db, namespace, &key, &text).await?;
        if let Some(notify) = &self.notify {
            notify.notify_one();
        }
        Ok(VectorReindexOutcome::Enqueued)
    }

    /// Re-enqueue a single KV entry for vector (re-)embedding — the KV-store
    /// counterpart of [`reindex_doc_vector`](DocStore::reindex_doc_vector). The
    /// KV value (a string) is the embedding text.
    ///
    /// Errors with [`DocStoreError::SemanticSearchNotEnabled`] when the namespace
    /// is not semantic-search-enabled.
    pub async fn kv_reindex_doc_vector(&self, namespace: &str, raw_key: &str) -> Result<VectorReindexOutcome, DocStoreError> {
        let schema = self.load_kv_schema(namespace)?;
        if !schema.is_semantic_search_enabled() {
            return Err(DocStoreError::SemanticSearchNotEnabled {
                namespace: namespace.to_owned(),
            });
        }
        let key_bytes = schema.key_type.serialize_key_from_str(raw_key)?;
        let ns = self.db.namespace(namespace.to_owned()).await?;
        let value = match ns.get(key_bytes.clone()).await? {
            Some(v) => v,
            None => return Ok(VectorReindexOutcome::NotFound),
        };
        let text = match std::str::from_utf8(&value) {
            Ok(t) if !t.is_empty() => t,
            _ => return Ok(VectorReindexOutcome::SkippedEmptyText),
        };
        vector_kv::enqueue_embed(&self.db, namespace, &key_bytes, text).await?;
        if let Some(notify) = &self.notify {
            notify.notify_one();
        }
        Ok(VectorReindexOutcome::Enqueued)
    }

    /// The configured field-index compaction threshold as a fraction (`0.0..1.0`).
    pub fn index_blob_waste_threshold(&self) -> f64 {
        self.db.index_blob_waste_threshold()
    }

    /// Returns the name and numeric ID of every KV namespace currently open in
    /// the underlying database.
    pub fn list_kv_namespaces(&self) -> Vec<(String, u32)> {
        self.db.list_namespaces()
    }

    /// Return `(ttl_secs, max_deletes_per_run)` for a namespace, or `None` if
    /// no TTL is registered.
    pub fn ttl_config_for_ns(&self, ns_id: u32) -> Option<(u64, usize)> {
        self.db.ttl_config_for_ns(ns_id)
    }

    // ── CRUD ──────────────────────────────────────────────────────────────
    //
    // Vector-index writes are decoupled from the write path.  When semantic
    // search is configured (`self.notify` is Some) and the namespace schema
    // has `semantic_search_enabled = true`, a pending-embed queue entry is
    // written atomically with the document — the background VecIndexWorker
    // processes it asynchronously.

    /// Insert or replace a document.
    ///
    /// The `id` must match the store's [`KeyType`].  The `doc` is serialized
    /// to JSON bytes before storage.
    ///
    /// When semantic search is configured and the namespace has
    /// `semantic_search_enabled = true`, the document is written first, then a
    /// pending embedding queue entry is enqueued.  The background
    /// `VecIndexWorker` processes the queue entry asynchronously — the vector
    /// index is eventually consistent with the document store.  A crash between
    /// the two writes leaves the document un-indexed until reconciliation
    /// re-enqueues it.
    pub async fn put(&self, namespace: &str, id: DocId, doc: serde_json::Value) -> Result<(), DocStoreError> {
        debug!("put namespace='{}' id={:?}", namespace, id);
        let schema = self.load_schema(namespace)?;
        schema.validate_doc(&doc)?;

        let key = id.to_bytes();
        let value = serde_json::to_vec(&doc).map_err(|e| DocStoreError::InvalidId(e.to_string()))?;

        if let Some(notify) = &self.notify
            && schema.is_semantic_search_enabled()
        {
            let text = build_embedding_text(&doc, &schema.embedding_fields);
            if !text.is_empty() {
                // Write the document first, then enqueue the embed marker as a
                // separate single op (no cross-namespace atomicity needed).
                let ns = self.db.namespace(namespace.to_owned()).await?;
                ns.put(key.clone(), value).await?;
                vector_kv::enqueue_embed(&self.db, namespace, &key, &text).await?;
                notify.notify_one();
                return Ok(());
            }
        }

        let ns = self.db.namespace(namespace.to_owned()).await?;
        ns.put(key, value).await?;
        Ok(())
    }

    /// Store a document without writing to the WAL (bulk-load path).
    ///
    /// The document write skips the WAL for throughput.  When semantic search
    /// is configured, the pending embedding queue entry is **WAL-backed** even
    /// on this path, so the worker can recover pending jobs after a crash and
    /// index any documents that survived the no-WAL write.
    pub async fn put_no_wal(&self, namespace: &str, id: DocId, doc: serde_json::Value) -> Result<(), DocStoreError> {
        let schema = self.load_schema(namespace)?;
        schema.validate_doc(&doc)?;

        let key = id.to_bytes();
        let value = serde_json::to_vec(&doc).map_err(|e| DocStoreError::InvalidId(e.to_string()))?;

        let ns = self.db.namespace(namespace.to_owned()).await?;
        ns.put_no_wal(key.clone(), value).await?;

        if let Some(notify) = &self.notify
            && schema.is_semantic_search_enabled()
        {
            let text = build_embedding_text(&doc, &schema.embedding_fields);
            if !text.is_empty() {
                vector_kv::enqueue_embed(&self.db, namespace, &key, &text).await?;
                notify.notify_one();
            }
        }

        Ok(())
    }

    /// Delete a document by ID.  No-op if the document does not exist.
    ///
    /// When semantic search is configured and the namespace has
    /// `semantic_search_enabled = true`, the pending queue entry and the vector
    /// index are removed first, then the document is deleted.  Each is a separate
    /// single-op write; ordering derived data before the document means a crash
    /// between them leaves an un-indexed document (reconciliation cleans it up),
    /// never an orphaned vector.
    pub async fn delete(&self, namespace: &str, id: DocId) -> Result<(), DocStoreError> {
        debug!("delete namespace='{}' id={:?}", namespace, id);
        let key = id.to_bytes();

        let schema = self.load_schema(namespace)?;
        if schema.is_semantic_search_enabled() {
            vector_kv::remove_queue_entry(&self.db, namespace, &key).await?;
            vector_kv::delete_vector(&self.db, namespace, &key).await?;
            let ns = self.db.namespace(namespace.to_owned()).await?;
            ns.delete(key).await?;
            return Ok(());
        }

        let ns = self.db.namespace(namespace.to_owned()).await?;
        ns.delete(key).await?;
        Ok(())
    }

    // ── Query ─────────────────────────────────────────────────────────────

    /// Fetch the query embeddings needed for a two-pass semantic search.
    ///
    /// Returns `(dense, sparse)`: the single whole-query embedding used in Pass 2
    /// dense re-ranking, and the sliding-window chunk embeddings used in Pass 1.
    /// Uses the system-wide TTL cache when possible, falling back to the embedding
    /// service (then populating the cache) on a miss.
    async fn cached_query_embeddings(&self, ctx: &SemanticSearchContext, query_text: &str) -> Result<(Vec<f32>, Vec<Vec<f32>>), DocStoreError> {
        let ttl = ctx.config.query_embedding_cache_ttl;
        if let Some(cached) = vector_kv::get_cached_query_embedding(&self.db, query_text, ctx.config.embedding_dim, ttl).await {
            debug!("query embedding cache hit");
            return Ok(cached);
        }
        debug!("query embedding cache miss, calling embedding service");
        let q = semantic_search::service::embed_query(&ctx.config, query_text)
            .await
            .map_err(|e| DocStoreError::EmbeddingFailed(e.to_string()))?;
        vector_kv::put_cached_query_embedding(&self.db, query_text, &q.dense, &q.sparse, ttl).await;
        Ok((q.dense, q.sparse))
    }

    /// Clear the system-wide query-embedding cache, returning the number of
    /// entries removed.
    ///
    /// The cache is keyed only by query text, so it must be cleared after any
    /// change to the chunking parameters (`window_size` / `sliding_size`) —
    /// otherwise cached sparse vectors built under the old chunking keep being
    /// served (up to the configured TTL) and silently degrade recall against
    /// freshly-indexed documents. Exposed via the admin API for this purpose.
    pub async fn clear_query_embedding_cache(&self) -> Result<usize, DocStoreError> {
        let ttl = self
            .semantic_ctx
            .as_ref()
            .map(|c| c.config.query_embedding_cache_ttl)
            .unwrap_or(vector_kv::DEFAULT_QUERY_EMBEDDING_CACHE_TTL);
        vector_kv::clear_cached_query_embeddings(&self.db, ttl).await
    }

    /// Run an approximate nearest-neighbour semantic search against `namespace`.
    ///
    /// Embeds `query_text` using the configured embedding service, then scores
    /// every quantised vector in the namespace's companion KV store and returns
    /// the top results sorted by descending dot-product similarity.
    ///
    /// Returns [`DocStoreError::EmbeddingFailed`] if no [`SemanticSearchContext`]
    /// is attached, if the namespace does not have `semantic_search_enabled`, or
    /// if the embedding service call fails.
    pub async fn search_semantic(
        &self,
        namespace: &str,
        query_text: &str,
        top_k: Option<usize>,
        pagination: Pagination,
    ) -> Result<Page<semantic_search::index::vector_index::QueryResult>, DocStoreError> {
        let ctx = self
            .semantic_ctx
            .as_ref()
            .ok_or_else(|| DocStoreError::EmbeddingFailed("semantic search not configured on this store".into()))?;

        let schema = self.load_schema(namespace)?;
        if !schema.semantic_search_enabled {
            return Err(DocStoreError::EmbeddingFailed(format!(
                "namespace '{namespace}' does not have semantic_search_enabled"
            )));
        }

        debug!("semantic search namespace='{}' top_k={:?}", namespace, top_k);
        let (query_dense, query_sparse) = self.cached_query_embeddings(ctx, query_text).await?;

        let db_store = vector_kv::DbVectorStore::new(&self.db, namespace)
            .await
            .map_err(|e| DocStoreError::EmbeddingFailed(e.to_string()))?;

        let all = semantic_search::service::search(
            &ctx.config,
            namespace,
            &ctx.cluster_index,
            &query_sparse,
            &query_dense,
            &db_store,
            None::<fn(&[u8]) -> bool>,
            top_k,
        )
        .await;

        debug!("semantic search namespace='{}' returned {} results", namespace, all.len());
        Ok(Page::from_vec(all, pagination))
    }

    /// Run an approximate nearest-neighbour semantic search restricted to documents
    /// that also satisfy an index `predicate`.
    ///
    /// This is a two-phase operation:
    /// 1. Execute `predicate` against the index layer to obtain a set of matching
    ///    document IDs (same semantics as [`query`]).
    /// 2. Run ANN search, but skip any candidate whose raw ID bytes are not in
    ///    that set — so only documents that pass both the semantic ranking *and*
    ///    the predicate are returned.
    ///
    /// Returns [`DocStoreError::EmbeddingFailed`] under the same conditions as
    /// [`search_semantic`], and propagates any index query errors from `predicate`.
    ///
    /// [`query`]: DocStore::query
    /// [`search_semantic`]: DocStore::search_semantic
    pub async fn search_semantic_filtered(
        &self,
        namespace: &str,
        query_text: &str,
        predicate: &str,
        top_k: Option<usize>,
        pagination: Pagination,
    ) -> Result<Page<semantic_search::index::vector_index::QueryResult>, DocStoreError> {
        // Phase 1: collect ALL doc IDs that satisfy the predicate (no pagination
        // here — the full set is needed as an ANN filter before scoring).
        let all_keys = self.query_all_keys(namespace, predicate).await?;
        let allowed_ids: std::collections::HashSet<Vec<u8>> = all_keys.into_iter().collect();

        // Phase 2: ANN search with the filter closure.
        let ctx = self
            .semantic_ctx
            .as_ref()
            .ok_or_else(|| DocStoreError::EmbeddingFailed("semantic search not configured on this store".into()))?;

        let schema = self.load_schema(namespace)?;
        if !schema.semantic_search_enabled {
            return Err(DocStoreError::EmbeddingFailed(format!(
                "namespace '{namespace}' does not have semantic_search_enabled"
            )));
        }

        let (query_dense, query_sparse) = self.cached_query_embeddings(ctx, query_text).await?;

        let db_store = vector_kv::DbVectorStore::new(&self.db, namespace)
            .await
            .map_err(|e| DocStoreError::EmbeddingFailed(e.to_string()))?;

        let all = semantic_search::service::search(
            &ctx.config,
            namespace,
            &ctx.cluster_index,
            &query_sparse,
            &query_dense,
            &db_store,
            Some(move |id: &[u8]| allowed_ids.contains(id)),
            top_k,
        )
        .await;

        Ok(Page::from_vec(all, pagination))
    }

    /// Retrieve a single document by its primary key.
    ///
    /// Returns `None` if no document with that ID exists.
    pub async fn get(&self, namespace: &str, id: DocId) -> Result<Option<serde_json::Value>, DocStoreError> {
        let ns = self.db.namespace(namespace.to_owned()).await?;
        match ns.get(id.to_bytes()).await? {
            None => Ok(None),
            Some(bytes) => {
                let doc = serde_json::from_slice(&bytes).map_err(|e| DocStoreError::InvalidId(e.to_string()))?;
                Ok(Some(doc))
            }
        }
    }

    /// Return documents whose IDs fall in `[start, end)`, one cursor page at a time.
    ///
    /// Pass `end = None` for an open-ended scan to the last key, and `cursor = None`
    /// for the first page; thereafter pass back the previous page's `next_cursor`.
    /// Only the page's documents are resolved from the value log. Results are
    /// ordered by ID ascending.
    pub async fn scan_range(
        &self,
        namespace: &str,
        start: DocId,
        end: Option<DocId>,
        cursor: Option<Vec<u8>>,
        limit: usize,
    ) -> Result<CursorPage<(DocId, serde_json::Value)>, DocStoreError> {
        let schema = self.load_schema(namespace)?;
        let ns = self.db.namespace(namespace.to_owned()).await?;

        let start_bytes = start.to_bytes();
        let end_bytes = end.map(|e| e.to_bytes());
        let scan_start = cursor.unwrap_or(start_bytes);
        let (pairs, next_cursor) = ns.scan(Some(scan_start), end_bytes, limit).await?;

        let results = pairs
            .into_iter()
            .map(|(k, v)| -> Result<_, DocStoreError> {
                let id = DocId::from_bytes(&k, schema.key_type)?;
                let doc = serde_json::from_slice(&v).map_err(|e| DocStoreError::InvalidId(e.to_string()))?;
                Ok((id, doc))
            })
            .collect::<Result<_, _>>()?;

        Ok(CursorPage::new(results, next_cursor))
    }

    /// Return documents whose binary key starts with `prefix`, one cursor page at a time.
    ///
    /// `prefix` is a raw byte slice of the key — callers should encode it in
    /// the same big-endian format used by [`DocId::to_bytes`].  A UUID prefix
    /// of `[0x55, 0x0e, 0x84, 0x00]` (4 bytes) matches every document whose
    /// UUID begins with `550e8400`.
    ///
    /// The prefix is scanned as the range `[prefix, prefix⁺)`, so only the page's
    /// documents are resolved. Pass `cursor = None` for the first page and pass
    /// back `next_cursor` thereafter. Returned in lexicographic key order, matching
    /// [`scan_range`].
    ///
    /// [`scan_range`]: DocStore::scan_range
    pub async fn scan_prefix(
        &self,
        namespace: &str,
        prefix: Vec<u8>,
        cursor: Option<Vec<u8>>,
        limit: usize,
    ) -> Result<CursorPage<(DocId, serde_json::Value)>, DocStoreError> {
        let schema = self.load_schema(namespace)?;
        let end_bytes = prefix_upper_bound(&prefix);
        let scan_start = cursor.unwrap_or(prefix);
        let ns = self.db.namespace(namespace.to_owned()).await?;
        let (pairs, next_cursor) = ns.scan(Some(scan_start), end_bytes, limit).await?;

        let results = pairs
            .into_iter()
            .map(|(k, v)| -> Result<_, DocStoreError> {
                let id = DocId::from_bytes(&k, schema.key_type)?;
                let doc = serde_json::from_slice(&v).map_err(|e| DocStoreError::InvalidId(e.to_string()))?;
                Ok((id, doc))
            })
            .collect::<Result<_, _>>()?;

        Ok(CursorPage::new(results, next_cursor))
    }

    /// Query documents using an index predicate.
    ///
    /// The `predicate` must reference only fields that have active indices in
    /// this store — full collection scans are not supported.
    ///
    /// # Example predicates
    /// ```text
    /// status = "active"
    /// age >= 18 AND verified = true
    /// status = "inactive" OR age < 18
    /// ```
    ///
    /// Returns a [`Page`] of `(DocId, document)` pairs for matching documents.
    pub async fn query(&self, namespace: &str, predicate: &str, pagination: Pagination) -> Result<Page<(DocId, serde_json::Value)>, DocStoreError> {
        let schema = self.load_schema(namespace)?;
        let ns_id = schema.ns_id.ok_or_else(|| DocStoreError::MissingNsId {
            namespace: namespace.to_owned(),
        })?;
        self.query_resolved(namespace, predicate, pagination, ns_id, schema.key_type).await
    }

    /// Like [`query`] but accepts a pre-resolved `ns_id` and `key_type` so the
    /// caller can supply values from an in-memory cache and avoid a disk read.
    ///
    /// [`query`]: DocStore::query
    pub async fn query_resolved(
        &self,
        namespace: &str,
        predicate: &str,
        pagination: Pagination,
        ns_id: u32,
        key_type: KeyType,
    ) -> Result<Page<(DocId, serde_json::Value)>, DocStoreError> {
        // Use the paginated variant so only the page window of keys is resolved
        // from the bitmap, not the full result set.
        let (page_keys, total) = self
            .db
            .query_index_paginated(ns_id, predicate.to_owned(), pagination.offset(), pagination.page_size)
            .await?;

        if total == 0 {
            return Ok(Page::from_slice(vec![], pagination, 0));
        }

        let ns = self.db.namespace(namespace.to_owned()).await?;
        let values = ns.get_multiple(page_keys.clone()).await;
        let mut results = Vec::with_capacity(page_keys.len());
        for (key_bytes, value_opt) in page_keys.into_iter().zip(values) {
            if let Some(bytes) = value_opt {
                let id = DocId::from_bytes(&key_bytes, key_type)?;
                let doc = serde_json::from_slice(&bytes).map_err(|e| DocStoreError::InvalidId(e.to_string()))?;
                results.push((id, doc));
            }
        }

        Ok(Page::from_slice(results, pagination, total))
    }

    /// Collect all raw key bytes that match `predicate` without pagination.
    ///
    /// Used internally by [`search_semantic_filtered`] to build the full
    /// candidate ID set before ANN scoring.
    async fn query_all_keys(&self, namespace: &str, predicate: &str) -> Result<Vec<Vec<u8>>, DocStoreError> {
        let schema = self.load_schema(namespace)?;
        let ns_id = schema.ns_id.ok_or_else(|| DocStoreError::MissingNsId {
            namespace: namespace.to_owned(),
        })?;
        Ok(self.db.query_index(ns_id, predicate.to_owned()).await?)
    }

    // ── Internal helpers ──────────────────────────────────────────────────

    fn schema_path(&self, namespace: &str) -> PathBuf {
        self.schema_dir.join(format!("{namespace}.json"))
    }

    /// Resolve the [`StoreType`] of an existing namespace by reading its
    /// persisted schema's discriminant, without committing to either full schema
    /// struct. Returns [`DocStoreError::NotFound`] if no schema exists (or it
    /// carries no parseable `store_type`).
    pub fn store_type(&self, namespace: &str) -> Result<StoreType, DocStoreError> {
        let path = self.schema_path(namespace);
        let json = std::fs::read_to_string(&path).map_err(|_| DocStoreError::NotFound {
            namespace: namespace.to_owned(),
        })?;
        crate::schema::peek_store_type(&json).ok_or_else(|| DocStoreError::NotFound {
            namespace: namespace.to_owned(),
        })
    }

    fn load_schema(&self, namespace: &str) -> Result<DocStoreSchema, DocStoreError> {
        DocStoreSchema::load(&self.schema_dir, namespace).map_err(|e| match e {
            SchemaError::NotFound { namespace } => DocStoreError::NotFound { namespace },
            other => DocStoreError::Schema(other),
        })
    }

    fn load_kv_schema(&self, namespace: &str) -> Result<KvStoreSchema, DocStoreError> {
        KvStoreSchema::load(&self.schema_dir, namespace).map_err(|e| match e {
            SchemaError::NotFound { namespace } => DocStoreError::NotFound { namespace },
            other => DocStoreError::Schema(other),
        })
    }

    fn load_all_schemas(&self) -> Result<Vec<DocStoreSchema>, DocStoreError> {
        load_all_schemas_from(&self.schema_dir)
    }

    fn load_all_kv_schemas(&self) -> Result<Vec<KvStoreSchema>, DocStoreError> {
        load_all_kv_schemas_from(&self.schema_dir)
    }
}

/// Load every persisted document-store schema from `schema_dir`.
fn load_all_schemas_from(schema_dir: &Path) -> Result<Vec<DocStoreSchema>, DocStoreError> {
    let mut schemas = Vec::new();
    let entries = match std::fs::read_dir(schema_dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(schemas),
        Err(e) => return Err(DocStoreError::Io(e)),
    };
    for entry in entries.filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("json") {
            let json = std::fs::read_to_string(&path)?;
            // Dispatch on the explicit `store_type` discriminant; skip anything
            // that isn't a doc store (or has no parseable discriminant).
            if crate::schema::peek_store_type(&json) == Some(StoreType::Doc)
                && let Ok(schema) = serde_json::from_str::<DocStoreSchema>(&json)
            {
                schemas.push(schema);
            }
        }
    }
    Ok(schemas)
}

/// Load every persisted KV-store schema from `schema_dir`.
fn load_all_kv_schemas_from(schema_dir: &Path) -> Result<Vec<KvStoreSchema>, DocStoreError> {
    let mut schemas = Vec::new();
    let entries = match std::fs::read_dir(schema_dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(schemas),
        Err(e) => return Err(DocStoreError::Io(e)),
    };
    for entry in entries.filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("json") {
            let json = std::fs::read_to_string(&path)?;
            // Dispatch on the explicit `store_type` discriminant; skip anything
            // that isn't a KV store (or has no parseable discriminant).
            if crate::schema::peek_store_type(&json) == Some(StoreType::Kv)
                && let Ok(schema) = serde_json::from_str::<KvStoreSchema>(&json)
            {
                schemas.push(schema);
            }
        }
    }
    Ok(schemas)
}

// ── Vector-index reconciliation ────────────────────────────────────────────────

/// Reconcile vector indexes across every semantic-search-enabled namespace.
///
/// For each such namespace, enqueue any document that has **neither** a
/// committed vector index entry **nor** a pending queue entry — i.e. the
/// `put` / `kv_put` crash window (document durably written, embed enqueue lost).
/// This is the vector-index analogue of how field indices self-heal on startup
/// via WAL replay: the difference is the work is routed into the async embedding
/// queue rather than rebuilt inline, and the [`VecIndexWorker`] drains it when
/// the embedding service is available.
///
/// Returns the total number of documents re-enqueued.  Errors on individual
/// namespaces are logged and skipped so one bad namespace cannot abort the rest.
///
/// [`VecIndexWorker`]: crate::vec_index_worker::VecIndexWorker
/// Outcome of a [`reconcile_all_vector_indexes`] pass.
struct ReconcileOutcome {
    /// Documents re-enqueued for embedding across all namespaces.
    reenqueued: usize,
    /// Number of namespaces whose reconciliation failed (each is also logged
    /// individually via `warn!`).  Non-zero means the pass did not fully
    /// complete and should be re-run.
    failed: usize,
}

/// Reconcile every semantic-search namespace's vector index.
///
/// With `check_bytes == false` (the cheap, default pass used at startup and by the
/// presence-only reconcile) a document is "indexed" when both companion halves are
/// **present** ([`vector_kv::has_complete_vector_index`]), and a count short-circuit
/// skips namespaces already fully covered. With `check_bytes == true` (the on-demand
/// *validating* pass) it instead deserializes each entry ([`vector_kv::has_valid_vector_index`])
/// to catch present-but-corrupt vectors, and skips the count short-circuit (corruption
/// is not count-detectable) — a full value-reading scan, hence run in the background.
async fn reconcile_all_vector_indexes(db: &AsyncDb, schema_dir: &Path, check_bytes: bool) -> ReconcileOutcome {
    // Scan the pending queue once and count entries per namespace, so each
    // namespace's cheap short-circuit can test `pending == 0` without re-scanning.
    let pending_by_ns = {
        let mut map: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for entry in vector_kv::list_queue_entries(db).await.unwrap_or_default() {
            *map.entry(entry.namespace).or_default() += 1;
        }
        map
    };
    let pending_for = |ns: &str| pending_by_ns.get(ns).copied().unwrap_or(0);

    let mut reenqueued = 0usize;
    let mut failed = 0usize;

    for schema in load_all_schemas_from(schema_dir).unwrap_or_default() {
        if !schema.is_semantic_search_enabled() {
            continue;
        }
        match reconcile_doc_namespace_vectors(db, &schema, pending_for(&schema.namespace), check_bytes).await {
            Ok(n) => reenqueued += n,
            Err(e) => {
                failed += 1;
                warn!("vec reconcile: doc namespace '{}' failed: {e}", schema.namespace);
            }
        }
    }

    for schema in load_all_kv_schemas_from(schema_dir).unwrap_or_default() {
        if !schema.is_semantic_search_enabled() {
            continue;
        }
        match reconcile_kv_namespace_vectors(db, &schema, pending_for(&schema.namespace), check_bytes).await {
            Ok(n) => reenqueued += n,
            Err(e) => {
                failed += 1;
                warn!("vec reconcile: kv namespace '{}' failed: {e}", schema.namespace);
            }
        }
    }

    ReconcileOutcome { reenqueued, failed }
}

/// Count keys in `namespace` without reading their values (LSM-only scan).
async fn count_keys(db: &AsyncDb, namespace: &str) -> Result<usize, DocStoreError> {
    let ns = db.namespace(namespace.to_owned()).await?;
    Ok(ns.keys().await?.len())
}

/// Count keys in a vector-index companion namespace (sparse-meta or dense)
/// without reading values.  Returns 0 if the companion namespace does not exist
/// yet.
async fn count_companion(db: &AsyncDb, companion_ns: String) -> usize {
    match db.namespace(companion_ns).await {
        Ok(ns) => ns.keys().await.map(|k| k.len()).unwrap_or(0),
        Err(_) => 0,
    }
}

/// Cheap short-circuit shared by the doc and KV reconcilers: when nothing is
/// queued for the namespace and every key already has a **complete** committed
/// vector index, there is nothing to reconcile and the full per-doc scan can be
/// skipped.
///
/// A complete index requires **both** a sparse-meta and a dense entry per key
/// (see [`vector_kv::has_complete_vector_index`]), so the short-circuit must
/// require both companion counts to reach `key_count` — otherwise a namespace
/// where every key has sparse-meta but some lost the no-WAL dense write would be
/// skipped despite being partially indexed.
///
/// Sound because the delete ordering guarantees vector-index, meta, and queue
/// entries never outlive their document — so the meta set, dense set, and queue
/// set are all subsets of the live keys. With no orphans, both counts reaching
/// `key_count` (and `pending == 0`) implies every live key is fully indexed.
/// Namespaces with empty-embedding-text documents simply fall through to the
/// full scan (which then enqueues nothing) — correctness is preserved, only the
/// optimisation is skipped. All counts are LSM-only key scans (no value-log
/// reads), so a clean boot avoids the expensive value-loading `iter`.
async fn nothing_to_reconcile(db: &AsyncDb, namespace: &str, pending_for_ns: usize) -> bool {
    if pending_for_ns != 0 {
        return false;
    }
    let Ok(key_count) = count_keys(db, namespace).await else {
        return false;
    };
    count_companion(db, vector_kv::sparse_vectors_meta_ns(namespace)).await >= key_count
        && count_companion(db, vector_kv::dense_vectors_ns(namespace)).await >= key_count
}

/// Reconcile one document-store namespace.  See [`reconcile_all_vector_indexes`].
async fn reconcile_doc_namespace_vectors(
    db: &AsyncDb,
    schema: &DocStoreSchema,
    pending_for_ns: usize,
    check_bytes: bool,
) -> Result<usize, DocStoreError> {
    let namespace = &schema.namespace;
    // The count short-circuit only sees presence, so it cannot detect corrupt-but-
    // present entries — skip it for the validating pass and scan every document.
    if !check_bytes && nothing_to_reconcile(db, namespace, pending_for_ns).await {
        return Ok(0);
    }

    let ns = db.namespace(namespace.clone()).await?;
    let all_docs = ns.iter().await?;

    let mut enqueued = 0usize;
    for (key, value) in &all_docs {
        if vector_kv::get_queue_entry(db, namespace, key).await?.is_some() {
            continue;
        }
        let indexed = if check_bytes {
            vector_kv::has_valid_vector_index(db, namespace, key).await?
        } else {
            vector_kv::has_complete_vector_index(db, namespace, key).await?
        };
        if indexed {
            continue;
        }
        let Ok(doc) = serde_json::from_slice::<serde_json::Value>(value) else {
            continue;
        };
        let text = build_embedding_text(&doc, &schema.embedding_fields);
        if !text.is_empty() {
            vector_kv::enqueue_embed(db, namespace, key, &text).await?;
            enqueued += 1;
        }
    }

    if enqueued > 0 {
        info!(
            "vec reconcile: doc namespace '{}' re-enqueued {} missing document(s)",
            namespace, enqueued
        );
    }
    Ok(enqueued)
}

/// Reconcile one KV-store namespace.  See [`reconcile_all_vector_indexes`].
async fn reconcile_kv_namespace_vectors(
    db: &AsyncDb,
    schema: &KvStoreSchema,
    pending_for_ns: usize,
    check_bytes: bool,
) -> Result<usize, DocStoreError> {
    let namespace = &schema.namespace;
    if !check_bytes && nothing_to_reconcile(db, namespace, pending_for_ns).await {
        return Ok(0);
    }

    let ns = db.namespace(namespace.clone()).await?;
    let all_entries = ns.iter().await?;

    let mut enqueued = 0usize;
    for (key, value_bytes) in &all_entries {
        if vector_kv::get_queue_entry(db, namespace, key).await?.is_some() {
            continue;
        }
        let indexed = if check_bytes {
            vector_kv::has_valid_vector_index(db, namespace, key).await?
        } else {
            vector_kv::has_complete_vector_index(db, namespace, key).await?
        };
        if indexed {
            continue;
        }
        if let Ok(text) = std::str::from_utf8(value_bytes)
            && !text.is_empty()
        {
            vector_kv::enqueue_embed(db, namespace, key, text).await?;
            enqueued += 1;
        }
    }

    if enqueued > 0 {
        info!("vec reconcile: kv namespace '{}' re-enqueued {} missing entry(ies)", namespace, enqueued);
    }
    Ok(enqueued)
}

// ── Namespace cleanup helper ──────────────────────────────────────────────────

/// Remove a namespace (and its vector-index companions) from the engine registry
/// and delete all on-disk directories and the schema file.
///
/// Shared by [`DocStore::drop`] and [`DocStore::drop_kv`] to
/// eliminate the near-identical cleanup sequences in each method.
async fn cleanup_store_namespaces(db: &AsyncDb, db_path: &Path, namespace: &str, ns_id: u32, schema_path: &Path) -> Result<(), DocStoreError> {
    // Primary namespace must exist; propagate error before touching files.
    // Vector companions are optional — ignore errors on removal.
    db.remove_namespace(namespace.to_owned()).await?;
    let _ = db.remove_namespace(vector_kv::sparse_vectors_ns(namespace)).await;
    let _ = db.remove_namespace(vector_kv::sparse_vectors_meta_ns(namespace)).await;
    let _ = db.remove_namespace(vector_kv::dense_vectors_ns(namespace)).await;

    for ns_name in &[
        namespace.to_owned(),
        vector_kv::sparse_vectors_ns(namespace),
        vector_kv::sparse_vectors_meta_ns(namespace),
        vector_kv::dense_vectors_ns(namespace),
    ] {
        let ns_dir = db_path.join(format!("ns_{}", ns_name));
        if ns_dir.exists() {
            std::fs::remove_dir_all(&ns_dir)?;
        }
    }

    let index_dir = db_path.join("index").join(ns_id.to_string());
    if index_dir.exists() {
        std::fs::remove_dir_all(&index_dir)?;
    }

    if schema_path.exists() {
        std::fs::remove_file(schema_path)?;
    }

    // Drop any in-memory vector corruption counters for this namespace so a
    // dropped store stops appearing in /admin/indices/vector/corruption-metrics.
    semantic_search::metrics::reset(namespace);

    Ok(())
}

// ── Background index rebuild ──────────────────────────────────────────────────

/// Scan every document in `ns_name` and re-put it so that the new extractor
/// is called for each, populating the freshly activated index with historical
/// data.
///
/// Progress is reported via `observer` (e.g. atomics + disk JSON) every 1 000
/// documents and on terminal status changes.  When `resume_after` is `Some(key)`,
/// all documents with keys ≤ that key are skipped (they were already processed
/// before the previous shutdown).
#[allow(clippy::too_many_arguments)]
/// Number of `(key, value)` pairs fetched per cursor page during an index
/// rebuild. Bounds peak memory to roughly one page of documents instead of the
/// whole namespace, and matches the progress/yield cadence below.
const REBUILD_PAGE_SIZE: usize = 1_000;

/// Smallest key strictly greater than `key`, used to advance a (inclusive)
/// scan cursor past an already-processed key. Appending a `0x00` byte yields a
/// key that sorts immediately after `key` in lexicographic order.
fn successor_key(key: &[u8]) -> Vec<u8> {
    let mut next = Vec::with_capacity(key.len() + 1);
    next.extend_from_slice(key);
    next.push(0);
    next
}

#[allow(clippy::too_many_arguments)] // cohesive set of build parameters; not worth a struct
async fn rebuild_index_for_namespace(
    db: Arc<AsyncDb>,
    ns_name: String,
    key_type: KeyType,
    field_id: FieldId,
    resume_after: Option<Vec<u8>>,
    observer: Arc<dyn IndexProgressObserver>,
) -> Result<(), DocStoreError> {
    let ns = db.namespace(ns_name.clone()).await?;

    // ── Pass 1: count keys, cursor-paginated. We need the exact total up front
    // (it drives the build's percent-complete), but materialising the whole
    // namespace just to count it is the memory spike this rebuild used to incur.
    // Scanning a page at a time and discarding each page bounds peak memory to
    // one page. We also count how many keys were already processed on a resumed
    // build, to seed `indexed`.
    let mut total: u64 = 0;
    let mut already_done: u64 = 0;
    {
        let mut cursor: Option<Vec<u8>> = None;
        loop {
            let (pairs, next_cursor) = ns.scan(cursor, None, REBUILD_PAGE_SIZE).await?;
            for (key, _value) in &pairs {
                total += 1;
                if resume_after.as_ref().is_some_and(|r| key <= r) {
                    already_done += 1;
                }
            }
            match next_cursor {
                Some(c) => cursor = Some(c),
                None => break,
            }
        }
    }

    info!("index build started: namespace='{}' field_id={} total={} docs", ns_name, field_id, total);

    // Seed the observer with the counts from the scan, then announce Running.
    // The observer (its `DiskProgress` link) is the single source of truth for
    // persistence: seeding before `on_status(Running)` makes the initial
    // in_progress record carry the real total/resume point.
    observer.on_progress(already_done, total, false, resume_after.as_deref());
    observer.on_status(BuildStatus::Running, None);

    let mut indexed: u64 = already_done;

    // ── Pass 2: rebuild, one cursor page at a time. On a resumed build the
    // cursor seeks strictly past the last processed key (the scan cursor is
    // inclusive, so start at its successor) instead of re-scanning and skipping.
    let mut cursor: Option<Vec<u8>> = resume_after.as_deref().map(successor_key);
    'pages: loop {
        let (pairs, next_cursor) = ns.scan(cursor, None, REBUILD_PAGE_SIZE).await?;
        if pairs.is_empty() {
            break;
        }
        for (key, value) in pairs {
            // Re-put triggers the active extractors, populating the new index.
            ns.put(key.clone(), value).await?;
            indexed += 1;
            // The observer persists to disk on its own cadence (every `every_n`).
            observer.on_progress(indexed, total, false, Some(&key));

            // Log progress and yield every 1 000 documents.
            if indexed.is_multiple_of(1_000) {
                info!(
                    "index build progress: namespace='{}' field_id={} {}/{} docs ({:.1}%)",
                    ns_name,
                    field_id,
                    indexed,
                    total,
                    if total > 0 { indexed as f64 / total as f64 * 100.0 } else { 0.0 }
                );
                tokio::task::yield_now().await;
            }
        }
        match next_cursor {
            Some(c) => cursor = Some(c),
            None => break 'pages,
        }
    }

    // Mark build complete.  The observer persists the terminal record from the
    // latest snapshot it has been fed via `on_progress`.
    observer.on_status(BuildStatus::Complete, None);

    info!(
        "index build complete: namespace='{}' field_id={} indexed={} docs",
        ns_name, field_id, indexed
    );
    let _ = key_type;
    Ok(())
}

// ── Lock-file cleanup ─────────────────────────────────────────────────────────

impl Drop for DocStore {
    fn drop(&mut self) {
        if self.lock_path.exists() {
            let _ = std::fs::remove_file(&self.lock_path);
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{AttributeType, IndexSpec, IndexType, KeyType, KvKeyType, KvValueType};
    use tempfile::TempDir;

    fn make_schema(namespace: &str, indices: Vec<IndexSpec>) -> DocStoreSchema {
        DocStoreSchema {
            store_type: StoreType::Doc,
            namespace: namespace.to_owned(),
            ns_id: None,
            key_type: KeyType::U64,
            attributes: vec![],
            indices,
            semantic_search_enabled: false,
            embedding_fields: vec![],
        }
    }

    async fn open_fresh(db_dir: &Path, schema_dir: &Path) -> DocStore {
        DocStore::open(db_dir, schema_dir).await.unwrap()
    }

    // ── create / list / drop ────────────────────────────────────────────────

    #[tokio::test]
    async fn test_create_and_list() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;

        let schema = make_schema(
            "users",
            vec![IndexSpec {
                field: "active".to_owned(),
                index_type: IndexType::Bool,
            }],
        );
        store.create(schema).await.unwrap();

        let list = store.list().unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0]["namespace"], "users");
        assert!(list[0]["ns_id"].as_u64().is_some());
    }

    #[tokio::test]
    async fn test_create_duplicate_is_rejected() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;

        store.create(make_schema("dup", vec![])).await.unwrap();
        let err = store.create(make_schema("dup", vec![])).await.unwrap_err();
        assert!(matches!(err, DocStoreError::AlreadyExists { .. }));
    }

    #[tokio::test]
    async fn test_drop_store_removes_schema_file() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;

        store.create(make_schema("tmp", vec![])).await.unwrap();
        assert!(schema_dir.path().join("tmp.json").exists());

        store.remove("tmp").await.unwrap();
        assert!(!schema_dir.path().join("tmp.json").exists());
    }

    // ── CRUD ────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_put_and_find_by_id() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store.create(make_schema("docs", vec![])).await.unwrap();

        let id = DocId::U64(42);
        let doc = serde_json::json!({"name": "Alice", "age": 30});
        store.put("docs", id, doc.clone()).await.unwrap();

        let found = store.get("docs", id).await.unwrap();
        assert_eq!(found, Some(doc));
    }

    #[tokio::test]
    async fn test_find_by_id_missing_returns_none() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store.create(make_schema("docs", vec![])).await.unwrap();

        let found = store.get("docs", DocId::U64(99)).await.unwrap();
        assert_eq!(found, None);
    }

    #[tokio::test]
    async fn test_delete_removes_document() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store.create(make_schema("docs", vec![])).await.unwrap();

        let id = DocId::U64(1);
        store.put("docs", id, serde_json::json!({"x": 1})).await.unwrap();
        store.delete("docs", id).await.unwrap();
        assert_eq!(store.get("docs", id).await.unwrap(), None);
    }

    // ── scan_prefix after delete ─────────────────────────────────────────

    /// Insert several documents, prefix-scan to verify them, delete one,
    /// then prefix-scan again and assert the deleted document is gone.
    #[tokio::test]
    async fn test_scan_by_prefix_after_delete() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store.create(make_schema("docs", vec![])).await.unwrap();

        // Insert 5 documents with sequential u64 IDs.
        for i in 1u64..=5 {
            store.put("docs", DocId::U64(i), serde_json::json!({"n": i})).await.unwrap();
        }

        // U64 keys are 8 bytes big-endian; an empty prefix matches all.
        let before = store.scan_prefix("docs", vec![], None, 100).await.unwrap();
        assert_eq!(before.results.len(), 5, "expected 5 docs before delete, got {}", before.results.len());

        // Delete doc with id=3.
        store.delete("docs", DocId::U64(3)).await.unwrap();

        // Point-get must return None.
        assert_eq!(store.get("docs", DocId::U64(3)).await.unwrap(), None, "doc 3 should be gone after delete");

        // Prefix scan must now return 4 docs, without doc 3.
        let after = store.scan_prefix("docs", vec![], None, 100).await.unwrap();
        assert_eq!(after.results.len(), 4, "expected 4 docs after delete, got {}", after.results.len());
        let ids_after: Vec<u64> = after
            .results
            .iter()
            .map(|(id, _)| match id {
                DocId::U64(v) => *v,
                _ => panic!("unexpected DocId variant"),
            })
            .collect();
        assert!(!ids_after.contains(&3), "deleted doc 3 must not appear in prefix scan");
        assert_eq!(ids_after, vec![1, 2, 4, 5]);
    }

    /// Same scenario but with `semantic_search_enabled = true`, which makes
    /// `delete()` take the semantic-search path (cancel pending embed + delete
    /// vector + delete doc).  No actual embedding service is needed because
    /// writes without `with_semantic_search()` attached fall back to the
    /// regular `ns.put()` path, while deletes always go through the
    /// semantic-search path when the schema flag is set.
    #[tokio::test]
    async fn test_scan_by_prefix_after_delete_semantic_search_enabled() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;

        // Create a schema with semantic_search_enabled + embedding_fields so
        // that `is_semantic_search_enabled()` returns true.
        let schema = DocStoreSchema {
            store_type: StoreType::Doc,
            namespace: "sem_docs".to_owned(),
            ns_id: None,
            key_type: KeyType::U64,
            attributes: vec![crate::schema::AttributeDef {
                name: "title".to_owned(),
                attr_type: AttributeType::Str,
                description: None,
            }],
            indices: vec![],
            semantic_search_enabled: true,
            embedding_fields: vec!["title".to_owned()],
        };
        store.create(schema).await.unwrap();

        // insert — self.notify is None (no SemanticSearchContext attached) so
        // put() falls through to the regular ns.put() path.
        for i in 1u64..=5 {
            store
                .put("sem_docs", DocId::U64(i), serde_json::json!({"title": format!("doc {}", i)}))
                .await
                .unwrap();
        }

        // Verify all 5 present.
        let before = store.scan_prefix("sem_docs", vec![], None, 100).await.unwrap();
        assert_eq!(before.results.len(), 5);

        // delete — schema.is_semantic_search_enabled() is true so this takes
        // the semantic-search path: remove_queue_entry + delete_vector +
        // ns.delete.
        store.delete("sem_docs", DocId::U64(2)).await.unwrap();

        // Point-get must return None.
        assert_eq!(
            store.get("sem_docs", DocId::U64(2)).await.unwrap(),
            None,
            "doc 2 should be gone after delete"
        );

        // Prefix scan must reflect the deletion.
        let after = store.scan_prefix("sem_docs", vec![], None, 100).await.unwrap();
        assert_eq!(
            after.results.len(),
            4,
            "expected 4 docs after deleting doc 2 (semantic path), got {}",
            after.results.len()
        );
        let ids_after: Vec<u64> = after
            .results
            .iter()
            .map(|(id, _)| match id {
                DocId::U64(v) => *v,
                _ => panic!("unexpected DocId variant"),
            })
            .collect();
        assert!(!ids_after.contains(&2), "deleted doc 2 must not appear in prefix scan (semantic path)");
        assert_eq!(ids_after, vec![1, 3, 4, 5]);
    }

    // ── Range query ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_query_range() {
        // ...existing test...
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store.create(make_schema("docs", vec![])).await.unwrap();

        for i in 1u64..=5 {
            store.put("docs", DocId::U64(i), serde_json::json!({"n": i})).await.unwrap();
        }

        let result = store.scan_range("docs", DocId::U64(2), Some(DocId::U64(4)), None, 100).await.unwrap();
        let ids: Vec<u64> = result
            .results
            .iter()
            .map(|(id, _)| match id {
                DocId::U64(v) => *v,
                _ => panic!(),
            })
            .collect();
        assert_eq!(ids, vec![2, 3]);
    }

    /// Walk a range scan page-by-page via `next_cursor` and confirm the union of
    /// pages is the full, in-order result with no key dropped or duplicated.
    #[tokio::test]
    async fn test_query_range_cursor_pagination_walk() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store.create(make_schema("docs", vec![])).await.unwrap();

        for i in 1u64..=5 {
            store.put("docs", DocId::U64(i), serde_json::json!({"n": i})).await.unwrap();
        }

        let mut cursor: Option<Vec<u8>> = None;
        let mut ids: Vec<u64> = Vec::new();
        let mut pages = 0;
        loop {
            let page = store.scan_range("docs", DocId::U64(1), None, cursor.clone(), 2).await.unwrap();
            assert!(page.results.len() <= 2, "page must not exceed the limit");
            pages += 1;
            for (id, _) in &page.results {
                match id {
                    DocId::U64(v) => ids.push(*v),
                    _ => panic!(),
                }
            }
            match page.next_cursor {
                Some(c) => cursor = Some(c),
                None => break,
            }
        }

        assert_eq!(ids, vec![1, 2, 3, 4, 5], "cursor walk must return every doc once, in order");
        assert_eq!(pages, 3, "5 docs at limit 2 → pages of 2, 2, 1");
    }

    /// Range scan must exclude a deleted document that falls within the range.
    #[tokio::test]
    async fn test_query_range_after_delete() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store.create(make_schema("docs", vec![])).await.unwrap();

        for i in 1u64..=6 {
            store.put("docs", DocId::U64(i), serde_json::json!({"n": i})).await.unwrap();
        }

        // Range [2, 6) before delete → 2, 3, 4, 5
        let before = store.scan_range("docs", DocId::U64(2), Some(DocId::U64(6)), None, 100).await.unwrap();
        let ids_before: Vec<u64> = before
            .results
            .iter()
            .map(|(id, _)| match id {
                DocId::U64(v) => *v,
                _ => panic!(),
            })
            .collect();
        assert_eq!(ids_before, vec![2, 3, 4, 5]);

        // Delete doc 3 (inside range) and doc 5 (inside range).
        store.delete("docs", DocId::U64(3)).await.unwrap();
        store.delete("docs", DocId::U64(5)).await.unwrap();

        // Range [2, 6) after delete → 2, 4
        let after = store.scan_range("docs", DocId::U64(2), Some(DocId::U64(6)), None, 100).await.unwrap();
        let ids_after: Vec<u64> = after
            .results
            .iter()
            .map(|(id, _)| match id {
                DocId::U64(v) => *v,
                _ => panic!(),
            })
            .collect();
        assert_eq!(ids_after, vec![2, 4]);

        // Docs outside the deleted set are untouched.
        assert!(store.get("docs", DocId::U64(1)).await.unwrap().is_some());
        assert!(store.get("docs", DocId::U64(6)).await.unwrap().is_some());
    }

    /// Open-ended range scan (no upper bound) after deletion.
    #[tokio::test]
    async fn test_query_range_open_ended_after_delete() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store.create(make_schema("docs", vec![])).await.unwrap();

        for i in 1u64..=4 {
            store.put("docs", DocId::U64(i), serde_json::json!({"v": i})).await.unwrap();
        }

        store.delete("docs", DocId::U64(2)).await.unwrap();

        // Open-ended range from 1 → should return 1, 3, 4
        let result = store.scan_range("docs", DocId::U64(1), None, None, 100).await.unwrap();
        let ids: Vec<u64> = result
            .results
            .iter()
            .map(|(id, _)| match id {
                DocId::U64(v) => *v,
                _ => panic!(),
            })
            .collect();
        assert_eq!(ids, vec![1, 3, 4]);
    }

    /// Range scan with `semantic_search_enabled` (semantic-search delete path).
    #[tokio::test]
    async fn test_query_range_after_delete_semantic_search_enabled() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;

        let schema = DocStoreSchema {
            store_type: StoreType::Doc,
            namespace: "sem_range".to_owned(),
            ns_id: None,
            key_type: KeyType::U64,
            attributes: vec![crate::schema::AttributeDef {
                name: "title".to_owned(),
                attr_type: AttributeType::Str,
                description: None,
            }],
            indices: vec![],
            semantic_search_enabled: true,
            embedding_fields: vec!["title".to_owned()],
        };
        store.create(schema).await.unwrap();

        for i in 1u64..=5 {
            store
                .put("sem_range", DocId::U64(i), serde_json::json!({"title": format!("doc {}", i)}))
                .await
                .unwrap();
        }

        // Delete via the semantic-search path.
        store.delete("sem_range", DocId::U64(3)).await.unwrap();

        // Range [1, 5) → should be 1, 2, 4
        let result = store
            .scan_range("sem_range", DocId::U64(1), Some(DocId::U64(5)), None, 100)
            .await
            .unwrap();
        let ids: Vec<u64> = result
            .results
            .iter()
            .map(|(id, _)| match id {
                DocId::U64(v) => *v,
                _ => panic!(),
            })
            .collect();
        assert_eq!(ids, vec![1, 2, 4], "deleted doc 3 must not appear in range scan (semantic path)");
    }

    // ── Index query ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_query_by_index() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;

        store
            .create(make_schema(
                "users",
                vec![IndexSpec {
                    field: "active".to_owned(),
                    index_type: IndexType::Bool,
                }],
            ))
            .await
            .unwrap();

        store
            .put("users", DocId::U64(1), serde_json::json!({"active": true,  "name": "Alice"}))
            .await
            .unwrap();
        store
            .put("users", DocId::U64(2), serde_json::json!({"active": false, "name": "Bob"}))
            .await
            .unwrap();
        store
            .put("users", DocId::U64(3), serde_json::json!({"active": true,  "name": "Carol"}))
            .await
            .unwrap();

        let active = store.query("users", "active = true", Pagination::default()).await.unwrap();
        let mut ids: Vec<u64> = active
            .results
            .iter()
            .map(|(id, _)| match id {
                DocId::U64(v) => *v,
                _ => panic!(),
            })
            .collect();
        ids.sort();
        assert_eq!(ids, vec![1, 3]);
    }

    // ── Schema amendment ────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_amend_add_and_remove_attribute() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store.create(make_schema("ns", vec![])).await.unwrap();

        store
            .amend(
                "ns",
                SchemaAmendment::AddAttribute {
                    name: "email".to_owned(),
                    attr_type: AttributeType::Str,
                    description: None,
                },
            )
            .unwrap();

        let loaded = DocStoreSchema::load(schema_dir.path(), "ns").unwrap();
        assert_eq!(loaded.attributes.len(), 1);
        assert_eq!(loaded.attributes[0].name, "email");

        store.amend("ns", SchemaAmendment::RemoveAttribute { name: "email".to_owned() }).unwrap();
        let loaded2 = DocStoreSchema::load(schema_dir.path(), "ns").unwrap();
        assert!(loaded2.attributes.is_empty());
    }

    #[tokio::test]
    async fn test_amend_cannot_remove_indexed_attribute() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store
            .create(make_schema(
                "ns",
                vec![IndexSpec {
                    field: "status".to_owned(),
                    index_type: IndexType::Str,
                }],
            ))
            .await
            .unwrap();

        let err = store
            .amend("ns", SchemaAmendment::RemoveAttribute { name: "status".to_owned() })
            .unwrap_err();
        assert!(
            matches!(err, DocStoreError::AttributeIsIndexed { .. }),
            "expected AttributeIsIndexed, got {:?}",
            err
        );
    }

    // ── Drop / add index ────────────────────────────────────────────────────

    /// `add_index` must enforce the per-namespace `MAX_INDICES` cap incrementally,
    /// not just at create/import time (`schema.save()` does not validate).
    #[tokio::test]
    async fn test_add_index_enforces_max_indices() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;

        // Start at the cap with MAX_INDICES indices.
        let indices = (0..crate::schema::MAX_INDICES)
            .map(|i| IndexSpec {
                field: format!("f{i}"),
                index_type: IndexType::Int,
            })
            .collect();
        store.create(make_schema("ns", indices)).await.unwrap();

        // One more must be rejected (not silently accepted past the cap).
        let result = store
            .add_index(
                "ns",
                IndexSpec {
                    field: "one_too_many".to_owned(),
                    index_type: IndexType::Int,
                },
            )
            .await;
        match result {
            Err(DocStoreError::Schema(crate::error::SchemaError::TooManyIndices { max, .. })) => {
                assert_eq!(max, crate::schema::MAX_INDICES);
            }
            Ok(_) => panic!("expected TooManyIndices, got Ok"),
            Err(e) => panic!("expected TooManyIndices, got {e:?}"),
        }

        // And the rejected field must not have been registered/persisted.
        let loaded = DocStoreSchema::load(schema_dir.path(), "ns").unwrap();
        assert_eq!(loaded.indices.len(), crate::schema::MAX_INDICES);
        assert!(!loaded.indices.iter().any(|s| s.field == "one_too_many"));
    }

    #[tokio::test]
    async fn test_drop_index_demotes_to_attribute() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store
            .create(make_schema(
                "ns",
                vec![IndexSpec {
                    field: "status".to_owned(),
                    index_type: IndexType::Str,
                }],
            ))
            .await
            .unwrap();

        store.drop_index("ns", "status").unwrap();

        let loaded = DocStoreSchema::load(schema_dir.path(), "ns").unwrap();
        // Index is gone
        assert!(loaded.indices.is_empty(), "index should be removed");
        // Field is preserved as a non-indexed attribute
        assert_eq!(loaded.attributes.len(), 1);
        assert_eq!(loaded.attributes[0].name, "status");
        assert_eq!(loaded.attributes[0].attr_type, AttributeType::Str);
    }

    /// After `drop_index` the in-memory bitmap must be gone: predicate queries
    /// on the dropped field must return an error in the same process, not stale
    /// results from the previously-populated bitmap.
    #[tokio::test]
    async fn test_drop_index_deactivates_in_memory_index() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store
            .create(make_schema(
                "ns",
                vec![IndexSpec {
                    field: "status".to_owned(),
                    index_type: IndexType::Str,
                }],
            ))
            .await
            .unwrap();

        store.put("ns", DocId::U64(1), serde_json::json!({"status": "active"})).await.unwrap();
        store.put("ns", DocId::U64(2), serde_json::json!({"status": "inactive"})).await.unwrap();

        // Sanity: query works before drop.
        let results = store.query("ns", "status = \"active\"", Pagination::default()).await.unwrap();
        assert_eq!(results.results.len(), 1, "should find one active doc before drop");

        store.drop_index("ns", "status").unwrap();

        // After drop, the same query must fail — not silently return stale hits.
        let err = store.query("ns", "status = \"active\"", Pagination::default()).await.unwrap_err();
        assert!(
            err.to_string().contains("unknown field 'status'"),
            "expected unknown field 'status', got: {err}",
        );
    }

    #[tokio::test]
    async fn test_add_index_and_wait() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store.create(make_schema("ns", vec![])).await.unwrap();

        // Insert some documents before the index exists
        for i in 0u64..5 {
            store
                .put("ns", DocId::U64(i), serde_json::json!({"status": "active", "n": i}))
                .await
                .unwrap();
        }

        // Add an index — background task builds it on existing data
        let handle = store
            .add_index(
                "ns",
                IndexSpec {
                    field: "status".to_owned(),
                    index_type: IndexType::Str,
                },
            )
            .await
            .unwrap();

        handle.wait().await.unwrap();

        // Schema must now include the index
        let loaded = DocStoreSchema::load(schema_dir.path(), "ns").unwrap();
        assert_eq!(loaded.indices.len(), 1);
        assert_eq!(loaded.indices[0].field, "status");

        // Query must return all 5 docs
        let result = store.query("ns", "status = \"active\"", Pagination::default()).await.unwrap();
        assert_eq!(result.total, 5);
    }

    /// The rebuild walks the namespace one cursor page at a time; index more
    /// documents than a single page so the page-boundary advancement (in both
    /// the count pass and the rebuild pass) is exercised, with a partial final
    /// page.
    #[tokio::test]
    async fn test_add_index_rebuild_spans_multiple_pages() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store.create(make_schema("ns", vec![])).await.unwrap();

        let n = REBUILD_PAGE_SIZE + 7; // > 1 page, partial last page
        for i in 0..n {
            let status = if i % 2 == 0 { "active" } else { "inactive" };
            store
                .put("ns", DocId::U64(i as u64), serde_json::json!({"status": status, "n": i}))
                .await
                .unwrap();
        }

        let handle = store
            .add_index(
                "ns",
                IndexSpec {
                    field: "status".to_owned(),
                    index_type: IndexType::Str,
                },
            )
            .await
            .unwrap();
        handle.wait().await.unwrap();

        // Every document across the page boundary must be indexed, partitioned
        // correctly — no key skipped or double-counted at a page edge.
        let active = store.query("ns", "status = \"active\"", Pagination::default()).await.unwrap();
        let inactive = store.query("ns", "status = \"inactive\"", Pagination::default()).await.unwrap();
        assert_eq!(active.total + inactive.total, n, "all docs indexed");
        assert_eq!(active.total, n.div_ceil(2), "even ids are active");
        assert_eq!(inactive.total, n / 2, "odd ids are inactive");
    }

    // ── Schema persists across restart ──────────────────────────────────────

    #[tokio::test]
    async fn test_schema_survives_restart() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();

        // First open: create store, write data
        {
            let store = open_fresh(db_dir.path(), schema_dir.path()).await;
            store
                .create(make_schema(
                    "users",
                    vec![IndexSpec {
                        field: "active".to_owned(),
                        index_type: IndexType::Bool,
                    }],
                ))
                .await
                .unwrap();
            store.put("users", DocId::U64(1), serde_json::json!({"active": true})).await.unwrap();
            store.put("users", DocId::U64(2), serde_json::json!({"active": false})).await.unwrap();
        }

        // Second open: no create() call — schema must be loaded automatically
        {
            let store = open_fresh(db_dir.path(), schema_dir.path()).await;
            let result = store.query("users", "active = true", Pagination::default()).await.unwrap();
            assert_eq!(result.total, 1);
            match result.results[0].0 {
                DocId::U64(1) => {}
                other => panic!("unexpected id {:?}", other),
            }
        }
    }

    /// Deletions must survive restart: a range scan on the reopened store
    /// must not return documents deleted in the previous session.
    #[tokio::test]
    async fn test_range_query_after_delete_survives_restart() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();

        // Session 1: create store, insert docs, delete one, then drop.
        {
            let store = open_fresh(db_dir.path(), schema_dir.path()).await;
            store.create(make_schema("docs", vec![])).await.unwrap();

            for i in 1u64..=5 {
                store.put("docs", DocId::U64(i), serde_json::json!({"n": i})).await.unwrap();
            }

            store.delete("docs", DocId::U64(3)).await.unwrap();
            store.delete("docs", DocId::U64(5)).await.unwrap();

            // Sanity check within the same session.
            let result = store.scan_range("docs", DocId::U64(1), None, None, 100).await.unwrap();
            assert_eq!(result.results.len(), 3, "same-session range should show 3 docs");
        }

        // Session 2: reopen — WAL recovery runs, deletions must hold.
        {
            let store = open_fresh(db_dir.path(), schema_dir.path()).await;

            let result = store.scan_range("docs", DocId::U64(1), None, None, 100).await.unwrap();
            let ids: Vec<u64> = result
                .results
                .iter()
                .map(|(id, _)| match id {
                    DocId::U64(v) => *v,
                    _ => panic!(),
                })
                .collect();
            assert_eq!(ids, vec![1, 2, 4], "after restart, deleted docs 3 and 5 must not appear in range scan");

            // Point-gets must also confirm deletion.
            assert_eq!(store.get("docs", DocId::U64(3)).await.unwrap(), None);
            assert_eq!(store.get("docs", DocId::U64(5)).await.unwrap(), None);
            // Surviving docs must still be readable.
            assert!(store.get("docs", DocId::U64(1)).await.unwrap().is_some());
            assert!(store.get("docs", DocId::U64(2)).await.unwrap().is_some());
            assert!(store.get("docs", DocId::U64(4)).await.unwrap().is_some());
        }
    }

    /// Prefix scan after delete must survive a restart.
    #[tokio::test]
    async fn test_prefix_scan_after_delete_survives_restart() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();

        {
            let store = open_fresh(db_dir.path(), schema_dir.path()).await;
            store.create(make_schema("docs", vec![])).await.unwrap();

            for i in 1u64..=4 {
                store.put("docs", DocId::U64(i), serde_json::json!({"v": i})).await.unwrap();
            }

            store.delete("docs", DocId::U64(2)).await.unwrap();
        }

        {
            let store = open_fresh(db_dir.path(), schema_dir.path()).await;

            let page = store.scan_prefix("docs", vec![], None, 100).await.unwrap();
            let ids: Vec<u64> = page
                .results
                .iter()
                .map(|(id, _)| match id {
                    DocId::U64(v) => *v,
                    _ => panic!(),
                })
                .collect();
            assert_eq!(ids, vec![1, 3, 4], "after restart, deleted doc 2 must not appear in prefix scan");
        }
    }

    /// Index (predicate) query after delete must survive a restart.
    ///
    /// The bitmap index is rebuilt from WAL on reopen, so a document deleted
    /// before shutdown must not appear in the predicate results after restart.
    #[tokio::test]
    async fn test_predicate_query_after_delete_survives_restart() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();

        {
            let store = open_fresh(db_dir.path(), schema_dir.path()).await;
            store
                .create(make_schema(
                    "users",
                    vec![IndexSpec {
                        field: "active".to_owned(),
                        index_type: IndexType::Bool,
                    }],
                ))
                .await
                .unwrap();

            store
                .put("users", DocId::U64(1), serde_json::json!({"active": true,  "name": "Alice"}))
                .await
                .unwrap();
            store
                .put("users", DocId::U64(2), serde_json::json!({"active": true,  "name": "Bob"}))
                .await
                .unwrap();
            store
                .put("users", DocId::U64(3), serde_json::json!({"active": false, "name": "Carol"}))
                .await
                .unwrap();
            store
                .put("users", DocId::U64(4), serde_json::json!({"active": true,  "name": "Dave"}))
                .await
                .unwrap();

            // Sanity: 3 active docs before delete.
            let before = store.query("users", "active = true", Pagination::default()).await.unwrap();
            assert_eq!(before.total, 3);

            // Delete one active doc.
            store.delete("users", DocId::U64(2)).await.unwrap();

            // Same session: only 2 active docs.
            let after = store.query("users", "active = true", Pagination::default()).await.unwrap();
            assert_eq!(after.total, 2);
        }

        // Reopen — index is rebuilt from the WAL tail.
        {
            let store = open_fresh(db_dir.path(), schema_dir.path()).await;

            let result = store.query("users", "active = true", Pagination::default()).await.unwrap();
            let mut ids: Vec<u64> = result
                .results
                .iter()
                .map(|(id, _)| match id {
                    DocId::U64(v) => *v,
                    _ => panic!(),
                })
                .collect();
            ids.sort();
            assert_eq!(ids, vec![1, 4], "after restart, deleted doc 2 must not appear in predicate query");

            // The inactive doc must still be found.
            let inactive = store.query("users", "active = false", Pagination::default()).await.unwrap();
            assert_eq!(inactive.total, 1);
            match inactive.results[0].0 {
                DocId::U64(3) => {}
                other => panic!("expected doc 3, got {:?}", other),
            }

            // Point-get must confirm doc 2 is gone.
            assert_eq!(store.get("users", DocId::U64(2)).await.unwrap(), None);
        }
    }

    /// Delete every on-disk per-field index `checkpoint` file under `db_dir`,
    /// rewinding each field's checkpoint offset to 0. This simulates a hard
    /// crash that occurred after writes but before any index checkpoint (the
    /// `Drop`/`shutdown` checkpoint never ran), forcing `activate_field_index`
    /// to replay the WAL tail on the next open.
    fn rewind_index_checkpoints(db_dir: &Path) {
        let index_dir = db_dir.join("index");
        let Ok(namespaces) = std::fs::read_dir(&index_dir) else { return };
        for ns in namespaces.flatten() {
            let Ok(fields) = std::fs::read_dir(ns.path()) else { continue };
            for field in fields.flatten() {
                let _ = std::fs::remove_file(field.path().join("checkpoint"));
            }
        }
    }

    /// Regression: the custom (key-type-derived) row-ID function must be
    /// installed *before* field indexes are activated, so the activation-time
    /// WAL-tail replay resolves row IDs the same way prior writes did. If the
    /// index is activated first, replay falls back to the dense `RowMap` and
    /// indexes the replayed keys under IDs (0, 1, 2, …) that disagree with the
    /// key-derived IDs — corrupting query key-resolution.
    ///
    /// The replay only fires when the index checkpoint is behind the WAL tail,
    /// which normally happens only on a hard crash; we reproduce that by
    /// rewinding the on-disk checkpoint between sessions.
    #[tokio::test]
    async fn row_id_fn_installed_before_index_activation_replay() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();

        {
            let store = open_fresh(db_dir.path(), schema_dir.path()).await;
            store
                .create(make_schema(
                    "users",
                    vec![IndexSpec {
                        field: "active".to_owned(),
                        index_type: IndexType::Bool,
                    }],
                ))
                .await
                .unwrap();

            // Non-sequential ids so the dense RowMap ids (0, 1, 2) the buggy
            // path would assign are clearly different from the key-derived ids.
            store
                .put("users", DocId::U64(100), serde_json::json!({"active": true,  "name": "A"}))
                .await
                .unwrap();
            store
                .put("users", DocId::U64(200), serde_json::json!({"active": false, "name": "B"}))
                .await
                .unwrap();
            store
                .put("users", DocId::U64(300), serde_json::json!({"active": true,  "name": "C"}))
                .await
                .unwrap();
        }

        // Simulate a crash after the writes but before an index checkpoint.
        rewind_index_checkpoints(db_dir.path());

        {
            let store = open_fresh(db_dir.path(), schema_dir.path()).await;
            let result = store.query("users", "active = true", Pagination::default()).await.unwrap();
            let mut ids: Vec<u64> = result
                .results
                .iter()
                .map(|(id, _)| match id {
                    DocId::U64(v) => *v,
                    other => panic!("expected U64 id, got {other:?}"),
                })
                .collect();
            ids.sort();
            assert_eq!(
                ids,
                vec![100, 300],
                "activation replay must index under key-derived row IDs, not dense RowMap IDs"
            );
            // The inactive doc must still resolve correctly too.
            let inactive = store.query("users", "active = false", Pagination::default()).await.unwrap();
            assert_eq!(inactive.total, 1);
            assert!(matches!(inactive.results[0].0, DocId::U64(200)));
        }
    }

    // ── DocId serialization ─────────────────────────────────────────────────

    #[test]
    fn test_doc_id_u64_roundtrip() {
        let id = DocId::U64(12345);
        let bytes = id.to_bytes();
        let restored = DocId::from_bytes(&bytes, KeyType::U64).unwrap();
        assert_eq!(id, restored);
    }

    #[test]
    fn test_doc_id_u128_roundtrip() {
        let id = DocId::U128(u128::MAX / 2);
        let bytes = id.to_bytes();
        let restored = DocId::from_bytes(&bytes, KeyType::U128).unwrap();
        assert_eq!(id, restored);
    }

    #[test]
    fn test_doc_id_uuid_roundtrip() {
        let id = DocId::Uuid(0xdeadbeef_cafebabe_12345678_9abcdef0);
        let bytes = id.to_bytes();
        let restored = DocId::from_bytes(&bytes, KeyType::Uuid).unwrap();
        assert_eq!(id, restored);
    }

    #[test]
    fn test_doc_id_ordering() {
        // big-endian encoding means byte-level ordering == numeric ordering
        let ids: Vec<DocId> = (0u64..5).map(DocId::U64).collect();
        let encoded: Vec<Vec<u8>> = ids.iter().map(|id| id.to_bytes()).collect();
        let sorted = {
            let mut c = encoded.clone();
            c.sort();
            c
        };
        assert_eq!(encoded, sorted);
    }

    #[test]
    fn test_invalid_key_size_rejected() {
        assert!(DocId::from_bytes(&[0u8; 3], KeyType::U64).is_err());
        assert!(DocId::from_bytes(&[0u8; 5], KeyType::U128).is_err());
    }

    // ── KV store helpers ────────────────────────────────────────────────────

    fn make_kv_schema(namespace: &str, key_type: KvKeyType, value_type: KvValueType) -> KvStoreSchema {
        use crate::schema::KvStoreSchema;
        KvStoreSchema {
            store_type: StoreType::Kv,
            namespace: namespace.to_owned(),
            ns_id: None,
            key_type,
            value_type,
            semantic_search_enabled: false,
        }
    }

    // ── KV lifecycle ────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_kv_create_and_list() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;

        store.create_kv(make_kv_schema("cache", KvKeyType::Str, KvValueType::Str)).await.unwrap();

        let list = store.list_kv().unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0]["namespace"], "cache");
        assert!(list[0]["ns_id"].as_u64().is_some());
        assert_eq!(list[0]["key_type"], "str");
        assert_eq!(list[0]["value_type"], "str");
    }

    #[tokio::test]
    async fn test_kv_get_schema_round_trip() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;

        store.create_kv(make_kv_schema("cache", KvKeyType::Int, KvValueType::F32)).await.unwrap();

        let schema = store.get_kv_schema("cache").unwrap();
        assert_eq!(schema.namespace, "cache");
        assert_eq!(schema.key_type, KvKeyType::Int);
        assert_eq!(schema.value_type, KvValueType::F32);
        // ns_id is assigned at creation and must survive the round-trip so an
        // exported schema reflects the persisted store.
        assert!(schema.ns_id.is_some());

        let err = store.get_kv_schema("missing").unwrap_err();
        assert!(matches!(err, DocStoreError::NotFound { .. }));
    }

    #[tokio::test]
    async fn test_kv_get_schema_rejects_doc_store() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;

        store.create(make_schema("docs", vec![])).await.unwrap();

        // A doc-store namespace must not be readable as a KV schema (key_type
        // discriminant differs), guarding the export endpoint against mixing types.
        assert!(store.get_kv_schema("docs").is_err());
    }

    #[tokio::test]
    async fn test_kv_create_duplicate_rejected() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;

        store.create_kv(make_kv_schema("kv", KvKeyType::Str, KvValueType::Int)).await.unwrap();
        let err = store.create_kv(make_kv_schema("kv", KvKeyType::Str, KvValueType::Int)).await.unwrap_err();
        assert!(matches!(err, DocStoreError::AlreadyExists { .. }));
    }

    #[tokio::test]
    async fn test_kv_drop_removes_schema_file() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;

        store.create_kv(make_kv_schema("tmp", KvKeyType::Str, KvValueType::Str)).await.unwrap();
        assert!(schema_dir.path().join("tmp.json").exists());

        store.remove_kv("tmp").await.unwrap();
        assert!(!schema_dir.path().join("tmp.json").exists());
    }

    #[tokio::test]
    async fn test_kv_list_does_not_include_doc_stores() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;

        store.create(make_schema("docs", vec![])).await.unwrap();
        store.create_kv(make_kv_schema("cache", KvKeyType::Str, KvValueType::Str)).await.unwrap();

        let kv_list = store.list_kv().unwrap();
        assert_eq!(kv_list.len(), 1, "list_kv should only return KV stores");
        assert_eq!(kv_list[0]["namespace"], "cache");

        let doc_list = store.list().unwrap();
        assert_eq!(doc_list.len(), 1, "list should only return doc stores");
        assert_eq!(doc_list[0]["namespace"], "docs");
    }

    // ── KV CRUD ─────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_kv_put_get_str_key_str_value() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store.create_kv(make_kv_schema("ns", KvKeyType::Str, KvValueType::Str)).await.unwrap();

        let key = serde_json::json!("hello");
        let val = serde_json::json!("world");
        store.kv_put("ns", &key, &val).await.unwrap();

        let got = store.kv_get("ns", &key).await.unwrap();
        assert_eq!(got, Some(val));
    }

    #[tokio::test]
    async fn test_kv_put_no_wal_get_round_trip() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store.create_kv(make_kv_schema("ns", KvKeyType::Str, KvValueType::Str)).await.unwrap();

        let key = serde_json::json!("hello");
        let val = serde_json::json!("world");
        // The no-WAL path must be readable in-process exactly like the WAL path;
        // only crash-durability differs.
        store.kv_put_no_wal("ns", &key, &val).await.unwrap();

        let got = store.kv_get("ns", &key).await.unwrap();
        assert_eq!(got, Some(val));
    }

    #[tokio::test]
    async fn test_kv_put_get_int_key_int_value() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store.create_kv(make_kv_schema("ns", KvKeyType::Int, KvValueType::Int)).await.unwrap();

        let key = serde_json::json!(99i64);
        let val = serde_json::json!(-42i64);
        store.kv_put("ns", &key, &val).await.unwrap();

        let got = store.kv_get("ns", &key).await.unwrap();
        assert_eq!(got, Some(val));
    }

    #[tokio::test]
    async fn test_kv_put_get_f32_value() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store.create_kv(make_kv_schema("ns", KvKeyType::Str, KvValueType::F32)).await.unwrap();

        let key = serde_json::json!("temp");
        let val = serde_json::json!(98.5f32);
        store.kv_put("ns", &key, &val).await.unwrap();

        let got = store.kv_get("ns", &key).await.unwrap().unwrap();
        let diff = (got.as_f64().unwrap() as f32 - 98.5f32).abs();
        assert!(diff < f32::EPSILON, "f32 value mismatch: {got}");
    }

    #[tokio::test]
    async fn test_kv_put_get_vec_f32_value() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store.create_kv(make_kv_schema("ns", KvKeyType::Str, KvValueType::VecF32)).await.unwrap();

        let key = serde_json::json!("embedding");
        let val = serde_json::json!([1.0f32, 0.5f32, -1.0f32]);
        store.kv_put("ns", &key, &val).await.unwrap();

        let got = store.kv_get("ns", &key).await.unwrap().unwrap();
        let arr = got.as_array().unwrap();
        assert_eq!(arr.len(), 3);
        let expected = [1.0f32, 0.5, -1.0];
        for (v, exp) in arr.iter().zip(expected.iter()) {
            assert!((v.as_f64().unwrap() as f32 - exp).abs() < f32::EPSILON);
        }
    }

    #[tokio::test]
    async fn test_kv_put_overwrites_existing() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store.create_kv(make_kv_schema("ns", KvKeyType::Str, KvValueType::Str)).await.unwrap();

        let key = serde_json::json!("k");
        store.kv_put("ns", &key, &serde_json::json!("v1")).await.unwrap();
        store.kv_put("ns", &key, &serde_json::json!("v2")).await.unwrap();

        assert_eq!(store.kv_get("ns", &key).await.unwrap(), Some(serde_json::json!("v2")));
    }

    #[tokio::test]
    async fn test_kv_get_missing_key_returns_none() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store.create_kv(make_kv_schema("ns", KvKeyType::Str, KvValueType::Str)).await.unwrap();

        let got = store.kv_get("ns", &serde_json::json!("ghost")).await.unwrap();
        assert_eq!(got, None);
    }

    #[tokio::test]
    async fn test_kv_delete_removes_entry() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store.create_kv(make_kv_schema("ns", KvKeyType::Str, KvValueType::Str)).await.unwrap();

        let key = serde_json::json!("gone");
        store.kv_put("ns", &key, &serde_json::json!("value")).await.unwrap();
        store.kv_delete("ns", "gone").await.unwrap();

        assert_eq!(store.kv_get("ns", &key).await.unwrap(), None);
    }

    #[tokio::test]
    async fn test_kv_get_by_str_str_key() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store.create_kv(make_kv_schema("ns", KvKeyType::Str, KvValueType::Str)).await.unwrap();

        store
            .kv_put("ns", &serde_json::json!("mykey"), &serde_json::json!("myval"))
            .await
            .unwrap();
        let got = store.kv_get_by_str("ns", "mykey").await.unwrap();
        assert_eq!(got, Some(serde_json::json!("myval")));
    }

    #[tokio::test]
    async fn test_kv_get_by_str_int_key() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store.create_kv(make_kv_schema("ns", KvKeyType::Int, KvValueType::Str)).await.unwrap();

        store.kv_put("ns", &serde_json::json!(7i64), &serde_json::json!("seven")).await.unwrap();
        let got = store.kv_get_by_str("ns", "7").await.unwrap();
        assert_eq!(got, Some(serde_json::json!("seven")));
    }

    #[tokio::test]
    async fn test_kv_wrong_value_type_rejected() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store.create_kv(make_kv_schema("ns", KvKeyType::Str, KvValueType::Int)).await.unwrap();

        // Put a string into an Int-typed namespace.
        let err = store
            .kv_put("ns", &serde_json::json!("k"), &serde_json::json!("not-int"))
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            DocStoreError::Schema(crate::error::SchemaError::KvValueTypeMismatch { .. })
        ));
    }

    // ── Restart persistence ─────────────────────────────────────────────────

    #[tokio::test]
    async fn test_kv_schema_survives_restart() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();

        // First run: create KV store and write data.
        {
            let store = open_fresh(db_dir.path(), schema_dir.path()).await;
            store
                .create_kv(make_kv_schema("session", KvKeyType::Str, KvValueType::Str))
                .await
                .unwrap();
            store
                .kv_put("session", &serde_json::json!("tok1"), &serde_json::json!("abc"))
                .await
                .unwrap();
            store
                .kv_put("session", &serde_json::json!("tok2"), &serde_json::json!("def"))
                .await
                .unwrap();
        } // store dropped here, lock released

        // Second run: no create_kv call — schema must load automatically.
        {
            let store = open_fresh(db_dir.path(), schema_dir.path()).await;
            assert_eq!(store.list_kv().unwrap().len(), 1, "KV schema not reloaded after restart");
            assert_eq!(
                store.kv_get("session", &serde_json::json!("tok1")).await.unwrap(),
                Some(serde_json::json!("abc"))
            );
            assert_eq!(
                store.kv_get("session", &serde_json::json!("tok2")).await.unwrap(),
                Some(serde_json::json!("def"))
            );
        }
    }

    #[tokio::test]
    async fn test_kv_and_doc_stores_coexist_after_restart() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();

        {
            let store = open_fresh(db_dir.path(), schema_dir.path()).await;
            store.create(make_schema("docs", vec![])).await.unwrap();
            store.put("docs", DocId::U64(1), serde_json::json!({"x": 1})).await.unwrap();
            store.create_kv(make_kv_schema("cache", KvKeyType::Str, KvValueType::Str)).await.unwrap();
            store.kv_put("cache", &serde_json::json!("a"), &serde_json::json!("b")).await.unwrap();
        }

        {
            let store = open_fresh(db_dir.path(), schema_dir.path()).await;
            // Doc store data intact
            assert_eq!(store.list().unwrap().len(), 1);
            assert_eq!(store.get("docs", DocId::U64(1)).await.unwrap(), Some(serde_json::json!({"x": 1})));
            // KV store data intact
            assert_eq!(store.list_kv().unwrap().len(), 1);
            assert_eq!(
                store.kv_get("cache", &serde_json::json!("a")).await.unwrap(),
                Some(serde_json::json!("b"))
            );
        }
    }

    // ── Vector-index queue methods ──────────────────────────────────────────

    #[tokio::test]
    async fn test_vector_index_max_retries_default() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        assert_eq!(store.vector_index_max_retries(), 5);
    }

    #[tokio::test]
    async fn test_with_vector_index_config_sets_max_retries() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path())
            .await
            .with_vector_index_config(VectorIndexConfig {
                max_retries: 10,
                retry_wait_secs: 1,
                concurrency: 2,
            });
        assert_eq!(store.vector_index_max_retries(), 10);
    }

    #[tokio::test]
    async fn test_pending_vector_index_count_empty() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        assert_eq!(store.pending_vector_index_count().await, 0);
    }

    #[tokio::test]
    async fn test_list_pending_queue_entries_empty() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        assert!(store.list_queue_entries().await.is_empty());
    }

    #[tokio::test]
    async fn test_pending_queue_count_and_list_after_enqueue() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;

        // Enqueue two entries directly via vector_kv primitives.
        vector_kv::enqueue_embed(&store.db, "ns_a", b"doc1", "text a").await.unwrap();
        vector_kv::enqueue_embed(&store.db, "ns_b", b"doc2", "text b").await.unwrap();

        assert_eq!(store.pending_vector_index_count().await, 2);

        let entries = store.list_queue_entries().await;
        assert_eq!(entries.len(), 2);

        let namespaces: std::collections::BTreeSet<_> = entries.iter().map(|e| e.namespace.as_str()).collect();
        assert!(namespaces.contains("ns_a"));
        assert!(namespaces.contains("ns_b"));
    }

    #[tokio::test]
    async fn test_delete_pending_queue_entry_removes_entry() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;

        vector_kv::enqueue_embed(&store.db, "ns", b"doc1", "text").await.unwrap();

        assert_eq!(store.pending_vector_index_count().await, 1);

        store.delete_queue_entry("ns", b"doc1").await.unwrap();

        assert_eq!(store.pending_vector_index_count().await, 0);
    }

    #[tokio::test]
    async fn test_delete_pending_queue_entry_noop_for_missing() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;

        // Deleting a non-existent entry should not error.
        store.delete_queue_entry("ns", b"ghost").await.unwrap();
        assert_eq!(store.pending_vector_index_count().await, 0);
    }

    #[tokio::test]
    async fn test_delete_pending_queue_entry_only_removes_target() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;

        vector_kv::enqueue_embed(&store.db, "ns", b"doc1", "t1").await.unwrap();
        vector_kv::enqueue_embed(&store.db, "ns", b"doc2", "t2").await.unwrap();

        store.delete_queue_entry("ns", b"doc1").await.unwrap();

        let entries = store.list_queue_entries().await;
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].doc_id_bytes, b"doc2");
    }

    // ── count_docs ─────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_count_docs_empty_namespace() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store.create(make_schema("empty", vec![])).await.unwrap();

        assert_eq!(store.count_docs("empty").await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_count_docs_reflects_inserts() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store.create(make_schema("counting", vec![])).await.unwrap();

        for i in 1u64..=5 {
            store.put("counting", DocId::U64(i), serde_json::json!({"n": i})).await.unwrap();
        }

        assert_eq!(store.count_docs("counting").await.unwrap(), 5);
    }

    #[tokio::test]
    async fn test_count_docs_decrements_after_delete() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store.create(make_schema("del_count", vec![])).await.unwrap();

        for i in 1u64..=3 {
            store.put("del_count", DocId::U64(i), serde_json::json!({"n": i})).await.unwrap();
        }
        assert_eq!(store.count_docs("del_count").await.unwrap(), 3);

        store.delete("del_count", DocId::U64(2)).await.unwrap();
        assert_eq!(store.count_docs("del_count").await.unwrap(), 2);
    }

    #[tokio::test]
    async fn test_count_docs_unknown_namespace_is_not_found() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;

        let err = store.count_docs("ghost").await.unwrap_err();
        assert!(matches!(err, DocStoreError::NotFound { .. }));
    }

    // ── get_schema (export) ────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_get_schema_round_trips_through_json() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;

        let original = make_schema("export_test", vec![]);
        store.create(original.clone()).await.unwrap();

        let retrieved = store.get_schema("export_test").unwrap();
        let json = serde_json::to_string(&retrieved).unwrap();
        let re_parsed: DocStoreSchema = serde_json::from_str(&json).unwrap();

        assert_eq!(re_parsed.namespace, "export_test");
        assert_eq!(re_parsed.key_type, original.key_type);
        assert_eq!(re_parsed.semantic_search_enabled, original.semantic_search_enabled);
    }

    #[tokio::test]
    async fn test_get_schema_unknown_namespace_is_not_found() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;

        let err = store.get_schema("no_such_ns").unwrap_err();
        assert!(matches!(err, DocStoreError::NotFound { .. }));
    }

    // ── import (create via schema JSON) ────────────────────────────────────────

    #[tokio::test]
    async fn test_import_schema_creates_usable_store() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;

        let schema: DocStoreSchema = serde_json::from_str(
            r#"{
            "namespace": "imported",
            "store_type": "doc",
            "key_type": "u64",
            "attributes": [],
            "indices": [],
            "semantic_search_enabled": false,
            "embedding_fields": []
        }"#,
        )
        .unwrap();

        store.create(schema).await.unwrap();

        let list = store.list().unwrap();
        assert!(list.iter().any(|v| v["namespace"] == "imported"));

        store.put("imported", DocId::U64(1), serde_json::json!({"x": 1})).await.unwrap();
        assert_eq!(store.count_docs("imported").await.unwrap(), 1);
    }

    /// Any `ns_id` present in an exported schema must be discarded so the store
    /// assigns a fresh ID rather than colliding with an existing namespace.
    #[tokio::test]
    async fn test_import_schema_stale_ns_id_is_replaced() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;

        // Schema carries a stale ns_id from a previous database instance.
        let mut schema = make_schema("ns_id_test", vec![]);
        schema.ns_id = Some(99999);

        store.create(schema).await.unwrap();

        let stored = store.get_schema("ns_id_test").unwrap();
        // The stored ns_id must be the one assigned by this store, not the
        // stale value that came in with the import body.
        assert_ne!(stored.ns_id, Some(99999));
        assert!(stored.ns_id.is_some(), "ns_id must be assigned after create");
    }

    #[tokio::test]
    async fn test_import_schema_duplicate_is_rejected() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;

        store.create(make_schema("dup_import", vec![])).await.unwrap();

        let err = store.create(make_schema("dup_import", vec![])).await.unwrap_err();
        assert!(matches!(err, DocStoreError::AlreadyExists { .. }));
    }

    #[tokio::test]
    async fn test_import_schema_invalid_namespace_rejected() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;

        for bad_ns in &["", "has space", "has/slash", "has.dot"] {
            let mut schema = make_schema("placeholder", vec![]);
            schema.namespace = bad_ns.to_string();
            let err = store.create(schema).await.unwrap_err();
            assert!(
                matches!(err, DocStoreError::Schema(crate::error::SchemaError::InvalidNamespace)),
                "expected InvalidNamespace for '{bad_ns}', got {err:?}"
            );
        }
    }

    #[tokio::test]
    async fn test_import_schema_too_many_indices_rejected() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;

        let indices = (0..=crate::schema::MAX_INDICES)
            .map(|i| IndexSpec {
                field: format!("f{i}"),
                index_type: IndexType::Int,
            })
            .collect();
        let err = store.create(make_schema("too_many", indices)).await.unwrap_err();
        assert!(matches!(err, DocStoreError::Schema(crate::error::SchemaError::TooManyIndices { .. })));
    }

    #[tokio::test]
    async fn test_import_schema_duplicate_index_field_rejected() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;

        let indices = vec![
            IndexSpec {
                field: "status".to_owned(),
                index_type: IndexType::Str,
            },
            IndexSpec {
                field: "status".to_owned(),
                index_type: IndexType::Str,
            },
        ];
        let err = store.create(make_schema("dup_field", indices)).await.unwrap_err();
        assert!(matches!(err, DocStoreError::Schema(crate::error::SchemaError::DuplicateFieldName { .. })));
    }

    /// Regression: a field that was originally declared as an attribute and later
    /// indexed ends up in both `attributes` and `indices` in the saved schema.
    /// `add_index` must move it out of `attributes`; importing such a schema must
    /// also survive without a DuplicateFieldName error.
    #[tokio::test]
    async fn test_add_index_removes_field_from_attributes() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;

        // Create with "agency" as a non-indexed attribute.
        let mut schema = make_schema("agency_ns", vec![]);
        schema.attributes.push(crate::schema::AttributeDef {
            name: "agency".to_owned(),
            attr_type: crate::schema::AttributeType::Str,
            description: None,
        });
        store.create(schema).await.unwrap();

        // Add an index on "agency" — this must remove it from attributes.
        store
            .add_index(
                "agency_ns",
                IndexSpec {
                    field: "agency".to_owned(),
                    index_type: IndexType::Str,
                },
            )
            .await
            .unwrap()
            .wait()
            .await
            .unwrap();

        let stored = store.get_schema("agency_ns").unwrap();
        assert!(
            stored.attributes.iter().all(|a| a.name != "agency"),
            "agency must not remain in attributes after indexing"
        );
        assert!(stored.indices.iter().any(|i| i.field == "agency"), "agency must be in indices");
    }

    /// Regression: importing an existing schema (with a new name) where a field
    /// appears in both `attributes` and `indices` must not fail with DuplicateFieldName.
    /// The import handler normalises the schema by dropping indexed fields from
    /// attributes before calling create.
    #[tokio::test]
    async fn test_create_with_field_in_both_attributes_and_indices_is_rejected() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;

        // Simulate a stale exported schema that has "agency" in both lists.
        let schema = DocStoreSchema {
            store_type: StoreType::Doc,
            namespace: "stale_export".to_owned(),
            ns_id: None,
            key_type: crate::schema::KeyType::U64,
            attributes: vec![crate::schema::AttributeDef {
                name: "agency".to_owned(),
                attr_type: crate::schema::AttributeType::Str,
                description: None,
            }],
            indices: vec![IndexSpec {
                field: "agency".to_owned(),
                index_type: IndexType::Str,
            }],
            semantic_search_enabled: false,
            embedding_fields: vec![],
        };

        // Direct create (without normalisation) must be rejected by validate().
        let err = store.create(schema).await.unwrap_err();
        assert!(matches!(err, DocStoreError::Schema(crate::error::SchemaError::DuplicateFieldName { .. })));
    }

    // ── Vector-index reconciliation ─────────────────────────────────────────

    /// Helper: create a semantic-search-enabled doc schema with a `title` field.
    async fn create_semantic_schema(store: &DocStore, namespace: &str) {
        let schema = DocStoreSchema {
            store_type: StoreType::Doc,
            namespace: namespace.to_owned(),
            ns_id: None,
            key_type: KeyType::U64,
            attributes: vec![crate::schema::AttributeDef {
                name: "title".to_owned(),
                attr_type: AttributeType::Str,
                description: None,
            }],
            indices: vec![],
            semantic_search_enabled: true,
            embedding_fields: vec!["title".to_owned()],
        };
        store.create(schema).await.unwrap();
    }

    /// Helper: commit a *complete* vector index (sparse meta + dense) for a doc,
    /// simulating a fully-indexed document. Reconciliation skips only documents
    /// that have both halves, so tests asserting "already indexed → skip" must
    /// write both.
    async fn commit_complete_index(store: &DocStore, namespace: &str, key: &[u8]) {
        use semantic_search::index::vector_index::QuantisationStyle;
        let sparse = semantic_search::VectorIndex::new(1, QuantisationStyle::SingleBit, 0.4, 0.0, 0.02, vec![]);
        let dense = semantic_search::VectorIndex::new(1, QuantisationStyle::MultiBit { number_of_bits: 8 }, 0.4, 0.0, 0.02, vec![]);
        crate::vector_kv::upsert_vectors(&store.db, namespace, key, &[sparse, dense])
            .await
            .unwrap();
    }

    /// Documents written through the crash window (doc present, but no queue
    /// entry and no vector index) must be re-enqueued by reconciliation.
    #[tokio::test]
    async fn test_reconcile_enqueues_missing_docs() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        create_semantic_schema(&store, "sem").await;

        // No SemanticSearchContext is attached, so put() does NOT enqueue —
        // this is exactly the crash window (doc durable, embed marker lost).
        for i in 1u64..=3 {
            store
                .put("sem", DocId::U64(i), serde_json::json!({"title": format!("doc {i}")}))
                .await
                .unwrap();
        }
        assert_eq!(store.pending_vector_index_count().await, 0, "no enqueue happened on write");

        let reconciled = store.reconcile_vector_indexes().await;
        assert_eq!(reconciled, 3, "all three missing docs must be re-enqueued");
        assert_eq!(store.pending_vector_index_count().await, 3);
    }

    /// A single-document vector reindex enqueues exactly that document, reports
    /// `NotFound` for a missing id, and rejects a non-semantic namespace.
    #[tokio::test]
    async fn test_reindex_doc_vector_enqueues_single_doc() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        create_semantic_schema(&store, "sem").await;

        // No SemanticSearchContext attached, so put() does not enqueue.
        for i in 1u64..=3 {
            store
                .put("sem", DocId::U64(i), serde_json::json!({"title": format!("doc {i}")}))
                .await
                .unwrap();
        }
        assert_eq!(store.pending_vector_index_count().await, 0);

        // Reindex just doc 2 → exactly one enqueue.
        assert_eq!(
            store.reindex_doc_vector("sem", DocId::U64(2)).await.unwrap(),
            VectorReindexOutcome::Enqueued
        );
        assert_eq!(store.pending_vector_index_count().await, 1);

        // A missing document reports NotFound and enqueues nothing more.
        assert_eq!(
            store.reindex_doc_vector("sem", DocId::U64(99)).await.unwrap(),
            VectorReindexOutcome::NotFound
        );
        assert_eq!(store.pending_vector_index_count().await, 1);

        // A namespace without semantic search is rejected.
        store
            .create(DocStoreSchema {
                store_type: StoreType::Doc,
                namespace: "plain".to_owned(),
                ns_id: None,
                key_type: KeyType::U64,
                attributes: vec![],
                indices: vec![],
                semantic_search_enabled: false,
                embedding_fields: vec![],
            })
            .await
            .unwrap();
        store.put("plain", DocId::U64(1), serde_json::json!({"title": "x"})).await.unwrap();
        assert!(matches!(
            store.reindex_doc_vector("plain", DocId::U64(1)).await,
            Err(DocStoreError::SemanticSearchNotEnabled { .. })
        ));
    }

    /// Reconciliation must skip documents that already have a pending queue
    /// entry, and must be idempotent on a second run.
    #[tokio::test]
    async fn test_reconcile_skips_queued_and_is_idempotent() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        create_semantic_schema(&store, "sem").await;

        store.put("sem", DocId::U64(1), serde_json::json!({"title": "hello"})).await.unwrap();
        store.put("sem", DocId::U64(2), serde_json::json!({"title": "world"})).await.unwrap();

        // First pass enqueues both missing docs.
        assert_eq!(store.reconcile_vector_indexes().await, 2);
        assert_eq!(store.pending_vector_index_count().await, 2);

        // Second pass is a no-op: both docs are already queued.
        assert_eq!(store.reconcile_vector_indexes().await, 0);
        assert_eq!(store.pending_vector_index_count().await, 2);
    }

    /// Reconciliation must skip documents that already have a committed vector
    /// index entry.
    #[tokio::test]
    async fn test_reconcile_skips_already_indexed() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        create_semantic_schema(&store, "sem").await;

        let key = DocId::U64(1).to_bytes();
        store.put("sem", DocId::U64(1), serde_json::json!({"title": "indexed"})).await.unwrap();

        // Simulate a complete committed vector index (sparse meta + dense) for doc 1.
        commit_complete_index(&store, "sem", &key).await;

        // Doc 1 is already indexed → reconciliation must not enqueue it.
        assert_eq!(store.reconcile_vector_indexes().await, 0);
        assert_eq!(store.pending_vector_index_count().await, 0);
    }

    /// The validating reconcile must NOT re-enqueue a document whose committed
    /// vectors are present *and* deserialize (no false positives on healthy docs).
    #[tokio::test]
    async fn test_validating_reconcile_skips_valid_complete_index() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        create_semantic_schema(&store, "sem").await;

        let key = DocId::U64(1).to_bytes();
        store.put("sem", DocId::U64(1), serde_json::json!({"title": "indexed"})).await.unwrap();
        commit_complete_index(&store, "sem", &key).await;

        assert_eq!(
            store.validate_and_reconcile_vector_indexes().await,
            0,
            "valid index must not be re-enqueued"
        );
        assert_eq!(store.pending_vector_index_count().await, 0);
    }

    /// A document whose committed vector bytes are *present but corrupt* is skipped
    /// by the presence-only reconcile (both halves exist) yet re-enqueued by the
    /// validating reconcile (the bytes fail to deserialize). Covers both the dense
    /// and a sparse composite entry.
    #[tokio::test]
    async fn test_validating_reconcile_reenqueues_present_but_corrupt() {
        // Corrupt dense.
        {
            let db_dir = TempDir::new().unwrap();
            let schema_dir = TempDir::new().unwrap();
            let store = open_fresh(db_dir.path(), schema_dir.path()).await;
            create_semantic_schema(&store, "sem").await;
            let key = DocId::U64(1).to_bytes();
            store.put("sem", DocId::U64(1), serde_json::json!({"title": "indexed"})).await.unwrap();
            commit_complete_index(&store, "sem", &key).await;

            // Overwrite the dense entry with bytes that are present but undeserializable.
            let dense_ns = store.db.namespace(crate::vector_kv::dense_vectors_ns("sem")).await.unwrap();
            dense_ns.put(key.clone(), b"not valid rkyv bytes".to_vec()).await.unwrap();

            assert_eq!(store.reconcile_vector_indexes().await, 0, "presence check sees both halves → skips");
            assert_eq!(store.pending_vector_index_count().await, 0);
            assert_eq!(
                store.validate_and_reconcile_vector_indexes().await,
                1,
                "corrupt dense must be re-enqueued"
            );
            assert_eq!(store.pending_vector_index_count().await, 1);
        }

        // Corrupt a sparse composite entry (commit_complete_index assigns cluster 1).
        {
            let db_dir = TempDir::new().unwrap();
            let schema_dir = TempDir::new().unwrap();
            let store = open_fresh(db_dir.path(), schema_dir.path()).await;
            create_semantic_schema(&store, "sem").await;
            let key = DocId::U64(1).to_bytes();
            store.put("sem", DocId::U64(1), serde_json::json!({"title": "indexed"})).await.unwrap();
            commit_complete_index(&store, "sem", &key).await;

            let sparse_ns = store.db.namespace(crate::vector_kv::sparse_vectors_ns("sem")).await.unwrap();
            sparse_ns
                .put(semantic_search::composite_key::encode(1, &key), b"garbage".to_vec())
                .await
                .unwrap();

            assert_eq!(store.reconcile_vector_indexes().await, 0, "presence check skips");
            assert_eq!(
                store.validate_and_reconcile_vector_indexes().await,
                1,
                "corrupt sparse must be re-enqueued"
            );
            assert_eq!(store.pending_vector_index_count().await, 1);
        }
    }

    /// KV-store namespaces are reconciled too: a string value written without a
    /// queue entry or vector index must be re-enqueued.
    #[tokio::test]
    async fn test_reconcile_kv_namespace() {
        use crate::schema::{KvKeyType, KvStoreSchema, KvValueType};

        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;

        let schema = KvStoreSchema {
            store_type: StoreType::Kv,
            namespace: "kv".to_owned(),
            ns_id: None,
            key_type: KvKeyType::Str,
            value_type: KvValueType::Str,
            semantic_search_enabled: true,
        };
        store.create_kv(schema).await.unwrap();

        store
            .kv_put("kv", &serde_json::json!("k1"), &serde_json::json!("some text"))
            .await
            .unwrap();
        assert_eq!(store.pending_vector_index_count().await, 0, "no enqueue on write (no ctx)");

        assert_eq!(store.reconcile_vector_indexes().await, 1);
        assert_eq!(store.pending_vector_index_count().await, 1);
    }

    /// The count short-circuit must NOT fire when a namespace is only partially
    /// indexed: one indexed doc + one missing doc means `indexed (1) < keys (2)`,
    /// so the full scan runs and the missing doc is enqueued.
    #[tokio::test]
    async fn test_reconcile_partial_index_not_short_circuited() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        create_semantic_schema(&store, "sem").await;

        // Doc 1 is indexed; doc 2 is written but missing (the crash window).
        store.put("sem", DocId::U64(1), serde_json::json!({"title": "one"})).await.unwrap();
        store.put("sem", DocId::U64(2), serde_json::json!({"title": "two"})).await.unwrap();
        commit_complete_index(&store, "sem", &DocId::U64(1).to_bytes()).await;

        // indexed (1) < keys (2) → no short-circuit → doc 2 enqueued, doc 1 skipped.
        assert_eq!(store.reconcile_vector_indexes().await, 1);
        let pending = store.list_queue_entries().await;
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].doc_id_bytes, DocId::U64(2).to_bytes());
    }

    /// A *partially* committed index counts as not-indexed: a doc with only the
    /// sparse side (dense write lost) and a doc with only the dense side (sparse
    /// write lost) must both be re-enqueued so the re-embed regenerates the
    /// missing half. Guards the `meta AND dense` tightening in
    /// [`vector_kv::has_complete_vector_index`] — under the old OR semantics
    /// either of these would have been skipped as "indexed".
    #[tokio::test]
    async fn test_reconcile_partial_index_is_reenqueued() {
        use semantic_search::index::vector_index::QuantisationStyle;

        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        create_semantic_schema(&store, "sem").await;

        store
            .put("sem", DocId::U64(1), serde_json::json!({"title": "sparse only"}))
            .await
            .unwrap();
        store.put("sem", DocId::U64(2), serde_json::json!({"title": "dense only"})).await.unwrap();

        // Doc 1: only the sparse meta committed — the dense write was lost.
        let key1 = DocId::U64(1).to_bytes();
        let sparse = semantic_search::VectorIndex::new(1, QuantisationStyle::SingleBit, 0.4, 0.0, 0.02, vec![]);
        crate::vector_kv::upsert_vectors(&store.db, "sem", &key1, &[sparse]).await.unwrap();

        // Doc 2: only the dense entry committed — the sparse write was lost.
        let key2 = DocId::U64(2).to_bytes();
        let dense = semantic_search::VectorIndex::new(1, QuantisationStyle::MultiBit { number_of_bits: 8 }, 0.4, 0.0, 0.02, vec![]);
        crate::vector_kv::upsert_vectors(&store.db, "sem", &key2, &[dense]).await.unwrap();

        // Both indexes are incomplete → both re-enqueued.
        assert_eq!(store.reconcile_vector_indexes().await, 2);
        let mut got: Vec<Vec<u8>> = store.list_queue_entries().await.into_iter().map(|e| e.doc_id_bytes).collect();
        got.sort();
        let mut want = vec![DocId::U64(1).to_bytes(), DocId::U64(2).to_bytes()];
        want.sort();
        assert_eq!(got, want);
    }

    /// Regression for the in-flight enqueue race: a live write that enqueues a
    /// document *while* the reconciliation pass is mid-scan must not produce a
    /// duplicate queue entry. The embed queue is keyed by `(namespace, doc_id)`,
    /// so a concurrent reconciliation re-enqueue and an in-flight write collapse
    /// to a single entry. This is the invariant that keeps the queue bounded by
    /// the number of distinct docs — without it, a racing writer could grow the
    /// queue unboundedly and feed the worker an endless re-index loop.
    ///
    /// The assertion holds for every interleaving (idempotent-by-key enqueue),
    /// so the test is deterministic despite running the two paths concurrently.
    #[tokio::test]
    async fn test_reconcile_inflight_enqueue_does_not_duplicate() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        create_semantic_schema(&store, "sem").await;

        // Five docs written through the crash window (durable, but no enqueue).
        for i in 1u64..=5 {
            store
                .put("sem", DocId::U64(i), serde_json::json!({"title": format!("doc {i}")}))
                .await
                .unwrap();
        }

        // Race a live enqueue for every doc against the reconciliation pass: each
        // direct enqueue stands in for a write that lands while reconciliation is
        // scanning the same key. Both paths target the same queue key per doc.
        let keys: Vec<Vec<u8>> = (1u64..=5).map(|i| DocId::U64(i).to_bytes()).collect();
        let live = async {
            for (i, key) in keys.iter().enumerate() {
                crate::vector_kv::enqueue_embed(&store.db, "sem", key, &format!("live {}", i + 1))
                    .await
                    .unwrap();
            }
        };
        let (_, _reconciled) = tokio::join!(live, store.reconcile_vector_indexes());

        // Idempotent by doc-id: exactly one entry per doc, never doubled —
        // regardless of how the live enqueues and the reconcile pass interleaved.
        assert_eq!(
            store.pending_vector_index_count().await,
            5,
            "concurrent enqueue + reconcile must not duplicate queue entries"
        );

        // The queue is converging, not looping: a follow-up reconcile is a clean
        // no-op because every doc now has a pending entry.
        assert_eq!(store.reconcile_vector_indexes().await, 0);
        assert_eq!(store.pending_vector_index_count().await, 5);
    }
}
