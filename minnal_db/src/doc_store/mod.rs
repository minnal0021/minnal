//! DocStore — document store built on top of minnal_db.
//!
//! A `DocStore` manages a collection of named document stores, each backed by
//! one minnal_db namespace.  Documents are stored as UTF-8 JSON values and
//! retrieved by a typed ID (`u64`, `u128`, or UUID as `u128`).
//!
//! # Lifecycle
//!
//! ```text
//! DocStore::open(db_path)               ← opens existing stores
//!   .create(schema)                     ← define a new store
//!   .put("users", id, doc)              ← write a document
//!   .get("users", id)            ← read by primary key
//!   .query("users", "status = \"active\"") ← index query
//!   .add_index("users", spec)           ← add an index (a build job)
//!   .drop_index("users", "status")      ← remove an index
//!   .amend("users", amendment)          ← add/remove non-index attributes
//!   .remove("users")                ← destroy everything
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::{AsyncDb, AsyncNamespace, DbConfig, FieldId};
#[cfg(feature = "semantic-search")]
use log::warn;
use log::{debug, info};

use self::pagination::clamp_cursor;
use self::schema::store::SchemaStore;
#[cfg(feature = "semantic-search")]
use self::schema::vector_settings::SearchSpec;
#[cfg(feature = "semantic-search")]
use crate::semantic_search::vector_kv;
#[cfg(feature = "semantic-search")]
use crate::semantic_search::workers::vec_index::{VecIndexWorker, VecIndexWorkerHandle};
use crate::support::file_lock::{DirLock, LockError};

mod diagnostics;
mod documents;
pub mod error;
mod helpers;
pub mod indexing;
pub mod key;
mod kv;
mod lifecycle;
pub mod pagination;
mod query;
pub mod schema;
#[cfg(test)]
mod tests;
mod vector;
mod write_gate;

pub use self::diagnostics::RowMapStats;
pub use self::indexing::progress::FieldBuildProgress;
#[cfg(feature = "semantic-search")]
pub use self::vector::VectorReindexOutcome;
pub use self::vector::{ReindexStats, VecReindexProgress};
#[cfg(feature = "semantic-search")]
pub(crate) use self::write_gate::{InFlightWrite, InFlightWrites};
pub(crate) use self::write_gate::{LifecycleLocks, WriteGate, WriteGuard};
pub use crate::namespace_lifecycle::{AuditEvent, AuditRecord, BuildStatus, JobRecord, LifecycleConfig, MAINTENANCE_NAMESPACE, NamespaceState};
pub(crate) use crate::namespace_lifecycle::{JobKey, MaintenanceStore};
#[cfg(feature = "semantic-search")]
pub use crate::semantic_search::context::{NamespaceSemantics, SemanticSearchContext};

#[cfg(feature = "semantic-search")]
pub(crate) use crate::semantic_search::workers::DocumentText;
#[allow(unused_imports)]
use helpers::*;

// ── Error types ────────────────────────────────────────────────────────────
pub use self::error::{DocStoreError, SchemaError};

// ── Key types ──────────────────────────────────────────────────────────────
pub use self::key::{DocId, MAX_STR_KEY_LEN, StrKey};

// ── Schema types ───────────────────────────────────────────────────────────
pub use self::schema::{
    AttributeDef, AttributeType, DocStoreSchema, IndexKind, IndexSpec, IndexType, KeyType, KvKeyType, KvStoreSchema, KvValueType, MAX_INDICES,
    SchemaAmendment, StoreType,
};

// ── Pagination types ───────────────────────────────────────────────────────
pub use self::pagination::{CursorPage, Page, Pagination};
pub use crate::support::prefix_upper_bound;

// ── Stored schemas ─────────────────────────────────────────────────────────
pub use self::schema::store::{SCHEMA_NAMESPACE, StoredSchema};

// ── Vector-index worker types (semantic-search only) ───────────────────────
#[cfg(feature = "semantic-search")]
pub use crate::semantic_search::vector_kv::{QueueEntry, QueueEntryKind};
#[cfg(feature = "semantic-search")]
pub use crate::semantic_search::workers::vec_index::VectorIndexConfig;

// ── Engine diagnostic types (re-exported for API consumers) ───────────────
pub use crate::{GCStats, KVError, LsmManifest, ManifestBucket, ManifestFile, ManifestLevel, Stats, ValueLogMetadata, WalMetadata};

// ── DocStore ──────────────────────────────────────────────────────────────────

// The two kinds of secondary index maintained by a document store.

/// A document store manager backed by a `minnal_db` database.
///
/// Each logical document store is one minnal_db namespace whose schema is kept
/// in the `system_schema` namespace and cached in memory (`SchemaStore`).
/// `DocStore` owns the `AsyncDb` instance.
pub struct DocStore {
    db: Arc<AsyncDb>,
    db_path: PathBuf,
    /// Every store's schema: the only way the doc store reads or writes one.
    schemas: Arc<SchemaStore>,
    /// Maintenance tasks and their progress (index builds, reindexes): the only
    /// way the doc store reads or writes one.
    maintenance: Arc<MaintenanceStore>,
    /// The index jobs of every store, recorded in `maintenance`.
    jobs: Arc<crate::namespace_lifecycle::Jobs>,
    /// Semantic-search context used by query paths (embedding + cluster index).
    /// `None` when semantic search is not configured.
    #[cfg(feature = "semantic-search")]
    semantic_ctx: Option<Arc<SemanticSearchContext>>,
    /// Wake signal sent to the [`VecIndexWorker`] after each write that
    /// enqueues a pending embedding.  `None` when semantic search is not
    /// configured.
    #[cfg(feature = "semantic-search")]
    notify: Option<Arc<tokio::sync::Notify>>,
    /// Handle to the background vector-index worker.  `None` when semantic
    /// search is not configured.  Wrapped in a `Mutex` so the handle can be
    /// taken out for an async graceful shutdown while `DocStore` is behind
    /// an `Arc`.
    #[cfg(feature = "semantic-search")]
    worker_handle: std::sync::Mutex<Option<VecIndexWorkerHandle>>,
    /// Configuration for the vector-index background worker.
    /// Set via [`DocStore::with_vector_index_config`] before calling
    /// [`DocStore::with_semantic_search`]; defaults are used otherwise.
    #[cfg(feature = "semantic-search")]
    vector_index_config: VectorIndexConfig,
    /// Writes in progress per store, which a drop waits for.
    writes: Arc<WriteGate>,
    /// Keeps a store's create, amendments and drop apart ([`LifecycleLocks`]).
    lifecycle: LifecycleLocks,
    /// Documents of semantic-search stores with a write in progress, which the
    /// vector worker skips (see [`InFlightWrites`]).
    #[cfg(feature = "semantic-search")]
    in_flight: Arc<InFlightWrites>,
    /// The lifecycle governor, once [`start_lifecycle`](Self::start_lifecycle)
    /// has started it.
    governor: std::sync::OnceLock<crate::namespace_lifecycle::Governor>,
    /// Test hook: failures to inject into a store's lifecycle tasks, per store
    /// (`(failures left, panic instead of an error)`).
    #[cfg(test)]
    lifecycle_faults: parking_lot::Mutex<std::collections::HashMap<String, (u32, bool)>>,
}

impl DocStore {
    // ── Construction ──────────────────────────────────────────────────────

    /// Open (or create) a `DocStore` at `db_path` with a custom [`DbConfig`].
    ///
    /// Prefer this over [`open`] when you need to tune the underlying engine
    /// (e.g. sync policy, GC intervals, WAL segment size).  For every stored
    /// schema, the corresponding namespace is opened and all field indices are
    /// activated automatically.
    ///
    /// [`open`]: DocStore::open
    pub async fn open_with_config(db_path: impl AsRef<Path>, config: DbConfig) -> Result<Self, DocStoreError> {
        let db_path = db_path.as_ref().to_path_buf();
        info!("opening doc store at {}", db_path.display());
        std::fs::create_dir_all(&db_path)?;

        // Handed to the database once it is open, which releases it only when
        // the database itself is dropped (`AsyncDb::hold_dir_lock`): a worker's
        // pass can keep the database alive after this DocStore is gone. Kept as
        // a local until then, so a failed open releases it.
        let lock = match DirLock::acquire(&db_path) {
            Ok(lock) => lock,
            Err(LockError::Held { pid }) => {
                return Err(DocStoreError::StoreLocked {
                    path: db_path,
                    owner_pid: pid,
                });
            }
            Err(LockError::Io(e)) => return Err(DocStoreError::Io(e)),
        };

        let db = Arc::new(AsyncDb::open_with_config(db_path.clone(), config.clone()).await?);
        db.hold_dir_lock(lock);
        // After WAL recovery (done by the open above), before anything reads a schema.
        let schemas = Arc::new(SchemaStore::open(&db).await?);
        let maintenance = Arc::new(MaintenanceStore::open(&db).await?);
        let audit = Arc::new(crate::namespace_lifecycle::AuditLog::open(&db, crate::namespace_lifecycle::DEFAULT_AUDIT_RETENTION).await?);
        let store = Self {
            db,
            db_path,
            schemas,
            jobs: Arc::new(crate::namespace_lifecycle::Jobs::new(Arc::clone(&maintenance), audit)),
            maintenance,
            #[cfg(feature = "semantic-search")]
            semantic_ctx: None,
            #[cfg(feature = "semantic-search")]
            notify: None,
            #[cfg(feature = "semantic-search")]
            worker_handle: std::sync::Mutex::new(None),
            #[cfg(feature = "semantic-search")]
            #[cfg(feature = "semantic-search")]
            vector_index_config: VectorIndexConfig::default(),
            writes: Arc::default(),
            lifecycle: LifecycleLocks::default(),
            #[cfg(feature = "semantic-search")]
            in_flight: Arc::default(),
            governor: std::sync::OnceLock::new(),
            #[cfg(test)]
            lifecycle_faults: parking_lot::Mutex::default(),
        };

        // Finish what a crash interrupted, before anything opens a store's
        // namespace: a creation (its record `Creating`) and a drop (`Deleting`).
        // A step that fails is logged and left for the next open; the store
        // keeps refusing requests meanwhile.
        for (namespace, state) in store.schemas.states() {
            let finished = match state {
                NamespaceState::Creating => store.provision(&namespace).await,
                NamespaceState::Deleting => store.teardown(&namespace).await,
                NamespaceState::Created | NamespaceState::IndexMaintenance => continue,
            };
            if let Err(e) = finished {
                log::error!("store '{namespace}' is left {}: finishing it at open failed: {e}", state.as_str());
            }
        }

        let schemas = store.load_all_schemas()?;
        info!("loaded {} schema(s)", schemas.len());
        for schema in schemas {
            if !store.schemas.state(&schema.namespace).is_some_and(NamespaceState::is_ready) {
                continue;
            }
            if let Some(ns_id) = schema.ns_id {
                debug!("activating indices for namespace '{}' (ns_id={})", schema.namespace, ns_id);
                activate_indices(&store.db, ns_id, &schema).await?;
            }
        }

        let kv_schemas = store.load_all_kv_schemas()?;
        info!("loaded {} KV schema(s)", kv_schemas.len());
        for schema in kv_schemas {
            if schema.ns_id.is_some() && store.schemas.state(&schema.namespace).is_some_and(NamespaceState::is_ready) {
                debug!("opening KV namespace '{}'", schema.namespace);
                store.db.namespace(schema.namespace.clone()).await?;
            }
        }

        // Before any vector worker can start (`with_semantic_search` comes after
        // this). A vector-index drop a crash interrupted is a live
        // `vector_cleanup` job, which the governor (or `run_pending_jobs`)
        // finishes.
        #[cfg(feature = "semantic-search")]
        store.ensure_vector_indexes().await?;

        // Start background workers after all indices are activated so the
        // index checkpoint worker's first immediate tick captures a complete
        // (not empty) index state.
        store.db.enable_all_workers(&config).await.map_err(DocStoreError::Db)?;

        info!("doc store ready");
        Ok(store)
    }

    /// Open (or create) a `DocStore` at `db_path` with default [`DbConfig`].
    ///
    /// For every stored schema, the corresponding namespace is opened and all
    /// field indices are activated automatically — no call to
    /// `register_index_field` is needed after the first `create`.
    pub async fn open(db_path: impl AsRef<Path>) -> Result<Self, DocStoreError> {
        Self::open_with_config(db_path, DbConfig::default()).await
    }

    /// Set the tuning parameters for the background vector-index worker.
    ///
    /// Must be called **before** [`with_semantic_search`]; configuration set
    /// after the worker has already started has no effect.
    ///
    /// [`with_semantic_search`]: DocStore::with_semantic_search
    #[cfg(feature = "semantic-search")]
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
    /// - The worker first drains whatever a crash left queued. Every write queues
    ///   its embed before it writes the document, so that queue lists every
    ///   document whose vectors may be out of date; nothing scans the documents
    ///   at startup (see `vector_kv`'s *Queue first*).
    ///
    /// Call [`shutdown_vec_index_worker`] for a clean stop before dropping the
    /// store.
    ///
    /// [`with_vector_index_config`]: DocStore::with_vector_index_config
    /// [`shutdown_vec_index_worker`]: DocStore::shutdown_vec_index_worker
    #[cfg(feature = "semantic-search")]
    pub fn with_semantic_search(mut self, ctx: SemanticSearchContext) -> Self {
        let ctx = Arc::new(ctx);
        let notify = Arc::new(tokio::sync::Notify::new());
        let handle = VecIndexWorker::start(
            Arc::clone(&self.db),
            Arc::clone(&ctx),
            self.worker_host(),
            Arc::clone(&notify),
            self.vector_index_config.clone(),
        );

        self.semantic_ctx = Some(ctx);
        self.notify = Some(notify);
        *self.worker_handle.lock().unwrap() = Some(handle);
        self
    }

    /// The engine under this store, for what is not scoped to one store: statistics
    /// (WAL, LSM, value log, operational metrics), maintenance (GC, compaction, index
    /// checkpoints) and raw namespaces.
    pub fn db(&self) -> Arc<AsyncDb> {
        Arc::clone(&self.db)
    }

    /// This store's side of the vector workers' [`WorkerHost`](crate::semantic_search::workers::WorkerHost).
    #[cfg(feature = "semantic-search")]
    fn worker_host(&self) -> Arc<self::vector::worker_host::DocStoreWorkerHost> {
        Arc::new(self::vector::worker_host::DocStoreWorkerHost {
            db: Arc::clone(&self.db),
            schemas: Arc::clone(&self.schemas),
            in_flight: Arc::clone(&self.in_flight),
        })
    }

    /// The configured maximum number of embedding attempts per queue entry.
    ///
    /// Entries whose `retry_count` reaches this value are skipped by the worker
    /// and must be removed manually via [`delete_queue_entry`].
    ///
    /// [`delete_queue_entry`]: DocStore::delete_queue_entry
    #[cfg(feature = "semantic-search")]
    pub fn vector_index_max_retries(&self) -> u32 {
        self.vector_index_config.max_retries
    }

    /// Signal the background vector-index worker to stop and await its exit.
    ///
    /// Call once during graceful server shutdown, before dropping the store.
    /// Any pending queue entries are preserved in the durable queue and will
    /// be processed on the next startup.
    #[cfg(feature = "semantic-search")]
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
        // First: a lifecycle task cut short is finished at the next open.
        if let Some(governor) = self.governor.get() {
            governor.stop();
        }
        #[cfg(feature = "semantic-search")]
        self.shutdown_vec_index_worker().await;
        self.db.shutdown().await.map_err(DocStoreError::Db)
    }

    // ── Internal helpers ──────────────────────────────────────────────────

    /// A store's schema, of whichever kind, from the in-memory cache (no I/O).
    /// `None` if there is no such store.
    pub fn schema(&self, namespace: &str) -> Option<StoredSchema> {
        self.schemas.get(namespace)
    }

    /// Every store's schema, sorted by name, from the in-memory cache (no I/O).
    pub fn schemas(&self) -> Vec<StoredSchema> {
        self.schemas.all()
    }

    /// Resolve the [`StoreType`] of an existing namespace. Returns
    /// [`DocStoreError::NotFound`] if it has no schema.
    pub fn store_type(&self, namespace: &str) -> Result<StoreType, DocStoreError> {
        self.schemas
            .get(namespace)
            .map(|s| s.store_type())
            .ok_or_else(|| DocStoreError::NotFound {
                namespace: namespace.to_owned(),
            })
    }

    /// The engine namespace holding store `namespace`'s data. Never creates it:
    /// a call racing the store's drop fails with [`DocStoreError::NotFound`]
    /// rather than bring the namespace back.
    /// The namespace of `namespace`, a store taking requests: `NotFound`,
    /// `StoreCreating` or `StoreDeleting` otherwise. Every read and write of a
    /// store's data comes through here or [`load_schema`](Self::load_schema).
    async fn store_ns(&self, namespace: &str) -> Result<AsyncNamespace, DocStoreError> {
        self.schemas.admit(namespace)?;
        self.db
            .existing_namespace(namespace.to_owned())
            .await?
            .ok_or_else(|| DocStoreError::NotFound {
                namespace: namespace.to_owned(),
            })
    }

    /// A document store's schema, for a request to its data: only a created
    /// store answers (`StoreCreating` / `StoreDeleting` otherwise). The admin
    /// views use [`view_schema`](Self::view_schema).
    fn load_schema(&self, namespace: &str) -> Result<Arc<DocStoreSchema>, DocStoreError> {
        let schema = self.schemas.doc(namespace)?;
        self.schemas.admit(namespace)?;
        Ok(schema)
    }

    /// A KV store's schema, for a request to its data; see [`load_schema`](Self::load_schema).
    fn load_kv_schema(&self, namespace: &str) -> Result<Arc<KvStoreSchema>, DocStoreError> {
        let schema = self.schemas.kv(namespace)?;
        self.schemas.admit(namespace)?;
        Ok(schema)
    }

    /// A document store's schema in whatever state the store is: for the
    /// admin views (export, diagnostics, progress), which work on a store being
    /// created or deleted too.
    fn view_schema(&self, namespace: &str) -> Result<Arc<DocStoreSchema>, DocStoreError> {
        self.schemas.doc(namespace)
    }

    /// A KV store's schema in whatever state; see [`view_schema`](Self::view_schema).
    fn view_kv_schema(&self, namespace: &str) -> Result<Arc<KvStoreSchema>, DocStoreError> {
        self.schemas.kv(namespace)
    }

    fn load_all_schemas(&self) -> Result<Vec<Arc<DocStoreSchema>>, DocStoreError> {
        Ok(self.schemas.docs())
    }

    fn load_all_kv_schemas(&self) -> Result<Vec<Arc<KvStoreSchema>>, DocStoreError> {
        Ok(self.schemas.kvs())
    }
}

// There is deliberately no `Drop` impl: the database lock is released when the
// `Db` holding the `DirLock` drops its file descriptor, and the lock file itself
// must NOT be unlinked. Removing the path while another process holds it open would
// leave that process locking an orphaned inode while a third process creates
// and locks a fresh file at the same path — two owners at once. See
// `support::file_lock`.
