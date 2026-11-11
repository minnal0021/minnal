//! DocStore — document store built on top of minnal_db.
//!
//! A `DocStore` manages a collection of named document stores, each backed by
//! one minnal_db namespace.  Documents are stored as UTF-8 JSON values and
//! retrieved by a typed ID (`u64`, `u128`, or UUID as `u128`).
//!
//! # Lifecycle
//!
//! ```text
//! DocStore::open(db_path)               ← opens existing stores, starts every worker
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
#[cfg(feature = "semantic-search")]
mod in_flight;
pub mod indexing;
pub mod key;
mod kv;
mod lifecycle;
mod lifecycle_locks;
mod open;
pub mod pagination;
mod query;
pub mod schema;
#[cfg(test)]
mod tests;
mod vector;
mod write_gate;

pub use self::diagnostics::RowMapStats;
#[cfg(feature = "semantic-search")]
pub(crate) use self::in_flight::{InFlightWrite, InFlightWrites};
pub use self::indexing::progress::FieldBuildProgress;
pub(crate) use self::lifecycle_locks::LifecycleLocks;
pub use self::open::DocStoreBuilder;
#[cfg(feature = "semantic-search")]
pub use self::vector::VectorReindexOutcome;
pub use self::vector::{ReindexStats, VecReindexProgress};
pub(crate) use self::write_gate::{StoreWrites, WriteGuard};
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
    /// Always set by [`DocStoreBuilder::open`]; `None` only in unit tests.
    #[cfg(feature = "semantic-search")]
    semantic_ctx: Option<Arc<SemanticSearchContext>>,
    /// Wake signal sent to the [`VecIndexWorker`] after each write that
    /// enqueues a pending embedding. Always set by [`DocStoreBuilder::open`];
    /// `None` only in unit tests.
    #[cfg(feature = "semantic-search")]
    notify: Option<Arc<tokio::sync::Notify>>,
    /// Handle to the background vector-index worker.  `None` when semantic
    /// search is not configured.  Wrapped in a `Mutex` so the handle can be
    /// taken out for an async graceful shutdown while `DocStore` is behind
    /// an `Arc`.
    #[cfg(feature = "semantic-search")]
    worker_handle: parking_lot::Mutex<Option<VecIndexWorkerHandle>>,
    /// Configuration for the vector-index background worker
    /// ([`DocStoreBuilder::vector_index_config`]).
    #[cfg(feature = "semantic-search")]
    vector_index_config: VectorIndexConfig,
    /// Keeps a store's create, amendments and drop apart ([`LifecycleLocks`]).
    lifecycle: LifecycleLocks,
    /// Documents of semantic-search stores with a write in progress, which the
    /// vector worker skips (see [`InFlightWrites`]).
    #[cfg(feature = "semantic-search")]
    in_flight: Arc<InFlightWrites>,
    /// The lifecycle governor, started by [`DocStoreBuilder::open`] once the
    /// store is whole (set right after the store is put in its `Arc`).
    governor: std::sync::OnceLock<crate::namespace_lifecycle::Governor>,
    /// Test hook: failures to inject into a store's lifecycle tasks, per store
    /// (`(failures left, panic instead of an error)`).
    #[cfg(test)]
    lifecycle_faults: parking_lot::Mutex<std::collections::HashMap<String, (u32, bool)>>,
}

impl DocStore {
    // ── Construction ──────────────────────────────────────────────────────

    /// Open (or create) a `DocStore` at `db_path` with the default
    /// configuration: the same as `DocStore::builder(db_path).open()`. See
    /// [`DocStoreBuilder`] for what opening starts.
    pub async fn open(db_path: impl AsRef<Path>) -> Result<Arc<Self>, DocStoreError> {
        Self::builder(db_path).open().await
    }

    /// A builder for a `DocStore` at `db_path`, to set its engine, lifecycle
    /// and semantic-search configuration before it opens.
    pub fn builder(db_path: impl AsRef<Path>) -> DocStoreBuilder {
        DocStoreBuilder::new(db_path)
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
        let handle = self.worker_handle.lock().take();
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
        // First, and waited for: no lifecycle task runs on while the engine
        // shuts down. A task cut short is finished at the next open.
        if let Some(governor) = self.governor.get() {
            governor.stop().await;
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

    /// The engine namespace holding the data of `namespace`, a store taking
    /// requests: `NotFound`, `StoreCreating` or `StoreDeleting` otherwise.
    /// Every read and write of a store's data comes through here or
    /// [`load_schema`](Self::load_schema). Never creates the namespace: a call
    /// racing the store's drop fails with [`DocStoreError::NotFound`] rather
    /// than bring it back.
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
