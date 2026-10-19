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
//!   .add_index("users", spec)           ← add an index (background build)
//!   .drop_index("users", "status")      ← remove an index
//!   .amend("users", amendment)          ← add/remove non-index attributes
//!   .remove("users")                ← destroy everything
//! ```

use std::path::{Path, PathBuf};

use crate::doc_store::schema_store::{SchemaStore, StoredSchema};
use std::sync::Arc;

use crate::{AsyncDb, DbConfig, FieldId};
use log::{debug, info};
#[cfg(feature = "semantic-search")]
use log::{error, warn};

use crate::doc_store::error::{DocStoreError, SchemaError};
use crate::doc_store::hex::hex_to_bytes;
use crate::doc_store::index_observer::{ChainedObserver, DiskProgress, InMemoryProgress, IndexProgressObserver};
use crate::doc_store::index_progress::BuildStatus;
#[cfg(feature = "semantic-search")]
use crate::doc_store::index_progress::now_ms;
use crate::doc_store::pagination::{CursorPage, Page, Pagination, clamp_cursor, prefix_upper_bound};
use crate::doc_store::schema::{DocStoreSchema, IndexSpec, KeyType, KvStoreSchema, SchemaAmendment, StoreType};
#[cfg(feature = "semantic-search")]
use crate::doc_store::vec_index_worker::{VecIndexWorker, VecIndexWorkerHandle, VectorIndexConfig};
#[cfg(feature = "semantic-search")]
use crate::doc_store::vector_settings::SearchSpec;
use crate::support::file_lock::{DirLock, LockError};
#[cfg(feature = "semantic-search")]
use crate::vector_kv;

mod admin;
mod diagnostics;
mod docs;
mod helpers;
mod index_build;
mod kv;
mod query;
mod reindex_all;
#[cfg(test)]
mod str_keys_tests;
#[cfg(test)]
mod test_support;
mod types;
mod vector;
#[cfg(test)]
mod vector_settings_tests;

pub use types::*;

#[allow(unused_imports)]
use helpers::*;
#[cfg(feature = "semantic-search")]
use vector::reconcile_all_vector_indexes;

// ── DocStore ──────────────────────────────────────────────────────────────────

// The two kinds of secondary index maintained by a document store.

/// A document store manager backed by a `minnal_db` database.
///
/// Each logical document store is one minnal_db namespace whose schema is kept
/// in the `system_schema` namespace and cached in memory ([`SchemaStore`]).
/// `DocStore` owns the `AsyncDb` instance.
pub struct DocStore {
    db: Arc<AsyncDb>,
    db_path: PathBuf,
    /// Every store's schema: the only way the doc store reads or writes one.
    schemas: Arc<SchemaStore>,
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
    /// Namespaces whose `index_all` / `kv_index_all` is still enqueueing in this
    /// process. A second reindex of one of them is refused; the on-disk record
    /// only reports progress (see [`DocStore::vec_reindex_progress`]).
    #[cfg(feature = "semantic-search")]
    vec_reindexing: Arc<parking_lot::Mutex<std::collections::HashSet<String>>>,
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
        let store = Self {
            db,
            db_path,
            schemas,
            #[cfg(feature = "semantic-search")]
            semantic_ctx: None,
            #[cfg(feature = "semantic-search")]
            notify: None,
            #[cfg(feature = "semantic-search")]
            worker_handle: std::sync::Mutex::new(None),
            #[cfg(feature = "semantic-search")]
            vec_reindexing: Arc::default(),
            #[cfg(feature = "semantic-search")]
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

        // Before any vector worker can start (`with_semantic_search` comes after
        // this): finish vector-index drops a crash interrupted.
        #[cfg(feature = "semantic-search")]
        store.finish_interrupted_vector_index_drops().await?;

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
    #[cfg(feature = "semantic-search")]
    pub fn with_semantic_search(mut self, ctx: SemanticSearchContext) -> Self {
        let ctx = Arc::new(ctx);
        let notify = Arc::new(tokio::sync::Notify::new());
        let handle = VecIndexWorker::start(
            Arc::clone(&self.db),
            Arc::clone(&ctx),
            Arc::clone(&self.schemas),
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
            let schemas = Arc::clone(&self.schemas);
            let notify = Arc::clone(&notify);
            let mut startup_pass = handle.startup_pass();
            tokio::spawn(async move {
                // After the worker's first pass over what a crash left queued: see
                // `VecIndexWorkerHandle::startup_pass`. (A worker gone before then
                // drops the sender; reconcile anyway.)
                let _ = startup_pass.wait_for(|done| *done).await;
                info!("startup vector-index reconciliation: scanning for documents missing a vector index");
                let outcome = reconcile_all_vector_indexes(&db, &schemas, false).await;
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

    fn load_schema(&self, namespace: &str) -> Result<Arc<DocStoreSchema>, DocStoreError> {
        self.schemas.doc(namespace)
    }

    fn load_kv_schema(&self, namespace: &str) -> Result<Arc<KvStoreSchema>, DocStoreError> {
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

// ── Tests ──────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::doc_store::store::test_support::*;

    // ── Database lock ───────────────────────────────────────────────────────

    /// A crash leaves the lock file on disk with no live owner. Recovery must
    /// still be reachable: the previous existence-check scheme refused to open
    /// here, so every `SIGKILL` needed a manual `rm db/.lock` before the WAL
    /// could be replayed.
    /// The lock lives as long as the database, not the `DocStore` handle. A
    /// background worker's pass holds the database (an `Arc<Db>`), so it can
    /// outlive a `DocStore` dropped without `shutdown`; while it does, its final
    /// flush and compaction are still to run. Releasing the lock with the handle
    /// let a reopen use the same files meanwhile.
    #[tokio::test]
    async fn the_lock_is_held_while_anything_still_holds_the_database() {
        let db_dir = TempDir::new().unwrap();

        let store = open_fresh(db_dir.path()).await;
        let pass = Arc::clone(&store.db); // what a worker's running pass holds
        drop(store);
        match DocStore::open_with_config(db_dir.path(), crate::doc_store::test_db_config()).await {
            Err(DocStoreError::StoreLocked { .. }) => {}
            Err(e) => panic!("expected StoreLocked, got {e:?}"),
            Ok(_) => panic!("a second instance opened the files while the first was still alive"),
        }

        drop(pass);
        let _reopened = open_fresh(db_dir.path()).await;
    }

    #[tokio::test]
    async fn test_stale_lock_file_from_a_crash_does_not_block_open() {
        let db_dir = TempDir::new().unwrap();

        // Exactly what a killed process leaves behind.
        std::fs::create_dir_all(db_dir.path()).unwrap();
        std::fs::write(db_dir.path().join(".lock"), "999999\nminnal: stale\n").unwrap();

        let store = open_fresh(db_dir.path()).await;
        store.create(make_schema("docs", vec![])).await.unwrap();
    }

    /// The other direction still holds: a second store on the same directory is
    /// refused while the first is open, and the error names the holder.
    #[tokio::test]
    async fn test_second_open_of_the_same_directory_is_refused() {
        let db_dir = TempDir::new().unwrap();

        let _held = open_fresh(db_dir.path()).await;

        match DocStore::open_with_config(db_dir.path(), crate::doc_store::test_db_config()).await {
            Err(DocStoreError::StoreLocked { owner_pid, .. }) => {
                assert_eq!(owner_pid, Some(std::process::id()));
            }
            Err(e) => panic!("expected StoreLocked, got {e:?}"),
            Ok(_) => panic!("two DocStores opened the same directory at once"),
        }
    }

    // ── Restart persistence ─────────────────────────────────────────────────

    #[tokio::test]
    async fn test_kv_schema_survives_restart() {
        let db_dir = TempDir::new().unwrap();

        // First run: create KV store and write data.
        {
            let store = open_fresh(db_dir.path()).await;
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
            let store = open_fresh(db_dir.path()).await;
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

        {
            let store = open_fresh(db_dir.path()).await;
            store.create(make_schema("docs", vec![])).await.unwrap();
            store.put("docs", DocId::U64(1), serde_json::json!({"x": 1})).await.unwrap();
            store.create_kv(make_kv_schema("cache", KvKeyType::Str, KvValueType::Str)).await.unwrap();
            store.kv_put("cache", &serde_json::json!("a"), &serde_json::json!("b")).await.unwrap();
        }

        {
            let store = open_fresh(db_dir.path()).await;
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
}
