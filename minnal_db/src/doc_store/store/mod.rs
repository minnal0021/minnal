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
use crate::doc_store::pagination::{CursorPage, Page, Pagination, prefix_upper_bound};
use crate::doc_store::schema::{DocStoreSchema, IndexSpec, KeyType, KvStoreSchema, SchemaAmendment, StoreType};
#[cfg(feature = "semantic-search")]
use crate::doc_store::vec_index_worker::{VecIndexWorker, VecIndexWorkerHandle, VectorIndexConfig};
#[cfg(feature = "semantic-search")]
use crate::vector_kv;

mod admin;
mod diagnostics;
mod docs;
mod helpers;
mod index_build;
mod kv;
mod query;
mod types;
mod vector;

pub use types::*;

#[allow(unused_imports)]
use helpers::*;
#[cfg(test)]
use index_build::REBUILD_PAGE_SIZE;
#[cfg(feature = "semantic-search")]
use vector::reconcile_all_vector_indexes;

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
            #[cfg(feature = "semantic-search")]
            semantic_ctx: None,
            #[cfg(feature = "semantic-search")]
            notify: None,
            #[cfg(feature = "semantic-search")]
            worker_handle: std::sync::Mutex::new(None),
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
        crate::doc_store::schema::peek_store_type(&json).ok_or_else(|| DocStoreError::NotFound {
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
            if crate::doc_store::schema::peek_store_type(&json) == Some(StoreType::Doc)
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
            if crate::doc_store::schema::peek_store_type(&json) == Some(StoreType::Kv)
                && let Ok(schema) = serde_json::from_str::<KvStoreSchema>(&json)
            {
                schemas.push(schema);
            }
        }
    }
    Ok(schemas)
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
    use crate::doc_store::schema::{AttributeType, IndexSpec, IndexType, KeyType, KvKeyType, KvValueType};
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
        DocStore::open_with_config(db_dir, schema_dir, crate::doc_store::test_db_config())
            .await
            .unwrap()
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
    #[cfg(feature = "semantic-search")]
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
            attributes: vec![crate::doc_store::schema::AttributeDef {
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
    #[cfg(feature = "semantic-search")]
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
            attributes: vec![crate::doc_store::schema::AttributeDef {
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
        let indices = (0..crate::doc_store::schema::MAX_INDICES)
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
            Err(DocStoreError::Schema(crate::doc_store::error::SchemaError::TooManyIndices { max, .. })) => {
                assert_eq!(max, crate::doc_store::schema::MAX_INDICES);
            }
            Ok(_) => panic!("expected TooManyIndices, got Ok"),
            Err(e) => panic!("expected TooManyIndices, got {e:?}"),
        }

        // And the rejected field must not have been registered/persisted.
        let loaded = DocStoreSchema::load(schema_dir.path(), "ns").unwrap();
        assert_eq!(loaded.indices.len(), crate::doc_store::schema::MAX_INDICES);
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
        use crate::doc_store::schema::KvStoreSchema;
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
            DocStoreError::Schema(crate::doc_store::error::SchemaError::KvValueTypeMismatch { .. })
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

    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn test_vector_index_max_retries_default() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        assert_eq!(store.vector_index_max_retries(), 5);
    }

    #[cfg(feature = "semantic-search")]
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

    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn test_pending_vector_index_count_empty() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        assert_eq!(store.pending_vector_index_count().await, 0);
    }

    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn test_list_pending_queue_entries_empty() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        assert!(store.list_queue_entries().await.is_empty());
    }

    #[cfg(feature = "semantic-search")]
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

    #[cfg(feature = "semantic-search")]
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

    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn test_delete_pending_queue_entry_noop_for_missing() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;

        // Deleting a non-existent entry should not error.
        store.delete_queue_entry("ns", b"ghost").await.unwrap();
        assert_eq!(store.pending_vector_index_count().await, 0);
    }

    #[cfg(feature = "semantic-search")]
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
                matches!(err, DocStoreError::Schema(crate::doc_store::error::SchemaError::InvalidNamespace)),
                "expected InvalidNamespace for '{bad_ns}', got {err:?}"
            );
        }
    }

    #[tokio::test]
    async fn test_import_schema_too_many_indices_rejected() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;

        let indices = (0..=crate::doc_store::schema::MAX_INDICES)
            .map(|i| IndexSpec {
                field: format!("f{i}"),
                index_type: IndexType::Int,
            })
            .collect();
        let err = store.create(make_schema("too_many", indices)).await.unwrap_err();
        assert!(matches!(
            err,
            DocStoreError::Schema(crate::doc_store::error::SchemaError::TooManyIndices { .. })
        ));
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
        assert!(matches!(
            err,
            DocStoreError::Schema(crate::doc_store::error::SchemaError::DuplicateFieldName { .. })
        ));
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
        schema.attributes.push(crate::doc_store::schema::AttributeDef {
            name: "agency".to_owned(),
            attr_type: crate::doc_store::schema::AttributeType::Str,
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
            key_type: crate::doc_store::schema::KeyType::U64,
            attributes: vec![crate::doc_store::schema::AttributeDef {
                name: "agency".to_owned(),
                attr_type: crate::doc_store::schema::AttributeType::Str,
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
        assert!(matches!(
            err,
            DocStoreError::Schema(crate::doc_store::error::SchemaError::DuplicateFieldName { .. })
        ));
    }

    // ── Vector-index reconciliation ─────────────────────────────────────────

    /// Without the `semantic-search` feature, creating a semantic-search-enabled
    /// store must be rejected at runtime rather than silently ignored.
    #[cfg(not(feature = "semantic-search"))]
    #[tokio::test]
    async fn test_semantic_create_rejected_without_feature() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        let schema = DocStoreSchema {
            store_type: StoreType::Doc,
            namespace: "sem".to_owned(),
            ns_id: None,
            key_type: KeyType::U64,
            attributes: vec![crate::doc_store::schema::AttributeDef {
                name: "title".to_owned(),
                attr_type: AttributeType::Str,
                description: None,
            }],
            indices: vec![],
            semantic_search_enabled: true,
            embedding_fields: vec!["title".to_owned()],
        };
        let err = store.create(schema).await.unwrap_err();
        assert!(matches!(err, DocStoreError::SemanticSearchNotCompiled), "got {err:?}");
    }

    /// Helper: create a semantic-search-enabled doc schema with a `title` field.
    #[cfg(feature = "semantic-search")]
    async fn create_semantic_schema(store: &DocStore, namespace: &str) {
        let schema = DocStoreSchema {
            store_type: StoreType::Doc,
            namespace: namespace.to_owned(),
            ns_id: None,
            key_type: KeyType::U64,
            attributes: vec![crate::doc_store::schema::AttributeDef {
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
    #[cfg(feature = "semantic-search")]
    async fn commit_complete_index(store: &DocStore, namespace: &str, key: &[u8]) {
        use crate::semantic_search::index::vector_index::QuantisationStyle;
        let sparse = crate::semantic_search::VectorIndex::new(1, QuantisationStyle::SingleBit, 0.4, 0.0, 0.02, vec![]);
        let dense = crate::semantic_search::VectorIndex::new(1, QuantisationStyle::MultiBit { number_of_bits: 8 }, 0.4, 0.0, 0.02, vec![]);
        crate::vector_kv::upsert_vectors(&store.db, namespace, key, &[sparse, dense])
            .await
            .unwrap();
    }

    /// Documents written through the crash window (doc present, but no queue
    /// entry and no vector index) must be re-enqueued by reconciliation.
    #[cfg(feature = "semantic-search")]
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
    #[cfg(feature = "semantic-search")]
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
    #[cfg(feature = "semantic-search")]
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
    #[cfg(feature = "semantic-search")]
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
    #[cfg(feature = "semantic-search")]
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
    #[cfg(feature = "semantic-search")]
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
                .put(crate::semantic_search::composite_key::encode(1, &key), b"garbage".to_vec())
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
    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn test_reconcile_kv_namespace() {
        use crate::doc_store::schema::{KvKeyType, KvStoreSchema, KvValueType};

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
    #[cfg(feature = "semantic-search")]
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
    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn test_reconcile_partial_index_is_reenqueued() {
        use crate::semantic_search::index::vector_index::QuantisationStyle;

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
        let sparse = crate::semantic_search::VectorIndex::new(1, QuantisationStyle::SingleBit, 0.4, 0.0, 0.02, vec![]);
        crate::vector_kv::upsert_vectors(&store.db, "sem", &key1, &[sparse]).await.unwrap();

        // Doc 2: only the dense entry committed — the sparse write was lost.
        let key2 = DocId::U64(2).to_bytes();
        let dense = crate::semantic_search::VectorIndex::new(1, QuantisationStyle::MultiBit { number_of_bits: 8 }, 0.4, 0.0, 0.02, vec![]);
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
    #[cfg(feature = "semantic-search")]
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
