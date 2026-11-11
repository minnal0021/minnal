//! Opening a doc store: [`DocStoreBuilder`], which takes the engine,
//! lifecycle and semantic-search configuration and opens the store with
//! everything it runs started.
//!
//! There is one way to open a store, and an embedded store is the same as
//! the API server's: the server opens its store through this builder too.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use log::{debug, info};

use super::*;
use crate::namespace_lifecycle::{AuditLog, DEFAULT_AUDIT_RETENTION, Governor, Jobs};
#[cfg(feature = "semantic-search")]
use crate::semantic_search::service::SemanticSearchConfig;

/// Configures a [`DocStore`] and opens it ([`open`](Self::open)).
///
/// Opening a store starts everything it runs, the same in a library and in
/// the API server:
/// - the engine's background workers (value-log GC, WAL GC, LSM compaction,
///   index checkpoints, TTL);
/// - with the `semantic-search` feature, the vector-index worker (embedding
///   on write) and partition maintenance, against the embedding service in
///   the semantic-search context (default: `http://localhost:8001`);
/// - the lifecycle governor, which runs every store's create, drop and index
///   job in the background, supervised, and resumes whatever a crash left.
///
/// Call [`DocStore::shutdown`] for a clean stop.
pub struct DocStoreBuilder {
    db_path: PathBuf,
    db_config: DbConfig,
    lifecycle: LifecycleConfig,
    audit_retention: Duration,
    #[cfg(feature = "semantic-search")]
    semantic_search: SemanticSearchContext,
    #[cfg(feature = "semantic-search")]
    vector_index_config: VectorIndexConfig,
    /// Test hook: give the store a worker notifier but start no vector
    /// worker, so a test drives the worker's steps by hand.
    #[cfg(all(test, feature = "semantic-search"))]
    vector_worker: bool,
}

impl DocStoreBuilder {
    pub(super) fn new(db_path: impl AsRef<Path>) -> Self {
        Self {
            db_path: db_path.as_ref().to_path_buf(),
            db_config: DbConfig::default(),
            lifecycle: LifecycleConfig::default(),
            audit_retention: DEFAULT_AUDIT_RETENTION,
            #[cfg(feature = "semantic-search")]
            semantic_search: SemanticSearchContext::new(SemanticSearchConfig::default()),
            #[cfg(feature = "semantic-search")]
            vector_index_config: VectorIndexConfig::default(),
            #[cfg(all(test, feature = "semantic-search"))]
            vector_worker: true,
        }
    }

    /// The engine's configuration (sync policy, GC intervals, WAL segment
    /// size, ...). Default: [`DbConfig::default`].
    pub fn db_config(mut self, config: DbConfig) -> Self {
        self.db_config = config;
        self
    }

    /// How the lifecycle governor runs again a task that fails. Default:
    /// [`LifecycleConfig::default`].
    pub fn lifecycle(mut self, config: LifecycleConfig) -> Self {
        self.lifecycle = config;
        self
    }

    /// How long lifecycle audit records are kept (default 30 days). It
    /// applies to every record, those already written included: a record
    /// expires once it is older than the retention the store opens with.
    pub fn audit_retention(mut self, retention: Duration) -> Self {
        self.audit_retention = retention;
        self
    }

    /// The semantic-search context: the embedding service, the models
    /// allowed, the index defaults and the maintenance threads. Default: the
    /// built-in service settings, any model.
    #[cfg(feature = "semantic-search")]
    pub fn semantic_search(mut self, ctx: SemanticSearchContext) -> Self {
        self.semantic_search = ctx;
        self
    }

    /// The vector-index worker's tuning (retries, concurrency). Default:
    /// [`VectorIndexConfig::default`].
    #[cfg(feature = "semantic-search")]
    pub fn vector_index_config(mut self, config: VectorIndexConfig) -> Self {
        self.vector_index_config = config;
        self
    }

    /// Test hook: start no vector worker (see the field).
    #[cfg(all(test, feature = "semantic-search"))]
    pub(crate) fn without_vector_worker(mut self) -> Self {
        self.vector_worker = false;
        self
    }

    /// Open (or create) the store and start everything it runs (see
    /// [`DocStoreBuilder`]). Every stored schema's namespace is opened and its
    /// field indexes activated; a create or drop a crash interrupted is
    /// finished first, and an index job is resumed by the governor.
    pub async fn open(self) -> Result<Arc<DocStore>, DocStoreError> {
        let db_path = self.db_path.clone();
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

        let db = Arc::new(AsyncDb::open_without_workers(db_path.clone(), self.db_config.clone()).await?);
        db.hold_dir_lock(lock);
        // After WAL recovery (done by the open above), before anything reads a schema.
        let schemas = Arc::new(SchemaStore::open(&db).await?);
        let maintenance = Arc::new(MaintenanceStore::open(&db).await?);
        let audit = Arc::new(AuditLog::open(&db, self.audit_retention).await?);

        #[cfg(feature = "semantic-search")]
        let semantic_ctx = Arc::new(self.semantic_search);
        #[cfg(feature = "semantic-search")]
        let notify = Arc::new(tokio::sync::Notify::new());

        let store = DocStore {
            db,
            db_path,
            schemas,
            jobs: Arc::new(Jobs::new(Arc::clone(&maintenance), audit)),
            maintenance,
            #[cfg(feature = "semantic-search")]
            semantic_ctx: Some(Arc::clone(&semantic_ctx)),
            #[cfg(feature = "semantic-search")]
            notify: Some(Arc::clone(&notify)),
            #[cfg(feature = "semantic-search")]
            worker_handle: parking_lot::Mutex::new(None),
            #[cfg(feature = "semantic-search")]
            vector_index_config: self.vector_index_config,
            lifecycle: LifecycleLocks::default(),
            #[cfg(feature = "semantic-search")]
            in_flight: Arc::default(),
            governor: std::sync::OnceLock::new(),
            #[cfg(test)]
            lifecycle_faults: parking_lot::Mutex::default(),
        };

        // Finish what a crash interrupted, before anything opens a store's
        // namespace: a creation (its record `Creating`) and a drop (`Deleting`).
        // `db.namespace` is get-or-create, so a store left `Deleting` must be
        // gone before anything below could open its namespace again. A step
        // that fails is logged; the governor runs it again once it starts.
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

        let doc_schemas = store.load_all_schemas()?;
        info!("loaded {} schema(s)", doc_schemas.len());
        for schema in doc_schemas {
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

        // Before the vector worker starts. A vector-index drop a crash
        // interrupted is a live `vector_cleanup` job, which the governor
        // finishes.
        #[cfg(feature = "semantic-search")]
        store.ensure_vector_indexes().await?;

        // Start background workers after all indices are activated so the
        // index checkpoint worker's first immediate tick captures a complete
        // (not empty) index state.
        store.db.enable_all_workers(&self.db_config).await.map_err(DocStoreError::Db)?;

        #[cfg(feature = "semantic-search")]
        {
            store.probe_embedding_models().await;
            #[cfg(test)]
            let start_worker = self.vector_worker;
            #[cfg(not(test))]
            let start_worker = true;
            if start_worker {
                // The worker first drains whatever a crash left queued. Every
                // write queues its embed before it writes the document, so that
                // queue lists every document whose vectors may be out of date;
                // nothing scans the documents at startup (see `vector_kv`'s
                // *Queue first*).
                let handle = VecIndexWorker::start(
                    Arc::clone(&store.db),
                    semantic_ctx,
                    store.worker_host(),
                    notify,
                    store.vector_index_config.clone(),
                );
                *store.worker_handle.lock() = Some(handle);
            }
        }

        // Last, once the store is whole: the governor at once resumes every
        // index job a crash left live.
        let store = Arc::new(store);
        store
            .governor
            .get_or_init(|| Governor::start(Arc::downgrade(&store), Arc::clone(&store.jobs), self.lifecycle));
        info!("doc store ready");
        Ok(store)
    }
}

#[cfg(feature = "semantic-search")]
impl DocStore {
    /// Check the embedding service for every model a semantic store uses, and
    /// log the result. A failure only logs: the stores open, and their writes
    /// queue embeds until the service answers.
    async fn probe_embedding_models(&self) {
        let Some(ctx) = &self.semantic_ctx else { return };
        let mut in_use = std::collections::BTreeSet::new();
        for schema in self.schemas.all().iter().filter(|s| s.is_semantic_search_enabled()) {
            let settings = match schema {
                StoredSchema::Doc(s) => s.vector_settings().ok(),
                StoredSchema::Kv(s) => s.vector_settings().ok(),
            };
            if let Some(settings) = settings {
                in_use.insert((settings.embedding_model, settings.embedding_dim));
            }
        }
        for (model, dim) in in_use {
            let probe = SemanticSearchConfig {
                model_name: model.clone(),
                embedding_dim: dim as usize,
                ..ctx.config.clone()
            };
            match crate::semantic_search::service::check_embedding_service(&probe).await {
                Ok(()) => info!("embedding service at {} serves '{model}' ({dim}-d)", probe.embedding_service_url),
                Err(e) => log::error!(
                    "embedding service at {} failed the check for '{model}' ({dim}-d); semantic search for this model is unavailable until it answers: {e}",
                    probe.embedding_service_url
                ),
            }
        }
    }
}
