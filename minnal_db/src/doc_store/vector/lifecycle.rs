//! A store's vector index over its lifetime: changing its search settings,
//! turning semantic search on (a `vector_enable` job: seed the partition, then
//! save the schema) and off (a `vector_cleanup` job: save the schema, then
//! remove the vector data).
//!
//! The two jobs conflict with each other and with every other vector job of
//! the store, so an enable can never land while a cleanup runs (FR-010: the
//! writes between them lost their queue entries).

use crate::doc_store::*;
use crate::namespace_lifecycle::{JobKind, JobRecord};

/// The progress of a `vector_enable` job: the amendment that turns semantic
/// search on, its settings settled.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct EnableProgress {
    amendment: SchemaAmendment,
}

impl DocStore {
    /// Change a store's search settings (`min_probes`, `max_probes`,
    /// `first_pass_top_k`, `top_k`) and/or its partition maintenance settings, for
    /// document and KV stores alike. Later searches, splits and merges use them;
    /// nothing is re-embedded. Fails with
    /// [`SchemaError::VectorIndexNotConfigured`] if the store has never had
    /// semantic search enabled, and with [`SchemaError::InvalidVectorSetting`]
    /// for a value out of range.
    pub async fn update_vector_search(
        &self,
        namespace: &str,
        search: &crate::doc_store::schema::vector_settings::SearchSpec,
        maintenance: Option<&crate::doc_store::schema::vector_settings::MaintenanceSpec>,
    ) -> Result<(), DocStoreError> {
        match self.store_type(namespace)? {
            StoreType::Doc => {
                self.amend(
                    namespace,
                    SchemaAmendment::UpdateVectorSearch {
                        search: *search,
                        maintenance: maintenance.copied(),
                    },
                )
                .await
            }
            StoreType::Kv => self
                .schemas
                .update_kv(namespace, |schema| Ok(schema.update_vector_search(search, maintenance)?)),
        }
    }

    /// Create `namespace`'s vector namespaces and give it its starting partition
    /// (one root posting at the zero centre), unless it already has one. Without
    /// an attached semantic context there is nothing to check the model against;
    /// the namespace gets its partition when it is next enabled with one.
    pub(in crate::doc_store) async fn seed_vector_index(
        &self,
        _namespace: &str,
        _settings: &crate::doc_store::schema::vector_settings::VectorIndexSettings,
        _vector_index: &mut Option<crate::doc_store::schema::vector_settings::VectorIndexSpec>,
    ) -> Result<(), DocStoreError> {
        #[cfg(feature = "semantic-search")]
        match &self.semantic_ctx {
            Some(ctx) => ctx.init_namespace(&self.db, _namespace, _settings).await?,
            None => vector_kv::create_vector_namespaces(&self.db, _namespace).await?,
        }
        Ok(())
    }

    /// Check that this server can embed and search with `settings` (its model
    /// has centroids of its dimension). Without an attached semantic context
    /// there is nothing to check against, so it passes.
    pub fn check_vector_model(&self, _settings: &crate::doc_store::schema::vector_settings::VectorIndexSettings) -> Result<(), DocStoreError> {
        #[cfg(feature = "semantic-search")]
        if let Some(ctx) = &self.semantic_ctx {
            ctx.check_model(_settings)?;
        }
        Ok(())
    }

    /// The defaults a namespace's vector-index settings are filled with when it
    /// first enables semantic search: the attached context's (from the server
    /// config), or the built-in ones.
    pub fn index_defaults(&self) -> crate::doc_store::schema::vector_settings::IndexDefaults {
        #[cfg(feature = "semantic-search")]
        if let Some(ctx) = &self.semantic_ctx {
            return ctx.index_defaults;
        }
        crate::doc_store::schema::vector_settings::IndexDefaults::default()
    }

    /// Turn semantic search off in a document or KV store's schema:
    /// `semantic_search_enabled` goes to `false`, and a document store's
    /// `embedding_fields` are cleared. `SemanticSearchNotEnabled` when it is off.
    pub(in crate::doc_store) fn turn_semantic_search_off(&self, namespace: &str) -> Result<(), DocStoreError> {
        let not_enabled = || DocStoreError::SemanticSearchNotEnabled {
            namespace: namespace.to_owned(),
        };
        match self.store_type(namespace)? {
            StoreType::Doc => self.schemas.update_doc(namespace, |schema| {
                if !schema.semantic_search_enabled && schema.embedding_fields.is_empty() {
                    return Err(not_enabled());
                }
                schema.semantic_search_enabled = false;
                schema.embedding_fields.clear();
                Ok(())
            }),
            StoreType::Kv => self.schemas.update_kv(namespace, |schema| {
                if !schema.semantic_search_enabled {
                    return Err(not_enabled());
                }
                schema.semantic_search_enabled = false;
                Ok(())
            }),
        }
    }

    /// Give every semantic-search store whatever it lacks of its vector namespaces
    /// and starting partition: a schema that promises a partition must have
    /// one, and writers never create it. Cheap and idempotent; called at open,
    /// before any job runs.
    #[cfg(feature = "semantic-search")]
    pub(in crate::doc_store) async fn ensure_vector_indexes(&self) -> Result<(), DocStoreError> {
        let docs = self
            .load_all_schemas()?
            .into_iter()
            .filter(|s| s.is_semantic_search_enabled())
            .map(|s| s.namespace.clone());
        let kvs = self
            .load_all_kv_schemas()?
            .into_iter()
            .filter(|s| s.is_semantic_search_enabled())
            .map(|s| s.namespace.clone());
        for namespace in docs.chain(kvs) {
            vector_kv::create_vector_namespaces(&self.db, &namespace).await?;
            if !vector_kv::ivf_exists(&self.db, &namespace).await? {
                let (settings, _) = self.schemas.vector_settings(&namespace)?;
                warn!("semantic store '{namespace}' has no partition; seeding one");
                vector_kv::init_ivf(&self.db, &namespace, settings.embedding_dim as usize).await?;
            }
        }
        Ok(())
    }

    // ── Enable: a `vector_enable` job ─────────────────────────────────────

    /// Turn semantic search on for document store `namespace` with
    /// `amendment` (one that does, its settings settled and its model
    /// checked), as a `vector_enable` job; returns once the schema says so.
    pub(in crate::doc_store) async fn enable_semantic_search(&self, namespace: &str, amendment: SchemaAmendment) -> Result<(), DocStoreError> {
        let key = self.start_vector_enable(namespace, amendment)?;
        self.run_job(namespace, key).await
    }

    /// Record the `vector_enable` job of [`enable_semantic_search`](Self::enable_semantic_search).
    pub(in crate::doc_store) fn start_vector_enable(&self, namespace: &str, amendment: SchemaAmendment) -> Result<JobKey, DocStoreError> {
        self.job_runner_ready()?;
        let ns_id = self.load_schema(namespace)?.ns_id.ok_or_else(|| DocStoreError::MissingNsId {
            namespace: namespace.to_owned(),
        })?;
        let key = JobKey::new(ns_id, JobKind::VectorEnable);
        self.start_job(namespace, key, "semantic search", &EnableProgress { amendment })?;
        Ok(key)
    }

    /// The body of a `vector_enable` job: seed the store's vector namespaces
    /// and partition, then save the schema with semantic search on. On a
    /// cancel, what the seeding made is removed again.
    pub(in crate::doc_store) async fn vector_enable_job(&self, namespace: &str, key: JobKey) -> Result<(), DocStoreError> {
        use crate::doc_store::indexing::record_error;
        let Some(record) = self.jobs.get(key)?.filter(JobRecord::is_live) else {
            return Ok(());
        };
        let progress: EnableProgress = record.progress().ok_or_else(|| record_error(key, "no amendment"))?;
        let current = self.load_schema(namespace)?;
        if current.semantic_search_enabled {
            return Ok(());
        }
        let cancelled = || self.jobs.is_cancelling(key);
        if cancelled() {
            return self.remove_vector_data(namespace).await;
        }
        let mut trial = DocStoreSchema::clone(&current);
        crate::doc_store::lifecycle::apply_amendment(namespace, &mut trial, progress.amendment.clone())?;
        let settings = trial.vector_settings()?;
        // Idempotent: a namespace that already has a partition keeps it.
        self.seed_vector_index(namespace, &settings, &mut trial.vector_index).await?;
        if cancelled() {
            return self.remove_vector_data(namespace).await;
        }
        // Applied to the schema as it is now, atomically with every other
        // schema write: a concurrent amendment cannot be lost.
        self.schemas.update_doc(namespace, |schema| {
            crate::doc_store::lifecycle::apply_amendment(namespace, schema, progress.amendment.clone())
        })
    }

    // ── Cleanup: a `vector_cleanup` job ───────────────────────────────────

    /// Turn semantic search off for a document or KV store and remove its
    /// vector index — its embedding queue entries and vector namespaces, the
    /// partition with them — as a `vector_cleanup` job; returns once it is
    /// gone. A document store's embedding fields stay as plain attributes,
    /// and its vector-index settings are kept for a later enable.
    ///
    /// Refused with [`DocStoreError::SemanticSearchNotEnabled`] for a store
    /// without semantic search, and [`DocStoreError::JobConflict`] while
    /// another vector job of it runs. It cannot be cancelled: a half-removed
    /// vector index serves neither searches nor a new enable.
    pub async fn drop_vector_index(&self, namespace: &str) -> Result<(), DocStoreError> {
        let key = self.start_vector_cleanup(namespace, || self.turn_semantic_search_off(namespace))?;
        self.run_job(namespace, key).await
    }

    /// [`drop_vector_index`](Self::drop_vector_index), returning the job's
    /// name once semantic search is off and the removal is handed to the
    /// governor.
    pub fn begin_drop_vector_index(&self, namespace: &str) -> Result<String, DocStoreError> {
        self.running_governor()?;
        let key = self.start_vector_cleanup(namespace, || self.turn_semantic_search_off(namespace))?;
        self.submit_job(namespace, key)?;
        Ok(key.kind.name().to_owned())
    }

    /// Start a `vector_cleanup` job of `namespace`, then `turn_off` its
    /// semantic search in the schema, before the job is handed on: writes stop
    /// enqueueing at once.
    pub(in crate::doc_store) fn start_vector_cleanup(
        &self,
        namespace: &str,
        turn_off: impl FnOnce() -> Result<(), DocStoreError>,
    ) -> Result<JobKey, DocStoreError> {
        self.job_runner_ready()?;
        self.schemas.admit(namespace)?;
        let schema = self.schemas.get(namespace).ok_or_else(|| DocStoreError::NotFound {
            namespace: namespace.to_owned(),
        })?;
        if !schema.is_semantic_search_enabled() {
            return Err(DocStoreError::SemanticSearchNotEnabled {
                namespace: namespace.to_owned(),
            });
        }
        let ns_id = schema.ns_id().ok_or_else(|| DocStoreError::MissingNsId {
            namespace: namespace.to_owned(),
        })?;
        let key = JobKey::new(ns_id, JobKind::VectorCleanup);
        self.start_job(namespace, key, "vector index", &serde_json::Value::Null)?;
        if let Err(e) = turn_off() {
            self.jobs.finish(self, namespace, key, Err(e.to_string()))?;
            return Err(e);
        }
        Ok(key)
    }

    /// The body of a `vector_cleanup` job: semantic search off (a crash can
    /// have come before the start turned it off), then the vector data
    /// removed.
    pub(in crate::doc_store) async fn vector_cleanup_job(&self, namespace: &str, _key: JobKey) -> Result<(), DocStoreError> {
        match self.turn_semantic_search_off(namespace) {
            Ok(()) | Err(DocStoreError::SemanticSearchNotEnabled { .. }) => {}
            Err(e) => return Err(e),
        }
        self.remove_vector_data(namespace).await
    }

    /// Remove `namespace`'s vector data, with its partition's maintenance
    /// stopped first. Idempotent.
    async fn remove_vector_data(&self, namespace: &str) -> Result<(), DocStoreError> {
        #[cfg(feature = "semantic-search")]
        return self
            .without_partition_maintenance(namespace, self.drop_vector_index_data_now(namespace))
            .await;
        #[cfg(not(feature = "semantic-search"))]
        {
            let _ = namespace; // no vector data without semantic search
            Ok(())
        }
    }

    /// Run `drop`, which removes `namespace`'s vector namespaces, with its
    /// partition's maintenance and vector writes stopped and finished first, no
    /// partition of it handed out meanwhile
    /// ([`SemanticSearchContext::begin_drop`]): a write after the removal would
    /// recreate what was removed. (A reindex enqueueing is a job, which the
    /// drop's own job or state refuses.)
    pub(in crate::doc_store) async fn without_partition_maintenance<T>(
        &self,
        namespace: &str,
        drop: impl std::future::Future<Output = Result<T, DocStoreError>>,
    ) -> Result<T, DocStoreError> {
        let _lifecycle = self.lifecycle.lock(namespace).await;
        #[cfg(feature = "semantic-search")]
        if let Some(ctx) = &self.semantic_ctx {
            // A guard, as in `without_writes`: left in place, the namespace's
            // partition is refused for good. Declared after `_lifecycle`, so the
            // drop has ended before another lifecycle operation starts.
            struct EndDrop<'a>(&'a SemanticSearchContext, &'a str);
            impl Drop for EndDrop<'_> {
                fn drop(&mut self) {
                    self.0.evict(self.1);
                    self.0.end_drop(self.1);
                }
            }
            let _end = EndDrop(ctx, namespace);
            ctx.begin_drop(namespace).await;
            return drop.await;
        }
        #[cfg(not(feature = "semantic-search"))]
        let _ = namespace; // no partitions without semantic search
        drop.await
    }

    #[cfg(feature = "semantic-search")]
    async fn drop_vector_index_data_now(&self, namespace: &str) -> Result<(), DocStoreError> {
        info!("drop_vector_index: clearing embedding queue for namespace='{namespace}'");
        self.delete_all_queue_entries(namespace).await?;

        // Remove the companion vector namespaces outright (reclaiming their
        // storage) rather than just emptying their entries — otherwise the empty
        // namespaces linger in /admin/storage/kv-namespaces as orphaned
        // "companion" stores after the index is dropped.
        // The partition and its journal go too: the partition is grown from the
        // data, so re-enabling starts again from one root posting.
        for companion in vector_kv::companion_namespaces(namespace) {
            if self.schemas.contains(&companion) {
                continue; // a store saved under that name, not a companion
            }
            if let Err(e) = self.db.remove_namespace(companion.clone()).await {
                // Best-effort: a missing companion is fine (nothing was indexed).
                debug!("drop_vector_index: removing companion '{companion}' for '{namespace}': {e}");
            }
        }

        // Clear the in-memory corruption counters so a dropped index stops
        // showing up in /admin/indices/vector/corruption-metrics.
        crate::semantic_search::metrics::reset(namespace);
        if let Some(ctx) = &self.semantic_ctx {
            ctx.evict(namespace);
        }

        // The ended records of the vector jobs describe an index that is gone.
        if let Some(ns_id) = self.schemas.get(namespace).and_then(|s| s.ns_id()) {
            self.jobs.forget(JobKey::vector_reindex(ns_id))?;
            self.jobs.forget(JobKey::new(ns_id, JobKind::PartitionRebuild))?;
        }

        info!("drop_vector_index: namespace='{namespace}' cleanup complete");
        Ok(())
    }
}
