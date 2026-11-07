//! A store's vector index over its lifetime: changing its search settings, seeding
//! its partition, turning semantic search off, and dropping its data, including
//! finishing a drop that a crash interrupted.

use crate::doc_store::*;

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

    /// Disable semantic search for a document or KV store: `semantic_search_enabled`
    /// goes to `false`, and a document store's `embedding_fields` are cleared. A
    /// document store can enable it again by amending its schema; a KV store chooses
    /// it only when it is created.
    ///
    /// Returns `Err(SemanticSearchNotEnabled)` when the namespace does not have
    /// semantic search configured, so callers can surface a proper 422.
    pub fn disable_semantic_search(&self, namespace: &str) -> Result<(), DocStoreError> {
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

    /// Finish every vector-index drop a crash interrupted; returns how many.
    ///
    /// Dropping a vector index saves the schema with semantic search off first
    /// (so new writes stop enqueueing), then clears the queue and removes the
    /// vector namespaces in the background ([`drop_vector_index_data`]). A crash
    /// in between left the vectors and the queue entries for good, and the worker
    /// would even process those entries, recreating the vector namespaces of a
    /// store without semantic search. Dropping a whole store has the same shape.
    /// So at open, before any worker starts, every namespace that has vector data
    /// or queue entries but no semantic-search schema gets the cleanup finished.
    ///
    /// [`drop_vector_index_data`]: DocStore::drop_vector_index_data
    #[cfg(feature = "semantic-search")]
    pub(in crate::doc_store) async fn finish_interrupted_vector_index_drops(&self) -> Result<usize, DocStoreError> {
        let mut semantic: std::collections::HashSet<String> = self
            .load_all_schemas()?
            .into_iter()
            .filter(|s| s.is_semantic_search_enabled())
            .map(|s| s.namespace.clone())
            .collect();
        semantic.extend(
            self.load_all_kv_schemas()?
                .into_iter()
                .filter(|s| s.is_semantic_search_enabled())
                .map(|s| s.namespace.clone()),
        );
        let mut leftover: std::collections::BTreeSet<String> = self
            .db
            .list_namespaces()
            .into_iter()
            // A store saved under such a name is not a companion, even though
            // its name is no longer accepted.
            .filter(|(name, _)| !self.schemas.contains(name))
            .filter_map(|(name, _)| vector_kv::vector_data_base(&name).map(str::to_owned))
            .collect();
        leftover.extend(vector_kv::queued_namespaces(&self.db).await?);
        leftover.retain(|ns| !semantic.contains(ns));
        for namespace in &leftover {
            warn!("finishing an interrupted vector-index drop for namespace='{namespace}'");
            self.drop_vector_index_data(namespace).await?;
        }
        Ok(leftover.len())
    }

    /// Give every semantic-search store whatever it lacks of its vector namespaces
    /// and starting partition. Writers never create them, and a crash after a drop
    /// removed them and before it re-seeded (semantic search was enabled again
    /// meanwhile) leaves a schema that promises a partition with none there; the
    /// enable route refuses such a store, as it already has a vector index. Called
    /// at open, before any drop can run.
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
    #[cfg(feature = "semantic-search")]
    pub async fn drop_vector_index_data(&self, namespace: &str) -> Result<(), DocStoreError> {
        self.without_partition_maintenance(namespace, async {
            self.drop_vector_index_data_now(namespace).await?;
            // Semantic search was turned back on before this drop ran, and the
            // partition its seeding kept is gone with the rest. The schema
            // promises one.
            let enabled = self.schemas.get(namespace).is_some_and(|s| s.is_semantic_search_enabled());
            if enabled {
                // No model check: the enable made it, and with the old partition
                // already gone, failing here would leave none.
                let (settings, _) = self.schemas.vector_settings(namespace)?;
                vector_kv::init_ivf(&self.db, namespace, settings.embedding_dim as usize).await?;
            }
            Ok(())
        })
        .await
    }

    /// Run `drop`, which removes `namespace`'s vector namespaces, with its
    /// partition's maintenance and vector writes stopped and finished first, no
    /// partition of it handed out meanwhile
    /// ([`SemanticSearchContext::begin_drop`]), and no reindex of it enqueueing:
    /// a write after the removal would recreate what was removed, and an
    /// enqueue after the queue is cleared would outlive the store.
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
            // A reindex sees the drop and stops at its next enqueue.
            self.wait_for_reindex(namespace).await;
            return drop.await;
        }
        #[cfg(not(feature = "semantic-search"))]
        let _ = namespace; // no partitions without semantic search
        drop.await
    }

    #[cfg(feature = "semantic-search")]
    async fn drop_vector_index_data_now(&self, namespace: &str) -> Result<(), DocStoreError> {
        info!("drop_vector_index_data: clearing embedding queue for namespace='{namespace}'");
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
                debug!("drop_vector_index_data: removing companion '{companion}' for '{namespace}': {e}");
            }
        }

        // Clear the in-memory corruption counters so a dropped index stops
        // showing up in /admin/indices/vector/corruption-metrics.
        crate::semantic_search::metrics::reset(namespace);
        if let Some(ctx) = &self.semantic_ctx {
            ctx.evict(namespace);
        }

        if let Some(ns_id) = self.schemas.get(namespace).and_then(|s| s.ns_id()) {
            let _ = std::fs::remove_file(vec_reindex_path(&self.db_path, ns_id));
        }

        info!("drop_vector_index_data: namespace='{namespace}' cleanup complete");
        Ok(())
    }
}
