//! Store lifecycle — create / list / drop for document and KV stores — and
//! schema amendment.

use super::*;

impl DocStore {
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

        #[cfg(not(feature = "semantic-search"))]
        if schema.semantic_search_enabled {
            return Err(DocStoreError::SemanticSearchNotCompiled);
        }

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

        #[cfg(not(feature = "semantic-search"))]
        if schema.semantic_search_enabled {
            return Err(DocStoreError::SemanticSearchNotCompiled);
        }

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

        // An amendment that turns on semantic search (e.g. EnableVectorIndex, or an
        // embedding attribute) is unsupported without the `semantic-search` feature.
        #[cfg(not(feature = "semantic-search"))]
        if schema.semantic_search_enabled {
            return Err(DocStoreError::SemanticSearchNotCompiled);
        }

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
    #[cfg(feature = "semantic-search")]
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
        crate::semantic_search::metrics::reset(namespace);

        if let Ok(schema) = self.load_schema(namespace)
            && let Some(ns_id) = schema.ns_id
        {
            let _ = std::fs::remove_file(vec_reindex_path(&self.db_path, ns_id));
        }

        info!("drop_vector_index_data: namespace='{namespace}' cleanup complete");
        Ok(())
    }
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

    #[cfg_attr(not(feature = "semantic-search"), allow(unused_mut))]
    let mut ns_names = vec![namespace.to_owned()];
    #[cfg(feature = "semantic-search")]
    {
        let _ = db.remove_namespace(vector_kv::sparse_vectors_ns(namespace)).await;
        let _ = db.remove_namespace(vector_kv::sparse_vectors_meta_ns(namespace)).await;
        let _ = db.remove_namespace(vector_kv::dense_vectors_ns(namespace)).await;
        ns_names.extend([
            vector_kv::sparse_vectors_ns(namespace),
            vector_kv::sparse_vectors_meta_ns(namespace),
            vector_kv::dense_vectors_ns(namespace),
        ]);
    }

    for ns_name in &ns_names {
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
    #[cfg(feature = "semantic-search")]
    crate::semantic_search::metrics::reset(namespace);

    Ok(())
}
