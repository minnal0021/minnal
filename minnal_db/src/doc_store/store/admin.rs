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
    // Clear this namespace's pending embed-queue entries FIRST.
    //
    // The queue outlives the store otherwise, and the vector worker then embeds
    // those entries — resolving `{ns}_sparse_vector` and friends through
    // get-or-create, which **recreates the sidecar namespaces of the store we are
    // deleting**. They survive on disk, return to the registry at every restart,
    // and are never reclaimed: three orphaned namespaces per dropped semantic
    // store. `drop_vector_index_data` has always cleared the queue; this path did
    // not.
    //
    // Best-effort, and deliberately not the only guard: the worker may already be
    // mid-entry, and a crash here would strand the rest. `VecIndexWorker` also
    // discards queue entries whose namespace has gone.
    #[cfg(feature = "semantic-search")]
    match vector_kv::list_queue_entries(db).await {
        Ok(entries) => {
            for entry in entries.into_iter().filter(|e| e.namespace == namespace) {
                if let Err(e) = vector_kv::remove_queue_entry(db, &entry.namespace, &entry.doc_id_bytes).await {
                    warn!("cleanup_store_namespaces: could not clear queue entry for '{namespace}': {e}");
                }
            }
        }
        Err(e) => warn!("cleanup_store_namespaces: could not scan the embed queue for '{namespace}': {e}"),
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doc_store::store::test_support::*;

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

    /// Regression: dropping a semantic store must clear its embed-queue entries.
    ///
    /// Left behind, the vector worker embeds them afterwards — and
    /// `upsert_vectors` resolves `{ns}_sparse_vector` and friends through
    /// get-or-create, so it **recreates the sidecar namespaces of the store that
    /// was just deleted**. Seen in a stress run: a store at ns_id 22 was dropped
    /// and its sidecars reappeared as ns_ids 30/34/38. Ids are monotonic and
    /// never reused, so they were created after the drop; they then survive on
    /// disk and return to the registry at every restart, forever.
    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn test_drop_store_clears_its_embed_queue() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;

        let mut schema = make_schema("sem_docs", vec![]);
        schema.attributes = vec![crate::doc_store::schema::AttributeDef {
            name: "title".to_owned(),
            attr_type: AttributeType::Str,
            description: None,
        }];
        schema.semantic_search_enabled = true;
        schema.embedding_fields = vec!["title".to_owned()];
        store.create(schema).await.unwrap();

        // Queue work for it, exactly as a write would. No embedding service is
        // involved — enqueuing is a plain durable write.
        for i in 0..5u32 {
            vector_kv::enqueue_embed(&store.db, "sem_docs", format!("doc{i}").as_bytes(), "the quiet harbour")
                .await
                .unwrap();
        }
        let queued = vector_kv::list_queue_entries(&store.db).await.unwrap();
        assert_eq!(
            queued.iter().filter(|e| e.namespace == "sem_docs").count(),
            5,
            "the queue should hold the enqueued work before the drop"
        );

        store.remove("sem_docs").await.unwrap();

        let after = vector_kv::list_queue_entries(&store.db).await.unwrap();
        let stranded: Vec<_> = after.iter().filter(|e| e.namespace == "sem_docs").collect();
        assert!(
            stranded.is_empty(),
            "{} queue entry/entries outlived the store they belong to — the worker will \
             embed them and recreate its vector sidecar namespaces",
            stranded.len()
        );
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
}
