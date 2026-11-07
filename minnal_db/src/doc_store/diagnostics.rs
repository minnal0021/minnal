//! Store-level diagnostics and per-document reindex hooks.
//!
//! Everything here is scoped to one store. The engine's own statistics and
//! maintenance are on [`DocStore::db`], which the API server calls directly.

use super::*;

impl DocStore {
    // ── Admin / diagnostics ───────────────────────────────────────────────

    /// Return the number of documents stored in `namespace`.
    pub async fn count_docs(&self, namespace: &str) -> Result<usize, DocStoreError> {
        self.load_schema(namespace)?;
        let ns = self.store_ns(namespace).await?;
        let keys = ns.keys().await?;
        Ok(keys.len())
    }

    /// Row-map usage of a document store: row IDs allocated against live
    /// documents, and the files' size. `None` when the store has no row map —
    /// `u64` keys (row IDs come from the key) or no index ever built.
    ///
    /// Counts the live documents by listing every key, so it costs a key scan
    /// of the store; meant for occasional admin checks, not a hot path.
    pub async fn rowmap_stats(&self, namespace: &str) -> Result<Option<RowMapStats>, DocStoreError> {
        let schema = self.load_schema(namespace)?;
        let ns_id = schema.ns_id.ok_or_else(|| DocStoreError::MissingNsId {
            namespace: namespace.to_owned(),
        })?;
        let Some(ids_allocated) = self.db.rowmap_ids_allocated(ns_id)? else {
            return Ok(None);
        };
        let dir = crate::layout::namespace_index_dir(&crate::layout::index_root(&self.db_path), ns_id).join("rowmap");
        let bytes_on_disk = std::fs::read_dir(&dir)
            .map(|entries| entries.filter_map(|e| e.ok()?.metadata().ok()).map(|m| m.len()).sum())
            .unwrap_or(0);
        let live_docs = self.count_docs(namespace).await? as u64;
        Ok(Some(RowMapStats {
            ids_allocated,
            live_docs,
            dead_ids: ids_allocated.saturating_sub(live_docs),
            bytes_on_disk,
        }))
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
    pub fn field_index_blob_stats(&self, namespace: &str, field: &str) -> Option<crate::IndexBlobStats> {
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
    /// Returns the [`crate::FieldReindexOutcome`]. Errors with
    /// [`DocStoreError::IndexNotFound`] when `field` is not an indexed field of
    /// the namespace.
    pub async fn reindex_doc_field(&self, namespace: &str, id: DocId, field: &str) -> Result<crate::FieldReindexOutcome, DocStoreError> {
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

    /// Report the health of every field index in a namespace: where its
    /// persisted state reaches, whether it is active, and any outstanding gap.
    pub async fn index_health(&self, namespace: &str) -> Result<Vec<crate::db::indexing::manager::FieldIndexHealth>, DocStoreError> {
        let schema = self.load_schema(namespace)?;
        let ns_id = schema.ns_id.ok_or_else(|| DocStoreError::MissingNsId {
            namespace: namespace.to_owned(),
        })?;
        Ok(self.db.index_health(ns_id).await?)
    }

    /// Repair a degraded field index: replay its recorded key worklist (or
    /// rebuild the whole field when the keys were not capturable), then clear
    /// the gap so queries stop reporting the field as degraded.
    ///
    /// Does not re-put documents, so it generates no WAL traffic and triggers no
    /// vector re-embedding.
    ///
    /// Errors with [`DocStoreError::IndexNotFound`] when `field` is not an
    /// indexed field of the namespace.
    pub async fn repair_index(&self, namespace: &str, field: &str) -> Result<crate::field_index::FieldRepairOutcome, DocStoreError> {
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
        Ok(self.db.repair_field_index(ns_id, fm.field_id).await?)
    }

    /// Re-enqueue a single document for vector (re-)embedding — the same enqueue
    /// the write path and [`index_all`](DocStore::index_all) use, scoped to one
    /// document. The async worker picks it up on its next pass.
    ///
    /// Errors with [`DocStoreError::SemanticSearchNotEnabled`] when the namespace
    /// is not semantic-search-enabled.
    #[cfg(feature = "semantic-search")]
    pub async fn reindex_doc_vector(&self, namespace: &str, id: DocId) -> Result<VectorReindexOutcome, DocStoreError> {
        let _write = self.writes.enter(namespace)?;
        let schema = self.load_schema(namespace)?;
        if !schema.is_semantic_search_enabled() {
            return Err(DocStoreError::SemanticSearchNotEnabled {
                namespace: namespace.to_owned(),
            });
        }
        let key = id.to_bytes();
        let ns = self.store_ns(namespace).await?;
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
    #[cfg(feature = "semantic-search")]
    pub async fn kv_reindex_doc_vector(&self, namespace: &str, raw_key: &str) -> Result<VectorReindexOutcome, DocStoreError> {
        let _write = self.writes.enter(namespace)?;
        let schema = self.load_kv_schema(namespace)?;
        if !schema.is_semantic_search_enabled() {
            return Err(DocStoreError::SemanticSearchNotEnabled {
                namespace: namespace.to_owned(),
            });
        }
        let key_bytes = schema.key_type.serialize_key_from_str(raw_key)?;
        let ns = self.store_ns(namespace).await?;
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doc_store::tests::test_support::*;

    // ── count_docs ─────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_count_docs_empty_namespace() {
        let db_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path()).await;
        store.create(make_schema("empty", vec![])).await.unwrap();

        assert_eq!(store.count_docs("empty").await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_count_docs_reflects_inserts() {
        let db_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path()).await;
        store.create(make_schema("counting", vec![])).await.unwrap();

        for i in 1u64..=5 {
            store.put("counting", DocId::U64(i), serde_json::json!({"n": i})).await.unwrap();
        }

        assert_eq!(store.count_docs("counting").await.unwrap(), 5);
    }

    #[tokio::test]
    async fn test_count_docs_decrements_after_delete() {
        let db_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path()).await;
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
        let store = open_fresh(db_dir.path()).await;

        let err = store.count_docs("ghost").await.unwrap_err();
        assert!(matches!(err, DocStoreError::NotFound { .. }));
    }
}
