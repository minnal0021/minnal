//! Engine diagnostics and per-document reindex hooks surfaced through `DocStore`.
//!
//! Most of what follows are one-line passthroughs to the underlying [`Db`]
//! (`db_stats`, `wal_metadata`, `lsm_manifests`, `compact`, …). They exist so the
//! API server holds a single handle rather than both a `DocStore` and a `Db`.
//!
//! That convenience is also the layering complaint in the 2026-07-25 review: the
//! API server reaches the engine *through* the document layer instead of
//! alongside it. Removing them is a breaking API change, so it is tracked
//! separately — see `FEATURE-REQUEST.md` FR-004 — rather than done here.

use super::*;

impl DocStore {
    // ── Admin / diagnostics ───────────────────────────────────────────────

    /// Return the number of documents stored in `namespace`.
    pub async fn count_docs(&self, namespace: &str) -> Result<usize, DocStoreError> {
        self.load_schema(namespace)?;
        let ns = self.db.namespace(namespace.to_owned()).await?;
        let keys = ns.keys().await?;
        Ok(keys.len())
    }

    /// Returns engine-wide value-log statistics.
    pub fn db_stats(&self) -> crate::Stats {
        self.db.stats()
    }

    /// Returns a snapshot of engine-wide operational metrics (runtime counters).
    pub fn ops_metrics(&self) -> crate::MetricsSnapshot {
        self.db.ops_metrics()
    }

    /// Operational metrics for a single namespace, by name.
    ///
    /// The only failure mode is an unknown namespace, so any underlying error is
    /// surfaced as [`DocStoreError::NotFound`] (→ 404).
    pub fn ops_metrics_for(&self, namespace: &str) -> Result<crate::MetricsSnapshot, DocStoreError> {
        self.db.ops_metrics_for(namespace).map_err(|_| DocStoreError::NotFound {
            namespace: namespace.to_owned(),
        })
    }

    /// Per-namespace operational metrics for every live namespace, keyed by name.
    pub fn ops_metrics_by_namespace(&self) -> Vec<(String, crate::MetricsSnapshot)> {
        self.db.ops_metrics_by_namespace()
    }

    /// Returns a snapshot of the shared WAL metadata.
    pub fn wal_metadata(&self) -> crate::WalMetadata {
        self.db.wal_metadata()
    }

    /// Returns a live LSM manifest snapshot for every active namespace.
    pub fn lsm_manifests(&self) -> Vec<(String, crate::LsmManifest)> {
        self.db.lsm_manifests()
    }

    /// Returns the in-memory (non-SSTable) LSM stats for every active namespace.
    pub fn lsm_runtime_stats(&self) -> Vec<(String, crate::LSMStats)> {
        self.db.lsm_runtime_stats()
    }

    /// Returns per-bucket value-log metadata for every active namespace.
    pub fn value_log_shard_stats(&self) -> Vec<(String, Vec<(u32, crate::ValueLogMetadata)>)> {
        self.db.value_log_shard_stats()
    }

    /// Physical (on-disk) vs logical value-log footprint per shard, per namespace.
    pub fn value_log_physical_stats(&self) -> Vec<(String, Vec<crate::ShardPhysicalStats>)> {
        self.db.value_log_physical_stats()
    }

    /// Per-page value-log garbage breakdown for one namespace (by name).
    pub fn value_log_segment_stats(&self, namespace: &str) -> Result<Vec<(u32, Vec<crate::SegmentStats>)>, DocStoreError> {
        self.db.value_log_segment_stats(namespace).map_err(DocStoreError::from)
    }

    /// Run value-log GC on every namespace and return per-namespace results.
    pub async fn garbage_collect_all(&self) -> Vec<(String, crate::GCStats)> {
        self.db.garbage_collect_all().await
    }

    /// Run WAL garbage collection (reclaims fully-persisted WAL segments).
    pub async fn garbage_collect_wal(&self) -> Result<(u64, u64), crate::KVError> {
        self.db.garbage_collect_wal().await
    }

    /// Trigger LSM compaction across all namespaces.
    pub async fn compact(&self) -> Result<(), crate::KVError> {
        self.db.compact().await
    }

    /// Force an index checkpoint across all namespaces: flush each namespace's
    /// dense row map and all active field indexes to disk, compacting any
    /// field-index bitmap store over the configured waste threshold. Returns the
    /// number of active field indexes checkpointed.
    ///
    /// This is the same pass the periodic index-checkpoint worker runs and that
    /// shutdown runs once on close — exposed for on-demand flush + compaction.
    pub async fn checkpoint_index(&self) -> Result<usize, crate::KVError> {
        self.db.checkpoint_index().await
    }

    /// Returns all indexed fields registered for a namespace.
    pub fn list_index_fields(&self, namespace_id: u32) -> Vec<crate::FieldMeta> {
        self.db.list_index_fields(namespace_id)
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
    pub async fn index_health(&self, namespace: &str) -> Result<Vec<crate::db::index_manager::FieldIndexHealth>, DocStoreError> {
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
    pub async fn repair_index(&self, namespace: &str, field: &str) -> Result<crate::db::namespace::FieldRepairOutcome, DocStoreError> {
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
        let schema = self.load_schema(namespace)?;
        if !schema.is_semantic_search_enabled() {
            return Err(DocStoreError::SemanticSearchNotEnabled {
                namespace: namespace.to_owned(),
            });
        }
        let key = id.to_bytes();
        let ns = self.db.namespace(namespace.to_owned()).await?;
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
        let schema = self.load_kv_schema(namespace)?;
        if !schema.is_semantic_search_enabled() {
            return Err(DocStoreError::SemanticSearchNotEnabled {
                namespace: namespace.to_owned(),
            });
        }
        let key_bytes = schema.key_type.serialize_key_from_str(raw_key)?;
        let ns = self.db.namespace(namespace.to_owned()).await?;
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

    /// The configured field-index compaction threshold as a fraction (`0.0..1.0`).
    pub fn index_blob_waste_threshold(&self) -> f64 {
        self.db.index_blob_waste_threshold()
    }

    /// Returns the name and numeric ID of every KV namespace currently open in
    /// the underlying database.
    pub fn list_kv_namespaces(&self) -> Vec<(String, u32)> {
        self.db.list_namespaces()
    }

    /// Return `(ttl_secs, max_deletes_per_run)` for a namespace, or `None` if
    /// no TTL is registered.
    pub fn ttl_config_for_ns(&self, ns_id: u32) -> Option<(u64, usize)> {
        self.db.ttl_config_for_ns(ns_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doc_store::store::test_support::*;

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
}
