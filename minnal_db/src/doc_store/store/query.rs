//! Reads that span more than one document: range and prefix scans, predicate
//! queries, and semantic search.

use super::*;

impl DocStore {
    // ── Query ─────────────────────────────────────────────────────────────

    /// Fetch the query embeddings needed for a two-pass semantic search.
    ///
    /// Returns `(dense, sparse)`: the single whole-query embedding used in Pass 2
    /// dense re-ranking, and the sliding-window chunk embeddings used in Pass 1.
    /// Uses the system-wide TTL cache when possible, falling back to the embedding
    /// service (then populating the cache) on a miss.
    #[cfg(feature = "semantic-search")]
    /// `pub(super)` rather than private: called from a sibling module that was
    /// the same file before the split.
    pub(super) async fn cached_query_embeddings(
        &self,
        ctx: &SemanticSearchContext,
        query_text: &str,
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>), DocStoreError> {
        let ttl = ctx.config.query_embedding_cache_ttl;
        if let Some(cached) = vector_kv::get_cached_query_embedding(&self.db, query_text, ctx.config.embedding_dim, ttl).await {
            debug!("query embedding cache hit");
            return Ok(cached);
        }
        debug!("query embedding cache miss, calling embedding service");
        let q = crate::semantic_search::service::embed_query(&ctx.config, query_text)
            .await
            .map_err(|e| DocStoreError::EmbeddingFailed(e.to_string()))?;
        vector_kv::put_cached_query_embedding(&self.db, query_text, &q.dense, &q.sparse, ttl).await;
        Ok((q.dense, q.sparse))
    }

    /// Clear the system-wide query-embedding cache, returning the number of
    /// entries removed.
    ///
    /// The cache is keyed only by query text, so it must be cleared after any
    /// change to the chunking parameters (`window_size` / `sliding_size`) —
    /// otherwise cached sparse vectors built under the old chunking keep being
    /// served (up to the configured TTL) and silently degrade recall against
    /// freshly-indexed documents. Exposed via the admin API for this purpose.
    #[cfg(feature = "semantic-search")]
    pub async fn clear_query_embedding_cache(&self) -> Result<usize, DocStoreError> {
        let ttl = self
            .semantic_ctx
            .as_ref()
            .map(|c| c.config.query_embedding_cache_ttl)
            .unwrap_or(vector_kv::DEFAULT_QUERY_EMBEDDING_CACHE_TTL);
        Ok(vector_kv::clear_cached_query_embeddings(&self.db, ttl).await?)
    }

    /// Run an approximate nearest-neighbour semantic search against `namespace`.
    ///
    /// Embeds `query_text` using the configured embedding service, then scores
    /// every quantised vector in the namespace's companion KV store and returns
    /// the top results sorted by descending dot-product similarity.
    ///
    /// Returns [`DocStoreError::EmbeddingFailed`] if no [`SemanticSearchContext`]
    /// is attached, if the namespace does not have `semantic_search_enabled`, or
    /// if the embedding service call fails.
    #[cfg(feature = "semantic-search")]
    pub async fn search_semantic(
        &self,
        namespace: &str,
        query_text: &str,
        top_k: Option<usize>,
        pagination: Pagination,
    ) -> Result<Page<crate::semantic_search::index::vector_index::QueryResult>, DocStoreError> {
        let ctx = self
            .semantic_ctx
            .as_ref()
            .ok_or_else(|| DocStoreError::EmbeddingFailed("semantic search not configured on this store".into()))?;

        let schema = self.load_schema(namespace)?;
        if !schema.semantic_search_enabled {
            return Err(DocStoreError::EmbeddingFailed(format!(
                "namespace '{namespace}' does not have semantic_search_enabled"
            )));
        }

        debug!("semantic search namespace='{}' top_k={:?}", namespace, top_k);
        let (query_dense, query_sparse) = self.cached_query_embeddings(ctx, query_text).await?;

        let db_store = vector_kv::DbVectorStore::new(&self.db, namespace)
            .await
            .map_err(|e| DocStoreError::EmbeddingFailed(e.to_string()))?;

        let all = crate::semantic_search::service::search(
            &ctx.config,
            namespace,
            &ctx.cluster_index,
            &query_sparse,
            &query_dense,
            &db_store,
            None::<fn(&[u8]) -> bool>,
            top_k,
        )
        .await;

        debug!("semantic search namespace='{}' returned {} results", namespace, all.len());
        Ok(Page::from_vec(all, pagination))
    }

    /// Run an approximate nearest-neighbour semantic search restricted to documents
    /// that also satisfy an index `predicate`.
    ///
    /// This is a two-phase operation:
    /// 1. Execute `predicate` against the index layer to obtain a set of matching
    ///    document IDs (same semantics as [`query`]).
    /// 2. Run ANN search, but skip any candidate whose raw ID bytes are not in
    ///    that set — so only documents that pass both the semantic ranking *and*
    ///    the predicate are returned.
    ///
    /// Returns [`DocStoreError::EmbeddingFailed`] under the same conditions as
    /// [`search_semantic`], and propagates any index query errors from `predicate`.
    ///
    /// [`query`]: DocStore::query
    /// [`search_semantic`]: DocStore::search_semantic
    #[cfg(feature = "semantic-search")]
    pub async fn search_semantic_filtered(
        &self,
        namespace: &str,
        query_text: &str,
        predicate: &str,
        top_k: Option<usize>,
        pagination: Pagination,
    ) -> Result<Page<crate::semantic_search::index::vector_index::QueryResult>, DocStoreError> {
        // Phase 1: collect ALL doc IDs that satisfy the predicate (no pagination
        // here — the full set is needed as an ANN filter before scoring).
        // A degraded predicate index means the candidate set itself is short, so
        // the ANN results are filtered against an incomplete allow-list. That has
        // to reach the caller just as it does for a plain query.
        let (all_keys, degraded_fields) = self.query_all_keys(namespace, predicate).await?;
        let allowed_ids: std::collections::HashSet<Vec<u8>> = all_keys.into_iter().collect();

        // Phase 2: ANN search with the filter closure.
        let ctx = self
            .semantic_ctx
            .as_ref()
            .ok_or_else(|| DocStoreError::EmbeddingFailed("semantic search not configured on this store".into()))?;

        let schema = self.load_schema(namespace)?;
        if !schema.semantic_search_enabled {
            return Err(DocStoreError::EmbeddingFailed(format!(
                "namespace '{namespace}' does not have semantic_search_enabled"
            )));
        }

        let (query_dense, query_sparse) = self.cached_query_embeddings(ctx, query_text).await?;

        let db_store = vector_kv::DbVectorStore::new(&self.db, namespace)
            .await
            .map_err(|e| DocStoreError::EmbeddingFailed(e.to_string()))?;

        let all = crate::semantic_search::service::search(
            &ctx.config,
            namespace,
            &ctx.cluster_index,
            &query_sparse,
            &query_dense,
            &db_store,
            Some(move |id: &[u8]| allowed_ids.contains(id)),
            top_k,
        )
        .await;

        Ok(Page::from_vec(all, pagination).with_degraded_fields(degraded_fields))
    }

    /// Return documents whose IDs fall in `[start, end)`, one cursor page at a time.
    ///
    /// Pass `end = None` for an open-ended scan to the last key, and `cursor = None`
    /// for the first page; thereafter pass back the previous page's `next_cursor`.
    /// Only the page's documents are resolved from the value log. Results are
    /// ordered by ID ascending.
    pub async fn scan_range(
        &self,
        namespace: &str,
        start: DocId,
        end: Option<DocId>,
        cursor: Option<Vec<u8>>,
        limit: usize,
    ) -> Result<CursorPage<(DocId, serde_json::Value)>, DocStoreError> {
        let schema = self.load_schema(namespace)?;
        let ns = self.db.namespace(namespace.to_owned()).await?;

        let start_bytes = start.to_bytes();
        let end_bytes = end.map(|e| e.to_bytes());
        let scan_start = cursor.unwrap_or(start_bytes);
        let (pairs, next_cursor) = ns.scan(Some(scan_start), end_bytes, limit).await?;

        let results = pairs
            .into_iter()
            .map(|(k, v)| -> Result<_, DocStoreError> {
                let id = DocId::from_bytes(&k, schema.key_type)?;
                let doc = serde_json::from_slice(&v).map_err(|e| DocStoreError::InvalidId(e.to_string()))?;
                Ok((id, doc))
            })
            .collect::<Result<_, _>>()?;

        Ok(CursorPage::new(results, next_cursor))
    }

    /// Return documents whose binary key starts with `prefix`, one cursor page at a time.
    ///
    /// `prefix` is a raw byte slice of the key — callers should encode it in
    /// the same big-endian format used by [`DocId::to_bytes`].  A UUID prefix
    /// of `[0x55, 0x0e, 0x84, 0x00]` (4 bytes) matches every document whose
    /// UUID begins with `550e8400`.
    ///
    /// The prefix is scanned as the range `[prefix, prefix⁺)`, so only the page's
    /// documents are resolved. Pass `cursor = None` for the first page and pass
    /// back `next_cursor` thereafter. Returned in lexicographic key order, matching
    /// [`scan_range`].
    ///
    /// [`scan_range`]: DocStore::scan_range
    pub async fn scan_prefix(
        &self,
        namespace: &str,
        prefix: Vec<u8>,
        cursor: Option<Vec<u8>>,
        limit: usize,
    ) -> Result<CursorPage<(DocId, serde_json::Value)>, DocStoreError> {
        let schema = self.load_schema(namespace)?;
        let end_bytes = prefix_upper_bound(&prefix);
        let scan_start = cursor.unwrap_or(prefix);
        let ns = self.db.namespace(namespace.to_owned()).await?;
        let (pairs, next_cursor) = ns.scan(Some(scan_start), end_bytes, limit).await?;

        let results = pairs
            .into_iter()
            .map(|(k, v)| -> Result<_, DocStoreError> {
                let id = DocId::from_bytes(&k, schema.key_type)?;
                let doc = serde_json::from_slice(&v).map_err(|e| DocStoreError::InvalidId(e.to_string()))?;
                Ok((id, doc))
            })
            .collect::<Result<_, _>>()?;

        Ok(CursorPage::new(results, next_cursor))
    }

    /// Query documents using an index predicate.
    ///
    /// The `predicate` must reference only fields that have active indices in
    /// this store — full collection scans are not supported.
    ///
    /// # Example predicates
    /// ```text
    /// status = "active"
    /// age >= 18 AND verified = true
    /// status = "inactive" OR age < 18
    /// ```
    ///
    /// Returns a [`Page`] of `(DocId, document)` pairs for matching documents.
    pub async fn query(&self, namespace: &str, predicate: &str, pagination: Pagination) -> Result<Page<(DocId, serde_json::Value)>, DocStoreError> {
        let schema = self.load_schema(namespace)?;
        let ns_id = schema.ns_id.ok_or_else(|| DocStoreError::MissingNsId {
            namespace: namespace.to_owned(),
        })?;
        self.query_resolved(namespace, predicate, pagination, ns_id, schema.key_type).await
    }

    /// Like [`query`] but accepts a pre-resolved `ns_id` and `key_type` so the
    /// caller can supply values from an in-memory cache and avoid a disk read.
    ///
    /// [`query`]: DocStore::query
    pub async fn query_resolved(
        &self,
        namespace: &str,
        predicate: &str,
        pagination: Pagination,
        ns_id: u32,
        key_type: KeyType,
    ) -> Result<Page<(DocId, serde_json::Value)>, DocStoreError> {
        // Use the paginated variant so only the page window of keys is resolved
        // from the bitmap, not the full result set.
        let outcome = self
            .db
            .query_index_paginated(ns_id, predicate.to_owned(), pagination.offset(), pagination.page_size)
            .await?;

        // Resolve degraded field ids to names for the caller — the engine speaks
        // in `FieldId`, but every doc-store surface above here speaks in field
        // names, and an operator reading a response needs the name.
        let degraded_fields = self.degraded_field_names(ns_id, &outcome.degraded_fields);
        let total = outcome.total;

        if total == 0 {
            return Ok(Page::from_slice(vec![], pagination, 0).with_degraded_fields(degraded_fields));
        }

        let page_keys = outcome.keys;
        let ns = self.db.namespace(namespace.to_owned()).await?;
        let values = ns.get_multiple(page_keys.clone()).await;
        let mut results = Vec::with_capacity(page_keys.len());
        for (key_bytes, value_opt) in page_keys.into_iter().zip(values) {
            if let Some(bytes) = value_opt {
                let id = DocId::from_bytes(&key_bytes, key_type)?;
                let doc = serde_json::from_slice(&bytes).map_err(|e| DocStoreError::InvalidId(e.to_string()))?;
                results.push((id, doc));
            }
        }

        Ok(Page::from_slice(results, pagination, total).with_degraded_fields(degraded_fields))
    }

    /// Map degraded `FieldId`s back to their field names for this namespace.
    ///
    /// An id with no matching registered field falls back to its numeric form
    /// rather than being dropped — losing a degradation signal because a name
    /// could not be resolved would be exactly the wrong failure mode.
    fn degraded_field_names(&self, ns_id: u32, degraded: &[crate::db::namespace::FieldId]) -> Vec<String> {
        if degraded.is_empty() {
            return Vec::new();
        }
        let fields = self.db.list_index_fields(ns_id);
        degraded
            .iter()
            .map(|&fid| {
                fields
                    .iter()
                    .find(|f| f.field_id == fid)
                    .map(|f| f.field_name.clone())
                    .unwrap_or_else(|| fid.to_string())
            })
            .collect()
    }

    /// Collect all raw key bytes that match `predicate` without pagination.
    ///
    /// Used internally by [`search_semantic_filtered`] to build the full
    /// candidate ID set before ANN scoring.
    #[cfg(feature = "semantic-search")]
    async fn query_all_keys(&self, namespace: &str, predicate: &str) -> Result<(Vec<Vec<u8>>, Vec<String>), DocStoreError> {
        let schema = self.load_schema(namespace)?;
        let ns_id = schema.ns_id.ok_or_else(|| DocStoreError::MissingNsId {
            namespace: namespace.to_owned(),
        })?;
        let outcome = self.db.query_index(ns_id, predicate.to_owned()).await?;
        let degraded = self.degraded_field_names(ns_id, &outcome.degraded_fields);
        Ok((outcome.keys, degraded))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doc_store::store::test_support::*;

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
}
