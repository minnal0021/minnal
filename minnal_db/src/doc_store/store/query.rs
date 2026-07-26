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
        let all_keys = self.query_all_keys(namespace, predicate).await?;
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

        Ok(Page::from_vec(all, pagination))
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
        let (page_keys, total) = self
            .db
            .query_index_paginated(ns_id, predicate.to_owned(), pagination.offset(), pagination.page_size)
            .await?;

        if total == 0 {
            return Ok(Page::from_slice(vec![], pagination, 0));
        }

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

        Ok(Page::from_slice(results, pagination, total))
    }

    /// Collect all raw key bytes that match `predicate` without pagination.
    ///
    /// Used internally by [`search_semantic_filtered`] to build the full
    /// candidate ID set before ANN scoring.
    #[cfg(feature = "semantic-search")]
    async fn query_all_keys(&self, namespace: &str, predicate: &str) -> Result<Vec<Vec<u8>>, DocStoreError> {
        let schema = self.load_schema(namespace)?;
        let ns_id = schema.ns_id.ok_or_else(|| DocStoreError::MissingNsId {
            namespace: namespace.to_owned(),
        })?;
        Ok(self.db.query_index(ns_id, predicate.to_owned()).await?)
    }
}
