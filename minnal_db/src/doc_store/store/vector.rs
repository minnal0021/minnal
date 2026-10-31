//! Vector-index queue administration and bulk (re)indexing.

use super::*;

impl DocStore {
    /// The health of `namespace`'s partition (document or KV store): every
    /// posting's size, state and lineage from memory, and with `codes` also the
    /// code metrics (residual inflation, error band, foreign and zero-centre
    /// shares), which read every chunk code. Returns the namespace's
    /// maintenance settings beside them.
    #[cfg(feature = "semantic-search")]
    pub async fn vector_partition_health(
        &self,
        namespace: &str,
        codes: bool,
    ) -> Result<
        (
            crate::doc_store::vector_settings::MaintenanceSettings,
            crate::ivf_split::health::PartitionHealth,
            Vec<crate::ivf_split::health::PostingHealth>,
        ),
        DocStoreError,
    > {
        let Some(ctx) = &self.semantic_ctx else {
            return Err(DocStoreError::EmbeddingFailed("semantic search not configured on this store".into()));
        };
        self.schemas.require_semantic_search(namespace)?;
        let (settings, ns_id) = self.schemas.vector_settings(namespace)?;
        let ns = ctx.for_namespace(&self.db, namespace, ns_id, &settings).await?;
        let (summary, postings) = crate::ivf_split::health::partition_health(&self.db, namespace, &ns.partition, codes).await?;
        if codes {
            record_history(
                &self.db,
                self.vector_partition_history_ttl(),
                namespace,
                crate::ivf_split::history::RecordKind::Codes,
                &summary,
            )
            .await;
        }
        Ok((settings.maintenance, summary, postings))
    }

    /// Every IVF centre of `namespace` (document or KV store) with the postings
    /// that name it, from memory; with `codes` also how many chunk codes decode
    /// against each, which reads every chunk code. No centre vectors are
    /// returned.
    #[cfg(feature = "semantic-search")]
    pub async fn vector_centres(&self, namespace: &str, codes: bool) -> Result<Vec<crate::ivf_split::health::CentreHealth>, DocStoreError> {
        let Some(ctx) = &self.semantic_ctx else {
            return Err(DocStoreError::EmbeddingFailed("semantic search not configured on this store".into()));
        };
        self.schemas.require_semantic_search(namespace)?;
        let (settings, ns_id) = self.schemas.vector_settings(namespace)?;
        let ns = ctx.for_namespace(&self.db, namespace, ns_id, &settings).await?;
        Ok(crate::ivf_split::health::centre_health(&self.db, namespace, &ns.partition, codes).await?)
    }

    /// Check whether `namespace`'s searches miss what a full read finds, the
    /// signal for a rebuild (FR-007): sample real queries from the query-embedding
    /// cache, or stored chunks when fewer than 20 are cached (no embedding call
    /// either way), and compare each search under the namespace's settings,
    /// or `overrides` of them, with one that probes every posting. The result is
    /// recorded in the partition history. Costs `params.sample` normal searches
    /// plus as many full reads, one at a time.
    #[cfg(feature = "semantic-search")]
    pub async fn vector_recall_check(
        &self,
        namespace: &str,
        params: &crate::ivf_split::recall_check::RecallCheckParams,
        overrides: &crate::doc_store::vector_settings::SearchSpec,
    ) -> Result<crate::ivf_split::recall_check::RecallCheck, DocStoreError> {
        if self.semantic_ctx.is_none() {
            return Err(DocStoreError::EmbeddingFailed("semantic search not configured on this store".into()));
        }
        self.schemas.require_semantic_search(namespace)?;
        // Document or KV store alike.
        let (settings, ns_id) = self.schemas.vector_settings(namespace)?;
        let ns = self.namespace_semantics(namespace, Some(ns_id), &settings, overrides).await?;
        // Real queries first: the cache holds the embeddings of recent searches
        // for this model and dimension (one bounded page read, no service call).
        let cached = vector_kv::sample_cached_query_embeddings(
            &self.db,
            &ns.config.model_name,
            ns.config.embedding_dim,
            ns.config.query_embedding_cache_ttl,
            CACHED_QUERIES_READ,
            params.seed,
        )
        .await?;
        let check = crate::ivf_split::recall_check::recall_check(&self.db, namespace, &ns.partition, &ns.config, params, cached).await?;
        record_history(
            &self.db,
            self.vector_partition_history_ttl(),
            namespace,
            crate::ivf_split::history::RecordKind::RecallCheck,
            &check,
        )
        .await;
        Ok(check)
    }

    /// The server's semantic-search settings, as configured (`[semantic_search]`
    /// in the server's TOML): the embedding service, the allowed models, the
    /// TTLs, the maintenance and rebuild threads, and the defaults a namespace's
    /// vector-index settings are filled with when it enables semantic search.
    /// `None` when semantic search is not configured.
    #[cfg(feature = "semantic-search")]
    pub fn vector_server_settings(&self) -> Option<serde_json::Value> {
        let ctx = self.semantic_ctx.as_ref()?;
        let (search, m) = (ctx.index_defaults.search, ctx.index_defaults.maintenance);
        Some(serde_json::json!({
            "embedding_service_url": ctx.config.embedding_service_url,
            "supported_models": ctx.supported_models,
            "embedding_request_timeout_secs": ctx.config.embedding_request_timeout.as_secs(),
            "embedding_connect_timeout_secs": ctx.config.embedding_connect_timeout.as_secs(),
            "query_embedding_cache_ttl_secs": ctx.config.query_embedding_cache_ttl.as_secs(),
            "partition_history_ttl_secs": ctx.config.partition_history_ttl.as_secs(),
            "maintenance_threads": ctx.maintenance_threads,
            "rebuild_threads": ctx.rebuild_threads,
            "max_rebuild_threads": crate::ivf_split::rebuild::MAX_REBUILD_THREADS,
            "recall_check": {
                "max_sample": crate::ivf_split::recall_check::MAX_SAMPLE,
                "min_cached_queries": crate::ivf_split::recall_check::MIN_GIVEN_QUERIES,
                "max_reference_entries": crate::ivf_split::recall_check::MAX_REFERENCE_ENTRIES,
                "default_min_overlap": crate::ivf_split::recall_check::DEFAULT_MIN_OVERLAP,
            },
            "namespace_defaults": {
                "search": {
                    "min_probes": search.min_probes,
                    "max_probes": search.max_probes,
                    "first_pass_top_k": search.first_pass_top_k,
                    "top_k": search.top_k,
                },
                "maintenance": {
                    "target_posting_size": m.target_posting_size,
                    "split_limit": m.split_limit(),
                    "merge_limit": m.merge_limit(),
                    "reassign_range": m.reassign_range,
                    "merge_ratio": m.merge_ratio.0,
                    "skip_uncertain_moves": m.skip_uncertain_moves,
                    "split_samples": m.split_samples,
                    "split_init_trials": m.split_init_trials,
                    "split_max_iters": m.split_max_iters,
                    "split_lambda_factor": m.split_lambda_factor.0,
                },
            },
        }))
    }

    /// How long partition-history records are kept (the server's
    /// `partition_history_ttl_secs`; default 30 days without semantic search).
    #[cfg(feature = "semantic-search")]
    pub fn vector_partition_history_ttl(&self) -> std::time::Duration {
        self.semantic_ctx
            .as_ref()
            .map_or(crate::ivf_split::history::DEFAULT_HISTORY_TTL, |c| c.config.partition_history_ttl)
    }

    /// `namespace`'s partition history, newest first: code-health reads, recall
    /// checks and rebuilds, kept for the server's `partition_history_ttl_secs`
    /// (default 30 days). `kind` keeps one
    /// kind only; at most `limit` records.
    #[cfg(feature = "semantic-search")]
    pub async fn vector_partition_history(
        &self,
        namespace: &str,
        kind: Option<crate::ivf_split::history::RecordKind>,
        limit: usize,
    ) -> Result<Vec<crate::ivf_split::history::HistoryRecord>, DocStoreError> {
        if self.semantic_ctx.is_none() {
            return Err(DocStoreError::EmbeddingFailed("semantic search not configured on this store".into()));
        }
        self.schemas.require_semantic_search(namespace)?;
        Ok(crate::ivf_split::history::read(&self.db, namespace, kind, limit, self.vector_partition_history_ttl()).await?)
    }

    /// Start rebuilding `namespace`'s partition from its codes (design doc M4)
    /// in the background: k-means into `postings` postings (default: as many as
    /// are active), every chunk moved to its nearest, then the usual splits and
    /// merges. No embedding call is made. Progress and the result show in
    /// [`vector_rebuild_status`](Self::vector_rebuild_status). Fails with
    /// [`DocStoreError::VecRebuildInProgress`] while one is running.
    #[cfg(feature = "semantic-search")]
    pub fn start_vector_rebuild(&self, namespace: &str, postings: Option<usize>) -> Result<(), DocStoreError> {
        let Some(ctx) = &self.semantic_ctx else {
            return Err(DocStoreError::EmbeddingFailed("semantic search not configured on this store".into()));
        };
        self.schemas.require_semantic_search(namespace)?;
        let (settings, ns_id) = self.schemas.vector_settings(namespace)?;
        ctx.claim_rebuild(namespace, postings)?;
        let (db, ctx, namespace) = (Arc::clone(&self.db), Arc::clone(ctx), namespace.to_owned());
        tokio::spawn(async move {
            let result = async {
                let ns = ctx.for_namespace(&db, &namespace, ns_id, &settings).await?;
                let params = crate::ivf_split::rebuild::RebuildParams {
                    postings,
                    threads: ctx.rebuild_threads,
                };
                let ops = settings.maintenance.ops();
                Ok::<_, DocStoreError>(crate::ivf_split::rebuild::rebuild_partition(&db, &namespace, &ns.partition, &ops, &params).await?)
            }
            .await;
            match &result {
                Ok(r) => log::info!(
                    "partition rebuild: namespace='{namespace}' {} chunks, {} -> {} postings ({} fitted, {} iterations, {} moved)",
                    r.chunks,
                    r.postings_before,
                    r.postings_after,
                    r.postings_fitted,
                    r.iterations,
                    r.chunks_moved
                ),
                Err(e) => log::warn!("partition rebuild: namespace='{namespace}' failed: {e}"),
            }
            ctx.finish_rebuild(&namespace, result.map_err(|e| e.to_string()));
            if let Some(status) = ctx.rebuild_status(&namespace) {
                record_history(
                    &db,
                    ctx.config.partition_history_ttl,
                    &namespace,
                    crate::ivf_split::history::RecordKind::Rebuild,
                    &status,
                )
                .await;
            }
        });
        Ok(())
    }

    /// `namespace`'s last partition rebuild since the server started.
    #[cfg(feature = "semantic-search")]
    pub fn vector_rebuild_status(&self, namespace: &str) -> Option<crate::ivf_split::rebuild::RebuildStatus> {
        self.semantic_ctx.as_ref()?.rebuild_status(namespace)
    }

    /// Remove a document from the vector index: it was deleted, or its embedding
    /// text is now empty.
    ///
    /// With a worker running, this writes a `Clear` tombstone before deleting the
    /// vectors ([`vector_kv::clear_vectors`]), so a worker already embedding the
    /// document's older text removes what it writes when it completes, and wakes
    /// the worker to retire the tombstone. With no worker nothing can be in
    /// flight, so the queue entry and the vectors are removed directly.
    #[cfg(feature = "semantic-search")]
    pub(super) async fn clear_doc_vectors(&self, namespace: &str, key: &[u8]) -> Result<(), DocStoreError> {
        // The partition first, so the count change below always reaches it. A
        // change made while no partition was loaded was dropped, which is right
        // only if the next load counts after the delete; a load already
        // counting (and not yet installed) would keep the deleted document for
        // good. Loading here installs it before the delete.
        let partition = self.loaded_partition(namespace).await;
        let delta = match &self.notify {
            Some(notify) => {
                let delta = vector_kv::clear_vectors(&self.db, namespace, key).await?;
                notify.notify_one();
                delta
            }
            None => {
                vector_kv::remove_queue_entry(&self.db, namespace, key).await?;
                vector_kv::delete_vector(&self.db, namespace, key).await?
            }
        };
        if let Some(ctx) = &self.semantic_ctx {
            match partition {
                Some(p) => {
                    p.snapshot().apply_delta(&delta);
                    if p.needs_maintenance(&delta) {
                        ctx.request_maintenance(namespace);
                    }
                }
                None => ctx.apply_posting_delta(namespace, &delta),
            }
        }
        Ok(())
    }

    /// `namespace`'s partition, loading it if needed; `None` when the namespace
    /// has none to load (no semantic search, or it is being dropped).
    #[cfg(feature = "semantic-search")]
    async fn loaded_partition(&self, namespace: &str) -> Option<Arc<crate::semantic_search::PartitionHandle>> {
        let ctx = self.semantic_ctx.as_ref()?;
        let (settings, ns_id) = self.schemas.vector_settings(namespace).ok()?;
        ctx.for_namespace(&self.db, namespace, ns_id, &settings).await.ok().map(|ns| ns.partition)
    }

    /// Record a write to a semantic-search store's document until the returned
    /// guard drops (see [`InFlightWrites`]). Taken before the write's queue entry
    /// and held until its document write returns; the drop wakes the worker.
    #[cfg(feature = "semantic-search")]
    pub(super) fn begin_vector_write(&self, namespace: &str, key: &[u8]) -> InFlightWrite {
        self.in_flight.begin(namespace, key, self.notify.clone())
    }

    /// For an upsert whose embedding text is empty: clear the document's vectors
    /// and any pending embed of its older text — but only if it has some, so a
    /// bulk load of documents without embedding text pays no extra writes.
    #[cfg(feature = "semantic-search")]
    pub(super) async fn clear_doc_vectors_if_any(&self, namespace: &str, key: &[u8]) -> Result<(), DocStoreError> {
        if vector_kv::has_any_vector_state(&self.db, namespace, key).await? {
            self.clear_doc_vectors(namespace, key).await?;
        }
        Ok(())
    }

    /// Number of entries currently waiting in the async vector-index queue.
    ///
    /// This is the count of documents that have been written but whose
    /// vector index entry has not yet been committed — either because the
    /// worker has not processed them yet or the embedding service is
    /// temporarily unavailable.
    #[cfg(feature = "semantic-search")]
    pub async fn pending_vector_index_count(&self) -> usize {
        vector_kv::list_queue_entries(&self.db).await.map(|e| e.len()).unwrap_or(0)
    }

    /// Return the live count of documents that have a committed vector index
    /// entry in `{namespace}_sparse_vector_meta`.
    ///
    /// Unlike the LSM manifest entry counts, this reads through the active
    /// memtable so it reflects recently indexed documents immediately.  It also
    /// excludes tombstones (the engine merges deletes before returning results),
    /// so the count decreases as soon as a document's vector is removed — no
    /// need to wait for LSM compaction.
    ///
    /// Returns 0 when the companion namespace has not been created yet (i.e. no
    /// document has been indexed in this namespace).
    #[cfg(feature = "semantic-search")]
    pub async fn count_indexed_docs(&self, namespace: &str) -> u64 {
        let meta_ns = vector_kv::sparse_vectors_meta_ns(namespace);
        match self.db.namespace(meta_ns).await {
            Ok(ns) => ns.iter().await.map(|v| v.len() as u64).unwrap_or(0),
            Err(_) => 0,
        }
    }

    /// Return all entries currently in the async vector-index queue.
    ///
    /// Includes entries that are still pending, currently being retried, and
    /// those that have exceeded the configured `max_retries` threshold and are
    /// waiting for manual removal via [`delete_queue_entry`].
    ///
    /// [`delete_queue_entry`]: DocStore::delete_queue_entry
    #[cfg(feature = "semantic-search")]
    pub async fn list_queue_entries(&self) -> Vec<vector_kv::QueueEntry> {
        vector_kv::list_queue_entries(&self.db).await.unwrap_or_default()
    }

    /// Look up a single pending embedding queue entry by namespace and document ID.
    ///
    /// Returns `None` when no entry exists for the pair.  This is an O(1) key
    /// lookup — it does not scan the whole queue.
    #[cfg(feature = "semantic-search")]
    pub async fn get_queue_entry(&self, namespace: &str, doc_id_bytes: &[u8]) -> Option<vector_kv::QueueEntry> {
        vector_kv::get_queue_entry(&self.db, namespace, doc_id_bytes).await.unwrap_or(None)
    }

    /// Remove a specific entry from the async vector-index queue.
    ///
    /// This is an admin operation intended for entries that have exceeded
    /// `max_retries` or that need to be manually cleared.  Returns `Ok` even
    /// if no matching entry exists.
    #[cfg(feature = "semantic-search")]
    pub async fn delete_queue_entry(&self, namespace: &str, doc_id_bytes: &[u8]) -> Result<(), DocStoreError> {
        vector_kv::remove_queue_entry(&self.db, namespace, doc_id_bytes).await?;
        Ok(())
    }

    /// Reset the retry count to zero for a specific queue entry, allowing the
    /// worker to attempt embedding it again on its next pass.
    ///
    /// Returns the entry as it was before the reset (so the caller can inspect
    /// its previous `retry_count`), or `None` when no matching entry exists.
    /// The worker is notified immediately after the reset so processing starts
    /// without waiting for the next scheduled wake-up.
    #[cfg(feature = "semantic-search")]
    pub async fn retry_queue_entry(&self, namespace: &str, doc_id_bytes: &[u8]) -> Result<Option<vector_kv::QueueEntry>, DocStoreError> {
        // Atomic reset that keeps the entry's kind and text: re-enqueueing a text
        // read a moment earlier could overwrite a newer upsert, or turn a `Clear`
        // tombstone into an embed.
        let Some(entry) = vector_kv::reset_queue_entry(&self.db, namespace, doc_id_bytes).await? else {
            return Ok(None);
        };
        if let Some(notify) = &self.notify {
            notify.notify_one();
        }
        Ok(Some(entry))
    }

    /// Reset the retry count to zero for every exhausted queue entry in
    /// `namespace` (those whose `retry_count` has reached `max_retries`), so
    /// they are picked up by the worker on its next pass.
    ///
    /// Returns the number of entries that were reset.  The worker is notified
    /// immediately when at least one entry was reset.
    #[cfg(feature = "semantic-search")]
    pub async fn retry_all_failed_queue_entries(&self, namespace: &str) -> Result<usize, DocStoreError> {
        let max_retries = self.vector_index_config.max_retries;
        let entries = vector_kv::list_queue_entries(&self.db).await?;
        let exhausted: Vec<_> = entries
            .into_iter()
            .filter(|e| e.namespace == namespace && e.retry_count >= max_retries)
            .collect();
        let count = exhausted.len();
        if count == 0 {
            return Ok(0);
        }
        for entry in &exhausted {
            vector_kv::reset_queue_entry(&self.db, &entry.namespace, &entry.doc_id_bytes).await?;
        }
        if let Some(notify) = &self.notify {
            notify.notify_one();
        }
        Ok(count)
    }

    /// Delete every entry in the async vector-index queue that belongs to
    /// `namespace`, regardless of retry count.
    ///
    /// Returns the number of entries removed.  Use this to completely drain the
    /// queue for a namespace before dropping it or when forcing a clean slate.
    #[cfg(feature = "semantic-search")]
    pub async fn delete_all_queue_entries(&self, namespace: &str) -> Result<usize, DocStoreError> {
        Ok(vector_kv::clear_queue_for(&self.db, namespace).await?)
    }

    /// Validate that `namespace` exists and has semantic search enabled.
    ///
    /// Returns `Ok(())` when the namespace is ready for `index_all`.
    /// Returns `Err(NotFound)` or `Err(SemanticSearchNotEnabled)` otherwise.
    /// This is a synchronous check (no I/O) suitable for upfront validation
    /// before spawning a background task.
    pub fn check_index_all_preconditions(&self, namespace: &str) -> Result<(), DocStoreError> {
        let schema = self.load_schema(namespace)?;
        if !schema.is_semantic_search_enabled() {
            return Err(DocStoreError::SemanticSearchNotEnabled {
                namespace: namespace.to_owned(),
            });
        }
        // Also reject if a reindex of this namespace is still enqueueing.
        #[cfg(feature = "semantic-search")]
        if self.vec_reindexing.lock().contains(namespace) {
            return Err(DocStoreError::VecReindexInProgress {
                namespace: namespace.to_owned(),
            });
        }
        Ok(())
    }

    /// Return the persisted vector-index reindex record for `namespace`, or
    /// `None` when no reindex has been started for that namespace.
    ///
    /// A `"running"` record is brought up to date as it is read: it becomes
    /// `"complete"` once every document was enqueued and none of the namespace's
    /// work is left in the queue, and `"failed"` if its enqueue never finished and
    /// no reindex of the namespace is enqueueing in this process (a crash cut it
    /// short). Records used to stay `"running"` for good, which also refused every
    /// later reindex.
    #[cfg(feature = "semantic-search")]
    pub async fn vec_reindex_progress(&self, namespace: &str) -> Option<VecReindexProgress> {
        let ns_id = match self.load_schema(namespace) {
            Ok(schema) => schema.ns_id?,
            Err(_) => self.load_kv_schema(namespace).ok()?.ns_id?,
        };
        let path = vec_reindex_path(&self.db_path, ns_id);
        let mut record = read_vec_reindex(&path)?;
        if record.status != "running" || self.vec_reindexing.lock().contains(namespace) {
            return Some(record);
        }
        if !record.enqueue_done {
            record.status = "failed".to_owned();
            record.completed_at_ms = Some(now_ms());
            record.error = Some("interrupted before every document was enqueued (a restart); run reindex-all again".to_owned());
            write_vec_reindex(&path, &record);
        } else if !vector_kv::list_queue_entries(&self.db)
            .await
            .ok()?
            .iter()
            .any(|e| e.namespace == namespace)
        {
            record.status = "complete".to_owned();
            record.completed_at_ms = Some(now_ms());
            write_vec_reindex(&path, &record);
        }
        Some(record)
    }

    /// Re-enqueue every document in `namespace` for vector indexing.
    ///
    /// 1. Rejects with [`DocStoreError::VecReindexInProgress`] when a
    ///    previous `index_all` reindex is still running.
    /// 2. All **exhausted** queue entries for the namespace (those whose
    ///    `retry_count` has reached `max_retries`) are removed first so they
    ///    don't block processing.
    /// 3. Every document whose embedding fields produce non-empty text is
    ///    enqueued with `retry_count = 0`.  Documents already in the queue
    ///    are overwritten (natural deduplication — only the latest text is
    ///    kept).
    /// 4. A reindex record is written to
    ///    `{db_path}/index/{ns_id}/vector_reindex.json` so the progress API
    ///    can track this reindex across server restarts.
    /// 5. The vector-index worker is notified to start processing immediately.
    ///
    /// The documents and the queue are read a page of `REINDEX_PAGE` (1,000) at a
    /// time, so memory stays bounded however large the store is. Writes go on
    /// meanwhile, so a page can enqueue a document's older text; the worker
    /// checks every entry against its document before embedding
    /// ([`vector_kv::check_entry_against_document`]) and corrects it.
    ///
    /// Returns [`DocStoreError::SemanticSearchNotEnabled`] when the namespace
    /// does not have semantic search configured.
    #[cfg(feature = "semantic-search")]
    pub async fn index_all(&self, namespace: &str) -> Result<ReindexStats, DocStoreError> {
        let schema = self.load_schema(namespace)?;
        if !schema.is_semantic_search_enabled() {
            return Err(DocStoreError::SemanticSearchNotEnabled {
                namespace: namespace.to_owned(),
            });
        }
        let ns_id = schema.ns_id.ok_or_else(|| DocStoreError::MissingNsId {
            namespace: namespace.to_owned(),
        })?;
        self.reindex_vectors(namespace, ns_id, "index_all", |value| {
            let doc = serde_json::from_slice::<serde_json::Value>(value).ok()?;
            Some(build_embedding_text(&doc, &schema.embedding_fields)).filter(|t| !t.is_empty())
        })
        .await
    }

    /// The body of [`index_all`](Self::index_all) and
    /// [`kv_index_all`](Self::kv_index_all): `text_of` turns a stored value into
    /// its embedding text (`None`: nothing to embed).
    #[cfg(feature = "semantic-search")]
    async fn reindex_vectors(
        &self,
        namespace: &str,
        ns_id: u32,
        label: &str,
        text_of: impl Fn(&[u8]) -> Option<String>,
    ) -> Result<ReindexStats, DocStoreError> {
        // 409 guard: one reindex of a namespace enqueues at a time. Held until
        // this function returns, however it returns.
        let reindex_path = vec_reindex_path(&self.db_path, ns_id);
        let _write = self.writes.enter(namespace)?;
        let _claim = ReindexClaim::take(&self.vec_reindexing, namespace)?;
        self.check_not_dropping(namespace)?;

        // Write "running" reindex record immediately so the guard works even
        // if the process crashes before we finish enqueueing.
        let started_at_ms = now_ms();
        std::fs::create_dir_all(reindex_path.parent().unwrap())?;
        let record = |enqueued: usize, exhausted_cleared: usize, enqueue_done: bool| VecReindexProgress {
            status: "running".to_owned(),
            started_at_ms,
            completed_at_ms: None,
            total_enqueued: enqueued,
            exhausted_cleared,
            error: None,
            enqueue_done,
        };
        write_vec_reindex(&reindex_path, &record(0, 0, false));

        let max_retries = self.vector_index_config.max_retries;
        let mut exhausted_cleared = 0usize;
        let mut enqueued = 0usize;
        let mut skipped_empty_text = 0usize;

        // Clear exhausted entries (durable single-op deletes), then enqueue every
        // document with embedding text. Enqueues are WAL-backed: the queue is the
        // durable record of what still needs (re-)indexing, so every enqueue must
        // survive a crash.
        let commit_result: Result<(), DocStoreError> = async {
            info!("{label}: namespace='{namespace}' clearing exhausted queue entries (max_retries={max_retries})");
            let prefix = vector_kv::queue_prefix(namespace);
            let end = prefix_upper_bound(&prefix);
            let mut cursor = Some(prefix);
            while let Some(start) = cursor.take() {
                let (page, next) = vector_kv::queue_page(&self.db, start, end.clone(), REINDEX_PAGE).await?;
                for entry in page.iter().filter(|e| e.retry_count >= max_retries) {
                    vector_kv::remove_queue_entry(&self.db, &entry.namespace, &entry.doc_id_bytes).await?;
                    exhausted_cleared += 1;
                }
                cursor = next;
            }

            info!("{label}: namespace='{namespace}' enqueueing documents, {REINDEX_PAGE} per page");
            let ns = self.store_ns(namespace).await?;
            let mut cursor: Option<Vec<u8>> = None;
            loop {
                let (page, next) = ns.scan(cursor.take(), None, REINDEX_PAGE).await?;
                for (key, value) in &page {
                    match text_of(value) {
                        Some(text) => {
                            self.check_not_dropping(namespace)?;
                            vector_kv::enqueue_embed(&self.db, namespace, key, &text).await?;
                            enqueued += 1;
                        }
                        None => skipped_empty_text += 1,
                    }
                }
                write_vec_reindex(&reindex_path, &record(enqueued, exhausted_cleared, false));
                match next {
                    Some(c) => cursor = Some(c),
                    None => break,
                }
            }
            Ok(())
        }
        .await;

        info!(
            "{label}: namespace='{namespace}' enqueue done — enqueued={enqueued} skipped_empty_text={skipped_empty_text} \
             exhausted_cleared={exhausted_cleared}"
        );

        if let Err(e) = commit_result {
            write_vec_reindex(
                &reindex_path,
                &VecReindexProgress {
                    status: "failed".to_owned(),
                    completed_at_ms: Some(now_ms()),
                    error: Some(e.to_string()),
                    ..record(enqueued, exhausted_cleared, false)
                },
            );
            return Err(e);
        }

        // The final count; status stays "running" until the worker finishes
        // (tracked via queue depth).
        write_vec_reindex(&reindex_path, &record(enqueued, exhausted_cleared, true));

        if let Some(notify) = &self.notify {
            notify.notify_one();
        }

        Ok(ReindexStats { exhausted_cleared, enqueued })
    }

    /// Like [`check_index_all_preconditions`] but for KV store namespaces.
    ///
    /// [`check_index_all_preconditions`]: DocStore::check_index_all_preconditions
    pub fn check_kv_index_all_preconditions(&self, namespace: &str) -> Result<(), DocStoreError> {
        let schema = self.load_kv_schema(namespace)?;
        if !schema.is_semantic_search_enabled() {
            return Err(DocStoreError::SemanticSearchNotEnabled {
                namespace: namespace.to_owned(),
            });
        }
        #[cfg(feature = "semantic-search")]
        if self.vec_reindexing.lock().contains(namespace) {
            return Err(DocStoreError::VecReindexInProgress {
                namespace: namespace.to_owned(),
            });
        }
        Ok(())
    }

    /// Like [`index_all`] but for KV store namespaces.
    ///
    /// Re-enqueues every KV entry whose value is a non-empty UTF-8 string.
    /// Only valid when `semantic_search_enabled = true` and `value_type = str`.
    ///
    /// [`index_all`]: DocStore::index_all
    #[cfg(feature = "semantic-search")]
    pub async fn kv_index_all(&self, namespace: &str) -> Result<ReindexStats, DocStoreError> {
        let schema = self.load_kv_schema(namespace)?;
        if !schema.is_semantic_search_enabled() {
            return Err(DocStoreError::SemanticSearchNotEnabled {
                namespace: namespace.to_owned(),
            });
        }
        let ns_id = schema.ns_id.ok_or_else(|| DocStoreError::MissingNsId {
            namespace: namespace.to_owned(),
        })?;
        self.reindex_vectors(namespace, ns_id, "kv_index_all", |value| {
            std::str::from_utf8(value).ok().filter(|t| !t.is_empty()).map(str::to_owned)
        })
        .await
    }
}

// ── Reindex claims ────────────────────────────────────────────────────────────

/// This process's claim on reindexing one namespace: present in
/// `DocStore::vec_reindexing` while held, removed on drop (also on an early return
/// or a panic).
#[cfg(feature = "semantic-search")]
impl DocStore {
    /// Fail when `namespace`, or its vector index, is being dropped. A reindex
    /// checks it after taking its claim and before every enqueue, and either
    /// drop waits for the reindex to end before it clears the queue (a store
    /// drop through the [`WriteGate`], a vector-index drop through the claim):
    /// an enqueue after that would outlive what was dropped.
    fn check_not_dropping(&self, namespace: &str) -> Result<(), DocStoreError> {
        if self.writes.is_closed(namespace) {
            return Err(DocStoreError::NotFound {
                namespace: namespace.to_owned(),
            });
        }
        if self.semantic_ctx.as_ref().is_some_and(|ctx| ctx.is_dropping(namespace)) {
            return Err(DocStoreError::SemanticSearchNotEnabled {
                namespace: namespace.to_owned(),
            });
        }
        Ok(())
    }

    /// Wait until no reindex of `namespace` holds its claim.
    pub(super) async fn wait_for_reindex(&self, namespace: &str) {
        while self.vec_reindexing.lock().contains(namespace) {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    }
}

#[cfg(feature = "semantic-search")]
struct ReindexClaim {
    set: Arc<parking_lot::Mutex<std::collections::HashSet<String>>>,
    namespace: String,
}

#[cfg(feature = "semantic-search")]
impl ReindexClaim {
    fn take(set: &Arc<parking_lot::Mutex<std::collections::HashSet<String>>>, namespace: &str) -> Result<Self, DocStoreError> {
        if !set.lock().insert(namespace.to_owned()) {
            return Err(DocStoreError::VecReindexInProgress {
                namespace: namespace.to_owned(),
            });
        }
        Ok(Self {
            set: Arc::clone(set),
            namespace: namespace.to_owned(),
        })
    }
}

#[cfg(feature = "semantic-search")]
impl Drop for ReindexClaim {
    fn drop(&mut self) {
        self.set.lock().remove(&self.namespace);
    }
}

/// Rows a vector reindex reads per page (documents, and queue entries while it
/// clears exhausted ones): enough to amortise each page's LSM read, small enough
/// that a page of large documents stays a few MB.
#[cfg(feature = "semantic-search")]
pub(crate) const REINDEX_PAGE: usize = 1_000;

// ── What a document holds now ─────────────────────────────────────────────────

/// What a document (or KV value) holds now, for the vector worker's check of a
/// queue entry against it ([`vector_kv::check_entry_against_document`]).
#[cfg(feature = "semantic-search")]
pub(crate) enum DocumentText {
    /// The store is gone (dropped, or being dropped): leave the entry to the drop.
    StoreGone,
    /// The document is absent, or has no embedding text: it must have no vectors.
    Nothing,
    /// The document's embedding text.
    Text(String),
}

#[cfg(feature = "semantic-search")]
impl DocumentText {
    /// The text the document's vectors must be embedded from, if any.
    pub(crate) fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(t) => Some(t),
            Self::StoreGone | Self::Nothing => None,
        }
    }
}

/// Read what `namespace`'s document `key` holds now: its embedding text, built
/// as a write builds it (the schema's embedding fields for a document store, the
/// value itself for a KV store). The caller has seen no write to it in progress
/// ([`InFlightWrites`]).
#[cfg(feature = "semantic-search")]
pub(crate) async fn document_text(db: &AsyncDb, schemas: &SchemaStore, namespace: &str, key: &[u8]) -> Result<DocumentText, DocStoreError> {
    // `list_namespaces` first: resolving a dropped namespace would recreate it.
    let Some(schema) = schemas.get(namespace) else {
        return Ok(DocumentText::StoreGone);
    };
    if !db.list_namespaces().iter().any(|(name, _)| name == namespace) {
        return Ok(DocumentText::StoreGone);
    }
    let Some(value) = db.namespace(namespace.to_owned()).await?.get(key.to_vec()).await? else {
        return Ok(DocumentText::Nothing);
    };
    let text = match schema {
        StoredSchema::Doc(schema) => match serde_json::from_slice::<serde_json::Value>(&value) {
            Ok(doc) => build_embedding_text(&doc, &schema.embedding_fields),
            Err(_) => String::new(),
        },
        StoredSchema::Kv(_) => String::from_utf8(value).unwrap_or_default(),
    };
    Ok(if text.is_empty() {
        DocumentText::Nothing
    } else {
        DocumentText::Text(text)
    })
}

/// Cached query embeddings a recall check reads at most (about 3 KB each at 768
/// dimensions), to sample from.
#[cfg(feature = "semantic-search")]
const CACHED_QUERIES_READ: usize = 2_000;

/// Record a partition-history entry; a failure loses the record, never the
/// result it describes.
#[cfg(feature = "semantic-search")]
async fn record_history(
    db: &AsyncDb,
    ttl: std::time::Duration,
    namespace: &str,
    kind: crate::ivf_split::history::RecordKind,
    data: &impl serde::Serialize,
) {
    if let Err(e) = crate::ivf_split::history::record(db, namespace, kind, data, ttl).await {
        log::warn!("partition history: namespace='{namespace}' {kind:?} not recorded: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doc_store::store::test_support::*;

    // ── Vector-index queue methods ──────────────────────────────────────────

    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn test_vector_index_max_retries_default() {
        let db_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path()).await;
        assert_eq!(store.vector_index_max_retries(), 5);
    }

    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn test_with_vector_index_config_sets_max_retries() {
        let db_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path()).await.with_vector_index_config(VectorIndexConfig {
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
        let store = open_fresh(db_dir.path()).await;
        assert_eq!(store.pending_vector_index_count().await, 0);
    }

    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn test_list_pending_queue_entries_empty() {
        let db_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path()).await;
        assert!(store.list_queue_entries().await.is_empty());
    }

    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn test_pending_queue_count_and_list_after_enqueue() {
        let db_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path()).await;

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
        let store = open_fresh(db_dir.path()).await;

        vector_kv::enqueue_embed(&store.db, "ns", b"doc1", "text").await.unwrap();

        assert_eq!(store.pending_vector_index_count().await, 1);

        store.delete_queue_entry("ns", b"doc1").await.unwrap();

        assert_eq!(store.pending_vector_index_count().await, 0);
    }

    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn test_delete_pending_queue_entry_noop_for_missing() {
        let db_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path()).await;

        // Deleting a non-existent entry should not error.
        store.delete_queue_entry("ns", b"ghost").await.unwrap();
        assert_eq!(store.pending_vector_index_count().await, 0);
    }

    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn test_delete_pending_queue_entry_only_removes_target() {
        let db_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path()).await;

        vector_kv::enqueue_embed(&store.db, "ns", b"doc1", "t1").await.unwrap();
        vector_kv::enqueue_embed(&store.db, "ns", b"doc2", "t2").await.unwrap();

        store.delete_queue_entry("ns", b"doc1").await.unwrap();

        let entries = store.list_queue_entries().await;
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].doc_id_bytes, b"doc2");
    }

    /// Without the `semantic-search` feature, creating a semantic-search-enabled
    /// store must be rejected at runtime rather than silently ignored.
    #[cfg(not(feature = "semantic-search"))]
    #[tokio::test]
    async fn test_semantic_create_rejected_without_feature() {
        let db_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path()).await;
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
            vector_index: None,
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
            vector_index: None,
        };
        store.create(schema).await.unwrap();
    }

    /// A single-document vector reindex enqueues exactly that document, reports
    /// `NotFound` for a missing id, and rejects a non-semantic namespace.
    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn test_reindex_doc_vector_enqueues_single_doc() {
        let db_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path()).await;
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
                vector_index: None,
            })
            .await
            .unwrap();
        store.put("plain", DocId::U64(1), serde_json::json!({"title": "x"})).await.unwrap();
        assert!(matches!(
            store.reindex_doc_vector("plain", DocId::U64(1)).await,
            Err(DocStoreError::SemanticSearchNotEnabled { .. })
        ));
    }

    // ── Queue races through the store's write paths (R3, R4) ─────────────────

    #[cfg(feature = "semantic-search")]
    fn race_vectors() -> Vec<crate::semantic_search::VectorIndex> {
        use crate::semantic_search::QuantisationStyle;
        vec![
            crate::semantic_search::VectorIndex::new(1, QuantisationStyle::MultiBit { number_of_bits: 8 }, 0.5, 0.0, 0.01, vec![]),
            crate::semantic_search::VectorIndex::new(2, QuantisationStyle::SingleBit, 0.5, 0.0, 0.01, vec![]),
        ]
    }

    #[cfg(feature = "semantic-search")]
    async fn open_kv_semantic(db_dir: &TempDir) -> DocStore {
        let store = with_worker_notify(open_fresh(db_dir.path()).await);
        let mut schema = make_kv_schema("sem_kv", KvKeyType::Str, KvValueType::Str);
        schema.semantic_search_enabled = true;
        store.create_kv(schema).await.unwrap();
        store
    }

    /// Cost of the open-time drop sweep's queue scan when a crash mid bulk-load
    /// left the whole corpus queued (57,600 entries of ~1.2 KB text): reading
    /// every entry against reading only the keys, and a full `DocStore` open.
    /// `MINNAL_QUEUE_DOCS` overrides the size. Run with `--ignored --nocapture`.
    #[cfg(feature = "semantic-search")]
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn open_cost_with_a_full_vector_queue() {
        let n: usize = std::env::var("MINNAL_QUEUE_DOCS").ok().and_then(|v| v.parse().ok()).unwrap_or(57_600);
        let db_dir = TempDir::new().unwrap();
        {
            let store = open_kv_semantic(&db_dir).await;
            let body = "Interest rates, bond yields and portfolio risk over the market cycle. ".repeat(17);
            for i in 0..n {
                vector_kv::enqueue_embed(&store.db, "sem_kv", format!("{i:08}").as_bytes(), &format!("{i} {body}"))
                    .await
                    .unwrap();
            }
            store.db.shutdown().await.unwrap();
        }
        let t = std::time::Instant::now();
        let store = open_fresh(db_dir.path()).await;
        let open = t.elapsed();
        let t = std::time::Instant::now();
        let entries = vector_kv::list_queue_entries(&store.db).await.unwrap();
        let full = t.elapsed();
        let t = std::time::Instant::now();
        let namespaces = vector_kv::queued_namespaces(&store.db).await.unwrap();
        let keys = t.elapsed();
        assert_eq!((entries.len(), namespaces.len()), (n, 1));
        eprintln!(
            "\n=== open with {n} queued entries ===\n  DocStore open (keys-only sweep included): {:.3} s\n  queue scan reading every entry (old sweep): {:.3} s\n  queue scan reading keys only (new sweep):   {:.3} s",
            open.as_secs_f64(),
            full.as_secs_f64(),
            keys.as_secs_f64()
        );
    }

    /// A vector-index drop cut short by a crash (schema saved with semantic
    /// search off, data and queue not yet cleared) is finished at the next open,
    /// before a worker could process the leftover queue entries.
    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn an_interrupted_vector_index_drop_is_finished_at_open() {
        let db_dir = TempDir::new().unwrap();
        {
            let store = open_kv_semantic(&db_dir).await;
            store
                .kv_put("sem_kv", &serde_json::json!("x"), &serde_json::json!("some text"))
                .await
                .unwrap(); // queued
            let entry = vector_kv::get_queue_entry(&store.db, "sem_kv", b"x").await.unwrap().unwrap();
            vector_kv::upsert_vectors(&store.db, "sem_kv", b"x", &entry.text, &race_vectors())
                .await
                .unwrap();
            store
                .kv_put("sem_kv", &serde_json::json!("y"), &serde_json::json!("other text"))
                .await
                .unwrap(); // still queued
            // The drop's first step only: the schema, then a crash.
            store
                .schemas
                .update_kv("sem_kv", |schema| {
                    schema.semantic_search_enabled = false;
                    Ok(())
                })
                .unwrap();
            store.db.shutdown().await.unwrap();
        }
        let store = open_fresh(db_dir.path()).await;
        let names: Vec<String> = store.db.list_namespaces().into_iter().map(|(n, _)| n).collect();
        assert!(
            !names.iter().any(|n| vector_kv::companion_base(n) == Some("sem_kv")),
            "vector namespaces of the dropped index remain: {names:?}"
        );
        assert!(store.list_queue_entries().await.is_empty(), "queue entries of the dropped index remain");
    }

    /// A finished reindex must not block the next one. Its record used to stay
    /// `"running"` for good, so every later `reindex-all` was refused as already
    /// in progress.
    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn a_finished_reindex_does_not_block_the_next() {
        let db_dir = TempDir::new().unwrap();
        let store = open_kv_semantic(&db_dir).await;
        store
            .kv_put("sem_kv", &serde_json::json!("x"), &serde_json::json!("some text"))
            .await
            .unwrap();
        store.kv_index_all("sem_kv").await.unwrap();
        // The worker finishes the work: the queue empties.
        for e in store.list_queue_entries().await {
            vector_kv::remove_queue_entry(&store.db, &e.namespace, &e.doc_id_bytes).await.unwrap();
        }
        assert_eq!(store.vec_reindex_progress("sem_kv").await.unwrap().status, "complete");
        store.check_kv_index_all_preconditions("sem_kv").unwrap();
        store.kv_index_all("sem_kv").await.expect("a second reindex must be allowed");
        // While its work is still queued, it is running.
        assert_eq!(store.vec_reindex_progress("sem_kv").await.unwrap().status, "running");
    }

    /// A reindex reads the store a page at a time: every entry is enqueued across
    /// page boundaries, and exhausted entries are cleared from every page of the
    /// queue, without touching another store's entries.
    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn a_reindex_covers_every_page() {
        let db_dir = TempDir::new().unwrap();
        let store = open_kv_semantic(&db_dir).await;
        let parent = store.db.namespace("sem_kv".to_string()).await.unwrap();
        let n = REINDEX_PAGE * 2 + 7;
        for i in 0..n {
            parent
                .put(format!("{i:06}").into_bytes(), format!("text {i}").into_bytes())
                .await
                .unwrap();
        }
        parent.put(b"empty".to_vec(), Vec::new()).await.unwrap(); // nothing to embed
        // Exhausted entries spread over more than one queue page, and one of
        // another store, which must stay.
        let max = store.vector_index_max_retries();
        for (ns, key) in [("sem_kv", "000000"), ("sem_kv", "001500"), ("other", "x")] {
            vector_kv::enqueue_embed(&store.db, ns, key.as_bytes(), "stale").await.unwrap();
            for _ in 0..max {
                let e = vector_kv::get_queue_entry(&store.db, ns, key.as_bytes()).await.unwrap().unwrap();
                vector_kv::record_queue_failure(&store.db, &e, "boom").await.unwrap();
            }
        }

        let stats = store.kv_index_all("sem_kv").await.unwrap();
        assert_eq!((stats.enqueued, stats.exhausted_cleared), (n, 2));
        let queued = vector_kv::list_queue_entries(&store.db).await.unwrap();
        assert_eq!(queued.iter().filter(|e| e.namespace == "sem_kv" && e.retry_count == 0).count(), n);
        assert!(queued.iter().any(|e| e.namespace == "other"), "another store's entry was removed");
        let progress = store.vec_reindex_progress("sem_kv").await.unwrap();
        assert_eq!((progress.total_enqueued, progress.exhausted_cleared), (n, 2));
    }

    /// A reindex interrupted by a crash mid-enqueue is reported as failed and does
    /// not block a new one.
    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn a_reindex_interrupted_by_a_crash_does_not_block_the_next() {
        let db_dir = TempDir::new().unwrap();
        {
            let store = open_kv_semantic(&db_dir).await;
            store
                .kv_put("sem_kv", &serde_json::json!("x"), &serde_json::json!("some text"))
                .await
                .unwrap();
            let ns_id = store.load_kv_schema("sem_kv").unwrap().ns_id.unwrap();
            // What a crash right after the reindex started leaves on disk.
            let path = vec_reindex_path(&store.db_path, ns_id);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            write_vec_reindex(
                &path,
                &VecReindexProgress {
                    status: "running".to_owned(),
                    started_at_ms: now_ms(),
                    completed_at_ms: None,
                    total_enqueued: 0,
                    exhausted_cleared: 0,
                    error: None,
                    enqueue_done: false,
                },
            );
            store.db.shutdown().await.unwrap();
        }
        let store = with_worker_notify(open_fresh(db_dir.path()).await);
        let progress = store.vec_reindex_progress("sem_kv").await.unwrap();
        assert_eq!(progress.status, "failed", "{progress:?}");
        store.kv_index_all("sem_kv").await.expect("a new reindex must be allowed");
    }

    /// A delete interrupted by a crash leaves the document, no vectors and a
    /// `Clear` tombstone. The worker compares the tombstone with the document,
    /// finds the document still there, and turns the tombstone into an embed of
    /// its text, so the surviving document is indexed again.
    #[cfg(feature = "semantic-search")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_delete_cut_short_by_a_crash_is_redone_as_an_embed() {
        use crate::semantic_search::service::SemanticSearchConfig;
        let db_dir = TempDir::new().unwrap();
        let store = open_kv_semantic(&db_dir).await;
        store
            .db
            .namespace("sem_kv".to_string())
            .await
            .unwrap()
            .put(b"x".to_vec(), b"hello".to_vec())
            .await
            .unwrap();
        vector_kv::remove_queue_entry(&store.db, "sem_kv", b"x").await.unwrap();
        vector_kv::clear_vectors(&store.db, "sem_kv", b"x").await.unwrap(); // the delete, cut short by a crash

        let ctx = SemanticSearchContext::new(SemanticSearchConfig {
            embedding_service_url: "http://127.0.0.1:9".into(), // nothing listens: embeds fail and stay queued
            embedding_connect_timeout: std::time::Duration::from_millis(200),
            ..SemanticSearchConfig::default()
        });
        let store = store.with_semantic_search(ctx);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        loop {
            let entry = vector_kv::get_queue_entry(&store.db, "sem_kv", b"x").await.unwrap();
            if entry
                .as_ref()
                .is_some_and(|e| e.kind == vector_kv::QueueEntryKind::Embed && e.text == "hello")
            {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "the surviving document was never queued again: {entry:?}"
            );
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        store.shutdown_vec_index_worker().await;
    }

    /// An upsert interrupted by a crash after its queue entry leaves an entry for
    /// text the store never received. The worker rewrites it to the text the
    /// store holds, so the vectors end up matching the value, not the lost write.
    #[cfg(feature = "semantic-search")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_upsert_cut_short_by_a_crash_embeds_what_the_store_holds() {
        use crate::semantic_search::service::SemanticSearchConfig;
        let db_dir = TempDir::new().unwrap();
        let store = open_kv_semantic(&db_dir).await;
        store
            .db
            .namespace("sem_kv".to_string())
            .await
            .unwrap()
            .put(b"x".to_vec(), b"on disk".to_vec())
            .await
            .unwrap();
        vector_kv::enqueue_embed(&store.db, "sem_kv", b"x", "never written").await.unwrap(); // its value write was lost

        let ctx = SemanticSearchContext::new(SemanticSearchConfig {
            embedding_service_url: "http://127.0.0.1:9".into(), // nothing listens: embeds fail and stay queued
            embedding_connect_timeout: std::time::Duration::from_millis(200),
            ..SemanticSearchConfig::default()
        });
        let store = store.with_semantic_search(ctx);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        loop {
            let entry = vector_kv::get_queue_entry(&store.db, "sem_kv", b"x").await.unwrap();
            if entry.as_ref().is_some_and(|e| e.text == "on disk") {
                break;
            }
            assert!(std::time::Instant::now() < deadline, "the entry was never rewritten: {entry:?}");
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        store.shutdown_vec_index_worker().await;
    }

    /// A write in progress is recorded until its guard drops, however the write
    /// ends: counted per concurrent write, and removed when a request's future is
    /// dropped mid-write (a client disconnect). A leaked entry would make the
    /// worker skip its document for good.
    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn a_write_in_progress_is_forgotten_however_it_ends() {
        let in_flight = Arc::new(InFlightWrites::default());
        let a = in_flight.begin("ns", b"x", None);
        let b = in_flight.begin("ns", b"x", None);
        assert!(in_flight.contains("ns", b"x"));
        assert!(!in_flight.contains("ns", b"y") && !in_flight.contains("other", b"x"));
        drop(a);
        assert!(in_flight.contains("ns", b"x"), "the second write is still in progress");
        drop(b);
        assert!(!in_flight.contains("ns", b"x"));

        // A request cancelled mid-write: its future is dropped at an await.
        let tracker = Arc::clone(&in_flight);
        let write = async move {
            let _writing = tracker.begin("ns", b"x", None);
            std::future::pending::<()>().await;
        };
        assert!(tokio::time::timeout(std::time::Duration::from_millis(20), write).await.is_err());
        assert!(!in_flight.contains("ns", b"x"), "a cancelled write left its entry behind");
    }

    /// The race the in-progress check closes. A writer queues new text and the
    /// worker embeds and completes it before the writer's document write lands;
    /// a reindex holding an older snapshot then queues the old text. Checked
    /// while the write is in progress, the old text would match the document
    /// and be embedded, and the new document would be left with stale vectors.
    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn a_stale_entry_waits_for_a_write_in_progress() {
        use crate::doc_store::vec_index_worker::{EntryCheck, check_entry};
        let db_dir = TempDir::new().unwrap();
        let store = open_kv_semantic(&db_dir).await;
        let parent = store.db.namespace("sem_kv".to_string()).await.unwrap();
        parent.put(b"x".to_vec(), b"old".to_vec()).await.unwrap();

        let writing = store.begin_vector_write("sem_kv", b"x");
        vector_kv::enqueue_embed(&store.db, "sem_kv", b"x", "new").await.unwrap();
        let entry = vector_kv::get_queue_entry(&store.db, "sem_kv", b"x").await.unwrap().unwrap();
        vector_kv::finish_embed(&store.db, &entry, &race_vectors()).await.unwrap(); // completed early
        vector_kv::enqueue_embed(&store.db, "sem_kv", b"x", "old").await.unwrap(); // the reindex's snapshot
        let stale = vector_kv::get_queue_entry(&store.db, "sem_kv", b"x").await.unwrap().unwrap();
        assert_eq!(
            check_entry(&store.db, &store.schemas, &store.in_flight, &stale).await.unwrap(),
            EntryCheck::InProgress
        );

        parent.put(b"x".to_vec(), b"new".to_vec()).await.unwrap(); // the write lands
        drop(writing);
        assert_eq!(
            check_entry(&store.db, &store.schemas, &store.in_flight, &stale).await.unwrap(),
            EntryCheck::Rewritten
        );
        let fixed = vector_kv::get_queue_entry(&store.db, "sem_kv", b"x").await.unwrap().unwrap();
        assert_eq!(fixed.text, "new");
        assert_eq!(
            check_entry(&store.db, &store.schemas, &store.in_flight, &fixed).await.unwrap(),
            EntryCheck::Proceed
        );
    }

    /// The running worker leaves an entry alone while its document has a write
    /// in progress, and checks it once the write ends (the guard's drop wakes it).
    #[cfg(feature = "semantic-search")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_worker_skips_a_document_with_a_write_in_progress() {
        use crate::semantic_search::service::SemanticSearchConfig;
        let db_dir = TempDir::new().unwrap();
        let store = open_kv_semantic(&db_dir).await;
        store
            .db
            .namespace("sem_kv".to_string())
            .await
            .unwrap()
            .put(b"x".to_vec(), b"on disk".to_vec())
            .await
            .unwrap();

        let ctx = SemanticSearchContext::new(SemanticSearchConfig {
            embedding_service_url: "http://127.0.0.1:9".into(), // nothing listens: embeds fail and stay queued
            embedding_connect_timeout: std::time::Duration::from_millis(200),
            ..SemanticSearchConfig::default()
        });
        let store = store.with_semantic_search(ctx);
        let writing = store.begin_vector_write("sem_kv", b"x");
        vector_kv::enqueue_embed(&store.db, "sem_kv", b"x", "being written").await.unwrap();
        store.notify.as_ref().unwrap().notify_one(); // a pass while the write is in progress
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        let entry = vector_kv::get_queue_entry(&store.db, "sem_kv", b"x").await.unwrap().unwrap();
        assert_eq!(
            (entry.text.as_str(), entry.retry_count),
            ("being written", 0),
            "touched while in progress"
        );

        drop(writing); // the write ends without its document write landing
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        loop {
            let entry = vector_kv::get_queue_entry(&store.db, "sem_kv", b"x").await.unwrap();
            if entry.as_ref().is_some_and(|e| e.text == "on disk") {
                break;
            }
            assert!(std::time::Instant::now() < deadline, "never checked after the write ended: {entry:?}");
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        store.shutdown_vec_index_worker().await;
    }

    /// The dense entry of `key` in `sem_kv`, if any.
    #[cfg(feature = "semantic-search")]
    async fn dense_entry(store: &DocStore, key: &[u8]) -> Option<Vec<u8>> {
        let dense = store.db.namespace(vector_kv::dense_vectors_ns("sem_kv")).await.unwrap();
        dense.get(key.to_vec()).await.unwrap()
    }

    /// R3 through the real delete path: deleting a key whose older value is being
    /// embedded leaves no vectors once that embed completes.
    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn kv_delete_during_embed_leaves_no_vectors() {
        let db_dir = TempDir::new().unwrap();
        let store = open_kv_semantic(&db_dir).await;
        store
            .kv_put("sem_kv", &serde_json::json!("x"), &serde_json::json!("some text"))
            .await
            .unwrap();
        let in_flight = vector_kv::get_queue_entry(&store.db, "sem_kv", b"x").await.unwrap().expect("queued");

        store.kv_delete("sem_kv", "x").await.unwrap();
        vector_kv::finish_embed(&store.db, &in_flight, &race_vectors()).await.unwrap();

        assert!(!vector_kv::has_any_vector_state(&store.db, "sem_kv", b"x").await.unwrap());
    }

    /// R4: a KV value updated to empty text must stop matching its old text —
    /// the vectors and any pending embed of the older value are cleared.
    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn kv_put_empty_text_clears_old_vectors() {
        let db_dir = TempDir::new().unwrap();
        let store = open_kv_semantic(&db_dir).await;
        store
            .kv_put("sem_kv", &serde_json::json!("x"), &serde_json::json!("some text"))
            .await
            .unwrap();
        let entry = vector_kv::get_queue_entry(&store.db, "sem_kv", b"x").await.unwrap().unwrap();
        vector_kv::finish_embed(&store.db, &entry, &race_vectors()).await.unwrap();
        assert!(dense_entry(&store, b"x").await.is_some());

        store.kv_put("sem_kv", &serde_json::json!("x"), &serde_json::json!("")).await.unwrap();

        assert!(dense_entry(&store, b"x").await.is_none(), "vectors of the old text must be gone");
        let pending = vector_kv::get_queue_entry(&store.db, "sem_kv", b"x").await.unwrap().expect("tombstone");
        assert_eq!(pending.kind, vector_kv::QueueEntryKind::Clear);
    }

    /// R4, doc store: a document whose embedding field is emptied is cleared the
    /// same way, including a pending embed of its older text.
    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn doc_put_empty_embedding_text_clears_pending_embed() {
        let db_dir = TempDir::new().unwrap();
        let store = with_worker_notify(open_fresh(db_dir.path()).await);
        create_semantic_schema(&store, "sem").await;
        store.put("sem", DocId::U64(1), serde_json::json!({"title": "old title"})).await.unwrap();
        let key = DocId::U64(1).to_bytes();
        assert_eq!(
            vector_kv::get_queue_entry(&store.db, "sem", &key).await.unwrap().unwrap().kind,
            vector_kv::QueueEntryKind::Embed
        );

        store.put("sem", DocId::U64(1), serde_json::json!({})).await.unwrap();

        let pending = vector_kv::get_queue_entry(&store.db, "sem", &key).await.unwrap().expect("tombstone");
        assert_eq!(pending.kind, vector_kv::QueueEntryKind::Clear, "the stale embed must not survive");
    }

    /// Writing empty text for a document that never had vectors costs nothing:
    /// no tombstone, so bulk loads of documents without embedding text pay no
    /// extra writes.
    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn empty_text_on_unindexed_doc_writes_no_queue_entry() {
        let db_dir = TempDir::new().unwrap();
        let store = open_kv_semantic(&db_dir).await;
        store.kv_put("sem_kv", &serde_json::json!("y"), &serde_json::json!("")).await.unwrap();
        assert!(vector_kv::get_queue_entry(&store.db, "sem_kv", b"y").await.unwrap().is_none());
    }
}
