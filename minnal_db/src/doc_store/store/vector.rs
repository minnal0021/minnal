//! Vector-index queue administration, bulk (re)indexing, and cross-namespace
//! reconciliation.

use super::*;

impl DocStore {
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
        match &self.notify {
            Some(notify) => {
                vector_kv::clear_vectors(&self.db, namespace, key).await?;
                notify.notify_one();
            }
            None => {
                vector_kv::remove_queue_entry(&self.db, namespace, key).await?;
                vector_kv::delete_vector(&self.db, namespace, key).await?;
            }
        }
        Ok(())
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

    /// Reconcile vector indexes across all semantic-search-enabled namespaces.
    ///
    /// Enqueues any document that has neither a committed vector index entry nor
    /// a pending queue entry — the `put` / `kv_put` crash window, plus the
    /// `put_no_wal` vector-write window where a crash before the memtable flush
    /// drops a just-indexed vector. A cheap count short-circuit skips namespaces
    /// that are already fully covered, so a clean run is inexpensive. Returns the
    /// number of documents re-enqueued and notifies the worker if any were.
    ///
    /// This same pass runs automatically as a background task on startup (see
    /// [`with_semantic_search`]); it is also exposed on demand over the admin
    /// REST API (`POST /admin/indices/vector/reconcile`) so an operator can
    /// re-run it — e.g. if the startup pass logged a failure.
    ///
    /// [`with_semantic_search`]: DocStore::with_semantic_search
    #[cfg(feature = "semantic-search")]
    pub async fn reconcile_vector_indexes(&self) -> usize {
        let outcome = reconcile_all_vector_indexes(&self.db, &self.schema_dir, false).await;
        if outcome.reenqueued > 0
            && let Some(notify) = &self.notify
        {
            notify.notify_one();
        }
        outcome.reenqueued
    }

    /// Like [`reconcile_vector_indexes`](Self::reconcile_vector_indexes), but also
    /// re-enqueues documents whose committed vector bytes are **present yet corrupt**
    /// (fail to deserialize) — not just those missing a half.
    ///
    /// This deserializes every entry and skips the count short-circuit, so it is a
    /// full value-reading scan of every semantic-search namespace — run it in the
    /// background, not on a latency-sensitive request path. Per-namespace failures
    /// are logged here (`warn!` per namespace inside the pass, plus an `error!`
    /// summary when any failed). Returns the number of documents re-enqueued and
    /// notifies the worker if any were.
    #[cfg(feature = "semantic-search")]
    pub async fn validate_and_reconcile_vector_indexes(&self) -> usize {
        let outcome = reconcile_all_vector_indexes(&self.db, &self.schema_dir, true).await;
        if outcome.failed > 0 {
            error!(
                "validating vector-index reconcile did not fully complete: {} namespace(s) failed, {} doc(s) re-enqueued",
                outcome.failed, outcome.reenqueued
            );
        }
        if outcome.reenqueued > 0
            && let Some(notify) = &self.notify
        {
            notify.notify_one();
        }
        outcome.reenqueued
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
        let entries = vector_kv::list_queue_entries(&self.db).await?;
        let to_delete: Vec<_> = entries.into_iter().filter(|e| e.namespace == namespace).collect();
        let count = to_delete.len();
        if count == 0 {
            return Ok(0);
        }
        for entry in &to_delete {
            vector_kv::remove_queue_entry(&self.db, &entry.namespace, &entry.doc_id_bytes).await?;
        }
        Ok(count)
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
        // Also reject if a reindex is currently running.
        if let Some(ns_id) = schema.ns_id {
            let path = vec_reindex_path(&self.db_path, ns_id);
            if let Some(c) = read_vec_reindex(&path)
                && c.status == "running"
            {
                return Err(DocStoreError::VecReindexInProgress {
                    namespace: namespace.to_owned(),
                });
            }
        }
        Ok(())
    }

    /// Return the persisted vector-index reindex record for `namespace`, or
    /// `None` when no reindex has been started for that namespace.
    pub fn vec_reindex_progress(&self, namespace: &str) -> Option<VecReindexProgress> {
        let schema = self.load_schema(namespace).ok()?;
        let ns_id = schema.ns_id?;
        read_vec_reindex(&vec_reindex_path(&self.db_path, ns_id))
    }

    /// Re-enqueue every document in `namespace` for vector indexing.
    ///
    /// The operation is a point-in-time snapshot of the doc store:
    ///
    /// 1. Rejects with [`DocStoreError::VecReindexInProgress`] when a
    ///    previous `index_all` reindex is still running.
    /// 2. All **exhausted** queue entries for the namespace (those whose
    ///    `retry_count` has reached `max_retries`) are removed first so they
    ///    don't block processing.
    /// 3. Every document whose embedding fields produce non-empty text is
    ///    enqueued with `retry_count = 0`.  Documents already in the queue
    ///    with a lower retry count are overwritten (natural deduplication
    ///    — only the latest text is kept).
    /// 4. A reindex record is written to
    ///    `{db_path}/index/{ns_id}/vector_reindex.json` so the progress API
    ///    can track this reindex across server restarts.
    /// 5. The vector-index worker is notified to start processing immediately.
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

        // 409 guard: reject concurrent reindexs.
        let reindex_path = vec_reindex_path(&self.db_path, ns_id);
        if let Some(c) = read_vec_reindex(&reindex_path)
            && c.status == "running"
        {
            return Err(DocStoreError::VecReindexInProgress {
                namespace: namespace.to_owned(),
            });
        }

        // Write "running" reindex record immediately so the guard works even
        // if the process crashes before we finish enqueueing.
        let started_at_ms = now_ms();
        std::fs::create_dir_all(reindex_path.parent().unwrap())?;
        write_vec_reindex(
            &reindex_path,
            &VecReindexProgress {
                status: "running".to_owned(),
                started_at_ms,
                completed_at_ms: None,
                total_enqueued: 0,
                exhausted_cleared: 0,
                error: None,
            },
        );

        let max_retries = self.vector_index_config.max_retries;

        info!(
            "index_all: namespace='{}' scanning pending queue for exhausted entries (max_retries={})",
            namespace, max_retries
        );

        // Collect exhausted entries to clear.
        let all_queue = vector_kv::list_queue_entries(&self.db).await?;
        let total_queue = all_queue.len();
        let exhausted: Vec<_> = all_queue
            .into_iter()
            .filter(|e| e.namespace == namespace && e.retry_count >= max_retries)
            .collect();
        let exhausted_cleared = exhausted.len();
        info!(
            "index_all: namespace='{}' queue scan done — total_queue={} exhausted_to_clear={}",
            namespace, total_queue, exhausted_cleared
        );

        // Scan all documents in the namespace.
        info!("index_all: namespace='{}' scanning all documents", namespace);
        let ns = self.db.namespace(namespace.to_owned()).await?;
        let all_docs = ns.iter().await?;
        let total_docs = all_docs.len();
        info!("index_all: namespace='{}' document scan done — total_docs={}", namespace, total_docs);

        // Clear exhausted entries (durable single-op deletes), then enqueue every
        // document whose embedding fields produce non-empty text. Enqueues are
        // WAL-backed: with the vector index itself written no-WAL, the queue is the
        // durable source of truth for what still needs (re-)indexing, so every
        // enqueue must survive a crash.
        let mut enqueued = 0usize;
        let mut skipped_empty_text = 0usize;
        let commit_result: Result<(), DocStoreError> = async {
            for entry in &exhausted {
                vector_kv::remove_queue_entry(&self.db, &entry.namespace, &entry.doc_id_bytes).await?;
            }
            for (key, value) in &all_docs {
                let doc = match serde_json::from_slice::<serde_json::Value>(value) {
                    Ok(d) => d,
                    Err(_) => continue,
                };
                let text = build_embedding_text(&doc, &schema.embedding_fields);
                if !text.is_empty() {
                    vector_kv::enqueue_embed(&self.db, namespace, key, &text).await?;
                    enqueued += 1;
                } else {
                    skipped_empty_text += 1;
                }
            }
            Ok(())
        }
        .await;

        info!(
            "index_all: namespace='{}' enqueue done — enqueued={} skipped_empty_text={}",
            namespace, enqueued, skipped_empty_text
        );

        if let Err(e) = commit_result {
            write_vec_reindex(
                &reindex_path,
                &VecReindexProgress {
                    status: "failed".to_owned(),
                    started_at_ms,
                    completed_at_ms: Some(now_ms()),
                    total_enqueued: 0,
                    exhausted_cleared,
                    error: Some(e.to_string()),
                },
            );
            return Err(e);
        }

        // Update reindex record with the final enqueued count; status stays
        // "running" until the worker finishes (tracked via queue depth).
        write_vec_reindex(
            &reindex_path,
            &VecReindexProgress {
                status: "running".to_owned(),
                started_at_ms,
                completed_at_ms: None,
                total_enqueued: enqueued,
                exhausted_cleared,
                error: None,
            },
        );

        info!("index_all: namespace='{}' batch committed", namespace);

        if let Some(notify) = &self.notify {
            notify.notify_one();
            info!("index_all: namespace='{}' worker notified", namespace);
        }

        info!(
            "index_all: namespace='{}' enqueued={} exhausted_cleared={}",
            namespace, enqueued, exhausted_cleared,
        );

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
        if let Some(ns_id) = schema.ns_id {
            let path = vec_reindex_path(&self.db_path, ns_id);
            if let Some(c) = read_vec_reindex(&path)
                && c.status == "running"
            {
                return Err(DocStoreError::VecReindexInProgress {
                    namespace: namespace.to_owned(),
                });
            }
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

        let reindex_path = vec_reindex_path(&self.db_path, ns_id);
        if let Some(c) = read_vec_reindex(&reindex_path)
            && c.status == "running"
        {
            return Err(DocStoreError::VecReindexInProgress {
                namespace: namespace.to_owned(),
            });
        }

        let started_at_ms = now_ms();
        std::fs::create_dir_all(reindex_path.parent().unwrap())?;
        write_vec_reindex(
            &reindex_path,
            &VecReindexProgress {
                status: "running".to_owned(),
                started_at_ms,
                completed_at_ms: None,
                total_enqueued: 0,
                exhausted_cleared: 0,
                error: None,
            },
        );

        let max_retries = self.vector_index_config.max_retries;

        info!(
            "kv_index_all: namespace='{}' scanning pending queue for exhausted entries (max_retries={})",
            namespace, max_retries
        );
        let all_queue = vector_kv::list_queue_entries(&self.db).await?;
        let exhausted: Vec<_> = all_queue
            .into_iter()
            .filter(|e| e.namespace == namespace && e.retry_count >= max_retries)
            .collect();
        let exhausted_cleared = exhausted.len();
        info!(
            "kv_index_all: namespace='{}' queue scan done — exhausted_to_clear={}",
            namespace, exhausted_cleared
        );

        info!("kv_index_all: namespace='{}' scanning all KV entries", namespace);
        let ns = self.db.namespace(namespace.to_owned()).await?;
        let all_entries = ns.iter().await?;
        info!("kv_index_all: namespace='{}' scan done — total_entries={}", namespace, all_entries.len());

        // Clear exhausted entries (durable single-op deletes), then enqueue every
        // entry with non-empty text. Enqueues are WAL-backed: the queue is the
        // durable source of truth for the no-WAL vector index — see index_all for
        // the rationale.
        let mut enqueued = 0usize;
        let mut skipped_empty_text = 0usize;
        let commit_result: Result<(), DocStoreError> = async {
            for entry in &exhausted {
                vector_kv::remove_queue_entry(&self.db, &entry.namespace, &entry.doc_id_bytes).await?;
            }
            for (key, value_bytes) in &all_entries {
                match std::str::from_utf8(value_bytes) {
                    Ok(text) if !text.is_empty() => {
                        vector_kv::enqueue_embed(&self.db, namespace, key, text).await?;
                        enqueued += 1;
                    }
                    _ => {
                        skipped_empty_text += 1;
                    }
                }
            }
            Ok(())
        }
        .await;

        info!(
            "kv_index_all: namespace='{}' enqueue done — enqueued={} skipped_empty_text={}",
            namespace, enqueued, skipped_empty_text
        );

        if let Err(e) = commit_result {
            write_vec_reindex(
                &reindex_path,
                &VecReindexProgress {
                    status: "failed".to_owned(),
                    started_at_ms,
                    completed_at_ms: Some(now_ms()),
                    total_enqueued: 0,
                    exhausted_cleared,
                    error: Some(e.to_string()),
                },
            );
            return Err(e);
        }

        write_vec_reindex(
            &reindex_path,
            &VecReindexProgress {
                status: "running".to_owned(),
                started_at_ms,
                completed_at_ms: None,
                total_enqueued: enqueued,
                exhausted_cleared,
                error: None,
            },
        );

        if let Some(notify) = &self.notify {
            notify.notify_one();
            info!("kv_index_all: namespace='{}' worker notified", namespace);
        }

        info!(
            "kv_index_all: namespace='{}' enqueued={} exhausted_cleared={}",
            namespace, enqueued, exhausted_cleared
        );

        Ok(ReindexStats { exhausted_cleared, enqueued })
    }
}

// ── Vector-index reconciliation ────────────────────────────────────────────────

/// Reconcile vector indexes across every semantic-search-enabled namespace.
///
/// For each such namespace, enqueue any document that has **neither** a
/// committed vector index entry **nor** a pending queue entry — i.e. the
/// `put` / `kv_put` crash window (document durably written, embed enqueue lost).
/// This is the vector-index analogue of how field indices self-heal on startup
/// via WAL replay: the difference is the work is routed into the async embedding
/// queue rather than rebuilt inline, and the [`VecIndexWorker`] drains it when
/// the embedding service is available.
///
/// Returns the total number of documents re-enqueued.  Errors on individual
/// namespaces are logged and skipped so one bad namespace cannot abort the rest.
///
/// [`VecIndexWorker`]: crate::doc_store::vec_index_worker::VecIndexWorker
/// Outcome of a [`reconcile_all_vector_indexes`] pass.
#[cfg(feature = "semantic-search")]
/// `pub(super)` so the parent's startup path can read the outcome; it was a
/// file-private type before the split.
pub(super) struct ReconcileOutcome {
    /// Documents re-enqueued for embedding across all namespaces.
    pub(super) reenqueued: usize,
    /// Number of namespaces whose reconciliation failed (each is also logged
    /// individually via `warn!`).  Non-zero means the pass did not fully
    /// complete and should be re-run.
    pub(super) failed: usize,
}

/// Reconcile every semantic-search namespace's vector index.
///
/// With `check_bytes == false` (the cheap, default pass used at startup and by the
/// presence-only reconcile) a document is "indexed" when both companion halves are
/// **present** ([`vector_kv::has_complete_vector_index`]), and a count short-circuit
/// skips namespaces already fully covered. With `check_bytes == true` (the on-demand
/// *validating* pass) it instead deserializes each entry ([`vector_kv::has_valid_vector_index`])
/// to catch present-but-corrupt vectors, and skips the count short-circuit (corruption
/// is not count-detectable) — a full value-reading scan, hence run in the background.
#[cfg(feature = "semantic-search")]
/// `pub(super)`: called from the parent module's construction path.
pub(super) async fn reconcile_all_vector_indexes(db: &AsyncDb, schema_dir: &Path, check_bytes: bool) -> ReconcileOutcome {
    // Scan the pending queue once and count entries per namespace, so each
    // namespace's cheap short-circuit can test `pending == 0` without re-scanning.
    let pending_by_ns = {
        let mut map: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for entry in vector_kv::list_queue_entries(db).await.unwrap_or_default() {
            *map.entry(entry.namespace).or_default() += 1;
        }
        map
    };
    let pending_for = |ns: &str| pending_by_ns.get(ns).copied().unwrap_or(0);

    let mut reenqueued = 0usize;
    let mut failed = 0usize;

    for schema in load_all_schemas_from(schema_dir).unwrap_or_default() {
        if !schema.is_semantic_search_enabled() {
            continue;
        }
        match reconcile_doc_namespace_vectors(db, &schema, pending_for(&schema.namespace), check_bytes).await {
            Ok(n) => reenqueued += n,
            Err(e) => {
                failed += 1;
                warn!("vec reconcile: doc namespace '{}' failed: {e}", schema.namespace);
            }
        }
    }

    for schema in load_all_kv_schemas_from(schema_dir).unwrap_or_default() {
        if !schema.is_semantic_search_enabled() {
            continue;
        }
        match reconcile_kv_namespace_vectors(db, &schema, pending_for(&schema.namespace), check_bytes).await {
            Ok(n) => reenqueued += n,
            Err(e) => {
                failed += 1;
                warn!("vec reconcile: kv namespace '{}' failed: {e}", schema.namespace);
            }
        }
    }

    ReconcileOutcome { reenqueued, failed }
}

/// Count keys in `namespace` without reading their values (LSM-only scan).
#[cfg(feature = "semantic-search")]
async fn count_keys(db: &AsyncDb, namespace: &str) -> Result<usize, DocStoreError> {
    let ns = db.namespace(namespace.to_owned()).await?;
    Ok(ns.keys().await?.len())
}

/// Count keys in a vector-index companion namespace (sparse-meta or dense)
/// without reading values.  Returns 0 if the companion namespace does not exist
/// yet.
#[cfg(feature = "semantic-search")]
async fn count_companion(db: &AsyncDb, companion_ns: String) -> usize {
    match db.namespace(companion_ns).await {
        Ok(ns) => ns.keys().await.map(|k| k.len()).unwrap_or(0),
        Err(_) => 0,
    }
}

/// Cheap short-circuit shared by the doc and KV reconcilers: when nothing is
/// queued for the namespace and every key already has a **complete** committed
/// vector index, there is nothing to reconcile and the full per-doc scan can be
/// skipped.
///
/// A complete index requires **both** a sparse-meta and a dense entry per key
/// (see [`vector_kv::has_complete_vector_index`]), so the short-circuit must
/// require both companion counts to reach `key_count` — otherwise a namespace
/// where every key has sparse-meta but some lost the no-WAL dense write would be
/// skipped despite being partially indexed.
///
/// Sound because the delete ordering guarantees vector-index, meta, and queue
/// entries never outlive their document — so the meta set, dense set, and queue
/// set are all subsets of the live keys. With no orphans, both counts reaching
/// `key_count` (and `pending == 0`) implies every live key is fully indexed.
/// Namespaces with empty-embedding-text documents simply fall through to the
/// full scan (which then enqueues nothing) — correctness is preserved, only the
/// optimisation is skipped. All counts are LSM-only key scans (no value-log
/// reads), so a clean boot avoids the expensive value-loading `iter`.
#[cfg(feature = "semantic-search")]
async fn nothing_to_reconcile(db: &AsyncDb, namespace: &str, pending_for_ns: usize) -> bool {
    if pending_for_ns != 0 {
        return false;
    }
    let Ok(key_count) = count_keys(db, namespace).await else {
        return false;
    };
    count_companion(db, vector_kv::sparse_vectors_meta_ns(namespace)).await >= key_count
        && count_companion(db, vector_kv::dense_vectors_ns(namespace)).await >= key_count
}

/// Reconcile one document-store namespace.  See [`reconcile_all_vector_indexes`].
#[cfg(feature = "semantic-search")]
async fn reconcile_doc_namespace_vectors(
    db: &AsyncDb,
    schema: &DocStoreSchema,
    pending_for_ns: usize,
    check_bytes: bool,
) -> Result<usize, DocStoreError> {
    let namespace = &schema.namespace;
    // The count short-circuit only sees presence, so it cannot detect corrupt-but-
    // present entries — skip it for the validating pass and scan every document.
    if !check_bytes && nothing_to_reconcile(db, namespace, pending_for_ns).await {
        return Ok(0);
    }

    let ns = db.namespace(namespace.clone()).await?;
    let all_docs = ns.iter().await?;

    let mut enqueued = 0usize;
    for (key, value) in &all_docs {
        if vector_kv::get_queue_entry(db, namespace, key).await?.is_some() {
            continue;
        }
        let indexed = if check_bytes {
            vector_kv::has_valid_vector_index(db, namespace, key).await?
        } else {
            vector_kv::has_complete_vector_index(db, namespace, key).await?
        };
        if indexed {
            continue;
        }
        let Ok(doc) = serde_json::from_slice::<serde_json::Value>(value) else {
            continue;
        };
        let text = build_embedding_text(&doc, &schema.embedding_fields);
        if !text.is_empty() {
            vector_kv::enqueue_embed(db, namespace, key, &text).await?;
            enqueued += 1;
        }
    }

    if enqueued > 0 {
        info!(
            "vec reconcile: doc namespace '{}' re-enqueued {} missing document(s)",
            namespace, enqueued
        );
    }
    Ok(enqueued)
}

/// Reconcile one KV-store namespace.  See [`reconcile_all_vector_indexes`].
#[cfg(feature = "semantic-search")]
async fn reconcile_kv_namespace_vectors(
    db: &AsyncDb,
    schema: &KvStoreSchema,
    pending_for_ns: usize,
    check_bytes: bool,
) -> Result<usize, DocStoreError> {
    let namespace = &schema.namespace;
    if !check_bytes && nothing_to_reconcile(db, namespace, pending_for_ns).await {
        return Ok(0);
    }

    let ns = db.namespace(namespace.clone()).await?;
    let all_entries = ns.iter().await?;

    let mut enqueued = 0usize;
    for (key, value_bytes) in &all_entries {
        if vector_kv::get_queue_entry(db, namespace, key).await?.is_some() {
            continue;
        }
        let indexed = if check_bytes {
            vector_kv::has_valid_vector_index(db, namespace, key).await?
        } else {
            vector_kv::has_complete_vector_index(db, namespace, key).await?
        };
        if indexed {
            continue;
        }
        if let Ok(text) = std::str::from_utf8(value_bytes)
            && !text.is_empty()
        {
            vector_kv::enqueue_embed(db, namespace, key, text).await?;
            enqueued += 1;
        }
    }

    if enqueued > 0 {
        info!("vec reconcile: kv namespace '{}' re-enqueued {} missing entry(ies)", namespace, enqueued);
    }
    Ok(enqueued)
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
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        assert_eq!(store.vector_index_max_retries(), 5);
    }

    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn test_with_vector_index_config_sets_max_retries() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path())
            .await
            .with_vector_index_config(VectorIndexConfig {
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
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        assert_eq!(store.pending_vector_index_count().await, 0);
    }

    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn test_list_pending_queue_entries_empty() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        assert!(store.list_queue_entries().await.is_empty());
    }

    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn test_pending_queue_count_and_list_after_enqueue() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;

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
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;

        vector_kv::enqueue_embed(&store.db, "ns", b"doc1", "text").await.unwrap();

        assert_eq!(store.pending_vector_index_count().await, 1);

        store.delete_queue_entry("ns", b"doc1").await.unwrap();

        assert_eq!(store.pending_vector_index_count().await, 0);
    }

    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn test_delete_pending_queue_entry_noop_for_missing() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;

        // Deleting a non-existent entry should not error.
        store.delete_queue_entry("ns", b"ghost").await.unwrap();
        assert_eq!(store.pending_vector_index_count().await, 0);
    }

    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn test_delete_pending_queue_entry_only_removes_target() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;

        vector_kv::enqueue_embed(&store.db, "ns", b"doc1", "t1").await.unwrap();
        vector_kv::enqueue_embed(&store.db, "ns", b"doc2", "t2").await.unwrap();

        store.delete_queue_entry("ns", b"doc1").await.unwrap();

        let entries = store.list_queue_entries().await;
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].doc_id_bytes, b"doc2");
    }

    // ── Vector-index reconciliation ─────────────────────────────────────────

    /// Without the `semantic-search` feature, creating a semantic-search-enabled
    /// store must be rejected at runtime rather than silently ignored.
    #[cfg(not(feature = "semantic-search"))]
    #[tokio::test]
    async fn test_semantic_create_rejected_without_feature() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
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
        };
        store.create(schema).await.unwrap();
    }

    /// Helper: commit a *complete* vector index (sparse meta + dense) for a doc,
    /// simulating a fully-indexed document. Reconciliation skips only documents
    /// that have both halves, so tests asserting "already indexed → skip" must
    /// write both.
    #[cfg(feature = "semantic-search")]
    async fn commit_complete_index(store: &DocStore, namespace: &str, key: &[u8]) {
        use crate::semantic_search::index::vector_index::QuantisationStyle;
        let sparse = crate::semantic_search::VectorIndex::new(1, QuantisationStyle::SingleBit, 0.4, 0.0, 0.02, vec![]);
        let dense = crate::semantic_search::VectorIndex::new(1, QuantisationStyle::MultiBit { number_of_bits: 8 }, 0.4, 0.0, 0.02, vec![]);
        crate::vector_kv::upsert_vectors(&store.db, namespace, key, &[sparse, dense])
            .await
            .unwrap();
    }

    /// Documents written through the crash window (doc present, but no queue
    /// entry and no vector index) must be re-enqueued by reconciliation.
    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn test_reconcile_enqueues_missing_docs() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        create_semantic_schema(&store, "sem").await;

        // No SemanticSearchContext is attached, so put() does NOT enqueue —
        // this is exactly the crash window (doc durable, embed marker lost).
        for i in 1u64..=3 {
            store
                .put("sem", DocId::U64(i), serde_json::json!({"title": format!("doc {i}")}))
                .await
                .unwrap();
        }
        assert_eq!(store.pending_vector_index_count().await, 0, "no enqueue happened on write");

        let reconciled = store.reconcile_vector_indexes().await;
        assert_eq!(reconciled, 3, "all three missing docs must be re-enqueued");
        assert_eq!(store.pending_vector_index_count().await, 3);
    }

    /// A single-document vector reindex enqueues exactly that document, reports
    /// `NotFound` for a missing id, and rejects a non-semantic namespace.
    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn test_reindex_doc_vector_enqueues_single_doc() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
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
            })
            .await
            .unwrap();
        store.put("plain", DocId::U64(1), serde_json::json!({"title": "x"})).await.unwrap();
        assert!(matches!(
            store.reindex_doc_vector("plain", DocId::U64(1)).await,
            Err(DocStoreError::SemanticSearchNotEnabled { .. })
        ));
    }

    /// Reconciliation must skip documents that already have a pending queue
    /// entry, and must be idempotent on a second run.
    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn test_reconcile_skips_queued_and_is_idempotent() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        create_semantic_schema(&store, "sem").await;

        store.put("sem", DocId::U64(1), serde_json::json!({"title": "hello"})).await.unwrap();
        store.put("sem", DocId::U64(2), serde_json::json!({"title": "world"})).await.unwrap();

        // First pass enqueues both missing docs.
        assert_eq!(store.reconcile_vector_indexes().await, 2);
        assert_eq!(store.pending_vector_index_count().await, 2);

        // Second pass is a no-op: both docs are already queued.
        assert_eq!(store.reconcile_vector_indexes().await, 0);
        assert_eq!(store.pending_vector_index_count().await, 2);
    }

    /// Reconciliation must skip documents that already have a committed vector
    /// index entry.
    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn test_reconcile_skips_already_indexed() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        create_semantic_schema(&store, "sem").await;

        let key = DocId::U64(1).to_bytes();
        store.put("sem", DocId::U64(1), serde_json::json!({"title": "indexed"})).await.unwrap();

        // Simulate a complete committed vector index (sparse meta + dense) for doc 1.
        commit_complete_index(&store, "sem", &key).await;

        // Doc 1 is already indexed → reconciliation must not enqueue it.
        assert_eq!(store.reconcile_vector_indexes().await, 0);
        assert_eq!(store.pending_vector_index_count().await, 0);
    }

    /// The validating reconcile must NOT re-enqueue a document whose committed
    /// vectors are present *and* deserialize (no false positives on healthy docs).
    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn test_validating_reconcile_skips_valid_complete_index() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        create_semantic_schema(&store, "sem").await;

        let key = DocId::U64(1).to_bytes();
        store.put("sem", DocId::U64(1), serde_json::json!({"title": "indexed"})).await.unwrap();
        commit_complete_index(&store, "sem", &key).await;

        assert_eq!(
            store.validate_and_reconcile_vector_indexes().await,
            0,
            "valid index must not be re-enqueued"
        );
        assert_eq!(store.pending_vector_index_count().await, 0);
    }

    /// A document whose committed vector bytes are *present but corrupt* is skipped
    /// by the presence-only reconcile (both halves exist) yet re-enqueued by the
    /// validating reconcile (the bytes fail to deserialize). Covers both the dense
    /// and a sparse composite entry.
    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn test_validating_reconcile_reenqueues_present_but_corrupt() {
        // Corrupt dense.
        {
            let db_dir = TempDir::new().unwrap();
            let schema_dir = TempDir::new().unwrap();
            let store = open_fresh(db_dir.path(), schema_dir.path()).await;
            create_semantic_schema(&store, "sem").await;
            let key = DocId::U64(1).to_bytes();
            store.put("sem", DocId::U64(1), serde_json::json!({"title": "indexed"})).await.unwrap();
            commit_complete_index(&store, "sem", &key).await;

            // Overwrite the dense entry with bytes that are present but undeserializable.
            let dense_ns = store.db.namespace(crate::vector_kv::dense_vectors_ns("sem")).await.unwrap();
            dense_ns.put(key.clone(), b"not valid rkyv bytes".to_vec()).await.unwrap();

            assert_eq!(store.reconcile_vector_indexes().await, 0, "presence check sees both halves → skips");
            assert_eq!(store.pending_vector_index_count().await, 0);
            assert_eq!(
                store.validate_and_reconcile_vector_indexes().await,
                1,
                "corrupt dense must be re-enqueued"
            );
            assert_eq!(store.pending_vector_index_count().await, 1);
        }

        // Corrupt a sparse composite entry (commit_complete_index assigns cluster 1).
        {
            let db_dir = TempDir::new().unwrap();
            let schema_dir = TempDir::new().unwrap();
            let store = open_fresh(db_dir.path(), schema_dir.path()).await;
            create_semantic_schema(&store, "sem").await;
            let key = DocId::U64(1).to_bytes();
            store.put("sem", DocId::U64(1), serde_json::json!({"title": "indexed"})).await.unwrap();
            commit_complete_index(&store, "sem", &key).await;

            let sparse_ns = store.db.namespace(crate::vector_kv::sparse_vectors_ns("sem")).await.unwrap();
            sparse_ns
                .put(crate::semantic_search::composite_key::encode(1, &key), b"garbage".to_vec())
                .await
                .unwrap();

            assert_eq!(store.reconcile_vector_indexes().await, 0, "presence check skips");
            assert_eq!(
                store.validate_and_reconcile_vector_indexes().await,
                1,
                "corrupt sparse must be re-enqueued"
            );
            assert_eq!(store.pending_vector_index_count().await, 1);
        }
    }

    /// KV-store namespaces are reconciled too: a string value written without a
    /// queue entry or vector index must be re-enqueued.
    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn test_reconcile_kv_namespace() {
        use crate::doc_store::schema::{KvKeyType, KvStoreSchema, KvValueType};

        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;

        let schema = KvStoreSchema {
            store_type: StoreType::Kv,
            namespace: "kv".to_owned(),
            ns_id: None,
            key_type: KvKeyType::Str,
            value_type: KvValueType::Str,
            semantic_search_enabled: true,
        };
        store.create_kv(schema).await.unwrap();

        store
            .kv_put("kv", &serde_json::json!("k1"), &serde_json::json!("some text"))
            .await
            .unwrap();
        assert_eq!(store.pending_vector_index_count().await, 0, "no enqueue on write (no ctx)");

        assert_eq!(store.reconcile_vector_indexes().await, 1);
        assert_eq!(store.pending_vector_index_count().await, 1);
    }

    /// The count short-circuit must NOT fire when a namespace is only partially
    /// indexed: one indexed doc + one missing doc means `indexed (1) < keys (2)`,
    /// so the full scan runs and the missing doc is enqueued.
    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn test_reconcile_partial_index_not_short_circuited() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        create_semantic_schema(&store, "sem").await;

        // Doc 1 is indexed; doc 2 is written but missing (the crash window).
        store.put("sem", DocId::U64(1), serde_json::json!({"title": "one"})).await.unwrap();
        store.put("sem", DocId::U64(2), serde_json::json!({"title": "two"})).await.unwrap();
        commit_complete_index(&store, "sem", &DocId::U64(1).to_bytes()).await;

        // indexed (1) < keys (2) → no short-circuit → doc 2 enqueued, doc 1 skipped.
        assert_eq!(store.reconcile_vector_indexes().await, 1);
        let pending = store.list_queue_entries().await;
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].doc_id_bytes, DocId::U64(2).to_bytes());
    }

    /// A *partially* committed index counts as not-indexed: a doc with only the
    /// sparse side (dense write lost) and a doc with only the dense side (sparse
    /// write lost) must both be re-enqueued so the re-embed regenerates the
    /// missing half. Guards the `meta AND dense` tightening in
    /// [`vector_kv::has_complete_vector_index`] — under the old OR semantics
    /// either of these would have been skipped as "indexed".
    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn test_reconcile_partial_index_is_reenqueued() {
        use crate::semantic_search::index::vector_index::QuantisationStyle;

        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        create_semantic_schema(&store, "sem").await;

        store
            .put("sem", DocId::U64(1), serde_json::json!({"title": "sparse only"}))
            .await
            .unwrap();
        store.put("sem", DocId::U64(2), serde_json::json!({"title": "dense only"})).await.unwrap();

        // Doc 1: only the sparse meta committed — the dense write was lost.
        let key1 = DocId::U64(1).to_bytes();
        let sparse = crate::semantic_search::VectorIndex::new(1, QuantisationStyle::SingleBit, 0.4, 0.0, 0.02, vec![]);
        crate::vector_kv::upsert_vectors(&store.db, "sem", &key1, &[sparse]).await.unwrap();

        // Doc 2: only the dense entry committed — the sparse write was lost.
        let key2 = DocId::U64(2).to_bytes();
        let dense = crate::semantic_search::VectorIndex::new(1, QuantisationStyle::MultiBit { number_of_bits: 8 }, 0.4, 0.0, 0.02, vec![]);
        crate::vector_kv::upsert_vectors(&store.db, "sem", &key2, &[dense]).await.unwrap();

        // Both indexes are incomplete → both re-enqueued.
        assert_eq!(store.reconcile_vector_indexes().await, 2);
        let mut got: Vec<Vec<u8>> = store.list_queue_entries().await.into_iter().map(|e| e.doc_id_bytes).collect();
        got.sort();
        let mut want = vec![DocId::U64(1).to_bytes(), DocId::U64(2).to_bytes()];
        want.sort();
        assert_eq!(got, want);
    }

    /// Regression for the in-flight enqueue race: a live write that enqueues a
    /// document *while* the reconciliation pass is mid-scan must not produce a
    /// duplicate queue entry. The embed queue is keyed by `(namespace, doc_id)`,
    /// so a concurrent reconciliation re-enqueue and an in-flight write collapse
    /// to a single entry. This is the invariant that keeps the queue bounded by
    /// the number of distinct docs — without it, a racing writer could grow the
    /// queue unboundedly and feed the worker an endless re-index loop.
    ///
    /// The assertion holds for every interleaving (idempotent-by-key enqueue),
    /// so the test is deterministic despite running the two paths concurrently.
    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn test_reconcile_inflight_enqueue_does_not_duplicate() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        create_semantic_schema(&store, "sem").await;

        // Five docs written through the crash window (durable, but no enqueue).
        for i in 1u64..=5 {
            store
                .put("sem", DocId::U64(i), serde_json::json!({"title": format!("doc {i}")}))
                .await
                .unwrap();
        }

        // Race a live enqueue for every doc against the reconciliation pass: each
        // direct enqueue stands in for a write that lands while reconciliation is
        // scanning the same key. Both paths target the same queue key per doc.
        let keys: Vec<Vec<u8>> = (1u64..=5).map(|i| DocId::U64(i).to_bytes()).collect();
        let live = async {
            for (i, key) in keys.iter().enumerate() {
                crate::vector_kv::enqueue_embed(&store.db, "sem", key, &format!("live {}", i + 1))
                    .await
                    .unwrap();
            }
        };
        let (_, _reconciled) = tokio::join!(live, store.reconcile_vector_indexes());

        // Idempotent by doc-id: exactly one entry per doc, never doubled —
        // regardless of how the live enqueues and the reconcile pass interleaved.
        assert_eq!(
            store.pending_vector_index_count().await,
            5,
            "concurrent enqueue + reconcile must not duplicate queue entries"
        );

        // The queue is converging, not looping: a follow-up reconcile is a clean
        // no-op because every doc now has a pending entry.
        assert_eq!(store.reconcile_vector_indexes().await, 0);
        assert_eq!(store.pending_vector_index_count().await, 5);
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
    async fn open_kv_semantic(db_dir: &TempDir, schema_dir: &TempDir) -> DocStore {
        let store = with_worker_notify(open_fresh(db_dir.path(), schema_dir.path()).await);
        let mut schema = make_kv_schema("sem_kv", KvKeyType::Str, KvValueType::Str);
        schema.semantic_search_enabled = true;
        store.create_kv(schema).await.unwrap();
        store
    }

    /// R3 through the real delete path: deleting a key whose older value is being
    /// embedded leaves no vectors once that embed completes.
    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn kv_delete_during_embed_leaves_no_vectors() {
        let (db_dir, schema_dir) = (TempDir::new().unwrap(), TempDir::new().unwrap());
        let store = open_kv_semantic(&db_dir, &schema_dir).await;
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
        let (db_dir, schema_dir) = (TempDir::new().unwrap(), TempDir::new().unwrap());
        let store = open_kv_semantic(&db_dir, &schema_dir).await;
        store
            .kv_put("sem_kv", &serde_json::json!("x"), &serde_json::json!("some text"))
            .await
            .unwrap();
        let entry = vector_kv::get_queue_entry(&store.db, "sem_kv", b"x").await.unwrap().unwrap();
        vector_kv::finish_embed(&store.db, &entry, &race_vectors()).await.unwrap();
        assert!(vector_kv::has_complete_vector_index(&store.db, "sem_kv", b"x").await.unwrap());

        store.kv_put("sem_kv", &serde_json::json!("x"), &serde_json::json!("")).await.unwrap();

        assert!(
            !vector_kv::has_complete_vector_index(&store.db, "sem_kv", b"x").await.unwrap(),
            "vectors of the old text must be gone"
        );
        let pending = vector_kv::get_queue_entry(&store.db, "sem_kv", b"x").await.unwrap().expect("tombstone");
        assert_eq!(pending.kind, vector_kv::QueueEntryKind::Clear);
    }

    /// R4, doc store: a document whose embedding field is emptied is cleared the
    /// same way, including a pending embed of its older text.
    #[cfg(feature = "semantic-search")]
    #[tokio::test]
    async fn doc_put_empty_embedding_text_clears_pending_embed() {
        let (db_dir, schema_dir) = (TempDir::new().unwrap(), TempDir::new().unwrap());
        let store = with_worker_notify(open_fresh(db_dir.path(), schema_dir.path()).await);
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
        let (db_dir, schema_dir) = (TempDir::new().unwrap(), TempDir::new().unwrap());
        let store = open_kv_semantic(&db_dir, &schema_dir).await;
        store.kv_put("sem_kv", &serde_json::json!("y"), &serde_json::json!("")).await.unwrap();
        assert!(vector_kv::get_queue_entry(&store.db, "sem_kv", b"y").await.unwrap().is_none());
    }
}
