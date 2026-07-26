//! Vector-index queue administration, bulk (re)indexing, and cross-namespace
//! reconciliation.

use super::*;

impl DocStore {
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
        let entry = match vector_kv::get_queue_entry(&self.db, namespace, doc_id_bytes).await? {
            None => return Ok(None),
            Some(e) => e,
        };
        vector_kv::enqueue_embed(&self.db, namespace, doc_id_bytes, &entry.text).await?;
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
            vector_kv::enqueue_embed(&self.db, &entry.namespace, &entry.doc_id_bytes, &entry.text).await?;
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
