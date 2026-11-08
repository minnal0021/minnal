//! Re-queueing a store's documents for embedding (`index_all`, `kv_index_all`),
//! its progress record, and the claim that keeps one reindex per store.

use crate::doc_store::*;
#[cfg(feature = "semantic-search")]
use crate::namespace_lifecycle::BuildStatus;

impl DocStore {
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
    /// A `Running` record is brought up to date as it is read: it becomes
    /// `Complete` once every document was enqueued and none of the namespace's
    /// work is left in the queue, and `Failed` if its enqueue never finished and
    /// no reindex of the namespace is enqueueing in this process (a crash cut it
    /// short).
    #[cfg(feature = "semantic-search")]
    pub async fn vec_reindex_progress(&self, namespace: &str) -> Option<VecReindexProgress> {
        let ns_id = match self.view_schema(namespace) {
            Ok(schema) => schema.ns_id?,
            Err(_) => self.view_kv_schema(namespace).ok()?.ns_id?,
        };
        let key = JobKey::vector_reindex(ns_id);
        let mut record: VecReindexProgress = self.maintenance.get(key).ok()??;
        if record.status != BuildStatus::Running || self.vec_reindexing.lock().contains(namespace) {
            return Some(record);
        }
        if !record.enqueue_done {
            record.status = BuildStatus::Failed;
            record.completed_at_ms = Some(now_ms());
            record.error = Some("interrupted before every document was enqueued (a restart); run reindex-all again".to_owned());
            self.save_vec_reindex(key, &record);
        } else if !vector_kv::list_queue_entries(&self.db)
            .await
            .ok()?
            .iter()
            .any(|e| e.namespace == namespace)
        {
            record.status = BuildStatus::Complete;
            record.completed_at_ms = Some(now_ms());
            self.save_vec_reindex(key, &record);
        }
        Some(record)
    }

    /// Save a vector reindex's record. A failed save is logged, not returned:
    /// the record it would have replaced is still there, and only the progress
    /// it shows is behind.
    #[cfg(feature = "semantic-search")]
    fn save_vec_reindex(&self, key: JobKey, record: &VecReindexProgress) {
        if let Err(e) = self.maintenance.put(key, record) {
            log::warn!("vector reindex progress: could not save {key:?}: {e}");
        }
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
    /// 4. A reindex record is saved in the maintenance namespace so the
    ///    progress API can track this reindex across server restarts.
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
        ns_id: crate::namespace::NamespaceId,
        label: &str,
        text_of: impl Fn(&[u8]) -> Option<String>,
    ) -> Result<ReindexStats, DocStoreError> {
        // 409 guard: one reindex of a namespace enqueues at a time. Held until
        // this function returns, however it returns.
        let key = JobKey::vector_reindex(ns_id);
        let _write = self.writes.enter(namespace)?;
        let _claim = ReindexClaim::take(&self.vec_reindexing, namespace)?;
        self.check_not_dropping(namespace)?;

        // Save the running record before enqueueing anything, so a crash part
        // way through leaves a record that says so.
        let started_at_ms = now_ms();
        let record = |enqueued: usize, exhausted_cleared: usize, enqueue_done: bool| VecReindexProgress {
            status: BuildStatus::Running,
            started_at_ms,
            completed_at_ms: None,
            total_enqueued: enqueued,
            exhausted_cleared,
            error: None,
            enqueue_done,
        };
        self.maintenance.put(key, &record(0, 0, false))?;

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
                self.save_vec_reindex(key, &record(enqueued, exhausted_cleared, false));
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
            self.save_vec_reindex(
                key,
                &VecReindexProgress {
                    status: BuildStatus::Failed,
                    completed_at_ms: Some(now_ms()),
                    error: Some(e.to_string()),
                    ..record(enqueued, exhausted_cleared, false)
                },
            );
            return Err(e);
        }

        // The final count; the status stays `Running` until the worker finishes
        // (tracked via queue depth).
        self.save_vec_reindex(key, &record(enqueued, exhausted_cleared, true));

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
    pub(in crate::doc_store) async fn wait_for_reindex(&self, namespace: &str) {
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
