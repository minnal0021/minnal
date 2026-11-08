//! Re-queueing a store's documents for embedding (`index_all`, `kv_index_all`)
//! as a `vector_reindex` job, and the progress view of it.
//!
//! The job pages through the store, enqueueing every document with embedding
//! text, and saves how far it got after each page, so a restart resumes it.
//! Enqueueing is idempotent (an entry is overwritten), and the vector worker
//! checks each entry against its document before embedding, so a page done
//! twice, or a document enqueued with an older text, costs nothing. A cancel
//! stops the job at its next page; what it queued is embedded as usual
//! (re-embedding a document changes nothing).

#[cfg(feature = "semantic-search")]
use crate::doc_store::*;
#[cfg(feature = "semantic-search")]
use crate::namespace_lifecycle::{BuildStatus, JobKind, JobRecord};

/// How far a vector reindex got: its job's progress.
#[cfg(feature = "semantic-search")]
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
struct ReindexProgress {
    /// Exhausted queue entries removed before enqueueing.
    exhausted_cleared: usize,
    /// Whether the exhausted entries are cleared.
    cleared: bool,
    /// Documents enqueued so far.
    enqueued: usize,
    /// Hex of the last key of the last page enqueued.
    resume_after_hex: Option<String>,
}

#[cfg(feature = "semantic-search")]
impl DocStore {
    /// Re-enqueue every document of document store `namespace` for vector
    /// indexing, as a `vector_reindex` job; returns once every document is
    /// enqueued (the worker embeds them after).
    ///
    /// Exhausted queue entries (`retry_count` at `max_retries`) are removed
    /// first, so they do not block processing; then every document whose
    /// embedding fields give text is enqueued with `retry_count = 0`. The
    /// documents and the queue are read a page of `REINDEX_PAGE` (1,000) at a
    /// time, so memory stays bounded however large the store is.
    ///
    /// Refused with [`DocStoreError::SemanticSearchNotEnabled`] when the store
    /// has no semantic search, and with [`DocStoreError::JobConflict`] while
    /// another vector job of the store runs.
    pub async fn index_all(&self, namespace: &str) -> Result<ReindexStats, DocStoreError> {
        self.load_schema(namespace)?;
        self.run_vector_reindex(namespace).await
    }

    /// [`index_all`](Self::index_all) for a KV store, whose values (non-empty
    /// UTF-8 strings) are the embedding text.
    pub async fn kv_index_all(&self, namespace: &str) -> Result<ReindexStats, DocStoreError> {
        self.load_kv_schema(namespace)?;
        self.run_vector_reindex(namespace).await
    }

    /// [`index_all`](Self::index_all) or [`kv_index_all`](Self::kv_index_all),
    /// whichever `namespace` is, returning the job's name once it is handed to
    /// the governor.
    pub fn begin_vector_reindex(&self, namespace: &str) -> Result<String, DocStoreError> {
        self.running_governor()?;
        let key = self.start_vector_reindex(namespace)?;
        self.submit_job(namespace, key)?;
        Ok(key.kind.name().to_owned())
    }

    async fn run_vector_reindex(&self, namespace: &str) -> Result<ReindexStats, DocStoreError> {
        let key = self.start_vector_reindex(namespace)?;
        self.run_job(namespace, key).await?;
        let progress: ReindexProgress = self.jobs.get(key)?.and_then(|r| r.progress()).unwrap_or_default();
        Ok(ReindexStats {
            exhausted_cleared: progress.exhausted_cleared,
            enqueued: progress.enqueued,
        })
    }

    pub(in crate::doc_store) fn start_vector_reindex(&self, namespace: &str) -> Result<JobKey, DocStoreError> {
        self.job_runner_ready()?;
        self.schemas.admit(namespace)?;
        self.schemas.require_semantic_search(namespace)?;
        let ns_id = self
            .schemas
            .get(namespace)
            .and_then(|s| s.ns_id())
            .ok_or_else(|| DocStoreError::MissingNsId {
                namespace: namespace.to_owned(),
            })?;
        let key = JobKey::vector_reindex(ns_id);
        self.start_job(namespace, key, "every document", &ReindexProgress::default())?;
        Ok(key)
    }

    /// The body of a `vector_reindex` job.
    pub(in crate::doc_store) async fn vector_reindex_job(&self, namespace: &str, key: JobKey) -> Result<(), DocStoreError> {
        use crate::doc_store::indexing::record_error;
        let Some(record) = self.jobs.get(key)?.filter(JobRecord::is_live) else {
            return Ok(());
        };
        let mut progress: ReindexProgress = record.progress().ok_or_else(|| record_error(key, "no reindex progress"))?;
        let embedding_fields = match self.schemas.get(namespace) {
            Some(StoredSchema::Doc(schema)) => Some(schema.embedding_fields.clone()),
            Some(StoredSchema::Kv(_)) => None,
            None => {
                return Err(DocStoreError::NotFound {
                    namespace: namespace.to_owned(),
                });
            }
        };
        let text_of = |value: &[u8]| -> Option<String> {
            match &embedding_fields {
                Some(fields) => {
                    let doc = serde_json::from_slice::<serde_json::Value>(value).ok()?;
                    Some(build_embedding_text(&doc, fields)).filter(|t| !t.is_empty())
                }
                None => std::str::from_utf8(value).ok().filter(|t| !t.is_empty()).map(str::to_owned),
            }
        };
        let cancelled = || self.jobs.is_cancelling(key);

        // Clear exhausted entries (durable single-op deletes), then enqueue
        // every document with embedding text. Enqueues are WAL-backed: the
        // queue is the durable record of what still needs (re-)indexing.
        if !progress.cleared {
            let max_retries = self.vector_index_config.max_retries;
            info!("vector reindex: namespace='{namespace}' clearing exhausted queue entries (max_retries={max_retries})");
            let prefix = vector_kv::queue_prefix(namespace);
            let end = prefix_upper_bound(&prefix);
            let mut cursor = Some(prefix);
            while let Some(start) = cursor.take() {
                let (page, next) = vector_kv::queue_page(&self.db, start, end.clone(), REINDEX_PAGE).await?;
                for entry in page.iter().filter(|e| e.retry_count >= max_retries) {
                    vector_kv::remove_queue_entry(&self.db, &entry.namespace, &entry.doc_id_bytes).await?;
                    progress.exhausted_cleared += 1;
                }
                cursor = next;
            }
            progress.cleared = true;
            self.save_job_progress(key, &progress);
        }

        info!("vector reindex: namespace='{namespace}' enqueueing documents, {REINDEX_PAGE} per page");
        let ns = self.store_ns(namespace).await?;
        let mut cursor: Option<Vec<u8>> = progress
            .resume_after_hex
            .as_deref()
            .and_then(crate::support::hex::hex_to_bytes)
            .map(|mut k| {
                // The scan cursor is inclusive: start just past the last key done.
                k.push(0);
                k
            });
        loop {
            if cancelled() {
                info!("vector reindex: namespace='{namespace}' cancelled after {} documents", progress.enqueued);
                return Ok(());
            }
            let (page, next) = ns.scan(cursor.take(), None, REINDEX_PAGE).await?;
            for (key, value) in &page {
                if let Some(text) = text_of(value) {
                    vector_kv::enqueue_embed(&self.db, namespace, key, &text).await?;
                    progress.enqueued += 1;
                }
            }
            if let Some((last, _)) = page.last() {
                progress.resume_after_hex = Some(crate::support::hex::bytes_to_hex(last));
            }
            self.save_job_progress(key, &progress);
            if let Some(notify) = &self.notify {
                notify.notify_one();
            }
            match next {
                Some(c) => cursor = Some(c),
                None => break,
            }
        }
        info!(
            "vector reindex: namespace='{namespace}' enqueue done — enqueued={} exhausted_cleared={}",
            progress.enqueued, progress.exhausted_cleared
        );
        Ok(())
    }

    /// `namespace`'s last vector reindex, or `None` if it has had none. Its
    /// job ends once every document is enqueued; it reads `Running` until
    /// the worker has also embedded what the namespace has queued.
    pub async fn vec_reindex_progress(&self, namespace: &str) -> Option<VecReindexProgress> {
        let ns_id = self.schemas.get(namespace)?.ns_id()?;
        let record = self.jobs.get(JobKey::new(ns_id, JobKind::VectorReindex)).ok()??;
        let progress: ReindexProgress = record.progress()?;
        let mut status = record.status;
        if status == BuildStatus::Complete
            && vector_kv::list_queue_entries(&self.db)
                .await
                .ok()?
                .iter()
                .any(|e| e.namespace == namespace)
        {
            status = BuildStatus::Running;
        }
        Some(VecReindexProgress {
            status,
            started_at_ms: record.started_at_ms,
            completed_at_ms: status.is_terminal().then_some(record.updated_at_ms),
            total_enqueued: progress.enqueued,
            exhausted_cleared: progress.exhausted_cleared,
            error: record.error,
            enqueue_done: record.status == BuildStatus::Complete,
        })
    }
}

/// Rows a vector reindex reads per page (documents, and queue entries while it
/// clears exhausted ones): enough to amortise each page's LSM read, small enough
/// that a page of large documents stays a few MB.
#[cfg(feature = "semantic-search")]
pub(crate) const REINDEX_PAGE: usize = 1_000;
