//! Background worker that processes the async vector-index queue.
//!
//! Every document write that targets a semantic-search-enabled namespace
//! enqueues a `(namespace, doc_id, text)` entry in the
//! [`PENDING_VEC_INDEX_NS`] KV namespace instead of calling the embedding
//! service inline.  This worker picks those entries up, calls the embedding
//! service, quantises the result, and writes the [`VectorIndex`] to the
//! companion `{ns}_sparse_vector`, `{ns}_dense_vector`, and
//! `{ns}_sparse_vector_meta` namespaces, then removes the queue entry.  The
//! vector writes and the queue delete are independent single-op writes; the
//! worker is idempotent, so a crash between them simply re-processes the entry
//! on the next pass.
//!
//! # Lifecycle
//!
//! [`VecIndexWorker::start`] spawns the task and returns a
//! [`VecIndexWorkerHandle`].  The store holds the handle and exposes
//! [`DocStore::shutdown_vec_index_worker`] for graceful shutdown.  Dropping
//! the handle without calling `shutdown` signals the flag so the task exits
//! on its next iteration — no entries are lost because the queue is durable.
//!
//! # Scheduling
//!
//! Entries are grouped by namespace and processed in **round-robin** order so
//! that a large backlog in one namespace cannot starve others.  Up to
//! [`VectorIndexConfig::concurrency`] embedding calls are in-flight at once
//! via a [`JoinSet`].
//!
//! # Retry / back-off
//!
//! On failure the entry's `retry_count` is incremented and persisted
//! atomically.  The worker logs the namespace, doc-id and error at `WARN`
//! level on every failure.  Once `retry_count` reaches
//! [`VectorIndexConfig::max_retries`] the entry is skipped on each pass
//! (left in the queue for inspection / manual deletion via the admin API).
//! After any failure in a pass the worker sleeps
//! [`VectorIndexConfig::retry_wait_secs`] before re-scanning.
//!
//! # Deduplication
//!
//! The queue key encodes `(namespace, doc_id)`.  Rapid successive writes to
//! the same document overwrite the queue entry — the worker makes exactly one
//! embedding call for the most-recent text.
//!
//! [`PENDING_VEC_INDEX_NS`]: crate::vector_kv::PENDING_VEC_INDEX_NS

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::AsyncDb;
use log::{debug, info, warn};
use tokio::sync::Notify;
use tokio::task::JoinSet;

use crate::doc_store::error::DocStoreError;
use crate::doc_store::store::SemanticSearchContext;
use crate::vector_kv::{self, QueueEntry};

// ── Config ────────────────────────────────────────────────────────────────────

/// Tuning knobs for the async vector-index background worker.
///
/// Pass this to [`DocStore::with_vector_index_config`] **before** calling
/// [`DocStore::with_semantic_search`].  Defaults are conservative values
/// suitable for most deployments.
///
/// [`DocStore::with_vector_index_config`]: crate::doc_store::store::DocStore::with_vector_index_config
/// [`DocStore::with_semantic_search`]: crate::doc_store::store::DocStore::with_semantic_search
#[derive(Debug, Clone)]
pub struct VectorIndexConfig {
    /// Seconds to sleep after a pass that contained at least one failure,
    /// before re-scanning the queue.  Default: 2.
    pub retry_wait_secs: u64,
    /// Maximum number of embedding attempts per queue entry.  Once an entry
    /// reaches this count it is skipped on every subsequent pass and must be
    /// removed manually via the admin API.  Default: 5.
    pub max_retries: u32,
    /// Maximum number of concurrent embedding calls in flight at once.
    /// Default: 4.
    pub concurrency: usize,
}

impl Default for VectorIndexConfig {
    fn default() -> Self {
        Self {
            retry_wait_secs: 2,
            max_retries: 5,
            concurrency: 4,
        }
    }
}

// ── Handle ────────────────────────────────────────────────────────────────────

/// Handle to a running [`VecIndexWorker`] background task.
///
/// Call [`shutdown`] for a clean stop; or simply drop the handle — the `Drop`
/// impl signals the shutdown flag so the task exits on its next iteration.
/// No queue entries are lost in either case.
///
/// [`shutdown`]: VecIndexWorkerHandle::shutdown
pub struct VecIndexWorkerHandle {
    shutdown: Arc<AtomicBool>,
    notify: Arc<Notify>,
    task: Option<tokio::task::JoinHandle<()>>,
}

impl VecIndexWorkerHandle {
    /// Signal the worker to stop and await its exit.
    pub async fn shutdown(mut self) {
        self.shutdown.store(true, Ordering::Release);
        self.notify.notify_one();
        if let Some(task) = self.task.take() {
            let _ = task.await;
        }
    }
}

impl Drop for VecIndexWorkerHandle {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        self.notify.notify_one();
        // The JoinHandle is detached here; the task will exit on its next
        // iteration when it sees the shutdown flag.
    }
}

// ── Worker ────────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub(crate) struct VecIndexWorker {
    db: Arc<AsyncDb>,
    ctx: Arc<SemanticSearchContext>,
    notify: Arc<Notify>,
    shutdown: Arc<AtomicBool>,
    config: Arc<VectorIndexConfig>,
}

impl VecIndexWorker {
    /// Spawn the worker and return a handle.
    ///
    /// On startup the worker drains any queue entries that survived a crash
    /// (crash recovery).  It then waits on `notify` signals from write
    /// operations and processes new entries as they arrive, with a 30 s
    /// fallback poll as a safety net.
    pub fn start(db: Arc<AsyncDb>, ctx: Arc<SemanticSearchContext>, notify: Arc<Notify>, config: VectorIndexConfig) -> VecIndexWorkerHandle {
        let shutdown = Arc::new(AtomicBool::new(false));
        let worker = VecIndexWorker {
            db,
            ctx,
            notify: Arc::clone(&notify),
            shutdown: Arc::clone(&shutdown),
            config: Arc::new(config),
        };
        let task = tokio::spawn(async move { worker.run().await });
        VecIndexWorkerHandle {
            shutdown,
            notify,
            task: Some(task),
        }
    }

    async fn run(self) {
        info!("vec index worker started — draining queue (crash recovery)");
        self.drain_queue().await;
        info!("vec index worker ready");

        loop {
            tokio::select! {
                _ = self.notify.notified() => {}
                _ = tokio::time::sleep(Duration::from_secs(30)) => {}
            }

            if self.shutdown.load(Ordering::Acquire) {
                info!("vec index worker shutting down");
                break;
            }

            self.drain_queue().await;
        }

        info!("vec index worker stopped");
    }

    /// Process all current actionable queue entries.
    ///
    /// Entries are grouped by namespace and visited in round-robin order so
    /// that no single namespace can delay others.  Up to `config.concurrency`
    /// embedding calls run concurrently.  Entries whose `retry_count` has
    /// reached `config.max_retries` are logged and skipped (left in the queue
    /// for admin inspection).  After any failure the worker sleeps
    /// `config.retry_wait_secs` before returning so the caller re-scans on
    /// the next pass.
    ///
    /// An `INFO`-level summary is emitted at the start and end of every pass
    /// so that progress through large queues (e.g. after a bulk load) is
    /// visible in the logs without enabling `DEBUG`.
    async fn drain_queue(&self) {
        loop {
            let all_entries = match vector_kv::list_queue_entries(&self.db).await {
                Ok(e) => e,
                Err(e) => {
                    warn!("vec index worker: queue scan failed: {e}");
                    return;
                }
            };

            if all_entries.is_empty() {
                return;
            }

            // Separate entries at max retries (leave in queue, admin must clear).
            let mut exhausted_count = 0usize;
            let mut by_namespace: BTreeMap<String, VecDeque<QueueEntry>> = BTreeMap::new();

            for entry in all_entries {
                if entry.retry_count >= self.config.max_retries {
                    exhausted_count += 1;
                } else {
                    by_namespace.entry(entry.namespace.clone()).or_default().push_back(entry);
                }
            }

            let actionable_count: usize = by_namespace.values().map(|q| q.len()).sum();
            let total_depth = actionable_count + exhausted_count;

            // Emit a per-pass start summary so queue depth is visible at INFO level.
            if exhausted_count > 0 {
                warn!(
                    "vec index worker: {exhausted_count} entry/entries have reached \
                     max_retries={} and are awaiting manual removal via the admin API",
                    self.config.max_retries,
                );
                info!(
                    "vec index worker: pass start — depth={total_depth} \
                     actionable={actionable_count} exhausted={exhausted_count} \
                     namespaces={}",
                    by_namespace.len(),
                );
            } else {
                info!(
                    "vec index worker: pass start — depth={total_depth} \
                     actionable={actionable_count} namespaces={}",
                    by_namespace.len(),
                );
            }

            if by_namespace.is_empty() {
                return;
            }

            // Build a round-robin ordered work list: one entry per namespace per
            // pass until all namespaces are drained.
            let mut work_queue: Vec<QueueEntry> = Vec::new();
            loop {
                let mut added = 0;
                for entries in by_namespace.values_mut() {
                    if let Some(e) = entries.pop_front() {
                        work_queue.push(e);
                        added += 1;
                    }
                }
                if added == 0 {
                    break;
                }
            }

            // Process work_queue with bounded concurrency.
            let concurrency = self.config.concurrency.max(1);
            let mut set: JoinSet<(QueueEntry, Result<(), DocStoreError>)> = JoinSet::new();
            let mut work_iter = work_queue.into_iter();
            let mut any_failed = false;
            let mut indexed_count = 0usize;
            let mut failed_count = 0usize;

            // Seed the JoinSet with the first batch of tasks.
            for entry in (&mut work_iter).take(concurrency) {
                let worker = self.clone();
                set.spawn(async move {
                    let result = worker.process_one(&entry).await;
                    (entry, result)
                });
            }

            while let Some(join_result) = set.join_next().await {
                if self.shutdown.load(Ordering::Acquire) {
                    set.abort_all();
                    return;
                }

                // Keep the concurrency slot filled.
                if let Some(entry) = work_iter.next() {
                    let worker = self.clone();
                    set.spawn(async move {
                        let result = worker.process_one(&entry).await;
                        (entry, result)
                    });
                }

                match join_result {
                    Ok((entry, Ok(()))) => {
                        indexed_count += 1;
                        debug!(
                            "vec index worker: indexed ns='{}' doc='{}'",
                            entry.namespace,
                            doc_id_display(&entry.doc_id_bytes),
                        );
                    }
                    Ok((entry, Err(e))) => {
                        any_failed = true;
                        failed_count += 1;
                        let new_retry = entry.retry_count + 1;
                        let exhausted = new_retry >= self.config.max_retries;
                        warn!(
                            "vec index worker: embedding failed \
                             ns='{}' doc='{}' attempt={}/{} exhausted={} error='{e}'",
                            entry.namespace,
                            doc_id_display(&entry.doc_id_bytes),
                            new_retry,
                            self.config.max_retries,
                            exhausted,
                        );
                        self.increment_retry(&entry, &e.to_string()).await;
                    }
                    Err(join_err) => {
                        warn!("vec index worker: task panicked: {join_err}");
                        any_failed = true;
                        failed_count += 1;
                    }
                }
            }

            // Per-pass completion summary visible at INFO level.
            info!(
                "vec index worker: pass complete — indexed={indexed_count} failed={failed_count} \
                 remaining={}",
                actionable_count.saturating_sub(indexed_count + failed_count),
            );

            if any_failed {
                tokio::time::sleep(Duration::from_secs(self.config.retry_wait_secs)).await;
            }

            if self.shutdown.load(Ordering::Acquire) {
                return;
            }

            // Loop: re-scan so any entries that arrived while we were processing
            // this batch are also picked up without waiting for the next notify.
        }
    }

    /// Embed `text`, quantise it with both multi-bit (single embedding) and single-bit
    /// (chunked embeddings), write all vector indexes, then remove the queue entry.
    ///
    /// The vector writes happen before the queue delete: a crash in between leaves
    /// the entry queued and the next pass re-processes it idempotently.
    async fn process_one(&self, entry: &QueueEntry) -> Result<(), DocStoreError> {
        let vector_indexes = crate::semantic_search::service::embed_document(&self.ctx.config, &self.ctx.cluster_index, &entry.text)
            .await
            .map_err(|e| DocStoreError::EmbeddingFailed(e.to_string()))?;

        vector_kv::upsert_vectors(&self.db, &entry.namespace, &entry.doc_id_bytes, &vector_indexes).await?;
        vector_kv::remove_queue_entry(&self.db, &entry.namespace, &entry.doc_id_bytes).await?;

        Ok(())
    }

    /// Increment the retry count and record the last error for a failed queue entry.
    async fn increment_retry(&self, entry: &QueueEntry, error: &str) {
        let new_count = entry.retry_count + 1;
        if let Err(e) = vector_kv::update_queue_retry(&self.db, &entry.namespace, &entry.doc_id_bytes, &entry.text, new_count, Some(error)).await {
            warn!(
                "vec index worker: failed to persist retry count for \
                 ns='{}' doc='{}': {e}",
                entry.namespace,
                doc_id_display(&entry.doc_id_bytes),
            );
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Display a doc-id as a UTF-8 string when possible, otherwise as hex.
fn doc_id_display(bytes: &[u8]) -> String {
    if let Ok(s) = std::str::from_utf8(bytes)
        && s.chars().all(|c| c.is_ascii_graphic() || c == ' ')
    {
        return s.to_owned();
    }
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_index_config_defaults() {
        let cfg = VectorIndexConfig::default();
        assert_eq!(cfg.retry_wait_secs, 2);
        assert_eq!(cfg.max_retries, 5);
        assert_eq!(cfg.concurrency, 4);
    }

    #[test]
    fn test_doc_id_display_printable_ascii() {
        assert_eq!(doc_id_display(b"my-doc-id"), "my-doc-id");
    }

    #[test]
    fn test_doc_id_display_ascii_with_space() {
        assert_eq!(doc_id_display(b"hello world"), "hello world");
    }

    #[test]
    fn test_doc_id_display_binary_falls_back_to_hex() {
        // Bytes 0x00–0x1f are not ascii_graphic, so hex path is taken.
        let bytes = [0x00u8, 0x01, 0xFF];
        assert_eq!(doc_id_display(&bytes), "0001ff");
    }

    #[test]
    fn test_doc_id_display_empty_slice() {
        assert_eq!(doc_id_display(&[]), "");
    }

    #[test]
    fn test_doc_id_display_non_utf8_is_hex() {
        // 0xFF 0xFE is not valid UTF-8.
        let bytes = [0xFFu8, 0xFE];
        assert_eq!(doc_id_display(&bytes), "fffe");
    }
}
