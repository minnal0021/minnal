//! Background worker that processes the async vector-index queue.
//!
//! Every document write that targets a semantic-search-enabled namespace
//! enqueues a `(namespace, doc_id, text)` entry in the
//! [`PENDING_VEC_INDEX_NS`] KV namespace instead of calling the embedding
//! service inline.  This worker picks those entries up, calls the embedding
//! service, quantises the result, and writes the [`VectorIndex`] to the
//! companion `{ns}_sparse_vector`, `{ns}_dense_vector`, and
//! `{ns}_sparse_vector_meta` namespaces, then removes the queue entry.
//!
//! # Completion waits for durability
//!
//! The vector writes are no-WAL (durable only after a memtable flush) while the
//! queue removal is WAL-backed (durable at once), so an entry is completed only
//! after its vectors are flushed. Written entries collect in a batch of up to
//! [`COMPLETION_BATCH`]; each batch, and whatever is pending at the end of a pass,
//! is made durable with one flush of the affected vector namespaces
//! ([`vector_kv::make_vector_writes_durable`]) and then completed. A crash before
//! that flush leaves the entries queued, and the next pass re-embeds them; writing
//! the same text's vectors again is harmless. See
//! `vector_kv::make_vector_writes_durable` for what completing first used to lose.
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

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::AsyncDb;
use log::{debug, info, warn};
use tokio::sync::Notify;
use tokio::task::JoinSet;

use crate::doc_store::error::DocStoreError;
use crate::doc_store::store::{NamespaceSemantics, SemanticSearchContext, load_vector_settings};
use crate::vector_kv::{self, QueueEntry, QueueEntryKind};

/// Written embed entries completed per flush of the vector namespaces. Larger
/// batches mean fewer flushes (each makes level-0 files for compaction to merge);
/// smaller ones complete entries sooner. Search is unaffected either way: vectors
/// are readable as soon as they are written.
const COMPLETION_BATCH: usize = 256;

/// What processing one queue entry left to do.
enum Processed {
    /// An embed entry's vectors are written but not yet durable; it is completed
    /// with its batch.
    Written,
    /// Nothing left: a `Clear` entry, whose writes are all WAL-backed, completes
    /// itself.
    Done,
}

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
    startup_pass: tokio::sync::watch::Receiver<bool>,
}

impl VecIndexWorkerHandle {
    /// Becomes `true` once the worker has made one pass over the queue it found
    /// at startup (the work a crash left behind), or found it empty.
    ///
    /// Startup reconciliation waits for this. A delete interrupted by a crash
    /// leaves the document, no vectors and a `Clear` tombstone; reconciliation
    /// never overwrites a tombstone, so run before the tombstone is processed it
    /// skipped the document, and the tombstone's removal then left it unindexed
    /// until the next restart (crash-audit test `delete_survives_a_crash_anywhere`).
    pub fn startup_pass(&self) -> tokio::sync::watch::Receiver<bool> {
        self.startup_pass.clone()
    }

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
    /// Where namespace schemas live: each entry is embedded with its
    /// namespace's model, dimension and chunking, read from its schema.
    schema_dir: Arc<PathBuf>,
    notify: Arc<Notify>,
    shutdown: Arc<AtomicBool>,
    config: Arc<VectorIndexConfig>,
    startup_pass: Arc<tokio::sync::watch::Sender<bool>>,
}

impl VecIndexWorker {
    /// Spawn the worker and return a handle.
    ///
    /// On startup the worker drains any queue entries that survived a crash
    /// (crash recovery).  It then waits on `notify` signals from write
    /// operations and processes new entries as they arrive, with a 30 s
    /// fallback poll as a safety net.
    pub fn start(
        db: Arc<AsyncDb>,
        ctx: Arc<SemanticSearchContext>,
        schema_dir: PathBuf,
        notify: Arc<Notify>,
        config: VectorIndexConfig,
    ) -> VecIndexWorkerHandle {
        let shutdown = Arc::new(AtomicBool::new(false));
        let (pass_tx, pass_rx) = tokio::sync::watch::channel(false);
        let worker = VecIndexWorker {
            db,
            ctx,
            schema_dir: Arc::new(schema_dir),
            notify: Arc::clone(&notify),
            shutdown: Arc::clone(&shutdown),
            config: Arc::new(config),
            startup_pass: Arc::new(pass_tx),
        };
        let task = tokio::spawn(async move { worker.run().await });
        VecIndexWorkerHandle {
            shutdown,
            notify,
            task: Some(task),
            startup_pass: pass_rx,
        }
    }

    async fn run(self) {
        info!("vec index worker started — draining queue (crash recovery)");
        self.drain_queue().await;
        self.startup_pass.send_replace(true); // also covers an empty queue
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

            // Entries whose namespace no longer exists must be discarded, not
            // embedded.
            //
            // Store deletion and the queue are separate writes, so a delete that
            // races this worker — or a crash between the two — can strand entries
            // for a namespace that is gone. Processing one calls `upsert_vectors`,
            // which resolves `{ns}_sparse_vector` and friends through
            // get-or-create, and so **recreates the dropped store's vector sidecar
            // namespaces**. Observed in a stress run: a store at ns_id 22 was
            // deleted and its sidecars reappeared as ns_ids 30/34/38 — ids are
            // monotonic and never reused, so they were created after the drop.
            // They then survive on disk, come back in the registry at every
            // restart, and are never cleaned up.
            let live_namespaces: std::collections::HashSet<String> = self.db.list_namespaces().into_iter().map(|(name, _)| name).collect();

            // Separate entries at max retries (leave in queue, admin must clear).
            let mut exhausted_count = 0usize;
            let mut orphaned_count = 0usize;
            let mut by_namespace: BTreeMap<String, VecDeque<QueueEntry>> = BTreeMap::new();

            for entry in all_entries {
                if !live_namespaces.contains(&entry.namespace) {
                    orphaned_count += 1;
                    if let Err(e) = vector_kv::remove_queue_entry(&self.db, &entry.namespace, &entry.doc_id_bytes).await {
                        warn!(
                            "vec index worker: could not drop orphaned queue entry \
                             ns='{}' doc='{}': {e}",
                            entry.namespace,
                            doc_id_display(&entry.doc_id_bytes),
                        );
                    }
                } else if entry.retry_count >= self.config.max_retries {
                    exhausted_count += 1;
                } else {
                    by_namespace.entry(entry.namespace.clone()).or_default().push_back(entry);
                }
            }

            if orphaned_count > 0 {
                info!(
                    "vec index worker: discarded {orphaned_count} queue entry/entries for \
                     namespaces that no longer exist"
                );
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

            // Each namespace's settings, read once per pass: not cached across
            // passes, because a dropped namespace can be recreated under the same
            // name with another model.
            let mut resolved_all: HashMap<String, Result<Arc<NamespaceSemantics>, String>> = HashMap::new();
            for ns in by_namespace.keys() {
                let resolved = match load_vector_settings(&self.schema_dir, ns) {
                    Ok((settings, ns_id)) => self.ctx.for_namespace(&self.db, ns, ns_id, &settings).await.map(Arc::new),
                    Err(e) => Err(e),
                }
                .map_err(|e| e.to_string());
                if let Err(e) = &resolved {
                    warn!("vec index worker: cannot embed for namespace '{ns}': {e}");
                }
                resolved_all.insert(ns.clone(), resolved);
            }
            let semantics = Arc::new(resolved_all);

            // Process work_queue with bounded concurrency.
            let concurrency = self.config.concurrency.max(1);
            let mut set: JoinSet<(QueueEntry, Result<Processed, DocStoreError>)> = JoinSet::new();
            let mut work_iter = work_queue.into_iter();
            let mut any_failed = false;
            let mut indexed_count = 0usize;
            let mut failed_count = 0usize;
            // Embed entries whose vectors are written but not yet durable.
            let mut written: Vec<QueueEntry> = Vec::new();

            // Seed the JoinSet with the first batch of tasks.
            for entry in (&mut work_iter).take(concurrency) {
                let worker = self.clone();
                let semantics = Arc::clone(&semantics);
                set.spawn(async move {
                    let result = worker.process_one(&entry, semantics.get(&entry.namespace)).await;
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
                    let semantics = Arc::clone(&semantics);
                    set.spawn(async move {
                        let result = worker.process_one(&entry, semantics.get(&entry.namespace)).await;
                        (entry, result)
                    });
                }

                match join_result {
                    Ok((entry, Ok(Processed::Written))) => {
                        written.push(entry);
                        if written.len() >= COMPLETION_BATCH {
                            let (done, failed) = self.complete_written(std::mem::take(&mut written)).await;
                            indexed_count += done;
                            failed_count += failed;
                            any_failed |= failed > 0;
                        }
                    }
                    Ok((entry, Ok(Processed::Done))) => {
                        indexed_count += 1;
                        debug!(
                            "vec index worker: cleared ns='{}' doc='{}'",
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

            if !written.is_empty() {
                let (done, failed) = self.complete_written(std::mem::take(&mut written)).await;
                indexed_count += done;
                failed_count += failed;
                any_failed |= failed > 0;
            }

            // The first pass over the startup queue is done (see `startup_pass`).
            self.startup_pass.send_replace(true);

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

    /// Carry out one queue entry.
    ///
    /// **Embed:** embed `text`, quantise it with both multi-bit (single embedding)
    /// and single-bit (chunked embeddings), and write the vectors. The entry is
    /// **not** completed here: it joins the pass's batch, which
    /// [`complete_written`](Self::complete_written) makes durable and then completes
    /// (see the module docs).
    ///
    /// **Clear:** delete the document's vectors and remove the tombstone.
    async fn process_one(&self, entry: &QueueEntry, semantics: Option<&Result<Arc<NamespaceSemantics>, String>>) -> Result<Processed, DocStoreError> {
        match entry.kind {
            QueueEntryKind::Embed => {
                let ns = match semantics {
                    Some(Ok(ns)) => ns,
                    Some(Err(e)) => return Err(DocStoreError::EmbeddingFailed(e.clone())),
                    None => return Err(DocStoreError::EmbeddingFailed(format!("no settings resolved for '{}'", entry.namespace))),
                };
                let vector_indexes = crate::semantic_search::service::embed_document(&ns.config, &*ns.ivf, &entry.text)
                    .await
                    .map_err(|e| DocStoreError::EmbeddingFailed(e.to_string()))?;
                vector_kv::upsert_vectors(&self.db, &entry.namespace, &entry.doc_id_bytes, &entry.text, &vector_indexes).await?;
                Ok(Processed::Written)
            }
            QueueEntryKind::Clear => {
                vector_kv::process_clear(&self.db, &entry.namespace, &entry.doc_id_bytes).await?;
                Ok(Processed::Done)
            }
        }
    }

    /// Make a batch of written embed entries durable, then complete each one.
    /// Returns `(completed, failed)`.
    ///
    /// If the flush fails, nothing is completed: every entry stays queued and the
    /// next pass redoes it. Each completion is conditional — the entry comes from
    /// this pass's snapshot, and the document may have been upserted or deleted
    /// since (see `vector_kv`'s conditional queue updates).
    async fn complete_written(&self, written: Vec<QueueEntry>) -> (usize, usize) {
        let mut namespaces: Vec<String> = written.iter().map(|e| e.namespace.clone()).collect();
        namespaces.sort_unstable();
        namespaces.dedup();
        if let Err(e) = vector_kv::make_vector_writes_durable(&self.db, &namespaces).await {
            warn!(
                "vec index worker: could not flush the vectors of {} written entr(y/ies); \
                 leaving them queued for the next pass: {e}",
                written.len(),
            );
            return (0, written.len());
        }
        let (mut done, mut failed) = (0usize, 0usize);
        for entry in written {
            match vector_kv::complete_embed(&self.db, &entry).await {
                Ok(()) => {
                    done += 1;
                    debug!(
                        "vec index worker: indexed ns='{}' doc='{}'",
                        entry.namespace,
                        doc_id_display(&entry.doc_id_bytes),
                    );
                }
                Err(e) => {
                    failed += 1;
                    warn!(
                        "vec index worker: completing ns='{}' doc='{}' failed: {e}",
                        entry.namespace,
                        doc_id_display(&entry.doc_id_bytes),
                    );
                    self.increment_retry(&entry, &e.to_string()).await;
                }
            }
        }
        (done, failed)
    }

    /// Increment the retry count and record the last error for a failed queue
    /// entry — only if the entry was not replaced while it was being processed.
    async fn increment_retry(&self, entry: &QueueEntry, error: &str) {
        match vector_kv::record_queue_failure(&self.db, entry, error).await {
            Ok(Some(_)) => {}
            Ok(None) => debug!(
                "vec index worker: ns='{}' doc='{}' was updated or deleted while failing; the newer entry is kept",
                entry.namespace,
                doc_id_display(&entry.doc_id_bytes),
            ),
            Err(e) => warn!(
                "vec index worker: failed to persist retry count for \
                 ns='{}' doc='{}': {e}",
                entry.namespace,
                doc_id_display(&entry.doc_id_bytes),
            ),
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
