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
//! The vector payloads go to the vector index WAL (durable only after its next
//! sync), and the queue removal skips the WAL (durable at the queue's next flush,
//! which can come at any moment), so an entry is completed only after its
//! vectors are synced. A removal a crash loses brings the entry back, and the
//! document is embedded again. Written entries collect in a
//! batch of up to [`COMPLETION_BATCH`]; each batch, and whatever is pending at the
//! end of a pass, is made durable with one fsync of the vector index WAL
//! ([`vector_kv::make_vector_writes_durable`]) and then completed. A crash before
//! that sync leaves the entries queued, and the next pass re-embeds them; writing
//! the same text's vectors again is harmless. See
//! `vector_kv::make_vector_writes_durable` for what completing first used to lose.
//!
//! # Checked against the document
//!
//! Every write queues its entry *before* it writes the document (see
//! `vector_kv`'s *Queue first*), so a crash or error between the two leaves an
//! entry that does not describe the document: an embed of text the store never
//! received, or a `Clear` for a document whose delete never ran. Before acting on
//! an entry the worker reads the document and compares
//! ([`vector_kv::check_entry_against_document`]), but only when no write to the
//! document is in progress ([`InFlightWrites`]; nothing waits on it). An entry
//! whose document has a write in progress is skipped: that write's own entry and
//! wake-up bring it back. Otherwise no write is half done, and an entry that does
//! not describe the document (a write cut short, or a re-index entry from an
//! older snapshot) is rewritten to describe it and carried out on the next pass.
//! That is the whole of crash recovery for the vector index: the worker drains
//! the queue; nothing scans the documents.
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
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::AsyncDb;
use log::{debug, info, warn};
use tokio::sync::Notify;
use tokio::task::JoinSet;

use crate::doc_store::error::DocStoreError;
use crate::doc_store::schema_store::SchemaStore;
use crate::doc_store::store::{DocumentText, InFlightWrites, NamespaceSemantics, SemanticSearchContext, document_text};
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
    /// Skipped: the namespace is being dropped, and the drop removes the entry.
    Dropped,
    /// The entry did not describe its document and was rewritten (or replaced
    /// meanwhile); the next pass carries out what is queued now.
    Rewritten,
    /// A write to the document is in progress; skipped, and that write's drop
    /// wakes the worker again.
    InProgress,
}

/// What [`check_entry`] found.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum EntryCheck {
    /// The entry describes its document: carry it out.
    Proceed,
    /// The entry was rewritten to describe its document (or replaced meanwhile).
    Rewritten,
    /// A write to the document is in progress: leave the entry for now.
    InProgress,
    /// The document's store is gone: leave the entry to the drop.
    StoreGone,
}

/// Compare `entry` with what its document holds now (see the module docs),
/// rewriting it when it does not describe the document. The document is read
/// only after `in_flight` shows no write to it in progress, so none is half done.
pub(crate) async fn check_entry(
    db: &AsyncDb,
    schemas: &SchemaStore,
    in_flight: &InFlightWrites,
    entry: &QueueEntry,
) -> Result<EntryCheck, DocStoreError> {
    if in_flight.contains(&entry.namespace, &entry.doc_id_bytes) {
        return Ok(EntryCheck::InProgress);
    }
    let current = document_text(db, schemas, &entry.namespace, &entry.doc_id_bytes).await?;
    if matches!(current, DocumentText::StoreGone) {
        return Ok(EntryCheck::StoreGone);
    }
    Ok(if vector_kv::check_entry_against_document(db, entry, current.as_text()).await? {
        EntryCheck::Proceed
    } else {
        EntryCheck::Rewritten
    })
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
    /// The partition maintenance task, which this worker wakes; stopped with it.
    maintenance: Option<crate::doc_store::ivf_maintenance::MaintenanceHandle>,
}

impl VecIndexWorkerHandle {
    /// Signal the worker to stop and await its exit.
    pub async fn shutdown(mut self) {
        self.shutdown.store(true, Ordering::Release);
        self.notify.notify_one();
        if let Some(task) = self.task.take() {
            let _ = task.await;
        }
        if let Some(m) = self.maintenance.take() {
            m.shutdown().await;
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
    /// Every store's schema: each entry is embedded with its namespace's
    /// model, dimension and chunking, read from its schema.
    schemas: Arc<SchemaStore>,
    /// Documents with a write in progress, which the worker skips.
    in_flight: Arc<InFlightWrites>,
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
    pub fn start(
        db: Arc<AsyncDb>,
        ctx: Arc<SemanticSearchContext>,
        schemas: Arc<SchemaStore>,
        in_flight: Arc<InFlightWrites>,
        notify: Arc<Notify>,
        config: VectorIndexConfig,
    ) -> VecIndexWorkerHandle {
        let shutdown = Arc::new(AtomicBool::new(false));
        let maintenance = crate::doc_store::ivf_maintenance::start(Arc::clone(&db), Arc::clone(&ctx), Arc::clone(&schemas));
        let worker = VecIndexWorker {
            db,
            ctx,
            schemas,
            in_flight,
            notify: Arc::clone(&notify),
            shutdown: Arc::clone(&shutdown),
            config: Arc::new(config),
        };
        let task = tokio::spawn(async move { worker.run().await });
        VecIndexWorkerHandle {
            shutdown,
            notify,
            task: Some(task),
            maintenance: Some(maintenance),
        }
    }

    async fn run(self) {
        // Splits a crash interrupted are finished first: nothing may write a
        // namespace's vectors while its journal is unfinished.
        crate::doc_store::ivf_maintenance::recover_all(&self.db, &self.ctx, &self.schemas).await;
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

            // Each namespace's settings, read once per pass: not cached across
            // passes, because a dropped namespace can be recreated under the same
            // name with another model.
            //
            // A store being dropped (or whose schema the drop already removed)
            // fails to resolve; its entries leave this pass rather than each
            // failing with a warning. The drop clears them from the queue, and
            // any it misses are discarded as orphans on the next pass.
            let mut resolved_all: HashMap<String, Result<Arc<NamespaceSemantics>, String>> = HashMap::new();
            let mut dropping: Vec<String> = Vec::new();
            for ns in by_namespace.keys() {
                let resolved = match self.schemas.vector_settings(ns) {
                    Ok((settings, ns_id)) => self.ctx.for_namespace(&self.db, ns, ns_id, &settings).await.map(Arc::new),
                    Err(e) => Err(e),
                }
                .map_err(|e| e.to_string());
                if let Err(e) = &resolved {
                    if self.ctx.is_dropping(ns) || !self.schemas.contains(ns) {
                        dropping.push(ns.clone());
                        continue;
                    }
                    warn!("vec index worker: cannot embed for namespace '{ns}': {e}");
                }
                resolved_all.insert(ns.clone(), resolved);
            }
            for ns in dropping {
                let skipped = by_namespace.remove(&ns).map_or(0, |q| q.len());
                info!("vec index worker: skipped {skipped} queue entry/entries for namespace '{ns}', which is being dropped");
            }
            let semantics = Arc::new(resolved_all);

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
            let mut set: JoinSet<(QueueEntry, Result<Processed, DocStoreError>)> = JoinSet::new();
            let mut work_iter = work_queue.into_iter();
            let mut any_failed = false;
            let mut indexed_count = 0usize;
            let mut failed_count = 0usize;
            let mut rewritten_count = 0usize;
            let mut in_progress_count = 0usize;
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
                    Ok((entry, Ok(Processed::Dropped))) => {
                        debug!(
                            "vec index worker: skipped ns='{}' doc='{}': the namespace is being dropped",
                            entry.namespace,
                            doc_id_display(&entry.doc_id_bytes),
                        );
                    }
                    Ok((_, Ok(Processed::InProgress))) => in_progress_count += 1,
                    Ok((entry, Ok(Processed::Rewritten))) => {
                        rewritten_count += 1;
                        debug!(
                            "vec index worker: ns='{}' doc='{}': the entry did not match the document; rewritten for the next pass",
                            entry.namespace,
                            doc_id_display(&entry.doc_id_bytes),
                        );
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

            // Nothing moved but entries whose document has a write in progress:
            // re-scanning now would spin until those writes finish. Each one wakes
            // the worker when it does (its `InFlightWrite` drop).
            if in_progress_count > 0 && indexed_count + failed_count + rewritten_count == 0 {
                return;
            }

            // Loop: re-scan so any entries that arrived while we were processing
            // this batch are also picked up without waiting for the next notify.
        }
    }

    /// Carry out one queue entry, after checking it against its document
    /// ([`check_against_document`](Self::check_against_document)).
    ///
    /// **Embed:** embed `text`, quantise it with both multi-bit (single embedding)
    /// and single-bit (chunked embeddings), and write the vectors. The entry is
    /// **not** completed here: it joins the pass's batch, which
    /// [`complete_written`](Self::complete_written) makes durable and then completes
    /// (see the module docs).
    ///
    /// **Clear:** delete the document's vectors and remove the tombstone.
    async fn process_one(&self, entry: &QueueEntry, semantics: Option<&Result<Arc<NamespaceSemantics>, String>>) -> Result<Processed, DocStoreError> {
        // First, and whatever the namespace's settings: the check needs only the
        // queue and the document.
        if let Some(skip) = self.check_against_document(entry).await? {
            return Ok(skip);
        }
        match entry.kind {
            QueueEntryKind::Embed => {
                let ns = match semantics {
                    Some(Ok(ns)) => ns,
                    Some(Err(e)) => return Err(DocStoreError::EmbeddingFailed(e.clone())),
                    None => return Err(DocStoreError::EmbeddingFailed(format!("no settings resolved for '{}'", entry.namespace))),
                };
                // The partition was resolved at the start of the pass; a drop
                // since then marks it. No embedding call for a dropped store.
                if ns.partition.is_dropped() {
                    return Ok(Processed::Dropped);
                }
                // The network call runs outside the routing epoch; routing,
                // the write and its count update run inside it, against the
                // snapshot current at that moment (a split may have published a
                // new one while the text was being embedded).
                let (dense, chunks) = crate::semantic_search::service::embed_document_floats(&ns.config, &entry.text)
                    .await
                    .map_err(|e| DocStoreError::EmbeddingFailed(e.to_string()))?;
                let _epoch = ns.partition.read_epoch().await;
                // Checked again inside the epoch, which the drop waits out
                // before removing the namespaces: a write after the removal
                // would recreate them.
                if ns.partition.is_dropped() {
                    return Ok(Processed::Dropped);
                }
                let ivf = ns.partition.snapshot();
                let vector_indexes = crate::semantic_search::service::index_embeddings(&ns.config, &*ivf, &dense, &chunks)
                    .map_err(|e| DocStoreError::EmbeddingFailed(e.to_string()))?;
                let delta = vector_kv::upsert_vectors(&self.db, &entry.namespace, &entry.doc_id_bytes, &entry.text, &vector_indexes).await?;
                ivf.apply_delta(&delta);
                if ns.partition.needs_maintenance(&delta) {
                    self.ctx.request_maintenance(&entry.namespace);
                }
                Ok(Processed::Written)
            }
            QueueEntryKind::Clear => {
                // As for an embed: the clear's deletes resolve the vector
                // namespaces with get-or-create too.
                let partition = match semantics {
                    Some(Ok(ns)) => Some(&ns.partition),
                    _ => None,
                };
                let _epoch = match partition {
                    Some(p) => Some(p.read_epoch().await),
                    None => None,
                };
                if partition.is_some_and(|p| p.is_dropped()) || self.ctx.is_dropping(&entry.namespace) {
                    return Ok(Processed::Dropped);
                }
                let delta = vector_kv::process_clear(&self.db, &entry.namespace, &entry.doc_id_bytes).await?;
                self.ctx.apply_posting_delta(&entry.namespace, &delta);
                Ok(Processed::Done)
            }
        }
    }

    /// [`check_entry`] for the worker: `None` when the entry describes its
    /// document and should be carried out; otherwise what to report instead.
    async fn check_against_document(&self, entry: &QueueEntry) -> Result<Option<Processed>, DocStoreError> {
        Ok(match check_entry(&self.db, &self.schemas, &self.in_flight, entry).await? {
            EntryCheck::Proceed => None,
            EntryCheck::Rewritten => Some(Processed::Rewritten),
            EntryCheck::InProgress => Some(Processed::InProgress),
            EntryCheck::StoreGone => Some(Processed::Dropped),
        })
    }

    /// Make a batch of written embed entries durable, then complete each one.
    /// Returns `(completed, failed)`.
    ///
    /// If the sync fails, nothing is completed: every entry stays queued and the
    /// next pass redoes it. Each completion is conditional — the entry comes from
    /// this pass's snapshot, and the document may have been upserted or deleted
    /// since (see `vector_kv`'s conditional queue updates).
    ///
    /// Up to `config.concurrency` completions run at once. A completion can still
    /// wait on the WAL — the cleanup when a document vanished mid-embed, and the
    /// re-queue when it came back — and every WAL write in the database waits on
    /// one lock held across an fsync, so one at a time a batch would hold one
    /// place in that line against every concurrent REST writer. Entries in a
    /// batch are distinct documents (the queue holds one entry per document), so
    /// their completions are independent.
    async fn complete_written(&self, written: Vec<QueueEntry>) -> (usize, usize) {
        if let Err(e) = vector_kv::make_vector_writes_durable(&self.db).await {
            warn!(
                "vec index worker: could not sync the vectors of {} written entr(y/ies); \
                 leaving them queued for the next pass: {e}",
                written.len(),
            );
            return (0, written.len());
        }
        let concurrency = self.config.concurrency.max(1);
        let mut set: JoinSet<(QueueEntry, Result<crate::semantic_search::PostingDelta, crate::KVError>)> = JoinSet::new();
        let mut pending = written.into_iter();
        for entry in (&mut pending).take(concurrency) {
            let db = self.db.clone();
            set.spawn(async move {
                let result = vector_kv::complete_embed(&db, &entry).await;
                (entry, result)
            });
        }
        let (mut done, mut failed) = (0usize, 0usize);
        while let Some(joined) = set.join_next().await {
            if let Some(entry) = pending.next() {
                let db = self.db.clone();
                set.spawn(async move {
                    let result = vector_kv::complete_embed(&db, &entry).await;
                    (entry, result)
                });
            }
            let (entry, result) = match joined {
                Ok(joined) => joined,
                Err(join_err) => {
                    warn!("vec index worker: completion task panicked: {join_err}");
                    failed += 1;
                    continue;
                }
            };
            match result {
                Ok(delta) => {
                    self.ctx.apply_posting_delta(&entry.namespace, &delta);
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
