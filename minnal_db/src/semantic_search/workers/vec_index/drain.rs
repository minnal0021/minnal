//! Draining the embed queue: the worker walks it in rounds, a page of each
//! namespace's entries per pass, and carries out every entry it can act on,
//! round-robin across namespaces with bounded concurrency.
//!
//! The queue's values are the documents' texts, and after a bulk load it can hold
//! every document, so it is never read whole: a pass holds at most
//! [`WORK_PAGE`] entries of each namespace. Each namespace has a cursor for the
//! round, so entries left in the queue (exhausted ones, or ones skipped for a
//! write in progress) are passed over rather than read again by every pass.

use super::*;

/// The most entries of one namespace a pass reads.
#[cfg(not(test))]
const WORK_PAGE: usize = 1_024;
/// Four in tests, so every worker test runs through many passes and rounds.
#[cfg(test)]
const WORK_PAGE: usize = 4;

/// What one pass did.
#[derive(Debug, Default)]
struct PassTotals {
    /// Entries carried out and completed.
    indexed: usize,
    /// Entries that failed (their retry count was raised).
    failed: usize,
    /// Entries rewritten to what their document holds, for the next round.
    rewritten: usize,
    /// Entries skipped because their document had a write in progress.
    in_progress: usize,
    /// Entries at `max_retries`, left for an operator.
    exhausted: usize,
    /// Whether anything failed.
    any_failed: bool,
    /// The worker is shutting down.
    stopped: bool,
}

impl PassTotals {
    fn add(&mut self, pass: &PassTotals) {
        self.indexed += pass.indexed;
        self.failed += pass.failed;
        self.rewritten += pass.rewritten;
        self.in_progress += pass.in_progress;
        self.exhausted += pass.exhausted;
        self.any_failed |= pass.any_failed;
    }

    /// Entries whose state changed: what makes another round worth running.
    fn moved(&self) -> usize {
        self.indexed + self.failed + self.rewritten
    }
}

impl<H: WorkerHost> VecIndexWorker<H> {
    /// Carry out every actionable queue entry, in rounds, until a round changes
    /// nothing.
    ///
    /// A round walks each queued namespace's entries from the start, a page per
    /// pass ([`run_pass`](Self::run_pass)), so every pass holds at most
    /// [`WORK_PAGE`] entries of each namespace and visits them round-robin: a
    /// large backlog in one namespace cannot delay the others. A namespace that
    /// is queued for the first time during a round joins it at its start. Entries
    /// whose `retry_count` has reached `config.max_retries` are passed over and
    /// left in the queue for an operator. After a round with any failure the
    /// worker sleeps `config.retry_wait_secs`.
    ///
    /// Another round runs while the last one changed something, so entries that
    /// arrived behind a cursor, and entries rewritten to match their document,
    /// are picked up without waiting for the next wake-up. A round that only
    /// skipped entries whose document has a write in progress ends the drain:
    /// running again would spin until those writes finish, and each one wakes the
    /// worker when it does (its `InFlightWrite` drop).
    ///
    /// Each round logs its start and end at `INFO`, so progress through a large
    /// queue (after a bulk load, say) is visible without `DEBUG`.
    pub(super) async fn drain_queue(&self) {
        loop {
            let depth = match vector_kv::count_queue_entries(&self.db).await {
                Ok(n) => n,
                Err(e) => {
                    warn!("vec index worker: queue count failed: {e}");
                    return;
                }
            };
            if depth == 0 {
                return;
            }
            info!("vec index worker: round start — depth={depth}");

            let mut round = PassTotals::default();
            // Each queued namespace's next key this round; `None` once walked to the end.
            let mut cursors: BTreeMap<String, Option<Vec<u8>>> = BTreeMap::new();
            loop {
                let namespaces = match vector_kv::queued_namespaces(&self.db).await {
                    Ok(n) => n,
                    Err(e) => {
                        warn!("vec index worker: queue scan failed: {e}");
                        return;
                    }
                };
                let mut entries = Vec::new();
                let mut read_any = false;
                for ns in namespaces {
                    let prefix = vector_kv::queue_prefix(&ns);
                    let cursor = cursors.entry(ns).or_insert_with(|| Some(prefix.clone()));
                    let Some(from) = cursor.take() else { continue };
                    let end = crate::support::prefix_upper_bound(&prefix);
                    match vector_kv::queue_page(&self.db, from, end, WORK_PAGE).await {
                        Ok((page, next)) => {
                            entries.extend(page);
                            *cursor = next;
                            read_any = true;
                        }
                        Err(e) => {
                            warn!("vec index worker: queue scan failed: {e}");
                            return;
                        }
                    }
                }
                if !read_any {
                    break;
                }
                if entries.is_empty() {
                    continue;
                }
                let pass = self.run_pass(entries).await;
                if pass.stopped {
                    return;
                }
                round.add(&pass);
            }

            info!(
                "vec index worker: round complete — indexed={} failed={} rewritten={} in_progress={}",
                round.indexed, round.failed, round.rewritten, round.in_progress,
            );
            if round.exhausted > 0 {
                warn!(
                    "vec index worker: {} entry/entries have reached max_retries={} and are \
                     awaiting manual removal via the admin API",
                    round.exhausted, self.config.max_retries,
                );
            }

            if round.any_failed {
                tokio::time::sleep(Duration::from_secs(self.config.retry_wait_secs)).await;
            }
            if self.shutdown.load(Ordering::Acquire) {
                return;
            }
            if round.moved() == 0 {
                return;
            }
        }
    }

    /// Carry out one pass's entries (at most a page of each namespace): discard
    /// entries of namespaces that no longer exist, pass over exhausted ones, and
    /// run the rest round-robin across namespaces, up to `config.concurrency`
    /// at once. Entries written in the pass are made durable and completed before
    /// it returns.
    async fn run_pass(&self, all_entries: Vec<QueueEntry>) -> PassTotals {
        // Entries whose namespace no longer exists must be discarded, not
        // embedded.
        //
        // Store deletion and the queue are separate writes, so a delete that
        // races this worker — or a crash between the two — can strand entries
        // for a namespace that is gone. Nothing would ever complete them.
        let live_namespaces: std::collections::HashSet<String> = self.db.list_namespaces().into_iter().map(|(name, _)| name).collect();

        // Separate entries at max retries (leave in queue, admin must clear).
        let mut exhausted_count = 0usize;
        let mut orphaned_count = 0usize;
        let mut by_namespace: BTreeMap<String, VecDeque<QueueEntry>> = BTreeMap::new();

        for entry in all_entries {
            if !live_namespaces.contains(&entry.namespace) {
                orphaned_count += 1;
                if let Err(e) = vector_kv::discard_orphaned_entry(&self.db, &entry).await {
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

        debug!(
            "vec index worker: pass — actionable={actionable_count} exhausted={exhausted_count} namespaces={}",
            by_namespace.len(),
        );

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
            let resolved = match self.host.vector_settings(ns) {
                Ok((settings, ns_id)) => self
                    .ctx
                    .for_namespace(&self.db, ns, ns_id, &settings)
                    .await
                    .map(Arc::new)
                    .map_err(WorkerError::from),
                Err(e) => Err(e),
            }
            .map_err(|e| e.to_string());
            if let Err(e) = &resolved {
                if self.ctx.is_dropping(ns) || !self.host.store_exists(ns) {
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
            return PassTotals {
                exhausted: exhausted_count,
                ..PassTotals::default()
            };
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
        let mut set: JoinSet<(QueueEntry, Result<Processed, WorkerError>)> = JoinSet::new();
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
                return PassTotals {
                    stopped: true,
                    ..PassTotals::default()
                };
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

        debug!(
            "vec index worker: pass complete — indexed={indexed_count} failed={failed_count} \
             rewritten={rewritten_count} in_progress={in_progress_count}"
        );
        PassTotals {
            indexed: indexed_count,
            failed: failed_count,
            rewritten: rewritten_count,
            in_progress: in_progress_count,
            exhausted: exhausted_count,
            any_failed,
            stopped: false,
        }
    }
}
