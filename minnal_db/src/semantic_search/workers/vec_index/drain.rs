//! One drain of the embed queue: the worker's pass over every entry it can act
//! on, grouped by namespace and run round-robin with bounded concurrency.

use super::*;

impl<H: WorkerHost> VecIndexWorker<H> {
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
    pub(super) async fn drain_queue(&self) {
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
}
