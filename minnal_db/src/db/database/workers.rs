//! What the background workers call: GC, compaction, and the `*Target` trait impls.

use super::*;

impl Database {
    /// Run value log GC on a specific namespace.
    ///
    /// Note the threshold passed here is the **per-segment** selection one, not the
    /// bucket-level trigger: an explicit `garbage_collect()` call has already decided
    /// the namespace is worth collecting, so what is left to decide is which *segments*
    /// to rewrite.
    pub fn garbage_collect_namespace(&self, namespace_id: u32) -> Result<GCStats> {
        let kv_store = self.get_store(namespace_id)?;
        let segment_threshold = self.config.threshold_config.segment_gc_threshold;
        let tail_threshold = self.config.threshold_config.effective_tail_gc_min_garbage_pct();
        kv_store.garbage_collect_with_thresholds(segment_threshold, tail_threshold)
    }

    /// Run value log GC on the default namespace
    pub fn garbage_collect(&self) -> Result<GCStats> {
        self.garbage_collect_namespace(DEFAULT_NAMESPACE_ID)
    }

    pub fn compact_lsm(&self) -> Result<()> {
        self.check_closed()?;
        let stores = self.stores.read();
        for kv_store in stores.values() {
            kv_store.compact_lsm()?;
        }
        Ok(())
    }

    pub fn has_lsm_compaction_work(&self) -> bool {
        let stores = self.stores.read();
        stores.values().any(|s| s.has_lsm_compaction_work())
    }

    /// What the compaction worker runs: every namespace's read-only memtables
    /// flushed, and only the buckets whose L0→L1 merge is due merged
    /// (`LSMTree::compaction_due`). [`compact_lsm`](Self::compact_lsm) merges
    /// everything.
    pub(crate) fn compact_lsm_due(&self) -> Result<()> {
        self.check_closed()?;
        let stores = self.stores.read();
        for kv_store in stores.values() {
            kv_store.compact_lsm_due()?;
        }
        Ok(())
    }

    pub(crate) fn has_due_lsm_compaction_work(&self) -> bool {
        let stores = self.stores.read();
        stores.values().any(|s| s.has_due_lsm_compaction_work())
    }

    /// Run value-log GC on every namespace and return per-namespace results.
    pub fn garbage_collect_all(&self) -> Vec<(String, GCStats)> {
        self.list_namespaces()
            .into_iter()
            .filter_map(|(name, ns_id)| self.garbage_collect_namespace(ns_id).ok().map(|stats| (name, stats)))
            .collect()
    }
}

// ── IndexCheckpointTarget impl ────────────────────────────────────────────

impl IndexCheckpointTarget for Database {
    fn is_closed(&self) -> bool {
        self.is_closed()
    }

    fn run_index_checkpoint(&self) -> Result<usize> {
        // The offset this checkpoint claims the indices reflect. NOT the raw tail:
        // a write registers as in flight when it appends to the WAL and stays so
        // until its store apply *and index update* are done, so the tail can hold
        // an entry the index has not seen. Recording that tail let a crash skip
        // the entry at replay — the document stayed out of every field index,
        // with no gap recorded (caught by the SIGKILL stress harness: 1–2 docs per
        // crash). `wal_cut_ceiling` stops below the oldest in-flight write, so
        // everything under it is applied before the flushes below run.
        let wal_tail = self.wal_flush_observer.wal_cut_ceiling();
        let fields = self.registry.read().all_indexed_fields();

        // Flush mmap bitmap data for each active field index, and collect only
        // those that are still active (dropped fields are deregistered from
        // namespace_index but remain in the registry schema).
        let waste_threshold = (self.config.threshold_config.index_blob_waste_threshold / 100.0).clamp(0.0, 1.0);
        let mut active_fields = Vec::with_capacity(fields.len());
        let stores = self.stores.read();

        // Flush each namespace's dense row map FIRST and advance its marker, so
        // the map is at least as durable as every field bitmap that references
        // its IDs (a crash must never leave a persisted bit whose row ID is not
        // reproducible from the row map). See `crate::field_index::RowMap` durability docs.
        let mut flushed_ns = std::collections::HashSet::new();
        for &(ns_id, _) in &fields {
            if flushed_ns.insert(ns_id)
                && let Some(store) = stores.get(&ns_id)
            {
                store.flush_rowmap(wal_tail)?;
            }
        }

        for &(ns_id, field_id) in &fields {
            if let Some(store) = stores.get(&ns_id) {
                let ns_index = store.namespace_index.read();
                if let Some(entry) = ns_index.get(field_id) {
                    let field_path = self.index_manager.field_path(ns_id, field_id);
                    // Write the field's buffered changes out, crash-ordered and
                    // with no fsync under the write lock:
                    //   1. stage  (write lock): append the changed blobs; no slot
                    //      points at them yet, reads keep using the overlay;
                    //   2. sync the value regions (read lock);
                    //   3. commit (write lock): point the slots at the blobs;
                    //   4. sync the key tables (read lock, below).
                    // A slot must never reach disk before its blob, and the kernel
                    // writes a shared map back in any order, hence the split.
                    // Every write below `wal_tail` finished its index update
                    // before `wal_tail` was taken (see above), so it is staged.
                    let stage = entry.index.write().stage();
                    if let Err(e) = entry.index.read().sync_values() {
                        entry.index.write().abort(stage);
                        return Err(KVError::Io(e));
                    }
                    entry.index.write().commit(stage);
                    // Sync under a read lock, then check waste cheaply. Only take
                    // the write lock (which serialises with index writers) when the
                    // bitmap store OR the keymap store has crossed the compaction
                    // threshold (the keymap accumulates dead space under
                    // distinct-value churn).
                    let (over_threshold, stats) = {
                        let idx = entry.index.read();
                        idx.sync_keys().map_err(KVError::Io)?;
                        let stats = idx.blob_stats();
                        let over = stats.bitmap_waste_ratio >= waste_threshold || stats.keymap_waste_ratio >= waste_threshold;
                        (over, stats)
                    };
                    // Guardrail: every spill appends each changed container and a
                    // new directory and leaves the old copies behind, so a field
                    // that spills often (hard-limit spills between checkpoints)
                    // piles up dead space (see index/CLAUDE.md). The compaction below
                    // reclaims it, but warn when a field's bitmap blob has grown
                    // large with a small live footprint so operators can spot
                    // runaway growth between checkpoints.
                    const LARGE_BITMAP_LOGICAL_BYTES: u64 = 64 * 1024 * 1024;
                    if stats.bitmap_logical_bytes >= LARGE_BITMAP_LOGICAL_BYTES && stats.bitmap_waste_ratio >= 0.5 {
                        warn!(
                            "[IndexCheckpoint] ns={ns_id} field={field_id}: bitmap blob logical={} MiB live={} MiB \
                             waste={:.0}% across {} distinct value(s) — old bitmap copies left by spills (likely \
                             large bitmaps spilling often at the overlay hard limit); compaction will reclaim it now, but review \
                             the field's update rate and the index_overlay_*_bytes limits",
                            stats.bitmap_logical_bytes / (1024 * 1024),
                            stats.bitmap_live_bytes / (1024 * 1024),
                            stats.bitmap_waste_ratio * 100.0,
                            stats.distinct_values,
                        );
                    }
                    if over_threshold {
                        let mut idx = entry.index.write();
                        if idx.maybe_compact(waste_threshold).map_err(KVError::Io)? {
                            idx.flush(&field_path).map_err(KVError::Io)?;
                            debug!("[IndexCheckpoint] compacted field-index stores ns={ns_id} field={field_id}");
                        }
                    }
                    active_fields.push((ns_id, field_id));
                }
            }
        }

        self.index_manager.checkpoint_fields(wal_tail, &active_fields)?;
        drop(stores);
        // Their markers are durable: the *checkpointed* event.
        self.wal_flush_observer.fields_checkpointed(&active_fields, wal_tail);

        // The index is now durable up to `wal_tail`, so any no-WAL writes before
        // this point are covered and their markers can go.
        for ns_id in self.no_wal_pending.lock().drain() {
            if let Err(e) = self.index_manager.clear_no_wal_pending(ns_id) {
                warn!("[IndexCheckpoint] failed to clear the no-WAL marker for ns={ns_id}: {e:?}");
            }
        }

        self.persist_rejected_update_gaps();
        Ok(active_fields.len())
    }
}

// ── WalGcTarget impl ──────────────────────────────────────────────────

impl WalGcTarget for Database {
    fn is_closed(&self) -> bool {
        self.is_closed()
    }

    fn flush_namespaces_pinning_wal(&self) -> usize {
        self.flush_namespaces_pinning_wal()
    }

    fn wal_pending_bytes(&self) -> u64 {
        self.wal_pending_bytes()
    }

    fn has_deletable_wal_segments(&self) -> bool {
        self.has_deletable_wal_segments()
    }

    fn garbage_collect_wal(&self) -> Result<(u64, u64)> {
        self.garbage_collect_wal()
    }

    fn flush_namespaces_pinning_vector_wal(&self) -> usize {
        self.flush_namespaces_pinning_vector_wal()
    }

    fn has_deletable_vector_wal_segments(&self) -> bool {
        self.has_deletable_vector_wal_segments()
    }

    fn garbage_collect_vector_wal(&self) -> Result<(u64, u64)> {
        self.garbage_collect_vector_wal()
    }
}

// ── LsmCompactionTarget impl ──────────────────────────────────────────

impl LsmCompactionTarget for Database {
    fn is_closed(&self) -> bool {
        self.is_closed()
    }

    fn flush_no_wal_memtables(&self) -> usize {
        self.flush_no_wal_memtables()
    }

    fn has_due_compaction_work(&self) -> bool {
        self.has_due_lsm_compaction_work()
    }

    fn compact_due(&self) -> Result<()> {
        self.compact_lsm_due()
    }
}

// ── ValueLogGcTarget impl ─────────────────────────────────────────────

impl ValueLogGcTarget for Database {
    fn is_closed(&self) -> bool {
        self.is_closed()
    }

    /// `waste_threshold` decides **whether** a namespace is collected; the separate,
    /// lower `segment_gc_threshold` then decides **which sealed segments** get rewritten.
    /// Passing the trigger through as the selection threshold (as this used to) makes
    /// garbage sitting just under it uncollectable: those segments read as "clean", are
    /// left in place with their garbage intact, and keep the bucket over its trigger
    /// forever.
    fn run_gc_if_needed(&self, waste_threshold: f64) {
        let stores = self.stores.read();
        let segment_threshold = self.config.threshold_config.segment_gc_threshold;
        let tail_threshold = self.config.threshold_config.effective_tail_gc_min_garbage_pct();
        // Fires on every tick whether or not there is anything to collect, so
        // DEBUG. The per-namespace results below stay at INFO.
        debug!(
            "[GCWorker] tick — checking {} namespace(s) against {:.2}% waste threshold \
             (segments rewritten at >= {:.2}% garbage, tail sealed at >= {:.2}%)",
            stores.len(),
            waste_threshold,
            segment_threshold,
            tail_threshold
        );
        for (ns_id, kv_store) in stores.iter() {
            let waste_ratio = kv_store.get_waste_ratio();
            // The namespace-average waste is over the trigger, OR some single bucket is —
            // the average alone hides a hot bucket behind near-empty ones, so a bucket well
            // over the trigger would never be collected. Check both.
            if waste_ratio < waste_threshold && !kv_store.has_bucket_over_waste(waste_threshold) {
                debug!(
                    "[GCWorker] ns_id={} waste {:.2}% below threshold (no bucket over it), skipping",
                    ns_id, waste_ratio
                );
                continue;
            }
            // Over the trigger, but is any of that garbage actually collectable? Garbage in
            // a bucket's active tail is not, until the tail is either filled or sealed — so
            // without this check a small, fully-deleted namespace would sit at 100% waste
            // and make the worker log "starting GC ... reclaimed 0 bytes" on every tick.
            if !kv_store.has_gc_work(segment_threshold, tail_threshold) {
                debug!(
                    "[GCWorker] ns_id={} waste {:.2}% is over the trigger but no segment is collectable yet, skipping",
                    ns_id, waste_ratio
                );
                continue;
            }
            let (garbage_bytes, written_bytes) = kv_store.waste_bytes();
            info!(
                "[GCWorker] ns_id={} waste {:.2}% ({} garbage / {} written bytes; waste = garbage / (live + garbage)) \
                 exceeds threshold {:.2}%, starting GC",
                ns_id, waste_ratio, garbage_bytes, written_bytes, waste_threshold
            );
            let start = std::time::Instant::now();
            match kv_store.garbage_collect_with_thresholds(segment_threshold, tail_threshold) {
                Ok(stats) => info!(
                    "[GCWorker] ns_id={} GC complete in {:?} — reclaimed {} bytes, live {} bytes, \
                     total reclaimed {} bytes across {} run(s)",
                    ns_id,
                    start.elapsed(),
                    stats.bytes_reclaimed,
                    stats.bytes_live,
                    stats.total_bytes_reclaimed,
                    stats.gc_run_count,
                ),
                Err(e) => error!("[GCWorker] ns_id={} GC failed: {:?}", ns_id, e),
            }
        }
    }
}

// ── TtlTarget impl ─────────────────────────────────────────────────────

impl TtlTarget for Database {
    fn is_closed(&self) -> bool {
        self.is_closed()
    }

    fn run_ttl_pass(&self) {
        // Snapshot the durable TTL config so the (brief) scan/delete work doesn't
        // hold the registry lock; new registrations take effect on the next pass.
        let ttl_configs = self.registry.read().ttl_configs();
        info!("[TtlWorker] tick — scanning {} TTL-enabled namespace(s)", ttl_configs.len());
        for (ns_id, (ttl, max_deletes)) in ttl_configs {
            // The namespace may have been dropped between snapshot and lookup.
            let Some(store) = self.stores.read().get(&ns_id).cloned() else {
                continue;
            };
            // The key stripe keeps expiry out of every same-key write's
            // seq-to-apply window (see `KVStore::expire_records_locked`).
            match store.expire_records_locked(ttl, max_deletes, |key| self.key_locks.guard(ns_id, key)) {
                Ok(deleted) => {
                    if deleted > 0 {
                        info!("[TtlWorker] ns_id={} expired {} record(s)", ns_id, deleted);
                    } else {
                        debug!("[TtlWorker] ns_id={} no records expired", ns_id);
                    }
                }
                Err(e) => warn!("[TtlWorker] ns_id={} TTL cleanup failed: {:?}", ns_id, e),
            }
        }
    }
}
