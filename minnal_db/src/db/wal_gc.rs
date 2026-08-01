//! WAL garbage collection, gated on the index-replay watermark.
//!
//! Split out of `database.rs`. WAL GC must not reclaim a segment an active field
//! index still needs to replay after a crash -- without that gate the index
//! silently loses those documents for the life of the database (FR-001).
//!
//! Three properties here are load-bearing; see `minnal_db/CLAUDE.md`:
//! the watermark is scoped to *active* fields, the pin drains by asking the
//! checkpoint worker rather than by waiting for its timer, and the backstop
//! produces gaps by design -- which is why the harvest in `record_backstop_gaps`
//! has to run *before* a segment is unlinked.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

use log::{debug, error, warn};

use crate::db::database::Database;
use crate::db::error::Result;
use crate::db::namespace::FieldId;
use crate::db::wal::WalMetadata;

/// What one WAL GC pass may reclaim, after the index-replay watermark has had
/// its say. Produced by `Database::plan_wal_gc`.
#[derive(Debug, Default, PartialEq, Eq)]
struct WalGcPlan {
    /// Segments to delete this pass, ascending. Includes any the backstop forced.
    reclaim: Vec<u64>,
    /// Segments held back by the watermark and left in place.
    deferred: usize,
    /// Segments the watermark wanted to hold but the backstop reclaimed anyway,
    /// ascending. Non-empty means a field index has been knowingly stranded —
    /// these are the segments whose keys must be harvested before the unlink.
    forced: Vec<u64>,
}

impl Database {
    pub fn get_wal_gc_stats(&self) -> (u64, u64) {
        let wal_metadata = self.wal_metadata.read();
        (wal_metadata.total_entries, wal_metadata.persisted_entries)
    }

    /// Returns `true` when at least one non-current WAL segment is fully persisted
    /// and ready to be deleted.  Entries in the active segment are not yet eligible
    /// — they will be marked persisted after the next memtable flush or clean shutdown.
    ///
    /// Accounts for the index-replay watermark, so the worker does not call GC
    /// every tick for segments GC would only refuse. It deliberately still
    /// reports `true` when the backstop is due to fire (`forced > 0`) — otherwise
    /// a wedged checkpoint worker would pin the WAL *and* suppress the very GC
    /// pass meant to bound it.
    pub fn has_deletable_wal_segments(&self) -> bool {
        let watermark = self.index_replay_watermark_segment();
        let wal_metadata = self.wal_metadata.read();
        if wal_metadata.tail == 0 {
            return false;
        }
        let current = self.wal.segment_id_for_offset(wal_metadata.tail - 1);
        let plan = self.plan_wal_gc(&wal_metadata, current, watermark);
        !plan.reclaim.is_empty()
    }

    /// The oldest WAL segment any **currently active** field index still needs in
    /// order to replay after a crash, or `None` when nothing is pinned.
    ///
    /// `min(checkpoint offset)` over active fields, mapped to its segment. Each
    /// field already records its replay offset in its `checkpoint` marker, so
    /// this needs no new per-entry or per-segment state — just one marker read
    /// per active field, on a path that runs every 60 s.
    ///
    /// **Scoped to active fields on purpose.** Two kinds of inactive field would
    /// otherwise pin the WAL without bound:
    ///
    /// - a **dropped** field, whose marker is frozen wherever it was last
    ///   checkpointed. Step 1 of FR-001 deletes those markers, but a field can be
    ///   inactive without being dropped (e.g. while rebuilding) and cleanup can
    ///   itself be interrupted, so the filter stays;
    /// - a field **being built for the first time**, which has no marker at all
    ///   and so reads as "replay from 0". A new index is populated by a build,
    ///   not by replay, so it is suppressed here exactly as `detect_replay_gap`
    ///   suppresses it — otherwise every `add_index` on an established database
    ///   would pin the entire WAL.
    pub(crate) fn index_replay_watermark_segment(&self) -> Option<u64> {
        let wal_tail = self.wal_metadata.read().tail;
        if wal_tail == 0 {
            return None;
        }
        let fields = self.registry.read().all_indexed_fields();
        if fields.is_empty() {
            return None;
        }

        let stores = self.stores.read();
        let mut watermark: Option<u64> = None;
        for (ns_id, field_id) in fields {
            let Some(store) = stores.get(&ns_id) else { continue };
            let ns_index = store.namespace_index.read();
            // Not activated → not replayed into → nothing to pin for.
            let Some(entry) = ns_index.get(field_id) else { continue };

            let state = self.index_manager.read_checkpoint_state(ns_id, field_id, wal_tail);
            if state == crate::db::index_manager::CheckpointState::Absent && entry.index.read().distinct_count() == 0 {
                continue;
            }
            let segment = self.wal.segment_id_for_offset(state.replay_offset());
            watermark = Some(watermark.map_or(segment, |w: u64| w.min(segment)));
        }
        watermark
    }

    /// Harvest the keys held by segments the backstop is about to reclaim, and
    /// record a durable gap for every active field index that needed them.
    ///
    /// **Must be called before the segment files are unlinked.** The affected
    /// keys live only in those files: a gap record holding a WAL range and
    /// segment ids cannot name them, and detection at the next open — where the
    /// original `detect_replay_gap` runs — is permanently too late, because by
    /// then the evidence is gone. This is the single moment row-scoped repair is
    /// possible at all.
    ///
    /// The WAL is shared across namespaces, so a segment's keys are attributed by
    /// [`WalEntry::namespace_id`](crate::db::wal::WalEntry) to each `(ns, field)`
    /// whose checkpoint sits at or below the reclaimed segment. A field already
    /// checkpointed past a segment did not need it and gets no gap.
    ///
    /// Best-effort: a failure here is logged, not propagated. Losing the record
    /// is bad, but aborting GC would leave the WAL growing without bound — the
    /// very outage the backstop exists to prevent.
    fn record_backstop_gaps(&self, forced: &[u64], wal_tail: u64) {
        // Which active fields needed each reclaimed segment, by checkpoint offset.
        let mut affected: Vec<(u32, FieldId, u64)> = Vec::new();
        {
            let fields = self.registry.read().all_indexed_fields();
            let stores = self.stores.read();
            for (ns_id, field_id) in fields {
                let Some(store) = stores.get(&ns_id) else { continue };
                let ns_index = store.namespace_index.read();
                let Some(entry) = ns_index.get(field_id) else { continue };
                let state = self.index_manager.read_checkpoint_state(ns_id, field_id, wal_tail);
                // Same suppression as the watermark: a never-checkpointed empty
                // index is populated by a build, not by replay, so it loses
                // nothing when a segment goes.
                if state == crate::db::index_manager::CheckpointState::Absent && entry.index.read().distinct_count() == 0 {
                    continue;
                }
                affected.push((ns_id, field_id, self.wal.segment_id_for_offset(state.replay_offset())));
            }
        }
        if affected.is_empty() {
            return;
        }

        // Harvest once per segment; the same keys usually serve several fields.
        let mut keys_by_ns: HashMap<u32, Vec<Vec<u8>>> = HashMap::new();
        for &segment_id in forced {
            match self.wal.scan_segment_keys(segment_id) {
                Ok(pairs) => {
                    for (ns_id, key) in pairs {
                        keys_by_ns.entry(ns_id).or_default().push(key);
                    }
                }
                Err(e) => warn!("[WAL GC] could not harvest keys from segment {segment_id} before reclaiming it: {e:?}"),
            }
        }

        let cap = Self::GAP_KEY_WORKLIST_CAP;
        let detected_at_ms = crate::db::kv_store::current_epoch_millis();
        for (ns_id, field_id, field_segment) in affected {
            // A field checkpointed past every reclaimed segment lost nothing.
            if !forced.iter().any(|&s| s >= field_segment) {
                continue;
            }
            let keys = keys_by_ns.get(&ns_id).cloned().unwrap_or_default();
            // Past the cap the worklist is dropped and the field is marked for a
            // full rebuild — a wedged checkpoint worker could otherwise strand a
            // key set large enough to be its own problem.
            let repair = if keys.is_empty() || keys.len() > cap {
                crate::db::index_manager::RepairMode::FullRebuild
            } else {
                crate::db::index_manager::RepairMode::RowScoped {
                    keys: keys.iter().map(|k| crate::support::hex::bytes_to_hex(k)).collect(),
                }
            };
            let gap = crate::db::index_manager::GapRecord {
                namespace_id: ns_id,
                field_id,
                cause: crate::db::index_manager::GapCause::BackstopReclaim,
                from: self.index_manager.read_checkpoint(ns_id, field_id, wal_tail),
                to: wal_tail,
                missing_segments: forced.to_vec(),
                detected_at_ms,
                repair,
            };
            if let Err(e) = self.index_manager.record_gap(gap, cap) {
                warn!("[WAL GC] failed to record index gap for ns={ns_id} field={field_id}: {e:?}");
            }
        }
    }

    /// Decide which fully-persisted, non-current WAL segments this GC pass may
    /// reclaim, given the index-replay watermark.
    ///
    /// Pure and side-effect free so `has_deletable_wal_segments` and
    /// `garbage_collect_wal` cannot disagree about what is reclaimable — if they
    /// did, the worker would either spin on segments GC refuses or skip the pass
    /// that is meant to fire the backstop.
    fn plan_wal_gc(&self, wal_metadata: &WalMetadata, current_segment_id: u64, watermark: Option<u64>) -> WalGcPlan {
        let ready: Vec<u64> = wal_metadata
            .tracked_segments()
            .take_while(|&s| s < current_segment_id)
            .filter(|&s| {
                let total = wal_metadata.segment_total(s);
                // `>=` rather than `==`: a fully-persisted segment has
                // persisted == total; accepting `>` too keeps GC unwedged if a
                // persisted counter is ever over-reported, instead of stranding
                // the segment forever.
                total > 0 && wal_metadata.segment_persisted(s) >= total
            })
            .collect();

        let Some(watermark) = watermark else {
            return WalGcPlan {
                reclaim: ready,
                deferred: 0,
                forced: Vec::new(),
            };
        };

        let (pinned, free): (Vec<u64>, Vec<u64>) = ready.into_iter().partition(|&s| s >= watermark);

        // Backstop: past the cap, reclaim the oldest pinned segments anyway
        // rather than let a wedged checkpoint worker grow the WAL without bound.
        // `tracked_segments` is ascending, so `partition` keeps `pinned` ascending
        // and `take` picks the oldest. This knowingly strands part of a field
        // index — see `ThresholdConfig::max_pinned_wal_segments`.
        let cap = self.config.threshold_config.max_pinned_wal_segments as usize;
        let force_count = if cap == 0 { 0 } else { pinned.len().saturating_sub(cap) };
        let forced: Vec<u64> = pinned.iter().take(force_count).copied().collect();

        let mut reclaim = free;
        reclaim.extend(forced.iter().copied());
        reclaim.sort_unstable();

        WalGcPlan {
            reclaim,
            deferred: pinned.len() - force_count,
            forced,
        }
    }

    pub fn garbage_collect_wal(&self) -> Result<(u64, u64)> {
        if self
            .wal_gc_in_progress
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            let (total, persisted) = self.get_wal_gc_stats();
            return Ok((0, total.saturating_sub(persisted)));
        }

        struct WalGcGuard<'a>(&'a AtomicBool);
        impl<'a> Drop for WalGcGuard<'a> {
            fn drop(&mut self) {
                self.0.store(false, Ordering::SeqCst);
            }
        }
        let _guard = WalGcGuard(&self.wal_gc_in_progress);

        // Compute the watermark BEFORE taking the metadata write lock: it reads
        // the registry, the stores and each active field's checkpoint marker, and
        // holding the WAL metadata lock across all of that would widen the
        // window against every writer for no benefit. A tail that advances in
        // between only makes the watermark more conservative (a stale, smaller
        // tail can make a marker read as `Unusable`, which pins from 0), never
        // less.
        let watermark = self.index_replay_watermark_segment();

        let mut wal_metadata = self.wal_metadata.write();
        let current_segment_id = if wal_metadata.tail == 0 {
            0
        } else {
            self.wal.segment_id_for_offset(wal_metadata.tail - 1)
        };

        // Persist the current sequence high-water mark BEFORE deleting any segment.
        // This ensures recover_sequence always finds a hint >= the max sequence
        // in any segment that survives, even if those segments are later deleted.
        wal_metadata.last_sequence = self.next_seq.load(Ordering::Relaxed).saturating_sub(1);

        let mut bytes_reclaimed = 0u64;
        let mut segments_deleted = 0u64;
        let plan = self.plan_wal_gc(&wal_metadata, current_segment_id, watermark);
        if !plan.forced.is_empty() {
            // The backstop is about to reclaim segments at least one active field
            // index still needed. Harvest the affected keys FIRST: they exist only
            // in these files, and once unlinked no gap record could ever name
            // them. This is the only moment row-scoped repair is possible.
            error!(
                "[WAL GC] index-replay watermark held {} segment(s), over the cap of {} — reclaiming the {} oldest ANYWAY. \
                 Field indices covering those segments are now INCOMPLETE. Is the index checkpoint worker running?",
                plan.deferred + plan.forced.len(),
                self.config.threshold_config.max_pinned_wal_segments,
                plan.forced.len(),
            );
            let wal_tail = wal_metadata.tail;
            self.record_backstop_gaps(&plan.forced, wal_tail);
        }
        for segment_id in plan.reclaim {
            let total = wal_metadata.segment_total(segment_id);
            let persisted = wal_metadata.segment_persisted(segment_id);
            if self.wal.delete_segment_file(segment_id).is_ok() {
                bytes_reclaimed = bytes_reclaimed.saturating_add(self.wal.segment_size());
                segments_deleted += 1;
                wal_metadata.total_entries = wal_metadata.total_entries.saturating_sub(total);
                wal_metadata.persisted_entries = wal_metadata.persisted_entries.saturating_sub(persisted);
                wal_metadata.clear_segment(segment_id);
            }
        }

        // Advance head past all consecutively deleted/empty segments so that a
        // subsequent scan_entries(head, tail) never tries to open a deleted file.
        // Segments with segment_total == 0 were either deleted in this run or in
        // a previous one; either way their files are gone.
        let head_segment = wal_metadata.head / self.wal.segment_size();
        let first_live = (head_segment..=current_segment_id)
            .find(|&sid| sid == current_segment_id || wal_metadata.segment_total(sid) > 0)
            .unwrap_or(current_segment_id);
        let new_head = first_live * self.wal.segment_size();
        if new_head > wal_metadata.head {
            wal_metadata.head = new_head;
        }
        // Trim per-segment counters below the new head so the dense vecs track
        // only the live segment window instead of growing with every segment
        // ever created (keeps `base_segment_id == head`'s segment).
        wal_metadata.trim_segments_before(first_live);

        wal_metadata.total_gc_runs = wal_metadata.total_gc_runs.saturating_add(1);
        wal_metadata.total_bytes_reclaimed = wal_metadata.total_bytes_reclaimed.saturating_add(bytes_reclaimed);
        let remaining = wal_metadata.total_entries.saturating_sub(wal_metadata.persisted_entries);
        drop(wal_metadata);
        self.flush_wal_metadata_internal()?;

        crate::db::metrics::Metrics::bump(&self.metrics.wal_gc_runs);
        crate::db::metrics::Metrics::add(&self.metrics.wal_segments_deleted, segments_deleted);

        // Liveness: the pin can only drain when a checkpoint advances the
        // fields' recorded offsets, so ask for one now and let the next GC tick
        // reclaim. Without this, retention would track the ~15 min checkpoint
        // timer instead of checkpoint latency. Debounced inside the trigger, and
        // deliberately uncapped — `request_if_over_cap` returns early when the
        // backpressure valve is disabled.
        if plan.deferred > 0 {
            debug!(
                "[WAL GC] {} segment(s) pinned by the index-replay watermark; requesting an index checkpoint",
                plan.deferred
            );
            if let Some(trigger) = self.index_checkpoint_trigger.read().as_ref() {
                trigger.request();
            }
        }

        Ok((bytes_reclaimed, remaining))
    }
}
