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
        // reclaim. Without this, retention would track the periodic checkpoint
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::index_checkpoint_worker::IndexCheckpointTarget;
    use crate::db::namespace::DEFAULT_NAMESPACE_ID;
    use crate::db::test_support::*;
    use crate::db::wal::WalEntry;
    use tempfile::TempDir;

    /// The flip side of the per-namespace watermark: an idle namespace holding
    /// un-flushed writes pins the WAL, so the GC worker must flush it. Without
    /// this the correctness fix above would trade data loss for unbounded WAL
    /// growth.
    #[test]
    fn test_wal_gc_flushes_namespaces_pinning_the_watermark() {
        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        let ns = db.create_namespace("idle").unwrap();
        for i in 0..10u32 {
            db.put_ns(ns, format!("k{i}").as_bytes(), b"v").unwrap();
        }

        assert_eq!(
            db.wal_flush_observer.namespaces_with_unflushed(),
            vec![ns],
            "a namespace with un-flushed WAL writes should be reported as pinning"
        );

        let flushed = db.flush_namespaces_pinning_wal();
        assert_eq!(flushed, 1, "the pinning namespace should have been flushed");
        assert!(
            db.wal_flush_observer.namespaces_with_unflushed().is_empty(),
            "nothing should still pin the watermark after the flush"
        );
    }

    /// A namespace that is DROPPED must stop constraining the global watermark.
    ///
    /// `remove_namespace` flushes and shuts the store down and marks that
    /// namespace's WAL entries persisted, but it leaves the namespace's entry in
    /// `ns_progress` with `last_write_offset > safe_offset` — so it is reported
    /// as pinning forever, and `flush_namespaces_pinning_wal` cannot clear it
    /// (the store is gone, so the loop skips it). The global cut is then frozen
    /// at the dead namespace's stale `safe_offset` and WAL GC never fires again.
    #[test]
    fn test_dropped_namespace_does_not_pin_the_wal_watermark() {
        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        let doomed = db.create_namespace("doomed").unwrap();
        let keeper = db.create_namespace("keeper").unwrap();

        // `doomed` writes and never fills its memtable, so it never flushes.
        for i in 0..10u32 {
            db.put_ns(doomed, format!("d{i}").as_bytes(), b"v").unwrap();
        }
        db.remove_namespace("doomed").unwrap();

        // `keeper` writes and flushes — everything in the WAL is now durable.
        for i in 0..10u32 {
            db.put_ns(keeper, format!("k{i}").as_bytes(), b"v").unwrap();
        }
        db.get_store(keeper).unwrap().flush_memtable_to_level0().unwrap();
        db.flush_namespaces_pinning_wal();

        let pinning = db.wal_flush_observer.namespaces_with_unflushed();
        let meta = db.wal_metadata.read();
        let (total, persisted) = (meta.total_entries, meta.persisted_entries);
        drop(meta);

        assert!(
            pinning.is_empty() && persisted == total,
            "a dropped namespace still pins the watermark: pinning={pinning:?}, \
             persisted={persisted} total={total} (WAL GC's gate is persisted >= total)"
        );
    }

    #[test]
    fn test_wal_gc_eligible_after_namespace_drop() {
        // Regression test: WAL entries for a dropped namespace must be marked
        // persisted on drop so that WAL GC can reclaim the segments they occupy.
        // Previously, removing a namespace left its WAL entries as Inserted forever,
        // blocking GC for any segment that contained those entries.
        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();

        let ns_id = db.create_namespace("temp").unwrap();
        db.put_ns(ns_id, b"k1", b"v1").unwrap();
        db.put_ns(ns_id, b"k2", b"v2").unwrap();

        let (total_before, persisted_before) = db.get_wal_gc_stats();
        assert!(total_before >= 2, "expected at least 2 WAL entries");

        db.remove_namespace("temp").unwrap();

        let (total_after, persisted_after) = db.get_wal_gc_stats();
        // After dropping the namespace, the persisted count must have increased
        // to cover the entries we just wrote, so GC is not blocked on them.
        assert!(
            persisted_after > persisted_before,
            "persisted count should increase after namespace drop (before={}, after={})",
            persisted_before,
            persisted_after,
        );
        assert_eq!(total_after, total_before, "total entry count should be unchanged");

        db.shutdown().unwrap();
    }

    #[test]
    fn test_database_wal_gc() {
        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();

        db.put(b"key", b"value").unwrap();

        let (total, _persisted) = db.get_wal_gc_stats();
        assert!(total >= 1);

        db.shutdown().unwrap();
    }

    #[test]
    fn test_wal_gc_deletes_fully_persisted_segments() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let db = Database::open_with_wal_segment_size(temp_dir.path(), create_db_config(), 256)?;

        for i in 0..120u32 {
            let key = format!("wal_gc_key_{:03}", i).into_bytes();
            let value = vec![b'v'; 64];
            db.put(&key, &value)?;
        }

        db.flush_all_namespaces()?;

        let segment_size = db.wal.segment_size();
        let mut tail = db.wal_metadata.read().tail;
        let mut remaining = segment_size.saturating_sub(tail % segment_size);
        if remaining == segment_size {
            db.put(b"wal_gc_pad", b"pad")?;
            db.flush_all_namespaces()?;
            tail = db.wal_metadata.read().tail;
            remaining = segment_size.saturating_sub(tail % segment_size);
        }

        let mut pad_value_len = 1usize;
        loop {
            let pad_entry = WalEntry::new_upsert(b"wal_gc_pad_crash".to_vec(), vec![0u8; pad_value_len]);
            let entry_len = 4u64 + pad_entry.to_bytes()?.len() as u64;
            if entry_len > remaining && entry_len < segment_size {
                db.simulate_crash_with_wal_entries(vec![pad_entry])?;
                break;
            }
            pad_value_len = pad_value_len.saturating_add(8);
        }

        let segment1_path = temp_dir.path().join("wal.log.seg000001");
        assert!(segment1_path.exists());

        let crash_entries = (0..3u32)
            .map(|i| WalEntry::new_upsert(format!("crash_{}", i).into_bytes(), vec![b'x'; 16]))
            .collect();
        db.simulate_crash_with_wal_entries(crash_entries)?;

        let (bytes_reclaimed, _) = db.garbage_collect_wal()?;
        assert!(bytes_reclaimed > 0);
        assert!(!segment1_path.exists());

        Ok(())
    }

    // Regression: the per-segment counter vecs must be trimmed as segments are
    // reclaimed, so the wal_metadata file stays proportional to the *live*
    // segment window instead of growing ~16 bytes for every segment ever created.
    #[test]
    fn test_wal_gc_trims_segment_counters_so_metadata_stays_bounded() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let db = Database::open_with_wal_segment_size(temp_dir.path(), create_db_config(), 256)?;

        let mut max_vec_len = 0usize;
        // Many write→persist→GC rounds create and reclaim far more segments than
        // are ever live at once.
        for round in 0..50u32 {
            for i in 0..10u32 {
                db.put(format!("k{}_{}", round, i).as_bytes(), &[b'v'; 32])?;
            }
            db.flush_all_namespaces()?;
            db.garbage_collect_wal()?;
            max_vec_len = max_vec_len.max(db.wal_metadata.read().segment_total_entries.len());
        }

        let m = db.wal_metadata.read();
        // The base advanced as old segments were reclaimed and trimmed...
        assert!(
            m.base_segment_id > 10,
            "base_segment_id should advance as segments are reclaimed, got {}",
            m.base_segment_id
        );
        // ...while the counter vecs stayed bounded by the live window rather than
        // growing to ~= total segments ever created (base + len).
        assert!(max_vec_len <= 8, "segment counter vec should stay bounded, peaked at {}", max_vec_len);
        assert_eq!(m.segment_total_entries.len(), m.segment_persisted_entries.len());
        drop(m);
        db.shutdown()?;
        Ok(())
    }

    #[test]
    fn test_wal_gc_shrinks_wal_directory_once_persisted() -> Result<()> {
        let temp_dir = TempDir::new()?;
        // Small segments so a modest number of writes fills several of them.
        let db = Database::open_with_wal_segment_size(temp_dir.path(), create_db_config(), 4096)?;

        for i in 0..400u32 {
            db.put(format!("wal_{:04}", i).as_bytes(), &[b'v'; 64])?;
        }
        // Persist the entries (flush memtables → SSTables), marking their WAL
        // segments fully persisted and therefore reclaimable.
        db.flush_all_namespaces()?;

        let wal_dir_bytes = |dir: &std::path::Path| -> u64 {
            std::fs::read_dir(dir)
                .unwrap()
                .flatten()
                .filter(|e| e.file_name().to_string_lossy().starts_with("wal.log.seg"))
                .map(|e| e.metadata().map(|m| m.len()).unwrap_or(0))
                .sum()
        };

        let before = wal_dir_bytes(temp_dir.path());
        assert!(before > 4096, "expected several WAL segments before GC, got {before} bytes");

        let (reclaimed, _) = db.garbage_collect_wal()?;
        assert!(reclaimed > 0, "WAL GC should reclaim fully-persisted segments");

        let after = wal_dir_bytes(temp_dir.path());
        assert!(after < before, "WAL directory should shrink: {before} -> {after} bytes");

        // Persisted data is still readable after the segments were deleted.
        assert_eq!(db.get(b"wal_0000")?, Some(vec![b'v'; 64]));
        Ok(())
    }

    /// The replay-gap condition is reachable through the ordinary write → persist
    /// → WAL-GC path, with nothing corrupted or hand-punched.
    ///
    /// WAL GC reclaims a segment once its entries are persisted to the **LSM**;
    /// it never consults the field-index checkpoint markers. So a field whose
    /// checkpoint still points into a reclaimed segment can no longer be repaired
    /// by replay — `scan_entries` skips the missing segments as holes and returns
    /// a partial result with no error, which is how this stayed invisible.
    ///
    /// This pins the *detectability* of that condition (the logging added at
    /// `activate_field_index` keys on exactly these two inputs), not a fix.
    /// Remediation is tracked in `FEATURE-REQUEST.md` (FR-001).
    #[test]
    fn wal_gc_can_strand_a_field_index_checkpoint_and_the_gap_is_detectable() -> Result<()> {
        use crate::db::index_manager::{CheckpointState, detect_replay_gap};

        let temp_dir = TempDir::new()?;
        let db = Database::open_with_wal_segment_size(temp_dir.path(), create_db_config(), 4096)?;

        // A field index checkpointed early: its marker records the WAL tail as of
        // now, which is what the crash-recovery replay would start from.
        for i in 0..40u32 {
            db.put(format!("gap_{:04}", i).as_bytes(), &[b'v'; 64])?;
        }
        let checkpoint_offset = db.wal_metadata.read().tail;
        assert!(checkpoint_offset > 0, "expected writes to have advanced the WAL tail");

        // More writes, then persist them to the LSM. That is all WAL GC looks at.
        for i in 40..400u32 {
            db.put(format!("gap_{:04}", i).as_bytes(), &[b'v'; 64])?;
        }
        db.flush_all_namespaces()?;

        let wal_tail = db.wal_metadata.read().tail;
        assert!(
            db.wal.segment_id_for_offset(wal_tail - 1) > 1,
            "test needs several WAL segments, tail is {wal_tail}"
        );
        assert!(
            db.wal.missing_segments(checkpoint_offset, wal_tail).is_empty(),
            "no segment should be missing before GC runs"
        );

        let (reclaimed, _) = db.garbage_collect_wal()?;
        assert!(reclaimed > 0, "WAL GC should have reclaimed fully-persisted segments");

        // The segments that would have healed a field index checkpointed at
        // `checkpoint_offset` are now gone — through nothing but normal operation.
        let missing = db.wal.missing_segments(checkpoint_offset, wal_tail);
        assert!(
            !missing.is_empty(),
            "WAL GC reclaimed past a field checkpoint at {checkpoint_offset}, so segments in \
             [{checkpoint_offset}, {wal_tail}) should be gone"
        );

        // ...and replay from that offset silently returns a partial result rather
        // than reporting the hole — the silence this detection exists to break.
        let wal_head = db.wal_metadata.read().head;
        let replayed = db.wal.scan_entries(wal_head.max(checkpoint_offset), wal_tail)?;
        // Observed: 10 segments reclaimed, head forced from the 4,671-byte
        // checkpoint up to 45,056, and only 15 of the ~360 writes in the window
        // still replayable. The other ~345 are gone from the field index for good.
        assert!(
            replayed.len() < 100,
            "replay should be short by the reclaimed segments' entries, got {} of ~360",
            replayed.len()
        );

        // Which is exactly what the detector reports on.
        let gap = detect_replay_gap(CheckpointState::At(checkpoint_offset), false, wal_tail, missing).expect("replay gap should be detected");
        assert_eq!(gap.from, checkpoint_offset);
        assert_eq!(gap.to, wal_tail);

        db.shutdown()?;
        Ok(())
    }

    /// FR-001 step 2, the inverse of the repro above: with a **real activated
    /// field index**, WAL GC must refuse to reclaim the segments that index still
    /// needs to replay, so no gap is detectable at all.
    ///
    /// The repro above works only because it has no active field index — the
    /// `checkpoint_offset` it strands is hypothetical. Here the index is genuinely
    /// active and checkpointed, which is what arms the watermark.
    #[test]
    fn wal_gc_does_not_reclaim_segments_an_active_field_index_still_needs() -> Result<()> {
        use crate::db::index_manager::{CheckpointState, detect_replay_gap};

        let temp_dir = TempDir::new()?;
        let db = Database::open_with_wal_segment_size(temp_dir.path(), create_db_config(), 4096)?;
        let ns = DEFAULT_NAMESPACE_ID;
        activate_status_index(&db, ns);

        // Enough writes before the checkpoint that several whole segments end up
        // *below* the watermark — otherwise the test could pass by GC simply
        // reclaiming nothing at all.
        for i in 0..200u32 {
            db.put(format!("gap_{:04}", i).as_bytes(), br#"{"status":"active"}"#)?;
        }
        // A real checkpoint: this is the offset the field would replay from.
        db.run_index_checkpoint()?;
        let checkpoint_offset = db.wal_metadata.read().tail;
        let watermark_segment = db.wal.segment_id_for_offset(checkpoint_offset);
        assert!(watermark_segment > 1, "test needs segments below the watermark, got {watermark_segment}");

        for i in 200..600u32 {
            db.put(format!("gap_{:04}", i).as_bytes(), br#"{"status":"active"}"#)?;
        }
        db.flush_all_namespaces()?;

        let wal_tail = db.wal_metadata.read().tail;
        assert!(
            db.wal.segment_id_for_offset(wal_tail - 1) > 1,
            "test needs several WAL segments, tail is {wal_tail}"
        );

        db.garbage_collect_wal()?;

        // The whole replay window survives, so the field index remains healable.
        let missing = db.wal.missing_segments(checkpoint_offset, wal_tail);
        assert!(
            missing.is_empty(),
            "WAL GC must not reclaim segments an active field index needs; missing: {missing:?}"
        );
        assert_eq!(
            detect_replay_gap(CheckpointState::At(checkpoint_offset), false, wal_tail, missing),
            None,
            "no gap should be detectable once the watermark gates GC"
        );

        // And the pin is not "reclaim nothing": segments below the watermark are
        // still collected, so retention is bounded by the checkpoint, not frozen.
        let head_segment = db.wal.segment_id_for_offset(db.wal_metadata.read().head);
        assert!(
            head_segment > 0 && head_segment <= watermark_segment,
            "segments below the watermark should be reclaimed and reclamation should stop at it \
             (head segment {head_segment}, watermark {watermark_segment})"
        );

        db.shutdown()?;
        Ok(())
    }

    /// FR-001 step 2: a namespace whose only field index has never been
    /// checkpointed and holds no data must not pin the WAL.
    ///
    /// A first-time build has no `checkpoint` marker, so it reads as "replay from
    /// 0". A new index is populated by a build rather than by replay, so pinning
    /// on it would hold the entire WAL from the moment of every `add_index` on an
    /// established database.
    #[test]
    fn a_never_checkpointed_empty_field_index_does_not_pin_wal_gc() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let db = Database::open_with_wal_segment_size(temp_dir.path(), create_db_config(), 4096)?;
        let ns = DEFAULT_NAMESPACE_ID;

        // Writes first, so there is a WAL worth reclaiming...
        for i in 0..400u32 {
            db.put(format!("k_{:04}", i).as_bytes(), b"raw")?;
        }
        db.flush_all_namespaces()?;

        // ...then a brand-new index: registered and activated, never checkpointed,
        // and empty (the extractor matches none of the values above).
        activate_status_index(&db, ns);
        assert_eq!(
            db.index_replay_watermark_segment(),
            None,
            "an empty, never-checkpointed field must not pin"
        );

        let (reclaimed, _) = db.garbage_collect_wal()?;
        assert!(reclaimed > 0, "a new index must not stop WAL GC reclaiming persisted segments");

        db.shutdown()?;
        Ok(())
    }

    /// FR-001 step 2: a namespace with no field indices keeps its previous WAL
    /// retention exactly — the watermark is inert.
    #[test]
    fn wal_retention_is_unchanged_without_field_indices() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let db = Database::open_with_wal_segment_size(temp_dir.path(), create_db_config(), 4096)?;

        for i in 0..400u32 {
            db.put(format!("k_{:04}", i).as_bytes(), b"raw")?;
        }
        db.flush_all_namespaces()?;

        assert_eq!(db.index_replay_watermark_segment(), None);
        assert!(db.has_deletable_wal_segments());
        let (reclaimed, _) = db.garbage_collect_wal()?;
        assert!(reclaimed > 0, "an unindexed database must reclaim exactly as before");

        db.shutdown()?;
        Ok(())
    }

    /// FR-001 step 3: the backstop records a durable, key-level gap.
    ///
    /// This is the whole point of harvesting inside GC: the affected keys live in
    /// the segments being unlinked, so a record written any later could never
    /// name them.
    #[test]
    fn backstop_records_a_gap_naming_the_affected_keys() -> Result<()> {
        use crate::db::index_manager::{GapCause, RepairMode};

        let temp_dir = TempDir::new()?;
        let mut config = create_db_config();
        config.threshold_config = config.threshold_config.with_max_pinned_wal_segments(2);
        let db = Database::open_with_wal_segment_size(temp_dir.path(), config, 4096)?;
        let ns = DEFAULT_NAMESPACE_ID;
        let field_id = activate_status_index(&db, ns);

        db.put(b"seed", br#"{"status":"active"}"#)?;
        db.run_index_checkpoint()?;

        for i in 0..400u32 {
            db.put(format!("gap_{:04}", i).as_bytes(), br#"{"status":"active"}"#)?;
        }
        db.flush_all_namespaces()?;
        db.garbage_collect_wal()?;

        let gap = db.index_manager.read_gap(ns, field_id).expect("the backstop must record a gap");
        assert_eq!(gap.cause, GapCause::BackstopReclaim);
        assert_eq!(gap.namespace_id, ns);
        assert_eq!(gap.field_id, field_id);
        assert!(!gap.missing_segments.is_empty(), "the gap should name the reclaimed segments");

        // The keys are the payload that makes row-scoped repair possible at all.
        let RepairMode::RowScoped { keys } = &gap.repair else {
            panic!("expected a row-scoped worklist, got {:?}", gap.repair);
        };
        assert!(!keys.is_empty(), "the worklist must name the affected keys");
        let decoded: Vec<Vec<u8>> = keys.iter().filter_map(|k| crate::support::hex::hex_to_bytes(k)).collect();
        assert_eq!(decoded.len(), keys.len(), "every recorded key must be valid hex");
        assert!(
            decoded.iter().any(|k| k.starts_with(b"gap_")),
            "the worklist should contain keys written into the reclaimed segments"
        );

        // And it survives the restart — a log line would not have.
        db.shutdown()?;
        let db = Database::open_with_wal_segment_size(temp_dir.path(), create_db_config(), 4096)?;
        let after = db.index_manager.read_gap(ns, field_id).expect("the gap must survive a restart");
        assert_eq!(after.repair, gap.repair);
        db.shutdown()?;
        Ok(())
    }

    /// FR-001 step 2: the backstop bounds pinned WAL.
    ///
    /// With the checkpoint worker never running, the watermark would otherwise
    /// hold every segment after the first checkpoint forever. Past
    /// `max_pinned_wal_segments` GC reclaims the oldest pinned segments anyway —
    /// knowingly stranding part of the index, which is what steps 3-5 repair.
    #[test]
    fn backstop_bounds_wal_pinned_by_the_index_watermark() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let mut config = create_db_config();
        config.threshold_config = config.threshold_config.with_max_pinned_wal_segments(2);
        let db = Database::open_with_wal_segment_size(temp_dir.path(), config, 4096)?;
        let ns = DEFAULT_NAMESPACE_ID;
        activate_status_index(&db, ns);

        // Checkpoint early, then never again: the watermark freezes here.
        db.put(b"seed", br#"{"status":"active"}"#)?;
        db.run_index_checkpoint()?;
        let checkpoint_offset = db.wal_metadata.read().tail;

        for i in 0..400u32 {
            db.put(format!("gap_{:04}", i).as_bytes(), br#"{"status":"active"}"#)?;
        }
        db.flush_all_namespaces()?;

        let wal_tail = db.wal_metadata.read().tail;
        let total_segments = db.wal.segment_id_for_offset(wal_tail - 1);
        assert!(total_segments > 4, "test needs more segments than the cap, got {total_segments}");

        // The worker must still be told there is work, or the backstop could
        // never fire on a database whose checkpoint worker is wedged.
        assert!(db.has_deletable_wal_segments(), "backstop work must be visible to the GC worker");

        let (reclaimed, _) = db.garbage_collect_wal()?;
        assert!(reclaimed > 0, "the backstop should have reclaimed the oldest pinned segments");

        // Pinned retention is now bounded by the cap rather than unbounded...
        let pinned_after = db
            .wal_metadata
            .read()
            .tracked_segments()
            .filter(|&s| s >= db.wal.segment_id_for_offset(checkpoint_offset) && s < db.wal.segment_id_for_offset(wal_tail - 1))
            .count();
        assert!(
            pinned_after <= 2,
            "pinned segments should be capped at max_pinned_wal_segments, got {pinned_after}"
        );

        // ...and the cost is exactly what FR-001 says it is: a real gap, which is
        // why the detection and repair arms still have to exist.
        assert!(
            !db.wal.missing_segments(checkpoint_offset, wal_tail).is_empty(),
            "the backstop reclaims pinned segments, so a gap is expected here"
        );

        db.shutdown()?;
        Ok(())
    }

    #[test]
    fn test_wal_gc_global_counters_stay_consistent_after_restart() -> Result<()> {
        let temp_dir = TempDir::new()?;

        {
            let db = Database::open_with_wal_segment_size(temp_dir.path(), create_db_config(), 256)?;

            for i in 0..120u32 {
                db.put(&format!("k{:03}", i).into_bytes(), &[b'v'; 32])?;
            }
            db.flush_all_namespaces()?;

            db.put(b"sentinel", b"v")?;
            db.flush_all_namespaces()?;

            let (bytes_reclaimed, _) = db.garbage_collect_wal()?;
            assert!(bytes_reclaimed > 0, "expected segments to be GC'd");

            let (total, persisted) = db.get_wal_gc_stats();
            assert_eq!(
                total.saturating_sub(persisted),
                0,
                "pending should be 0 immediately after GC (total={total}, persisted={persisted})"
            );
        }

        {
            let db = Database::open_with_wal_segment_size(temp_dir.path(), create_db_config(), 256)?;
            let (total, persisted) = db.get_wal_gc_stats();
            assert_eq!(
                total.saturating_sub(persisted),
                0,
                "pending should still be 0 after reopen (total={total}, persisted={persisted})"
            );
        }

        Ok(())
    }

    #[test]
    fn test_gc_updates_last_sequence_before_deleting_segments() -> Result<()> {
        use std::sync::atomic::Ordering;
        let temp_dir = TempDir::new()?;
        let db = Database::open_with_wal_segment_size(temp_dir.path(), create_db_config(), 256)?;

        for i in 0..60u32 {
            let key = format!("gc_key_{}", i).into_bytes();
            db.put(&key, b"some_value_padding_xxx")?;
        }

        let seq_before_gc = db.next_seq.load(Ordering::Relaxed).saturating_sub(1);

        {
            let wal_metadata = db.wal_metadata.read();
            let tail = wal_metadata.tail;
            let head = wal_metadata.head;
            drop(wal_metadata);
            db.wal_flush_observer.mark_persisted_range(head, tail);
        }
        {
            let mut wal_metadata = db.wal_metadata.write();
            for i in 0..wal_metadata.segment_total_entries.len() {
                wal_metadata.segment_persisted_entries[i] = wal_metadata.segment_total_entries[i];
            }
        }

        db.garbage_collect_wal()?;

        let last_seq = db.wal_metadata.read().last_sequence;
        assert!(
            last_seq >= seq_before_gc,
            "last_sequence ({}) must be >= highest written seq ({}) after GC",
            last_seq,
            seq_before_gc
        );
        Ok(())
    }
}
