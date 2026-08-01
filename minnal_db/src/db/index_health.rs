//! Field-index health, repair, and the durable gap records behind them.
//!
//! Split out of `database.rs`. This is the remediation half of FR-001: an index
//! that is missing updates records the fact durably, reports it to anyone who
//! queries it, and can be repaired without re-putting documents.
//!
//! Three separate ways an index diverges from the store feed the same records --
//! WAL GC's backstop, no-WAL writes, and updates a field index rejected -- and
//! only the first is a WAL problem, which is why they live here rather than
//! beside the collector.

use log::{error, info, warn};

use crate::db::database::Database;
// `run_index_checkpoint` is a trait method, so the trait must be in scope here.
use crate::db::error::{KVError, Result};
use crate::db::index_checkpoint_worker::IndexCheckpointTarget;
use crate::db::namespace::{FieldId, FieldMeta, FieldRepairOutcome};

impl Database {
    /// Repair a field index that has an outstanding gap, then clear the gap.
    ///
    /// Two modes, chosen by the gap record itself:
    ///
    /// - **Row-scoped** (the normal case): walk the recorded key worklist,
    ///   re-derive each key's field value from its *current* stored bytes, and
    ///   rewrite that one row. Work is proportional to the damage, not to the
    ///   store. Reading current values rather than the lost ones is what makes
    ///   this idempotent and convergent: a key overwritten since the loss is
    ///   already correct, and a key written five times needs one repair.
    /// - **Full rebuild**: re-extract every key in the namespace. Used when the
    ///   worklist crossed its cap, or for a no-WAL gap whose keys were never
    ///   knowable.
    ///
    /// The gap is cleared **only on success**, so a failed or interrupted repair
    /// leaves the field visibly degraded and retryable rather than quietly
    /// marked healthy.
    ///
    /// Does not re-put documents: no WAL traffic, no vector re-embedding, no
    /// other field's extractor. Returns [`FieldRepairOutcome::NotDegraded`] when
    /// there is nothing to repair.
    pub fn repair_field_index(&self, namespace_id: u32, field_id: FieldId) -> Result<FieldRepairOutcome> {
        use crate::db::index_manager::RepairMode;
        use crate::db::namespace::FieldReindexOutcome;

        let Some(gap) = self.index_manager.read_gap(namespace_id, field_id) else {
            return Ok(FieldRepairOutcome::NotDegraded);
        };
        let store = self.get_store(namespace_id)?;
        if store.namespace_index.read().get(field_id).is_none() {
            return Err(KVError::Serialization(format!(
                "Field {} in namespace {} has no active index to repair; activate it first",
                field_id, namespace_id
            )));
        }

        let outcome = match &gap.repair {
            RepairMode::RowScoped { keys } => {
                // A key recorded as un-decodable hex would silently shrink the
                // worklist, so fail loudly rather than report a repair that
                // skipped rows.
                let mut decoded = Vec::with_capacity(keys.len());
                for hex in keys {
                    let key = crate::support::hex::hex_to_bytes(hex).ok_or_else(|| {
                        KVError::Serialization(format!("Gap record for ns={namespace_id} field={field_id} holds a malformed key: {hex}"))
                    })?;
                    decoded.push(key);
                }

                let mut reindexed = 0usize;
                let mut absent = 0usize;
                for key in &decoded {
                    match store.reindex_field(field_id, key)? {
                        FieldReindexOutcome::Reindexed => reindexed += 1,
                        // The key has no current value. `reindex_field` has
                        // already cleared its row, which is the repair when it
                        // was the *delete* that went missing.
                        FieldReindexOutcome::KeyNotFound => absent += 1,
                        FieldReindexOutcome::FieldNotActive => {
                            return Err(KVError::Serialization(format!(
                                "Field {field_id} in namespace {namespace_id} was deactivated mid-repair"
                            )));
                        }
                    }
                }
                FieldRepairOutcome::RowScoped {
                    keys_total: decoded.len(),
                    reindexed,
                    absent,
                }
            }
            RepairMode::FullRebuild => {
                let keys = store.keys()?;
                let scanned = keys.len();
                for key in &keys {
                    match store.reindex_field(field_id, key)? {
                        FieldReindexOutcome::FieldNotActive => {
                            return Err(KVError::Serialization(format!(
                                "Field {field_id} in namespace {namespace_id} was deactivated mid-repair"
                            )));
                        }
                        _ => continue,
                    }
                }
                FieldRepairOutcome::FullRebuild { scanned }
            }
        };

        // Make the repaired index durable before dropping the record of why it
        // needed repairing. The reverse order could clear the gap and then lose
        // the repair to a crash, leaving a silently incomplete index with
        // nothing left to say so.
        self.run_index_checkpoint()?;
        self.index_manager.clear_gap(namespace_id, field_id)?;
        info!("[REPAIR] ns={namespace_id} field={field_id}: repaired ({outcome:?}), gap cleared");
        Ok(outcome)
    }

    /// Report the health of every registered field index in a namespace.
    ///
    /// The operator-facing counterpart of the per-query `degraded_fields`
    /// signal: that tells a caller their *answer* may be short, this tells an
    /// operator *which indices* need repair and why. Dropped fields are omitted —
    /// they have no index to be healthy or otherwise.
    pub fn index_health(&self, namespace_id: u32) -> Result<Vec<crate::db::index_manager::FieldIndexHealth>> {
        let store = self.get_store(namespace_id)?;
        let wal_tail = self.wal_metadata.read().tail;

        let fields: Vec<FieldMeta> = {
            let registry = self.registry.read();
            registry
                .schema(namespace_id)
                .map(|s| s.list_fields().into_iter().filter(|f| !f.dropped).collect())
                .unwrap_or_default()
        };

        let ns_index = store.namespace_index.read();
        Ok(fields
            .into_iter()
            .map(|f| {
                let checkpoint_offset = match self.index_manager.read_checkpoint_state(namespace_id, f.field_id, wal_tail) {
                    crate::db::index_manager::CheckpointState::At(offset) => Some(offset),
                    _ => None,
                };
                crate::db::index_manager::FieldIndexHealth {
                    namespace_id,
                    field_id: f.field_id,
                    field_name: f.field_name,
                    checkpoint_offset,
                    active: ns_index.get(f.field_id).is_some(),
                    gap: self.index_manager.read_gap(namespace_id, f.field_id),
                }
            })
            .collect())
    }

    /// Note that a namespace has taken a no-WAL write whose field-index update
    /// no checkpoint has made durable yet.
    ///
    /// The durable marker is written **once** per checkpoint interval, not per
    /// write: the in-memory set is the debounce, so the fsync cost is amortised
    /// to roughly one per checkpoint rather than one per bulk-loaded document.
    /// Cleared by `run_index_checkpoint` once that namespace's index state is on
    /// disk.
    ///
    /// Only namespaces with at least one field index matter — an unindexed
    /// namespace has no index to diverge — so this is free for the raw-KV case.
    ///
    /// Best-effort: a marker that cannot be written is logged. Failing the write
    /// itself would be a worse trade (the caller asked for the fast path), but it
    /// does mean a crash could then go unreported.
    pub(crate) fn note_no_wal_index_exposure(&self, namespace_id: u32) {
        {
            let stores = self.stores.read();
            match stores.get(&namespace_id) {
                Some(store) if !store.namespace_index.read().is_empty() => {}
                _ => return,
            }
        }
        if !self.no_wal_pending.lock().insert(namespace_id) {
            return; // already marked since the last checkpoint
        }
        if let Err(e) = self.index_manager.set_no_wal_pending(namespace_id) {
            warn!("[NO-WAL] failed to mark ns={namespace_id} as having uncheckpointed no-WAL index updates: {e:?}");
        }
    }

    /// Record a full-rebuild gap for every field of a namespace that had no-WAL
    /// writes outstanding when the process died.
    ///
    /// Runs at open. The affected keys are **not** recoverable here and never
    /// were: no-WAL writes leave no WAL entries, so unlike the backstop path
    /// there is nothing to harvest. A coarse full rebuild is the honest answer,
    /// and it is still infinitely better than the silence it replaces.
    ///
    /// Fields with no checkpoint marker are skipped: they hold no persisted index
    /// state, so they are a first-time build rather than a damaged index — the
    /// same judgement `detect_replay_gap` and the watermark make.
    pub(crate) fn record_no_wal_gaps_after_unclean_shutdown(&self) {
        let namespaces: Vec<(u32, Vec<FieldId>)> = {
            let registry = self.registry.read();
            registry
                .list()
                .into_iter()
                .filter(|&(_, ns_id)| self.index_manager.no_wal_pending(ns_id))
                .filter_map(|(_, ns_id)| registry.schema(ns_id).map(|s| (ns_id, s.live_field_ids())))
                .collect()
        };
        if namespaces.is_empty() {
            return;
        }

        let wal_tail = self.wal_metadata.read().tail;
        let detected_at_ms = crate::db::kv_store::current_epoch_millis();
        for (ns_id, fields) in namespaces {
            for field_id in fields {
                if self.index_manager.read_checkpoint_state(ns_id, field_id, wal_tail) == crate::db::index_manager::CheckpointState::Absent {
                    continue;
                }
                error!(
                    "[NO-WAL] ns={ns_id} field={field_id}: the previous run ended uncleanly with no-WAL writes outstanding. \
                     Those writes have no WAL entries, so their index updates cannot be replayed — the field index is INCOMPLETE \
                     and needs a full rebuild."
                );
                let gap = crate::db::index_manager::GapRecord {
                    namespace_id: ns_id,
                    field_id,
                    cause: crate::db::index_manager::GapCause::NoWalWrites,
                    from: 0,
                    to: 0,
                    missing_segments: Vec::new(),
                    detected_at_ms,
                    repair: crate::db::index_manager::RepairMode::FullRebuild,
                };
                if let Err(e) = self.index_manager.record_gap(gap, Self::GAP_KEY_WORKLIST_CAP) {
                    warn!("[NO-WAL] failed to record gap for ns={ns_id} field={field_id}: {e:?}");
                }
            }
            // The condition is now recorded, so the marker has done its job.
            if let Err(e) = self.index_manager.clear_no_wal_pending(ns_id) {
                warn!("[NO-WAL] failed to clear the no-WAL marker for ns={ns_id}: {e:?}");
            }
        }
    }

    /// Turn field-index updates rejected since the last checkpoint into durable
    /// gap records.
    ///
    /// Deliberately deferred to here rather than done on the write path: a gap
    /// record is an fsync, and a field that rejects every write would otherwise
    /// pay one per write. Deferring is sound because the buffer and the index
    /// share a durability point — a crash that loses a buffered key also lost
    /// the index update it describes, and WAL replay re-runs the write,
    /// re-rejects it, and re-buffers the key.
    ///
    /// Best-effort: a failure to record is logged, never propagated, so a
    /// reporting problem cannot fail a checkpoint that is otherwise persisting
    /// real index state.
    pub(crate) fn persist_rejected_update_gaps(&self) {
        if self.rejected_index_updates.is_empty() {
            return;
        }
        let detected_at_ms = crate::db::kv_store::current_epoch_millis();
        for ((ns_id, field_id), rejected) in self.rejected_index_updates.drain() {
            let repair = if rejected.overflowed {
                crate::db::index_manager::RepairMode::FullRebuild
            } else {
                crate::db::index_manager::RepairMode::RowScoped {
                    keys: rejected.keys.iter().map(|k| crate::support::hex::bytes_to_hex(k)).collect(),
                }
            };
            warn!(
                "[IndexCheckpoint] ns={ns_id} field={field_id}: recording a gap for {} rejected index update(s){}",
                rejected.keys.len(),
                if rejected.overflowed {
                    " (overflowed — full rebuild required)"
                } else {
                    ""
                }
            );
            let gap = crate::db::index_manager::GapRecord {
                namespace_id: ns_id,
                field_id,
                cause: crate::db::index_manager::GapCause::RejectedUpdate,
                // A rejected update is not a WAL replay window — the key is known
                // exactly, so there is no range to record.
                from: 0,
                to: 0,
                missing_segments: Vec::new(),
                detected_at_ms,
                repair,
            };
            if let Err(e) = self.index_manager.record_gap(gap, Self::GAP_KEY_WORKLIST_CAP) {
                warn!("[IndexCheckpoint] failed to record rejected-update gap for ns={ns_id} field={field_id}: {e:?}");
            }
        }
    }
}
