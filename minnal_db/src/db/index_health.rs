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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::namespace::DEFAULT_NAMESPACE_ID;
    use crate::db::test_support::*;
    use std::sync::Arc;
    use tempfile::TempDir;

    /// FR-001 step 1: dropping a field index deletes its on-disk directory.
    ///
    /// Before this, `deactivate_field_index` was an in-memory deregister only, so
    /// the bitmap blob, keymap and — the load-bearing part — the frozen
    /// `checkpoint` marker survived until the whole namespace was dropped.
    #[test]
    fn test_drop_field_index_deletes_its_directory() {
        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        let ns = DEFAULT_NAMESPACE_ID;

        let field_id = activate_status_index(&db, ns);
        db.put(b"doc:1", br#"{"status":"active"}"#).unwrap();
        db.run_index_checkpoint().unwrap();

        let field_path = db.index_manager.field_path(ns, field_id);
        assert!(field_path.join("checkpoint").exists(), "checkpoint marker should exist before the drop");

        db.drop_field_index(ns, field_id).unwrap();
        assert!(!field_path.exists(), "the field directory must be gone after a drop");

        // The namespace's row map is a sibling of the field directories and must
        // survive — other fields resolve their row IDs through it.
        assert!(db.index_manager.rowmap_path(ns).exists(), "dropping a field must not touch the row map");

        // Idempotent: dropping again is not an error.
        db.drop_field_index(ns, field_id).unwrap();

        db.shutdown().unwrap();
    }

    /// FR-001 step 1: a dropped field is excluded from the checkpoint worker's
    /// field set, so its frozen marker can never hold up WAL GC (and the
    /// checkpoint does not try to write into a directory that is gone).
    #[test]
    fn test_dropped_field_is_excluded_from_checkpoint_fields() {
        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        let ns = DEFAULT_NAMESPACE_ID;

        let field_id = activate_status_index(&db, ns);
        db.put(b"doc:1", br#"{"status":"active"}"#).unwrap();

        assert!(db.registry.read().all_indexed_fields().contains(&(ns, field_id)));

        db.drop_field_index(ns, field_id).unwrap();

        assert!(
            !db.registry.read().all_indexed_fields().contains(&(ns, field_id)),
            "a dropped field must not appear in all_indexed_fields"
        );
        // Checkpointing after the drop must still succeed — the field directory
        // no longer exists, so including it would fail the whole pass.
        db.run_index_checkpoint().unwrap();

        db.shutdown().unwrap();
    }

    /// FR-001 step 1: the drop is persisted, so a drop interrupted after the
    /// schema write but before the files were deleted is completed at the next
    /// open rather than leaking until the namespace is dropped.
    #[test]
    fn test_interrupted_field_drop_is_completed_at_open() {
        let dir = TempDir::new().unwrap();
        let ns = DEFAULT_NAMESPACE_ID;

        let field_path;
        {
            let db = Database::open(dir.path(), create_db_config()).unwrap();
            let field_id = activate_status_index(&db, ns);
            db.put(b"doc:1", br#"{"status":"active"}"#).unwrap();
            db.run_index_checkpoint().unwrap();
            field_path = db.index_manager.field_path(ns, field_id);

            // Simulate a crash between step (1) persist and step (3) delete:
            // mark the field dropped durably, then leave every file in place.
            db.registry.write().mark_schema_field_dropped(ns, field_id).unwrap();
            assert!(field_path.exists(), "precondition: files still present at the crash point");
            db.shutdown().unwrap();
        }

        let db = Database::open(dir.path(), create_db_config()).unwrap();
        assert!(!field_path.exists(), "open must finish the interrupted drop and reclaim the directory");
        db.shutdown().unwrap();
    }

    /// FR-001 step 1: activating a dropped field must fail loudly.
    ///
    /// This is the guard that closes the silent-incomplete window directly: a
    /// dropped field's directory is gone, so activation would open an empty index
    /// whose `Absent` + empty shape `detect_replay_gap` deliberately suppresses as
    /// a normal first build — an index that queries as complete while holding
    /// nothing.
    #[test]
    fn test_activating_a_dropped_field_is_rejected() {
        use crate::db::namespace_index::ExtractorFn;
        use crate::index::{IndexValue, IndexValueType};
        use std::sync::Arc;

        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        let ns = DEFAULT_NAMESPACE_ID;

        let field_id = activate_status_index(&db, ns);
        db.put(b"doc:1", br#"{"status":"active"}"#).unwrap();
        db.drop_field_index(ns, field_id).unwrap();

        let extractor: ExtractorFn = Arc::new(|bytes: &[u8]| {
            let s = std::str::from_utf8(bytes).ok()?;
            let v: serde_json::Value = serde_json::from_str(s).ok()?;
            Some(IndexValue::Str(v["status"].as_str()?.to_string()))
        });
        let err = db.activate_field_index(ns, field_id, IndexValueType::Str, extractor.clone()).unwrap_err();
        assert!(err.to_string().contains("was dropped"), "expected a dropped-field rejection, got: {err}");

        // Re-registering clears the flag and reuses the same field id, so the
        // documented add-after-drop path still works.
        let reused = db.register_index_field(ns, "status", IndexValueType::Str).unwrap();
        assert_eq!(reused, field_id, "re-registering a dropped field must reuse its id");
        db.activate_field_index(ns, field_id, IndexValueType::Str, extractor).unwrap();

        db.shutdown().unwrap();
    }

    /// FR-001 step 1: the `dropped` flag survives a restart. Without persistence
    /// the drop could not be completed at open, and a dropped field would silently
    /// become activatable again over an empty directory.
    #[test]
    fn test_dropped_flag_survives_restart() {
        let dir = TempDir::new().unwrap();
        let ns = DEFAULT_NAMESPACE_ID;

        let field_id = {
            let db = Database::open(dir.path(), create_db_config()).unwrap();
            let field_id = activate_status_index(&db, ns);
            db.put(b"doc:1", br#"{"status":"active"}"#).unwrap();
            db.drop_field_index(ns, field_id).unwrap();
            db.shutdown().unwrap();
            field_id
        };

        let db = Database::open(dir.path(), create_db_config()).unwrap();
        let registry = db.registry.read();
        let meta = registry
            .schema(ns)
            .unwrap()
            .get_field(field_id)
            .expect("field definition is retained for id reuse");
        assert!(meta.dropped, "the dropped flag must survive a restart");
        drop(registry);
        db.shutdown().unwrap();
    }

    /// Targeted single-field reindex: re-deriving a key's value for one field
    /// must repair a stale index entry, be idempotent, and report the right
    /// outcome for a missing key or an inactive field.
    #[test]
    fn test_reindex_field_repairs_and_reports_outcome() {
        use crate::db::namespace::FieldReindexOutcome;
        use crate::db::namespace_index::ExtractorFn;
        use crate::index::{IndexValue, IndexValueType};
        use std::sync::Arc;

        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        let ns = DEFAULT_NAMESPACE_ID;

        let field_id = db.register_index_field(ns, "status", IndexValueType::Str).unwrap();
        let extractor: ExtractorFn = Arc::new(|bytes: &[u8]| {
            let s = std::str::from_utf8(bytes).ok()?;
            let v: serde_json::Value = serde_json::from_str(s).ok()?;
            Some(IndexValue::Str(v["status"].as_str()?.to_string()))
        });
        db.activate_field_index(ns, field_id, IndexValueType::Str, extractor).unwrap();

        db.put(b"doc:1", br#"{"status":"active"}"#).unwrap();
        assert_eq!(db.query_keys(ns, "status = \"active\"").unwrap().keys, vec![b"doc:1".to_vec()]);

        // Reindexing an up-to-date entry is a no-op: still queryable, no duplicate.
        assert_eq!(db.reindex_field(ns, field_id, b"doc:1").unwrap(), FieldReindexOutcome::Reindexed);
        assert_eq!(db.query_keys(ns, "status = \"active\"").unwrap().keys, vec![b"doc:1".to_vec()]);

        // A key with no value reports KeyNotFound and changes nothing.
        assert_eq!(db.reindex_field(ns, field_id, b"missing").unwrap(), FieldReindexOutcome::KeyNotFound);

        // An unregistered field id reports FieldNotActive rather than erroring.
        assert_eq!(db.reindex_field(ns, 9999, b"doc:1").unwrap(), FieldReindexOutcome::FieldNotActive);

        db.shutdown().unwrap();
    }

    /// Deactivating a field index and deleting its on-disk directory must not
    /// cause shutdown to fail with ENOENT when run_index_checkpoint is called.
    #[test]
    fn test_shutdown_after_drop_index_does_not_error() {
        use crate::db::namespace_index::ExtractorFn;
        use crate::index::{IndexValue, IndexValueType};
        use std::sync::Arc;

        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        let ns = DEFAULT_NAMESPACE_ID;

        let field_id = db.register_index_field(ns, "status", IndexValueType::Str).unwrap();
        let extractor: ExtractorFn = Arc::new(|bytes: &[u8]| {
            let s = std::str::from_utf8(bytes).ok()?;
            let v: serde_json::Value = serde_json::from_str(s).ok()?;
            Some(IndexValue::Str(v["status"].as_str()?.to_string()))
        });
        db.activate_field_index(ns, field_id, IndexValueType::Str, extractor).unwrap();
        db.put(b"doc:1", br#"{"status":"active"}"#).unwrap();

        // Simulate drop_index: deactivate in-memory then delete on-disk directory.
        db.deactivate_field_index(ns, field_id).unwrap();
        let index_dir = crate::db::layout::namespace_index_dir(&crate::db::layout::index_root(dir.path()), ns).join(field_id.to_string());
        if index_dir.exists() {
            std::fs::remove_dir_all(&index_dir).unwrap();
        }

        // Shutdown must succeed even though the field's directory is gone.
        db.shutdown().unwrap();
    }

    /// FR-001 step 3: a rejected index update becomes a durable one-key gap
    /// rather than only a `warn!`.
    #[test]
    fn a_rejected_index_update_is_recorded_as_a_gap() -> Result<()> {
        use crate::db::index_manager::{GapCause, RepairMode};
        use crate::db::namespace_index::ExtractorFn;
        use crate::index::{IndexValue, IndexValueType};

        let temp_dir = TempDir::new()?;
        let db = Database::open(temp_dir.path(), create_db_config())?;
        let ns = DEFAULT_NAMESPACE_ID;

        // A Str-typed field whose extractor yields an Int for one document: the
        // index refuses the update, which used to be a log line and nothing else.
        let field_id = db.register_index_field(ns, "status", IndexValueType::Str)?;
        let extractor: ExtractorFn = Arc::new(|bytes: &[u8]| {
            if bytes.starts_with(b"bad") {
                Some(IndexValue::Int(1))
            } else {
                Some(IndexValue::Str(String::from_utf8_lossy(bytes).into_owned()))
            }
        });
        db.activate_field_index(ns, field_id, IndexValueType::Str, extractor)?;

        db.put(b"ok:1", b"fine")?;
        db.put(b"doc:bad", b"bad-value")?;
        db.run_index_checkpoint()?;

        let gap = db.index_manager.read_gap(ns, field_id).expect("a rejected update must be recorded");
        assert_eq!(gap.cause, GapCause::RejectedUpdate);
        let RepairMode::RowScoped { keys } = &gap.repair else {
            panic!("a rejected update names its key exactly, so repair is row-scoped: {:?}", gap.repair);
        };
        assert_eq!(keys.len(), 1, "only the rejected key should be listed, got {keys:?}");
        assert_eq!(crate::support::hex::hex_to_bytes(&keys[0]).unwrap(), b"doc:bad".to_vec());

        db.shutdown()?;
        Ok(())
    }

    /// FR-001 step 3: no-WAL writes have no WAL entries, so an unclean shutdown
    /// leaves the field index incomplete with nothing to replay. The marker turns
    /// that silence into a recorded, repairable full rebuild.
    #[test]
    fn unclean_shutdown_with_no_wal_writes_records_a_full_rebuild_gap() -> Result<()> {
        use crate::db::index_manager::{GapCause, RepairMode};

        let temp_dir = TempDir::new()?;
        let ns = DEFAULT_NAMESPACE_ID;

        let field_id = {
            let db = Database::open(temp_dir.path(), create_db_config())?;
            let field_id = activate_status_index(&db, ns);
            db.put(b"doc:1", br#"{"status":"active"}"#)?;
            // Checkpoint so the field has persisted state — otherwise it is a
            // first-time build and legitimately has nothing to lose.
            db.run_index_checkpoint()?;
            assert!(!db.index_manager.no_wal_pending(ns), "a checkpoint clears the marker");

            db.put_ns_no_wal(ns, b"doc:2", br#"{"status":"active"}"#)?;
            assert!(db.index_manager.no_wal_pending(ns), "a no-WAL write must set the marker before writing");

            // Drop without shutdown() — the unclean case. A clean shutdown runs a
            // checkpoint first, which is exactly what clears the marker.
            std::mem::forget(db);
            field_id
        };

        let db = Database::open(temp_dir.path(), create_db_config())?;
        let gap = db
            .index_manager
            .read_gap(ns, field_id)
            .expect("an unclean shutdown with no-WAL writes outstanding must record a gap");
        assert_eq!(gap.cause, GapCause::NoWalWrites);
        assert_eq!(
            gap.repair,
            RepairMode::FullRebuild,
            "the affected keys were never in the WAL, so they cannot be named"
        );
        assert!(!db.index_manager.no_wal_pending(ns), "the marker is cleared once the gap is recorded");

        db.shutdown()?;
        Ok(())
    }

    /// FR-001 step 3: a clean shutdown must NOT report a no-WAL gap — it
    /// checkpoints first, which is what makes those index updates durable.
    #[test]
    fn clean_shutdown_after_no_wal_writes_records_no_gap() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let ns = DEFAULT_NAMESPACE_ID;

        let field_id = {
            let db = Database::open(temp_dir.path(), create_db_config())?;
            let field_id = activate_status_index(&db, ns);
            db.put(b"doc:1", br#"{"status":"active"}"#)?;
            db.run_index_checkpoint()?;
            db.put_ns_no_wal(ns, b"doc:2", br#"{"status":"active"}"#)?;
            db.shutdown()?;
            field_id
        };

        let db = Database::open(temp_dir.path(), create_db_config())?;
        assert_eq!(
            db.index_manager.read_gap(ns, field_id),
            None,
            "a clean shutdown checkpoints before exiting, so there is no gap to report"
        );
        db.shutdown()?;
        Ok(())
    }

    /// FR-001 step 3: gaps accumulate rather than overwrite. A second detection
    /// before anyone repairs the first must not discard the first worklist.
    #[test]
    fn gap_records_merge_instead_of_overwriting() {
        use crate::db::index_manager::{GapCause, GapRecord, RepairMode};

        let mk = |from: u64, to: u64, segs: Vec<u64>, keys: &[&str]| GapRecord {
            namespace_id: 0,
            field_id: 1,
            cause: GapCause::BackstopReclaim,
            from,
            to,
            missing_segments: segs,
            detected_at_ms: from,
            repair: RepairMode::RowScoped {
                keys: keys.iter().map(|k| k.to_string()).collect(),
            },
        };

        let mut first = mk(100, 200, vec![1, 2], &["aa", "bb"]);
        first.merge(mk(50, 300, vec![2, 3], &["bb", "cc"]), 10);

        assert_eq!(first.from, 50, "the earliest start wins");
        assert_eq!(first.to, 300, "the latest end wins");
        assert_eq!(first.missing_segments, vec![1, 2, 3], "segment ids union and dedupe");
        assert_eq!(first.detected_at_ms, 100, "the original detection time is kept");
        let RepairMode::RowScoped { keys } = &first.repair else {
            panic!("expected the worklists to union, got {:?}", first.repair);
        };
        assert_eq!(keys, &["aa", "bb", "cc"], "worklists union and dedupe");

        // Crossing the cap downgrades to a full rebuild...
        let mut capped = mk(0, 10, vec![], &["aa", "bb"]);
        capped.merge(mk(0, 10, vec![], &["cc", "dd"]), 3);
        assert_eq!(capped.repair, RepairMode::FullRebuild, "past the cap the worklist is dropped");

        // ...and a full rebuild is absorbing: it can never be downgraded back to
        // a row-scoped repair that would miss the un-nameable rows.
        let mut full = mk(0, 10, vec![], &["aa"]);
        full.repair = RepairMode::FullRebuild;
        full.merge(mk(0, 10, vec![], &["bb"]), 100);
        assert_eq!(full.repair, RepairMode::FullRebuild);
    }

    /// FR-001 step 4: index health names the degraded fields for an operator.
    #[test]
    fn index_health_reports_gaps_per_field() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let db = Database::open(temp_dir.path(), create_db_config())?;
        let ns = DEFAULT_NAMESPACE_ID;

        let status_field = activate_status_index(&db, ns);
        activate_named_index(&db, ns, "tier");
        db.put(b"doc:1", br#"{"status":"active","tier":"gold"}"#)?;
        db.run_index_checkpoint()?;

        let health = db.index_health(ns)?;
        assert_eq!(health.len(), 2, "both registered fields should be reported");
        assert!(health.iter().all(|h| h.active), "both fields are activated");
        assert!(health.iter().all(|h| !h.is_degraded()), "nothing is degraded yet");
        assert!(
            health.iter().all(|h| h.checkpoint_offset.is_some()),
            "a checkpointed field should report its offset"
        );

        record_test_gap(&db, ns, status_field);
        let health = db.index_health(ns)?;
        let degraded: Vec<&str> = health.iter().filter(|h| h.is_degraded()).map(|h| h.field_name.as_str()).collect();
        assert_eq!(degraded, vec!["status"], "only the damaged field is reported degraded");

        db.shutdown()?;
        Ok(())
    }

    /// FR-001 step 5: row-scoped repair fixes exactly the recorded keys and
    /// clears the gap, so queries stop reporting degradation.
    #[test]
    fn row_scoped_repair_restores_the_index_and_clears_the_gap() -> Result<()> {
        use crate::db::index_manager::{GapCause, GapRecord, RepairMode};

        let temp_dir = TempDir::new()?;
        let db = Database::open(temp_dir.path(), create_db_config())?;
        let ns = DEFAULT_NAMESPACE_ID;
        let field_id = activate_status_index(&db, ns);

        db.put(b"doc:1", br#"{"status":"active"}"#)?;
        db.put(b"doc:2", br#"{"status":"active"}"#)?;

        // Simulate the damage: drop doc:2's row from the index behind the write
        // path's back, exactly as a lost replay would.
        {
            let store = db.get_store(ns)?;
            let row_id = store.resolve_row_id_get(b"doc:2").expect("doc:2 has a row id");
            let ns_index = store.namespace_index.read();
            ns_index.get(field_id).unwrap().index.write().remove_all_for_row(row_id);
        }
        assert_eq!(
            db.query_keys(ns, "status = \"active\"")?.keys,
            vec![b"doc:1".to_vec()],
            "precondition: the index is now short one document"
        );

        db.index_manager.record_gap(
            GapRecord {
                namespace_id: ns,
                field_id,
                cause: GapCause::BackstopReclaim,
                from: 0,
                to: 0,
                missing_segments: vec![],
                detected_at_ms: 1,
                repair: RepairMode::RowScoped {
                    keys: vec![crate::support::hex::bytes_to_hex(b"doc:2")],
                },
            },
            1000,
        )?;

        let outcome = db.repair_field_index(ns, field_id)?;
        assert_eq!(
            outcome,
            FieldRepairOutcome::RowScoped {
                keys_total: 1,
                reindexed: 1,
                absent: 0
            }
        );

        let after = db.query_keys(ns, "status = \"active\"")?;
        let mut keys = after.keys.clone();
        keys.sort();
        assert_eq!(keys, vec![b"doc:1".to_vec(), b"doc:2".to_vec()], "the missing row is back");
        assert!(!after.is_degraded(), "a successful repair clears the gap");
        assert!(db.index_manager.read_gap(ns, field_id).is_none());

        // Repairing a healthy field is a no-op, not an error.
        assert_eq!(db.repair_field_index(ns, field_id)?, FieldRepairOutcome::NotDegraded);

        db.shutdown()?;
        Ok(())
    }

    /// FR-001 step 5: a key on the worklist whose value is **gone** must have its
    /// row cleared, not skipped.
    ///
    /// The lost update can be the *delete*. `reindex_field` used to return
    /// `KeyNotFound` and do nothing, leaving a stale row that queries as a hit —
    /// a repair that leaves the index wrong in the opposite direction.
    #[test]
    fn repair_clears_the_row_of_a_key_whose_delete_was_lost() -> Result<()> {
        use crate::db::index_manager::{GapCause, GapRecord, RepairMode};

        let temp_dir = TempDir::new()?;
        let db = Database::open(temp_dir.path(), create_db_config())?;
        let ns = DEFAULT_NAMESPACE_ID;
        let field_id = activate_status_index(&db, ns);

        db.put(b"doc:1", br#"{"status":"active"}"#)?;
        db.put(b"doc:2", br#"{"status":"active"}"#)?;

        // Delete doc:2 from storage only, leaving its index row behind — what a
        // lost delete looks like after a crash.
        {
            let store = db.get_store(ns)?;
            store.delete_from_storage(b"doc:2")?;
            let row_id = store.resolve_row_id_get(b"doc:2").expect("row id still known");
            let ns_index = store.namespace_index.read();
            ns_index
                .get(field_id)
                .unwrap()
                .index
                .write()
                .set(&crate::index::IndexValue::Str("active".into()), row_id)
                .unwrap();
        }
        assert!(
            db.query_keys(ns, "status = \"active\"")?.keys.contains(&b"doc:2".to_vec()),
            "precondition: the index still returns the deleted document"
        );

        db.index_manager.record_gap(
            GapRecord {
                namespace_id: ns,
                field_id,
                cause: GapCause::BackstopReclaim,
                from: 0,
                to: 0,
                missing_segments: vec![],
                detected_at_ms: 1,
                repair: RepairMode::RowScoped {
                    keys: vec![crate::support::hex::bytes_to_hex(b"doc:2")],
                },
            },
            1000,
        )?;

        let outcome = db.repair_field_index(ns, field_id)?;
        assert_eq!(
            outcome,
            FieldRepairOutcome::RowScoped {
                keys_total: 1,
                reindexed: 0,
                absent: 1
            },
            "the key is absent, and that still counts as repaired"
        );
        assert!(
            !db.query_keys(ns, "status = \"active\"")?.keys.contains(&b"doc:2".to_vec()),
            "the stale row must be gone — a lost delete is as wrong as a lost write"
        );

        db.shutdown()?;
        Ok(())
    }

    /// FR-001 step 5: a `FullRebuild` gap re-extracts the whole field.
    #[test]
    fn full_rebuild_repair_reindexes_every_key() -> Result<()> {
        use crate::db::index_manager::{GapCause, GapRecord, RepairMode};

        let temp_dir = TempDir::new()?;
        let db = Database::open(temp_dir.path(), create_db_config())?;
        let ns = DEFAULT_NAMESPACE_ID;
        let field_id = activate_status_index(&db, ns);

        for i in 0..5u32 {
            db.put(format!("doc:{i}").as_bytes(), br#"{"status":"active"}"#)?;
        }

        // Wipe several rows — a no-WAL gap cannot name which, hence full rebuild.
        {
            let store = db.get_store(ns)?;
            let ns_index = store.namespace_index.read();
            let entry = ns_index.get(field_id).unwrap();
            for i in 0..3u32 {
                let row_id = store.resolve_row_id_get(format!("doc:{i}").as_bytes()).unwrap();
                entry.index.write().remove_all_for_row(row_id);
            }
        }
        assert_eq!(
            db.query_keys(ns, "status = \"active\"")?.keys.len(),
            2,
            "precondition: three rows are missing"
        );

        db.index_manager.record_gap(
            GapRecord {
                namespace_id: ns,
                field_id,
                cause: GapCause::NoWalWrites,
                from: 0,
                to: 0,
                missing_segments: vec![],
                detected_at_ms: 1,
                repair: RepairMode::FullRebuild,
            },
            1000,
        )?;

        assert_eq!(db.repair_field_index(ns, field_id)?, FieldRepairOutcome::FullRebuild { scanned: 5 });
        assert_eq!(
            db.query_keys(ns, "status = \"active\"")?.keys.len(),
            5,
            "a full rebuild restores every row"
        );
        assert!(db.index_manager.read_gap(ns, field_id).is_none());

        db.shutdown()?;
        Ok(())
    }
}
