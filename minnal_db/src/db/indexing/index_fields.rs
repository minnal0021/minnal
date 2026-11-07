//! The field registry: registering, activating, dropping and measuring field indexes.

use crate::db::database::Database;
use crate::db::error::{KVError, Result};
use crate::field_index::{DynFieldIndex, IndexValueType};
use crate::field_index::{FieldId, FieldMeta, FieldReindexOutcome};
use crate::kv_store::KVStore;
use crate::kv_store::index_set::{ExtractorFn, IndexEntry};

use log::{debug, error, info, warn};
use std::collections::HashMap;
use std::sync::Arc;

impl Database {
    /// Register an indexed field for a namespace.
    ///
    /// Assigns a monotonic `FieldId` via the namespace schema, then creates
    /// `{db_path}/field-index/{namespace_id}/{field_id}/` on disk.
    /// The field name must be unique within the namespace.
    ///
    /// The `value_type` is stored in the schema and validated when
    /// `activate_field_index` is called, so type mismatches are caught at
    /// activation time rather than at query time.
    pub fn register_index_field(&self, namespace_id: u32, field_name: &str, value_type: IndexValueType) -> Result<FieldId> {
        let field_id = self.registry.write().register_schema_field(namespace_id, field_name, value_type)?;
        self.index_manager.ensure_field_path(namespace_id, field_id)?;
        Ok(field_id)
    }

    /// Register a custom row-ID function (and optionally its inverse) for a namespace.
    ///
    /// When set, every key written to or deleted from `namespace_id` will be
    /// assigned a row ID produced by `row_id_fn` rather than the default
    /// dense row-ID map.  Providing `row_to_key_fn` (the exact inverse) additionally
    /// enables O(|hits|) query resolution with zero memory overhead: each matching
    /// key is reconstructed directly from its row ID without any in-memory map.
    ///
    /// This must be called **before** [`activate_field_index`] so that the WAL
    /// replay during activation uses the same row IDs that future writes will
    /// produce.
    ///
    /// [`query_keys`]: Self::query_keys
    /// [`activate_field_index`]: Self::activate_field_index
    pub fn set_row_id_fn(
        &self,
        namespace_id: u32,
        row_id_fn: crate::kv_store::index_set::RowIdFn,
        row_to_key_fn: Option<crate::kv_store::index_set::RowToKeyFn>,
    ) -> Result<()> {
        self.get_store(namespace_id)?.set_row_id_fn(row_id_fn, row_to_key_fn)
    }

    /// Return all indexed fields registered for a namespace, sorted by `FieldId`.
    pub fn list_index_fields(&self, namespace_id: u32) -> Vec<FieldMeta> {
        self.registry.read().schema(namespace_id).map(|s| s.list_fields()).unwrap_or_default()
    }

    /// Return the number of distinct indexed values for a field.
    ///
    /// Returns `None` when the field is not active (not yet registered via
    /// `activate_field_index`).
    pub fn field_index_distinct_count(&self, namespace_id: u32, field_id: FieldId) -> Option<usize> {
        let store = self.get_store(namespace_id).ok()?;
        let ns_index = store.namespace_index.read();
        ns_index.get(field_id).map(|e| e.index.read().distinct_count())
    }

    /// Reclaimable dead-space ratios (`0.0..1.0`) for a field's two append-only
    /// stores, as `(bitmap_waste, keymap_waste)`. The bitmap store grows with
    /// per-document churn; the keymap store grows under distinct-value churn.
    /// Compaction reclaims either at the index checkpoint once it crosses
    /// [`index_blob_waste_threshold`](Self::index_blob_waste_threshold).
    ///
    /// Returns `None` when the field is not active.
    pub fn field_index_waste(&self, namespace_id: u32, field_id: FieldId) -> Option<(f64, f64)> {
        let store = self.get_store(namespace_id).ok()?;
        let ns_index = store.namespace_index.read();
        ns_index.get(field_id).map(|e| {
            let idx = e.index.read();
            (idx.bitmap_waste_ratio(), idx.keymap_waste_ratio())
        })
    }

    /// On-disk blob growth/waste metrics for an active field index — the bitmap
    /// and keymap store sizes (logical vs. live bytes) and waste ratios. Use this
    /// to monitor the append-only write amplification that low-cardinality fields
    /// suffer (a value rewritten per document leaves a stale blob copy each time),
    /// e.g. to alert before disk fills between compactions.
    ///
    /// Returns `None` when the field is not active.
    pub fn field_index_blob_stats(&self, namespace_id: u32, field_id: FieldId) -> Option<crate::field_index::IndexBlobStats> {
        let store = self.get_store(namespace_id).ok()?;
        let ns_index = store.namespace_index.read();
        ns_index.get(field_id).map(|e| e.index.read().blob_stats())
    }

    /// Reindex a single field for a single key: re-derive the field value from
    /// the key's current stored bytes and rewrite its entry in that field's
    /// index, using the same extractor + `DynFieldIndex` ops as the put path.
    /// Touches only the named field — no value rewrite, no other-field or vector
    /// re-indexing. See [`KVStore::reindex_field`].
    ///
    /// Returns `FieldReindexOutcome::FieldNotActive` when the namespace is not
    /// open or `field_id` has no live index there.
    pub fn reindex_field(&self, namespace_id: u32, field_id: FieldId, key: &[u8]) -> Result<FieldReindexOutcome> {
        let Ok(store) = self.get_store(namespace_id) else {
            return Ok(FieldReindexOutcome::FieldNotActive);
        };
        self.reindex_key(&store, namespace_id, field_id, key)
    }

    /// [`KVStore::reindex_field`] under `key`'s stripe — the only way anything
    /// outside the write path should rewrite an index row.
    ///
    /// Reindexing reads the key's current value and then sets its row. A put to
    /// the same key updates the index after its own apply, under the stripe, so
    /// without the stripe the two interleave: reindex reads v1, the put applies v2
    /// and moves the row v1→v2, then reindex sets it back to v1 — the index now
    /// disagrees with the store until something rewrites the key.
    pub(crate) fn reindex_key(&self, store: &KVStore, namespace_id: u32, field_id: FieldId, key: &[u8]) -> Result<FieldReindexOutcome> {
        let _stripe = self.key_locks.guard(namespace_id, key);
        store.reindex_field(field_id, key)
    }

    /// The configured field-index compaction threshold as a fraction
    /// (`0.0..1.0`) — a store is compacted at the next checkpoint once its waste
    /// ratio reaches this. Mirrors the clamp applied in `run_index_checkpoint`.
    pub fn index_blob_waste_threshold(&self) -> f64 {
        (self.config.threshold_config.index_blob_waste_threshold / 100.0).clamp(0.0, 1.0)
    }

    /// Wire up a live extractor for a previously-registered field.
    ///
    /// Loads the on-disk index state (BlobStore + keymap mmap store) and replays any
    /// WAL entries written since the last checkpoint, so the index is fully
    /// current before it becomes visible.  After this call, every `put` and
    /// `delete` on the namespace's KVStore will keep the in-memory
    /// [`DynFieldIndex`] up to date automatically.
    ///
    /// # Arguments
    /// * `namespace_id` – the namespace the field belongs to
    /// * `field_id`     – the [`FieldId`] returned by `register_index_field`
    /// * `value_type`   – runtime type for the index entries
    /// * `extractor`    – closure that extracts a typed value from raw document bytes
    pub fn activate_field_index(&self, namespace_id: u32, field_id: FieldId, value_type: IndexValueType, extractor: ExtractorFn) -> Result<()> {
        self.activate_field_index_inner(namespace_id, field_id, value_type, extractor, true)
    }

    /// [`activate_field_index`](Self::activate_field_index) for a new index that
    /// a build is about to fill: no WAL replay, so it starts empty.
    ///
    /// The replay reads the current value of every key in the WAL window, and
    /// writers do not see the field until it is registered at the end. On a store
    /// taking writes, a key deleted after the replay read it therefore stays in
    /// the index: the delete had no index to update, and the build, which only
    /// visits keys that exist, never comes back to it. When the key is written
    /// again it is a fresh insert, so the row ends up in two buckets. A build
    /// indexes every key's current value anyway, so the replay only adds that risk.
    pub fn activate_field_index_for_build(
        &self,
        namespace_id: u32,
        field_id: FieldId,
        value_type: IndexValueType,
        extractor: ExtractorFn,
    ) -> Result<()> {
        self.activate_field_index_inner(namespace_id, field_id, value_type, extractor, false)
    }

    fn activate_field_index_inner(
        &self,
        namespace_id: u32,
        field_id: FieldId,
        value_type: IndexValueType,
        extractor: ExtractorFn,
        replay_wal: bool,
    ) -> Result<()> {
        // Validate: field_id must be registered and its stored type must match.
        {
            let registry = self.registry.read();
            let schema = registry
                .schema(namespace_id)
                .ok_or_else(|| KVError::Serialization(format!("Namespace {} not found", namespace_id)))?;
            let meta = schema
                .get_field(field_id)
                .ok_or_else(|| KVError::Serialization(format!("Field id {} is not registered in namespace {}", field_id, namespace_id)))?;
            if meta.field_type != value_type {
                return Err(KVError::Serialization(format!(
                    "Type mismatch for field '{}' (id {}): registered as {:?}, activation supplies {:?}",
                    meta.field_name, field_id, meta.field_type, value_type
                )));
            }
            // A dropped field's directory is gone (or is about to be), so
            // activating it would open an empty index and silently serve
            // incomplete results — `detect_replay_gap` suppresses exactly this
            // shape (`Absent` + empty) as a normal first build. Fail loudly and
            // make the caller re-register, which clears the flag and forces a
            // rebuild. See `FEATURE-REQUEST.md` (FR-001).
            if meta.dropped {
                return Err(KVError::Serialization(format!(
                    "Field '{}' (id {}) in namespace {} was dropped; re-register it before activating so the index is rebuilt",
                    meta.field_name, field_id, namespace_id
                )));
            }
        }

        let store = self.get_store(namespace_id)?;

        // Everything below this point creates or maps files under
        // `field-index/{ns_id}/`. Two threads doing that for the same field truncate
        // each other's live mappings and the process dies with SIGBUS — see
        // `index_activate_lock`. Taken after `get_store`, which may itself wait
        // on the strictly-outermost `namespace_create_lock`.
        let _activate = self.index_activate_lock.lock();

        // Ensure the namespace's dense row-ID map is loaded before any
        // resolution happens — the write and replay paths resolve through it.
        store.ensure_rowmap(&self.index_manager.rowmap_path(namespace_id))?;

        // Open (or create) the file-backed mmap index in the field directory.
        let field_path = self.index_manager.field_path(namespace_id, field_id);
        let mut dyn_index = DynFieldIndex::open(value_type, &field_path).map_err(KVError::Io)?;
        dyn_index.set_overlay_budget(self.index_overlay_budget.clone());
        if dyn_index.damaged_at_open() > 0 {
            error!(
                "[activate_field_index] ns={namespace_id} field={field_id}: {} damaged entr(y/ies) in the field's index files \
                 were dropped at open (torn by a crash). Rows they held are missing from this field index until it is \
                 rebuilt; recording a full-rebuild gap.",
                dyn_index.damaged_at_open()
            );
            let gap = crate::db::indexing::manager::GapRecord {
                namespace_id,
                field_id,
                cause: crate::db::indexing::manager::GapCause::DamagedIndexFile,
                from: 0,
                to: 0,
                missing_segments: Vec::new(),
                detected_at_ms: crate::kv_store::current_epoch_millis(),
                repair: crate::db::indexing::manager::RepairMode::FullRebuild,
            };
            if let Err(e) = self.index_manager.record_gap(gap, Self::GAP_KEY_WORKLIST_CAP) {
                warn!("[activate_field_index] ns={namespace_id} field={field_id}: failed to record the damage gap: {e:?}");
            }
        }

        // Replay the WAL tail into dyn_index *before* registering the entry.
        // Recovering on the unshared dyn_index (not yet wrapped in Arc<RwLock<>>)
        // guarantees no concurrent put can race with recovery and overwrite a
        // newer value with a stale one from the WAL scan.
        if replay_wal {
            let wal_tail = self.wal_metadata.read().tail;
            if wal_tail > 0 {
                let checkpoint_state = self.index_manager.read_checkpoint_state(namespace_id, field_id, wal_tail);
                let checkpoint_offset = checkpoint_state.replay_offset();
                if checkpoint_offset < wal_tail {
                    let wal_head = self.wal_metadata.read().head;

                    // Report — but do not act on — a replay window the WAL can no
                    // longer satisfy. WAL GC reclaims a segment once its entries are
                    // persisted to the *LSM*; it never consults these checkpoint
                    // markers, so the segments that would heal this field's index may
                    // already be gone. `scan_entries` skips a missing segment as a
                    // hole and returns what it can, silently, which is exactly how
                    // this stayed invisible. Checking segment *presence* rather than
                    // `checkpoint_offset < wal_head` matters: GC reclaims
                    // fully-persisted segments out of order, so a hole can sit in the
                    // middle of the window while `head` is still below the checkpoint.
                    //
                    // Detection only — the replay below is unchanged, and the missing
                    // updates are unrecoverable from here. Recording, surfacing and
                    // repairing this is tracked in `FEATURE-REQUEST.md` (FR-001).
                    if let Some(gap) = crate::db::indexing::manager::detect_replay_gap(
                        checkpoint_state,
                        dyn_index.distinct_count() == 0,
                        wal_tail,
                        self.wal.missing_segments(checkpoint_offset, wal_tail),
                    ) {
                        error!(
                            "[activate_field_index] ns={} field={}: field index is INCOMPLETE and cannot be repaired \
                             by WAL replay — {} WAL segment(s) covering the replay window [{}, {}) have been reclaimed \
                             (missing segment ids: {:?}, checkpoint marker: {:?}). Every write recorded in those \
                             segments is absent from this field index, so queries on it will return incomplete \
                             results until the index is rebuilt. Re-index this field to restore full coverage.",
                            namespace_id,
                            field_id,
                            gap.missing_segments.len(),
                            gap.from,
                            gap.to,
                            gap.missing_segments,
                            checkpoint_state,
                        );
                    }

                    let entries = self.wal.scan_entries(wal_head.max(checkpoint_offset), wal_tail)?;
                    let mut affected_keys = std::collections::BTreeSet::<Vec<u8>>::new();
                    for (_, wal_entry) in &entries {
                        if wal_entry.namespace_id == namespace_id {
                            affected_keys.insert(wal_entry.key.clone());
                        }
                    }
                    debug!(
                        "[activate_field_index] ns={} field={} WAL replay: \
                         {} entries scanned, {} keys affected",
                        namespace_id,
                        field_id,
                        entries.len(),
                        affected_keys.len()
                    );

                    // Defense-in-depth tripwire for the `set_row_id_fn` ordering
                    // contract. If we are about to rebuild an already-populated
                    // field index through the dense RowMap fallback (no RowIdFn
                    // registered) while that RowMap has never allocated an ID,
                    // the existing on-disk entries were almost certainly built
                    // under a custom RowIdFn that should have been registered
                    // BEFORE activation — replaying now would mix row-ID schemes.
                    // Legitimate RowMap-based namespaces don't trip this: their
                    // RowMap is non-empty once any write has occurred.
                    if !affected_keys.is_empty() {
                        let scheme_mismatch = store.rowmap_active() && store.rowmap_is_empty() && dyn_index.distinct_count() > 0;
                        debug_assert!(
                            !scheme_mismatch,
                            "activate_field_index ns={namespace_id} field={field_id}: replaying a non-empty field index \
                             through the dense RowMap fallback with no RowIdFn registered and an unused RowMap — register \
                             set_row_id_fn BEFORE activate_field_index (see its ordering contract); replaying now mixes row-ID schemes"
                        );
                        if scheme_mismatch {
                            warn!(
                                "[activate_field_index] ns={} field={}: possible row-ID scheme mismatch — \
                                 rebuilding a non-empty index via the RowMap fallback with no RowIdFn set; \
                                 register set_row_id_fn before activation",
                                namespace_id, field_id
                            );
                        }
                    }

                    // Bound the blob growth this replay produces.
                    //
                    // Replay changes the index's in-memory write buffer, which is
                    // written to the append-only bitmap store only when it spills
                    // (the hard limit of the shared budget, set on `dyn_index`
                    // above) and once at the end. Each spill can leave dead
                    // copies behind, and the write path's backpressure valve is a
                    // no-op here: it signals the checkpoint *worker*, which is not
                    // started until every field has been activated.
                    //
                    // When every insert appended a whole bitmap, replaying one
                    // 5-distinct-value field over 16k documents reached 2.6 GB on
                    // disk and left the database unopenable. Buffering removes
                    // that shape, but a replay window larger than the budget still
                    // spills repeatedly, so keep the guard: compact inline, on the
                    // same `dead_bytes` cap the write path uses. We hold `dyn_index` exclusively here (it is not
                    // yet published behind the `RwLock`), so this needs no locks
                    // and cannot race a writer.
                    let waste_threshold = (self.config.threshold_config.index_blob_waste_threshold / 100.0).clamp(0.0, 1.0);
                    // A cap of 0 disables the *write-path* valve, which is a
                    // throughput choice. It must not disable this: replay is not
                    // the write path, and an unbounded replay leaves the database
                    // unopenable rather than merely slow.
                    let cap = match self.config.threshold_config.index_blob_backpressure_bytes {
                        0 => crate::db::config::DEFAULT_INDEX_BLOB_BACKPRESSURE_BYTES,
                        n => n,
                    };
                    let mut compactions = 0usize;

                    // Group the replay by VALUE before writing anything. Replay
                    // knows its complete key set up front, so it resolves every
                    // row first and then changes each value's bitmap once
                    // (`insert_many`), and clears all affected rows in one pass.
                    // The write buffer would coalesce per-key writes anyway; the
                    // grouping still saves a bitmap lookup per key.
                    let mut rows_by_value: HashMap<crate::field_index::IndexValue, Vec<u128>> = HashMap::new();
                    let mut rows_to_clear: Vec<u128> = Vec::new();

                    for key in &affected_keys {
                        match store.get(key)? {
                            // Live key: resolve its dense ID and bucket it under
                            // the value its current bytes extract to.
                            Some(ref bytes) => {
                                let row_id = store.resolve_row_id_alloc(key);
                                rows_to_clear.push(row_id);
                                if let Some(v) = extractor(bytes) {
                                    rows_by_value.entry(v).or_default().push(row_id);
                                }
                            }
                            // Deleted key: clear any existing entry without
                            // allocating a (dead) ID for a key that no longer exists.
                            None => {
                                if let Some(row_id) = store.resolve_row_id_get(key) {
                                    rows_to_clear.push(row_id);
                                }
                            }
                        }
                    }

                    // Clear every affected row first, so the scalar
                    // one-value-per-row invariant holds no matter which bucket
                    // each row used to live in. This must precede the inserts:
                    // clearing afterwards would undo them.
                    //
                    // Batched: one probe of every bucket for the whole row set,
                    // rather than one per row.
                    dyn_index.remove_all_for_rows(&rows_to_clear);

                    for (value, row_ids) in rows_by_value {
                        if dyn_index.reclaimable_dead_bytes() >= cap {
                            dyn_index.flush(&field_path).map_err(KVError::Io)?;
                            if dyn_index.maybe_compact(waste_threshold).map_err(KVError::Io)? {
                                dyn_index.flush(&field_path).map_err(KVError::Io)?;
                                compactions += 1;
                            }
                        }
                        if let Err(e) = dyn_index.insert_many(&value, &row_ids) {
                            warn!(
                                "[activate_field_index] index update rejected \
                                 ns={} field={}: {}",
                                namespace_id, field_id, e
                            );
                        }
                    }
                    dyn_index.flush(&field_path).map_err(KVError::Io)?;
                    if compactions > 0 {
                        debug!(
                            "[activate_field_index] ns={namespace_id} field={field_id}: \
                             compacted the index blob {compactions} time(s) during replay to keep it bounded"
                        );
                    }
                }
            }
        }

        let entry = IndexEntry {
            field_id,
            extractor,
            index: Arc::new(parking_lot::RwLock::new(dyn_index)),
        };
        // The field now needs the WAL from its checkpoint (`field_consumers`).
        // Registered with the field, so the two cannot disagree.
        let wal_tail = self.wal_metadata.read().tail;
        let position = match self.index_manager.read_checkpoint_state(namespace_id, field_id, wal_tail) {
            crate::db::indexing::manager::CheckpointState::At(p) => Some(p),
            crate::db::indexing::manager::CheckpointState::Absent => None,
            crate::db::indexing::manager::CheckpointState::Unusable => Some(0),
        };
        let mut namespace_index = store.namespace_index.write();
        namespace_index.register(entry);
        self.wal_flush_observer.add_field_consumer(namespace_id, field_id, position);
        Ok(())
    }

    /// Remove a previously-activated field index from the in-memory registry.
    ///
    /// After this call the field's bitmap is dropped and any predicate query
    /// that references it returns [`KVError::Query`] carrying
    /// [`crate::field_index::query::QueryError::InactiveField`].  The on-disk checkpoint
    /// files are left untouched; callers are responsible for removing them.
    pub fn deactivate_field_index(&self, namespace_id: u32, field_id: FieldId) -> Result<()> {
        let store = self.get_store(namespace_id)?;
        let mut namespace_index = store.namespace_index.write();
        namespace_index.deregister(field_id);
        self.wal_flush_observer.remove_field_consumer(namespace_id, field_id);
        Ok(())
    }

    /// Drop a field index: persist the drop, deregister it, and delete its
    /// on-disk directory.
    ///
    /// Unlike [`deactivate_field_index`](Self::deactivate_field_index) — which
    /// is an in-memory deregister a caller may legitimately use while a field is
    /// rebuilt — this is the permanent operation. The field's [`FieldMeta`] is
    /// retained in the schema so a later `register_field` reuses the same
    /// [`FieldId`], but it is marked `dropped`: excluded from the checkpoint
    /// worker and refused by `activate_field_index`.
    ///
    /// **Step order is load-bearing** (mirrors `remove_namespace`):
    ///
    /// 1. persist `dropped` to `config.json`,
    /// 2. deregister the in-memory index,
    /// 3. delete `field-index/{ns_id}/{field_id}/`.
    ///
    /// A crash anywhere in that sequence leaves the drop *recorded*, so
    /// `complete_interrupted_field_drops`
    /// finishes it at the next open. The reverse order — files first — would
    /// leave a live-looking field with no data, which reads as `Absent` + empty
    /// and is suppressed by `detect_replay_gap` as a normal first build: a
    /// silently incomplete index. See `FEATURE-REQUEST.md` (FR-001).
    ///
    /// Idempotent: dropping an already-dropped or unregistered field still
    /// deregisters and reclaims any leftover directory.
    pub fn drop_field_index(&self, namespace_id: u32, field_id: FieldId) -> Result<()> {
        // (1) Persist the drop before anything becomes unreachable.
        self.registry.write().mark_schema_field_dropped(namespace_id, field_id)?;

        // (2) Deregister in memory. Best-effort on the store lookup: a namespace
        // that is not open has no in-memory index to deregister, but its files
        // still need reclaiming below.
        if let Ok(store) = self.get_store(namespace_id) {
            store.namespace_index.write().deregister(field_id);
        }
        // Not active any more, whether or not its store is open.
        self.wal_flush_observer.remove_field_consumer(namespace_id, field_id);

        // (3) Reclaim disk.
        self.index_manager.remove_field_path(namespace_id, field_id)
    }

    /// Free every row ID of a namespace: replace its dense row map with an
    /// empty one, so the next field build allocates fresh dense IDs for the
    /// live keys only. Returns `false` (and does nothing) for a namespace whose
    /// row IDs come from a key-derived `RowIdFn`.
    ///
    /// Row IDs are referenced only by field-index bitmaps (gap records and build
    /// progress store keys), so this is safe exactly when no field index of the
    /// namespace exists: every registered field must be dropped (its `dropped`
    /// flag persisted and its directory removed by
    /// [`drop_field_index`](Self::drop_field_index)) and none active. Writers
    /// resolve a row ID only while the namespace has an active field, so none
    /// can race this. Refuses with an error otherwise.
    ///
    /// Holds `index_activate_lock` for the whole reset, so no field can be
    /// activated (and allocate an ID) in the middle of it. The reset itself is
    /// crash-safe (`RowMap::reset`).
    pub fn reset_rowmap(&self, namespace_id: u32) -> Result<bool> {
        let store = self.get_store(namespace_id)?;
        if store.row_id_fn.read().is_some() {
            return Ok(false);
        }
        let _activate = self.index_activate_lock.lock();
        let live_fields: Vec<String> = self
            .registry
            .read()
            .schema(namespace_id)
            .map(|s| s.list_fields().into_iter().filter(|f| !f.dropped).map(|f| f.field_name).collect())
            .unwrap_or_default();
        let ns_index = store.namespace_index.read();
        if !ns_index.is_empty() || !live_fields.is_empty() {
            return Err(KVError::Serialization(format!(
                "cannot reset the row map of namespace {namespace_id}: field indices still exist ({}); drop them first",
                live_fields.join(", ")
            )));
        }
        store.reset_rowmap(&self.index_manager.rowmap_path(namespace_id))?;
        info!("[INDEX] reset the row map of namespace {namespace_id}: every row ID freed");
        Ok(true)
    }

    /// Row IDs a namespace's dense row map has allocated (every key ever
    /// indexed, live or not), or `None` when it uses a key-derived `RowIdFn`
    /// or has no row map loaded (no field index was ever activated).
    pub fn rowmap_ids_allocated(&self, namespace_id: u32) -> Result<Option<u64>> {
        Ok(self.get_store(namespace_id)?.rowmap_ids_allocated())
    }

    /// Finish deleting the directories of field indices whose drop was
    /// interrupted by a crash, and reclaim any that were dropped before this
    /// cleanup existed.
    ///
    /// Runs at open, after the registry is loaded and before any field is
    /// activated. Idempotent — `remove_field_path` treats a missing directory as
    /// success — so the common case (nothing dropped, or every drop completed)
    /// costs one `remove_dir_all` per dropped field and touches no disk.
    ///
    /// Failures are logged, not propagated: a leftover directory wastes disk but
    /// cannot affect correctness, and refusing to open the database over it would
    /// be a far worse outcome.
    pub(in crate::db) fn complete_interrupted_field_drops(&self) {
        let dropped: Vec<(u32, FieldId)> = {
            let registry = self.registry.read();
            registry
                .list()
                .into_iter()
                .filter_map(|(_, ns_id)| registry.schema(ns_id).map(|s| (ns_id, s.dropped_field_ids())))
                .flat_map(|(ns_id, fields)| fields.into_iter().map(move |fid| (ns_id, fid)))
                .collect()
        };
        for (ns_id, field_id) in dropped {
            match self.index_manager.remove_field_path(ns_id, field_id) {
                Ok(()) => debug!("[INDEX] reclaimed dropped field index ns={ns_id} field={field_id}"),
                Err(e) => warn!("[INDEX] failed to reclaim dropped field index ns={ns_id} field={field_id}: {e:?}"),
            }
        }
    }
}
