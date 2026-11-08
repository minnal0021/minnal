//! Field-index lifecycle: add, drop, and the background (re)build that
//! populates a newly added index.

use crate::doc_store::*;
use crate::namespace::NamespaceId;

impl DocStore {
    // ── Admin: index management ────────────────────────────────────────────

    /// Drop an index from an existing store.
    ///
    /// Removes the `IndexSpec` from `indices` and **demotes the field to a
    /// non-indexed `AttributeDef`** in `attributes`, preserving the type
    /// declaration.  The field's data remains in every stored document; only
    /// the live index files (`field-index/{ns_id}/{field_id}/`) are deleted.
    ///
    /// The field registration in `config.json` is left in place so that a
    /// later [`add_index`] call for the same field reuses the same `field_id`.
    ///
    /// [`add_index`]: DocStore::add_index
    pub fn drop_index(&self, namespace: &str, field: &str) -> Result<(), DocStoreError> {
        info!("dropping index '{}' from namespace '{}'", field, namespace);
        // Persist the schema BEFORE anything is deregistered or deleted.
        //
        // The reverse order (the original) had a crash window in which the files
        // were gone but this schema still listed the index: the field would be
        // re-activated at the next open, find no checkpoint and no data, and read
        // as `Absent` + empty — the shape `detect_replay_gap` suppresses as a
        // normal first build. The index would come up silently incomplete. See
        // `FEATURE-REQUEST.md` (FR-001) — *Dropped-index cleanup*.
        let ns_id = self.schemas.update_doc(namespace, |schema| {
            let ns_id = schema.ns_id.ok_or_else(|| DocStoreError::MissingNsId {
                namespace: namespace.to_owned(),
            })?;

            // Confirm the index exists in the schema
            let idx_pos = schema
                .indices
                .iter()
                .position(|s| s.field == field)
                .ok_or_else(|| DocStoreError::IndexNotFound {
                    namespace: namespace.to_owned(),
                    field: field.to_owned(),
                })?;

            let spec = schema.indices.remove(idx_pos);

            // Demote to a non-indexed attribute so the field is still documented
            // in the schema. Skip if an AttributeDef with this name already exists
            // (shouldn't happen, but be safe).
            if !schema.attributes.iter().any(|a| a.name == spec.field) {
                schema.attributes.push(crate::doc_store::schema::AttributeDef {
                    name: spec.field.clone(),
                    attr_type: index_type_to_attr_type(spec.index_type),
                    description: Some("previously indexed; field data is still present in stored documents".to_owned()),
                });
            }
            Ok(ns_id)
        })?;

        // Drop through the engine, which owns the index paths: it persists the
        // field's `dropped` flag, deregisters the in-memory bitmap, and deletes
        // `field-index/{ns_id}/{field_id}/`. A crash part-way leaves the drop recorded,
        // so the next open finishes it.
        //
        // The build record goes first. Left behind (a crash after the engine's
        // drop), a running record would make a later re-add of the field resume
        // the old build instead of starting a new one; a missing record only
        // means there is no build to resume.
        let fields = self.db.list_index_fields(ns_id);
        if let Some(meta) = fields.iter().find(|f| f.field_name == field) {
            self.maintenance.delete(TaskKey::field_build(ns_id, meta.field_id))?;
            self.db.drop_field_index(ns_id, meta.field_id)?;
        }

        info!("index '{}' dropped from namespace '{}'", field, namespace);
        Ok(())
    }

    /// Add a new index to an existing store and build it in the background.
    ///
    /// Returns an [`IndexBuildHandle`] immediately.  The index is activated
    /// before the handle is returned, so new writes are indexed straight away.
    /// The background task then walks every existing document and indexes its
    /// current value in this field (`Db::reindex_field`, under the key's stripe),
    /// without rewriting the document.
    ///
    /// Returns [`DocStoreError::IndexAlreadyExists`] if the field is already
    /// indexed.
    pub async fn add_index(&self, namespace: &str, spec: IndexSpec) -> Result<IndexBuildHandle, DocStoreError> {
        self.add_index_inner(namespace, spec, false).await
    }

    /// [`add_index`](Self::add_index). `resuming` skips the "a build is already
    /// in progress" guard: a re-add that finishes an interrupted run (an
    /// interrupted `reindex-all`) finds its own running record from before
    /// the crash, and must take it over rather than refuse.
    pub(super) async fn add_index_inner(&self, namespace: &str, spec: IndexSpec, resuming: bool) -> Result<IndexBuildHandle, DocStoreError> {
        // Held by the build task until it ends: registering the field creates
        // its index directory, and the build reads and writes the store.
        let write = self.writes.enter(namespace)?;
        // Also held until the build ends. An earlier build of this field can
        // still be running after a drop until it notices; adding the field
        // again meanwhile would give one index two builders.
        let claim = self
            .builds
            .claim(namespace, &spec.field)
            .ok_or_else(|| DocStoreError::IndexBuildInProgress {
                namespace: namespace.to_owned(),
                field: spec.field.clone(),
            })?;
        let schema = self.load_schema(namespace)?;
        let ns_id = schema.ns_id.ok_or_else(|| DocStoreError::MissingNsId {
            namespace: namespace.to_owned(),
        })?;

        // Saving the schema does not run `validate`, so check the name here.
        crate::doc_store::schema::validate_index_field_name(&spec.field)?;

        if schema.indices.iter().any(|s| s.field == spec.field) {
            return Err(DocStoreError::IndexAlreadyExists {
                namespace: namespace.to_owned(),
                field: spec.field.clone(),
            });
        }

        // Enforce the per-namespace index cap here too: saving the schema does not
        // validate, so without this check `add_index` could push the namespace
        // past MAX_INDICES one field at a time even though create/import reject
        // it. (Checked again when the schema is saved.)
        if schema.indices.len() >= crate::doc_store::schema::MAX_INDICES {
            return Err(DocStoreError::Schema(SchemaError::TooManyIndices {
                count: schema.indices.len() + 1,
                max: crate::doc_store::schema::MAX_INDICES,
            }));
        }

        // Guard against a second caller racing in while a background build is
        // still running (e.g. after a drop_index + immediate re-add). Read from
        // the build's record rather than in-process state, so the check also
        // works when called directly (not via the API).
        if !resuming {
            let ivt = to_ivt(spec.index_type);
            if let Ok(fid) = self.db.register_index_field(ns_id, &spec.field, ivt) {
                let running = self.maintenance.get::<FieldBuildRecord>(TaskKey::field_build(ns_id, fid))?;
                if running.is_some_and(|r| r.is_running()) {
                    return Err(DocStoreError::IndexBuildInProgress {
                        namespace: namespace.to_owned(),
                        field: spec.field.clone(),
                    });
                }
            }
        }

        let ivt = to_ivt(spec.index_type);
        let field_id = self.db.register_index_field(ns_id, &spec.field, ivt)?;
        info!(
            "adding index '{}' (type={:?}, field_id={}) to namespace '{}' (ns_id={})",
            spec.field, spec.index_type, field_id, namespace, ns_id
        );
        let extractor = json_extractor(spec.field.clone(), spec.index_type);
        // Starts empty unless an interrupted build is being taken over, whose
        // index holds what it did before the crash (the WAL replay brings that
        // up to date). The store takes writes meanwhile only in the first case;
        // see `activate_field_index_for_build`.
        let record_key = TaskKey::field_build(ns_id, field_id);
        let interrupted = self.maintenance.get::<FieldBuildRecord>(record_key)?.filter(FieldBuildRecord::is_running);
        let takes_over_a_build = interrupted.as_ref().is_some_and(|r| r.resume_after_hex.is_some());
        if takes_over_a_build {
            self.db.activate_field_index(ns_id, field_id, ivt, extractor).await?;
        } else {
            self.db.activate_field_index_for_build(ns_id, field_id, ivt, extractor).await?;
        }

        // Record the build as running BEFORE the schema lists the index. The
        // build task saves its own first record only after counting every
        // document, so without this a crash in between left an index the
        // schema lists, half built, with no record for `resume_pending_builds`
        // to find: it was never resumed and silently stayed incomplete. A
        // record already running (an interrupted build being taken over) is
        // kept, so its resume point survives.
        let start = match interrupted {
            Some(record) => record,
            None => {
                let record = FieldBuildRecord::started();
                self.maintenance.put(record_key, &record)?;
                record
            }
        };

        // Persist updated schema — if the field was previously declared as a
        // non-indexed attribute, move it to indices (remove from attributes).
        // Checked again under the schema write: a concurrent add of the same
        // field, or one that reached the cap, must not be saved twice.
        self.schemas.update_doc(namespace, |schema| {
            if schema.indices.iter().any(|s| s.field == spec.field) {
                return Err(DocStoreError::IndexAlreadyExists {
                    namespace: namespace.to_owned(),
                    field: spec.field.clone(),
                });
            }
            if schema.indices.len() >= crate::doc_store::schema::MAX_INDICES {
                return Err(DocStoreError::Schema(SchemaError::TooManyIndices {
                    count: schema.indices.len() + 1,
                    max: crate::doc_store::schema::MAX_INDICES,
                }));
            }
            schema.attributes.retain(|a| a.name != spec.field);
            schema.indices.push(spec.clone());
            Ok(())
        })?;

        // ── Background rebuild ────────────────────────────────────────────
        // Check for a previously interrupted build and resume from where it left off.
        let resume_after: Option<Vec<u8>> = start.resume_after_hex.as_deref().and_then(hex_to_bytes);
        if resume_after.is_some() {
            info!("resuming interrupted index build for namespace '{}' field '{}'", namespace, spec.field);
        } else {
            info!("starting background index build for namespace '{}' field '{}'", namespace, spec.field);
        }

        let mem = Arc::new(InMemoryProgress::with_initial(start.total, start.indexed));
        let record = Arc::new(RecordProgress::new(Arc::clone(&self.maintenance), record_key, 1_000, &start));
        let observer: Arc<dyn IndexProgressObserver> = Arc::new(ChainedObserver(vec![
            Arc::clone(&mem) as Arc<dyn IndexProgressObserver>,
            record as Arc<dyn IndexProgressObserver>,
        ]));

        let observer_clone = Arc::clone(&observer);
        let fail_observer = Arc::clone(&observer);
        let db_clone = Arc::clone(&self.db);
        let ns_name = namespace.to_owned();
        let field_name = spec.field.clone();
        let key_type = schema.key_type;

        let task = tokio::spawn(async move {
            let result = rebuild_index_for_namespace(db_clone, ns_name, ns_id, key_type, field_id, resume_after, observer_clone, &write).await;
            if let Err(ref e) = result {
                // Notify the whole chain (in-memory *and* the record), so the
                // failure is saved, not just visible to live pollers.
                fail_observer.on_status(BuildStatus::Failed, Some(&e.to_string()));
            }
            drop(write);
            drop(claim);
            result
        });

        Ok(IndexBuildHandle {
            namespace: namespace.to_owned(),
            field: field_name,
            mem,
            task,
        })
    }

    /// Resume any index builds that were interrupted by a previous shutdown.
    ///
    /// Scans all persisted schemas.  For each index whose build record is
    /// still running, a new background task is spawned that continues from the
    /// record's resume point rather than rescanning every document.
    ///
    /// Call this once at startup, right after [`open`] / [`open_with_config`],
    /// and store the returned handles the same way you would handles returned
    /// by [`add_index`].
    ///
    /// [`open`]: DocStore::open
    /// [`open_with_config`]: DocStore::open_with_config
    /// [`add_index`]: DocStore::add_index
    pub async fn resume_pending_builds(&self) -> Result<Vec<IndexBuildHandle>, DocStoreError> {
        // Finish any `reindex-all` a crash interrupted first: it re-adds indexes
        // and starts their builds, which the loop below must not start again.
        let mut handles = self.resume_interrupted_reindex_all().await?;
        let started: std::collections::HashSet<(String, String)> = handles.iter().map(|h| (h.namespace.clone(), h.field.clone())).collect();

        for schema in self.load_all_schemas()? {
            let ns_id = match schema.ns_id {
                Some(id) => id,
                None => continue,
            };

            for spec in &schema.indices {
                if started.contains(&(schema.namespace.clone(), spec.field.clone())) {
                    continue;
                }
                // Being dropped: nothing to resume.
                let Ok(write) = self.writes.enter(&schema.namespace) else { continue };
                let Some(claim) = self.builds.claim(&schema.namespace, &spec.field) else {
                    continue;
                };
                let ivt = to_ivt(spec.index_type);
                let field_id = self.db.register_index_field(ns_id, &spec.field, ivt)?;
                let record_key = TaskKey::field_build(ns_id, field_id);

                // No record, or a finished one: no build of this field is left
                // to finish. A record is a WAL-backed write, so it is never torn.
                let Some(start) = self.maintenance.get::<FieldBuildRecord>(record_key)?.filter(FieldBuildRecord::is_running) else {
                    continue;
                };

                info!(
                    "resuming interrupted index build: namespace='{}' field='{}' progress={}/{}",
                    schema.namespace, spec.field, start.indexed, start.total
                );
                let resume_after = start.resume_after_hex.as_deref().and_then(hex_to_bytes);

                let mem = Arc::new(InMemoryProgress::with_initial(start.total, start.indexed));
                let record = Arc::new(RecordProgress::new(Arc::clone(&self.maintenance), record_key, 1_000, &start));
                let observer: Arc<dyn IndexProgressObserver> = Arc::new(ChainedObserver(vec![
                    Arc::clone(&mem) as Arc<dyn IndexProgressObserver>,
                    record as Arc<dyn IndexProgressObserver>,
                ]));

                let observer_clone = Arc::clone(&observer);
                let fail_observer = Arc::clone(&observer);
                let db_clone = Arc::clone(&self.db);
                let ns_name = schema.namespace.clone();
                let field_name = spec.field.clone();
                let key_type = schema.key_type;

                let task = tokio::spawn(async move {
                    let result =
                        rebuild_index_for_namespace(db_clone, ns_name, ns_id, key_type, field_id, resume_after, observer_clone, &write).await;
                    if let Err(ref e) = result {
                        // Notify the whole chain (in-memory *and* the record), so
                        // the failure is saved, not just visible to live pollers.
                        fail_observer.on_status(BuildStatus::Failed, Some(&e.to_string()));
                    }
                    drop(write);
                    drop(claim);
                    result
                });

                handles.push(IndexBuildHandle {
                    namespace: schema.namespace.clone(),
                    field: field_name,
                    mem,
                    task,
                });
            }
        }

        Ok(handles)
    }

    /// A field of `namespace` whose index build is running in this process, if
    /// any. `reindex-all` is refused while there is one.
    pub fn field_index_build_running(&self, namespace: &str) -> Option<String> {
        self.builds.any(namespace)
    }

    /// The saved record of a field's index build, or `None` if no build of it
    /// was started.
    ///
    /// The fallback for the progress API when no build of the field is running
    /// in this process (e.g. after it completed, or before a restart resumes it).
    pub fn index_build_record(&self, namespace: &str, field: &str) -> Result<Option<FieldBuildRecord>, DocStoreError> {
        let schema = self.view_schema(namespace)?;
        let Some(ns_id) = schema.ns_id else { return Ok(None) };
        let fields = self.db.list_index_fields(ns_id);
        let Some(meta) = fields.iter().find(|f| f.field_name == field) else {
            return Ok(None);
        };
        self.maintenance.get(TaskKey::field_build(ns_id, meta.field_id))
    }
}

// ── Background index rebuild ──────────────────────────────────────────────────

/// Scan every document in `ns_name` and re-put it so that the new extractor
/// is called for each, populating the freshly activated index with historical
/// data.
///
/// Progress is reported via `observer` (atomics and the build's record) every 1 000
/// documents and on terminal status changes.  When `resume_after` is `Some(key)`,
/// all documents with keys ≤ that key are skipped (they were already processed
/// before the previous shutdown).
#[allow(clippy::too_many_arguments)]
/// Number of `(key, value)` pairs fetched per cursor page during an index
/// rebuild. Bounds peak memory to roughly one page of documents instead of the
/// whole namespace, and matches the progress/yield cadence below.
const REBUILD_PAGE_SIZE: usize = 1_000;

/// Smallest key strictly greater than `key`, used to advance a (inclusive)
/// scan cursor past an already-processed key. Appending a `0x00` byte yields a
/// key that sorts immediately after `key` in lexicographic order.
fn successor_key(key: &[u8]) -> Vec<u8> {
    let mut next = Vec::with_capacity(key.len() + 1);
    next.extend_from_slice(key);
    next.push(0);
    next
}

#[allow(clippy::too_many_arguments)] // cohesive set of build parameters; not worth a struct
async fn rebuild_index_for_namespace(
    db: Arc<AsyncDb>,
    ns_name: String,
    ns_id: NamespaceId,
    key_type: KeyType,
    field_id: FieldId,
    resume_after: Option<Vec<u8>>,
    observer: Arc<dyn IndexProgressObserver>,
    write: &WriteGuard,
) -> Result<(), DocStoreError> {
    // Never get-or-create: a build outliving its store would bring it back.
    let dropped = || DocStoreError::NotFound { namespace: ns_name.clone() };
    let ns = db.existing_namespace(ns_name.clone()).await?.ok_or_else(dropped)?;

    // ── Pass 1: count keys, cursor-paginated. We need the exact total up front
    // (it drives the build's percent-complete), but materialising the whole
    // namespace just to count it is the memory spike this rebuild used to incur.
    // Scanning a page at a time and discarding each page bounds peak memory to
    // one page. We also count how many keys were already processed on a resumed
    // build, to seed `indexed`.
    let mut total: u64 = 0;
    let mut already_done: u64 = 0;
    {
        let mut cursor: Option<Vec<u8>> = None;
        loop {
            if write.store_dropping() {
                return Err(dropped());
            }
            let (pairs, next_cursor) = ns.scan(cursor, None, REBUILD_PAGE_SIZE).await?;
            for (key, _value) in &pairs {
                total += 1;
                if resume_after.as_ref().is_some_and(|r| key <= r) {
                    already_done += 1;
                }
            }
            match next_cursor {
                Some(c) => cursor = Some(c),
                None => break,
            }
        }
    }

    info!("index build started: namespace='{}' field_id={} total={} docs", ns_name, field_id, total);

    // Seed the observer with the counts from the scan, then announce Running.
    // The observer (its `RecordProgress` link) is the single source of truth
    // for the saved record: seeding before `on_status(Running)` makes the
    // running record carry the real total and resume point.
    observer.on_progress(already_done, total, false, resume_after.as_deref());
    observer.on_status(BuildStatus::Running, None);

    let mut indexed: u64 = already_done;
    // The rows this build sets live in memory until an index checkpoint writes
    // them out; they are not in the WAL, so a crash before then loses them. The
    // saved resume point therefore names a key only once a checkpoint that
    // started after the key was indexed has finished. Naming it sooner skipped,
    // on resume, every key a crash had taken back out of the index.
    let mut unsaved: std::collections::VecDeque<(u64, Vec<u8>)> = std::collections::VecDeque::new();

    // ── Pass 2: rebuild, one cursor page at a time. On a resumed build the
    // cursor seeks strictly past the last processed key (the scan cursor is
    // inclusive, so start at its successor) instead of re-scanning and skipping.
    let mut cursor: Option<Vec<u8>> = resume_after.as_deref().map(successor_key);
    'pages: loop {
        let (pairs, next_cursor) = ns.scan(cursor, None, REBUILD_PAGE_SIZE).await?;
        if pairs.is_empty() {
            break;
        }
        for (key, _value) in pairs {
            // The store is being dropped, and the drop waits for this build.
            if write.store_dropping() {
                return Err(dropped());
            }
            // Index the key's CURRENT value in this one field, under its key
            // stripe (`Db::reindex_field`). This used to re-put the scanned bytes,
            // which was wrong twice over: re-putting unchanged bytes is a no-op for
            // the index (`DynFieldIndex::update` skips old == new), so only the
            // documents WAL replay happened to cover at activation got indexed —
            // on a store whose WAL had been reclaimed, 28 of 300; and writing back
            // the bytes read a page earlier overwrote any newer write to that key.
            let outcome = db.reindex_field(ns_id, field_id, key.clone()).await?;
            if outcome == crate::field_index::FieldReindexOutcome::FieldNotActive {
                return Err(crate::db::error::KVError::Serialization(format!(
                    "index build for namespace '{ns_name}' field_id={field_id}: the field was deactivated mid-build"
                ))
                .into());
            }
            indexed += 1;
            if indexed.is_multiple_of(1_000) {
                unsaved.push_back((db.index_checkpoints_started(), key.clone()));
            }
            let mut resume_point = None;
            while unsaved.front().is_some_and(|(started, _)| db.index_checkpoint_covers(*started)) {
                resume_point = unsaved.pop_front().map(|(_, key)| key);
            }
            // The observer saves the record on its own cadence (every `every_n`).
            observer.on_progress(indexed, total, false, resume_point.as_deref());

            // Log progress and yield every 1 000 documents.
            if indexed.is_multiple_of(1_000) {
                info!(
                    "index build progress: namespace='{}' field_id={} {}/{} docs ({:.1}%)",
                    ns_name,
                    field_id,
                    indexed,
                    total,
                    if total > 0 { indexed as f64 / total as f64 * 100.0 } else { 0.0 }
                );
                tokio::task::yield_now().await;
            }
        }
        match next_cursor {
            Some(c) => cursor = Some(c),
            None => break 'pages,
        }
    }

    // Write the built rows out before the record says complete: a complete
    // build is never resumed, so rows a crash took back would stay missing.
    db.checkpoint_index().await?;
    // Mark build complete.  The observer persists the terminal record from the
    // latest snapshot it has been fed via `on_progress`.
    observer.on_status(BuildStatus::Complete, None);

    info!(
        "index build complete: namespace='{}' field_id={} indexed={} docs",
        ns_name, field_id, indexed
    );
    let _ = key_type;
    Ok(())
}

#[cfg(test)]
mod tests;
