//! Field-index lifecycle: add, drop, and the background (re)build that
//! populates a newly added index.

use super::*;

impl DocStore {
    // ── Admin: index management ────────────────────────────────────────────

    /// Drop an index from an existing store.
    ///
    /// Removes the `IndexSpec` from `indices` and **demotes the field to a
    /// non-indexed `AttributeDef`** in `attributes`, preserving the type
    /// declaration.  The field's data remains in every stored document; only
    /// the live index files (`index/{ns_id}/{field_id}/`) are deleted.
    ///
    /// The field registration in `config.json` is left in place so that a
    /// later [`add_index`] call for the same field reuses the same `field_id`.
    ///
    /// [`add_index`]: DocStore::add_index
    pub fn drop_index(&self, namespace: &str, field: &str) -> Result<(), DocStoreError> {
        info!("dropping index '{}' from namespace '{}'", field, namespace);
        let mut schema = self.load_schema(namespace)?;
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

        // Deactivate the in-memory bitmap and delete on-disk checkpoint files.
        let fields = self.db.list_index_fields(ns_id);
        if let Some(meta) = fields.iter().find(|f| f.field_name == field) {
            self.db.deactivate_field_index(ns_id, meta.field_id)?;
            let index_dir = self.db_path.join("index").join(ns_id.to_string()).join(meta.field_id.to_string());
            if index_dir.exists() {
                std::fs::remove_dir_all(&index_dir)?;
            }
        }

        schema.save(&self.schema_dir)?;
        info!("index '{}' dropped from namespace '{}'", field, namespace);
        Ok(())
    }

    /// Add a new index to an existing store and build it in the background.
    ///
    /// Returns an [`IndexBuildHandle`] immediately.  The index is activated
    /// before the handle is returned, so new writes are indexed straight away.
    /// The background task then scans all existing documents and re-puts each
    /// one so the extractor is called for historical data too.
    ///
    /// Returns [`DocStoreError::IndexAlreadyExists`] if the field is already
    /// indexed.
    pub async fn add_index(&self, namespace: &str, spec: IndexSpec) -> Result<IndexBuildHandle, DocStoreError> {
        let mut schema = self.load_schema(namespace)?;
        let ns_id = schema.ns_id.ok_or_else(|| DocStoreError::MissingNsId {
            namespace: namespace.to_owned(),
        })?;

        if schema.indices.iter().any(|s| s.field == spec.field) {
            return Err(DocStoreError::IndexAlreadyExists {
                namespace: namespace.to_owned(),
                field: spec.field.clone(),
            });
        }

        // Enforce the per-namespace index cap here too: `schema.save()` below does
        // not validate (only `validate_and_save` does), so without this check
        // `add_index` could push the namespace past MAX_INDICES one field at a
        // time even though create/import reject it.
        if schema.indices.len() >= crate::doc_store::schema::MAX_INDICES {
            return Err(DocStoreError::Schema(SchemaError::TooManyIndices {
                count: schema.indices.len() + 1,
                max: crate::doc_store::schema::MAX_INDICES,
            }));
        }

        // Guard against a second caller racing in while a background build is
        // still running (e.g. after a drop_index + immediate re-add).
        // We detect this via the on-disk progress file rather than in-process
        // state so the check also works when called directly (not via the API).
        {
            let ivt = to_ivt(spec.index_type);
            if let Ok(fid) = self.db.register_index_field(ns_id, &spec.field, ivt) {
                let progress_path = build_progress_path(&self.db_path, ns_id, fid);
                if read_disk_progress(&progress_path).map(|p| p.status == "in_progress").unwrap_or(false) {
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
        self.db.activate_field_index(ns_id, field_id, ivt, extractor).await?;

        // Persist updated schema — if the field was previously declared as a
        // non-indexed attribute, move it to indices (remove from attributes).
        schema.attributes.retain(|a| a.name != spec.field);
        schema.indices.push(spec.clone());
        schema.save(&self.schema_dir)?;

        // ── Background rebuild ────────────────────────────────────────────
        // Check for a previously interrupted build and resume from where it left off.
        let progress_path = build_progress_path(&self.db_path, ns_id, field_id);
        let resume_after: Option<Vec<u8>> = read_disk_progress(&progress_path)
            .filter(|p| p.status == "in_progress")
            .and_then(|p| p.last_key_hex)
            .and_then(|h| hex_to_bytes(&h));
        if resume_after.is_some() {
            info!("resuming interrupted index build for namespace '{}' field '{}'", namespace, spec.field);
        } else {
            info!("starting background index build for namespace '{}' field '{}'", namespace, spec.field);
        }

        let mem = Arc::new(InMemoryProgress::new());
        let disk = Arc::new(DiskProgress::new(&progress_path, 1_000));
        let observer: Arc<dyn IndexProgressObserver> = Arc::new(ChainedObserver(vec![
            Arc::clone(&mem) as Arc<dyn IndexProgressObserver>,
            Arc::clone(&disk) as Arc<dyn IndexProgressObserver>,
        ]));

        let observer_clone = Arc::clone(&observer);
        let fail_observer = Arc::clone(&observer);
        let db_clone = Arc::clone(&self.db);
        let ns_name = namespace.to_owned();
        let field_name = spec.field.clone();
        let key_type = schema.key_type;

        let task = tokio::spawn(async move {
            let result = rebuild_index_for_namespace(db_clone, ns_name, key_type, field_id, resume_after, observer_clone).await;
            if let Err(ref e) = result {
                // Notify the whole chain (in-memory *and* disk), so the failure
                // is persisted, not just visible to live pollers.
                fail_observer.on_status(BuildStatus::Failed, Some(&e.to_string()));
            }
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
    /// Scans all persisted schemas.  For each index whose
    /// `build_progress.json` has `status == "in_progress"`, a new background
    /// task is spawned that continues from the last checkpointed key rather
    /// than rescanning every document.
    ///
    /// Call this once at startup, right after [`open`] / [`open_with_config`],
    /// and store the returned handles the same way you would handles returned
    /// by [`add_index`].
    ///
    /// [`open`]: DocStore::open
    /// [`open_with_config`]: DocStore::open_with_config
    /// [`add_index`]: DocStore::add_index
    pub async fn resume_pending_builds(&self) -> Result<Vec<IndexBuildHandle>, DocStoreError> {
        let mut handles = Vec::new();

        for schema in self.load_all_schemas()? {
            let ns_id = match schema.ns_id {
                Some(id) => id,
                None => continue,
            };

            for spec in &schema.indices {
                let ivt = to_ivt(spec.index_type);
                let field_id = self.db.register_index_field(ns_id, &spec.field, ivt)?;
                let progress_path = build_progress_path(&self.db_path, ns_id, field_id);

                let Some(disk) = read_disk_progress(&progress_path) else { continue };
                if disk.status != "in_progress" {
                    continue;
                }

                info!(
                    "resuming interrupted index build: namespace='{}' field='{}' progress={}/{}",
                    schema.namespace, spec.field, disk.indexed, disk.total
                );
                let resume_after = disk.last_key_hex.as_deref().and_then(hex_to_bytes);

                let mem = Arc::new(InMemoryProgress::with_initial(disk.total, disk.indexed));
                let disk_obs = Arc::new(DiskProgress::new(&progress_path, 1_000));
                let observer: Arc<dyn IndexProgressObserver> = Arc::new(ChainedObserver(vec![
                    Arc::clone(&mem) as Arc<dyn IndexProgressObserver>,
                    Arc::clone(&disk_obs) as Arc<dyn IndexProgressObserver>,
                ]));

                let observer_clone = Arc::clone(&observer);
                let fail_observer = Arc::clone(&observer);
                let db_clone = Arc::clone(&self.db);
                let ns_name = schema.namespace.clone();
                let field_name = spec.field.clone();
                let key_type = schema.key_type;

                let task = tokio::spawn(async move {
                    let result = rebuild_index_for_namespace(db_clone, ns_name, key_type, field_id, resume_after, observer_clone).await;
                    if let Err(ref e) = result {
                        // Notify the whole chain (in-memory *and* disk), so the
                        // failure is persisted, not just visible to live pollers.
                        fail_observer.on_status(BuildStatus::Failed, Some(&e.to_string()));
                    }
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

    /// Return the persisted build progress for a specific index, or `None` if
    /// no `build_progress.json` exists for that field.
    ///
    /// This is the fallback used by the progress API when no active in-memory
    /// handle exists (e.g. after the build completed and was drained on shutdown).
    pub fn index_build_disk_progress(&self, namespace: &str, field: &str) -> Option<DiskBuildProgress> {
        let schema = self.load_schema(namespace).ok()?;
        let ns_id = schema.ns_id?;
        let fields = self.db.list_index_fields(ns_id);
        let meta = fields.iter().find(|f| f.field_name == field)?;
        let path = build_progress_path(&self.db_path, ns_id, meta.field_id);
        read_disk_progress(&path)
    }
}

// ── Background index rebuild ──────────────────────────────────────────────────

/// Scan every document in `ns_name` and re-put it so that the new extractor
/// is called for each, populating the freshly activated index with historical
/// data.
///
/// Progress is reported via `observer` (e.g. atomics + disk JSON) every 1 000
/// documents and on terminal status changes.  When `resume_after` is `Some(key)`,
/// all documents with keys ≤ that key are skipped (they were already processed
/// before the previous shutdown).
#[allow(clippy::too_many_arguments)]
/// Number of `(key, value)` pairs fetched per cursor page during an index
/// rebuild. Bounds peak memory to roughly one page of documents instead of the
/// whole namespace, and matches the progress/yield cadence below.
pub(super) const REBUILD_PAGE_SIZE: usize = 1_000;

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
    key_type: KeyType,
    field_id: FieldId,
    resume_after: Option<Vec<u8>>,
    observer: Arc<dyn IndexProgressObserver>,
) -> Result<(), DocStoreError> {
    let ns = db.namespace(ns_name.clone()).await?;

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
    // The observer (its `DiskProgress` link) is the single source of truth for
    // persistence: seeding before `on_status(Running)` makes the initial
    // in_progress record carry the real total/resume point.
    observer.on_progress(already_done, total, false, resume_after.as_deref());
    observer.on_status(BuildStatus::Running, None);

    let mut indexed: u64 = already_done;

    // ── Pass 2: rebuild, one cursor page at a time. On a resumed build the
    // cursor seeks strictly past the last processed key (the scan cursor is
    // inclusive, so start at its successor) instead of re-scanning and skipping.
    let mut cursor: Option<Vec<u8>> = resume_after.as_deref().map(successor_key);
    'pages: loop {
        let (pairs, next_cursor) = ns.scan(cursor, None, REBUILD_PAGE_SIZE).await?;
        if pairs.is_empty() {
            break;
        }
        for (key, value) in pairs {
            // Re-put triggers the active extractors, populating the new index.
            ns.put(key.clone(), value).await?;
            indexed += 1;
            // The observer persists to disk on its own cadence (every `every_n`).
            observer.on_progress(indexed, total, false, Some(&key));

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
