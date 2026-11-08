//! `reindex-all`: drop every field index of a store, free its row IDs, and
//! rebuild every index — crash-safe through an intent file.
//!
//! Dropping an index demotes it to a plain attribute in the saved schema, so a
//! crash between the drops and the re-adds would leave the store with no
//! indexes at all. The intent file (`field-index/{ns_id}/reindex_all.json`) records
//! the index specs and how far the run got; [`DocStore::resume_pending_builds`]
//! finishes an interrupted run at startup.
//!
//! Between the drops and the re-adds no field index of the namespace exists, so
//! no row ID is referenced anywhere: that is when the row map is reset
//! ([`AsyncDb::reset_rowmap`]), and the rebuilt indexes get dense IDs for the
//! live documents only (FR-006).

use crate::doc_store::*;
use crate::namespace::NamespaceId;

/// Name of the intent file in the namespace's index directory.
const INTENT_FILE: &str = "reindex_all.json";

/// How far a `reindex-all` run got. Each phase is finished before the file
/// moves to the next, so a resumed run repeats at most one idempotent step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum Phase {
    /// Dropping the indexes; the row map has not been reset yet.
    Dropping,
    /// Indexes dropped and the row map reset; re-adding the indexes.
    Readding,
}

/// Contents of the intent file.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct Intent {
    /// The indexes to rebuild, as the schema listed them before the run.
    specs: Vec<IndexSpec>,
    phase: Phase,
}

fn intent_path(db_path: &Path, ns_id: NamespaceId) -> PathBuf {
    crate::layout::namespace_index_dir(&crate::layout::index_root(db_path), ns_id).join(INTENT_FILE)
}

fn write_intent(path: &Path, intent: &Intent) -> Result<(), DocStoreError> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let bytes = serde_json::to_vec_pretty(intent).map_err(|e| DocStoreError::Io(std::io::Error::other(e)))?;
    crate::support::write_atomic_durable(path, &bytes)?;
    Ok(())
}

fn read_intent(path: &Path) -> Option<Intent> {
    let bytes = std::fs::read(path).ok()?;
    match serde_json::from_slice(&bytes) {
        Ok(intent) => Some(intent),
        Err(e) => {
            log::warn!("ignoring unreadable reindex-all intent file {}: {e}", path.display());
            None
        }
    }
}

fn remove_intent(path: &Path) -> Result<(), DocStoreError> {
    match std::fs::remove_file(path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e.into()),
    }
    if let Some(dir) = path.parent() {
        crate::support::fsync_dir(dir)?;
    }
    Ok(())
}

impl DocStore {
    /// Drop every field index of `namespace`, free its row IDs, and rebuild
    /// every index in the background.
    ///
    /// Returns one build handle per index, as [`add_index`](Self::add_index)
    /// does. Crash-safe: an interrupted run is finished by
    /// [`resume_pending_builds`](Self::resume_pending_builds) at the next start.
    /// A store with no indexes returns no handles and changes nothing.
    pub async fn reindex_all_attribute_indices(&self, namespace: &str) -> Result<Vec<IndexBuildHandle>, DocStoreError> {
        self.reindex_all_until(namespace, None).await
    }

    /// [`reindex_all_attribute_indices`](Self::reindex_all_attribute_indices),
    /// stopping after `stop_after` steps (each drop, the reset and each re-add
    /// is one step) to simulate a crash in tests.
    pub(super) async fn reindex_all_until(&self, namespace: &str, stop_after: Option<usize>) -> Result<Vec<IndexBuildHandle>, DocStoreError> {
        // One reindex-all of a store at a time, apart from its amendments and drop.
        let _lifecycle = self.lifecycle.lock(namespace).await;
        let _write = self.writes.enter(namespace)?;
        let schema = self.load_schema(namespace)?;
        let ns_id = schema.ns_id.ok_or_else(|| DocStoreError::MissingNsId {
            namespace: namespace.to_owned(),
        })?;
        // Dropping an index whose build is running deletes its directory under
        // it; refuse before anything changes.
        if let Some(field) = self.builds.any(namespace) {
            return Err(DocStoreError::IndexBuildInProgress {
                namespace: namespace.to_owned(),
                field,
            });
        }
        let path = intent_path(&self.db_path, ns_id);
        // A run that stopped part-way left its intent, which lists every index it
        // set out to rebuild, some of them dropped already. Finish that one: a new
        // intent from the schema as it is now would leave those out for good.
        if let Some(intent) = read_intent(&path) {
            log::warn!(
                "reindex-all '{namespace}': finishing an earlier run that stopped part-way (phase {:?})",
                intent.phase
            );
            return self.drive_reindex_all(namespace, ns_id, intent, stop_after).await;
        }
        if schema.indices.is_empty() {
            return Ok(Vec::new());
        }
        let intent = Intent {
            specs: schema.indices.clone(),
            phase: Phase::Dropping,
        };
        write_intent(&path, &intent)?;
        info!("reindex-all '{namespace}': rebuilding {} index(es)", intent.specs.len());
        self.drive_reindex_all(namespace, ns_id, intent, stop_after).await
    }

    /// Run (or resume) a `reindex-all` from `intent`.
    async fn drive_reindex_all(
        &self,
        namespace: &str,
        ns_id: NamespaceId,
        mut intent: Intent,
        stop_after: Option<usize>,
    ) -> Result<Vec<IndexBuildHandle>, DocStoreError> {
        let path = intent_path(&self.db_path, ns_id);
        let mut steps = 0usize;
        let stop = |steps: usize| stop_after.is_some_and(|n| steps >= n);

        if intent.phase == Phase::Dropping {
            for spec in &intent.specs {
                if stop(steps) {
                    return Ok(Vec::new());
                }
                // Already dropped by an interrupted run: nothing to do.
                if self.load_schema(namespace)?.indices.iter().any(|s| s.field == spec.field) {
                    self.drop_index(namespace, &spec.field)?;
                }
                steps += 1;
            }
            if stop(steps) {
                return Ok(Vec::new());
            }
            // No field index of the namespace exists now, so no row ID is
            // referenced: free them all before the rebuild allocates new ones.
            self.db.reset_rowmap(ns_id)?;
            intent.phase = Phase::Readding;
            write_intent(&path, &intent)?;
            steps += 1;
        }

        let mut handles = Vec::new();
        for spec in &intent.specs {
            if stop(steps) {
                return Ok(handles);
            }
            // Already re-added by an interrupted run: its build resumes from
            // its own progress file.
            if !self.load_schema(namespace)?.indices.iter().any(|s| s.field == spec.field) {
                handles.push(self.add_index_inner(namespace, spec.clone(), true).await?);
            }
            steps += 1;
        }
        remove_intent(&path)?;
        info!("reindex-all '{namespace}': every index re-added; builds running");
        Ok(handles)
    }

    /// Finish every `reindex-all` that a crash interrupted. Called first by
    /// [`resume_pending_builds`](Self::resume_pending_builds); returns the build
    /// handles it started, keyed so that function does not start them twice.
    pub(super) async fn resume_interrupted_reindex_all(&self) -> Result<Vec<IndexBuildHandle>, DocStoreError> {
        let mut handles = Vec::new();
        for schema in self.load_all_schemas()? {
            let Some(ns_id) = schema.ns_id else { continue };
            let path = intent_path(&self.db_path, ns_id);
            let Some(intent) = read_intent(&path) else { continue };
            info!("resuming interrupted reindex-all of '{}' (phase {:?})", schema.namespace, intent.phase);
            handles.extend(self.drive_reindex_all(&schema.namespace, ns_id, intent, None).await?);
        }
        Ok(handles)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doc_store::schema::IndexType;
    use crate::doc_store::tests::test_support::*;

    const DOCS: u128 = 300;

    /// A uuid-keyed store with `active` and `tier` indexes over `DOCS`
    /// documents, 80% of them then deleted. Returns the store and the live
    /// `(id, active, tier)` set.
    async fn seeded(db_dir: &Path) -> (DocStore, Vec<(u128, bool, i64)>) {
        let store = open_fresh(db_dir).await;
        let mut schema = make_schema(
            "events",
            vec![
                IndexSpec {
                    field: "active".to_owned(),
                    index_type: IndexType::Bool,
                },
                IndexSpec {
                    field: "tier".to_owned(),
                    index_type: IndexType::Int,
                },
            ],
        );
        schema.key_type = KeyType::Uuid;
        store.create(schema).await.unwrap();
        let mut live = Vec::new();
        for i in 0..DOCS {
            let id = i.wrapping_mul(0x9E37_79B9_7F4A_7C15_F39C_C060_5CED_C835);
            let (active, tier) = (i % 3 == 0, (i % 4) as i64);
            store
                .put("events", DocId::Uuid(id), serde_json::json!({"active": active, "tier": tier}))
                .await
                .unwrap();
            if i % 5 == 0 {
                live.push((id, active, tier));
            } else {
                store.delete("events", DocId::Uuid(id)).await.unwrap();
            }
        }
        (store, live)
    }

    async fn ids(store: &DocStore, predicate: &str) -> (Vec<u128>, Vec<String>) {
        let page = store
            .query(
                "events",
                predicate,
                Pagination {
                    page_no: 1,
                    page_size: 10_000,
                },
            )
            .await
            .unwrap();
        let mut ids: Vec<u128> = page
            .results
            .iter()
            .map(|(id, _)| match id {
                DocId::Uuid(u) => *u,
                other => panic!("expected a uuid id, got {other:?}"),
            })
            .collect();
        ids.sort_unstable();
        (ids, page.degraded_fields)
    }

    /// Every index is back, answers match the live documents exactly, the row
    /// map holds the live documents only, and no intent file is left.
    async fn assert_rebuilt(store: &DocStore, live: &[(u128, bool, i64)], what: &str) {
        let schema = store.load_schema("events").unwrap();
        let fields: Vec<&str> = schema.indices.iter().map(|s| s.field.as_str()).collect();
        assert_eq!(fields.len(), 2, "{what}: both indexes are back ({fields:?})");
        let ns_id = schema.ns_id.unwrap();
        let mut want: Vec<u128> = live.iter().filter(|d| d.1).map(|d| d.0).collect();
        want.sort_unstable();
        let (got, degraded) = ids(store, "active = true").await;
        assert_eq!(got, want, "{what}: active = true");
        assert!(degraded.is_empty(), "{what}: finished builds are not degraded ({degraded:?})");
        for t in 0..4 {
            let mut want: Vec<u128> = live.iter().filter(|d| d.2 == t).map(|d| d.0).collect();
            want.sort_unstable();
            assert_eq!(ids(store, &format!("tier = {t}")).await.0, want, "{what}: tier = {t}");
        }
        assert_eq!(
            store.db.rowmap_ids_allocated(ns_id).unwrap(),
            Some(live.len() as u64),
            "{what}: the row map numbers the live documents only"
        );
        assert!(!intent_path(&store.db_path, ns_id).exists(), "{what}: intent file removed");
    }

    #[tokio::test]
    async fn reindex_all_frees_the_row_ids_of_deleted_documents() {
        let db_dir = TempDir::new().unwrap();
        let (store, live) = seeded(db_dir.path()).await;
        let ns_id = store.load_schema("events").unwrap().ns_id.unwrap();
        assert_eq!(store.db.rowmap_ids_allocated(ns_id).unwrap(), Some(DOCS as u64));

        for h in store.reindex_all_attribute_indices("events").await.unwrap() {
            h.wait().await.unwrap();
        }
        assert_rebuilt(&store, &live, "after reindex-all").await;
        drop(store);

        let store = open_fresh(db_dir.path()).await;
        assert!(store.resume_pending_builds().await.unwrap().is_empty(), "nothing left to resume");
        assert_rebuilt(&store, &live, "after a restart").await;
    }

    /// A crash after each step (two drops, the reset, two re-adds) is finished
    /// at the next start: the store ends with both indexes, exact answers and a
    /// compacted row map.
    #[tokio::test]
    async fn an_interrupted_reindex_all_is_finished_at_the_next_start() {
        for stop in 0..5usize {
            let db_dir = TempDir::new().unwrap();
            let (store, live) = seeded(db_dir.path()).await;
            for h in store.reindex_all_until("events", Some(stop)).await.unwrap() {
                // The build is killed with the process.
                h.task.abort();
                let _ = h.task.await;
            }
            drop(store);

            let store = open_fresh(db_dir.path()).await;
            for h in store.resume_pending_builds().await.unwrap() {
                h.wait().await.unwrap();
            }
            assert_rebuilt(&store, &live, &format!("crash after step {stop}")).await;
        }
    }

    /// The two crash windows around a re-add: the build killed before it reports
    /// anything (only the `in_progress` record `add_index` writes first), and a
    /// crash after that record but before the schema lists the index. Both are
    /// finished at the next start.
    #[tokio::test]
    async fn a_re_add_interrupted_before_its_build_reports_is_finished() {
        let db_dir = TempDir::new().unwrap();
        let (store, live) = seeded(db_dir.path()).await;
        let ns_id = store.load_schema("events").unwrap().ns_id.unwrap();
        // Two drops, the reset, and the first re-add; its build is killed at once.
        for h in store.reindex_all_until("events", Some(4)).await.unwrap() {
            h.task.abort();
            let _ = h.task.await;
        }
        let blank = DiskBuildProgress {
            status: "in_progress".to_owned(),
            total: 0,
            indexed: 0,
            last_key_hex: None,
            error: None,
        };
        let fields = store.db.list_index_fields(ns_id);
        let field_dir = |name: &str| {
            let fid = fields.iter().find(|f| f.field_name == name).unwrap().field_id;
            build_progress_path(&store.db_path, ns_id, fid)
        };
        // `active`: as if the process died before the build wrote anything.
        std::fs::write(field_dir("active"), serde_json::to_vec(&blank).unwrap()).unwrap();
        // `tier`: re-registered (which clears its `dropped` flag) and its record
        // written, the schema not yet saved.
        store.db.register_index_field(ns_id, "tier", to_ivt(IndexType::Int)).unwrap();
        let tier = field_dir("tier");
        std::fs::create_dir_all(tier.parent().unwrap()).unwrap();
        std::fs::write(&tier, serde_json::to_vec(&blank).unwrap()).unwrap();
        drop(store);

        let store = open_fresh(db_dir.path()).await;
        for h in store.resume_pending_builds().await.unwrap() {
            h.wait().await.unwrap();
        }
        assert_rebuilt(&store, &live, "re-adds interrupted before reporting").await;
    }

    /// A query on a field whose index is still building (or whose build
    /// failed) reports it in `degraded_fields`: the index holds only the
    /// documents the build reached. Other fields are not tainted.
    #[tokio::test]
    async fn a_field_being_built_is_reported_degraded() {
        let db_dir = TempDir::new().unwrap();
        let (store, _) = seeded(db_dir.path()).await;
        let ns_id = store.load_schema("events").unwrap().ns_id.unwrap();
        let fid = store
            .db
            .list_index_fields(ns_id)
            .iter()
            .find(|f| f.field_name == "active")
            .unwrap()
            .field_id;
        let progress = build_progress_path(&store.db_path, ns_id, fid);
        let set = |status: &str| {
            let p = DiskBuildProgress {
                status: status.to_owned(),
                total: 10,
                indexed: 3,
                last_key_hex: None,
                error: None,
            };
            std::fs::write(&progress, serde_json::to_vec(&p).unwrap()).unwrap();
        };
        assert!(ids(&store, "active = true").await.1.is_empty());
        set("in_progress");
        assert_eq!(ids(&store, "active = true").await.1, vec!["active".to_owned()]);
        assert_eq!(ids(&store, "active = true AND tier = 1").await.1, vec!["active".to_owned()]);
        assert!(ids(&store, "tier = 1").await.1.is_empty(), "an untouched field does not taint the query");
        set("failed");
        assert_eq!(ids(&store, "NOT active = true").await.1, vec!["active".to_owned()]);
        set("complete");
        assert!(ids(&store, "active = true").await.1.is_empty());
    }

    #[tokio::test]
    async fn drop_all_resets_the_row_map() {
        let db_dir = TempDir::new().unwrap();
        let (store, _) = seeded(db_dir.path()).await;
        let ns_id = store.load_schema("events").unwrap().ns_id.unwrap();
        let before = store.rowmap_stats("events").await.unwrap().unwrap();
        assert_eq!(
            (before.ids_allocated, before.live_docs, before.dead_ids),
            (DOCS as u64, DOCS as u64 / 5, DOCS as u64 * 4 / 5)
        );
        assert_eq!(store.drop_all_attribute_indices("events").unwrap().len(), 2);
        assert_eq!(store.db.rowmap_ids_allocated(ns_id).unwrap(), Some(0));
        let after = store.rowmap_stats("events").await.unwrap().unwrap();
        assert_eq!((after.ids_allocated, after.dead_ids), (0, 0));
        assert!(
            after.bytes_on_disk < before.bytes_on_disk,
            "{} -> {} bytes",
            before.bytes_on_disk,
            after.bytes_on_disk
        );
    }

    /// A power loss can leave a build's progress record empty, as its writes
    /// are not fsynced. The next start rebuilds that field from the first key;
    /// it used to skip the field, leaving its index half built for good.
    #[tokio::test]
    async fn a_build_with_an_unreadable_progress_record_is_rebuilt_at_the_next_start() {
        let db_dir = TempDir::new().unwrap();
        let (store, live) = seeded(db_dir.path()).await;
        for h in store.reindex_all_attribute_indices("events").await.unwrap() {
            // The build is killed with the process.
            h.task.abort();
            let _ = h.task.await;
        }
        let ns_id = store.load_schema("events").unwrap().ns_id.unwrap();
        for field in store.db.list_index_fields(ns_id) {
            std::fs::write(build_progress_path(&store.db_path, ns_id, field.field_id), b"").unwrap();
        }
        drop(store);

        let store = open_fresh(db_dir.path()).await;
        let handles = store.resume_pending_builds().await.unwrap();
        assert_eq!(handles.len(), 2, "both fields are rebuilt");
        for h in handles {
            h.wait().await.unwrap();
        }
        assert_rebuilt(&store, &live, "after rebuilding from an unreadable record").await;
    }

    /// While a build of the store runs, `reindex-all` is refused before it
    /// changes anything. It used to drop the indexes, delete the running build's
    /// directory under it, fail, and leave the indexes out of the schema.
    #[tokio::test]
    async fn reindex_all_is_refused_while_a_build_runs() {
        let db_dir = TempDir::new().unwrap();
        let (store, _) = seeded(db_dir.path()).await;
        let running = store.builds.claim("events", "tier").unwrap();
        assert!(matches!(
            store.reindex_all_attribute_indices("events").await,
            Err(DocStoreError::IndexBuildInProgress { field, .. }) if field == "tier"
        ));
        assert_eq!(store.load_schema("events").unwrap().indices.len(), 2);
        drop(running);
        for h in store.reindex_all_attribute_indices("events").await.unwrap() {
            h.wait().await.unwrap();
        }
        assert_eq!(store.load_schema("events").unwrap().indices.len(), 2);
    }

    /// A run that stopped after dropping some indexes is finished by the next
    /// call, from its own list. Starting over from the schema, which no longer
    /// listed the dropped ones, lost them for good.
    #[tokio::test]
    async fn a_reindex_all_that_stopped_part_way_is_finished_by_the_next_call() {
        let db_dir = TempDir::new().unwrap();
        let (store, live) = seeded(db_dir.path()).await;
        store.reindex_all_until("events", Some(1)).await.unwrap();
        assert_eq!(store.load_schema("events").unwrap().indices.len(), 1, "one index dropped, then stopped");
        for h in store.reindex_all_attribute_indices("events").await.unwrap() {
            h.wait().await.unwrap();
        }
        assert_eq!(store.load_schema("events").unwrap().indices.len(), 2);
        assert_rebuilt(&store, &live, "after finishing the earlier run").await;
    }

    /// Writers changing and deleting keys while `reindex-all` runs: afterwards
    /// every key is listed under exactly its current value. A re-added field's
    /// WAL replay used to leave a deleted key in the index, and a later write of
    /// it put the row in two buckets.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn reindex_all_under_concurrent_writes_leaves_no_stale_entries() {
        use std::sync::Arc;
        let db_dir = TempDir::new().unwrap();
        let store = Arc::new(open_fresh(db_dir.path()).await);
        let mut schema = make_schema(
            "rw",
            vec![IndexSpec {
                field: "y".to_owned(),
                index_type: IndexType::Int,
            }],
        );
        schema.key_type = KeyType::Str;
        store.create(schema).await.unwrap();
        const KEYS: usize = 1500;
        for i in 0..KEYS {
            store
                .put(
                    "rw",
                    DocId::Str(StrKey::new(&format!("k{i:04}")).unwrap()),
                    serde_json::json!({"y": i % 10}),
                )
                .await
                .unwrap();
        }
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut writers = Vec::new();
        for w in 0..6u64 {
            let (store, stop) = (Arc::clone(&store), Arc::clone(&stop));
            writers.push(tokio::spawn(async move {
                let mut n = w;
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    n = n.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let id = DocId::Str(StrKey::new(&format!("k{:04}", (n >> 33) as usize % KEYS)).unwrap());
                    if n % 10 == 0 {
                        let _ = store.delete("rw", id).await;
                    } else {
                        store.put("rw", id, serde_json::json!({"y": (n >> 40) % 10})).await.unwrap();
                    }
                }
            }));
        }
        for _ in 0..8 {
            for h in store.reindex_all_attribute_indices("rw").await.unwrap() {
                h.wait().await.unwrap();
            }
        }
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        for w in writers {
            w.await.unwrap();
        }
        let mut stale = Vec::new();
        for v in 0..10 {
            let page = store.query("rw", &format!("y = {v}"), Pagination::new(1, 10_000)).await.unwrap();
            for (id, doc) in page.results {
                if doc["y"] != v {
                    stale.push((id, v, doc["y"].clone()));
                }
            }
        }
        let mut listed = 0;
        for v in 0..10 {
            listed += store.query("rw", &format!("y = {v}"), Pagination::new(1, 10_000)).await.unwrap().total;
        }
        let live = store.count_docs("rw").await.unwrap();
        assert!(
            stale.is_empty() && listed == live,
            "stale {stale:?}, {listed} listed for {live} documents"
        );
    }
}
