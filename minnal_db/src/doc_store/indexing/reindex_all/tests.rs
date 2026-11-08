//! Tests of `reindex-all` and drop-all: the rebuilt indexes, a crash after
//! every step, and the conflicts with other jobs.

use super::*;
use crate::doc_store::schema::IndexType;
use crate::doc_store::tests::test_support::*;
use crate::namespace_lifecycle::BuildStatus;

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
/// map holds the live documents only, no job is live and the store is
/// `Created`.
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
    assert!(store.jobs.live(ns_id).unwrap().is_empty(), "{what}: no job is live");
    assert_eq!(store.schemas.state("events"), Some(NamespaceState::Created), "{what}");
}

#[tokio::test]
async fn reindex_all_frees_the_row_ids_of_deleted_documents() {
    let db_dir = TempDir::new().unwrap();
    let (store, live) = seeded(db_dir.path()).await;
    let ns_id = store.load_schema("events").unwrap().ns_id.unwrap();
    assert_eq!(store.db.rowmap_ids_allocated(ns_id).unwrap(), Some(DOCS as u64));

    store.reindex_all_attribute_indices("events").await.unwrap();
    assert_rebuilt(&store, &live, "after reindex-all").await;
    drop(store);

    let store = open_fresh(db_dir.path()).await;
    store.run_pending_jobs().await.unwrap();
    assert_rebuilt(&store, &live, "after a restart").await;
}

/// A crash after each step (two drops, the reset, two re-adds, two builds)
/// is finished at the next start: the store ends with both indexes, exact
/// answers and a compacted row map.
#[tokio::test]
async fn an_interrupted_reindex_all_is_finished_at_the_next_start() {
    for stop in 0..7usize {
        let db_dir = TempDir::new().unwrap();
        let (store, live) = seeded(db_dir.path()).await;
        store.reindex_all_until("events", true, stop).await.unwrap();
        assert_eq!(store.schemas.state("events"), Some(NamespaceState::IndexMaintenance));
        drop(store);

        let store = open_fresh(db_dir.path()).await;
        store.run_pending_jobs().await.unwrap();
        assert_rebuilt(&store, &live, &format!("crash after step {stop}")).await;
    }
}

/// A drop-all cut short is finished at the next start too, and leaves no
/// index and an empty row map.
#[tokio::test]
async fn an_interrupted_drop_all_is_finished_at_the_next_start() {
    for stop in 0..3usize {
        let db_dir = TempDir::new().unwrap();
        let (store, _) = seeded(db_dir.path()).await;
        store.reindex_all_until("events", false, stop).await.unwrap();
        drop(store);

        let store = open_fresh(db_dir.path()).await;
        store.run_pending_jobs().await.unwrap();
        let schema = store.load_schema("events").unwrap();
        assert!(schema.indices.is_empty(), "crash after step {stop}");
        assert_eq!(store.db.rowmap_ids_allocated(schema.ns_id.unwrap()).unwrap(), Some(0));
        assert_eq!(store.schemas.state("events"), Some(NamespaceState::Created));
    }
}

/// A query on a field whose index is still building (or whose build
/// failed) reports it in `degraded_fields`: the index holds only the
/// documents the build reached. Other fields are not tainted. A field a
/// `reindex-all` has not rebuilt yet is degraded the same way.
#[tokio::test]
async fn a_field_being_built_is_reported_degraded() {
    let db_dir = TempDir::new().unwrap();
    let (store, live) = seeded(db_dir.path()).await;
    let ns_id = store.load_schema("events").unwrap().ns_id.unwrap();
    let fid = store
        .db
        .list_index_fields(ns_id)
        .iter()
        .find(|f| f.field_name == "active")
        .unwrap()
        .field_id;
    let set = |status: BuildStatus| {
        let mut record = JobRecord::started(Some("active".to_owned()), serde_json::Value::Null);
        record.status = status;
        store.maintenance.put(JobKey::field_build(ns_id, fid), &record).unwrap();
    };
    assert!(ids(&store, "active = true").await.1.is_empty());
    set(BuildStatus::Running);
    assert_eq!(ids(&store, "active = true").await.1, vec!["active".to_owned()]);
    assert_eq!(ids(&store, "active = true AND tier = 1").await.1, vec!["active".to_owned()]);
    assert!(ids(&store, "tier = 1").await.1.is_empty(), "an untouched field does not taint the query");
    set(BuildStatus::Failed);
    assert_eq!(ids(&store, "NOT active = true").await.1, vec!["active".to_owned()]);
    set(BuildStatus::Complete);
    assert!(ids(&store, "active = true").await.1.is_empty());

    // Both indexes dropped, re-added and not built yet.
    store.reindex_all_until("events", true, 5).await.unwrap();
    assert_eq!(
        ids(&store, "active = true AND tier = 1").await.1,
        vec!["active".to_owned(), "tier".to_owned()]
    );
    store.run_pending_jobs().await.unwrap();
    assert_rebuilt(&store, &live, "after the builds").await;
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
    assert_eq!(store.drop_all_attribute_indices("events").await.unwrap().len(), 2);
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

/// While a build of the store is live, `reindex-all` is refused before it
/// changes anything. It used to drop the indexes, delete the running build's
/// directory under it, fail, and leave the indexes out of the schema.
#[tokio::test]
async fn reindex_all_is_refused_while_a_build_runs() {
    let db_dir = TempDir::new().unwrap();
    let (store, live) = seeded(db_dir.path()).await;
    let spec = IndexSpec {
        field: "extra".to_owned(),
        index_type: IndexType::Int,
    };
    store.start_field_build("events", spec).await.unwrap();
    assert!(matches!(
        store.reindex_all_attribute_indices("events").await,
        Err(DocStoreError::JobConflict { running, .. }) if running.starts_with("field_build")
    ));
    assert_eq!(store.load_schema("events").unwrap().indices.len(), 3);
    store.run_pending_jobs().await.unwrap();
    store.drop_index("events", "extra").await.unwrap();
    store.reindex_all_attribute_indices("events").await.unwrap();
    assert_rebuilt(&store, &live, "after the build").await;
}

/// A run that failed after dropping some indexes is taken over by the next
/// one, from its own list. Starting over from the schema, which no longer
/// lists the dropped ones, would lose them for good.
#[tokio::test]
async fn a_reindex_all_that_failed_part_way_is_taken_over_by_the_next() {
    let db_dir = TempDir::new().unwrap();
    let (store, live) = seeded(db_dir.path()).await;
    let ns_id = store.load_schema("events").unwrap().ns_id.unwrap();
    store.reindex_all_until("events", true, 1).await.unwrap();
    store
        .jobs
        .finish(&store, "events", JobKey::reindex_all(ns_id), Err("simulated".to_owned()))
        .unwrap();
    assert_eq!(store.load_schema("events").unwrap().indices.len(), 1, "one index dropped, then failed");
    store.reindex_all_attribute_indices("events").await.unwrap();
    assert_rebuilt(&store, &live, "after taking over the failed run").await;
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
        store.reindex_all_attribute_indices("rw").await.unwrap();
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
