//! What a store keeps across a restart: schemas, deletes, row ids.

use crate::doc_store::tests::test_support::*;
use crate::doc_store::*;

// ── Schema persists across restart ──────────────────────────────────────

#[tokio::test]
async fn test_schema_survives_restart() {
    let db_dir = TempDir::new().unwrap();

    // First open: create store, write data
    {
        let store = open_fresh(db_dir.path()).await;
        store
            .create(make_schema(
                "users",
                vec![IndexSpec {
                    field: "active".to_owned(),
                    index_type: IndexType::Bool,
                }],
            ))
            .await
            .unwrap();
        store.put("users", DocId::U64(1), serde_json::json!({"active": true})).await.unwrap();
        store.put("users", DocId::U64(2), serde_json::json!({"active": false})).await.unwrap();
    }

    // Second open: no create() call — schema must be loaded automatically
    {
        let store = open_fresh(db_dir.path()).await;
        let result = store.query("users", "active = true", Pagination::default()).await.unwrap();
        assert_eq!(result.total, 1);
        match result.results[0].0 {
            DocId::U64(1) => {}
            other => panic!("unexpected id {:?}", other),
        }
    }
}

/// Deletions must survive restart: a range scan on the reopened store
/// must not return documents deleted in the previous session.
#[tokio::test]
async fn test_range_query_after_delete_survives_restart() {
    let db_dir = TempDir::new().unwrap();

    // Session 1: create store, insert docs, delete one, then drop.
    {
        let store = open_fresh(db_dir.path()).await;
        store.create(make_schema("docs", vec![])).await.unwrap();

        for i in 1u64..=5 {
            store.put("docs", DocId::U64(i), serde_json::json!({"n": i})).await.unwrap();
        }

        store.delete("docs", DocId::U64(3)).await.unwrap();
        store.delete("docs", DocId::U64(5)).await.unwrap();

        // Sanity check within the same session.
        let result = store.scan_range("docs", DocId::U64(1), None, None, 100).await.unwrap();
        assert_eq!(result.results.len(), 3, "same-session range should show 3 docs");
    }

    // Session 2: reopen — WAL recovery runs, deletions must hold.
    {
        let store = open_fresh(db_dir.path()).await;

        let result = store.scan_range("docs", DocId::U64(1), None, None, 100).await.unwrap();
        let ids: Vec<u64> = result
            .results
            .iter()
            .map(|(id, _)| match id {
                DocId::U64(v) => *v,
                _ => panic!(),
            })
            .collect();
        assert_eq!(ids, vec![1, 2, 4], "after restart, deleted docs 3 and 5 must not appear in range scan");

        // Point-gets must also confirm deletion.
        assert_eq!(store.get("docs", DocId::U64(3)).await.unwrap(), None);
        assert_eq!(store.get("docs", DocId::U64(5)).await.unwrap(), None);
        // Surviving docs must still be readable.
        assert!(store.get("docs", DocId::U64(1)).await.unwrap().is_some());
        assert!(store.get("docs", DocId::U64(2)).await.unwrap().is_some());
        assert!(store.get("docs", DocId::U64(4)).await.unwrap().is_some());
    }
}

/// Prefix scan after delete must survive a restart.
#[tokio::test]
async fn test_prefix_scan_after_delete_survives_restart() {
    let db_dir = TempDir::new().unwrap();

    {
        let store = open_fresh(db_dir.path()).await;
        store.create(make_schema("docs", vec![])).await.unwrap();

        for i in 1u64..=4 {
            store.put("docs", DocId::U64(i), serde_json::json!({"v": i})).await.unwrap();
        }

        store.delete("docs", DocId::U64(2)).await.unwrap();
    }

    {
        let store = open_fresh(db_dir.path()).await;

        let page = store.scan_prefix("docs", vec![], None, 100).await.unwrap();
        let ids: Vec<u64> = page
            .results
            .iter()
            .map(|(id, _)| match id {
                DocId::U64(v) => *v,
                _ => panic!(),
            })
            .collect();
        assert_eq!(ids, vec![1, 3, 4], "after restart, deleted doc 2 must not appear in prefix scan");
    }
}

/// Index (predicate) query after delete must survive a restart.
///
/// The bitmap index is rebuilt from WAL on reopen, so a document deleted
/// before shutdown must not appear in the predicate results after restart.
#[tokio::test]
async fn test_predicate_query_after_delete_survives_restart() {
    let db_dir = TempDir::new().unwrap();

    {
        let store = open_fresh(db_dir.path()).await;
        store
            .create(make_schema(
                "users",
                vec![IndexSpec {
                    field: "active".to_owned(),
                    index_type: IndexType::Bool,
                }],
            ))
            .await
            .unwrap();

        store
            .put("users", DocId::U64(1), serde_json::json!({"active": true,  "name": "Alice"}))
            .await
            .unwrap();
        store
            .put("users", DocId::U64(2), serde_json::json!({"active": true,  "name": "Bob"}))
            .await
            .unwrap();
        store
            .put("users", DocId::U64(3), serde_json::json!({"active": false, "name": "Carol"}))
            .await
            .unwrap();
        store
            .put("users", DocId::U64(4), serde_json::json!({"active": true,  "name": "Dave"}))
            .await
            .unwrap();

        // Sanity: 3 active docs before delete.
        let before = store.query("users", "active = true", Pagination::default()).await.unwrap();
        assert_eq!(before.total, 3);

        // Delete one active doc.
        store.delete("users", DocId::U64(2)).await.unwrap();

        // Same session: only 2 active docs.
        let after = store.query("users", "active = true", Pagination::default()).await.unwrap();
        assert_eq!(after.total, 2);
    }

    // Reopen — index is rebuilt from the WAL tail.
    {
        let store = open_fresh(db_dir.path()).await;

        let result = store.query("users", "active = true", Pagination::default()).await.unwrap();
        let mut ids: Vec<u64> = result
            .results
            .iter()
            .map(|(id, _)| match id {
                DocId::U64(v) => *v,
                _ => panic!(),
            })
            .collect();
        ids.sort();
        assert_eq!(ids, vec![1, 4], "after restart, deleted doc 2 must not appear in predicate query");

        // The inactive doc must still be found.
        let inactive = store.query("users", "active = false", Pagination::default()).await.unwrap();
        assert_eq!(inactive.total, 1);
        match inactive.results[0].0 {
            DocId::U64(3) => {}
            other => panic!("expected doc 3, got {:?}", other),
        }

        // Point-get must confirm doc 2 is gone.
        assert_eq!(store.get("users", DocId::U64(2)).await.unwrap(), None);
    }
}

/// Delete every on-disk per-field index `checkpoint` file under `db_dir`,
/// rewinding each field's checkpoint offset to 0. This simulates a hard
/// crash that occurred after writes but before any index checkpoint (the
/// `Drop`/`shutdown` checkpoint never ran), forcing `activate_field_index`
/// to replay the WAL tail on the next open.
fn rewind_index_checkpoints(db_dir: &Path) {
    let index_dir = crate::layout::index_root(db_dir);
    let Ok(namespaces) = std::fs::read_dir(&index_dir) else { return };
    for ns in namespaces.flatten() {
        let Ok(fields) = std::fs::read_dir(ns.path()) else { continue };
        for field in fields.flatten() {
            let _ = std::fs::remove_file(field.path().join("checkpoint"));
        }
    }
}

/// The custom (key-type-derived) row-ID function must be
/// installed *before* field indexes are activated, so the activation-time
/// WAL-tail replay resolves row IDs the same way prior writes did. If the
/// index is activated first, replay falls back to the dense `RowMap` and
/// indexes the replayed keys under IDs (0, 1, 2, …) that disagree with the
/// key-derived IDs — corrupting query key-resolution.
///
/// The replay only fires when the index checkpoint is behind the WAL tail,
/// which normally happens only on a hard crash; we reproduce that by
/// rewinding the on-disk checkpoint between sessions.
#[tokio::test]
async fn row_id_fn_installed_before_index_activation_replay() {
    let db_dir = TempDir::new().unwrap();

    {
        let store = open_fresh(db_dir.path()).await;
        store
            .create(make_schema(
                "users",
                vec![IndexSpec {
                    field: "active".to_owned(),
                    index_type: IndexType::Bool,
                }],
            ))
            .await
            .unwrap();

        // Non-sequential ids so the dense RowMap ids (0, 1, 2) the buggy
        // path would assign are clearly different from the key-derived ids.
        store
            .put("users", DocId::U64(100), serde_json::json!({"active": true,  "name": "A"}))
            .await
            .unwrap();
        store
            .put("users", DocId::U64(200), serde_json::json!({"active": false, "name": "B"}))
            .await
            .unwrap();
        store
            .put("users", DocId::U64(300), serde_json::json!({"active": true,  "name": "C"}))
            .await
            .unwrap();
    }

    // Simulate a crash after the writes but before an index checkpoint.
    rewind_index_checkpoints(db_dir.path());

    {
        let store = open_fresh(db_dir.path()).await;
        let result = store.query("users", "active = true", Pagination::default()).await.unwrap();
        let mut ids: Vec<u64> = result
            .results
            .iter()
            .map(|(id, _)| match id {
                DocId::U64(v) => *v,
                other => panic!("expected U64 id, got {other:?}"),
            })
            .collect();
        ids.sort();
        assert_eq!(
            ids,
            vec![100, 300],
            "activation replay must index under key-derived row IDs, not dense RowMap IDs"
        );
        // The inactive doc must still resolve correctly too.
        let inactive = store.query("users", "active = false", Pagination::default()).await.unwrap();
        assert_eq!(inactive.total, 1);
        assert!(matches!(inactive.results[0].0, DocId::U64(200)));
    }
}

/// Random uuid keys must not scatter the field-index rows: the store takes
/// dense row IDs from the row map, so a two-value field over 2,000
/// documents fits in one container per value. Deriving the row ID from
/// the uuid gave every document a container of its own in every bitmap.
/// Queries still resolve the uuid keys, across a restart.
#[tokio::test]
async fn uuid_keyed_store_packs_its_index_rows_densely() {
    let db_dir = TempDir::new().unwrap();
    let mut active_ids = Vec::new();
    {
        let store = open_fresh(db_dir.path()).await;
        let mut schema = make_schema(
            "events",
            vec![IndexSpec {
                field: "active".to_owned(),
                index_type: IndexType::Bool,
            }],
        );
        schema.key_type = KeyType::Uuid;
        store.create(schema).await.unwrap();
        for i in 0..2_000u128 {
            // Spread over the whole u128 range, like v4 uuids.
            let id = i.wrapping_mul(0x9E37_79B9_7F4A_7C15_F39C_C060_5CED_C835);
            let active = i % 3 == 0;
            if active {
                active_ids.push(id);
            }
            store.put("events", DocId::Uuid(id), serde_json::json!({"active": active})).await.unwrap();
        }
        store.db.checkpoint_index().await.unwrap();
        let stats = store.field_index_blob_stats("events", "active").unwrap();
        assert!(
            // Dense: two array containers (2 bytes per row) and two
            // directories, ~4 KB. One container per row would be over 80 KB.
            stats.bitmap_live_bytes < 8_192,
            "two values over 2,000 rows should be two small containers, got {} bytes live",
            stats.bitmap_live_bytes
        );
    }
    let store = open_fresh(db_dir.path()).await;
    let page = store
        .query(
            "events",
            "active = true",
            Pagination {
                page_no: 1,
                page_size: 1_000,
            },
        )
        .await
        .unwrap();
    let mut got: Vec<u128> = page
        .results
        .iter()
        .map(|(id, _)| match id {
            DocId::Uuid(u) => *u,
            other => panic!("expected a uuid id, got {other:?}"),
        })
        .collect();
    got.sort();
    active_ids.sort();
    assert_eq!(page.total, active_ids.len());
    assert_eq!(got, active_ids);
}
