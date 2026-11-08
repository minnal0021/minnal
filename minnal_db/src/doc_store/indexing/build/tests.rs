//! Tests of adding, building and dropping a field index.

use super::*;
use crate::doc_store::tests::test_support::*;

// ── Drop / add index ────────────────────────────────────────────────────

/// `add_index` must enforce the per-namespace `MAX_INDICES` cap incrementally,
/// not just at create/import time (saving a schema does not validate).
#[tokio::test]
async fn test_add_index_enforces_max_indices() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;

    // Start at the cap with MAX_INDICES indices.
    let indices = (0..crate::doc_store::schema::MAX_INDICES)
        .map(|i| IndexSpec {
            field: format!("f{i}"),
            index_type: IndexType::Int,
        })
        .collect();
    store.create(make_schema("ns", indices)).await.unwrap();

    // One more must be rejected (not silently accepted past the cap).
    let result = store
        .add_index(
            "ns",
            IndexSpec {
                field: "one_too_many".to_owned(),
                index_type: IndexType::Int,
            },
        )
        .await;
    match result {
        Err(DocStoreError::Schema(crate::doc_store::error::SchemaError::TooManyIndices { max, .. })) => {
            assert_eq!(max, crate::doc_store::schema::MAX_INDICES);
        }
        Ok(_) => panic!("expected TooManyIndices, got Ok"),
        Err(e) => panic!("expected TooManyIndices, got {e:?}"),
    }

    // And the rejected field must not have been registered/persisted.
    let loaded = store.load_schema("ns").unwrap();
    assert_eq!(loaded.indices.len(), crate::doc_store::schema::MAX_INDICES);
    assert!(!loaded.indices.iter().any(|s| s.field == "one_too_many"));
}

#[tokio::test]
async fn test_drop_index_demotes_to_attribute() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;
    store
        .create(make_schema(
            "ns",
            vec![IndexSpec {
                field: "status".to_owned(),
                index_type: IndexType::Str,
            }],
        ))
        .await
        .unwrap();

    store.drop_index("ns", "status").unwrap();

    let loaded = store.load_schema("ns").unwrap();
    // Index is gone
    assert!(loaded.indices.is_empty(), "index should be removed");
    // Field is preserved as a non-indexed attribute
    assert_eq!(loaded.attributes.len(), 1);
    assert_eq!(loaded.attributes[0].name, "status");
    assert_eq!(loaded.attributes[0].attr_type, AttributeType::Str);
}

/// After `drop_index` the in-memory bitmap must be gone: predicate queries
/// on the dropped field must return an error in the same process, not stale
/// results from the previously-populated bitmap.
#[tokio::test]
async fn test_drop_index_deactivates_in_memory_index() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;
    store
        .create(make_schema(
            "ns",
            vec![IndexSpec {
                field: "status".to_owned(),
                index_type: IndexType::Str,
            }],
        ))
        .await
        .unwrap();

    store.put("ns", DocId::U64(1), serde_json::json!({"status": "active"})).await.unwrap();
    store.put("ns", DocId::U64(2), serde_json::json!({"status": "inactive"})).await.unwrap();

    // Sanity: query works before drop.
    let results = store.query("ns", "status = \"active\"", Pagination::default()).await.unwrap();
    assert_eq!(results.results.len(), 1, "should find one active doc before drop");

    store.drop_index("ns", "status").unwrap();

    // After drop, the same query must fail — not silently return stale hits.
    let err = store.query("ns", "status = \"active\"", Pagination::default()).await.unwrap_err();
    assert!(
        err.to_string().contains("unknown field 'status'"),
        "expected unknown field 'status', got: {err}",
    );
}

/// `add_index` saves the schema without running `validate`, so it must check
/// the new field's name itself.
#[tokio::test]
async fn test_add_index_rejects_a_field_name_no_query_can_reference() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;
    store.create(make_schema("ns", vec![])).await.unwrap();

    for bad in ["address.city", "created-at", "in"] {
        let result = store
            .add_index(
                "ns",
                IndexSpec {
                    field: bad.to_owned(),
                    index_type: IndexType::Str,
                },
            )
            .await;
        assert!(
            matches!(result, Err(DocStoreError::Schema(SchemaError::InvalidFieldName { .. }))),
            "'{bad}' must be rejected"
        );
    }
    assert!(store.load_schema("ns").unwrap().indices.is_empty(), "nothing was added");
}

#[tokio::test]
async fn test_add_index_and_wait() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;
    store.create(make_schema("ns", vec![])).await.unwrap();

    // Insert some documents before the index exists
    for i in 0u64..5 {
        store
            .put("ns", DocId::U64(i), serde_json::json!({"status": "active", "n": i}))
            .await
            .unwrap();
    }

    // Add an index — background task builds it on existing data
    let handle = store
        .add_index(
            "ns",
            IndexSpec {
                field: "status".to_owned(),
                index_type: IndexType::Str,
            },
        )
        .await
        .unwrap();

    handle.wait().await.unwrap();

    // Schema must now include the index
    let loaded = store.load_schema("ns").unwrap();
    assert_eq!(loaded.indices.len(), 1);
    assert_eq!(loaded.indices[0].field, "status");

    // Query must return all 5 docs
    let result = store.query("ns", "status = \"active\"", Pagination::default()).await.unwrap();
    assert_eq!(result.total, 5);
}

/// The rebuild walks the namespace one cursor page at a time; index more
/// documents than a single page so the page-boundary advancement (in both
/// the count pass and the rebuild pass) is exercised, with a partial final
/// page.
/// Dropping a store while one of its field indexes builds must stop the
/// build at its next document and leave nothing behind. The build holds a
/// write guard, so the drop waits for it: without the build's checks of
/// the guard, the drop waited out the whole build. (The guard is what keeps
/// a build that starts or registers its field after the removal from
/// recreating the namespace or its index directory.)
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_drop_stops_a_running_index_build() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;
    store.create(make_schema("ns", vec![])).await.unwrap();
    for i in 0..20_000u64 {
        store.put_no_wal("ns", DocId::U64(i), serde_json::json!({"status": "a"})).await.unwrap();
    }
    let ns_id = store.load_schema("ns").unwrap().ns_id.unwrap();
    let handle = store
        .add_index(
            "ns",
            IndexSpec {
                field: "status".to_owned(),
                index_type: IndexType::Str,
            },
        )
        .await
        .unwrap();
    while handle.mem.indexed.load(std::sync::atomic::Ordering::Relaxed) < 100 {
        tokio::task::yield_now().await;
    }
    store.remove("ns").await.unwrap();
    assert!(handle.wait().await.is_err(), "the build ran on after the drop");
    let names: Vec<String> = store.db.list_namespaces().into_iter().map(|(n, _)| n).collect();
    assert!(!names.iter().any(|n| n == "ns"), "the dropped store's namespace is back");
    let index_dir = crate::layout::namespace_index_dir(&crate::layout::index_root(db_dir.path()), ns_id);
    assert!(!index_dir.exists(), "the dropped store's index directory is back");
    store.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_add_index_rebuild_spans_multiple_pages() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;
    store.create(make_schema("ns", vec![])).await.unwrap();

    let n = REBUILD_PAGE_SIZE + 7; // > 1 page, partial last page
    for i in 0..n {
        let status = if i % 2 == 0 { "active" } else { "inactive" };
        store
            .put("ns", DocId::U64(i as u64), serde_json::json!({"status": status, "n": i}))
            .await
            .unwrap();
    }

    let handle = store
        .add_index(
            "ns",
            IndexSpec {
                field: "status".to_owned(),
                index_type: IndexType::Str,
            },
        )
        .await
        .unwrap();
    handle.wait().await.unwrap();

    // Every document across the page boundary must be indexed, partitioned
    // correctly — no key skipped or double-counted at a page edge.
    let active = store.query("ns", "status = \"active\"", Pagination::default()).await.unwrap();
    let inactive = store.query("ns", "status = \"inactive\"", Pagination::default()).await.unwrap();
    assert_eq!(active.total + inactive.total, n, "all docs indexed");
    assert_eq!(active.total, n.div_ceil(2), "even ids are active");
    assert_eq!(inactive.total, n / 2, "odd ids are inactive");
}

/// A new index must cover every existing document, not just the ones whose
/// WAL entries are still around for activation to replay. The rebuild used to
/// re-put each document's bytes, which the index ignores when nothing changed
/// (`DynFieldIndex::update` skips old == new), so on a store whose WAL had
/// been reclaimed it indexed 28 of these 300 documents and reported success.
#[tokio::test]
async fn test_add_index_covers_documents_whose_wal_was_reclaimed() {
    let db_dir = TempDir::new().unwrap();
    // The smallest WAL segments a configuration allows, and documents
    // large enough to fill several, so the documents' segments are
    // reclaimable.
    let config = || {
        let mut c = crate::support::test_db_config();
        c.wal_segment_size = crate::db::wal::MIN_CONFIGURED_SEGMENT_SIZE;
        c
    };
    let n = 300u64;
    {
        let store = DocStore::open_with_config(db_dir.path(), config()).await.unwrap();
        store.create(make_schema("ns", vec![])).await.unwrap();
        for i in 0..n {
            store
                .put("ns", DocId::U64(i), serde_json::json!({"status": "active", "pad": "x".repeat(800)}))
                .await
                .unwrap();
        }
        store.db.shutdown().await.unwrap(); // flush, so the WAL is persisted
    }
    let store = DocStore::open_with_config(db_dir.path(), config()).await.unwrap();
    // The WAL worker's first tick comes right after open, so it can reclaim
    // these segments first, or be part-way through (this call then reclaims
    // nothing). So check that they are gone, not who removed them: only the
    // active segment is left, and it is not the first.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        store.db.garbage_collect_wal().await.unwrap();
        let segments = store.db.wal_stats().segments;
        if matches!(segments.as_slice(), [only] if only.segment_id > 0) {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "setup: the documents' WAL must be reclaimed: {segments:?}"
        );
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }

    let handle = store
        .add_index(
            "ns",
            IndexSpec {
                field: "status".to_owned(),
                index_type: IndexType::Str,
            },
        )
        .await
        .unwrap();
    handle.wait().await.unwrap();

    let found = store.query("ns", "status = \"active\"", Pagination::default()).await.unwrap();
    assert_eq!(found.total, n as usize, "every existing document must be indexed");
}
