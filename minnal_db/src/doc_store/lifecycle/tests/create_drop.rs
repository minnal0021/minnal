//! Creating, listing and dropping stores, and names a store cannot take.

use crate::doc_store::tests::test_support::*;
use crate::doc_store::*;
use crate::namespace::NamespaceId;

/// What a create cut short leaves: an empty namespace, which a new store
/// replaces, field-index registrations included.
#[tokio::test]
async fn a_new_store_replaces_an_empty_namespace_without_a_schema() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;
    for name in ["ghost_doc", "ghost_kv"] {
        let orphan = store.db.namespace(name.to_owned()).await.unwrap();
        // Emptied, not just never written: only what a scan finds counts.
        orphan.put(b"k".to_vec(), b"v".to_vec()).await.unwrap();
        orphan.delete(b"k".to_vec()).await.unwrap();
    }
    let before = store.db.existing_namespace("ghost_doc".to_owned()).await.unwrap().unwrap().id();
    store.create(make_schema("ghost_doc", vec![])).await.unwrap();
    let after = store.db.existing_namespace("ghost_doc".to_owned()).await.unwrap().unwrap().id();
    assert_ne!(before, after, "the store must get a namespace of its own");
    store
        .create_kv(make_kv_schema("ghost_kv", KvKeyType::Str, KvValueType::Str))
        .await
        .unwrap();
    store.shutdown().await.unwrap();
}

/// Keys written directly through `Db` are not a store's to delete or adopt.
#[tokio::test]
async fn a_store_cannot_take_the_name_of_a_namespace_holding_keys() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;
    let raw = store.db.namespace("users".to_owned()).await.unwrap();
    raw.put(1u64.to_be_bytes().to_vec(), b"{}".to_vec()).await.unwrap();

    let err = store.create(make_schema("users", vec![])).await.unwrap_err();
    assert!(matches!(err, DocStoreError::NamespaceNotEmpty { .. }), "{err:?}");
    let err = store
        .create_kv(make_kv_schema("users", KvKeyType::Str, KvValueType::Str))
        .await
        .unwrap_err();
    assert!(matches!(err, DocStoreError::NamespaceNotEmpty { .. }), "{err:?}");

    assert!(store.list().unwrap().is_empty());
    assert_eq!(raw.get(1u64.to_be_bytes().to_vec()).await.unwrap(), Some(b"{}".to_vec()), "the keys stay");
    store.shutdown().await.unwrap();
}

#[tokio::test]
async fn a_store_name_cannot_end_like_a_vector_companion() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;
    for suffix in crate::namespace::VECTOR_COMPANION_SUFFIXES {
        let name = format!("foo{suffix}");
        let err = store.create(make_schema(&name, vec![])).await.unwrap_err();
        assert!(
            matches!(err, DocStoreError::Schema(SchemaError::ReservedNamespace { .. })),
            "{name}: {err:?}"
        );
        let err = store
            .create_kv(make_kv_schema(&name, KvKeyType::Str, KvValueType::Str))
            .await
            .unwrap_err();
        assert!(
            matches!(err, DocStoreError::Schema(SchemaError::ReservedNamespace { .. })),
            "{name}: {err:?}"
        );
    }
    assert!(store.list().unwrap().is_empty());
    store.shutdown().await.unwrap();
}

/// A database can hold a store saved before such names were refused. It is
/// a store, not a companion: neither a restart nor dropping `foo` removes it.
#[cfg(feature = "semantic-search")]
#[tokio::test]
async fn a_store_saved_under_a_companion_name_survives_a_restart_and_a_drop() {
    const NAME: &str = "foo_dense_vector";
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;
    store.create(make_schema("foo", vec![])).await.unwrap();
    let mut schema = make_schema(NAME, vec![]);
    schema.ns_id = Some(store.db.namespace(NAME.to_owned()).await.unwrap().id());
    store
        .schemas
        .insert(StoredSchema::Doc(Arc::new(schema)), NamespaceState::Created)
        .unwrap();
    store.put(NAME, DocId::U64(1), serde_json::json!({})).await.unwrap();
    store.shutdown().await.unwrap();
    drop(store);

    let store = open_fresh(db_dir.path()).await;
    assert!(store.get(NAME, DocId::U64(1)).await.unwrap().is_some(), "lost by the restart");
    store.remove("foo").await.unwrap();
    assert!(store.get(NAME, DocId::U64(1)).await.unwrap().is_some(), "lost by dropping foo");
    store.shutdown().await.unwrap();
}

/// A drop's future can be dropped at any await (the client of its route
/// disconnecting). Once it has moved the store to `Deleting`, the store stays
/// so, refusing requests, until the governor finishes the drop; a retry waits
/// for that.
#[tokio::test]
async fn a_cancelled_drop_leaves_the_store_deleting_and_a_retry_finishes_it() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;
    store.create(make_schema("ns", vec![])).await.unwrap();
    let held = store.lifecycle.lock("ns").await;
    let cancelled = tokio::time::timeout(std::time::Duration::from_millis(50), store.remove("ns")).await;
    assert!(cancelled.is_err(), "the drop waits for the lifecycle lock");
    drop(held);
    assert_eq!(store.schemas.state("ns"), Some(NamespaceState::Deleting));
    assert!(matches!(
        store.put("ns", DocId::U64(1), serde_json::json!({})).await,
        Err(DocStoreError::StoreDeleting { .. })
    ));
    store.remove("ns").await.unwrap();
    assert!(!store.schemas.contains("ns"));
    store.shutdown().await.unwrap();
}

// ── create / list / drop ────────────────────────────────────────────────

#[tokio::test]
async fn test_create_and_list() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;

    let schema = make_schema(
        "users",
        vec![IndexSpec {
            field: "active".to_owned(),
            index_type: IndexType::Bool,
        }],
    );
    store.create(schema).await.unwrap();

    let list = store.list().unwrap();
    assert_eq!(list.len(), 1);
    assert_eq!(list[0]["namespace"], "users");
    assert!(serde_json::from_value::<NamespaceId>(list[0]["ns_id"].clone()).is_ok());
}

#[tokio::test]
async fn test_create_duplicate_is_rejected() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;

    store.create(make_schema("dup", vec![])).await.unwrap();
    let err = store.create(make_schema("dup", vec![])).await.unwrap_err();
    assert!(matches!(err, DocStoreError::AlreadyExists { .. }));
}

#[tokio::test]
async fn system_names_are_reserved_for_both_kinds_of_store() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;
    for name in ["system", "system_schema", "system_mine"] {
        assert!(
            matches!(
                store.create(make_schema(name, vec![])).await,
                Err(DocStoreError::Schema(SchemaError::ReservedNamespace { .. }))
            ),
            "{name}"
        );
        assert!(
            matches!(
                store.create_kv(make_kv_schema(name, KvKeyType::Str, KvValueType::Str)).await,
                Err(DocStoreError::Schema(SchemaError::ReservedNamespace { .. }))
            ),
            "{name}"
        );
    }
    // Only the prefix is reserved.
    store.create(make_schema("systems", vec![])).await.unwrap();
    store.create(make_schema("my_system_x", vec![])).await.unwrap();
}

/// Concurrent amendments to one store each read, change and save the schema;
/// none may overwrite another's change.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_amendments_to_one_store_all_land() {
    let db_dir = TempDir::new().unwrap();
    let store = Arc::new(open_fresh(db_dir.path()).await);
    store.create(make_schema("ns", vec![])).await.unwrap();
    let mut tasks = tokio::task::JoinSet::new();
    for i in 0..32 {
        let store = Arc::clone(&store);
        tasks.spawn(async move {
            store
                .amend(
                    "ns",
                    SchemaAmendment::AddAttribute {
                        name: format!("a{i}"),
                        attr_type: AttributeType::Str,
                        description: None,
                    },
                )
                .await
        });
    }
    while let Some(r) = tasks.join_next().await {
        r.unwrap().unwrap();
    }
    assert_eq!(store.load_schema("ns").unwrap().attributes.len(), 32);
    let fresh = SchemaStore::open(&store.db).await.unwrap();
    assert_eq!(fresh.doc("ns").unwrap().attributes.len(), 32, "every amendment persisted");
}

/// After every kind of schema change, the cache holds exactly what a fresh
/// read of `system_schema` finds, and so does a reopened store.
#[tokio::test]
async fn the_cache_matches_storage_after_every_schema_change() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;
    let snapshot = |schemas: &SchemaStore| -> Vec<serde_json::Value> {
        schemas
            .all()
            .into_iter()
            .map(|s| match s {
                StoredSchema::Doc(s) => serde_json::to_value(&*s).unwrap(),
                StoredSchema::Kv(s) => serde_json::to_value(&*s).unwrap(),
            })
            .collect()
    };
    let check = |store: &DocStore, step: &str| {
        let schemas = Arc::clone(&store.schemas);
        let db = Arc::clone(&store.db);
        let step = step.to_owned();
        async move {
            let fresh = SchemaStore::open(&db).await.unwrap();
            assert_eq!(snapshot(&schemas), snapshot(&fresh), "after {step}");
        }
    };
    store.create(make_schema("docs", vec![])).await.unwrap();
    store.create_kv(make_kv_schema("kv", KvKeyType::Str, KvValueType::Str)).await.unwrap();
    check(&store, "create").await;
    store
        .amend(
            "docs",
            SchemaAmendment::AddAttribute {
                name: "status".to_owned(),
                attr_type: AttributeType::Str,
                description: None,
            },
        )
        .await
        .unwrap();
    check(&store, "amend").await;
    let spec = IndexSpec {
        field: "level".to_owned(),
        index_type: IndexType::Int,
    };
    store.add_index("docs", spec).await.unwrap();
    check(&store, "add_index").await;
    store.drop_index("docs", "level").await.unwrap();
    check(&store, "drop_index").await;
    store.remove_attribute("docs", "status").await.unwrap();
    check(&store, "remove_attribute").await;
    store.remove_kv("kv").await.unwrap();
    check(&store, "remove_kv").await;

    let before = snapshot(&store.schemas);
    store.shutdown().await.unwrap();
    drop(store);
    let reopened = open_fresh(db_dir.path()).await;
    assert_eq!(snapshot(&reopened.schemas), before, "after reopen");
    assert_eq!(
        reopened
            .load_schema("docs")
            .unwrap()
            .attributes
            .iter()
            .map(|a| a.name.as_str())
            .collect::<Vec<_>>(),
        ["level"]
    );
}

#[tokio::test]
async fn test_drop_store_removes_schema() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;

    store.create(make_schema("tmp", vec![])).await.unwrap();
    assert!(store.schemas.contains("tmp"));

    store.remove("tmp").await.unwrap();
    assert!(!store.schemas.contains("tmp"));
}

/// A store drop interrupted after it removed the store's namespace (the
/// schema goes last) must be finishable: by dropping the store again, and
/// at the next open. Dropping again failed, since the namespace it removes
/// first was gone; and the next open recreated a KV store's namespace (under
/// a new id its schema does not name) and failed on a document store's
/// field indexes.
#[tokio::test]
async fn an_interrupted_drop_is_finished_by_a_retry_or_the_next_open() {
    let index = || IndexSpec {
        field: "active".to_owned(),
        index_type: IndexType::Bool,
    };
    for retry in [true, false] {
        let db_dir = TempDir::new().unwrap();
        {
            let store = open_fresh(db_dir.path()).await;
            store.create(make_schema("docs", vec![index()])).await.unwrap();
            store.put("docs", DocId::U64(1), serde_json::json!({"active": true})).await.unwrap();
            store.create_kv(make_kv_schema("kv", KvKeyType::Str, KvValueType::Str)).await.unwrap();
            store.kv_put("kv", &serde_json::json!("k"), &serde_json::json!("v")).await.unwrap();
            // Interrupted part-way: the stores are `Deleting` and their
            // namespaces gone, their records not.
            for name in ["docs", "kv"] {
                store
                    .schemas
                    .transition(name, &[NamespaceState::Created], NamespaceState::Deleting)
                    .unwrap();
                store.db.remove_namespace(name.to_owned()).await.unwrap();
            }
            assert!(matches!(store.get("docs", DocId::U64(1)).await, Err(DocStoreError::StoreDeleting { .. })));
            if retry {
                store.remove("docs").await.unwrap();
                store.remove_kv("kv").await.unwrap();
            }
            store.shutdown().await.unwrap();
        }
        let store = open_fresh(db_dir.path()).await;
        assert!(
            !store.schemas.contains("docs") && !store.schemas.contains("kv"),
            "retry={retry}: a schema survived"
        );
        let names: Vec<String> = store.db.list_namespaces().into_iter().map(|(n, _)| n).collect();
        assert!(
            !names.iter().any(|n| n == "docs" || n == "kv"),
            "retry={retry}: a namespace is back: {names:?}"
        );
        // The names are free again.
        store.create(make_schema("docs", vec![index()])).await.unwrap();
        store.put("docs", DocId::U64(2), serde_json::json!({"active": true})).await.unwrap();
        assert_eq!(store.query("docs", "active = true", Pagination::default()).await.unwrap().total, 1);
        store.shutdown().await.unwrap();
    }
}

/// Dropping a semantic store must clear its embed-queue entries. Left behind,
/// they outlive the store, and the vector worker processing them can bring its
/// vector namespaces back under new ids.
#[cfg(feature = "semantic-search")]
#[tokio::test]
async fn test_drop_store_clears_its_embed_queue() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;

    let mut schema = make_schema("sem_docs", vec![]);
    schema.attributes = vec![crate::doc_store::schema::AttributeDef {
        name: "title".to_owned(),
        attr_type: AttributeType::Str,
        description: None,
    }];
    schema.semantic_search_enabled = true;
    schema.embedding_fields = vec!["title".to_owned()];
    store.create(schema).await.unwrap();

    // Queue work for it, exactly as a write would. No embedding service is
    // involved — enqueuing is a plain durable write.
    for i in 0..5u32 {
        vector_kv::enqueue_embed(&store.db, "sem_docs", format!("doc{i}").as_bytes(), "the quiet harbour")
            .await
            .unwrap();
    }
    let queued = vector_kv::list_queue_entries(&store.db).await.unwrap();
    assert_eq!(
        queued.iter().filter(|e| e.namespace == "sem_docs").count(),
        5,
        "the queue should hold the enqueued work before the drop"
    );

    store.remove("sem_docs").await.unwrap();

    let after = vector_kv::list_queue_entries(&store.db).await.unwrap();
    let stranded: Vec<_> = after.iter().filter(|e| e.namespace == "sem_docs").collect();
    assert!(
        stranded.is_empty(),
        "{} queue entry/entries outlived the store they belong to — the worker will \
         embed them and recreate its vector sidecar namespaces",
        stranded.len()
    );
}
