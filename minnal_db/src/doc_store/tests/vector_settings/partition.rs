//! A namespace's own partition: postings split and merge by themselves, a
//! rebuild runs in the background, and a new or dropped namespace starts from
//! one root posting.

use super::support::*;
use super::*;

/// A namespace grows from its root by itself: the worker wakes maintenance
/// when a posting passes its split limit, and the splits keep every key in
/// an active posting with exact counts, also across a reopen.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn postings_split_by_themselves_as_a_namespace_grows() {
    let (url, _seen) = spawn_service();
    let db_dir = TempDir::new().unwrap();
    let store = open_with(db_dir.path(), |b| b.semantic_search(context(&url))).await;
    store
        .create(semantic_schema(
            "g",
            Some(spec(r#"{"embedding_dim":8,"maintenance":{"target_posting_size":16}}"#)),
        ))
        .await
        .unwrap();
    for i in 0..40u64 {
        store.put("g", DocId::U64(i), serde_json::json!({"text": TEXT})).await.unwrap();
    }
    wait_for_empty_queue(&store).await;
    let ctx = Arc::clone(store.semantic_ctx.as_ref().unwrap());
    let (settings, ns_id) = store.schemas.vector_settings("g").unwrap();
    let partition = ctx.for_namespace(&store.db, "g", ns_id, &settings).await.unwrap().partition;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    while partition.snapshot().posting_infos()[&0].state != crate::semantic_search::PostingState::Retired {
        assert!(std::time::Instant::now() < deadline, "the root was never split");
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    let check = |partition: Arc<crate::semantic_search::PartitionHandle>| {
        let db = Arc::clone(&store.db);
        async move {
            let ivf = partition.snapshot();
            let keys = crate::semantic_search::ivf_split::recount_chunks(&db, "g").await.unwrap();
            for (p, n) in &keys {
                assert_eq!(ivf.posting_infos()[p].state, crate::semantic_search::PostingState::Active, "posting {p}");
                assert_eq!(ivf.posting_chunks(*p), *n, "posting {p}");
            }
            assert_eq!(keys.values().sum::<u64>(), ivf.counts().snapshot().values().map(|c| c.1).sum::<u64>());
            ivf.postings()
        }
    };
    // Let any further splits settle, then check.
    let _ = partition.lock_maintenance().await;
    let postings = check(Arc::clone(&partition)).await;
    assert!(postings > 1, "{postings} posting(s)");
    store.shutdown_vec_index_worker().await;
    store.shutdown().await.unwrap();
    drop((store, ctx, partition));

    let store = open_with(db_dir.path(), |b| b.semantic_search(context(&url))).await;
    let ctx = Arc::clone(store.semantic_ctx.as_ref().unwrap());
    let partition = ctx.for_namespace(&store.db, "g", ns_id, &settings).await.unwrap().partition;
    let ivf = partition.snapshot();
    let keys = crate::semantic_search::ivf_split::recount_chunks(&store.db, "g").await.unwrap();
    for (p, n) in &keys {
        assert_eq!(ivf.posting_chunks(*p), *n, "posting {p} after reopen");
    }
    assert_eq!(ivf.postings(), postings);
    let all = store
        .search_semantic("g", "hello", &SearchSpec::default(), Pagination::new(1, 100))
        .await
        .unwrap();
    assert_eq!(all.results.len(), 40, "every document is still found");
    store.shutdown_vec_index_worker().await;
    store.shutdown().await.unwrap();
}

/// Deleting most of a grown namespace's documents merges its emptied
/// postings by themselves: the deletes wake maintenance, and the merges keep
/// every key in an active posting with exact counts.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn postings_merge_by_themselves_as_documents_are_deleted() {
    let (url, _seen) = spawn_service();
    let db_dir = TempDir::new().unwrap();
    let store = open_with(db_dir.path(), |b| b.semantic_search(context(&url))).await;
    store
        .create(semantic_schema(
            "g",
            Some(spec(r#"{"embedding_dim":8,"maintenance":{"target_posting_size":16,"merge_ratio":0.3}}"#)),
        ))
        .await
        .unwrap();
    for i in 0..60u64 {
        store
            .put("g", DocId::U64(i), serde_json::json!({"text": format!("{TEXT} Number {i}.")}))
            .await
            .unwrap();
    }
    wait_for_empty_queue(&store).await;
    let ctx = Arc::clone(store.semantic_ctx.as_ref().unwrap());
    let (settings, ns_id) = store.schemas.vector_settings("g").unwrap();
    let partition = ctx.for_namespace(&store.db, "g", ns_id, &settings).await.unwrap().partition;
    let settle = |partition: Arc<crate::semantic_search::PartitionHandle>| async move {
        // Maintenance is woken asynchronously; wait until it has nothing
        // left: the partition unchanged over ten idle polls in a row.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        let (mut last, mut quiet) = (None, 0);
        while quiet < 10 {
            assert!(std::time::Instant::now() < deadline, "maintenance never settled");
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            let _idle = partition.lock_maintenance().await;
            let c = partition.counters().read();
            let now = Some((partition.snapshot().postings(), c.splits, c.merges));
            quiet = if now == last { quiet + 1 } else { 0 };
            last = now;
        }
    };
    settle(Arc::clone(&partition)).await;
    let grown = partition.snapshot().postings();
    assert!(grown > 2, "{grown} posting(s)");
    for i in 0..54u64 {
        store.delete("g", DocId::U64(i)).await.unwrap();
    }
    wait_for_empty_queue(&store).await;
    settle(Arc::clone(&partition)).await;
    let ivf = partition.snapshot();
    assert!(partition.counters().read().merges > 0, "{:?}", partition.counters().read());
    assert!(ivf.postings() < grown, "{} posting(s), grown to {grown}", ivf.postings());
    let keys = crate::semantic_search::ivf_split::recount_chunks(&store.db, "g").await.unwrap();
    for (p, n) in &keys {
        assert_eq!(ivf.posting_infos()[p].state, crate::semantic_search::PostingState::Active, "posting {p}");
        assert_eq!(ivf.posting_chunks(*p), *n, "posting {p}");
    }
    assert_eq!(keys.values().sum::<u64>(), ivf.counts().snapshot().values().map(|c| c.1).sum::<u64>());
    let all = store
        .search_semantic("g", "hello", &SearchSpec::default(), Pagination::new(1, 100))
        .await
        .unwrap();
    assert_eq!(all.results.len(), 6, "every remaining document is found");
    store.shutdown_vec_index_worker().await;
    store.shutdown().await.unwrap();
}

/// A rebuild started through the store re-clusters a grown namespace from
/// its codes in the background: no embedding request, every key in an
/// active posting with exact counts, every document still found; a second
/// rebuild is refused while one runs.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_rebuild_runs_in_the_background_without_the_embedding_service() {
    use crate::semantic_search::ivf_split::rebuild::RebuildState;
    let (url, seen) = spawn_service();
    let db_dir = TempDir::new().unwrap();
    let store = open_with(db_dir.path(), |b| b.semantic_search(context(&url))).await;
    store
        .create(semantic_schema(
            "g",
            Some(spec(r#"{"embedding_dim":8,"maintenance":{"target_posting_size":16}}"#)),
        ))
        .await
        .unwrap();
    for i in 0..40u64 {
        store
            .put("g", DocId::U64(i), serde_json::json!({"text": format!("{TEXT} Number {i}.")}))
            .await
            .unwrap();
    }
    wait_for_empty_queue(&store).await;
    let ctx = Arc::clone(store.semantic_ctx.as_ref().unwrap());
    let (settings, ns_id) = store.schemas.vector_settings("g").unwrap();
    let partition = ctx.for_namespace(&store.db, "g", ns_id, &settings).await.unwrap().partition;
    let _ = partition.lock_maintenance().await;
    let calls = seen.lock().len();

    assert_eq!(store.vector_rebuild_status("g"), None);
    store
        .create(semantic_schema("plain", Some(spec(r#"{"embedding_dim":8}"#))))
        .await
        .unwrap();
    store.drop_vector_index("plain").await.unwrap();
    assert!(matches!(
        store.start_vector_rebuild("plain", None),
        Err(DocStoreError::SemanticSearchNotEnabled { .. })
    ));
    assert!(matches!(store.start_vector_rebuild("missing", None), Err(DocStoreError::NotFound { .. })));
    let busy = partition.lock_maintenance().await; // holds the rebuild back
    assert_eq!(store.start_vector_rebuild("g", Some(2)).unwrap(), "partition_rebuild");
    assert!(matches!(store.start_vector_rebuild("g", None), Err(DocStoreError::JobConflict { .. })));
    assert!(
        matches!(store.remove("g").await, Err(DocStoreError::StoreInMaintenance { .. })),
        "a store with a rebuild running cannot be dropped"
    );
    drop(busy);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let status = loop {
        let s = store.vector_rebuild_status("g").unwrap();
        if s.state != RebuildState::Running {
            break s;
        }
        assert!(std::time::Instant::now() < deadline, "the rebuild never finished");
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    };
    assert_eq!(status.state, RebuildState::Done, "{status:?}");
    let report = status.report.unwrap();
    assert_eq!(report.postings_fitted, 2, "{report:?}");
    assert_eq!(report.chunks_moved, report.chunks);
    assert_eq!(seen.lock().len(), calls, "a rebuild makes no embedding request");

    // A recall check makes none either; it and the rebuild are in the history.
    let params = crate::semantic_search::ivf_split::recall_check::RecallCheckParams {
        sample: 10,
        ..Default::default()
    };
    let check = store.vector_recall_check("g", &params, &SearchSpec::default()).await.unwrap();
    assert_eq!(check.sampled, 10, "{check:?}");
    assert_eq!(seen.lock().len(), calls, "a recall check makes no embedding request");
    use crate::semantic_search::ivf_split::history::RecordKind;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let kinds = loop {
        let kinds: Vec<RecordKind> = store
            .vector_partition_history("g", None, 10)
            .await
            .unwrap()
            .iter()
            .map(|r| r.kind)
            .collect();
        if kinds.contains(&RecordKind::Rebuild) || std::time::Instant::now() > deadline {
            break kinds;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    };
    assert!(
        kinds.contains(&RecordKind::Rebuild) && kinds.contains(&RecordKind::RecallCheck),
        "{kinds:?}"
    );
    let last = store.vector_partition_history("g", Some(RecordKind::RecallCheck), 1).await.unwrap();
    assert_eq!(last[0].data["sampled"], 10);

    let ivf = partition.snapshot();
    let keys = crate::semantic_search::ivf_split::recount_chunks(&store.db, "g").await.unwrap();
    for (p, n) in &keys {
        assert_eq!(ivf.posting_infos()[p].state, crate::semantic_search::PostingState::Active, "posting {p}");
        assert_eq!(ivf.posting_chunks(*p), *n, "posting {p}");
    }
    assert_eq!(keys.values().sum::<u64>(), report.chunks);
    let all = store
        .search_semantic("g", "hello", &SearchSpec::default(), Pagination::new(1, 100))
        .await
        .unwrap();
    assert_eq!(all.results.len(), 40, "every document is still found");

    // The escape hatch still works after a rebuild: `reindex-all` enqueues
    // every document, re-embeds each through the service, and every
    // document is found afterwards.
    let before = seen.lock().len();
    let stats = store.index_all("g").await.unwrap();
    assert_eq!(stats.enqueued, 40, "{stats:?}");
    wait_for_empty_queue(&store).await;
    assert!(seen.lock().len() > before, "reindex-all calls the embedding service");
    let all = store
        .search_semantic("g", "hello", &SearchSpec::default(), Pagination::new(1, 100))
        .await
        .unwrap();
    assert_eq!(all.results.len(), 40, "every document is found after reindex-all");
    store.shutdown_vec_index_worker().await;
    store.shutdown().await.unwrap();
}

// ── M2b: the namespace's own partition ──

#[tokio::test]
async fn a_new_namespace_starts_from_one_root_posting_and_a_drop_resets_it() {
    let (url, _seen) = spawn_service();
    let db_dir = TempDir::new().unwrap();
    let store = open_with(db_dir.path(), |b| b.semantic_search(context(&url))).await;
    store.create(semantic_schema("g", Some(spec(r#"{"embedding_dim":8}"#)))).await.unwrap();
    let vi = store.get_schema("g").unwrap().vector_index.unwrap();
    assert_eq!(vi.quantisation.unwrap().rotation_seed.as_deref(), Some("0x6d696e6e616c0001"));
    let root = vector_kv::load_ivf(&store.db, "g", 1).await.unwrap().expect("a root posting");
    assert_eq!((root.postings(), root.centres()), (1, 1));
    store.put("g", DocId::U64(1), serde_json::json!({"text": TEXT})).await.unwrap();
    wait_for_empty_queue(&store).await;
    assert_eq!(sparse_clusters(&store, "g").await, vec![0], "every chunk is in the root");

    // Dropping the vector index drops the partition with the data, also
    // across a reopen.
    store.drop_vector_index("g").await.unwrap();
    store.shutdown_vec_index_worker().await;
    store.shutdown().await.unwrap();
    drop(store);
    let store = open_with(db_dir.path(), |b| b.semantic_search(context(&url))).await;
    assert!(!vector_kv::ivf_exists(&store.db, "g").await.unwrap(), "gone with the data");

    // Re-enabling over the same field (still an attribute after the drop)
    // starts again from a root.
    store
        .amend(
            "g",
            SchemaAmendment::EnableVectorIndex {
                fields: vec!["text".into()],
                vector_index: None,
            },
        )
        .await
        .unwrap();
    assert!(vector_kv::ivf_exists(&store.db, "g").await.unwrap());
    store.shutdown_vec_index_worker().await;
    store.shutdown().await.unwrap();
}

#[tokio::test]
async fn dropping_a_store_removes_its_partition_and_a_recreated_store_gets_its_own() {
    let (url, _seen) = spawn_service();
    let db_dir = TempDir::new().unwrap();
    let store = open_with(db_dir.path(), |b| b.semantic_search(context(&url))).await;
    store.create(semantic_schema("s", Some(spec(r#"{"embedding_dim":8}"#)))).await.unwrap();
    store.put("s", DocId::U64(1), serde_json::json!({"text": TEXT})).await.unwrap();
    wait_for_empty_queue(&store).await;
    assert_eq!(results(&store, "s").await.len(), 1, "loads gemma's partition into the registry");
    store.remove("s").await.unwrap();
    let names: Vec<String> = store.db.list_namespaces().into_iter().map(|(n, _)| n).collect();
    assert!(!names.iter().any(|n| n.starts_with("s_ivf")), "{names:?}");

    // Same name, other model: its own partition, not the evicted one.
    store
        .create(semantic_schema("s", Some(spec(r#"{"embedding_model":"qwen","embedding_dim":8}"#))))
        .await
        .unwrap();
    store.put("s", DocId::U64(1), serde_json::json!({"text": TEXT})).await.unwrap();
    wait_for_empty_queue(&store).await;
    assert_eq!(sparse_clusters(&store, "s").await, vec![0], "a fresh root");
    assert_eq!(results(&store, "s").await.len(), 1);
    store.shutdown_vec_index_worker().await;
    store.shutdown().await.unwrap();
}
