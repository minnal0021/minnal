//! Searching: probe traces, entry counts, two models in one process,
//! per-request overrides, and models the service lacks.

use super::support::*;
use super::*;
use crate::semantic_search::chunk_document;
use crate::semantic_search::service::MAX_TRACE_TOP_K;

/// A probe trace covers the window it is asked for, at absolute ranks: by
/// default the namespace's `max_probes` nearest postings (all of a small
/// namespace), any window being a slice of that order. Its baseline is the
/// namespace's own search, and a window that is empty, too wide or past the
/// last posting is refused.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_probe_trace_covers_the_window_asked_for() {
    let (url, _seen) = spawn_text_service();
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
        // Different chunks per document: identical chunks cannot be split
        // past one posting per distinct chunk.
        let text = format!("Doc {i} one. Doc {i} two. Doc {i} three. Doc {i} four. Doc {i} five. Doc {i} six.");
        store.put("g", DocId::U64(i), serde_json::json!({"text": text})).await.unwrap();
    }
    wait_for_empty_queue(&store).await;
    let partition = partition(&store, "g").await;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    while partition.snapshot().postings() <= 2 {
        assert!(std::time::Instant::now() < deadline, "the split children were never split");
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    let _quiet = partition.lock_maintenance().await;
    let active = partition.snapshot().postings();
    assert!(active > 2, "{active} postings");
    // Fewer results than documents, so some reported documents are not returned.
    let top5 = SearchSpec {
        top_k: Some(5),
        ..SearchSpec::default()
    };
    store.update_vector_search("g", &top5, None).await.unwrap();

    let full = store.vector_probe_trace("g", "hello", None, None, None).await.unwrap();
    assert_eq!((full.start, full.end, full.active_postings), (0, active, active));
    assert_eq!(full.postings.iter().map(|p| p.rank).collect::<Vec<_>>(), (0..active).collect::<Vec<_>>());
    // Each document has several chunks, which can lie in different postings.
    assert!(full.postings.iter().map(|p| p.documents).sum::<usize>() >= 40);
    let search = store
        .search_semantic("g", "hello", &SearchSpec::default(), Pagination::new(1, 100))
        .await
        .unwrap();
    let b = &full.baseline;
    assert_eq!((b.n_probes, b.top_k, b.first_pass_top_k, b.results), (active, 5, 1000, 5));
    assert_eq!(search.results.len(), 5);
    assert_eq!(b.lowest_pass2, search.results.iter().map(|r| r.dot_product).min_by(f32::total_cmp));
    // Every reported document says whether that search returns it.
    let returned: Vec<&[u8]> = search.results.iter().map(|r| r.document_id.as_slice()).collect();
    for s in full.postings.iter().flat_map(|p| [&p.min, &p.max]).flatten() {
        assert_eq!(s.in_results, Some(returned.contains(&s.document_id.as_slice())));
    }
    let flags: Vec<bool> = full
        .postings
        .iter()
        .flat_map(|p| [&p.min, &p.max])
        .flatten()
        .filter_map(|s| s.in_results)
        .collect();
    assert!(flags.contains(&false), "{flags:?}");

    let tail = store.vector_probe_trace("g", "hello", Some(1), None, None).await.unwrap();
    assert_eq!((tail.start, tail.end), (1, active), "end defaults to the last posting");
    assert_eq!(tail.postings, full.postings[1..]);
    let one = store.vector_probe_trace("g", "hello", Some(1), Some(2), None).await.unwrap();
    assert_eq!(one.postings, full.postings[1..2]);

    // A deeper baseline returns more documents, and lowers the bar to be one.
    let deep = store.vector_probe_trace("g", "hello", None, None, Some(30)).await.unwrap();
    let without_flags = |postings: &[crate::semantic_search::service::ProbedPosting]| {
        let mut postings = postings.to_vec();
        for s in postings.iter_mut().flat_map(|p| [&mut p.min, &mut p.max]).flatten() {
            s.in_results = None;
        }
        postings
    };
    assert_eq!(
        without_flags(&deep.postings),
        without_flags(&full.postings),
        "top_k changes only the baseline"
    );
    let (d, deep_search) = (&deep.baseline, {
        let top30 = SearchSpec {
            top_k: Some(30),
            ..SearchSpec::default()
        };
        store.search_semantic("g", "hello", &top30, Pagination::new(1, 100)).await.unwrap()
    });
    assert_eq!((d.top_k, d.first_pass_top_k, d.results), (30, 1000, deep_search.results.len()));
    assert!(d.results > b.results);
    assert_eq!(d.lowest_pass2, deep_search.results.iter().map(|r| r.dot_product).min_by(f32::total_cmp));
    assert!(d.lowest_pass2 <= b.lowest_pass2);
    let deep_returned: Vec<&[u8]> = deep_search.results.iter().map(|r| r.document_id.as_slice()).collect();
    for s in deep.postings.iter().flat_map(|p| [&p.min, &p.max]).flatten() {
        assert_eq!(s.in_results, Some(deep_returned.contains(&s.document_id.as_slice())));
    }
    // Past the namespace's first_pass_top_k, Pass 1 keeps top_k candidates.
    let widest = store.vector_probe_trace("g", "hello", None, None, Some(MAX_TRACE_TOP_K)).await.unwrap();
    assert_eq!((widest.baseline.top_k, widest.baseline.first_pass_top_k), (10_000, 10_000));
    assert_eq!(widest.baseline.results, 40, "every document");
    // The namespace's own settings are untouched.
    let after = store
        .search_semantic("g", "hello", &SearchSpec::default(), Pagination::new(1, 100))
        .await
        .unwrap();
    assert_eq!(after.results.len(), 5);

    let refused_k = |k: u32| {
        let store = &store;
        async move {
            match store.vector_probe_trace("g", "hello", None, None, Some(k)).await {
                Err(DocStoreError::InvalidProbeTrace { reason, .. }) => reason,
                other => panic!("top_k {k} was not refused: {other:?}"),
            }
        }
    };
    assert!(refused_k(0).await.contains("between 1 and 10000"));
    assert!(refused_k(MAX_TRACE_TOP_K + 1).await.contains("between 1 and 10000"));

    let refused = |start: Option<u32>, end: Option<u32>| {
        let store = &store;
        async move {
            match store.vector_probe_trace("g", "hello", start, end, None).await {
                Err(DocStoreError::InvalidProbeTrace { reason, .. }) => reason,
                other => panic!("{start:?}..{end:?} was not refused: {other:?}"),
            }
        }
    };
    assert!(refused(Some(2), Some(2)).await.contains("below end"));
    assert!(refused(None, Some(0)).await.contains("below end"));
    assert!(refused(Some(active as u32), None).await.contains("below end"));
    assert!(refused(Some(0), Some(active as u32 + 1)).await.contains("past"));
    assert!(refused(Some(0), Some(1025)).await.contains("more than 1024"));
    drop(_quiet);
    store.shutdown_vec_index_worker().await;
    store.shutdown().await.unwrap();
}

/// A trace's baseline accepts any `top_k` a namespace's Pass-1 cut accepts.
#[test]
fn the_trace_top_k_cap_is_the_largest_first_pass_cut() {
    assert_eq!(MAX_TRACE_TOP_K, crate::doc_store::schema::vector_settings::MAX_FIRST_PASS_TOP_K);
}

/// The loaded partition's entry counts track the worker's writes and the
/// doc store's deletes, matching a fresh count of the stored keys.
#[tokio::test]
async fn entry_counts_track_indexing_updates_and_deletes() {
    let (url, _seen) = spawn_service();
    let db_dir = TempDir::new().unwrap();
    let store = open_with(db_dir.path(), |b| b.semantic_search(context(&url))).await;
    store
        .create(semantic_schema(
            "g",
            Some(spec(r#"{"embedding_dim":8,"chunking":{"window_size":1,"sliding_size":1}}"#)),
        ))
        .await
        .unwrap();
    let ctx = Arc::clone(store.semantic_ctx.as_ref().unwrap());
    let (settings, ns_id) = store.schemas.vector_settings("g").unwrap();
    let check = || async {
        let ivf = ctx.for_namespace(&store.db, "g", ns_id, &settings).await.unwrap().partition.snapshot();
        let stored = vector_kv::count_posting_entries(&store.db, "g").await.unwrap();
        let loaded: std::collections::HashMap<u32, u64> = ivf
            .counts()
            .snapshot()
            .into_iter()
            .filter(|&(_, (e, _))| e > 0)
            .map(|(p, (e, _))| (p, e))
            .collect();
        assert_eq!(loaded, stored);
        // The chunk counts agree with the document records too.
        let from_meta = vector_kv::count_postings(&store.db, "g").await.unwrap();
        assert_eq!(
            ivf.counts()
                .snapshot()
                .into_iter()
                .filter(|&(_, (e, _))| e > 0)
                .collect::<std::collections::HashMap<_, _>>(),
            from_meta
        );
        ivf.total_entries()
    };

    for i in 0..6u64 {
        store.put("g", DocId::U64(i), serde_json::json!({"text": TEXT})).await.unwrap();
    }
    wait_for_empty_queue(&store).await;
    assert!(check().await > 0, "the documents were indexed");
    // Re-embed with other text, then delete some (doc-store path) and blank one.
    for i in 0..3u64 {
        store
            .put("g", DocId::U64(i), serde_json::json!({"text": format!("Other words {i}. And more.")}))
            .await
            .unwrap();
    }
    wait_for_empty_queue(&store).await;
    check().await;
    store.delete("g", DocId::U64(4)).await.unwrap();
    store.delete("g", DocId::U64(5)).await.unwrap();
    store.put("g", DocId::U64(3), serde_json::json!({"text": ""})).await.unwrap();
    wait_for_empty_queue(&store).await;
    check().await;
    store.shutdown().await.unwrap();
}

#[tokio::test]
async fn two_namespaces_with_different_models_each_use_their_own() {
    let (url, seen) = spawn_service();
    let db_dir = TempDir::new().unwrap();
    let store = open_with(db_dir.path(), |b| b.semantic_search(context(&url))).await;
    store
        .create(semantic_schema(
            "g",
            Some(spec(
                r#"{"embedding_model":"gemma","embedding_dim":8,"chunking":{"window_size":2,"sliding_size":1}}"#,
            )),
        ))
        .await
        .unwrap();
    store
        .create(semantic_schema("q", Some(spec(r#"{"embedding_model":"qwen","embedding_dim":8}"#))))
        .await
        .unwrap();

    store.put("g", DocId::U64(1), serde_json::json!({"text": TEXT})).await.unwrap();
    store.put("q", DocId::U64(1), serde_json::json!({"text": TEXT})).await.unwrap();
    wait_for_empty_queue(&store).await;

    // Each document was embedded by its namespace's model, at its dimension,
    // chunked by its namespace's window (one whole-text payload + chunks).
    let text = format!("text: {TEXT}");
    let docs: Vec<Seen> = seen.lock().iter().filter(|s| s.kind == "document").cloned().collect();
    let expect = |model: &str, w: usize, s: usize| Seen {
        model: model.into(),
        kind: "document".into(),
        dimensions: DIM,
        payloads: 1 + chunk_document(&text, w, s).len(),
    };
    assert!(docs.contains(&expect("gemma", 2, 1)), "{docs:?}");
    assert!(docs.contains(&expect("qwen", 4, 2)), "{docs:?}");
    assert_ne!(
        expect("gemma", 2, 1).payloads,
        expect("qwen", 4, 2).payloads,
        "the test must tell the chunkings apart"
    );

    // Each namespace's chunks sit in its own partition (each still its root).
    assert_eq!(sparse_clusters(&store, "g").await, vec![0]);
    assert_eq!(sparse_clusters(&store, "q").await, vec![0]);

    // Searches embed with the namespace's model and cache per model.
    let none = SearchSpec::default();
    let queries = |seen: &parking_lot::Mutex<Vec<Seen>>| {
        seen.lock()
            .iter()
            .filter(|s| s.kind == "query")
            .map(|s| s.model.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        store
            .search_semantic("g", "hello", &none, Pagination::default())
            .await
            .unwrap()
            .results
            .len(),
        1
    );
    assert_eq!(
        store
            .search_semantic("g", "hello", &none, Pagination::default())
            .await
            .unwrap()
            .results
            .len(),
        1
    );
    assert_eq!(queries(&seen), vec!["gemma"], "the second gemma search is a cache hit");
    assert_eq!(
        store
            .search_semantic("q", "hello", &none, Pagination::default())
            .await
            .unwrap()
            .results
            .len(),
        1
    );
    assert_eq!(queries(&seen), vec!["gemma", "qwen"], "qwen never reuses gemma's cached vector");

    store.shutdown_vec_index_worker().await;
    store.shutdown().await.unwrap();
}

#[tokio::test]
async fn per_request_overrides_are_validated_and_applied() {
    let (url, _seen) = spawn_service();
    let db_dir = TempDir::new().unwrap();
    let store = open_with(db_dir.path(), |b| b.semantic_search(context(&url))).await;
    store.create(semantic_schema("g", Some(spec(r#"{"embedding_dim":8}"#)))).await.unwrap();
    for i in 1..=3u64 {
        store.put("g", DocId::U64(i), serde_json::json!({"text": TEXT})).await.unwrap();
    }
    wait_for_empty_queue(&store).await;

    let page = Pagination::default();
    let all = store.search_semantic("g", "q", &SearchSpec::default(), page).await.unwrap();
    assert_eq!(all.results.len(), 3);
    let one = SearchSpec {
        top_k: Some(1),
        ..Default::default()
    };
    assert_eq!(store.search_semantic("g", "q", &one, page).await.unwrap().results.len(), 1);
    assert!(
        store.kv_search_semantic("g", "q", &one, page).await.is_err(),
        "g is a doc store, not a KV store"
    );

    for (bad, field) in [
        (
            SearchSpec {
                min_probes: Some(0),
                ..Default::default()
            },
            "search.min_probes",
        ),
        (
            SearchSpec {
                top_k: Some(1001),
                ..Default::default()
            },
            "search.top_k",
        ),
        (
            SearchSpec {
                first_pass_top_k: Some(5),
                ..Default::default()
            },
            "search.top_k",
        ),
    ] {
        let err = store.search_semantic("g", "q", &bad, page).await.unwrap_err();
        assert!(
            matches!(&err, DocStoreError::Schema(SchemaError::InvalidVectorSetting { field: f, .. }) if *f == field),
            "{bad:?}: {err:?}"
        );
        let err = store.search_semantic_filtered("g", "q", "text = 'x'", &bad, page).await.unwrap_err();
        assert!(matches!(err, DocStoreError::Schema(SchemaError::InvalidVectorSetting { .. })), "{err:?}");
    }
    store.shutdown_vec_index_worker().await;
    store.shutdown().await.unwrap();
}

#[tokio::test]
async fn a_model_or_dimension_the_server_lacks_is_rejected_on_create_and_enable() {
    let (url, _seen) = spawn_service();
    let db_dir = TempDir::new().unwrap();
    let store = open_with(db_dir.path(), |b| b.semantic_search(context(&url))).await;

    let err = store
        .create(semantic_schema("e5", Some(spec(r#"{"embedding_model":"e5","embedding_dim":8}"#))))
        .await
        .unwrap_err();
    assert!(
        matches!(&err, DocStoreError::UnsupportedEmbeddingModel { model, supported } if model == "e5" && supported == "gemma, qwen"),
        "{err:?}"
    );
    let err = store
        .create(semantic_schema("wide", Some(spec(r#"{"embedding_dim":16}"#))))
        .await
        .unwrap_err();
    assert!(matches!(err, DocStoreError::EmbeddingDimMismatch { dim: 16, expected: 8, .. }), "{err:?}");
    assert!(!store.schemas.contains("e5") && !store.schemas.contains("wide"));

    store.create(make_schema("later", vec![])).await.unwrap();
    let err = store
        .amend(
            "later",
            SchemaAmendment::EnableVectorIndex {
                fields: vec!["text".into()],
                vector_index: Some(spec(r#"{"embedding_model":"e5","embedding_dim":8}"#)),
            },
        )
        .await
        .unwrap_err();
    assert!(matches!(err, DocStoreError::UnsupportedEmbeddingModel { .. }), "{err:?}");
    assert!(!store.get_schema("later").unwrap().semantic_search_enabled, "nothing saved");
    store.shutdown_vec_index_worker().await;
    store.shutdown().await.unwrap();
}
