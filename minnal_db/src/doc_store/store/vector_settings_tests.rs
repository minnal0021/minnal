//! A namespace's vector-index settings through the store: defaults written on
//! create, fixed fields kept across a vector-index drop, search updates, model
//! checks, per-request overrides, and two namespaces with different models in
//! one process (design doc M2a).

use super::test_support::*;
use super::*;
#[cfg(feature = "semantic-search")]
use crate::doc_store::vector_settings::SearchSpec;
use crate::doc_store::vector_settings::VectorIndexSpec;

fn spec(json: &str) -> VectorIndexSpec {
    serde_json::from_str(json).unwrap()
}

/// A doc store schema with semantic search on over one `text` field.
fn semantic_schema(namespace: &str, vector_index: Option<VectorIndexSpec>) -> DocStoreSchema {
    let mut s = make_schema(namespace, vec![]);
    s.attributes = vec![crate::doc_store::schema::AttributeDef {
        name: "text".into(),
        attr_type: AttributeType::Str,
        description: None,
    }];
    s.semantic_search_enabled = true;
    s.embedding_fields = vec!["text".into()];
    s.vector_index = vector_index;
    s
}

/// The `vector_index` object as saved in `system_schema`.
#[cfg(feature = "semantic-search")]
fn saved_vector_index(store: &DocStore, namespace: &str) -> serde_json::Value {
    let json: serde_json::Value = match store.schemas.get(namespace).unwrap() {
        StoredSchema::Doc(s) => serde_json::to_value(&*s).unwrap(),
        StoredSchema::Kv(s) => serde_json::to_value(&*s).unwrap(),
    };
    json["vector_index"].clone()
}

// Creates semantic stores, which need the feature compiled in.
#[cfg(feature = "semantic-search")]
#[tokio::test]
async fn create_writes_every_default_into_the_saved_schema() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;
    store.create(semantic_schema("docs", None)).await.unwrap();
    let mut kv = make_kv_schema("kvs", KvKeyType::Str, KvValueType::Str);
    kv.semantic_search_enabled = true;
    kv.vector_index = Some(spec(r#"{"embedding_model":"Qwen","chunking":{"window_size":6}}"#));
    store.create_kv(kv).await.unwrap();

    let expected = serde_json::json!({
        "embedding_model": "gemma", "embedding_dim": 768,
        "chunking": {"window_size": 4, "sliding_size": 2},
        "quantisation": {"pass1_bits": 1, "pass2_bits": 8, "rotation_seed": "0x6d696e6e616c0001", "pass2_centre": "zero"},
        "search": {"min_probes": 1, "max_probes": 256, "first_pass_top_k": 1000, "top_k": 100},
        "maintenance": {"target_posting_size": 128, "reassign_range": 64, "merge_ratio": 10.0 / 118.0,
                        "skip_uncertain_moves": false, "split_samples": 1000, "split_init_trials": 3,
                        "split_max_iters": 100, "split_lambda_factor": 100.0}
    });
    assert_eq!(saved_vector_index(&store, "docs"), expected);
    let kv_saved = saved_vector_index(&store, "kvs");
    assert_eq!(kv_saved["embedding_model"], "qwen", "lower-cased on save");
    assert_eq!(kv_saved["chunking"], serde_json::json!({"window_size": 6, "sliding_size": 2}));
    assert_eq!(kv_saved["search"]["top_k"], 100);

    // A store without semantic search keeps no settings.
    store.create(make_schema("plain", vec![])).await.unwrap();
    assert!(saved_vector_index(&store, "plain").is_null());
    store.shutdown().await.unwrap();
}

#[tokio::test]
async fn create_rejects_invalid_settings_and_saves_nothing() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;
    for (json, field) in [
        (r#"{"embedding_dim":7}"#, "embedding_dim"),
        (r#"{"embedding_model":"bad model"}"#, "embedding_model"),
        (r#"{"chunking":{"window_size":4,"sliding_size":5}}"#, "chunking.sliding_size"),
        (r#"{"quantisation":{"pass2_bits":4}}"#, "quantisation.pass2_bits"),
        (r#"{"search":{"first_pass_top_k":10,"top_k":20}}"#, "search.top_k"),
    ] {
        let err = store.create(semantic_schema("bad", Some(spec(json)))).await.unwrap_err();
        assert!(
            matches!(&err, DocStoreError::Schema(SchemaError::InvalidVectorSetting { field: f, .. }) if *f == field),
            "{json}: {err:?}"
        );
        assert!(!store.schemas.contains("bad"), "{json}: nothing saved");
    }
    store.shutdown().await.unwrap();
}

// Creates semantic stores, which need the feature compiled in.
#[cfg(feature = "semantic-search")]
#[tokio::test]
async fn fixed_settings_survive_a_vector_index_drop_and_re_enable() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;
    store
        .create(semantic_schema(
            "docs",
            Some(spec(r#"{"embedding_model":"qwen","chunking":{"window_size":6,"sliding_size":3}}"#)),
        ))
        .await
        .unwrap();
    store.disable_semantic_search("docs").unwrap();
    let after_drop = store.get_schema("docs").unwrap();
    assert!(!after_drop.semantic_search_enabled);
    assert_eq!(after_drop.vector_settings().unwrap().embedding_model, "qwen", "settings are kept on drop");

    // Re-enabling with another model, dimension or chunking is rejected...
    for json in [
        r#"{"embedding_model":"gemma"}"#,
        r#"{"embedding_dim":1024}"#,
        r#"{"chunking":{"window_size":4}}"#,
        r#"{"quantisation":{"pass1_bits":2}}"#,
    ] {
        let err = store
            .amend(
                "docs",
                SchemaAmendment::EnableVectorIndex {
                    fields: vec!["body".into()],
                    vector_index: Some(spec(json)),
                },
            )
            .await
            .unwrap_err();
        assert!(
            matches!(
                err,
                DocStoreError::Schema(SchemaError::VectorSettingFixed { .. }) | DocStoreError::Schema(SchemaError::InvalidVectorSetting { .. })
            ),
            "{json}: {err:?}"
        );
        assert!(!store.get_schema("docs").unwrap().semantic_search_enabled, "{json}: schema untouched");
    }
    // ...and with them omitted or repeated it keeps the original ones, while
    // search settings may change.
    store
        .amend(
            "docs",
            SchemaAmendment::AddEmbeddingAttribute {
                name: "body".into(),
                description: None,
                vector_index: Some(spec(r#"{"embedding_model":"QWEN","search":{"max_probes":8}}"#)),
            },
        )
        .await
        .unwrap();
    let s = store.get_schema("docs").unwrap().vector_settings().unwrap();
    assert_eq!(
        (s.embedding_model.as_str(), s.window_size, s.sliding_size, s.search.max_probes),
        ("qwen", 6, 3, 8)
    );
    store.shutdown().await.unwrap();
}

// Creates semantic stores, which need the feature compiled in.
#[cfg(feature = "semantic-search")]
#[tokio::test]
async fn update_vector_search_changes_only_search_settings() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;
    store.create(semantic_schema("docs", None)).await.unwrap();
    let mut kv = make_kv_schema("kvs", KvKeyType::Str, KvValueType::Str);
    kv.semantic_search_enabled = true;
    store.create_kv(kv).await.unwrap();
    store
        .create_kv(make_kv_schema("plain_kv", KvKeyType::Str, KvValueType::Str))
        .await
        .unwrap();

    let update = SearchSpec {
        max_probes: Some(160),
        top_k: Some(10),
        ..Default::default()
    };
    store.update_vector_search("docs", &update, None).await.unwrap();
    store.update_vector_search("kvs", &update, None).await.unwrap();
    for ns in ["docs", "kvs"] {
        let saved = saved_vector_index(&store, ns);
        assert_eq!(
            saved["search"],
            serde_json::json!({
                "min_probes": 1,
                "max_probes": 160,
                "first_pass_top_k": 1000,
                "top_k": 10
            }),
            "{ns}"
        );
        assert_eq!(saved["embedding_model"], "gemma", "{ns}");
    }

    let bad = SearchSpec {
        max_probes: Some(0),
        ..Default::default()
    };
    assert!(matches!(
        store.update_vector_search("docs", &bad, None).await.unwrap_err(),
        DocStoreError::Schema(SchemaError::InvalidVectorSetting {
            field: "search.max_probes",
            ..
        })
    ));
    assert!(matches!(
        store.update_vector_search("plain_kv", &update, None).await.unwrap_err(),
        DocStoreError::Schema(SchemaError::VectorIndexNotConfigured { .. })
    ));
    assert!(matches!(
        store.update_vector_search("docs", &SearchSpec::default(), None).await.unwrap_err(),
        DocStoreError::Schema(SchemaError::InvalidVectorSetting { field: "search", .. })
    ));
    store.shutdown().await.unwrap();
}

#[cfg(feature = "semantic-search")]
mod with_service {
    use super::*;
    use crate::semantic_search::chunk_document;
    use crate::semantic_search::service::MAX_TRACE_TOP_K;
    use crate::semantic_search::service::SemanticSearchConfig;

    const DIM: usize = 8;

    /// One embedding request the mock saw.
    #[derive(Debug, Clone, PartialEq)]
    struct Seen {
        model: String,
        kind: String,
        dimensions: usize,
        payloads: usize,
    }

    /// A mock embedding service serving `gemma` and `qwen` at any dimension:
    /// one vector per payload, a different one per model, and every request
    /// recorded.
    fn spawn_service() -> (String, std::sync::Arc<parking_lot::Mutex<Vec<Seen>>>) {
        spawn_slow_service(std::time::Duration::ZERO)
    }

    /// [`spawn_service`], answering each request `delay` after it arrives (one
    /// at a time).
    fn spawn_slow_service(delay: std::time::Duration) -> (String, std::sync::Arc<parking_lot::Mutex<Vec<Seen>>>) {
        spawn_service_with(delay, false)
    }

    /// [`spawn_service`] whose vectors come from each payload's text, so
    /// different texts get different vectors whichever request they arrive in.
    /// The plain mock's vectors depend only on a payload's position in its
    /// request, so documents embedded one at a time all get the same few.
    fn spawn_text_service() -> (String, std::sync::Arc<parking_lot::Mutex<Vec<Seen>>>) {
        spawn_service_with(std::time::Duration::ZERO, true)
    }

    fn spawn_service_with(delay: std::time::Duration, vectors_from_text: bool) -> (String, std::sync::Arc<parking_lot::Mutex<Vec<Seen>>>) {
        use std::hash::{Hash, Hasher};
        use std::io::{BufRead, BufReader, Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let seen = std::sync::Arc::new(parking_lot::Mutex::new(Vec::new()));
        let log = seen.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { break };
                let mut reader = BufReader::new(stream);
                let mut line = String::new();
                if reader.read_line(&mut line).is_err() {
                    continue;
                }
                let path = line.split_whitespace().nth(1).unwrap_or_default().to_string();
                let mut len = 0usize;
                loop {
                    let mut h = String::new();
                    if reader.read_line(&mut h).is_err() || h == "\r\n" || h.is_empty() {
                        break;
                    }
                    if let Some(v) = h.to_ascii_lowercase().strip_prefix("content-length:") {
                        len = v.trim().parse().unwrap_or(0);
                    }
                }
                let mut body = vec![0u8; len];
                let _ = reader.read_exact(&mut body);
                let parts: Vec<&str> = path.split('/').collect(); // ["", "embedding", model, kind]
                let (status, out) = if parts.len() == 4 && matches!(parts[2], "gemma" | "qwen") {
                    let req: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
                    let n = req["payloads"].as_array().map_or(0, |a| a.len());
                    let dims = req["dimensions"].as_u64().unwrap_or(0) as usize;
                    log.lock().push(Seen {
                        model: parts[2].into(),
                        kind: parts[3].into(),
                        dimensions: dims,
                        payloads: n,
                    });
                    std::thread::sleep(delay);
                    // gemma points along +x, qwen along +y; each payload slightly different.
                    let axis = if parts[2] == "gemma" { 0 } else { 1 };
                    let embs: Vec<Vec<f32>> = (0..n)
                        .map(|i| {
                            let mut v = vec![0.01f32 * (i as f32 + 1.0); dims];
                            if vectors_from_text {
                                for (d, x) in v.iter_mut().enumerate() {
                                    let mut h = std::collections::hash_map::DefaultHasher::new();
                                    (req["payloads"][i].to_string(), d).hash(&mut h);
                                    *x = (h.finish() % 1000) as f32 / 1000.0;
                                }
                            }
                            v[axis] = 1.0;
                            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                            v.iter().map(|x| x / norm).collect()
                        })
                        .collect();
                    ("200 OK", serde_json::json!({ "embeddings": embs }).to_string())
                } else {
                    ("404 Not Found", serde_json::json!({ "detail": "Unknown model" }).to_string())
                };
                let mut stream = reader.into_inner();
                let _ = write!(
                    stream,
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{out}",
                    out.len()
                );
                let _ = stream.flush();
            }
        });
        (url, seen)
    }

    fn context(url: &str) -> SemanticSearchContext {
        SemanticSearchContext::new(SemanticSearchConfig {
            embedding_service_url: url.to_string(),
            ..SemanticSearchConfig::default()
        })
        .with_supported_models([("gemma".to_string(), 8), ("qwen".to_string(), 8)])
    }

    async fn wait_for_empty_queue(store: &DocStore) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while !vector_kv::list_queue_entries(&store.db).await.unwrap().is_empty() {
            assert!(std::time::Instant::now() < deadline, "the vector queue never drained");
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
    }

    /// The cluster ids of a namespace's stored chunk keys.
    async fn sparse_clusters(store: &DocStore, namespace: &str) -> Vec<u32> {
        let rows = store
            .db
            .namespace(vector_kv::sparse_vectors_ns(namespace))
            .await
            .unwrap()
            .scan_prefix(Vec::new())
            .await
            .unwrap();
        let mut ids: Vec<u32> = rows.iter().map(|(k, _)| u32::from_be_bytes(k[..4].try_into().unwrap())).collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    const TEXT: &str = "One sentence here. Two sentences now. Three of them. Four in a row. Five so far. Six at last.";

    /// A namespace grows from its root by itself: the worker wakes maintenance
    /// when a posting passes its split limit, and the splits keep every key in
    /// an active posting with exact counts, also across a reopen.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn postings_split_by_themselves_as_a_namespace_grows() {
        let (url, _seen) = spawn_service();
        let db_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path()).await.with_semantic_search(context(&url));
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
                let keys = crate::ivf_split::recount_chunks(&db, "g").await.unwrap();
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

        let store = open_fresh(db_dir.path()).await.with_semantic_search(context(&url));
        let ctx = Arc::clone(store.semantic_ctx.as_ref().unwrap());
        let partition = ctx.for_namespace(&store.db, "g", ns_id, &settings).await.unwrap().partition;
        let ivf = partition.snapshot();
        let keys = crate::ivf_split::recount_chunks(&store.db, "g").await.unwrap();
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
        let store = open_fresh(db_dir.path()).await.with_semantic_search(context(&url));
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
        let keys = crate::ivf_split::recount_chunks(&store.db, "g").await.unwrap();
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
        use crate::ivf_split::rebuild::RebuildState;
        let (url, seen) = spawn_service();
        let db_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path()).await.with_semantic_search(context(&url));
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
        store.disable_semantic_search("plain").unwrap();
        assert!(matches!(
            store.start_vector_rebuild("plain", None),
            Err(DocStoreError::SemanticSearchNotEnabled { .. })
        ));
        assert!(matches!(store.start_vector_rebuild("missing", None), Err(DocStoreError::NotFound { .. })));
        let busy = partition.lock_maintenance().await; // holds the rebuild back
        store.start_vector_rebuild("g", Some(2)).unwrap();
        assert!(matches!(
            store.start_vector_rebuild("g", None),
            Err(DocStoreError::VecRebuildInProgress { .. })
        ));
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
        let params = crate::ivf_split::recall_check::RecallCheckParams {
            sample: 10,
            ..Default::default()
        };
        let check = store.vector_recall_check("g", &params, &SearchSpec::default()).await.unwrap();
        assert_eq!(check.sampled, 10, "{check:?}");
        assert_eq!(seen.lock().len(), calls, "a recall check makes no embedding request");
        use crate::ivf_split::history::RecordKind;
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
        let keys = crate::ivf_split::recount_chunks(&store.db, "g").await.unwrap();
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

    /// A probe trace covers the window it is asked for, at absolute ranks: by
    /// default the namespace's `max_probes` nearest postings (all of a small
    /// namespace), any window being a slice of that order. Its baseline is the
    /// namespace's own search, and a window that is empty, too wide or past the
    /// last posting is refused.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_probe_trace_covers_the_window_asked_for() {
        let (url, _seen) = spawn_text_service();
        let db_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path()).await.with_semantic_search(context(&url));
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
        assert_eq!(MAX_TRACE_TOP_K, crate::doc_store::vector_settings::MAX_FIRST_PASS_TOP_K);
    }

    /// The loaded partition's entry counts track the worker's writes and the
    /// doc store's deletes, matching a fresh count of the stored keys.
    #[tokio::test]
    async fn entry_counts_track_indexing_updates_and_deletes() {
        let (url, _seen) = spawn_service();
        let db_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path()).await.with_semantic_search(context(&url));
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
        let store = open_fresh(db_dir.path()).await.with_semantic_search(context(&url));
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
        let store = open_fresh(db_dir.path()).await.with_semantic_search(context(&url));
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
        let store = open_fresh(db_dir.path()).await.with_semantic_search(context(&url));

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

    // ── M2b: the namespace's own partition ──

    async fn results(store: &DocStore, ns: &str) -> Vec<(Vec<u8>, u32)> {
        let page = store
            .search_semantic(ns, "hello", &SearchSpec::default(), Pagination::default())
            .await
            .unwrap();
        page.results.iter().map(|r| (r.document_id.clone(), r.dot_product.to_bits())).collect()
    }

    #[tokio::test]
    async fn a_new_namespace_starts_from_one_root_posting_and_a_drop_resets_it() {
        let (url, _seen) = spawn_service();
        let db_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path()).await.with_semantic_search(context(&url));
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
        store.disable_semantic_search("g").unwrap();
        store.drop_vector_index_data("g").await.unwrap();
        store.shutdown_vec_index_worker().await;
        store.shutdown().await.unwrap();
        drop(store);
        let store = open_fresh(db_dir.path()).await.with_semantic_search(context(&url));
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
        let store = open_fresh(db_dir.path()).await.with_semantic_search(context(&url));
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

    // ── Review fixes (2026-10-17) ──

    /// The partition of `namespace`, loaded as every user of it loads it.
    async fn partition(store: &DocStore, namespace: &str) -> std::sync::Arc<crate::semantic_search::PartitionHandle> {
        let (settings, ns_id) = store.schemas.vector_settings(namespace).unwrap();
        store
            .semantic_ctx
            .as_ref()
            .unwrap()
            .for_namespace(&store.db, namespace, ns_id, &settings)
            .await
            .unwrap()
            .partition
    }

    /// Names of `namespace`'s companion namespaces that exist.
    fn companions_left(store: &DocStore, namespace: &str) -> Vec<String> {
        let prefix = format!("{namespace}_");
        store
            .db
            .list_namespaces()
            .into_iter()
            .map(|(n, _)| n)
            .filter(|n| n.starts_with(&prefix))
            .collect()
    }

    /// Dropping a vector index, or its whole store, while a maintenance
    /// operation runs must wait for the operation and stop maintenance on that
    /// partition. A write it made after the drop would land in a store being
    /// removed, or leave stray postings in a re-enabled partition.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_drop_waits_for_running_maintenance_and_stops_it() {
        for drop_store in [false, true] {
            let (url, _seen) = spawn_service();
            let db_dir = TempDir::new().unwrap();
            let store = open_fresh(db_dir.path()).await.with_semantic_search(context(&url));
            store.create(semantic_schema("g", Some(spec(r#"{"embedding_dim":8}"#)))).await.unwrap();
            store.put("g", DocId::U64(1), serde_json::json!({"text": TEXT})).await.unwrap();
            wait_for_empty_queue(&store).await;
            let handle = partition(&store, "g").await;
            let ops = store.schemas.vector_settings("g").unwrap().0.maintenance.ops();
            let store = std::sync::Arc::new(store);

            // An operation in progress: it holds the maintenance lock and writes.
            let running = handle.lock_maintenance().await;
            let dropping = tokio::spawn({
                let store = std::sync::Arc::clone(&store);
                async move {
                    if drop_store {
                        store.remove("g").await.unwrap();
                    } else {
                        store.disable_semantic_search("g").unwrap();
                        store.drop_vector_index_data("g").await.unwrap();
                    }
                }
            });
            tokio::time::sleep(std::time::Duration::from_millis(300)).await;
            store
                .db
                .namespace(vector_kv::ivf_journal_ns("g"))
                .await
                .unwrap()
                .put(b"op".to_vec(), b"record".to_vec())
                .await
                .unwrap();
            drop(running);
            dropping.await.unwrap();
            // Maintenance on the partition it held, as a queued request would run it.
            assert!(
                crate::ivf_split::maintain_partition(&store.db, "g", &handle, &ops).await.is_err(),
                "drop_store={drop_store}: maintenance went on after the drop"
            );
            let left = companions_left(&store, "g");
            assert!(left.is_empty(), "drop_store={drop_store}: dropped namespaces are back: {left:?}");
            store.shutdown_vec_index_worker().await;
            store.shutdown().await.unwrap();
        }
    }

    /// Embed-queue entries of `namespace`.
    async fn queued_for(store: &DocStore, namespace: &str) -> usize {
        vector_kv::list_queue_entries(&store.db)
            .await
            .unwrap()
            .iter()
            .filter(|e| e.namespace == namespace)
            .count()
    }

    /// Dropping a store while the vector worker is part-way through a pass of
    /// its entries must stop the pass on that store: the worker resolved the
    /// store's partition at the start of the pass and went on embedding the
    /// rest of its entries after the drop (embedding calls for a store that is
    /// gone, maintenance requests that failed, vector writes racing the
    /// removal of the namespaces they recreate).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_drop_stops_the_worker_pass_on_its_store() {
        let (url, seen) = spawn_slow_service(std::time::Duration::from_millis(20));
        let db_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path()).await.with_semantic_search(context(&url));
        store.create(semantic_schema("g", Some(spec(r#"{"embedding_dim":8}"#)))).await.unwrap();
        for id in 1..=60 {
            store.put("g", DocId::U64(id), serde_json::json!({"text": TEXT})).await.unwrap();
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while seen.lock().len() < 4 {
            assert!(std::time::Instant::now() < deadline, "the worker never started embedding");
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        store.remove("g").await.unwrap();
        let at_drop = seen.lock().len();
        tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
        let after = seen.lock().len() - at_drop;
        // Requests already sent when the drop ended may still be answered: at
        // most two (whole document, chunks) per concurrent entry.
        assert!(after <= 8, "{after} embedding request(s) after the drop");
        assert!(!store.db.list_namespaces().iter().any(|(n, _)| n == "g"), "the dropped store is back");
        let left = companions_left(&store, "g");
        assert!(left.is_empty(), "dropped namespaces are back: {left:?}");
        assert_eq!(queued_for(&store, "g").await, 0, "queue entries outlived the store");
        store.shutdown_vec_index_worker().await;
        store.shutdown().await.unwrap();
    }

    /// The worker reads a document to check its queue entry, outside any
    /// partition's epoch. That read racing a drop must not bring the store back:
    /// the drop's sweep of leftover directories then deleted the new store's
    /// files, and the next metadata write, at shutdown at the latest, failed with
    /// `NotFound`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_document_read_racing_a_drop_does_not_bring_the_store_back() {
        let (url, _seen) = spawn_service();
        let db_dir = TempDir::new().unwrap();
        let store = Arc::new(open_fresh(db_dir.path()).await.with_semantic_search(context(&url)));
        for round in 0..40 {
            let name = format!("g{round}");
            store.create(semantic_schema(&name, Some(spec(r#"{"embedding_dim":8}"#)))).await.unwrap();
            let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let readers: Vec<_> = (0..4)
                .map(|_| {
                    let (store, name, stop) = (Arc::clone(&store), name.clone(), Arc::clone(&stop));
                    tokio::spawn(async move {
                        while !stop.load(std::sync::atomic::Ordering::Acquire) {
                            let _ = super::super::vector::document_text(&store.db, &store.schemas, &name, b"k").await;
                            tokio::task::yield_now().await;
                        }
                    })
                })
                .collect();
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            store.remove(&name).await.unwrap();
            stop.store(true, std::sync::atomic::Ordering::Release);
            for r in readers {
                r.await.unwrap();
            }
            let names: Vec<String> = store.db.list_namespaces().into_iter().map(|(n, _)| n).collect();
            assert!(!names.contains(&name), "round {round}: the dropped store is back");
        }
        store.shutdown().await.unwrap();
    }

    /// Create a store, run `call` from several tasks while it is dropped, and
    /// check that nothing of the store came back. Repeated, since the window is
    /// small, and shut down last: a store left without its files fails there.
    async fn assert_call_racing_a_drop_recreates_nothing<F, Fut>(rounds: u32, with_docs: bool, call: F)
    where
        F: Fn(Arc<DocStore>, String, u64) -> Fut + Clone + Send + Sync + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        let (url, _seen) = spawn_service();
        let db_dir = TempDir::new().unwrap();
        let store = Arc::new(open_fresh(db_dir.path()).await.with_semantic_search(context(&url)));
        for round in 0..rounds {
            let name = format!("g{round}");
            store.create(semantic_schema(&name, Some(spec(r#"{"embedding_dim":8}"#)))).await.unwrap();
            if with_docs {
                store.put(&name, DocId::U64(0), serde_json::json!({"text": TEXT})).await.unwrap();
                wait_for_empty_queue(&store).await;
            }
            let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let tasks: Vec<_> = (0..4u64)
                .map(|task| {
                    let (store, name, stop, call) = (Arc::clone(&store), name.clone(), Arc::clone(&stop), call.clone());
                    tokio::spawn(async move {
                        let mut i = task * 1_000_000 + 1;
                        while !stop.load(std::sync::atomic::Ordering::Acquire) {
                            call(Arc::clone(&store), name.clone(), i).await;
                            i += 1;
                            tokio::task::yield_now().await;
                        }
                    })
                })
                .collect();
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            store.remove(&name).await.unwrap();
            stop.store(true, std::sync::atomic::Ordering::Release);
            for t in tasks {
                t.await.unwrap();
            }
            let names: Vec<String> = store.db.list_namespaces().into_iter().map(|(n, _)| n).collect();
            assert!(!names.contains(&name), "round {round}: the dropped store is back");
            let left = companions_left(&store, &name);
            assert!(left.is_empty(), "round {round}: dropped namespaces are back: {left:?}");
        }
        store.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_search_racing_a_drop_recreates_nothing() {
        assert_call_racing_a_drop_recreates_nothing(15, true, |store, name, _| async move {
            let _ = store
                .search_semantic(&name, "hello", &SearchSpec::default(), Pagination::new(1, 10))
                .await;
        })
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn counting_indexed_documents_racing_a_drop_recreates_nothing() {
        assert_call_racing_a_drop_recreates_nothing(5, false, |store, name, _| async move {
            let _ = store.count_indexed_docs(&name).await;
        })
        .await;
    }

    /// A document with no embedding text checks for vector state to clear
    /// (`has_any_vector_state`).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_write_without_embedding_text_racing_a_drop_recreates_nothing() {
        // No `text` at all: `{"text": ""}` still builds the embedding text "text: ".
        assert_call_racing_a_drop_recreates_nothing(10, false, |store, name, i| async move {
            let _ = store.put(&name, DocId::U64(i), serde_json::json!({})).await;
        })
        .await;
    }

    /// The drop route answers and runs in the background, so an enable can land
    /// before the drop starts. The schema then says semantic search is on, and
    /// the drop must not leave it without a partition.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn enabling_before_a_background_drop_runs_leaves_a_partition() {
        let (url, _seen) = spawn_service();
        let db_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path()).await.with_semantic_search(context(&url));
        store.create(semantic_schema("g", Some(spec(r#"{"embedding_dim":8}"#)))).await.unwrap();
        store.put("g", DocId::U64(1), serde_json::json!({"text": TEXT})).await.unwrap();
        wait_for_empty_queue(&store).await;

        store.disable_semantic_search("g").unwrap();
        let enable = || SchemaAmendment::EnableVectorIndex {
            fields: vec!["text".into()],
            vector_index: None,
        };
        store.amend("g", enable()).await.unwrap();
        store.drop_vector_index_data("g").await.unwrap();

        assert!(store.get_schema("g").unwrap().semantic_search_enabled);
        partition(&store, "g").await;
        assert_eq!(store.count_indexed_docs("g").await, 0, "the drop removed the old vectors");
        store.put("g", DocId::U64(2), serde_json::json!({"text": TEXT})).await.unwrap();
        wait_for_empty_queue(&store).await;
        assert_eq!(store.count_indexed_docs("g").await, 1, "the store indexes again");
        store.shutdown().await.unwrap();
    }

    /// A drop can only stop a partition that is installed. One still loading
    /// when the drop begins must not be handed out afterwards unmarked: the
    /// worker trusts the mark, and its write would recreate what the drop removed.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_partition_loading_during_a_vector_index_drop_is_not_handed_out() {
        let (url, _seen) = spawn_service();
        let db_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path()).await.with_semantic_search(context(&url));
        store.create(semantic_schema("g", Some(spec(r#"{"embedding_dim":8}"#)))).await.unwrap();
        store.put("g", DocId::U64(1), serde_json::json!({"text": TEXT})).await.unwrap();
        wait_for_empty_queue(&store).await;
        store.shutdown_vec_index_worker().await;
        let store = std::sync::Arc::new(store);
        store.semantic_ctx.as_ref().unwrap().evict("g"); // the next use loads

        let (reached, release) = crate::doc_store::store::types::load_pause::arm("g");
        let (settings, ns_id) = store.schemas.vector_settings("g").unwrap();
        let loading = tokio::spawn({
            let store = std::sync::Arc::clone(&store);
            async move {
                let ctx = store.semantic_ctx.as_ref().unwrap();
                ctx.for_namespace(&store.db, "g", ns_id, &settings).await.map(|ns| ns.partition)
            }
        });
        reached.notified().await; // read, not yet installed
        store.disable_semantic_search("g").unwrap();
        store.drop_vector_index_data("g").await.unwrap();
        release.notify_one();
        let loaded = loading.await.unwrap();
        assert!(
            matches!(loaded, Err(DocStoreError::VectorIndexNotSeeded { .. })),
            "the load read a partition the drop removed: is_dropped={:?}",
            loaded.as_ref().map(|p| p.is_dropped()).ok()
        );
        assert!(companions_left(&store, "g").is_empty());
        store.shutdown().await.unwrap();
    }

    /// As above for a whole store's drop.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_partition_loading_during_a_store_drop_is_not_handed_out() {
        let (url, _seen) = spawn_service();
        let db_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path()).await.with_semantic_search(context(&url));
        store.create(semantic_schema("g", Some(spec(r#"{"embedding_dim":8}"#)))).await.unwrap();
        store.put("g", DocId::U64(1), serde_json::json!({"text": TEXT})).await.unwrap();
        wait_for_empty_queue(&store).await;
        store.shutdown_vec_index_worker().await;
        let store = std::sync::Arc::new(store);
        store.semantic_ctx.as_ref().unwrap().evict("g");

        let (reached, release) = crate::doc_store::store::types::load_pause::arm("g");
        let (settings, ns_id) = store.schemas.vector_settings("g").unwrap();
        let loading = tokio::spawn({
            let store = std::sync::Arc::clone(&store);
            async move {
                let ctx = store.semantic_ctx.as_ref().unwrap();
                ctx.for_namespace(&store.db, "g", ns_id, &settings).await.map(|ns| ns.partition)
            }
        });
        reached.notified().await;
        store.remove("g").await.unwrap();
        release.notify_one();
        assert!(loading.await.unwrap().is_err(), "a dropped store's partition was handed out");
        assert!(companions_left(&store, "g").is_empty());
        store.shutdown().await.unwrap();
    }

    /// A vector-index drop does not stop writes. One that read the schema before
    /// semantic search went off still clears the document's vectors after the
    /// drop, and that must not bring the namespaces back.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_write_without_text_after_a_vector_index_drop_recreates_nothing() {
        let (url, _seen) = spawn_service();
        let db_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path()).await.with_semantic_search(context(&url));
        store.create(semantic_schema("g", Some(spec(r#"{"embedding_dim":8}"#)))).await.unwrap();
        store.disable_semantic_search("g").unwrap();
        store.drop_vector_index_data("g").await.unwrap();
        assert!(companions_left(&store, "g").is_empty(), "set-up: the drop removed everything");
        store.clear_doc_vectors_if_any("g", &DocId::U64(1).to_bytes()).await.unwrap();
        let left = companions_left(&store, "g");
        assert!(left.is_empty(), "recreated: {left:?}");
        store.shutdown().await.unwrap();
    }

    /// Every companion exists from the moment a store is semantic, however the
    /// store got there: writers rely on it and never create one. A store saved
    /// without them gets them at open rather than silently indexing nothing.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_semantic_store_has_its_vector_namespaces_from_the_start() {
        let (url, _seen) = spawn_service();
        let db_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path()).await.with_semantic_search(context(&url));
        store.create(semantic_schema("g", Some(spec(r#"{"embedding_dim":8}"#)))).await.unwrap();
        let mut left = companions_left(&store, "g");
        left.sort();
        let mut all = vector_kv::companion_namespaces("g");
        all.sort();
        assert_eq!(left, all, "seeding creates them all");
        store.db.remove_namespace(vector_kv::sparse_vectors_ns("g")).await.unwrap();
        store.shutdown().await.unwrap();
        drop(store);

        let store = DocStore::open_with_config(db_dir.path(), crate::doc_store::test_db_config())
            .await
            .unwrap()
            .with_semantic_search(context(&url));
        let names = companions_left(&store, "g");
        assert!(names.contains(&vector_kv::sparse_vectors_ns("g")), "open restores it: {names:?}");
        store.put("g", DocId::U64(1), serde_json::json!({"text": TEXT})).await.unwrap();
        wait_for_empty_queue(&store).await;
        assert_eq!(store.count_indexed_docs("g").await, 1);
        store.shutdown().await.unwrap();
    }

    /// Enables and drops racing in any order: whenever the schema ends with
    /// semantic search on, the partition is there.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn enables_racing_drops_always_leave_a_partition() {
        let (url, _seen) = spawn_service();
        let db_dir = TempDir::new().unwrap();
        let store = std::sync::Arc::new(open_fresh(db_dir.path()).await.with_semantic_search(context(&url)));
        store.create(semantic_schema("g", Some(spec(r#"{"embedding_dim":8}"#)))).await.unwrap();
        for round in 0..30 {
            store.disable_semantic_search("g").unwrap();
            let dropping = tokio::spawn({
                let store = std::sync::Arc::clone(&store);
                async move { store.drop_vector_index_data("g").await.unwrap() }
            });
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
            dropping.await.unwrap();
            let (settings, ns_id) = store.schemas.vector_settings("g").unwrap();
            let loaded = store.semantic_ctx.as_ref().unwrap().for_namespace(&store.db, "g", ns_id, &settings).await;
            assert!(loaded.is_ok(), "round {round}: semantic search is on, but: {:?}", loaded.err());
        }
        store.shutdown().await.unwrap();
    }

    /// A drop waiting out one store's maintenance (a rebuild's k-means can take
    /// minutes) holds up that store's admin operations and no other's.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_waiting_drop_holds_up_only_its_own_store() {
        let (url, _seen) = spawn_service();
        let db_dir = TempDir::new().unwrap();
        let store = std::sync::Arc::new(open_fresh(db_dir.path()).await.with_semantic_search(context(&url)));
        store.create(semantic_schema("g", Some(spec(r#"{"embedding_dim":8}"#)))).await.unwrap();
        store.create(make_schema("other", vec![])).await.unwrap();
        let add = |name: &str| SchemaAmendment::AddAttribute {
            name: name.to_owned(),
            attr_type: AttributeType::Str,
            description: None,
        };
        let loaded = partition(&store, "g").await;
        let running = loaded.lock_maintenance().await;
        let dropping = tokio::spawn({
            let store = std::sync::Arc::clone(&store);
            async move { store.remove("g").await.unwrap() }
        });
        let ctx = store.semantic_ctx.as_ref().unwrap();
        while !ctx.is_dropping("g") {
            tokio::task::yield_now().await;
        }

        let within = std::time::Duration::from_secs(5);
        tokio::time::timeout(within, store.create(make_schema("new", vec![])))
            .await
            .expect("a create of another store waited for the drop")
            .unwrap();
        tokio::time::timeout(within, store.amend("other", add("a")))
            .await
            .expect("an amendment of another store waited for the drop")
            .unwrap();
        let own = tokio::time::timeout(std::time::Duration::from_millis(200), store.amend("g", add("b"))).await;
        assert!(own.is_err(), "an amendment of the store being dropped must wait");

        drop(running);
        dropping.await.unwrap();
        store.shutdown().await.unwrap();
    }

    /// Counting indexed documents only reads, so asking about a store that does
    /// not exist must not create anything.
    #[tokio::test]
    async fn counting_indexed_documents_of_an_unknown_store_creates_nothing() {
        let (url, _seen) = spawn_service();
        let db_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path()).await.with_semantic_search(context(&url));
        assert_eq!(store.count_indexed_docs("nope").await, 0);
        let left = companions_left(&store, "nope");
        assert!(left.is_empty(), "{left:?}");
        store.shutdown().await.unwrap();
    }

    /// Dropping a store while a reindex of it enqueues must leave none of its
    /// entries in the queue: the drop cleared the queue once, and the reindex,
    /// enqueueing from its own scan of the documents, went on after it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_drop_during_a_reindex_leaves_no_queue_entries() {
        let (url, _seen) = spawn_service();
        let db_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path()).await.with_semantic_search(context(&url));
        store.create(semantic_schema("g", Some(spec(r#"{"embedding_dim":8}"#)))).await.unwrap();
        // No worker: the queue keeps what the reindex enqueues.
        store.shutdown_vec_index_worker().await;
        for id in 1..=3000 {
            store.put("g", DocId::U64(id), serde_json::json!({"text": TEXT})).await.unwrap();
        }
        let store = std::sync::Arc::new(store);
        // The puts' own entries, so the reindex's are the ones counted below.
        store.delete_all_queue_entries("g").await.unwrap();
        let reindex = tokio::spawn({
            let store = std::sync::Arc::clone(&store);
            async move { store.index_all("g").await }
        });
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while queued_for(&store, "g").await < 50 {
            assert!(std::time::Instant::now() < deadline, "the reindex never enqueued");
            assert!(!reindex.is_finished(), "the reindex finished before the drop; make it longer");
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        store.remove("g").await.unwrap();
        assert!(reindex.await.unwrap().is_err(), "the reindex went on after the drop");
        assert_eq!(queued_for(&store, "g").await, 0, "queue entries outlived the store");
        store.shutdown().await.unwrap();
    }

    /// A worker pass that starts while a store is being dropped must leave the
    /// store's entries alone: their settings no longer resolve, and each one
    /// failed with a warning and a retry (a log line per queued document of a
    /// large store, and a retry wait after the pass).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_pass_during_a_drop_skips_the_dropped_store() {
        let (url, seen) = spawn_service();
        let db_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path()).await.with_semantic_search(context(&url));
        store.create(semantic_schema("g", Some(spec(r#"{"embedding_dim":8}"#)))).await.unwrap();
        // No worker while the entries are queued.
        store.shutdown_vec_index_worker().await;
        for id in 1..=20 {
            store.put("g", DocId::U64(id), serde_json::json!({"text": TEXT})).await.unwrap();
        }
        let ctx = std::sync::Arc::clone(store.semantic_ctx.as_ref().unwrap());
        ctx.begin_drop("g").await;
        let at_drop = seen.lock().len();
        let worker = crate::doc_store::vec_index_worker::VecIndexWorker::start(
            std::sync::Arc::clone(&store.db),
            std::sync::Arc::clone(&ctx),
            std::sync::Arc::clone(&store.schemas),
            std::sync::Arc::clone(&store.in_flight),
            std::sync::Arc::new(tokio::sync::Notify::new()),
            store.vector_index_config.clone(),
        );
        // A worker always makes its startup pass before it can see the
        // shutdown, so the pass has run once `shutdown` returns.
        worker.shutdown().await;
        ctx.end_drop("g");
        let entries: Vec<_> = vector_kv::list_queue_entries(&store.db)
            .await
            .unwrap()
            .into_iter()
            .filter(|e| e.namespace == "g")
            .collect();
        assert_eq!(entries.len(), 20, "the pass removed entries the drop owns");
        assert!(
            entries.iter().all(|e| e.retry_count == 0),
            "the pass failed entries of a store being dropped"
        );
        assert_eq!(seen.lock().len(), at_drop, "embedding calls for a store being dropped");
        store.shutdown().await.unwrap();
    }

    /// Writes that keep arriving while a store is dropped must neither bring
    /// its namespace back nor leave embed-queue entries behind: a `put` read the
    /// schema (removed last), then resolved the namespace with get-or-create
    /// after the drop had removed it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn writes_during_a_drop_leave_nothing_behind() {
        let (url, _seen) = spawn_service();
        let db_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path()).await.with_semantic_search(context(&url));
        store.create(semantic_schema("g", Some(spec(r#"{"embedding_dim":8}"#)))).await.unwrap();
        store.shutdown_vec_index_worker().await;
        for id in 1..=500 {
            store.put("g", DocId::U64(id), serde_json::json!({"text": TEXT})).await.unwrap();
        }
        let store = std::sync::Arc::new(store);
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let writers: Vec<_> = (0..4u64)
            .map(|w| {
                let (store, stop) = (std::sync::Arc::clone(&store), std::sync::Arc::clone(&stop));
                tokio::spawn(async move {
                    let mut id = 1_000 + w * 1_000_000;
                    while !stop.load(std::sync::atomic::Ordering::Acquire) {
                        let _ = store.put("g", DocId::U64(id), serde_json::json!({"text": TEXT})).await;
                        let _ = store.delete("g", DocId::U64(id - 1)).await;
                        id += 1;
                        // A refused write returns at once: let the drop run.
                        tokio::task::yield_now().await;
                    }
                })
            })
            .collect();
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        store.remove("g").await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        stop.store(true, std::sync::atomic::Ordering::Release);
        for w in writers {
            w.await.unwrap();
        }
        let names: Vec<String> = store.db.list_namespaces().into_iter().map(|(n, _)| n).collect();
        assert!(!names.iter().any(|n| n == "g"), "the dropped store's namespace is back");
        let left = companions_left(&store, "g");
        assert!(left.is_empty(), "dropped namespaces are back: {left:?}");
        assert_eq!(queued_for(&store, "g").await, 0, "queue entries outlived the store");
        assert!(store.get("g", DocId::U64(1)).await.is_err(), "a read of the dropped store found it");
        store.shutdown().await.unwrap();
    }

    /// A document deleted while its namespace's partition is first loaded
    /// (after the load counted the store, before it was installed) must not
    /// leave the counts one too high: a delete's count change was dropped when
    /// no partition was loaded, so the load's count, which saw the document,
    /// stayed.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_delete_during_the_first_load_keeps_the_counts_exact() {
        let (url, _seen) = spawn_service();
        let db_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path()).await.with_semantic_search(context(&url));
        store.create(semantic_schema("g", Some(spec(r#"{"embedding_dim":8}"#)))).await.unwrap();
        for id in 1..=3 {
            store.put("g", DocId::U64(id), serde_json::json!({"text": TEXT})).await.unwrap();
        }
        wait_for_empty_queue(&store).await;
        // No worker from here: its own first load would install a partition
        // counted after the delete and hide the foreground path under test.
        store.shutdown_vec_index_worker().await;
        let store = std::sync::Arc::new(store);
        store.semantic_ctx.as_ref().unwrap().evict("g"); // the next use loads and counts

        let (reached, release) = crate::doc_store::store::types::load_pause::arm("g");
        let loading = tokio::spawn({
            let store = std::sync::Arc::clone(&store);
            async move { partition(&store, "g").await }
        });
        reached.notified().await; // counted, not yet installed
        let deleting = tokio::spawn({
            let store = std::sync::Arc::clone(&store);
            async move { store.delete("g", DocId::U64(2)).await.unwrap() }
        });
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        release.notify_one();
        deleting.await.unwrap();
        let handle = loading.await.unwrap();

        let truth: std::collections::HashMap<u32, (u64, u64)> = vector_kv::count_postings(&store.db, "g").await.unwrap();
        let counted: std::collections::HashMap<u32, (u64, u64)> =
            handle.snapshot().counts().snapshot().into_iter().filter(|(_, c)| c.0 > 0).collect();
        assert_eq!(counted, truth, "the loaded counts differ from the store's");
        store.shutdown().await.unwrap();
    }

    /// Every user of a partition gets it only after its journal is recovered:
    /// a write before recovery could route to a split's children that recovery
    /// then retires, chunks and all.
    #[tokio::test]
    async fn a_partition_is_handed_out_only_after_its_journal_is_recovered() {
        let (url, _seen) = spawn_service();
        let db_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path()).await.with_semantic_search(context(&url));
        store.create(semantic_schema("g", Some(spec(r#"{"embedding_dim":8}"#)))).await.unwrap();
        for id in 1..=5 {
            store.put("g", DocId::U64(id), serde_json::json!({"text": TEXT})).await.unwrap();
        }
        wait_for_empty_queue(&store).await;
        let handle = partition(&store, "g").await;
        // A split cut short once it is published (the copy fails), as a crash
        // would leave it.
        crate::vector_kv::crash_point::arm(7);
        let split = crate::ivf_split::split_posting(&store.db, "g", &handle, 0, &crate::ivf_split::MaintenanceOps::split_only(1)).await;
        crate::vector_kv::crash_point::disarm();
        assert!(split.is_err(), "set-up: the split was cut short");
        let journal = || async {
            store
                .db
                .namespace(vector_kv::ivf_journal_ns("g"))
                .await
                .unwrap()
                .scan_prefix(Vec::new())
                .await
                .unwrap()
                .len()
        };
        assert_eq!(journal().await, 1, "set-up: the split is in the journal");

        // A restart: the partition is loaded again, by its next user.
        store.semantic_ctx.as_ref().unwrap().evict("g");
        let handle = partition(&store, "g").await;
        assert_eq!(journal().await, 0, "the partition was handed out with its journal unfinished");
        assert!(!handle.needs_recovery());
        assert_eq!(
            handle.snapshot().posting_infos()[&0].state,
            crate::semantic_search::PostingState::Retired,
            "the split was finished"
        );
        store.shutdown_vec_index_worker().await;
        store.shutdown().await.unwrap();
    }
}
