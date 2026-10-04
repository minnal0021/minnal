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

/// The `vector_index` object as saved on disk.
#[cfg(feature = "semantic-search")]
fn saved_vector_index(schema_dir: &Path, namespace: &str) -> serde_json::Value {
    let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(schema_dir.join(format!("{namespace}.json"))).unwrap()).unwrap();
    json["vector_index"].clone()
}

// Creates semantic stores, which need the feature compiled in.
#[cfg(feature = "semantic-search")]
#[tokio::test]
async fn create_writes_every_default_into_the_saved_schema() {
    let (db_dir, schema_dir) = (TempDir::new().unwrap(), TempDir::new().unwrap());
    let store = open_fresh(db_dir.path(), schema_dir.path()).await;
    store.create(semantic_schema("docs", None)).await.unwrap();
    let mut kv = make_kv_schema("kvs", KvKeyType::Str, KvValueType::Str);
    kv.semantic_search_enabled = true;
    kv.vector_index = Some(spec(r#"{"embedding_model":"Qwen","chunking":{"window_size":6}}"#));
    store.create_kv(kv).await.unwrap();

    let expected = serde_json::json!({
        "embedding_model": "gemma", "embedding_dim": 768,
        "chunking": {"window_size": 4, "sliding_size": 2},
        "quantisation": {"pass1_bits": 1, "pass2_bits": 8},
        "search": {"n_probes": 64, "first_pass_top_k": 1000, "top_k": 100}
    });
    assert_eq!(saved_vector_index(schema_dir.path(), "docs"), expected);
    let kv_saved = saved_vector_index(schema_dir.path(), "kvs");
    assert_eq!(kv_saved["embedding_model"], "qwen", "lower-cased on save");
    assert_eq!(kv_saved["chunking"], serde_json::json!({"window_size": 6, "sliding_size": 2}));
    assert_eq!(kv_saved["search"]["top_k"], 100);

    // A store without semantic search keeps no settings.
    store.create(make_schema("plain", vec![])).await.unwrap();
    assert!(saved_vector_index(schema_dir.path(), "plain").is_null());
    store.shutdown().await.unwrap();
}

#[tokio::test]
async fn create_rejects_invalid_settings_and_saves_nothing() {
    let (db_dir, schema_dir) = (TempDir::new().unwrap(), TempDir::new().unwrap());
    let store = open_fresh(db_dir.path(), schema_dir.path()).await;
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
        assert!(!schema_dir.path().join("bad.json").exists(), "{json}: nothing saved");
    }
    store.shutdown().await.unwrap();
}

// Creates semantic stores, which need the feature compiled in.
#[cfg(feature = "semantic-search")]
#[tokio::test]
async fn fixed_settings_survive_a_vector_index_drop_and_re_enable() {
    let (db_dir, schema_dir) = (TempDir::new().unwrap(), TempDir::new().unwrap());
    let store = open_fresh(db_dir.path(), schema_dir.path()).await;
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
                vector_index: Some(spec(r#"{"embedding_model":"QWEN","search":{"n_probes":8}}"#)),
            },
        )
        .unwrap();
    let s = store.get_schema("docs").unwrap().vector_settings().unwrap();
    assert_eq!(
        (s.embedding_model.as_str(), s.window_size, s.sliding_size, s.search.n_probes),
        ("qwen", 6, 3, 8)
    );
    store.shutdown().await.unwrap();
}

// Creates semantic stores, which need the feature compiled in.
#[cfg(feature = "semantic-search")]
#[tokio::test]
async fn update_vector_search_changes_only_search_settings() {
    let (db_dir, schema_dir) = (TempDir::new().unwrap(), TempDir::new().unwrap());
    let store = open_fresh(db_dir.path(), schema_dir.path()).await;
    store.create(semantic_schema("docs", None)).await.unwrap();
    let mut kv = make_kv_schema("kvs", KvKeyType::Str, KvValueType::Str);
    kv.semantic_search_enabled = true;
    store.create_kv(kv).await.unwrap();
    store
        .create_kv(make_kv_schema("plain_kv", KvKeyType::Str, KvValueType::Str))
        .await
        .unwrap();

    let update = SearchSpec {
        n_probes: Some(16),
        top_k: Some(10),
        ..Default::default()
    };
    store.update_vector_search("docs", &update).unwrap();
    store.update_vector_search("kvs", &update).unwrap();
    for ns in ["docs", "kvs"] {
        let saved = saved_vector_index(schema_dir.path(), ns);
        assert_eq!(
            saved["search"],
            serde_json::json!({"n_probes": 16, "first_pass_top_k": 1000, "top_k": 10}),
            "{ns}"
        );
        assert_eq!(saved["embedding_model"], "gemma", "{ns}");
    }

    let bad = SearchSpec {
        n_probes: Some(0),
        ..Default::default()
    };
    assert!(matches!(
        store.update_vector_search("docs", &bad).unwrap_err(),
        DocStoreError::Schema(SchemaError::InvalidVectorSetting {
            field: "search.n_probes",
            ..
        })
    ));
    assert!(matches!(
        store.update_vector_search("plain_kv", &update).unwrap_err(),
        DocStoreError::Schema(SchemaError::VectorIndexNotConfigured { .. })
    ));
    assert!(matches!(
        store.update_vector_search("docs", &SearchSpec::default()).unwrap_err(),
        DocStoreError::Schema(SchemaError::InvalidVectorSetting { field: "search", .. })
    ));
    store.shutdown().await.unwrap();
}

#[cfg(feature = "semantic-search")]
mod with_service {
    use super::*;
    use crate::semantic_search::cluster::Cluster;
    use crate::semantic_search::service::SemanticSearchConfig;
    use crate::semantic_search::{ClusterIndex, chunk_document};

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
                    // gemma points along +x, qwen along +y; each payload slightly different.
                    let axis = if parts[2] == "gemma" { 0 } else { 1 };
                    let embs: Vec<Vec<f32>> = (0..n)
                        .map(|i| {
                            let mut v = vec![0.01f32 * (i as f32 + 1.0); dims];
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

    /// Centroids for one model: two clusters with ids starting at `first_id`.
    fn centroids(first_id: u32, axis: usize) -> Arc<ClusterIndex> {
        let mut a = vec![0.0f32; DIM];
        a[axis] = 1.0;
        let mut b = vec![0.0f32; DIM];
        b[(axis + 2) % DIM] = 1.0;
        Arc::new(ClusterIndex::from_clusters(
            [(first_id, Cluster::new(first_id, a)), (first_id + 1, Cluster::new(first_id + 1, b))]
                .into_iter()
                .collect(),
        ))
    }

    fn context(url: &str) -> SemanticSearchContext {
        SemanticSearchContext::new(
            SemanticSearchConfig {
                embedding_service_url: url.to_string(),
                ..SemanticSearchConfig::default()
            },
            [("gemma".to_string(), centroids(1, 0)), ("qwen".to_string(), centroids(101, 1))],
        )
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

    #[tokio::test]
    async fn two_namespaces_with_different_models_each_use_their_own() {
        let (url, seen) = spawn_service();
        let (db_dir, schema_dir) = (TempDir::new().unwrap(), TempDir::new().unwrap());
        let store = open_fresh(db_dir.path(), schema_dir.path()).await.with_semantic_search(context(&url));
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

        // Each namespace's chunks sit in its own model's clusters.
        assert!(sparse_clusters(&store, "g").await.iter().all(|c| (1..=2).contains(c)));
        assert!(sparse_clusters(&store, "q").await.iter().all(|c| (101..=102).contains(c)));

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
        let (db_dir, schema_dir) = (TempDir::new().unwrap(), TempDir::new().unwrap());
        let store = open_fresh(db_dir.path(), schema_dir.path()).await.with_semantic_search(context(&url));
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
                    n_probes: Some(0),
                    ..Default::default()
                },
                "search.n_probes",
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
        let (db_dir, schema_dir) = (TempDir::new().unwrap(), TempDir::new().unwrap());
        let store = open_fresh(db_dir.path(), schema_dir.path()).await.with_semantic_search(context(&url));

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
        assert!(
            matches!(
                err,
                DocStoreError::EmbeddingDimMismatch {
                    dim: 16,
                    centroid_dim: 8,
                    ..
                }
            ),
            "{err:?}"
        );
        assert!(!schema_dir.path().join("e5.json").exists() && !schema_dir.path().join("wide.json").exists());

        store.create(make_schema("later", vec![])).await.unwrap();
        let err = store
            .amend(
                "later",
                SchemaAmendment::EnableVectorIndex {
                    fields: vec!["text".into()],
                    vector_index: Some(spec(r#"{"embedding_model":"e5","embedding_dim":8}"#)),
                },
            )
            .unwrap_err();
        assert!(matches!(err, DocStoreError::UnsupportedEmbeddingModel { .. }), "{err:?}");
        assert!(!store.get_schema("later").unwrap().semantic_search_enabled, "nothing saved");
        store.shutdown_vec_index_worker().await;
        store.shutdown().await.unwrap();
    }
}
