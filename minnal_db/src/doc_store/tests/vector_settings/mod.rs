//! A namespace's vector-index settings through the store: defaults written on
//! create, fixed fields kept across a vector-index drop, search updates, model
//! checks, per-request overrides, and two namespaces with different models in
//! one process (design doc M2a).

#[cfg(feature = "semantic-search")]
mod drop_cleanup;
#[cfg(feature = "semantic-search")]
mod drop_partitions;
#[cfg(feature = "semantic-search")]
mod drop_races;
#[cfg(feature = "semantic-search")]
mod partition;
#[cfg(feature = "semantic-search")]
mod search;
#[cfg(feature = "semantic-search")]
mod support;
#[cfg(feature = "semantic-search")]
mod vector_jobs;

use super::test_support::*;
#[cfg(feature = "semantic-search")]
use crate::doc_store::schema::vector_settings::SearchSpec;
use crate::doc_store::schema::vector_settings::VectorIndexSpec;
use crate::doc_store::*;

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
    store.drop_vector_index("docs").await.unwrap();
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
