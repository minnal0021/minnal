//! Exporting a schema and creating a store from one.

use crate::doc_store::tests::test_support::*;
use crate::doc_store::*;
use crate::namespace::NamespaceId;

// ── get_schema (export) ────────────────────────────────────────────────────

#[tokio::test]
async fn test_get_schema_round_trips_through_json() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;

    let original = make_schema("export_test", vec![]);
    store.create(original.clone()).await.unwrap();

    let retrieved = store.get_schema("export_test").unwrap();
    let json = serde_json::to_string(&retrieved).unwrap();
    let re_parsed: DocStoreSchema = serde_json::from_str(&json).unwrap();

    assert_eq!(re_parsed.namespace, "export_test");
    assert_eq!(re_parsed.key_type, original.key_type);
    assert_eq!(re_parsed.semantic_search_enabled, original.semantic_search_enabled);
}

#[tokio::test]
async fn test_get_schema_unknown_namespace_is_not_found() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;

    let err = store.get_schema("no_such_ns").unwrap_err();
    assert!(matches!(err, DocStoreError::NotFound { .. }));
}

// ── import (create via schema JSON) ────────────────────────────────────────

#[tokio::test]
async fn test_import_schema_creates_usable_store() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;

    let schema: DocStoreSchema = serde_json::from_str(
        r#"{
        "namespace": "imported",
        "store_type": "doc",
        "key_type": "u64",
        "attributes": [],
        "indices": [],
        "semantic_search_enabled": false,
        "embedding_fields": []
    }"#,
    )
    .unwrap();

    store.create(schema).await.unwrap();

    let list = store.list().unwrap();
    assert!(list.iter().any(|v| v["namespace"] == "imported"));

    store.put("imported", DocId::U64(1), serde_json::json!({"x": 1})).await.unwrap();
    assert_eq!(store.count_docs("imported").await.unwrap(), 1);
}

/// Any `ns_id` present in an exported schema must be discarded so the store
/// assigns a fresh ID rather than colliding with an existing namespace.
#[tokio::test]
async fn test_import_schema_stale_ns_id_is_replaced() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;

    // Schema carries a stale ns_id from a previous database instance.
    let mut schema = make_schema("ns_id_test", vec![]);
    schema.ns_id = Some(NamespaceId::from_u64(99999));

    store.create(schema).await.unwrap();

    let stored = store.get_schema("ns_id_test").unwrap();
    // The stored ns_id must be the one assigned by this store, not the
    // stale value that came in with the import body.
    assert_ne!(stored.ns_id, Some(NamespaceId::from_u64(99999)));
    assert!(stored.ns_id.is_some(), "ns_id must be assigned after create");
}

#[tokio::test]
async fn test_import_schema_duplicate_is_rejected() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;

    store.create(make_schema("dup_import", vec![])).await.unwrap();

    let err = store.create(make_schema("dup_import", vec![])).await.unwrap_err();
    assert!(matches!(err, DocStoreError::AlreadyExists { .. }));
}

#[tokio::test]
async fn test_import_schema_invalid_namespace_rejected() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;

    for bad_ns in &["", "has space", "has/slash", "has.dot"] {
        let mut schema = make_schema("placeholder", vec![]);
        schema.namespace = bad_ns.to_string();
        let err = store.create(schema).await.unwrap_err();
        assert!(
            matches!(err, DocStoreError::Schema(crate::doc_store::error::SchemaError::InvalidNamespace)),
            "expected InvalidNamespace for '{bad_ns}', got {err:?}"
        );
    }
}

#[tokio::test]
async fn test_import_schema_too_many_indices_rejected() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;

    let indices = (0..=crate::doc_store::schema::MAX_INDICES)
        .map(|i| IndexSpec {
            field: format!("f{i}"),
            index_type: IndexType::Int,
        })
        .collect();
    let err = store.create(make_schema("too_many", indices)).await.unwrap_err();
    assert!(matches!(
        err,
        DocStoreError::Schema(crate::doc_store::error::SchemaError::TooManyIndices { .. })
    ));
}

#[tokio::test]
async fn test_import_schema_duplicate_index_field_rejected() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;

    let indices = vec![
        IndexSpec {
            field: "status".to_owned(),
            index_type: IndexType::Str,
        },
        IndexSpec {
            field: "status".to_owned(),
            index_type: IndexType::Str,
        },
    ];
    let err = store.create(make_schema("dup_field", indices)).await.unwrap_err();
    assert!(matches!(
        err,
        DocStoreError::Schema(crate::doc_store::error::SchemaError::DuplicateFieldName { .. })
    ));
}

/// A field that was first declared as an attribute and later
/// indexed ends up in both `attributes` and `indices` in the saved schema.
/// `add_index` must move it out of `attributes`; importing such a schema must
/// also survive without a DuplicateFieldName error.
#[tokio::test]
async fn test_add_index_removes_field_from_attributes() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;

    // Create with "agency" as a non-indexed attribute.
    let mut schema = make_schema("agency_ns", vec![]);
    schema.attributes.push(crate::doc_store::schema::AttributeDef {
        name: "agency".to_owned(),
        attr_type: crate::doc_store::schema::AttributeType::Str,
        description: None,
    });
    store.create(schema).await.unwrap();

    // Add an index on "agency" — this must remove it from attributes.
    store
        .add_index(
            "agency_ns",
            IndexSpec {
                field: "agency".to_owned(),
                index_type: IndexType::Str,
            },
        )
        .await
        .unwrap();

    let stored = store.get_schema("agency_ns").unwrap();
    assert!(
        stored.attributes.iter().all(|a| a.name != "agency"),
        "agency must not remain in attributes after indexing"
    );
    assert!(stored.indices.iter().any(|i| i.field == "agency"), "agency must be in indices");
}

/// Importing an existing schema (with a new name) where a field
/// appears in both `attributes` and `indices` must not fail with DuplicateFieldName.
/// The import handler normalises the schema by dropping indexed fields from
/// attributes before calling create.
#[tokio::test]
async fn test_create_with_field_in_both_attributes_and_indices_is_rejected() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;

    // Simulate a stale exported schema that has "agency" in both lists.
    let schema = DocStoreSchema {
        store_type: StoreType::Doc,
        namespace: "stale_export".to_owned(),
        ns_id: None,
        key_type: crate::doc_store::schema::KeyType::U64,
        attributes: vec![crate::doc_store::schema::AttributeDef {
            name: "agency".to_owned(),
            attr_type: crate::doc_store::schema::AttributeType::Str,
            description: None,
        }],
        indices: vec![IndexSpec {
            field: "agency".to_owned(),
            index_type: IndexType::Str,
        }],
        semantic_search_enabled: false,
        embedding_fields: vec![],
        vector_index: None,
    };

    // Direct create (without normalisation) must be rejected by validate().
    let err = store.create(schema).await.unwrap_err();
    assert!(matches!(
        err,
        DocStoreError::Schema(crate::doc_store::error::SchemaError::DuplicateFieldName { .. })
    ));
}
