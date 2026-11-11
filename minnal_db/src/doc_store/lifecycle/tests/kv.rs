//! The KV store lifecycle.

use crate::doc_store::tests::test_support::*;
use crate::doc_store::*;
use crate::namespace::NamespaceId;

// ── KV lifecycle ────────────────────────────────────────────────────────

#[tokio::test]
async fn test_kv_create_and_list() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;

    store.create_kv(make_kv_schema("cache", KvKeyType::Str, KvValueType::Str)).await.unwrap();

    let list = store.list_kv().unwrap();
    assert_eq!(list.len(), 1);
    assert_eq!(list[0]["namespace"], "cache");
    assert!(list[0]["ns_id"].as_str().is_some_and(|s| s.parse::<NamespaceId>().is_ok()));
    assert_eq!(list[0]["key_type"], "str");
    assert_eq!(list[0]["value_type"], "str");
}

#[tokio::test]
async fn test_kv_get_schema_round_trip() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;

    store.create_kv(make_kv_schema("cache", KvKeyType::Int, KvValueType::F32)).await.unwrap();

    let schema = store.get_kv_schema("cache").unwrap();
    assert_eq!(schema.namespace, "cache");
    assert_eq!(schema.key_type, KvKeyType::Int);
    assert_eq!(schema.value_type, KvValueType::F32);
    // ns_id is assigned at creation and must survive the round-trip so an
    // exported schema reflects the persisted store.
    assert!(schema.ns_id.is_some());

    let err = store.get_kv_schema("missing").unwrap_err();
    assert!(matches!(err, DocStoreError::NotFound { .. }));
}

#[tokio::test]
async fn test_kv_get_schema_rejects_doc_store() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;

    store.create(make_schema("docs", vec![])).await.unwrap();

    // A doc-store namespace must not be readable as a KV schema (key_type
    // discriminant differs), guarding the export endpoint against mixing types.
    assert!(store.get_kv_schema("docs").is_err());
}

#[tokio::test]
async fn test_kv_create_duplicate_rejected() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;

    store.create_kv(make_kv_schema("kv", KvKeyType::Str, KvValueType::Int)).await.unwrap();
    let err = store.create_kv(make_kv_schema("kv", KvKeyType::Str, KvValueType::Int)).await.unwrap_err();
    assert!(matches!(err, DocStoreError::AlreadyExists { .. }));
}

#[tokio::test]
async fn test_kv_drop_removes_schema() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;

    store.create_kv(make_kv_schema("tmp", KvKeyType::Str, KvValueType::Str)).await.unwrap();
    assert!(store.schemas.contains("tmp"));

    store.remove_kv("tmp").await.unwrap();
    assert!(!store.schemas.contains("tmp"));
}

#[tokio::test]
async fn test_kv_list_does_not_include_doc_stores() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;

    store.create(make_schema("docs", vec![])).await.unwrap();
    store.create_kv(make_kv_schema("cache", KvKeyType::Str, KvValueType::Str)).await.unwrap();

    let kv_list = store.list_kv().unwrap();
    assert_eq!(kv_list.len(), 1, "list_kv should only return KV stores");
    assert_eq!(kv_list[0]["namespace"], "cache");

    let doc_list = store.list().unwrap();
    assert_eq!(doc_list.len(), 1, "list should only return doc stores");
    assert_eq!(doc_list[0]["namespace"], "docs");
}
