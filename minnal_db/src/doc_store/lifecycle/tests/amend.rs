//! Amending a store's attributes.

use crate::doc_store::tests::test_support::*;
use crate::doc_store::*;

// ── Schema amendment ────────────────────────────────────────────────────

#[tokio::test]
async fn test_amend_add_and_remove_attribute() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;
    store.create(make_schema("ns", vec![])).await.unwrap();

    store
        .amend(
            "ns",
            SchemaAmendment::AddAttribute {
                name: "email".to_owned(),
                attr_type: AttributeType::Str,
                description: None,
            },
        )
        .await
        .unwrap();

    let loaded = store.load_schema("ns").unwrap();
    assert_eq!(loaded.attributes.len(), 1);
    assert_eq!(loaded.attributes[0].name, "email");

    store
        .amend("ns", SchemaAmendment::RemoveAttribute { name: "email".to_owned() })
        .await
        .unwrap();
    let loaded2 = store.load_schema("ns").unwrap();
    assert!(loaded2.attributes.is_empty());
}

#[tokio::test]
async fn test_amend_cannot_remove_indexed_attribute() {
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

    let err = store
        .amend("ns", SchemaAmendment::RemoveAttribute { name: "status".to_owned() })
        .await
        .unwrap_err();
    assert!(
        matches!(err, DocStoreError::AttributeIsIndexed { .. }),
        "expected AttributeIsIndexed, got {:?}",
        err
    );
}
