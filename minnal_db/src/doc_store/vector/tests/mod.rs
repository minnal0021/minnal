//! Tests of the vector-index administration, one file per topic, and the
//! helpers they share. The queue races through the store's write paths (a
//! delete or an emptied embedding field during an embed) are in `reindex` and
//! `writes`.

#[cfg(feature = "semantic-search")]
use super::*;
use crate::doc_store::tests::test_support::*;
#[cfg(not(feature = "semantic-search"))]
use crate::doc_store::*;

#[cfg(feature = "semantic-search")]
mod drop;
mod queue;
#[cfg(feature = "semantic-search")]
mod reindex;
#[cfg(feature = "semantic-search")]
mod writes;

/// Helper: create a semantic-search-enabled doc schema with a `title` field.
#[cfg(feature = "semantic-search")]
async fn create_semantic_schema(store: &DocStore, namespace: &str) {
    let schema = DocStoreSchema {
        store_type: StoreType::Doc,
        namespace: namespace.to_owned(),
        ns_id: None,
        key_type: KeyType::U64,
        attributes: vec![crate::doc_store::schema::AttributeDef {
            name: "title".to_owned(),
            attr_type: AttributeType::Str,
            description: None,
        }],
        indices: vec![],
        semantic_search_enabled: true,
        embedding_fields: vec!["title".to_owned()],
        vector_index: None,
    };
    store.create(schema).await.unwrap();
}

#[cfg(feature = "semantic-search")]
fn race_vectors() -> Vec<crate::semantic_search::VectorIndex> {
    use crate::semantic_search::QuantisationStyle;
    vec![
        crate::semantic_search::VectorIndex::new(1, QuantisationStyle::MultiBit { number_of_bits: 8 }, 0.5, 0.0, 0.01, vec![]),
        crate::semantic_search::VectorIndex::new(2, QuantisationStyle::SingleBit, 0.5, 0.0, 0.01, vec![]),
    ]
}

#[cfg(feature = "semantic-search")]
async fn open_kv_semantic(db_dir: &TempDir) -> Arc<DocStore> {
    let store = open_without_vector_worker(db_dir.path()).await;
    let mut schema = make_kv_schema("sem_kv", KvKeyType::Str, KvValueType::Str);
    schema.semantic_search_enabled = true;
    store.create_kv(schema).await.unwrap();
    store
}

/// The dense entry of `key` in `sem_kv`, if any.
#[cfg(feature = "semantic-search")]
async fn dense_entry(store: &DocStore, key: &[u8]) -> Option<Vec<u8>> {
    let dense = store.db.namespace(vector_kv::dense_vectors_ns("sem_kv")).await.unwrap();
    dense.get(key.to_vec()).await.unwrap()
}
