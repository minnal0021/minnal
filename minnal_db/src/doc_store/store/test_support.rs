//! Shared fixtures for the `store` module's unit tests.
//!
//! The per-module `mod tests` blocks all need the same schema builder and store
//! opener, so they live here rather than being duplicated seven times.

use super::*;

pub(super) use crate::doc_store::schema::{AttributeType, IndexSpec, IndexType, KeyType, KvKeyType, KvValueType};
pub(super) use tempfile::TempDir;

pub(super) fn make_schema(namespace: &str, indices: Vec<IndexSpec>) -> DocStoreSchema {
    DocStoreSchema {
        store_type: StoreType::Doc,
        namespace: namespace.to_owned(),
        ns_id: None,
        key_type: KeyType::U64,
        attributes: vec![],
        indices,
        semantic_search_enabled: false,
        embedding_fields: vec![],
    }
}

pub(super) async fn open_fresh(db_dir: &Path, schema_dir: &Path) -> DocStore {
    DocStore::open_with_config(db_dir, schema_dir, crate::doc_store::test_db_config())
        .await
        .unwrap()
}

pub(super) fn make_kv_schema(namespace: &str, key_type: KvKeyType, value_type: KvValueType) -> KvStoreSchema {
    use crate::doc_store::schema::KvStoreSchema;
    KvStoreSchema {
        store_type: StoreType::Kv,
        namespace: namespace.to_owned(),
        ns_id: None,
        key_type,
        value_type,
        semantic_search_enabled: false,
    }
}
