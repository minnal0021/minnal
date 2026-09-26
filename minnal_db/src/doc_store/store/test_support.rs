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

/// Give `store` a worker notifier **without starting a worker**, so the write and
/// delete paths do their vector-queue work exactly as they do beside a running
/// worker, while a test drives the worker's steps by hand
/// (`vector_kv::finish_embed`, `vector_kv::process_clear`) at chosen points.
#[cfg(feature = "semantic-search")]
pub(super) fn with_worker_notify(mut store: DocStore) -> DocStore {
    store.notify = Some(Arc::new(tokio::sync::Notify::new()));
    store
}
