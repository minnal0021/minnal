//! Shared fixtures for the `store` module's unit tests.
//!
//! The per-module `mod tests` blocks all need the same schema builder and store
//! opener, so they live here rather than being duplicated seven times.

use crate::doc_store::*;

pub(crate) use crate::doc_store::schema::{AttributeType, IndexSpec, IndexType, KeyType, KvKeyType, KvValueType};
pub(crate) use tempfile::TempDir;

pub(crate) fn make_schema(namespace: &str, indices: Vec<IndexSpec>) -> DocStoreSchema {
    DocStoreSchema {
        store_type: StoreType::Doc,
        namespace: namespace.to_owned(),
        ns_id: None,
        key_type: KeyType::U64,
        attributes: vec![],
        indices,
        semantic_search_enabled: false,
        embedding_fields: vec![],
        vector_index: None,
    }
}

/// Open the store at `db_dir`, as a restart would.
///
/// A test that simulates a kill drops its store without `shutdown` and opens
/// it again. A background worker's pass can still hold the old database, and
/// with it the directory lock, for a while after the drop (on a real disk its
/// fsyncs make that long enough to hit). A restart waits for the old process
/// to exit; this waits, up to a bound, for the old instance in this process to
/// let go. A lock held by another process still fails at once.
pub(crate) async fn open_fresh(db_dir: &Path) -> DocStore {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    loop {
        match DocStore::open_with_config(db_dir, crate::support::test_db_config()).await {
            Err(DocStoreError::StoreLocked { owner_pid, .. }) if owner_pid == Some(std::process::id()) && std::time::Instant::now() < deadline => {
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
            opened => return opened.unwrap(),
        }
    }
}

pub(crate) fn make_kv_schema(namespace: &str, key_type: KvKeyType, value_type: KvValueType) -> KvStoreSchema {
    use crate::doc_store::schema::KvStoreSchema;
    KvStoreSchema {
        store_type: StoreType::Kv,
        namespace: namespace.to_owned(),
        ns_id: None,
        key_type,
        value_type,
        semantic_search_enabled: false,
        vector_index: None,
    }
}

/// Give `store` a worker notifier **without starting a worker**, so the write and
/// delete paths do their vector-queue work exactly as they do beside a running
/// worker, while a test drives the worker's steps by hand
/// (`vector_kv::finish_embed`, `vector_kv::process_clear`) at chosen points.
#[cfg(feature = "semantic-search")]
pub(crate) fn with_worker_notify(mut store: DocStore) -> DocStore {
    store.notify = Some(Arc::new(tokio::sync::Notify::new()));
    store
}
