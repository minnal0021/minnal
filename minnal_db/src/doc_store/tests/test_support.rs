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

/// The embedding service tests point at unless they bring their own mock: a
/// port nothing listens on, so no test reaches a real service by accident
/// (one usually runs on the default port on a development machine).
#[cfg(feature = "semantic-search")]
pub(crate) const NO_EMBEDDING_SERVICE: &str = "http://127.0.0.1:9";

/// A builder for the store at `db_dir` with the test settings: the test
/// engine config, a lifecycle that runs again after 5 ms, and (with semantic
/// search) [`NO_EMBEDDING_SERVICE`].
pub(crate) fn test_builder(db_dir: &Path) -> DocStoreBuilder {
    let builder = DocStore::builder(db_dir)
        .db_config(crate::support::test_db_config())
        .lifecycle(LifecycleConfig {
            max_restarts: 3,
            restart_backoff: std::time::Duration::from_millis(5),
            ..LifecycleConfig::default()
        });
    #[cfg(feature = "semantic-search")]
    let builder = builder.semantic_search(SemanticSearchContext::new(crate::semantic_search::service::SemanticSearchConfig {
        embedding_service_url: NO_EMBEDDING_SERVICE.to_owned(),
        ..Default::default()
    }));
    builder
}

/// Open the store at `db_dir`, as a restart would, with the test settings.
pub(crate) async fn open_fresh(db_dir: &Path) -> Arc<DocStore> {
    open_with(db_dir, |b| b).await
}

/// Open the store at `db_dir` with the test settings changed by `configure`.
///
/// A test that simulates a kill drops its store without `shutdown` and opens
/// it again. A background worker's pass can still hold the old database, and
/// with it the directory lock, for a while after the drop (on a real disk its
/// fsyncs make that long enough to hit). A restart waits for the old process
/// to exit; this waits, up to a bound, for the old instance in this process to
/// let go. A lock held by another process still fails at once.
pub(crate) async fn open_with(db_dir: &Path, configure: impl Fn(DocStoreBuilder) -> DocStoreBuilder) -> Arc<DocStore> {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    loop {
        match configure(test_builder(db_dir)).open().await {
            Err(DocStoreError::StoreLocked { owner_pid, .. }) if owner_pid == Some(std::process::id()) && std::time::Instant::now() < deadline => {
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
            opened => return opened.unwrap(),
        }
    }
}

/// Wait up to 30 s until no job of `namespace` is live: the governor has
/// finished (or given up on) every job it resumed or was handed.
pub(crate) async fn wait_for_jobs(store: &DocStore, namespace: &str) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        let live: Vec<String> = store
            .jobs(namespace)
            .unwrap()
            .into_iter()
            .filter(|(_, r)| r.is_live())
            .map(|(n, _)| n)
            .collect();
        if live.is_empty() {
            return;
        }
        assert!(std::time::Instant::now() < deadline, "jobs of '{namespace}' still live: {live:?}");
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
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

/// Open the store at `db_dir` with a worker notifier but **no vector
/// worker**, so the write and delete paths do their vector-queue work exactly
/// as they do beside a running worker, while a test drives the worker's steps
/// by hand (`vector_kv::finish_embed`, `vector_kv::process_clear`) at chosen
/// points.
#[cfg(feature = "semantic-search")]
pub(crate) async fn open_without_vector_worker(db_dir: &Path) -> Arc<DocStore> {
    open_with(db_dir, DocStoreBuilder::without_vector_worker).await
}
