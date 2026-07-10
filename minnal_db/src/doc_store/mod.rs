pub mod error;
pub mod hex;
pub mod index_manager;
pub mod index_observer;
pub mod index_progress;
pub mod pagination;
pub mod schema;
pub mod store;
#[cfg(feature = "semantic-search")]
pub(crate) mod vec_index_worker;

// ── Error types ────────────────────────────────────────────────────────────
pub use self::error::{DocStoreError, SchemaError};

// ── Schema types ───────────────────────────────────────────────────────────
pub use self::schema::{
    AttributeDef, AttributeType, DocStoreSchema, IndexKind, IndexSpec, IndexType, KeyType, KvKeyType, KvStoreSchema, KvValueType, MAX_INDICES,
    SchemaAmendment, StoreType,
};

// ── Pagination types ───────────────────────────────────────────────────────
pub use self::pagination::{CursorPage, Page, Pagination, prefix_upper_bound};

// ── Store types ────────────────────────────────────────────────────────────
pub use self::store::{DiskBuildProgress, DocId, DocStore, IndexBuildHandle, IndexBuildProgress};
#[cfg(feature = "semantic-search")]
pub use self::store::{ReindexStats, SemanticSearchContext, VecReindexProgress, VectorReindexOutcome};

// ── Index manager ──────────────────────────────────────────────────────────
pub use self::index_manager::IndexBuildManager;

// ── Vector-index worker types (semantic-search only) ───────────────────────
#[cfg(feature = "semantic-search")]
pub use self::vec_index_worker::VectorIndexConfig;
#[cfg(feature = "semantic-search")]
pub use crate::vector_kv::QueueEntry;

// ── Engine diagnostic types (re-exported for API consumers) ───────────────
pub use crate::{GCStats, KVError, LsmManifest, ManifestBucket, ManifestFile, ManifestLevel, Stats, ValueLogMetadata, WalMetadata};

/// Test-only `DbConfig` with a deliberately small bucket count.
///
/// Every namespace eagerly opens `2 × num_buckets` file descriptors (one LSM L1
/// file plus one value-log file per bucket), held for its lifetime. With the
/// production default of 16 buckets a namespace-heavy test suite run at high
/// `cargo test` parallelism (one test per core) exhausts the typical 1024 fd
/// soft limit on many-core machines ("Too many open files"). Two buckets cuts
/// the per-namespace fd cost ~8× while still exercising multi-bucket routing.
#[cfg(test)]
pub(crate) fn test_db_config() -> crate::DbConfig {
    crate::DbConfig {
        num_buckets: 2,
        ..Default::default()
    }
}
