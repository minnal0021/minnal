pub mod error;
pub mod hex;
pub mod index_manager;
pub mod index_observer;
pub mod index_progress;
pub mod pagination;
pub mod schema;
pub mod store;
pub(crate) mod vec_index_worker;
pub(crate) mod vector_kv;

// ── Error types ────────────────────────────────────────────────────────────
pub use error::{DocStoreError, SchemaError};

// ── Schema types ───────────────────────────────────────────────────────────
pub use schema::{
    AttributeDef, AttributeType, DocStoreSchema, IndexKind, IndexSpec, IndexType, KeyType, KvKeyType, KvStoreSchema, KvValueType, MAX_INDICES,
    SchemaAmendment, StoreType,
};

// ── Pagination types ───────────────────────────────────────────────────────
pub use pagination::{CursorPage, Page, Pagination, prefix_upper_bound};

// ── Store types ────────────────────────────────────────────────────────────
pub use store::{
    DiskBuildProgress, DocId, DocStore, IndexBuildHandle, IndexBuildProgress, ReindexStats, SemanticSearchContext, VecReindexProgress,
    VectorReindexOutcome,
};

// ── Index manager ──────────────────────────────────────────────────────────
pub use index_manager::IndexBuildManager;

// ── Vector-index worker types ──────────────────────────────────────────────
pub use vec_index_worker::VectorIndexConfig;
pub use vector_kv::QueueEntry;

// ── Engine diagnostic types (re-exported for API consumers) ───────────────
pub use minnal_db::{GCStats, KVError, LsmManifest, ManifestBucket, ManifestFile, ManifestLevel, Stats, ValueLogMetadata, WalMetadata};

/// Test-only `DbConfig` with a deliberately small bucket count.
///
/// Every namespace eagerly opens `2 × num_buckets` file descriptors (one LSM L1
/// file plus one value-log file per bucket), held for its lifetime. With the
/// production default of 16 buckets a namespace-heavy test suite run at high
/// `cargo test` parallelism (one test per core) exhausts the typical 1024 fd
/// soft limit on many-core machines ("Too many open files"). Two buckets cuts
/// the per-namespace fd cost ~8× while still exercising multi-bucket routing.
#[cfg(test)]
pub(crate) fn test_db_config() -> minnal_db::DbConfig {
    minnal_db::DbConfig {
        num_buckets: 2,
        ..Default::default()
    }
}
