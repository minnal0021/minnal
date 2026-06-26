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
    SchemaAmendment,
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
