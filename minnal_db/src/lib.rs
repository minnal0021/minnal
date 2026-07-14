//! # MinnalDB
//!
//! An embedded Minnal-style key-value store for Rust applications.
//!
//! MinnalDB separates keys (stored in an LSM tree) from values (stored in a
//! value log), reducing write amplification for large values while keeping
//! lookups efficient.
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use minnal_db::Db;
//!
//! fn main() -> Result<(), minnal_db::KVError> {
//!     let db = Db::open("/tmp/my_db")?;
//!
//!     db.put(b"hello", b"world")?;
//!
//!     if let Some(val) = db.get(b"hello")? {
//!         println!("{}", String::from_utf8_lossy(&val));
//!     }
//!
//!     db.shutdown()?;
//!     Ok(())
//! }
//! ```
//!
//! ## Async usage
//!
//! ```rust,no_run
//! use minnal_db::AsyncDb;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), minnal_db::KVError> {
//!     let db = AsyncDb::open("/tmp/my_async_db").await?;
//!
//!     db.put(b"hello".to_vec(), b"world".to_vec()).await?;
//!
//!     if let Some(val) = db.get(b"hello".to_vec()).await? {
//!         println!("{}", String::from_utf8_lossy(&val));
//!     }
//!
//!     db.shutdown().await?;
//!     Ok(())
//! }
//! ```

// minnal_db's storage engine relies on Unix positional I/O (`pread`/`pwrite`,
// via `std::os::unix::fs::FileExt`) and the server on POSIX signals. Windows is
// unsupported; fail early with a clear message rather than a cryptic
// `unresolved import std::os::unix` further down.
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
compile_error!("minnal_db supports only Linux and macOS (its storage engine uses Unix pread/pwrite)");

pub mod db;
mod store;
mod support;

// ── Folded layers ─────────────────────────────────────────────────────────────
// `index` is part of the base engine (field indexing is core). `semantic_search`
// and `doc_store` are opt-in via cargo features.
#[cfg(feature = "doc-store")]
pub mod doc_store;
pub mod index;
#[cfg(feature = "semantic-search")]
pub mod semantic_search;
// Vector-index storage bridge (`DbVectorStore`, upsert/delete, query-embedding
// cache) over raw namespaces — usable with `kv-store` alone, no `doc-store`.
#[cfg(feature = "semantic-search")]
pub mod vector_kv;

// ── Facade API (primary entry points) ─────────────────────────────────────────

/// The primary synchronous entry point.
///
/// ```rust,no_run
/// use minnal_db::Db;
/// let db = Db::open("/tmp/db").unwrap();
/// db.put(b"k", b"v").unwrap();
/// db.shutdown().unwrap();
/// ```
pub use db::facade::Db;

/// Async wrapper — offloads I/O via `spawn_blocking`.
pub use db::facade::AsyncDb;

/// Scoped handle to a single namespace (sync).
pub use db::facade::Namespace;

/// Scoped handle to a single namespace (async).
pub use db::facade::AsyncNamespace;

// ── Configuration ─────────────────────────────────────────────────────────────

/// Top-level database configuration.
pub use db::config::DbConfig;

/// Controls when the value-log GC is triggered.
pub use db::config::ThresholdConfig;

/// Default field-index bitmap compaction threshold (percentage of dead space).
pub use db::config::{DEFAULT_INDEX_BLOB_WASTE_THRESHOLD, DEFAULT_PAGE_GC_THRESHOLD};
pub use store::value_log::DEFAULT_SEGMENT_SIZE_BYTES;

/// Intervals at which the background workers run.
pub use db::config::ScheduledTaskConfig;

/// Controls how many writes to batch before syncing.
pub use db::config::SyncConfig;

/// TOML configuration file parser.
pub use db::toml_config::MinnalTomlConfig;

// ── Types ─────────────────────────────────────────────────────────────────────

/// The unified error type returned by all operations.
pub use db::error::KVError;

// ── Index types ───────────────────────────────────────────────────────────────

/// Discriminant used when registering a field index.
pub use crate::index::IndexValueType;

/// A typed field value extracted from a document by an extractor closure.
pub use crate::index::IndexValue;

/// On-disk blob growth/waste metrics for one field index (logical vs. live
/// bytes + waste ratios), returned by [`Db::field_index_blob_stats`].
pub use crate::index::IndexBlobStats;

/// Unique identifier for a registered field within a namespace.
pub use db::namespace::FieldId;
/// Metadata for a registered field: id, name, and value type.
pub use db::namespace::FieldMeta;
/// Outcome of a targeted single-field reindex ([`Db::reindex_field`]).
pub use db::namespace::FieldReindexOutcome;

/// Extractor closure type: maps raw document bytes to an [`IndexValue`].
pub use db::namespace_index::ExtractorFn;

/// Row-ID function type: derives a stable 128-bit row ID from raw key bytes.
///
/// Register one via [`Db::set_row_id_fn`] to replace the default dense row-ID
/// map with an explicit identifier (e.g. a UUID embedded in the key).
pub use db::namespace_index::RowIdFn;

/// Inverse of [`RowIdFn`]: reconstructs the raw key bytes from a row ID.
///
/// Registering this alongside [`RowIdFn`] enables O(|hits|) query resolution
/// in [`Db::query_index`] with zero memory overhead — no in-memory map needed.
pub use db::namespace_index::RowToKeyFn;

/// Namespace ID for the implicit default namespace.
pub use db::namespace::DEFAULT_NAMESPACE_ID;

/// Name and ID for the system-wide namespace (always present, never removable).
pub use db::namespace::{SYSTEM_NAMESPACE, SYSTEM_NAMESPACE_ID};

/// Snapshot statistics returned by [`Db::stats`].
pub use db::stats::Stats;

/// Per-GC-run statistics returned by [`Db::garbage_collect`].
pub use db::stats::GCStats;

/// Engine-wide operational metrics (runtime counters) and their serializable
/// snapshot, returned by [`Db::ops_metrics`].
pub use db::metrics::{Metrics, MetricsSnapshot};

/// Snapshot of WAL state returned by [`Db::wal_metadata`].
pub use db::wal::WalMetadata;

/// Live LSM manifest snapshot returned by [`Db::lsm_manifests`].
pub use store::lsm::lsm_manifest::LsmManifest;

/// Per-bucket entry inside a [`ManifestLevel`].
pub use store::lsm::lsm_manifest::ManifestBucket;
/// Per-file entry inside a [`ManifestBucket`].
pub use store::lsm::lsm_manifest::ManifestFile;
/// Per-level entry in an [`LsmManifest`].
pub use store::lsm::lsm_manifest::ManifestLevel;
/// In-memory (non-SSTable) LSM statistics, returned by [`Db::lsm_runtime_stats`].
pub use store::lsm::lsm_tree::LSMStats;

/// Per-bucket value-log metadata returned by [`Db::value_log_shard_stats`].
pub use store::value_log::ValueLogMetadata;

/// Physical (on-disk) value-log shard footprint, returned by
/// [`Db::value_log_physical_stats`].
pub use store::value_log::sharded::ShardPhysicalStats;

/// Per-page value-log garbage breakdown, returned by [`Db::value_log_page_stats`].
pub use store::value_log::SegmentStats;

// ── Document store (folded layer) ─────────────────────────────────────────────
//
// The full `doc_store` and `semantic_search` module trees are public modules
// above (`minnal_db::doc_store::…`, `minnal_db::semantic_search::…`). These
// re-export the most-used document-store types at the crate root for ergonomics.
// Engine diagnostic types (`KVError`, `Stats`, manifests, …) are intentionally
// NOT re-listed here — they already live at the crate root.
#[cfg(feature = "doc-store")]
pub use doc_store::{
    AttributeDef, AttributeType, CursorPage, DiskBuildProgress, DocId, DocStore, DocStoreError, DocStoreSchema, IndexBuildHandle, IndexBuildManager,
    IndexBuildProgress, IndexKind, IndexSpec, IndexType, KeyType, KvKeyType, KvStoreSchema, KvValueType, MAX_INDICES, Page, Pagination,
    SchemaAmendment, SchemaError, StoreType, prefix_upper_bound,
};

// Document-store types that only exist alongside `semantic-search`.
#[cfg(all(feature = "doc-store", feature = "semantic-search"))]
pub use doc_store::{QueueEntry, ReindexStats, SemanticSearchContext, VecReindexProgress, VectorIndexConfig, VectorReindexOutcome};

// ── rkyv re-exports (for typed API) ──────────────────────────────────────────

/// Re-exported rkyv derive macros so users can `#[derive(Archive, Serialize, Deserialize)]`
/// without adding rkyv as a direct dependency.
pub mod rkyv_derives {
    pub use rkyv::{Archive, Deserialize, Serialize};
}

/// Zero-copy access to an archived value, re-exported from rkyv.
///
/// Useful inside a field-index [`ExtractorFn`] over values written with
/// [`Db::put_typed`]: validate-and-borrow the archived struct, then pull out
/// the indexed field — no full deserialisation. Pair with [`rancor`] for the
/// error type, e.g. `access::<ArchivedT, rancor::Error>(bytes)`.
pub use rkyv::access;

/// rkyv's error-handling module, re-exported for use with [`access`].
pub use rkyv::rancor;

/// The archived form of a type `T`, re-exported from rkyv.
///
/// Spells the target type for [`access`] when no named `Archived*` struct is
/// generated — e.g. `access::<Archived<u64>, rancor::Error>(key_bytes)` to read
/// back a `u64` key written with [`Db::put_typed`].
pub use rkyv::Archived;

// ── LSM configuration ────────────────────────────────────────────────────────

/// LSM-tree configuration (compaction threshold, thread count, data dir).
pub mod lsm {
    pub use crate::store::lsm::lsm_tree::LSMConfig;
}
