//! The engine's background workers. `db` starts each one and implements its
//! `*Target` trait; each is a single task that fans out over namespaces.
//!
//! Every background worker in the crate:
//!
//! | Worker | What it does | Lives in |
//! |---|---|---|
//! | [`lsm_compaction`] | Merges a namespace's SSTables | here |
//! | [`value_log_gc`] | Rewrites value-log segments to reclaim dead space | here |
//! | [`wal_gc`] | Deletes WAL segments nothing needs any more | here |
//! | [`ttl`] | Removes expired entries from TTL namespaces | here |
//! | [`index_checkpoint`] | Spills field-index write buffers and records their WAL offset | here |
//! | `vec_index` | Embeds documents from the vector-index queue | `semantic_search::workers` |
//! | `ivf_maintenance` | Splits, merges and reassigns a namespace's postings | `semantic_search::workers` |
//!
//! The last two are feature-gated and started by `DocStore`, and reach its schemas
//! and in-flight write tracking through `semantic_search::workers::WorkerHost`. `db`
//! must not import `doc_store` or `semantic_search`.

pub(crate) mod index_checkpoint;
pub(crate) mod lsm_compaction;
pub(crate) mod ttl;
pub(crate) mod value_log_gc;
pub(crate) mod wal_gc;
