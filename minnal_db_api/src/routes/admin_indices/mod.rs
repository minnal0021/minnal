//! Index management admin endpoints — `/admin/indices/*`
//!
//! ```text
//! GET    /admin/indices/progress                          → all active index builds (field + vector)
//! GET    /admin/indices/vector/queue/summary              → global queue depth / lag
//! GET    /admin/indices/vector/queue/retried              → globally retried entries (all namespaces)
//! DELETE /admin/indices/vector/query-cache                → clear the system-wide query embedding cache
//! GET    /admin/indices/vector/corruption-metrics         → corrupt-skip counts per namespace (all namespaces)
//! GET    /admin/indices/{ns}/progress                     → per-namespace index progress
//! GET    /admin/indices/{ns}/vector/corruption-metrics    → corrupt-skip counts for one namespace
//! POST   /admin/indices/{ns}/attribute/reindex-all        → drop + rebuild all field indices (202)
//! DELETE /admin/indices/{ns}/attribute/drop-all           → drop all field indices (202)
//! POST   /admin/indices/{ns}/attribute/{field}/reindex/{doc_id} → reindex one doc in one field index (200)
//! GET    /admin/indices/{ns}/{field}/blob-stats           → one field's on-disk blob growth/waste (404 if not active)
//! POST   /admin/indices/{ns}/vector/reindex-all           → re-enqueue all docs for embedding (202)
//! POST   /admin/indices/{ns}/vector/reindex/{doc_id}      → re-enqueue one doc for embedding (doc + kv) (200)
//! POST   /admin/indices/{ns}/vector/reindex-failed        → reset exhausted queue entries (200)
//! DELETE /admin/indices/{ns}/vector/drop-all              → clear all vector index data (202)
//! GET    /admin/indices/{ns}/vector/probe-trace           → one query's postings at probe ranks start..end, with min/max Pass-1 scores, against a search at top_k
//! GET    /admin/indices/{ns}/vector/queue                 → all queue entries for namespace
//! GET    /admin/indices/{ns}/vector/queue/retried         → retried entries for namespace
//! GET    /admin/indices/{ns}/vector/queue/{doc_id}        → look up one queue entry
//! DELETE /admin/indices/{ns}/vector/queue/{doc_id}        → remove one queue entry
//! POST   /admin/indices/{ns}/vector/queue/{doc_id}/retry  → retry one exhausted entry
//! ```
//!
//! One file per topic: `progress`, `attribute`, `vector_ops`, `partition`, `queue`.

mod attribute;
mod partition;
mod progress;
mod queue;
mod vector_ops;

pub use attribute::*;
pub use partition::*;
pub use progress::*;
pub use queue::*;
pub use vector_ops::*;
