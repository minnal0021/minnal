//! Runtime measurements of the vector index: the per-namespace corruption
//! counters search keeps ([`corruption`], re-exported here).

pub mod corruption;

pub use corruption::{VectorMetricsSnapshot, record_dense_corrupt_skipped, record_sparse_corrupt_skipped, reset, snapshot, snapshot_all};
