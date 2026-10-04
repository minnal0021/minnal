//! Measuring the vector index: the per-namespace corruption counters search
//! keeps at runtime ([`corruption`], re-exported here), and the `#[ignore]`d BEIR
//! relevance evaluation of the production pipeline (`beir_eval`, tests only).

#[cfg(test)]
pub(in crate::semantic_search) mod beir_eval;
pub mod corruption;

pub use corruption::{VectorMetricsSnapshot, record_dense_corrupt_skipped, record_sparse_corrupt_skipped, reset, snapshot, snapshot_all};
