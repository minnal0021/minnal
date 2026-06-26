//! Vector-search corruption metrics — cumulative, process-wide counters.
//!
//! These count how often a stored vector-index entry was **skipped during search
//! because its bytes failed to deserialize** (corruption, or a write-path bug).
//! Together with the per-skip `warn!` logging they make index degradation
//! *trackable*, not just log-visible: sample [`snapshot`] twice to compute a rate,
//! or alert on a non-zero/rising value. They are monotonically-increasing counters
//! since process start.
//!
//! Implemented as a single process-global `static` of `Relaxed` atomics, mirroring
//! the `minnal_db` ops-metrics pattern. A corrupt entry is a rare, off-hot-path
//! event, so a global counter (rather than a handle threaded through
//! [`crate::service::search`]) is both sufficient and the least invasive.

use std::sync::atomic::{AtomicU64, Ordering};

use serde::Serialize;

/// Process-wide vector-search corruption counters.
#[derive(Debug)]
struct VectorMetrics {
    /// Pass-1 sparse (SingleBit) entries skipped because their bytes failed to deserialize.
    sparse_corrupt_skipped: AtomicU64,
    /// Pass-2 dense (MultiBit) entries skipped because their bytes failed to deserialize.
    dense_corrupt_skipped: AtomicU64,
}

impl VectorMetrics {
    const fn new() -> Self {
        Self {
            sparse_corrupt_skipped: AtomicU64::new(0),
            dense_corrupt_skipped: AtomicU64::new(0),
        }
    }
}

static METRICS: VectorMetrics = VectorMetrics::new();

/// Record that a Pass-1 sparse entry was skipped because it failed to deserialize.
#[inline]
pub fn record_sparse_corrupt_skipped() {
    METRICS.sparse_corrupt_skipped.fetch_add(1, Ordering::Relaxed);
}

/// Record that a Pass-2 dense entry was skipped because it failed to deserialize.
#[inline]
pub fn record_dense_corrupt_skipped() {
    METRICS.dense_corrupt_skipped.fetch_add(1, Ordering::Relaxed);
}

/// An immutable, serializable read of the corruption counters.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct VectorMetricsSnapshot {
    /// Sparse (Pass-1) entries skipped since process start due to a deserialize failure.
    pub sparse_corrupt_skipped: u64,
    /// Dense (Pass-2) entries skipped since process start due to a deserialize failure.
    pub dense_corrupt_skipped: u64,
    /// Convenience total of the two corruption counters.
    pub total_corrupt_skipped: u64,
}

/// Snapshot the current corruption counters.
pub fn snapshot() -> VectorMetricsSnapshot {
    let sparse = METRICS.sparse_corrupt_skipped.load(Ordering::Relaxed);
    let dense = METRICS.dense_corrupt_skipped.load(Ordering::Relaxed);
    VectorMetricsSnapshot {
        sparse_corrupt_skipped: sparse,
        dense_corrupt_skipped: dense,
        total_corrupt_skipped: sparse + dense,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The counters are process-global, so assert *deltas* with `>=` rather than
    // absolute values — other tests in the binary may increment them concurrently.

    #[test]
    fn record_increments_each_counter() {
        let before = snapshot();
        record_sparse_corrupt_skipped();
        record_dense_corrupt_skipped();
        record_dense_corrupt_skipped();
        let after = snapshot();
        assert!(after.sparse_corrupt_skipped >= before.sparse_corrupt_skipped + 1);
        assert!(after.dense_corrupt_skipped >= before.dense_corrupt_skipped + 2);
        assert_eq!(after.total_corrupt_skipped, after.sparse_corrupt_skipped + after.dense_corrupt_skipped);
    }
}
