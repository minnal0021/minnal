//! Vector-search corruption metrics — cumulative, per-namespace counters.
//!
//! These count how often a stored vector-index entry was **skipped during search
//! because its bytes failed to deserialize** (corruption, or a write-path bug).
//! Together with the per-skip `warn!` logging they make index degradation
//! *trackable*, not just log-visible: sample [`snapshot`] twice to compute a rate,
//! or alert on a non-zero/rising value. They are monotonically-increasing counters
//! since process start.
//!
//! Counters are **kept per namespace** so a degraded namespace is distinguishable
//! from a healthy one. They live **in memory only** — a process-global map of
//! `Relaxed` atomics behind a `RwLock`, reset to zero on restart. A corrupt entry
//! is a rare, off-hot-path event, so the lock cost on record is irrelevant and a
//! global map (rather than a handle threaded through [`crate::service::search`]) is
//! both sufficient and the least invasive.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, RwLock};

use serde::Serialize;

/// Per-namespace vector-search corruption counters.
#[derive(Debug, Default)]
struct VectorMetrics {
    /// Pass-1 sparse (SingleBit) entries skipped because their bytes failed to deserialize.
    sparse_corrupt_skipped: AtomicU64,
    /// Pass-2 dense (MultiBit) entries skipped because their bytes failed to deserialize.
    dense_corrupt_skipped: AtomicU64,
}

/// Process-wide map of `namespace -> counters`. Read-locked on the common record
/// path (the namespace's counters already exist); write-locked only the first
/// time a namespace records a corruption.
static METRICS: LazyLock<RwLock<BTreeMap<String, VectorMetrics>>> = LazyLock::new(|| RwLock::new(BTreeMap::new()));

/// Increment one counter for `namespace`, selected by `select`.
fn record(namespace: &str, select: impl Fn(&VectorMetrics) -> &AtomicU64) {
    // Fast path: the namespace already has counters — a shared read lock suffices
    // because the counter itself is an atomic.
    if let Some(m) = METRICS.read().expect("vector metrics lock poisoned").get(namespace) {
        select(m).fetch_add(1, Ordering::Relaxed);
        return;
    }
    // Slow path (first corruption for this namespace): insert then increment.
    let mut map = METRICS.write().expect("vector metrics lock poisoned");
    let m = map.entry(namespace.to_owned()).or_default();
    select(m).fetch_add(1, Ordering::Relaxed);
}

/// Record that a Pass-1 sparse entry in `namespace` was skipped because it failed to deserialize.
#[inline]
pub fn record_sparse_corrupt_skipped(namespace: &str) {
    record(namespace, |m| &m.sparse_corrupt_skipped);
}

/// Record that a Pass-2 dense entry in `namespace` was skipped because it failed to deserialize.
#[inline]
pub fn record_dense_corrupt_skipped(namespace: &str) {
    record(namespace, |m| &m.dense_corrupt_skipped);
}

/// An immutable, serializable read of one namespace's corruption counters.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct VectorMetricsSnapshot {
    /// Sparse (Pass-1) entries skipped since process start due to a deserialize failure.
    pub sparse_corrupt_skipped: u64,
    /// Dense (Pass-2) entries skipped since process start due to a deserialize failure.
    pub dense_corrupt_skipped: u64,
    /// Convenience total of the two corruption counters.
    pub total_corrupt_skipped: u64,
}

impl VectorMetricsSnapshot {
    fn read_from(m: &VectorMetrics) -> Self {
        let sparse = m.sparse_corrupt_skipped.load(Ordering::Relaxed);
        let dense = m.dense_corrupt_skipped.load(Ordering::Relaxed);
        Self {
            sparse_corrupt_skipped: sparse,
            dense_corrupt_skipped: dense,
            total_corrupt_skipped: sparse + dense,
        }
    }
}

/// Snapshot the corruption counters for one `namespace`. Returns all-zero for a
/// namespace that has never recorded a corruption.
pub fn snapshot(namespace: &str) -> VectorMetricsSnapshot {
    METRICS
        .read()
        .expect("vector metrics lock poisoned")
        .get(namespace)
        .map(VectorMetricsSnapshot::read_from)
        .unwrap_or_default()
}

/// Snapshot the corruption counters for every namespace that has recorded one,
/// keyed by namespace name (sorted).
pub fn snapshot_all() -> BTreeMap<String, VectorMetricsSnapshot> {
    METRICS
        .read()
        .expect("vector metrics lock poisoned")
        .iter()
        .map(|(ns, m)| (ns.clone(), VectorMetricsSnapshot::read_from(m)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // The counters are process-global, so assert *deltas* with `>=` rather than
    // absolute values, and use test-unique namespaces so concurrent tests in the
    // binary don't perturb each other's namespace counts.

    #[test]
    fn record_increments_each_counter_for_its_namespace() {
        let ns = "metrics_test_ns_a";
        let before = snapshot(ns);
        record_sparse_corrupt_skipped(ns);
        record_dense_corrupt_skipped(ns);
        record_dense_corrupt_skipped(ns);
        let after = snapshot(ns);
        assert!(after.sparse_corrupt_skipped > before.sparse_corrupt_skipped);
        assert!(after.dense_corrupt_skipped >= before.dense_corrupt_skipped + 2);
        assert_eq!(after.total_corrupt_skipped, after.sparse_corrupt_skipped + after.dense_corrupt_skipped);
    }

    #[test]
    fn counters_are_isolated_per_namespace() {
        let ns1 = "metrics_test_ns_b";
        let ns2 = "metrics_test_ns_c";
        record_sparse_corrupt_skipped(ns1);
        // ns2 must not see ns1's increment.
        assert_eq!(snapshot(ns2).sparse_corrupt_skipped, 0);
        assert!(snapshot(ns1).sparse_corrupt_skipped >= 1);
        // The all-namespaces view includes the namespace that recorded.
        assert!(snapshot_all().contains_key(ns1));
    }

    #[test]
    fn snapshot_of_unknown_namespace_is_zero() {
        let s = snapshot("metrics_test_ns_never_used");
        assert_eq!(s.sparse_corrupt_skipped, 0);
        assert_eq!(s.dense_corrupt_skipped, 0);
        assert_eq!(s.total_corrupt_skipped, 0);
    }
}
