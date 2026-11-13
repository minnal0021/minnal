//! What a namespace reports about its value log: [`Stats`] for its current
//! state, [`GCStats`] for one garbage-collection pass.

/// The outcome of one value-log GC pass, with the namespace's running totals.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GCStats {
    /// Bytes of the segment files this pass unlinked (what it rewrote elsewhere is
    /// not subtracted).
    pub bytes_reclaimed: u64,
    /// Record bytes still referenced by the LSM after the pass, across all buckets.
    pub bytes_live: u64,
    /// GC passes this namespace has run, across all buckets, since it was created.
    pub gc_run_count: u64,
    /// Bytes every GC pass has freed since the namespace was created.
    pub total_bytes_reclaimed: u64,
    /// How long this pass took.
    pub gc_duration_ms: u128,
}

/// A namespace's value-log state now.
#[derive(Debug, serde::Serialize)]
pub struct Stats {
    /// Value-log segment files currently on disk across all buckets.
    pub segment_count: u64,
    /// Bytes those segment files occupy on disk.
    pub disk_bytes: u64,
    /// Record bytes no key points at any more: what GC can reclaim.
    pub garbage_size: u64,
    /// `garbage_size` as a percentage of all record bytes (live + garbage), the
    /// figure GC's trigger compares against `value_log_waste_threshold`.
    pub waste_ratio: f64,
    /// GC passes this namespace has run, across all buckets, since it was created.
    pub total_gc_runs: u64,
    /// Bytes every GC pass has freed since the namespace was created.
    pub total_bytes_reclaimed: u64,
    /// Record bytes the LSM still points at.
    pub live_bytes: u64,
}
