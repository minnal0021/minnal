//! Engine-wide operational metrics — cumulative runtime counters.
//!
//! Unlike the structural snapshots ([`Stats`](crate::Stats),
//! [`WalMetadata`](crate::WalMetadata), LSM manifests) which report *how big*
//! the engine is, these report *what it is doing*: read/write throughput, the
//! effectiveness of the read fast path and bloom filters, flush/compaction/GC
//! activity, and durability work. They are monotonically-increasing counters
//! (plus a few accumulated durations); callers compute rates by sampling twice.
//!
//! All counters are `Relaxed` atomics on paths that already do far more work, so
//! the overhead is negligible. A single shared [`Metrics`] is created in
//! `Database::open` and threaded into every `KVStore`/`LSMTree` via a
//! `OnceLock`, so a standalone store/tree (in tests) simply records nothing.

use std::sync::atomic::{AtomicU64, Ordering};

use serde::Serialize;

/// Shared, engine-wide operational counters. Cheap to read on hot paths
/// (`Relaxed` atomic adds). Snapshot with [`Metrics::snapshot`].
#[derive(Debug, Default)]
pub struct Metrics {
    // ── Reads (user-facing point reads, at `KVStore::get`) ──────────────
    pub reads: AtomicU64,
    pub read_hits: AtomicU64,
    pub read_misses: AtomicU64,
    pub scans: AtomicU64,
    pub scan_rows: AtomicU64,

    // ── LSM point-lookup internals (`get_with_seq`; includes GC-guard reads) ─
    /// Total point lookups through the LSM (≥ `reads` — also counts GC validation).
    pub lookups: AtomicU64,
    /// Lookups served by the active-memtable fast path (no lower-layer scan).
    pub fast_path_hits: AtomicU64,
    /// Lookups that scanned at least one Level-0 file.
    pub l0_probes: AtomicU64,
    /// Lookups that scanned the Level-1 file (i.e. not bloom-rejected).
    pub l1_probes: AtomicU64,
    /// L1 lookups short-circuited by the bloom filter ("definitely absent").
    pub bloom_rejects: AtomicU64,

    // ── Writes ──────────────────────────────────────────────────────────
    pub puts: AtomicU64,
    pub deletes: AtomicU64,
    pub no_wal_puts: AtomicU64,
    pub wal_bytes_appended: AtomicU64,
    /// WAL fsyncs (one per WAL-backed write — durability cost).
    pub wal_fsyncs: AtomicU64,
    /// In-memory applies that failed after retry (data still durable in the WAL).
    pub apply_failures: AtomicU64,

    // ── Flush / compaction ──────────────────────────────────────────────
    pub memtable_flushes: AtomicU64,
    pub l0_l1_compactions: AtomicU64,
    pub compaction_bytes_merged: AtomicU64,
    pub compaction_duration_ms: AtomicU64,

    // ── Garbage collection ──────────────────────────────────────────────
    pub vlog_gc_runs: AtomicU64,
    pub vlog_gc_duration_ms: AtomicU64,
    pub wal_gc_runs: AtomicU64,
    pub wal_segments_deleted: AtomicU64,
}

impl Metrics {
    /// Record a user-facing point read and whether it found a live value.
    #[inline]
    pub fn record_read(&self, hit: bool) {
        self.reads.fetch_add(1, Ordering::Relaxed);
        if hit {
            self.read_hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.read_misses.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record a multi-key scan and how many rows it returned.
    #[inline]
    pub fn record_scan(&self, rows: u64) {
        self.scans.fetch_add(1, Ordering::Relaxed);
        self.scan_rows.fetch_add(rows, Ordering::Relaxed);
    }

    #[inline]
    pub fn bump(counter: &AtomicU64) {
        counter.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn add(counter: &AtomicU64, n: u64) {
        counter.fetch_add(n, Ordering::Relaxed);
    }

    /// Atomically read every counter into a serializable snapshot.
    pub fn snapshot(&self) -> MetricsSnapshot {
        let g = |c: &AtomicU64| c.load(Ordering::Relaxed);
        MetricsSnapshot {
            reads: g(&self.reads),
            read_hits: g(&self.read_hits),
            read_misses: g(&self.read_misses),
            scans: g(&self.scans),
            scan_rows: g(&self.scan_rows),
            lookups: g(&self.lookups),
            fast_path_hits: g(&self.fast_path_hits),
            l0_probes: g(&self.l0_probes),
            l1_probes: g(&self.l1_probes),
            bloom_rejects: g(&self.bloom_rejects),
            puts: g(&self.puts),
            deletes: g(&self.deletes),
            no_wal_puts: g(&self.no_wal_puts),
            wal_bytes_appended: g(&self.wal_bytes_appended),
            wal_fsyncs: g(&self.wal_fsyncs),
            apply_failures: g(&self.apply_failures),
            memtable_flushes: g(&self.memtable_flushes),
            l0_l1_compactions: g(&self.l0_l1_compactions),
            compaction_bytes_merged: g(&self.compaction_bytes_merged),
            compaction_duration_ms: g(&self.compaction_duration_ms),
            vlog_gc_runs: g(&self.vlog_gc_runs),
            vlog_gc_duration_ms: g(&self.vlog_gc_duration_ms),
            wal_gc_runs: g(&self.wal_gc_runs),
            wal_segments_deleted: g(&self.wal_segments_deleted),
        }
    }
}

/// A point-in-time copy of every [`Metrics`] counter. Serializable for the
/// admin API.
#[derive(Debug, Clone, Serialize)]
pub struct MetricsSnapshot {
    pub reads: u64,
    pub read_hits: u64,
    pub read_misses: u64,
    pub scans: u64,
    pub scan_rows: u64,
    pub lookups: u64,
    pub fast_path_hits: u64,
    pub l0_probes: u64,
    pub l1_probes: u64,
    pub bloom_rejects: u64,
    pub puts: u64,
    pub deletes: u64,
    pub no_wal_puts: u64,
    pub wal_bytes_appended: u64,
    pub wal_fsyncs: u64,
    pub apply_failures: u64,
    pub memtable_flushes: u64,
    pub l0_l1_compactions: u64,
    pub compaction_bytes_merged: u64,
    pub compaction_duration_ms: u64,
    pub vlog_gc_runs: u64,
    pub vlog_gc_duration_ms: u64,
    pub wal_gc_runs: u64,
    pub wal_segments_deleted: u64,
}
