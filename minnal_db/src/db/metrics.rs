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
//! the overhead is negligible. Each namespace's `KVStore`/`LSMTree` owns its own
//! [`Metrics`] instance (threaded in via a `OnceLock`), so counters are recorded
//! **per namespace**; a standalone store/tree (in tests) simply records nothing.
//! `Database` keeps one extra global [`Metrics`] for the counters that belong to
//! no single namespace (WAL GC) plus a fold of every dropped namespace's final
//! totals (so engine-wide aggregates stay monotonic across namespace drops). The
//! engine-wide snapshot is the sum of all live per-namespace snapshots and that
//! global instance ([`Database::metrics_snapshot`](crate::db::database::Database)).

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
    pub no_wal_deletes: AtomicU64,
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
    /// Value-log segment files GC has unlinked.
    pub vlog_segments_reclaimed: AtomicU64,
    /// Bytes returned to the filesystem by unlinking those segments.
    pub vlog_gc_bytes_reclaimed: AtomicU64,
    /// Bytes of *survivors* GC rewrote to relocate them out of those segments — the
    /// cost of the pass. `bytes_rewritten / bytes_reclaimed` is GC's write
    /// amplification, and is the number to watch: it should sit near 1, and a value
    /// far above it means GC is repeatedly relocating data it cannot actually free.
    pub vlog_gc_bytes_rewritten: AtomicU64,
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
            no_wal_deletes: g(&self.no_wal_deletes),
            wal_bytes_appended: g(&self.wal_bytes_appended),
            wal_fsyncs: g(&self.wal_fsyncs),
            apply_failures: g(&self.apply_failures),
            memtable_flushes: g(&self.memtable_flushes),
            l0_l1_compactions: g(&self.l0_l1_compactions),
            compaction_bytes_merged: g(&self.compaction_bytes_merged),
            compaction_duration_ms: g(&self.compaction_duration_ms),
            vlog_gc_runs: g(&self.vlog_gc_runs),
            vlog_gc_duration_ms: g(&self.vlog_gc_duration_ms),
            vlog_segments_reclaimed: g(&self.vlog_segments_reclaimed),
            vlog_gc_bytes_reclaimed: g(&self.vlog_gc_bytes_reclaimed),
            vlog_gc_bytes_rewritten: g(&self.vlog_gc_bytes_rewritten),
            wal_gc_runs: g(&self.wal_gc_runs),
            wal_segments_deleted: g(&self.wal_segments_deleted),
        }
    }

    /// Fold a snapshot into these counters (each field `fetch_add`-ed).
    ///
    /// Used when a namespace is dropped: its final per-namespace totals are
    /// folded into the `Database`-level global instance so engine-wide aggregates
    /// remain monotonic even though the per-namespace counters disappear with it.
    pub fn add_snapshot(&self, o: &MetricsSnapshot) {
        let a = |c: &AtomicU64, v: u64| {
            c.fetch_add(v, Ordering::Relaxed);
        };
        a(&self.reads, o.reads);
        a(&self.read_hits, o.read_hits);
        a(&self.read_misses, o.read_misses);
        a(&self.scans, o.scans);
        a(&self.scan_rows, o.scan_rows);
        a(&self.lookups, o.lookups);
        a(&self.fast_path_hits, o.fast_path_hits);
        a(&self.l0_probes, o.l0_probes);
        a(&self.l1_probes, o.l1_probes);
        a(&self.bloom_rejects, o.bloom_rejects);
        a(&self.puts, o.puts);
        a(&self.deletes, o.deletes);
        a(&self.no_wal_puts, o.no_wal_puts);
        a(&self.no_wal_deletes, o.no_wal_deletes);
        a(&self.wal_bytes_appended, o.wal_bytes_appended);
        a(&self.wal_fsyncs, o.wal_fsyncs);
        a(&self.apply_failures, o.apply_failures);
        a(&self.memtable_flushes, o.memtable_flushes);
        a(&self.l0_l1_compactions, o.l0_l1_compactions);
        a(&self.compaction_bytes_merged, o.compaction_bytes_merged);
        a(&self.compaction_duration_ms, o.compaction_duration_ms);
        a(&self.vlog_gc_runs, o.vlog_gc_runs);
        a(&self.vlog_gc_duration_ms, o.vlog_gc_duration_ms);
        a(&self.wal_gc_runs, o.wal_gc_runs);
        a(&self.wal_segments_deleted, o.wal_segments_deleted);
    }
}

/// A point-in-time copy of every [`Metrics`] counter. Serializable for the
/// admin API.
#[derive(Debug, Clone, Default, Serialize)]
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
    pub no_wal_deletes: u64,
    pub wal_bytes_appended: u64,
    pub wal_fsyncs: u64,
    pub apply_failures: u64,
    pub memtable_flushes: u64,
    pub l0_l1_compactions: u64,
    pub compaction_bytes_merged: u64,
    pub compaction_duration_ms: u64,
    pub vlog_gc_runs: u64,
    pub vlog_gc_duration_ms: u64,
    pub vlog_segments_reclaimed: u64,
    pub vlog_gc_bytes_reclaimed: u64,
    pub vlog_gc_bytes_rewritten: u64,
    pub wal_gc_runs: u64,
    pub wal_segments_deleted: u64,
}

impl MetricsSnapshot {
    /// Add every field of `o` into `self`. Used to build the engine-wide
    /// aggregate by summing all per-namespace snapshots.
    pub fn accumulate(&mut self, o: &MetricsSnapshot) {
        self.reads += o.reads;
        self.read_hits += o.read_hits;
        self.read_misses += o.read_misses;
        self.scans += o.scans;
        self.scan_rows += o.scan_rows;
        self.lookups += o.lookups;
        self.fast_path_hits += o.fast_path_hits;
        self.l0_probes += o.l0_probes;
        self.l1_probes += o.l1_probes;
        self.bloom_rejects += o.bloom_rejects;
        self.puts += o.puts;
        self.deletes += o.deletes;
        self.no_wal_puts += o.no_wal_puts;
        self.no_wal_deletes += o.no_wal_deletes;
        self.wal_bytes_appended += o.wal_bytes_appended;
        self.wal_fsyncs += o.wal_fsyncs;
        self.apply_failures += o.apply_failures;
        self.memtable_flushes += o.memtable_flushes;
        self.l0_l1_compactions += o.l0_l1_compactions;
        self.compaction_bytes_merged += o.compaction_bytes_merged;
        self.compaction_duration_ms += o.compaction_duration_ms;
        self.vlog_gc_runs += o.vlog_gc_runs;
        self.vlog_gc_duration_ms += o.vlog_gc_duration_ms;
        self.vlog_segments_reclaimed += o.vlog_segments_reclaimed;
        self.vlog_gc_bytes_reclaimed += o.vlog_gc_bytes_reclaimed;
        self.vlog_gc_bytes_rewritten += o.vlog_gc_bytes_rewritten;
        self.wal_gc_runs += o.wal_gc_runs;
        self.wal_segments_deleted += o.wal_segments_deleted;
    }
}
