//! What the engine did during a cell: its counters before and after, and the
//! time series sampled while it ran.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use minnal_db::MetricsSnapshot;
use minnal_db::doc_store::DocStore;
use serde::Serialize;

use crate::engine;
use crate::runner::Ctx;

/// Engine counters taken before and after a cell.
#[derive(Serialize, Default, Clone, Copy)]
pub(crate) struct Counters {
    pub(crate) puts: u64,
    pub(crate) deletes: u64,
    reads: u64,
    scans: u64,
    wal_writes: u64,
    memtable_flushes: u64,
    l0_l1_compactions: u64,
    compaction_bytes_merged: u64,
    vlog_gc_runs: u64,
    vlog_gc_bytes_reclaimed: u64,
    wal_gc_runs: u64,
    wal_segments_deleted: u64,
    pub(crate) vector_wal_appends: u64,
    vector_wal_fsyncs: u64,
    vector_wal_gc_runs: u64,
    vector_wal_segments_deleted: u64,
    splits: u64,
    merges: u64,
    reassigns: u64,
    postings: u64,
    /// Highest field-index checkpoint offset: advances with each checkpoint
    /// that wrote something.
    pub(crate) checkpoint_offset: u64,
    disk_bytes: u64,
    rss_bytes: u64,
    #[serde(flatten)]
    memory: engine::Memory,
}

impl Counters {
    pub(crate) async fn take(ctx: &Ctx) -> Self {
        let m: MetricsSnapshot = ctx.target.store.db().ops_metrics();
        let (mut splits, mut merges, mut reassigns, mut postings) = (0, 0, 0, 0);
        if let Ok((_, h, _)) = ctx.target.store.vector_partition_health(&ctx.target.ns, false).await {
            (splits, merges, reassigns, postings) = (h.maintenance.splits, h.maintenance.merges, h.maintenance.reassigns, h.postings as u64);
        }
        let checkpoint_offset = ctx
            .target
            .store
            .index_health(&ctx.target.ns)
            .await
            .map(|fields| fields.iter().filter_map(|f| f.checkpoint_offset).max().unwrap_or(0))
            .unwrap_or(0);
        Self {
            puts: m.puts,
            deletes: m.deletes,
            reads: m.reads,
            scans: m.scans,
            wal_writes: m.wal_writes,
            memtable_flushes: m.memtable_flushes,
            l0_l1_compactions: m.l0_l1_compactions,
            compaction_bytes_merged: m.compaction_bytes_merged,
            vlog_gc_runs: m.vlog_gc_runs,
            vlog_gc_bytes_reclaimed: m.vlog_gc_bytes_reclaimed,
            wal_gc_runs: m.wal_gc_runs,
            wal_segments_deleted: m.wal_segments_deleted,
            vector_wal_appends: m.vector_wal_appends,
            vector_wal_fsyncs: m.vector_wal_fsyncs,
            vector_wal_gc_runs: m.vector_wal_gc_runs,
            vector_wal_segments_deleted: m.vector_wal_segments_deleted,
            splits,
            merges,
            reassigns,
            postings,
            checkpoint_offset,
            disk_bytes: engine::disk_bytes(&ctx.out.join("engine")),
            rss_bytes: engine::rss_bytes(),
            memory: engine::memory(),
        }
    }

    /// Counters accumulated from `before` to `self`; gauges (`postings`,
    /// `disk_bytes`, `rss_bytes`, `checkpoint_offset`) are the value after.
    pub(crate) fn since(&self, b: &Counters) -> Counters {
        let d = |a: u64, b: u64| a.saturating_sub(b);
        Counters {
            puts: d(self.puts, b.puts),
            deletes: d(self.deletes, b.deletes),
            reads: d(self.reads, b.reads),
            scans: d(self.scans, b.scans),
            wal_writes: d(self.wal_writes, b.wal_writes),
            memtable_flushes: d(self.memtable_flushes, b.memtable_flushes),
            l0_l1_compactions: d(self.l0_l1_compactions, b.l0_l1_compactions),
            compaction_bytes_merged: d(self.compaction_bytes_merged, b.compaction_bytes_merged),
            vlog_gc_runs: d(self.vlog_gc_runs, b.vlog_gc_runs),
            vlog_gc_bytes_reclaimed: d(self.vlog_gc_bytes_reclaimed, b.vlog_gc_bytes_reclaimed),
            wal_gc_runs: d(self.wal_gc_runs, b.wal_gc_runs),
            wal_segments_deleted: d(self.wal_segments_deleted, b.wal_segments_deleted),
            vector_wal_appends: d(self.vector_wal_appends, b.vector_wal_appends),
            vector_wal_fsyncs: d(self.vector_wal_fsyncs, b.vector_wal_fsyncs),
            vector_wal_gc_runs: d(self.vector_wal_gc_runs, b.vector_wal_gc_runs),
            vector_wal_segments_deleted: d(self.vector_wal_segments_deleted, b.vector_wal_segments_deleted),
            splits: d(self.splits, b.splits),
            merges: d(self.merges, b.merges),
            reassigns: d(self.reassigns, b.reassigns),
            postings: self.postings,
            checkpoint_offset: self.checkpoint_offset,
            disk_bytes: self.disk_bytes,
            rss_bytes: self.rss_bytes,
            memory: self.memory,
        }
    }
}

pub(crate) async fn queue_depth(store: &DocStore) -> usize {
    // Counted from the keys alone: the values are the documents' texts.
    store.pending_vector_index_count().await.unwrap_or(0)
}

/// One sample of the time series kept during a cell.
#[derive(Serialize)]
pub(crate) struct Sample {
    t_s: f64,
    queue: usize,
    rss_bytes: u64,
    #[serde(flatten)]
    memory: engine::Memory,
    disk_bytes: u64,
    puts: u64,
    deletes: u64,
    reads: u64,
    memtable_flushes: u64,
    l0_l1_compactions: u64,
    vector_wal_appends: u64,
}

pub(crate) fn spawn_monitor(ctx: &Ctx, every: Duration, stop: Arc<AtomicBool>) -> tokio::task::JoinHandle<Vec<Sample>> {
    let (store, dir) = (Arc::clone(&ctx.target.store), ctx.out.join("engine"));
    tokio::spawn(async move {
        let t0 = Instant::now();
        let mut out = Vec::new();
        while !stop.load(Ordering::SeqCst) {
            let m = store.db().ops_metrics();
            out.push(Sample {
                t_s: t0.elapsed().as_secs_f64(),
                queue: queue_depth(&store).await,
                rss_bytes: engine::rss_bytes(),
                memory: engine::memory(),
                disk_bytes: engine::disk_bytes(&dir),
                puts: m.puts,
                deletes: m.deletes,
                reads: m.reads,
                memtable_flushes: m.memtable_flushes,
                l0_l1_compactions: m.l0_l1_compactions,
                vector_wal_appends: m.vector_wal_appends,
            });
            tokio::time::sleep(every).await;
        }
        out
    })
}
