//! A run: a fixed plan of cells, each measured, checked and saved before the
//! next starts. A restarted run resumes from the manifest.
//!
//! Plan, each block repeated `repeats` times (repeat 1 of every cell, then
//! repeat 2, ...), so drift over the run shows up between repeats:
//!
//! 1. `single-{class}-r{k}`: one client sends one class back to back, while a
//!    trickle of writes (`trickle_rate`/s, `trickle_mix`) keeps vector
//!    indexing, flushes and maintenance working. The trickle is recorded
//!    separately.
//! 2. `closed-{n}-r{k}`: `n` clients, each sending the next operation from
//!    `mix` as soon as its last returns.
//! 3. `soak-{n}`: `soak_clients` clients for `soak_s`, sampled every 10 s.
//!
//! Every cell: wait for the vector queue to drain, take counters, run
//! `warm_s` unmeasured, measure, wait for the queue to drain again (so
//! ingestion lag samples complete), take counters, check validity. An invalid
//! cell is saved with its reasons and stops the run; fix the cause and run
//! again to resume at that cell.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use minnal_db::MetricsSnapshot;
use minnal_db::doc_store::schema::KeyType;
use minnal_db::doc_store::{DocStore, Pagination};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::engine::{self, Config, EngineSettings};
use crate::replay::{Replay, ReplayStats};
use crate::stats::Recording;
use crate::workload::{CLASSES, Cursors, Data, Generator, Op, Outcome, Target, execute, search_spec, weights};

/// Shared by every generator of a run: one hot set of `get` ids.
const HOT_SEED: u64 = 0x005e_ed0f_ae71;

#[derive(Clone, Copy, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Kind {
    Single { class: usize },
    Closed { clients: usize },
    Soak { clients: usize },
}

#[derive(Clone, Serialize)]
struct Cell {
    name: String,
    #[serde(flatten)]
    kind: Kind,
    measure_s: u64,
    warm_s: u64,
}

fn plan(cfg: &Config) -> Vec<Cell> {
    let mut cells = Vec::new();
    let cell = |name: String, kind, measure_s| Cell {
        name,
        kind,
        measure_s,
        warm_s: cfg.warm_s,
    };
    for r in 1..=cfg.repeats {
        for (class, c) in CLASSES.iter().enumerate() {
            cells.push(cell(format!("single-{c}-r{r}"), Kind::Single { class }, cfg.single_s));
        }
    }
    for r in 1..=cfg.repeats {
        for &clients in &cfg.clients {
            cells.push(cell(format!("closed-{clients:02}-r{r}"), Kind::Closed { clients }, cfg.concurrent_s));
        }
    }
    cells.push(cell(
        format!("soak-{:02}", cfg.soak_clients),
        Kind::Soak { clients: cfg.soak_clients },
        cfg.soak_s,
    ));
    cells
}

#[derive(Serialize, Deserialize, Default)]
struct Manifest {
    /// Cell name -> "done" or "invalid".
    cells: BTreeMap<String, String>,
    cursors: Cursors,
    /// Unix seconds of the first start and of every resume.
    starts: Vec<u64>,
    primed: bool,
    complete: bool,
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn write_json(path: &Path, v: &impl Serialize) -> Result<(), String> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(v).map_err(|e| e.to_string())?).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))
}

struct Ctx {
    cfg: Config,
    out: PathBuf,
    data: Arc<Data>,
    target: Arc<Target>,
    replay: Arc<Replay>,
    cursors: Arc<Mutex<Cursors>>,
    mix: [f64; 8],
    trickle_mix: [f64; 8],
}

/// Engine counters taken before and after a cell.
#[derive(Serialize, Default, Clone, Copy)]
struct Counters {
    puts: u64,
    deletes: u64,
    reads: u64,
    scans: u64,
    wal_fsyncs: u64,
    memtable_flushes: u64,
    l0_l1_compactions: u64,
    compaction_bytes_merged: u64,
    vlog_gc_runs: u64,
    vlog_gc_bytes_reclaimed: u64,
    wal_gc_runs: u64,
    wal_segments_deleted: u64,
    vector_wal_appends: u64,
    vector_wal_fsyncs: u64,
    vector_wal_gc_runs: u64,
    vector_wal_segments_deleted: u64,
    splits: u64,
    merges: u64,
    reassigns: u64,
    postings: u64,
    /// Highest field-index checkpoint offset: advances with each checkpoint
    /// that wrote something.
    checkpoint_offset: u64,
    disk_bytes: u64,
    rss_bytes: u64,
    #[serde(flatten)]
    memory: engine::Memory,
}

impl Counters {
    async fn take(ctx: &Ctx) -> Self {
        let m: MetricsSnapshot = ctx.target.store.ops_metrics();
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
            wal_fsyncs: m.wal_fsyncs,
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
    fn since(&self, b: &Counters) -> Counters {
        let d = |a: u64, b: u64| a.saturating_sub(b);
        Counters {
            puts: d(self.puts, b.puts),
            deletes: d(self.deletes, b.deletes),
            reads: d(self.reads, b.reads),
            scans: d(self.scans, b.scans),
            wal_fsyncs: d(self.wal_fsyncs, b.wal_fsyncs),
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

async fn queue_depth(store: &DocStore) -> usize {
    store.list_queue_entries().await.len()
}

/// Wait until the vector queue is empty; `None` if it took longer than `max`.
async fn drain(store: &DocStore, max: Duration) -> Option<f64> {
    let t = Instant::now();
    let mut reported = Duration::ZERO;
    loop {
        let entries = store.list_queue_entries().await;
        if entries.is_empty() {
            return Some(t.elapsed().as_secs_f64());
        }
        if t.elapsed() > max {
            return None;
        }
        // A queue that does not drain says why every 30 s.
        if t.elapsed() >= reported + Duration::from_secs(30) {
            reported = t.elapsed();
            let clear = entries
                .iter()
                .filter(|e| matches!(e.kind, minnal_db::vector_kv::QueueEntryKind::Clear))
                .count();
            let retried = entries.iter().filter(|e| e.retry_count > 0).count();
            let e = &entries[0];
            eprintln!(
                "[{}] queue not drained after {:.0} s: {} entries ({clear} clear, {retried} retried); first: {} {:02x?} {:?} retries {} error {:?}",
                now_unix(),
                t.elapsed().as_secs_f64(),
                entries.len(),
                e.namespace,
                e.doc_id_bytes,
                e.kind,
                e.retry_count,
                e.last_error
            );
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

fn seed_of(name: &str, salt: u64) -> u64 {
    // FNV-1a: the same cell always sends the same operations.
    name.bytes()
        .fold(0xcbf2_9ce4_8422_2325u64 ^ salt, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
}

/// What one stretch of load produced.
#[derive(Default)]
struct Load {
    measured: Recording,
    trickle: Recording,
    /// First ranked reply per test query.
    trec: BTreeMap<String, Vec<(u64, f32)>>,
    lag_tasks: Vec<tokio::task::JoinHandle<Option<f64>>>,
}

/// After an insert is acknowledged, poll its queue entry until it is gone: the
/// time until the document is searchable.
fn spawn_lag(target: &Arc<Target>, id: u64, acked: Instant) -> tokio::task::JoinHandle<Option<f64>> {
    let target = Arc::clone(target);
    tokio::spawn(async move {
        while acked.elapsed() < Duration::from_secs(1800) {
            if target.store.get_queue_entry(&target.ns, &id.to_be_bytes()).await.is_none() {
                return Some(acked.elapsed().as_secs_f64() * 1e3);
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        None
    })
}

/// Bring the store back to the base's size, unmeasured, before a cell: a
/// single-class insert or delete cell moves it by thousands, and later cells
/// would otherwise run on a different store. Inserts or deletes in order, 16 at
/// a time, as the load would; returns how many (+ inserted, - deleted).
async fn rebalance(ctx: &Ctx, cell: &Cell) -> Result<i64, String> {
    let base = ctx.data.base.len();
    let gap = ctx.cursors.lock().documents(base) - base as i64;
    if gap == 0 {
        return Ok(0);
    }
    let class = if gap < 0 { 1 } else { 3 };
    let left = Arc::new(std::sync::atomic::AtomicI64::new(gap.abs()));
    let workers: Vec<_> = (0..16)
        .map(|w| {
            let (target, left) = (Arc::clone(&ctx.target), Arc::clone(&left));
            let mut g = Generator::new(Arc::clone(&ctx.data), HOT_SEED, seed_of(&cell.name, 0x5eed ^ w), Arc::clone(&ctx.cursors));
            tokio::spawn(async move {
                while left.fetch_sub(1, Ordering::SeqCst) > 0 {
                    let op = g.op(class).ok_or("rebalance: no document to delete")?;
                    execute(&target, &op).await.map_err(|e| format!("rebalance: {e}"))?;
                }
                Ok::<(), String>(())
            })
        })
        .collect();
    for w in workers {
        w.await.map_err(|e| e.to_string())??;
    }
    Ok(-gap)
}

/// Run `kind`'s load for `secs`. `salt` separates warm-up from measurement.
async fn run_load(ctx: &Ctx, cell: &Cell, secs: u64, salt: u64) -> Load {
    let deadline = Instant::now() + Duration::from_secs(secs);
    let (clients, single) = match cell.kind {
        Kind::Single { class } => (1, Some(class)),
        Kind::Closed { clients } | Kind::Soak { clients } => (clients, None),
    };
    let stop = Arc::new(AtomicBool::new(false));
    let trickle = single.map(|_| {
        let (target, data, cursors, stop) = (
            Arc::clone(&ctx.target),
            Arc::clone(&ctx.data),
            Arc::clone(&ctx.cursors),
            Arc::clone(&stop),
        );
        let (mix, period) = (ctx.trickle_mix, Duration::from_secs_f64(1.0 / ctx.cfg.trickle_rate));
        let seed = seed_of(&cell.name, salt ^ 0x7c);
        tokio::spawn(async move {
            let mut g = Generator::new(data, HOT_SEED, seed, cursors);
            let mut rec = Recording::default();
            let mut tick = tokio::time::interval(period);
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            while !stop.load(Ordering::SeqCst) {
                tick.tick().await;
                let class = g.class(&mix);
                let Some(op) = g.op(class) else {
                    rec.0[class].unavailable += 1;
                    continue;
                };
                let t = Instant::now();
                match execute(&target, &op).await {
                    Ok(_) => rec.0[class].ok(t.elapsed()),
                    Err(e) => rec.0[class].error(t.elapsed(), e),
                }
            }
            rec
        })
    });
    let tasks: Vec<_> = (0..clients)
        .map(|k| {
            let (target, data, cursors) = (Arc::clone(&ctx.target), Arc::clone(&ctx.data), Arc::clone(&ctx.cursors));
            let (mix, lag_every) = (ctx.mix, ctx.cfg.lag_every);
            let seed = seed_of(&cell.name, salt ^ (k as u64) << 8);
            tokio::spawn(async move {
                let mut g = Generator::new(data, HOT_SEED, seed, cursors);
                let mut load = Load::default();
                let mut inserts = 0u64;
                while Instant::now() < deadline {
                    let class = single.unwrap_or_else(|| g.class(&mix));
                    let Some(op) = g.op(class) else {
                        load.measured.0[class].unavailable += 1;
                        tokio::time::sleep(Duration::from_millis(1)).await;
                        continue;
                    };
                    let t = Instant::now();
                    let r = execute(&target, &op).await;
                    let took = t.elapsed();
                    let rec = &mut load.measured.0[class];
                    match r {
                        Ok(Outcome::Ok) => rec.ok(took),
                        Ok(Outcome::NotFound) => rec.not_found(took),
                        Ok(Outcome::Ranked(hits)) => {
                            rec.ok(took);
                            if let Op::Search { qid, .. } = &op {
                                load.trec.entry(qid.clone()).or_insert(hits);
                            }
                        }
                        Err(e) => rec.error(took, e),
                    }
                    if let Op::Insert(d) = &op {
                        inserts += 1;
                        if lag_every > 0 && inserts.is_multiple_of(lag_every) {
                            load.lag_tasks.push(spawn_lag(&target, d.id, Instant::now()));
                        }
                    }
                }
                load
            })
        })
        .collect();
    let mut load = Load::default();
    for t in tasks {
        if let Ok(l) = t.await {
            load.measured.merge(&l.measured);
            for (q, h) in l.trec {
                load.trec.entry(q).or_insert(h);
            }
            load.lag_tasks.extend(l.lag_tasks);
        }
    }
    stop.store(true, Ordering::SeqCst);
    if let Some(t) = trickle
        && let Ok(rec) = t.await
    {
        load.trickle = rec;
    }
    load
}

/// One sample of the time series kept during a cell.
#[derive(Serialize)]
struct Sample {
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

fn spawn_monitor(ctx: &Ctx, every: Duration, stop: Arc<AtomicBool>) -> tokio::task::JoinHandle<Vec<Sample>> {
    let (store, dir) = (Arc::clone(&ctx.target.store), ctx.out.join("engine"));
    tokio::spawn(async move {
        let t0 = Instant::now();
        let mut out = Vec::new();
        while !stop.load(Ordering::SeqCst) {
            let m = store.ops_metrics();
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

pub(crate) fn write_trec(path: &Path, trec: &BTreeMap<String, Vec<(u64, f32)>>, run: &str) -> Result<(), String> {
    let mut s = String::new();
    for (q, hits) in trec {
        for (rank, (id, score)) in hits.iter().enumerate() {
            s.push_str(&format!("{q} Q0 {id} {} {score} {run}\n", rank + 1));
        }
    }
    std::fs::write(path, s).map_err(|e| format!("{}: {e}", path.display()))
}

/// Every test query once, alone: the ranked replies.
async fn quality_pass(ctx: &Ctx) -> Result<BTreeMap<String, Vec<(u64, f32)>>, String> {
    let mut trec = BTreeMap::new();
    for (qid, text) in &ctx.data.queries {
        let page = ctx
            .target
            .store
            .search_semantic(&ctx.target.ns, text, &search_spec(), Pagination::new(1, 100))
            .await
            .map_err(|e| format!("quality pass, query {qid}: {e}"))?;
        let hits = page
            .results
            .into_iter()
            .filter_map(|r| Some((u64::from_be_bytes(r.document_id.as_slice().try_into().ok()?), r.dot_product)))
            .collect();
        trec.insert(qid.clone(), hits);
    }
    Ok(trec)
}

#[derive(Serialize)]
struct CellResult<'a> {
    cell: &'a Cell,
    valid: bool,
    reasons: Vec<String>,
    started_unix: u64,
    /// Writes that brought the store back to the base's size before the cell
    /// (+ inserted, - deleted), unmeasured.
    rebalanced: i64,
    documents_before: i64,
    documents_after: i64,
    /// Texts in the store twice at the cell's end ([`Cursors::repeated_texts`]).
    repeated_texts: u64,
    drain_before_s: Option<f64>,
    drain_after_s: Option<f64>,
    measured_s: f64,
    total_ops_per_s: f64,
    classes: BTreeMap<&'static str, crate::stats::ClassSummary>,
    trickle: BTreeMap<&'static str, crate::stats::ClassSummary>,
    errors: Vec<String>,
    ingestion_lag_ms: Value,
    replay: ReplayStats,
    engine: Counters,
    engine_after: Counters,
    cursors_after: Cursors,
    /// The soak only (the last cell): memory before and after asking the
    /// allocator to return what it keeps free, so a growth can be split into
    /// live allocations and memory freed but kept.
    trim: Option<(engine::Memory, engine::Memory)>,
}

async fn run_cell(ctx: &Ctx, cell: &Cell) -> Result<bool, String> {
    let cfg = &ctx.cfg;
    let max_drain = Duration::from_secs(cfg.drain_max_s);
    let started_unix = now_unix();
    let rebalanced = rebalance(ctx, cell).await?;
    let documents_before = ctx.cursors.lock().documents(ctx.data.base.len());
    eprintln!(
        "[{}] {}: rebalanced by {rebalanced:+} to {documents_before} documents; draining",
        started_unix, cell.name
    );
    let drain_before_s = drain(&ctx.target.store, max_drain).await;
    let before = Counters::take(ctx).await;
    let replay_before = ctx.replay.stats();
    let stop = Arc::new(AtomicBool::new(false));
    let every = if matches!(cell.kind, Kind::Soak { .. }) { 10 } else { 1 };
    let monitor = spawn_monitor(ctx, Duration::from_secs(every), Arc::clone(&stop));

    if cell.warm_s > 0 {
        let warm = run_load(ctx, cell, cell.warm_s, 1).await;
        for t in warm.lag_tasks {
            t.abort();
        }
    }
    let t = Instant::now();
    let load = run_load(ctx, cell, cell.measure_s, 2).await;
    let measured = t.elapsed();
    let drain_after_s = drain(&ctx.target.store, max_drain).await;
    let mut lags: Vec<f64> = Vec::new();
    for h in load.lag_tasks {
        if let Ok(Ok(Some(ms))) = tokio::time::timeout(Duration::from_secs(5), h).await {
            lags.push(ms);
        }
    }
    stop.store(true, Ordering::SeqCst);
    let timeline = monitor.await.unwrap_or_default();
    let after = Counters::take(ctx).await;
    let trim = matches!(cell.kind, Kind::Soak { .. }).then(engine::trim);
    let replay = ctx.replay.stats().since(&replay_before);
    let delta = after.since(&before);

    // Validity.
    let mut reasons = Vec::new();
    let errors = load.measured.errors() + load.trickle.errors();
    if errors > 0 {
        reasons.push(format!("{errors} operation(s) failed"));
    }
    if load.measured.unavailable() + load.trickle.unavailable() > 0 {
        reasons.push("a delete had no document available".into());
    }
    if replay.misses > 0 {
        reasons.push(format!("{} embedding payload(s) missing from the replay table", replay.misses));
    }
    let measured_classes: Vec<usize> = match cell.kind {
        Kind::Single { class } => vec![class],
        _ => (0..8).filter(|&c| ctx.mix[c] > 0.0).collect(),
    };
    for c in measured_classes {
        let n = load.measured.count(c);
        if n < cfg.min_samples {
            reasons.push(format!("{}: {n} operations, fewer than {}", CLASSES[c], cfg.min_samples));
        }
    }
    if drain_before_s.is_none() || drain_after_s.is_none() {
        reasons.push(format!("the vector queue did not drain within {} s", cfg.drain_max_s));
    }
    if delta.puts > 0 && delta.vector_wal_appends == 0 {
        reasons.push("documents were written but the vector worker indexed nothing".into());
    }
    if delta.puts + delta.deletes > 0 && after.checkpoint_offset <= before.checkpoint_offset {
        reasons.push("documents were written but no field-index checkpoint advanced".into());
    }
    let valid = reasons.is_empty();

    let mut sorted = lags.clone();
    sorted.sort_by(f64::total_cmp);
    let pct = |p: f64| {
        sorted
            .get(((p * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len().max(1)) - 1)
            .copied()
    };
    // One copy: a lock taken inside the struct literal would be held to its
    // end, and a second `lock()` there would wait on it forever.
    let cursors = *ctx.cursors.lock();
    let result = CellResult {
        cell,
        valid,
        reasons: reasons.clone(),
        started_unix,
        rebalanced,
        documents_before,
        documents_after: cursors.documents(ctx.data.base.len()),
        repeated_texts: cursors.repeated_texts(ctx.data.deletable.len(), ctx.data.stream.len()),
        drain_before_s,
        drain_after_s,
        measured_s: measured.as_secs_f64(),
        total_ops_per_s: load.measured.completed() as f64 / measured.as_secs_f64(),
        classes: load.measured.summary(measured),
        trickle: load.trickle.summary(measured),
        errors: [load.measured.error_samples(), load.trickle.error_samples()].concat(),
        ingestion_lag_ms: json!({"samples": sorted.len(), "p50": pct(0.5), "p90": pct(0.9), "p99": pct(0.99), "max": sorted.last()}),
        replay,
        engine: delta,
        engine_after: after,
        trim,
        cursors_after: cursors,
    };
    let dir = ctx.out.join("cells");
    write_json(&dir.join(format!("{}.json", cell.name)), &result)?;
    let lines: String = timeline.iter().map(|s| serde_json::to_string(s).unwrap_or_default() + "\n").collect();
    std::fs::write(dir.join(format!("{}.timeline.jsonl", cell.name)), lines).map_err(|e| e.to_string())?;
    if !load.trec.is_empty() {
        write_trec(&dir.join(format!("{}.trec", cell.name)), &load.trec, &cell.name)?;
    }
    eprintln!(
        "[{}] {}: {} — {:.1} ops/s, errors {errors}{}",
        now_unix(),
        cell.name,
        if valid { "valid" } else { "INVALID" },
        result.total_ops_per_s,
        if valid { String::new() } else { format!("; {}", reasons.join("; ")) }
    );
    Ok(valid)
}

fn git_commit() -> String {
    let run = |args: &[&str]| {
        std::process::Command::new("git")
            .args(args)
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default()
    };
    let dirty = run(&["status", "--porcelain", "--untracked-files=no"]);
    format!(
        "{}{}",
        run(&["rev-parse", "--short", "HEAD"]),
        if dirty.is_empty() { "" } else { " (uncommitted changes)" }
    )
}

/// `engine_bench run CONFIG [--smoke]`.
pub async fn run(args: &[String]) -> Result<(), String> {
    let path = args.first().ok_or("usage: engine_bench run CONFIG.json [--smoke]")?;
    let smoke = args.iter().any(|a| a == "--smoke");
    let mut cfg = Config::load(path)?;
    if smoke {
        cfg = cfg.smoke();
    }
    let out = PathBuf::from(&cfg.out);
    std::fs::create_dir_all(out.join("cells")).map_err(|e| format!("{}: {e}", out.display()))?;
    let manifest_path = out.join("manifest.json");
    let mut manifest: Manifest = match std::fs::read(&manifest_path) {
        Ok(b) => serde_json::from_slice(&b).map_err(|e| format!("{}: {e}", manifest_path.display()))?,
        Err(_) => {
            if out.join("engine").exists() {
                return Err(format!("{} holds an engine but no manifest; use a fresh out directory", out.display()));
            }
            engine::copy_snapshot(&cfg.snapshot, &out.join("engine"))?;
            Manifest::default()
        }
    };
    if manifest.complete {
        return Err(format!("{} is complete", out.display()));
    }
    manifest.starts.push(now_unix());
    engine::log_to(&out.join("engine.log"))?;

    let replay = Replay::open(&cfg.table, None)?;
    // The replay table is in this process too: its share of memory is what is
    // resident before the engine opens.
    let memory_before_engine = engine::memory();
    let url = replay.serve().await?;
    let (store, settings): (Arc<DocStore>, EngineSettings) = engine::open(&cfg, &out.join("engine"), &url, None).await?;
    let schema = store
        .list()
        .map_err(|e| e.to_string())?
        .into_iter()
        .find(|s| s["namespace"] == cfg.namespace)
        .ok_or_else(|| format!("no store named {}", cfg.namespace))?;
    let target = Arc::new(Target {
        store: Arc::clone(&store),
        ns: cfg.namespace.clone(),
        ns_id: schema["ns_id"].as_u64().ok_or("schema has no ns_id")? as u32,
        key_type: serde_json::from_value::<KeyType>(schema["key_type"].clone()).map_err(|e| e.to_string())?,
    });
    let data = Arc::new(Data::load(&cfg.data)?);
    let ctx = Ctx {
        mix: weights(&cfg.mix)?,
        trickle_mix: weights(&cfg.trickle_mix)?,
        cursors: Arc::new(Mutex::new(manifest.cursors)),
        out: out.clone(),
        data,
        target,
        replay: Arc::clone(&replay),
        cfg: cfg.clone(),
    };
    if ctx
        .trickle_mix
        .iter()
        .enumerate()
        .any(|(c, w)| *w > 0.0 && !crate::workload::WRITES.contains(&c))
    {
        return Err("trickle_mix may hold only insert, update and delete".into());
    }
    // The store keeps its size: as many deletes as inserts, so a long cell (the
    // soak) does not grow it, and the inserts alive at once (about the
    // deletable base's count) stay fewer than the stream's texts, so no text
    // is in the store twice.
    for (name, w) in [("mix", ctx.mix), ("trickle_mix", ctx.trickle_mix)] {
        if w[1] != w[3] {
            return Err(format!(
                "{name}: insert ({}) and delete ({}) must be equal so the store keeps its size",
                w[1], w[3]
            ));
        }
    }
    if ctx.data.deletable.len() + 1000 > ctx.data.stream.len() {
        return Err(format!(
            "{} deletable base documents against a stream of {}: inserts alive at once could repeat a text",
            ctx.data.deletable.len(),
            ctx.data.stream.len()
        ));
    }
    let cells = plan(&cfg);
    write_json(
        &out.join("run.json"),
        &json!({
            "config": cfg, "smoke": smoke, "commit": git_commit(), "host_cores": std::thread::available_parallelism().map_or(0, |n| n.get()),
            "engine": settings, "plan": cells,
            "data": {"stable": ctx.data.stable.len(), "stable_text": ctx.data.stable_text.len(), "base": ctx.data.base.len(),
                     "stream": ctx.data.stream.len(), "queries": ctx.data.queries.len()},
            "replay": {"table": cfg.table, "vectors": replay.vectors(), "models": replay.models(),
                       "table_bytes": std::fs::metadata(&cfg.table).map_or(0, |m| m.len())},
            "starts": manifest.starts, "memory_before_engine": memory_before_engine,
        }),
    )?;

    let result = run_plan(&ctx, &cells, &mut manifest, &manifest_path).await;
    eprintln!("[{}] shutting down the engine", now_unix());
    store.shutdown().await.map_err(|e| format!("shutdown: {e}"))?;
    result
}

async fn run_plan(ctx: &Ctx, cells: &[Cell], manifest: &mut Manifest, manifest_path: &Path) -> Result<(), String> {
    let save = |m: &mut Manifest, ctx: &Ctx| {
        m.cursors = *ctx.cursors.lock();
        write_json(manifest_path, m)
    };
    if !manifest.primed {
        eprintln!("[{}] draining the vector queue left in the snapshot", now_unix());
        drain(&ctx.target.store, Duration::from_secs(ctx.cfg.drain_max_s))
            .await
            .ok_or("the vector queue did not drain at start")?;
        let before = ctx.replay.stats();
        // Caches every test query's embedding (searches in the run use cached
        // queries only) and records quality at the start.
        let trec = quality_pass(ctx).await?;
        write_trec(&ctx.out.join("quality-start.trec"), &trec, "start")?;
        let misses = ctx.replay.stats().since(&before).misses;
        if misses > 0 {
            return Err(format!("{misses} query payload(s) missing from the replay table"));
        }
        manifest.primed = true;
        save(manifest, ctx)?;
    }
    for cell in cells {
        if manifest.cells.get(&cell.name).map(String::as_str) == Some("done") {
            continue;
        }
        let valid = run_cell(ctx, cell).await?;
        manifest.cells.insert(cell.name.clone(), if valid { "done" } else { "invalid" }.into());
        save(manifest, ctx)?;
        if !valid {
            return Err(format!(
                "cell {} is invalid (see cells/{}.json); the run stopped there",
                cell.name, cell.name
            ));
        }
    }
    eprintln!("[{}] quality at rest", now_unix());
    drain(&ctx.target.store, Duration::from_secs(ctx.cfg.drain_max_s))
        .await
        .ok_or("the vector queue did not drain before quality at rest")?;
    let trec = quality_pass(ctx).await?;
    write_trec(&ctx.out.join("quality-rest.trec"), &trec, "rest")?;
    manifest.complete = true;
    save(manifest, ctx)?;
    eprintln!("[{}] run complete", now_unix());
    Ok(())
}
