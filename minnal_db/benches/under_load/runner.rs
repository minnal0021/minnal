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
//! 3. `open-{rate}-r{k}`: operations from `mix` arriving at `rate` a second
//!    (each of `open_rates`) on a Poisson schedule, timed from their scheduled
//!    arrival (see `open_loop`).
//! 4. `soak-{n}`: `soak_clients` clients for `soak_s`, sampled every 10 s.
//!
//! `single_s` or `soak_s` set to 0 leaves those cells out, for a run that
//! compares builds on the concurrent cells only.
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

use minnal_db::doc_store::schema::KeyType;
use minnal_db::doc_store::{DocStore, Pagination};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::counters::{Counters, spawn_monitor};
use crate::engine::{self, Config, EngineSettings};
use crate::load::{rebalance, run_load};
use crate::replay::{Replay, ReplayStats};
use crate::workload::{CLASSES, Cursors, Data, Target, search_spec, weights};

#[derive(Clone, Copy, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum Kind {
    Single { class: usize },
    Closed { clients: usize },
    Soak { clients: usize },
    Open { rate: u32 },
}

#[derive(Clone, Serialize)]
pub(crate) struct Cell {
    pub(crate) name: String,
    #[serde(flatten)]
    pub(crate) kind: Kind,
    pub(crate) measure_s: u64,
    pub(crate) warm_s: u64,
}

fn plan(cfg: &Config) -> Vec<Cell> {
    let mut cells = Vec::new();
    let cell = |name: String, kind, measure_s| Cell {
        name,
        kind,
        measure_s,
        warm_s: cfg.warm_s,
    };
    if cfg.single_s > 0 {
        for r in 1..=cfg.repeats {
            for (class, c) in CLASSES.iter().enumerate() {
                cells.push(cell(format!("single-{c}-r{r}"), Kind::Single { class }, cfg.single_s));
            }
        }
    }
    for r in 1..=cfg.repeats {
        for &clients in &cfg.clients {
            cells.push(cell(format!("closed-{clients:02}-r{r}"), Kind::Closed { clients }, cfg.concurrent_s));
        }
    }
    for r in 1..=cfg.repeats {
        for &rate in &cfg.open_rates {
            cells.push(cell(format!("open-{rate:05}-r{r}"), Kind::Open { rate }, cfg.concurrent_s));
        }
    }
    if cfg.soak_s > 0 {
        cells.push(cell(
            format!("soak-{:02}", cfg.soak_clients),
            Kind::Soak { clients: cfg.soak_clients },
            cfg.soak_s,
        ));
    }
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

pub(crate) fn now_unix() -> u64 {
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

pub(crate) struct Ctx {
    pub(crate) cfg: Config,
    pub(crate) out: PathBuf,
    pub(crate) data: Arc<Data>,
    pub(crate) target: Arc<Target>,
    pub(crate) replay: Arc<Replay>,
    pub(crate) cursors: Arc<Mutex<Cursors>>,
    pub(crate) mix: [f64; 8],
    pub(crate) trickle_mix: [f64; 8],
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
                .filter(|e| matches!(e.kind, minnal_db::semantic_search::vector_kv::QueueEntryKind::Clear))
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
    /// Open-loop cells only: how the arrivals kept to their schedule, and
    /// whether the rate saturated the engine.
    #[serde(skip_serializing_if = "Option::is_none")]
    open: Option<crate::open_loop::OpenStats>,
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
    // An open-loop cell's rates are over its arrival window; operations still
    // in flight at its end finish after it.
    let measured = match cell.kind {
        Kind::Open { .. } => Duration::from_secs(cell.measure_s),
        _ => t.elapsed(),
    };
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
        open: load.open.clone(),
    };
    let dir = ctx.out.join("cells");
    write_json(&dir.join(format!("{}.json", cell.name)), &result)?;
    let lines: String = timeline.iter().map(|s| serde_json::to_string(s).unwrap_or_default() + "\n").collect();
    std::fs::write(dir.join(format!("{}.timeline.jsonl", cell.name)), lines).map_err(|e| e.to_string())?;
    if !load.trec.is_empty() {
        write_trec(&dir.join(format!("{}.trec", cell.name)), &load.trec, &cell.name)?;
    }
    let saturated = match &load.open {
        Some(o) if o.saturated => format!(" — SATURATED: {} of {} arrivals shed", o.shed, o.scheduled),
        _ => String::new(),
    };
    eprintln!(
        "[{}] {}: {} — {:.1} ops/s, errors {errors}{}{saturated}",
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
        ns_id: serde_json::from_value(schema["ns_id"].clone()).map_err(|e| format!("schema has no ns_id: {e}"))?,
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
            "config": cfg, "smoke": smoke, "commit": git_commit(), "allocator": crate::ALLOCATOR, "host_cores": std::thread::available_parallelism().map_or(0, |n| n.get()),
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
