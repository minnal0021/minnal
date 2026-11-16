//! Generating load: the closed-loop clients and the write trickle beside a
//! single-operation cell, the ingestion-lag probes, and the unmeasured
//! writes that bring the store back to its size before a cell.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::runner::{Cell, Ctx, Kind};
use crate::stats::Recording;
use crate::workload::{CLASSES, Generator, Op, Outcome, Target, execute};

/// Shared by every generator of a run: one hot set of `get` ids.
pub(crate) const HOT_SEED: u64 = 0x005e_ed0f_ae71;

pub(crate) fn seed_of(name: &str, salt: u64) -> u64 {
    // FNV-1a: the same cell always sends the same operations.
    name.bytes()
        .fold(0xcbf2_9ce4_8422_2325u64 ^ salt, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
}

/// What one stretch of load produced.
#[derive(Default)]
pub(crate) struct Load {
    pub(crate) measured: Recording,
    pub(crate) trickle: Recording,
    /// First ranked reply per test query.
    pub(crate) trec: BTreeMap<String, Vec<(u64, f32)>>,
    pub(crate) lag_tasks: Vec<tokio::task::JoinHandle<Option<f64>>>,
}

/// After an insert is acknowledged, poll its queue entry until it is gone: the
/// time until the document is searchable.
pub(crate) fn spawn_lag(target: &Arc<Target>, id: u64, acked: Instant) -> tokio::task::JoinHandle<Option<f64>> {
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

/// Deletes per second a delete-only cell's starting pool covers, for its whole
/// warm-up and measurement. Deletes remove the deletable base and then earlier
/// inserts, so a cell that starts at the base's size can delete only about
/// `deletable` documents (10,000 in the FiQA data) before it runs out, and
/// fast deletes do that within one cell. The fastest delete-only rate measured
/// is about 120 a second; this leaves three times that.
const DELETE_CELL_MAX_RATE: u64 = 400;

/// Documents a cell starts with beyond the base: enough for a delete-only
/// cell's deletes ([`DELETE_CELL_MAX_RATE`]), none for any other cell.
fn extra_documents(cell: &Cell) -> i64 {
    match cell.kind {
        Kind::Single { class } if CLASSES[class] == "delete" => ((cell.warm_s + cell.measure_s) * DELETE_CELL_MAX_RATE) as i64,
        _ => 0,
    }
}

/// Bring the store back to the base's size, unmeasured, before a cell: a
/// single-class insert or delete cell moves it by thousands, and later cells
/// would otherwise run on a different store. A delete-only cell starts larger
/// ([`extra_documents`]), so it cannot run out of documents to delete. Inserts
/// or deletes in order, 16 at a time, as the load would; returns how many
/// (+ inserted, - deleted).
pub(crate) async fn rebalance(ctx: &Ctx, cell: &Cell) -> Result<i64, String> {
    let base = ctx.data.base.len();
    let gap = ctx.cursors.lock().documents(base) - (base as i64 + extra_documents(cell));
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
pub(crate) async fn run_load(ctx: &Ctx, cell: &Cell, secs: u64, salt: u64) -> Load {
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
