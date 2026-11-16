//! Fixed-arrival-rate (open-loop) load: operations drawn from `mix` arrive on a
//! seeded Poisson schedule, `rate` a second on average, whatever the engine is
//! doing, and each is timed from its **scheduled** arrival. Time an operation
//! spends waiting (behind the dispatcher, the runtime or the engine's own
//! queues, such as the search gate) is in its latency, so nothing is hidden by
//! coordinated omission.
//!
//! The closed-loop cells cannot show this. There a client sends its next
//! operation only when its last returns, so making one class faster frees its
//! clients to queue somewhere else: after the field-index worker made writes
//! faster, more of the same clients waited at the search gate and search got
//! slower at a fixed client count, although each search was served faster. At
//! a fixed arrival rate the offered load is the same for every build, and each
//! class's latency is its own.
//!
//! Past the engine's capacity there is no steady state: the operations in
//! flight grow without bound. When they reach [`MAX_IN_FLIGHT`] the dispatcher
//! sheds arrivals (sends nothing, counts them) until they fall again, and the
//! cell is reported **saturated**: its latencies are then a lower bound, not a
//! measurement at that rate.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use serde::Serialize;
use tokio::sync::mpsc::UnboundedSender;

use crate::load::{HOT_SEED, Load, seed_of, spawn_lag};
use crate::runner::{Cell, Ctx};
use crate::workload::{Generator, Op, Outcome, Rng, Target, execute};

/// Operations in flight at which arrivals are shed. At the rates measured
/// (about 2,000 a second at most, a few ms each) a steady state holds a few
/// dozen; this many means the queues are growing.
pub(crate) const MAX_IN_FLIGHT: usize = 1024;

/// How a fixed-arrival-rate stretch kept to its schedule.
#[derive(Serialize, Clone)]
pub(crate) struct OpenStats {
    /// Target arrivals per second.
    pub rate: u32,
    /// Arrivals scheduled in the stretch: sent, shed, or a delete with nothing
    /// to delete.
    pub scheduled: u64,
    /// Arrivals sent.
    pub sent: u64,
    /// Arrivals not sent because [`MAX_IN_FLIGHT`] operations were in flight.
    pub shed: u64,
    /// `shed > 0`: the rate is past what the engine sustains.
    pub saturated: bool,
    /// Most operations in flight at once.
    pub max_in_flight: usize,
    /// Mean operations in flight (Little's law: rate × mean latency).
    pub mean_in_flight: f64,
    /// From an operation's scheduled arrival to the moment it started: the
    /// dispatcher's sleep and the runtime, not the engine. Already inside the
    /// latencies; reported so a late harness is visible.
    pub dispatch_lag_p50_ms: f64,
    pub dispatch_lag_p99_ms: f64,
    pub dispatch_lag_max_ms: f64,
}

/// One finished operation, sent to the collector.
struct Done {
    class: usize,
    op: Op,
    result: Result<Outcome, String>,
    /// From the scheduled arrival to the reply.
    took: Duration,
    /// From the scheduled arrival to the start.
    dispatch_lag: Duration,
    finished: Instant,
}

/// What the dispatcher counted.
struct Dispatched {
    scheduled: u64,
    sent: u64,
    shed: u64,
    max_in_flight: usize,
    /// Deletes with nothing to delete, per class.
    unavailable: [u64; 8],
}

/// Run `mix` at `rate` arrivals a second for `secs`.
pub(crate) async fn run_open(ctx: &Ctx, cell: &Cell, rate: u32, secs: u64, salt: u64) -> Load {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Done>();
    let in_flight = Arc::new(AtomicUsize::new(0));
    // Sum over operations of their time in flight, in µs: mean in flight is
    // this over the stretch's length.
    let busy_us = Arc::new(AtomicU64::new(0));

    // The collector records off the operations' path.
    let (target, lag_every) = (Arc::clone(&ctx.target), ctx.cfg.lag_every);
    let collector = tokio::spawn(async move {
        let mut load = Load::default();
        let mut lags: Vec<u32> = Vec::new();
        let mut inserts = 0u64;
        while let Some(d) = rx.recv().await {
            lags.push(d.dispatch_lag.as_micros().min(u32::MAX as u128) as u32);
            if let Op::Insert(doc) = &d.op {
                inserts += 1;
                if lag_every > 0 && inserts.is_multiple_of(lag_every) {
                    load.lag_tasks.push(spawn_lag(&target, doc.id, d.finished));
                }
            }
            load.record(d.class, &d.op, d.result, d.took);
        }
        (load, lags)
    });

    // The dispatcher is an OS thread: Tokio's timer ticks in whole
    // milliseconds, so at 2,000 arrivals a second it would send them in bunches
    // up to a millisecond late, longer than a `get` takes. A thread's sleep is
    // within tens of microseconds, and it spins no core.
    let dispatcher = Dispatcher {
        generator: Generator::new(Arc::clone(&ctx.data), HOT_SEED, seed_of(&cell.name, salt), Arc::clone(&ctx.cursors)),
        arrivals: Rng(seed_of(&cell.name, salt ^ 0xa771_7a15)),
        mix: ctx.mix,
        target: Arc::clone(&ctx.target),
        runtime: tokio::runtime::Handle::current(),
        tx,
        in_flight: Arc::clone(&in_flight),
        busy_us: Arc::clone(&busy_us),
    };
    let (done_tx, done_rx) = tokio::sync::oneshot::channel();
    std::thread::Builder::new()
        .name("open-loop-dispatch".into())
        .spawn(move || {
            let _ = done_tx.send(dispatcher.run(rate, secs));
        })
        .expect("spawn the open-loop dispatcher");
    let d = done_rx.await.expect("the open-loop dispatcher died");
    // The dispatcher dropped its sender; the collector ends once every
    // operation still in flight has reported.
    let (mut load, mut lags) = collector.await.unwrap_or_default();
    for (c, n) in load.measured.0.iter_mut().zip(d.unavailable) {
        c.unavailable += n;
    }
    lags.sort_unstable();
    let pct = |p: f64| {
        if lags.is_empty() {
            return 0.0;
        }
        lags[((p * lags.len() as f64).ceil() as usize).clamp(1, lags.len()) - 1] as f64 / 1e3
    };
    load.open = Some(OpenStats {
        rate,
        scheduled: d.scheduled,
        sent: d.sent,
        shed: d.shed,
        saturated: d.shed > 0,
        max_in_flight: d.max_in_flight,
        mean_in_flight: busy_us.load(Ordering::Relaxed) as f64 / 1e6 / secs as f64,
        dispatch_lag_p50_ms: pct(0.5),
        dispatch_lag_p99_ms: pct(0.99),
        dispatch_lag_max_ms: lags.last().map_or(0.0, |&x| x as f64 / 1e3),
    });
    load
}

/// Sends one stretch's arrivals, on its own thread.
struct Dispatcher {
    generator: Generator,
    arrivals: Rng,
    mix: [f64; 8],
    target: Arc<Target>,
    runtime: tokio::runtime::Handle,
    tx: UnboundedSender<Done>,
    in_flight: Arc<AtomicUsize>,
    busy_us: Arc<AtomicU64>,
}

impl Dispatcher {
    /// Send arrivals at `rate` a second for `secs`.
    fn run(mut self, rate: u32, secs: u64) -> Dispatched {
        let mut d = Dispatched {
            scheduled: 0,
            sent: 0,
            shed: 0,
            max_in_flight: 0,
            unavailable: [0; 8],
        };
        let end = Instant::now() + Duration::from_secs(secs);
        let mut next = Instant::now();
        loop {
            // Exponential gaps: a Poisson process at `rate`.
            next += Duration::from_secs_f64(-(1.0 - self.arrivals.unit()).ln() / rate as f64);
            if next >= end {
                return d;
            }
            let now = Instant::now();
            if next > now {
                std::thread::sleep(next - now);
            }
            d.scheduled += 1;
            let in_flight = self.in_flight.load(Ordering::SeqCst);
            if in_flight >= MAX_IN_FLIGHT {
                d.shed += 1;
                continue;
            }
            let class = self.generator.class(&self.mix);
            let Some(op) = self.generator.op(class) else {
                // A delete with nothing to delete: invalidates the cell, as in
                // the closed loop.
                d.unavailable[class] += 1;
                continue;
            };
            d.sent += 1;
            d.max_in_flight = d.max_in_flight.max(in_flight + 1);
            self.in_flight.fetch_add(1, Ordering::SeqCst);
            self.spawn(class, op, next);
        }
    }

    /// Run `op`, scheduled at `at`, on the runtime.
    fn spawn(&self, class: usize, op: Op, at: Instant) {
        let (target, tx) = (Arc::clone(&self.target), self.tx.clone());
        let (in_flight, busy_us) = (Arc::clone(&self.in_flight), Arc::clone(&self.busy_us));
        self.runtime.spawn(async move {
            let dispatch_lag = at.elapsed();
            let result = execute(&target, &op).await;
            let took = at.elapsed();
            in_flight.fetch_sub(1, Ordering::SeqCst);
            busy_us.fetch_add(took.as_micros() as u64, Ordering::Relaxed);
            let _ = tx.send(Done {
                class,
                op,
                result,
                took,
                dispatch_lag,
                finished: Instant::now(),
            });
        });
    }
}
