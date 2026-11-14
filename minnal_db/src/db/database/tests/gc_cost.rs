//! What a value-log GC pass costs, and what it costs the writers running beside
//! it. `#[ignore]`d: a measurement, run by hand in release on a real disk
//! (WAL fsyncs are most of a write's cost there):
//!
//! `TMPDIR=/var/tmp cargo test --release -p minnal_db --lib -- --ignored gc_cost_under_writers --nocapture`

use super::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

const KEYS: u64 = 300_000;
const VALUE_LEN: usize = 1024;
const WRITERS: usize = 8;

fn key_of(i: u64) -> Vec<u8> {
    format!("{:016x}", i.wrapping_mul(0x9E37_79B9_7F4A_7C15)).into_bytes()
}

/// Overwrite two keys in three, `rounds` times, without the WAL: garbage in
/// sealed segments for GC to collect.
fn make_garbage(db: &Database, ns: NamespaceId, rounds: u8) {
    for round in 0..rounds {
        for i in (0..KEYS).filter(|i| i % 3 != 0) {
            db.put_ns_no_wal(ns, &key_of(i), &[round; VALUE_LEN]).unwrap();
        }
    }
    let store = db.get_store(ns).unwrap();
    store.flush_memtable_to_level0().unwrap();
    // Shadowed records in L0 count as garbage once the merge drops them.
    store.compact_lsm().unwrap();
    store.sync_value_log().unwrap();
}

/// `WRITERS` threads doing WAL-backed puts of random keys until `stop`; each
/// returns its write latencies.
fn spawn_writers(db: &Arc<Database>, ns: NamespaceId, stop: &Arc<AtomicBool>) -> Vec<std::thread::JoinHandle<Vec<(Instant, Duration)>>> {
    (0..WRITERS)
        .map(|w| {
            let (db, stop) = (Arc::clone(db), Arc::clone(stop));
            std::thread::spawn(move || {
                let mut rng = 0x2545_F491_4F6C_DD1Du64 ^ (w as u64 + 1).wrapping_mul(0x9E37_79B9);
                let mut latencies = Vec::new();
                while !stop.load(Ordering::Relaxed) {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    let started = Instant::now();
                    db.put_ns(ns, &key_of(rng % KEYS), &[0xEE; VALUE_LEN]).unwrap();
                    latencies.push((started, started.elapsed()));
                }
                latencies
            })
        })
        .collect()
}

fn summary(mut latencies: Vec<Duration>, window: Duration) -> String {
    latencies.sort();
    let at = |p: f64| latencies[((latencies.len() as f64 * p) as usize).min(latencies.len() - 1)];
    format!(
        "{} writes ({:.0}/s), p50 {:?}, p99 {:?}, p99.9 {:?}, max {:?}",
        latencies.len(),
        latencies.len() as f64 / window.as_secs_f64(),
        at(0.50),
        at(0.99),
        at(0.999),
        latencies.last().unwrap()
    )
}

#[test]
#[ignore]
fn gc_cost_under_writers() {
    let dir = TempDir::new().unwrap();
    let mut config = create_db_config();
    config.segment_size_bytes = 4 * 1024 * 1024;
    let db = Arc::new(Database::open(dir.path(), config).unwrap());
    let ns = db.create_namespace("gc_cost").unwrap();
    let store = db.get_store(ns).unwrap();
    for i in 0..KEYS {
        db.put_ns_no_wal(ns, &key_of(i), &[0x11; VALUE_LEN]).unwrap();
    }

    // GC alone, no writers.
    make_garbage(&db, ns, 3);
    let started = Instant::now();
    let alone = store.garbage_collect_with_thresholds(10.0, 10.0).unwrap();
    let alone_time = started.elapsed();
    println!(
        "GC alone: {:.1} MiB reclaimed in {alone_time:?} ({:.0} MiB/s)",
        alone.bytes_reclaimed as f64 / 1048576.0,
        alone.bytes_reclaimed as f64 / 1048576.0 / alone_time.as_secs_f64()
    );

    // Writers alone: the latency baseline.
    let stop = Arc::new(AtomicBool::new(false));
    let writers = spawn_writers(&db, ns, &stop);
    let window = Duration::from_secs(5);
    std::thread::sleep(window);
    stop.store(true, Ordering::Relaxed);
    let baseline: Vec<Duration> = writers.into_iter().flat_map(|w| w.join().unwrap()).map(|(_, d)| d).collect();
    println!("writers alone: {}", summary(baseline, window));

    // GC with writers running beside it, latencies taken over the pass only.
    make_garbage(&db, ns, 3);
    let stop = Arc::new(AtomicBool::new(false));
    let writers = spawn_writers(&db, ns, &stop);
    std::thread::sleep(Duration::from_millis(500)); // writers up to speed
    let started = Instant::now();
    let during = store.garbage_collect_with_thresholds(10.0, 10.0).unwrap();
    let gc_time = started.elapsed();
    stop.store(true, Ordering::Relaxed);
    // Only writes that started while the pass ran.
    let with_gc: Vec<Duration> = writers
        .into_iter()
        .flat_map(|w| w.join().unwrap())
        .filter(|(at, _)| *at >= started)
        .map(|(_, d)| d)
        .collect();
    println!(
        "GC with writers: {:.1} MiB reclaimed in {gc_time:?} ({:.0} MiB/s)",
        during.bytes_reclaimed as f64 / 1048576.0,
        during.bytes_reclaimed as f64 / 1048576.0 / gc_time.as_secs_f64()
    );
    println!("writers during GC: {}", summary(with_gc, gc_time));

    for i in (0..KEYS).step_by(7) {
        assert!(db.get_ns(ns, &key_of(i)).unwrap().is_some(), "GC lost a key");
    }
}
