//! Randomised power losses against concurrent writers and background work, and
//! process kills of a child that writes until killed.

use super::*;

// ── Randomised power-loss stress ─────────────────────────────────────────────

/// A tiny deterministic generator (no extra dependency).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// What a key should read after a power loss: its last acknowledged value,
/// or — if a write to it was in flight — possibly that write's value.
/// A key in a namespace, and a value (`None`: deleted).
type NsKey = (NamespaceId, Vec<u8>);
type Value = Option<Vec<u8>>;

#[derive(Default)]
struct Expected {
    acked: HashMap<NsKey, Value>,
    /// Per writer, the operation it has started and not yet seen return.
    in_flight: HashMap<usize, (NsKey, Value)>,
}

/// Writers, a background thread doing what the workers do (flush, WAL GC,
/// field-index checkpoints, value-log GC, LSM compaction, TTL expiry, field
/// reindexing), small memtables, small WAL and value-log segments, so flushes,
/// rotations, retirement, relocation and expiry all happen often. At random
/// moments every thread is stopped — writers either between operations or
/// inside a commit — and the power-loss image is checked.
///
/// The `ttl` namespace is expired continuously (a TTL of zero), so its keys
/// read as the last acknowledged value or absent; its field index must still
/// agree with what it stores.
fn power_loss_stress(seed: u64, images: usize) {
    const WRITERS: usize = 8;
    const KEYS_PER_WRITER: u64 = 40;
    let dir = TempDir::new().unwrap();
    let mut config = create_db_config();
    config.lsm_config.skip_list_capacity = 64;
    config.segment_size_bytes = 64 * 1024;
    let db = Database::open_with_wal_segment_size(dir.path(), config, 4096).unwrap();
    let docs = db.create_namespace("docs").unwrap();
    let other = db.create_namespace("other").unwrap();
    let ttl = db.create_namespace("ttl").unwrap();
    let docs_status = activate_status_index(&db, docs);
    activate_status_index(&db, ttl);

    // Each writer owns its keys, and no two writers' keys share a key-stripe
    // lock (stripes are per namespace): a writer stopped inside a commit holds its key's stripe, and a
    // second writer waiting on that stripe could never reach a stopping point.
    let mut taken = std::collections::HashSet::new();
    let keys: Vec<Vec<(NamespaceId, Vec<u8>)>> = (0..WRITERS)
        .map(|w| {
            let mut mine = Vec::new();
            let mut i = 0u64;
            while (mine.len() as u64) < KEYS_PER_WRITER {
                let ns = match i % 5 {
                    0 | 3 => other,
                    4 => ttl,
                    _ => docs,
                };
                let key = format!("w{w}/k{i}").into_bytes();
                if taken.insert((ns, db.stripe_of(ns, &key))) {
                    mine.push((ns, key));
                }
                i += 1;
            }
            mine
        })
        .collect();
    let expected = Mutex::new(Expected::default());
    let pause = AtomicBool::new(false);
    let parked = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let park = |pause: &AtomicBool, parked: &AtomicUsize| {
        if pause.load(Ordering::SeqCst) {
            parked.fetch_add(1, Ordering::SeqCst);
            while pause.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_micros(200));
            }
            parked.fetch_sub(1, Ordering::SeqCst);
        }
    };

    std::thread::scope(|s| {
        for (w, mine) in keys.iter().enumerate() {
            let (db, expected, pause, parked, stop) = (&db, &expected, &pause, &parked, &stop);
            s.spawn(move || {
                let mut rng = Rng(seed ^ (w as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                while !stop.load(Ordering::SeqCst) {
                    park(pause, parked);
                    let (ns, key) = mine[rng.below(KEYS_PER_WRITER) as usize].clone();
                    let value = if rng.below(5) == 0 {
                        None
                    } else {
                        let status = if rng.below(2) == 0 { "active" } else { "idle" };
                        Some(format!(r#"{{"status":"{status}","n":{}}}"#, rng.next()).into_bytes())
                    };
                    expected.lock().in_flight.insert(w, ((ns, key.clone()), value.clone()));
                    let result = match (&value, rng.below(4)) {
                        (Some(v), 0) => db.merge_ns(ns, &key, v, |_, op| Ok(Some(op.to_vec()))).map(|_| ()),
                        (Some(v), _) => db.put_ns(ns, &key, v),
                        (None, _) => db.delete_ns(ns, &key),
                    };
                    result.unwrap();
                    let mut e = expected.lock();
                    e.acked.insert((ns, key), value);
                    e.in_flight.remove(&w);
                }
            });
        }
        {
            let (db, pause, parked, stop, keys) = (&db, &pause, &parked, &stop, &keys);
            s.spawn(move || {
                let mut i = 0u64;
                let mut rng = Rng(seed ^ 0xBAC6_0000);
                let stores = [docs, other, ttl].map(|ns| db.get_store(ns).unwrap());
                while !stop.load(Ordering::SeqCst) {
                    park(pause, parked);
                    match i % 7 {
                        0 => {
                            db.flush_namespaces_pinning_wal();
                        }
                        1 => {
                            db.garbage_collect_wal().unwrap();
                        }
                        2 => {
                            db.run_index_checkpoint().unwrap();
                        }
                        3 => {
                            for store in &stores {
                                store.garbage_collect_with_thresholds(0.0, 0.0).unwrap();
                            }
                        }
                        4 => {
                            for store in &stores {
                                if rng.below(4) == 0 {
                                    store.compact_lsm().unwrap();
                                } else {
                                    store.compact_lsm_due().unwrap();
                                }
                            }
                        }
                        5 => {
                            // As the TTL worker does (`run_ttl_pass`).
                            stores[2]
                                .expire_records_with(Duration::ZERO, 20, |_| db.note_no_wal_index_exposure(ttl))
                                .unwrap();
                        }
                        _ => {
                            for _ in 0..8 {
                                let mine = &keys[rng.below(WRITERS as u64) as usize];
                                let (ns, key) = &mine[rng.below(KEYS_PER_WRITER) as usize];
                                if *ns == docs {
                                    db.reindex_field(docs, docs_status, key).unwrap();
                                }
                            }
                        }
                    }
                    i += 1;
                    std::thread::sleep(Duration::from_millis(2));
                }
            });
        }

        // If a check fails, release every thread so the scope can end and the
        // panic surface, instead of leaving them parked for ever.
        struct Release<'a>(&'a AtomicBool, &'a AtomicBool, &'a Database);
        impl Drop for Release<'_> {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
                self.1.store(false, Ordering::SeqCst);
                self.2.wal.freeze_commits(false);
            }
        }
        let _release = Release(&stop, &pause, &db);

        let mut rng = Rng(seed);
        for round in 0..images {
            std::thread::sleep(Duration::from_millis(5 + rng.below(40)));
            // Stop the world: writers park between operations or wait inside
            // a commit (appended, not acknowledged); the maintenance thread
            // parks between passes.
            db.wal.freeze_commits(true);
            pause.store(true, Ordering::SeqCst);
            wait_for("every thread to stop", || {
                parked.load(Ordering::SeqCst) + db.wal.frozen_writers() == WRITERS + 1
            });
            let (_image_dir, image) = open_image(&db);
            check_image(&image, &expected.lock(), seed, round);
            drop(image);
            pause.store(false, Ordering::SeqCst);
            db.wal.freeze_commits(false);
        }
        stop.store(true, Ordering::SeqCst);
    });
}

fn check_image(image: &Database, expected: &Expected, seed: u64, round: usize) {
    let in_flight: HashMap<&(NamespaceId, Vec<u8>), &Option<Vec<u8>>> = expected.in_flight.values().map(|(k, v)| (k, v)).collect();
    let ttl = image.get_namespace_id("ttl").unwrap();
    for ((ns, key), want) in &expected.acked {
        // Namespace ids are kept in the copied registry.
        let got = image.get_ns(*ns, key).unwrap();
        let ok = got == *want
            || in_flight.get(&(*ns, key.clone())).is_some_and(|v| got == **v)
            // An expiring namespace may have expired it.
            || (*ns == ttl && got.is_none());
        let text = String::from_utf8_lossy(key);
        assert!(ok, "seed {seed} image {round}: ns {ns} {text} reads {got:?}, acknowledged {want:?}");
    }
    // The field index, rebuilt from its checkpoint and the WAL, agrees with the
    // documents themselves.
    for name in ["docs", "ttl"] {
        let ns = image.get_namespace_id(name).unwrap();
        let field = activate_status_index(image, ns);
        // TTL's deletes are in no WAL, so a crash can leave its index behind;
        // that must be recorded as a gap, and repair must bring it level.
        if name == "ttl" && image.index_manager.read_gap(ns, field).is_some() {
            image.repair_field_index(ns, field).unwrap();
        }
        let mut indexed = image.query_keys(ns, "status = \"active\"").unwrap().keys;
        indexed.sort();
        let mut stored: Vec<Vec<u8>> = image
            .get_store(ns)
            .unwrap()
            .scan_prefix_batch(b"")
            .unwrap()
            .into_iter()
            .filter(|(_, v)| v.windows(17).any(|w| w == br#""status":"active""#))
            .map(|(k, _)| k)
            .collect();
        stored.sort();
        assert_eq!(
            indexed, stored,
            "seed {seed} image {round}: {name}'s field index disagrees with the documents"
        );
    }
    // Every layer holds a newer version than the layers below it.
    for name in ["docs", "other", "ttl"] {
        let ns = image.get_namespace_id(name).unwrap();
        let violations = image.get_store(ns).unwrap().lsm.layer_order_violations().unwrap();
        assert!(violations.is_empty(), "seed {seed} image {round}: {name}: {violations:?}");
    }
}

#[test]
fn every_acknowledged_write_survives_random_power_losses() {
    for seed in [0x5EED_0001u64, 0x5EED_0002] {
        power_loss_stress(seed, 6);
    }
}

/// The long variant: `MINNAL_POWER_LOSS_SEEDS` seeds (default 20) of 20
/// images each.
#[test]
#[ignore = "long power-loss stress; run with --ignored"]
fn every_acknowledged_write_survives_many_random_power_losses() {
    let seeds: u64 = std::env::var("MINNAL_POWER_LOSS_SEEDS").ok().and_then(|s| s.parse().ok()).unwrap_or(20);
    for seed in 0..seeds {
        power_loss_stress(0xC0FFEE ^ seed.wrapping_mul(0x2545_F491_4F6C_DD1D), 20);
    }
}

// ── Process kill ─────────────────────────────────────────────────────────────
//
// A real child process writes concurrently (with flushes, WAL GC and index
// checkpoints running) and is killed with SIGKILL at a random moment; the parent
// reopens the database and checks it, over several kill-and-recover cycles on
// the same directory. A kill keeps the page cache, so this checks recovery and
// retirement end to end rather than fsync coverage (the power-loss tests above
// do that).

const KILL_DIR_VAR: &str = "MINNAL_KILL_TEST_DIR";
const KILL_WRITERS: usize = 6;

/// The child half of [`acknowledged_writes_survive_repeated_kills`]: writes
/// until killed. Each writer logs `I <key> <value|->` before an operation and
/// `A <key> <value|->` once it returned, in its own file.
#[test]
#[ignore = "spawned as a child process by acknowledged_writes_survive_repeated_kills"]
fn kill_child_writes_until_killed() {
    use std::io::Write;
    let Ok(dir) = std::env::var(KILL_DIR_VAR) else {
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let mut config = create_db_config();
    config.lsm_config.skip_list_capacity = 64;
    let db = Database::open_with_wal_segment_size(&dir.join("db"), config, 4096).unwrap();
    let ns = db.get_namespace_id("docs").unwrap_or_else(|| db.create_namespace("docs").unwrap());
    activate_status_index(&db, ns);
    let round: u64 = std::env::var("MINNAL_KILL_ROUND").unwrap().parse().unwrap();

    std::thread::scope(|s| {
        for w in 0..KILL_WRITERS {
            let (db, dir) = (&db, &dir);
            s.spawn(move || {
                let mut log = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(dir.join(format!("w{w}.log")))
                    .unwrap();
                let mut rng = Rng((round + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (w as u64 + 1));
                let mut n = 0u64;
                loop {
                    n += 1;
                    let key = format!("w{w}-k{}", rng.below(30));
                    let value = (rng.below(5) != 0).then(|| {
                        format!(
                            r#"{{"status":"{}","r":{round},"n":{n}}}"#,
                            if rng.below(2) == 0 { "active" } else { "idle" }
                        )
                    });
                    let shown = value.clone().unwrap_or_else(|| "-".into());
                    log.write_all(format!("I {key} {shown}\n").as_bytes()).unwrap();
                    match (&value, rng.below(4)) {
                        (Some(v), 0) => db.merge_ns(ns, key.as_bytes(), v.as_bytes(), |_, op| Ok(Some(op.to_vec()))).map(|_| ()),
                        (Some(v), _) => db.put_ns(ns, key.as_bytes(), v.as_bytes()),
                        (None, _) => db.delete_ns(ns, key.as_bytes()),
                    }
                    .unwrap();
                    log.write_all(format!("A {key} {shown}\n").as_bytes()).unwrap();
                }
            });
        }
        s.spawn(|| {
            let mut i = 0u64;
            loop {
                i += 1;
                match i % 3 {
                    0 => {
                        db.flush_namespaces_pinning_wal();
                    }
                    1 => {
                        db.garbage_collect_wal().unwrap();
                    }
                    _ => {
                        db.run_index_checkpoint().unwrap();
                    }
                }
                std::thread::sleep(Duration::from_millis(3));
            }
        });
    });
}

/// For one key: its last acknowledged value (`None` if never acknowledged;
/// `Some(None)` for a delete), and the values of operations that started after
/// it (killed before they returned; each may or may not be in).
type KeyLog = (Option<Option<String>>, Vec<Option<String>>);

/// Each key's [`KeyLog`], from the child's logs.
fn read_kill_logs(dir: &std::path::Path) -> HashMap<String, KeyLog> {
    let mut keys: HashMap<String, KeyLog> = HashMap::new();
    for w in 0..KILL_WRITERS {
        let Ok(text) = std::fs::read_to_string(dir.join(format!("w{w}.log"))) else {
            continue;
        };
        // Only complete lines: a kill can land in the middle of writing one, and
        // a torn `A` line would read as a truncated acknowledged value.
        for line in text.split_inclusive('\n').filter(|l| l.ends_with('\n')).map(str::trim_end) {
            let mut parts = line.splitn(3, ' ');
            let (Some(kind), Some(key), Some(value)) = (parts.next(), parts.next(), parts.next()) else {
                continue;
            };
            let value = (value != "-").then(|| value.to_string());
            let entry = keys.entry(key.to_string()).or_default();
            match kind {
                "I" => entry.1.push(value),
                "A" => {
                    entry.0 = Some(value);
                    entry.1.clear();
                }
                _ => {}
            }
        }
    }
    keys
}

#[test]
fn acknowledged_writes_survive_repeated_kills() {
    use std::os::unix::process::ExitStatusExt;
    let dir = TempDir::new().unwrap();
    let mut rng = Rng(0xD1CE);
    for round in 0..4u64 {
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "db::wal::power_loss_tests::stress::kill_child_writes_until_killed",
                "--exact",
                "--ignored",
                "--nocapture",
            ])
            .env(KILL_DIR_VAR, dir.path())
            .env("MINNAL_KILL_ROUND", round.to_string())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap();
        std::thread::sleep(Duration::from_millis(300 + rng.below(700)));
        child.kill().unwrap();
        let status = child.wait().unwrap();
        assert_eq!(
            status.signal(),
            Some(libc::SIGKILL),
            "round {round}: the child stopped by itself: {status:?}"
        );

        let db = Database::open(&dir.path().join("db"), create_db_config()).unwrap();
        let ns = db.get_namespace_id("docs").expect("the namespace survives");
        let logs = read_kill_logs(dir.path());
        assert!(!logs.is_empty(), "round {round}: the child wrote nothing before it was killed");
        for (key, (acked, pending)) in &logs {
            let got = db.get_ns(ns, key.as_bytes()).unwrap().map(|v| String::from_utf8(v).unwrap());
            let ok = acked.as_ref().is_some_and(|a| *a == got) || pending.contains(&got) || (acked.is_none() && got.is_none());
            if !ok {
                // Keep the evidence: the database and the child's logs.
                let kept = dir.keep();
                panic!(
                    "round {round}: {key} reads {got:?}; acknowledged {acked:?}, in flight {pending:?}; kept {}",
                    kept.display()
                );
            }
        }
        // The field index agrees with the documents after recovery.
        activate_status_index(&db, ns);
        let mut indexed = db.query_keys(ns, "status = \"active\"").unwrap().keys;
        indexed.sort();
        let mut stored: Vec<Vec<u8>> = db
            .get_store(ns)
            .unwrap()
            .scan_prefix_batch(b"")
            .unwrap()
            .into_iter()
            .filter(|(_, v)| v.windows(17).any(|w| w == br#""status":"active""#))
            .map(|(k, _)| k)
            .collect();
        stored.sort();
        assert_eq!(indexed, stored, "round {round}: the field index disagrees with the documents");
        db.shutdown().unwrap();
    }
}
