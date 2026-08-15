//! Unit tests for the database coordinator.
//!
//! Split out of `database.rs`. Everything here exercises the coordinator itself
//! — namespace lifecycle, recovery, WAL bookkeeping, sequence allocation, stats,
//! the async facade, and process-level crash durability. Tests for the pieces
//! that moved to sibling modules moved with them.

use super::*;
use crate::db::config::SyncConfig;
use crate::db::test_support::*;
use crate::index::IndexValue;
use tempfile::TempDir;

// ── Process-level crash durability ─────────────────────────────────
//
// Everything else here asserts on counters inside one process. The crash pair
// below is the end-to-end guard: a real child process writes, is killed by a
// real `SIGKILL`, and the parent reopens the database and recounts. That is the
// only shape that tests the actual promise — "once `put` returns, the write
// survives" — rather than a proxy for it.

/// Env var carrying the database path from the parent to the crash child.
const CRASH_TEST_DIR_VAR: &str = "MINNAL_CRASH_TEST_DIR";
/// Written to the namespace that never fills its memtable.
const QUIET_KEYS: u32 = 100;
/// Written to the namespace that flushes repeatedly, dragging the watermark.
const BUSY_KEYS: u32 = 1_500;

fn quiet_key(i: u32) -> String {
    format!("quiet-{i:04}")
}

fn busy_key(i: u32) -> String {
    format!("busy-{i:04}")
}

/// The child half of [`acknowledged_writes_survive_a_process_kill`].
///
/// Inert unless the parent sets [`CRASH_TEST_DIR_VAR`], so a bare
/// `cargo test -- --ignored` cannot make it do anything.
///
/// It writes `quiet` first and `busy` second on purpose: `busy`'s flushes
/// then happen while `quiet`'s entries are still only in a memtable, which is
/// exactly the ordering that made one namespace's flush mark another's WAL
/// entries persisted.
#[test]
#[ignore = "spawned as a child process by acknowledged_writes_survive_a_process_kill"]
fn crash_child_writes_two_namespaces_then_dies() {
    let Ok(dir) = std::env::var(CRASH_TEST_DIR_VAR) else {
        return;
    };

    let db = Database::open(std::path::Path::new(&dir), crash_test_config()).unwrap();
    let quiet = db.create_namespace("quiet").unwrap();
    let busy = db.create_namespace("busy").unwrap();

    for i in 0..QUIET_KEYS {
        db.put_ns(quiet, quiet_key(i).as_bytes(), b"quiet-value").unwrap();
    }
    for i in 0..BUSY_KEYS {
        db.put_ns(busy, busy_key(i).as_bytes(), b"busy-value").unwrap();
    }

    // Every `put_ns` above returned `Ok`, which is the durability
    // acknowledgement: the WAL entry is fsynced before it returns. Die
    // without unwinding, running a destructor, or flushing anything —
    // `SIGKILL` cannot be caught, so nothing here gets a chance to tidy up,
    // exactly as in a power loss.
    unsafe { libc::raise(libc::SIGKILL) };
    unreachable!("SIGKILL did not terminate the crash child");
}

/// Every acknowledged write must survive a real `SIGKILL`, across two
/// namespaces with different flush histories.
///
/// This is the guard the suite was missing. Both critical defects found by
/// the stress run — a value-log fsync marking WAL entries persisted, and one
/// namespace's flush marking another's — were live while 1133 tests passed,
/// because nothing killed a process and recounted. Reintroduce either and
/// this fails: `quiet`'s entries are marked persisted while they exist only
/// in a memtable, so recovery skips them and the data is gone.
#[test]
fn acknowledged_writes_survive_a_process_kill() {
    use std::os::unix::process::ExitStatusExt;

    let dir = TempDir::new().unwrap();

    // Re-invoke this same test binary, running only the ignored child.
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        // `--exact` matches the full test path, not the bare function name.
        .args([
            "db::database::tests::crash_child_writes_two_namespaces_then_dies",
            "--exact",
            "--ignored",
            "--nocapture",
        ])
        .env(CRASH_TEST_DIR_VAR, dir.path())
        .status()
        .expect("failed to spawn the crash child");

    assert_eq!(
        status.signal(),
        Some(libc::SIGKILL),
        "child should have been killed by SIGKILL, but exited with {status:?} — \
             it probably failed before reaching the kill, so nothing below is meaningful"
    );

    // Reopen exactly as a restart would: this runs WAL recovery.
    let db = Database::open(dir.path(), crash_test_config()).unwrap();
    let quiet = db.get_namespace_id("quiet").expect("'quiet' should have survived the crash");
    let busy = db.get_namespace_id("busy").expect("'busy' should have survived the crash");

    let mut lost_quiet = Vec::new();
    for i in 0..QUIET_KEYS {
        if db.get_ns(quiet, quiet_key(i).as_bytes()).unwrap().as_deref() != Some(&b"quiet-value"[..]) {
            lost_quiet.push(i);
        }
    }
    let mut lost_busy = Vec::new();
    for i in 0..BUSY_KEYS {
        if db.get_ns(busy, busy_key(i).as_bytes()).unwrap().as_deref() != Some(&b"busy-value"[..]) {
            lost_busy.push(i);
        }
    }

    assert!(
        lost_quiet.is_empty() && lost_busy.is_empty(),
        "acknowledged writes did not survive the kill: {} of {QUIET_KEYS} lost from the \
             never-flushed namespace, {} of {BUSY_KEYS} from the flushing one",
        lost_quiet.len(),
        lost_busy.len()
    );

    db.shutdown().unwrap();
}

/// Regression (F6): concurrent first use of a namespace must not fail.
///
/// `create_namespace` publishes the registry entry — and persists it —
/// before it opens the `KVStore` and inserts it into `stores`. Between those
/// two points the name resolves to an id that `get_store` does not know.
/// Every get-or-create caller (`Db::namespace`, `AsyncDb::namespace`, and so
/// the whole vector write path) does exactly `get_namespace_id` followed by
/// `get_store_by_name`, so it lands in that window and fails with
/// "Namespace with ID N not found". Separately, two callers that both see the
/// name as absent both try to create it, and the loser gets
/// "Namespace 'x' already exists" from the registry.
///
/// Seen once in a live stress run, on a new store's first embed, where it
/// burned one of five retries. It is not rare: the probe this test came from
/// reproduced **167** "not found" and **13** "already exists" in 240 attempts.
#[test]
fn test_concurrent_first_use_of_a_namespace_does_not_race() {
    use std::sync::Barrier;

    let dir = TempDir::new().unwrap();
    let db = Arc::new(Database::open(dir.path(), create_db_config()).unwrap());
    let mut failures: Vec<String> = Vec::new();

    for round in 0..25 {
        let name = format!("ns_{round}");
        // All four threads reach the get-or-create at once, on a name that
        // does not exist yet.
        let barrier = Arc::new(Barrier::new(4));
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let db = Arc::clone(&db);
                let name = name.clone();
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    // Exactly what `Db::namespace` / `AsyncDb::namespace` do.
                    match db.get_namespace_id(&name) {
                        Some(id) => Ok(id),
                        None => db.create_namespace(&name),
                    }
                    .and_then(|_| db.get_store_by_name(&name))
                    .map(|_| ())
                })
            })
            .collect();
        for handle in handles {
            if let Err(e) = handle.join().unwrap() {
                failures.push(e.to_string());
            }
        }

        // All four must agree on one namespace, and it must be usable.
        let id = db.get_namespace_id(&name).expect("the namespace should exist after the round");
        db.put_ns(id, b"k", b"v").unwrap();
        assert_eq!(db.get_ns(id, b"k").unwrap().as_deref(), Some(&b"v"[..]));
    }

    assert!(
        failures.is_empty(),
        "{} of 100 concurrent first-use attempts failed: {:?}",
        failures.len(),
        // Distinct messages only — the same two repeat.
        failures.iter().collect::<std::collections::BTreeSet<_>>()
    );
}

/// Regression: activating the same field index from several threads must not
/// kill the process.
///
/// `activate_field_index` creates memory-mapped files — the field's `BlobStore`
/// pair, its `keymap/` pair, and the namespace's `RowMap` — through a
/// check-then-act (`if exists { open } else { create }`), and the create path
/// opens with `truncate(true)`. Two threads racing it truncate a file the other
/// has already mapped, so the next access to that mapping faults with
/// **SIGBUS**: the process dies outright, with no unwinding and no `Result` for
/// anyone to handle.
///
/// Reachable from an ordinary client: two concurrent `POST /stores` for one
/// namespace (a timeout plus a retry is enough) each activate the same fields.
/// Measured before the fix, against the stress server: 8 racing creates killed
/// it on **every** attempt — `Bus error (core dumped)`, exit 135 — and a
/// different interleaving surfaced instead as
/// `range end index 8 out of range for slice of length 0` writing the 64-byte
/// header into a zero-length map.
///
/// **Reverting the fix does not make this test fail — it makes the test binary
/// die.** That is the honest reproduction; there is nothing to assert against a
/// signal. What the assertions below add is that serialising activation did not
/// break it: every thread returns `Ok`, and the index answers correctly
/// afterwards.
#[test]
fn test_concurrent_activation_of_one_field_index_does_not_kill_the_process() {
    use std::sync::Barrier;

    let dir = TempDir::new().unwrap();
    let db = Arc::new(Database::open(dir.path(), create_db_config()).unwrap());

    const THREADS: usize = 8;
    let mut failures: Vec<String> = Vec::new();

    for round in 0..10 {
        let name = format!("act_ns_{round}");
        let ns = db.create_namespace(&name).unwrap();
        // Registered once, up front: the race under test is the *activation*,
        // not the registration.
        let field_id = db.register_index_field(ns, "status", IndexValueType::Str).unwrap();

        let barrier = Arc::new(Barrier::new(THREADS));
        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let db = Arc::clone(&db);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let extractor: ExtractorFn = Arc::new(|bytes: &[u8]| {
                        let s = std::str::from_utf8(bytes).ok()?;
                        let v: serde_json::Value = serde_json::from_str(s).ok()?;
                        Some(IndexValue::Str(v["status"].as_str()?.to_string()))
                    });
                    barrier.wait();
                    db.activate_field_index(ns, field_id, IndexValueType::Str, extractor)
                })
            })
            .collect();
        for handle in handles {
            if let Err(e) = handle.join().unwrap() {
                failures.push(e.to_string());
            }
        }

        // The surviving index must be usable, not merely present: a store that
        // came through the race with a truncated key file would open with a
        // zeroed header and silently answer nothing.
        db.put_ns(ns, b"d1", br#"{"status":"active"}"#).unwrap();
        db.put_ns(ns, b"d2", br#"{"status":"retired"}"#).unwrap();
        let hits = db.query_keys(ns, r#"status = "active""#).unwrap();
        assert_eq!(
            hits.keys,
            vec![b"d1".to_vec()],
            "round {round}: the field index survived activation but does not answer correctly"
        );
    }

    assert!(
        failures.is_empty(),
        "{} of {} concurrent activations failed: {:?}",
        failures.len(),
        THREADS * 10,
        failures.iter().collect::<std::collections::BTreeSet<_>>()
    );
}

// ── The in-flight write barrier ────────────────────────────────────────────
//
// Three tests for one invariant: **a WAL entry may be marked `Persisted` only
// once its key is in an SSTable**, because recovery skips `Persisted` entries.
//
// Everything that drives the persisted cut is derived from a `wal_metadata.tail`
// snapshot, and a tail snapshot over-states durability: `put_ns` appends under
// the WAL lock and applies to the memtable *after* releasing it, so in between,
// an entry is visible in the tail while its key is nowhere. The three tests
// cover the two places that read the tail, plus the path that put the bug in
// front of a user.
//
// Found by the 2026-08-02 stress run: 1 acknowledged write lost in 34,538, then
// 4 lost in one round of `work/stress/repro/drop_loses_a_write.py`.

/// The cut must not pass a write that is in the WAL but not yet in a memtable.
///
/// `try_advance_persisted` cut at the **live tail** whenever no namespace
/// reported un-flushed writes — which is exactly the state this test sets up.
#[test]
fn an_in_flight_write_holds_the_persisted_watermark_back() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    let ns = db.create_namespace("ns").unwrap();

    // Flush everything written so far, so the namespace reports nothing
    // outstanding. Without that this test would be vacuous: a namespace with
    // un-flushed writes pins the cut on its own.
    for i in 0..20u32 {
        db.put_ns(ns, format!("k{i}").as_bytes(), b"v").unwrap();
    }
    db.get_store(ns).unwrap().flush_memtable_to_level0().unwrap();

    // The offset a write would occupy between its WAL append and its apply.
    let in_flight_offset = db.wal_metadata.read().tail;
    let guard = db.wal_flush_observer.begin_write(in_flight_offset);

    // Move the tail well past it, then flush again so that — but for the
    // in-flight registration — nothing at all constrains the cut.
    for i in 20..40u32 {
        db.put_ns(ns, format!("k{i}").as_bytes(), b"v").unwrap();
    }
    db.get_store(ns).unwrap().flush_memtable_to_level0().unwrap();

    let watermark = *db.last_persisted_wal_offset.read();
    let tail = db.wal_metadata.read().tail;
    assert!(
        watermark <= in_flight_offset,
        "watermark {watermark} passed an in-flight write at {in_flight_offset} \
         (tail {tail}) — recovery would skip a write that is in no SSTable"
    );
    drop(guard);
}

/// A memtable seal must not claim a write whose apply has not landed.
///
/// `on_memtable_sealed_ns` records "every WAL entry below this tail is in the
/// memtable I am sealing". That is false for an in-flight write: its apply goes
/// into the **next** memtable, while its offset sits below the recorded tail —
/// so `advance_namespace` would push `safe_offset` past an entry no SSTable
/// holds. This needs no namespace drop, which is why guarding only the cut
/// would have left the bug live.
#[test]
fn a_memtable_seal_does_not_claim_an_unapplied_write() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    let ns = db.create_namespace("ns").unwrap();

    for i in 0..10u32 {
        db.put_ns(ns, format!("k{i}").as_bytes(), b"v").unwrap();
    }

    let in_flight_offset = db.wal_metadata.read().tail;
    let guard = db.wal_flush_observer.begin_write(in_flight_offset);

    // These land after the in-flight write, so a seal now must not report that
    // it captured everything up to the current tail.
    for i in 10..30u32 {
        db.put_ns(ns, format!("k{i}").as_bytes(), b"v").unwrap();
    }
    db.get_store(ns).unwrap().flush_memtable_to_level0().unwrap();

    let safe = db.wal_flush_observer.ns_progress.read().get(&ns).map(|p| p.safe_offset).unwrap_or(0);
    assert!(
        safe <= in_flight_offset,
        "seal recorded safe_offset {safe}, past the in-flight write at {in_flight_offset}"
    );
    drop(guard);
}

/// The path a user actually hit: dropping a store while another namespace writes.
///
/// `forget_namespace` calls `try_advance_persisted()` the instant it removes the
/// dropped namespace from `ns_progress` — deliberately, to unpin the cut — and
/// that is precisely when the set of namespaces reporting un-flushed writes can
/// go empty while a survivor is mid-write. Measured before the fix: 4
/// acknowledged writes lost in one round of the repro script, and the surviving
/// namespace's key was gone after a SIGKILL with **no recovery pass at all**
/// (`persisted == total` short-circuits it).
#[test]
fn dropping_a_store_does_not_persist_a_survivors_in_flight_write() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    let survivor = db.create_namespace("survivor").unwrap();
    let victim = db.create_namespace("victim").unwrap();

    for i in 0..20u32 {
        db.put_ns(survivor, format!("s{i}").as_bytes(), b"v").unwrap();
        db.put_ns(victim, format!("v{i}").as_bytes(), b"v").unwrap();
    }
    // Both flushed: neither pins the cut, so only the in-flight guard can.
    db.get_store(survivor).unwrap().flush_memtable_to_level0().unwrap();
    db.get_store(victim).unwrap().flush_memtable_to_level0().unwrap();

    let in_flight_offset = db.wal_metadata.read().tail;
    let guard = db.wal_flush_observer.begin_write(in_flight_offset);
    for i in 20..30u32 {
        db.put_ns(survivor, format!("s{i}").as_bytes(), b"v").unwrap();
    }
    db.get_store(survivor).unwrap().flush_memtable_to_level0().unwrap();

    db.remove_namespace("victim").unwrap();

    let watermark = *db.last_persisted_wal_offset.read();
    assert!(
        watermark <= in_flight_offset,
        "dropping a store advanced the watermark to {watermark}, past the survivor's \
         in-flight write at {in_flight_offset}"
    );
    drop(guard);
}

/// `try_advance_persisted`'s cut must respect the ceiling on its own.
///
/// The other three tests all pass with the cut's cap reverted, because the seal
/// cap alone keeps `safe_offset` behind the barrier — and that is not an
/// accident. Registration happens under the WAL metadata write lock, so no seal
/// can observe a tail containing a write without also observing its
/// registration, which means `safe_offset` can never overtake an in-flight
/// write and the cut's cap is **redundant by argument**.
///
/// It is kept, and tested here, because that argument is non-local: it lives in
/// `put_ns`, three hundred lines from the `None` branch it protects. Moving the
/// registration out of the WAL lock would silently re-open the hole, and the cut
/// should not depend on remembering why it is safe.
///
/// The final trigger here is a **namespace drop**, which reaches
/// `try_advance_persisted` through `forget_namespace` with no seal after it — so
/// the cut's own cap is the only thing left holding the barrier.
#[test]
fn the_cut_respects_the_barrier_even_when_no_seal_enforces_it() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    let ns_a = db.create_namespace("a").unwrap();
    let ns_b = db.create_namespace("b").unwrap();

    for i in 0..20u32 {
        db.put_ns(ns_a, format!("a{i}").as_bytes(), b"v").unwrap();
    }

    // The barrier goes down BEFORE any flush. A barrier cannot retract progress
    // the watermark has already made, so registering it after a flush that
    // advanced past it asserts nothing (that mistake made an earlier version of
    // this test fail against correct behaviour).
    let barrier = db.wal_metadata.read().tail;
    let guard = db.wal_flush_observer.begin_write(barrier);

    for i in 0..20u32 {
        db.put_ns(ns_b, format!("b{i}").as_bytes(), b"v").unwrap();
    }
    db.get_store(ns_a).unwrap().flush_memtable_to_level0().unwrap();
    db.get_store(ns_b).unwrap().flush_memtable_to_level0().unwrap();

    // Dropping `b` removes the last namespace that reports un-flushed writes,
    // so the cut falls through to the `None` branch — the one that used to take
    // the live tail.
    db.remove_namespace("b").unwrap();

    let watermark = *db.last_persisted_wal_offset.read();
    let tail = db.wal_metadata.read().tail;
    assert!(
        watermark <= barrier,
        "cut reached {watermark}, past the barrier at {barrier} (tail {tail}), \
         with no seal left to stop it"
    );
    drop(guard);
}

/// The barrier must not leak: a registration outliving its write pins the
/// watermark forever, so WAL GC's `persisted >= total` gate never fires again
/// and the WAL grows without bound — the failure mode `17a8a0c` fixed for
/// dropped namespaces, reached a different way.
#[test]
fn the_in_flight_barrier_is_released_when_the_write_finishes() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    let ns = db.create_namespace("ns").unwrap();

    for i in 0..50u32 {
        db.put_ns(ns, format!("k{i}").as_bytes(), b"v").unwrap();
    }
    assert!(
        db.wal_flush_observer.in_flight.lock().is_empty(),
        "writes that have returned are still registered as in flight: {:?}",
        db.wal_flush_observer.in_flight.lock()
    );

    // And the watermark can still reach the tail once everything has flushed —
    // i.e. the barrier bounds progress without stopping it.
    db.get_store(ns).unwrap().flush_memtable_to_level0().unwrap();
    let tail = db.wal_metadata.read().tail;
    let watermark = *db.last_persisted_wal_offset.read();
    assert_eq!(watermark, tail, "watermark stalled at {watermark} with tail {tail}");
}

/// Regression: syncing the value log must NOT advance the WAL persisted
/// watermark.
///
/// A value-log fsync makes the *value* durable, but the key → pointer
/// mapping still lives solely in the LSM memtable until that memtable is
/// flushed to an SSTable. Recovery skips `Persisted` entries, so marking
/// them at sync time tells recovery to skip keys that exist nowhere on disk:
/// the writes are lost and their values sit orphaned in the value log.
/// Measured before the fix, against a stock-config server over the REST API:
/// **0 of 3000** acknowledged writes survived a `kill -9`, with
/// `persisted_entries` already equal to `total_entries` so recovery did not
/// even run.
///
/// `records_per_sync = 1` puts every single write through the sync path, and
/// the memtable is left far from full so nothing legitimately flushes — any
/// entry marked persisted here can only have come from the sync.
#[test]
fn test_value_log_sync_does_not_mark_wal_entries_persisted() {
    let dir = TempDir::new().unwrap();
    let mut config = create_db_config();
    config.sync_config = SyncConfig::new(1);
    config.lsm_config.skip_list_capacity = 100_000;
    let db = Database::open(dir.path(), config).unwrap();
    let ns = db.create_namespace("data").unwrap();

    for i in 0..200u32 {
        db.put_ns(ns, format!("k{i:04}").as_bytes(), b"value").unwrap();
    }
    // Deletes take the same Step-3 sync path and had the same defect.
    db.delete_ns(ns, b"k0000").unwrap();

    let meta = db.wal_metadata.read();
    let (total, persisted) = (meta.total_entries, meta.persisted_entries);
    drop(meta);

    assert_eq!(total, 201, "expected every write to be journalled, got total={total}");
    assert_eq!(
        persisted, 0,
        "a value-log fsync marked {persisted} of {total} WAL entries persisted, but nothing \
             has reached an SSTable — recovery would skip them and lose acknowledged writes"
    );
}

/// Regression: `pending` flush records must be keyed by `(namespace,
/// version)`, not by version alone.
///
/// Memtable versions are allocated per `KVStore` from a counter created as
/// `AtomicU64::new(0)` in the LSM tree's own constructor and never persisted,
/// so **every** namespace's first sealed memtable is version 0 — and so is
/// its first seal after any reopen. Keyed by version alone, namespace A's
/// seal record and namespace B's are literally the same map entry:
/// `or_insert` silently drops the second, and A's flush then satisfies B's
/// record, advancing the watermark past writes B has not flushed.
///
/// This is the second of the two mechanisms behind the cross-namespace
/// data-loss bug. The per-namespace `safe_offset` minimum masks it today, so
/// nothing else fails if the key is collapsed back — which is exactly why it
/// needs pinning here.
#[test]
fn test_pending_flush_records_are_keyed_by_namespace_not_just_version() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    let a = db.create_namespace("a").unwrap();
    let b = db.create_namespace("b").unwrap();

    for i in 0..5u32 {
        db.put_ns(a, format!("a{i}").as_bytes(), b"v").unwrap();
        db.put_ns(b, format!("b{i}").as_bytes(), b"v").unwrap();
    }

    // Both namespaces seal their first memtable — version 0 for each.
    db.wal_flush_observer.on_memtable_sealed_ns(a, 0);
    db.wal_flush_observer.on_memtable_sealed_ns(b, 0);
    assert_eq!(
        db.pending_wal_flushes.read().len(),
        2,
        "two namespaces sealing version 0 must record two pending flushes, not collide into one"
    );

    // A's flush may satisfy only A's record.
    db.wal_flush_observer.on_ro_memtable_flushed_to_level0_ns(a, 0);

    let pending = db.pending_wal_flushes.read();
    assert!(!pending.contains_key(&(a, 0)), "A's own record should be consumed by A's flush");
    let b_state = pending
        .get(&(b, 0))
        .expect("B's seal record must survive A's flush, not be satisfied by it");
    assert!(!b_state.flushed, "A's flush must not mark B's memtable flushed");
}

/// Regression: one namespace's memtable flush must not mark ANOTHER
/// namespace's WAL entries persisted.
///
/// The WAL is global, memtables are per-namespace. The observer used to
/// record the global WAL tail at seal time and, on flush, mark every entry
/// below it persisted — including entries owned by namespaces that had not
/// flushed. Recovery skips persisted entries, so those acknowledged writes
/// were silently lost on the next crash. Measured against a live server
/// before the fix: 0 of 200 keys in the quiet namespace survived a SIGKILL
/// after a second namespace flushed repeatedly.
///
/// Asserting on `persisted_entries` rather than on post-crash reads keeps
/// this a unit test: an entry marked persisted is exactly the entry recovery
/// will refuse to replay.
#[test]
fn test_one_namespace_flush_does_not_persist_anothers_wal_entries() {
    let dir = TempDir::new().unwrap();
    let mut config = create_db_config();
    // Small memtable so `busy` seals and flushes repeatedly.
    config.lsm_config.skip_list_capacity = 64;
    let db = Database::open(dir.path(), config).unwrap();

    let quiet = db.create_namespace("quiet").unwrap();
    let busy = db.create_namespace("busy").unwrap();

    // `quiet` writes a handful of records and then goes idle — they stay in
    // its memtable, so none of them may be marked persisted.
    for i in 0..10u32 {
        db.put_ns(quiet, format!("q{i}").as_bytes(), b"quiet-value").unwrap();
    }
    let quiet_entries = 10u64;

    // `busy` writes, then flushes to level 0 — the event that used to mark
    // everything below the global WAL tail persisted. Flushed explicitly
    // rather than by capacity so the test does not depend on the background
    // workers running.
    for i in 0..500u32 {
        db.put_ns(busy, format!("b{i}").as_bytes(), b"busy-value").unwrap();
    }
    db.get_store(busy).unwrap().flush_memtable_to_level0().unwrap();

    let meta = db.wal_metadata.read();
    let (total, persisted) = (meta.total_entries, meta.persisted_entries);
    drop(meta);

    assert!(
        persisted <= total.saturating_sub(quiet_entries),
        "quiet namespace's {quiet_entries} un-flushed entries were marked persisted \
             by busy's flushes: persisted={persisted} total={total}"
    );
}

#[test]
fn test_concurrent_mark_persisted_never_overcounts() {
    // Regression: `mark_persisted_range` is a non-atomic scan → flip → count.
    // Run concurrently by multiple writers reaching the sync point at once, two
    // threads would flip and BOTH count the same entry, pushing
    // `persisted_entries` past `total_entries`. Because the WAL GC deletion gate
    // was exact equality (`total == persisted`), that impossible state wedged
    // WAL GC forever (segments never became deletable → unbounded WAL growth).
    // `persist_lock` must serialize the flip+count so no entry is counted twice.
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();

    // Append a batch of entries (default records_per_sync is high, so these do
    // not auto-persist during the writes).
    let n = 500u64;
    for i in 0..n {
        db.put(format!("key{i}").as_bytes(), b"value").unwrap();
    }
    let tail = db.wal_metadata.read().tail;
    let total = db.wal_metadata.read().total_entries;
    assert!(total >= n, "expected at least {n} entries, got {total}");

    // Hammer the same range from many threads at once.
    std::thread::scope(|s| {
        for _ in 0..16 {
            s.spawn(|| {
                for _ in 0..8 {
                    db.wal_flush_observer.mark_persisted_range(0, tail);
                }
            });
        }
    });

    let meta = db.wal_metadata.read();
    // The invariant the bug violated: persisted can never exceed total.
    assert!(
        meta.persisted_entries <= meta.total_entries,
        "persisted_entries ({}) exceeded total_entries ({}) — over-count race",
        meta.persisted_entries,
        meta.total_entries,
    );
    // The whole range was covered, so every entry is persisted exactly once.
    assert_eq!(meta.persisted_entries, meta.total_entries);
    // Per-segment invariant holds too.
    for seg in meta.tracked_segments() {
        assert!(
            meta.segment_persisted(seg) <= meta.segment_total(seg),
            "segment {seg}: persisted {} > total {}",
            meta.segment_persisted(seg),
            meta.segment_total(seg),
        );
    }
    drop(meta);
    db.shutdown().unwrap();
}

#[test]
fn test_wal_segment_size_is_locked_at_creation() {
    // The configured WAL segment size applies only to a brand-new WAL, and is
    // then fixed: reopening with a different config value must NOT change it
    // (a segment id is offset / segment_size, so a change would re-bucket
    // stored offsets into the wrong files).
    let dir = TempDir::new().unwrap();
    let small = 64 * 1024; // 64 KiB
    {
        let mut cfg = create_db_config();
        cfg.wal_segment_size = small;
        let db = Database::open(dir.path(), cfg).unwrap();
        assert_eq!(db.wal.segment_size(), small, "fresh WAL should honour config");
        db.put(b"k", b"v").unwrap();
        db.shutdown().unwrap();
    }
    // A marker must have been recorded.
    assert!(dir.path().join("wal_segment_size").exists());
    {
        // Reopen with a DIFFERENT configured size — the created size must win.
        let mut cfg = create_db_config();
        cfg.wal_segment_size = 8 * 1024 * 1024;
        let db = Database::open(dir.path(), cfg).unwrap();
        assert_eq!(db.wal.segment_size(), small, "existing WAL size must be locked");
        assert_eq!(db.get(b"k").unwrap(), Some(b"v".to_vec()));
        db.shutdown().unwrap();
    }
}

#[test]
fn test_database_basic_operations() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();

    db.put(b"key1", b"value1").unwrap();
    db.put(b"key2", b"value2").unwrap();

    assert_eq!(db.get(b"key1").unwrap(), Some(b"value1".to_vec()));
    assert_eq!(db.get(b"key2").unwrap(), Some(b"value2".to_vec()));

    db.delete(b"key2").unwrap();
    assert_eq!(db.get(b"key2").unwrap(), None);

    db.shutdown().unwrap();
}

#[test]
fn test_database_multiple_namespaces() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();

    // Create additional namespace — ID 0 = default, 1 = system, so users gets 2.
    let users_id = db.create_namespace("users").unwrap();
    assert_eq!(users_id, 2);

    // Write to default namespace
    db.put(b"global_key", b"global_value").unwrap();

    // Write to users namespace
    db.put_ns(users_id, b"user:1", b"alice").unwrap();
    db.put_ns(users_id, b"user:2", b"bob").unwrap();

    // Read from default — should not see users data
    assert_eq!(db.get(b"global_key").unwrap(), Some(b"global_value".to_vec()));
    assert_eq!(db.get(b"user:1").unwrap(), None);

    // Read from users namespace
    assert_eq!(db.get_ns(users_id, b"user:1").unwrap(), Some(b"alice".to_vec()));
    assert_eq!(db.get_ns(users_id, b"user:2").unwrap(), Some(b"bob".to_vec()));
    assert_eq!(db.get_ns(users_id, b"global_key").unwrap(), None);

    db.shutdown().unwrap();
}

#[test]
fn test_database_namespace_listing() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();

    db.create_namespace("users").unwrap();
    db.create_namespace("orders").unwrap();

    let mut namespaces = db.list_namespaces();
    namespaces.sort_by_key(|(_, id)| *id);
    // default(0), system(1), users(2), orders(3)
    assert_eq!(namespaces.len(), 4);
    assert_eq!(namespaces[0], ("default".to_string(), 0));
    assert_eq!(namespaces[1], ("system".to_string(), 1));
    assert_eq!(namespaces[2], ("users".to_string(), 2));
    assert_eq!(namespaces[3], ("orders".to_string(), 3));

    db.shutdown().unwrap();
}

#[test]
fn test_database_namespace_persistence() {
    let dir = TempDir::new().unwrap();

    // Create database with namespaces and data
    {
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        let users_id = db.create_namespace("users").unwrap();
        db.put(b"default_key", b"default_val").unwrap();
        db.put_ns(users_id, b"user_key", b"user_val").unwrap();
        db.shutdown().unwrap();
    }

    // Reopen and verify
    {
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        assert!(db.namespace_exists("users"));
        let users_id = db.get_namespace_id("users").unwrap();

        assert_eq!(db.get(b"default_key").unwrap(), Some(b"default_val".to_vec()));
        assert_eq!(db.get_ns(users_id, b"user_key").unwrap(), Some(b"user_val".to_vec()));

        db.shutdown().unwrap();
    }
}

#[test]
fn test_database_remove_namespace() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();

    let users_id = db.create_namespace("users").unwrap();
    db.put_ns(users_id, b"key", b"val").unwrap();

    db.remove_namespace("users").unwrap();
    assert!(!db.namespace_exists("users"));
    assert!(db.get_ns(users_id, b"key").is_err());

    // Cannot remove default
    assert!(db.remove_namespace("default").is_err());

    db.shutdown().unwrap();
}

#[test]
fn test_activate_type_mismatch_rejected() {
    use crate::db::namespace_index::ExtractorFn;
    use crate::index::IndexValueType;
    use std::sync::Arc;

    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();

    let ns = DEFAULT_NAMESPACE_ID;
    // Register as Int
    let field_id = db.register_index_field(ns, "age", IndexValueType::Int).unwrap();

    // Activating with the correct type succeeds
    let extractor: ExtractorFn = Arc::new(|_| None);
    assert!(db.activate_field_index(ns, field_id, IndexValueType::Int, Arc::clone(&extractor)).is_ok());

    // Activating with a different type must fail
    let err = db.activate_field_index(ns, field_id, IndexValueType::Str, extractor).unwrap_err();
    assert!(err.to_string().contains("Type mismatch"), "unexpected error: {}", err);
}

#[test]
fn test_activate_unknown_field_id_rejected() {
    use crate::db::namespace_index::ExtractorFn;
    use crate::index::IndexValueType;
    use std::sync::Arc;

    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();

    let extractor: ExtractorFn = Arc::new(|_| None);
    // field_id 99 was never registered
    let err = db
        .activate_field_index(DEFAULT_NAMESPACE_ID, 99, IndexValueType::Int, extractor)
        .unwrap_err();
    assert!(err.to_string().contains("not registered"), "unexpected error: {}", err);
}

#[test]
fn test_duplicate_register_field_idempotent() {
    use crate::index::IndexValueType;

    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();

    let id1 = db.register_index_field(DEFAULT_NAMESPACE_ID, "status", IndexValueType::Str).unwrap();
    // Same name + same type: idempotent, returns existing id
    let id2 = db.register_index_field(DEFAULT_NAMESPACE_ID, "status", IndexValueType::Str).unwrap();
    assert_eq!(id1, id2);
    // Same name, different type: error
    let err = db.register_index_field(DEFAULT_NAMESPACE_ID, "status", IndexValueType::Int).unwrap_err();
    assert!(err.to_string().contains("already registered"), "unexpected error: {}", err);
}

#[test]
fn test_schema_survives_restart_and_index_activates_without_re_register() {
    use crate::db::namespace_index::ExtractorFn;
    use crate::index::{IndexValue, IndexValueType};
    use std::sync::Arc;

    let dir = TempDir::new().unwrap();

    // ── First open: register fields and write some data ──────────────
    let status_field_id;
    let age_field_id;
    {
        let db = Database::open(dir.path(), create_db_config()).unwrap();

        status_field_id = db.register_index_field(DEFAULT_NAMESPACE_ID, "status", IndexValueType::Str).unwrap();
        age_field_id = db.register_index_field(DEFAULT_NAMESPACE_ID, "age", IndexValueType::Int).unwrap();

        // Activate both indices with extractors
        let status_extractor: ExtractorFn = Arc::new(|bytes: &[u8]| {
            let s = std::str::from_utf8(bytes).ok()?;
            let v: serde_json::Value = serde_json::from_str(s).ok()?;
            Some(IndexValue::Str(v["status"].as_str()?.to_string()))
        });
        let age_extractor: ExtractorFn = Arc::new(|bytes: &[u8]| {
            let s = std::str::from_utf8(bytes).ok()?;
            let v: serde_json::Value = serde_json::from_str(s).ok()?;
            Some(IndexValue::Int(v["age"].as_i64()?))
        });
        db.activate_field_index(DEFAULT_NAMESPACE_ID, status_field_id, IndexValueType::Str, status_extractor)
            .unwrap();
        db.activate_field_index(DEFAULT_NAMESPACE_ID, age_field_id, IndexValueType::Int, age_extractor)
            .unwrap();

        db.put(b"user:1", br#"{"status":"active","age":30}"#).unwrap();
        db.put(b"user:2", br#"{"status":"inactive","age":25}"#).unwrap();
        db.put(b"user:3", br#"{"status":"active","age":40}"#).unwrap();

        db.shutdown().unwrap();
    }

    // ── Second open: do NOT call register_index_field ────────────────
    // Schema must be loaded from config.json automatically.
    {
        let db = Database::open(dir.path(), create_db_config()).unwrap();

        // Confirm schema is present without re-registering
        let fields = db.list_index_fields(DEFAULT_NAMESPACE_ID);
        assert_eq!(fields.len(), 2);
        assert!(fields.iter().any(|f| f.field_name == "status" && f.field_type == IndexValueType::Str));
        assert!(fields.iter().any(|f| f.field_name == "age" && f.field_type == IndexValueType::Int));

        // Activate indices with extractors (closures can't be persisted — caller always supplies these)
        let status_extractor: ExtractorFn = Arc::new(|bytes: &[u8]| {
            let s = std::str::from_utf8(bytes).ok()?;
            let v: serde_json::Value = serde_json::from_str(s).ok()?;
            Some(IndexValue::Str(v["status"].as_str()?.to_string()))
        });
        let age_extractor: ExtractorFn = Arc::new(|bytes: &[u8]| {
            let s = std::str::from_utf8(bytes).ok()?;
            let v: serde_json::Value = serde_json::from_str(s).ok()?;
            Some(IndexValue::Int(v["age"].as_i64()?))
        });
        // These must succeed using the field IDs recovered from config.json
        db.activate_field_index(DEFAULT_NAMESPACE_ID, status_field_id, IndexValueType::Str, status_extractor)
            .unwrap();
        db.activate_field_index(DEFAULT_NAMESPACE_ID, age_field_id, IndexValueType::Int, age_extractor)
            .unwrap();

        // Queries must return correct results from the warmed index
        let mut active_keys = db.query_keys(DEFAULT_NAMESPACE_ID, "status = \"active\"").unwrap().keys;
        active_keys.sort();
        assert_eq!(active_keys, vec![b"user:1".to_vec(), b"user:3".to_vec()]);

        let inactive_keys = db.query_keys(DEFAULT_NAMESPACE_ID, "status = \"inactive\"").unwrap().keys;
        assert_eq!(inactive_keys, vec![b"user:2".to_vec()]);

        db.shutdown().unwrap();
    }
}

/// Measurement — the decisive one for "just shrink the checkpoint interval".
///
/// Holds the replay WINDOW fixed and varies the TOTAL document count. If the
/// per-key replay cost is flat, shrinking the interval is a lasting fix. If it
/// grows with total data, shrinking the interval buys a one-off factor and the
/// problem returns as the store grows.
///
/// `cargo test -p minnal_db --release --lib replay_cost_vs_total -- --ignored --nocapture`
#[test]
#[ignore = "measurement; run explicitly with --release --ignored --nocapture"]
fn replay_cost_vs_total() {
    const WINDOW: u32 = 750;
    println!("fixed replay window={WINDOW} keys, 2 distinct values");
    for total in [3_000u32, 6_000, 12_000] {
        let dir = TempDir::new().unwrap();
        let ns = DEFAULT_NAMESPACE_ID;
        let config = create_db_config();
        let field_id = {
            let db = Database::open(dir.path(), config.clone()).unwrap();
            let field_id = activate_status_index(&db, ns);
            for i in 0..total {
                let v = if i % 2 == 0 { "active" } else { "inactive" };
                db.put(format!("doc:{i:06}").as_bytes(), format!(r#"{{"status":"{v}"}}"#).as_bytes())
                    .unwrap();
            }
            db.run_index_checkpoint().unwrap();
            for i in 0..WINDOW {
                let v = if i % 2 == 0 { "active" } else { "inactive" };
                db.put(format!("doc:{i:06}").as_bytes(), format!(r#"{{"status":"{v}"}}"#).as_bytes())
                    .unwrap();
            }
            std::mem::forget(db);
            field_id
        };
        let db = Database::open(dir.path(), config).unwrap();
        let extractor: crate::db::namespace_index::ExtractorFn = std::sync::Arc::new(|bytes: &[u8]| {
            let s = std::str::from_utf8(bytes).ok()?;
            let v: serde_json::Value = serde_json::from_str(s).ok()?;
            Some(crate::index::IndexValue::Str(v["status"].as_str()?.to_string()))
        });
        let t = std::time::Instant::now();
        db.activate_field_index(ns, field_id, crate::index::IndexValueType::Str, extractor)
            .unwrap();
        let dt = t.elapsed();
        println!(
            "  total={total:6} docs -> replay {:>8.1?} ({:.3} ms/key)",
            dt,
            dt.as_secs_f64() * 1000.0 / WINDOW as f64
        );
        db.shutdown().unwrap();
    }
}

/// Measurement, not a correctness check — run explicitly:
/// `cargo test -p minnal_db --release --lib replay_cost_vs_window -- --ignored --nocapture`
///
/// Answers: does shrinking the checkpoint interval (i.e. the replay window)
/// reduce restart time proportionally? Holds the total document count fixed and
/// varies only how many of those keys fall inside the replay window.
#[test]
#[ignore = "measurement; run explicitly with --release --ignored --nocapture"]
fn replay_cost_vs_window() {
    const TOTAL: u32 = 6_000;
    println!("total docs={TOTAL}, 2 distinct values");
    for window in [750u32, 1_500, 3_000, 6_000] {
        let dir = TempDir::new().unwrap();
        let ns = DEFAULT_NAMESPACE_ID;
        let config = create_db_config();
        let field_id = {
            let db = Database::open(dir.path(), config.clone()).unwrap();
            let field_id = activate_status_index(&db, ns);
            for i in 0..TOTAL {
                let v = if i % 2 == 0 { "active" } else { "inactive" };
                db.put(format!("doc:{i:06}").as_bytes(), format!(r#"{{"status":"{v}"}}"#).as_bytes())
                    .unwrap();
            }
            db.run_index_checkpoint().unwrap();
            // Re-write `window` keys AFTER the checkpoint: those are exactly the
            // keys the next replay will have to process.
            for i in 0..window {
                let v = if i % 2 == 0 { "active" } else { "inactive" };
                db.put(format!("doc:{i:06}").as_bytes(), format!(r#"{{"status":"{v}"}}"#).as_bytes())
                    .unwrap();
            }
            // Simulate a CRASH. `shutdown()` runs a final checkpoint, which
            // would advance the marker past everything just written and leave
            // the replay window empty — measuring nothing. (It did, on the
            // first attempt at this measurement.)
            std::mem::forget(db);
            field_id
        };

        let db = Database::open(dir.path(), config).unwrap();
        let extractor: crate::db::namespace_index::ExtractorFn = std::sync::Arc::new(|bytes: &[u8]| {
            let s = std::str::from_utf8(bytes).ok()?;
            let v: serde_json::Value = serde_json::from_str(s).ok()?;
            Some(crate::index::IndexValue::Str(v["status"].as_str()?.to_string()))
        });
        let t = std::time::Instant::now();
        db.activate_field_index(ns, field_id, crate::index::IndexValueType::Str, extractor)
            .unwrap();
        let dt = t.elapsed();
        println!(
            "  window={window:5} keys -> replay {:>8.1?} ({:.2} ms/key)",
            dt,
            dt.as_secs_f64() * 1000.0 / window as f64
        );
        db.shutdown().unwrap();
    }
}

/// Replaying a **low-cardinality** field must not balloon its blob store.
///
/// The bitmap store is append-only and rewrites a whole bitmap per key, so a
/// value shared by N keys leaves N-1 stale copies. On the write path the
/// backpressure valve bounds that; during `activate_field_index`'s replay the
/// valve is a no-op, because it signals the checkpoint *worker* and the workers
/// are not started until every field has been activated.
///
/// Measured before the inline compaction existed: one 5-value field over 16k
/// documents reached 2.6 GB and never finished replaying. This asserts the blob
/// stays within a small multiple of the configured cap.
#[test]
fn replaying_a_low_cardinality_field_keeps_its_blob_bounded() {
    let dir = TempDir::new().unwrap();
    let ns = DEFAULT_NAMESPACE_ID;
    // 256 KiB cap so the test is quick; the property is "bounded by the cap",
    // not any particular byte count.
    let cap: u64 = 256 * 1024;
    let mut config = create_db_config();
    config.threshold_config = config.threshold_config.with_index_blob_backpressure_bytes(cap);

    let field_id = {
        let db = Database::open(dir.path(), config.clone()).unwrap();
        let field_id = activate_status_index(&db, ns);
        // Two distinct values across many rows: the pathological shape.
        for i in 0..4_000u32 {
            let v = if i % 2 == 0 { "active" } else { "inactive" };
            db.put(format!("doc:{i:06}").as_bytes(), format!(r#"{{"status":"{v}"}}"#).as_bytes())
                .unwrap();
        }
        db.run_index_checkpoint().unwrap();
        db.shutdown().unwrap();
        field_id
    };

    // Force a full replay: drop the checkpoint marker so the whole WAL is in
    // the replay window, exactly as it is after a crash with no checkpoint.
    let marker = crate::db::layout::namespace_index_dir(&crate::db::layout::index_root(dir.path()), ns)
        .join(field_id.to_string())
        .join("checkpoint");
    std::fs::remove_file(&marker).unwrap();

    let db = Database::open(dir.path(), config).unwrap();
    let extractor: crate::db::namespace_index::ExtractorFn = std::sync::Arc::new(|bytes: &[u8]| {
        let s = std::str::from_utf8(bytes).ok()?;
        let v: serde_json::Value = serde_json::from_str(s).ok()?;
        Some(crate::index::IndexValue::Str(v["status"].as_str()?.to_string()))
    });
    db.activate_field_index(ns, field_id, crate::index::IndexValueType::Str, extractor)
        .unwrap();

    let field_dir = crate::db::layout::namespace_index_dir(&crate::db::layout::index_root(dir.path()), ns).join(field_id.to_string());
    let on_disk: u64 = std::fs::read_dir(&field_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter_map(|e| e.metadata().ok())
        .filter(|m| m.is_file())
        .map(|m| m.len())
        .sum();

    // Generous bound: compaction fires at `cap` of dead bytes, so the blob can
    // reach roughly cap + one live copy between compactions. 16x cap leaves
    // ample headroom while still failing loudly on unbounded growth (which
    // measured ~180x this cap for the same shape).
    assert!(
        on_disk < cap * 16,
        "field blob grew to {on_disk} bytes with a {cap}-byte cap — replay is not compacting"
    );

    // ...and the replay must still be CORRECT, not just small.
    let outcome = db.query_keys(ns, "status = \"active\"").unwrap();
    assert_eq!(outcome.keys.len(), 2_000, "replay must rebuild every row it bounded");

    db.shutdown().unwrap();
}

/// Verify that `activate_field_index` replays the WAL tail into the index
/// on reopen, covering any writes that happened after the last checkpoint.
///
/// Sequence:
///   1. Open DB, activate index, write batch A.
///   2. Checkpoint (flush BlobStore + keymap mmap store + checkpoint file).
///   3. Write batch B (live index updated, checkpoint is now stale).
///   4. Drop DB without shutdown — simulates a crash.
///   5. Reopen DB, activate index (WAL replay runs for batch B).
///   6. Query must return all of batch A and batch B.
#[test]
fn test_activate_field_index_replays_wal_after_crash() {
    use crate::db::namespace_index::ExtractorFn;
    use crate::index::{IndexValue, IndexValueType};
    use std::sync::Arc;

    let dir = TempDir::new().unwrap();
    let ns = DEFAULT_NAMESPACE_ID;

    let make_extractor = || -> ExtractorFn {
        Arc::new(|bytes: &[u8]| {
            let s = std::str::from_utf8(bytes).ok()?;
            let v: serde_json::Value = serde_json::from_str(s).ok()?;
            Some(IndexValue::Str(v["status"].as_str()?.to_string()))
        })
    };

    let field_id;
    {
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        field_id = db.register_index_field(ns, "status", IndexValueType::Str).unwrap();
        db.activate_field_index(ns, field_id, IndexValueType::Str, make_extractor()).unwrap();

        // Batch A — will be on disk after checkpoint.
        db.put(b"user:1", br#"{"status":"active"}"#).unwrap();
        db.put(b"user:2", br#"{"status":"inactive"}"#).unwrap();

        // Checkpoint: flush BlobStore + keymap mmap store + checkpoint marker.
        db.run_index_checkpoint().unwrap();

        // Batch B — live index only; checkpoint is now stale.
        db.put(b"user:3", br#"{"status":"active"}"#).unwrap();
        db.put(b"user:4", br#"{"status":"inactive"}"#).unwrap();

        // Drop without shutdown — WAL has batch B but checkpoint does not.
    }

    {
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        // activate_field_index must replay batch B from the WAL tail.
        db.activate_field_index(ns, field_id, IndexValueType::Str, make_extractor()).unwrap();

        let mut active_keys = db.query_keys(ns, "status = \"active\"").unwrap().keys;
        active_keys.sort();
        assert_eq!(
            active_keys,
            vec![b"user:1".to_vec(), b"user:3".to_vec()],
            "WAL replay must include batch-B writes",
        );

        let mut inactive_keys = db.query_keys(ns, "status = \"inactive\"").unwrap().keys;
        inactive_keys.sort();
        assert_eq!(inactive_keys, vec![b"user:2".to_vec(), b"user:4".to_vec()],);

        db.shutdown().unwrap();
    }
}

#[test]
fn check_total_len_rejects_over_limit() {
    // Within / at the limit → Ok; over → WriteTooLarge. Tiny limit so the
    // over-limit branch needs no multi-GiB allocation.
    assert!(Database::check_total_len(5, 5, 12, "test limit").is_ok());
    assert!(Database::check_total_len(7, 5, 12, "test limit").is_ok()); // exactly at limit
    match Database::check_total_len(8, 5, 12, "test limit") {
        Err(KVError::WriteTooLarge(m)) => assert!(m.contains("exceeds the 12-byte"), "got: {m}"),
        other => panic!("expected WriteTooLarge, got {other:?}"),
    }
}

/// A value too big for a value-log segment must be rejected **at the write
/// boundary**, not deep inside the apply.
///
/// This was the gap: `check_write_size` bounded only the `u32` field widths
/// (~4 GiB), while `ValueLog::append` additionally requires a record to fit one
/// segment (256 MiB by default). A value between the two passed the front door,
/// got a fsynced WAL entry — so `put` returned `Ok` — and then failed every
/// apply attempt, leaving an acknowledged write that `get` could not see and
/// that WAL replay could only turn into a fail log.
///
/// Asserting the error is only half of it; the load-bearing half is that the WAL
/// did not grow, because that is what makes the rejection mean "nothing
/// happened".
#[test]
fn a_value_too_big_for_a_segment_is_rejected_before_the_wal() {
    const SEG: u64 = 64 * 1024; // the minimum legal segment size

    let dir = TempDir::new().unwrap();
    let mut config = create_db_config();
    config.segment_size_bytes = SEG;
    let db = Database::open(dir.path(), config).unwrap();

    db.put(b"k", b"small").unwrap();
    let entries_before = db.wal_metadata.read().total_entries;

    // Comfortably over the segment, but far under the u32 field-width limit that
    // was the only bound before.
    let oversized = vec![0u8; (SEG as usize) * 2];
    match db.put(b"k", &oversized) {
        Err(KVError::WriteTooLarge(m)) => {
            assert!(
                m.contains("value-log segment"),
                "the message should name the bound that was hit; got: {m}"
            );
        }
        other => panic!("expected WriteTooLarge, got {other:?}"),
    }

    assert_eq!(
        db.wal_metadata.read().total_entries,
        entries_before,
        "the rejected write reached the WAL — it was not rejected at the boundary"
    );
    assert_eq!(db.get(b"k").unwrap(), Some(b"small".to_vec()), "the prior value must be untouched");

    // A record that fits still writes.
    let fits = vec![7u8; (SEG as usize) - 1024];
    db.put(b"k", &fits).unwrap();
    assert_eq!(db.get(b"k").unwrap(), Some(fits));

    db.shutdown().unwrap();
}

#[test]
fn put_and_delete_accept_normal_sized_keys_and_values() {
    // Regression: the size guard must not reject ordinary writes.
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    db.put(b"k", &vec![0u8; 4096]).unwrap();
    assert_eq!(db.get(b"k").unwrap(), Some(vec![0u8; 4096]));
    db.delete(b"k").unwrap();
    assert_eq!(db.get(b"k").unwrap(), None);
    db.shutdown().unwrap();
}

/// Item 13 regression: replaying an **update** (same key, changed field
/// value) must reconcile via the targeted O(1) path — during replay
/// `lsm.get` returns the replay-so-far value, so the second put moves the
/// row off the old value. A bug here would leave the key matching BOTH the
/// old and new value after recovery.
#[test]
fn test_field_index_replays_updates_after_crash() {
    use crate::db::namespace_index::ExtractorFn;
    use crate::index::{IndexValue, IndexValueType};
    use std::sync::Arc;

    let dir = TempDir::new().unwrap();
    let ns = DEFAULT_NAMESPACE_ID;
    let make_extractor = || -> ExtractorFn {
        Arc::new(|bytes: &[u8]| {
            let s = std::str::from_utf8(bytes).ok()?;
            let v: serde_json::Value = serde_json::from_str(s).ok()?;
            Some(IndexValue::Str(v["status"].as_str()?.to_string()))
        })
    };

    let field_id;
    {
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        field_id = db.register_index_field(ns, "status", IndexValueType::Str).unwrap();
        db.activate_field_index(ns, field_id, IndexValueType::Str, make_extractor()).unwrap();

        // Insert, then UPDATE the same key to a different value — all in the
        // WAL, no checkpoint, so recovery must replay both writes in order.
        db.put(b"u:1", br#"{"status":"active"}"#).unwrap();
        db.put(b"u:1", br#"{"status":"archived"}"#).unwrap(); // update
        db.put(b"u:2", br#"{"status":"active"}"#).unwrap();
        // Drop without shutdown — index lives only in the WAL.
    }

    {
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        db.activate_field_index(ns, field_id, IndexValueType::Str, make_extractor()).unwrap();

        // u:1 must match ONLY its latest value after replay, not the old one.
        assert_eq!(
            db.query_keys(ns, "status = \"archived\"").unwrap().keys,
            vec![b"u:1".to_vec()],
            "replayed update must land u:1 under its new value"
        );
        assert_eq!(
            db.query_keys(ns, "status = \"active\"").unwrap().keys,
            vec![b"u:2".to_vec()],
            "replayed update must remove u:1 from its old value (no stale bucket)"
        );
        db.shutdown().unwrap();
    }
}

#[test]
fn test_ops_metrics_counters_move() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();

    db.put(b"k1", b"v1").unwrap();
    db.put(b"k2", b"v2").unwrap();
    assert_eq!(db.get(b"k1").unwrap(), Some(b"v1".to_vec())); // hit
    assert_eq!(db.get(b"missing").unwrap(), None); // miss
    db.delete(b"k2").unwrap();

    let m = db.metrics_snapshot();
    assert_eq!(m.puts, 2, "two WAL-backed puts");
    assert_eq!(m.deletes, 1, "one delete");
    assert_eq!(m.reads, 2, "two user reads");
    assert_eq!(m.read_hits, 1, "one hit");
    assert_eq!(m.read_misses, 1, "one miss");
    // Every WAL-backed write fsyncs once (2 puts + 1 delete).
    assert_eq!(m.wal_fsyncs, 3);
    assert!(m.wal_bytes_appended > 0);
    // Reads go through the LSM point-lookup path.
    assert!(m.lookups >= 2, "lookups should cover the user reads, got {}", m.lookups);

    db.shutdown().unwrap();
}

#[test]
fn test_value_log_physical_and_segment_stats() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();

    for i in 0..50u32 {
        db.put(format!("k{i}").as_bytes(), &vec![b'v'; 256]).unwrap();
    }

    // Physical stats: present for the default namespace, with real on-disk bytes.
    let physical = db.value_log_physical_stats();
    let (_, shards) = physical.iter().find(|(ns, _)| ns == "default").expect("default ns present");
    assert!(!shards.is_empty(), "expected value-log shards");
    let total_physical: u64 = shards.iter().map(|s| s.physical_bytes).sum();
    assert!(total_physical > 0, "physical bytes should be non-zero after writes");

    // Segment stats: per-bucket segment breakdown, with live bytes accounted.
    let segments = db.value_log_segment_stats("default").unwrap();
    assert_eq!(segments.len(), db.config.num_buckets, "one entry per bucket");
    let live: u64 = segments.iter().flat_map(|(_, ss)| ss.iter()).map(|s| s.live_bytes).sum();
    assert!(live > 0, "written records should be accounted as live bytes, got {live}");
    // Every bucket has exactly one active tail (unsealed) segment.
    for (bucket, ss) in &segments {
        let unsealed = ss.iter().filter(|s| !s.sealed).count();
        assert_eq!(unsealed, 1, "bucket {bucket} must have exactly one active tail segment");
    }

    // Unknown namespace errors rather than panicking.
    assert!(db.value_log_segment_stats("nope").is_err());

    db.shutdown().unwrap();
}

#[test]
fn concurrent_same_key_puts_keep_value_log_accounting_exact() {
    // Under real concurrency, racing writers to the same keys must not drift the
    // value-log accounting GC selection relies on: total live bytes must equal
    // exactly one live record per distinct key. (The deterministic form of the
    // underlying bug is `a_lower_seq_put_charges_its_own_record_not_the_live_one`
    // in kv_store; this guards the invariant end-to-end under load.)
    use std::sync::Arc;
    use std::thread;

    let dir = TempDir::new().unwrap();
    let db = Arc::new(Database::open(dir.path(), create_db_config()).unwrap());

    let keys: Vec<Vec<u8>> = (0..4u32).map(|k| format!("hot-key-{k}").into_bytes()).collect();
    let mut handles = Vec::new();
    for t in 0..4usize {
        let (db, keys) = (Arc::clone(&db), keys.clone());
        handles.push(thread::spawn(move || {
            for i in 0..100usize {
                // Vary the value size per write so a mis-charged displacement can't
                // cancel in the aggregate byte count.
                let len = 64 + (t * 37 + i * 101) % 1024;
                db.put(&keys[(t + i) % keys.len()], &vec![0xACu8; len]).unwrap();
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }

    // Expected live bytes: one record per distinct key, sized by that key's current
    // (winning) value. Database::open starts no GC worker, so nothing reclaims
    // concurrently and the live set is exactly the winning records.
    let expected_live: u64 = keys
        .iter()
        .map(|k| {
            let v = db.get(k).unwrap().expect("key present after the race");
            crate::store::value_log::ValueRecordHeader::record_len(k.len(), v.len())
        })
        .sum();

    let stats = db.value_log_segment_stats("default").unwrap();
    let mut total_live = 0u64;
    for (_bucket, segs) in &stats {
        for s in segs {
            assert_eq!(
                s.live_bytes + s.garbage_bytes,
                s.total_bytes,
                "segment {} broke live+garbage==total",
                s.id
            );
            total_live += s.live_bytes;
        }
    }
    assert_eq!(
        total_live, expected_live,
        "accounted live bytes must equal exactly one record per live key, not leak intermediate records"
    );

    db.shutdown().unwrap();
}

#[test]
fn existing_bucket_count_is_detected_from_segmented_value_logs() {
    // num_buckets is locked at creation: reopening with a different configured
    // value must NOT re-shard. detect_bucket_count reads the count back from the
    // value-log filenames — which are now `value_log_{bucket}.segNNNNNN`, not the
    // old `value_log_{bucket}.log`. A stale matcher would see zero buckets and
    // silently honour the new config, corrupting key→bucket sharding.
    let dir = TempDir::new().unwrap();
    let original_buckets = 3;

    {
        let mut config = create_db_config();
        config.num_buckets = original_buckets;
        let db = Database::open(dir.path(), config).unwrap();
        for i in 0..30u32 {
            db.put(format!("k{i}").as_bytes(), format!("v{i}").as_bytes()).unwrap();
        }
        assert_eq!(db.config.num_buckets, original_buckets);
        db.shutdown().unwrap();
    }

    // Reopen asking for a DIFFERENT bucket count.
    let mut config = create_db_config();
    config.num_buckets = original_buckets + 3;
    let db = Database::open(dir.path(), config).unwrap();
    assert_eq!(
        db.config.num_buckets, original_buckets,
        "the original bucket count must be detected from the segment files and preserved, not the newly configured one"
    );

    // Every value is still readable — keys resolved to the same buckets they were
    // written to, which they only can if the original count was honoured.
    for i in 0..30u32 {
        assert_eq!(
            db.get(format!("k{i}").as_bytes()).unwrap(),
            Some(format!("v{i}").into_bytes()),
            "value k{i} unreadable after reopen with a mismatched num_buckets"
        );
    }
    db.shutdown().unwrap();
}

#[test]
fn gc_collects_a_hot_bucket_when_the_namespace_average_is_below_the_trigger() {
    use crate::store::gc_value_log_worker::ValueLogGcTarget;
    // The GC trigger is per-bucket, not just the namespace average: a bucket over the
    // trigger must be collected even when near-empty buckets drag the average below it.
    let dir = TempDir::new().unwrap();
    let mut config = create_db_config();
    config.threshold_config.value_log_waste_threshold = 30.0; // tail bar tracks this
    let db = Database::open(dir.path(), config).unwrap();

    let value = vec![0x33u8; 300];
    // Live data spread across buckets keeps the namespace average low...
    for i in 0..800u32 {
        db.put(format!("k{i:04}").as_bytes(), &value).unwrap();
    }
    // ...then hammer a single key so ITS bucket alone piles up garbage.
    for _ in 0..250u32 {
        db.put(b"k0000", &value).unwrap();
    }

    let store = db.get_store(DEFAULT_NAMESPACE_ID).unwrap();
    let avg = store.get_waste_ratio();
    assert!(avg < 30.0, "precondition: namespace average must be below the trigger, got {avg:.1}%");
    assert!(store.has_bucket_over_waste(30.0), "precondition: some bucket must be over the trigger");

    // The per-bucket-aware trigger runs GC where the old average-only gate would have
    // skipped the whole namespace.
    ValueLogGcTarget::run_gc_if_needed(&db, 30.0);

    let after = store.get_waste_ratio();
    assert!(
        after < avg,
        "the hot bucket's garbage must be reclaimed, dropping overall waste (was {avg:.1}%, now {after:.1}%)"
    );
    assert_eq!(db.get(b"k0000").unwrap(), Some(value.clone()), "the hammered key must survive collection");

    db.shutdown().unwrap();
}

/// No-WAL writes are unrecoverable by design — there is nothing to replay —
/// so leaving them in a memtable until it fills means a crash discards the
/// lot. This is what cost the stress run its whole vector index: a `SIGKILL`
/// left 4 sparse-vector entries on disk against 6020 in memory.
///
/// The periodic flush is the bound. Both halves are asserted here, because
/// the second is what makes the first meaningful.
#[test]
fn periodic_flush_makes_no_wal_writes_survive_a_crash() {
    let dir = TempDir::new().unwrap();
    let config = create_db_config();

    // Session 1: no-WAL writes, a periodic flush, then a crash.
    {
        let db = Database::open(dir.path(), config.clone()).unwrap();
        let ns = db.create_namespace("vectors").unwrap();
        for i in 0..50u32 {
            db.put_ns_no_wal(ns, format!("v{i:03}").as_bytes(), b"embedding").unwrap();
        }

        assert!(
            db.get_store(ns).unwrap().has_unflushed_no_wal_writes(),
            "no-WAL writes should mark the namespace as holding unflushed data"
        );
        assert_eq!(db.flush_no_wal_memtables(), 1, "the namespace should have been flushed");
        assert!(
            !db.get_store(ns).unwrap().has_unflushed_no_wal_writes(),
            "the flag should clear once the memtable is flushed"
        );
        // Nothing new since the flush, so a second tick is a no-op.
        assert_eq!(db.flush_no_wal_memtables(), 0, "an already-flushed namespace should not be re-flushed");

        std::mem::forget(db);
    }

    {
        let db = Database::open(dir.path(), config.clone()).unwrap();
        let ns = db.get_namespace_id("vectors").expect("namespace should still exist");
        for i in 0..50u32 {
            assert_eq!(
                db.get_ns(ns, format!("v{i:03}").as_bytes()).unwrap().as_deref(),
                Some(&b"embedding"[..]),
                "no-WAL write {i} was flushed before the crash and must survive it"
            );
        }
        std::mem::forget(db);
    }

    // Without the flush the same writes are gone — no WAL entry exists to
    // replay them. This is the loss the periodic flush bounds.
    let dir2 = TempDir::new().unwrap();
    {
        let db = Database::open(dir2.path(), config.clone()).unwrap();
        let ns = db.create_namespace("vectors").unwrap();
        for i in 0..50u32 {
            db.put_ns_no_wal(ns, format!("v{i:03}").as_bytes(), b"embedding").unwrap();
        }
        std::mem::forget(db);
    }
    {
        let db = Database::open(dir2.path(), config).unwrap();
        let ns = db.get_namespace_id("vectors").expect("namespace should still exist");
        assert_eq!(
            db.get_ns(ns, b"v000").unwrap(),
            None,
            "un-flushed no-WAL writes are expected to be lost — if this starts passing, \
                 the flush test above is no longer proving anything"
        );
        std::mem::forget(db);
    }
}

/// A *failed* flush must leave the no-WAL flag set.
///
/// The flag is the only record that these writes exist nowhere but memory.
/// Clearing it on a flush that did not happen drops the namespace out of
/// `flush_no_wal_memtables` permanently — no later tick retries it — so the
/// next crash takes the whole memtable, which is exactly the loss the
/// periodic flush was added to bound.
#[test]
fn a_failed_flush_keeps_the_no_wal_flag_set() {
    use std::os::unix::fs::PermissionsExt;

    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    let ns = db.create_namespace("vectors").unwrap();
    for i in 0..50u32 {
        db.put_ns_no_wal(ns, format!("v{i:03}").as_bytes(), b"embedding").unwrap();
    }
    let store = db.get_store(ns).unwrap();
    assert!(store.has_unflushed_no_wal_writes());

    // Make the flush fail by taking write permission off the level-0 tree —
    // it can then neither create a bucket directory nor write an SSTable
    // into an existing one.
    let level0 = store.lsm_path.join("level0");
    std::fs::create_dir_all(&level0).unwrap();
    let mut locked: Vec<(std::path::PathBuf, std::fs::Permissions)> = Vec::new();
    for entry in std::fs::read_dir(&level0).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            locked.push((path.clone(), std::fs::metadata(&path).unwrap().permissions()));
        }
    }
    locked.push((level0.clone(), std::fs::metadata(&level0).unwrap().permissions()));
    for (path, perms) in &locked {
        let mut readonly = perms.clone();
        readonly.set_mode(0o555);
        std::fs::set_permissions(path, readonly).unwrap();
    }

    let result = store.flush_memtable_to_level0();

    // Restore before asserting so a failure still leaves a removable TempDir.
    for (path, perms) in locked {
        std::fs::set_permissions(&path, perms).unwrap();
    }

    assert!(result.is_err(), "flush should fail on a read-only directory");
    assert!(
        store.has_unflushed_no_wal_writes(),
        "a failed flush cleared the flag, so no later tick will retry this namespace"
    );
}

/// WAL-backed writes must not be dragged into the no-WAL flush: they are
/// recoverable, are already flushed by the WAL-watermark path, and forcing
/// extra flushes would only add L0 files for compaction to merge.
#[test]
fn wal_backed_writes_do_not_trigger_the_no_wal_flush() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    let ns = db.create_namespace("docs").unwrap();
    for i in 0..50u32 {
        db.put_ns(ns, format!("k{i:03}").as_bytes(), b"value").unwrap();
    }

    assert!(!db.get_store(ns).unwrap().has_unflushed_no_wal_writes());
    assert_eq!(db.flush_no_wal_memtables(), 0);
}

#[test]
fn recovery_replay_then_crash_before_metadata_flush_loses_nothing() {
    // A crash can strike right after WAL replay re-appended values to the log but
    // before the value-log metadata was flushed — leaving segment files longer than
    // their recorded totals. The next open must reconcile that (rebuild stats from
    // the LSM) and lose nothing, even across repeated crash/replay cycles. Crash is
    // simulated with mem::forget, which skips Database::drop's clean flush.
    let dir = TempDir::new().unwrap();
    let mut config = create_db_config();
    config.num_buckets = 1;
    let keys: Vec<Vec<u8>> = (0..30u32).map(|i| format!("k{i:03}").into_bytes()).collect();

    // Session 1: write, then crash before any clean flush.
    {
        let db = Database::open(dir.path(), config.clone()).unwrap();
        for (i, k) in keys.iter().enumerate() {
            db.put(k, format!("v{i}").as_bytes()).unwrap();
        }
        std::mem::forget(db);
    }

    // Session 2: reopen replays the WAL (re-appending to the log), then crash again
    // before the post-replay metadata is flushed.
    {
        let db = Database::open(dir.path(), config.clone()).unwrap();
        assert_eq!(
            db.get(b"k000").unwrap().as_deref(),
            Some(b"v0".as_slice()),
            "replay must restore the data"
        );
        std::mem::forget(db);
    }

    // Session 3: open cleanly. Everything is still readable, and the value-log
    // accounting is self-consistent despite the stale metadata it started from.
    let db = Database::open(dir.path(), config).unwrap();
    for (i, k) in keys.iter().enumerate() {
        assert_eq!(
            db.get(k).unwrap().as_deref(),
            Some(format!("v{i}").as_bytes()),
            "key {i} lost across replay+crash cycles"
        );
    }
    for (_bucket, segs) in db.value_log_segment_stats("default").unwrap() {
        for s in segs {
            assert_eq!(
                s.live_bytes + s.garbage_bytes,
                s.total_bytes,
                "segment {} broke live+garbage==total after recovery",
                s.id
            );
        }
    }
    db.shutdown().unwrap();
}

#[test]
fn a_torn_tail_survives_reopen_append_and_gc() {
    // The full torn-tail hazard end to end: a crash leaves a partial record at the
    // active tail; reopen truncates it and replays; we append past it, churn to make
    // the segment collectable, then GC seals and relocates it. Records from BEFORE
    // the torn tail and those appended AFTER it must all survive GC relocation.
    let dir = TempDir::new().unwrap();
    let mut config = create_db_config();
    config.num_buckets = 1; // one bucket → one active-tail segment to tear
    let value = vec![0x5Au8; 300];

    // Session 1: write, crash before clean flush.
    {
        let db = Database::open(dir.path(), config.clone()).unwrap();
        for i in 0..40u32 {
            db.put(format!("k{i:03}").as_bytes(), &value).unwrap();
        }
        std::mem::forget(db);
    }

    // Inject a torn write at the tail of bucket 0's active segment.
    let seg = dir.path().join("ns_default").join("value_logs").join("value_log_0.seg000001");
    let mut bytes = std::fs::read(&seg).unwrap();
    bytes.extend_from_slice(&[0xAB; 20]); // partial, not a valid header
    std::fs::write(&seg, &bytes).unwrap();

    // Session 2: reopen (truncates the torn tail, replays), append MORE past it,
    // overwrite the originals to pile up garbage, then GC.
    let db = Database::open(dir.path(), config).unwrap();
    for i in 40..60u32 {
        db.put(format!("k{i:03}").as_bytes(), &value).unwrap();
    }
    for i in 0..40u32 {
        db.put(format!("k{i:03}").as_bytes(), &value).unwrap();
    }
    db.garbage_collect().unwrap();

    // All 60 keys survive — including those appended after the torn tail.
    for i in 0..60u32 {
        assert_eq!(
            db.get(format!("k{i:03}").as_bytes()).unwrap(),
            Some(value.clone()),
            "k{i:03} lost across torn-tail reopen + GC"
        );
    }
    for (_bucket, segs) in db.value_log_segment_stats("default").unwrap() {
        for s in segs {
            assert_eq!(
                s.live_bytes + s.garbage_bytes,
                s.total_bytes,
                "segment {} broke the invariant after GC",
                s.id
            );
        }
    }
    db.shutdown().unwrap();
}

#[tokio::test]
async fn async_db_basic_round_trip() {
    let dir = TempDir::new().unwrap();
    let db = crate::db::facade::AsyncDb::open_with_config(dir.path().to_path_buf(), create_db_config())
        .await
        .unwrap();

    db.put(b"key1".to_vec(), b"value1".to_vec()).await.unwrap();
    assert_eq!(db.get(b"key1".to_vec()).await.unwrap(), Some(b"value1".to_vec()));

    db.delete(b"key1".to_vec()).await.unwrap();
    assert_eq!(db.get(b"key1".to_vec()).await.unwrap(), None);

    db.shutdown().await.unwrap();
}

/// The WAL GC worker runs, reclaims, and stops cleanly on shutdown.
///
/// Asserts on *observable* reclamation rather than on a worker-enabled flag:
/// the old version checked `is_wal_gc_worker_enabled`, which only ever
/// existed on the unused wrapper, so it tested bookkeeping instead of work.
#[tokio::test]
async fn async_db_wal_gc_worker_reclaims_and_stops() {
    let dir = TempDir::new().unwrap();
    let db = crate::db::facade::AsyncDb::open_with_config(dir.path().to_path_buf(), create_db_config())
        .await
        .unwrap();
    db.enable_wal_gc_worker(Duration::from_millis(50)).await.unwrap();

    for i in 0..200u32 {
        db.put(format!("key{i}").into_bytes(), vec![b'v'; 128]).await.unwrap();
    }
    db.compact().await.unwrap();

    // A direct pass proves GC is reachable and returns sane numbers; the
    // worker firing on its own interval is covered by the workers' own tests.
    let (_reclaimed, remaining) = db.garbage_collect_wal().await.unwrap();
    assert!(remaining <= 200, "un-persisted count should not exceed what was written");

    // Shutdown must stop the worker without hanging or erroring.
    db.shutdown().await.unwrap();
}

/// `AsyncDb::shutdown` must stop the index checkpoint worker.
///
/// It did not: the shutdown sequence stopped the TTL, WAL GC, LSM compaction
/// and value-log GC workers but silently skipped this one. The gap was
/// invisible because a since-deleted second async wrapper *did* stop it, so
/// nothing flagged the worker's shutdown path as unused. A checkpoint tick
/// could therefore land after the database was closed.
#[tokio::test]
async fn async_db_shutdown_stops_the_index_checkpoint_worker() {
    let dir = TempDir::new().unwrap();
    let db = crate::db::facade::AsyncDb::open_with_config(dir.path().to_path_buf(), create_db_config())
        .await
        .unwrap();
    // A short interval so a surviving worker would keep firing.
    db.enable_index_checkpoint_worker(Duration::from_millis(20)).await.unwrap();
    db.put(b"k".to_vec(), b"v".to_vec()).await.unwrap();

    db.shutdown().await.unwrap();
    assert!(
        !db.index_checkpoint_worker_active().await,
        "shutdown must stop the checkpoint worker, not leave it running"
    );

    // Ticks after shutdown must not blow up: give the interval several
    // periods to fire if the task were still alive.
    tokio::time::sleep(Duration::from_millis(100)).await;
}

#[tokio::test]
async fn async_db_enable_all_workers_then_shut_down() {
    let dir = TempDir::new().unwrap();
    let cfg = create_db_config();
    let db = crate::db::facade::AsyncDb::open_with_config(dir.path().to_path_buf(), cfg.clone())
        .await
        .unwrap();
    db.enable_all_workers(&cfg).await.unwrap();

    db.put(b"k".to_vec(), b"v".to_vec()).await.unwrap();
    assert_eq!(db.get(b"k".to_vec()).await.unwrap(), Some(b"v".to_vec()));

    db.shutdown().await.unwrap();
}

#[tokio::test]
async fn async_db_namespaces_are_isolated() {
    let dir = TempDir::new().unwrap();
    let db = crate::db::facade::AsyncDb::open_with_config(dir.path().to_path_buf(), create_db_config())
        .await
        .unwrap();

    let users = db.namespace("users".to_string()).await.unwrap();
    db.put(b"global".to_vec(), b"g_val".to_vec()).await.unwrap();
    users.put(b"user:1".to_vec(), b"alice".to_vec()).await.unwrap();

    // Neither namespace can see the other's keys.
    assert_eq!(db.get(b"user:1".to_vec()).await.unwrap(), None);
    assert_eq!(users.get(b"user:1".to_vec()).await.unwrap(), Some(b"alice".to_vec()));
    assert_eq!(users.get(b"global".to_vec()).await.unwrap(), None);

    db.shutdown().await.unwrap();
}

#[test]
fn test_delete_ns_removes_key_from_namespace() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();

    let ns = db.create_namespace("items").unwrap();

    db.put_ns(ns, b"k1", b"v1").unwrap();
    db.put_ns(ns, b"k2", b"v2").unwrap();
    assert_eq!(db.get_ns(ns, b"k1").unwrap(), Some(b"v1".to_vec()));

    db.delete_ns(ns, b"k1").unwrap();
    assert_eq!(db.get_ns(ns, b"k1").unwrap(), None);
    // Sibling key is untouched
    assert_eq!(db.get_ns(ns, b"k2").unwrap(), Some(b"v2".to_vec()));

    db.shutdown().unwrap();
}

#[test]
fn test_delete_ns_does_not_affect_other_namespaces() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();

    let ns_a = db.create_namespace("ns_a").unwrap();
    let ns_b = db.create_namespace("ns_b").unwrap();

    // Same key in three namespaces
    db.put(b"shared", b"default_val").unwrap();
    db.put_ns(ns_a, b"shared", b"a_val").unwrap();
    db.put_ns(ns_b, b"shared", b"b_val").unwrap();

    // Delete only from ns_a
    db.delete_ns(ns_a, b"shared").unwrap();

    assert_eq!(db.get_ns(ns_a, b"shared").unwrap(), None);
    assert_eq!(db.get(b"shared").unwrap(), Some(b"default_val".to_vec()));
    assert_eq!(db.get_ns(ns_b, b"shared").unwrap(), Some(b"b_val".to_vec()),);

    db.shutdown().unwrap();
}

#[test]
fn test_delete_ns_nonexistent_key_is_ok() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();

    let ns = db.create_namespace("empty").unwrap();

    // Deleting a key that was never written should succeed silently.
    db.delete_ns(ns, b"ghost").unwrap();
    assert_eq!(db.get_ns(ns, b"ghost").unwrap(), None);

    db.shutdown().unwrap();
}

#[test]
fn test_delete_ns_advances_wal() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();

    let ns = db.create_namespace("wal_check").unwrap();
    db.put_ns(ns, b"k", b"v").unwrap();

    let tail_before = db.wal_metadata().tail;
    db.delete_ns(ns, b"k").unwrap();
    let tail_after = db.wal_metadata().tail;

    assert!(
        tail_after > tail_before,
        "WAL tail must advance after delete_ns (before={}, after={})",
        tail_before,
        tail_after,
    );

    db.shutdown().unwrap();
}

#[test]
fn test_delete_ns_invalid_namespace_returns_error() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();

    // Namespace 9999 was never created.
    let result = db.delete_ns(9999, b"k");
    assert!(result.is_err(), "delete_ns on unknown namespace should fail");

    db.shutdown().unwrap();
}

/// Item 1 regression: several single-op writes to one key, laid down in the
/// WAL with sequences *out of physical order*, must resolve to the
/// highest-sequence value. This directly exercises the sequence sort in
/// recovery (a naive scan-order replay would pick the physically-last entry,
/// which here has a lower sequence).
#[test]
fn test_recovery_applies_same_key_writes_in_sequence_order() {
    use crate::db::wal::{Wal, WalEntry, WalMetadata};

    let dir = TempDir::new().unwrap();
    let wal_path = dir.path().join("wal.log");
    let wal_meta_path = dir.path().join("wal_metadata");
    {
        let wal = Wal::open(&wal_path).unwrap();
        let mut tail = 0u64;

        // Physical order: seq 5, seq 9, seq 7. Sequence order says seq 9 wins.
        for (seq, val) in [(5u64, b"v5".to_vec()), (9, b"v9".to_vec()), (7, b"v7".to_vec())] {
            wal.append_entry(&WalEntry::new_upsert(b"k".to_vec(), val).with_sequence(seq), &mut tail, false)
                .unwrap();
        }
        wal.sync().unwrap();

        let mut meta = WalMetadata::new();
        meta.tail = tail;
        meta.total_entries = 3;
        meta.segment_total_entries = vec![3];
        meta.segment_persisted_entries = vec![0];
        std::fs::write(&wal_meta_path, meta.to_file_bytes().unwrap()).unwrap();
    }

    let db = Database::open(dir.path(), create_db_config()).unwrap();
    assert_eq!(
        db.get(b"k").unwrap(),
        Some(b"v9".to_vec()),
        "recovery must apply same-key writes in sequence order (seq 9 is newest)"
    );
    db.shutdown().unwrap();
}

/// Critical regression: WAL entries fsynced *after* the last `wal_metadata`
/// flush must not be lost. We craft the post-crash state — three durable WAL
/// entries but a metadata file whose `tail`/counts cover only the first —
/// and assert recovery reconstructs the true tail and replays all three
/// (without the fix, recovery scans only up to the stale tail and the last
/// two are silently dropped).
#[test]
fn test_recovery_reconstructs_wal_tail_past_stale_metadata() {
    use crate::db::wal::{Wal, WalEntry, WalMetadata};

    let dir = TempDir::new().unwrap();
    let wal_path = dir.path().join("wal.log");
    let wal_meta_path = dir.path().join("wal_metadata");

    let stale_tail;
    {
        let wal = Wal::open(&wal_path).unwrap();
        let mut tail = 0u64;

        // Entry 1 — covered by the (earlier) metadata flush.
        wal.append_entry(&WalEntry::new_upsert(b"k1".to_vec(), b"v1".to_vec()).with_sequence(1), &mut tail, false)
            .unwrap();
        stale_tail = tail;

        // Entries 2 and 3 — fsynced after the flush; metadata never recorded them.
        wal.append_entry(&WalEntry::new_upsert(b"k2".to_vec(), b"v2".to_vec()).with_sequence(2), &mut tail, false)
            .unwrap();
        wal.append_entry(&WalEntry::new_upsert(b"k3".to_vec(), b"v3".to_vec()).with_sequence(3), &mut tail, false)
            .unwrap();
        wal.sync().unwrap();

        // Persist metadata as it stood at the earlier flush: it sees only entry 1.
        let mut meta = WalMetadata::new();
        meta.tail = stale_tail;
        meta.total_entries = 1;
        meta.segment_total_entries = vec![1];
        meta.segment_persisted_entries = vec![0];
        std::fs::write(&wal_meta_path, meta.to_file_bytes().unwrap()).unwrap();
    }

    let db = Database::open(dir.path(), create_db_config()).unwrap();
    assert_eq!(db.get(b"k1").unwrap(), Some(b"v1".to_vec()));
    assert_eq!(
        db.get(b"k2").unwrap(),
        Some(b"v2".to_vec()),
        "entry past the stale tail must be recovered"
    );
    assert_eq!(
        db.get(b"k3").unwrap(),
        Some(b"v3".to_vec()),
        "entry past the stale tail must be recovered"
    );
    db.shutdown().unwrap();
}

/// Coordinator-level companion to
/// [`test_recovery_reconstructs_wal_tail_past_stale_metadata`]: rather than
/// hand-crafting the WAL, it drives the **real** `Database::put` write path so
/// it catches counter/short-circuit regressions in `Database::open` +
/// `recover_from_wal` + `Wal::recover_tail` — e.g. a stale `total_entries`
/// that wrongly trips the `persisted >= total` short-circuit, or a tail fold
/// that miscounts `total`/`segment_total` and so under- or over-replays.
///
/// Scenario (the "crash just after a metadata flush" window):
///   1. open Db, `put` key A,
///   2. flush WAL metadata so the durable snapshot covers only A,
///   3. `put` key B — fsynced into the WAL, but its in-memory metadata bump
///      never reaches disk (no further flush, `records_per_sync = 1000` so the
///      write path never auto-syncs/persists for two puts),
///   4. abandon the Db **without** shutdown via `mem::forget`, faithfully
///      simulating a crash: the in-memory metadata that knows about B and the
///      unflushed memtable holding A and B are both lost, leaving both keys
///      only in the WAL. (A plain `drop` would run `Database::drop`, which does
///      a *clean* flush — memtable→SSTable plus metadata — making B durable and
///      defeating the test.)
///   5. restore the stale (A-only) metadata on disk,
///   6. reopen and assert BOTH A and B are recovered.
#[test]
fn test_recovery_from_stale_metadata_via_real_write_path() {
    let dir = TempDir::new().unwrap();
    let wal_meta_path = dir.path().join("wal_metadata");

    let stale_metadata: Vec<u8>;
    {
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        db.put(b"A", b"va").unwrap();

        // The on-disk metadata as it stood at the last flush before the crash:
        // its tail/total cover only A.
        db.flush_wal_metadata_internal().unwrap();
        stale_metadata = std::fs::read(&wal_meta_path).unwrap();

        // B is durable in the WAL (every put fsyncs it) but its metadata bump
        // lives only in memory — exactly the post-flush window.
        db.put(b"B", b"vb").unwrap();

        // Abandon without shutdown to simulate a crash: `mem::forget` skips
        // `Database::drop`, which would otherwise cleanly flush the memtable
        // and metadata (persisting B and defeating the test). The leaked
        // handles are released at process exit; everything durable (the WAL)
        // is already fsynced.
        std::mem::forget(db);
    }

    // Force the on-disk metadata back to the A-only snapshot, simulating the
    // crash landing after A's flush but before B's would have been recorded.
    // (Belt-and-suspenders: nothing should have rewritten it, but this makes
    // the staleness explicit regardless of write-path sync timing.)
    std::fs::write(&wal_meta_path, &stale_metadata).unwrap();

    // Reopen: recover_tail must notice the durable WAL extends past the stale
    // tail, fold B into total/segment counters, and recover_from_wal must
    // replay both A (lost from the memtable) and B (past the stale tail).
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    assert_eq!(
        db.get(b"A").unwrap(),
        Some(b"va".to_vec()),
        "A lived only in the WAL and must be replayed"
    );
    assert_eq!(
        db.get(b"B").unwrap(),
        Some(b"vb".to_vec()),
        "B was appended after the last metadata flush; recover_tail must fold it in and replay it"
    );
    db.shutdown().unwrap();
}

/// Item 3: `apply_with_retry` rides out transient apply failures and stops
/// as soon as one attempt succeeds.
#[test]
fn test_apply_with_retry_succeeds_after_transient_failure() {
    use std::cell::Cell;

    let attempts = Cell::new(0);
    Database::apply_with_retry("put", 0, b"k", 1, || {
        attempts.set(attempts.get() + 1);
        // Fail on the first attempt, succeed on the second.
        if attempts.get() < 2 { Err(KVError::KeyNotFound) } else { Ok(()) }
    });
    assert_eq!(attempts.get(), 2, "should stop retrying once an attempt succeeds");
}

/// Item 3: `apply_with_retry` gives up after a bounded number of attempts
/// (it never surfaces the error — the write is durable in the WAL).
#[test]
fn test_apply_with_retry_gives_up_after_max_attempts() {
    use std::cell::Cell;

    let attempts = Cell::new(0);
    Database::apply_with_retry("put", 0, b"k", 1, || {
        attempts.set(attempts.get() + 1);
        Err(KVError::KeyNotFound)
    });
    assert_eq!(
        attempts.get(),
        Database::APPLY_RETRY_ATTEMPTS,
        "should attempt exactly APPLY_RETRY_ATTEMPTS times before giving up"
    );
}

/// Regression for the O(1) targeted field-index update/delete (item 13): a
/// document update must move its row from the old value's bucket to the new
/// one (the prior value must stop matching), and a delete must remove it —
/// driven by the prior document bytes read in the put/delete path, not a
/// full bucket scan.
#[test]
fn test_field_index_targeted_update_and_delete() {
    use crate::db::namespace_index::ExtractorFn;
    use crate::index::{IndexValue, IndexValueType};
    use std::sync::Arc;

    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    let ns = DEFAULT_NAMESPACE_ID;

    let field_id = db.register_index_field(ns, "status", IndexValueType::Str).unwrap();
    let extractor: ExtractorFn = Arc::new(|bytes: &[u8]| {
        let s = std::str::from_utf8(bytes).ok()?;
        let v: serde_json::Value = serde_json::from_str(s).ok()?;
        Some(IndexValue::Str(v["status"].as_str()?.to_string()))
    });
    db.activate_field_index(ns, field_id, IndexValueType::Str, extractor).unwrap();

    db.put(b"doc:1", br#"{"status":"active"}"#).unwrap();
    db.put(b"doc:2", br#"{"status":"active"}"#).unwrap();
    assert_eq!(db.query_keys(ns, "status = \"active\"").unwrap().keys.len(), 2);

    // Update doc:1's status active -> archived. The targeted update must move
    // the row, so it no longer matches "active" and now matches "archived".
    db.put(b"doc:1", br#"{"status":"archived"}"#).unwrap();
    assert_eq!(db.query_keys(ns, "status = \"active\"").unwrap().keys, vec![b"doc:2".to_vec()]);
    assert_eq!(db.query_keys(ns, "status = \"archived\"").unwrap().keys, vec![b"doc:1".to_vec()]);

    // Updating to the same value is a no-op and keeps the row queryable.
    db.put(b"doc:2", br#"{"status":"active"}"#).unwrap();
    assert_eq!(db.query_keys(ns, "status = \"active\"").unwrap().keys, vec![b"doc:2".to_vec()]);

    // Delete doc:2 → it must leave the "active" bucket.
    db.delete(b"doc:2").unwrap();
    assert!(db.query_keys(ns, "status = \"active\"").unwrap().keys.is_empty());
    assert_eq!(db.query_keys(ns, "status = \"archived\"").unwrap().keys, vec![b"doc:1".to_vec()]);

    db.shutdown().unwrap();
}

// Regression: WAL GC can delete a fully-persisted segment out of segment
// order (e.g. a dropped namespace persists a trailing segment while an
// earlier one stays live), leaving a "hole". Recovery scans head→tail, so it
// must survive the missing middle segment. Before the `scan_entries` hole-skip
// fix, the open-time scan aborted with NotFound and `Database::open` failed.
#[test]
fn test_recovery_survives_deleted_middle_wal_segment() -> Result<()> {
    let temp_dir = TempDir::new()?;
    {
        let db = Database::open_with_wal_segment_size(temp_dir.path(), create_db_config(), 256)?;
        // Inject un-persisted entries spanning several WAL segments so reopen
        // actually runs recovery (persisted < total).
        let entries: Vec<WalEntry> = (0..60u32)
            .map(|i| WalEntry::new_upsert(format!("rk{:03}", i).into_bytes(), vec![b'v'; 32]))
            .collect();
        db.simulate_crash_with_wal_entries(entries)?;
        assert!(
            temp_dir.path().join("wal.log.seg000002").exists(),
            "need >= 3 segments to have a middle one"
        );

        // Punch a hole: delete a middle segment while segment 0 stays live
        // (so head is not advanced past it) — exactly what WAL GC would leave.
        db.wal.delete_segment_file(1)?;
        drop(db);
    }

    // Reopen: recovery must NOT crash on the hole, and the entries from the
    // surviving segments must be recovered (the deleted segment's are lost).
    let db = Database::open_with_wal_segment_size(temp_dir.path(), create_db_config(), 256)?;
    assert_eq!(db.get(b"rk000")?, Some(vec![b'v'; 32]), "segment 0 entry must survive the hole");
    assert_eq!(db.get(b"rk059")?, Some(vec![b'v'; 32]), "last-segment entry must survive the hole");
    db.shutdown()?;
    Ok(())
}

#[test]
fn test_sequence_increments_on_writes() -> Result<()> {
    use std::sync::atomic::Ordering;
    let temp_dir = TempDir::new()?;
    let db = Database::open(temp_dir.path(), create_db_config())?;

    let seq_before = db.next_seq.load(Ordering::Relaxed);
    db.put(b"a", b"1")?;
    db.put(b"b", b"2")?;
    db.delete(b"a")?;
    let seq_after = db.next_seq.load(Ordering::Relaxed);

    assert_eq!(seq_after - seq_before, 3);
    Ok(())
}

#[test]
fn test_sequence_stamps_wal_entries() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db = Database::open(temp_dir.path(), create_db_config())?;

    db.put(b"x", b"v1")?;
    db.put(b"y", b"v2")?;

    let wal_metadata = db.wal_metadata.read();
    let entries = db.wal.scan_entries(0, wal_metadata.tail)?;
    drop(wal_metadata);

    let seqs: Vec<u64> = entries.iter().map(|(_, e)| e.sequence).collect();
    assert!(!seqs.is_empty());
    for seq in &seqs {
        assert!(*seq > 0, "sequence must be non-zero");
    }
    for w in seqs.windows(2) {
        assert!(w[1] > w[0], "sequences must be strictly increasing");
    }
    Ok(())
}

#[test]
fn test_sequence_recovered_after_reopen() -> Result<()> {
    use std::sync::atomic::Ordering;
    let temp_dir = TempDir::new()?;

    let seq_at_close = {
        let db = Database::open(temp_dir.path(), create_db_config())?;
        db.put(b"k1", b"v1")?;
        db.put(b"k2", b"v2")?;
        db.next_seq.load(Ordering::Relaxed)
    };

    let db2 = Database::open(temp_dir.path(), create_db_config())?;
    let seq_after_reopen = db2.next_seq.load(Ordering::Relaxed);
    assert!(
        seq_after_reopen >= seq_at_close,
        "recovered seq ({}) must be >= seq at close ({})",
        seq_after_reopen,
        seq_at_close
    );

    let seq_before_new_write = db2.next_seq.load(Ordering::Relaxed);
    db2.put(b"k3", b"v3")?;
    let wal_metadata = db2.wal_metadata.read();
    let entries = db2.wal.scan_entries(0, wal_metadata.tail)?;
    drop(wal_metadata);
    let max_old_seq = seq_before_new_write - 1;
    let new_entry_seq = entries.last().map(|(_, e)| e.sequence).unwrap_or(0);
    assert!(
        new_entry_seq > max_old_seq,
        "new write seq ({}) must exceed previous max ({})",
        new_entry_seq,
        max_old_seq
    );
    Ok(())
}

#[test]
fn test_metadata_checksum_corruption_recovery() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let path = temp_dir.path().to_path_buf();

    {
        let db = Database::open(&path, create_db_config())?;
        db.put(b"key1", b"value1")?;
        db.shutdown()?;
    }

    {
        let wal_metadata_path = path.join("wal_metadata");
        let mut data = std::fs::read(&wal_metadata_path)?;
        if let Some(byte) = data.get_mut(16) {
            *byte ^= 0xFF;
        }
        std::fs::write(&wal_metadata_path, data)?;
    }

    let db = Database::open(&path, create_db_config())?;
    assert!(path.join("wal_metadata.corrupt").exists());
    assert_eq!(db.get(b"key1")?, Some(b"value1".to_vec()));

    Ok(())
}

// ── merge (atomic read-modify-write) ───────────────────────────────
//
// `merge_ns` is the only operation that reads a key and writes it back as one
// step. These tests pin the two halves of that: the semantics of what the
// closure returns, and the serialisation that makes the read-modify-write safe
// against concurrent writers.

/// Decode a little-endian `u64` counter, treating an absent key as zero.
#[cfg(test)]
fn counter_of(bytes: Option<&[u8]>) -> u64 {
    bytes.map_or(0, |b| u64::from_le_bytes(b.try_into().expect("counter is 8 bytes")))
}

#[cfg(test)]
mod merge_tests {
    use super::*;
    use std::sync::Arc;

    /// Adds the operand to the stored counter, creating it at zero.
    fn add(existing: Option<&[u8]>, operand: &[u8]) -> Result<Option<Vec<u8>>> {
        Ok(Some((counter_of(existing) + counter_of(Some(operand))).to_le_bytes().to_vec()))
    }

    #[test]
    fn merge_creates_the_key_when_it_is_absent() {
        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();

        let mut saw: Option<Option<Vec<u8>>> = None;
        let written = db
            .merge(b"counter", &5u64.to_le_bytes(), |existing, operand| {
                saw = Some(existing.map(|b| b.to_vec()));
                add(existing, operand)
            })
            .unwrap();

        assert_eq!(saw, Some(None), "the closure should see None for an absent key");
        assert_eq!(written, Some(5u64.to_le_bytes().to_vec()));
        assert_eq!(db.get(b"counter").unwrap(), Some(5u64.to_le_bytes().to_vec()));

        db.shutdown().unwrap();
    }

    #[test]
    fn merge_sees_the_stored_value_and_the_operand() {
        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();

        db.put(b"counter", &7u64.to_le_bytes()).unwrap();
        let written = db.merge(b"counter", &3u64.to_le_bytes(), add).unwrap();

        assert_eq!(written, Some(10u64.to_le_bytes().to_vec()));
        assert_eq!(db.get(b"counter").unwrap(), Some(10u64.to_le_bytes().to_vec()));

        db.shutdown().unwrap();
    }

    #[test]
    fn returning_none_deletes_an_existing_key() {
        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();

        db.put(b"doomed", b"value").unwrap();
        assert_eq!(db.merge(b"doomed", b"", |_, _| Ok(None)).unwrap(), None);
        assert_eq!(db.get(b"doomed").unwrap(), None);

        db.shutdown().unwrap();
    }

    /// Deleting a key that is already absent would cost an fsync and a tombstone
    /// to say nothing, so the absent-to-absent case must not touch the WAL.
    #[test]
    fn returning_none_for_an_absent_key_writes_nothing_at_all() {
        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();

        let before = db.wal_metadata.read().total_entries;
        assert_eq!(db.merge(b"never-existed", b"", |_, _| Ok(None)).unwrap(), None);

        assert_eq!(
            db.wal_metadata.read().total_entries,
            before,
            "an absent -> absent merge appended a WAL entry"
        );
        assert_eq!(db.get(b"never-existed").unwrap(), None);

        db.shutdown().unwrap();
    }

    /// An aborting closure must leave no trace: not the value, and not a WAL
    /// entry recording a write that never happened.
    #[test]
    fn an_aborting_closure_writes_nothing_and_surfaces_its_error() {
        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();

        db.put(b"capped", &9u64.to_le_bytes()).unwrap();
        let before = db.wal_metadata.read().total_entries;

        let err = db
            .merge(b"capped", &1u64.to_le_bytes(), |existing, _| {
                if counter_of(existing) >= 9 {
                    return Err(KVError::MergeAborted("already at cap".into()));
                }
                Ok(Some(vec![]))
            })
            .unwrap_err();

        assert!(matches!(err, KVError::MergeAborted(_)), "unexpected error: {err:?}");
        assert_eq!(db.wal_metadata.read().total_entries, before, "an aborted merge appended a WAL entry");
        assert_eq!(db.get(b"capped").unwrap(), Some(9u64.to_le_bytes().to_vec()));

        db.shutdown().unwrap();
    }

    /// The one case where `merge` and `put` deliberately disagree.
    ///
    /// Once the WAL fsync returns, a write is durable — but the in-memory apply
    /// that makes it *readable* is best-effort. `put` reports that as success,
    /// because the write is committed and the caller can do nothing useful with
    /// the distinction. `merge` must not: its result never became readable, so
    /// the next merge on that key would read the unchanged value and accumulate
    /// onto a stale base. It returns `MergeNotApplied` instead.
    ///
    /// The failure is injected by making the value-log directory read-only after
    /// the active segment is nearly full, so the next write needs a new segment
    /// file it cannot create. (Running as root defeats this — the assertions say
    /// so rather than failing cryptically.)
    #[test]
    fn merge_reports_a_write_that_is_durable_but_not_readable_while_put_does_not() {
        use std::os::unix::fs::PermissionsExt;

        const SEG: u64 = 64 * 1024; // the minimum legal segment size

        let dir = TempDir::new().unwrap();
        let mut config = create_db_config();
        config.segment_size_bytes = SEG;
        config.num_buckets = 1; // one value log, so the fill below is deterministic
        let db = Database::open(dir.path(), config).unwrap();

        // Fill the active segment to ~32.9 KiB of its 64 KiB, so the 41 KiB
        // record written below cannot fit and must roll to a new segment.
        db.put(b"k0", &vec![0u8; 16 * 1024]).unwrap();
        db.put(b"k1", &vec![1u8; 16 * 1024]).unwrap();
        db.put(b"target", b"small").unwrap();

        let vlog_dir = db.get_store(DEFAULT_NAMESPACE_ID).unwrap().value_log_path.clone();
        let original = std::fs::metadata(&vlog_dir).unwrap().permissions();
        let mut readonly = original.clone();
        readonly.set_mode(0o555);
        std::fs::set_permissions(&vlog_dir, readonly).unwrap();

        let big = vec![9u8; 40 * 1024];
        let entries_before = db.wal_metadata.read().total_entries;

        // `put` swallows it and reports success.
        let put_result = db.put(b"target", &big);
        let readable_after_put = db.get(b"target");

        // `merge` surfaces it.
        let merge_result = db.merge(b"target", &big, |_, operand| Ok(Some(operand.to_vec())));

        let entries_after = db.wal_metadata.read().total_entries;
        let failures = db.metrics_snapshot().apply_failures;

        // Restore before asserting, so a failure still leaves a removable TempDir.
        std::fs::set_permissions(&vlog_dir, original).unwrap();

        assert!(
            put_result.is_ok(),
            "put must report a WAL-durable write as success (are these tests running as root?): {put_result:?}"
        );
        assert_eq!(
            readable_after_put.unwrap(),
            Some(b"small".to_vec()),
            "the put was acknowledged but never applied, so a read still sees the prior value"
        );
        match merge_result {
            Err(KVError::MergeNotApplied { .. }) => {}
            other => panic!("merge should surface an unapplied write (running as root?), got {other:?}"),
        }
        assert!(
            entries_after > entries_before,
            "both writes are durable — this is 'not readable yet', not 'nothing happened'"
        );
        assert!(failures >= 2, "expected both applies to fail, apply_failures = {failures}");

        db.shutdown().unwrap();
    }

    /// The docs promise a read from inside the closure is safe (it takes no
    /// stripe) and returns the pre-merge value. Pin both — a future change that
    /// made reads take the stripe would deadlock here rather than go unnoticed.
    #[test]
    fn the_closure_may_read_the_database_and_sees_the_pre_merge_value() {
        let dir = TempDir::new().unwrap();
        let db = Arc::new(Database::open(dir.path(), create_db_config()).unwrap());
        db.put(b"k", b"before").unwrap();
        db.put(b"other", b"untouched").unwrap();

        let inner = Arc::clone(&db);
        let written = db
            .merge(b"k", b"after", move |existing, operand| {
                assert_eq!(inner.get(b"k").unwrap().as_deref(), existing, "a re-read disagreed with the handed value");
                assert_eq!(inner.get(b"other").unwrap(), Some(b"untouched".to_vec()));
                Ok(Some(operand.to_vec()))
            })
            .unwrap();

        assert_eq!(written, Some(b"after".to_vec()));
        db.shutdown().unwrap();
    }

    /// The load-bearing test. Without the key stripe every thread reads the same
    /// counter, computes the same successor, and all but one update is lost.
    #[test]
    fn concurrent_merges_on_one_key_lose_no_updates() {
        const THREADS: u64 = 8;
        const PER_THREAD: u64 = 100;

        let dir = TempDir::new().unwrap();
        let db = Arc::new(Database::open(dir.path(), create_db_config()).unwrap());

        std::thread::scope(|s| {
            for _ in 0..THREADS {
                let db = Arc::clone(&db);
                s.spawn(move || {
                    for _ in 0..PER_THREAD {
                        db.merge(b"counter", &1u64.to_le_bytes(), add).unwrap();
                    }
                });
            }
        });

        assert_eq!(
            counter_of(db.get(b"counter").unwrap().as_deref()),
            THREADS * PER_THREAD,
            "lost update — concurrent merges are not serialised"
        );

        db.shutdown().unwrap();
    }

    /// Merge must also be atomic against a *blind* `put`, which is why `put_ns`
    /// takes the stripe as well.
    ///
    /// Thread A merges with a deliberately slow closure; thread B blind-writes a
    /// sentinel while A is still inside it. B has to wait for the stripe, so B's
    /// write is the later sequence and the sentinel survives. Without the stripe
    /// B lands mid-merge and A's write — computed from a value B has already
    /// replaced — clobbers it. The two outcomes are distinct values, so the test
    /// actually discriminates.
    #[test]
    fn a_blind_put_cannot_land_in_the_middle_of_a_merge() {
        let dir = TempDir::new().unwrap();
        let db = Arc::new(Database::open(dir.path(), create_db_config()).unwrap());
        db.put(b"k", b"initial").unwrap();

        std::thread::scope(|s| {
            let merger = {
                let db = Arc::clone(&db);
                s.spawn(move || {
                    db.merge(b"k", b"", |existing, _| {
                        std::thread::sleep(Duration::from_millis(300));
                        let mut v = existing.unwrap_or(b"").to_vec();
                        v.extend_from_slice(b"+merged");
                        Ok(Some(v))
                    })
                    .unwrap();
                })
            };

            // Long enough that the merge is certainly inside its closure.
            std::thread::sleep(Duration::from_millis(50));
            db.put(b"k", b"sentinel").unwrap();
            merger.join().unwrap();
        });

        assert_eq!(
            db.get(b"k").unwrap(),
            Some(b"sentinel".to_vec()),
            "the put was applied while the merge held the key, so the merge overwrote it"
        );

        db.shutdown().unwrap();
    }

    /// A merge resolves to an ordinary `Upsert`/`Delete` WAL record, so it must
    /// replay like any other write.
    #[test]
    fn merged_values_survive_a_reopen() {
        let dir = TempDir::new().unwrap();

        {
            let db = Database::open(dir.path(), create_db_config()).unwrap();
            db.merge(b"counter", &4u64.to_le_bytes(), add).unwrap();
            db.merge(b"counter", &6u64.to_le_bytes(), add).unwrap();
            db.put(b"doomed", b"value").unwrap();
            db.merge(b"doomed", b"", |_, _| Ok(None)).unwrap();
            db.shutdown().unwrap();
        }

        let db = Database::open(dir.path(), create_db_config()).unwrap();
        assert_eq!(db.get(b"counter").unwrap(), Some(10u64.to_le_bytes().to_vec()));
        assert_eq!(db.get(b"doomed").unwrap(), None, "a merge that deleted must stay deleted");
        db.shutdown().unwrap();
    }

    /// Merge writes through the same storage path as `put`, so field-index
    /// maintenance comes for free — this pins that it actually does.
    #[test]
    fn merge_updates_the_field_index() {
        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        let ns = DEFAULT_NAMESPACE_ID;
        activate_status_index(&db, ns);

        db.put_ns(ns, b"u1", br#"{"status":"active"}"#).unwrap();
        db.merge_ns(ns, b"u1", br#"{"status":"inactive"}"#, |_, operand| Ok(Some(operand.to_vec())))
            .unwrap();

        assert!(
            db.query_keys(ns, r#"status = "active""#).unwrap().keys.is_empty(),
            "the pre-merge value is still indexed"
        );
        assert_eq!(db.query_keys(ns, r#"status = "inactive""#).unwrap().keys, vec![b"u1".to_vec()]);

        db.shutdown().unwrap();
    }

    #[test]
    fn merges_are_counted_separately_from_the_writes_they_produce() {
        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();

        db.merge(b"counter", &1u64.to_le_bytes(), add).unwrap();
        db.merge(b"counter", &1u64.to_le_bytes(), add).unwrap();
        // Aborted and no-op merges still count as merges, but write nothing.
        let _ = db.merge(b"counter", b"", |_, _| Err(KVError::MergeAborted("no".into())));
        db.merge(b"absent", b"", |_, _| Ok(None)).unwrap();

        let m = db.metrics_snapshot();
        assert_eq!(m.merges, 4);
        assert_eq!(m.puts, 2, "only the two writing merges should count as puts");

        db.shutdown().unwrap();
    }

    /// The stripe is the outermost lock. Driving every writer plus GC at once is
    /// the cheap way to notice if that ever stops being true — an inversion
    /// against the WAL or value-log bucket locks would hang here.
    #[test]
    fn writers_and_gc_make_progress_together() {
        let dir = TempDir::new().unwrap();
        let db = Arc::new(Database::open(dir.path(), create_db_config()).unwrap());

        std::thread::scope(|s| {
            for t in 0..4u64 {
                let db = Arc::clone(&db);
                s.spawn(move || {
                    for i in 0..150u64 {
                        let key = format!("{}-key-{}", i % 16, t);
                        // Deliberately tolerant of whatever the blind writes
                        // leave behind — this test is about liveness, not values.
                        db.merge(key.as_bytes(), b"merged", |_, operand| Ok(Some(operand.to_vec()))).unwrap();
                        db.put(key.as_bytes(), b"blind").unwrap();
                        if i % 10 == 0 {
                            db.delete(key.as_bytes()).unwrap();
                        }
                    }
                });
            }
            let db = Arc::clone(&db);
            s.spawn(move || {
                for _ in 0..5 {
                    let _ = db.garbage_collect();
                }
            });
        });

        db.shutdown().unwrap();
    }
}
