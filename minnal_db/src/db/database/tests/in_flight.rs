//! The in-flight write barrier: what a flush, a seal or the WAL cut may claim while a write is still applying.

use super::*;
use crate::namespace::NamespaceId;

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
/// `compute_cut` cut at the **live tail** whenever no namespace
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

    let watermark = db.wal_metadata.read().cut;
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
/// `forget_namespace` calls `compute_cut` the instant it removes the
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

    let watermark = db.wal_metadata.read().cut;
    assert!(
        watermark <= in_flight_offset,
        "dropping a store advanced the watermark to {watermark}, past the survivor's \
         in-flight write at {in_flight_offset}"
    );
    drop(guard);
}

/// `compute_cut`'s cut must respect the ceiling on its own.
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
/// `compute_cut` through `forget_namespace` with no seal after it — so
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

    let watermark = db.wal_metadata.read().cut;
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
    let watermark = db.wal_metadata.read().cut;
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
    // Creating the namespace wrote and flushed the system catalog, which moved
    // the cut past its records; nothing after this may move it.
    let cut_before = db.wal_metadata.read().cut;

    for i in 0..200u32 {
        db.put_ns(ns, format!("k{i:04}").as_bytes(), b"value").unwrap();
    }
    // Deletes take the same Step-3 sync path and had the same defect.
    db.delete_ns(ns, b"k0000").unwrap();

    let meta = db.wal_metadata.read().clone();
    assert!(meta.tail > 0, "expected every write to be journalled");
    assert_eq!(
        meta.cut, cut_before,
        "a value-log fsync moved the cut to {} (tail {}), but nothing has reached an SSTable — \
             recovery would skip those entries and lose acknowledged writes",
        meta.cut, meta.tail
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
    db.wal_flush_observer.prepare_written(a, 0).unwrap().commit();

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
/// Asserting on the saved progress rather than on post-crash reads keeps this
/// a unit test: an entry below its namespace's `replay_from` is exactly the
/// entry recovery will refuse to replay.
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
    // its memtable, so the cut must stay at or below the first of them.
    let quiet_start = db.wal_metadata.read().tail;
    for i in 0..10u32 {
        db.put_ns(quiet, format!("q{i}").as_bytes(), b"quiet-value").unwrap();
    }

    // `busy` writes, then flushes to level 0 — the event that used to mark
    // everything below the global WAL tail persisted. Flushed explicitly
    // rather than by capacity so the test does not depend on the background
    // workers running.
    for i in 0..500u32 {
        db.put_ns(busy, format!("b{i}").as_bytes(), b"busy-value").unwrap();
    }
    db.get_store(busy).unwrap().flush_memtable_to_level0().unwrap();

    let meta = db.wal_metadata.read().clone();
    assert!(
        meta.replay_from(quiet) <= quiet_start,
        "busy's flushes retired quiet's un-flushed entries: quiet replays from {} \
             but its first entry is at {quiet_start} (cut {})",
        meta.replay_from(quiet),
        meta.cut
    );
    assert!(meta.replay_from(busy) > quiet_start, "busy's own flush should have retired its entries");
}

/// Concurrent flushes in many namespaces each save the WAL progress. The
/// saves are serialised (`WalPersistObserver::persist_lock`), so the file on
/// disk always ends equal to the live progress: an older save landing after a
/// newer one would leave recovery an older cut than WAL GC has acted on.
#[test]
fn concurrent_flushes_leave_the_saved_progress_equal_to_the_live_one() {
    let dir = TempDir::new().unwrap();
    let mut config = create_db_config();
    config.lsm_config.skip_list_capacity = 32;
    let db = Database::open(dir.path(), config).unwrap();
    let namespaces: Vec<NamespaceId> = (0..8).map(|i| db.create_namespace(&format!("ns{i}")).unwrap()).collect();

    std::thread::scope(|s| {
        for &ns in &namespaces {
            let db = &db;
            s.spawn(move || {
                for i in 0..300u32 {
                    db.put_ns(ns, format!("k{i:04}").as_bytes(), b"value").unwrap();
                    if i % 50 == 49 {
                        db.get_store(ns).unwrap().flush_memtable_to_level0().unwrap();
                    }
                }
            });
        }
    });

    let live = db.wal_metadata.read().clone();
    let saved = crate::db::wal::WalMetadata::from_file_bytes(&std::fs::read(wal_metadata_path(dir.path())).unwrap()).unwrap();
    assert!(live.cut > 0, "the flushes should have moved the cut");
    assert_eq!((saved.cut, &saved.safe), (live.cut, &live.safe), "the saved progress is not the live one");
    db.shutdown().unwrap();
}

/// Retiring entries never writes into the log: a flush moves the cut by
/// saving the WAL metadata, and the log's bytes stay exactly as appended.
///
/// The log is append-only on purpose. Rewriting a record in place (as marking
/// each entry persisted once did) costs a random write and an fsync per touched
/// segment inside the flush, and means a frame can change after it is written.
#[test]
fn a_flush_retires_entries_without_writing_into_the_log() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    let ns = db.create_namespace("bulk").unwrap();

    for i in 0..600u32 {
        db.put_ns(ns, format!("k{i:05}").as_bytes(), b"v").unwrap();
    }
    let before = std::fs::read(wal_segment_path(dir.path(), 0)).unwrap();
    db.get_store(ns).unwrap().flush_memtable_to_level0().unwrap();
    db.garbage_collect_wal().unwrap();
    let after = std::fs::read(wal_segment_path(dir.path(), 0)).unwrap();

    let meta = db.wal_metadata.read().clone();
    assert_eq!(meta.cut, meta.tail, "the flush should have retired every entry");
    assert!(before == after, "retiring entries changed the log's bytes");
}

/// The index checkpoint must never claim an offset past a write whose index
/// update has not happened. A write is in flight from its WAL append until its
/// apply (index update included) returns; the checkpoint used the raw WAL tail,
/// which includes such a write, so after a crash replay started past it and the
/// document stayed out of every field index with no gap recorded. Found by the
/// SIGKILL stress harness (1–2 documents per crash).
#[test]
fn index_checkpoint_stops_below_an_in_flight_write() {
    use crate::db::indexing::manager::CheckpointState;
    use crate::db::workers::index_checkpoint::IndexCheckpointTarget;
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    let ns = crate::namespace::DEFAULT_NAMESPACE_ID;
    let field_id = activate_status_index(&db, ns);
    db.put(b"doc:1", br#"{"status":"active"}"#).unwrap();

    // A write appended at the current tail but not yet applied...
    let in_flight_at = db.wal_metadata.read().tail;
    let in_flight = db.wal_flush_observer.begin_write(in_flight_at);
    // ...while later writes move the tail on.
    db.put(b"doc:2", br#"{"status":"active"}"#).unwrap();
    let tail = db.wal_metadata.read().tail;
    assert!(tail > in_flight_at, "setup: the tail must have moved past the in-flight write");

    db.run_index_checkpoint().unwrap();
    assert_eq!(
        db.index_manager.read_checkpoint_state(ns, field_id, tail),
        CheckpointState::At(in_flight_at),
        "the checkpoint claimed index coverage past a write the index has not seen"
    );

    drop(in_flight);
    db.run_index_checkpoint().unwrap();
    assert_eq!(db.index_manager.read_checkpoint_state(ns, field_id, tail), CheckpointState::At(tail));
    db.shutdown().unwrap();
}
