//! The log wired into `Database`: logging, flushing, replay after a crash.

use super::*;
use crate::namespace::NamespaceId;

use crate::db::database::Database;
use crate::db::test_support::create_db_config;

/// Logs and applies one write to `ns`, as step 4's worker will.
fn log_and_apply(db: &Database, ns: NamespaceId, key: &[u8], value: &[u8]) {
    let seq = db.next_seq.fetch_add(1, Ordering::SeqCst);
    let logged = db.vector_index_wal.append(ns, key, value, seq).unwrap();
    // Applied with the logged seq, and synced before the caller goes on,
    // as step 4's worker will (it completes an entry only after a sync).
    db.get_store(ns).unwrap().replay_upsert(key, value, seq).unwrap();
    drop(logged);
    db.vector_index_wal.sync().unwrap();
}

#[test]
fn a_namespace_flush_retires_its_logged_writes() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    let ns = db.create_namespace("v_sparse_vector").unwrap();
    for i in 0..5u64 {
        log_and_apply(&db, ns, &key(i), &value(i));
    }
    assert_eq!(db.vector_index_wal.retired_cut(), 0);
    db.get_store(ns).unwrap().flush_memtable_to_level0().unwrap();
    assert_eq!(db.vector_index_wal.retired_cut(), db.vector_index_wal.tail());
    let stats = db.vector_index_wal_stats().unwrap();
    assert_eq!(stats.unretired_bytes, 0);
    assert!(stats.holding_cut.is_empty());
}

#[test]
fn the_gc_pass_flushes_an_idle_namespace_holding_the_cut() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    let busy = db.create_namespace("busy").unwrap();
    let idle = db.create_namespace("idle").unwrap();
    log_and_apply(&db, idle, b"k", b"v");
    log_and_apply(&db, busy, b"k", b"v");
    db.get_store(busy).unwrap().flush_memtable_to_level0().unwrap();
    let stats = db.vector_index_wal_stats().unwrap();
    assert_eq!(stats.holding_cut.iter().map(|h| h.namespace.as_str()).collect::<Vec<_>>(), vec!["idle"]);
    assert_eq!(db.vector_index_wal.retired_cut(), 0);

    assert_eq!(db.flush_namespaces_pinning_vector_wal(), 1);
    assert_eq!(db.vector_index_wal.retired_cut(), db.vector_index_wal.tail());
    assert_eq!(db.flush_namespaces_pinning_vector_wal(), 0, "nothing holds the cut now");
    assert_eq!(db.metrics_snapshot().vector_wal_pin_flushes, 1);
}

#[test]
fn dropping_a_namespace_releases_the_cut() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    let kept = db.create_namespace("kept").unwrap();
    let dropped = db.create_namespace("dropped").unwrap();
    log_and_apply(&db, dropped, b"k", b"v");
    log_and_apply(&db, kept, b"k", b"v");
    db.get_store(kept).unwrap().flush_memtable_to_level0().unwrap();
    assert_eq!(db.vector_index_wal.retired_cut(), 0);
    db.remove_namespace("dropped").unwrap();
    assert_eq!(db.vector_index_wal.retired_cut(), db.vector_index_wal.tail());
}

#[test]
fn a_database_reopens_with_its_retired_cut() {
    let dir = TempDir::new().unwrap();
    let cut = {
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        let ns = db.create_namespace("v").unwrap();
        log_and_apply(&db, ns, b"k", b"v");
        db.get_store(ns).unwrap().flush_memtable_to_level0().unwrap();
        let cut = db.vector_index_wal.retired_cut();
        db.shutdown().unwrap();
        cut
    };
    assert!(cut > 0);
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    assert_eq!(db.vector_index_wal.retired_cut(), cut);
}

#[test]
fn a_restart_keeps_records_nobody_flushed() {
    let dir = TempDir::new().unwrap();
    let (v, first) = {
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        let v = db.create_namespace("v").unwrap();
        db.create_namespace("u").unwrap();
        let first = db.vector_index_wal.tail();
        log_and_apply(&db, v, b"k1", b"x");
        log_and_apply(&db, v, b"k2", b"y");
        // A namespace that no longer exists after the restart.
        db.vector_index_wal.append(NamespaceId::from_u128(999), b"gone", b"z", 1).unwrap();
        db.vector_index_wal.sync().unwrap();
        db.mark_crashed();
        (v, first)
    };

    let db = Database::open(dir.path(), create_db_config()).unwrap();
    // 999 was released at open; v's records were replayed, flushed and
    // retired by recovery.
    assert!(db.vector_index_wal.namespaces_holding_cut().is_empty());
    assert!(db.vector_index_wal.retired_cut() > first);
    assert_eq!(db.metrics_snapshot().vector_wal_records_skipped, 1, "999's record is not replayed");
    assert_eq!(db.metrics_snapshot().vector_wal_records_replayed, 2);
    // Another namespace's flush moves the cut to the tail; v's records must
    // be on disk by then, which a second crash shows.
    let u = db.registry.read().get_id("u").unwrap();
    log_and_apply(&db, u, b"k", b"v");
    db.get_store(u).unwrap().flush_memtable_to_level0().unwrap();
    assert_eq!(db.vector_index_wal.retired_cut(), db.vector_index_wal.tail());
    db.mark_crashed();
    drop(db);
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    assert_eq!(db.get_ns(v, b"k1").unwrap(), Some(b"x".to_vec()));
    assert_eq!(db.get_ns(v, b"k2").unwrap(), Some(b"y".to_vec()));
}

/// The timeline from the step 3 review: a chunk key whose delete reached
/// L1 (tombstone dropped) while another vector namespace held the global
/// cut below the key's record. Replaying from the global cut brings the
/// key back, with nothing newer left to beat it.
#[test]
fn replay_does_not_bring_back_a_key_whose_delete_reached_l1() {
    let dir = TempDir::new().unwrap();
    let (sparse, k) = {
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        let sparse = db.create_namespace("docs_sparse_vector").unwrap();
        let dense = db.create_namespace("docs_dense_vector").unwrap();
        let k: &[u8] = b"posting-7|doc-42";

        // 1-2. A chunk key in the sparse namespace, a dense code in the other.
        log_and_apply(&db, sparse, k, b"chunk code");
        let r_start = db.vector_index_wal.tail();
        log_and_apply(&db, dense, b"doc-42", b"dense code");
        // 3. Sparse flushes; dense still holds the global cut below R.
        db.get_store(sparse).unwrap().flush_memtable_to_level0().unwrap();
        assert_eq!(db.vector_index_wal.namespaces_holding_cut(), vec![dense]);
        assert!(db.vector_index_wal.retired_cut() < r_start, "the cut stays below R");
        // 4-6. The chunk moves away: K is deleted (main WAL), flushed and
        // merged into L1, which drops the tombstone.
        db.delete_ns(sparse, k).unwrap();
        let store = db.get_store(sparse).unwrap();
        store.flush_memtable_to_level0().unwrap();
        store.compact_lsm().unwrap();
        assert_eq!(db.get_ns(sparse, k).unwrap(), None);
        // The namespace keeps working: its next compaction pass also removes
        // the L0 files the merge made obsolete, tombstone included.
        log_and_apply(&db, sparse, b"posting-9|doc-42", b"chunk code");
        store.flush_memtable_to_level0().unwrap();
        store.compact_lsm().unwrap();
        let m = db.wal_metadata();
        assert_eq!(m.cut, m.tail, "main-WAL recovery has nothing to replay");
        // 7. Crash.
        db.mark_crashed();
        (sparse, k.to_vec())
    };

    // 8. Restart: open replays the vector index WAL. R lies below the sparse
    // namespace's own durable safe offset, so it is not replayed; the dense
    // namespace's record is.
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    assert_eq!(
        db.get_ns(sparse, &k).unwrap(),
        None,
        "a deleted chunk key must stay deleted after recovery"
    );
    let dense = db.registry.read().get_id("docs_dense_vector").unwrap();
    assert_eq!(db.get_ns(dense, b"doc-42").unwrap(), Some(b"dense code".to_vec()));
    assert_eq!(db.metrics_snapshot().vector_wal_records_replayed, 1);
}

fn reopen_db(dir: &TempDir) -> Database {
    Database::open(dir.path(), create_db_config()).unwrap()
}

#[test]
fn records_are_replayed_after_a_crash_and_retired_by_recovery() {
    let dir = TempDir::new().unwrap();
    let ns = {
        let db = reopen_db(&dir);
        let ns = db.create_namespace("v_sparse_vector").unwrap();
        for i in 0..20u64 {
            log_and_apply(&db, ns, &key(i), &value(i));
        }
        db.mark_crashed(); // nothing flushed
        ns
    };
    let db = reopen_db(&dir);
    for i in 0..20u64 {
        assert_eq!(db.get_ns(ns, &key(i)).unwrap(), Some(value(i)), "key {i} replayed");
    }
    assert_eq!(db.metrics_snapshot().vector_wal_records_replayed, 20);
    // Recovery flushed what it replayed and retired it: nothing holds the cut.
    assert!(db.vector_index_wal.namespaces_holding_cut().is_empty());
    assert_eq!(db.vector_index_wal.retired_cut(), db.vector_index_wal.tail());
    db.mark_crashed();
    drop(db);

    let db = reopen_db(&dir);
    assert_eq!(db.metrics_snapshot().vector_wal_records_replayed, 0, "everything was retired");
    for i in 0..20u64 {
        assert_eq!(db.get_ns(ns, &key(i)).unwrap(), Some(value(i)), "key {i} survives a second crash");
    }
}

#[test]
fn a_write_after_restart_beats_a_replayed_record() {
    let dir = TempDir::new().unwrap();
    let (ns, seq) = {
        let db = reopen_db(&dir);
        let ns = db.create_namespace("v").unwrap();
        db.put_ns(ns, b"main-wal", b"x").unwrap(); // the main WAL's last seq is below the record's
        log_and_apply(&db, ns, b"k", b"old");
        let seq = db.vector_index_wal.last_seq();
        db.mark_crashed();
        (ns, seq)
    };
    let db = reopen_db(&dir);
    assert!(db.next_seq.load(Ordering::SeqCst) > seq, "the counter is raised above the log");
    log_and_apply(&db, ns, b"k", b"new");
    assert_eq!(db.get_ns(ns, b"k").unwrap(), Some(b"new".to_vec()));
}

#[test]
fn a_crash_loses_appends_that_were_never_synced() {
    let dir = TempDir::new().unwrap();
    let ns = {
        let db = reopen_db(&dir);
        let ns = db.create_namespace("v").unwrap();
        log_and_apply(&db, ns, b"synced", b"1"); // syncs
        let seq = db.next_seq.fetch_add(1, Ordering::SeqCst);
        let logged = db.vector_index_wal.append(ns, b"unsynced", b"2", seq).unwrap();
        db.get_store(ns).unwrap().replay_upsert(b"unsynced", b"2", seq).unwrap();
        drop(logged);
        db.mark_crashed();
        ns
    };
    let db = reopen_db(&dir);
    assert_eq!(db.get_ns(ns, b"synced").unwrap(), Some(b"1".to_vec()));
    assert_eq!(db.get_ns(ns, b"unsynced").unwrap(), None, "a power loss takes what was never synced");
}

#[test]
fn a_logged_delete_replays_with_its_seq() {
    let dir = TempDir::new().unwrap();
    let ns = {
        let db = reopen_db(&dir);
        let ns = db.create_namespace("v").unwrap();
        db.put_ns(ns, b"gone", b"1").unwrap();
        db.put_ns(ns, b"back", b"1").unwrap();
        db.delete_ns_logged(ns, b"gone").unwrap();
        db.delete_ns_logged(ns, b"back").unwrap();
        db.put_ns(ns, b"back", b"2").unwrap(); // newer than its logged delete
        db.delete_ns_logged(ns, b"unsynced").unwrap();
        db.sync_vector_index_wal().unwrap();
        assert_eq!(db.get_ns(ns, b"gone").unwrap(), None);
        db.mark_crashed();
        ns
    };
    let db = reopen_db(&dir);
    assert_eq!(db.metrics_snapshot().vector_wal_records_replayed, 3);
    assert_eq!(db.get_ns(ns, b"gone").unwrap(), None, "the delete replays");
    assert_eq!(db.get_ns(ns, b"back").unwrap(), Some(b"2".to_vec()), "a newer put beats it");
}

/// The prefix rule: a flush syncs the log before it writes L0 files. Here
/// a meta write logged to one namespace is followed by a key delete logged
/// to another, and only the second namespace is flushed before the crash.
/// Without the sync its L0 file would keep the delete while the log lost
/// the meta write before it.
#[test]
fn a_flush_syncs_the_log_so_a_crash_keeps_a_prefix() {
    let dir = TempDir::new().unwrap();
    let (meta, sparse) = {
        let db = reopen_db(&dir);
        let meta = db.create_namespace("v_sparse_vector_meta").unwrap();
        let sparse = db.create_namespace("v_sparse_vector").unwrap();
        db.put_ns(sparse, b"7|doc", b"codes").unwrap();
        db.put_ns(meta, b"doc", b"lists 7").unwrap();
        db.put_ns_logged(meta, b"doc", b"lists nothing").unwrap();
        db.delete_ns_logged(sparse, b"7|doc").unwrap();
        let before = db.metrics_snapshot();
        db.get_store(sparse).unwrap().flush_memtable_to_level0().unwrap();
        let after = db.metrics_snapshot();
        assert_eq!(after.vector_wal_flush_syncs - before.vector_wal_flush_syncs, 1);
        assert_eq!(after.vector_wal_fsyncs - before.vector_wal_fsyncs, 1);
        // Nothing unsynced now: the next flush does not sync.
        db.put_ns(sparse, b"8|doc", b"codes").unwrap();
        db.get_store(sparse).unwrap().flush_memtable_to_level0().unwrap();
        assert_eq!(db.metrics_snapshot().vector_wal_flush_syncs, after.vector_wal_flush_syncs);
        db.mark_crashed();
        (meta, sparse)
    };
    let db = reopen_db(&dir);
    assert_eq!(db.get_ns(sparse, b"7|doc").unwrap(), None);
    assert_eq!(
        db.get_ns(meta, b"doc").unwrap(),
        Some(b"lists nothing".to_vec()),
        "the meta write logged before the delete survives with it"
    );
}

#[test]
fn a_flush_of_a_namespace_that_never_logged_does_not_sync() {
    let dir = TempDir::new().unwrap();
    let db = reopen_db(&dir);
    let logged = db.create_namespace("v").unwrap();
    let plain = db.create_namespace("plain").unwrap();
    db.put_ns_logged(logged, b"k", b"v").unwrap(); // unsynced
    db.put_ns(plain, b"k", b"v").unwrap();
    db.get_store(plain).unwrap().flush_memtable_to_level0().unwrap();
    assert_eq!(db.metrics_snapshot().vector_wal_flush_syncs, 0);
    assert!(db.vector_index_wal.durable_tail() < db.vector_index_wal.tail());
}

#[test]
fn a_crash_right_after_recovery_loses_nothing() {
    let dir = TempDir::new().unwrap();
    let ns = {
        let db = reopen_db(&dir);
        let ns = db.create_namespace("v").unwrap();
        for i in 0..5u64 {
            log_and_apply(&db, ns, &key(i), &value(i));
        }
        db.mark_crashed();
        ns
    };
    // Replays (recovery makes the records durable in L0), then crashes
    // before anything else runs.
    let db = reopen_db(&dir);
    assert_eq!(db.metrics_snapshot().vector_wal_records_replayed, 5);
    db.mark_crashed();
    drop(db);
    let db = reopen_db(&dir);
    assert_eq!(db.metrics_snapshot().vector_wal_records_replayed, 0, "retired by the first recovery");
    for i in 0..5u64 {
        assert_eq!(db.get_ns(ns, &key(i)).unwrap(), Some(value(i)));
    }
}

/// Two crashes in a row. Recovery replays the chunk key and, from the main
/// WAL, its delete; main-WAL recovery then compacts into L1 (tombstone
/// dropped) and marks the delete persisted, and a later compaction pass
/// removes the obsolete L0 file. Had recovery not recorded what it
/// replayed, the second crash would replay the key with nothing left to
/// delete it.
#[test]
fn a_second_crash_does_not_bring_back_a_deleted_key() {
    let dir = TempDir::new().unwrap();
    let k: &[u8] = b"posting-7|doc-42";
    let sparse = {
        let db = reopen_db(&dir);
        let sparse = db.create_namespace("docs_sparse_vector").unwrap();
        log_and_apply(&db, sparse, k, b"chunk code"); // vector index WAL, synced
        db.delete_ns(sparse, k).unwrap(); // main WAL, later seq
        db.mark_crashed(); // neither in an SSTable
        sparse
    };
    {
        let db = reopen_db(&dir);
        assert_eq!(db.get_ns(sparse, k).unwrap(), None);
        // The compaction worker's pass: nothing to seal (the memtable is
        // empty), but it removes the L0 files recovery's merge made obsolete.
        db.get_store(sparse).unwrap().compact_lsm().unwrap();
        db.mark_crashed();
    }
    let db = reopen_db(&dir);
    assert_eq!(db.get_ns(sparse, k).unwrap(), None, "a deleted chunk key must stay deleted");
}

/// A namespace can hold the cut with an empty memtable (here: a logged
/// record whose apply failed). The GC tick's flush must still release it,
/// which takes a seal of the empty memtable.
#[test]
fn a_pin_flush_releases_a_holder_whose_memtable_is_empty() {
    let dir = TempDir::new().unwrap();
    let db = reopen_db(&dir);
    let ns = db.create_namespace("v").unwrap();
    let seq = db.next_seq.fetch_add(1, Ordering::SeqCst);
    drop(db.vector_index_wal.append(ns, b"k", b"v", seq).unwrap()); // never applied
    db.vector_index_wal.sync().unwrap();
    assert_eq!(db.vector_index_wal.namespaces_holding_cut(), vec![ns]);
    assert_eq!(db.flush_namespaces_pinning_vector_wal(), 1);
    assert!(db.vector_index_wal.namespaces_holding_cut().is_empty());
    assert_eq!(db.vector_index_wal.retired_cut(), db.vector_index_wal.tail());
}

/// A flush whose marker cannot be written registers no L0 file, keeps the
/// data readable, and is completed by the next flush; the namespace's
/// progress is not wedged.
#[test]
fn a_flush_whose_marker_fails_registers_nothing_and_is_retried() {
    let dir = TempDir::new().unwrap();
    let db = reopen_db(&dir);
    let ns = db.create_namespace("v").unwrap();
    log_and_apply(&db, ns, b"k1", b"x");
    let store = db.get_store(ns).unwrap();
    let l0_files = || -> usize {
        let m = store.lsm.build_manifest_snapshot().unwrap();
        m.levels
            .iter()
            .filter(|l| l.level == 0)
            .flat_map(|l| &l.buckets)
            .map(|b| b.files.len())
            .sum()
    };
    db.vector_index_wal.fail_next_marker_write();
    assert!(store.flush_memtable_to_level0().is_err());
    assert_eq!(l0_files(), 0, "nothing registered");
    assert_eq!(db.get_ns(ns, b"k1").unwrap(), Some(b"x".to_vec()));
    assert_eq!(db.vector_index_wal.namespaces_holding_cut(), vec![ns]);

    store.flush_memtable_to_level0().unwrap();
    assert!(l0_files() > 0);
    assert!(db.vector_index_wal.namespaces_holding_cut().is_empty());
    // Later flushes keep retiring.
    log_and_apply(&db, ns, b"k2", b"y");
    store.flush_memtable_to_level0().unwrap();
    assert!(db.vector_index_wal.namespaces_holding_cut().is_empty());
    assert_eq!(db.vector_index_wal.retired_cut(), db.vector_index_wal.tail());
}

/// A sealed memtable is not on disk until its L0 files are written; its
/// records must still replay. Retiring at the seal would skip them.
#[test]
fn records_of_a_sealed_but_unwritten_memtable_replay_after_a_crash() {
    let dir = TempDir::new().unwrap();
    let config = || {
        let mut c = create_db_config();
        c.skip_list_capacity = 8; // seals every few writes; no worker writes them to L0
        c
    };
    let ns = {
        let db = Database::open(dir.path(), config()).unwrap();
        let ns = db.create_namespace("v").unwrap();
        for i in 0..30u64 {
            log_and_apply(&db, ns, &key(i), &value(i));
        }
        assert!(db.metrics_snapshot().memtable_flushes > 0, "the memtable sealed");
        db.mark_crashed();
        ns
    };
    let db = Database::open(dir.path(), config()).unwrap();
    for i in 0..30u64 {
        assert_eq!(db.get_ns(ns, &key(i)).unwrap(), Some(value(i)), "key {i}");
    }
}

/// A clean shutdown compacts every store, which drops tombstones in L1.
/// The vector index WAL must have retired what those stores flushed by
/// then, or the next open replays records whose keys were deleted.
#[test]
fn a_clean_shutdown_does_not_bring_back_deleted_keys() {
    let dir = TempDir::new().unwrap();
    let ns = {
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        let ns = db.create_namespace("v_sparse_vector").unwrap();
        log_and_apply(&db, ns, b"moved-away", b"chunk code");
        log_and_apply(&db, ns, b"kept", b"chunk code");
        db.delete_ns(ns, b"moved-away").unwrap();
        db.shutdown().unwrap(); // flushes, then compacts into L1
        ns
    };
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    assert_eq!(
        db.get_ns(ns, b"moved-away").unwrap(),
        None,
        "a deleted key came back after a clean shutdown"
    );
    assert_eq!(db.get_ns(ns, b"kept").unwrap(), Some(b"chunk code".to_vec()));
    assert_eq!(
        db.metrics_snapshot().vector_wal_records_replayed,
        0,
        "a clean shutdown retires everything"
    );
}
