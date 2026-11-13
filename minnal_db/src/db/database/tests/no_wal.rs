//! Writes that skip the WAL: when they are flushed so they survive a crash.

use super::*;

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

        db.crash();
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
        db.crash();
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
        db.crash();
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
        db.crash();
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
    let level0 = crate::layout::namespace_lsm_dir(&crate::layout::namespace_data_dir(dir.path(), "vectors")).join("level0");
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
