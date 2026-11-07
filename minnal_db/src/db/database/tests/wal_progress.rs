//! Saving WAL retirement progress (the cut and each namespace's safe offset) for both logs, and undoing it on failure.

use super::*;

// ── WAL retirement progress (cut + per-namespace safe offsets) ──────────────
//
// A flush saves the WAL's progress on its *written* event, before its L0
// files are registered, and the progress goes live only once that save and
// the vector index WAL's both succeeded. These pin the failure paths and the
// reason the offsets are per namespace.

#[test]
fn a_failed_progress_save_undoes_the_flush_and_a_retry_completes_it() {
    let dir = TempDir::new().unwrap();
    {
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        let ns = db.create_namespace("ns").unwrap();
        for i in 0..20u32 {
            db.put_ns(ns, format!("k{i}").as_bytes(), b"v").unwrap();
        }
        let cut_before = db.wal_metadata.read().cut;

        db.wal_flush_observer.script_saves(&[true]);
        assert!(
            db.get_store(ns).unwrap().flush_memtable_to_level0().is_err(),
            "the failed save must veto the flush"
        );
        assert_eq!(db.wal_metadata.read().cut, cut_before, "a failed save must not move the live cut");
        assert_eq!(saved_wal_metadata(&dir).cut, cut_before);
        for i in 0..20u32 {
            assert_eq!(db.get_ns(ns, format!("k{i}").as_bytes()).unwrap(), Some(b"v".to_vec()));
        }

        db.get_store(ns).unwrap().flush_memtable_to_level0().unwrap();
        let meta = db.wal_metadata.read().clone();
        assert_eq!(meta.cut, meta.tail, "the retry should retire everything");
        assert_eq!(saved_wal_metadata(&dir).cut, meta.cut);
        db.mark_crashed();
    }
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    let ns = db.get_namespace_id("ns").unwrap();
    for i in 0..20u32 {
        assert_eq!(db.get_ns(ns, format!("k{i}").as_bytes()).unwrap(), Some(b"v".to_vec()), "k{i} lost");
    }
}

/// The vector index WAL saves its progress in the same event, after the main
/// WAL. If it fails, the flush is undone and its L0 files deleted, so the main
/// WAL must put back the progress it already saved: left in place, recovery
/// would skip entries whose only on-disk copy was just deleted.
#[test]
fn a_vector_wal_failure_after_the_main_save_puts_the_main_progress_back() {
    let dir = TempDir::new().unwrap();
    {
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        let ns = db.create_namespace("docs_sparse_vector").unwrap();
        for i in 0..20u32 {
            db.put_ns(ns, format!("main{i}").as_bytes(), b"v").unwrap();
        }
        db.put_ns_logged(ns, b"logged", b"code").unwrap();
        db.sync_vector_index_wal().unwrap();
        let before = saved_wal_metadata(&dir);

        db.vector_index_wal.fail_next_marker_write();
        assert!(db.get_store(ns).unwrap().flush_memtable_to_level0().is_err());
        let after = saved_wal_metadata(&dir);
        assert_eq!(
            (after.cut, &after.safe),
            (before.cut, &before.safe),
            "the main WAL's saved progress was not put back"
        );
        assert_eq!(db.wal_metadata.read().cut, before.cut, "the live cut moved for an undone flush");
        assert!(db.wal_flush_observer.check_poisoned().is_ok());
        db.mark_crashed();
    }
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    let ns = db.get_namespace_id("docs_sparse_vector").unwrap();
    for i in 0..20u32 {
        assert_eq!(db.get_ns(ns, format!("main{i}").as_bytes()).unwrap(), Some(b"v".to_vec()), "main{i} lost");
    }
    assert_eq!(db.get_ns(ns, b"logged").unwrap(), Some(b"code".to_vec()));
}

/// If putting the progress back fails too, the file on disk may claim entries
/// are in L0 that are not, so the main WAL stops accepting writes.
#[test]
fn a_failed_rollback_stops_the_main_wal() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    let ns = db.create_namespace("docs_sparse_vector").unwrap();
    db.put_ns(ns, b"main", b"v").unwrap();
    db.put_ns_logged(ns, b"logged", b"code").unwrap();
    db.sync_vector_index_wal().unwrap();

    db.vector_index_wal.fail_next_marker_write();
    db.wal_flush_observer.script_saves(&[false, true]);
    assert!(db.get_store(ns).unwrap().flush_memtable_to_level0().is_err());
    assert!(
        matches!(db.put_ns(ns, b"after", b"v"), Err(KVError::WalPoisoned(_))),
        "writes must be refused"
    );
    assert!(matches!(db.delete_ns(ns, b"main"), Err(KVError::WalPoisoned(_))));
}

/// Both logs' stats show positions the same way, `{segment_id, byte}`, and
/// the main WAL's pending bytes count entries only.
#[test]
fn both_logs_report_positions_as_segment_and_byte() {
    let dir = TempDir::new().unwrap();
    let db = Database::open_with_wal_segment_size(dir.path(), create_db_config(), 512).unwrap();
    for i in 0..40u32 {
        db.put(format!("k{i:03}").as_bytes(), &[b'v'; 40]).unwrap();
    }
    let s = db.wal_stats();
    let m = db.wal_metadata.read().clone();
    assert_eq!(s.tail, crate::db::wal::segmented_log::LogPosition::of(m.tail));
    assert!(s.tail.segment_id >= 2);
    assert_eq!(s.pending_bytes, db.wal.bytes_between(m.cut, m.tail));
    assert!(s.pending_bytes > 0 && s.segments.iter().any(|seg| !seg.retired));
    assert_eq!(s.durable_tail, s.tail, "every acknowledged write is fsynced");
    assert_eq!(s.stopped, None);
    let json = serde_json::to_value(&s).unwrap();
    assert!(json["tail"]["segment_id"].is_u64() && json["tail"]["byte"].is_u64(), "{json}");
    db.wal.fail_next_sync();
    assert!(db.put(b"after", b"v").is_err());
    assert!(db.wal_stats().stopped.is_some(), "a failed fsync must show in the stats");
    let v = serde_json::to_value(db.vector_index_wal_stats().unwrap()).unwrap();
    assert!(v["tail"]["segment_id"].is_u64() && v["tail"]["byte"].is_u64(), "{v}");
}
