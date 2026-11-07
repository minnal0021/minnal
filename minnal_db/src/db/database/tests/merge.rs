//! `merge` (atomic read-modify-write): what the closure returns, and the serialisation that keeps it safe against concurrent writers.

use super::*;

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

        let before = db.wal_metadata.read().tail;
        assert_eq!(db.merge(b"never-existed", b"", |_, _| Ok(None)).unwrap(), None);

        assert_eq!(db.wal_metadata.read().tail, before, "an absent -> absent merge appended a WAL entry");
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
        let before = db.wal_metadata.read().tail;

        let err = db
            .merge(b"capped", &1u64.to_le_bytes(), |existing, _| {
                if counter_of(existing) >= 9 {
                    return Err(KVError::MergeAborted("already at cap".into()));
                }
                Ok(Some(vec![]))
            })
            .unwrap_err();

        assert!(matches!(err, KVError::MergeAborted(_)), "unexpected error: {err:?}");
        assert_eq!(db.wal_metadata.read().tail, before, "an aborted merge appended a WAL entry");
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
        let entries_before = db.wal_metadata.read().tail;

        // `put` swallows it and reports success.
        let put_result = db.put(b"target", &big);
        let readable_after_put = db.get(b"target");

        // `merge` surfaces it.
        let merge_result = db.merge(b"target", &big, |_, operand| Ok(Some(operand.to_vec())));

        let entries_after = db.wal_metadata.read().tail;
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

    // ── merge_ns_no_wal ──────────────────────────────────────────────

    /// The same results as `merge` — write, delete, abort, absent no-op — with
    /// no WAL entry and no fsync for any of them.
    #[test]
    fn a_no_wal_merge_writes_like_merge_without_the_wal() {
        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        let ns = DEFAULT_NAMESPACE_ID;
        db.put_ns(ns, b"doomed", b"value").unwrap();
        let before = db.metrics_snapshot();

        let written = db.merge_ns_no_wal(ns, b"counter", &5u64.to_le_bytes(), add).unwrap();
        assert_eq!(written, Some(5u64.to_le_bytes().to_vec()));
        db.merge_ns_no_wal(ns, b"counter", &2u64.to_le_bytes(), add).unwrap();
        assert_eq!(db.get_ns(ns, b"counter").unwrap(), Some(7u64.to_le_bytes().to_vec()));

        assert_eq!(db.merge_ns_no_wal(ns, b"doomed", b"", |_, _| Ok(None)).unwrap(), None);
        assert_eq!(db.get_ns(ns, b"doomed").unwrap(), None, "the merge deleted the key");

        let aborted = db.merge_ns_no_wal(ns, b"counter", b"", |_, _| Err(KVError::MergeAborted("no".into())));
        assert!(matches!(aborted, Err(KVError::MergeAborted(_))));
        assert_eq!(
            db.get_ns(ns, b"counter").unwrap(),
            Some(7u64.to_le_bytes().to_vec()),
            "an abort writes nothing"
        );
        db.merge_ns_no_wal(ns, b"absent", b"", |_, _| Ok(None)).unwrap();

        let after = db.metrics_snapshot();
        assert_eq!(after.wal_writes, before.wal_writes, "a no-WAL merge must not fsync the WAL");
        assert_eq!(after.merges - before.merges, 5);
        assert_eq!(after.no_wal_puts - before.no_wal_puts, 2, "the two writing merges");
        assert_eq!(
            after.no_wal_deletes - before.no_wal_deletes,
            1,
            "only the delete of a present key writes a tombstone"
        );
        assert_eq!(after.puts, before.puts, "no WAL-backed put");
        assert_eq!(after.deletes, before.deletes, "no WAL-backed delete");

        db.shutdown().unwrap();
    }

    /// The stripe covers the read, the closure and the write, as for `merge`:
    /// see `a_blind_put_cannot_land_in_the_middle_of_a_merge`.
    #[test]
    fn a_blind_put_cannot_land_in_the_middle_of_a_no_wal_merge() {
        let dir = TempDir::new().unwrap();
        let db = Arc::new(Database::open(dir.path(), create_db_config()).unwrap());
        let ns = DEFAULT_NAMESPACE_ID;
        db.put_ns(ns, b"k", b"initial").unwrap();

        std::thread::scope(|s| {
            let merger = {
                let db = Arc::clone(&db);
                s.spawn(move || {
                    db.merge_ns_no_wal(ns, b"k", b"", |existing, _| {
                        std::thread::sleep(Duration::from_millis(300));
                        let mut v = existing.unwrap_or(b"").to_vec();
                        v.extend_from_slice(b"+merged");
                        Ok(Some(v))
                    })
                    .unwrap();
                })
            };

            std::thread::sleep(Duration::from_millis(50));
            db.put_ns(ns, b"k", b"sentinel").unwrap();
            merger.join().unwrap();
        });

        assert_eq!(
            db.get_ns(ns, b"k").unwrap(),
            Some(b"sentinel".to_vec()),
            "the put was applied while the no-WAL merge held the key, so the merge overwrote it"
        );

        db.shutdown().unwrap();
    }

    /// A no-WAL merge is lost if the process dies before its memtable is
    /// flushed, and kept once it is: the WAL-backed write it replaced replays
    /// in the first case and not in the second.
    #[test]
    fn a_no_wal_merge_survives_a_crash_only_once_flushed() {
        for flush in [false, true] {
            let dir = TempDir::new().unwrap();
            {
                let db = Database::open(dir.path(), create_db_config()).unwrap();
                let ns = db.create_namespace("queue").unwrap();
                db.put_ns(ns, b"entry", b"queued").unwrap();
                db.merge_ns_no_wal(ns, b"entry", b"", |_, _| Ok(None)).unwrap();
                if flush {
                    assert_eq!(db.flush_no_wal_memtables(), 1);
                }
                db.crash();
            }
            let db = Database::open(dir.path(), create_db_config()).unwrap();
            let ns = db.get_namespace_id("queue").unwrap();
            let expected = if flush { None } else { Some(b"queued".to_vec()) };
            assert_eq!(db.get_ns(ns, b"entry").unwrap(), expected, "flushed before the crash: {flush}");
            db.crash();
        }
    }

    /// The one crash state a no-WAL completion depends on outside itself: its
    /// tombstone is in L0 while the older WAL-backed put it removed still
    /// replays (a crash between writing the L0 file and marking the put
    /// persisted). The put replays with its own, lower seq, so the tombstone
    /// must win.
    #[test]
    fn a_replayed_older_put_loses_to_a_flushed_no_wal_delete() {
        let dir = TempDir::new().unwrap();
        {
            let db = Database::open(dir.path(), create_db_config()).unwrap();
            let ns = db.create_namespace("queue").unwrap();
            db.put_ns(ns, b"entry", b"queued").unwrap();
            db.merge_ns_no_wal(ns, b"entry", b"", |_, _| Ok(None)).unwrap();
            assert_eq!(db.flush_no_wal_memtables(), 1);
            // The put again, unpersisted, with a seq below the tombstone's.
            db.simulate_crash_with_wal_entries(vec![WalEntry::new_upsert_ns(ns, b"entry".to_vec(), b"queued".to_vec()).with_sequence(1)])
                .unwrap();
            db.crash();
        }
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        let ns = db.get_namespace_id("queue").unwrap();
        assert_eq!(
            db.get_ns(ns, b"entry").unwrap(),
            None,
            "the replayed put came back over the newer tombstone in L0"
        );
        db.crash();
    }
}
