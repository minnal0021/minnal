//! Retirement and GC: the cut, its marker and what holds it back.

use super::*;

/// Appends `n` records for `ns` and lets them count as applied.
fn applied(wal: &VectorIndexWal, ns: u32, n: u64) -> Vec<Appended> {
    (0..n).map(|i| wal.append(ns, &key(i), &value(i), i + 1).unwrap().appended()).collect()
}

fn seal_and_flush(wal: &VectorIndexWal, ns: u32, version: u64) {
    wal.on_memtable_sealed(ns, version);
    wal.on_memtable_written(ns, version).unwrap();
}

#[test]
fn a_flush_retires_its_memtables_records_and_the_cut_survives_reopen() {
    let dir = TempDir::new().unwrap();
    let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
    applied(&wal, 1, 3);
    assert_eq!(wal.retired_cut(), 0);
    assert_eq!(wal.namespaces_holding_cut(), vec![1]);
    seal_and_flush(&wal, 1, 0);
    assert_eq!(wal.retired_cut(), wal.tail());
    assert!(wal.namespaces_holding_cut().is_empty());
    let cut = wal.retired_cut();
    drop(wal);
    let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
    assert_eq!(wal.retired_cut(), cut);
    assert_eq!(rows_from(&wal, wal.retired_cut()), vec![]);
}

#[test]
fn memtables_retire_in_version_order_only() {
    let dir = TempDir::new().unwrap();
    let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
    applied(&wal, 1, 1);
    wal.on_memtable_sealed(1, 0);
    applied(&wal, 1, 1);
    wal.on_memtable_sealed(1, 1);
    wal.on_memtable_written(1, 1).unwrap();
    assert_eq!(wal.retired_cut(), 0, "version 0 is not flushed yet");
    wal.on_memtable_written(1, 0).unwrap();
    assert_eq!(wal.retired_cut(), wal.tail());
}

#[test]
fn a_record_not_yet_applied_holds_the_cut() {
    let dir = TempDir::new().unwrap();
    let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
    applied(&wal, 1, 2);
    let in_flight = wal.append(1, b"late", b"v", 9).unwrap();
    // Its apply will land in the next memtable, so this seal must not claim it.
    seal_and_flush(&wal, 1, 0);
    assert_eq!(wal.retired_cut(), in_flight.start);
    assert_eq!(wal.namespaces_holding_cut(), vec![1]);
    drop(in_flight);
    seal_and_flush(&wal, 1, 1);
    assert_eq!(wal.retired_cut(), wal.tail());
}

#[test]
fn another_namespaces_record_in_flight_does_not_hold_a_flushed_namespace() {
    let dir = TempDir::new().unwrap();
    let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
    applied(&wal, 1, 2);
    let other = wal.append(2, b"k", b"v", 9).unwrap();
    applied(&wal, 1, 1);
    seal_and_flush(&wal, 1, 0);
    // Namespace 1 flushed everything it wrote; only 2 holds the cut.
    assert_eq!(wal.namespaces_holding_cut(), vec![2]);
    assert_eq!(wal.retired_cut(), 0);
    drop(other);
    seal_and_flush(&wal, 2, 0);
    assert_eq!(wal.retired_cut(), wal.tail());
    assert!(wal.namespaces_holding_cut().is_empty());
}

#[test]
fn the_cut_is_the_slowest_unflushed_namespace() {
    let dir = TempDir::new().unwrap();
    let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
    applied(&wal, 2, 1);
    seal_and_flush(&wal, 2, 0);
    let after_two = wal.retired_cut();
    assert_eq!(after_two, wal.tail());
    applied(&wal, 1, 2);
    applied(&wal, 2, 1);
    seal_and_flush(&wal, 1, 0);
    assert_eq!(wal.namespaces_holding_cut(), vec![2]);
    assert_eq!(wal.retired_cut(), after_two, "namespace 2's safe offset");
    seal_and_flush(&wal, 2, 1);
    assert_eq!(wal.retired_cut(), wal.tail());
}

#[test]
fn a_flush_without_a_seal_record_retires_nothing() {
    let dir = TempDir::new().unwrap();
    let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
    applied(&wal, 1, 2);
    wal.on_memtable_written(1, 0).unwrap();
    assert_eq!(wal.retired_cut(), 0);
    assert_eq!(wal.namespaces_holding_cut(), vec![1]);
    seal_and_flush(&wal, 1, 1);
    assert_eq!(wal.retired_cut(), wal.tail());
}

#[test]
fn events_for_a_namespace_that_never_appended_are_ignored() {
    let dir = TempDir::new().unwrap();
    let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
    seal_and_flush(&wal, 7, 0);
    applied(&wal, 1, 1);
    seal_and_flush(&wal, 7, 1);
    assert_eq!(wal.retired_cut(), 0);
    assert_eq!(wal.namespaces_holding_cut(), vec![1]);
}

#[test]
fn forgetting_a_namespace_releases_the_cut() {
    let dir = TempDir::new().unwrap();
    let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
    applied(&wal, 1, 1);
    applied(&wal, 2, 1);
    seal_and_flush(&wal, 1, 0);
    assert_eq!(wal.retired_cut(), 0);
    wal.forget_namespace(2);
    assert_eq!(wal.retired_cut(), wal.tail());
    assert!(wal.namespaces_holding_cut().is_empty());
}

#[test]
fn gc_deletes_only_segments_wholly_below_the_cut() {
    let dir = TempDir::new().unwrap();
    let metrics = Arc::new(Metrics::default());
    let (wal, _) = VectorIndexWal::open(dir.path(), 256, Arc::clone(&metrics)).unwrap();
    applied(&wal, 1, 8); // segments 0, 1, 2
    assert!(!wal.has_deletable_segments().unwrap());
    assert_eq!(wal.garbage_collect().unwrap(), (0, 0));
    seal_and_flush(&wal, 1, 0);
    assert_eq!(split(wal.retired_cut()).0, 2);
    assert!(wal.has_deletable_segments().unwrap());
    // Segment files are their full size.
    assert_eq!(wal.garbage_collect().unwrap(), (2, 2 * 256));
    assert_eq!(wal.segments().unwrap().iter().map(|s| s.0).collect::<Vec<_>>(), vec![2]);
    assert!(!wal.has_deletable_segments().unwrap());
    let s = metrics.snapshot();
    assert_eq!((s.vector_wal_gc_runs, s.vector_wal_segments_deleted), (2, 2));

    let more = applied(&wal, 1, 2);
    let cut = wal.retired_cut();
    drop(wal);
    let (wal, _) = open_at(dir.path(), 256);
    assert_eq!(wal.retired_cut(), cut);
    let rows = rows_from(&wal, cut);
    assert_eq!(
        rows.iter().map(|r| r.0).collect::<Vec<_>>(),
        more.iter().map(|a| a.start).collect::<Vec<_>>()
    );
}

#[test]
fn a_bad_retired_cut_marker_fails_the_open() {
    let fresh = || {
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), 256);
        applied(&wal, 1, 8);
        seal_and_flush(&wal, 1, 0);
        wal.garbage_collect().unwrap();
        dir
    };
    let marker = |dir: &TempDir| dir.path().join(RETIRED_CUT_FILE);

    let dir = fresh();
    std::fs::write(
        marker(&dir),
        progress_bytes(&Progress {
            cut: position(9, 16),
            safe: vec![],
        }),
    )
    .unwrap();
    assert!(corruption(reopen(dir.path(), 256)).contains("past the last segment"));

    let dir = fresh();
    let cut = read_progress(dir.path()).unwrap().unwrap().cut;
    std::fs::write(
        marker(&dir),
        progress_bytes(&Progress {
            cut,
            safe: vec![(1, cut - 1)],
        }),
    )
    .unwrap();
    assert!(corruption(reopen(dir.path(), 256)).contains("below the cut"));

    let dir = fresh();
    std::fs::write(
        marker(&dir),
        progress_bytes(&Progress {
            cut: position(0, 16),
            safe: vec![],
        }),
    )
    .unwrap();
    assert!(corruption(reopen(dir.path(), 256)).contains("below the oldest segment"));

    let dir = fresh();
    std::fs::remove_file(marker(&dir)).unwrap();
    assert!(corruption(reopen(dir.path(), 256)).contains("no progress was saved"));

    let dir = fresh();
    let mut bytes = std::fs::read(marker(&dir)).unwrap();
    bytes[5] ^= 0xff;
    std::fs::write(marker(&dir), bytes).unwrap();
    assert!(corruption(reopen(dir.path(), 256)).contains("damaged"));
}

#[test]
fn the_cut_never_moves_back() {
    let dir = TempDir::new().unwrap();
    let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
    applied(&wal, 1, 2);
    seal_and_flush(&wal, 1, 0);
    let cut = wal.retired_cut();
    // A namespace that starts writing now has a safe offset of 0 until it
    // flushes; that must not drag the cut back.
    applied(&wal, 2, 1);
    wal.forget_namespace(3);
    assert_eq!(wal.retired_cut(), cut);
    drop(wal);
    let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
    assert_eq!(wal.retired_cut(), cut);
}

#[test]
fn stats_report_positions_unretired_bytes_and_holders() {
    let dir = TempDir::new().unwrap();
    let (wal, _) = open_at(dir.path(), 256);
    applied(&wal, 1, 4); // segment 0 sealed with 3, segment 1 holds 1
    applied(&wal, 2, 1);
    let name = |ns: u32| format!("ns{ns}");
    let s = wal.stats(name).unwrap();
    assert_eq!(s.segment_count, 2);
    assert_eq!(
        s.tail,
        VectorIndexWalPosition {
            segment_id: 1,
            byte: SEGMENT_HEADER_LEN + 2 * FRAME
        }
    );
    assert_eq!(s.retired_cut, VectorIndexWalPosition { segment_id: 0, byte: 0 });
    assert_eq!(s.unretired_bytes, u64::from(5 * FRAME), "record bytes only: no headers, no end marker");
    assert_eq!(s.last_sequence, 4);
    assert_eq!(s.holding_cut.iter().map(|h| h.namespace.as_str()).collect::<Vec<_>>(), vec!["ns1", "ns2"]);

    seal_and_flush(&wal, 1, 0);
    seal_and_flush(&wal, 2, 0);
    let s = wal.stats(name).unwrap();
    assert_eq!(s.retired_cut, s.tail);
    assert_eq!(s.unretired_bytes, 0);
    assert!(s.holding_cut.is_empty());
    // Sealing segment 0 fsynced it; nothing has synced segment 1 yet.
    assert_eq!(
        s.durable_tail,
        VectorIndexWalPosition {
            segment_id: 0,
            byte: SEGMENT_HEADER_LEN + 3 * FRAME
        }
    );
}

#[test]
fn a_reopened_log_holds_the_cut_for_records_nobody_flushed() {
    let dir = TempDir::new().unwrap();
    let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
    applied(&wal, 3, 1);
    seal_and_flush(&wal, 3, 0);
    let cut = wal.retired_cut();
    let unflushed = applied(&wal, 1, 2);
    drop(wal); // a crash: the in-memory state is gone, the records are not

    let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
    // Namespace 3 has nothing above the cut, so only 1 holds it.
    assert_eq!(wal.namespaces_holding_cut(), vec![1]);
    assert_eq!(wal.tracked_namespaces(), vec![1]);
    // Another namespace's flush must not carry the cut past 1's records.
    applied(&wal, 2, 1);
    seal_and_flush(&wal, 2, 0);
    assert_eq!(wal.retired_cut(), cut);
    assert!(wal.retired_cut() <= unflushed[0].start);
    // Only 1's own flush releases it.
    seal_and_flush(&wal, 1, 0);
    assert_eq!(wal.retired_cut(), wal.tail());
}

#[test]
fn the_marker_round_trips_with_per_namespace_offsets() {
    let p = Progress {
        cut: position(1, 40),
        safe: vec![(2, position(1, 90)), (7, position(2, 16))],
    };
    let dir = TempDir::new().unwrap();
    crate::support::write_atomic_durable(&dir.path().join(RETIRED_CUT_FILE), &progress_bytes(&p)).unwrap();
    assert_eq!(read_progress(dir.path()).unwrap(), Some(p.clone()));
    assert_eq!(p.replay_from(2), position(1, 90));
    assert_eq!(p.replay_from(7), position(2, 16));
    assert_eq!(p.replay_from(3), position(1, 40), "an unlisted namespace replays from the cut");
    let mut bytes = progress_bytes(&p);
    bytes.truncate(bytes.len() - 5);
    std::fs::write(dir.path().join(RETIRED_CUT_FILE), &bytes).unwrap();
    assert!(read_progress(dir.path()).is_err());
}

#[test]
fn replay_skips_each_namespaces_records_below_its_own_safe_offset() {
    let dir = TempDir::new().unwrap();
    let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
    let flushed = applied(&wal, 1, 2);
    let held = applied(&wal, 2, 1);
    let after = applied(&wal, 1, 1);
    wal.on_memtable_sealed(1, 0); // covers all of namespace 1's records so far
    wal.on_memtable_written(1, 0).unwrap();
    assert_eq!(wal.retired_cut(), 0, "namespace 2 holds the global cut");
    drop(wal);

    let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
    let mut replayed = Vec::new();
    wal.replay(|r| -> Result<()> {
        replayed.push((r.ns_id, r.seq));
        Ok(())
    })
    .unwrap();
    // Only namespace 2's record: namespace 1's are below its own offset,
    // although they are above the global cut.
    assert_eq!(replayed, vec![(2, 1)]);
    let _ = (flushed, held, after);
}

#[test]
fn a_marker_past_the_durable_tail_starts_a_new_segment() {
    let dir = TempDir::new().unwrap();
    let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
    applied(&wal, 1, 3); // never synced
    seal_and_flush(&wal, 1, 0); // but written to L0: the marker names the tail
    let marked = wal.retired_cut();
    wal.lose_unsynced(); // power loss
    drop(wal);

    let (wal, report) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
    assert_eq!(report.tail, position(1, SEGMENT_HEADER_LEN));
    assert!(wal.tail() > marked, "new appends land above every position the marker names");
    let fresh = wal.append(1, b"new", b"v", 9).unwrap().appended();
    drop(wal);
    let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
    let mut replayed = Vec::new();
    wal.replay(|r| -> Result<()> {
        replayed.push(r.key.to_vec());
        Ok(())
    })
    .unwrap();
    assert_eq!(replayed, vec![b"new".to_vec()], "the new record at {:?} is replayed", fresh);
}

/// A marker that cannot be written changes nothing: the event fails (so
/// the LSM undoes the flush), the cut and the marker stay, no later marker
/// carries the failed flush, and the retry retires the records.
#[test]
fn a_failed_marker_write_changes_nothing_and_the_retry_retires() {
    let dir = TempDir::new().unwrap();
    let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
    let ns1 = applied(&wal, 1, 3);
    wal.on_memtable_sealed(1, 0);
    let marker_before = read_progress(dir.path()).unwrap();
    wal.fail_next_marker_write();
    assert!(wal.on_memtable_written(1, 0).is_err());
    assert_eq!(wal.retired_cut(), 0);
    assert_eq!(wal.namespaces_holding_cut(), vec![1]);
    assert_eq!(read_progress(dir.path()).unwrap(), marker_before);

    // Another namespace's flush writes a marker: it must not carry 1's.
    applied(&wal, 2, 1);
    seal_and_flush(&wal, 2, 0);
    assert!(wal.retired_cut() <= ns1[0].start, "1's records are not retired");
    let marker = read_progress(dir.path()).unwrap().unwrap();
    assert!(marker.replay_from(1) <= ns1[0].start);

    // The LSM retries the same memtable version.
    wal.on_memtable_written(1, 0).unwrap();
    assert!(wal.namespaces_holding_cut().is_empty());
    assert_eq!(wal.retired_cut(), wal.tail());
}

/// A dropped namespace is never tracked again: a write that raced the
/// drop would otherwise hold the cut until the next restart.
#[test]
fn an_append_after_a_namespace_is_forgotten_does_not_hold_the_cut() {
    let dir = TempDir::new().unwrap();
    let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
    applied(&wal, 1, 1);
    wal.forget_namespace(1);
    applied(&wal, 1, 1); // a late write that raced the drop
    assert!(wal.namespaces_holding_cut().is_empty());
    wal.on_memtable_sealed(1, 0);
    wal.on_memtable_written(1, 0).unwrap();
    applied(&wal, 2, 1);
    seal_and_flush(&wal, 2, 0);
    assert_eq!(wal.retired_cut(), wal.tail(), "nothing holds the cut");
}

/// After an fsync fails, the log refuses appends and syncs until it is
/// reopened, and the stats say why.
#[test]
fn a_failed_fsync_stops_the_log_until_it_is_reopened() {
    let dir = TempDir::new().unwrap();
    let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
    applied(&wal, 1, 1);
    wal.fail_next_sync();
    assert!(matches!(wal.sync(), Err(VectorIndexWalError::Log(LogError::Poisoned(_)))));
    assert!(matches!(
        wal.append(1, b"k", b"v", 9),
        Err(VectorIndexWalError::Log(LogError::Poisoned(_)))
    ));
    assert!(
        matches!(wal.sync(), Err(VectorIndexWalError::Log(LogError::Poisoned(_)))),
        "a later sync must not succeed"
    );
    assert!(wal.stats(|n| n.to_string()).unwrap().stopped.is_some());
    drop(wal);
    let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
    wal.append(1, b"k", b"v", 9).unwrap();
    wal.sync().unwrap();
    assert!(wal.stats(|n| n.to_string()).unwrap().stopped.is_none());
}

/// Concurrent syncs share fsyncs (commit rounds, as in the main WAL), and
/// each still returns only once every append made before it is durable.
#[test]
fn concurrent_syncs_share_fsyncs() {
    let dir = TempDir::new().unwrap();
    let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
    let before = wal.metrics.vector_wal_fsyncs.load(Ordering::SeqCst);
    let seq = AtomicU64::new(1);
    std::thread::scope(|s| {
        for t in 0..16u32 {
            let (wal, seq) = (&wal, &seq);
            s.spawn(move || {
                for i in 0..50u32 {
                    let write = wal
                        .append(t, format!("k{i}").as_bytes(), b"v", seq.fetch_add(1, Ordering::SeqCst))
                        .unwrap();
                    let end = write.end;
                    drop(write);
                    wal.sync().unwrap();
                    assert!(wal.durable_tail() >= end, "a sync returned before its append was durable");
                }
            });
        }
    });
    let fsyncs = wal.metrics.vector_wal_fsyncs.load(Ordering::SeqCst) - before;
    assert!(
        fsyncs < 800,
        "16 threads syncing concurrently never shared an fsync ({fsyncs} fsyncs for 800 syncs)"
    );
}

/// Syncs that arrive while others are pending share one fsync: held at the
/// commit point together, then released, eight syncs cost one fsync, not
/// eight. (Without commit rounds each would fsync on its own, since none
/// sees another's fsync finish first.)
#[test]
fn syncs_waiting_together_share_one_fsync() {
    let dir = TempDir::new().unwrap();
    let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
    let before = wal.metrics.vector_wal_fsyncs.load(Ordering::SeqCst);
    wal.log.rounds.freeze(true);
    std::thread::scope(|s| {
        let syncs: Vec<_> = (0..8u32)
            .map(|t| {
                let wal = &wal;
                s.spawn(move || {
                    drop(wal.append(t, b"k", b"v", u64::from(t) + 1).unwrap());
                    wal.sync()
                })
            })
            .collect();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while wal.log.rounds.frozen_callers() < 8 {
            assert!(std::time::Instant::now() < deadline, "the syncs never reached the commit point");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        wal.log.rounds.freeze(false);
        for sync in syncs {
            sync.join().unwrap().unwrap();
        }
    });
    assert_eq!(
        wal.metrics.vector_wal_fsyncs.load(Ordering::SeqCst) - before,
        1,
        "eight waiting syncs should share one fsync"
    );
    assert_eq!(wal.durable_tail(), wal.tail());
}

/// A failed fsync fails every sync waiting on that round, and stops the log.
#[test]
fn a_failed_round_fails_every_waiting_sync() {
    let dir = TempDir::new().unwrap();
    let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
    wal.log.rounds.freeze(true);
    std::thread::scope(|s| {
        let syncs: Vec<_> = (0..4u32)
            .map(|t| {
                let wal = &wal;
                s.spawn(move || {
                    drop(wal.append(t, b"k", b"v", u64::from(t) + 1).unwrap());
                    wal.sync()
                })
            })
            .collect();
        while wal.log.rounds.frozen_callers() < 4 {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        wal.fail_next_sync();
        wal.log.rounds.freeze(false);
        for sync in syncs {
            assert!(
                matches!(sync.join().unwrap(), Err(VectorIndexWalError::Log(LogError::Poisoned(_)))),
                "a waiting sync succeeded"
            );
        }
    });
    assert!(wal.stats(|n| n.to_string()).unwrap().stopped.is_some());
}

/// Segments wholly below the cut are retired: open does not read them, so
/// damage there (they only wait for GC) does not stop the database.
#[test]
fn open_does_not_read_segments_below_the_cut() {
    let dir = TempDir::new().unwrap();
    let (wal, _) = open_at(dir.path(), 256);
    let records = applied(&wal, 1, 8); // segments 0, 1, 2
    seal_and_flush(&wal, 1, 0);
    assert_eq!(split(wal.retired_cut()).0, 2);
    let last_seq = wal.last_seq();
    drop(wal);
    flip(dir.path(), 0, u64::from(SEGMENT_HEADER_LEN + FRAME_HEADER_LEN + 2));
    let (wal, report) = open_at(dir.path(), 256);
    assert_eq!(report.tail, records.last().unwrap().end);
    assert!(wal.namespaces_holding_cut().is_empty());
    // Nothing at or above the cut, so no seq to report from the log.
    assert!(report.last_seq < last_seq);
}

/// A seal runs under a memtable's write lock, for every namespace of the
/// database. It must never wait for `retirement`, which is held across
/// the marker's fsyncs.
#[test]
fn a_seal_never_waits_for_a_marker_write() {
    let dir = TempDir::new().unwrap();
    let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
    applied(&wal, 1, 1);
    let held = wal.retirement.lock(); // as during a marker write
    std::thread::scope(|s| {
        let (tx, rx) = std::sync::mpsc::channel();
        let wal = &wal;
        s.spawn(move || {
            wal.on_memtable_sealed(1, 0); // tracked
            wal.on_memtable_sealed(7, 0); // untracked
            tx.send(()).unwrap();
        });
        let done = rx.recv_timeout(std::time::Duration::from_secs(5));
        drop(held);
        assert!(done.is_ok(), "a seal waited for the retirement lock");
    });
}
