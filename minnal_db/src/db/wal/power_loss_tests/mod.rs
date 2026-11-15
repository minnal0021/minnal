//! Power-loss and crash tests for the main WAL's write path: group commit,
//! retirement by saved progress, checksummed frames.
//!
//! A process kill leaves the page cache intact, so it cannot show a write that
//! returned before an fsync covered it. These tests use a power-loss model
//! instead (`Database::power_loss_image`): every writer and worker is stopped
//! at one moment, the database directory is copied, and in the copy each log
//! is cut back to what an fsync actually covered. Writers stopped inside a
//! commit (`Wal::freeze_commits`) have appended but not been acknowledged —
//! the state a power loss would catch them in. The copy is then opened, which
//! runs recovery, and checked: every acknowledged write is there, and every
//! unacknowledged one is either whole or absent.

use crate::namespace::NamespaceId;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use parking_lot::Mutex;
use tempfile::TempDir;

use crate::db::database::Database;
use crate::db::error::KVError;
use crate::db::test_support::*;
use crate::db::workers::index_checkpoint::IndexCheckpointTarget;

/// Open the power-loss image of `db` in a fresh directory.
fn open_image(db: &Database) -> (TempDir, Database) {
    let dir = TempDir::new().unwrap();
    let image = dir.path().join("db");
    db.power_loss_image(&image).unwrap();
    let reopened = Database::open(&image, create_db_config()).unwrap();
    (dir, reopened)
}

/// Wait until `cond` holds, failing the test after 10 s.
fn wait_for(what: &str, cond: impl Fn() -> bool) {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while !cond() {
        assert!(std::time::Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn an_acknowledged_write_survives_a_power_loss_and_an_unacknowledged_one_does_not() {
    let dir = TempDir::new().unwrap();
    let db = Arc::new(Database::open(dir.path(), create_db_config()).unwrap());
    let ns = db.create_namespace("ns").unwrap();
    db.put_ns(ns, b"acked", b"v").unwrap();

    db.wal.freeze_commits(true);
    let writer = {
        let db = Arc::clone(&db);
        std::thread::spawn(move || db.put_ns(ns, b"in-flight", b"v"))
    };
    wait_for("the writer to reach its commit", || db.wal.frozen_writers() == 1);
    assert_eq!(
        db.get_ns(ns, b"in-flight").unwrap(),
        None,
        "a write was readable before an fsync covered it"
    );
    let (_image_dir, image) = open_image(&db);
    db.wal.freeze_commits(false);
    writer.join().unwrap().unwrap();

    let ns_image = image.get_namespace_id("ns").unwrap();
    assert_eq!(image.get_ns(ns_image, b"acked").unwrap(), Some(b"v".to_vec()));
    assert_eq!(
        image.get_ns(ns_image, b"in-flight").unwrap(),
        None,
        "an unsynced append survived the power loss"
    );
}

/// Concurrent writers share fsyncs, and every one of them is durable once it
/// returns.
#[test]
fn concurrent_writers_share_fsyncs_and_all_survive_a_power_loss() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    let ns = db.create_namespace("ns").unwrap();
    let before = db.metrics_snapshot();
    std::thread::scope(|s| {
        for t in 0..16u32 {
            let db = &db;
            s.spawn(move || {
                for i in 0..100u32 {
                    db.put_ns(ns, format!("t{t}-{i}").as_bytes(), b"v").unwrap();
                }
            });
        }
    });
    let after = db.metrics_snapshot();
    let (rounds, writes) = (
        after.wal_commit_rounds - before.wal_commit_rounds,
        after.wal_commit_writes - before.wal_commit_writes,
    );
    assert_eq!(writes, 1600);
    assert!(
        rounds < writes,
        "16 concurrent writers never shared an fsync ({rounds} rounds for {writes} writes)"
    );

    let (_image_dir, image) = open_image(&db);
    let ns_image = image.get_namespace_id("ns").unwrap();
    for t in 0..16u32 {
        for i in 0..100u32 {
            assert_eq!(
                image.get_ns(ns_image, format!("t{t}-{i}").as_bytes()).unwrap(),
                Some(b"v".to_vec()),
                "t{t}-{i} lost"
            );
        }
    }
}

#[test]
fn a_lone_writer_issues_one_fsync_per_write() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    let ns = db.create_namespace("ns").unwrap();
    let before = db.metrics_snapshot();
    for i in 0..100u32 {
        db.put_ns(ns, format!("k{i}").as_bytes(), b"v").unwrap();
    }
    let after = db.metrics_snapshot();
    assert_eq!(after.wal_commit_rounds - before.wal_commit_rounds, 100);
    assert_eq!(after.wal_commit_writes - before.wal_commit_writes, 100);
}

/// The commit delay makes a round wait for more writers only when others are
/// already waiting; a lone writer never waits for it.
#[test]
fn the_commit_delay_never_slows_a_lone_writer() {
    let dir = TempDir::new().unwrap();
    let mut config = create_db_config();
    config.sync_config = config.sync_config.with_commit_delay_us(200_000);
    let db = Database::open(dir.path(), config).unwrap();
    let ns = db.create_namespace("ns").unwrap();
    let started = std::time::Instant::now();
    for i in 0..10u32 {
        db.put_ns(ns, format!("k{i}").as_bytes(), b"v").unwrap();
    }
    assert!(started.elapsed() < Duration::from_secs(1), "a lone writer waited for the commit delay");
}

/// Every fsync is made to cost 500 µs, as on a real disk: on a RAM-backed
/// `/tmp` an fsync costs nothing, and how many writers share a round would
/// depend on lock timing rather than on what the delay buys.
#[test]
fn the_commit_delay_groups_concurrent_writers_into_larger_rounds() {
    fn rounds_for(delay_us: u64) -> u64 {
        let dir = TempDir::new().unwrap();
        let mut config = create_db_config();
        config.sync_config = config.sync_config.with_commit_delay_us(delay_us);
        let db = Database::open(dir.path(), config).unwrap();
        db.wal.set_fsync_delay(Duration::from_micros(500));
        let ns = db.create_namespace("ns").unwrap();
        let before = db.metrics_snapshot().wal_commit_rounds;
        std::thread::scope(|s| {
            for t in 0..8u32 {
                let db = &db;
                s.spawn(move || {
                    for i in 0..40u32 {
                        db.put_ns(ns, format!("t{t}-{i}").as_bytes(), b"v").unwrap();
                    }
                });
            }
        });
        db.metrics_snapshot().wal_commit_rounds - before
    }
    let (without, with) = (rounds_for(0), rounds_for(2_000));
    assert!(with < without, "a 2 ms delay should form fewer, larger rounds ({with} vs {without})");
}

#[test]
fn a_failed_fsync_fails_the_write_and_stops_the_wal() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    let ns = db.create_namespace("ns").unwrap();
    db.put_ns(ns, b"before", b"v").unwrap();

    db.wal.fail_next_sync();
    assert!(matches!(db.put_ns(ns, b"failed", b"v"), Err(KVError::WalPoisoned(_))));
    assert_eq!(db.get_ns(ns, b"failed").unwrap(), None, "a write whose fsync failed must not be readable");
    assert!(
        matches!(db.put_ns(ns, b"after", b"v"), Err(KVError::WalPoisoned(_))),
        "later writes must be refused"
    );
    assert!(matches!(db.delete_ns(ns, b"before"), Err(KVError::WalPoisoned(_))));
    assert_eq!(db.get_ns(ns, b"before").unwrap(), Some(b"v".to_vec()));
}

/// Every writer waiting on a round whose fsync fails gets the error, and none
/// of their writes becomes readable.
#[test]
fn a_failed_round_fails_every_writer_in_it() {
    let dir = TempDir::new().unwrap();
    let db = Arc::new(Database::open(dir.path(), create_db_config()).unwrap());
    let ns = db.create_namespace("ns").unwrap();

    db.wal.freeze_commits(true);
    let writers: Vec<_> = (0..4u32)
        .map(|i| {
            let db = Arc::clone(&db);
            std::thread::spawn(move || db.put_ns(ns, format!("k{i}").as_bytes(), b"v"))
        })
        .collect();
    wait_for("four writers to reach their commit", || db.wal.frozen_writers() == 4);
    db.wal.fail_next_sync();
    db.wal.freeze_commits(false);
    for w in writers {
        assert!(
            matches!(w.join().unwrap(), Err(KVError::WalPoisoned(_))),
            "a writer in the failed round returned Ok"
        );
    }
    for i in 0..4u32 {
        assert_eq!(db.get_ns(ns, format!("k{i}").as_bytes()).unwrap(), None);
    }
}

/// A segment is fsynced as it is left. Commit rounds fsync only the active
/// segment, so without that, writes acknowledged into a segment that filled up
/// before the next round would be lost.
#[test]
fn leaving_a_segment_makes_it_durable() {
    let dir = TempDir::new().unwrap();
    let db = Database::open_with_wal_segment_size(dir.path(), create_db_config(), 512).unwrap();
    let ns = db.create_namespace("ns").unwrap();
    for i in 0..200u32 {
        db.put_ns(ns, format!("k{i:03}").as_bytes(), &[b'v'; 40]).unwrap();
    }
    assert!(db.wal.active_segment() > 5, "setup: the writes should span segments");
    let (_image_dir, image) = open_image(&db);
    let ns_image = image.get_namespace_id("ns").unwrap();
    for i in 0..200u32 {
        assert_eq!(
            image.get_ns(ns_image, format!("k{i:03}").as_bytes()).unwrap(),
            Some(vec![b'v'; 40]),
            "k{i:03} lost"
        );
    }
}

/// A frame that fills a segment up to the room kept for its end marker leaves
/// nothing for the next one, which rotates first. The rotation must fsync the
/// full segment: a commit round fsyncs only the active segment, so a write in
/// the full segment, acknowledged by a round that ran after the move, would
/// otherwise have no fsync behind it. Found by the power-loss stress below.
#[test]
fn a_segment_filled_to_its_last_byte_is_durable_when_left() {
    use crate::db::wal::segmented_log::{END_MARKER_LEN, FRAME_HEADER_LEN, split};
    const SEGMENT: u64 = 512;
    let dir = TempDir::new().unwrap();
    let db = Arc::new(Database::open_with_wal_segment_size(dir.path(), create_db_config(), SEGMENT).unwrap());
    let ns = db.create_namespace("ns").unwrap();
    // Find a value length whose frame ends exactly at the segment's end, padding
    // with small writes until one exists (frames are padded for alignment).
    let fill = loop {
        let remaining = SEGMENT - u64::from(END_MARKER_LEN) - u64::from(split(db.wal_metadata.read().tail).1);
        let fits = (0..remaining as usize).find(|&len| {
            let entry = crate::db::wal::WalEntry::new_upsert_ns(ns, b"full".to_vec(), vec![b'f'; len]).with_sequence(1);
            u64::from(FRAME_HEADER_LEN) + entry.to_bytes().unwrap().len() as u64 == remaining
        });
        match fits {
            Some(len) => break len,
            None => db.put_ns(ns, b"pad", b"p").unwrap(),
        }
    };

    db.wal.freeze_commits(true);
    let first = {
        let db = Arc::clone(&db);
        std::thread::spawn(move || db.put_ns(ns, b"full", &vec![b'f'; fill]))
    };
    wait_for("the first writer to reach its commit", || db.wal.frozen_writers() == 1);
    assert_eq!(
        u64::from(split(db.wal_metadata.read().tail).1),
        SEGMENT - u64::from(END_MARKER_LEN),
        "setup: the first frame must fill the segment"
    );
    let second = {
        let db = Arc::clone(&db);
        std::thread::spawn(move || db.put_ns(ns, b"next", b"n"))
    };
    wait_for("the second writer to reach its commit", || db.wal.frozen_writers() == 2);
    db.wal.freeze_commits(false);
    first.join().unwrap().unwrap();
    second.join().unwrap().unwrap();

    let (_image_dir, image) = open_image(&db);
    let ns_image = image.get_namespace_id("ns").unwrap();
    assert_eq!(
        image.get_ns(ns_image, b"full").unwrap(),
        Some(vec![b'f'; fill]),
        "the write that filled the segment was lost"
    );
    assert_eq!(image.get_ns(ns_image, b"next").unwrap(), Some(b"n".to_vec()));
}

/// A metadata save while a write waits for its fsync, then a power loss that
/// takes the write: the reopened log finds its tail from the log itself (the
/// metadata saves none), so a write after the reopen lands where the next
/// open reads it, and survives a second power loss.
#[test]
fn a_write_after_losing_an_unsynced_tail_survives_the_next_power_loss() {
    let dir = TempDir::new().unwrap();
    let db = Arc::new(Database::open(dir.path(), create_db_config()).unwrap());
    let ns = db.create_namespace("ns").unwrap();
    db.put_ns(ns, b"acked", b"v").unwrap();

    db.wal.freeze_commits(true);
    let writer = {
        let db = Arc::clone(&db);
        std::thread::spawn(move || db.put_ns(ns, b"in-flight", &[b'x'; 200]))
    };
    wait_for("the writer to reach its commit", || db.wal.frozen_writers() == 1);
    db.flush_wal_metadata_internal().unwrap();

    // End to end: lose the unsynced bytes, write after reopening, crash again.
    let image_dir = TempDir::new().unwrap();
    let image_path = image_dir.path().join("db");
    db.power_loss_image(&image_path).unwrap();
    db.wal.freeze_commits(false);
    writer.join().unwrap().unwrap();
    {
        let image = Database::open(&image_path, create_db_config()).unwrap();
        let ns_image = image.get_namespace_id("ns").unwrap();
        image.put_ns(ns_image, b"after-reopen", b"v").unwrap();
        image.mark_crashed();
    }
    let image = Database::open(&image_path, create_db_config()).unwrap();
    let ns_image = image.get_namespace_id("ns").unwrap();
    assert_eq!(image.get_ns(ns_image, b"acked").unwrap(), Some(b"v".to_vec()));
    assert_eq!(
        image.get_ns(ns_image, b"after-reopen").unwrap(),
        Some(b"v".to_vec()),
        "a write acknowledged after the reopen was lost behind a hole"
    );
}

// ── Preallocated segments ────────────────────────────────────────────────────

/// Every segment is created at its full size, so an fsync never has to commit
/// a new file size (design note §15).
#[test]
fn wal_segments_are_created_at_full_size() {
    const SEGMENT: u64 = 4096;
    let dir = TempDir::new().unwrap();
    let db = Database::open_with_wal_segment_size(dir.path(), create_db_config(), SEGMENT).unwrap();
    let ns = db.create_namespace("ns").unwrap();
    assert_eq!(std::fs::metadata(wal_segment_path(dir.path(), 0)).unwrap().len(), SEGMENT);
    for i in 0..200u32 {
        db.put_ns(ns, format!("k{i:03}").as_bytes(), &[b'v'; 40]).unwrap();
    }
    let last = db.wal.active_segment();
    assert!(last >= 2, "setup: the writes should span segments");
    for id in 1..=last {
        let path = wal_segment_path(dir.path(), id);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), SEGMENT, "segment {id} is not preallocated");
    }
}

/// Reopening zeroes everything past the recovered tail (so a stale frame there
/// cannot replay later) and leaves the segment at its full size.
#[test]
fn reopening_zeroes_past_the_tail_and_keeps_the_segment_preallocated() {
    use std::os::unix::fs::FileExt;
    const SEGMENT: u64 = 4096;
    let dir = TempDir::new().unwrap();
    let tail = {
        let db = Database::open_with_wal_segment_size(dir.path(), create_db_config(), SEGMENT).unwrap();
        let ns = db.create_namespace("ns").unwrap();
        db.put_ns(ns, b"k", b"v").unwrap();
        let tail = db.wal_metadata.read().tail;
        db.mark_crashed();
        tail
    };
    // Garbage past the tail, as a torn write would leave.
    let (segment, byte) = crate::db::wal::segmented_log::split(tail);
    let wal = wal_segment_path(dir.path(), segment);
    std::fs::OpenOptions::new()
        .write(true)
        .open(&wal)
        .unwrap()
        .write_at(&[0xAB; 64], u64::from(byte))
        .unwrap();

    let db = Database::open_with_wal_segment_size(dir.path(), create_db_config(), SEGMENT).unwrap();
    assert_eq!(db.wal_metadata.read().tail, tail);
    let bytes = std::fs::read(&wal).unwrap();
    assert_eq!(bytes.len() as u64, SEGMENT, "the segment lost its preallocation");
    assert!(bytes[byte as usize..].iter().all(|&b| b == 0), "bytes past the tail were not zeroed");
}

/// Measurement, not a check: one writer's `put` latency on the real disk
/// (the repository's `target/` directory), per value size, once the log has
/// moved past its first (sparse) segment onto zero-filled ones (design note
/// §17). 2 MiB segments so that happens quickly.
#[test]
#[ignore = "measurement; run explicitly with --release --ignored --nocapture"]
fn measure_one_writer_put_latency() {
    const SEGMENT: u64 = 2 << 20;
    for size in [128usize, 2048, 8192] {
        let dir = tempfile::tempdir_in(concat!(env!("CARGO_MANIFEST_DIR"), "/../target")).unwrap();
        let db = Database::open_with_wal_segment_size(dir.path(), create_db_config(), SEGMENT).unwrap();
        let ns = db.create_namespace("ns").unwrap();
        let mut i = 0u64;
        while db.wal_metadata.read().tail < SEGMENT * 2 + SEGMENT / 4 {
            db.put_ns(ns, format!("warm{i}").as_bytes(), &vec![b'v'; size]).unwrap();
            i += 1;
        }
        let n = 2000u32;
        let started = std::time::Instant::now();
        for k in 0..n {
            db.put_ns(ns, format!("k{k:05}").as_bytes(), &vec![b'v'; size]).unwrap();
        }
        let per = started.elapsed() / n;
        println!("one-writer put, {size} B: {per:?} each, {:.0}/s", 1.0 / per.as_secs_f64());
    }
}

mod flush;
mod stress;
