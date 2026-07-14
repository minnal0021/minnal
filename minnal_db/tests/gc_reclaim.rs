//! Integration tests for the segmented value log: segment rollover, per-segment GC,
//! the unlink-after-durable ordering, and monotone segment ids across restarts.

use minnal_db::{Db, DbConfig};
use std::path::{Path, PathBuf};

/// A small segment size, so a handful of values roll several segments.
const SEGMENT_SIZE: u64 = 1024 * 1024;

fn config(segment_size: u64) -> DbConfig {
    DbConfig {
        num_buckets: 1,
        segment_size_bytes: segment_size,
        ..DbConfig::default()
    }
}

/// Every value-log segment file under the db dir (their namespace path is internal).
fn segment_files(dir: &Path) -> Vec<PathBuf> {
    walkdir(dir)
        .into_iter()
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("value_log_") && n.contains(".seg"))
        })
        .collect()
}

fn walkdir(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                out.extend(walkdir(&p));
            } else {
                out.push(p);
            }
        }
    }
    out
}

/// Writes roll over into new segment files at the configured size, and every value
/// stays readable across the rollover and a reopen.
#[test]
fn writes_roll_into_new_segments_and_survive_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open_with_config(dir.path(), config(SEGMENT_SIZE)).unwrap();

    let value = vec![0x11u8; 100 * 1024];
    for i in 0..30 {
        db.put(format!("key{i:03}").as_bytes(), &value).unwrap();
    }

    // 30 × 100 KiB cannot fit in one 1 MiB segment.
    assert!(
        segment_files(dir.path()).len() > 1,
        "expected the writes to roll into several segments, found {:?}",
        segment_files(dir.path())
    );

    for i in 0..30 {
        assert_eq!(db.get(format!("key{i:03}").as_bytes()).unwrap(), Some(value.clone()));
    }
    db.shutdown().unwrap();

    let db = Db::open_with_config(dir.path(), config(SEGMENT_SIZE)).unwrap();
    for i in 0..30 {
        assert_eq!(
            db.get(format!("key{i:03}").as_bytes()).unwrap(),
            Some(value.clone()),
            "key{i:03} lost across reopen"
        );
    }
    db.shutdown().unwrap();
}

/// GC reclaims whole segments: it rewrites only the survivors of the segments it
/// selects, unlinks those files, and leaves untouched segments alone. Every live
/// value must survive.
#[test]
fn gc_reclaims_whole_segments_and_preserves_live_values() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open_with_config(dir.path(), config(SEGMENT_SIZE)).unwrap();

    // One key we keep, and one we overwrite until its old records dominate.
    let keep = vec![0x22u8; 100 * 1024];
    db.put(b"keep", &keep).unwrap();

    let churn = vec![0x33u8; 100 * 1024];
    for _ in 0..40 {
        db.put(b"hot", &churn).unwrap();
    }
    let final_value = vec![0x44u8; 100 * 1024];
    db.put(b"hot", &final_value).unwrap();

    let before = db.stats();
    let files_before = segment_files(dir.path()).len();
    assert!(before.garbage_size > 0, "expected accumulated garbage");
    assert!(db.waste_ratio() >= 30.0, "expected enough waste to be worth collecting");

    let gc = db.garbage_collect().unwrap();
    assert!(gc.bytes_reclaimed > 0, "GC should reclaim whole segments");

    let after = db.stats();
    assert!(after.garbage_size < before.garbage_size / 2, "most garbage should be gone");
    assert!(
        segment_files(dir.path()).len() < files_before,
        "GC should have unlinked segment files: {files_before} -> {}",
        segment_files(dir.path()).len()
    );
    assert!(after.disk_bytes < before.disk_bytes, "on-disk footprint should shrink");

    // Both live values survive the relocation.
    assert_eq!(db.get(b"hot").unwrap(), Some(final_value.clone()), "live value lost by GC");
    assert_eq!(db.get(b"keep").unwrap(), Some(keep.clone()), "untouched value lost by GC");

    // ...and they still resolve after a reopen, i.e. the re-pointed pointers are durable.
    db.shutdown().unwrap();
    let db = Db::open_with_config(dir.path(), config(SEGMENT_SIZE)).unwrap();
    assert_eq!(db.get(b"hot").unwrap(), Some(final_value), "relocated value lost across reopen");
    assert_eq!(db.get(b"keep").unwrap(), Some(keep), "relocated value lost across reopen");
    db.shutdown().unwrap();
}

/// GC must stay bounded across repeated cycles: the same churn, collected again and
/// again, must not grow the on-disk footprint without limit.
#[test]
fn gc_stays_bounded_across_repeated_cycles() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open_with_config(dir.path(), config(SEGMENT_SIZE)).unwrap();

    let blob = vec![0xAAu8; 100 * 1024];
    let mut disk_each_cycle = Vec::new();

    for cycle in 0u8..4 {
        for _ in 0..20 {
            db.put(b"hot", &blob).unwrap();
        }
        let final_value = vec![cycle; 100 * 1024];
        db.put(b"hot", &final_value).unwrap();

        db.garbage_collect().unwrap();
        assert_eq!(db.get(b"hot").unwrap(), Some(final_value), "cycle {cycle}: value lost");
        disk_each_cycle.push(db.stats().disk_bytes);
    }

    // The live set is one value, so the footprint must settle rather than climb.
    let first = disk_each_cycle[1];
    for (cycle, &bytes) in disk_each_cycle.iter().enumerate().skip(1) {
        assert!(
            bytes <= first * 2,
            "cycle {cycle}: disk footprint growing without bound ({disk_each_cycle:?})"
        );
    }
    db.shutdown().unwrap();
}

/// Regression: GC used to leave sub-threshold garbage on disk **forever**.
///
/// `value_log_waste_threshold` (the bucket trigger — "is this namespace worth
/// collecting?") was passed straight through as the per-*page* dirty threshold, so
/// garbage spread just under it could never be collected. The page threshold is now a
/// separate, lower knob (`page_gc_threshold`) that selects **segments**.
#[test]
fn gc_reclaims_garbage_below_the_bucket_trigger() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = config(SEGMENT_SIZE);
    assert!(
        cfg.threshold_config.page_gc_threshold < cfg.threshold_config.value_log_waste_threshold,
        "the segment threshold must sit below the bucket trigger, or sub-threshold garbage is uncollectable"
    );
    let db = Db::open_with_config(dir.path(), cfg).unwrap();

    let value = vec![0x11u8; 90 * 1024];
    let n = 60;
    for i in 0..n {
        db.put(format!("key{i:04}").as_bytes(), &value).unwrap();
    }
    // Overwrite a strided ~20%: garbage lands thinly spread across segments — enough
    // to put the bucket over its 30% trigger, but well under what a coupled 30%
    // per-segment threshold would have demanded of any single segment.
    let overwrite = vec![0x22u8; 90 * 1024];
    for i in (0..n).step_by(5) {
        db.put(format!("key{i:04}").as_bytes(), &overwrite).unwrap();
    }

    let before = db.stats();
    assert!(before.garbage_size > 0, "expected accumulated garbage");

    let gc = db.garbage_collect().unwrap();
    assert!(gc.bytes_reclaimed > 0, "GC reclaimed nothing — sub-threshold garbage is stranded again");

    let after = db.stats();
    assert!(
        after.garbage_size * 2 < before.garbage_size,
        "GC left most of the garbage behind: {} -> {} bytes",
        before.garbage_size,
        after.garbage_size
    );

    for i in 0..n {
        let expected = if i % 5 == 0 { &overwrite } else { &value };
        assert_eq!(
            db.get(format!("key{i:04}").as_bytes()).unwrap().as_ref(),
            Some(expected),
            "value for key{i:04} lost or corrupted by GC"
        );
    }
    db.shutdown().unwrap();
}

/// Segment ids are **monotone and never reused**, across restarts as well as within a
/// process. This is the landmine of the whole design: if GC unlinks the highest
/// segment and the next id were derived from `max(existing files) + 1`, that id would
/// be handed out again — and a reader holding the old pointer would silently read a
/// *different* key's record. A persisted high-water mark prevents it.
#[test]
fn segment_ids_never_repeat_across_restarts() {
    let dir = tempfile::tempdir().unwrap();
    let mut seen: Vec<u32> = Vec::new();

    for round in 0u8..4 {
        let db = Db::open_with_config(dir.path(), config(SEGMENT_SIZE)).unwrap();

        // Churn enough to roll segments and give GC something to reclaim, including
        // the newest segments — the case that would tempt id reuse.
        let blob = vec![round; 100 * 1024];
        for i in 0..15 {
            db.put(format!("k{i}").as_bytes(), &blob).unwrap();
        }
        for _ in 0..15 {
            db.put(b"hot", &blob).unwrap();
        }
        db.garbage_collect().unwrap();

        for (_, segments) in db.value_log_segment_stats("default").unwrap() {
            for s in segments {
                assert!(
                    !seen.contains(&s.id),
                    "segment id {} was reused (already seen in an earlier round); \
                     ids must come from a persisted monotone counter, never max(files)+1. seen={seen:?}",
                    s.id
                );
                seen.push(s.id);
            }
        }
        db.shutdown().unwrap();
    }

    assert!(seen.len() > 4, "expected several segments across the rounds, saw {seen:?}");
}

/// The segment size is **not** fixed at creation (unlike `num_buckets`): it is not
/// encoded in any pointer, so existing segments keep the size they were written at and
/// new ones use whatever is configured now. Reopening with a different size must work
/// and must not lose a single value.
#[test]
fn segment_size_can_change_across_restarts() {
    let dir = tempfile::tempdir().unwrap();

    let db = Db::open_with_config(dir.path(), config(SEGMENT_SIZE)).unwrap();
    let small = vec![0x55u8; 100 * 1024];
    for i in 0..12 {
        db.put(format!("a{i}").as_bytes(), &small).unwrap();
    }
    db.shutdown().unwrap();

    // Reopen with 4x the segment size — the old segments are still readable.
    let db = Db::open_with_config(dir.path(), config(SEGMENT_SIZE * 4)).unwrap();
    for i in 0..12 {
        assert_eq!(
            db.get(format!("a{i}").as_bytes()).unwrap(),
            Some(small.clone()),
            "a{i} unreadable after the segment size changed"
        );
    }
    for i in 0..12 {
        db.put(format!("b{i}").as_bytes(), &small).unwrap();
    }
    db.shutdown().unwrap();

    let db = Db::open_with_config(dir.path(), config(SEGMENT_SIZE * 4)).unwrap();
    for i in 0..12 {
        assert_eq!(db.get(format!("a{i}").as_bytes()).unwrap(), Some(small.clone()));
        assert_eq!(db.get(format!("b{i}").as_bytes()).unwrap(), Some(small.clone()));
    }
    db.shutdown().unwrap();
}

/// A deleted key must stay deleted through a GC pass. GC's re-point is a
/// compare-and-set against the LSM, so a key deleted since GC's scan is skipped rather
/// than resurrected by having its old record copied forward.
#[test]
fn gc_does_not_resurrect_deleted_keys() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open_with_config(dir.path(), config(SEGMENT_SIZE)).unwrap();

    let blob = vec![0x66u8; 100 * 1024];
    for i in 0..20 {
        db.put(format!("d{i}").as_bytes(), &blob).unwrap();
    }
    // Delete half of them, then churn so their segments become GC candidates.
    for i in (0..20).step_by(2) {
        db.delete(format!("d{i}").as_bytes()).unwrap();
    }
    for _ in 0..10 {
        db.put(b"churn", &blob).unwrap();
    }

    db.garbage_collect().unwrap();

    for i in 0..20 {
        let key = format!("d{i}");
        let got = db.get(key.as_bytes()).unwrap();
        if i % 2 == 0 {
            assert_eq!(got, None, "{key} was resurrected by GC");
        } else {
            assert_eq!(got, Some(blob.clone()), "{key} was lost by GC");
        }
    }

    db.shutdown().unwrap();
    let db = Db::open_with_config(dir.path(), config(SEGMENT_SIZE)).unwrap();
    for i in (0..20).step_by(2) {
        assert_eq!(db.get(format!("d{i}").as_bytes()).unwrap(), None, "d{i} resurrected across reopen");
    }
    db.shutdown().unwrap();
}
