//! Integration test: value-log metadata stats drive GC, and GC logically
//! reclaims garbage while preserving live values.

use minnal_db::{Db, DbConfig};
use std::path::{Path, PathBuf};

fn one_bucket_config() -> DbConfig {
    DbConfig {
        num_buckets: 1,
        ..DbConfig::default()
    }
}

const VALUE_LOG_PAGE_SIZE: u64 = 64 * 1024 * 1024;

/// Find `value_log_0.log` under the db dir (its exact namespace path is internal).
fn find_value_log(dir: &Path) -> Option<PathBuf> {
    walkdir(dir).into_iter().find(|p| p.file_name().is_some_and(|n| n == "value_log_0.log"))
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

#[test]
fn value_log_gc_triggers_on_waste_then_reclaims_metadata_and_values() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open_with_config(dir.path(), one_bucket_config()).unwrap();

    // Overwrite a single key with large values so the value log piles up garbage
    // spanning more than one 64 MiB page (each overwrite supersedes the prior
    // record, which becomes reclaimable).
    let blob = vec![0xAAu8; 8 * 1024 * 1024];
    for _ in 0..10 {
        db.put(b"hot", &blob).unwrap();
    }
    let final_value = vec![0xBBu8; 8 * 1024 * 1024];
    db.put(b"hot", &final_value).unwrap();

    // The metadata-driven waste ratio is the exact signal the background GC
    // worker triggers on — it must now exceed the default 30% threshold.
    let waste_threshold = 30.0;
    let before = db.stats();
    assert!(
        db.waste_ratio() >= waste_threshold,
        "waste ratio {} should cross the GC trigger threshold {}",
        db.waste_ratio(),
        waste_threshold
    );
    assert!(before.garbage_size > 0, "expected accumulated garbage");

    assert!(find_value_log(dir.path()).is_some(), "value_log_0.log should exist before GC");

    // Run value-log GC.
    let gc = db.garbage_collect().unwrap();
    assert!(gc.bytes_reclaimed > 0, "GC should reclaim garbage bytes");

    let after = db.stats();

    // The garbage is gone: the metadata no longer reports waste...
    assert!(after.waste_ratio < before.waste_ratio, "waste ratio should drop after GC");
    assert!(after.garbage_size < before.garbage_size / 2, "garbage should be largely reclaimed");
    assert_eq!(after.garbage_size, 0, "GC should clear metadata-tracked garbage");
    assert!(
        after.live_bytes <= final_value.len() as u64 + 1024,
        "only the final live value should remain logically live, got live_bytes={}",
        after.live_bytes
    );
    assert!(
        after.tail <= VALUE_LOG_PAGE_SIZE,
        "GC should compact the single live value back into the first logical page, got tail={}",
        after.tail
    );

    // The latest value survives the compaction.
    assert_eq!(db.get(b"hot").unwrap(), Some(final_value));

    db.shutdown().unwrap();
}

/// Regression: an overwrite-heavy bucket used to leak. GC rewrote survivors
/// *past* `tail` each cycle, abandoning the low pages as holes; the page scan
/// then stopped at the first hole, so after the first cycle GC reclaimed nothing
/// and the value log grew without bound. GC must now stay bounded across cycles.
#[test]
fn value_log_gc_stays_bounded_across_repeated_cycles() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open_with_config(dir.path(), one_bucket_config()).unwrap();
    let blob = vec![0xAAu8; 8 * 1024 * 1024];

    let mut live_each_cycle = Vec::new();
    let mut tail_each_cycle = Vec::new();
    for cycle in 0u8..3 {
        for _ in 0..10 {
            db.put(b"hot", &blob).unwrap();
        }
        let final_value = vec![cycle; 8 * 1024 * 1024];
        db.put(b"hot", &final_value).unwrap();

        let gc = db.garbage_collect().unwrap();
        assert!(gc.bytes_reclaimed > 0, "cycle {cycle}: GC must keep reclaiming garbage");
        assert_eq!(db.stats().garbage_size, 0, "cycle {cycle}: garbage not fully reclaimed");
        assert_eq!(db.get(b"hot").unwrap(), Some(final_value), "cycle {cycle}: value lost");

        let stats = db.stats();
        live_each_cycle.push(stats.live_bytes);
        tail_each_cycle.push(stats.tail);
    }

    // The logical live footprint holds steady at one surviving value across
    // cycles. Avoid asserting physical disk blocks here: sparse-hole accounting
    // differs across Linux filesystems and WSL/native Ubuntu setups.
    let expected_live = blob.len() as u64;
    for (cycle, live) in live_each_cycle.into_iter().enumerate() {
        assert!(
            live <= expected_live + 1024,
            "cycle {cycle}: logical live bytes should stay bounded near one value, got {live}"
        );
    }
    for (cycle, tail) in tail_each_cycle.into_iter().enumerate() {
        assert!(
            tail <= VALUE_LOG_PAGE_SIZE,
            "cycle {cycle}: logical tail should stay within the first page after GC, got {tail}"
        );
    }

    db.shutdown().unwrap();
}
