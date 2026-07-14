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

/// A non-default page size, to prove `page_size_bytes` actually drives the layout.
const CUSTOM_PAGE_SIZE: u64 = 1024 * 1024;

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

/// `page_size_bytes` must actually reach the value log — it was parsed into
/// `DbConfig` and then ignored, so pages were always 64 MiB no matter the config.
/// Drive a configured page size end to end: writes must roll over at that size,
/// GC must rewrite survivors into pages of that size, and every value must
/// survive the compaction.
#[test]
fn configured_page_size_drives_layout_and_survives_gc() {
    let dir = tempfile::tempdir().unwrap();
    let config = DbConfig {
        num_buckets: 1,
        page_size_bytes: CUSTOM_PAGE_SIZE,
        ..DbConfig::default()
    };
    let db = Db::open_with_config(dir.path(), config).unwrap();

    // Values large enough that a handful of them cannot share one 1 MiB page.
    let blob = |b: u8| vec![b; 300 * 1024];

    // Keep one key live; overwrite another repeatedly to pile up garbage.
    db.put(b"keep", &blob(0x11)).unwrap();
    for _ in 0..12 {
        db.put(b"hot", &blob(0x22)).unwrap();
    }
    let final_value = blob(0x33);
    db.put(b"hot", &final_value).unwrap();

    // Pages are laid out at the CONFIGURED size, not the 64 MiB default — with a
    // 64 MiB page all of this would still be sitting in page 0.
    let pages = db.value_log_page_stats("default").unwrap();
    let offsets: Vec<u64> = pages.iter().flat_map(|(_, ps)| ps.iter().map(|p| p.page_offset)).collect();
    assert!(
        offsets.iter().any(|&o| o > 0),
        "expected writes to roll onto later pages at a {CUSTOM_PAGE_SIZE}-byte page size, got offsets {offsets:?}"
    );
    for o in &offsets {
        assert!(
            o.is_multiple_of(CUSTOM_PAGE_SIZE),
            "page offset {o} is not aligned to the configured page size {CUSTOM_PAGE_SIZE}"
        );
    }

    // GC must rewrite survivors using the same page size (its replacement file is
    // opened with the bucket's size — a mismatch here would corrupt every pointer).
    assert!(db.waste_ratio() >= 30.0, "expected enough garbage to trigger GC");
    let gc = db.garbage_collect().unwrap();
    assert!(gc.bytes_reclaimed > 0, "GC should reclaim garbage bytes");

    for (_, ps) in db.value_log_page_stats("default").unwrap() {
        for p in ps {
            assert!(
                p.page_offset.is_multiple_of(CUSTOM_PAGE_SIZE),
                "GC produced a page at {} — not aligned to the configured page size",
                p.page_offset
            );
        }
    }

    // Both values survive the compaction and still resolve through their pointers.
    assert_eq!(db.get(b"hot").unwrap(), Some(final_value.clone()), "live value lost across GC");
    assert_eq!(db.get(b"keep").unwrap(), Some(blob(0x11)), "untouched value lost across GC");

    db.shutdown().unwrap();

    // And the pointers still resolve after a reopen at the same page size.
    let config = DbConfig {
        num_buckets: 1,
        page_size_bytes: CUSTOM_PAGE_SIZE,
        ..DbConfig::default()
    };
    let db = Db::open_with_config(dir.path(), config).unwrap();
    assert_eq!(db.get(b"hot").unwrap(), Some(final_value), "value lost across reopen");
    assert_eq!(db.get(b"keep").unwrap(), Some(blob(0x11)), "value lost across reopen");
    db.shutdown().unwrap();
}

/// The page size is fixed at creation: reopening a database whose value log was
/// written with a different page size must fail loudly rather than reinterpret
/// every stored pointer (`page_offset` is page-aligned and a record's slot entry
/// sits at `page_size - segment_id * 8`).
#[test]
fn reopening_with_a_different_page_size_is_refused() {
    let dir = tempfile::tempdir().unwrap();

    let db = Db::open_with_config(
        dir.path(),
        DbConfig {
            num_buckets: 1,
            page_size_bytes: CUSTOM_PAGE_SIZE,
            ..DbConfig::default()
        },
    )
    .unwrap();
    db.put(b"k", b"v").unwrap();
    db.shutdown().unwrap();

    let Err(err) = Db::open_with_config(
        dir.path(),
        DbConfig {
            num_buckets: 1,
            page_size_bytes: CUSTOM_PAGE_SIZE * 2,
            ..DbConfig::default()
        },
    ) else {
        panic!("reopening with a different page size must fail");
    };
    let msg = err.to_string();
    assert!(msg.contains("page size"), "error should name the page-size mismatch, got: {msg}");

    // The original database is untouched and still opens at its own page size.
    let db = Db::open_with_config(
        dir.path(),
        DbConfig {
            num_buckets: 1,
            page_size_bytes: CUSTOM_PAGE_SIZE,
            ..DbConfig::default()
        },
    )
    .unwrap();
    assert_eq!(db.get(b"k").unwrap(), Some(b"v".to_vec()));
    db.shutdown().unwrap();
}

/// Regression: GC used to leave sub-threshold garbage on disk **forever**.
///
/// `value_log_waste_threshold` (the bucket trigger — "is this namespace worth
/// collecting?") was passed straight through as the *per-page* dirty threshold
/// ("is this page worth rewriting?"). Because they were the same number, garbage
/// spread just *under* it could never be collected: those pages read as "clean",
/// so GC copied them byte-for-byte into the compacted file with their garbage
/// intact, reported success, and left the bucket still over its trigger — so the
/// next pass copied everything again and still reclaimed nothing. A treadmill.
///
/// The page threshold is now a separate, lower knob (`page_gc_threshold`, 10% by
/// default), so garbage below the bucket trigger is actually reclaimed.
#[test]
fn gc_reclaims_garbage_below_the_bucket_trigger() {
    let dir = tempfile::tempdir().unwrap();
    // 1 MiB pages: small enough that ~90 KiB values spread garbage across many
    // pages, which is what creates the sub-threshold shape.
    let config = DbConfig {
        num_buckets: 1,
        page_size_bytes: CUSTOM_PAGE_SIZE,
        ..DbConfig::default()
    };
    // The bucket trigger stays at its default 30%; the page threshold is 10%.
    assert!(
        config.threshold_config.page_gc_threshold < config.threshold_config.value_log_waste_threshold,
        "the page threshold must sit below the bucket trigger, or sub-threshold garbage is uncollectable"
    );
    let db = Db::open_with_config(dir.path(), config).unwrap();

    let value = vec![0x11u8; 90 * 1024];
    let n = 200;
    for i in 0..n {
        db.put(format!("key{i:04}").as_bytes(), &value).unwrap();
    }

    // Overwrite a strided ~20% of the keys: garbage lands spread thinly across
    // pages — enough to put the BUCKET over its 30% trigger once counted, but well
    // under what the old coupled 30% *page* threshold demanded of any single page.
    let overwrite = vec![0x22u8; 90 * 1024];
    for i in (0..n).step_by(5) {
        db.put(format!("key{i:04}").as_bytes(), &overwrite).unwrap();
    }

    let before = db.stats();
    assert!(before.garbage_size > 0, "expected accumulated garbage");

    let gc = db.garbage_collect().unwrap();
    let after = db.stats();

    assert!(gc.bytes_reclaimed > 0, "GC reclaimed nothing — sub-threshold garbage is stranded again");
    assert!(
        after.garbage_size * 4 < before.garbage_size,
        "GC left most of the garbage behind: {} -> {} bytes. The page threshold is probably coupled \
         to the bucket trigger again.",
        before.garbage_size,
        after.garbage_size
    );

    // ...and every value still reads back correctly after the compaction.
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
