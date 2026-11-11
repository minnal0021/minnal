//! Writes and their bookkeeping: size limits, sequences, retries, value-log accounting and stats.

use super::*;
use crate::namespace::NamespaceId;

#[test]
fn check_total_len_rejects_over_limit() {
    // Within / at the limit → Ok; over → WriteTooLarge. Tiny limit so the
    // over-limit branch needs no multi-GiB allocation.
    assert!(Database::check_total_len(5, 5, 12, "test limit").is_ok());
    assert!(Database::check_total_len(7, 5, 12, "test limit").is_ok()); // exactly at limit
    match Database::check_total_len(8, 5, 12, "test limit") {
        Err(KVError::WriteTooLarge(m)) => assert!(m.contains("exceeds the 12-byte"), "got: {m}"),
        other => panic!("expected WriteTooLarge, got {other:?}"),
    }
}

/// A value too big for a value-log segment must be rejected **at the write
/// boundary**, not deep inside the apply.
///
/// This was the gap: `check_write_size` bounded only the `u32` field widths
/// (~4 GiB), while `ValueLog::append` additionally requires a record to fit one
/// segment (256 MiB by default). A value between the two passed the front door,
/// got a fsynced WAL entry — so `put` returned `Ok` — and then failed every
/// apply attempt, leaving an acknowledged write that `get` could not see and
/// that WAL replay could only turn into a fail log.
///
/// Asserting the error is only half of it; the load-bearing half is that the WAL
/// did not grow, because that is what makes the rejection mean "nothing
/// happened".
#[test]
fn a_value_too_big_for_a_segment_is_rejected_before_the_wal() {
    const SEG: u64 = 64 * 1024; // the minimum legal segment size

    let dir = TempDir::new().unwrap();
    let mut config = create_db_config();
    config.segment_size_bytes = SEG;
    let db = Database::open(dir.path(), config).unwrap();

    db.put(b"k", b"small").unwrap();
    let entries_before = db.wal_metadata.read().tail;

    // Comfortably over the segment, but far under the u32 field-width limit that
    // was the only bound before.
    let oversized = vec![0u8; (SEG as usize) * 2];
    match db.put(b"k", &oversized) {
        Err(KVError::WriteTooLarge(m)) => {
            assert!(
                m.contains("value-log segment"),
                "the message should name the bound that was hit; got: {m}"
            );
        }
        other => panic!("expected WriteTooLarge, got {other:?}"),
    }

    assert_eq!(
        db.wal_metadata.read().tail,
        entries_before,
        "the rejected write reached the WAL — it was not rejected at the boundary"
    );
    assert_eq!(db.get(b"k").unwrap(), Some(b"small".to_vec()), "the prior value must be untouched");

    // A record that fits still writes.
    let fits = vec![7u8; (SEG as usize) - 1024];
    db.put(b"k", &fits).unwrap();
    assert_eq!(db.get(b"k").unwrap(), Some(fits));

    db.shutdown().unwrap();
}

#[test]
fn put_and_delete_accept_normal_sized_keys_and_values() {
    // Regression: the size guard must not reject ordinary writes.
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    db.put(b"k", &vec![0u8; 4096]).unwrap();
    assert_eq!(db.get(b"k").unwrap(), Some(vec![0u8; 4096]));
    db.delete(b"k").unwrap();
    assert_eq!(db.get(b"k").unwrap(), None);
    db.shutdown().unwrap();
}

#[test]
fn test_ops_metrics_counters_move() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();

    db.put(b"k1", b"v1").unwrap();
    db.put(b"k2", b"v2").unwrap();
    assert_eq!(db.get(b"k1").unwrap(), Some(b"v1".to_vec())); // hit
    assert_eq!(db.get(b"missing").unwrap(), None); // miss
    db.delete(b"k2").unwrap();

    let m = db.metrics_snapshot();
    assert_eq!(m.puts, 2, "two WAL-backed puts");
    assert_eq!(m.deletes, 1, "one delete");
    assert_eq!(m.reads, 2, "two user reads");
    assert_eq!(m.read_hits, 1, "one hit");
    assert_eq!(m.read_misses, 1, "one miss");
    // Every WAL-backed write fsyncs once (2 puts + 1 delete).
    assert_eq!(m.wal_writes, 3);
    assert!(m.wal_bytes_appended > 0);
    // Reads go through the LSM point-lookup path.
    assert!(m.lookups >= 2, "lookups should cover the user reads, got {}", m.lookups);

    db.shutdown().unwrap();
}

#[test]
fn test_value_log_physical_and_segment_stats() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();

    for i in 0..50u32 {
        db.put(format!("k{i}").as_bytes(), &vec![b'v'; 256]).unwrap();
    }

    // Physical stats: present for the default namespace, with real on-disk bytes.
    let physical = db.value_log_physical_stats();
    let (_, shards) = physical.iter().find(|(ns, _)| ns == "default").expect("default ns present");
    assert!(!shards.is_empty(), "expected value-log shards");
    let total_physical: u64 = shards.iter().map(|s| s.physical_bytes).sum();
    assert!(total_physical > 0, "physical bytes should be non-zero after writes");

    // Segment stats: per-bucket segment breakdown, with live bytes accounted.
    let segments = db.value_log_segment_stats("default").unwrap();
    assert_eq!(segments.len(), db.config.num_buckets, "one entry per bucket");
    let live: u64 = segments.iter().flat_map(|(_, ss)| ss.iter()).map(|s| s.live_bytes).sum();
    assert!(live > 0, "written records should be accounted as live bytes, got {live}");
    // Every bucket has exactly one active tail (unsealed) segment.
    for (bucket, ss) in &segments {
        let unsealed = ss.iter().filter(|s| !s.sealed).count();
        assert_eq!(unsealed, 1, "bucket {bucket} must have exactly one active tail segment");
    }

    // Unknown namespace errors rather than panicking.
    assert!(db.value_log_segment_stats("nope").is_err());

    db.shutdown().unwrap();
}

#[test]
fn concurrent_same_key_puts_keep_value_log_accounting_exact() {
    // Under real concurrency, racing writers to the same keys must not drift the
    // value-log accounting GC selection relies on: total live bytes must equal
    // exactly one live record per distinct key. (The deterministic form of the
    // underlying bug is `a_lower_seq_put_charges_its_own_record_not_the_live_one`
    // in kv_store; this guards the invariant end-to-end under load.)
    use std::sync::Arc;
    use std::thread;

    let dir = TempDir::new().unwrap();
    let db = Arc::new(Database::open(dir.path(), create_db_config()).unwrap());

    let keys: Vec<Vec<u8>> = (0..4u32).map(|k| format!("hot-key-{k}").into_bytes()).collect();
    let mut handles = Vec::new();
    for t in 0..4usize {
        let (db, keys) = (Arc::clone(&db), keys.clone());
        handles.push(thread::spawn(move || {
            for i in 0..100usize {
                // Vary the value size per write so a mis-charged displacement can't
                // cancel in the aggregate byte count.
                let len = 64 + (t * 37 + i * 101) % 1024;
                db.put(&keys[(t + i) % keys.len()], &vec![0xACu8; len]).unwrap();
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }

    // Expected live bytes: one record per distinct key, sized by that key's current
    // (winning) value. Database::open starts no GC worker, so nothing reclaims
    // concurrently and the live set is exactly the winning records.
    let expected_live: u64 = keys
        .iter()
        .map(|k| {
            let v = db.get(k).unwrap().expect("key present after the race");
            crate::storage::value_log::ValueRecordHeader::record_len(k.len(), v.len())
        })
        .sum();

    let stats = db.value_log_segment_stats("default").unwrap();
    let mut total_live = 0u64;
    for (_bucket, segs) in &stats {
        for s in segs {
            assert_eq!(
                s.live_bytes + s.garbage_bytes,
                s.total_bytes,
                "segment {} broke live+garbage==total",
                s.id
            );
            total_live += s.live_bytes;
        }
    }
    assert_eq!(
        total_live, expected_live,
        "accounted live bytes must equal exactly one record per live key, not leak intermediate records"
    );

    db.shutdown().unwrap();
}

#[test]
fn existing_bucket_count_is_detected_from_segmented_value_logs() {
    // num_buckets is locked at creation: reopening with a different configured
    // value must NOT re-shard. detect_bucket_count reads the count back from the
    // value-log filenames — which are now `value_log_{bucket}.segNNNNNN`, not the
    // old `value_log_{bucket}.log`. A stale matcher would see zero buckets and
    // silently honour the new config, corrupting key→bucket sharding.
    let dir = TempDir::new().unwrap();
    let original_buckets = 3;

    {
        let mut config = create_db_config();
        config.num_buckets = original_buckets;
        let db = Database::open(dir.path(), config).unwrap();
        for i in 0..30u32 {
            db.put(format!("k{i}").as_bytes(), format!("v{i}").as_bytes()).unwrap();
        }
        assert_eq!(db.config.num_buckets, original_buckets);
        db.shutdown().unwrap();
    }

    // Reopen asking for a DIFFERENT bucket count.
    let mut config = create_db_config();
    config.num_buckets = original_buckets + 3;
    let db = Database::open(dir.path(), config).unwrap();
    assert_eq!(
        db.config.num_buckets, original_buckets,
        "the original bucket count must be detected from the segment files and preserved, not the newly configured one"
    );

    // Every value is still readable — keys resolved to the same buckets they were
    // written to, which they only can if the original count was honoured.
    for i in 0..30u32 {
        assert_eq!(
            db.get(format!("k{i}").as_bytes()).unwrap(),
            Some(format!("v{i}").into_bytes()),
            "value k{i} unreadable after reopen with a mismatched num_buckets"
        );
    }
    db.shutdown().unwrap();
}

#[test]
fn gc_collects_a_hot_bucket_when_the_namespace_average_is_below_the_trigger() {
    use crate::db::workers::value_log_gc::ValueLogGcTarget;
    // The GC trigger is per-bucket, not just the namespace average: a bucket over the
    // trigger must be collected even when near-empty buckets drag the average below it.
    let dir = TempDir::new().unwrap();
    let mut config = create_db_config();
    config.threshold_config.value_log_waste_threshold = 30.0; // tail bar tracks this
    let db = Database::open(dir.path(), config).unwrap();

    let value = vec![0x33u8; 300];
    // Live data spread across buckets keeps the namespace average low...
    for i in 0..800u32 {
        db.put(format!("k{i:04}").as_bytes(), &value).unwrap();
    }
    // ...then hammer a single key so ITS bucket alone piles up garbage.
    for _ in 0..250u32 {
        db.put(b"k0000", &value).unwrap();
    }

    let store = db.get_store(DEFAULT_NAMESPACE_ID).unwrap();
    let avg = store.get_waste_ratio();
    assert!(avg < 30.0, "precondition: namespace average must be below the trigger, got {avg:.1}%");
    assert!(store.has_bucket_over_waste(30.0), "precondition: some bucket must be over the trigger");

    // The per-bucket-aware trigger runs GC where the old average-only gate would have
    // skipped the whole namespace.
    ValueLogGcTarget::run_gc_if_needed(&db, 30.0);

    let after = store.get_waste_ratio();
    assert!(
        after < avg,
        "the hot bucket's garbage must be reclaimed, dropping overall waste (was {avg:.1}%, now {after:.1}%)"
    );
    assert_eq!(db.get(b"k0000").unwrap(), Some(value.clone()), "the hammered key must survive collection");

    db.shutdown().unwrap();
}

/// Item 3: `apply_with_retry` rides out transient apply failures and stops
/// as soon as one attempt succeeds.
#[test]
fn test_apply_with_retry_succeeds_after_transient_failure() {
    use std::cell::Cell;

    let attempts = Cell::new(0);
    Database::apply_with_retry("put", NamespaceId::from_u64(0), b"k", 1, || {
        attempts.set(attempts.get() + 1);
        // Fail on the first attempt, succeed on the second.
        if attempts.get() < 2 { Err(KVError::KeyNotFound) } else { Ok(()) }
    });
    assert_eq!(attempts.get(), 2, "should stop retrying once an attempt succeeds");
}

/// Item 3: `apply_with_retry` gives up after a bounded number of attempts
/// (it never surfaces the error — the write is durable in the WAL).
#[test]
fn test_apply_with_retry_gives_up_after_max_attempts() {
    use std::cell::Cell;

    let attempts = Cell::new(0);
    Database::apply_with_retry("put", NamespaceId::from_u64(0), b"k", 1, || {
        attempts.set(attempts.get() + 1);
        Err(KVError::KeyNotFound)
    });
    assert_eq!(
        attempts.get(),
        Database::APPLY_RETRY_ATTEMPTS,
        "should attempt exactly APPLY_RETRY_ATTEMPTS times before giving up"
    );
}

#[test]
fn test_sequence_increments_on_writes() -> Result<()> {
    use std::sync::atomic::Ordering;
    let temp_dir = TempDir::new()?;
    let db = Database::open(temp_dir.path(), create_db_config())?;

    let seq_before = db.next_seq.load(Ordering::Relaxed);
    db.put(b"a", b"1")?;
    db.put(b"b", b"2")?;
    db.delete(b"a")?;
    let seq_after = db.next_seq.load(Ordering::Relaxed);

    assert_eq!(seq_after - seq_before, 3);
    Ok(())
}

#[test]
fn test_sequence_stamps_wal_entries() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db = Database::open(temp_dir.path(), create_db_config())?;

    db.put(b"x", b"v1")?;
    db.put(b"y", b"v2")?;

    let wal_metadata = db.wal_metadata.read();
    let entries = db.wal.scan_entries(0, wal_metadata.tail)?;
    drop(wal_metadata);

    let seqs: Vec<u64> = entries.iter().map(|(_, e)| e.sequence).collect();
    assert!(!seqs.is_empty());
    for seq in &seqs {
        assert!(*seq > 0, "sequence must be non-zero");
    }
    for w in seqs.windows(2) {
        assert!(w[1] > w[0], "sequences must be strictly increasing");
    }
    Ok(())
}

#[test]
fn test_sequence_recovered_after_reopen() -> Result<()> {
    use std::sync::atomic::Ordering;
    let temp_dir = TempDir::new()?;

    let seq_at_close = {
        let db = Database::open(temp_dir.path(), create_db_config())?;
        db.put(b"k1", b"v1")?;
        db.put(b"k2", b"v2")?;
        db.next_seq.load(Ordering::Relaxed)
    };

    let db2 = Database::open(temp_dir.path(), create_db_config())?;
    let seq_after_reopen = db2.next_seq.load(Ordering::Relaxed);
    assert!(
        seq_after_reopen >= seq_at_close,
        "recovered seq ({}) must be >= seq at close ({})",
        seq_after_reopen,
        seq_at_close
    );

    let seq_before_new_write = db2.next_seq.load(Ordering::Relaxed);
    db2.put(b"k3", b"v3")?;
    let wal_metadata = db2.wal_metadata.read();
    let entries = db2.wal.scan_entries(0, wal_metadata.tail)?;
    drop(wal_metadata);
    let max_old_seq = seq_before_new_write - 1;
    let new_entry_seq = entries.last().map(|(_, e)| e.sequence).unwrap_or(0);
    assert!(
        new_entry_seq > max_old_seq,
        "new write seq ({}) must exceed previous max ({})",
        new_entry_seq,
        max_old_seq
    );
    Ok(())
}

/// An entry too large for a WAL segment is refused as `WriteTooLarge` before
/// anything is written, and the database keeps working.
#[test]
fn a_write_too_large_for_a_wal_segment_is_refused() {
    let dir = TempDir::new().unwrap();
    let db = Database::open_with_wal_segment_size(dir.path(), create_db_config(), 4096).unwrap();
    let tail = db.wal_metadata.read().tail;
    assert!(matches!(db.put(b"big", &vec![b'x'; 8192]), Err(KVError::WriteTooLarge(_))));
    assert_eq!(db.wal_metadata.read().tail, tail, "a refused write must not grow the WAL");
    assert_eq!(db.get(b"big").unwrap(), None);
    db.put(b"small", b"v").unwrap();
    assert_eq!(db.get(b"small").unwrap(), Some(b"v".to_vec()));
}
