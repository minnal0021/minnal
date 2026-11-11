//! Registering and activating field indexes, their WAL replay, write buffers and row map.

use super::*;

/// Regression: activating the same field index from several threads must not
/// kill the process.
///
/// `activate_field_index` creates memory-mapped files — the field's `BlobStore`
/// pair, its `keymap/` pair, and the namespace's `RowMap` — through a
/// check-then-act (`if exists { open } else { create }`), and the create path
/// opens with `truncate(true)`. Two threads racing it truncate a file the other
/// has already mapped, so the next access to that mapping faults with
/// **SIGBUS**: the process dies outright, with no unwinding and no `Result` for
/// anyone to handle.
///
/// Reachable from an ordinary client: two concurrent `POST /stores` for one
/// namespace (a timeout plus a retry is enough) each activate the same fields.
/// Measured before the fix, against the stress server: 8 racing creates killed
/// it on **every** attempt — `Bus error (core dumped)`, exit 135 — and a
/// different interleaving surfaced instead as
/// `range end index 8 out of range for slice of length 0` writing the 64-byte
/// header into a zero-length map.
///
/// **Reverting the fix does not make this test fail — it makes the test binary
/// die.** That is the honest reproduction; there is nothing to assert against a
/// signal. What the assertions below add is that serialising activation did not
/// break it: every thread returns `Ok`, and the index answers correctly
/// afterwards.
#[test]
fn test_concurrent_activation_of_one_field_index_does_not_kill_the_process() {
    use std::sync::Barrier;

    let dir = TempDir::new().unwrap();
    let db = Arc::new(Database::open(dir.path(), create_db_config()).unwrap());

    const THREADS: usize = 8;
    let mut failures: Vec<String> = Vec::new();

    for round in 0..10 {
        let name = format!("act_ns_{round}");
        let ns = db.create_namespace(&name).unwrap();
        // Registered once, up front: the race under test is the *activation*,
        // not the registration.
        let field_id = db.register_index_field(ns, "status", IndexValueType::Str).unwrap();

        let barrier = Arc::new(Barrier::new(THREADS));
        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let db = Arc::clone(&db);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let extractor: ExtractorFn = Arc::new(|bytes: &[u8]| {
                        let s = std::str::from_utf8(bytes).ok()?;
                        let v: serde_json::Value = serde_json::from_str(s).ok()?;
                        Some(IndexValue::Str(v["status"].as_str()?.to_string()))
                    });
                    barrier.wait();
                    db.activate_field_index(ns, field_id, IndexValueType::Str, extractor)
                })
            })
            .collect();
        for handle in handles {
            if let Err(e) = handle.join().unwrap() {
                failures.push(e.to_string());
            }
        }

        // The surviving index must be usable, not merely present: a store that
        // came through the race with a truncated key file would open with a
        // zeroed header and silently answer nothing.
        db.put_ns(ns, b"d1", br#"{"status":"active"}"#).unwrap();
        db.put_ns(ns, b"d2", br#"{"status":"retired"}"#).unwrap();
        let hits = db.query_keys(ns, r#"status = "active""#).unwrap();
        assert_eq!(
            hits.keys,
            vec![b"d1".to_vec()],
            "round {round}: the field index survived activation but does not answer correctly"
        );
    }

    assert!(
        failures.is_empty(),
        "{} of {} concurrent activations failed: {:?}",
        failures.len(),
        THREADS * 10,
        failures.iter().collect::<std::collections::BTreeSet<_>>()
    );
}

#[test]
fn test_activate_type_mismatch_rejected() {
    use crate::field_index::IndexValueType;
    use crate::kv_store::index_set::ExtractorFn;
    use std::sync::Arc;

    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();

    let ns = DEFAULT_NAMESPACE_ID;
    // Register as Int
    let field_id = db.register_index_field(ns, "age", IndexValueType::Int).unwrap();

    // Activating with the correct type succeeds
    let extractor: ExtractorFn = Arc::new(|_| None);
    assert!(db.activate_field_index(ns, field_id, IndexValueType::Int, Arc::clone(&extractor)).is_ok());

    // Activating with a different type must fail
    let err = db.activate_field_index(ns, field_id, IndexValueType::Str, extractor).unwrap_err();
    assert!(err.to_string().contains("Type mismatch"), "unexpected error: {}", err);
}

#[test]
fn test_activate_unknown_field_id_rejected() {
    use crate::field_index::IndexValueType;
    use crate::kv_store::index_set::ExtractorFn;
    use std::sync::Arc;

    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();

    let extractor: ExtractorFn = Arc::new(|_| None);
    // field_id 99 was never registered
    let err = db
        .activate_field_index(DEFAULT_NAMESPACE_ID, 99, IndexValueType::Int, extractor)
        .unwrap_err();
    assert!(err.to_string().contains("not registered"), "unexpected error: {}", err);
}

#[test]
fn test_duplicate_register_field_idempotent() {
    use crate::field_index::IndexValueType;

    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();

    let id1 = db.register_index_field(DEFAULT_NAMESPACE_ID, "status", IndexValueType::Str).unwrap();
    // Same name + same type: idempotent, returns existing id
    let id2 = db.register_index_field(DEFAULT_NAMESPACE_ID, "status", IndexValueType::Str).unwrap();
    assert_eq!(id1, id2);
    // Same name, different type: error
    let err = db.register_index_field(DEFAULT_NAMESPACE_ID, "status", IndexValueType::Int).unwrap_err();
    assert!(err.to_string().contains("already registered"), "unexpected error: {}", err);
}

#[test]
fn test_schema_survives_restart_and_index_activates_without_re_register() {
    use crate::field_index::{IndexValue, IndexValueType};
    use crate::kv_store::index_set::ExtractorFn;
    use std::sync::Arc;

    let dir = TempDir::new().unwrap();

    // ── First open: register fields and write some data ──────────────
    let status_field_id;
    let age_field_id;
    {
        let db = Database::open(dir.path(), create_db_config()).unwrap();

        status_field_id = db.register_index_field(DEFAULT_NAMESPACE_ID, "status", IndexValueType::Str).unwrap();
        age_field_id = db.register_index_field(DEFAULT_NAMESPACE_ID, "age", IndexValueType::Int).unwrap();

        // Activate both indices with extractors
        let status_extractor: ExtractorFn = Arc::new(|bytes: &[u8]| {
            let s = std::str::from_utf8(bytes).ok()?;
            let v: serde_json::Value = serde_json::from_str(s).ok()?;
            Some(IndexValue::Str(v["status"].as_str()?.to_string()))
        });
        let age_extractor: ExtractorFn = Arc::new(|bytes: &[u8]| {
            let s = std::str::from_utf8(bytes).ok()?;
            let v: serde_json::Value = serde_json::from_str(s).ok()?;
            Some(IndexValue::Int(v["age"].as_i64()?))
        });
        db.activate_field_index(DEFAULT_NAMESPACE_ID, status_field_id, IndexValueType::Str, status_extractor)
            .unwrap();
        db.activate_field_index(DEFAULT_NAMESPACE_ID, age_field_id, IndexValueType::Int, age_extractor)
            .unwrap();

        db.put(b"user:1", br#"{"status":"active","age":30}"#).unwrap();
        db.put(b"user:2", br#"{"status":"inactive","age":25}"#).unwrap();
        db.put(b"user:3", br#"{"status":"active","age":40}"#).unwrap();

        db.shutdown().unwrap();
    }

    // ── Second open: do NOT call register_index_field ────────────────
    // Schema must be loaded from the system catalog automatically.
    {
        let db = Database::open(dir.path(), create_db_config()).unwrap();

        // Confirm schema is present without re-registering
        let fields = db.list_index_fields(DEFAULT_NAMESPACE_ID);
        assert_eq!(fields.len(), 2);
        assert!(fields.iter().any(|f| f.field_name == "status" && f.field_type == IndexValueType::Str));
        assert!(fields.iter().any(|f| f.field_name == "age" && f.field_type == IndexValueType::Int));

        // Activate indices with extractors (closures can't be persisted — caller always supplies these)
        let status_extractor: ExtractorFn = Arc::new(|bytes: &[u8]| {
            let s = std::str::from_utf8(bytes).ok()?;
            let v: serde_json::Value = serde_json::from_str(s).ok()?;
            Some(IndexValue::Str(v["status"].as_str()?.to_string()))
        });
        let age_extractor: ExtractorFn = Arc::new(|bytes: &[u8]| {
            let s = std::str::from_utf8(bytes).ok()?;
            let v: serde_json::Value = serde_json::from_str(s).ok()?;
            Some(IndexValue::Int(v["age"].as_i64()?))
        });
        // These must succeed using the field IDs recovered from the catalog
        db.activate_field_index(DEFAULT_NAMESPACE_ID, status_field_id, IndexValueType::Str, status_extractor)
            .unwrap();
        db.activate_field_index(DEFAULT_NAMESPACE_ID, age_field_id, IndexValueType::Int, age_extractor)
            .unwrap();

        // Queries must return correct results from the warmed index
        let mut active_keys = db.query_keys(DEFAULT_NAMESPACE_ID, "status = \"active\"").unwrap().keys;
        active_keys.sort();
        assert_eq!(active_keys, vec![b"user:1".to_vec(), b"user:3".to_vec()]);

        let inactive_keys = db.query_keys(DEFAULT_NAMESPACE_ID, "status = \"inactive\"").unwrap().keys;
        assert_eq!(inactive_keys, vec![b"user:2".to_vec()]);

        db.shutdown().unwrap();
    }
}

/// Measurement — the decisive one for "just shrink the checkpoint interval".
///
/// Holds the replay WINDOW fixed and varies the TOTAL document count. If the
/// per-key replay cost is flat, shrinking the interval is a lasting fix. If it
/// grows with total data, shrinking the interval buys a one-off factor and the
/// problem returns as the store grows.
///
/// `cargo test -p minnal_db --release --lib replay_cost_vs_total -- --ignored --nocapture`
#[test]
#[ignore = "measurement; run explicitly with --release --ignored --nocapture"]
fn replay_cost_vs_total() {
    const WINDOW: u32 = 750;
    println!("fixed replay window={WINDOW} keys, 2 distinct values");
    for total in [3_000u32, 6_000, 12_000] {
        let dir = TempDir::new().unwrap();
        let ns = DEFAULT_NAMESPACE_ID;
        let config = create_db_config();
        let field_id = {
            let db = Database::open(dir.path(), config.clone()).unwrap();
            let field_id = activate_status_index(&db, ns);
            for i in 0..total {
                let v = if i % 2 == 0 { "active" } else { "inactive" };
                db.put(format!("doc:{i:06}").as_bytes(), format!(r#"{{"status":"{v}"}}"#).as_bytes())
                    .unwrap();
            }
            db.run_index_checkpoint().unwrap();
            for i in 0..WINDOW {
                let v = if i % 2 == 0 { "active" } else { "inactive" };
                db.put(format!("doc:{i:06}").as_bytes(), format!(r#"{{"status":"{v}"}}"#).as_bytes())
                    .unwrap();
            }
            db.crash();
            field_id
        };
        let db = Database::open(dir.path(), config).unwrap();
        let extractor: crate::kv_store::index_set::ExtractorFn = std::sync::Arc::new(|bytes: &[u8]| {
            let s = std::str::from_utf8(bytes).ok()?;
            let v: serde_json::Value = serde_json::from_str(s).ok()?;
            Some(crate::field_index::IndexValue::Str(v["status"].as_str()?.to_string()))
        });
        let t = std::time::Instant::now();
        db.activate_field_index(ns, field_id, crate::field_index::IndexValueType::Str, extractor)
            .unwrap();
        let dt = t.elapsed();
        println!(
            "  total={total:6} docs -> replay {:>8.1?} ({:.3} ms/key)",
            dt,
            dt.as_secs_f64() * 1000.0 / WINDOW as f64
        );
        db.shutdown().unwrap();
    }
}

/// Measurement, not a correctness check — run explicitly:
/// `cargo test -p minnal_db --release --lib replay_cost_vs_window -- --ignored --nocapture`
///
/// Answers: does shrinking the checkpoint interval (i.e. the replay window)
/// reduce restart time proportionally? Holds the total document count fixed and
/// varies only how many of those keys fall inside the replay window.
#[test]
#[ignore = "measurement; run explicitly with --release --ignored --nocapture"]
fn replay_cost_vs_window() {
    const TOTAL: u32 = 6_000;
    println!("total docs={TOTAL}, 2 distinct values");
    for window in [750u32, 1_500, 3_000, 6_000] {
        let dir = TempDir::new().unwrap();
        let ns = DEFAULT_NAMESPACE_ID;
        let config = create_db_config();
        let field_id = {
            let db = Database::open(dir.path(), config.clone()).unwrap();
            let field_id = activate_status_index(&db, ns);
            for i in 0..TOTAL {
                let v = if i % 2 == 0 { "active" } else { "inactive" };
                db.put(format!("doc:{i:06}").as_bytes(), format!(r#"{{"status":"{v}"}}"#).as_bytes())
                    .unwrap();
            }
            db.run_index_checkpoint().unwrap();
            // Re-write `window` keys AFTER the checkpoint: those are exactly the
            // keys the next replay will have to process.
            for i in 0..window {
                let v = if i % 2 == 0 { "active" } else { "inactive" };
                db.put(format!("doc:{i:06}").as_bytes(), format!(r#"{{"status":"{v}"}}"#).as_bytes())
                    .unwrap();
            }
            // Simulate a CRASH. `shutdown()` runs a final checkpoint, which
            // would advance the marker past everything just written and leave
            // the replay window empty — measuring nothing. (It did, on the
            // first attempt at this measurement.)
            db.crash();
            field_id
        };

        let db = Database::open(dir.path(), config).unwrap();
        let extractor: crate::kv_store::index_set::ExtractorFn = std::sync::Arc::new(|bytes: &[u8]| {
            let s = std::str::from_utf8(bytes).ok()?;
            let v: serde_json::Value = serde_json::from_str(s).ok()?;
            Some(crate::field_index::IndexValue::Str(v["status"].as_str()?.to_string()))
        });
        let t = std::time::Instant::now();
        db.activate_field_index(ns, field_id, crate::field_index::IndexValueType::Str, extractor)
            .unwrap();
        let dt = t.elapsed();
        println!(
            "  window={window:5} keys -> replay {:>8.1?} ({:.2} ms/key)",
            dt,
            dt.as_secs_f64() * 1000.0 / window as f64
        );
        db.shutdown().unwrap();
    }
}

/// Every field index of a database charges one write-buffer budget, and a
/// write that leaves it over the hard limit writes its field's buffer out
/// before returning. Two fields, a 32 KiB hard limit and 6,000 documents: the
/// total must stay within the limit plus one changed bitmap per field (the
/// documented bound), and every query must stay correct.
#[test]
fn field_index_write_buffers_share_one_budget_and_stay_bounded() {
    let dir = TempDir::new().unwrap();
    let ns = DEFAULT_NAMESPACE_ID;
    let (soft, hard) = (16 * 1024u64, 32 * 1024u64);
    let mut config = create_db_config();
    config.threshold_config = config.threshold_config.with_index_overlay_bytes(soft, hard);
    let db = Database::open(dir.path(), config).unwrap();
    activate_named_index(&db, ns, "status");
    activate_named_index(&db, ns, "tier");

    let budget = db.index_overlay_budget.clone();
    assert_eq!(budget.hard_limit(), hard, "the configured limit reaches the shared budget");
    // One changed bitmap per field: here at most one full bitset container
    // (8 KiB) plus overhead.
    let slack = 2 * 12 * 1024;
    let mut peak = 0;
    for i in 0..6_000u32 {
        let status = ["active", "inactive", "banned"][i as usize % 3];
        let tier = ["gold", "silver"][i as usize % 2];
        db.put(
            format!("doc:{i:06}").as_bytes(),
            format!(r#"{{"status":"{status}","tier":"{tier}"}}"#).as_bytes(),
        )
        .unwrap();
        peak = peak.max(budget.used());
    }
    assert!(peak >= soft, "both fields must charge the shared budget (peaked at only {peak} bytes)");
    assert!(peak <= hard + slack, "write buffers peaked at {peak} bytes, limit {hard} + slack {slack}");
    assert_eq!(db.query_keys(ns, "status = \"active\"").unwrap().keys.len(), 2_000);
    assert_eq!(db.query_keys(ns, "tier = \"gold\"").unwrap().keys.len(), 3_000);

    db.run_index_checkpoint().unwrap();
    assert_eq!(budget.used(), 0, "a checkpoint writes every buffer out");
    db.shutdown().unwrap();
}

/// A field-index slot torn by a crash (half old, half new after a power loss)
/// fails its checksum at open. The slot is dropped, the field records a
/// full-rebuild gap so the loss is visible, and repair restores every row.
#[test]
fn a_torn_index_slot_records_a_full_rebuild_gap_and_repair_restores_it() {
    let dir = TempDir::new().unwrap();
    let ns = DEFAULT_NAMESPACE_ID;
    let field_id = {
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        let field_id = activate_status_index(&db, ns);
        for i in 0..100u32 {
            let v = if i % 2 == 0 { "active" } else { "inactive" };
            db.put(format!("doc:{i:03}").as_bytes(), format!(r#"{{"status":"{v}"}}"#).as_bytes())
                .unwrap();
        }
        db.run_index_checkpoint().unwrap();
        db.shutdown().unwrap();
        field_id
    };

    // Tear the first checksummed live slot of the field's bitmap key table.
    let keys = crate::layout::namespace_index_dir(&crate::layout::index_root(dir.path()), ns)
        .join(field_id.to_string())
        .join("blobs.keys");
    let mut bytes = std::fs::read(&keys).unwrap();
    let slot = (0..)
        .map(|i| 64 + i * 48)
        .take_while(|&b| b + 48 <= bytes.len())
        .find(|&b| bytes[b] == 3)
        .expect("a checksummed live slot");
    bytes[slot + 30] ^= 0x5A;
    std::fs::write(&keys, &bytes).unwrap();

    let db = Database::open(dir.path(), create_db_config()).unwrap();
    let extractor: crate::kv_store::index_set::ExtractorFn = std::sync::Arc::new(|bytes: &[u8]| {
        let s = std::str::from_utf8(bytes).ok()?;
        let v: serde_json::Value = serde_json::from_str(s).ok()?;
        Some(crate::field_index::IndexValue::Str(v["status"].as_str()?.to_string()))
    });
    db.activate_field_index(ns, field_id, crate::field_index::IndexValueType::Str, extractor)
        .unwrap();
    let gap = db.index_manager.read_gap(ns, field_id).expect("the damage must be recorded");
    assert_eq!(gap.cause, crate::db::indexing::manager::GapCause::DamagedIndexFile);
    assert_eq!(gap.repair, crate::db::indexing::manager::RepairMode::FullRebuild);

    db.repair_field_index(ns, field_id).unwrap();
    assert!(db.index_manager.read_gap(ns, field_id).is_none(), "repair clears the gap");
    assert_eq!(db.query_keys(ns, "status = \"active\"").unwrap().keys.len(), 50);
    assert_eq!(db.query_keys(ns, "status = \"inactive\"").unwrap().keys.len(), 50);
    db.shutdown().unwrap();
}

/// Replaying a **low-cardinality** field must not balloon its blob store.
///
/// The bitmap store is append-only: each spill appends the changed containers
/// and a new directory, leaving the old copies as dead space, and a replay
/// window larger than the write-buffer budget spills repeatedly. On the write path the
/// backpressure valve bounds that; during `activate_field_index`'s replay the
/// valve is a no-op, because it signals the checkpoint *worker* and the workers
/// are not started until every field has been activated.
///
/// Measured before the inline compaction existed: one 5-value field over 16k
/// documents reached 2.6 GB and never finished replaying. This asserts the blob
/// stays within a small multiple of the configured cap.
#[test]
fn replaying_a_low_cardinality_field_keeps_its_blob_bounded() {
    let dir = TempDir::new().unwrap();
    let ns = DEFAULT_NAMESPACE_ID;
    // 256 KiB cap so the test is quick; the property is "bounded by the cap",
    // not any particular byte count.
    let cap: u64 = 256 * 1024;
    let mut config = create_db_config();
    config.threshold_config = config.threshold_config.with_index_blob_backpressure_bytes(cap);

    let field_id = {
        let db = Database::open(dir.path(), config.clone()).unwrap();
        let field_id = activate_status_index(&db, ns);
        // Two distinct values across many rows: the pathological shape.
        for i in 0..4_000u32 {
            let v = if i % 2 == 0 { "active" } else { "inactive" };
            db.put(format!("doc:{i:06}").as_bytes(), format!(r#"{{"status":"{v}"}}"#).as_bytes())
                .unwrap();
        }
        db.run_index_checkpoint().unwrap();
        db.shutdown().unwrap();
        field_id
    };

    // Force a full replay: drop the checkpoint marker so the whole WAL is in
    // the replay window, exactly as it is after a crash with no checkpoint.
    let marker = crate::layout::namespace_index_dir(&crate::layout::index_root(dir.path()), ns)
        .join(field_id.to_string())
        .join("checkpoint");
    std::fs::remove_file(&marker).unwrap();

    let db = Database::open(dir.path(), config).unwrap();
    let extractor: crate::kv_store::index_set::ExtractorFn = std::sync::Arc::new(|bytes: &[u8]| {
        let s = std::str::from_utf8(bytes).ok()?;
        let v: serde_json::Value = serde_json::from_str(s).ok()?;
        Some(crate::field_index::IndexValue::Str(v["status"].as_str()?.to_string()))
    });
    db.activate_field_index(ns, field_id, crate::field_index::IndexValueType::Str, extractor)
        .unwrap();

    let field_dir = crate::layout::namespace_index_dir(&crate::layout::index_root(dir.path()), ns).join(field_id.to_string());
    let on_disk: u64 = std::fs::read_dir(&field_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter_map(|e| e.metadata().ok())
        .filter(|m| m.is_file())
        .map(|m| m.len())
        .sum();

    // Generous bound: compaction fires at `cap` of dead bytes, so the blob can
    // reach roughly cap + one live copy between compactions. 16x cap leaves
    // ample headroom while still failing loudly on unbounded growth (which
    // measured ~180x this cap for the same shape).
    assert!(
        on_disk < cap * 16,
        "field blob grew to {on_disk} bytes with a {cap}-byte cap — replay is not compacting"
    );

    // ...and the replay must still be CORRECT, not just small.
    let outcome = db.query_keys(ns, "status = \"active\"").unwrap();
    assert_eq!(outcome.keys.len(), 2_000, "replay must rebuild every row it bounded");

    db.shutdown().unwrap();
}

/// Verify that `activate_field_index` replays the WAL tail into the index
/// on reopen, covering any writes that happened after the last checkpoint.
///
/// Sequence:
///   1. Open DB, activate index, write batch A.
///   2. Checkpoint (flush BlobStore + keymap mmap store + checkpoint file).
///   3. Write batch B (live index updated, checkpoint is now stale).
///   4. Drop DB without shutdown — simulates a crash.
///   5. Reopen DB, activate index (WAL replay runs for batch B).
///   6. Query must return all of batch A and batch B.
#[test]
fn test_activate_field_index_replays_wal_after_crash() {
    use crate::field_index::{IndexValue, IndexValueType};
    use crate::kv_store::index_set::ExtractorFn;
    use std::sync::Arc;

    let dir = TempDir::new().unwrap();
    let ns = DEFAULT_NAMESPACE_ID;

    let make_extractor = || -> ExtractorFn {
        Arc::new(|bytes: &[u8]| {
            let s = std::str::from_utf8(bytes).ok()?;
            let v: serde_json::Value = serde_json::from_str(s).ok()?;
            Some(IndexValue::Str(v["status"].as_str()?.to_string()))
        })
    };

    let field_id;
    {
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        field_id = db.register_index_field(ns, "status", IndexValueType::Str).unwrap();
        db.activate_field_index(ns, field_id, IndexValueType::Str, make_extractor()).unwrap();

        // Batch A — will be on disk after checkpoint.
        db.put(b"user:1", br#"{"status":"active"}"#).unwrap();
        db.put(b"user:2", br#"{"status":"inactive"}"#).unwrap();

        // Checkpoint: flush BlobStore + keymap mmap store + checkpoint marker.
        db.run_index_checkpoint().unwrap();

        // Batch B — live index only; checkpoint is now stale.
        db.put(b"user:3", br#"{"status":"active"}"#).unwrap();
        db.put(b"user:4", br#"{"status":"inactive"}"#).unwrap();

        // Drop without shutdown — WAL has batch B but checkpoint does not.
    }

    {
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        // activate_field_index must replay batch B from the WAL tail.
        db.activate_field_index(ns, field_id, IndexValueType::Str, make_extractor()).unwrap();

        let mut active_keys = db.query_keys(ns, "status = \"active\"").unwrap().keys;
        active_keys.sort();
        assert_eq!(
            active_keys,
            vec![b"user:1".to_vec(), b"user:3".to_vec()],
            "WAL replay must include batch-B writes",
        );

        let mut inactive_keys = db.query_keys(ns, "status = \"inactive\"").unwrap().keys;
        inactive_keys.sort();
        assert_eq!(inactive_keys, vec![b"user:2".to_vec(), b"user:4".to_vec()],);

        db.shutdown().unwrap();
    }
}

/// Item 13 regression: replaying an **update** (same key, changed field
/// value) must reconcile via the targeted O(1) path — during replay
/// `lsm.get` returns the replay-so-far value, so the second put moves the
/// row off the old value. A bug here would leave the key matching BOTH the
/// old and new value after recovery.
#[test]
fn test_field_index_replays_updates_after_crash() {
    use crate::field_index::{IndexValue, IndexValueType};
    use crate::kv_store::index_set::ExtractorFn;
    use std::sync::Arc;

    let dir = TempDir::new().unwrap();
    let ns = DEFAULT_NAMESPACE_ID;
    let make_extractor = || -> ExtractorFn {
        Arc::new(|bytes: &[u8]| {
            let s = std::str::from_utf8(bytes).ok()?;
            let v: serde_json::Value = serde_json::from_str(s).ok()?;
            Some(IndexValue::Str(v["status"].as_str()?.to_string()))
        })
    };

    let field_id;
    {
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        field_id = db.register_index_field(ns, "status", IndexValueType::Str).unwrap();
        db.activate_field_index(ns, field_id, IndexValueType::Str, make_extractor()).unwrap();

        // Insert, then UPDATE the same key to a different value — all in the
        // WAL, no checkpoint, so recovery must replay both writes in order.
        db.put(b"u:1", br#"{"status":"active"}"#).unwrap();
        db.put(b"u:1", br#"{"status":"archived"}"#).unwrap(); // update
        db.put(b"u:2", br#"{"status":"active"}"#).unwrap();
        // Drop without shutdown — index lives only in the WAL.
    }

    {
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        db.activate_field_index(ns, field_id, IndexValueType::Str, make_extractor()).unwrap();

        // u:1 must match ONLY its latest value after replay, not the old one.
        assert_eq!(
            db.query_keys(ns, "status = \"archived\"").unwrap().keys,
            vec![b"u:1".to_vec()],
            "replayed update must land u:1 under its new value"
        );
        assert_eq!(
            db.query_keys(ns, "status = \"active\"").unwrap().keys,
            vec![b"u:2".to_vec()],
            "replayed update must remove u:1 from its old value (no stale bucket)"
        );
        db.shutdown().unwrap();
    }
}

/// Regression for the O(1) targeted field-index update/delete (item 13): a
/// document update must move its row from the old value's bucket to the new
/// one (the prior value must stop matching), and a delete must remove it —
/// driven by the prior document bytes read in the put/delete path, not a
/// full bucket scan.
#[test]
fn test_field_index_targeted_update_and_delete() {
    use crate::field_index::{IndexValue, IndexValueType};
    use crate::kv_store::index_set::ExtractorFn;
    use std::sync::Arc;

    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    let ns = DEFAULT_NAMESPACE_ID;

    let field_id = db.register_index_field(ns, "status", IndexValueType::Str).unwrap();
    let extractor: ExtractorFn = Arc::new(|bytes: &[u8]| {
        let s = std::str::from_utf8(bytes).ok()?;
        let v: serde_json::Value = serde_json::from_str(s).ok()?;
        Some(IndexValue::Str(v["status"].as_str()?.to_string()))
    });
    db.activate_field_index(ns, field_id, IndexValueType::Str, extractor).unwrap();

    db.put(b"doc:1", br#"{"status":"active"}"#).unwrap();
    db.put(b"doc:2", br#"{"status":"active"}"#).unwrap();
    assert_eq!(db.query_keys(ns, "status = \"active\"").unwrap().keys.len(), 2);

    // Update doc:1's status active -> archived. The targeted update must move
    // the row, so it no longer matches "active" and now matches "archived".
    db.put(b"doc:1", br#"{"status":"archived"}"#).unwrap();
    assert_eq!(db.query_keys(ns, "status = \"active\"").unwrap().keys, vec![b"doc:2".to_vec()]);
    assert_eq!(db.query_keys(ns, "status = \"archived\"").unwrap().keys, vec![b"doc:1".to_vec()]);

    // Updating to the same value is a no-op and keeps the row queryable.
    db.put(b"doc:2", br#"{"status":"active"}"#).unwrap();
    assert_eq!(db.query_keys(ns, "status = \"active\"").unwrap().keys, vec![b"doc:2".to_vec()]);

    // Delete doc:2 → it must leave the "active" bucket.
    db.delete(b"doc:2").unwrap();
    assert!(db.query_keys(ns, "status = \"active\"").unwrap().keys.is_empty());
    assert_eq!(db.query_keys(ns, "status = \"archived\"").unwrap().keys, vec![b"doc:1".to_vec()]);

    db.shutdown().unwrap();
}

/// FR-006: once every field index of a namespace is dropped, its row map can be
/// reset, freeing every row ID; the rebuilt index then packs the live keys into
/// IDs 0..live. While any field still exists the reset is refused.
#[test]
fn reset_rowmap_frees_every_row_id_once_no_field_exists() {
    let dir = TempDir::new().unwrap();
    let ns = DEFAULT_NAMESPACE_ID;
    let rowmap_dir = |db: &Database| db.index_manager.rowmap_path(ns);
    let rowmap_bytes = |d: &std::path::Path| -> u64 { std::fs::read_dir(d).unwrap().map(|e| e.unwrap().metadata().unwrap().len()).sum() };
    {
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        let field = activate_named_index(&db, ns, "status");
        for i in 0..5_000u32 {
            db.put(format!("doc:{i:06}").as_bytes(), br#"{"status":"active"}"#).unwrap();
        }
        for i in 0..5_000u32 {
            if i % 10 != 0 {
                db.delete(format!("doc:{i:06}").as_bytes()).unwrap();
            }
        }
        db.run_index_checkpoint().unwrap();
        assert_eq!(db.rowmap_ids_allocated(ns).unwrap(), Some(5_000));
        let before = rowmap_bytes(&rowmap_dir(&db));

        let err = db.reset_rowmap(ns).unwrap_err().to_string();
        assert!(err.contains("status"), "refused while the field exists: {err}");
        assert_eq!(db.rowmap_ids_allocated(ns).unwrap(), Some(5_000), "a refused reset changes nothing");

        db.drop_field_index(ns, field).unwrap();
        assert!(db.reset_rowmap(ns).unwrap());
        assert_eq!(db.rowmap_ids_allocated(ns).unwrap(), Some(0));
        assert!(rowmap_bytes(&rowmap_dir(&db)) < before, "the files shrink");

        // Rebuild: re-register and re-index the 500 live documents.
        activate_named_index(&db, ns, "status");
        for i in (0..5_000u32).step_by(10) {
            db.put(format!("doc:{i:06}").as_bytes(), br#"{"status":"active"}"#).unwrap();
        }
        assert_eq!(db.rowmap_ids_allocated(ns).unwrap(), Some(500));
        let mut keys = db.query_keys(ns, "status = \"active\"").unwrap().keys;
        keys.sort();
        let want: Vec<Vec<u8>> = (0..5_000u32).step_by(10).map(|i| format!("doc:{i:06}").into_bytes()).collect();
        assert_eq!(keys, want);
        db.shutdown().unwrap();
    }
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    activate_named_index(&db, ns, "status");
    assert_eq!(db.rowmap_ids_allocated(ns).unwrap(), Some(500), "the compacted map survives a restart");
    assert_eq!(db.query_keys(ns, "status = \"active\"").unwrap().keys.len(), 500);
    db.shutdown().unwrap();
}
