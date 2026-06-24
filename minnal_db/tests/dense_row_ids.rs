//! Dense, sequential row IDs.
//!
//! The field-index row ID is assigned from a per-namespace monotonic counter
//! (the `RowMap` sidecar) instead of a random hash, so consecutive documents
//! share a RoaringBitmap high key and pack densely into one container rather than
//! scattering one-per-doc. These tests cover the payoff (bitmap size stays ~flat
//! as document count grows) and the durability contract (IDs survive a clean
//! restart and are correctly rebuilt by WAL replay after a crash).

use std::sync::Arc;

use minnal_db::rkyv_derives::{Archive, Deserialize, Serialize};
use minnal_db::{Archived, DEFAULT_NAMESPACE_ID, Db, ExtractorFn, IndexValue, IndexValueType, KVError, access, rancor};

#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize)]
struct User {
    status: String,
    age: i64,
}

fn status_extractor() -> ExtractorFn {
    Arc::new(|bytes: &[u8]| {
        let user = access::<ArchivedUser, rancor::Error>(bytes).ok()?;
        Some(IndexValue::Str(user.status.as_str().to_string()))
    })
}

/// Sum the sizes of every `blobs.vals` value file under `root` — the append-only
/// region holding the serialised bitmaps. With dense IDs a single-value bitmap is
/// one container regardless of doc count; with sparse hash IDs it would be one
/// container per doc.
fn total_blob_value_bytes(root: &std::path::Path) -> u64 {
    fn walk(dir: &std::path::Path, total: &mut u64) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, total);
            } else if path.file_name().and_then(|n| n.to_str()) == Some("blobs.vals") {
                *total += std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            }
        }
    }
    let mut total = 0;
    walk(root, &mut total);
    total
}

fn put_user(db: &Db, key: u64, status: &str) -> Result<(), KVError> {
    db.put_typed(
        &key,
        &User {
            status: status.into(),
            age: key as i64,
        },
    )
}

/// With dense IDs, 2000 documents sharing one field value pack into a *single*
/// RoaringBitmap container (ids 0..1999 share high key 0), so the compacted
/// bitmap is a few KiB. With sparse hash IDs the same 2000 docs would scatter
/// into ~2000 one-element containers (~50+ KiB serialised), so a tight absolute
/// bound is a clean density signature.
#[test]
fn dense_bitmap_stays_compact() -> Result<(), KVError> {
    let dir = tempfile::TempDir::new().unwrap();
    let db = Db::open(dir.path())?;
    let field = db.register_index_field(DEFAULT_NAMESPACE_ID, "status", IndexValueType::Str)?;
    db.activate_field_index(DEFAULT_NAMESPACE_ID, field, IndexValueType::Str, status_extractor())?;

    let n = 2_000u64;
    for i in 0..n {
        put_user(&db, i, "active")?;
    }
    // Compact away the append churn so we measure the live bitmap, not the
    // per-insert rewrite history.
    db.checkpoint_index()?;

    let bytes = total_blob_value_bytes(dir.path());
    assert!(
        bytes < 24 * 1024,
        "dense bitmap over {n} docs should be a few KiB, got {bytes} bytes \
         (sparse hash IDs would be ~50+ KiB of one-element containers)"
    );

    // Sanity: the query still returns every doc.
    let active = db.query_index(DEFAULT_NAMESPACE_ID, r#"status = "active""#)?;
    assert_eq!(active.len(), n as usize);

    db.shutdown()?;
    Ok(())
}

/// IDs and query results survive a clean shutdown + reopen (the row map is loaded
/// from its on-disk marker; no replay needed).
#[test]
fn ids_survive_clean_restart() -> Result<(), KVError> {
    let dir = tempfile::TempDir::new().unwrap();
    let field;
    {
        let db = Db::open(dir.path())?;
        field = db.register_index_field(DEFAULT_NAMESPACE_ID, "status", IndexValueType::Str)?;
        db.activate_field_index(DEFAULT_NAMESPACE_ID, field, IndexValueType::Str, status_extractor())?;
        for i in 0..200u64 {
            put_user(&db, i, if i % 2 == 0 { "active" } else { "inactive" })?;
        }
        db.shutdown()?; // flushes the row map + index
    }

    // Reopen: the field is still registered (schema persisted); re-supply the
    // extractor. The row map is loaded from disk, so query resolution works.
    let db = Db::open(dir.path())?;
    db.activate_field_index(DEFAULT_NAMESPACE_ID, field, IndexValueType::Str, status_extractor())?;

    let active = db.query_index(DEFAULT_NAMESPACE_ID, r#"status = "active""#)?;
    let mut ids: Vec<u64> = active
        .iter()
        .map(|kb| access::<Archived<u64>, rancor::Error>(kb).unwrap().to_native())
        .collect();
    ids.sort_unstable();
    assert_eq!(ids, (0..200u64).filter(|i| i % 2 == 0).collect::<Vec<_>>());

    // An update after restart must reuse the doc's existing ID, not duplicate it.
    put_user(&db, 0, "inactive")?;
    let active_after = db.query_index(DEFAULT_NAMESPACE_ID, r#"status = "active""#)?;
    assert_eq!(active_after.len(), 99, "doc 0 must move out of 'active', not linger under a stale id");

    db.shutdown()?;
    Ok(())
}

/// IDs and queries are correctly rebuilt by WAL replay after a crash that lands
/// between the last checkpoint and the writes that followed it (the row map +
/// field index are reconstructed together from the WAL tail).
#[test]
fn ids_rebuilt_by_wal_replay_after_crash() -> Result<(), KVError> {
    let dir = tempfile::TempDir::new().unwrap();
    let field;
    {
        let db = Db::open(dir.path())?;
        field = db.register_index_field(DEFAULT_NAMESPACE_ID, "status", IndexValueType::Str)?;
        db.activate_field_index(DEFAULT_NAMESPACE_ID, field, IndexValueType::Str, status_extractor())?;

        // Checkpoint persists the row map + index for the first 100 docs…
        for i in 0..100u64 {
            put_user(&db, i, if i % 2 == 0 { "active" } else { "inactive" })?;
        }
        db.checkpoint_index()?;

        // …then 100 more docs land in the WAL but are NOT checkpointed.
        for i in 100..200u64 {
            put_user(&db, i, if i % 2 == 0 { "active" } else { "inactive" })?;
        }
        // Simulate a crash: drop without shutdown. The WAL is durable per-write,
        // but the post-checkpoint index/row-map state is in memory only.
        std::mem::drop(db);
    }

    // Reopen: activation replays the WAL tail, re-resolving every touched key
    // through the row map (durable keys keep their id; post-checkpoint keys are
    // re-allocated deterministically) and rebuilding the bitmaps.
    let db = Db::open(dir.path())?;
    db.activate_field_index(DEFAULT_NAMESPACE_ID, field, IndexValueType::Str, status_extractor())?;

    let active = db.query_index(DEFAULT_NAMESPACE_ID, r#"status = "active""#)?;
    let mut ids: Vec<u64> = active
        .iter()
        .map(|kb| access::<Archived<u64>, rancor::Error>(kb).unwrap().to_native())
        .collect();
    ids.sort_unstable();
    assert_eq!(
        ids,
        (0..200u64).filter(|i| i % 2 == 0).collect::<Vec<_>>(),
        "every active doc across the checkpoint boundary must be recovered exactly once"
    );

    db.shutdown()?;
    Ok(())
}
