//! Field-level indexing over typed (rkyv) struct values with a `u64` key.
//!
//! Mirrors the "Field-Level Indexing" example in the top-level README — keep
//! the two in sync. Demonstrates indexing fields pulled out of an archived
//! struct written with `put_typed`, then querying them with the predicate DSL.

use std::sync::Arc;

use minnal_db::rkyv_derives::{Archive, Deserialize, Serialize};
use minnal_db::{Archived, DEFAULT_NAMESPACE_ID, Db, DbConfig, ExtractorFn, IndexValue, IndexValueType, KVError, access, rancor};

/// Small bucket count keeps the eager per-namespace fd footprint low so the
/// suite survives high `cargo test` parallelism (each namespace opens
/// `2 × num_buckets` files). Two buckets still exercises multi-bucket routing.
fn open_test_db(path: &std::path::Path) -> Result<Db, KVError> {
    Db::open_with_config(
        path,
        DbConfig {
            num_buckets: 2,
            ..DbConfig::default()
        },
    )
}

/// The value type. Derives the rkyv traits re-exported from `minnal_db`, so the
/// derive macro also generates `ArchivedUser` for zero-copy field access.
#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize)]
struct User {
    status: String,
    age: i64,
}

#[test]
fn field_index_over_typed_struct_value() -> Result<(), KVError> {
    let dir = tempfile::TempDir::new().unwrap();
    let db = open_test_db(dir.path())?;

    // 1. Declare which fields to index on the default namespace. Returns a FieldId.
    let status_field = db.register_index_field(DEFAULT_NAMESPACE_ID, "status", IndexValueType::Str)?;
    let age_field = db.register_index_field(DEFAULT_NAMESPACE_ID, "age", IndexValueType::Int)?;

    // 2. Activate each field with an *extractor*. The stored bytes are an rkyv
    //    archive of `User`, so we borrow it zero-copy with `access` and read the
    //    field straight off `ArchivedUser` — no full deserialisation.
    let status_extractor: ExtractorFn = Arc::new(|bytes: &[u8]| {
        let user = access::<ArchivedUser, rancor::Error>(bytes).ok()?;
        Some(IndexValue::Str(user.status.as_str().to_string()))
    });
    let age_extractor: ExtractorFn = Arc::new(|bytes: &[u8]| {
        let user = access::<ArchivedUser, rancor::Error>(bytes).ok()?;
        Some(IndexValue::Int(user.age.to_native()))
    });
    db.activate_field_index(DEFAULT_NAMESPACE_ID, status_field, IndexValueType::Str, status_extractor)?;
    db.activate_field_index(DEFAULT_NAMESPACE_ID, age_field, IndexValueType::Int, age_extractor)?;

    // 3. Write typed records with a plain `u64` key. Each `put_typed` rkyv-
    //    serialises key and value, runs the extractors, and updates the indices.
    db.put_typed(
        &1u64,
        &User {
            status: "active".into(),
            age: 30,
        },
    )?;
    db.put_typed(
        &2u64,
        &User {
            status: "inactive".into(),
            age: 25,
        },
    )?;
    db.put_typed(
        &3u64,
        &User {
            status: "active".into(),
            age: 42,
        },
    )?;
    db.put_typed(
        &4u64,
        &User {
            status: "active".into(),
            age: 18,
        },
    )?;

    // 4. Query the index with the predicate DSL. Returns the raw (rkyv) key bytes.
    let keys = db.query_index(DEFAULT_NAMESPACE_ID, r#"status = "active" AND age > 20"#)?.keys;

    // 5. Resolve each matched key: decode the archived u64, then `get_typed`.
    let mut ids: Vec<u64> = keys
        .iter()
        .map(|kb| access::<Archived<u64>, rancor::Error>(kb).expect("key is an archived u64").to_native())
        .collect();
    ids.sort_unstable();

    // user 1 (active, 30) and user 3 (active, 42); user 4 is active but 18, user 2 is inactive.
    assert_eq!(ids, vec![1, 3]);
    for id in &ids {
        let user = db.get_typed::<u64, User>(id)?.expect("key exists");
        assert_eq!(user.status, "active");
        assert!(user.age > 20);
    }

    db.shutdown()?;
    Ok(())
}

/// Sum the sizes of every `blobs.vals` value file under `root` (the append-only
/// region where bitmap dead space accumulates).
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

/// Field-index writes are buffered in memory and written once per checkpoint.
/// The bitmap store is append-only, so writing each document's change straight
/// to it left one dead copy of the whole bitmap per document: 500 documents over
/// two values bloated the value region past 256 KiB before any checkpoint. Now
/// nothing reaches the store before the checkpoint, the checkpoint writes each
/// value's bitmap once, and the index stays correct throughout.
#[test]
fn field_index_writes_are_buffered_until_the_checkpoint() -> Result<(), KVError> {
    let dir = tempfile::TempDir::new().unwrap();
    let db = open_test_db(dir.path())?;

    let status_field = db.register_index_field(DEFAULT_NAMESPACE_ID, "status", IndexValueType::Str)?;
    let status_extractor: ExtractorFn = Arc::new(|bytes: &[u8]| {
        let user = access::<ArchivedUser, rancor::Error>(bytes).ok()?;
        Some(IndexValue::Str(user.status.as_str().to_string()))
    });
    db.activate_field_index(DEFAULT_NAMESPACE_ID, status_field, IndexValueType::Str, status_extractor)?;
    db.checkpoint_index()?;
    let empty = total_blob_value_bytes(dir.path());

    let n = 500u64;
    for i in 0..n {
        let status = if i % 2 == 0 { "active" } else { "inactive" };
        db.put_typed(
            &i,
            &User {
                status: status.into(),
                age: i as i64,
            },
        )?;
    }
    // Queries see the buffered changes before any checkpoint.
    let active = db.query_index(DEFAULT_NAMESPACE_ID, r#"status = "active""#)?.keys;
    assert_eq!(active.len() as u64, n / 2, "buffered writes must be visible to queries");
    // Before the checkpoint the store has not grown (a background checkpoint
    // tick may land here, which writes each bitmap once, still no bloat).
    assert!(
        total_blob_value_bytes(dir.path()) < empty + 64 * 1024,
        "writes must not bloat the bitmap store before the checkpoint"
    );

    // The checkpoint writes each value's bitmap once: two small blobs, no bloat.
    db.checkpoint_index()?;
    let written = total_blob_value_bytes(dir.path());
    assert!(
        written < empty + 64 * 1024,
        "one copy per value expected, value region grew by {} bytes",
        written - empty
    );
    let active = db.query_index(DEFAULT_NAMESPACE_ID, r#"status = "active""#)?.keys;
    assert_eq!(active.len() as u64, n / 2, "every even-keyed doc must still match after the checkpoint");

    // A checkpoint with nothing buffered writes nothing.
    db.checkpoint_index()?;
    assert_eq!(
        total_blob_value_bytes(dir.path()),
        written,
        "an idle checkpoint must not rewrite the store"
    );

    db.shutdown()?;
    Ok(())
}
