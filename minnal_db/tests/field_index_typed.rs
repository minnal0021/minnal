//! Field-level indexing over typed (rkyv) struct values with a `u64` key.
//!
//! Mirrors the "Field-Level Indexing" example in the top-level README — keep
//! the two in sync. Demonstrates indexing fields pulled out of an archived
//! struct written with `put_typed`, then querying them with the predicate DSL.

use std::sync::Arc;

use minnal_db::rkyv_derives::{Archive, Deserialize, Serialize};
use minnal_db::{Archived, DEFAULT_NAMESPACE_ID, Db, ExtractorFn, IndexValue, IndexValueType, KVError, access, rancor};

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
    let db = Db::open(dir.path())?;

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
    let keys = db.query_index(DEFAULT_NAMESPACE_ID, r#"status = "active" AND age > 20"#)?;

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

/// The bitmap blob store is append-only: every per-document insert rewrites the
/// whole bitmap for that field value and orphans the previous copy. A low-
/// cardinality field over many documents therefore balloons the value region.
/// The checkpoint must reclaim that dead space via compaction once it crosses
/// the waste threshold — and the live index must stay correct afterward.
#[test]
fn checkpoint_compacts_bloated_field_index() -> Result<(), KVError> {
    let dir = tempfile::TempDir::new().unwrap();
    let db = Db::open(dir.path())?;

    let status_field = db.register_index_field(DEFAULT_NAMESPACE_ID, "status", IndexValueType::Str)?;
    let status_extractor: ExtractorFn = Arc::new(|bytes: &[u8]| {
        let user = access::<ArchivedUser, rancor::Error>(bytes).ok()?;
        Some(IndexValue::Str(user.status.as_str().to_string()))
    });
    db.activate_field_index(DEFAULT_NAMESPACE_ID, status_field, IndexValueType::Str, status_extractor)?;

    // 500 docs over just two status values → each value's bitmap is rewritten
    // ~250 times, leaving heavy dead space in the append-only value region. The
    // bitmap store is a file-backed mmap that grows on disk during the inserts,
    // so the bloat is already on disk before any checkpoint runs.
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
    let bloated = total_blob_value_bytes(dir.path());
    assert!(bloated > 256 * 1024, "expected a bloated value region, got {bloated} bytes");

    // An in-place checkpoint compacts because the bitmap store's waste (~99%) is
    // well over the 50% threshold — and the db stays open and queryable.
    db.checkpoint_index()?;
    let compacted = total_blob_value_bytes(dir.path());
    assert!(
        compacted * 4 < bloated,
        "compaction must reclaim most of the value region: {compacted} not « {bloated}"
    );

    // The live index must be correct after compaction.
    let active = db.query_index(DEFAULT_NAMESPACE_ID, r#"status = "active""#)?;
    assert_eq!(active.len() as u64, n / 2, "every even-keyed doc must still match after compaction");

    // A second checkpoint is now a cheap no-op (waste reads back ≈0).
    let before_second = total_blob_value_bytes(dir.path());
    db.checkpoint_index()?;
    assert_eq!(
        total_blob_value_bytes(dir.path()),
        before_second,
        "already-compacted store must not be rewritten"
    );

    db.shutdown()?;
    Ok(())
}
