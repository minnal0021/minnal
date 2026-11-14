//! Paginated predicate queries return exactly the unpaginated result set.
//!
//! `query_index_paginated` resolves a page by windowing the RoaringBitmap that
//! query evaluation produces. That windowing skips whole containers using each
//! container's **cached cardinality**, whereas a full `query_index` iterates
//! container **contents** — two different mechanisms that must agree. These
//! tests walk a query page by page and assert the concatenation equals the full
//! result, which is the end-to-end guard on that agreement.
//!
//! Both resolution paths are covered, because they window the bitmap
//! separately:
//!
//! - **`RowToKeyFn` registered** — what the document store actually uses
//!   (`activate_indices` registers a key-derived row-ID function for every key
//!   type), so this is the path behind `POST /stores/{ns}/query`.
//! - **dense `RowMap`** — the default when no custom row-ID function is set.
//!
//! Values are stored as raw UTF-8 rather than rkyv so the tests can choose exact
//! key bytes, which is what lets the second one place row IDs in chosen
//! containers without writing 65k documents.

use std::sync::Arc;

use minnal_db::{AsyncDb, DEFAULT_NAMESPACE_ID, DbConfig, ExtractorFn, IndexValue, IndexValueType, KVError, RowIdFn, RowToKeyFn};

/// Small bucket count keeps the per-namespace fd footprint low under parallel
/// `cargo test`, matching the other integration suites.
fn small_config() -> DbConfig {
    DbConfig {
        num_buckets: 2,
        ..DbConfig::default()
    }
}

/// The stored value *is* the status string.
fn status_extractor() -> ExtractorFn {
    Arc::new(|bytes: &[u8]| Some(IndexValue::Str(String::from_utf8_lossy(bytes).into_owned())))
}

const PREDICATE: &str = r#"status = "active""#;

/// Page through `PREDICATE` in `page_size` chunks and assert the walk reproduces
/// the unpaginated result exactly — same keys, same order, no gaps, no repeats —
/// and that the reported total is stable across every page.
async fn assert_paged_walk_matches_full_query(db: &AsyncDb, page_size: usize) -> Result<(), KVError> {
    let full = db.query_index(DEFAULT_NAMESPACE_ID, PREDICATE).await?;
    assert!(!full.keys.is_empty(), "fixture should match something");
    assert!(!full.is_degraded(), "fixture index should be healthy");

    let mut walked: Vec<Vec<u8>> = Vec::new();
    let mut offset = 0usize;
    loop {
        let page = db.query_index_paginated(DEFAULT_NAMESPACE_ID, PREDICATE, offset, page_size).await?;
        assert_eq!(page.total, full.keys.len(), "reported total should not drift at offset {offset}");
        if page.keys.is_empty() {
            break;
        }
        assert!(page.keys.len() <= page_size, "page at offset {offset} exceeded the limit");
        walked.extend(page.keys);
        offset += page_size;
        assert!(offset <= full.keys.len() + page_size, "paging failed to terminate");
    }

    assert_eq!(walked, full.keys, "page-by-page walk (page_size {page_size}) must equal the full query");

    // An offset at the end is empty rather than wrapping or erroring.
    let past_end = db
        .query_index_paginated(DEFAULT_NAMESPACE_ID, PREDICATE, full.keys.len(), page_size)
        .await?;
    assert!(past_end.keys.is_empty(), "offset at the result length should yield nothing");
    assert_eq!(past_end.total, full.keys.len());
    Ok(())
}

#[tokio::test]
async fn paged_query_matches_full_query_via_dense_row_map() -> Result<(), KVError> {
    let dir = tempfile::TempDir::new().unwrap();
    let db = AsyncDb::open_with_config(dir.path().to_path_buf(), small_config()).await?;
    let field = db.register_index_field(DEFAULT_NAMESPACE_ID, "status", IndexValueType::Str)?;
    db.activate_field_index(DEFAULT_NAMESPACE_ID, field, IndexValueType::Str, status_extractor())
        .await?;

    // Interleave matching and non-matching docs so the result bitmap is gappy
    // rather than one solid run.
    for i in 0u64..500 {
        let status = if i.is_multiple_of(3) { "inactive" } else { "active" };
        db.put(i.to_be_bytes().to_vec(), status.as_bytes().to_vec()).await?;
    }

    for page_size in [1, 7, 64, 333, 10_000] {
        assert_paged_walk_matches_full_query(&db, page_size).await?;
    }

    db.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn paged_query_matches_full_query_via_row_id_fn_across_containers() -> Result<(), KVError> {
    let dir = tempfile::TempDir::new().unwrap();
    let db = AsyncDb::open_with_config(dir.path().to_path_buf(), small_config()).await?;

    // A key-derived row ID, as the document store registers for every key type.
    let row_id: RowIdFn = Arc::new(|k: &[u8]| {
        let arr: [u8; 8] = k[..8].try_into().unwrap_or_default();
        u64::from_be_bytes(arr) as u128
    });
    let row_to_key: RowToKeyFn = Arc::new(|id: u128| (id as u64).to_be_bytes().to_vec());
    db.set_row_id_fn(DEFAULT_NAMESPACE_ID, row_id, row_to_key).await?;

    let field = db.register_index_field(DEFAULT_NAMESPACE_ID, "status", IndexValueType::Str)?;
    db.activate_field_index(DEFAULT_NAMESPACE_ID, field, IndexValueType::Str, status_extractor())
        .await?;

    // Keys are spaced beyond 2^16 apart so the derived row IDs land in several
    // RoaringBitmap containers without needing 65k documents. Container skipping
    // in the paginated path is only exercised when more than one exists.
    for block in 0u64..4 {
        let base = block * 70_000;
        for i in 0..120u64 {
            let key = base + i * 3;
            let status = if i.is_multiple_of(4) { "inactive" } else { "active" };
            db.put(key.to_be_bytes().to_vec(), status.as_bytes().to_vec()).await?;
        }
    }

    for page_size in [1, 5, 50, 400, 10_000] {
        assert_paged_walk_matches_full_query(&db, page_size).await?;
    }

    db.shutdown().await?;
    Ok(())
}
