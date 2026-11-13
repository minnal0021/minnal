//! Query embedding cache tests.

use super::super::*;
use crate::AsyncDb;
use std::time::Duration;
use tempfile::TempDir;

async fn open_db(dir: &TempDir) -> AsyncDb {
    AsyncDb::open_with_config(dir.path().to_owned(), crate::support::test_db_config())
        .await
        .unwrap()
}

/// TTL used by the cache round-trip tests.
const TEST_TTL: Duration = crate::defaults::semantic_search::QUERY_EMBEDDING_CACHE_TTL;
const MODEL: &str = "gemma";

// ── Encoding helpers ──────────────────────────────────────────────────────

#[test]
fn test_f32_vec_list_roundtrip_single() {
    let embeddings = vec![vec![1.0f32, 2.0, 3.0, 4.0]];
    let bytes = f32_vec_list_to_bytes(&embeddings);
    let decoded = bytes_to_f32_vec_list(&bytes, 4).unwrap();
    assert_eq!(decoded, embeddings);
}

#[test]
fn test_f32_vec_list_roundtrip_multiple() {
    let embeddings = vec![vec![1.0f32, 2.0], vec![3.0f32, 4.0], vec![5.0f32, 6.0]];
    let bytes = f32_vec_list_to_bytes(&embeddings);
    let decoded = bytes_to_f32_vec_list(&bytes, 2).unwrap();
    assert_eq!(decoded, embeddings);
}

#[test]
fn test_f32_vec_list_roundtrip_empty_list() {
    let embeddings: Vec<Vec<f32>> = vec![];
    let bytes = f32_vec_list_to_bytes(&embeddings);
    let decoded = bytes_to_f32_vec_list(&bytes, 4).unwrap();
    assert_eq!(decoded, embeddings);
}

#[test]
fn test_f32_vec_list_rejects_wrong_dim() {
    let embeddings = vec![vec![1.0f32, 2.0, 3.0, 4.0]];
    let bytes = f32_vec_list_to_bytes(&embeddings);
    // encoded as dim=4, but we decode claiming dim=3
    assert!(bytes_to_f32_vec_list(&bytes, 3).is_none());
}

#[test]
fn test_f32_vec_list_rejects_too_short() {
    assert!(bytes_to_f32_vec_list(&[0, 1], 4).is_none());
}

#[test]
fn test_f32_vec_list_rejects_truncated_payload() {
    // count=1 but only 8 bytes of payload instead of the required 4*4=16
    let mut bytes = 1u32.to_be_bytes().to_vec();
    bytes.extend_from_slice(&[0u8; 8]);
    assert!(bytes_to_f32_vec_list(&bytes, 4).is_none());
}

// ── Cache round-trip ──────────────────────────────────────────────────────

#[tokio::test]
async fn test_cache_miss_returns_none() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;
    assert!(get_cached_query_embedding(&db, MODEL, "unknown", 4, TEST_TTL).await.is_none());
}

#[tokio::test]
async fn test_cache_roundtrip() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;
    let dense = vec![0.1f32, 0.2, 0.3, 0.4];
    put_cached_query_embedding(&db, MODEL, "hello", &dense, TEST_TTL).await;
    let cached = get_cached_query_embedding(&db, MODEL, "hello", 4, TEST_TTL).await.unwrap();
    for (a, b) in cached.iter().zip(dense.iter()) {
        assert!((a - b).abs() < 1e-6, "f32 mismatch after cache round-trip");
    }
}

#[tokio::test]
async fn test_cache_miss_on_dim_mismatch() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;
    let dense = vec![1.0f32, 2.0, 3.0, 4.0];
    put_cached_query_embedding(&db, MODEL, "query", &dense, TEST_TTL).await;
    // stored as dim=4; ask for dim=3 — should be a cache miss
    assert!(get_cached_query_embedding(&db, MODEL, "query", 3, TEST_TTL).await.is_none());
}

#[tokio::test]
async fn test_cache_different_queries_are_independent() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;
    let dense_a = vec![1.0f32, 0.0];
    let dense_b = vec![0.0f32, 1.0];
    put_cached_query_embedding(&db, MODEL, "query-a", &dense_a, TEST_TTL).await;
    put_cached_query_embedding(&db, MODEL, "query-b", &dense_b, TEST_TTL).await;
    let cached_a = get_cached_query_embedding(&db, MODEL, "query-a", 2, TEST_TTL).await.unwrap();
    let cached_b = get_cached_query_embedding(&db, MODEL, "query-b", 2, TEST_TTL).await.unwrap();
    assert!((cached_a[0] - 1.0f32).abs() < 1e-6);
    assert!((cached_b[1] - 1.0f32).abs() < 1e-6);
}

/// The recall check's sample of cached queries holds only the asked model at
/// the asked dimension, at most `limit`, and every one of them whatever the
/// seed's starting point (the read wraps).
#[tokio::test]
async fn sampled_cached_queries_are_the_models_own_and_bounded() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;
    for (i, q) in ["alpha", "beta", "gamma", "zeta", "~tilde"].iter().enumerate() {
        put_cached_query_embedding(&db, MODEL, q, &[i as f32, 1.0], TEST_TTL).await;
    }
    put_cached_query_embedding(&db, "other", "alpha", &[9.0, 9.0], TEST_TTL).await;
    put_cached_query_embedding(&db, MODEL, "three", &[9.0, 9.0, 9.0], TEST_TTL).await;
    for seed in [0u64, 0x62, 0x7a, 255] {
        let mut got = sample_cached_query_embeddings(&db, MODEL, 2, TEST_TTL, 100, seed).await.unwrap();
        got.sort_by(|a, b| a[0].total_cmp(&b[0]));
        let firsts: Vec<f32> = got.iter().map(|v| v[0]).collect();
        assert_eq!(firsts, [0.0, 1.0, 2.0, 3.0, 4.0], "seed {seed}");
        assert_eq!(sample_cached_query_embeddings(&db, MODEL, 2, TEST_TTL, 2, seed).await.unwrap().len(), 2);
    }
    assert!(sample_cached_query_embeddings(&db, "none", 2, TEST_TTL, 10, 0).await.unwrap().is_empty());
}

#[tokio::test]
async fn test_clear_removes_all_entries() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;
    put_cached_query_embedding(&db, MODEL, "query-a", &[1.0f32, 0.0], TEST_TTL).await;
    put_cached_query_embedding(&db, MODEL, "query-b", &[0.0f32, 1.0], TEST_TTL).await;

    let cleared = clear_cached_query_embeddings(&db, TEST_TTL).await.unwrap();
    assert_eq!(cleared, 2);
    assert!(get_cached_query_embedding(&db, MODEL, "query-a", 2, TEST_TTL).await.is_none());
    assert!(get_cached_query_embedding(&db, MODEL, "query-b", 2, TEST_TTL).await.is_none());

    // Clearing an already-empty cache is a no-op that reports zero.
    assert_eq!(clear_cached_query_embeddings(&db, TEST_TTL).await.unwrap(), 0);
}

/// Cache populates must stay off the WAL (no fsync on the query path),
/// while the admin clear goes through it so it survives a crash. This pins
/// the durability split for the TTL cache namespace.
#[tokio::test]
async fn test_cache_populates_bypass_wal_and_clears_use_it() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;

    // Baseline after namespace setup, so we only measure the cache writes. The
    // cache's namespace is created on first use, which writes the system
    // catalog: its puts are left out.
    let catalog = || db.ops_metrics_for(crate::CATALOG_NAMESPACE).unwrap();
    let (before, catalog_before) = (db.ops_metrics(), catalog());

    put_cached_query_embedding(&db, MODEL, "q1", &[1.0f32, 0.0], TEST_TTL).await;
    put_cached_query_embedding(&db, MODEL, "q2", &[0.0f32, 1.0], TEST_TTL).await;
    let cleared = clear_cached_query_embeddings(&db, TEST_TTL).await.unwrap();
    assert_eq!(cleared, 2);

    let after = db.ops_metrics();
    let catalog_puts = catalog().puts - catalog_before.puts;
    assert_eq!(after.no_wal_puts - before.no_wal_puts, 2, "two cache populates use put_no_wal");
    assert_eq!(after.puts - before.puts - catalog_puts, 0, "no WAL-backed puts from the cache");
    assert_eq!(after.deletes - before.deletes, 2, "two cache clears use the WAL-backed delete");
    assert_eq!(after.no_wal_deletes - before.no_wal_deletes, 0, "no cache clear bypasses the WAL");
}

/// Populate the cache and flush it to L0 — the state a long-running server
/// is in once the periodic no-WAL flush tick has persisted cache entries.
async fn populate_and_flush(db: &AsyncDb) {
    put_cached_query_embedding(db, MODEL, "q1", &[1.0f32, 0.0], TEST_TTL).await;
    put_cached_query_embedding(db, MODEL, "q2", &[0.0f32, 1.0], TEST_TTL).await;
    assert!(db.coordinator_for_test().flush_no_wal_memtables() >= 1);
}

/// A clear must survive a graceful restart.
#[tokio::test]
async fn test_cache_clear_survives_graceful_restart() {
    let dir = TempDir::new().unwrap();
    {
        let db = open_db(&dir).await;
        populate_and_flush(&db).await;
        assert_eq!(clear_cached_query_embeddings(&db, TEST_TTL).await.unwrap(), 2);
        db.shutdown().await.unwrap();
    }
    let db = open_db(&dir).await;
    assert!(get_cached_query_embedding(&db, MODEL, "q1", 2, TEST_TTL).await.is_none());
    assert!(get_cached_query_embedding(&db, MODEL, "q2", 2, TEST_TTL).await.is_none());
}

/// A clear must survive a crash that lands before the next no-WAL flush
/// tick. The cleared entries were already flushed to disk, so if the
/// clear's tombstones only lived in the memtable, the crash would drop them
/// and resurrect every cleared entry on reopen.
#[tokio::test]
async fn test_cache_clear_survives_crash_before_flush_tick() {
    let dir = TempDir::new().unwrap();
    {
        let db = open_db(&dir).await;
        populate_and_flush(&db).await;
        assert_eq!(clear_cached_query_embeddings(&db, TEST_TTL).await.unwrap(), 2);
        // Crash: skip shutdown and every Drop-time flush.
        db.crash().await;
    }
    let db = open_db(&dir).await;
    assert!(get_cached_query_embedding(&db, MODEL, "q1", 2, TEST_TTL).await.is_none());
    assert!(get_cached_query_embedding(&db, MODEL, "q2", 2, TEST_TTL).await.is_none());
}

/// The same query text cached for two models keeps two entries: switching
/// models must not serve the other model's vector.
#[tokio::test]
async fn test_cache_is_keyed_by_model() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;
    put_cached_query_embedding(&db, "gemma", "q", &[1.0f32, 0.0], TEST_TTL).await;
    assert!(get_cached_query_embedding(&db, "qwen", "q", 2, TEST_TTL).await.is_none());
    put_cached_query_embedding(&db, "qwen", "q", &[0.0f32, 1.0], TEST_TTL).await;
    assert_eq!(
        get_cached_query_embedding(&db, "gemma", "q", 2, TEST_TTL).await.unwrap(),
        vec![1.0f32, 0.0]
    );
    assert_eq!(
        get_cached_query_embedding(&db, "qwen", "q", 2, TEST_TTL).await.unwrap(),
        vec![0.0f32, 1.0]
    );
    assert_ne!(query_cache_key("ab", 2, "c"), query_cache_key("a", 2, "bc"));
}

/// The same model at two dimensions never shares an entry: each namespace
/// picks its own dimension.
#[tokio::test]
async fn test_cache_is_keyed_by_dimension() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;
    put_cached_query_embedding(&db, MODEL, "q", &[1.0f32, 0.0], TEST_TTL).await;
    assert!(get_cached_query_embedding(&db, MODEL, "q", 4, TEST_TTL).await.is_none());
    put_cached_query_embedding(&db, MODEL, "q", &[0.0f32, 0.0, 1.0, 0.0], TEST_TTL).await;
    assert_eq!(get_cached_query_embedding(&db, MODEL, "q", 2, TEST_TTL).await.unwrap(), vec![1.0f32, 0.0]);
    assert_eq!(
        get_cached_query_embedding(&db, MODEL, "q", 4, TEST_TTL).await.unwrap(),
        vec![0.0f32, 0.0, 1.0, 0.0]
    );
    assert_ne!(query_cache_key(MODEL, 2, "q"), query_cache_key(MODEL, 4, "q"));
}

/// Re-embedding the same query overwrites its entry.
#[tokio::test]
async fn test_cache_reembed_overwrites() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;
    put_cached_query_embedding(&db, MODEL, "q", &[1.0f32, 0.0], TEST_TTL).await;
    put_cached_query_embedding(&db, MODEL, "q", &[0.0f32, 1.0], TEST_TTL).await;
    assert_eq!(get_cached_query_embedding(&db, MODEL, "q", 2, TEST_TTL).await.unwrap(), vec![0.0f32, 1.0]);
}
