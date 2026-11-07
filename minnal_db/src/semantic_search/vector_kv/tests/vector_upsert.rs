//! Vector upsert tests.

use super::super::*;
use crate::AsyncDb;
use crate::semantic_search::index::distance_estimator::{DotProductEstimatorState, MultiBitQuanDotProductEstimator};
use crate::semantic_search::index::vector_index::{QuantisationStyle, VectorKvStore, score_rkyv_bytes};
use tempfile::TempDir;

async fn open_db(dir: &TempDir) -> AsyncDb {
    let db = AsyncDb::open_with_config(dir.path().to_owned(), crate::support::test_db_config())
        .await
        .unwrap();
    // `upsert_vectors` never creates the vector namespaces, so a write cannot
    // bring back a dropped store's. Create them, as seeding a real store does.
    db.namespace("docs".to_owned()).await.unwrap();
    create_vector_namespaces(&db, "docs").await.unwrap();
    db
}

const MULTI8: QuantisationStyle = QuantisationStyle::MultiBit { number_of_bits: 8 };

// With zero estimator state and empty packed_vector, estimated_distance = 1.0 - addition_factor.
fn zero_estimator() -> MultiBitQuanDotProductEstimator {
    MultiBitQuanDotProductEstimator(DotProductEstimatorState {
        cluster_id: 1,
        query_to_centroid_dot_product: 0.0,
        scaled_query_sum: 0.0,
    })
}

// SingleBit entry → goes to sparse namespace, scannable by cluster.
#[tokio::test]
async fn test_upsert_single_bit_entry_is_scannable_by_cluster() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;
    let ns = "docs";
    let doc_id = b"doc1";

    let vi = VectorIndex::new(1, QuantisationStyle::SingleBit, 0.4, 0.0, 0.02, vec![]);
    upsert_vectors(&db, ns, doc_id, "text", &[vi]).await.unwrap();

    let store = DbVectorStore::new(&db, ns).await.unwrap();
    let entries = store.scan_sparse_cluster(1).await.unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].0, doc_id);
}

// MultiBit entry → goes to dense namespace, fetchable by doc_id.
#[tokio::test]
async fn test_upsert_multi_bit_entry_is_fetchable_by_doc_id() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;
    let ns = "docs";
    let doc_id = b"doc2";

    let vi = VectorIndex::new(1, MULTI8, 0.3, 0.0, 0.05, vec![]);
    upsert_vectors(&db, ns, doc_id, "text", &[vi]).await.unwrap();

    let store = DbVectorStore::new(&db, ns).await.unwrap();
    let bytes = store.get_dense_entry(doc_id).await.unwrap().unwrap();
    let vis = VectorIndex::list_from_bytes(&bytes).unwrap();
    assert_eq!(vis.len(), 1);

    let (score, _) = score_rkyv_bytes(&bytes, &[], &zero_estimator()).unwrap();
    assert!((score - 0.7).abs() < 1e-6);
}

// Multiple MultiBit chunks stored in dense namespace — SimMax picks the best.
#[tokio::test]
async fn test_multi_bit_multi_chunk_simmax() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;
    let ns = "docs";
    let doc_id = b"doc3";

    // Three chunks with scores 0.7, 0.9, 0.5 — SimMax must return 0.9.
    let chunks = vec![
        VectorIndex::new(1, MULTI8, 0.3, 0.0, 0.01, vec![]), // 1 - 0.3 = 0.7
        VectorIndex::new(1, MULTI8, 0.1, 0.0, 0.05, vec![]), // 1 - 0.1 = 0.9 ← winner
        VectorIndex::new(1, MULTI8, 0.5, 0.0, 0.02, vec![]), // 1 - 0.5 = 0.5
    ];

    upsert_vectors(&db, ns, doc_id, "text", &chunks).await.unwrap();

    let store = DbVectorStore::new(&db, ns).await.unwrap();
    let bytes = store.get_dense_entry(doc_id).await.unwrap().unwrap();

    let (score, error_bound) = score_rkyv_bytes(&bytes, &[], &zero_estimator()).unwrap();
    assert!((score - 0.9).abs() < 1e-6, "SimMax should pick 0.9, got {score}");
    assert!((error_bound - 0.05).abs() < 1e-6);
}

#[tokio::test]
async fn test_upsert_vectors_empty_list_is_noop() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;
    let ns = "docs";

    upsert_vectors(&db, ns, b"doc4", "text", &[]).await.unwrap();

    // An empty list must write nothing to either vector namespace.
    let store = DbVectorStore::new(&db, ns).await.unwrap();
    assert!(store.get_dense_entry(b"doc4").await.unwrap().is_none(), "empty list must write nothing");
}

// Cluster reassignment for SingleBit deletes the stale sparse composite key.
#[tokio::test]
async fn test_sparse_cluster_change_deletes_stale_key() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;
    let ns = "docs";
    let doc_id = b"doc5";

    let vi1 = VectorIndex::new(1, QuantisationStyle::SingleBit, 0.4, 0.0, 0.01, vec![]);
    upsert_vectors(&db, ns, doc_id, "text", &[vi1]).await.unwrap();

    let store = DbVectorStore::new(&db, ns).await.unwrap();
    assert_eq!(store.scan_sparse_cluster(1).await.unwrap().len(), 1);
    assert_eq!(store.scan_sparse_cluster(2).await.unwrap().len(), 0);

    let vi2 = VectorIndex::new(2, QuantisationStyle::SingleBit, 0.35, 0.0, 0.02, vec![]);
    upsert_vectors(&db, ns, doc_id, "text", &[vi2]).await.unwrap();

    assert_eq!(
        store.scan_sparse_cluster(1).await.unwrap().len(),
        0,
        "stale cluster-1 key must be deleted"
    );
    assert_eq!(store.scan_sparse_cluster(2).await.unwrap().len(), 1, "new cluster-2 key must exist");
}

// ── Sparse-meta encoding round-trip ──────────────────────────────────────

#[test]
fn test_encode_decode_sparse_meta_roundtrip() {
    let postings = vec![(10u32, 3u32), (20, 1), (30, 7)];
    let encoded = encode_sparse_meta(0xdead_beef, &postings);
    let decoded = decode_sparse_meta(&encoded).unwrap();
    assert_eq!(
        decoded,
        SparseMeta {
            text_hash: 0xdead_beef,
            postings
        }
    );
}

#[test]
fn a_meta_written_before_chunk_counts_still_decodes() {
    let mut old = 9u64.to_be_bytes().to_vec();
    old.extend_from_slice(&2u16.to_be_bytes());
    old.extend_from_slice(&5u32.to_be_bytes());
    old.extend_from_slice(&8u32.to_be_bytes());
    let m = decode_sparse_meta(&old).unwrap();
    assert_eq!((m.text_hash, m.postings), (9, vec![(5, 0), (8, 0)]));
}

#[test]
fn test_encode_decode_sparse_meta_empty() {
    let encoded = encode_sparse_meta(7, &[]);
    let decoded = decode_sparse_meta(&encoded).unwrap();
    assert!(decoded.postings.is_empty());
    assert_eq!(decoded.text_hash, 7);
}

#[test]
fn test_decode_sparse_meta_rejects_truncated() {
    // Shorter than the 8-byte text hash + 2-byte count header.
    assert!(decode_sparse_meta(&[0x00, 0x01, 0x02]).is_none());
    assert!(decode_sparse_meta(&[0u8; 9]).is_none());
}

#[test]
fn test_decode_sparse_meta_rejects_wrong_length() {
    // Header claims count=1 but only 3 bytes of payload instead of 4.
    let mut bad = vec![0u8; 8]; // text hash
    bad.extend_from_slice(&[0x00, 0x01]); // count = 1
    bad.extend_from_slice(&[0x01, 0x00, 0x00]); // only 3 bytes instead of 4
    assert!(decode_sparse_meta(&bad).is_none());
}

/// The text hash is part of the stored format: pin it, so a change to the
/// hash function (which would make every stored index look stale) fails here.
#[test]
fn text_hash_is_stable() {
    assert_eq!(text_hash(""), 0xcbf2_9ce4_8422_2325);
    assert_eq!(text_hash("a"), 0xaf63_dc4c_8601_ec8c);
}

// ── Durability of the vector write path ───────────────────────────────────

/// The vector payloads go to the vector index WAL: no main-WAL entry and no
/// fsync per write, not flagged as no-WAL (they are recoverable, so the
/// compaction worker's no-WAL flush tick must not flush them), and one
/// fsync makes the whole write durable. The meta is the one WAL-backed
/// write (it must be durable before any key it lists).
///
/// Deliberately service-free: it needs the `semantic-search` feature
/// compiled but no embedding service running.
#[tokio::test]
async fn test_vector_writes_go_to_the_vector_index_wal() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;
    let ns = "docs";
    let doc_id = b"doc-durability";

    let vector_namespaces = [sparse_vectors_ns(ns), sparse_vectors_meta_ns(ns), dense_vectors_ns(ns)];
    // Create the namespaces up front so the baseline below measures the
    // vector writes alone, not namespace setup.
    for name in &vector_namespaces {
        db.namespace(name.clone()).await.unwrap();
        create_vector_namespaces(&db, name).await.unwrap();
    }
    let coordinator = db.coordinator_for_test();

    let before = db.ops_metrics();
    let vi_sparse = VectorIndex::new(3, QuantisationStyle::SingleBit, 0.4, 0.0, 0.02, vec![]);
    let vi_dense = VectorIndex::new(3, MULTI8, 0.3, 0.0, 0.01, vec![]);
    upsert_vectors(&db, ns, doc_id, "text", &[vi_sparse, vi_dense]).await.unwrap();
    let after = db.ops_metrics();

    assert_eq!(
        after.vector_wal_appends - before.vector_wal_appends,
        2,
        "the sparse chunk and the dense entry are logged"
    );
    assert_eq!(after.no_wal_puts - before.no_wal_puts, 0, "nothing takes the unlogged no-WAL path");
    assert_eq!(after.puts - before.puts, 1, "only the meta takes the WAL-backed put path");
    assert_eq!(after.wal_writes - before.wal_writes, 1, "only the meta may add a main-WAL fsync");
    assert_eq!(after.vector_wal_fsyncs - before.vector_wal_fsyncs, 0, "logging does not fsync");
    for name in &vector_namespaces {
        assert!(
            !coordinator.get_store_by_name(name).unwrap().has_unflushed_no_wal_writes(),
            "{name} is flagged as holding unrecoverable no-WAL writes"
        );
    }

    make_vector_writes_durable(&db).await.unwrap();
    let synced = db.ops_metrics();
    assert_eq!(synced.vector_wal_fsyncs - after.vector_wal_fsyncs, 1, "the barrier is one fsync");
    assert_eq!(synced.memtable_flushes, after.memtable_flushes, "and no memtable flush");
}

// ── delete_vector ─────────────────────────────────────────────────────────

#[tokio::test]
async fn test_delete_vector_removes_sparse_and_dense() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;
    let ns = "docs";
    let doc_id = b"del_doc";

    let vi_sb = VectorIndex::new(5, QuantisationStyle::SingleBit, 0.4, 0.0, 0.02, vec![]);
    let vi_mb = VectorIndex::new(5, MULTI8, 0.3, 0.0, 0.01, vec![]);
    upsert_vectors(&db, ns, doc_id, "text", &[vi_sb, vi_mb]).await.unwrap();

    let store = DbVectorStore::new(&db, ns).await.unwrap();
    assert_eq!(store.scan_sparse_cluster(5).await.unwrap().len(), 1, "sparse must exist before delete");
    assert!(store.get_dense_entry(doc_id).await.unwrap().is_some(), "dense must exist before delete");

    delete_vector(&db, ns, doc_id).await.unwrap();

    assert_eq!(store.scan_sparse_cluster(5).await.unwrap().len(), 0, "sparse must be gone after delete");
    assert!(store.get_dense_entry(doc_id).await.unwrap().is_none(), "dense must be gone after delete");

    // Second delete is a no-op (meta is gone).
    delete_vector(&db, ns, doc_id).await.unwrap();
}

#[tokio::test]
async fn test_delete_vector_noop_when_never_indexed() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;
    // Should not panic or return an error.
    delete_vector(&db, "docs", b"ghost").await.unwrap();
}

// ── Namespace isolation ───────────────────────────────────────────────────

// MultiBit-only upsert must not write to the sparse namespace or sparse meta.
#[tokio::test]
async fn test_dense_only_upsert_leaves_sparse_namespace_empty() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;
    let ns = "docs";
    let doc_id = b"dense_only_doc";

    let vi = VectorIndex::new(3, MULTI8, 0.3, 0.0, 0.01, vec![]);
    upsert_vectors(&db, ns, doc_id, "text", &[vi]).await.unwrap();

    let store = DbVectorStore::new(&db, ns).await.unwrap();
    assert!(store.get_dense_entry(doc_id).await.unwrap().is_some(), "dense entry must exist");
    assert!(store.scan_sparse_cluster(3).await.unwrap().is_empty(), "no sparse entry must be written");

    // The meta is still written (it records the text), listing no clusters.
    let meta_ns = db.namespace(sparse_vectors_meta_ns(ns)).await.unwrap();
    let meta = decode_sparse_meta(&meta_ns.get(doc_id.to_vec()).await.unwrap().expect("meta written")).unwrap();
    assert!(meta.postings.is_empty(), "a dense-only document lists no clusters");
    assert_eq!(meta.text_hash, text_hash("text"));
}

// SingleBit-only upsert must not write to the dense namespace.
#[tokio::test]
async fn test_sparse_only_upsert_leaves_dense_namespace_empty() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;
    let ns = "docs";
    let doc_id = b"sparse_only_doc";

    let vi = VectorIndex::new(7, QuantisationStyle::SingleBit, 0.4, 0.0, 0.02, vec![]);
    upsert_vectors(&db, ns, doc_id, "text", &[vi]).await.unwrap();

    let store = DbVectorStore::new(&db, ns).await.unwrap();
    assert_eq!(store.scan_sparse_cluster(7).await.unwrap().len(), 1, "sparse entry must exist");
    assert!(store.get_dense_entry(doc_id).await.unwrap().is_none(), "no dense entry must be written");
}

// ── delete_vector with multiple sparse clusters ───────────────────────────

#[tokio::test]
async fn test_delete_vector_removes_all_sparse_clusters() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;
    let ns = "docs";
    let doc_id = b"multi_cluster_doc";

    let vi_sb1 = VectorIndex::new(1, QuantisationStyle::SingleBit, 0.4, 0.0, 0.02, vec![]);
    let vi_sb2 = VectorIndex::new(2, QuantisationStyle::SingleBit, 0.35, 0.0, 0.03, vec![]);

    upsert_vectors(&db, ns, doc_id, "text", &[vi_sb1, vi_sb2]).await.unwrap();

    let store = DbVectorStore::new(&db, ns).await.unwrap();
    assert_eq!(store.scan_sparse_cluster(1).await.unwrap().len(), 1);
    assert_eq!(store.scan_sparse_cluster(2).await.unwrap().len(), 1);

    delete_vector(&db, ns, doc_id).await.unwrap();

    assert_eq!(store.scan_sparse_cluster(1).await.unwrap().len(), 0, "cluster-1 must be deleted");
    assert_eq!(store.scan_sparse_cluster(2).await.unwrap().len(), 0, "cluster-2 must be deleted");
}
