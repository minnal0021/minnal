//! Dual-style (multi-bit + single-bit) upsert tests.

use super::super::*;
use crate::AsyncDb;
use crate::semantic_search::index::vector_index::{QuantisationStyle, VectorKvStore};
use tempfile::TempDir;

const MULTI8: QuantisationStyle = QuantisationStyle::MultiBit { number_of_bits: 8 };

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

/// One MultiBit + two SingleBit entries stored together. MultiBit goes to
/// dense namespace (fetchable by doc_id); SingleBit goes to sparse namespace
/// (scannable by cluster).
#[tokio::test]
async fn test_dual_style_upsert_routes_to_correct_namespace() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;
    let ns = "docs";
    let doc_id = b"dual_doc";

    let vi_mb = VectorIndex::new(1, MULTI8, 0.3, 0.0, 0.01, vec![]);
    let vi_sb1 = VectorIndex::new(1, QuantisationStyle::SingleBit, 0.4, 0.0, 0.02, vec![]);
    let vi_sb2 = VectorIndex::new(2, QuantisationStyle::SingleBit, 0.35, 0.0, 0.03, vec![]);

    upsert_vectors(&db, ns, doc_id, "text", &[vi_mb, vi_sb1, vi_sb2]).await.unwrap();

    let store = DbVectorStore::new(&db, ns).await.unwrap();

    let dense = store.get_dense_entry(doc_id).await.unwrap();
    assert!(dense.is_some(), "MultiBit entry must be in dense namespace");
    let vis = VectorIndex::list_from_bytes(&dense.unwrap()).unwrap();
    assert_eq!(vis.len(), 1);
    assert!((vis[0].addition_factor - 0.3).abs() < 1e-6);

    let sb1 = store.scan_sparse_cluster(1).await.unwrap();
    assert_eq!(sb1.len(), 1, "SingleBit cluster 1: exactly 1 entry");
    assert_eq!(sb1[0].0, doc_id);

    let sb2 = store.scan_sparse_cluster(2).await.unwrap();
    assert_eq!(sb2.len(), 1, "SingleBit cluster 2: exactly 1 entry");
    assert_eq!(sb2[0].0, doc_id);
}

/// delete_vector removes all three entries (MB + 2 SB) and the sparse meta record.
#[tokio::test]
async fn test_dual_style_delete_removes_all_entries() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;
    let ns = "docs";
    let doc_id = b"dual_del_doc";

    let vi_mb = VectorIndex::new(1, MULTI8, 0.3, 0.0, 0.01, vec![]);
    let vi_sb1 = VectorIndex::new(1, QuantisationStyle::SingleBit, 0.4, 0.0, 0.02, vec![]);
    let vi_sb2 = VectorIndex::new(2, QuantisationStyle::SingleBit, 0.35, 0.0, 0.03, vec![]);

    upsert_vectors(&db, ns, doc_id, "text", &[vi_mb, vi_sb1, vi_sb2]).await.unwrap();

    let store = DbVectorStore::new(&db, ns).await.unwrap();
    assert!(store.get_dense_entry(doc_id).await.unwrap().is_some(), "MB entry exists before delete");
    assert_eq!(store.scan_sparse_cluster(1).await.unwrap().len(), 1, "SB cluster-1 exists before delete");
    assert_eq!(store.scan_sparse_cluster(2).await.unwrap().len(), 1, "SB cluster-2 exists before delete");

    delete_vector(&db, ns, doc_id).await.unwrap();

    assert!(store.get_dense_entry(doc_id).await.unwrap().is_none(), "MB entry must be deleted");
    assert!(store.scan_sparse_cluster(1).await.unwrap().is_empty(), "SB cluster-1 must be deleted");
    assert!(store.scan_sparse_cluster(2).await.unwrap().is_empty(), "SB cluster-2 must be deleted");

    // Second delete is a no-op (sparse meta is gone).
    delete_vector(&db, ns, doc_id).await.unwrap();
}

/// Upserting with a new set replaces stale sparse entries; dense is overwritten.
#[tokio::test]
async fn test_dual_style_upsert_replaces_stale_sparse_entries_on_update() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;
    let ns = "docs";
    let doc_id = b"dual_update_doc";

    // First write: MB cluster 1, SB cluster 1, SB cluster 2.
    let vi_mb = VectorIndex::new(1, MULTI8, 0.3, 0.0, 0.01, vec![]);
    let vi_sb1 = VectorIndex::new(1, QuantisationStyle::SingleBit, 0.4, 0.0, 0.02, vec![]);
    let vi_sb2 = VectorIndex::new(2, QuantisationStyle::SingleBit, 0.35, 0.0, 0.03, vec![]);

    upsert_vectors(&db, ns, doc_id, "text", &[vi_mb, vi_sb1, vi_sb2]).await.unwrap();

    let store = DbVectorStore::new(&db, ns).await.unwrap();
    assert!(store.get_dense_entry(doc_id).await.unwrap().is_some());
    assert_eq!(store.scan_sparse_cluster(2).await.unwrap().len(), 1);

    // Second write: MB updated (overwrites dense), SB now in cluster 3 only (cluster 2 stale).
    let vi_mb2 = VectorIndex::new(1, MULTI8, 0.25, 0.0, 0.01, vec![]);
    let vi_sb3 = VectorIndex::new(3, QuantisationStyle::SingleBit, 0.5, 0.0, 0.04, vec![]);

    upsert_vectors(&db, ns, doc_id, "text", &[vi_mb2, vi_sb3]).await.unwrap();

    let dense = store.get_dense_entry(doc_id).await.unwrap().unwrap();
    let vis = VectorIndex::list_from_bytes(&dense).unwrap();
    assert!((vis[0].addition_factor - 0.25).abs() < 1e-6, "dense must have updated addition_factor");

    assert!(store.scan_sparse_cluster(2).await.unwrap().is_empty(), "stale SB cluster 2 removed");
    assert_eq!(store.scan_sparse_cluster(3).await.unwrap().len(), 1, "new SB cluster 3 present");
    assert!(store.scan_sparse_cluster(1).await.unwrap().is_empty(), "stale SB cluster 1 removed");
}

/// Multiple documents can share the same sparse cluster without interfering.
#[tokio::test]
async fn test_dual_style_multiple_docs_same_cluster_no_interference() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;
    let ns = "docs";
    let doc_a = b"doc_alpha";
    let doc_b = b"doc_beta";

    for doc in [&doc_a[..], &doc_b[..]] {
        let vi_mb = VectorIndex::new(1, MULTI8, 0.3, 0.0, 0.01, vec![]);
        let vi_sb = VectorIndex::new(1, QuantisationStyle::SingleBit, 0.4, 0.0, 0.02, vec![]);
        upsert_vectors(&db, ns, doc, "text", &[vi_mb, vi_sb]).await.unwrap();
    }

    let store = DbVectorStore::new(&db, ns).await.unwrap();
    assert!(store.get_dense_entry(doc_a).await.unwrap().is_some(), "doc_a dense entry exists");
    assert!(store.get_dense_entry(doc_b).await.unwrap().is_some(), "doc_b dense entry exists");
    assert_eq!(store.scan_sparse_cluster(1).await.unwrap().len(), 2, "both docs in SB cluster 1");

    // Delete doc_a; doc_b must still be present.
    delete_vector(&db, ns, doc_a).await.unwrap();

    assert!(store.get_dense_entry(doc_a).await.unwrap().is_none(), "doc_a dense entry deleted");
    assert!(store.get_dense_entry(doc_b).await.unwrap().is_some(), "doc_b dense entry preserved");
    assert_eq!(store.scan_sparse_cluster(1).await.unwrap().len(), 1, "only doc_b remains in SB cluster 1");

    let sb_remaining = store.scan_sparse_cluster(1).await.unwrap();
    assert_eq!(sb_remaining[0].0, doc_b, "remaining sparse entry belongs to doc_b");
}
