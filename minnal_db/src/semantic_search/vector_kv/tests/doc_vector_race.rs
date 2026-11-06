use super::super::*;
use tempfile::TempDir;

/// Search reads through these, and a search can outlive its store's drop.
/// A namespace with no partition or no documents reads as empty, and reading
/// it creates nothing.
#[tokio::test]
async fn reading_a_partition_or_its_documents_creates_no_namespaces() {
    let dir = TempDir::new().unwrap();
    let db = AsyncDb::open_with_config(dir.path().to_path_buf(), crate::db::test_support::create_db_config())
        .await
        .unwrap();
    let created = |what: &str| {
        let names: Vec<String> = db.list_namespaces().into_iter().map(|(n, _)| n).collect();
        assert!(!names.iter().any(|n| n.starts_with("gone_")), "{what}: {names:?}");
    };
    assert!(load_ivf(&db, "gone", 0).await.unwrap().is_none());
    created("load_ivf");
    assert!(count_postings(&db, "gone").await.unwrap().is_empty());
    created("count_postings");
    let store = DbVectorStore::new(&db, "gone").await.unwrap();
    created("DbVectorStore::new");
    assert!(store.scan_sparse_cluster(1).await.unwrap().is_empty());
    assert_eq!(store.get_dense_entry(b"x").await.unwrap(), None);
    assert_eq!(store.get_dense_entries_batch(&[b"x".to_vec()]).await.unwrap(), vec![None]);
    assert_eq!(store.scan_sparse_clusters_batch(&[1]).await.unwrap().iter().count(), 0);
    created("reading through the store");
    db.shutdown().await.unwrap();
}

/// The worker clears a document after its pass, outside any partition's
/// epoch, so this can run after a drop removed the vector namespaces. There
/// is nothing to delete then, and it must not recreate them.
#[tokio::test]
async fn deleting_vectors_does_not_create_their_namespaces() {
    let dir = TempDir::new().unwrap();
    let db = AsyncDb::open_with_config(dir.path().to_path_buf(), crate::db::test_support::create_db_config())
        .await
        .unwrap();
    db.namespace("docs".to_string()).await.unwrap();
    let delta = delete_vector(&db, "docs", b"x").await.unwrap();
    assert!(delta.removed.is_empty());
    let names: Vec<String> = db.list_namespaces().into_iter().map(|(n, _)| n).collect();
    assert!(!names.iter().any(|n| n.starts_with("docs_")), "{names:?}");
    db.shutdown().await.unwrap();
}

/// Only seeding creates the vector namespaces. A write that outlives a drop,
/// past any check its caller made, is skipped and brings nothing back.
#[tokio::test]
async fn a_vector_write_after_a_drop_creates_no_namespaces() {
    let dir = TempDir::new().unwrap();
    let db = AsyncDb::open_with_config(dir.path().to_owned(), crate::support::test_db_config())
        .await
        .unwrap();
    db.namespace("docs".to_string()).await.unwrap();
    init_ivf(&db, "docs", 8).await.unwrap();
    for name in companion_namespaces("docs") {
        assert!(db.existing_namespace(name.clone()).await.unwrap().is_some(), "seeding creates {name}");
        db.remove_namespace(name).await.unwrap();
    }
    // Even with the store itself still there, as after a vector-index drop.
    let vectors = [VectorIndex::new(1, QuantisationStyle::SingleBit, 0.5, 0.0, 0.01, vec![])];
    let delta = upsert_vectors(&db, "docs", b"x", "text", &vectors).await.unwrap();
    assert_eq!(delta, PostingDelta::default());
    let names: Vec<String> = db.list_namespaces().into_iter().map(|(n, _)| n).collect();
    assert!(!names.iter().any(|n| n.starts_with("docs_")), "{names:?}");
    db.shutdown().await.unwrap();
}

/// A delete racing an upsert of the same document's vectors must never leave
/// a sparse cluster key that the document's cluster list does not record —
/// every clean-up finds keys through that list, so such a key is orphaned for
/// good. Unserialised, 260 of 300 rounds of this left one.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn racing_upsert_and_delete_leave_no_orphaned_cluster_keys() {
    let dir = TempDir::new().unwrap();
    let db = std::sync::Arc::new(
        AsyncDb::open_with_config(dir.path().to_owned(), crate::support::test_db_config())
            .await
            .unwrap(),
    );
    db.namespace("docs".to_string()).await.unwrap();
    create_vector_namespaces(&db, "docs").await.unwrap();
    let vectors = |clusters: &[u32]| -> Vec<VectorIndex> {
        let mut out = vec![VectorIndex::new(
            1,
            QuantisationStyle::MultiBit { number_of_bits: 8 },
            1.0,
            0.0,
            0.01,
            vec![],
        )];
        out.extend(
            clusters
                .iter()
                .map(|&c| VectorIndex::new(c, QuantisationStyle::SingleBit, 1.0, 0.0, 0.01, vec![])),
        );
        out
    };
    let meta_ns = db.namespace(sparse_vectors_meta_ns("docs")).await.unwrap();
    let sparse_ns = db.namespace(sparse_vectors_ns("docs")).await.unwrap();
    for round in 0..150 {
        let doc = format!("d{round}").into_bytes();
        upsert_vectors(&db, "docs", &doc, "text", &vectors(&[3])).await.unwrap();
        let (db_a, db_b, doc_a, doc_b) = (db.clone(), db.clone(), doc.clone(), doc.clone());
        let upsert = tokio::spawn(async move { upsert_vectors(&db_a, "docs", &doc_a, "text", &vectors(&[5, 7])).await.unwrap() });
        let delete = tokio::spawn(async move { delete_vector(&db_b, "docs", &doc_b).await.unwrap() });
        upsert.await.unwrap();
        delete.await.unwrap();

        let listed = meta_ns
            .get(doc.clone())
            .await
            .unwrap()
            .and_then(|b| decode_sparse_meta(&b))
            .map(|m| m.cluster_ids())
            .unwrap_or_default();
        for cluster in [3u32, 5, 7] {
            let present = sparse_ns.get(composite_key::encode(cluster, &doc)).await.unwrap().is_some();
            assert!(
                !present || listed.contains(&cluster),
                "round {round}: cluster {cluster} key orphaned (list {listed:?})"
            );
        }
    }
}
