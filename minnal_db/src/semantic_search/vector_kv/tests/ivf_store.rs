use super::super::*;
use std::collections::HashMap;
use tempfile::TempDir;

/// A document can hold more than 65,535 chunks in one posting; its count
/// must come back exact, not saturated (the counts drive splits and merges).
#[test]
fn a_meta_keeps_chunk_counts_above_u16() {
    let m = decode_sparse_meta(&encode_sparse_meta(7, &[(3, 70_000), (9, 2)])).unwrap();
    assert_eq!(m.postings, vec![(3, 70_000), (9, 2)]);
    // The narrower forms still decode.
    let mut six = 7u64.to_be_bytes().to_vec();
    six.extend_from_slice(&1u16.to_be_bytes());
    six.extend_from_slice(&3u32.to_be_bytes());
    six.extend_from_slice(&5u16.to_be_bytes());
    assert_eq!(decode_sparse_meta(&six).unwrap().postings, vec![(3, 5)]);
}

fn centroids() -> HashMap<u32, Vec<f32>> {
    (1..=4u32)
        .map(|id| {
            let mut v = vec![0.0f32; 16];
            v[id as usize] = 1.0;
            (id, v)
        })
        .collect()
}

#[tokio::test]
async fn seed_then_load_round_trips_and_survives_a_reopen() {
    let dir = TempDir::new().unwrap();
    let db = AsyncDb::open_with_config(dir.path().to_owned(), crate::DbConfig::default())
        .await
        .unwrap();
    assert!(!ivf_exists(&db, "ns").await.unwrap());
    assert!(load_ivf(&db, "ns", 7).await.unwrap().is_none());
    seed_ivf(&db, "ns", &centroids()).await.unwrap();
    assert!(ivf_exists(&db, "ns").await.unwrap());
    db.shutdown().await.unwrap();
    drop(db);

    let db = AsyncDb::open_with_config(dir.path().to_owned(), crate::DbConfig::default())
        .await
        .unwrap();
    let ivf = load_ivf(&db, "ns", 7).await.unwrap().unwrap();
    let expected = crate::semantic_search::NamespaceIvf::seeded(&centroids(), 7).unwrap();
    use crate::semantic_search::IvfLayout;
    // Distinct distances to every centroid, so no tie is broken by map order.
    let q: Vec<f32> = (0..16).map(|i| 1.0 / (i as f32 + 1.0)).collect();
    assert_eq!(ivf.probe(std::slice::from_ref(&q), 4), expected.probe(std::slice::from_ref(&q), 4));
    assert_eq!(ivf.rotate(&q), expected.rotate(&q));
    for id in 1..=4 {
        assert_eq!(ivf.centre_of(id), Some(id));
        assert_eq!(ivf.centre(id), expected.centre(id));
    }
    db.shutdown().await.unwrap();
}

#[tokio::test]
async fn an_interrupted_init_is_not_loaded_and_is_redone() {
    let dir = TempDir::new().unwrap();
    let db = AsyncDb::open_with_config(dir.path().to_owned(), crate::DbConfig::default())
        .await
        .unwrap();
    // A crash after a centre, before any posting.
    let centres_ns = db.namespace(ivf_centres_ns("ns")).await.unwrap();
    centres_ns.put(9u32.to_be_bytes().to_vec(), f32s_to_bytes(&[1.0; 16])).await.unwrap();
    assert!(!ivf_exists(&db, "ns").await.unwrap());
    assert!(load_ivf(&db, "ns", 7).await.unwrap().is_none());
    init_ivf(&db, "ns", 16).await.unwrap();
    let ivf = load_ivf(&db, "ns", 7).await.unwrap().unwrap();
    assert_eq!((ivf.postings(), ivf.centres()), (1, 1), "the stray centre was wiped");
    db.shutdown().await.unwrap();
}

/// `init_ivf` is durable on return without any flush: a crash right after
/// it (memtables lost) still finds the root partition.
#[tokio::test]
async fn a_new_partition_survives_a_crash_right_after_it_is_written() {
    let dir = TempDir::new().unwrap();
    let open = || AsyncDb::open_with_config(dir.path().to_owned(), crate::DbConfig::default());
    let db = open().await.unwrap();
    let before = db.ops_metrics();
    init_ivf(&db, "ns", 16).await.unwrap();
    let after = db.ops_metrics();
    assert_eq!(after.no_wal_puts, before.no_wal_puts, "the partition is written WAL-backed");
    assert_eq!(after.memtable_flushes, before.memtable_flushes, "and needs no flush");
    db.crash().await;
    let db = open().await.unwrap();
    let ivf = load_ivf(&db, "ns", 7).await.unwrap().expect("the root partition survived the crash");
    assert_eq!((ivf.postings(), ivf.centres()), (1, 1));
    db.shutdown().await.unwrap();
}

#[tokio::test]
async fn a_new_partition_is_one_root_posting_at_the_zero_centre() {
    use crate::semantic_search::{IvfLayout, PostingOrigin, PostingState};
    let dir = TempDir::new().unwrap();
    let db = AsyncDb::open_with_config(dir.path().to_owned(), crate::DbConfig::default())
        .await
        .unwrap();
    init_ivf(&db, "ns", 16).await.unwrap();
    db.shutdown().await.unwrap();
    drop(db);
    let db = AsyncDb::open_with_config(dir.path().to_owned(), crate::DbConfig::default())
        .await
        .unwrap();
    let ivf = load_ivf(&db, "ns", 7).await.unwrap().unwrap();
    let root = ivf.posting_infos()[&0];
    assert_eq!(
        (root.state, root.origin, root.parent_id, root.centre_id),
        (PostingState::Active, PostingOrigin::Root, None, 0)
    );
    assert_eq!(ivf.centre(0), Some(&[0.0f32; 16][..]));
    let v: Vec<f32> = (0..16).map(|i| i as f32).collect();
    assert_eq!(IvfLayout::route(&ivf, &v), Some(0), "everything routes to the root");
    db.shutdown().await.unwrap();
}

#[test]
fn posting_rows_round_trip_and_pre_m3_rows_still_decode() {
    use crate::semantic_search::{Posting, PostingOrigin, PostingState};
    let p = Posting {
        posting_id: 5,
        routing_centroid: vec![0.5, -1.0, 2.0, 0.0],
        centre_id: 9,
        state: PostingState::Draining,
        parent_id: Some(2),
        origin: PostingOrigin::Split,
    };
    assert_eq!(decode_posting(5, &encode_posting(&p)), Some(p.clone()));
    let root = Posting {
        parent_id: None,
        ..p.clone()
    };
    assert_eq!(decode_posting(5, &encode_posting(&root)), Some(root));
    // A retired posting drops its routing centroid.
    let retired = Posting {
        state: PostingState::Retired,
        ..p.clone()
    };
    let row = encode_posting(&retired);
    assert_eq!(row.len(), 11);
    assert_eq!(
        decode_posting(5, &row),
        Some(Posting {
            routing_centroid: Vec::new(),
            ..retired
        })
    );
    // centre_id ‖ state ‖ centroid, as seeded before M3.
    let mut old = 9u32.to_be_bytes().to_vec();
    old.push(0);
    old.extend(f32s_to_bytes(&[0.5, -1.0, 2.0, 0.0]));
    let o = decode_posting(5, &old).unwrap();
    assert_eq!(
        (o.centre_id, o.state, o.origin, o.parent_id),
        (9, PostingState::Active, PostingOrigin::Seed, None)
    );
    assert_eq!(o.routing_centroid, vec![0.5, -1.0, 2.0, 0.0]);
}

/// The entry counts a search budgets by: counted from the keys at load, and
/// kept equal to a fresh recount by the deltas every vector write returns.
#[tokio::test]
async fn entry_counts_follow_writes_and_are_recounted_at_load() {
    use crate::semantic_search::IvfLayout;
    use crate::semantic_search::index::vector_index::QuantisationStyle;
    let chunks = |postings: &[u32]| -> Vec<VectorIndex> {
        postings
            .iter()
            .map(|&p| VectorIndex::new(p, QuantisationStyle::SingleBit, 0.5, 0.0, 0.01, vec![]))
            .chain([VectorIndex::new(
                u32::MAX,
                QuantisationStyle::MultiBit { number_of_bits: 8 },
                0.5,
                0.0,
                0.01,
                vec![],
            )])
            .collect()
    };
    let counts = |ivf: &crate::semantic_search::NamespaceIvf| (1..=4).map(|p| ivf.posting_entries(p).unwrap()).collect::<Vec<_>>();
    let dir = TempDir::new().unwrap();
    let db = AsyncDb::open_with_config(dir.path().to_owned(), crate::DbConfig::default())
        .await
        .unwrap();
    db.namespace("ns".to_string()).await.unwrap();
    seed_ivf(&db, "ns", &centroids()).await.unwrap();
    let ivf = load_ivf(&db, "ns", 7).await.unwrap().unwrap();
    assert_eq!(counts(&ivf), [0, 0, 0, 0]);
    let recount = |db: AsyncDb| async move {
        let c = count_posting_entries(&db, "ns").await.unwrap();
        (1..=4).map(|p| c.get(&p).copied().unwrap_or(0)).collect::<Vec<_>>()
    };

    // Two chunks in one posting are one entry.
    ivf.apply_delta(&upsert_vectors(&db, "ns", b"a", "t", &chunks(&[1, 1, 2])).await.unwrap());
    ivf.apply_delta(&upsert_vectors(&db, "ns", b"b", "t", &chunks(&[2, 3])).await.unwrap());
    assert_eq!(counts(&ivf), [1, 2, 1, 0]);
    assert_eq!([1, 2, 3].map(|p| ivf.posting_chunks(p)), [2, 2, 1]);
    // A re-embed moves "a" out of 1 and into 4; posting 2 keeps it with
    // two chunks instead of one.
    ivf.apply_delta(&upsert_vectors(&db, "ns", b"a", "u", &chunks(&[2, 2, 4])).await.unwrap());
    assert_eq!(counts(&ivf), [0, 2, 1, 1]);
    assert_eq!([1, 2, 3, 4].map(|p| ivf.posting_chunks(p)), [0, 3, 1, 1]);
    assert_eq!(counts(&ivf), recount(db.clone()).await);
    let from_meta = count_postings(&db, "ns").await.unwrap();
    assert_eq!(
        ivf.counts().snapshot().into_iter().filter(|(_, c)| c.0 > 0).collect::<HashMap<_, _>>(),
        from_meta
    );
    // Deletes, including of a document that has no vectors.
    ivf.apply_delta(&delete_vector(&db, "ns", b"b").await.unwrap());
    ivf.apply_delta(&delete_vector(&db, "ns", b"ghost").await.unwrap());
    ivf.apply_delta(&clear_vectors(&db, "ns", b"a").await.unwrap());
    assert_eq!(counts(&ivf), [0, 0, 0, 0]);
    assert_eq!(counts(&ivf), recount(db.clone()).await);

    // Counted afresh at load, including after a reopen.
    upsert_vectors(&db, "ns", b"c", "t", &chunks(&[3, 4])).await.unwrap();
    upsert_vectors(&db, "ns", b"d", "t", &chunks(&[4])).await.unwrap();
    db.shutdown().await.unwrap();
    drop(db);
    let db = AsyncDb::open_with_config(dir.path().to_owned(), crate::DbConfig::default())
        .await
        .unwrap();
    let ivf = load_ivf(&db, "ns", 7).await.unwrap().unwrap();
    assert_eq!(counts(&ivf), [0, 0, 1, 2]);
    assert_eq!(ivf.total_entries(), 3);
    db.shutdown().await.unwrap();
}

#[test]
fn partition_namespaces_are_companions_and_go_with_the_data() {
    assert_eq!(companion_base("x_ivf_centres"), Some("x"));
    assert_eq!(companion_base("x_ivf_postings"), Some("x"));
    assert_eq!(companion_base("x_ivf_journal"), Some("x"));
    assert_eq!(vector_data_base("x_ivf_centres"), Some("x"));
    assert_eq!(vector_data_base("x_sparse_vector_meta"), Some("x"));
    assert_eq!(vector_data_base("x_dense_vector"), Some("x"));
    assert_eq!(companion_namespaces("x").len(), COMPANION_SUFFIXES.len());
}
