use std::collections::{BTreeMap, BTreeSet};

use tempfile::TempDir;

use super::*;
use crate::semantic_search::index::vector_index::QuantisationStyle;
use crate::semantic_search::quantisation::rabitq::{index_embedding_rotated, index_embedding_zero_centred};
use crate::vector_kv::{init_ivf, load_ivf, upsert_vectors};

const NS: &str = "docs";
const DIM: usize = 16;
const SEED: u64 = 7;

fn settings(limit: u64) -> SplitSettings {
    SplitSettings {
        split_limit: limit,
        params: SplitParams::default(),
    }
}

/// A deterministic unit vector near one of four directions (`topic`).
fn chunk(topic: usize, n: u64) -> Vec<f32> {
    let mut s = n.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (topic as u64);
    let mut v: Vec<f32> = (0..DIM)
        .map(|i| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let noise = ((s >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 0.6;
            noise + if i % 4 == topic % 4 { 1.0 } else { 0.0 }
        })
        .collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter_mut().for_each(|x| *x /= norm);
    v
}

/// The chunks of document `d`: three, on topic `d % 2` (`+ 2` when `alt`).
fn doc_chunks(d: u64, alt: bool) -> Vec<Vec<f32>> {
    let topic = (d % 2) as usize + if alt { 2 } else { 0 };
    (0..3).map(|j| chunk(topic, d * 10 + j)).collect()
}

fn doc_id(d: u64) -> Vec<u8> {
    d.to_be_bytes().to_vec()
}

async fn open(dir: &TempDir) -> AsyncDb {
    let db = AsyncDb::open_with_config(dir.path().to_owned(), crate::support::test_db_config())
        .await
        .unwrap();
    db.namespace(NS.to_string()).await.unwrap();
    db
}

async fn handle(db: &AsyncDb) -> PartitionHandle {
    PartitionHandle::new(load_ivf(db, NS, SEED).await.unwrap().expect("a partition"))
}

/// Index a document as the vector worker does: route and write under the
/// routing epoch, against the current snapshot.
async fn index_doc(db: &AsyncDb, h: &PartitionHandle, d: u64, chunks: &[Vec<f32>]) -> Result<(), crate::KVError> {
    let _epoch = h.read_epoch().await;
    let ivf = h.snapshot();
    let mut codes: Vec<VectorIndex> = chunks
        .iter()
        .map(|c| index_embedding_rotated(&*ivf, c, QuantisationStyle::SingleBit).unwrap())
        .collect();
    codes.push(index_embedding_zero_centred(
        &*ivf,
        &chunks[0],
        QuantisationStyle::MultiBit { number_of_bits: 8 },
    ));
    let delta = upsert_vectors(db, NS, &doc_id(d), "text", &codes).await?;
    ivf.apply_delta(&delta);
    Ok(())
}

/// A namespace of `docs` documents (three chunks each), all in the root.
async fn setup(dir: &TempDir, docs: u64) -> AsyncDb {
    let db = open(dir).await;
    init_ivf(&db, NS, DIM).await.unwrap();
    let h = handle(&db).await;
    for d in 0..docs {
        index_doc(&db, &h, d, &doc_chunks(d, false)).await.unwrap();
    }
    db.flush_namespaces(crate::vector_kv::companion_namespaces(NS)).await.unwrap();
    db
}

/// Everything that must hold once no split is in progress: no journal record;
/// every key in an active posting, listed in its document's meta with its exact
/// chunk count; each document holding exactly `chunks[doc]` chunks; and the
/// counts equal to what the keys say.
async fn check(db: &AsyncDb, h: &PartitionHandle, chunks: &BTreeMap<Vec<u8>, usize>) -> Result<(), String> {
    if !records(db, NS).await.unwrap().is_empty() {
        return Err("journal not empty".into());
    }
    let ivf = h.snapshot();
    let mut keys: BTreeMap<Vec<u8>, BTreeMap<u32, usize>> = BTreeMap::new();
    for (k, v) in db.namespace(sparse_vectors_ns(NS)).await.unwrap().scan_prefix(Vec::new()).await.unwrap() {
        let (p, doc) = composite_key::decode(&k).unwrap();
        let state = ivf.posting_infos().get(&p).map(|i| i.state);
        if state != Some(PostingState::Active) {
            return Err(format!("a key in posting {p}, which is {state:?}"));
        }
        let n = VectorIndex::list_from_bytes(&v).unwrap().len();
        keys.entry(doc.to_vec()).or_default().insert(p, n);
    }
    let meta_ns = db.namespace(sparse_vectors_meta_ns(NS)).await.unwrap();
    let mut counts: HashMap<u32, (u64, u64)> = HashMap::new();
    for (doc, &want) in chunks {
        let have = keys.remove(doc).unwrap_or_default();
        let total: usize = have.values().sum();
        if total != want {
            return Err(format!("document {doc:?} holds {total} chunks, expected {want}: {have:?}"));
        }
        let meta: BTreeMap<u32, usize> = meta_ns
            .get(doc.clone())
            .await
            .unwrap()
            .and_then(|b| decode_sparse_meta(&b))
            .map(|m| m.postings.into_iter().map(|(p, n)| (p, n as usize)).collect())
            .unwrap_or_default();
        if meta != have {
            return Err(format!("document {doc:?}: meta {meta:?}, keys {have:?}"));
        }
        for (p, n) in have {
            let c = counts.entry(p).or_default();
            c.0 += 1;
            c.1 += n as u64;
        }
    }
    if let Some((doc, ps)) = keys.into_iter().next() {
        return Err(format!("keys {ps:?} of unknown document {doc:?}"));
    }
    let loaded: HashMap<u32, (u64, u64)> = ivf.counts().snapshot().into_iter().filter(|(_, c)| c.0 > 0).collect();
    if loaded != counts {
        return Err(format!("counts {loaded:?}, keys say {counts:?}"));
    }
    Ok(())
}

fn expected(docs: u64) -> BTreeMap<Vec<u8>, usize> {
    (0..docs).map(|d| (doc_id(d), 3)).collect()
}

#[tokio::test]
async fn an_oversized_posting_splits_until_every_posting_is_within_the_limit() {
    let dir = TempDir::new().unwrap();
    let db = setup(&dir, 60).await; // 180 chunks in the root
    let h = handle(&db).await;
    let made = split_oversized(&db, NS, &h, &settings(40)).await.unwrap();
    assert!(made >= 4, "{made} split(s)");
    let ivf = h.snapshot();
    for (&id, info) in ivf.posting_infos() {
        match info.state {
            PostingState::Active => assert!(ivf.posting_chunks(id) <= 40, "posting {id}: {}", ivf.posting_chunks(id)),
            PostingState::Retired => assert_eq!(ivf.posting_chunks(id), 0),
            PostingState::Draining => panic!("posting {id} left draining"),
        }
    }
    assert_eq!(ivf.posting_infos()[&0].state, PostingState::Retired, "the root was split");
    check(&db, &h, &expected(60)).await.unwrap();
    // The two topics no longer share a posting.
    assert_ne!(ivf.route(&chunk(0, 1)), ivf.route(&chunk(1, 1)));

    // The partition survives a reopen, with the same counts.
    let postings = ivf.postings();
    db.shutdown().await.unwrap();
    drop((db, h));
    let db = open(&dir).await;
    let h = handle(&db).await;
    assert_eq!(h.snapshot().postings(), postings);
    check(&db, &h, &expected(60)).await.unwrap();
    db.shutdown().await.unwrap();
}

#[tokio::test]
async fn a_posting_within_the_limit_or_of_identical_chunks_is_left_alone() {
    let dir = TempDir::new().unwrap();
    let db = setup(&dir, 10).await;
    let h = handle(&db).await;
    assert_eq!(split_posting(&db, NS, &h, 0, &settings(40)).await.unwrap(), SplitOutcome::NotNeeded);
    // 30 identical chunks cannot be divided.
    for d in 10..20 {
        index_doc(&db, &h, d, &vec![chunk(0, 1); 3]).await.unwrap();
    }
    let dir2 = TempDir::new().unwrap();
    let db2 = open(&dir2).await;
    init_ivf(&db2, NS, DIM).await.unwrap();
    let h2 = handle(&db2).await;
    for d in 0..20 {
        index_doc(&db2, &h2, d, &vec![chunk(0, 1); 3]).await.unwrap();
    }
    assert_eq!(split_posting(&db2, NS, &h2, 0, &settings(40)).await.unwrap(), SplitOutcome::Unsplittable);
    assert_eq!(
        split_oversized(&db2, NS, &h2, &settings(40)).await.unwrap(),
        0,
        "not retried at the same size"
    );
    check(&db2, &h2, &(0..20).map(|d| (doc_id(d), 3)).collect()).await.unwrap();
    db.shutdown().await.unwrap();
    db2.shutdown().await.unwrap();
}

#[derive(Clone, Copy, Debug)]
enum Flushed {
    None,
    Sparse,
    Meta,
    All,
}

/// A crash before every write of a split, under several flush patterns: after
/// reopening and recovering, every invariant holds and a later split finishes
/// the job.
#[tokio::test]
async fn a_split_survives_a_crash_before_any_write() {
    let mut failures = Vec::new();
    let mut writes_seen = 0;
    for flushed in [Flushed::None, Flushed::Sparse, Flushed::Meta, Flushed::All] {
        for n in 0.. {
            let dir = TempDir::new().unwrap();
            let db = setup(&dir, 20).await; // 60 chunks
            let h = handle(&db).await;
            crate::vector_kv::crash_point::arm(n);
            let result = split_posting(&db, NS, &h, 0, &settings(40)).await;
            let crashed = crate::vector_kv::crash_point::disarm();
            if !crashed {
                assert!(matches!(result.unwrap(), SplitOutcome::Split { .. }));
                writes_seen = n;
            }
            let names = match flushed {
                Flushed::None => vec![],
                Flushed::Sparse => vec![sparse_vectors_ns(NS)],
                Flushed::Meta => vec![sparse_vectors_meta_ns(NS)],
                Flushed::All => crate::vector_kv::companion_namespaces(NS),
            };
            db.flush_namespaces(names).await.unwrap();
            drop(h);
            db.crash().await; // unflushed no-WAL writes are lost
            let db = open(&dir).await;
            let h = handle(&db).await;
            recover(&db, NS, &h).await.unwrap();
            split_oversized(&db, NS, &h, &settings(40)).await.unwrap();
            if let Err(e) = check(&db, &h, &expected(20)).await {
                failures.push(format!("crash before write {n}, flushed {flushed:?}: {e}"));
            }
            db.shutdown().await.unwrap();
            if !crashed {
                break;
            }
        }
    }
    assert!(writes_seen > 20, "the scenario must cover a whole split ({writes_seen} writes)");
    assert!(failures.is_empty(), "{} failure(s):\n  {}", failures.len(), failures.join("\n  "));
}

/// Re-embeds and deletes racing a split: no chunk is lost or orphaned, nothing
/// is written to a draining or retired posting, and every count stays exact.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn writes_racing_a_split_lose_nothing() {
    for round in 0..5u64 {
        let dir = TempDir::new().unwrap();
        let db = setup(&dir, 40).await;
        let h = std::sync::Arc::new(handle(&db).await);
        let splitter = {
            let (db, h) = (db.clone(), std::sync::Arc::clone(&h));
            tokio::spawn(async move { split_oversized(&db, NS, &h, &settings(40)).await.unwrap() })
        };
        let writer = {
            let (db, h) = (db.clone(), std::sync::Arc::clone(&h));
            tokio::spawn(async move {
                for d in (round % 2..40).step_by(2) {
                    index_doc(&db, &h, d, &doc_chunks(d, true)).await.unwrap();
                }
                for d in [1u64, 3, 5] {
                    let delta = crate::vector_kv::delete_vector(&db, NS, &doc_id(d)).await.unwrap();
                    h.snapshot().apply_delta(&delta);
                }
            })
        };
        writer.await.unwrap();
        splitter.await.unwrap();
        split_oversized(&db, NS, &h, &settings(40)).await.unwrap();
        let mut want = expected(40);
        for d in [1u64, 3, 5] {
            want.insert(doc_id(d), 0);
        }
        check(&db, &h, &want).await.unwrap_or_else(|e| panic!("round {round}: {e}"));
        db.shutdown().await.unwrap();
    }
}

#[test]
fn journal_records_round_trip() {
    let r = SplitRecord {
        state: JournalState::Copied,
        posting: 3,
        centres: [7, 8],
        children: [9, 10],
    };
    assert_eq!(SplitRecord::decode(&r.encode()), Some(r));
    assert_eq!(SplitRecord::decode(&[1, 0, 9]), None);
}

/// Searches running while postings split never miss a document: every
/// full-scan search returns every document, before, during and after.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_search_during_a_split_misses_nothing() {
    use crate::semantic_search::ProbeSettings;
    use crate::semantic_search::service::{SemanticSearchConfig, search};
    let dir = TempDir::new().unwrap();
    let db = setup(&dir, 60).await;
    let h = std::sync::Arc::new(handle(&db).await);
    let config = SemanticSearchConfig {
        embedding_dim: DIM,
        number_of_bits_for_dense_quantisation: 8,
        probe: ProbeSettings::fixed(4096),
        first_pass_sparse_search_top_k: 1000,
        top_k_results: 1000,
        ..SemanticSearchConfig::default()
    };
    let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let searcher = {
        let (db, h, done) = (db.clone(), std::sync::Arc::clone(&h), std::sync::Arc::clone(&done));
        tokio::spawn(async move {
            let store = crate::vector_kv::DbVectorStore::new(&db, NS).await.unwrap();
            let q = chunk(0, 999);
            let mut searches = 0;
            loop {
                let finished = done.load(std::sync::atomic::Ordering::Acquire);
                let found = {
                    let _epoch = h.read_epoch().await;
                    let ivf = h.snapshot();
                    search(&config, NS, &*ivf, std::slice::from_ref(&q), &q, &store, None::<fn(&[u8]) -> bool>, None)
                        .await
                        .unwrap()
                };
                let docs: BTreeSet<Vec<u8>> = found.into_iter().map(|r| r.document_id).collect();
                assert_eq!(docs.len(), 60, "search {searches} found {} documents", docs.len());
                searches += 1;
                if finished {
                    return searches;
                }
            }
        })
    };
    split_oversized(&db, NS, &h, &settings(30)).await.unwrap();
    done.store(true, std::sync::atomic::Ordering::Release);
    let searches = searcher.await.unwrap();
    assert!(searches > 1, "{searches} search(es)");
    assert!(h.snapshot().postings() > 4);
    db.shutdown().await.unwrap();
}
