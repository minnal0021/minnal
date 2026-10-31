use std::collections::{BTreeMap, BTreeSet};

use tempfile::TempDir;

use super::*;
use crate::semantic_search::index::vector_index::QuantisationStyle;
use crate::semantic_search::quantisation::rabitq::{index_embedding_rotated, index_embedding_zero_centred};
use crate::vector_kv::{init_ivf, load_ivf, upsert_vectors};

pub(super) const NS: &str = "docs";
const DIM: usize = 16;
const SEED: u64 = 7;

pub(super) fn settings(limit: u64) -> MaintenanceOps {
    MaintenanceOps::split_only(limit)
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

pub(super) async fn handle(db: &AsyncDb) -> PartitionHandle {
    PartitionHandle::new(load_ivf(db, NS, SEED).await.unwrap().expect("a partition"))
}

/// Index a document as the vector worker does: route and write under the
/// routing epoch, against the current snapshot.
async fn index_doc(db: &AsyncDb, h: &PartitionHandle, d: u64, chunks: &[Vec<f32>]) -> Result<(), crate::KVError> {
    index_doc_at(db, h, d, chunks, 1).await
}

/// [`index_doc`] with chunk codes `bits` wide.
async fn index_doc_at(db: &AsyncDb, h: &PartitionHandle, d: u64, chunks: &[Vec<f32>], bits: usize) -> Result<(), crate::KVError> {
    let _epoch = h.read_epoch().await;
    let ivf = h.snapshot();
    let mut codes: Vec<VectorIndex> = chunks
        .iter()
        .map(|c| index_embedding_rotated(&*ivf, c, QuantisationStyle::pass1(bits)).unwrap())
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
pub(super) async fn setup(dir: &TempDir, docs: u64) -> AsyncDb {
    setup_at(dir, docs, 1).await
}

/// [`setup`] with chunk codes `bits` wide.
async fn setup_at(dir: &TempDir, docs: u64, bits: usize) -> AsyncDb {
    let db = open(dir).await;
    init_ivf(&db, NS, DIM).await.unwrap();
    let h = handle(&db).await;
    for d in 0..docs {
        index_doc_at(&db, &h, d, &doc_chunks(d, false), bits).await.unwrap();
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
async fn maintenance_stops_between_operations_when_asked_and_resumes_later() {
    use std::sync::atomic::AtomicBool;
    let dir = TempDir::new().unwrap();
    let db = setup(&dir, 60).await;
    let h = handle(&db).await;
    // Set before the first operation: nothing is split, so server shutdown does
    // not wait for a whole partition to grow.
    let stopped = super::maintain_partition_until(&db, NS, &h, &settings(40), &AtomicBool::new(true))
        .await
        .unwrap();
    assert_eq!(stopped, MaintenanceReport::default());
    assert_eq!(h.snapshot().posting_infos()[&0].state, PostingState::Active);
    check(&db, &h, &expected(60)).await.unwrap();
    // The next request does the work.
    assert!(maintain_partition(&db, NS, &h, &settings(40)).await.unwrap().splits >= 4);
    check(&db, &h, &expected(60)).await.unwrap();
}

#[tokio::test]
async fn an_oversized_posting_splits_until_every_posting_is_within_the_limit() {
    let dir = TempDir::new().unwrap();
    let db = setup(&dir, 60).await; // 180 chunks in the root
    let h = handle(&db).await;
    let made = maintain_partition(&db, NS, &h, &settings(40)).await.unwrap().splits;
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

/// A retired posting is stored without its routing centroid, and the
/// partition still reloads and keeps every chunk.
#[tokio::test]
async fn retired_postings_are_stored_without_their_routing_centroid() {
    let dir = TempDir::new().unwrap();
    let db = setup(&dir, 60).await;
    let h = handle(&db).await;
    assert!(maintain_partition(&db, NS, &h, &settings(40)).await.unwrap().splits >= 4);
    let rows = db.namespace(ivf_postings_ns(NS)).await.unwrap().scan_prefix(Vec::new()).await.unwrap();
    let (mut retired, mut live) = (0, 0);
    for (k, v) in &rows {
        let p = decode_posting(u32::from_be_bytes(k[..4].try_into().unwrap()), v).unwrap();
        if p.state == PostingState::Retired {
            retired += 1;
            assert_eq!(v.len(), 11, "retired posting {} keeps {} bytes", p.posting_id, v.len());
        } else {
            live += 1;
            assert_eq!(p.routing_centroid.len(), DIM, "posting {} lost its centroid", p.posting_id);
        }
    }
    assert!(retired >= 4 && live >= 5, "{retired} retired, {live} live");
    let postings = h.snapshot().postings();
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
        maintain_partition(&db2, NS, &h2, &settings(40)).await.unwrap().splits,
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

/// The operation a crash audit stops part-way.
#[derive(Clone, Copy, Debug)]
enum Audited {
    /// `split_posting` of the root, which must split.
    Split,
    /// `maintain_partition`.
    Maintain,
    /// `rebuild_partition` into four postings.
    Rebuild,
}

/// Run `op`; `false` if it finished but did not do what the audit expects.
async fn run_audited(db: &AsyncDb, h: &PartitionHandle, op: Audited, ops: &MaintenanceOps) -> Result<bool, crate::KVError> {
    Ok(match op {
        Audited::Split => matches!(split_posting(db, NS, h, 0, ops).await?, SplitOutcome::Split { .. }),
        Audited::Maintain => maintain_partition(db, NS, h, ops).await.map(|_| true)?,
        Audited::Rebuild => rebuild::rebuild_partition(db, NS, h, ops, &rebuild_params(4)).await.map(|_| true)?,
    })
}

fn copy_dir(from: &std::path::Path, to: &std::path::Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let dest = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &dest);
        } else {
            std::fs::copy(entry.path(), dest).unwrap();
        }
    }
}

/// Stop `op` before its `n`th write for every `n`, under every flush pattern,
/// each case on a fresh copy of the shut-down database in `template`: after
/// the crash, reopening, recovering and maintaining, every invariant holds and
/// each document holds `want` chunks. Returns the writes of an uncrashed run
/// and every failure.
///
/// The crash point is thread-local, so the cases are spread over a few
/// threads, each with its own current-thread runtime: thread `t` takes every
/// `n ≡ t` (mod threads) until a run finishes without crashing.
fn crash_audit(template: &std::path::Path, want: &BTreeMap<Vec<u8>, usize>, op: Audited, ops: MaintenanceOps) -> (usize, Vec<String>) {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).clamp(1, 8);
    let results: Vec<(usize, Vec<String>)> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|t| {
                scope.spawn(move || {
                    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
                    rt.block_on(crash_cases(template, want, op, ops, t, threads))
                })
            })
            .collect();
        workers.into_iter().map(|w| w.join().unwrap()).collect()
    });
    let writes = results.iter().map(|(w, _)| *w).min().unwrap();
    (writes, results.into_iter().flat_map(|(_, f)| f).collect())
}

async fn crash_cases(
    template: &std::path::Path,
    want: &BTreeMap<Vec<u8>, usize>,
    op: Audited,
    ops: MaintenanceOps,
    first: usize,
    step: usize,
) -> (usize, Vec<String>) {
    let mut failures = Vec::new();
    let mut writes_seen = usize::MAX;
    for flushed in [Flushed::None, Flushed::Sparse, Flushed::Meta, Flushed::All] {
        for n in (first..).step_by(step) {
            let dir = TempDir::new().unwrap();
            copy_dir(template, dir.path());
            let db = open(&dir).await;
            let h = handle(&db).await;
            crate::vector_kv::crash_point::arm(n);
            let result = run_audited(&db, &h, op, &ops).await;
            let crashed = crate::vector_kv::crash_point::disarm();
            if !crashed {
                assert!(result.unwrap(), "{op:?} did not do its job");
                writes_seen = writes_seen.min(n);
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
            recover(&db, NS, &h, &ops).await.unwrap();
            maintain_partition(&db, NS, &h, &ops).await.unwrap();
            if let Err(e) = check(&db, &h, want).await {
                failures.push(format!("{op:?}, crash before write {n}, flushed {flushed:?}: {e}"));
            }
            db.shutdown().await.unwrap();
            if !crashed {
                break;
            }
        }
    }
    (writes_seen, failures)
}

/// A crash before every write of a split, under several flush patterns: after
/// reopening and recovering, every invariant holds and a later split finishes
/// the job.
#[tokio::test]
async fn a_split_survives_a_crash_before_any_write() {
    let template = TempDir::new().unwrap();
    setup(&template, 20).await.shutdown().await.unwrap(); // 60 chunks
    let (writes, failures) = crash_audit(template.path(), &expected(20), Audited::Split, settings(40));
    assert!(writes > 20, "the scenario must cover a whole split ({writes} writes)");
    assert!(failures.is_empty(), "{} failure(s):\n  {}", failures.len(), failures.join("\n  "));
}

/// A split moves every document of the posting, but its per-document writes
/// go to the vector index WAL, one fsync per batch: the main WAL sees only the
/// operation's own records (journal, centres, postings), however many
/// documents move.
#[tokio::test]
async fn a_split_writes_nothing_per_document_to_the_main_wal() {
    let dir = TempDir::new().unwrap();
    let docs = 40;
    let db = setup(&dir, docs).await;
    let h = handle(&db).await;
    let before = db.ops_metrics();
    assert!(matches!(
        split_posting(&db, NS, &h, 0, &settings(40)).await.unwrap(),
        SplitOutcome::Split { .. }
    ));
    let after = db.ops_metrics();
    let main = after.wal_writes - before.wal_writes;
    let logged = after.vector_wal_fsyncs - before.vector_wal_fsyncs;
    assert!(main < docs / 2, "{main} main-WAL fsyncs for {docs} moved documents");
    // A copy and a delete pass, each a sync per batch of 16 (plus reassign's).
    assert!(logged >= 2 * docs.div_ceil(16), "{logged} vector-index-WAL fsyncs");
    check(&db, &h, &expected(docs)).await.unwrap();
    db.shutdown().await.unwrap();
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
            tokio::spawn(async move { maintain_partition(&db, NS, &h, &settings(40)).await.unwrap() })
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
        maintain_partition(&db, NS, &h, &settings(40)).await.unwrap();
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
    for (state, op) in [
        (
            JournalState::Copied,
            Op::Split {
                posting: 3,
                centres: [7, 8],
                children: [9, 10],
            },
        ),
        (JournalState::Published, Op::Merge { posting: 4, kept: 11 }),
        (
            JournalState::Planned,
            Op::Reassign {
                parent: 3,
                parent_centre: 3,
                children: [9, 10],
                scope: vec![9, 10, 1, 2, 5],
            },
        ),
        (
            JournalState::Published,
            Op::Rebuild {
                first_centre: 20,
                first_posting: 30,
                postings: 4,
                drained: vec![1, 2, 5],
            },
        ),
    ] {
        let r = Record { state, op };
        assert_eq!(Record::decode(&r.encode()), Some(r));
    }
    // A split record is 23 bytes, as before reassignment and merges existed.
    let split = Record {
        state: JournalState::Planned,
        op: Op::Split {
            posting: 0,
            centres: [1, 2],
            children: [1, 2],
        },
    };
    assert_eq!(split.encode().len(), 23);
    assert_eq!(Record::decode(&[1, 0, 9]), None);
    assert_eq!(Record::decode(&[1, 1, 1, 0, 0, 0, 1]), None, "a merge needs two ids");
    assert_eq!(Record::decode(&[1, 2, 1, 0, 0, 0, 1]), None, "a reassignment needs four");
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
    maintain_partition(&db, NS, &h, &settings(30)).await.unwrap();
    done.store(true, std::sync::atomic::Ordering::Release);
    let searches = searcher.await.unwrap();
    assert!(searches > 1, "{searches} search(es)");
    assert!(h.snapshot().postings() > 4);
    db.shutdown().await.unwrap();
}

/// Diagnostic over an existing database (`MINNAL_E2E_DB`, namespace
/// `MINNAL_E2E_NS`): every key in an active posting, counts equal to a recount
/// of the stored codes, no unfinished split; and the inflation of a few codes.
#[tokio::test]
#[ignore]
async fn e2e_partition_invariants() {
    let Ok(path) = std::env::var("MINNAL_E2E_DB") else { return };
    let ns = std::env::var("MINNAL_E2E_NS").unwrap_or_else(|_| "e2e".into());
    let db = AsyncDb::open_with_config(std::path::PathBuf::from(path), crate::DbConfig::default())
        .await
        .unwrap();
    let ivf = load_ivf(&db, &ns, crate::semantic_search::cluster::DEFAULT_ROTATION_SEED)
        .await
        .unwrap()
        .unwrap();
    assert!(records(&db, &ns).await.unwrap().is_empty(), "unfinished maintenance operations");
    let keys = recount_chunks(&db, &ns).await.unwrap();
    for (p, n) in &keys {
        assert_eq!(ivf.posting_infos()[p].state, PostingState::Active, "posting {p}");
        assert_eq!(ivf.posting_chunks(*p), *n, "posting {p}");
    }
    let counted: u64 = ivf.counts().snapshot().values().map(|c| c.1).sum();
    assert_eq!(keys.values().sum::<u64>(), counted);
    eprintln!("{} postings, {} chunks: invariants hold", keys.len(), counted);
    db.shutdown().await.unwrap();
}

// ── Reassignment and merges (M3b) ──

/// Maintenance with merges at `merge_limit` and reassignment over `range`
/// neighbours.
pub(super) fn full(split_limit: u64, merge_limit: u64, range: usize) -> MaintenanceOps {
    MaintenanceOps {
        merge_limit,
        reassign_range: range,
        ..MaintenanceOps::split_only(split_limit)
    }
}

async fn delete_doc(db: &AsyncDb, h: &PartitionHandle, d: u64) {
    let delta = crate::vector_kv::delete_vector(db, NS, &doc_id(d)).await.unwrap();
    h.snapshot().apply_delta(&delta);
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Scenario {
    /// Topics 0 and 1 split the root at 40 chunks; then topics 2 and 3 arrive
    /// and fill both halves past the limit, so the next splits leave chunks in
    /// a neighbour that a new centre beats.
    Reassign,
    /// The root splits into a topic-0 and a topic-1 posting; then eight of the
    /// ten topic-1 documents are deleted, leaving 6 chunks to merge.
    Merge,
}

const DELETED: [u64; 8] = [1, 3, 5, 7, 9, 11, 13, 15];

/// The state just before the maintenance a scenario is about, and the chunks
/// each document must end with.
pub(super) async fn prepare(dir: &TempDir, scenario: Scenario) -> (AsyncDb, BTreeMap<Vec<u8>, usize>) {
    prepare_at(dir, scenario, 1).await
}

/// [`prepare`] with chunk codes `bits` wide.
async fn prepare_at(dir: &TempDir, scenario: Scenario, bits: usize) -> (AsyncDb, BTreeMap<Vec<u8>, usize>) {
    let db = setup_at(dir, 20, bits).await;
    let h = handle(&db).await;
    maintain_partition(&db, NS, &h, &settings(40)).await.unwrap();
    let mut want = expected(20);
    match scenario {
        Scenario::Reassign => {
            for d in 20..40 {
                index_doc_at(&db, &h, d, &doc_chunks(d, true), bits).await.unwrap();
                want.insert(doc_id(d), 3);
            }
        }
        Scenario::Merge => {
            for d in DELETED {
                delete_doc(&db, &h, d).await;
                want.insert(doc_id(d), 0);
            }
        }
    }
    db.flush_namespaces(crate::vector_kv::companion_namespaces(NS)).await.unwrap();
    db.shutdown().await.unwrap();
    drop((db, h));
    (open(dir).await, want)
}

fn scenario_ops(scenario: Scenario) -> MaintenanceOps {
    match scenario {
        Scenario::Reassign => full(40, 0, 64),
        Scenario::Merge => full(40, 12, 0),
    }
}

/// Squared distance of every chunk's reconstruction from its posting's centre,
/// summed: what reassignment lowers.
async fn spread(db: &AsyncDb, h: &PartitionHandle) -> f64 {
    let ivf = h.snapshot();
    let mut total = 0.0;
    for (k, v) in db.namespace(sparse_vectors_ns(NS)).await.unwrap().scan_prefix(Vec::new()).await.unwrap() {
        let (p, _) = composite_key::decode(&k).unwrap();
        let c = ivf.rotated_posting_centre(p).unwrap();
        for code in VectorIndex::list_from_bytes(&v).unwrap() {
            total += sq_dist(&reconstruct(&ivf, &code).unwrap(), c);
        }
    }
    total
}

#[tokio::test]
async fn reassign_moves_chunks_to_a_nearer_posting() {
    for bits in [1, 2, 4] {
        reassign_moves_chunks_to_a_nearer_posting_at(bits).await;
    }
}

async fn reassign_moves_chunks_to_a_nearer_posting_at(bits: usize) {
    let mut spreads = Vec::new();
    for range in [0, 64] {
        let dir = TempDir::new().unwrap();
        let (db, want) = prepare_at(&dir, Scenario::Reassign, bits).await;
        let h = handle(&db).await;
        let report = maintain_partition(&db, NS, &h, &full(40, 0, range)).await.unwrap();
        assert!(report.splits >= 2, "{report:?}");
        let c = h.counters().read();
        assert_eq!(c.reassigns, c.splits, "one reassignment per split: {c:?}");
        if range == 64 {
            assert!(c.reassign_moves > 0 && c.reassign_candidates >= c.reassign_moves, "{c:?}");
        }
        check(&db, &h, &want).await.unwrap();
        spreads.push(spread(&db, &h).await);
        db.shutdown().await.unwrap();
    }
    assert!(
        spreads[1] < spreads[0],
        "{bits} bits: reassigning over the neighbours brings chunks nearer their centres: {spreads:?}"
    );
}

#[tokio::test]
async fn deletes_merge_small_postings_and_never_the_last() {
    let dir = TempDir::new().unwrap();
    let (db, want) = prepare(&dir, Scenario::Merge).await;
    let h = handle(&db).await;
    let before = h.snapshot().postings();
    let report = maintain_partition(&db, NS, &h, &scenario_ops(Scenario::Merge)).await.unwrap();
    assert_eq!(report.merges, 1, "{report:?}");
    assert_eq!(h.snapshot().postings(), before - 1);
    check(&db, &h, &want).await.unwrap();
    // Every document gone: the last posting stays, empty.
    for d in 0..20 {
        delete_doc(&db, &h, d).await;
    }
    maintain_partition(&db, NS, &h, &scenario_ops(Scenario::Merge)).await.unwrap();
    let ivf = h.snapshot();
    assert_eq!(ivf.postings(), 1);
    assert_eq!(ivf.active_centres().len(), 1);
    check(&db, &h, &(0..20).map(|d| (doc_id(d), 0)).collect()).await.unwrap();
    db.shutdown().await.unwrap();
}

/// A crash before every write of a reassignment or a merge, under several flush
/// patterns: after reopening and recovering, every invariant holds.
#[tokio::test]
async fn reassign_and_merge_survive_a_crash_before_any_write() {
    let mut failures = Vec::new();
    for (scenario, bits) in [(Scenario::Reassign, 1), (Scenario::Merge, 1), (Scenario::Reassign, 4)] {
        let template = TempDir::new().unwrap();
        let (db, want) = prepare_at(&template, scenario, bits).await;
        db.shutdown().await.unwrap();
        let (writes, f) = crash_audit(template.path(), &want, Audited::Maintain, scenario_ops(scenario));
        assert!(writes > 10, "{scenario:?} at {bits} bits must cover a whole run ({writes} writes)");
        failures.extend(f.into_iter().map(|e| format!("{scenario:?} at {bits} bits: {e}")));
    }
    assert!(failures.is_empty(), "{} failure(s):\n  {}", failures.len(), failures.join("\n  "));
}

/// Re-embeds and deletes racing reassignment and merges: no chunk is lost or
/// orphaned and every count stays exact.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn writes_racing_reassign_and_merge_lose_nothing() {
    for round in 0..4u64 {
        let dir = TempDir::new().unwrap();
        let (db, mut want) = prepare(&dir, Scenario::Reassign).await;
        let h = std::sync::Arc::new(handle(&db).await);
        let ops = full(40, 12, 64);
        let maintainer = {
            let (db, h) = (db.clone(), std::sync::Arc::clone(&h));
            tokio::spawn(async move { maintain_partition(&db, NS, &h, &ops).await.unwrap() })
        };
        let deleted: Vec<u64> = (round % 2..40).step_by(3).collect();
        let writer = {
            let (db, h, deleted) = (db.clone(), std::sync::Arc::clone(&h), deleted.clone());
            tokio::spawn(async move {
                for d in (round % 3..40).step_by(4) {
                    index_doc(&db, &h, d, &doc_chunks(d, d < 20)).await.unwrap();
                }
                for d in deleted {
                    delete_doc(&db, &h, d).await;
                }
            })
        };
        writer.await.unwrap();
        maintainer.await.unwrap();
        maintain_partition(&db, NS, &h, &ops).await.unwrap();
        for d in deleted {
            want.insert(doc_id(d), 0);
        }
        check(&db, &h, &want).await.unwrap_or_else(|e| panic!("round {round}: {e}"));
        db.shutdown().await.unwrap();
    }
}

/// Searches running while chunks are reassigned and postings merged never miss
/// a document.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_search_during_reassign_and_merge_misses_nothing() {
    use crate::semantic_search::ProbeSettings;
    use crate::semantic_search::service::{SemanticSearchConfig, search};
    for scenario in [Scenario::Reassign, Scenario::Merge] {
        let dir = TempDir::new().unwrap();
        let (db, want) = prepare(&dir, scenario).await;
        let live = want.values().filter(|&&n| n > 0).count();
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
                    assert_eq!(docs.len(), live, "{scenario:?}: search {searches} found {} documents", docs.len());
                    searches += 1;
                    if finished {
                        return searches;
                    }
                }
            })
        };
        let report = maintain_partition(&db, NS, &h, &scenario_ops(scenario)).await.unwrap();
        done.store(true, std::sync::atomic::Ordering::Release);
        let searches = searcher.await.unwrap();
        assert!(searches > 1, "{searches} search(es)");
        assert!(report.splits + report.merges > 0, "{report:?}");
        db.shutdown().await.unwrap();
    }
}

#[test]
fn the_gate_skips_moves_within_the_code_error() {
    use crate::semantic_search::index::vector_index::QuantisationStyle;
    let code = |error_bound: f32| VectorIndex {
        cluster_id: 0,
        centre_id: 0,
        quantisation_style: QuantisationStyle::SingleBit,
        addition_factor: 0.0,
        scaling_factor: 0.0,
        error_bound,
        packed_vector: vec![],
    };
    let (a, b) = ([0.0f32, 0.0], [1.0f32, 0.0]);
    let x = [0.4f32, 0.0]; // D(x, a) = 0.16, D(x, b) = 0.36: a wins by 0.2; ‖a − b‖ = 1
    assert!(nearer(&x, &a, &b, &code(0.5), false), "no gate: the nearer centre wins");
    assert!(nearer(&x, &a, &b, &code(0.09), true), "0.2 > 2 · 0.09");
    assert!(!nearer(&x, &a, &b, &code(0.11), true), "0.2 < 2 · 0.11: too close to call");
    assert!(!nearer(&x, &b, &a, &code(0.0), false), "the farther centre never wins");
}

#[tokio::test]
async fn the_gate_moves_fewer_chunks() {
    let mut moves = Vec::new();
    for gate in [false, true] {
        let dir = TempDir::new().unwrap();
        let (db, want) = prepare(&dir, Scenario::Reassign).await;
        let h = handle(&db).await;
        let ops = MaintenanceOps {
            skip_uncertain_moves: gate,
            ..scenario_ops(Scenario::Reassign)
        };
        maintain_partition(&db, NS, &h, &ops).await.unwrap();
        check(&db, &h, &want).await.unwrap();
        let c = h.counters().read();
        moves.push((c.reassign_candidates, c.reassign_moves));
        db.shutdown().await.unwrap();
    }
    assert!(moves[1].0 <= moves[0].0 && moves[1].1 <= moves[0].1, "{moves:?}");
}

/// A split that only isolates a few outliers would be merged straight back,
/// and the merge's re-routing would push the chunks back into the big half:
/// splits and merges took turns hundreds of times on SciFact. Such a split is
/// refused, so maintenance ends at once with the posting left whole.
#[tokio::test]
async fn a_split_that_the_next_merge_would_undo_is_refused() {
    let dir = TempDir::new().unwrap();
    let db = open(&dir).await;
    init_ivf(&db, NS, DIM).await.unwrap();
    let h = handle(&db).await;
    // 20 documents of one repeated chunk, then two whose chunks sit on another
    // topic: the only division 2-means can make isolates those 6.
    for d in 0..20 {
        index_doc(&db, &h, d, &vec![chunk(0, 1); 3]).await.unwrap();
    }
    for d in 20..22 {
        index_doc(&db, &h, d, &(0..3).map(|j| chunk(3, d * 10 + j)).collect::<Vec<_>>())
            .await
            .unwrap();
    }
    let ops = full(60, 6, 64); // 66 chunks: over the limit; the outliers are 6
    let report = maintain_partition(&db, NS, &h, &ops).await.unwrap();
    assert_eq!(
        report,
        MaintenanceReport::default(),
        "nothing to do: the only split isolates the outliers"
    );
    assert!(h.is_unsplittable(0, 66));
    check(&db, &h, &expected(22)).await.unwrap();
    // Once the posting's size changes (it is not retried at the same size),
    // with a merge limit below the outliers, the same split goes ahead.
    index_doc(&db, &h, 22, &vec![chunk(0, 1); 3]).await.unwrap();
    let report = maintain_partition(&db, NS, &h, &full(60, 5, 64)).await.unwrap();
    assert_eq!(report.splits, 1, "{report:?}");
    check(&db, &h, &expected(23)).await.unwrap();
    db.shutdown().await.unwrap();
}

/// Diagnostic over an existing database (`MINNAL_E2E_DB`, `MINNAL_E2E_NS`,
/// `MINNAL_E2E_SPLIT_LIMIT`): every active posting over the split limit, and the
/// sides a split's 2-means would give it.
#[tokio::test]
#[ignore]
async fn e2e_oversized_postings() {
    let Ok(path) = std::env::var("MINNAL_E2E_DB") else { return };
    let ns = std::env::var("MINNAL_E2E_NS").unwrap_or_else(|_| "e2e".into());
    let limit: u64 = std::env::var("MINNAL_E2E_SPLIT_LIMIT").map_or(32, |v| v.parse().unwrap());
    let db = AsyncDb::open_with_config(std::path::PathBuf::from(path), crate::DbConfig::default())
        .await
        .unwrap();
    let ivf = load_ivf(&db, &ns, crate::semantic_search::cluster::DEFAULT_ROTATION_SEED)
        .await
        .unwrap()
        .unwrap();
    let mut over = 0;
    for (&p, info) in ivf.posting_infos() {
        let chunks = ivf.posting_chunks(p);
        if info.state != PostingState::Active || chunks <= limit {
            continue;
        }
        over += 1;
        let entries = posting_entries(&db, &ns, p).await.unwrap();
        let points: Vec<Vec<f32>> = entries.iter().flat_map(|(_, c)| c.iter().filter_map(|c| reconstruct(&ivf, c))).collect();
        let mut rng = SplitRng::new(u64::from(p).rotate_left(32) ^ chunks);
        let sides = balanced_two_means(&points, &SplitParams::default(), &mut rng).map(|r| {
            let ones = r.labels.iter().filter(|&&l| l == 1).count();
            (r.labels.len() - ones, ones)
        });
        let docs = entries.len();
        eprintln!("posting {p}: {chunks} chunks in {docs} documents, 2-means sides {sides:?}");
    }
    eprintln!("{over} active posting(s) over {limit}");
    db.shutdown().await.unwrap();
}

// ── Rebuild (M4) ──

fn rebuild_params(postings: usize) -> rebuild::RebuildParams {
    rebuild::RebuildParams {
        postings: Some(postings),
        threads: 1,
    }
}

/// Four topics (40 documents), split while only two had arrived: topics 2 and 3
/// share postings with 0 and 1. A rebuild into four gives each its own.
#[tokio::test]
async fn a_rebuild_fits_postings_to_the_chunks_they_hold() {
    for bits in [1, 2, 4] {
        a_rebuild_fits_postings_to_the_chunks_they_hold_at(bits).await;
    }
}

async fn a_rebuild_fits_postings_to_the_chunks_they_hold_at(bits: usize) {
    let dir = TempDir::new().unwrap();
    let (db, want) = prepare_at(&dir, Scenario::Reassign, bits).await;
    let h = handle(&db).await;
    let before = spread(&db, &h).await;
    let ops = full(1000, 0, 0); // nothing splits or merges afterwards
    let report = rebuild::rebuild_partition(&db, NS, &h, &ops, &rebuild_params(4)).await.unwrap();
    assert_eq!((report.chunks, report.postings_fitted, report.postings_after), (120, 4, 4), "{report:?}");
    assert_eq!(report.chunks_moved, 120);
    let c = h.counters().read();
    assert_eq!((c.rebuilds, c.rebuild_moves, c.chunk_moves()), (1, 120, 0), "{c:?}");
    check(&db, &h, &want).await.unwrap();
    let ivf = h.snapshot();
    let routes: BTreeSet<u32> = (0..4).map(|t| ivf.route(&chunk(t, 1)).unwrap()).collect();
    assert_eq!(routes.len(), 4, "each topic routed to its own posting");
    for (&id, info) in ivf.posting_infos() {
        if info.state == PostingState::Active {
            assert_eq!(info.origin, PostingOrigin::Rebuild, "posting {id}");
        }
    }
    let after = spread(&db, &h).await;
    assert!(after < before, "chunks nearer their centres: {before} -> {after}");

    // Reopened, the partition and counts are the same.
    db.shutdown().await.unwrap();
    drop((db, h));
    let db = open(&dir).await;
    let h = handle(&db).await;
    check(&db, &h, &want).await.unwrap();
    db.shutdown().await.unwrap();
}

/// A crash before every write of a rebuild, under several flush patterns:
/// after reopening and recovering, every invariant holds.
#[tokio::test]
async fn a_rebuild_survives_a_crash_before_any_write() {
    for bits in [1, 4] {
        let template = TempDir::new().unwrap();
        let (db, want) = prepare_at(&template, Scenario::Reassign, bits).await;
        db.shutdown().await.unwrap();
        let (writes, failures) = crash_audit(template.path(), &want, Audited::Rebuild, full(40, 3, 8));
        assert!(writes > 40, "{bits} bits: the scenario must cover a whole rebuild ({writes} writes)");
        assert!(
            failures.is_empty(),
            "{bits} bits: {} failure(s):\n  {}",
            failures.len(),
            failures.join("\n  ")
        );
    }
}

/// Re-embeds and deletes racing a rebuild: no chunk is lost or orphaned and
/// every count stays exact.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn writes_racing_a_rebuild_lose_nothing() {
    for round in 0..4u64 {
        let dir = TempDir::new().unwrap();
        let (db, mut want) = prepare(&dir, Scenario::Reassign).await;
        let h = std::sync::Arc::new(handle(&db).await);
        let ops = full(40, 3, 8);
        let rebuilder = {
            let (db, h) = (db.clone(), std::sync::Arc::clone(&h));
            tokio::spawn(async move { rebuild::rebuild_partition(&db, NS, &h, &ops, &rebuild_params(4)).await.unwrap() })
        };
        let deleted: Vec<u64> = (round % 2..40).step_by(3).collect();
        let writer = {
            let (db, h, deleted) = (db.clone(), std::sync::Arc::clone(&h), deleted.clone());
            tokio::spawn(async move {
                for d in (round % 3..40).step_by(4) {
                    index_doc(&db, &h, d, &doc_chunks(d, d < 20)).await.unwrap();
                }
                for d in deleted {
                    delete_doc(&db, &h, d).await;
                }
            })
        };
        writer.await.unwrap();
        rebuilder.await.unwrap();
        maintain_partition(&db, NS, &h, &ops).await.unwrap();
        for d in deleted {
            want.insert(doc_id(d), 0);
        }
        check(&db, &h, &want).await.unwrap_or_else(|e| panic!("round {round}: {e}"));
        db.shutdown().await.unwrap();
    }
}

/// Searches running while the partition is rebuilt never miss a document.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_search_during_a_rebuild_misses_nothing() {
    use crate::semantic_search::ProbeSettings;
    use crate::semantic_search::service::{SemanticSearchConfig, search};
    let dir = TempDir::new().unwrap();
    let (db, want) = prepare(&dir, Scenario::Reassign).await;
    let live = want.len();
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
                assert_eq!(docs.len(), live, "search {searches} found {} documents", docs.len());
                searches += 1;
                if finished {
                    return searches;
                }
            }
        })
    };
    let report = rebuild::rebuild_partition(&db, NS, &h, &full(1000, 0, 0), &rebuild_params(4))
        .await
        .unwrap();
    done.store(true, std::sync::atomic::Ordering::Release);
    let searches = searcher.await.unwrap();
    assert!(searches > 1, "{searches} search(es)");
    assert_eq!(report.postings_fitted, 4, "{report:?}");
    db.shutdown().await.unwrap();
}

// ── Review fixes (2026-10-17) ───────────────────────────────────────────────

/// A transient error after a split's `Planned` record, then an in-process
/// retry, then a restart. The retry must not run on top of the failed
/// attempt's record: the failed attempt's children ids are free in memory, so
/// the retry reused them, and recovery at the next start then "abandoned" the
/// stale record by retiring the retry's live postings, chunks and all.
#[tokio::test]
async fn a_failed_split_retried_in_process_survives_a_restart() {
    // Through both entry points: the maintenance loop and a single split.
    for single in [false, true] {
        let dir = TempDir::new().unwrap();
        let docs = 20; // 60 chunks over a limit of 40
        let db = setup(&dir, docs).await;
        let h = handle(&db).await;
        let ops = settings(40);
        let run = || async {
            if single {
                split_posting(&db, NS, &h, 0, &ops).await.map(|_| ())
            } else {
                maintain_partition(&db, NS, &h, &ops).await.map(|_| ())
            }
        };
        // The record goes through, the next write (the first centre) fails.
        crate::vector_kv::crash_point::arm(1);
        assert!(run().await.is_err(), "the injected error");
        crate::vector_kv::crash_point::disarm();
        run().await.unwrap();
        check(&db, &h, &expected(docs)).await.unwrap_or_else(|e| panic!("single={single}: {e}"));

        db.shutdown().await.unwrap();
        drop(db);
        let db = open(&dir).await;
        let h = handle(&db).await;
        recover(&db, NS, &h, &ops).await.unwrap();
        check(&db, &h, &expected(docs))
            .await
            .unwrap_or_else(|e| panic!("single={single}, after a restart: {e}"));
        db.shutdown().await.unwrap();
    }
}

/// A write that routed by the snapshot before a publish (to the posting being
/// drained) and lands after it must still be drained: the publish waits for
/// every write that entered the epoch before it, so the drain lists the
/// posting's documents only once no such write is left.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_write_routed_before_a_publish_is_drained_with_the_rest() {
    let dir = TempDir::new().unwrap();
    let docs = 20;
    let db = setup(&dir, docs).await;
    let h = std::sync::Arc::new(handle(&db).await);
    // A writer enters and routes by the current snapshot: everything to the root.
    let entered = h.read_epoch().await;
    let routed = h.snapshot();
    let splitting = tokio::spawn({
        let (db, h) = (db.clone(), std::sync::Arc::clone(&h));
        async move { maintain_partition(&db, NS, &h, &settings(40)).await }
    });
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    // ... and only now writes.
    let chunks = doc_chunks(docs, false);
    let mut codes: Vec<VectorIndex> = chunks
        .iter()
        .map(|c| index_embedding_rotated(&*routed, c, QuantisationStyle::SingleBit).unwrap())
        .collect();
    codes.push(index_embedding_zero_centred(
        &*routed,
        &chunks[0],
        QuantisationStyle::MultiBit { number_of_bits: 8 },
    ));
    let delta = upsert_vectors(&db, NS, &doc_id(docs), "text", &codes).await.unwrap();
    routed.apply_delta(&delta);
    drop(entered);
    splitting.await.unwrap().unwrap();
    check(&db, &h, &expected(docs + 1)).await.unwrap();
    db.shutdown().await.unwrap();
}

/// A crash leaves a split `Published` (parent draining, children active and
/// partly filled). A rebuild asked for before startup recovery has run must
/// not lose chunks: it would drain and retire the children, and recovery would
/// then finish the split by copying the parent's chunks into retired postings.
#[tokio::test]
async fn a_rebuild_before_recovery_loses_nothing() {
    let dir = TempDir::new().unwrap();
    let docs = 20;
    let db = setup(&dir, docs).await;
    let h = handle(&db).await;
    let ops = settings(40);
    // Planned record, two centres, two children, the parent draining, the
    // Published record: seven writes; the eighth (the first copy) fails.
    crate::vector_kv::crash_point::arm(7);
    assert!(split_posting(&db, NS, &h, 0, &ops).await.is_err());
    crate::vector_kv::crash_point::disarm();
    let found = records(&db, NS).await.unwrap();
    assert!(
        matches!(
            found.as_slice(),
            [(
                _,
                Record {
                    state: JournalState::Published,
                    op: Op::Split { .. }
                }
            )]
        ),
        "set-up: a published split is pending: {found:?}"
    );
    db.crash().await;

    let db = open(&dir).await;
    let h = handle(&db).await;
    let params = rebuild::RebuildParams {
        postings: Some(2),
        threads: 1,
    };
    rebuild::rebuild_partition(&db, NS, &h, &ops, &params).await.unwrap();
    recover(&db, NS, &h, &ops).await.unwrap();
    check(&db, &h, &expected(docs)).await.unwrap();
    db.shutdown().await.unwrap();
}

/// A crash between recording a split's reassignment and deleting the split's
/// record leaves both. Recovery must run that reassignment once.
#[tokio::test]
async fn recovery_runs_a_recorded_reassignment_once() {
    let docs = 20;
    let ops = settings(40);
    for n in 8..400 {
        let dir = TempDir::new().unwrap();
        let db = setup(&dir, docs).await;
        let h = handle(&db).await;
        crate::vector_kv::crash_point::arm(n);
        let result = split_posting(&db, NS, &h, 0, &ops).await;
        crate::vector_kv::crash_point::disarm();
        assert!(result.is_err(), "no crash point left a split and its reassignment recorded");
        let kinds: Vec<u8> = records(&db, NS)
            .await
            .unwrap()
            .iter()
            .map(|(_, r)| match r.op {
                Op::Split { .. } => KIND_SPLIT,
                Op::Reassign { .. } => KIND_REASSIGN,
                _ => u8::MAX,
            })
            .collect();
        if kinds != [KIND_SPLIT, KIND_REASSIGN] {
            db.crash().await;
            continue;
        }
        db.crash().await;
        let db = open(&dir).await;
        let h = handle(&db).await;
        recover(&db, NS, &h, &ops).await.unwrap();
        assert_eq!(
            h.counters().read().reassigns,
            1,
            "crash after write {n}: the reassignment ran more than once"
        );
        check(&db, &h, &expected(docs)).await.unwrap();
        db.shutdown().await.unwrap();
        return;
    }
    panic!("no crash point left a split and its reassignment recorded");
}

/// A posting that was found unsplittable at some size and is later split
/// (or merged away) leaves no mark behind: the marks would otherwise grow with
/// every retired posting for the life of the process.
#[tokio::test]
async fn a_retired_postings_marks_are_dropped() {
    let dir = TempDir::new().unwrap();
    let db = setup(&dir, 20).await;
    let h = handle(&db).await;
    h.mark_unsplittable(0, 999);
    h.mark_unmergeable(0, 999);
    maintain_partition(&db, NS, &h, &settings(40)).await.unwrap();
    assert_eq!(h.snapshot().posting_infos()[&0].state, PostingState::Retired, "set-up: the root split");
    assert!(!h.is_unsplittable(0, 999), "a retired posting's unsplittable mark stayed");
    assert!(!h.is_unmergeable(0, 999), "a retired posting's unmergeable mark stayed");
    db.shutdown().await.unwrap();
}

/// A search builds Pass-1 estimators only for the centres its probed codes
/// were encoded against, not for every centre the namespace ever had (splits
/// and rebuilds add centres and never remove them).
#[tokio::test]
async fn a_search_builds_estimators_only_for_the_centres_its_codes_use() {
    use crate::semantic_search::ProbeSettings;
    use crate::semantic_search::service::{SemanticSearchConfig, search, test_hooks};
    const SEARCH_NS: &str = "estimators_for_used_centres_only";
    let dir = TempDir::new().unwrap();
    let db = setup(&dir, 10).await; // every code in the root, against centre 0
    let base = handle(&db).await.snapshot();
    let unused: Vec<(u32, Vec<f32>)> = (1..=2000u32).map(|i| (i, chunk(1, 5000 + u64::from(i)))).collect();
    let ivf = base.evolve(&unused, &[]).unwrap();
    let config = SemanticSearchConfig {
        embedding_dim: DIM,
        number_of_bits_for_dense_quantisation: 8,
        probe: ProbeSettings::fixed(4096),
        first_pass_sparse_search_top_k: 1000,
        top_k_results: 1000,
        ..SemanticSearchConfig::default()
    };
    let store = crate::vector_kv::DbVectorStore::new(&db, NS).await.unwrap();
    let q = chunk(0, 999);
    let found = search(
        &config,
        SEARCH_NS,
        &ivf,
        std::slice::from_ref(&q),
        &q,
        &store,
        None::<fn(&[u8]) -> bool>,
        None,
    )
    .await
    .unwrap();
    assert_eq!(found.len(), 10, "every document found");
    assert_eq!(test_hooks::estimator_centres(SEARCH_NS), 1, "estimators built for unused centres");
    db.shutdown().await.unwrap();
}
