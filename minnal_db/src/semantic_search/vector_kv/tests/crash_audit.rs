//! Crash audit of the vector write paths (design doc M0-2).
//!
//! Each scenario runs a sequence of vector-index operations with a crash injected
//! before its Nth write (`crash_point`), for every N, and under several flush
//! patterns: before the crash, some vector namespaces may already have been
//! flushed by their own memtable filling up, independently of the others. The
//! database is then dropped without shutting down, reopened, and put through what
//! production runs after a restart (the worker draining the queue), and every
//! invariant is checked.

use super::super::*;
use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::pin::Pin;
use tempfile::TempDir;

const NS: &str = "docs";
type Step = for<'a> fn(&'a AsyncDb) -> Pin<Box<dyn Future<Output = Result<(), crate::KVError>> + 'a>>;

/// Which vector namespaces had flushed (by their own memtable filling)
/// before the crash.
#[derive(Clone, Copy, Debug)]
enum Flushed {
    None,
    SparseOnly,
    MetaOnly,
    DenseOnly,
    All,
}

fn hash(text: &str) -> u64 {
    text.bytes()
        .fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
}

/// The vectors a text embeds to, deterministically: a dense entry tagged with
/// the text, and two chunks whose clusters depend on the text, so a re-embed
/// with new text moves them.
fn vectors_for_text(text: &str) -> Vec<VectorIndex> {
    let h = hash(text);
    let tag = (h % 997) as f32;
    vec![
        VectorIndex::new(0, QuantisationStyle::MultiBit { number_of_bits: 8 }, tag, 0.0, 0.01, vec![]),
        VectorIndex::new(10 + (h % 5) as u32, QuantisationStyle::SingleBit, tag, 0.0, 0.01, vec![]),
        VectorIndex::new(20 + ((h >> 8) % 5) as u32, QuantisationStyle::SingleBit, tag, 0.0, 0.01, vec![]),
    ]
}

fn clusters_for_text(text: &str) -> BTreeSet<u32> {
    vectors_for_text(text).iter().skip(1).map(|v| v.cluster_id).collect()
}

async fn open(dir: &TempDir) -> AsyncDb {
    let db = AsyncDb::open_with_config(dir.path().to_owned(), crate::support::test_db_config())
        .await
        .unwrap();
    db.namespace(NS.to_string()).await.unwrap();
    create_vector_namespaces(&db, NS).await.unwrap();
    db
}

// ── What the document store does ──

/// A document write, in `DocStore::put`'s order: the embed entry first, then
/// the document (WAL).
async fn put_doc(db: &AsyncDb, doc: &[u8], text: &str) -> Result<(), crate::KVError> {
    enqueue_embed(db, NS, doc, text).await?;
    crash_point!();
    db.namespace(NS.to_string()).await?.put(doc.to_vec(), text.as_bytes().to_vec()).await
}

/// A document delete, in `DocStore::delete`'s order: the tombstone and the
/// vectors first (`clear_vectors`), then the document (WAL).
async fn delete_doc(db: &AsyncDb, doc: &[u8]) -> Result<(), crate::KVError> {
    clear_vectors(db, NS, doc).await?;
    crash_point!();
    db.namespace(NS.to_string()).await?.delete(doc.to_vec()).await
}

/// The vector worker draining the queue: each entry is checked against its
/// document (no write is in progress here: the scenarios run one step at a
/// time), then carried out; a rewritten entry is carried out on the next pass.
async fn worker_pass(db: &AsyncDb) -> Result<(), crate::KVError> {
    for _ in 0..4 {
        let entries = list_queue_entries(db).await?;
        if entries.is_empty() {
            break;
        }
        for entry in entries {
            let current = db.namespace(NS.to_string()).await?.get(entry.doc_id_bytes.clone()).await?;
            let current = current.map(|b| String::from_utf8(b).unwrap());
            if !check_entry_against_document(db, &entry, current.as_deref()).await? {
                continue;
            }
            match entry.kind {
                QueueEntryKind::Embed => finish_embed(db, &entry, &vectors_for_text(&entry.text)).await?,
                QueueEntryKind::Clear => process_clear(db, &entry.namespace, &entry.doc_id_bytes).await?,
            };
        }
    }
    Ok(())
}

// ── Invariants ──

/// Everything that must hold after recovery: the vector index matches what
/// the store holds. A document present has exactly its current text's
/// vectors; one absent has none; no queue work is left; no sparse key exists
/// that its document's meta does not list.
async fn check(db: &AsyncDb, docs: &[&[u8]]) -> Result<(), String> {
    let parent = db.namespace(NS.to_string()).await.unwrap();
    let mut texts: BTreeMap<Vec<u8>, Option<String>> = BTreeMap::new();
    for doc in docs {
        let text = parent.get(doc.to_vec()).await.unwrap().map(|b| String::from_utf8(b).unwrap());
        texts.insert(doc.to_vec(), text);
    }
    let expected = &texts;
    let queue = list_queue_entries(db).await.unwrap();
    if !queue.is_empty() {
        return Err(format!("queue not drained: {} entr(y/ies)", queue.len()));
    }
    let mut keys: BTreeMap<Vec<u8>, BTreeSet<u32>> = BTreeMap::new();
    for (key, _) in db.namespace(sparse_vectors_ns(NS)).await.unwrap().iter().await.unwrap() {
        let (cluster, doc) = composite_key::decode(&key).unwrap();
        keys.entry(doc.to_vec()).or_default().insert(cluster);
    }
    let meta_ns = db.namespace(sparse_vectors_meta_ns(NS)).await.unwrap();
    for (doc, text) in expected {
        let meta: Option<BTreeSet<u32>> = meta_ns
            .get(doc.clone())
            .await
            .unwrap()
            .and_then(|b| decode_sparse_meta(&b))
            .map(|m| m.cluster_ids().into_iter().collect());
        let present = keys.remove(doc).unwrap_or_default();
        let dense = DbVectorStore::new(db, NS).await.unwrap().get_dense_entry(doc).await.unwrap();
        let dense_tag = dense.map(|raw| VectorIndex::access_list(&raw).unwrap().iter().next().unwrap().addition_factor());
        match text.as_deref() {
            Some(t) => {
                let want = clusters_for_text(t);
                let tag = vectors_for_text(t)[0].addition_factor;
                if present != want {
                    return Err(format!("sparse keys {present:?}, expected {want:?} for text {t:?}"));
                }
                if meta.as_ref() != Some(&want) {
                    return Err(format!("meta {meta:?}, expected {want:?}"));
                }
                if dense_tag != Some(tag) {
                    return Err(format!("dense tag {dense_tag:?}, expected {tag} (stale or missing vectors)"));
                }
            }
            None => {
                if !present.is_empty() || meta.is_some() || dense_tag.is_some() {
                    return Err(format!(
                        "vectors of a deleted document remain: keys {present:?}, meta {meta:?}, dense {dense_tag:?}"
                    ));
                }
            }
        }
    }
    if let Some((doc, clusters)) = keys.into_iter().next() {
        return Err(format!("sparse keys {clusters:?} for unknown document {doc:?}"));
    }
    Ok(())
}

/// Run `scenario` after `setup` with a crash before every one of its writes
/// in turn, under every flush pattern, and check `expected` after recovery.
/// Returns the failures, one line each.
async fn audit(setup: Step, scenario: Step, docs: &[&[u8]]) -> Vec<String> {
    let mut failures = Vec::new();
    for flushed in [Flushed::None, Flushed::SparseOnly, Flushed::MetaOnly, Flushed::DenseOnly, Flushed::All] {
        for n in 0.. {
            let dir = TempDir::new().unwrap();
            let db = open(&dir).await;
            setup(&db).await.unwrap();
            make_vector_writes_durable(&db).await.unwrap();
            crash_point::arm(n);
            let result = scenario(&db).await;
            let crashed = crash_point::disarm();
            if !crashed {
                result.unwrap();
            }
            let names: Vec<String> = match flushed {
                Flushed::None => vec![],
                Flushed::SparseOnly => vec![sparse_vectors_ns(NS)],
                Flushed::MetaOnly => vec![sparse_vectors_meta_ns(NS)],
                Flushed::DenseOnly => vec![dense_vectors_ns(NS)],
                Flushed::All => vec![sparse_vectors_ns(NS), sparse_vectors_meta_ns(NS), dense_vectors_ns(NS)],
            };
            db.flush_namespaces(names).await.unwrap();
            db.crash().await; // crash: unflushed no-WAL writes are lost
            // Restart: the worker drains the queue the crash left. Nothing
            // scans the documents.
            let db = open(&dir).await;
            worker_pass(&db).await.unwrap();
            if let Err(e) = check(&db, docs).await {
                failures.push(format!("crash before write {n}, flushed {flushed:?}: {e}"));
            }
            db.shutdown().await.unwrap();
            if !crashed {
                break;
            }
        }
    }
    failures
}

fn assert_clean(name: &str, failures: Vec<String>) {
    assert!(failures.is_empty(), "{name}: {} failure(s):\n  {}", failures.len(), failures.join("\n  "));
}

fn nothing(_: &AsyncDb) -> Pin<Box<dyn Future<Output = Result<(), crate::KVError>> + '_>> {
    Box::pin(async { Ok(()) })
}

fn index_old(db: &AsyncDb) -> Pin<Box<dyn Future<Output = Result<(), crate::KVError>> + '_>> {
    Box::pin(async move {
        put_doc(db, b"x", "old text").await?;
        worker_pass(db).await
    })
}

#[tokio::test]
async fn first_index_survives_a_crash_anywhere() {
    let failures = audit(nothing, index_old, &[b"x"]).await;
    assert_clean("first index", failures);
}

#[tokio::test]
async fn reembed_survives_a_crash_anywhere() {
    fn reembed(db: &AsyncDb) -> Pin<Box<dyn Future<Output = Result<(), crate::KVError>> + '_>> {
        Box::pin(async move {
            put_doc(db, b"x", "new text").await?;
            worker_pass(db).await
        })
    }
    let failures = audit(index_old, reembed, &[b"x"]).await;
    assert_clean("re-embed", failures);
}

#[tokio::test]
async fn delete_survives_a_crash_anywhere() {
    fn delete(db: &AsyncDb) -> Pin<Box<dyn Future<Output = Result<(), crate::KVError>> + '_>> {
        Box::pin(async move {
            delete_doc(db, b"x").await?;
            worker_pass(db).await
        })
    }
    let failures = audit(index_old, delete, &[b"x"]).await;
    assert_clean("delete", failures);
}

/// A re-embed in flight when the document is deleted: the worker's write
/// lands after the delete.
#[tokio::test]
async fn delete_racing_a_reembed_survives_a_crash_anywhere() {
    fn race(db: &AsyncDb) -> Pin<Box<dyn Future<Output = Result<(), crate::KVError>> + '_>> {
        Box::pin(async move {
            put_doc(db, b"x", "new text").await?;
            let in_flight = get_queue_entry(db, NS, b"x").await?.expect("queued");
            delete_doc(db, b"x").await?;
            finish_embed(db, &in_flight, &vectors_for_text("new text")).await?;
            worker_pass(db).await
        })
    }
    let failures = audit(index_old, race, &[b"x"]).await;
    assert_clean("delete racing a re-embed", failures);
}

/// A bulk re-index enqueues from a snapshot; its entry can land after the
/// document's delete was fully processed. The worker must not index a
/// document that no longer exists.
#[tokio::test]
async fn a_stale_entry_for_a_deleted_document_survives_a_crash_anywhere() {
    fn stale(db: &AsyncDb) -> Pin<Box<dyn Future<Output = Result<(), crate::KVError>> + '_>> {
        Box::pin(async move {
            delete_doc(db, b"x").await?;
            worker_pass(db).await?;
            enqueue_embed(db, NS, b"x", "old text").await?; // the re-index's stale snapshot
            worker_pass(db).await
        })
    }
    let failures = audit(index_old, stale, &[b"x"]).await;
    assert_clean("a stale entry for a deleted document", failures);
}

/// A re-embed and a delete back to back, each cut short by the crash at every
/// point: the queue alone must bring the index back to the document.
#[tokio::test]
async fn reembed_then_delete_survives_a_crash_anywhere() {
    fn both(db: &AsyncDb) -> Pin<Box<dyn Future<Output = Result<(), crate::KVError>> + '_>> {
        Box::pin(async move {
            put_doc(db, b"x", "new text").await?;
            delete_doc(db, b"x").await?;
            put_doc(db, b"y", "other text").await?;
            worker_pass(db).await
        })
    }
    let failures = audit(index_old, both, &[b"x", b"y"]).await;
    assert_clean("re-embed then delete", failures);
}
