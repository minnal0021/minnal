//! The field-index worker: a write queues its field-index changes and returns;
//! the worker, the index checkpoint, a writer past the queue's cap and
//! `sync_field_indexes` apply them (`kv_store/index_queue.rs`).

use super::*;
use crate::db::workers::index_checkpoint::IndexCheckpointTarget;
use crate::field_index::{IndexValue, IndexValueType};
use crate::kv_store::index_queue::{QUEUE_CAP, QueuedChange};
use crate::kv_store::index_set::{ExtractorFn, IndexApplyWake};

/// A worker that never runs: changes stay queued until something else
/// applies them.
struct NeverRuns;

impl IndexApplyWake for NeverRuns {
    fn wake(&self) {}
}

fn worker_never_runs(db: &Database, ns: NamespaceId) {
    db.get_store(ns).unwrap().set_index_apply_wake(Some(Arc::new(NeverRuns)));
}

fn active_keys(db: &Database, ns: NamespaceId) -> Vec<Vec<u8>> {
    let mut keys = db.query_keys(ns, "status = \"active\"").unwrap().keys;
    keys.sort();
    keys
}

fn progress(db: &Database, ns: NamespaceId, field_id: crate::field_index::FieldId) -> crate::FieldQueueProgress {
    db.index_health(ns)
        .unwrap()
        .into_iter()
        .find(|h| h.field_id == field_id)
        .and_then(|h| h.apply_queue)
        .expect("an active field reports its queue")
}

/// With a worker running, a write leaves its field-index change queued and
/// returns: queries do not see it until it is applied, and the queue's
/// progress says what is pending. The barrier applies it.
#[test]
fn a_write_returns_with_its_field_index_change_queued() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    let ns = DEFAULT_NAMESPACE_ID;
    let field_id = activate_status_index(&db, ns);
    worker_never_runs(&db, ns);

    db.put(b"doc:1", br#"{"status":"active"}"#).unwrap();
    assert!(active_keys(&db, ns).is_empty(), "the change was applied on the write path");
    let p = progress(&db, ns, field_id);
    assert_eq!((p.pending, p.queued, p.applied_by_writers), (1, 1, 0));

    assert_eq!(db.sync_field_indexes(), 1);
    assert_eq!(active_keys(&db, ns), vec![b"doc:1".to_vec()]);
    let p = progress(&db, ns, field_id);
    assert_eq!((p.pending, p.applied_by_barriers), (0, 1));
    assert!(p.applied_up_to_seq > 0, "the applied seq is reported");
    db.shutdown().unwrap();
}

/// A key's changes apply in the order of its versions, whoever applies them:
/// puts, updates and deletes of a few keys, all left queued, end in the index
/// exactly as the last write of each key left it.
#[test]
fn queued_changes_apply_in_the_order_their_writes_did() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    let ns = DEFAULT_NAMESPACE_ID;
    activate_status_index(&db, ns);
    worker_never_runs(&db, ns);

    let mut want = std::collections::BTreeMap::new();
    for round in 0..30u32 {
        for k in 0..5u32 {
            let key = format!("doc:{k}");
            match (round + k) % 3 {
                0 => {
                    db.put(key.as_bytes(), br#"{"status":"active"}"#).unwrap();
                    want.insert(key, true);
                }
                1 => {
                    db.put(key.as_bytes(), br#"{"status":"archived"}"#).unwrap();
                    want.insert(key, false);
                }
                _ => {
                    db.delete(key.as_bytes()).unwrap();
                    want.insert(key, false);
                }
            }
        }
    }
    db.sync_field_indexes();
    let expected: Vec<Vec<u8>> = want.into_iter().filter(|(_, active)| *active).map(|(k, _)| k.into_bytes()).collect();
    assert_eq!(active_keys(&db, ns), expected);
    db.shutdown().unwrap();
}

/// The index checkpoint records the WAL offset its fields reflect, and crash
/// replay starts there. A change still queued at a checkpoint would be behind
/// that offset and lost to a crash, so the checkpoint applies the queue before
/// it stages the field.
#[test]
fn the_checkpoint_applies_queued_changes_before_it_records_its_offset() {
    let dir = TempDir::new().unwrap();
    let ns = DEFAULT_NAMESPACE_ID;
    let field_id = {
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        let field_id = activate_status_index(&db, ns);
        worker_never_runs(&db, ns);
        db.put(b"doc:1", br#"{"status":"active"}"#).unwrap();
        db.run_index_checkpoint().unwrap();
        assert_eq!(progress(&db, ns, field_id).pending, 0, "the checkpoint left the change queued");
        db.crash();
        field_id
    };
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    let extractor: ExtractorFn = Arc::new(|bytes: &[u8]| {
        let v: serde_json::Value = serde_json::from_slice(bytes).ok()?;
        Some(IndexValue::Str(v["status"].as_str()?.to_string()))
    });
    db.activate_field_index(ns, field_id, IndexValueType::Str, extractor).unwrap();
    assert_eq!(
        active_keys(&db, ns),
        vec![b"doc:1".to_vec()],
        "a change queued at the checkpoint was lost to the crash"
    );
    db.shutdown().unwrap();
}

/// A queue past its cap means the worker has fallen behind: the writer that
/// finds it so applies batches until it is back under the cap, so memory and
/// lag stay bounded.
#[test]
fn a_writer_applies_its_fields_queue_once_it_is_past_the_cap() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    let ns = DEFAULT_NAMESPACE_ID;
    let field_id = activate_status_index(&db, ns);
    worker_never_runs(&db, ns);
    let store = db.get_store(ns).unwrap();
    let queue = Arc::clone(&store.namespace_index.read().get(field_id).unwrap().queue);
    for row in 0..QUEUE_CAP as u128 {
        queue.push(
            b"filler",
            1_000_000 + row,
            QueuedChange::Insert(IndexValue::Str("filler".into())),
            None,
            0,
        );
    }

    db.put(b"doc:1", br#"{"status":"active"}"#).unwrap();
    let p = progress(&db, ns, field_id);
    assert!(p.pending <= QUEUE_CAP as u64, "the queue stayed past its cap: {}", p.pending);
    assert!(p.applied_by_writers > 0, "the writer applied nothing");
    db.shutdown().unwrap();
}

/// End to end, with the engine's own worker: a write's change reaches the
/// index without anyone asking, and the worker is what applied it.
#[tokio::test(flavor = "multi_thread")]
async fn the_worker_applies_queued_changes_in_the_background() {
    use std::time::{Duration, Instant};

    let dir = TempDir::new().unwrap();
    // Only the field-index worker may apply the change: no checkpoint tick
    // after the first, which fires as the checkpoint worker starts.
    let mut config = create_db_config();
    config.scheduled_task_config = config.scheduled_task_config.with_index_checkpoint_interval(Duration::from_secs(3600));
    let db = crate::db::facade::AsyncDb::open_with_config(dir.path().to_path_buf(), config)
        .await
        .unwrap();
    let ns = DEFAULT_NAMESPACE_ID;
    let field_id = db.register_index_field(ns, "status", IndexValueType::Str).unwrap();
    let extractor: ExtractorFn = Arc::new(|bytes: &[u8]| {
        let v: serde_json::Value = serde_json::from_slice(bytes).ok()?;
        Some(IndexValue::Str(v["status"].as_str()?.to_string()))
    });
    db.activate_field_index(ns, field_id, IndexValueType::Str, extractor).await.unwrap();

    tokio::time::sleep(Duration::from_millis(100)).await;
    db.put(b"doc:1".to_vec(), br#"{"status":"active"}"#.to_vec()).await.unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let keys = db.query_index(ns, "status = \"active\"").await.unwrap().keys;
        if keys == vec![b"doc:1".to_vec()] {
            break;
        }
        assert!(Instant::now() < deadline, "the worker never applied the change");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let health = db.index_health(ns).await.unwrap();
    let p = health.iter().find(|h| h.field_id == field_id).unwrap().apply_queue.unwrap();
    assert_eq!(p.pending, 0);
    assert!(p.applied_by_worker >= 1, "something other than the worker applied it: {p:?}");
    db.shutdown().await.unwrap();
}
