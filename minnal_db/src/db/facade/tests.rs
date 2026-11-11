//! Unit tests for the facade: `AsyncDb` and `AsyncNamespace`.

use super::*;
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use tempfile::TempDir;

/// Config for facade tests: keeps the eager per-namespace fd footprint small
/// so the suite survives high `cargo test` parallelism. See
/// [`crate::support::TEST_NUM_BUCKETS`].
fn test_config() -> DbConfig {
    DbConfig {
        num_buckets: crate::support::TEST_NUM_BUCKETS,
        ..DbConfig::default()
    }
}

async fn open(dir: &TempDir) -> AsyncDb {
    AsyncDb::open_with_config(dir.path().to_path_buf(), test_config()).await.unwrap()
}

fn b(s: &str) -> Vec<u8> {
    s.as_bytes().to_vec()
}

#[tokio::test]
async fn test_db_open_and_crud() {
    let dir = TempDir::new().unwrap();
    let db = open(&dir).await;

    db.put(b("k1"), b("v1")).await.unwrap();
    db.put(b("k2"), b("v2")).await.unwrap();

    assert_eq!(db.get(b("k1")).await.unwrap(), Some(b("v1")));
    assert_eq!(db.get(b("k2")).await.unwrap(), Some(b("v2")));
    assert_eq!(db.get(b("k3")).await.unwrap(), None);

    db.delete(b("k1")).await.unwrap();
    assert_eq!(db.get(b("k1")).await.unwrap(), None);

    db.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_db_iteration() {
    let dir = TempDir::new().unwrap();
    let db = open(&dir).await;

    db.put(b("a"), b("1")).await.unwrap();
    db.put(b("b"), b("2")).await.unwrap();
    db.put(b("c"), b("3")).await.unwrap();

    let all = db.iter().await.unwrap();
    assert_eq!(all, vec![(b("a"), b("1")), (b("b"), b("2")), (b("c"), b("3"))]);

    let keys = db.keys().await.unwrap();
    assert_eq!(keys, vec![b("a"), b("b"), b("c")]);

    db.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_db_range() {
    let dir = TempDir::new().unwrap();
    let db = open(&dir).await;

    for (k, v) in [("a", "1"), ("b", "2"), ("c", "3"), ("d", "4")] {
        db.put(b(k), b(v)).await.unwrap();
    }

    let range = db.range(b("b"), Some(b("d"))).await.unwrap();
    assert_eq!(range.len(), 2);
    assert_eq!(range[0].0, b("b"));
    assert_eq!(range[1].0, b("c"));

    db.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_db_scan_prefix() {
    let dir = TempDir::new().unwrap();
    let db = open(&dir).await;

    db.put(b("user:1"), b("alice")).await.unwrap();
    db.put(b("user:2"), b("bob")).await.unwrap();
    db.put(b("order:1"), b("item")).await.unwrap();

    let users = db.scan_prefix(b("user:")).await.unwrap();
    assert_eq!(users.len(), 2);
    assert_eq!(users[0].0, b("user:1"));

    db.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_db_namespaces() {
    let dir = TempDir::new().unwrap();
    let db = open(&dir).await;

    // Default namespace
    db.put(b("global"), b("value")).await.unwrap();

    // Create and use a namespace
    let users = db.namespace("users".to_string()).await.unwrap();
    users.put(b("u1"), b("alice")).await.unwrap();
    users.put(b("u2"), b("bob")).await.unwrap();

    // Isolation: default doesn't see namespace data, and the other way round
    assert_eq!(db.get(b("u1")).await.unwrap(), None);
    assert_eq!(users.get(b("global")).await.unwrap(), None);

    assert_eq!(users.get(b("u1")).await.unwrap(), Some(b("alice")));
    assert_eq!(users.keys().await.unwrap().len(), 2);

    // default + users
    assert!(db.list_namespaces().len() >= 2);

    db.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_per_namespace_ops_metrics() {
    let dir = TempDir::new().unwrap();
    let db = open(&dir).await;

    let a = db.namespace("a".to_string()).await.unwrap();
    a.put(b("k1"), b("v")).await.unwrap();
    a.put(b("k2"), b("v")).await.unwrap();
    let ns_b = db.namespace("b".to_string()).await.unwrap();
    ns_b.put(b("k1"), b("v")).await.unwrap();

    // Counters are isolated per namespace.
    assert_eq!(db.ops_metrics_for("a").unwrap().puts, 2);
    assert_eq!(db.ops_metrics_for("b").unwrap().puts, 1);

    // Engine-wide view is the sum across namespaces, the system catalog's
    // own records (one per namespace created) among them.
    let catalog_puts = || db.ops_metrics_for(crate::CATALOG_NAMESPACE).unwrap().puts;
    assert_eq!(db.ops_metrics().puts - catalog_puts(), 3);

    // by-namespace map carries each namespace's own counters.
    let by_ns: std::collections::HashMap<String, _> = db.ops_metrics_by_namespace().into_iter().collect();
    assert_eq!(by_ns["a"].puts, 2);
    assert_eq!(by_ns["b"].puts, 1);

    // Unknown namespace is an error, not a zero snapshot.
    assert!(db.ops_metrics_for("missing").is_err());

    // Dropping a namespace folds its totals into the global accumulator, so
    // the engine aggregate stays monotonic, but the namespace is no longer
    // individually queryable.
    drop(a);
    db.remove_namespace("a".to_string()).await.unwrap();
    assert_eq!(db.ops_metrics().puts - catalog_puts(), 3, "engine total stays monotonic after drop");
    assert!(db.ops_metrics_for("a").is_err(), "dropped ns is no longer queryable");
    assert!(!db.ops_metrics_by_namespace().iter().any(|(n, _)| n == "a"));

    db.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_remove_namespace_deletes_storage_and_survives_reopen() {
    let dir = TempDir::new().unwrap();
    let doomed_dir = dir.path().join("ns_doomed");
    let keeper_dir = dir.path().join("ns_keeper");

    {
        let db = open(&dir).await;

        // Two namespaces so we can confirm the survivor is untouched.
        let doomed = db.namespace("doomed".to_string()).await.unwrap();
        doomed.put(b("k1"), b("v1")).await.unwrap();
        doomed.put(b("k2"), b("v2")).await.unwrap();
        drop(doomed);
        let keeper = db.namespace("keeper".to_string()).await.unwrap();
        keeper.put(b("alive"), b("yes")).await.unwrap();

        assert!(doomed_dir.exists(), "data dir should exist before removal");

        db.remove_namespace("doomed".to_string()).await.unwrap();

        // On-disk storage for the removed namespace is gone; the survivor remains.
        assert!(!doomed_dir.exists(), "data dir should be deleted on removal");
        assert!(keeper_dir.exists(), "survivor data dir must remain");
        assert!(!db.list_namespaces().iter().any(|(n, _)| n == "doomed"));

        db.shutdown().await.unwrap();
    }

    // Reopen: recovery must not resurrect the removed namespace, and the
    // surviving namespace's data must still be intact.
    let db = open(&dir).await;
    assert!(!db.list_namespaces().iter().any(|(n, _)| n == "doomed"));
    let keeper = db.namespace("keeper".to_string()).await.unwrap();
    assert_eq!(keeper.get(b("alive")).await.unwrap(), Some(b("yes")));
    db.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_db_namespace_gc() {
    let dir = TempDir::new().unwrap();
    let db = open(&dir).await;

    let ns = db.namespace("test_ns".to_string()).await.unwrap();
    ns.put(b("k"), b("v")).await.unwrap();
    let gc_stats = ns.garbage_collect().await.unwrap();
    assert_eq!(gc_stats.bytes_reclaimed, 0);

    db.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_db_maintenance() {
    let dir = TempDir::new().unwrap();
    let db = open(&dir).await;

    db.put(b("k"), b("v")).await.unwrap();

    let _stats = db.stats();
    let _ratio = db.waste_ratio();
    let _gc = db.garbage_collect().await.unwrap();
    db.compact().await.unwrap();

    db.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_async_namespace_no_wal_crud() {
    let dir = TempDir::new().unwrap();
    let db = open(&dir).await;
    let ns = db.namespace("cache".to_string()).await.unwrap();

    // No-WAL write is readable like any other write.
    ns.put_no_wal(b("q1"), b("emb1")).await.unwrap();
    assert_eq!(ns.get(b("q1")).await.unwrap(), Some(b("emb1")));

    // No-WAL delete removes it.
    ns.delete_no_wal(b("q1")).await.unwrap();
    assert_eq!(ns.get(b("q1")).await.unwrap(), None);

    // No-WAL delete tombstones a key written *with* the WAL too.
    ns.put(b("q2"), b("emb2")).await.unwrap();
    ns.delete_no_wal(b("q2")).await.unwrap();
    assert_eq!(ns.get(b("q2")).await.unwrap(), None);

    // The no-WAL paths bump their own counters and skip the WAL fsync path:
    // one WAL-backed put (q2) accounts for the only wal_fsync from these ops.
    let m = db.ops_metrics();
    assert_eq!(m.no_wal_puts, 1, "one put_no_wal");
    assert_eq!(m.no_wal_deletes, 2, "two delete_no_wal calls");

    db.shutdown().await.unwrap();
}

// ── Typed API tests ───────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Archive, RkyvSerialize, RkyvDeserialize)]
struct UserId(u64);

#[derive(Debug, Clone, PartialEq, Archive, RkyvSerialize, RkyvDeserialize)]
struct UserProfile {
    name: String,
    age: u32,
}

#[tokio::test]
async fn test_db_typed_crud() {
    let dir = TempDir::new().unwrap();
    let db = open(&dir).await;

    let key = UserId(42);
    let value = UserProfile {
        name: "Alice".to_string(),
        age: 30,
    };

    db.put_typed(&key, &value).await.unwrap();
    let got: Option<UserProfile> = db.get_typed(&key).await.unwrap();
    assert_eq!(got, Some(value));

    // Update
    let updated = UserProfile {
        name: "Alice".to_string(),
        age: 31,
    };
    db.put_typed(&key, &updated).await.unwrap();
    let got: Option<UserProfile> = db.get_typed(&key).await.unwrap();
    assert_eq!(got, Some(updated));

    // Delete
    db.delete_typed(&key).await.unwrap();
    let got: Option<UserProfile> = db.get_typed(&key).await.unwrap();
    assert_eq!(got, None);

    db.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_namespace_typed_crud() {
    let dir = TempDir::new().unwrap();
    let db = open(&dir).await;

    let ns = db.namespace("users".to_string()).await.unwrap();
    let key = UserId(1);
    let value = UserProfile {
        name: "Bob".to_string(),
        age: 25,
    };

    ns.put_typed(&key, &value).await.unwrap();
    let got: Option<UserProfile> = ns.get_typed(&key).await.unwrap();
    assert_eq!(got, Some(value));

    // Isolation: the default namespace doesn't have it
    let from_default: Option<UserProfile> = db.get_typed(&key).await.unwrap();
    assert_eq!(from_default, None);

    db.shutdown().await.unwrap();
}

// ── Typed Iteration tests ─────────────────────────────────────────

#[tokio::test]
async fn test_db_typed_iter() {
    let dir = TempDir::new().unwrap();
    let db = open(&dir).await;

    for (id, name, age) in [(1, "Alice", 30), (2, "Bob", 25), (3, "Charlie", 35)] {
        db.put_typed(&UserId(id), &UserProfile { name: name.into(), age }).await.unwrap();
    }

    let all: Vec<(UserId, UserProfile)> = db.iter_typed().await.unwrap();
    assert_eq!(all.len(), 3);
    assert_eq!(all[0].1.name, "Alice");

    let keys: Vec<UserId> = db.keys_typed().await.unwrap();
    assert_eq!(keys.len(), 3);

    db.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_namespace_typed_iter() {
    let dir = TempDir::new().unwrap();
    let db = open(&dir).await;

    let ns = db.namespace("items".to_string()).await.unwrap();
    ns.put_typed(&1u64, &"first".to_string()).await.unwrap();
    ns.put_typed(&2u64, &"second".to_string()).await.unwrap();

    let all: Vec<(u64, String)> = ns.iter_typed().await.unwrap();
    assert_eq!(all.len(), 2);

    let keys: Vec<u64> = ns.keys_typed().await.unwrap();
    assert_eq!(keys.len(), 2);

    db.shutdown().await.unwrap();
}
