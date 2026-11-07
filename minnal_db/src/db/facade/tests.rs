//! Unit tests for the facade.

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

#[test]
fn test_db_open_and_crud() {
    let dir = TempDir::new().unwrap();
    let db = Db::open_with_config(dir.path(), test_config()).unwrap();

    db.put(b"k1", b"v1").unwrap();
    db.put(b"k2", b"v2").unwrap();

    assert_eq!(db.get(b"k1").unwrap(), Some(b"v1".to_vec()));
    assert_eq!(db.get(b"k2").unwrap(), Some(b"v2".to_vec()));
    assert_eq!(db.get(b"k3").unwrap(), None);

    db.delete(b"k1").unwrap();
    assert_eq!(db.get(b"k1").unwrap(), None);

    db.shutdown().unwrap();
}

#[test]
fn test_db_iteration() {
    let dir = TempDir::new().unwrap();
    let db = Db::open_with_config(dir.path(), test_config()).unwrap();

    db.put(b"a", b"1").unwrap();
    db.put(b"b", b"2").unwrap();
    db.put(b"c", b"3").unwrap();

    let all = db.iter().unwrap();
    assert_eq!(all.len(), 3);
    assert_eq!(all[0], (b"a".to_vec(), b"1".to_vec()));
    assert_eq!(all[1], (b"b".to_vec(), b"2".to_vec()));
    assert_eq!(all[2], (b"c".to_vec(), b"3".to_vec()));

    let keys = db.keys().unwrap();
    assert_eq!(keys, vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()]);

    db.shutdown().unwrap();
}

#[test]
fn test_db_range() {
    let dir = TempDir::new().unwrap();
    let db = Db::open_with_config(dir.path(), test_config()).unwrap();

    db.put(b"a", b"1").unwrap();
    db.put(b"b", b"2").unwrap();
    db.put(b"c", b"3").unwrap();
    db.put(b"d", b"4").unwrap();

    let range = db.range(b"b", Some(b"d")).unwrap();
    assert_eq!(range.len(), 2);
    assert_eq!(range[0].0, b"b".to_vec());
    assert_eq!(range[1].0, b"c".to_vec());

    db.shutdown().unwrap();
}

#[test]
fn test_db_scan_prefix() {
    let dir = TempDir::new().unwrap();
    let db = Db::open_with_config(dir.path(), test_config()).unwrap();

    db.put(b"user:1", b"alice").unwrap();
    db.put(b"user:2", b"bob").unwrap();
    db.put(b"order:1", b"item").unwrap();

    let users = db.scan_prefix(b"user:").unwrap();
    assert_eq!(users.len(), 2);
    assert_eq!(users[0].0, b"user:1".to_vec());

    db.shutdown().unwrap();
}

#[test]
fn test_db_namespaces() {
    let dir = TempDir::new().unwrap();
    let db = Db::open_with_config(dir.path(), test_config()).unwrap();

    // Default namespace
    db.put(b"global", b"value").unwrap();

    // Create and use a namespace
    let users = db.namespace("users").unwrap();
    users.put(b"u1", b"alice").unwrap();
    users.put(b"u2", b"bob").unwrap();

    // Isolation: default doesn't see namespace data
    assert_eq!(db.get(b"u1").unwrap(), None);
    assert_eq!(users.get(b"global").unwrap(), None);

    // Namespace data
    assert_eq!(users.get(b"u1").unwrap(), Some(b"alice".to_vec()));

    // Namespace iteration
    let keys = users.keys().unwrap();
    assert_eq!(keys.len(), 2);

    // List namespaces
    let ns_list = db.list_namespaces();
    assert!(ns_list.len() >= 2); // default + users

    db.shutdown().unwrap();
}

#[test]
fn test_per_namespace_ops_metrics() {
    let dir = TempDir::new().unwrap();
    let db = Db::open_with_config(dir.path(), test_config()).unwrap();

    let a = db.namespace("a").unwrap();
    a.put(b"k1", b"v").unwrap();
    a.put(b"k2", b"v").unwrap();
    let b = db.namespace("b").unwrap();
    b.put(b"k1", b"v").unwrap();

    // Counters are isolated per namespace.
    assert_eq!(db.ops_metrics_for("a").unwrap().puts, 2);
    assert_eq!(db.ops_metrics_for("b").unwrap().puts, 1);

    // Engine-wide view is the sum across namespaces.
    assert_eq!(db.ops_metrics().puts, 3);

    // by-namespace map carries each namespace's own counters.
    let by_ns: std::collections::HashMap<String, _> = db.ops_metrics_by_namespace().into_iter().collect();
    assert_eq!(by_ns["a"].puts, 2);
    assert_eq!(by_ns["b"].puts, 1);

    // Unknown namespace is an error, not a zero snapshot.
    assert!(db.ops_metrics_for("missing").is_err());

    // Dropping a namespace folds its totals into the global accumulator, so
    // the engine aggregate stays monotonic, but the namespace is no longer
    // individually queryable.
    db.remove_namespace("a").unwrap();
    assert_eq!(db.ops_metrics().puts, 3, "engine total stays monotonic after drop");
    assert!(db.ops_metrics_for("a").is_err(), "dropped ns is no longer queryable");
    assert!(!db.ops_metrics_by_namespace().iter().any(|(n, _)| n == "a"));

    db.shutdown().unwrap();
}

#[test]
fn test_remove_namespace_deletes_storage_and_survives_reopen() {
    let dir = TempDir::new().unwrap();
    let doomed_dir = dir.path().join("ns_doomed");
    let keeper_dir = dir.path().join("ns_keeper");

    {
        let db = Db::open_with_config(dir.path(), test_config()).unwrap();

        // Two namespaces so we can confirm the survivor is untouched.
        let doomed = db.namespace("doomed").unwrap();
        doomed.put(b"k1", b"v1").unwrap();
        doomed.put(b"k2", b"v2").unwrap();
        let keeper = db.namespace("keeper").unwrap();
        keeper.put(b"alive", b"yes").unwrap();

        assert!(doomed_dir.exists(), "data dir should exist before removal");

        db.remove_namespace("doomed").unwrap();

        // On-disk storage for the removed namespace is gone; the survivor remains.
        assert!(!doomed_dir.exists(), "data dir should be deleted on removal");
        assert!(keeper_dir.exists(), "survivor data dir must remain");
        assert!(!db.list_namespaces().iter().any(|(n, _)| n == "doomed"));

        db.shutdown().unwrap();
    }

    // Reopen: recovery must not resurrect the removed namespace, and the
    // surviving namespace's data must still be intact.
    let db = Db::open_with_config(dir.path(), test_config()).unwrap();
    assert!(!db.list_namespaces().iter().any(|(n, _)| n == "doomed"));
    let keeper = db.namespace("keeper").unwrap();
    assert_eq!(keeper.get(b"alive").unwrap(), Some(b"yes".to_vec()));
    db.shutdown().unwrap();
}

#[test]
fn test_db_namespace_gc() {
    let dir = TempDir::new().unwrap();
    let db = Db::open_with_config(dir.path(), test_config()).unwrap();

    let ns = db.namespace("test_ns").unwrap();
    ns.put(b"k", b"v").unwrap();
    let gc_stats = ns.garbage_collect().unwrap();
    assert_eq!(gc_stats.bytes_reclaimed, 0);

    db.shutdown().unwrap();
}

#[test]
fn test_db_maintenance() {
    let dir = TempDir::new().unwrap();
    let db = Db::open_with_config(dir.path(), test_config()).unwrap();

    db.put(b"k", b"v").unwrap();

    let _stats = db.stats();
    let _ratio = db.waste_ratio();
    let _gc = db.garbage_collect().unwrap();
    let _compact = db.compact();

    db.shutdown().unwrap();
}

#[tokio::test]
async fn test_async_db_crud() {
    let dir = TempDir::new().unwrap();
    let db = AsyncDb::open_with_config(dir.path().to_path_buf(), test_config()).await.unwrap();

    db.put(b"k1".to_vec(), b"v1".to_vec()).await.unwrap();
    assert_eq!(db.get(b"k1".to_vec()).await.unwrap(), Some(b"v1".to_vec()));

    db.delete(b"k1".to_vec()).await.unwrap();
    assert_eq!(db.get(b"k1".to_vec()).await.unwrap(), None);

    db.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_async_db_namespace() {
    let dir = TempDir::new().unwrap();
    let db = AsyncDb::open_with_config(dir.path().to_path_buf(), test_config()).await.unwrap();

    let ns = db.namespace("events".to_string()).await.unwrap();
    ns.put(b"e1".to_vec(), b"click".to_vec()).await.unwrap();
    assert_eq!(ns.get(b"e1".to_vec()).await.unwrap(), Some(b"click".to_vec()));

    // Default namespace should not see it
    assert_eq!(db.get(b"e1".to_vec()).await.unwrap(), None);

    db.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_async_namespace_no_wal_crud() {
    let dir = TempDir::new().unwrap();
    let db = AsyncDb::open_with_config(dir.path().to_path_buf(), test_config()).await.unwrap();
    let ns = db.namespace("cache".to_string()).await.unwrap();

    // No-WAL write is readable like any other write.
    ns.put_no_wal(b"q1".to_vec(), b"emb1".to_vec()).await.unwrap();
    assert_eq!(ns.get(b"q1".to_vec()).await.unwrap(), Some(b"emb1".to_vec()));

    // No-WAL delete removes it.
    ns.delete_no_wal(b"q1".to_vec()).await.unwrap();
    assert_eq!(ns.get(b"q1".to_vec()).await.unwrap(), None);

    // No-WAL delete tombstones a key written *with* the WAL too.
    ns.put(b"q2".to_vec(), b"emb2".to_vec()).await.unwrap();
    ns.delete_no_wal(b"q2".to_vec()).await.unwrap();
    assert_eq!(ns.get(b"q2".to_vec()).await.unwrap(), None);

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

#[test]
fn test_db_typed_crud() {
    let dir = TempDir::new().unwrap();
    let db = Db::open_with_config(dir.path(), test_config()).unwrap();

    let key = UserId(42);
    let value = UserProfile {
        name: "Alice".to_string(),
        age: 30,
    };

    db.put_typed(&key, &value).unwrap();

    let got: Option<UserProfile> = db.get_typed::<UserId, UserProfile>(&key).unwrap();
    assert_eq!(got, Some(value.clone()));

    // Update
    let updated = UserProfile {
        name: "Alice".to_string(),
        age: 31,
    };
    db.put_typed(&key, &updated).unwrap();
    let got: Option<UserProfile> = db.get_typed(&key).unwrap();
    assert_eq!(got, Some(updated));

    // Delete
    db.delete_typed(&key).unwrap();
    let got: Option<UserProfile> = db.get_typed(&key).unwrap();
    assert_eq!(got, None);

    db.shutdown().unwrap();
}

#[test]
fn test_namespace_typed_crud() {
    let dir = TempDir::new().unwrap();
    let db = Db::open_with_config(dir.path(), test_config()).unwrap();

    let ns = db.namespace("users").unwrap();
    let key = UserId(1);
    let value = UserProfile {
        name: "Bob".to_string(),
        age: 25,
    };

    ns.put_typed(&key, &value).unwrap();
    let got: Option<UserProfile> = ns.get_typed(&key).unwrap();
    assert_eq!(got, Some(value));

    // Isolation: default namespace shouldn't have it
    let from_default: Option<UserProfile> = db.get_typed(&key).unwrap();
    assert_eq!(from_default, None);

    db.shutdown().unwrap();
}

#[tokio::test]
async fn test_async_db_typed_crud() {
    let dir = TempDir::new().unwrap();
    let db = AsyncDb::open_with_config(dir.path().to_path_buf(), test_config()).await.unwrap();

    let key = UserId(99);
    let value = UserProfile {
        name: "Charlie".to_string(),
        age: 40,
    };

    db.put_typed(&key, &value).await.unwrap();
    let got: Option<UserProfile> = db.get_typed(&key).await.unwrap();
    assert_eq!(got, Some(value));

    db.delete_typed(&key).await.unwrap();
    let got: Option<UserProfile> = db.get_typed(&key).await.unwrap();
    assert_eq!(got, None);

    db.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_async_namespace_typed_crud() {
    let dir = TempDir::new().unwrap();
    let db = AsyncDb::open_with_config(dir.path().to_path_buf(), test_config()).await.unwrap();

    let ns = db.namespace("products".to_string()).await.unwrap();
    let key = 42u64;
    let value = "Widget".to_string();

    ns.put_typed(&key, &value).await.unwrap();
    let got: Option<String> = ns.get_typed(&key).await.unwrap();
    assert_eq!(got, Some("Widget".to_string()));

    db.shutdown().await.unwrap();
}

// ── Typed Iteration tests ─────────────────────────────────────────

#[test]
fn test_db_typed_iter() {
    let dir = TempDir::new().unwrap();
    let db = Db::open_with_config(dir.path(), test_config()).unwrap();

    db.put_typed(
        &UserId(1),
        &UserProfile {
            name: "Alice".into(),
            age: 30,
        },
    )
    .unwrap();
    db.put_typed(&UserId(2), &UserProfile { name: "Bob".into(), age: 25 }).unwrap();
    db.put_typed(
        &UserId(3),
        &UserProfile {
            name: "Charlie".into(),
            age: 35,
        },
    )
    .unwrap();

    // iter_typed
    let all: Vec<(UserId, UserProfile)> = db.iter_typed().unwrap();
    assert_eq!(all.len(), 3);
    assert_eq!(all[0].1.name, "Alice");

    // keys_typed
    let keys: Vec<UserId> = db.keys_typed().unwrap();
    assert_eq!(keys.len(), 3);

    db.shutdown().unwrap();
}

#[test]
fn test_namespace_typed_iter() {
    let dir = TempDir::new().unwrap();
    let db = Db::open_with_config(dir.path(), test_config()).unwrap();

    let ns = db.namespace("users").unwrap();
    ns.put_typed(&UserId(10), &UserProfile { name: "Dan".into(), age: 20 }).unwrap();
    ns.put_typed(&UserId(20), &UserProfile { name: "Eve".into(), age: 28 }).unwrap();

    let all: Vec<(UserId, UserProfile)> = ns.iter_typed().unwrap();
    assert_eq!(all.len(), 2);

    let keys: Vec<UserId> = ns.keys_typed().unwrap();
    assert_eq!(keys.len(), 2);

    db.shutdown().unwrap();
}

#[tokio::test]
async fn test_async_db_typed_iter() {
    let dir = TempDir::new().unwrap();
    let db = AsyncDb::open_with_config(dir.path().to_path_buf(), test_config()).await.unwrap();

    db.put_typed(
        &UserId(1),
        &UserProfile {
            name: "Alice".into(),
            age: 30,
        },
    )
    .await
    .unwrap();
    db.put_typed(&UserId(2), &UserProfile { name: "Bob".into(), age: 25 }).await.unwrap();

    let all: Vec<(UserId, UserProfile)> = db.iter_typed().await.unwrap();
    assert_eq!(all.len(), 2);

    let keys: Vec<UserId> = db.keys_typed().await.unwrap();
    assert_eq!(keys.len(), 2);

    db.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_async_namespace_typed_iter() {
    let dir = TempDir::new().unwrap();
    let db = AsyncDb::open_with_config(dir.path().to_path_buf(), test_config()).await.unwrap();

    let ns = db.namespace("items".to_string()).await.unwrap();
    ns.put_typed(&1u64, &"first".to_string()).await.unwrap();
    ns.put_typed(&2u64, &"second".to_string()).await.unwrap();

    let all: Vec<(u64, String)> = ns.iter_typed().await.unwrap();
    assert_eq!(all.len(), 2);

    let keys: Vec<u64> = ns.keys_typed().await.unwrap();
    assert_eq!(keys.len(), 2);

    db.shutdown().await.unwrap();
}
