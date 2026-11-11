//! Creating, listing, persisting and removing namespaces, and their isolation.

use super::*;
use crate::namespace::NamespaceId;

/// Regression (F6): concurrent first use of a namespace must not fail.
///
/// `create_namespace` publishes the registry entry — and persists it —
/// before it opens the `KVStore` and inserts it into `stores`. Between those
/// two points the name resolves to an id that `get_store` does not know.
/// Every get-or-create caller (`Db::namespace`, `AsyncDb::namespace`, and so
/// the whole vector write path) does exactly `get_namespace_id` followed by
/// `get_store_by_name`, so it lands in that window and fails with
/// "Namespace with ID N not found". Separately, two callers that both see the
/// name as absent both try to create it, and the loser gets
/// "Namespace 'x' already exists" from the registry.
///
/// Seen once in a live stress run, on a new store's first embed, where it
/// burned one of five retries. It is not rare: the probe this test came from
/// reproduced **167** "not found" and **13** "already exists" in 240 attempts.
#[test]
fn test_concurrent_first_use_of_a_namespace_does_not_race() {
    use std::sync::Barrier;

    let dir = TempDir::new().unwrap();
    let db = Arc::new(Database::open(dir.path(), create_db_config()).unwrap());
    let mut failures: Vec<String> = Vec::new();

    for round in 0..25 {
        let name = format!("ns_{round}");
        // All four threads reach the get-or-create at once, on a name that
        // does not exist yet.
        let barrier = Arc::new(Barrier::new(4));
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let db = Arc::clone(&db);
                let name = name.clone();
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    // Exactly what `Db::namespace` / `AsyncDb::namespace` do.
                    match db.get_namespace_id(&name) {
                        Some(id) => Ok(id),
                        None => db.create_namespace(&name),
                    }
                    .and_then(|_| db.get_store_by_name(&name))
                    .map(|_| ())
                })
            })
            .collect();
        for handle in handles {
            if let Err(e) = handle.join().unwrap() {
                failures.push(e.to_string());
            }
        }

        // All four must agree on one namespace, and it must be usable.
        let id = db.get_namespace_id(&name).expect("the namespace should exist after the round");
        db.put_ns(id, b"k", b"v").unwrap();
        assert_eq!(db.get_ns(id, b"k").unwrap().as_deref(), Some(&b"v"[..]));
    }

    assert!(
        failures.is_empty(),
        "{} of 100 concurrent first-use attempts failed: {:?}",
        failures.len(),
        // Distinct messages only — the same two repeat.
        failures.iter().collect::<std::collections::BTreeSet<_>>()
    );
}

/// `lookup_open_store` answers from memory and reports the creation window
/// instead of guessing: a name with no namespace is `Absent`, a created one is
/// `Open` with the id and store `get_store` gives, and a name the registry has
/// published before its store is in the store map (the window
/// `test_concurrent_first_use_of_a_namespace_does_not_race` describes) is
/// `NotReady`, never `Absent`. `AsyncDb::existing_namespace` relies on that to
/// send only the window to the blocking path.
#[test]
fn lookup_open_store_reports_the_creation_window_as_not_ready() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();

    assert!(matches!(db.lookup_open_store("missing"), OpenStoreLookup::Absent));

    let id = db.create_namespace("ready").unwrap();
    match db.lookup_open_store("ready") {
        OpenStoreLookup::Open(found, store) => {
            assert_eq!(found, id);
            assert!(Arc::ptr_eq(&store, &db.get_store(id).unwrap()));
        }
        _ => panic!("a created namespace must be open"),
    }

    // The middle of `create_namespace`: the name is published, the store not yet.
    let half = db.registry.write().create("half").unwrap();
    assert!(matches!(db.lookup_open_store("half"), OpenStoreLookup::NotReady));
    assert_eq!(db.get_namespace_id("half"), Some(half));
}

#[test]
fn test_database_basic_operations() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();

    db.put(b"key1", b"value1").unwrap();
    db.put(b"key2", b"value2").unwrap();

    assert_eq!(db.get(b"key1").unwrap(), Some(b"value1".to_vec()));
    assert_eq!(db.get(b"key2").unwrap(), Some(b"value2".to_vec()));

    db.delete(b"key2").unwrap();
    assert_eq!(db.get(b"key2").unwrap(), None);

    db.shutdown().unwrap();
}

#[test]
fn test_database_multiple_namespaces() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();

    let users_id = db.create_namespace("users").unwrap();
    assert_ne!(users_id, NamespaceId::DEFAULT);
    assert_ne!(users_id, NamespaceId::SYSTEM);

    // Write to default namespace
    db.put(b"global_key", b"global_value").unwrap();

    // Write to users namespace
    db.put_ns(users_id, b"user:1", b"alice").unwrap();
    db.put_ns(users_id, b"user:2", b"bob").unwrap();

    // Read from default — should not see users data
    assert_eq!(db.get(b"global_key").unwrap(), Some(b"global_value".to_vec()));
    assert_eq!(db.get(b"user:1").unwrap(), None);

    // Read from users namespace
    assert_eq!(db.get_ns(users_id, b"user:1").unwrap(), Some(b"alice".to_vec()));
    assert_eq!(db.get_ns(users_id, b"user:2").unwrap(), Some(b"bob".to_vec()));
    assert_eq!(db.get_ns(users_id, b"global_key").unwrap(), None);

    db.shutdown().unwrap();
}

#[test]
fn test_database_namespace_listing() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();

    let users = db.create_namespace("users").unwrap();
    let orders = db.create_namespace("orders").unwrap();

    let mut namespaces = db.list_namespaces();
    namespaces.sort();
    assert_eq!(
        namespaces,
        vec![
            ("default".to_string(), NamespaceId::DEFAULT),
            ("orders".to_string(), orders),
            ("system".to_string(), NamespaceId::SYSTEM),
            ("users".to_string(), users),
        ]
    );

    db.shutdown().unwrap();
}

#[test]
fn test_database_namespace_persistence() {
    let dir = TempDir::new().unwrap();

    // Create database with namespaces and data
    {
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        let users_id = db.create_namespace("users").unwrap();
        db.put(b"default_key", b"default_val").unwrap();
        db.put_ns(users_id, b"user_key", b"user_val").unwrap();
        db.shutdown().unwrap();
    }

    // Reopen and verify
    {
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        assert!(db.namespace_exists("users"));
        let users_id = db.get_namespace_id("users").unwrap();

        assert_eq!(db.get(b"default_key").unwrap(), Some(b"default_val".to_vec()));
        assert_eq!(db.get_ns(users_id, b"user_key").unwrap(), Some(b"user_val".to_vec()));

        db.shutdown().unwrap();
    }
}

#[test]
fn test_database_remove_namespace() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();

    let users_id = db.create_namespace("users").unwrap();
    db.put_ns(users_id, b"key", b"val").unwrap();

    db.remove_namespace("users").unwrap();
    assert!(!db.namespace_exists("users"));
    assert!(db.get_ns(users_id, b"key").is_err());

    // Cannot remove default
    assert!(db.remove_namespace("default").is_err());

    db.shutdown().unwrap();
}

#[test]
fn test_delete_ns_removes_key_from_namespace() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();

    let ns = db.create_namespace("items").unwrap();

    db.put_ns(ns, b"k1", b"v1").unwrap();
    db.put_ns(ns, b"k2", b"v2").unwrap();
    assert_eq!(db.get_ns(ns, b"k1").unwrap(), Some(b"v1".to_vec()));

    db.delete_ns(ns, b"k1").unwrap();
    assert_eq!(db.get_ns(ns, b"k1").unwrap(), None);
    // Sibling key is untouched
    assert_eq!(db.get_ns(ns, b"k2").unwrap(), Some(b"v2".to_vec()));

    db.shutdown().unwrap();
}

#[test]
fn test_delete_ns_does_not_affect_other_namespaces() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();

    let ns_a = db.create_namespace("ns_a").unwrap();
    let ns_b = db.create_namespace("ns_b").unwrap();

    // Same key in three namespaces
    db.put(b"shared", b"default_val").unwrap();
    db.put_ns(ns_a, b"shared", b"a_val").unwrap();
    db.put_ns(ns_b, b"shared", b"b_val").unwrap();

    // Delete only from ns_a
    db.delete_ns(ns_a, b"shared").unwrap();

    assert_eq!(db.get_ns(ns_a, b"shared").unwrap(), None);
    assert_eq!(db.get(b"shared").unwrap(), Some(b"default_val".to_vec()));
    assert_eq!(db.get_ns(ns_b, b"shared").unwrap(), Some(b"b_val".to_vec()),);

    db.shutdown().unwrap();
}

#[test]
fn test_delete_ns_nonexistent_key_is_ok() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();

    let ns = db.create_namespace("empty").unwrap();

    // Deleting a key that was never written should succeed silently.
    db.delete_ns(ns, b"ghost").unwrap();
    assert_eq!(db.get_ns(ns, b"ghost").unwrap(), None);

    db.shutdown().unwrap();
}

#[test]
fn test_delete_ns_advances_wal() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();

    let ns = db.create_namespace("wal_check").unwrap();
    db.put_ns(ns, b"k", b"v").unwrap();

    let tail_before = db.wal_metadata().tail;
    db.delete_ns(ns, b"k").unwrap();
    let tail_after = db.wal_metadata().tail;

    assert!(
        tail_after > tail_before,
        "WAL tail must advance after delete_ns (before={}, after={})",
        tail_before,
        tail_after,
    );

    db.shutdown().unwrap();
}

#[test]
fn test_delete_ns_invalid_namespace_returns_error() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();

    // Namespace 9999 was never created.
    let result = db.delete_ns(NamespaceId::from_u128(9999), b"k");
    assert!(result.is_err(), "delete_ns on unknown namespace should fail");

    db.shutdown().unwrap();
}

/// A namespace removed while it is being created must not leave a store the
/// registry does not list, or a listed store whose directory was deleted: its
/// next metadata write, at shutdown at the latest, fails with `NotFound`.
#[test]
fn removing_a_namespace_while_it_is_created_leaves_no_store_without_its_files() {
    for round in 0..50 {
        let dir = TempDir::new().unwrap();
        let db = Arc::new(Database::open(dir.path(), create_db_config()).unwrap());
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let creator = {
            let (db, barrier) = (Arc::clone(&db), Arc::clone(&barrier));
            std::thread::spawn(move || {
                barrier.wait();
                db.create_namespace("racing").unwrap();
            })
        };
        barrier.wait();
        // Keep trying until the create's registry entry is there to remove.
        while db.remove_namespace("racing").is_err() {
            std::hint::spin_loop();
        }
        creator.join().unwrap();
        assert_eq!(db.shutdown().map_err(|e| format!("{e:?}")), Ok(()), "round {round}");
    }
}

/// The sweep for an interrupted drop's leftovers must not delete a live store's
/// files, and does delete a directory no namespace owns.
#[test]
fn only_an_unregistered_namespace_loses_its_directory() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    db.create_namespace("live").unwrap();
    let live = crate::layout::namespace_data_dir(dir.path(), "live");
    db.remove_unregistered_namespace_dir("live").unwrap();
    assert!(live.exists());
    let leftover = crate::layout::namespace_data_dir(dir.path(), "leftover");
    std::fs::create_dir_all(&leftover).unwrap();
    db.remove_unregistered_namespace_dir("leftover").unwrap();
    assert!(!leftover.exists());
    db.remove_unregistered_namespace_dir("never_existed").unwrap();
    db.shutdown().unwrap();
}

/// Removing a namespace flushes and compacts it, which can take seconds. Lookups
/// of other namespaces run on async threads and must not wait for that.
#[test]
fn removing_a_namespace_does_not_hold_up_lookups_of_others() {
    let dir = TempDir::new().unwrap();
    let db = Arc::new(Database::open(dir.path(), create_db_config()).unwrap());
    db.create_namespace("other").unwrap();
    db.create_namespace("going").unwrap();
    let going = db.get_store_by_name("going").unwrap();

    // The removal's flush stops on this, so it stays in progress.
    let held = going.lsm.hold_memtable();
    let remover = {
        let db = Arc::clone(&db);
        std::thread::spawn(move || db.remove_namespace("going").unwrap())
    };
    while db.namespace_exists("going") {
        std::thread::yield_now();
    }
    std::thread::sleep(Duration::from_millis(50)); // into the flush

    let (tx, rx) = std::sync::mpsc::channel();
    {
        let db = Arc::clone(&db);
        std::thread::spawn(move || tx.send(matches!(db.lookup_open_store("other"), OpenStoreLookup::Open(..))).unwrap());
    }
    let found = rx.recv_timeout(Duration::from_secs(5));
    assert!(!remover.is_finished(), "the removal must still be in progress");
    drop(held);
    remover.join().unwrap();
    assert_eq!(found, Ok(true), "the lookup waited for the removal");
    db.shutdown().unwrap();
}

/// A create that fails after the registry saved the name must take the name back:
/// the next open of the database would try to open that namespace and fail.
#[test]
fn a_failed_namespace_create_leaves_the_database_openable() {
    let dir = TempDir::new().unwrap();
    {
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        // A file where the namespace's directory has to go makes the open fail.
        std::fs::write(crate::layout::namespace_data_dir(dir.path(), "blocked"), b"in the way").unwrap();
        assert!(db.create_namespace("blocked").is_err());
        assert!(!db.list_namespaces().iter().any(|(name, _)| name == "blocked"));
        // Without the name in the registry, a later attempt starts over.
        std::fs::remove_file(crate::layout::namespace_data_dir(dir.path(), "blocked")).unwrap();
        db.create_namespace("blocked").unwrap();
        db.remove_namespace("blocked").unwrap();
        std::fs::write(crate::layout::namespace_data_dir(dir.path(), "blocked"), b"in the way").unwrap();
        assert!(db.create_namespace("blocked").is_err());
    }
    Database::open(dir.path(), create_db_config()).expect("the database still opens");
}
