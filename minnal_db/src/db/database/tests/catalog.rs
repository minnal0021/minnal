//! The system catalog: namespaces, TTL settings and field indexes kept in the
//! `system_catalog` namespace, replayed first after a crash.

use super::*;
use std::sync::atomic::Ordering;
use std::time::Duration;

/// Everything a registry change saves, made while the catalog is not flushed
/// (its records in the WAL only), survives a crash: the catalog's records are
/// replayed before the namespaces they list are opened, and the namespaces'
/// own writes are then replayed into them.
#[test]
fn catalog_records_only_in_the_wal_survive_a_crash() {
    let dir = TempDir::new().unwrap();
    let (users, status, cache) = {
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        db.catalog_flush_paused.store(true, Ordering::SeqCst);
        let users = db.create_namespace("users").unwrap();
        let status = db.register_index_field(users, "status", IndexValueType::Str).unwrap();
        db.put_ns(users, b"u1", br#"{"status":"active"}"#).unwrap();
        let cache = db.create_namespace("cache").unwrap();
        db.update_registry(|r| r.set_ttl_config(cache, Duration::from_secs(60), 7)).unwrap();
        let m = db.wal_metadata.read().clone();
        assert!(
            m.replay_from(NamespaceId::CATALOG) < m.tail,
            "setup: the catalog's records are in the WAL only"
        );
        db.crash();
        (users, status, cache)
    };
    for round in 0..2 {
        // The second round reopens a database whose first reopen replayed the
        // catalog: replay must have flushed it before retiring the log.
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        assert_eq!(db.get_namespace_id("users"), Some(users), "round {round}");
        assert_eq!(db.get_ns(users, b"u1").unwrap(), Some(br#"{"status":"active"}"#.to_vec()));
        let fields = db.list_index_fields(users);
        assert_eq!(fields.len(), 1);
        assert_eq!((fields[0].field_id, fields[0].field_name.as_str()), (status, "status"));
        assert_eq!(db.registry.read().ttl_config(cache), Some((Duration::from_secs(60), 7)));
        let next = db.create_namespace(&format!("next{round}")).unwrap();
        assert!(next > users && next > cache && next.as_u64() > status, "the counter carries on: {next}");
        db.crash();
    }
}

/// A database keeps no registry file and no per-namespace config file: the
/// catalog is the only record.
#[test]
fn no_registry_or_config_file_is_written() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    let ns = db.create_namespace("users").unwrap();
    activate_status_index(&db, ns);
    db.update_registry(|r| r.set_ttl_config(ns, Duration::from_secs(60), 7)).unwrap();
    db.shutdown().unwrap();
    let mut found = Vec::new();
    let mut stack = vec![dir.path().to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if matches!(p.file_name().and_then(|n| n.to_str()), Some("namespace_registry" | "config.json")) {
                found.push(p);
            }
        }
    }
    assert!(found.is_empty(), "{found:?}");
    assert!(crate::layout::namespace_data_dir(dir.path(), crate::CATALOG_NAMESPACE).is_dir());
}

/// The catalog never holds back the main WAL: it is flushed after every
/// change, so a flush of the namespaces written retires every entry.
#[test]
fn the_catalog_does_not_hold_the_wal_cut() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    let ns = db.create_namespace("data").unwrap();
    activate_status_index(&db, ns);
    db.put_ns(ns, b"k", br#"{"status":"a"}"#).unwrap();
    db.get_store(ns).unwrap().flush_memtable_to_level0().unwrap();
    let m = db.wal_metadata.read().clone();
    assert_eq!(m.cut, m.tail, "the catalog's records hold the cut");
    db.shutdown().unwrap();
}

/// A dropped namespace's catalog records go with it, and a re-created name
/// gets a new id that survives a reopen.
#[test]
fn a_dropped_namespace_leaves_no_record_and_a_recreated_one_gets_a_new_id() {
    let dir = TempDir::new().unwrap();
    let (first, second) = {
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        let first = db.create_namespace("users").unwrap();
        activate_status_index(&db, first);
        db.remove_namespace("users").unwrap();
        let second = db.create_namespace("users").unwrap();
        db.shutdown().unwrap();
        (first, second)
    };
    assert_ne!(first, second);
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    assert_eq!(db.get_namespace_id("users"), Some(second));
    assert!(db.list_index_fields(second).is_empty(), "the old namespace's field went with it");
    let catalog = db.get_store(NamespaceId::CATALOG).unwrap();
    let field_records = catalog.scan_prefix_batch(b"field/").unwrap();
    assert!(field_records.is_empty(), "{} field records left", field_records.len());
    db.shutdown().unwrap();
}

/// A reopen that replays only catalog records flushes them before it retires
/// the log: otherwise a second crash would lose them (nothing else replayed,
/// so nothing else would have made recovery flush).
#[test]
fn replaying_only_catalog_records_flushes_them_before_the_log_is_retired() {
    let dir = TempDir::new().unwrap();
    let cache = {
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        db.catalog_flush_paused.store(true, Ordering::SeqCst);
        let cache = db.create_namespace("cache").unwrap();
        db.crash();
        cache
    };
    Database::open(dir.path(), create_db_config()).unwrap().crash();
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    assert_eq!(db.get_namespace_id("cache"), Some(cache), "lost to the second crash");
    db.shutdown().unwrap();
}
