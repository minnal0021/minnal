//! Tests of the engine handle's reads (iteration, prefix scans, ranges) and
//! its open and close, including crashes and the files a database holds.

use crate::db::config::{DbConfig, ScheduledTaskConfig, SyncConfig, ThresholdConfig};
use crate::db::error::Result;
use crate::db::facade::Db;
use crate::storage::lsm::tree::LSMConfig;
use std::time::Duration;
use tempfile::TempDir;

fn create_db_config() -> DbConfig {
    let gc_interval = Duration::from_secs(5);
    let wal_gc_interval = Duration::from_secs(5);
    let lsm_compaction_interval = Duration::from_secs(5);

    let sync_config = SyncConfig::default();
    let threshold_config = ThresholdConfig::new(2.5);
    let scheduled_task_config = ScheduledTaskConfig::new(gc_interval, wal_gc_interval, lsm_compaction_interval);
    let lsm_config = LSMConfig::default();
    let mut config = DbConfig::new(threshold_config, scheduled_task_config, sync_config, lsm_config);
    config.num_buckets = crate::support::TEST_NUM_BUCKETS;
    config
}

#[test]
fn test_iter_all() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db = Db::open_with_config(temp_dir.path(), create_db_config())?;

    for i in 0..10 {
        let key = format!("key{:02}", i).into_bytes();
        let value = format!("value{}", i).into_bytes();
        db.put(&key, &value)?;
    }

    assert_eq!(db.iter()?.len(), 10);
    Ok(())
}

#[test]
fn test_prefix_scan() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db = Db::open_with_config(temp_dir.path(), create_db_config())?;

    db.put(b"user:1", b"Alice")?;
    db.put(b"user:2", b"Bob")?;
    db.put(b"product:1", b"Laptop")?;

    assert_eq!(db.scan_prefix(b"user:")?.len(), 2);
    Ok(())
}

#[test]
fn shutdown_marks_db_as_closed() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");
    let db = Db::open_with_config(temp_dir.path(), create_db_config()).expect("failed to open db");

    db.put(b"key1", b"value1").expect("failed to put key1");
    assert_eq!(db.get(b"key1").expect("Failed to get key1"), Some(b"value1".to_vec()));

    db.shutdown().expect("Failed to close database");
    assert!(db.is_closed());
}

#[test]
fn test_reopen_after_close() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let path = temp_dir.path().to_path_buf();

    {
        let db = Db::open_with_config(&path, create_db_config())?;
        db.put(b"key1", b"value1")?;
        db.put(b"key2", b"value2")?;
        db.shutdown()?;
    }

    {
        let db = Db::open_with_config(&path, create_db_config())?;
        assert_eq!(db.get(b"key1")?, Some(b"value1".to_vec()));
        assert_eq!(db.get(b"key2")?, Some(b"value2".to_vec()));
    }

    Ok(())
}

#[test]
fn test_auto_close_on_drop() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let path = temp_dir.path().to_path_buf();

    {
        let db = Db::open_with_config(&path, create_db_config())?;
        db.put(b"key1", b"value1")?;
        db.shutdown()?;
    }

    let db = Db::open_with_config(&path, create_db_config())?;
    assert_eq!(db.get(b"key1")?, Some(b"value1".to_vec()));

    Ok(())
}

/// Regression: skip list used to reject u128 value 0 (encoding of
/// ShardedValuePointer { bucket: 0, page_offset: 0, segment_id: 0 }).
#[test]
fn test_zero_byte_values_roundtrip() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db = Db::open_with_config(temp_dir.path(), create_db_config())?;

    db.put(b"key_zero", b"\x00")?;
    assert_eq!(db.get(b"key_zero")?, Some(b"\x00".to_vec()));

    db.put(b"key_leading_zero", b"\x00\x01\x02")?;
    assert_eq!(db.get(b"key_leading_zero")?, Some(b"\x00\x01\x02".to_vec()));

    db.put(b"key_all_zeros", b"\x00\x00\x00\x00")?;
    assert_eq!(db.get(b"key_all_zeros")?, Some(b"\x00\x00\x00\x00".to_vec()));

    db.put(b"ke\x00y", b"value_with_zero_key")?;
    assert_eq!(db.get(b"ke\x00y")?, Some(b"value_with_zero_key".to_vec()));

    db.put(b"key_zero", b"\x00\xff")?;
    assert_eq!(db.get(b"key_zero")?, Some(b"\x00\xff".to_vec()));

    Ok(())
}

#[test]
fn test_iter_returns_all_pairs_in_order() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db = Db::open_with_config(temp_dir.path(), create_db_config())?;

    db.put(b"c", b"3")?;
    db.put(b"a", b"1")?;
    db.put(b"b", b"2")?;

    let pairs = db.iter()?;
    assert_eq!(pairs.len(), 3);
    assert_eq!(pairs[0], (b"a".to_vec(), b"1".to_vec()));
    assert_eq!(pairs[1], (b"b".to_vec(), b"2".to_vec()));
    assert_eq!(pairs[2], (b"c".to_vec(), b"3".to_vec()));
    Ok(())
}

#[test]
fn test_keys_iter_no_value_log_access() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db = Db::open_with_config(temp_dir.path(), create_db_config())?;

    db.put(b"x", b"v1")?;
    db.put(b"y", b"v2")?;
    db.delete(b"x")?;

    assert_eq!(db.keys()?, vec![b"y".to_vec()]);
    Ok(())
}

#[test]
fn test_values_iter() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db = Db::open_with_config(temp_dir.path(), create_db_config())?;

    db.put(b"a", b"apple")?;
    db.put(b"b", b"banana")?;

    let vals: Vec<Vec<u8>> = db.iter()?.into_iter().map(|(_, v)| v).collect();
    assert_eq!(vals.len(), 2);
    assert!(vals.contains(&b"apple".to_vec()));
    assert!(vals.contains(&b"banana".to_vec()));
    Ok(())
}

#[test]
fn test_range_inclusive_start_exclusive_end() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db = Db::open_with_config(temp_dir.path(), create_db_config())?;

    for ch in b'a'..=b'e' {
        db.put(&[ch], &[ch])?;
    }

    let pairs = db.range(b"b", Some(b"d" as &[u8]))?;
    let keys: Vec<_> = pairs.iter().map(|(k, _)| k.clone()).collect();
    assert_eq!(keys, vec![b"b".to_vec(), b"c".to_vec()]);
    Ok(())
}

#[test]
fn test_range_no_upper_bound() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db = Db::open_with_config(temp_dir.path(), create_db_config())?;

    db.put(b"a", b"1")?;
    db.put(b"b", b"2")?;
    db.put(b"c", b"3")?;

    let pairs = db.range(b"b", None)?;
    let keys: Vec<_> = pairs.iter().map(|(k, _)| k.clone()).collect();
    assert_eq!(keys, vec![b"b".to_vec(), b"c".to_vec()]);
    Ok(())
}

#[test]
fn test_scan_prefix_returns_only_matching_keys() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db = Db::open_with_config(temp_dir.path(), create_db_config())?;

    db.put(b"user:alice", b"1")?;
    db.put(b"user:bob", b"2")?;
    db.put(b"order:1", b"3")?;

    let pairs = db.scan_prefix(b"user:")?;
    assert_eq!(pairs.len(), 2);
    assert!(pairs.iter().all(|(k, _)| k.starts_with(b"user:")));
    Ok(())
}

#[test]
fn test_scan_prefix_after_delete() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db = Db::open_with_config(temp_dir.path(), create_db_config())?;

    db.put(b"doc:1", b"alpha")?;
    db.put(b"doc:2", b"beta")?;
    db.put(b"doc:3", b"gamma")?;
    db.put(b"doc:4", b"delta")?;
    db.put(b"other:1", b"unrelated")?;

    let before = db.scan_prefix(b"doc:")?;
    assert_eq!(before.len(), 4, "expected 4 doc: records before delete");
    let keys_before: Vec<Vec<u8>> = before.iter().map(|(k, _)| k.clone()).collect();
    assert!(keys_before.contains(&b"doc:1".to_vec()));
    assert!(keys_before.contains(&b"doc:2".to_vec()));
    assert!(keys_before.contains(&b"doc:3".to_vec()));
    assert!(keys_before.contains(&b"doc:4".to_vec()));

    db.delete(b"doc:2")?;

    assert_eq!(db.get(b"doc:2")?, None);

    let after = db.scan_prefix(b"doc:")?;
    assert_eq!(after.len(), 3, "expected 3 doc: records after deleting doc:2, got {}", after.len());
    let keys_after: Vec<Vec<u8>> = after.iter().map(|(k, _)| k.clone()).collect();
    assert!(!keys_after.contains(&b"doc:2".to_vec()), "doc:2 should not appear after deletion");
    assert!(keys_after.contains(&b"doc:1".to_vec()));
    assert!(keys_after.contains(&b"doc:3".to_vec()));
    assert!(keys_after.contains(&b"doc:4".to_vec()));

    let val_1 = after.iter().find(|(k, _)| k == b"doc:1").map(|(_, v)| v.clone());
    assert_eq!(val_1, Some(b"alpha".to_vec()));

    assert_eq!(db.scan_prefix(b"other:")?.len(), 1);

    Ok(())
}

#[test]
fn test_range_after_delete_excludes_deleted_keys() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db = Db::open_with_config(temp_dir.path(), create_db_config())?;

    for ch in b'a'..=b'f' {
        db.put(&[ch], &[ch])?;
    }

    db.delete(b"c")?;
    db.delete(b"e")?;

    let pairs = db.range(b"b", Some(b"f" as &[u8]))?;
    let keys: Vec<_> = pairs.iter().map(|(k, _)| k.clone()).collect();
    assert_eq!(keys, vec![b"b".to_vec(), b"d".to_vec()]);

    let pairs2 = db.range(b"c", None)?;
    let keys2: Vec<_> = pairs2.iter().map(|(k, _)| k.clone()).collect();
    assert_eq!(keys2, vec![b"d".to_vec(), b"f".to_vec()]);

    Ok(())
}

#[test]
fn test_iter_after_delete_excludes_deleted_keys() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db = Db::open_with_config(temp_dir.path(), create_db_config())?;

    db.put(b"keep", b"yes")?;
    db.put(b"drop", b"no")?;
    db.delete(b"drop")?;

    let pairs = db.iter()?;
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0].0, b"keep".to_vec());
    Ok(())
}

#[test]
fn test_iter_empty_store() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db = Db::open_with_config(temp_dir.path(), create_db_config())?;

    assert_eq!(db.iter()?.len(), 0);
    assert_eq!(db.keys()?.len(), 0);
    assert_eq!(db.range(b"a", Some(b"z" as &[u8]))?.len(), 0);
    assert_eq!(db.scan_prefix(b"x")?.len(), 0);
    Ok(())
}

/// A no-WAL write takes a sequence the WAL never records. After a crash the
/// counter recovered from the WAL alone restarted below a no-WAL value
/// already flushed to an SSTable, so every later write to that key lost to
/// the stale value under highest-sequence-wins. `KVStore::set_seq_counter`
/// now raises the counter above every stored sequence at open.
#[tokio::test]
async fn a_write_after_a_crash_beats_a_flushed_no_wal_value() {
    let dir = TempDir::new().unwrap();
    let open = || crate::db::facade::AsyncDb::open_with_config(dir.path().to_owned(), create_db_config());
    let db = open().await.unwrap();
    let ns = db.namespace("n".to_string()).await.unwrap();
    ns.put(b"k".to_vec(), b"wal".to_vec()).await.unwrap();
    // Push the sequence well past what the WAL has seen, without the WAL.
    for i in 0..100u32 {
        ns.put_no_wal(i.to_be_bytes().to_vec(), b"x".to_vec()).await.unwrap();
    }
    ns.put_no_wal(b"k".to_vec(), b"no-wal".to_vec()).await.unwrap();
    db.flush_namespaces(vec!["n".to_string()]).await.unwrap();
    drop(ns);
    db.crash().await;

    let db = open().await.unwrap();
    let ns = db.namespace("n".to_string()).await.unwrap();
    assert_eq!(ns.get(b"k".to_vec()).await.unwrap().as_deref(), Some(&b"no-wal"[..]));
    ns.put(b"k".to_vec(), b"after".to_vec()).await.unwrap();
    assert_eq!(
        ns.get(b"k".to_vec()).await.unwrap().as_deref(),
        Some(&b"after"[..]),
        "a WAL put lost to the stale value"
    );
    ns.put_no_wal(b"k".to_vec(), b"after2".to_vec()).await.unwrap();
    assert_eq!(ns.get(b"k".to_vec()).await.unwrap().as_deref(), Some(&b"after2"[..]));
    ns.delete(b"k".to_vec()).await.unwrap();
    assert_eq!(ns.get(b"k".to_vec()).await.unwrap(), None, "a delete lost to the stale value");
    db.shutdown().await.unwrap();
}

/// The crash tests' simulated crash loses what a process kill loses: a
/// no-WAL write still in the memtable is gone, a WAL-backed one replays, and
/// a flushed no-WAL one stays. `KVStore`'s `Drop` used to flush every
/// memtable even after `crash()`, so no-WAL writes always survived and no
/// crash test could catch a missing barrier flush.
#[tokio::test]
async fn a_simulated_crash_loses_unflushed_no_wal_writes() {
    let dir = TempDir::new().unwrap();
    let open = || crate::db::facade::AsyncDb::open_with_config(dir.path().to_owned(), create_db_config());
    let db = open().await.unwrap();
    let ns = db.namespace("n".to_string()).await.unwrap();
    ns.put_no_wal(b"flushed".to_vec(), b"1".to_vec()).await.unwrap();
    db.flush_namespaces(vec!["n".to_string()]).await.unwrap();
    ns.put(b"wal".to_vec(), b"1".to_vec()).await.unwrap();
    ns.put_no_wal(b"memtable".to_vec(), b"1".to_vec()).await.unwrap();
    drop(ns);
    db.crash().await;

    let db = open().await.unwrap();
    let ns = db.namespace("n".to_string()).await.unwrap();
    assert!(ns.get(b"flushed".to_vec()).await.unwrap().is_some());
    assert!(ns.get(b"wal".to_vec()).await.unwrap().is_some());
    assert_eq!(ns.get(b"memtable".to_vec()).await.unwrap(), None, "an unflushed no-WAL write survived");
    db.shutdown().await.unwrap();
}

/// A database with its workers running, holding one written namespace.
#[cfg(target_os = "linux")]
async fn open_with_workers(dir: &std::path::Path) -> crate::db::facade::AsyncDb {
    let db = crate::db::facade::AsyncDb::open_with_config(dir.to_owned(), create_db_config())
        .await
        .unwrap();
    let ns = db.namespace("n".to_string()).await.unwrap();
    ns.put(b"k".to_vec(), b"v".to_vec()).await.unwrap();
    db.flush_namespaces(vec!["n".to_string()]).await.unwrap();
    assert!(!crate::support::open_files_under(dir).is_empty(), "set-up: the database holds files");
    db
}

/// However a database goes away, it closes every file. A leak here is what
/// runs a long-lived process, or a test binary, out of descriptors.
#[cfg(target_os = "linux")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_database_closes_every_file_when_it_goes() {
    let dir = TempDir::new().unwrap();
    let db = open_with_workers(dir.path()).await;
    db.shutdown().await.unwrap();
    drop(db);
    crate::support::assert_no_open_files_under(dir.path(), "after shutdown").await;

    let dir = TempDir::new().unwrap();
    drop(open_with_workers(dir.path()).await);
    crate::support::assert_no_open_files_under(dir.path(), "dropped without shutdown").await;

    let dir = TempDir::new().unwrap();
    open_with_workers(dir.path()).await.crash().await;
    crate::support::assert_no_open_files_under(dir.path(), "after a simulated crash").await;
}

/// Removing a namespace closes its files while the database stays open.
#[cfg(target_os = "linux")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn removing_a_namespace_closes_its_files() {
    let dir = TempDir::new().unwrap();
    let db = open_with_workers(dir.path()).await;
    let files = dir.path().join("ns_n");
    assert!(!crate::support::open_files_under(&files).is_empty(), "set-up: the namespace holds files");
    db.remove_namespace("n".to_string()).await.unwrap();
    crate::support::assert_no_open_files_under(&files, "after remove_namespace").await;
    db.shutdown().await.unwrap();
}
