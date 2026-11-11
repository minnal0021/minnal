//! The async facade over the coordinator, and its workers.

use super::*;

#[tokio::test]
async fn async_db_basic_round_trip() {
    let dir = TempDir::new().unwrap();
    let db = crate::db::facade::AsyncDb::open_with_config(dir.path().to_path_buf(), create_db_config())
        .await
        .unwrap();

    db.put(b"key1".to_vec(), b"value1".to_vec()).await.unwrap();
    assert_eq!(db.get(b"key1".to_vec()).await.unwrap(), Some(b"value1".to_vec()));

    db.delete(b"key1".to_vec()).await.unwrap();
    assert_eq!(db.get(b"key1".to_vec()).await.unwrap(), None);

    db.shutdown().await.unwrap();
}

/// The WAL GC worker flushes writes left in an idle memtable, so the WAL they
/// hold can be reclaimed, and stops cleanly on shutdown.
///
/// Nothing else here seals the memtable, so seeing the pending bytes reach zero
/// means the worker did it. Check for that with a deadline, not at one moment:
/// a tick can land at any point after the writes.
#[tokio::test]
async fn async_db_wal_gc_worker_reclaims_and_stops() {
    let dir = TempDir::new().unwrap();
    let db = crate::db::facade::AsyncDb::open_without_workers(dir.path().to_path_buf(), create_db_config())
        .await
        .unwrap();
    db.enable_wal_gc_worker(Duration::from_millis(50)).await.unwrap();

    for i in 0..200u32 {
        db.put(format!("key{i}").into_bytes(), vec![b'v'; 128]).await.unwrap();
    }

    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        let (_reclaimed, remaining) = db.garbage_collect_wal().await.unwrap();
        if remaining == 0 {
            break;
        }
        assert!(std::time::Instant::now() < deadline, "the worker left {remaining} bytes pending");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(db.get(b"key0".to_vec()).await.unwrap(), Some(vec![b'v'; 128]), "the flush kept the data");

    // Shutdown must stop the worker without hanging or erroring.
    db.shutdown().await.unwrap();
}

/// `AsyncDb::shutdown` must stop the index checkpoint worker.
///
/// It did not: the shutdown sequence stopped the TTL, WAL GC, LSM compaction
/// and value-log GC workers but silently skipped this one. The gap was
/// invisible because a since-deleted second async wrapper *did* stop it, so
/// nothing flagged the worker's shutdown path as unused. A checkpoint tick
/// could therefore land after the database was closed.
#[tokio::test]
async fn async_db_shutdown_stops_the_index_checkpoint_worker() {
    let dir = TempDir::new().unwrap();
    let db = crate::db::facade::AsyncDb::open_without_workers(dir.path().to_path_buf(), create_db_config())
        .await
        .unwrap();
    // A short interval so a surviving worker would keep firing.
    db.enable_index_checkpoint_worker(Duration::from_millis(20)).await.unwrap();
    db.put(b"k".to_vec(), b"v".to_vec()).await.unwrap();

    db.shutdown().await.unwrap();
    assert!(
        !db.index_checkpoint_worker_active().await,
        "shutdown must stop the checkpoint worker, not leave it running"
    );

    // Ticks after shutdown must not blow up: give the interval several
    // periods to fire if the task were still alive.
    tokio::time::sleep(Duration::from_millis(100)).await;
}

/// An open starts every worker, and a shutdown stops them all.
#[tokio::test]
async fn an_open_starts_every_worker_and_a_shutdown_stops_them() {
    let dir = TempDir::new().unwrap();
    let db = crate::db::facade::AsyncDb::open_with_config(dir.path().to_path_buf(), create_db_config())
        .await
        .unwrap();
    assert_eq!(
        db.running_workers().await,
        ["wal_gc", "lsm_compaction", "value_log_gc", "index_checkpoint"],
        "every worker runs from the open"
    );

    db.put(b"k".to_vec(), b"v".to_vec()).await.unwrap();
    assert_eq!(db.get(b"k".to_vec()).await.unwrap(), Some(b"v".to_vec()));

    db.shutdown().await.unwrap();
    assert!(db.running_workers().await.is_empty(), "the shutdown stopped them all");
}

#[tokio::test]
async fn async_db_namespaces_are_isolated() {
    let dir = TempDir::new().unwrap();
    let db = crate::db::facade::AsyncDb::open_with_config(dir.path().to_path_buf(), create_db_config())
        .await
        .unwrap();

    let users = db.namespace("users".to_string()).await.unwrap();
    db.put(b"global".to_vec(), b"g_val".to_vec()).await.unwrap();
    users.put(b"user:1".to_vec(), b"alice".to_vec()).await.unwrap();

    // Neither namespace can see the other's keys.
    assert_eq!(db.get(b"user:1".to_vec()).await.unwrap(), None);
    assert_eq!(users.get(b"user:1".to_vec()).await.unwrap(), Some(b"alice".to_vec()));
    assert_eq!(users.get(b"global".to_vec()).await.unwrap(), None);

    db.shutdown().await.unwrap();
}
