#[cfg(test)]
use crate::db::config::{DbConfig, ScheduledTaskConfig, SyncConfig, ThresholdConfig};
#[cfg(test)]
use crate::db::error::{KVError, Result};
#[cfg(test)]
use crate::store::lsm::lsm_tree::LSMConfig;
#[cfg(test)]
use std::time::Duration;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::facade::{AsyncDb, Db};
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
    fn put_get_update_delete_round_trips() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let db = Db::open_with_config(temp_dir.path(), create_db_config())?;

        db.put(b"key1", b"value1")?;
        db.put(b"key2", b"value2")?;

        assert_eq!(db.get(b"key1")?, Some(b"value1".to_vec()));
        assert_eq!(db.get(b"key2")?, Some(b"value2".to_vec()));

        db.put(b"key1", b"new_value1")?;
        assert_eq!(db.get(b"key1")?, Some(b"new_value1".to_vec()));

        db.delete(b"key2")?;
        assert_eq!(db.get(b"key2")?, None);

        Ok(())
    }

    #[tokio::test]
    async fn test_concurrent_operations() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let async_db = AsyncDb::open_with_config(temp_dir.path().to_path_buf(), create_db_config()).await?;

        let mut handles = vec![];

        for i in 0..100 {
            let db_clone = async_db.clone();
            let handle = tokio::spawn(async move {
                let key = format!("key{}", i).into_bytes();
                let value = format!("value{}", i).into_bytes();
                db_clone.put(key, value).await
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap()?;
        }

        for i in 0..100 {
            let key = format!("key{}", i).into_bytes();
            let expected = format!("value{}", i).into_bytes();
            let value = async_db.get(key).await?;
            assert_eq!(value, Some(expected));
        }

        Ok(())
    }

    #[test]
    fn test_garbage_collection() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let db = Db::open_with_config(temp_dir.path(), create_db_config())?;

        for i in 0..5 {
            let key = format!("key{}", i).into_bytes();
            let value = vec![0u8; 50];
            db.put(&key, &value)?;
        }

        for round in 0..3 {
            for i in 0..5 {
                let key = format!("key{}", i).into_bytes();
                let value = vec![(round + 1) as u8; 50];
                db.put(&key, &value)?;
            }
        }

        db.garbage_collect()?;

        for i in 0..5 {
            let key = format!("key{}", i).into_bytes();
            assert!(db.get(&key)?.is_some());
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_iteration_after_gc() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let db = AsyncDb::open_with_config(temp_dir.path().to_path_buf(), create_db_config()).await?;

        for i in 0..50 {
            let key = format!("key{}", i).into_bytes();
            db.put(key, vec![0u8; 128]).await?;
        }

        for i in 0..50 {
            let key = format!("key{}", i).into_bytes();
            db.put(key, vec![1u8; 128]).await?;
        }

        db.garbage_collect().await?;

        assert!(db.iter().await.is_ok());

        db.shutdown().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_iteration_after_repeated_gc() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let db = AsyncDb::open_with_config(temp_dir.path().to_path_buf(), create_db_config()).await?;

        for i in 0..100 {
            let key = format!("key{}", i).into_bytes();
            db.put(key, vec![0u8; 256]).await?;
        }

        for i in 0..100 {
            let key = format!("key{}", i).into_bytes();
            db.put(key, vec![1u8; 256]).await?;
        }

        for _round in 0..10 {
            db.garbage_collect().await?;
            assert!(db.iter().await.is_ok());
        }

        db.shutdown().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_gc_write_stress() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let db = AsyncDb::open_with_config(temp_dir.path().to_path_buf(), create_db_config()).await?;

        let writer_count = 4usize;
        let writes_per_writer = 100usize;
        let mut handles = Vec::with_capacity(writer_count + 1);

        for writer_id in 0..writer_count {
            let db_clone = db.clone();
            let handle = tokio::spawn(async move {
                for i in 0..writes_per_writer {
                    let key = format!("stress:key:{}:{}", writer_id, i).into_bytes();
                    let value = vec![writer_id as u8; 64];
                    db_clone.put(key, value).await?;
                    if i % 10 == 0 {
                        tokio::task::yield_now().await;
                    }
                }
                Ok::<(), KVError>(())
            });
            handles.push(handle);
        }

        let gc_db = db.clone();
        let gc_handle = tokio::spawn(async move {
            for _ in 0..20 {
                gc_db.garbage_collect().await?;
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            Ok::<(), KVError>(())
        });
        handles.push(gc_handle);

        for handle in handles {
            handle.await.unwrap()?;
        }

        for writer_id in 0..writer_count {
            let key = format!("stress:key:{}:{}", writer_id, writes_per_writer - 1).into_bytes();
            assert!(db.get(key).await?.is_some());
        }

        db.shutdown().await?;
        Ok(())
    }

    // ── GC race regression tests ──────────────────────────────────────────────

    #[test]
    fn test_gc_write_before_gc_survives() -> Result<()> {
        // A write that completes immediately before GC starts must not be lost.
        let temp_dir = TempDir::new()?;
        let db = Db::open_with_config(temp_dir.path(), create_db_config())?;

        for round in 0..4u8 {
            for i in 0u32..50 {
                db.put(&format!("base:{i:04}").into_bytes(), &[round; 128])?;
            }
        }

        db.put(b"sentinel", b"sentinel_value")?;
        db.garbage_collect()?;

        assert_eq!(db.get(b"sentinel")?, Some(b"sentinel_value".to_vec()), "sentinel key lost after GC");
        Ok(())
    }

    #[test]
    fn test_gc_concurrent_writes_no_data_loss() -> Result<()> {
        // Regression: concurrent writes and GC must not lose any data.
        let temp_dir = TempDir::new()?;
        let db = Db::open_with_config(temp_dir.path(), create_db_config())?;

        for round in 0..4u8 {
            for i in 0u32..50 {
                db.put(&format!("base:{i:04}").into_bytes(), &[round; 128])?;
            }
        }

        const NUM_NEW: usize = 200;

        std::thread::scope(|s| {
            s.spawn(|| {
                for i in 0..NUM_NEW {
                    let key = format!("new:{i:04}").into_bytes();
                    db.put(&key, &[(i % 256) as u8; 64]).unwrap();
                }
            });
            s.spawn(|| {
                for _ in 0..5 {
                    db.garbage_collect().unwrap();
                }
            });
        });

        for i in 0..NUM_NEW {
            let key = format!("new:{i:04}").into_bytes();
            let expected = vec![(i % 256) as u8; 64];
            assert_eq!(db.get(&key)?, Some(expected), "key new:{i:04} was lost or corrupted after concurrent GC");
        }

        for i in 0u32..50 {
            let key = format!("base:{i:04}").into_bytes();
            assert!(db.get(&key)?.is_some(), "base key {i} lost after GC");
        }
        Ok(())
    }

    #[test]
    fn test_gc_multi_round_all_values_survive() -> Result<()> {
        // Multiple sequential GC rounds must not lose any values.
        let temp_dir = TempDir::new()?;
        let db = Db::open_with_config(temp_dir.path(), create_db_config())?;

        for round in 0..6u8 {
            for i in 0u32..40 {
                db.put(&format!("key:{i:04}").into_bytes(), &[round; 64])?;
            }
            db.garbage_collect()?;
        }

        for i in 0u32..40 {
            let key = format!("key:{i:04}").into_bytes();
            assert_eq!(db.get(&key)?, Some(vec![5u8; 64]), "key {i} has wrong value or is missing");
        }
        Ok(())
    }
}

#[cfg(test)]
mod iterator_tests {
    use super::*;
    use crate::db::facade::Db;
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
}
