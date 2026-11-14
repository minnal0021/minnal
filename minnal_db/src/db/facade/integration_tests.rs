//! Integration tests for the `AsyncDb` facade, and for `Db`, the engine
//! handle it wraps.
//!
//! Lives here rather than in `tests/` because it reaches internal configuration
//! types.

#[cfg(test)]
use crate::db::config::{DbConfig, ScheduledTaskConfig, SyncConfig, ThresholdConfig};
#[cfg(test)]
use crate::db::error::{KVError, Result};
#[cfg(test)]
use crate::storage::lsm::tree::LSMConfig;
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

    // ── GC racing writes ──────────────────────────────────────────────────────

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
        // Concurrent writes and GC must not lose any data.
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

    // ── merge ─────────────────────────────────────────────────────────

    /// The typed value used by the `merge_typed` tests.
    #[derive(Debug, Clone, PartialEq, crate::rkyv_derives::Archive, crate::rkyv_derives::Serialize, crate::rkyv_derives::Deserialize)]
    struct Tally {
        count: u64,
        label: String,
    }

    fn u64_of(bytes: Option<&[u8]>) -> u64 {
        bytes.map_or(0, |b| u64::from_le_bytes(b.try_into().unwrap()))
    }

    #[test]
    fn merge_on_the_default_namespace_round_trips() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let db = Db::open_with_config(temp_dir.path(), create_db_config())?;

        let first = db.merge(b"visits", &2u64.to_le_bytes(), |existing, operand| {
            Ok(Some((u64_of(existing) + u64_of(Some(operand))).to_le_bytes().to_vec()))
        })?;
        assert_eq!(first, Some(2u64.to_le_bytes().to_vec()), "merge returns the value it wrote");

        db.merge(b"visits", &3u64.to_le_bytes(), |existing, operand| {
            Ok(Some((u64_of(existing) + u64_of(Some(operand))).to_le_bytes().to_vec()))
        })?;
        assert_eq!(u64_of(db.get(b"visits")?.as_deref()), 5);

        db.shutdown()?;
        Ok(())
    }

    /// A namespace's merge must stay inside its own keyspace.
    #[tokio::test]
    async fn merge_is_scoped_to_its_namespace() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let db = AsyncDb::open_with_config(temp_dir.path().to_path_buf(), create_db_config()).await?;

        db.put(b"k".to_vec(), b"default-value".to_vec()).await?;
        let orders = db.namespace("orders".to_string()).await?;
        let written = orders
            .merge(b"k".to_vec(), b"ns-value".to_vec(), |existing, operand| {
                assert_eq!(existing, None, "the namespace saw the default namespace's value");
                Ok(Some(operand.to_vec()))
            })
            .await?;

        assert_eq!(written, Some(b"ns-value".to_vec()));
        assert_eq!(orders.get(b"k".to_vec()).await?, Some(b"ns-value".to_vec()));
        assert_eq!(db.get(b"k".to_vec()).await?, Some(b"default-value".to_vec()));

        db.shutdown().await?;
        Ok(())
    }

    /// A namespace's no-WAL writes, `get_multiple` and `scan_prefixes_batch`.
    #[tokio::test]
    async fn a_namespace_has_batch_and_no_wal_operations() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let db = AsyncDb::open_with_config(temp_dir.path().to_path_buf(), create_db_config()).await?;
        let ns = db.namespace("mirror".to_string()).await?;
        let key = |prefix: u32, n: u8| {
            let mut k = prefix.to_be_bytes().to_vec();
            k.push(n);
            k
        };
        ns.put_no_wal(key(7, 1), b"a".to_vec()).await?;
        ns.put(key(7, 2), b"b".to_vec()).await?;
        ns.put(key(9, 1), b"c".to_vec()).await?;
        assert_eq!(
            ns.get_multiple(vec![key(7, 1), key(8, 1), key(7, 1)]).await?,
            vec![Some(b"a".to_vec()), None, Some(b"a".to_vec())]
        );
        let by_prefix = ns.scan_prefixes_batch(vec![7, 8]).await?;
        assert_eq!(by_prefix.iter().filter(|(p, _, _)| *p == 7).count(), 2);
        assert!(by_prefix.iter().all(|(p, _, _)| p == 7));
        ns.delete_no_wal(key(7, 1)).await?;
        assert_eq!(ns.get(key(7, 1)).await?, None);
        db.shutdown().await?;
        Ok(())
    }

    /// `get_typed` must validate what it reads: the bytes under a key need not
    /// be an archive of the requested type. It used unchecked access, so reading
    /// raw or differently-typed bytes followed archived pointers out of bounds —
    /// undefined behaviour from a safe function. Now it is an error.
    #[tokio::test]
    async fn get_typed_rejects_bytes_that_are_not_an_archive_of_that_type() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let db = AsyncDb::open_with_config(temp_dir.path().to_path_buf(), create_db_config()).await?;

        // Too short to hold a `Tally`: unchecked access read past the buffer.
        let short_key = rkyv::to_bytes::<rkyv::rancor::Error>(&"short".to_string()).unwrap().to_vec();
        db.put(short_key, vec![0x01, 0x02, 0x03]).await?;
        assert!(matches!(
            db.get_typed::<String, Tally>(&"short".to_string()).await,
            Err(KVError::Serialization(_))
        ));

        // A real `Tally` archive whose out-of-line label bytes are corrupted into
        // invalid UTF-8: unchecked access handed back an invalid `String`.
        let long_label = Tally {
            count: 1,
            label: "a label long enough to be stored out of line".to_string(),
        };
        let mut bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&long_label).unwrap().to_vec();
        let at = bytes.windows(7).position(|w| w == b"a label").expect("label bytes in the archive");
        bytes[at] = 0xFF;
        let corrupt_key = rkyv::to_bytes::<rkyv::rancor::Error>(&"corrupt".to_string()).unwrap().to_vec();
        db.put(corrupt_key, bytes).await?;
        assert!(matches!(
            db.get_typed::<String, Tally>(&"corrupt".to_string()).await,
            Err(KVError::Serialization(_))
        ));

        // A value stored as one type, read back as another.
        db.put_typed(&"n".to_string(), &7u8).await?;
        assert!(matches!(
            db.get_typed::<String, Tally>(&"n".to_string()).await,
            Err(KVError::Serialization(_))
        ));

        // A correctly typed value still round-trips.
        let tally = Tally {
            count: 3,
            label: "ok".to_string(),
        };
        db.put_typed(&"t".to_string(), &tally).await?;
        assert_eq!(db.get_typed::<String, Tally>(&"t".to_string()).await?, Some(tally));
        db.shutdown().await?;
        Ok(())
    }

    #[tokio::test]
    async fn merge_typed_accumulates_a_typed_value() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let db = AsyncDb::open_with_config(temp_dir.path().to_path_buf(), create_db_config()).await?;

        let step = Tally {
            count: 4,
            label: "clicks".to_string(),
        };
        let first: Option<Tally> = db
            .merge_typed(&"stats".to_string(), &step, |current, operand| {
                assert!(current.is_none());
                Ok(Some(operand))
            })
            .await?;
        assert_eq!(first.unwrap().count, 4);

        let second: Option<Tally> = db
            .merge_typed(&"stats".to_string(), &step, |current, operand| {
                let current = current.expect("the stored tally should be visible");
                Ok(Some(Tally {
                    count: current.count + operand.count,
                    label: current.label,
                }))
            })
            .await?;
        let second = second.expect("merge_typed returns the new value");
        assert_eq!(second.count, 8);
        assert_eq!(second.label, "clicks");

        // And it is what actually landed in storage.
        let stored: Option<Tally> = db.get_typed(&"stats".to_string()).await?;
        assert_eq!(stored.unwrap().count, 8);

        db.shutdown().await?;
        Ok(())
    }

    #[tokio::test]
    async fn merge_typed_aborts_without_touching_the_stored_value() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let db = AsyncDb::open_with_config(temp_dir.path().to_path_buf(), create_db_config()).await?;

        let tally = Tally {
            count: 1,
            label: "clicks".to_string(),
        };
        db.put_typed(&"stats".to_string(), &tally).await?;

        let err = db
            .merge_typed(&"stats".to_string(), &tally, |_current, _operand| -> Result<Option<Tally>> {
                Err(KVError::MergeAborted("declined".into()))
            })
            .await
            .unwrap_err();
        assert!(matches!(err, KVError::MergeAborted(_)), "unexpected error: {err:?}");

        let stored: Option<Tally> = db.get_typed(&"stats".to_string()).await?;
        assert_eq!(stored.unwrap().count, 1);

        db.shutdown().await?;
        Ok(())
    }

    /// The async twin of the coordinator's concurrency test — 100 tasks racing
    /// on one key through `spawn_blocking`.
    #[tokio::test]
    async fn concurrent_async_merges_lose_no_updates() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let db = AsyncDb::open_with_config(temp_dir.path().to_path_buf(), create_db_config()).await?;

        let mut handles = vec![];
        for _ in 0..100 {
            let db = db.clone();
            handles.push(tokio::spawn(async move {
                db.merge(b"hits".to_vec(), 1u64.to_le_bytes().to_vec(), |existing, operand| {
                    Ok(Some((u64_of(existing) + u64_of(Some(operand))).to_le_bytes().to_vec()))
                })
                .await
                .unwrap()
            }));
        }
        for handle in handles {
            handle.await.unwrap();
        }

        assert_eq!(u64_of(db.get(b"hits".to_vec()).await?.as_deref()), 100);

        db.shutdown().await?;
        Ok(())
    }

    #[tokio::test]
    async fn async_namespace_merge_and_merge_typed_work() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let db = AsyncDb::open_with_config(temp_dir.path().to_path_buf(), create_db_config()).await?;
        let ns = db.namespace("orders".to_string()).await?;

        let written = ns
            .merge(b"o1".to_vec(), b"shipped".to_vec(), |existing, operand| {
                assert_eq!(existing, None);
                Ok(Some(operand.to_vec()))
            })
            .await?;
        assert_eq!(written, Some(b"shipped".to_vec()));

        let tally: Option<Tally> = db
            .merge_typed(&"stats".to_string(), &Tally { count: 3, label: "a".into() }, |current, operand| {
                Ok(Some(Tally {
                    count: current.map_or(0, |c| c.count) + operand.count,
                    label: operand.label,
                }))
            })
            .await?;
        assert_eq!(tally.unwrap().count, 3);

        // Deleting through the async surface.
        assert_eq!(ns.merge(b"o1".to_vec(), Vec::new(), |_, _| Ok(None)).await?, None);
        assert_eq!(ns.get(b"o1".to_vec()).await?, None);

        db.shutdown().await?;
        Ok(())
    }

    /// `existing_namespace` finds a namespace only while it exists, and never
    /// creates one: a caller racing a removal must not bring it back.
    #[tokio::test]
    async fn existing_namespace_never_creates_one() {
        let temp_dir = TempDir::new().unwrap();
        let db = AsyncDb::open_with_config(temp_dir.path().to_path_buf(), create_db_config())
            .await
            .unwrap();
        let registered = |db: &AsyncDb| db.list_namespaces().into_iter().any(|(n, _)| n == "ns");
        assert!(db.existing_namespace("ns".to_owned()).await.unwrap().is_none());
        assert!(!registered(&db), "the lookup created the namespace");
        db.namespace("ns".to_owned())
            .await
            .unwrap()
            .put(b"k".to_vec(), b"v".to_vec())
            .await
            .unwrap();
        let ns = db.existing_namespace("ns".to_owned()).await.unwrap().expect("it exists");
        assert_eq!(ns.get(b"k".to_vec()).await.unwrap(), Some(b"v".to_vec()));
        db.remove_namespace("ns".to_owned()).await.unwrap();
        assert!(db.existing_namespace("ns".to_owned()).await.unwrap().is_none());
        assert!(!registered(&db), "the lookup brought the namespace back");
        db.shutdown().await.unwrap();
    }
}
