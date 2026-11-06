//! Opening a `DocStore`.

use crate::doc_store::tests::test_support::*;
use crate::doc_store::*;

// ── Database lock ───────────────────────────────────────────────────────

/// A crash leaves the lock file on disk with no live owner. Recovery must
/// still be reachable: the previous existence-check scheme refused to open
/// here, so every `SIGKILL` needed a manual `rm db/.lock` before the WAL
/// could be replayed.
/// The lock lives as long as the database, not the `DocStore` handle. A
/// background worker's pass holds the database (an `Arc<Db>`), so it can
/// outlive a `DocStore` dropped without `shutdown`; while it does, its final
/// flush and compaction are still to run. Releasing the lock with the handle
/// let a reopen use the same files meanwhile.
#[tokio::test]
async fn the_lock_is_held_while_anything_still_holds_the_database() {
    let db_dir = TempDir::new().unwrap();

    let store = open_fresh(db_dir.path()).await;
    let pass = Arc::clone(&store.db); // what a worker's running pass holds
    drop(store);
    match DocStore::open_with_config(db_dir.path(), crate::doc_store::test_db_config()).await {
        Err(DocStoreError::StoreLocked { .. }) => {}
        Err(e) => panic!("expected StoreLocked, got {e:?}"),
        Ok(_) => panic!("a second instance opened the files while the first was still alive"),
    }

    drop(pass);
    let _reopened = open_fresh(db_dir.path()).await;
}

#[tokio::test]
async fn test_stale_lock_file_from_a_crash_does_not_block_open() {
    let db_dir = TempDir::new().unwrap();

    // Exactly what a killed process leaves behind.
    std::fs::create_dir_all(db_dir.path()).unwrap();
    std::fs::write(db_dir.path().join(".lock"), "999999\nminnal: stale\n").unwrap();

    let store = open_fresh(db_dir.path()).await;
    store.create(make_schema("docs", vec![])).await.unwrap();
}

/// The other direction still holds: a second store on the same directory is
/// refused while the first is open, and the error names the holder.
#[tokio::test]
async fn test_second_open_of_the_same_directory_is_refused() {
    let db_dir = TempDir::new().unwrap();

    let _held = open_fresh(db_dir.path()).await;

    match DocStore::open_with_config(db_dir.path(), crate::doc_store::test_db_config()).await {
        Err(DocStoreError::StoreLocked { owner_pid, .. }) => {
            assert_eq!(owner_pid, Some(std::process::id()));
        }
        Err(e) => panic!("expected StoreLocked, got {e:?}"),
        Ok(_) => panic!("two DocStores opened the same directory at once"),
    }
}

// ── Restart persistence ─────────────────────────────────────────────────

#[tokio::test]
async fn test_kv_schema_survives_restart() {
    let db_dir = TempDir::new().unwrap();

    // First run: create KV store and write data.
    {
        let store = open_fresh(db_dir.path()).await;
        store
            .create_kv(make_kv_schema("session", KvKeyType::Str, KvValueType::Str))
            .await
            .unwrap();
        store
            .kv_put("session", &serde_json::json!("tok1"), &serde_json::json!("abc"))
            .await
            .unwrap();
        store
            .kv_put("session", &serde_json::json!("tok2"), &serde_json::json!("def"))
            .await
            .unwrap();
    } // store dropped here, lock released

    // Second run: no create_kv call — schema must load automatically.
    {
        let store = open_fresh(db_dir.path()).await;
        assert_eq!(store.list_kv().unwrap().len(), 1, "KV schema not reloaded after restart");
        assert_eq!(
            store.kv_get("session", &serde_json::json!("tok1")).await.unwrap(),
            Some(serde_json::json!("abc"))
        );
        assert_eq!(
            store.kv_get("session", &serde_json::json!("tok2")).await.unwrap(),
            Some(serde_json::json!("def"))
        );
    }
}

#[tokio::test]
async fn test_kv_and_doc_stores_coexist_after_restart() {
    let db_dir = TempDir::new().unwrap();

    {
        let store = open_fresh(db_dir.path()).await;
        store.create(make_schema("docs", vec![])).await.unwrap();
        store.put("docs", DocId::U64(1), serde_json::json!({"x": 1})).await.unwrap();
        store.create_kv(make_kv_schema("cache", KvKeyType::Str, KvValueType::Str)).await.unwrap();
        store.kv_put("cache", &serde_json::json!("a"), &serde_json::json!("b")).await.unwrap();
    }

    {
        let store = open_fresh(db_dir.path()).await;
        // Doc store data intact
        assert_eq!(store.list().unwrap().len(), 1);
        assert_eq!(store.get("docs", DocId::U64(1)).await.unwrap(), Some(serde_json::json!({"x": 1})));
        // KV store data intact
        assert_eq!(store.list_kv().unwrap().len(), 1);
        assert_eq!(
            store.kv_get("cache", &serde_json::json!("a")).await.unwrap(),
            Some(serde_json::json!("b"))
        );
    }
}
