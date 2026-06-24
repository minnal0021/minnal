use minnal_db::lsm::LSMConfig;
use minnal_db::{AsyncDb, DbConfig, KVError, ScheduledTaskConfig, SyncConfig, ThresholdConfig};
use minnal_db::{DEFAULT_NAMESPACE_ID, ExtractorFn, FieldId, FieldMeta, IndexValue, IndexValueType};
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use std::sync::Arc;
use std::time::Duration;

type Result<T> = std::result::Result<T, KVError>;

// ── helpers ──────────────────────────────────────────────────────────────────

fn print_section(label: &str) {
    log::info!("\n── {} ──────────────────────────────────────────", label);
}

fn show_kv(key: &[u8], value: &[u8]) {
    log::info!("  {} = {}", String::from_utf8_lossy(key), String::from_utf8_lossy(value),);
}

// ── demo functions ───────────────────────────────────────────────────────────

/// put / get / update / delete on the default namespace
async fn demo_basic_crud(db: &AsyncDb) -> Result<()> {
    print_section("CRUD (default namespace)");

    db.put(b"user:1".to_vec(), b"Alice".to_vec()).await?;
    db.put(b"user:2".to_vec(), b"Bob".to_vec()).await?;
    db.put(b"user:3".to_vec(), b"Charlie".to_vec()).await?;
    db.put(b"order:1".to_vec(), b"Laptop".to_vec()).await?;
    db.put(b"order:2".to_vec(), b"Phone".to_vec()).await?;
    log::info!("inserted 5 keys");

    let val = db.get(b"user:1".to_vec()).await?;
    log::info!("get user:1 = {:?}", val.as_deref().map(String::from_utf8_lossy));

    let missing = db.get(b"user:999".to_vec()).await?;
    log::info!("get user:999 (missing) = {:?}", missing);

    db.put(b"user:2".to_vec(), b"Bobby".to_vec()).await?;
    let updated = db.get(b"user:2".to_vec()).await?;
    log::info!("update user:2 -> {:?}", updated.as_deref().map(String::from_utf8_lossy));

    db.delete(b"user:3".to_vec()).await?;
    let after_delete = db.get(b"user:3".to_vec()).await?;
    log::info!("delete user:3 -> {:?}", after_delete);

    Ok(())
}

/// iter, keys, range, scan_prefix
async fn demo_iteration(db: &AsyncDb) -> Result<()> {
    print_section("Iteration");

    log::info!("iter (all):");
    let all = db.iter().await?;
    for (k, v) in &all {
        show_kv(k, v);
    }

    log::info!("keys:");
    let keys = db.keys().await?;
    for k in &keys {
        log::info!("  {}", String::from_utf8_lossy(k));
    }

    log::info!("range [order:1, user:1):");
    let range = db.range(b"order:1".to_vec(), Some(b"user:1".to_vec())).await?;
    for (k, v) in &range {
        show_kv(k, v);
    }

    log::info!("scan_prefix \"user:\":");
    let prefix = db.scan_prefix(b"user:".to_vec()).await?;
    for (k, v) in &prefix {
        show_kv(k, v);
    }

    Ok(())
}

/// Scoped namespace handles with isolation
async fn demo_namespaces(db: &AsyncDb) -> Result<()> {
    print_section("Namespaces");

    let users = db.namespace("users".to_string()).await?;
    let orders = db.namespace("orders".to_string()).await?;
    log::info!("created namespaces: users (id={}), orders (id={})", users.id(), orders.id());

    users.put(b"alice".to_vec(), b"admin".to_vec()).await?;
    users.put(b"bob".to_vec(), b"viewer".to_vec()).await?;
    orders.put(b"ord-100".to_vec(), b"shipped".to_vec()).await?;
    log::info!("inserted: 2 users, 1 order");

    let from_default = db.get(b"alice".to_vec()).await?;
    let from_users = users.get(b"alice".to_vec()).await?;
    log::info!("default.get(alice) = {:?} (isolated)", from_default);
    log::info!("users.get(alice)   = {:?}", from_users.as_deref().map(String::from_utf8_lossy));

    log::info!("users.keys:");
    let user_keys = users.keys().await?;
    for k in &user_keys {
        log::info!("  {}", String::from_utf8_lossy(k));
    }

    log::info!("all namespaces: {:?}", db.list_namespaces());

    users.delete(b"bob".to_vec()).await?;
    log::info!("deleted 'bob' from users");

    db.remove_namespace("users".to_string()).await?;
    db.remove_namespace("orders".to_string()).await?;
    log::info!("removed namespaces, remaining: {:?}", db.list_namespaces());

    Ok(())
}

/// stats, garbage_collect, compact
async fn demo_maintenance(db: &AsyncDb) -> Result<()> {
    print_section("Maintenance");

    let stats = db.stats();
    log::info!("stats: waste_ratio={:.2}%, live_bytes={}", stats.waste_ratio, stats.live_bytes);

    log::info!("waste_ratio() = {:.2}%", db.waste_ratio());

    let gc = db.garbage_collect().await?;
    log::info!(
        "garbage_collect: reclaimed={} bytes, live={} bytes, runs={}",
        gc.bytes_reclaimed,
        gc.bytes_live,
        gc.gc_run_count,
    );

    db.compact().await?;
    log::info!("compact: done");

    Ok(())
}

// ── Typed data models ────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Archive, RkyvSerialize, RkyvDeserialize)]
struct UserId(u64);

#[derive(Debug, Clone, PartialEq, Archive, RkyvSerialize, RkyvDeserialize)]
struct UserProfile {
    name: String,
    email: String,
    age: u32,
}

/// Typed CRUD and iteration — no manual serialization needed
async fn demo_typed_api(db: &AsyncDb) -> Result<()> {
    print_section("Typed API (rkyv ser/de)");

    db.put_typed(
        &UserId(1),
        &UserProfile {
            name: "Alice".into(),
            email: "alice@example.com".into(),
            age: 30,
        },
    )
    .await?;
    db.put_typed(
        &UserId(2),
        &UserProfile {
            name: "Bob".into(),
            email: "bob@example.com".into(),
            age: 25,
        },
    )
    .await?;
    db.put_typed(
        &UserId(3),
        &UserProfile {
            name: "Charlie".into(),
            email: "charlie@example.com".into(),
            age: 35,
        },
    )
    .await?;
    log::info!("put_typed: inserted 3 user profiles");

    let got: Option<UserProfile> = db.get_typed(&UserId(1)).await?;
    log::info!("get_typed UserId(1) = {:?}", got.as_ref().map(|p| &p.name));

    db.put_typed(
        &UserId(2),
        &UserProfile {
            name: "Bobby".into(),
            email: "bobby@new.com".into(),
            age: 26,
        },
    )
    .await?;
    let got: Option<UserProfile> = db.get_typed(&UserId(2)).await?;
    log::info!("update UserId(2) -> {:?}", got.as_ref().map(|p| &p.name));

    db.delete_typed(&UserId(3)).await?;
    let got: Option<UserProfile> = db.get_typed(&UserId(3)).await?;
    log::info!("delete UserId(3) -> {:?}", got);

    // ── Typed iteration (in a dedicated namespace to avoid raw-byte keys) ──

    let typed_ns = db.namespace("typed_demo".to_string()).await?;
    typed_ns
        .put_typed(
            &UserId(1),
            &UserProfile {
                name: "Alice".into(),
                email: "alice@example.com".into(),
                age: 30,
            },
        )
        .await?;
    typed_ns
        .put_typed(
            &UserId(2),
            &UserProfile {
                name: "Bobby".into(),
                email: "bobby@new.com".into(),
                age: 26,
            },
        )
        .await?;

    log::info!("iter_typed (typed_demo ns):");
    let all: Vec<(UserId, UserProfile)> = typed_ns.iter_typed().await?;
    for (id, profile) in &all {
        log::info!("  UserId({}) = {} (age {})", id.0, profile.name, profile.age);
    }

    log::info!("keys_typed (typed_demo ns):");
    let keys: Vec<UserId> = typed_ns.keys_typed().await?;
    for id in &keys {
        log::info!("  UserId({})", id.0);
    }

    // ── Second namespace to show isolation ───────────────────────────

    let profiles = db.namespace("profiles".to_string()).await?;
    profiles
        .put_typed(
            &UserId(10),
            &UserProfile {
                name: "Dan".into(),
                email: "dan@example.com".into(),
                age: 20,
            },
        )
        .await?;
    profiles
        .put_typed(
            &UserId(20),
            &UserProfile {
                name: "Eve".into(),
                email: "eve@example.com".into(),
                age: 28,
            },
        )
        .await?;

    log::info!("namespace 'profiles' iter_typed:");
    let ns_all: Vec<(UserId, UserProfile)> = profiles.iter_typed().await?;
    for (id, profile) in &ns_all {
        log::info!("  UserId({}) = {} (age {})", id.0, profile.name, profile.age);
    }

    let from_default: Option<UserProfile> = db.get_typed(&UserId(10)).await?;
    log::info!("default.get_typed UserId(10) = {:?} (isolated)", from_default);

    db.remove_namespace("typed_demo".to_string()).await?;
    db.remove_namespace("profiles".to_string()).await?;

    Ok(())
}

/// Field index: register → activate → query
///
/// Documents are JSON-like byte strings.  Three fields are indexed:
///   - "status"   (Str)  e.g. "active" / "inactive"
///   - "age"      (Int)  e.g. 30
///   - "verified" (Bool) e.g. true
///
/// After inserting documents, a DSL query filters them using AND / OR / comparisons.
async fn demo_index(db: &AsyncDb) -> Result<()> {
    print_section("Field Index");

    let ns = DEFAULT_NAMESPACE_ID;

    // ── 1. Register fields ────────────────────────────────────────────
    let status_id: FieldId = db.register_index_field(ns, "status", IndexValueType::Str)?;
    let age_id: FieldId = db.register_index_field(ns, "age", IndexValueType::Int)?;
    let verified_id: FieldId = db.register_index_field(ns, "verified", IndexValueType::Bool)?;
    log::info!("registered fields: status={status_id}, age={age_id}, verified={verified_id}");

    // ── 2. Activate with extractors ───────────────────────────────────
    // Each extractor parses its field out of a simple "key=value,..." byte string.
    fn extract_str(field: &str, bytes: &[u8]) -> Option<IndexValue> {
        let s = std::str::from_utf8(bytes).ok()?;
        s.split(',')
            .find(|kv| kv.starts_with(field))?
            .split_once('=')
            .map(|(_, v)| IndexValue::Str(v.to_string()))
    }
    fn extract_int(field: &str, bytes: &[u8]) -> Option<IndexValue> {
        let s = std::str::from_utf8(bytes).ok()?;
        s.split(',')
            .find(|kv| kv.starts_with(field))?
            .split_once('=')?
            .1
            .parse::<i64>()
            .ok()
            .map(IndexValue::Int)
    }
    fn extract_bool(field: &str, bytes: &[u8]) -> Option<IndexValue> {
        let s = std::str::from_utf8(bytes).ok()?;
        s.split(',')
            .find(|kv| kv.starts_with(field))?
            .split_once('=')?
            .1
            .parse::<bool>()
            .ok()
            .map(IndexValue::Bool)
    }

    let status_extractor: ExtractorFn = Arc::new(|b| extract_str("status", b));
    let age_extractor: ExtractorFn = Arc::new(|b| extract_int("age", b));
    let verified_extractor: ExtractorFn = Arc::new(|b| extract_bool("verified", b));

    db.activate_field_index(ns, status_id, IndexValueType::Str, status_extractor).await?;
    db.activate_field_index(ns, age_id, IndexValueType::Int, age_extractor).await?;
    db.activate_field_index(ns, verified_id, IndexValueType::Bool, verified_extractor).await?;
    log::info!("all field indices active");

    // ── 3. Insert documents ───────────────────────────────────────────
    let docs: &[(&[u8], &[u8])] = &[
        (b"user:1", b"status=active,age=30,verified=true"),
        (b"user:2", b"status=active,age=17,verified=false"),
        (b"user:3", b"status=inactive,age=45,verified=true"),
        (b"user:4", b"status=active,age=22,verified=true"),
        (b"user:5", b"status=inactive,age=60,verified=false"),
    ];
    for (k, v) in docs {
        db.put(k.to_vec(), v.to_vec()).await?;
    }
    log::info!("inserted {} documents", docs.len());

    // ── 4. Query ──────────────────────────────────────────────────────
    // Active users aged 18 or over
    let keys = db.query_index(ns, "status = \"active\" AND age >= 18").await?;
    log::info!("status=active AND age>=18 ({} results):", keys.len());
    for k in &keys {
        log::info!("  {}", String::from_utf8_lossy(k));
    }

    // Verified users
    let keys = db.query_index(ns, "verified = true").await?;
    log::info!("verified=true ({} results):", keys.len());
    for k in &keys {
        log::info!("  {}", String::from_utf8_lossy(k));
    }

    // Inactive OR very young
    let keys = db.query_index(ns, "status = \"inactive\" OR age < 18").await?;
    log::info!("status=inactive OR age<18 ({} results):", keys.len());
    for k in &keys {
        log::info!("  {}", String::from_utf8_lossy(k));
    }

    // ── 5. Update and re-query ────────────────────────────────────────
    // user:2 turns 18 and gets verified
    db.put(b"user:2".to_vec(), b"status=active,age=18,verified=true".to_vec()).await?;
    let keys = db.query_index(ns, "status = \"active\" AND age >= 18").await?;
    log::info!("after updating user:2 — status=active AND age>=18 ({} results):", keys.len());
    for k in &keys {
        log::info!("  {}", String::from_utf8_lossy(k));
    }

    // ── 6. Delete and re-query ────────────────────────────────────────
    db.delete(b"user:3".to_vec()).await?;
    let keys = db.query_index(ns, "status = \"inactive\"").await?;
    log::info!("after deleting user:3 — status=inactive ({} results):", keys.len());
    for k in &keys {
        log::info!("  {}", String::from_utf8_lossy(k));
    }

    Ok(())
}

/// Open with explicit configuration
async fn demo_custom_config() -> Result<()> {
    print_section("Custom config");

    let db_path = "/tmp/minnal_db_custom";
    let _ = std::fs::remove_dir_all(db_path);

    let config = DbConfig::new(
        ThresholdConfig::new(2.5),
        ScheduledTaskConfig::new(Duration::from_secs(30), Duration::from_secs(30), Duration::from_secs(30)),
        SyncConfig::new(500),
        LSMConfig::default(),
    );

    let db = AsyncDb::open_with_config(db_path, config).await?;
    log::info!("opened with custom config (sync_every=500, gc_interval=30s)");

    db.put(b"config-test".to_vec(), b"works".to_vec()).await?;
    let val = db.get(b"config-test".to_vec()).await?;
    log::info!("get config-test = {:?}", val.as_deref().map(String::from_utf8_lossy));

    db.shutdown().await?;
    log::info!("shutdown complete");

    let _ = std::fs::remove_dir_all(db_path);
    Ok(())
}

/// Schema persistence: register once, survive restart
///
/// Demonstrates that indexed field definitions written to `config.json` are
/// automatically reloaded on the next open.  The caller only needs to supply
/// extractors (Rust closures — unpersistable) after a restart; no
/// `register_index_field` call is required.
///
/// Pass 1 — fresh database:
///   • register "status" (Str) and "age" (Int) fields
///   • activate with extractors, write three JSON documents
///   • shut down → schema saved to `ns_default/config.json`
///
/// Pass 2 — reopened database:
///   • list fields loaded from config.json (no re-registration)
///   • activate with fresh extractors
///   • run queries and verify results match
async fn demo_schema_persistence() -> Result<()> {
    print_section("Schema Persistence (doc store restart)");

    let db_path = "/tmp/minnal_db_schema_demo";
    let _ = std::fs::remove_dir_all(db_path);

    // ── Pass 1: first open — register schema, write data ─────────────────
    log::info!("[pass 1] opening database for the first time");
    {
        let db = AsyncDb::open(db_path).await?;

        let ns = DEFAULT_NAMESPACE_ID;
        let status_id = db.register_index_field(ns, "status", IndexValueType::Str)?;
        let age_id = db.register_index_field(ns, "age", IndexValueType::Int)?;
        log::info!("[pass 1] registered fields: status (id={status_id}), age (id={age_id})");

        let status_ex: ExtractorFn = Arc::new(|b: &[u8]| {
            let v: serde_json::Value = serde_json::from_slice(b).ok()?;
            Some(IndexValue::Str(v["status"].as_str()?.to_string()))
        });
        let age_ex: ExtractorFn = Arc::new(|b: &[u8]| {
            let v: serde_json::Value = serde_json::from_slice(b).ok()?;
            Some(IndexValue::Int(v["age"].as_i64()?))
        });
        db.activate_field_index(ns, status_id, IndexValueType::Str, status_ex).await?;
        db.activate_field_index(ns, age_id, IndexValueType::Int, age_ex).await?;

        let docs: &[(&[u8], &[u8])] = &[
            (b"user:1", br#"{"status":"active","age":30}"#),
            (b"user:2", br#"{"status":"inactive","age":25}"#),
            (b"user:3", br#"{"status":"active","age":17}"#),
        ];
        for (k, v) in docs {
            db.put(k.to_vec(), v.to_vec()).await?;
        }
        log::info!("[pass 1] inserted {} documents", docs.len());

        db.shutdown().await?;
        log::info!("[pass 1] shutdown — schema written to ns_default/config.json");
    }

    // ── Pass 2: reopen — no register_index_field ─────────────────────────
    log::info!("[pass 2] reopening database (no register_index_field call)");
    {
        let db = AsyncDb::open(db_path).await?;

        let ns = DEFAULT_NAMESPACE_ID;

        // Schema is populated from config.json automatically
        let fields: Vec<FieldMeta> = db.list_index_fields(ns);
        log::info!("[pass 2] fields loaded from config.json:");
        for f in &fields {
            log::info!("  field_id={} name={:<10} type={:?}", f.field_id, f.field_name, f.field_type);
        }

        // Look up the stored field IDs — the caller no longer needs to hard-code them
        let status_id = fields.iter().find(|f| f.field_name == "status").unwrap().field_id;
        let age_id = fields.iter().find(|f| f.field_name == "age").unwrap().field_id;

        // Extractors are closures — always supplied by the caller on each open
        let status_ex: ExtractorFn = Arc::new(|b: &[u8]| {
            let v: serde_json::Value = serde_json::from_slice(b).ok()?;
            Some(IndexValue::Str(v["status"].as_str()?.to_string()))
        });
        let age_ex: ExtractorFn = Arc::new(|b: &[u8]| {
            let v: serde_json::Value = serde_json::from_slice(b).ok()?;
            Some(IndexValue::Int(v["age"].as_i64()?))
        });
        db.activate_field_index(ns, status_id, IndexValueType::Str, status_ex).await?;
        db.activate_field_index(ns, age_id, IndexValueType::Int, age_ex).await?;
        log::info!("[pass 2] indices activated");

        let active = db.query_index(ns, "status = \"active\"").await?;
        log::info!("[pass 2] status=active ({} results):", active.len());
        for k in &active {
            log::info!("  {}", String::from_utf8_lossy(k));
        }

        let adults = db.query_index(ns, "status = \"active\" AND age >= 18").await?;
        log::info!("[pass 2] active AND age>=18 ({} results):", adults.len());
        for k in &adults {
            log::info!("  {}", String::from_utf8_lossy(k));
        }

        db.shutdown().await?;
        log::info!("[pass 2] shutdown complete");
    }

    let _ = std::fs::remove_dir_all(db_path);
    Ok(())
}

// ── main ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let db_path = "/tmp/minnal_db";
    let _ = std::fs::remove_dir_all(db_path);

    let log_dir = std::path::Path::new(db_path).join("log");
    std::fs::create_dir_all(&log_dir).expect("failed to create log directory");
    let file_appender = tracing_appender::rolling::daily(&log_dir, "minnal_db.log");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

    use tracing_subscriber::prelude::*;
    let filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".parse().unwrap());
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_subscriber::fmt::layer().with_writer(non_blocking).with_ansi(false))
        .with(filter)
        .init();

    let db = AsyncDb::open(db_path).await?;
    log::info!("[Main] Database opened at {db_path}");

    demo_basic_crud(&db).await?;
    demo_iteration(&db).await?;
    demo_namespaces(&db).await?;
    demo_typed_api(&db).await?;
    demo_index(&db).await?;
    demo_maintenance(&db).await?;

    print_section("Shutdown");
    db.shutdown().await?;
    log::info!("shutdown complete");

    let _ = std::fs::remove_dir_all(db_path);

    demo_custom_config().await?;
    demo_schema_persistence().await?;

    Ok(())
}
