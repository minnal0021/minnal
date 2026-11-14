//! Every key of the config file reaches the configuration it sets, a missing
//! key takes the same default as a config built in code, and an unknown key
//! or table is refused.

use std::path::PathBuf;
use std::time::Duration;

use super::*;
use crate::db::config::{DbConfig, ScheduledTaskConfig, SyncConfig, ThresholdConfig};
use crate::storage::lsm::tree::LSMConfig;

fn parse(text: &str) -> ConfigFile {
    ConfigFile::parse(text, &[]).map(|(c, _)| c).unwrap_or_else(|e| panic!("{e}\n{text}"))
}

fn parse_err(text: &str) -> String {
    match ConfigFile::parse(text, &[]) {
        Ok(_) => panic!("accepted:\n{text}"),
        Err(e) => e.to_string(),
    }
}

/// Every key set to a value different from its default.
const EVERY_KEY: &str = r#"
[storage]
db_path = "/srv/minnal/db"
log_dir = "/srv/minnal/log"

[memtable]
max_capacity = 1234

[sharding]
num_buckets = 3

[lsm]
compaction_threshold_percent = 77

[sync]
records_per_sync = 17
commit_delay_us = 250

[thresholds]
value_log_waste_threshold = 41.0
segment_gc_threshold = 7.5
tail_gc_min_garbage_pct = 55.0
index_blob_waste_threshold = 61.0
index_blob_backpressure_bytes = 1000
index_overlay_soft_bytes = 2000
index_overlay_hard_bytes = 3000
max_pinned_wal_segments = 9

[scheduled_tasks]
value_log_gc_interval_secs = 11
wal_gc_interval_secs = 12
lsm_compaction_interval_secs = 13
ttl_cleanup_interval_secs = 14
index_checkpoint_interval_ms = 15

[wal]
segment_size_bytes = 131072

[value_log]
segment_size_bytes = 524288
verify_checksums_on_read = true

[recovery]
fail_log_dir = "/srv/minnal/fail"

[lifecycle]
max_restarts = 8
restart_backoff_ms = 250
audit_retention_days = 9

[vector_index]
retry_wait_secs = 21
max_retries = 22
concurrency = 23

[semantic_search]
embedding_service_url = "http://embed:9000"
query_embedding_cache_ttl_secs = 31
partition_history_ttl_secs = 32
embedding_request_timeout_secs = 33
embedding_connect_timeout_secs = 4

[[semantic_search.supported_models]]
name = "Gemma"
dimension = 512

[semantic_search.maintenance]
threads = 5
rebuild_threads = 1
target_posting_size = 256
reassign_range = 32
merge_ratio = 0.125
skip_uncertain_moves = true
split_samples = 500
split_init_trials = 4
split_max_iters = 50
split_lambda_factor = 50.0

[semantic_search.search_defaults]
min_probes = 2
max_probes = 64
first_pass_top_k = 500
top_k = 50
"#;

/// The destructuring below names every field, with no `..`: a field added to
/// any of these types fails to compile here until it is given a key, mapped,
/// and checked.
#[test]
fn every_key_reaches_the_engine_config() {
    let file = parse(EVERY_KEY);
    assert_eq!(file.db_path(), PathBuf::from("/srv/minnal/db"));
    assert_eq!(file.storage.log_dir, "/srv/minnal/log");

    let DbConfig {
        threshold_config,
        sync_config,
        scheduled_task_config,
        lsm_config,
        num_buckets,
        skip_list_capacity,
        wal_segment_size,
        segment_size_bytes,
        fail_log_dir,
        verify_checksums_on_read,
    } = file.db_config();
    assert_eq!(num_buckets, 3);
    assert_eq!(skip_list_capacity, 1234);
    assert_eq!(wal_segment_size, 131072);
    assert_eq!(segment_size_bytes, 524288);
    assert_eq!(fail_log_dir, Some(PathBuf::from("/srv/minnal/fail")));
    assert!(verify_checksums_on_read);

    let ThresholdConfig {
        value_log_waste_threshold,
        segment_gc_threshold,
        tail_gc_min_garbage_pct,
        index_blob_waste_threshold,
        index_blob_backpressure_bytes,
        index_overlay_soft_bytes,
        index_overlay_hard_bytes,
        max_pinned_wal_segments,
    } = threshold_config;
    assert_eq!(value_log_waste_threshold, 41.0);
    assert_eq!(segment_gc_threshold, 7.5);
    assert_eq!(tail_gc_min_garbage_pct, Some(55.0));
    assert_eq!(index_blob_waste_threshold, 61.0);
    assert_eq!(index_blob_backpressure_bytes, 1000);
    assert_eq!(index_overlay_soft_bytes, 2000);
    assert_eq!(index_overlay_hard_bytes, 3000);
    assert_eq!(max_pinned_wal_segments, 9);

    let SyncConfig {
        records_per_sync,
        commit_delay_us,
    } = sync_config;
    assert_eq!((records_per_sync, commit_delay_us), (17, 250));

    let ScheduledTaskConfig {
        value_log_gc_interval,
        wal_gc_interval,
        lsm_compaction_interval,
        ttl_cleanup_interval,
        index_checkpoint_interval,
    } = scheduled_task_config;
    assert_eq!(value_log_gc_interval, Duration::from_secs(11));
    assert_eq!(wal_gc_interval, Duration::from_secs(12));
    assert_eq!(lsm_compaction_interval, Duration::from_secs(13));
    assert_eq!(ttl_cleanup_interval, Duration::from_secs(14));
    assert_eq!(index_checkpoint_interval, Duration::from_millis(15));

    // The directory, bucket count and capacity are set when the database
    // opens (`every_key_reaches_the_opened_database`).
    let LSMConfig {
        compaction_threshold_percent,
        data_dir: _,
        num_buckets: _,
        skip_list_capacity: _,
    } = lsm_config;
    assert_eq!(compaction_threshold_percent, 77);
}

#[cfg(feature = "doc-store")]
#[test]
fn every_key_reaches_the_lifecycle_config() {
    let file = parse(EVERY_KEY);
    let crate::namespace_lifecycle::LifecycleConfig {
        max_restarts,
        restart_backoff,
        restart_window,
    } = file.lifecycle_config();
    assert_eq!(max_restarts, 8);
    assert_eq!(restart_backoff, Duration::from_millis(250));
    assert_eq!(restart_window, crate::defaults::lifecycle::RESTART_WINDOW, "not in the file");
    assert_eq!(file.audit_retention(), Duration::from_secs(9 * 86_400));
}

#[cfg(all(feature = "doc-store", feature = "semantic-search"))]
#[test]
fn every_key_reaches_the_semantic_search_config() {
    use crate::vector_index::vector_settings::{IndexDefaults, MaintenanceSettings, Ratio, SearchSettings};

    let file = parse(EVERY_KEY);
    let crate::semantic_search::workers::vec_index::VectorIndexConfig {
        retry_wait_secs,
        max_retries,
        concurrency,
    } = file.vector_index_config();
    assert_eq!((retry_wait_secs, max_retries, concurrency), (21, 22, 23));

    let service = file.semantic_search_config();
    assert_eq!(service.embedding_service_url, "http://embed:9000");
    assert_eq!(service.query_embedding_cache_ttl, Duration::from_secs(31));
    assert_eq!(service.partition_history_ttl, Duration::from_secs(32));
    assert_eq!(service.embedding_request_timeout, Duration::from_secs(33));
    assert_eq!(service.embedding_connect_timeout, Duration::from_secs(4));

    assert_eq!(file.semantic_search.supported_models(), vec![("gemma".to_string(), 512)]);
    assert_eq!(
        (file.semantic_search.maintenance.threads, file.semantic_search.maintenance.rebuild_threads),
        (5, 1)
    );
    assert_eq!(
        file.semantic_search.index_defaults(),
        IndexDefaults {
            search: SearchSettings {
                min_probes: 2,
                max_probes: 64,
                first_pass_top_k: 500,
                top_k: 50,
            },
            maintenance: MaintenanceSettings {
                target_posting_size: 256,
                reassign_range: 32,
                merge_ratio: Ratio(0.125),
                skip_uncertain_moves: true,
                split_samples: 500,
                split_init_trials: 4,
                split_max_iters: 50,
                split_lambda_factor: Ratio(50.0),
            },
        }
    );
}

/// The parser's defaults and the code's are the same values.
#[test]
fn an_empty_file_is_the_default_config() {
    let file = parse("");
    assert_eq!(file, ConfigFile::default());
    assert_eq!(file.db_config(), DbConfig::default());
    assert_eq!(file.db_path(), PathBuf::from(crate::defaults::storage::DB_PATH));
    #[cfg(feature = "doc-store")]
    {
        assert_eq!(file.lifecycle_config(), crate::namespace_lifecycle::LifecycleConfig::default());
        assert_eq!(file.audit_retention(), crate::defaults::lifecycle::AUDIT_RETENTION);
        assert_eq!(
            file.semantic_search.index_defaults(),
            crate::vector_index::vector_settings::IndexDefaults::default()
        );
    }
    #[cfg(all(feature = "doc-store", feature = "semantic-search"))]
    assert_eq!(
        file.vector_index_config(),
        crate::semantic_search::workers::vec_index::VectorIndexConfig::default()
    );
    #[cfg(feature = "semantic-search")]
    assert_eq!(
        format!("{:?}", file.semantic_search_config()),
        format!("{:?}", crate::semantic_search::service::SemanticSearchConfig::default())
    );
}

/// `config/sample.toml` is the reference: with its commented-out defaults
/// switched on, it is the default config, apart from the example values it
/// names on purpose.
#[test]
fn the_sample_config_is_the_default_config() {
    let sample = include_str!("../../../config/sample.toml");
    let (file, host) = ConfigFile::parse(sample, API_SERVER_TABLES).unwrap();
    assert!(host.contains_key("api") && host.contains_key("logging"));

    let uncommented: String = sample
        .lines()
        .map(|l| match l.strip_prefix("# ") {
            Some(kv) if is_commented_default(kv) => kv.to_string(),
            _ => l.to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n");
    let (switched_on, _) = ConfigFile::parse(&uncommented, API_SERVER_TABLES).unwrap();
    assert_eq!(
        file.db_config(),
        switched_on.db_config(),
        "a commented default differs from the value in force"
    );

    let mut expected = ConfigFile::default();
    // Named on purpose: the models the companion embedding service serves.
    expected.semantic_search.supported_models = file.semantic_search.supported_models.clone();
    // Written to four places in the sample.
    assert!((switched_on.semantic_search.maintenance.merge_ratio - expected.semantic_search.maintenance.merge_ratio).abs() < 1e-4);
    expected.semantic_search.maintenance.merge_ratio = switched_on.semantic_search.maintenance.merge_ratio;
    assert_eq!(switched_on, expected);
}

/// Keys whose default is "unset": the sample shows them with an example value.
const UNSET_BY_DEFAULT: [&str; 2] = ["tail_gc_min_garbage_pct", "fail_log_dir"];

/// `# key = value`, where the value is a number, a string or a boolean: a
/// commented-out default, not prose.
fn is_commented_default(kv: &str) -> bool {
    let Some((key, value)) = kv.split_once(" = ") else { return false };
    let value = value.trim_start();
    key.chars().all(|c| c.is_ascii_lowercase() || c == '_')
        && !UNSET_BY_DEFAULT.contains(&key)
        && (value.starts_with(|c: char| c.is_ascii_digit()) || value.starts_with('"') || value.starts_with("true") || value.starts_with("false"))
}

#[test]
fn unknown_keys_and_tables_are_refused() {
    for table in [
        "storage",
        "memtable",
        "sharding",
        "lsm",
        "sync",
        "thresholds",
        "scheduled_tasks",
        "wal",
        "value_log",
        "recovery",
        "lifecycle",
        "vector_index",
        "semantic_search",
        "semantic_search.maintenance",
        "semantic_search.search_defaults",
    ] {
        let err = parse_err(&format!("[{table}]\nno_such_key = 1\n"));
        assert!(err.contains("no_such_key"), "[{table}]: {err}");
    }
    assert!(parse_err("[shardng]\nnum_buckets = 4\n").contains("shardng"));
    assert!(parse_err("[storage]\nschema_dir = \"/tmp/x\"\n").contains("schema_dir"));
    // A namespace's own settings are not server settings.
    for key in ["model = \"gemma\"", "embedding_dim = 768", "max_probes = 64"] {
        parse_err(&format!("[semantic_search]\n{key}\n"));
    }
}

#[test]
fn tables_the_caller_owns_are_handed_back_unread() {
    let text = "[api]\nlisten_addr = \"0.0.0.0:1\"\nanything = true\n[sharding]\nnum_buckets = 4\n";
    let (file, host) = ConfigFile::parse(text, &["api"]).unwrap();
    assert_eq!(file.sharding.num_buckets, 4);
    assert_eq!(host["api"]["listen_addr"].as_str(), Some("0.0.0.0:1"));
    assert!(ConfigFile::parse(text, &[]).is_err(), "not named, [api] is unknown");
}

#[test]
fn out_of_range_values_are_refused() {
    for bad in [
        "[sharding]\nnum_buckets = 0\n",
        "[lifecycle]\naudit_retention_days = 0\n",
        "[semantic_search.maintenance]\nthreads = 0\n",
        "[semantic_search.maintenance]\nthreads = 65\n",
        "[[semantic_search.supported_models]]\nname = \"\"\ndimension = 8\n",
        "[[semantic_search.supported_models]]\nname = \"m\"\ndimension = 0\n",
        "[[semantic_search.supported_models]]\nname = \"m\"\ndimension = 8\n[[semantic_search.supported_models]]\nname = \"M\"\ndimension = 8\n",
    ] {
        parse_err(bad);
    }
    #[cfg(feature = "semantic-search")]
    parse_err("[semantic_search.maintenance]\nrebuild_threads = 3\n");
    #[cfg(feature = "doc-store")]
    for bad in [
        "[semantic_search.maintenance]\ntarget_posting_size = 4\n",
        "[semantic_search.maintenance]\nmerge_ratio = 0.7\n",
        "[semantic_search.search_defaults]\nmax_probes = 0\n",
    ] {
        parse_err(bad);
    }
}

#[test]
fn a_bad_file_names_itself_in_the_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("minnal.toml");
    std::fs::write(&path, "[sharding]\nnum_buckets = 0\n").unwrap();
    let err = ConfigFile::from_file(&path, &[]).unwrap_err().to_string();
    assert!(err.contains("minnal.toml") && err.contains("num_buckets"), "{err}");
    let missing = ConfigFile::from_file(&dir.path().join("nope.toml"), &[]).unwrap_err().to_string();
    assert!(missing.contains("nope.toml"), "{missing}");
}

/// The open copies the bucket count and memtable capacity into the LSM's
/// config; everything else is used as the file gave it.
#[tokio::test]
async fn every_key_reaches_the_opened_database() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("minnal.toml");
    let db_path = dir.path().join("db");
    let fail_dir = dir.path().join("fail");
    let text = format!(
        "[api]\nlisten_addr = \"127.0.0.1:1\"\n[storage]\ndb_path = \"{}\"\n[sharding]\nnum_buckets = 3\n[memtable]\nmax_capacity = 777\n\
         [sync]\nrecords_per_sync = 5\n[recovery]\nfail_log_dir = \"{}\"\n[wal]\nsegment_size_bytes = 131072\n",
        db_path.display(),
        fail_dir.display()
    );
    std::fs::write(&path, text).unwrap();

    let db = crate::AsyncDb::open_with_config_file(path.clone()).await.unwrap();
    let opened = db.coordinator_for_test().config.clone();
    db.shutdown().await.unwrap();

    let mut expected = ConfigFile::from_file(&path, API_SERVER_TABLES).unwrap().0.db_config();
    expected.lsm_config.num_buckets = 3;
    expected.lsm_config.skip_list_capacity = 777;
    let mut opened_lsm_dirless = opened.clone();
    opened_lsm_dirless.lsm_config.data_dir = expected.lsm_config.data_dir.clone();
    assert_eq!(opened_lsm_dirless, expected);
    assert_eq!(opened.fail_log_dir, Some(fail_dir));

    // A database keeps the bucket count it was created with.
    let text = std::fs::read_to_string(&path).unwrap().replace("num_buckets = 3", "num_buckets = 5");
    std::fs::write(&path, text).unwrap();
    let db = crate::AsyncDb::open_with_config_file(path).await.unwrap();
    assert_eq!(db.coordinator_for_test().config.num_buckets, 3);
    db.shutdown().await.unwrap();
}
