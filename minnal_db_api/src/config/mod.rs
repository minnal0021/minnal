//! Configuration for the minnal_db_api server.
//!
//! [`DocStoreApiConfig`] is a superset of `minnal.toml`: it shares the same
//! TOML sections as `MinnalTomlConfig` and adds server extras such as
//! `[api].listen_addr` (the HTTP bind address) and `storage.log_dir`.
//! Store schemas live inside the database (the `system_schema` namespace), so
//! there is no schema directory to configure.
//!
//! A plain `minnal.toml` (without `[api]`) is valid here; the missing fields
//! fall back to their built-in defaults.
//!
//! # Example `minnal_db_api.toml`
//!
//! ```toml
//! [storage]
//! db_path    = "/var/lib/minnal/db"
//!
//! [api]
//! listen_addr = "0.0.0.0:8080"
//!
//! [sync]
//! records_per_sync = 500
//!
//! [scheduled_tasks]
//! value_log_gc_interval_secs    = 30
//! wal_gc_interval_secs          = 30
//! lsm_compaction_interval_secs  = 30
//!
//! [semantic_search]
//! embedding_service_url = "http://192.168.1.155:8001"
//! ```
//!
//! A namespace's model, dimension, chunking, code widths and search defaults
//! are not server settings: they live in its schema (`vector_index`).

use std::path::{Path, PathBuf};
use std::time::Duration;

use minnal_db::VectorIndexConfig;
use minnal_db::{DbConfig, ScheduledTaskConfig, SyncConfig, ThresholdConfig, lsm::LSMConfig};
use serde::Deserialize;

// ── Supported embedding models ────────────────────────────────────────────────

/// One entry in the optional `[[semantic_search.supported_models]]` TOML array.
///
/// When the array is given, a namespace may use only the models it lists, each
/// at its dimension. When it is empty (the default), any model the embedding
/// service serves is allowed: the server probes the service when a namespace
/// enables semantic search and rejects a model it does not serve, or serves at
/// another dimension (422).
///
/// The name is also the model key the embedding service is asked for: a
/// namespace naming this model sends it, lower-cased, as the `{model}` path
/// segment of every request (`{url}/embedding/{model}/document` and
/// `.../query`), so a listed name must be one the service serves (the
/// companion service serves `gemma` and `qwen`).
#[derive(Debug, Clone, Deserialize)]
pub struct SupportedModelEntry {
    /// Model identifier (matched case-insensitively).
    pub name: String,
    /// Embedding dimensionality produced by this model. Must be non-zero.
    pub dimension: u16,
}

impl SupportedModelEntry {
    /// Check the name is non-empty and the dimension non-zero.
    fn validate(&self) -> Result<(), String> {
        if self.name.trim().is_empty() {
            return Err("supported_models: an entry has an empty name".to_string());
        }
        if self.dimension == 0 {
            return Err(format!("supported_models: '{}' declares a zero dimension", self.name));
        }
        Ok(())
    }
}

// ── Top-level config ──────────────────────────────────────────────────────────

/// Superset configuration for the minnal_db_api HTTP server.
#[derive(Debug, Default, Deserialize)]
pub struct DocStoreApiConfig {
    pub storage: StorageSection,
    #[serde(default)]
    pub api: ApiSection,
    #[serde(default)]
    pub logging: LoggingSection,
    #[serde(default)]
    pub memtable: MemtableSection,
    #[serde(default)]
    pub sharding: ShardingSection,
    #[serde(default)]
    pub lsm: LsmSection,
    #[serde(default)]
    pub sync: SyncSection,
    #[serde(default)]
    pub thresholds: ThresholdSection,
    #[serde(default)]
    pub scheduled_tasks: ScheduledTaskSection,
    #[serde(default)]
    pub wal: WalSection,
    #[serde(default)]
    pub value_log: ValueLogSection,
    #[serde(default)]
    pub semantic_search: SemanticSearchSection,
    #[serde(default)]
    pub vector_index: VectorIndexSection,
    #[serde(default)]
    pub lifecycle: LifecycleSection,
}

impl DocStoreApiConfig {
    /// Load and validate a `DocStoreApiConfig` from a TOML file.
    ///
    /// Returns an error if the file cannot be read, fails to parse, or holds an
    /// invalid setting.
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path).map_err(|e| format!("cannot read '{}': {e}", path.display()))?;
        let config: Self = toml::from_str(&content).map_err(|e| format!("cannot parse '{}': {e}", path.display()))?;
        config.validate_supported_models()?;
        config
            .semantic_search
            .index_defaults()
            .map_err(|e| format!("invalid [semantic_search] defaults in '{}': {e}", path.display()))?;
        Ok(config)
    }

    /// Check every `semantic_search.supported_models` entry: a non-empty name
    /// and a non-zero dimension, each name listed once.
    fn validate_supported_models(&self) -> Result<(), String> {
        let mut seen = std::collections::HashSet::new();
        for entry in &self.semantic_search.supported_models {
            entry.validate()?;
            if !seen.insert(entry.name.to_lowercase()) {
                return Err(format!("supported_models: '{}' is listed twice", entry.name));
            }
        }
        Ok(())
    }

    /// Path to the minnal_db data directory.
    pub fn db_path(&self) -> PathBuf {
        PathBuf::from(&self.storage.db_path)
    }

    /// Path to the directory where rolling log files are written.
    pub fn log_dir(&self) -> PathBuf {
        PathBuf::from(&self.storage.log_dir)
    }

    /// HTTP address to bind the API server on.
    pub fn listen_addr(&self) -> &str {
        &self.api.listen_addr
    }

    /// Fallback log level used when `RUST_LOG` is not set.
    pub fn log_level(&self) -> &str {
        &self.logging.level
    }

    /// Convert the `[vector_index]` section to a [`VectorIndexConfig`].
    pub fn to_vector_index_config(&self) -> VectorIndexConfig {
        VectorIndexConfig {
            retry_wait_secs: self.vector_index.retry_wait_secs,
            max_retries: self.vector_index.max_retries,
            concurrency: self.vector_index.concurrency,
        }
    }

    /// Convert to the [`DbConfig`] consumed by `DocStoreBuilder::db_config`.
    pub fn to_db_config(&self) -> DbConfig {
        let scheduled = ScheduledTaskConfig::new(
            Duration::from_secs(self.scheduled_tasks.value_log_gc_interval_secs),
            Duration::from_secs(self.scheduled_tasks.wal_gc_interval_secs),
            Duration::from_secs(self.scheduled_tasks.lsm_compaction_interval_secs),
        )
        .with_ttl_cleanup_interval(Duration::from_secs(self.scheduled_tasks.ttl_cleanup_interval_secs))
        .with_index_checkpoint_interval(Duration::from_millis(self.scheduled_tasks.index_checkpoint_interval_ms));

        DbConfig {
            threshold_config: ThresholdConfig {
                value_log_waste_threshold: self.thresholds.value_log_waste_threshold,
                segment_gc_threshold: self.thresholds.segment_gc_threshold,
                tail_gc_min_garbage_pct: self.thresholds.tail_gc_min_garbage_pct,
                index_blob_waste_threshold: self.thresholds.index_blob_waste_threshold,
                index_blob_backpressure_bytes: self.thresholds.index_blob_backpressure_bytes,
                index_overlay_soft_bytes: self.thresholds.index_overlay_soft_bytes,
                index_overlay_hard_bytes: self.thresholds.index_overlay_hard_bytes,
                max_pinned_wal_segments: self.thresholds.max_pinned_wal_segments,
            },
            sync_config: SyncConfig::new(self.sync.records_per_sync).with_commit_delay_us(self.sync.commit_delay_us),
            scheduled_task_config: scheduled,
            // data_dir is overridden inside AsyncDb::open_with_config at open-time.
            lsm_config: LSMConfig::new(self.lsm.compaction_threshold_percent, PathBuf::from("lsm_data")),
            num_buckets: self.sharding.num_buckets,
            skip_list_capacity: self.memtable.max_capacity,
            wal_segment_size: self.wal.segment_size_bytes,
            segment_size_bytes: self.value_log.segment_size_bytes,
            fail_log_dir: None,
            verify_checksums_on_read: self.value_log.verify_checksums_on_read,
        }
    }
}

// ── Sections ──────────────────────────────────────────────────────────────────

/// Storage paths. Both fields have sensible defaults so the config file may
/// omit this entire section. Unknown keys are rejected: a leftover
/// `schema_dir` (store schemas live in the database) fails at startup instead of
/// being silently ignored.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StorageSection {
    #[serde(default = "default_db_path")]
    pub db_path: String,
    #[serde(default = "default_log_dir")]
    pub log_dir: String,
}

impl Default for StorageSection {
    fn default() -> Self {
        Self {
            db_path: default_db_path(),
            log_dir: default_log_dir(),
        }
    }
}

fn default_db_path() -> String {
    "./data/db".into()
}
fn default_log_dir() -> String {
    "./data/log".into()
}

/// HTTP server settings.
#[derive(Debug, Deserialize)]
pub struct ApiSection {
    #[serde(default = "default_listen_addr")]
    pub listen_addr: String,
}

impl Default for ApiSection {
    fn default() -> Self {
        Self {
            listen_addr: default_listen_addr(),
        }
    }
}

fn default_listen_addr() -> String {
    "0.0.0.0:8080".into()
}

/// Logging settings.
#[derive(Debug, Deserialize)]
pub struct LoggingSection {
    /// Minimum log level when `RUST_LOG` is not set.
    ///
    /// Accepted values: `"error"`, `"warn"`, `"info"`, `"debug"`, `"trace"`.
    /// `RUST_LOG` always takes precedence over this setting.
    #[serde(default = "default_log_level")]
    pub level: String,
}

impl Default for LoggingSection {
    fn default() -> Self {
        Self { level: default_log_level() }
    }
}

fn default_log_level() -> String {
    "info".into()
}

/// How the store lifecycle governor restarts a store's create or drop that
/// fails (`[lifecycle]`).
#[derive(Debug, Deserialize)]
pub struct LifecycleSection {
    /// Runs again of a failed create or drop before the store is parked (left
    /// as it is, with a CRITICAL log, until the server restarts).
    #[serde(default = "default_lifecycle_max_restarts")]
    pub max_restarts: u32,
    /// Wait before each run again, in milliseconds.
    #[serde(default = "default_lifecycle_restart_backoff_ms")]
    pub restart_backoff_ms: u64,
    /// Days a lifecycle audit record is kept (`GET /admin/lifecycle/audit`).
    #[serde(default = "default_audit_retention_days")]
    pub audit_retention_days: u64,
}

impl Default for LifecycleSection {
    fn default() -> Self {
        Self {
            max_restarts: default_lifecycle_max_restarts(),
            restart_backoff_ms: default_lifecycle_restart_backoff_ms(),
            audit_retention_days: default_audit_retention_days(),
        }
    }
}

impl LifecycleSection {
    /// How long a lifecycle audit record is kept.
    pub fn audit_retention(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.audit_retention_days.max(1) * 86_400)
    }

    /// The governor's configuration.
    pub fn config(&self) -> minnal_db::LifecycleConfig {
        minnal_db::LifecycleConfig {
            max_restarts: self.max_restarts,
            restart_backoff: std::time::Duration::from_millis(self.restart_backoff_ms),
        }
    }
}

fn default_lifecycle_max_restarts() -> u32 {
    minnal_db::LifecycleConfig::default().max_restarts
}

fn default_audit_retention_days() -> u64 {
    minnal_db::namespace_lifecycle::DEFAULT_AUDIT_RETENTION.as_secs() / 86_400
}

fn default_lifecycle_restart_backoff_ms() -> u64 {
    minnal_db::LifecycleConfig::default().restart_backoff.as_millis() as u64
}

// ── DB engine sections (mirrors MinnalTomlConfig) ─────────────────────────────

#[derive(Debug, Deserialize)]
pub struct MemtableSection {
    #[serde(default = "default_max_capacity")]
    pub max_capacity: usize,
}

impl Default for MemtableSection {
    fn default() -> Self {
        Self {
            max_capacity: default_max_capacity(),
        }
    }
}

fn default_max_capacity() -> usize {
    100_000
}

#[derive(Debug, Deserialize)]
pub struct ShardingSection {
    #[serde(default = "default_num_buckets")]
    pub num_buckets: usize,
}

impl Default for ShardingSection {
    fn default() -> Self {
        Self {
            num_buckets: default_num_buckets(),
        }
    }
}

fn default_num_buckets() -> usize {
    8
}

#[derive(Debug, Deserialize)]
pub struct LsmSection {
    #[serde(default = "default_compaction_threshold_percent")]
    pub compaction_threshold_percent: usize,
}

impl Default for LsmSection {
    fn default() -> Self {
        Self {
            compaction_threshold_percent: default_compaction_threshold_percent(),
        }
    }
}

fn default_compaction_threshold_percent() -> usize {
    95
}

#[derive(Debug, Deserialize)]
pub struct SyncSection {
    #[serde(default = "default_records_per_sync")]
    pub records_per_sync: usize,
    /// The main WAL's commit delay in microseconds; default 0 (see
    /// `minnal_db::SyncConfig`).
    #[serde(default)]
    pub commit_delay_us: u64,
}

impl Default for SyncSection {
    fn default() -> Self {
        Self {
            records_per_sync: default_records_per_sync(),
            commit_delay_us: 0,
        }
    }
}

fn default_records_per_sync() -> usize {
    1_000
}

#[derive(Debug, Deserialize)]
pub struct ThresholdSection {
    #[serde(default = "default_waste_threshold")]
    pub value_log_waste_threshold: f64,
    #[serde(default = "default_segment_gc_threshold")]
    pub segment_gc_threshold: f64,
    /// Garbage share at which a bucket's active tail is sealed for GC. Omit to track
    /// `value_log_waste_threshold` (the recommended default).
    #[serde(default)]
    pub tail_gc_min_garbage_pct: Option<f64>,
    #[serde(default = "default_index_blob_waste_threshold")]
    pub index_blob_waste_threshold: f64,
    #[serde(default = "default_index_blob_backpressure_bytes")]
    pub index_blob_backpressure_bytes: u64,
    /// Soft limit on memory held by all field indexes' write buffers (changes
    /// not yet written to their files): request an early index checkpoint.
    #[serde(default = "default_index_overlay_soft_bytes")]
    pub index_overlay_soft_bytes: u64,
    /// Hard limit on the same memory: the writer that crosses it writes its
    /// field's buffer out before returning.
    #[serde(default = "default_index_overlay_hard_bytes")]
    pub index_overlay_hard_bytes: u64,
    /// Cap on WAL segments the index-replay watermark may hold back from WAL GC
    /// before the backstop reclaims the oldest anyway. `0` disables the backstop.
    #[serde(default = "default_max_pinned_wal_segments")]
    pub max_pinned_wal_segments: u32,
}

impl Default for ThresholdSection {
    fn default() -> Self {
        Self {
            value_log_waste_threshold: default_waste_threshold(),
            segment_gc_threshold: default_segment_gc_threshold(),
            tail_gc_min_garbage_pct: None,
            index_blob_waste_threshold: default_index_blob_waste_threshold(),
            index_blob_backpressure_bytes: default_index_blob_backpressure_bytes(),
            index_overlay_soft_bytes: default_index_overlay_soft_bytes(),
            index_overlay_hard_bytes: default_index_overlay_hard_bytes(),
            max_pinned_wal_segments: default_max_pinned_wal_segments(),
        }
    }
}

fn default_waste_threshold() -> f64 {
    30.0
}

fn default_segment_gc_threshold() -> f64 {
    minnal_db::DEFAULT_SEGMENT_GC_THRESHOLD
}

fn default_index_blob_waste_threshold() -> f64 {
    minnal_db::DEFAULT_INDEX_BLOB_WASTE_THRESHOLD
}

fn default_index_blob_backpressure_bytes() -> u64 {
    minnal_db::DEFAULT_INDEX_BLOB_BACKPRESSURE_BYTES
}

fn default_index_overlay_soft_bytes() -> u64 {
    minnal_db::DEFAULT_INDEX_OVERLAY_SOFT_BYTES
}

fn default_index_overlay_hard_bytes() -> u64 {
    minnal_db::DEFAULT_INDEX_OVERLAY_HARD_BYTES
}

fn default_max_pinned_wal_segments() -> u32 {
    minnal_db::DEFAULT_MAX_PINNED_WAL_SEGMENTS
}

#[derive(Debug, Deserialize)]
pub struct ScheduledTaskSection {
    #[serde(default = "default_gc_interval_secs")]
    pub value_log_gc_interval_secs: u64,
    #[serde(default = "default_gc_interval_secs")]
    pub wal_gc_interval_secs: u64,
    #[serde(default = "default_gc_interval_secs")]
    pub lsm_compaction_interval_secs: u64,
    #[serde(default = "default_ttl_cleanup_secs")]
    pub ttl_cleanup_interval_secs: u64,
    /// Index checkpoint interval in **milliseconds** — this is the crash-replay
    /// window, so the useful range is sub-second to a few seconds.
    #[serde(default = "default_index_checkpoint_interval_ms")]
    pub index_checkpoint_interval_ms: u64,
}

impl Default for ScheduledTaskSection {
    fn default() -> Self {
        Self {
            value_log_gc_interval_secs: default_gc_interval_secs(),
            wal_gc_interval_secs: default_gc_interval_secs(),
            lsm_compaction_interval_secs: default_gc_interval_secs(),
            ttl_cleanup_interval_secs: default_ttl_cleanup_secs(),
            index_checkpoint_interval_ms: default_index_checkpoint_interval_ms(),
        }
    }
}

fn default_gc_interval_secs() -> u64 {
    60
}
fn default_index_checkpoint_interval_ms() -> u64 {
    minnal_db::DEFAULT_INDEX_CHECKPOINT_INTERVAL_MS
}

fn default_ttl_cleanup_secs() -> u64 {
    3_600
}

#[derive(Debug, Deserialize)]
pub struct WalSection {
    #[serde(default = "default_wal_segment_size")]
    pub segment_size_bytes: u64,
}

impl Default for WalSection {
    fn default() -> Self {
        Self {
            segment_size_bytes: default_wal_segment_size(),
        }
    }
}

fn default_wal_segment_size() -> u64 {
    64 * 1024 * 1024
}

#[derive(Debug, Deserialize)]
pub struct ValueLogSection {
    #[serde(default = "default_segment_size")]
    pub segment_size_bytes: u64,
    /// Re-verify each value's CRC32 on every read. Defaults to `false`
    /// (latency first); see `DbConfig::verify_checksums_on_read`.
    #[serde(default)]
    pub verify_checksums_on_read: bool,
}

impl Default for ValueLogSection {
    fn default() -> Self {
        Self {
            segment_size_bytes: default_segment_size(),
            verify_checksums_on_read: false,
        }
    }
}

fn default_segment_size() -> u64 {
    minnal_db::DEFAULT_SEGMENT_SIZE_BYTES
}

mod semantic_search;
#[cfg(test)]
mod tests;

pub use semantic_search::*;
