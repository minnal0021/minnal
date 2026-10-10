//! Configuration for the minnal_db_api server.
//!
//! [`DocStoreApiConfig`] is a superset of `minnal.toml`: it shares the same
//! TOML sections as `MinnalTomlConfig` and adds two extras:
//! - `storage.schema_dir` — where per-store schema files are persisted
//! - `[api].listen_addr`  — the HTTP bind address
//!
//! A plain `minnal.toml` (without `schema_dir` or `[api]`) is valid here;
//! the missing fields fall back to their built-in defaults.
//!
//! # Example `minnal_db_api.toml`
//!
//! ```toml
//! [storage]
//! db_path    = "/var/lib/minnal/db"
//! schema_dir = "/var/lib/minnal/schemas"
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
//! # centroid_dir = "service/embedding_support"   # {centroid_dir}/{model}/clusters.json
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

/// Default directory under which each model's cluster centroid file lives, one
/// sub-directory per model: `{centroid_dir}/{name}/clusters.json`.
const DEFAULT_CENTROID_DIR: &str = "service/embedding_support";

/// One entry in the `[[semantic_search.supported_models]]` TOML array.
///
/// Each entry declares that a particular embedding model is available to this
/// instance and records the dimensionality it produces. The model set is fully
/// **data-driven**: any `name` is accepted as long as its cluster centroid file
/// is present at `{centroid_dir}/{name}/clusters.json` and the centroids match
/// the declared `dimension` — there is no hard-coded list of recognised models.
///
/// The name is also the model key the embedding service is asked for: a
/// namespace naming this model sends it, lower-cased, as the `{model}` path
/// segment of every request (`{url}/embedding/{model}/document` and
/// `.../query`), so a declared name must be one the service serves (the
/// companion service serves `gemma` and `qwen`).
#[derive(Debug, Clone, Deserialize)]
pub struct SupportedModelEntry {
    /// Model identifier. The corresponding cluster file is expected at
    /// `{centroid_dir}/{name}/clusters.json` (the name is lower-cased to form the
    /// directory, so the lookup is case-insensitive).
    pub name: String,
    /// Embedding dimensionality produced by this model. Must be non-zero and
    /// must equal the dimension of the centroids in the cluster file (validated
    /// at startup).
    pub dimension: u16,
}

impl SupportedModelEntry {
    /// Lower-cased identifier used as the sub-directory name under the
    /// centroid directory.
    pub fn dir_name(&self) -> String {
        self.name.to_lowercase()
    }

    /// Path to this model's cluster centroid file, rooted at `support_dir`
    /// (`{support_dir}/{name}/clusters.json`).
    pub fn cluster_file_path(&self, support_dir: &Path) -> PathBuf {
        support_dir.join(self.dir_name()).join("clusters.json")
    }

    /// Validate this entry against cluster files rooted at `support_dir`.
    ///
    /// Checks the name is non-empty and the dimension is non-zero, then loads
    /// the cluster file — which confirms it exists, is well-formed, and that its
    /// centroids match the declared `dimension`.
    fn validate(&self, support_dir: &Path) -> Result<(), String> {
        if self.name.trim().is_empty() {
            return Err("supported_models: an entry has an empty name".to_string());
        }
        if self.dimension == 0 {
            return Err(format!("supported_models: '{}' declares a zero dimension", self.name));
        }

        let cluster_file = self.cluster_file_path(support_dir);
        let cluster_path = cluster_file
            .to_str()
            .ok_or_else(|| format!("supported_models: cluster file path for '{}' is not valid UTF-8", self.name))?;

        minnal_db::semantic_search::ClusterIndex::load_with_dim(cluster_path, self.dimension as usize).map_err(|e| {
            format!(
                "supported_models: cluster file for '{}' (expected at '{}') is unusable: {e}",
                self.name, cluster_path
            )
        })?;
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
}

impl DocStoreApiConfig {
    /// Load and validate a `DocStoreApiConfig` from a TOML file.
    ///
    /// Returns an error if the file cannot be read, fails to parse, or if any
    /// supported model is missing its cluster file on disk.
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

    /// Check that every entry in `semantic_search.supported_models`:
    /// - has a non-empty `name`,
    /// - declares a non-zero `dimension`, and
    /// - has a usable cluster file at `{centroid_dir}/{name}/clusters.json`
    ///   whose centroids match the declared `dimension`.
    ///
    /// The model set is **data-driven**: any name is accepted provided its
    /// cluster file exists and is consistent. Loading the file via
    /// [`ClusterIndex::load_with_dim`](minnal_db::semantic_search::ClusterIndex::load_with_dim)
    /// validates existence, well-formedness, and centroid dimension in one step.
    fn validate_supported_models(&self) -> Result<(), String> {
        let dir = self.semantic_search.centroid_dir();
        for entry in &self.semantic_search.supported_models {
            entry.validate(&dir)?;
        }
        Ok(())
    }

    /// Path to the minnal_db data directory.
    pub fn db_path(&self) -> PathBuf {
        PathBuf::from(&self.storage.db_path)
    }

    /// Path to the directory where schema JSON files are stored.
    pub fn schema_dir(&self) -> PathBuf {
        PathBuf::from(&self.storage.schema_dir)
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

    /// Convert to the [`DbConfig`] consumed by `DocStore::open_with_config`.
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
            sync_config: SyncConfig {
                records_per_sync: self.sync.records_per_sync,
            },
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
/// omit this entire section.
#[derive(Debug, Deserialize)]
pub struct StorageSection {
    #[serde(default = "default_db_path")]
    pub db_path: String,
    #[serde(default = "default_schema_dir")]
    pub schema_dir: String,
    #[serde(default = "default_log_dir")]
    pub log_dir: String,
}

impl Default for StorageSection {
    fn default() -> Self {
        Self {
            db_path: default_db_path(),
            schema_dir: default_schema_dir(),
            log_dir: default_log_dir(),
        }
    }
}

fn default_db_path() -> String {
    "./data/db".into()
}
fn default_schema_dir() -> String {
    "./data/schemas".into()
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
}

impl Default for SyncSection {
    fn default() -> Self {
        Self {
            records_per_sync: default_records_per_sync(),
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

// ── Vector index worker section ───────────────────────────────────────────────

/// Tuning parameters for the async vector-index background worker.
#[derive(Debug, Deserialize)]
pub struct VectorIndexSection {
    /// Seconds to wait after a pass that contained at least one failure before
    /// re-scanning the queue.  Default: 2.
    #[serde(default = "default_retry_wait_secs")]
    pub retry_wait_secs: u64,
    /// Maximum number of embedding attempts per queue entry before the entry
    /// is skipped and flagged for manual removal via the admin API.  Default: 5.
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    /// Maximum number of concurrent embedding calls the worker keeps in flight
    /// at once.  Default: 4.
    #[serde(default = "default_concurrency")]
    pub concurrency: usize,
}

impl Default for VectorIndexSection {
    fn default() -> Self {
        Self {
            retry_wait_secs: default_retry_wait_secs(),
            max_retries: default_max_retries(),
            concurrency: default_concurrency(),
        }
    }
}

fn default_retry_wait_secs() -> u64 {
    2
}
fn default_max_retries() -> u32 {
    5
}
fn default_concurrency() -> usize {
    4
}

/// Semantic-search settings that belong to the server: how to reach the
/// embedding service, and which models' centroids it loads.
///
/// Everything that shapes one namespace's index lives in that namespace's
/// schema (`vector_index`). The `maintenance` and `search_defaults` tables here
/// only give the values a namespace's schema is filled with when it first
/// enables semantic search; from then on the namespace owns them. Unknown keys
/// are rejected, so a misplaced key (`model`, `embedding_dim`,
/// `probe_budget_entries` at this level, ...) fails at startup instead of being
/// silently ignored.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticSearchSection {
    /// Base URL of the embedding service, e.g. `http://192.168.1.155:8001`.
    ///
    /// Requests are a batch POST to `{url}/embedding/{model}/document` and
    /// `{url}/embedding/{model}/query` with body
    /// `{"payloads": [str, ...], "dimensions": N}`, returning
    /// `{"embeddings": [[f32], ...]}`; `{model}` and `N` come from the
    /// namespace. Chunking happens in minnal.
    #[serde(default = "default_embedding_service_url")]
    pub embedding_service_url: String,

    /// Directory holding one cluster-centroid set per model, at
    /// `{centroid_dir}/{model}/clusters.json`. Relative paths resolve against
    /// the working directory. Default: `service/embedding_support`.
    #[serde(default)]
    pub centroid_dir: Option<PathBuf>,

    /// Embedding models this instance serves, each with the dimension it
    /// produces. Every entry's centroid file must exist and match its
    /// dimension (validated at startup). When empty, every
    /// `{centroid_dir}/{model}/clusters.json` found at startup is loaded.
    #[serde(default)]
    pub supported_models: Vec<SupportedModelEntry>,

    /// Time-to-live, in seconds, for cached query embeddings in the system-wide
    /// `system_qemb_cache` namespace. After this duration a cached entry is
    /// evicted by the TTL worker. Default: 86400 (1 day).
    #[serde(default = "default_query_embedding_cache_ttl_secs")]
    pub query_embedding_cache_ttl_secs: u64,

    /// Overall timeout, in seconds, for a single embedding-service HTTP request
    /// (connect + send + receive). Caps how long indexing/search can block on a
    /// slow or hanging service. Default: 30.
    #[serde(default = "default_embedding_request_timeout_secs")]
    pub embedding_request_timeout_secs: u64,

    /// Timeout, in seconds, for just the TCP connect phase to the embedding
    /// service. Fails fast when the host is unreachable. Should be shorter than
    /// `embedding_request_timeout_secs` (the overall cap). Default: 10.
    #[serde(default = "default_embedding_connect_timeout_secs")]
    pub embedding_connect_timeout_secs: u64,

    /// Partition maintenance: defaults copied into a namespace's schema
    /// (`vector_index.maintenance`) when it enables semantic search, plus the
    /// engine-only `threads`.
    #[serde(default)]
    pub maintenance: MaintenanceSection,

    /// Search defaults copied into a namespace's schema (`vector_index.search`)
    /// when it enables semantic search.
    #[serde(default)]
    pub search_defaults: minnal_db::doc_store::vector_settings::SearchSpec,
}

/// `[semantic_search.maintenance]`: how each namespace's partition is split,
/// reassigned and merged (design doc M3). All but `threads` are defaults for a
/// namespace's schema; unset keys keep the built-in values.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaintenanceSection {
    /// Target posting size in chunks; a posting splits above twice this.
    pub target_posting_size: Option<u32>,
    /// Nearest postings re-checked after a split (0 = none).
    pub reassign_range: Option<u32>,
    /// Merge limit as a share of the split limit (SPFresh: 10/118).
    pub merge_ratio: Option<f64>,
    /// Move a chunk only when its code shows it is clearly closer to the new
    /// posting (the gain exceeds the code's error bound).
    pub skip_uncertain_moves: Option<bool>,
    /// Balanced 2-means: members sampled per iteration.
    pub split_samples: Option<u32>,
    /// Balanced 2-means: random starting pairs tried.
    pub split_init_trials: Option<u32>,
    /// Balanced 2-means: iteration cap.
    pub split_max_iters: Option<u32>,
    /// Balanced 2-means: balance factor.
    pub split_lambda_factor: Option<f64>,
    /// Engine-only: maintenance threads per process. Default 2.
    pub threads: Option<usize>,
}

/// Default maintenance threads per process.
pub const DEFAULT_MAINTENANCE_THREADS: usize = 2;

impl MaintenanceSection {
    fn spec(&self) -> minnal_db::doc_store::vector_settings::MaintenanceSpec {
        use minnal_db::doc_store::vector_settings::{MaintenanceSpec, Ratio};
        MaintenanceSpec {
            target_posting_size: self.target_posting_size,
            reassign_range: self.reassign_range,
            merge_ratio: self.merge_ratio.map(Ratio),
            skip_uncertain_moves: self.skip_uncertain_moves,
            split_samples: self.split_samples,
            split_init_trials: self.split_init_trials,
            split_max_iters: self.split_max_iters,
            split_lambda_factor: self.split_lambda_factor.map(Ratio),
        }
    }

    /// Maintenance threads, defaulted and checked (1–64).
    pub fn threads(&self) -> Result<usize, String> {
        let t = self.threads.unwrap_or(DEFAULT_MAINTENANCE_THREADS);
        if !(1..=64).contains(&t) {
            return Err(format!("maintenance.threads must be between 1 and 64, got {t}"));
        }
        Ok(t)
    }
}

impl Default for SemanticSearchSection {
    fn default() -> Self {
        Self {
            embedding_service_url: default_embedding_service_url(),
            centroid_dir: None,
            supported_models: Vec::new(),
            query_embedding_cache_ttl_secs: default_query_embedding_cache_ttl_secs(),
            embedding_request_timeout_secs: default_embedding_request_timeout_secs(),
            embedding_connect_timeout_secs: default_embedding_connect_timeout_secs(),
            maintenance: MaintenanceSection::default(),
            search_defaults: Default::default(),
        }
    }
}

fn default_embedding_service_url() -> String {
    "http://localhost:8001".into()
}
fn default_query_embedding_cache_ttl_secs() -> u64 {
    86_400
}
fn default_embedding_request_timeout_secs() -> u64 {
    30
}
fn default_embedding_connect_timeout_secs() -> u64 {
    10
}

// ── Resolved semantic-search config ──────────────────────────────────────────

/// Resolved semantic-search server settings (see [`SemanticSearchSection`]).
#[derive(Debug, Clone)]
pub struct ResolvedSemanticSearchConfig {
    /// Base URL of the embedding service, e.g. `http://192.168.1.155:8001`.
    pub embedding_service_url: String,

    /// Directory holding `{model}/clusters.json` per model.
    pub centroid_dir: PathBuf,

    /// Declared models; empty means "every centroid set found".
    pub supported_models: Vec<SupportedModelEntry>,

    /// Time-to-live for cached query embeddings in the system-wide cache.
    pub query_embedding_cache_ttl: std::time::Duration,

    /// Overall timeout for a single embedding-service HTTP request.
    pub embedding_request_timeout: std::time::Duration,

    /// Timeout for just the TCP connect phase to the embedding service.
    pub embedding_connect_timeout: std::time::Duration,
}

impl SemanticSearchSection {
    /// The defaults a namespace's vector-index settings are filled with, built
    /// from `search_defaults` and `maintenance` over the built-in values and
    /// validated (also checks `maintenance.threads`).
    pub fn index_defaults(&self) -> Result<minnal_db::doc_store::vector_settings::IndexDefaults, String> {
        use minnal_db::doc_store::vector_settings::IndexDefaults;
        let builtin = IndexDefaults::default();
        self.maintenance.threads()?;
        Ok(IndexDefaults {
            search: self.search_defaults.apply(builtin.search).map_err(|e| e.to_string())?,
            maintenance: self.maintenance.spec().apply(builtin.maintenance).map_err(|e| e.to_string())?,
        })
    }

    /// The centroid directory, defaulted.
    pub fn centroid_dir(&self) -> PathBuf {
        self.centroid_dir.clone().unwrap_or_else(|| PathBuf::from(DEFAULT_CENTROID_DIR))
    }

    /// Resolve this section into a [`ResolvedSemanticSearchConfig`].
    pub fn resolve(&self) -> ResolvedSemanticSearchConfig {
        ResolvedSemanticSearchConfig {
            embedding_service_url: self.embedding_service_url.clone(),
            centroid_dir: self.centroid_dir(),
            supported_models: self.supported_models.clone(),
            query_embedding_cache_ttl: std::time::Duration::from_secs(self.query_embedding_cache_ttl_secs),
            embedding_request_timeout: std::time::Duration::from_secs(self.embedding_request_timeout_secs),
            embedding_connect_timeout: std::time::Duration::from_secs(self.embedding_connect_timeout_secs),
        }
    }
}

impl ResolvedSemanticSearchConfig {
    /// The service settings as the library's per-call config. Its
    /// per-namespace fields stay at their defaults: each namespace's settings
    /// overwrite them (`SemanticSearchContext::for_namespace`).
    pub fn service_config(&self) -> minnal_db::semantic_search::service::SemanticSearchConfig {
        minnal_db::semantic_search::service::SemanticSearchConfig {
            embedding_service_url: self.embedding_service_url.clone(),
            query_embedding_cache_ttl: self.query_embedding_cache_ttl,
            embedding_request_timeout: self.embedding_request_timeout,
            embedding_connect_timeout: self.embedding_connect_timeout,
            ..Default::default()
        }
    }

    /// Load one centroid set per model: the declared models (each checked
    /// against its dimension), or, when none are declared, every
    /// `{centroid_dir}/{model}/clusters.json` found. Returns the loaded sets
    /// (lower-cased model name → index) and a message for each set that could
    /// not be loaded.
    pub fn load_cluster_indexes(&self) -> (Vec<(String, std::sync::Arc<minnal_db::semantic_search::ClusterIndex>)>, Vec<String>) {
        use minnal_db::semantic_search::ClusterIndex;
        let mut loaded = Vec::new();
        let mut problems = Vec::new();
        let mut load = |name: String, dim: Option<usize>| {
            let path = self.centroid_dir.join(&name).join("clusters.json");
            let path_str = path.to_string_lossy();
            let result = match dim {
                Some(d) => ClusterIndex::load_with_dim(&path_str, d),
                None => ClusterIndex::load(&path_str),
            };
            match result {
                Ok(index) => loaded.push((name, std::sync::Arc::new(index))),
                Err(e) => problems.push(format!("model '{name}': cannot load '{}': {e}", path.display())),
            }
        };
        if self.supported_models.is_empty() {
            let mut names: Vec<String> = std::fs::read_dir(&self.centroid_dir)
                .into_iter()
                .flatten()
                .flatten()
                .filter(|e| e.path().join("clusters.json").is_file())
                .filter_map(|e| e.file_name().to_str().map(str::to_lowercase))
                .collect();
            names.sort_unstable();
            for name in names {
                load(name, None);
            }
        } else {
            for entry in &self.supported_models {
                load(entry.dir_name(), Some(entry.dimension as usize));
            }
        }
        (loaded, problems)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a unique temp directory and write a `{model}/clusters.json` with
    /// `n` centroids of dimension `dim` (JSONL, one object per line). Returns the
    /// support-dir root the entry validates against.
    fn write_cluster_file(model: &str, dim: usize, n: usize) -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let unique = format!("minnal_cfg_test_{}_{}", std::process::id(), SEQ.fetch_add(1, Ordering::Relaxed));
        let support_dir = std::env::temp_dir().join(unique);
        let model_dir = support_dir.join(model.to_lowercase());
        std::fs::create_dir_all(&model_dir).unwrap();
        let mut body = String::new();
        for id in 0..n {
            let centroid: Vec<f32> = (0..dim).map(|j| (id + j) as f32).collect();
            body.push_str(&serde_json::json!({ "cluster_id": id, "centroid": centroid }).to_string());
            body.push('\n');
        }
        std::fs::write(model_dir.join("clusters.json"), body).unwrap();
        support_dir
    }

    #[test]
    fn validate_accepts_any_model_name_with_a_matching_cluster_file() {
        // Data-driven: a name that is NOT "qwen" is accepted as long as its
        // cluster file exists and the centroid dimension matches.
        let support_dir = write_cluster_file("brandnew", 1024, 3);
        let entry = SupportedModelEntry {
            name: "BrandNew".to_string(),
            dimension: 1024,
        };
        assert!(entry.validate(&support_dir).is_ok());
        std::fs::remove_dir_all(&support_dir).ok();
    }

    #[test]
    fn validate_rejects_dimension_mismatch() {
        let support_dir = write_cluster_file("modelx", 768, 2);
        let err = SupportedModelEntry {
            name: "modelx".to_string(),
            dimension: 512,
        }
        .validate(&support_dir)
        .unwrap_err();
        assert!(err.contains("modelx"), "got: {err}");
        std::fs::remove_dir_all(&support_dir).ok();
    }

    #[test]
    fn validate_rejects_missing_cluster_file() {
        let support_dir = std::env::temp_dir().join(format!("minnal_cfg_missing_{}", std::process::id()));
        let err = SupportedModelEntry {
            name: "ghost".to_string(),
            dimension: 768,
        }
        .validate(&support_dir)
        .unwrap_err();
        assert!(err.contains("ghost"), "got: {err}");
    }

    #[test]
    fn validate_rejects_empty_name_and_zero_dimension() {
        let support_dir = std::env::temp_dir();
        assert!(
            SupportedModelEntry {
                name: "  ".to_string(),
                dimension: 768
            }
            .validate(&support_dir)
            .is_err()
        );
        assert!(
            SupportedModelEntry {
                name: "ok".to_string(),
                dimension: 0
            }
            .validate(&support_dir)
            .is_err()
        );
    }

    #[test]
    fn resolve_carries_embedding_timeouts() {
        // Defaults: request 30s, connect 10s. resolve must surface both as Durations,
        // with connect shorter than the overall request cap.
        let section = SemanticSearchSection::default();
        assert_eq!(section.embedding_request_timeout_secs, 30);
        assert_eq!(section.embedding_connect_timeout_secs, 10);
        let resolved = section.resolve();
        assert_eq!(resolved.embedding_request_timeout, std::time::Duration::from_secs(30));
        assert_eq!(resolved.embedding_connect_timeout, std::time::Duration::from_secs(10));
        assert!(resolved.embedding_connect_timeout < resolved.embedding_request_timeout);
    }

    #[test]
    fn a_namespace_setting_left_in_the_server_config_is_rejected() {
        for key in [
            "model = \"qwen\"",
            "embedding_dim = 768",
            "n_probes = 64",
            "probe_budget_entries = 70000",
            "max_probes = 1024",
            "window_size = 4",
            "cluster_path = \"x\"",
        ] {
            let toml = format!("[storage]\ndb_path = \"/tmp/x\"\nschema_dir = \"/tmp/y\"\n[semantic_search]\n{key}\n");
            let err = toml::from_str::<DocStoreApiConfig>(&toml).unwrap_err().to_string();
            assert!(err.contains("unknown field"), "{key}: {err}");
        }
        let ok = "[storage]\ndb_path = \"/tmp/x\"\nschema_dir = \"/tmp/y\"\n[semantic_search]\nembedding_service_url = \"http://h:1\"\ncentroid_dir = \"/c\"\n";
        let cfg: DocStoreApiConfig = toml::from_str(ok).unwrap();
        assert_eq!(cfg.semantic_search.centroid_dir(), PathBuf::from("/c"));
    }

    #[test]
    fn maintenance_and_search_defaults_are_read_and_validated() {
        let base = "[storage]\ndb_path = \"/tmp/x\"\nschema_dir = \"/tmp/y\"\n";
        let cfg: DocStoreApiConfig = toml::from_str(base).unwrap();
        let d = cfg.semantic_search.index_defaults().unwrap();
        assert_eq!(d, minnal_db::doc_store::vector_settings::IndexDefaults::default());
        assert_eq!(cfg.semantic_search.maintenance.threads().unwrap(), DEFAULT_MAINTENANCE_THREADS);

        let toml = format!(
            "{base}[semantic_search.maintenance]\ntarget_posting_size = 256\nskip_uncertain_moves = true\nthreads = 4\n\
             [semantic_search.search_defaults]\nprobe_budget_entries = 40000\nprobe_budget_fraction = 0.3\n"
        );
        let cfg: DocStoreApiConfig = toml::from_str(&toml).unwrap();
        let d = cfg.semantic_search.index_defaults().unwrap();
        assert_eq!((d.maintenance.target_posting_size, d.maintenance.skip_uncertain_moves), (256, true));
        assert_eq!(d.maintenance.reassign_range, 64, "unset keys keep the built-in value");
        assert_eq!(d.search.probe_budget_entries, 40_000);
        assert_eq!(d.search.probe_budget_fraction.map(|r| r.0), Some(0.3));
        assert_eq!(cfg.semantic_search.maintenance.threads().unwrap(), 4);

        for (table, key) in [
            ("maintenance", "target_posting_size = 4"),
            ("maintenance", "merge_ratio = 0.7"),
            ("maintenance", "threads = 0"),
            ("search_defaults", "probe_budget_fraction = 2.0"),
        ] {
            let toml = format!("{base}[semantic_search.{table}]\n{key}\n");
            let cfg: DocStoreApiConfig = toml::from_str(&toml).unwrap();
            assert!(cfg.semantic_search.index_defaults().is_err(), "{key}");
        }
        for (table, key) in [("maintenance", "error_bound_gate = true"), ("search_defaults", "n_probes = 64")] {
            let toml = format!("{base}[semantic_search.{table}]\n{key}\n");
            assert!(toml::from_str::<DocStoreApiConfig>(&toml).is_err(), "{key}");
        }
    }

    #[test]
    fn the_sample_config_parses_with_every_commented_default_switched_on() {
        let sample = include_str!("../../config/sample.toml");
        let cfg: DocStoreApiConfig = toml::from_str(sample).unwrap();
        cfg.semantic_search.index_defaults().unwrap();
        // Uncomment the documented defaults of the two semantic-search tables:
        // they must parse and equal the built-in values.
        let mut table = "";
        let uncommented: String = sample
            .lines()
            .map(|l| {
                if l.starts_with('[') {
                    table = l;
                }
                match l.strip_prefix("# ") {
                    Some(kv) if table.starts_with("[semantic_search.") && kv.contains(" = ") && !kv.contains("fraction") && !kv.contains("floor") => {
                        kv.to_string()
                    }
                    _ => l.to_string(),
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        let cfg: DocStoreApiConfig = toml::from_str(&uncommented).unwrap();
        let d = cfg.semantic_search.index_defaults().unwrap();
        let builtin = minnal_db::doc_store::vector_settings::IndexDefaults::default();
        assert_eq!(d.search, builtin.search);
        assert_eq!(d.maintenance.target_posting_size, builtin.maintenance.target_posting_size);
        assert!((d.maintenance.merge_ratio.0 - builtin.maintenance.merge_ratio.0).abs() < 1e-4);
    }

    #[test]
    fn centroid_dir_defaults_to_the_bundled_sets() {
        assert_eq!(
            SemanticSearchSection::default().centroid_dir(),
            PathBuf::from("service/embedding_support")
        );
    }

    #[test]
    fn without_declared_models_every_centroid_set_found_is_loaded() {
        let dir = write_cluster_file("gemma", 8, 3);
        let qwen = write_cluster_file("qwen", 16, 2);
        std::fs::rename(qwen.join("qwen"), dir.join("qwen")).unwrap();
        std::fs::create_dir_all(dir.join("empty")).unwrap(); // no clusters.json: not a model
        std::fs::create_dir_all(dir.join("broken")).unwrap();
        std::fs::write(dir.join("broken").join("clusters.json"), "not json").unwrap();
        let section = SemanticSearchSection {
            centroid_dir: Some(dir.clone()),
            ..SemanticSearchSection::default()
        };
        let (loaded, problems) = section.resolve().load_cluster_indexes();
        let got: Vec<(String, usize)> = loaded.iter().map(|(m, i)| (m.clone(), i.dim())).collect();
        assert_eq!(got, vec![("gemma".to_string(), 8), ("qwen".to_string(), 16)]);
        assert_eq!(problems.len(), 1, "{problems:?}");
        assert!(problems[0].contains("broken"), "{problems:?}");
        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_dir_all(&qwen).ok();
    }

    #[test]
    fn declared_models_are_the_only_ones_loaded_and_are_checked_against_their_dimension() {
        let dir = write_cluster_file("gemma", 8, 3);
        let other = write_cluster_file("qwen", 8, 2);
        std::fs::rename(other.join("qwen"), dir.join("qwen")).unwrap();
        let section = SemanticSearchSection {
            centroid_dir: Some(dir.clone()),
            supported_models: vec![
                SupportedModelEntry {
                    name: "Gemma".into(),
                    dimension: 8,
                },
                SupportedModelEntry {
                    name: "qwen".into(),
                    dimension: 16,
                },
            ],
            ..SemanticSearchSection::default()
        };
        let (loaded, problems) = section.resolve().load_cluster_indexes();
        assert_eq!(loaded.iter().map(|(m, _)| m.as_str()).collect::<Vec<_>>(), vec!["gemma"]);
        assert!(problems.len() == 1 && problems[0].contains("qwen"), "{problems:?}");
        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_dir_all(&other).ok();
    }

    #[test]
    fn service_config_carries_only_service_settings() {
        let section = SemanticSearchSection {
            embedding_service_url: "http://h:1".into(),
            embedding_request_timeout_secs: 7,
            ..SemanticSearchSection::default()
        };
        let c = section.resolve().service_config();
        assert_eq!(c.embedding_service_url, "http://h:1");
        assert_eq!(c.embedding_request_timeout, std::time::Duration::from_secs(7));
    }
}
