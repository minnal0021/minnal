//! Configuration for the minnal_doc_store_api server.
//!
//! [`DocStoreApiConfig`] is a superset of `minnal.toml`: it shares the same
//! TOML sections as `MinnalTomlConfig` and adds two extras:
//! - `storage.schema_dir` — where per-store schema files are persisted
//! - `[api].listen_addr`  — the HTTP bind address
//!
//! A plain `minnal.toml` (without `schema_dir` or `[api]`) is valid here;
//! the missing fields fall back to their built-in defaults.
//!
//! # Example `minnal_doc_store_api.toml`
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
//! number_of_bits_for_dense_quantisation = 8
//! # cluster_path = "service/embedding_support/qwen/clusters.json"
//! embedding_dim = 768
//! n_probes = 32
//! embedding_service_url = "http://192.168.1.155:8001"
//! model = "qwen"
//! ```

use std::path::{Path, PathBuf};
use std::time::Duration;

use minnal_db::VectorIndexConfig;
use minnal_db::{DbConfig, ScheduledTaskConfig, SyncConfig, ThresholdConfig, lsm::LSMConfig};
use serde::Deserialize;

// ── Supported embedding models ────────────────────────────────────────────────

/// Directory under which each model's cluster centroid file lives, one
/// sub-directory per model: `service/embedding_support/{name}/clusters.json`.
const EMBEDDING_SUPPORT_DIR: &str = "service/embedding_support";

/// One entry in the `[[semantic_search.supported_models]]` TOML array.
///
/// Each entry declares that a particular embedding model is available to this
/// instance and records the dimensionality it produces. The model set is fully
/// **data-driven**: any `name` is accepted as long as its cluster centroid file
/// is present at `service/embedding_support/{name}/clusters.json` and the
/// centroids match the declared `dimension` — there is no hard-coded list of
/// recognised models.
///
/// The name is purely internal: minnal never sends it to the embedding service
/// (requests always go to `{url}/embedding/document` and `.../query` with no
/// model segment). It only selects which cluster file and embedding dimension
/// this instance uses.
#[derive(Debug, Clone, Deserialize)]
pub struct SupportedModelEntry {
    /// Model identifier. The corresponding cluster file is expected at
    /// `service/embedding_support/{name}/clusters.json` (the name is lower-cased
    /// to form the directory, so the lookup is case-insensitive).
    pub name: String,
    /// Embedding dimensionality produced by this model. Must be non-zero and
    /// must equal the dimension of the centroids in the cluster file (validated
    /// at startup).
    pub dimension: u16,
}

impl SupportedModelEntry {
    /// Lower-cased identifier used as the sub-directory name under
    /// `service/embedding_support/`.
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

/// Superset configuration for the minnal_doc_store_api HTTP server.
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
        config.validate_semantic_search()?;
        Ok(config)
    }

    /// Validate the `[semantic_search]` knobs that have hard implementation limits.
    ///
    /// `number_of_bits_for_dense_quantisation` feeds RaBitQ multi-bit quantisation,
    /// which only supports a bounded range: each quantised code is packed into a
    /// `u8` and an out-of-range width panics (or silently truncates) deep in the
    /// indexing path. Reject it at startup instead.
    fn validate_semantic_search(&self) -> Result<(), String> {
        use minnal_db::semantic_search::quantisation::rabitq::{MAX_MULTI_BIT_QUANTISATION_BITS, MIN_MULTI_BIT_QUANTISATION_BITS};
        let bits = self.semantic_search.number_of_bits_for_dense_quantisation;
        if !(MIN_MULTI_BIT_QUANTISATION_BITS..=MAX_MULTI_BIT_QUANTISATION_BITS).contains(&bits) {
            return Err(format!(
                "semantic_search.number_of_bits_for_dense_quantisation must be in \
                 {MIN_MULTI_BIT_QUANTISATION_BITS}..={MAX_MULTI_BIT_QUANTISATION_BITS}, got {bits}"
            ));
        }
        Ok(())
    }

    /// Check that every entry in `semantic_search.supported_models`:
    /// - has a non-empty `name`,
    /// - declares a non-zero `dimension`, and
    /// - has a usable cluster file at
    ///   `service/embedding_support/{name}/clusters.json` (relative to the
    ///   current working directory) whose centroids match the declared
    ///   `dimension`.
    ///
    /// The model set is **data-driven**: any name is accepted provided its
    /// cluster file exists and is consistent. Loading the file via
    /// [`ClusterIndex::load_with_dim`](minnal_db::semantic_search::ClusterIndex::load_with_dim)
    /// validates existence, well-formedness, and centroid dimension in one step
    /// — the replacement for the old hard-coded canonical-dimension check.
    ///
    /// Finally, when the list is non-empty, the active `semantic_search.model`
    /// must name one of the declared entries (case-insensitive). An empty list
    /// disables that cross-check.
    fn validate_supported_models(&self) -> Result<(), String> {
        for entry in &self.semantic_search.supported_models {
            entry.validate(Path::new(EMBEDDING_SUPPORT_DIR))?;
        }
        self.validate_active_model_listed()
    }

    /// When `supported_models` is non-empty, the active `semantic_search.model`
    /// must name one of the declared entries (matched case-insensitively,
    /// mirroring the lower-cased cluster-file directory).
    ///
    /// An empty list disables the check — there is nothing declared to match
    /// against, which keeps configs that omit the list (including the built-in
    /// default) valid.
    fn validate_active_model_listed(&self) -> Result<(), String> {
        let models = &self.semantic_search.supported_models;
        if models.is_empty() {
            return Ok(());
        }
        let active = self.semantic_search.model.to_lowercase();
        if !models.iter().any(|m| m.name.to_lowercase() == active) {
            return Err(format!(
                "semantic_search.model '{}' is not listed in supported_models (declared: {})",
                self.semantic_search.model,
                models.iter().map(|m| m.name.as_str()).collect::<Vec<_>>().join(", "),
            ));
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
        .with_ttl_cleanup_interval(Duration::from_secs(self.scheduled_tasks.ttl_cleanup_interval_secs));

        DbConfig {
            threshold_config: ThresholdConfig {
                value_log_waste_threshold: self.thresholds.value_log_waste_threshold,
                index_blob_waste_threshold: self.thresholds.index_blob_waste_threshold,
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
            page_size_bytes: self.value_log.page_size_bytes,
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
    #[serde(default = "default_index_blob_waste_threshold")]
    pub index_blob_waste_threshold: f64,
}

impl Default for ThresholdSection {
    fn default() -> Self {
        Self {
            value_log_waste_threshold: default_waste_threshold(),
            index_blob_waste_threshold: default_index_blob_waste_threshold(),
        }
    }
}

fn default_waste_threshold() -> f64 {
    30.0
}

fn default_index_blob_waste_threshold() -> f64 {
    minnal_db::DEFAULT_INDEX_BLOB_WASTE_THRESHOLD
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
}

impl Default for ScheduledTaskSection {
    fn default() -> Self {
        Self {
            value_log_gc_interval_secs: default_gc_interval_secs(),
            wal_gc_interval_secs: default_gc_interval_secs(),
            lsm_compaction_interval_secs: default_gc_interval_secs(),
            ttl_cleanup_interval_secs: default_ttl_cleanup_secs(),
        }
    }
}

fn default_gc_interval_secs() -> u64 {
    60
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
    #[serde(default = "default_page_size")]
    pub page_size_bytes: u64,
    /// Re-verify each value's CRC32 on every read. Defaults to `false`
    /// (latency first); see `DbConfig::verify_checksums_on_read`.
    #[serde(default)]
    pub verify_checksums_on_read: bool,
}

impl Default for ValueLogSection {
    fn default() -> Self {
        Self {
            page_size_bytes: default_page_size(),
            verify_checksums_on_read: false,
        }
    }
}

fn default_page_size() -> u64 {
    64 * 1024 * 1024
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

/// Semantic search / vector quantisation settings.
#[derive(Debug, Deserialize)]
pub struct SemanticSearchSection {
    /// Number of bits used to quantise each embedding dimension.
    ///
    /// Higher values give better recall at the cost of more memory and CPU.
    /// Must be in `2..=8` (validated at startup); each quantised code is packed
    /// into a `u8`, so wider widths cannot be represented.  Default: 8.
    #[serde(default = "default_number_of_bits_for_dense_quantisation")]
    pub number_of_bits_for_dense_quantisation: usize,

    /// Path to the cluster centroids file used by the IVF index.
    ///
    /// Accepts any absolute or relative path.  When `None` (the default), the
    /// path is resolved at runtime as `{db_path}/semantic_search/clusters.json`.
    #[serde(default)]
    pub cluster_path: Option<PathBuf>,

    /// Dimensionality of the embedding vectors.  Default: 768.
    #[serde(default = "default_embedding_dim")]
    pub embedding_dim: usize,

    /// Number of IVF clusters to probe during the first-pass sparse (single-bit) search.
    ///
    /// Higher values increase recall at the cost of query latency.  Default: 32.
    #[serde(default = "default_n_probes")]
    pub n_probes: usize,

    /// Number of candidates retained after the first-pass sparse (single-bit) search
    /// before re-scoring with the dense multi-bit index.  Default: 1000.
    #[serde(default = "default_first_pass_sparse_search_top_k")]
    pub first_pass_sparse_search_top_k: usize,

    /// Base URL of the embedding service, e.g. `http://192.168.1.155:8001`.
    ///
    /// Requests are a batch POST to `{url}/embedding/document` and
    /// `{url}/embedding/query` with body `{"payloads": [str, ...], "dimensions": N}`,
    /// returning `{"embeddings": [[f32], ...]}`. Chunking happens in minnal.
    #[serde(default = "default_embedding_service_url")]
    pub embedding_service_url: String,

    /// Embedding model to use.  Selects which cluster file and embedding
    /// dimension this instance uses; should name one of the
    /// [`supported_models`](Self::supported_models) entries.  Default: `"qwen"`.
    #[serde(default = "default_model")]
    pub model: String,

    /// Maximum number of results returned by a semantic search query.  Default: 100.
    #[serde(default = "default_top_k_results")]
    pub top_k_results: usize,

    /// Embedding models this instance is configured to serve.
    ///
    /// Each entry declares a `name` and the `dimension` it produces.  The set is
    /// data-driven — any name is accepted.  On startup the server validates that
    /// the corresponding cluster file
    /// (`service/embedding_support/{name}/clusters.json`) exists and that its
    /// centroids match the declared `dimension`.
    #[serde(default)]
    pub supported_models: Vec<SupportedModelEntry>,

    /// Tokens/sentences per sliding-window chunk for single-bit chunked embeddings.
    /// Default: 4.
    #[serde(default = "default_window_size")]
    pub window_size: usize,

    /// How far the window advances between chunks for single-bit embeddings.
    /// Default: 2.
    #[serde(default = "default_sliding_size")]
    pub sliding_size: usize,

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
}

impl Default for SemanticSearchSection {
    fn default() -> Self {
        Self {
            number_of_bits_for_dense_quantisation: default_number_of_bits_for_dense_quantisation(),
            cluster_path: None,
            embedding_dim: default_embedding_dim(),
            n_probes: default_n_probes(),
            first_pass_sparse_search_top_k: default_first_pass_sparse_search_top_k(),
            embedding_service_url: default_embedding_service_url(),
            model: default_model(),
            top_k_results: default_top_k_results(),
            supported_models: Vec::new(),
            window_size: default_window_size(),
            sliding_size: default_sliding_size(),
            query_embedding_cache_ttl_secs: default_query_embedding_cache_ttl_secs(),
            embedding_request_timeout_secs: default_embedding_request_timeout_secs(),
            embedding_connect_timeout_secs: default_embedding_connect_timeout_secs(),
        }
    }
}

fn default_number_of_bits_for_dense_quantisation() -> usize {
    8
}
fn default_embedding_dim() -> usize {
    768
}
fn default_n_probes() -> usize {
    32
}
fn default_first_pass_sparse_search_top_k() -> usize {
    1000
}
fn default_embedding_service_url() -> String {
    "http://localhost:8001".into()
}
fn default_model() -> String {
    "qwen".into()
}
fn default_top_k_results() -> usize {
    100
}
fn default_window_size() -> usize {
    4
}
fn default_sliding_size() -> usize {
    2
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

/// Resolved semantic search configuration, with all paths made absolute.
///
/// Build this from [`SemanticSearchSection`] via
/// [`SemanticSearchSection::resolve`], which fills in the `cluster_path`
/// default relative to the doc-store data directory.
#[derive(Debug, Clone)]
pub struct ResolvedSemanticSearchConfig {
    /// Absolute path to the cluster centroids file.
    pub cluster_path: PathBuf,

    /// Dimensionality of the embedding vectors.
    pub embedding_dim: usize,

    /// Number of IVF clusters to probe per query.
    pub n_probes: usize,

    /// Number of bits used to quantise each embedding dimension.
    pub number_of_bits_for_dense_quantisation: usize,

    /// Base URL of the embedding service, e.g. `http://192.168.1.155:8001`.
    pub embedding_service_url: String,

    /// Embedding model name, e.g. `"qwen"`.
    pub model_name: String,

    /// Maximum number of results returned by a semantic search query.
    pub top_k_results: usize,

    /// Tokens/sentences per chunk for the single-bit sliding-window embedding call.
    pub window_size: usize,

    /// How far the window advances between chunks for single-bit embeddings.
    pub sliding_size: usize,

    /// Candidates retained after the first-pass sparse (single-bit) search before dense re-ranking.
    pub first_pass_sparse_search_top_k: usize,

    /// Time-to-live for cached query embeddings in the system-wide cache.
    pub query_embedding_cache_ttl: std::time::Duration,

    /// Overall timeout for a single embedding-service HTTP request.
    pub embedding_request_timeout: std::time::Duration,

    /// Timeout for just the TCP connect phase to the embedding service.
    pub embedding_connect_timeout: std::time::Duration,
}

impl SemanticSearchSection {
    /// Resolve this section into a [`ResolvedSemanticSearchConfig`], filling in
    /// the `cluster_path` default (`{db_path}/semantic_search/clusters.json`)
    /// when no explicit path was provided.
    pub fn resolve(&self, db_path: &Path) -> ResolvedSemanticSearchConfig {
        ResolvedSemanticSearchConfig {
            cluster_path: self
                .cluster_path
                .clone()
                .unwrap_or_else(|| db_path.join("semantic_search").join("clusters.json")),
            embedding_dim: self.embedding_dim,
            n_probes: self.n_probes,
            number_of_bits_for_dense_quantisation: self.number_of_bits_for_dense_quantisation,
            embedding_service_url: self.embedding_service_url.clone(),
            model_name: self.model.clone(),
            top_k_results: self.top_k_results,
            window_size: self.window_size,
            sliding_size: self.sliding_size,
            first_pass_sparse_search_top_k: self.first_pass_sparse_search_top_k,
            query_embedding_cache_ttl: std::time::Duration::from_secs(self.query_embedding_cache_ttl_secs),
            embedding_request_timeout: std::time::Duration::from_secs(self.embedding_request_timeout_secs),
            embedding_connect_timeout: std::time::Duration::from_secs(self.embedding_connect_timeout_secs),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_with_bits(bits: usize) -> DocStoreApiConfig {
        let mut cfg = DocStoreApiConfig::default();
        cfg.semantic_search.number_of_bits_for_dense_quantisation = bits;
        cfg
    }

    #[test]
    fn validate_semantic_search_accepts_supported_bit_widths() {
        for bits in 2..=8 {
            assert!(config_with_bits(bits).validate_semantic_search().is_ok(), "bits={bits} should be valid");
        }
    }

    #[test]
    fn validate_semantic_search_rejects_out_of_range_bit_widths() {
        // 0/1 are below the multi-bit range (1 is the single-bit path); 9 silently
        // truncates the u8 code; 10+ would index the START table out of bounds.
        for bits in [0usize, 1, 9, 10, 64] {
            let err = config_with_bits(bits).validate_semantic_search().unwrap_err();
            assert!(
                err.contains("number_of_bits_for_dense_quantisation"),
                "bits={bits} should be rejected, got: {err}"
            );
        }
    }

    #[test]
    fn default_config_bit_width_is_valid() {
        let cfg = DocStoreApiConfig::default();
        assert_eq!(cfg.semantic_search.number_of_bits_for_dense_quantisation, 8);
        assert!(cfg.validate_semantic_search().is_ok());
    }

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
    fn active_model_check_skipped_when_list_empty() {
        // The built-in default: model = "qwen", supported_models = []. The
        // cross-check is disabled, so this stays valid (no break).
        let cfg = DocStoreApiConfig::default();
        assert!(cfg.semantic_search.supported_models.is_empty());
        assert!(cfg.validate_active_model_listed().is_ok());
    }

    #[test]
    fn active_model_check_passes_when_listed_case_insensitively() {
        let mut cfg = DocStoreApiConfig::default();
        cfg.semantic_search.model = "Qwen".to_string();
        cfg.semantic_search.supported_models = vec![SupportedModelEntry {
            name: "qwen".to_string(),
            dimension: 768,
        }];
        assert!(cfg.validate_active_model_listed().is_ok());
    }

    #[test]
    fn active_model_check_rejects_unlisted_model() {
        let mut cfg = DocStoreApiConfig::default();
        cfg.semantic_search.model = "qwen".to_string();
        cfg.semantic_search.supported_models = vec![SupportedModelEntry {
            name: "other".to_string(),
            dimension: 768,
        }];
        let err = cfg.validate_active_model_listed().unwrap_err();
        assert!(err.contains("qwen") && err.contains("supported_models"), "got: {err}");
    }

    #[test]
    fn resolve_carries_embedding_timeouts() {
        // Defaults: request 30s, connect 10s. resolve must surface both as Durations,
        // with connect shorter than the overall request cap.
        let section = SemanticSearchSection::default();
        assert_eq!(section.embedding_request_timeout_secs, 30);
        assert_eq!(section.embedding_connect_timeout_secs, 10);
        let resolved = section.resolve(Path::new("/tmp/db"));
        assert_eq!(resolved.embedding_request_timeout, std::time::Duration::from_secs(30));
        assert_eq!(resolved.embedding_connect_timeout, std::time::Duration::from_secs(10));
        assert!(resolved.embedding_connect_timeout < resolved.embedding_request_timeout);
    }
}
