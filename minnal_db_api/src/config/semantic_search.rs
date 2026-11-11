//! The `[vector_index]` and `[semantic_search]` sections, and the
//! semantic-search config they resolve to.

use super::*;

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
    /// at once.  Default: 6.
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
    6
}

/// Semantic-search settings that belong to the server: how to reach the
/// embedding service, and optionally which models namespaces may use.
///
/// Everything that shapes one namespace's index lives in that namespace's
/// schema (`vector_index`). The `maintenance` and `search_defaults` tables here
/// only give the values a namespace's schema is filled with when it first
/// enables semantic search; from then on the namespace owns them. Unknown keys
/// are rejected, so a misplaced key (`model`, `embedding_dim`,
/// `max_probes` at this level, ...) fails at startup instead of being
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

    /// Embedding models namespaces may use, each with the dimension it
    /// produces. When empty (the default), any model the embedding service
    /// serves.
    #[serde(default)]
    pub supported_models: Vec<SupportedModelEntry>,

    /// Time-to-live, in seconds, for cached query embeddings in the system-wide
    /// `system_qemb_cache` namespace. After this duration a cached entry is
    /// evicted by the TTL worker. Default: 86400 (1 day).
    #[serde(default = "default_query_embedding_cache_ttl_secs")]
    pub query_embedding_cache_ttl_secs: u64,

    /// Time-to-live, in seconds, for the partition history in the system-wide
    /// `system_partition_history` namespace: each namespace's recorded
    /// code-health reads (`partition?codes=true`), recall checks and rebuilds.
    /// Default: 2592000 (30 days).
    #[serde(default = "default_partition_history_ttl_secs")]
    pub partition_history_ttl_secs: u64,

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
    pub search_defaults: minnal_db::doc_store::schema::vector_settings::SearchSpec,
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
    /// Engine-only: threads a partition rebuild's k-means uses, 1 or 2.
    /// Default 2.
    pub rebuild_threads: Option<usize>,
}

/// Default maintenance threads per process.
pub const DEFAULT_MAINTENANCE_THREADS: usize = 2;

impl MaintenanceSection {
    fn spec(&self) -> minnal_db::doc_store::schema::vector_settings::MaintenanceSpec {
        use minnal_db::doc_store::schema::vector_settings::{MaintenanceSpec, Ratio};
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

    /// Rebuild threads, defaulted and checked (1 to
    /// [`MAX_REBUILD_THREADS`](minnal_db::semantic_search::ivf_split::rebuild::MAX_REBUILD_THREADS)).
    pub fn rebuild_threads(&self) -> Result<usize, String> {
        let max = minnal_db::semantic_search::ivf_split::rebuild::MAX_REBUILD_THREADS;
        let t = self.rebuild_threads.unwrap_or(max);
        if !(1..=max).contains(&t) {
            return Err(format!("maintenance.rebuild_threads must be between 1 and {max}, got {t}"));
        }
        Ok(t)
    }
}

impl Default for SemanticSearchSection {
    fn default() -> Self {
        Self {
            embedding_service_url: default_embedding_service_url(),
            supported_models: Vec::new(),
            query_embedding_cache_ttl_secs: default_query_embedding_cache_ttl_secs(),
            partition_history_ttl_secs: default_partition_history_ttl_secs(),
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
fn default_partition_history_ttl_secs() -> u64 {
    30 * 86_400
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

    /// Allowed models; empty means any the embedding service serves.
    pub supported_models: Vec<SupportedModelEntry>,

    /// Time-to-live for cached query embeddings in the system-wide cache.
    pub query_embedding_cache_ttl: std::time::Duration,

    /// Time-to-live for the partition history.
    pub partition_history_ttl: std::time::Duration,

    /// Overall timeout for a single embedding-service HTTP request.
    pub embedding_request_timeout: std::time::Duration,

    /// Timeout for just the TCP connect phase to the embedding service.
    pub embedding_connect_timeout: std::time::Duration,
}

impl SemanticSearchSection {
    /// The defaults a namespace's vector-index settings are filled with, built
    /// from `search_defaults` and `maintenance` over the built-in values and
    /// validated (also checks `maintenance.threads`).
    pub fn index_defaults(&self) -> Result<minnal_db::doc_store::schema::vector_settings::IndexDefaults, String> {
        use minnal_db::doc_store::schema::vector_settings::IndexDefaults;
        let builtin = IndexDefaults::default();
        self.maintenance.threads()?;
        self.maintenance.rebuild_threads()?;
        Ok(IndexDefaults {
            search: self.search_defaults.apply(builtin.search).map_err(|e| e.to_string())?,
            maintenance: self.maintenance.spec().apply(builtin.maintenance).map_err(|e| e.to_string())?,
        })
    }

    /// Resolve this section into a [`ResolvedSemanticSearchConfig`].
    pub fn resolve(&self) -> ResolvedSemanticSearchConfig {
        ResolvedSemanticSearchConfig {
            embedding_service_url: self.embedding_service_url.clone(),
            supported_models: self.supported_models.clone(),
            query_embedding_cache_ttl: std::time::Duration::from_secs(self.query_embedding_cache_ttl_secs),
            partition_history_ttl: std::time::Duration::from_secs(self.partition_history_ttl_secs),
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
            partition_history_ttl: self.partition_history_ttl,
            embedding_request_timeout: self.embedding_request_timeout,
            embedding_connect_timeout: self.embedding_connect_timeout,
            ..Default::default()
        }
    }

    /// The allowed models as (lower-cased name, dimension); empty: any.
    pub fn supported_models(&self) -> Vec<(String, u32)> {
        self.supported_models
            .iter()
            .map(|m| (m.name.to_lowercase(), u32::from(m.dimension)))
            .collect()
    }
}
