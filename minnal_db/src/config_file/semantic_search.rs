//! `[vector_index]` (the embedding worker) and `[semantic_search]` (the
//! embedding service, the system-wide caches, and the defaults a namespace's
//! vector-index settings start from).
//!
//! The tables parse in every build, so one file serves every feature set;
//! what they configure exists only with `semantic-search` (and `doc-store`
//! for the worker and the partition settings).

use serde::Deserialize;

use crate::defaults;

/// `[vector_index]`: the background worker that embeds documents.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VectorIndexSection {
    /// Seconds to wait after a pass with a failure before the queue is read
    /// again.
    pub retry_wait_secs: u64,
    /// Embedding attempts per queue entry before it is skipped and left for
    /// an operator.
    pub max_retries: u32,
    /// Embedding calls in flight at once.
    pub concurrency: usize,
}

impl Default for VectorIndexSection {
    fn default() -> Self {
        Self {
            retry_wait_secs: defaults::vector_index::RETRY_WAIT_SECS,
            max_retries: defaults::vector_index::MAX_RETRIES,
            concurrency: defaults::vector_index::CONCURRENCY,
        }
    }
}

/// One entry of `[[semantic_search.supported_models]]`.
///
/// When the list is given, a namespace may use only the models it names, each
/// at its dimension; when it is empty, any model the embedding service serves.
/// The name is also what the service is asked for: it is the `{model}` path
/// segment of every request, lower-cased.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SupportedModel {
    /// Model name, matched without regard to case.
    pub name: String,
    /// Dimension the model's embeddings have.
    pub dimension: u16,
}

/// `[semantic_search]`: the embedding service and the system-wide caches.
///
/// Everything that shapes one namespace's index is in that namespace's schema
/// (`vector_index`). `maintenance` and `search_defaults` only give the values
/// a namespace's schema starts from when it enables semantic search.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SemanticSearchSection {
    /// Base URL of the embedding service. Requests go to
    /// `{url}/embedding/{model}/document` and `{url}/embedding/{model}/query`.
    pub embedding_service_url: String,
    /// Models namespaces may use; empty: any the service serves.
    pub supported_models: Vec<SupportedModel>,
    /// How long a cached query embedding is kept, in seconds.
    pub query_embedding_cache_ttl_secs: u64,
    /// How long a partition-history record is kept, in seconds.
    pub partition_history_ttl_secs: u64,
    /// The whole of one embedding request, in seconds.
    pub embedding_request_timeout_secs: u64,
    /// The TCP connect alone, in seconds.
    pub embedding_connect_timeout_secs: u64,
    /// `[semantic_search.maintenance]`.
    pub maintenance: MaintenanceSection,
    /// `[semantic_search.search_defaults]`.
    pub search_defaults: SearchDefaultsSection,
}

impl Default for SemanticSearchSection {
    fn default() -> Self {
        use defaults::semantic_search as d;
        Self {
            embedding_service_url: d::EMBEDDING_SERVICE_URL.into(),
            supported_models: Vec::new(),
            query_embedding_cache_ttl_secs: d::QUERY_EMBEDDING_CACHE_TTL.as_secs(),
            partition_history_ttl_secs: d::PARTITION_HISTORY_TTL.as_secs(),
            embedding_request_timeout_secs: d::EMBEDDING_REQUEST_TIMEOUT.as_secs(),
            embedding_connect_timeout_secs: d::EMBEDDING_CONNECT_TIMEOUT.as_secs(),
            maintenance: MaintenanceSection::default(),
            search_defaults: SearchDefaultsSection::default(),
        }
    }
}

/// `[semantic_search.maintenance]`: how each namespace's partition is split,
/// reassigned and merged. `threads` and `rebuild_threads` apply to the
/// process; the rest are copied into a namespace's schema when it enables
/// semantic search.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MaintenanceSection {
    /// Partition-maintenance threads per process, 1 to 64.
    pub threads: usize,
    /// Threads a partition rebuild's k-means uses, 1 or 2.
    pub rebuild_threads: usize,
    /// Chunks per posting aimed at; a posting splits above twice this.
    pub target_posting_size: u32,
    /// Nearest postings re-checked after a split (0: none).
    pub reassign_range: u32,
    /// The merge limit as a share of the split limit.
    pub merge_ratio: f64,
    /// Move a chunk only when its code shows it is clearly closer to the new
    /// posting.
    pub skip_uncertain_moves: bool,
    /// Balanced 2-means: members sampled per iteration.
    pub split_samples: u32,
    /// Balanced 2-means: random starting pairs tried.
    pub split_init_trials: u32,
    /// Balanced 2-means: iteration cap.
    pub split_max_iters: u32,
    /// Balanced 2-means: balance factor.
    pub split_lambda_factor: f64,
}

impl Default for MaintenanceSection {
    fn default() -> Self {
        use defaults::semantic_search::maintenance as d;
        Self {
            threads: d::THREADS,
            rebuild_threads: d::REBUILD_THREADS,
            target_posting_size: d::TARGET_POSTING_SIZE,
            reassign_range: d::REASSIGN_RANGE,
            merge_ratio: d::MERGE_RATIO,
            skip_uncertain_moves: d::SKIP_UNCERTAIN_MOVES,
            split_samples: d::SPLIT_SAMPLES,
            split_init_trials: d::SPLIT_INIT_TRIALS,
            split_max_iters: d::SPLIT_MAX_ITERS,
            split_lambda_factor: d::SPLIT_LAMBDA_FACTOR,
        }
    }
}

/// `[semantic_search.search_defaults]`: search settings copied into a
/// namespace's schema when it enables semantic search; a request can override
/// them.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SearchDefaultsSection {
    /// Floor on the postings a query reads.
    pub min_probes: u32,
    /// Postings a query reads: its nearest this many.
    pub max_probes: u32,
    /// Candidates Pass 1 hands to Pass 2.
    pub first_pass_top_k: u32,
    /// Results a search returns.
    pub top_k: u32,
}

impl Default for SearchDefaultsSection {
    fn default() -> Self {
        use defaults::semantic_search::search_defaults as d;
        Self {
            min_probes: d::MIN_PROBES,
            max_probes: d::MAX_PROBES,
            first_pass_top_k: d::FIRST_PASS_TOP_K,
            top_k: d::TOP_K,
        }
    }
}

impl SemanticSearchSection {
    /// The allowed models as (lower-cased name, dimension); empty: any.
    pub fn supported_models(&self) -> Vec<(String, u32)> {
        self.supported_models
            .iter()
            .map(|m| (m.name.to_lowercase(), u32::from(m.dimension)))
            .collect()
    }

    /// Check what a typed value cannot rule out. The ranges of the schema
    /// defaults are checked where the schema's own rules live, so only with
    /// `doc-store`.
    pub(super) fn validate(&self) -> Result<(), String> {
        let mut seen = std::collections::HashSet::new();
        for m in &self.supported_models {
            if m.name.trim().is_empty() {
                return Err("semantic_search.supported_models: an entry has an empty name".into());
            }
            if m.dimension == 0 {
                return Err(format!("semantic_search.supported_models: '{}' has dimension 0", m.name));
            }
            if !seen.insert(m.name.to_lowercase()) {
                return Err(format!("semantic_search.supported_models: '{}' is listed twice", m.name));
            }
        }
        let t = self.maintenance.threads;
        if !(1..=64).contains(&t) {
            return Err(format!("semantic_search.maintenance.threads must be between 1 and 64, got {t}"));
        }
        #[cfg(feature = "semantic-search")]
        {
            let max = crate::semantic_search::ivf_split::rebuild::MAX_REBUILD_THREADS;
            let r = self.maintenance.rebuild_threads;
            if !(1..=max).contains(&r) {
                return Err(format!(
                    "semantic_search.maintenance.rebuild_threads must be between 1 and {max}, got {r}"
                ));
            }
        }
        #[cfg(feature = "doc-store")]
        self.index_defaults().validate().map_err(|e| format!("semantic_search: {e}"))?;
        Ok(())
    }

    /// The values a namespace's vector-index settings start from.
    #[cfg(feature = "doc-store")]
    pub fn index_defaults(&self) -> crate::vector_index::vector_settings::IndexDefaults {
        use crate::vector_index::vector_settings::{IndexDefaults, MaintenanceSettings, Ratio, SearchSettings};
        let m = &self.maintenance;
        let s = &self.search_defaults;
        IndexDefaults {
            search: SearchSettings {
                min_probes: s.min_probes,
                max_probes: s.max_probes,
                first_pass_top_k: s.first_pass_top_k,
                top_k: s.top_k,
            },
            maintenance: MaintenanceSettings {
                target_posting_size: m.target_posting_size,
                reassign_range: m.reassign_range,
                merge_ratio: Ratio(m.merge_ratio),
                skip_uncertain_moves: m.skip_uncertain_moves,
                split_samples: m.split_samples,
                split_init_trials: m.split_init_trials,
                split_max_iters: m.split_max_iters,
                split_lambda_factor: Ratio(m.split_lambda_factor),
            },
        }
    }
}

#[cfg(feature = "semantic-search")]
impl super::ConfigFile {
    /// The embedding-service settings. The per-namespace fields stay at their
    /// defaults: each namespace's own settings replace them when it is used.
    pub fn semantic_search_config(&self) -> crate::semantic_search::service::SemanticSearchConfig {
        use std::time::Duration;
        let s = &self.semantic_search;
        crate::semantic_search::service::SemanticSearchConfig {
            embedding_service_url: s.embedding_service_url.clone(),
            query_embedding_cache_ttl: Duration::from_secs(s.query_embedding_cache_ttl_secs),
            partition_history_ttl: Duration::from_secs(s.partition_history_ttl_secs),
            embedding_request_timeout: Duration::from_secs(s.embedding_request_timeout_secs),
            embedding_connect_timeout: Duration::from_secs(s.embedding_connect_timeout_secs),
            ..Default::default()
        }
    }
}

#[cfg(all(feature = "semantic-search", feature = "doc-store"))]
impl super::ConfigFile {
    /// The embedding worker's configuration.
    pub fn vector_index_config(&self) -> crate::semantic_search::workers::vec_index::VectorIndexConfig {
        crate::semantic_search::workers::vec_index::VectorIndexConfig {
            retry_wait_secs: self.vector_index.retry_wait_secs,
            max_retries: self.vector_index.max_retries,
            concurrency: self.vector_index.concurrency,
        }
    }

    /// The semantic-search context a doc store opens with: the service, the
    /// allowed models, the index defaults and the maintenance threads.
    pub fn semantic_search_context(&self) -> crate::semantic_search::context::SemanticSearchContext {
        let s = &self.semantic_search;
        crate::semantic_search::context::SemanticSearchContext::new(self.semantic_search_config())
            .with_supported_models(s.supported_models())
            .with_index_defaults(s.index_defaults())
            .with_maintenance_threads(s.maintenance.threads)
            .with_rebuild_threads(s.maintenance.rebuild_threads)
    }
}
