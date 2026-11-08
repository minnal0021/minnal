//! Inspecting and rebuilding a namespace's partition: its health, its centres, a
//! recall check, a probe trace, its history and a background rebuild. All of it runs
//! on what the context holds. The store above decides whether a namespace may be
//! asked about and resolves its settings.

use crate::namespace::NamespaceId;
use std::collections::HashSet;
use std::sync::Arc;

use crate::AsyncDb;
use crate::semantic_search::context::{ContextError, NamespaceSemantics, SemanticSearchContext};
use crate::semantic_search::ivf_split::health::{self, CentreHealth, PartitionHealth, PostingHealth};
use crate::semantic_search::ivf_split::history::{self, HistoryRecord, RecordKind};
use crate::semantic_search::ivf_split::rebuild::{self, RebuildParams};
use crate::semantic_search::ivf_split::recall_check::{self, RecallCheck, RecallCheckParams};
use crate::semantic_search::service::{self, MAX_TRACE_PROBES, MAX_TRACE_TOP_K, ProbeTrace, SearchBaseline};
use crate::semantic_search::vector_kv;
use crate::vector_index::vector_settings::VectorIndexSettings;

/// Cached query embeddings a recall check reads at most (about 3 KB each at 768
/// dimensions), to sample from.
const CACHED_QUERIES_READ: usize = 2_000;

/// Record a partition-history entry; a failure loses the record, never the
/// result it describes.
async fn record_history(db: &AsyncDb, ttl: std::time::Duration, namespace: &str, kind: RecordKind, data: &impl serde::Serialize) {
    if let Err(e) = history::record(db, namespace, kind, data, ttl).await {
        log::warn!("partition history: namespace='{namespace}' {kind:?} not recorded: {e}");
    }
}

impl SemanticSearchContext {
    /// `namespace`'s partition summary and every posting's health; with `codes`
    /// also the code metrics, which read every chunk code and are recorded in the
    /// partition history.
    pub(crate) async fn partition_health(
        &self,
        db: &AsyncDb,
        namespace: &str,
        ns_id: NamespaceId,
        settings: &VectorIndexSettings,
        codes: bool,
    ) -> Result<(PartitionHealth, Vec<PostingHealth>), ContextError> {
        let ns = self.for_namespace(db, namespace, ns_id, settings).await?;
        let (summary, postings) = health::partition_health(db, namespace, &ns.partition, codes).await?;
        if codes {
            record_history(db, self.config.partition_history_ttl, namespace, RecordKind::Codes, &summary).await;
        }
        Ok((summary, postings))
    }

    /// Every IVF centre of `namespace` with the postings that name it.
    pub(crate) async fn centre_health(
        &self,
        db: &AsyncDb,
        namespace: &str,
        ns_id: NamespaceId,
        settings: &VectorIndexSettings,
        codes: bool,
    ) -> Result<Vec<CentreHealth>, ContextError> {
        let ns = self.for_namespace(db, namespace, ns_id, settings).await?;
        Ok(health::centre_health(db, namespace, &ns.partition, codes).await?)
    }

    /// Compare `namespace`'s searches with full reads, preferring real queries from
    /// the query-embedding cache, and record the result in the partition history.
    pub(crate) async fn recall_check(
        &self,
        db: &AsyncDb,
        namespace: &str,
        ns: &NamespaceSemantics,
        params: &RecallCheckParams,
    ) -> Result<RecallCheck, ContextError> {
        // Real queries first: the cache holds the embeddings of recent searches
        // for this model and dimension (one bounded page read, no service call).
        let cached = vector_kv::sample_cached_query_embeddings(
            db,
            &ns.config.model_name,
            ns.config.embedding_dim,
            ns.config.query_embedding_cache_ttl,
            CACHED_QUERIES_READ,
            params.seed,
        )
        .await?;
        let check = recall_check::recall_check(db, namespace, &ns.partition, &ns.config, params, cached).await?;
        record_history(db, self.config.partition_history_ttl, namespace, RecordKind::RecallCheck, &check).await;
        Ok(check)
    }

    /// How a query walks `namespace`'s partition over probe ranks `start..end`,
    /// beside a baseline search at `top_k`.
    #[allow(clippy::too_many_arguments)] // the request's own parameters; not worth a struct
    pub(crate) async fn probe_trace(
        &self,
        db: &AsyncDb,
        namespace: &str,
        ns: &NamespaceSemantics,
        query_text: &str,
        start: Option<u32>,
        end: Option<u32>,
        top_k: Option<u32>,
    ) -> Result<ProbeTrace, ContextError> {
        let invalid = |reason: String| {
            Err(ContextError::InvalidProbeTrace {
                namespace: namespace.to_owned(),
                reason,
            })
        };
        if let Some(k) = top_k.filter(|&k| k == 0 || k > MAX_TRACE_TOP_K) {
            return invalid(format!("top_k ({k}) must be between 1 and {MAX_TRACE_TOP_K}"));
        }
        // The baseline searches as the namespace does, at this trace's top_k.
        let mut config = ns.config.clone();
        if let Some(k) = top_k {
            config.top_k_results = k as usize;
        }
        config.first_pass_sparse_search_top_k = config.first_pass_sparse_search_top_k.max(config.top_k_results);
        let (query_dense, query_sparse) = ns.query_embeddings(db, query_text).await?;
        let db_store = vector_kv::DbVectorStore::new(db, namespace)
            .await
            .map_err(|e| ContextError::EmbeddingFailed(e.to_string()))?;
        // Under the routing epoch, as a search: a split cannot delete a
        // posting's keys while this may still scan by the routing before it.
        let _epoch = ns.partition.read_epoch().await;
        let partition = ns.partition.snapshot();
        let active = partition.postings();
        let max_width = MAX_TRACE_PROBES as usize;
        let start_rank = start.map_or(0, |s| s as usize);
        let width = if start.is_some() {
            max_width
        } else {
            ns.config.probe.per_query().min(max_width)
        };
        let end_rank = end.map_or_else(|| start_rank.saturating_add(width).min(active), |e| e as usize);
        if start_rank >= end_rank {
            return invalid(format!("start ({start_rank}) must be below end ({end_rank})"));
        }
        if end_rank - start_rank > max_width {
            return invalid(format!("end - start ({}) is more than {max_width}", end_rank - start_rank));
        }
        if end_rank > active {
            return invalid(format!("end ({end_rank}) is past the namespace's {active} postings"));
        }
        let results = service::search(
            &config,
            namespace,
            &*partition,
            &query_sparse,
            &query_dense,
            &db_store,
            None::<fn(&[u8]) -> bool>,
            None,
        )
        .await?;
        let baseline = SearchBaseline {
            n_probes: config.probe.per_query().min(active),
            top_k: config.top_k_results,
            first_pass_top_k: config.first_pass_sparse_search_top_k,
            results: results.len(),
            lowest_pass2: results.iter().map(|r| r.dot_product).min_by(f32::total_cmp),
        };
        let mut postings = service::pass1_probe_trace(&config, namespace, &*partition, &query_dense, start_rank..end_rank, &db_store).await?;
        let returned: HashSet<&[u8]> = results.iter().map(|r| r.document_id.as_slice()).collect();
        for s in postings.iter_mut().flat_map(|p| [&mut p.min, &mut p.max]).flatten() {
            s.in_results = Some(returned.contains(s.document_id.as_slice()));
        }
        Ok(ProbeTrace {
            start: start_rank,
            end: end_rank,
            active_postings: active,
            pass1_bits: ns.config.pass1_bits,
            postings,
            baseline,
        })
    }

    /// The semantic-search settings this context was configured with.
    pub(crate) fn server_settings(&self) -> serde_json::Value {
        let (search, m) = (self.index_defaults.search, self.index_defaults.maintenance);
        serde_json::json!({
            "embedding_service_url": self.config.embedding_service_url,
            "supported_models": self.supported_models,
            "embedding_request_timeout_secs": self.config.embedding_request_timeout.as_secs(),
            "embedding_connect_timeout_secs": self.config.embedding_connect_timeout.as_secs(),
            "query_embedding_cache_ttl_secs": self.config.query_embedding_cache_ttl.as_secs(),
            "partition_history_ttl_secs": self.config.partition_history_ttl.as_secs(),
            "maintenance_threads": self.maintenance_threads,
            "rebuild_threads": self.rebuild_threads,
            "max_rebuild_threads": rebuild::MAX_REBUILD_THREADS,
            "recall_check": {
                "max_sample": recall_check::MAX_SAMPLE,
                "min_cached_queries": recall_check::MIN_GIVEN_QUERIES,
                "max_reference_entries": recall_check::MAX_REFERENCE_ENTRIES,
                "default_min_overlap": recall_check::DEFAULT_MIN_OVERLAP,
            },
            "namespace_defaults": {
                "search": {
                    "min_probes": search.min_probes,
                    "max_probes": search.max_probes,
                    "first_pass_top_k": search.first_pass_top_k,
                    "top_k": search.top_k,
                },
                "maintenance": {
                    "target_posting_size": m.target_posting_size,
                    "split_limit": m.split_limit(),
                    "merge_limit": m.merge_limit(),
                    "reassign_range": m.reassign_range,
                    "merge_ratio": m.merge_ratio.0,
                    "skip_uncertain_moves": m.skip_uncertain_moves,
                    "split_samples": m.split_samples,
                    "split_init_trials": m.split_init_trials,
                    "split_max_iters": m.split_max_iters,
                    "split_lambda_factor": m.split_lambda_factor.0,
                },
            },
        })
    }

    /// `namespace`'s partition history, newest first; `kind` keeps one kind only.
    pub(crate) async fn partition_history(
        &self,
        db: &AsyncDb,
        namespace: &str,
        kind: Option<RecordKind>,
        limit: usize,
    ) -> Result<Vec<HistoryRecord>, ContextError> {
        Ok(history::read(db, namespace, kind, limit, self.config.partition_history_ttl).await?)
    }

    /// Start rebuilding `namespace`'s partition in the background into `postings`
    /// postings; its progress and result show in [`rebuild_status`](Self::rebuild_status).
    pub(crate) fn start_rebuild(
        self: &Arc<Self>,
        db: &Arc<AsyncDb>,
        namespace: &str,
        ns_id: NamespaceId,
        settings: VectorIndexSettings,
        postings: Option<usize>,
    ) -> Result<(), ContextError> {
        self.claim_rebuild(namespace, postings)?;
        let (db, ctx, namespace) = (Arc::clone(db), Arc::clone(self), namespace.to_owned());
        tokio::spawn(async move {
            let result = async {
                let ns = ctx.for_namespace(&db, &namespace, ns_id, &settings).await?;
                let params = RebuildParams {
                    postings,
                    threads: ctx.rebuild_threads,
                };
                let ops = settings.maintenance.ops();
                Ok::<_, ContextError>(rebuild::rebuild_partition(&db, &namespace, &ns.partition, &ops, &params).await?)
            }
            .await;
            match &result {
                Ok(r) => log::info!(
                    "partition rebuild: namespace='{namespace}' {} chunks, {} -> {} postings ({} fitted, {} iterations, {} moved)",
                    r.chunks,
                    r.postings_before,
                    r.postings_after,
                    r.postings_fitted,
                    r.iterations,
                    r.chunks_moved
                ),
                Err(e) => log::warn!("partition rebuild: namespace='{namespace}' failed: {e}"),
            }
            ctx.finish_rebuild(&namespace, result.map_err(|e| e.to_string()));
            if let Some(status) = ctx.rebuild_status(&namespace) {
                record_history(&db, ctx.config.partition_history_ttl, &namespace, RecordKind::Rebuild, &status).await;
            }
        });
        Ok(())
    }
}
