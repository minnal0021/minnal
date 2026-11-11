//! A store's vector partition through the doc store: health, centres, recall
//! checks, probe traces, history, and rebuilds (a `partition_rebuild` job).

use crate::doc_store::*;

impl DocStore {
    /// What a partition admin call runs with: the search context, and `namespace`'s
    /// settings and id once it is known to have semantic search on.
    #[cfg(feature = "semantic-search")]
    fn vector_admin_target(
        &self,
        namespace: &str,
    ) -> Result<
        (
            &Arc<SemanticSearchContext>,
            crate::doc_store::schema::vector_settings::VectorIndexSettings,
            crate::namespace::NamespaceId,
        ),
        DocStoreError,
    > {
        let Some(ctx) = &self.semantic_ctx else {
            return Err(DocStoreError::EmbeddingFailed("semantic search not configured on this store".into()));
        };
        self.schemas.require_semantic_search(namespace)?;
        let (settings, ns_id) = self.schemas.vector_settings(namespace)?;
        Ok((ctx, settings, ns_id))
    }

    /// The health of `namespace`'s partition (document or KV store): every
    /// posting's size, state and lineage from memory, and with `codes` also the
    /// code metrics (residual inflation, error band, foreign and zero-centre
    /// shares), which read every chunk code. Returns the namespace's
    /// maintenance settings beside them.
    #[cfg(feature = "semantic-search")]
    pub async fn vector_partition_health(
        &self,
        namespace: &str,
        codes: bool,
    ) -> Result<
        (
            crate::doc_store::schema::vector_settings::MaintenanceSettings,
            crate::semantic_search::ivf_split::health::PartitionHealth,
            Vec<crate::semantic_search::ivf_split::health::PostingHealth>,
        ),
        DocStoreError,
    > {
        let (ctx, settings, ns_id) = self.vector_admin_target(namespace)?;
        let (summary, postings) = ctx.partition_health(&self.db, namespace, ns_id, &settings, codes).await?;
        Ok((settings.maintenance, summary, postings))
    }

    /// Every IVF centre of `namespace` (document or KV store) with the postings
    /// that name it, from memory; with `codes` also how many chunk codes decode
    /// against each, which reads every chunk code. No centre vectors are
    /// returned.
    #[cfg(feature = "semantic-search")]
    pub async fn vector_centres(
        &self,
        namespace: &str,
        codes: bool,
    ) -> Result<Vec<crate::semantic_search::ivf_split::health::CentreHealth>, DocStoreError> {
        let (ctx, settings, ns_id) = self.vector_admin_target(namespace)?;
        Ok(ctx.centre_health(&self.db, namespace, ns_id, &settings, codes).await?)
    }

    /// Check whether `namespace`'s searches miss what a full read finds, the
    /// signal for a rebuild (FR-007): sample real queries from the query-embedding
    /// cache, or stored chunks when fewer than 20 are cached (no embedding call
    /// either way), and compare each search under the namespace's settings,
    /// or `overrides` of them, with one that probes every posting. The result is
    /// recorded in the partition history. Costs `params.sample` normal searches
    /// plus as many full reads, one at a time.
    #[cfg(feature = "semantic-search")]
    pub async fn vector_recall_check(
        &self,
        namespace: &str,
        params: &crate::semantic_search::ivf_split::recall_check::RecallCheckParams,
        overrides: &crate::doc_store::schema::vector_settings::SearchSpec,
    ) -> Result<crate::semantic_search::ivf_split::recall_check::RecallCheck, DocStoreError> {
        let (ctx, settings, ns_id) = self.vector_admin_target(namespace)?;
        // Document or KV store alike.
        let ns = self.namespace_semantics(namespace, Some(ns_id), &settings, overrides).await?;
        Ok(ctx.recall_check(&self.db, namespace, &ns, params).await?)
    }

    /// How a semantic query walks `namespace`'s partition (document or KV
    /// store): the postings at probe ranks `start..end` (0 is the posting
    /// nearest the query), each with its distance from the query, the documents
    /// scored in it, and its lowest and highest Pass-1 document score, plus a
    /// [`SearchBaseline`](crate::semantic_search::service::SearchBaseline): the
    /// same query searched as the namespace searches it. Each reported document
    /// says whether that search returns it.
    ///
    /// `start` defaults to 0. `end` defaults to `start` plus the namespace's
    /// `max_probes` when `start` is not given, or plus
    /// [`MAX_TRACE_PROBES`](crate::semantic_search::service::MAX_TRACE_PROBES)
    /// when it is; either way capped at the namespace's postings. The window must
    /// satisfy `start < end <= postings` and be at most `MAX_TRACE_PROBES` wide,
    /// or [`DocStoreError::InvalidProbeTrace`].
    ///
    /// `top_k` sets how many results the baseline search returns (default the
    /// namespace's `top_k`, at most
    /// [`MAX_TRACE_TOP_K`](crate::semantic_search::service::MAX_TRACE_TOP_K));
    /// its Pass 1 keeps the namespace's `first_pass_top_k` candidates, or `top_k`
    /// when that is more. The namespace's own settings do not change.
    ///
    /// The query is embedded as a search embeds it (from the query-embedding
    /// cache when it is there). Costs one search plus one Pass 1 over the window.
    #[cfg(feature = "semantic-search")]
    pub async fn vector_probe_trace(
        &self,
        namespace: &str,
        query_text: &str,
        start: Option<u32>,
        end: Option<u32>,
        top_k: Option<u32>,
    ) -> Result<crate::semantic_search::service::ProbeTrace, DocStoreError> {
        let (ctx, settings, ns_id) = self.vector_admin_target(namespace)?;
        let ns = self
            .namespace_semantics(namespace, Some(ns_id), &settings, &SearchSpec::default())
            .await?;
        Ok(ctx.probe_trace(&self.db, namespace, &ns, query_text, start, end, top_k).await?)
    }

    /// The server's semantic-search settings, as configured (`[semantic_search]`
    /// in the server's TOML): the embedding service, the allowed models, the
    /// TTLs, the maintenance and rebuild threads, and the defaults a namespace's
    /// vector-index settings are filled with when it enables semantic search.
    /// `None` when semantic search is not configured.
    #[cfg(feature = "semantic-search")]
    pub fn vector_server_settings(&self) -> Option<serde_json::Value> {
        Some(self.semantic_ctx.as_ref()?.server_settings())
    }

    /// How long partition-history records are kept (the server's
    /// `partition_history_ttl_secs`; default 30 days without semantic search).
    #[cfg(feature = "semantic-search")]
    pub fn vector_partition_history_ttl(&self) -> std::time::Duration {
        self.semantic_ctx
            .as_ref()
            .map_or(crate::semantic_search::ivf_split::history::DEFAULT_HISTORY_TTL, |c| {
                c.config.partition_history_ttl
            })
    }

    /// `namespace`'s partition history, newest first: code-health reads, recall
    /// checks and rebuilds, kept for the server's `partition_history_ttl_secs`
    /// (default 30 days). `kind` keeps one
    /// kind only; at most `limit` records.
    #[cfg(feature = "semantic-search")]
    pub async fn vector_partition_history(
        &self,
        namespace: &str,
        kind: Option<crate::semantic_search::ivf_split::history::RecordKind>,
        limit: usize,
    ) -> Result<Vec<crate::semantic_search::ivf_split::history::HistoryRecord>, DocStoreError> {
        let Some(ctx) = &self.semantic_ctx else {
            return Err(DocStoreError::EmbeddingFailed("semantic search not configured on this store".into()));
        };
        self.schemas.require_semantic_search(namespace)?;
        Ok(ctx.partition_history(&self.db, namespace, kind, limit).await?)
    }

    /// Rebuild `namespace`'s partition from its codes (design doc M4), as a
    /// `partition_rebuild` job in the background; returns the job's name. K-means
    /// into `postings` postings (default: as many as are active), every chunk
    /// moved to its nearest, then the usual splits and merges. No embedding call
    /// is made. Progress and the result show in
    /// [`vector_rebuild_status`](Self::vector_rebuild_status). Refused with
    /// [`DocStoreError::JobConflict`] while another vector job of the store runs.
    #[cfg(feature = "semantic-search")]
    pub fn start_vector_rebuild(&self, namespace: &str, postings: Option<usize>) -> Result<String, DocStoreError> {
        self.running_governor()?;
        let key = self.start_partition_rebuild(namespace, postings)?;
        self.submit_job(namespace, key)?;
        Ok(key.kind.name().to_owned())
    }

    /// [`start_vector_rebuild`](Self::start_vector_rebuild), returning once
    /// the rebuild has ended.
    #[cfg(feature = "semantic-search")]
    pub async fn rebuild_vector_partition(&self, namespace: &str, postings: Option<usize>) -> Result<(), DocStoreError> {
        let key = self.start_partition_rebuild(namespace, postings)?;
        self.run_job(namespace, key).await
    }

    #[cfg(feature = "semantic-search")]
    pub(in crate::doc_store) fn start_partition_rebuild(&self, namespace: &str, postings: Option<usize>) -> Result<JobKey, DocStoreError> {
        self.job_runner_ready()?;
        let (_, _, ns_id) = self.vector_admin_target(namespace)?;
        let key = JobKey::new(ns_id, crate::namespace_lifecycle::JobKind::PartitionRebuild);
        let progress = RebuildProgress {
            postings,
            above: None,
            report: None,
        };
        self.start_job(namespace, key, "partition", &progress)?;
        Ok(key)
    }

    /// The body of a `partition_rebuild` job.
    ///
    /// A run that started before (a crash, or a failure) is not run again:
    /// the partition's journal finished the rebuild if it got past its
    /// publish, and abandoned it otherwise, and the job is closed with that
    /// outcome. K-means is minutes on a large store, and repeating it after
    /// an error would repeat whatever caused it.
    #[cfg(feature = "semantic-search")]
    pub(in crate::doc_store) async fn partition_rebuild_job(&self, namespace: &str, key: JobKey) -> Result<(), DocStoreError> {
        use crate::doc_store::indexing::record_error;
        let Some(record) = self.jobs.get(key)?.filter(crate::namespace_lifecycle::JobRecord::is_live) else {
            return Ok(());
        };
        let mut progress: RebuildProgress = record.progress().ok_or_else(|| record_error(key, "no rebuild progress"))?;
        let (ctx, settings, ns_id) = self.vector_admin_target(namespace)?;
        if let Some(above) = progress.above {
            if ctx.rebuilt_above(&self.db, namespace, ns_id, &settings, above).await? {
                return Ok(());
            }
            // Final, so the governor records the job failed without running it
            // again.
            return Err(DocStoreError::LifecycleTaskAbandoned(
                "the rebuild was cut short before it published, and recovery abandoned it; start it again".to_owned(),
            ));
        }
        let cancelled = || self.jobs.is_cancelling(key);
        if cancelled() {
            return Ok(());
        }
        progress.above = Some(ctx.max_posting_id(&self.db, namespace, ns_id, &settings).await?);
        self.jobs.save_progress(key, serde_json::to_value(&progress)?)?;
        if let Some(report) = ctx.rebuild(&self.db, namespace, ns_id, &settings, progress.postings, &cancelled).await? {
            // A cancel asked after the plan was recorded came too late.
            self.jobs.refuse_cancel(key)?;
            progress.report = Some(report);
            self.save_job_progress(key, &progress);
        }
        Ok(())
    }

    /// `namespace`'s last partition rebuild (its `partition_rebuild` job), if
    /// it has had one.
    #[cfg(feature = "semantic-search")]
    pub fn vector_rebuild_status(&self, namespace: &str) -> Option<crate::semantic_search::ivf_split::rebuild::RebuildStatus> {
        use crate::semantic_search::ivf_split::rebuild::{RebuildState, RebuildStatus};
        let ns_id = self.schemas.get(namespace)?.ns_id()?;
        let record = self
            .jobs
            .get(JobKey::new(ns_id, crate::namespace_lifecycle::JobKind::PartitionRebuild))
            .ok()??;
        let progress: RebuildProgress = record.progress()?;
        let state = match record.status {
            BuildStatus::Complete => RebuildState::Done,
            BuildStatus::Failed => RebuildState::Failed,
            BuildStatus::Cancelled => RebuildState::Cancelled,
            BuildStatus::Pending | BuildStatus::Running | BuildStatus::Cancelling => RebuildState::Running,
        };
        Some(RebuildStatus {
            state,
            postings_requested: progress.postings,
            started_at_ms: record.started_at_ms,
            finished_at_ms: record.status.is_terminal().then_some(record.updated_at_ms),
            report: progress.report,
            error: record.error,
        })
    }
}

/// The progress of a `partition_rebuild` job.
#[cfg(feature = "semantic-search")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct RebuildProgress {
    /// Postings asked for (`None`: as many as are active).
    postings: Option<usize>,
    /// The highest posting id before the rebuild started; set when it starts.
    above: Option<u32>,
    /// What it did, once done.
    report: Option<crate::semantic_search::ivf_split::rebuild::RebuildReport>,
}
