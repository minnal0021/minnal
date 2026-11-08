//! The semantic-search context attached to a `DocStore`: the shared service
//! configuration, each namespace's loaded partition, and what a drop or a rebuild
//! coordinates through.

use crate::namespace::NamespaceId;
use std::sync::Arc;

use crate::semantic_search::PartitionHandle;
use crate::semantic_search::service::SemanticSearchConfig;
use crate::vector_index::vector_settings::{SearchSettings, SearchSpec, VectorIndexSettings, VectorSettingsError};

/// Why the [`SemanticSearchContext`] refused or could not serve a namespace. The
/// document store reports each one as its matching `DocStoreError`.
#[derive(Debug, thiserror::Error)]
pub enum ContextError {
    /// A namespace names an embedding model this server's `supported_models`
    /// does not list.
    #[error("embedding model '{model}' is not supported by this server (supported: {supported})")]
    UnsupportedEmbeddingModel { model: String, supported: String },

    /// A namespace's embedding dimension is not the one the server lists for
    /// its model (`supported_models`).
    #[error("embedding_dim {dim} does not match model '{model}', which this server runs at {expected} dimensions")]
    EmbeddingDimMismatch { model: String, dim: u32, expected: usize },

    /// A semantic namespace has no centres and postings to search or index with.
    #[error("namespace '{namespace}' has no seeded centres; restart to seed them, or drop its vector index and enable it again")]
    VectorIndexNotSeeded { namespace: String },

    /// Semantic search is not enabled for the namespace, or it is being dropped.
    #[error("semantic search is not enabled for namespace '{namespace}'")]
    SemanticSearchNotEnabled { namespace: String },

    /// A partition rebuild of the namespace is already running.
    #[error("a partition rebuild for namespace '{namespace}' is already running")]
    VecRebuildInProgress { namespace: String },

    /// The embedding service could not embed a query, or a query's stores could
    /// not be opened.
    #[error("embedding failed: {0}")]
    EmbeddingFailed(String),

    /// A probe trace was asked for something it cannot show.
    #[error("invalid probe trace for namespace '{namespace}': {reason}")]
    InvalidProbeTrace { namespace: String, reason: String },

    /// A vector-index setting was rejected.
    #[error("schema error: {0}")]
    Settings(#[from] VectorSettingsError),

    /// The underlying database returned an error.
    #[error("database error: {0}")]
    Db(#[from] crate::KVError),
}

/// Semantic-search configuration attached to a [`DocStore`](crate::doc_store::DocStore).
///
/// When present, writes to namespaces with `semantic_search_enabled = true`
/// enqueue a pending embedding job instead of calling the embedding service
/// inline.  A background `VecIndexWorker` processes the queue
/// asynchronously, making the write path independent of embedding service
/// availability.
///
/// Construct once at startup and attach via
/// [`DocStore::with_semantic_search`](crate::doc_store::DocStore::with_semantic_search).
///
/// It holds what is engine-wide: the embedding service settings, the defaults
/// a namespace's vector-index settings start from, and optionally the models
/// this server allows (each with its dimension). A namespace that enables semantic search starts with one root
/// posting ([`init_namespace`]) and grows its own partition by splitting
/// (design doc M3); it is loaded once into a registry of
/// [`PartitionHandle`]s. Everything that shapes a namespace's index comes from
/// its schema; [`for_namespace`] combines the two into the per-call
/// [`SemanticSearchConfig`] and the namespace's partition.
///
/// [`init_namespace`]: SemanticSearchContext::init_namespace
///
/// [`for_namespace`]: SemanticSearchContext::for_namespace
pub struct SemanticSearchContext {
    /// Embedding service settings: URL, timeouts and the query-cache TTL. Its
    /// per-namespace fields are ignored; [`for_namespace`](Self::for_namespace)
    /// overwrites them from the namespace's settings.
    pub config: SemanticSearchConfig,
    /// The models this server allows, lower-cased name → dimension. Empty:
    /// any model the embedding service serves (checked by probing it when a
    /// namespace enables semantic search).
    pub supported_models: std::collections::BTreeMap<String, u32>,
    /// What a namespace's vector-index settings are filled with when it first
    /// enables semantic search (the server's TOML defaults, or built-in).
    pub index_defaults: crate::vector_index::vector_settings::IndexDefaults,
    /// Namespaces maintained at once by the partition maintenance task.
    pub maintenance_threads: usize,
    /// Threads a partition rebuild's k-means uses (at most
    /// [`MAX_REBUILD_THREADS`](crate::semantic_search::ivf_split::rebuild::MAX_REBUILD_THREADS)).
    pub rebuild_threads: usize,
    /// Each namespace's last rebuild since start.
    rebuilds: parking_lot::Mutex<std::collections::HashMap<String, crate::semantic_search::ivf_split::rebuild::RebuildStatus>>,
    /// Reaches the partition maintenance task, once it runs.
    maintenance_tx: std::sync::OnceLock<tokio::sync::mpsc::UnboundedSender<String>>,
    /// Each namespace's partition, loaded once from its stores. Keyed by name and
    /// `ns_id`, so a namespace dropped and recreated under the same name never
    /// reuses the old one.
    ivfs: parking_lot::RwLock<std::collections::HashMap<(String, NamespaceId), Arc<PartitionHandle>>>,
    /// Namespaces whose vector index is being dropped: no partition of theirs
    /// is loaded or handed out until the drop ends ([`begin_drop`](Self::begin_drop)).
    dropping: parking_lot::Mutex<std::collections::HashSet<String>>,
    /// How many drops have begun, of any namespace. A load compares it before
    /// and after: [`begin_drop`](Self::begin_drop) can only stop a partition
    /// that is installed, so a load it could not see must not install what it read.
    drops_begun: std::sync::atomic::AtomicU64,
}

/// What one namespace's semantic operations run with: the per-call config built
/// from its settings, and its model's centroids.
pub struct NamespaceSemantics {
    /// Service settings plus the namespace's model, dimension, chunking, code
    /// widths and search defaults.
    pub config: SemanticSearchConfig,
    /// The namespace's own partition (its centres, postings, rotation and
    /// counts). Take [`PartitionHandle::read_epoch`] before routing a write or
    /// running a search, then use its [`snapshot`](PartitionHandle::snapshot).
    pub partition: Arc<PartitionHandle>,
    /// The namespace's search settings, from which `config.probe` was computed
    /// (kept so a request's overrides apply to them, not to the computed budget).
    pub search: SearchSettings,
    /// The namespace's partition maintenance settings.
    pub maintenance: crate::vector_index::vector_settings::MaintenanceSettings,
}

impl SemanticSearchContext {
    /// A context with the given service settings, allowing any model.
    pub fn new(config: SemanticSearchConfig) -> Self {
        Self {
            config,
            supported_models: std::collections::BTreeMap::new(),
            index_defaults: crate::vector_index::vector_settings::IndexDefaults::default(),
            maintenance_threads: crate::semantic_search::workers::ivf_maintenance::DEFAULT_MAINTENANCE_THREADS,
            rebuild_threads: crate::semantic_search::ivf_split::rebuild::MAX_REBUILD_THREADS,
            rebuilds: parking_lot::Mutex::new(std::collections::HashMap::new()),
            maintenance_tx: std::sync::OnceLock::new(),
            ivfs: parking_lot::RwLock::new(std::collections::HashMap::new()),
            dropping: parking_lot::Mutex::new(std::collections::HashSet::new()),
            drops_begun: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Maintain at most `threads` namespaces' partitions at once.
    pub fn with_maintenance_threads(mut self, threads: usize) -> Self {
        self.maintenance_threads = threads.max(1);
        self
    }

    /// Allow only these models, each at its dimension (names are lower-cased).
    pub fn with_supported_models(mut self, models: impl IntoIterator<Item = (String, u32)>) -> Self {
        self.supported_models = models.into_iter().map(|(m, d)| (m.to_lowercase(), d)).collect();
        self
    }

    /// Run rebuilds' k-means on `threads` threads (1 to
    /// [`MAX_REBUILD_THREADS`](crate::semantic_search::ivf_split::rebuild::MAX_REBUILD_THREADS)).
    pub fn with_rebuild_threads(mut self, threads: usize) -> Self {
        self.rebuild_threads = threads.clamp(1, crate::semantic_search::ivf_split::rebuild::MAX_REBUILD_THREADS);
        self
    }

    /// Mark a rebuild of `namespace` as running, unless one already is.
    pub(crate) fn claim_rebuild(&self, namespace: &str, postings: Option<usize>) -> Result<(), ContextError> {
        use crate::semantic_search::ivf_split::rebuild::{RebuildState, RebuildStatus};
        let mut rebuilds = self.rebuilds.lock();
        if rebuilds.get(namespace).is_some_and(|s| s.state == RebuildState::Running) {
            return Err(ContextError::VecRebuildInProgress {
                namespace: namespace.to_owned(),
            });
        }
        rebuilds.insert(
            namespace.to_owned(),
            RebuildStatus {
                state: RebuildState::Running,
                postings_requested: postings,
                started_at_ms: crate::support::now_ms(),
                finished_at_ms: None,
                report: None,
                error: None,
            },
        );
        Ok(())
    }

    /// Record how `namespace`'s running rebuild ended.
    pub(crate) fn finish_rebuild(&self, namespace: &str, result: Result<crate::semantic_search::ivf_split::rebuild::RebuildReport, String>) {
        use crate::semantic_search::ivf_split::rebuild::RebuildState;
        if let Some(status) = self.rebuilds.lock().get_mut(namespace) {
            status.finished_at_ms = Some(crate::support::now_ms());
            match result {
                Ok(report) => {
                    status.state = RebuildState::Done;
                    status.report = Some(report);
                }
                Err(e) => {
                    status.state = RebuildState::Failed;
                    status.error = Some(e);
                }
            }
        }
    }

    /// `namespace`'s last rebuild since start, if any.
    pub fn rebuild_status(&self, namespace: &str) -> Option<crate::semantic_search::ivf_split::rebuild::RebuildStatus> {
        self.rebuilds.lock().get(namespace).cloned()
    }

    pub(crate) fn set_maintenance_sender(&self, tx: tokio::sync::mpsc::UnboundedSender<String>) {
        let _ = self.maintenance_tx.set(tx);
    }

    /// Ask the maintenance task to split `namespace`'s oversized postings. A
    /// no-op until the task runs, and while `namespace` is being dropped.
    pub fn request_maintenance(&self, namespace: &str) {
        if self.is_dropping(namespace) {
            return;
        }
        if let Some(tx) = self.maintenance_tx.get() {
            let _ = tx.send(namespace.to_owned());
        }
    }

    /// Use `defaults` for namespaces that enable semantic search from now on.
    pub fn with_index_defaults(mut self, defaults: crate::vector_index::vector_settings::IndexDefaults) -> Self {
        self.index_defaults = defaults;
        self
    }

    /// Give `namespace` its starting partition, one root posting at the zero
    /// centre, unless it already holds one (a re-enable, or a namespace seeded
    /// from a centroid file before M3). Checks the model first. Durable on
    /// return.
    pub async fn init_namespace(&self, db: &crate::AsyncDb, namespace: &str, settings: &VectorIndexSettings) -> Result<(), ContextError> {
        self.check_model(settings)?;
        crate::semantic_search::vector_kv::create_vector_namespaces(db, namespace).await?;
        if !crate::semantic_search::vector_kv::ivf_exists(db, namespace).await? {
            crate::semantic_search::vector_kv::init_ivf(db, namespace, settings.embedding_dim as usize).await?;
            self.ivfs.write().retain(|(name, _), _| name != namespace);
        }
        Ok(())
    }

    /// Forget every loaded partition of `namespace` (it was dropped).
    pub fn evict(&self, namespace: &str) {
        self.ivfs.write().retain(|(name, _), _| name != namespace);
    }

    /// Start dropping `namespace`'s vector index: refuse to load or hand out
    /// its partition from now on, stop maintenance and vector writes on every
    /// loaded one, and wait for any maintenance operation and vector write in
    /// progress to finish. Only then may the caller remove the namespaces: a
    /// write still using them could land in a store being removed, or in the new
    /// partition of one enabled again. The vector worker holds a partition for a whole pass,
    /// so it checks [`PartitionHandle::is_dropped`] inside the routing epoch
    /// before each write. Call [`end_drop`](Self::end_drop) once they are gone.
    pub async fn begin_drop(&self, namespace: &str) {
        self.dropping.lock().insert(namespace.to_owned());
        // Before the partitions are collected, under the lock a load installs
        // with: a load either sees this count change, or installs in time to be
        // collected below.
        self.drops_begun.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let loaded: Vec<Arc<PartitionHandle>> = {
            let mut ivfs = self.ivfs.write();
            let mut out = Vec::new();
            ivfs.retain(|(name, _), p| {
                if name == namespace {
                    out.push(Arc::clone(p));
                    false
                } else {
                    true
                }
            });
            out
        };
        for partition in loaded {
            partition.mark_dropped();
            drop(partition.lock_maintenance().await);
            // A write that entered the epoch before the mark may not have seen
            // it; one entering after does (the epoch's lock orders the two).
            partition.epoch_barrier().await;
        }
    }

    /// Whether [`begin_drop`](Self::begin_drop) has started dropping
    /// `namespace` and [`end_drop`](Self::end_drop) has not yet ended it.
    pub(crate) fn is_dropping(&self, namespace: &str) -> bool {
        self.dropping.lock().contains(namespace)
    }

    /// The drop [`begin_drop`](Self::begin_drop) started is done; the name may
    /// hold a vector index again.
    pub fn end_drop(&self, namespace: &str) {
        self.dropping.lock().remove(namespace);
    }

    /// Apply one document's [`PostingDelta`] to `namespace`'s entry counts, if
    /// its partition is loaded. When it is not, there is nothing to update: the
    /// load counts from the store.
    ///
    /// [`PostingDelta`]: crate::semantic_search::PostingDelta
    pub fn apply_posting_delta(&self, namespace: &str, delta: &crate::semantic_search::PostingDelta) {
        if delta.added.is_empty() && delta.removed.is_empty() {
            return;
        }
        let mut wake = false;
        for ((name, _), partition) in self.ivfs.read().iter() {
            if name == namespace {
                partition.snapshot().apply_delta(delta);
                wake |= partition.needs_maintenance(delta);
            }
        }
        if wake {
            self.request_maintenance(namespace);
        }
    }

    /// Check that `settings` can run here: when the server lists its models,
    /// the namespace's is one of them, at the listed dimension.
    pub fn check_model(&self, settings: &VectorIndexSettings) -> Result<(), ContextError> {
        if self.supported_models.is_empty() {
            return Ok(());
        }
        let dim = *self
            .supported_models
            .get(&settings.embedding_model)
            .ok_or_else(|| ContextError::UnsupportedEmbeddingModel {
                model: settings.embedding_model.clone(),
                supported: self.supported_models.keys().map(String::as_str).collect::<Vec<_>>().join(", "),
            })?;
        if dim != settings.embedding_dim {
            return Err(ContextError::EmbeddingDimMismatch {
                model: settings.embedding_model.clone(),
                dim: settings.embedding_dim,
                expected: dim as usize,
            });
        }
        Ok(())
    }

    /// The per-call config and partition for `namespace` (`ns_id`) with `settings`,
    /// loading the partition from its stores on first use.
    pub async fn for_namespace(
        &self,
        db: &crate::AsyncDb,
        namespace: &str,
        ns_id: NamespaceId,
        settings: &VectorIndexSettings,
    ) -> Result<NamespaceSemantics, ContextError> {
        let key = (namespace.to_owned(), ns_id);
        let partition = loop {
            if self.is_dropping(namespace) {
                return Err(ContextError::SemanticSearchNotEnabled {
                    namespace: namespace.to_owned(),
                });
            }
            if let Some(p) = self.ivfs.read().get(&key) {
                break Arc::clone(p);
            }
            let drops_begun = self.drops_begun.load(std::sync::atomic::Ordering::SeqCst);
            let ivf = crate::semantic_search::vector_kv::load_ivf(db, namespace, settings.rotation_seed)
                .await?
                .ok_or_else(|| ContextError::VectorIndexNotSeeded {
                    namespace: namespace.to_owned(),
                })?;
            #[cfg(test)]
            load_pause::point(namespace).await;
            // A drop that began during the load cannot see this partition, so
            // it would never be marked dropped, and writers trust that mark. The
            // drop may be over by now, so `is_dropping` cannot tell: start again
            // from what the stores hold now. (Any namespace's drop counts; an
            // unrelated one only costs a second read.)
            let installed = {
                let mut ivfs = self.ivfs.write();
                (self.drops_begun.load(std::sync::atomic::Ordering::SeqCst) == drops_begun).then(|| {
                    // Two first uses can race to load; keep whichever landed first,
                    // so every count update goes to the one partition in use. Nothing
                    // writes this namespace's vectors before its partition is loaded
                    // (every writer goes through here), so the counts are exact.
                    Arc::clone(ivfs.entry(key.clone()).or_insert_with(|| Arc::new(PartitionHandle::new(ivf))))
                })
            };
            if let Some(p) = installed {
                break p;
            }
        };
        partition.set_limits(
            u64::from(settings.maintenance.split_limit()),
            u64::from(settings.maintenance.merge_limit()),
        );
        // Nothing uses a partition whose journal is unfinished: a write could
        // route to postings recovery then retires, and maintenance would start
        // on top of the unfinished operation. A recovery failure is returned,
        // so writes wait (their queue entries stay) rather than go ahead.
        let found = crate::semantic_search::ivf_split::ensure_recovered(db, namespace, &partition, &settings.maintenance.ops()).await?;
        if found > 0 {
            log::info!("partition recovery: namespace='{namespace}' finished {found} interrupted operation(s)");
        }
        let config = SemanticSearchConfig {
            model_name: settings.embedding_model.clone(),
            embedding_dim: settings.embedding_dim as usize,
            window_size: settings.window_size as usize,
            sliding_size: settings.sliding_size as usize,
            number_of_bits_for_dense_quantisation: settings.pass2_bits as usize,
            pass1_bits: settings.pass1_bits as usize,
            probe: probe_settings(&settings.search),
            first_pass_sparse_search_top_k: settings.search.first_pass_top_k as usize,
            top_k_results: settings.search.top_k as usize,
            ..self.config.clone()
        };
        Ok(NamespaceSemantics {
            config,
            partition,
            search: settings.search,
            maintenance: settings.maintenance,
        })
    }
}

impl NamespaceSemantics {
    /// Fetch the query embeddings needed for a two-pass semantic search.
    ///
    /// Returns `(dense, sparse)`: the single whole-query embedding used in Pass 2
    /// dense re-ranking, and Pass 1's query vectors (that same vector: queries are
    /// not chunked).
    /// Uses the system-wide TTL cache when possible, falling back to the embedding
    /// service (then populating the cache) on a miss.
    pub(crate) async fn query_embeddings(&self, db: &crate::AsyncDb, query_text: &str) -> Result<(Vec<f32>, Vec<Vec<f32>>), ContextError> {
        let config = &self.config;
        let ttl = config.query_embedding_cache_ttl;
        // Queries are not chunked: one whole-query vector serves both passes.
        if let Some(dense) =
            crate::semantic_search::vector_kv::get_cached_query_embedding(db, &config.model_name, query_text, config.embedding_dim, ttl).await
        {
            log::debug!("query embedding cache hit");
            return Ok((dense.clone(), vec![dense]));
        }
        log::debug!("query embedding cache miss, calling embedding service");
        let q = crate::semantic_search::service::embed_query(config, query_text)
            .await
            .map_err(|e| ContextError::EmbeddingFailed(e.to_string()))?;
        crate::semantic_search::vector_kv::put_cached_query_embedding(db, &config.model_name, query_text, &q.dense, ttl).await;
        Ok((q.dense, q.sparse))
    }

    /// This namespace's config with one request's search overrides applied
    /// (validated against the same ranges as the schema).
    pub fn with_overrides(mut self, overrides: &SearchSpec) -> Result<Self, ContextError> {
        let s = overrides.apply(self.search)?;
        self.config.probe = probe_settings(&s);
        self.search = s;
        self.config.first_pass_sparse_search_top_k = s.first_pass_top_k as usize;
        self.config.top_k_results = s.top_k as usize;
        Ok(self)
    }
}

/// The probe settings a namespace's search settings describe.
fn probe_settings(s: &SearchSettings) -> crate::semantic_search::ProbeSettings {
    crate::semantic_search::ProbeSettings {
        min_probes: s.min_probes as usize,
        max_probes: s.max_probes as usize,
    }
}

/// Test-only: stop one partition load of a namespace after it has counted the
/// store and before it is installed, so a test can act in between.
#[cfg(test)]
pub(crate) mod load_pause {
    use std::collections::HashMap;
    use std::sync::Arc;
    use tokio::sync::Notify;

    type Pause = (Arc<Notify>, Arc<Notify>);
    static PAUSES: std::sync::LazyLock<parking_lot::Mutex<HashMap<String, Pause>>> =
        std::sync::LazyLock::new(|| parking_lot::Mutex::new(HashMap::new()));

    /// Pause the next load of `namespace`: returns `(reached, release)`. Global to
    /// the test binary, so `namespace` must be a name no other test uses: a load
    /// of the same name in a test running alongside would take this pause.
    pub(crate) fn arm(namespace: &str) -> Pause {
        let pause = (Arc::new(Notify::new()), Arc::new(Notify::new()));
        PAUSES.lock().insert(namespace.to_owned(), pause.clone());
        pause
    }

    pub(crate) async fn point(namespace: &str) {
        let pause = PAUSES.lock().remove(namespace);
        if let Some((reached, release)) = pause {
            reached.notify_one();
            release.notified().await;
        }
    }
}
