mod config;
mod error;
mod id;
mod routes;

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Instant,
};

use config::DocStoreApiConfig;
use minnal_doc_store::{DocStore, DocStoreSchema, IndexBuildManager, KvStoreSchema, SemanticSearchContext};
use semantic_search::ClusterIndex;
use semantic_search::service::SemanticSearchConfig as EmbeddingServiceConfig;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

/// Shared application state passed to every handler via axum's `State` extractor.
#[derive(Clone)]
pub struct AppState {
    /// The underlying document store.
    pub store: Arc<DocStore>,
    /// In-memory schema cache for doc stores, keyed by namespace name.
    ///
    /// Populated at startup and kept in sync with the store after each mutation.
    /// Handlers use this to resolve the [`KeyType`] needed to parse `{id}` path
    /// segments without re-reading from disk on every request.
    ///
    /// [`KeyType`]: minnal_doc_store::KeyType
    pub schemas: Arc<RwLock<HashMap<String, DocStoreSchema>>>,
    /// In-memory schema cache for KV stores, keyed by namespace name.
    pub kv_schemas: Arc<RwLock<HashMap<String, KvStoreSchema>>>,
    /// Registry of active background index-build tasks.
    ///
    /// Used to:
    ///   - await every in-progress build on graceful shutdown.
    ///   - serve live progress snapshots via the progress endpoints.
    pub index_manager: Arc<IndexBuildManager>,
    /// Pre-built IVF cluster index for semantic search.
    ///
    /// `None` when the cluster file does not exist yet (semantic search
    /// unavailable until the index is built).  Once set at startup it is
    /// never mutated, so no lock is needed.
    pub cluster_index: Option<Arc<ClusterIndex>>,
    /// Monotonic timestamp recorded when the server process started.
    ///
    /// Used by the `GET /admin/storage/health` endpoint to report uptime.
    pub started_at: Instant,
    /// Tracks namespaces with an active exclusive attribute-index operation
    /// (drop-all, reindex-all, or single-field cleanup).
    pub attr_index_ops: Arc<std::sync::Mutex<HashSet<String>>>,
    /// Tracks namespaces whose vector index is currently being dropped (background cleanup).
    pub vec_index_cleanup: Arc<std::sync::Mutex<HashSet<String>>>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = load_config();

    let log_dir = cfg.log_dir();
    std::fs::create_dir_all(&log_dir)?;
    let file_appender = tracing_appender::rolling::daily(&log_dir, "minnal_doc_store_api.log");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

    use tracing_subscriber::prelude::*;
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_subscriber::fmt::layer().with_writer(non_blocking).with_ansi(false))
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| cfg.log_level().parse().unwrap_or_else(|_| "info".parse().unwrap())),
        )
        .init();

    info!(
        db_path    = %cfg.db_path().display(),
        schema_dir = %cfg.schema_dir().display(),
        listen     = %cfg.listen_addr(),
        "starting minnal_doc_store_api",
    );

    let db_config = cfg.to_db_config();
    let store = DocStore::open_with_config(cfg.db_path(), cfg.schema_dir(), db_config).await?;

    let schemas: HashMap<String, DocStoreSchema> = store
        .list()?
        .into_iter()
        .filter_map(|v| serde_json::from_value::<DocStoreSchema>(v).ok())
        .map(|s| (s.namespace.clone(), s))
        .collect();

    let kv_schemas: HashMap<String, KvStoreSchema> = store
        .list_kv()?
        .into_iter()
        .filter_map(|v| serde_json::from_value::<KvStoreSchema>(v).ok())
        .map(|s| (s.namespace.clone(), s))
        .collect();

    info!("loaded {} doc schema(s) and {} KV schema(s) into cache", schemas.len(), kv_schemas.len());

    let index_manager = Arc::new(IndexBuildManager::new());

    // Resume any index builds interrupted by a previous shutdown.
    match store.resume_pending_builds().await {
        Ok(handles) => {
            if !handles.is_empty() {
                info!("resuming {} interrupted index build(s)", handles.len());
                for h in handles {
                    index_manager.insert_field_build(h);
                }
            }
        }
        Err(e) => error!("failed to resume pending index builds: {e}"),
    }

    let semantic_cfg = cfg.semantic_search.resolve(cfg.db_path().as_path());
    let cluster_index_opt = {
        let path = semantic_cfg.cluster_path.to_string_lossy();
        info!(path = %path, "loading cluster index");
        match ClusterIndex::load_with_dim(&path, semantic_cfg.n_probes, semantic_cfg.embedding_dim) {
            Ok(idx) => {
                info!(
                    clusters = idx.clusters.len(),
                    n_probes = semantic_cfg.n_probes,
                    "loaded cluster index for semantic search"
                );
                Some(Arc::new(idx))
            }
            Err(e) => {
                warn!(
                    path = %path,
                    "cluster index not loaded — semantic search unavailable: {e}",
                );
                None
            }
        }
    };

    // If the cluster index loaded successfully, attach a SemanticSearchContext
    // to the store so that put/delete on semantic-search-enabled namespaces
    // automatically maintain the companion vector KV store.
    let store = if let Some(cluster_index) = cluster_index_opt.clone() {
        let embedding_cfg = EmbeddingServiceConfig {
            embedding_service_url: semantic_cfg.embedding_service_url.clone(),
            model_name: semantic_cfg.model_name.clone(),
            embedding_dim: semantic_cfg.embedding_dim,
            top_k_results: semantic_cfg.top_k_results,
            number_of_bits_for_dense_quantisation: semantic_cfg.number_of_bits_for_dense_quantisation,
            n_probes: semantic_cfg.n_probes,
            window_size: semantic_cfg.window_size,
            sliding_size: semantic_cfg.sliding_size,
            first_pass_sparse_search_top_k: semantic_cfg.first_pass_sparse_search_top_k,
            query_embedding_cache_ttl: semantic_cfg.query_embedding_cache_ttl,
            embedding_request_timeout: semantic_cfg.embedding_request_timeout,
            embedding_connect_timeout: semantic_cfg.embedding_connect_timeout,
        };

        // Probe the embedding service so operators get an early warning if it
        // is unreachable or misconfigured.  Failure is non-fatal: the server
        // starts anyway and semantic search requests will surface the error at
        // call time.
        match semantic_search::service::check_embedding_service(&embedding_cfg).await {
            Ok(()) => info!(
                url = %embedding_cfg.embedding_service_url,
                dim = embedding_cfg.embedding_dim,
                "embedding service reachable",
            ),
            Err(e) => error!(
                url = %embedding_cfg.embedding_service_url,
                "embedding service health check failed — semantic search will be unavailable: {e}",
            ),
        }

        store
            .with_vector_index_config(cfg.to_vector_index_config())
            .with_semantic_search(SemanticSearchContext {
                config: embedding_cfg,
                cluster_index,
            })
    } else {
        store
    };

    let state = AppState {
        store: Arc::new(store),
        schemas: Arc::new(RwLock::new(schemas)),
        kv_schemas: Arc::new(RwLock::new(kv_schemas)),
        index_manager: Arc::clone(&index_manager),
        cluster_index: cluster_index_opt,
        started_at: Instant::now(),
        attr_index_ops: Arc::new(std::sync::Mutex::new(HashSet::new())),
        vec_index_cleanup: Arc::new(std::sync::Mutex::new(HashSet::new())),
    };

    let shutdown_store = Arc::clone(&state.store);
    let app = routes::router().with_state(state);
    let listener = tokio::net::TcpListener::bind(cfg.listen_addr()).await?;
    info!("listening on http://{}", cfg.listen_addr());

    axum::serve(listener, app).with_graceful_shutdown(shutdown_signal()).await?;

    // Drain and await all in-progress index builds before exiting.
    index_manager.drain_all().await;

    // Stop all background workers (vec-index, GC, WAL GC, LSM compaction,
    // index checkpoint, TTL) and flush all in-memory state to disk.
    if let Err(e) = shutdown_store.shutdown().await {
        error!("error during store shutdown: {e}");
    }

    info!("shutdown complete");
    Ok(())
}

/// Resolves on SIGINT (Ctrl-C) or SIGTERM.
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c().await.expect("failed to install Ctrl-C handler");
    };

    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }

    info!("shutdown signal received — draining connections");
}

/// Resolve and load configuration.
///
/// Priority:
/// 1. First positional CLI argument: `minnal_doc_store_api /path/to/config.toml`
/// 2. `MINNAL_CONFIG_FILE` environment variable
/// 3. All built-in defaults (no file required)
fn load_config() -> DocStoreApiConfig {
    let config_path = std::env::args().nth(1).or_else(|| std::env::var("MINNAL_CONFIG_FILE").ok());

    match config_path {
        Some(path) => {
            let p = std::path::Path::new(&path);
            match DocStoreApiConfig::from_file(p) {
                Ok(cfg) => {
                    info!("loaded config from '{path}'");
                    cfg
                }
                Err(e) => {
                    eprintln!("error: {e}");
                    std::process::exit(1);
                }
            }
        }
        None => {
            info!("no config file specified — using built-in defaults");
            DocStoreApiConfig::default()
        }
    }
}
