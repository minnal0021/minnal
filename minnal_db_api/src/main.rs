mod config;
mod config_report;
mod error;
mod id;
mod limits;
mod routes;

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Instant,
};

use config::DocStoreApiConfig;
use minnal_db::semantic_search::service::SemanticSearchConfig as EmbeddingServiceConfig;
use minnal_db::{DocStore, DocStoreSchema, IndexBuildManager, KvStoreSchema, SemanticSearchContext};
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
    /// [`KeyType`]: minnal_db::KeyType
    pub schemas: Arc<RwLock<HashMap<String, DocStoreSchema>>>,
    /// In-memory schema cache for KV stores, keyed by namespace name.
    pub kv_schemas: Arc<RwLock<HashMap<String, KvStoreSchema>>>,
    /// Registry of active background index-build tasks.
    ///
    /// Used to:
    ///   - await every in-progress build on graceful shutdown.
    ///   - serve live progress snapshots via the progress endpoints.
    pub index_manager: Arc<IndexBuildManager>,
    /// Embedding-service settings for semantic search (URL, timeouts, cache
    /// TTL), used to probe a namespace's model when it enables semantic search.
    ///
    /// `None` when no model's centroids loaded at startup, so semantic search
    /// is unavailable. Set once at startup and never mutated.
    pub embedding_service: Option<EmbeddingServiceConfig>,
    /// Monotonic timestamp recorded when the server process started.
    ///
    /// Used by the `GET /admin/storage/health` endpoint to report uptime.
    pub started_at: Instant,
    /// Tracks namespaces with an active exclusive attribute-index operation
    /// (drop-all, reindex-all, or single-field cleanup).
    pub attr_index_ops: Arc<parking_lot::Mutex<HashSet<String>>>,
    /// Tracks namespaces whose vector index is currently being dropped (background cleanup).
    pub vec_index_cleanup: Arc<parking_lot::Mutex<HashSet<String>>>,
    /// Set while a (background) vector-index reconcile/validate pass is running, so
    /// the on-demand endpoint can reject overlapping runs instead of stacking
    /// expensive full scans.
    pub vec_reconcile_running: Arc<std::sync::atomic::AtomicBool>,
    /// Set while a (background) index checkpoint (field-index flush + compaction)
    /// is running, so the on-demand endpoint can reject overlapping runs instead
    /// of stacking expensive flush/compaction passes.
    pub index_checkpoint_running: Arc<std::sync::atomic::AtomicBool>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (cfg, raw_toml) = load_config();

    let log_dir = cfg.log_dir();
    std::fs::create_dir_all(&log_dir)?;
    let file_appender = tracing_appender::rolling::daily(&log_dir, "minnal_db_api.log");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

    // `minnal_db` logs exclusively through the `log` facade (it has no direct
    // `tracing` dependency). Those records only reach the subscriber below via
    // the `log` → `tracing` bridge that `.init()` installs — which exists only
    // because `tracing-subscriber`'s default `tracing-log` feature is enabled.
    // Do NOT disable that feature (e.g. via `default-features = false`) without
    // installing the bridge another way, or all engine logs go silent.
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
        "starting minnal_db_api",
    );

    // Dump the effective configuration (with per-value source) so a deployment can be
    // debugged from the log without guessing which knob actually took effect.
    config_report::log_config_table(&cfg, raw_toml.as_ref());

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

    let semantic_cfg = cfg.semantic_search.resolve();
    let (cluster_indexes, problems) = semantic_cfg.load_cluster_indexes();
    for problem in &problems {
        warn!("centroid set not loaded: {problem}");
    }
    let embedding_service = (!cluster_indexes.is_empty()).then(|| semantic_cfg.service_config());

    // With at least one model's centroids loaded, attach a SemanticSearchContext
    // so put/delete on semantic-search-enabled namespaces maintain their vector
    // index. Each namespace picks its model; its centroids come from here.
    let store = if let Some(service) = embedding_service.clone() {
        info!(
            dir = %semantic_cfg.centroid_dir.display(),
            models = %cluster_indexes
                .iter()
                .map(|(m, i)| format!("{m} ({} clusters, {}-d)", i.len(), i.dim()))
                .collect::<Vec<_>>()
                .join(", "),
            "loaded cluster centroids for semantic search",
        );

        // Probe the embedding service for each (model, dimension) an existing
        // semantic namespace uses, so operators get an early warning if it is
        // unreachable or misconfigured. Non-fatal: semantic requests surface the
        // error at call time.
        let mut in_use: std::collections::BTreeSet<(String, u32)> = std::collections::BTreeSet::new();
        for settings in schemas
            .values()
            .filter(|s| s.semantic_search_enabled)
            .filter_map(|s| s.vector_settings().ok())
            .chain(
                kv_schemas
                    .values()
                    .filter(|s| s.is_semantic_search_enabled())
                    .filter_map(|s| s.vector_settings().ok()),
            )
        {
            in_use.insert((settings.embedding_model, settings.embedding_dim));
        }
        for (model, dim) in in_use {
            let probe = EmbeddingServiceConfig {
                model_name: model.clone(),
                embedding_dim: dim as usize,
                ..service.clone()
            };
            match minnal_db::semantic_search::service::check_embedding_service(&probe).await {
                Ok(()) => info!(url = %probe.embedding_service_url, model = %model, dim, "embedding service reachable"),
                Err(e) => error!(
                    url = %probe.embedding_service_url, model = %model, dim,
                    "embedding service check failed — semantic search for this model will be unavailable: {e}",
                ),
            }
        }

        store
            .with_vector_index_config(cfg.to_vector_index_config())
            .with_semantic_search(SemanticSearchContext::new(service, cluster_indexes))
    } else {
        warn!(
            dir = %semantic_cfg.centroid_dir.display(),
            "no cluster centroids loaded — semantic search unavailable",
        );
        store
    };

    let state = AppState {
        store: Arc::new(store),
        schemas: Arc::new(RwLock::new(schemas)),
        kv_schemas: Arc::new(RwLock::new(kv_schemas)),
        index_manager: Arc::clone(&index_manager),
        embedding_service,
        started_at: Instant::now(),
        attr_index_ops: Arc::new(parking_lot::Mutex::new(HashSet::new())),
        vec_index_cleanup: Arc::new(parking_lot::Mutex::new(HashSet::new())),
        vec_reconcile_running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        index_checkpoint_running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
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
/// 1. First positional CLI argument: `minnal_db_api /path/to/config.toml`
/// 2. `MINNAL_CONFIG_FILE` environment variable
/// 3. All built-in defaults (no file required)
///
/// Returns the resolved config plus the file parsed as a plain [`toml::Table`] (the
/// second element is `None` when no file was given, or if the raw re-parse fails),
/// used by [`config_report`] to report each value's source.
fn load_config() -> (DocStoreApiConfig, Option<toml::Table>) {
    let config_path = std::env::args().nth(1).or_else(|| std::env::var("MINNAL_CONFIG_FILE").ok());

    match config_path {
        Some(path) => {
            let p = std::path::Path::new(&path);
            match DocStoreApiConfig::from_file(p) {
                Ok(cfg) => {
                    info!("loaded config from '{path}'");
                    // Re-parse the raw file to a plain table so the startup report can
                    // tell which keys were actually set vs left at their default.
                    let raw = std::fs::read_to_string(p).ok().and_then(|s| toml::from_str::<toml::Table>(&s).ok());
                    (cfg, raw)
                }
                Err(e) => {
                    eprintln!("error: {e}");
                    std::process::exit(1);
                }
            }
        }
        None => {
            info!("no config file specified — using built-in defaults");
            (DocStoreApiConfig::default(), None)
        }
    }
}
