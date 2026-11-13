mod config;
mod config_report;
mod error;
mod id;
mod limits;
mod routes;

use std::{collections::HashMap, sync::Arc, time::Instant};

use config::DocStoreApiConfig;
use minnal_db::doc_store::StoredSchema;
use minnal_db::semantic_search::service::SemanticSearchConfig as EmbeddingServiceConfig;
use minnal_db::{AsyncDb, DocStore, DocStoreSchema, KvStoreSchema};
use tracing::{error, info};

/// jemalloc, not the system allocator: under concurrent load glibc malloc keeps
/// much of the memory it frees resident, and searches ran slower with it
/// (`minnal_db/QUICKSTART.md`, "Choosing a memory allocator").
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// Shared application state passed to every handler via axum's `State` extractor.
#[derive(Clone)]
pub struct AppState {
    /// The underlying document store.
    pub store: Arc<DocStore>,
    /// The engine under it (the store's own handle), for the admin routes on the
    /// engine's statistics and maintenance.
    pub db: Arc<AsyncDb>,
    /// Embedding-service settings for semantic search (URL, timeouts, cache
    /// TTL), used to probe a namespace's model when it enables semantic search.
    ///
    /// `None` only where a server is built without semantic search (tests).
    /// Set once at startup and never mutated.
    pub embedding_service: Option<EmbeddingServiceConfig>,
    /// Monotonic timestamp recorded when the server process started.
    ///
    /// Used by the `GET /admin/storage/health` endpoint to report uptime.
    pub started_at: Instant,
    /// Set while a (background) index checkpoint (field-index flush + compaction)
    /// is running, so the on-demand endpoint can reject overlapping runs instead
    /// of stacking expensive flush/compaction passes.
    pub index_checkpoint_running: Arc<std::sync::atomic::AtomicBool>,
}

impl AppState {
    /// Every document store's schema by name, from the store's schema cache
    /// (no I/O; always current, whoever changed a schema).
    pub fn doc_schemas(&self) -> HashMap<String, Arc<DocStoreSchema>> {
        self.store
            .schemas()
            .into_iter()
            .filter_map(|s| match s {
                StoredSchema::Doc(s) => Some((s.namespace.clone(), s)),
                StoredSchema::Kv(_) => None,
            })
            .collect()
    }

    /// Every KV store's schema by name; see [`doc_schemas`](Self::doc_schemas).
    pub fn kv_schemas(&self) -> HashMap<String, Arc<KvStoreSchema>> {
        self.store
            .schemas()
            .into_iter()
            .filter_map(|s| match s {
                StoredSchema::Kv(s) => Some((s.namespace.clone(), s)),
                StoredSchema::Doc(_) => None,
            })
            .collect()
    }
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
        listen     = %cfg.listen_addr(),
        "starting minnal_db_api",
    );

    // Dump the effective configuration (with per-value source) so a deployment can be
    // debugged from the log without guessing which knob actually took effect.
    config_report::log_config_table(&cfg, raw_toml.as_ref());

    let service = cfg.file.semantic_search_config();
    let supported = cfg.file.semantic_search.supported_models();
    info!(
        models = %if supported.is_empty() {
            "any the embedding service serves".to_string()
        } else {
            supported.iter().map(|(m, d)| format!("{m} ({d}-d)")).collect::<Vec<_>>().join(", ")
        },
        "semantic search enabled",
    );

    // The same open as an embedded store's: every worker, the vector worker
    // and the lifecycle governor start here, and the governor finishes
    // whatever a crash left part-way. Each namespace picks its model and
    // grows its own partition; the server only lists the allowed models, if
    // any.
    let store = DocStore::builder(cfg.db_path()).config_file(&cfg.file).open().await?;
    info!(
        "loaded {} store schema(s) from {}",
        store.schemas().len(),
        minnal_db::doc_store::SCHEMA_NAMESPACE
    );

    let state = AppState {
        db: store.db(),
        store,
        embedding_service: Some(service),
        started_at: Instant::now(),
        index_checkpoint_running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
    };

    let shutdown_store = Arc::clone(&state.store);
    let app = routes::router().with_state(state);
    let listener = tokio::net::TcpListener::bind(cfg.listen_addr()).await?;
    info!("listening on http://{}", cfg.listen_addr());

    axum::serve(listener, app).with_graceful_shutdown(shutdown_signal()).await?;

    // Index jobs still running stop with the process; the governor resumes
    // them at the next start, from their records.
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

    info!("shutdown signal received — draining connections (Ctrl-C again to force exit)");

    // The handler stays installed, so without this a second Ctrl-C would be
    // swallowed while a slow shutdown runs. Forcing exit is crash-equivalent:
    // WAL-backed writes replay at the next start, and vector work not yet
    // completed is still queued and is redone by the vector worker.
    tokio::spawn(async {
        if tokio::signal::ctrl_c().await.is_ok() {
            error!("second Ctrl-C — exiting without a clean shutdown");
            std::process::exit(130);
        }
    });
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
