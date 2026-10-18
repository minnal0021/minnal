//! Raw HTTP client for the external embedding service (batch interface).
//!
//! The service no longer chunks text — chunking/tokenisation lives in
//! [`crate::semantic_search::chunking`].  Each call posts a list of already-prepared payload
//! strings to the endpoint for the configured model and gets back one embedding
//! per payload:
//!
//! ```text
//! POST {base}/embedding/{model}/document   {"payloads":[...],"dimensions":D}  ->  {"embeddings":[[f32], ...]}
//! POST {base}/embedding/{model}/query      (same request/response shape)
//! GET  {base}/healthcheck                  -> {"status":..., "models":{"<model>":{"status":"ok"|..., ...}, ...}}
//! ```
//!
//! A service can load several models, so the model is named on every request
//! (`{model}` is e.g. `gemma` or `qwen`); one it does not serve answers 404.
//! A "single" whole-text embedding is just a one-element `payloads` array (a
//! query is always one payload: queries are not chunked); a document passes its
//! whole text plus one payload per sentence-window chunk.

use std::sync::OnceLock;
use std::time::Duration;

use log::{debug, warn};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// How long an idle pooled connection is kept before being dropped.
const POOL_IDLE_TIMEOUT: Duration = Duration::from_secs(90);

static HTTP_CLIENT: OnceLock<reqwest::Client> = OnceLock::new();

fn build_client(connect_timeout: Duration) -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(connect_timeout)
        .pool_idle_timeout(POOL_IDLE_TIMEOUT)
        .build()
        .unwrap_or_else(|e| {
            warn!("failed to build configured HTTP client ({e}); falling back to default (no connect timeout)");
            reqwest::Client::new()
        })
}

/// The shared, connection-pooled HTTP client.
///
/// Built once with the given `connect_timeout` (a client-level setting that caps
/// the TCP connect phase, so an unreachable host fails fast) and a pool-idle
/// timeout. **The connect timeout is bound on this first build and reused for the
/// process lifetime** — that is fine because it comes from a single config value,
/// so every caller passes the same one. The per-call overall *request* timeout is
/// applied separately at each request site (see [`embed`] / [`check_health`]), so
/// a slow or hanging service can never stall an indexing or search call.
fn client(connect_timeout: Duration) -> &'static reqwest::Client {
    HTTP_CLIENT.get_or_init(|| build_client(connect_timeout))
}

// ── Errors ────────────────────────────────────────────────────────────────────

/// Errors that can arise when calling the embedding service.
#[derive(Debug, Error)]
pub enum EmbeddingError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    /// The service answered with a non-2xx status. `detail` is the service's
    /// error message when it sent one (e.g. `Unknown model 'foo'; available:
    /// gemma, qwen` for a model it does not serve), otherwise the raw body.
    #[error("Embedding service returned HTTP {status} for {url}: {detail}")]
    Status { status: u16, url: String, detail: String },

    /// `/healthcheck` lists the models the service loads, and the configured
    /// model is not among them.
    #[error("Embedding service does not serve model '{model}' (available: {available})")]
    ModelNotServed { model: String, available: String },

    /// The service loads the configured model but reports it is not ready
    /// (`loading` while it starts, `unreachable` if its model server is down).
    #[error("Embedding service model '{model}' is not ready (status: {status})")]
    ModelNotReady { model: String, status: String },

    #[error("Embedding service returned an empty response")]
    EmptyResponse,

    #[error("Embedding service returned {got} embeddings for {sent} payloads")]
    CountMismatch { sent: usize, got: usize },

    #[error("Embedding service returned dimension {actual}, expected {expected}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("cluster assignment failed: {0}")]
    Cluster(#[from] crate::semantic_search::cluster::ClusterIndexError),
}

impl EmbeddingError {
    /// `true` when the service answered but rejects this model or dimension
    /// (not served, wrong dimension, or a 4xx), as opposed to a transient
    /// failure (unreachable, timed out, 5xx, model still loading). A request
    /// that would create a namespace for a rejected model should fail; a
    /// transient failure is only worth a warning, since the embed queue retries.
    pub fn is_configuration_error(&self) -> bool {
        match self {
            EmbeddingError::ModelNotServed { .. } | EmbeddingError::DimensionMismatch { .. } => true,
            EmbeddingError::Status { status, .. } => (400..500).contains(status) && *status != 408 && *status != 429,
            _ => false,
        }
    }
}

// ── Request / response types ──────────────────────────────────────────────────

/// Which embedding endpoint a batch is destined for.
///
/// The service exposes separate document and query endpoints because the model
/// embeds the two asymmetrically; this selects the last URL path segment,
/// under `{base_url}/embedding/{model}/`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingTarget {
    /// Documents being indexed → `/embedding/{model}/document`.
    Document,
    /// Search queries → `/embedding/{model}/query`.
    Query,
}

impl EmbeddingTarget {
    /// The path segment under `{base_url}/embedding/{model}/` for this target.
    fn path_segment(self) -> &'static str {
        match self {
            EmbeddingTarget::Document => "document",
            EmbeddingTarget::Query => "query",
        }
    }
}

/// Batch embed request — one embedding is returned per payload string.
#[derive(Serialize)]
struct BatchEmbedRequest<'a> {
    payloads: &'a [String],
    dimensions: usize,
}

// The service may include other keys; serde ignores any not named here.
#[derive(Deserialize)]
struct BatchEmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

/// The parts of `/healthcheck` the client reads. `models` maps each model the
/// service loads to its state; a service that omits it is judged by the HTTP
/// status alone.
#[derive(Deserialize)]
struct HealthResponse {
    #[serde(default)]
    models: Option<std::collections::BTreeMap<String, ModelHealth>>,
}

#[derive(Deserialize)]
struct ModelHealth {
    #[serde(default)]
    status: Option<String>,
}

/// The service's error body (FastAPI's `{"detail": "..."}`).
#[derive(Deserialize)]
struct ErrorBody {
    detail: serde_json::Value,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// POST `payloads` to `model`'s embedding endpoint for `target` and return one
/// vector per payload.
///
/// A model the service does not serve fails with [`EmbeddingError::Status`]
/// (404, carrying the service's list of available models).
///
/// `request_timeout` caps the whole round trip (connect + send + receive) so a slow
/// service cannot stall the caller; `connect_timeout` caps just the TCP connect phase
/// when the shared client is first built (see [`client`]). Returns an empty vector
/// without making a request when `payloads` is empty.
pub async fn embed(
    base_url: &str,
    model: &str,
    target: EmbeddingTarget,
    payloads: &[String],
    dimension: usize,
    request_timeout: Duration,
    connect_timeout: Duration,
) -> Result<Vec<Vec<f32>>, EmbeddingError> {
    let url = format!("{}/embedding/{}/{}", base_url, model, target.path_segment());
    post_embed_batch(&url, payloads, dimension, request_timeout, connect_timeout).await
}

/// GET `{base_url}/healthcheck` and check that `model` is loaded and ready.
///
/// The service answers 503 while *any* of its models is still loading, so the
/// verdict comes from `model`'s own entry in the `models` map rather than the
/// HTTP status: [`EmbeddingError::ModelNotServed`] if it is absent,
/// [`EmbeddingError::ModelNotReady`] if its status is not `ok`. A response
/// without a `models` map must simply be 2xx.
///
/// `request_timeout` caps the whole round trip so the startup probe cannot hang;
/// `connect_timeout` caps the TCP connect phase (bound at first client build).
pub async fn check_health(base_url: &str, model: &str, request_timeout: Duration, connect_timeout: Duration) -> Result<(), EmbeddingError> {
    let url = format!("{}/healthcheck", base_url);
    debug!("embedding service health check url={}", url);
    let response = client(connect_timeout).get(&url).timeout(request_timeout).send().await.map_err(|e| {
        warn!("embedding service health check failed: {}", e);
        e
    })?;
    let status = response.status();
    let body = response.text().await?;
    let models = serde_json::from_str::<HealthResponse>(&body).ok().and_then(|h| h.models);
    let Some(models) = models else {
        if status.is_success() {
            return Ok(());
        }
        warn!("embedding service returned non-2xx status: {}", status);
        return Err(status_error(status, url, &body));
    };
    match models.get(model) {
        None => Err(EmbeddingError::ModelNotServed {
            model: model.to_string(),
            available: models.keys().cloned().collect::<Vec<_>>().join(", "),
        }),
        Some(m) if m.status.as_deref() == Some("ok") => Ok(()),
        Some(m) => Err(EmbeddingError::ModelNotReady {
            model: model.to_string(),
            status: m.status.clone().unwrap_or_else(|| "unknown".to_string()),
        }),
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

async fn post_embed_batch(
    url: &str,
    payloads: &[String],
    dimension: usize,
    request_timeout: Duration,
    connect_timeout: Duration,
) -> Result<Vec<Vec<f32>>, EmbeddingError> {
    if payloads.is_empty() {
        return Ok(Vec::new());
    }
    debug!("embedding batch url={} payloads={} dim={}", url, payloads.len(), dimension);
    let response = client(connect_timeout)
        .post(url)
        .timeout(request_timeout)
        .json(&BatchEmbedRequest {
            payloads,
            dimensions: dimension,
        })
        .send()
        .await?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(status_error(status, url.to_string(), &body));
    }
    let BatchEmbedResponse { embeddings } = response.json().await?;
    if embeddings.is_empty() {
        return Err(EmbeddingError::EmptyResponse);
    }
    if embeddings.len() != payloads.len() {
        return Err(EmbeddingError::CountMismatch {
            sent: payloads.len(),
            got: embeddings.len(),
        });
    }
    for emb in &embeddings {
        if emb.len() != dimension {
            return Err(EmbeddingError::DimensionMismatch {
                expected: dimension,
                actual: emb.len(),
            });
        }
    }
    Ok(embeddings)
}

/// Build [`EmbeddingError::Status`], preferring the service's `detail` message
/// over the raw body.
fn status_error(status: reqwest::StatusCode, url: String, body: &str) -> EmbeddingError {
    let detail = match serde_json::from_str::<ErrorBody>(body) {
        Ok(ErrorBody {
            detail: serde_json::Value::String(msg),
        }) => msg,
        Ok(ErrorBody { detail }) => detail.to_string(),
        Err(_) => body.chars().take(500).collect(),
    };
    EmbeddingError::Status {
        status: status.as_u16(),
        url,
        detail,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::time::Instant;

    /// A server that accepts connections but never replies must not stall the
    /// caller: the per-request timeout must fire and surface an error. Uses a
    /// plain `std::net::TcpListener` on a background OS thread (no extra deps) —
    /// the TCP connect succeeds, the HTTP response never comes.
    #[tokio::test]
    async fn embed_times_out_when_server_never_responds() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("local_addr");
        // Hold accepted sockets open without ever writing a response.
        std::thread::spawn(move || {
            let mut held = Vec::new();
            for stream in listener.incoming() {
                match stream {
                    Ok(s) => held.push(s), // keep the socket alive, send nothing
                    Err(_) => break,
                }
            }
        });

        let base = format!("http://{addr}");
        let timeout = Duration::from_millis(300);
        let start = Instant::now();
        let result = embed(
            &base,
            "gemma",
            EmbeddingTarget::Document,
            &["hello".to_string()],
            8,
            timeout,
            Duration::from_secs(5),
        )
        .await;
        let elapsed = start.elapsed();

        assert!(result.is_err(), "a non-responding server must yield an error, not hang");
        assert!(
            elapsed < Duration::from_secs(5),
            "request must time out promptly (~{timeout:?}), took {elapsed:?}",
        );
    }

    /// An empty payload list short-circuits without any network call, so even a
    /// dead address returns `Ok(vec![])` immediately regardless of the timeout.
    #[tokio::test]
    async fn embed_empty_payloads_makes_no_request() {
        let result = embed(
            "http://127.0.0.1:1", // unroutable; must never be contacted
            "gemma",
            EmbeddingTarget::Document,
            &[],
            8,
            Duration::from_millis(1),
            Duration::from_millis(1),
        )
        .await;
        assert!(matches!(result, Ok(v) if v.is_empty()));
    }
}
