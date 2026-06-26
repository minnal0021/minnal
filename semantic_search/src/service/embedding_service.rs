//! Raw HTTP client for the external embedding service (batch interface).
//!
//! The service no longer chunks text — chunking/tokenisation lives in
//! [`crate::chunking`].  Each call posts a list of already-prepared payload
//! strings and gets back one embedding per payload:
//!
//! ```text
//! POST {base}/embedding/document   {"payloads":[...],"dimensions":D}  ->  {"embeddings":[[f32], ...]}
//! POST {base}/embedding/query      (same request/response shape)
//! ```
//!
//! A "single" whole-text embedding is just a one-element `payloads` array;
//! chunked embeddings pass one payload per sliding-window chunk.  The old
//! `{model}` path segment is gone (the model is fixed server-side).

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

    #[error("Embedding service returned an empty response")]
    EmptyResponse,

    #[error("Embedding service returned {got} embeddings for {sent} payloads")]
    CountMismatch { sent: usize, got: usize },

    #[error("Embedding service returned dimension {actual}, expected {expected}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("cluster assignment failed: {0}")]
    Cluster(#[from] crate::cluster::ClusterIndexError),
}

// ── Request / response types ──────────────────────────────────────────────────

/// Which embedding endpoint a batch is destined for.
///
/// The service exposes separate document and query endpoints because the model
/// embeds the two asymmetrically; this selects the URL path segment under
/// `{base_url}/embedding/`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingTarget {
    /// Documents being indexed → `/embedding/document`.
    Document,
    /// Search queries → `/embedding/query`.
    Query,
}

impl EmbeddingTarget {
    /// The path segment under `{base_url}/embedding/` for this target.
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

// ── Public API ────────────────────────────────────────────────────────────────

/// POST `payloads` to the embedding endpoint for `target` and return one vector per payload.
///
/// `request_timeout` caps the whole round trip (connect + send + receive) so a slow
/// service cannot stall the caller; `connect_timeout` caps just the TCP connect phase
/// when the shared client is first built (see [`client`]). Returns an empty vector
/// without making a request when `payloads` is empty.
pub async fn embed(
    base_url: &str,
    target: EmbeddingTarget,
    payloads: &[String],
    dimension: usize,
    request_timeout: Duration,
    connect_timeout: Duration,
) -> Result<Vec<Vec<f32>>, EmbeddingError> {
    let url = format!("{}/embedding/{}", base_url, target.path_segment());
    post_embed_batch(&url, payloads, dimension, request_timeout, connect_timeout).await
}

/// GET `{base_url}/healthcheck` and return `Ok(())` on a successful response.
///
/// `request_timeout` caps the whole round trip so the startup probe cannot hang;
/// `connect_timeout` caps the TCP connect phase (bound at first client build).
pub async fn check_health(base_url: &str, request_timeout: Duration, connect_timeout: Duration) -> Result<(), EmbeddingError> {
    let url = format!("{}/healthcheck", base_url);
    debug!("embedding service health check url={}", url);
    client(connect_timeout)
        .get(&url)
        .timeout(request_timeout)
        .send()
        .await
        .map_err(|e| {
            warn!("embedding service health check failed: {}", e);
            e
        })?
        .error_for_status()
        .map_err(|e| {
            warn!("embedding service returned non-2xx status: {}", e);
            e
        })?;
    Ok(())
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
        .await?
        .error_for_status()?;
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
