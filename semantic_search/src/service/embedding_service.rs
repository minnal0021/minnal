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

use log::{debug, warn};
use serde::{Deserialize, Serialize};
use thiserror::Error;

static HTTP_CLIENT: OnceLock<reqwest::Client> = OnceLock::new();

fn client() -> &'static reqwest::Client {
    HTTP_CLIENT.get_or_init(reqwest::Client::new)
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
/// Returns an empty vector without making a request when `payloads` is empty.
pub async fn embed(base_url: &str, target: EmbeddingTarget, payloads: &[String], dimension: usize) -> Result<Vec<Vec<f32>>, EmbeddingError> {
    let url = format!("{}/embedding/{}", base_url, target.path_segment());
    post_embed_batch(&url, payloads, dimension).await
}

/// GET `{base_url}/healthcheck` and return `Ok(())` on a successful response.
pub async fn check_health(base_url: &str) -> Result<(), EmbeddingError> {
    let url = format!("{}/healthcheck", base_url);
    debug!("embedding service health check url={}", url);
    client()
        .get(&url)
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

async fn post_embed_batch(url: &str, payloads: &[String], dimension: usize) -> Result<Vec<Vec<f32>>, EmbeddingError> {
    if payloads.is_empty() {
        return Ok(Vec::new());
    }
    debug!("embedding batch url={} payloads={} dim={}", url, payloads.len(), dimension);
    let response = client()
        .post(url)
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
