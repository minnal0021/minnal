//! Index jobs and the lifecycle audit log:
//!
//! ```text
//! GET    /admin/jobs                    → every store's live jobs
//! GET    /admin/stores/{ns}/jobs        → one store's jobs, live and ended
//! DELETE /admin/stores/{ns}/jobs/{job}  → cancel a live job (202)
//! GET    /admin/lifecycle/audit         → lifecycle events, newest first (?namespace=&limit=&cursor=)
//! ```
//!
//! A job is named by its kind, and a field job by its field too:
//! `field_build:status`, `reindex_all`, `vector_cleanup`.

use axum::{
    Json,
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
};
use minnal_db::JobRecord;
use serde::Deserialize;
use tracing::info;

use crate::{AppState, error::AppError};

/// A job as the routes show it.
fn job_json(name: &str, record: &JobRecord) -> serde_json::Value {
    serde_json::json!({
        "job": name,
        "status": record.status,
        "subject": record.subject,
        "started_at_ms": record.started_at_ms,
        "updated_at_ms": record.updated_at_ms,
        "error": record.error,
        "progress": record.progress,
    })
}

/// `GET /admin/jobs` — every store's live jobs.
pub async fn live_jobs(State(state): State<AppState>) -> impl IntoResponse {
    let jobs: Vec<serde_json::Value> = state
        .store
        .live_jobs()
        .iter()
        .map(|(ns, name, record)| {
            let mut job = job_json(name, record);
            job["namespace"] = serde_json::Value::String(ns.clone());
            job
        })
        .collect();
    Json(jobs)
}

/// `GET /admin/stores/{ns}/jobs` — `{ns}`'s jobs with a record, live and
/// ended (each kind's latest run), and the store's state.
pub async fn store_jobs(State(state): State<AppState>, Path(ns): Path<String>) -> Result<impl IntoResponse, AppError> {
    let jobs = state.store.jobs(&ns).map_err(|e| AppError::from(e).with_ns(&ns))?;
    Ok(Json(serde_json::json!({
        "namespace": ns,
        "state": state.store.store_state(&ns).map(|s| s.as_str()),
        "jobs": jobs.iter().map(|(name, record)| job_json(name, record)).collect::<Vec<_>>(),
    })))
}

/// `DELETE /admin/stores/{ns}/jobs/{job}` — ask a live job to stop. It stops
/// at its next batch and undoes its work: `202 Accepted`. `404` for no live
/// job of that name, `409` for one that cannot be stopped part-way
/// (`field_drop`, `reindex_all`, `vector_cleanup`).
pub async fn cancel_job(State(state): State<AppState>, Path((ns, job)): Path<(String, String)>) -> Result<impl IntoResponse, AppError> {
    state.store.cancel_job(&ns, &job).map_err(|e| AppError::from(e).with_ns(&ns))?;
    info!(namespace = %ns, job = %job, "job cancel requested");
    Ok((
        StatusCode::ACCEPTED,
        Json(serde_json::json!({"namespace": ns, "job": job, "status": "cancelling"})),
    ))
}

/// Query of `GET /admin/lifecycle/audit`.
#[derive(Debug, Deserialize)]
pub struct AuditParams {
    /// Only this store's events (a dropped store's too).
    pub namespace: Option<String>,
    /// Records per page (default 100, at most 1,000).
    pub limit: Option<usize>,
    /// The `next_cursor` of the page before.
    pub cursor: Option<String>,
}

/// `GET /admin/lifecycle/audit` — the lifecycle audit log, newest first:
/// every store create and drop, every index job's start, cancel and end, and
/// every task the governor gave up on, kept for `[lifecycle]
/// audit_retention_days`. `{results, next_cursor}`; `next_cursor` is `null`
/// at the end.
pub async fn lifecycle_audit(State(state): State<AppState>, Query(params): Query<AuditParams>) -> Result<impl IntoResponse, AppError> {
    let cursor = params.cursor.as_deref().map(crate::routes::decode_cursor).transpose()?;
    let (results, next) = state
        .store
        .lifecycle_audit(params.namespace.as_deref(), cursor, params.limit.unwrap_or(100))
        .await?;
    Ok(Json(serde_json::json!({
        "results": results,
        "next_cursor": next.as_deref().map(crate::routes::encode_cursor),
    })))
}
