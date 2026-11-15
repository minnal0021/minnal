//! A vector index's partition: health, server settings, recall checks, probe traces, history, rebuilds, postings and centres.

use axum::{
    Json,
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
};
use serde::Deserialize;

use crate::AppState;

// ── GET /admin/indices/{ns}/vector/partition and /postings ───────────────────

/// Query of `GET /admin/indices/{ns}/vector/partition`.
#[derive(Debug, Deserialize)]
pub struct PartitionParams {
    /// Also compute the code metrics (reads every chunk code of the namespace).
    #[serde(default)]
    pub codes: bool,
}

fn maintenance_json(m: &minnal_db::doc_store::schema::vector_settings::MaintenanceSettings) -> serde_json::Value {
    serde_json::json!({
        "target_posting_size": m.target_posting_size,
        "split_limit": m.split_limit(),
        "merge_limit": m.merge_limit(),
    })
}

/// `GET /admin/indices/{ns}/vector/partition[?codes=true]` — the namespace's
/// partition at a glance: postings (K), entries, chunks, size percentiles,
/// splits, unfinished maintenance operations, and maintenance counters (splits,
/// merges, reassignments, chunk moves) since the partition loaded; with
/// `codes=true` also the entry-weighted code
/// metrics and the postings with the highest centre drift (re-encode
/// candidates). Without `codes` it reads memory only.
pub async fn vector_partition(
    State(state): State<AppState>,
    Path(ns): Path<String>,
    Query(params): Query<PartitionParams>,
) -> axum::response::Response {
    let (maintenance, summary, _) = match state.store.vector_partition_health(&ns, params.codes).await {
        Ok(r) => r,
        Err(e) => return crate::routes::AppError::from(e).with_ns(&ns).into_response(),
    };
    // The last recall check, from the history (never run here: it is costly).
    let last_check = state
        .store
        .vector_partition_history(&ns, Some(minnal_db::semantic_search::ivf_split::history::RecordKind::RecallCheck), 1)
        .await
        .ok()
        .and_then(|mut r| r.pop());
    let mut body = serde_json::to_value(&summary).unwrap_or_default();
    let rebuild = state.store.vector_rebuild_status(&ns);
    if let (Some(obj), serde_json::Value::Object(m)) = (body.as_object_mut(), maintenance_json(&maintenance)) {
        obj.insert("namespace".into(), serde_json::Value::String(ns));
        obj.extend(m);
        obj.insert("rebuild".into(), serde_json::to_value(rebuild).unwrap_or_default());
        obj.insert("last_recall_check".into(), serde_json::to_value(last_check).unwrap_or_default());
    }
    Json(body).into_response()
}

/// `GET /admin/indices/vector/settings` — the server's semantic-search
/// settings (`[semantic_search]` in its config): embedding service, allowed
/// models, TTLs, maintenance and rebuild threads, the recall check's limits, and
/// the defaults new namespaces' vector-index settings are filled with. `422`
/// when semantic search is not configured.
pub async fn vector_server_settings(State(state): State<AppState>) -> axum::response::Response {
    match state.store.vector_server_settings() {
        Some(v) => Json(v).into_response(),
        None => (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({"error": "semantic search is not configured on this server"})),
        )
            .into_response(),
    }
}

/// Query of `GET /admin/indices/{ns}/vector/recall-check`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecallCheckParams {
    /// Queries to sample: default 100, at most 1,000.
    #[serde(default)]
    pub sample: Option<usize>,
    /// Sample seed; the same seed (and query source) samples the same queries
    /// before and after a rebuild.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Overlap@10 below which the verdict is `below_target` (default 0.95).
    #[serde(default)]
    pub min_overlap: Option<f64>,
    /// Override of the namespace's `min_probes`.
    #[serde(default)]
    pub min_probes: Option<u32>,
    /// Override of the namespace's `max_probes`: check what probing another
    /// number of postings would find (validated like a search's).
    #[serde(default)]
    pub max_probes: Option<u32>,
}

/// `GET /admin/indices/{ns}/vector/recall-check` — does the namespace's search
/// miss what a full read finds? Samples real queries from the
/// query-embedding cache (this namespace's model and dimension); with fewer than
/// 20 cached it samples stored chunks instead (their 1-bit reconstructions).
/// Neither makes an embedding call; `query_source` in the result says which was
/// used. Each query is searched under the namespace's settings and again probing
/// every posting (the nearest postings up to 500,000 entries on a larger
/// namespace), and the result gives the overlap at @10–@100, the share read, the
/// cost, and a verdict: `read_in_full` (a rebuild cannot help), `ok`,
/// `below_target`, or `empty`. Synchronous; it costs `sample` searches plus
/// `sample` reference reads. The result is recorded in the partition history.
pub async fn vector_recall_check(
    State(state): State<AppState>,
    Path(ns): Path<String>,
    Query(params): Query<RecallCheckParams>,
) -> axum::response::Response {
    use minnal_db::semantic_search::ivf_split::recall_check as rc;
    let bad = |msg: String| (StatusCode::BAD_REQUEST, Json(serde_json::json!({ "error": msg }))).into_response();
    let defaults = rc::RecallCheckParams::default();
    let sample = params.sample.unwrap_or(defaults.sample);
    if !(1..=rc::MAX_SAMPLE).contains(&sample) {
        return bad(format!("sample must be between 1 and {}, got {sample}", rc::MAX_SAMPLE));
    }
    let min_overlap = params.min_overlap.unwrap_or(defaults.min_overlap);
    if !(0.0..=1.0).contains(&min_overlap) {
        return bad(format!("min_overlap must be between 0 and 1, got {min_overlap}"));
    }
    let check = rc::RecallCheckParams {
        sample,
        seed: params.seed.unwrap_or(defaults.seed),
        min_overlap,
    };
    let overrides = crate::routes::semantic_search::search_overrides(
        None,
        crate::routes::semantic_search::ProbeOverrides {
            min_probes: params.min_probes,
            max_probes: params.max_probes,
        },
        None,
    );
    match state.store.vector_recall_check(&ns, &check, &overrides).await {
        Ok(r) => {
            let mut body = serde_json::to_value(&r).unwrap_or_default();
            if let Some(obj) = body.as_object_mut() {
                obj.insert("namespace".into(), serde_json::Value::String(ns));
            }
            Json(body).into_response()
        }
        Err(e) => crate::routes::AppError::from(e).with_ns(&ns).into_response(),
    }
}

/// Query of `GET /admin/indices/{ns}/vector/probe-trace`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProbeTraceParams {
    /// The query text, embedded as a search embeds it.
    #[serde(default)]
    pub query: Option<String>,
    /// First probe rank to trace (0 is the posting nearest the query); default 0.
    #[serde(default)]
    pub start: Option<u32>,
    /// One past the last probe rank to trace: at most the namespace's postings
    /// and at most `MAX_TRACE_PROBES` (1,024) past `start`. Default `start` plus
    /// the namespace's `max_probes` (no `start`) or plus `MAX_TRACE_PROBES`
    /// (with one), capped at the postings.
    #[serde(default)]
    pub end: Option<u32>,
    /// Results the baseline search returns: 1 to `MAX_TRACE_TOP_K` (10,000);
    /// default the namespace's `top_k`. Its Pass 1 keeps at least this many
    /// candidates. The namespace's settings do not change.
    #[serde(default)]
    pub top_k: Option<u32>,
}

/// A document id as the store's clients see it: per the document store's
/// `key_type` or the KV store's key type, hex when it does not decode.
fn vector_id_json(state: &AppState, ns: &str) -> impl Fn(&[u8]) -> serde_json::Value {
    let schema = state.store.schema(ns);
    move |bytes: &[u8]| {
        let decoded = match &schema {
            Some(minnal_db::doc_store::StoredSchema::Doc(s)) => minnal_db::DocId::from_bytes(bytes, s.key_type).ok().map(crate::id::doc_id_to_value),
            Some(minnal_db::doc_store::StoredSchema::Kv(s)) => s.key_type.deserialize_key(bytes).ok(),
            None => None,
        };
        decoded.unwrap_or_else(|| serde_json::Value::String(minnal_db::bytes_to_hex(bytes)))
    }
}

/// `GET /admin/indices/{ns}/vector/probe-trace?query=…&start=S&end=E&top_k=K` — how
/// one semantic query walks the namespace's partition: the postings at probe
/// ranks `S..E`, each with its distance from the query, the documents scored in
/// it, and its lowest and highest Pass-1 document score (with the document's id
/// and its Pass-2 score, what search results are ranked by), plus `baseline`:
/// the same query searched as the namespace searches it (returning `K` results
/// when given), and the Pass-2 score of its last result. Costs one embedding
/// (none when the query is cached), one search and one Pass 1 over the window.
/// `400` for a missing query, a window that is empty, past the namespace's
/// postings, or wider than `MAX_TRACE_PROBES` (1,024), or a `top_k` outside
/// 1 to `MAX_TRACE_TOP_K` (10,000).
pub async fn vector_probe_trace(
    State(state): State<AppState>,
    Path(ns): Path<String>,
    Query(params): Query<ProbeTraceParams>,
) -> axum::response::Response {
    let bad = |msg: String| (StatusCode::BAD_REQUEST, Json(serde_json::json!({ "error": msg }))).into_response();
    let query = match params.query.as_deref().map(str::trim) {
        Some(q) if !q.is_empty() => q.to_owned(),
        _ => return bad("query is required".into()),
    };
    let trace = match state.store.vector_probe_trace(&ns, &query, params.start, params.end, params.top_k).await {
        Ok(t) => t,
        Err(e) => return crate::routes::AppError::from(e).with_ns(&ns).into_response(),
    };
    let id = vector_id_json(&state, &ns);
    let score = |s: &Option<minnal_db::semantic_search::service::PostingDocScore>| {
        s.as_ref()
            .map(|s| serde_json::json!({ "dot_product": s.dot_product, "pass2": s.pass2, "in_results": s.in_results, "id": id(&s.document_id) }))
    };
    let postings: Vec<serde_json::Value> = trace
        .postings
        .iter()
        .map(|p| {
            serde_json::json!({
                "rank": p.rank,
                "posting_id": p.posting_id,
                "centre_id": p.centre_id,
                "distance": p.distance,
                "documents": p.documents,
                "min": score(&p.min),
                "max": score(&p.max),
            })
        })
        .collect();
    Json(serde_json::json!({
        "namespace": ns,
        "query": query,
        "start": trace.start,
        "end": trace.end,
        "active_postings": trace.active_postings,
        "pass1_bits": trace.pass1_bits,
        "baseline": {
            "n_probes": trace.baseline.n_probes,
            "top_k": trace.baseline.top_k,
            "first_pass_top_k": trace.baseline.first_pass_top_k,
            "results": trace.baseline.results,
            "lowest_pass2": trace.baseline.lowest_pass2,
        },
        "postings": postings,
    }))
    .into_response()
}

/// Query of `GET /admin/indices/{ns}/vector/history`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HistoryParams {
    /// `codes`, `recall_check` or `rebuild`; all kinds when absent.
    #[serde(default)]
    pub kind: Option<String>,
    /// Records to return, newest first: default 50, at most 1,000.
    #[serde(default)]
    pub limit: Option<usize>,
}

fn parse_history_kind(kind: &str) -> Option<minnal_db::semantic_search::ivf_split::history::RecordKind> {
    use minnal_db::semantic_search::ivf_split::history::RecordKind;
    match kind {
        "codes" => Some(RecordKind::Codes),
        "recall_check" => Some(RecordKind::RecallCheck),
        "rebuild" => Some(RecordKind::Rebuild),
        _ => None,
    }
}

/// `GET /admin/indices/{ns}/vector/history` — the namespace's partition history,
/// newest first: every code-health read (`partition?codes=true`), recall check
/// and rebuild still within `partition_history_ttl_secs` (default 30 days),
/// without recomputing any of them.
pub async fn vector_history(State(state): State<AppState>, Path(ns): Path<String>, Query(params): Query<HistoryParams>) -> axum::response::Response {
    let bad = |msg: String| (StatusCode::BAD_REQUEST, Json(serde_json::json!({ "error": msg }))).into_response();
    let kind = match params.kind.as_deref() {
        None => None,
        Some(k) => match parse_history_kind(k) {
            Some(kind) => Some(kind),
            None => return bad(format!("kind must be codes, recall_check or rebuild, got {k:?}")),
        },
    };
    let limit = params.limit.unwrap_or(50);
    if !(1..=minnal_db::semantic_search::ivf_split::history::MAX_HISTORY_LIMIT).contains(&limit) {
        return bad(format!(
            "limit must be between 1 and {}, got {limit}",
            minnal_db::semantic_search::ivf_split::history::MAX_HISTORY_LIMIT
        ));
    }
    match state.store.vector_partition_history(&ns, kind, limit).await {
        Ok(records) => Json(serde_json::json!({
            "namespace": ns,
            "ttl_secs": state.store.vector_partition_history_ttl().as_secs(),
            "records": records,
        }))
        .into_response(),
        Err(e) => crate::routes::AppError::from(e).with_ns(&ns).into_response(),
    }
}

/// Body of `POST /admin/indices/{ns}/vector/rebuild` (optional).
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RebuildBody {
    /// Postings to fit; default: as many as are active now.
    #[serde(default)]
    pub postings: Option<usize>,
}

/// `POST /admin/indices/{ns}/vector/rebuild` — re-cluster the namespace's
/// partition from its stored codes (k-means, no embedding calls) and move every
/// chunk to its nearest new posting, in the background. 202 on
/// start; 409 while one is running. Progress and the result are the `rebuild`
/// field of `GET /admin/indices/{ns}/vector/partition`.
pub async fn vector_rebuild(State(state): State<AppState>, Path(ns): Path<String>, body: axum::body::Bytes) -> axum::response::Response {
    // The body is optional: an empty one (with or without a JSON content type)
    // asks for the defaults.
    let body: RebuildBody = if body.iter().all(u8::is_ascii_whitespace) {
        RebuildBody::default()
    } else {
        match serde_json::from_slice(&body) {
            Ok(b) => b,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({"error": format!("invalid rebuild body: {e}")})),
                )
                    .into_response();
            }
        }
    };
    let postings = body.postings;
    if postings == Some(0) {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "postings must be at least 1"}))).into_response();
    }
    match state.store.start_vector_rebuild(&ns, postings) {
        Ok(job) => (
            StatusCode::ACCEPTED,
            Json(serde_json::json!({
                "namespace": ns,
                "job": job,
                "state": minnal_db::NamespaceState::IndexMaintenance.as_str(),
                "rebuild": state.store.vector_rebuild_status(&ns),
            })),
        )
            .into_response(),
        Err(e) => crate::routes::AppError::from(e).with_ns(&ns).into_response(),
    }
}

/// Query of `GET /admin/indices/{ns}/vector/postings`.
#[derive(Debug, Deserialize)]
pub struct PostingsParams {
    /// `id` (default), `entries`, `chunks` or `drift` (implies `codes`).
    #[serde(default)]
    pub sort: Option<String>,
    /// `asc` or `desc` (default).
    #[serde(default)]
    pub order: Option<String>,
    /// Opaque cursor from a previous page's `next_cursor`.
    #[serde(default)]
    pub cursor: Option<String>,
    /// Page size: default 1,000, at most 10,000.
    #[serde(default)]
    pub limit: Option<usize>,
    /// Include each posting's code metrics (reads every chunk code).
    #[serde(default)]
    pub codes: bool,
    /// Only postings in this state: `active`, `draining` or `retired`.
    #[serde(default)]
    pub state: Option<String>,
}

/// `GET /admin/indices/{ns}/vector/postings` — every posting (retired ones
/// included, as the parents of the split tree) with its size, state, centre,
/// parent and origin, cursor-paginated, for size-distribution charts. Sorted by
/// `sort` (`id`, `entries`, `chunks`, `drift`) and `order`, filtered by `state`.
pub async fn vector_postings(
    State(state): State<AppState>,
    Path(ns): Path<String>,
    Query(params): Query<PostingsParams>,
) -> axum::response::Response {
    let bad = |msg: String| (StatusCode::BAD_REQUEST, Json(serde_json::json!({ "error": msg }))).into_response();
    let sort = params.sort.as_deref().unwrap_or("id");
    if !matches!(sort, "id" | "entries" | "chunks" | "drift") {
        return bad(format!("sort must be id, entries, chunks or drift, got {sort:?}"));
    }
    let descending = match params.order.as_deref().unwrap_or("desc") {
        "desc" => true,
        "asc" => false,
        other => return bad(format!("order must be asc or desc, got {other:?}")),
    };
    let limit = params.limit.unwrap_or(1000);
    if !(1..=10_000).contains(&limit) {
        return bad(format!("limit must be between 1 and 10000, got {limit}"));
    }
    let start = match &params.cursor {
        Some(c) => match c.parse::<usize>() {
            Ok(n) => n,
            Err(_) => return bad("invalid cursor".into()),
        },
        None => 0,
    };
    if let Some(s) = params.state.as_deref()
        && !matches!(s, "active" | "draining" | "retired")
    {
        return bad(format!("state must be active, draining or retired, got {s:?}"));
    }
    let codes = params.codes || sort == "drift";
    let (maintenance, summary, mut postings) = match state.store.vector_partition_health(&ns, codes).await {
        Ok(r) => r,
        Err(e) => return crate::routes::AppError::from(e).with_ns(&ns).into_response(),
    };
    if let Some(s) = params.state.as_deref() {
        postings.retain(|p| p.state == s);
    }
    let key = |p: &minnal_db::semantic_search::ivf_split::health::PostingHealth| -> f64 {
        match sort {
            "entries" => p.entries as f64,
            "chunks" => p.chunks as f64,
            "drift" => p.codes.map_or(f64::NEG_INFINITY, |c| c.drift_mean),
            _ => f64::from(p.posting_id),
        }
    };
    postings.sort_by(|a, b| {
        let o = key(a).total_cmp(&key(b)).then(a.posting_id.cmp(&b.posting_id));
        if descending { o.reverse() } else { o }
    });
    let page: Vec<_> = postings.iter().skip(start).take(limit).collect();
    let next_cursor = (start + page.len() < postings.len()).then(|| (start + page.len()).to_string());
    let mut body = serde_json::json!({
        "namespace": ns,
        "postings": summary.postings,
        "entries": summary.entries,
        "chunks": summary.chunks,
        "matched": postings.len(),
        "results": page,
        "next_cursor": next_cursor,
    });
    if let (Some(obj), serde_json::Value::Object(m)) = (body.as_object_mut(), maintenance_json(&maintenance)) {
        obj.extend(m);
    }
    Json(body).into_response()
}

/// Query of `GET /admin/indices/{ns}/vector/centres`.
#[derive(Debug, Deserialize)]
pub struct CentresParams {
    /// `id` (default), `codes` (implies `codes`) or `postings`.
    #[serde(default)]
    pub sort: Option<String>,
    /// `asc` or `desc` (default).
    #[serde(default)]
    pub order: Option<String>,
    /// Opaque cursor from a previous page's `next_cursor`.
    #[serde(default)]
    pub cursor: Option<String>,
    /// Page size: default 1,000, at most 10,000.
    #[serde(default)]
    pub limit: Option<usize>,
    /// Count the chunk codes encoded against each centre (reads every chunk code).
    #[serde(default)]
    pub codes: bool,
    /// Only centres whose postings' most live state is this: `active`,
    /// `draining`, `retired` or `none`.
    #[serde(default)]
    pub state: Option<String>,
}

/// `GET /admin/indices/{ns}/vector/centres` — every IVF centre's metadata (no
/// vectors): the postings that name it, their state and origin, its norm and,
/// with `codes`, how many chunk codes still decode against it. `by_state`
/// counts every centre; with `codes`, `unused` counts centres that no live
/// posting names and no code uses (nothing collects them yet: centres are
/// append-only).
pub async fn vector_centres(State(state): State<AppState>, Path(ns): Path<String>, Query(params): Query<CentresParams>) -> axum::response::Response {
    let bad = |msg: String| (StatusCode::BAD_REQUEST, Json(serde_json::json!({ "error": msg }))).into_response();
    let sort = params.sort.as_deref().unwrap_or("id");
    if !matches!(sort, "id" | "codes" | "postings") {
        return bad(format!("sort must be id, codes or postings, got {sort:?}"));
    }
    let descending = match params.order.as_deref().unwrap_or("desc") {
        "desc" => true,
        "asc" => false,
        other => return bad(format!("order must be asc or desc, got {other:?}")),
    };
    if let Some(s) = params.state.as_deref()
        && !matches!(s, "active" | "draining" | "retired" | "none")
    {
        return bad(format!("state must be active, draining, retired or none, got {s:?}"));
    }
    let limit = params.limit.unwrap_or(1000);
    if !(1..=10_000).contains(&limit) {
        return bad(format!("limit must be between 1 and 10000, got {limit}"));
    }
    let start = match &params.cursor {
        Some(c) => match c.parse::<usize>() {
            Ok(n) => n,
            Err(_) => return bad("invalid cursor".into()),
        },
        None => 0,
    };
    let codes = params.codes || sort == "codes";
    let mut centres = match state.store.vector_centres(&ns, codes).await {
        Ok(r) => r,
        Err(e) => return crate::routes::AppError::from(e).with_ns(&ns).into_response(),
    };
    let mut by_state = serde_json::Map::new();
    for c in &centres {
        let n = by_state.entry(c.state).or_insert(serde_json::json!(0));
        *n = serde_json::json!(n.as_u64().unwrap_or(0) + 1);
    }
    let total = centres.len();
    let unused = codes.then(|| {
        centres
            .iter()
            .filter(|c| matches!(c.state, "retired" | "none") && c.codes == Some(0))
            .count()
    });
    if let Some(s) = params.state.as_deref() {
        centres.retain(|c| c.state == s);
    }
    let key = |c: &minnal_db::semantic_search::ivf_split::health::CentreHealth| -> u64 {
        match sort {
            "codes" => c.codes.unwrap_or(0),
            "postings" => c.postings.len() as u64,
            _ => u64::from(c.centre_id),
        }
    };
    centres.sort_by(|a, b| {
        let o = key(a).cmp(&key(b)).then(a.centre_id.cmp(&b.centre_id));
        if descending { o.reverse() } else { o }
    });
    let page: Vec<_> = centres.iter().skip(start).take(limit).collect();
    let next_cursor = (start + page.len() < centres.len()).then(|| (start + page.len()).to_string());
    Json(serde_json::json!({
        "namespace": ns,
        "centres": total,
        "by_state": by_state,
        "unused": unused,
        "matched": centres.len(),
        "results": page,
        "next_cursor": next_cursor,
    }))
    .into_response()
}
