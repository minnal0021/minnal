//! Storage diagnostics and operations — `GET/POST /admin/storage/*`
//!
//! ```text
//! GET  /admin/storage/health                          → server liveness + uptime
//! GET  /admin/storage/stats                           → value-log statistics across all namespaces
//! GET  /admin/storage/ops-metrics                      → engine-wide operational counters (reads/writes/compaction/GC + read-path efficiency)
//! GET  /admin/storage/ops-metrics/by-namespace         → the same counters, broken out per namespace
//! GET  /admin/storage/stores/{ns}/ops-metrics          → operational counters for one namespace
//! GET  /admin/storage/wal                             → WAL metadata snapshot
//! GET  /admin/storage/lsm                             → LSM manifest for every namespace
//! GET  /admin/storage/value-log                       → per-namespace, per-bucket utilization + every segment file (like /wal)
//! GET  /admin/storage/value-log/{ns}/segments         → per-segment garbage breakdown for one namespace (deep dive)
//! GET  /admin/storage/namespaces                      → doc-store namespace registry with field schemas
//! GET  /admin/storage/namespaces/physical             → every physical engine namespace (incl. companions/system), each with a role
//! GET  /admin/storage/stores/{ns}/kv-meta             → engine storage metadata for one store (doc or KV)
//! GET  /admin/storage/system/stores                   → list all stores in the system namespace
//! GET  /admin/storage/system/stores/{ns}/meta         → metadata for one system KV store
//! GET  /admin/storage/index-waste                     → per-field field-index bitmap/keymap waste + threshold
//! POST /admin/storage/gc                              → trigger value-log GC across all namespaces
//! POST /admin/storage/gc/wal                          → trigger WAL GC
//! POST /admin/storage/compact                         → trigger LSM compaction
//! POST /admin/storage/index-checkpoint                → flush + compact field indexes (and row maps) — runs in background (202)
//! ```

use std::collections::HashMap;

use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
};
use minnal_db::{FieldMeta, LsmManifest, ValueLogMetadata};
use serde::Serialize;
use tracing::{error, info};

use minnal_db::{KvKeyType, KvValueType, StoreType};

use crate::{AppState, error::AppError};

// ── Health ────────────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct HealthResponse {
    status: &'static str,
    uptime_s: u64,
}

pub async fn health(State(state): State<AppState>) -> impl IntoResponse {
    Json(HealthResponse {
        status: "ok",
        uptime_s: state.started_at.elapsed().as_secs(),
    })
}

// ── Stats ─────────────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct StatsResponse {
    /// Value-log segment files on disk across **every** namespace.
    segment_count: u64,
    /// Bytes those segment files occupy.
    disk_bytes: u64,
    garbage_bytes: u64,
    live_bytes: u64,
    waste_ratio_pct: f64,
    total_gc_runs: u64,
    total_bytes_reclaimed: u64,
    /// How many namespaces are included in the figures above.
    namespaces: usize,
}

/// `GET /admin/storage/stats` — value-log statistics aggregated across **all**
/// namespaces.
///
/// (It used to report only the *default* namespace while being documented as
/// engine-wide, which made a busy database look empty.)
pub async fn stats(State(state): State<AppState>) -> impl IntoResponse {
    let disk_bytes: u64 = state
        .store
        .value_log_physical_stats()
        .into_iter()
        .flat_map(|(_ns, shards)| shards.into_iter().map(|s| s.physical_bytes))
        .sum();

    let mut out = StatsResponse {
        segment_count: 0,
        disk_bytes,
        garbage_bytes: 0,
        live_bytes: 0,
        waste_ratio_pct: 0.0,
        total_gc_runs: 0,
        total_bytes_reclaimed: 0,
        namespaces: 0,
    };

    let per_ns = state.store.value_log_shard_stats();
    out.namespaces = per_ns.len();
    for (_ns, buckets) in per_ns {
        for (_bucket, m) in buckets {
            out.segment_count = out.segment_count.saturating_add(m.segments.len() as u64);
            out.live_bytes = out.live_bytes.saturating_add(m.live_bytes());
            out.garbage_bytes = out.garbage_bytes.saturating_add(m.garbage_bytes());
            out.total_gc_runs = out.total_gc_runs.saturating_add(m.total_gc_runs);
            out.total_bytes_reclaimed = out.total_bytes_reclaimed.saturating_add(m.total_bytes_reclaimed);
        }
    }
    let written = out.live_bytes.saturating_add(out.garbage_bytes);
    out.waste_ratio_pct = if written > 0 {
        (out.garbage_bytes as f64 / written as f64) * 100.0
    } else {
        0.0
    };
    Json(out)
}

// ── Operational metrics ────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct ReadMetrics {
    reads: u64,
    read_hits: u64,
    read_misses: u64,
    read_hit_ratio: f64,
    scans: u64,
    scan_rows: u64,
}

#[derive(Serialize)]
pub struct LsmLookupMetrics {
    /// Point lookups through the LSM (≥ `reads`: also counts GC-validation reads).
    lookups: u64,
    /// Lookups served by the active-memtable fast path (no lower-layer scan).
    fast_path_hits: u64,
    fast_path_hit_ratio: f64,
    l0_probes: u64,
    l1_probes: u64,
    bloom_rejects: u64,
}

#[derive(Serialize)]
pub struct WriteMetrics {
    puts: u64,
    deletes: u64,
    no_wal_puts: u64,
    no_wal_deletes: u64,
    wal_bytes_appended: u64,
    wal_fsyncs: u64,
    apply_failures: u64,
}

#[derive(Serialize)]
pub struct CompactionMetrics {
    memtable_flushes: u64,
    l0_l1_compactions: u64,
    compaction_bytes_merged: u64,
    compaction_duration_ms: u64,
}

#[derive(Serialize)]
pub struct GcMetrics {
    vlog_gc_runs: u64,
    vlog_gc_duration_ms: u64,
    /// Value-log segment files unlinked by GC.
    vlog_segments_reclaimed: u64,
    /// Bytes handed back to the filesystem by unlinking those segments.
    vlog_gc_bytes_reclaimed: u64,
    /// Bytes of survivors GC rewrote to relocate them out of those segments — the
    /// cost of the work.
    vlog_gc_bytes_rewritten: u64,
    /// `bytes_rewritten / bytes_reclaimed`: **GC's write amplification**, and the
    /// number to watch. Well below 1 is healthy — GC frees far more than it rewrites.
    /// Above 1 means it is relocating more data than it frees (the segments it picks
    /// are mostly survivors); `thresholds.segment_gc_threshold` is the knob for that.
    /// `null` until GC has reclaimed anything.
    vlog_gc_write_amplification: Option<f64>,
    wal_gc_runs: u64,
    wal_segments_deleted: u64,
}

/// The grouped operational-metrics body (without the engine-level `uptime_s`),
/// reused by the engine, per-namespace, and by-namespace endpoints.
#[derive(Serialize)]
pub struct OpsMetricsBody {
    reads: ReadMetrics,
    lsm_lookups: LsmLookupMetrics,
    writes: WriteMetrics,
    compaction: CompactionMetrics,
    gc: GcMetrics,
}

#[derive(Serialize)]
pub struct OpsMetricsResponse {
    uptime_s: u64,
    #[serde(flatten)]
    body: OpsMetricsBody,
}

/// One namespace's operational metrics (engine `uptime_s` omitted; the WAL-GC
/// counters under `gc` are engine-global and always `0` here).
#[derive(Serialize)]
pub struct NamespaceOpsMetrics {
    namespace: String,
    #[serde(flatten)]
    body: OpsMetricsBody,
}

fn ratio(num: u64, denom: u64) -> f64 {
    if denom == 0 { 0.0 } else { num as f64 / denom as f64 }
}

/// Build the grouped metrics body (with derived ratios) from a raw snapshot.
fn ops_metrics_body(m: &minnal_db::MetricsSnapshot) -> OpsMetricsBody {
    OpsMetricsBody {
        reads: ReadMetrics {
            reads: m.reads,
            read_hits: m.read_hits,
            read_misses: m.read_misses,
            read_hit_ratio: ratio(m.read_hits, m.reads),
            scans: m.scans,
            scan_rows: m.scan_rows,
        },
        lsm_lookups: LsmLookupMetrics {
            lookups: m.lookups,
            fast_path_hits: m.fast_path_hits,
            fast_path_hit_ratio: ratio(m.fast_path_hits, m.lookups),
            l0_probes: m.l0_probes,
            l1_probes: m.l1_probes,
            bloom_rejects: m.bloom_rejects,
        },
        writes: WriteMetrics {
            puts: m.puts,
            deletes: m.deletes,
            no_wal_puts: m.no_wal_puts,
            no_wal_deletes: m.no_wal_deletes,
            wal_bytes_appended: m.wal_bytes_appended,
            wal_fsyncs: m.wal_fsyncs,
            apply_failures: m.apply_failures,
        },
        compaction: CompactionMetrics {
            memtable_flushes: m.memtable_flushes,
            l0_l1_compactions: m.l0_l1_compactions,
            compaction_bytes_merged: m.compaction_bytes_merged,
            compaction_duration_ms: m.compaction_duration_ms,
        },
        gc: GcMetrics {
            vlog_gc_runs: m.vlog_gc_runs,
            vlog_gc_duration_ms: m.vlog_gc_duration_ms,
            vlog_segments_reclaimed: m.vlog_segments_reclaimed,
            vlog_gc_bytes_reclaimed: m.vlog_gc_bytes_reclaimed,
            vlog_gc_bytes_rewritten: m.vlog_gc_bytes_rewritten,
            vlog_gc_write_amplification: (m.vlog_gc_bytes_reclaimed > 0).then(|| m.vlog_gc_bytes_rewritten as f64 / m.vlog_gc_bytes_reclaimed as f64),
            wal_gc_runs: m.wal_gc_runs,
            wal_segments_deleted: m.wal_segments_deleted,
        },
    }
}

/// `GET /admin/storage/ops-metrics` — engine-wide operational counters since
/// startup (cumulative; sample twice to compute rates). The engine view is the
/// sum of every namespace's counters plus the global WAL-GC/retired totals.
/// Includes derived read-path ratios (`read_hit_ratio`, `fast_path_hit_ratio`).
pub async fn ops_metrics(State(state): State<AppState>) -> impl IntoResponse {
    let m = state.store.ops_metrics();
    Json(OpsMetricsResponse {
        uptime_s: state.started_at.elapsed().as_secs(),
        body: ops_metrics_body(&m),
    })
}

/// `GET /admin/storage/stores/{ns}/ops-metrics` — operational counters for one
/// namespace. The WAL-GC counters are engine-global and read `0` here.
pub async fn ops_metrics_ns(State(state): State<AppState>, Path(ns): Path<String>) -> Result<impl IntoResponse, AppError> {
    let m = state.store.ops_metrics_for(&ns).map_err(|e| AppError::from(e).with_ns(&ns))?;
    Ok(Json(OpsMetricsResponse {
        uptime_s: state.started_at.elapsed().as_secs(),
        body: ops_metrics_body(&m),
    }))
}

/// `GET /admin/storage/ops-metrics/by-namespace` — per-namespace operational
/// counters for every live namespace (engine-global WAL-GC counters excluded).
pub async fn ops_metrics_by_namespace(State(state): State<AppState>) -> impl IntoResponse {
    let per_ns: Vec<NamespaceOpsMetrics> = state
        .store
        .ops_metrics_by_namespace()
        .into_iter()
        .map(|(namespace, m)| NamespaceOpsMetrics {
            namespace,
            body: ops_metrics_body(&m),
        })
        .collect();
    Json(per_ns)
}

// ── WAL ───────────────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct WalSegmentStats {
    segment_id: u64,
    total_entries: u64,
    persisted_entries: u64,
    pending_entries: u64,
}

#[derive(Serialize)]
pub struct WalResponse {
    head: u64,
    tail: u64,
    total_entries: u64,
    persisted_entries: u64,
    pending_entries: u64,
    total_gc_runs: u64,
    total_bytes_reclaimed: u64,
    /// Absolute segment id the per-segment counters start at; counters for
    /// segments below this were reclaimed and trimmed.
    base_segment_id: u64,
    /// Number of tracked segments still carrying entries (the live window).
    live_segments: usize,
    /// Highest write sequence the WAL has observed (GC high-water mark).
    last_sequence: u64,
    segments: Vec<WalSegmentStats>,
}

pub async fn wal(State(state): State<AppState>) -> impl IntoResponse {
    let m = state.store.wal_metadata();
    let pending = m.total_entries.saturating_sub(m.persisted_entries);
    // The per-segment vecs are dense from `base_segment_id`, so the real segment
    // id is base + relative index (not the bare index).
    let segments: Vec<WalSegmentStats> = m
        .segment_total_entries
        .iter()
        .zip(m.segment_persisted_entries.iter())
        .enumerate()
        .map(|(i, (&total, &persisted))| WalSegmentStats {
            segment_id: m.base_segment_id + i as u64,
            total_entries: total,
            persisted_entries: persisted,
            pending_entries: total.saturating_sub(persisted),
        })
        .collect();
    let live_segments = m.segment_total_entries.iter().filter(|&&t| t > 0).count();
    Json(WalResponse {
        head: m.head,
        tail: m.tail,
        total_entries: m.total_entries,
        persisted_entries: m.persisted_entries,
        pending_entries: pending,
        total_gc_runs: m.total_gc_runs,
        total_bytes_reclaimed: m.total_bytes_reclaimed,
        base_segment_id: m.base_segment_id,
        live_segments,
        last_sequence: m.last_sequence,
        segments,
    })
}

// ── LSM ───────────────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct LsmFileInfo {
    path: String,
    created_at_ms: u128,
    entry_count: u64,
    /// On-disk size of the SSTable file in bytes (0 if it can't be stat'd).
    size_bytes: u64,
}

/// In-memory LSM state (active + sealed memtables) that the on-disk manifest
/// does not capture.
#[derive(Serialize)]
pub struct LsmInMemoryInfo {
    memtable_entries: usize,
    read_only_entries: usize,
    read_only_count: usize,
    compaction_in_progress: bool,
}

#[derive(Serialize)]
pub struct LsmBucketInfo {
    bucket: u32,
    file_count: usize,
    total_entries: u64,
    files: Vec<LsmFileInfo>,
}

#[derive(Serialize)]
pub struct LsmLevelInfo {
    level: u8,
    bucket_count: usize,
    total_entries: u64,
    buckets: Vec<LsmBucketInfo>,
}

#[derive(Serialize)]
pub struct LsmNamespaceInfo {
    namespace: String,
    manifest_version: u32,
    created_at_ms: u64,
    level_count: usize,
    total_entries: u64,
    /// On-disk size summed across all SSTable files (bytes).
    total_size_bytes: u64,
    /// Live in-memory state, present on the `/admin/storage/lsm` listing.
    #[serde(skip_serializing_if = "Option::is_none")]
    in_memory: Option<LsmInMemoryInfo>,
    levels: Vec<LsmLevelInfo>,
}

pub fn build_lsm_namespace_info(ns: String, m: LsmManifest) -> LsmNamespaceInfo {
    let levels: Vec<LsmLevelInfo> = m
        .levels
        .iter()
        .map(|lvl| {
            let buckets: Vec<LsmBucketInfo> = lvl
                .buckets
                .iter()
                .map(|b| {
                    let total_entries: u64 = b.files.iter().map(|f| f.entry_count).sum();
                    let files: Vec<LsmFileInfo> = b
                        .files
                        .iter()
                        .map(|f| LsmFileInfo {
                            size_bytes: std::fs::metadata(&f.path).map(|md| md.len()).unwrap_or(0),
                            path: f.path.clone(),
                            created_at_ms: f.created_at_ms,
                            entry_count: f.entry_count,
                        })
                        .collect();
                    LsmBucketInfo {
                        bucket: b.bucket,
                        file_count: files.len(),
                        total_entries,
                        files,
                    }
                })
                .collect();
            let total_entries: u64 = buckets.iter().map(|b| b.total_entries).sum();
            LsmLevelInfo {
                level: lvl.level,
                bucket_count: buckets.len(),
                total_entries,
                buckets,
            }
        })
        .collect();
    let total_entries: u64 = levels.iter().map(|l| l.total_entries).sum();
    let total_size_bytes: u64 = levels
        .iter()
        .flat_map(|l| l.buckets.iter())
        .flat_map(|b| b.files.iter())
        .map(|f| f.size_bytes)
        .sum();
    LsmNamespaceInfo {
        namespace: ns,
        manifest_version: m.version,
        created_at_ms: m.created_at_ms,
        level_count: levels.len(),
        total_entries,
        total_size_bytes,
        in_memory: None,
        levels,
    }
}

pub async fn lsm(State(state): State<AppState>) -> impl IntoResponse {
    // In-memory stats (active + sealed memtables) keyed by namespace, to enrich
    // the on-disk manifest view.
    let runtime: HashMap<String, minnal_db::LSMStats> = state.store.lsm_runtime_stats().into_iter().collect();
    let result: Vec<LsmNamespaceInfo> = state
        .store
        .lsm_manifests()
        .into_iter()
        .map(|(ns, m)| {
            let mut info = build_lsm_namespace_info(ns.clone(), m);
            info.in_memory = runtime.get(&ns).map(|s| LsmInMemoryInfo {
                memtable_entries: s.memtable_entries,
                read_only_entries: s.read_only_entries,
                read_only_count: s.read_only_count,
                compaction_in_progress: s.compaction_in_progress,
            });
            info
        })
        .collect();
    Json(result)
}

// ── Namespaces ────────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct FieldInfo {
    field_id: u32,
    field_name: String,
    field_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    distinct_count: Option<usize>,
}

impl FieldInfo {
    fn from_meta(m: FieldMeta, store: &minnal_db::DocStore, ns: &str) -> Self {
        let distinct_count = store.field_index_distinct_count(ns, &m.field_name);
        Self {
            field_id: m.field_id,
            field_name: m.field_name,
            field_type: format!("{:?}", m.field_type),
            distinct_count,
        }
    }
}

#[derive(Serialize)]
pub struct DocStoreNamespaceInfo {
    name: String,
    ns_id: u32,
    key_type: String,
    semantic_search_enabled: bool,
    indexed_fields: Vec<FieldInfo>,
}

#[derive(Serialize)]
pub struct KvStoreNamespaceInfo {
    name: String,
    ns_id: u32,
    key_type: String,
    value_type: String,
    semantic_search_enabled: bool,
}

#[derive(Serialize)]
pub struct NamespacesResponse {
    doc_stores: Vec<DocStoreNamespaceInfo>,
    kv_stores: Vec<KvStoreNamespaceInfo>,
}

pub async fn namespaces(State(state): State<AppState>) -> impl IntoResponse {
    let schemas = state.schemas.read().await;
    let mut doc_stores: Vec<DocStoreNamespaceInfo> = schemas
        .iter()
        .map(|(name, schema)| {
            let ns_id = schema.ns_id.unwrap_or(0);
            let indexed_fields = if let Some(id) = schema.ns_id {
                // Filter to the schema's current indices — the db registry retains
                // dropped fields (for field_id reuse), which must not show here.
                let active: std::collections::HashSet<&str> = schema.indices.iter().map(|s| s.field.as_str()).collect();
                state
                    .store
                    .list_index_fields(id)
                    .into_iter()
                    .filter(|m| active.contains(m.field_name.as_str()))
                    .map(|m| FieldInfo::from_meta(m, &state.store, name))
                    .collect()
            } else {
                vec![]
            };
            DocStoreNamespaceInfo {
                name: name.clone(),
                ns_id,
                key_type: format!("{:?}", schema.key_type).to_lowercase(),
                semantic_search_enabled: schema.semantic_search_enabled,
                indexed_fields,
            }
        })
        .collect();
    doc_stores.sort_by(|a, b| a.name.cmp(&b.name));

    let kv_schemas = state.kv_schemas.read().await;
    let mut kv_stores: Vec<KvStoreNamespaceInfo> = kv_schemas
        .values()
        .map(|s| KvStoreNamespaceInfo {
            name: s.namespace.clone(),
            ns_id: s.ns_id.unwrap_or(0),
            key_type: kv_key_type_str(s.key_type),
            value_type: kv_value_type_str(s.value_type),
            semantic_search_enabled: s.semantic_search_enabled,
        })
        .collect();
    kv_stores.sort_by(|a, b| a.name.cmp(&b.name));

    Json(NamespacesResponse { doc_stores, kv_stores })
}

// ── Value-log utilization ─────────────────────────────────────────────────────

/// One value-log segment file. Mirrors the per-segment breakdown the WAL exposes.
#[derive(Serialize, Clone)]
pub struct SegmentInfo {
    segment_id: u32,
    /// The file's size on disk (16-byte header + records). Segments are dense
    /// append-only files, so this is exact — there are no sparse holes.
    file_bytes: u64,
    /// Record bytes in the file (live + garbage), excluding the header.
    total_bytes: u64,
    live_bytes: u64,
    garbage_bytes: u64,
    garbage_ratio_pct: f64,
    /// Sealed segments are immutable and are what GC selects from. Exactly one
    /// segment per shard is unsealed — the active tail — and it is never collected.
    sealed: bool,
}

fn segment_info(s: &minnal_db::SegmentStats) -> SegmentInfo {
    SegmentInfo {
        segment_id: s.id,
        file_bytes: s.file_bytes(),
        total_bytes: s.total_bytes,
        live_bytes: s.live_bytes,
        garbage_bytes: s.garbage_bytes,
        garbage_ratio_pct: s.garbage_ratio_pct(),
        sealed: s.sealed,
    }
}

#[derive(Serialize)]
pub struct ValueLogShardInfo {
    bucket: u32,
    /// Segment files this shard currently holds.
    segment_count: usize,
    /// The shard's active tail — the only segment still being appended to.
    active_segment_id: u32,
    /// Next segment id to be handed out. Monotone and never reused, so it also
    /// counts how many segments this shard has ever created.
    next_segment_id: u64,
    /// Sealed (immutable) segments — the ones GC can select from.
    sealed_segment_count: usize,
    live_bytes: u64,
    garbage_bytes: u64,
    waste_ratio_pct: f64,
    total_gc_runs: u64,
    total_bytes_reclaimed: u64,
    /// Every segment file in this shard, newest id last — the same per-segment
    /// breakdown `/admin/storage/wal` gives for WAL segments.
    segments: Vec<SegmentInfo>,
    /// Bytes the shard's segment files occupy on disk.
    #[serde(skip_serializing_if = "Option::is_none")]
    physical_bytes: Option<u64>,
    /// Record bytes tracked (live + garbage), excluding segment file headers.
    #[serde(skip_serializing_if = "Option::is_none")]
    logical_bytes: Option<u64>,
}

#[derive(Serialize)]
pub struct ValueLogNamespaceInfo {
    namespace: String,
    total_live_bytes: u64,
    total_garbage_bytes: u64,
    waste_ratio_pct: f64,
    /// On-disk physical bytes summed across shards (present on the listing).
    #[serde(skip_serializing_if = "Option::is_none")]
    total_physical_bytes: Option<u64>,
    shards: Vec<ValueLogShardInfo>,
}

pub fn build_vlog_namespace_info(ns: String, buckets: Vec<(u32, ValueLogMetadata)>) -> ValueLogNamespaceInfo {
    let mut total_live = 0u64;
    let mut total_garbage = 0u64;
    let shards: Vec<ValueLogShardInfo> = buckets
        .into_iter()
        .map(|(bucket, m)| {
            let live = m.live_bytes();
            let garbage = m.garbage_bytes();
            total_live = total_live.saturating_add(live);
            total_garbage = total_garbage.saturating_add(garbage);
            let total_written = live.saturating_add(garbage);
            let waste_pct = if total_written > 0 {
                (garbage as f64 / total_written as f64) * 100.0
            } else {
                0.0
            };
            let mut segments: Vec<SegmentInfo> = m.segments.iter().map(segment_info).collect();
            segments.sort_by_key(|s| s.segment_id);
            ValueLogShardInfo {
                bucket,
                segment_count: m.segments.len(),
                active_segment_id: m.active_segment_id,
                next_segment_id: m.next_segment_id,
                sealed_segment_count: m.segments.iter().filter(|s| s.sealed).count(),
                live_bytes: live,
                garbage_bytes: garbage,
                waste_ratio_pct: waste_pct,
                total_gc_runs: m.total_gc_runs,
                total_bytes_reclaimed: m.total_bytes_reclaimed,
                segments,
                physical_bytes: None,
                logical_bytes: None,
            }
        })
        .collect();
    let total_written = total_live.saturating_add(total_garbage);
    let waste_pct = if total_written > 0 {
        (total_garbage as f64 / total_written as f64) * 100.0
    } else {
        0.0
    };
    ValueLogNamespaceInfo {
        namespace: ns,
        total_live_bytes: total_live,
        total_garbage_bytes: total_garbage,
        waste_ratio_pct: waste_pct,
        total_physical_bytes: None,
        shards,
    }
}

pub async fn value_log(State(state): State<AppState>) -> impl IntoResponse {
    // On-disk footprint per (namespace, bucket) — cheap stat, merged
    // into the logical metadata view so callers see the true on-disk usage.
    let physical: HashMap<String, HashMap<u32, minnal_db::ShardPhysicalStats>> = state
        .store
        .value_log_physical_stats()
        .into_iter()
        .map(|(ns, shards)| (ns, shards.into_iter().map(|s| (s.bucket, s)).collect()))
        .collect();

    let result: Vec<ValueLogNamespaceInfo> = state
        .store
        .value_log_shard_stats()
        .into_iter()
        .map(|(ns, buckets)| {
            let mut info = build_vlog_namespace_info(ns.clone(), buckets);
            if let Some(ns_phys) = physical.get(&ns) {
                let mut total_physical = 0u64;
                for shard in &mut info.shards {
                    if let Some(p) = ns_phys.get(&shard.bucket) {
                        shard.physical_bytes = Some(p.physical_bytes);
                        shard.logical_bytes = Some(p.logical_bytes);
                        total_physical = total_physical.saturating_add(p.physical_bytes);
                    }
                }
                info.total_physical_bytes = Some(total_physical);
            }
            info
        })
        .collect();
    Json(result)
}

// ── Value-log per-segment garbage breakdown (deep dive, per namespace) ─────────

#[derive(Serialize)]
pub struct ValueLogSegmentsShard {
    bucket: u32,
    segment_count: usize,
    segments: Vec<SegmentInfo>,
}

#[derive(Serialize)]
pub struct ValueLogSegmentsResponse {
    namespace: String,
    shards: Vec<ValueLogSegmentsShard>,
}

/// `GET /admin/storage/value-log/{ns}/segments` — per-segment garbage breakdown for
/// one namespace: shows *which* segments GC would collect next.
///
/// Cheap: the counters are maintained in memory by writers, so this reads no segment
/// data at all. (The per-page version it replaces had to scan every record.)
pub async fn value_log_segments(
    State(state): State<AppState>,
    Path(ns): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    match state.store.value_log_segment_stats(&ns) {
        Ok(per_bucket) => {
            let shards = per_bucket
                .into_iter()
                .map(|(bucket, segments)| ValueLogSegmentsShard {
                    bucket,
                    segment_count: segments.len(),
                    segments: segments.iter().map(segment_info).collect(),
                })
                .collect();
            Ok(Json(ValueLogSegmentsResponse { namespace: ns, shards }))
        }
        Err(e) => Err((StatusCode::NOT_FOUND, Json(serde_json::json!({ "error": e.to_string() })))),
    }
}

// ── KV namespaces ─────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct KvNamespaceInfo {
    name: String,
    id: u32,
    role: &'static str,
}

pub async fn physical_namespaces(State(state): State<AppState>) -> impl IntoResponse {
    let doc_names: std::collections::HashSet<String> = state.schemas.read().await.keys().cloned().collect();
    let kv_names: std::collections::HashSet<String> = state.kv_schemas.read().await.keys().cloned().collect();

    let mut result: Vec<KvNamespaceInfo> = state
        .store
        .list_kv_namespaces()
        .into_iter()
        .map(|(name, id)| {
            let role = classify_namespace_role(&name, &doc_names, &kv_names);
            KvNamespaceInfo { name, id, role }
        })
        .collect();
    result.sort_by(|a, b| a.name.cmp(&b.name));
    Json(result)
}

fn classify_namespace_role(name: &str, doc_names: &std::collections::HashSet<String>, kv_names: &std::collections::HashSet<String>) -> &'static str {
    if is_system_ns(name) {
        return "system";
    }
    if doc_names.contains(name) {
        return "doc_store";
    }
    if kv_names.contains(name) {
        return "kv_store";
    }
    for suffix in ["_sparse_vector", "_dense_vector", "_sparse_vector_meta"] {
        if let Some(prefix) = name.strip_suffix(suffix)
            && (doc_names.contains(prefix) || kv_names.contains(prefix))
        {
            return "companion";
        }
    }
    "unknown"
}

// ── GC triggers ───────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct NamespaceGCResult {
    namespace: String,
    bytes_reclaimed: u64,
    bytes_live: u64,
    gc_run_count: u64,
    total_bytes_reclaimed: u64,
    gc_duration_ms: u128,
}

#[derive(Serialize)]
pub struct GCResponse {
    namespaces_collected: usize,
    results: Vec<NamespaceGCResult>,
}

pub async fn trigger_gc(State(state): State<AppState>) -> impl IntoResponse {
    info!("admin: value-log GC triggered manually");
    let results = state.store.garbage_collect_all().await;
    let count = results.len();
    let total_reclaimed: u64 = results.iter().map(|(_, s)| s.bytes_reclaimed).sum();
    info!(
        namespaces = count,
        total_bytes_reclaimed = total_reclaimed,
        "admin: value-log GC complete"
    );
    let ns_results: Vec<NamespaceGCResult> = results
        .into_iter()
        .map(|(ns, s)| NamespaceGCResult {
            namespace: ns,
            bytes_reclaimed: s.bytes_reclaimed,
            bytes_live: s.bytes_live,
            gc_run_count: s.gc_run_count,
            total_bytes_reclaimed: s.total_bytes_reclaimed,
            gc_duration_ms: s.gc_duration_ms,
        })
        .collect();
    Json(GCResponse {
        namespaces_collected: count,
        results: ns_results,
    })
}

#[derive(Serialize)]
pub struct WalGCResponse {
    total_entries: u64,
    persisted_entries: u64,
}

pub async fn trigger_gc_wal(State(state): State<AppState>) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    info!("admin: WAL GC triggered manually");
    match state.store.garbage_collect_wal().await {
        Ok((total, persisted)) => {
            info!(total_entries = total, persisted_entries = persisted, "admin: WAL GC complete");
            Ok(Json(WalGCResponse {
                total_entries: total,
                persisted_entries: persisted,
            }))
        }
        Err(e) => Err((StatusCode::CONFLICT, Json(serde_json::json!({ "error": e.to_string() })))),
    }
}

pub async fn trigger_compact(State(state): State<AppState>) -> Result<StatusCode, (StatusCode, Json<serde_json::Value>)> {
    info!("LSM compaction triggered manually");
    match state.store.compact().await {
        Ok(()) => {
            info!("LSM compaction complete");
            Ok(StatusCode::NO_CONTENT)
        }
        Err(e) => {
            error!(error = %e, "LSM compaction failed");
            Err((StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() }))))
        }
    }
}

/// Force an index checkpoint: flush every namespace's dense row map and all
/// active field indexes to disk, compacting any field-index bitmap store whose
/// waste exceeds `thresholds.index_blob_waste_threshold`. This is the same pass
/// the periodic worker and clean shutdown run.
///
/// Because the compaction can take a long time on a large/wasted index (each
/// over-threshold field is rebuilt under its own write lock), this **returns
/// immediately with `202 Accepted`** and runs the pass in the **background**; the
/// checkpointed-field count is written to the log (`info!` on completion,
/// `error!` on failure). Overlapping runs are rejected with `409 Conflict` so
/// expensive flush/compaction passes cannot stack.
pub async fn trigger_index_checkpoint(State(state): State<AppState>) -> Result<StatusCode, (StatusCode, Json<serde_json::Value>)> {
    use std::sync::atomic::Ordering;

    // Reject overlapping runs — a checkpoint can be a full flush + compaction.
    if state
        .index_checkpoint_running
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return Err((
            StatusCode::CONFLICT,
            Json(serde_json::json!({ "error": "an index checkpoint is already running" })),
        ));
    }

    info!("index checkpoint accepted — running in background");
    let store = std::sync::Arc::clone(&state.store);
    let running = std::sync::Arc::clone(&state.index_checkpoint_running);
    tokio::spawn(async move {
        // Clear the running flag however the task exits (including a panic), so a
        // failure can never wedge the endpoint into a permanent 409.
        struct ResetOnDrop(std::sync::Arc<std::sync::atomic::AtomicBool>);
        impl Drop for ResetOnDrop {
            fn drop(&mut self) {
                self.0.store(false, std::sync::atomic::Ordering::Release);
            }
        }
        let _reset = ResetOnDrop(running);

        match store.checkpoint_index().await {
            Ok(fields_checkpointed) => info!(fields_checkpointed, "index checkpoint complete"),
            Err(e) => error!(error = %e, "index checkpoint failed"),
        }
    });

    Ok(StatusCode::ACCEPTED)
}

// ── Index waste ────────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct FieldWasteInfo {
    field_id: u32,
    field_name: String,
    field_type: String,
    /// Reclaimable fraction (`0.0..1.0`) of the bitmap value region; `null` when
    /// the field is not active (e.g. still building).
    #[serde(skip_serializing_if = "Option::is_none")]
    bitmap_waste_ratio: Option<f64>,
    /// Reclaimable fraction (`0.0..1.0`) of the keymap value region.
    #[serde(skip_serializing_if = "Option::is_none")]
    keymap_waste_ratio: Option<f64>,
    /// True if either store has reached the compaction threshold.
    over_threshold: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    distinct_count: Option<usize>,
}

#[derive(Serialize)]
pub struct NamespaceWasteInfo {
    namespace: String,
    ns_id: u32,
    fields: Vec<FieldWasteInfo>,
}

#[derive(Serialize)]
pub struct IndexWasteResponse {
    /// Compaction threshold as a fraction (`0.0..1.0`): a store is compacted at
    /// the next index checkpoint once its waste reaches this.
    threshold: f64,
    namespaces: Vec<NamespaceWasteInfo>,
}

/// `GET /admin/storage/index-waste` — per-field reclaimable dead space in the
/// field-index bitmap and keymap stores, alongside the compaction threshold.
/// Use it to decide whether to force a `POST /admin/storage/index-checkpoint`.
pub async fn index_waste(State(state): State<AppState>) -> impl IntoResponse {
    let threshold = state.store.index_blob_waste_threshold();
    let schemas = state.schemas.read().await;
    let mut namespaces: Vec<NamespaceWasteInfo> = schemas
        .iter()
        .filter_map(|(name, schema)| {
            let ns_id = schema.ns_id?;
            // The db registry retains dropped fields (for field_id reuse), so use
            // the schema's current `indices` as the source of truth for which
            // indexes are still active — otherwise a dropped index lingers here.
            let active: std::collections::HashSet<&str> = schema.indices.iter().map(|s| s.field.as_str()).collect();
            let fields = state
                .store
                .list_index_fields(ns_id)
                .into_iter()
                .filter(|m| active.contains(m.field_name.as_str()))
                .map(|m| {
                    let waste = state.store.field_index_waste(name, &m.field_name);
                    FieldWasteInfo {
                        field_id: m.field_id,
                        field_type: format!("{:?}", m.field_type),
                        bitmap_waste_ratio: waste.map(|(b, _)| b),
                        keymap_waste_ratio: waste.map(|(_, k)| k),
                        over_threshold: waste.is_some_and(|(b, k)| b >= threshold || k >= threshold),
                        distinct_count: state.store.field_index_distinct_count(name, &m.field_name),
                        field_name: m.field_name,
                    }
                })
                .collect();
            Some(NamespaceWasteInfo {
                namespace: name.clone(),
                ns_id,
                fields,
            })
        })
        .collect();
    namespaces.sort_by(|a, b| a.namespace.cmp(&b.namespace));
    Json(IndexWasteResponse { threshold, namespaces })
}

// ── Per-namespace KV store metrics ────────────────────────────────────────────

#[derive(Serialize)]
pub struct AssociatedKvStore {
    name: String,
    purpose: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    lsm: Option<LsmNamespaceInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    value_log: Option<ValueLogNamespaceInfo>,
}

#[derive(Serialize)]
pub struct DocStoreKvMeta {
    namespace: String,
    ns_id: u32,
    semantic_search_enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    lsm: Option<LsmNamespaceInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    value_log: Option<ValueLogNamespaceInfo>,
    associated_stores: Vec<AssociatedKvStore>,
}

fn companion_purpose(suffix: &str) -> String {
    match suffix {
        "vectors" => "quantised embeddings (cluster_id ‖ doc_id → VectorIndex)".to_owned(),
        "vec_meta" => "embedding cluster-id lookup (doc_id → cluster_id, used for deletion)".to_owned(),
        other => format!("companion KV store (suffix: {other})"),
    }
}

pub fn build_namespace_meta(
    store: &minnal_db::DocStore,
    ns: &str,
    doc_names: &std::collections::HashSet<String>,
    kv_names: &std::collections::HashSet<String>,
) -> (Option<LsmNamespaceInfo>, Option<ValueLogNamespaceInfo>, Vec<AssociatedKvStore>) {
    let kv_set: std::collections::HashSet<String> = store.list_kv_namespaces().into_iter().map(|(n, _)| n).collect();
    let mut lsm_map: HashMap<String, minnal_db::LsmManifest> = store.lsm_manifests().into_iter().collect();
    let mut vlog_map: HashMap<String, Vec<(u32, ValueLogMetadata)>> = store.value_log_shard_stats().into_iter().collect();

    let lsm_info = lsm_map.remove(ns).map(|m| build_lsm_namespace_info(ns.to_owned(), m));
    let vlog_info = vlog_map.remove(ns).map(|b| build_vlog_namespace_info(ns.to_owned(), b));

    let companion_prefix = format!("{ns}_");
    let mut companion_names: Vec<String> = kv_set
        .iter()
        .filter(|name| name.starts_with(&companion_prefix) && !doc_names.contains(*name) && !kv_names.contains(*name) && !is_system_ns(name))
        .cloned()
        .collect();
    companion_names.sort();

    let associated_stores: Vec<AssociatedKvStore> = companion_names
        .into_iter()
        .map(|name| {
            let suffix = &name[companion_prefix.len()..];
            let purpose = companion_purpose(suffix);
            let lsm = lsm_map.remove(&name).map(|m| build_lsm_namespace_info(name.clone(), m));
            let vlog = vlog_map.remove(&name).map(|b| build_vlog_namespace_info(name.clone(), b));
            AssociatedKvStore {
                name,
                purpose,
                lsm,
                value_log: vlog,
            }
        })
        .collect();

    (lsm_info, vlog_info, associated_stores)
}

#[derive(Serialize)]
pub struct KvStoreKvMeta {
    namespace: String,
    ns_id: u32,
    key_type: String,
    value_type: String,
    semantic_search_enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    lsm: Option<LsmNamespaceInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    value_log: Option<ValueLogNamespaceInfo>,
    associated_stores: Vec<AssociatedKvStore>,
}

/// `GET /admin/storage/stores/{ns}/kv-meta` — engine-level (LSM + value-log)
/// storage metadata for one store of **either** kind, plus its companion KV
/// stores. The kind is resolved from the store's `store_type`; the KV-store
/// response additionally carries `key_type`/`value_type`.
pub async fn store_kv_meta(
    State(state): State<AppState>,
    Path(ns): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let not_found = || {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": format!("store '{}' does not exist", ns) })),
        )
    };

    match state.store.store_type(&ns) {
        Ok(StoreType::Doc) => {
            let (ns_id, semantic_search_enabled, doc_names, kv_names) = {
                let schemas = state.schemas.read().await;
                let s = schemas.get(&ns).filter(|s| s.ns_id.is_some()).ok_or_else(not_found)?;
                let doc_names: std::collections::HashSet<String> = schemas.keys().cloned().collect();
                let kv_names: std::collections::HashSet<String> = state.kv_schemas.read().await.keys().cloned().collect();
                (s.ns_id.unwrap(), s.semantic_search_enabled, doc_names, kv_names)
            };
            let (lsm_info, vlog_info, associated_stores) = build_namespace_meta(&state.store, &ns, &doc_names, &kv_names);
            Ok(Json(serde_json::json!(DocStoreKvMeta {
                namespace: ns,
                ns_id,
                semantic_search_enabled,
                lsm: lsm_info,
                value_log: vlog_info,
                associated_stores,
            })))
        }
        Ok(StoreType::Kv) => {
            let (ns_id, key_type, value_type, semantic_search_enabled, doc_names, kv_names) = {
                let kv_schemas = state.kv_schemas.read().await;
                let s = kv_schemas.get(&ns).filter(|s| s.ns_id.is_some()).ok_or_else(not_found)?;
                let doc_names: std::collections::HashSet<String> = state.schemas.read().await.keys().cloned().collect();
                let kv_names: std::collections::HashSet<String> = kv_schemas.keys().cloned().collect();
                (
                    s.ns_id.unwrap(),
                    kv_key_type_str(s.key_type),
                    kv_value_type_str(s.value_type),
                    s.semantic_search_enabled,
                    doc_names,
                    kv_names,
                )
            };
            let (lsm_info, vlog_info, associated_stores) = build_namespace_meta(&state.store, &ns, &doc_names, &kv_names);
            Ok(Json(serde_json::json!(KvStoreKvMeta {
                namespace: ns,
                ns_id,
                key_type,
                value_type,
                semantic_search_enabled,
                lsm: lsm_info,
                value_log: vlog_info,
                associated_stores,
            })))
        }
        Err(_) => Err(not_found()),
    }
}

// ── Type helpers ──────────────────────────────────────────────────────────────

pub fn kv_key_type_str(t: KvKeyType) -> String {
    match t {
        KvKeyType::Str => "str".to_owned(),
        KvKeyType::Int => "int".to_owned(),
    }
}

pub fn kv_value_type_str(t: KvValueType) -> String {
    match t {
        KvValueType::Int => "int".to_owned(),
        KvValueType::Str => "str".to_owned(),
        KvValueType::F32 => "f32".to_owned(),
        KvValueType::VecF32 => "vec_f32".to_owned(),
    }
}

pub fn is_system_ns(name: &str) -> bool {
    name == minnal_db::SYSTEM_NAMESPACE || name.starts_with("system_")
}

fn system_ns_purpose(name: &str) -> &'static str {
    match name {
        "system" => "system namespace root (reserved; never stores user data)",
        "system_qemb_cache" => "query-embedding cache shared across all semantic-search namespaces (TTL configurable; default 1 day)",
        _ => "internal system namespace",
    }
}

// ── System namespace stores ───────────────────────────────────────────────────

#[derive(Serialize)]
pub struct SystemKvStoreInfo {
    name: String,
    ns_id: u32,
    purpose: &'static str,
    ttl_enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    ttl_secs: Option<u64>,
    /// Maximum records deleted per TTL worker run (present only when a TTL is configured).
    #[serde(skip_serializing_if = "Option::is_none")]
    ttl_max_deletes_per_run: Option<u64>,
    lsm_entry_count: u64,
}

#[derive(Serialize)]
pub struct SystemDocStoreInfo {
    namespace: String,
    ns_id: u32,
    semantic_search_enabled: bool,
}

#[derive(Serialize)]
pub struct SystemStoresResponse {
    kv_stores: Vec<SystemKvStoreInfo>,
    doc_stores: Vec<SystemDocStoreInfo>,
}

#[derive(Serialize)]
pub struct SystemStoreMeta {
    name: String,
    ns_id: u32,
    purpose: &'static str,
    ttl_enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    ttl_secs: Option<u64>,
    /// Maximum records deleted per TTL worker run (present only when a TTL is configured).
    #[serde(skip_serializing_if = "Option::is_none")]
    ttl_max_deletes_per_run: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    lsm: Option<LsmNamespaceInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    value_log: Option<ValueLogNamespaceInfo>,
}

pub async fn system_stores(State(state): State<AppState>) -> impl IntoResponse {
    let lsm_counts: HashMap<String, u64> = state
        .store
        .lsm_manifests()
        .into_iter()
        .map(|(ns, m)| {
            let total: u64 = m
                .levels
                .iter()
                .flat_map(|l| l.buckets.iter())
                .flat_map(|b| b.files.iter())
                .map(|f| f.entry_count)
                .sum();
            (ns, total)
        })
        .collect();

    let kv_stores: Vec<SystemKvStoreInfo> = state
        .store
        .list_kv_namespaces()
        .into_iter()
        .filter(|(name, _)| is_system_ns(name))
        .map(|(name, ns_id)| {
            let ttl_cfg = state.store.ttl_config_for_ns(ns_id);
            let ttl_secs = ttl_cfg.map(|(ttl, _)| ttl);
            let ttl_max_deletes_per_run = ttl_cfg.map(|(_, max_del)| max_del as u64);
            let lsm_entry_count = lsm_counts.get(&name).copied().unwrap_or(0);
            SystemKvStoreInfo {
                purpose: system_ns_purpose(&name),
                ttl_enabled: ttl_secs.is_some(),
                ttl_secs,
                ttl_max_deletes_per_run,
                lsm_entry_count,
                name,
                ns_id,
            }
        })
        .collect();

    let schemas = state.schemas.read().await;
    let doc_stores: Vec<SystemDocStoreInfo> = schemas
        .values()
        .filter(|s| is_system_ns(&s.namespace))
        .map(|s| SystemDocStoreInfo {
            namespace: s.namespace.clone(),
            ns_id: s.ns_id.unwrap_or(0),
            semantic_search_enabled: s.semantic_search_enabled,
        })
        .collect();

    Json(SystemStoresResponse { kv_stores, doc_stores })
}

pub async fn system_store_meta(
    State(state): State<AppState>,
    Path(ns): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    if !is_system_ns(&ns) {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": format!("'{}' is not a system namespace", ns)
            })),
        ));
    }

    let kv_map: HashMap<String, u32> = state.store.list_kv_namespaces().into_iter().collect();

    let ns_id = match kv_map.get(&ns) {
        Some(&id) => id,
        None => {
            return Err((
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({
                    "error": format!("system namespace '{}' has not been opened yet", ns)
                })),
            ));
        }
    };

    let lsm_info = state
        .store
        .lsm_manifests()
        .into_iter()
        .find(|(name, _)| name == &ns)
        .map(|(name, m)| build_lsm_namespace_info(name, m));

    let vlog_info = state
        .store
        .value_log_shard_stats()
        .into_iter()
        .find(|(name, _)| name == &ns)
        .map(|(name, buckets)| build_vlog_namespace_info(name, buckets));

    let ttl_cfg = state.store.ttl_config_for_ns(ns_id);
    let ttl_secs = ttl_cfg.map(|(ttl, _)| ttl);
    let ttl_max_deletes_per_run = ttl_cfg.map(|(_, max_del)| max_del as u64);
    Ok(Json(SystemStoreMeta {
        purpose: system_ns_purpose(&ns),
        ttl_enabled: ttl_secs.is_some(),
        ttl_secs,
        ttl_max_deletes_per_run,
        lsm: lsm_info,
        value_log: vlog_info,
        name: ns,
        ns_id,
    }))
}
