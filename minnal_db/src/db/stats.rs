// Statistics about garbage collection
#[derive(Debug, Clone, serde::Serialize)]
#[allow(dead_code)]
pub struct GCStats {
    pub bytes_reclaimed: u64,
    pub bytes_live: u64,
    pub gc_run_count: u64,
    pub total_bytes_reclaimed: u64,
    pub gc_duration_ms: u128,
}

#[derive(Debug, serde::Serialize)]
#[allow(dead_code)]
pub struct Stats {
    pub head: u64,
    pub tail: u64,
    pub garbage_size: u64,
    pub waste_ratio: f64,
    pub free_space_ratio: f64,
    pub total_gc_runs: u64,
    pub total_bytes_reclaimed: u64,
    pub live_bytes: u64,
}
