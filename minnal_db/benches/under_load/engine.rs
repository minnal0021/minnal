//! Opening the engine the way the API server does (`minnal_db_api/src/main.rs`),
//! from the same server TOML, with its embedding URL pointed at the replay.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use minnal_db::ConfigFile;
use minnal_db::config_file::API_SERVER_TABLES;
use minnal_db::doc_store::DocStore;
use serde::{Deserialize, Serialize};

/// The benchmark's configuration file (JSON).
#[derive(Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// The API server's TOML: engine, vector worker and semantic-search settings.
    pub server_toml: String,
    /// A database directory (`db/`) with the base documents loaded
    /// and indexed; copied, never written.
    pub snapshot: String,
    /// `base.jsonl`, `stream.jsonl`, `ids.json`, `queries.jsonl`, `qrels.tsv`.
    pub data: String,
    /// The embedding replay table (`engine_bench record` builds it).
    pub table: String,
    /// Where the run's engine copy, results and manifest go.
    pub out: String,
    pub namespace: String,
    /// The mix the concurrent cells and the soak draw from: `{class: share}`.
    pub mix: BTreeMap<String, f64>,
    /// Writes per second running beside every single-operation cell.
    pub trickle_rate: f64,
    /// The trickle's mix of `insert`, `update`, `delete`.
    pub trickle_mix: BTreeMap<String, f64>,
    pub repeats: u32,
    pub single_s: u64,
    pub concurrent_s: u64,
    pub clients: Vec<usize>,
    pub soak_s: u64,
    pub soak_clients: usize,
    /// Unmeasured load before each cell's measurement.
    pub warm_s: u64,
    /// How long the vector queue may take to drain between cells.
    pub drain_max_s: u64,
    /// Fewest operations a measured class may have.
    pub min_samples: u64,
    /// Measure ingestion lag on every Nth insert of each client.
    pub lag_every: u64,
}

impl Config {
    pub fn load(path: &str) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))
    }

    /// The smoke variant: every cell, short.
    pub fn smoke(mut self) -> Self {
        self.single_s = 10;
        self.concurrent_s = 10;
        self.soak_s = 30;
        self.warm_s = 2;
        self.min_samples = 3;
        self
    }
}

/// What the run applied, for `run.json`.
#[derive(Serialize, Clone)]
pub struct EngineSettings {
    pub vector_worker_concurrency: usize,
    pub vector_worker_max_retries: u32,
    pub maintenance_threads: usize,
    pub rebuild_threads: usize,
    pub supported_models: Vec<(String, u32)>,
    pub server_toml: String,
}

/// Open the doc store at `dir` (`db/`) with every background worker and the
/// vector worker running, configured from the server's TOML as the server
/// configures itself, but embedding through `embed_url`. `concurrency`
/// replaces the vector worker's.
pub async fn open(cfg: &Config, dir: &Path, embed_url: &str, concurrency: Option<usize>) -> Result<(Arc<DocStore>, EngineSettings), String> {
    let (mut file, _) = ConfigFile::from_file(Path::new(&cfg.server_toml), API_SERVER_TABLES).map_err(|e| e.to_string())?;
    file.semantic_search.embedding_service_url = embed_url.to_string();
    if let Some(c) = concurrency {
        file.vector_index.concurrency = c;
    }
    let settings = EngineSettings {
        vector_worker_concurrency: file.vector_index.concurrency,
        vector_worker_max_retries: file.vector_index.max_retries,
        maintenance_threads: file.semantic_search.maintenance.threads,
        rebuild_threads: file.semantic_search.maintenance.rebuild_threads,
        supported_models: file.semantic_search.supported_models(),
        server_toml: cfg.server_toml.clone(),
    };
    let store = DocStore::builder(dir.join("db"))
        .config_file(&file)
        .open()
        .await
        .map_err(|e| format!("open {}: {e}", dir.display()))?;
    Ok((store, settings))
}

/// Copy `snapshot/db` to `dir`.
pub fn copy_snapshot(snapshot: &str, dir: &Path) -> Result<(), String> {
    fn copy(from: &Path, to: &Path) -> std::io::Result<()> {
        std::fs::create_dir_all(to)?;
        for e in std::fs::read_dir(from)? {
            let e = e?;
            let target = to.join(e.file_name());
            if e.file_type()?.is_dir() {
                copy(&e.path(), &target)?;
            } else {
                std::fs::copy(e.path(), target)?;
            }
        }
        Ok(())
    }
    // Schemas live in the database (`system_schema`), so the database is all there is to copy.
    copy(&Path::new(snapshot).join("db"), &dir.join("db")).map_err(|e| format!("copy {snapshot}/db: {e}"))?;
    Ok(())
}

/// Bytes under `dir`.
pub fn disk_bytes(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .map(|it| {
            it.filter_map(Result::ok)
                .map(|e| match e.file_type() {
                    Ok(t) if t.is_dir() => disk_bytes(&e.path()),
                    _ => e.metadata().map_or(0, |m| m.len()),
                })
                .sum()
        })
        .unwrap_or(0)
}

/// This process's resident memory, from `/proc/self/status`.
pub fn rss_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
        })
        .map_or(0, |kb| kb * 1024)
}

/// The process's memory, split so that growth can be attributed: resident
/// pages that are anonymous (heap, stacks) or file-backed (memory-mapped files,
/// which the kernel can drop), and glibc's own account of its heap: what is
/// allocated and not freed, and what is freed but kept (resident until trimmed).
/// The heap figures are zero unless the bench is built with the system allocator
/// on glibc (`--cfg minnal_bench_alloc="system"`).
#[derive(serde::Serialize, Default, Clone, Copy)]
pub struct Memory {
    pub rss_anon_bytes: u64,
    pub rss_file_bytes: u64,
    /// Allocated and not freed (`mallinfo2`: `uordblks + hblkhd`).
    pub heap_live_bytes: u64,
    /// Freed but kept by the allocator (`fordblks`).
    pub heap_free_bytes: u64,
}

pub fn memory() -> Memory {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let kb = |key: &str| {
        status
            .lines()
            .find(|l| l.starts_with(key))
            .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
            .map_or(0, |kb| kb * 1024)
    };
    #[cfg_attr(not(minnal_bench_alloc = "system"), allow(unused_mut))]
    let mut m = Memory {
        rss_anon_bytes: kb("RssAnon:"),
        rss_file_bytes: kb("RssFile:"),
        ..Memory::default()
    };
    #[cfg(all(target_os = "linux", target_env = "gnu", minnal_bench_alloc = "system"))]
    {
        // SAFETY: mallinfo2 takes no arguments and returns a plain struct.
        let i = unsafe { libc::mallinfo2() };
        m.heap_live_bytes = (i.uordblks + i.hblkhd) as u64;
        m.heap_free_bytes = i.fordblks as u64;
    }
    m
}

/// Ask the allocator to return the free memory it keeps (`malloc_trim(0)`);
/// returns the memory before and after.
pub fn trim() -> (Memory, Memory) {
    let before = memory();
    #[cfg(all(target_os = "linux", target_env = "gnu", minnal_bench_alloc = "system"))]
    // SAFETY: malloc_trim only releases free memory back to the system.
    unsafe {
        libc::malloc_trim(0);
    }
    (before, memory())
}

/// The engine's log (it logs through the `log` facade) to a file, INFO and up.
struct FileLog(parking_lot::Mutex<std::io::BufWriter<std::fs::File>>);

impl log::Log for FileLog {
    fn enabled(&self, m: &log::Metadata) -> bool {
        m.level() <= log::Level::Info
    }
    fn log(&self, r: &log::Record) {
        if self.enabled(r.metadata()) {
            let t = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64();
            let mut w = self.0.lock();
            let _ = writeln!(w, "{t:.3} {} {}: {}", r.level(), r.target(), r.args());
            let _ = w.flush();
        }
    }
    fn flush(&self) {
        let _ = self.0.lock().flush();
    }
}

pub fn log_to(path: &Path) -> Result<(), String> {
    let f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let logger: &'static FileLog = Box::leak(Box::new(FileLog(parking_lot::Mutex::new(std::io::BufWriter::new(f)))));
    log::set_logger(logger).map_err(|e| e.to_string())?;
    log::set_max_level(log::LevelFilter::Info);
    Ok(())
}
