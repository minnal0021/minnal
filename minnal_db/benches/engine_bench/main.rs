//! `engine_bench`: what each document-store operation costs in process while
//! the engine is working: every background worker running (flushes,
//! compaction, value-log and WAL GC, field-index checkpoints, vector indexing,
//! partition maintenance) and no external call (embeddings are replayed from a
//! table recorded once).
//!
//! ```text
//! cargo bench -p minnal_db --features doc-store,semantic-search --bench engine_bench -- record CONFIG.json --upstream URL
//! cargo bench -p minnal_db --features doc-store,semantic-search --bench engine_bench -- run CONFIG.json [--smoke]
//! cargo bench -p minnal_db --features doc-store,semantic-search --bench engine_bench -- quality CONFIG.json DB_DIR OUT_DIR
//! ```
//!
//! `record` builds the embedding replay table from the real service (once).
//! `run` runs the plan (see `runner`); `--smoke` runs every cell for a few
//! seconds, to check the harness end to end before a real run. A run resumes
//! from its manifest when started again with the same CONFIG. Reports:
//! `minnal_db/docs/benchmarks/engine/report.py`. `quality` measures where search
//! quality is lost on a database (see `quality`).
//!
//! Without a command it does nothing, so `cargo bench` over every suite (the
//! Criterion report) skips it.
//!
//! The allocator is jemalloc, as in the API server, unless the bench is built
//! with `RUSTFLAGS='--cfg minnal_bench_alloc="system"'` (or `"mimalloc"`), to
//! compare allocators on the same engine. The library sets no allocator.

mod engine;
mod quality;
mod record;
mod replay;
mod runner;
mod stats;
mod workload;

#[cfg(not(any(minnal_bench_alloc = "system", minnal_bench_alloc = "mimalloc")))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[cfg(minnal_bench_alloc = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// The allocator this build uses, recorded in each run.
pub const ALLOCATOR: &str = if cfg!(minnal_bench_alloc = "system") {
    "system"
} else if cfg!(minnal_bench_alloc = "mimalloc") {
    "mimalloc"
} else {
    "jemalloc"
};

fn main() {
    // `cargo bench` passes `--bench` to every bench target.
    let args: Vec<String> = std::env::args().skip(1).filter(|a| a != "--bench").collect();
    let command = args.first().cloned().unwrap_or_default();
    if !matches!(command.as_str(), "run" | "record" | "quality") {
        eprintln!("engine_bench: no command (run | record | quality); skipped. See the module docs.");
        return;
    }
    let runtime = tokio::runtime::Builder::new_multi_thread().enable_all().build().expect("tokio runtime");
    let result = runtime.block_on(async {
        match command.as_str() {
            "run" => runner::run(&args[1..]).await,
            "quality" => quality::quality(&args[1..]).await,
            _ => record::record(&args[1..]).await,
        }
    });
    if let Err(e) = result {
        eprintln!("engine_bench: {e}");
        std::process::exit(2);
    }
}
