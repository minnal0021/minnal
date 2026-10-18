//! `engine_bench quality CONFIG DB_DIR OUT_DIR`: where search quality is lost
//! on a database (a snapshot, or a finished run's `engine`). On a scratch copy,
//! with embeddings replayed, the test queries run three ways:
//!
//! - `default`: the namespace's search settings, as a run searches;
//! - `all`: every posting probed, so no relevant chunk is missed by routing;
//! - `all-wide`: every posting probed and 10,000 Pass-1 candidates, so neither
//!   routing nor the Pass-1 cut can miss one;
//!
//! then the partition is rebuilt from its codes and the three run again. One
//! TREC file per pass in OUT_DIR, plus `quality.json` (the rebuild's report and
//! each pass's time). nDCG: `minnal_db/docs/benchmarks/engine/quality.py`.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use minnal_db::doc_store::vector_settings::{MAX_FIRST_PASS_TOP_K, MAX_PROBE_BUDGET_ENTRIES, MAX_PROBES, SearchSpec};
use minnal_db::doc_store::{DocStore, Pagination};
use minnal_db::ivf_split::rebuild::RebuildState;
use serde_json::json;

use crate::engine::{self, Config};
use crate::replay::Replay;
use crate::runner::write_trec;
use crate::workload::{Data, search_spec};

fn specs() -> [(&'static str, SearchSpec); 3] {
    let all = SearchSpec {
        probe_budget_entries: Some(MAX_PROBE_BUDGET_ENTRIES),
        min_probes: Some(MAX_PROBES),
        max_probes: Some(MAX_PROBES),
        ..search_spec()
    };
    let wide = SearchSpec {
        first_pass_top_k: Some(MAX_FIRST_PASS_TOP_K),
        ..all
    };
    [("default", search_spec()), ("all", all), ("all-wide", wide)]
}

async fn pass(store: &DocStore, ns: &str, data: &Data, spec: &SearchSpec) -> Result<BTreeMap<String, Vec<(u64, f32)>>, String> {
    let mut trec = BTreeMap::new();
    for (qid, text) in &data.queries {
        let page = store
            .search_semantic(ns, text, spec, Pagination::new(1, 100))
            .await
            .map_err(|e| format!("query {qid}: {e}"))?;
        let hits = page
            .results
            .into_iter()
            .filter_map(|r| Some((u64::from_be_bytes(r.document_id.as_slice().try_into().ok()?), r.dot_product)))
            .collect();
        trec.insert(qid.clone(), hits);
    }
    Ok(trec)
}

pub async fn quality(args: &[String]) -> Result<(), String> {
    let usage = "usage: engine_bench quality CONFIG.json DB_DIR OUT_DIR";
    let (Some(path), Some(db_dir), Some(out)) = (args.first(), args.get(1), args.get(2)) else {
        return Err(usage.into());
    };
    let cfg = Config::load(path)?;
    let out = PathBuf::from(out);
    let scratch = out.join("engine");
    if scratch.exists() {
        return Err(format!("{} exists; use a fresh OUT_DIR", scratch.display()));
    }
    engine::copy_snapshot(db_dir, &scratch)?;
    engine::log_to(&out.join("engine.log"))?;
    let replay = Replay::open(&cfg.table, None)?;
    let url = replay.serve().await?;
    let (store, _) = engine::open(&cfg, &scratch, &url, None).await?;
    let data = Data::load(&cfg.data)?;
    let ns = cfg.namespace.as_str();

    while !store.list_queue_entries().await.is_empty() {
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    let mut passes = serde_json::Map::new();
    let mut rebuild = serde_json::Value::Null;
    for stage in ["before", "rebuilt"] {
        if stage == "rebuilt" {
            let t = Instant::now();
            store.start_vector_rebuild(ns, None).map_err(|e| format!("rebuild: {e}"))?;
            let status = loop {
                tokio::time::sleep(Duration::from_secs(1)).await;
                match store.vector_rebuild_status(ns) {
                    Some(s) if s.state != RebuildState::Running => break s,
                    _ => {}
                }
            };
            if status.state != RebuildState::Done {
                return Err(format!("rebuild failed: {:?}", status.error));
            }
            rebuild = json!({"report": status.report, "seconds": t.elapsed().as_secs_f64()});
            eprintln!("rebuild: {rebuild}");
        }
        for (name, spec) in specs() {
            let t = Instant::now();
            let trec = pass(&store, ns, &data, &spec).await?;
            let run = format!("{stage}-{name}");
            write_trec(&out.join(format!("{run}.trec")), &trec, &run)?;
            let secs = t.elapsed().as_secs_f64();
            eprintln!("{run}: {} queries in {secs:.1} s", trec.len());
            passes.insert(run, json!({"seconds": secs, "queries": trec.len()}));
        }
    }
    let misses = replay.stats().misses;
    std::fs::write(
        out.join("quality.json"),
        serde_json::to_vec_pretty(&json!({"db": db_dir, "passes": passes, "rebuild": rebuild, "replay_misses": misses}))
            .map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    store.shutdown().await.map_err(|e| format!("shutdown: {e}"))?;
    drop(store);
    std::fs::remove_dir_all(&scratch).map_err(|e| format!("{}: {e}", scratch.display()))?;
    if misses > 0 {
        return Err(format!("{misses} replay misses: the passes above are not valid"));
    }
    Ok(())
}
