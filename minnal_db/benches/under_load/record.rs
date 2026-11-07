//! `engine_bench record CONFIG --upstream URL`: build the replay table once,
//! from the real embedding service, by making the engine itself embed every
//! text a run can send. The engine turns each text into the payloads it sends
//! (whole text plus its sentence windows), so recording what it sends makes
//! the table match a run byte for byte.
//!
//! On a scratch copy of the snapshot (`{table}.engine`, removed afterwards),
//! with the replay in record mode in front of the service:
//!
//! 1. every test query once (the query endpoint);
//! 2. every stable document rewritten with its own text (score-only updates
//!    re-embed it unchanged);
//! 3. every stable document no test query judges relevant, with `" Revised."`
//!    appended (text-changing updates);
//! 4. every stream document (inserts);
//! 5. every stream text with `" (copy)"` appended, under the ids inserts past
//!    the stream use;
//!
//! waiting for the vector queue to drain after each phase. A table that already
//! holds a payload is not asked again, so an interrupted recording resumes.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use minnal_db::doc_store::{DocId, DocStore, Pagination};

use crate::engine::{self, Config};
use crate::replay::Replay;
use crate::workload::{Data, Doc, search_spec};

async fn put_all(store: &Arc<DocStore>, ns: &str, phase: &str, docs: Vec<Doc>) -> Result<(), String> {
    let t = Instant::now();
    let n = docs.len();
    let docs = Arc::new(parking_lot::Mutex::new(docs));
    let workers: Vec<_> = (0..16)
        .map(|_| {
            let (store, ns, docs) = (Arc::clone(store), ns.to_string(), Arc::clone(&docs));
            tokio::spawn(async move {
                loop {
                    let Some(d) = docs.lock().pop() else { return Ok::<(), String>(()) };
                    store
                        .put(&ns, DocId::U64(d.id), d.body())
                        .await
                        .map_err(|e| format!("put {}: {e}", d.id))?;
                }
            })
        })
        .collect();
    for w in workers {
        w.await.map_err(|e| e.to_string())??;
    }
    eprintln!("{phase}: {n} writes in {:.0} s", t.elapsed().as_secs_f64());
    Ok(())
}

async fn drain(store: &DocStore, replay: &Replay, phase: &str) {
    let t = Instant::now();
    while !store.list_queue_entries().await.is_empty() {
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    let s = replay.stats();
    eprintln!(
        "{phase}: queue drained in {:.0} s; table {} vectors; {} requests, {} payloads, {} from the table",
        t.elapsed().as_secs_f64(),
        replay.vectors(),
        s.requests,
        s.payloads,
        s.hits
    );
}

/// A replayed vector must reach the engine as exactly the f32 recorded: ask
/// the replay for a sample over HTTP and parse it as the engine does
/// (`Vec<Vec<f32>>`, `embedding_service.rs`), comparing bits.
async fn verify(replay: &Replay, url: &str) -> Result<(), String> {
    #[derive(serde::Deserialize)]
    struct Resp {
        embeddings: Vec<Vec<f32>>,
    }
    let client = reqwest::Client::new();
    let sample = replay.sample(500);
    for (model, endpoint, dims, text, want) in &sample {
        let resp: Resp = client
            .post(format!("{url}/embedding/{model}/{endpoint}"))
            .json(&serde_json::json!({"payloads": [text], "dimensions": dims}))
            .send()
            .await
            .map_err(|e| e.to_string())?
            .json()
            .await
            .map_err(|e| e.to_string())?;
        let got = resp.embeddings.first().ok_or("verify: empty reply")?;
        if got.len() != want.len() || got.iter().zip(want).any(|(a, b)| a.to_bits() != b.to_bits()) {
            return Err(format!(
                "verify: the replayed vector for {model}/{endpoint} {text:.60} differs from the recorded one"
            ));
        }
    }
    eprintln!("verify: {} replayed vectors identical to the recorded ones, bit for bit", sample.len());
    Ok(())
}

pub async fn record(args: &[String]) -> Result<(), String> {
    let usage = "usage: engine_bench record CONFIG.json --upstream URL";
    let path = args.first().ok_or(usage)?;
    let upstream = args.iter().position(|a| a == "--upstream").and_then(|i| args.get(i + 1)).ok_or(usage)?;
    let cfg = Config::load(path)?;
    let scratch = PathBuf::from(format!("{}.engine", cfg.table));
    if scratch.exists() {
        std::fs::remove_dir_all(&scratch).map_err(|e| format!("{}: {e}", scratch.display()))?;
    }
    engine::copy_snapshot(&cfg.snapshot, &scratch)?;
    engine::log_to(&PathBuf::from(format!("{}.log", cfg.table)))?;
    let replay = Replay::open(&cfg.table, Some(upstream))?;
    eprintln!("table {}: {} vectors already recorded", cfg.table, replay.vectors());
    let url = replay.serve().await?;
    // 64 documents at a time: how many are embedded at once, not what is
    // sent. The service is round-trip bound, not GPU bound, at 16.
    let (store, _) = engine::open(&cfg, &scratch, &url, Some(64)).await?;
    let data = Data::load(&cfg.data)?;
    let ns = cfg.namespace.as_str();

    drain(&store, &replay, "start").await;
    for (qid, text) in &data.queries {
        store
            .search_semantic(ns, text, &search_spec(), Pagination::new(1, 100))
            .await
            .map_err(|e| format!("query {qid}: {e}"))?;
    }
    drain(&store, &replay, "queries").await;
    let own = data.stable.iter().map(|i| data.base[i].clone()).collect();
    put_all(&store, ns, "stable, own text", own).await?;
    drain(&store, &replay, "stable, own text").await;
    let revised = data
        .stable_text
        .iter()
        .map(|i| {
            let d = &data.base[i];
            d.with_text(format!("{} Revised.", d.text))
        })
        .collect();
    put_all(&store, ns, "stable, revised", revised).await?;
    drain(&store, &replay, "stable, revised").await;
    let len = data.stream.len() as u64;
    put_all(&store, ns, "stream", (0..len).map(|n| data.stream_doc(n)).collect()).await?;
    drain(&store, &replay, "stream").await;
    put_all(&store, ns, "stream copies", (len..2 * len).map(|n| data.stream_doc(n)).collect()).await?;
    drain(&store, &replay, "stream copies").await;

    store.shutdown().await.map_err(|e| format!("shutdown: {e}"))?;
    drop(store);
    verify(&replay, &url).await?;
    std::fs::remove_dir_all(&scratch).map_err(|e| format!("{}: {e}", scratch.display()))?;
    eprintln!(
        "table {}: {} vectors, {:.0} MiB",
        cfg.table,
        replay.vectors(),
        std::fs::metadata(&cfg.table).map_or(0, |m| m.len()) as f64 / (1 << 20) as f64
    );
    Ok(())
}
