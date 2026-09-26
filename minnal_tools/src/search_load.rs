//! `search_load` — closed-loop concurrent load against a semantic-search endpoint.
//!
//! `-c N` workers share one query list and one request counter. Each worker sends
//! a query, waits for the reply, records its latency, and sends the next one, so
//! at most `N` requests are in flight at any time. Prints one JSON summary line
//! (throughput, mean/p50/p95/p99/max latency in ms, error count).
//!
//! # Usage
//!
//! ```text
//! minnal_tools search_load [--kv] [-c N] [--duration SECS | --passes P] [--top-k K]
//!                          [--qids QRELS.tsv] [--cold TAG] [--record RUN] [--trace FILE]
//!                          <url> <namespace> <queries.jsonl>
//! ```
//!
//! `queries.jsonl` holds one object per line with `_id` and `text` fields (the
//! BEIR layout). Flags:
//!
//! | Flag | Effect |
//! |---|---|
//! | `--kv` | Query a KV store (`/stores/{ns}/kv/semantic-search`) rather than a doc store |
//! | `-c N` | Concurrent workers (default 1) |
//! | `--duration SECS` | Run for this long, cycling through the queries |
//! | `--passes P` | Otherwise send every query `P` times (default 1) |
//! | `--top-k K` | `top_k` and `page_size` of each request (default 20) |
//! | `--qids FILE` | Keep only queries whose id is in the first column of this TSV (header skipped), e.g. BEIR `qrels/test.tsv` |
//! | `--cold TAG` | Append ` [TAG-i]` to request `i`, so no query hits the query-embedding cache |
//! | `--record RUN` | Write the first reply to each query as a TREC run file (`qid Q0 id rank score tag`), to diff results under load against a sequential run |
//! | `--trace FILE` | Write `start_ms latency_ms` per request, sorted by start, to find stalled requests |
//!
//! Warm the query-embedding cache with one sequential pass before measuring
//! warm latency; otherwise the first pass includes the embedding call.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use reqwest::Client;
use serde_json::{Value, json};

fn usage() -> ! {
    eprintln!(concat!(
        "usage: minnal_tools search_load [--kv] [-c N] [--duration SECS | --passes P] [--top-k K]\n",
        "                                [--qids QRELS.tsv] [--cold TAG] [--record RUN] [--trace FILE]\n",
        "                                <url> <namespace> <queries.jsonl>",
    ));
    std::process::exit(1);
}

struct Options {
    kv: bool,
    concurrency: usize,
    duration: Option<Duration>,
    passes: usize,
    top_k: usize,
    qids: Option<String>,
    cold: Option<String>,
    record: Option<String>,
    trace: Option<String>,
    positional: Vec<String>,
}

fn parse_args(args: &[String]) -> Result<Options, String> {
    let mut opts = Options {
        kv: false,
        concurrency: 1,
        duration: None,
        passes: 1,
        top_k: 20,
        qids: None,
        cold: None,
        record: None,
        trace: None,
        positional: Vec::new(),
    };
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let mut value = || it.next().cloned().ok_or_else(|| format!("{arg} needs a value"));
        let number = |v: String| v.parse::<usize>().map_err(|e| format!("{arg}: {e}"));
        match arg.as_str() {
            "--kv" => opts.kv = true,
            "-c" => opts.concurrency = number(value()?)?.max(1),
            "--duration" => opts.duration = Some(Duration::from_secs_f64(value()?.parse().map_err(|e| format!("{arg}: {e}"))?)),
            "--passes" => opts.passes = number(value()?)?,
            "--top-k" => opts.top_k = number(value()?)?,
            "--qids" => opts.qids = Some(value()?),
            "--cold" => opts.cold = Some(value()?),
            "--record" => opts.record = Some(value()?),
            "--trace" => opts.trace = Some(value()?),
            flag if flag.starts_with('-') => return Err(format!("unknown flag {flag}")),
            _ => opts.positional.push(arg.clone()),
        }
    }
    if opts.positional.len() != 3 {
        return Err("expected <url> <namespace> <queries.jsonl>".into());
    }
    Ok(opts)
}

/// `(id, text)` for every query, filtered to `qids` when given, sorted by id
/// (numerically when the ids are numbers) so runs are reproducible.
fn load_queries(jsonl: &str, qids: Option<&HashSet<String>>) -> Result<Vec<(String, String)>, String> {
    let mut queries = Vec::new();
    for (line_no, line) in jsonl.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        let v: Value = serde_json::from_str(line).map_err(|e| format!("line {}: {e}", line_no + 1))?;
        let (Some(id), Some(text)) = (v["_id"].as_str(), v["text"].as_str()) else {
            return Err(format!("line {}: needs string `_id` and `text` fields", line_no + 1));
        };
        if qids.is_none_or(|q| q.contains(id)) {
            queries.push((id.to_string(), text.to_string()));
        }
    }
    queries.sort_by(|a, b| match (a.0.parse::<u64>(), b.0.parse::<u64>()) {
        (Ok(x), Ok(y)) => x.cmp(&y),
        _ => a.0.cmp(&b.0),
    });
    Ok(queries)
}

/// The value at percentile `p` (0.0–1.0) of an ascending slice, nearest-rank.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (sorted.len() as f64 * p).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

/// One worker's output: `(start_ms, latency_ms)` per successful request, the
/// error count, and the recorded replies `(request index, qid, hits)`.
type WorkerResult = (Vec<(f64, f64)>, usize, Vec<(usize, String, Vec<(String, f64)>)>);

pub async fn run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let opts = parse_args(args).unwrap_or_else(|e| {
        eprintln!("search_load: {e}");
        usage()
    });
    let (url, ns, queries_path) = (&opts.positional[0], &opts.positional[1], &opts.positional[2]);

    let qids = match &opts.qids {
        Some(path) => Some(
            std::fs::read_to_string(path)?
                .lines()
                .skip(1)
                .filter_map(|l| l.split('\t').next().map(str::to_string))
                .collect::<HashSet<_>>(),
        ),
        None => None,
    };
    let queries = Arc::new(load_queries(&std::fs::read_to_string(queries_path)?, qids.as_ref())?);
    if queries.is_empty() {
        return Err("no queries to send".into());
    }
    let total = queries.len() * opts.passes;
    let endpoint = Arc::new(if opts.kv {
        format!("{url}/stores/{ns}/kv/semantic-search")
    } else {
        format!("{url}/stores/{ns}/semantic-search")
    });

    let client = Client::builder().pool_max_idle_per_host(opts.concurrency).build()?;
    let next = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();
    let deadline = opts.duration.map(|d| start + d);

    let workers: Vec<_> = (0..opts.concurrency)
        .map(|_| {
            let (client, endpoint, next, queries) = (client.clone(), endpoint.clone(), next.clone(), queries.clone());
            let (cold, top_k) = (opts.cold.clone(), opts.top_k);
            tokio::spawn(async move {
                let mut result: WorkerResult = (Vec::new(), 0, Vec::new());
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let done = match deadline {
                        Some(d) => Instant::now() >= d,
                        None => i >= total,
                    };
                    if done {
                        return result;
                    }
                    let (qid, text) = &queries[i % queries.len()];
                    let query = match &cold {
                        Some(tag) => format!("{text} [{tag}-{i}]"),
                        None => text.clone(),
                    };
                    let body = json!({ "query": query, "top_k": top_k, "page_size": top_k });
                    let sent = Instant::now();
                    let reply = match client.post(endpoint.as_str()).json(&body).send().await {
                        Ok(r) if r.status().is_success() => r.json::<Value>().await.map_err(|e| e.to_string()),
                        Ok(r) => Err(format!("HTTP {}", r.status())),
                        Err(e) => Err(e.to_string()),
                    };
                    match reply {
                        Ok(v) => {
                            let (started_ms, took_ms) = (sent.duration_since(start).as_secs_f64() * 1e3, sent.elapsed().as_secs_f64() * 1e3);
                            result.0.push((started_ms, took_ms));
                            if i < queries.len() {
                                let hits = v["results"]
                                    .as_array()
                                    .map(|hits| {
                                        hits.iter()
                                            .map(|h| {
                                                let id = h.get("key").or_else(|| h.get("id")).unwrap_or(&Value::Null);
                                                let id = id.as_str().map_or_else(|| id.to_string(), str::to_string);
                                                (id, h["dot_product"].as_f64().unwrap_or(f64::NAN))
                                            })
                                            .collect()
                                    })
                                    .unwrap_or_default();
                                result.2.push((i, qid.clone(), hits));
                            }
                        }
                        Err(e) => {
                            eprintln!("query {qid}: {e}");
                            result.1 += 1;
                        }
                    }
                }
            })
        })
        .collect();

    let mut timings = Vec::new();
    let mut errors = 0;
    let mut replies = BTreeMap::new();
    for worker in workers {
        let (t, e, r) = worker.await?;
        timings.extend(t);
        errors += e;
        replies.extend(r.into_iter().map(|(i, qid, hits)| (i, (qid, hits))));
    }
    let elapsed = start.elapsed().as_secs_f64();

    if let Some(path) = &opts.trace {
        timings.sort_by(|a, b| a.0.total_cmp(&b.0));
        std::fs::write(path, timings.iter().map(|(s, l)| format!("{s:.2} {l:.2}\n")).collect::<String>())?;
    }
    if let Some(path) = &opts.record {
        let mut out = String::new();
        for (qid, hits) in replies.values() {
            for (rank, (id, score)) in hits.iter().enumerate() {
                out.push_str(&format!("{qid} Q0 {id} {} {score:.6} minnal\n", rank + 1));
            }
        }
        std::fs::write(path, out)?;
    }

    let mut latencies: Vec<f64> = timings.iter().map(|t| t.1).collect();
    latencies.sort_by(f64::total_cmp);
    let mean = latencies.iter().sum::<f64>() / latencies.len().max(1) as f64;
    let ms = |v: f64| (v * 100.0).round() / 100.0;
    println!(
        "{}",
        json!({
            "ns": ns, "c": opts.concurrency, "cold": opts.cold.is_some(), "n": latencies.len(), "errors": errors,
            "elapsed_s": ms(elapsed), "qps": (latencies.len() as f64 / elapsed * 10.0).round() / 10.0,
            "mean": ms(mean), "p50": ms(percentile(&latencies, 0.50)), "p95": ms(percentile(&latencies, 0.95)),
            "p99": ms(percentile(&latencies, 0.99)), "max": ms(percentile(&latencies, 1.0)),
        })
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_is_nearest_rank() {
        let v: Vec<f64> = (1..=100).map(f64::from).collect();
        assert_eq!(percentile(&v, 0.50), 50.0);
        assert_eq!(percentile(&v, 0.99), 99.0);
        assert_eq!(percentile(&v, 1.0), 100.0);
        assert_eq!(percentile(&v, 0.0), 1.0);
        assert_eq!(percentile(&[], 0.5), 0.0);
    }

    #[test]
    fn load_queries_filters_by_qid_and_sorts_numerically() {
        let jsonl = "{\"_id\":\"10\",\"text\":\"ten\"}\n{\"_id\":\"9\",\"text\":\"nine\"}\n\n{\"_id\":\"2\",\"text\":\"two\"}\n";
        let all = load_queries(jsonl, None).unwrap();
        assert_eq!(all.iter().map(|q| q.0.as_str()).collect::<Vec<_>>(), ["2", "9", "10"]);
        let keep: HashSet<String> = ["10".to_string(), "2".to_string()].into();
        let some = load_queries(jsonl, Some(&keep)).unwrap();
        assert_eq!(some, [("2".to_string(), "two".to_string()), ("10".to_string(), "ten".to_string())]);
    }

    #[test]
    fn load_queries_rejects_a_line_without_text() {
        assert!(load_queries("{\"_id\":\"1\"}\n", None).unwrap_err().contains("line 1"));
    }

    #[test]
    fn parse_args_reads_flags_and_positionals() {
        let args: Vec<String> = ["--kv", "-c", "32", "--duration", "15", "http://h:8080", "fiqa", "q.jsonl"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let opts = parse_args(&args).unwrap();
        assert!(opts.kv);
        assert_eq!(opts.concurrency, 32);
        assert_eq!(opts.duration, Some(Duration::from_secs(15)));
        assert_eq!(opts.positional, ["http://h:8080", "fiqa", "q.jsonl"]);
        assert!(parse_args(&args[..5]).is_err(), "missing positionals must be rejected");
        assert!(parse_args(&["--bogus".to_string()]).is_err());
    }
}
