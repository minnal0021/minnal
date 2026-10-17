//! `mixed_load` — one step of the M5 system benchmark: a mixed REST workload
//! (CRUD, scans, field queries, semantic search) against a running server, at a
//! fixed offered rate (open loop) or with N clients (closed loop).
//!
//! # Usage
//!
//! ```text
//! minnal_tools mixed_load <step.json>
//! ```
//!
//! `step.json` describes the step (all fields required unless noted):
//!
//! | Field | Meaning |
//! |---|---|
//! | `url`, `namespace` | Server and document store |
//! | `data_dir` | `base.jsonl`, `stream.jsonl`, `ids.json`, `queries.jsonl` (`work/bench/analysis/m5/prepare.py`) |
//! | `out_dir` | Where `summary.json`, `timeline.jsonl`, `search.trec` and `lag.json` go |
//! | `mix` | Share of each class: `get`, `insert`, `update`, `delete`, `scan`, `field_query`, `search`, `filtered_search` |
//! | `rate` | Offered requests per second (open loop) |
//! | `closed` | Optional: run N clients in closed loop instead of a rate |
//! | `duration_s` | Length of the step (its longest, with `min_samples_per_class`) |
//! | `min_samples_per_class` | Optional: end the step as soon as every class with a share has this many completed requests (and `min_duration_s` has passed). A p99 from n samples is known to within a rank range of `±1.96·√(0.0099·n)`: at 1,000 samples, the 98.4th to 99.6th percentile. So a step stops once its rarest class is measured that well, instead of running a fixed time that leaves rare classes short at low rates and over-samples every class at high ones. Rates in the summary are over the time the step actually ran |
//! | `min_duration_s` | Optional: the shortest a step with `min_samples_per_class` runs (default 0), so periodic background work (WAL GC ticks, flushes, maintenance) falls inside it |
//! | `slo_p99_ms` | Optional, with `min_samples_per_class`: `class → p99 target (ms)`. The step then ends early only on a clear pass, every class's p99 so far within its target; a step showing a miss runs its full `duration_s`, so a fail is always judged on the full step. (Near the saturation point the first minutes of a step run slower than the rest, so an early stop is pessimistic: it could fail a step that passes over the full length, never pass one that fails.) |
//! | `seed` | Seeds the schedule: the same seed sends the same requests at the same times |
//! | `state_file` | Insert and delete cursors, carried from step to step |
//! | `server_pid` | Optional: sample the server's CPU, memory, I/O and open files |
//! | `poll_interval_s` | Admin and resource polling (default 5) |
//! | `lag_sample_every` | Measure ingestion lag on every Nth insert (default 20) |
//! | `search_cold_share` | Share of searches given a unique suffix, so they miss the query-embedding cache (default 0.3) |
//! | `trace` | Optional: also write `trace.tsv`, one line per request: class, scheduled start (ms since the step began, and Unix ms), latency (ms), status |
//! | `search_page_size` | Results per search reply, each with its document (default 100: the top 100, which nDCG@100 needs) |
//!
//! **Open loop.** Arrivals are Poisson at `rate`, precomputed from `seed`. A
//! dispatcher sends each request at its scheduled time whatever is still in
//! flight, and its latency counts **from the scheduled time**, so time spent
//! waiting to be sent is included (no coordinated omission). The summary
//! reports how late the dispatcher was, to show the generator kept up.
//!
//! **Classes.**
//!
//! - `get`: ids drawn Zipf (s = 1.0) over the stable base documents.
//! - `insert`: the ingestion stream in order, then new ids reusing its texts.
//! - `update`: a stable document rewritten whole, half with a new `score`
//!   only, half with changed text (text changes avoid documents a test query
//!   judges relevant, so quality stays comparable).
//! - `delete`: the deletable base documents in order, then inserted ones.
//! - `scan`: one page of 50 from a random starting key.
//! - `field_query`: 1–3 terms over `category`, `year`, `score`.
//! - `search` / `filtered_search`: the FiQA test queries, top 100 (a share made
//!   cache misses); filtered adds `score < S` (5–50% of documents).
//!
//! A `get` or `delete` answered 404 is counted as not found, not as an error.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use reqwest::Client;
use serde_json::{Value, json};
use tokio::sync::Mutex;

const CLASSES: [&str; 8] = ["get", "insert", "update", "delete", "scan", "field_query", "search", "filtered_search"];
/// Requests in flight past which the open loop drops new ones (counted).
const MAX_IN_FLIGHT: usize = 50_000;

fn usage() -> ! {
    eprintln!("usage: minnal_tools mixed_load <step.json>");
    std::process::exit(1);
}

/// SplitMix64: seeded, no dependency.
#[derive(Clone)]
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Zipf over `n` ranks with exponent `s`: a cumulative table, sampled by binary
/// search.
struct Zipf(Vec<f64>);

impl Zipf {
    fn new(n: usize, s: f64) -> Self {
        let mut acc = 0.0;
        let cdf = (1..=n)
            .map(|k| {
                acc += 1.0 / (k as f64).powf(s);
                acc
            })
            .collect::<Vec<_>>();
        let total = acc;
        Self(cdf.into_iter().map(|c| c / total).collect())
    }
    fn sample(&self, rng: &mut Rng) -> usize {
        let u = rng.unit();
        self.0.partition_point(|&c| c < u).min(self.0.len() - 1)
    }
}

#[derive(Clone)]
struct Doc {
    id: u64,
    text: String,
    category: String,
    year: i64,
    score: i64,
}

impl Doc {
    fn from_json(v: &Value) -> Option<Self> {
        Some(Self {
            id: v["id"].as_u64()?,
            text: v["text"].as_str()?.to_string(),
            category: v["category"].as_str()?.to_string(),
            year: v["year"].as_i64()?,
            score: v["score"].as_i64()?,
        })
    }
    fn body(&self) -> Value {
        json!({"text": self.text, "category": self.category, "year": self.year, "score": self.score})
    }
}

struct Data {
    stable: Vec<u64>,
    /// Stable documents no test query judges relevant (text updates).
    stable_text: Vec<u64>,
    deletable: Vec<u64>,
    base: HashMap<u64, Doc>,
    stream: Vec<Doc>,
    queries: Vec<(String, String)>,
}

fn load_data(dir: &str) -> Result<Data, String> {
    let read = |name: &str| std::fs::read_to_string(format!("{dir}/{name}")).map_err(|e| format!("{dir}/{name}: {e}"));
    let docs = |text: String| -> Result<Vec<Doc>, String> {
        text.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                serde_json::from_str::<Value>(l)
                    .ok()
                    .and_then(|v| Doc::from_json(&v))
                    .ok_or_else(|| format!("bad line: {l:.80}"))
            })
            .collect()
    };
    let ids: Value = serde_json::from_str(&read("ids.json")?).map_err(|e| e.to_string())?;
    let list = |k: &str| {
        ids[k]
            .as_array()
            .map(|a| a.iter().filter_map(Value::as_u64).collect::<Vec<u64>>())
            .unwrap_or_default()
    };
    let relevant: std::collections::HashSet<u64> = list("relevant").into_iter().collect();
    let stable = list("stable");
    let stable_text = stable.iter().copied().filter(|i| !relevant.contains(i)).collect();
    let base = docs(read("base.jsonl")?)?.into_iter().map(|d| (d.id, d)).collect();
    let queries = read("queries.jsonl")?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| {
            let v: Value = serde_json::from_str(l).ok()?;
            Some((v["_id"].as_str()?.to_string(), v["text"].as_str()?.to_string()))
        })
        .collect();
    Ok(Data {
        stable,
        stable_text,
        deletable: list("deletable"),
        base,
        stream: docs(read("stream.jsonl")?)?,
        queries,
    })
}

/// One request, fully decided when the schedule is built.
#[derive(Clone)]
enum Op {
    Get(u64),
    Insert(Doc),
    Update(Doc),
    Delete(u64),
    Scan(u64),
    FieldQuery(String),
    Search { qid: String, text: String, cold: bool },
    FilteredSearch { text: String, predicate: String },
}

impl Op {
    fn class(&self) -> usize {
        match self {
            Op::Get(_) => 0,
            Op::Insert(_) => 1,
            Op::Update(_) => 2,
            Op::Delete(_) => 3,
            Op::Scan(_) => 4,
            Op::FieldQuery(_) => 5,
            Op::Search { .. } => 6,
            Op::FilteredSearch { .. } => 7,
        }
    }
}

/// Cursors carried from step to step: the next stream insert and the next
/// delete.
#[derive(Default)]
struct Cursors {
    inserted: u64,
    deleted: u64,
}

struct Generator {
    data: Arc<Data>,
    rng: Rng,
    zipf: Zipf,
    hot: Vec<u64>,
    cdf: Vec<(f64, usize)>,
    cursors: Cursors,
    cold_share: f64,
    seq: u64,
}

impl Generator {
    fn new(data: Arc<Data>, mix: &BTreeMap<String, f64>, seed: u64, cursors: Cursors, cold_share: f64) -> Self {
        let mut rng = Rng(seed);
        // A random hot set: Zipf ranks over a seeded shuffle of the stable ids.
        let mut hot = data.stable.clone();
        for i in (1..hot.len()).rev() {
            let j = rng.below(i + 1);
            hot.swap(i, j);
        }
        let total: f64 = CLASSES.iter().map(|c| mix.get(*c).copied().unwrap_or(0.0)).sum();
        let mut acc = 0.0;
        let cdf = CLASSES
            .iter()
            .enumerate()
            .filter_map(|(k, c)| {
                let w = mix.get(*c).copied().unwrap_or(0.0);
                (w > 0.0).then(|| {
                    acc += w / total;
                    (acc, k)
                })
            })
            .collect();
        Self {
            data,
            zipf: Zipf::new(hot.len(), 1.0),
            hot,
            rng,
            cdf,
            cursors,
            cold_share,
            seq: seed << 20,
        }
    }

    fn stream_doc(&self, n: u64) -> Doc {
        let s = &self.data.stream;
        let mut d = s[(n % s.len() as u64) as usize].clone();
        if n >= s.len() as u64 {
            // Past the stream: new ids, the stream's texts again.
            d.id = 10_000_000_000 + n;
            d.text = format!("{} (copy {})", d.text, n / s.len() as u64);
        }
        d
    }

    fn next(&mut self) -> Op {
        let u = self.rng.unit();
        let class = self.cdf.iter().find(|(c, _)| u < *c).map_or(self.cdf.last().map_or(0, |c| c.1), |c| c.1);
        self.seq += 1;
        match class {
            0 => Op::Get(self.hot[self.zipf.sample(&mut self.rng)]),
            1 => {
                let d = self.stream_doc(self.cursors.inserted);
                self.cursors.inserted += 1;
                Op::Insert(d)
            }
            2 => {
                let text_change = self.rng.unit() < 0.5;
                let data = Arc::clone(&self.data);
                let pool = if text_change { &data.stable_text } else { &data.stable };
                let id = pool[self.rng.below(pool.len())];
                let mut d = self.data.base[&id].clone();
                if text_change {
                    d.text = format!("{} Revised {}.", d.text, self.seq);
                } else {
                    d.score = self.rng.below(1001) as i64;
                }
                Op::Update(d)
            }
            3 => {
                let n = self.cursors.deleted;
                let data = Arc::clone(&self.data);
                let del = &data.deletable;
                let id = if (n as usize) < del.len() {
                    del[n as usize]
                } else {
                    // Then the inserted documents, oldest first, never ahead of
                    // the inserts.
                    let k = n - del.len() as u64;
                    if k + 100 > self.cursors.inserted {
                        return Op::Get(self.hot[self.zipf.sample(&mut self.rng)]);
                    }
                    self.stream_doc(k).id
                };
                self.cursors.deleted += 1;
                Op::Delete(id)
            }
            4 => Op::Scan(self.data.stable[self.rng.below(self.data.stable.len())]),
            5 => {
                let mut terms = Vec::new();
                let n = 1 + self.rng.below(3);
                let mut kinds = [0usize, 1, 2];
                for i in (1..3).rev() {
                    let j = self.rng.below(i + 1);
                    kinds.swap(i, j);
                }
                for &k in &kinds[..n] {
                    terms.push(match k {
                        0 => format!("category = \"c{}\"", self.rng.below(12)),
                        1 => format!("year >= {}", 2010 + self.rng.below(16)),
                        _ => format!("score < {}", 100 + self.rng.below(900)),
                    });
                }
                Op::FieldQuery(terms.join(" AND "))
            }
            6 | 7 => {
                let data = Arc::clone(&self.data);
                let (qid, text) = &data.queries[self.rng.below(data.queries.len())];
                let cold = self.rng.unit() < self.cold_share;
                let text = if cold { format!("{text} [m5 {}]", self.seq) } else { text.clone() };
                if class == 6 {
                    Op::Search {
                        qid: qid.clone(),
                        text,
                        cold,
                    }
                } else {
                    Op::FilteredSearch {
                        text,
                        predicate: format!("score < {}", 50 + self.rng.below(451)),
                    }
                }
            }
            _ => unreachable!(),
        }
    }
}

/// What one class recorded.
#[derive(Default)]
struct ClassStats {
    latencies_us: Vec<u64>,
    statuses: BTreeMap<String, u64>,
    not_found: u64,
    errors: u64,
}

#[derive(Default)]
struct Shared {
    classes: [ClassStatsCell; 8],
    /// First top-100 reply per test query: `qid → [(doc id, score)]`.
    trec: Mutex<BTreeMap<String, Vec<(String, f64)>>>,
    lags_ms: Mutex<Vec<f64>>,
    /// `(class, start ms, latency ms, status)` per request, when tracing.
    trace: std::sync::Mutex<Vec<(u8, f64, f64, u16)>>,
}

/// One class's stats behind a std mutex (held per request, briefly).
#[derive(Default)]
struct ClassStatsCell(std::sync::Mutex<ClassStats>);

impl ClassStatsCell {
    fn record(&self, latency: Duration, status: Option<u16>, not_found_ok: bool) {
        let mut s = self.0.lock().unwrap();
        s.latencies_us.push(latency.as_micros() as u64);
        match status {
            Some(code) if (200..300).contains(&code) => {}
            Some(404) if not_found_ok => s.not_found += 1,
            Some(code) => {
                s.errors += 1;
                *s.statuses.entry(code.to_string()).or_default() += 1;
            }
            None => {
                s.errors += 1;
                *s.statuses.entry("transport".into()).or_default() += 1;
            }
        }
    }
}

fn hex(id: u64) -> String {
    id.to_be_bytes().iter().map(|b| format!("{b:02x}")).collect()
}

struct Ctx {
    client: Client,
    url: String,
    ns: String,
    shared: Shared,
    lag_every: u64,
    inserts: AtomicU64,
    page: u64,
    tracing: bool,
    origin: Instant,
}

/// Send `op`; returns the HTTP status (None on a transport error).
async fn send(ctx: &Arc<Ctx>, op: &Op) -> Option<u16> {
    let (url, ns, c) = (&ctx.url, &ctx.ns, &ctx.client);
    let resp = match op {
        Op::Get(id) => c.get(format!("{url}/stores/{ns}/docs/{id}")).send().await,
        Op::Insert(d) | Op::Update(d) => c.put(format!("{url}/stores/{ns}/docs/{}", d.id)).json(&d.body()).send().await,
        Op::Delete(id) => c.delete(format!("{url}/stores/{ns}/docs/{id}")).send().await,
        Op::Scan(id) => c.get(format!("{url}/stores/{ns}/docs?start={id}&limit=50")).send().await,
        Op::FieldQuery(p) => {
            c.post(format!("{url}/stores/{ns}/query?page_size=20"))
                .json(&json!({"predicate": p}))
                .send()
                .await
        }
        Op::Search { text, .. } => {
            c.post(format!("{url}/stores/{ns}/semantic-search"))
                .json(&json!({"query": text, "top_k": 100, "page_size": ctx.page}))
                .send()
                .await
        }
        Op::FilteredSearch { text, predicate } => {
            c.post(format!("{url}/stores/{ns}/semantic-search/filtered"))
                .json(&json!({"query": text, "predicate": predicate, "top_k": 100, "page_size": ctx.page}))
                .send()
                .await
        }
    };
    let resp = resp.ok()?;
    let status = resp.status().as_u16();
    let body = resp.bytes().await.ok()?;
    if let Op::Search { qid, cold: false, .. } = op
        && status == 200
    {
        let mut trec = ctx.shared.trec.lock().await;
        if !trec.contains_key(qid)
            && let Ok(v) = serde_json::from_slice::<Value>(&body)
        {
            let hits = v["results"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|h| Some((h["id"].to_string().trim_matches('"').to_string(), h["dot_product"].as_f64()?)))
                        .collect()
                })
                .unwrap_or_default();
            trec.insert(qid.clone(), hits);
        }
    }
    Some(status)
}

/// After an insert is acknowledged, poll its queue entry until it is gone: the
/// time until the document is searchable.
async fn measure_lag(ctx: Arc<Ctx>, id: u64, acked: Instant) {
    let url = format!("{}/admin/indices/{}/vector/queue/{}", ctx.url, ctx.ns, hex(id));
    while acked.elapsed() < Duration::from_secs(1800) {
        match ctx.client.get(&url).send().await {
            Ok(r) if r.status().as_u16() == 404 => {
                ctx.shared.lags_ms.lock().await.push(acked.elapsed().as_secs_f64() * 1e3);
                return;
            }
            _ => tokio::time::sleep(Duration::from_millis(200)).await,
        }
    }
}

async fn run_op(ctx: Arc<Ctx>, op: Op, scheduled: Instant) {
    let status = send(&ctx, &op).await;
    let latency = scheduled.elapsed();
    let class = op.class();
    if ctx.tracing {
        let at = scheduled.duration_since(ctx.origin).as_secs_f64() * 1e3;
        ctx.shared
            .trace
            .lock()
            .unwrap()
            .push((class as u8, at, latency.as_secs_f64() * 1e3, status.unwrap_or(0)));
    }
    ctx.shared.classes[class].record(latency, status, matches!(op, Op::Get(_) | Op::Delete(_)));
    if let (Op::Insert(d), Some(code)) = (&op, status)
        && (200..300).contains(&code)
        && ctx.inserts.fetch_add(1, Ordering::Relaxed).is_multiple_of(ctx.lag_every)
    {
        tokio::spawn(measure_lag(Arc::clone(&ctx), d.id, Instant::now()));
    }
}

/// Server and host resources: CPU ticks, RSS, I/O bytes, open files, GPU.
fn resources(pid: Option<u64>) -> Value {
    let read = |p: &str| std::fs::read_to_string(p).unwrap_or_default();
    let mut out = json!({});
    if let Some(pid) = pid {
        let stat = read(&format!("/proc/{pid}/stat"));
        let fields: Vec<&str> = stat.rsplit_once(')').map(|(_, r)| r.split_whitespace().collect()).unwrap_or_default();
        let ticks = |i: usize| fields.get(i).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
        out["cpu_ticks"] = json!(ticks(11) + ticks(12)); // utime + stime
        let status = read(&format!("/proc/{pid}/status"));
        out["rss_kb"] = json!(
            status
                .lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse::<u64>().ok())
        );
        let io = read(&format!("/proc/{pid}/io"));
        for k in ["read_bytes", "write_bytes", "wchar", "rchar"] {
            out[k] = json!(
                io.lines()
                    .find(|l| l.starts_with(&format!("{k}:")))
                    .and_then(|l| l.split_whitespace().nth(1))
                    .and_then(|v| v.parse::<u64>().ok())
            );
        }
        out["open_files"] = json!(std::fs::read_dir(format!("/proc/{pid}/fd")).map(|d| d.count()).ok());
    }
    let gpu = "/sys/class/drm/card1/device";
    out["gpu_busy_percent"] = json!(read(&format!("{gpu}/gpu_busy_percent")).trim().parse::<u64>().ok());
    out["vram_used"] = json!(read(&format!("{gpu}/mem_info_vram_used")).trim().parse::<u64>().ok());
    out["loadavg"] = json!(read("/proc/loadavg").trim());
    out
}

/// Poll the admin endpoints and resources every `every` until `stop`.
async fn poll(ctx: Arc<Ctx>, every: Duration, pid: Option<u64>, start: Instant, stop: Arc<tokio::sync::Notify>, path: String) {
    use std::io::Write;
    let mut file = std::io::BufWriter::new(std::fs::File::create(path).expect("timeline file"));
    let endpoints = [
        ("ops", "/admin/storage/ops-metrics".to_string()),
        ("lsm", "/admin/storage/lsm".to_string()),
        ("wal", "/admin/storage/wal".to_string()),
        ("vector_index_wal", "/admin/storage/vector-index-wal".to_string()),
        ("value_log", "/admin/storage/value-log".to_string()),
        ("queue", "/admin/indices/vector/queue/summary".to_string()),
        ("partition", format!("/admin/indices/{}/vector/partition", ctx.ns)),
        ("index_health", format!("/admin/indices/{}/health", ctx.ns)),
    ];
    loop {
        let mut snap = json!({"t": start.elapsed().as_secs_f64(), "resources": resources(pid)});
        for (name, path) in &endpoints {
            let v = match ctx.client.get(format!("{}{path}", ctx.url)).send().await {
                Ok(r) => r.json::<Value>().await.unwrap_or(Value::Null),
                Err(_) => Value::Null,
            };
            snap[*name] = v;
        }
        let _ = writeln!(file, "{snap}");
        let _ = file.flush();
        tokio::select! {
            _ = tokio::time::sleep(every) => {}
            _ = stop.notified() => {
                let _ = writeln!(file, "{}", json!({"t": start.elapsed().as_secs_f64(), "resources": resources(pid), "final": true}));
                return;
            }
        }
    }
}

fn percentile(sorted: &[u64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let i = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[i] as f64 / 1e3
}

/// Log-spaced latency histogram (2% buckets): `bucket lower bound in µs → count`.
fn histogram(sorted: &[u64]) -> BTreeMap<u64, u64> {
    let mut h = BTreeMap::new();
    for &v in sorted {
        let b = ((v.max(1) as f64).ln() / 1.02f64.ln()).floor();
        *h.entry(1.02f64.powf(b).round() as u64).or_insert(0) += 1;
    }
    h
}

/// Whether the step may end early: every class in `required` has at least
/// `target` completed requests and, where it has a p99 target, is within it.
fn has_samples(ctx: &Ctx, required: &[(usize, Option<f64>)], target: u64) -> bool {
    required.iter().all(|&(k, slo)| {
        let mut latencies = {
            let s = ctx.shared.classes[k].0.lock().unwrap();
            if (s.latencies_us.len() as u64) < target {
                return false;
            }
            if slo.is_none() {
                return true;
            }
            s.latencies_us.clone()
        };
        latencies.sort_unstable();
        slo.is_none_or(|slo| percentile(&latencies, 0.99) <= slo)
    })
}

pub async fn run(args: &[String]) -> Result<(), String> {
    let [path] = args else { usage() };
    let step: Value =
        serde_json::from_str(&std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?).map_err(|e| format!("{path}: {e}"))?;
    let s = |k: &str| step[k].as_str().map(str::to_string).ok_or_else(|| format!("step.{k} is required"));
    let (url, ns, data_dir, out_dir, state_file) = (s("url")?, s("namespace")?, s("data_dir")?, s("out_dir")?, s("state_file")?);
    let duration = Duration::from_secs_f64(step["duration_s"].as_f64().ok_or("step.duration_s is required")?);
    let min_samples = step["min_samples_per_class"].as_u64();
    let min_duration = Duration::from_secs_f64(step["min_duration_s"].as_f64().unwrap_or(0.0));
    let slo: BTreeMap<String, f64> = match step.get("slo_p99_ms") {
        Some(v) if !v.is_null() => serde_json::from_value(v.clone()).map_err(|e| format!("step.slo_p99_ms: {e}"))?,
        _ => BTreeMap::new(),
    };
    let seed = step["seed"].as_u64().ok_or("step.seed is required")?;
    let closed = step["closed"].as_u64().map(|n| n as usize);
    let rate = step["rate"].as_f64().unwrap_or(0.0);
    if closed.is_none() && rate <= 0.0 {
        return Err("step.rate or step.closed is required".into());
    }
    let mix: BTreeMap<String, f64> = serde_json::from_value(step["mix"].clone()).map_err(|e| format!("step.mix: {e}"))?;
    // The classes a sample-based stop waits for: every one with a share.
    let required: Vec<(usize, Option<f64>)> = CLASSES
        .iter()
        .enumerate()
        .filter(|(_, c)| mix.get(**c).copied().unwrap_or(0.0) > 0.0)
        .map(|(k, c)| (k, slo.get(*c).copied()))
        .collect();
    let pid = step["server_pid"].as_u64();
    let poll_every = Duration::from_secs_f64(step["poll_interval_s"].as_f64().unwrap_or(5.0));
    let cold_share = step["search_cold_share"].as_f64().unwrap_or(0.3);
    std::fs::create_dir_all(&out_dir).map_err(|e| format!("{out_dir}: {e}"))?;

    let data = Arc::new(load_data(&data_dir)?);
    let state: Value = std::fs::read_to_string(&state_file)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or(json!({}));
    let cursors = Cursors {
        inserted: state["inserted"].as_u64().unwrap_or(0),
        deleted: state["deleted"].as_u64().unwrap_or(0),
    };
    let mut generator = Generator::new(Arc::clone(&data), &mix, seed, cursors, cold_share);

    let client = Client::builder()
        .pool_max_idle_per_host(4096)
        .tcp_nodelay(true)
        .timeout(Duration::from_secs(60))
        .build()
        .map_err(|e| e.to_string())?;
    let ctx = Arc::new(Ctx {
        client,
        url,
        ns,
        shared: Shared::default(),
        lag_every: step["lag_sample_every"].as_u64().unwrap_or(20).max(1),
        inserts: AtomicU64::new(0),
        page: step["search_page_size"].as_u64().unwrap_or(100),
        tracing: step["trace"].as_bool().unwrap_or(false),
        origin: Instant::now(),
    });
    let origin_unix_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64() * 1e3);

    let start = Instant::now();
    let stop = Arc::new(tokio::sync::Notify::new());
    let poller = tokio::spawn(poll(
        Arc::clone(&ctx),
        poll_every,
        pid,
        start,
        Arc::clone(&stop),
        format!("{out_dir}/timeline.jsonl"),
    ));

    let in_flight = Arc::new(AtomicUsize::new(0));
    let mut dropped = 0u64;
    let mut max_late = Duration::ZERO;
    let mut late_sum = 0.0f64;
    let mut sent = 0u64;
    let mut tasks = tokio::task::JoinSet::new();
    // How long the step ran: its full duration, or until the samples were in.
    let mut ran = duration;
    let mut stopped_by = "duration";
    if let Some(n) = closed {
        // Closed loop: n clients, each sending its next request when the last
        // returns; latency from the send.
        let ops = Arc::new(Mutex::new(generator));
        let deadline = start + duration;
        let enough = Arc::new(std::sync::atomic::AtomicBool::new(false));
        for _ in 0..n {
            let (ctx, ops, enough) = (Arc::clone(&ctx), Arc::clone(&ops), Arc::clone(&enough));
            tasks.spawn(async move {
                while Instant::now() < deadline && !enough.load(Ordering::Acquire) {
                    let op = ops.lock().await.next();
                    run_op(Arc::clone(&ctx), op, Instant::now()).await;
                }
            });
        }
        if let Some(target) = min_samples {
            // Checked once a second: the clients stop at their next request.
            while !tasks.is_empty() {
                tokio::select! {
                    _ = tasks.join_next() => {}
                    _ = tokio::time::sleep(Duration::from_secs(1)) => {
                        if !enough.load(Ordering::Acquire) && start.elapsed() >= min_duration && has_samples(&ctx, &required, target) {
                            ran = start.elapsed();
                            stopped_by = "samples";
                            enough.store(true, Ordering::Release);
                        }
                    }
                }
            }
        }
        while tasks.join_next().await.is_some() {}
        let generator = Arc::try_unwrap(ops).ok().expect("clients finished").into_inner();
        sent = ctx.shared.classes.iter().map(|c| c.0.lock().unwrap().latencies_us.len() as u64).sum();
        write_state(&state_file, &generator.cursors)?;
    } else {
        // Open loop: Poisson arrivals at `rate`.
        let mut t = 0.0f64;
        let mut next_check = 1.0f64;
        let mut arrivals = Rng(seed ^ 0xa11);
        loop {
            t += -(1.0 - arrivals.unit()).ln() / rate;
            if t >= duration.as_secs_f64() {
                break;
            }
            if let Some(target) = min_samples
                && t >= next_check
            {
                next_check = t + 1.0;
                if t >= min_duration.as_secs_f64() && has_samples(&ctx, &required, target) {
                    ran = Duration::from_secs_f64(t);
                    stopped_by = "samples";
                    break;
                }
            }
            let op = generator.next();
            let scheduled = start + Duration::from_secs_f64(t);
            tokio::time::sleep_until(scheduled.into()).await;
            let late = scheduled.elapsed();
            max_late = max_late.max(late);
            late_sum += late.as_secs_f64();
            sent += 1;
            if in_flight.load(Ordering::Relaxed) >= MAX_IN_FLIGHT {
                dropped += 1;
                continue;
            }
            in_flight.fetch_add(1, Ordering::Relaxed);
            let (ctx, in_flight) = (Arc::clone(&ctx), Arc::clone(&in_flight));
            tasks.spawn(async move {
                run_op(ctx, op, scheduled).await;
                in_flight.fetch_sub(1, Ordering::Relaxed);
            });
            while tasks.try_join_next().is_some() {}
        }
        write_state(&state_file, &generator.cursors)?;
        // Let the requests in flight finish (bounded by the client timeout).
        while tasks.join_next().await.is_some() {}
    }
    let wall = start.elapsed().as_secs_f64();
    // Ingestion-lag pollers still running are given up to 60 s more.
    let lag_deadline = Instant::now() + Duration::from_secs(60);
    while Instant::now() < lag_deadline && Arc::strong_count(&ctx) > 2 {
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    stop.notify_one();
    let _ = poller.await;

    // ── Summary ──
    let mut classes = serde_json::Map::new();
    let mut total_done = 0u64;
    let mut total_errors = 0u64;
    for (k, name) in CLASSES.iter().enumerate() {
        let mut s = ctx.shared.classes[k].0.lock().unwrap();
        s.latencies_us.sort_unstable();
        let n = s.latencies_us.len() as u64;
        total_done += n;
        total_errors += s.errors;
        let l = &s.latencies_us;
        classes.insert(
            name.to_string(),
            json!({
                "count": n,
                "rate": n as f64 / ran.as_secs_f64(),
                "errors": s.errors,
                "not_found": s.not_found,
                "statuses": s.statuses,
                "p50_ms": percentile(l, 0.50),
                "p95_ms": percentile(l, 0.95),
                "p99_ms": percentile(l, 0.99),
                "p999_ms": percentile(l, 0.999),
                "max_ms": l.last().map_or(0.0, |&v| v as f64 / 1e3),
                "histogram_us": histogram(l),
            }),
        );
    }
    let mut lags = ctx.shared.lags_ms.lock().await.clone();
    lags.sort_by(f64::total_cmp);
    let lag_pct = |p: f64| {
        if lags.is_empty() {
            0.0
        } else {
            lags[((lags.len() - 1) as f64 * p).round() as usize]
        }
    };
    let summary = json!({
        "step": step,
        "mode": if closed.is_some() { "closed" } else { "open" },
        "offered_rate": rate,
        "clients": closed,
        "duration_s": ran.as_secs_f64(),
        "max_duration_s": duration.as_secs_f64(),
        "stopped_by": stopped_by,
        "wall_s": wall,
        "sent": sent,
        "completed": total_done,
        "completed_rate": total_done as f64 / ran.as_secs_f64(),
        "errors": total_errors,
        "error_rate": total_errors as f64 / total_done.max(1) as f64,
        "dropped": dropped,
        "dispatcher_late_max_ms": max_late.as_secs_f64() * 1e3,
        "dispatcher_late_mean_ms": late_sum * 1e3 / sent.max(1) as f64,
        "ingestion_lag": {"samples": lags.len(), "p50_ms": lag_pct(0.5), "p99_ms": lag_pct(0.99), "max_ms": lags.last().copied().unwrap_or(0.0)},
        "classes": classes,
    });
    std::fs::write(format!("{out_dir}/summary.json"), serde_json::to_string_pretty(&summary).unwrap()).map_err(|e| e.to_string())?;
    std::fs::write(format!("{out_dir}/lag.json"), serde_json::to_string(&lags).unwrap()).map_err(|e| e.to_string())?;
    let mut trec = String::new();
    for (qid, hits) in ctx.shared.trec.lock().await.iter() {
        for (rank, (doc, score)) in hits.iter().enumerate() {
            trec.push_str(&format!("{qid} Q0 {doc} {} {score} m5\n", rank + 1));
        }
    }
    std::fs::write(format!("{out_dir}/search.trec"), trec).map_err(|e| e.to_string())?;
    if ctx.tracing {
        let mut rows = std::mem::take(&mut *ctx.shared.trace.lock().unwrap());
        rows.sort_by(|a, b| a.1.total_cmp(&b.1));
        let mut out = String::from("class\tstart_ms\tstart_unix_ms\tlatency_ms\tstatus\n");
        for (c, at, lat, st) in rows {
            out.push_str(&format!("{}\t{at:.1}\t{:.1}\t{lat:.2}\t{st}\n", CLASSES[c as usize], origin_unix_ms + at));
        }
        std::fs::write(format!("{out_dir}/trace.tsv"), out).map_err(|e| e.to_string())?;
    }
    println!(
        "{}",
        json!({
            "mode": summary["mode"], "offered_rate": rate, "clients": closed, "completed_rate": summary["completed_rate"],
            "error_rate": summary["error_rate"], "dropped": dropped, "late_max_ms": summary["dispatcher_late_max_ms"],
            "p99_ms": CLASSES.iter().map(|c| (c.to_string(), summary["classes"][c]["p99_ms"].clone())).collect::<serde_json::Map<_, _>>(),
            "lag_p50_ms": lag_pct(0.5),
        })
    );
    Ok(())
}

fn write_state(path: &str, c: &Cursors) -> Result<(), String> {
    std::fs::write(path, json!({"inserted": c.inserted, "deleted": c.deleted}).to_string()).map_err(|e| format!("{path}: {e}"))
}
