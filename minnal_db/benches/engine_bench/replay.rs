//! Embedding replay: answers the embedding service's API from a table of
//! embeddings recorded once, in this process, so a run measures the engine and
//! makes no external call. The engine's own embedding client talks to it over
//! localhost exactly as it would to the service.
//!
//! API (`minnal_db/src/semantic_search/service/embedding_service.rs`):
//! `POST /embedding/{model}/{document|query}` with `{"payloads": [...],
//! "dimensions": N}` returns `{"embeddings": [[...], ...]}`; `GET /healthcheck`
//! lists the models as `ok`.
//!
//! - **Replay**: answers only from the table. A payload missing from it fails
//!   the request (HTTP 500) and is counted; a cell with misses is invalid.
//! - **Record**: forwards payloads missing from the table to the real service
//!   and appends what it returns.
//!
//! Table: an append-only file of records
//! `[key_len u32][key: model \0 endpoint \0 dims \0 text][dims u32][dims x f32]`,
//! little-endian; vectors are stored as the f32 the engine parses.

use std::collections::{BTreeSet, HashMap};
use std::io::{Read, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::{Mutex, RwLock};
use serde::Deserialize;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

#[derive(Default, Clone, Copy, serde::Serialize)]
pub struct ReplayStats {
    pub requests: u64,
    pub payloads: u64,
    pub hits: u64,
    pub misses: u64,
}

impl ReplayStats {
    pub fn since(&self, before: &ReplayStats) -> ReplayStats {
        ReplayStats {
            requests: self.requests - before.requests,
            payloads: self.payloads - before.payloads,
            hits: self.hits - before.hits,
            misses: self.misses - before.misses,
        }
    }
}

pub struct Replay {
    table: RwLock<HashMap<Vec<u8>, Vec<f32>>>,
    models: RwLock<BTreeSet<String>>,
    /// Record mode: the real service and the table file to append to.
    record: Option<(reqwest::Client, String, Mutex<std::fs::File>)>,
    requests: AtomicU64,
    payloads: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
}

fn key(model: &str, endpoint: &str, dims: usize, text: &str) -> Vec<u8> {
    format!("{model}\0{endpoint}\0{dims}\0{text}").into_bytes()
}

/// Read one table record; `None` at the end of the file or at a record cut
/// short by an interrupted recording (ignored). `buf` is reused for the vector
/// bytes.
fn read_record(r: &mut impl Read, buf: &mut Vec<u8>) -> std::io::Result<Option<(Vec<u8>, Vec<f32>)>> {
    fn exact(r: &mut impl Read, out: &mut [u8]) -> std::io::Result<bool> {
        match r.read_exact(out) {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(false),
            Err(e) => Err(e),
        }
    }
    let mut n = [0u8; 4];
    if !exact(r, &mut n)? {
        return Ok(None);
    }
    let mut k = vec![0u8; u32::from_le_bytes(n) as usize];
    if !exact(r, &mut k)? || !exact(r, &mut n)? {
        return Ok(None);
    }
    buf.resize(4 * u32::from_le_bytes(n) as usize, 0);
    if !exact(r, buf)? {
        return Ok(None);
    }
    let v = buf.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    Ok(Some((k, v)))
}

impl Replay {
    /// Load `table` (missing: empty). `upstream` set: record mode.
    pub fn open(table: &str, upstream: Option<&str>) -> Result<Arc<Self>, String> {
        let (mut map, mut models) = (HashMap::new(), BTreeSet::new());
        match std::fs::File::open(table) {
            // Streamed a record at a time, so loading never holds the file and
            // the parsed table at once.
            Ok(f) => {
                let mut r = std::io::BufReader::with_capacity(1 << 20, f);
                let mut buf = Vec::new();
                while let Some((k, v)) = read_record(&mut r, &mut buf).map_err(|e| format!("replay table {table}: {e}"))? {
                    if let Some(m) = k.split(|b| *b == 0).next() {
                        models.insert(String::from_utf8_lossy(m).into_owned());
                    }
                    map.insert(k, v);
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound && upstream.is_some() => {}
            Err(e) => return Err(format!("replay table {table}: {e}")),
        }
        let record = match upstream {
            Some(url) => {
                let file = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(table)
                    .map_err(|e| format!("{table}: {e}"))?;
                Some((reqwest::Client::new(), url.trim_end_matches('/').to_string(), Mutex::new(file)))
            }
            None => None,
        };
        Ok(Arc::new(Self {
            table: RwLock::new(map),
            models: RwLock::new(models),
            record,
            requests: AtomicU64::new(0),
            payloads: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }))
    }

    pub fn vectors(&self) -> usize {
        self.table.read().len()
    }

    /// Up to `n` recorded entries: `(model, endpoint, dims, text, vector)`.
    pub fn sample(&self, n: usize) -> Vec<(String, String, usize, String, Vec<f32>)> {
        let t = self.table.read();
        let step = (t.len() / n.max(1)).max(1);
        t.iter()
            .step_by(step)
            .take(n)
            .filter_map(|(k, v)| {
                let s = String::from_utf8_lossy(k);
                let mut p = s.splitn(4, '\0');
                let (m, e, d, text) = (p.next()?, p.next()?, p.next()?.parse().ok()?, p.next()?);
                Some((m.to_string(), e.to_string(), d, text.to_string(), v.clone()))
            })
            .collect()
    }

    pub fn models(&self) -> Vec<String> {
        self.models.read().iter().cloned().collect()
    }

    pub fn stats(&self) -> ReplayStats {
        ReplayStats {
            requests: self.requests.load(Ordering::SeqCst),
            payloads: self.payloads.load(Ordering::SeqCst),
            hits: self.hits.load(Ordering::SeqCst),
            misses: self.misses.load(Ordering::SeqCst),
        }
    }

    /// Listen on a free localhost port; returns the base URL.
    pub async fn serve(self: &Arc<Self>) -> Result<String, String> {
        let listener = TcpListener::bind("127.0.0.1:0").await.map_err(|e| e.to_string())?;
        let url = format!("http://{}", listener.local_addr().map_err(|e| e.to_string())?);
        let me = Arc::clone(self);
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let me = Arc::clone(&me);
                tokio::spawn(async move {
                    let _ = me.connection(stream).await;
                });
            }
        });
        Ok(url)
    }

    /// One keep-alive HTTP/1.1 connection.
    async fn connection(&self, mut stream: TcpStream) -> std::io::Result<()> {
        let mut buf: Vec<u8> = Vec::with_capacity(64 * 1024);
        loop {
            let head_end = loop {
                if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                    break p + 4;
                }
                if stream.read_buf(&mut buf).await? == 0 {
                    return Ok(());
                }
            };
            let head = String::from_utf8_lossy(&buf[..head_end]).into_owned();
            let mut lines = head.split("\r\n");
            let mut request = lines.next().unwrap_or("").split(' ');
            let (method, path) = (request.next().unwrap_or("").to_string(), request.next().unwrap_or("").to_string());
            let len = lines
                .filter_map(|l| l.split_once(':'))
                .find(|(k, _)| k.trim().eq_ignore_ascii_case("content-length"))
                .and_then(|(_, v)| v.trim().parse::<usize>().ok())
                .unwrap_or(0);
            while buf.len() < head_end + len {
                if stream.read_buf(&mut buf).await? == 0 {
                    return Ok(());
                }
            }
            let body = buf[head_end..head_end + len].to_vec();
            buf.drain(..head_end + len);
            let (status, reply) = self.handle(&method, &path, &body).await;
            let reason = if status == 200 { "OK" } else { "Error" };
            let mut out = format!(
                "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                reply.len()
            )
            .into_bytes();
            out.extend_from_slice(reply.as_bytes());
            stream.write_all(&out).await?;
        }
    }

    async fn handle(&self, method: &str, path: &str, body: &[u8]) -> (u16, String) {
        if method == "GET" && path == "/healthcheck" {
            let mut models = self.models();
            if models.is_empty() {
                models.push("gemma".into());
            }
            let m: serde_json::Map<String, serde_json::Value> = models.into_iter().map(|m| (m, serde_json::json!({"status": "ok"}))).collect();
            return (200, serde_json::json!({"status": "healthy", "models": m}).to_string());
        }
        let parts: Vec<&str> = path.trim_matches('/').split('/').collect();
        if method != "POST" || parts.len() != 3 || parts[0] != "embedding" || !matches!(parts[2], "document" | "query") {
            return (404, format!("{{\"detail\":\"unknown path {path}\"}}"));
        }
        #[derive(Deserialize)]
        struct Req {
            payloads: Vec<String>,
            dimensions: usize,
        }
        let Ok(req) = serde_json::from_slice::<Req>(body) else {
            return (400, "{\"detail\":\"bad body\"}".into());
        };
        let (model, endpoint) = (parts[1], parts[2]);
        let keys: Vec<Vec<u8>> = req.payloads.iter().map(|p| key(model, endpoint, req.dimensions, p)).collect();
        self.requests.fetch_add(1, Ordering::SeqCst);
        self.payloads.fetch_add(keys.len() as u64, Ordering::SeqCst);
        let mut found: Vec<Option<Vec<f32>>> = {
            let t = self.table.read();
            keys.iter().map(|k| t.get(k).cloned()).collect()
        };
        let missing: Vec<usize> = (0..keys.len()).filter(|i| found[*i].is_none()).collect();
        self.hits.fetch_add((keys.len() - missing.len()) as u64, Ordering::SeqCst);
        if !missing.is_empty() {
            let Some((client, upstream, file)) = &self.record else {
                self.misses.fetch_add(missing.len() as u64, Ordering::SeqCst);
                return (
                    500,
                    format!(
                        "{{\"detail\":\"replay miss: {} of {} payloads not in the table\"}}",
                        missing.len(),
                        keys.len()
                    ),
                );
            };
            #[derive(Deserialize)]
            struct Resp {
                embeddings: Vec<Vec<f32>>,
            }
            let payloads: Vec<&String> = missing.iter().map(|i| &req.payloads[*i]).collect();
            let sent = client
                .post(format!("{upstream}{path}"))
                .json(&serde_json::json!({"payloads": payloads, "dimensions": req.dimensions}))
                .send()
                .await;
            let resp = match sent {
                Ok(r) if r.status().is_success() => r.json::<Resp>().await,
                Ok(r) => return (r.status().as_u16(), r.text().await.unwrap_or_default()),
                Err(e) => return (502, format!("{{\"detail\":\"upstream: {e}\"}}")),
            };
            let Ok(resp) = resp else {
                return (502, "{\"detail\":\"upstream: bad body\"}".into());
            };
            let mut t = self.table.write();
            let mut f = file.lock();
            for (i, v) in missing.iter().zip(resp.embeddings) {
                let k = &keys[*i];
                if !t.contains_key(k) {
                    let mut rec = Vec::with_capacity(8 + k.len() + 4 * v.len());
                    rec.extend_from_slice(&(k.len() as u32).to_le_bytes());
                    rec.extend_from_slice(k);
                    rec.extend_from_slice(&(v.len() as u32).to_le_bytes());
                    v.iter().for_each(|x| rec.extend_from_slice(&x.to_le_bytes()));
                    if let Err(e) = f.write_all(&rec) {
                        return (500, format!("{{\"detail\":\"table write: {e}\"}}"));
                    }
                    t.insert(k.clone(), v.clone());
                    self.models.write().insert(model.to_string());
                }
                found[*i] = Some(v);
            }
        }
        // Serialised from f32 directly (shortest form that parses back to the
        // same f32); `json!` would widen each value to f64 first.
        #[derive(serde::Serialize)]
        struct Out {
            embeddings: Vec<Vec<f32>>,
        }
        let embeddings = found.into_iter().map(|v| v.unwrap_or_default()).collect();
        (200, serde_json::to_string(&Out { embeddings }).unwrap_or_default())
    }
}
