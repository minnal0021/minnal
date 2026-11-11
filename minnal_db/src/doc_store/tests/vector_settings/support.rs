//! The mock embedding service and the helpers the semantic-search tests share.

use super::*;
use crate::semantic_search::service::SemanticSearchConfig;

pub(super) const DIM: usize = 8;

/// One embedding request the mock saw.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Seen {
    pub(super) model: String,
    pub(super) kind: String,
    pub(super) dimensions: usize,
    pub(super) payloads: usize,
}

/// A mock embedding service serving `gemma` and `qwen` at any dimension:
/// one vector per payload, a different one per model, and every request
/// recorded.
pub(super) fn spawn_service() -> (String, std::sync::Arc<parking_lot::Mutex<Vec<Seen>>>) {
    spawn_slow_service(std::time::Duration::ZERO)
}

/// [`spawn_service`], answering each request `delay` after it arrives (one
/// at a time).
pub(super) fn spawn_slow_service(delay: std::time::Duration) -> (String, std::sync::Arc<parking_lot::Mutex<Vec<Seen>>>) {
    spawn_service_with(delay, false)
}

/// [`spawn_service`] whose vectors come from each payload's text, so
/// different texts get different vectors whichever request they arrive in.
/// The plain mock's vectors depend only on a payload's position in its
/// request, so documents embedded one at a time all get the same few.
pub(super) fn spawn_text_service() -> (String, std::sync::Arc<parking_lot::Mutex<Vec<Seen>>>) {
    spawn_service_with(std::time::Duration::ZERO, true)
}

pub(super) fn spawn_service_with(delay: std::time::Duration, vectors_from_text: bool) -> (String, std::sync::Arc<parking_lot::Mutex<Vec<Seen>>>) {
    use std::hash::{Hash, Hasher};
    use std::io::{BufRead, BufReader, Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let seen = std::sync::Arc::new(parking_lot::Mutex::new(Vec::new()));
    let log = seen.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { break };
            let mut reader = BufReader::new(stream);
            let mut line = String::new();
            if reader.read_line(&mut line).is_err() {
                continue;
            }
            let path = line.split_whitespace().nth(1).unwrap_or_default().to_string();
            let mut len = 0usize;
            loop {
                let mut h = String::new();
                if reader.read_line(&mut h).is_err() || h == "\r\n" || h.is_empty() {
                    break;
                }
                if let Some(v) = h.to_ascii_lowercase().strip_prefix("content-length:") {
                    len = v.trim().parse().unwrap_or(0);
                }
            }
            let mut body = vec![0u8; len];
            let _ = reader.read_exact(&mut body);
            let parts: Vec<&str> = path.split('/').collect(); // ["", "embedding", model, kind]
            let (status, out) = if parts.len() == 4 && matches!(parts[2], "gemma" | "qwen") {
                let req: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
                let n = req["payloads"].as_array().map_or(0, |a| a.len());
                let dims = req["dimensions"].as_u64().unwrap_or(0) as usize;
                log.lock().push(Seen {
                    model: parts[2].into(),
                    kind: parts[3].into(),
                    dimensions: dims,
                    payloads: n,
                });
                std::thread::sleep(delay);
                // gemma points along +x, qwen along +y; each payload slightly different.
                let axis = if parts[2] == "gemma" { 0 } else { 1 };
                let embs: Vec<Vec<f32>> = (0..n)
                    .map(|i| {
                        let mut v = vec![0.01f32 * (i as f32 + 1.0); dims];
                        if vectors_from_text {
                            for (d, x) in v.iter_mut().enumerate() {
                                let mut h = std::collections::hash_map::DefaultHasher::new();
                                (req["payloads"][i].to_string(), d).hash(&mut h);
                                *x = (h.finish() % 1000) as f32 / 1000.0;
                            }
                        }
                        v[axis] = 1.0;
                        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                        v.iter().map(|x| x / norm).collect()
                    })
                    .collect();
                ("200 OK", serde_json::json!({ "embeddings": embs }).to_string())
            } else {
                ("404 Not Found", serde_json::json!({ "detail": "Unknown model" }).to_string())
            };
            let mut stream = reader.into_inner();
            let _ = write!(
                stream,
                "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{out}",
                out.len()
            );
            let _ = stream.flush();
        }
    });
    (url, seen)
}

pub(super) fn context(url: &str) -> SemanticSearchContext {
    SemanticSearchContext::new(SemanticSearchConfig {
        embedding_service_url: url.to_string(),
        ..SemanticSearchConfig::default()
    })
    .with_supported_models([("gemma".to_string(), 8), ("qwen".to_string(), 8)])
}

pub(super) async fn wait_for_empty_queue(store: &DocStore) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while !vector_kv::list_queue_entries(&store.db).await.unwrap().is_empty() {
        assert!(std::time::Instant::now() < deadline, "the vector queue never drained");
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}

/// The cluster ids of a namespace's stored chunk keys.
pub(super) async fn sparse_clusters(store: &DocStore, namespace: &str) -> Vec<u32> {
    let rows = store
        .db
        .namespace(vector_kv::sparse_vectors_ns(namespace))
        .await
        .unwrap()
        .scan_prefix(Vec::new())
        .await
        .unwrap();
    let mut ids: Vec<u32> = rows.iter().map(|(k, _)| u32::from_be_bytes(k[..4].try_into().unwrap())).collect();
    ids.sort_unstable();
    ids.dedup();
    ids
}

pub(super) const TEXT: &str = "One sentence here. Two sentences now. Three of them. Four in a row. Five so far. Six at last.";

pub(super) async fn results(store: &DocStore, ns: &str) -> Vec<(Vec<u8>, u32)> {
    let page = store
        .search_semantic(ns, "hello", &SearchSpec::default(), Pagination::default())
        .await
        .unwrap();
    page.results.iter().map(|r| (r.document_id.clone(), r.dot_product.to_bits())).collect()
}

/// The partition of `namespace`, loaded as every user of it loads it.
pub(super) async fn partition(store: &DocStore, namespace: &str) -> std::sync::Arc<crate::semantic_search::PartitionHandle> {
    let (settings, ns_id) = store.schemas.vector_settings(namespace).unwrap();
    store
        .semantic_ctx
        .as_ref()
        .unwrap()
        .for_namespace(&store.db, namespace, ns_id, &settings)
        .await
        .unwrap()
        .partition
}

/// Names of `namespace`'s companion namespaces that exist.
pub(super) fn companions_left(store: &DocStore, namespace: &str) -> Vec<String> {
    let prefix = format!("{namespace}_");
    store
        .db
        .list_namespaces()
        .into_iter()
        .map(|(n, _)| n)
        .filter(|n| n.starts_with(&prefix))
        .collect()
}

/// Embed-queue entries of `namespace`.
pub(super) async fn queued_for(store: &DocStore, namespace: &str) -> usize {
    vector_kv::list_queue_entries(&store.db)
        .await
        .unwrap()
        .iter()
        .filter(|e| e.namespace == namespace)
        .count()
}
