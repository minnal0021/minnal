//! Frozen embeddings: every document, chunk and query vector of one BEIR dataset,
//! embedded once and saved as raw little-endian `f32`, so every benchmark run
//! indexes exactly the same vectors without calling the embedding service.
//!
//! Layout of `{bench_root}/{dataset}/{model}/`:
//!
//! | File | Contents |
//! |---|---|
//! | `dense.f32` | one whole-document vector per indexed doc (`docs × dim`) |
//! | `chunks.f32` | every sliding-window chunk vector, grouped by doc in doc order (`chunks × dim`) |
//! | `chunk_doc.u32` | the doc row of each chunk |
//! | `queries.f32` | one vector per judged query (`queries × dim`), in query-id order |
//! | `manifest.json` | ids, dimension, model, chunking parameters; written **last**, so its presence means the dump is complete |
//!
//! Documents whose text is empty are skipped (the service cannot embed them, and
//! production never indexes them either), and so are documents the service
//! refuses (production's vector worker fails on those too); both are listed in
//! the manifest.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;

use super::super::metrics::beir_eval::{read_jsonl, read_qrels, str_field};
use crate::semantic_search::chunking::chunk_document;
use crate::semantic_search::service::embedding_service::{EmbeddingTarget, embed};

/// Most payloads sent in one embedding request while dumping. A document's
/// payloads (`[whole_text, chunk₀, …]`) always travel together.
const MAX_PAYLOADS_PER_REQUEST: usize = 96;
/// Embedding requests in flight while dumping.
const DUMP_CONCURRENCY: usize = 8;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// What a dump contains and how it was made.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub(super) struct Manifest {
    pub dataset: String,
    pub model: String,
    pub split: String,
    pub dim: usize,
    pub window_size: usize,
    pub sliding_size: usize,
    /// Corpus id of each indexed doc row.
    pub doc_ids: Vec<String>,
    pub n_chunks: usize,
    /// Corpus ids of docs skipped because their text was empty.
    pub skipped_empty_docs: Vec<String>,
    /// Corpus ids of docs the embedding service refused (for example over its
    /// token limit). Production fails to index these too.
    #[serde(default)]
    pub skipped_embed_failed: Vec<String>,
    /// Judged query ids, sorted; row `q` of `queries.f32` is `query_ids[q]`.
    pub query_ids: Vec<String>,
    pub embed_url: String,
    pub dump_seconds: f64,
}

/// A loaded dump.
pub(super) struct Frozen {
    pub manifest: Manifest,
    dense: Vec<f32>,
    chunks: Vec<f32>,
    pub chunk_doc: Vec<u32>,
    /// `chunk_start[d]..chunk_start[d + 1]` are doc `d`'s chunk rows.
    chunk_start: Vec<usize>,
    queries: Vec<f32>,
}

impl Frozen {
    pub fn dim(&self) -> usize {
        self.manifest.dim
    }
    pub fn n_docs(&self) -> usize {
        self.manifest.doc_ids.len()
    }
    pub fn n_chunks(&self) -> usize {
        self.chunk_doc.len()
    }
    pub fn n_queries(&self) -> usize {
        self.manifest.query_ids.len()
    }
    pub fn dense(&self, doc: usize) -> &[f32] {
        &self.dense[doc * self.dim()..(doc + 1) * self.dim()]
    }
    pub fn chunk(&self, chunk: usize) -> &[f32] {
        &self.chunks[chunk * self.dim()..(chunk + 1) * self.dim()]
    }
    pub fn chunk_rows(&self, doc: usize) -> std::ops::Range<usize> {
        self.chunk_start[doc]..self.chunk_start[doc + 1]
    }
    /// Doc `doc`'s chunk vectors, owned, as `index_embeddings` takes them.
    pub fn doc_chunks(&self, doc: usize) -> Vec<Vec<f32>> {
        self.chunk_rows(doc).map(|c| self.chunk(c).to_vec()).collect()
    }
    pub fn query(&self, q: usize) -> &[f32] {
        &self.queries[q * self.dim()..(q + 1) * self.dim()]
    }
}

/// The parameters a dump must match to be reused.
pub(super) struct DumpSpec<'a> {
    pub dataset: &'a str,
    pub dataset_dir: &'a Path,
    pub split: &'a str,
    pub model: &'a str,
    pub dim: usize,
    pub window_size: usize,
    pub sliding_size: usize,
    pub embed_url: &'a str,
}

/// Load the dump in `dir`, creating it through the embedding service first if it
/// does not exist. Panics if an existing dump was made with other parameters.
pub(super) async fn load_or_dump(dir: &Path, spec: &DumpSpec<'_>) -> Frozen {
    let manifest_path = dir.join("manifest.json");
    if !manifest_path.exists() {
        dump(dir, spec).await;
    }
    let manifest: Manifest = serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    assert_eq!(
        (
            manifest.model.as_str(),
            manifest.split.as_str(),
            manifest.dim,
            manifest.window_size,
            manifest.sliding_size
        ),
        (spec.model, spec.split, spec.dim, spec.window_size, spec.sliding_size),
        "{} was dumped with other parameters; move it aside to re-dump",
        dir.display()
    );
    let dense = read_f32(&dir.join("dense.f32"));
    let chunks = read_f32(&dir.join("chunks.f32"));
    let chunk_doc = read_u32(&dir.join("chunk_doc.u32"));
    let queries = read_f32(&dir.join("queries.f32"));
    let n_docs = manifest.doc_ids.len();
    assert_eq!(dense.len(), n_docs * manifest.dim, "dense.f32 size");
    assert_eq!(chunks.len(), chunk_doc.len() * manifest.dim, "chunks.f32 size");
    assert_eq!(queries.len(), manifest.query_ids.len() * manifest.dim, "queries.f32 size");
    let mut chunk_start = vec![0usize; n_docs + 1];
    for &d in &chunk_doc {
        chunk_start[d as usize + 1] += 1;
    }
    for d in 0..n_docs {
        chunk_start[d + 1] += chunk_start[d];
    }
    assert!(chunk_doc.windows(2).all(|w| w[0] <= w[1]), "chunks must be grouped by doc in doc order");
    Frozen {
        manifest,
        dense,
        chunks,
        chunk_doc,
        chunk_start,
        queries,
    }
}

/// `title. text`, as `beir_eval` builds a document body.
pub(super) fn doc_body(doc: &serde_json::Value) -> String {
    let (title, text) = (str_field(doc, "title"), str_field(doc, "text"));
    if title.is_empty() {
        text.to_string()
    } else {
        format!("{title}. {text}")
    }
}

/// Judged queries of `split`, sorted by id: `(id, text)`.
pub(super) fn judged_queries(dataset_dir: &Path, split: &str) -> Vec<(String, String)> {
    let qrels = read_qrels(&dataset_dir.join("qrels").join(format!("{split}.tsv")));
    let mut queries: Vec<(String, String)> = read_jsonl(&dataset_dir.join("queries.jsonl"))
        .iter()
        .map(|q| (str_field(q, "_id").to_string(), str_field(q, "text").to_string()))
        .filter(|(id, text)| qrels.contains_key(id) && !text.is_empty())
        .collect();
    queries.sort_by(|a, b| a.0.cmp(&b.0));
    queries
}

async fn dump(dir: &Path, spec: &DumpSpec<'_>) {
    let started = Instant::now();
    std::fs::create_dir_all(dir).unwrap();
    let corpus = read_jsonl(&spec.dataset_dir.join("corpus.jsonl"));
    let mut doc_ids = Vec::new();
    let mut skipped = Vec::new();
    let mut payloads: Vec<Vec<String>> = Vec::new(); // per doc: [body, chunk₀, …]
    for d in &corpus {
        let body = doc_body(d);
        let id = str_field(d, "_id").to_string();
        if body.trim().is_empty() {
            skipped.push(id);
            continue;
        }
        let mut p = vec![body.clone()];
        p.extend(chunk_document(&body, spec.window_size, spec.sliding_size));
        doc_ids.push(id);
        payloads.push(p);
    }
    let n_payloads: usize = payloads.iter().map(Vec::len).sum();
    eprintln!(
        "  dumping {}: {} docs ({} skipped as empty), {} payloads via {}",
        spec.dataset,
        doc_ids.len(),
        skipped.len(),
        n_payloads,
        spec.embed_url
    );

    // Group whole docs into requests of at most MAX_PAYLOADS_PER_REQUEST payloads.
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let (mut cur, mut n) = (Vec::new(), 0usize);
    for (i, p) in payloads.iter().enumerate() {
        if !cur.is_empty() && n + p.len() > MAX_PAYLOADS_PER_REQUEST {
            groups.push(std::mem::take(&mut cur));
            n = 0;
        }
        cur.push(i);
        n += p.len();
    }
    if !cur.is_empty() {
        groups.push(cur);
    }

    let payloads = Arc::new(payloads);
    let per_doc: Vec<Option<Vec<Vec<f32>>>> = {
        let sem = Arc::new(Semaphore::new(DUMP_CONCURRENCY));
        let mut set = tokio::task::JoinSet::new();
        for (g, docs) in groups.into_iter().enumerate() {
            let permit = sem.clone().acquire_owned().await.unwrap();
            let (payloads, url, model, dim) = (payloads.clone(), spec.embed_url.to_string(), spec.model.to_string(), spec.dim);
            set.spawn(async move {
                let _permit = permit;
                let flat: Vec<String> = docs.iter().flat_map(|&i| payloads[i].iter().cloned()).collect();
                match embed(&url, &model, EmbeddingTarget::Document, &flat, dim, REQUEST_TIMEOUT, CONNECT_TIMEOUT).await {
                    Ok(embs) => {
                        let mut it = embs.into_iter();
                        docs.iter()
                            .map(|&i| (i, Some(it.by_ref().take(payloads[i].len()).collect())))
                            .collect::<Vec<_>>()
                    }
                    // One refused doc fails its whole request: retry each doc alone.
                    Err(group_err) => {
                        let mut out = Vec::with_capacity(docs.len());
                        for &i in &docs {
                            match embed(
                                &url,
                                &model,
                                EmbeddingTarget::Document,
                                &payloads[i],
                                dim,
                                REQUEST_TIMEOUT,
                                CONNECT_TIMEOUT,
                            )
                            .await
                            {
                                Ok(embs) => out.push((i, Some(embs))),
                                Err(e) => {
                                    eprintln!("    doc row {i} refused (group {g}: {group_err}): {e}");
                                    out.push((i, None));
                                }
                            }
                        }
                        out
                    }
                }
            });
        }
        let mut out: Vec<Option<Vec<Vec<f32>>>> = vec![None; payloads.len()];
        let (mut done, mut last) = (0usize, Instant::now());
        while let Some(res) = set.join_next().await {
            for (i, embs) in res.unwrap() {
                out[i] = embs;
                done += 1;
            }
            if last.elapsed() > Duration::from_secs(30) {
                eprintln!("    {done}/{} docs, {:.0}s", payloads.len(), started.elapsed().as_secs_f64());
                last = Instant::now();
            }
        }
        out
    };

    let mut dense = Vec::with_capacity(per_doc.len() * spec.dim);
    let mut chunks = Vec::new();
    let mut chunk_doc: Vec<u32> = Vec::new();
    let mut kept_ids = Vec::with_capacity(doc_ids.len());
    let mut refused = Vec::new();
    for (embs, id) in per_doc.iter().zip(doc_ids) {
        let Some(embs) = embs else {
            refused.push(id);
            continue;
        };
        let d = kept_ids.len() as u32;
        dense.extend_from_slice(&embs[0]);
        for c in &embs[1..] {
            chunks.extend_from_slice(c);
            chunk_doc.push(d);
        }
        kept_ids.push(id);
    }
    let doc_ids = kept_ids;

    // Queries: one request each, as `embed_query` sends them.
    let queries = judged_queries(spec.dataset_dir, spec.split);
    let query_vecs: Vec<Vec<f32>> = {
        let sem = Arc::new(Semaphore::new(DUMP_CONCURRENCY));
        let mut set = tokio::task::JoinSet::new();
        for (q, (qid, text)) in queries.iter().enumerate() {
            let permit = sem.clone().acquire_owned().await.unwrap();
            let (url, model, dim, text, qid) = (spec.embed_url.to_string(), spec.model.to_string(), spec.dim, text.clone(), qid.clone());
            set.spawn(async move {
                let _permit = permit;
                let mut e = embed(&url, &model, EmbeddingTarget::Query, &[text], dim, REQUEST_TIMEOUT, CONNECT_TIMEOUT)
                    .await
                    .unwrap_or_else(|e| panic!("embedding query {qid} failed: {e}"));
                (q, e.remove(0))
            });
        }
        let mut out = vec![Vec::new(); queries.len()];
        while let Some(res) = set.join_next().await {
            let (q, v) = res.unwrap();
            out[q] = v;
        }
        out
    };
    let flat_queries: Vec<f32> = query_vecs.concat();

    write_f32(&dir.join("dense.f32"), &dense);
    write_f32(&dir.join("chunks.f32"), &chunks);
    write_u32(&dir.join("chunk_doc.u32"), &chunk_doc);
    write_f32(&dir.join("queries.f32"), &flat_queries);
    let manifest = Manifest {
        dataset: spec.dataset.to_string(),
        model: spec.model.to_string(),
        split: spec.split.to_string(),
        dim: spec.dim,
        window_size: spec.window_size,
        sliding_size: spec.sliding_size,
        doc_ids,
        n_chunks: chunk_doc.len(),
        skipped_empty_docs: skipped,
        skipped_embed_failed: refused,
        query_ids: queries.into_iter().map(|(id, _)| id).collect(),
        embed_url: spec.embed_url.to_string(),
        dump_seconds: started.elapsed().as_secs_f64(),
    };
    std::fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
    eprintln!(
        "  dumped {} docs ({} refused by the service), {} chunks, {} queries in {:.0}s to {}",
        manifest.doc_ids.len(),
        manifest.skipped_embed_failed.len(),
        manifest.n_chunks,
        manifest.query_ids.len(),
        manifest.dump_seconds,
        dir.display()
    );
}

fn write_f32(path: &PathBuf, v: &[f32]) {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    for x in v {
        f.write_all(&x.to_le_bytes()).unwrap();
    }
    f.flush().unwrap();
}

fn write_u32(path: &PathBuf, v: &[u32]) {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    for x in v {
        f.write_all(&x.to_le_bytes()).unwrap();
    }
    f.flush().unwrap();
}

fn read_f32(path: &Path) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn read_u32(path: &Path) -> Vec<u32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}
