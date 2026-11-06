//! The workload: what each class of operation sends, and how it is executed
//! against the engine in process.
//!
//! - `get`: ids drawn Zipf (s = 1.0) over a seeded shuffle of the stable base
//!   documents (a hot set).
//! - `insert`: the ingestion stream in order, then new ids reusing its texts
//!   with a fixed `" (copy)"` appended, so the set of texts the engine embeds is
//!   finite and the replay table covers it. Inserts and deletes are equally
//!   likely and each cell starts from the base's size (`runner::rebalance`; a
//!   delete-only cell starts larger, so it has enough to delete), so
//!   the inserted documents alive at once are a window of the most recent
//!   inserts shorter than the stream: no text is in the store twice, except
//!   briefly when a single-class insert cell outruns that margin (counted per
//!   cell as `repeated_texts`).
//! - `update`: a stable document rewritten whole: half change only `score`,
//!   half append a fixed `" Revised."` to the text (only documents no test query
//!   judges relevant, so search quality stays comparable).
//! - `delete`: the deletable base documents in order, then inserted ones,
//!   oldest first and never closer than 100 to the newest insert.
//! - `scan`: one page of 50 from a random stable key.
//! - `field_query`: 1-3 terms over `category`, `year`, `score`; page of 20.
//! - `search` / `filtered_search`: a FiQA test query, top 100; filtered adds
//!   `score < S` (5-50% of documents).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use minnal_db::doc_store::vector_settings::SearchSpec;
use minnal_db::doc_store::{DocId, DocStore, Pagination, schema::KeyType};
use serde_json::{Value, json};

pub const CLASSES: [&str; 8] = ["get", "insert", "update", "delete", "scan", "field_query", "search", "filtered_search"];
pub const WRITES: [usize; 3] = [1, 2, 3];

/// SplitMix64: seeded, no dependency.
#[derive(Clone)]
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn below(&mut self, n: usize) -> usize {
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
pub struct Doc {
    pub id: u64,
    pub text: String,
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
    pub fn body(&self) -> Value {
        json!({"text": self.text, "category": self.category, "year": self.year, "score": self.score})
    }
    pub fn with_text(&self, text: String) -> Self {
        Self { text, ..self.clone() }
    }
}

/// The benchmark data: `base.jsonl`, `stream.jsonl`, `ids.json`, `queries.jsonl`.
pub struct Data {
    pub stable: Vec<u64>,
    /// Stable documents no test query judges relevant (text updates).
    pub stable_text: Vec<u64>,
    pub deletable: Vec<u64>,
    pub base: HashMap<u64, Doc>,
    pub stream: Vec<Doc>,
    /// `(query id, text)`.
    pub queries: Vec<(String, String)>,
}

impl Data {
    pub fn load(dir: &str) -> Result<Self, String> {
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
        let relevant: HashSet<u64> = list("relevant").into_iter().collect();
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
        Ok(Self {
            stable,
            stable_text,
            deletable: list("deletable"),
            base,
            stream: docs(read("stream.jsonl")?)?,
            queries,
        })
    }

    /// The `n`th insert: the stream, then its texts again under new ids with
    /// the one fixed suffix.
    pub fn stream_doc(&self, n: u64) -> Doc {
        let s = &self.stream;
        let mut d = s[(n % s.len() as u64) as usize].clone();
        if n >= s.len() as u64 {
            d.id = 10_000_000_000 + n;
            d.text = format!("{} (copy)", d.text);
        }
        d
    }
}

/// Insert and delete cursors, shared by every client and carried across cells
/// (saved in the manifest).
#[derive(Default, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Cursors {
    pub inserted: u64,
    pub deleted: u64,
}

impl Cursors {
    /// Documents in the store, given the base it started from: every delete
    /// removes a document that exists (the deletable base, then inserts in
    /// order).
    pub fn documents(&self, base: usize) -> i64 {
        base as i64 + self.inserted as i64 - self.deleted as i64
    }

    /// Texts in the store twice: inserts alive at once (deletes reach inserts
    /// after the `deletable` base, oldest first) past the stream's length.
    pub fn repeated_texts(&self, deletable: usize, stream: usize) -> u64 {
        let alive = self.inserted - self.deleted.saturating_sub(deletable as u64).min(self.inserted);
        alive.saturating_sub(stream as u64)
    }
}

#[derive(Clone)]
pub enum Op {
    Get(u64),
    Insert(Doc),
    Update(Doc),
    Delete(u64),
    Scan(u64),
    FieldQuery(String),
    Search { qid: String, text: String },
    FilteredSearch { text: String, predicate: String },
}

/// Draws operations. Each client has its own; the cursors are shared.
pub struct Generator {
    data: Arc<Data>,
    rng: Rng,
    zipf: Zipf,
    hot: Vec<u64>,
    cursors: Arc<parking_lot::Mutex<Cursors>>,
}

impl Generator {
    /// `hot_seed` must be the same for every generator of a run, so all clients
    /// share one hot set; `seed` differs per client.
    pub fn new(data: Arc<Data>, hot_seed: u64, seed: u64, cursors: Arc<parking_lot::Mutex<Cursors>>) -> Self {
        let mut shuffle = Rng(hot_seed);
        let mut hot = data.stable.clone();
        for i in (1..hot.len()).rev() {
            let j = shuffle.below(i + 1);
            hot.swap(i, j);
        }
        Self {
            zipf: Zipf::new(hot.len(), 1.0),
            hot,
            data,
            rng: Rng(seed),
            cursors,
        }
    }

    /// A class drawn from `weights` (indexed like [`CLASSES`]).
    pub fn class(&mut self, weights: &[f64; 8]) -> usize {
        let total: f64 = weights.iter().sum();
        let mut u = self.rng.unit() * total;
        for (k, w) in weights.iter().enumerate() {
            if u < *w {
                return k;
            }
            u -= w;
        }
        weights.iter().rposition(|w| *w > 0.0).unwrap_or(0)
    }

    /// An operation of `class`, or `None` when none is available (a delete
    /// that would come too close to the newest insert).
    pub fn op(&mut self, class: usize) -> Option<Op> {
        Some(match class {
            0 => Op::Get(self.hot[self.zipf.sample(&mut self.rng)]),
            1 => {
                let n = {
                    let mut c = self.cursors.lock();
                    c.inserted += 1;
                    c.inserted - 1
                };
                Op::Insert(self.data.stream_doc(n))
            }
            2 => {
                let text_change = self.rng.unit() < 0.5;
                let pool = if text_change { &self.data.stable_text } else { &self.data.stable };
                let id = pool[self.rng.below(pool.len())];
                let d = &self.data.base[&id];
                Op::Update(if text_change {
                    d.with_text(format!("{} Revised.", d.text))
                } else {
                    Doc {
                        score: self.rng.below(1001) as i64,
                        ..d.clone()
                    }
                })
            }
            3 => {
                let mut c = self.cursors.lock();
                let n = c.deleted;
                let del = &self.data.deletable;
                let id = if (n as usize) < del.len() {
                    del[n as usize]
                } else {
                    let k = n - del.len() as u64;
                    if k + 100 > c.inserted {
                        return None;
                    }
                    self.data.stream_doc(k).id
                };
                c.deleted += 1;
                Op::Delete(id)
            }
            4 => Op::Scan(self.data.stable[self.rng.below(self.data.stable.len())]),
            5 => {
                let n = 1 + self.rng.below(3);
                let mut kinds = [0usize, 1, 2];
                for i in (1..3).rev() {
                    let j = self.rng.below(i + 1);
                    kinds.swap(i, j);
                }
                let terms: Vec<String> = kinds[..n]
                    .iter()
                    .map(|k| match k {
                        0 => format!("category = \"c{}\"", self.rng.below(12)),
                        1 => format!("year >= {}", 2010 + self.rng.below(16)),
                        _ => format!("score < {}", 100 + self.rng.below(900)),
                    })
                    .collect();
                Op::FieldQuery(terms.join(" AND "))
            }
            6 | 7 => {
                let (qid, text) = self.data.queries[self.rng.below(self.data.queries.len())].clone();
                if class == 6 {
                    Op::Search { qid, text }
                } else {
                    Op::FilteredSearch {
                        text,
                        predicate: format!("score < {}", 50 + self.rng.below(451)),
                    }
                }
            }
            _ => unreachable!(),
        })
    }
}

/// What a completed operation returned.
pub enum Outcome {
    Ok,
    NotFound,
    /// A search's ranked document ids (`search` only).
    Ranked(Vec<(u64, f32)>),
}

/// The namespace and what the REST layer caches about it.
pub struct Target {
    pub store: Arc<DocStore>,
    pub ns: String,
    pub ns_id: u32,
    pub key_type: KeyType,
}

pub fn search_spec() -> SearchSpec {
    SearchSpec {
        top_k: Some(100),
        ..SearchSpec::default()
    }
}

fn ranked(results: Vec<minnal_db::semantic_search::index::vector_index::QueryResult>) -> Vec<(u64, f32)> {
    results
        .into_iter()
        .filter_map(|r| Some((u64::from_be_bytes(r.document_id.as_slice().try_into().ok()?), r.dot_product)))
        .collect()
}

/// Run `op` with the same library calls the REST handlers make. A search
/// returns ids and scores only; the REST route then fetches each result's
/// document, which is not part of this measurement.
pub async fn execute(t: &Target, op: &Op) -> Result<Outcome, String> {
    let s = &t.store;
    let e = |e: minnal_db::doc_store::DocStoreError| e.to_string();
    match op {
        Op::Get(id) => Ok(match s.get(&t.ns, DocId::U64(*id)).await.map_err(e)? {
            Some(_) => Outcome::Ok,
            None => Outcome::NotFound,
        }),
        Op::Insert(d) | Op::Update(d) => s.put(&t.ns, DocId::U64(d.id), d.body()).await.map(|_| Outcome::Ok).map_err(e),
        Op::Delete(id) => s.delete(&t.ns, DocId::U64(*id)).await.map(|_| Outcome::Ok).map_err(e),
        Op::Scan(id) => s.scan_range(&t.ns, DocId::U64(*id), None, None, 50).await.map(|_| Outcome::Ok).map_err(e),
        Op::FieldQuery(p) => s
            .query_resolved(&t.ns, p, Pagination::new(1, 20), t.ns_id, t.key_type)
            .await
            .map(|_| Outcome::Ok)
            .map_err(e),
        Op::Search { text, .. } => s
            .search_semantic(&t.ns, text, &search_spec(), Pagination::new(1, 100))
            .await
            .map(|p| Outcome::Ranked(ranked(p.results)))
            .map_err(e),
        Op::FilteredSearch { text, predicate } => s
            .search_semantic_filtered(&t.ns, text, predicate, &search_spec(), Pagination::new(1, 100))
            .await
            .map(|_| Outcome::Ok)
            .map_err(e),
    }
}

/// Class weights from a `{class: share}` map.
pub fn weights(mix: &BTreeMap<String, f64>) -> Result<[f64; 8], String> {
    let mut w = [0.0; 8];
    for (k, v) in mix {
        let i = CLASSES.iter().position(|c| c == k).ok_or_else(|| format!("unknown class {k}"))?;
        w[i] = *v;
    }
    Ok(w)
}
