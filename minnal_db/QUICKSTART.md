# minnal_db — Embedded Quickstart

`minnal_db` is a **single embeddable crate** you link directly into your Rust
process — no server to run, no network hop. It gives you:

- an **LSM + value-log key-value engine** (WiscKey-style), always compiled in;
- **RoaringBitmap field indexing** with a predicate query DSL, also always in;
- a **JSON document store**, behind the `doc-store` cargo feature;
- **quantised ANN semantic search**, behind the `semantic-search` cargo feature.

Everything runs in your process, background workers included — compaction,
value-log GC, WAL GC and TTL expiry all tick along inside your binary.

**Semantic search is the one part that reaches outside.** Its ANN index,
quantisation and search run in-process like the rest, but turning text into
vectors does not: that calls an external embedding service over HTTP, and the
IVF centroids it quantises against are a data file you download separately.
[§6](#6-semantic-search-doc-store--kv-store--semantic-search) covers both.

For running minnal as a REST **service** instead, see
[`minnal_db_api`](../minnal_db_api/README.md).

> **Platform:** Linux and macOS only — the engine uses `pread`/`pwrite`.

> **Semantic search needs two things this crate does not ship: an embedding
> service and a cluster-centroid file.** The default `kv-store` and `doc-store`
> features have **no external dependencies** — but the moment you enable
> `semantic-search`, indexing and querying vectors need both of these in place:
>
> 1. an **embedding service** reachable over HTTP (default
>    `http://localhost:8001`), which minnal calls to turn text into vectors. To
>    get started, run the companion reference service,
>    [minnal0021/embedding_service](https://github.com/minnal0021/embedding_service),
>    which serves the **gemma** embedding model, then point
>    `SemanticSearchConfig::embedding_service_url` (embedded) or
>    `semantic_search.embedding_service_url` (REST server) at it. Without a
>    service reachable, writes still succeed but vector indexing lags and
>    semantic *queries* error.
> 2. a **cluster-centroid file** — pre-computed IVF centroids, downloaded
>    separately (they are data, not code, and are not inside the published
>    crate). Two are available, one per supported model:
>    [gemma](https://github.com/minnal0021/minnal/raw/main/service/embedding_support/gemma/clusters.json)
>    and [qwen](https://github.com/minnal0021/minnal/raw/main/service/embedding_support/qwen/clusters.json).
>    Use the one matching the model your service serves.
>
> Both are set up in [§6](#6-semantic-search-doc-store--kv-store--semantic-search).

---

## 1. Select your features

Capabilities are opt-in through cargo features, so you compile — and pull
dependencies for — only what you actually use. There are three to choose from.

**`kv-store`** is the default, and it is really a name for the base engine: the
LSM tree and value log, namespaces, TTL expiry, typed zero-copy values, and
RoaringBitmap field indexing with its predicate query DSL. The engine and the
field index are compiled in no matter which features you pick, so this one adds
nothing beyond the base dependencies — it exists so you can name the default
explicitly.

**`doc-store`** puts a JSON document layer on top of that engine: document
schemas, document CRUD, background index builders, and cursor pagination. It
pulls in one small dependency, `json_dotpath`.

**`semantic-search`** adds quantised IVF + RaBitQ approximate-nearest-neighbour
vector search over stored vectors, along with the client that talks to the
embedding service. It works on raw KV namespaces through
`vector_kv::DbVectorStore`, so you do not need the document store to use it. This
is the only feature with a meaningful dependency cost — `reqwest`, `simsimd` and
`rayon` — and the only one that needs anything external at runtime.

### How they combine

The storage layer and semantic search are independent choices, which gives you
four useful combinations. On its own, the default `kv-store` is the leanest
build: the KV engine and field indexing, no vector dependencies compiled in.
Adding `semantic-search` to it gives you vector search over raw namespaces while
staying below the document layer. Choosing `doc-store` instead gets you JSON
documents with field-index queries and still no vector dependencies. Enabling
`doc-store` and `semantic-search` together is the full build: documents,
embed-on-write, and vector search over them.

If you enable `doc-store` without `semantic-search`, asking for a
semantic-search-enabled store is **rejected at runtime**
(`DocStoreError::SemanticSearchNotCompiled`) rather than silently ignored, so a
missing feature surfaces as a clear error instead of a store that quietly never
indexes anything.

---

## 2. Add the dependency

Pick the line below that matches the build you settled on above and drop it into
your `Cargo.toml`. The default needs no `features` key at all; everything else is
one or two feature names.

```toml
# Lean KV engine + field indexing (default)
minnal_db = "0.2"

# JSON document store, no vector dependencies
minnal_db = { version = "0.2", features = ["doc-store"] }

# Document store + semantic search (full)
minnal_db = { version = "0.2", features = ["doc-store", "semantic-search"] }

# Raw KV + semantic search, without the document layer
minnal_db = { version = "0.2", features = ["semantic-search"] }
```

---

## 3. Using the key-value engine (default `kv-store`)

`Db` (and its async twin, `AsyncDb`) is the entry point: you open a database
directory, and from there put, get, delete, scan by prefix, or walk a key range.
Keys and values are plain bytes — minnal does not interpret them. A database also
carries any number of **namespaces**, each an isolated keyspace of its own, which
is how you keep unrelated data apart inside one database.

Whichever API you use, call `shutdown()` when you are done: it stops the
background workers and flushes buffered writes.

### The synchronous API

```rust
use minnal_db::Db;

let db = Db::open("/tmp/mydb")?;

db.put(b"hello", b"world")?;
let val = db.get(b"hello")?;          // Some(b"world")
db.delete(b"hello")?;

for (key, value) in db.scan_prefix(b"user:")? { /* … */ }
for (key, value) in db.range(b"a", Some(b"z"))? { /* … */ }

// Namespaces — each has its own isolated keyspace
let ns = db.namespace("orders")?;
ns.put(b"o1", b"shipped")?;

db.shutdown()?;
```

### The asynchronous API

`AsyncDb` mirrors the same operations for a tokio runtime. The shape is
identical; the calls take owned buffers and are awaited.

```rust
use minnal_db::AsyncDb;

let db = AsyncDb::open("/tmp/mydb").await?;
db.put(b"hello".to_vec(), b"world".to_vec()).await?;
let val = db.get(b"hello".to_vec()).await?;
db.shutdown().await?;
```

---

## 4. Indexing fields and querying them (default `kv-store`)

Because minnal stores opaque value bytes, it cannot know where a field lives
inside your values — so you tell it. You supply an *extractor closure*
(`&[u8] -> Option<IndexValue>`) per indexed field, which pulls that field out of
whatever encoding you use, be it JSON, a fixed binary layout, or anything else.
From then on, indexing runs inline on every `put`, so a query issued immediately
after a write already sees that write.

Setting this up is four steps, shown in full below: declare the fields you want
indexed, activate each one with its extractor, write records as usual, and query
with the predicate DSL.

```rust
use std::sync::Arc;
use minnal_db::{Db, ExtractorFn, IndexValue, IndexValueType, KVError, DEFAULT_NAMESPACE_ID};

fn main() -> Result<(), KVError> {
    let db = Db::open("/tmp/users_db")?;

    // 1. Declare which fields to index (persisted — re-activate, don't re-register, on restart).
    let status_field = db.register_index_field(DEFAULT_NAMESPACE_ID, "status", IndexValueType::Str)?;
    let age_field    = db.register_index_field(DEFAULT_NAMESPACE_ID, "age",    IndexValueType::Int)?;

    // 2. Activate each field with an extractor over your value bytes (here: JSON).
    let status_extractor: ExtractorFn = Arc::new(|bytes: &[u8]| {
        let v: serde_json::Value = serde_json::from_slice(bytes).ok()?;
        Some(IndexValue::Str(v["status"].as_str()?.to_string()))
    });
    let age_extractor: ExtractorFn = Arc::new(|bytes: &[u8]| {
        let v: serde_json::Value = serde_json::from_slice(bytes).ok()?;
        Some(IndexValue::Int(v["age"].as_i64()?))
    });
    db.activate_field_index(DEFAULT_NAMESPACE_ID, status_field, IndexValueType::Str, status_extractor)?;
    db.activate_field_index(DEFAULT_NAMESPACE_ID, age_field,    IndexValueType::Int, age_extractor)?;

    // 3. Write records — each put updates the RoaringBitmap indices automatically.
    db.put(b"user:1", br#"{"status":"active","age":30}"#)?;
    db.put(b"user:2", br#"{"status":"inactive","age":25}"#)?;
    db.put(b"user:3", br#"{"status":"active","age":42}"#)?;
    db.put(b"user:4", br#"{"status":"active","age":18}"#)?;

    // 4. Query with the predicate DSL (=, !=, <, <=, >, >=, AND, OR, BETWEEN, IN).
    let keys = db.query_index(DEFAULT_NAMESPACE_ID, r#"status = "active" AND age > 20"#)?;
    for key in keys {
        if let Some(value) = db.get(&key)? {
            println!("{} => {}", String::from_utf8_lossy(&key), String::from_utf8_lossy(&value));
        }
    }
    // → user:1 and user:3

    db.shutdown()?;
    Ok(())
}
```

Two things are worth knowing before you build on this. The `IndexValue` your
extractor returns has to match the type you registered the field with — one of
`Bool`, `Int` (an `i64`), or `Str`. And when a query can match a large number of
records, reach for `query_index_paginated(ns, predicate, offset, limit)` instead:
paired with a `RowToKeyFn` registered through `set_row_id_fn`, it resolves only
`offset + limit` keys rather than the whole match set.

---

## 5. Storing JSON documents (`doc-store` feature)

With `features = ["doc-store"]` you get the `DocStore` handle, which puts a JSON
document model over the same engine: typed document IDs, declarative schemas, and
index builds that run in the background. Rather than registering extractors
yourself as in §4, you name the fields you want indexed in the schema and the
store wires up the rest. The schema is the same shape as the REST create payload
documented in [`minnal_db_api`](../minnal_db_api/README.md), so what you learn
here carries over to the server.

```rust
use minnal_db::{DocId, DocStore, DocStoreSchema, IndexSpec, IndexType, KeyType, Pagination, StoreType};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // db data dir + schema dir
    let store = DocStore::open("/tmp/docs", "/tmp/docs/schemas").await?;

    // Define a "users" doc store keyed by u64, with `status` (string) and
    // `age` (int) field indices so both can be queried. (`ns_id` is assigned
    // by `create`; `attributes` are optional non-indexed field declarations.)
    let schema = DocStoreSchema {
        namespace: "users".into(),
        store_type: StoreType::Doc,
        ns_id: None,
        key_type: KeyType::U64,
        attributes: vec![],
        indices: vec![
            IndexSpec { field: "status".into(), index_type: IndexType::Str },
            IndexSpec { field: "age".into(), index_type: IndexType::Int },
        ],
        semantic_search_enabled: false,
        embedding_fields: vec![],
    };
    store.create(schema).await?;

    store.put("users", DocId::U64(1), serde_json::json!({ "status": "active", "age": 30 })).await?;
    let doc = store.get("users", DocId::U64(1)).await?;

    let page = store.query("users", r#"status = "active" AND age > 20"#, Pagination::default()).await?;
    for (id, doc) in page.results { println!("{id:?} => {doc}"); }

    store.shutdown().await?;
    Ok(())
}
```

Add `"semantic-search"` to the features to enable embed-on-write and vector
search on document (or `value_type = "str"` KV) stores — see §6.

---

## 6. Semantic search (`doc-store` / `kv-store` + `semantic-search`)

With `semantic-search` enabled you can run quantised ANN vector search on either
a **document** store or a **KV store whose value type is `str`**
(`value_type = "str"`). It needs two things you must supply:

1. an external **embedding service** (default `http://localhost:8001`) — run the
   companion [minnal0021/embedding_service](https://github.com/minnal0021/embedding_service),
   which serves the **gemma** model; and
2. a **cluster-centroid file** — the pre-computed IVF centroids the coarse
   quantiser assigns chunks to. See below.

### Getting the cluster-centroid file

The centroid file is data, not code, and it is **not shipped inside the crate** —
`cargo add minnal_db` will not put one on your disk. You download it yourself and
point `ClusterIndex::load_with_dim` (when embedding) or
`semantic_search.cluster_path` (when running the REST server) at wherever you put
it.

Two ready-made centroid sets live in the minnal repository, one per supported
model:

| Model | Download | View on GitHub | Centroids | Dim | Size |
|---|---|---|:---:|:---:|:---:|
| **gemma** (used by the reference embedding service) | [`gemma/clusters.json`](https://github.com/minnal0021/minnal/raw/main/service/embedding_support/gemma/clusters.json) | [source](https://github.com/minnal0021/minnal/blob/main/service/embedding_support/gemma/clusters.json) | 256 | 768 | 4.4 MB |
| **qwen** | [`qwen/clusters.json`](https://github.com/minnal0021/minnal/raw/main/service/embedding_support/qwen/clusters.json) | [source](https://github.com/minnal0021/minnal/blob/main/service/embedding_support/qwen/clusters.json) | 256 | 768 | 4.4 MB |

```sh
# Fetch the gemma centroids next to your application (~4.4 MB)
curl -L --create-dirs -o clusters/clusters.json \
  https://github.com/minnal0021/minnal/raw/main/service/embedding_support/gemma/clusters.json
```

Pick the file that matches the model your embedding service serves. Both sets are
768-dimensional, so minnal cannot tell them apart: `load_with_dim` checks the
dimension and nothing more. Pair the wrong file with your service and startup
succeeds, queries return results, and recall is quietly worse — you get no error
at any point. Matching the two is on you.

Neither file is privileged: they are ordinary JSONL, one
`{"cluster_id": <u32>, "centroid": [f32; dim]}` object per line, and the only
thing special about them is the model they were fitted on. If you serve a
different model, run k-means over a sample of your own corpus's embeddings and
load the result exactly the same way. Whichever file you use, it is read into
memory once at startup and never mutated afterwards — around 750 KB resident for
256 centroids of 768 `f32`.

One last thing to expect before the code: indexing is **asynchronous**. A write
returns as soon as the embed job is enqueued, and a background worker — started
for you by `with_semantic_search` — calls the service, quantises the result and
stores the vector. A query issued immediately after a write may therefore not see
it yet. If the service is down, writes still succeed and the worker keeps
retrying; only vector *indexing* falls behind, while semantic *queries* error.

### Shared setup: attaching a `SemanticSearchContext`

Both storage layers are driven the same way. You load the cluster index, build a
`SemanticSearchConfig`, and hand both to the store as a `SemanticSearchContext`.
Attach it once, at startup — it is also what spawns the embed-worker and the
one-shot reconciliation pass.

```rust
use std::sync::Arc;
use minnal_db::{
    AttributeDef, AttributeType, DocId, DocStore, DocStoreSchema, KeyType, KvKeyType,
    KvStoreSchema, KvValueType, Pagination, SemanticSearchContext, StoreType,
};
use minnal_db::semantic_search::ClusterIndex;
use minnal_db::semantic_search::service::SemanticSearchConfig;

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let embedding_dim = 768; // must match the model the embedding service serves
// Path to the centroid file you downloaded above — gemma and qwen centroids are
// both published in the minnal repo under service/embedding_support/{model}/.
let cluster_index = Arc::new(ClusterIndex::load_with_dim(
    "clusters/clusters.json",
    embedding_dim,
)?);
// `embedding_dim` (768) is already the default; it is spelled out here because it
// must agree with both the centroid file and the model the service serves.
// `model_name` is inert when embedded — nothing is sent to the service, and it is
// not used to pick the cluster file (that is the `ClusterIndex` above) — but set it
// to match, since it defaults to "qwen".
let config = SemanticSearchConfig {
    embedding_dim,
    model_name: "gemma".into(),
    ..Default::default()
};

// `with_semantic_search` also starts the background embed-worker + a one-shot
// startup reconciliation, so attach it once, up front.
let store = DocStore::open("/tmp/sem", "/tmp/sem/schemas")
    .await?
    .with_semantic_search(SemanticSearchContext { config, cluster_index });
```

### Searching a document store

For documents you mark the store as semantic-search-enabled in its schema and
list which fields should be embedded. Every write to those fields then enqueues
an embed job automatically, and `search_semantic` runs the two-pass ANN search
over what has been indexed so far. Results come back with the document's key
bytes in `document_id`.

```rust
// A doc store whose `body` field is embedded (semantic_search_enabled + embedding_fields).
let schema = DocStoreSchema {
    namespace: "articles".into(),
    store_type: StoreType::Doc,
    ns_id: None,
    key_type: KeyType::U64,
    attributes: vec![AttributeDef { name: "body".into(), attr_type: AttributeType::Str, description: None }],
    indices: vec![],
    semantic_search_enabled: true,
    embedding_fields: vec!["body".into()],
};
store.create(schema).await?;

// Each put enqueues an async embed job and returns immediately.
store.put("articles", DocId::U64(1), serde_json::json!({ "body": "thunder and lightning over the sea" })).await?;
store.put("articles", DocId::U64(2), serde_json::json!({ "body": "a quiet afternoon in the library" })).await?;

// Two-pass ANN search over the stored vectors (top 5). `document_id` is the doc's key bytes.
let hits = store.search_semantic("articles", "a storm at night", Some(5), Pagination::default()).await?;
for r in hits.results {
    let id = u64::from_be_bytes(r.document_id[..8].try_into().unwrap());
    println!("doc {id}  score={:.4}", r.dot_product);
}
```

### Searching a KV store

A KV store works the same way with less ceremony: declare it with
`value_type = "str"` and semantic search enabled, and every string value you
write gets embedded. Here `document_id` in the results is the raw string key
rather than encoded key bytes.

```rust
// A KV store whose string values are embedded (value_type = str + semantic_search_enabled).
let kv_schema = KvStoreSchema {
    namespace: "notes".into(),
    store_type: StoreType::Kv,
    ns_id: None,
    key_type: KvKeyType::Str,
    value_type: KvValueType::Str,
    semantic_search_enabled: true,
};
store.create_kv(kv_schema).await?;

store.kv_put("notes", &serde_json::json!("n1"), &serde_json::json!("meeting about the Q3 budget")).await?;
store.kv_put("notes", &serde_json::json!("n2"), &serde_json::json!("weekend hiking trip planning")).await?;

// `document_id` here is the raw string key.
let hits = store.kv_search_semantic("notes", "financial planning", Some(5), Pagination::default()).await?;
for r in hits.results {
    println!("key {}  score={:.4}", String::from_utf8_lossy(&r.document_id), r.dot_product);
}

store.shutdown().await?; // stops the embed-worker cleanly
# Ok(()) }
```
