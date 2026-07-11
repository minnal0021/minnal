# minnal_db — Embedded Quickstart

`minnal_db` is a **single embeddable crate**. It ships an LSM + value-log
key-value engine with RoaringBitmap field indexing, and — behind cargo
features — a JSON document store and quantised ANN semantic search. Link it
directly into your Rust process: no server, no network hop, all background
workers (compaction, value-log GC, WAL GC, TTL) run in-process.

For running minnal as a REST **service** instead, see
[`minnal_db_api`](../minnal_db_api/README.md).

> **Platform:** Linux and macOS only — the engine uses `pread`/`pwrite`.

> **Semantic search needs an embedding service.** The default `kv-store` and
> `doc-store` features have **no external dependencies**. But the moment you
> enable `semantic-search`, indexing and querying vectors require an external
> **embedding service** reachable over HTTP (default `http://localhost:8001`) —
> minnal calls it to turn text into vectors. To get started, run the companion
> reference service, [minnal0021/embedding_service](https://github.com/minnal0021/embedding_service),
> which serves the **gemma** embedding model, then point
> `SemanticSearchConfig::embedding_service_url` (embedded) or
> `semantic_search.embedding_service_url` (REST server) at it. Without a service
> reachable, writes still succeed but vector indexing lags and semantic
> *queries* error. See [§6](#6-semantic-search-doc-store--kv-store--semantic-search).

---

## 1. Select your features

Capabilities are opt-in through cargo features, so you only compile (and only
pull dependencies for) what you use. **Field indexing is always included** in
the base engine; `doc-store` and `semantic-search` are independent knobs.

### Individual features

| Feature | Default | What it adds | Extra dependencies pulled in |
|---|:---:|---|---|
| `kv-store` | ✅ | LSM + value-log KV engine, namespaces, TTL, typed (zero-copy) values, **RoaringBitmap field indexing + predicate query DSL** | — (base only) |
| `doc-store` | | JSON document store: schema, document CRUD, background index builders, cursor pagination | `json_dotpath` |
| `semantic-search` | | Quantised IVF + RaBitQ ANN vector search over stored vectors, usable on **raw KV namespaces** (`vector_kv::DbVectorStore`), plus the embedding-service client | `reqwest`, `simsimd`, `rayon`, `futures` |

### How they combine (storage layer × semantic search)

| Cargo features | Capability | Vector deps compiled in? |
|---|---|:---:|
| *(default)* `kv-store` | KV engine + field indexing | ❌ none |
| `kv-store` + `semantic-search` | + vector search on raw namespaces | ✅ yes |
| `doc-store` | JSON documents + field-index queries | ❌ none |
| `doc-store` + `semantic-search` | full: documents + embed-on-write + vector search | ✅ yes |

Notes:

- `semantic-search` is **orthogonal** — it works on top of either a raw KV store
  or the document store.
- With `doc-store` but **not** `semantic-search`, requesting a semantic-search
  store is **rejected at runtime** (`DocStoreError::SemanticSearchNotCompiled`)
  rather than silently ignored.
- `semantic-search` requires an **external embedding service** at query/index
  time (default `http://localhost:8001`); nothing else needs anything external.

---

## 2. Add the dependency

```toml
# Lean KV engine + field indexing (default)
minnal_db = "0.1"

# JSON document store, no vector dependencies
minnal_db = { version = "0.1", features = ["doc-store"] }

# Document store + semantic search (full)
minnal_db = { version = "0.1", features = ["doc-store", "semantic-search"] }

# Raw KV + semantic search, without the document layer
minnal_db = { version = "0.1", features = ["semantic-search"] }
```

---

## 3. Key-value usage (default `kv-store`)

### Synchronous

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

### Asynchronous

```rust
use minnal_db::AsyncDb;

let db = AsyncDb::open("/tmp/mydb").await?;
db.put(b"hello".to_vec(), b"world".to_vec()).await?;
let val = db.get(b"hello".to_vec()).await?;
db.shutdown().await?;
```

---

## 4. Field indexing + predicate queries (default `kv-store`)

`minnal_db` stores **opaque value bytes** — the field index is driven by an
*extractor closure* you supply (`&[u8] -> Option<IndexValue>`), so you decide
how to pull an indexed field out of your own value encoding (JSON, a fixed
binary layout, …). Indexing runs **inline on every `put`**, so a query
immediately after a write sees it.

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

The `IndexValue` your extractor returns must match the registered
`IndexValueType` (`Bool`, `Int` (i64), or `Str`). For large result sets use
`query_index_paginated(ns, predicate, offset, limit)` with a `RowToKeyFn`
registered via `set_row_id_fn` so only `offset + limit` keys are resolved.

---

## 5. Document store (`doc-store` feature)

With `features = ["doc-store"]`, the `DocStore` handle adds a JSON document
model — typed IDs, declarative schemas, and background index builds — on top of
the same engine. The schema mirrors the REST create payload documented in
[`minnal_db_api`](../minnal_db_api/README.md).

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
a **document** store or a **`value_type = "str"` KV** store. It needs an external
**embedding service** (default `http://localhost:8001`) and a set of pre-computed
IVF cluster centroids (bundled at `service/embedding_support/qwen/clusters.json`).

Indexing is **asynchronous**: a write returns immediately after enqueuing an
embed job; a background worker (started by `with_semantic_search`) calls the
service, quantises the result, and stores the vector. So a query right after a
write may not see it yet, and if the service is down, writes still succeed and
the worker retries — only vector *indexing* lags and semantic *queries* error.

### Shared setup — attach a `SemanticSearchContext`

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
let cluster_index = Arc::new(ClusterIndex::load_with_dim(
    "service/embedding_support/qwen/clusters.json",
    embedding_dim,
)?);
let config = SemanticSearchConfig { embedding_dim, ..Default::default() };

// `with_semantic_search` also starts the background embed-worker + a one-shot
// startup reconciliation, so attach it once, up front.
let store = DocStore::open("/tmp/sem", "/tmp/sem/schemas")
    .await?
    .with_semantic_search(SemanticSearchContext { config, cluster_index });
```

### Document store

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

### KV store

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

The `cluster_path` and `embedding_dim` must match the embedding model the
service actually serves (see the model-pinning notes in
`src/semantic_search/CLAUDE.md`). To run vector search over the **REST** API
instead, use [`minnal_db_api`](../minnal_db_api/README.md).
