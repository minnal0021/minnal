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
vectors does not: that calls an external embedding service over HTTP.
[§6](#6-semantic-search-doc-store--kv-store--semantic-search) covers it.

For running minnal as a REST **service** instead, see
[`minnal_db_api`](../minnal_db_api/README.md).

> **Platform:** Linux and macOS only — the engine uses `pread`/`pwrite`.

> **Semantic search needs one thing this crate does not ship: an embedding
> service.** The default `kv-store` and `doc-store` features have **no external
> dependencies** — but the moment you enable `semantic-search`, indexing and
> querying vectors need an **embedding service** reachable over HTTP (default
> `http://localhost:8001`), which minnal calls to turn text into vectors. To get
> started, run the companion reference service,
> [minnal0021/embedding_service](https://github.com/minnal0021/embedding_service),
> which serves the **gemma** and **qwen** embedding models, then point
> `SemanticSearchConfig::embedding_service_url` (embedded) or
> `semantic_search.embedding_service_url` (REST server) at it. Each store names
> its model in its schema. Without a service reachable, writes still succeed but
> vector indexing lags and semantic *queries* error. It is set up in
> [§6](#6-semantic-search-doc-store--kv-store--semantic-search).

---

## 1. Select your features

Capabilities are opt-in through cargo features, so you compile — and pull
dependencies for — only what you actually use. There are three to choose from.

**`kv-store`** is the default, and it is really a name for the base engine: the
LSM tree and value log, namespaces, TTL expiry, typed values (via `rkyv`), and
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
directory, and from there put, get, delete, merge, scan by prefix, or walk a key
range.
Keys and values are plain bytes — minnal does not interpret them. A database also
carries any number of **namespaces**, each an isolated keyspace of its own, which
is how you keep unrelated data apart inside one database.

Whichever API you use, call `shutdown()` when you are done: it stops the
background workers and flushes buffered writes.

> **One rule about key bytes:** the storage bucket is chosen by hashing only the
> **first 8 bytes** of the key, so keys that share a constant 8-byte prefix
> (`user:profile:…`) all land in the same bucket — one hot shard instead of
> `num_buckets`. Put the varying part of the key first. At the raw-engine level
> there is no length limit; the typed doc/KV stores of §5–§6 cap string keys at
> 50 bytes.

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

### Read-modify-write with `merge`

`put` overwrites blindly, which is a problem whenever the new value depends on
the old one — a counter, a running total, a set you keep adding to. Doing it by
hand as `get` → compute → `put` silently loses updates: two callers read the same
value, and whichever writes second wipes out the other's increment.

`merge` is that sequence done atomically. It takes the same `(key, value)` as
`put` plus a closure, and the closure decides what actually gets stored:

```rust
let db = Db::open("/tmp/mydb")?;

// Increment a counter by the operand, creating it at zero.
let total = db.merge(b"visits", &1u64.to_le_bytes(), |existing, operand| {
    let current = existing.map_or(0, |b| u64::from_le_bytes(b.try_into().unwrap()));
    let step = u64::from_le_bytes(operand.try_into().unwrap());
    Ok(Some((current + step).to_le_bytes().to_vec()))
})?;
// → Some(new count). Run it from ten threads and none of them is lost.
```

The closure receives `(existing, operand)` — `existing` is `None` when the key is
absent — and its return value drives the write:

| Returns | Effect |
|---|---|
| `Ok(Some(new))` | write `new` |
| `Ok(None)` | delete the key (nothing is written if it was already absent) |
| `Err(e)` | abort — nothing is written, and `e` comes back to you |

`Err` is how you decline a write, and `KVError::MergeAborted` exists for exactly
that so you needn't borrow an error variant that means something else:

```rust
// A counter that refuses to go past a cap.
let result = db.merge(b"quota", &1u64.to_le_bytes(), |existing, operand| {
    let current = existing.map_or(0, |b| u64::from_le_bytes(b.try_into().unwrap()));
    if current >= 100 {
        return Err(minnal_db::KVError::MergeAborted("quota exhausted".into()));
    }
    let step = u64::from_le_bytes(operand.try_into().unwrap());
    Ok(Some((current + step).to_le_bytes().to_vec()))
});
```

`merge_typed` is the same thing over rkyv values, so the closure works with your
own type instead of bytes:

```rust
let total: Option<u64> = db.merge_typed(&"visits".to_string(), &1u64, |current, step| {
    Ok(Some(current.unwrap_or(0) + step))
})?;
```

Both are available on `Db`, `Namespace`, `AsyncDb` and `AsyncNamespace`. Three
things to know:

- **The guarantee is per key, not a transaction.** A merge is indivisible for the
  key it touches and says nothing about any other key. There is still no
  multi-key transaction primitive.
- **`merge` can fail after your closure ran.** `KVError::MergeNotApplied` means
  the write reached the WAL — it is durable and will appear after the next open —
  but did not reach memory, so its result is not readable and a follow-up merge
  would compute from the old value. `put` deliberately returns `Ok` in the same
  situation; `merge` does not, because for a read-modify-write the distinction
  matters.
- **The closure runs while a lock is held**, so keep it quick and never write
  back into the database from inside it — a re-entrant write to the same key
  deadlocks. (Reading is safe, just pointless: you get the same value the
  closure was already handed.)

### The asynchronous API

`AsyncDb` mirrors the same operations for a tokio runtime. The shape is
identical; the calls take owned buffers and are awaited. `merge` additionally
needs its closure to be `Send + 'static`, because it runs on a blocking worker.

```rust
use minnal_db::AsyncDb;

let db = AsyncDb::open("/tmp/mydb").await?;
db.put(b"hello".to_vec(), b"world".to_vec()).await?;
let val = db.get(b"hello".to_vec()).await?;

db.merge(b"visits".to_vec(), 1u64.to_le_bytes().to_vec(), |existing, operand| {
    let current = existing.map_or(0, |b| u64::from_le_bytes(b.try_into().unwrap()));
    let step = u64::from_le_bytes(operand.try_into().unwrap());
    Ok(Some((current + step).to_le_bytes().to_vec()))
})
.await?;

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

    // 4. Query with the predicate DSL (=, !=, <, <=, >, >=, IN, AND, OR, NOT).
    let outcome = db.query_index(DEFAULT_NAMESPACE_ID, r#"status = "active" AND age > 20"#)?;
    if !outcome.degraded_fields.is_empty() {
        eprintln!("an index is missing updates; results may be incomplete");
    }
    for key in outcome.keys {
        if let Some(value) = db.get(&key)? {
            println!("{} => {}", String::from_utf8_lossy(&key), String::from_utf8_lossy(&value));
        }
    }
    // → user:1 and user:3

    db.shutdown()?;
    Ok(())
}
```

A few things are worth knowing before you build on this:

- The `IndexValue` your extractor returns has to match the type you registered
  the field with — one of `Bool`, `Int` (an `i64`), or `Str`.
- A query returns a `QueryOutcome`: the matching `keys`, the `total` number of
  matches, and `degraded_fields`. A field lands in `degraded_fields` when its
  index is known to be missing updates (after a crash that outran its WAL
  retention, say), so a non-empty list means the result may be incomplete.
- When a query can match many records, use
  `query_index_paginated(ns, predicate, offset, limit)`: it resolves only the
  requested window of keys rather than the whole match set.

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
    // The data directory; store schemas are kept inside it.
    let store = DocStore::open("/tmp/docs").await?;

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

#### Key types

`key_type` fixes the shape of every document ID in the store:

| `key_type` | `DocId` variant | Stored as |
|---|---|---|
| `KeyType::U64` | `DocId::U64(u64)` | 8 bytes, big-endian |
| `KeyType::U128` | `DocId::U128(u128)` | 16 bytes, big-endian |
| `KeyType::Uuid` | `DocId::Uuid(u128)` | 16 bytes, big-endian |
| `KeyType::Str` | `DocId::Str(StrKey)` | the UTF-8 bytes, verbatim |

Integer keys are big-endian and string keys verbatim so that in both cases the
byte order matches the natural order of the ID — which is what makes
`scan_range` and `scan_prefix` over IDs meaningful.

**String keys are capped at `MAX_STR_KEY_LEN` (50) UTF-8 bytes and may not be
empty.** The cap is counted in bytes, not characters, so a 20-character key of
3-byte code points is 60 bytes and is rejected. `StrKey::new` is the only
constructor and does the validation, so an invalid key cannot reach storage:

```rust
let schema = DocStoreSchema {
    namespace: "slugs".into(),
    store_type: StoreType::Doc,
    ns_id: None,
    key_type: KeyType::Str,
    attributes: vec![],
    indices: vec![IndexSpec { field: "status".into(), index_type: IndexType::Str }],
    semantic_search_enabled: false,
    embedding_fields: vec![],
};
store.create(schema).await?;

let id = DocId::Str(StrKey::new("acme-corp-2026")?);
store.put("slugs", id, serde_json::json!({ "status": "active" })).await?;

// Prefix scans take the raw key bytes, which for a string store is the string.
let page = store.scan_prefix("slugs", b"acme-".to_vec(), None, 100).await?;
```

The same cap applies to `KvKeyType::Str` keys in a KV store.

> **Key design: vary the first 8 bytes.** The storage bucket is chosen by
> hashing only the **first 8 bytes** of the key. Fixed-width integer and UUID
> keys spread across buckets by construction, but string keys with a constant
> leading prefix (`user:profile:…`, `job_content_…`) all hash to the *same*
> bucket — one hot shard, one value log taking every write, and unbalanced
> compaction. Put the varying part first (`a3f9-user-profile`, not
> `user:profile:a3f9`), or keep the constant prefix under 8 bytes.

Add `"semantic-search"` to the features to enable embed-on-write and vector
search on document (or `value_type = "str"` KV) stores — see §6.

---

## 6. Semantic search (`doc-store` / `kv-store` + `semantic-search`)

With `semantic-search` enabled you can run quantised ANN vector search on either
a **document** store or a **KV store whose value type is `str`**
(`value_type = "str"`). It needs one thing you must supply: an external
**embedding service** (default `http://localhost:8001`). Run the companion
[minnal0021/embedding_service](https://github.com/minnal0021/embedding_service),
which serves the **gemma** and **qwen** models.

Everything that shapes a store's vector index lives in its schema's
`vector_index`: the model (`embedding_model`, default `"gemma"`, sent on every
request to the service), its dimension (`embedding_dim`, default 768), how
documents are chunked, and the search defaults. The model, dimension and
chunking are fixed once semantic search is enabled. There is no centroid file to
download: each store starts with one IVF posting and grows its own partition as
documents arrive, splitting postings that grow too large and merging ones that
deletes empty, in the background.

One last thing to expect before the code: indexing is **asynchronous**. A write
returns as soon as the embed job is enqueued, and a background worker — started
for you by `with_semantic_search` — calls the service, quantises the result and
stores the vector. A query issued immediately after a write may therefore not see
it yet. If the service is down, writes still succeed and the worker keeps
retrying; only vector *indexing* falls behind, while semantic *queries* error.

### Shared setup: attaching a `SemanticSearchContext`

Both storage layers are driven the same way. You build a `SemanticSearchConfig`
naming the embedding service and hand it to the store as a
`SemanticSearchContext`. Attach it once, at startup — it is also what spawns the
embed-worker, the partition maintenance task and the one-shot reconciliation
pass.

```rust
use minnal_db::doc_store::vector_settings::{SearchSpec, VectorIndexSpec};
use minnal_db::semantic_search::service::SemanticSearchConfig;
use minnal_db::{
    AttributeDef, AttributeType, DocId, DocStore, DocStoreSchema, KeyType, KvKeyType,
    KvStoreSchema, KvValueType, Pagination, SemanticSearchContext, StoreType,
};

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
// Only the service is configured here; each store picks its model below.
let config = SemanticSearchConfig {
    embedding_service_url: "http://localhost:8001".into(),
    ..Default::default()
};

// `with_semantic_search` also starts the background embed-worker, the partition
// maintenance and a one-shot startup reconciliation, so attach it once, up front.
let store = DocStore::open("/tmp/sem")
    .await?
    .with_semantic_search(SemanticSearchContext::new(config));
```

### Searching a document store

For documents you mark the store as semantic-search-enabled in its schema and
list which fields should be embedded. Every write to those fields then enqueues
an embed job automatically, and `search_semantic` runs the two-pass ANN search
over what has been indexed so far. Results come back with the document's key
bytes in `document_id`.

```rust
// A doc store whose `body` field is embedded with gemma (768 dimensions).
let schema = DocStoreSchema {
    namespace: "articles".into(),
    store_type: StoreType::Doc,
    ns_id: None,
    key_type: KeyType::U64,
    attributes: vec![AttributeDef { name: "body".into(), attr_type: AttributeType::Str, description: None }],
    indices: vec![],
    semantic_search_enabled: true,
    embedding_fields: vec!["body".into()],
    vector_index: Some(VectorIndexSpec {
        embedding_model: Some("gemma".into()),
        ..Default::default()
    }),
};
store.create(schema).await?;

// Each put enqueues an async embed job and returns immediately.
store.put("articles", DocId::U64(1), serde_json::json!({ "body": "thunder and lightning over the sea" })).await?;
store.put("articles", DocId::U64(2), serde_json::json!({ "body": "a quiet afternoon in the library" })).await?;

// Two-pass ANN search over the stored vectors (top 5). `document_id` is the doc's key bytes.
let top5 = SearchSpec { top_k: Some(5), ..Default::default() };
let hits = store.search_semantic("articles", "a storm at night", &top5, Pagination::default()).await?;
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
// A KV store whose string values are embedded, with the default settings (gemma, 768).
let kv_schema = KvStoreSchema {
    namespace: "notes".into(),
    store_type: StoreType::Kv,
    ns_id: None,
    key_type: KvKeyType::Str,
    value_type: KvValueType::Str,
    semantic_search_enabled: true,
    vector_index: None,
};
store.create_kv(kv_schema).await?;

store.kv_put("notes", &serde_json::json!("n1"), &serde_json::json!("meeting about the Q3 budget")).await?;
store.kv_put("notes", &serde_json::json!("n2"), &serde_json::json!("weekend hiking trip planning")).await?;

// `document_id` here is the raw string key.
let hits = store.kv_search_semantic("notes", "financial planning", &top5, Pagination::default()).await?;
for r in hits.results {
    println!("key {}  score={:.4}", String::from_utf8_lossy(&r.document_id), r.dot_product);
}

store.shutdown().await?; // stops the embed-worker cleanly
# Ok(()) }
```

---

## 7. Choosing a memory allocator

**Use jemalloc in the program that embeds `minnal_db`.** Every request
allocates and frees short-lived buffers, from many threads at once. The system
allocator on Linux, glibc `malloc`, holds on to much of what it frees, and the
same work ran slower with it. Under 16 concurrent clients, measured in the
engine benchmark, switching the same binary to jemalloc gave:

| | glibc `malloc` | jemalloc |
|---|---:|---:|
| Operations per second | 412 | 444 |
| Semantic search, median / p99 | 33 / 91 ms | 23 / 61 ms |
| Resident memory after the load | 5.2 GiB | 1.3 GiB |
| Highest resident memory over the whole run | 5.7 GiB | 2.7 GiB |

Search quality and write latency were unchanged. The full comparison,
including glibc's tuning options, is in
[benchmark.md → Choosing the memory allocator](benchmark.md#choosing-the-memory-allocator).
The `minnal_db_api` server already uses jemalloc.

`minnal_db` is a library, so it does not choose an allocator for you: Rust
lets only the final program set one. You set it in your binary crate, the one
with `main`, never in a library crate that others depend on.

### Switching to jemalloc

Add the allocator to your binary's `Cargo.toml`:

```toml
[dependencies]
minnal_db = "0.2"
tikv-jemallocator = "0.7"
```

and declare it once in `src/main.rs`. Nothing else changes: every allocation in
the process, minnal's included, now goes through jemalloc.

```rust
use minnal_db::Db;

// Every heap allocation in this program, minnal's background workers included,
// now goes through jemalloc.
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db = Db::open("/tmp/mydb")?;
    db.put(b"hello", b"world")?;
    assert_eq!(db.get(b"hello")?.as_deref(), Some(&b"world"[..]));
    db.shutdown()?;
    Ok(())
}
```

jemalloc is a C library that `tikv-jemallocator` compiles from source the first
time you build, so the build needs a C compiler and `make`, as most Rust
toolchains on Linux and macOS already have.

### Checking it is in use

jemalloc reads its options from the `_RJEM_MALLOC_CONF` environment variable
(the `_RJEM_` prefix keeps it apart from any other jemalloc in the process).
`stats_print:true` makes it print its statistics when the program exits, so
seeing them confirms the program runs on jemalloc:

```sh
_RJEM_MALLOC_CONF=stats_print:true ./target/release/my_app 2>&1 | grep -m1 -A1 'Begin jemalloc'
# ___ Begin jemalloc statistics ___
# Version: "5.3.1-0-g81034ce1f1373e37dc865038e1bc8eeecf559ce8"
```

The same variable tunes it. jemalloc returns freed pages to the operating
system about ten seconds after they are freed (`dirty_decay_ms:10000`), and it
does so while the program allocates, so a program that goes idle keeps them
until it is busy again. `background_thread:true` makes a jemalloc thread return
them on time even when the program is idle. `dirty_decay_ms:1000` returns them
sooner, at some CPU cost. The comparison above used the defaults.

### Staying on glibc

If you cannot change the allocator, glibc's own settings help only a little.
`GLIBC_TUNABLES=glibc.malloc.arena_max=8` caps the number of memory regions it
spreads threads over. That cut resident memory to 2 GiB in the benchmark, but
also halved throughput, because threads then wait for each other's locks. Treat
any glibc setting as something to measure on your own workload first.
