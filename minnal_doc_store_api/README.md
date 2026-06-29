# minnal doc store

A lightweight embedded document and key-value store with a REST API, built on top of [minnal_db](../minnal_db).

**Document stores** (`/stores`) — JSON objects stored by a typed primary key, with optional field-level indices and semantic search.

**KV stores** (`/kv-stores`) — schema-lite namespaces for raw typed key-value data, with optional semantic search on string values.

---

## Table of contents

- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Concepts](#concepts)
  - [Namespace](#namespace)
  - [Key types](#key-types)
  - [Indices](#indices)
  - [Attributes](#attributes)
  - [Semantic search (async vector indexing)](#semantic-search-async-vector-indexing)
  - [KV store](#kv-store-concept)
- [REST API reference](#rest-api-reference)
  - [Quick reference](#quick-reference)
  - [Store lifecycle](#store-lifecycle)
  - [Index management](#index-management)
  - [Document CRUD](#document-crud)
  - [Queries](#queries)
  - [KV store lifecycle](#kv-store-lifecycle)
  - [KV CRUD](#kv-crud)
  - [KV range scan](#kv-range-scan)
  - [KV prefix scan](#kv-prefix-scan)
  - [KV semantic search](#kv-semantic-search)
  - [Admin Stores API](#admin-stores-api)
  - [Operational & Storage Metrics](#operational--storage-metrics)
  - [Admin Storage API](#admin-storage-api)
  - [Admin Indices API](#admin-indices-api)
- [Predicate syntax](#predicate-syntax)
- [Error responses](#error-responses)
- [Bulk loading data](#bulk-loading-data)
- [On-disk layout](#on-disk-layout)

---

## Quick start

### 1 — Build

```bash
cargo build --release -p minnal_doc_store_api
```

The binary is at `target/release/minnal_doc_store_api`.

### 2 — Start the server

Run with built-in defaults (data stored under `./data/`):

```bash
./target/release/minnal_doc_store_api
```

Or point it at a config file:

```bash
./target/release/minnal_doc_store_api /path/to/config.toml
```

The server listens on `0.0.0.0:8080` by default.

### 3 — Create a store

```bash
curl -s -X POST http://localhost:8080/stores \
  -H 'Content-Type: application/json' \
  -d '{
    "namespace": "users",
    "store_type": "doc",
    "key_type": "uuid",
    "attributes": [],
    "indices": [
      {"field": "status", "index_type": "str"},
      {"field": "age",    "index_type": "int"}
    ]
  }'
# → 201 Created
```

### 4 — Write a document

```bash
curl -s -X PUT \
  "http://localhost:8080/stores/users/docs/550e8400-e29b-41d4-a716-446655440000" \
  -H 'Content-Type: application/json' \
  -d '{"name": "Alice", "status": "active", "age": 30}'
# → 204 No Content
```

### 5 — Read it back

```bash
curl -s \
  "http://localhost:8080/stores/users/docs/550e8400-e29b-41d4-a716-446655440000"
# → {"name":"Alice","status":"active","age":30}
```

### 6 — Query by index

```bash
curl -s -X POST http://localhost:8080/stores/users/query \
  -H 'Content-Type: application/json' \
  -d '{"predicate": "status = \"active\" AND age >= 18"}'
# → [{"id":"550e8400-...","doc":{"name":"Alice","status":"active","age":30}}]
```

---

## Configuration

All settings have built-in defaults so no config file is required to get started.

```toml
# minnal_doc_store_api.toml

[storage]
db_path    = "./data/db"       # where minnal_db stores its files
schema_dir = "./data/schemas"  # where per-store schema JSON files live

[api]
listen_addr = "0.0.0.0:8080"

# ── tuning (optional) ──────────────────────────────────────────────────────

[sync]
records_per_sync = 1000        # flush WAL after this many writes

[scheduled_tasks]
value_log_gc_interval_secs   = 60
wal_gc_interval_secs         = 60
lsm_compaction_interval_secs = 60
ttl_cleanup_interval_secs    = 3600

[thresholds]
value_log_waste_threshold = 30.0   # GC when >30 % of value-log is stale

[memtable]
max_capacity = 100000          # skip-list capacity (entries)

[sharding]
num_buckets = 8                # value-log shards

[wal]
segment_size_bytes = 67108864  # 64 MiB per WAL segment

[value_log]
page_size_bytes = 67108864     # 64 MiB per value-log page

[vector_index]
retry_wait_secs = 2            # seconds to wait before retrying after an embedding failure
max_retries     = 5            # max attempts per queue entry before it needs manual removal
concurrency     = 4            # max concurrent embedding calls in flight
```

The config file path is resolved in this order:
1. First CLI argument: `minnal_doc_store_api /path/to/config.toml`
2. `MINNAL_CONFIG_FILE` environment variable
3. Built-in defaults (no file needed)

---

## Concepts

A handful of terms recur throughout the API. This section defines them once — namespaces, key types, indices, attributes, and the schema-lite KV store — so the endpoint reference that follows can stay terse.

### Namespace

Each logical document store is called a **namespace**. A namespace:

- Has a unique name made up of ASCII letters, digits, `_`, or `-`
- Stores any number of JSON documents, each identified by a typed primary key
- Has its own schema, indices, and on-disk storage

Multiple namespaces can coexist in the same server instance.

### Key types

The primary key type is chosen at store creation and cannot be changed.

| `key_type` | Description | URL format |
|-----------|-------------|------------|
| `uuid`    | 128-bit UUID | `550e8400-e29b-41d4-a716-446655440000` |
| `u64`     | Unsigned 64-bit integer | `42` |
| `u128`    | Unsigned 128-bit integer | `340282366920938463463374607431768211455` |

Keys are stored in big-endian byte order so that numeric range scans return results in ascending order.

### Indices

Indices enable fast predicate queries (no full collection scan). Each index:

- Covers a single top-level JSON field
- Has a value type: `str`, `int`, or `bool`
- Is built as a [RoaringBitmap](https://roaringbitmap.org/) field index under the hood

**Limits:** up to **5 indices** per namespace.

Adding an index to a store that already has documents triggers a **background rebuild**. The API returns `202 Accepted` immediately; the index is live for new writes straight away and historical documents are back-filled asynchronously. Use the [progress endpoint](#get-adminindicesprogress) to monitor the rebuild.

Rebuilds are **restartable** — if the server shuts down mid-rebuild, it automatically resumes from the last checkpoint on the next startup.

### Attributes

Non-indexed fields can be declared as **attributes** in the schema. Attributes:

- Are not indexed and do not support predicate queries
- Can be added, removed, or type-updated at any time via `PATCH /stores/{ns}/schema`
- Serve as schema documentation and allow future type-checking

You cannot amend an indexed field's declaration directly — drop the index first, amend, then re-add if needed.

### Semantic search (async vector indexing)

When a store has `semantic_search_enabled = true`, embedding happens **asynchronously** and never blocks a write:

1. A `PUT`/upsert writes the document (or KV value) first, then enqueues a pending-embed marker in a durable system queue as a separate write. The request returns immediately — it does not contact the embedding service.
2. A background worker drains the queue: it calls the embedding service, writes the quantised `VectorIndex` to the companion namespaces (`{ns}_sparse_vector`, `{ns}_dense_vector`, `{ns}_sparse_vector_meta`), then removes the queue entry. The worker is idempotent and retries failures with back-off (`[vector_index]` config); the vector index is **eventually consistent** with the store.

**Robust search results.** The document and its vector index are separate writes that can drift — a crash window on write, or a delete racing the async indexer can leave an index entry whose document is gone. To keep results trustworthy, semantic search **fetches each candidate's document and drops any that no longer exist**, so a search hit always resolves to a live document; orphaned index entries never surface as dangling results.

**Reconciliation.** Because the document write and the queue enqueue are two separate writes, a crash between them could leave a document written but not queued; likewise the quantised vector payloads are written with `put_no_wal`, so a crash before the memtable flush can drop a just-indexed vector (or flush one half and lose the other). Reconciliation heals both: across every semantic-search-enabled namespace it re-enqueues any document missing **both** a *complete* committed vector index and a pending queue entry. "Complete" means **both** the sparse-meta and dense entries are present — a document with only one half is a partially committed index and is re-enqueued so the re-embed regenerates the missing half. This mirrors how field indices self-heal via WAL replay, but routes the recovered work into the async queue. It runs **automatically as a background task on store startup** (presence-only: a cheap count short-circuit — nothing queued **and** both the sparse-meta and dense key counts already cover the live-key count — skips namespaces already fully covered, so a clean boot is inexpensive). The on-demand [`POST /admin/indices/vector/reconcile`](#post-adminindicesvectorreconcile) endpoint runs a stronger **validating** variant that *also* re-enqueues documents whose committed bytes are present but **fail to deserialize** (corruption the presence check cannot see) — it deserializes every entry and skips the short-circuit, so it is a full scan that runs in the background (`202 Accepted`). To force a full rebuild of one namespace regardless, use [`POST /admin/indices/{ns}/vector/reindex-all`](#post-adminindicesnsvectorreindex-all).

Requires the external embedding service and a cluster index (`semantic_search.cluster_path`) to be available at startup.

### KV store concept

A **KV store** is a schema-lite namespace managed under `/kv-stores`. Unlike a document store it has:

- No field indices and no predicate queries
- Typed keys (`str` or `int`) and typed values (`str`, `int`, `f32`, `vec_f32`)
- Optional semantic search when `value_type = str`

KV stores share the same underlying minnal_db namespace registry, WAL, LSM compaction, and value-log GC as document stores. The schema is persisted alongside doc-store schemas in `schema_dir` and distinguished on disk by a mandatory `store_type` field (`"doc"` vs `"kv"`) that every schema must declare.

**Key types:**

| `key_type` | URL path segment | Storage |
|-----------|-----------------|---------|
| `str` | any UTF-8 string | raw UTF-8 bytes |
| `int` | decimal integer | big-endian `i64` (ordered scans work correctly) |

**Value types:**

| `value_type` | JSON body | Bytes |
|-------------|-----------|-------|
| `str` | JSON string | UTF-8 |
| `int` | JSON integer | little-endian `i64` |
| `f32` | JSON number | little-endian `f32` |
| `vec_f32` | JSON array of numbers | packed little-endian `f32` array |

---

## REST API reference

All request and response bodies are JSON, and errors are returned as `{"error": "<message>"}`. The endpoints fall into two families — `/stores/*` for document stores and `/kv-stores/*` for KV stores — plus a set of `/admin/*` routes for backup, diagnostics, and bulk index operations. The quick-reference tables below list every endpoint at a glance; the sections that follow document each one in detail, grouped by task.

### Quick reference

**Document stores:**

| Method | Path | Response | Purpose |
|--------|------|----------|---------|
| `POST` | `/stores` | `201` | Create a document store with schema |
| `GET` | `/stores` | `200` | List all document stores |
| `DELETE` | `/stores/{ns}` | `204` | Drop a document store and all its data |
| `GET` | `/stores/{ns}/schema` | `200` | Fetch the current schema |
| `PATCH` | `/stores/{ns}/schema` | `204` | Add / update / remove a non-indexed attribute |
| `GET` | `/stores/{ns}/indices` | `200` | List indices and vector campaign status |
| `POST` | `/stores/{ns}/indices` | `202` | Add an index (background rebuild if data exists) |
| `DELETE` | `/stores/{ns}/indices/vector` | `202` | Drop the vector index (background cleanup) |
| `DELETE` | `/stores/{ns}/indices/{field}` | `202` | Drop a field index (background cleanup) |
| `PUT` | `/stores/{ns}/docs/{id}` | `204` | Upsert a document |
| `GET` | `/stores/{ns}/docs/{id}` | `200` | Retrieve a document by primary key |
| `DELETE` | `/stores/{ns}/docs/{id}` | `204` | Delete a document |
| `GET` | `/stores/{ns}/docs?start=&end=` | `200` | Range scan in primary-key order |
| `POST` | `/stores/{ns}/query` | `200` | Index predicate query |

**KV stores:**

| Method | Path | Response | Purpose |
|--------|------|----------|---------|
| `POST` | `/kv-stores` | `201` | Create a KV store |
| `GET` | `/kv-stores` | `200` | List all KV stores |
| `DELETE` | `/kv-stores/{ns}` | `204` | Drop a KV store and all its data |
| `GET` | `/kv-stores/{ns}/schema` | `200` | Fetch the current KV-store schema |
| `PUT` | `/kv-stores/{ns}/kv/{key}` | `204` | Set a value |
| `GET` | `/kv-stores/{ns}/kv/{key}` | `200` | Get a value by key |
| `DELETE` | `/kv-stores/{ns}/kv/{key}` | `204` | Delete a key |
| `GET` | `/kv-stores/{ns}/kv?start=&end=` | `200` | Range scan in key order |
| `GET` | `/kv-stores/{ns}/kv/prefix?prefix=` | `200` | Prefix scan (`key_type = str` most useful) |
| `POST` | `/kv-stores/{ns}/semantic-search` | `200` | ANN search (`value_type = str` only) |

**Schema export / import (admin):**

| Method | Path | Response | Purpose |
|--------|------|----------|---------|
| `GET` | `/admin/stores/{ns}/schema/export` | `200` | Download a doc-store schema as a JSON attachment |
| `POST` | `/admin/stores/import` | `201` | Create a doc store from an exported schema |
| `GET` | `/admin/stores/{ns}/row-count` | `200` | Number of documents in a doc-store namespace |
| `GET` | `/admin/kv-stores/{ns}/schema/export` | `200` | Download a KV-store schema as a JSON attachment |
| `POST` | `/admin/kv-stores/import` | `201` | Create a KV store from an exported schema |

---

### Store lifecycle

These endpoints create, list, inspect, and drop document stores, and amend their non-indexed attributes. A store must exist before any document can be written to it.

#### `GET /stores`

List all stores.

```bash
curl http://localhost:8080/stores
```

```json
[
  {
    "namespace": "users",
    "store_type": "doc",
    "key_type": "uuid",
    "attributes": [],
    "indices": [
      {"field": "status", "index_type": "str"},
      {"field": "age",    "index_type": "int"}
    ]
  }
]
```

---

#### `POST /stores`

Create a new store.

**Request body:**

| Field        | Type             | Required | Description                          |
|-------------|-----------------|----------|--------------------------------------|
| `namespace`  | string           | yes      | Unique name (`[a-zA-Z0-9_-]+`)       |
| `key_type`   | `uuid`/`u64`/`u128` | yes  | Primary key type                     |
| `indices`    | array            | yes      | Zero to 5 index specs (may be empty) |
| `attributes` | array            | no       | Non-indexed field declarations       |

**Index spec:**

```json
{"field": "status", "index_type": "str"}
```

`index_type` must be `str`, `int`, or `bool`.

**Attribute definition:**

```json
{"name": "email", "attr_type": "str", "description": "user email address"}
```

**Example:**

```bash
curl -X POST http://localhost:8080/stores \
  -H 'Content-Type: application/json' \
  -d '{
    "namespace": "orders",
    "store_type": "doc",
    "key_type": "u64",
    "indices": [
      {"field": "state",      "index_type": "str"},
      {"field": "amount_usd", "index_type": "int"},
      {"field": "paid",       "index_type": "bool"}
    ],
    "attributes": [
      {"name": "customer_id", "attr_type": "str"}
    ]
  }'
# → 201 Created
```

---

#### `DELETE /stores/{ns}`

Permanently delete a store and all its documents, indices, and schema.

```bash
curl -X DELETE http://localhost:8080/stores/orders
# → 204 No Content
```

---

#### `GET /stores/{ns}/schema`

Fetch the current schema for a namespace.

```bash
curl http://localhost:8080/stores/users/schema
```

```json
{
  "namespace": "users",
  "store_type": "doc",
  "key_type": "uuid",
  "indices": [
    {"field": "status", "index_type": "str"},
    {"field": "age",    "index_type": "int"}
  ],
  "attributes": [
    {"name": "email", "attr_type": "str", "description": "contact email"}
  ],
  "semantic_search_enabled": false,
  "embedding_fields": []
}
```

Returns `404 Not Found` if the namespace does not exist.

---

#### `PATCH /stores/{ns}/schema`

Amend the schema. Cannot be used to change indexed fields — drop the index first.

**Removing an embedding field:** if `remove_attribute` targets a field listed in `embedding_fields` and it is the last such field, semantic search is disabled synchronously and all vector index data (queue entries and the `{ns}_sparse_vector`, `{ns}_dense_vector`, `{ns}_sparse_vector_meta` companion namespaces) is cleaned up in a background task. This is equivalent to calling `DELETE /stores/{ns}/indices/vector`.

**Add an attribute:**

```bash
curl -X PATCH http://localhost:8080/stores/users/schema \
  -H 'Content-Type: application/json' \
  -d '{"op": "add_attribute", "name": "email", "attr_type": "str", "description": "contact email"}'
# → 204 No Content
```

**Update an attribute:**

```bash
curl -X PATCH http://localhost:8080/stores/users/schema \
  -H 'Content-Type: application/json' \
  -d '{"op": "update_attribute", "name": "email", "attr_type": "str", "description": "primary email"}'
# → 204 No Content
```

**Remove an attribute:**

```bash
curl -X PATCH http://localhost:8080/stores/users/schema \
  -H 'Content-Type: application/json' \
  -d '{"op": "remove_attribute", "name": "email"}'
# → 204 No Content
```

**Add an embedding attribute (enable the vector index):** declares a `str`
attribute that feeds the namespace's vector index and turns on semantic search.

```bash
curl -X PATCH http://localhost:8080/stores/users/schema \
  -H 'Content-Type: application/json' \
  -d '{"op": "add_embedding_attribute", "name": "bio", "description": "embedded text"}'
# → 204 No Content
```

A namespace has **at most one vector index**. Once semantic search is enabled,
this op returns **`409 Conflict`** (`SemanticSearchAlreadyEnabled`) — drop the
vector index first (`DELETE /stores/{ns}/indices/vector`) before re-adding.

**Enable the vector index over multiple fields:** `add_embedding_attribute` only
adds one field. To create a multi-field vector index after the store exists, use
`enable_vector_index` with the full field list in a single call (valid only when
no vector index is present):

```bash
curl -X PATCH http://localhost:8080/stores/users/schema \
  -H 'Content-Type: application/json' \
  -d '{"op": "enable_vector_index", "fields": ["bio", "headline"]}'
# → 204 No Content
```

So the post-create workflow to **change which fields are embedded** is: drop the
vector index (`DELETE /stores/{ns}/indices/vector` — data is preserved) then
`enable_vector_index` with the new field set. Like `add_embedding_attribute`, it
returns `409` if a vector index already exists, and `400` if `fields` is empty,
contains duplicates, or names an existing index/attribute.

| `op` value                | Required fields          |
|---------------------------|--------------------------|
| `add_attribute`           | `name`, `attr_type`      |
| `update_attribute`        | `name`, `attr_type`      |
| `remove_attribute`        | `name`                   |
| `add_embedding_attribute` | `name`                   |
| `enable_vector_index`     | `fields` (non-empty)     |

All ops except `enable_vector_index` accept an optional `"description"` string.

---

### Index management

These endpoints add and drop the secondary indices that make predicate queries fast, including the vector index that powers semantic search. Adding or dropping an index on a store that already holds documents runs as a background job and returns `202 Accepted`; monitor it through the [admin progress endpoints](#admin-indices-api).

#### `GET /stores/{ns}/indices`

List all active or recently completed field-index builds and the vector campaign status (if any) for the namespace.

```bash
curl http://localhost:8080/stores/users/indices
```

Returns an array of `IndexBuildSnapshot` objects. Use `GET /admin/indices/{ns}/progress` for the live monitoring view including queue depth.

---

#### `POST /stores/{ns}/indices`

Add a new field index. If the store already has documents the rebuild runs in the background.

```bash
curl -X POST http://localhost:8080/stores/users/indices \
  -H 'Content-Type: application/json' \
  -d '{"field": "country", "index_type": "str"}'
# → 202 Accepted
```

Returns `409 Conflict` if the field is already indexed or a rebuild is in progress. Returns `400 Bad Request` (`TooManyIndices`) if the namespace is already at the **5-index limit** — the cap is enforced here, not just at create/import time. Monitor progress via `GET /admin/indices/{ns}/progress`.

---

#### `DELETE /stores/{ns}/indices/vector`

Disable semantic search and drop all vector index data for the namespace. The schema is updated synchronously before the background cleanup runs, preventing new embeddings from being enqueued during cleanup. Returns `202 Accepted`.

```bash
curl -X DELETE http://localhost:8080/stores/users/indices/vector
# → 202 Accepted
```

Returns `409 Conflict` if a vector cleanup or reindex campaign is already in progress. Returns `422 Unprocessable Entity` if semantic search is not enabled for the namespace.

---

#### `DELETE /stores/{ns}/indices/{field}`

Drop a field index. The bitmap files are deleted in a background task; the field is demoted to a plain attribute in the schema. Returns `202 Accepted`.

```bash
curl -X DELETE http://localhost:8080/stores/users/indices/country
# → 202 Accepted
```

Returns `409 Conflict` if an attribute index operation is already active for this namespace.

---

### Document CRUD

All document endpoints accept the `{id}` path segment formatted according to the store's `key_type`:

| `key_type` | Example path segment |
|-----------|----------------------|
| `uuid`    | `550e8400-e29b-41d4-a716-446655440000` |
| `u64`     | `42` |
| `u128`    | `99999999999999999999` |

---

#### `PUT /stores/{ns}/docs/{id}`

Insert or replace a document (upsert).

```bash
curl -X PUT "http://localhost:8080/stores/users/docs/1" \
  -H 'Content-Type: application/json' \
  -d '{"name": "Bob", "status": "inactive", "age": 25}'
# → 204 No Content
```

---

#### `GET /stores/{ns}/docs/{id}`

Retrieve a document by its primary key.

```bash
curl "http://localhost:8080/stores/users/docs/1"
# → {"name":"Bob","status":"inactive","age":25}
```

Returns `404 Not Found` if no document with that ID exists.

---

#### `DELETE /stores/{ns}/docs/{id}`

Delete a document. No-op if the document does not exist.

```bash
curl -X DELETE "http://localhost:8080/stores/users/docs/1"
# → 204 No Content
```

---

### Queries

There are two ways to retrieve documents beyond a single primary-key lookup: a **range scan** that walks documents in key order, and an **index predicate query** that returns the documents matching a boolean expression over indexed fields. (Semantic similarity search is documented separately, under the KV and store sections.)

#### `GET /stores/{ns}/docs?start=&end=`

Range scan over documents in primary-key order. **Cursor-paginated** — each page
resolves only its own documents from the value log, so memory stays bounded
regardless of how many keys match.

| Parameter | Required | Description                                     |
|-----------|----------|-------------------------------------------------|
| `start`   | yes      | First key to include (inclusive)                |
| `end`     | no       | Last key to include (exclusive). Omit for open-ended scan |
| `limit`   | no       | Max documents per page (default: 20)            |
| `cursor`  | no       | Opaque token from a prior page's `next_cursor`; omit for the first page |

```bash
# u64 store — first page of documents with IDs 10 through 99
curl "http://localhost:8080/stores/orders/docs?start=10&end=100&limit=2"
```

```bash
# uuid store — scan from a given UUID to the end of the collection
curl "http://localhost:8080/stores/users/docs?start=00000000-0000-0000-0000-000000000000"
```

Response — a page of `{id, doc}` pairs ordered by key, plus `next_cursor`
(`null` when the scan is exhausted). To fetch the next page, pass the returned
`next_cursor` back as `cursor`:

```json
{
  "results": [
    {"id": 10, "doc": {"state": "shipped", "amount_usd": 50, "paid": true}},
    {"id": 11, "doc": {"state": "pending", "amount_usd": 20, "paid": false}}
  ],
  "next_cursor": "000000000000000c"
}
```

---

#### `POST /stores/{ns}/query`

Query documents using an index predicate. Only indexed fields may appear in the predicate.

Offset-paginated (the bitmap index gives an exact `total` and random page access). Query params `page_no` / `page_size` (defaults 1 / 20) override the same-named body fields; `limit` is accepted as an alias for `page_size` (`page_size` wins if both are given).

```bash
curl -X POST "http://localhost:8080/stores/orders/query?limit=5" \
  -H 'Content-Type: application/json' \
  -d '{"predicate": "state = \"shipped\" AND paid = true"}'
```

Response — `{id, doc}` pairs plus the pagination envelope:

```json
{
  "results": [
    {"id": 10, "doc": {"state": "shipped", "amount_usd": 50, "paid": true}}
  ],
  "page_no": 1,
  "page_size": 5,
  "total": 1
}
```

---

### KV store lifecycle

The KV-store endpoints mirror the document-store lifecycle, but for schema-lite namespaces under `/kv-stores`: create, list, inspect, and drop. A KV store fixes its key and value types at creation and has no field indices — see the [KV store concept](#kv-store-concept) for the available type combinations.

#### `GET /kv-stores`

List all KV stores.

```bash
curl http://localhost:8080/kv-stores
```

```json
[
  {
    "namespace": "session-cache",
    "store_type": "kv",
    "key_type": "str",
    "value_type": "str",
    "semantic_search_enabled": false
  }
]
```

---

#### `POST /kv-stores`

Create a new KV store.

**Request body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `namespace` | string | yes | Unique name (`[a-zA-Z0-9_-]+`) |
| `key_type` | `str` / `int` | yes | Key type |
| `value_type` | `str` / `int` / `f32` / `vec_f32` | yes | Value type |
| `semantic_search_enabled` | bool | no | Enable ANN search (requires `value_type = str` and a cluster index at startup) |

```bash
curl -X POST http://localhost:8080/kv-stores \
  -H 'Content-Type: application/json' \
  -d '{"namespace": "session-cache", "store_type": "kv", "key_type": "str", "value_type": "str"}'
# → 201 Created

# With semantic search
curl -X POST http://localhost:8080/kv-stores \
  -H 'Content-Type: application/json' \
  -d '{
    "namespace": "product-descriptions",
    "store_type": "kv",
    "key_type": "str",
    "value_type": "str",
    "semantic_search_enabled": true
  }'
# → 201 Created
```

Returns `409 Conflict` if a store with that namespace already exists (doc or KV).

---

#### `DELETE /kv-stores/{ns}`

Permanently delete a KV store. Irreversible.

```bash
curl -X DELETE http://localhost:8080/kv-stores/session-cache
# → 204 No Content
```

---

#### `GET /kv-stores/{ns}/schema`

Fetch the current schema for a KV store as JSON.

```bash
curl http://localhost:8080/kv-stores/session-cache/schema
```

```json
{
  "namespace": "session-cache",
  "ns_id": 7,
  "store_type": "kv",
  "key_type": "str",
  "value_type": "str",
  "semantic_search_enabled": false
}
```

Returns `404 Not Found` if the namespace does not exist as a KV store. To download the schema as a file (e.g. for backup or migration to another deployment), use [`GET /admin/kv-stores/{ns}/schema/export`](#get-adminkv-storesnsschemaexport).

---

### KV CRUD

All KV endpoints use `{key}` as a URL path segment. The segment is interpreted according to the store's `key_type`:

| `key_type` | Example path segment |
|-----------|----------------------|
| `str` | `user-42`, `hello world` (URL-encode spaces) |
| `int` | `42`, `-100` |

The request / response body is a raw JSON value matching `value_type`:

| `value_type` | Example body |
|-------------|-------------|
| `str` | `"hello"` |
| `int` | `42` |
| `f32` | `3.14` |
| `vec_f32` | `[1.0, -0.5, 0.25]` |

---

#### `PUT /kv-stores/{ns}/kv/{key}`

Insert or replace a value (upsert). When `semantic_search_enabled = true` the value text is enqueued for async embedding — the request returns immediately and the vector index is updated in the background.

```bash
curl -X PUT http://localhost:8080/kv-stores/session-cache/kv/user-42 \
  -H 'Content-Type: application/json' \
  -d '"eyJhbGciOiJIUzI1NiJ9..."'
# → 204 No Content

# vec_f32 example
curl -X PUT http://localhost:8080/kv-stores/embeddings/kv/doc-1 \
  -H 'Content-Type: application/json' \
  -d '[0.12, -0.45, 0.89]'
# → 204 No Content
```

---

#### `GET /kv-stores/{ns}/kv/{key}`

Retrieve a value by key. Returns the value as a JSON body.

```bash
curl http://localhost:8080/kv-stores/session-cache/kv/user-42
# → "eyJhbGciOiJIUzI1NiJ9..."
```

Returns `404 Not Found` if the key does not exist.

---

#### `DELETE /kv-stores/{ns}/kv/{key}`

Delete a key. No-op if the key does not exist. Also removes the companion vector index entry when semantic search is enabled.

```bash
curl -X DELETE http://localhost:8080/kv-stores/session-cache/kv/user-42
# → 204 No Content
```

---

### KV range scan

`GET /kv-stores/{ns}/kv?start=&end=`

Scan all entries whose key falls in `[start, end)`, returned in ascending key
order. **Cursor-paginated** — each page resolves only its own values, so memory
stays bounded regardless of how many keys match.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `start` | yes | First key to include (inclusive). Format matches the store's `key_type`: a plain string for `str`, a decimal integer for `int`. |
| `end` | no | Last key to include (exclusive). Omit for an open-ended scan to the last key. |
| `limit` | no | Max entries per page (default: 20) |
| `cursor` | no | Opaque token from a prior page's `next_cursor`; omit for the first page. |

```bash
# str key store — scan all sessions for users with IDs "user-10" through "user-20"
curl "http://localhost:8080/kv-stores/session-cache/kv?start=user-10&end=user-21"

# int key store — fetch entries with keys 100 through 199
curl "http://localhost:8080/kv-stores/counters/kv?start=100&end=200"

# first page of 50, then follow next_cursor for the next page
curl "http://localhost:8080/kv-stores/session-cache/kv?start=user-&limit=50"
curl "http://localhost:8080/kv-stores/session-cache/kv?start=user-&limit=50&cursor=757365722d3530"
```

Response — a page of `{key, value}` pairs ordered by key, plus `next_cursor`
(`null` when the scan is exhausted). Pass it back as `cursor` for the next page:

```json
{
  "results": [
    {"key": "user-10", "value": "eyJhbGciOiJIUzI1NiJ9..."},
    {"key": "user-11", "value": "eyJhbGciOiJIUzI1NiJ9..."}
  ],
  "next_cursor": "757365722d3132"
}
```

`key` is rendered as a JSON string for `key_type = str` and a JSON number for `key_type = int`. `value` matches the store's `value_type`.

---

### KV prefix scan

`GET /kv-stores/{ns}/kv/prefix?prefix=`

Scan all entries whose key starts with `prefix`. Most useful for `key_type = str` stores where keys share a common string prefix (e.g. `"user-"` to find all user entries).

For `key_type = int`, `prefix` is parsed as a decimal integer and serialised as a big-endian 8-byte value — this matches only the exact key and is rarely more useful than a point lookup; use range scan for numeric key ranges instead.

Results are returned in ascending key order. **Cursor-paginated** like the range
scan — each page resolves only its own values.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `prefix` | yes | Key prefix. Plain UTF-8 string for `key_type = str`; decimal integer for `key_type = int`. |
| `limit` | no | Max entries per page (default: 20) |
| `cursor` | no | Opaque token from a prior page's `next_cursor`; omit for the first page. |

```bash
# Find all session-cache entries whose key starts with "user-"
curl "http://localhost:8080/kv-stores/session-cache/kv/prefix?prefix=user-"

# Paginate through a large prefix result set via next_cursor
curl "http://localhost:8080/kv-stores/session-cache/kv/prefix?prefix=user-&limit=100"
curl "http://localhost:8080/kv-stores/session-cache/kv/prefix?prefix=user-&limit=100&cursor=757365722d393939"
```

Response — same format as range scan:

```json
{
  "results": [
    {"key": "user-10",  "value": "eyJhbGciOiJIUzI1NiJ9..."},
    {"key": "user-42",  "value": "eyJhbGciOiJIUzI1NiJ9..."},
    {"key": "user-999", "value": "eyJhbGciOiJIUzI1NiJ9..."}
  ],
  "next_cursor": null
}
```

---

### KV semantic search

`POST /kv-stores/{ns}/semantic-search`

Requires `semantic_search_enabled = true` and `value_type = str` on the KV store. The stored string values are used as the text that was embedded at write time.

**Request body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | yes | Query text to embed and search |
| `top_k` | integer | no | Maximum number of candidates to return (default: all in probed clusters) |
| `page_size` | integer | no | Page size (default: 20). Also accepts `limit` as an alias (query param); `page_size` wins if both are given. |
| `page_no` | integer | no | 1-based page number (default: 1) |

```bash
curl -X POST http://localhost:8080/kv-stores/product-descriptions/semantic-search \
  -H 'Content-Type: application/json' \
  -d '{"query": "lightweight waterproof running shoes", "top_k": 5}'
```

Response:

```json
{
  "results": [
    {
      "key": "sku-1042",
      "dot_product": 0.93,
      "error_bound": 0.02,
      "cluster_id": 7,
      "is_primary": true,
      "value": "Ultra-light trail runner with waterproof membrane"
    }
  ],
  "page_no": 1,
  "page_size": 20,
  "total": 1
}
```

| Field | Meaning |
|-------|---------|
| `key` | The key, serialised as a JSON string or number according to `key_type` |
| `dot_product` | Estimated cosine similarity (higher = more similar) |
| `error_bound` | Theoretical max deviation of the estimate from the true dot product |
| `cluster_id` | IVF cluster this entry was indexed under |
| `is_primary` | `true` if the entry's cluster is the closest cluster to the query |
| `value` | The stored value, or `null` if the key no longer exists |

---

### Admin Stores API

Schema export / import for backup and for migrating a namespace definition between deployments. The exported file is the schema only — it does **not** include documents or values. Importing creates an empty store; reload the data separately (e.g. via the [bulk loader](#bulk-loading-data) or the CRUD endpoints).

#### `GET /admin/stores/{ns}/schema/export`

Download a doc-store schema as a JSON file attachment (`Content-Disposition: attachment; filename="{ns}-schema.json"`).

```bash
curl -OJ http://localhost:8080/admin/stores/users/schema/export
# → writes users-schema.json
```

Returns `404 Not Found` if the namespace does not exist as a doc store.

---

#### `POST /admin/stores/import`

Create a doc store from a previously exported schema. The internal `ns_id` is stripped and reassigned, so an exported schema can be imported into a fresh deployment. Equivalent to `POST /stores` with the schema body.

```bash
curl -X POST http://localhost:8080/admin/stores/import \
  -H 'Content-Type: application/json' \
  --data-binary @users-schema.json
# → 201 Created
```

Returns `409 Conflict` if a store with that namespace already exists.

---

#### `GET /admin/kv-stores/{ns}/schema/export`

Download a KV-store schema as a JSON file attachment (`Content-Disposition: attachment; filename="{ns}-kv-schema.json"`).

```bash
curl -OJ http://localhost:8080/admin/kv-stores/session-cache/schema/export
# → writes session-cache-kv-schema.json
```

Returns `404 Not Found` if the namespace does not exist as a KV store.

---

#### `POST /admin/kv-stores/import`

Create a KV store from a previously exported schema. The internal `ns_id` is stripped and reassigned. Equivalent to `POST /kv-stores` with the schema body.

```bash
curl -X POST http://localhost:8080/admin/kv-stores/import \
  -H 'Content-Type: application/json' \
  --data-binary @session-cache-kv-schema.json
# → 201 Created
```

Returns `409 Conflict` if a store with that namespace already exists.

---

#### `GET /admin/stores/{ns}/row-count`

Number of documents in a doc-store namespace.

```bash
curl http://localhost:8080/admin/stores/users/row-count
```

```json
{ "namespace": "users", "count": 1280 }
```

---

### Operational & Storage Metrics

This is a consolidated reference for everything reported by the metrics/diagnostic
endpoints under `/admin/storage`, with a **"survives restart?"** column for each
value. There are two distinct kinds:

- **Operational metrics** (`GET /admin/storage/ops-metrics`) — engine-wide runtime
  counters of *what the engine is doing* (throughput, read-path efficiency,
  compaction/GC activity). They are in-memory `AtomicU64`s created fresh in
  `Database::open`, so they are **not persisted** — every restart starts them at
  zero. Cumulative since process start; sample twice to compute a rate.
- **Storage metrics** (the other `/admin/storage/*` GET endpoints) — structural
  snapshots of *how big the engine is* and *how much dead space it holds*. These
  are **recomputed from on-disk state** (WAL metadata, LSM manifests, value-log
  metadata, index blob stores) on every request, so a restart does not reset them
  — they reflect whatever is durably on disk.

**"Survives restart?" legend:** **No** = in-memory only, zero at process start.
**Yes** = read/derived from persisted on-disk state, so unaffected by a restart.

#### Endpoint overview

| Endpoint | Kind | Survives restart? |
|----------|------|-------------------|
| `GET /admin/storage/ops-metrics` | Runtime counters | **No** (see startup-repopulation note) |
| `GET /admin/storage/health` | Liveness / uptime | `uptime_s` **No**; `status` n/a |
| `GET /admin/storage/stats` | Value-log aggregate | **Yes** |
| `GET /admin/storage/wal` | WAL metadata | **Yes** |
| `GET /admin/storage/lsm` | LSM manifest | **Yes** (except `in_memory.*` → **No**) |
| `GET /admin/storage/value-log` | Value-log per shard | **Yes** |
| `GET /admin/storage/value-log/{ns}/pages` | Value-log per page | **Yes** |
| `GET /admin/storage/index-waste` | Field-index dead space | **Yes** (derived from on-disk state, not a stored counter) |
| `GET /admin/storage/namespaces` | Registry + schema | **Yes** |
| `GET /admin/storage/kv-namespaces` | Engine KV namespaces | **Yes** |
| `GET /admin/storage/stores/{ns}/kv-meta` | Per-ns LSM+value-log | **Yes** (except `in_memory.*`) |
| `GET /admin/storage/kv-stores/{ns}/kv-meta` | Per-ns LSM+value-log | **Yes** (except `in_memory.*`) |
| `GET /admin/storage/system/stores` | System namespaces | **Yes** |
| `GET /admin/storage/system/stores/{ns}/meta` | One system store | **Yes** (except `in_memory.*`) |

#### Operational metrics — `GET /admin/storage/ops-metrics`

All counters below are **in-memory and reset to zero on restart**. They are grouped
in the response under `reads`, `lsm_lookups`, `writes`, `compaction`, `gc`, plus a
top-level `uptime_s`.

| Field | Group | Meaning | Survives restart? |
|-------|-------|---------|-------------------|
| `uptime_s` | (top) | Seconds since the server process started | **No** |
| `reads` | reads | User-facing point reads (`GET` by key) | **No** |
| `read_hits` | reads | Reads that found a live value | **No** |
| `read_misses` | reads | Reads that found nothing (absent/tombstoned) | **No** |
| `read_hit_ratio` | reads | `read_hits / reads` (derived) | **No** |
| `scans` | reads | Multi-key scans (range/prefix) executed | **No** |
| `scan_rows` | reads | Total rows returned across all scans | **No** |
| `lookups` | lsm_lookups | LSM point lookups (≥ `reads` — also counts GC-validation reads and WAL-replay probes) | **No** |
| `fast_path_hits` | lsm_lookups | Lookups served by the active-memtable fast path (no lower-layer scan) | **No** |
| `fast_path_hit_ratio` | lsm_lookups | `fast_path_hits / lookups` (derived) | **No** |
| `l0_probes` | lsm_lookups | Lookups that scanned at least one L0 SSTable | **No** |
| `l1_probes` | lsm_lookups | Lookups that scanned the L1 SSTable (not bloom-rejected) | **No** |
| `bloom_rejects` | lsm_lookups | L1 lookups short-circuited by the bloom filter ("definitely absent") | **No** |
| `puts` | writes | WAL-backed upserts applied | **No** |
| `deletes` | writes | WAL-backed deletes applied | **No** |
| `no_wal_puts` | writes | Upserts written bypassing the WAL (`skip_wal`, vector payloads, query-embedding cache) | **No** |
| `no_wal_deletes` | writes | Deletes written bypassing the WAL (query-embedding cache populate/clear) | **No** |
| `wal_bytes_appended` | writes | Total bytes appended to the WAL | **No** |
| `wal_fsyncs` | writes | WAL fsyncs (one per WAL-backed write — durability cost) | **No** |
| `apply_failures` | writes | In-memory applies that failed after retry (data still durable in WAL) | **No** |
| `memtable_flushes` | compaction | Memtable → L0 SSTable flushes | **No** |
| `l0_l1_compactions` | compaction | L0 → L1 compactions run | **No** |
| `compaction_bytes_merged` | compaction | Total bytes merged during compactions | **No** |
| `compaction_duration_ms` | compaction | Cumulative time spent compacting (ms) | **No** |
| `vlog_gc_runs` | gc | Value-log GC passes run | **No** |
| `vlog_gc_duration_ms` | gc | Cumulative value-log GC time (ms) | **No** |
| `wal_gc_runs` | gc | WAL GC passes run | **No** |
| `wal_segments_deleted` | gc | WAL segments reclaimed by GC | **No** |

> **Startup-repopulation note.** Although these counters start at zero, they are
> wired in *before* recovery, so the work the engine does on the way up bumps some
> of them before any user request arrives — they are **regenerated, not persisted**.
> In particular WAL replay does one `lsm.get` per replayed entry (bumping
> `lookups`), and startup vector-index reconciliation re-enqueues missing docs via
> the normal write path (bumping `puts` / `wal_fsyncs` / `wal_bytes_appended`). So
> a non-zero reading right after a restart is expected and reflects startup
> activity, not pre-restart totals — `lookups` in particular jumps with the size of
> the WAL replay. Vector-search corruption counters
> ([`/admin/indices/vector/corruption-metrics`](#get-adminindicesvectorcorruption-metrics))
> follow the same in-memory, reset-on-restart model.

#### Storage metrics — field reference

All fields below are **recomputed from on-disk state on every request** (survives
restart = **Yes**), except the explicitly-flagged in-memory ones.

**`GET /admin/storage/stats`** — engine-wide value-log aggregate:

| Field | Meaning |
|-------|---------|
| `head`, `tail` | Value-log logical start/end offsets |
| `garbage_bytes` | Reclaimable dead bytes across the value log |
| `waste_ratio_pct` | Dead / total written (%) |
| `free_space_ratio_pct` | Free fraction of the allocated region (%) |
| `total_gc_runs` | Value-log GC passes ever run (persisted in metadata) |
| `total_bytes_reclaimed` | Bytes ever reclaimed by value-log GC (persisted) |
| `live_bytes` | Live (non-garbage) bytes |

**`GET /admin/storage/wal`** — WAL metadata:

| Field | Meaning |
|-------|---------|
| `head`, `tail` | WAL byte offsets of the live window |
| `total_entries` | Entries currently tracked |
| `persisted_entries` | Entries already applied + persisted |
| `pending_entries` | `total − persisted` (replayed on next open) |
| `total_gc_runs`, `total_bytes_reclaimed` | WAL GC activity (persisted) |
| `base_segment_id` | Absolute id the per-segment counters start at (lower segments trimmed) |
| `live_segments` | Tracked segments still carrying entries |
| `last_sequence` | Highest write sequence the WAL has observed |
| `segments[]` | Per-segment `{segment_id, total_entries, persisted_entries, pending_entries}` |

**`GET /admin/storage/lsm`** — per-namespace LSM manifest:

| Field | Meaning | Survives restart? |
|-------|---------|-------------------|
| `manifest_version`, `created_at_ms` | Manifest version + creation time | **Yes** |
| `level_count`, `total_entries`, `total_size_bytes` | Levels, key count, on-disk SSTable bytes | **Yes** |
| `levels[].buckets[].files[]` | Per-file `{path, created_at_ms, entry_count, size_bytes}` | **Yes** |
| `in_memory.memtable_entries` | Live active-memtable entry count | **No** |
| `in_memory.read_only_entries` / `read_only_count` | Sealed (read-only) memtable entries / count | **No** |
| `in_memory.compaction_in_progress` | Whether a compaction is running now | **No** |

**`GET /admin/storage/value-log`** — per-namespace, per-shard utilisation:

| Field | Meaning |
|-------|---------|
| `total_live_bytes`, `total_garbage_bytes`, `waste_ratio_pct`, `total_physical_bytes` | Namespace rollups across shards |
| `shards[].bucket`, `head`, `tail` | Shard id and offsets |
| `shards[].live_bytes`, `garbage_bytes`, `waste_ratio_pct` | Per-shard utilisation |
| `shards[].total_gc_runs`, `total_bytes_reclaimed` | Per-shard GC activity (persisted) |
| `shards[].physical_bytes` | Blocks actually allocated on disk (`st_blocks`; excludes sparse holes) |
| `shards[].logical_bytes` | File length including sparse holes (≥ `physical_bytes`) |

**`GET /admin/storage/value-log/{ns}/pages`** — per-page garbage breakdown:

| Field | Meaning |
|-------|---------|
| `shards[].pages[].page_offset` | Page start offset within the shard |
| `live_bytes`, `garbage_bytes`, `garbage_ratio_pct` | Per-page live/dead bytes and ratio |
| `total_records`, `garbage_records` | Records on the page, and how many are garbage |

**`GET /admin/storage/index-waste`** — field-index dead space:

> **Derived, not a stored counter.** The waste ratios are **not** persisted
> figures — they are computed on demand from the field-index blob store
> (`waste_ratio = (logical_bytes − live_bytes) / logical_bytes`, where both inputs
> come from the on-disk `blobs.keys` header + slot table). Because those files are
> persisted, a restart reproduces the **identical** value (it "survives restart"),
> but unlike the ops-metrics counters there is no accumulated history: the ratio
> always reflects *current* dead space and reads back to ≈0 right after a
> compaction. `distinct_count` is likewise derived from the field's in-memory
> value→slot map, which is rebuilt from persisted state on open.

| Field | Meaning |
|-------|---------|
| `threshold` | Compaction threshold (fraction `0.0..1.0`) — config, not on-disk state |
| `namespaces[].fields[].bitmap_waste_ratio` | Reclaimable fraction of the bitmap blob store — *derived from on-disk state* (`null` if field still building) |
| `namespaces[].fields[].keymap_waste_ratio` | Reclaimable fraction of the keymap blob store — *derived from on-disk state* |
| `namespaces[].fields[].over_threshold` | True if either store has reached the threshold (compacted next checkpoint) |
| `namespaces[].fields[].distinct_count` | Distinct indexed values for the field — *derived from the rebuilt-on-open value map* |

The listing endpoints — `GET /admin/storage/namespaces`, `/kv-namespaces`,
`/stores/{ns}/kv-meta`, `/kv-stores/{ns}/kv-meta`, `/system/stores`, and
`/system/stores/{ns}/meta` — return registry/schema descriptors (names, `ns_id`,
key/value types, `semantic_search_enabled`, TTL config, indexed fields) plus, where
relevant, the same LSM/value-log blocks documented above. All are derived from the
persisted registry + on-disk state and so **survive restart** (their nested
`in_memory.*` LSM blocks, when present, are the only **No** values).

---

### Admin Storage API

Storage diagnostics and engine operations. Not intended for application traffic.

#### Quick reference

| Method | Path | Response | Purpose |
|--------|------|----------|---------|
| `GET` | `/admin/storage/health` | `200` | Liveness probe — uptime in seconds |
| `GET` | `/admin/storage/stats` | `200` | Engine-wide value-log statistics |
| `GET` | `/admin/storage/ops-metrics` | `200` | Engine-wide operational counters since startup (reads/writes/lookups/compaction/GC) |
| `GET` | `/admin/storage/wal` | `200` | WAL metadata snapshot |
| `GET` | `/admin/storage/lsm` | `200` | LSM manifest for every namespace |
| `GET` | `/admin/storage/value-log` | `200` | Per-namespace, per-shard value-log utilisation |
| `GET` | `/admin/storage/value-log/{ns}/pages` | `200` | Per-page garbage breakdown for one namespace |
| `GET` | `/admin/storage/namespaces` | `200` | Namespace registry (doc stores + KV stores) |
| `GET` | `/admin/storage/kv-namespaces` | `200` | All engine KV namespaces, annotated by role |
| `GET` | `/admin/storage/stores/{ns}/kv-meta` | `200` | KV-layer metrics for one doc store namespace |
| `GET` | `/admin/storage/kv-stores/{ns}/kv-meta` | `200` | KV-layer metrics for one KV store namespace |
| `GET` | `/admin/storage/system/stores` | `200` | List system-namespace KV and doc stores |
| `GET` | `/admin/storage/system/stores/{ns}/meta` | `200` | Full metadata for one system KV store |
| `GET` | `/admin/storage/index-waste` | `200` | Per-field field-index bitmap/keymap waste + compaction threshold |
| `POST` | `/admin/storage/gc` | `200` | Trigger value-log GC across all namespaces |
| `POST` | `/admin/storage/gc/wal` | `200` | Trigger WAL GC |
| `POST` | `/admin/storage/compact` | `204` | Trigger LSM compaction across all namespaces |
| `POST` | `/admin/storage/index-checkpoint` | `202` | Flush + compact field indexes (and row maps) across all namespaces — runs in background; `409` if one is already running |

---

#### `GET /admin/storage/health`

```bash
curl http://localhost:8080/admin/storage/health
# → {"status":"ok","uptime_s":3601}
```

---

#### `POST /admin/storage/gc`

Trigger value-log GC immediately across all namespaces. Returns per-namespace results.

```bash
curl -X POST http://localhost:8080/admin/storage/gc
```
```json
{
  "namespaces_collected": 3,
  "results": [
    {"namespace": "products", "bytes_reclaimed": 1048576, "bytes_live": 5242880}
  ]
}
```

---

#### `POST /admin/storage/compact`

Trigger LSM compaction across all namespaces.

```bash
curl -X POST http://localhost:8080/admin/storage/compact
# → 204 No Content
```

---

#### `GET /admin/storage/index-waste`

Report the reclaimable dead space in each field index's two append-only stores, alongside the compaction `threshold` (a fraction). The **bitmap** store grows with per-document churn; the **keymap** store grows under distinct-value churn. `over_threshold` is `true` when either store has reached the threshold and will be compacted at the next index checkpoint. Use this to decide whether to force a `POST /admin/storage/index-checkpoint`. Fields still building report `null` waste.

This is the fleet-wide view of waste *ratios*; for the absolute on-disk byte *growth* of a single field (logical vs. live bytes) — which a ratio alone hides — use [`GET /admin/indices/{ns}/{field}/blob-stats`](#get-adminindicesnsfieldblob-stats).

```bash
curl http://localhost:8080/admin/storage/index-waste
```
```json
{
  "threshold": 0.5,
  "namespaces": [
    {
      "namespace": "users",
      "ns_id": 3,
      "fields": [
        {
          "field_id": 0,
          "field_name": "status",
          "field_type": "Str",
          "bitmap_waste_ratio": 0.12,
          "keymap_waste_ratio": 0.0,
          "over_threshold": false,
          "distinct_count": 5
        }
      ]
    }
  ]
}
```

---

#### `POST /admin/storage/index-checkpoint`

Force an index checkpoint immediately. This runs the **same pass** as the periodic index-checkpoint worker (default every 15 min) and clean shutdown: it flushes each namespace's dense row map and all active field indexes to disk, and compacts any field-index bitmap store whose waste exceeds `thresholds.index_blob_waste_threshold`. Use it to reclaim field-index dead space on demand rather than waiting for the next tick.

This is the **only** way to trigger field-index compaction on demand — `/admin/storage/compact` is LSM/value-log compaction, a separate subsystem.

Because the flush + compaction can take a long time on a large/wasted index, this **returns `202 Accepted` immediately and runs the pass in the background**; the checkpointed-field count is written to the server log on completion (and any failure is logged there too). If a checkpoint is already running, the request is rejected with `409 Conflict` so passes cannot stack.

```bash
curl -i -X POST http://localhost:8080/admin/storage/index-checkpoint
```
```text
HTTP/1.1 202 Accepted
```
```json
// 409 Conflict when one is already running
{ "error": "an index checkpoint is already running" }
```

---

### Admin Indices API

Index monitoring and bulk operations. All write operations that touch index data return `202 Accepted` and run in the background. Only one attribute-index operation and one vector-index operation may be active per namespace at a time.

#### Quick reference

| Method | Path | Response | Purpose |
|--------|------|----------|---------|
| `GET` | `/admin/indices/progress` | `200` | All active index builds across every namespace |
| `GET` | `/admin/indices/vector/queue/summary` | `200` | Global queue depth / lag by namespace |
| `GET` | `/admin/indices/vector/queue/retried` | `200` | All entries with `retry_count > 0` (global) |
| `GET` | `/admin/indices/vector/corruption-metrics` | `200` | Per-namespace counts of vector entries skipped during search due to corrupt bytes (all namespaces) |
| `GET` | `/admin/indices/{ns}/vector/corruption-metrics` | `200` | Corrupt-skip counts for a single namespace |
| `POST` | `/admin/indices/vector/reconcile` | `202` | Background validating reconcile: re-enqueue docs missing **or with corrupt** vectors (all namespaces); `409` if already running |
| `GET` | `/admin/indices/{ns}/progress` | `200` | Index progress for one namespace |
| `POST` | `/admin/indices/{ns}/attribute/reindex-all` | `202` | Drop + rebuild all field indices |
| `DELETE` | `/admin/indices/{ns}/attribute/drop-all` | `202` | Drop all field indices (no rebuild) |
| `POST` | `/admin/indices/{ns}/attribute/{field}/reindex/{doc_id}` | `200` | Reindex one document in one field index (doc stores) |
| `GET` | `/admin/indices/{ns}/{field}/blob-stats` | `200` | One field index's on-disk blob growth/waste (`404` if not active) |
| `POST` | `/admin/indices/{ns}/vector/reindex-all` | `202` | Re-enqueue all docs for embedding |
| `POST` | `/admin/indices/{ns}/vector/reindex/{doc_id}` | `200` | Re-enqueue one document for embedding (doc + KV stores) |
| `POST` | `/admin/indices/{ns}/vector/reindex-failed` | `200` | Reset exhausted queue entries |
| `DELETE` | `/admin/indices/{ns}/vector/drop-all` | `202` | Clear all vector index data |
| `GET` | `/admin/indices/{ns}/vector/queue` | `200` | All queue entries for one namespace |
| `GET` | `/admin/indices/{ns}/vector/queue/retried` | `200` | Retried entries for one namespace |
| `GET` | `/admin/indices/{ns}/vector/queue/{doc_id}` | `200` | Look up one queue entry |
| `DELETE` | `/admin/indices/{ns}/vector/queue/{doc_id}` | `204` | Remove one queue entry |
| `POST` | `/admin/indices/{ns}/vector/queue/{doc_id}/retry` | `200` | Reset retry count for one exhausted entry |

---

#### `GET /admin/indices/progress`

All active or recently completed index builds across every namespace, grouped into field (attribute) builds and vector-index progress.

```bash
curl http://localhost:8080/admin/indices/progress
```
```json
{
  "attribute_builds": [
    {
      "kind": "Attribute",
      "id": {"Field": {"namespace": "users", "field": "country"}},
      "status": "Running",
      "total": 50000,
      "indexed": 12340,
      "failed": 0,
      "started_at_ms": 1746789000000,
      "updated_at_ms": 1746789010000,
      "completed_at_ms": null,
      "last_error": null
    }
  ],
  "vector_progress": [
    {
      "namespace": "products",
      "indexed_approx": 970,
      "pending": 30,
      "exhausted": 0,
      "progress_pct": 97.0
    }
  ]
}
```

`progress_pct = indexed_approx / (indexed_approx + pending) * 100`. Exhausted entries are excluded from the denominator.

Use `GET /admin/indices/{ns}/progress` for the same view scoped to one namespace.

---

#### `GET /admin/indices/vector/corruption-metrics`

Counts of vector-index entries that search **skipped because their bytes failed to deserialize** (corruption or a write-path bug), **broken down per namespace**. In-memory and monotonically increasing since startup (reset on restart) — sample twice to compute a rate, or alert on a non-zero/rising value. Split by pass: `sparse_corrupt_skipped` (Pass 1), `dense_corrupt_skipped` (Pass 2), plus their `total`. A namespace that has never recorded a corruption is omitted; an empty object means none have.

A rising value means stored vectors are corrupt and queries are silently degraded; run the validating [`POST /admin/indices/vector/reconcile`](#post-adminindicesvectorreconcile) to re-embed the affected documents.

```bash
curl http://localhost:8080/admin/indices/vector/corruption-metrics
```
```json
{
  "products": { "sparse_corrupt_skipped": 2, "dense_corrupt_skipped": 0, "total_corrupt_skipped": 2 }
}
```

---

#### `GET /admin/indices/{ns}/vector/corruption-metrics`

The same corrupt-skip counts for a single namespace `{ns}`. Returns all-zero if the namespace has never recorded a corruption.

```bash
curl http://localhost:8080/admin/indices/products/vector/corruption-metrics
```
```json
{ "sparse_corrupt_skipped": 2, "dense_corrupt_skipped": 0, "total_corrupt_skipped": 2 }
```

---

#### `POST /admin/indices/vector/reconcile`

**Validating** vector-index reconciliation across **every** semantic-search-enabled namespace. Re-enqueues any document whose committed vector index is missing a half (the sparse-meta or dense entry) **or is present but corrupt** — i.e. the stored bytes fail to deserialize — as well as documents left un-enqueued by the write-then-enqueue crash window. Unlike the cheap presence-only pass that runs at startup, this **deserializes every entry**, so it cannot use the count short-circuit and performs a full value-reading scan.

Because that scan can take a long time on a large corpus, the endpoint **returns `202 Accepted` immediately and runs the pass in the background.** The re-enqueued count and any per-namespace failures are written to the server log (`info!` on completion, `warn!`/`error!` on failure); the async worker then embeds and indexes the re-enqueued documents. Overlapping runs are rejected with **`409 Conflict`** so the expensive scan cannot stack. A presence-only reconcile still runs automatically on startup.

```bash
curl -i -X POST http://localhost:8080/admin/indices/vector/reconcile
# HTTP/1.1 202 Accepted    (or 409 if one is already running)
```

Watch the server log for `vector reconcile (validating) complete` and the re-enqueued count. To force a full rebuild of a single namespace regardless of current index state, use [`POST /admin/indices/{ns}/vector/reindex-all`](#post-adminindicesnsvectorreindex-all) instead.

---

#### `GET /admin/indices/vector/queue/summary`

Queue depth and lag, grouped by namespace. Each entry is classified as *actionable* (worker will process it), *retrying* (failed at least once, still within retry budget), or *exhausted* (budget spent — needs manual reset or removal).

```bash
curl http://localhost:8080/admin/indices/vector/queue/summary
```
```json
{
  "max_retries_configured": 5,
  "total_pending": 49,
  "total_actionable": 47,
  "total_retrying": 5,
  "total_exhausted": 2,
  "by_namespace": [
    {"namespace": "products", "pending": 30, "actionable": 30, "retrying": 1, "exhausted": 0}
  ]
}
```

---

#### `GET /admin/indices/vector/queue/retried`

All queue entries with `retry_count > 0` across every namespace, paginated.

```bash
curl "http://localhost:8080/admin/indices/vector/queue/retried?page_no=1&page_size=20"
```
```json
{
  "total": 2,
  "page_no": 1,
  "page_size": 20,
  "entries": [
    {
      "namespace": "products",
      "doc_id_hex": "550e8400e29b41d4",
      "doc_id_str": null,
      "retry_count": 3,
      "last_error": "embedding service timeout",
      "text_preview": "ultra-light trail runner with waterproof membrane…"
    }
  ]
}
```

`doc_id_str` is set when the doc-id bytes are printable ASCII; `null` otherwise. `text_preview` is capped at 120 characters.

---

#### `POST /admin/indices/{ns}/attribute/reindex-all`

Drop every field index for `{ns}` and rebuild them all from scratch. Returns `202 Accepted`; progress is visible via `GET /admin/indices/{ns}/progress`.

```bash
curl -X POST http://localhost:8080/admin/indices/users/attribute/reindex-all
# → 202 Accepted
```

Returns `409` when an attribute operation is already active for `{ns}`. Returns `422` when the namespace has no indices to rebuild.

---

#### `DELETE /admin/indices/{ns}/attribute/drop-all`

Drop every field index for `{ns}` without rebuilding. Returns `202 Accepted`.

```bash
curl -X DELETE http://localhost:8080/admin/indices/users/attribute/drop-all
# → 202 Accepted
```

Returns `409` when an attribute operation is already active. Returns `422` when the namespace has no indices.

---

#### `POST /admin/indices/{ns}/attribute/{field}/reindex/{doc_id}`

Reindex a **single document's** entry in **one** field index. The field value is re-derived from the document's *current* stored bytes using the same logic as the write path (clear the row's old buckets, re-extract via the field's extractor, insert), so it repairs a single stale or missing index entry without rewriting the document. Only the named field is touched — no other field index is rebuilt and no vector re-embedding is triggered. The operation is O(1) and runs synchronously, returning `200`.

Document stores only — field indices do not exist on KV stores. `{doc_id}` is parsed in the namespace's key format (the same format as `GET /stores/{ns}/docs/{id}`).

```bash
curl -X POST http://localhost:8080/admin/indices/users/status/reindex/42
```
```json
{ "status": "reindexed", "namespace": "users", "field": "status", "doc_id": "42" }
```

Returns `404` when the namespace is unknown, `{field}` is not an indexed field of it, or no document with `{doc_id}` exists. Returns `409` when the field index is registered but not yet active (still building). Returns `400` when `{doc_id}` is not valid for the namespace's key type.

---

#### `GET /admin/indices/{ns}/{field}/blob-stats`

On-disk blob growth for a **single** field index. Each field keeps two append-only stores — the **bitmap** store (one blob per distinct value, re-appended whole on every document write) and the **keymap** store (slot → value) — and this reports, per store, the **logical** bytes (everything ever appended = live + stale) versus the **live** bytes (what survives compaction), their waste ratios, and the field's `distinct_values` count. `over_threshold` is `true` when either store has reached the compaction `waste_threshold`.

Unlike [`GET /admin/storage/index-waste`](#get-adminstorageindex-waste), which reports only waste *ratios* across all fields, this surfaces the absolute blob *growth* between compactions that a ratio hides — the failure mode of low-cardinality, high-churn fields (e.g. a boolean over many documents), whose bitmap blob can balloon to many times its live footprint. A large, high-waste `bitmap_logical_bytes` is the signal to force a [`POST /admin/storage/index-checkpoint`](#post-adminstorageindex-checkpoint). The same condition is logged as a checkpoint warning when a field's bitmap blob exceeds 64 MiB logical with ≥50% waste.

```bash
curl http://localhost:8080/admin/indices/users/status/blob-stats
```
```json
{
  "namespace": "users",
  "field": "status",
  "waste_threshold": 0.5,
  "over_threshold": true,
  "distinct_values": 2,
  "bitmap_logical_bytes": 104857600,
  "bitmap_live_bytes": 8192,
  "bitmap_waste_ratio": 0.99,
  "keymap_logical_bytes": 256,
  "keymap_live_bytes": 256,
  "keymap_waste_ratio": 0.0
}
```

Returns `404` when `{field}` has no active index in `{ns}` (unknown field, or still building).

---

#### `POST /admin/indices/{ns}/vector/reindex-all`

Re-enqueue every document in `{ns}` for embedding (equivalent to a fresh full index build). Returns `202 Accepted`.

```bash
curl -X POST http://localhost:8080/admin/indices/products/vector/reindex-all
# → 202 Accepted
```

Returns `409` when a campaign or cleanup is already running. Returns `422` when semantic search is not enabled.

---

#### `POST /admin/indices/{ns}/vector/reindex/{doc_id}`

Re-enqueue a **single** document for vector (re-)embedding — the same enqueue the write path and `vector/reindex-all` use, scoped to one document. The async worker embeds and indexes it on its next pass, so this returns `200` immediately (not `202`: the enqueue itself is synchronous and durable, only the embedding is deferred).

Works for both document stores and semantic-search KV stores (the namespace kind is detected automatically); for a KV store `{doc_id}` is the key string, otherwise it is parsed in the namespace's key format. The JSON `status` is:

- `enqueued` — the document was queued for embedding.
- `not_found` — no document with that id exists.
- `skipped_empty_text` — the document produced no embedding text, so nothing was queued.

```bash
curl -X POST http://localhost:8080/admin/indices/products/vector/reindex/sku-123
```
```json
{ "status": "enqueued", "namespace": "products", "doc_id": "sku-123" }
```

Returns `422` when the namespace is not semantic-search-enabled, `404` when it does not exist, and `400` when `{doc_id}` is not valid for the namespace's key type.

---

#### `POST /admin/indices/{ns}/vector/reindex-failed`

Reset `retry_count` to zero for every exhausted queue entry in `{ns}`, making them actionable again. Returns `200` with `{ "retried": N }`.

```bash
curl -X POST http://localhost:8080/admin/indices/products/vector/reindex-failed
# → {"retried": 3}
```

---

#### `DELETE /admin/indices/{ns}/vector/drop-all`

Disable semantic search and clear all vector index data: queue entries and the `{ns}_sparse_vector`, `{ns}_dense_vector`, and `{ns}_sparse_vector_meta` companion namespaces. The schema is updated synchronously before the background cleanup runs. Returns `202 Accepted`.

```bash
curl -X DELETE http://localhost:8080/admin/indices/products/vector/drop-all
# → 202 Accepted
```

Returns `409` when a cleanup or campaign is already running. Returns `422` when semantic search is not enabled.

---

#### `GET /admin/indices/{ns}/vector/queue`

All pending queue entries for one namespace, paginated.

```bash
curl "http://localhost:8080/admin/indices/products/vector/queue?page_no=1&page_size=20"
```

Returns the same `{total, page_no, page_size, entries}` shape as `/admin/indices/vector/queue/retried`.

---

#### `GET /admin/indices/{ns}/vector/queue/{doc_id_hex}`

Look up one queue entry by its hex-encoded doc-id.

```bash
curl http://localhost:8080/admin/indices/products/vector/queue/550e8400e29b41d4
```

Returns `400` for invalid hex. Returns `404` if not found.

---

#### `DELETE /admin/indices/{ns}/vector/queue/{doc_id_hex}`

Remove one queue entry manually. Use for entries that are exhausted and should not be retried.

```bash
curl -X DELETE \
  http://localhost:8080/admin/indices/products/vector/queue/550e8400e29b41d4
# → 204 No Content
```

Returns `400` for invalid hex. Returns `404` if not found.

---

#### `POST /admin/indices/{ns}/vector/queue/{doc_id_hex}/retry`

Reset the retry count for one exhausted entry, making it actionable again.

```bash
curl -X POST \
  http://localhost:8080/admin/indices/products/vector/queue/550e8400e29b41d4/retry
```

Returns `422` if the entry has not yet exhausted its retry budget.

---

## Predicate syntax

Predicates reference indexed field names. Operators and examples:

| Operator   | Example                          | Applicable types   |
|-----------|----------------------------------|--------------------|
| `=`        | `status = "active"`              | str, int, bool     |
| `!=`       | `status != "deleted"`            | str, int, bool     |
| `<`        | `age < 18`                       | int                |
| `<=`       | `age <= 65`                      | int                |
| `>`        | `amount_usd > 100`               | int                |
| `>=`       | `age >= 18`                      | int                |
| `AND`      | `status = "active" AND age >= 18` | —                 |
| `OR`       | `status = "active" OR status = "trial"` | —          |
| `NOT`      | `NOT paid = false`               | —                  |

String values must be quoted with `"`. Boolean values are `true` or `false` (unquoted).

**Examples:**

```
# Single condition
status = "active"

# Compound condition
status = "active" AND age >= 18 AND verified = true

# OR condition
country = "US" OR country = "CA"

# Negation
NOT status = "deleted"
```

---

## Error responses

All errors return JSON with an `"error"` key:

```json
{"error": "doc store 'orders' not found"}
```

| HTTP status | When                                                     |
|-------------|----------------------------------------------------------|
| `400`       | Invalid ID / key format, bad schema, type mismatch, malformed request |
| `404`       | Namespace, document, or KV key not found                 |
| `409`       | Store already exists, field already indexed, build in progress, attribute/vector operation already active for namespace |
| `422`       | Semantic search requested but not enabled / cluster index not loaded; no indices to rebuild; entry not yet exhausted |
| `500`       | Internal database error                                  |

---

## Bulk loading data

Use the `bulk_load` tool from the `tools` crate (binary `minnal_tools`) to bulk-load a [JSONL](https://jsonlines.org/) file (one JSON object per line) into a store. The loader streams each line to the running server over the REST API, so **the server must already be up**. By default the namespace must already exist; pass `--schema <schema.json>` to import the schema first.

### Build

```bash
cargo build --release -p tools
# binary: target/release/minnal_tools
```

### Usage

```
minnal_tools bulk_load [--no-wal] [--schema <schema.json>] <url> <namespace> <id_field> <data.jsonl>
```

| Argument     | Description                                                  |
|-------------|--------------------------------------------------------------|
| `url`        | Base URL of the running doc store REST API (e.g. `http://localhost:8080`) |
| `namespace`  | Name of the target doc store                                |
| `id_field`   | JSON field name whose value becomes the document ID          |
| `data.jsonl` | Path to the JSONL file                                       |

| Flag                | Description                                                  |
|---------------------|-------------------------------------------------------------|
| `--schema <file>`   | Import the schema (`POST /admin/stores/import`) before loading. An existing store is reused, so re-runs are safe. The schema's `namespace` must match the `namespace` argument. Without this flag the namespace must already exist. |
| `--no-wal`          | Append `?skip_wal=true` to each write for maximum throughput. Data written this way is **unrecoverable on a crash** — only use when re-running the load is acceptable (e.g. an initial import from a source of truth). |

The tool first calls `GET /stores` to confirm the namespace exists and resolve its `key_type` (after importing the schema if `--schema` was given), then `PUT`s each document. The `id_field` value is parsed according to the store's `key_type` — UUID string for `uuid`, integer (number or numeric string) for `u64`/`u128`. The id field is **not** removed from the stored document.

### Example

Given `users.jsonl`:

```jsonl
{"id": "550e8400-e29b-41d4-a716-446655440001", "name": "Alice", "status": "active", "age": 30}
{"id": "550e8400-e29b-41d4-a716-446655440002", "name": "Bob",   "status": "inactive", "age": 25}
```

Load (with the server running at `http://localhost:8080`):

```bash
./target/release/minnal_tools bulk_load http://localhost:8080 users id users.jsonl
```

Output:

```
namespace 'users' found  key_type=Uuid
id_field='id'
  1000 documents loaded…
done  loaded=2  skipped=0  total=2  elapsed=0.05s
```

Lines with a missing or unparseable `id_field`, invalid JSON, or a rejected `PUT` are skipped with a warning and counted in `skipped`; valid lines continue to be processed. When there are any failures, the offending lines and their reasons are written to a sibling `<data>.errors` file.

---

## On-disk layout

```
{db_path}/
  ns_{namespace}/             ← one directory per namespace (doc store, KV store, or vector companion)
    wal/                      ← write-ahead log segments (64 MiB each)
    lsm/                      ← LSM tree (sorted key files)
    value_log/                ← value blobs (sharded)
  ns_{namespace}_sparse_vector/      ← companion store: 1-bit sliding-window chunk embeddings, keyed by [cluster_id ‖ doc_id] (pass-1 ANN)
  ns_{namespace}_dense_vector/       ← companion store: multi-bit whole-doc embeddings, keyed by doc_id (pass-2 re-rank)
  ns_{namespace}_sparse_vector_meta/ ← companion store: per-doc cluster membership, used only by delete/upsert cleanup
  index/
    {ns_id}/
      {field_id}/             ← doc stores only — KV stores have no field indices
        blobs.keys            ← RoaringBitmap key file (mmap hash table)
        blobs.vals            ← RoaringBitmap blob data
        keymap.idx            ← field-value → slot mapping
        checkpoint            ← WAL offset at last index flush
        build_progress.json   ← index rebuild progress (created on add_index)

{schema_dir}/
  {namespace}.json            ← schema for each store (doc or KV — distinguished by mandatory store_type field)
```

Schema files are written atomically (tmp-then-rename). Every schema declares a mandatory `store_type` field — `"doc"` or `"kv"` — which is the authoritative on-disk discriminant between the two store kinds (doc schemas additionally use `key_type` values `"uuid"`/`"u64"`/`"u128"`, KV schemas `"str"`/`"int"`, but that is no longer what distinguishes them). Both types are stored in the same directory and are mutually exclusive — you cannot create a doc store and a KV store with the same namespace name.
