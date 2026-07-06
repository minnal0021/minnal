# minnal

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this software except in compliance with the License. You may obtain a copy of the License at:

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

---

**minnal** (மின்னல்) means *lightning* in Tamil.

Minnal is a layered document database written in Rust. It combines a high-performance LSM + value-log key-value engine with a JSON document layer, RoaringBitmap field indexing, and quantised approximate nearest-neighbour semantic search.

It works in **two ways**, from the same engine:

- **Embedded** — link `minnal_db` (and the upper layers) directly into your Rust application and call the API in-process, with no server or network hop. See [Embedded Use](#embedded-use-minnal_db-as-a-library).
- **Server (REST)** — run `minnal_doc_store_api` to expose the same stores over an HTTP REST API for any client or language. See [Start the Server](#start-the-server).

> **Platform support:** Linux and macOS only. Windows is not supported — the storage engine relies on Unix positional I/O (`pread`/`pwrite`) and the server requires POSIX signals.

> **Companion UI:** [minnal0021/minnal_ui](https://github.com/minnal0021/minnal_ui) is a lightweight web UI that can be used to drive the minnal doc store.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [Layer 1 — minnal\_db (Key-Value Engine)](#layer-1--minnal_db-key-value-engine)
  - [Layer 2 — index (RoaringBitmap Field Indexing)](#layer-2--index-roaringbitmap-field-indexing)
  - [Layer 3 — semantic\_search (Vector Quantisation + ANN)](#layer-3--semantic_search-vector-quantisation--ann)
  - [Layer 4 — minnal\_doc\_store (Document Store + KV Store)](#layer-4--minnal_doc_store-document-store--kv-store)
  - [Layer 5 — minnal\_doc\_store\_api (REST API)](#layer-5--minnal_doc_store_api-rest-api)
- [Salient Features](#salient-features)
- [Embedded Use (minnal\_db as a Library)](#embedded-use-minnal_db-as-a-library)
  - [Field-Level Indexing](#field-level-indexing)
- [How to Use](#how-to-use)
  - [Build](#build)
  - [Start the Server](#start-the-server)
  - [Quickstart: create a document store and bulk-load it](#quickstart-create-a-document-store-and-bulk-load-it)
  - [Configuration](#configuration)
  - [Store Lifecycle](#store-lifecycle)
  - [Document CRUD](#document-crud)
  - [Predicate Queries](#predicate-queries)
  - [Semantic Search](#semantic-search)
  - [Admin and Monitoring](#admin-and-monitoring)
  - [Index Management](#index-management)
  - [KV Store](#kv-store)
  - [Bulk Loading](#bulk-loading)
  - [Logging](#logging)
  - [Write Durability and Recovery](#write-durability-and-recovery)
- [Scripts and Config](#scripts-and-config)
  - [Server scripts](#server-scripts)
  - [Example scripts](#example-scripts)
  - [Cluster centroids](#cluster-centroids)
  - [Sample config](#sample-config)
- [Crate Structure](#crate-structure)
- [Acknowledgements](#acknowledgements)

---

## Overview

Minnal is designed for use cases that need fast key-value storage, structured predicate queries on JSON fields, *and* semantic (embedding-based) similarity search — in a single embedded process. The only piece that lives outside the process is the embedding service, and only when semantic search is enabled — it generates the vector embeddings for documents (at index time) and for queries (at search time) that semantic search relies on. Everything else — quantisation, indexing, and the ANN search itself — runs in-process.

A typical deployment looks like this:

```
REST client
    │
    ▼
minnal_doc_store_api   ← Axum HTTP server
    │
    ├── minnal_doc_store   ← JSON schema, doc lifecycle, index builders
    │       ├── minnal_db  ← LSM + value-log KV engine (WiscKey-style)
    │       ├── index      ← RoaringBitmap field indices, predicate evaluator
    │       └── semantic_search  ← IVF clustering, RaBitQ quantisation, ANN search
    │
    └── embedding service  ← external HTTP service (e.g. E5, Instructor)
```

The diagram shows the **server** deployment. For **embedded** use, drop the top `minnal_doc_store_api` box and call `minnal_doc_store` (or `minnal_db` directly) in-process — everything below the REST boundary is the same code either way.

Each namespace (logical store) maintains its own KV storage, field indices, and a companion vector store for quantised embeddings.

---

## Architecture

Minnal is built as five layers, each consuming the one below it. The bottom layer is a raw key-value engine; every layer above adds a capability — structured indexing, vector search, a document model, and finally an HTTP interface — without the lower layers needing to know about it. Reading the layers in order is the fastest way to understand how a single REST request ends up touching the LSM tree at the bottom.

### Layer 1 — minnal\_db (Key-Value Engine)

At the foundation sits `minnal_db`, an embedded, multi-namespace key-value store built on the **WiscKey** key-value separation principle: keys (with value-log pointers) live in a small LSM tree, while values live in a separate append-only sharded value log. This dramatically reduces compaction write-amplification — the LSM tree stays small and cheap to rewrite, and large values are only touched during dedicated GC passes.

Key design choices:

| Property | Detail |
|---|---|
| Key storage | Single skip-list memtable → sharded L0/L1 SSTables (per-bucket LSM) |
| Value storage | Sharded append-only value log (configurable page size, 64 MiB default) |
| Durability | Write-ahead log (WAL) with configurable fsync cadence |
| Concurrency | `parking_lot` RwLock per bucket; epoch-based reclamation (`crossbeam-epoch`) |
| Serialization | Zero-copy `rkyv` for typed values; `crc32fast` checksums on every record |
| TTL | Native per-record expiry tracked in the value log |
| Namespaces | Logical isolation within a single DB; each namespace has its own LSM shards and value-log shards |
| Background workers | LSM compaction, value-log GC, WAL cleanup, TTL eviction — each on its own tokio task |
| Hashing | SIMD-accelerated Murmur3 (`mm3h`) for bucket assignment |
| Prefix scan | SIMD-accelerated prefix scan across all layers — see below |

The WAL is written before every mutation. On startup the engine replays any WAL segments that postdate the last LSM flush, ensuring no committed write is lost after a crash.

#### Prefix scan with SIMD acceleration

`scan_prefix` returns all live key-value pairs whose keys start with a given byte prefix. It merges results across all storage layers — active memtable, read-only memtable, L0 SSTables, and the L1 SSTable — deduplicating by key and honouring tombstones. The full API is available on `Db`, `Namespace`, `AsyncDb`, and `AsyncNamespace`, including a zero-copy typed variant (`scan_prefix_typed<K, V>`) that deserialises keys and values with `rkyv`.

Three layers of acceleration keep prefix scans fast even as data accumulates across layers:

**1 — Stored `u64` key prefix for O(1) integer comparison**

Every skip-list node and every SSTable record stores the first 8 bytes of its key as a big-endian `u64` field (`key_prefix`). The same field drives bucket assignment (via a Murmur3 hash of the `u64`). For prefixes up to 8 bytes, matching reduces to a single integer comparison against the stored `u64` — no byte-by-byte string walk needed. Only prefixes longer than 8 bytes fall back to `starts_with`.

**2 — File-level bounds pruning**

Each SSTable file records the `min_key_prefix` and `max_key_prefix` of all entries it contains (both stored as `u64`). Before scanning a file, the engine checks whether the search prefix can possibly fall within those bounds. Files whose prefix range excludes the target are skipped entirely, with no I/O.

**3 — AVX2/AVX512 key comparison in skip-list traversal**

When two keys share the same `u64` prefix (i.e. the first 8 bytes are identical), the skip list calls `compare_bytes_simd` to resolve the full lexicographic order. The implementation selects the widest available SIMD path at compile time:

- **AVX512** — 64 bytes per iteration using `_mm512_loadu_epi8` / `_mm512_cmpeq_epi8_mask`. The equality mask is a 64-bit integer; `trailing_ones()` finds the first differing byte without a branch per byte.
- **AVX2** — 32 bytes per iteration using `_mm256_cmpeq_epi8` / `_mm256_movemask_epi8`. The 32-bit equality mask is inspected with `trailing_ones()` in the same way.
- **Scalar fallback** — standard byte-by-byte comparison on targets without AVX2.

The SIMD paths are also used for all ordered key lookups and range scans in the skip list, not just prefix scans, so the benefit extends across all read paths.

### Layer 2 — index (RoaringBitmap Field Indexing)

The KV engine can find a document by its primary key, but answering a question like "which documents have `status = active` and `age >= 18`?" needs a secondary index. That is what the stand-alone `index` crate provides: fast predicate queries over document fields.

Each indexed field maintains a `FieldIndex` — a persistent hash table that maps field values to `RoaringBitmap` sets of document row IDs. The bitmaps live in memory-mapped files (`memmap2`), so the OS page cache handles warm and cold access naturally, and the bitwise `AND`/`OR`/`NOT` operations that combine them are SIMD-accelerated. A query string such as `status = "active" AND age >= 18` is turned into an AST by a small lexer and parser, then evaluated against the live indices. Three value types are supported — `str`, `int`, and `bool`.

Index writes are checkpointed against WAL offsets, so a crash partway through a rebuild resumes exactly where it left off rather than starting over.

For the full design — how field attributes are created, updated, stored, and recovered after a crash — see [`index/Index-Architecture.md`](index/Index-Architecture.md).

### Layer 3 — semantic\_search (Vector Quantisation + ANN)

Field indices answer *exact* questions. Semantic search answers *fuzzy* ones — "find the records that mean something similar to this query" — by comparing embedding vectors rather than literal values. It applies to any text-valued store: the indexed embedding fields of a JSON document store, or the string values of a `value_type = str` KV store (both expose a `semantic-search` endpoint). The `semantic_search` crate implements this as IVF (Inverted File Index) approximate nearest-neighbour search with **RaBitQ** quantisation and a **two-pass** sparse→dense search algorithm. The subsections below build it up from how the space is partitioned, through how values are quantised at write time, to how a query is resolved.

#### IVF Clustering

The embedding space is partitioned into a fixed number of clusters. The cluster centroids are pre-computed offline (e.g. with k-means over a representative corpus) and stored as a JSONL file (one `{cluster_id, centroid}` per line). At startup, minnal loads the centroids once into a read-only index shared across all requests.

This partitioning is what makes Pass 1 cheap: every document's sparse chunks are stored keyed by their assigned `cluster_id`, so a query only has to scan the chunks in a handful of relevant clusters instead of the whole corpus. At query time the relevant clusters are chosen by an **exact** nearest-centroid scan — each query embedding's distance to every centroid is computed and the `n_probes` nearest are selected (introselect, ~O(C) where C is the cluster count). There is deliberately **no precomputed neighbour graph over the clusters**: at the cluster counts in use the exhaustive scan takes microseconds and yields the *exact* nearest clusters, whereas an approximate graph traversal would trade recall for no meaningful speed-up. (A neighbour graph would only pay off at much larger, sparser cluster counts.)

#### Dual Quantisation at Index Time

Each document write triggers a **single embedding service call** that embeds one ordered payload list — `[whole_text, chunk₀, chunk₁, …]` — in one round trip (one GPU batch of `N+1`), then splits the order-preserving response by position into two complementary sets of quantised entries:

1. **MultiBit (dense)** — one embedding for the entire document, quantised to `number_of_bits_for_dense_quantisation` bits per dimension. Stored in a `{ns}_dense_vector` companion namespace keyed directly by `doc_id`. At 8 bits, a 768-dimensional embedding compresses from ~3 KB (f32) to ~768 bytes — a 4× reduction with high recall fidelity.

2. **SingleBit (sparse)** — the document text is split into overlapping chunks using a sliding window (`window_size`, `sliding_size`). Each chunk is independently embedded and quantised to 1 bit per dimension. Stored in a `{ns}_sparse_vector` namespace under composite keys `[cluster_id (4 bytes) || doc_id]`. At 1 bit, each chunk entry is ~96 bytes — a 32× reduction, enabling fast cluster prefix scans over large corpora.

A third namespace, `{ns}_sparse_vector_meta`, records which clusters each document's sparse chunks were assigned to, used only for stale-key cleanup on delete and upsert.

#### Two-Pass Search

**Pass 1 — Sparse (SingleBit):**
1. The query text is embedded (or fetched from the `system_qemb_cache` TTL namespace; TTL configurable via `query_embedding_cache_ttl_secs`, default 1 day).
2. The top-`n_probes` clusters by Euclidean distance are identified; the union of probe sets across all query chunks is scanned.
3. All SingleBit entries for the probed clusters are scanned from `{ns}_sparse_vector`.
4. An optional attribute-predicate filter is applied per candidate (only in this pass) — non-matching documents are excluded.
5. Candidates are scored with `SingleBitQuanDotProductEstimator` and aggregated via **SimMax** (max score per `doc_id` across all clusters × all query chunks).
6. The top `first_pass_sparse_search_top_k` candidates are retained.

**Pass 2 — Dense (MultiBit):**
1. MultiBit entries for all sparse candidates are fetched directly from `{ns}_dense_vector` by `doc_id` in parallel — O(1) per lookup, no cluster scan.
2. Candidates are grouped by `cluster_id` and scored in parallel with `MultiBitQuanDotProductEstimator`.
3. The top-K results are returned, sorted by estimated dot product (higher = more similar).

The attribute-predicate filter is applied **only in Pass 1**. Pass 2 operates on the already-filtered candidate list.

#### RaBitQ Quantisation

RaBitQ encodes a floating-point vector as a compact bit string plus scalar correction factors (`addition_factor`, `scaling_factor`) and a per-vector `error_bound`. The error bound guarantees that the estimated dot product deviates from the true dot product by at most `error_bound` with high probability.

Accuracy is validated in the test suite: at 8 bits, the relative error against the exact dot product (computed on full-precision f32 vectors with simsimd) is consistently below 0.1%.

#### Embedding Service

The embedding service is an external HTTP endpoint. Chunking/tokenisation happens in minnal (`semantic_search::chunking`); the service receives a batch of pre-prepared payload strings and returns one embedding per payload:

```
POST {base_url}/embedding/document   {"payloads": [str, ...], "dimensions": N}  →  {"embeddings": [[f32], ...]}
POST {base_url}/embedding/query      (same request/response shape)
GET  {base_url}/healthcheck
```

A whole-text embedding is a one-element `payloads` array; chunked embeddings send one payload per sliding-window chunk. Configure the base URL and model via `[semantic_search]` in the TOML config (defaults: `http://localhost:8001`, model `qwen`).

#### Adding a New Embedding Model

The set of models is **data-driven** — no code change or recompile is required. A model is defined entirely by (a) a cluster-centroid file on disk and (b) a config entry that names it. Adding one (assuming an external embedding service that serves it) is:

1. **Generate the cluster centroids offline.** Run k-means (or any IVF clustering) over a representative corpus *embedded with the new model*. The number of centroids is the IVF cluster count (more clusters → finer partitioning, the trade-off knob against `n_probes`). Every centroid vector must have the model's embedding dimension.

2. **Write them as a JSONL file** — one JSON object per line (not a JSON array), each with a unique `cluster_id` and its `centroid` vector:

   ```jsonl
   {"cluster_id": 0, "centroid": [0.0123, -0.0456, 0.0789, ...]}
   {"cluster_id": 1, "centroid": [-0.0021,  0.0095, 0.0310, ...]}
   {"cluster_id": 2, "centroid": [ 0.0440, -0.0177, 0.0002, ...]}
   ```

   Rules enforced at startup: every centroid must be the **same, non-zero dimension**, that dimension must match the `dimension` declared in config, `cluster_id`s must be **unique**, and no centroid may be empty. A malformed file is rejected at boot, not at first query.

3. **Place the file** at `service/embedding_support/{model}/clusters.json`, where `{model}` is the lower-cased model name (e.g. `service/embedding_support/e5/clusters.json`).

4. **Declare the model** in the TOML config and select it as active:

   ```toml
   [semantic_search]
   model         = "e5"        # active model (must match an entry below; case-insensitive)
   embedding_dim = 1024        # must equal the centroid dimension
   cluster_path  = "service/embedding_support/e5/clusters.json"

   [[semantic_search.supported_models]]
   name      = "e5"            # → service/embedding_support/e5/clusters.json
   dimension = 1024            # must equal the centroid dimension
   ```

   At startup the server validates that each declared model's cluster file exists and that its centroids match the declared `dimension`, and — when the list is non-empty — that the active `model` is one of the declared entries.

5. **Point the embedding service at the model** so `/embedding/document` and `/embedding/query` return vectors of the matching dimension. The model name is *not* sent to the service (the URL has no model segment); which concrete model produces the embeddings is the service's own concern. minnal uses the name only to pick the cluster file and dimension.

6. **Re-index affected namespaces.** Existing vectors were quantised against the previous model's centroids/dimension and are not comparable. Since secondary indices are reconstructable, re-embed each namespace you want searchable under the new model with `POST /admin/indices/{ns}/vector/reindex-all` — it re-enqueues every document for embedding (a fresh full build) and returns `202 Accepted`. For a clean slate first (recommended when the dimension changes), clear the old vectors with `DELETE /admin/indices/{ns}/vector/drop-all` before re-indexing.

For the full end-to-end design — embedding generation, dual quantisation, index structure, two-pass query execution, storage layout, crash recovery, and hybrid search — see [`semantic_search/Semantic-Search-Architecture.md`](semantic_search/Semantic-Search-Architecture.md).

### Layer 4 — minnal\_doc\_store (Document Store + KV Store)

The three layers below are independent engines — one stores bytes, one indexes fields, one searches vectors. `minnal_doc_store` is where they come together: it sits on top of `minnal_db`, `index`, and `semantic_search` and presents them as a coherent document model, exposing two distinct store types behind a unified `DocStore` handle.

**Document stores (`DocStoreSchema`)**

Each document namespace declares:
- A primary key type (`uuid`, `u64`, or `u128`)
- Up to 5 field indices (typed: `str`, `int`, or `bool`)
- Any number of non-indexed attribute declarations
- Whether semantic search is enabled and which fields to embed

**KV stores (`KvStoreSchema`)**

A simpler, schema-lite alternative backed by the same minnal_db namespace. Each KV namespace declares:
- A key type: `str` (UTF-8 bytes) or `int` (big-endian `i64` for ordered scans)
- A value type: `int`, `str`, `f32`, or `vec_f32`
- Optionally `semantic_search_enabled = true` when `value_type = str`

KV stores have no field indices; field-index predicate queries belong to document stores only. Both store types support range scans and prefix scans. The two schema types share the same `schema_dir` and are distinguished on disk by a mandatory `store_type` field (`"doc"` vs `"kv"`) that every schema must declare.

Schemas are serialised as JSON and written atomically (tmp-then-rename) to a `schema_dir`.

**Index builds**

Adding an index to a namespace that already has documents triggers a background rebuild. Progress is tracked in a `build_progress.json` checkpoint file. If the server is stopped mid-build, it automatically detects the incomplete state on the next startup and resumes from the last checkpoint — no data is lost.

**Semantic search integration — async vector indexing**

When `semantic_search_enabled` is `true` and an embedding field is declared, every `put` (or `kv_put`) call automatically:
1. Extracts the nominated text field from the JSON document.
2. Writes the document, then enqueues a `(namespace, doc_id, text)` entry in a durable system-wide vector-index queue (`system_pending_vec_index` KV namespace) as a separate single-op write. A crash between the two leaves the document un-indexed but never acked-but-lost — startup reconciliation re-enqueues it (see below).
3. A background `VecIndexWorker` drains the queue: it calls the embedding service, quantises the result, writes the `VectorIndex` to the companion stores, then removes the queue entry. The worker is idempotent, so a crash between those writes simply re-processes the entry.

Document writes return immediately without blocking on the embedding service. Vector index entries may lag slightly behind the most recent writes. If the embedding service is temporarily unavailable, entries are retried with configurable back-off (see `[vector_index]` below).

**Robust search results.** Because the document and its vector index are separate writes that can drift (a crash window on write, or a delete racing the async indexer), a search candidate's `doc_id` might not resolve to a live document. The search path fetches each hit's document and **filters out any that no longer exist**, so a result always resolves to a real document — orphaned index entries never surface as dangling hits.

**Reconciliation.** Reconciliation re-enqueues any document missing **both** a *complete* committed vector index and a pending queue entry — self-healing the write-then-enqueue crash window the same way field indices self-heal via WAL replay, but routing the recovered work into the async queue. It also covers the vector write itself: the quantised payloads are written with `put_no_wal`, so a crash before the memtable flush drops a just-indexed vector — possibly flushing one half and losing the other. A "complete" index requires **both** the sparse-meta and dense entries, so a partially committed index counts as not-indexed and is re-enqueued, letting the re-embed regenerate the missing half. It runs **automatically as a background task on startup**, and on demand via `POST /admin/indices/vector/reconcile` (e.g. to re-run after the startup pass logs a failure). A cheap count short-circuit (nothing queued **and** both the sparse-meta and dense key counts already cover the live-key count, all LSM-only key scans) skips the full per-document scan for namespaces already fully covered.

Rapid successive writes to the same document overwrite the queue entry — the worker makes exactly one embedding call for the most-recent text (deduplication by `(namespace, doc_id)`).

### Layer 5 — minnal\_doc\_store\_api (REST API)

An [Axum](https://github.com/tokio-rs/axum) HTTP server that wraps `minnal_doc_store` with a full REST interface. It loads the schema cache and cluster index at startup, then serves all store, document, index, and semantic-search endpoints.

Server state:

| Field | Type | Purpose |
|---|---|---|
| `store` | `Arc<DocStore>` | Shared document + KV store |
| `schemas` | `Arc<RwLock<HashMap<String, DocStoreSchema>>>` | In-memory doc-store schema cache |
| `kv_schemas` | `Arc<RwLock<HashMap<String, KvStoreSchema>>>` | In-memory KV-store schema cache (key-type lookup on hot path) |
| `index_manager` | `Arc<IndexBuildManager>` | Active background index builds (field and vector) |
| `cluster_index` | `Option<Arc<ClusterIndex>>` | Pre-loaded IVF cluster index |
| `attr_index_ops` | `Arc<Mutex<HashSet<String>>>` | Namespaces with an active attribute index operation |
| `vec_index_cleanup` | `Arc<Mutex<HashSet<String>>>` | Namespaces with an active vector index cleanup |

---

## Salient Features

The layered architecture above produces a specific set of capabilities. The table below summarises what minnal offers and where each feature comes from in the stack.

| Feature | Detail |
|---|---|
| **WiscKey-style KV engine** | Keys and values stored separately. LSM tree stays small; large values avoid compaction churn. |
| **Multi-namespace isolation** | Each logical store has independent LSM, value-log, and index shards. |
| **Crash-safe WAL** | All mutations go through a WAL before they reach the memtable. Replay on startup. |
| **Background GC** | Value-log GC reclaims space from deleted/overwritten entries. Tunable waste threshold. |
| **SIMD-accelerated prefix scan** | `scan_prefix` merges across all storage layers. Short prefixes (≤ 8 bytes) match against a stored `u64` key fingerprint — a single integer comparison. Skip-list traversal uses AVX2 (32 bytes/cycle) or AVX512 (64 bytes/cycle) for full-key ordering when fingerprints tie. SSTable files are skipped entirely when their `min`/`max` key prefix bounds exclude the target. |
| **RoaringBitmap field indices** | Compressed, SIMD-accelerated bitmap indices for `str`, `int`, and `bool` predicates. |
| **Restartable index builds** | Checkpoint files track progress; interrupted builds resume automatically on startup. |
| **IVF + RaBitQ two-pass vector search** | Sparse first pass scans compact 1-bit chunk embeddings (32× compression vs f32) over IVF clusters for fast candidate selection; dense second pass re-ranks using multi-bit whole-doc embeddings (4× compression) for high-precision scoring. |
| **Async vector indexing** | Document writes enqueue a pending embedding job and return immediately. A background worker (round-robin across namespaces, bounded concurrency) drains the queue and retries on failure. Configurable retry budget and concurrency under `[vector_index]`. |
| **Quantisation error bounds** | Per-document `error_bound` field gives a theoretical guarantee on the dot-product estimate. |
| **Filtered semantic search** | Combine ANN scoring with an index predicate in a single request. |
| **Schema amendments** | Non-indexed attributes can be added, updated, or removed at any time without downtime. |
| **Three doc key types** | `uuid`, `u64`, `u128` — stored big-endian so range scans return ascending order. |
| **KV store** | Schema-lite namespaces (`store_type: "kv"`, data under `/stores/{ns}/kv`) for raw key-value data. Key types: `str`, `int`. Value types: `str`, `int`, `f32`, `vec_f32`. Range scan and prefix scan exposed via REST. Same durability guarantees as doc stores; no field indices. |
| **JSONL bulk loader** | `minnal_tools bulk_load` streams arbitrarily large JSONL files into a namespace via the REST API — document stores by default, KV stores with `--kv` — optionally importing the store's schema first (`--schema`). |
| **Self-contained core** | Storage, field indexing, vector quantisation, ANN search, and the server all run in a single Rust process. The **one** external dependency is the embedding service — and only when semantic search is enabled: embedding *generation* is delegated to an HTTP endpoint, while quantisation and search stay in-process. KV and document storage, field indexing, and predicate queries need nothing external. |

---

## Embedded Use (minnal_db as a Library)

Minnal has two usage modes. The [How to Use](#how-to-use) section below covers the **full server** — the Axum REST API over the whole stack (documents, predicate queries, semantic search). But the bottom layer, [`minnal_db`](#layer-1--minnal_db-key-value-engine), is also a standalone **embedded key-value store** you can link directly into any Rust process — no server, no daemon. You call `Db::open` (or `AsyncDb::open`) on a directory and get a durable, namespaced KV store with all background workers (compaction, value-log GC, WAL GC, TTL) running in-process.

The dependency direction is strictly downward — `minnal_db` knows nothing of the layers above it. Its one workspace dependency is `index`, because **secondary (field-level) indexing is a built-in capability of the KV engine itself**, not something the document store layers on.

### What an embedded store can and cannot do

| Capability | Where it lives | Embeddable via `minnal_db`? |
|---|---|---|
| Key-value CRUD, namespaces, TTL, typed (rkyv) values | `minnal_db` | ✅ Yes |
| RoaringBitmap **field/secondary index** + predicate query DSL | `index`, wired into `minnal_db` | ✅ Yes |
| JSON schema, document lifecycle, extractor generation | `minnal_doc_store` | ❌ No — higher layer |
| Semantic / vector (IVF + RaBitQ) search | `semantic_search` + `minnal_doc_store` | ❌ No — higher layer |

MinnalDB stores **opaque value bytes** — it never assumes a format. The field index is driven by an *extractor closure* you supply (`&[u8] -> Option<IndexValue>`), so you decide how to pull an indexed field out of your own value encoding (JSON, bincode, rkyv, a fixed binary layout, …). Deriving those extractors from a JSON schema is precisely what `minnal_doc_store` adds on top; the indexing machinery itself is engine-level. See [`minnal_db/README.md`](minnal_db/README.md#minnaldb-as-an-embedded-store) for the full engine documentation.

> **Not published to crates.io.** Because `index` is a path dependency, embedding `minnal_db` elsewhere means pulling in both crates (via path or git), not `cargo add minnal_db`. **Platform:** Linux and macOS only (`pread`/`pwrite`).

### Field-Level Indexing

The example below opens a store, indexes two fields (`status`, `age`), writes a few records keyed by `u64`, and runs a predicate query — all in-process, with no document-store layer involved. Here the values are a **typed `rkyv` struct** (`User`), so the extractors read the indexed fields straight off the zero-copy archive; the engine still only ever sees opaque value bytes.

```rust
use std::sync::Arc;
use minnal_db::rkyv_derives::{Archive, Deserialize, Serialize};
use minnal_db::{
    access, rancor, Archived, Db, ExtractorFn, IndexValue, IndexValueType, KVError,
    DEFAULT_NAMESPACE_ID,
};

// The value type. Deriving the rkyv traits also generates `ArchivedUser`,
// which the extractors below borrow zero-copy. The derive macros are
// re-exported from minnal_db, so no direct rkyv dependency is needed for them.
#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize)]
struct User {
    status: String,
    age: i64,
}

fn main() -> Result<(), KVError> {
    let db = Db::open("/tmp/users_db")?;

    // 1. Declare which fields to index on the default namespace. Returns a FieldId.
    //    The schema is persisted in config.json, so after a restart you only
    //    re-activate (step 2) — you don't re-register.
    let status_field = db.register_index_field(DEFAULT_NAMESPACE_ID, "status", IndexValueType::Str)?;
    let age_field    = db.register_index_field(DEFAULT_NAMESPACE_ID, "age",    IndexValueType::Int)?;

    // 2. Activate each field with an *extractor*: a closure that pulls the field
    //    out of the raw stored value. minnal_db has no idea what your value bytes
    //    mean — here they're an rkyv archive of `User`, so we borrow it zero-copy
    //    with `access` and read the field off `ArchivedUser` (no full decode).
    let status_extractor: ExtractorFn = Arc::new(|bytes: &[u8]| {
        let user = access::<ArchivedUser, rancor::Error>(bytes).ok()?;
        Some(IndexValue::Str(user.status.as_str().to_string()))
    });
    let age_extractor: ExtractorFn = Arc::new(|bytes: &[u8]| {
        let user = access::<ArchivedUser, rancor::Error>(bytes).ok()?;
        Some(IndexValue::Int(user.age.to_native()))
    });
    db.activate_field_index(DEFAULT_NAMESPACE_ID, status_field, IndexValueType::Str, status_extractor)?;
    db.activate_field_index(DEFAULT_NAMESPACE_ID, age_field,    IndexValueType::Int, age_extractor)?;

    // 3. Write records with a plain `u64` key. Each put_typed rkyv-serialises key
    //    and value, runs the extractors, and updates the RoaringBitmap indices
    //    automatically — there is no separate "index" call.
    db.put_typed(&1u64, &User { status: "active".into(),   age: 30 })?;
    db.put_typed(&2u64, &User { status: "inactive".into(), age: 25 })?;
    db.put_typed(&3u64, &User { status: "active".into(),   age: 42 })?;
    db.put_typed(&4u64, &User { status: "active".into(),   age: 18 })?;

    // 4. Query the index with the predicate DSL (=, !=, <, <=, >, >=, AND, OR,
    //    BETWEEN, IN). Returns the raw (rkyv) key bytes of matching records.
    let keys = db.query_index(DEFAULT_NAMESPACE_ID, r#"status = "active" AND age > 20"#)?;

    // 5. Resolve matched keys: decode the archived u64, then get_typed the value.
    for key in keys {
        let id = access::<Archived<u64>, rancor::Error>(&key)
            .expect("key is an archived u64")
            .to_native();
        if let Some(user) = db.get_typed::<u64, User>(&id)? {
            println!("user {id} => status={}, age={}", user.status, user.age);
        }
    }
    // → user 1 and user 3  (active AND age > 20; user 4 is active but 18, user 2 is inactive)

    db.shutdown()?;
    Ok(())
}
```

Key points about the embedded field-index contract:

- **The extractor is yours, not the engine's.** MinnalDB stores opaque bytes; the `&[u8] -> Option<IndexValue>` closure is where you interpret them — here, a zero-copy `rkyv` archive, but it could equally be JSON, bincode, or a fixed binary layout. Return `None` to skip indexing a record for that field.
- **Match the types.** The `IndexValue` your extractor returns must match the `IndexValueType` you registered (`Bool`, `Int` (i64), or `Str`), or activation fails.
- **Closures can't be persisted — re-activate on restart.** The schema (field names, IDs, types) survives in `config.json`, so on reopen you skip `register_index_field` and just call `activate_field_index` again with the same closures. Activation also replays any un-checkpointed WAL tail into the index.
- **Indexing is synchronous with the write.** The field index is updated inline on `put`, so a query immediately after a write sees it (unlike the async semantic-search pipeline in the layers above).
- **Pagination.** Use `query_index_paginated(ns, predicate, offset, limit)` and register a `RowToKeyFn` via `set_row_id_fn` so only `offset + limit` keys are resolved (O(|hits|), no in-memory key map). An `AsyncDb` equivalent of the whole API exists for tokio contexts.

---

## How to Use

This section is a practical walkthrough: build the binaries, start the server, then exercise each part of the API with `curl`. The examples assume the server is running locally on its default port (`8080`); every request is self-contained, so you can copy any block and run it directly.

### Build

```bash
# Build all crates (debug)
cargo build

# Build optimised binaries
cargo build --release -p minnal_doc_store_api
cargo build --release -p tools     # minnal_tools (bulk_load, …)
```

### Start the Server

#### Release workflow (recommended)

Use `release.sh` to build optimised binaries and stage everything under `./work/bin/`:

```bash
# Build and stage release binaries + config
./service/scripts/release.sh

# Start the server
./work/bin/start.sh
```

`release.sh` generates `./work/bin/minnal.toml` from `config/sample.toml` with all data paths rewritten to `./work/doc_store/` as the base.  Run both commands from the workspace root.

#### Development workflow

```bash
# Debug build and run directly
cargo run -p minnal_doc_store_api -- config/sample.toml
```

The server listens on `0.0.0.0:8080` by default (configurable via `[api] listen_addr`). To stop it, send SIGINT (`Ctrl-C`) or SIGTERM.

### Quickstart: create a document store and bulk-load it

This is the fastest end-to-end path: stage the release, start the server, load a small bundled sample dataset, and query it — all from the workspace root.

A document store is described by a **schema**: a key type, zero or more attribute **indices** (RoaringBitmap field indexes you can filter on), and an optional set of **embedding fields** for semantic search. The schema below is deliberately minimal — one index field (`agency`) and one semantic-search field (`jobTitle`):

**1. Build and stage the release (with sample data), then start the server**

```bash
# Build optimised binaries, config, AND the sample data (-s) under ./work/
./service/scripts/release.sh -s

# Start the server (listens on :8080)
./work/bin/start.sh
```

The `-s` flag stages [`tools/sample_data/`](tools/sample_data) into `./work/sample_data/`, including the two files this quickstart uses: `jobs-mini-schema.json` and `jobs-mini.jsonl` (ten rows). It also stages a KV-store sample — `job-content-kv-schema.json` and `job-content-kv.jsonl` (twenty long job descriptions) — used by the [KV Store](#kv-store) bulk-load example.

**2. Look at the schema**

`jobs-mini-schema.json` is deliberately minimal — `u64` keys, one attribute index (`agency`), and one semantic-search field (`jobTitle`):

```json
{
  "namespace": "jobs",
  "store_type": "doc",
  "key_type": "u64",
  "attributes": [
    { "name": "jobTitle", "attr_type": "str" }
  ],
  "indices": [
    { "field": "agency", "index_type": "str" }
  ],
  "semantic_search_enabled": true,
  "embedding_fields": ["jobTitle"]
}
```

Each line of `jobs-mini.jsonl` is one document, e.g.:

```json
{"jobId": "17039645", "agency": "Ministry of Transport", "jobTitle": "Executive, Shared Mobility"}
```

**3. Import the schema and load the rows in one shot**

With `--schema`, the `bulk_load` tool POSTs the schema to `POST /admin/stores/import` (creating the `jobs` store) and then streams the JSONL in, using `jobId` as the `u64` document key. Re-running is safe — an existing store is reused.

```bash
./work/bin/run_tool.sh bulk_load --schema ./work/sample_data/jobs-mini-schema.json \
  http://localhost:8080 jobs jobId ./work/sample_data/jobs-mini.jsonl
# → schema imported — store 'jobs' created
# → done  loaded=10  skipped=0  total=10  elapsed=…s
```

**4. Query the store**

Filter on the indexed `agency` field. Attribute indexing is synchronous with the write, so the rows are queryable immediately:

```bash
curl -s -X POST http://localhost:8080/stores/jobs/query \
  -H 'Content-Type: application/json' \
  -d '{"predicate": "agency = \"National Library Board\""}'
# → {"results":[{"id":13354909,"doc":{...}},
#               {"id":13354910,"doc":{...}},
#               {"id":15120777,"doc":{...}}],
#    "total":3,"page_no":1,"page_size":20}
```

> **Semantic search needs an external embedding service.** The schema above sets `semantic_search_enabled: true`, but the actual embeddings are produced asynchronously by an embedding service (default `http://localhost:8001`). **Without it, the load and all attribute queries above still work** — only `POST /stores/jobs/semantic-search` returns nothing until embeddings exist. See [Semantic Search](#semantic-search) below to enable it.

### Configuration

See [`config/sample.toml`](config/sample.toml) for the full annotated reference. The most important sections:

```toml
[storage]
db_path    = "./data/db"       # where minnal_db stores its files
schema_dir = "./data/schemas"  # where per-store schema JSON files live

[api]
listen_addr = "0.0.0.0:8080"

[memtable]
max_capacity = 100_000         # skip-list capacity before flush

[sharding]
num_buckets = 8                # value-log shards (fixed at DB creation)

[sync]
records_per_sync = 1_000       # WAL fsync cadence

[thresholds]
value_log_waste_threshold = 30.0   # GC when >30% of value-log is stale

[scheduled_tasks]
value_log_gc_interval_secs   = 60
wal_gc_interval_secs         = 60
lsm_compaction_interval_secs = 60
ttl_cleanup_interval_secs    = 3_600

[wal]
segment_size_bytes = 67_108_864    # 64 MiB per WAL segment

[value_log]
page_size_bytes = 67_108_864       # 64 MiB per value-log page

[vector_index]
# Seconds to wait after a pass with at least one embedding failure before re-scanning.
retry_wait_secs = 2
# Max embedding attempts per queue entry; exhausted entries need manual removal.
max_retries = 5
# Max concurrent embedding calls in flight at once (entries are round-robin across namespaces).
concurrency = 4

[semantic_search]
# Path to the JSONL cluster centroids file (one {"cluster_id":…,"centroid":[…]} per line).
cluster_path = "service/embedding_support/qwen/clusters.json"
# Bits per dimension for the dense (multi-bit) Pass 2 quantisation. 4 = compact, 8 = high recall.
number_of_bits_for_dense_quantisation = 8
# IVF clusters probed in the sparse first pass (higher = better recall, slower).
# n_probes = 32
# Candidates kept after Pass 1 before dense re-ranking.
# first_pass_sparse_search_top_k = 1000
# Sliding-window chunk parameters for Pass 1 single-bit embeddings.
# window_size = 4
# sliding_size = 2
```

> **Note:** `window_size` and `sliding_size` control how text is split into chunks for the sparse (Pass 1) embeddings, and the *same* values are used to chunk both documents at index time and queries at search time. Changing them after documents have been indexed makes stored chunks and new query chunks inconsistent, which silently degrades recall (no error is raised). Treat them as a fixed indexing decision: if you change either value, re-embed the corpus with `POST /admin/indices/{ns}/vector/reindex-all`.

The config file is located by (in order): first CLI argument → `MINNAL_CONFIG_FILE` env var → built-in defaults.

### Store Lifecycle

```bash
# Create a store with UUID keys, two indices, and semantic search
curl -s -X POST http://localhost:8080/stores \
  -H 'Content-Type: application/json' \
  -d '{
    "namespace": "profiles",
    "store_type": "doc",
    "key_type": "uuid",
    "semantic_search_enabled": true,
    "embedding_fields": ["bio"],
    "indices": [
      {"field": "status",   "index_type": "str"},
      {"field": "seniority","index_type": "str"}
    ],
    "attributes": [
      {"name": "name",  "attr_type": "str"},
      {"name": "email", "attr_type": "str", "description": "primary contact"},
      {"name": "bio",   "attr_type": "str", "description": "embedded for semantic search"}
    ]
  }'
# → 201 Created

# List all stores
curl -s http://localhost:8080/stores

# Amend a non-indexed attribute
curl -s -X PATCH http://localhost:8080/stores/profiles/schema \
  -H 'Content-Type: application/json' \
  -d '{"op": "add_attribute", "name": "team", "attr_type": "str"}'
# → 204 No Content

# Drop a store (irreversible)
curl -s -X DELETE http://localhost:8080/stores/profiles
# → 204 No Content
```

Schema amendment operations: `add_attribute`, `update_attribute`, `remove_attribute`. Indexed fields cannot be amended directly — drop the index first.

### Document CRUD

```bash
# Insert (upsert)
curl -s -X PUT \
  "http://localhost:8080/stores/profiles/docs/550e8400-e29b-41d4-a716-446655440000" \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "Alice",
    "status": "active",
    "seniority": "senior",
    "bio": "distributed systems engineer with a focus on storage engines"
  }'
# → 204 No Content

# Read by ID
curl -s "http://localhost:8080/stores/profiles/docs/550e8400-e29b-41d4-a716-446655440000"
# → {"name":"Alice","status":"active","seniority":"senior","bio":"..."}

# Delete
curl -s -X DELETE \
  "http://localhost:8080/stores/profiles/docs/550e8400-e29b-41d4-a716-446655440000"
# → 204 No Content

# Range scan (key order, inclusive start, exclusive end)
curl -s "http://localhost:8080/stores/profiles/docs?start=00000000-0000-0000-0000-000000000000"
# → [{"id":"550e8400-...","doc":{...}}, ...]
```

### Predicate Queries

Only indexed fields may appear in predicates. Operators: `=`, `!=`, `<`, `<=`, `>`, `>=`, `AND`, `OR`, `NOT`.

```bash
curl -s -X POST http://localhost:8080/stores/profiles/query \
  -H 'Content-Type: application/json' \
  -d '{"predicate": "status = \"active\" AND seniority = \"senior\""}'
# → [{"id":"550e8400-...","doc":{...}}, ...]
```

### Semantic Search

Semantic search requires:
- The store was created with `"semantic_search_enabled": true` — a doc store also needs `"embedding_fields"` set; a `value_type = str` KV store embeds the stored string directly.
- A cluster centroids file is configured and loaded at startup (`[semantic_search] cluster_path = …`).
- An embedding service is reachable at the configured URL.

```bash
# Unfiltered — all candidates ranked by similarity
curl -s -X POST http://localhost:8080/stores/profiles/semantic-search \
  -H 'Content-Type: application/json' \
  -d '{"query": "senior engineer with distributed systems experience"}'

# Filtered — ANN scoring restricted to documents that pass the predicate
curl -s -X POST http://localhost:8080/stores/profiles/semantic-search/filtered \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "senior engineer with distributed systems experience",
    "predicate": "status = \"active\""
  }'
```

Response (both endpoints) — ordered array, highest similarity first:

```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "dot_product": 0.94,
    "error_bound": 0.02,
    "cluster_id": 7,
    "is_primary": true
  }
]
```

| Field | Meaning |
|---|---|
| `dot_product` | Estimated cosine similarity to the query (higher = more similar) |
| `error_bound` | Theoretical max deviation of the estimate from the true dot product |
| `cluster_id` | IVF cluster this document was indexed under |
| `is_primary` | `true` if the document's cluster is the closest cluster to the query |

### Admin and Monitoring

The `/admin` routes are split into two groups:

- `/admin/storage/*` — storage engine diagnostics and GC/compaction triggers
- `/admin/indices/*` — index monitoring and bulk index operations

They are intended for operations and monitoring — not application traffic.

```bash
# Liveness check
curl http://localhost:8080/admin/storage/health
# → {"status":"ok","uptime_s":42}

# Value-log GC statistics
curl http://localhost:8080/admin/storage/stats

# LSM manifest (all namespaces)
curl http://localhost:8080/admin/storage/lsm

# All active index builds + vector progress across all namespaces
curl http://localhost:8080/admin/indices/progress

# Index progress for one namespace
curl http://localhost:8080/admin/indices/products/progress
# → {"attribute_builds":[...],"vector_progress":[{"namespace":"products","indexed_approx":970,"pending":10,"exhausted":2,"progress_pct":99.0}]}

# Vector-index queue depth / lag by namespace
curl http://localhost:8080/admin/indices/vector/queue/summary
# → {"max_retries_configured":5,"total_pending":12,...}

# All retried queue entries (retry_count > 0) across all namespaces
curl http://localhost:8080/admin/indices/vector/queue/retried

# Queue entries for one namespace
curl http://localhost:8080/admin/indices/products/vector/queue

# Retried entries for one namespace
curl http://localhost:8080/admin/indices/products/vector/queue/retried

# Remove a stuck queue entry (doc_id_hex from the listing above)
curl -X DELETE \
  http://localhost:8080/admin/indices/products/vector/queue/550e8400e29b41d4
# → 204 No Content

# Reset all exhausted entries for a namespace
curl -X POST http://localhost:8080/admin/indices/products/vector/reindex-failed
# → {"retried":3}

# Re-enqueue all docs for a fresh full vector rebuild
curl -X POST http://localhost:8080/admin/indices/products/vector/reindex-all
# → 202 Accepted

# Drop and rebuild all field indices for a namespace
curl -X POST http://localhost:8080/admin/indices/products/attribute/reindex-all
# → 202 Accepted

# Trigger value-log GC
curl -X POST http://localhost:8080/admin/storage/gc

# Trigger LSM compaction
curl -X POST http://localhost:8080/admin/storage/compact

# Monitor field-index bitmap/keymap waste (decide if a checkpoint is worthwhile)
curl http://localhost:8080/admin/storage/index-waste

# Force an index checkpoint (flush + compact field indexes and row maps)
curl -X POST http://localhost:8080/admin/storage/index-checkpoint
```

For the full admin API reference see [`minnal_doc_store_api/README.md`](minnal_doc_store_api/README.md). For a consolidated table of every metric these endpoints report — with explanations and whether each value survives a restart — see [Operational & Storage Metrics](minnal_doc_store_api/README.md#operational--storage-metrics).

### Index Management

```bash
# Add an index (returns 202; background rebuild if documents already exist)
curl -s -X POST http://localhost:8080/stores/profiles/indices \
  -H 'Content-Type: application/json' \
  -d '{"field": "team", "index_type": "str"}'
# → 202 Accepted

# Monitor rebuild progress (per namespace)
curl -s http://localhost:8080/admin/indices/profiles/progress
# → {"attribute_builds":[{"id":{"Field":{"namespace":"profiles","field":"team"}},"status":"Running","total":50000,"indexed":12340,...}],"vector_progress":[...]}

# Drop a field index (background cleanup, field becomes a plain attribute)
curl -s -X DELETE http://localhost:8080/stores/profiles/indices/team
# → 202 Accepted

# Drop the vector index (disables semantic search, background cleanup)
curl -s -X DELETE http://localhost:8080/stores/profiles/indices/vector
# → 202 Accepted

# Admin: drop + rebuild all field indices for a namespace
curl -s -X POST http://localhost:8080/admin/indices/profiles/attribute/reindex-all
# → 202 Accepted
```

### KV Store

A KV store is a schema-lite namespace for raw key-value data. It has no field indices but supports the same WAL durability, LSM compaction, value-log GC, and optional semantic search as document stores.

```bash
# Create a KV store (str key → str value)
curl -s -X POST http://localhost:8080/stores \
  -H 'Content-Type: application/json' \
  -d '{"namespace": "session-cache", "store_type": "kv", "key_type": "str", "value_type": "str"}'
# → 201 Created

# Create a KV store with semantic search (str key → str value, ANN enabled)
curl -s -X POST http://localhost:8080/stores \
  -H 'Content-Type: application/json' \
  -d '{"namespace": "product-descriptions", "store_type": "kv", "key_type": "str", "value_type": "str",
       "semantic_search_enabled": true}'
# → 201 Created

# List all stores (doc and KV; each entry carries its store_type)
curl -s http://localhost:8080/stores

# Write a value
curl -s -X PUT http://localhost:8080/stores/session-cache/kv/user-42 \
  -H 'Content-Type: application/json' \
  -d '"eyJhbGciOiJIUzI1NiJ9..."'
# → 204 No Content

# Read a value
curl -s http://localhost:8080/stores/session-cache/kv/user-42
# → "eyJhbGciOiJIUzI1NiJ9..."

# Delete a key
curl -s -X DELETE http://localhost:8080/stores/session-cache/kv/user-42
# → 204 No Content

# Range scan — entries with keys "user-10" through "user-20" (exclusive end)
curl -s "http://localhost:8080/stores/session-cache/kv?start=user-10&end=user-21"
# → {"results":[{"key":"user-10","value":"..."},...],"page_no":1,"page_size":20,"total":2}

# Prefix scan — all entries whose key starts with "user-"
curl -s "http://localhost:8080/stores/session-cache/kv/prefix?prefix=user-"
# → {"results":[{"key":"user-10","value":"..."},...],"page_no":1,"page_size":20,"total":3}

# Semantic search (value_type = str, semantic_search_enabled = true only)
curl -s -X POST http://localhost:8080/stores/product-descriptions/kv/semantic-search \
  -H 'Content-Type: application/json' \
  -d '{"query": "lightweight waterproof running shoes", "top_k": 5}'

# Drop a KV store (irreversible)
curl -s -X DELETE http://localhost:8080/stores/session-cache
# → 204 No Content
```

**Key and value types:**

| `key_type` | URL path segment format |
|-----------|------------------------|
| `str` | any UTF-8 string |
| `int` | decimal integer (stored big-endian for ordered scans) |

| `value_type` | JSON body format |
|-------------|-----------------|
| `str` | JSON string: `"hello"` |
| `int` | JSON integer: `42` |
| `f32` | JSON number: `3.14` |
| `vec_f32` | JSON array of numbers: `[1.0, -0.5, 0.25]` |

To populate a KV store from a JSONL file in one step, use `bulk_load --kv` — the
release bundles a sample str→str store (`job-content-kv-schema.json` +
`job-content-kv.jsonl`, 20 long job descriptions keyed by job id). See
[Bulk Loading](#bulk-loading) below.

### Bulk Loading

A single `minnal_tools bulk_load` command streams a JSONL file line by line and
PUTs each row via the REST API. It loads a **document store** by default; pass
`--kv` to load a **KV store** instead. By default the namespace must already
exist; pass `--schema <schema.json>` to import the schema first, which turns it
into a one-step "fresh server → populated store" load:

| Invocation                      | Imports via | Creates the store? | Use when |
|---------------------------------|-------------|--------------------|----------|
| `bulk_load --schema …`          | `POST /admin/stores/import`    | Yes — imports the schema first, then loads | Fresh server → populated doc store in one step |
| `bulk_load` (no `--schema`)     | —           | No — the namespace must already exist | Adding more rows to a doc store you created earlier |
| `bulk_load --kv --schema …`     | `POST /admin/stores/import` | Yes — imports the schema first, then loads | Fresh server → populated KV store in one step |
| `bulk_load --kv` (no `--schema`) | —          | No — the namespace must already exist | Adding more rows to a KV store you created earlier |

When `--schema` is given, the tool validates that the schema's `key_type` matches
the selected store kind — e.g. passing `--kv` with a document schema (`u64` /
`u128` / `uuid`) fails fast with a clear message, and vice versa.

**Document stores** (default) — each line becomes one document; `id_field` names
the field holding the document ID:

```bash
# Import <schema.json> first, then load <data.jsonl>. Re-runs are safe — an
# existing store is reused. The schema's namespace must match <namespace>.
./work/bin/run_tool.sh bulk_load --schema <schema.json> <url> <namespace> <id_field> <data.jsonl>
./work/bin/run_tool.sh bulk_load --schema ./work/sample_data/jobs-mini-schema.json http://localhost:8080 jobs jobId ./work/sample_data/jobs-mini.jsonl

# Load <data.jsonl> into an existing <namespace>.
./work/bin/run_tool.sh bulk_load <url> <namespace> <id_field> <data.jsonl>
./work/bin/run_tool.sh bulk_load http://localhost:8080 profiles id profiles.jsonl

# Development: build and run directly via cargo (instead of the staged binary)
cargo run -p tools -- bulk_load http://localhost:8080 profiles id profiles.jsonl
```

The `id_field` value is parsed according to the namespace's `key_type` (`u64`,
`u128`, or `uuid`).

**KV stores** (`--kv`) — each line supplies a key and a value via separate
fields; `key_field` and `value_field` name them. The value is sent verbatim and
validated against the namespace's `value_type`:

```bash
# Import <schema.json> first, then load <data.jsonl>. The bundled sample is a
# str→str KV store of 20 long job-description blobs keyed by job id.
./work/bin/run_tool.sh bulk_load --kv --schema <schema.json> <url> <namespace> <key_field> <value_field> <data.jsonl>
./work/bin/run_tool.sh bulk_load --kv --schema ./work/sample_data/job-content-kv-schema.json \
  http://localhost:8080 job-content key value ./work/sample_data/job-content-kv.jsonl

# Load <data.jsonl> into an existing KV <namespace>.
./work/bin/run_tool.sh bulk_load --kv <url> <namespace> <key_field> <value_field> <data.jsonl>
./work/bin/run_tool.sh bulk_load --kv http://localhost:8080 job-content key value ./work/sample_data/job-content-kv.jsonl
```

The `key_field` value is parsed according to the namespace's `key_type` (`str` or
`int`).

For both store kinds each line must be a valid JSON object, and rows with missing
or unparseable keys/IDs are skipped, counted in `skipped`, and written to a
sibling `<data>.errors` file. Pass `--no-wal` (before the positional arguments)
for maximum throughput when re-running the load is acceptable — data written that
way is unrecoverable on a crash. `--no-wal` works for both document and KV stores.

See [Quickstart](#quickstart-create-a-document-store-and-bulk-load-it) for a
complete `bulk_load` walkthrough with a sample schema and rows.

### Logging

Minnal uses the [`tracing`](https://docs.rs/tracing) crate throughout. Log verbosity can be set either in the TOML config file or via the `RUST_LOG` environment variable. `RUST_LOG` always takes precedence over the config file setting.

#### TOML configuration

The `[logging]` section controls the default log level and the `[storage]` section controls where rolling log files are written:

```toml
[storage]
# Directory where rolling log files are written.
log_dir = "./data/log"

[logging]
# Minimum log level when RUST_LOG is not set.
# Accepted values: "error", "warn", "info", "debug", "trace"
level = "info"
```

`log_dir` is relative to the working directory (the workspace root when launched via `./work/bin/start.sh` or `cargo run`).

#### RUST_LOG environment variable

The `env-filter` feature lets you target specific crates or modules, overriding the `[logging] level` value entirely:

```bash
# Warnings and above only (quiet)
RUST_LOG=warn ./target/release/minnal_doc_store_api config/sample.toml

# Info level (default recommended)
RUST_LOG=info ./target/release/minnal_doc_store_api config/sample.toml

# Debug messages
RUST_LOG=debug ./target/release/minnal_doc_store_api config/sample.toml

# Per-crate control — debug for the KV engine, info elsewhere
RUST_LOG=info,minnal_db=debug ./target/release/minnal_doc_store_api config/sample.toml

# Narrow to a single module
RUST_LOG=minnal_db::db::database=trace ./target/release/minnal_doc_store_api config/sample.toml
```

Level hierarchy (most → least verbose): `trace > debug > info > warn > error`.

The same variable works with the staged binary and `cargo run`:

```bash
RUST_LOG=info ./work/bin/start.sh
RUST_LOG=info cargo run -p minnal_doc_store_api -- config/sample.toml
```

### Write Durability and Recovery

All mutating requests (document and KV store writes and deletes) follow a WAL-first durability model.

#### Request lifecycle

1. The write is appended to the Write-Ahead Log and fsynced to disk.
2. If WAL persistence fails, the request returns **500 Internal Server Error**. No data is lost because nothing was committed.
3. Once the WAL entry is durable, the write is applied to the in-memory structures (value log + memtable). The in-memory apply is best-effort: if it fails, the error is logged at `ERROR` level but the request still returns **204 No Content**. The data is safe in the WAL and will be recovered on the next startup.
4. All SSTable flushes, LSM compaction, value-log GC, and WAL GC happen asynchronously in background workers and do not affect the request latency or response.

In short: a successful response (204) means the write is durable in the WAL. It does not mean it has been flushed to an SSTable.

#### WAL recovery on startup

When the server starts, it replays any WAL entries that have not yet been flushed to SSTables:

1. Each entry is applied to the appropriate in-memory store.
2. If an entry fails to apply, it is **retried once**.
3. If the retry also fails, the failed operation is written to a **fail log file** and skipped.

#### Fail log files

Fail logs are written to the directory configured by `recovery.fail_log_dir` (defaults to `<db_path>/fail_logs`). A separate file is created for each recovery run, named:

```
fail_log_YYYY-MM-DDTHH-MM-SS.json
```

Each file is a JSON document with this structure:

```json
{
  "recovery_timestamp": "2026-05-03T10-22-01",
  "db_path": "/data/db",
  "failed_operations": [
    {
      "name": "put_doc",
      "operation": "Put",
      "namespace_id": 3,
      "key": "550e8400-e29b-41d4-a716-446655440000",
      "value": { "name": "Alice", "status": "active" },
      "error": "..."
    }
  ]
}
```

A user or operator can inspect these files and take corrective action — replay the missing writes via the REST API, delete the affected keys, or ignore them if the data is no longer needed.

---

## Scripts and Config

### Server scripts

| Script | Purpose |
|---|---|
| [`service/scripts/release.sh`](service/scripts/release.sh) | Build release binaries and stage them under `./work/bin/`. Generates `minnal.toml` (paths rewritten for `./work/doc_store/`), `start.sh`, and `run_tool.sh` in the same directory. |
| [`service/scripts/build_docker.sh`](service/scripts/build_docker.sh) | Build the Docker image from the workspace root. Accepts an optional image tag argument. |

**Generated by `release.sh` into `./work/bin/`:**

| Script | Purpose |
|---|---|
| `./work/bin/start.sh` | Start `minnal_doc_store_api` using the staged binary and `minnal.toml`. |
| `./work/bin/run_tool.sh` | Run `minnal_tools` (e.g. `bulk_load`) using the staged binary. |

### Example scripts

The [`service/scripts/examples/`](service/scripts/examples/) directory contains self-contained `curl` demos that exercise the full API:

| Script | What it covers |
|---|---|
| [`service/scripts/examples/stores.sh`](service/scripts/examples/stores.sh) | Store lifecycle: create, list, amend schema, drop. Also covers error cases (duplicate store, amending an indexed field). |
| [`service/scripts/examples/docs.sh`](service/scripts/examples/docs.sh) | Full document CRUD, range scans, predicate queries, and **semantic search** — both filtered and unfiltered, with multiple query examples. Requires an embedding service running at the configured URL (default: `http://localhost:8001`). |
| [`service/scripts/examples/indices.sh`](service/scripts/examples/indices.sh) | Add index, poll progress until complete, drop index. Includes error cases (already indexed, field not found). |

Run any script against a running server:

```bash
./work/bin/start.sh                                # terminal 1
./service/scripts/examples/stores.sh              # terminal 2
./service/scripts/examples/docs.sh
./service/scripts/examples/indices.sh
```

### Cluster centroids

[`service/embedding_support/qwen/clusters.json`](service/embedding_support/qwen/clusters.json) contains 50 pre-computed cluster centroids for 768-dimensional embeddings. This sample cluster file is for the **Qwen Embedding model** — to use a different embedding model, generate your own centroids (see below). The file is in JSONL format — one cluster per line:

```json
{"cluster_id": 1, "centroid": [-0.6687, 0.6142, ...]}
{"cluster_id": 2, "centroid": [0.1234, -0.9876, ...]}
```

To generate your own centroids for a different model or dimensionality, run k-means (e.g. with `faiss` or `sklearn`) over a representative sample of your corpus embeddings and write the output in this JSONL format. Point the server at your file via `[semantic_search] cluster_path` in the config.

The cluster index is loaded once at startup and held in memory. For 50 clusters of 768 dimensions, the in-memory footprint is negligible (~150 KB).

### Sample config

[`config/sample.toml`](config/sample.toml) is a fully annotated configuration file covering all tunable parameters with their defaults and a brief explanation of each. It is the recommended starting point for new deployments.

---

## Crate Structure

```
minnal/
├── minnal_db/          ← LSM + value-log KV engine
│   └── README.md       ← detailed KV engine documentation
├── index/              ← RoaringBitmap field indexing + predicate evaluator
├── semantic_search/    ← IVF clustering, RaBitQ quantisation, embedding client
├── minnal_doc_store/   ← JSON document store (schema, indices, semantic search integration)
├── minnal_doc_store_api/  ← Axum REST API server
│   └── README.md       ← full REST API reference with curl examples
├── tools/              ← minnal_tools binary (bulk_load and future tools)
├── service/
│   ├── scripts/
│   │   ├── release.sh      ← build release binaries + stage to ./work/bin/
│   │   ├── build_docker.sh ← build the Docker image
│   │   └── examples/       ← curl demo scripts (stores, docs, indices)
│   ├── docker/
│   │   └── Dockerfile
│   └── embedding_support/
│       └── qwen/
│           └── clusters.json   ← 50 pre-computed IVF cluster centroids (JSONL)
├── config/
│   └── sample.toml     ← annotated reference configuration
└── work/bin/           ← generated by release.sh (not committed)
    ├── minnal_doc_store_api
    ├── minnal_tools
    ├── minnal.toml     ← config with ./work/doc_store paths
    ├── start.sh        ← start the server
    └── run_tool.sh     ← run bulk_load / other tools
```

For the detailed KV engine documentation see [`minnal_db/README.md`](minnal_db/README.md).
For the full REST API reference with all endpoints, error codes, predicate syntax, and on-disk layout see [`minnal_doc_store_api/README.md`](minnal_doc_store_api/README.md).

---

## Acknowledgements

### WiscKey

The storage engine at the core of minnal is directly inspired by the WiscKey paper. The key insight — storing only keys in the LSM tree and large values in a separate append-only log — is the foundational design decision of `minnal_db`.

> **WiscKey: Separating Keys from Values in SSD-Conscious Storage**
> Lanyue Lu, Thanumalayan Sankaranarayana Pillai, Andrea C. Arpaci-Dusseau, Remzi H. Arpaci-Dusseau
> *USENIX FAST '16*
> https://www.usenix.org/conference/fast16/technical-sessions/presentation/lu

### RaBitQ

The quantisation scheme used in `semantic_search` is RaBitQ, which provides a tight theoretical error bound on the estimated inner product between a quantised document vector and a full-precision query vector. This bound is what populates the `error_bound` field in semantic search responses.

> **RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search**
> Jianyang Gao, Cheng Long
> *ACM SIGMOD '24*
> https://dl.acm.org/doi/10.1145/3654970

The reference C++ implementation of RaBitQ, which informed the multi-bit quantisation and distance estimation code in this project, is available at:

> https://github.com/gaoj0017/RaBitQ

### RoaringBitmap

The field indexing layer in the `index` crate uses Roaring Bitmaps as its compressed bitmap representation. Roaring Bitmaps partition a 32-bit integer space into 65 536 chunks and choose the most space-efficient container type (array, bitset, or run-length encoded) per chunk, giving excellent compression on both sparse and dense sets while keeping set operations fast.

> **Roaring Bitmaps: Implementation of an Optimized Software Library**
> Daniel Lemire, Owen Kaser, Nathan Kurz, Luca Deri, Chris O'Hern, François Saint-Jacques, Gregory Ssi-Yan-Kai
> *Software: Practice and Experience, 2018*
> https://arxiv.org/abs/1709.07821

The canonical Java reference implementation, which established the on-disk format and container selection heuristics, is available at:

> https://github.com/RoaringBitmap/RoaringBitmap
