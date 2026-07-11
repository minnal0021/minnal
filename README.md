# minnal

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this software except in compliance with the License. You may obtain a copy of the License at:

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

---

**minnal** (மின்னல்) means *lightning* in Tamil.

Minnal is a layered document database written in Rust. It combines a high-performance LSM + value-log key-value engine with a JSON document layer, RoaringBitmap field indexing, and quantised approximate nearest-neighbour semantic search.

It works in **two ways**, from the same engine:

- **Embedded** — add the `minnal_db` crate to your Rust application and call the API in-process, with no server or network hop. Capabilities are opt-in via cargo features (`kv-store` default, `doc-store`, `semantic-search`). See the **[Embedded Quickstart](minnal_db/QUICKSTART.md)**.
- **Server (REST)** — run `minnal_db_api` to expose the same stores over an HTTP REST API for any client or language. See **[minnal_db_api](minnal_db_api/README.md)**.

For a hands-on server walkthrough — build, bulk-load, and every endpoint — see the **[Quickstart & Usage Guide](QUICKSTART.md)**.

> **Platform support:** Linux and macOS only. Windows is not supported — the storage engine relies on Unix positional I/O (`pread`/`pwrite`) and the server requires POSIX signals.

> **Companion UI:** [minnal0021/minnal_ui](https://github.com/minnal0021/minnal_ui) is a lightweight web UI that can be used to drive the minnal doc store.

> **Companion embedding service:** [minnal0021/embedding_service](https://github.com/minnal0021/embedding_service) serves the **gemma** embedding model over HTTP — the external dependency semantic search needs. Run it to get started with the minnal doc store's semantic search, then point `semantic_search.embedding_service_url` at it (default `http://localhost:8001`).

---

## Table of Contents

- [Overview](#overview)
- [Concepts Primer](#concepts-primer)
- [Architecture](#architecture)
  - [Layer 1 — KV Engine (`kv-store`, always on)](#layer-1--kv-engine-kv-store-always-on)
  - [Layer 2 — Field Indexing (RoaringBitmap, part of `kv-store`)](#layer-2--field-indexing-roaringbitmap-part-of-kv-store)
  - [Layer 3 — Semantic Search (Vector Quantisation + ANN, `semantic-search`)](#layer-3--semantic-search-vector-quantisation--ann-semantic-search)
  - [Layer 4 — Document Store + KV Store (`doc-store`)](#layer-4--document-store--kv-store-doc-store)
  - [Layer 5 — REST API (`minnal_db_api`)](#layer-5--rest-api-minnal_db_api)
- [Salient Features](#salient-features)
- [Quickstart & Usage](#quickstart--usage)
- [Repository Structure](#repository-structure)
- [Acknowledgements](#acknowledgements)

---

## Overview

Minnal is designed for use cases that need fast key-value storage, structured predicate queries on JSON fields, *and* semantic (embedding-based) similarity search — in a single embedded process. The only piece that lives outside the process is the embedding service, and only when semantic search is enabled — it generates the vector embeddings for documents (at index time) and for queries (at search time) that semantic search relies on. Everything else — quantisation, indexing, and the ANN search itself — runs in-process.

Everything below the REST boundary is a **single crate, `minnal_db`**, whose
capabilities are selected with cargo features — the KV engine and field indexing
are always present; the JSON document store and semantic search are opt-in:

| Layer (feature) | Role |
|---|---|
| KV engine + field indexing (`kv-store`, default) | LSM + value-log store, namespaces, TTL; RoaringBitmap field indices + predicate evaluator |
| Semantic search (`semantic-search`) | IVF clustering, RaBitQ quantisation, two-pass ANN — usable on raw KV or documents |
| Document store (`doc-store`) | JSON schema, document lifecycle, background index builders |
| `minnal_db_api` (separate binary crate) | Axum HTTP server exposing all of the above over REST |

The **only** component outside the process is the embedding service, and only
when `semantic-search` is enabled. For **embedded** use, add `minnal_db` and call
it in-process; the **server** simply wraps the same crate. Each namespace
maintains its own KV storage, field indices, and a companion vector store for
quantised embeddings.

---

## Concepts Primer

The sections below assume some familiarity with a handful of database-internals
terms. If you already know them, skip ahead to [Architecture](#architecture);
otherwise, here's the short version of each:

| Term | In one sentence |
|---|---|
| **LSM tree** (Log-Structured Merge tree) | An on-disk index tuned for fast writes: new data lands in a small in-memory buffer first and is merged into sorted files on disk later, in the background, instead of updating records in place. |
| **WiscKey** | An LSM variant used here: store only keys (plus a pointer) in the LSM tree, and put the bulky values in a separate append-only log, so compacting the tree never has to rewrite large values. |
| **IVF** (Inverted File index) | A way to make similarity search fast: pre-group all vectors into clusters by proximity, so a query only has to compare against the handful of clusters nearest to it instead of every stored vector. |
| **ANN** (Approximate Nearest Neighbour search) | Finding the vectors most similar to a query vector without an exhaustive, exact comparison against every stored vector — trading a small amount of accuracy for a large speed-up. |
| **RaBitQ** | A technique for compressing ("quantising") a high-precision vector into a compact bit string, with a mathematical bound on how far the compressed version's similarity score can drift from the true one. |
| **RoaringBitmap** | A compressed representation of a set of integers. Used here to record "which documents have field `X = Y`", so that combining several such conditions with AND/OR/NOT is both fast and memory-efficient. |

---

## Architecture

Minnal is built as layers, each adding a capability over the one below — a raw
key-value engine at the bottom, then structured indexing, vector search, a
document model, and finally an HTTP interface. All the library layers live in a
**single crate, `minnal_db`**, selected with cargo features (`kv-store` default,
`doc-store`, `semantic-search`); the REST server (`minnal_db_api`) is a thin
binary crate on top. Reading the layers in order is the fastest way to
understand how a single REST request ends up touching the LSM tree at the bottom.

### Layer 1 — KV Engine (`kv-store`, always on)

At the foundation sits `minnal_db`, an embedded, multi-namespace key-value store built on the **WiscKey** key-value separation principle: keys (with value-log pointers) live in a small LSM tree, while values live in a separate append-only sharded value log. This dramatically reduces compaction write-amplification — the LSM tree stays small and cheap to rewrite, and large values are only touched during dedicated GC passes.

Key design choices:

| Property | Detail |
|---|---|
| Key storage | Single skip-list memtable → sharded L0/L1 SSTables (per-bucket LSM) |
| Value storage | Sharded append-only value log (configurable page size, 64 MiB default) |
| Durability | Write-ahead log (WAL) with configurable fsync cadence |
| Concurrency | A reader/writer lock per bucket, plus lock-free reclamation so readers never block a cleanup pass |
| Serialization | Compact binary encoding for typed values, with a checksum on every record |
| TTL | Native per-record expiry tracked in the value log |
| Namespaces | Logical isolation within a single DB; each namespace has its own LSM shards and value-log shards |
| Background workers | LSM compaction, value-log GC, WAL cleanup, TTL eviction — each on its own async task |
| Hashing | Fast, SIMD-accelerated hashing for bucket assignment |
| Prefix scan | SIMD-accelerated prefix scan across all layers — see below |

The WAL is written before every mutation. On startup the engine replays any WAL segments that postdate the last LSM flush, ensuring no committed write is lost after a crash.

#### Prefix scan with SIMD acceleration

`scan_prefix` returns all live key-value pairs whose keys start with a given byte prefix. It merges results across all storage layers — active memtable, read-only memtable, L0 SSTables, and the L1 SSTable — deduplicating by key and honouring tombstones. The full API is available on `Db`, `Namespace`, `AsyncDb`, and `AsyncNamespace`, including a zero-copy typed variant for callers who store typed values.

Three layers of acceleration keep prefix scans fast even as data accumulates across layers:

**1 — A stored key prefix for instant comparison**

Every skip-list node and every SSTable record stores the first 8 bytes of its key as a single integer. For prefixes up to 8 bytes, matching reduces to one integer comparison — no byte-by-byte string walk needed. Only prefixes longer than 8 bytes fall back to a direct byte comparison.

**2 — File-level bounds pruning**

Each SSTable file records the smallest and largest key prefix it contains. Before scanning a file, the engine checks whether the search prefix can possibly fall within those bounds. Files whose range excludes the target are skipped entirely, with no I/O.

**3 — SIMD key comparison in skip-list traversal**

When two keys share the same 8-byte prefix, the skip list falls back to a full byte-by-byte comparison to resolve the order — and that comparison itself is SIMD-accelerated on x86_64 (processing many bytes per CPU cycle instead of one). On Apple Silicon this particular comparison currently runs as a plain, non-SIMD loop; it's still correct, just not vectorised yet. (SIMD acceleration elsewhere in the engine, such as the field-index bitmap operations and vector search, does cover Apple Silicon — see [SIMD coverage by architecture](minnal_db/README.md#simd-coverage-by-architecture) for the full breakdown.)

The SIMD paths are also used for all ordered key lookups and range scans in the skip list, not just prefix scans, so the benefit extends across all read paths.

### Layer 2 — Field Indexing (RoaringBitmap, part of `kv-store`)

The KV engine can find a document by its primary key, but answering a question like "which documents have `status = active` and `age >= 18`?" needs a secondary index. Field indexing is built into the engine (the folded `index` module): fast predicate queries over document fields.

Each indexed field maintains a `FieldIndex` — a persistent hash table that maps field values to `RoaringBitmap` sets of document row IDs. The bitmaps live in memory-mapped files, so the OS page cache handles warm and cold access naturally, and the bitwise `AND`/`OR`/`NOT` operations that combine them are SIMD-accelerated on both x86_64 and Apple Silicon. A query string such as `status = "active" AND age >= 18` is turned into an AST by a small lexer and parser, then evaluated against the live indices. Three value types are supported — `str`, `int`, and `bool`.

Index writes are checkpointed against WAL offsets, so a crash partway through a rebuild resumes exactly where it left off rather than starting over.

For the full design — how field attributes are created, updated, stored, and recovered after a crash — see [`Index-Architecture.md`](minnal_db/src/index/Index-Architecture.md).

### Layer 3 — Semantic Search (Vector Quantisation + ANN, `semantic-search`)

Field indices answer *exact* questions. Semantic search answers *fuzzy* ones — "find the records that mean something similar to this query" — by comparing embedding vectors rather than literal values. It applies to any text-valued store: the indexed embedding fields of a JSON document store, or the string values of a `value_type = str` KV store (both expose a `semantic-search` endpoint).

The `semantic-search` feature implements this as IVF (Inverted File Index) approximate nearest-neighbour search with **RaBitQ** quantisation and a **two-pass** sparse→dense search algorithm. The subsections below build it up step by step: how the embedding space is partitioned, how values are quantised at write time, and how a query is resolved.

#### IVF Clustering

The embedding space is partitioned into a fixed number of clusters. The cluster centroids are pre-computed offline (e.g. with k-means over a representative corpus) and stored as a JSONL file (one `{cluster_id, centroid}` per line). At startup, minnal loads the centroids once into a read-only index shared across all requests.

This partitioning is what makes Pass 1 cheap: every document's sparse chunks are stored keyed by their assigned `cluster_id`, so a query only has to scan the chunks in a handful of relevant clusters instead of the whole corpus. At query time the relevant clusters are chosen by an **exact** nearest-centroid scan — each query embedding's distance to every centroid is computed and the `n_probes` nearest are selected. There is deliberately **no precomputed neighbour graph over the clusters**: at the cluster counts in use, this exhaustive scan takes microseconds and yields the *exact* nearest clusters, whereas an approximate graph traversal would trade away recall for no meaningful speed-up. (A neighbour graph would only pay off at much larger, sparser cluster counts.)

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
5. Candidates are scored and aggregated via **SimMax** (max score per `doc_id` across all clusters × all query chunks).
6. The top `first_pass_sparse_search_top_k` candidates are retained.

**Pass 2 — Dense (MultiBit):**
1. MultiBit entries for all sparse candidates are fetched directly from `{ns}_dense_vector` by `doc_id` in parallel — a direct lookup per candidate, no cluster scan.
2. Candidates are grouped by `cluster_id` and scored in parallel.
3. The top-K results are returned, sorted by estimated dot product (higher = more similar).

The attribute-predicate filter is applied **only in Pass 1**. Pass 2 operates on the already-filtered candidate list.

#### RaBitQ Quantisation

RaBitQ encodes a floating-point vector as a compact bit string plus scalar correction factors (`addition_factor`, `scaling_factor`) and a per-vector `error_bound`. The error bound guarantees that the estimated dot product deviates from the true dot product by at most `error_bound` with high probability.

Accuracy is validated in the test suite: at 8 bits, the relative error against the exact dot product (computed on full-precision f32 vectors with simsimd) is consistently below 0.1%.

#### Embedding Service

The embedding service is an external HTTP endpoint. Chunking/tokenisation happens in minnal (`semantic_search::chunking`); the service receives a batch of pre-prepared payload strings (one per chunk) and returns a parallel array of embeddings — one embedding per payload, in the same order:

```
POST {base_url}/embedding/document   {"payloads": [str, ...], "dimensions": N}  →  {"embeddings": [[f32], ...]}
POST {base_url}/embedding/query      (same request/response shape)
GET  {base_url}/healthcheck
```

#### Adding a New Embedding Model

The set of models is **data-driven** — no code change or recompile is required. A model is defined entirely by (a) a cluster-centroid file on disk and (b) a config entry that names it. Adding one (assuming an external embedding service that serves it) is:

1. **Generate the cluster centroids offline.** Run k-means (or any IVF clustering) over a representative corpus *embedded with the new model*. The number of centroids is the IVF cluster count (more clusters → finer partitioning, the trade-off knob against `n_probes`). Every centroid vector must have the model's embedding dimension.

2. **Write them as a JSONL file** — one `{cluster_id, centroid}` object per line, each centroid dimensioned to match the model. For the exact file format and the validation rules enforced at startup, see [`Semantic-Search-Architecture.md` §3 — IVF Index Structure](minnal_db/src/semantic_search/Semantic-Search-Architecture.md#3-ivf-index-structure).

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

For the full end-to-end design — embedding generation, dual quantisation, index structure, two-pass query execution, storage layout, crash recovery, and hybrid search — see [`Semantic-Search-Architecture.md`](minnal_db/src/semantic_search/Semantic-Search-Architecture.md).

### Layer 4 — Document Store + KV Store (`doc-store`)

The engine, field indexing, and vector search are independent capabilities — one stores bytes, one indexes fields, one searches vectors. The `doc-store` feature is where they come together: it presents them as a coherent document model, exposing two distinct store types behind a unified `DocStore` handle.

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

**Reconciliation.** Because the document write and its embed enqueue are two separate steps, a crash between them can leave a document that never gets indexed. A reconciliation pass finds and re-enqueues any document that's missing its vector index, self-healing the gap automatically — it runs on every startup, and can also be triggered on demand via `POST /admin/indices/vector/reconcile`. For the exact crash windows it closes, see [`Semantic-Search-Architecture.md`](minnal_db/src/semantic_search/Semantic-Search-Architecture.md#forward-reconciliation-startup--on-demand).

### Layer 5 — REST API (`minnal_db_api`)

An [Axum](https://github.com/tokio-rs/axum) HTTP server (the `minnal_db_api` binary crate) that wraps `minnal_db` (with `doc-store` + `semantic-search`) in a full REST interface. At startup it loads every store's schema and the IVF cluster index into memory, so requests can validate and route without touching disk first. It then serves all store, document, index, and semantic-search endpoints, tracking which namespaces have an index build or cleanup in progress so concurrent admin operations on the same namespace are rejected rather than racing each other.

---

## Salient Features

The layered architecture above produces a specific set of capabilities. The table below summarises what minnal offers and where each feature comes from in the stack.

| Feature | Detail |
|---|---|
| **WiscKey-style KV engine** | Keys and values stored separately. LSM tree stays small; large values avoid compaction churn. |
| **Multi-namespace isolation** | Each logical store has independent LSM, value-log, and index shards. |
| **Crash-safe WAL** | All mutations go through a WAL before they reach the memtable. Replay on startup. |
| **Background GC** | Value-log GC reclaims space from deleted/overwritten entries. Tunable waste threshold. |
| **SIMD-accelerated prefix scan** | `scan_prefix` merges across all storage layers, using a stored key fingerprint and file-level bounds to skip work before falling back to a SIMD-accelerated byte comparison. See [Prefix scan with SIMD acceleration](#prefix-scan-with-simd-acceleration) above. |
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

## Quickstart & Usage

The hands-on walkthrough — building the binaries, running the server, bulk-loading
sample data, and every REST and embedded example — lives in a dedicated guide:

**→ [Quickstart & Usage Guide](QUICKSTART.md)**

It is organised by how you run minnal and which store type you use:

- **[Getting Started](QUICKSTART.md#getting-started-bulk-load-a-store-and-query-it)** — the fastest end-to-end path: stage the release, start the server, bulk-load a bundled sample dataset, and query it.
- **[Using minnal as a Service (REST)](QUICKSTART.md#using-minnal-as-a-service-rest)** — start the server, then work with [document and KV stores](QUICKSTART.md#document-and-kv-stores-rest-api) (CRUD, predicate queries, semantic search — full endpoint reference in [`minnal_db_api`](minnal_db_api/README.md)), plus [bulk loading](QUICKSTART.md#bulk-loading), [configuration](QUICKSTART.md#configuration), [admin/monitoring](QUICKSTART.md#admin-and-monitoring), [logging](QUICKSTART.md#logging), and [durability & recovery](QUICKSTART.md#write-durability-and-recovery).
- **[Embedded Quickstart](minnal_db/QUICKSTART.md)** — add `minnal_db` to a Rust process for in-process storage; feature selection (`kv-store`/`doc-store`/`semantic-search`) and code examples.
- **[Scripts & Config](QUICKSTART.md#scripts-and-config)** — the `release.sh`/`start.sh`/`run_tool.sh` helpers, the bundled `curl` example scripts, cluster centroids, and the annotated sample config.

---

## Repository Structure

The workspace is a **single publishable library crate** (`minnal_db`, with the
former `index`, `semantic_search`, and `minnal_doc_store` crates folded in as
feature-gated modules) plus two binary crates.

```
minnal/
├── minnal_db/              ← the library crate (all layers, feature-gated)
│   ├── QUICKSTART.md       ← embedded quickstart + feature selection
│   ├── README.md           ← detailed KV engine documentation
│   └── src/
│       ├── db/ store/      ← LSM + value-log KV engine
│       ├── index/          ← RoaringBitmap field indexing (part of kv-store)
│       ├── semantic_search/← IVF clustering, RaBitQ quantisation (semantic-search)
│       ├── vector_kv.rs    ← vector↔KV bridge, usable on raw namespaces
│       └── doc_store/      ← JSON document store (doc-store)
├── minnal_db_api/          ← Axum REST API server (binary)
│   └── README.md           ← full REST API reference with curl examples
├── minnal_tools/            ← minnal_tools binary (bulk_load and future tools)
├── service/
│   ├── scripts/            ← release.sh / build_docker.sh / curl examples
│   ├── docker/Dockerfile
│   └── embedding_support/qwen/clusters.json   ← pre-computed IVF centroids (JSONL)
├── config/sample.toml      ← annotated reference configuration
└── work/bin/               ← generated by release.sh (not committed)
    ├── minnal_db_api        ├── minnal.toml    ├── start.sh    └── run_tool.sh
```

Feature selection (`kv-store` default, `doc-store`, `semantic-search`) and the
embedded API are covered in [`minnal_db/QUICKSTART.md`](minnal_db/QUICKSTART.md);
the detailed engine internals in [`minnal_db/README.md`](minnal_db/README.md).
For the full REST API reference — endpoints, error codes, predicate syntax, and
on-disk layout — see [`minnal_db_api/README.md`](minnal_db_api/README.md).

---

## Acknowledgements

### WiscKey

The storage engine at the core of minnal is directly inspired by the WiscKey paper. The key insight — storing only keys in the LSM tree and large values in a separate append-only log — is the foundational design decision of `minnal_db`.

> **WiscKey: Separating Keys from Values in SSD-Conscious Storage**
> Lanyue Lu, Thanumalayan Sankaranarayana Pillai, Andrea C. Arpaci-Dusseau, Remzi H. Arpaci-Dusseau
> *USENIX FAST '16*
> https://www.usenix.org/conference/fast16/technical-sessions/presentation/lu

### RaBitQ

The quantisation scheme used in the `semantic-search` layer is RaBitQ, which provides a tight theoretical error bound on the estimated inner product between a quantised document vector and a full-precision query vector. This bound is what populates the `error_bound` field in semantic search responses.

> **RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search**
> Jianyang Gao, Cheng Long
> *ACM SIGMOD '24*
> https://dl.acm.org/doi/10.1145/3654970

The reference C++ implementation of RaBitQ, which informed the multi-bit quantisation and distance estimation code in this project, is available at:

> https://github.com/gaoj0017/RaBitQ

### RoaringBitmap

The field indexing layer (the folded `index` module) uses Roaring Bitmaps as its compressed bitmap representation. Roaring Bitmaps partition a 32-bit integer space into 65 536 chunks and choose the most space-efficient container type (array, bitset, or run-length encoded) per chunk, giving excellent compression on both sparse and dense sets while keeping set operations fast.

> **Roaring Bitmaps: Implementation of an Optimized Software Library**
> Daniel Lemire, Owen Kaser, Nathan Kurz, Luca Deri, Chris O'Hern, François Saint-Jacques, Gregory Ssi-Yan-Kai
> *Software: Practice and Experience, 2018*
> https://arxiv.org/abs/1709.07821

The canonical Java reference implementation, which established the on-disk format and container selection heuristics, is available at:

> https://github.com/RoaringBitmap/RoaringBitmap
