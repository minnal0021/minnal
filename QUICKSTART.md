# minnal — Quickstart & Usage Guide

A practical walkthrough of every way to use minnal: build the binaries, then
exercise each part of the API. For what minnal *is* — the architecture, the
five layers, and the design of the storage engine and semantic search — see the
[main README](README.md).

Minnal can be used along **two axes** — how you run it (embedded in-process, or
as a REST service) and which **store type** you talk to (a JSON *document store*,
or a schema-lite *KV store*):

| Mode | Document store | KV store |
|---|---|---|
| **Embedded** — link `minnal_db` directly into a Rust process | ❌ Not available — the document model is a higher layer | ✅ Key-value CRUD, TTL, typed values, **RoaringBitmap field index + predicate queries** |
| **Service (REST)** — run `minnal_db_api` | ✅ Full: CRUD, predicate queries, **semantic search** | ✅ CRUD, range/prefix scans, optional **semantic search** |

The two modes share the same engine — everything below the REST boundary is the
same code whether you call it in-process or over HTTP. The rest of this guide is
organised by mode: the [fastest end-to-end path](#getting-started-bulk-load-a-store-and-query-it)
first, then the full [service](#using-minnal-as-a-service-rest) walkthrough, then
[embedded](#using-minnal-embedded-as-a-library) use.

## Table of Contents

- [Build](#build)
- [Getting Started: bulk-load a store and query it](#getting-started-bulk-load-a-store-and-query-it)
- [Using minnal as a Service (REST)](#using-minnal-as-a-service-rest)
  - [Start the Server](#start-the-server)
  - [Configuration](#configuration)
  - [Document Stores](#document-stores)
    - [Store Lifecycle](#store-lifecycle)
    - [Document CRUD](#document-crud)
    - [Predicate Queries](#predicate-queries)
    - [Semantic Search](#semantic-search)
    - [Index Management](#index-management)
  - [KV Stores](#kv-stores)
  - [Bulk Loading](#bulk-loading)
  - [Admin and Monitoring](#admin-and-monitoring)
  - [Logging](#logging)
  - [Write Durability and Recovery](#write-durability-and-recovery)
- [Using minnal Embedded (as a Library)](#using-minnal-embedded-as-a-library)
  - [What an embedded store can and cannot do](#what-an-embedded-store-can-and-cannot-do)
  - [Field-Level Indexing](#field-level-indexing)
- [Scripts and Config](#scripts-and-config)
  - [Server scripts](#server-scripts)
  - [Example scripts](#example-scripts)
  - [Cluster centroids](#cluster-centroids)
  - [Sample config](#sample-config)

---

## Build

```bash
# Build all crates (debug)
cargo build

# Build optimised binaries
cargo build --release -p minnal_db_api
cargo build --release -p minnal_tools     # minnal_tools (bulk_load, …)
```

> **Semantic search needs an external embedding service.** Everything else —
> key-value and document storage, field indexing, and predicate queries — runs
> entirely in-process with no external dependency. But **semantic search** relies
> on an embedding service to turn text into vectors, both when documents are
> indexed and when a query is run. If you plan to use semantic search, set up and
> run one first. The companion
> [minnal0021/embedding_service](https://github.com/minnal0021/embedding_service)
> serves the **gemma** model over HTTP — see its
> [README](https://github.com/minnal0021/embedding_service#readme) to get started,
> then point `semantic_search.embedding_service_url` at it (default
> `http://localhost:8001`).

---

## Getting Started: bulk-load a store and query it

This is the fastest end-to-end path: stage the release, start the server, load a
small bundled sample dataset, and query it — all from the workspace root.

A document store is described by a **schema**: a key type, zero or more attribute
**indices** (RoaringBitmap field indexes you can filter on), and an optional set
of **embedding fields** for semantic search. The schema below is deliberately
minimal — one index field (`agency`) and one semantic-search field (`jobTitle`).

**1. Build and stage the release (with sample data), then start the server**

```bash
# Build optimised binaries, config, AND the sample data (-s) under ./work/
./service/scripts/release.sh -s

# Start the server (listens on :8080)
./work/bin/start.sh
```

The `-s` flag stages [`minnal_tools/sample_data/`](minnal_tools/sample_data) into
`./work/sample_data/`, including the two files this quickstart uses:
`jobs-mini-schema.json` and `jobs-mini.jsonl` (ten rows). It also stages a
KV-store sample — `job-content-kv-schema.json` and `job-content-kv.jsonl` (twenty
long job descriptions) — used by the [KV Stores](#kv-stores) bulk-load example.

**2. Look at the schema**

`jobs-mini-schema.json` is deliberately minimal — `u64` keys, one attribute index
(`agency`), and one semantic-search field (`jobTitle`):

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

With `--schema`, the `bulk_load` tool POSTs the schema to
`POST /admin/stores/import` (creating the `jobs` store) and then streams the JSONL
in, using `jobId` as the `u64` document key. Re-running is safe — an existing
store is reused.

```bash
./work/bin/run_tool.sh bulk_load --schema ./work/sample_data/jobs-mini-schema.json \
  http://localhost:8080 jobs jobId ./work/sample_data/jobs-mini.jsonl
# → schema imported — store 'jobs' created
# → done  loaded=10  skipped=0  total=10  elapsed=…s
```

**4. Query the store**

Filter on the indexed `agency` field. Attribute indexing is synchronous with the
write, so the rows are queryable immediately:

```bash
curl -s -X POST http://localhost:8080/stores/jobs/query \
  -H 'Content-Type: application/json' \
  -d '{"predicate": "agency = \"National Library Board\""}'
# → {"results":[{"id":13354909,"doc":{...}},
#               {"id":13354910,"doc":{...}},
#               {"id":15120777,"doc":{...}}],
#    "total":3,"page_no":1,"page_size":20}
```

> **Semantic search needs an external embedding service.** The schema above sets
> `semantic_search_enabled: true`, but the actual embeddings are produced
> asynchronously by an embedding service (default `http://localhost:8001`).
> **Without it, the load and all attribute queries above still work** — only
> `POST /stores/jobs/semantic-search` returns nothing until embeddings exist. See
> [Semantic Search](#semantic-search) below to enable it.

For the full `bulk_load` reference (KV stores, existing namespaces, `--no-wal`),
see [Bulk Loading](#bulk-loading).

---

## Using minnal as a Service (REST)

The examples below assume the server is running locally on its default port
(`8080`); every request is self-contained, so you can copy any block and run it
directly.

> **Prefer a GUI?** [minnal0021/minnal_ui](https://github.com/minnal0021/minnal_ui)
> is a companion single-page web app you can run alongside the REST server and
> point at its address for a rich graphical experience — browsing stores, editing
> documents, and running queries and semantic search without hand-writing `curl`.
> See the [minnal_ui README](https://github.com/minnal0021/minnal_ui#readme) for
> its quickstart.

### Start the Server

#### Release workflow (recommended)

Use `release.sh` to build optimised binaries and stage everything under
`./work/bin/`:

```bash
# Build and stage release binaries + config
./service/scripts/release.sh

# Start the server
./work/bin/start.sh
```

`release.sh` generates `./work/bin/minnal.toml` from `config/sample.toml` with all
data paths rewritten to `./work/doc_store/` as the base.  Run both commands from
the workspace root.

#### Development workflow

```bash
# Debug build and run directly
cargo run -p minnal_db_api -- config/sample.toml
```

The server listens on `0.0.0.0:8080` by default (configurable via
`[api] listen_addr`). To stop it, send SIGINT (`Ctrl-C`) or SIGTERM.

### Configuration

See [`config/sample.toml`](config/sample.toml) for the full annotated reference.
The most important sections:

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

> **Note:** `window_size` and `sliding_size` control how text is split into chunks
> for the sparse (Pass 1) embeddings, and the *same* values are used to chunk both
> documents at index time and queries at search time. Changing them after
> documents have been indexed makes stored chunks and new query chunks
> inconsistent, which silently degrades recall (no error is raised). Treat them as
> a fixed indexing decision: if you change either value, re-embed the corpus with
> `POST /admin/indices/{ns}/vector/reindex-all`.

The config file is located by (in order): first CLI argument → `MINNAL_CONFIG_FILE`
env var → built-in defaults.

### Document Stores

A document store holds JSON documents keyed by `uuid`, `u64`, or `u128`, with up
to five typed field indices and optional semantic search.

#### Store Lifecycle

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

Schema amendment operations: `add_attribute`, `update_attribute`,
`remove_attribute`. Indexed fields cannot be amended directly — drop the index
first.

#### Document CRUD

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

#### Predicate Queries

Only indexed fields may appear in predicates. Operators: `=`, `!=`, `<`, `<=`,
`>`, `>=`, `AND`, `OR`, `NOT`.

```bash
curl -s -X POST http://localhost:8080/stores/profiles/query \
  -H 'Content-Type: application/json' \
  -d '{"predicate": "status = \"active\" AND seniority = \"senior\""}'
# → [{"id":"550e8400-...","doc":{...}}, ...]
```

#### Semantic Search

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

#### Index Management

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

### KV Stores

A KV store is a schema-lite namespace for raw key-value data. It has no field
indices but supports the same WAL durability, LSM compaction, value-log GC, and
optional semantic search as document stores.

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
cargo run -p minnal_tools -- bulk_load http://localhost:8080 profiles id profiles.jsonl
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

See [Getting Started](#getting-started-bulk-load-a-store-and-query-it) for a
complete `bulk_load` walkthrough with a sample schema and rows.

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

For the full admin API reference see
[`minnal_db_api/README.md`](minnal_db_api/README.md). For a
consolidated table of every metric these endpoints report — with explanations and
whether each value survives a restart — see
[Operational & Storage Metrics](minnal_db_api/README.md#operational--storage-metrics).

### Logging

Minnal uses the [`tracing`](https://docs.rs/tracing) crate throughout. Log
verbosity can be set either in the TOML config file or via the `RUST_LOG`
environment variable. `RUST_LOG` always takes precedence over the config file
setting.

#### TOML configuration

The `[logging]` section controls the default log level and the `[storage]` section
controls where rolling log files are written:

```toml
[storage]
# Directory where rolling log files are written.
log_dir = "./data/log"

[logging]
# Minimum log level when RUST_LOG is not set.
# Accepted values: "error", "warn", "info", "debug", "trace"
level = "info"
```

`log_dir` is relative to the working directory (the workspace root when launched
via `./work/bin/start.sh` or `cargo run`).

#### RUST_LOG environment variable

The `env-filter` feature lets you target specific crates or modules, overriding
the `[logging] level` value entirely:

```bash
# Warnings and above only (quiet)
RUST_LOG=warn ./target/release/minnal_db_api config/sample.toml

# Info level (default recommended)
RUST_LOG=info ./target/release/minnal_db_api config/sample.toml

# Debug messages
RUST_LOG=debug ./target/release/minnal_db_api config/sample.toml

# Per-crate control — debug for the KV engine, info elsewhere
RUST_LOG=info,minnal_db=debug ./target/release/minnal_db_api config/sample.toml

# Narrow to a single module
RUST_LOG=minnal_db::db::database=trace ./target/release/minnal_db_api config/sample.toml
```

Level hierarchy (most → least verbose): `trace > debug > info > warn > error`.

The same variable works with the staged binary and `cargo run`:

```bash
RUST_LOG=info ./work/bin/start.sh
RUST_LOG=info cargo run -p minnal_db_api -- config/sample.toml
```

### Write Durability and Recovery

All mutating requests (document and KV store writes and deletes) follow a
WAL-first durability model.

#### Request lifecycle

1. The write is appended to the Write-Ahead Log and fsynced to disk.
2. If WAL persistence fails, the request returns **500 Internal Server Error**. No data is lost because nothing was committed.
3. Once the WAL entry is durable, the write is applied to the in-memory structures (value log + memtable). The in-memory apply is best-effort: if it fails, the error is logged at `ERROR` level but the request still returns **204 No Content**. The data is safe in the WAL and will be recovered on the next startup.
4. All SSTable flushes, LSM compaction, value-log GC, and WAL GC happen asynchronously in background workers and do not affect the request latency or response.

In short: a successful response (204) means the write is durable in the WAL. It
does not mean it has been flushed to an SSTable.

#### WAL recovery on startup

When the server starts, it replays any WAL entries that have not yet been flushed
to SSTables:

1. Each entry is applied to the appropriate in-memory store.
2. If an entry fails to apply, it is **retried once**.
3. If the retry also fails, the failed operation is written to a **fail log file** and skipped.

#### Fail log files

Fail logs are written to the directory configured by `recovery.fail_log_dir`
(defaults to `<db_path>/fail_logs`). A separate file is created for each recovery
run, named:

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

A user or operator can inspect these files and take corrective action — replay the
missing writes via the REST API, delete the affected keys, or ignore them if the
data is no longer needed.

---

## Using minnal Embedded (as a Library)

To embed minnal in a Rust process, add the `minnal_db` crate and call it
in-process — no server, no daemon. Capabilities are selected by cargo feature
(`kv-store` default, `doc-store`, `semantic-search`), so you compile only what
you use and the lean default pulls no vector dependencies.

See the dedicated **[Embedded Quickstart](minnal_db/QUICKSTART.md)** — it covers
feature selection as a table, the key-value and field-index APIs with runnable
examples, and the document-store handle. Full engine internals are in
[`minnal_db/README.md`](minnal_db/README.md).

## Scripts and Config

### Server scripts

| Script | Purpose |
|---|---|
| [`service/scripts/release.sh`](service/scripts/release.sh) | Build release binaries and stage them under `./work/bin/`. Generates `minnal.toml` (paths rewritten for `./work/doc_store/`), `start.sh`, and `run_tool.sh` in the same directory. |
| [`service/scripts/build_docker.sh`](service/scripts/build_docker.sh) | Build the Docker image from the workspace root. Accepts an optional image tag argument. |

**Generated by `release.sh` into `./work/bin/`:**

| Script | Purpose |
|---|---|
| `./work/bin/start.sh` | Start `minnal_db_api` using the staged binary and `minnal.toml`. |
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
