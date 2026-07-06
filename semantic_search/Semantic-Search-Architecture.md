# Semantic Search Architecture

This document describes how semantic search works end-to-end: embedding generation, dual quantisation, index structure, two-pass query execution, storage layout, crash recovery, and hybrid search.

---

## 1. Embeddings

### Overview

Minnal does **not** generate embeddings itself. It relies on an **external embedding service** and treats it as the sole source of vectors: minnal prepares payloads, the service returns embeddings, and minnal quantises and indexes them as-is.

The service is reached over **HTTP**. Its base URL is configured under `[semantic_search]` in the TOML config (`embedding_service_url`) and defaults to `http://localhost:8001`. Minnal does not negotiate or version models with the service — every request goes to fixed endpoints with no model identifier in the URL or body. Choosing the concrete model, and pinning its exact version, is entirely the embedding service's responsibility, decided server-side and applied uniformly to every request.

The `model` name in the config (e.g. `qwen`) is therefore *not* used to select or version a model at request time. It is only an indication of which *family* of model the deployment is built around, and within minnal it serves a single purpose: selecting the matching cluster-centroid file and embedding dimension for this instance (validated at startup against `[[semantic_search.supported_models]]`). Keeping the config name aligned with whatever the embedding service actually serves is an operational convention, not something minnal enforces against the service.

### Service interface

The service exposes a **batch interface** — chunking/tokenisation happens in minnal (`chunking/mod.rs`), not in the service. A whole-text embedding is just a one-element `payloads` array; chunked (sliding-window) embeddings send one payload per chunk.

Two `POST` endpoints, identical in request/response shape, differing only in which side of the asymmetric model they target (documents are embedded differently from queries):

| Method & path | Purpose |
|---|---|
| `POST {base_url}/embedding/document` | Embed document payloads (indexing). |
| `POST {base_url}/embedding/query` | Embed query payloads (search). |

**Request body** (`application/json`):

```json
{
  "payloads": ["first text to embed", "second text", "..."],
  "dimensions": 768
}
```

| Parameter | Type | Meaning |
|---|---|---|
| `payloads` | array of strings | One entry per text to embed. Order is significant — the response must preserve it. |
| `dimensions` | unsigned integer | The embedding dimensionality minnal expects back, taken from `embedding_dim`. Every returned vector must have exactly this length. |

**Response body** (`application/json`):

```json
{
  "embeddings": [[0.0123, -0.0456, ...], [0.0789, 0.0011, ...]]
}
```

| Field | Type | Meaning |
|---|---|---|
| `embeddings` | array of `f32` arrays | One vector per input payload, in the **same order** as `payloads`. Each vector must be L2-normalised and of length `dimensions`. |

Minnal validates the response and errors if the number of returned embeddings differs from the number of payloads sent (`CountMismatch`), or if any vector's length differs from the requested `dimensions` (`DimensionMismatch`). An empty `payloads` array short-circuits to an empty result with no HTTP request. Extra fields in the response JSON are ignored.

A health endpoint is also expected: `GET {base_url}/healthcheck` should return a 2xx status. Minnal probes this at startup and logs a warning if the service is unreachable (startup is non-fatal — failures surface at query time).

### Embedding dimension

The expected embedding size is configurable via `embedding_dim` under `[semantic_search]` and **defaults to 768**. The bundled cluster-centroid file (`service/embedding_support/qwen/clusters.json`) contains **768-dimensional** centroids, so the default works out of the box.

> ⚠️ **Changing `embedding_dim` requires regenerating the cluster-centroid file.** The centroids in `clusters.json` must have the *same dimensionality* as the embeddings the service returns — IVF cluster assignment computes Euclidean distance between an embedding and each centroid, which is only defined for equal-length vectors. If you point minnal at a service that produces a different embedding size, you **must** also supply a matching `clusters.json` (see [§3](#3-ivf-index-structure)) whose centroids have that dimensionality, and re-index the corpus. The default 768-dimensional centroids shipped in `clusters.json` are only valid for 768-dimensional embeddings.

### Normalisation

Every embedding returned by the service — both the **dense** whole-text vectors and the **sparse** per-chunk vectors — is expected to be **L2-normalised** (unit length) by the service. Minnal relies on this: downstream quantisation and the dot-product distance estimators assume unit vectors, so dot products can be treated as cosine similarity. Minnal does **not** re-normalise the embeddings it receives; producing unit-length vectors is the embedding service's responsibility.

---

## 2. Dual Quantisation (RaBitQ)

Before storage, embeddings are compressed using **RaBitQ** (`quantisation/rabitq/mod.rs`). At index time, a **single embedding call** is made per document — one ordered payload list `[whole_text, chunk₀, chunk₁, …]` in one round trip — and the order-preserving response is split into two classes of quantised entry:

### MultiBit (dense, whole-document)

1. A single embedding is obtained for the entire document text.
2. The nearest cluster centroid is found by Euclidean L2.
3. The residual `embedding − centroid` is quantised to N bits per dimension (`number_of_bits_for_dense_quantisation`, default 8).

The result is stored in the **dense namespace** (`{ns}_dense_vector`) keyed directly by `doc_id`. For 768-dimensional embeddings at 8 bits, each entry compresses from ~3 KB (f32) to ~768 bytes — a 4× reduction.

### SingleBit (sparse, sliding-window chunks)

1. The document text is split into overlapping chunks using a sliding window (`window_size`, `sliding_size`).
2. The embedding service returns one embedding per chunk.
3. Each chunk embedding is independently assigned to its nearest IVF cluster and quantised to 1 bit per dimension.

The result is stored in the **sparse namespace** (`{ns}_sparse_vector`) under composite keys `[cluster_id (4B BE) ‖ doc_id]`. For 768-dimensional embeddings, a SingleBit entry is ~96 bytes — a 32× reduction versus f32, enabling fast first-pass scans over large cluster partitions.

A third companion namespace, **`{ns}_sparse_vector_meta`**, records which clusters each document's chunks were assigned to. This is used only by delete and upsert operations to clean up stale composite keys; it is never consulted during search.

### The `VectorIndex` struct

All quantised entries share the same `VectorIndex` struct:

| Field | Purpose |
|---|---|
| `cluster_id` | The IVF cluster this entry belongs to |
| `binary_quantised_vector` | Packed bit codes (96 bytes SingleBit, ~768 bytes 8-bit MultiBit) |
| `addition_factor` | Scalar correction coefficient for distance estimation |
| `scaling_factor` | Scalar correction coefficient for distance estimation |
| `error_bound` | Theoretical max deviation of estimated dot product from true dot product |
| `quantisation_style` | `SingleBit` or `MultiBit { number_of_bits }` |

---

## 3. IVF Index Structure

The index is **Inverted File (IVF) with flat scanning** (`cluster/mod.rs`):

- A `ClusterIndex` maps `cluster_id → centroid (Vec<f32>)`.
- **Cluster probing is exact and exhaustive.** For each query embedding, `find_top_n_cluster_ids` computes the Euclidean distance to **every** centroid and returns the `n_probes` nearest (using `select_nth_unstable` to select the top-`n_probes` in ~O(C) rather than a full O(C log C) sort). The union of these sets across all query embeddings is scanned once.
- **There is deliberately no neighbour graph / approximate cluster traversal.** Coarse-assignment cost is **T·C·D** (query chunks × centroids × dim) — the scan runs once per query chunk. At C≈256 this is genuinely cheap: ~50 µs at 4 chunks, ~1.1 ms at 100 chunks (`bench_distance_estimation` → `coarse_assignment`). It grows *linearly with query length*, so it is not entirely free for long, many-chunk queries on a warm query-embedding cache (where the embedding round-trip no longer hides it), but it is a small fraction of a query at realistic lengths. A neighbour graph is the wrong lever regardless: it is approximate on the most recall-sensitive stage (missing the right coarse cluster drops the document entirely), barely reduces work at a few hundred nodes, and does nothing about the per-chunk (T) factor. The exact optimisations, measured:
  - **Contiguous centroid matrix** (streamed by `find_top_n_cluster_ids_batch`) instead of a `HashMap` of per-`Cluster` `Vec`s — ~12% faster, no recall change. **Done.**
  - **Parallelising the per-chunk scans (rayon)** — *tried and reverted*: ~2.4× **slower** at 100 chunks, because each per-chunk scan is only microseconds and the thread-pool dispatch dwarfs the work.
  - A true **blocked GEMM** (reuse each centroid tile across query chunks) is the only remaining lever, and only pays at far larger C. Revisit it — and graph-over-centroids — only if C reaches the *tens of thousands* (orders-of-magnitude corpus growth forcing many more centroids), behind a config flag with recall/latency benchmarking.

Clusters are loaded from the file at `cluster_path` (`clusters.json`) at startup. The file is **JSONL** — one JSON object per line, each describing a single cluster centroid with exactly two attributes:

| Attribute | Type | Meaning |
|---|---|---|
| `cluster_id` | unsigned integer (`u32`) | Stable identifier for the cluster; used as the 4-byte big-endian key prefix in `{ns}_sparse_vector` and to look up centroids during probing. |
| `centroid` | array of `f32` | The centroid vector. Its length **must** equal the configured embedding dimension. |

```jsonl
{"cluster_id": 0, "centroid": [0.0123, -0.0456, 0.0789, ...]}
{"cluster_id": 1, "centroid": [-0.0210,  0.0337, 0.0042, ...]}
{"cluster_id": 2, "centroid": [0.0500, -0.0011, -0.0623, ...]}
```

Lines are parsed independently (`read_clusters_from_file` in `cluster/mod.rs`): any line missing `cluster_id` is an error, and `centroid` is deserialised directly into a `Vec<f32>`. Extra attributes on a line are ignored, blank lines are skipped, and `cluster_id` values need not be contiguous or sorted.

---

## 4. Two-Pass Search Execution

`service/mod.rs` implements a two-pass ANN search:

```
Raw query text
  → embedding cache lookup (system_qemb_cache TTL namespace)
  → on miss: embed_query — ONE batch POST to {base_url}/embedding/query
       · payload[0]:  whole query → 1 embedding         (Pass 2 dense)
       · payload[1..]: chunk_query → N chunk embeddings  (Pass 1 sparse)
  → cache both together (TTL configurable; default 1 day)

Pass 1 — Sparse (SingleBit), ColBERT MaxSim:
  for each query chunk q_i:
    find top-n_probes cluster IDs by Euclidean distance
  union cluster sets across all query tokens
  scan_sparse_cluster(cluster_id) for each cluster in parallel
  for each (cluster, document d):
    apply doc_filter (RoaringBitmap predicate) — skip if fails
    for each query token q_i:
      score = max_j SingleBitEstimator(q_i, d_j)   ← best chunk for this token
      per_token_max[doc_id][i] = max(per_token_max[doc_id][i], score)
  final sparse score: S(q, d) = Σ_i per_token_max[d][i]  (ColBERT MaxSim)
  sort descending by score, truncate to first_pass_sparse_search_top_k

Pass 2 — Dense (MultiBit):
  get_dense_entry(doc_id) for each sparse candidate in parallel
  for each candidate (rayon parallel):
    look up its centroid via the stored VectorIndex's cluster_id
    score with MultiBitQuanDotProductEstimator against the single
      whole-query dense embedding
  build top-k min-heap
  return sorted descending
```

### Why two passes?

- **SingleBit** entries are very compact and cluster-contiguous in storage, making prefix-scan over a cluster fast. The 1-bit score is coarse but sufficient to narrow the candidate set from millions to ~1000.
- **MultiBit** entries are stored directly by `doc_id` for O(1) lookup, and their higher-precision scores produce final ranked results with low quantisation error.

### ColBERT MaxSim aggregation (Pass 1)

Pass 1 uses **ColBERT MaxSim** to aggregate scores across multiple query tokens and document chunks:

```
S(q, d) = Σ_i  max_j ⟨q_i, d_j⟩
```

where `i` iterates over query tokens and `j` over document chunks. For each query token, the best-matching chunk of the document wins (inner `max`); those per-token bests are then **summed** (outer `Σ`). This means a document that matches many query tokens scores proportionally higher than one that only matches one — even if that one match is equally strong.

Chunks whose cluster is not probed contribute **0** to their query token's term (rather than −∞), so documents are never penalised for having chunks in far-away clusters.

### Dense re-ranking (Pass 2)

Pass 2 scores each candidate against a **single whole-query dense embedding** — symmetric with the document's whole-text dense vector. There is no aggregation across multiple query vectors: each `doc_id` is scored once by one `MultiBitQuanDotProductEstimator`, so each document appears exactly once in the final output.

### Predicate filtering

`doc_filter` is an optional closure `Fn(&[u8]) -> bool` applied **only in Pass 1**, per document, before scoring. Documents that fail the filter are excluded from both passes. Pass 2 operates on the already-filtered `sparse_ranked` list and never re-evaluates the predicate.

This is how the `search_semantic_filtered` REST endpoint works: it resolves the bitmap predicate into a `HashSet<Vec<u8>>` of matching doc IDs and passes a membership check as the `doc_filter` closure.

---

## 5. Storage Layout

Three companion KVStore namespaces per semantic-search-enabled store (`vector_kv.rs`):

| Namespace | Key | Value |
|---|---|---|
| `{ns}_sparse_vector` | `cluster_id (4B BE u32) ‖ doc_id` | rkyv `Vec<VectorIndex>` (SingleBit only) |
| `{ns}_sparse_vector_meta` | `doc_id` | `count (2B BE u16) ‖ [cluster_id (4B BE)]×N` |
| `{ns}_dense_vector` | `doc_id` | rkyv `Vec<VectorIndex>` (MultiBit only) |

`_sparse_vector` uses a composite key with a cluster prefix so a `scan_prefix(cluster_id)` efficiently retrieves all SingleBit vectors in a cluster. `_sparse_vector_meta` is a reverse lookup used only to find and delete stale sparse keys when a document is updated (its chunks may move to different clusters) or deleted. `_dense_vector` is keyed directly by `doc_id` for O(1) fetch in Pass 2.

---

## 6. Query Embedding Cache

Query embeddings are cached in a system-wide TTL namespace `system_qemb_cache` shared across all doc-store namespaces. Keys are raw UTF-8 query strings; values are packed big-endian `f32` vectors. The TTL is **configurable** via `[semantic_search] query_embedding_cache_ttl_secs` and **defaults to 1 day** (86400 s) — once it elapses, stale entries are evicted automatically by the TTL worker. Cache misses fall back to the embedding service transparently.

**Durability — no-WAL on all CRUD.** Every write to this namespace is no-WAL: both populating an entry on a cache miss (`put_no_wal`) and clearing the cache (`delete_no_wal`). Unlike the vector-payload cleanup deletes (§ above), which stay WAL-backed because a lost delete would orphan an index entry, the cache is TTL-bounded and fully regenerable — a dropped populate or a lost clear-delete just produces a future cache miss that re-fetches from the embedding service. Keeping it off the WAL removes a per-populate fsync from the query hot path (the latency motivation for caching in the first place).

---

## 7. Async Write Path & Durability

Vector indexing is **asynchronous and decoupled from document writes**. A document write never contacts the embedding service inline; it durably enqueues work that a background worker drains later. This keeps writes fast and the system resilient to an embedding service that is slow, restarting, or entirely down.

### Write path

1. `put` / `kv_put` writes the document, then extracts the embedding field and enqueues a `(namespace, doc_id, text)` entry in the durable `system_pending_vec_index` KV namespace as a separate, independent write — the document write and the enqueue are not atomic. The document write returns immediately without contacting the embedding service. A crash between the two writes leaves the document un-indexed (reconciled by re-index), never acked-but-lost.
2. `VecIndexWorker` (`minnal_doc_store/src/vec_index_worker.rs`) consumes the queue in the background — see [Background worker](#background-worker).

### Embedding service availability

The embedding service being unreachable is **never fatal**:

- **At startup**, the server probes `GET {base_url}/healthcheck`. On failure it logs an error and **starts anyway**; the failure surfaces later at call time. The background worker starts regardless of the probe result.
- **At query time**, a search that cannot reach the service returns an error to the caller (`EmbeddingFailed`) — no crash, no partial index corruption.
- **At index time**, the worker simply fails the affected queue entries and retries them on later passes (see [Retry & exhaustion](#retry--exhaustion)). Because the queue is durable, no indexing work is lost while the service is down — it drains once the service returns.

### Background worker

`VecIndexWorker` runs as a single background tokio task. On startup it first **drains any surviving queue entries** (crash recovery), then loops, woken by a write `notify` signal or a 30 s fallback poll. Each pass:

- Scans all pending queue entries and **skips exhausted ones** (`retry_count ≥ max_retries`).
- Groups the remainder by namespace and visits them in **round-robin** order so no single namespace can starve others.
- Calls `embed_document` (one embedding call per document — whole text + chunks in a single ordered batch) for up to `concurrency` entries at once.
- **On success:** writes the `VectorIndex` entries across all three companion namespaces, then removes the queue entry. These are independent writes (not atomic), so a crash between them just re-processes the entry idempotently on the next pass.
- **On failure:** increments the entry's `retry_count`, persists it, and logs a `WARN` with namespace, doc-id, attempt number, and whether the budget is now exhausted.

The worker's behaviour is tuned by the `[vector_index]` TOML section: `concurrency` (default `4`), `max_retries` (default `5`), and `retry_wait_secs` (default `2`, slept after any pass containing a failure).

### Retry & exhaustion

A failed entry is retried up to `max_retries` times (incrementing `retry_count` each failure, with a `retry_wait_secs` back-off between passes). Once `retry_count` reaches `max_retries` the entry is **exhausted**:

- It is **skipped on every subsequent pass** and **left in the queue** (never auto-deleted) for inspection.
- Each pass logs a `WARN` that *N* entries have reached `max_retries` and await action.
- **The document itself is untouched** — it remains stored and fully readable via normal reads/scans. Only its vector index is missing, so it simply won't appear in semantic-search results until re-indexed.

Exhausted entries can be inspected or removed individually:

```
GET    /admin/indices/{ns}/vector/queue            → list queued entries (incl. retry_count, last_error)
GET    /admin/indices/{ns}/vector/queue/{doc_id}   → inspect one entry
DELETE /admin/indices/{ns}/vector/queue/{doc_id}   → drop one entry
```

### Recovering exhausted entries (re-indexing)

The queue is keyed by `(namespace, doc_id)`, so an entry is a **single row that is overwritten**, never duplicated — you can never have two competing rows for the same document. Re-enqueueing therefore **resets** the existing entry rather than appending a second one. Every re-enqueue path writes `retry_count = 0`, so an exhausted (`retry_count = 5`) entry becomes actionable again and the worker retries it on its next pass:

| Trigger | Endpoint / call | Effect on an exhausted entry |
|---|---|---|
| A fresh write to the same document | `put` / `kv_put` → `enqueue_embed` | Overwrites the key → `retry_count = 0` |
| Re-index one entry | `POST /admin/indices/{ns}/vector/queue/{doc_id}/retry` | Resets that entry → `retry_count = 0` |
| Re-index all failed | `POST /admin/indices/{ns}/vector/reindex-failed` | Resets every exhausted entry in `{ns}` → `retry_count = 0` |
| Full re-index | `POST /admin/indices/{ns}/vector/reindex-all` | **Deletes** existing exhausted entries, then re-enqueues every document at `retry_count = 0` |

### Queue entry format

Queue keys encode `(namespace, doc_id)` (length-prefixed namespace ‖ doc-id bytes) so rapid successive writes to the same document overwrite the entry — the worker makes exactly one dual-embedding call for the most-recent text. Queue values are versioned binary, holding the `retry_count` and (v2) the last error text:

- v1: `0x01 ‖ retry_count (4 B BE) ‖ text_bytes`
- v2: `0x02 ‖ retry_count (4 B BE) ‖ error_len (4 B BE) ‖ error_bytes ‖ text_bytes`

### Durability guarantees

#### Queue durability & crash recovery

The pending-embed queue lives in a standard minnal_db namespace, so every enqueue and dequeue is WAL-backed and survives a crash. On startup the worker drains any entries that were still queued before entering its normal notify-driven loop, so work that was in flight at the moment of a crash is simply re-attempted.

The three companion-namespace vector **writes** — sparse chunks (`{ns}_sparse_vector`), sparse-meta (`{ns}_sparse_vector_meta`), and dense (`{ns}_dense_vector`) — are written **without** the WAL (`put_no_wal`): the quantised payloads are bulky and fully reconstructable by re-embedding, so a per-chunk WAL fsync plus a second copy of every payload in the WAL is pure overhead. The tradeoff is that a crash before the memtable flush can drop a just-indexed vector (or flush one half and lose the other); that window is healed by reconciliation (below).

The cleanup **deletes** are the exception — they stay **WAL-backed**. These are the stale-cluster deletes during an upsert (composite keys for clusters the re-embedded document no longer belongs to) and the full reverse-lookup cleanup on document deletion (`delete_vector` reads `{ns}_sparse_vector_meta` to find and delete every composite key, then the dense entry). They are tiny key-only tombstones, so the no-WAL throughput argument does not apply, and — unlike a lost payload write — a lost delete is **not** self-healing: reconciliation re-enqueues only documents *missing* an index (missing data), whereas a lost delete leaves an *extra* orphan composite key (excess data) that a re-embed would never revisit. Keeping the deletes durable prevents that phantom entirely. Every operation is idempotent: if a multi-namespace write fails partway, a retry always converges to the correct final state.

#### Orphaned index entries are filtered at read time

A document and its vector index are separate writes that can drift apart — for example during a write crash window, or when a delete races the async indexer. As a result, a search candidate's `doc_id` may not resolve to a live document.

The search path guards against this by fetching each hit's document and dropping any that no longer exist (`decode_results` / `hydrate_kv_results` in the API layer), so an orphaned vector-index entry never surfaces as a dangling search result. This is a cheap read-time filter only — it does not delete the orphan; that is reconciliation's job.

#### Forward reconciliation (startup + on demand)

Reconciliation closes two crash windows: (1) the `put` / `kv_put` window where a document was durably written but its separate embed enqueue was lost to a crash, and (2) the `put_no_wal` vector-write window — the quantised vector payloads are written without the WAL for throughput, so a crash before the memtable flush drops a just-indexed vector. `DocStore::reconcile_vector_indexes` scans every semantic-search-enabled namespace and re-enqueues any document that has **neither** a *complete* committed vector index **nor** a pending queue entry. It is the vector-index analogue of how field indices self-heal via WAL replay, except the recovered work is routed back into the async embedding queue.

"Complete" means **both** halves are present: the sparse-meta record (`{ns}_sparse_vector_meta`) **and** the dense entry (`{ns}_dense_vector`). A normally-indexed document always has both, so a doc with only one half is a *partially* committed index — the second crash window can flush one side and lose the other — and reconciliation treats it as not-indexed and re-enqueues it. The re-embed then regenerates the missing half idempotently. (Requiring only *either* half would silently leave such a document permanently half-indexed.)

It runs **automatically as a background task on store startup** (`with_semantic_search` spawns it; it never blocks startup, and on failure it logs an error so an operator can re-run it). The startup pass is *presence-only* (cheap count short-circuit). It is also exposed on demand at `POST /admin/indices/vector/reconcile`, which runs a stronger **validating** pass (`DocStore::validate_and_reconcile_vector_indexes`): it deserializes every committed entry to also catch present-but-corrupt vectors (which the presence check cannot), so it skips the short-circuit and runs as a full background scan returning `202 Accepted` (with a `409` guard against overlapping runs).

A cheap **count short-circuit** skips the full per-document scan for a namespace when nothing is queued and **both** companion-key counts (sparse-meta *and* dense) already cover the live-key count (all are LSM-only key scans, with no value reads). Requiring both — not just the sparse-meta count — is what keeps the short-circuit consistent with the complete-index rule: a namespace where every key has sparse-meta but some lost their dense write must not be skipped. Namespaces with empty-embedding-text documents fall through to the full scan, which is still correct because it enqueues nothing.

Reverse reconciliation — deleting orphan index entries for documents that were deleted — is intentionally **not** performed here: it is destructive and races the async indexer, so the read-time filter above handles the user-visible symptom instead.

---

## 8. Hybrid Search

The `search()` function accepts an optional `doc_filter: Option<F>` closure (`service/mod.rs`). It is applied **only in Pass 1** (`line ~281`), so only documents passing the predicate are scored:

```rust
if doc_filter.as_ref().is_some_and(|f| !f(doc_id)) { continue; }
```

This enables queries like "find documents semantically similar to X **and** matching status='active'". The filter operates on raw `doc_id` bytes, so predicate evaluation is external to the vector pipeline. Pass 2 receives the already-filtered candidate list from Pass 1 and does not re-evaluate the filter.

---

## 9. Configuration

All parameters are under `[semantic_search]` in the TOML config:

| Parameter | Default | Description |
|---|---|---|
| `number_of_bits_for_dense_quantisation` | `8` | Bits per dimension for MultiBit (dense) quantisation. 4 = compact, 8 = high recall. Only affects Pass 2 precision. |
| `n_probes` | `32` | Number of IVF clusters probed per query in the sparse pass. Higher = better recall, slower — see *Tuning & profiling* below. |
| `first_pass_sparse_search_top_k` | `1000` | Candidates retained after Pass 1 before dense re-ranking. |
| `window_size` | `4` | Sentences/tokens per sliding-window chunk for SingleBit embeddings. |
| `sliding_size` | `2` | Window advance step. Smaller than `window_size` → overlapping chunks. |
| `cluster_path` | — | Path to the JSONL cluster centroids file. |
| `embedding_service_url` | `http://localhost:8001` | Base URL of the external embedding service. |
| `top_k_results` | `100` | Maximum results returned per query (overridable per-request). |
| `query_embedding_cache_ttl_secs` | `86400` | TTL (seconds) for cached query embeddings in `system_qemb_cache`. Default is 1 day. |

### Tuning & profiling `n_probes`

`n_probes` is the primary recall/latency knob. Two on-demand harnesses in
`minnal_doc_store/src/vector_kv.rs` (both `#[ignore]`d tests, run from the
`minnal_doc_store/` crate root) measure the two axes it trades off:

- **Latency** — `real_kv_search_profile` runs the real two-pass `search()` over a real
  `minnal_db`-backed store (synthetic vectors, but real LSM + value-log `pread` I/O) and
  phase-times coarse cluster pick, Pass-1 sparse-scan I/O, Pass-2 dense-fetch I/O, the
  scoring remainder, and the whole query — sweeping `n_probes ∈ {10, 32, 128}` at several
  chunks-per-doc. It needs no embedding service.
  ```sh
  cargo test -p minnal_doc_store --lib real_kv_search_profile --release -- --ignored --nocapture
  ```
- **Recall** — `real_recall_vs_nprobes` indexes a real text corpus through the real
  embedding service (the production `embed_document` → `upsert_vectors` path) and reports
  `recall@k(n_probes) = |top_k(n_probes) ∩ top_k(exhaustive)| / k`, using the pipeline's
  own **exhaustive-probe** (all clusters) ranking as ground truth — so it isolates the
  recall lost by *reducing* `n_probes`, holding quantisation and re-ranking fixed. Requires
  the embedding service and a JSONL corpus; env-gated via `MINNAL_EMBED_URL`,
  `MINNAL_RECALL_CORPUS`, `MINNAL_RECALL_DOCS`, `MINNAL_RECALL_QUERIES` (soft-skips if either
  is absent).
  ```sh
  MINNAL_EMBED_URL=http://<host>:8001 \
    cargo test -p minnal_doc_store --lib real_recall_vs_nprobes --release -- --ignored --nocapture
  ```

Measured tradeoff (recall: 2000-doc real news corpus, 50 queries; latency: 5000-doc
synthetic store, 8 chunks/doc, warm cache):

| `n_probes` | recall@10 | recall@100 | entire `search()` |
|---|---|---|---|
| 10 | 0.968 | 0.935 | 14.5 ms |
| **32 (default)** | **0.986** | **0.978** | **18.7 ms** |
| 128 | 1.000 | 0.999 | 26.4 ms |

`32` is the default: it recovers most of the recall lost at `10` while staying ~29% cheaper
than `128`. The dominant lever is **Pass-1 sparse-scan I/O**, which scales roughly linearly
with `n_probes` (entries scanned grow in step); the SIMD dot products are a minority of the
cost. Pass-2 dense fetch is roughly fixed — it re-ranks a probe-independent
`first_pass_sparse_search_top_k` candidate set — so its share *shrinks* as `n_probes` rises.

---

## Key Files

| Component | File |
|---|---|
| Embedding service client + two-pass search | `semantic_search/src/service/mod.rs` |
| RaBitQ quantisation (encode + decode) | `semantic_search/src/quantisation/rabitq/mod.rs` |
| `VectorIndex` struct + `VectorKvStore` trait | `semantic_search/src/index/vector_index.rs` |
| Distance estimators (SingleBit, MultiBit) | `semantic_search/src/index/distance_estimator.rs` |
| Cluster index (centroids) + exact top-`n_probes` probing | `semantic_search/src/cluster/mod.rs` |
| Composite key encoding (cluster ‖ doc_id) | `semantic_search/src/index/composite_key.rs` |
| Coarse-assignment micro-benchmarks | `semantic_search/benches/bench_distance_estimation.rs` |
| Vector KV storage (three namespaces) + query cache | `minnal_doc_store/src/vector_kv.rs` |
| Latency + recall profiling harnesses (see §9) | `minnal_doc_store/src/vector_kv.rs` (ignored tests) |
| Async vector-index background worker | `minnal_doc_store/src/vec_index_worker.rs` |
| Document store | `minnal_doc_store/src/store.rs` |
