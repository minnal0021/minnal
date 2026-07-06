# semantic_search — Vector Quantisation + ANN Search

Implements IVF (Inverted File Index) clustering with RaBitQ quantisation for two-pass approximate nearest-neighbour search over dense embeddings. Vectors are stored in `minnal_db`; this crate handles quantisation, cluster assignment, and search.

## Key files

| File | Role |
|---|---|
| `src/lib.rs` | Public re-exports |
| `src/chunking/mod.rs` | Text chunking: `chunk_document` (sentence-split), `chunk_query` (word-tokenise), `sliding_windows` |
| `src/cluster/mod.rs` | `Cluster`, `ClusterIndex` — IVF cluster centroids, nearest-cluster lookup |
| `src/index/vector_index.rs` | `VectorIndex` struct, `VectorKvStore` trait (`scan_sparse_cluster`, `get_dense_entry`) |
| `src/index/composite_key.rs` | Composite key layout: `cluster_id (4B BE) ‖ doc_id` |
| `src/index/distance_estimator.rs` | `SingleBitQuanDotProductEstimator` (Pass 1) and `MultiBitQuanDotProductEstimator` (Pass 2) |
| `src/quantisation/rabitq/` | RaBitQ multi-bit and single-bit quantisation (encode + decode) |
| `src/service/mod.rs` | HTTP client for the external embedding service; `search()` two-pass ANN |
| `src/vector_math/mod.rs` | `vector_math` module — L2 normalisation, residuals, RaBitQ quantisation/bit-packing helpers (SIMD via `simsimd`) |

## How it works

### Chunking (done here, not by the service)

The embedding service no longer chunks text — `src/chunking/mod.rs` does. Documents are split on **sentence** boundaries (`chunk_document`), queries on **word** boundaries (`chunk_query`); both are then grouped into overlapping **sliding windows** (`window_size` units per window, advancing `sliding_size` units; the last window keeps the remainder) and each window is joined into one payload string. The service receives a list of already-prepared payloads and returns one embedding per payload.

### Indexing (dual quantisation)

Each document goes through a **single** embedding service call (`embed_document`) that embeds one ordered payload list — `[whole_text, chunk₀, chunk₁, …]` — and splits the (order-preserving) response by position:

1. **MultiBit (dense)** — `payload[0]`, one embedding for the whole document text, quantised to `number_of_bits_for_dense_quantisation` bits per dimension. Stored in `{ns}_dense_vector` keyed by `doc_id`. Used for high-precision re-ranking in Pass 2.
2. **SingleBit (sparse, sliding-window)** — `payload[1..]`, `chunk_document`'s N sliding-window chunks → N embeddings. Each chunk is independently assigned to its nearest IVF cluster and quantised to 1 bit per dimension. Stored in `{ns}_sparse_vector` under composite keys `[cluster_id (4B BE) ‖ doc_id]`. Used for fast first-pass cluster scans.

Folding both into one call is one round trip and one GPU batch of `N+1` (the service preserves payload order, so position 0 is always the dense vector). The query path (`embed_query`) does the same with `chunk_query`.

A third namespace, `{ns}_sparse_vector_meta`, records which clusters each document's sparse chunks belong to, for use only by delete and upsert cleanup.

**Durability split (no-WAL writes, WAL-backed deletes).** All three vector **writes** — `{ns}_sparse_vector`, `{ns}_sparse_vector_meta`, `{ns}_dense_vector` — are written with `put_no_wal` (`vector_kv::upsert_vectors`): the quantised payloads are bulky and reconstructable by re-embedding, so a per-chunk WAL fsync plus a second copy in the WAL is pure overhead. A crash before the memtable flush can drop a just-indexed vector (or flush one half and lose the other); that window is healed by vector-index reconciliation, which treats a doc as indexed only when **both** the sparse-meta and dense halves are present (`has_complete_vector_index`). The cleanup **deletes** are the exception and stay **WAL-backed** — the stale-cluster deletes on upsert and the full reverse-lookup `delete_vector` (reads `{ns}_sparse_vector_meta` to delete every composite key, then the dense entry). They are tiny key-only tombstones, and unlike a lost write a lost delete is *not* self-healing: reconciliation re-enqueues only docs *missing* an index, whereas a lost delete leaves an *orphan* composite key a re-embed never revisits.

The **query-embedding cache** (`system_qemb_cache`, below) is a *separate* no-WAL story: it is no-WAL for **all** CRUD — both populate (`put_no_wal`) and clear (`delete_no_wal`). The vector-cleanup "deletes stay WAL-backed" reasoning does **not** apply because the cache is TTL-bounded and fully regenerable from the embedding service, so a dropped cache write or a lost clear-delete is harmless — the stale entry just expires (or is re-fetched on the next miss). Keeping it off the WAL removes a per-populate fsync from the query hot path. This relies on the engine's `AsyncNamespace::delete_no_wal` (the no-WAL counterpart of `put_no_wal`).

### Search (two-pass ANN)

`embed_query` returns both query inputs from one service call (cached together in the `system_qemb_cache` TTL namespace, dense as element 0): the **sparse** chunk embeddings (`chunk_query` → N vectors) for Pass 1, and a **dense** single whole-query embedding for Pass 2. `search()` takes both as separate arguments and returns empty if either is empty. The cache is keyed only by query text — clear it (`DELETE /admin/indices/vector/query-cache`) after changing `window_size`/`sliding_size`, or stale chunkings are served until the TTL expires (configurable via `query_embedding_cache_ttl_secs`, default 1 day).

**Pass 1 — sparse (SingleBit), ColBERT MaxSim:**
1. Use the sparse query chunk embeddings (or fetch from the `system_qemb_cache` TTL namespace).
2. For each query chunk, find the top-`n_probes` clusters by Euclidean distance; union across all query chunks. This is an **exact, exhaustive** scan — `ClusterIndex::find_top_n_cluster_ids_batch` computes the distance to *every* centroid (over a contiguous centroid matrix) and selects the `n_probes` nearest per chunk (`select_nth_unstable`, ~O(C)). There is **no neighbour graph / approximate traversal**: coarse-assignment cost is **T·C·D** (query chunks × centroids × dim), which at C≈256 is genuinely cheap (~50 µs at 4 chunks, ~1.1 ms at 100) — small vs the rest of a query even on a warm cache. A graph is the wrong lever — approximate on the most recall-sensitive stage, marginal at a few hundred nodes, no help on the per-chunk T factor. The contiguous-matrix scan is a measured ~12% win; **parallelising the per-chunk scans was tried and reverted (~2.4× slower — the work is microseconds, so the thread pool costs more than it saves)**; a blocked GEMM is the only remaining lever and only pays at far larger C. See `Semantic-Search-Architecture.md`.
3. `scan_sparse_cluster(cluster_id)` for each probed cluster in parallel.
4. Apply the optional `doc_filter` (RoaringBitmap predicate) — skip non-matching docs.
5. Score with `SingleBitQuanDotProductEstimator` using **ColBERT MaxSim**:
   - For each query token `q_i` and each document `d`, find `max_j ⟨q_i, d_j⟩` over all chunks `d_j` of `d` in the probed cluster.
   - Accumulate the per-token max across all probed clusters (same chunk may appear in several).
   - Final score: `S(q, d) = Σ_i max_j ⟨q_i, d_j⟩` — documents matching more query tokens score higher.
6. Retain top `first_pass_sparse_search_top_k` candidates.

**Pass 2 — dense (MultiBit):**
1. `get_dense_entry(doc_id)` for each sparse candidate in parallel.
2. Score each candidate with `MultiBitQuanDotProductEstimator` against the single whole-query dense embedding (symmetric with the document's whole-text dense vector).
3. Build top-k min-heap and return sorted descending.

The `doc_filter` is applied **only in Pass 1**. Pass 2 operates on the already-filtered candidate list.

## External dependency

**The embedding service must be running** for any vector insert or query to work. Without it, `semantic_search` calls will return an error. The service is not part of this workspace.

Requests use a **batch interface** (chunking happens in minnal, not the service):
- `POST {base_url}/embedding/document` — body `{"payloads": [str, ...], "dimensions": N}` → `{"embeddings": [[f32], ...]}` (one vector per payload)
- `POST {base_url}/embedding/query` — same request/response shape
- `GET {base_url}/healthcheck`

A whole-text ("single") embedding is just a one-element `payloads` array; chunked embeddings send one payload per sliding-window chunk. The `{model}` path segment from the old API is gone (the model is fixed server-side). Default base URL: `http://localhost:8001`.

### Startup probe & the model-pinning gap (operational, not enforced)

`check_embedding_service` (called once at startup, non-fatal) does more than ping `/healthcheck`: it embeds a fixed probe payload through **both** the document and query endpoints and validates the returned **dimension** against `embedding_dim` (a service on a different dimension fails the probe instead of degrading search silently), then soft-warns if the probe vector is not unit-norm. **What it cannot validate is the model itself** — the service exposes no model family/version metadata, so a *wrong model with the same dimension* passes every check while the bundled cluster centroids are for a different embedding distribution (silently degraded recall). **Model pinning is therefore an operational guarantee, not an enforced one:** deployment must ensure `[[semantic_search.supported_models]]` / `cluster_path` match the model the service actually serves. If the service later exposes a model/version endpoint, enforce it here.

## Cluster centroids

Pre-built centroids are at `service/embedding_support/qwen/clusters.json`. Set `semantic_search.cluster_path` in config to point at this file. The file is ~784 KB of JSON — do not read it; it is data, not code.

## Configuration (from TOML)

```toml
[semantic_search]
# Bits per dimension for the dense (multi-bit) quantisation used in Pass 2.
# 4 = compact, 8 = better recall (default).
number_of_bits_for_dense_quantisation = 8

cluster_path = "service/embedding_support/qwen/clusters.json"

# embedding_service_url = "http://localhost:8001"

# Number of IVF clusters probed in Pass 1.  Higher = better recall, slower.
# n_probes = 128

# Candidates kept after Pass 1 before dense re-ranking.
# first_pass_sparse_search_top_k = 1000

# Sliding-window chunk parameters for SingleBit embeddings.
# window_size = 4
# sliding_size = 2
```

> **`window_size` / `sliding_size` are effectively an on-disk decision — changing them requires a full corpus re-index.** Both document indexing (`embed_document`) and query embedding (`store.rs` search paths) read the *same* values, so Pass-1 ColBERT MaxSim only compares query chunks against document chunks that were produced with identical chunking. If you edit these after documents are indexed, new queries use the new chunking while stored sparse vectors keep the old — the dot products stay comparable enough to return results but recall degrades **silently** (no error). After changing either value you must do **both**: (1) re-embed the corpus (`POST /admin/indices/{ns}/vector/reindex-all`), and (2) clear the query-embedding cache (`DELETE /admin/indices/vector/query-cache`) — the cache is keyed only by query text, so without this it keeps serving old-chunking sparse vectors for up to its configured TTL (`query_embedding_cache_ttl_secs`, default 1 day). For the same reason, do **not** add a query-only override for these.

## Key types

- `VectorIndex` — a quantised embedding entry: holds `cluster_id`, packed bit codes (`binary_quantised_vector`), scalar correction coefficients (`addition_factor`, `scaling_factor`, `error_bound`), and `quantisation_style` (`SingleBit` or `MultiBit { number_of_bits }`).
- `VectorKvStore` — trait over the storage backend; exposes `scan_sparse_cluster(cluster_id)` (returns all SingleBit entries for that cluster) and `get_dense_entry(doc_id_bytes)` (returns the MultiBit bytes for a document).
- `Cluster` — a single centroid + its pre-computed norms for fast distance estimation.
- `ClusterIndex` — the full set of centroids, loaded from `cluster_path` on startup.
- `SemanticSearchConfig` — all tunable parameters for the search pipeline.
