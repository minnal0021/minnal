# semantic_search — Vector Quantisation + ANN Search

Implements IVF (Inverted File Index) clustering with RaBitQ quantisation for two-pass approximate nearest-neighbour search over dense embeddings. Vectors are stored in `minnal_db`; this crate handles quantisation, cluster assignment, and search.

## Key files

| File | Role |
|---|---|
| `src/lib.rs` | Public re-exports |
| `src/chunking/mod.rs` | Document chunking: `chunk_document` (sentence-split → `sliding_windows`). Queries are not chunked. |
| `src/cluster/mod.rs` | `Cluster`, `ClusterIndex` — IVF cluster centroids, nearest-cluster lookup |
| `src/index/vector_index.rs` | `VectorIndex` struct, `VectorKvStore` trait (`scan_sparse_cluster`, `get_dense_entry`) |
| `src/index/composite_key.rs` | Composite key layout: `cluster_id (4B BE) ‖ doc_id` |
| `src/index/distance_estimator.rs` | `SingleBitQuanDotProductEstimator` (Pass 1) and `MultiBitQuanDotProductEstimator` (Pass 2) |
| `src/quantisation/rabitq/` | RaBitQ multi-bit and single-bit quantisation (encode + decode) |
| `src/service/mod.rs` | HTTP client for the external embedding service; `embed_document` / `embed_query`; `search()` two-pass ANN |
| `src/beir_eval.rs` | `#[ignore]`d BEIR relevance eval of the production pipeline (nDCG@10, candidate recall, probes, latency) |
| `query-embedding-report.md` | BEIR evaluation behind whole-query Pass-1 embedding (vs chunked queries) |
| `src/vector_math/mod.rs` | `vector_math` module — L2 normalisation, residuals, RaBitQ quantisation/bit-packing helpers (SIMD via `simsimd`) |

## How it works

### Chunking (done here, not by the service)

The embedding service no longer chunks text — `src/chunking/mod.rs` does. **Only documents are chunked**: `chunk_document` splits on **sentence** boundaries and groups the sentences into overlapping **sliding windows** (`window_size` sentences per window, advancing `sliding_size`; the last window keeps the remainder), each joined into one payload string. **Queries are not chunked** — one whole-query embedding serves both passes (see *Search* below). The service receives a list of already-prepared payloads and returns one embedding per payload.

### Indexing (dual quantisation)

Each document goes through a **single** embedding service call (`embed_document`) that embeds one ordered payload list — `[whole_text, chunk₀, chunk₁, …]` — and splits the (order-preserving) response by position:

1. **MultiBit (dense)** — `payload[0]`, one embedding for the whole document text, quantised to `number_of_bits_for_dense_quantisation` bits per dimension. Stored in `{ns}_dense_vector` keyed by `doc_id`. Used for high-precision re-ranking in Pass 2.
2. **SingleBit (sparse, sliding-window)** — `payload[1..]`, `chunk_document`'s N sliding-window chunks → N embeddings. Each chunk is independently assigned to its nearest IVF cluster and quantised to 1 bit per dimension. Stored in `{ns}_sparse_vector` under composite keys `[cluster_id (4B BE) ‖ doc_id]`. Used for fast first-pass cluster scans.

Folding both into one call is one round trip and one GPU batch of `N+1` (the service preserves payload order, so position 0 is always the dense vector). The query path (`embed_query`) sends a single payload: the whole query.

A third namespace, `{ns}_sparse_vector_meta`, records which clusters each document's sparse chunks belong to, for use only by delete and upsert cleanup.

**Durability split (no-WAL writes, WAL-backed deletes).** All three vector **writes** — `{ns}_sparse_vector`, `{ns}_sparse_vector_meta`, `{ns}_dense_vector` — are written with `put_no_wal` (`vector_kv::upsert_vectors`): the quantised payloads are bulky and reconstructable by re-embedding, so a per-chunk WAL fsync plus a second copy in the WAL is pure overhead. A crash before the memtable flush can drop a just-indexed vector (or flush one half and lose the other); that window is healed by vector-index reconciliation, which treats a doc as indexed only when **both** the sparse-meta and dense halves are present (`has_complete_vector_index`). The cleanup **deletes** are the exception and stay **WAL-backed** — the stale-cluster deletes on upsert and the full reverse-lookup `delete_vector` (reads `{ns}_sparse_vector_meta` to delete every composite key, then the dense entry). They are tiny key-only tombstones, and unlike a lost write a lost delete is *not* self-healing: reconciliation re-enqueues only docs *missing* an index, whereas a lost delete leaves an *orphan* composite key a re-embed never revisits.

The **query-embedding cache** (`system_qemb_cache`, below) follows the same split. Populates are no-WAL (`put_no_wal`): the cache is TTL-bounded and fully regenerable from the embedding service, so a dropped populate is just a future miss, and staying off the WAL removes a per-populate fsync from the query hot path. The admin clear (`DELETE /admin/indices/vector/query-cache`) deletes **WAL-backed**: the periodic no-WAL flush tick has usually already persisted the entries being cleared, so a no-WAL tombstone lost to a crash before the next tick would resurrect them — serving entries the operator explicitly cleared until the TTL expires. The clear is a rare admin operation, so the per-delete fsync is off the query path.

### Search (two-pass ANN)

`embed_query` makes **one** embedding of the whole query (one payload, one service call) and uses it for **both** passes: as Pass 2's dense vector and as Pass 1's single MaxSim query vector (`QueryEmbeddings { sparse: vec![dense.clone()], dense }`). Queries used to be split into 4-word sliding windows; a BEIR evaluation (`query-embedding-report.md`) found the whole-query vector never worse and often better. With a tight first-pass cut it kept more relevant docs (ArguAna candidate recall 0.991 vs 0.870). It is also far cheaper: one embedding instead of 1+N, and `n_probes` clusters instead of the union over every fragment (up to 74% faster search). Document-style sentence windows for long queries gained nothing either, so there is **no query-chunking option**. `search()` still accepts several Pass-1 query vectors (general MaxSim) and returns empty if either input is empty. The embedding is cached in the `system_qemb_cache` TTL namespace, keyed by query text (`query_embedding_cache_ttl_secs`, default 1 day). Since queries are not chunked, chunking settings never invalidate it; clear it (`DELETE /admin/indices/vector/query-cache`) only after changing the embedding model or service.

**Pass 1 — sparse (SingleBit), ColBERT MaxSim over document chunks:**
1. Use the Pass-1 query vector (the whole-query embedding; fetched from the `system_qemb_cache` TTL namespace on a hit).
2. Find its top-`n_probes` clusters by Euclidean distance (with several query vectors, the union across them). This is an **exact, exhaustive** scan — `ClusterIndex::find_top_n_cluster_ids_batch` computes the distance to *every* centroid (over a contiguous centroid matrix) and selects the `n_probes` nearest (`select_nth_unstable`, ~O(C)). There is **no neighbour graph / approximate traversal**: coarse-assignment cost is **T·C·D** (query vectors × centroids × dim), and production `T = 1`, so it is ~microseconds at C≈256. A graph is the wrong lever — approximate on the most recall-sensitive stage, marginal at a few hundred nodes. The contiguous-matrix scan was a measured ~12% win back when queries were chunked (T up to 100); **parallelising the per-vector scans was tried and reverted (~2.4× slower — the work is microseconds, so the thread pool costs more than it saves)**; a blocked GEMM is the only remaining lever and only pays at far larger C. See `Semantic-Search-Architecture.md`.
3. `scan_sparse_cluster(cluster_id)` for each probed cluster in parallel.
4. Apply the optional `doc_filter` (RoaringBitmap predicate) — skip non-matching docs.
5. Score with `SingleBitQuanDotProductEstimator` using **ColBERT MaxSim**:
   - For each query vector `q_i` and each document `d`, find `max_j ⟨q_i, d_j⟩` over all chunks `d_j` of `d` in the probed cluster.
   - Accumulate the per-vector max across all probed clusters (same chunk may appear in several).
   - Final score: `S(q, d) = Σ_i max_j ⟨q_i, d_j⟩`. With production's single whole-query vector this is `max_j ⟨q, d_j⟩`, the document's best-matching chunk.
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

A whole-text ("single") embedding is just a one-element `payloads` array (every query is one); a document sends its whole text plus one payload per sentence-window chunk. The `{model}` path segment from the old API is gone (the model is fixed server-side). Default base URL: `http://localhost:8001`.

### Startup probe & the model-pinning gap (operational, not enforced)

`check_embedding_service` (called once at startup, non-fatal) does more than ping `/healthcheck`: it embeds a fixed probe payload through **both** the document and query endpoints and validates the returned **dimension** against `embedding_dim` (a service on a different dimension fails the probe instead of degrading search silently), then soft-warns if the probe vector is not unit-norm. **What it cannot validate is the model itself** — the service exposes no model family/version metadata, so a *wrong model with the same dimension* passes every check while the bundled cluster centroids are for a different embedding distribution (silently degraded recall). **Model pinning is therefore an operational guarantee, not an enforced one:** deployment must ensure `[[semantic_search.supported_models]]` / `cluster_path` match the model the service actually serves. If the service later exposes a model/version endpoint, enforce it here.

## Cluster centroids

Pre-built centroids ship per model at `service/embedding_support/{model}/clusters.json` — currently **gemma** (what the companion embedding service serves) and **qwen**, each 256 centroids × 768 dims. Set `semantic_search.cluster_path` to the one matching the model the service actually serves; both are 768-dim, so a mismatch passes `load_with_dim` and degrades recall silently. Each file is ~4.4 MB of JSONL — do not read it; it is data, not code.

**The gemma set was regenerated on 2026-09-25 for the PyTorch embedding service.** The earlier ONNX/FastEmbed service mean-pooled raw hidden states and skipped EmbeddingGemma's two Dense projections, so its vectors, and the old centroids fit on them, lived in a different space (cosine ≈ 0 vs the real embeddings). Any vector index built with the old service is incompatible with this file: re-index it (`POST /admin/indices/{ns}/vector/reindex-all`) and clear the query-embedding cache (`DELETE /admin/indices/vector/query-cache`). Provenance, reproducible with the embedding service's `generate_sample_embeddings.sh` + `generate_cluster_centroids.sh`: ELI5 QA pairs (first 25,024 records, `question + "\n" + answer`, document prompt), K-means k=256, seed 42. It is a **general-purpose example set**: well spread on general text (253/256 clusters used by held-out ELI5), but specialist text collapses (SciFact: 43% of docs in one cluster), so domain-specific deployments should fit centroids on their own data.

## Configuration (from TOML)

```toml
[semantic_search]
# Bits per dimension for the dense (multi-bit) quantisation used in Pass 2.
# 4 = compact, 8 = better recall (default).
number_of_bits_for_dense_quantisation = 8

cluster_path = "service/embedding_support/gemma/clusters.json"

# embedding_service_url = "http://localhost:8001"

# Number of IVF clusters probed in Pass 1.  Higher = better recall, slower.
# n_probes = 32

# Candidates kept after Pass 1 before dense re-ranking.
# first_pass_sparse_search_top_k = 1000

# Sentence-window chunk parameters for document SingleBit embeddings
# (queries are not chunked).
# window_size = 4
# sliding_size = 2
```

> **`window_size` / `sliding_size` shape the stored document chunks — changing them requires a full corpus re-index** (`POST /admin/indices/{ns}/vector/reindex-all`); already-indexed documents keep their old chunks until re-embedded, and nothing errors. They do **not** affect queries (queries are not chunked), so the query-embedding cache needs no clear. **Don't reintroduce query chunking** without re-running the BEIR harness: word-window query chunks were measured worse and far costlier than the whole-query vector, and sentence windows gained nothing (`query-embedding-report.md`).

## Key types

- `VectorIndex` — a quantised embedding entry: holds `cluster_id`, packed bit codes (`binary_quantised_vector`), scalar correction coefficients (`addition_factor`, `scaling_factor`, `error_bound`), and `quantisation_style` (`SingleBit` or `MultiBit { number_of_bits }`).
- `VectorKvStore` — trait over the storage backend; exposes `scan_sparse_cluster(cluster_id)` (returns all SingleBit entries for that cluster) and `get_dense_entry(doc_id_bytes)` (returns the MultiBit bytes for a document).
- `Cluster` — a single centroid + its pre-computed norms for fast distance estimation.
- `ClusterIndex` — the full set of centroids, loaded from `cluster_path` on startup.
- `SemanticSearchConfig` — all tunable parameters for the search pipeline.
