# Semantic Search Architecture

This document describes how semantic search works end to end: embedding generation, dual quantisation, the dynamically partitioned index, two-pass query execution (with optional predicate filtering), storage layout, and crash recovery.

---

## 1. Embeddings

### Overview

Minnal does **not** generate embeddings itself. It relies on an **external embedding service** and treats it as the sole source of vectors: minnal prepares payloads, the service returns embeddings, and minnal quantises and indexes them as-is.

> **Companion embedding service:** [minnal0021/embedding_service](https://github.com/minnal0021/embedding_service) is a reference implementation that serves two embedding models over HTTP — **gemma** (EmbeddingGemma-300M) and **qwen** (Qwen3-Embedding-8B) — the external dependency described here. Run it, then point `semantic_search.embedding_service_url` at it (default `http://localhost:8001`).

The service is reached over **HTTP**. Its base URL is configured under `[semantic_search]` in the TOML config (`embedding_service_url`) and defaults to `http://localhost:8001`.

One service can load several models, so **minnal names the model on every request**. Each store chooses its model in its schema (`vector_index.embedding_model`, e.g. `gemma` or `qwen`; default `gemma`; lower-cased before use), and that name is sent as the `{model}` path segment of both embedding endpoints, so it decides which of the service's models embeds the text. The service answers 404 for a model it was not started with. Which build of a model sits behind a name (its quantisation, say) is still the service's choice.

Every stored vector of a store must come from the same model: the same text embeds to unrelated vectors under different models. That is why a store's model, dimension and chunking are fixed once semantic search is enabled. When a store enables semantic search, the API server embeds a probe text with the store's model and rejects the request (422) if the service does not serve the model or returns another dimension. A server can also restrict the models it accepts (`[[semantic_search.supported_models]]`, each with its dimension); without that list it accepts any model the service serves.

### Service interface

Each request carries a **`payloads` array of strings and returns one embedding per string**, so a single HTTP call can embed many strings at once. Minnal decides how many strings to send: chunking/tokenisation happens in minnal (`chunking/mod.rs`), not in the service, so the service never splits a string — it embeds exactly the strings it is given. To embed a whole text, minnal sends a `payloads` array with a single string — every query is exactly that, since queries are not chunked; to embed a chunked (sliding-window) document, it sends one string per chunk — all in the same request.

Two `POST` endpoints per model, identical in request/response shape, differing only in which side of the asymmetric model they target (documents are embedded differently from queries). `{model}` is the configured `model`:

| Method & path | Purpose |
|---|---|
| `POST {base_url}/embedding/{model}/document` | Embed document payloads (indexing). |
| `POST {base_url}/embedding/{model}/query` | Embed query payloads (search). |

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

Minnal validates the response and errors if the number of returned embeddings differs from the number of payloads sent (`CountMismatch`), or if any vector's length differs from the requested `dimensions` (`DimensionMismatch`). A non-2xx answer becomes a `Status` error carrying the service's `detail` message — for an unknown model, `Unknown model 'foo'; available: gemma, qwen`. An empty `payloads` array short-circuits to an empty result with no HTTP request. Extra fields in the response JSON are ignored.

A health endpoint is also expected: `GET {base_url}/healthcheck`, listing the loaded models and the state of each:

```json
{"status": "healthy", "models": {"gemma": {"status": "ok", ...}, "qwen": {"status": "ok", ...}}}
```

The service answers 503 while *any* of its models is still loading, so minnal judges only the configured model's entry: it must be present (`ModelNotServed` otherwise) with status `ok` (`ModelNotReady` otherwise). A response without a `models` map just has to be 2xx. Minnal probes this at startup, then embeds a probe text through both endpoints to check the dimension (see [Embedding service availability](#embedding-service-availability)).

### Embedding dimension

Each store sets the embedding size in its schema (`vector_index.embedding_dim`, **default 768**), sent to the service as `dimensions` on every request. Nothing else has to match it: a store's IVF centres are fitted from its own vectors (see [§3](#3-dynamic-partitioning-the-ivf-index)), so a model of any dimension works without preparing a file. The rotation codes are computed in needs an even dimension of at least 8.

### Normalisation

Every embedding returned by the service — both the **dense** whole-text vectors and the **sparse** per-chunk vectors — is expected to be **L2-normalised** (unit length) by the service. Minnal relies on this: downstream quantisation and the dot-product distance estimators assume unit vectors, so dot products can be treated as cosine similarity. Minnal does **not** re-normalise the embeddings it receives; producing unit-length vectors is the embedding service's responsibility.

---

## 2. Dual Quantisation (RaBitQ)

Before storage, embeddings are compressed using **RaBitQ** (`quantisation/rabitq/mod.rs`). At index time, a **single embedding call** is made per document — one ordered payload list `[whole_text, chunk₀, chunk₁, …]` in one round trip — and the order-preserving response is split into two classes of quantised entry:

### MultiBit (dense, whole-document)

1. A single embedding is obtained for the entire document text.
2. It is quantised to 8 bits per dimension against a **zero centre**: the code depends on no IVF centre, so it never needs re-encoding when the partition changes, and one estimator scores every candidate.

The result is stored in the **dense namespace** (`{ns}_dense_vector`) keyed directly by `doc_id`. For 768-dimensional embeddings at 8 bits, each entry compresses from ~3 KB (f32) to ~768 bytes — a 4× reduction.

### SingleBit (sparse, sliding-window chunks)

1. The document text is split into overlapping chunks using a sliding window (`window_size`, `sliding_size`).
2. The embedding service returns one embedding per chunk.
3. Each chunk embedding is routed to its nearest IVF posting and quantised to 1 bit per dimension, as a residual against that posting's centre.

The result is stored in the **sparse namespace** (`{ns}_sparse_vector`) under composite keys `[posting_id (4B BE) ‖ doc_id]`. For 768-dimensional embeddings, a SingleBit entry is ~96 bytes — a 32× reduction versus f32, enabling fast first-pass scans of the probed postings.

A third companion namespace, **`{ns}_sparse_vector_meta`**, records which postings hold each document's chunks, and how many. Delete and upsert use it to clean up stale composite keys, and each posting's size is counted from it when the partition loads; search never consults it.

### The `VectorIndex` struct

All quantised entries share the same `VectorIndex` struct:

| Field | Purpose |
|---|---|
| `cluster_id` | The IVF posting this entry is filed under (its key prefix) |
| `centre_id` | The centre the code was encoded against. A code is always decoded through it, never through the posting it is filed under, so maintenance can move a code between postings without re-encoding it |
| `packed_vector` | Packed bit codes (96 bytes SingleBit, ~768 bytes 8-bit MultiBit) |
| `addition_factor` | Scalar correction coefficient for distance estimation |
| `scaling_factor` | Scalar correction coefficient for distance estimation |
| `error_bound` | Theoretical max deviation of estimated dot product from true dot product |
| `quantisation_style` | `SingleBit` or `MultiBit { number_of_bits }` |

**Random rotation.** As in the RaBitQ paper, codes are computed in a randomly rotated space: before quantising, the residual is rotated by one orthogonal transform `P`, seeded per store (`vector_index.quantisation.rotation_seed`) (`semantic_search/quantisation/rotation.rs`, a fast Walsh–Hadamard-based rotation that handles 768 dimensions without padding), and each query is rotated once per search before it meets the codes. Rotation preserves inner products, so the estimators' formulas are unchanged; only their inputs are rotated. Cluster probing and the query-to-centroid term stay in the original space. The rotation is part of the stored format, so `embedding_dim` must be even and at least 8.

---

## 3. Dynamic Partitioning (the IVF index)

Pass 1 of a search (§4) must not compare the query with every chunk in the store. Like most vector indexes, minnal divides the chunks into groups of nearby vectors and reads only the groups nearest the query. This is an **IVF** (inverted file) index. This section explains how a store builds those groups and keeps them in shape as documents are added and deleted, what can be tuned, and what each design choice trades away.

The design follows LIRE (Lightweight Incremental REbalancing), the update protocol of the SPFresh vector index (see [Acknowledgements](#acknowledgements)) with a few deliberate differences, listed in [§3.6](#36-where-minnal-differs-from-spfresh). The full study behind the numbers here (the simulations, every gate run and the rejected alternatives) is in [`Dynamic-RaBitQ-Index-Design.md`](Dynamic-RaBitQ-Index-Design.md).

### 3.1 Terms

- **Chunk**: the **embedding** of one sliding-window piece of a document's text (§2, *SingleBit*), stored as a 1-bit code. In this section "chunk" always means that embedding, never the text: partitioning groups embeddings, and the text is not stored in the index. A document has one chunk per window, and its chunks can sit in different postings.
- **Posting**: one group of chunks. Every chunk is in exactly one posting. A posting is stored as a key prefix: its chunks live under `posting_id ‖ doc_id` keys in `{ns}_sparse_vector`, so reading a posting is one prefix scan.
- **Entry**: one such key. It holds all of one document's chunks in that posting. Search cost is counted in entries read.
- **Centre**: a vector stored in the store's centre table (`{ns}_ivf_centres`). Each posting has a **routing centre**: new chunks go to the posting whose centre is nearest, and a query reads the postings nearest to it. Each chunk code is also encoded *against* a centre (it stores the chunk's difference from that centre); that is the code's `centre_id`.
- **Target posting size** (`target_posting_size`, default 128 chunks): the size the partition aims for. A posting **splits** once it holds more than twice the target (the **split limit**, 256) and **merges** once it shrinks to `round(split limit × merge_ratio)` chunks or fewer (the **merge limit**, 22).
- **Read share**: the entries a search reads as a share of all entries in the store. Partition quality is compared at equal read shares, because a partition can always find more by reading more.
- **Recall@10**: the share of the true top 10 that a search returns, where the true top 10 is the same two-pass ranking computed exactly, on full-precision vectors, over every chunk. It measures what partitioning and probing lose. **nDCG** (normalised discounted cumulative gain) scores the final ranking against human relevance judgements.

### 3.2 Why each store builds its own partition

The groups must match the store's own text. A fixed set of centroids fitted on general text put 43% of SciFact's documents (a scientific-claims dataset) into one of 256 groups, so a query probing that group read nearly half the index. Smaller, on-topic postings find more of the right documents per entry read: in a simulation on FiQA (57,638 financial Q&A documents, 173,000 chunks), with the gemma model, recall@10 at 10,000 entries read was 0.975 with postings of target 128, 0.942 at target 1,024, and 0.811 with the general-purpose centroids.

So no centroid file is prepared, shipped or loaded. Each store starts with one posting and grows its partition from its own vectors, without the embedding service and without keeping any float vectors.

### 3.3 Overall design

```
enable semantic search
        │
        ▼
 one root posting (centre = zero vector)
        │   documents arrive; each chunk goes to its nearest active posting
        ▼
 ┌───────────── background maintenance, per store ─────────────┐
 │  posting > split limit  ──►  SPLIT into two  ──►  REASSIGN   │
 │                                                  neighbours  │
 │  posting ≤ merge limit  ──►  MERGE into a nearby posting     │
 └──────────────────────────────────────────────────────────────┘
        │
        ▼
 optional, by hand: REBUILD (k-means over every stored code)
```

**Insert.** A chunk's embedding is compared with every active posting's routing centre, filed under the nearest, and encoded against that posting's centre. If the posting is now over the split limit, the write wakes the maintenance task.

**Split.** The maintenance task rebuilds an approximate vector for every chunk of the posting from its 1-bit code (its *reconstruction*), then divides the posting in two with balanced 2-means (two-group k-means with a penalty that keeps the halves of similar size). The two halves become new postings whose centres are the **means** of their members; the old posting is retired. If the smaller half would be at or under the merge limit, the split is refused and the posting stays whole until its contents change (see [§3.5](#35-design-choices-and-trade-offs)).

**Reassign.** A split moves two centres, so some chunks nearby are now in the wrong posting. After each split, the task checks the chunks of the two halves and of the `reassign_range` (64) postings nearest the old centre. A chunk in one of the halves is a candidate if the old centre is nearer to it than its new one; a chunk in a neighbouring posting is a candidate if a new centre is nearer to it than both the old centre and its own posting's centre (SPFresh's two conditions). Each candidate moves to its nearest active posting, which may be where it already is.

**Merge.** Deletes and moves shrink postings. A posting at or under the merge limit is folded into the first of the 64 postings nearest its centre that has room (the two together stay under the split limit). The larger of the two is kept. A moved chunk that ends up farther from its new centre than its old one is routed again to its nearest posting.

**Rebuild** (by hand). `POST /admin/indices/{ns}/vector/rebuild` re-clusters the whole store: k-means over the reconstructions of every stored chunk (k-means++ on a 50,000-chunk sample, then 20 Lloyd iterations), with as many centres as there are active postings unless the request body sets `{"postings": n}`. Every chunk then moves to its nearest new posting, and ordinary maintenance splits whatever k-means left oversized. It makes no embedding calls and runs in the background (202; 409 while one is running).

**Codes keep their centre.** A moved chunk changes only its key: its code and its `centre_id` stay as they were, and search decodes every code through its own `centre_id`, never through the posting it is filed under. So no maintenance step re-encodes anything or needs the original text. Whole-document (Pass 2) codes are encoded against the zero vector (§2), so they never depend on the partition at all.

**Crash safety and concurrency.** The store has no multi-key transactions, so every operation is written as a plan in a WAL-backed journal (`{ns}_ivf_journal`) and carried out as single-key writes that are safe to repeat: create the new postings, copy each document's chunks to their destination, make the copies durable, then delete the originals and retire the old posting. Copies go to the vector index WAL, which is fsynced once per operation (the *barrier*) before any original is deleted; otherwise a crash could lose the copy and the original both. Before anything uses a store's partition (a write, a search, a delete, a rebuild), recovery finishes every operation that had published its new postings and abandons one that had not; a maintenance error makes the next user recover again. Searches and writes carry on during maintenance and never wait for it: each enters the store's routing epoch while it chooses postings and reads or writes them, and maintenance, after publishing a new snapshot of the partition (and before deleting any original), waits only for the searches and writes that entered before. Dropping a store or its vector index first stops its maintenance and waits for the operation in progress. During a move a chunk may briefly sit in two postings; MaxSim (§4) takes each document's best chunk, so a duplicate changes no score.

**Who runs it.** One maintenance task per process serves every store, at most `[semantic_search.maintenance] threads` (default 2) stores at a time and never one store twice. The 2-means runs on the blocking thread pool, and a rebuild's k-means on a pool of its own (`rebuild_threads`, 1 or 2), never on the pool that search scoring uses. Per-document moves run in batches of 16 documents, one batch at a time, because a batch holds those documents' write locks and every write to them waits.

**Probing at search time.** Pass 1 computes the distance from the query to **every** active posting's routing centre (nothing is pruned), then takes postings nearest first until their entries reach the probe budget (`probe_budget_entries`, default 70,000), reading at least `min_probes` and at most `max_probes` postings. The cost is T × C × D (query vectors × postings × dimensions); production sends one query vector, so it is one scan of the centres: about 9 µs at 256 postings, growing linearly with their number. Entry counts per posting are kept exactly in memory, loaded from `{ns}_sparse_vector_meta` at startup and updated by every write.

### 3.4 Parameters

Every store owns its partition settings: they are copied into its schema (`vector_index.maintenance` and `vector_index.search`) when it enables semantic search, from the server's `[semantic_search.maintenance]` and `[semantic_search.search_defaults]` (or the built-in defaults). A later change to the server config affects only stores enabled afterwards. A store's search settings can be changed later and overridden per request.

| Setting | Default | Allowed | What it controls |
|---|---|---|---|
| `target_posting_size` | 128 | 16–65,536 | Chunks per posting the partition aims for; split limit = 2 × this |
| `reassign_range` | 64 | 0–1,024 | Postings near a split re-checked for misplaced chunks (0: only the two halves) |
| `merge_ratio` | 10/118 ≈ 0.0847 | above 0, below 0.5 | Merge limit as a share of the split limit (SPFresh's ratio) |
| `skip_uncertain_moves` | `false` | | Move a chunk only when its code shows it is clearly nearer, by more than the code's error bound |
| `split_samples` | 1,000 | 2–1,000,000 | Balanced 2-means: chunks sampled per iteration |
| `split_init_trials` | 3 | 1–100 | Balanced 2-means: random starting pairs tried |
| `split_max_iters` | 100 | 1–10,000 | Balanced 2-means: iteration cap |
| `split_lambda_factor` | 100 | above 0 | Balanced 2-means: balance weight, λ ≤ 1 / (factor × samples) |
| `search.probe_budget_entries` | 70,000 | | Entries Pass 1 reads |
| `search.min_probes` / `max_probes` | 1 / 1,024 | | Bounds on the postings Pass 1 reads |
| `search.probe_budget_fraction` / `probe_budget_floor` | unset / 20,000 | fraction 0–1 | Optional budget that scales with the store: `clamp(fraction × entries, floor, probe_budget_entries)` |

The `split_*` settings are the values of SPFresh's reference implementation (SPTAG); at the default target a posting being split holds far fewer than 1,000 chunks, so every chunk takes part in every iteration. Two server-only settings sit under `[semantic_search.maintenance]`: `threads` (default 2, 1–64: stores maintained at once) and `rebuild_threads` (default 2, 1 or 2: a rebuild's k-means threads).

### 3.5 Design choices and trade-offs

Measured on SciFact (5,183 documents, about 23,000 chunks) and FiQA, with the gemma and qwen embedding models, in three insertion orders (file order, random order, and topic by topic). "Static k-means" is a partition fitted once on all the full-precision vectors, with the same number of postings: a strong fixed baseline.

| Choice | What it gains | What it costs |
|---|---|---|
| **Grow from one posting**, no seeding phase | Searchable from the first document; nothing to prepare; handles growth, topic drift and deletes the same way | Postings are placed one split at a time, so the partition trails static k-means. On FiQA recall@10 is 0.4–1.0 points behind at a 5% read share; on SciFact 0.9–1.8. Seeding with k-means on the first floats gained at most 0.3 points, for about 90 MB of staged floats |
| **Target 128 chunks** | More recall per millisecond than 256 (FiQA: 0.971 against about 0.966 at 3.7 ms) | About twice as many postings, so more probes to fill the same budget (603 against 293 at 70,000 entries, about 3.5 µs each: p50 14.9 ms against 13.8 ms) and more splits while indexing |
| **Mean centres**, not SPFresh's medoid (the member nearest the mean) | With 1-bit codes a medoid is one member's noisy reconstruction, and the partition fragments (FiQA: 12,000–24,000 postings, recall@10 0.58–0.69 at a 1% read). The mean averages the noise away, and beat the medoid even on exact vectors | None measured |
| **1-bit codes for maintenance**, no stored floats | No float copy of the store (FiQA: about 20 MB of codes instead of 530 MB of floats). A k-means on 1-bit reconstructions matched one on floats (recall@10 0.978 against 0.976) | Each split decides from approximate vectors; 2-bit codes would add at most about a point at read shares of 2% or less |
| **Codes keep their centre** when moved | No re-encoding, no embedding calls, no compounding error (re-encoding from a 1-bit code measured 0.861 against 0.918) | A moved code is a little less precise than a fresh encode against its new posting's centre. This is reported as *centre drift*; no ranking cost from it has been measured |
| **No replicas** (each chunk in one posting) | One write per chunk; SPFresh's 4–5 copies per chunk cost 15–20 key moves per insert | None measured: replicated partitions lost to unreplicated ones of the same size |
| **Reassignment over 64 neighbours** | SciFact's gap to static k-means at a 5% read narrowed from up to 3.5 points to under 2; when topics arrive one at a time, recall@10 at a 1% read rose by up to 10 points | About 2 key moves per inserted chunk (1.9–2.2 in random order, up to 2.8 topic by topic); FiQA indexing (embedding excluded) 47 s against 35 s without it, about 0.2 ms per document beside an embedding call of about 100 ms |
| **Eager merges**, with refused splits | A small posting is merged as soon as it appears, instead of costing a probe on every query that reaches it | Needs the refused-split rule: on SciFact, 2-means split a 262-chunk posting into 259 and 3, the merge sent the 3 back, and the two alternated about 900 times in one pass. A refused posting stays a little over the split limit until its contents change |
| **Error-bound gate off** by default | Keeps recall when topics arrive one at a time | On, it saves 12–17% of key moves for up to 0.9 points of recall@10 at a 1% read |
| **Exact scan of every centre** at search and maintenance time | No approximation in the most recall-sensitive step; microseconds at thousands of postings | Linear in the number of postings: at about 100,000 postings (roughly 17 million chunks at FiQA's average posting size) the centres would need an index of their own |
| **Rebuild only by hand** | A rebuild never runs unasked; it costs FiQA 143–151 s on 2 threads | A grown partition can trail a fresh k-means. A rebuild gained up to 4.4 points of recall@10 at a 1% read share and up to 1.3 at 5%; at the default budget nDCG did not change in any run. Use the recall check (§3.7) before and after to see whether it helped |

Deletes do not wear the partition down. Deleting half the documents at random, or whole topics, left FiQA within ±0.8 points of what the partition reached before the deletes. **Known limitation:** SciFact with qwen, after half of its topics were deleted, lost 2.5 points; a small store (87 postings left) is the worst case.

### 3.6 Where minnal differs from SPFresh

| SPFresh | minnal | Why |
|---|---|---|
| Postings hold full vectors | Postings hold 1-bit codes; maintenance works on their reconstructions | No float copy of the store |
| A split's centres are medoids | The means of the two halves | Medoids of noisy reconstructions fragment the partition |
| Up to 8 replicas per vector | One posting per chunk | Replicas did not pay at equal entries read |
| Merges are found lazily, when a search probes a small posting | Merges are found eagerly, when a write or delete leaves a posting small | Maintenance sees every size change; a small posting costs a probe on every query reaching it |
| A split always writes both halves | A split whose smaller half is at or under the merge limit is refused | Without it, eager merges and splits undo each other in a loop |
| A posting that will not split is truncated | It is kept whole and tried again when it changes | minnal never drops data |
| Nearest centres found by an approximate graph | Every centre scanned exactly | Microseconds at this scale |
| A moved vector keeps its value | A moved code keeps its `centre_id`; only its key changes | Re-encoding a 1-bit code compounds its error |
| — | Optional error-bound gate on move decisions | Uses the `error_bound` every code already stores |

### 3.7 Watching the partition

| Endpoint | Shows |
|---|---|
| `GET /admin/indices/{ns}/vector/partition` | Postings, entries, posting-size spread, splits, merges and key moves since start, and a running rebuild's progress. With `?codes=true` it also reads every code to report centre drift and the mean error band |
| `GET /admin/indices/{ns}/vector/postings` | Every live posting with its size, centre, parent and how it was created (`root`, `split`, `rebuild`); cursor-paginated |
| `GET /admin/indices/{ns}/vector/recall-check` | How much of a full read's top 10…100 a normal search finds within its budget, on a sample of real queries from the query-embedding cache (or, with fewer than 20 cached, stored chunks). Verdict `read_in_full`, `ok`, `below_target` (overlap@10 under `min_overlap`, default 0.95) or `empty` (no chunks stored). Pass the same `seed` before and after a rebuild to compare |
| `GET /admin/indices/{ns}/vector/history` | Past recall checks, `?codes=true` reads and rebuilds (kept `partition_history_ttl_secs`, default 30 days) |
| `POST /admin/indices/{ns}/vector/rebuild` | Starts a rebuild (body optional: `{"postings": n}`) |

Use the recall check, not centre drift, to decide on a rebuild. Drift measures how precise the codes are, and a rebuild makes it higher (every code ends up in a posting with a new centre). What a rebuild changes is which postings hold which chunks, and that is what the recall check measures. A low overlap that does not rise after a rebuild means the probe budget, not the partition, is the limit.

---

## 4. Two-Pass Search Execution

`service/mod.rs` implements a two-pass ANN search:

```
Raw query text
  → embedding cache lookup (system_qemb_cache TTL namespace)
  → on miss: embed_query — ONE POST to {base_url}/embedding/{model}/query
       · payload[0]: whole query → 1 embedding q
                     (Pass 2 dense vector AND Pass 1's single MaxSim vector;
                      queries are not chunked)
  → cache it (TTL configurable; default 1 day)

Pass 1 — Sparse (SingleBit), ColBERT MaxSim over document chunks:
  rank every posting by the distance of q to its routing centre; take them
    nearest first until their entries reach the probe budget
    (several query vectors → union of their posting sets)
  scan_sparse_clusters_batch(probed ids)        ← one LSM pass per layer for all
                                                  probed clusters, then one value-log
                                                  pread per entry
  ── SCORING_GATE permit (see "Concurrency" below) ──
  flatten every probed entry (cluster, d, chunks) into one list
  for each entry (rayon parallel over ENTRIES, not clusters):
    apply doc_filter (RoaringBitmap predicate) — skip if fails
    row[entry][i] = max_j SingleBitEstimator(q_i, d_j)   ← best chunk in this entry
  sort kept entry indices by doc_id; for each run of one document's entries:
    per_vector_max[i] = max over the run of row[entry][i]
    S(q, d) = Σ_i per_vector_max[i]              (= max_j ⟨q, d_j⟩ for one vector)
  select_nth_unstable → keep first_pass_sparse_search_top_k (unordered)
  ── release permit; free the scanned entries on the blocking pool ──

Pass 2 — Dense (MultiBit):
  get_dense_entries_batch(candidate doc_ids)    ← one batch read
  ── SCORING_GATE permit ──
  for each candidate (rayon parallel):
    score with MultiBitQuanDotProductEstimator (zero centre) against the single
      whole-query dense embedding
  build top-k min-heap
  return sorted descending
```

### Where a query's time goes

Almost all of a warm query's time goes into Pass 1's sparse scan: merging the
LSM layers for each probed cluster, then one value-log read per entry, then
decoding and scoring. Pass 2 is small and roughly fixed, since it re-scores a
capped candidate set. So latency follows **how many sparse entries the probed
postings hold**, which the probe budget caps. On FiQA (57,638 documents,
173,000 chunks) with gemma and the default 70,000-entry budget, a warm query
takes about 15 ms.

That makes small, on-topic postings the main lever: they find more of the right
documents per entry read. Centroids fitted on general text, rather than on the
store's own vectors, put most of a specialist corpus in a few clusters (43% of
SciFact's documents fell in one of 256), so each probe read a large share of the
index. That is why every store builds its own partition (§3).

### Concurrency

Pass 1 and Pass 2 each score in parallel on rayon's global pool, which every
concurrent search shares. A rayon worker waiting inside a `join` can steal
another search's job and cannot return to its own search until that one
finishes; under steady load those nestings pile up and requests stall.
`SCORING_GATE` allows at most 2 scoring sections at a time, process-wide, and
admits waiters in FIFO order. On FiQA with 32 concurrent clients this took p99
latency from 9.7 s to 308 ms and raised throughput from 110 to 125 queries per
second. Rules for changing it are in `CLAUDE.md` → *Concurrency: the scoring gate*.

### Why two passes?

- **SingleBit** entries are very compact and cluster-contiguous in storage, making prefix-scan over a cluster fast. The 1-bit score is coarse but sufficient to narrow the candidate set from millions to ~1000.
- **MultiBit** entries are stored directly by `doc_id` for O(1) lookup, and their higher-precision scores produce final ranked results with low quantisation error.

### ColBERT MaxSim aggregation (Pass 1)

Pass 1 uses **ColBERT MaxSim** to aggregate scores across query vectors and document chunks:

```
S(q, d) = Σ_i  max_j ⟨q_i, d_j⟩
```

where `i` iterates over query vectors and `j` over document chunks. For each query vector, the best-matching chunk of the document wins (inner `max`); those per-vector bests are then **summed** (outer `Σ`).

**In production there is one query vector** — the whole-query embedding — so this reduces to `S(q, d) = max_j ⟨q, d_j⟩`: a document scores by its single best-matching chunk. `search()` keeps the general `Σ_i` form, which would let a document score by matching several query fragments. Splitting queries that way was evaluated on BEIR (`query-embedding-report.md`) and lost: 4-word fragments carry little meaning on their own and, with a tight first-pass cut, let relevant documents drop out (ArguAna candidate recall 0.870 against 0.991 for the whole query), while costing 96 query vectors and 171 probed clusters per query against 1 and 32. Sentence-sized query chunks matched the whole query on quality at extra cost. So queries are not chunked.

Chunks whose cluster is not probed contribute **0** to their query vector's term (rather than −∞), so documents are never penalised for having chunks in far-away clusters.

### Dense re-ranking (Pass 2)

Pass 2 scores each candidate against a **single whole-query dense embedding** — symmetric with the document's whole-text dense vector. There is no aggregation across multiple query vectors: each `doc_id` is scored once by one `MultiBitQuanDotProductEstimator`, so each document appears exactly once in the final output.

### Predicate filtering

`doc_filter` is an optional closure `Fn(&[u8]) -> bool` applied **only in Pass 1**, per document, before scoring. Documents that fail the filter are excluded from both passes. Pass 2 operates on the already-filtered `sparse_ranked` list and never re-evaluates the predicate.

This is how filtered semantic search (`POST /stores/{ns}/semantic-search/filtered`) works: it evaluates the index predicate into the set of matching doc IDs and passes a membership check as the `doc_filter` closure. That combines "semantically similar to X" with "and `status = 'active'`" in one query, with predicate evaluation kept outside the vector pipeline.

---

## 5. Storage Layout

Companion KVStore namespaces per semantic-search-enabled store (`vector_kv.rs`):

| Namespace | Key | Value |
|---|---|---|
| `{ns}_sparse_vector` | `posting_id (4B BE u32) ‖ doc_id` | rkyv `Vec<VectorIndex>` (SingleBit only) |
| `{ns}_sparse_vector_meta` | `doc_id` | `text_hash (8B) ‖ n (2B BE) ‖ n × (posting_id (4B BE) ‖ chunks (2B BE))` |
| `{ns}_dense_vector` | `doc_id` | rkyv `Vec<VectorIndex>` (MultiBit only) |
| `{ns}_ivf_centres` | `centre_id (4B BE)` | the centre, `f32` × D; append-only |
| `{ns}_ivf_postings` | `posting_id (4B BE)` | routing centre, `centre_id`, state (`Active`, `Draining`, `Retired`), parent, origin |
| `{ns}_ivf_journal` | operation id (8B BE) | a maintenance operation in progress (split, reassignment, merge, rebuild) |

`_sparse_vector` uses a composite key with a posting prefix so a `scan_prefix(posting_id)` efficiently retrieves all SingleBit vectors in a posting. `_sparse_vector_meta` is a reverse lookup used to find and delete stale sparse keys when a document is updated or deleted, and its chunk counts give each posting's exact size; its text hash tells reconciliation which text the vectors came from. `_dense_vector` is keyed directly by `doc_id` for O(1) fetch in Pass 2. The three `_ivf_` namespaces hold the store's partition and are WAL-backed: unlike codes, they cannot be regenerated by re-embedding.

---

## 6. Query Embedding Cache

Query embeddings are cached in a system-wide TTL namespace `system_qemb_cache` shared across all doc-store namespaces. Keys are the model name, a NUL byte, then the UTF-8 query string, so switching `model` never serves the previous model's vectors; values are packed big-endian `f32` vectors. The TTL is **configurable** via `[semantic_search] query_embedding_cache_ttl_secs` and **defaults to 1 day** (86400 s) — once it elapses, stale entries are evicted automatically by the TTL worker. Cache misses fall back to the embedding service transparently.

**Durability — no-WAL populate, WAL-backed clear.** Populating an entry on a cache miss is no-WAL (`put_no_wal`): the cache is TTL-bounded and fully regenerable, so a dropped populate just produces a future cache miss that re-fetches from the embedding service, and staying off the WAL removes a per-populate fsync from the query hot path (the latency motivation for caching in the first place). Clearing the cache (`DELETE /admin/indices/vector/query-cache`) uses WAL-backed deletes, like the vector-index cleanup deletes (see §7, *Durability guarantees*): the periodic no-WAL flush tick has usually already persisted the cached entries, so a clear tombstone lost to a crash before the next tick would resurrect entries the operator explicitly cleared.

---

## 7. Async Write Path & Durability

Vector indexing is **asynchronous and decoupled from document writes**. A document write never contacts the embedding service inline; it durably enqueues work that a background worker drains later. This keeps writes fast and the system resilient to an embedding service that is slow, restarting, or entirely down.

### Write path

1. `put` / `kv_put` writes the document, then extracts the embedding field and enqueues a `(namespace, doc_id, text)` entry in the durable `system_pending_vec_index` KV namespace as a separate, independent write — the document write and the enqueue are not atomic. The document write returns immediately without contacting the embedding service. A crash between the two writes leaves the document un-indexed (reconciled by re-index), never acked-but-lost.
2. `VecIndexWorker` (`minnal_db/src/doc_store/vec_index_worker.rs`) consumes the queue in the background — see [Background worker](#background-worker).

### Embedding service availability

The embedding service being unreachable is **never fatal**:

- **At startup**, the server probes `GET {base_url}/healthcheck` (the configured model must be listed and ready), then embeds a probe text through both of the model's endpoints and checks the vectors have `embedding_dim` dimensions. On failure it logs an error and **starts anyway**; the failure surfaces later at call time. The background worker starts regardless of the probe result.
- **At query time**, a search that cannot reach the service returns an error to the caller (`EmbeddingFailed`) — no crash, no partial index corruption.
- **At index time**, the worker simply fails the affected queue entries and retries them on later passes (see [Retry & exhaustion](#retry--exhaustion)). Because the queue is durable, no indexing work is lost while the service is down — it drains once the service returns.

### Background worker

`VecIndexWorker` runs as a single background tokio task. On startup it first **drains any surviving queue entries** (crash recovery), then loops, woken by a write `notify` signal or a 30 s fallback poll. Each pass:

- Scans all pending queue entries and **skips exhausted ones** (`retry_count ≥ max_retries`).
- Groups the remainder by namespace and visits them in **round-robin** order so no single namespace can starve others.
- Calls `embed_document` (one embedding call per document — whole text + chunks in a single ordered batch) for up to `concurrency` entries at once.
- **On success:** writes the `VectorIndex` entries across all three companion namespaces, then **completes** the entry. These are independent writes (not atomic), so a crash between them just re-processes the entry idempotently on the next pass.
- **Completion is conditional.** A pass works from a snapshot of the whole queue, so an upsert or delete of a document can land between the snapshot and that entry's completion, for minutes during a bulk load. Completion (and failure bookkeeping) is an atomic `merge` on the queue key that only acts if the stored entry is still the one the worker took (same kind and text):
  - a newer upsert's entry is left for the next pass, so its text still gets embedded (otherwise the entry would be deleted and the stale vectors kept);
  - a failure never overwrites a newer entry with the older text;
  - if the entry has become a `Clear` tombstone, the document was deleted mid-embed, so the worker deletes the vectors it just wrote and retires the tombstone.
- **Clear tombstones.** Deletes, and upserts whose embedding text is now empty, write a `Clear` entry *before* deleting the vectors synchronously. The worker retires it by deleting the vectors again (idempotent) and removing it, unless a re-upsert has replaced it meanwhile. An empty-text upsert of a document that never had vectors or a pending entry writes nothing.
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
| Re-index one entry | `POST /admin/indices/{ns}/vector/queue/{doc_id}/retry` | Resets that entry → `retry_count = 0`, atomically, keeping its kind and text |
| Re-index all failed | `POST /admin/indices/{ns}/vector/reindex-failed` | Resets every exhausted entry in `{ns}` → `retry_count = 0`, the same way |
| Full re-index | `POST /admin/indices/{ns}/vector/reindex-all` | **Deletes** existing exhausted entries, then re-enqueues every document at `retry_count = 0` |

### Queue entry format

Queue keys encode `(namespace, doc_id)` (length-prefixed namespace ‖ doc-id bytes) so rapid successive writes to the same document overwrite the entry — the worker makes exactly one dual-embedding call for the most-recent text. A queue value is:

```
0x03 ‖ kind (1 B: 0 = embed, 1 = clear) ‖ retry_count (4 B BE) ‖ error_len (4 B BE) ‖ error_bytes ‖ text_bytes
```

The leading byte is a format version. The decoder also reads versions 1 and 2, which lack the kind (and, in version 1, the error) and are treated as embed entries. The admin queue listing shows each entry's `kind`.

### Durability guarantees

#### Queue durability & crash recovery

The pending-embed queue lives in a standard minnal_db namespace, so every enqueue and dequeue is WAL-backed and survives a crash. On startup the worker drains any entries that were still queued before entering its normal notify-driven loop, so work that was in flight at the moment of a crash is simply re-attempted.

The two companion-namespace vector **payloads** — sparse chunks (`{ns}_sparse_vector`) and dense (`{ns}_dense_vector`) — skip the main WAL: the quantised payloads are bulky and fully reconstructable by re-embedding, so a per-chunk WAL fsync plus a second copy in the WAL would be pure overhead. They go to a separate log, the **vector index WAL** (one per database), with no fsync per write; the worker makes a whole batch durable with one fsync of it before completing the batch's queue entries, and after a crash the log's unretired records are replayed. The sparse-meta (`{ns}_sparse_vector_meta`), which lists every posting holding the document's chunks, is WAL-backed and written before the chunks. A crash before the batch's fsync can still lose its vectors; those entries were not completed, so the worker redoes them, and reconciliation (below) covers the rest. The design is in `Vector-Index-WAL-Design.md`.

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

## 8. Configuration

The server's `[semantic_search]` table holds only what concerns the embedding service:

| Parameter | Default | Description |
|---|---|---|
| `embedding_service_url` | `http://localhost:8001` | Base URL of the external embedding service. |
| `query_embedding_cache_ttl_secs` | `86400` | TTL (seconds) for cached query embeddings in `system_qemb_cache`. Default is 1 day. |
| `embedding_request_timeout_secs` / `embedding_connect_timeout_secs` | `30` / `10` | Caps on one embedding request, and on its connect phase. |
| `[[supported_models]]` | none | Optional allow-list (`name`, `dimension`). Without it, any model the service serves. |
| `[maintenance]` `threads` / `rebuild_threads` | `2` / `2` | Stores maintained at once; k-means threads of a rebuild (1 or 2), on a pool of its own. |
| `partition_history_ttl_secs` | `2592000` | How long recall checks, code reads and rebuilds stay in the partition history (30 days). |

Everything that shapes one store's index lives in its schema's `vector_index`. `[semantic_search.maintenance]` and `[semantic_search.search_defaults]` only give the values a schema is filled with when the store first enables semantic search:

| Setting | Default | Description |
|---|---|---|
| `embedding_model` | `gemma` | Model requested from the service on every request (`/embedding/{model}/…`). Fixed once set. |
| `embedding_dim` | `768` | Dimension of the vectors the service returns. Fixed once set. |
| `chunking.window_size` / `sliding_size` | `4` / `2` | Sentences per document chunk, and the step between chunks (queries are not chunked). Fixed once set. |
| `quantisation` | 1 / 8 bits, zero Pass-2 centre | Code widths and the rotation seed; read-only. |
| `search.probe_budget_entries` | `70000` | Pass 1 probes the nearest postings until their entries reach this. |
| `search.min_probes` / `max_probes` | `1` / `1024` | Bounds on postings probed. |
| `search.first_pass_top_k` | `1000` | Candidates kept after Pass 1 for dense re-ranking. |
| `search.top_k` | `100` | Results returned. |
| `maintenance.*` | see §3.4 | How the store's partition splits, reassigns and merges. |

Search settings can be changed later (`update_vector_search` amendment) and overridden per request.

### Profiling search latency

`real_kv_search_profile` in `minnal_db/src/vector_kv.rs` (an `#[ignore]`d test)
runs the real two-pass `search()` over a real `minnal_db`-backed store (synthetic
vectors and centres, but real LSM + value-log `pread` I/O) and phase-times
posting choice, Pass-1 sparse-scan I/O, Pass-2 dense-fetch I/O, the scoring
remainder, and the whole query, at several fixed probe counts and chunks per
document. It needs no embedding service.

```sh
cargo test -p minnal_db --no-default-features --features doc-store,semantic-search --lib real_kv_search_profile --release -- --ignored --nocapture
```

Pass-1 sparse-scan I/O is 54–88% of `search()` at 8 chunks per document and
grows roughly linearly with the entries scanned. The SIMD dot products are a
minority of the cost. Pass 2 re-scores a fixed `first_pass_top_k` candidates, so
its share shrinks as more is read. Recall and nDCG against entries read are
measured on frozen BEIR embeddings by the vector bench
(`semantic_search/vector_bench`, design doc M0).

---

## Key Files

| Component | File |
|---|---|
| Embedding service client + two-pass search | `minnal_db/src/semantic_search/service/mod.rs` |
| RaBitQ quantisation (encode + decode) | `minnal_db/src/semantic_search/quantisation/rabitq/mod.rs` |
| `VectorIndex` struct + `VectorKvStore` trait | `minnal_db/src/semantic_search/index/vector_index.rs` |
| Distance estimators (SingleBit, MultiBit) | `minnal_db/src/semantic_search/index/distance_estimator.rs` |
| A store's partition (postings, centres, routing epoch) + probing by budget | `minnal_db/src/semantic_search/cluster/layout.rs` |
| Split, reassignment, merge, rebuild and their journal | `minnal_db/src/ivf_split.rs`, `minnal_db/src/ivf_split/` |
| Recall check and partition history | `minnal_db/src/ivf_split/recall_check.rs`, `ivf_split/history.rs` |
| Background maintenance task and crash recovery | `minnal_db/src/doc_store/ivf_maintenance.rs` |
| Per-store vector-index settings (defaults, validation) | `minnal_db/src/doc_store/vector_settings.rs` |
| Balanced 2-means and k-means | `minnal_db/src/semantic_search/cluster/split.rs`, `cluster/kmeans.rs` |
| Composite key encoding (posting ‖ doc_id) | `minnal_db/src/semantic_search/index/composite_key.rs` |
| Scoring, coarse-assignment and end-to-end benchmarks | `minnal_db/benches/bench_distance_estimation.rs` |
| Vector KV storage (three namespaces) + query cache | `minnal_db/src/vector_kv.rs` |
| Latency profiling harness (see §8) | `minnal_db/src/vector_kv.rs` (ignored test) |
| Async vector-index background worker | `minnal_db/src/doc_store/vec_index_worker.rs` |
| Document store | `minnal_db/src/doc_store/store/` |

---

## Acknowledgements

### SPFresh

The way each store grows and maintains its partition (§3) follows SPFresh's LIRE protocol: split a posting that grows too large, reassign the nearby vectors the split leaves misplaced, and merge postings that shrink too small, all in place and while the index keeps serving reads and writes. minnal takes LIRE's split, reassignment and merge rules, its reassignment range and merge ratio, and the balanced 2-means parameters of its reference implementation, SPTAG. Where minnal departs from them, and why, is listed in §3.6.

> **SPFresh: Incremental In-Place Update for Billion-Scale Vector Search**
> Yuming Xu, Hengyu Liang, Jin Li, Shuotao Xu, Qi Chen, Qianxi Zhang, Cheng Li, Ziyue Yang, Fan Yang, Yuqing Yang, Peng Cheng, Mao Yang
> *ACM SOSP '23*
> https://arxiv.org/pdf/2410.14452 (the protocol is §3, *LIRE Protocol Design*)

The reference implementation (`ExtraDynamicSearcher.h` for the update protocol, `BKTree.h` for balanced k-means) is part of SPTAG:

> https://github.com/microsoft/SPTAG

### RaBitQ

The 1-bit and multi-bit codes (§2), their error bound, and the random rotation they are computed in come from RaBitQ:

> **RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search**
> Jianyang Gao, Cheng Long
> *ACM SIGMOD '24*
> https://dl.acm.org/doi/10.1145/3654970
