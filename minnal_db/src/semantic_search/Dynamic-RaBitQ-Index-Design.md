# Dynamic RaBitQ index: design and milestones

**Status:** draft under review; first-round decisions recorded (see *Decisions*
at the end) · branch `dynamic-rabitq-index` · 2026-10-03

**Goal.** Remove the pre-trained centroid files. Each namespace should build and
maintain its own IVF partition as documents arrive. Search quality (nDCG, recall)
and query latency must not regress at any step.

**Approach.** Five milestones (M0–M4). Each changes one thing and ends at a
benchmark gate, so a regression shows up in the milestone that caused it.

| Milestone | What changes | Changes partitioning? | Gate (summary) |
|---|---|:---:|---|
| **M0** Benchmark | Frozen-embedding harness plus a baseline on `main` | — | Reproducible: two runs give identical quality numbers |
| **M1** Rotation | Random orthogonal rotation of codes and query (`FhtKacRotator`) | No | Quality ≥ baseline; latency within noise |
| **M2** Namespace-owned index | Model and dimension move into the schema; centres and postings become per-namespace data, seeded from the model's file; dense codes use a zero centre; probing by entry budget | No (same seeds) | M2a and M2b give byte-identical results to M1; M2c and M2d ≥ M1 |
| **M3** Dynamic partitions | No centroid file: grow from one posting, split, reassign, merge | **Yes** | Within 2 pts of corpus-fitted centroids; largest posting under 1% of entries |
| **M4** Rebuild and clean-up | Re-cluster from codes; drop centroid files and config | Yes | Rebuild uses no embedding calls; within 1 pt of fitted |

The research behind this design (production systems, papers, the 109k-passage
experiment, worked examples) is in the *Dynamic Clustering for minnal's IVF +
RaBitQ* doc. The design below starts from its recommendations. It splits them
into smaller steps and adds what the doc leaves open: the benchmark,
per-namespace model selection, and how a namespace bootstraps from nothing.

---

## Terms used here

- **Posting** (posting list): the set of chunk entries stored under one key
  prefix in `{ns}_sparse_vector`. Today a posting is a cluster, and the prefix
  is the `cluster_id`.
- **Routing centroid**: the vector a posting is found by. Inserts go to the
  nearest one; queries probe the nearest few. It may move.
- **Code centre** (centre): the vector `c` a code is encoded against
  (`x = c + r`; the code stores `r`). Once a code is written, its centre must
  never change.
- **Rotation `P`**: a random orthogonal matrix. Codes store `Pᵀr`, and queries are
  rotated once as `q' = Pᵀq`.
- **Entries scanned**: the number of Pass-1 entries a query reads. Latency
  tracks this almost linearly (Semantic-Search-Architecture.md, *Where a query's
  time goes*). It is the cost axis every milestone is compared on.
- **Exact pipeline**: the same two-pass ranking computed on full-precision
  floats with no partitioning (exact MaxSim over all chunks, then exact dense
  rerank). It is the ground truth for ANN recall.

---

## What today's code ties together

One `cluster_id` currently does three jobs: it routes (which posting), it is
the code centre, and it is the key prefix. The centroid file must therefore
exist before the first insert and never change. Concretely:

- `rabitq::index_embedding` picks the nearest cluster and encodes against its
  centroid (`quantisation/rabitq/mod.rs`).
- Pass 1 builds one estimator per probed cluster from that cluster's centroid
  (`service/mod.rs`, `cluster_estimators`).
- Pass 2 decodes each dense entry through
  `cluster_index.clusters.get(&cluster_id)?` (`service/mod.rs`, around line 610).
  A missing id silently drops the document.
- `ClusterIndex` is **process-global** (`SemanticSearchContext.cluster_index`),
  loaded from `cluster_path`. The **model is engine-wide** too
  (`SemanticSearchConfig::model_name`). So every namespace shares one model and
  one partition, and a model/file mismatch is only a startup warning.

---

## M0 — Benchmark harness and baseline

Every later milestone is judged against M0's numbers, so the harness has to be
deterministic, quick to rerun, and sensitive to exactly what the later
milestones change.

### Frozen embeddings

Embed each corpus **once** and store the f32 vectors. Every milestone then
indexes the same vectors through the real write path. There are no
embedding-service calls, no GPU nondeterminism, and a re-index takes minutes
instead of the ~40 minutes the service needs for FiQA.

| Dataset | Docs | Chunks (window 4, slide 2) | Judged queries |
|---|---:|---:|---:|
| SciFact | 5,183 | 21,774 | 300 |
| FiQA | 57,638 | 172,998 | 648 |

(Counts from the step-0 dump in `splade-lite/data/ivf_step0/`. That dump is
f16, so M0 re-dumps f32 through the running gemma service. Production quantises
f32, and f16 rounding would shift sign bits near zero.)

Files go outside git, under `work/bench/{dataset}/gemma/`: `dense.f32`,
`chunks.f32`, `chunk_doc.u32`, `queries.f32`, plus a small JSON manifest (ids,
dims, model, chunking params, service commit).

### Harness

Extend `semantic_search/beir_eval.rs` rather than writing a new tool:

1. **Split `embed_document`** into *fetch embeddings* and
   `index_embeddings(config, ns_index, dense, chunks) -> Vec<VectorIndex>`. This
   is a pure refactor and the harness's only hook into production code. Later
   milestones change `index_embeddings` and the harness follows automatically.
2. Add a `MINNAL_BEIR_EMBEDDINGS=dir` mode: load frozen vectors, index them
   through `index_embeddings` + `upsert_vectors`, and run queries through the
   real `search()`.
3. **Ground truth from floats**, computed once and cached next to the vectors:
   - `exact_pass1[q]`: exact MaxSim (`max_j ⟨q, d_j⟩`) over all chunks, top 1000.
   - `exact_final[q]`: `exact_pass1` reranked by exact dense `⟨q, d⟩`, top 100.
4. **Insertion order** is a knob: `corpus` (file order), `shuffled` (seeded),
   and `drifting` (docs sorted by their nearest bundled centroid, so topics arrive
   one at a time; this is the adversarial case for M3).

### Metrics (all per query, then aggregated)

| Group | Metric | Why |
|---|---|---|
| Relevance (qrels) | nDCG@10, MRR@10, Recall@100 | What users see; already in `beir_eval` |
| ANN fidelity | Pass-1 recall@{100,1000} vs `exact_pass1`; final recall@10 vs `exact_final` | Isolates index damage from model quality; more sensitive than nDCG |
| Estimator | RMSE and bias of Pass-1 and Pass-2 estimates vs exact inner products, on a fixed sample of (query, entry) pairs; share of errors inside `error_bound` | What rotation (M1) and centring (M2b) change directly |
| Cost | entries scanned, postings probed, Pass-1/Pass-2 ms, total ms p50/p95/p99 (warm, sequential) | Latency gate |
| Partition shape | postings, largest posting share, p99 posting size, coefficient of variation | Catches skew (today: 43% of SciFact in one cluster) |
| Footprint | bytes in `_sparse_vector`, `_dense_vector`, `_meta`, centre tables; indexing throughput (docs/s, embedding excluded) | Space goal; M2 adds 4 bytes per entry |

Every configuration runs as a **sweep** over `n_probes ∈ {4, 8, 16, 32, 64, 128,
256}` (from M2c on, also over entry budgets). The comparison is the
**recall-vs-entries-scanned curve**, not a single point. Later milestones change
how many postings exist and how big they are, so the same `n_probes` stops
meaning the same cost.

Output: `work/bench/results/{milestone}/{dataset}.{md,tsv,json}`, plus a
`compare` step that prints paired per-query deltas against the baseline
(win / loss / tie counts, mean Δ with a bootstrap 95% CI). Per the report style
we use, a mean alone is not enough.

### Online check (milestone ends only)

The harness is in-process. At the end of each milestone, also run the release
binary on a **scratch server and scratch DB** (another port) loaded with the same
corpora, and use `minnal_tools search_load` at c = 1, 8 and 32. This catches what
the in-process harness can't: the query cache, the REST path, and the
`SCORING_GATE` under load.

### Gates (confirmed)

- **Quality:** nDCG@10 Δ ≥ −0.005 with the bootstrap CI covering 0 or above, on
  both datasets. ANN recall@10 not lower by more than 0.5 pt at equal entries
  scanned.
- **Latency:** p50 and p95 at the production setting within +5% (or +0.3 ms,
  whichever is larger). Three repeat runs on an idle host.
- **Footprint:** within +5% unless the milestone states otherwise.
- **M0 exit:** baseline numbers for `main` (`4397d16`) on both datasets, and two
  back-to-back runs giving identical quality numbers.

---

## M1 — Rotation

### What changes

One seeded `FhtKacRotator` per namespace. Codes are computed in rotated space,
and the query is rotated once per search. Routing, centroids, keys and
partitioning are unchanged.

### The key simplification: rotate the inputs, not the formulas

Every term the quantisers and estimators compute is an inner product among the
residual `r`, the centre `c`, the code `s` and the query `q`
(`get_index_calculation_data`, `quantise_using_single_bit`, both
`estimate_from_parts`). A rotation preserves inner products, so the existing
code stays correct if it is handed **rotated vectors consistently**:

- **Index:** `quantise(Pᵀx, Pᵀc, bits)`, with no change to `quantise`. The
  multi-bit `addition_factor` uses `⟨c, s⟩`, where `s` lives in rotated space,
  so the centre must be rotated too. Keep a `Pᵀc` copy per centroid (256 × 768
  rotations ≈ 0.4 ms per namespace at load).
- **Query:** `q' = Pᵀq` once. `query_to_centroid_dot_product` can use plain `q`
  and `c` (`⟨q, c⟩ = ⟨q', Pᵀc⟩`). `packed_ip_best`, `multi_bit_dot_best`,
  `query_sum` and `scaled_query_sum` must use `q'`.
- **Routing:** unchanged, in original space with plain `q`.
- **Query cache:** keeps the raw embedding (it is keyed by model and shared
  across namespaces). Rotate per search: ≈1.5 µs against a ~17 ms FiQA query.

The bug to guard against is mixing the two spaces: a rotated code scored with
an unrotated `Σq`, or the reverse. The tests below target exactly that.

### `rotation.rs` review (the supplied implementation)

I built, tested and linted `~/Downloads/rotation.rs` (440 lines, no
dependencies) unchanged, on this host (AVX-512, rustc 1.96):

- **Passes:** its 5 tests, `clippy --all-targets -D warnings`, and `rustfmt
  --check` with minnal's `rustfmt.toml`.
- **Algebra checked by hand:** the inverse is correct. Each normalised step is a
  symmetric involution (sign flip, `H/√n`, Kac butterfly `/√2`), so applying them
  in reverse order inverts the rotation, and the four unnormalised Kac steps are
  covered by the final `0.25`. The AVX-512 and AVX2 lane masks and permutes match
  the `i ^ 1`, `i ^ 2`, `i ^ 4`, `i ^ 8` butterflies.
- **Extra tests I ran:** every path (portable, AVX2, AVX-512) for both rotate
  and unrotate, at dims 8, 96, 384, 768, 1024 and 1536, matched the portable path
  to within 1e-6. Over 2,000 random 768-d unit vectors, **no coordinate changed
  sign** between the dispatched and portable paths. That matters because a
  1-bit code is the sign pattern.

Gaps to close when it moves into `semantic_search/rotation.rs`:

1. **The format is unpinned.** A change to `splitmix64`, `ROUNDS`, the mask
   layout or the block order would silently change every code. Add a golden test:
   seed 42, a fixed input, and the expected output stored in `test_data/`.
2. **The AVX2 dispatch is untested on AVX-512 hosts**, and nothing compares the
   SIMD inverse with the portable one. Add the per-path test above (it calls the
   `#[target_feature]` methods directly).
3. **`new()` panics** on an odd or short dim. Validate the dimension (even,
   ≥ 8) at schema validation (it moves into the schema in M2a; until then, at
   config validation) so the panic is unreachable, and keep the assert as a
   guard.
4. **No NEON path.** Apple Silicon uses the auto-vectorised portable path.
   Codes are portable across machines (identical within 1e-6, no sign flips
   above), so only speed differs; measure it when a Mac is available.

### Tests (beyond the rotator's own)

- **Estimator error:** on 2,000 gemma (query, chunk) pairs from the frozen
  set, rotated-code RMSE ≤ unrotated RMSE for 1-bit and for 8-bit, bias ≈ 0, and
  ≥ 95% of 1-bit errors inside `error_bound` (the bound assumes the rotation).
- **Consistency:** quantise rotated and score rotated must equal the
  unrotated estimator applied to `(P x, P c, P q)` within float tolerance. This
  catches a missed `q'`.
- **End to end:** a 2,000-doc in-memory index with exhaustive probing gives
  final top-10 equal to `exact_final` on ≥ 97% of positions (the threshold gets
  set from the M0 baseline value, not guessed).

### Index format and rollout

M1 changes what every stored code means. Add `index_format: u16` to a
per-namespace vector-index metadata record (`{ns}_vector_meta`, a new
companion namespace, WAL-backed). It holds `index_format` and `rotation_seed`
and grows in M2. A namespace whose format doesn't match the binary's is refused
for search with a clear "re-index required" error. Silently mis-scoring it is the
alternative, and it's worse. Greenfield rules apply: existing indexes are
not migrated, they are rebuilt.

`reindex-all` re-enqueues documents **without clearing** old vectors, so a
namespace would serve a mix of old and new codes while it runs. Rebuilding
after a format change is therefore *drop vector index → re-enable*, which clears
it first.

Seed: stored per namespace, defaulting to a fixed constant. A per-namespace seed
gives no quality gain over one fixed seed, but recording it makes the format
explicit.

### Gate

nDCG@10 and ANN recall ≥ M0 on SciFact and FiQA; estimator RMSE lower; latency
within noise. The 109k WordLlama experiment saw only +0.3 pt from rotation, and
gemma has a stronger common direction, so a larger gain is plausible but
unproven. **If M1 shows no gain and no loss, keep it anyway.** M3's
reconstruction-based maintenance depends on it: without rotation, sign bits of
similar vectors are correlated.

---

## M2 — Namespace-owned model, centres and postings (still static)

M2 moves everything a namespace's index depends on into that namespace. No
partition moves yet: postings are seeded from the model's centroid file, so
M2a's results can be checked byte-for-byte against M1.

### M2a — Model and dimension in the namespace schema

**Change.** Add `embedding_model: Option<String>` and
`embedding_dim: Option<u32>` to `DocStoreSchema` and `KvStoreSchema`.

- **Validation:** both are required when semantic search is enabled (on a doc
  store, when any field is an embedding field). The model must be non-empty and
  is lower-cased on save. The dimension must be even and ≥ 8, which is what the
  rotator needs. Both are ignored when semantic search is off, the same way
  `embedding_fields` is ignored.
- **Fixed once selected.** A store's model and dimension are set the first time
  semantic search is enabled (at create, or by the first `EnableVectorIndex` /
  `AddEmbeddingAttribute`) and can never change afterwards, including across a
  drop and re-enable of the vector index. A later amendment may omit them, or
  repeat the same values; a different value is rejected. Vectors from different
  models live in unrelated spaces, so there is no in-place switch. A
  "drop and re-index with another model" operation can be added later as a
  separate feature.
- **REST:** the create-store and enable-vector-index bodies carry
  `embedding_model` and `embedding_dim`.
- **Engine config:** remove `SemanticSearchConfig::model_name` and
  `embedding_dim`, and the `[semantic_search] model` and `embedding_dim` keys.
  What stays engine-wide is service-level: URL, timeouts, chunking, bits,
  `top_k`, probing.

**Where the model and dimension are read today and must come from the
namespace instead:**

| Site | Today | After |
|---|---|---|
| `embed_document`, `embed_query` (`service/mod.rs`) | `config.model_name`, `config.embedding_dim` | `model` and `dim` arguments from the namespace (`dim` is also the `dimensions` sent in the request) |
| `search()` dimension check (`service/mod.rs`) | `cluster_index.dim()` | the namespace's `embedding_dim` |
| Query-cache read (`get_cached_query_embedding`) | `ctx.config.embedding_dim` | the namespace's dimension; the cache key gains the dimension too, so the same model at two dimensions never shares an entry |
| Query cache (`doc_store/store/query.rs`) | `ctx.config.model_name` | namespace's model; the key already includes the model, so namespaces with different models never share entries |
| Vector worker (`vec_index_worker.rs`) | context config | the queue entry's namespace → schema → model |
| `check_embedding_service` (startup, `main.rs`) | probes one model at one dimension | probes each **distinct** (model, dim) pair used by an existing semantic namespace; also probe at create/enable time (a 404 `Unknown model` or a dimension mismatch fails the request; an unreachable service only warns, since the queue tolerates outages) |
| `centroid_mismatch` (`minnal_db_api/src/config.rs`) | compares `cluster_path` with the model's bundled file | **removed**: in M2b the seed file is chosen by the model, so there is nothing to mismatch |
| `beir_eval.rs`, `config_report.rs`, `routes/stores.rs`, tests | config | schema |
| `QUICKSTART.md`, `minnal_db/QUICKSTART.md`, READMEs, `config/sample.toml`, `service/scripts/examples/docs.sh` | `[semantic_search] model` / `embedding_dim` | `embedding_model` / `embedding_dim` in the store body (examples rerun per the docs rules) |

**Existing stores.** Schemas on disk without these fields fail validation.
Existing data is not a concern (greenfield): such stores are recreated.

**Gate.** Pure refactor: harness results byte-identical to M1. Add tests for
schema validation (missing, empty, mixed-case model; odd, small or zero
dimension; set on a non-semantic store), for immutability (a different value is
rejected on every amendment path, including after a vector-index drop), and for
two namespaces with different models in one process (each embeds with its own
model and keeps its own cache entries).

### M2b — Per-namespace centres and postings, seeded from the model's file

**New per-namespace data** (companion namespaces, added to `COMPANION_SUFFIXES`
so drop/hide/cleanup cover them; all **WAL-backed**, because unlike codes they
cannot be rebuilt by re-embedding):

- `{ns}_ivf_centres`: `centre_id (u32 BE) → [f32; D]`. Append-only; never
  rewritten or deleted while any code points at it. `centre_id` is allocated
  densely so the hot path can index a `Vec`.
- `{ns}_ivf_postings`: `posting_id (u32 BE) → { routing_centroid [f32; D],
  centre_id, state: Active | Draining | Retired }`. Entry counts are kept in
  memory and recomputed at load from the key prefixes; they feed M2c and M3.
- `{ns}_vector_meta` (from M1): plus `seeded_from` (the file and its hash).

**Bootstrap.** When a namespace's vector index is created and the postings
table is empty, seed it from `{centroid_dir}/{model}/clusters.json`: posting id
= centre id = cluster id, and routing centroid = centre. `centroid_dir` is
engine-wide (default `service/embedding_support`) and replaces `cluster_path`.
After seeding, the namespace never reads the file again, so the bundled file can
later change without corrupting existing namespaces. A model with no bundled
file, or a file whose dimension differs from the namespace's, fails the enable
request during M2; from M3 on, such a namespace simply starts with no file.

**In memory.** A per-namespace `NamespaceIvf` replaces the global
`Arc<ClusterIndex>`. It holds the rotator, centres (plain and rotated) in a
dense `Vec`, and a routing snapshot (contiguous centroid matrix, ids, counts).
It is published as `parking_lot::RwLock<Arc<Snapshot>>`: a search clones the
`Arc` under a brief read lock, and the maintenance writer in M3 swaps it.
(`arc-swap` would do the same, but it would be a new dependency; we can discuss
it if read-lock contention ever shows in a profile.) A registry maps namespace
→ `NamespaceIvf`; it is loaded on open and dropped with the store.

**`VectorIndex` gains `centre_id: u32`.** For chunk codes it equals the posting
id at seeding time, but the code is decoded via `centre_id` and **never** via
the key prefix. Pass 1 needs `⟨q, c⟩` per entry rather than per probed cluster:
precompute `qc[centre_id]` once per query for the centres present in the probed
postings (each snapshot keeps posting → centre set). The hot loop then does one
indexed load per entry. **This sub-step's latency gate is the one to watch.**

**Gate.** Search results byte-identical to M1 (same seeds, same ids, same
centres). Hot-path latency within noise. Footprint +4 bytes per entry (measured,
not assumed; rkyv padding may round it).

### M2c — Dense (Pass-2) codes against a zero centre

Dense entries are fetched by `doc_id` and never routed. At 8 bits the choice of
centre changed recall@10 by 0.2 pt at most (0.992–0.994 across five choices),
and a zero centre removes the routing-id hazard entirely. With `c = 0` the
`⟨q, c⟩` term vanishes, so Pass 2 needs **one** estimator per query and loses its
`est_cache` HashMap. Dense entries get `centre_id = ZERO_CENTRE` (reserved). If
the dense codes ever drop to 4 bits, switch back to a real centre (worth ≈1.5 pt
there).

Also: a dense entry whose `centre_id` is unknown is a **counted, logged error**,
not a silent `?` drop.

**Gate:** Pass-2 estimator RMSE within 1.3× of M2b, nDCG@10 ≥ M2b − 0.002,
latency ≤ M2b.

### M2d — Probe by entry budget

Replace the fixed `n_probes` with `probe_budget_entries`: probe the nearest
postings in order until the sum of their entry counts reaches the budget
(bounded by `min_probes` and `max_probes`). On today's static postings this
should reproduce the M2c recall-vs-entries curve point for point. It stops
latency from drifting as M3 changes posting counts and sizes, and it fixes the
cost of a query at a known budget. Default the budget to what `n_probes = 64`
scans on FiQA at M0.

**Gate:** the curves overlay M2c within noise; at the default budget, latency
≤ M2c and recall ≥ M2c.

---

## M3 — Dynamic partitions (no centroid file)

### Choosing the approach

**Undecided.** When a namespace starts clustering, and when postings split, is
still open. The options are kept open below, and M3-pre's simulation is the
first step towards choosing. Two options were proposed; with the M2 split in
place they can combine into one design.

| Option | How it works | Assessment |
|---|---|---|
| **A. Brute force until critical mass, then cluster** | Keep full embeddings until N chunks, flat-scan until then, then k-means and index | **Candidate bootstrap.** Exact k-means on real floats gives the best first centroids. But A alone doesn't handle growth past N or topic drift, so it would still need a split mechanism. |
| **B. Searchable from doc 1; centroids move until critical mass, then sealed** | Segments whose centroids settle, then freeze | **Candidate, at posting granularity, possibly without sealing.** Sealing exists only because codes depend on the centroid. Once codes have their own frozen centre (M2), a routing centroid can keep moving forever at the cost of a key move, never a re-encode. Segments that queries must each probe (Lucene/Milvus style) would add a per-segment probe cost and a merge policy on top of the LSM; postings are already global key prefixes, which avoids both. |
| **C. Split as you grow (SPFresh/LIRE)** | One posting from the start; split at 2× target with 2-means on code reconstructions; reassign neighbours; merge small postings | **Likely steady state.** It is the only option here that handles unbounded growth and drift without the embedding service. |

**One candidate lifecycle** (a combination of the above, to be tested against
the alternatives in M3-pre, not a decision):

1. **Flat phase** (chunks < `N_seed`, default ~10k; tuned in M3-pre). One
   posting, centre = zero. Search is an exhaustive Pass 1, which is fine at this
   size (the ivf_dynamic simulation put ~17k entries scanned at ≈5 ms on FiQA). The f32 chunk embeddings are also kept
   in `{ns}_ivf_staging` (no-WAL; ~30 MB at 10k chunks, re-embeddable).
2. **Seed** (at `N_seed`, background). Run exact k-means++ on the staged floats
   (`k = N_seed / target_posting_size`). Write centres and postings. **Re-encode**
   the staged chunks exactly against their new centres (the floats are still
   there), move their keys, delete the staging floats, publish the snapshot. This
   is option A's quality without its growth problem.
3. **Grow** (forever after). Each insert is routed with its full-precision vector
   and encoded against the target posting's pinned centre. The routing centroid
   tracks a running mean. Postings past `2 × target` split; neighbours are
   reassigned; postings under `target / 4` merge. All maintenance works from 1-bit
   codes and their reconstructions, and **never re-encodes a code** (re-centring a
   1-bit code was measured worse than leaving it, 0.861 vs 0.918).
4. **Rebuild** (M4) when drift indicators say the partition has degraded.

Tiny namespaces stay in phase 1 indefinitely, which is correct behaviour for
them; their staging floats are bounded by `N_seed`.

### M3-pre — Settle the numbers by simulation before coding

The ivf_dynamic simulator (`splade-lite/data/ivf_dynamic/simulate_dynamic.py`)
already models "keep the code centre, move only the key". Run it on the M0 f32
dumps, with rotation and the M2 centre rules, over three insertion orders
(`corpus`, `shuffled`, `drifting`). Compare:

- (i) grow from one posting with no staging (option C alone)
- (ii) the candidate lifecycle above, with a float-seeded bootstrap (A + C)
- (iii) flat until `N_seed`, then k-means from codes with no staging (B + C)
- (iv) static k-means fitted on the whole corpus (upper bound)
- (v) the bundled gemma file (today)

Sweep `target_posting_size ∈ {128, 256, 512, 1024}`, `N_seed ∈ {2k, 10k, 30k}`,
and reassign-neighbours `k ∈ {0, 2, 8}`. Report recall vs entries scanned
(including the early-life curve at 1k, 5k, 10k and 20k chunks), moves per insert
(write amplification), and largest posting share.

The results go to review before M3a starts; the bootstrap and split policy are
chosen then. A natural rule is to prefer the simplest variant within 1 pt of the
best at equal entries scanned. This step needs no Rust, and it stops us building
the wrong variant.

### M3a — Grow and split (split only, no reassign)

- **Background maintenance task**, one writer per namespace, woken by
  posting-size events. It runs off the request path, on `spawn_blocking` plus its
  own small thread budget, not the rayon pool searches share (the
  `SCORING_GATE` rules).
- **Split protocol** (copy, then delete; never a hole):
  1. Read posting P's entries. Reconstruct `x̂ = c + scaling_factor ·
     P(2b − 1)` (unrotated), and run balanced 2-means.
  2. WAL: append two centres; write postings P1 and P2 (`Active`); mark P
     `Draining`; write a split record `{P → P1, P2, state: Planned}`.
  3. **Routing epoch.** Inserts route and write while holding a per-namespace
     tokio `RwLock` read guard, inside the existing `lock_doc_vectors`. The split
     takes the write guard just long enough to publish a snapshot where P is
     `Draining`: still probed by search, never chosen by inserts. After that
     point, no insert that routed with the old snapshot is still writing.
     Split record → `Published`.
  4. Move each P entry to P1 or P2 (nearest routing centroid by code estimate).
     Under the doc lock: put the new key, update `{ns}_sparse_vector_meta`,
     delete the old key. Then rescan P once; it must be empty. P → `Retired`;
     split record → `Done`.
- **Duplicates are harmless** in the window (MaxSim takes each document's
  max), and holes cannot happen (copy before delete).
- **Recovery:** on open, finish any `Published` split (step 4 is idempotent)
  and abandon any `Planned` one. Orphan centres are harmless. A startup check
  moves any entry found under a `Retired` prefix.

**Gate (SciFact and FiQA, empty namespace, no file, all three orders):**
recall within 2 pts of (iv) at equal entries scanned; largest posting < 1%
(today 43% on SciFact); nDCG@10 ≥ M2d − 0.005. Plus these tests:

- a **kill-during-split** test at each step boundary (fault injection), next to
  `racing_upsert_and_delete_leave_no_orphaned_cluster_keys`
- racing inserts and deletes against a split: no lost or orphaned keys, and the
  meta stays consistent
- search results during a split: a superset of before or after, never missing a
  document present in both

### M3b — Reassign (LIRE) and merge

Neighbour reassignment with the `k` chosen in M3-pre, and merging of postings
below `target / 4`. Measured separately because each move costs a put plus a
tombstone in the LSM. **Gate:** recall gain ≥ the cost the simulation predicted;
compaction and write-amplification overhead reported; no latency regression.

### M3c — Bootstrap phase

Whichever bootstrap M3-pre's review chooses: for example a flat phase, or a
float-seeded k-means with exact re-encode, or none if (i) is enough. **Gate:**
the early-life recall curve (measured at 1k, 5k, 10k and 20k chunks) ≥ (i).

---

## M4 — Rebuild from codes, then remove the files

- **Rebuild:** an admin endpoint (`POST /admin/indices/{ns}/vector/rebuild`)
  re-runs k-means over 1-bit reconstructions (optionally seeded from the 8-bit
  dense vectors) and rewrites keys only. Triggered by hand, or by drift
  indicators (posting-size coefficient of variation, mean `‖x̂ − routing
  centroid‖` per posting, probes needed to reach the budget). **Gate:** a FiQA
  rebuild makes zero embedding-service calls and lands within 1 pt of (iv).
- **Delete:** `centroid_dir`, the bundled `service/embedding_support/*/clusters.json`
  files, `cluster_path` remnants, the M2b seeding code, and the "Cluster
  centroids" sections of the docs. No import path is kept.
- **Optional later**, each gated on its own run: SOAR spilling, a second
  centroid level above ~15–20k postings, 2-bit chunk codes.

---

## Risks

| Risk | Where | Mitigation |
|---|---|---|
| Mixed old/new codes in one namespace | M1, M2 | `index_format` check refuses search; rebuild by drop + re-enable, not `reindex-all` |
| Rotated/unrotated mix-up (`q` vs `q'`, `c` vs `Pᵀc`) | M1 | Consistency test against explicitly rotated inputs; estimator-RMSE gate |
| Hot-path cost of per-entry centre lookup | M2b | Dense `Vec` index; latency gate on this sub-step alone |
| Split races with inserts and deletes; crash mid-split | M3a | Routing epoch + doc lock; WAL split record; fault-injection tests |
| Write amplification from reassignment | M3b | Measured in M3-pre; `k` is a knob (0 = off) |
| Early-life quality before the first splits | M3 | M3-pre decides float seeding; flat scan while small |
| Two models in one process | M2a | Per-namespace probe and cache key; test with gemma + qwen |

## Decisions (2026-10-03)

| # | Question | Decision |
|---|---|---|
| 1 | Gate thresholds (nDCG −0.005, ANN recall −0.5 pt, latency +5%) | Accepted |
| 2 | Where `embedding_dim` lives | In the namespace schema, next to `embedding_model` (M2a) |
| 3 | Changing a store's model | Not supported: model and dimension are fixed once selected. A drop-and-re-index operation may be added later |
| 4 | Existing data | Not a concern; stores are recreated after format changes |
| 5 | When to start clustering and when to split | Open; M3-pre's simulation comes first and its results are reviewed before M3a |
| 6 | Bundled centroid files | Deleted in M4 |
