# Dynamic RaBitQ index: design and milestones

**Status:** draft under review; first-round decisions recorded (see *Decisions*
at the end) · branch `dynamic-rabitq-index` · 2026-10-03

**Goal.** Remove the pre-trained centroid files. Each namespace should build and
maintain its own IVF partition as documents arrive. Search quality (nDCG, recall)
and query latency must not regress at any step.

**Approach.** Five milestones (M0–M4), plus two correctness steps (M0-1, M0-2)
found while designing M3. Each changes one thing and ends at a
benchmark gate, so a regression shows up in the milestone that caused it.

| Milestone | What changes | Changes partitioning? | Gate (summary) |
|---|---|:---:|---|
| **M0** Benchmark ✓ | Frozen-embedding harness plus a baseline on `main` | — | Reproducible: two runs give identical quality numbers |
| **M0-1** Durable re-embed ✓ | The vector worker completes a queue entry only after its vectors are flushed | — | Crash regression test passes; end-to-end indexing throughput within 5% |
| **M0-2** Write-path crash audit ✓ | Every multi-step vector write traced for crash safety, with a shared crash-test helper; gaps fixed | — | One crash test per path; all pass |
| **M1** Rotation ✓ | Random orthogonal rotation of codes and query (`FhtKacRotator`) | No | Quality ≥ baseline; latency within noise |
| **M2** Namespace-owned index | Model, dimension, chunking, code widths and search settings move into the schema; centres and postings become per-namespace data, seeded from the model's file; dense codes use a zero centre; probing by entry budget | No (same seeds) | M2a and M2b give byte-identical results to M1; M2c and M2d ≥ M1 |
| **M3** Dynamic partitions | No centroid file: grow from one posting, split, reassign, merge; then optional per-namespace re-encoding (`stored` or `service`) | **Yes** | Within 2 pts of corpus-fitted centroids; largest posting under 1% of entries |
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

Postings and centres are numbered separately, written `P…` and `C…` below.
Today they share one number (the `cluster_id`), which is why they are easy to
confuse.

### Worked example: a split

Toy 2-D vectors. Six chunks share one posting: three about cricket, three about
interest rates.

**Before.** The centre table holds `C7 = (0.5, 0.5)`. Posting P7 has routing
centroid (0.5, 0.5) and encodes new inserts against C7. Every entry stores
`centre_id 7`:

| Chunk | Topic | Vector `x` | Key | Residual `x − c` (not stored) | Stored code: signs of `x − c` | `centre_id` |
|---|---|---|---|---|---|---|
| a | cricket | (0.9, 0.2) | `P7‖a` | (0.4, −0.3) | `+ −` | 7 |
| b | cricket | (1.0, 0.1) | `P7‖b` | (0.5, −0.4) | `+ −` | 7 |
| c | cricket | (0.8, 0.0) | `P7‖c` | (0.3, −0.5) | `+ −` | 7 |
| d | rates | (0.1, 0.9) | `P7‖d` | (−0.4, 0.4) | `− +` | 7 |
| e | rates | (0.2, 1.0) | `P7‖e` | (−0.3, 0.5) | `− +` | 7 |
| f | rates | (0.0, 0.8) | `P7‖f` | (−0.5, 0.3) | `− +` | 7 |

An entry stores what it stores today: the sign of each residual coordinate (96
bytes at 768 dimensions, after rotation) plus three scalar factors. The float
residual is shown only to make the arithmetic visible. The only new per-entry
field is the 4-byte `centre_id`; the centre vectors themselves (3 KB each at 768
dimensions) live once per namespace in the centre table.

A cricket query scans all six entries, half of them about rates.

**The split.** 2-means over P7's entries (computed from their codes) finds a
cricket group around (0.9, 0.1) and a rates group around (0.1, 0.9). Two
centres are appended, `C8 = (0.9, 0.1)` and `C9 = (0.1, 0.9)`. Postings P8 and
P9 are created and P7 is retired. Each entry's **key** moves; its value is
copied byte for byte.

**After.** The centre table holds C7, C8 and C9; **C7 stays**, because six
codes still point at it.

| Posting | Routing centroid | Centre for new inserts | State |
|---|---|---|---|
| P7 | — | — | Retired |
| P8 | (0.9, 0.1) | C8 | Active |
| P9 | (0.1, 0.9) | C9 | Active |

| Chunk | Key | Stored code | `centre_id` |
|---|---|---|---|
| a, b, c | `P7‖…` → `P8‖…` | unchanged | 7 |
| d, e, f | `P7‖…` → `P9‖…` | unchanged | 7 |

A new cricket chunk `g = (0.95, 0.15)` arriving later routes to P8 and is
encoded against C8: residual (0.05, 0.05), stored as `+ +`, `centre_id 8`,
key `P8‖g`. So P8
holds codes with two different centres, and each entry names its own.

A cricket query now routes to P8 and scans four entries (a, b, c, g) instead of
six, none of them about rates. It scores a, b and c with `⟨q, C7⟩ + estimate
from bits`, exactly as before the split, and g with `⟨q, C8⟩ + …`.

What each part buys:

- **The split makes search cheaper:** fewer entries scanned, all on topic. No
  code is rewritten.
- **New centres make new codes more precise:** g's residual is about 0.07 long
  against about 0.5 for a, and the 1-bit estimate's error grows with that
  length.
- **Old codes stay valid but slightly less precise.** Rewriting them would need
  the original vector (an embedding call) or re-centring from the 1-bit code,
  which measured worse than leaving them alone (0.861 vs 0.918).

A centre can be removed only when no entry points at it any more: after its
chunks are deleted or re-embedded. At 768 dimensions it costs 3 KB, so removing
unreferenced centres is a small clean-up job for M4.

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

The harness is `semantic_search/vector_bench/` (`#[cfg(test)]`, `#[ignore]`d
tests). It reuses `metrics/beir_eval.rs`'s readers and scoring, and its module docs give
the commands.

1. **`embed_document` is split** into the service call and
   `index_embeddings(config, cluster_index, dense, chunks) -> Vec<VectorIndex>`.
   This pure refactor is the harness's only hook into production code. Later
   milestones change `index_embeddings` and the harness follows automatically.
2. **`vector_bench`** loads the frozen vectors (dumping them through the service
   on the first run), indexes them through `index_embeddings` +
   `upsert_vectors` into a fresh database (default `DbConfig`, 8 docs in flight),
   reopens and compacts it, and runs queries through the real `search()`.
   Documents the service refuses are left out of the dump and listed in its
   manifest (see *Finding* below).
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
| Cost | entries and chunks scanned, total ms p50/p95/p99 (warm, sequential, 3 repeats), queries whose results differ between repeats | Latency gate. Pass 1 and Pass 2 are not timed separately: `search()` has no timing hook, and adding one is not worth a production change |
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

### Baseline (measured 2026-10-03)

gemma served by llama.cpp (`embeddinggemma-300M-Q8_0`, embedding service
`b71a853`, which truncates inputs longer than the context), bundled gemma
centroids, default `DbConfig`, corpus insertion order. SciFact measured at
minnal `54bea36`, FiQA at `d27a65a` (same search and indexing code). Results and per-query
data: `work/bench/results/m0-baseline/`.

| | SciFact | FiQA |
|---|---:|---:|
| Docs indexed / chunks | 5,183 / 22,787 | 57,600 / 172,998 |
| Docs left out (empty text) | 0 | 38 |
| nDCG@10 at 64 probes | 0.7908 | 0.4737 |
| ANN recall@10 at 64 probes | 0.9813 | 0.9927 |
| Pass-1 recall at 64 probes / all clusters | 0.797 / 0.836 | 0.805 / 0.808 |
| Entries scanned at 64 probes / all clusters | 5,587 / 8,197 | 69,491 / 89,690 |
| Latency p50 / p95 at 64 probes | 3.08 / 3.39 ms | 14.61 / 16.91 ms |
| Largest cluster (share of chunks) | 36.2% | 14.8% |
| Pass-1 / Pass-2 estimator RMSE | 0.0267 / 0.00030 | 0.0233 / 0.00024 |
| Footprint (sparse + meta + dense) | 7.0 MiB | 67.9 MiB |

What it shows:

- **Partition skew.** At 64 of 256 probes a FiQA query already scans 77% of the
  index, and SciFact's largest cluster holds over a third of its chunks. This is
  the problem M3 targets.
- **Pass 1 loses candidates even when it scans everything.** With every
  cluster probed, the 1-bit Pass 1 keeps only 84% (SciFact) and 81% (FiQA) of
  the exact top-1000 candidates. Final ANN recall stays above 0.98 because Pass 2
  recovers most of it, but this is the error M1's rotation should reduce.
- **Reproducible.** A second run (`m0-repro`) gave identical quality numbers for
  every query on both datasets; p50 latency moved by at most 2%.
- **Embedding service.** gemma on llama.cpp scores the same as the earlier
  PyTorch service through the server pipeline on `main` at 64 probes (SciFact
  0.7851 → 0.7849, FiQA 0.4714 → 0.4718; paired CIs include 0). It used to
  refuse payloads longer than the model's context (21 FiQA documents could not
  be indexed at all); embedding service PR #11 truncates them instead, as the
  PyTorch service did, and the FiQA baseline above includes them.

---

## M0-1 — Make a re-embed durable before its queue entry is completed

A correctness fix in today's vector pipeline, found while designing the M3
journal. It comes before M1 because every later milestone re-embeds documents,
and the M3 journal relies on the same rule.

### The problem

The vector worker handles a queue entry in `vector_kv::finish_embed`:

1. `upsert_vectors` deletes the document's stale cluster keys (**WAL-backed**,
   durable when the call returns), then writes the new sparse keys, the new
   sparse meta and the new dense entry (**no-WAL**, durable only after the next
   memtable flush).
2. `complete_entry` removes the queue entry with a `merge` (**WAL-backed**,
   durable at once).

So the queue entry can be gone for good while the vectors it vouches for are
still only in memory. A crash then (a process kill is enough: unflushed no-WAL
writes live only in the memtable) leaves an already-indexed document like this:

| What | After the crash |
|---|---|
| Queue entry | Gone (its removal was durable) |
| New sparse keys, new meta, new dense entry | Lost |
| Stale cluster keys | Deleted (the deletes were durable) |
| Old meta and old dense entry | Still on disk, flushed at the original index |

`has_complete_vector_index` finds the old meta and dense entry and reports the
document complete, so startup reconciliation skips it. The document keeps the
**old text's** vectors, minus the chunks whose clusters changed, until something
re-embeds it. Only the on-demand validating reconcile (`check_bytes = true`)
notices the missing keys, and nothing at all notices the stale vectors.

A first-time index is not affected: with no old meta the document reads as
incomplete and reconciliation re-enqueues it.

### The fix: complete only after a flush, in batches

The queue is already a redo log for embedding; it just must not be truncated
before the work it records is durable.

- The worker splits `finish_embed` into **write** (`upsert_vectors`) and
  **complete** (`complete_entry`, plus the existing `Cleared` and
  vanished-document checks, unchanged).
- After writing, an entry joins a pending list. When the list reaches a batch
  size (a constant, e.g. 256) or the pass ends, the worker:
  1. flushes the vector namespaces those entries wrote to (`{ns}_sparse_vector`,
     `{ns}_sparse_vector_meta`, `{ns}_dense_vector`) that hold unflushed no-WAL
     writes, once each;
  2. only if the flush succeeded, completes every entry in the list.
- A flush failure completes nothing: the entries stay queued and the next pass
  redoes them, which is safe because re-embedding the same text writes the same
  vectors.
- A crash before the flush leaves the queue entries in place, so the restarted
  worker re-embeds them. The stale-key deletes that did survive are harmless:
  the re-run writes the full new set.
- `Clear` entries need no barrier: `process_clear` and its tombstone removal are
  all WAL-backed.

**New internal API.** `AsyncDb::flush_no_wal_writes(namespaces)` (crate-internal):
on a blocking thread, for each named namespace that `has_unflushed_no_wal_writes`,
call `KVStore::flush_memtable_to_level0`. That flush already fsyncs the value log
before the L0 file counts as durable (`SyncValueLogBeforePersist`, installed on
every store); a test confirms it for these namespaces.

**Cost.** One memtable flush per vector namespace per batch, rather than per
document, so more L0 files during bulk indexing (FiQA: ~57.6k documents ÷ 256 ≈
225 flushes per namespace) for the compaction worker to merge. Completion is
delayed by up to one batch; search is not affected, since the vectors are
readable as soon as they are written.

### Tests and gate

- **Regression test (fails today):** index doc x and flush; re-embed x with
  vectors whose sparse chunk lands in a different cluster; crash with
  `std::mem::forget(db)` before any flush; reopen. Expect the queue entry for x
  to still be there, and processing it to leave exactly the new vectors. Today
  the entry is gone and x keeps its old dense entry.
- A crash **after** the barrier but before completion: on reopen the entry is
  re-processed and the result is the same vectors.
- A flush failure (injected) completes nothing.
- The existing R1–R3 race and crash tests keep passing; the conditional
  completion they cover is unchanged, only deferred.
- **Gate:** the tests above, and end-to-end indexing throughput (embedding
  call included, as in production) within 5%. The flush cost itself is
  measured by `vector_bench_worker_completion`: the same frozen documents
  through write-then-complete, once completing each entry at once (the old
  behaviour) and once flushing every 256 entries first.

**Measured (2026-10-03).** With embedding excluded, so that only the flushes
differ: SciFact 2,455 → 2,444 docs/s (−0.4%), FiQA 2,541 → 2,085 docs/s (−18%,
about 225 flushes of ~22 ms each). End to end the embedding call
dominates: the service embeds about 10 docs/s, so a batch of 256 takes ~25 s
against one ~22 ms flush, about 0.1%. Vectors are searchable as soon as they are
written, so the later completion does not delay search.

**Status: passed (2026-10-03).** Regression tests pass and fail without the fix;
end-to-end throughput cost about 0.1%.

---

## M0-2 — Crash audit of the vector-index write paths

M0-1 came from tracing one write path. The same class of bug (a WAL-backed
write made durable before the no-WAL data it depends on) may exist in others.
M0-2 traces every multi-step vector-index write, records what a crash at each
step can lose and what redoes it, and adds a crash test per path. Any gap found
is fixed inside M0-2, with a regression test that fails before the fix.

It is an audit, not new infrastructure. The redo-only journal stays in M3a,
where its first real user (split, merge, reassign) defines what it must record.
Today's paths already have redo sources: the vector queue, the `Clear`
tombstone, and the reindex progress record.

### Paths to audit

| Path | Redo source today | What to check |
|---|---|---|
| Re-embed (`finish_embed` → `upsert_vectors`) | Vector queue | Fixed by M0-1; its tests join this suite |
| Delete / clear (`clear_vectors` → `process_clear`) | `Clear` tombstone, written first | All writes WAL-backed; confirm a crash at each step leaves the tombstone or no vectors |
| Vanished-document cleanup in `finish_embed` | Next worker pass? | A crash after the vector write and before its `delete_vector`: is anything left that re-checks the document? |
| Startup reconciliation (`enqueue_embed_if_absent`) | Itself, at the next start | Its cheap completeness check passes stale vectors (M0-1's case); confirm that after M0-1 nothing reaches that state |
| `reindex-all` | `vector_reindex.json` progress record | Resume after a crash mid-enqueue; old and new codes mixed while it runs |
| Drop vector index / drop store | Registry-first ordering | No sidecar data or queue entries left behind after a crash at each step |
| Query-cache clear | WAL-backed deletes | Already reasoned in `semantic_search/CLAUDE.md`; one test to pin it |

For each path the audit records the write sequence, which writes are WAL-backed
and which no-WAL, the crash points between them, and what restores consistency.
The table goes into `semantic_search/CLAUDE.md` next to the existing durability
notes.

### Shared crash-test helper

One helper, reused later by the M3a split tests:

1. run a scripted sequence of steps up to step N;
2. crash with `std::mem::forget(db)`, which loses every unflushed no-WAL write,
   as a real process crash does;
3. reopen, then run what production runs after a restart: startup
   reconciliation and the worker's next pass;
4. check the invariants:
   - every sparse key a document has is listed in its meta;
   - no vectors remain for a deleted document;
   - an indexed document's vectors match its current text (the dense entry's
     tag, as the existing R1–R3 tests check);
   - no queue entry is lost while its work is undone.

Each path's test runs the helper for every N.

### Gate

One crash test per path, each covering every crash point, all passing; any gap
found fixed with a test that failed before the fix. No change to indexing
throughput beyond M0-1's.

### Findings and fixes (2026-10-03)

Five gaps, each fixed with a test that fails without the fix (checked by undoing
the fix and rerunning). Search results are byte-identical to the M0 baseline on
both datasets; the codes themselves did not change.

| # | Gap | Consequence | Fix | Test |
|---|---|---|---|---|
| 1 | The meta and the chunk keys are no-WAL writes to different namespaces, which flush independently | A key durable while its meta is lost: a later delete misses it for good, a reused id inherits it | Meta written WAL-backed, after the stale deletes and before the new keys | `delete_racing_a_reembed_survives_a_crash_anywhere` |
| 2 | A crash between a document write and its embed enqueue | New text saved, old vectors kept, nothing queued; they looked complete forever | The meta records a hash of the embedded text; reconciliation compares it with the current text (no count short-circuit) | `reembed_survives_a_crash_anywhere`, `test_reconcile_reembeds_vectors_of_stale_text` |
| 3 | The vanished-document cleanup ran after the queue completion | Vectors of a deleted document kept, with nothing left to remove them | Cleanup runs before the completion | `embedding_a_vanished_document_survives_a_crash_anywhere` |
| 4 | Startup reconciliation ran concurrently with the worker's crash-recovery pass | A delete cut short left a document unindexed until the next restart | Reconciliation waits for the worker's first pass | `startup_reconciliation_runs_after_the_crash_queue_is_processed` |
| 5 | An interrupted vector-index drop was never finished | Vector data and queue entries of a dropped index kept; the worker recreated its namespaces | Finished at open, before any worker starts | `an_interrupted_vector_index_drop_is_finished_at_open` |

One more, not crash-related, found on the same path: a reindex record never left
`"running"` (nothing wrote `"complete"`), so every second `reindex-all` on a store
was refused with 409. Concurrency is now an in-memory claim held while the
reindex enqueues, and the record is brought up to date when read (`"complete"`
once the queue drains, `"failed"` if a crash cut the enqueue short). Tests:
`a_finished_reindex_does_not_block_the_next`,
`a_reindex_interrupted_by_a_crash_does_not_block_the_next`.

Paths checked and already safe: the delete path (`clear_vectors` →
`process_clear`, all WAL-backed, covered by `delete_survives_a_crash_anywhere`)
and the query-cache clear (pinned by `test_cache_clear_survives_crash_before_flush_tick`).

**Cost.** One WAL fsync per document upsert (the meta): indexing throughput,
embedding excluded, is within run-to-run noise on this NVMe host (FiQA 15,287 →
15,415 docs/s). Startup reconciliation now reads every document: 1.44 s for
57,600 entries, once per start, in the background (it was 0.08 s with the count
short-circuit, which could not see stale vectors).

**Status: passed.**

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

Gaps to close when it moves into `semantic_search/quantisation/rotation.rs`:

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

### Rotation seed

M1 uses one fixed seed (`ROTATION_SEED`), held by the process-wide `ClusterIndex`
together with a rotated copy of every centroid. A per-namespace seed needs
per-namespace rotated centres, which arrive with M2b's per-namespace index, so
the seed record (`{ns}_vector_meta`) moves there. The choice of seed does not
affect quality.

The new code format is simply the format from M1 on. There is no migration and
no format-version check: stores indexed before M1 are recreated (greenfield).

### Gate

nDCG@10 and ANN recall ≥ M0 on SciFact and FiQA; estimator RMSE lower; latency
within noise. The 109k WordLlama experiment saw only +0.3 pt from rotation, and
gemma has a stronger common direction, so a larger gain is plausible but
unproven. **If M1 shows no gain and no loss, keep it anyway.** M3's
reconstruction-based maintenance depends on it: without rotation, sign bits of
similar vectors are correlated.

### Result (2026-10-03)

**Passed.** Rotation is in (`semantic_search/quantisation/rotation.rs`, wired through
`ClusterIndex::rotate` / `rotated_centroid`, `index_embedding_rotated` and
`search()`), and on gemma it changes almost nothing:

| At 64 probes, vs M0 | SciFact | FiQA |
|---|---|---|
| nDCG@10 | −0.0002 [−0.0006, 0.0000]; 0 better, 2 worse, 298 same | −0.0008 [−0.0024, +0.0003]; 12 / 9 / 627 |
| ANN recall@10 | +0.0033 [+0.0003, +0.0063] | +0.0005 [−0.0014, +0.0022] |
| Pass-1 estimator RMSE | 0.02671 → 0.02655 (−0.6%) | 0.02328 → 0.02276 (−2.2%) |
| Pass-1 recall, all clusters probed | 0.836 → 0.835 | 0.808 → 0.809 |
| p50 / p95 latency, alternated runs | 3.06, 3.05 → 3.10, 3.04 / 3.35, 3.37 → 3.40, 3.32 ms | 14.91, 14.85 → 14.84, 14.67 / 17.40, 17.24 → 17.21, 17.00 ms |

Latency was measured by running the pre-rotation and rotation binaries
alternately, twice; a first comparison against the M0 baseline from hours
earlier showed +2–6%, which alternated runs show to be machine drift. Rotating a
query costs microseconds.

What it means: gemma's sign bits are already close to independent, so the 1-bit
estimator's error barely moves, and Pass-1's candidate losses (16–19% of the
exact top-1000 with every cluster probed) are not caused by a missing rotation.
They are the 1-bit estimator's own variance reordering near-equal scores at
the bottom of the list: the index keeps 99.6% of FiQA's exact top 100, and a
first-pass cut of 4,000 instead of 1,000 gives the same nDCG@10. The only
quality lever is probing (`pass1-recall-study.md`). Rotation stays, as planned:
M3's maintenance reconstructs vectors from codes and relies on it.

**On qwen** (`Qwen3-Embedding-8B`, Q4_K_M, 768 dimensions, centroids refitted
with the gemma recipe), rotation does measurably more, because qwen's residuals
are concentrated in fewer dimensions (top 10% of dimensions hold 21–24% of the
variance, against 15–20% for gemma). Same alternated A/B procedure, 64 probes:

| qwen, at 64 probes | SciFact | FiQA |
|---|---|---|
| nDCG@10 | +0.0012 [−0.0002, +0.0037]; 1 better, 1 worse, 298 same | +0.0001 [−0.0010, +0.0016]; 6 / 8 / 634 |
| ANN recall@10 | +0.0010 | +0.0011 |
| Pass-1 estimator RMSE | 0.0278 → 0.0235 (−15%) | 0.0239 → 0.0213 (−11%) |
| Pass-1 estimator bias | −0.0105 → −0.0001 | −0.0035 → −0.0001 |
| Pass-1 recall | 0.7966 → 0.8084 | 0.8085 → 0.8225 |
| p50 / p95 latency, alternated runs | 3.15, 3.15 → 3.07, 3.13 / 3.50, 3.46 → 3.41, 3.49 ms | 15.48, 15.43 → 15.45, 15.51 / 17.83, 17.69 → 17.72, 17.80 ms |

Without rotation qwen's 1-bit estimate is biased; rotation removes the bias and
keeps more of the exact candidates, at no cost. The final ranking does not move,
for the reason the study gives: the extra candidates sit near rank 1,000.

An audit against the RaBitQ papers and RaBitQ-Library (`rabitq-rotation-audit.md`)
found the rotation and both passes' similarity formulas correct. FhtKac scores
the same as a dense Haar rotation on both models. The audit also fixed a
multi-bit rescale-search bug that affects 2- and 4-bit codes only.

Also in M1: the API rejects an `embedding_dim` the rotation cannot handle (odd,
or under 8) at startup, and the rotator's format is pinned by a fixed-seed test
(`quantisation/test_data/rotation_golden_768.json`).

---

## M2 — Namespace-owned model, centres and postings (still static)

M2 moves everything a namespace's index depends on into that namespace. No
partition moves yet: postings are seeded from the model's centroid file, so
M2a's results can be checked byte-for-byte against M1.

### M2a — The namespace owns its vector-index settings

**Change.** Every setting that shapes a namespace's vector index, or how it is
searched, moves out of the engine config into one `vector_index` object in
`DocStoreSchema` and `KvStoreSchema`. The engine config keeps only what is about
the embedding service itself (URL, timeouts, query-cache TTL) and the set of
models the server supports.

```json
"vector_index": {
  "embedding_model": "gemma",
  "embedding_dim": 768,
  "chunking":     { "window_size": 4, "sliding_size": 2 },
  "quantisation": { "pass1_bits": 1, "pass2_bits": 8 },
  "search":       { "n_probes": 64, "first_pass_top_k": 1000, "top_k": 100 }
}
```

| Field | Default | Valid values | After it is set |
|---|---|---|---|
| `embedding_model` | `gemma` | non-empty, at most 64 characters of `[a-z0-9._-]`; lower-cased on save; must be a declared supported model | fixed |
| `embedding_dim` | 768 | even, 8 to 4096; must equal the supported model's declared dimension | fixed |
| `chunking.window_size` | 4 | 1 to 64 sentences | fixed |
| `chunking.sliding_size` | 2 | 1 to `window_size` (a larger step would skip sentences) | fixed |
| `quantisation.pass1_bits` | 1 | 1 only | read-only |
| `quantisation.pass2_bits` | 8 | 8 only | read-only |
| `search.n_probes` | 64 | 1 to 4096; a search probes at most the namespace's posting count | changeable |
| `search.first_pass_top_k` | 1,000 | `top_k` to 10,000 | changeable |
| `search.top_k` | 100 | 1 to 1,000 (`MAX_RESULT_LIMIT`) | changeable |

- **Defaults are written into the schema.** Any field the caller omits is
  filled with its built-in default the first time semantic search is enabled
  (at create, or by `EnableVectorIndex` / `AddEmbeddingAttribute`), and the
  filled-in object is what is saved. A schema read back always shows the values
  in force, and a later change of a built-in default never changes an existing
  namespace.
- **Only used while semantic search is on.** A store without semantic search
  may omit the object; if it is given, it is validated but unused, like
  `embedding_fields`.
- **Fixed means fixed for the namespace's life.** The model, dimension and
  chunking decide what the stored vectors are, so they cannot change once set,
  including across a drop and re-enable of the vector index (the object is kept
  when the vector index is dropped). Repeating the same value is accepted; a
  different value is rejected with an error that names the field. Changing them
  means a new namespace, or a later "drop and re-index" feature.
- **Read-only bits.** `pass1_bits` and `pass2_bits` record the code widths the
  index uses. A request may omit them or repeat the current value; any other
  value is rejected. They are fields rather than constants so a future version
  can make them choosable without a format change.
- **Search settings change at any time** through a new amendment,
  `UpdateVectorSearch { n_probes?, first_pass_top_k?, top_k? }`, on doc and KV
  stores alike. No re-index: they only affect later searches.
- **Per-request overrides.** A search request may pass `n_probes`,
  `first_pass_top_k` and `top_k` for that query only. They are checked against
  the same ranges, and `top_k ≤ first_pass_top_k` is checked on the values in
  effect for the query.
- **Errors are specific.** Each failure names the field, the value and the rule,
  for example `vector_index.chunking.sliding_size must be at most window_size
  (4), got 6`. Cross-field rules (`sliding_size ≤ window_size`,
  `top_k ≤ first_pass_top_k`) are checked together with the single-field ones,
  on create, on every amendment, on schema import and on every override.

**Centroids are chosen by the model.** Today one global `ClusterIndex` is loaded
from `cluster_path`, which would probe a qwen namespace with gemma's centroids.
In M2a the server keeps one `ClusterIndex` per supported model, loaded from
`{centroid_dir}/{model}/clusters.json` (`centroid_dir` defaults to
`service/embedding_support`) and checked against the declared dimension at
startup. A namespace uses its model's index. `cluster_path` and the
`centroid_mismatch` startup warning are removed: there is no longer a separate
file choice to get wrong. Per-namespace copies of the centres come in M2b.

**Embedding-service checks.** At startup the server probes each distinct
(model, dimension) pair used by an existing semantic namespace. At create and
enable time it probes the requested pair: a 404 `Unknown model` or a dimension
mismatch fails the request; an unreachable service only warns, because the
embed queue tolerates outages.

**Library layering.** `semantic_search::service` keeps taking a
`SemanticSearchConfig` per call, which raw `vector_kv` namespaces (no schema)
build themselves. The doc store builds it per namespace from the engine-wide
service settings and the namespace's `vector_index`, so `search()` and
`index_embeddings` do not change.

**Where the settings are read today and must come from the namespace:**

| Site | Today | After |
|---|---|---|
| `embed_document`, `embed_query` (`service/mod.rs`) | `config.model_name`, `embedding_dim`, `window_size`, `sliding_size` | the namespace's `vector_index`, through the per-namespace config |
| `index_embeddings`, Pass 2 (`service/mod.rs`) | `config.number_of_bits_for_dense_quantisation` | `quantisation.pass2_bits` |
| `search()` (`service/mod.rs`) | `config.n_probes`, `first_pass_sparse_search_top_k`, `top_k_results` | `search.*`, or the request's override |
| Query cache (`doc_store/store/query.rs`, `vector_kv`) | key `model ‖ text`, dimension from the engine config | key `model ‖ dim ‖ text`, both from the namespace, so the same model at two dimensions never shares an entry |
| Vector worker (`vec_index_worker.rs`) | the context's config and its one `ClusterIndex` | the queue entry's namespace → schema → per-namespace config and its model's `ClusterIndex` |
| `SemanticSearchContext` (`doc_store/store/types.rs`) | one config, one `ClusterIndex` | service settings and the per-model `ClusterIndex` set |
| `check_embedding_service` (`main.rs`) | one model at one dimension | each distinct pair in use, plus a probe at create and enable |
| API config (`minnal_db_api/src/config.rs`, `config_report.rs`) | `[semantic_search]` model, dimension, chunking, bits, probing, `cluster_path` | `embedding_service_url`, timeouts, cache TTL, `centroid_dir`, `supported_models` |
| Search routes (`routes/semantic_search.rs`) | `top_k` only | `top_k`, `n_probes`, `first_pass_top_k` |
| `config/sample.toml`, `service/scripts/examples/docs.sh`, Docker image | old `[semantic_search]` keys; the image copies one centroid file | new keys; the image copies the centroid directory |

Reader-facing docs (READMEs, QUICKSTARTs) are updated in the final doc pass
after all milestones. Stores created before M2a are recreated (greenfield).

**Gate.** Pure refactor: with every setting at its default, harness results are
byte-identical to M1. New tests:

- every validation range and both cross-field rules, at their edges, for
  create, amendment, import and per-request overrides;
- defaults written into the saved schema when fields are omitted;
- fixed fields rejected on every amendment path, including after a
  vector-index drop and re-enable; read-only bits rejected unless omitted or
  equal;
- `UpdateVectorSearch` changes later searches without re-embedding anything;
- two namespaces with different models (gemma and qwen) and different
  settings in one process: each embeds with its own model and chunking, probes
  its own model's centroids, and keeps its own query-cache entries.

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
After seeding, the namespace never reads the file again. A model with no bundled
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
- **Every maintenance operation goes through a journal** (next section),
  so a crash at any point is finished by recovery, never left half done.

#### Maintenance journal

The store has no transactions, and the design does not need them: no step has to
change several keys at once. It needs single-key atomic state changes (a
WAL-backed `put`), steps that are safe to repeat, and a guaranteed order in
which writes become durable. The journal provides the first two; the barrier
rule below provides the third.

**Format.** One WAL-backed companion namespace, `{ns}_ivf_journal`:

| Key | Value |
|---|---|
| `op_id` (u64 BE, increasing) | `{ kind: Split \| Merge \| Reassign \| Rebuild, plan, state }` |

- **The plan is a rule, not a list of entries.** For a split: "entries of P go
  to whichever of R1 or R2 is nearer, measured from their codes", with P, the
  child posting ids, their routing centroids and their new centre ids stored in
  the record. Re-running the plan gives the same assignment every time, and the
  record stays small however large P is. A rebuild covers the whole namespace,
  so it is written as one record per batch of postings.
- **Redo only, no undo log.** Every step is copy-then-delete, so the old state
  stays intact until the plan commits, and recovery always rolls forward. The
  only "undo" is abandoning a plan that never published, which just deletes its
  empty child postings. An undo log would have to capture the previous values of
  no-WAL entries, which are themselves not durable.
- **One writer per namespace** (the maintenance task), so operations never
  interleave and replaying open records in `op_id` order is correct.

**Split, step by step.** Each state change is one WAL-backed put of the journal
record.

| Step | Writes | Durable how | Record state after |
|---|---|---|---|
| 1. Plan | Read P, reconstruct `x̂ = c + scaling_factor · P(2b − 1)` (unrotated), run balanced 2-means. Write the record | WAL | `Planned` |
| 2. Prepare | Append the two centres; write postings P1, P2 (`Active`); mark P `Draining` | WAL | `Planned` |
| 3. Publish | Take the routing-epoch write guard (below), publish a snapshot where inserts never choose P but search still probes it | WAL | `Published` |
| 4. Copy | Per doc, under `lock_doc_vectors`: add P1/P2 to the doc's meta, then write its chunks under `P1‖doc` / `P2‖doc` | **no-WAL** | `Published` |
| 5. Barrier | Flush the sparse and meta namespaces to L0 (the flush fsyncs the value log first) | flush | `Copied` |
| 6. Delete | Per doc, under the lock: delete `P‖doc`, then remove P from the doc's meta. Rescan P; it must be empty | WAL (deletes), no-WAL (meta) | `Copied` |
| 7. Retire | Mark P `Retired`; publish the snapshot | WAL | `Done` |

**Recovery** runs at open, before the vector worker and reconciliation start.
It finishes every record that is not `Done`, in `op_id` order:

| Last durable state | Recovery |
|---|---|
| No record | Nothing happened. Any centres appended are unreferenced and harmless. |
| `Planned` | Abandon: delete the child postings (they are empty, since inserts only route to them after `Published`), restore P to `Active`, mark the record `Done`. |
| `Published` | Redo steps 4–7. Step 4 is safe to repeat; it may re-copy entries that did survive. |
| `Copied` | Redo steps 6–7. |
| `Done` | Nothing. `Done` records are deleted, or the last N kept for diagnostics. |

A merge or a reassignment follows the same table with its own plan. A merge
moves all of P's entries to their nearest remaining postings; a reassignment
moves the entries the plan selects from neighbouring postings.

**Rules the steps depend on.**

1. **Barrier before delete.** Copies are no-WAL, deletes are WAL-backed, and a
   WAL-backed write is durable the moment it returns while a no-WAL one is
   durable only after the next memtable flush. Without step 5, a crash after a
   delete but before the flush loses the copy and the chunk is gone for good.
   Copy order alone does not help; durability order does. One flush per
   operation, not per entry. (Making the copies WAL-backed would also work, at
   roughly one extra fsync per insert at the measured 0.5–0.9 moves per insert.)
2. **The meta is always a superset.** `{ns}_sparse_vector_meta` is the only
   record of which postings hold a doc's chunks, and `delete_vector` deletes
   exactly what it lists. A posting missing from the meta becomes an orphan: the
   chunk of a deleted doc keeps appearing in search. So a posting is added to
   the meta before the copy and removed only after the delete. An extra posting
   is harmless (deleting an absent key does nothing). The barrier makes this
   hold across the two no-WAL meta writes.
3. **Values are lists.** The key `posting ‖ doc_id` holds all of that doc's
   chunks in that posting, so a move splits or merges lists. Every move runs
   under `lock_doc_vectors`, which serialises all writers of a doc's vector keys,
   so a plain get and put is enough. The move re-reads its source under the
   lock and skips it if an upsert or delete already changed it. A repeated move
   must not append a chunk that is already there: chunk codes carry no id, so
   de-duplicate by comparing bytes.
4. **Routing epoch.** Inserts route and write while holding a per-namespace
   tokio `RwLock` read guard, inside `lock_doc_vectors`. Step 3 takes the write
   guard only to publish, so once it returns no insert that routed with the old
   snapshot is still writing to P.
5. **Recovery comes first.** Reconciliation treats `Draining` and `Retired`
   postings as part of the journal's work: an entry under a retired prefix is
   moved, not reported as an orphan.

Search tolerates the intermediate states: duplicates during a move score the
same (MaxSim takes each document's maximum), and holes cannot occur.

**Same rule in today's code.** `upsert_vectors` and the vector queue have the
rule 1 problem already: the queue entry's removal is durable before the
vectors it vouches for. M0-1 fixes that first, with the same barrier.

**Gate (SciFact and FiQA, empty namespace, no file, all three orders):**
recall within 2 pts of (iv) at equal entries scanned; largest posting < 1%
(today 43% on SciFact); nDCG@10 ≥ M2d − 0.005. Plus these tests:

- a **crash at every step boundary** of the table above, next to
  `racing_upsert_and_delete_leave_no_orphaned_cluster_keys`. A crash here is
  `std::mem::forget(db)` at the chosen step, as the existing vector crash tests
  do: unflushed no-WAL writes live only in the memtable, so a process crash
  loses them. That needs a hook to stop at each step boundary. After reopen and recovery, check
  that every posting key a doc has is listed in its meta, that no chunk is
  missing, and that no chunk of a deleted doc remains
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

### After M3: re-check the Pass-1 estimator form

Pass 1 estimates a chunk's residual inner product against the raw query `q`
(`⟨q,c⟩ + est⟨r, q⟩`). The RaBitQ paper and library estimate it against the query
residual instead (`⟨q,c⟩ + ⟨r,c⟩ + est⟨r, q − c⟩`, with `⟨r,c⟩` exact and stored).
That form's error scales with `‖q − c‖` rather than `‖q‖ = 1`. With the bundled
256 centroids, queries sit about as far from the probed centroids as from the
origin (`‖q − c‖` 1.10–1.15 for gemma, 0.99–1.04 for qwen), so it gains nothing:
nDCG@10 moves by at most 0.0004 across eight model–dataset pairs
(`rabitq-rotation-audit.md`, row 9).

M3's smaller, namespace-specific postings bring centres closer to the data, and
possibly to the queries that probe them. Once M3 is in, rerun
`vector_bench_pass1_study`, which prints `‖q − c‖` and both forms side by side.
Adopt the paper's form only if `‖q − c‖` falls well below 1 and nDCG@10 improves.
It costs one stored float per chunk (`⟨r,c⟩`, with `⟨ō,c⟩` folded in as the
library does, so search needs no extra per-cluster work) and a re-encode, which
the per-namespace format from M2b can carry.

---

### Re-encode strategies (per-namespace policy)

After a split, merge or rebuild, a moved code keeps its original centre (the
code-only design above). That code is still **correct**: it scores against the
centre it names, just with a wider error than a fresh encode against the new,
nearer centre would have. Re-encoding moved codes is therefore an optional
**precision** improvement, never a correctness requirement, and it can wait.

Each namespace chooses where re-encoding gets its vectors, as
`reencode_source` in its schema next to `embedding_model`:

| Policy | Source of vectors for re-encoding | Extra storage | Load on the embedding service | Moved codes |
|---|---|---|---|---|
| `none` (default) | None: only keys move | none | none | keep their original centre; slightly less precise |
| `stored` | The raw embeddings, kept in `{ns}_raw_vector` | about 3 KB per chunk as f32 at 768 dims (FiQA: ~530 MB, against ~19 MB of 1-bit codes); half as f16 | none | re-encoded exactly against the new centre |
| `service` | The embedding service, called again | none | heavy: re-embedding FiQA takes ~95 min at today's ~10 docs/s, competing with new documents | re-encoded exactly against the new centre |

Users pick the trade-off per namespace: disk for local, fast, service-free
maintenance (`stored`), or no extra disk at the cost of embedding-service
capacity (`service`), or neither (`none`).

**Rules shared by all three.**

- **Code-only is the floor.** Split, merge and rebuild ask a *vector source*
  for a document's vectors. When it has none (policy `none`, a stored vector
  lost in a crash, the service unavailable), the operation proceeds code-only.
  So M3 never depends on either optional source.
- **New documents always come first.** Re-encoding runs from a separate
  low-priority queue, never ahead of the embed queue. For `service` it is capped
  at a configured share of embedding requests; for `stored` it is local CPU only.
  A pending re-encode costs precision, not correctness, so delaying it is
  always safe.
- **Re-encodes are journalled** like any other maintenance operation (see
  *Maintenance journal*): the new code and its `centre_id` replace the old one
  under the doc lock, and a crash mid-way leaves the old, still-valid code.
- **One model per store.** The model is fixed per store (M2a), so stored vectors
  can never come from a different model than the service would use.

**`stored` in more detail.** The raw vectors are written no-WAL alongside the
codes, at index time, and deleted with them (the meta superset rule covers the
key). Losing one to a crash only drops that document back to `none` until it is
next embedded. Beyond re-encoding, stored vectors give exact k-means for
rebuilds (M4), changing the rotation or the dense bit width without the service,
and a per-namespace exact ground truth for checking recall on live data.

**`service` in more detail.** A re-encode re-embeds the document's text (from the
store itself), so it needs no extra storage, but each one costs a full embedding
call. Its queue entries are a separate kind from new-document embeds so the
worker can always prefer the latter.

**Sequencing.** M3a–M3c ship `none`, which every namespace needs. M3-pre's
simulation also measures what an exact re-encode is worth on gemma. On the 109k
WordLlama test it lifted recall@10 before reranking from 0.918 to 0.955; after
the Pass-2 rerank the gain will be smaller. Both optional strategies are planned:

- **M3d — `stored`:** the raw-vector namespace, the vector-source interface,
  the low-priority re-encode queue. Gate: after a FiQA split-heavy ingest and a
  full re-encode, recall within 0.5 pt of a fresh index fitted on the same
  postings; no change to new-document indexing throughput.
- **M3e — `service`:** the same interface backed by the embedding service, with
  the request cap. Gate: under continuous new-document load, new-document
  indexing throughput stays within 5% while re-encodes drain.

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
| Rotated/unrotated mix-up (`q` vs `q'`, `c` vs `Pᵀc`) | M1 | Consistency test against explicitly rotated inputs; estimator-RMSE gate |
| Hot-path cost of per-entry centre lookup | M2b | Dense `Vec` index; latency gate on this sub-step alone |
| Split races with inserts and deletes; crash mid-split | M3a | Maintenance journal (redo only); barrier before delete; meta kept a superset; routing epoch + doc lock; crash tests that discard unflushed no-WAL writes |
| No-WAL write lost while a WAL-backed delete or queue completion survives | M0-1, M3a | Queue completion after flush (M0-1); barrier before delete (M3a) |
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
| 7 | Which settings the namespace owns (2026-10-04) | Model (default gemma) and dimension (default 768), chunking (fixed once set), code widths (read-only, 1 and 8), search settings (changeable, with per-request overrides); defaults written into the schema (M2a) |
