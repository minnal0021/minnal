# Pass-1 latency and code width

Status: Parts A and B built and measured (see *Part A as built* and *Part B as
built*), and the scan's merge moved onto its workers (*Merging on the scan's
workers*); Part C is a proposal, deferred.

This document covers two linked changes to Pass 1 of semantic search:

- **Part A** makes Pass 1 faster without changing any stored format.
- **Part B** lets a namespace choose 1-, 2- or 4-bit chunk codes for Pass 1
  when semantic search is enabled on it (1 bit by default), trading search
  latency and storage for better chunk placement and Pass-1 recall.
- **Part C** is a layout change (contiguous per-posting blocks) to take up if
  Parts A and B leave the scan as the main cost.

The measurements say where to start: scoring is about 3% of a search, and
reading and arranging the entries is about 85%. So Part A comes first, and
Part B is designed so that wider codes do not undo it.

## Terms used here

- **Posting**: one IVF partition cell. Each has a centre; each chunk embedding
  is filed under one posting.
- **Probe**: a posting that a search reads. `max_probes` is how many of the
  nearest postings a search reads.
- **Entry**: one stored record in `{ns}_sparse_vector`, keyed
  `posting_id (4 bytes) ‖ doc_id`. Its value is an rkyv list of that
  document's chunk codes in that posting (usually one chunk).
- **Code width**: bits stored per dimension of a chunk code. Pass 1 uses 1, 2
  or 4 (the namespace's `pass1_bits`); Pass 2 uses 8.
- **Placement**: whether a chunk is filed in the posting nearest its
  full-precision embedding. Maintenance (splits, reassignment, rebuilds) moves
  chunks using their codes, never the embedding, so a coarse code can leave a
  chunk in a farther posting, where a search finds it only if it probes that
  posting too.
- **Pass-1 recall**: the share of the exact top 1,000 documents (by best
  chunk, full precision) among the 1,000 candidates Pass 1 hands to Pass 2.
- **L1, L0**: the two on-disk levels of a namespace's LSM tree. Keys and value
  pointers live there; the values themselves live in the **value log**.
- **Bucket**: the store shards both the LSM and the value log into buckets by
  the first 8 bytes of the key (8 buckets in the measured store).

## Where a search's time goes

Measured on a copy of `musics_200k` (1,397 postings, 208,364 entries, gemma,
768 dimensions, warm page cache), Ryzen 9 9950X3D. One client, searches one at
a time, query embeddings already cached, top_k 10. Phase times are medians
from wall-clock stamps inside `search()` and `KVStore::scan_prefixes_batch`
(the patch is `work/search_quality/pass1_timing.patch`, not committed).

| Phase (ms) | 256 probes | 1,024 probes | all 1,397 |
|---|---:|---:|---:|
| Entries read | 40,895 | 161,049 | 208,364 |
| Choose postings | 0.15 | 0.28 | 0.31 |
| LSM key scan (`scan_prefixes`) | 2.37 | **11.43** | 13.68 |
| Value-log reads (one `pread` per entry) | 1.55 | **5.46** | 6.91 |
| Re-sort scan results into key order | 0.49 | 2.09 | 2.83 |
| Flatten entries into one list | 0.32 | 1.75 | 2.24 |
| **Score every chunk (1-bit)** | **0.30** | **0.89** | **1.12** |
| Sort entries by document | 0.69 | 2.39 | 2.96 |
| Fold each document's chunks (MaxSim) | 0.42 | 1.94 | 2.65 |
| Keep the top 1,000 | 0.07 | 0.26 | 0.31 |
| Pass 2: fetch and score 1,000 candidates | 0.97 | 1.50 | 1.72 |
| **Total** | **7.33** | **28.00** | **34.77** |

What the numbers say:

1. **The LSM key scan is the largest phase.** For each probed posting, in each
   L1 bucket, the scan calls `fstat`, validates a seek point, and starts a new
   buffered frame reader. At 1,024 probes and 8 buckets that is 8,192 fresh
   starts per search, each to read about 20 entries (about 1.8 KB). Every frame
   is also CRC-checked.
2. **Each entry is its own `pread`.** 161,049 system calls at 1,024 probes.
   The entries a search reads are mostly adjacent in the value log, because
   splits and rebuilds copy a posting's chunks together:

   | Probes | Exactly adjacent to the previous entry | Within 4 KiB | Reads if merged over gaps ≤ 4 KiB | Bytes read / bytes needed |
   |---|---:|---:|---:|---:|
   | 256 | 83% | 97.6% | 966 | 12.6 / 7.6 MB |
   | 1,024 | 95% | 99.8% | 266 | 32.8 / 29.9 MB |

   This is the best case: entries written by ordinary inserts land wherever the
   log is when they are written, and are only gathered by the next split or
   rebuild of their posting.
3. **Arranging entries costs more than scoring them.** The re-sort, the
   flatten, the sort by document (a `memcmp` per comparison of byte-slice
   document ids) and the fold together take 8.2 ms at 1,024 probes. Scoring
   takes 0.9 ms.
4. **Search time is linear in entries read**: about 165 ns of wall time per
   entry beyond a fixed 1.3 ms, against about 5.5 ns to score one. A full read costs only 24% more than
   1,024 probes.

## Part A — faster Pass 1, no format change

Each step stands alone and is measured alone, with the same harness and the
same store copy.

### A1. One cursor per L1 file for the whole probe set

`scan_prefixes` walks each overlapping L1 bucket file once, in key order, with
one `FrameReader` and one `fstat`, jumping forward to the next probed posting's
sparse-index block only when that block lies past the cursor. This is the
pattern `get_multiple` already uses (`scan_sorted_candidates`). When the probed
postings cover most of a file, the jumps disappear and the walk becomes one
sequential read.

Correctness does not change: layers still fold by highest sequence
(`fold_newest`), and a hint is used only after `valid_scan_start` accepts it.

Expected: 11.4 → 2–3 ms at 1,024 probes. Guard: the existing scan-versus-brute-
force tests, plus a test that the per-posting and single-cursor scans return
identical entries for random probe sets.

### A2. Merged value-log reads

Inside each bucket's read thread, sort the pointers by `(segment, offset)`,
merge neighbours whose gap is at most a small threshold into one run, read
each run with one `pread` into the bucket buffer, and slice the values out.
Start with a 512-byte gap: on the table above a 4 KiB gap read 1.7× the bytes
needed at 256 probes, and most entries are exactly adjacent anyway.

The pointer-resolution rule is unchanged: runs are read inside the same
bracket as today. A run that hits a reclaimed segment (`SegmentMissing`)
retries its keys through `get`, as single reads do now.

Expected: 5.5 → about 1 ms at 1,024 probes on this store. Less on a store
whose recent inserts have not been through a split or rebuild; the design
does not depend on the best case.

### A3. Stop rearranging the scan result

`scan_prefixes_batch` sorts its entries back into key order (2.1 ms), and
`search()` then copies every entry into a flat list (1.8 ms). Neither order
is needed: scoring is per entry, and grouping by document sorts again. The scan
returns each bucket's packed entries as they were read, and scoring iterates
them in place.

### A4. Group by document without comparison sorts

Doc-store keys are fixed-width for `u64`, `u128` and `uuid` stores, so
document ids can be compared as integers, not byte slices. Replace the
comparison sort of entry indices with one of:

- a radix sort on the document id's integer form, or
- an open-addressed table keyed by document id, sized to the entry count,
  holding each document's running MaxSim row (no sort at all).

`str` stores (variable-length ids up to 50 bytes) keep the comparison sort,
or hash the id into the same table.

Expected: sort + fold 4.3 → about 1 ms at 1,024 probes.

### A5. No thread spawns per search

Every search spawns one OS thread per overlapping L1 bucket and one per
value-log bucket (`std::thread::scope`). Replace them with a small fixed pool
for scan I/O. It must not be rayon's global pool, which the scoring gate
protects (see `CLAUDE.md` → *Concurrency: the scoring gate*). This saves tens
of microseconds per search, so it is last and kept only if it measures.

### A6. Pass 2 (measured; nothing to change in its I/O)

Pass 2 fetches each first-pass candidate's whole-document code
(`get_dense_entries_batch`, one batch `get_multiple`) and scores it. On
`musics_200k` with Part A in place, one client, 1,000 candidates (ms):

| Step | 256 probes | 1,024 probes |
|---|---:|---:|
| Cut to the top 1,000 and look up their ids | 0.10 | 0.21 |
| LSM lookup of 1,000 keys | 0.50 | 0.79 |
| Value-log reads | 0.23 | 0.26 |
| Score and keep the top k | 0.18 | 0.18 |
| **Pass 2** | **1.0 of 5.1** | **1.4 of 19.0** |

Each candidate's key lies in a different sparse-index block of the dense
namespace's L1, so the batch lookup jumps for nearly every key, and each jump
validated its hint with two reads and a buffer before an 8 KiB refill. Folding
that check into a 2 KiB read at the jump was built and measured: 0.83 → 0.79 ms,
within noise, so it was removed. The same 1,000 lookups take 0.50 ms after a
256-probe scan and 0.79 ms after a 1,024-probe one: the cost is the dense
namespace's data being out of cache after a large scan, not the number of reads.

Pass 2's cost follows the number of candidates, `first_pass_top_k` (1,000 by
default, a per-namespace search setting). It is a fifth of a 256-probe search.
Lowering it is the lever, and whether that keeps quality is a Part B question
(see B5): a better-ordered Pass 1 is what would let it drop.

### Expected result of Part A

At 1,024 probes, roughly 28 → 10–12 ms. A full read, roughly 35 → 13 ms.
These are estimates to be replaced by measurements after each step.

### Part A as built, and what it measured

Built on branch `pass1-scan-latency`. Two steps were added after the first
measurements showed where the time still went, and one planned idea was
dropped:

- **Merging the buckets' scan results** (`merge_l1_runs`). Each L1 bucket's
  entries come back in key order, and the scan then sorted all of them again
  into one key order: 6–7 ms of a 160k-entry scan. The runs are now merged
  directly, comparing the first 16 key bytes as an integer and reading the
  bytes only on a tie. Runs are found from the keys themselves, so any input
  is merged correctly; input in more than 64 runs is sorted instead.
- **Looking up document ids only for the survivors of the first-pass cut.**
  Grouping looked up every probed document's id (161k of them, at scattered
  positions), where only the 1,000 kept for Pass 2 need one.
- **A5 (a fixed thread pool for scan I/O) was not built.** Merging the bucket
  runs on several threads was also tried and removed: it saved 0.4 ms of 19.
- **A6 (Pass 2) changed nothing**: its one change was within noise (see A6).

Search results are unchanged: 1,800 searches on `musics_200k` (600 queries at
256, 1,024 and every posting) returned the same ids, in the same order, with
the same scores, before and after. Every reordering is pinned to the keys' byte
order by tests (`key_head_then_key_is_byte_order`,
`merging_l1_runs_equals_sorting_and_keeping_the_newest`,
`doc_order_sorts_and_groups_as_the_id_bytes_do`,
`scan_prefixes_over_l1_matches_a_reference_for_any_probe_set`,
`batch_reads_return_what_single_reads_do`).

**`musics_200k`, one client, 1,024 probes, phase times (ms):**

| Phase | Before | After |
|---|---:|---:|
| LSM key scan: bucket reads | 11.4 (all of it) | 4.0 |
| LSM key scan: merging buckets into key order | | 4.4 |
| Value-log reads | 5.5 | 4.2 |
| Re-sort, flatten, score | 4.7 | 1.4 |
| Group by document | 4.3 | 2.0 |
| Total | 28.0 | 18.3 |

**`musics_200k`, production build, closed-loop clients** (20 s per cell, top_k
10, query embeddings cached; p50 / p99 ms and searches per second):

| Probes | Clients | Before p50 / p99 | After p50 / p99 | Before / after per second |
|---|---:|---:|---:|---:|
| 256 | 1 | 7.8 / 8.7 | 5.4 / 6.4 | 129 / 182 |
| 256 | 16 | 47.6 / 56.7 | 37.0 / 45.2 | 337 / 431 |
| 1,024 | 1 | 28.5 / 31.2 | 18.0 / 20.7 | 35 / 55 |
| 1,024 | 4 | 53.4 / 62.9 | 37.5 / 47.4 | 75 / 106 |
| 1,024 | 16 | 179.3 / 227.5 | 149.3 / 188.3 | 90 / 108 |
| all 1,397 | 1 | 35.7 / 39.5 | 23.2 / 26.5 | 28 / 43 |
| all 1,397 | 16 | 232.4 / 292.3 | 204.0 / 242.7 | 69 / 81 |

One client gains 30–37%. Under load the gain falls to 12–24% at the median,
because throughput saturates: at 1,024 probes it stops rising at four clients
(106 searches per second, against 75 before). Past that point latency is
queueing, and what limits throughput is the CPU time per search, mostly the
scan's per-entry work across its threads. The p90 and p95 columns, and 2 and 8
clients, are in `work/search_quality/server_{before,after}.txt`.

**Criterion, `bench_pass1_search`** (100,000 synthetic documents on disk, 1,024
postings; mean ms):

| Case | Before | A1 + merge | + A2 | + A3, A4, survivors | Change |
|---|---:|---:|---:|---:|---:|
| Search, every posting, grouped | 14.90 | 12.88 | 11.93 | 8.78 | −41% |
| Search, every posting, scattered | 15.61 | 14.12 | 12.63 | 9.00 | −42% |
| Search, 18% of postings, grouped | 3.77 | 3.52 | 3.21 | 2.69 | −29% |
| Scan alone, every posting, scattered | 10.11 | 8.15 | 6.82 | 5.21 | −48% |

What is left at 1,024 probes on `musics_200k` (18 ms): the bucket reads (4.0 ms,
per-entry frame CRC and key copies), the cross-bucket merge (4.4 ms, one
thread), the value-log reads (4.2 ms) and Pass 2 (1.5 ms). Part C would remove
the first three for postings held in memory.

## Part B — Pass-1 code width of 1, 2 or 4 bits

### B1. The setting

`vector_index.quantisation.pass1_bits` is **1** (the default), **2** or **4**.
It is chosen when semantic search is first enabled on a store (store creation,
or a later enable) and fixed for the store's life, like the model and the
chunking: a dropped and re-enabled vector index keeps it. Using another width
means a new store, because every chunk would have to be embedded again: a wider
code cannot be made from a narrower one, and the embeddings are not kept. KV
stores take the same field.

```json
"vector_index": { "quantisation": { "pass1_bits": 4 } }
```

### B2. Encoding

- **1 bit**: a sign per dimension (`QuantisationStyle::SingleBit`), 64
  dimensions per word.
- **2 and 4 bits**: the multi-bit RaBitQ quantiser Pass 2 uses, applied to the
  chunk against its posting's centre, stored as **bit planes**
  (`QuantisationStyle::BitPlanes { number_of_bits }`): plane `j` holds bit `j`
  of every dimension's code, packed like a 1-bit code, and the planes follow
  one another. A code takes `bits × dimensions / 8` bytes (192 or 384 at 768
  dimensions).
- The style is stored in every code, and it is also how the write path tells a
  chunk code (`SingleBit` or `BitPlanes`, to `{ns}_sparse_vector`) from a
  whole-document code (`MultiBit`, to `{ns}_dense_vector`). `BitPlanes` comes
  after the existing styles, so stored 1-bit and 8-bit entries read as before.
- Search skips, and counts as corrupt, any chunk code whose style is not the
  namespace's width.

### B3. Scoring

The bit-plane layout reuses the 1-bit kernel: the code's dot product with the
query is `Σ_j 2^j · ⟨plane_j, q⟩`, one AVX-512 1-bit pass per plane, in the
multi-bit estimate. A 1-bit search runs exactly the code it ran before; its
results are identical, query by query, to a build without wider codes. A
quantised query with integer dot products (AVX-512 VNNI) was not built:
scoring is a small share of a search, and what costs time at wider widths is
reading more bytes (B5).

### B4. Partition maintenance

Splits (2-means), rebuilds (k-means) and the recall check reconstruct each
chunk from its code (`reconstruct_code`): `c + s·(2b − 1)` at 1 bit, and
`c − s·(u + c_b)` for a bit-plane code with per-dimension value `u` and shift
`c_b = −(2^(b−1) − 0.5)`. Reassignment decides from per-byte lookup tables
without reconstructing; for bit planes it sums one table pass per plane. Its
test (`the_scorer_decides_as_the_reconstruction_does`) checks it against the
reconstruction at 1, 2 and 4 bits. Codes still keep the centre they were
encoded against when moved, and maintenance still makes no embedding calls.

### Part B as built, and what it measured

Measured with `vector_bench` (frozen embeddings through the production write
path and search; partition grown from one posting with maintenance running)
on SciFact (5,183 documents, 22,787 chunks) and FiQA (57,600 documents,
172,998 chunks), with gemma and qwen. Grown runs twice, alternating widths;
rebuild runs once (`MINNAL_BENCH_REBUILD=1`, k-means on the codes after
indexing). Commit `2ab19bf`; a build from `main` at 1 bit gave identical
per-query results. Search latency and indexing time are from commit
`1b7e83e` (grown runs, twice each), whose results and placement match.

**Placement**: chunks in their nearest posting (in one of the 2 nearest).

| Store | Partition | 1 bit | 2 bits | 4 bits |
|---|---|---:|---:|---:|
| FiQA gemma | grown | 89.6% (95.9%) | 94.1% (98.1%) | 97.1% (99.2%) |
| FiQA gemma | rebuilt | 91.0% (97.0%) | 96.2% (99.1%) | 98.7% (99.7%) |
| FiQA qwen | grown | 91.0% (96.8%) | 95.0% (98.5%) | 97.5% (99.3%) |
| FiQA qwen | rebuilt | 92.4% (97.6%) | 96.7% (99.2%) | 98.9% (99.8%) |
| SciFact gemma | grown | 91.0% (96.3%) | 94.5% (97.8%) | 97.6% (99.7%) |
| SciFact qwen | grown | 90.5% (95.6%) | 95.5% (98.1%) | 98.8% (99.8%) |

**Search, FiQA, 256 probes** (the default `max_probes`), grown partition:

| | 1 bit | 2 bits | 4 bits |
|---|---:|---:|---:|
| Pass-1 recall, gemma / qwen | 0.824 / 0.841 | 0.905 / 0.920 | 0.952 / 0.964 |
| ANN recall@100, gemma / qwen | 0.986 / 0.988 | 0.988 / 0.991 | 0.989 / 0.991 |
| ANN recall@10, gemma / qwen | 0.994 / 0.996 | 0.993 / 0.997 | 0.993 / 0.997 |
| nDCG@10, gemma / qwen | 0.4760 / 0.5916 | 0.4745 / 0.5919 | 0.4741 / 0.5919 |
| Search p50, gemma / qwen (ms) | 4.11 / 4.07 | 3.94 / 3.94 | 4.29 / 4.25 |
| Search p50 at 512 probes (ms) | 5.85 / 5.70 | 5.88 / 5.88 | 7.04 / 6.98 |

(ANN recall@k: the share of the exact top k, full-precision two-pass ranking,
in the index's top k.)

**Cost**:

| | 1 bit | 2 bits | 4 bits |
|---|---:|---:|---:|
| Pass-1 storage per chunk (key, code, factors, value-log record) | 142 B | 238 B | 430 B |
| FiQA Pass-1 index | 23.4 MiB | 39.3 MiB | 70.9 MiB |
| FiQA indexing, gemma / qwen (s) | 42 / 40 | 44 / 42 | 50 / 49 |
| FiQA rebuild, gemma / qwen (s) | 145 / 138 | 95 / 93 | 113 / 110 |

What the numbers say:

- **Placement and Pass-1 recall improve at every width**, on both datasets and
  both models. At 4 bits about a third as many chunks are misplaced as at 1
  bit, and Pass 1 keeps 95–96% of the exact top 1,000 instead of 82–84%.
- **nDCG and the top 10 do not change.** From 64 probes up, every paired
  nDCG difference against 1 bit, at every cutoff from 10 to 100, is within
  ±0.003 and its 95% interval includes zero. Pass 2 rescores the 1,000
  candidates with the whole-document code and recovers what a coarse Pass 1
  ranks too low. Below 32 probes the differences are larger but go both ways
  (SciFact qwen at 8 probes: −0.029 at 2 bits, −0.018 at 4).
- **Recall@100 improves slightly** (0.2–0.5 points), the end-to-end effect of
  the better candidate list.
- **Better placement does not let a search read fewer postings** on these
  stores: recall at a given probe count is the same at every width.
- **Search latency barely changes at 2 bits and grows with the probe count at
  4**: at 256 probes a 2-bit search is within 4% of a 1-bit one and a 4-bit
  search 4–5% slower; at 512 probes, +1–3% and +20–22%. The wider codes are
  more bytes to read, and the read is most of a search.
- **Rebuilds are faster** at 2 and 4 bits: k-means converges in fewer
  iterations on finer reconstructions.

A separate sweep on the frozen embeddings (`vector_bench_pass1_width`: fixed
partition, every probe count and a first-pass cut of 1,000 or 300) agrees: the
same probe count reaches a full read's nDCG at every width; at a cut of 300
the 2-bit code stayed within 0.001 nDCG of a full read on FiQA qwen, where 1
bit did not.

### Choosing a width

1 bit is the default and the right choice for most stores. A wider code buys
better Pass-1 output at the price of search latency and Pass-1 storage:

| Width | Choose it when | Costs (FiQA, 256 probes) |
|---|---|---|
| **1** (default) | Ranked retrieval of documents, where Pass 2 decides the final order: the top results are the same at every width. Large stores where Pass-1 storage matters | — |
| **2** | Pass 1's own candidate list matters: a small `first_pass_top_k`, long result lists (recall@100 and beyond), or anything that uses the candidates directly. Stores that are rebuilt often | Search latency within 4% of 1 bit; 1.7× Pass-1 storage |
| **4** | Chunk placement matters most: near-duplicate or entity matching where one chunk must be found, short records whose chunk and document are nearly the same text, stores where misplaced chunks cost many extra probes. The most accurate Pass 1 | +4–5% search latency (+20% at 512 probes); 3× Pass-1 storage; +20% indexing time |

The width cannot be changed later, so measure on a sample first: build a test
store at each width and compare the recall check
(`GET /admin/indices/{ns}/vector/recall-check`) and search latency at the
store's `max_probes`.

## Merging on the scan's workers, and seeking in L0

Two changes to the LSM scan (`LSMTree::scan_prefixes`), with no format change.

**Where the time went.** A profile of the Criterion scan (1,024 postings, keys
scattered in the value log, 1-bit codes) put about 92% of the scan's CPU on the
calling thread and only about 8% on the per-bucket reading threads. About half
of the calling thread's time was the merge of the buckets' results into one key
order. A posting's keys (`posting id ‖ document id`) are hashed to buckets by
their first 8 bytes, so every posting is spread over every bucket, and the
buckets' sorted runs had to be merged entry by entry on one thread.

**Merging on the workers.** The probed posting ids are split into contiguous
groups, one worker per group (as many groups as overlapping buckets). Each worker
reads its group's postings from every bucket and merges those runs itself.
Groups are ascending, disjoint ranges of ids, so the workers' outputs follow one
another in key order and need no merge after them. Seeks per posting are
unchanged.

**Seeking in L0.** L0 files were read whole on every scan. L0 is merged into L1
only once it holds a tenth of L1's entries, and under the engine benchmark's
standard mix at 16 clients the sparse namespace's L0 averaged 19% of L1's bytes
(peak 43%). Each L0 file is now skipped when its key range misses every probed
posting, and otherwise read with the same per-posting cursor as L1, seeking
through the file's sparse index.

**Measured** (`bench_pass1_search`, 100,000 documents, 8 buckets, everything in
L1, so the L0 change does not show here; mean ms, same machine and build
settings):

| Case | Before | After | Change |
|---|---:|---:|---:|
| Scan, scattered, 184 postings | 1.11 | 0.88 | −21% |
| Scan, scattered, 750 postings | 3.43 | 2.37 | −31% |
| Scan, scattered, 1,024 postings | 4.73 | 3.11 | −34% |
| Scan, grouped, 1,024 postings | 4.59 | 2.93 | −36% |
| Search, scattered, 184 postings | 2.97 | 2.72 | −9% |
| Search, scattered, 750 postings | 6.59 | 5.43 | −18% |
| Search, scattered, 1,024 postings | 8.40 | 6.94 | −17% |
| Search, scattered, 1,024 postings, 4-bit codes | 9.96 | 8.52 | −15% |

Results are unchanged: `scan_prefixes_over_l1_and_l0_matches_a_reference_for_any_probe_set`
compares every probe set against a reference with keys in L1 alone and with
rewrites and deletes in three L0 files above it, and
`scan_prefixes_seeks_through_level0_files` checks that one posting of a large
L0 file is found by seeking (it fails if the hints are ignored).

## Part C — contiguous per-posting blocks (if still needed)

After Part A, each entry still carries a key in the LSM, a 36-byte value-log
record header plus its key, an rkyv list wrapper and per-entry validation.
That is about half the bytes at 1 bit. Part C removes them from the search
path without changing what is stored.

### C1. A derived, in-memory block per posting (recommended form)

- For each posting, an immutable block: document ids, centre ids and scale
  factors as arrays, and all codes contiguous and 64-byte aligned (one array
  per field, not one record per entry).
- Built the first time a search probes the posting, from the same KV entries
  the scan reads today.
- Replaced when anything changes the posting. Every vector write already
  returns a `PostingDelta` naming the postings it touched, inside the routing
  epoch, and maintenance publishes a new snapshot when it moves chunks. Both
  are the invalidation points.
- The KV entries stay the source of truth. Nothing about writes, the vector
  index WAL, the maintenance journal or crash recovery changes, because the
  blocks are never persisted.
- A memory budget (a new server setting, for example
  `[semantic_search] pass1_block_cache_bytes`), evicting least-recently-probed
  postings. A probed posting without a block takes the Part A path.
- Memory for `musics_200k`: about 25 MB at 1 bit, about 85 MB at 4 bits.

Costs and risks: a write-heavy namespace rebuilds blocks often (one posting's
block per write, built on the next probe), and a rebuild of the partition
invalidates every block at once. The budget bounds memory, not churn.

### C2. Persisted blocks (not recommended now)

Storing each posting's block as one value would make every insert or
maintenance move rewrite a block, and would change the vector index WAL,
the maintenance journal and every crash audit. Revisit only if C1's memory
budget cannot hold the probed postings of a real workload.

## How each step is judged

- **Latency**: the copy at `work/search_quality/pass1_db`, the scratch server
  on port 8090, `work/search_quality/er_probes/load.py` at 256 / 1,024 / all
  probes, p50 and p95, plus the phase-timing patch for the breakdown. Also
  `vector_bench` on SciFact and FiQA for stores of ordinary text.
- **Quality (Part B, and A4 if it changes any order)**: `vector_bench` on
  SciFact and FiQA with gemma and qwen, nDCG at 10, 20, 30, 40, 50 and 100,
  paired per query, plus Pass-1 recall, ANN recall and chunk placement
  (`MINNAL_BENCH_PASS1_BITS`, the placement table in its report).
- **Storage and indexing cost (Part B)**: bytes per chunk, index size, FiQA
  indexing time with maintenance.
- **Correctness**: the existing scan, crash-audit and concurrency tests, plus
  the new tests named in each step. Part A changes no results, so search
  output must be byte-identical before and after on the gate datasets.

## Order of work

1. A1 and A2 (the two largest costs, independent of each other). Measure.
2. A3 and A4. Measure.
3. Part B behind `pass1_bits`, with 1 bit unchanged. Measure each width.
4. Part C1 only if, after 1–3, the scan is still most of a search, or 4-bit
   stores need it.

## Decisions

1. Part A first, before any format work.
2. Pass-1 widths 1, 2 and 4 (no 8), default 1, chosen when semantic search is
   enabled and fixed after that, no conversion between widths. Offered as an
   explicit trade-off (*Choosing a width*).
3. Scoring of 2- and 4-bit codes uses the float multi-bit estimate over bit
   planes; a quantised query (integer dot products) is added only if scoring
   shows up in a profile, and only if it passes the nDCG gate.
4. Part C1 deferred: after Parts A and B the scan is not yet the case for it.
