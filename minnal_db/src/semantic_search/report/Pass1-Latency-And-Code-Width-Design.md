# Pass-1 latency and code width

Status: Part A built and measured (see *Part A as built*); Parts B and C are
proposals, for review.

This document covers two linked changes to Pass 1 of semantic search:

- **Part A** makes Pass 1 faster without changing any stored format.
- **Part B** lets a namespace choose 1-, 2-, 4- or 8-bit chunk codes for
  Pass 1 when it is created, trading storage for better estimates and better
  partition maintenance.
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
- **Code width**: bits stored per dimension of a chunk code. Pass 1 uses 1 bit
  today; Pass 2 uses 8.
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

## Part B — Pass-1 code width of 1, 2, 4 or 8 bits

### B1. The setting

`vector_index.quantisation.pass1_bits` already exists in every namespace's
schema and is fixed for the namespace's life. Today `check_bits` accepts only
1. It will accept **1, 2, 4 or 8**, set when semantic search is first enabled
on the namespace (store creation, or a later enable) and read-only afterwards,
as it is now. The default stays 1. KV stores take the same field through the
shared `vector_settings` code.

Changing the width of an existing namespace means dropping its vector index
and enabling it again, which re-embeds every document. No conversion between
widths is offered: a narrower code can be cut from a wider one, but a wider
one needs the embedding.

### B2. Encoding

- **1 bit**: unchanged (`QuantisationStyle::SingleBit`).
- **2, 4, 8 bits**: the existing multi-bit RaBitQ quantiser
  (`quantise(embedding, centre, bits)`), which Pass 2 already uses, applied to
  the chunk embedding against its posting's centre, stored as
  `QuantisationStyle::MultiBit { number_of_bits }`. `VectorIndex` already
  carries the style, so no new record type is needed.
- Search rejects a sparse entry whose style is not the namespace's width, as it
  rejects a non-`SingleBit` entry today.

Storage per chunk at 768 dimensions:

| Width | Code bytes | Entry with fields and value-log record | musics_200k (210k chunks) | 1M chunks |
|---|---:|---:|---:|---:|
| 1 bit | 96 | ~180 B | ~38 MB | ~180 MB |
| 2 bits | 192 | ~276 B | ~58 MB | ~276 MB |
| 4 bits | 384 | ~468 B | ~98 MB | ~468 MB |
| 8 bits | 768 | ~852 B | ~179 MB | ~852 MB |

(Measured at 1 bit: 38 MB of value log for 208k entries. Wider rows add the
code bytes only.)

### B3. Scoring

Scoring is 0.9 ms at 1 bit and 161k entries, so compute is not the limit at
any width. **Bytes are.** At 8 bits, 1,024 probes read about 134 MB of values
per search, and copying that out of the page cache alone costs on the order of
10 ms. So:

- the scan must read less (fewer probes, which is what wider codes should buy;
  see B5), and
- Part C's in-place block scan becomes worthwhile at 4 and 8 bits, where it
  avoids copying the codes out of the page cache at all.

For the scoring itself:

- **2, 4, 8 bits** start with the existing multi-bit estimator
  (`MultiBitQuanDotProductEstimator`, float query). If it shows in the profile,
  quantise the rotated query once per search to 8-bit integers and use
  AVX-512 VNNI integer dot products (`vpdpbusd`: 64 byte-multiplies per
  instruction, 12 instructions per 768-dimension code), behind the existing
  runtime dispatch (`simd::*_best`) with the scalar fallback. The measured CPU
  has AVX-512 VNNI and VPOPCNTDQ. Query quantisation adds a small estimation
  error, so it goes through the same quality gate as the width itself.
- **1 bit** keeps the float path. A 4-bit quantised query with popcount over
  bit planes (the RaBitQ paper's method) is an option if scoring ever matters
  at 1 bit; today it is 3% of the search.

### B4. Partition maintenance on wider codes

Splits, reassignment and rebuilds decide where chunks go from code
reconstructions (`reconstruct_single_bit`), because maintenance makes no
embedding calls. At 1 bit those reconstructions are coarse: on `musics_200k`,
13% of sampled chunks are not in their nearest posting (2nd nearest 8.6%, 3rd
to 16th 4.8%). Codes of 4 or 8 bits reconstruct the chunk almost exactly, so
maintenance at those widths places chunks where a fresh embedding would.

Needed:

- `reconstruct_multi_bit` for splits' 2-means and rebuild k-means.
- The reassignment `Scorer` decides without reconstructing, using per-byte
  lookup tables that assume 1-bit signs (`SignDot`). For 2 and 4 bits the same
  idea works with tables over 4 or 2 dimensions per byte; for 8 bits,
  reconstruct and compute directly. Its correctness test
  (`the_scorer_decides_as_the_reconstruction_does`) is extended to each width.
- Measure maintenance cost at each width on FiQA indexing, the established
  benchmark (`vector_bench`, background workers running).

### B5. Where the latency gain from wider codes must come from

A wider code does not by itself rank better. A study at 2 bits found Pass-1
recall up from 0.806 to 0.900 but the same exact top 100 and the same nDCG
(`pass1-recall-study.md`): Pass 2 already corrects Pass 1's order. Wider codes
cost more bytes per entry read, so any gain must come from reading fewer
entries:

1. **Fewer probes for the same result.** Better placement (B4) and better
   estimates should let a smaller `max_probes` reach the overlap with a full
   read that 1 bit needs more probes for. Measured with the recall check
   (`GET /admin/indices/{ns}/vector/recall-check`, same seed per width) and,
   on `musics_200k`, with the entity-resolution sweep
   (`work/search_quality/er_probes/`).
2. **A smaller first-pass cut.** If Pass 1's order is closer to exact,
   `first_pass_top_k` can drop below 1,000, which shrinks Pass 2's fetch.
3. **Possibly no Pass 2 at 8 bits** for stores that only need chunk-level
   MaxSim. Pass 2 scores the whole-document embedding, a different signal from
   the best chunk, so this is a per-store question to measure, not a default.

If none of these show a gain on the gate datasets, wider codes are only worth
it where placement matters (stores like `musics_200k`), and the default stays
1 bit.

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
- Memory for `musics_200k`: about 25 MB at 1 bit, about 170 MB at 8 bits.

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
  paired per query. Also Pass-1 recall, recall-check overlap at several probe
  counts, chunk placement (`probe_tool placement`), and the entity-resolution
  sweep on `musics_200k`.
- **Storage and indexing cost (Part B)**: bytes per chunk, index size, FiQA
  indexing time with maintenance.
- **Correctness**: the existing scan, crash-audit and concurrency tests, plus
  the new tests named in each step. Part A changes no results, so search
  output must be byte-identical before and after on the gate datasets.

## Order of work

1. A1 and A2 (the two largest costs, independent of each other). Measure.
2. A3 and A4. Measure.
3. Part B behind `pass1_bits`, with 1 bit unchanged. Gate each width.
4. Part C1 only if, after 1–3, the scan is still most of a search, or 4- and
   8-bit stores need it.

## Decisions needed

1. Start with Part A as described, before any format work?
2. Widths 1, 2, 4 and 8, default 1, fixed at creation, no conversion: agreed?
3. For integer scoring of 2/4/8-bit codes: accept a quantised query, subject
   to the nDCG gate?
4. Part C1: acceptable in principle (a memory budget for derived blocks), and
   what default budget?
