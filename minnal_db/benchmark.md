# minnal_db — Benchmark Report

This is a full benchmark run of the `minnal_db` engine: the key-value store, the
field-index predicate evaluator, and the semantic-search distance math. It was
measured at one point in time on a single machine, so treat the absolute numbers
as specific to the hardware below. The durable signal is in the *relative*
comparisons — how cost changes with value size, key count, a read versus a
write, the raw API versus the typed one, or data in memory versus data on disk.
Those ratios hold up across machines even when the raw microsecond figures
don't.

The [README's Performance section](README.md#performance) explains the design
choices behind these numbers and lists the ballpark figures this report is
checked against; the final section here closes the loop on that comparison.

---

## Environment

| | |
|---|---|
| OS | Ubuntu 26.04 LTS, kernel 7.0.0-30-generic, x86_64 — bare metal (no virtualisation layer between the benchmarks and the disk) |
| CPU | AMD Ryzen 9 9950X3D — 16 cores / 32 threads, boost to 5.76 GHz, AVX-512 |
| RAM | 96 GB |
| Disk | NVMe SSD (`nvme0n1`, non-rotational, 1.8 TB) — `target/` and all benchmark temp dirs live here |
| rustc / cargo | 1.96.0 |
| gnuplot | 6.0 patchlevel 2 (used to render the charts in this report) |
| Repo | branch `main` @ commit `4e8ef22` |
| Date | 2026-08-22 |

---

## How this was run

Everything here comes from `cargo bench -p minnal_db --all-features`, run from
the repo root. That builds and runs all eleven benchmark suites; the
`--all-features` flag only matters for the semantic-search suite, which needs
the vector code compiled in. Criterion writes full interactive per-benchmark
plots (violin plots, PDFs, regression lines) under
`target/criterion/*/report/index.html` — this document is a curated summary of
them, not a replacement. Every figure quoted here is Criterion's **mean point
estimate**.

The whole report is reproducible in one step:

```sh
minnal_db/docs/benchmarks/tools/run_report.sh
```

which runs every suite, extracts Criterion's means, and re-renders the charts
here. It archives any previous `target/criterion` rather than deleting it, so
Criterion never reports a cross-machine "regression" against a baseline measured
somewhere else.

Everything ran with Criterion's defaults — 100 samples per case, a 3-second
warm-up, a 10-second measurement window, 95% confidence intervals. The one
exception is the mixed read/write suite, which re-seeds thousands of durable keys
before every measured batch; at 100 samples that took 20-45 minutes per case, so
it was cut to 20 samples, trading some confidence-interval width for a run that
finishes in minutes.

**How much a repeat run moves.** Criterion's confidence intervals describe
spread *within* a measurement, which is not the same as reproducibility. Across
runs on this hardware: fsync-bound writes are the steadiest thing here (±1.5%),
CPU-bound work — the semantic-search stages, predicate evaluation, bitmap
paging — holds to ±5%, and the sub-microsecond cases (memtable hits, miss paths)
swing 20-40% if the machine is doing anything else at the time. Read the small
numbers as orders of magnitude unless the host is idle.

### A word on memory versus disk

minnal_db keeps recently-written data in an in-memory table (very fast to read)
and moves older data down onto an on-disk SSTable (slower, but that on-disk tier
is where most of the data lives most of the time). A benchmark that only ever
reads freshly-written, still-in-memory keys paints too rosy a picture of a real
running database, where the working set has aged onto disk.

So every read benchmark here measures **both** tiers side by side: one case with
the data resident in memory, and one where the same data has been flushed and
compacted onto disk and the store reopened, so the read genuinely goes through
the on-disk lookup path — its key-range check, bloom filter, and bounded index
scan. Write benchmarks aren't split this way, because a write always lands in the
in-memory table regardless of what's already on disk, so the tier distinction
doesn't apply to write cost. The lookup benchmark goes one step further and adds
a *mixed* case, described in its own section, where a single read-only workload
reads across both tiers at once.

There's deliberately no third tier for "flushed to disk but not yet compacted."
That intermediate state exists internally, but there's no way through the public
API to hold the database in it long enough to measure — the only operation that
produces it immediately compacts it away in the same call. So it isn't measured
as its own tier.

---

## Writes

Every write is a fully durable transaction — the change is synced to disk before
the call returns, with no batching and no way to relax that per call. That
durability has a fixed cost, and it shows. Writing a 128-byte value takes
2.27 ms; writing a 64-kilobyte value — 512 times more data — takes 2.34 ms, a
3% difference. Latency is essentially flat across that entire value-size range,
because the disk sync itself, not the amount of data, is what a write pays for.
Payload size only starts to move the needle once the synced write is large
enough to matter on its own.

![Write latency across value sizes](docs/benchmarks/write.png)

## `merge` against the rest of CRUD

`merge` is the atomic read-modify-write: it reads a key, runs the caller's
closure over the old value and an operand, and writes the result back, with a
per-key lock held across all three so nothing can land in between. The obvious
question is what that costs against a plain `put` — and the honest answer is
that on this hardware the write benchmarks cannot see it, because a merge and a
put both pay exactly one WAL fsync and everything else is rounding error.

| Operation | Mean | Against `put/2KiB` |
|---|---:|---:|
| `crud/put/2KiB` | 2.269 ms | — |
| `crud/merge_sorted_set/2KiB` | 2.268 ms | −0.07% |
| `crud/delete/2KiB` | 2.272 ms | +0.14% |
| `crud/get_then_put_sorted_set/2KiB` | 2.276 ms | +0.32% |
| `crud/merge_counter/8B` | 2.283 ms | +0.59% |
| `crud/put/8B` | 2.304 ms | +1.54% |
| `crud/get/2KiB` | 448 ns | — |

All six write rows sit inside a 1.6% band, and every one of them is the same
number: the drive's fsync latency. So the way to size what a merge adds is not
to difference two millisecond figures — it is to measure the parts that aren't
the fsync, which the suite does separately and which reproduce to within a
couple of percent:

| Component | Mean |
|---|---:|
| the read a merge does that a put doesn't (`crud/get/2KiB`) | 448 ns |
| the per-key stripe: hash + uncontended lock (`stripe/*`) | 50 ns |
| the caller's own closure — counter (`closure/counter`) | 9.8 ns |
| the caller's own closure — 2 KiB sorted set (`closure/sorted_set`) | 167 ns |

Summed, a merge does roughly **0.5–0.7 µs** of work a blind `put` does not, or
about **0.03%** of the 2.27 ms the fsync costs. Two consequences worth stating
plainly:

- **Atomicity is free here.** `get_then_put_sorted_set` is the same work with no
  guarantee — the hand-rolled version a caller writes without `merge` — and it
  measures 0.32% *slower*, not faster. The striped lock costs 50 ns; nothing in
  the durable write path notices.
- **The closure is the only part you control.** At 167 ns for a 2 KiB sorted-set
  splice it is still four orders of magnitude under the fsync, so a merge stays
  fsync-bound until the closure does something genuinely expensive.

One caveat, and it is the check `bench_merge.rs`'s header comment tells you to
apply before trusting any of these write rows: **the 8 B put measured 1.5%
slower than the 2 KiB put**, which no amount of real work can produce. That
inversion is the tell that the write rows are noise-limited rather than
measuring payload cost — here by only 1.5%, but on a host with less predictable
fsync latency the same inversion runs to tens of percent. The rule it implies is
the one this section follows throughout: read the decomposition, not the
difference between two write rows.

![merge against the other CRUD operations](docs/benchmarks/merge.png)

*Bar labels: `crud/*` are the durable operations (every one an fsync);
`closure/*` are the two merge closures run with no database underneath;
`stripe/*` reproduces the per-key lock's hash-and-acquire against the same
primitives the engine uses (`KeyLocks` is crate-private, so the bench cannot
call it directly).*

## Reads

Reading a key that's still in memory is fast and stays fast regardless of value
size — under a microsecond for small and medium values. Pushing the same read
onto disk adds a roughly constant penalty: a small-value lookup goes from about
0.40 µs in memory to about 3.5 µs on disk, and that gap of roughly 3.1 µs holds
steady across small and medium values (4 KiB: 0.57 µs against 3.6 µs, the same
3.1 µs step). It's a fixed per-lookup disk cost — the bloom-filter check, the
index probe, and one read from the value log — that value size barely affects
until the payload gets big enough that reading the bytes themselves starts to
count (a 64-kilobyte value takes about 3.4 µs in memory and 6.6 µs on disk,
where the extra bytes finally show up).

The miss path — looking up a key that was never written — is a special case
worth calling out. It costs about 0.27 µs on disk and 0.33 µs in memory: both
sub-microsecond, and, surprisingly, *cheaper* on disk than in memory. The
on-disk side is really just measuring the bloom filter proving a key absent,
which is a cheaper thing to do than the in-memory skip-list search that has to
walk to the insertion point to be sure. The same inversion shows up, more
clearly, in the point-lookup sweep below.

![Read latency, memory versus disk](docs/benchmarks/read.png)

*Bar labels: `memtable` = still in the in-memory table, `l1` = flushed and
compacted onto disk.*

## Point lookups as the key count grows

This is a closer look at the same on-disk lookup path, sweeping the number of
keys in the store from a thousand up to a hundred thousand, and adding a mixed
case the plain read benchmark doesn't cover. In the mixed case a single
read-only workload alternates on every call between a key resident in memory and
one resident on disk, so its cost is the blended price of a 50/50 tier mix —
a closer approximation of a real steady-state database than either pure extreme.

The headline is that an on-disk hit barely gets slower as the store grows: it
rises only about 4% (3.49 to 3.64 µs) across a hundredfold increase in key
count. That near-flat curve is the intended behaviour, not a fluke — the on-disk
lookup fast-rejects using the key range, then a bloom filter, then a bounded
index search that caps any residual scanning to a fixed number of entries no
matter how large the file gets. In-memory reads stay faster throughout (0.36 to
0.60 µs), though that gap narrows as the store grows, since the on-disk side
stays flat while the in-memory skip-list search cost creeps up mildly with key
count — a 67% rise over the same hundredfold sweep, against the on-disk 4%.

Two findings here are counterintuitive.

The first is about misses. On disk, minnal_db keeps a quick filter that can
usually say "definitely not here" without looking any further — so an on-disk
miss stays flat and cheap (0.19 to 0.21 µs) no matter how many keys are in the
store. In memory there's no such shortcut: to be sure a key is missing, the
engine still has to search through the in-memory data the same way it would to
find a real key. So an in-memory miss costs about the same as an in-memory hit
(slightly more, in fact, since a hit can stop early), and grows the same way as
the store grows (0.46 to 0.68 µs) — the two tiers end up flipped for misses
compared to hits, by roughly 3x at the largest key count.

The second is about the mixed workload: reading it costs about 6-12% more
than simply averaging the memory and disk numbers would predict, and the
overhead widens with key count. The likely
reason is that memory and disk lookups go through two quite different code
paths internally, and jumping back and forth between them on every single call
adds a small overhead of its own, on top of whatever each lookup costs by
itself. A workload that read a whole batch of in-memory keys and then a whole
batch of on-disk keys — rather than alternating one by one — probably
wouldn't pay this extra cost.

![Lookup latency across tiers and key counts](docs/benchmarks/sstable_lookup.png)

*Bar labels: `memtable` = in memory, `l1` = on disk, `mixed` = the blended
workload described above; `hit` = the key exists, `miss` = it doesn't; the
trailing number is how many keys are in the store for that case.*

## Scans

This suite covers the four ways to read many keys at once — prefix scan, range
scan, cursor pagination, and a full async iteration — each measured against data
in memory and on disk.

Prefix and range scans give the cleanest comparison: on disk they cost a
consistent **~2x** their in-memory equivalent across every result-set size
(1.9x–2.2x, with the ratio drifting up only slightly as the result grows). So
the on-disk penalty here is close to a constant per-scan overhead rather than a
per-element one. Full iteration shows the mildest tier gap of the four — 2.1x
for 128-byte values, shrinking to 1.27x at 4 KiB — because iteration resolves
every value, so value-log I/O already dominates its cost before the tier
distinction even enters, and the larger the values, the more that dominates.

Cursor pagination has the surprise. At its smallest page size it runs *faster*
on disk than in memory (98 µs versus 145 µs), the reverse of everything else
here. By the largest page size the expected order returns and disk is slower
again (735 µs versus 396 µs). The most likely explanation is that at very small
pages the pagination bookkeeping itself dominates the cost regardless of which
tier the data sits in, so the usual tier gap simply doesn't get a chance to
show.

![Scan latency across scan types and tiers](docs/benchmarks/scan.png)

*Bar labels: `prefix`/`range`/`cursor`/`iter` are the four scan types above;
`memtable`/`l1` = in memory vs. on disk. The trailing number means whatever
that scan type varies — matching keys for `prefix`, results returned for
`range`, page size for `cursor`, value size for `iter`.*

## Mixed read/write workload

These cases run blended workloads — 80% reads with 20% writes, and an even
50/50 split — with the read set living in memory or on disk, reporting the
average cost per operation. (This is the suite run at the reduced sample size, so
its confidence intervals are wider than the rest.)

All four cases land within 1.9% of each other (2.248–2.290 ms per operation).
Neither the tier nor the read/write ratio makes a visible difference, and the
reason is the same in both cases: a write costs about 2.27 ms (the disk sync)
while a read costs a few
microseconds on either tier, so the writes utterly dominate the average. Even at
50% writes, the write half alone accounts for roughly 99.9% of the per-operation
time. The read tier and the read/write ratio would only start to matter to this
average at write fractions far lower than anything swept here.

![Mixed workload latency](docs/benchmarks/mixed.png)

*Bar labels: `memtable`/`l1` = whether the pre-seeded read set lives in
memory or on disk; the percentages are the read/write split.*

## Write-ahead log

This suite breaks the write path apart: the durable WAL append on its own, the
serialization round-trip, the recovery scan, the full write throughput, and the
cost of tagging an entry with its namespace. The three pieces sit in three very
different cost bands, so they're shown as three separate charts below rather
than one crowded chart spanning six orders of magnitude.

The clearest result is how little of the write cost is anything *but* the disk
sync. The full write path (WAL plus value log plus in-memory index, 2.295 ms) is
within 1% of the durable WAL append measured alone (2.271 ms) — so the value log
and the memtable insert together account for roughly 24 µs, about 1% of a write.
Tagging an entry with a namespace ID shows no cost either: the default namespace
measures 2.267 ms against 2.321 ms for an arbitrary non-default one, a 2.4%
difference in the same direction and of the same size as the drift the write
rows show generally (see the 8 B-versus-2 KiB inversion in the merge section).
Writing a `u32` into a record cannot cost 54 µs; this is the measurement floor,
not a finding. In other words, the disk sync *is* the write path's cost: the
per-write sync is a deliberate durability choice, and this confirms it in
measurement, not just in design intent.

![Durable write cost: fsync'd append, full put, and namespace tagging](docs/benchmarks/wal_durability.png)

*Bar labels: `append_fsync` = a raw, durable WAL append in isolation;
`full_put_throughput` = the entire write path (WAL + value log + in-memory
index) end to end; `namespace_overhead` = the same raw append tagged with a
namespace id (`ns_id_0` is the default namespace, `ns_id_42` an arbitrary
non-default one, to check tagging isn't adding cost).*

Serializing and deserializing a WAL entry, by contrast, is three to five
orders of magnitude cheaper than the sync itself — serializing runs 34.7 ns for
a 128-byte entry up to 1.10 µs at 64 KiB, deserializing 13.3 ns to 547 ns,
against roughly 2.27 ms — so it's nowhere near the critical path.

![Serialization round-trip cost](docs/benchmarks/wal_serialization.png)

*Bar labels: `to_bytes` = serializing a WAL entry, `from_bytes` =
deserializing one back.*

The recovery scan, which reads entries back the way a restart would, runs at
about 3.5 million entries per second for small values — 287 µs for 1,000
128-byte entries, 1.44 ms for 5,000 of them, so the rate holds as the scan grows
— and 2.6 million/s at 4 KiB.

![Recovery scan throughput](docs/benchmarks/wal_scan.png)

## Typed API

minnal_db offers a typed convenience layer that serializes and deserializes your
own types with zero-copy on top of the raw-bytes API. The question this suite
answers is what that convenience costs, and the answer is essentially nothing on
top of what the underlying operation already costs.

Typed writes and deletes land within noise of raw writes (2.273 ms and 2.275 ms
against the raw 2.271 ms) — the disk sync swamps everything else regardless of
which API you call. On the read side, a typed read shows the same ~3.0 µs
memory-to-disk step that a raw read does, so the zero-copy deserialization adds
no tier penalty of its own. The typed iteration, range, and prefix scans all sit
in the same consistent ~1.9-2.2x on-disk-versus-memory band as their raw
counterparts, inheriting their tier sensitivity directly from the underlying
scan rather than from anything specific to the typed layer.

The keys-only operation is the one that behaves differently. Fetching just the
keys shows *no* memory-versus-disk gap at small key counts — at 100 keys the two
tiers are statistically indistinguishable (261 µs on disk against 264 µs in
memory) — and only opens up a gap (1.75x) at 1,000. That fits: fetching keys
never touches the value log, so at small counts the tier difference is too small
to clear the noise floor, and only becomes visible once there are enough keys
for it to accumulate.

![Typed API latencies](docs/benchmarks/typed.png)

*Bar labels: `get`/`put`/`delete`/`iter`/`keys`/`range`/`scan_prefix` are the
typed operations, mirroring the raw-bytes API's method names; `memtable`/`l1`
= in memory vs. on disk.*

## Field-index predicates

These benchmarks evaluate field-index queries — the bitmap-backed predicate
engine — over a hundred thousand rows across three indexed fields.

Equality lookups are cheap: a string-equality predicate runs at 11.4 µs, and
combining two equality predicates with AND or OR costs 32.9 µs. A *range*
predicate over an integer field is a different story entirely — 1.10 ms,
roughly 97x more expensive than equality and the single costliest operation in
the suite by a wide margin. Parsing overhead, by contrast, is negligible:
evaluating a query from its string form (761 µs) and evaluating an
already-parsed one (754 µs) differ by under 1%, so the query parser is not where
the cost lives.

One result runs against intuition: selectivity doesn't track cost. Sweeping a
compound query so that it matches progressively *fewer* rows makes it *slower*,
not faster — 11.2 µs at 25% of rows, 32.6 µs at 12%, and 454 µs at 6%. The most
selective case in the sweep is the slowest one by 40x. Narrowing the result set
here does not mean less work.

![Predicate evaluation latencies](docs/benchmarks/predicate.png)

*Bar labels: `str_eq` = a string-equality predicate, `int_range` = an
integer-range predicate, `compound_and`/`compound_or` = two predicates
combined with AND/OR, `three_way_and` = three predicates combined,
`parse_vs_eval` = isolating query-string parsing cost (`parse_and_eval`)
from evaluating an already-parsed query (`eval_only`), `selectivity/AND/Npct`
= an AND query matching about N% of rows.*

## Paging through query results

A field-index query produces a bitmap of matching rows, and a paginated query
has to window it — return rows *n* through *n+limit*. This suite compares the
obvious way to do that (`iter().skip(n)`, walking the bitmap from the start) with
the one the engine uses (`iter_page(n)`, which skips whole bitmap containers
using their cached cardinality). It runs over 200,000 matching rows at a
50-row page, which is what a REST client paging through results actually does.

Fetching a single page is where the difference is starkest, and it grows with
how deep into the result set the page sits:

| Page offset | `iter().skip(n)` | `iter_page(n)` | Speedup |
|---|---:|---:|---:|
| 0 | 77.1 µs | 0.45 µs | 170x |
| 1,000 | 77.0 µs | 0.80 µs | 96x |
| 50,000 | 79.0 µs | 18.1 µs | 4.4x |
| 199,000 | 242.4 µs | 0.37 µs | 661x |

The 50,000 row is the weak case and the interesting one: that offset lands in
the *middle* of a container, so the walk within that one container still has to
happen — a cost bounded by the container's cardinality (at most 65,536), but not
eliminated. Offsets that land on a container boundary skip straight through, so
paging deep into the set (199,000) is actually *cheaper* than paging near the
front, because there is less of a partial container left to walk.

Walking every page of the result set end to end — the realistic client
behaviour — is **5.3x** faster overall (2.46 ms against 466 µs). That is the
quadratic-versus-linear difference: the old path re-walks the bitmap from the
start on every page, so its cost grows with the square of the number of pages,
while container-skipping stays linear.

![Offset paging over a result bitmap](docs/benchmarks/bitmap_paging.png)

*Bar labels: `bitmap_page_at_offset/*` is one page fetched at the given offset;
`bitmap_full_paged_walk/*` is every page of the result set in sequence.
`iter_skip` is the old path, `iter_page` the current one.*

## Semantic search

These benchmarks measure the vector-search pipeline used for semantic queries,
run against the real cluster centroids bundled with the project (256 clusters)
and synthetic 768-dimension embeddings, so no external embedding service needs
to be running.

Scoring a batch of candidate documents against a query is fast and cheap — a
couple of microseconds for a hundred candidates up to about 16 µs for a
thousand, scaling linearly, and roughly the same whether it's the coarse first
pass or the more precise second pass. The extra precision of the second pass
costs nothing extra at this scale. Picking out which few clusters are worth
searching, the equivalent of choosing the right shelf before scanning it, takes
under 10 µs and barely changes whether 8 or 128 clusters are requested.

A full end-to-end search's cost tracks how long the *query* is, not how many
clusters get probed (that's held fixed). A query ten times longer costs only
about 2.2x as much — sub-linear — because the final re-ranking step only ever
looks at a capped number of candidates no matter how many fed into it. Growing
the *document* side instead, by giving each document more searchable chunks,
doesn't get that cap: cost there scales roughly in step with the extra chunks
(eight times the chunks per document costs about 5.4x as much), because the first
pass scans all of them rather than capping.

![Semantic search latency by stage](docs/benchmarks/distance_estimation.png)

*A guide to the bars. `single_bit` and `multi_bit` are the coarse first pass
and the more precise second pass. Under `top_n_cluster_selection`, `select_nth`
is the current cluster-picking algorithm and `full_sort` is the older
sort-everything baseline it is measured against. `coarse_assignment` contrasts
two ways of assigning a multi-chunk query to its nearest clusters: `serial_hashmap`
walks the query's chunks one at a time, looking each one up individually in a
scattered `HashMap` of clusters, while `batched_matrix` — the current approach —
scores all of a query's chunks at once against a single contiguous matrix of
cluster centroids. That contiguity is what produces the speedup, through better
cache locality and more vectorizable math. `pass1_scoring` peels the first pass
apart into cumulative layers, starting from the bare distance math
(`dot_arithmetic`), then adding the cost of reading the on-disk format
(`plus_archived`), and finally the real scoring data structure (`plus_hashmap`),
which is the closest of the three to what production actually pays. Finally,
`end_to_end_search` and `end_to_end_multichunk` are complete searches that vary
the query length and the per-document chunk count respectively.*

---

## How this compares to the README's published figures

The README publishes single-threaded ballpark throughput figures, and the
measurements here corroborate them:

| Operation | README ballpark | Measured here |
|---|---|---|
| Write (small values, single writer) | ~400–500 ops/s | 440 ops/s — in range |
| `merge` (small values, single writer) | ~400–500 ops/s | 438 ops/s — in range |
| Read (warm, from memory) | 1M–2.5M ops/s | 2.53M ops/s — top of range |
| Read (cold, from disk) | 100k–300k ops/s | 287k ops/s — in range |
| Range / prefix scan | bounded by result size | consistent — see the Scans section |

Write throughput lands squarely in the published range. The WAL is synced on
every write by design — only the value log's sync cadence is tunable — so 440
writes per second is the expected, sync-bound ceiling for a single writer at
2.27 ms per sync on this drive, not a number a configuration change could raise.
`merge` inherits exactly that ceiling, which is the point of its own section
above.

The cold-read figure is corroborated twice over. Two different benchmarks — one
in the reads section, one in the point-lookup section — both measure a
small-value lookup forced onto disk, using different setups, and arrive within
0.02% of each other (287k ops/s from both). Read throughput meets the README's
ballpark on both tiers.

Sharding gives concurrent writers real headroom beyond that single-writer
figure: writes fan out across the buckets (16 by default), so at 440 writes per
second per writer, sixteen writers spread one-per-bucket would land somewhere
around 7k writes per second. That aggregate concurrent number isn't measured
here — this whole report covers single-threaded latency only — and would be
worth a dedicated concurrent-writer benchmark if it's needed.
