# Field-index updates: buffer in memory, write at checkpoint

**Question.** Every field-index update today appends to a memory-mapped file,
and a background compaction claws the space back. Can updates be held in
memory and written once per checkpoint instead, **without unbounded memory
growth**? Which design is best?

**Short answer.** Yes:

- Keep changed bitmap **containers** in an in-memory overlay with a hard byte
  budget, and write them to the file at the checkpoint.
- Start by replacing the in-memory bitmap's storage, which is itself
  append-only and is the worst memory hazard in the index today.
- The background compaction stays, but it runs about a thousand times less
  often.

*Measured on `roaring_bit_map_fixes` at `ec5fb43` (release build, one thread,
file-backed `DynFieldIndex`, no checkpoint or compaction running).*

**Outcome.** Steps 1, 2 and the crash-ordering part of step 3 (3a) are
implemented; the container-granular file format (3b–3d) is deferred as FR-005.
Measured results and how the implementation differs from this proposal are in
[Results](#results) at the end. The rest of this document describes the code as
it was at `ec5fb43`.

## Terms

- **Field index**: one `DynFieldIndex` per indexed field. It maps each distinct
  field value to a **bitmap** of the row IDs that hold it.
- **Slot**: the stable number a distinct value gets (`ordering: BTreeMap<V,
  slot>`, in memory). The bitmap for a slot is stored under that number.
- **Container**: the unit a Roaring bitmap is made of. Row ID `r` lives in
  container `r >> 16` and holds the low 16 bits. A container is an *array*
  (≤ 4,096 values, 2 bytes each), a *bitset* (8 KB) or a *run* list. So one
  container covers 65,536 consecutive row IDs and is at most 8 KB.
- **Blob store** (`BlobStore`): the field's on-disk store, two memory-mapped
  files. `blobs.keys` is a hash table of slot → (offset, length). `blobs.vals`
  is an append-only region of serialised bitmaps.
- **Checkpoint**: every 1.75 s, `run_index_checkpoint` flushes every field's
  files and then records the WAL offset they reflect. After a crash each field
  replays the WAL from that offset on top of what is on disk.

## How an update works today

`FieldIndex::insert(value, row)`:

1. **Load**: read the slot's blob and deserialise the **whole** bitmap into a
   new `RoaringBitmap`.
2. **Mutate**: `bm.insert(row)`.
3. **Store**: serialise the whole bitmap again and `upsert` it. The upsert
   **appends** the new copy at the end of `blobs.vals` and repoints the slot.
   The old copy becomes dead space.

So every update writes the full size of the value's bitmap. Bytes appended per
insert (no checkpoint or compaction):

| Field | Rows | Distinct values | Time per insert | Appended | Live after | Appended per row |
|---|---|---|---|---|---|---|
| bool | 50,000 | 2 | 9.8 µs | 361 MiB | 16 KiB | 7.6 KB |
| bool | 200,000 | 2 | 14.4 µs | **3,076 MiB** | 54 KiB | 16 KB |
| int | 50,000 | 16 | 5.9 µs | 151 MiB | 96 KiB | 3.2 KB |
| int | 50,000 | 1,000 | 3.1 µs | 4.6 MiB | 137 KiB | 96 B |

Updates with the old value unknown (`set`, int field with 16 values) cost
89 µs each and append 13 KB each: they load every bucket to find the row.

Growth is quadratic in a value's row count, so low-cardinality fields (booleans,
status enums) are the worst case. Two mechanisms keep the files bounded:

- **Backpressure**: a write that pushes a field's dead bytes past 64 MiB
  triggers an immediate checkpoint.
- **Compaction**: at a checkpoint, a field over 50% waste is rewritten under
  its write lock, which stalls that field's reads and writes for the
  rewrite and its fsyncs.

At 200,000 rows each boolean write appends about 27 KB, so the field crosses
64 MiB about every 2,400 writes. WAL
replay needs its own inline compaction for the same reason. Before that
compaction existed, a replay over 16k documents reached 2.6 GB and left the
database unopenable.

### The in-memory bitmap is also append-only

`RoaringBitmap::new()` (used for every loaded bitmap, every query result and
every replay batch) is backed by an **anonymous memory map** with the same
append-only layout (`ContainerStore`). `bm.insert(row)` deserialises the
container, changes it, and appends a new copy of it; nothing is reclaimed until
the bitmap is dropped.

| | Result |
|---|---|
| 65,536 inserts into one new bitmap (one container) | 2.6 µs per insert |
| Peak resident memory | 3 MiB → **514 MiB** |
| Size of the finished bitmap, serialised | **8 KB** |

This is real memory, not disk: any code path that inserts many rows into one
in-memory bitmap uses up to 8 KB per insert. WAL replay's `insert_many` does
exactly that (16k rows for one value: up to 128 MiB, transient). Query
operators are not affected in the same way: `or_inplace` builds a fresh bitmap
per operation and drops the old one.

**Consequence for the design:** an in-memory write buffer built on today's
`RoaringBitmap` would cause the memory pressure it is meant to avoid. The
buffer needs a heap-backed bitmap whose containers change in place.

### A crash-safety gap in the same path

Updates write straight into the live shared memory maps. The kernel may write
those pages to disk at any time and in any order, and `BlobStore::flush`
flushes `blobs.keys` **before** `blobs.vals`. After a power loss or OS crash
(not a process crash, which keeps the page cache), a slot can therefore point
at value bytes that never reached the disk. Those bytes read as zeros, which
deserialise as an **empty bitmap without error**. Replay then rebuilds the slot
from the replay window alone, and every row indexed under that value before
the last checkpoint is silently lost from the index.

An in-place hash-table rehash torn by the same writeback is also unrecoverable.
This was found by reading the code and has not been reproduced; it needs a
power-loss test (`dm-flakey` or a VM). Any design that writes at checkpoint
time with an explicit order closes it.

## Options

| | Design | Memory | Bytes appended | Compaction | Effort |
|---|---|---|---|---|---|
| **0** | Today: append the whole bitmap per update | small | whole bitmap **per update** | constant, under the write lock | — |
| **1** | Fully in-memory index, snapshot at checkpoint | **whole index** (unbounded for high-cardinality fields) | whole index per checkpoint | none | medium |
| **2** | Store containers in the engine's own LSM (no-WAL puts, flushed at checkpoint) | memtable budget | dirty containers per flush | reuses LSM and value-log GC | medium |
| **3** | Overlay of dirty bitmaps, write whole bitmaps at checkpoint | bounded by budget | whole bitmap **per dirty value per checkpoint** | rare | small to medium |
| **4** | Overlay of dirty **containers**, write containers at checkpoint | bounded by budget | **dirty containers** per checkpoint | rare | medium |

**Option 1** is rejected on the memory constraint. Its size is the whole index,
and a high-cardinality field (one value per document, such as a unique tag)
holds one container per value: tens of millions of entries.

**Option 2** gets garbage collection for free but puts every query's container
reads behind LSM lookups and value-log reads. A predicate reads every container
of every value it touches, which today is a zero-copy slice of a mapped file.
It also couples the index's durability to namespace flush scheduling. Rejected
on read latency, though it remains the fallback if a custom store becomes a
maintenance burden.

**Options 3 and 4** share the overlay and the budget. They differ in what a
checkpoint writes for a value that changed:

| Workload (2 values, 1.75 s checkpoint) | Today | Option 3 | Option 4 |
|---|---|---|---|
| 200k rows, 1,750 updates per checkpoint | 47 MB | 54 KB | ≤ 54 KB |
| 10M rows, ingest of new rows (dense IDs grow at the end) | ~2 GB | 2.4 MB | **~25 KB** |
| 10M rows, random updates over all rows | ~2 GB | 2.4 MB | ~2.4 MB |

Derived from the measured bitmap sizes: about 27 KB per value at 200k rows,
and 1.2 MB per value at 10M rows (153 bitset containers). Option 4's ingest
figure is one 8 KB container plus a 4 KB directory per value. Option 3's cost
grows with the bitmap. Option 4's cost grows with what changed, and ingest,
the common bulk case, changes only the last container of each value.

## Recommended design: option 4, in three steps

### Step 1: heap-backed in-memory bitmap

Give `ContainerStore` a heap variant: a sorted map from container key to
`Container`, mutated in place. `RoaringBitmap::new()` uses it, and the memory-map
variant stays for file-backed bitmaps.

- Fixes the 514 MiB case: memory becomes the bitmap's real size.
- Removes two `mmap` calls and two `munmap` calls from every bitmap load and
  every query operator.
- Changes no on-disk format and no crash behaviour.

This step stands on its own, and steps 2 and 3 depend on it.

### Step 2: dirty-container overlay with a hard memory budget

Each `FieldIndex` gains an overlay: per slot, the containers changed since the
last checkpoint (`container key → Container`, or "deleted").

- **Writes** change the overlay only. They load just the one container the row
  lands in: from the overlay if present, else from the file. They no longer
  read or write the whole bitmap.
- **Reads** take each container from the overlay if present, else from the
  file.
- **Old value unknown** (`remove_all_for_row`): check the row's container in
  each slot instead of deserialising every bitmap. It is still O(distinct
  values), but each check is a container probe.
- **WAL replay** goes through the same path. The overlay coalesces the window
  by itself, so replay's group-by-value code and inline compaction can go.

**The memory bound.** A database-wide byte counter covers every field's
overlay. It counts each container's heap size plus a fixed per-entry overhead;
one update adds at most one container per field, so at most 8 KB.

| Level | Default (proposed) | Action |
|---|---|---|
| soft | 32 MiB | request a checkpoint (the existing `IndexCheckpointTrigger`) |
| hard | 64 MiB | the writer that crosses it **spills** its own field synchronously before returning |

A spill writes the field's dirty containers exactly as a checkpoint does, but
does not advance the checkpoint marker. That is safe because the files are
already allowed to run ahead of the marker (replay re-applies writes
idempotently). The hard limit therefore holds even if the checkpoint worker is
slow or stalled. A spill costs an fsync, so it should happen only when more
than 64 MiB of containers change within one checkpoint interval.

### Step 3: container-granular files and an ordered checkpoint write

Change the on-disk layout from "one blob per slot" to "one blob per container":

- `blobs.vals`: appended container blobs, plus per-slot **directories** (a
  sorted list of container key, offset and length).
- `blobs.keys`: slot → its latest directory.

A checkpoint (or a spill) runs per field in four steps:

1. **Write, under the field lock (memory copies only):** append every dirty
   container and each changed slot's new directory past the end of
   `blobs.vals`. No live byte is overwritten. Remember the slot updates, and
   replace the overlay entries with their new file offsets, so readers keep
   seeing them and their heap memory is freed at once.
2. **No lock:** `msync` the appended range of `blobs.vals`.
3. **Under the lock:** apply the slot updates to `blobs.keys`.
4. **No lock:** `msync` `blobs.keys`, then record the checkpoint offset (as
   today).

No fsync happens under a lock. The longest hold is copying at most the budget
(64 MiB) into the map.

**Crash cases:**

- Before step 3, the key table on disk is unchanged.
- During or after step 3, each slot points at either its old directory or its
  new one, and both are durable.
- Replay re-applies everything after the last marker either way.

Three details make this hold:

- **Slots that cannot tear:** 64-byte slots aligned in the file (none crosses a
  sector), each with a checksum. A slot that fails its checksum at open marks
  the field for a full rebuild (the existing `FullRebuild` gap record) instead
  of being trusted.
- **Write position recomputed at open**, as the end of the furthest live blob
  rather than read from the header. A stale header must never let new appends
  overwrite live data.
- **Rehash** into a new key file plus an atomic rename, never in place.

### What garbage collection is left

Each checkpoint still leaves the previous copies of the containers and
directories it replaced, so dead space accumulates. But it grows per changed
container per checkpoint, not per update:

- The boolean field at 200k rows reaches 64 MiB of waste every 2,400 writes
  today. Here it grows by at most 54 KB per checkpoint however many writes the
  checkpoint holds, so about 36 minutes of continuous writes.
- The existing staged-swap compaction can stay as it is, triggered by the
  waste ratio, and now fires rarely.
- If its write-lock stall matters later, a segmented `blobs.vals` (like the
  value log: copy the live containers off a sealed segment, swap pointers
  briefly under the lock) can replace it.

## What changes for users and operators

- `thresholds.index_blob_backpressure_bytes` changes meaning, from dead disk
  bytes to the overlay's soft limit. A new hard limit is added. Both are global
  across fields.
- The file format of field indexes changes in step 3. Under the greenfield
  policy, indexes are rebuilt from the documents (`FullRebuild`) on first open.
  There is no migration.
- Query results, the replay contract and the checkpoint marker are unchanged.

## Tests to add

- The 514 MiB case: peak memory after 65,536 inserts stays within a few times
  8 KB (step 1).
- The budget holds under a write burst across many fields, with the checkpoint
  worker paused: overlay bytes never exceed the hard limit (step 2).
- The tables above re-run as before/after numbers: bytes appended per insert
  and per checkpoint, and update latency.
- A crash at every step of the checkpoint write (`crash_point!`-style
  countdown), with "pages written back" simulated by applying a random subset
  of the dirty pages. Reopen and replay must produce the same index as a full
  rebuild. A power-loss run in a VM covers what a simulation cannot.
- Today's index integration tests and the SIGKILL stress harness, unchanged.

## Open questions

- Budget defaults (32 / 64 MiB) are a guess. Size them against the largest
  realistic write rate per checkpoint.
- Should step 1 ship alone first? It is self-contained, fixes a real memory
  hazard, and makes steps 2–3 smaller to review.
- The power-loss gap exists today. Should it be fixed in the current format
  ahead of step 3 (flush values before keys, and stop the per-update
  shared-map writes), or wait for step 3?

## Results

Implemented on `roaring_bit_map_fixes`: step 1 (`4e63447`), step 2 (`cb46a67`,
`3ac93bd`), step 3a (`571c113`), steps 3b–3d (`0c04268`, FR-005). Measured in
release builds against `0270020` (the last commit before any code change), with
separate target directories, unless a section names another baseline.

### Write volume (step 2)

File-backed field, one spill every 1,750 writes (1,000 writes/s at the 1.75 s
checkpoint interval):

| Field | Appended before | Appended after | Time per write before → after |
|---|---:|---:|---|
| bool, 200k rows | 3,076 MiB | 3.6 MiB | 14.5 → 0.03 µs |
| bool, 50k rows | 361 MiB | 0.43 MiB | |
| 16 values | 151 MiB | 1.5 MiB | |
| 1,000 values | 4.6 MiB | 2.7 MiB | |

The 1,000-value field gains least: each spill rewrites most of its many small
bitmaps.

### Write volume past the hard limit (steps 3b–3d)

With whole bitmaps in the overlay, a field whose changed bitmaps exceed the hard
limit spilled its whole bitmap on every write. With container-granular files a
spill appends only the changed containers and one directory per changed bitmap.
Test `past_the_hard_limit_each_spill_appends_containers_not_bitmaps`: a boolean
with 32 containers per value (264 KB per bitmap) and an 8 KiB hard limit, so
every write spills. Each write moves one row between the two values and appends
18.2 KB (two 8 KiB containers and two 32-entry directories), where spilling
both whole bitmaps would append about 527 KB.

### Memory and insert cost (step 1)

- 65,536 inserts into one in-memory bitmap: 2.9 µs → 0.004 µs each, and
  +384 MiB of RSS → no growth.
- Index insert, file-backed, 50k rows: bool 10.1 → 2.1 µs, 16 values 5.9 →
  0.96 µs, 1,000 values 3.2 → 0.23 µs.

### Query cost

`bench_predicate` became 8–21× faster with step 1, because a query no longer
builds its working bitmaps in an anonymous memory map: `str_eq` 11.5 → 0.55 µs,
`int_range` 1.11 ms → 134 µs, `three_way_and` 799 → 88 µs. Steps 2 and 3a left
it unchanged within noise. With indexes written to and read from their files
(at `571c113`): `str_eq` 0.55 µs, `int_range` 148 µs, `compound_and` 2.2 µs,
`three_way_and` 93 µs, `parse_and_eval` 88 µs.

Steps 3b–3d, against `b29d1ae` (alternated, two rounds): equal or faster on every
case. `str_eq` 635 → 345–366 ns, `int_range`
140–145 → 135–137 µs, `compound_and` 2.26–2.29 → 1.98–2.04 µs, `three_way_and`
89–92 → 84–86 µs, `parse_and_eval` 82–84 → 79–82 µs.

### Crash safety (step 3a)

The power-loss gap described above is closed in the current file format: a
spill appends, syncs the values, then points the slots and syncs them; slots
carry a checksum; open repairs torn slots and reconciles the bitmap and keymap
stores; damaged files trigger a full rebuild of the field. A checkpoint adds one
value-region `msync` per store and no measurable cost on the write path.

Steps 3b–3d keep that write order and add a checksum to each directory; open
treats a torn directory, or one that lists a container past the end of the file,
like a torn slot. Test `every_spill_phase_crash_recovers_to_the_ground_truth`
crashes after each spill phase (stage, sync values, commit, sync keys), with
each unsynced 4 KiB page taken at random from the old or the new image, over 24
seeds (half of them compacted first). Every image reopens, replays the window
since the last durable spill, and matches the ground truth; removing the replay
makes it fail.

### Restart time (WAL replay)

After a crash, each field index replays the WAL written since its last
checkpoint. It scans that stretch of the WAL, reads the current value of every
key it touched, and updates the bitmaps. Measured with the ignored tests
`replay_cost_vs_window` and `replay_cost_vs_total` in `db/database_tests.rs`:
one string field with two values, 6,000 documents, baseline and new binaries
alternated over two rounds:

| Keys to replay | Before | After |
|---:|---:|---:|
| 750 | 5.9–6.1 ms | 2.8–2.9 ms |
| 1,500 | 9.7–10.4 ms | 5.3–5.5 ms |
| 3,000 | 18.6 ms | 10.4–10.5 ms |
| 6,000 | 42–56 ms | 26–29 ms |

Replay now costs about 3.7 µs per key. Of the 2.8 ms for 750 keys, reading the
keys' current values takes 2.3 ms and the WAL scan 0.2 ms. The bitmap updates
take under 0.1 ms. The per-key cost does not grow with the store: 750 keys
replay in 2.9–3.5 ms over 6,000, 12,000 and 24,000 documents. These tests leak
each crashed database, whose background threads keep running. That adds about
9 ms, outside the replay itself, to whichever case runs third, so the raw test
output shows a jump at the third size whatever its order.

What that means at the default 1.75 s checkpoint interval: a durable write
fsyncs under the one WAL lock, so a database takes at most about 440 single-op
writes per second, and a checkpoint leaves at most about 770 writes to replay.
That is about 3 ms of replay per indexed field, against about 6 ms before. If
the checkpoint falls behind, each second of lag adds about 1.6 ms per field. A
longer interval would scale restart time linearly (a 15-minute interval: up to
about 400,000 keys, roughly 1.5 s per field, extrapolated rather than measured)
and would also hold that much more WAL.

### How the implementation differs from the proposal

- **Slots stay 48 bytes**, not the proposed 64. They carry a CRC-32 instead,
  so a slot torn across a sector boundary is detected and dropped at open.
- **Directories are flat**: one directory per bitmap, rewritten whole at each
  spill that changes the bitmap (28 bytes per container, so 4.3 KB at 10M rows).
  A two-level directory would only matter for bitmaps with far more containers.
- **`index_blob_backpressure_bytes` keeps its meaning** (dead disk bytes per
  field). The overlay got two new settings instead, `index_overlay_soft_bytes`
  and `index_overlay_hard_bytes`, shared by all fields of a database. The
  backpressure valve now rarely fires.
- **A hard-limit spill syncs on the writer's thread**, under the field's write
  lock, so the write that crosses the limit pays two `msync`s. The checkpoint
  path syncs under a read lock.

### Open questions, as resolved

- Budget defaults stay at 32 / 64 MiB, now measured (see *1M-document test*
  below): the write buffers of four indexed fields peak at about 8.5 MB at the
  server's top write rate, so the soft limit has about 4× headroom.
- Step 1 shipped together with steps 2 and 3a on one branch.
- The power-loss gap was fixed in the current format (step 3a), ahead of and
  independent of the deferred format change.

### 1M-document test

A scratch API server (release build, NVMe disk) with a uuid-keyed document
store of four indexed fields: `active` (bool, 70% true), `tier` (16 values),
`country` (200 values, skewed) and `price` (50,000 values). Documents come from
a seeded generator that also writes the expected count for every value.

**Load and correctness, WAL on.** 1M documents in 39 min 45 s, about 416
writes/s (one WAL fsync per write; more client threads do not help). Write
buffers peaked at 0.58 MB during the load and 4.75 MB during 200k random
updates. After the load, and again after the updates and 50k deletes (950,000
documents), every checked query — equality on each field, ranges, `IN`, `AND`,
`OR`, `NOT` — returned the generator's count, with no degraded fields. A
`kill -9` in the middle of the updates came back healthy in 2.88 s (12,845 WAL
entries replayed), with no gap on any field. On disk: field indexes 48 MB (most
of it `price`'s 50,000 directories), row map 75 MB, whole database 293 MB.

**Write-buffer budget.** The same 1M documents and 200k updates without the WAL
(about 5,000 writes/s), on a fresh store per setting:

| Soft / hard | Load | Updates | Peak | Soft crossings | Hard spills |
|---|---:|---:|---:|---:|---:|
| 2 / 4 MiB | 5,287/s | 2,393/s | 4.2 MB | 926 | 1 (8.9 ms) |
| 8 / 16 MiB | 5,251/s | 5,103/s | 8.4 MB | 2 | 0 |
| 32 / 64 MiB (default) | 5,259/s | 5,123/s | 8.5 MB | 0 | 0 |

Random updates fill the buffers far faster than appends, which only touch each
value's last container. The default never fired; a limit below the natural
peak only adds early checkpoints, which at 2/4 MiB halved the update rate.

**Query latency** (HTTP, page of 20, p50, 950,000 documents): 0.3–0.5 ms for
equality, `AND`, `OR` and `NOT`; 0.8–1.1 ms for `IN` and `tier >= 8`; 5.1 ms for
`price < 1000`, which merges about 1,000 values' bitmaps. Evaluating a
predicate on the index itself takes 0.3–3 µs; the rest is fetching the page's
documents. That fetch took 12–13 ms per query before `4ef7c04`, because the
batch read loaded each touched bucket's whole L1 SSTable.

**uuid keys before and after** (100k documents, `b29d1ae` against `4ef7c04`):
`b29d1ae` took row IDs from the uuids and read whole SSTables per query page.

| | `b29d1ae` | `4ef7c04` |
|---|---:|---:|
| `active` bitmap, live bytes | 3.4 MB | 33 KB |
| `tier` / `country` bitmaps | 3.4 MB each | 198 / 202 KB |
| `active = true` | 6.2 ms | 0.39 ms |
| `tier >= 8` | 24.7 ms | 0.47 ms |
| `price < 1000` | 66.9 ms | 0.95 ms |
| `NOT active = true` | 38.0 ms | 0.36 ms |

`price` stays at about 4 MB either way: its 50,000 values hold about two rows
each, which no row-ID scheme can pack.

**`bench_predicate`** at `4ef7c04` against `b29d1ae` (alternated, two rounds):
equal or faster on every case — `str_eq` 631–686 → 367–370 ns, `compound_and`
2.35 → 2.02–2.04 µs, `int_range` 143–144 → 139 µs, `three_way_and` 89–92 →
87 µs; AND at 6% selectivity within noise (58.6–60.4 vs 59.1–60.7 µs).

