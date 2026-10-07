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
