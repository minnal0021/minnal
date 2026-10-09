# Index Architecture

This document describes the field index end to end: the custom RoaringBitmap
engine at its core, how per-field indexes are defined and built on top of it, how
they are stored and kept compact, and how the whole structure is brought back
after a crash. It is meant to be read in order — each section builds on the one
before it, from the raw bitmap up to the recovery model.

Two parts of `minnal_db` share the work. The bitmap engine itself — `bitmap.rs`,
`container/`, `container_store.rs`, `blob_store.rs`, `rowmap.rs`, `storage.rs`,
all under `src/index/` — knows nothing about databases. The *lifecycle* around
it — schema registration, activation, checkpointing, WAL replay — lives in the
engine (`src/db/database.rs`, `kv_store.rs`, `index_manager.rs`), which owns one
set of indexes per namespace. Paths below are relative to `minnal_db/src/index/`
unless they start with `minnal_db/`.

---

## The shape of the whole thing

At the top sits a query DSL; at the bottom sits a hand-rolled RoaringBitmap. In
between, each indexed field owns a structure that maps field values to the sets of
documents that hold them:

```
                        QUERY DSL  (query/: lexer → parser → eval)
                              │  "age > 30 AND status = \"active\""
                              ▼
        ┌───────────────────────────────────────────────┐
        │ DynFieldIndex  (one per indexed field)          │
        │   ordering: BTreeMap<Value, slot_id>  (heap)    │  ← sorted, drives ranges
        │   overlay:  slot_id → changed containers (heap) │  ← changes not yet written
        │   bitmaps:  BlobStore  slot_id → container dir  │  ← off-heap, on disk
        │   keymap:   BlobStore  slot_id → value bytes    │  ← rebuilds ordering on open
        └───────────────────────────────────────────────┘
                              │  each bitmap is a …
                              ▼
        ┌───────────────────────────────────────────────┐
        │ RoaringBitmap  (custom, u128 keyspace)          │
        │   ContainerStore  high-key → Container          │
        │     Container = Array | Bitset | Run            │
        └───────────────────────────────────────────────┘

        RowMap (one per namespace)   key ⇄ dense row-id (0,1,2,…)
            └─ the row IDs that get inserted into the bitmaps above
```

Two design choices shape everything that follows, and are worth holding in mind
before the details.

The first is that **the RoaringBitmap is written from scratch.** There is no
`roaring` crate dependency; the module uses only `memmap2`, `parking_lot` and
`rkyv`. The entire container model (array, bitset, run, with
promotion and demotion between them), every cross-type set operation, and the
memory-mapped store underneath are all implemented here, in `bitmap.rs` and
`container/`.

The second is that **bitmaps live off-heap and are materialised only on demand.**
A field index keeps just a small `BTreeMap` resident in memory. Each value's
bitmap is stored container by container in a memory-mapped `BlobStore`, and is
decoded into a live `RoaringBitmap` only when a query actually touches it. This is
what lets a namespace carry many large indexes without a proportional heap cost.
The one exception is a container that a write has changed: it stays in memory
(the field's **overlay**) until the next checkpoint writes it out, under a memory
budget shared by every field of the database.

The sections below walk up this stack: first how a field comes into existence,
then the row IDs that populate its bitmaps, then the bitmap engine itself, then
the per-field structure built on top, and finally the disk format, compaction,
checkpointing, and recovery that keep it all durable.

---

## 1. Defining a field

An index begins with a field definition (`minnal_db/src/db/namespace.rs`):

```rust
pub struct FieldDef {
    pub field_id: FieldId,           // unique u32, monotonically increasing, never reused
    pub field_name: String,
    pub field_type: IndexValueType,  // Bool | Int | Str
}
```

Fields are registered through `Database::register_index_field`, which is
idempotent: registering the same name-and-type pair twice returns the existing
`FieldId` rather than allocating a new one. A fresh registration assigns the next
monotonic `FieldId`, creates the field's on-disk home at
`{db_path}/index/{namespace_id}/{field_id}/`, and persists the updated field list
to `{db_path}/ns_{namespace}/config.json`.

That `config.json` is the authoritative schema record. It is written atomically on
every registration and loaded on every `Database::open()`, so the set of indexed
fields survives restarts without any separate bookkeeping.

A field's type is one of three, drawn from `IndexValueType` in
`src/field/value.rs`: `Bool`, `Int` (an i64), and `Str`. These match the variants
of the `IndexValue` enum that the rest of the crate operates on.

---

## 2. Dense row IDs — the `RowMap`

A field-index bitmap is a set of **row IDs**, not keys. Which IDs those are turns
out to be the single biggest lever on how large the index becomes, so it is worth
understanding before the bitmaps themselves.

The reason comes from how a RoaringBitmap is laid out (see §3): it groups values
by their high bits into *containers*. If row IDs are scattered across the `u128`
space — as they would be if derived by hashing each key — then almost every
document falls into its own high-key bucket, and the bitmap degenerates into
roughly one container per document: pathologically sparse, and enormous.
Assigning **dense, monotonic** IDs instead (`0, 1, 2, …`) means consecutive
documents share a high key and pack into a single container. That packing is the
entire reason the bitmaps stay small, and it is why dense IDs from the `RowMap`
are the default.

`RowMap` (`src/rowmap.rs`) is a per-namespace, memory-mapped sidecar in three
parts:

| Part | Backing | Role |
|---|---|---|
| `key → id` | open-addressing table in `rows.slots` (a file, so the kernel can page it out) | `O(1)` lookup on every put/delete/replay; **rebuilt from the id array on every open** — the file is never trusted as a source of truth |
| `id → key` | `rows.idarray` (append-only, indexed by ID) → `rows.keybytes` (append-only key bytes) | `O(1)` resolution of query hits back to keys |
| counter | `next_id`, stored in the `rowmap.ckpt` marker | next dense ID to assign |

The `key → id` table keys on the **full key bytes** — FNV-1a picks only the probe
start, after which keys are compared byte for byte — so two keys can never
collide onto the same ID. Entries are **never removed**: a key that is deleted and
later recreated reuses its original ID. That gives the table a useful invariant —
no tombstones, and `count == next_id` at all times. It also means the map grows
with every key ever written. IDs are freed only by resetting the whole map
(`Database::reset_rowmap`), which is allowed only when no field index of the
namespace exists — a freed ID handed to another key would otherwise match
whatever an old bitmap still says about the deleted one. The document store's
`reindex-all` resets the map between dropping and rebuilding its indexes, so
the rebuilt indexes number the live documents only.

When it comes time to assign an ID for a key, `KVStore::resolve_row_id_alloc`
(`kv_store.rs`) consults two sources in order:

1. A caller-supplied `RowIdFn` wins if present — an escape hatch for keys that
   embed their own ID, paired with a `RowToKeyFn` for the reverse direction. It
   suits keys that are already dense or clustered; the document store uses it
   for `u64` keys only. Random keys (v4 UUIDs, say) would put every row in a
   container of its own, so every value's bitmap would hold one container per
   document.
2. Otherwise the dense `RowMap`, the normal path.

These are the only two sources: a namespace's row map is always loaded before any
field index is activated (`ensure_rowmap` in `activate_field_index`), so a write
that reaches index maintenance always has one or the other available.

On disk the row map lives at `{db_path}/index/{namespace_id}/rowmap/`, a sibling
of the per-field directories. Because the directory is named `rowmap` rather than
a number, it can never collide with a numeric `FieldId`.

---

## 3. The RoaringBitmap engine

With row IDs in hand, we can look at what actually stores them. A RoaringBitmap
is a compressed set of integers; this crate's version operates over a `u128`
keyspace and picks its internal encoding adaptively as the data changes.

### 3.1 Keyspace and containers

A `RoaringBitmap` (`src/bitmap.rs`) holds a set of `u128` values. Each value is
split (`decompose`) into a **high key** — the upper 112 bits — and a **low
value** — the lower 16 bits:

```
value: u128  ──►  high (112 bits) ──► selects a container
                  low  (16 bits)  ──► the bit within that container
```

The high key selects a `Container`, and the low 16-bit value is the specific bit
stored inside it. A `Container` (`src/container/mod.rs`) comes in three encodings,
each suited to a different data shape, and the engine moves between them
automatically as a container's contents change:

| Container | Backing | Best for | Boundary |
|---|---|---|---|
| `ArrayContainer` | sorted `Vec<u16>` | sparse (few values) | promotes to `Bitset` at `ARRAY_TO_BITSET_THRESHOLD` (4096) elements |
| `BitsetContainer` | fixed `[u64; 1024]` (8 KB) | dense | demotes back to `Array` below the threshold |
| `RunContainer` | run-length `(start, len)` pairs | long consecutive runs | chosen by `optimize()` / set ops when `is_efficient()` (fewer bytes than the alternatives) |

`Container::insert` and `remove` promote and demote across the array/bitset
boundary as cardinality crosses the threshold, while `Container::optimize()`
re-evaluates all three encodings at once — including switching a container to or
from the run form when that is smaller. Cardinality itself is free to read:
`cardinality()` is `O(1)`, because the running bit count is cached in the store
header and maintained on every mutation.

### 3.2 Set operations

On top of the containers, `RoaringBitmap` offers the full complement of set
algebra: `and` / `or` / `and_not` and their in-place variants, `flip`,
range-scoped `range_and` / `range_or`, `rank` / `select`, and the bulk
constructors `from_sorted_iter` / `from_unsorted_iter`. A binary operation walks
the two bitmaps' high keys in sorted-merge order at the top level, and for each
high key that both share it hands the work down to the container layer.

That container layer, in `container/ops.rs`, implements **all nine cross-type
combinations** of the binary operations — Array×Array, Array×Bitset, Bitset×Run,
and so on — each producing whichever container type best fits its result and
promoting or demoting as needed. The hot loops lean on the SIMD helpers in
`src/simd_support/`: popcount, bitwise AND/OR/AND-NOT, sorted-array merge, and bit
extraction.

Each helper carries three implementations selected per target: **AVX-512** and
**AVX2** on x86_64 (chosen at runtime via `is_x86_feature_detected!`, so one
binary runs correctly on any x86 CPU), **NEON** on aarch64 (Apple Silicon / ARM64
— NEON is baseline on every aarch64 target, so it is compiled in unconditionally
with no runtime probe), and a portable **scalar** fallback everywhere else and for
the sub-vector tail. All three are unit-tested against the scalar reference, so
the vectorised paths are drop-in equivalents that only change speed, not results.

### 3.3 `ContainerStore` — the mmap backing

Each `RoaringBitmap`'s containers live in a `ContainerStore`
(`src/container_store.rs`), a two-file, memory-mapped map from `u128` high key to
`Container`:

```
containers.keys   64-byte header (magic "MINNALBI", version, capacity, count,
                  tombstone count, cardinality, value_write_pos)
                  + 48-byte open-addressing linear-probing slots
                    (state EMPTY/OCCUPIED/TOMBSTONE, high-key, val offset, val len)
containers.vals   append-only rkyv-serialised Container blobs, 16-byte aligned
```

A store has one of two backings behind the same API:

- **Heap** — a sorted map of containers changed in place, with a cached total
  count. Every transient bitmap uses it: a bitmap built in memory, the result of
  a set operation, and a field-index bitmap loaded from its blob for a query or a
  write. Its memory is the bitmap's real size.
- **File-backed** (`RoaringBitmap::create` / `open`) — the two-file layout above,
  for a bitmap that is its own persistent object. Growth is handled by
  remapping. Values in `containers.vals` are **append-only**: every change writes
  a new copy of the container, and the old one is not reclaimed. That is why
  transient bitmaps never use this backing — 65,536 inserts into one container
  would leave about 500 MiB of copies behind an 8 KB bitmap.

---

## 4. The field index

A single bitmap answers "which rows hold this exact value". A field index
(`FieldIndex<V>`, `src/field/field_index.rs`) is the layer that turns a whole
field into many such bitmaps and routes queries to the right ones:

```
ordering:  BTreeMap<V, u128>   // value → slot_id  (sorted; drives range queries)
bitmaps:   BlobStore           // slot_id → directory of that value's containers
next_slot: u128
```

The `ordering` map is kept on the heap — the values and their slot IDs, but
none of the bitmap data. The bitmaps live in a `BlobStore` (`src/blob_store.rs`),
which uses the same two-file mmap layout as `ContainerStore` (a header
"MINNALBS", 48-byte slots, and an append-only `blobs.vals`) but stores byte
blobs. In a field's bitmap store, each container of each bitmap is its own blob,
and a value's slot points at a small **directory** blob listing where its
containers are (§5).

### Writes go to an in-memory overlay

A write does not touch the `BlobStore`. Inserting `(value, row_id)`:

1. Find or allocate the `slot_id` for `value` in `ordering`.
2. Split `row_id` into its container key (upper 112 bits) and its low 16 bits.
3. If the field's **overlay** (`slot_id → container key → Container`, on the
   heap) does not hold that container yet, load it: binary-search the slot's
   stored directory in place for the container key, and decode just that
   container's blob. A container the bitmap does not have yet starts empty.
   The overlay never copies a directory, so its memory follows the containers
   that changed, not the size of the bitmaps.
4. Insert the low 16 bits into the overlay's copy of the container, in place.

Reads consult the overlay first and fall back to the store, container by
container, so a query always sees every write. Removal is the mirror image; a
container that empties is recorded as removed, and a bitmap whose containers
have all gone is empty. A document **update** removes the row from its old
value's bitmap and inserts it under the new one. When the caller knows the old
value — the document store always does — that touches one bitmap
(`DynFieldIndex::update`). When it does not, `remove_all_for_row` clears the row
from *every* value bucket; it checks membership by decoding only the container
the row falls in, and adds only the buckets that held the row to the overlay.

The overlay is written to the store by a **spill**, which appends each changed
container plus a new directory for its bitmap, and points the slot at the new
directory (§6). A spill happens at every index checkpoint (§7), and sooner if
memory runs short. A bitmap that empties keeps its slot until the spill, which
then frees it.

### The memory budget

Every field of a database charges its overlay to one shared counter (an
`IndexOverlayBudget`, `src/overlay_budget.rs`) with two limits:

| Setting (TOML `[thresholds]`) | Default | When crossed |
|---|---|---|
| `index_overlay_soft_bytes` | 32 MiB | the write path asks for an index checkpoint now, which spills every field |
| `index_overlay_hard_bytes` | 64 MiB | the write that crossed it spills its own field before returning |

The hard limit keeps memory bounded even when checkpoints fall behind, at a
cost to that one write: it appends and syncs the field's changes while holding
the field's write lock. Total overlay memory stays below the hard limit plus,
for each field written after it was crossed, the containers that one write
changed (a container is at most 8 KB).

Because the overlay holds containers, not whole bitmaps, a write that has to
spill writes only what changed: on a boolean over hundreds of millions of rows a
spill appends one or two containers and their bitmaps' directories, not the
bitmaps.

`GET /admin/storage/ops-metrics` reports the budget under `index_overlay`: bytes
held now and at peak, both limits, how often the soft limit was crossed, and how
many writes spilled past the hard limit and for how long.

Queries run through `evaluate`, which uses `ordering` to locate the slots a
predicate needs and then OR-folds their bitmaps together. The lookup shape follows
the operator: a single `get` for `Eq` and `In`, a `BTreeMap::range` for the
ordered comparisons (`Lt`/`Le`/`Gt`/`Ge`/`Between`), and a full scan minus the one
excluded slot for `Ne`.

### Rebuilding `ordering` on open — the keymap

Because `ordering` is heap-only, nothing on disk records it directly, so it has to
be reconstructed at startup. That is the purpose of a second `BlobStore`, under
`keymap/`, which persists `slot_id → raw value bytes` (one byte for a bool, an
8-byte little-endian i64, or UTF-8 for a string). Scanning it on open rebuilds
`BTreeMap<V, slot_id>` exactly.

The keymap is written **once per distinct value**, not once per document — a new
value's entry is queued and written by the next spill, with the bitmaps — so a
stable set of values never bloats it. Its one failure mode is distinct-value
*churn* — values that appear and are then fully removed, freeing their slots —
which leaves dead entries behind. Those are reclaimed by the same compaction pass
that handles the bitmaps (§6).

---

## 5. On-disk layout

Pulling the pieces together, a namespace's index directory looks like this:

```
{db_path}/index/{namespace_id}/
  rowmap/                      ← per-namespace dense row-ID map (§2)
    rows.keybytes              ← append-only raw key bytes
    rows.idarray               ← append-only id → (key_off, key_len)
    rows.slots                 ← key → id hash table, rebuilt from rows.idarray on open
    rowmap.ckpt                ← marker: magic "MINNALRM", next_id, keybytes_pos, wal_offset
  {field_id}/                  ← one directory per indexed field
    blobs.keys                 ← mmap hash table: slot_id → (offset, len, CRC) of its directory
    blobs.vals                 ← append-only: container blobs, and one directory per bitmap
    keymap/
      blobs.keys               ← mmap hash table: slot_id → (offset, len) into keymap/blobs.vals
      blobs.vals               ← append-only: raw value bytes (1B bool, 8B LE i64, UTF-8 str)
    checkpoint                 ← 8-byte LE u64: WAL offset the files reflect (last checkpoint)
    gap.json                   ← only when the field is missing updates (§8)
```

The bitmap blobs in `blobs.vals` warrant a note, because they are **not** a copy
of the `ContainerStore` mmap. Each container is its own blob (rkyv-serialised
`Container`, 16-byte aligned), and each bitmap is a **directory** blob listing
its containers (`src/storage.rs`):

```
[4B  magic "MBD1"]
[4B  LE u32   count]
count × [16B LE u128 container key][8B LE u64 offset][4B LE u32 len]   (sorted by key)
[4B  LE u32   CRC-32 of everything before it]
```

A directory lets a reader decode one container without the rest of the bitmap,
and lets a spill append only the containers that changed: the new directory
points at the new copies of those and at the existing blobs of the rest. The
checksum lets open-time repair tell a torn directory from a good one (§8); a
container blob is validated by rkyv's checked access when it is read. The
`unaligned` rkyv feature lets the archived containers be read straight out of
the map without an aligned copy. Neither store has a log of its own: both are
derived from the KV data, and the engine's WAL rebuilds them after a crash (§8).

The on-disk format version (in the `MINNALBS` header) is 2. A store with any
other version refuses to open; delete the field's directory and rebuild it.

---

## 6. Spills, dead space and compaction

### What a spill writes

For each changed bitmap, a spill appends the containers that changed and a new
directory to `blobs.vals`, and points the slot at the new directory. The old
directory and the old copies of the changed containers stay in the file as dead
space, because the value region is append-only (§3.3); containers that did not
change are shared by the old and new directory. So the file grows by what
changed at each spill, not at each write and not by the size of the bitmaps: a
value that gains a thousand rows between two checkpoints, all in its last
container, appends one container and one directory.

The keymap store has the same shape of problem in a milder form. It is written
once per distinct value, so a fixed value set never grows it — but distinct-value
churn leaves dead entries, as described in §4. So compaction covers **both**
stores, each triggered on its own `waste_ratio()`. A freed slot's keymap entry is
removed one spill after its bitmap, so the two stores never disagree about a slot
id that is still in use.

### The order of a spill's writes

The files are shared memory maps, which the kernel writes back to disk at any
time and in any order. A slot that reached disk before the blob it points at
would, after a power loss, point at zeros. So a spill runs in four steps:

1. **Stage** — append every changed container, a new directory for each changed
   bitmap, and every new keymap entry. No slot points at them yet; reads keep
   using the overlay.
2. **Sync values** — msync both stores' value regions. If this fails, the spill
   is abandoned and the overlay keeps every change.
3. **Commit** — point the slots at the staged directories and remove emptied slots.
   If a bitmap changed after the stage, only its containers changed since then
   stay in the overlay for the next spill; the ones just written leave it.
4. **Sync keys** — msync both key tables.

A slot is still rewritten in place, so a crash can tear one. Each slot carries a
CRC-32 of its other bytes. On open, a store drops any slot that is torn or points
past the end of its value file — and, in a bitmap store, any slot whose directory
fails its checksum or lists a container past the end of the file — and
recomputes its counts and write position from the slots that remain (§8). Growing a key table writes the larger table to a new
file and renames it into place.

### Compaction

`BlobStore::compact()` reclaims dead space: it rebuilds the value region from the
live slots only — for a bitmap slot, the containers its directory lists followed
by a new directory pointing at their new positions — dropping dead space and
tombstones, and shrinks the file.
`waste_ratio()` reports the reclaimable fraction — alignment padding counts as
live, so it reads ≈0 immediately after a compaction. The trigger is a single
threshold, `ThresholdConfig::index_blob_waste_threshold` (a percentage, default 50,
set in TOML as `thresholds.index_blob_waste_threshold`).

Compaction runs in two places. The main one is `Database::run_index_checkpoint`,
for each field over the waste threshold. That pass is reached four ways, all
running the same code: the periodic `IndexCheckpointWorker` (1750 ms by default),
an early request from the write path (§7), a clean shutdown, and the on-demand
`Db::checkpoint_index()` exposed over REST as `POST
/admin/storage/index-checkpoint`. The other is WAL replay at open (§8), which
compacts a field inline once its dead space passes
`thresholds.index_blob_backpressure_bytes`, because the checkpoint worker has not
started yet. (Do not confuse either with `Db::compact()` / `POST
/admin/storage/compact`, which is the unrelated KV-engine LSM and value-log
compaction.)

### Why compaction must never rewrite in place

Recovery (§8) does not rebuild the index from scratch. It loads the on-disk store
as it stands and replays only the **WAL tail** on top of it. That makes in-place
rewriting during compaction dangerous: a crash that left new key offsets pointing
into a half-rewritten value region would be inherited as **silent corruption** —
any value bucket the replay window happened not to touch would simply stay wrong,
and nothing would ever detect it.

So `compact()` never touches the live files. It builds the compacted pair in
memory and swaps it in through a staged sequence, with a marker file as the atomic
commit point:

```
build compacted (key table, val region) in memory   (~1× live size, read from mmap)
  write blobs.keys.new + blobs.vals.new   → fsync both files + dir   (live pair untouched)
  write compact.commit marker             → fsync dir                ◄── COMMIT POINT
  rename *.new → live; fsync dir; remove marker; fsync dir
  remap onto the new files
```

`BlobStore::open()` calls `recover_compaction()` before anything else. If the
marker is **present**, the staged files are known-complete and the swap is
finished idempotently; if it is **absent**, the staged files are partial and are
discarded. A two-file rename is not atomic as a pair, so it is the marker — not
any lock — that makes the swap atomic.

The invariant this buys is simple and load-bearing: **after any crash the on-disk
pair is fully old or fully new, never a torn mix.** Anyone touching this path must
preserve the ordering (stage+fsync → marker+fsync → rename → drop marker) and keep
`recover_compaction` in `open`.

---

## 7. Checkpointing

Checkpointing is the routine that writes every field's overlay to disk and,
where needed, invokes the compaction of §6. The `index_checkpoint_worker` runs it
every **1.75 seconds** by default (`scheduled_tasks.index_checkpoint_interval_ms`),
and the identical sequence runs on clean shutdown and on demand via
`Database::run_index_checkpoint()`. The write path can also request one early,
for two reasons:

- the overlays together pass `index_overlay_soft_bytes` (§4), or
- one field's reclaimable dead bytes pass `thresholds.index_blob_backpressure_bytes`
  (64 MiB by default, 0 turns this off). Dead space grows per spill, so this
  matters only for a field that keeps spilling at the hard limit between
  checkpoints.

The interval matters beyond disk space. After a crash each field replays the WAL
from its last checkpoint, so the interval bounds how much replay a restart does,
and WAL GC keeps any segment a field still needs for that replay. Each tick first
picks the WAL offset to record — the current tail, but never past a write that
has been appended to the WAL and not yet applied to the index — then processes
every namespace's row map, then each active field in turn:

1. **Flush the `RowMap` first.** `flush_rowmap(wal_tail)` msyncs `rows.*` and
   atomically writes the `rowmap.ckpt` marker. The ordering here is load-bearing:
   the row map is made durable *before* any field index, so it is always at least
   as current as every persisted bitmap bit. Otherwise a crash could leave a
   persisted bit whose row ID the row map could no longer reproduce.
2. **Spill each field** in the four steps of §6. Stage and commit take the
   field's **write** lock; the two syncs run under its **read** lock, so queries
   on the field keep running while it syncs (writes to it wait).
3. **Compact if either store is over threshold.** When `bitmap_waste_ratio()` or
   `keymap_waste_ratio()` is ≥ `index_blob_waste_threshold`, take the field's
   **write** lock and `maybe_compact()` → `BlobStore::compact()` (§6) on whichever
   store(s) qualify. This stalls *that one field's* reads and writes for the
   compaction's I/O — dominated by the fsyncs — and only for fields actually over
   the threshold, not every field every tick.
4. **Record the offset.** Atomically write the chosen WAL offset to
   `{field_path}/checkpoint` via tmp-then-rename.

That final offset is the hinge between checkpointing and recovery: it marks
exactly how much of the WAL is already reflected on disk, which is where the next
section picks up.

---

## 8. Crash recovery

The index is a *derived* structure — everything in it can be reconstructed from
the KV data and the WAL — so recovery is a **checkpoint + WAL replay** model
rather than a rebuild. Each on-disk component is loaded as it was last flushed,
and then the WAL entries written since its checkpoint are replayed to bring it
current.

```
Database::open()
  ├── 1. Load WAL metadata (CRC-checked); if corrupt, rename → .corrupt, start fresh
  ├── 2. Recover next sequence: max(WAL scan, metadata hint) + 1
  ├── 3. Load namespace schemas from config.json
  ├── 4. Open KVStores
  └── 5. recover_from_wal()                       ← data-layer recovery
        ├── Scan WAL head→tail; replay un-persisted Upsert/Delete into the KVStore
        ├── Flush and compact affected stores
        └── Mark entries Persisted, flush WAL metadata

Per indexed namespace:
  RowMap::open()                                  ← row-map recovery
    ├── Read rowmap.ckpt → (next_id, keybytes_pos)
    ├── Rebuild the key→id table from rows.idarray[0..next_id]
    └── Ignore any torn tail appended past the marker

activate_field_index()                            ← index-layer recovery
  ├── BlobStore::open() → recover_compaction() (finish or discard a staged swap)
  │                      → drop torn (CRC) or out-of-range slots, and slots whose
  │                        directory is torn or out of range; recompute header
  ├── Rebuild ordering from the keymap store
  ├── Reconcile the two stores: drop values with no bitmap, bitmaps with no value;
  │       new slot ids start above both
  ├── Any slot dropped as damaged → record a full-rebuild gap
  ├── Read checkpoint file → last_flushed_wal_offset
  ├── Scan WAL last_flushed_wal_offset → tail; collect this namespace's keys
  ├── For each key: remove old index entries, re-insert the current KVStore value
  │       (grouped by value; re-allocating dense row IDs continues deterministically
  │        from next_id, because replay is in sequence order; bounded by the overlay
  │        budget, compacting inline if dead space grows large)
  └── Flush the index before making it visible (under Arc<RwLock>)
```

The order across these three passes is what makes them safe. The KV data is
replayed before any index is activated, so the index always re-inserts from
current values. The row map is recovered before the fields, so every replayed
insert has a stable ID to reuse. And because replay runs in sequence order, a
freshly allocated dense ID picks up deterministically from `next_id` — the same ID
a key would have received the first time.

Each component's guarantee, in one place:

| Component | Recovery mechanism |
|---|---|
| Field schema | `config.json` written synchronously on registration |
| KVStore data | WAL replay in `recover_from_wal()` before any index activation |
| Row map | `rows.*` + marker loaded; key→id table rebuilt from the id array; torn tail ignored |
| Index bitmaps + keymap | mmap loaded (with `recover_compaction`); torn slots dropped; the two stores reconciled; then WAL tail replayed from the last checkpoint offset |
| Spill in flight | slots are pointed at blobs only after the blobs are synced, so a crash leaves at worst unreferenced bytes; replay restores the changes |
| Compaction in flight | `compact.commit` marker → finish the staged swap; absent → discard `*.new` |
| Checkpoint markers | tmp-then-rename; a partial write is never observed |
| Sequence numbers | `max(WAL scan, metadata hint) + 1` |

All of this rests on WAL replay being idempotent and on there being no transaction
that spans multiple operations: each write's intent is durable in the WAL before
the operation is acknowledged, so replaying it twice is harmless and replaying it
once is enough.

### Why recovery lives inside `activate_field_index`

The replay step above deliberately runs *inside* field activation rather than in
a pass of its own. A separate recovery pass would leave a window between
recovering the index and registering it, and a put arriving in that window could
be overwritten by the replayed state. Replaying inside `activate_field_index()`,
before the index is wrapped in `Arc<RwLock>` and made visible to anyone, leaves
no such window.

### When replay is not possible: gaps

Replay needs the WAL from the field's checkpoint onwards. WAL GC keeps those
segments, up to a limit (`thresholds.max_pinned_wal_segments`, 32 by default).
Past the limit it deletes the oldest anyway rather than let the WAL grow without
bound, and first records which keys those segments touched. That record is a
**gap**, written to `gap.json` beside the field's checkpoint. Three other
situations also record one: writes made without the WAL (bulk loads with
`skip_wal`) followed by an unclean shutdown, an index update the write path
could not apply, and index files that held damaged slots when opened (the rows
those slots held may predate the replay window, so the field is rebuilt).

A field with a gap stays queryable, but its answers may be missing rows:

- Queries that use the field list it in `degraded_fields`, so a caller can tell
  "nothing matched" from "the index is incomplete". The document store also
  lists a field whose index build is still running or failed: such an index
  holds only the documents the build reached.
- `GET /admin/indices/{ns}/health` lists every degraded field.
- `POST /admin/indices/{ns}/attribute/{field}/repair` re-indexes the keys the gap
  names, or rebuilds the field when the keys could not be captured, then
  checkpoints and clears the gap.
