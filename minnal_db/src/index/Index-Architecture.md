# Index Architecture

This document describes the `index` crate end to end: the custom RoaringBitmap
engine at its core, how per-field indexes are defined and built on top of it, how
they are stored and kept compact, and how the whole structure is brought back
after a crash. It is meant to be read in order — each section builds on the one
before it, from the raw bitmap up to the recovery model.

Two crates share the work. The bitmap engine itself — `bitmap.rs`, `container/`,
`container_store.rs`, `blob_store.rs`, `rowmap.rs`, `storage.rs` — lives in the
`index` crate and knows nothing about databases. The *lifecycle* around it —
schema registration, activation, checkpointing, WAL replay — lives in `minnal_db`
(`src/db/database.rs`, `kv_store.rs`, `index_manager.rs`), which owns one set of
indexes per namespace. File and line references throughout point at the code as
of this writing.

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
        │   bitmaps:  BlobStore  slot_id → serialised bm  │  ← off-heap, on disk
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
`roaring` crate dependency — `index/Cargo.toml` pulls only `memmap2`,
`parking_lot`, and `rkyv`. The entire container model (array, bitset, run, with
promotion and demotion between them), every cross-type set operation, and the
memory-mapped store underneath are all implemented here, in `bitmap.rs` and
`container/`.

The second is that **bitmaps live off-heap and are materialised only on demand.**
A field index keeps just a small `BTreeMap` resident in memory. Each value's
bitmap is a serialised byte blob in a memory-mapped `BlobStore`, deserialised into
a live `RoaringBitmap` only when a query or a write actually touches it. This is
what lets a namespace carry many large indexes without a proportional heap cost.

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
| `key → id` | in-memory open-addressing table (anonymous mmap) | `O(1)` lookup on every put/delete/replay; **rebuilt from the id array on open** — never a persisted source of truth |
| `id → key` | `rows.idarray` (append-only, indexed by ID) → `rows.keybytes` (append-only key bytes) | `O(1)` resolution of query hits back to keys |
| counter | `next_id`, stored in the `rowmap.ckpt` marker | next dense ID to assign |

The `key → id` table keys on the **full key bytes** — FNV-1a picks only the probe
start, after which keys are compared byte for byte — so two keys can never
collide onto the same ID. Entries are **never removed**: a key that is deleted and
later recreated reuses its original ID. That gives the table a useful invariant —
no tombstones, and `count == next_id` at all times.

When it comes time to assign an ID for a key, `KVStore::resolve_row_id_alloc`
(`kv_store.rs`) consults two sources in order:

1. A caller-supplied `RowIdFn` wins if present — an escape hatch for keys that
   embed their own ID (a UUID, say), paired with a `RowToKeyFn` for the reverse
   direction. Such IDs are scattered rather than dense, so this trades bitmap
   compactness for a stable, caller-controlled identifier.
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

A store is either **anonymous** — an in-memory mmap used for transient bitmaps,
such as set-operation results and the field-index bitmaps materialised on demand —
or **file-backed**. The public API is identical either way, and growth is handled
transparently by remapping. The one rule that matters for correctness later is
that `value_write_pos` only ever advances: values are **append-only** within a
store, never overwritten in place.

---

## 4. The field index

A single bitmap answers "which rows hold this exact value". A field index
(`FieldIndex<V>`, `src/field/field_index.rs`) is the layer that turns a whole
field into many such bitmaps and routes queries to the right ones:

```
ordering:  BTreeMap<V, u128>   // value → slot_id  (sorted; drives range queries)
bitmaps:   BlobStore           // slot_id → serialised RoaringBitmap bytes
next_slot: u128
```

Only the `ordering` map is kept on the heap — the values and their slot IDs, but
none of the bitmap data. The bitmaps live in a `BlobStore` (`src/blob_store.rs`),
which uses the same two-file mmap layout as `ContainerStore` (a header
"MINNALBS", 48-byte slots, and an append-only `blobs.vals`) but stores arbitrary
byte blobs rather than containers.

Inserting `(value, row_id)` walks that structure top to bottom:

1. Find or allocate the `slot_id` for `value` in `ordering`.
2. `load_bitmap(slot_id)` reads the blob and `storage::deserialize`s it into an
   in-memory (anonymous) `RoaringBitmap`.
3. `bitmap.insert(row_id)`.
4. `store_bitmap` re-serialises the whole bitmap and hands it to
   `BlobStore::upsert`, which **appends** the new blob.

Removal is the mirror image, and when a bitmap empties out its slot is freed
(`remove_key`) and its `ordering` entry dropped. One helper deserves a name:
`remove_all_for_row` clears a single row from *every* value bucket at once, which
is how a document **update** is handled — clear the row from its old values, then
insert it under the new ones.

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

The keymap is written **once per distinct value**, not once per document, so a
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
    rowmap.ckpt                ← marker: magic "MINNALRM", next_id, keybytes_pos, wal_offset
  {field_id}/                  ← one directory per indexed field
    blobs.keys                 ← mmap hash table: slot_id → (offset, len) into blobs.vals
    blobs.vals                 ← append-only: serialised RoaringBitmap blobs
    keymap/
      blobs.keys               ← mmap hash table: slot_id → (offset, len) into keymap/blobs.vals
      blobs.vals               ← append-only: raw value bytes (1B bool, 8B LE i64, UTF-8 str)
    checkpoint                 ← 8-byte LE u64: WAL offset at last flush
```

The bitmap blobs in `blobs.vals` warrant a note, because they are **not** a copy
of the `ContainerStore` mmap. A stored bitmap is an independent, length-prefixed
encoding (`src/storage.rs`):

```
[4B  LE u32   container_count]
for each container (sorted by high key):
  [16B LE u128  high key]
  [4B  LE u32   blob_len]
  [blob_len bytes  rkyv-serialised Container]
```

`deserialize` rebuilds an anonymous `RoaringBitmap` from these bytes; the
`unaligned` rkyv feature lets the archived containers be read straight out of the
buffer without an aligned copy. The keymap store, for its part, needs no WAL of
its own — its mmap writes are OS-mediated and durable once flushed.

---

## 6. Append-only growth and crash-safe compaction

The append-only rule from §3.3 — values are only ever appended, never overwritten
in place — has a cost that this section and the next exist to manage. Every
`FieldIndex::insert` re-serialises the **whole** bitmap and `upsert`s it, and
`upsert` only ever appends. So a value shared by *N* documents
leaves *N−1* stale copies of its bitmap behind it in `blobs.vals` — `O(N²)`
cumulative bytes. For a low-cardinality field this is catastrophic: a boolean
spread across tens of thousands of documents can bloat to gigabytes. Dense row
IDs (§2) keep each individual bitmap small; compaction is what reclaims the
accumulated churn.

The keymap store has the same shape of problem in a milder form. It is written
once per distinct value, so a fixed value set never grows it — but distinct-value
churn leaves dead entries, exactly as described in §4. So compaction covers
**both** stores, each triggered on its own `waste_ratio()`. Because the two stores
key on the same slot IDs and free the same slots together (a value's slot is
released in both the moment its bitmap empties), compacting them independently
still leaves them mutually consistent.

`BlobStore::compact()` does the reclaiming: it rebuilds the value region from the
live slots only, dropping dead space and tombstones, and shrinks the file.
`waste_ratio()` reports the reclaimable fraction — alignment padding counts as
live, so it reads ≈0 immediately after a compaction. The trigger is a single
threshold, `ThresholdConfig::index_blob_waste_threshold` (a percentage, default 50,
set in TOML as `thresholds.index_blob_waste_threshold`).

Where that trigger fires is deliberately narrow. `compact()` is invoked from
exactly one place — `Database::run_index_checkpoint` — and only for a field over
the waste threshold. That checkpoint pass is reached three ways, all running the
same code: the periodic `IndexCheckpointWorker` (~15 min), a clean shutdown, and
the on-demand `Db::checkpoint_index()` exposed over REST as `POST
/admin/storage/index-checkpoint`. There is no write-path compaction and no
standalone one. (Do not confuse this with `Db::compact()` / `POST
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

Checkpointing is the routine that flushes the index to disk and, where needed,
invokes the compaction of §6. The `index_checkpoint_worker` runs it every **15
minutes** by default, and the identical sequence runs on clean shutdown and on
demand via `Database::run_index_checkpoint()`. Each tick processes the namespace's
row map first, then each active field in turn:

1. **Flush the `RowMap` first.** `flush_rowmap(wal_tail)` msyncs `rows.*` and
   atomically writes the `rowmap.ckpt` marker. The ordering here is load-bearing:
   the row map is made durable *before* any field index, so it is always at least
   as current as every persisted bitmap bit. Otherwise a crash could leave a
   persisted bit whose row ID the row map could no longer reproduce.
2. **Flush each field**, under that field's `index` **read** lock: msync the
   bitmap and keymap mmaps, then compute `bitmap_waste_ratio()` and
   `keymap_waste_ratio()`.
3. **Compact if either is over threshold.** When a store's waste is ≥
   `index_blob_waste_threshold`, take the field's **write** lock and
   `maybe_compact()` → `BlobStore::compact()` (§6) on whichever store(s) qualify.
   This stalls *that one field's* reads and writes for the compaction's I/O —
   dominated by the fsyncs — and only for fields actually over the threshold, not
   every field every tick.
4. **Record the offset.** Atomically write the current WAL tail offset to
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
  ├── Rebuild ordering from the keymap store
  ├── Read checkpoint file → last_flushed_wal_offset
  ├── Scan WAL last_flushed_wal_offset → tail; collect this namespace's keys
  ├── For each key: remove old index entries, re-insert the current KVStore value
  │       (re-allocating dense row IDs continues deterministically from next_id,
  │        because replay is in sequence order)
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
| Index bitmaps + keymap | mmap loaded (with `recover_compaction`), then WAL tail replayed from the last checkpoint offset |
| Compaction in flight | `compact.commit` marker → finish the staged swap; absent → discard `*.new` |
| Checkpoint markers | tmp-then-rename; a partial write is never observed |
| Sequence numbers | `max(WAL scan, metadata hint) + 1` |

All of this rests on WAL replay being idempotent and on there being no transaction
that spans multiple operations: each write's intent is durable in the WAL before
the operation is acknowledged, so replaying it twice is harmless and replaying it
once is enough.

### Why recovery lives inside `activate_field_index`

The replay step above deliberately runs *inside* field activation rather than in a
pass of its own. It did not always. Before commit `9399ff5` ("Simplified index
recovery process"), WAL replay ran in a separate `recover_indices()` step, which
opened a race: a concurrent put arriving between recovery and the index being
registered could have its effect silently overwritten by the replayed state.
Folding replay into `activate_field_index()` — before the index is wrapped in
`Arc<RwLock>` and made visible to anyone — closes that window completely.
