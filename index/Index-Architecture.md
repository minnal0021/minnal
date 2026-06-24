# Index Architecture

This document describes the `index` crate end to end: the custom RoaringBitmap
engine at its core, how per-field indexes are defined, built, stored, compacted,
and recovered after a crash, and how dense row IDs keep the bitmaps small.

> **Scope note.** The bitmap engine (`bitmap.rs`, `container/`, `container_store.rs`,
> `blob_store.rs`, `rowmap.rs`, `storage.rs`) lives in the `index` crate. The
> *lifecycle* glue — schema registration, activation, checkpointing, WAL replay —
> lives in `minnal_db` (`src/db/database.rs`, `kv_store.rs`, `index_manager.rs`),
> which owns the indexes per namespace. File/line references below point at the
> code as of this writing.

---

## 0. The big picture

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

Two things are worth internalising before the details:

1. **The RoaringBitmap is hand-rolled.** There is no `roaring` crate dependency
   (`index/Cargo.toml` pulls only `memmap2`, `parking_lot`, `rkyv`). The whole
   container model — array/bitset/run, promotion/demotion, all cross-type set
   operations, and the mmap-backed store underneath — is implemented in this
   crate (`bitmap.rs` + `container/`).

2. **Bitmaps are stored off-heap as serialised blobs, and materialised on
   demand.** A `FieldIndex` keeps only a small `BTreeMap` on the heap; each
   value's bitmap lives as a serialised byte blob in a memory-mapped `BlobStore`
   and is deserialised into an in-memory `RoaringBitmap` only when touched.

---

## 1. Field definition & creation

**Schema** (`minnal_db/src/db/namespace.rs`):

```rust
pub struct FieldDef {
    pub field_id: FieldId,           // unique u32, monotonically increasing, never reused
    pub field_name: String,
    pub field_type: IndexValueType,  // Bool | Int | Str
}
```

Registration happens via `Database::register_index_field`:

- Assigns a new monotonic `FieldId`, or returns the existing one if the same
  name+type was already registered (idempotent).
- Creates the on-disk directory `{db_path}/index/{namespace_id}/{field_id}/`.
- Persists the updated field list to `{db_path}/ns_{namespace}/config.json`.

`config.json` is the authoritative schema record: loaded on every
`Database::open()`, written atomically whenever a field is registered.

The three indexable value types (`IndexValueType` in `src/field/value.rs`) are
`Bool`, `Int` (i64), and `Str`, matching the `IndexValue` enum variants.

---

## 2. Dense row IDs — the `RowMap`

Field-index bitmaps are sets of **row IDs**, not keys. How those IDs are chosen
determines how the bitmaps pack — and is the single biggest lever on index size.

**Why dense.** A RoaringBitmap groups values by their high 112 bits into
*containers* (see §3). If row IDs are random `u128`s (the old Murmur3-hash
scheme), every document lands in its own high-key bucket, so the bitmap
degenerates to ~one container per document — pathologically sparse and huge.
Assigning **dense, monotonic** IDs (`0, 1, 2, …`) makes consecutive documents
share a high key and fill a single container, which is the whole reason the
bitmaps stay small.

**`RowMap`** (`src/rowmap.rs`) is a per-namespace, mmap-backed sidecar with three
parts:

| Part | Backing | Role |
|---|---|---|
| `key → id` | in-memory open-addressing table (anonymous mmap) | `O(1)` lookup on every put/delete/replay; **rebuilt from the id array on open** — never a persisted source of truth |
| `id → key` | `rows.idarray` (append-only, indexed by ID) → `rows.keybytes` (append-only key bytes) | `O(1)` resolution of query hits back to keys |
| counter | `next_id`, stored in the `rowmap.ckpt` marker | next dense ID to assign |

The hash table keys on the **full key bytes** (FNV-1a is used only to pick the
probe start, then keys are compared byte-for-byte), so there are no ID
collisions. Entries are **never removed** — a deleted-then-recreated key reuses
its original ID — so the table has no tombstones and `count == next_id` always.

**Row-ID precedence** (`KVStore::resolve_row_id_alloc`, `kv_store.rs`):

1. A caller-supplied `RowIdFn` (escape hatch for keys that embed their own ID,
   e.g. a UUID) wins, paired with a `RowToKeyFn` for reverse resolution.
2. Otherwise the dense `RowMap`.
3. Otherwise the legacy Murmur3 `key_to_row_id` — a defensive fallback only
   reachable for an indexed namespace with no row map loaded (should not occur).

On disk at `{db_path}/index/{namespace_id}/rowmap/` — a sibling of the per-field
directories (`rowmap` can never collide with a numeric `FieldId`).

---

## 3. The RoaringBitmap engine

### 3.1 `u128` keyspace and containers

A `RoaringBitmap` (`src/bitmap.rs`) holds a set of `u128` values. Each value is
split (`decompose`) into a **high key** (upper 112 bits) and a **low value**
(lower 16 bits):

```
value: u128  ──►  high (112 bits) ──► selects a container
                  low  (16 bits)  ──► the bit within that container
```

The high key selects a `Container`; the low 16-bit value is stored inside it.
A `Container` (`src/container/mod.rs`) is one of three encodings, chosen
adaptively for the data it holds:

| Container | Backing | Best for | Boundary |
|---|---|---|---|
| `ArrayContainer` | sorted `Vec<u16>` | sparse (few values) | promotes to `Bitset` at `ARRAY_TO_BITSET_THRESHOLD` (4096) elements |
| `BitsetContainer` | fixed `[u64; 1024]` (8 KB) | dense | demotes back to `Array` below the threshold |
| `RunContainer` | run-length `(start, len)` pairs | long consecutive runs | chosen by `optimize()` / set ops when `is_efficient()` (fewer bytes than the alternatives) |

`Container::insert` / `remove` auto-promote and auto-demote as cardinality
crosses the threshold; `Container::optimize()` re-evaluates all three encodings
(including switching to/from `Run`). `cardinality()` is `O(1)` — the total bit
count is cached in the store header and maintained on every mutation.

### 3.2 Set operations

`RoaringBitmap` provides `and` / `or` / `and_not` (plus in-place variants),
`flip`, range-scoped `range_and` / `range_or`, `rank` / `select`, and bulk
`from_sorted_iter` / `from_unsorted_iter`. The top level walks the two bitmaps'
high keys in sorted-merge order; per matching high key it dispatches to the
container layer.

`container/ops.rs` implements **all nine cross-type combinations** of the binary
operations (Array×Array, Array×Bitset, Bitset×Run, …), each returning the
best container type for its result (and demoting/promoting as needed). The hot
paths use SIMD helpers in `src/simd_support/` (popcount, bitwise AND/OR/AND-NOT,
sorted-array merge, bit extraction).

### 3.3 `ContainerStore` — the mmap backing

Each `RoaringBitmap` is backed by a `ContainerStore` (`src/container_store.rs`):
a two-file, memory-mapped `u128 → Container` map.

```
containers.keys   64-byte header (magic "MINNALBI", version, capacity, count,
                  tombstone count, cardinality, value_write_pos)
                  + 48-byte open-addressing linear-probing slots
                    (state EMPTY/OCCUPIED/TOMBSTONE, high-key, val offset, val len)
containers.vals   append-only rkyv-serialised Container blobs, 16-byte aligned
```

A store is either **anonymous** (an in-memory mmap, used for transient bitmaps —
set-operation results and the field-index bitmaps materialised on demand) or
**file-backed**. The public API is identical; growth is handled transparently by
remapping. `value_write_pos` only ever advances — values are **append-only**
within a store, never overwritten in place.

---

## 4. Field index in-memory structure

**`FieldIndex<V>`** (`src/field/field_index.rs`):

```
ordering:  BTreeMap<V, u128>   // value → slot_id  (sorted; drives range queries)
bitmaps:   BlobStore           // slot_id → serialised RoaringBitmap bytes
next_slot: u128
```

The heap carries only the `ordering` map (values + slot IDs, no bitmap data).
The bitmaps themselves live in a `BlobStore` (`src/blob_store.rs`) — the same
two-file mmap layout as `ContainerStore` (header "MINNALBS" + 48-byte slots +
append-only `blobs.vals`), but with arbitrary byte blobs as values.

**Insert** `(value, row_id)`:

1. Look up or allocate `slot_id` for `value` in `ordering`.
2. `load_bitmap(slot_id)` → read the blob, `storage::deserialize` it into an
   in-memory (anonymous) `RoaringBitmap`.
3. `bitmap.insert(row_id)`.
4. `store_bitmap` → `storage::serialize` the whole bitmap, `BlobStore::upsert`
   (which **appends** the new blob).

**Remove** is symmetric; when a bitmap becomes empty its slot is freed
(`remove_key`) and the `ordering` entry dropped. `remove_all_for_row` clears a
row from every value bucket — used to handle document **updates** (clear old
buckets, then re-insert the new value).

Predicate evaluation (`evaluate`) uses `ordering` to find the relevant slots —
a single `get` for `Eq`/`In`, a `BTreeMap::range` for `Lt`/`Le`/`Gt`/`Ge`/
`Between`, a full scan minus one for `Ne` — then OR-folds their bitmaps.

### Keymap — reconstructing `ordering` on open

The `ordering` map is heap-only, so it must be rebuilt at startup. A second
`BlobStore` under `keymap/` persists `slot_id → raw value bytes` (1 byte for
bool, 8-byte LE i64, UTF-8 for strings). It is written **once per distinct
value** (not per document), so a fixed value set never bloats it — but under
**distinct-value churn** (values that appear and are later fully removed,
freeing their slot) it accumulates dead entries and is compacted on its own
waste ratio (§6). On open, scanning it reconstructs `BTreeMap<V, slot_id>`
exactly.

---

## 5. Storage format on disk

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

### Serialised RoaringBitmap format (`src/storage.rs`)

A bitmap blob in `blobs.vals` is **not** a copy of the `ContainerStore` mmap; it
is an independent length-prefixed encoding:

```
[4B  LE u32   container_count]
for each container (sorted by high key):
  [16B LE u128  high key]
  [4B  LE u32   blob_len]
  [blob_len bytes  rkyv-serialised Container]
```

`deserialize` rebuilds an anonymous `RoaringBitmap` from these bytes (the
`unaligned` rkyv feature lets archived containers be read without an aligned
copy).

The keymap store is **immediately durable** — mmap writes are OS-mediated and do
not require separate WAL entries.

---

## 6. Append-only growth & crash-safe compaction

Every `FieldIndex::insert` re-serialises the **whole** bitmap and `upsert`s it,
and `upsert` always **appends** (the value region only grows). So a field value
shared by *N* documents leaves *N−1* stale bitmap copies behind — `O(N²)`
cumulative bytes, and **catastrophic for low-cardinality fields** (a boolean over
tens of thousands of docs can bloat to gigabytes). The dense row IDs (§2) shrink
each individual bitmap; compaction reclaims the append churn.

The same applies, more mildly, to the **keymap** store: a distinct value is
written once, so a fixed value set never bloats — but under distinct-value churn
(values created then fully removed) its freed slots leave dead bytes. `maybe_compact`
therefore compacts **both** stores, each on its own `waste_ratio()`. They key on
the same slot IDs and free the same slots together (when a value's bitmap empties),
so compacting them independently keeps them consistent.

**`BlobStore::compact()`** rebuilds the value region from live slots only,
drops dead space and tombstones, and shrinks the file. `waste_ratio()` reports
the reclaimable fraction (alignment padding counts as live, so it reads ≈0 right
after a compaction). Compaction is driven at the checkpoint (§7) for any field
whose waste crosses `ThresholdConfig::index_blob_waste_threshold` (percent,
default 50; TOML `thresholds.index_blob_waste_threshold`).

> **The checkpoint pass is the *only* trigger for compaction.** `compact()` is
> invoked from exactly one place — `Database::run_index_checkpoint` — and only
> for a field over the waste threshold. That pass is reached three ways, all
> running the *same* code: the periodic `IndexCheckpointWorker` (~15 min), clean
> shutdown, and the on-demand `Db::checkpoint_index()` (exposed over REST as
> `POST /admin/storage/index-checkpoint`). There is no write-path or standalone
> compaction. (Note: `Db::compact()` / `POST /admin/storage/compact` is the
> unrelated KV-engine LSM/value-log compaction, not this.)

### Compaction must never rewrite in place

Index recovery (§8) loads the on-disk store and replays only the **WAL tail** on
top of it — it does *not* rebuild from scratch. So a crash that left new key
offsets pointing into a half-rewritten value region would be inherited as
**silent corruption** that replay cannot heal (any value bucket not touched in
the replay window stays wrong).

`compact()` therefore uses a **staged file swap**:

```
build compacted (key table, val region) in memory   (~1× live size, read from mmap)
  write blobs.keys.new + blobs.vals.new   → fsync both files + dir   (live pair untouched)
  write compact.commit marker             → fsync dir                ◄── COMMIT POINT
  rename *.new → live; fsync dir; remove marker; fsync dir
  remap onto the new files
```

`BlobStore::open()` calls `recover_compaction()` first: **marker present** ⇒ the
staged files are complete, finish the swap idempotently; **marker absent** ⇒ the
staged files are partial, discard them. Two-file renames are not atomic as a
pair — the marker is what makes the swap atomic, not any lock.

> **Invariant:** after any crash the on-disk pair is fully **old** or fully
> **new**, never a torn mix. If you touch this path, preserve the ordering
> (stage+fsync → marker+fsync → rename → drop marker) and keep
> `recover_compaction` in `open`.

---

## 7. Checkpointing

A background worker (`index_checkpoint_worker`) runs every **15 minutes** by
default; the same sequence runs on clean shutdown and on demand via
`Database::run_index_checkpoint()`. Per tick, for the namespace then each active
field:

1. **Flush the `RowMap` first.** `flush_rowmap(wal_tail)` msyncs `rows.*` and
   atomically writes the `rowmap.ckpt` marker. **Load-bearing ordering:** the row
   map is flushed *before* any field index, so it is always at least as durable
   as every persisted bitmap bit — otherwise a crash could leave a bit whose row
   ID can't be reproduced.
2. Per field, under the field's `index` **read** lock: `msync` the bitmap and
   keymap mmaps; compute `bitmap_waste_ratio()` and `keymap_waste_ratio()`.
3. If **either** is ≥ `index_blob_waste_threshold`, take the field's **write**
   lock and `maybe_compact()` → `BlobStore::compact()` (§6) on whichever store(s)
   are over. This stalls *that field's* reads and writes for the compaction's I/O
   (dominated by the fsyncs), and only for fields over the threshold — not every
   tick.
4. Atomically write the current WAL tail offset to `{field_path}/checkpoint` via
   **tmp-then-rename**.

---

## 8. Crash recovery

Recovery is a **checkpoint + WAL replay** model. The index is a *derived*
structure: its durable on-disk state is brought current by replaying the WAL
entries written since its last checkpoint.

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

### Guarantees

| Component | Recovery mechanism |
|---|---|
| Field schema | `config.json` written synchronously on registration |
| KVStore data | WAL replay in `recover_from_wal()` before any index activation |
| Row map | `rows.*` + marker loaded; key→id table rebuilt from the id array; torn tail ignored |
| Index bitmaps + keymap | mmap loaded (with `recover_compaction`), then WAL tail replayed from the last checkpoint offset |
| Compaction in flight | `compact.commit` marker → finish the staged swap; absent → discard `*.new` |
| Checkpoint markers | tmp-then-rename; a partial write is never observed |
| Sequence numbers | `max(WAL scan, metadata hint) + 1` |

WAL replay is idempotent, and there is no transaction spanning multiple
operations — each write's intent is durable in the WAL before the op is
acknowledged.

### Why index recovery lives in `activate_field_index`

Prior to commit `9399ff5` ("Simplified index recovery process"), index WAL replay
ran in a separate `recover_indices()` pass. That introduced a race: a concurrent
put arriving between recovery and index registration could have its effect
overwritten by the replayed state. Moving replay into `activate_field_index()` —
before the index is wrapped in `Arc<RwLock>` and made visible — closes that
window entirely.
