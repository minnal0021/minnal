# index — RoaringBitmap Field Indexing

Provides per-field bitmap indexes over document key-spaces and a predicate query evaluator. `minnal_db` stores a `DynFieldIndex` per indexed field (held in its own per-namespace `NamespaceIndex`), and `minnal_doc_store` uses the query DSL to answer structured queries.

## Key files

| File | Role |
|---|---|
| `src/lib.rs` | Public re-exports |
| `src/field/field_index.rs` | `FieldIndex<V>` — per-field bitmap index (`BTreeMap<V, slot>` + blob-backed bitmaps + an in-memory **overlay** of bitmaps changed since the last spill) |
| `src/field/value.rs` | `DynFieldIndex` (type-erased field index), `IndexValue`, `IndexValueType` |
| `src/field/predicate.rs` | `Predicate` — a single field predicate (eq, ne, lt, le, gt, ge, between, in) |
| `src/query/lexer.rs` | Tokeniser for the query string DSL |
| `src/query/parser.rs` | Parser: query string → `RawExpr` AST (`Op`, `RawValue`) |
| `src/query/eval.rs` | Evaluator: `RawExpr` against live `DynFieldIndex`es (via a `SchemaMap` + `get_index` closure) → `RoaringBitmap` of row IDs |
| `src/bitmap.rs` | `RoaringBitmap` — **custom** from-scratch Roaring implementation over a `u128` keyspace (no `roaring` crate dependency); backed by a `ContainerStore` (heap for transient bitmaps, mmap files for persistent ones) |
| `src/container/` | The three container types (`Array`, `Bitset`, `Run`) + the `Container` enum dispatch and all cross-type set ops (`mod.rs`, `array.rs`, `bitset.rs`, `run.rs`, `ops.rs`) |
| `src/container_store.rs` | `ContainerStore` — `u128 → Container` map behind one bitmap. **Heap** backing (`BTreeMap`, changed in place) for every transient bitmap (`RoaringBitmap::new`, loaded copies, query results); **Mapped** two-file store (`containers.keys` + `containers.vals`) for `create`/`open`. Never put transient bitmaps on the mapped backing: it appends a container copy per change (65,536 inserts → 514 MiB for an 8 KB bitmap) |
| `src/blob_store.rs` | `BlobStore` — mmap-backed `u128 → bytes` two-file store (same layout as `ContainerStore`); holds each `FieldIndex`'s bitmaps off-heap in the **directory** layout (`BlobLayout::Directory`: slot → directory blob → container blobs; the keymap uses `Flat`). Space accounting, open-time repair and compaction read directories. On-disk `VERSION` 2. Crash-ordered writes (`append_value` → `sync_values` → `set_slot` → `sync_keys`), checksummed slots, repair at open. Also defines the `pub(crate) GrowableMmap` helper that `RowMap` reuses (`ContainerStore` keeps its own private one) |
| `src/overlay_budget.rs` | `IndexOverlayBudget` — one atomic byte counter per `Database`, shared by every field's overlay, with a soft limit (request a checkpoint) and a hard limit (the writer spills its own field) |
| `src/rowmap.rs` | `RowMap` — per-namespace dense row-ID map (`key↔id` + counter), mmap sidecar |
| `src/storage.rs` | On-disk bitmap encoding: one rkyv blob per `Container` (`encode_container` / `decode_container`) and the checksummed per-bitmap **directory** (`MBD1`: sorted container key → offset, len; `encode_dir` / `decode_dir`) |
| `src/simd_support/` | SIMD helpers for container set ops (popcount, bitwise, array merge, extract, sum, run-bitset). Each kernel has AVX-512/AVX2 (x86_64, runtime-detected) **and NEON (aarch64/Apple Silicon, baseline, compile-time)** paths plus a scalar fallback; all are unit-tested against the scalar reference. When adding a kernel, provide all three or guard the missing arch with `cfg`. |

## How it works

1. Each indexed field gets a `DynFieldIndex` (minnal_db holds them in its per-namespace `NamespaceIndex`).
2. On every document write, the extractor closure (`ExtractorFn`) maps raw bytes → `IndexValue`, and the value is inserted into the field's bitmap at the document's row ID.
3. At query time, `eval.rs` walks the parsed `RawExpr`, evaluates each field predicate against that field's `DynFieldIndex`, combines the per-field bitmaps (AND / OR / NOT), and returns a `RoaringBitmap` of matching row IDs.
4. Callers resolve row IDs back to keys via `RowToKeyFn` (O(|hits|)) or an in-memory map.

### End-to-end: ingestion → overlay → spill → compaction

```
INGEST (synchronous, on the write path)                    [minnal_doc_store → minnal_db]
  Db::put(ns, key, value)
    └─ KVStore::put_to_storage_inner                       (kv_store.rs)
         1. WAL append + fsync          ← durability barrier (the DOC is durable here)
         2. value-log write → pointer; memtable insert
         3. update_indices_on_put(key, value)              ← indexing runs INLINE, in memory
              row_id = resolve_row_id_alloc(key)           (dense RowMap id; or registered RowIdFn)
              for each registered field, under entry.index.write():
                v = extractor(value)                       (ExtractorFn: bytes → IndexValue)
                DynFieldIndex::insert / update / set:      (clear old bucket(s) if any, insert new)
                  slot_id = ordering[v]                    (BTreeMap<V,u128>, in heap)
                  overlay[slot_id][row_id >> 16] ← container (loaded from the store on first
                                                   touch via the slot's directory), changed in place
                  budget.charge(Δ heap bytes)
                  if budget.over_hard(): spill() THIS field  ← stage, msync vals, commit, msync keys
              after the lock: over_soft() → IndexCheckpointTrigger::request()
                              dead_bytes ≥ index_blob_backpressure_bytes → request_if_over_cap()
         (nothing touches the index files here unless the hard limit is crossed;
          after a crash the overlay is rebuilt by WAL replay from the checkpoint offset)

CHECKPOINT (background, every 1.75 s by default / soft limit / backpressure / shutdown / Db::checkpoint_index)
  IndexCheckpointWorker → Database::run_index_checkpoint   (database.rs)
    wal_tail = wal_cut_ceiling()           (below any in-flight write)
    flush every row map first (its marker before any field's)
    per active field — spill, crash-ordered, no fsync under the write lock:
      1. stage()        WRITE lock: append every changed container, a new directory per
                                    changed bitmap, and pending keymap entries to the
                                    value regions; no slot points at them yet
      2. sync_values()  READ lock:  msync both stores' value regions
                                    (failure → abort(stage): overlay kept, nothing pointed)
      3. commit(stage)  WRITE lock: point slots at the staged blobs; remove emptied slots;
                                    drop overlay entries unchanged since stage (version check)
      4. sync_keys()    READ lock:  msync both key tables
      if bitmap or keymap waste ≥ index_blob_waste_threshold (default 50%):
        WRITE lock: maybe_compact() → BlobStore::compact(); flush()
    checkpoint_fields(wal_tail)            ← records WAL offset now reflected on disk (tmp+rename)

COMPACTION (staged, crash-safe swap)                       [index/blob_store.rs]
  build compacted val_buf + key table in memory (~1× live size, read from mmap)
    write blobs.vals.new + blobs.keys.new  → fsync files + dir   (live pair untouched)
    write compact.commit marker            → fsync dir           ◄── COMMIT POINT
    rename *.new → live; fsync dir; remove marker; fsync dir
    remap self.val / self.key onto new files

RECOVERY (on open)                                         [index + minnal_db]
  BlobStore::open → recover_compaction():  marker present ⇒ finish swap; absent ⇒ drop *.new
                  → repair_at_open():      tombstone torn (bad CRC) / out-of-range slots, and
                                           slots whose directory is torn or lists a container
                                           past the file; recompute counts + write cursor
  DynFieldIndex::open → reconcile_at_open(): drop values with no bitmap, remove bitmaps no
                                           value maps to, new slot ids above both stores
  activate_field_index: damaged_at_open > 0 ⇒ gap (GapCause::DamagedIndexFile, FullRebuild);
                        replay WAL tail from checkpoint offset (bounded by the shared budget)
```

## Supported value types (`IndexValueType`)

`Bool`, `Int` (i64), `Str` — matching the `IndexValue` enum variants in `src/field/value.rs`.

## Query DSL (parsed by `query/`)

The string DSL is used by `minnal_doc_store` to accept structured query strings from the REST API. Example: `age > 30 AND status = "active"`. The lexer/parser produce a `RawExpr`; the evaluator runs it against the live field indexes (resolved by name through a `SchemaMap` + `get_index` closure).

**Operator precedence is the conventional boolean ordering: `NOT` > `AND` > `OR`** (`NOT` binds tightest, `OR` loosest). So `a = 1 OR b = 2 AND c = 3` means `a = 1 OR (b = 2 AND c = 3)`, and `NOT a = 1 AND b = 2` means `(NOT a = 1) AND b = 2`. `AND` and `OR` are each left-associative; use parentheses to override grouping. (The evaluator validates the **whole** AST before evaluating, and AND short-circuits only after validation — see `query/eval.rs`.)

**`NOT` uses document-store semantics, not SQL.** `NOT` is complemented against a universe scoped to the fields the inner expression references, so `NOT status = "active"` returns rows that **have a `status` value** other than `"active"` — a row with **no `status` field is excluded** (a missing field is "no value", not "a differing value"). There is no `EXISTS`/`MISSING` operator yet; add one if you need to match rows by field presence. See the `parse_and_evaluate` rustdoc in `query/eval.rs`.

### Query complexity is capped — the whole pipeline is recursive

The parser (`parse_term`), the evaluator (`validate` / `eval_expr` / `collect_field_ids`) and `RawExpr`'s compiler-generated **drop glue** all recurse over the expression tree, and a stack overflow *aborts the process* (it does not unwind, so `panic = "abort"` and handlers returning `Result` are both irrelevant to it). Since `POST /stores/{ns}/query` passes the caller's predicate string straight through, an unbounded tree is a remote kill switch for every namespace at once. Two caps in `query/parser.rs` bound it, and **both are needed**:

| Const | Value | Bounds |
|---|---|---|
| `MAX_PARSE_DEPTH` | 64 | nesting levels entered via `(` and `NOT` — i.e. *parser* stack depth |
| `MAX_PARSE_NODES` | 512 | total `RawExpr` nodes — i.e. *tree* depth, hence every other walk |

Exceeding either yields `QueryError::TooComplex`.

**The node budget is not redundant with the depth limit.** A flat `a = 1 AND a = 1 AND …` chain needs no parser recursion at all (`parse_and` loops) but builds a left-nested tree one level deep per operand. Verified before the fix: 5,000 `(` aborted inside the parser, while 50,000 `AND` terms parsed **successfully** and then overflowed the stack in the **drop alone**, before evaluation ever ran. A depth counter by itself would have closed only the first.

`MAX_PARSE_NODES` is pinned from both sides — `parser::tests::long_flat_and_chain_is_rejected_by_the_node_budget` rejects oversize input, and `eval::tests::a_maximally_complex_accepted_query_evaluates_within_a_small_stack` evaluates a max-size chain on a deliberately small 2 MiB stack (Rust's default thread stack, and tokio's default worker stack). Measured on a debug build: 1024 nodes fits, 2048 overflows, so 512 keeps ~4× headroom. **Raise the cap and that eval test fails** — which is the point. Note an `IN` list is a single node regardless of length, so the natural "match many values" query shape is unaffected by the budget.

## Persistence

Field indexes live under `{db_path}/index/{ns_id}/{field_id}/` (layout in `db/index_manager.rs`). `IndexCheckpointWorker` in `minnal_db` spills every field's overlay and records the WAL offset the files reflect; on open, `minnal_db` replays the WAL from that offset to bring the index back in sync.

### Writes are buffered in an overlay of changed containers — compaction reclaims dead space

`FieldIndex` stores each distinct value's `RoaringBitmap` **container by container** in a `BlobStore` (`blobs.keys` open-addressing table + `blobs.vals` append-only value region): every container is its own blob, and the value's slot points at a **directory** blob listing them (`storage.rs`). **A write never touches the blob store.** It changes an in-memory copy of the one container the row falls in, held in the field's **overlay** (`slot → container key → Option<Container>`, `None` = container emptied), which reads consult first. The overlay entry also caches the slot's decoded directory (`SlotChanges::base`) so repeated writes to a big bitmap do not re-decode it; the cache is dropped when the stored directory changes under it (a commit that keeps the entry, or `compact_bitmaps`), because compaction moves every offset. The overlay is written out by a **spill** — at each index checkpoint, or by the writer itself when the shared `IndexOverlayBudget` passes its hard limit — which appends each changed container plus a new directory per changed bitmap and repoints the slot. Unchanged containers are shared between the old and new directory; the old directory and old copies of changed containers become dead space.

Why: before the overlay, every insert re-serialised and appended the whole bitmap, which is O(N²) cumulative bytes per value — a boolean over 200k rows appended 3 GB for 54 KB live, and a 5-value field replayed over 16k docs left a 2.6 GB file that could not be opened. The overlay (step 2) made that one bitmap per changed value per spill (bool 200k rows 3,076 MiB → 3.6 MiB). It still held **whole** bitmaps, so once the changed bitmaps exceeded the hard limit (a boolean over ~300M rows) every write spilled its whole bitmap again; the container-granular layout (FR-005, steps 3b–3d) removed that: past the hard limit each write now appends about one container and a directory (`past_the_hard_limit_each_spill_appends_containers_not_bitmaps`). See `Field-Index-Write-Back-Evaluation.md` → *Results*. **Do not reintroduce a per-write blob write** (`BlobStore::upsert` is `#[cfg(test)]` for this reason and because it is not crash-ordered), and **do not go back to whole-bitmap blobs**.

The budget (`thresholds.index_overlay_soft_bytes` 32 MiB / `index_overlay_hard_bytes` 64 MiB) is **one counter per `Database`**, shared by every field (`Database::index_overlay_budget`, set on each `DynFieldIndex` at activation). It charges each overlay container's heap bytes plus a fixed overhead, each slot's overhead, and its cached directory. Soft → `IndexCheckpointTrigger::request()` (uncapped; fires even with backpressure disabled). Hard → `spill_if_over_budget` on the writer's thread, under the field's write lock, including the two msyncs — so a write that crosses the hard limit is slow. A soft limit above the hard one is clamped; a hard limit of 0 spills every write. `IndexOverlayBudget::stats()` (peak, soft crossings, hard spills and their time) is served under `index_overlay` in `GET /admin/storage/ops-metrics`.

**Format version.** `BlobStore` `VERSION` is 2 (directory layout). A version-1 store fails `validate_open` with `InvalidData` and the field cannot activate — greenfield: delete the database (or the field's directory) and rebuild. There is no migration.

`BlobStore::compact()` rebuilds the value region from live slots only, clears tombstones, and shrinks the file. `BlobStore::waste_ratio()` reports the reclaimable fraction (alignment padding counts as live). The checkpoint calls `DynFieldIndex::maybe_compact` per field when waste crosses `ThresholdConfig::index_blob_waste_threshold` (percent, default 50). `Db::checkpoint_index()` forces a flush+compaction on demand.

**Backpressure is now a secondary valve.** The write path also reads `BlobStore::dead_bytes()` — an **O(1)** running count of reclaimable bytes (incremented by `set_slot`-replace / `remove_key`, reset by `compact`, seeded from `logical - live` at `open`), surfaced as `DynFieldIndex::reclaimable_dead_bytes()` — and fires `request_if_over_cap` at `thresholds.index_blob_backpressure_bytes` (default 64 MiB, 0 disables). Dead space now grows per spill, not per write, so this fires only when hard-limit spills pile up between checkpoints. Keep it an absolute **byte** cap, not a ratio: a field with one large, often-rewritten bitmap sits near 100% waste after a few spills, so a ratio trigger would fire nearly every write. **WAL replay at open** compacts inline on the same cap (and on the default cap when it is 0), because the checkpoint worker is not running yet and an unbounded replay can leave the database unopenable. See `db/index_checkpoint_worker.rs`.

`maybe_compact` covers **both** of a field's BlobStores. The `keymap/` store (slot_id → value) is written once per distinct value (queued in `keymap_pending`, written by the next spill), so a fixed value set never bloats it — but under **distinct-value churn** (values fully removed, freeing their slot) it accumulates dead entries, so it is compacted on its own `keymap_waste_ratio()`.

### Spill write order and crash safety — preserve it

Both files of a `BlobStore` are `MAP_SHARED` mmaps the kernel writes back **at any time, in any order**, and slots are rewritten in place. So a spill is two-phase, and every step is load-bearing:

1. `stage` (write lock) — `append_value` every changed container, a new directory for each changed bitmap, and every pending keymap entry. No slot points at them, so a crash here leaves only unreachable bytes. Reads keep using the overlay.
2. `sync_values` (read lock) — msync both stores' value regions. On failure, `abort(stage)`: the overlay still holds everything and the pending keymap work is re-queued.
3. `commit` (write lock) — `set_slot` each staged blob; remove emptied slots. An overlay entry is dropped only if its **version** is unchanged since the stage (a write between stage and commit keeps its entry for the next spill — this is what makes releasing the lock between phases safe).
4. `sync_keys` (read lock) — msync the key tables, bitmaps first, then keymap.

No fsync runs under the write lock on the checkpoint path. `DynFieldIndex::spill` runs the same four steps in one call (hard-limit spill, `flush`), and returns early while the checkpoint has a stage in flight (`stage_in_flight`).

Further invariants:

- **Slots are checksummed.** A live slot is written with state byte **3** and a CRC-32 of its first 36 bytes; state 1 (no checksum) is still read but never written. `BlobStore::open` → `repair_at_open` tombstones any slot that is torn (CRC mismatch) or points past the value file, counts it in `damaged_at_open`, and **recomputes the header** (counts, and the write cursor as `max(live blob end)`) so a stale header can never make a new append overwrite live data.
- **A freed slot's keymap entry is removed one spill later** (`keymap_free_next`), after the bitmap's removal is durable, so a crash can never free a slot id whose old bitmap is still on disk.
- **The two stores are reconciled at open, not kept atomic.** There is no marker across the bitmap and keymap stores. `DynFieldIndex::open` → `reconcile_at_open` drops values whose bitmap is missing, removes bitmaps no value maps to, and raises `next_slot` above every id either store holds. Content lost this way (and anything after the last checkpoint) comes back from **WAL replay**; a non-zero `damaged_at_open` makes `activate_field_index` record a `GapCause::DamagedIndexFile` full-rebuild gap, since dropped slots lost rows that replay from the checkpoint offset may not cover.
- **Rehash never rewrites the live key table in place** (persistent stores): sync values, build the doubled table, write `blobs.keys.rehash`, rename over `blobs.keys`, fsync the dir, remap.

Regression tests live beside the code in `blob_store.rs`, `field/value.rs` (`open_removes_an_orphan_bitmap_and_never_reuses_its_slot`, …) and `db/database_tests.rs` (damaged file → gap).

#### Compaction is crash-safe via a staged file swap — do NOT rewrite in place

Compaction must **never** mutate the live `blobs.keys` / `blobs.vals` in place. Index recovery (`activate_field_index`) loads the on-disk store and replays only the WAL tail *on top* of it — it does **not** rebuild from scratch — so a crash that leaves new key offsets pointing into a half-rewritten value region is inherited as **silent corruption** that replay cannot heal (any value bucket not touched by a key in the replay window stays wrong).

So `compact()` (persistent stores) builds the compacted pair in memory, stages it as `blobs.keys.new` / `blobs.vals.new` (live pair untouched), fsyncs both + the dir, then writes a `compact.commit` marker (fsynced) as the **commit point**, renames both files into place, fsyncs, and removes the marker. `BlobStore::open()` calls `recover_compaction()` first: marker present ⇒ the staged files are complete, finish the swap idempotently; marker absent ⇒ staged files are partial, discard them. **Invariant: after any crash the on-disk pair is fully old or fully new, never a torn mix.** Two-file renames are not atomic as a pair — the marker is what makes the swap atomic, not the write lock. If you touch this path, preserve the ordering (stage+fsync → marker+fsync → rename → drop marker) and keep `recover_compaction` in `open`.

Cost notes: peak heap is ~1× the live (compacted) index — the compacted `val_buf` is built straight from the mmap and written without a pad-copy; don't reintroduce an `iter_entries()` snapshot or a `to_vec()` pad. Compaction runs under the field's `entry.index` **write lock** (`run_index_checkpoint`), so it stalls queries *and* writes **on that field only** for the compaction's I/O duration (dominated by the fsyncs, not the remap). It fires only for fields over the waste threshold at a checkpoint tick, not every tick. If that stall ever matters, the planned fix is two-phase locking (stage+fsync unlocked, swap+remap under the write lock, abort-and-retry on concurrent write) — not reverting to in-place.
