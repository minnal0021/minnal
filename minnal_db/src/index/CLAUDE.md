# index — RoaringBitmap Field Indexing

Provides per-field bitmap indexes over document key-spaces and a predicate query evaluator. `minnal_db` stores a `DynFieldIndex` per indexed field (held in its own per-namespace `NamespaceIndex`), and `minnal_doc_store` uses the query DSL to answer structured queries.

## Key files

| File | Role |
|---|---|
| `src/lib.rs` | Public re-exports |
| `src/field/field_index.rs` | `FieldIndex<V>` — per-field bitmap index (`BTreeMap<V, slot>` + blob-backed bitmaps) |
| `src/field/value.rs` | `DynFieldIndex` (type-erased field index), `IndexValue`, `IndexValueType` |
| `src/field/predicate.rs` | `Predicate` — a single field predicate (eq, ne, lt, le, gt, ge, between, in) |
| `src/query/lexer.rs` | Tokeniser for the query string DSL |
| `src/query/parser.rs` | Parser: query string → `RawExpr` AST (`Op`, `RawValue`) |
| `src/query/eval.rs` | Evaluator: `RawExpr` against live `DynFieldIndex`es (via a `SchemaMap` + `get_index` closure) → `RoaringBitmap` of row IDs |
| `src/bitmap.rs` | `RoaringBitmap` — **custom** from-scratch Roaring implementation over a `u128` keyspace (no `roaring` crate dependency); backed by a `ContainerStore` |
| `src/container/` | The three container types (`Array`, `Bitset`, `Run`) + the `Container` enum dispatch and all cross-type set ops (`mod.rs`, `array.rs`, `bitset.rs`, `run.rs`, `ops.rs`) |
| `src/container_store.rs` | `ContainerStore` — mmap-backed `u128 → Container` two-file store (`containers.keys` + `containers.vals`) backing a single bitmap; anonymous or file-backed |
| `src/blob_store.rs` | `BlobStore` — mmap-backed `u128 → bytes` two-file store (same layout as `ContainerStore`); holds each `FieldIndex`'s serialised bitmaps off-heap. Also defines the `pub(crate) GrowableMmap` helper that `RowMap` reuses (`ContainerStore` keeps its own private one) |
| `src/rowmap.rs` | `RowMap` — per-namespace dense row-ID map (`key↔id` + counter), mmap sidecar |
| `src/storage.rs` | `serialize` / `deserialize` — `RoaringBitmap` ⇄ bytes (length-prefixed, rkyv per `Container`) for blob storage |
| `src/simd_support/` | SIMD helpers for container set ops (popcount, bitwise, array merge, extract, sum, run-bitset). Each kernel has AVX-512/AVX2 (x86_64, runtime-detected) **and NEON (aarch64/Apple Silicon, baseline, compile-time)** paths plus a scalar fallback; all are unit-tested against the scalar reference. When adding a kernel, provide all three or guard the missing arch with `cfg`. |

## How it works

1. Each indexed field gets a `DynFieldIndex` (minnal_db holds them in its per-namespace `NamespaceIndex`).
2. On every document write, the extractor closure (`ExtractorFn`) maps raw bytes → `IndexValue`, and the value is inserted into the field's bitmap at the document's row ID.
3. At query time, `eval.rs` walks the parsed `RawExpr`, evaluates each field predicate against that field's `DynFieldIndex`, combines the per-field bitmaps (AND / OR / NOT), and returns a `RoaringBitmap` of matching row IDs.
4. Callers resolve row IDs back to keys via `RowToKeyFn` (O(|hits|)) or an in-memory map.

### End-to-end: ingestion → index → checkpoint → compaction

```
INGEST (synchronous, on the write path)                    [minnal_doc_store → minnal_db]
  Db::put(ns, key, value)
    └─ KVStore::put_to_storage_inner                       (kv_store.rs)
         1. WAL append + fsync          ← durability barrier (the DOC is durable here)
         2. value-log write → pointer; memtable insert
         3. update_indices_on_put(key, value)              ← indexing runs INLINE, in-memory
              row_id = resolve_row_id_alloc(key)           (dense RowMap id; or registered RowIdFn)
              for each registered field, under entry.index.write():
                v = extractor(value)                       (ExtractorFn: bytes → IndexValue)
                FieldIndex::remove_all_for_row(row_id)     (clear old buckets — handles updates)
                FieldIndex::insert(v, row_id):
                  slot_id = ordering[v]                    (BTreeMap<V,u128>, in heap)
                  bm = load_bitmap(slot_id); bm.insert(row_id)
                  BlobStore::upsert(slot_id, serialize(bm))  ← APPEND-ONLY: old copy orphaned
         (index mutation is mmap-only here — NOT yet fsynced; rebuilt from WAL on recovery)

CHECKPOINT (background, every ~15 min / BACKPRESSURE / shutdown / Db::checkpoint_index)   [minnal_db]
  IndexCheckpointWorker → Database::run_index_checkpoint   (database.rs)
    per active field:
      ── under entry.index.READ lock ──   flush() mmap;  waste = bitmap_waste_ratio()
      if waste ≥ index_blob_waste_threshold (default 50%):
        ── under entry.index.WRITE lock ──  maybe_compact() → BlobStore::compact();  flush()
                                            (stalls THIS field's reads+writes for the I/O)
    checkpoint_fields(wal_tail)            ← records WAL offset now reflected on disk (tmp+rename)

  BACKPRESSURE (write-path valve — bounds transient disk between the ~15 min ticks):
    the ingest path checks the field's O(1) DynFieldIndex::reclaimable_dead_bytes()
    after each update; when it reaches thresholds.index_blob_backpressure_bytes
    (default 64 MiB) it fires IndexCheckpointTrigger → an immediate checkpoint
    (debounced via a shared `pending` flag; cap 0 disables). This is an absolute
    BYTE cap, not a ratio: a low-cardinality field crosses any ratio in a handful
    of writes and stays ~100%, so a ratio trigger would fire almost every write.
    The cap self-debounces because compaction resets dead_bytes to 0.

COMPACTION (staged, crash-safe swap)                       [index/blob_store.rs]
  build compacted val_buf + key table in memory (~1× live size, read from mmap)
    write blobs.vals.new + blobs.keys.new  → fsync files + dir   (live pair untouched)
    write compact.commit marker            → fsync dir           ◄── COMMIT POINT
    rename *.new → live; fsync dir; remove marker; fsync dir
    remap self.val / self.key onto new files
  ⇒ invariant: on-disk pair is fully OLD or fully NEW after any crash, never torn

RECOVERY (on open)                                         [index + minnal_db]
  BlobStore::open → recover_compaction():  marker present ⇒ finish swap; absent ⇒ drop *.new
  activate_field_index:  load consistent store, then replay WAL tail from checkpoint offset
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

Index snapshots are written to the database directory alongside the LSM/value-log data. `IndexCheckpointWorker` in `minnal_db` drives periodic snapshots; on open, `minnal_db` replays any un-checkpointed WAL entries to bring the index back in sync.

### Blob store is append-only — compaction reclaims dead space

`FieldIndex` holds each distinct value's `RoaringBitmap` as a single blob in `BlobStore` (`blobs.keys` open-addressing table + `blobs.vals` append-only value region). **`insert` re-serialises the *whole* bitmap and `upsert`s it on every document** — and `upsert` always appends the new blob, orphaning the previous copy (`value_write_pos` only advances; nothing is reclaimed in place). So a field value shared by N documents leaves N−1 stale bitmap copies. This is O(N²) cumulative bytes per value and is **catastrophic for low-cardinality fields** (few distinct values, each covering many docs) — e.g. a boolean over 40k docs can bloat to multiple GB. The random u128 row IDs compound it: bitmaps are pathologically sparse (≈one container per doc), so each rewrite is large.

`BlobStore::compact()` rebuilds the value region from live slots only (dropping dead space), clears tombstones, and shrinks the file. `BlobStore::waste_ratio()` reports the reclaimable fraction (alignment padding counts as live, so it reads ≈0 right after a compaction). The index checkpoint (`run_index_checkpoint`) calls `DynFieldIndex::maybe_compact` per field, compacting any store whose waste crosses `ThresholdConfig::index_blob_waste_threshold` (percent, default 50; TOML `thresholds.index_blob_waste_threshold`). `Db::checkpoint_index()` forces a flush+compaction on demand.

**Backpressure — don't rely on the ~15 min tick alone.** Between checkpoints a low-cardinality high-churn field can amplify to gigabytes (measured: a boolean over 12k docs bloated to ~1 GB before the timer fired). So the write path also carries a valve: it reads `BlobStore::dead_bytes()` — an **O(1)** running counter of reclaimable bytes (incremented on each `upsert`-replace / `remove_key`, reset by `compact`, seeded from `logical-live` at `open`) — surfaced as `DynFieldIndex::reclaimable_dead_bytes()`, and when a field crosses `ThresholdConfig::index_blob_backpressure_bytes` (default 64 MiB; TOML `thresholds.index_blob_backpressure_bytes`, 0 disables) it fires an `IndexCheckpointTrigger` to run a checkpoint immediately. The trigger is threaded to each `KVStore` like the LSM compaction trigger (set when the checkpoint worker is enabled; inherited by namespaces opened later) and is debounced through a shared `pending` flag so a hot field enqueues at most one checkpoint at a time. **It is an absolute byte cap, not a ratio** — a ratio hits ~100% almost immediately for exactly these fields, so it would fire nearly every write and cost O(capacity) per check; the byte cap bounds peak disk to ~cap per field and self-debounces (compaction zeroes the count). See `db/index_checkpoint_worker.rs`.

`maybe_compact` covers **both** of a field's BlobStores. The bitmap store bloats per **document** (whole-bitmap rewrite per insert). The `keymap/` store (slot_id → value) is written once per distinct value, so a fixed value set never bloats it — but under **distinct-value churn** (values that appear and are later fully removed, freeing their slot) it accumulates dead entries, so it is compacted on its own `keymap_waste_ratio()` crossing the same threshold. The two stores key on the same slot IDs and tombstone the same slots together (a value's slot is freed in both when its bitmap empties), so they are compacted independently while staying consistent.

#### Compaction is crash-safe via a staged file swap — do NOT rewrite in place

Compaction must **never** mutate the live `blobs.keys` / `blobs.vals` in place. Index recovery (`activate_field_index`) loads the on-disk store and replays only the WAL tail *on top* of it — it does **not** rebuild from scratch — so a crash that leaves new key offsets pointing into a half-rewritten value region is inherited as **silent corruption** that replay cannot heal (any value bucket not touched by a key in the replay window stays wrong).

So `compact()` (persistent stores) builds the compacted pair in memory, stages it as `blobs.keys.new` / `blobs.vals.new` (live pair untouched), fsyncs both + the dir, then writes a `compact.commit` marker (fsynced) as the **commit point**, renames both files into place, fsyncs, and removes the marker. `BlobStore::open()` calls `recover_compaction()` first: marker present ⇒ the staged files are complete, finish the swap idempotently; marker absent ⇒ staged files are partial, discard them. **Invariant: after any crash the on-disk pair is fully old or fully new, never a torn mix.** Two-file renames are not atomic as a pair — the marker is what makes the swap atomic, not the write lock. If you touch this path, preserve the ordering (stage+fsync → marker+fsync → rename → drop marker) and keep `recover_compaction` in `open`.

Cost notes: peak heap is ~1× the live (compacted) index — the compacted `val_buf` is built straight from the mmap and written without a pad-copy; don't reintroduce an `iter_entries()` snapshot or a `to_vec()` pad. Compaction runs under the field's `entry.index` **write lock** (`run_index_checkpoint`), so it stalls queries *and* writes **on that field only** for the compaction's I/O duration (dominated by the fsyncs, not the remap). It fires only for fields over the waste threshold at a checkpoint tick, not every tick. If that stall ever matters, the planned fix is two-phase locking (stage+fsync unlocked, swap+remap under the write lock, abort-and-retry on concurrent write) — not reverting to in-place.
