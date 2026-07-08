# minnal_db — KV Engine

WiscKey-style embedded store: keys live in an LSM tree, values in a separate value log, reducing write amplification for large values.

## Key files

| File | Role |
|---|---|
| `src/lib.rs` | Public re-exports only — all types surfaced here |
| `src/db/facade.rs` | `Db` and `AsyncDb` — the only entry points callers use |
| `src/db/database.rs` | Internal coordinator: owns all subsystems, routes ops |
| `src/db/namespace.rs` | Namespace metadata, `FieldMeta`, `FieldId` |
| `src/db/namespace_index.rs` | Per-namespace index registry, `ExtractorFn`, `RowIdFn`, `RowToKeyFn` |
| `src/db/config.rs` | `DbConfig`, `SyncConfig`, `ThresholdConfig`, `ScheduledTaskConfig` |
| `src/db/toml_config.rs` | TOML → `DbConfig` parser (`MinnalTomlConfig`) |
| `src/db/wal.rs` | WAL segment structure, `WalEntry`, recovery |
| `src/db/fail_log.rs` | Recovery fail-log writer — JSON files for entries that fail after retry |
| `src/db/error.rs` | `KVError` — the single error type for all operations |
| `src/store/lsm/lsm_tree.rs` | LSM tree + compaction logic |
| `src/store/lsm/skip_list.rs` | In-memory memtable (skip list) |
| `src/store/lsm/lsm_manifest.rs` | SSTable manifest (public, returned by `Db::lsm_manifests`) |
| `src/store/value_log/sharded.rs` | Sharded value log: reads and writes, GC |
| `src/support/simd_support.rs` | SIMD helpers |

## Data flow (write path)

```
Db::put(key, value)
  → Database::put
      → WAL append (fsynced)
      → Value log write → pointer
      → Memtable insert with pointer (best-effort w/ bounded retry; ERROR-logged
          if all retries fail — data IS durable in WAL and replayed on next startup)
      → When memtable hits threshold → flush to SSTable
      → Background: LSM compaction, value-log GC, WAL GC, TTL cleanup
```

Every write is a single-op WAL transaction — there is no batch/transaction primitive. The in-memory apply is best-effort with bounded retry (`Database::apply_with_retry`): once the WAL fsync succeeds the op is durable and the call returns `Ok` regardless of the apply outcome. Recovery **replays all un-persisted entries in global sequence order** (so writes to the same key resolve to the last writer), retries each once, and writes persistent failures to `fail_logs/<timestamp>.json`.

## Concurrency & correctness invariants

These are load-bearing; preserve them when touching the store/GC paths:

- **Sequence == WAL order, and the memtable resolves conflicts by it.** Writers allocate the global sequence (`Database::next_seq`, shared into every `KVStore` via `set_seq_counter`) *inside* the WAL append lock, so on-disk WAL order == sequence order. The memtable resolves same-key conflicts **highest-sequence-wins** (`SkipList::try_insert_with_seq` / `remove_with_seq`, serial-number `u32` comparison): a lower-sequence write applied later is dropped. This makes the live winner for racing same-key writes identical to recovery's winner (recovery replays in sequence order), so a value observed live survives a crash. Non-WAL writes (TTL expiry, bulk, tests) allocate from the same counter (`KVStore::alloc_seq`) so there is one consistent sequence space.
- **GC vs. writes/deletes.** `put_to_storage`/`delete_from_storage` hold the value-log **bucket write lock** across both the value-log write/tombstone and the LSM insert/delete. GC compacts a bucket under the same lock, so its scan→reinsert is atomic w.r.t. those mutations.
- **GC reinsert is compare-and-set and sequence-preserving.** After the file swap, GC re-points a key only if it still maps to the exact pointer GC copied (validated via `lsm.get_with_seq`), and re-inserts under the key's *existing* sequence so the relocation neither resurrects a deletion nor changes the key's version. The CAS guard runs under the value-log bucket write lock (atomic w.r.t. concurrent deletes), and `get_with_seq` is seq-aware (below), so a re-point fires only when no higher-seq tombstone exists in any *non-obsolete* layer. Note this is a guard, not a guarantee that no recency-vs-seq inversion ever exists on disk: out-of-order ro→L0 flush can stamp a value's L0 file with a larger `created_at_ms` than a higher-seq tombstone's file. That is why same-key resolution everywhere is by seq, not file order (see below) — the seq-aware merge collapses such an inversion correctly instead of baking it into L1. The GC-journal replay applies the same rule (skip keys now absent; preserve sequence).
- **Cross-layer reads resolve by highest sequence, not layer order.** `get` / `get_with_seq` (`lsm_tree.rs`) gather the newest-`seq` entry across the active memtable, every read-only memtable, all L0 files, and L1, tombstone⇒`None` — they do **not** early-return at the first layer holding the key, because a GC re-point could (in principle) leave a low-seq value in a newer layer above a higher-seq tombstone. **Fast path:** an active-memtable hit whose `seq` is `>=` the per-tree `max_lower_seq` is authoritative and returns immediately — so normal writes (always newest seq) keep the early-return; only a GC re-point's low seq falls through to the full scan. `max_lower_seq` is the max seq of everything below the active memtable; its contract is **bound ≥ max seq of every lower layer**, so it is folded at flush (records leaving the active memtable) **and from every L0 *and* L1 file at open** — folding only L1 at open left it too low after a restart and let a low-seq active entry wrongly short-circuit above a higher-seq L0 tombstone (resurrection). Measured cost vs. layer-order resolution: ~5–10% on reads.
- **All same-key conflict resolution is by highest `seq`, never by layer/file recency.** Point reads (`get`/`get_with_seq`), the GC liveness scan (`key_pointer_pairs`), and the L0→L1 merge (`merge_level0_to_level1` — both its cross-L0-file `by_key` collapse *and* `two_way_merge` with L1) all resolve a key's winner by the **highest global write `seq`** (wraparound-aware `seq_newer`/`seq_newer_or_eq`; exact ties prefer the newer layer). This is sound because every write — WAL-backed and non-WAL (TTL/bulk/test) alike — draws from one global counter (`Database::next_seq` / `KVStore::alloc_seq`), so seqs are globally comparable. **Recency is NOT a safe proxy for seq.** File `created_at_ms` is stamped at ro→L0 *flush* time, which can run out of seal/seq order (a higher-version memtable can flush to L0 before a lower one), and a GC re-point preserves a low seq into a newer layer — either leaves a lower-seq value in a "newer" file above a higher-seq tombstone. When the merge resolved by recency it baked the wrong winner into L1: it dropped the higher-seq tombstone and kept the lower-seq value → **resurrection of deleted keys** (the symmetric inversion made `key_pointer_pairs` treat a live value as dead → value-log GC drop = data loss). Confirmed by repro: a value's L0 file had a *larger* `created_at_ms` than the higher-seq tombstone's file. So keep these paths seq-resolved; do not reintroduce last-write-wins-by-file-order. (`scan_prefix` / `scan_prefix_in_bucket` still merge oldest→newest for *key set* assembly, but value correctness comes from L1 always being seq-correct plus the per-record seq-validity check on the value read; a transient L0 inversion there is read-committed and self-heals at the next merge.)
- **SSTable lookups are tri-state** (`SsLookup::Found(ptr, seq)/Deleted(seq)/Missing`): both `Found` and `Deleted` carry the entry's seq so cross-layer resolution can compare them; an L0 tombstone must shadow a live L1 entry, so "deleted" and "absent" must stay distinct. The active memtable is checked via `SkipList::entry` (not `get_value`) for the same reason — a memtable tombstone must shadow lower layers.
- **Lock-free reads use the bucket swap generation.** `KVStore::get` samples the generation before/after the pointer+value read and only trusts an unchanged generation; otherwise it retries, then falls back to a lock-held read. `key_value_pairs` (GC's scan) reads L1 + L0 + read-only memtables + active memtable.
- **Per-write WAL fsync is deliberate — do NOT add group-commit.** Each `put`/`delete` fsyncs the WAL before returning (durable-on-return against power loss). Group-committing independent writes would acknowledge them before they hit stable storage, weakening the guarantee. `records_per_sync` tunes only the value-log fsync cadence (safe — the WAL already has a durable copy).

## Typed value API

`put_typed<T>` / `get_typed<T>` use `rkyv` for zero-copy serialisation. Types must derive `rkyv::Archive + rkyv::Serialize + rkyv::Deserialize`. These are re-exported via `minnal_db::rkyv_derives` so downstream crates don't need a direct `rkyv` dependency.

## Scan / read API

Two families of multi-key reads, differing in **memory profile**:

- **Unbounded** — `scan_prefix_batch` / `scan_range_batch` (facade: `scan_prefix` / `range`) materialise the *entire* matching result set — keys **and all values** — in one `Vec`. Memory scales with total matches. Use only when the result is known-small or genuinely needs to be whole.
- **Bounded (cursor-paginated)** — `scan_page_batch(cursor, end, limit)` (facade: `Namespace::scan` / `AsyncNamespace::scan`) returns at most `limit` pairs plus a `next_cursor` (the raw key the next page starts at, or `None` at the end). **Each page resolves only its own values**, so peak memory is O(page), not O(total). Prefer this for anything user-facing or unbounded. `end` (exclusive) bounds the scan to `[cursor, end)`; a prefix scan passes the prefix's upper bound (`minnal_doc_store::prefix_upper_bound`) as `end` so it stops at the prefix instead of walking the keyspace tail.

Semantics: cursor pagination is **read-committed across the walk**, not a point-in-time snapshot — a key deleted/overwritten between pages reflects the newer state (each page is its own generation-stable read + `refetch_dropped` fallback). Acceptable here because writes are single-op (no transactions).

POINTER-RESOLUTION INVARIANT (load-bearing): the LSM scan returns value-log *pointers* that are LSM-complete but **must be resolved inside the same `read_generation_stable` bracket** (or under the value-log bucket write lock). Resolving a pointer outside that bracket reopens the wrong-file window closed in `420ac8e`. The `end` bound added for pagination only filters keys in the LSM merge — it does **not** change this resolution rule.

Caveat (not yet fixed): `scan_page_batch` bounds the *value* memory per page, but the LSM merge still builds a keys+pointers map for everything `>= cursor` before `take(limit)` (no values — the small cost, shrinks as the cursor advances). Bounding that transient is the deferred "bounded k-way merge" follow-up; don't reverse the oldest→newest merge order if you attempt it (load-bearing for tombstone suppression — see [`lsm_tree.rs`](src/store/lsm/lsm_tree.rs) `scan_prefix` docs).

## Namespaces

Each `Db` has a default namespace plus any number of named namespaces. Each namespace has its own keyspace. Field indexes are registered per-namespace via `Db::register_field` / `Db::register_extractor`.

`remove_namespace` reclaims on-disk storage (the `ns_{name}` data dir + `index/{ns_id}` subtree). The step order is load-bearing for crash safety: **persist the registry deletion first**, then flush/drop the store, mark its WAL entries persisted, and only then delete files. Recovery skips WAL entries whose `namespace_id` is absent from the registry (`recover_from_wal`), so a crash mid-cleanup never resurrects the namespace. File deletion never touches the shared WAL — keep it that way.

Marking the namespace's WAL entries persisted is only the *clean-shutdown* fast path; a crash before it leaves those entries un-persisted, which is **not a permanent WAL leak**. Un-persisted entries force `persisted_entries < total_entries`, so the next startup always runs recovery, and recovery's cleanup pass marks **every** scanned entry persisted (orphans included, no per-namespace filter) and advances the watermark to the WAL tail — so WAL GC reclaims those segments on its next cycle. Don't "fix" the orphan entries by filtering them out of that cleanup pass; the unfiltered sweep is what self-heals the leak.

## Row IDs

By default, field-index row IDs are **dense, monotonic** integers (0, 1, 2, …) assigned per namespace by a `RowMap` sidecar (`index::RowMap`, on disk under `index/{ns}/rowmap/`). Dense IDs make the RoaringBitmaps pack into a few containers instead of scattering one-per-doc, which is the whole reason field-index bitmaps stay small. `resolve_row_id_alloc` (put/replay) allocates on first sighting; `resolve_row_id_get` (delete/query fallback) never allocates; `query_keys` resolves hits back to keys via `RowMap::key_for` (O(|hits|)).

The map is a **derived** structure with the same durability model as the field index: writes mutate mmap in memory, flushed at the index checkpoint, rebuilt by WAL replay on open. **Load-bearing ordering:** `run_index_checkpoint` flushes the row map (advancing its `rowmap.ckpt` marker) **before** any field index, so the map is always at least as durable as every persisted bitmap bit — otherwise a crash could leave a bit whose row ID can't be reproduced. The row map's `key → id` slot table is in-memory (rebuilt from the on-disk id array on open), so it is never a persisted source of truth; a torn tail past the marker is ignored. See `index::RowMap` docs.

**Escape hatch (unchanged):** register `RowIdFn` + `RowToKeyFn` together to bypass the row map and derive the ID directly from the key (e.g., a UUID in the key) — O(|hits|) query resolution with zero map. These are the only two row-ID sources (`resolve_row_id_alloc`/`resolve_row_id_get`): a `RowIdFn` if registered, otherwise the dense `RowMap`. The row map is always loaded before any field is activated (`ensure_rowmap` in `activate_field_index`), so an indexed namespace always has one or the other — resolution never falls through. (The old Murmur3 `key_to_row_id` fallback was removed once dense row IDs became the default.)

## Sharding

`num_buckets` (default 16) shards both the value log and LSM SSTables. Keys are hash-distributed. **Cannot change `num_buckets` on an existing database** — the value is locked at creation time.

## Background workers

All background workers (`lsm_worker`, `gc_value_log_worker`, `wal_worker`, `ttl_worker`, `index_checkpoint_worker`) are spawned on `Db::open` and stopped on `Db::shutdown`. Always call `shutdown()` — dropping without it may lose buffered writes.

Each worker is a **single global task** that fans out over namespaces — there is no per-namespace worker. `ttl_worker` follows the same shape as `gc_value_log_worker`: one `TtlWorker` task (handle in `Database::ttl_worker`) driven by `TtlTarget::run_ttl_pass`, which iterates `NamespaceRegistry::ttl_configs` (`ns_id -> (Duration, max_deletes_per_run)`) and calls `KVStore::expire_records` for each TTL-enabled namespace.

**TTL config is persisted** in the namespace registry file (`RegistryData::ttl_configs`), not just held in memory — it survives restarts. The registry — not `store.ttl` — is the source of truth for what the worker processes: `namespace_with_ttl` calls `registry.set_ttl_config` (persists) and starts the worker on first use; `shutdown_ttl_worker(ns_id)` calls `remove_ttl_config` (persists, stops expiring that namespace without killing the task); `remove_namespace` drops the config as part of `registry.remove`. On open, each store is reopened with its persisted ttl (`open_with_ttl`) and `enable_all_workers` restarts the worker when any TTL config exists. **Do not clear `ttl_configs` on shutdown** — that would wipe the durable config. The worker only starts when at least one TTL namespace exists, so a database with none runs no TTL task. (Sync-mode `Db::namespace_with_ttl` has no `max_deletes` and no worker, so it sets `store.ttl` in memory only and is intentionally *not* persisted.)

## WAL ownership

All WAL logic (append, GC, recovery, sequence tracking) lives exclusively in `database.rs` (`Database`, the internal coordinator). Do not add WAL code anywhere else.

## Public API surface

`Db` / `AsyncDb` (facade) with namespaces are the only entry points. `Namespace` is the scoped per-namespace handle returned by `Db::namespace()`.

`Database`, `AsyncDatabase`, and `KVStore` are **internal** types: `Database`/`AsyncDatabase` is the coordinator behind `Db`/`AsyncDb`; `KVStore` is the per-namespace store behind `Namespace`. They are no longer re-exported from the crate root (the deprecated `pub use` aliases were removed). Refer to them only via internal paths (`crate::db::database::Database`, `crate::db::kv_store::KVStore`).

`MinnalStore` / `AsyncMinnalStore` have been removed entirely. `src/db/db.rs` now holds only integration tests for the facade.
