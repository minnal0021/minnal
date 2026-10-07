# Field-index write-back: implementation plan

Design and measurements: [`Field-Index-Write-Back-Evaluation.md`](Field-Index-Write-Back-Evaluation.md).
Branch `roaring_bit_map_fixes`. Each step ends with tests, clippy for all four
feature combinations, a commit, and a tick here, so work can stop and resume
at any step boundary.

On-disk storage stays memory-mapped files throughout. What changes is the
in-memory bitmap (step 1), when the files are written (step 2), and their
layout and write order (step 3).

## Resume here

To pick up: read this file, run `git log --oneline main..` on the branch, and
continue at the first unticked item.

## Step 1: heap-backed in-memory bitmap

`ContainerStore` gets two backings: **heap** (a `BTreeMap<u128, Container>`
changed in place, with a cached cardinality) for `RoaringBitmap::new()`, and
**mapped** (today's two files) for `RoaringBitmap::create` / `open`.

- [x] 1a. `ContainerStore` = enum of `Heap` / `Mapped`, same public methods;
      add `with_container` (read without clone) and `modify` (change in place)
- [x] 1b. `RoaringBitmap` mutators and readers use the in-place methods, so an
      insert into a heap bitmap allocates nothing beyond the container itself
- [x] 1c. Tests: 65,536 inserts into one bitmap stay within a small multiple of
      8 KB of memory (measure the store's own size, not process RSS); every
      existing bitmap / container_store / index test passes
- [x] 1d. Re-measure (insert cost, query cost via `bench_predicate`), commit

## Step 2: dirty-container overlay with a memory budget

- [x] 2a. `FieldIndex` overlay: per slot, changed containers (`Some` or deleted).
      Writes go to the overlay, loading only the row's container; reads merge
      overlay over file
- [x] 2b. Global budget (`IndexOverlayBudget`, atomic byte counter shared by all
      fields of a `Database`): soft limit requests a checkpoint, hard limit makes
      the writer spill its field before returning. Config:
      `thresholds.index_overlay_soft_bytes` (32 MiB),
      `thresholds.index_overlay_hard_bytes` (64 MiB)
- [x] 2c. Checkpoint / spill writes the overlay out in today's format (one blob
      per changed slot) and clears it; checkpoint marker only after the flush
- [x] 2d. `remove_all_for_row(s)` probes the row's container per slot instead of
      deserialising every bitmap
- [x] 2e. WAL replay: bounded by the shared budget (set before replay). Kept
      its group-by-value batching (saves a lookup per key) and the inline
      compaction guard (a window larger than the budget still spills
      repeatedly); comments updated
- [ ] 2f. Tests: budget never exceeded with the checkpoint worker paused;
      overlay reads equal file reads; replay equivalence; existing index and
      crash tests. Re-measure, commit

## Step 3: container-granular files, ordered checkpoint write

Why step 3 matters beyond write volume: in step 2 the overlay holds **whole**
bitmaps. Once the bitmaps being written add up to more than the hard limit,
every write spills a whole bitmap again — today's per-write behaviour (seen in
a test with a deliberately tiny budget: 400k writes took 51 s, debug build).
In production that is a low-cardinality field whose bitmaps exceed 64 MiB
together (a boolean over ~300M rows). Step 3's overlay holds only changed
containers (≤ 8 KB each), which removes the case.

- [ ] 3a. New on-disk format: `blobs.vals` holds container blobs and per-slot
      directories; `blobs.keys` maps slot → directory. 64-byte checksummed
      slots; write position recomputed at open; rehash into a new file + rename.
      Format version bump (greenfield: old indexes rebuild)
- [ ] 3b. Ordered write: append + `msync` values, then slot updates, then
      `msync` keys, then the checkpoint marker; no fsync under the field lock
- [ ] 3c. Torn-slot detection at open → `FullRebuild` gap record
- [ ] 3d. Compaction rewritten for the new layout (same staged-swap protocol)
- [ ] 3e. Crash tests: crash before each write step, with a random subset of
      dirty pages "written back"; reopen + replay must equal a full rebuild.
      Re-measure, commit

## Step 4: docs (every place that describes index storage)

Each step also updates the doc comments of the code it changes. This step
brings the reader-facing docs and agent notes in line once the design has
landed, following the root `CLAUDE.md` *Writing docs* rules: newcomer-first,
no history, every fact checked against the code, REST examples run against a
scratch server, a cold read of the whole diff before committing.

Before starting, re-run the search; the list below is what it found on
2026-10-07:

    grep -rlnI -i "blob store\|BlobStore\|index_blob_\|append-only\|ContainerStore\|anonymous mmap\|blobs.vals\|bitmap" \
        --include=*.md --include=*.toml --include=*.rs . | grep -v "^./target\|^./work/"

- [ ] 4a. **Agent notes** (may keep the "why it changed"):
      `minnal_db/src/index/CLAUDE.md` (ingest/checkpoint/compaction diagram,
      *Blob store is append-only*, backpressure, crash-safety sections),
      `minnal_db/CLAUDE.md` (index checkpoint / replay batching notes),
      root `CLAUDE.md` (key knobs: `index_blob_*` thresholds, new overlay limits)
- [ ] 4b. **Architecture doc**: `minnal_db/src/index/Index-Architecture.md`
- [ ] 4c. **Operator docs**: `minnal_db_api/README.md` — index checkpoint,
      `GET /admin/storage/index-waste`, `GET /admin/indices/{ns}/{field}/blob-stats`,
      `POST /admin/storage/index-checkpoint`, config/metrics tables (new overlay
      fields and limits; changed meaning of the backpressure setting);
      `minnal_db/README.md`, `QUICKSTART.md`s and `benchmark.md` if they mention
      index storage or its knobs
- [ ] 4d. **Config**: `config/sample.toml` comments; `minnal_db/src/db/config.rs`
      and `toml_config.rs` doc comments; `minnal_db_api/src/config.rs` and
      `config_report.rs` (reported knobs)
- [ ] 4e. **Code doc comments not touched in steps 1–3**: `bitmap.rs` type docs,
      `field/field_index.rs`, `field/value.rs` (crash-atomicity section),
      `blob_store.rs`, `db/kv_store.rs` and `db/database.rs` (replay and
      backpressure comments), `db/index_checkpoint_worker.rs`,
      `db/index_manager.rs`, admin route docs in
      `minnal_db_api/src/routes/admin_indices.rs` / `admin_storage.rs`,
      `doc_store/store/diagnostics.rs`
- [ ] 4f. **Feature requests**: `FEATURE-REQUEST.md` entries that describe
      the blob store or backpressure (mark what this work closes)
- [ ] 4g. Evaluation doc: add a *Results* section with before/after numbers

## Log

(one line per finished step: date, commit, headline numbers)

- **Step 1** (2026-10-07): transient bitmaps on the heap. 65,536 inserts into
  one bitmap: 2.9 µs → 0.004 µs each, +384 MiB → no RSS growth. Index insert
  (file-backed, 50k rows): bool 10.1 → 2.1 µs, 16 values 5.9 → 0.96 µs, 1,000
  values 3.2 → 0.23 µs; bytes appended unchanged (step 2). `bench_predicate`
  8–21× faster on every case (`str_eq` 11.5 µs → 0.55 µs, `int_range` 1.11 ms →
  134 µs, `three_way_and` 799 → 88 µs). Measured against `0270020`, release,
  separate target dirs.
