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

- [ ] 1a. `ContainerStore` = enum of `Heap` / `Mapped`, same public methods;
      add `with_container` (read without clone) and `modify` (change in place)
- [ ] 1b. `RoaringBitmap` mutators and readers use the in-place methods, so an
      insert into a heap bitmap allocates nothing beyond the container itself
- [ ] 1c. Tests: 65,536 inserts into one bitmap stay within a small multiple of
      8 KB of memory (measure the store's own size, not process RSS); every
      existing bitmap / container_store / index test passes
- [ ] 1d. Re-measure (insert cost, query cost via `bench_predicate`), commit

## Step 2: dirty-container overlay with a memory budget

- [ ] 2a. `FieldIndex` overlay: per slot, changed containers (`Some` or deleted).
      Writes go to the overlay, loading only the row's container; reads merge
      overlay over file
- [ ] 2b. Global budget (`IndexOverlayBudget`, atomic byte counter shared by all
      fields of a `Database`): soft limit requests a checkpoint, hard limit makes
      the writer spill its field before returning. Config:
      `thresholds.index_overlay_soft_bytes` (32 MiB),
      `thresholds.index_overlay_hard_bytes` (64 MiB)
- [ ] 2c. Checkpoint / spill writes the overlay out in today's format (one blob
      per changed slot) and clears it; checkpoint marker only after the flush
- [ ] 2d. `remove_all_for_row(s)` probes the row's container per slot instead of
      deserialising every bitmap
- [ ] 2e. WAL replay uses the normal path; remove its group-by-value code and
      inline compaction
- [ ] 2f. Tests: budget never exceeded with the checkpoint worker paused;
      overlay reads equal file reads; replay equivalence; existing index and
      crash tests. Re-measure, commit

## Step 3: container-granular files, ordered checkpoint write

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

## Step 4: docs

- [ ] 4a. `index/CLAUDE.md`, `Index-Architecture.md`, root `CLAUDE.md` key knobs,
      `config/sample.toml`; evaluation doc gets a results section

## Log

(one line per finished step: date, commit, headline numbers)
