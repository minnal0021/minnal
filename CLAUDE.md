# Project: minnal

**minnal** (மின்னல்) means *lightning* in Tamil. It is a layered, embedded document database: an LSM + value-log KV engine at the bottom, RoaringBitmap field indexing and quantised ANN semantic search in the middle, and a JSON document store with a REST API on top.

**Platform:** Linux and macOS only. The storage engine uses `pread`/`pwrite` and the server requires POSIX signals — Windows is not supported.

## Architecture (bottom → top)

```
minnal_doc_store_api   ← Axum REST API server (binary)
    └── minnal_doc_store   ← JSON schema, doc lifecycle, KV store lifecycle, index builders
            ├── minnal_db      ← LSM + value-log KV engine (WiscKey-style)
            ├── index          ← RoaringBitmap field indices + predicate evaluator
            └── semantic_search ← IVF clustering, RaBitQ quantisation, ANN search
tools/                 ← standalone bulk-loader binary
```

When changing behaviour, edit the lowest layer that owns it. Don't reach through layers.

## Workspace crates

| Crate | Role |
|---|---|
| `minnal_db/` | Core KV engine — LSM tree, value log, WAL, namespaces, TTL |
| `index/` | RoaringBitmap field indexing, predicate query evaluator |
| `semantic_search/` | Vector quantisation + approximate nearest-neighbour search |
| `minnal_doc_store/` | JSON document layer (schema, CRUD, index builders) + raw KV store layer (`KvStoreSchema`) |
| `minnal_doc_store_api/` | Axum HTTP server; reads TOML config on startup; serves both `/stores` (doc) and `/kv-stores` (KV) routes |
| `tools/` | Bulk loader binary |

## Build & test

```sh
cargo build                          # full workspace
cargo test                           # full workspace
cargo test -p minnal_db              # single crate
cargo bench -p minnal_db             # run benchmarks (criterion, HTML reports in target/criterion/)
cargo clippy -- -D warnings          # lint (run before committing)
cargo fmt --check                    # format check
```

Benchmarks live in `minnal_db/benches/`: `bench_write`, `bench_read`, `bench_scan`, `bench_mixed`, `bench_wal`, `bench_typed`.

## Project structure (minnal_db)

```
src/lib.rs          — public API surface (re-exports only)
src/db/             — facade, config, namespace, WAL, fail_log, TTL, stats, error
src/store/          — LSM tree, value log, GC workers
src/support/        — SIMD helpers and shared utilities
tests/              — integration tests only; unit tests live inline in each module
```

## Code conventions

- Use `thiserror` for all error types (library code and public APIs)
- All public APIs must have doc comments
- Prefer `impl Trait` over `Box<dyn Trait>` where possible
- Use `rkyv` for zero-copy serialisation of stored values (re-exported as `minnal_db::rkyv_derives`)
- Use `parking_lot` primitives (`RwLock`, `Mutex`) — not `std::sync`
- Use `Db` / `AsyncDb` with namespaces as the public entry points. `Database`, `AsyncDatabase`, `KVStore`, `MinnalStore`, and `AsyncMinnalStore` are no longer re-exported from the crate root — the first three are internal types behind the `Db`/`Namespace` facade; the last two were removed entirely.
- Writes are single-op only (`put`/`delete`, each its own WAL-fsynced transaction). There is no multi-op batch/transaction primitive — secondary structures (field index, vector index) are derived and reconstructable, so cross-namespace writes are ordered single ops (e.g. doc-store writes the document, then enqueues the embed marker), reconciled on recovery/re-index rather than committed atomically. **Exception:** the quantised vector-index payloads (`{ns}_sparse_vector`, `{ns}_sparse_vector_meta`, `{ns}_dense_vector`) are written **no-WAL** (`put_no_wal`) since they are bulky and re-embeddable — a crash before flush drops them and reconciliation re-enqueues the affected docs. Their cleanup **deletes** stay WAL-backed (a lost delete would orphan an index entry, which reconciliation cannot heal). See `semantic_search/CLAUDE.md` → *Durability split*.
- Multi-key reads come in **unbounded** (`scan_prefix` / `range` — whole result set in memory) and **bounded cursor-paginated** (`scan_page_batch` / facade `scan(cursor, end, limit)` → `next_cursor`) forms. User-facing scan/range endpoints (doc `GET /stores/{ns}/docs[ /prefix]`, KV `GET /kv-stores/{ns}/kv[ /prefix]`) are cursor-paginated: they take `cursor` + `limit` and return `{results, next_cursor}` (no `total`/`page_no`), so each page resolves only its own values. The cursor is opaque raw key bytes (hex-encoded at the REST boundary). See `minnal_db/CLAUDE.md` → *Scan / read API* for the memory profile and the pointer-resolution invariant.
- Run `cargo clippy` before committing

## Dependencies (notable)

| Crate | Used for |
|---|---|
| `tokio` (multi-thread) | Async runtime throughout |
| `rkyv` | Zero-copy serialisation of values and index entries |
| `serde` + `serde_json` | Config, doc-store JSON, REST payloads |
| `parking_lot` | All internal locks |
| `mm3h` | MurmurHash3 for default row-ID derivation |
| `memmap2` | Memory-mapped SSTable reads in `index/` |
| `axum` | REST API in `minnal_doc_store_api/` |
| `simsimd` | SIMD distance computations in `semantic_search/` |
| `rayon` | Parallel index build in `semantic_search/` |

Do not add new dependencies without discussion.

## Configuration

Both binaries accept a TOML config file:

```sh
# KV engine standalone
minnal_db /path/to/minnal.toml

# Full API server
minnal_doc_store_api /path/to/config.toml
# or via env
MINNAL_CONFIG_FILE=/path/to/config.toml minnal_doc_store_api
```

Reference config:
- `config/sample.toml` — full API server config (covers all KV engine options plus `[api]`, `[logging]`, `[semantic_search]`)

Key knobs: `sharding.num_buckets` (cannot change after data exists), `sync.records_per_sync` (value-log fsync cadence — the WAL is fsynced on every write regardless), `thresholds.value_log_waste_threshold` (GC trigger), `thresholds.index_blob_waste_threshold` (field-index bitmap compaction trigger at checkpoint), `recovery.fail_log_dir` (where fail-log JSON files are written; defaults to `<db_path>/fail_logs`).

## Semantic search

`semantic_search` uses a **two-pass ANN search** with dual RaBitQ quantisation:
- **Pass 1 (sparse)** — 1-bit chunk embeddings (sliding-window) stored in `{ns}_sparse_vector`, prefix-scanned by IVF cluster for fast candidate selection. Scored with **ColBERT MaxSim**: `S(q,d) = Σ_i max_j ⟨q_i, d_j⟩` — sum over query tokens of the per-token best chunk score.
- **Pass 2 (dense)** — multi-bit whole-doc embeddings stored in `{ns}_dense_vector`, fetched by `doc_id` for high-precision re-ranking.

Requires an **external embedding service** (default: `http://localhost:8001`). Vector search will not work without it. Cluster centroids are bundled at `service/embedding_support/qwen/clusters.json`; set `semantic_search.cluster_path` in config to point at them. The key config knobs are `number_of_bits_for_dense_quantisation`, `n_probes`, `first_pass_sparse_search_top_k`, `window_size`, and `sliding_size`. See `semantic_search/Semantic-Search-Architecture.md` for the full design.
