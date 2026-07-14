# Project: minnal

**minnal** (மின்னல்) means *lightning* in Tamil. It is a layered, embedded document database: an LSM + value-log KV engine at the bottom, RoaringBitmap field indexing and quantised ANN semantic search in the middle, and a JSON document store with a REST API on top.

**Platform:** Linux and macOS only. The storage engine uses `pread`/`pwrite` and the server requires POSIX signals — Windows is not supported.

## Architecture (single crate, feature-gated layers)

`minnal_db` is **one publishable library crate**; the former `index`,
`semantic_search`, and `minnal_doc_store` crates are folded in as modules and
selected by cargo feature. `minnal_db_api` and `tools` are `publish = false`
binary crates on top.

```
minnal_db_api          ← Axum REST API server (binary; enables doc-store + semantic-search)
tools                  ← standalone bulk-loader binary
minnal_db  (library, features: kv-store [default], doc-store, semantic-search)
    src/db/ src/store/     ← LSM + value-log KV engine (WiscKey-style)   [always]
    src/index/            ← RoaringBitmap field indices + predicate evaluator   [always, part of kv-store]
    src/vector_kv.rs      ← vector↔KV bridge (DbVectorStore, upsert/delete, qemb cache)   #[cfg(semantic-search)]
    src/semantic_search/  ← IVF clustering, RaBitQ quantisation, ANN search   #[cfg(semantic-search)]
    src/doc_store/        ← JSON schema, doc lifecycle, KV store lifecycle, index builders   #[cfg(doc-store)]
```

When changing behaviour, edit the lowest layer/module that owns it. Don't reach across modules.

## Features (2×2: storage layer × semantic search)

| Features | Capability | Vector deps (reqwest/simsimd/rayon/futures) |
|---|---|:---:|
| `kv-store` (default) | KV engine + field indexing | ❌ |
| `kv-store` + `semantic-search` | + vector search on raw namespaces (`vector_kv`) | ✅ |
| `doc-store` | JSON documents + field queries | ❌ |
| `doc-store` + `semantic-search` | full: docs + embed-on-write + vector search | ✅ |

`doc-store` and `semantic-search` are **orthogonal**; field indexing is always in
the base. With `doc-store` but not `semantic-search`, a semantic-search store is
rejected at runtime (`DocStoreError::SemanticSearchNotCompiled`). Greenfield —
make new behaviour the default; no migration tooling.

## Build & test

```sh
cargo build                                              # default (kv-store)
cargo build --all-features                               # everything
cargo test --workspace --all-features                    # full suite
cargo test -p minnal_db --no-default-features --features doc-store   # one feature set
cargo bench -p minnal_db                                 # criterion (HTML in target/criterion/)
cargo clippy --workspace --all-targets -- -D warnings    # lint before committing
cargo fmt --check                                        # format check
```

When touching feature-gated code, verify each combo compiles: default / `--features doc-store` / `--features semantic-search` / `--all-features` (clippy `-D warnings` must be clean in all). Benchmarks live in `minnal_db/benches/`; `bench_distance_estimation` needs `--features semantic-search`.

## Project structure (minnal_db)

```
src/lib.rs             — public API surface (re-exports; feature-gated)
src/db/                — facade, config, namespace, WAL, fail_log, TTL, stats, error
src/store/             — LSM tree, value log, GC workers
src/support/           — SIMD helpers and shared utilities (incl. test_db_config)
src/index/             — folded field-indexing module (always compiled)
src/semantic_search/   — folded ANN module            #[cfg(semantic-search)]
src/vector_kv.rs       — vector↔KV storage bridge       #[cfg(semantic-search)]
src/doc_store/         — folded document-store module   #[cfg(doc-store)]
tests/                 — integration tests only; unit tests live inline in each module
```

## Code conventions

- Use `thiserror` for all error types (library code and public APIs)
- All public APIs must have doc comments
- Prefer `impl Trait` over `Box<dyn Trait>` where possible
- Use `rkyv` for zero-copy serialisation of stored values (re-exported as `minnal_db::rkyv_derives`)
- Use `parking_lot` primitives (`RwLock`, `Mutex`) — not `std::sync`
- Use `Db` / `AsyncDb` with namespaces as the public entry points. `Database`, `AsyncDatabase`, `KVStore`, `MinnalStore`, and `AsyncMinnalStore` are no longer re-exported from the crate root — the first three are internal types behind the `Db`/`Namespace` facade; the last two were removed entirely.
- Writes are single-op only (`put`/`delete`, each its own WAL-fsynced transaction). There is no multi-op batch/transaction primitive — secondary structures (field index, vector index) are derived and reconstructable, so cross-namespace writes are ordered single ops (e.g. doc-store writes the document, then enqueues the embed marker), reconciled on recovery/re-index rather than committed atomically. **Exception:** the quantised vector-index payloads (`{ns}_sparse_vector`, `{ns}_sparse_vector_meta`, `{ns}_dense_vector`) are written **no-WAL** (`put_no_wal`) since they are bulky and re-embeddable — a crash before flush drops them and reconciliation re-enqueues the affected docs. Their cleanup **deletes** stay WAL-backed (a lost delete would orphan an index entry, which reconciliation cannot heal). The system-wide query-embedding cache (`system_qemb_cache`) goes further — it is no-WAL on **all** CRUD (`put_no_wal` + `delete_no_wal`), since it is TTL-bounded and fully regenerable, so even a lost delete is harmless (the stale entry just expires). See `minnal_db/src/semantic_search/CLAUDE.md` → *Durability split*.
- Multi-key reads come in **unbounded** (`scan_prefix` / `range` — whole result set in memory) and **bounded cursor-paginated** (`scan_page_batch` / facade `scan(cursor, end, limit)` → `next_cursor`) forms. User-facing scan/range endpoints (doc `GET /stores/{ns}/docs[ /prefix]`, KV `GET /stores/{ns}/kv[ /prefix]`) are cursor-paginated: they take `cursor` + `limit` and return `{results, next_cursor}` (no `total`/`page_no`), so each page resolves only its own values. The cursor is opaque raw key bytes (hex-encoded at the REST boundary). See `minnal_db/CLAUDE.md` → *Scan / read API* for the memory profile and the pointer-resolution invariant.
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
| `axum` | REST API in `minnal_db_api/` |
| `simsimd` | SIMD distance computations in `semantic_search/` |
| `rayon` | Parallel index build in `semantic_search/` |

Do not add new dependencies without discussion.

## Configuration

`minnal_db` is a library crate (embed it and build a `DbConfig` in code). The
API server binary accepts a TOML config file:

```sh
# Full API server
minnal_db_api /path/to/config.toml
# or via env
MINNAL_CONFIG_FILE=/path/to/config.toml minnal_db_api
```

Reference config:
- `config/sample.toml` — full API server config (covers all KV engine options plus `[api]`, `[logging]`, `[semantic_search]`)

Key knobs: `sharding.num_buckets` (cannot change after data exists), `value_log.page_size_bytes` (value-log page size, default 64 MiB — also fixed at creation: it is encoded in every value pointer), `sync.records_per_sync` (value-log fsync cadence — the WAL is fsynced on every write regardless), `thresholds.value_log_waste_threshold` (GC trigger), `thresholds.index_blob_waste_threshold` (field-index bitmap compaction trigger at checkpoint), `recovery.fail_log_dir` (where fail-log JSON files are written; defaults to `<db_path>/fail_logs`).

## Semantic search

`semantic_search` uses a **two-pass ANN search** with dual RaBitQ quantisation:
- **Pass 1 (sparse)** — 1-bit chunk embeddings (sliding-window) stored in `{ns}_sparse_vector`, prefix-scanned by IVF cluster for fast candidate selection. Scored with **ColBERT MaxSim**: `S(q,d) = Σ_i max_j ⟨q_i, d_j⟩` — sum over query tokens of the per-token best chunk score.
- **Pass 2 (dense)** — multi-bit whole-doc embeddings stored in `{ns}_dense_vector`, fetched by `doc_id` for high-precision re-ranking.

Requires an **external embedding service** (default: `http://localhost:8001`). Vector search will not work without it. Cluster centroids are bundled at `service/embedding_support/qwen/clusters.json`; set `semantic_search.cluster_path` in config to point at them. The key config knobs are `number_of_bits_for_dense_quantisation`, `n_probes`, `first_pass_sparse_search_top_k`, `window_size`, and `sliding_size`. See `minnal_db/src/semantic_search/Semantic-Search-Architecture.md` for the full design.
