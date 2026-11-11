# System catalog: the namespace registry in a namespace

The engine keeps its list of namespaces — each one's id, TTL settings and
field indexes — inside the database itself, in a reserved namespace called
`system_catalog`. There is no registry file and no per-namespace config file.
This document says what the catalog holds, how ids are handed out, and how
opening and recovery use it.

## Terms used here

- **Namespace** — one keyspace of the engine: its own LSM tree, value log
  and field indexes, in `{db_path}/ns_{name}/`. Doc stores and KV stores are
  namespaces, and so are their internal companions (`{ns}_sparse_vector`,
  `{ns}_ivf_postings`, …) and the `system_*` namespaces.
- **`NamespaceId`** — a namespace's 64-bit id. Every main-WAL record and
  every vector-index-WAL record names its namespace by this id (8 bytes).
- **`FieldId`** — the 64-bit id of one field index of a namespace. It names
  the index's directory, `{db_path}/field-index/{ns_id}/{field_id}/`.
- **Registry** — the engine's in-memory copy of the catalog
  (`NamespaceRegistry`, `src/namespace/registry.rs`). Every lookup reads it;
  nothing reads the catalog's files on a request.

## The catalog namespace

| Namespace | Id | Holds |
|---|---|---|
| `default` | 0 | the default keyspace |
| `system` | 1 | system-wide stores such as the query-embedding cache |
| `system_catalog` | 2 | every namespace's record, every field index's record, the id counter |

All three always exist. `system_catalog`'s name (so its directory,
`ns_system_catalog/`) and its id are compiled in, so it is opened before
anything else is read. It cannot be removed, has no TTL and no field
indexes, and is written only by the registry, always WAL-backed. The doc
store refuses store names starting with `system_`, so no REST call can
reach it.

### Keys and values

| Key | Value |
|---|---|
| `next_id` | the next id to hand out (u64, little-endian) |
| `ns/{name}` | the namespace's id and TTL settings (lifetime, deletes per pass) |
| `field/{ns_id}/{field_name}` | the field's id, value type and `dropped` flag |

Record values start with a version byte, then an rkyv payload read with
validation. `ns_id` in a field key is 8 big-endian bytes, so one namespace's
fields are one prefix scan. The record layout lives in one module,
`src/namespace/catalog.rs`.

## Ids: one counter

Namespace ids and field ids come from one counter, `next_id`, starting at
1024; ids below it are reserved for namespaces with fixed ids. Creating a
namespace or a new field writes the counter's new value **first**, then the
record that uses the id. A crash between the two leaves a gap in the ids,
never an id handed out twice. So a name dropped and created again gets a new
id, and the old namespace's WAL records, field-index files and caches can
never be taken for the new one's.

A field that is dropped keeps its record, marked `dropped`, and a later
re-add reuses its id: the doc store keys a field's job records by its id, so
a re-added field carries on from its own records. The flag is saved before
the field's files are deleted, so a drop a crash cut short is finished at the
next open (`complete_interrupted_field_drops`).

## Changing the registry

Every change goes through `Database::update_registry`:

1. take the registry's write lock;
2. make the change on a copy, which records the catalog writes it needs;
3. write them, in order, each WAL-backed and durable on return;
4. swap the copy in, release the lock;
5. flush the catalog's memtable.

A change that fails, or whose writes fail, leaves the registry as it was, so
the registry is always what the catalog holds. Holding the lock across the
writes keeps changes in the order they are saved; the catalog's write never
reads the registry (its store is always open), so it cannot wait on the lock
it is called under.

Step 5 keeps the catalog from holding back the main WAL. Its writes are few
and its memtable would rarely fill, so without a flush they would keep their
WAL segments until WAL GC flushed the catalog, and every open would replay
them. Changes are rare (a create, a drop, a field or TTL change), so one small
flush each is cheap. It is best-effort: the writes are durable in the WAL
already.

## Opening and recovery

Recovery replays the main WAL into every namespace it knows and skips the
records of namespaces it does not. With the namespace list inside a
namespace, nothing is known until the catalog's own records are replayed, so
opening (`Database::finish_open`) goes:

1. Open the main WAL and its metadata.
2. Open the catalog's store (fixed name and id).
3. **Replay the catalog's records** of the main WAL, at their original seqs
   (`recover_catalog_from_wal`).
4. Load the registry from the catalog, and open every other namespace's
   store with its saved TTL.
5. Release vector-index-WAL records of namespaces no longer listed, finish
   interrupted field drops, note no-WAL gaps.
6. Recover both WALs into the other namespaces as before; the main-WAL pass
   skips catalog records (already replayed).
7. Wire the flush observers, then write what loading left pending: a new
   database's first records (`next_id`, `default`, `system`), and the
   deletes of records loading left out (a namespace whose name cannot be a
   directory, a field of a namespace that no longer exists).

Step 3 reads the main WAL a second time. The part that needs replay is
bounded by flushes and by the WAL GC backstop, and the second read mostly
comes from the page cache, so the cost is small and paid only at open.

Two orderings are load-bearing:

- **Step 6 counts the catalog's replayed records.** Recovery flushes every
  store and then retires the whole log only if it replayed something; a
  reopen that replayed catalog records alone must flush them too, or a
  second crash would lose them (`replaying_only_catalog_records_flushes_them_before_the_log_is_retired`).
- **Step 7 comes after recovery and the observers.** Recovery's retirement is
  right only while every write it covers was replayed and flushed; a write
  made before it, and not replayed, would be retired from the WAL while still
  only in a memtable.

Within the WAL, a namespace's catalog record always precedes its own records:
the record is durable before the namespace's store opens.

## What stays in files

Only configuration moved. The engine's own state stays in files, because the
namespaces need it to open:

- the main WAL's segments and metadata, and the vector index WAL's segments and
  `retired_cut`;
- each LSM tree's SSTables and each value log's segments and metadata;
- each field index's `checkpoint` and `gap.json`, and the `no_wal_pending`
  marker;
- `fail_logs/`.

The API server's TOML config file is unchanged.

## Not changed

- The doc store's `system_schema` keeps the doc store's own record of each
  store (field declarations, vector settings, lifecycle state). The catalog
  is the engine's record of ids and field indexes; each layer owns its own.

## Compatibility

A database written before the catalog (with a `namespace_registry` file and
128-bit ids) does not open and must be recreated. There is no migration.

## Where the code is

| Path | What |
|---|---|
| `src/namespace/id.rs` | `NamespaceId`, the fixed ids, `FIRST_ALLOCATED` |
| `src/namespace/catalog.rs` | The catalog's keys and record encoding |
| `src/namespace/registry.rs` | `NamespaceRegistry`: the in-memory registry and its changes |
| `src/namespace/schema.rs` | `NamespaceSchema`: one namespace's field indexes |
| `src/db/namespace/catalog.rs` | Opening the catalog, loading the registry, `update_registry` |
| `src/db/wal/recovery.rs` | `recover_catalog_from_wal`, then `recover_from_wal` |
| `src/db/database/tests/catalog.rs` | Crash, flush and record tests |
