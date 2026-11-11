# System catalog: namespace and field registry in a namespace

**Status:** proposed, not implemented.

This document describes how the engine keeps its list of namespaces and
their field-index ids inside the database itself — in a reserved namespace,
`system_catalog` — instead of in files beside it.

## Terms used here

- **Namespace** — one keyspace of the engine: its own LSM tree, value log
  and field indexes, in `{db_path}/ns_{name}/`. Doc stores and KV stores are
  namespaces, and so are their internal companions (`{ns}_sparse_vector`,
  `{ns}_ivf_postings`, …) and the `system_*` namespaces.
- **`NamespaceId`** — a namespace's 128-bit id (a version-4 UUID), drawn at
  random when it is created and never reused. Every main-WAL and vector-index-WAL
  record names the namespace it belongs to by this id.
- **`FieldId`** — the id of one field index of a namespace. It names the
  index's directory, `{db_path}/field-index/{ns_id}/{field_id}/`.
- **Catalog** — the engine's record of which namespaces exist (name ↔ id, TTL)
  and which field indexes each has (field name ↔ field id, value type).

## What the catalog holds today, and where

| File | Format | Holds |
|---|---|---|
| `{db_path}/namespace_registry` | binary: magic `NSRG`, version, CRC32, rkyv payload | every namespace's name ↔ id; TTL settings (lifetime, deletes per pass) of each TTL namespace |
| `{db_path}/ns_{name}/config.json` | JSON, one per namespace that has field indexes | each field's id, name, value type and a `dropped` flag; the next field id to hand out |

Both are written with an atomic, fsynced tmp-then-rename and read once, at
open, into `NamespaceRegistry` (`src/namespace/registry.rs`), which serves every
lookup from memory.

This works, but it is a second storage mechanism beside the engine's own:

- Two durable formats to maintain, each with its own checksum and versioning,
  and neither covered by the WAL's recovery or fail-log tooling.
- Field ids are a per-namespace counter (`u32`). A dropped field keeps its
  definition, marked `dropped`, so a later re-add can reuse the id — the flag is
  also what lets an interrupted drop finish at the next open. That reuse rule
  and the flag exist only because ids are counters.
- The doc store already keeps its schemas in a namespace (`system_schema`,
  `doc_store/schema/store.rs`); the engine is the one layer still using files.

## The design

### A reserved namespace with a compiled-in id

The engine always has a namespace named `system_catalog` whose id is a constant
in the source:

```rust
impl NamespaceId {
    pub const DEFAULT: Self = Self(0);
    pub const SYSTEM: Self = Self(1);
    /// The catalog's fixed id: known before anything else is read.
    pub const CATALOG: Self = Self(2);
}
```

Opening a database opens `system_catalog` first, creating it if absent —
the same for an embedded `AsyncDb`, an embedded doc store and the API server,
since all three open through the engine. Because both its name (which fixes its
data directory, `ns_system_catalog/`) and its id are known in advance, it can be
opened without reading anything else.

It is a separate namespace, not keys inside `system`: `system` holds the
query-embedding cache, which is written without the WAL and expires by TTL. The
catalog must never be either. Its rules:

- every write is WAL-backed (`put` / `delete`), durable when it returns;
- no TTL, no no-WAL writes, no field indexes, no vector index;
- it cannot be removed, and it does not list itself;
- the name falls under `system_*`, which `validate_store_name` already reserves.

### Keys and values

| Key | Value (rkyv) |
|---|---|
| `ns/{name}` | `{ id: NamespaceId, ttl: Option<{ ttl_millis, max_deletes }> }` |
| `field/{ns_id}/{field_name}` | `{ field_id: FieldId, value_type }` |

`ns_id` is the 16-byte big-endian id, so all fields of one namespace share a
prefix and are listed with one prefix scan. `default` and `system` are written
like any other namespace, with their fixed ids.

### In memory, written through

`NamespaceRegistry` keeps its maps (name ↔ id, schemas, TTL) and is loaded from
a scan of `system_catalog` at open instead of from the files. Every change
follows the rule `SchemaStore` uses for `system_schema`:

1. take the registry's writer lock;
2. write the catalog key (WAL-backed, fsynced on return);
3. update the in-memory map.

So the map is always what is on disk, in write order, and reads stay
lock-and-lookup with no I/O. The writer lock is the existing `RwLock` around
the registry; a catalog write is one WAL commit, which the group commit shares
with concurrent writers.

### Field ids become UUIDs

`FieldId` becomes a 128-bit random id, drawn like a `NamespaceId`. On disk it
appears only in directory names, so the change costs no space in any record.
With ids never reused:

- a re-added field gets a **new** id and an empty directory; there is no reuse
  rule and no `next_field_id` counter;
- dropping a field is: delete its catalog key, then delete its directory. There
  is no `dropped` flag. A crash between the two leaves a directory the catalog
  does not list, which the open-time sweep (below) removes, in place of
  `complete_interrupted_field_drops`;
- the checkpoint worker's field list is simply the catalog's fields; nothing
  needs filtering out.

`FieldId::MAX`, which the doc store's job registry uses as a "whole namespace"
sentinel in `JobKey::field_build`, becomes a named constant of the new type.

### Opening and recovery

Recovery replays the main WAL into every known namespace and skips records of
namespaces it does not know (`db/wal/recovery.rs`). With the catalog inside a
namespace, nothing is known until the catalog's own records have been
replayed. Open becomes:

1. Open the main WAL and its metadata (unchanged).
2. Open `system_catalog` (fixed name and id).
3. **Replay the catalog's records only**: scan the WAL from its replay point
   and apply the records whose namespace is `NamespaceId::CATALOG`.
4. Load `NamespaceRegistry` from the catalog.
5. Open every other namespace's store, then replay the rest of the WAL as today
   (catalog records already applied are skipped by their namespace).
6. Sweep: remove every `ns_*` directory and every `field-index/{ns_id}` or
   `field-index/{ns_id}/{field_id}` directory the catalog does not list. This
   finishes a drop of a namespace or a field that a crash cut short.

Step 3 reads the WAL a second time. The WAL that needs replay is bounded by
flushes and by the WAL GC backstop (`thresholds.max_pinned_wal_segments`), and
the second read mostly comes from the page cache, so the cost is small and
paid only at open. The alternative — one pass that holds every other
namespace's records in memory until the catalog is loaded — costs memory in
proportion to the WAL and is not simpler.

Ordering within the WAL needs nothing new: a namespace's catalog `put` commits
before any write to that namespace can be issued, so it is always earlier in
the log than the namespace's own records.

### Create and remove

Creating a namespace keeps its current order, with the file write replaced by
a catalog `put`: write `ns/{name}` (durable), then open the store. A failed open
deletes the key again, as it removes the registry entry today.

Removing a namespace: delete `ns/{name}` and its `field/{ns_id}/…` keys
(durable point — recovery now skips its records), close the store, forget its
WAL progress, delete its directories. A crash after the first delete leaves
directories that step 6 of open removes.

The removes are several single-key deletes, not one transaction; the
`ns/{name}` delete goes first, so that once it is durable the namespace is
gone, and leftover `field/` keys of an unknown namespace are deleted by the
sweep at open.

### Layout

`layout.rs` gains nothing and loses nothing: the catalog is an ordinary
namespace directory. The `namespace_registry` file and every `config.json` are
gone, so `registry.rs` no longer builds file paths.

## What stays in files

Only configuration moves. The engine's own state files stay, because the
namespaces depend on them to open and so cannot live inside one:

- the main WAL's segments and metadata, and the vector index WAL's segments and
  `retired_cut`;
- each LSM tree's manifest and each value log's segments;
- each field index's `checkpoint` and `gap.json`, and the `no_wal_pending`
  marker;
- `fail_logs/`.

The server's TOML config file is unchanged.

## Not changed

- The doc store's `system_schema` keeps the doc store's own record of each
  store (field declarations, vector settings, lifecycle state). The catalog is
  the engine's record of field ids; each layer owns its own, as now.
- `NamespaceId` stays random and never reused.

## Compatibility

The project is greenfield: an existing database, with its `namespace_registry`
file, will not open after this change and must be rebuilt. No migration tool.

## Work, in order

1. `FieldId` to a 128-bit random id; drop the `dropped` flag and reuse rule;
   add the open-time sweep for field directories. (Self-contained; still on
   `config.json`.)
2. `system_catalog` namespace, `NamespaceId::CATALOG`, catalog keys and the
   write-through registry; delete `namespace_registry` and `config.json`.
3. Two-step recovery and the namespace-directory sweep.
4. Tests: crash between each pair of steps in create, remove, field add and
   field drop (`crash()` simulation), recovery with catalog records interleaved
   with data records, a re-added field getting a new id, and the sweep removing
   only what the catalog does not list.
5. Docs: `WAL-Architecture.md` (recovery order), `Namespace-Lifecycle.md`,
   the layout comment in `layout.rs`, and the agent notes.
