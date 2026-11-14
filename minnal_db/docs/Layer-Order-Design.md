# Layer order: reads stop at the first copy

A key can have copies in several layers of a namespace's LSM tree. Every read,
scan and merge takes the copy in the **newest layer that holds the key** and
ignores the rest. That is correct because every writer keeps one rule: **a copy
in a newer layer is a newer version than every copy below it.** This document
defines a copy's version, says how each writer keeps the rule, and what a crash
can leave behind.

## Terms used here

- **Layers** — newest first: the **active memtable** (takes every write), the
  **read-only memtables** (sealed, waiting to be flushed), the **Level 0 (L0)
  files** (one per flush per bucket, ordered by a stamp taken at flush), and
  **Level 1 (L1)** (one merged file per bucket). See `README.md` → *LSM Tree*.
- **User seq** — the global write counter (`Database::next_seq`, shared by every
  namespace's `KVStore`). Every user write — put, delete, `merge`, the logged and
  no-WAL writes — draws a fresh one.
- **System action** — a write the engine makes on its own, which changes no
  value a user wrote: value-log GC moving a record to a new segment, and TTL
  expiry deleting an expired one.
- **System seq** — how many system actions have rewritten one user write.
- **Version** — a copy's `(user seq, system seq)`, compared user seq first
  (`storage/lsm/version.rs`). A user write is at system seq 0. A system action
  writes its copy at the version it replaces with the system seq raised by one,
  so the copy is newer than what it replaces and older than every later user
  write. Versions are unique: two copies at one version are one write.
- **Key stripe** — one of 1,024 mutexes per namespace (`kv_store/key_locks.rs`);
  a key hashes to one. Every user write holds its key's stripe from drawing its
  user seq until the write is in the memtable.
- **Bucket lock** — the value-log bucket's write mutex, held across a write's
  value-log append and its memtable insert.
- **Replay** — re-applying a log's records at open, each at its original user
  seq: the main WAL's and the vector index WAL's records from each namespace's
  **safe offset**, the position below which its records are known to be in L0
  files.

## The rule

A write lands in the active memtable, and memtables move down one layer at a
time, so the rule holds as long as **every write applied to the memtable is
newer than every stored copy of its key**, and **memtables move down in the
order they were frozen**. Each writer keeps it as follows.

| Writer | Version it writes | Why it is newer than every stored copy |
|---|---|---|
| User writes | a fresh user seq | drawn under the key's stripe, after the previous write to the key was applied |
| Value-log GC's re-point | the current copy's next system version | decided under the bucket lock, from the current copy |
| TTL expiry | the expired copy's next system version | decided under the bucket lock, from the current copy |
| Replay | the record's original user seq | checked against the current copy; dropped unless newer |
| A memtable's flush to L0 | (moves copies, writes none) | memtables reach L0 in the order they were frozen |

With the rule held, the reads are simple. A point read checks the active
memtable, the read-only memtables newest first, the bucket's L0 files newest
first, then L1, and returns the first copy it finds (`point_read.rs`); a
tombstone found first means the key is deleted. A batch read does the same per
key (`batch_read.rs`). A scan folds its layers oldest first, each layer's copy
replacing the older one (`fold_newest`). The L0→L1 merge keeps the newer file's
copy and drops a tombstone at the bottom (`compaction.rs`).

## User writes and the key stripe

Every user write takes its key's stripe before it draws its user seq and
releases it after the write is in the memtable. While a writer holds the
stripe, no other user write to the key is between drawing a seq and applying
it, so a seq drawn under the stripe is newer than every copy of the key already
stored. It is also what makes `merge` (read, decide, write) indivisible for one
key.

Each namespace's `KVStore` owns its own 1,024 stripes (`KVStore::lock_key`),
about 1 KiB per namespace, so a slow `merge` closure in one namespace never
holds up writes to another. Reads, scans, GC and TTL take no stripe. The lock
order is **key stripe → WAL → value-log bucket**, and nothing waits for a
stripe while holding either of the others.

## System actions: GC and TTL expiry

A system action keeps the user seq of the copy it replaces and raises its system
seq. Under the bucket lock it reads the key's current copy, checks it is still
the record it acted on (a compare-and-set), and writes its own copy one system
version above it. That copy is newer than the one it replaces, so it can go into
the active memtable. It is older than every later user write: a writer that drew
its user seq before the system action, and is waiting for the bucket lock, has a
higher user seq and wins whenever it lands — live, and after a crash too. So GC
and TTL need no key stripe.

**Value-log GC** (`kv_store/gc.rs`) rewrites a sealed segment's live records
into the active tail:

1. **Find the survivors** (no lock): read the segment's keys, look them all up
   in one batch, keep each record the LSM still points at.
2. **Re-point** (bucket lock): look the survivors up again in one batch, and for
   each key that still points at the copied record, append the copy and insert
   it at the current copy's next system version.
3. **Flush** the memtable to L0, then **unlink** the old segment.

The copy keeps the record's creation time (`epoch`), so TTL judges a value by
when it was written. A crash before step 3's flush loses the copy, and the
durable layers still point at the old segment, which still exists.

**TTL expiry** (`kv_store/maintenance.rs`, `delete_if_current`) deletes a key
only if it still points at the record judged expired, with a tombstone at that
copy's next system version.

A version's system seq is a `u32`: about four billion system actions on one
unchanged write. A copy that has used them all is re-pointed or expired as a user
write instead — under its stripe, at a fresh user seq.

## Replay drops what is not newer

Replay is the one path that applies a write at an old user seq. A record past its
namespace's safe offset can be older than a copy of its key already on disk:

- **The safe offset is a lower bound.** A memtable's seal records how far its
  writes reach, lowered to the oldest write still in flight, so records past
  the offset may already be in L0.
- **The other log.** A key can be written through both logs (a vector chunk:
  deleted through the main WAL, re-written through the vector index WAL). A
  newer write in one log can be flushed, and not replayed, while an older one
  in the other log replays.
- **No-WAL writes** are in no log. A `merge_no_wal` at user seq 170 can be in
  L0 while a logged put of the same key at user seq 160 replays.

So `replay_upsert` and `replay_delete` check, under the bucket lock, whether the
record is strictly newer than the key's current copy, tombstones included
(`LSMTree::current_pointer_and_wins`). If not, the record is dropped: nothing is
appended to the value log, no garbage is counted and no field index changes. An
equal version is the same write, already stored.

Both logs keep their own files and replay as they do, the vector index WAL
first; with the check, the order between them does not matter. One merged log
was considered and rejected: every client write would then wait on fsyncs
carrying bulky vector payloads, and it would not cover the no-WAL writes.

The check cannot see tombstones that a merge into L1 dropped. The per-namespace
safe offsets cover that case: a record is replayed only if no later write of its
namespace has been retired (`WAL-Architecture.md`).

## Memtables reach L0 in order

A read-only memtable's records move to L0 when it is flushed, and its files get
a stamp that orders them among the bucket's L0 files. Every flush goes through
`LSMTree::flush_read_only_memtables`, which flushes oldest first and stops at
the first failure; a second flusher of the same memtable waits for the first.
`flush_ro_memtable_to_level0` refuses a memtable while an older one is still
unflushed (`LSMError::FlushOutOfOrder`), before anything is written.

## What a crash can leave behind

After a merge into L1 the merged L0 files are deleted, and a crash can come
first. Read again at open, a merged file would sit above the L1 it was merged
into, and an older one could hold an older copy of a key than L1 does (L1 having
taken a newer copy from a later merge). So each L1 file records in its header
the newest L0 stamp merged into it (`merged_through`), and the open deletes
every L0 file stamped at or below it instead of reading it. A merge takes every
file registered when it starts, and files register in stamp order, so those are
exactly the merged files. A merge that leaves L1 empty writes a header-only
file, so the stamp is kept. The open also starts new stamps above it, so a clock
that steps backwards cannot name a new file as if it were merged.

The merged files are also deleted oldest first, each unlink made durable before
the next, stopping at the first one a reader still holds
(`cleanup_obsolete_level0_files`), so whatever survives is the newest of them: a
second line of defence.

## A write that never applied

A WAL-backed put or delete whose in-memory apply fails on every retry is durable
but not readable: reads still see the value before it. The store records the
key (`KVStore::note_unapplied_write`) until the next user write to it. A `merge`
on such a key logs an ERROR naming the unapplied write's seq, reads the value it
can see as the source of truth, and writes over it; its write is newer, so the
unapplied one stays superseded after a crash too. GC and TTL never supersede it:
their versions keep the older user seq, so replay applies the unapplied write
over them.

## Tests

`LSMTree::layer_order_violations` (test-only) walks every layer and reports each
key with a copy that is not older than the copy above it. The differential tests compare the reads, which
stop at the first copy, with `reference_get`, which reads every layer and keeps
the highest version.

| Test | Fails if |
|---|---|
| `gc_writes_its_copy_as_the_newest_version` | GC writes its copy at the version it replaces |
| `a_write_that_drew_its_seq_before_gc_still_wins` | GC writes its copy at a fresh user seq |
| `a_write_that_drew_its_seq_before_ttl_expiry_still_wins` | TTL's tombstone takes a fresh user seq |
| `a_replayed_record_not_newer_than_a_stored_copy_is_dropped` | replay applies a record older than, or equal to, the current copy |
| `the_layer_order_holds_under_every_writer` | any of the above (random puts, deletes, replays, GC, flushes and merges) |
| `test_read_shortcuts_agree_with_brute_force_resolution` | a point or batch read disagrees with the brute-force read |
| `test_scans_agree_with_brute_force_resolution` | a scan disagrees with the brute-force read |
| `test_get_multiple_takes_the_newer_layers_copy_at_every_boundary` | a batch read takes an older layer's copy at some layer boundary |
| `a_newer_memtable_cannot_reach_level0_before_an_older_one` | a newer memtable can be flushed before an older one |
| `a_crash_while_merged_level0_files_are_deleted_does_not_bring_a_key_back` | a merged L0 file a crash left is read again |
| `merged_level0_files_left_by_a_crash_are_deleted_at_open` | the open reads merged L0 files again, or a new stamp falls at or below a merged one |
| `an_empty_merge_still_records_what_it_merged` | a merge that empties L1 loses its `merged_through` |
| `a_merge_over_a_put_that_never_applied_keeps_the_value_it_read` | a failed apply is not recorded, or replay brings it back over the merge |
| `a_held_stripe_does_not_block_writes_to_another_namespace` | stripes are shared between namespaces |

The GC, TTL, failed-apply and open-time deletion tests were checked by
reverting each change: each revert fails at least one of them.

## What it costs

GC and TTL take no lock beyond the bucket lock they always held, and a system
version costs four bytes per memtable node and SSTable entry.
`gc_cost_under_writers` (`db/database/tests/gc_cost.rs`, `#[ignore]`d; run in
release with `TMPDIR` on a real disk) runs a GC pass over 300,000 keys of 1 KiB
values in 4 MiB segments, alone and beside 8 writers doing WAL-backed puts of
random keys. Three runs on NVMe:

| | Result |
|---|---|
| GC alone, 720 MiB reclaimed | 497–500 MiB/s |
| GC with writers, 632 MiB reclaimed | 457–467 MiB/s |
| Writers during GC: p50 / p99 | 4.7 / 11.5–13.0 ms |
| Writers during GC: p99.9 / max | 16.2–19.0 / 18.6–20.4 ms |
| Writers without GC: p50 / p99 | 4.6 / 6.9–7.0 ms |

A first-hit read and the seq-resolved read it replaced were measured on a tree
with L1, six L0 files of overwrites, a read-only and an active memtable: first
hit was about 1% faster on keys in L0 or only in L1 and on misses, and faster on
keys in a read-only memtable, which then needed no L0 check.
