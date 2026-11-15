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
  value-log append, its memtable insert and its in-memory field-index change.
  It is held for memory work only: nothing waits for I/O under it that it can
  avoid (see *Holding the bucket lock* below).
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
holds up writes to another. Reads, scans, GC, TTL and field reindexing take no
stripe. The lock order is **key stripe → WAL → value-log bucket → a field's
change queue**, and nothing waits for a lock while holding one later in that
order. A field index's own lock is taken with none of them held (below).

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
2. **Re-point** (bucket lock, one batch at a time): look a batch of survivors
   up again, and for each key that still points at the copied record, append
   the copy and insert it at the current copy's next system version. A batch is
   at most 1,024 records or 1 MiB of values, and the lock is released between
   batches, so writers to the bucket wait for one batch, not for a segment (up
   to `segment_size_bytes`, 256 MiB by default). Survivors are re-pointed as the
   segment is read, so only one batch of values is in memory.
3. **Flush** the memtable to L0, then **unlink** the old segment.

The copy keeps the record's creation time (`epoch`), so TTL judges a value by
when it was written. A crash before step 3's flush loses the copy, and the
durable layers still point at the old segment, which still exists.

**TTL expiry** (`kv_store/maintenance.rs`, `delete_if_current`) deletes a key
only if it still points at the record judged expired, with a tombstone at that
copy's next system version.

## Field indexes follow the same order

In a namespace with field indexes, a write also moves the key's row in each
field's bitmaps, and those changes must land in the order the key's versions
do: each is a change relative to the key's previous value (take the row out of
the old value's bitmap, add it to the new one's). User writes are ordered by
the stripe, but TTL expiry takes none, so the order is fixed under the bucket
lock: every writer **queues** its change there, with its memtable insert, on
the field's change queue (`kv_store/index_queue.rs`). Applied in any other
order, a TTL tombstone and a put of the same value racing it could change the
index the wrong way round: the put adds the row, then the expiry's removal
takes it out again, and a live document is missing from every query on the
field.

The change is applied after the bucket lock is released, not under it. The
field's own lock can be held for milliseconds (a query merging many values'
bitmaps, the index checkpoint's `msync`), and a writer waiting for it under the
bucket lock would hold up every write to the bucket, GC and TTL with it. After
releasing the bucket lock, the writer takes the field's write lock and applies
every change queued so far, in queue order. Changes leave the queue only under
that lock, so a writer that gets it finds its own change applied, by itself or
by an earlier writer, and a write returns only once its change is in the index.
A spill of a field past the overlay budget's hard limit follows, under the
field's lock only; allocating a new key's row id (which can wait for a row-map
flush) is done before the bucket lock is taken.

A field reindex (`KVStore::reindex_field`: repair, a new index's build) reads
the key's value with no lock and queues its row by compare-and-set: under the
bucket lock, only if the key still points at the record it read; otherwise it
reads again.

## Holding the bucket lock

Every writer to a bucket, GC and TTL wait while one of them holds its lock, so
it is released as early as possible: a holder does memory work and the
lookups its compare-and-set needs, and leaves anything else for after the
lock. GC re-points one batch per hold; a write's field-index changes are
applied after it, and nothing waits for another lock under it that can be held
across I/O. One thing still waits for I/O under it, rarely: sealing a full
segment (one fsync of it and a directory fsync, once per segment).

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
appended to the value log, no garbage is counted and no field index changes.

A record with the same user seq as the stored copy is the same write, already
stored, unless a power loss took its value. A flush fsyncs the value log just
before the logs retire the memtable's records, not before it writes the L0
file (an fsync there costs concurrent writers), and open finds L0 files by
listing their directories. So a power loss between the two leaves an L0 copy
pointing at a record that is gone, or at an offset a later append has reused,
while the log still holds the record. Replay checks the stored copy's record
(whole, its checksum, carrying the write's seq: `ShardedValueLog::holds_record`)
and, if it is not there, **heals** it: it appends the value again and points
the key at it at the stored copy's next system version, as a GC relocation
does. That copy is newer than the broken one and older than any later user
write, and a second replay of the record finds it intact and drops it.

Values no log covers (a no-WAL write's, a GC copy) could not be written again,
so a flush fsyncs those before it writes an L0 file pointing at them
(`ValueLog::mark_unlogged`, `sync_unlogged`).

Both logs keep their own files and replay as they do, the vector index WAL
first; with the check, the order between them does not matter. One merged log
was considered and rejected: every client write would then wait on fsyncs
carrying bulky vector payloads, and it would not cover the no-WAL writes.

The check cannot see tombstones that a merge into L1 dropped. For a logged
delete the per-namespace safe offsets cover that case: a record is replayed
only if no later write of its namespace has been retired, and the delete is a
later record, replayed after it (`WAL-Architecture.md`). A TTL tombstone is in
no log, and the argument is the flush order instead: the tombstone's L0 file
becomes mergeable only after its flush saved the progress its seal recorded,
and that seal's safe offset is past the expired put unless a write appended
before the put was still not applied at the seal — a write whose apply took
longer than the TTL itself.

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
every L0 file stamped at or below it instead of reading it. The header carries
a CRC, and a damaged one fails the open: a raised `merged_through` would delete
live L0 files. An L0 file whose name is not a stamp fails the open too, rather
than being given one. A merge takes every
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
key (`KVStore::note_unapplied_write`) until a user write to it applies. A
`merge` on such a key logs an ERROR naming the unapplied write's seq, reads the
value it can see as the source of truth, and writes over it; its write is newer,
so the unapplied one stays superseded after a crash too. A merge that aborts or
writes nothing leaves the mark, and the next merge reports it again. GC and TTL
never supersede it: their versions keep the older user seq, so if the
unapplied record is replayed it applies over them. It is not always replayed:
the barrier that holds a namespace's safe offset behind a write is released
when the write returns, applied or not, so a later flush can retire it. The
fail log is then the operator's record of it.

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
| `replay_writes_again_a_value_a_power_loss_took_from_under_an_l0_file` | replay drops a record whose stored copy's value a power loss took |
| `an_l0_file_never_points_at_an_unlogged_value_a_power_loss_could_take` | a flush writes an L0 file pointing at a no-WAL value or a GC copy before fsyncing it |
| `ttl_expiry_and_reindexing_racing_puts_keep_the_index_in_step` | an index change is applied out of the order the bucket lock gives, or a reindex sets its row unchecked |
| `a_write_waiting_for_a_field_index_does_not_hold_its_bucket_lock` | a write waits for a field's lock while holding its bucket lock |
| `gc_releases_the_bucket_lock_between_batches_of_survivors` | GC re-points a whole segment under one hold of the lock |
| `a_damaged_merged_through_fails_the_open_and_deletes_nothing` | the L1 header's CRC is not checked |
| `an_aborted_merge_keeps_the_unapplied_marker` | a merge clears the unapplied mark before it writes |

The GC, TTL, failed-apply, open-time deletion, flush-fsync and index-order
tests were checked by reverting each change: each revert fails at least one of
them.

## What it costs

GC and TTL take no lock beyond the bucket lock they always held, and a system
version costs four bytes per memtable node and SSTable entry.
`gc_cost_under_writers` (`db/database/tests/gc_cost.rs`, `#[ignore]`d; run in
release with `TMPDIR` on a real disk, `GC_COST_SEGMENT_MIB` for the segment
size) runs a GC pass over 300,000 keys of 1 KiB values, alone and beside 8
writers doing WAL-backed puts of random keys. Two runs each on NVMe
(2026-11-15), with the re-point batched as described above and, for
comparison, with one hold of the bucket lock per segment:

| 256 MiB segments (the default) | One batch per hold | One segment per hold |
|---|---|---|
| GC alone | 730–735 MiB/s | 817–837 MiB/s |
| GC with writers | 685–686 MiB/s | 764–766 MiB/s |
| Writes completed during GC | 1,667–1,689/s | 1,299–1,324/s |
| Writers during GC: p99 | 11.8–13.5 ms | 26.1–73.9 ms |
| Writers during GC: p99.9 / max | 23.9–29.0 / 23.9–29.0 ms | 190–195 / 193–195 ms |
| Writers without GC: p50 / p99 | 4.6 / 6.8–7.0 ms | 4.6 / 6.8–6.9 ms |

With 4 MiB segments the two are close (writers' p99 during GC 10.1–12.9 ms
batched, 10.2–19.7 ms per segment). Batching costs GC about a tenth of its
throughput, one batched lookup per batch instead of per segment, and takes the
worst write during a pass from the time to rewrite a segment to the time to
rewrite a batch.

A first-hit read and the seq-resolved read it replaced were measured on a tree
with L1, six L0 files of overwrites, a read-only and an active memtable: first
hit was about 1% faster on keys in L0 or only in L1 and on misses, and faster on
keys in a read-only memtable, which then needed no L0 check.
