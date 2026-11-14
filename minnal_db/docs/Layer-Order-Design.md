# Layer order: a newer layer holds a newer version

A key can have copies in several layers of a namespace's LSM tree. The engine
keeps one rule about them: **a copy in a newer layer always carries a higher
write sequence than every copy in an older layer.** This document says what
the rule is, which writers could break it, and how each one keeps it: value-log
GC writes its copies as new versions under the key's lock, replay drops records
older than what is stored, and memtables reach Level 0 in the order they were
frozen.

## Terms used here

- **Sequence (seq)** — the global write counter every write draws from
  (`Database::next_seq`, shared by every namespace's `KVStore`). Seqs are
  unique: two copies of a key at one seq are one write.
- **Version** — one copy of a key: a value pointer or a tombstone, with the seq
  of the write that made it.
- **Layers** — newest first: the **active memtable** (takes every write), the
  **read-only memtables** (sealed, waiting to be flushed), the **Level 0 (L0)
  files** (one per flush per bucket, ordered by a stamp taken at flush), and
  **Level 1 (L1)** (one merged file per bucket). See `README.md` → *LSM Tree*.
- **Key stripe** — one of 1,024 mutexes per namespace (`kv_store/key_locks.rs`);
  a key hashes to one. Every writer holds its key's stripe from drawing its seq
  until its write is in the memtable.
- **Bucket lock** — the value-log bucket's write mutex, held across a write's
  value-log append and its memtable insert.
- **Replay** — re-applying a log's records at open, each at its original seq:
  the main WAL's and the vector index WAL's records from each namespace's
  **safe offset**, the position below which its records are known to be in L0
  files.
- **No-WAL write** — a write recorded in no log (`put_no_wal`, `delete_no_wal`,
  `merge_no_wal`, TTL expiry); it is durable only once its memtable is flushed.

## The rule and why it matters

If every newer layer holds newer versions, the first layer that has a key holds
its current version. If not, an older version in a newer layer can sit above a
newer one below it: a value above its own later delete, which brings the key
back, or a tombstone above a later value, which loses it.

The engine resolves every read, scan and merge by highest seq across all
layers (`point_read.rs`, `fold_newest`, `merge_level0_to_level1`), which gives
the right answer even where the rule fails. The rule is what makes that
resolution simple to reason about, and what lets a read stop at the first layer
that has the key.

A write lands in the active memtable, and memtables move down one layer at a
time, so the rule holds as long as **every write applied to the memtable is
newer than every stored copy of its key**, and **memtables move down in the
order they were frozen**. Four kinds of writer apply versions; each keeps the
rule as follows.

| Writer | Seq it carries | Why it is the newest |
|---|---|---|
| Live writes (put, delete, merge, logged and no-WAL writes, TTL expiry) | drawn at write time | drawn under the key's stripe, after the previous write to the key was applied |
| Value-log GC's re-point | drawn at re-point time | drawn under the key's stripe, like a live write |
| Replay | the record's original seq | checked against every stored copy; dropped unless newer |
| A memtable's flush to L0 | (moves versions, writes none) | memtables reach L0 in the order they were frozen |

## Key stripes, per namespace

Every writer takes its key's stripe before it draws its seq and releases it
after the write is in the memtable. While a writer holds the stripe, no other
write to the key is between drawing a seq and applying it, so a seq drawn under
the stripe is newer than every copy of the key already stored, and older than
every write to it still to come.

Each namespace's `KVStore` owns its own 1,024 stripes (`KVStore::lock_key`),
about 1 KiB per namespace. Only writes to the same namespace can wait on each
other: GC, a TTL pass or a slow `merge` closure in one namespace never holds up
writes to another. Reads and scans take no stripe.

The lock order is **key stripe → WAL → value-log bucket**. An uncontended
acquire is one atomic compare-and-swap (tens of nanoseconds), against a WAL
fsync of about 0.7 ms on every WAL-backed write.

## Value-log GC writes a new version

GC rewrites a sealed segment's live records into the active tail and points
their keys at the copies (`kv_store/gc.rs`):

1. **Find the survivors** (no lock): read the segment's keys, look them all up
   in the LSM in one batch, keep each record the LSM still points at.
2. **Re-point** (bucket lock): look the survivors up again in one batch. For
   each key that still points at the copied record, **try** its stripe:
   - free: draw a fresh seq, append the copy to the tail, insert it into the
     memtable at that seq;
   - busy: defer the key.
3. **Deferred keys**: for each, take its stripe, then the bucket lock, check
   the key still points at the copied record, and re-point it as above.
4. **Flush** the memtable to L0, then **unlink** the old segment.

Why a fresh seq, under the stripe: the copy goes into the active memtable, the
newest layer, so it has to be the newest version. A seq drawn with the stripe
held is. Without the stripe, a writer that drew a lower seq before GC and is
waiting for the bucket lock would apply after GC's copy and lose to it, though
its write is newer.

Why *try* the stripe: step 2 holds the bucket lock, which comes after the
stripe in the lock order. A writer holding the stripe may be waiting for that
bucket lock, so waiting for the stripe there would deadlock. A stripe is busy
for one write at most (its WAL fsync included), and the deferred path takes the two locks
in order. Step 2 keeps the batched lookup, so GC's cost stays one lookup per
segment rather than one per record (`gc_resolves_liveness_in_batches_not_per_record`).

The copy keeps the record's creation time (`epoch`), so TTL judges a value by
when it was written, not by when GC moved it. Readers are unchanged: a reader
holding the old pointer either reads the old record (the same bytes) or finds
the segment gone (`SegmentMissing`) and looks the key up again.

Crash safety is unchanged: the copy is not in any log, so the memtable is
flushed before the old segment is unlinked. A crash before the flush loses the
copy; the durable layers still point at the old segment, which still exists.

## Replay drops what is not newer

Replay is the one path that applies a write at an old seq. A record past its
namespace's safe offset can be older than a copy of its key that is already on
disk:

- **The safe offset is a lower bound.** A memtable's seal records how far its
  writes reach, lowered to the oldest write still in flight, so records past
  the offset may already be in L0 at the same seq.
- **The other log.** A key can be written through both logs (a vector chunk:
  deleted through the main WAL, re-written through the vector index WAL). A
  newer write in one log can be flushed, and not replayed, while an older one
  in the other log replays.
- **No-WAL writes** are in no log. A TTL delete or a `merge_no_wal` at seq 170
  can be in L0 while a logged put of the same key at seq 160 replays.

So `replay_upsert` and `replay_delete` check, under the bucket lock, whether
the record is strictly newer than every stored copy of its key, tombstones
included (`LSMTree::current_pointer_and_wins`). If not, the record is dropped:
nothing is appended to the value log, no garbage is counted and no field index
changes. An equal seq is the same write already stored.

Both logs keep their own files and replay as they do: the vector index WAL
first, then the main WAL. With the check, the order between them does not
matter. One merged log was considered and rejected: every client write would
then wait on fsyncs carrying bulky vector payloads, embedding traffic would
queue on the main WAL's lock, and the main WAL's retention would hold those
payloads. It would also not cover the no-WAL writes.

The check cannot see tombstones that a merge into L1 dropped. The per-namespace
safe offsets cover that case: a record is replayed only if no
later write of its namespace has been retired (`WAL-Architecture.md`).

## Memtables reach L0 in order

A read-only memtable's records move to L0 when it is flushed, and its files get
a stamp that orders them among the bucket's L0 files. If a newer memtable were
flushed first, its files would be older by stamp and, until the older memtable
was flushed, the older memtable would sit above them.

Every flush goes through `LSMTree::flush_read_only_memtables`, which flushes
oldest first and stops at the first failure; a second flusher of the same
memtable waits for the first. `flush_ro_memtable_to_level0` refuses a memtable
while an older one is still unflushed (`LSMError::FlushOutOfOrder`), before
anything is written, so a new caller that picks one memtable out of order fails
loudly instead of breaking the rule.

## Tests

`LSMTree::layer_order_violations` (test-only) walks every layer and reports
each key with a copy that is not older than the copy above it.

| Test | Fails if |
|---|---|
| `gc_writes_its_copy_as_the_newest_version` | GC writes its copy at the record's old seq |
| `a_write_that_drew_its_seq_before_gc_still_wins` | GC re-points without the key's stripe |
| `a_replayed_record_not_newer_than_a_stored_copy_is_dropped` | replay applies a record older than, or equal to, a stored copy |
| `the_layer_order_holds_under_every_writer` | any of the above (random puts, deletes, replays, GC, flushes and merges) |
| `a_newer_memtable_cannot_reach_level0_before_an_older_one` | a newer memtable can be flushed before an older one |
| `a_held_stripe_does_not_block_writes_to_another_namespace` | stripes are shared between namespaces |

The first four were checked by reverting each change: each revert fails at
least one of them.

## Reads still resolve by sequence

With the rule kept by every writer, a read could stop at the first layer that
holds the key. That was measured against the seq-resolved read and not adopted:
on a tree with L1, six L0 files of overwrites, a read-only and an active
memtable, stopping at the first hit saved about 1% on keys whose newest copy is
in L0 or only in L1, and on misses. It saved more only on keys in a read-only
memtable, which exists only until it is flushed. The seq-resolved read already
skips almost every lower layer by its sequence bounds (each L0 file's and L1's
newest seq), checked before the file's bloom filter. Keeping it means a writer
that ever broke the rule would still be read correctly.

The measurement is `first_hit_against_seq_resolution`
(`storage/lsm/tree/tests/first_hit.rs`, `#[ignore]`d; run it in release).
Scans and the L0→L1 merge compare one seq per entry they already read, so
resolving them by layer would save no I/O.

GC's cost with the stripe has still to be measured: GC throughput, and writers'
p99 latency during a GC pass.
