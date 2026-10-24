# Vector index WAL — design note

**Status:** agreed 2026-10-15 (decisions at the end). Built in the steps of §9.
**Scope:** how the vector worker makes a document's vector writes durable before
it removes the document's queue entry. Nothing here changes what search reads,
the main WAL's write path, or the doc store's API.

## Terms used here

The same terms, for both logs, are in `minnal_db/README.md` → *WAL terms*.

**Writes and logs**

- **Main WAL** — the database's write-ahead log (`db/wal.rs`). Every ordinary
  write appends one entry and fsyncs it before the call returns, under one lock
  for the whole database.
- **Vector index WAL** — the log this note designs: one per database, in
  `<db_path>/vector_index_wal/`, shared by every namespace.
- **Logged write** — a write recorded in the vector index WAL instead of the
  main WAL: `Database::put_ns_logged` / `delete_ns_logged`, called through a
  namespace as `put_logged` / `delete_logged`. Appended and applied with no
  fsync; durable at the next **sync** (`sync_vector_index_wal`), one fsync for
  everything appended before it. Concurrent syncs share one fsync, through the
  same commit rounds as the main WAL (`minnal_db/docs/Main-WAL-Rework-Design.md`
  §16).
- **No-WAL write** — `put_no_wal` (and `delete_no_wal`, `merge_no_wal`): the
  value goes to the value log and the key's pointer goes to the memtable, with
  no log entry at all. The pointer exists only in memory until the memtable is
  flushed to a level-0 (L0) file.
- **Sequence (seq)** — the global write counter every write draws from,
  whichever log it uses; the highest seq for a key wins everywhere (memtable,
  L0, L1, replay).
- **Key stripe lock** (`db/key_locks.rs`) — 1,024 locks, each key hashed to one
  (its *stripe*). Every write takes its key's stripe before anything else and
  holds it until the write is applied, so one key's writes apply in seq order
  whichever log they use.
- **Barrier** — a point the worker (or maintenance) must not pass until its
  earlier writes are durable; here, one sync of this log.

**Memtables and flushes**

- **Memtable** — a namespace's in-memory sorted buffer of recent writes.
  **Seal:** turning a full (or force-flushed) memtable read-only, ready to
  flush. **Flush:** writing a sealed memtable to L0 files and fsyncing them.
- **L0 / L1** — the LSM tree's on-disk levels. **Compaction** merges L0 files
  into L1, and drops tombstones (the records deletes leave) there.
- **Flush observer** (`LsmFlushObserver`) — how a namespace's LSM tree reports
  each flush, in order: **sealed** (under the memtable's write lock),
  **flushing** (before any L0 file is written), **written** (L0 files durable,
  not yet registered, so nothing can merge them) and **flushed** (registered).

**The log's own positions** (§5)

- **Segment** — one file of the log, `{segment id}.log`; a full one is
  **sealed** with an end marker and the next is started.
- **Position** — `(segment id, byte)`; positions grow in log order.
- **Tail / durable tail** — where the next record goes / below which everything
  is fsynced.
- **Torn tail** — a partly written last record after a crash; `open` cuts it off.
- **In flight** — a record appended but not yet applied to its memtable. The
  `LoggedWrite` guard an append returns keeps it in flight until dropped.
- **Ceiling** — at a seal, how far the namespace's applied records reach: the
  tail, lowered to the oldest of the namespace's records still in flight.
- **Retired** — a record whose memtable is in L0 files, so replay must skip it.
- **Safe offset** — per namespace: every one of its records below this position
  is retired. Its replay starts here.
- **Retired cut** — the lowest safe offset of any namespace that still has
  records above its own, capped by the oldest record still in flight. Records
  below it are retired for every namespace; GC deletes segments wholly below it.
- **Marker** — the `retired_cut` file, holding the cut and each namespace's safe
  offset above it. It is written at the *written* event, before the L0 files
  are registered.
- **Holding the cut** — a namespace whose unflushed records keep the cut from
  moving. The WAL GC tick flushes such namespaces.
- **Prefix rule** — what survives any crash is a prefix of the log, across all
  namespaces: a sync makes a prefix durable, and a flush syncs the log before it
  writes L0 files (§5).
- **Poisoned** — the log after a failed fsync: it refuses appends and syncs
  until the database is reopened (§6).

**The vector side**

- **Vector namespaces** — for a store `ns`: `{ns}_sparse_vector` (1-bit chunk
  codes, keyed by posting and document), `{ns}_sparse_vector_meta` (the
  **meta**: which postings hold the document's chunks, and the text hash) and
  `{ns}_dense_vector` (the 8-bit whole-document code).
- **Posting** — a group of chunks with one centre, the unit search reads.
- **Queue entry** — a document waiting in `system_pending_vec_index` to be
  embedded. Removing it ("completing" it) says the document's vectors are done.
- **Partition maintenance** — splitting, merging and reassigning postings
  (`ivf_split`). A **drain** empties a posting into others (a split or a merge
  of postings), then **retires** it; a **reassign** moves chunks between active
  postings. Each runs through a **journal** record that recovery **redoes**
  after a crash.

## 1. The problem

The worker writes a document's vectors, then completes its queue entry. The
chunk codes and the dense code are no-WAL writes, while the completion is a
WAL-backed delete, durable at once. If the process dies after the completion but
before the vectors are flushed, the document is indexed nowhere and nothing
re-queues it. So the worker completes an entry only after a **barrier**:
`make_vector_writes_durable` flushes the three vector namespaces
(`vector_kv.rs`), once per batch of up to 256 written entries and once at the end
of every pass.

Under steady traffic a pass holds one or two documents, so the barrier flushes
three namespaces per document. Each flush writes and fsyncs one L0 file per
bucket that has data, plus the directory. Measured on the M5 benchmark (FiQA,
balanced mix, 2026-10-15), at 50 requests/s:

| | Memtable flushes/s | L0→L1 merges/s | Compaction rewrite | Disk writes |
|---|---|---|---|---|
| Before the compaction fix | 20.2 | 42.4 | 17.0 MB/s | 18.4 MB/s |
| With the compaction fix (size-ratio trigger) | 21.7 | 6.8 | 2.7 MB/s | 4.1 MB/s |

The compaction fix (`LSMTree::compaction_due`) removed most of the bytes, but
not the flushes: about 20 a second, each a dozen small files and fsyncs, on the
same disk as the main WAL's fsync that every client write waits for. The flushed
memtables hold about 13 entries each against a capacity of 100,000.

## 2. The idea

Give the vector writes a log of their own. The worker appends each vector write
to the **vector index WAL** (`vector_index_wal`) as it applies it, and before completing a batch it fsyncs
that log once: one sequential file, no memtable flush. The rule stays exactly the
same — *an entry is completed only after its vectors are durable* — only the way
they are made durable changes.

There is **one** vector index WAL per database, shared by every vector namespace;
each record carries its namespace id. One log because:

- a completion batch can cover documents of several stores, and one log makes its
  barrier one fsync however many namespaces the batch wrote;
- one set of segments and one GC pass, however many stores have semantic search;
- it mirrors the main WAL, which is also shared and namespace-tagged, so its
  retirement rules carry over.

The cost is that a namespace that writes a little and goes idle holds back the
cut for every namespace; §5 flushes such namespaces on the GC tick.

After a crash, the vector writes still in the log and not yet flushed are
replayed into their memtables, with their original sequence numbers. A
completed entry therefore always has its vectors, and nothing has to be
re-embedded.

### What is logged

Exactly the writes the worker makes as no-WAL today, as key and value bytes:

| Write | Namespace | Bytes (gemma, 768 dims) |
|---|---|---|
| Chunk codes of one posting | `{ns}_sparse_vector` | ~130 B per chunk |
| Whole-document code | `{ns}_dense_vector` | ~800 B |

That is about 1.2 KB per FiQA document (3 chunks on average). The float
embeddings are never logged: the namespaces hold codes, and replay rebuilds the
namespaces, so it needs nothing else. Replay never re-encodes and does not
depend on the partition's centres (each code carries its own `centre_id`).

Not logged:

- The worker's meta write (`{ns}_sparse_vector_meta`) and the worker's and
  document deletes' vector deletes stay in the **main WAL**. The meta must be
  durable before any key it lists is written, and a lost delete would orphan an
  index entry (see `upsert_vectors`).
- The query-embedding cache stays no-WAL: it is a cache.
- The partition's centres and postings stay in the main WAL, the root posting
  written when a namespace enables semantic search included: they are small
  records, each durable on return.

Partition maintenance (split, merge and rebuild drains, and reassignment,
`ivf_split`) logs **every** per-document write: the copies, the destination's
meta, the drained posting's meta drop and key delete, and reassign's source
meta and key shrink. One sync per batch of documents, before the batch
releases their locks, makes them durable. Its writes span two namespaces (meta
and chunk keys) whose order matters, which is safe because of the prefix rule
in §5: whatever survives a crash is a prefix of the log. Design and measured
gain: `Fewer-WAL-Fsyncs-Design.md`, Part 2.

### Why a separate log and not the main WAL

- **The main WAL keeps its own write path.** Its lock, fsync and GC are its
  own; recovery reads one more log, and a flush saves both logs' progress. All new logic lives in one module, so a fault in it cannot reach ordinary
  writes.
- **No contention with client writes.** The vector index WAL has its own lock and its
  own fsync. Embedding traffic stops queueing on the one lock every client write
  takes.
- **Fsync per batch is natural.** No client waits on this log, so one fsync per
  batch of completions is the design. The main WAL's rule (each write returns
  only once an fsync covers it) is about client acknowledgements and does not
  apply.

## 3. Write path

For each logged write (`Database::put_ns_logged` / `delete_ns_logged`; the
worker's `upsert_vectors` and partition maintenance call them):

1. Take the key's stripe lock (as every write does).
2. Allocate the write's seq.
3. Append `(namespace id, key, value or delete, seq)` to the vector index WAL, under its append
   lock, and register the append's offset as **in flight** (§5).
4. Apply to the memtable with that seq (`put_to_storage` with a given seq).
5. Clear the in-flight registration.

The append is a `write`, not an fsync. Order (3) before (4) mirrors the main WAL:
a write is in the log before it is visible.

The batch barrier becomes:

1. `vector_index_wal.sync()` — one fsync of the log's active segment (and its
   directory, if the segment is new). Returns once every append made before the
   call is on stable storage.
2. Complete each queue entry, as now.

`make_vector_writes_durable` keeps its name and its contract; its body changes
from "flush three namespaces" to "fsync the vector index WAL".

These writes no longer mark their namespace as holding un-flushed no-WAL writes
(`note_no_wal_write`): they are recoverable, so the compaction worker's
no-WAL flush tick does not need to flush them.

## 4. Record and file format

- **Segments.** `vector_index_wal/{segment id}.log`, each a fixed size (64 MiB),
  appended in order. A segment is created at its full size, and from the
  log's second segment on it is zero-filled ahead in the background (as
  `{segment id}.log.prep`, renamed into place when the log reaches it), so a
  sync commits no new file size or block allocation. A new segment's header
  and directory entry are fsynced before anything in it is relied on.
- **Header** per segment: magic, format version, segment id.
- **End of a segment's records.** A full segment is sealed with an end marker:
  a length of `0xFFFF_FFFF`, which no record can have. A length of 0 is
  unwritten space: the records end there.
- **Record:** length, CRC32 of the body, then the body: namespace id (u32), seq
  (u64), key length and key, value length and value. A **delete** has value
  length `u32::MAX` and no value bytes; no put can be that long, since a record
  must fit a segment. Replay applies a delete with its seq.
- **Torn tail.** A record whose length runs past the end, or whose CRC fails, is
  the end of the log, as is unwritten space. Open zeroes everything after the
  end in that segment before any append, so records a power loss kept beyond
  a lost one can never follow new records. Only the last segment may end this
  way; a bad record, or unwritten space before the end marker, in an earlier
  segment fails the open, as a corrupt main WAL does.

The format is versioned so a later change can be rejected rather than misread.

## 5. Retirement: when a record is no longer needed

**First, the prefix rule.** A sync makes a prefix of the log durable, and a torn
tail drops only a suffix. A memtable written to L0 is another way a logged write
reaches disk, so before a flush writes any L0 file, the log is synced if the
namespace has records past the durable tail
(`LsmFlushObserver::on_ro_memtable_flushing`, sent after the seal and before
the first L0 file is written). L0 files
are found at open by listing their directories, so this must happen before the
files are written, not when they are registered. Then no L0 file holds a logged
write the log could still lose, and what survives any crash is exactly a prefix
of the log, across all namespaces. The sync is usually a no-op, since the
writers sync each batch.

A record is **retired** once the memtable holding its write has been flushed to
L0: from then on the SSTables have it and replay must skip it. This is the main
WAL's rule, applied per vector namespace, and it reuses the same mechanism:

- Each vector namespace's flush observer (`LsmFlushObserverHub`) also reports to
  a **vector-index-WAL observer**, alongside the main WAL observer.
- **On seal** of namespace `n`'s memtable version `v`, the observer records
  `(n, v) → ceiling`, where `ceiling` is the vector index WAL's tail lowered to the
  oldest of **`n`'s** records appended but not yet applied. Every record of `n`
  below the ceiling is in that memtable or an older one: the seal runs under the
  memtable's write lock, so nothing is applied into the next memtable meanwhile.
  (The ceiling matters: a write appended but not yet applied goes into the
  *next* memtable while its offset sits below the tail — the bug the main WAL's
  `wal_cut_ceiling` closes.) It is per namespace, unlike the main WAL's, because
  another namespace's write in flight says nothing about `n`'s memtable; with a
  global ceiling, `n` could flush everything it wrote and still hold the cut,
  with an empty memtable that no later flush would seal.
- **When `(n, v)`'s L0 files are written and fsynced, before they are
  registered**, the observer advances `n`'s `safe_offset`, strictly in version
  order (an out-of-order flush cannot skip an older pending memtable). The LSM
  reports this moment separately (`LsmFlushObserver::on_ro_memtable_written`);
  until the files are registered nothing can read or merge them. A memtable
  with no seal record retires nothing: `n`'s first record may have landed in
  the next memtable after that seal.
- **If that progress cannot be made durable** — the marker write fails, or
  the value-log fsync before it — the event returns an error and the LSM undoes
  the flush: it removes the files, keeps the memtable read-only, and flushes it
  again later under the same version. The log's progress does not change
  meanwhile, so no later marker (written for another namespace) carries the
  failed flush. Registering the files anyway would make them mergeable while
  the marker is behind, which is the hazard of §7.
- **A seal never waits on the marker's I/O.** The seal callback runs under the
  memtable's write lock, for every namespace of the database, vector or not; a
  seal that waited for a marker fsync would stall that namespace's reads and
  writes. Seal records and which namespaces are tracked sit under a lock held
  only briefly; the marker is written under a separate one.
- The marker file (tmp + rename + fsync) holds the **retired cut** — the minimum
  `safe_offset` over vector namespaces still holding un-flushed logged writes,
  capped by the ceiling of all namespaces — and **each namespace's own
  `safe_offset`** above it. It is written in that same event, so it is durable
  before the files are registered. The cut only moves forward. At open the
  marker says where each namespace's replay starts (§7).

A namespace that writes a little and goes idle would hold the cut back, as in the
main WAL. The vector index WAL's GC tick flushes vector namespaces that pin it, as
`flush_namespaces_pinning_wal` does for the main WAL, and **seals the memtable
even when it is empty**: a namespace can hold the cut with nothing in its
memtable (its last record was logged but its apply failed), and only a seal
releases it. These flushes are rare (once per tick at most, per idle
namespace), not per document.

A **dropped namespace** is forgotten and never tracked again (namespace ids are
not reused), so a write that raced the drop cannot hold the cut.

## 6. Garbage collection

On the main WAL GC tick (60 s): delete every vector-index-WAL segment that lies wholly
below the retired cut. Delete in order, oldest first; fsync the directory after.
A segment is never deleted while it holds an unretired record. No field index
reads these namespaces, so there is no index-replay watermark to respect.

The pass shares the main WAL GC's tick, not its decisions: it runs after the
main WAL's pass, uses only its own retired cut, and an error in either pass is
logged without skipping the other. A stuck main-WAL GC must not leave the vector
index WAL growing, nor the reverse.

Disk use is bounded by what the vector namespaces have not flushed: at most one
memtable each (100,000 entries), plus the segment in progress.

**A failed fsync stops the log.** After an `fsync` fails, the kernel may have
dropped the pages it could not write, and a later `fsync` that succeeds says
nothing about them. Carrying on would let later batches complete over that
hole, and a crash would then read the hole as a torn tail and drop those
batches' records. So the first failure stops the log: appends and syncs fail
until the database is reopened, the vector worker's entries stay queued, and
`GET /admin/storage/vector-index-wal` reports `stopped` with the error.

## 7. Recovery

At open, after the registry and stores are loaded and **before** any worker
starts:

1. Read the marker: the retired cut and each namespace's `safe_offset` (zero if
   absent).
2. Read every vector-index-WAL record at or above the cut, stopping at the torn
   tail. Segments wholly below the cut are retired and are not read at all
   (they only wait for GC). Keep only those at or above **their own namespace's** `safe_offset`
   (the cut, for a namespace the marker does not list).
3. Skip records whose namespace is no longer in the registry (dropped store or
   dropped vector index), as main-WAL recovery does.
4. Replay each remaining record with its original seq (`replay_upsert`), which
   writes the value to the value log again and the pointer to the memtable. The
   memtable keeps the highest seq per key, so a replayed record never overrides a
   newer write, from either log.
5. Raise the global seq counter above the highest seq in the vector index WAL, as it
   is raised above the main WAL and every SSTable (`set_seq_counter`). Without
   this, a new write could draw a seq below a replayed one and lose to it.
6. Flush every namespace replayed into to L0, fsync its value log, and record it
   in the marker as safe up to the log's tail. No flush observer is attached
   during recovery, so nothing else records that these records are now on
   disk — and the next step compacts every store into L1, which drops
   tombstones. With the marker left behind, a second crash would replay a
   record whose delete that compaction erased, and the key would come back
   (`a_second_crash_does_not_bring_back_a_deleted_key`).
7. Main-WAL recovery runs as it does now. Both logs replay with original seqs
   into memtables that resolve by seq, so their relative order does not matter.

If the marker names a position past the log's durable end, a power loss took
records of a memtable that had already been written to L0: their data is safe
in SSTables, but the log lost them. The open then seals the last segment and
starts the next, so every new record lies above every position the marker
names; otherwise replay would skip new records below a namespace's
`safe_offset`.

Before any of this, opening the log rebuilds which namespaces hold the cut: the
in-memory seal records and per-namespace progress are lost in a crash, so every
namespace with a record at or above the cut starts out holding it, and
namespaces no longer in the registry are released. Otherwise the first flush of
any namespace would find nobody holding the cut and retire the replayed
records while they exist only in memtables.

A **clean shutdown** flushes every store while the flush observers are still
attached, and only then detaches them and compacts. Compaction into L1 drops
tombstones, so the log must have retired what those flushes covered first;
otherwise the next open would replay records whose keys were deleted during
the session. A store whose flush fails there keeps its observers, so the
flush its shutdown runs before compacting is still seen (and undone if the
marker still cannot be written). Shutdown carries on with the other stores
and reports the first error.

Then the vector worker starts and finds the queue as the crash left it.

### Why replaying an old seq is safe here

`minnal_db/CLAUDE.md` warns that re-applying an old seq can bring back a deleted
key: L1 drops tombstones, so if the key's later delete already reached L1, the
old write wins again. The main WAL avoids this because its deletes are in the
same log and replay alongside its puts. A vector key's later writes can be in
**either** log (the worker's and document deletes are in the main WAL,
maintenance's in this one), so the vector index WAL has to avoid it by itself:

- A later write to the same key (a delete of a stale chunk key, a re-embed)
  holds the key's stripe from seq allocation to apply, so it is applied after
  the replayed record, into the same memtable of that namespace or a later one.
- Memtables are written to L0 in version order, so if that later write has been
  written (let alone merged into L1 with its tombstone dropped), the replayed
  record's memtable was written first.
- That memtable's write moved the namespace's `safe_offset` past the record,
  durably, before its files could be merged.
- So replay, which starts each namespace at its own `safe_offset`, never applies
  a record that has a written later write. The global cut alone would not do:
  another namespace can hold it below a record whose namespace flushed long
  ago, and that record's key can be deleted and merged into L1 meanwhile.
  `replay_does_not_bring_back_a_key_whose_delete_reached_l1` reproduces exactly
  that, and fails when replay starts every namespace at the cut.

## 8. Crash points and tests

Every crash point below gets a test in the style of `vector_kv`'s crash audits
(`crash_point!` countdown, then reopen, worker pass, reconciliation, worker
pass), and each test checks the same invariants: every completed document has
its vectors; no document is indexed that is deleted; no chunk key is listed by
no meta.

| Crash after… | On disk | After restart |
|---|---|---|
| append, before memtable apply | record (maybe unsynced) | replayed, or lost if unsynced; entry still queued either way → re-embedded |
| apply, before `sync` | records maybe unsynced | same as above: not completed, so redone |
| `sync`, before completion | records durable | replayed; entry still queued → re-embedded (harmless rewrite) |
| completion | records durable, entry gone | replayed → document indexed |
| seal, before flush | records unretired | replayed; flush redone later |
| flush, before the cut marker is written | records flushed but look unretired | replayed with original seqs: highest seq wins, so no change (the CLAUDE.md hazard cannot arise: nothing later to this key was flushed after them — §7) |
| segment delete, before directory fsync | segment may reappear | wholly below the cut → skipped |
| mid-replay | nothing new is retired until a flush | replay runs again from the same cut |
| recovery, after its flush and marker | replayed records in L0 and retired | not replayed again; later compactions cannot bring a deleted key back |

Power loss differs from a process kill: unsynced appends vanish. The crash
simulation (`AsyncDb::crash`) must model that by cutting each segment of the
vector index WAL back to what an fsync actually covered, or the tests cannot
see a missing fsync. (The same blind spot hid
no-WAL writes until 2026-10-11.)

**Mutation checks** — each must make a test fail, or the test is not guarding
anything:

1. Replay starts every namespace at the global cut (or ignores it) → a key
   deleted and compacted to L1 comes back.
2. `sync` skipped before completion (with the crash simulation truncating
   unsynced appends) → a completed document has no vectors.
3. Retire on seal instead of on flush → a crash between seal and flush loses a
   completed document's vectors.
4. No ceiling (seal records the raw tail) → an in-flight write is retired early
   and lost.
5. Seq counter not raised above the vector index WAL → a write after restart loses to
   a replayed older one.
6. Recovery does not record what it replayed → after two crashes a deleted
   chunk key is back (`a_second_crash_does_not_bring_back_a_deleted_key`).
7. A failed marker write is not undone, or the LSM registers the files anyway
   → a later marker carries the failed flush, or its files are mergeable
   (`a_failed_marker_write_changes_nothing_and_the_retry_retires`,
   `a_flush_whose_marker_fails_registers_nothing_and_is_retried`).
8. The GC tick's flush does not seal an empty memtable → a namespace holds the
   cut for good (`a_pin_flush_releases_a_holder_whose_memtable_is_empty`).
9. A forgotten namespace is tracked again by a late append → it holds the cut
   until restart (`an_append_after_a_namespace_is_forgotten_does_not_hold_the_cut`).
10. A seal takes the lock held across the marker write
    (`a_seal_never_waits_for_a_marker_write`).

The barrier tests (`crash_before_reembed_is_synced_keeps_new_vectors`,
`a_failed_sync_completes_nothing` and the crash audits) guard the fsync.

## 9. Plan

Each step is its own commit, tested in all four feature combinations, with the
main WAL's tests untouched and passing.

1. **The log** (`db/vector_index_wal.rs`): segments, record format, append, `sync`,
   torn-tail reading. Unit tests, including corrupt and torn records.
2. **Retirement and GC:** the observer (seal/flush, ceiling, version order), the
   cut marker, segment deletion, flushing idle namespaces, and the log's stats
   (§9a) in the storage admin endpoints. Tests mirroring the main WAL observer's.
3. **Recovery:** replay at open, seq counter raise, dropped namespaces; crash
   simulation truncating unsynced appends.
4. **Switch the worker:** logged writes in `upsert_vectors`, `sync` in
   `make_vector_writes_durable`. Crash audits and the five mutation checks.
5. **Measure:** the M5 quick check before and after (flushes/s, fsyncs/s, write
   p99, sustainable rate), the vector bench (indexing time, recall and nDCG at
   @10–@100 must be identical), and a restart after a crash mid-ingest.
6. **Partition maintenance:** the drain and reassignment copies (`copy_into_batch`)
   on the vector index WAL, and their three barriers (drain, rebuild,
   reassignment) one fsync each instead of a flush of two namespaces. Built
   with a fix it exposed: a clean shutdown must flush with the observers still
   attached (§7).
7. **minnal_ui**, after step 5: show the log where the main WAL already appears,
   since it is system-wide. Admin → Storage gets a *Vector index WAL* card beside
   the *WAL* card (the on-disk state of §9a, with its segment list); Admin → Ops
   Metrics gets a *Vector index WAL* card in the engine-wide column (the counters
   of §9a, plus appends per fsync, which is the batch size this design relies
   on). Both cards are hidden when the server does not report them. Tested on a
   scratch server: empty, during ingest, idle past a GC tick, after a SIGKILL
   restart (minnal_ui has one, dark, theme).

### 9a. What the log reports

On-disk state, which survives a restart: `GET /admin/storage/vector-index-wal`
returns the segments (id and bytes, their count and total), the tail, the
durable tail, the retired cut, the unretired bytes between them (what a restart
would replay), the last seq appended, and the namespaces holding the cut back
with their `safe_offset` (why the log is not shrinking). Positions are shown as
`{segment_id, byte}`. It is a separate endpoint because
the two logs have different segment layouts and stopping rules. After a failed fsync it also reports `stopped`, with the error
(the log refuses appends until the database is reopened, §6).

Counters since startup, engine-wide only (as the main WAL's GC counters are; a
per-namespace view reads 0), in `GET /admin/storage/ops-metrics`:

| Counter | Counts | Step |
|---|---|---|
| `vector_wal_appends`, `vector_wal_bytes_appended` | records and bytes appended | 1 |
| `vector_wal_fsyncs` | `sync` calls that fsynced | 1 |
| `vector_wal_flush_syncs` | syncs a memtable flush asked for (the prefix rule, §5) | Fewer-WAL-Fsyncs Part 2 |
| `vector_wal_gc_runs`, `vector_wal_segments_deleted` | GC passes, segments deleted | 2 |
| `vector_wal_pin_flushes` | flushes of an idle namespace holding the cut back | 2 |
| `vector_wal_records_replayed` | records replayed at open | 3 |
| `vector_wal_records_skipped` | records of dropped namespaces skipped at open | 3 |

### 9b. Measured: the quick M5 check

**Setup.** The M5 REST harness (`minnal_db/docs/benchmarks/m5/run_m5.py balanced
--quick`, at commit `d7434ce`) on the settled FiQA snapshot (40,000 documents embedded with gemma),
balanced mix, 1-minute steps from 50 req/s ×1.5 with two bisections, a 2-minute
soak at the highest passing rate, closed-loop 1/4/16 clients, and a cold-query
step (30% cache misses). **Before**: the build with the vector worker's flush
barrier (2e88f1d). **After**: the vector index WAL (df8cdc6). Run before, after,
before, after on one host, 2026-10-16; results in
`work/bench/results/m5-quick-vwal-{before,after}-{1,2}/`.

**Background work at the same offered rate.** This is what the log was built to
remove, and it is not noise: both runs of each build agree.

| Offered | Build | Memtable flushes/s | L0→L1 merges/s | Disk writes MB/s | Main-WAL fsyncs/s | Vector-WAL fsyncs/s | Server CPU (cores) |
|---|---|---|---|---|---|---|---|
| 50 req/s | before | 22.8, 21.6 | 7.1, 6.5 | 4.2, 3.9 | 149, 137 | — | 1.48, 1.46 |
| 50 req/s | after | 0.9, 0.9 | 0.7, 0.7 | 1.3, 1.3 | 151, 150 | 8.5, 8.4 | 1.46, 1.45 |
| 75 req/s | before | 31.5, 29.1 | 8.6, 8.5 | 5.0, 5.0 | 131, 144 | — | 2.14, 2.14 |
| 75 req/s | after | 0.5, 0.5 | 0.4, 0.5 | 1.1, 1.2 | 131, 131 | 13.0, 13.1 | 2.11, 2.13 |

Memtable flushes fall by 96–98% and disk writes by 70–77%. The main WAL's
fsyncs do not change (it never carried the payloads), and neither does CPU. At
these rates the worker's batches are small, about 3 records per vector-WAL
fsync; under a backlog they grow to hundreds (508 per fsync in one soak). The
remaining ~1 flush/s and the no-WAL puts (36/s at 50 req/s, from 62) came from
partition maintenance's copies, which at the measured commit (df8cdc6) still
had their own flush barrier; step 6 puts them on the log.

**Latency and throughput.** Write p99 at 50 req/s: insert 25/34 → 19/20 ms,
update 22/40 → 18/19 ms, delete 76/82 → 71/78 ms (before/after, both runs each).
At 75 req/s the runs overlap. Closed loop, completed req/s is unchanged (1
client 114/113 → 111/118, 16 clients 464/463 → 453/422), and write p99 at 16
clients is the same or lower (delete 380/298 → 305/272 ms).

The **highest passing rate** does not settle the question with 1-minute steps.
Before reached 69 and 103 req/s, after 232 and 103; both after-2 and before-2
stopped at 112 req/s on a single class just over its target (update 51 ms,
insert 56 ms). In after-1's soak at 232 req/s the vector queue grew by 1,501
entries, so that rate is not sustainable: inserts and updates there ask for
about 46 embeddings a second, past the embedding pipeline's rate, which is now
the limit before storage is. The full M5 run, with 10-minute steps, measures
the sustainable rate.

**To check in the full run.** Search p99 at 16 closed-loop clients was higher
after in both pairs (112/114 → 126/135 ms) while write p99 fell. One
explanation to test: the vector namespaces' memtables now hold up to a
memtable's worth of entries between flushes instead of a few dozen, so a
search's prefix scans read more from memory. The cold-query step is not
comparable across builds here: it runs at each build's highest passing rate,
and after-1's ran with an embedding backlog.

### 9c. Measured: the vector bench

**Setup.** The vector bench (`semantic_search::vector_bench`) indexes a frozen
set of gemma embeddings with background workers running, growing the partition
from one posting (target 128), then searches. SciFact (5,183 documents) and
FiQA (57,600), corpus order, three latency passes. **Before**: a test binary
built at 2e88f1d (the worker's flush barrier). **After**: one built at 9f91bc7
(the vector index WAL, steps 1–6). Run before, after, before, after on one
idle host, 2026-10-17; results in
`work/bench/results/vwal-bench-{before,after}-r{1,2}/`. (The `git` field in
those files names the checkout the bench ran from, not the binary's commit.)

**Results are identical.** In all four before/after pairs, every probe setting
(fixed 4–128 postings and every entry budget) scans the same entries and gives
the same nDCG@10, ANN recall@10 and Pass-1 recall, with no query won or lost;
at 64 postings nDCG is identical at @10, @20, @30, @40, @50 and @100. The
partition is the same too (SciFact 133 postings and 132 splits, FiQA 979 and
978; the same reassignment and drain moves), as are the index size and the
estimator errors.

**Indexing is faster**, because the barrier after each batch is one fsync of
the log instead of a flush of three namespaces:

| Dataset | Before (s) | After (s) | Change |
|---|---|---|---|
| SciFact | 4.14, 4.11 | 3.32, 3.32 | −20% |
| FiQA | 53.4, 52.8 | 46.0, 44.9 | −14% |

Search p50, averaged over every probe setting, is +1.8% and +2.6% on SciFact
and +1.7% and −0.1% on FiQA (a few hundredths of a millisecond). That is close
to run-to-run noise, but it points the same way as the search p99 question
above, so the full M5 run checks both.

## 10. Alternatives considered

- **Group the barrier** (flush once per interval, complete entries afterwards).
  Fewer flushes, but entries wait in the queue, and every flush still makes small
  L0 files. Superseded by this design.
- **A conditional completion record in the main WAL** ("remove entry E, valid
  only if the vector namespaces flushed past seq S"). No forced flushes, but it
  needs a cross-namespace pin in the main WAL's cut and GC — new
  rules in the most sensitive code. Rejected for that reason.
- **The value log as the WAL** (WiscKey's approach). No-WAL values already go to
  the value log; only the key's pointer is memory-only. Recovery could rebuild
  pointers from the value log's tail. Rejected: value-log GC relocates records
  (with old seqs) into the same tail, the value log knows nothing of memtable
  boundaries for retirement, and it would change the value log's format and GC —
  more intrusive than a separate log.
- **Make the vector writes WAL-backed.** Correct, but each write would wait for
  a main-WAL fsync (~4 per document), and the bulky codes would go into the log
  every client write shares.

## Decisions (review, 2026-10-15)

1. **Code location:** `db/vector_index_wal.rs` holds the log (record body,
   observer, retirement, replay) on the segmented log shared with the main WAL
   (`db/segmented_log.rs`: segments, frames, append, sync, open, GC);
   `database.rs` calls it at open and from the GC tick.
   `minnal_db/CLAUDE.md`'s WAL-ownership rule names both files.
2. **Fsync policy:** one `sync` per completion batch; no time bound.
3. **Segment size:** 64 MiB, as the main WAL.
4. **Admin stats:** the vector index WAL's state and counters (§9a) are reported
   from the step that creates them, before the worker uses the log, and shown in
   minnal_ui's Storage and Ops Metrics tabs (step 7).
