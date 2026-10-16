# Vector index WAL — design note

**Status:** agreed 2026-10-15 (decisions at the end). Built in the steps of §9.
**Scope:** how the vector worker makes a document's vector writes durable before
it removes the document's queue entry. Nothing here changes what search reads,
the main WAL's write path, or the doc store's API.

## Terms used here

- **Main WAL** — the database's write-ahead log (`db/wal.rs`). Every ordinary
  write appends one entry and fsyncs it before the call returns.
- **No-WAL write** — `put_no_wal`: the value goes to the value log and the key's
  pointer goes to the memtable, with no WAL entry. The pointer exists only in
  memory until the memtable is flushed to a level-0 (L0) file.
- **Vector namespaces** — for a store `ns`: `{ns}_sparse_vector` (1-bit chunk
  codes, keyed by posting and document), `{ns}_sparse_vector_meta` (which postings
  hold the document's chunks, and the text hash) and `{ns}_dense_vector` (the
  8-bit whole-document code).
- **Queue entry** — a document waiting in `system_pending_vec_index` to be
  embedded. Removing it ("completing" it) says the document's vectors are done.
- **Flush** — writing a namespace's memtable to L0 files and fsyncing them.
- **Seal** — turning a full (or force-flushed) memtable read-only, ready to flush.
- **Sequence (seq)** — the global write counter every write draws from; the
  highest seq for a key wins everywhere (memtable, L0, L1, replay).

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

Not logged, and unchanged:

- The meta (`{ns}_sparse_vector_meta`) and every vector delete stay in the
  **main WAL**. The meta must be durable before any key it lists is written, and
  a lost delete would orphan an index entry (see `upsert_vectors`).
- The query-embedding cache stays no-WAL: it is a cache.
- The partition's centres and postings keep their own flush when seeded.
- Partition maintenance's drain copies (`ivf_split.rs`) keep their barrier in the
  first version; moving them to the vector index WAL is a follow-up (§9).

### Why a separate log and not the main WAL

- **The main WAL does not change.** Its write path, lock, fsync-per-write rule,
  persisted watermark and GC stay as they are; only recovery reads one more
  log. All new logic lives in one module, so a fault in it cannot reach ordinary
  writes.
- **No contention with client writes.** The vector index WAL has its own lock and its
  own fsync. Embedding traffic stops queueing on the one lock every client write
  takes.
- **Fsync per batch is natural.** No client waits on this log, so one fsync per
  batch of completions is the design. The main WAL's rule (one fsync per
  acknowledged write, no group commit) is about client acknowledgements and does
  not apply.

## 3. Write path

For each vector write, inside `upsert_vectors`, in place of `put_no_wal`:

1. Take the key's stripe lock (as every write does).
2. Allocate the write's seq.
3. Append `(namespace id, key, value, seq)` to the vector index WAL, under its append
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

- **Segments.** `vector_index_wal/{segment id}.log`, each up to a fixed size (64 MiB),
  appended in order. A new segment's directory entry is fsynced before anything
  in it is relied on.
- **Header** per segment: magic, format version, segment id.
- **Record:** length, CRC32 of the body, then the body: namespace id (u32), seq
  (u64), key length and key, value length and value.
- **Torn tail.** A record whose length runs past the end, or whose CRC fails, is
  the end of the log; everything after it in that segment is ignored. Only the
  last segment may end this way; a bad record in an earlier segment fails the
  open, as a corrupt main WAL does.

The format is versioned so a later change can be rejected rather than misread.

## 5. Retirement: when a record is no longer needed

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
- **On flush** of `(n, v)`, the observer advances `n`'s `safe_offset`, strictly
  in version order (an out-of-order flush cannot skip an older pending memtable).
  A flush with no seal record retires nothing: `n`'s first record may have
  landed in the next memtable after that seal.
- The vector index WAL's **retired cut** is the minimum `safe_offset` over vector
  namespaces still holding un-flushed logged writes, capped by the ceiling of
  all namespaces. It only moves forward, and it is written to a small marker
  file (tmp + rename + fsync) before it is used; at open it says where replay
  starts.

A namespace that writes a little and goes idle would hold the cut back, as in the
main WAL. The vector index WAL's GC tick flushes vector namespaces that pin it, exactly
as `flush_namespaces_pinning_wal` does for the main WAL. These flushes are rare
(once per tick at most, per idle namespace), not per document.

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

## 7. Recovery

At open, after the registry and stores are loaded and **before** any worker
starts:

1. Read the retired cut from the marker file (zero if absent).
2. Read every vector-index-WAL record at or above it, stopping at the torn tail.
3. Skip records whose namespace is no longer in the registry (dropped store or
   dropped vector index), as main-WAL recovery does.
4. Replay each remaining record with its original seq (`replay_upsert`), which
   writes the value to the value log again and the pointer to the memtable. The
   memtable keeps the highest seq per key, so a replayed record never overrides a
   newer write, from either log.
5. Raise the global seq counter above the highest seq in the vector index WAL, as it
   is raised above the main WAL and every SSTable (`set_seq_counter`). Without
   this, a new write could draw a seq below a replayed one and lose to it.
6. Main-WAL recovery runs as it does now. Both logs replay with original seqs
   into memtables that resolve by seq, so their relative order does not matter.

Then the vector worker starts and finds the queue as the crash left it.

### Why replaying an old seq is safe here

`minnal_db/CLAUDE.md` warns that re-applying an old seq can bring back a deleted
key: L1 drops tombstones, so if the key's later delete already reached L1, the
old write wins again. Replay avoids this the same way main-WAL replay does — it
only replays **unretired** records:

- A later write to the same key (a delete of a stale chunk key, a re-embed)
  holds the key's stripe from seq allocation to apply, so it is applied after
  the replayed record, into the same memtable or a later one.
- Memtables flush in version order, so if that later write has been flushed (let
  alone merged into L1 with its tombstone dropped), the replayed record's
  memtable was flushed first, and the record is retired.
- So an unretired record has no flushed later write to override. The test for
  this is the first mutation in §8.

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

Power loss differs from a process kill: unsynced appends vanish. The crash
simulation (`AsyncDb::crash`) must model that by truncating the vector index WAL to its
last `sync`, or the tests cannot see a missing fsync. (The same blind spot hid
no-WAL writes until 2026-10-11.)

**Mutation checks** — each must make a test fail, or the test is not guarding
anything:

1. Replay ignores the retired cut (replays every record) → a key deleted and
   compacted to L1 comes back.
2. `sync` skipped before completion (with the crash simulation truncating
   unsynced appends) → a completed document has no vectors.
3. Retire on seal instead of on flush → a crash between seal and flush loses a
   completed document's vectors.
4. No ceiling (seal records the raw tail) → an in-flight write is retired early
   and lost.
5. Seq counter not raised above the vector index WAL → a write after restart loses to
   a replayed older one.

The existing barrier tests (`crash_before_reembed_is_flushed_keeps_new_vectors`,
`a_failed_flush_completes_nothing` and the crash audits) must keep passing,
retargeted at the fsync.

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
6. **Later, separately:** partition maintenance's drain copies on the vector index WAL
   instead of their flush barrier.
7. **minnal_ui**, after step 5: show the log where the main WAL already appears,
   since it is system-wide. Admin → Storage gets a *Vector index WAL* card beside
   the *WAL* card (the on-disk state of §9a, with its segment list); Admin → Ops
   Metrics gets a *Vector index WAL* card in the engine-wide column (the counters
   of §9a, plus appends per fsync, which is the batch size this design relies
   on). Both cards are hidden when the server does not report them. Tested on a
   scratch server: empty, during ingest, idle past a GC tick, after a SIGKILL
   restart, in light and dark themes.

### 9a. What the log reports

On-disk state, which survives a restart: `GET /admin/storage/vector-index-wal`
returns the segments (id and bytes, their count and total), the tail, the
durable tail, the retired cut, the unretired bytes between them (what a restart
would replay), the last seq appended, and the namespaces holding the cut back
with their `safe_offset` (why the log is not shrinking). Positions are shown as
`{segment_id, byte}`. It is a separate endpoint because
the main WAL's response counts entries and persisted entries, which this log
does not have.

Counters since startup, engine-wide only (as the main WAL's GC counters are; a
per-namespace view reads 0), in `GET /admin/storage/ops-metrics`:

| Counter | Counts | Step |
|---|---|---|
| `vector_wal_appends`, `vector_wal_bytes_appended` | records and bytes appended | 1 |
| `vector_wal_fsyncs` | `sync` calls that fsynced | 1 |
| `vector_wal_gc_runs`, `vector_wal_segments_deleted` | GC passes, segments deleted | 2 |
| `vector_wal_pin_flushes` | flushes of an idle namespace holding the cut back | 2 |
| `vector_wal_records_replayed` | records replayed at open | 3 |

## 10. Alternatives considered

- **Group the barrier** (flush once per interval, complete entries afterwards).
  Fewer flushes, but entries wait in the queue, and every flush still makes small
  L0 files. Superseded by this design.
- **A conditional completion record in the main WAL** ("remove entry E, valid
  only if the vector namespaces flushed past seq S"). No forced flushes, but it
  needs a cross-namespace pin in the main WAL's persisted watermark and GC — new
  rules in the most sensitive code. Rejected for that reason.
- **The value log as the WAL** (WiscKey's approach). No-WAL values already go to
  the value log; only the key's pointer is memory-only. Recovery could rebuild
  pointers from the value log's tail. Rejected: value-log GC relocates records
  (with old seqs) into the same tail, the value log knows nothing of memtable
  boundaries for retirement, and it would change the value log's format and GC —
  more intrusive than a separate log.
- **Make the vector writes WAL-backed.** Correct, but each write would take the
  main WAL lock and fsync (~2.3 ms each, ~4 per document), adding embedding
  traffic to the lock every client write waits on.

## Decisions (review, 2026-10-15)

1. **Code location:** `db/vector_index_wal.rs` holds the log (format, append,
   sync, observer, GC); `database.rs` calls it at open and from the GC tick.
   `minnal_db/CLAUDE.md`'s WAL-ownership rule names both files.
2. **Fsync policy:** one `sync` per completion batch; no time bound.
3. **Segment size:** 64 MiB, as the main WAL.
4. **Admin stats:** the vector index WAL's state and counters (§9a) are reported
   from the step that creates them, before the worker uses the log, and shown in
   minnal_ui's Storage and Ops Metrics tabs (step 7).
