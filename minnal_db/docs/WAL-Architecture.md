# Write-ahead logs

minnal has two write-ahead logs, built on one shared implementation:

- **The main WAL** (`db/wal.rs`, `<db_path>/wal/`): every `put`, `delete` and
  `merge` that is not a no-WAL write. Each returns only once an fsync covers
  its record.
- **The vector index WAL** (`db/vector_index_wal.rs`,
  `<db_path>/vector_index_wal/`): the vector worker's payload writes and
  partition maintenance's per-document writes, made durable by one fsync per
  batch (`sync`).

Both are a **segmented log** (`db/segmented_log.rs`, `SegmentedLog`). It owns
everything about how bytes reach the disk and come back after a crash. Each
log adds only a `LogFormat` (its name, a four-byte magic and a body version),
the encoding of its records' bodies, and its own rule for when records are
retired.

## Terms

- **Segment** — one file of a log, `{segment_id:010}.log`. Ids are contiguous;
  only the newest, the **active** segment, takes appends.
- **Frame** — one record on disk: `[len u32][crc32(body) u32][body]`.
- **Position** — a place in a log: `(segment_id << 32) | byte_in_segment`. The
  admin API shows it as `{segment_id, byte}`. Records start at byte 16, after
  the segment header. Names that say *offset* (a namespace's *safe offset*, a
  field's *checkpoint offset*) are positions too.
- **Tail** — where the next record goes. **Durable tail** — everything below it
  is fsynced.
- **Cut** (retired cut) — every record below it is in an SSTable; replay starts
  there, and GC deletes segments wholly below it.
- **Safe offset** — per namespace: every one of its records below it is in
  L0 files. It can be above the cut, which is the lowest of them.
- **Poisoned** — stopped after a failed fsync: appends and syncs fail until
  the database is reopened.

## On disk

```text
segment: [magic 4B | format_version u32 | segment_id u64]   16-byte header
         [frame] [frame] ...
         [end marker: u32 0xFFFF_FFFF]                      sealed segments only
         [zeros to the end of the file]                     unwritten space
frame:   [len u32][crc32(body) u32][body]
```

All integers are little-endian. A segment is created at its full size, and
from the second segment on it is written in full with zeros ahead of need
(`db/segment_prep.rs`: under a `.prep` name, renamed into place when the log
reaches it). So an fsync never commits a new file size or block allocation,
and a zero length always means unwritten space. The end marker is a length no
frame can have; its four bytes are reserved in every segment.

| Log | Magic | Body |
|---|---|---|
| Main WAL | `MWAL` | an rkyv `WalEntry`: operation, namespace, sequence, op name, key, value |
| Vector index WAL | `MVIW` | `[ns_id u32][seq u64][key_len u32][key][value_len u32][value]`; a delete has `value_len = u32::MAX` and no value |

Each log also keeps a **progress file** (`segmented_log::ProgressFile`) in its
directory: the cut and every safe offset above it, plus the log's own extra
bytes, checksummed and written atomically.

```text
[magic 4B][version u32][cut u64][count u32][count × (ns_id u32, safe u64)]
[extra_len u32][extra][crc32 of everything before it]
```

| Log | File | Magic | Extra bytes |
|---|---|---|---|
| Main WAL | `wal/metadata` | `WALM` | WAL GC runs, bytes reclaimed, sequence high-water mark (3 × u64) |
| Vector index WAL | `vector_index_wal/retired_cut` | `MVIC` | none |

A progress file whose magic, version, checksum or length is wrong, whose
namespaces are not sorted, or that lists a safe offset below the cut, is
corruption.

## Writing

`SegmentedLog::append(body_parts, under_lock)` writes one frame under the
log's append lock. The body is passed as slices and copied once into the
frame. `under_lock(start, end)` runs after the write and before the new tail
is published, which is where a log registers the record as not yet applied
to a memtable. A record that does not fit a segment is refused before
anything is written (`LogError::RecordTooLarge`; the main WAL reports it as
`KVError::WriteTooLarge`).

When a record does not fit what is left of the active segment, the segment is
**sealed**: the end marker is written, the segment is fsynced, and only then is
the next segment created or installed. So only the last segment can ever hold
bytes no fsync covered. Any failure while sealing poisons the log.

`commit(end, delay)` and `sync()` run as **commit rounds**
(`db/commit_rounds.rs`): a caller whose bytes are not yet durable either waits
for the round in progress or, if none is, leads one — it reads the active
segment's file and tail under the append lock, fsyncs outside every lock, and
publishes the durable tail. Concurrent callers share one fsync. A round that
fails poisons the log before it wakes anyone, and a round that succeeds while
the log was poisoned acknowledges nothing: after a failed fsync the kernel may
have dropped the pages, and a later fsync that succeeds says nothing about
them. The main WAL commits every write (`sync.commit_delay_us` can make a
leader wait for more writers when others are already waiting); the vector
index WAL syncs per batch.

## Opening

`SegmentedLog::open(dir, format, segment_size, from, visit)` reads the log from
the saved cut (`OpenFrom::cut`) and hands each record at or above it to
`visit`. Segments wholly below the cut are retired and not read.

- **The last segment** is read in full to find the tail. A frame there whose
  length runs past the end or whose CRC fails is a **torn tail**: everything
  from it to the end of the file is zeroed (a power loss can keep pages past a
  lost record, and a later record laid over the gap could otherwise end where
  a stale frame begins). A last segment whose header is missing or all zeros
  was being created and holds nothing; it is recreated.
- **Any other damage fails the open**: a bad frame or unwritten space before
  the end marker of an earlier segment, a missing segment between two that
  exist, a header with another magic, version or id, a cut below the oldest
  segment, or no saved progress while segments start above 0. Nothing is cut
  or deleted. The error names the segment and byte.
- **Saved progress past the tail.** A flush can save progress covering
  records a power loss then takes (after a failed fsync, or before a sync).
  New records must land above every position the progress names, or the next
  replay would skip them, so open seals the last segment and starts a new one.
- The segment size is not encoded in positions, so it can change between
  opens; new segments take the configured size.

**The main WAL's metadata.** The head and the tail are not saved: open takes
the tail from the log and the head from the oldest segment on disk. The
sequence counter starts above the saved high-water mark and every entry at or
above the cut; entries below the cut are in SSTables, which raise it too.
If the metadata is missing or unreadable, the cut is unknown and the whole log
would be replayed. That is safe only while no tombstone can have been dropped,
and compaction drops tombstones only into L1. So with L1 data and segments to
replay the open fails (a replayed put could bring back a key deleted since);
with no L1 data — a new database, or a crash during its first flushes — the
log is replayed whole and the damaged file kept as `metadata.corrupt`.

## Reading

`for_each(start, end, f)` streams the records in `[start, end)` with one
reused buffer. `for_each_with` takes a function that turns the log's errors
into the caller's error type, for a caller whose error does not convert from
`LogError`. Segments below the oldest one on disk are skipped (they are
retired; a reader racing GC skips a segment deleted under it the same way);
`missing_segments` reports them to a caller that must know, such as a field
index replaying from its checkpoint. A missing segment above the oldest, or a
damaged record, is corruption. A body that passes its CRC but does not decode
is corruption too.

## Retirement and GC

A record is retired once the memtable holding its write is in L0 files. Both
logs learn this from the LSM's flush events, and both save their progress in
the *written* event, before the flush's L0 files are registered and so before
compaction can merge them. Each log keeps a safe offset per namespace, and
replay applies a record only at or above its own namespace's offset: a
namespace that flushed past a record may since have had a delete of that key
merged into L1, where its tombstone is dropped.

- **Main WAL**: the cut is also held back by every active field index until it
  checkpoints past it (field indexes replay the WAL after a crash).
  `Database::garbage_collect_wal` deletes segments wholly below the lower of
  the two, oldest first, harvesting the keys of any segment the
  `max_pinned_wal_segments` backstop forces out (see `minnal_db/CLAUDE.md`).
- **Vector index WAL**: each namespace's retirement is tracked separately
  (seal records, a per-namespace ceiling); see the module docs of
  `db/vector_index_wal.rs` and `semantic_search/report/Vector-Index-WAL-Design.md`.

`delete_segments_below` never deletes the active segment and stops at the
first failure, so what is left is always one run of segments.

## Errors

`LogError` is the log-level error: `Io`, `Corruption`, `RecordTooLarge`,
`Poisoned`. Each log wraps it in one variant of its own error type, which
names the log in messages (`WalError::Log`, `VectorIndexWalError::Log`); the
main WAL adds encoding errors and an invalid configured segment size.

| Error | What it means | What to do |
|---|---|---|
| `Poisoned` | An fsync failed; the log refuses writes | Check the disk, then restart the server |
| `Corruption` naming a segment and byte | A damaged record or segment outside the torn tail | Restore the database from a backup; nothing was cut or deleted |
| `Corruption`: WAL metadata missing or unreadable with L1 data | The main WAL's cut is unknown | Restore `wal/metadata`; or, if every write is flushed, remove the `wal` directory (writes not yet in an SSTable are lost) |
| `InvalidSegmentSize` | `wal.segment_size_bytes` outside [64 KiB, 4 GiB] | Fix the configuration |

## Adding a log

1. Declare a `LogFormat` with a new magic and version 1, and a `ProgressFile`
   with its own magic.
2. Encode bodies as slices for `append`; decode them in `visit` and in your
   reader, and report a body that does not decode as `LogError::Corruption`.
3. Wrap `LogError` in one variant of your error type.
4. Decide when records retire, save the progress in the flush's written event
   before the L0 files are registered, and replay per namespace.
5. Bump the version whenever the body layout changes; nothing else detects it.

## Tests

The format and its rules are tested once, in `segmented_log.rs`: records
across segments and reopens, the segment size changing, reading from the cut,
the torn tail, damage that fails the open, foreign and missing segments, lost
progress, a blank last segment, progress past the tail, shared fsyncs,
poisoning by a failed fsync or rotation, oversized records, the progress
file, and a randomised power-loss loop. The power-loss model (test-only)
records what each fsync covered and cuts every segment back to it
(`lose_unsynced`, `Database::power_loss_image`). A test-only fsync delay
(`set_fsync_delay`) and a pause point in `CommitRounds` keep the fsync-sharing
and poisoning tests independent of the machine. Each log keeps its own tests
on top, and `db/power_loss_tests.rs` runs the main WAL through power losses
and repeated `SIGKILL`s.
