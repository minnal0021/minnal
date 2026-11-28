# Collecting unused IVF centres

**Status:** proposed, for review. Implements FR-008 (`FEATURE-REQUEST.md`).
**Code it touches:** `semantic_search/vector_kv/partition.rs`,
`semantic_search/cluster/layout.rs`, a new `semantic_search/ivf_split/centre_gc.rs`,
`semantic_search/workers/ivf_maintenance.rs`, the config file, the partition
admin routes and minnal_ui's Vector Indexes tab.

## Summary

A store's IVF centres are never deleted, so they accumulate for the store's
whole life. This design removes the ones nothing uses any more, in two passes
separated by a quarantine:

1. A pass **marks** every centre it finds unused, recording when it did so.
2. A later pass, at least the quarantine after the mark, **scans again** and
   collects a marked centre only if it is still unused. A marked centre that is
   used again is unmarked and reported as an error, never deleted.

Each pass runs under conditions that make "unused" a fact rather than a guess
(the namespace's maintenance lock, an epoch barrier, no unfinished partition
operation). The second scan turns any mistake in that reasoning into a visible
alarm instead of chunks silently dropping out of search. The passes run in the
partition maintenance task, on a timer.

## Terms used here

- **Posting** — one IVF cluster: the chunk codes filed under it, keyed
  `posting_id ‖ doc_id` in `{ns}_sparse_vector`. A posting is *active* (searched
  and written to), *draining* (being emptied by a split, merge or rebuild) or
  *retired* (empty; kept as a small record for the split lineage and so its id
  is never reused).
- **Centre** — a vector a chunk code is encoded against. Each posting names one
  centre (`centre_id`); new chunks routed to a posting are encoded against that
  centre. Centres are stored one per row in `{ns}_ivf_centres`, keyed by their
  4-byte id.
- **Code** — one chunk's quantised vector (`VectorIndex`). It stores the
  `centre_id` it was encoded against and can only be decoded and scored
  through that centre.
- **Snapshot** — the in-memory partition (`NamespaceIvf`): postings, centres,
  counts. It is immutable; maintenance builds the next one and publishes it.
- **Routing epoch** — every vector write and every search runs inside it
  (`PartitionHandle::read_epoch`). An **epoch barrier**
  (`PartitionHandle::epoch_barrier`) waits until every write and search that
  entered before it has left.
- **Maintenance lock** — held by every split, merge, reassignment and rebuild
  of a namespace for its whole run (`PartitionHandle::lock_maintenance`).
- **Journal** — `{ns}_ivf_journal`, the record of a maintenance operation in
  progress; recovery finishes or abandons whatever it holds.
- **Candidate** — a centre that no posting other than a retired one names.
- **Unused** — a candidate that no code decodes against.

## Why centres accumulate

A split adds two centres and a rebuild one per posting it fits. When a split or
rebuild moves a chunk to another posting, the code keeps the `centre_id` it was
encoded against, because decoding it needs that centre. So a retired posting's
centre usually stays in use by the chunks that moved out of it. A centre
becomes unused only once every code encoded against it has been deleted or
re-embedded.

Each live centre costs about 6 KB of memory at 768 dimensions (its vector and
the rotated copy the codes are scored in) and one stored row. The partition is
loaded whole, so this grows the partition's memory and its load for good. On
FiQA, measured with `GET /admin/indices/{ns}/vector/centres?codes=true`, 23 of
1,893 centres were unused; a store with heavy delete or re-embed churn leaves
many more.

## What can make a centre used

Collection is safe exactly when no code that decodes against a centre can exist
or appear after the centre is judged unused. These are the only ways a code
with a given `centre_id` comes into being:

| Path | Which centres it uses |
|---|---|
| A document write (the vector worker) | The centre of the active posting the chunk is routed to, in the snapshot current when it routes. It routes and writes inside the routing epoch. |
| A split, merge, reassignment or rebuild moving a chunk | The centre the code already has: a move copies a code, it never re-encodes. |
| Recovery redoing a journalled operation | The same: it copies existing codes. |
| Vector index WAL replay after a crash | A code that was written before the crash. If it was in the store when a pass scanned, the pass saw it (scans read the memtable too); if its write was lost, the embed queue re-embeds the document through the first path. |
| Recovery abandoning a planned operation | Reactivates *draining* postings only; a retired posting is never active again (`ivf_split::abandon`). |

So after a posting retires, no new code is encoded against its centre, and
codes using it can only be copied, deleted or re-embedded. A pass can therefore
judge a centre unused if, while it scans:

1. **No code moves between postings.** A pass reads one posting at a time; a
   code moved from a posting it has not read yet to one it already read would be
   missed. Every move happens under the maintenance lock, so the pass holds it.
2. **No write is still using an older snapshot.** A write that routed while a
   posting was active can still be encoding against its centre after the
   posting retires. The pass runs an epoch barrier after taking the lock.
3. **No operation is unfinished.** Recovery of a journalled operation refers to
   the centre ids it recorded. The pass skips the namespace while the journal
   holds anything.
4. **It considers only candidates.** Every centre of an active or draining
   posting is kept: those are the only centres new writes and redo steps use.

## The design

### A centre's states

A centre is in one of three states, stored in its own row:

| State | Meaning | Row value | Length |
|---|---|---|---|
| live | in use, or not yet judged | the vector (`dim` little-endian `f32`s) | `4·dim` |
| marked | judged unused by a pass, awaiting confirmation | the vector ‖ `M` ‖ `marked_at_ms` (`u64` BE) | `4·dim + 9` |
| collected | removed; the row is kept so its id is never reused | `C` ‖ `collected_at_ms` (`u64` BE) | `9` |

A length one more than a multiple of four can never be a plain vector, and
`dim` is at least 8 (`MIN_EMBEDDING_DIM`), so the three are unambiguous; the
tag byte is checked as well. A malformed row fails the partition load, as it
does today.

Why the state lives in the row:

- There is no new namespace: nothing to add to the companion list, the store
  drop or `validate_store_name`, and the state goes when the store does.
- The state survives a restart, so a quarantine can span one.
- **A collected centre keeps its row, so its id is never handed out again.**
  New centre ids are taken above the highest id the snapshot holds
  (`NamespaceIvf::max_ids`); with the row deleted, collecting the highest centre
  would let the next split reuse its id, and a retired posting naming the old id
  would point at a different centre. A tombstone costs 13 bytes on disk and
  nothing in memory. Retired postings follow the same rule (strip the data,
  keep the row).

Every state change is one WAL-backed put of one row, durable when it returns
and harmless to repeat.

### A pass

One pass over one namespace:

1. Take the maintenance lock. If the journal holds a record, stop: the next
   interval tries again. Run an epoch barrier.
2. Candidates: every live or marked centre that no active or draining posting
   names. If there are none, stop without reading a code.
3. Scan every live posting's codes once (`compute_code_metrics`, the code-metrics
   pass, which also refreshes the code-metrics cache) and take the **unused**
   candidates: those no code decodes against.
4. For each **marked** centre:
   - **Used again** (not in the unused set): unmark it (rewrite the row without
     the trailer), log an ERROR naming the namespace, the centre and when it was
     marked, and count it. By the argument above this cannot happen; if it does,
     a path that makes codes was missed, and nothing was deleted.
   - **Still unused, and `now − marked_at ≥ quarantine`**: collect it. Write the
     tombstone row, then publish a snapshot without the centre.
   - **Still unused, quarantine not over**: leave it.
5. Mark every unused centre not yet marked, with the current time.
6. Release the lock and record the pass in the partition history (a new kind,
   `centre_gc`): candidates, newly marked, unmarked, collected, and live centres
   before and after.

In `mark` mode (below), step 4 never collects; everything else runs.

### Why the second scan, not just the time

The mark records that the centre was unused when it was marked. It says nothing
about the time since. Deleting a centre because its mark is old would rely on
the mark alone, so the quarantine adds safety only because the second pass scans
again: a centre is collected only after two scans, at least the quarantine
apart, both found it unused under the conditions above. The quarantine gives
anything that could still use a centre (a write held up for minutes, a path
nobody thought of) time to show itself before the second scan.

### Snapshot changes

- **The highest centre id is kept apart from the live centres.** `CentreSet`
  gains a high-water id: the highest id of any row, tombstones included, raised
  as centres are added and never lowered. `max_ids` uses it.
- **A snapshot without some centres.** `CentreSet` is append-only today
  (`evolve` refuses an existing id). It gains `without(ids)`, which builds the
  next set without those centres and keeps the high-water id. A search or write
  still holding the old snapshot keeps its centres (they are shared `Arc`s), so
  a search that started before the publish scores as before.
- **Loading.** `NamespaceIvf` requires the centre of every active and draining
  posting to exist, as now; a retired posting may name a collected centre. A
  marked centre loads as live (it is still needed if a pass unmarks it).
- **Readers of a retired posting's centre.** Partition and centre health,
  drift, the recall check and reassignment read either live postings or codes,
  and by the argument above codes only use centres that are not collected.
  Each is audited and tested with a collected centre present; one that meets a
  missing centre treats it as absent, never as an error that fails the request.
- **Search's safety net is unchanged:** a code whose centre is missing is
  skipped and counted in `sparse_corrupt_skipped`
  (`GET /admin/indices/vector/corruption-metrics`). After a correct collection
  that count does not move.

### Writes in order, and crashes

| Crash after | State on reopen | What happens next |
|---|---|---|
| a mark | the centre is marked | confirmed or unmarked by a later pass |
| an unmark | the centre is live | judged again by a later pass |
| a tombstone, before the publish | the row is collected; the load leaves the centre out | nothing to do |
| the publish | as above | nothing to do |

No journal record is needed: each step is a single durable row write, and every
reachable state is valid. `stored_centre`, which journal recovery uses to read
a centre by id, decodes all three states (a collected centre reads as absent).

### Where it runs

There is no vector GC worker: the engine's GC workers sit below
`semantic_search` and know nothing about centres. The **partition maintenance
task** (`workers/ivf_maintenance.rs`) is the natural home. It already works one
namespace at a time under that namespace's maintenance lock and limits how many
namespaces run at once (`maintenance.threads`). Today it wakes only when a write
leaves a posting over its split limit or under its merge limit. It gains a
timer: every `centre_gc_interval_secs` it queues each semantic namespace for a
pass. A namespace with no candidates costs one look at the snapshot.

An admin route runs one pass at once, for checking and for tests:
`POST /admin/indices/{ns}/vector/centres/collect` → the pass's counts.

### Configuration

Three keys in `[semantic_search.maintenance]`, each with a constant in
`defaults.rs`, a field, a mapping line, a row in the config report and an entry
in `config/sample.toml` (the config rules in the root `CLAUDE.md`):

| Key | Values | Proposed default |
|---|---|---|
| `centre_gc` | `off`, `mark` (mark and unmark, never collect), `collect` | `mark` |
| `centre_gc_interval_secs` | how often each namespace gets a pass | 3,600 |
| `centre_gc_quarantine_secs` | how long a mark must stand before its centre may be collected | 86,400 |

### What the admin surfaces show

- `GET …/vector/centres`: each centre's state (`live`, `marked` with
  `marked_at_ms`), and counts of marked and collected centres.
- `GET …/vector/partition`: live centres beside total centre ids, and the last
  pass (time and counts).
- `GET …/vector/history`: `centre_gc` records.
- minnal_ui's Centres and History tabs show the same, and an unmark (the
  "used again" alarm) is shown as an error.

### Cost

A pass that finds candidates reads every chunk code of the namespace once,
under the maintenance lock: about half a second for 170,000 chunks, so around
30 s for 10 million. While it runs, splits and merges of that namespace wait;
writes and searches do not (they need only the routing epoch, and the barrier
waits only for those already running). At an interval of an hour that is a
small share of the time. If it is ever too long, a variant can scan without the
lock and discard its result if any maintenance operation ran meanwhile (the
maintenance counters say so).

## Risks

| Risk | Consequence | What keeps it safe |
|---|---|---|
| A centre judged unused that a code still uses | That code's chunks drop out of search silently | The four conditions on every scan; a second scan at least the quarantine later must agree; a centre used again is unmarked and reported, never deleted; search counts any code without its centre |
| A retired posting names a collected centre | The partition fails to load and the store's search stops | Validation only for active and draining postings; a restart test after collection |
| A centre id handed out twice | A retired posting names a different centre | Tombstone rows keep the high-water id; a test collects the highest centre and then splits |
| A pass blocks splits for long | Postings grow past the split limit while it runs | Runs once per interval and only with candidates; the lock-free variant if needed |
| Collection is enabled before it is trusted | — | `mark` mode first: watch marks and any "used again" alarm before switching to `collect` |

## Tests

- **Mark, then collect:** a pass marks a centre whose codes were all deleted;
  a pass before the quarantine leaves it; a pass after collects it; the
  partition, a search and a restart all work afterwards.
- **Used again:** a code written against a marked candidate behind the rules'
  back (the simulated bug) makes the next pass unmark it, log the error and
  collect nothing.
- **A racing write:** a write routed to an active posting while a pass runs is
  scored correctly afterwards. Mutation check: removing "keep the centres of
  active and draining postings" makes it fail.
- **No id reuse:** collect the highest centre, split, and check the new
  centres' ids are above it, also after a restart.
- **Crashes:** a crash at every row write of a pass (`crash_point!`, as the
  vector crash audits do), then a reopen: the partition loads and later passes
  finish the job.
- **Unfinished operation:** a pass with a journal record does nothing.
- **Search quality:** nDCG at 10, 20, 30, 40, 50 and 100 unchanged on SciFact
  and FiQA after collection, against no collection (the frozen evaluation).

## Building it

1. The row codec and the loader (three states, high-water id, retired postings
   may name a collected centre), with `stored_centre` using the codec. Nothing
   writes the new states yet.
2. `CentreSet::without` and the pass (`ivf_split/centre_gc.rs`), driven by tests
   only.
3. The timer, the configuration and the admin route.
4. The admin surfaces, minnal_ui, and the history record.
5. A mark-only run on SciFact and FiQA, then the search-quality comparison
   with collection.

## Decisions for review

1. **Defaults.** `centre_gc = "mark"`, an interval of one hour and a quarantine
   of a day, until marks have been watched on real stores; or `collect` from the
   start with a quarantine of minutes.
2. **Where the state lives.** In the centre's row (this design), or in a
   separate table of marks.
3. **Collection at the end of a rebuild.** FR-008 asks for it. With a periodic
   pass a rebuild needs no special step, and a rebuild frees no centre anyway:
   the codes it moves keep their old centres. This design drops it.
