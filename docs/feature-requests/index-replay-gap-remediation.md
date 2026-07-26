# Feature request: capture, surface and remediate field-index replay gaps

**Status:** proposed
**Filed:** 2026-07-26
**Area:** `minnal_db` (WAL GC, index checkpoint), `minnal_db_api` (admin surface)
**Related:** detection-only logging landed separately (see *What already exists* below)

## Summary

The field index is a derived structure kept durable by two independent
mechanisms — an inline in-memory update on every write, and a periodic
checkpoint that makes it durable and records the WAL offset it reflects. After a
crash the gap between the last checkpoint and the crash point is healed by
replaying the WAL from the checkpoint offset.

Nothing guarantees the WAL that heals it still exists. WAL GC reclaims a segment
once its entries are persisted **to the LSM**, and never consults the index
checkpoint. When the needed segments are gone the replay window silently
narrows, and the affected documents are absent from the field index **for the
lifetime of the database** — queries return incomplete results with no error.

Today the only remedy is to drop and re-add the index, which the operator has no
reason to do because nothing tells them the index is incomplete.

## Why this is a feature, not a bug fix

The engine can *detect* the condition cheaply (that part is being added now), but
detection only produces a log line. To actually remediate, minnal needs
mechanisms it does not currently have:

1. **A place to record the condition.** A log line is lost on restart and cannot
   be queried. There is no persisted "this index is incomplete" state.
2. **A way to surface it.** No admin API or UI reports index health, so an
   operator cannot discover the problem without reading logs.
3. **A re-index trigger.** `doc_store::rebuild_index_for_namespace` exists but is
   only reachable via `add_index`, is all-or-nothing, and **re-puts every
   document** — which re-amplifies the WAL. There is no "repair this index" or
   "re-index these rows" entry point.

Item 3 is the substantive one: knowing *when* to re-index is the open question,
and it is what this request is really about.

## Scope

### Must have

- **Persisted gap record.** When a replay gap is detected, record it durably
  alongside the field's checkpoint marker: namespace, field, the WAL range that
  could not be replayed, and when it was detected. Survives restart; cleared only
  by a successful re-index.
- **Admin surface.** Expose index health — at minimum, per field: last checkpoint
  offset, whether a gap is outstanding, and the detected range. Feeds an admin UI
  panel and makes the condition alertable.
- **Re-index entry point.** An operation that rebuilds a field index from current
  data **without re-putting documents** (extract-and-insert over a cursor scan,
  i.e. `update_indices_on_put` minus the storage write). Must clear the field
  index first — a crash can leave partially written post-checkpoint data that a
  merge-only pass would not remove.

### Should have

- **Row-scoped re-index.** Repair specific keys rather than the whole namespace.
  Viable when the gap's WAL range is still partially readable, or when an
  external process knows which documents are suspect. Much cheaper than a full
  rebuild on a large namespace.
- **Detection at the moment of loss.** Detection currently fires at the next
  `activate_field_index` (i.e. next open). Checking the index checkpoint
  watermark inside WAL GC would catch it when it happens and let the admin UI
  show it live.

### Could have

- **Prevention: gate WAL GC on the index checkpoint.** Analysis below. Removes
  most occurrences, but does not remove the need for remediation (the backstop
  path still produces gaps by design).

## Design notes from the investigation

Recorded so the analysis is not repeated. These were worked out while
investigating the finding; none of it is implemented.

### Prevention (deferred, not obviously worth it alone)

Hold WAL GC at an index-replay watermark: `min(read_checkpoint)` over fields, and
refuse to reclaim segments above it.

The trap: it must be scoped to **currently active** fields, not
`registry.all_indexed_fields()`. Registered-but-inactive fields — notably
*dropped* indices, whose checkpoint files `deactivate_field_index` deliberately
leaves on disk — have frozen markers. Pinning on those grows the WAL without
bound, trading silent divergence for a disk-full outage.

It also needs a liveness story, because the WAL GC interval (60 s) and the index
checkpoint interval (15 min) differ by 15×:

- Normal path: when GC finds segments blocked by the pin, fire the existing
  `IndexCheckpointTrigger` (already wired and debounced for blob-store
  backpressure) and defer those segments one tick. WAL retention then tracks
  checkpoint latency rather than the 15-minute timer.
- Backstop: cap pinned WAL by bytes/segments. If the checkpoint worker is
  disabled or wedged, reclaim anyway, record the gap, and let remediation heal
  it. **This is why prevention does not remove the need for this feature.**

### Remediation strategies considered

| Strategy | Trade-off |
|---|---|
| Rebuild synchronously at activation | Always correct, self-healing, no caller changes. `open` blocks on a full namespace scan — unbounded for a large store. |
| Fail activation with a distinct error | Engine stays simple and fast; `doc_store` drives repair with its existing resumable progress-observer machinery. Until it does, open hard-fails on the crash path. |
| Activate degraded + background rebuild | Fast open, visible resumable progress. Most moving parts, and the query path must honour the stale flag or it recreates the silent-wrong-answer problem it is meant to fix. |

### Row-ID consistency (applies to any rebuild)

A single-field rebuild stays consistent with the namespace's other fields:

- `doc_store` namespaces register a key-derived `RowIdFn`, a pure function of the
  key, so IDs are stable by construction.
- Dense-`RowMap` namespaces: `get_or_alloc` returns the existing ID for known
  keys, and `run_index_checkpoint` flushes the row map **before** any field
  marker. So any key the restored row map lacks was written after that marker and
  cannot be referenced by another field's persisted bitmap.

## What already exists

Detection-only logging is being added now, separately from this request: at
`activate_field_index`, if the WAL segments covering `[checkpoint_offset,
wal_tail)` are not all present, an `ERROR` is logged naming the namespace, field
and missing segments. It changes no behaviour — replay still proceeds with
whatever WAL survives — and it does not persist, surface, or repair anything.

Note the check is segment-presence based, not `checkpoint_offset < wal_head`:
WAL GC reclaims fully-persisted segments **out of order**, and `scan_entries`
deliberately skips missing segments as holes (`wal.rs`), so a gap can open in the
middle of the replay window while `head` still sits below the checkpoint offset.
Any future work here should keep that in mind — comparing against `head` alone
under-reports.

## Acceptance criteria

- A replay gap detected at open is recorded durably and still visible after a
  restart.
- The admin API reports outstanding gaps per namespace/field.
- An operator can trigger a re-index (full, and ideally row-scoped) that clears
  the gap record on success.
- Re-indexing does not re-put documents through the WAL.
- A namespace with no field indices is unaffected on every path.
