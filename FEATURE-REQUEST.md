# minnal — feature requests

Open feature requests, newest at the bottom. Items here are **deliberately not
bug fixes**: each needs a product decision, a new mechanism, or an API change
that is not the engine's to make unilaterally. Straightforward defects are fixed
in place and tracked in git history instead.

Each entry records the design analysis behind it so the reasoning is not
re-derived from scratch later. Code references were verified at filing time;
re-check them before starting work.

| # | Title | Area | Severity | Status |
|---|---|---|---|---|
| [FR-001](#fr-001--prevent-capture-surface-and-remediate-field-index-replay-gaps) | Prevent, capture, surface and remediate field-index replay gaps | `minnal_db`, `minnal_db_api` | High | **Planned — 2026-08-01** |
| [FR-002](#fr-002--api-authentication-tls-and-a-safe-bind-default) | API authentication, TLS, and a safe bind default | `minnal_db_api` | **Critical** | Proposed |
| [FR-003](#fr-003--surface-write-apply-failures-to-the-caller-of-put) | Surface write-apply failures to the caller of `put` | `minnal_db` | Medium | Proposed |
| [FR-004](#fr-004--let-the-api-server-talk-to-the-engine-directly) | Let the API server talk to the engine directly | `minnal_db_api`, `minnal_db` | Low | Proposed |

---

## FR-001 — Prevent, capture, surface and remediate field-index replay gaps

**Filed:** 2026-07-26
**Updated:** 2026-07-31 — prevention promoted from *Could have* to the first
**Must have**, and full remediation scheduled alongside it. Scope is now
*prevent **and** repair*, not repair alone.
**Area:** `minnal_db` (WAL GC, index checkpoint), `minnal_db_api` (admin surface)
**Severity:** High — silent incomplete query results
**Related:** detection-only logging landed in `51ee29f` (see *What already
exists*). Unblocks the persisted row-map slot table (see *Downstream: the row
map* below).

### Summary

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

Measured severity (`wal_gc_can_strand_a_field_index_checkpoint_and_the_gap_is_detectable`,
reproduced through ordinary writes → flush → GC with nothing hand-corrupted):
10 WAL segments reclaimed, only **15 of ~360** writes in the replay window still
replayable.

### Why this is a feature, not a bug fix

The engine can *detect* the condition cheaply (that part has landed), but
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

Item 3 is the substantive one: knowing *when* to re-index is the open question.

**Updated 2026-07-31:** prevention (gating WAL GC on the index-replay watermark)
now leads the work, which shrinks that open question rather than answering it —
after prevention, the only gaps left come from the deliberate backstop path, so
"when to re-index" narrows to "when the backstop fired". The remediation arms are
still required; see *Implementation order*.

### Scope

#### Must have

- **Prevention: gate WAL GC on the index-replay watermark.** WAL GC must not
  reclaim a segment the index checkpoint still needs. Full design in *Prevention*
  below. This is the arm that removes the condition rather than reporting it:
  once it lands, everything below becomes the rare backstop path instead of the
  normal one. (Sequenced second, after dropped-index cleanup — see
  *Implementation order*.)
- **Dropping an index must delete its files and release the WAL.** Today
  `deactivate_field_index` only deregisters the in-memory index; the whole
  `index/{ns_id}/{field_id}/` subtree survives — bitmap blob, keymap **and a
  frozen checkpoint marker**. That is both a disk leak (reclaimed only when the
  entire namespace is dropped) and the thing that would poison the watermark
  above. Dropping a field must reclaim its directory and stop it holding WAL GC.
  Design in *Dropped-index cleanup* below.
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

#### Should have

- *(**Row-scoped re-index** was here; promoted to Must have on 2026-07-31 — it is
  now the chosen repair mode. See* Product decisions *below.)*
- *(**Detection at the moment of loss** was here; promoted to Must have on
  2026-07-31 — row-scoped repair depends on it. See* Product decisions *below.)*

#### Could have

- *(Prevention was here until 2026-07-31; it is now the first Must have.)*

### Design notes from the investigation

Recorded so the analysis is not repeated. None of it is implemented.

#### Prevention (promoted to Must have 2026-07-31 — full design)

Hold WAL GC at an **index-replay watermark** and refuse to reclaim segments at or
above it:

```text
index_replay_watermark = min(checkpoint offset) over currently-active fields
```

##### A watermark, not a per-entry flag

The obvious framing is "add an *index updated* flag next to *persisted* on each
WAL entry". That is more machinery than the problem needs. Every field already
records the WAL offset its persisted index reflects — the `checkpoint` file that
`IndexManager::read_checkpoint_state` reads. GC only has to compare a segment id
against the watermark's segment: O(1) per segment, no new per-entry state and no
new per-segment counters alongside `segment_total` / `segment_persisted`.

##### The trap: active fields only

The watermark must be scoped to **currently active** fields, not
`registry.all_indexed_fields()`. Two ways an inactive field poisons it:

- **Dropped indices.** `deactivate_field_index` deliberately leaves their
  checkpoint files on disk, so their markers are frozen at whenever they were
  last checkpointed. Pinning on those grows the WAL without bound — trading
  silent divergence for a disk-full outage.
- **A field being built for the first time.** No marker, so it reads as "replay
  from 0" and would pin the entire WAL. But a new index is populated by a
  *build*, not by WAL replay. `detect_replay_gap` already suppresses exactly this
  case (`CheckpointState::Absent` + empty index); the watermark must apply the
  same suppression, or every `add_index` on an established database pins
  everything.

Active means present in `store.namespace_index` — the same test
`run_index_checkpoint` uses to build its `active_fields` list.

##### Dropped-index cleanup (root fix for the frozen marker)

Filtering the watermark to active fields is a read-side guard. The write-side
root cause is that dropping an index leaves everything behind:

`deactivate_field_index` (`database.rs`) is a two-line in-memory deregister. It
does not touch disk, and `IndexManager` has `remove_namespace_path` (whole
namespace) but **no per-field removal**. So `index/{ns_id}/{field_id}/` — bitmap
blob store, keymap store, `checkpoint` marker — survives a drop indefinitely,
reclaimed only when the entire namespace is dropped.

Required: dropping a field index deletes its directory, so there is no stale
marker to pin WAL GC and no orphaned blob to leak. Add
`IndexManager::remove_field_path(ns_id, field_id)` alongside the existing
namespace-level call.

**Ordering is load-bearing — deregister first, delete files second.** This
mirrors `remove_namespace`, which persists the registry deletion before touching
any file. Taken the other way round, a crash between "files deleted" and
"registry updated" leaves a field that is still registered, still activated at
the next open, and now finds no checkpoint and no data — which reads as
`Absent` + empty index, exactly the state `detect_replay_gap` **suppresses** as a
normal first build. The index would come up silently incomplete: precisely the
failure this FR exists to remove. Registry-first instead leaves orphaned files
after a crash, which is a bounded disk leak and nothing worse.

Keep the active-fields filter on the watermark even once cleanup lands. A field
can be inactive without being dropped, and cleanup can itself be interrupted;
the filter is what makes the watermark correct rather than merely tidy.

##### Liveness

WAL GC runs every 60 s; the index checkpoint every 15 min (`DEFAULT_CHECKPOINT_INTERVAL`).
Pinning naively would hold up to 15 minutes of WAL at all times.

- **Normal path:** when GC finds segments blocked by the pin, fire the existing
  `IndexCheckpointTrigger` (already wired and debounced for blob-store
  backpressure) and defer those segments one tick. WAL retention then tracks
  *checkpoint latency* rather than the 15-minute timer. Needs one addition: an
  uncapped `request()` on the trigger, because `request_if_over_cap` returns
  early when `cap_bytes == 0` (valve disabled) and the pin must still drain.
- **Backstop:** cap pinned WAL by **segment count** (decided 2026-07-31 — simpler
  to reason about than a byte budget; revisit only if a configurable segment size
  makes the disk cost misleading). If the checkpoint worker is
  disabled or wedged, reclaim the oldest pinned segments anyway, record the gap,
  and let remediation heal it. **This is why prevention does not remove the need
  for the rest of this feature** — the backstop produces gaps by design, just
  rarely.

##### The row map comes along for free

`run_index_checkpoint` flushes each namespace's `RowMap` **before** any field
marker, both against the same `wal_tail`. So the row map's durable position is
always at or ahead of `min(field offsets)`, and pinning on the fields covers it.
No separate row-map watermark is needed.

Related cleanup: the row-map marker *writes* a `wal_offset` (`rowmap.rs`, bytes
32..40) that `read_marker` never reads back — it is write-only today. Either wire
it up as part of this work or drop the field.

#### Downstream: the row map

Prevention is the missing precondition for persisting the `RowMap` slot table.

The `key → id` slot table is currently anonymous memory rebuilt from the id array
on every open — O(N distinct keys ever written), and ~57–114 bytes of
unreclaimable RSS per key while open. Making it a file that can be *adopted* at
open (rather than rebuilt) requires trusting it up to the checkpoint and letting
WAL replay correct the tail. That is only sound if the WAL tail is guaranteed to
still exist — which is precisely what this watermark guarantees and what its
absence today makes unsafe.

So the ordering is: **prevention first, then the persisted slot table.** Without
the retention guarantee, adopting a slot file can hand back a row ID the id array
never durably assigned, which is silent bitmap/key divergence — the same class of
failure this FR exists to eliminate.

#### Remediation strategies considered

| Strategy | Trade-off |
|---|---|
| Rebuild synchronously at activation | Always correct, self-healing, no caller changes. `open` blocks on a full namespace scan — unbounded for a large store. |
| Fail activation with a distinct error | Engine stays simple and fast; `doc_store` drives repair with its existing resumable progress-observer machinery. Until it does, open hard-fails on the crash path. |
| Activate degraded + background rebuild | Fast open, visible resumable progress. Most moving parts, and the query path must honour the stale flag or it recreates the silent-wrong-answer problem it is meant to fix. |

#### Row-ID consistency (applies to any rebuild)

A single-field rebuild stays consistent with the namespace's other fields:

- `doc_store` namespaces register a key-derived `RowIdFn`, a pure function of the
  key, so IDs are stable by construction.
- Dense-`RowMap` namespaces: `get_or_alloc` returns the existing ID for known
  keys, and `run_index_checkpoint` flushes the row map **before** any field
  marker. So any key the restored row map lacks was written after that marker and
  cannot be referenced by another field's persisted bitmap.

### What already exists

Detection-only logging landed separately: at `activate_field_index`, if the WAL
segments covering `[checkpoint_offset, wal_tail)` are not all present, an `ERROR`
is logged naming the namespace, field and missing segments. It changes no
behaviour — replay still proceeds with whatever WAL survives — and it does not
persist, surface, or repair anything.

Note the check is segment-presence based, not `checkpoint_offset < wal_head`:
WAL GC reclaims fully-persisted segments **out of order**, and `scan_entries`
deliberately skips missing segments as holes (`wal.rs`), so a gap can open in the
middle of the replay window while `head` still sits below the checkpoint offset.
Any future work here should keep that in mind — comparing against `head` alone
under-reports.

### Product decisions (2026-07-31)

**A degraded index stays queryable.** Chosen over failing activation or blocking
open on a full rebuild. Fast open, and a partially-complete index is more useful
than none.

**Repair is row-scoped**, not whole-namespace. Bounded work proportional to the
damage rather than to the store.

Each decision drags a requirement with it. Neither is optional.

##### Queryable + degraded ⇒ the answer must say so

The strategy table below already flags this: *"the query path must honour the
stale flag or it recreates the silent-wrong-answer problem it is meant to fix."*
Serving a degraded index while returning results that look complete **is the
original bug**, just triggered by repair instead of by GC.

So a query touching a field with an outstanding gap must carry that fact back to
the caller. An operator polling the admin surface is not sufficient — the caller
holding an incomplete result set is the one who needs to know.

**Decided 2026-07-31: the engine returns it.** `query_keys` /
`query_keys_paginated` change from `Vec<Vec<u8>>` to a type carrying the keys
**and** the degraded status — roughly:

```rust
pub struct QueryOutcome {
    pub keys: Vec<Vec<u8>>,
    /// Fields referenced by the predicate that have an outstanding replay gap.
    /// Empty ⇒ results are complete.
    pub degraded_fields: Vec<FieldId>,
}
```

The alternative — leaving the signature alone and flagging degradation only in
the `doc_store` / REST response — was rejected. It would deliver the guarantee to
HTTP callers and leave **embedders** exactly where they started: `db.query_index(..)`
is a documented first-class path (it is the `QUICKSTART.md` field-index example),
and it would keep returning a complete-looking `Vec` over an incomplete index.
That is the same silence this FR exists to remove, relocated rather than fixed.

The cost is a public API change on a hot path, touching every call site. That is
accepted deliberately: the breakage is a **compile error**, not a silent
behaviour change, so it is paid once and then the status cannot be ignored by
construction.

Note the required field set is already computed where it is needed —
`query_keys` builds a `schema_map` and parses the predicate, so the fields a
query actually touches are known at exactly the point the gap records must be
consulted. Only fields referenced by *this* predicate count; an unrelated
degraded field elsewhere in the namespace must not mark a query degraded.

Propagation: `doc_store::query` / `query_resolved` carry it into their `Page`,
and `POST /stores/{ns}/query` surfaces it in the JSON response.

##### Row-scoped ⇒ detection must happen at the moment of loss

This is the sharp consequence. Row-scoped repair needs the list of affected keys
— and **those keys live in the very segments that were deleted**. The gap record
as designed holds a WAL *range* and missing segment ids, from which the affected
keys cannot be recovered. Detection at next open is therefore too late to ever
support row-scoped repair: by then the evidence is gone.

The only moment the keys are still knowable is inside WAL GC, immediately before
the backstop reclaims a pinned segment. So:

- When the backstop forces reclamation of a segment still needed by an active
  field, **scan that segment first** and record the distinct keys it holds into
  the gap record as a repair worklist, alongside the range and segment ids.
- Repair then iterates that worklist: read each key's current value from the
  store, run the field's extractor, insert the index entry — i.e.
  `update_indices_on_put` minus the storage write. Bounded by the damage.
- Clear the gap record once the worklist drains.

The extra scan is I/O, but only on the backstop path, which prevention makes rare
by construction.

**Bound the worklist.** A wedged checkpoint worker could strand a very large key
set. Cap the recorded worklist; past the cap, drop it and mark the field for a
full rebuild instead. Repair then has two modes — row-scoped (normal) and full
(cap exceeded) — and the gap record says which applies.

### Implementation order (2026-08-01)

Total remediation — prevention **and** repair, not prevention alone.

1. **Dropped-index cleanup.** `remove_field_path` + deregister-then-delete
   ordering. Independent of everything else and removes the frozen markers the
   watermark would otherwise trip over.
2. **Prevention.** Index-replay watermark gating WAL GC, active-fields scoped,
   `IndexCheckpointTrigger::request()` for liveness, backstop cap.
3. **Detection at the moment of loss + persisted gap record.** Scan a pinned
   segment before the backstop reclaims it, record range, segment ids **and the
   affected-key worklist** (capped). Must survive restart. This has to precede
   repair — it is what makes row-scoped repair possible at all.
4. **Admin surface + degraded-query signal.** Index health per namespace/field,
   and the query response indicator so a caller reading a degraded index knows
   its results are incomplete.
5. **Row-scoped re-index entry point.** Walk the worklist, re-extract and insert
   without re-putting documents; full-rebuild fallback when the cap was
   exceeded; clears the gap record on success.

Steps 1–2 make the condition rare; 3–4 make the remaining cases visible and
fixable. Shipping 1–2 alone would leave the backstop path silent, which is the
same failure mode in a smaller box — so this is not a stopping point.

Once this lands, the persisted row-map slot table becomes sound; see *Downstream:
the row map*.

### Acceptance criteria

- WAL GC does not reclaim a segment that any active field index still needs for
  replay, proven by the inverse of
  `wal_gc_can_strand_a_field_index_checkpoint_and_the_gap_is_detectable`.
- A dropped field index leaves no files under `index/{ns_id}/{field_id}/` and
  does not pin WAL GC; a crash mid-drop never yields a registered field whose
  data is gone.
- Pinned WAL is bounded: with the checkpoint worker stopped, retention stops at
  the backstop cap rather than growing without limit.
- A replay gap detected at open is recorded durably and still visible after a
  restart.
- The admin API reports outstanding gaps per namespace/field.
- A gap recorded by the backstop names the affected keys, and a restart preserves
  that worklist.
- An operator can trigger a row-scoped re-index that repairs only those keys and
  clears the gap record on success; the full-rebuild fallback is reachable when
  the worklist cap was exceeded.
- Re-indexing does not re-put documents through the WAL.
- A query against a field with an outstanding gap returns results **and** an
  indication that the index is degraded — never a complete-looking result set.
- A namespace with no field indices is unaffected on every path, and WAL
  retention is unchanged for it.

---

## FR-002 — API authentication, TLS, and a safe bind default

**Filed:** 2026-07-26
**Area:** `minnal_db_api`
**Severity:** Critical — unauthenticated destructive operations, reachable on all interfaces
**Source:** security audit 2026-07-18, item 1 (plus item 4, folded in)

### Summary

`minnal_db_api` wires every route with **no authentication middleware** and binds
to **all interfaces in plaintext** by default. Anyone who can reach the port has
full destructive control of the database.

Verified at filing:

- No auth or authorization layer anywhere in the router (`routes/mod.rs`) or
  server setup (`main.rs`) — the only `layer(..)` calls are tracing subscribers.
- `default_listen_addr()` returns `"0.0.0.0:8080"` (`config.rs:343`), and
  `config/sample.toml:43` ships the same.

Unauthenticated destructive routes include:

| Route | Effect |
|---|---|
| `DELETE /stores/{ns}` | drops an entire namespace and its on-disk data |
| `DELETE /stores/{ns}/indices/{field}` | drops a field index |
| `DELETE /stores/{ns}/indices/vector` | drops the vector index |
| `DELETE /admin/indices/{ns}/attribute/drop-all` | drops all attribute indices |
| `DELETE /admin/indices/{ns}/vector/drop-all` | drops all vector indices |
| `POST /admin/storage/gc`, `/gc/wal`, `/compact`, `/index-checkpoint` | forces expensive storage operations on demand |
| `POST /admin/stores/import` | imports schema |
| all `/stores/{ns}/docs/*` and `/stores/{ns}/kv/*` | full CRUD |

Plus every read route, so this is a data-disclosure exposure as well as a
destructive one.

### Why this is a feature, not a bug fix

Adding auth is a product decision, not a defect repair. It requires choosing a
trust model (who are the principals, how are credentials issued and rotated),
selecting a scheme, deciding whether TLS is terminated in-process or by a
proxy, and accepting a breaking change for existing deployments. None of those
are the engine's call.

### Scope

#### Must have

- **An authentication layer** on the router, with a documented trust model.
  Minimum viable: a shared API key or bearer token via middleware. At absolute
  minimum, gate `/admin/*`.
- **Bind to `127.0.0.1` by default.** Exposing on all interfaces should be an
  explicit opt-in, in both `default_listen_addr()` and `config/sample.toml`.
- **Gate `?skip_wal=true`.** It routes `PUT /stores/{ns}/kv/{key}` and
  `PUT /stores/{ns}/docs/{id}` to non-durable writes (`routes/kv.rs`,
  `routes/docs.rs`) and is intended for the bulk loader, but is currently
  unrestricted — any client can silently opt out of durability. Fold into the
  admin-auth boundary or move off the public path. *(security audit item 4)*
- **Document the trust model** in the API README, including what is assumed
  about the network the server sits on.

#### Should have

- **TLS**, or an explicit documented statement that termination is the
  deployment's responsibility and the listener must not be public.
- **Authorization tiers** — separating read, write and admin so a compromised
  application credential cannot drop namespaces.

#### Could have

- Per-namespace scoping of credentials.
- Audit logging of admin operations with principal identity.

### Acceptance criteria

- No route is reachable without authentication (or, if scoped down, no
  `/admin/*` route is).
- The default configuration binds to loopback only.
- `skip_wal` is unavailable to an unauthenticated caller.
- The trust model is written down.
- Existing deployments have a documented migration path — this is a breaking
  change by design.

---

## FR-003 — Surface write-apply failures to the caller of `put`

**Filed:** 2026-07-26
**Area:** `minnal_db` (public API)
**Severity:** Medium — API contract; no data loss
**Source:** engine correctness review 2026-07-25, item 4

### Summary

`put` can return `Ok(())` for a write that a subsequent `get` reports as absent.

The write path is: WAL append + fsync (durability barrier), then a best-effort
in-memory apply with bounded retry. `apply_with_retry` (`database.rs:688`) tries
`APPLY_RETRY_ATTEMPTS` (3) times, logs an `error!` on final failure, and returns
`bool`. `put_ns` bumps the `apply_failures` metric on that `false` and then
returns `Ok(())` regardless (`database.rs:760-773`); `delete_ns` mirrors it at
`:846`.

**The data is not lost** — it is durable in the WAL and replayed on the next
open. But between the failed apply and the next restart, the write is invisible
to reads while the caller has been told it succeeded.

`Db::put`'s public doc comment (`facade.rs:132`) is a bare one-liner — "Insert or
update a key-value pair" — and says nothing about this. The behaviour is only
described in internal comments.

*(Correcting the original audit note: `apply_failures` is **not** unconsumed. It
is exposed on the admin storage endpoint at `admin_storage.rs:238`. So an
operator can observe it; what is missing is an in-band signal for the caller.)*

### Why this is a feature, not a bug fix

The honest fix changes `put`'s return type — a breaking change to the crate's
primary API, affecting every caller including `doc_store` and the REST layer.
That deserves a deliberate decision about what the contract *should* be, not a
drive-by patch. There is also a real design question: whether "durable but not
yet readable" should be an error, a distinct success variant, or simply
documented.

### Scope

#### Must have

- **Document the actual contract** on `Db::put` / `Db::delete` and their async
  equivalents: what `Ok` guarantees (durability) and what it does not
  (read-back visibility). This is worth doing even if the return type never
  changes.

#### Should have

- **An in-band signal.** Options, in rough order of invasiveness:
  - a distinct success variant (e.g. `WriteOutcome::{Applied, DurableNotApplied}`)
  - an error variant, making the failure impossible to ignore
  - an opt-in strict mode where an apply failure is an error

#### Could have

- A **health signal** derived from `apply_failures` — a non-zero count means
  reads are serving stale state for some keys until restart, which is worth
  surfacing more prominently than a raw counter.
- Reconsider whether a persistent apply failure should trigger a
  flush/reopen of the affected store rather than waiting for a restart.

### Acceptance criteria

- The public docs state the durability-vs-visibility distinction.
- If the return type changes, `doc_store` and `minnal_db_api` are updated to
  handle the new outcome rather than discarding it.
- A test covers the failing-apply path and asserts the chosen contract.


---

## FR-004 — Let the API server talk to the engine directly

**Filed:** 2026-07-26
**Area:** `minnal_db_api`, `minnal_db` (`doc_store`)
**Severity:** Low — layering; no defect, no user-visible symptom
**Source:** engine correctness review 2026-07-25, item 5 (the half the
decomposition could not reach)

### Summary

`DocStore` carries roughly 25 one-line passthroughs to the underlying `Db` —
`db_stats`, `ops_metrics*`, `wal_metadata`, `lsm_manifests`, `lsm_runtime_stats`,
`value_log_*_stats`, `garbage_collect_all`, `garbage_collect_wal`, `compact`,
`checkpoint_index`, `index_blob_waste_threshold`, and others. They exist so the
API server can hold one handle, and they drag the whole engine diagnostic
surface up into the document layer.

That cuts against the project's own rule — *edit the lowest layer that owns it,
don't reach across modules* — because the API server reaches the engine
**through** `DocStore` rather than alongside it. `routes/admin_storage.rs` is
almost entirely calls of this shape.

They are now collected in `doc_store/store/diagnostics.rs` with a module doc
comment pointing here, so at least the coupling is visible in one place.

### Why this is a feature, not a bug fix

Removing a public method is a breaking change to the crate's API. The decision
is not the engine's to make unilaterally: it needs a view on who consumes
`minnal_db` besides this repo's own API server, and whether a major-version bump
is acceptable. The 2026-07-26 decomposition was carried out under an explicit
"no existing interface may break" constraint, which put this half out of reach
by construction.

### Scope

#### Must have

- **`AppState` holds an `Arc<AsyncDb>` alongside its `Arc<DocStore>`**, and the
  admin/diagnostic routes use it directly. `DocStore` already owns an
  `Arc<AsyncDb>`, so this is a wiring change at construction, not a second open.
- **Delete the passthroughs** once no route calls them, in one commit, with the
  breaking change called out in the changelog.

#### Should have

- An audit of the remaining `DocStore` surface for the same shape — the ~25
  counted here are the obvious ones, but `list_kv_namespaces` and
  `ttl_config_for_ns` are arguably in the same family.

#### Could have

- A deprecation cycle (`#[deprecated]` on the passthroughs for one release
  before removal) if there are downstream consumers to warn.

### Acceptance criteria

- No route in `minnal_db_api` reaches the engine through `DocStore`.
- `doc_store/store/diagnostics.rs` shrinks to the genuinely document-scoped
  operations (`count_docs`, the `field_index_*` helpers, `reindex_doc_*`), or
  disappears.
- The removal is a single, clearly-labelled breaking commit.
