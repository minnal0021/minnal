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
| [FR-001](#fr-001--capture-surface-and-remediate-field-index-replay-gaps) | Capture, surface and remediate field-index replay gaps | `minnal_db`, `minnal_db_api` | High | Proposed |
| [FR-002](#fr-002--api-authentication-tls-and-a-safe-bind-default) | API authentication, TLS, and a safe bind default | `minnal_db_api` | **Critical** | Proposed |
| [FR-003](#fr-003--surface-write-apply-failures-to-the-caller-of-put) | Surface write-apply failures to the caller of `put` | `minnal_db` | Medium | Proposed |

---

## FR-001 — Capture, surface and remediate field-index replay gaps

**Filed:** 2026-07-26
**Area:** `minnal_db` (WAL GC, index checkpoint), `minnal_db_api` (admin surface)
**Severity:** High — silent incomplete query results
**Related:** detection-only logging landed in `51ee29f` (see *What already exists*)

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

Item 3 is the substantive one: knowing *when* to re-index is the open question,
and it is what this request is really about.

### Scope

#### Must have

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

- **Row-scoped re-index.** Repair specific keys rather than the whole namespace.
  Viable when the gap's WAL range is still partially readable, or when an
  external process knows which documents are suspect. Much cheaper than a full
  rebuild on a large namespace.
- **Detection at the moment of loss.** Detection currently fires at the next
  `activate_field_index` (i.e. next open). Checking the index checkpoint
  watermark inside WAL GC would catch it when it happens and let the admin UI
  show it live.

#### Could have

- **Prevention: gate WAL GC on the index checkpoint.** Analysis below. Removes
  most occurrences, but does not remove the need for remediation (the backstop
  path still produces gaps by design).

### Design notes from the investigation

Recorded so the analysis is not repeated. None of it is implemented.

#### Prevention (deferred, not obviously worth it alone)

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

### Acceptance criteria

- A replay gap detected at open is recorded durably and still visible after a
  restart.
- The admin API reports outstanding gaps per namespace/field.
- An operator can trigger a re-index (full, and ideally row-scoped) that clears
  the gap record on success.
- Re-indexing does not re-put documents through the WAL.
- A namespace with no field indices is unaffected on every path.

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
