# Namespace lifecycle

How a store (a document or KV store, one engine *namespace*) is created,
changed by index jobs, and dropped; what keeps those steps safe under
crashes, concurrent requests and failures; and where the code lives.

## Terms used here

- **Store** — a document or KV store of the doc store. Each has a *record*:
  its schema in the system namespace `system_schema`, which also holds its
  state.
- **State** — where the store is in its life: `creating`, `created`,
  `index_maintenance` or `deleting`. The state alone decides which requests
  the store takes.
- **Index job** — bounded, operator-started work on a store's indexes: build
  or drop a field index, rebuild all of them, turn semantic search on or off,
  re-queue every document for embedding, re-cluster the vector partition.
- **Governor** — one background task per doc store that runs every creation,
  drop and index job, and restarts what fails.
- **Host** — the layer that owns the stores and does each step's work (the doc
  store). The lifecycle layer (`src/namespace_lifecycle/`) is generic over it
  and never imports it.

## States

```text
Creating ──► Created ⇄ IndexMaintenance
                │
                └──► Deleting ──► (gone)
```

| State | Data requests | Admin views | Start an index job | Drop |
|---|---|---|---|---|
| `creating` | refused (`StoreCreating`, 409) | yes | refused | refused |
| `created` | yes | yes | yes → `index_maintenance` | yes → `deleting` |
| `index_maintenance` | yes | yes | yes, unless it conflicts | refused (`StoreInMaintenance`, 409) |
| `deleting` | refused (`StoreDeleting`, 409) | yes | refused | already dropping (202 again) |

Every move is a compare-and-set on the store's record under the schema
store's single writer (`SchemaStore::transition`): written through the WAL,
durable before the in-memory copy changes. A move the state does not allow
fails with the error the table names.

`index_maintenance` exists so that a drop cannot remove a store under a
running job, and so that a restart knows which stores have work to resume.
Continuous background work (embedding the queue, splitting and merging
postings, checkpoints) is not a job: a busy store must stay droppable.

## Creating and dropping a store

**Create.** The record is written first, in `creating`: from then on the
name is taken, and a second create of it is refused. The governor then runs
the store's *provision* task: create the engine namespace (refusing one that
already holds keys), activate the field indexes, and for a semantic store
create the vector namespaces and seed its partition. Last, the state moves to
`created`. `POST /stores` answers 202 once the record is durable.

**Drop.** The state moves to `deleting` first, durably: from that moment the
store refuses every request. The governor then runs the *teardown* task:

1. Wait for the writes already admitted (the write gate, below).
2. Stop partition maintenance on the store and wait out the searches and
   writes inside its partition.
3. Clear the store's embedding queue entries, remove its engine namespace,
   and clear the queue again (an entry the vector worker rewrote meanwhile).
4. Remove the vector namespaces, any directory left of them, and the field
   index directory.
5. Remove the store's job records, then its record, last.

The record is removed last because the engine's `namespace` call is
get-or-create: with the record gone first, a create of the same name could
reopen the old namespace with its documents.

Both tasks are idempotent: each step skips what is already done, so a task
cut short (an error, a panic, a crash) is finished by running it again. A
drop whose caller gives up (an HTTP client that disconnects) goes on: the
governor runs it, not the caller, so the store stays `deleting` until the
governor finishes it, and a retry waits for that.

## The governor

One task per doc store, not per store inside it, started when the doc store
opens: an embedded doc store and the API server's run the same governor, and
there is no way to open one without it. It takes tasks from a channel and
runs each in a coordinator task of its own while the task lasts; it does no
work itself, and costs nothing while no task runs.

- **One creation or drop of a store at a time, and nothing beside it.** Jobs
  of one store may run together, subject to the conflict table.
- **Supervision.** A task that fails or panics is run again after
  `restart_backoff`, up to `max_restarts` times (`[lifecycle]` in the server
  config, defaults 3 and 1 s); the restart count is kept in memory. Past
  that, a job is recorded `failed`; a creation or drop *parks* its store:
  a CRITICAL error is logged, the store stays in its state, and later tasks
  for it fail at once, until the process restarts.
- **The governor itself is supervised the same way.** If it gives up, its
  channel closes and every create, drop and job start fails with
  `GovernorUnavailable` (503). Stores that exist keep serving data requests.
- **At start** it picks up every store still `creating` or `deleting`, and
  every live job, so whatever a crash or a parked task left is finished. It
  finds them before the open returns, so a request that comes later is not
  mistaken for one of them. (The open itself first finishes a create or drop
  a crash cut short, before it opens any store's namespace; the governor
  runs one again only if that failed.)

The library's `create`, `remove`, `add_index` and the other plain forms hand
their task to the governor and wait for it to end; the `begin_*` forms return
once it is handed over.

## Index jobs

| Job | Started by (library / REST) | What the body does | Its saved progress | Cancel |
|---|---|---|---|---|
| `field_build` | `add_index` / `POST /stores/{ns}/indices` | Activate the index empty, list it in the schema, index every document | index spec, counts, resume key | yes: the half-built index is dropped |
| `field_drop` | `drop_index` / `DELETE /stores/{ns}/indices/{field}` | Demote the index to an attribute, then delete its files | — | no |
| `reindex_all` | `reindex_all_attribute_indices`, `drop_all_attribute_indices` / the attribute `reindex-all` and `drop-all` routes | Drop every index, reset the row map, list them again, build them one after the other | specs, phase, fields built, the current build | no: the old indexes are gone after its first step |
| `vector_enable` | `amend` turning semantic search on / `PATCH …/schema` | Seed the partition, then save the schema | the amendment | yes: what the seeding made is removed |
| `vector_cleanup` | `drop_vector_index`, `remove_attribute` of the last embedding field / `DELETE …/indices/vector` | Turn semantic search off (at the start), clear the queue, remove the vector namespaces | — | no: a half-removed vector index serves nothing |
| `vector_reindex` | `index_all`, `kv_index_all` / `POST …/vector/reindex-all` | Clear exhausted queue entries, enqueue every document a page at a time | counts, resume key | yes: stops at the next page |
| `partition_rebuild` | `start_vector_rebuild` / `POST …/vector/rebuild` | Re-cluster the partition from its codes | postings asked for, the highest posting id before, the report | until the plan is recorded |

Each `begin_*` library form returns once the job is handed to the governor
(the REST routes use them and answer 202 with the job's name); the plain forms
wait for the job to end.

**Record, then state.** A start writes the job's record (`JobRecord`, under its
`JobKey` in `system_namespace_maintenance`) and only then moves the store to
`index_maintenance`. An end writes the record's final status (`complete`,
`failed`, `cancelled`) and then moves the store back to `created` if no job of
it is live. One lock in the jobs registry covers every start and end across
both writes: without it an end could see no live job, a start write its record
and find the store still in maintenance, and the end then move it to `created`
with a job running. A crash between the two writes leaves a state that
disagrees with the records; at the next start the records win
(`Jobs::repair`).

**Progress lives in the record.** A job's body saves its own progress in its
record (`JobRecord.progress`, opaque to the lifecycle layer) and resumes from
it after a restart. A build saves its resume key only once an index checkpoint
covers it, so a resumed build never skips a document a crash took back out of
the index.

**Conflicts.** Two jobs that touch the same index cannot run together:

| Job | Refused while this runs |
|---|---|
| a field job on field *f* | another field job on *f*, or `reindex_all` |
| `reindex_all` | any field job, or another `reindex_all` |
| any vector job | any other vector job |

`vector_reindex` and `partition_rebuild` are the one vector pair that could
share a store safely, but `vector_enable` and `vector_cleanup` conflict with
both. A refused start answers `JobConflict` (409), naming both jobs.

**Cancel goes through the record.** `cancel_job` (`DELETE
/admin/stores/{ns}/jobs/{job}`) only marks the record `cancelling`. The body
reads that between batches, undoes its own work and returns, and the job ends
`cancelled`. The governor never reaches into a running body. A rebuild that has
recorded its plan runs to its end, and the cancel is withdrawn
(`Jobs::refuse_cancel`): the job ends `complete`.

**A rebuild found already started is not run again.** K-means takes minutes on
a large store, and repeating it after an error would repeat whatever caused
it. The partition's journal has already finished the rebuild (it got past its
publish) or abandoned it (it had not); the job is closed `complete` or
`failed` to match.

## Writes and drops: the write gate

A drop must not remove a store under a write that is in progress: the write
could recreate the namespace after its removal, or leave an embedding queue
entry the drop already cleared. Every document and KV write holds a guard
from the write gate for its whole duration, and is admitted only while the
store's state takes requests. The state check and the count are one step
under the gate's lock, and teardown reads the count under the same lock
after the move to `deleting`: a write either saw the old state and is counted,
so the drop waits for it, or sees `deleting` and is refused.

## The lifecycle audit log

Every create (`creating`, `created`), drop (`deleting`, `deleted`), job event
(`job_started`, `job_cancel_requested`, `job_completed`, `job_failed`,
`job_cancelled`), task a restart picked up (`resumed`), parked store
(`task_parked`) and governor stop (`governor_stopped`) is written to the TTL
namespace `system_lifecycle_audit`. Records are kept for `[lifecycle]
audit_retention_days` (default 30) and outlive the stores they describe. Keys
order the log newest first; `GET /admin/lifecycle/audit` reads it a page at a
time, for every store or one. A record that cannot be written is logged and
lost: an audit failure never fails the operation it describes.

## Where the code is

| Path | What it holds |
|---|---|
| `src/namespace_lifecycle/state.rs` | `NamespaceState`: what each state admits, which moves it refuses |
| `src/namespace_lifecycle/governor.rs` | The governor, its supervision and the start-up pick-up |
| `src/namespace_lifecycle/jobs/` | Job keys and records, the conflict table, the `Jobs` registry |
| `src/namespace_lifecycle/records/` | `system_namespace_maintenance` (`MaintenanceStore`) |
| `src/namespace_lifecycle/audit.rs` | The audit log |
| `src/namespace_lifecycle/host.rs` | `LifecycleHost`: what the lifecycle needs from the doc store |
| `src/doc_store/lifecycle/` | The doc store as host: create, provision, drop, teardown, job plumbing (`jobs.rs`) |
| `src/doc_store/indexing/` | The field-job bodies: `build/`, `reindex_all/` |
| `src/doc_store/vector/` | The vector-job bodies: `lifecycle.rs` (enable, cleanup), `reindex.rs`, `partition.rs` (rebuild) |
| `src/doc_store/write_gate.rs` | The write gate |

## Tests that pin the rules

| Rule | Test |
|---|---|
| Each state admits what the table says | `namespace_lifecycle::state` tests, `doc_store::lifecycle::tests::states` |
| A failed or panicking task runs again; past the limit it parks | `a_failed_or_panicked_task_is_run_again`, `a_failed_or_panicked_job_is_run_again` |
| A restart picks up creations, drops and jobs, and records it | `what_a_crash_left_is_picked_up_at_start` |
| The records win over a state a crash left behind | `the_governor_frees_a_store_left_in_maintenance_with_no_job` |
| Two drops at once remove the store once | `two_drops_at_once_remove_the_store_once` |
| No write outlives a drop | `writes_during_a_drop_leave_nothing_behind` |
| A cancel stops a build and drops the half-built index | `a_cancel_stops_a_running_build_and_drops_its_index` |
| A build, a reindex-all and a drop-all cut short at any step finish at the next start | `a_build_cut_short_is_finished_at_the_next_start`, `an_interrupted_reindex_all_is_finished_at_the_next_start`, `an_interrupted_drop_all_is_finished_at_the_next_start` |
| An enable is refused while a cleanup runs (FR-010) | `an_enable_is_refused_while_a_vector_cleanup_runs` |
| A rebuild cut short is closed with what the journal did | `a_rebuild_cut_short_is_closed_with_what_the_journal_did` |
| A store's life is in the audit log after it is gone | `a_stores_life_is_in_the_audit_log_after_it_is_gone` |
