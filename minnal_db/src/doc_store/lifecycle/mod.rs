//! Store lifecycle — create / list / drop for document and KV stores — and
//! schema amendment. A store's vector index has its own: `vector/lifecycle.rs`.
//!
//! The states and the governor that runs creations and drops are the
//! `namespace_lifecycle` layer's; this is the work of each step, and `host`
//! is where the doc store plugs into the governor. `jobs` starts and runs
//! index jobs and lists them; their bodies live with the indexes they work on
//! (`indexing/`).

mod amend;
mod create;
mod drop;
mod host;
mod jobs;
mod provision;
#[cfg(test)]
mod tests;

pub(in crate::doc_store) use amend::apply_amendment;
