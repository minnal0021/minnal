//! Store lifecycle — create / list / drop for document and KV stores — and
//! schema amendment. A store's vector index has its own: `vector/lifecycle.rs`.
//!
//! The states and the governor that runs creations and drops are the
//! `namespace_lifecycle` layer's; this is the work of each step, and `host`
//! is where the doc store plugs into the governor.

mod amend;
mod create;
mod drop;
mod host;
mod provision;
#[cfg(test)]
mod tests;
