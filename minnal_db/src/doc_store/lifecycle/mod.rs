//! Store lifecycle — create / list / drop for document and KV stores — and
//! schema amendment. A store's vector index has its own: `vector/lifecycle.rs`.

mod amend;
mod create;
mod drop;
#[cfg(test)]
mod tests;

pub(super) use drop::finish_interrupted_drops;
