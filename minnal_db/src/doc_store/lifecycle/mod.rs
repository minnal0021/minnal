//! Store lifecycle — create / list / drop for document and KV stores — and
//! schema amendment. A store's vector index has its own: `vector/lifecycle.rs`.

mod amend;
mod create;
mod drop;
mod provision;
#[cfg(test)]
mod tests;
