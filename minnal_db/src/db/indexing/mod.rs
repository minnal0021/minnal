//! Field indexes as the engine runs them; the data structures are in
//! `crate::field_index`. `manager` handles the on-disk directories, checkpoints
//! and gap records, `health` reports and repairs, and `query` runs predicates.

pub(crate) mod health;
pub mod manager;
pub(crate) mod query;
