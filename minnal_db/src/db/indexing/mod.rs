//! Field indexes as the engine runs them; the data structures are in
//! `crate::field_index`. `index_fields` registers, activates and drops them,
//! `manager` handles the on-disk directories, checkpoints and gap records,
//! `health` reports and repairs, and `query` runs predicates.

pub(crate) mod health;
mod index_fields;
pub mod manager;
pub(crate) mod query;
