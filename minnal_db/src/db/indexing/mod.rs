//! Field indexes as the engine runs them; the data structures are in
//! `crate::field_index`. `index_fields` registers, activates and drops them,
//! `manager` handles the on-disk directories, checkpoints and gap records,
//! `health` reports and repairs, `query` runs predicates, and `row_filter`
//! turns a predicate's matches into a test on keys for a filtered search.

pub(crate) mod health;
mod index_fields;
pub mod manager;
pub(crate) mod query;
#[cfg(all(feature = "doc-store", feature = "semantic-search"))]
pub(crate) mod row_filter;
