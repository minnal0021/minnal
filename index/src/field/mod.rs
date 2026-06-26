//! Index-level API for KV store query support.
//!
//! # Overview
//!
//! Two building blocks compose the query pipeline:
//!
//! | Type | Role |
//! |------|------|
//! | [`FieldIndex<V>`] | Single-field index: `BTreeMap<V, RoaringBitmap>` |
//! | [`Predicate<V>`] | Per-field filter (Eq, Ne, Lt, Le, Gt, Ge, Between, In) |
//!
//! A [`FieldIndex`] maps each distinct field value to a bitmap of row IDs, and
//! [`FieldIndex::evaluate`] turns a [`Predicate`] into the matching bitmap. The
//! type-erased [`DynFieldIndex`] wraps the per-type variants for the dynamic
//! storage layer; the query DSL in `crate::query` evaluates parsed predicates
//! directly against it.
//!
//! # Quick start
//!
//! ```rust
//! use index::field::{FieldIndex, Predicate};
//!
//! let mut age_idx = FieldIndex::<u32>::new();
//! age_idx.insert(25, 1);
//! age_idx.insert(30, 2);
//! age_idx.insert(25, 3);
//!
//! // rows whose age < 30
//! let rows = age_idx.evaluate(&Predicate::Lt(30));
//! assert_eq!(rows.cardinality(), 2); // rows 1 and 3
//! ```

/// Identifier for a registered field index.
pub type FieldId = u32;

pub(crate) mod field_index;
pub(crate) mod predicate;
pub mod value;

pub use field_index::FieldIndex;
pub use predicate::Predicate;
pub use value::{DynFieldIndex, IndexBlobStats, IndexValue, IndexValueType};
