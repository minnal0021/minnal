//! Field indexes: per-field bitmap indexes and the query language over them.
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
//! storage layer; the query DSL in `crate::field_index::query` evaluates parsed predicates
//! directly against it.
//!
//! # Quick start
//!
//! ```rust
//! use minnal_db::field_index::{FieldIndex, Predicate};
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

pub mod bitmap;
pub mod blob_store;
pub mod dyn_index;
pub mod overlay_budget;
pub(crate) mod predicate;
pub mod query;
pub mod rowmap;
pub(crate) mod typed_index;
pub mod value;

pub use self::bitmap::RoaringBitmap;
pub use self::bitmap::shared::SharedRoaringBitmap;
pub use self::dyn_index::{DynFieldIndex, IndexBlobStats};
pub use self::overlay_budget::{IndexOverlayBudget, IndexOverlayStats};
pub use self::predicate::Predicate;
pub use self::rowmap::RowMap;
pub use self::typed_index::FieldIndex;
pub use self::value::{IndexValue, IndexValueType};

/// Identifier for a registered field index.
pub type FieldId = u32;
