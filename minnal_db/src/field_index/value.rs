//! The typed values a field index holds.
//!
//! [`IndexValue`] and [`IndexValueType`] are the bridge between the caller's
//! document model and the typed [`FieldIndex<V>`](crate::field_index::FieldIndex).
//! The caller (DocStore or KVStore index hooks) extracts a field value from raw
//! bytes and returns an [`IndexValue`]; the database layer then dispatches to the
//! correct [`DynFieldIndex`](crate::field_index::DynFieldIndex) variant.

/// A single typed field value extracted from a document.
///
/// `Eq`/`Hash` are derived so callers can group rows by value before writing —
/// WAL replay does this to write each value's bitmap once instead of once per
/// row. Every variant is a plain hashable scalar, so there is no float-equality
/// hazard in deriving them.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IndexValue {
    Bool(bool),
    Int(i64),
    Str(String),
}

/// Discriminant for the value type stored in a [`DynFieldIndex`](crate::field_index::DynFieldIndex).
///
/// Used when creating a new field index so the correct variant is allocated
/// before any values are inserted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum IndexValueType {
    Bool,
    Int,
    Str,
}
