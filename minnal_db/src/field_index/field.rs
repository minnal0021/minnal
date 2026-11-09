//! A registered field and the results of working on it: the types shared by the
//! namespace registry, the store's live indexes and the engine's index upkeep.

use super::value::IndexValueType;

/// Unique identifier for an indexed field within a namespace.
/// Assigned monotonically; never reused after a field is removed.
pub type FieldId = u32;

/// Metadata for a single indexed field within a namespace.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldMeta {
    /// Unique field identifier within the namespace.
    pub field_id: FieldId,
    /// Human-readable field name (e.g. `"status"`, `"user_id"`).
    pub field_name: String,
    /// The value type this field was registered with.
    ///
    /// `activate_field_index` validates that the caller-supplied type matches
    /// this so mismatches are caught at activation time rather than at query time.
    pub field_type: IndexValueType,
    /// Whether this field's index has been dropped.
    ///
    /// A dropped field keeps its entry so a later re-add reuses the same
    /// [`FieldId`], but it is excluded from
    /// [`NamespaceRegistry::all_indexed_fields`](crate::namespace::NamespaceRegistry::all_indexed_fields) and cannot be activated until
    /// it is re-registered. See `FEATURE-REQUEST.md` (FR-001) —
    /// *Dropped-index cleanup*.
    pub dropped: bool,
}

/// The result of a field-index query: the matching keys **and** whether the
/// indices that produced them are known to be incomplete.
///
/// Returning a bare `Vec` of keys was the original defect in a different
/// costume: a degraded index served results that *looked* complete, so a caller
/// could not tell a genuinely empty result from a missing one. The status is
/// carried here rather than only at the REST boundary because
/// [`AsyncDb::query_index`](crate::AsyncDb::query_index) is a first-class documented path
/// — flagging only in `doc_store` would fix HTTP callers and leave embedders
/// exactly where they started.
///
/// See `FEATURE-REQUEST.md` (FR-001) — *Queryable + degraded ⇒ the answer must
/// say so*.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct QueryOutcome {
    /// The matching keys. For a paginated query, only the requested window.
    pub keys: Vec<Vec<u8>>,
    /// Total matching rows, i.e. the cardinality of the evaluated bitmap.
    ///
    /// May exceed `keys.len()`: a paginated query returns one window, and any
    /// query can hold a row whose key cannot be resolved back (a row map that
    /// has lost the entry).
    pub total: usize,
    /// Fields referenced by *this* predicate that have an outstanding gap
    /// record, so the results may be incomplete. Empty ⇒ complete.
    ///
    /// Only fields the predicate actually touched are listed — an unrelated
    /// degraded field elsewhere in the namespace does not taint this query.
    pub degraded_fields: Vec<FieldId>,
    /// Every field the predicate referenced, sorted and deduplicated. Lets a
    /// caller check other conditions per field — the document store flags a
    /// field whose index build is still running.
    pub touched_fields: Vec<FieldId>,
}

impl QueryOutcome {
    /// Whether any field this query touched has an outstanding gap, i.e. the
    /// results may be missing rows.
    pub fn is_degraded(&self) -> bool {
        !self.degraded_fields.is_empty()
    }
}

/// What a field-index repair did ([`AsyncDb::repair_field_index`](crate::AsyncDb::repair_field_index)).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum FieldRepairOutcome {
    /// The field had no outstanding gap; nothing to do.
    NotDegraded,
    /// The recorded worklist was replayed key by key.
    RowScoped {
        /// Keys in the worklist.
        keys_total: usize,
        /// Keys whose current value was re-extracted and re-indexed.
        reindexed: usize,
        /// Keys with no current value. Their rows were cleared — the key may
        /// have been deleted by the very write that went missing.
        absent: usize,
    },
    /// The whole field was rebuilt from current data, because the worklist was
    /// unavailable (over the cap, or never knowable for a no-WAL gap).
    FullRebuild {
        /// Keys scanned and re-extracted.
        scanned: usize,
    },
}

/// Outcome of a targeted single-field reindex ([`AsyncDb::reindex_field`](crate::AsyncDb::reindex_field)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldReindexOutcome {
    /// The field index entry for the key was re-derived from its current value
    /// and rewritten.
    Reindexed,
    /// The key has no stored value (absent or deleted), so there was nothing to
    /// index.
    KeyNotFound,
    /// No active index exists for the given field id in the namespace (it was
    /// never registered, or is not currently activated — e.g. still building).
    FieldNotActive,
}
