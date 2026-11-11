//! `NamespaceSchema`: one namespace's field indexes — each field's id, name,
//! value type and `dropped` flag — as kept in the system catalog.

use std::collections::HashMap;

use crate::error::{KVError, Result};
use crate::field_index::{FieldId, FieldMeta, IndexValueType};

/// The schema of one namespace: its registered field indexes, by name.
#[derive(Debug, Clone, Default)]
pub struct NamespaceSchema {
    fields: HashMap<String, FieldMeta>,
}

/// What [`NamespaceSchema::register_field`] changed.
pub(crate) enum Registered {
    /// The field was live already: nothing changed.
    Unchanged(FieldId),
    /// A new field, or a dropped one made live again: its record must be saved.
    Changed(FieldMeta),
}

impl NamespaceSchema {
    /// Add a field as read from the catalog.
    pub(crate) fn insert(&mut self, meta: FieldMeta) {
        self.fields.insert(meta.field_name.clone(), meta);
    }

    /// Register an indexed field. A new field takes its id from `new_id`.
    ///
    /// **Idempotent**: a live field with the same name and type keeps its id
    /// and changes nothing. A name registered with a *different* type is an
    /// error.
    ///
    /// Re-registering a **dropped** field clears its `dropped` flag and reuses
    /// its [`FieldId`]: the doc store keys a field's job records by its id, so
    /// a re-added field carries on from its own records. The caller rebuilds
    /// the index; the flag only records that the field is live again.
    pub(crate) fn register_field(&mut self, field_name: &str, field_type: IndexValueType, new_id: impl FnOnce() -> FieldId) -> Result<Registered> {
        if let Some(existing) = self.fields.get_mut(field_name) {
            if existing.field_type != field_type {
                return Err(KVError::Serialization(format!(
                    "Field '{}' is already registered as {:?}, cannot re-register as {:?}",
                    field_name, existing.field_type, field_type
                )));
            }
            if !existing.dropped {
                return Ok(Registered::Unchanged(existing.field_id));
            }
            existing.dropped = false;
            return Ok(Registered::Changed(existing.clone()));
        }
        let meta = FieldMeta {
            field_id: new_id(),
            field_name: field_name.to_string(),
            field_type,
            dropped: false,
        };
        self.fields.insert(field_name.to_string(), meta.clone());
        Ok(Registered::Changed(meta))
    }

    /// Mark a field's index as dropped; `None` if no live field has `field_id`.
    ///
    /// The [`FieldMeta`] is kept, so a later
    /// [`register_field`](Self::register_field) reuses the same [`FieldId`].
    /// Saving the flag is what makes a drop completable after a crash — see
    /// `FEATURE-REQUEST.md` (FR-001).
    pub(crate) fn mark_field_dropped(&mut self, field_id: FieldId) -> Option<FieldMeta> {
        let f = self.fields.values_mut().find(|f| f.field_id == field_id && !f.dropped)?;
        f.dropped = true;
        Some(f.clone())
    }

    /// The ids of every field whose index has been dropped but whose
    /// definition is kept, sorted ascending.
    ///
    /// Used at open to finish deleting the on-disk directories of drops that
    /// were interrupted by a crash.
    pub fn dropped_field_ids(&self) -> Vec<FieldId> {
        let mut ids: Vec<FieldId> = self.fields.values().filter(|f| f.dropped).map(|f| f.field_id).collect();
        ids.sort();
        ids
    }

    /// Look up a field by name — the primary query-time access pattern, O(1).
    pub fn get_field_by_name(&self, field_name: &str) -> Option<&FieldMeta> {
        self.fields.get(field_name)
    }

    /// Look up a field by ID — used for index path resolution, O(n).
    pub fn get_field(&self, field_id: FieldId) -> Option<&FieldMeta> {
        self.fields.values().find(|f| f.field_id == field_id)
    }

    /// Return all fields sorted by `FieldId`.
    pub fn list_fields(&self) -> Vec<FieldMeta> {
        let mut fields: Vec<FieldMeta> = self.fields.values().cloned().collect();
        fields.sort_by_key(|f| f.field_id);
        fields
    }

    /// Return all registered `FieldId`s sorted ascending, **including dropped
    /// fields** (whose definitions are kept for id reuse).
    pub fn field_ids(&self) -> Vec<FieldId> {
        let mut ids: Vec<FieldId> = self.fields.values().map(|f| f.field_id).collect();
        ids.sort();
        ids
    }

    /// Return the `FieldId`s of every field whose index has *not* been dropped,
    /// sorted ascending.
    pub fn live_field_ids(&self) -> Vec<FieldId> {
        let mut ids: Vec<FieldId> = self.fields.values().filter(|f| !f.dropped).map(|f| f.field_id).collect();
        ids.sort();
        ids
    }
}
