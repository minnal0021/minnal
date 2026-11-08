//! `StoreState`: where a store is in its lifecycle.
//!
//! ```text
//! Creating ──► Created ──► Deleting ──► (gone)
//! ```
//!
//! A store is visible from the moment its record is written (`Creating`), but
//! takes no requests until every namespace and index it needs exists
//! (`Created`). A drop moves it to `Deleting`, which again refuses every request
//! except the admin views, while its data is removed; its record goes last,
//! which is the moment the store is gone.
//!
//! The state is lifecycle metadata, not part of the schema a user writes: it is
//! kept as a `"state"` key in the store's record in `system_schema`, beside the
//! schema's own fields, and only `SchemaStore` reads or writes it.

use serde::{Deserialize, Serialize};

use crate::doc_store::error::DocStoreError;

/// Where a store is in its lifecycle. See the module docs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoreState {
    /// Its record is written; its namespaces and indexes are being set up.
    /// Refuses every request but the admin views; cannot be dropped yet.
    Creating,
    /// Ready: takes reads, writes, queries and searches.
    Created,
    /// Being dropped. Refuses every request but the admin views.
    Deleting,
}

impl StoreState {
    /// The JSON key the state is kept under in a store's record.
    pub(crate) const KEY: &'static str = "state";

    /// `Ok` if a store in this state takes reads and writes, else the error a
    /// request gets.
    pub(crate) fn admit(self, namespace: &str) -> Result<(), DocStoreError> {
        match self {
            Self::Created => Ok(()),
            Self::Creating => Err(DocStoreError::StoreCreating {
                namespace: namespace.to_owned(),
            }),
            Self::Deleting => Err(DocStoreError::StoreDeleting {
                namespace: namespace.to_owned(),
            }),
        }
    }

    /// The state as it appears in JSON (`"creating"`, …).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Creating => "creating",
            Self::Created => "created",
            Self::Deleting => "deleting",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_created_store_takes_requests() {
        assert!(StoreState::Created.admit("s").is_ok());
        assert!(matches!(StoreState::Creating.admit("s"), Err(DocStoreError::StoreCreating { .. })));
        assert!(matches!(StoreState::Deleting.admit("s"), Err(DocStoreError::StoreDeleting { .. })));
    }

    #[test]
    fn the_json_names_match_as_str() {
        for s in [StoreState::Creating, StoreState::Created, StoreState::Deleting] {
            assert_eq!(serde_json::to_value(s).unwrap(), serde_json::json!(s.as_str()));
        }
    }
}
