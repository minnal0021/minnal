//! `NamespaceState`: where a namespace is in its lifecycle.
//!
//! ```text
//! Creating ──► Created ⇄ IndexMaintenance
//!                 │
//!                 └──► Deleting ──► (gone)
//! ```
//!
//! A namespace is visible from the moment its record is written (`Creating`),
//! but takes no requests until every namespace and index it needs exists
//! (`Created`). While an index job runs (a field-index build, a reindex, a
//! vector cleanup; see `jobs`) it is in `IndexMaintenance`: it takes every
//! request as when created, but cannot be dropped, and goes back to `Created`
//! when its last job ends. A drop, only from `Created`, moves it to
//! `Deleting`, which again refuses every request except the admin views while
//! its data is removed; its record goes last, the moment it is gone.
//!
//! The state is lifecycle metadata, not part of the schema a user writes. The
//! layer that owns the namespace keeps it: the doc store keeps it as a
//! `"state"` key in the store's record in `system_schema`, beside the schema's
//! own fields.

use serde::{Deserialize, Serialize};

use crate::namespace_lifecycle::LifecycleError;

/// Where a namespace is in its lifecycle. See the module docs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NamespaceState {
    /// Its record is written; its namespaces and indexes are being set up.
    /// Refuses every request but the admin views; cannot be dropped yet.
    Creating,
    /// Ready: takes reads, writes, queries and searches.
    Created,
    /// Ready, with at least one index job running. Takes every request a
    /// created namespace takes, but cannot be dropped until its jobs end.
    IndexMaintenance,
    /// Being dropped. Refuses every request but the admin views.
    Deleting,
}

impl NamespaceState {
    /// The JSON key the state is kept under in a store's record.
    pub(crate) const KEY: &'static str = "state";

    /// Every state, for tests that go through them all.
    #[cfg(test)]
    pub(crate) const ALL: [Self; 4] = [Self::Creating, Self::Created, Self::IndexMaintenance, Self::Deleting];

    /// Whether a namespace in this state takes reads, writes and index jobs.
    pub fn is_ready(self) -> bool {
        matches!(self, Self::Created | Self::IndexMaintenance)
    }

    /// `Ok` if a namespace in this state takes reads and writes, else the
    /// error a request gets.
    pub(crate) fn admit(self, namespace: &str) -> Result<(), LifecycleError> {
        match self {
            Self::Created | Self::IndexMaintenance => Ok(()),
            Self::Creating => Err(LifecycleError::Creating {
                namespace: namespace.to_owned(),
            }),
            Self::Deleting => Err(LifecycleError::Deleting {
                namespace: namespace.to_owned(),
            }),
        }
    }

    /// The error a move from this state to `to` gets when this state is not
    /// one it may start from: the state's own refusal where it has one (a
    /// namespace being created or deleted, or with index jobs running, for a
    /// drop), else [`LifecycleError::InvalidTransition`].
    pub(crate) fn refusal(self, namespace: &str, to: Self) -> LifecycleError {
        let namespace = namespace.to_owned();
        match self {
            Self::Creating => LifecycleError::Creating { namespace },
            Self::Deleting => LifecycleError::Deleting { namespace },
            Self::IndexMaintenance if to == Self::Deleting => LifecycleError::InMaintenance { namespace },
            from => LifecycleError::InvalidTransition { namespace, from, to },
        }
    }

    /// The state as it appears in JSON (`"creating"`, …).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Creating => "creating",
            Self::Created => "created",
            Self::IndexMaintenance => "index_maintenance",
            Self::Deleting => "deleting",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_created_store_and_one_in_maintenance_take_requests() {
        assert!(NamespaceState::Created.admit("s").is_ok());
        assert!(NamespaceState::IndexMaintenance.admit("s").is_ok());
        assert!(matches!(NamespaceState::Creating.admit("s"), Err(LifecycleError::Creating { .. })));
        assert!(matches!(NamespaceState::Deleting.admit("s"), Err(LifecycleError::Deleting { .. })));
        for s in NamespaceState::ALL {
            assert_eq!(s.is_ready(), s.admit("s").is_ok(), "{s:?}");
        }
    }

    #[test]
    fn a_drop_during_maintenance_is_refused_as_such() {
        let drop = |from: NamespaceState| from.refusal("s", NamespaceState::Deleting);
        assert!(matches!(drop(NamespaceState::IndexMaintenance), LifecycleError::InMaintenance { .. }));
        assert!(matches!(drop(NamespaceState::Creating), LifecycleError::Creating { .. }));
        assert!(matches!(
            NamespaceState::IndexMaintenance.refusal("s", NamespaceState::Creating),
            LifecycleError::InvalidTransition { .. }
        ));
    }

    #[test]
    fn the_json_names_match_as_str() {
        for s in NamespaceState::ALL {
            assert_eq!(serde_json::to_value(s).unwrap(), serde_json::json!(s.as_str()));
        }
    }
}
