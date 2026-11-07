//! Which field-index builds are running in this process, per store.
//!
//! A `reindex-all` drops every index of a store and re-adds it. Started while an
//! earlier build of the store is still running, it deleted that build's
//! directory under it and stopped part-way, with the indexes already removed from
//! the schema. It is refused instead while any build of the store runs, and a
//! field cannot be added again until its own last build has ended.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// The builds running now, keyed by store, then field.
#[derive(Default)]
pub(crate) struct RunningBuilds(parking_lot::Mutex<HashMap<String, HashSet<String>>>);

/// A running build of one field; the build ends when this drops.
pub(crate) struct BuildClaim {
    builds: Arc<RunningBuilds>,
    namespace: String,
    field: String,
}

impl RunningBuilds {
    /// Claim `field` of `namespace` for a build; `None` while one already runs.
    pub(crate) fn claim(self: &Arc<Self>, namespace: &str, field: &str) -> Option<BuildClaim> {
        let mut running = self.0.lock();
        if !running.entry(namespace.to_owned()).or_default().insert(field.to_owned()) {
            return None;
        }
        Some(BuildClaim {
            builds: Arc::clone(self),
            namespace: namespace.to_owned(),
            field: field.to_owned(),
        })
    }

    /// A field of `namespace` with a build running, if any.
    pub(crate) fn any(&self, namespace: &str) -> Option<String> {
        self.0.lock().get(namespace).and_then(|fields| fields.iter().next().cloned())
    }
}

impl Drop for BuildClaim {
    fn drop(&mut self) {
        let mut running = self.builds.0.lock();
        if let Some(fields) = running.get_mut(&self.namespace) {
            fields.remove(&self.field);
            if fields.is_empty() {
                running.remove(&self.namespace);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_field_is_claimed_once_until_its_build_ends() {
        let builds = Arc::new(RunningBuilds::default());
        let claim = builds.claim("s", "a").unwrap();
        assert!(builds.claim("s", "a").is_none());
        assert_eq!(builds.any("s").as_deref(), Some("a"));
        assert!(builds.any("other").is_none());
        drop(claim);
        assert!(builds.any("s").is_none());
        assert!(builds.claim("s", "a").is_some());
    }
}
