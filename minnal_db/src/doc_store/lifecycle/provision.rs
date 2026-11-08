//! Setting a store up while it is `Creating`: its namespace, its field indexes
//! and, with semantic search, its vector namespaces and first partition. Then
//! it moves to `Created`.
//!
//! Every task is idempotent, so a run cut short (an error, a crash, a restart
//! of the task) is finished by running the whole list again from the top. No
//! request reaches the store meanwhile: it takes none until it is created.

use super::super::*;
use crate::namespace::NamespaceId;

impl DocStore {
    /// Run the creation tasks of `namespace`, whose record is `Creating`, and
    /// move it to `Created`. Does nothing for a store in any other state, or
    /// none: it was created already, or is gone.
    pub(in crate::doc_store) async fn provision(&self, namespace: &str) -> Result<(), DocStoreError> {
        #[cfg(test)]
        self.inject_lifecycle_fault(namespace)?;
        let _lifecycle = self.lifecycle.lock(namespace).await;
        if self.schemas.state(namespace) != Some(StoreState::Creating) {
            return Ok(());
        }
        let ns_id = self.provision_namespace(namespace).await?;
        let semantic = match self.schemas.get(namespace) {
            Some(StoredSchema::Doc(schema)) => {
                activate_indices(&self.db, ns_id, &schema).await?;
                schema.semantic_search_enabled.then(|| schema.vector_settings()).transpose()?
            }
            Some(StoredSchema::Kv(schema)) => schema.is_semantic_search_enabled().then(|| schema.vector_settings()).transpose()?,
            None => return Ok(()),
        };
        if let Some(settings) = semantic {
            self.seed_vector_index(namespace, &settings, &mut None).await?;
        }
        self.schemas.transition(namespace, &[StoreState::Creating], StoreState::Created)?;
        info!("store '{namespace}' created (ns_id={ns_id})");
        Ok(())
    }

    /// The store's namespace: the one an earlier run created (its id is in the
    /// record), or a new one, whose id the record then takes. A namespace of
    /// that name with another id was left by a run cut short before it could
    /// record the id; it is empty (the store took no writes), so it goes.
    async fn provision_namespace(&self, namespace: &str) -> Result<NamespaceId, DocStoreError> {
        let recorded = self.schemas.get(namespace).and_then(|s| s.ns_id());
        if let Some(existing) = self.db.existing_namespace(namespace.to_owned()).await?
            && Some(existing.id()) == recorded
        {
            return Ok(existing.id());
        }
        self.clear_orphaned_namespace(namespace).await?;
        let ns_id = self.db.namespace(namespace.to_owned()).await?.id();
        match self.schemas.get(namespace) {
            Some(StoredSchema::Doc(_)) => self.schemas.update_doc(namespace, |s| {
                s.ns_id = Some(ns_id);
                Ok(())
            })?,
            Some(StoredSchema::Kv(_)) => self.schemas.update_kv(namespace, |s| {
                s.ns_id = Some(ns_id);
                Ok(())
            })?,
            None => {}
        }
        Ok(ns_id)
    }

    /// Deal with a namespace called `name` that has no store, before a store
    /// takes the name. An empty one goes: it is all a create cut short leaves,
    /// and it may hold field-index registrations the new store must not
    /// inherit. One holding keys is refused with
    /// [`DocStoreError::NamespaceNotEmpty`]: it was written directly through
    /// [`Db`](crate::Db), and those keys are not this store's to delete or
    /// adopt. The default namespace always exists, so it is left alone.
    pub(super) async fn clear_orphaned_namespace(&self, name: &str) -> Result<(), DocStoreError> {
        self.refuse_a_namespace_holding_keys(name).await?;
        if name == crate::namespace::DEFAULT_NAMESPACE {
            return Ok(());
        }
        if self.db.existing_namespace(name.to_owned()).await?.is_some() {
            log::warn!("removing empty namespace '{name}', left without a store, before creating a store with its name");
            self.db.remove_namespace(name.to_owned()).await?;
        }
        Ok(())
    }

    /// Test hook: fail (or panic in) this run of `namespace`'s lifecycle task if
    /// a failure is set for it ([`DocStore::lifecycle_faults`]).
    #[cfg(test)]
    pub(in crate::doc_store) fn inject_lifecycle_fault(&self, namespace: &str) -> Result<(), DocStoreError> {
        let mut faults = self.lifecycle_faults.lock();
        let Some((left, panic)) = faults.get_mut(namespace) else { return Ok(()) };
        if *left == 0 {
            return Ok(());
        }
        *left -= 1;
        let panic = *panic;
        drop(faults);
        if panic {
            panic!("injected lifecycle panic for '{namespace}'");
        }
        Err(DocStoreError::Io(std::io::Error::other(format!(
            "injected lifecycle failure for '{namespace}'"
        ))))
    }

    /// [`DocStoreError::NamespaceNotEmpty`] if a namespace called `name` holds
    /// keys. Checked before a create writes anything, so the caller hears it.
    pub(super) async fn refuse_a_namespace_holding_keys(&self, name: &str) -> Result<(), DocStoreError> {
        if name == crate::namespace::DEFAULT_NAMESPACE {
            return Ok(());
        }
        let Some(existing) = self.db.existing_namespace(name.to_owned()).await? else {
            return Ok(());
        };
        if existing.scan(None, None, 1).await?.0.is_empty() {
            Ok(())
        } else {
            Err(DocStoreError::NamespaceNotEmpty { namespace: name.to_owned() })
        }
    }
}
