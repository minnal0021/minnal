//! `TaskKey`: which store a maintenance record belongs to, and which task it is.
//!
//! Encoded as the store's namespace id (16 bytes), a task tag (1 byte) and the
//! task's subject: the field id (4 bytes, big-endian) for a field-index build,
//! nothing otherwise. The namespace id comes first, so one store's records are a
//! prefix scan, and a store dropped and created again (which gets a new id)
//! never sees the old store's records. The id is random, so the leading bytes
//! vary and the records spread over the value-log buckets.

use crate::FieldId;
use crate::namespace::NamespaceId;

/// A maintenance record's key. See the module docs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct TaskKey {
    /// The store's namespace.
    pub store: NamespaceId,
    /// The task.
    pub task: Task,
}

/// Which task a record describes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum Task {
    /// Building one field's index (`DocStore::add_index`).
    FieldBuild(FieldId),
    /// Dropping and rebuilding every field index (`reindex-all`).
    ReindexAll,
    /// Re-queueing every document for embedding (`DocStore::index_all`).
    VectorReindex,
}

const TAG_FIELD_BUILD: u8 = 1;
const TAG_REINDEX_ALL: u8 = 2;
const TAG_VECTOR_REINDEX: u8 = 3;

impl TaskKey {
    /// The key of `store`'s build of field `field`.
    pub fn field_build(store: NamespaceId, field: FieldId) -> Self {
        Self {
            store,
            task: Task::FieldBuild(field),
        }
    }

    /// The key of `store`'s `reindex-all`.
    pub fn reindex_all(store: NamespaceId) -> Self {
        Self {
            store,
            task: Task::ReindexAll,
        }
    }

    /// The key of `store`'s vector reindex. Its records exist only with
    /// semantic search, but every build decodes the key.
    #[cfg_attr(not(feature = "semantic-search"), allow(dead_code))]
    pub fn vector_reindex(store: NamespaceId) -> Self {
        Self {
            store,
            task: Task::VectorReindex,
        }
    }

    /// The key as stored.
    pub fn to_bytes(self) -> Vec<u8> {
        let mut out = Vec::with_capacity(21);
        out.extend_from_slice(&self.store.to_bytes());
        match self.task {
            Task::FieldBuild(field) => {
                out.push(TAG_FIELD_BUILD);
                out.extend_from_slice(&field.to_be_bytes());
            }
            Task::ReindexAll => out.push(TAG_REINDEX_ALL),
            Task::VectorReindex => out.push(TAG_VECTOR_REINDEX),
        }
        out
    }

    /// Decode a stored key; `None` for bytes no `to_bytes` produced.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let store = NamespaceId::from_bytes(bytes.get(..16)?.try_into().ok()?);
        let task = match (bytes.get(16)?, &bytes[17..]) {
            (&TAG_FIELD_BUILD, field) => Task::FieldBuild(FieldId::from_be_bytes(field.try_into().ok()?)),
            (&TAG_REINDEX_ALL, []) => Task::ReindexAll,
            (&TAG_VECTOR_REINDEX, []) => Task::VectorReindex,
            _ => return None,
        };
        Some(Self { store, task })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_round_trip_and_start_with_the_store() {
        let store = NamespaceId::random().unwrap();
        for key in [
            TaskKey::field_build(store, 7),
            TaskKey::reindex_all(store),
            TaskKey::vector_reindex(store),
        ] {
            let bytes = key.to_bytes();
            assert_eq!(&bytes[..16], &store.to_bytes());
            assert_eq!(TaskKey::from_bytes(&bytes), Some(key));
        }
    }

    #[test]
    fn foreign_bytes_are_not_a_key() {
        let store = NamespaceId::random().unwrap().to_bytes();
        for bad in [
            &store[..15],
            &[&store[..], &[9]].concat(),
            &[&store[..], &[TAG_REINDEX_ALL, 0]].concat(),
            &[&store[..], &[TAG_FIELD_BUILD, 1, 2]].concat(),
        ] {
            assert_eq!(TaskKey::from_bytes(bad), None, "{bad:?}");
        }
    }
}
