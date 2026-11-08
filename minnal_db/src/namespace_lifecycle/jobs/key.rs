//! `JobKey`: which namespace a job belongs to, and which job it is. It is also
//! the key of the job's record in `system_namespace_maintenance`.
//!
//! Encoded as the namespace id (16 bytes), a job tag (1 byte) and the job's
//! subject: the field id (4 bytes, big-endian) for a field job, nothing
//! otherwise. The namespace id comes first, so one namespace's records are a
//! prefix scan, and a namespace dropped and created again (which gets a new id)
//! never sees the old one's records. The id is random, so the leading bytes
//! vary and the records spread over the value-log buckets.

use std::fmt;

use crate::FieldId;
use crate::namespace::NamespaceId;

/// A job's key. See the module docs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct JobKey {
    /// The namespace the job works on.
    pub namespace: NamespaceId,
    /// The job.
    pub kind: JobKind,
}

/// Which job a key names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum JobKind {
    /// Building one field's index.
    FieldBuild(FieldId),
    /// Dropping one field's index.
    FieldDrop(FieldId),
    /// Dropping and rebuilding every field index.
    ReindexAll,
    /// Turning semantic search on: creating the vector index.
    VectorEnable,
    /// Turning semantic search off: removing the vector index's data.
    VectorCleanup,
    /// Re-queueing every document for embedding.
    VectorReindex,
    /// Re-clustering the vector index's partition from its codes.
    PartitionRebuild,
}

const TAG_FIELD_BUILD: u8 = 1;
const TAG_REINDEX_ALL: u8 = 2;
const TAG_VECTOR_REINDEX: u8 = 3;
const TAG_FIELD_DROP: u8 = 4;
const TAG_VECTOR_ENABLE: u8 = 5;
const TAG_VECTOR_CLEANUP: u8 = 6;
const TAG_PARTITION_REBUILD: u8 = 7;

impl JobKind {
    /// The job's name, without its field: `field_build`, `reindex_all`, …
    pub fn name(self) -> &'static str {
        match self {
            Self::FieldBuild(_) => "field_build",
            Self::FieldDrop(_) => "field_drop",
            Self::ReindexAll => "reindex_all",
            Self::VectorEnable => "vector_enable",
            Self::VectorCleanup => "vector_cleanup",
            Self::VectorReindex => "vector_reindex",
            Self::PartitionRebuild => "partition_rebuild",
        }
    }

    /// The field a field job works on.
    pub fn field(self) -> Option<FieldId> {
        match self {
            Self::FieldBuild(f) | Self::FieldDrop(f) => Some(f),
            _ => None,
        }
    }

    /// Whether a cancel may stop the job part-way and undo it. Not a field
    /// drop (it is short, and its undo would be a full build), a
    /// `reindex-all` (its first step drops the old indexes, which cannot come
    /// back), or a vector cleanup (a half-removed vector index serves neither
    /// searches nor a new enable).
    pub fn cancellable(self) -> bool {
        !matches!(self, Self::FieldDrop(_) | Self::ReindexAll | Self::VectorCleanup)
    }
}

/// `field_build:7` for a field job (the field's id), else the name alone.
impl fmt::Display for JobKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.field() {
            Some(field) => write!(f, "{}:{field}", self.name()),
            None => f.write_str(self.name()),
        }
    }
}

impl JobKey {
    /// The key of `namespace`'s `kind` job.
    pub fn new(namespace: NamespaceId, kind: JobKind) -> Self {
        Self { namespace, kind }
    }

    /// The key of `namespace`'s build of field `field`.
    pub fn field_build(namespace: NamespaceId, field: FieldId) -> Self {
        Self::new(namespace, JobKind::FieldBuild(field))
    }

    /// The key of `namespace`'s drop of field `field`'s index.
    pub fn field_drop(namespace: NamespaceId, field: FieldId) -> Self {
        Self::new(namespace, JobKind::FieldDrop(field))
    }

    /// The key of `namespace`'s `reindex-all`.
    pub fn reindex_all(namespace: NamespaceId) -> Self {
        Self::new(namespace, JobKind::ReindexAll)
    }

    /// The key of `namespace`'s vector reindex. Its records exist only with
    /// semantic search, but every build decodes the key.
    #[cfg_attr(not(feature = "semantic-search"), allow(dead_code))]
    pub fn vector_reindex(namespace: NamespaceId) -> Self {
        Self::new(namespace, JobKind::VectorReindex)
    }

    /// The key as stored.
    pub fn to_bytes(self) -> Vec<u8> {
        let mut out = Vec::with_capacity(21);
        out.extend_from_slice(&self.namespace.to_bytes());
        let tag = match self.kind {
            JobKind::FieldBuild(_) => TAG_FIELD_BUILD,
            JobKind::FieldDrop(_) => TAG_FIELD_DROP,
            JobKind::ReindexAll => TAG_REINDEX_ALL,
            JobKind::VectorEnable => TAG_VECTOR_ENABLE,
            JobKind::VectorCleanup => TAG_VECTOR_CLEANUP,
            JobKind::VectorReindex => TAG_VECTOR_REINDEX,
            JobKind::PartitionRebuild => TAG_PARTITION_REBUILD,
        };
        out.push(tag);
        if let Some(field) = self.kind.field() {
            out.extend_from_slice(&field.to_be_bytes());
        }
        out
    }

    /// Decode a stored key; `None` for bytes no `to_bytes` produced.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let namespace = NamespaceId::from_bytes(bytes.get(..16)?.try_into().ok()?);
        let field = |b: &[u8]| -> Option<FieldId> { Some(FieldId::from_be_bytes(b.try_into().ok()?)) };
        let kind = match (bytes.get(16)?, &bytes[17..]) {
            (&TAG_FIELD_BUILD, b) => JobKind::FieldBuild(field(b)?),
            (&TAG_FIELD_DROP, b) => JobKind::FieldDrop(field(b)?),
            (&TAG_REINDEX_ALL, []) => JobKind::ReindexAll,
            (&TAG_VECTOR_ENABLE, []) => JobKind::VectorEnable,
            (&TAG_VECTOR_CLEANUP, []) => JobKind::VectorCleanup,
            (&TAG_VECTOR_REINDEX, []) => JobKind::VectorReindex,
            (&TAG_PARTITION_REBUILD, []) => JobKind::PartitionRebuild,
            _ => return None,
        };
        Some(Self { namespace, kind })
    }
}

/// One job of every kind (two fields for the field jobs), for tests.
#[cfg(test)]
pub(crate) const EVERY_KIND: [JobKind; 9] = [
    JobKind::FieldBuild(7),
    JobKind::FieldBuild(8),
    JobKind::FieldDrop(7),
    JobKind::FieldDrop(8),
    JobKind::ReindexAll,
    JobKind::VectorEnable,
    JobKind::VectorCleanup,
    JobKind::VectorReindex,
    JobKind::PartitionRebuild,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_round_trip_and_start_with_the_namespace() {
        let namespace = NamespaceId::random().unwrap();
        for kind in EVERY_KIND {
            let key = JobKey::new(namespace, kind);
            let bytes = key.to_bytes();
            assert_eq!(&bytes[..16], &namespace.to_bytes());
            assert_eq!(JobKey::from_bytes(&bytes), Some(key));
        }
    }

    #[test]
    fn foreign_bytes_are_not_a_key() {
        let namespace = NamespaceId::random().unwrap().to_bytes();
        for bad in [
            &namespace[..15],
            &[&namespace[..], &[9]].concat(),
            &[&namespace[..], &[TAG_REINDEX_ALL, 0]].concat(),
            &[&namespace[..], &[TAG_FIELD_BUILD, 1, 2]].concat(),
            &[&namespace[..], &[TAG_FIELD_DROP]].concat(),
        ] {
            assert_eq!(JobKey::from_bytes(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn a_job_reads_as_its_name_and_field() {
        assert_eq!(JobKind::FieldBuild(7).to_string(), "field_build:7");
        assert_eq!(JobKind::PartitionRebuild.to_string(), "partition_rebuild");
    }
}
