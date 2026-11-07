pub mod chunking;
pub mod cluster;
#[cfg(feature = "doc-store")]
pub mod context;
#[cfg(test)]
mod eval;
pub mod index;
// Partition maintenance: the journal, split, reassign, merge and rebuild
// executors and their crash recovery (design doc M3a).
pub mod ivf_split;
pub mod metrics;
pub mod quantisation;
pub mod service;
pub(crate) mod simd_support;
// Vector-index storage bridge (`DbVectorStore`, upsert/delete, embed queue,
// query-embedding cache) over raw namespaces: usable with `kv-store` alone, no
// `doc-store`.
#[cfg(feature = "doc-store")]
mod partition_admin;
pub mod vector_kv;
pub mod vector_math;
#[cfg(feature = "doc-store")]
pub(crate) mod workers;

pub use self::chunking::chunk_document;
pub use self::cluster::Cluster;
pub use self::cluster::ClusterIndex;
pub use self::cluster::{
    CountTable, IvfLayout, MaintenanceCounters, MaintenanceTotals, NamespaceIvf, PartitionHandle, Posting, PostingDelta, PostingInfo, PostingOrigin,
    PostingState, ProbeSettings, WithEntryCounts, ZERO_CENTRE, select_probes,
};
pub use self::index::composite_key;
pub use self::index::vector_index::{QuantisationStyle, VectorIndex};
pub use self::quantisation::rabitq::{index_embedding_in_cluster, index_embedding_rotated, index_embedding_to_cluster, index_embedding_zero_centred};
