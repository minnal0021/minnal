pub mod chunking;
pub mod cluster;
pub mod index;
pub mod metrics;
pub mod quantisation;
pub mod service;
pub(crate) mod simd;
#[cfg(test)]
mod vector_bench;
pub mod vector_math;

pub use self::chunking::chunk_document;
pub use self::cluster::Cluster;
pub use self::cluster::ClusterIndex;
pub use self::cluster::{
    CountTable, IvfLayout, NamespaceIvf, PartitionHandle, Posting, PostingDelta, PostingInfo, PostingOrigin, PostingState, ProbeSettings,
    WithEntryCounts, ZERO_CENTRE, select_probes,
};
pub use self::index::composite_key;
pub use self::index::vector_index::{QuantisationStyle, VectorIndex};
pub use self::quantisation::rabitq::{index_embedding_in_cluster, index_embedding_rotated, index_embedding_to_cluster, index_embedding_zero_centred};
