//! [`AsyncDb`]'s maintenance and statistics: garbage collection, compaction,
//! index checkpoints, and the engine's metrics and storage stats.

use super::*;

impl AsyncDb {
    // ── Maintenance ───────────────────────────────────────────────────

    pub async fn garbage_collect(&self) -> Result<GCStats> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || db.garbage_collect())
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// Run value-log GC on every namespace and return per-namespace results.
    pub async fn garbage_collect_all(&self) -> Vec<(String, GCStats)> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || db.garbage_collect_all()).await.unwrap_or_default()
    }

    /// Run WAL garbage collection (reclaims segments wholly below the cut).
    ///
    /// Returns `(bytes_reclaimed, pending_bytes)`: the bytes of segment files
    /// deleted by this pass, and the bytes of WAL recovery would replay
    /// afterwards (from the cut to the tail). If another pass is already
    /// running, nothing is reclaimed and the result is `(0, pending_bytes)`.
    pub async fn garbage_collect_wal(&self) -> Result<(u64, u64)> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || db.garbage_collect_wal())
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    pub async fn compact(&self) -> Result<()> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || db.compact())
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// Async [`Db::flush_namespaces`].
    #[cfg(any(test, feature = "semantic-search"))]
    pub(crate) async fn flush_namespaces(&self, names: Vec<String>) -> Result<()> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || db.flush_namespaces(&names))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// Async [`Db::sync_vector_index_wal`].
    #[cfg(feature = "semantic-search")]
    pub(crate) async fn sync_vector_index_wal(&self) -> Result<()> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || db.sync_vector_index_wal())
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// Force an index checkpoint across all namespaces: flush each namespace's
    /// dense row map and all active field indexes to disk, compacting any
    /// field-index bitmap store whose waste exceeds the configured
    /// `index_blob_waste_threshold`. Returns the number of active field indexes
    /// checkpointed.
    ///
    /// This runs the same pass as the periodic `IndexCheckpointWorker` and the
    /// one performed on clean shutdown; exposed for callers that want to force a
    /// flush + compaction on demand. Runs the blocking checkpoint on the
    /// blocking thread pool.
    /// How many index checkpoints have started so far; see
    /// [`index_checkpoint_covers`](Self::index_checkpoint_covers).
    pub fn index_checkpoints_started(&self) -> u64 {
        self.inner.inner.index_checkpoints_started()
    }

    /// Whether an index checkpoint that started after `started` checkpoints had
    /// has finished, so index changes made before `started` was read are on disk.
    /// A field build's changes are not in the WAL: only this makes them durable.
    pub fn index_checkpoint_covers(&self, started: u64) -> bool {
        self.inner.inner.index_checkpoint_covers(started)
    }

    pub async fn checkpoint_index(&self) -> Result<usize> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || db.checkpoint_index())
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    pub fn stats(&self) -> Stats {
        self.inner.stats()
    }

    /// Snapshot of engine-wide operational metrics (runtime counters).
    pub fn ops_metrics(&self) -> crate::support::metrics::MetricsSnapshot {
        self.inner.ops_metrics()
    }

    /// Field-index write buffers: bytes held now and at peak, the soft and
    /// hard limits, and how often each fired.
    pub fn index_overlay_stats(&self) -> crate::IndexOverlayStats {
        self.inner.index_overlay_stats()
    }

    /// Operational metrics for a single namespace, by name.
    pub fn ops_metrics_for(&self, namespace: &str) -> Result<crate::support::metrics::MetricsSnapshot> {
        self.inner.ops_metrics_for(namespace)
    }

    /// Per-namespace operational metrics for every live namespace, keyed by name.
    pub fn ops_metrics_by_namespace(&self) -> Vec<(String, crate::support::metrics::MetricsSnapshot)> {
        self.inner.ops_metrics_by_namespace()
    }

    /// Returns a snapshot of the current WAL metadata.
    pub fn wal_metadata(&self) -> crate::db::wal::WalMetadata {
        self.inner.wal_metadata()
    }

    /// Returns the main WAL's state: bounds, cut, per-namespace progress and
    /// each segment from the head to the tail.
    pub fn wal_stats(&self) -> crate::WalStats {
        self.inner.wal_stats()
    }

    /// Returns the vector index WAL's state: segments, tail, retired cut and
    /// the namespaces holding the cut back.
    pub fn vector_index_wal_stats(&self) -> Result<crate::VectorIndexWalStats> {
        self.inner.vector_index_wal_stats()
    }

    /// Returns a live LSM manifest snapshot for every active namespace.
    pub fn lsm_manifests(&self) -> Vec<(String, crate::storage::lsm::manifest::LsmManifest)> {
        self.inner.lsm_manifests()
    }

    /// Returns the in-memory (non-SSTable) LSM stats for every active namespace.
    pub fn lsm_runtime_stats(&self) -> Vec<(String, crate::storage::lsm::tree::LSMStats)> {
        self.inner.lsm_runtime_stats()
    }

    /// Returns per-bucket value-log metadata for every active namespace.
    pub fn value_log_shard_stats(&self) -> Vec<(String, Vec<(u32, crate::storage::value_log::ValueLogMetadata)>)> {
        self.inner.value_log_shard_stats()
    }

    /// Physical (on-disk) vs logical value-log footprint per shard, per namespace.
    pub fn value_log_physical_stats(&self) -> Vec<(String, Vec<crate::storage::value_log::sharded::ShardPhysicalStats>)> {
        self.inner.value_log_physical_stats()
    }

    /// Per-page value-log garbage breakdown for one namespace (by name).
    pub fn value_log_segment_stats(&self, namespace: &str) -> Result<Vec<(u32, Vec<crate::storage::value_log::SegmentStats>)>> {
        self.inner.value_log_segment_stats(namespace)
    }

    pub fn waste_ratio(&self) -> f64 {
        self.inner.waste_ratio()
    }
}
