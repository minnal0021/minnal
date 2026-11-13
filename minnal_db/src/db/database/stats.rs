//! Stats and metrics snapshots.

use super::*;

impl Database {
    pub fn stats(&self) -> Stats {
        self.default_store().map(|s| s.stats()).unwrap_or_else(|_| Stats {
            segment_count: 0,
            disk_bytes: 0,
            garbage_size: 0,
            waste_ratio: 0.0,
            total_gc_runs: 0,
            total_bytes_reclaimed: 0,
            live_bytes: 0,
        })
    }

    pub fn get_waste_ratio(&self) -> f64 {
        self.default_store().map(|s| s.get_waste_ratio()).unwrap_or(0.0)
    }

    /// Returns per-bucket value-log metadata for every active namespace.
    ///
    /// Each entry is `(namespace_name, Vec<(bucket_id, ValueLogMetadata)>)`.
    pub fn value_log_shard_stats(&self) -> Vec<(String, Vec<(u32, ValueLogMetadata)>)> {
        let namespaces = self.list_namespaces();
        let stores = self.stores.read();
        namespaces
            .into_iter()
            .filter_map(|(name, ns_id)| stores.get(&ns_id).map(|s| (name, s.value_log.all_bucket_metadata())))
            .collect()
    }

    /// Physical (on-disk `st_blocks`) vs logical (file length) value-log
    /// footprint per shard for every namespace. Cheap: one `stat` per shard.
    pub fn value_log_physical_stats(&self) -> Vec<(String, Vec<crate::storage::value_log::sharded::ShardPhysicalStats>)> {
        let namespaces = self.list_namespaces();
        let stores = self.stores.read();
        namespaces
            .into_iter()
            .filter_map(|(name, ns_id)| stores.get(&ns_id).map(|s| (name, s.value_log.physical_stats())))
            .collect()
    }

    /// Per-segment live/garbage breakdown for one namespace's value-log shards.
    ///
    /// Cheap: the counters are maintained in memory by writers, so this reads no
    /// segment data at all.
    pub fn value_log_segment_stats(&self, namespace: &str) -> Result<Vec<(u32, Vec<crate::storage::value_log::SegmentStats>)>> {
        let store = self.get_store_by_name(namespace)?;
        Ok(store.value_log.all_segment_stats())
    }

    /// Returns a snapshot of the current WAL metadata.
    pub fn wal_metadata(&self) -> WalMetadata {
        self.wal_metadata.read().clone()
    }

    /// Snapshot of the engine-wide operational counters (runtime metrics).
    ///
    /// Counters are recorded per namespace, so the engine-wide view is the sum
    /// of every live namespace's snapshot plus the global instance — which holds
    /// the WAL-GC counters (not attributable to one namespace) and the folded
    /// totals of every dropped namespace (so the aggregate stays monotonic).
    pub fn metrics_snapshot(&self) -> crate::support::metrics::MetricsSnapshot {
        let mut agg = self.metrics.snapshot();
        for store in self.stores.read().values() {
            if let Some(m) = store.metrics() {
                agg.accumulate(&m.snapshot());
            }
        }
        agg
    }

    /// Operational counters for a single namespace, by name.
    ///
    /// Note: the WAL-GC counters (`wal_gc_runs`, `wal_segments_deleted`) and
    /// the `vector_wal_*` counters belong to the two shared logs and are not
    /// attributed to any namespace — they are always `0` here and only
    /// populated in [`Self::metrics_snapshot`].
    pub fn metrics_snapshot_for(&self, name: &str) -> Result<crate::support::metrics::MetricsSnapshot> {
        let ns_id = self
            .registry
            .read()
            .get_id(name)
            .ok_or_else(|| KVError::Serialization(format!("Namespace '{}' not found", name)))?;
        let stores = self.stores.read();
        let store = stores
            .get(&ns_id)
            .ok_or_else(|| KVError::Serialization(format!("Namespace '{}' not found", name)))?;
        Ok(store.metrics().map(|m| m.snapshot()).unwrap_or_default())
    }

    /// Per-namespace operational counters for every live namespace, keyed by
    /// namespace name. Excludes the global WAL-GC/retired accumulator.
    pub fn metrics_snapshot_by_namespace(&self) -> Vec<(String, crate::support::metrics::MetricsSnapshot)> {
        let registry = self.registry.read();
        let mut out: Vec<(String, crate::support::metrics::MetricsSnapshot)> = self
            .stores
            .read()
            .iter()
            .filter_map(|(ns_id, store)| {
                let name = registry.get_name(*ns_id)?.to_owned();
                Some((name, store.metrics().map(|m| m.snapshot()).unwrap_or_default()))
            })
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// Returns a live manifest snapshot for every active namespace.
    ///
    /// Each entry is `(namespace_name, manifest)`.  The manifest is built
    /// from the in-memory LSM state so it does not touch disk.
    pub fn lsm_manifests(&self) -> Vec<(String, crate::storage::lsm::manifest::LsmManifest)> {
        let namespaces = self.list_namespaces();
        let stores = self.stores.read();
        namespaces
            .into_iter()
            .filter_map(|(name, ns_id)| stores.get(&ns_id).and_then(|s| s.lsm.build_manifest_snapshot().ok()).map(|m| (name, m)))
            .collect()
    }

    /// Returns the in-memory (non-SSTable) LSM stats for every active namespace,
    /// as `(namespace_name, stats)`. Complements [`lsm_manifests`](Self::lsm_manifests),
    /// which only reflects on-disk SSTables.
    pub fn lsm_runtime_stats(&self) -> Vec<(String, crate::storage::lsm::tree::LSMStats)> {
        let namespaces = self.list_namespaces();
        let stores = self.stores.read();
        namespaces
            .into_iter()
            .filter_map(|(name, ns_id)| stores.get(&ns_id).map(|s| (name, s.lsm.stats())))
            .collect()
    }
}
