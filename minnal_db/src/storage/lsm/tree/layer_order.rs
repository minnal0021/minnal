//! Test-only check of what reads rely on: the first copy of a key found, newest
//! layer first, is its newest version.

use super::*;
use std::collections::HashMap;

impl LSMTree {
    /// Every key whose first copy, newest layer first, is not its newest
    /// version, described as `"key: <first layer> holds V, but <layer> below
    /// holds W"`; empty when reads can stop at the first copy.
    ///
    /// Layers, newest first, as a read sees them: the active memtable, the
    /// read-only memtables not yet flushed (newest version first), then per
    /// bucket the live Level-0 files (newest stamp first) and Level 1. A flushed
    /// read-only memtable and an obsolete L0 file are skipped, as reads skip them.
    ///
    /// The first copy must be strictly newer than every copy below it: versions
    /// are unique, so an equal one is one write applied twice. The exception is
    /// L1 holding the first copy's version, and older L0 copies between the two:
    /// merged L0 files that a reopen found before their deletion are read again
    /// above the L1 they were merged into, and `cleanup_obsolete_level0_files`
    /// leaves only the newest merged files, so the first of them holds the
    /// version L1 holds.
    pub(crate) fn layer_order_violations(&self) -> Result<Vec<String>> {
        // Per key, its first copy: (version, layer).
        let mut first: HashMap<Vec<u8>, (Version, String)> = HashMap::new();
        let mut violations = Vec::new();
        let mut visit = |layer: &str, key: &[u8], version: Version, is_level1: bool| match first.get(key) {
            None => {
                first.insert(key.to_vec(), (version, layer.to_string()));
            }
            Some((top, top_layer)) => {
                if version > *top || (version == *top && !is_level1) {
                    violations.push(format!(
                        "{}: {top_layer} holds {top:?}, but {layer} below holds {version:?}",
                        String::from_utf8_lossy(key)
                    ));
                }
            }
        };

        for record in self.memtable.read().collect_records() {
            visit("the active memtable", &record.key, record.version, false);
        }
        let mut read_only = self.read_only_memtables.read().clone();
        read_only.retain(|ro| !ro.is_flushed_to_level0());
        read_only.sort_by_key(|ro| std::cmp::Reverse(ro.version()));
        for ro in &read_only {
            let layer = format!("read-only memtable {}", ro.version());
            for record in ro.records().iter() {
                visit(&layer, &record.key, record.version, false);
            }
        }
        for bucket in 0u32..self.config.num_buckets as u32 {
            let mut files = self.level0_entries_snapshot(bucket);
            files.retain(|file| !file.is_obsolete());
            files.sort_by_key(|file| std::cmp::Reverse(file.created_at_ms));
            for file in &files {
                let layer = format!("L0 file {}", file.path.display());
                for entry in self.read_sstable_entries_from_path(&file.path)? {
                    visit(&layer, &entry.key, entry.version(), false);
                }
            }
            for entry in self.read_sstable_entries(bucket)? {
                visit(&format!("L1 bucket {bucket}"), &entry.key, entry.version(), true);
            }
        }
        Ok(violations)
    }
}
