//! Test-only check of the layer-order invariant: every copy of a key in a
//! newer layer has a higher write sequence than every copy in an older one.

use super::*;
use std::collections::HashMap;

impl LSMTree {
    /// Every key whose copies break the layer order, described as
    /// `"key: seq S in <layer> is not older than seq T above it"`; empty when the
    /// invariant holds.
    ///
    /// Layers, newest first, as a read sees them: the active memtable, the
    /// read-only memtables not yet flushed (newest version first), then per
    /// bucket the live Level-0 files (newest stamp first) and Level 1. A flushed
    /// read-only memtable and an obsolete L0 file are skipped, as reads skip them.
    /// Equal seqs count as a break: seqs are unique, so two copies at one seq are
    /// one write applied twice.
    pub(crate) fn layer_order_violations(&self) -> Result<Vec<String>> {
        let mut newest_above: HashMap<Vec<u8>, u64> = HashMap::new();
        let mut violations = Vec::new();
        let mut visit = |layer: &str, key: &[u8], seq: u64| {
            if let Some(&above) = newest_above.get(key)
                && !seq_newer(above, seq)
            {
                violations.push(format!(
                    "{}: seq {seq} in {layer} is not older than seq {above} above it",
                    String::from_utf8_lossy(key)
                ));
            }
            newest_above.insert(key.to_vec(), seq);
        };

        for record in self.memtable.read().collect_records() {
            visit("the active memtable", &record.key, record.seq);
        }
        let mut read_only = self.read_only_memtables.read().clone();
        read_only.retain(|ro| !ro.is_flushed_to_level0());
        read_only.sort_by_key(|ro| std::cmp::Reverse(ro.version()));
        for ro in &read_only {
            let layer = format!("read-only memtable {}", ro.version());
            for record in ro.records().iter() {
                visit(&layer, &record.key, record.seq);
            }
        }
        for bucket in 0u32..self.config.num_buckets as u32 {
            let mut files = self.level0_entries_snapshot(bucket);
            files.retain(|file| !file.is_obsolete());
            files.sort_by_key(|file| std::cmp::Reverse(file.created_at_ms));
            for file in &files {
                let layer = format!("L0 file {}", file.path.display());
                for entry in self.read_sstable_entries_from_path(&file.path)? {
                    visit(&layer, &entry.key, entry.seq);
                }
            }
            for entry in self.read_sstable_entries(bucket)? {
                visit(&format!("L1 bucket {bucket}"), &entry.key, entry.seq);
            }
        }
        Ok(violations)
    }
}
