//! Replaying both logs at open.

use crate::db::database::{Database, display_key};
use crate::db::error::Result;
use crate::db::wal::{WalEntry, WalOperationType};

use log::{debug, error, info, warn};
use std::sync::atomic::Ordering;

impl Database {
    /// Replay the vector index WAL's unretired records into their memtables,
    /// with their original seqs, then make them durable in L0 and record that.
    ///
    /// Runs at open, before main-WAL recovery and before the flush observers
    /// are wired: both logs replay with original seqs into memtables that
    /// resolve by seq, so their order does not matter. Each record is applied
    /// only if it lies at or above its namespace's durable safe offset
    /// (`VectorIndexWal::replay`); records of a namespace no longer in the
    /// registry are skipped. A record that fails twice goes to the fail log,
    /// as in main-WAL recovery, and is retired with the rest.
    ///
    /// **Load-bearing:** no observer sees the replay, so the namespaces it
    /// replayed into are flushed to L0, their value logs fsynced, and the log
    /// told they are safe up to its tail, all **before** `recover_from_wal`
    /// runs. That recovery compacts every store into L1, which drops
    /// tombstones; had the marker stayed behind, a second crash would replay a
    /// record whose delete the compaction erased, and the key would come
    /// back (`a_second_crash_does_not_bring_back_a_deleted_key`). It would
    /// also leave an idle namespace holding the cut.
    pub(in crate::db) fn recover_from_vector_index_wal(&self) -> Result<()> {
        // A new write must never draw a seq below a replayed record's, or it
        // would lose to it.
        self.next_seq
            .fetch_max(self.vector_index_wal.last_seq().saturating_add(1), Ordering::SeqCst);

        let stores = self.stores.read().clone();
        let (mut replayed, mut skipped) = (0u64, 0u64);
        let mut failed: Vec<(WalEntry, String)> = Vec::new();
        self.vector_index_wal.replay(|r| -> Result<()> {
            let Some(store) = stores.get(&r.ns_id) else {
                skipped += 1;
                return Ok(());
            };
            let apply = || match r.value {
                Some(value) => store.replay_upsert(r.key, value, r.seq),
                None => store.replay_delete(r.key, r.seq),
            };
            match apply().or_else(|_| apply()) {
                Ok(()) => replayed += 1,
                Err(e) => {
                    let entry = match r.value {
                        Some(value) => WalEntry::new_upsert_ns(r.ns_id, r.key.to_vec(), value.to_vec()),
                        None => WalEntry::new_delete_ns(r.ns_id, r.key.to_vec()),
                    }
                    .with_sequence(r.seq)
                    .with_op_name("vector_index_wal_replay");
                    failed.push((entry, e.to_string()));
                }
            }
            Ok(())
        })?;

        crate::support::metrics::Metrics::add(&self.metrics.vector_wal_records_replayed, replayed);
        crate::support::metrics::Metrics::add(&self.metrics.vector_wal_records_skipped, skipped);
        if replayed + skipped > 0 || !failed.is_empty() {
            info!(
                "[RECOVERY] vector index WAL: replayed {replayed} record(s), skipped {skipped} of dropped namespaces, {} failed",
                failed.len()
            );
        }
        if !failed.is_empty() {
            error!(
                "[RECOVERY] vector index WAL: {} record(s) failed to replay twice; see the fail log",
                failed.len()
            );
            let failures: Vec<(&WalEntry, String)> = failed.iter().map(|(e, msg)| (e, msg.clone())).collect();
            crate::db::wal::wal_fail_log::write_fail_log(&self.fail_log_dir, &self.db_path, &failures);
        }

        // Every namespace still holding the cut had its records replayed above
        // (dropped ones were forgotten at open). Flush them and record it.
        let holding = self.vector_index_wal.namespaces_holding_cut();
        if !holding.is_empty() {
            let mut flushed = Vec::with_capacity(holding.len());
            for ns_id in holding {
                let Some(store) = stores.get(&ns_id) else {
                    continue;
                };
                store.flush_memtable_to_level0()?;
                store.sync_value_log()?;
                flushed.push(ns_id);
            }
            self.vector_index_wal.retire_replayed(&flushed)?;
        }
        Ok(())
    }

    pub(in crate::db) fn recover_from_wal(&self) -> Result<()> {
        let wal_metadata = self.wal_metadata.read().clone();
        let tail = wal_metadata.tail;
        let start = wal_metadata.cut.max(wal_metadata.head);

        if start >= tail {
            debug!("[RECOVERY] No recovery needed.");
            return Ok(());
        }

        info!("[RECOVERY] Starting WAL recovery: replaying from {start} to {tail}");

        let all_entries = self.wal.scan_entries(start, tail)?;

        // Collect every eligible entry into a single flat list, then replay
        // strictly in global sequence order.
        //
        // Replaying in sequence order is essential for correctness: the *last*
        // write to a key by sequence must win after recovery, exactly as it did
        // before the crash. Sequence == WAL append order (writers allocate the
        // sequence under the WAL lock), so sorting by sequence reproduces the
        // original order.
        //
        // Eligibility:
        //   • Entries below their namespace's saved safe offset are skipped:
        //     they are in L0 files already (`WalMetadata::replay_from`). Per
        //     namespace, not just from the cut: a namespace that flushed past an
        //     entry may since have had a later delete of that key merged into L1,
        //     its tombstone dropped, and replaying the entry would bring the key
        //     back with nothing newer to beat it.
        //   • Entries for a namespace that no longer exists are skipped: a
        //     dropped namespace's entries above the cut stay in the log until
        //     WAL GC passes them.
        //     The namespace is intentionally gone, so replaying its writes would
        //     be wrong — and there is no KVStore to replay them into. We drop
        //     them with a WARN rather than routing them to the fail log, which is
        //     reserved for writes that genuinely failed to apply. Namespace IDs
        //     are monotonic and never reused, so a missing store is unambiguous.
        use std::collections::HashSet;
        let known_namespaces: HashSet<u32> = self.stores.read().keys().copied().collect();
        let mut skipped_unknown_ns_count = 0u64;
        let mut warned_namespaces: HashSet<u32> = HashSet::new();
        let mut eligible: Vec<WalEntry> = Vec::new();
        let mut skipped_flushed_count = 0u64;
        for (pointer, entry) in all_entries {
            if pointer.offset < wal_metadata.replay_from(entry.namespace_id) {
                skipped_flushed_count += 1;
                continue;
            }
            if !known_namespaces.contains(&entry.namespace_id) {
                // Log once per namespace to avoid flooding the log when a deleted
                // namespace had many un-persisted entries at crash time.
                if warned_namespaces.insert(entry.namespace_id) {
                    warn!(
                        "[RECOVERY] Skipping WAL entries for unknown namespace ns={} \
                         (likely a namespace deleted just before the crash). Its writes \
                         will not be replayed.",
                        entry.namespace_id
                    );
                }
                skipped_unknown_ns_count += 1;
                continue;
            }
            eligible.push(entry);
        }
        // Stable sort by sequence so entries replay in their original order.
        eligible.sort_by_key(|e| e.sequence);

        // Apply each entry in sequence order, retrying once *immediately* on
        // failure. Immediate (rather than deferred) retry is what preserves the
        // ordering guarantee: a failed entry has no effect, so retrying it before
        // moving on means a later write to the same key still lands afterwards.
        let mut recovered_count = 0u64;
        let mut error_count = 0u64;
        let mut failed_entries: Vec<(WalEntry, String)> = Vec::new();

        for entry in eligible {
            match self.apply_wal_entry(&entry) {
                Ok(()) => recovered_count += 1,
                Err(e) => {
                    warn!(
                        "[RECOVERY] Apply failed for op '{}' key='{}' (ns={}): {}. Retrying once.",
                        entry.op_name,
                        display_key(&entry.key),
                        entry.namespace_id,
                        e
                    );
                    match self.apply_wal_entry(&entry) {
                        Ok(()) => recovered_count += 1,
                        Err(e2) => {
                            error!(
                                "[RECOVERY] Retry also failed for op '{}' key='{}' (ns={}): {}. \
                                 Writing to fail log.",
                                entry.op_name,
                                display_key(&entry.key),
                                entry.namespace_id,
                                e2
                            );
                            error_count += 1;
                            failed_entries.push((entry, e2.to_string()));
                        }
                    }
                }
            }
        }

        // Write any persistent failures to the fail log — one record per failed op.
        if !failed_entries.is_empty() {
            let failures: Vec<(&WalEntry, String)> = failed_entries.iter().map(|(e, msg)| (e, msg.clone())).collect();
            crate::db::wal::wal_fail_log::write_fail_log(&self.fail_log_dir, &self.db_path, &failures);
        }

        // Flush all namespaces that had recovered entries, then retire the whole
        // log: every replayed entry is in L0 now (or in the fail log, which is
        // the operator's recovery path). A flush that fails leaves the progress
        // where it was, so the next open replays again.
        let mut flushed = true;
        if recovered_count > 0 {
            let stores = self.stores.read();
            for kv_store in stores.values() {
                if let Err(e) = kv_store.flush_and_compact_all() {
                    warn!("[RECOVERY] Flushing a namespace after replay failed; the WAL keeps its entries: {e:?}");
                    flushed = false;
                }
            }
        }
        if flushed {
            self.wal_flush_observer.retire_all()?;
        }

        info!(
            "[RECOVERY] Complete. Recovered: {}, Skipped (already in L0): {}, \
             Skipped (unknown/deleted namespaces): {}, Failed after retry (see fail log): {}",
            recovered_count, skipped_flushed_count, skipped_unknown_ns_count, error_count
        );
        Ok(())
    }

    /// Apply a single WAL entry to its KV store.  Returns `Err` if the
    /// namespace is missing or the underlying storage write fails.
    ///
    /// `recover_from_wal` filters out entries for deleted namespaces before
    /// calling this, so in the recovery path a missing namespace here indicates
    /// a genuine inconsistency rather than an ordinary post-deletion entry.
    fn apply_wal_entry(&self, entry: &WalEntry) -> Result<()> {
        let kv_store = self.get_store(entry.namespace_id)?;
        let seq = entry.sequence;
        match entry.operation {
            WalOperationType::Upsert => {
                if let Some(value) = &entry.value {
                    kv_store.replay_upsert(&entry.key, value, seq)
                } else {
                    Ok(())
                }
            }
            WalOperationType::Delete => kv_store.replay_delete(&entry.key, seq),
        }
    }
}
