//! Value-log GC: relocate a segment's survivors, make that durable, then unlink it.

use super::*;

/// What one bucket's GC pass did.
pub(super) struct BucketGCResult {
    bucket: u32,
    /// Bytes of survivors rewritten into the active tail — the *cost* of the pass.
    /// Compare against the bytes reclaimed for write amplification.
    bytes_rewritten: u64,
    /// Segments whose survivors have been relocated, but which cannot be unlinked
    /// until the LSM re-point is durable (see `garbage_collect_with_threshold`).
    pub(super) pending_unlink: Vec<u32>,
}

/// A record phase 1 found live: what phase 2 needs to copy and re-point it.
struct Survivor {
    key: Vec<u8>,
    value: Vec<u8>,
    /// The record's creation time, carried over to the copy: TTL expiry judges
    /// a value by when it was written, not by when GC last moved it.
    epoch: u64,
    /// Where the LSM pointed when phase 1 found the record live.
    old_pointer: u128,
}

impl KVStore {
    /// GC one bucket: rewrite the survivors of its worst segments, and hand back the
    /// segments that become safe to unlink *once the re-point is durable*.
    ///
    /// **Phase 1 (no lock):** read the segment sequentially. Every record carries its
    /// **key**, so liveness is one batched LSM lookup of the segment's keys: a record
    /// survives iff the LSM still points at exactly this location. So GC's cost is
    /// proportional to the segment rather than to the whole bucket or the whole key
    /// set. The segment is sealed and immutable, so nothing can change it under us.
    ///
    /// **Phase 2 (bucket lock):** relocate each survivor as a **compare-and-set** —
    /// append it to the active tail and update the LSM only if the key *still* maps
    /// to the old location. Phase 1 runs without the lock, so a key deleted since
    /// then must stay deleted and one overwritten since then must keep its newer
    /// value. The copy keeps the write's user seq and takes the next system seq
    /// ([`Version::next_system`]): newer than the copy it replaces, older than any
    /// user write to come.
    ///
    /// The old segment is deliberately **not** unlinked here — see
    /// [`garbage_collect_with_threshold`](Self::garbage_collect_with_threshold).
    pub(super) fn compact_bucket(&self, bucket: u32, segment_gc_threshold_pct: f64, tail_gc_threshold_pct: f64) -> Result<BucketGCResult> {
        let log = self.value_log.get_bucket_log(bucket)?;
        let mut result = BucketGCResult {
            bucket,
            bytes_rewritten: 0,
            pending_unlink: Vec::new(),
        };

        // The active tail is normally off-limits, but a small or idle namespace may never
        // fill a segment — so its garbage would be uncollectable forever while the waste
        // trigger kept firing. If the tail is garbage enough (>= `tail_gc_threshold_pct`),
        // seal it (under the bucket lock, since this rolls the segment writers append to)
        // so it can be collected in this same pass. See `TAIL_GC_MIN_GARBAGE_PCT`.
        {
            let _bucket_guard = self.value_log.lock_bucket_for_write(bucket)?;
            if let Some(sealed) = log.seal_tail_for_gc(tail_gc_threshold_pct)? {
                debug!(
                    "[KVStore '{}'] bucket {}: sealed the active tail (segment {}) — it was mostly garbage",
                    self.name, bucket, sealed
                );
            }
        }

        let candidates = log.gc_candidates(segment_gc_threshold_pct);
        if candidates.is_empty() {
            return Ok(result);
        }

        for segment in candidates {
            // ── Phase 1: what is still live in this segment? ──────────────────
            //
            // Liveness is resolved for the whole segment in ONE batched lookup,
            // not one point-get per record. A point-get can read every L0 file of
            // the bucket that its summary cannot rule out, and doing that per
            // record makes GC cost grow with records × L0 entries, quadratic in
            // churn: measured at ~165 µs per record per L0 file read, 8,000
            // records against 16 L0 files took 22 s. The vector-index queue,
            // which churns hardest, accumulates the most L0 files.
            //
            // `get_multiple` inverts the loop: each L0 file is read once, matched
            // against every pending key at once, so the cost is one pass per file
            // rather than one search per key.
            //
            // The price is a second pass over the segment — pass A takes keys
            // only, pass B re-reads to materialise just the survivors' values.
            // That trades random LSM lookups for one extra sequential read of a
            // file we are about to rewrite anyway, and keeps peak memory at
            // keys + survivors rather than the whole segment.
            // Pass A: the distinct keys this segment holds.
            let keys: Vec<Vec<u8>> = {
                let mut seen: std::collections::HashSet<Vec<u8>> = std::collections::HashSet::new();
                log.for_each_record(segment.id, |key, _value, _meta, _location| {
                    seen.insert(key.to_vec());
                })?;
                seen.into_iter().collect()
            };

            // Where the LSM currently points for each of them, if anywhere.
            let resolved = self.lsm.get_multiple(&keys)?;
            let live: std::collections::HashMap<&[u8], u128> = keys
                .iter()
                .zip(resolved.iter())
                .filter_map(|(key, found)| found.map(|(pointer, _seq)| (key.as_slice(), pointer)))
                .collect();

            // Pass B: keep only the records the LSM still points at.
            //
            // Note this yields at most ONE survivor per key: two records for the
            // same key in this segment have different locations, so only the one
            // whose pointer matches can match. Phase 2 relies on that.
            let mut survivors: Vec<Survivor> = Vec::new();
            log.for_each_record(segment.id, |key, value, meta, location| {
                let Ok(pointer) = ShardedValuePointer::new(bucket, location, self.value_log.num_buckets()) else {
                    return;
                };
                let old_u128 = pointer.to_u128();
                // Live iff the LSM still points at exactly this record.
                if live.get(key) == Some(&old_u128) {
                    survivors.push(Survivor {
                        key: key.to_vec(),
                        value,
                        epoch: meta.epoch,
                        old_pointer: old_u128,
                    });
                }
            })?;

            // ── Phase 2: relocate survivors, re-point by compare-and-set ──────
            //
            // A copy is the same user write, moved: it keeps the current copy's
            // user seq and takes the next system seq (`Version::next_system`).
            // That makes it newer than the copy it replaces, so the newest layer
            // still holds the newest version, and older than every user write to
            // the key: a writer that drew its seq before GC and is waiting for
            // the bucket lock wins whenever it lands. So GC takes no key stripe.
            // A version whose system seq is used up (about four billion moves of
            // one unchanged write) is re-pointed after the batch as a new user
            // write, under its stripe.
            let mut exhausted: Vec<&Survivor> = Vec::new();
            {
                let _bucket_guard = self.value_log.lock_bucket_for_write(bucket)?;

                // Batched for the same reason as phase 1 — a per-survivor
                // lookup pays the same full L0 scan, and here it is paid while
                // holding the bucket write lock, which blocks every writer to
                // this bucket (`put_to_storage` / `delete_from_storage` hold it
                // across their value-log append *and* their LSM update).
                //
                // Still a compare-and-set: holding that lock excludes every
                // writer's apply, so one batched read under it sees exactly what
                // per-key reads under it would. The re-points below do mutate
                // the LSM as we go, but survivors carry no duplicate keys (see
                // phase 1), so no re-point can invalidate another's answer.
                let survivor_keys: Vec<Vec<u8>> = survivors.iter().map(|s| s.key.clone()).collect();
                let current = self.lsm.get_multiple(&survivor_keys)?;

                for (survivor, current) in survivors.iter().zip(current) {
                    let Some((pointer, version)) = current else {
                        continue; // deleted since phase 1 — do not resurrect
                    };
                    if pointer != survivor.old_pointer {
                        continue; // overwritten since phase 1 — do not revert
                    }
                    let Some(moved) = version.next_system() else {
                        exhausted.push(survivor);
                        continue;
                    };
                    result.bytes_rewritten += self.relocate_locked(bucket, survivor, moved)?;
                }
            }

            // Stripe, then bucket (the write path's order), then the
            // compare-and-set again, and a fresh user seq drawn with the stripe
            // held: newer than every stored copy, older than any write to come.
            for survivor in exhausted {
                let _stripe = self.lock_key(&survivor.key);
                let _bucket_guard = self.value_log.lock_bucket_for_write(bucket)?;
                if self.lsm.get(&survivor.key)? != Some(survivor.old_pointer) {
                    continue;
                }
                let version = Version::user(self.alloc_seq());
                result.bytes_rewritten += self.relocate_locked(bucket, survivor, version)?;
            }

            result.pending_unlink.push(segment.id);
        }

        Ok(result)
    }

    /// Copy `survivor` to the active tail and point its key at the copy, at
    /// `version`. The caller holds `bucket`'s write lock and has checked the key
    /// still points at `survivor.old_pointer`. Returns the bytes written.
    fn relocate_locked(&self, bucket: u32, survivor: &Survivor, version: Version) -> Result<u64> {
        let key = survivor.key.as_slice();
        let meta = ValueRecordMeta {
            epoch: survivor.epoch,
            seq: version.user,
        };
        let new_pointer = self.value_log.append_to_locked_bucket(bucket, key, &survivor.value, meta, false)?;
        let applied = match self.lsm.insert_with_seq(key, new_pointer.to_u128(), version) {
            Ok(applied) => applied,
            Err(e) => {
                // Nothing points at the copy: count it as garbage, as the put
                // path does. The key keeps its old pointer, and this segment
                // stays until a later pass relocates it.
                self.value_log.note_displaced(new_pointer, key.len());
                return Err(e.into());
            }
        };
        // The old record, if the memtable held it, is now unreferenced (it lives
        // in a segment about to go). Held in a lower layer, it is dropped by the
        // next merge: counting it then finds no segment to charge once this one
        // is deleted.
        self.note_applied(applied, Some(new_pointer), key.len());
        Ok(new_pointer.record_len(key.len()))
    }

    /// Run value-log GC across every bucket.
    ///
    /// `segment_gc_threshold_pct` selects **segments** (`ThresholdConfig::segment_gc_threshold`).
    /// The bucket-level `value_log_waste_threshold` decides whether a namespace is
    /// collected at all, and is applied by the caller.
    ///
    /// # The ordering IS the crash-safety story
    ///
    /// ```text
    /// 1. append survivors into the active segment
    /// 2. CAS re-point the keys            (bucket lock)
    /// 3. flush the memtable to L0         ← the re-point is now DURABLE
    /// 4. unlink the old segments          ← only now
    /// ```
    ///
    /// At every crash point the durable LSM still refers to a segment that exists. Die
    /// before 3 and the keys still point at the old segment, which is still there; die
    /// between 3 and 4 and they point at the new one while the old segment is merely
    /// dead weight, which the next pass reclaims. Nothing has to be rolled forward or
    /// back, so GC keeps no journal and no commit marker.
    ///
    /// **Unlinking before step 3 would be data loss**: the durable LSM would point into
    /// a segment that no longer exists, and the WAL entries that could replay those
    /// writes are long gone.
    pub fn garbage_collect_with_threshold(&self, segment_gc_threshold_pct: f64) -> Result<GCStats> {
        self.garbage_collect_with_thresholds(segment_gc_threshold_pct, TAIL_GC_MIN_GARBAGE_PCT)
    }

    /// As [`garbage_collect_with_threshold`](Self::garbage_collect_with_threshold), but
    /// with an explicit tail-seal bar (`tail_gc_threshold_pct`) — the garbage share at
    /// which a bucket's never-filled active tail is sealed so it can be collected. The GC
    /// worker passes the configured
    /// [`effective_tail_gc_min_garbage_pct`](crate::db::config::ThresholdConfig::effective_tail_gc_min_garbage_pct),
    /// which by default tracks the waste trigger so no garbage strands between the two.
    pub fn garbage_collect_with_thresholds(&self, segment_gc_threshold_pct: f64, tail_gc_threshold_pct: f64) -> Result<GCStats> {
        let start_time = std::time::Instant::now();

        if self
            .value_log_gc_in_progress
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Ok(self.gc_stats_now(0, start_time));
        }
        struct InProgressGuard<'a>(&'a AtomicBool);
        impl Drop for InProgressGuard<'_> {
            fn drop(&mut self) {
                self.0.store(false, Ordering::SeqCst);
            }
        }
        let _in_progress = InProgressGuard(&self.value_log_gc_in_progress);

        // Steps 1-2, per bucket, in parallel: buckets are independent, each with its
        // own lock.
        let bucket_count = self.value_log.num_buckets();
        let results: Vec<BucketGCResult> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..bucket_count as u32)
                .map(|bucket| s.spawn(move || self.compact_bucket(bucket, segment_gc_threshold_pct, tail_gc_threshold_pct)))
                .collect();
            handles
                .into_iter()
                .filter_map(|h| match h.join() {
                    Ok(Ok(r)) => Some(r),
                    Ok(Err(e)) => {
                        error!("[KVStore '{}'] value-log GC failed for a bucket: {:?}", self.name, e);
                        None
                    }
                    Err(_) => {
                        error!("[KVStore '{}'] value-log GC thread panicked", self.name);
                        None
                    }
                })
                .collect()
        });

        if results.iter().all(|r| r.pending_unlink.is_empty()) {
            return Ok(self.gc_stats_now(0, start_time));
        }

        // Step 3: make the relocated copies durable, then the re-points to them, BEFORE
        // unlinking anything. Load-bearing — `test_gc_crash_the_instant_segments_are_
        // unlinked_loses_nothing` fails if the flush and the unlink are ever swapped.
        // The fsync matters for power loss rather than a process crash: survivors are
        // appended without one, so without it the old copy could be unlinked (and the
        // unlink made durable) while the new copy is still only in the page cache.
        self.value_log.sync_all()?;
        self.lsm.flush_memtable_to_level0()?;

        // Step 4: the old segments are now unreferenced by anything durable.
        let mut bytes_reclaimed = 0u64;
        let mut bytes_rewritten = 0u64;
        let mut segments_reclaimed = 0usize;
        for result in &results {
            let Ok(log) = self.value_log.get_bucket_log(result.bucket) else {
                continue;
            };
            let mut reclaimed_here = 0u64;
            for &segment_id in &result.pending_unlink {
                match log.unlink_segment(segment_id) {
                    Ok(bytes) => {
                        reclaimed_here += bytes;
                        segments_reclaimed += 1;
                    }
                    Err(e) => warn!(
                        "[KVStore '{}'] could not unlink value-log segment {} of bucket {}: {:?}",
                        self.name, segment_id, result.bucket, e
                    ),
                }
            }
            log.record_gc_run(reclaimed_here);
            bytes_reclaimed += reclaimed_here;
            bytes_rewritten += result.bytes_rewritten;
        }

        // Test-only crash point (see `gc_crash_after_unlink`): the segments are gone, and
        // nothing after this line has run.
        #[cfg(test)]
        if self.gc_crash_after_unlink.load(Ordering::SeqCst) {
            return Ok(self.gc_stats_now(bytes_reclaimed, start_time));
        }

        self.value_log.flush_all_metadata()?;

        info!(
            "[KVStore '{}'] value-log GC: reclaimed {} bytes across {} segment(s), rewrote {} bytes of survivors, in {:?}",
            self.name,
            bytes_reclaimed,
            segments_reclaimed,
            bytes_rewritten,
            start_time.elapsed()
        );
        if let Some(m) = self.metrics() {
            Metrics::bump(&m.vlog_gc_runs);
            Metrics::add(&m.vlog_gc_duration_ms, start_time.elapsed().as_millis() as u64);
            Metrics::add(&m.vlog_segments_reclaimed, segments_reclaimed as u64);
            Metrics::add(&m.vlog_gc_bytes_reclaimed, bytes_reclaimed);
            // Together with bytes_reclaimed this is GC's write amplification — the cost
            // of a pass against what it actually freed.
            Metrics::add(&m.vlog_gc_bytes_rewritten, bytes_rewritten);
        }

        Ok(self.gc_stats_now(bytes_reclaimed, start_time))
    }

    fn gc_stats_now(&self, bytes_reclaimed: u64, start_time: std::time::Instant) -> GCStats {
        let (total_gc_runs, total_bytes_reclaimed, bytes_live) = self.aggregate_value_log_stats();
        GCStats {
            bytes_reclaimed,
            bytes_live,
            gc_run_count: total_gc_runs,
            total_bytes_reclaimed,
            gc_duration_ms: start_time.elapsed().as_millis(),
        }
    }

    /// Does this namespace have value-log garbage GC can actually collect right now?
    ///
    /// The bucket-level waste ratio alone is not enough: garbage sitting in a bucket's
    /// active tail is not collectable until the tail is sealed, so a small, fully deleted
    /// namespace can show 100% waste while GC has nothing to do. Without this check the
    /// worker would wake, log "starting GC", reclaim nothing, and repeat on every tick.
    pub(crate) fn has_gc_work(&self, segment_gc_threshold_pct: f64, tail_gc_threshold_pct: f64) -> bool {
        (0..self.value_log.num_buckets() as u32).any(|b| {
            self.value_log
                .get_bucket_log(b)
                .is_ok_and(|log| log.has_gc_work(segment_gc_threshold_pct, tail_gc_threshold_pct))
        })
    }

    /// Waste across the whole namespace: `garbage / (live + garbage)`, as a percentage.
    /// This is the **trigger** the GC worker compares against `value_log_waste_threshold`.
    pub fn get_waste_ratio(&self) -> f64 {
        self.value_log.total_garbage_ratio()
    }

    /// True when at least one bucket is at or above `threshold_pct` waste. The GC worker
    /// checks this alongside the namespace-average [`get_waste_ratio`](Self::get_waste_ratio),
    /// so a hot bucket over the trigger is collected even when near-empty buckets drag the
    /// average below it.
    pub(crate) fn has_bucket_over_waste(&self, threshold_pct: f64) -> bool {
        self.value_log.any_bucket_over_waste(threshold_pct)
    }

    /// The raw `(garbage_bytes, written_bytes)` behind [`get_waste_ratio`](Self::get_waste_ratio),
    /// where `written = live + garbage` — useful in logs, so a high ratio over a tiny
    /// absolute volume is obvious.
    pub(crate) fn waste_bytes(&self) -> (u64, u64) {
        let garbage = self.value_log.total_garbage_bytes();
        (garbage, garbage.saturating_add(self.value_log.total_live_bytes()))
    }

    pub(crate) fn aggregate_value_log_stats(&self) -> (u64, u64, u64) {
        let mut total_gc_runs = 0u64;
        let mut total_bytes_reclaimed = 0u64;
        let mut bytes_live = 0u64;
        for (_bucket, metadata) in self.value_log.all_bucket_metadata() {
            total_gc_runs = total_gc_runs.saturating_add(metadata.total_gc_runs);
            total_bytes_reclaimed = total_bytes_reclaimed.saturating_add(metadata.total_bytes_reclaimed);
            bytes_live = bytes_live.saturating_add(metadata.live_bytes());
        }
        (total_gc_runs, total_bytes_reclaimed, bytes_live)
    }
}
