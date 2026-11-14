use super::*;
use tempfile::TempDir;

const SEG: u64 = 64 * 1024; // the minimum legal segment size

fn meta(seq: u64) -> ValueRecordMeta {
    ValueRecordMeta {
        epoch: 1_700_000_000_000,
        seq,
    }
}

/// Unsynced appends are tracked until something makes them durable: `sync`,
/// an append with `sync = true`, or sealing the segment (which fsyncs it, so
/// only the fresh tail can hold unsynced bytes afterwards).
/// A segment written in another format must stop the open, not be read with
/// this layout: its records would fail to parse, and the tail repair would
/// truncate the segment to its header. A file too short to hold a header has
/// no records to misread and still opens.
#[test]
fn open_refuses_a_segment_in_another_format() -> Result<()> {
    let dir = TempDir::new()?;
    let id = {
        let log = ValueLog::open(dir.path(), 0, SEG)?;
        log.append(b"k", &[7u8; 100], meta(1), true)?;
        log.inner.read().active_id
    };
    let path = ValueLog::segment_path(dir.path(), 0, id);
    let len = std::fs::metadata(&path)?.len();
    let file = OpenOptions::new().write(true).open(&path)?;
    file.write_all_at(&(SEGMENT_FORMAT_VERSION - 1).to_le_bytes(), 4)?;
    drop(file);

    let err = ValueLog::open(dir.path(), 0, SEG).err().expect("an older format must not open");
    assert!(matches!(err, ValueLogError::UnsupportedSegmentFormat { .. }), "{err}");
    assert_eq!(std::fs::metadata(&path)?.len(), len, "the refused segment must be left untouched");

    let fresh = TempDir::new()?;
    File::create(ValueLog::segment_path(fresh.path(), 0, 1))?;
    ValueLog::open(fresh.path(), 0, SEG)?;
    Ok(())
}

#[test]
fn unsynced_bytes_track_what_a_power_loss_could_take() -> Result<()> {
    let dir = TempDir::new()?;
    let log = ValueLog::open(dir.path(), 0, SEG)?;
    assert_eq!(log.unsynced_bytes(), 0);

    log.append(b"k1", &[1u8; 100], meta(1), false)?;
    let one = ValueRecordHeader::record_len(2, 100);
    assert_eq!(log.unsynced_bytes(), one);
    log.sync()?;
    assert_eq!(log.unsynced_bytes(), 0);

    log.append(b"k2", &[2u8; 100], meta(2), false)?;
    log.append(b"k3", &[3u8; 100], meta(3), true)?; // a synced append covers what came before
    assert_eq!(log.unsynced_bytes(), 0);

    // Fill past a segment boundary without syncing: after the roll only the new
    // tail's records are unsynced — the sealed segment was fsynced as it sealed.
    let first = log.inner.read().active_id;
    let mut in_new_tail = 0;
    for i in 0..1_000u64 {
        log.append(b"kk", &[9u8; 1000], meta(10 + i), false)?;
        if log.inner.read().active_id != first {
            in_new_tail += ValueRecordHeader::record_len(2, 1000);
            if in_new_tail >= 3 * ValueRecordHeader::record_len(2, 1000) {
                break;
            }
        }
    }
    assert_ne!(log.inner.read().active_id, first, "setup: the tail must have rolled");
    assert_eq!(log.unsynced_bytes(), in_new_tail);
    Ok(())
}

/// A batch read merges nearby records into one `pread` per run; every value it
/// returns must still be exactly what a single read of that record returns,
/// for adjacent records, records with gaps above and below the merge limit,
/// records in several segments, repeated positions and a pointer that names
/// no record (which fails alone without taking its run down with it).
#[test]
fn batch_reads_return_what_single_reads_do() -> Result<()> {
    let dir = TempDir::new()?;
    let log = ValueLog::open(dir.path(), 0, SEG)?;
    let mut s = 0x853c_49e6_748f_ea9b_u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    // Values of 1–2,000 bytes cross the merge gap both ways and roll several segments.
    let mut locs = Vec::new();
    for i in 0..600u64 {
        let len = 1 + (next() % 2_000) as usize;
        let value: Vec<u8> = (0..len).map(|j| (i as usize * 31 + j) as u8).collect();
        locs.push((log.append(&i.to_be_bytes(), &value, meta(i + 1), false)?, value));
    }
    assert!(log.inner.read().active_id > 2, "setup: records must span several segments");

    for density in [100u64, 60, 10, 1] {
        // Which record each batch position reads; the batch's index is the position.
        let mut records: Vec<usize> = (0..locs.len()).filter(|_| next() % 100 < density).collect();
        if let Some(&first) = records.first() {
            records.push(first); // the same record twice
        }
        let mut batch: Vec<(usize, ValueLocation)> = records.iter().enumerate().map(|(pos, &r)| (pos, locs[r].0)).collect();
        // A pointer into the middle of a real record names no record.
        let bogus = batch.len();
        let mut bad = locs[3].0;
        bad.rec_offset += 1;
        batch.push((bogus, bad));

        let mut out = Vec::new();
        let results = log.read_values_batch_into(&batch, &mut out);
        assert_eq!(results.len(), batch.len());
        for (pos, range) in results {
            if pos == bogus {
                assert!(range.is_none(), "a pointer to no record must fail");
                continue;
            }
            let range = range.expect("a real record must read");
            let expected = &locs[records[pos]].1;
            assert_eq!(&out[range], expected.as_slice(), "density {density}%, position {pos}");
            assert_eq!(&log.read_value(batch[pos].1)?, expected);
        }
    }
    Ok(())
}

#[test]
fn record_round_trips_key_and_value() -> Result<()> {
    let dir = TempDir::new()?;
    let log = ValueLog::open(dir.path(), 0, SEG)?;

    let loc = log.append(b"the-key", b"the-value", meta(7), true)?;
    // The pointer carries the value length, which is what makes a read one pread.
    assert_eq!(loc.value_len, b"the-value".len() as u32);

    let (value, m) = log.read_value_and_meta(loc)?;
    assert_eq!(value, b"the-value");
    assert_eq!(m.seq, 7);

    // The key is in the record too — this is what lets GC decide liveness without
    // inverting the whole LSM.
    let mut seen = Vec::new();
    log.for_each_record(loc.segment_id, |key, value, _, _| {
        seen.push((key.to_vec(), value));
    })?;
    assert_eq!(seen, vec![(b"the-key".to_vec(), b"the-value".to_vec())]);
    Ok(())
}

#[test]
fn appends_roll_into_new_segments_when_full() -> Result<()> {
    let dir = TempDir::new()?;
    let log = ValueLog::open(dir.path(), 0, SEG)?;

    let value = vec![0xAB; 8 * 1024];
    let mut locations = Vec::new();
    for i in 0..20 {
        locations.push(log.append(format!("k{i}").as_bytes(), &value, meta(i as u64), false)?);
    }

    let segments: Vec<u32> = locations.iter().map(|l| l.segment_id).collect();
    assert!(
        segments.windows(2).any(|w| w[0] != w[1]),
        "20 × 8 KiB records cannot fit one 64 KiB segment — expected a rollover, got {segments:?}"
    );
    // Ids only ever go up.
    assert!(segments.windows(2).all(|w| w[0] <= w[1]), "segment ids must be monotone");

    for (i, loc) in locations.iter().enumerate() {
        assert_eq!(log.read_value(*loc)?, value, "record {i} unreadable after rollover");
    }
    Ok(())
}

#[test]
fn segment_ids_are_never_reused_even_after_the_newest_is_unlinked() -> Result<()> {
    // The landmine: if the next id were `max(existing files) + 1`, unlinking the
    // newest segment would hand its id out again — and a reader holding the old
    // pointer would silently read a different record.
    let dir = TempDir::new()?;
    let log = ValueLog::open(dir.path(), 0, SEG)?;

    let value = vec![0xCD; 8 * 1024];
    let mut ids = Vec::new();
    for i in 0..12 {
        ids.push(log.append(format!("k{i}").as_bytes(), &value, meta(i as u64), false)?.segment_id);
    }
    let max_ever = *ids.iter().max().unwrap();

    // Unlink the newest SEALED segment: on disk, `max(files)` now skips its id.
    let highest_sealed = *ids.iter().filter(|&&id| id != log.inner.read().active_id).max().unwrap();
    log.unlink_segment(highest_sealed)?;

    // Keep appending so fresh segments get created, and check every NEWLY created
    // id is above everything ever handed out — including the one we just freed.
    for i in 20..60 {
        let id = log.append(format!("n{i}").as_bytes(), &value, meta(i as u64), false)?.segment_id;
        if !ids.contains(&id) {
            assert!(
                id > max_ever,
                "segment id {id} was reused (max ever handed out was {max_ever}); \
                 ids must come from the monotone counter, never max(existing files) + 1"
            );
        }
    }
    Ok(())
}

#[test]
fn segment_id_high_water_mark_survives_restart() -> Result<()> {
    let dir = TempDir::new()?;
    let value = vec![0xEF; 8 * 1024];

    let last_id = {
        let log = ValueLog::open(dir.path(), 0, SEG)?;
        let mut id = 0;
        for i in 0..12 {
            id = log.append(format!("k{i}").as_bytes(), &value, meta(i as u64), false)?.segment_id;
        }
        // Drop every sealed segment, so the filesystem no longer remembers those ids.
        let sealed: Vec<u32> = log.segment_stats().iter().filter(|s| s.sealed).map(|s| s.id).collect();
        for id in sealed {
            log.unlink_segment(id)?;
        }
        log.flush_metadata()?;
        id
    };

    // Reopen: the id allocator must not regress, even though the files that used
    // those ids are gone. The persisted high-water mark is what guarantees it.
    let log = ValueLog::open(dir.path(), 0, SEG)?;
    let next = log.append(b"after-restart", &value, meta(99), false)?.segment_id;
    assert!(
        next >= last_id,
        "segment id regressed across restart: was {last_id}, handed out {next} — the high-water mark was lost"
    );
    Ok(())
}

#[test]
fn displaced_records_become_garbage_without_touching_the_record() -> Result<()> {
    let dir = TempDir::new()?;
    let log = ValueLog::open(dir.path(), 0, SEG)?;

    let old = log.append(b"k", &vec![1u8; 1000], meta(1), false)?;
    let stats = log.segment_stats();
    let live_before = stats[0].live_bytes;
    assert_eq!(stats[0].garbage_bytes, 0);

    log.note_displaced(old, b"k".len());

    let stats = log.segment_stats();
    assert_eq!(stats[0].garbage_bytes, live_before, "the displaced record's bytes become garbage");
    assert_eq!(stats[0].live_bytes, 0);

    // The record itself is untouched — its segment may be sealed and immutable, and
    // liveness is the LSM's business, not the record's.
    assert_eq!(log.read_value(old)?, vec![1u8; 1000]);
    Ok(())
}

#[test]
fn gc_candidates_are_sealed_segments_over_the_threshold_worst_first() -> Result<()> {
    let dir = TempDir::new()?;
    let log = ValueLog::open(dir.path(), 0, SEG)?;

    let value = vec![0x11; 8 * 1024];
    let mut locs = Vec::new();
    for i in 0..20 {
        locs.push(log.append(format!("k{i}").as_bytes(), &value, meta(i as u64), false)?);
    }
    // Garbage the records in the earliest segment only.
    let first = locs[0].segment_id;
    for loc in locs.iter().filter(|l| l.segment_id == first) {
        log.note_displaced(*loc, 2);
    }

    let candidates = log.gc_candidates(30.0);
    assert!(candidates.iter().any(|c| c.id == first), "the garbage-heavy segment must be a candidate");
    assert!(
        candidates.iter().all(|c| c.sealed),
        "the active tail must never be collected — it is still being appended to"
    );
    assert!(
        candidates.windows(2).all(|w| w[0].garbage_ratio_pct() >= w[1].garbage_ratio_pct()),
        "candidates must be worst-first"
    );
    Ok(())
}

#[test]
fn a_reader_holding_an_open_handle_survives_the_unlink() -> Result<()> {
    // POSIX: an open fd outlives unlink. This is what lets GC delete a segment
    // without coordinating with in-flight readers.
    let dir = TempDir::new()?;
    let log = ValueLog::open(dir.path(), 0, SEG)?;

    let value = vec![0x77; 8 * 1024];
    let mut locs = Vec::new();
    for i in 0..12 {
        locs.push(log.append(format!("k{i}").as_bytes(), &value, meta(i as u64), false)?);
    }
    let sealed = log.segment_stats().iter().find(|s| s.sealed).map(|s| s.id).expect("a sealed segment");
    let victim = *locs.iter().find(|l| l.segment_id == sealed).unwrap();

    let handle = log.segment_file(sealed)?; // a reader's in-flight handle
    log.unlink_segment(sealed)?;

    // The held handle still reads correctly...
    assert_eq!(log.read_value_from_file(&handle, victim)?.0, value);
    // ...while a fresh resolution reports the segment is gone, so the caller
    // re-resolves through the LSM. Loud, local, retryable — never a wrong value.
    assert!(
        matches!(log.read_value(victim), Err(ValueLogError::SegmentMissing(id)) if id == sealed),
        "a pointer into a reclaimed segment must fail loudly, not silently resolve"
    );
    Ok(())
}

#[test]
fn invalid_segment_sizes_are_rejected() {
    let dir = TempDir::new().unwrap();
    for bad in [0, 4096, SEG + 1, u32::MAX as u64 + 4096] {
        assert!(
            matches!(ValueLog::open(dir.path(), 0, bad), Err(ValueLogError::InvalidSegmentSize(got)) if got == bad),
            "segment size {bad} must be rejected"
        );
    }
}

#[test]
fn a_value_too_large_for_a_segment_is_rejected() -> Result<()> {
    let dir = TempDir::new()?;
    let log = ValueLog::open(dir.path(), 0, SEG)?;
    let huge = vec![0u8; SEG as usize];
    assert!(
        matches!(log.append(b"k", &huge, meta(1), false), Err(ValueLogError::ValueTooLarge { .. })),
        "a record that cannot fit a segment must be refused, not silently truncated"
    );
    Ok(())
}

#[test]
fn stats_rebuild_from_the_lsm_when_metadata_is_lost() -> Result<()> {
    let dir = TempDir::new()?;
    let value = vec![0x33; 4 * 1024];
    let (live_loc, dead_loc) = {
        let log = ValueLog::open(dir.path(), 0, SEG)?;
        let live = log.append(b"live", &value, meta(1), false)?;
        let dead = log.append(b"dead", &value, meta(2), false)?;
        log.flush_metadata()?;
        (live, dead)
    };

    // Lose the metadata file, as a crash before the flush would.
    std::fs::remove_file(dir.path().join("value_log_0.metadata"))?;

    let log = ValueLog::open(dir.path(), 0, SEG)?;
    assert!(log.stats_need_rebuild(), "a missing metadata file must be flagged for rebuild");

    // The caller recomputes liveness from the LSM's pointers: only `live` is still
    // referenced, so `dead`'s bytes are garbage.
    let live_bytes = ValueRecordHeader::record_len(b"live".len(), value.len());
    let mut live_map = HashMap::new();
    live_map.insert(live_loc.segment_id, live_bytes);
    log.rebuild_stats(&live_map);

    assert!(!log.stats_need_rebuild());
    let stats = log.segment_stats();
    let seg = stats.iter().find(|s| s.id == live_loc.segment_id).unwrap();
    assert_eq!(seg.live_bytes, live_bytes);
    assert!(seg.garbage_bytes > 0, "the unreferenced record must be counted as garbage");
    assert_eq!(log.read_value(dead_loc)?, value, "the bytes are still there — just unreferenced");
    Ok(())
}

#[test]
fn a_corrupted_value_is_detected_when_checksums_are_verified() -> Result<()> {
    // Values carry a CRC32 of their payload, and `verify_checksums_on_read`
    // makes reads re-check it. Without it the structural checks (length, offset)
    // would serve bit-rotted bytes.
    let dir = TempDir::new()?;
    let log = ValueLog::open(dir.path(), 0, SEG)?;
    let loc = log.append(b"k", b"the-original-value", meta(1), true)?;

    // Flip a byte inside the value payload, in place, behind the log's back.
    let path = ValueLog::segment_path(dir.path(), 0, loc.segment_id);
    let mut bytes = std::fs::read(&path)?;
    let value_at = loc.rec_offset as usize + ValueRecordHeader::SIZE;
    bytes[value_at] ^= 0xFF;
    std::fs::write(&path, &bytes)?;

    // A fresh log (so nothing is cached) with verification off still returns the
    // corrupted bytes — that is the documented latency-first default.
    let log = ValueLog::open(dir.path(), 0, SEG)?;
    assert_ne!(log.read_value(loc)?, b"the-original-value", "sanity: the bytes really were corrupted");

    // With verification on, the read fails instead of serving them.
    log.set_verify_checksums_on_read(true);
    assert!(
        matches!(log.read_value(loc), Err(ValueLogError::CorruptedLog)),
        "a corrupted value must be rejected, not served"
    );
    Ok(())
}

#[test]
fn a_truncated_record_read_is_corruption_not_a_zero_filled_value() -> Result<()> {
    // `FileExt::read_at` can return a short count near EOF and leave the tail of the
    // buffer zeroed. A read of a record the pointer says is present must not serve
    // those zero bytes as the value — a file truncated below the record is corruption,
    // and must fail loudly rather than silently returning a wrong value.
    let dir = TempDir::new()?;
    let log = ValueLog::open(dir.path(), 0, SEG)?;
    let loc = log.append(b"k", b"the-full-value-bytes", meta(1), true)?;

    // Chop the last 8 bytes behind the log's back — no reopen, so the read path (not
    // the on-open tail-truncation) is what has to catch the short read.
    let path = ValueLog::segment_path(dir.path(), 0, loc.segment_id);
    let full = std::fs::metadata(&path)?.len();
    let f = OpenOptions::new().write(true).open(&path)?;
    f.set_len(full - 8)?;
    f.sync_all()?;

    assert!(
        matches!(log.read_value(loc), Err(ValueLogError::CorruptedLog)),
        "a truncated record must fail with CorruptedLog, not return a zero-filled value"
    );
    Ok(())
}

#[test]
fn a_torn_tail_from_a_crash_mid_append_is_ignored() -> Result<()> {
    // A crash can leave a partial record at the end of the active segment. Nothing
    // references it — the LSM only ever learned about records whose append
    // returned — so the scan must stop cleanly there and keep everything before it.
    let dir = TempDir::new()?;
    let log = ValueLog::open(dir.path(), 0, SEG)?;
    let a = log.append(b"a", b"value-a", meta(1), true)?;
    let b = log.append(b"b", b"value-b", meta(2), true)?;

    // Simulate the torn write: garbage bytes appended past the last good record.
    let path = ValueLog::segment_path(dir.path(), 0, a.segment_id);
    let mut bytes = std::fs::read(&path)?;
    bytes.extend_from_slice(&[0xAB; 20]); // shorter than a header, and not one anyway
    std::fs::write(&path, &bytes)?;

    let log = ValueLog::open(dir.path(), 0, SEG)?;
    let mut seen = Vec::new();
    log.for_each_record(a.segment_id, |key, _, _, _| seen.push(key.to_vec()))?;
    assert_eq!(
        seen,
        vec![b"a".to_vec(), b"b".to_vec()],
        "the complete records must survive the torn tail"
    );

    // ...and both are still readable by pointer.
    assert_eq!(log.read_value(a)?, b"value-a");
    assert_eq!(log.read_value(b)?, b"value-b");
    Ok(())
}

#[test]
fn a_segment_longer_than_its_saved_count_is_flagged_for_rebuild() -> Result<()> {
    // No roll here, unlike `stale_but_present_metadata_is_rebuilt_not_trusted`: a
    // new segment has no saved accounting and is flagged for that reason, so that
    // test passes even if a disagreeing size is ignored.
    let dir = TempDir::new()?;
    let value = vec![0x42; 1024];
    let rec = ValueRecordHeader::record_len(b"k0".len(), value.len());
    let seg = {
        let log = ValueLog::open(dir.path(), 0, SEG)?;
        let first = log.append(b"k0", &value, meta(0), true)?;
        log.flush_metadata()?;
        let second = log.append(b"k1", &value, meta(1), true)?;
        assert_eq!(second.segment_id, first.segment_id, "setup: one segment only");
        first.segment_id
    };

    let log = ValueLog::open(dir.path(), 0, SEG)?;
    assert!(
        log.stats_need_rebuild(),
        "a segment holding bytes its saved count does not cover must be flagged"
    );
    let stats = log.segment_stats().into_iter().find(|s| s.id == seg).unwrap();
    assert_eq!(stats.total_bytes, 2 * rec, "the file is believed, not the saved count");
    assert_eq!(stats.live_bytes, 2 * rec, "everything counts as live until the rebuild");
    assert_eq!(stats.garbage_bytes, 0);
    Ok(())
}

#[test]
fn a_tail_cut_back_below_its_saved_count_is_flagged_for_rebuild() -> Result<()> {
    // The file's length matches the saved count, so only walking the records shows
    // the disagreement: a crash can keep a file's length and lose its last page.
    let dir = TempDir::new()?;
    let value = vec![0x42; 1024];
    let rec = ValueRecordHeader::record_len(b"k0".len(), value.len());
    let seg = {
        let log = ValueLog::open(dir.path(), 0, SEG)?;
        let first = log.append(b"k0", &value, meta(0), true)?;
        log.append(b"k1", &value, meta(1), true)?;
        log.flush_metadata()?;
        first.segment_id
    };
    let path = ValueLog::segment_path(dir.path(), 0, seg);
    let file = OpenOptions::new().write(true).open(&path)?;
    let len = file.metadata()?.len();
    file.write_all_at(&vec![0u8; rec as usize], len - rec)?;
    drop(file);

    let log = ValueLog::open(dir.path(), 0, SEG)?;
    assert!(
        log.stats_need_rebuild(),
        "a tail shorter than the saved count must be flagged, even though the file length agrees"
    );
    let stats = log.segment_stats().into_iter().find(|s| s.id == seg).unwrap();
    assert_eq!(stats.total_bytes, rec, "only the first record is left");
    assert_eq!(stats.live_bytes, rec, "everything counts as live until the rebuild");
    assert_eq!(stats.garbage_bytes, 0);
    Ok(())
}

#[test]
fn stale_but_present_metadata_is_rebuilt_not_trusted() -> Result<()> {
    // Metadata is not flushed on every append. A crash can leave the metadata file
    // present and parseable but describing FEWER bytes than a segment actually
    // holds: a segment that was the active tail when metadata was last flushed can
    // take more records and then *seal* (roll) before the crash. Trusting the stale
    // count undercounts that segment's garbage — and the from-LSM rebuild multiplies
    // the error, since it derives `garbage = total_bytes - live`. Open must reconcile
    // total_bytes to the file length, and re-seal the rolled segment, before the
    // rebuild runs. (`stats_rebuild_from_the_lsm_when_metadata_is_lost` covers only
    // a *missing* metadata file, where the whole persisted view is discarded.)
    let dir = TempDir::new()?;
    let value = vec![0x42; 20 * 1024];
    let rec = ValueRecordHeader::record_len(b"k0".len(), value.len());

    let locs = {
        let log = ValueLog::open(dir.path(), 0, SEG)?;
        // One record, then flush: metadata records this segment as the active tail
        // holding a single record.
        let r0 = log.append(b"k0", &value, meta(0), true)?;
        log.flush_metadata()?;
        // Two more fill it and a fourth rolls into a new segment — sealing the first
        // with three records, two of which the stale metadata never learned about.
        // Dropped without another flush = the crash state (no flush-on-drop).
        let r1 = log.append(b"k1", &value, meta(1), true)?;
        let r2 = log.append(b"k2", &value, meta(2), true)?;
        let r3 = log.append(b"k3", &value, meta(3), true)?;
        vec![r0, r1, r2, r3]
    };
    let sealed_seg = locs[0].segment_id;
    assert!(
        locs[1].segment_id == sealed_seg && locs[2].segment_id == sealed_seg,
        "k0..k2 must share the first segment for this test to exercise a sealed stale segment"
    );
    assert_ne!(
        locs[3].segment_id, sealed_seg,
        "k3 must have rolled into a new segment, sealing the first"
    );

    let log = ValueLog::open(dir.path(), 0, SEG)?;
    assert!(
        log.stats_need_rebuild(),
        "a segment file longer than its persisted total_bytes must be flagged for rebuild"
    );

    let seg = log.segment_stats().into_iter().find(|s| s.id == sealed_seg).unwrap();
    assert_eq!(
        seg.total_bytes,
        3 * rec,
        "total_bytes must reflect the file (three records), not the stale one-record count"
    );
    assert!(
        seg.sealed,
        "a segment that rolled before the crash must be re-sealed on reopen, or GC would never collect it"
    );

    // Rebuild from the LSM: only k0 (here) and k3 (in the next segment) are still
    // live; k1 and k2 are garbage. The garbage must count both records beyond the
    // stale one-record total — only possible once total_bytes was corrected.
    let mut live = HashMap::new();
    live.insert(sealed_seg, rec); // k0
    live.insert(locs[3].segment_id, rec); // k3
    log.rebuild_stats(&live);
    let seg = log.segment_stats().into_iter().find(|s| s.id == sealed_seg).unwrap();
    assert_eq!(seg.live_bytes, rec);
    assert_eq!(
        seg.garbage_bytes,
        2 * rec,
        "both records past the stale count must be reclaimable garbage"
    );

    // Nothing was lost — every record still reads back.
    for (i, loc) in locs.iter().enumerate() {
        assert_eq!(log.read_value(*loc)?, value, "record {i} unreadable after stale-metadata recovery");
    }
    Ok(())
}

#[test]
fn records_appended_after_a_torn_tail_stay_visible_to_gc() -> Result<()> {
    // The dangerous sequence: a crash leaves a torn record at the tail, then on
    // reopen we resume appending into that same not-full segment. If the torn bytes
    // were left in place, a record appended after them would sit beyond a gap that
    // GC's sequential scan stops at — GC would relocate only the records before the
    // gap and then unlink the segment, silently losing the later live record.
    let dir = TempDir::new()?;
    let (a, b) = {
        let log = ValueLog::open(dir.path(), 0, SEG)?;
        let a = log.append(b"a", b"value-a", meta(1), true)?;
        let b = log.append(b"b", b"value-b", meta(2), true)?;
        (a, b)
    };

    // Simulate the torn write: partial bytes appended past the last good record.
    let path = ValueLog::segment_path(dir.path(), 0, a.segment_id);
    let mut bytes = std::fs::read(&path)?;
    bytes.extend_from_slice(&[0xAB; 20]); // shorter than a header, and not one anyway
    std::fs::write(&path, &bytes)?;

    // Reopen (recovery) and resume appending into the reused tail.
    let log = ValueLog::open(dir.path(), 0, SEG)?;
    let c = log.append(b"c", b"value-c", meta(3), true)?;
    assert_eq!(
        c.segment_id, a.segment_id,
        "c must land in the reused tail, exercising the torn-tail path"
    );

    // GC learns a segment's contents by scanning it. All three records — including
    // the one appended after the torn tail — must be visited in order, or GC would
    // drop c while unlinking the segment.
    let mut seen = Vec::new();
    log.for_each_record(a.segment_id, |key, _, _, _| seen.push(key.to_vec()))?;
    assert_eq!(
        seen,
        vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()],
        "the record appended after the torn tail must be visible to GC"
    );

    // The torn bytes were truncated, not skipped: c sits immediately after b, and
    // every pointer still resolves to its own value.
    assert_eq!(log.read_value(a)?, b"value-a");
    assert_eq!(log.read_value(b)?, b"value-b");
    assert_eq!(log.read_value(c)?, b"value-c");
    Ok(())
}

#[test]
fn reads_stay_correct_when_the_fd_cache_evicts() -> Result<()> {
    // The fd cache is bounded (MAX_OPEN_SEGMENTS_PER_BUCKET). A bucket with more
    // segments than that must still read correctly from the evicted ones — they are
    // immutable, so an evicted handle is simply reopened on demand.
    let dir = TempDir::new()?;
    let log = ValueLog::open(dir.path(), 0, SEG)?;

    // ~2 records per 64 KiB segment → comfortably more segments than the cache holds.
    let value = vec![0x5A; 30 * 1024];
    let mut locs = Vec::new();
    for i in 0..(MAX_OPEN_SEGMENTS_PER_BUCKET * 3) {
        locs.push(log.append(format!("k{i:04}").as_bytes(), &value, meta(i as u64), false)?);
    }
    let distinct: std::collections::HashSet<u32> = locs.iter().map(|l| l.segment_id).collect();
    assert!(
        distinct.len() > MAX_OPEN_SEGMENTS_PER_BUCKET,
        "this test needs more segments ({}) than the fd cache holds ({MAX_OPEN_SEGMENTS_PER_BUCKET})",
        distinct.len()
    );

    // Read the OLDEST records last — they are the ones evicted from the cache.
    for (i, loc) in locs.iter().enumerate() {
        assert_eq!(log.read_value(*loc)?, value, "record {i} unreadable after its segment was evicted");
    }
    Ok(())
}

#[test]
fn live_and_garbage_always_sum_to_total() -> Result<()> {
    let dir = TempDir::new()?;
    let log = ValueLog::open(dir.path(), 0, SEG)?;

    let value = vec![0x9C; 4 * 1024];
    let mut locs = Vec::new();
    for i in 0..12 {
        locs.push(log.append(format!("k{i}").as_bytes(), &value, meta(i as u64), false)?);
    }
    for (i, loc) in locs.iter().enumerate() {
        if i % 3 == 0 {
            log.note_displaced(*loc, format!("k{i}").len());
        }
    }

    for s in log.segment_stats() {
        assert_eq!(
            s.live_bytes + s.garbage_bytes,
            s.total_bytes,
            "segment {} broke the live+garbage==total invariant that GC selection relies on",
            s.id
        );
    }
    Ok(())
}
