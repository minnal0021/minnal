//! SSTable format: headers, versions, frames, the entry codec, and opening damaged files.

use super::*;

// The entry CRC covers only the body, so it cannot detect a *layout* change:
// the bytes are identical and only their interpretation differs. That is why
// SSTable files carry a magic + format version: without one, a file in
// another layout would pass the CRC and be misread, yielding garbage pointers
// instead of an error. These pin the guard that makes such a file fail loudly.
#[test]
fn test_pre_header_sstable_is_rejected_not_misread() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let path = temp_dir.path().join("legacy.dat");

    // A file without a header: entry frames straight from offset 0.
    {
        let entry = SStableEntry {
            key: b"k".to_vec(),
            key_prefix: key_prefix_of(b"k"),
            value: 42,
            tombstone: false,
            seq: 7,
        };
        let payload = encode_sstable_entry(&entry)?;
        let mut f = File::create(&path)?;
        f.write_all(&(payload.len() as u32).to_le_bytes())?;
        f.write_all(&payload)?;
        f.sync_all()?;
    }

    let err = open_sstable_for_scan(&path).expect_err("a pre-header SSTable must be rejected, not parsed");
    assert!(
        matches!(err, LSMError::Corruption(ref m) if m.contains("magic")),
        "expected a bad-magic corruption error, got: {err:?}"
    );
    Ok(())
}

#[test]
fn test_sstable_with_unknown_format_version_is_rejected() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let path = temp_dir.path().join("future.dat");

    // Correct magic, but a version this build does not know how to read.
    let mut header = sstable_header_bytes();
    header[4..8].copy_from_slice(&(SSTABLE_FORMAT_VERSION + 1).to_le_bytes());
    let mut f = File::create(&path)?;
    f.write_all(&header)?;
    f.sync_all()?;
    drop(f);

    let err = open_sstable_for_scan(&path).expect_err("an unknown format version must be rejected");
    assert!(
        matches!(err, LSMError::Corruption(ref m) if m.contains("format version")),
        "expected a format-version corruption error, got: {err:?}"
    );
    Ok(())
}

// The guard is only useful if it fires on the path a real database takes at
// startup — `LSMTree::open` deliberately tolerates a partial metadata load, so
// this checks that leniency does not swallow an unreadable format.
#[test]
fn test_open_rejects_l1_file_written_in_an_older_format() -> Result<()> {
    let temp_dir = TempDir::new()?;
    {
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;
        lsm.insert_with_seq(b"k", 0xAAAA, 1)?;
        lsm.flush_memtable_to_level0()?;
        lsm.compact_all()?; // → a real, header-bearing L1 file
    }

    // Strip the header, leaving a file this build cannot identify.
    let bucket = get_bucket_for_key(b"k", crate::support::TEST_NUM_BUCKETS);
    let l1 = LSMTree::level1_path_from(temp_dir.path(), bucket);
    let body = std::fs::read(&l1)?;
    assert!(body.len() as u64 > SSTABLE_DATA_START, "setup: L1 file should hold entries");
    std::fs::write(&l1, &body[SSTABLE_DATA_START as usize..])?;

    // `LSMTree` is not `Debug`, so match on the result rather than `expect_err`.
    match LSMTree::open(temp_dir.path(), test_lsm_config()) {
        Ok(_) => panic!("open must reject an unreadable L1 format, but it succeeded"),
        Err(LSMError::Corruption(m)) => assert!(m.contains("magic"), "expected a bad-magic corruption error, got: {m}"),
        Err(e) => panic!("expected a corruption error, got: {e:?}"),
    }
    Ok(())
}

// A frame's 4-byte length prefix sits *outside* the entry CRC (which covers only
// the body), so a flipped bit there is undetectable until after the payload has
// been read. A reader that allocated from it first could request up to 4 GiB
// from a file of a few dozen bytes.
//
// A reader without the guard also ends in an error here, so the assertion has
// to be on *which* error: the guard reports the impossible length itself,
// whereas an unguarded reader allocates, tries to fill the buffer, and surfaces
// an unrelated `Io(UnexpectedEof)`, having already committed the allocation.
#[test]
fn test_corrupt_frame_length_is_rejected_before_allocating() -> Result<()> {
    let temp_dir = TempDir::new()?;

    for bogus in [u32::MAX, 1 << 30, 0] {
        let path = temp_dir.path().join(format!("frame_{bogus}.dat"));
        let mut f = File::create(&path)?;
        f.write_all(&sstable_header_bytes())?;
        f.write_all(&bogus.to_le_bytes())?; // claims `bogus` bytes…
        f.write_all(&[0xABu8; 8])?; // …but the file holds 8
        f.sync_all()?;
        drop(f);

        let file_len = std::fs::metadata(&path)?.len();
        assert!(file_len < 64, "setup: the file must be far smaller than the claimed frame");

        let err = LSMTree::level0_file_summary(&path)
            .map(|_| ())
            .expect_err("a frame longer than its file must be rejected");
        match err {
            LSMError::Corruption(m) => assert!(
                m.contains("declares"),
                "expected the frame-length guard to fire, got a different corruption: {m}"
            ),
            other => panic!("expected the frame-length guard to fire before allocating, got: {other:?}"),
        }
    }
    Ok(())
}

/// Write `bodies` as consecutive `u32 LE len ‖ body` frames after `lead` junk bytes.
fn write_frames(path: &Path, lead: usize, bodies: &[Vec<u8>]) -> Result<()> {
    let mut f = File::create(path)?;
    f.write_all(&vec![0xEEu8; lead])?;
    for body in bodies {
        f.write_all(&(body.len() as u32).to_le_bytes())?;
        f.write_all(body)?;
    }
    f.sync_all()?;
    Ok(())
}

// The buffered frame reader must return exactly the frames a per-entry
// `read_exact_at` loop returns, including frames that straddle a refill boundary
// and a frame larger than the maximum read size — starting from a mid-file
// offset (the sparse-index seek), and end cleanly at EOF.
#[test]
fn test_frame_reader_returns_every_frame_across_refills() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let path = temp_dir.path().join("frames.dat");
    let mut bodies: Vec<Vec<u8>> = (0..5000u32).map(|i| vec![(i % 251) as u8; 1 + (i as usize * 7) % 97]).collect();
    bodies.insert(2500, vec![0x5Au8; FrameReader::MAX_READ + 12_345]); // bigger than any refill
    let lead = 16;
    write_frames(&path, lead, &bodies)?;

    let file = File::open(&path)?;
    let file_len = file.metadata()?.len();
    let mut reader = FrameReader::new(&file, lead as u64);
    let mut read = Vec::new();
    while let Some(body) = reader.next_frame(file_len)? {
        read.push(body.to_vec());
    }
    assert_eq!(read.len(), bodies.len());
    assert!(read == bodies, "frames must round-trip byte for byte");
    Ok(())
}

// EOF inside a frame is corruption, not a clean end: a truncated length prefix
// or a truncated body must surface `UnexpectedEof`, as `read_exact_at` does,
// rather than silently ending the scan early.
#[test]
fn test_frame_reader_truncated_frame_is_an_error() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let path = temp_dir.path().join("frames.dat");
    write_frames(&path, 0, &[vec![1u8; 10], vec![2u8; 10]])?;
    let full = std::fs::read(&path)?;

    // Cut 1: inside the second frame's body. Cut 2: inside its length prefix.
    for cut in [full.len() - 3, 14 + 2] {
        std::fs::write(&path, &full[..cut])?;
        let file = File::open(&path)?;
        let file_len = file.metadata()?.len();
        let mut reader = FrameReader::new(&file, 0);
        assert_eq!(reader.next_frame(file_len)?.map(<[u8]>::to_vec), Some(vec![1u8; 10]));
        match reader.next_frame(file_len) {
            Err(LSMError::Io(e)) => assert_eq!(e.kind(), std::io::ErrorKind::UnexpectedEof, "cut at {cut}"),
            other => panic!("cut at {cut}: expected UnexpectedEof, got {:?}", other.map(|o| o.map(<[u8]>::to_vec))),
        }
    }
    Ok(())
}

// The L1 publish must never leave the bucket's L1 path missing or empty while
// data lives only there: after a crash, `LSMTree::open` would create a 0-byte L1,
// read the bucket as empty, and every key that lived only in L1 would be lost.
// The publish is one atomic `rename(2)`, which replaces the destination, so this
// test watches the path through many compactions and checks no intermediate
// state has it absent or empty.
//
// Note the setup must call `cleanup_obsolete_level0_files()`: without it the merged
// L0 file is still on disk and answers the read, masking the loss entirely.
#[test]
fn test_crash_between_l1_publish_steps_keeps_l1() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let bucket = get_bucket_for_key(b"k", crate::support::TEST_NUM_BUCKETS);
    let l1 = LSMTree::level1_path_from(temp_dir.path(), bucket);
    {
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;
        lsm.insert_with_seq(b"k", 0xAAAA, 1)?;
        lsm.flush_memtable_to_level0()?;
        lsm.compact_all()?;
        lsm.cleanup_obsolete_level0_files();
        assert_eq!(lsm.get(b"k")?, Some(0xAAAA), "setup: key should be readable from L1");
    }

    // The key must now live *only* in L1, or this test cannot observe a loss.
    let l0_dir = LSMTree::level0_bucket_dir_from(temp_dir.path(), bucket);
    let l0_files = std::fs::read_dir(&l0_dir).map(|r| r.count()).unwrap_or(0);
    assert_eq!(l0_files, 0, "setup: no L0 file may remain, or it would mask the loss");
    assert!(std::fs::metadata(&l1)?.len() > SSTABLE_DATA_START, "setup: L1 must hold the entry");

    // A missing path could only be seen *during* a publish, so inspecting the
    // final state proves nothing. Watch the L1 path from another thread while
    // compactions run, and assert it is never absent: that is the state a crash
    // would freeze, and the state `LSMTree::open` turns into an empty bucket.
    let lsm = Arc::new(LSMTree::open(temp_dir.path(), test_lsm_config())?);
    let stop = Arc::new(AtomicBool::new(false));
    let vanished = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    let observer = {
        let (l1, stop, vanished) = (l1.clone(), Arc::clone(&stop), Arc::clone(&vanished));
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                // "Absent" and "present but 0 bytes" are the same failure: both make
                // `LSMTree::open` short-circuit into an empty bucket.
                match std::fs::metadata(&l1) {
                    Err(_) => {
                        vanished.fetch_add(1, Ordering::Relaxed);
                    }
                    Ok(m) if m.len() == 0 => {
                        vanished.fetch_add(1, Ordering::Relaxed);
                    }
                    Ok(_) => {}
                }
                std::hint::spin_loop();
            }
        })
    };

    // seq 1 belongs to the key seeded above, so these start at 2.
    for round in 0..200u64 {
        lsm.insert_with_seq(format!("k{round:04}").as_bytes(), 0xCCCC + round as u128, 2 + round)?;
        lsm.flush_memtable_to_level0()?;
        lsm.compact_bucket(bucket)?;
    }

    stop.store(true, Ordering::Relaxed);
    observer.join().expect("observer thread panicked");

    assert_eq!(
        vanished.load(Ordering::Relaxed),
        0,
        "the L1 path was absent/empty mid-publish — a crash in that window makes \
         LSMTree::open create a 0-byte L1 and lose every key that lived only in L1"
    );
    assert_eq!(lsm.get(b"k")?, Some(0xAAAA), "the original L1-only key must survive the merges");
    Ok(())
}

// Reads of a *registered* L0 file skip header validation (`open_registered_l0`),
// so `LSMTree::open` is the only thing standing between an L0 file in an unknown
// layout and a garbage reinterpretation of it. This pins that single guard: without
// it, the stripped file would be loaded and its entries parsed under the wrong
// layout, with nothing downstream to notice.
#[test]
fn test_open_rejects_l0_file_written_in_an_older_format() -> Result<()> {
    let temp_dir = TempDir::new()?;
    {
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;
        lsm.insert_with_seq(b"k", 0xAAAA, 1)?;
        lsm.flush_memtable_to_level0()?; // → a real, header-bearing L0 file
    }

    let bucket = get_bucket_for_key(b"k", crate::support::TEST_NUM_BUCKETS);
    let dir = LSMTree::level0_bucket_dir_from(temp_dir.path(), bucket);
    let l0 = std::fs::read_dir(&dir)?.next().expect("setup: an L0 file should exist")?.path();
    let body = std::fs::read(&l0)?;
    // Strip the header, leaving enough bytes that this is a *format* mismatch and
    // not the sub-header "incomplete debris" case, which is skipped by design.
    let stripped = &body[SSTABLE_DATA_START as usize..];
    assert!(
        stripped.len() as u64 > SSTABLE_DATA_START,
        "setup: stripped L0 body must stay longer than a header"
    );
    std::fs::write(&l0, stripped)?;

    match LSMTree::open(temp_dir.path(), test_lsm_config()) {
        Ok(_) => panic!("open must reject an unreadable L0 format, but it succeeded"),
        Err(LSMError::Corruption(m)) => assert!(m.contains("magic"), "expected a bad-magic corruption error, got: {m}"),
        Err(e) => panic!("expected a corruption error, got: {e:?}"),
    }
    Ok(())
}

/// A `.dat` at its final name is always complete (L0 files are renamed into
/// place), so one too short to hold a header is damage and fails the open.
#[test]
fn test_open_rejects_a_level0_file_shorter_than_a_header() -> Result<()> {
    let temp_dir = TempDir::new()?;
    {
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;
        lsm.insert_with_seq(b"k", 0xBBBB, 2)?;
        lsm.flush_memtable_to_level0()?;
    }

    let bucket = get_bucket_for_key(b"k", crate::support::TEST_NUM_BUCKETS);
    let dir = LSMTree::level0_bucket_dir_from(temp_dir.path(), bucket);
    let damaged = std::fs::read_dir(&dir)?.next().expect("setup: an L0 file should exist")?.path();
    std::fs::write(&damaged, b"")?;

    match LSMTree::open(temp_dir.path(), test_lsm_config()) {
        Err(LSMError::Corruption(msg)) => assert!(msg.contains("shorter than an SSTable header"), "{msg}"),
        other => panic!("expected Corruption, got {:?}", other.map(|_| ())),
    }
    Ok(())
}

/// A process killed while a flush writes an L0 file leaves it cut anywhere,
/// including inside a frame's body (a kill can land mid-`write`). The file
/// is written under a temporary name and renamed only once complete, so what
/// a kill leaves is a `.dat.tmp`: open removes it, and since its flush never
/// retired its WAL entries, replay restores its records. Written in place,
/// it would be a torn `.dat` that fails every later open
/// (`acknowledged_writes_survive_repeated_kills` hits that about once in 800
/// kills).
#[test]
fn test_open_removes_a_level0_file_a_kill_left_unfinished() -> Result<()> {
    let temp_dir = TempDir::new()?;
    {
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;
        lsm.insert_with_seq(b"k", 0xBBBB, 2)?;
        lsm.flush_memtable_to_level0()?;
    }
    let bucket = get_bucket_for_key(b"k", crate::support::TEST_NUM_BUCKETS);
    let dir = LSMTree::level0_bucket_dir_from(temp_dir.path(), bucket);
    let done = std::fs::read_dir(&dir)?.next().expect("setup: an L0 file should exist")?.path();
    assert_eq!(done.extension().unwrap(), "dat", "a finished flush leaves only `.dat` files");
    // The state a kill mid-flush leaves: the temporary, cut inside a frame body.
    let torn = dir.join(format!("{}{LEVEL0_TMP_SUFFIX}", done.file_stem().unwrap().to_string_lossy()));
    std::fs::rename(&done, &torn)?;
    let cut = SSTABLE_DATA_START + 4 + 3;
    assert!(std::fs::metadata(&torn)?.len() > cut, "setup: the frame must be longer than the cut");
    std::fs::OpenOptions::new().write(true).open(&torn)?.set_len(cut)?;

    let lsm = match LSMTree::open(temp_dir.path(), test_lsm_config()) {
        Ok(lsm) => lsm,
        Err(e) => panic!("open must remove an unfinished L0 file, not fail: {e:?}"),
    };
    assert!(!torn.exists(), "the unfinished file was not removed");
    assert_eq!(lsm.get(b"k")?, None, "its records are not in L0 (WAL replay restores them)");
    lsm.scan_prefix(b"")?;
    lsm.compact_all()?;
    Ok(())
}

#[test]
fn test_sstable_entry_codec_roundtrip_and_corruption() {
    let entry = SStableEntry {
        key: b"some_key".to_vec(),
        key_prefix: key_prefix_of(b"some_key"),
        value: 0xdead_beef,
        tombstone: false,
        seq: 42,
    };

    let mut payload = encode_sstable_entry(&entry).expect("encode");

    // Round-trip: verify passes and the body decodes to the same entry.
    let body = verify_sstable_payload(&payload).expect("verify clean payload");
    let archived = unsafe { rkyv::access_unchecked::<ArchivedSStableEntry>(body) };
    assert_eq!(archived.key.as_slice(), b"some_key");
    assert_eq!(archived.value.to_native(), 0xdead_beef);
    assert_eq!(archived.seq.to_native(), 42);

    // Flip a byte in the rkyv body (past the 4-byte CRC prefix) → mismatch.
    let last = payload.len() - 1;
    payload[last] ^= 0xFF;
    assert!(matches!(verify_sstable_payload(&payload), Err(LSMError::Corruption(_))));

    // A truncated payload (shorter than the CRC word) is also rejected.
    assert!(matches!(verify_sstable_payload(&[0u8; 2]), Err(LSMError::Corruption(_))));
}
