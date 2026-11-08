//! The log itself: appending, reading back, torn tails, corruption and fsyncs.

use super::*;
use crate::namespace::NamespaceId;

#[test]
fn appended_records_read_back_in_order() {
    let dir = TempDir::new().unwrap();
    let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
    let appended = append_n(&wal, 50);
    assert!(appended.windows(2).all(|w| w[0].end == w[1].start && w[0].start < w[1].start));
    assert_eq!(rows_from(&wal, 0), expected(&appended, 0..50));
    assert_eq!(wal.tail(), appended[49].end);
}

#[test]
fn a_delete_reads_back_as_a_delete_and_a_torn_one_is_dropped() {
    let dir = TempDir::new().unwrap();
    let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
    let put = wal.append(NamespaceId::from_u128(1), b"k", b"v", 1).unwrap().appended();
    let del = wal.append_delete(NamespaceId::from_u128(1), b"k", 2).unwrap().appended();
    let empty = wal.append(NamespaceId::from_u128(1), b"e", b"", 3).unwrap().appended(); // an empty value is still a put
    wal.sync().unwrap();
    assert_eq!(del.end - del.start, u64::from(FRAME_HEADER_LEN) + BODY_FIXED_LEN as u64 + 1);
    let want = vec![
        (put.start, 1, 1, b"k".to_vec(), Some(b"v".to_vec())),
        (del.start, 1, 2, b"k".to_vec(), None),
        (empty.start, 1, 3, b"e".to_vec(), Some(vec![])),
    ];
    assert_eq!(rows_from(&wal, 0), want);
    drop(wal);
    let (wal, report) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
    assert_eq!(report.last_seq, 3);
    assert_eq!(rows_from(&wal, 0), want);
    drop(wal);

    // Torn inside the delete: it goes, with everything after it.
    set_len(dir.path(), 0, u64::from(split(del.end).1) - 1);
    let (wal, report) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
    assert_eq!(report.tail, del.start);
    assert_eq!(rows_from(&wal, 0), want[..1].to_vec());
}

#[test]
fn reopen_finds_the_tail_and_the_highest_seq() {
    let dir = TempDir::new().unwrap();
    let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
    wal.append(NamespaceId::from_u128(1), b"a", b"x", 10).unwrap();
    let last = wal.append(NamespaceId::from_u128(1), b"b", b"y", 5).unwrap().appended(); // seqs are not in log order
    wal.sync().unwrap();
    drop(wal);

    let (wal, report) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
    assert_eq!(
        report,
        OpenReport {
            tail: last.end,
            last_seq: 10,
            torn_bytes_dropped: 0
        }
    );
    assert_eq!(wal.last_seq(), 10);
    let more = wal.append(NamespaceId::from_u128(2), b"c", b"z", 11).unwrap();
    assert_eq!(more.start, last.end);
    let rows = rows_from(&wal, 0);
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[2], (more.start, 2, 11, b"c".to_vec(), Some(b"z".to_vec())));
}

#[test]
fn records_span_segments_and_every_sealed_segment_ends_in_a_marker() {
    let dir = TempDir::new().unwrap();
    let (wal, _) = open_at(dir.path(), SMALL_SEGMENT);
    let appended = append_n(&wal, 20);
    let segs = wal.segments().unwrap();
    assert_eq!(segs.len(), 7);
    for &(id, len) in &segs {
        assert_eq!(len, u64::from(SMALL_SEGMENT), "segment {id} is created at its full size");
    }
    for &(id, _) in &segs[..segs.len() - 1] {
        let f = File::open(segment_path(dir.path(), id)).unwrap();
        check_header(&f, id).unwrap();
        let mut m = [0u8; 4];
        f.read_exact_at(&mut m, u64::from(SEGMENT_HEADER_LEN + 3 * FRAME)).unwrap();
        assert_eq!(m, END_MARKER.to_le_bytes(), "segment {id} ends in the end marker");
    }
    assert_eq!(split(appended[3].start), (1, SEGMENT_HEADER_LEN));
    assert_eq!(rows_from(&wal, 0), expected(&appended, 0..20));
    drop(wal);
    let (wal, report) = open_at(dir.path(), SMALL_SEGMENT);
    assert_eq!(report.tail, appended[19].end);
    assert_eq!(rows_from(&wal, 0), expected(&appended, 0..20));
}

#[test]
fn reading_starts_at_a_record_or_a_segment_start() {
    let dir = TempDir::new().unwrap();
    let (wal, _) = open_at(dir.path(), SMALL_SEGMENT);
    let appended = append_n(&wal, 20);
    assert_eq!(rows_from(&wal, appended[7].start), expected(&appended, 7..20));
    assert_eq!(rows_from(&wal, position(2, 0)), expected(&appended, 6..20));
    assert_eq!(rows_from(&wal, wal.tail()), vec![]);
}

#[test]
fn a_torn_tail_is_dropped_and_later_appends_read_back() {
    for cut in [2u64, u64::from(FRAME_HEADER_LEN) + 5, u64::from(FRAME) - 1] {
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
        let appended = append_n(&wal, 5);
        drop(wal);
        let (_, last_byte) = split(appended[4].start);
        set_len(dir.path(), 0, u64::from(last_byte) + cut);

        let (wal, report) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
        assert_eq!(report.tail, appended[4].start, "cut {cut}");
        assert_eq!(report.torn_bytes_dropped, cut);
        assert_eq!(report.last_seq, 4);
        let fresh = wal.append(NamespaceId::from_u128(9), b"new", b"v", 99).unwrap().appended();
        assert_eq!(fresh.start, appended[4].start);
        let mut want = expected(&appended, 0..4);
        want.push((fresh.start, 9, 99, b"new".to_vec(), Some(b"v".to_vec())));
        assert_eq!(rows_from(&wal, 0), want);
        drop(wal);
        let (wal, report) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
        assert_eq!(report.torn_bytes_dropped, 0);
        assert_eq!(rows_from(&wal, 0), want, "nothing stale after the new record");
    }
}

#[test]
fn a_crc_failure_is_a_torn_tail_only_in_the_last_segment() {
    let dir = TempDir::new().unwrap();
    let (wal, _) = open_at(dir.path(), SMALL_SEGMENT);
    let appended = append_n(&wal, 8); // segments 0, 1, 2
    drop(wal);

    // Last record of the last segment.
    let (seg, byte) = split(appended[7].start);
    assert_eq!(seg, 2);
    flip(dir.path(), seg, u64::from(byte) + 20);
    let (wal, report) = reopen(dir.path(), SMALL_SEGMENT).unwrap();
    assert_eq!(report.tail, appended[7].start);
    assert_eq!(rows_from(&wal, 0), expected(&appended, 0..7));
    drop(wal);

    // A record of an earlier segment.
    let (seg, byte) = split(appended[1].start);
    flip(dir.path(), seg, u64::from(byte) + 20);
    let msg = corruption(reopen(dir.path(), SMALL_SEGMENT));
    assert!(msg.contains("segment 0") && msg.contains("CRC"), "{msg}");
}

#[test]
fn a_corrupt_length_is_rejected_before_allocating() {
    let dir = TempDir::new().unwrap();
    let (wal, _) = open_at(dir.path(), SMALL_SEGMENT);
    let appended = append_n(&wal, 8);
    drop(wal);
    let write_len = |at: u64| {
        let (seg, byte) = split(at);
        let f = OpenOptions::new().write(true).open(segment_path(dir.path(), seg)).unwrap();
        // Not `u32::MAX`: that is the end marker.
        f.write_all_at(&(u32::MAX - 1).to_le_bytes(), u64::from(byte)).unwrap();
    };

    write_len(appended[7].start);
    let (_, report) = reopen(dir.path(), SMALL_SEGMENT).unwrap();
    assert_eq!(report.tail, appended[7].start);
    // The frame claims more than the segment holds: capped at its end.
    assert_eq!(
        report.torn_bytes_dropped,
        u64::from(SMALL_SEGMENT) - u64::from(split(appended[7].start).1)
    );

    write_len(appended[4].start);
    let msg = corruption(reopen(dir.path(), SMALL_SEGMENT));
    assert!(msg.contains("segment 1") && msg.contains("runs past the end"), "{msg}");
}

#[test]
fn a_bad_header_or_a_missing_segment_fails_the_open() {
    let fresh = || {
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), SMALL_SEGMENT);
        append_n(&wal, 8); // segments 0, 1, 2
        dir
    };

    let dir = fresh();
    flip(dir.path(), 2, 0);
    assert!(corruption(reopen(dir.path(), SMALL_SEGMENT)).contains("no vector index WAL header"));

    let dir = fresh();
    flip(dir.path(), 0, 4);
    assert!(corruption(reopen(dir.path(), SMALL_SEGMENT)).contains("format version"));

    let dir = fresh();
    let f = OpenOptions::new().write(true).open(segment_path(dir.path(), 1)).unwrap();
    f.write_all_at(&7u64.to_le_bytes(), 8).unwrap();
    assert!(corruption(reopen(dir.path(), SMALL_SEGMENT)).contains("segment file 1 holds segment 7"));

    let dir = fresh();
    std::fs::remove_file(segment_path(dir.path(), 1)).unwrap();
    assert!(corruption(reopen(dir.path(), SMALL_SEGMENT)).contains("segment 1 is missing"));
}

#[test]
fn a_crash_mid_rotation_is_finished_at_open() {
    // Sealed, but the next segment was never created.
    let dir = TempDir::new().unwrap();
    let (wal, _) = open_at(dir.path(), SMALL_SEGMENT);
    let appended = append_n(&wal, 4); // record 3 starts segment 1
    drop(wal);
    std::fs::remove_file(segment_path(dir.path(), 1)).unwrap();
    let (wal, report) = reopen(dir.path(), SMALL_SEGMENT).unwrap();
    assert_eq!(report.tail, position(1, SEGMENT_HEADER_LEN));
    assert_eq!(rows_from(&wal, 0), expected(&appended, 0..3));
    drop(wal);

    // The next segment exists but its header never landed.
    set_len(dir.path(), 1, 5);
    let (wal, report) = reopen(dir.path(), SMALL_SEGMENT).unwrap();
    assert_eq!(report.tail, position(1, SEGMENT_HEADER_LEN));
    assert_eq!(report.torn_bytes_dropped, 5);
    let more = wal.append(NamespaceId::from_u128(0), b"k", b"v", 50).unwrap();
    assert_eq!(more.start, position(1, SEGMENT_HEADER_LEN));
    assert_eq!(rows_from(&wal, 0).len(), 4);
}

#[test]
fn a_record_too_large_for_a_segment_writes_nothing() {
    let dir = TempDir::new().unwrap();
    let (wal, _) = open_at(dir.path(), SMALL_SEGMENT);
    append_n(&wal, 2);
    let (tail, segs) = (wal.tail(), wal.segments().unwrap());
    let max = u64::from(SMALL_SEGMENT - SEGMENT_HEADER_LEN - END_MARKER_LEN);
    let fits = vec![0u8; (max - u64::from(FRAME_HEADER_LEN) - BODY_FIXED_LEN as u64) as usize];
    let too_big = vec![0u8; fits.len() + 1];
    match wal.append(NamespaceId::from_u128(0), b"", &too_big, 9) {
        Err(VectorIndexWalError::Log(LogError::RecordTooLarge { len, max: m })) => assert_eq!((len, m), (max + 1, max)),
        other => panic!("expected RecordTooLarge, got {other:?}"),
    }
    assert_eq!((wal.tail(), wal.segments().unwrap()), (tail, segs));
    // The largest record that fits is accepted, alone in its segment.
    let a = wal.append(NamespaceId::from_u128(0), b"", &fits, 10).unwrap();
    assert_eq!(split(a.start), (1, SEGMENT_HEADER_LEN));
    assert_eq!(rows_from(&wal, a.start).len(), 1);
}

#[test]
fn sync_covers_earlier_appends_and_skips_when_idle() {
    let dir = TempDir::new().unwrap();
    let metrics = Arc::new(Metrics::default());
    let (wal, report) = VectorIndexWal::open(dir.path(), SMALL_SEGMENT, Arc::clone(&metrics)).unwrap();
    assert_eq!(wal.durable_tail(), report.tail);
    wal.sync().unwrap();
    assert_eq!(metrics.vector_wal_fsyncs.load(Relaxed), 0, "nothing to sync");

    let appended = append_n(&wal, 2);
    assert!(wal.durable_tail() < wal.tail());
    wal.sync().unwrap();
    wal.sync().unwrap();
    assert_eq!(wal.durable_tail(), appended[1].end);
    assert_eq!(metrics.vector_wal_fsyncs.load(Relaxed), 1);

    // Sealing a segment makes it durable without a sync.
    let appended = append_n(&wal, 2);
    assert_eq!(wal.durable_tail(), position(0, split(appended[0].end).1));
}

#[test]
fn concurrent_appends_and_syncs_lose_nothing() {
    let dir = TempDir::new().unwrap();
    let (wal, _) = open_at(dir.path(), 4096);
    let wal = Arc::new(wal);
    let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let syncer = {
        let (wal, done) = (Arc::clone(&wal), Arc::clone(&done));
        std::thread::spawn(move || {
            while !done.load(Relaxed) {
                wal.sync().unwrap();
                assert!(wal.durable_tail() <= wal.tail());
            }
        })
    };
    let writers: Vec<_> = (0..8u64)
        .map(|t| {
            let wal = Arc::clone(&wal);
            std::thread::spawn(move || {
                for i in 0..200u64 {
                    let seq = t * 1000 + i;
                    wal.append(NamespaceId::from_u128(u128::from(t as u32)), &seq.to_le_bytes(), &[t as u8; 30], seq)
                        .unwrap();
                }
            })
        })
        .collect();
    writers.into_iter().for_each(|w| w.join().unwrap());
    done.store(true, Relaxed);
    syncer.join().unwrap();
    wal.sync().unwrap();
    assert_eq!(wal.durable_tail(), wal.tail());

    let mut seqs: Vec<u64> = rows_from(&wal, 0)
        .into_iter()
        .map(|(_, ns, seq, k, v)| {
            assert_eq!(k, seq.to_le_bytes());
            assert_eq!(v, Some(vec![ns as u8; 30]));
            seq
        })
        .collect();
    seqs.sort_unstable();
    let want: Vec<u64> = (0..8u64).flat_map(|t| (0..200).map(move |i| t * 1000 + i)).collect();
    assert_eq!(seqs, want);
}

#[test]
fn counters_count_appends_bytes_and_fsyncs() {
    let dir = TempDir::new().unwrap();
    let metrics = Arc::new(Metrics::default());
    let (wal, _) = VectorIndexWal::open(dir.path(), DEFAULT_SEGMENT_SIZE, Arc::clone(&metrics)).unwrap();
    append_n(&wal, 7);
    wal.sync().unwrap();
    append_n(&wal, 3);
    wal.sync().unwrap();
    let s = metrics.snapshot();
    assert_eq!(s.vector_wal_appends, 10);
    assert_eq!(s.vector_wal_bytes_appended, 10 * u64::from(FRAME));
    assert_eq!(s.vector_wal_fsyncs, 2);
}

#[test]
fn an_empty_log_opens_and_reopens_the_same() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("vector_index_wal");
    let (wal, report) = open_at(&path, DEFAULT_SEGMENT_SIZE);
    let empty = OpenReport {
        tail: position(0, SEGMENT_HEADER_LEN),
        last_seq: 0,
        torn_bytes_dropped: 0,
    };
    assert_eq!(report, empty);
    assert_eq!(wal.segments().unwrap(), vec![(0, u64::from(DEFAULT_SEGMENT_SIZE))]);
    assert_eq!(rows_from(&wal, 0), vec![]);
    drop(wal);
    std::fs::write(path.join("notes.txt"), b"not a segment").unwrap();
    let (wal, report) = open_at(&path, DEFAULT_SEGMENT_SIZE);
    assert_eq!(report, empty);
    assert_eq!(wal.segments().unwrap().len(), 1);
}
