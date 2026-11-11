//! Preallocated segments (design note §18).

use super::*;
use crate::namespace::NamespaceId;

/// The bytes of segment `segment_id` from `from` to `to`.
fn bytes_of(dir: &Path, segment_id: u64, from: u64, to: u64) -> Vec<u8> {
    let f = File::open(segment_path(dir, segment_id)).unwrap();
    let mut b = vec![0u8; (to - from) as usize];
    f.read_exact_at(&mut b, from).unwrap();
    b
}

fn zero(dir: &Path, segment_id: u64, from: u64, to: u64) {
    let f = OpenOptions::new().write(true).open(segment_path(dir, segment_id)).unwrap();
    f.write_all_at(&vec![0u8; (to - from) as usize], from).unwrap();
}

/// The last segment's records end at unwritten space; a reopen finds the
/// same tail and the file keeps its full size.
#[test]
fn a_preallocated_segment_reopens_at_the_same_tail() {
    let dir = TempDir::new().unwrap();
    let (wal, _) = open_at(dir.path(), 4096);
    let appended = append_n(&wal, 5);
    wal.sync().unwrap();
    let tail = wal.tail();
    drop(wal);
    let (wal, report) = reopen(dir.path(), 4096).unwrap();
    assert_eq!((report.tail, report.torn_bytes_dropped), (tail, 0));
    assert_eq!(rows_from(&wal, 0), expected(&appended, 0..5));
    assert_eq!(wal.segments().unwrap(), vec![(0, 4096)]);
}

/// An earlier segment whose end marker is gone reads as unwritten space
/// before its end: it was never sealed, which only the last segment may be.
#[test]
fn unwritten_space_in_an_earlier_segment_is_corruption() {
    let dir = TempDir::new().unwrap();
    let (wal, _) = open_at(dir.path(), SMALL_SEGMENT);
    append_n(&wal, 4); // segment 0 sealed with 3
    drop(wal);
    let marker = u64::from(SEGMENT_HEADER_LEN + 3 * FRAME);
    zero(dir.path(), 0, marker, marker + 4);
    let msg = corruption(reopen(dir.path(), SMALL_SEGMENT));
    assert!(msg.contains("segment 0 is not sealed"), "{msg}");
}

/// A last segment whose header is all zeros was being created when the
/// power went (its header is fsynced before any record goes in): it is
/// created again. Other bad headers stay corruption.
#[test]
fn a_last_segment_with_a_blank_header_is_created_again() {
    let dir = TempDir::new().unwrap();
    let (wal, _) = open_at(dir.path(), SMALL_SEGMENT);
    let appended = append_n(&wal, 3);
    wal.append(NamespaceId::from_u64(0), b"k", b"v", 99).unwrap(); // starts segment 1
    drop(wal);
    zero(dir.path(), 1, 0, u64::from(SMALL_SEGMENT));
    let (wal, report) = reopen(dir.path(), SMALL_SEGMENT).unwrap();
    assert_eq!(report.tail, position(1, SEGMENT_HEADER_LEN));
    assert_eq!(rows_from(&wal, 0), expected(&appended, 0..3));
    let f = File::open(segment_path(dir.path(), 1)).unwrap();
    check_header(&f, 1).unwrap();
    drop(wal);

    flip(dir.path(), 1, 2);
    let msg = corruption(reopen(dir.path(), SMALL_SEGMENT));
    assert!(msg.contains("no vector index WAL header"), "{msg}");
}

/// A segment of the earlier format (end marker 0) fails the open: no
/// conversion.
#[test]
fn a_segment_of_an_older_version_fails_the_open() {
    let dir = TempDir::new().unwrap();
    let (wal, _) = open_at(dir.path(), SMALL_SEGMENT);
    append_n(&wal, 2);
    drop(wal);
    let f = OpenOptions::new().write(true).open(segment_path(dir.path(), 0)).unwrap();
    f.write_all_at(&1u32.to_le_bytes(), 4).unwrap();
    let msg = corruption(reopen(dir.path(), SMALL_SEGMENT));
    assert!(msg.contains(&format!("format version 1, expected {}", FORMAT.version)), "{msg}");
}

/// Past half of the active segment the next one is zero-filled ahead,
/// with its header, under a `.prep` name; moving to it installs it with
/// its blocks written, and the log reads back whole.
#[test]
fn the_next_segment_is_prepared_ahead_and_installed() {
    use std::os::unix::fs::MetadataExt;
    let dir = TempDir::new().unwrap();
    let (wal, _) = open_at(dir.path(), 64 * 1024);
    let per_segment = u64::from((64 * 1024 - SEGMENT_HEADER_LEN - END_MARKER_LEN) / FRAME);
    let mut appended = append_n(&wal, per_segment / 2 + 1);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while !wal.log.prep.pending().is_some_and(|(id, done)| id == 1 && done) {
        assert!(std::time::Instant::now() < deadline, "segment 1 was never prepared");
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let prep = prep_path(dir.path(), 1);
    assert!(prep.exists() && !segment_path(dir.path(), 1).exists());
    let before = appended.len() as u64;
    appended.extend((before..per_segment + 2).map(|i| {
        wal.append(NamespaceId::from_u64(u64::from((i % 3) as u32)), &key(i), &value(i), i + 1)
            .unwrap()
            .appended()
    }));
    assert!(!prep.exists(), "the prepared file was moved into place");
    let m = std::fs::metadata(segment_path(dir.path(), 1)).unwrap();
    assert_eq!(m.len(), 64 * 1024);
    assert!(m.blocks() * 512 >= 64 * 1024, "the installed segment's blocks are written");
    wal.sync().unwrap();
    drop(wal);
    let (wal, _) = reopen(dir.path(), 64 * 1024).unwrap();
    assert_eq!(rows_from(&wal, 0), expected(&appended, 0..per_segment + 2));
}

/// A segment a previous run left half-prepared is removed at open and
/// never read as part of the log.
#[test]
fn a_leftover_prepared_segment_is_removed_at_open() {
    let dir = TempDir::new().unwrap();
    let (wal, _) = open_at(dir.path(), SMALL_SEGMENT);
    let appended = append_n(&wal, 2);
    drop(wal);
    let prep = prep_path(dir.path(), 1);
    std::fs::write(&prep, header_bytes(1)).unwrap();
    let (wal, report) = reopen(dir.path(), SMALL_SEGMENT).unwrap();
    assert!(!prep.exists());
    assert_eq!(split(report.tail).0, 0);
    assert_eq!(rows_from(&wal, 0), expected(&appended, 0..2));
}

/// A power loss kept records past one it lost. Open zeroes everything
/// past the tail, so a record later laid over the gap — ending exactly
/// where an old record starts, as same-size records do — is followed by
/// unwritten space, not by the old records.
#[test]
fn records_past_a_lost_one_never_rejoin_the_log() {
    let dir = TempDir::new().unwrap();
    let (wal, _) = open_at(dir.path(), 64 * 1024);
    let appended = append_n(&wal, 40);
    wal.sync().unwrap();
    drop(wal);
    let (lost_seg, lost) = split(appended[10].start);
    zero(dir.path(), lost_seg, u64::from(lost), u64::from(split(appended[10].end).1));

    let (wal, report) = reopen(dir.path(), 64 * 1024).unwrap();
    assert_eq!(report.tail, appended[10].start);
    let past = bytes_of(dir.path(), 0, u64::from(lost), 64 * 1024);
    assert!(past.iter().all(|&b| b == 0), "open zeroes every byte past the tail");
    let new = wal
        .append(NamespaceId::from_u64(0), &key(9_000), &value(9_000), 9_001)
        .unwrap()
        .appended();
    assert_eq!(new.end, appended[11].start, "a same-size record ends where an old one started");
    wal.sync().unwrap();
    drop(wal);

    let (wal, _) = reopen(dir.path(), 64 * 1024).unwrap();
    let mut want = expected(&appended, 0..10);
    want.push((new.start, 0, 9_001, key(9_000), Some(value(9_000))));
    assert_eq!(rows_from(&wal, 0), want);
}
