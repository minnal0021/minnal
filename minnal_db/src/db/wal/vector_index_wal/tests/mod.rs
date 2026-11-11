//! Tests for the vector index WAL, with the fixtures they share.

use super::*;
use crate::db::wal::segmented_log::{END_MARKER, END_MARKER_LEN, FRAME_HEADER_LEN, SEGMENT_HEADER_LEN, position, prep_path, segment_path, split};
use crate::namespace::NamespaceId;
use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::sync::atomic::Ordering::Relaxed;
use tempfile::TempDir;

fn check_header(file: &File, segment_id: u64) -> Result<()> {
    Ok(segmented_log::check_header(&FORMAT, file, segment_id)?)
}

fn header_bytes(segment_id: u64) -> [u8; SEGMENT_HEADER_LEN as usize] {
    segmented_log::header_bytes(&FORMAT, segment_id)
}

type Row = (u64, u128, u64, Vec<u8>, Option<Vec<u8>>);

fn open_at(dir: &Path, segment_size: u32) -> (VectorIndexWal, OpenReport) {
    VectorIndexWal::open(dir, segment_size, Arc::new(Metrics::default())).unwrap()
}

fn rows_from(wal: &VectorIndexWal, start: u64) -> Vec<Row> {
    let mut rows = Vec::new();
    wal.for_each_from(start, |at, r| -> Result<()> {
        rows.push((at, r.ns_id.as_u128(), r.seq, r.key.to_vec(), r.value.map(<[u8]>::to_vec)));
        Ok(())
    })
    .unwrap();
    rows
}

fn key(i: u64) -> Vec<u8> {
    format!("key-{i:04}").into_bytes()
}

fn value(i: u64) -> Vec<u8> {
    vec![i as u8; 40]
}

/// Appends records 0..n (ns `i % 3`, seq `i + 1`) and returns where each landed.
fn append_n(wal: &VectorIndexWal, n: u64) -> Vec<Appended> {
    (0..n)
        .map(|i| {
            wal.append(NamespaceId::from_u128(u128::from(i % 3)), &key(i), &value(i), i + 1)
                .unwrap()
                .appended()
        })
        .collect()
}

fn expected(appended: &[Appended], range: std::ops::Range<u64>) -> Vec<Row> {
    range
        .map(|i| (appended[i as usize].start, u128::from(i % 3), i + 1, key(i), Some(value(i))))
        .collect()
}

/// Flips one byte of a segment file.
fn flip(dir: &Path, segment_id: u64, byte: u64) {
    let f = OpenOptions::new().read(true).write(true).open(segment_path(dir, segment_id)).unwrap();
    let mut b = [0u8; 1];
    f.read_exact_at(&mut b, byte).unwrap();
    f.write_all_at(&[b[0] ^ 0xff], byte).unwrap();
}

fn set_len(dir: &Path, segment_id: u64, len: u64) {
    OpenOptions::new()
        .write(true)
        .open(segment_path(dir, segment_id))
        .unwrap()
        .set_len(len)
        .unwrap();
}

fn corruption(r: Result<(VectorIndexWal, OpenReport)>) -> String {
    match r {
        Err(VectorIndexWalError::Log(LogError::Corruption(m))) => m,
        Err(e) => panic!("expected corruption, got {e}"),
        Ok(_) => panic!("expected corruption, the open succeeded"),
    }
}

fn reopen(dir: &Path, segment_size: u32) -> Result<(VectorIndexWal, OpenReport)> {
    VectorIndexWal::open(dir, segment_size, Arc::new(Metrics::default()))
}

/// Frame length of the records `append_n` writes.
const FRAME: u32 = FRAME_HEADER_LEN + BODY_FIXED_LEN as u32 + 8 + 40;

/// A segment that holds exactly three of `append_n`'s records (and its end
/// marker), so a few appends span several segments.
const SMALL_SEGMENT: u32 = segmented_log::SEGMENT_HEADER_LEN + 3 * FRAME + segmented_log::END_MARKER_LEN;

mod database;
mod log;
mod power_loss;
mod preallocated;
mod retirement;
