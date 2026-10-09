//! On-disk encoding of a field index's bitmaps.
//!
//! A bitmap is stored **one container at a time**, so a change rewrites only
//! the containers it touched. In a field's bitmap store (`blobs.vals`):
//!
//! - each [`Container`] is its own blob (rkyv bytes, see [`encode_container`]);
//! - each bitmap is a **directory** blob listing its containers, sorted by
//!   container key, with the offset and length of each container blob;
//! - the slot for a value points at its directory.
//!
//! Directory layout:
//!
//! ```text
//! [4B  magic "MBD1"]
//! [4B  LE u32 count]
//! count × [16B LE u128 container key][8B LE u64 offset][4B LE u32 len]
//! [4B  LE u32 crc32 of every byte before it]
//! ```
//!
//! The checksum lets open-time repair tell a directory torn by a crash from a
//! good one; container blobs are validated by rkyv's checked access when read.

use rkyv::api::high::{HighDeserializer, HighValidator};
use rkyv::ser::allocator::ArenaHandle;
use rkyv::util::AlignedVec;
use rkyv::{Archive, Deserialize, Serialize, rancor};

use crate::index::container::Container;

/// Errors that can occur during bitmap serialization and deserialization.
#[derive(Debug, thiserror::Error)]
#[error("bitmap storage error: {0}")]
pub struct StorageError(String);

impl From<rancor::Error> for StorageError {
    fn from(e: rancor::Error) -> Self {
        Self(e.to_string())
    }
}

const DIR_MAGIC: [u8; 4] = *b"MBD1";
const DIR_HEADER: usize = 8;
const DIR_ENTRY: usize = 28;
const DIR_CRC: usize = 4;

/// One directory entry: where a container's blob lives in the value region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirEntry {
    /// Container key (the row id's upper 112 bits).
    pub key: u128,
    /// Byte offset of the container blob in the value region.
    pub offset: u64,
    /// Length of the container blob.
    pub len: u32,
}

/// Encode one container as a blob.
pub fn encode_container(container: &Container) -> Result<AlignedVec, StorageError>
where
    Container: for<'a> Serialize<rkyv::api::high::HighSerializer<AlignedVec, ArenaHandle<'a>, rancor::Error>>,
{
    rkyv::to_bytes::<rancor::Error>(container).map_err(StorageError::from)
}

/// Decode one container blob.
///
/// Checked access: validate the archive's structure before reading it, so a
/// corrupt blob yields a `StorageError` instead of UB. The `unaligned` rkyv
/// feature makes archived primitives 1-byte aligned, so the raw slice is
/// sufficiently aligned and no AlignedVec copy is needed.
pub fn decode_container(blob: &[u8]) -> Result<Container, StorageError>
where
    <Container as Archive>::Archived:
        Deserialize<Container, HighDeserializer<rancor::Error>> + for<'a> rkyv::bytecheck::CheckBytes<HighValidator<'a, rancor::Error>>,
{
    let archived = rkyv::access::<rkyv::Archived<Container>, rancor::Error>(blob).map_err(StorageError::from)?;
    rkyv::deserialize::<Container, rancor::Error>(archived).map_err(StorageError::from)
}

/// Encode a directory. `entries` must be sorted by key.
pub fn encode_dir(entries: &[DirEntry]) -> Vec<u8> {
    debug_assert!(entries.windows(2).all(|w| w[0].key < w[1].key), "directory entries must be sorted");
    let mut out = Vec::with_capacity(DIR_HEADER + entries.len() * DIR_ENTRY + DIR_CRC);
    out.extend_from_slice(&DIR_MAGIC);
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for e in entries {
        out.extend_from_slice(&e.key.to_le_bytes());
        out.extend_from_slice(&e.offset.to_le_bytes());
        out.extend_from_slice(&e.len.to_le_bytes());
    }
    let crc = crc32fast::hash(&out);
    out.extend_from_slice(&crc.to_le_bytes());
    out
}

/// Decode a directory, checking its magic, length and checksum.
pub fn decode_dir(bytes: &[u8]) -> Result<Vec<DirEntry>, StorageError> {
    if bytes.len() < DIR_HEADER + DIR_CRC || bytes[0..4] != DIR_MAGIC {
        return Err(StorageError("not a bitmap directory".into()));
    }
    let count = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let body = count
        .checked_mul(DIR_ENTRY)
        .and_then(|n| n.checked_add(DIR_HEADER))
        .ok_or_else(|| StorageError("directory count overflows".into()))?;
    if bytes.len() != body + DIR_CRC {
        return Err(StorageError(format!("directory length {} does not match count {count}", bytes.len())));
    }
    let stored = u32::from_le_bytes(bytes[body..body + DIR_CRC].try_into().unwrap());
    if stored != crc32fast::hash(&bytes[..body]) {
        return Err(StorageError("directory checksum mismatch".into()));
    }
    let entries: Vec<DirEntry> = bytes[DIR_HEADER..body]
        .chunks_exact(DIR_ENTRY)
        .map(|c| DirEntry {
            key: u128::from_le_bytes(c[0..16].try_into().unwrap()),
            offset: u64::from_le_bytes(c[16..24].try_into().unwrap()),
            len: u32::from_le_bytes(c[24..28].try_into().unwrap()),
        })
        .collect();
    if !entries.windows(2).all(|w| w[0].key < w[1].key) {
        return Err(StorageError("directory entries out of order".into()));
    }
    Ok(entries)
}

/// The entry for container `key` in a sorted directory.
pub fn find_entry(entries: &[DirEntry], key: u128) -> Option<DirEntry> {
    entries.binary_search_by_key(&key, |e| e.key).ok().map(|i| entries[i])
}

/// Number of entries in an encoded directory. Checks the magic and that the
/// length matches the count, not the checksum (see [`lookup_dir`]).
pub fn dir_count(bytes: &[u8]) -> Result<usize, StorageError> {
    if bytes.len() < DIR_HEADER + DIR_CRC || bytes[0..4] != DIR_MAGIC {
        return Err(StorageError("not a bitmap directory".into()));
    }
    let count = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    match count.checked_mul(DIR_ENTRY).and_then(|n| n.checked_add(DIR_HEADER + DIR_CRC)) {
        Some(n) if n == bytes.len() => Ok(count),
        _ => Err(StorageError(format!("directory length {} does not match count {count}", bytes.len()))),
    }
}

/// The entry for container `key` in an encoded directory, by binary search
/// over its bytes: O(log n), no allocation.
///
/// Checks the magic and length but not the checksum, which costs a pass over
/// the whole directory. That is safe because a directory is never changed
/// once written: one torn by a crash is caught by the store's repair at open,
/// and [`decode_dir`] checks the checksum on every full read.
pub fn lookup_dir(bytes: &[u8], key: u128) -> Result<Option<DirEntry>, StorageError> {
    let count = dir_count(bytes)?;
    let entry = |i: usize| {
        let c = &bytes[DIR_HEADER + i * DIR_ENTRY..DIR_HEADER + (i + 1) * DIR_ENTRY];
        DirEntry {
            key: u128::from_le_bytes(c[0..16].try_into().unwrap()),
            offset: u64::from_le_bytes(c[16..24].try_into().unwrap()),
            len: u32::from_le_bytes(c[24..28].try_into().unwrap()),
        }
    };
    let (mut lo, mut hi) = (0usize, count);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let e = entry(mid);
        match e.key.cmp(&key) {
            std::cmp::Ordering::Equal => return Ok(Some(e)),
            std::cmp::Ordering::Less => lo = mid + 1,
            std::cmp::Ordering::Greater => hi = mid,
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn container_of(values: impl IntoIterator<Item = u16>) -> Container {
        let mut c = Container::new_array();
        for v in values {
            c.insert(v);
        }
        c
    }

    #[test]
    fn container_round_trip() {
        for c in [container_of([1, 2, 3]), container_of(0..10_000), container_of([65_535])] {
            let bytes = encode_container(&c).unwrap();
            assert_eq!(decode_container(&bytes).unwrap(), c);
        }
    }

    #[test]
    fn dir_round_trip() {
        let entries = vec![
            DirEntry { key: 0, offset: 16, len: 40 },
            DirEntry {
                key: 7,
                offset: 64,
                len: 8_200,
            },
            DirEntry {
                key: u128::MAX >> 16,
                offset: 9_000,
                len: 12,
            },
        ];
        let bytes = encode_dir(&entries);
        assert_eq!(decode_dir(&bytes).unwrap(), entries);
        assert_eq!(find_entry(&entries, 7).unwrap().offset, 64);
        assert!(find_entry(&entries, 8).is_none());
        assert_eq!(dir_count(&bytes).unwrap(), 3);
        for e in &entries {
            assert_eq!(lookup_dir(&bytes, e.key).unwrap(), Some(*e));
        }
        for missing in [1, 6, 8, u128::MAX] {
            assert_eq!(lookup_dir(&bytes, missing).unwrap(), None);
        }
        assert_eq!(lookup_dir(&encode_dir(&[]), 0).unwrap(), None);
        assert!(lookup_dir(&bytes[..bytes.len() - 1], 7).is_err());
        assert!(decode_dir(&encode_dir(&[])).unwrap().is_empty());
    }

    #[test]
    fn dir_rejects_torn_or_foreign_bytes() {
        let bytes = encode_dir(&[DirEntry { key: 3, offset: 0, len: 10 }]);
        for i in 0..bytes.len() {
            let mut torn = bytes.clone();
            torn[i] ^= 0x40;
            assert!(decode_dir(&torn).is_err(), "flipping byte {i} must be detected");
        }
        assert!(decode_dir(&bytes[..bytes.len() - 1]).is_err());
        assert!(decode_dir(&[]).is_err());
        assert!(decode_dir(&[0u8; 64]).is_err());
    }

    /// A container blob whose rkyv payload is corrupted must be rejected, not
    /// deserialized into garbage or UB.
    #[test]
    fn corrupt_container_payload_errors() {
        let mut bytes = encode_container(&container_of(0..300)).unwrap().to_vec();
        for b in &mut bytes {
            *b ^= 0xFF;
        }
        assert!(decode_container(&bytes).is_err());
        assert!(decode_container(&[0xFFu8; 32]).is_err());
    }
}
