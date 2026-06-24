use rkyv::api::high::HighDeserializer;
use rkyv::ser::allocator::ArenaHandle;
use rkyv::util::AlignedVec;
use rkyv::{Archive, Deserialize, Serialize, rancor};

use crate::bitmap::RoaringBitmap;
use crate::container::Container;

/// Errors that can occur during bitmap serialization and deserialization.
#[derive(Debug, thiserror::Error)]
#[error("bitmap storage error: {0}")]
pub struct StorageError(String);

impl From<rancor::Error> for StorageError {
    fn from(e: rancor::Error) -> Self {
        Self(e.to_string())
    }
}

/// Serialize a [`RoaringBitmap`] to bytes.
///
/// Format:
/// ```text
/// [4B LE u32  container_count]
/// for each (key, container) in sorted key order:
///   [16B LE u128  key]
///   [4B  LE u32   blob_len]
///   [blob_len bytes  rkyv-serialized Container]
/// ```
pub fn serialize(bitmap: &RoaringBitmap) -> Result<Vec<u8>, StorageError>
where
    Container: for<'a> Serialize<rkyv::api::high::HighSerializer<AlignedVec, ArenaHandle<'a>, rancor::Error>>,
{
    let entries = bitmap.store.sorted_entries();
    let mut out = Vec::new();
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for (key, container) in &entries {
        out.extend_from_slice(&key.to_le_bytes());
        let blob: AlignedVec = rkyv::to_bytes::<rancor::Error>(container).map_err(StorageError::from)?;
        out.extend_from_slice(&(blob.len() as u32).to_le_bytes());
        out.extend_from_slice(&blob);
    }
    Ok(out)
}

/// Deserialize a [`RoaringBitmap`] from bytes written by [`serialize`].
pub fn deserialize(bytes: &[u8]) -> Result<RoaringBitmap, StorageError>
where
    <Container as Archive>::Archived: Deserialize<Container, HighDeserializer<rancor::Error>>,
{
    let mut bm = RoaringBitmap::new();
    let mut pos = 0usize;

    if pos + 4 > bytes.len() {
        return Err(StorageError("truncated: missing container count".into()));
    }
    let count = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;

    for _ in 0..count {
        if pos + 16 > bytes.len() {
            return Err(StorageError("truncated: missing key".into()));
        }
        let key = u128::from_le_bytes(bytes[pos..pos + 16].try_into().unwrap());
        pos += 16;

        if pos + 4 > bytes.len() {
            return Err(StorageError("truncated: missing blob length".into()));
        }
        let blob_len = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        if pos + blob_len > bytes.len() {
            return Err(StorageError("truncated: blob data".into()));
        }
        let blob = &bytes[pos..pos + blob_len];
        pos += blob_len;

        // SAFETY: `blob` was written by our own `serialize` using the same rkyv
        // version and feature set. The `unaligned` feature makes all archived
        // primitives 1-byte aligned, so no AlignedVec copy is needed.
        let archived = unsafe { rkyv::access_unchecked::<rkyv::Archived<Container>>(blob) };
        let container: Container = rkyv::deserialize::<Container, rancor::Error>(archived).map_err(StorageError::from)?;

        bm.store.upsert(key, &container);
    }
    Ok(bm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let mut bm = RoaringBitmap::new();
        for i in 0..500u128 {
            bm.insert(i);
        }
        bm.insert(u128::MAX);
        bm.insert(0xDEAD_BEEF_0000_0042);

        let bytes = serialize(&bm).expect("serialize failed");
        let restored = deserialize(&bytes).expect("deserialize failed");

        assert_eq!(bm.cardinality(), restored.cardinality());
        let orig: Vec<u128> = bm.iter().collect();
        let rest: Vec<u128> = restored.iter().collect();
        assert_eq!(orig, rest);
    }

    #[test]
    fn round_trip_empty() {
        let bm = RoaringBitmap::new();
        let bytes = serialize(&bm).expect("serialize failed");
        let restored = deserialize(&bytes).expect("deserialize failed");
        assert!(restored.is_empty());
    }

    #[test]
    fn serialize_produces_sorted_keys() {
        let mut bm = RoaringBitmap::new();
        bm.insert(0x2_0000);
        bm.insert(0x1_0000);
        bm.insert(0x3_0000);
        let bytes = serialize(&bm).expect("serialize");
        let restored = deserialize(&bytes).expect("deserialize");
        let vals: Vec<u128> = restored.iter().collect();
        assert!(vals.windows(2).all(|w| w[0] < w[1]));
    }

    // ── Truncation handling ──────────────────────────────────────────────
    //
    // These exercise the length-framing checks that run *before* the unsafe
    // `access_unchecked`, so they are safe to feed crafted byte buffers.

    #[test]
    fn deserialize_empty_input_errors() {
        let err = deserialize(&[]).unwrap_err();
        assert!(err.to_string().contains("container count"), "got: {err}");
        // Fewer than 4 bytes is still a missing count.
        assert!(deserialize(&[0, 0, 0]).is_err());
    }

    #[test]
    fn deserialize_truncated_after_count_errors() {
        // Claims one container, but no key bytes follow.
        let bytes = 1u32.to_le_bytes();
        let err = deserialize(&bytes).unwrap_err();
        assert!(err.to_string().contains("missing key"), "got: {err}");
    }

    #[test]
    fn deserialize_truncated_blob_length_errors() {
        // count = 1, a full 16-byte key, then nothing where the blob length goes.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&0u128.to_le_bytes());
        let err = deserialize(&bytes).unwrap_err();
        assert!(err.to_string().contains("blob length"), "got: {err}");
    }

    #[test]
    fn deserialize_truncated_blob_data_errors() {
        // count = 1, key, blob_len = 10, but only 2 blob bytes are present.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&0u128.to_le_bytes());
        bytes.extend_from_slice(&10u32.to_le_bytes());
        bytes.extend_from_slice(&[0xAB, 0xCD]);
        let err = deserialize(&bytes).unwrap_err();
        assert!(err.to_string().contains("blob data"), "got: {err}");
    }
}
