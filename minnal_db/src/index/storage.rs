use rkyv::api::high::{HighDeserializer, HighValidator};
use rkyv::ser::allocator::ArenaHandle;
use rkyv::util::AlignedVec;
use rkyv::{Archive, Deserialize, Serialize, rancor};

use crate::index::bitmap::RoaringBitmap;
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

/// Walk the frames of a serialized bitmap, calling `f(key, rkyv_bytes)` for
/// each container in order. Stops early when `f` returns `Ok(false)`. Checks
/// every length against the buffer, so truncated input is a [`StorageError`].
fn for_each_frame(bytes: &[u8], mut f: impl FnMut(u128, &[u8]) -> Result<bool, StorageError>) -> Result<(), StorageError> {
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

        if !f(key, blob)? {
            break;
        }
    }
    Ok(())
}

/// Decode one container's rkyv bytes.
///
/// Checked access: validate the archive's structure before reading it, so a
/// corrupt blob yields a `StorageError` instead of UB. The `unaligned` rkyv
/// feature makes archived primitives 1-byte aligned, so the raw slice is
/// sufficiently aligned and no AlignedVec copy is needed.
fn decode_container(blob: &[u8]) -> Result<Container, StorageError>
where
    <Container as Archive>::Archived:
        Deserialize<Container, HighDeserializer<rancor::Error>> + for<'a> rkyv::bytecheck::CheckBytes<HighValidator<'a, rancor::Error>>,
{
    let archived = rkyv::access::<rkyv::Archived<Container>, rancor::Error>(blob).map_err(StorageError::from)?;
    rkyv::deserialize::<Container, rancor::Error>(archived).map_err(StorageError::from)
}

/// Deserialize a [`RoaringBitmap`] from bytes written by [`serialize`].
///
/// Each container blob is accessed with **checked** rkyv validation
/// ([`rkyv::access`]) so a corrupt or malicious on-disk blob is reported as a
/// [`StorageError`] rather than triggering a panic or undefined behaviour — see
/// `FieldIndex::load_bitmap`, which treats this failure as recoverable.
pub fn deserialize(bytes: &[u8]) -> Result<RoaringBitmap, StorageError> {
    let mut bm = RoaringBitmap::new();
    for_each_frame(bytes, |key, blob| {
        bm.store.upsert(key, &decode_container(blob)?);
        Ok(true)
    })?;
    Ok(bm)
}

/// True if a bitmap serialized by [`serialize`] contains any of `values`.
///
/// Decodes only the containers those values fall in, not the whole bitmap, so
/// it is cheap to ask of every value bucket of a field (clearing a row whose
/// old value is unknown).
pub fn contains_any(bytes: &[u8], values: &[u128]) -> Result<bool, StorageError> {
    let mut found = false;
    for_each_frame(bytes, |key, blob| {
        let wanted: Vec<u16> = values
            .iter()
            .filter(|&&v| crate::index::bitmap::decompose(v).0 == key)
            .map(|&v| crate::index::bitmap::decompose(v).1)
            .collect();
        if wanted.is_empty() {
            return Ok(true);
        }
        let c = decode_container(blob)?;
        found = wanted.iter().any(|&low| c.contains(low));
        Ok(!found)
    })?;
    Ok(found)
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
    fn contains_any_matches_a_full_decode() {
        let mut bm = RoaringBitmap::new();
        for v in [3u128, 70_000, 70_001, 5 << 40] {
            bm.insert(v);
        }
        let bytes = serialize(&bm).unwrap();
        for probe in [&[3u128][..], &[70_001], &[4, 5 << 40], &[4], &[200_000, 1], &[]] {
            let expected = probe.iter().any(|&v| bm.contains(v));
            assert_eq!(contains_any(&bytes, probe).unwrap(), expected, "probe {probe:?}");
        }
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
    // These exercise the length-framing checks that run *before* the rkyv
    // access, so they fail fast on the framing without reaching deserialization.

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

    // ── Corrupt rkyv payload with *valid* framing ─────────────────────────
    //
    // These pass the length-framing checks and reach the rkyv access. With the
    // previous `access_unchecked` they were undefined behaviour / a likely
    // panic; with checked `rkyv::access` they must return a `StorageError`.

    /// A real blob whose rkyv payload bytes are corrupted in place (framing left
    /// intact) must be rejected, not deserialized into garbage or UB.
    #[test]
    fn deserialize_corrupt_rkyv_payload_with_valid_framing_errors() {
        let mut bm = RoaringBitmap::new();
        for i in 0..300u128 {
            bm.insert(i); // a non-trivial container with an internal Vec/pointer
        }
        let mut bytes = serialize(&bm).expect("serialize");

        // Layout: [4B count][16B key][4B blob_len][blob …] — corrupt the blob.
        let blob_start = 4 + 16 + 4;
        assert!(bytes.len() > blob_start, "expected a non-empty blob");
        for b in &mut bytes[blob_start..] {
            *b ^= 0xFF; // flip every payload byte → relative pointer/len now out of bounds
        }

        let err = deserialize(&bytes).expect_err("corrupt rkyv payload must error, not panic");
        assert!(err.to_string().contains("bitmap storage error"), "got: {err}");
    }

    /// Valid framing wrapping an all-`0xFF` garbage blob (definitely not a valid
    /// archive) must also be rejected gracefully.
    #[test]
    fn deserialize_garbage_blob_with_valid_framing_errors() {
        let garbage = [0xFFu8; 32];
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1u32.to_le_bytes()); // count = 1
        bytes.extend_from_slice(&0u128.to_le_bytes()); // key
        bytes.extend_from_slice(&(garbage.len() as u32).to_le_bytes()); // blob_len matches
        bytes.extend_from_slice(&garbage);

        // Must not panic; returns a controlled error.
        assert!(deserialize(&bytes).is_err());
    }
}
