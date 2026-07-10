//! Utilities for encoding and decoding composite KV store keys.
//!
//! # Key format
//!
//! ```text
//! [ cluster_id: 4 bytes, big-endian u32 ][ document_id: variable bytes ]
//! ```
//!
//! The cluster prefix is always exactly 4 bytes, which lets the store use it
//! as a range-scan prefix.  The document-id suffix is opaque — callers choose
//! an encoding that matches their document-id type (`u64`, `u128`, raw bytes,
//! etc.) and use the corresponding helpers to round-trip the value.

/// Encode a composite key from a cluster ID and arbitrary document-id bytes.
pub fn encode(cluster_id: u32, doc_id_bytes: &[u8]) -> Vec<u8> {
    let mut key = Vec::with_capacity(4 + doc_id_bytes.len());
    key.extend_from_slice(&cluster_id.to_be_bytes());
    key.extend_from_slice(doc_id_bytes);
    key
}

/// Encode a composite key from a cluster ID and a `u64` document ID.
pub fn encode_u64(cluster_id: u32, doc_id: u64) -> Vec<u8> {
    encode(cluster_id, &doc_id.to_be_bytes())
}

/// Encode a composite key from a cluster ID and a `u128` document ID.
pub fn encode_u128(cluster_id: u32, doc_id: u128) -> Vec<u8> {
    encode(cluster_id, &doc_id.to_be_bytes())
}

/// Encode a composite key from a cluster ID and a UUID represented as its
/// raw 16-byte array (e.g. `uuid::Uuid::as_bytes()`).
///
/// This avoids a `uuid` crate dependency while remaining compatible with any
/// UUID library — all of them expose a `&[u8; 16]` byte view.
pub fn encode_uuid_bytes(cluster_id: u32, uuid_bytes: &[u8; 16]) -> Vec<u8> {
    encode(cluster_id, uuid_bytes)
}

/// Decode a composite key into `(cluster_id, doc_id_bytes)`.
///
/// Returns `None` if `key` is shorter than the 4-byte cluster prefix.
pub fn decode(key: &[u8]) -> Option<(u32, &[u8])> {
    if key.len() < 4 {
        return None;
    }
    let cluster_id = u32::from_be_bytes(key[..4].try_into().unwrap());
    Some((cluster_id, &key[4..]))
}

/// Interpret document-id bytes as a `u64`.
///
/// Returns `None` if `bytes` is not exactly 8 bytes.
pub fn doc_id_as_u64(bytes: &[u8]) -> Option<u64> {
    bytes.try_into().ok().map(u64::from_be_bytes)
}

/// Interpret document-id bytes as a `u128`.
///
/// Returns `None` if `bytes` is not exactly 16 bytes.
pub fn doc_id_as_u128(bytes: &[u8]) -> Option<u128> {
    bytes.try_into().ok().map(u128::from_be_bytes)
}

/// Interpret document-id bytes as a UUID raw byte array (`[u8; 16]`).
///
/// Returns `None` if `bytes` is not exactly 16 bytes.  The returned array can
/// be passed directly to any UUID library, e.g. `uuid::Uuid::from_bytes(...)`.
pub fn doc_id_as_uuid_bytes(bytes: &[u8]) -> Option<[u8; 16]> {
    bytes.try_into().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_u64_roundtrip() {
        let key = encode_u64(42, 1234567890);
        let (cluster_id, doc_bytes) = decode(&key).unwrap();
        assert_eq!(cluster_id, 42);
        assert_eq!(doc_id_as_u64(doc_bytes).unwrap(), 1234567890);
    }

    #[test]
    fn test_encode_u128_roundtrip() {
        let key = encode_u128(7, u128::MAX - 1);
        let (cluster_id, doc_bytes) = decode(&key).unwrap();
        assert_eq!(cluster_id, 7);
        assert_eq!(doc_id_as_u128(doc_bytes).unwrap(), u128::MAX - 1);
    }

    #[test]
    fn test_encode_raw_bytes_roundtrip() {
        let raw: &[u8] = &[0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE];
        let key = encode(99, raw);
        let (cluster_id, doc_bytes) = decode(&key).unwrap();
        assert_eq!(cluster_id, 99);
        assert_eq!(doc_bytes, raw);
    }

    #[test]
    fn test_cluster_id_zero() {
        let key = encode_u64(0, 1);
        let (cluster_id, _) = decode(&key).unwrap();
        assert_eq!(cluster_id, 0);
    }

    #[test]
    fn test_cluster_id_max() {
        let key = encode_u64(u32::MAX, 0);
        let (cluster_id, doc_bytes) = decode(&key).unwrap();
        assert_eq!(cluster_id, u32::MAX);
        assert_eq!(doc_id_as_u64(doc_bytes).unwrap(), 0);
    }

    #[test]
    fn test_decode_too_short_returns_none() {
        assert!(decode(&[0x00, 0x01, 0x02]).is_none());
    }

    #[test]
    fn test_decode_exactly_4_bytes_gives_empty_doc_id() {
        let key = 5u32.to_be_bytes();
        let (cluster_id, doc_bytes) = decode(&key).unwrap();
        assert_eq!(cluster_id, 5);
        assert!(doc_bytes.is_empty());
    }

    #[test]
    fn test_doc_id_as_u64_wrong_length_returns_none() {
        assert!(doc_id_as_u64(&[0x01, 0x02, 0x03]).is_none());
    }

    #[test]
    fn test_doc_id_as_u128_wrong_length_returns_none() {
        assert!(doc_id_as_u128(&[0x01; 8]).is_none());
    }

    #[test]
    fn test_encode_uuid_bytes_roundtrip() {
        let uuid: [u8; 16] = [
            0x55, 0x0e, 0x84, 0x00, 0xe2, 0x9b, 0x41, 0xd4, 0xa7, 0x16, 0x44, 0x66, 0x55, 0x44, 0x00, 0x00,
        ];
        let key = encode_uuid_bytes(12, &uuid);
        let (cluster_id, doc_bytes) = decode(&key).unwrap();
        assert_eq!(cluster_id, 12);
        assert_eq!(doc_id_as_uuid_bytes(doc_bytes).unwrap(), uuid);
    }

    #[test]
    fn test_doc_id_as_uuid_bytes_wrong_length_returns_none() {
        assert!(doc_id_as_uuid_bytes(&[0x01; 8]).is_none());
        assert!(doc_id_as_uuid_bytes(&[0x01; 17]).is_none());
    }

    #[test]
    fn test_key_is_prefix_sortable() {
        // Keys with the same cluster_id should share the same 4-byte prefix.
        let key1 = encode_u64(3, 100);
        let key2 = encode_u64(3, 200);
        assert_eq!(&key1[..4], &key2[..4]);
    }
}
