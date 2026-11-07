//! Validated UTF-8 string keys for typed stores.
//!
//! Both store kinds accept string keys — document stores via [`KeyType::Str`]
//! and KV stores via [`KvKeyType::Str`] — and both bound them to
//! [`MAX_STR_KEY_LEN`] bytes. [`StrKey`] is the single choke point: it is the
//! only way to build one, so holding a `StrKey` is proof the key passed
//! validation.
//!
//! The cap is a **typed-store policy**, not an engine limit. The raw KV engine
//! (`Db::put`) still treats keys as uninterpreted bytes of any length; the
//! bound exists so that a key is small enough to be cheap in the row map, the
//! LSM sparse index, and every SSTable entry that carries it.
//!
//! [`KeyType::Str`]: crate::doc_store::schema::KeyType::Str
//! [`KvKeyType::Str`]: crate::doc_store::schema::kv::KvKeyType::Str

use std::cmp::Ordering;
use std::fmt;

use crate::doc_store::error::{DocStoreError, SchemaError};
use crate::doc_store::schema::KeyType;

/// Maximum length of a string key, in **UTF-8 bytes**.
///
/// Counted in bytes rather than `char`s because bytes are what every layer
/// below actually stores: a 50-`char` key of 4-byte code points would cost 200
/// bytes in the row map, the WAL, and each SSTable entry.
pub const MAX_STR_KEY_LEN: usize = 50;

/// A validated UTF-8 string key of at most [`MAX_STR_KEY_LEN`] bytes.
///
/// Stored inline in a fixed-size buffer rather than a `String`, which keeps the
/// type `Copy` (and therefore [`DocId`] `Copy`, so no doc-store signature has
/// to take it by reference) and makes the length bound *structural* — an
/// over-long `StrKey` cannot be constructed.
///
/// Ordering is lexicographic over the key bytes, matching the byte ordering the
/// storage engine uses, so range scans over `StrKey` document IDs behave the
/// same way they do over big-endian integer IDs.
///
/// [`DocId`]: crate::doc_store::DocId
#[derive(Clone, Copy)]
pub struct StrKey {
    /// Number of meaningful bytes in `buf`. Always `1..=MAX_STR_KEY_LEN`.
    len: u8,
    /// Key bytes; only `buf[..len]` is meaningful, the tail is unspecified.
    buf: [u8; MAX_STR_KEY_LEN],
}

impl StrKey {
    /// Validate `s` and store it inline.
    ///
    /// Returns [`SchemaError::EmptyStrKey`] for an empty key and
    /// [`SchemaError::StrKeyTooLong`] for one over [`MAX_STR_KEY_LEN`] bytes.
    pub fn new(s: &str) -> Result<Self, SchemaError> {
        Self::from_bytes(s.as_bytes())
    }

    /// Validate raw key bytes and store them inline.
    ///
    /// Used on the read path, where the bytes come back from storage: a key
    /// written through [`StrKey::new`] always round-trips, so a failure here
    /// means the stored key was not written as a string key (wrong `key_type`)
    /// or is corrupt.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SchemaError> {
        if bytes.is_empty() {
            return Err(SchemaError::EmptyStrKey);
        }
        if bytes.len() > MAX_STR_KEY_LEN {
            return Err(SchemaError::StrKeyTooLong {
                max: MAX_STR_KEY_LEN,
                len: bytes.len(),
            });
        }
        if std::str::from_utf8(bytes).is_err() {
            return Err(SchemaError::StrKeyNotUtf8);
        }
        let mut buf = [0u8; MAX_STR_KEY_LEN];
        buf[..bytes.len()].copy_from_slice(bytes);
        Ok(Self { len: bytes.len() as u8, buf })
    }

    /// The key bytes — exactly what is stored as the database key.
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf[..self.len as usize]
    }

    /// The key as a string slice. Infallible: the bytes were UTF-8-validated at
    /// construction and the buffer is private.
    pub fn as_str(&self) -> &str {
        // SAFETY-equivalent without unsafe: validated in `from_bytes`, and the
        // buffer cannot be mutated from outside this module.
        std::str::from_utf8(self.as_bytes()).unwrap_or_default()
    }

    /// Length of the key in bytes.
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// Always `false` — an empty key cannot be constructed. Present so the type
    /// satisfies the usual `len`/`is_empty` pairing.
    pub fn is_empty(&self) -> bool {
        false
    }
}

// The derived impls would be wrong: `buf` carries unspecified bytes past `len`,
// and a derived `Ord` on the struct would compare `len` before the bytes —
// ordering "z" before "aa" and breaking the lexicographic key ordering that
// range scans depend on. Compare the meaningful bytes only.

impl PartialEq for StrKey {
    fn eq(&self, other: &Self) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}

impl Eq for StrKey {}

impl Ord for StrKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_bytes().cmp(other.as_bytes())
    }
}

impl PartialOrd for StrKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl std::hash::Hash for StrKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_bytes().hash(state);
    }
}

impl fmt::Debug for StrKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_str(), f)
    }
}

impl fmt::Display for StrKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A document identifier, typed to match the [`KeyType`] of the store.
///
/// Integer keys are stored in big-endian byte order and string keys verbatim,
/// so in both cases lexicographic byte order corresponds to the natural order
/// of the ID — which is what makes range scans over IDs work.
///
/// `DocId` is `Copy`: [`StrKey`] stores its bytes inline rather than in a
/// `String`, so passing an ID by value never allocates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DocId {
    /// 128-bit UUID represented as a `u128`.
    Uuid(u128),
    /// Unsigned 64-bit integer.
    U64(u64),
    /// Unsigned 128-bit integer.
    U128(u128),
    /// UTF-8 string key, validated to at most
    /// [`MAX_STR_KEY_LEN`] bytes.
    Str(StrKey),
}

impl DocId {
    /// Serialize this ID to its raw database key bytes — big-endian for the
    /// integer types, the UTF-8 bytes themselves for [`DocId::Str`].
    pub fn to_bytes(self) -> Vec<u8> {
        match self {
            DocId::Uuid(v) | DocId::U128(v) => v.to_be_bytes().to_vec(),
            DocId::U64(v) => v.to_be_bytes().to_vec(),
            DocId::Str(k) => k.as_bytes().to_vec(),
        }
    }

    /// The store [`KeyType`] this ID belongs to.
    pub fn key_type(self) -> KeyType {
        match self {
            DocId::Uuid(_) => KeyType::Uuid,
            DocId::U64(_) => KeyType::U64,
            DocId::U128(_) => KeyType::U128,
            DocId::Str(_) => KeyType::Str,
        }
    }

    /// Deserialize bytes back to a `DocId` given the store's [`KeyType`].
    pub fn from_bytes(bytes: &[u8], key_type: KeyType) -> Result<Self, DocStoreError> {
        match key_type {
            KeyType::Str => Ok(DocId::Str(StrKey::from_bytes(bytes)?)),
            KeyType::Uuid => {
                let arr: [u8; 16] = bytes
                    .try_into()
                    .map_err(|_| DocStoreError::InvalidId(format!("expected 16 bytes for UUID key, got {}", bytes.len())))?;
                Ok(DocId::Uuid(u128::from_be_bytes(arr)))
            }
            KeyType::U64 => {
                let arr: [u8; 8] = bytes
                    .try_into()
                    .map_err(|_| DocStoreError::InvalidId(format!("expected 8 bytes for u64 key, got {}", bytes.len())))?;
                Ok(DocId::U64(u64::from_be_bytes(arr)))
            }
            KeyType::U128 => {
                let arr: [u8; 16] = bytes
                    .try_into()
                    .map_err(|_| DocStoreError::InvalidId(format!("expected 16 bytes for u128 key, got {}", bytes.len())))?;
                Ok(DocId::U128(u128::from_be_bytes(arr)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_a_key_at_the_limit_and_rejects_one_past_it() {
        let at_limit = "x".repeat(MAX_STR_KEY_LEN);
        assert_eq!(StrKey::new(&at_limit).unwrap().as_str(), at_limit);

        let over = "x".repeat(MAX_STR_KEY_LEN + 1);
        assert!(matches!(
            StrKey::new(&over),
            Err(SchemaError::StrKeyTooLong { max, len }) if max == MAX_STR_KEY_LEN && len == MAX_STR_KEY_LEN + 1
        ));
    }

    #[test]
    fn rejects_an_empty_key() {
        assert!(matches!(StrKey::new(""), Err(SchemaError::EmptyStrKey)));
        assert!(matches!(StrKey::from_bytes(&[]), Err(SchemaError::EmptyStrKey)));
    }

    #[test]
    fn the_limit_counts_bytes_not_chars() {
        // 20 × 3-byte code points = 60 bytes: over the limit despite being
        // only 20 characters.
        let twenty_chars = "日".repeat(20);
        assert_eq!(twenty_chars.chars().count(), 20);
        assert_eq!(twenty_chars.len(), 60);
        assert!(matches!(StrKey::new(&twenty_chars), Err(SchemaError::StrKeyTooLong { .. })));

        // 16 × 3 = 48 bytes fits.
        let sixteen_chars = "日".repeat(16);
        assert_eq!(StrKey::new(&sixteen_chars).unwrap().len(), 48);
    }

    #[test]
    fn from_bytes_rejects_non_utf8() {
        assert!(matches!(StrKey::from_bytes(&[0xff, 0xfe]), Err(SchemaError::StrKeyNotUtf8)));
    }

    /// The derive trap: `#[derive(PartialEq)]` would compare the whole 50-byte
    /// buffer (equal here only by luck of zero-init) and `#[derive(Ord)]` would
    /// compare `len` first, ordering "z" before "aa".
    #[test]
    fn eq_and_ord_ignore_the_buffer_tail_and_are_lexicographic() {
        let a = StrKey::new("aa").unwrap();
        let z = StrKey::new("z").unwrap();
        assert!(a < z, "shorter key must not sort first just for being shorter");

        // A key that is a strict prefix of another sorts before it.
        let acme = StrKey::new("acme").unwrap();
        let acme_corp = StrKey::new("acme-corp").unwrap();
        assert!(acme < acme_corp);

        // Equality is over the meaningful bytes only.
        assert_eq!(StrKey::new("abc").unwrap(), StrKey::new("abc").unwrap());
        assert_ne!(acme, acme_corp);
    }

    #[test]
    fn ordering_matches_raw_byte_ordering() {
        let mut keys = ["pear", "apple", "z", "aa", "Apple"]
            .into_iter()
            .map(|s| StrKey::new(s).unwrap())
            .collect::<Vec<_>>();
        keys.sort();

        let mut raw = ["pear", "apple", "z", "aa", "Apple"].map(|s| s.as_bytes().to_vec());
        raw.sort();

        let sorted: Vec<Vec<u8>> = keys.iter().map(|k| k.as_bytes().to_vec()).collect();
        assert_eq!(sorted, raw.to_vec(), "StrKey ordering must match raw key-byte ordering");
    }

    #[test]
    fn round_trips_through_bytes() {
        let key = StrKey::new("acme-corp-2026").unwrap();
        let restored = StrKey::from_bytes(key.as_bytes()).unwrap();
        assert_eq!(key, restored);
        assert_eq!(restored.as_str(), "acme-corp-2026");
    }

    #[test]
    fn debug_and_display_show_the_key_not_the_buffer() {
        let key = StrKey::new("slug").unwrap();
        assert_eq!(format!("{key}"), "slug");
        assert_eq!(format!("{key:?}"), "\"slug\"");
    }
}

#[cfg(test)]
mod doc_id_tests {
    use super::*;

    // ── DocId serialization ─────────────────────────────────────────────────

    #[test]
    fn test_doc_id_u64_roundtrip() {
        let id = DocId::U64(12345);
        let bytes = id.to_bytes();
        let restored = DocId::from_bytes(&bytes, KeyType::U64).unwrap();
        assert_eq!(id, restored);
    }

    #[test]
    fn test_doc_id_u128_roundtrip() {
        let id = DocId::U128(u128::MAX / 2);
        let bytes = id.to_bytes();
        let restored = DocId::from_bytes(&bytes, KeyType::U128).unwrap();
        assert_eq!(id, restored);
    }

    #[test]
    fn test_doc_id_uuid_roundtrip() {
        let id = DocId::Uuid(0xdeadbeef_cafebabe_12345678_9abcdef0);
        let bytes = id.to_bytes();
        let restored = DocId::from_bytes(&bytes, KeyType::Uuid).unwrap();
        assert_eq!(id, restored);
    }

    #[test]
    fn test_doc_id_ordering() {
        // big-endian encoding means byte-level ordering == numeric ordering
        let ids: Vec<DocId> = (0u64..5).map(DocId::U64).collect();
        let encoded: Vec<Vec<u8>> = ids.iter().map(|id| id.to_bytes()).collect();
        let sorted = {
            let mut c = encoded.clone();
            c.sort();
            c
        };
        assert_eq!(encoded, sorted);
    }

    #[test]
    fn test_invalid_key_size_rejected() {
        assert!(DocId::from_bytes(&[0u8; 3], KeyType::U64).is_err());
        assert!(DocId::from_bytes(&[0u8; 5], KeyType::U128).is_err());
    }

    #[test]
    fn test_doc_id_str_roundtrip() {
        let id = DocId::Str(StrKey::new("acme-corp-2026").unwrap());
        let bytes = id.to_bytes();
        assert_eq!(bytes, b"acme-corp-2026", "a str key is stored verbatim, not encoded");
        let restored = DocId::from_bytes(&bytes, KeyType::Str).unwrap();
        assert_eq!(id, restored);
    }

    /// The same property the integer types get from big-endian encoding: the
    /// order of the stored key bytes is the order of the IDs, which is what
    /// makes `scan_range` over document IDs meaningful.
    #[test]
    fn test_doc_id_str_ordering_matches_byte_ordering() {
        let ids: Vec<DocId> = ["aa", "acme", "acme-corp", "b", "z"]
            .into_iter()
            .map(|s| DocId::Str(StrKey::new(s).unwrap()))
            .collect();

        let encoded: Vec<Vec<u8>> = ids.iter().map(|id| id.to_bytes()).collect();
        let sorted = {
            let mut c = encoded.clone();
            c.sort();
            c
        };
        assert_eq!(encoded, sorted, "str ids must sort lexicographically by their key bytes");

        // And the `DocId` ordering itself agrees — a derived `Ord` on the
        // inline buffer would compare length first and put "z" before "aa".
        let mut by_doc_id = ids.clone();
        by_doc_id.sort();
        assert_eq!(by_doc_id, ids);
    }

    #[test]
    fn test_invalid_str_key_rejected() {
        // Over the cap, empty, and non-UTF-8 all fail to decode rather than
        // producing a truncated or lossy id.
        assert!(DocId::from_bytes(&[b'x'; crate::doc_store::key::MAX_STR_KEY_LEN + 1], KeyType::Str).is_err());
        assert!(DocId::from_bytes(&[], KeyType::Str).is_err());
        assert!(DocId::from_bytes(&[0xff, 0xfe], KeyType::Str).is_err());
    }
}
