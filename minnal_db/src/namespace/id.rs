//! `NamespaceId`: a namespace's identity, a 64-bit id.
//!
//! An id comes from the system catalog's counter when a namespace is created,
//! and is never handed out again, so a name dropped and created again gets a
//! new id: WAL records, field-index files and caches of the old namespace can
//! never be taken for the new one's. Ids below [`NamespaceId::FIRST_ALLOCATED`]
//! are reserved for namespaces with fixed ids.
//!
//! Written as a decimal number in JSON, directory names and logs, and as 8
//! big-endian bytes in the binary formats (both WALs and their progress files).

use std::fmt;
#[cfg(test)]
use std::io::Read;
use std::str::FromStr;

use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};

/// A namespace's id. See the module docs.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct NamespaceId(u64);

impl NamespaceId {
    /// The `default` namespace's fixed id.
    pub const DEFAULT: Self = Self(0);
    /// The `system` namespace's fixed id.
    pub const SYSTEM: Self = Self(1);
    /// The `system_catalog` namespace's fixed id: it holds every other
    /// namespace's record, so it is known before anything is read.
    pub const CATALOG: Self = Self(2);
    /// The first id the catalog's counter hands out (namespaces and field
    /// indexes alike). Ids below it are reserved for namespaces with fixed ids.
    pub const FIRST_ALLOCATED: u64 = 1024;
    /// Bytes in the binary form ([`to_bytes`](Self::to_bytes)).
    pub const LEN: usize = 8;

    /// Test-only: a random id below 2^53, for tests that need ids no other
    /// test uses.
    #[cfg(test)]
    pub fn random() -> std::io::Result<Self> {
        let mut bytes = [0u8; 8];
        std::fs::File::open("/dev/urandom")?.read_exact(&mut bytes)?;
        Ok(Self(u64::from_be_bytes(bytes) >> 11))
    }

    /// The id from its 64-bit value.
    pub const fn from_u64(value: u64) -> Self {
        Self(value)
    }

    /// The id's 64-bit value.
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// The id as 8 big-endian bytes, the form the binary formats store.
    pub const fn to_bytes(self) -> [u8; 8] {
        self.0.to_be_bytes()
    }

    /// The id from 8 big-endian bytes.
    pub const fn from_bytes(bytes: [u8; 8]) -> Self {
        Self(u64::from_be_bytes(bytes))
    }
}

/// Tests name namespaces by small numbers; `id == 3` reads better there than
/// `id == NamespaceId::from_u64(3)`.
#[cfg(test)]
impl PartialEq<u64> for NamespaceId {
    fn eq(&self, other: &u64) -> bool {
        self.0 == *other
    }
}

impl From<&ArchivedNamespaceId> for NamespaceId {
    fn from(archived: &ArchivedNamespaceId) -> Self {
        Self(archived.0.to_native())
    }
}

impl fmt::Display for NamespaceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Debug for NamespaceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NamespaceId({self})")
    }
}

/// Text that is not a namespace id (a decimal number).
#[derive(Debug, thiserror::Error)]
#[error("not a namespace id (a decimal number): {0:?}")]
pub struct BadNamespaceId(pub String);

impl FromStr for NamespaceId {
    type Err = BadNamespaceId;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // `u64::from_str` takes a leading `+`; an id's text never has one.
        if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
            return Err(BadNamespaceId(s.to_owned()));
        }
        s.parse().map(Self).map_err(|_| BadNamespaceId(s.to_owned()))
    }
}

impl serde::Serialize for NamespaceId {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u64(self.0)
    }
}

impl<'de> serde::Deserialize<'de> for NamespaceId {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        u64::deserialize(deserializer).map(Self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_ids_differ_and_stay_exact_in_json() {
        let a = NamespaceId::random().unwrap();
        let b = NamespaceId::random().unwrap();
        assert_ne!(a, b);
        assert!(a.as_u64() < 1 << 53 && b.as_u64() < 1 << 53);
    }

    #[test]
    fn text_bytes_and_json_round_trip() {
        let id = NamespaceId::from_u64(1234567);
        assert_eq!(id.to_string(), "1234567");
        assert_eq!(id.to_string().parse::<NamespaceId>().unwrap(), id);
        assert_eq!(NamespaceId::from_bytes(id.to_bytes()), id);
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, "1234567");
        assert_eq!(serde_json::from_str::<NamespaceId>(&json).unwrap(), id);
        assert_eq!(NamespaceId::DEFAULT.to_string(), "0");
    }

    #[test]
    fn malformed_text_is_rejected() {
        for bad in ["", "+1", "-1", "1a", " 1", "18446744073709551616"] {
            assert!(bad.parse::<NamespaceId>().is_err(), "{bad}");
        }
    }
}
