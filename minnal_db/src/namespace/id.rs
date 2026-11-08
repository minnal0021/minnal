//! `NamespaceId`: a namespace's identity, a random 128-bit id (a version-4 UUID).
//!
//! An id is drawn when a namespace is created and never handed out again, so a
//! name dropped and created again gets a new id: WAL records, field-index files
//! and caches of the old namespace can never be taken for the new one's. No
//! counter is kept anywhere, so creating a namespace reads no shared state to
//! pick its id.
//!
//! Written as the usual hyphenated UUID text in JSON, directory names and logs,
//! and as 16 bytes in the binary formats (the registry and both WALs).

use std::fmt;
use std::io::Read;
use std::str::FromStr;

use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};

/// A namespace's id. See the module docs.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct NamespaceId(u128);

impl NamespaceId {
    /// The `default` namespace's fixed id.
    pub const DEFAULT: Self = Self(0);
    /// The `system` namespace's fixed id.
    pub const SYSTEM: Self = Self(1);

    /// A new random id (version-4 UUID), read from the operating system's
    /// random source.
    pub fn random() -> std::io::Result<Self> {
        let mut bytes = [0u8; 16];
        std::fs::File::open("/dev/urandom")?.read_exact(&mut bytes)?;
        // RFC 9562 version 4, variant 10.
        bytes[6] = (bytes[6] & 0x0f) | 0x40;
        bytes[8] = (bytes[8] & 0x3f) | 0x80;
        Ok(Self(u128::from_be_bytes(bytes)))
    }

    /// The id from its 128-bit value.
    pub const fn from_u128(value: u128) -> Self {
        Self(value)
    }

    /// The id's 128-bit value.
    pub const fn as_u128(self) -> u128 {
        self.0
    }

    /// The id as 16 big-endian bytes, the form the binary formats store.
    pub const fn to_bytes(self) -> [u8; 16] {
        self.0.to_be_bytes()
    }

    /// The id from 16 big-endian bytes.
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(u128::from_be_bytes(bytes))
    }
}

/// Tests name namespaces by small numbers; `id == 3` reads better there than
/// `id == NamespaceId::from_u128(3)`.
#[cfg(test)]
impl PartialEq<u128> for NamespaceId {
    fn eq(&self, other: &u128) -> bool {
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
        let b = self.to_bytes();
        let hex = |r: std::ops::Range<usize>| b[r].iter().map(|x| format!("{x:02x}")).collect::<String>();
        write!(f, "{}-{}-{}-{}-{}", hex(0..4), hex(4..6), hex(6..8), hex(8..10), hex(10..16))
    }
}

impl fmt::Debug for NamespaceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NamespaceId({self})")
    }
}

/// Text that is not a hyphenated UUID.
#[derive(Debug, thiserror::Error)]
#[error("not a namespace id (a hyphenated UUID): {0:?}")]
pub struct BadNamespaceId(pub String);

impl FromStr for NamespaceId {
    type Err = BadNamespaceId;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let bad = || BadNamespaceId(s.to_owned());
        let groups: Vec<&str> = s.split('-').collect();
        if groups.iter().map(|g| g.len()).collect::<Vec<_>>() != [8, 4, 4, 4, 12] {
            return Err(bad());
        }
        let hex: String = groups.concat();
        u128::from_str_radix(&hex, 16).map(Self).map_err(|_| bad())
    }
}

impl serde::Serialize for NamespaceId {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> serde::Deserialize<'de> for NamespaceId {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = <std::borrow::Cow<'de, str>>::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_ids_are_version_4_and_differ() {
        let a = NamespaceId::random().unwrap();
        let b = NamespaceId::random().unwrap();
        assert_ne!(a, b);
        let text = a.to_string();
        assert_eq!(text.len(), 36);
        assert_eq!(&text[14..15], "4", "version nibble: {text}");
        assert!(matches!(&text[19..20], "8" | "9" | "a" | "b"), "variant: {text}");
    }

    #[test]
    fn text_bytes_and_json_round_trip() {
        let id = NamespaceId::random().unwrap();
        assert_eq!(id.to_string().parse::<NamespaceId>().unwrap(), id);
        assert_eq!(NamespaceId::from_bytes(id.to_bytes()), id);
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(serde_json::from_str::<NamespaceId>(&json).unwrap(), id);
        assert_eq!(NamespaceId::DEFAULT.to_string(), "00000000-0000-0000-0000-000000000000");
    }

    #[test]
    fn malformed_text_is_rejected() {
        for bad in [
            "",
            "1",
            "00000000000000000000000000000000",
            "0000000-00000-0000-0000-000000000000",
            "zzzzzzzz-0000-0000-0000-000000000000",
        ] {
            assert!(bad.parse::<NamespaceId>().is_err(), "{bad}");
        }
    }
}
