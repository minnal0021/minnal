//! The system catalog's records: how the namespace registry is kept in the
//! `system_catalog` namespace.
//!
//! The catalog is an ordinary namespace with a fixed id
//! ([`NamespaceId::CATALOG`]) and a fixed name, so it is opened before anything
//! else is read. It holds three kinds of key:
//!
//! | Key | Value |
//! |---|---|
//! | `next_id` | the next id to hand out (u64, little-endian) |
//! | `ns/{name}` | a namespace's id and TTL settings |
//! | `field/{ns_id}/{field_name}` | a field index's id, value type and `dropped` flag |
//!
//! `ns_id` in a field key is 8 big-endian bytes, so one namespace's fields
//! are one prefix scan. Every write is WAL-backed and durable on return; the
//! registry is changed only after its catalog writes return
//! (`Database::update_registry`), so the in-memory registry is always what is
//! on disk.

use std::time::Duration;

use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};

use super::NamespaceId;
use crate::error::{KVError, Result};
use crate::field_index::{FieldId, FieldMeta, IndexValueType};

/// The catalog namespace's name. Its data directory is `ns_system_catalog`.
pub const CATALOG_NAMESPACE: &str = "system_catalog";

const NEXT_ID_KEY: &[u8] = b"next_id";
const NS_PREFIX: &[u8] = b"ns/";
const FIELD_PREFIX: &[u8] = b"field/";

/// The first byte of every record value: its layout's version.
const RECORD_VERSION: u8 = 1;

/// A write to the catalog, made by `Database::update_registry`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CatalogWrite {
    Put { key: Vec<u8>, value: Vec<u8> },
    Delete { key: Vec<u8> },
}

/// One decoded catalog record.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum CatalogEntry {
    NextId(u64),
    Namespace {
        name: String,
        id: NamespaceId,
        ttl: Option<(Duration, usize)>,
    },
    Field {
        ns_id: NamespaceId,
        meta: FieldMeta,
    },
}

#[derive(Archive, RkyvSerialize, RkyvDeserialize)]
struct NamespaceRecord {
    id: u64,
    ttl: Option<TtlRecord>,
}

#[derive(Archive, RkyvSerialize, RkyvDeserialize)]
struct TtlRecord {
    /// Record lifetime in milliseconds.
    ttl_millis: u64,
    /// Cap on records tombstoned per TTL pass.
    max_deletes: u64,
}

#[derive(Archive, RkyvSerialize, RkyvDeserialize)]
struct FieldRecord {
    field_id: u64,
    value_type: u8,
    dropped: bool,
}

fn value_type_tag(t: IndexValueType) -> u8 {
    match t {
        IndexValueType::Bool => 0,
        IndexValueType::Int => 1,
        IndexValueType::Str => 2,
    }
}

fn value_type_of(tag: u8) -> Option<IndexValueType> {
    match tag {
        0 => Some(IndexValueType::Bool),
        1 => Some(IndexValueType::Int),
        2 => Some(IndexValueType::Str),
        _ => None,
    }
}

/// A record value: its version byte, then its rkyv payload.
fn versioned(payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + payload.len());
    out.push(RECORD_VERSION);
    out.extend_from_slice(payload);
    out
}

fn corrupt(key: &[u8], why: impl std::fmt::Display) -> KVError {
    KVError::Serialization(format!(
        "the system catalog's record {:?} is unreadable: {why}",
        String::from_utf8_lossy(key)
    ))
}

/// The payload of a record value, after its version byte.
fn payload<'v>(key: &[u8], value: &'v [u8]) -> Result<&'v [u8]> {
    match value.split_first() {
        Some((&RECORD_VERSION, rest)) => Ok(rest),
        Some((v, _)) => Err(corrupt(key, format!("record version {v}, expected {RECORD_VERSION}"))),
        None => Err(corrupt(key, "empty value")),
    }
}

/// The write saving the counter's next id.
pub(crate) fn next_id_write(next: u64) -> CatalogWrite {
    CatalogWrite::Put {
        key: NEXT_ID_KEY.to_vec(),
        value: next.to_le_bytes().to_vec(),
    }
}

/// The key of namespace `name`'s record.
pub(crate) fn namespace_key(name: &str) -> Vec<u8> {
    [NS_PREFIX, name.as_bytes()].concat()
}

/// The write saving namespace `name`'s record.
pub(crate) fn namespace_write(name: &str, id: NamespaceId, ttl: Option<(Duration, usize)>) -> CatalogWrite {
    let record = NamespaceRecord {
        id: id.as_u64(),
        ttl: ttl.map(|(ttl, max_deletes)| TtlRecord {
            ttl_millis: ttl.as_millis() as u64,
            max_deletes: max_deletes as u64,
        }),
    };
    CatalogWrite::Put {
        key: namespace_key(name),
        value: versioned(&rkyv::to_bytes::<rkyv::rancor::Error>(&record).expect("a namespace record serialises")),
    }
}

/// The key of field `field_name` of namespace `ns_id`.
pub(crate) fn field_key(ns_id: NamespaceId, field_name: &str) -> Vec<u8> {
    [FIELD_PREFIX, &ns_id.to_bytes(), b"/", field_name.as_bytes()].concat()
}

/// The write saving a field's record.
pub(crate) fn field_write(ns_id: NamespaceId, meta: &FieldMeta) -> CatalogWrite {
    let record = FieldRecord {
        field_id: meta.field_id,
        value_type: value_type_tag(meta.field_type),
        dropped: meta.dropped,
    };
    CatalogWrite::Put {
        key: field_key(ns_id, &meta.field_name),
        value: versioned(&rkyv::to_bytes::<rkyv::rancor::Error>(&record).expect("a field record serialises")),
    }
}

/// Decode one stored record. A key of no known kind, or a value that does not
/// decode, is an error: the catalog is written only by this module.
pub(crate) fn decode(key: &[u8], value: &[u8]) -> Result<CatalogEntry> {
    if key == NEXT_ID_KEY {
        let bytes: [u8; 8] = value.try_into().map_err(|_| corrupt(key, "not 8 bytes"))?;
        return Ok(CatalogEntry::NextId(u64::from_le_bytes(bytes)));
    }
    if let Some(name) = key.strip_prefix(NS_PREFIX) {
        let name = std::str::from_utf8(name).map_err(|e| corrupt(key, e))?.to_owned();
        let record = rkyv::from_bytes::<NamespaceRecord, rkyv::rancor::Error>(payload(key, value)?).map_err(|e| corrupt(key, e))?;
        return Ok(CatalogEntry::Namespace {
            name,
            id: NamespaceId::from_u64(record.id),
            ttl: record.ttl.map(|t| (Duration::from_millis(t.ttl_millis), t.max_deletes as usize)),
        });
    }
    if let Some(rest) = key.strip_prefix(FIELD_PREFIX) {
        let n = NamespaceId::LEN;
        if rest.len() <= n || rest[n] != b'/' {
            return Err(corrupt(key, "a field key without its namespace id"));
        }
        let ns_id = NamespaceId::from_bytes(rest[..n].try_into().unwrap());
        let field_name = std::str::from_utf8(&rest[n + 1..]).map_err(|e| corrupt(key, e))?.to_owned();
        let record = rkyv::from_bytes::<FieldRecord, rkyv::rancor::Error>(payload(key, value)?).map_err(|e| corrupt(key, e))?;
        let field_type = value_type_of(record.value_type).ok_or_else(|| corrupt(key, format!("value type {}", record.value_type)))?;
        return Ok(CatalogEntry::Field {
            ns_id,
            meta: FieldMeta {
                field_id: record.field_id as FieldId,
                field_name,
                field_type,
                dropped: record.dropped,
            },
        });
    }
    Err(corrupt(key, "a key of no known kind"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decoded(w: CatalogWrite) -> CatalogEntry {
        match w {
            CatalogWrite::Put { key, value } => decode(&key, &value).unwrap(),
            CatalogWrite::Delete { .. } => panic!("a delete has no value"),
        }
    }

    #[test]
    fn every_record_round_trips() {
        assert_eq!(decoded(next_id_write(1029)), CatalogEntry::NextId(1029));
        let ttl = Some((Duration::from_secs(3600), 500));
        assert_eq!(
            decoded(namespace_write("users", NamespaceId::from_u64(1024), ttl)),
            CatalogEntry::Namespace {
                name: "users".into(),
                id: NamespaceId::from_u64(1024),
                ttl
            }
        );
        let meta = FieldMeta {
            field_id: 1030,
            field_name: "status".into(),
            field_type: IndexValueType::Str,
            dropped: true,
        };
        assert_eq!(
            decoded(field_write(NamespaceId::from_u64(1024), &meta)),
            CatalogEntry::Field {
                ns_id: NamespaceId::from_u64(1024),
                meta
            }
        );
    }

    #[test]
    fn one_namespaces_fields_share_a_prefix_no_other_namespace_matches() {
        let a = field_key(NamespaceId::from_u64(1024), "x");
        let b = field_key(NamespaceId::from_u64(1025), "x");
        let prefix = field_key(NamespaceId::from_u64(1024), "");
        assert!(a.starts_with(&prefix) && !b.starts_with(&prefix));
    }

    #[test]
    fn damaged_records_are_errors() {
        let CatalogWrite::Put { key, mut value } = namespace_write("users", NamespaceId::from_u64(1024), None) else {
            unreachable!()
        };
        assert!(decode(b"unknown", b"").is_err());
        assert!(decode(b"next_id", b"1234").is_err());
        assert!(decode(&key, &[]).is_err());
        value[0] = 9;
        assert!(decode(&key, &value).is_err(), "an unknown version");
        assert!(decode(b"field/1234", &value).is_err(), "a field key without its namespace id");
    }
}
