//! Schema for **raw key-value** store namespaces.
//!
//! Split out of `schema.rs`, which held two unrelated vocabularies in one file:
//! documents (attributes, indexes, amendments) and raw key-value stores (key and
//! value encodings). They share only [`StoreType`], the discriminant that says
//! which of the two a persisted schema is.
//!
//! Key encodings are chosen so that byte order matches value order — integer
//! keys are big-endian, which is what makes range scans work on them.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::doc_store::error::SchemaError;
use crate::doc_store::key::StrKey;
use crate::doc_store::schema::{StoreType, peek_store_type};

/// Key type for a KV store namespace.
///
/// `Bool` is intentionally absent — a boolean is not a useful lookup key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvKeyType {
    #[serde(rename = "str")]
    Str,
    #[serde(rename = "int")]
    Int,
}

/// Value type for a KV store namespace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvValueType {
    #[serde(rename = "int")]
    Int,
    #[serde(rename = "str")]
    Str,
    #[serde(rename = "f32")]
    F32,
    #[serde(rename = "vec_f32")]
    VecF32,
}

/// Schema definition for a KV store namespace.
///
/// One `KvStoreSchema` maps to exactly one namespace in the underlying
/// `minnal_db`.  Values are stored as raw bytes according to `value_type`.
/// Indices and attribute declarations are not supported; use `DocStoreSchema`
/// if you need them.
///
/// On disk the JSON is distinguished from `DocStoreSchema` by the mandatory
/// [`store_type`](Self::store_type) field, which must be `"kv"`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvStoreSchema {
    pub namespace: String,
    /// Mandatory store-kind discriminant; must be [`StoreType::Kv`].
    pub store_type: StoreType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ns_id: Option<u32>,
    pub key_type: KvKeyType,
    pub value_type: KvValueType,
    /// Enable ANN semantic search on this namespace.  Only valid when
    /// `value_type = str`; the stored string is the text that gets embedded.
    #[serde(default)]
    pub semantic_search_enabled: bool,
}

impl KvStoreSchema {
    /// Validate the schema without saving.
    pub fn validate(&self) -> Result<(), SchemaError> {
        if self.store_type != StoreType::Kv {
            return Err(SchemaError::WrongStoreType {
                namespace: self.namespace.clone(),
                expected: "kv",
                found: "doc",
            });
        }
        if self.namespace.is_empty() || !self.namespace.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
            return Err(SchemaError::InvalidNamespace);
        }
        if self.semantic_search_enabled && self.value_type != KvValueType::Str {
            return Err(SchemaError::KvSemanticSearchOnlyForStr);
        }
        Ok(())
    }

    /// Returns `true` when semantic search is enabled and the value type is `Str`.
    pub fn is_semantic_search_enabled(&self) -> bool {
        self.semantic_search_enabled && self.value_type == KvValueType::Str
    }

    /// Persist the schema to `schema_dir/<namespace>.json` (atomic write).
    pub fn save(&self, schema_dir: &Path) -> Result<(), SchemaError> {
        let path = schema_dir.join(format!("{}.json", self.namespace));
        let tmp = path.with_extension("tmp");
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&tmp, &json)?;
        std::fs::rename(&tmp, &path)?;
        Ok(())
    }

    /// Load the KV schema for `namespace` from `schema_dir/<namespace>.json`.
    ///
    /// Returns [`SchemaError::WrongStoreType`] if the file on disk is a document
    /// store rather than a KV store.
    pub fn load(schema_dir: &Path, namespace: &str) -> Result<Self, SchemaError> {
        let path = schema_dir.join(format!("{namespace}.json"));
        if !path.exists() {
            return Err(SchemaError::NotFound {
                namespace: namespace.to_owned(),
            });
        }
        let json = std::fs::read_to_string(path)?;
        if let Some(found) = peek_store_type(&json)
            && found != StoreType::Kv
        {
            return Err(SchemaError::WrongStoreType {
                namespace: namespace.to_owned(),
                expected: "kv",
                found: found.label(),
            });
        }
        serde_json::from_str(&json).map_err(SchemaError::Serialize)
    }
}

impl KvKeyType {
    /// Serialize a JSON value as raw key bytes for storage in minnal_db.
    ///
    /// Int keys use big-endian encoding so that lexicographic byte order
    /// matches numeric order, enabling range scans. Str keys are stored
    /// verbatim after validation through [`StrKey`], which bounds them to
    /// [`MAX_STR_KEY_LEN`](crate::doc_store::key::MAX_STR_KEY_LEN) bytes.
    pub fn serialize_key(&self, key: &serde_json::Value) -> Result<Vec<u8>, SchemaError> {
        match self {
            KvKeyType::Str => {
                let s = key.as_str().ok_or(SchemaError::KvKeyTypeMismatch { expected: "string" })?;
                Ok(StrKey::new(s)?.as_bytes().to_vec())
            }
            KvKeyType::Int => {
                let n = key.as_i64().ok_or(SchemaError::KvKeyTypeMismatch { expected: "integer" })?;
                Ok(n.to_be_bytes().to_vec())
            }
        }
    }

    /// Parse a raw URL path segment and serialize it as key bytes.
    ///
    /// Str keys go through the same [`StrKey`] validation as
    /// [`serialize_key`](Self::serialize_key) — this is the choke point for the
    /// operations that address a key by string (get, delete, scan bounds), so
    /// no over-long or empty key reaches storage through any of them.
    pub fn serialize_key_from_str(&self, raw: &str) -> Result<Vec<u8>, SchemaError> {
        match self {
            KvKeyType::Str => Ok(StrKey::new(raw)?.as_bytes().to_vec()),
            KvKeyType::Int => {
                let n: i64 = raw.parse().map_err(|_| SchemaError::KvKeyTypeMismatch { expected: "integer" })?;
                Ok(n.to_be_bytes().to_vec())
            }
        }
    }

    /// Deserialize raw key bytes back to a JSON value.
    pub fn deserialize_key(&self, bytes: &[u8]) -> Result<serde_json::Value, SchemaError> {
        match self {
            KvKeyType::Str => {
                let s = std::str::from_utf8(bytes).map_err(|_| SchemaError::KvValueCorrupt)?;
                Ok(serde_json::Value::String(s.to_owned()))
            }
            KvKeyType::Int => {
                let arr: [u8; 8] = bytes.try_into().map_err(|_| SchemaError::KvValueCorrupt)?;
                Ok(serde_json::Value::from(i64::from_be_bytes(arr)))
            }
        }
    }
}

impl KvValueType {
    /// Serialize a JSON value to raw bytes for storage.
    pub fn serialize_value(&self, value: &serde_json::Value) -> Result<Vec<u8>, SchemaError> {
        match self {
            KvValueType::Int => {
                let n = value.as_i64().ok_or(SchemaError::KvValueTypeMismatch { expected: "integer" })?;
                Ok(n.to_le_bytes().to_vec())
            }
            KvValueType::Str => {
                let s = value.as_str().ok_or(SchemaError::KvValueTypeMismatch { expected: "string" })?;
                Ok(s.as_bytes().to_vec())
            }
            KvValueType::F32 => {
                let n = value.as_f64().ok_or(SchemaError::KvValueTypeMismatch { expected: "number (f32)" })?;
                Ok((n as f32).to_le_bytes().to_vec())
            }
            KvValueType::VecF32 => {
                let arr = value.as_array().ok_or(SchemaError::KvValueTypeMismatch {
                    expected: "array of numbers",
                })?;
                let mut bytes = Vec::with_capacity(arr.len() * 4);
                for v in arr {
                    let n = v.as_f64().ok_or(SchemaError::KvValueTypeMismatch {
                        expected: "array element must be a number",
                    })?;
                    bytes.extend_from_slice(&(n as f32).to_le_bytes());
                }
                Ok(bytes)
            }
        }
    }

    /// Deserialize raw bytes back to a JSON value.
    pub fn deserialize_value(&self, bytes: &[u8]) -> Result<serde_json::Value, SchemaError> {
        match self {
            KvValueType::Int => {
                let arr: [u8; 8] = bytes.try_into().map_err(|_| SchemaError::KvValueCorrupt)?;
                Ok(serde_json::Value::from(i64::from_le_bytes(arr)))
            }
            KvValueType::Str => {
                let s = std::str::from_utf8(bytes).map_err(|_| SchemaError::KvValueCorrupt)?;
                Ok(serde_json::Value::String(s.to_owned()))
            }
            KvValueType::F32 => {
                let arr: [u8; 4] = bytes.try_into().map_err(|_| SchemaError::KvValueCorrupt)?;
                let f = f32::from_le_bytes(arr);
                Ok(serde_json::json!(f))
            }
            KvValueType::VecF32 => {
                if !bytes.len().is_multiple_of(4) {
                    return Err(SchemaError::KvValueCorrupt);
                }
                let floats: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                serde_json::to_value(&floats).map_err(SchemaError::Serialize)
            }
        }
    }
}

/// A minimal valid KV schema, shared by this module's tests and by the
/// store-type discrimination tests in [`crate::doc_store::schema`] — those sit
/// on the boundary between the two schema families, so they need a builder from
/// each side.
#[cfg(test)]
pub(crate) fn valid_kv_schema() -> KvStoreSchema {
    KvStoreSchema {
        store_type: StoreType::Kv,
        namespace: "cache".to_owned(),
        ns_id: None,
        key_type: KvKeyType::Str,
        value_type: KvValueType::Str,
        semantic_search_enabled: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── KvStoreSchema validation ──────────────────────────────────────────────

    #[test]
    fn kv_schema_valid_passes() {
        assert!(valid_kv_schema().validate().is_ok());
    }

    #[test]
    fn kv_schema_invalid_namespace_rejected() {
        let mut s = valid_kv_schema();
        s.namespace = "my namespace".to_owned();
        assert!(matches!(s.validate(), Err(SchemaError::InvalidNamespace)));
    }

    #[test]
    fn kv_schema_empty_namespace_rejected() {
        let mut s = valid_kv_schema();
        s.namespace = String::new();
        assert!(matches!(s.validate(), Err(SchemaError::InvalidNamespace)));
    }

    #[test]
    fn kv_schema_semantic_search_on_non_str_value_rejected() {
        for vt in [KvValueType::Int, KvValueType::F32, KvValueType::VecF32] {
            let s = KvStoreSchema {
                store_type: StoreType::Kv,
                namespace: "ns".to_owned(),
                ns_id: None,
                key_type: KvKeyType::Str,
                value_type: vt,
                semantic_search_enabled: true,
            };
            assert!(
                matches!(s.validate(), Err(SchemaError::KvSemanticSearchOnlyForStr)),
                "expected error for value_type={vt:?}"
            );
        }
    }

    #[test]
    fn kv_schema_semantic_search_on_str_is_valid() {
        let s = KvStoreSchema {
            store_type: StoreType::Kv,
            namespace: "ns".to_owned(),
            ns_id: None,
            key_type: KvKeyType::Str,
            value_type: KvValueType::Str,
            semantic_search_enabled: true,
        };
        assert!(s.validate().is_ok());
        assert!(s.is_semantic_search_enabled());
    }

    #[test]
    fn kv_schema_semantic_search_disabled_by_default() {
        assert!(!valid_kv_schema().is_semantic_search_enabled());
    }

    #[test]
    fn kv_schema_save_and_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let schema = KvStoreSchema {
            store_type: StoreType::Kv,
            namespace: "tokens".to_owned(),
            ns_id: Some(7),
            key_type: KvKeyType::Int,
            value_type: KvValueType::F32,
            semantic_search_enabled: false,
        };
        schema.save(dir.path()).unwrap();
        let loaded = KvStoreSchema::load(dir.path(), "tokens").unwrap();
        assert_eq!(schema, loaded);
    }

    #[test]
    fn kv_schema_load_missing_returns_not_found() {
        let dir = tempfile::tempdir().unwrap();
        assert!(matches!(KvStoreSchema::load(dir.path(), "nope"), Err(SchemaError::NotFound { .. })));
    }

    // ── KvKeyType serialisation ───────────────────────────────────────────────

    #[test]
    fn kv_key_str_serialize_roundtrip() {
        let key = serde_json::Value::String("hello world".to_owned());
        let bytes = KvKeyType::Str.serialize_key(&key).unwrap();
        let restored = KvKeyType::Str.deserialize_key(&bytes).unwrap();
        assert_eq!(key, restored);
    }

    #[test]
    fn kv_key_int_serialize_roundtrip() {
        for n in [0i64, 1, -1, i64::MIN, i64::MAX, 42] {
            let key = serde_json::Value::from(n);
            let bytes = KvKeyType::Int.serialize_key(&key).unwrap();
            let restored = KvKeyType::Int.deserialize_key(&bytes).unwrap();
            assert_eq!(key, restored, "roundtrip failed for n={n}");
        }
    }

    #[test]
    fn kv_key_int_ordering_preserved_by_big_endian() {
        // Big-endian encoding means lexicographic byte order == numeric order.
        let keys: Vec<i64> = vec![0, 1, 42, 100, i64::MAX];
        let encoded: Vec<Vec<u8>> = keys
            .iter()
            .map(|&n| KvKeyType::Int.serialize_key(&serde_json::Value::from(n)).unwrap())
            .collect();
        let mut sorted = encoded.clone();
        sorted.sort();
        assert_eq!(encoded, sorted);
    }

    #[test]
    fn kv_key_str_rejects_non_string() {
        let err = KvKeyType::Str.serialize_key(&serde_json::json!(42));
        assert!(matches!(err, Err(SchemaError::KvKeyTypeMismatch { .. })));
    }

    #[test]
    fn kv_key_int_rejects_non_integer() {
        let err = KvKeyType::Int.serialize_key(&serde_json::json!("not-a-number"));
        assert!(matches!(err, Err(SchemaError::KvKeyTypeMismatch { .. })));
    }

    #[test]
    fn kv_key_int_rejects_float() {
        let err = KvKeyType::Int.serialize_key(&serde_json::json!(1.5));
        assert!(matches!(err, Err(SchemaError::KvKeyTypeMismatch { .. })));
    }

    #[test]
    fn kv_key_from_str_str_key() {
        let bytes = KvKeyType::Str.serialize_key_from_str("mykey").unwrap();
        assert_eq!(bytes, b"mykey");
    }

    #[test]
    fn kv_key_from_str_int_key() {
        let bytes = KvKeyType::Int.serialize_key_from_str("42").unwrap();
        let expected = 42i64.to_be_bytes().to_vec();
        assert_eq!(bytes, expected);
    }

    #[test]
    fn kv_key_from_str_invalid_int_rejected() {
        assert!(matches!(
            KvKeyType::Int.serialize_key_from_str("not-a-number"),
            Err(SchemaError::KvKeyTypeMismatch { .. })
        ));
    }

    /// Both encoders must enforce the cap: `serialize_key` covers `kv_put`,
    /// `serialize_key_from_str` covers get/delete/scan-bounds. An unbounded key
    /// used to reach the row map, where the length check *panics*.
    #[test]
    fn kv_str_keys_are_length_capped_by_both_encoders() {
        let over = "x".repeat(crate::doc_store::key::MAX_STR_KEY_LEN + 1);

        assert!(matches!(
            KvKeyType::Str.serialize_key(&serde_json::Value::String(over.clone())),
            Err(SchemaError::StrKeyTooLong { .. })
        ));
        assert!(matches!(
            KvKeyType::Str.serialize_key_from_str(&over),
            Err(SchemaError::StrKeyTooLong { .. })
        ));

        let at_limit = "x".repeat(crate::doc_store::key::MAX_STR_KEY_LEN);
        assert!(KvKeyType::Str.serialize_key(&serde_json::Value::String(at_limit.clone())).is_ok());
        assert!(KvKeyType::Str.serialize_key_from_str(&at_limit).is_ok());
    }

    #[test]
    fn kv_str_keys_reject_empty() {
        assert!(matches!(
            KvKeyType::Str.serialize_key(&serde_json::Value::String(String::new())),
            Err(SchemaError::EmptyStrKey)
        ));
        assert!(matches!(KvKeyType::Str.serialize_key_from_str(""), Err(SchemaError::EmptyStrKey)));
    }

    /// The cap counts UTF-8 bytes, so a key well under 50 characters can still
    /// be rejected — and the byte count is what the row map and every SSTable
    /// entry actually pay for.
    #[test]
    fn kv_str_key_cap_counts_bytes_not_chars() {
        let twenty_chars = "日".repeat(20); // 60 bytes
        assert!(matches!(
            KvKeyType::Str.serialize_key_from_str(&twenty_chars),
            Err(SchemaError::StrKeyTooLong { len: 60, .. })
        ));
    }

    // ── KvValueType serialisation ─────────────────────────────────────────────

    #[test]
    fn kv_value_int_roundtrip() {
        for n in [0i64, -1, 1, i64::MIN, i64::MAX] {
            let v = serde_json::Value::from(n);
            let bytes = KvValueType::Int.serialize_value(&v).unwrap();
            let restored = KvValueType::Int.deserialize_value(&bytes).unwrap();
            assert_eq!(v, restored, "roundtrip failed for n={n}");
        }
    }

    #[test]
    fn kv_value_str_roundtrip() {
        let v = serde_json::Value::String("the quick brown fox".to_owned());
        let bytes = KvValueType::Str.serialize_value(&v).unwrap();
        let restored = KvValueType::Str.deserialize_value(&bytes).unwrap();
        assert_eq!(v, restored);
    }

    #[test]
    fn kv_value_f32_roundtrip() {
        // Use values exactly representable as f32 to avoid precision surprises.
        for f in [0.0f32, 1.0, -1.0, 0.5, 1024.25] {
            let v = serde_json::json!(f);
            let bytes = KvValueType::F32.serialize_value(&v).unwrap();
            let restored = KvValueType::F32.deserialize_value(&bytes).unwrap();
            let got = restored.as_f64().unwrap() as f32;
            assert!((got - f).abs() < f32::EPSILON, "roundtrip failed for f={f}");
        }
    }

    #[test]
    fn kv_value_vec_f32_roundtrip() {
        let v = serde_json::json!([1.0f32, -0.5f32, 0.25f32]);
        let bytes = KvValueType::VecF32.serialize_value(&v).unwrap();
        let restored = KvValueType::VecF32.deserialize_value(&bytes).unwrap();
        let arr = restored.as_array().unwrap();
        assert_eq!(arr.len(), 3);
        let expected = [1.0f32, -0.5, 0.25];
        for (got, exp) in arr.iter().zip(expected.iter()) {
            assert!((got.as_f64().unwrap() as f32 - exp).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn kv_value_vec_f32_empty_roundtrip() {
        let v = serde_json::json!([]);
        let bytes = KvValueType::VecF32.serialize_value(&v).unwrap();
        assert!(bytes.is_empty());
        let restored = KvValueType::VecF32.deserialize_value(&bytes).unwrap();
        assert_eq!(restored.as_array().unwrap().len(), 0);
    }

    #[test]
    fn kv_value_int_rejects_str() {
        assert!(matches!(
            KvValueType::Int.serialize_value(&serde_json::json!("text")),
            Err(SchemaError::KvValueTypeMismatch { .. })
        ));
    }

    #[test]
    fn kv_value_str_rejects_number() {
        assert!(matches!(
            KvValueType::Str.serialize_value(&serde_json::json!(42)),
            Err(SchemaError::KvValueTypeMismatch { .. })
        ));
    }

    #[test]
    fn kv_value_f32_rejects_string() {
        assert!(matches!(
            KvValueType::F32.serialize_value(&serde_json::json!("bad")),
            Err(SchemaError::KvValueTypeMismatch { .. })
        ));
    }

    #[test]
    fn kv_value_vec_f32_rejects_non_array() {
        assert!(matches!(
            KvValueType::VecF32.serialize_value(&serde_json::json!(42)),
            Err(SchemaError::KvValueTypeMismatch { .. })
        ));
    }

    #[test]
    fn kv_value_vec_f32_rejects_mixed_array() {
        assert!(matches!(
            KvValueType::VecF32.serialize_value(&serde_json::json!([1.0, "bad"])),
            Err(SchemaError::KvValueTypeMismatch { .. })
        ));
    }

    #[test]
    fn kv_value_int_corrupt_bytes_rejected() {
        assert!(matches!(
            KvValueType::Int.deserialize_value(&[0u8; 5]), // wrong length
            Err(SchemaError::KvValueCorrupt)
        ));
    }

    #[test]
    fn kv_value_f32_corrupt_bytes_rejected() {
        assert!(matches!(
            KvValueType::F32.deserialize_value(&[0u8; 3]), // wrong length
            Err(SchemaError::KvValueCorrupt)
        ));
    }

    #[test]
    fn kv_value_vec_f32_unaligned_bytes_rejected() {
        assert!(matches!(
            KvValueType::VecF32.deserialize_value(&[0u8; 7]), // not multiple of 4
            Err(SchemaError::KvValueCorrupt)
        ));
    }

    #[test]
    fn kv_value_str_invalid_utf8_rejected() {
        assert!(matches!(
            KvValueType::Str.deserialize_value(&[0xFF, 0xFE]),
            Err(SchemaError::KvValueCorrupt)
        ));
    }
}
