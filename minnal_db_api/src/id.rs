use minnal_db::{DocId, DocStoreError, KeyType, StrKey};

use crate::error::AppError;

/// Serialize a [`DocId`] to a JSON-friendly value.
///
/// - `Uuid` → hyphenated UUID string
/// - `U64`  → JSON number
/// - `U128` → decimal string (too large for JSON number)
/// - `Str`  → the key string itself
pub fn doc_id_to_value(id: DocId) -> serde_json::Value {
    match id {
        DocId::Uuid(v) => serde_json::Value::String(format_uuid(v)),
        DocId::U64(v) => serde_json::json!(v),
        DocId::U128(v) => serde_json::Value::String(v.to_string()),
        DocId::Str(k) => serde_json::Value::String(k.as_str().to_owned()),
    }
}

/// Path segments under `/stores/{ns}/docs/` that are claimed by a static route
/// and therefore cannot address a document.
///
/// `GET /stores/{ns}/docs/prefix` is the prefix-scan endpoint, so a document
/// whose string key is `prefix` could be written but never fetched or deleted
/// by URL. Rejecting it at the boundary turns an invisible, unreachable record
/// into an immediate 400.
pub const RESERVED_DOC_KEYS: &[&str] = &["prefix"];

/// The same, for `/stores/{ns}/kv/` — `prefix` is the KV prefix scan and
/// `semantic-search` the KV ANN query.
pub const RESERVED_KV_KEYS: &[&str] = &["prefix", "semantic-search"];

/// Reject a string key that collides with a static route segment.
///
/// Only applies to string-keyed stores: an integer or UUID key can never
/// stringify to one of these.
pub fn check_reserved_key(raw: &str, reserved: &[&str]) -> Result<(), AppError> {
    if reserved.contains(&raw) {
        return Err(DocStoreError::InvalidId(format!(
            "'{raw}' is a reserved key: it collides with the /{raw} endpoint, so a record stored under it could not be read back"
        ))
        .into());
    }
    Ok(())
}

/// Parse a URL path segment into a [`DocId`] using the store's [`KeyType`].
///
/// - `Uuid`  → expects `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`
/// - `U64`   → decimal integer
/// - `U128`  → decimal integer
/// - `Str`   → the segment itself, validated to at most
///   [`MAX_STR_KEY_LEN`](minnal_db::MAX_STR_KEY_LEN) bytes and rejected if it
///   names a reserved route segment. Axum percent-decodes path segments, so
///   `acme%20corp` arrives here as `acme corp`.
pub fn parse_doc_id(s: &str, key_type: KeyType) -> Result<DocId, AppError> {
    match key_type {
        KeyType::Uuid => {
            let hex = s.replace('-', "");
            let v = u128::from_str_radix(&hex, 16).map_err(|_| DocStoreError::InvalidId(format!("invalid UUID: '{s}'")))?;
            Ok(DocId::Uuid(v))
        }
        KeyType::U64 => {
            let v = s.parse::<u64>().map_err(|_| DocStoreError::InvalidId(format!("invalid u64 id: '{s}'")))?;
            Ok(DocId::U64(v))
        }
        KeyType::U128 => {
            let v = s
                .parse::<u128>()
                .map_err(|_| DocStoreError::InvalidId(format!("invalid u128 id: '{s}'")))?;
            Ok(DocId::U128(v))
        }
        KeyType::Str => {
            check_reserved_key(s, RESERVED_DOC_KEYS)?;
            Ok(DocId::Str(StrKey::new(s).map_err(DocStoreError::Schema)?))
        }
    }
}

fn format_uuid(v: u128) -> String {
    let b = v.to_be_bytes();
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use minnal_db::{MAX_STR_KEY_LEN, SchemaError};

    #[test]
    fn str_ids_round_trip_through_the_url_form() {
        let id = parse_doc_id("acme-corp-2026", KeyType::Str).unwrap();
        assert_eq!(id, DocId::Str(StrKey::new("acme-corp-2026").unwrap()));
        assert_eq!(doc_id_to_value(id), serde_json::json!("acme-corp-2026"));
    }

    /// Axum percent-decodes before the extractor runs, so a decoded space must
    /// be accepted here rather than treated as invalid.
    #[test]
    fn str_ids_accept_decoded_spaces() {
        let id = parse_doc_id("acme corp", KeyType::Str).unwrap();
        assert_eq!(doc_id_to_value(id), serde_json::json!("acme corp"));
    }

    #[test]
    fn str_ids_are_length_capped_and_non_empty() {
        let over = "x".repeat(MAX_STR_KEY_LEN + 1);
        let err = parse_doc_id(&over, KeyType::Str).unwrap_err();
        assert!(matches!(err.inner, DocStoreError::Schema(SchemaError::StrKeyTooLong { .. })));

        let err = parse_doc_id("", KeyType::Str).unwrap_err();
        assert!(matches!(err.inner, DocStoreError::Schema(SchemaError::EmptyStrKey)));

        assert!(parse_doc_id(&"x".repeat(MAX_STR_KEY_LEN), KeyType::Str).is_ok());
    }

    /// `/stores/{ns}/docs/prefix` is the prefix-scan route, so a document keyed
    /// `prefix` would be write-only. Fail the write instead.
    #[test]
    fn reserved_route_segments_are_rejected_for_str_keys() {
        let err = parse_doc_id("prefix", KeyType::Str).unwrap_err();
        assert!(matches!(err.inner, DocStoreError::InvalidId(_)));

        // Not reserved on the KV side only — `semantic-search` is a KV route,
        // and `/stores/{ns}/docs/semantic-search` does not exist.
        assert!(parse_doc_id("semantic-search", KeyType::Str).is_ok());
        assert!(check_reserved_key("semantic-search", RESERVED_KV_KEYS).is_err());
    }

    /// The reserved list applies only to string keys — the integer types can
    /// never produce one of these segments.
    #[test]
    fn integer_key_types_are_unaffected() {
        assert_eq!(parse_doc_id("42", KeyType::U64).unwrap(), DocId::U64(42));
        assert!(parse_doc_id("prefix", KeyType::U64).is_err(), "still invalid, but as a bad u64");
    }
}
