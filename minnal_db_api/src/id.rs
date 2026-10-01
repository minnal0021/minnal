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
        KeyType::Uuid => Ok(DocId::Uuid(
            parse_uuid(s).ok_or_else(|| DocStoreError::InvalidId(format!("invalid UUID: '{s}'")))?,
        )),
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

/// Parse a UUID in its canonical form (`8-4-4-4-12` hex digits) or as 32 plain
/// hex digits. Stripping every `-` and parsing the rest as a number accepted
/// `1`, `1-2-3` and `+ff…`, and let non-canonical spellings alias one document.
fn parse_uuid(s: &str) -> Option<u128> {
    let hex: String = match s.len() {
        32 => s.to_owned(),
        36 if s.char_indices().all(|(i, c)| (c == '-') == matches!(i, 8 | 13 | 18 | 23)) => s.replace('-', ""),
        _ => return None,
    };
    let bytes: [u8; 16] = minnal_db::doc_store::hex::hex_to_bytes(&hex)?.try_into().ok()?;
    Some(u128::from_be_bytes(bytes))
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

    /// Canonical and plain 32-digit UUIDs parse to the same id; anything else is
    /// rejected. Stripping all hyphens and parsing a number accepted `1`, `1-2-3`
    /// and a leading `+`.
    #[test]
    fn uuid_ids_must_be_canonical_or_32_hex_digits() {
        let want = DocId::Uuid(0x550e8400_e29b_41d4_a716_446655440000);
        assert_eq!(parse_doc_id("550e8400-e29b-41d4-a716-446655440000", KeyType::Uuid).unwrap(), want);
        assert_eq!(parse_doc_id("550e8400e29b41d4a716446655440000", KeyType::Uuid).unwrap(), want);
        assert_eq!(parse_doc_id("550E8400-E29B-41D4-A716-446655440000", KeyType::Uuid).unwrap(), want);
        for bad in [
            "1",
            "1-2-3",
            "+50e8400e29b41d4a716446655440000",
            "550e8400-e29b41d4-a716-446655440000",
            "550e8400-e29b-41d4-a716-44665544000é",
            "",
        ] {
            assert!(parse_doc_id(bad, KeyType::Uuid).is_err(), "'{bad}' must be rejected");
        }
    }

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
