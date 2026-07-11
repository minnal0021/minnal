use minnal_db::{DocId, DocStoreError, KeyType};

use crate::error::AppError;

/// Serialize a [`DocId`] to a JSON-friendly value.
///
/// - `Uuid` → hyphenated UUID string
/// - `U64`  → JSON number
/// - `U128` → decimal string (too large for JSON number)
pub fn doc_id_to_value(id: DocId) -> serde_json::Value {
    match id {
        DocId::Uuid(v) => serde_json::Value::String(format_uuid(v)),
        DocId::U64(v) => serde_json::json!(v),
        DocId::U128(v) => serde_json::Value::String(v.to_string()),
    }
}

/// Parse a URL path segment into a [`DocId`] using the store's [`KeyType`].
///
/// - `Uuid`  → expects `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`
/// - `U64`   → decimal integer
/// - `U128`  → decimal integer
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
    }
}

fn format_uuid(v: u128) -> String {
    let b = v.to_be_bytes();
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15],
    )
}
