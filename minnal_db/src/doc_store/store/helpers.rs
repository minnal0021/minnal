//! Shared helpers used across the `DocStore` impl modules.

use std::sync::Arc;

use crate::doc_store::error::DocStoreError;
use crate::doc_store::schema::{DocStoreSchema, IndexType, KeyType};
use crate::{AsyncDb, ExtractorFn, IndexValue, IndexValueType, RowIdFn, RowToKeyFn};

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Convert an [`IndexType`] to the equivalent [`AttributeType`].
pub(super) fn index_type_to_attr_type(t: IndexType) -> crate::doc_store::schema::AttributeType {
    match t {
        IndexType::Bool => crate::doc_store::schema::AttributeType::Bool,
        IndexType::Int => crate::doc_store::schema::AttributeType::Int,
        IndexType::Str => crate::doc_store::schema::AttributeType::Str,
    }
}

/// Convert an [`IndexType`] to the minnal_db [`IndexValueType`].
pub(super) fn to_ivt(t: IndexType) -> IndexValueType {
    match t {
        IndexType::Bool => IndexValueType::Bool,
        IndexType::Int => IndexValueType::Int,
        IndexType::Str => IndexValueType::Str,
    }
}

/// Build a JSON-field extractor closure for a given field name and type.
pub(super) fn json_extractor(field: String, index_type: IndexType) -> ExtractorFn {
    Arc::new(move |bytes: &[u8]| {
        let v: serde_json::Value = serde_json::from_slice(bytes).ok()?;
        match index_type {
            IndexType::Bool => v.get(&field)?.as_bool().map(IndexValue::Bool),
            IndexType::Int => v.get(&field)?.as_i64().map(IndexValue::Int),
            IndexType::Str => v.get(&field)?.as_str().map(|s| IndexValue::Str(s.to_string())),
        }
    })
}

/// Concatenate the specified embedding fields from a document into a single
/// text string for embedding.  Each field contributes `"field_name: value\n"`.
/// Fields that are absent or not strings are silently skipped.
#[cfg(feature = "semantic-search")]
pub(super) fn build_embedding_text(doc: &serde_json::Value, fields: &[String]) -> String {
    fields
        .iter()
        .filter_map(|f| doc.get(f)?.as_str().map(|v| format!("{f}: {v}")))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Activate all field indices defined in `schema` for namespace `ns_id`.
///
/// Also registers a [`RowIdFn`] + [`RowToKeyFn`] pair for the **fixed-width**
/// key types (`U64`, `U128`, `Uuid`) so that index queries resolve matching
/// keys in O(|hits|) rather than the O(n_keys) fallback scan. `Str` keys
/// deliberately register nothing — see below.
///
/// Called once on open and once after creating a new store.
pub(super) async fn activate_indices(db: &AsyncDb, ns_id: u32, schema: &DocStoreSchema) -> Result<(), DocStoreError> {
    // Register the row-ID functions FIRST — before activating any field index.
    //
    // `activate_field_index` replays the WAL tail and resolves each affected
    // key's row ID through the store's *current* resolver. The resolver
    // precedence is RowIdFn > dense RowMap > legacy hash, so if the custom
    // key-derived RowIdFn is not yet installed, replay would fall back to the
    // RowMap and index those WAL-tail keys under dense IDs that disagree with
    // the key-derived IDs used by prior persisted entries and all future writes
    // — mixing two row-ID schemes in one field index (stale hits, wrong key
    // resolution). The engine documents this ordering requirement on
    // `Database::set_row_id_fn`; honour it here.
    //
    // The functions exist for key types whose raw bytes are injective into u128,
    // which also gives O(|hits|) key resolution in index queries.
    //
    // `Str` keys are variable length and can be up to MAX_STR_KEY_LEN bytes, so
    // they are NOT injective into a u128 and get no function pair: they fall
    // through to the dense `RowMap`, which compares full key bytes (so distinct
    // keys never share a row ID) and answers the reverse `row_id -> key` lookup
    // in O(1) from its id array. Deriving an ID from the key bytes here instead
    // would be actively unsafe — the fixed-width closures below slice `k[..8]` /
    // `k[..16]`, which panics on a shorter key and silently truncates a longer
    // one into a colliding row ID.
    let fns: Option<(RowIdFn, RowToKeyFn)> = match schema.key_type {
        KeyType::U64 => Some((
            std::sync::Arc::new(|k: &[u8]| {
                let arr: [u8; 8] = k[..8].try_into().unwrap_or_default();
                u64::from_be_bytes(arr) as u128
            }),
            std::sync::Arc::new(|id: u128| (id as u64).to_be_bytes().to_vec()),
        )),
        KeyType::U128 | KeyType::Uuid => Some((
            std::sync::Arc::new(|k: &[u8]| {
                let arr: [u8; 16] = k[..16].try_into().unwrap_or_default();
                u128::from_be_bytes(arr)
            }),
            std::sync::Arc::new(|id: u128| id.to_be_bytes().to_vec()),
        )),
        KeyType::Str => None,
    };
    if let Some((row_id_fn, row_to_key_fn)) = fns {
        db.set_row_id_fn(ns_id, row_id_fn, Some(row_to_key_fn)).await.map_err(DocStoreError::Db)?;
    }

    for spec in &schema.indices {
        let ivt = to_ivt(spec.index_type);
        // register_index_field is idempotent — returns existing field_id on restart
        let field_id = db.register_index_field(ns_id, &spec.field, ivt)?;
        let extractor = json_extractor(spec.field.clone(), spec.index_type);
        db.activate_field_index(ns_id, field_id, ivt, extractor).await?;
    }

    Ok(())
}
