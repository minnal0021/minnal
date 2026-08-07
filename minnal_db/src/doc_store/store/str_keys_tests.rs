//! End-to-end coverage for `KeyType::Str` document stores.
//!
//! String keys are the one key type whose row IDs are **not** derived from the
//! key bytes: they are variable length, so `activate_indices` registers no
//! `RowIdFn` and the namespace falls through to the dense `RowMap`. That makes
//! two things worth pinning that the fixed-width key types get for free:
//!
//! 1. **Index queries must resolve hits back to the original key.** With a
//!    `RowToKeyFn` this is a pure function; here it is a lookup in a sidecar
//!    that is flushed at a checkpoint and rebuilt by WAL replay at open. So the
//!    tests query, restart, and query again.
//! 2. **Ordering must stay lexicographic**, since range and prefix scans over
//!    IDs depend on the stored key bytes ordering the way the IDs do.

use super::test_support::*;
use super::*;
use crate::doc_store::key::{MAX_STR_KEY_LEN, StrKey};

fn str_schema(namespace: &str, indices: Vec<IndexSpec>) -> DocStoreSchema {
    DocStoreSchema {
        store_type: StoreType::Doc,
        namespace: namespace.to_owned(),
        ns_id: None,
        key_type: KeyType::Str,
        attributes: vec![],
        indices,
        semantic_search_enabled: false,
        embedding_fields: vec![],
    }
}

fn sk(s: &str) -> DocId {
    DocId::Str(StrKey::new(s).unwrap())
}

fn status_index() -> Vec<IndexSpec> {
    vec![IndexSpec {
        field: "status".to_owned(),
        index_type: IndexType::Str,
    }]
}

/// Collect the string form of every ID in a result set.
fn ids_of<T>(results: &[(DocId, T)]) -> Vec<String> {
    results
        .iter()
        .map(|(id, _)| match id {
            DocId::Str(k) => k.as_str().to_owned(),
            other => panic!("expected a str id, got {other:?}"),
        })
        .collect()
}

#[tokio::test]
async fn put_get_delete_round_trips_a_string_key() {
    let db_dir = TempDir::new().unwrap();
    let schema_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path(), schema_dir.path()).await;
    store.create(str_schema("slugs", vec![])).await.unwrap();

    store
        .put("slugs", sk("acme-corp-2026"), serde_json::json!({"status": "active"}))
        .await
        .unwrap();

    let doc = store.get("slugs", sk("acme-corp-2026")).await.unwrap();
    assert_eq!(doc, Some(serde_json::json!({"status": "active"})));

    // A different key is a different document — no truncation or prefix
    // collapsing, which is what a fixed-width `RowIdFn` would have caused.
    assert_eq!(store.get("slugs", sk("acme-corp-2027")).await.unwrap(), None);

    store.delete("slugs", sk("acme-corp-2026")).await.unwrap();
    assert_eq!(store.get("slugs", sk("acme-corp-2026")).await.unwrap(), None);
}

/// Two keys sharing a long common prefix must stay distinct. Deriving a row ID
/// from the first 8 or 16 key bytes — what the U64/Uuid closures do — would map
/// these to the same row and corrupt the index.
#[tokio::test]
async fn keys_sharing_a_long_prefix_are_distinct_documents() {
    let db_dir = TempDir::new().unwrap();
    let schema_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path(), schema_dir.path()).await;
    store.create(str_schema("slugs", status_index())).await.unwrap();

    let a = "organisation-profile-alice";
    let b = "organisation-profile-bob";
    store.put("slugs", sk(a), serde_json::json!({"status": "active"})).await.unwrap();
    store.put("slugs", sk(b), serde_json::json!({"status": "archived"})).await.unwrap();

    assert_eq!(store.get("slugs", sk(a)).await.unwrap(), Some(serde_json::json!({"status": "active"})));
    assert_eq!(store.get("slugs", sk(b)).await.unwrap(), Some(serde_json::json!({"status": "archived"})));

    let active = store.query("slugs", "status = 'active'", Pagination::new(1, 100)).await.unwrap();
    assert_eq!(ids_of(&active.results), vec![a], "each key must own its own row in the index");
}

/// The reason string keys can use the dense `RowMap` at all: it answers the
/// reverse `row_id -> key` lookup, so an index hit resolves to the caller's
/// original key rather than to an integer they never supplied.
#[tokio::test]
async fn index_query_resolves_hits_back_to_string_keys() {
    let db_dir = TempDir::new().unwrap();
    let schema_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path(), schema_dir.path()).await;
    store.create(str_schema("slugs", status_index())).await.unwrap();

    for (slug, status) in [("alpha", "active"), ("beta", "archived"), ("gamma", "active")] {
        store.put("slugs", sk(slug), serde_json::json!({ "status": status })).await.unwrap();
    }

    let active = store.query("slugs", "status = 'active'", Pagination::new(1, 100)).await.unwrap();
    let mut ids = ids_of(&active.results);
    ids.sort();
    assert_eq!(ids, vec!["alpha", "gamma"]);
    assert_eq!(active.total, 2);
    assert!(active.degraded_fields.is_empty(), "a healthy index must not report degradation");
}

/// The row map is a derived structure: flushed at the index checkpoint and
/// rebuilt from the WAL at open. If it did not come back, index hits would
/// resolve to nothing (or to the wrong key) after a restart — and unlike the
/// key-derived `RowIdFn` types, string stores have no fallback that would hide
/// it. Also the case the `activate_field_index` scheme-mismatch tripwire
/// watches for: reopening a populated index with no `RowIdFn` registered.
#[tokio::test]
async fn string_keys_and_index_hits_survive_a_restart() {
    let db_dir = TempDir::new().unwrap();
    let schema_dir = TempDir::new().unwrap();

    {
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store.create(str_schema("slugs", status_index())).await.unwrap();
        for (slug, status) in [("alpha", "active"), ("beta", "archived"), ("gamma", "active")] {
            store.put("slugs", sk(slug), serde_json::json!({ "status": status })).await.unwrap();
        }
        store.shutdown().await.unwrap();
    }

    let store = open_fresh(db_dir.path(), schema_dir.path()).await;

    // Point reads still work…
    assert_eq!(
        store.get("slugs", sk("beta")).await.unwrap(),
        Some(serde_json::json!({"status": "archived"}))
    );

    // …and so does index resolution, which is the part that depends on the row
    // map having been rebuilt with the same IDs the persisted bitmaps hold.
    let active = store.query("slugs", "status = 'active'", Pagination::new(1, 100)).await.unwrap();
    let mut ids = ids_of(&active.results);
    ids.sort();
    assert_eq!(ids, vec!["alpha", "gamma"], "index hits must resolve to the same keys after a restart");
    assert!(active.degraded_fields.is_empty());

    // A write after the restart joins the same ID space.
    store.put("slugs", sk("delta"), serde_json::json!({"status": "active"})).await.unwrap();
    let active = store.query("slugs", "status = 'active'", Pagination::new(1, 100)).await.unwrap();
    let mut ids = ids_of(&active.results);
    ids.sort();
    assert_eq!(ids, vec!["alpha", "delta", "gamma"]);
}

/// Range scans work on string keys for the same reason they work on big-endian
/// integers: the stored bytes order the way the IDs do. `end` is exclusive.
#[tokio::test]
async fn range_scan_walks_string_keys_in_lexicographic_order() {
    let db_dir = TempDir::new().unwrap();
    let schema_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path(), schema_dir.path()).await;
    store.create(str_schema("slugs", vec![])).await.unwrap();

    for slug in ["aa", "acme", "acme-corp", "b", "z"] {
        store.put("slugs", sk(slug), serde_json::json!({"s": slug})).await.unwrap();
    }

    let all = store.scan_range("slugs", sk("a"), None, None, 100).await.unwrap();
    assert_eq!(
        ids_of(&all.results),
        vec!["aa", "acme", "acme-corp", "b", "z"],
        "a shorter key must not sort last just for being shorter"
    );

    let bounded = store.scan_range("slugs", sk("acme"), Some(sk("b")), None, 100).await.unwrap();
    assert_eq!(ids_of(&bounded.results), vec!["acme", "acme-corp"]);
}

#[tokio::test]
async fn range_scan_paginates_string_keys_by_cursor() {
    let db_dir = TempDir::new().unwrap();
    let schema_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path(), schema_dir.path()).await;
    store.create(str_schema("slugs", vec![])).await.unwrap();

    for slug in ["s-01", "s-02", "s-03", "s-04", "s-05"] {
        store.put("slugs", sk(slug), serde_json::json!({"s": slug})).await.unwrap();
    }

    let mut seen: Vec<String> = Vec::new();
    let mut cursor: Option<Vec<u8>> = None;
    loop {
        let page = store.scan_range("slugs", sk("s-"), None, cursor.clone(), 2).await.unwrap();
        seen.extend(ids_of(&page.results));
        match page.next_cursor {
            Some(next) => cursor = Some(next),
            None => break,
        }
    }
    assert_eq!(seen, vec!["s-01", "s-02", "s-03", "s-04", "s-05"]);
}

/// The prefix a caller supplies is raw key bytes, which for a string store is
/// just the string — this is what the REST layer forwards for `str` stores
/// instead of hex.
#[tokio::test]
async fn prefix_scan_matches_a_raw_string_prefix() {
    let db_dir = TempDir::new().unwrap();
    let schema_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path(), schema_dir.path()).await;
    store.create(str_schema("slugs", vec![])).await.unwrap();

    for slug in ["acme-1", "acme-2", "beta-1"] {
        store.put("slugs", sk(slug), serde_json::json!({"s": slug})).await.unwrap();
    }

    let page = store.scan_prefix("slugs", b"acme-".to_vec(), None, 100).await.unwrap();
    assert_eq!(ids_of(&page.results), vec!["acme-1", "acme-2"]);

    let all = store.scan_prefix("slugs", vec![], None, 100).await.unwrap();
    assert_eq!(all.results.len(), 3, "an empty prefix matches every key");
}

/// A key at the cap must be storable and readable — the boundary is where an
/// off-by-one in the inline buffer would show up.
#[tokio::test]
async fn a_key_at_the_length_cap_round_trips() {
    let db_dir = TempDir::new().unwrap();
    let schema_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path(), schema_dir.path()).await;
    store.create(str_schema("slugs", status_index())).await.unwrap();

    let max_key = "k".repeat(MAX_STR_KEY_LEN);
    store.put("slugs", sk(&max_key), serde_json::json!({"status": "active"})).await.unwrap();

    assert_eq!(
        store.get("slugs", sk(&max_key)).await.unwrap(),
        Some(serde_json::json!({"status": "active"}))
    );
    let active = store.query("slugs", "status = 'active'", Pagination::new(1, 100)).await.unwrap();
    assert_eq!(ids_of(&active.results), vec![max_key]);
}

/// Non-ASCII keys are stored as their UTF-8 bytes and must come back intact.
#[tokio::test]
async fn multibyte_keys_round_trip() {
    let db_dir = TempDir::new().unwrap();
    let schema_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path(), schema_dir.path()).await;
    store.create(str_schema("slugs", vec![])).await.unwrap();

    let key = "மின்னல்";
    store.put("slugs", sk(key), serde_json::json!({"n": 1})).await.unwrap();
    assert_eq!(store.get("slugs", sk(key)).await.unwrap(), Some(serde_json::json!({"n": 1})));

    let page = store.scan_prefix("slugs", vec![], None, 100).await.unwrap();
    assert_eq!(ids_of(&page.results), vec![key]);
}

/// KV stores share the same validated key type, so the cap holds there too —
/// and this is the path that already accepted unbounded keys before.
#[tokio::test]
async fn kv_string_keys_are_capped_at_the_same_length() {
    let db_dir = TempDir::new().unwrap();
    let schema_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path(), schema_dir.path()).await;
    store.create_kv(make_kv_schema("cache", KvKeyType::Str, KvValueType::Str)).await.unwrap();

    let ok = "k".repeat(MAX_STR_KEY_LEN);
    store.kv_put("cache", &serde_json::json!(ok), &serde_json::json!("v")).await.unwrap();
    assert_eq!(store.kv_get_by_str("cache", &ok).await.unwrap(), Some(serde_json::json!("v")));

    let over = "k".repeat(MAX_STR_KEY_LEN + 1);
    let err = store
        .kv_put("cache", &serde_json::json!(over), &serde_json::json!("v"))
        .await
        .unwrap_err();
    assert!(
        matches!(err, DocStoreError::Schema(SchemaError::StrKeyTooLong { .. })),
        "expected StrKeyTooLong, got {err:?}"
    );

    // The read side is capped by the same encoder, so an over-long key is
    // rejected rather than silently missing.
    let err = store.kv_get_by_str("cache", &over).await.unwrap_err();
    assert!(matches!(err, DocStoreError::Schema(SchemaError::StrKeyTooLong { .. })));

    let err = store.kv_put("cache", &serde_json::json!(""), &serde_json::json!("v")).await.unwrap_err();
    assert!(matches!(err, DocStoreError::Schema(SchemaError::EmptyStrKey)));
}

/// The vector sidecar namespaces key on raw document-ID bytes (`queue_key`,
/// `composite_key::encode`, the dense-vector namespace) rather than on a
/// fixed-width integer, so a variable-length string ID needs no new code there.
/// This pins that: it drives the semantic-search delete path, which is the one
/// that touches all three (`remove_queue_entry` + `delete_vector` + the doc
/// delete) with the ID bytes. No embedding service is needed — with no
/// `SemanticSearchContext` attached, `put` falls through to the plain write
/// while `delete` still takes the semantic path.
#[cfg(feature = "semantic-search")]
#[tokio::test]
async fn semantic_search_delete_path_handles_string_ids() {
    let db_dir = TempDir::new().unwrap();
    let schema_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path(), schema_dir.path()).await;

    let schema = DocStoreSchema {
        store_type: StoreType::Doc,
        namespace: "sem_slugs".to_owned(),
        ns_id: None,
        key_type: KeyType::Str,
        attributes: vec![crate::doc_store::schema::AttributeDef {
            name: "title".to_owned(),
            attr_type: AttributeType::Str,
            description: None,
        }],
        indices: vec![],
        semantic_search_enabled: true,
        embedding_fields: vec!["title".to_owned()],
    };
    store.create(schema).await.unwrap();

    // Long, prefix-sharing slugs: the case where a fixed-width ID derivation
    // would have collapsed two documents into one vector-index entry.
    let slugs = ["report-quarterly-alpha", "report-quarterly-beta", "report-quarterly-gamma"];
    for slug in slugs {
        store
            .put("sem_slugs", sk(slug), serde_json::json!({ "title": format!("title of {slug}") }))
            .await
            .unwrap();
    }

    store.delete("sem_slugs", sk("report-quarterly-beta")).await.unwrap();

    assert_eq!(store.get("sem_slugs", sk("report-quarterly-beta")).await.unwrap(), None);
    let after = store.scan_prefix("sem_slugs", vec![], None, 100).await.unwrap();
    assert_eq!(
        ids_of(&after.results),
        vec!["report-quarterly-alpha", "report-quarterly-gamma"],
        "only the deleted slug is removed — the sibling keys sharing its prefix survive"
    );
}
