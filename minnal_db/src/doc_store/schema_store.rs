//! Store schemas: persisted in the `system_schema` namespace, served from memory.
//!
//! Every schema the doc store reads comes from `SchemaStore`, and nothing else
//! touches `system_schema`. The key is the store's name and the value its schema
//! as JSON (the same JSON export and import use).
//!
//! **Writes are write-through and serialised.** Each write takes `writer`,
//! persists the schema (a WAL-backed put or delete, durable on return), then
//! swaps the cached copy under the map's write lock. So the cache always holds
//! what is on disk, in the order it was written, and a read-modify-write
//! (`SchemaStore::update_doc`, `SchemaStore::update_kv`) cannot lose a
//! concurrent update. Readers clone an `Arc` under the read lock and never wait
//! for an fsync. `writer` is never held across an `.await`.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::{Mutex, RwLock};

use crate::doc_store::error::{DocStoreError, SchemaError};
use crate::doc_store::schema::{DocStoreSchema, KvStoreSchema, StoreType, peek_store_type};
#[cfg(feature = "semantic-search")]
use crate::doc_store::vector_settings::VectorIndexSettings;
use crate::{AsyncDb, AsyncNamespace};

/// The system namespace that holds every store's schema.
pub const SCHEMA_NAMESPACE: &str = "system_schema";

/// One store's schema, of either kind.
#[derive(Debug, Clone)]
pub enum StoredSchema {
    /// A document store.
    Doc(Arc<DocStoreSchema>),
    /// A KV store.
    Kv(Arc<KvStoreSchema>),
}

impl StoredSchema {
    /// The store's name.
    pub fn namespace(&self) -> &str {
        match self {
            Self::Doc(s) => &s.namespace,
            Self::Kv(s) => &s.namespace,
        }
    }

    /// Which kind of store it is.
    pub fn store_type(&self) -> StoreType {
        match self {
            Self::Doc(_) => StoreType::Doc,
            Self::Kv(_) => StoreType::Kv,
        }
    }

    /// The store's namespace id, once it has one.
    pub fn ns_id(&self) -> Option<u32> {
        match self {
            Self::Doc(s) => s.ns_id,
            Self::Kv(s) => s.ns_id,
        }
    }

    /// Whether semantic search is on for the store.
    pub fn is_semantic_search_enabled(&self) -> bool {
        match self {
            Self::Doc(s) => s.is_semantic_search_enabled(),
            Self::Kv(s) => s.is_semantic_search_enabled(),
        }
    }

    #[cfg(feature = "semantic-search")]
    fn vector_settings(&self) -> Result<VectorIndexSettings, SchemaError> {
        match self {
            Self::Doc(s) => s.vector_settings(),
            Self::Kv(s) => s.vector_settings(),
        }
    }

    fn to_json(&self) -> Result<Vec<u8>, SchemaError> {
        Ok(match self {
            Self::Doc(s) => serde_json::to_vec_pretty(&**s)?,
            Self::Kv(s) => serde_json::to_vec_pretty(&**s)?,
        })
    }

    /// Parse a stored schema; `store_type` says which kind it is.
    fn from_json(bytes: &[u8]) -> Result<Self, SchemaError> {
        let json = String::from_utf8_lossy(bytes);
        Ok(match peek_store_type(&json) {
            Some(StoreType::Kv) => Self::Kv(Arc::new(serde_json::from_str(&json)?)),
            _ => Self::Doc(Arc::new(serde_json::from_str(&json)?)),
        })
    }
}

/// A cached schema and its vector-index settings, resolved once when cached so
/// a search does not resolve them again (`None` when they do not resolve; the
/// reader then resolves them to get the error).
#[derive(Debug)]
struct Entry {
    schema: StoredSchema,
    #[cfg(feature = "semantic-search")]
    vector: Option<VectorIndexSettings>,
}

impl Entry {
    fn new(schema: StoredSchema) -> Arc<Self> {
        Arc::new(Self {
            #[cfg(feature = "semantic-search")]
            vector: schema.vector_settings().ok(),
            schema,
        })
    }
}

/// Every store's schema, persisted in [`SCHEMA_NAMESPACE`] and cached in memory.
pub(crate) struct SchemaStore {
    ns: AsyncNamespace,
    map: RwLock<HashMap<String, Arc<Entry>>>,
    /// Held by every write across its persist and its cache swap.
    writer: Mutex<()>,
}

impl SchemaStore {
    /// Open the schema namespace (creating it in a new database) and load every
    /// schema it holds. Call once the database is open, before anything reads a
    /// schema.
    pub(crate) async fn open(db: &AsyncDb) -> Result<Self, DocStoreError> {
        let ns = db.namespace(SCHEMA_NAMESPACE.to_owned()).await?;
        let mut map = HashMap::new();
        for (key, value) in ns.scan_prefix(Vec::new()).await? {
            let name = String::from_utf8_lossy(&key).into_owned();
            let schema = StoredSchema::from_json(&value).map_err(|e| {
                DocStoreError::Schema(SchemaError::Corrupt {
                    namespace: name.clone(),
                    reason: e.to_string(),
                })
            })?;
            map.insert(name, Entry::new(schema));
        }
        Ok(Self {
            ns,
            map: RwLock::new(map),
            writer: Mutex::new(()),
        })
    }

    fn entry(&self, namespace: &str) -> Option<Arc<Entry>> {
        self.map.read().get(namespace).cloned()
    }

    fn not_found(namespace: &str) -> DocStoreError {
        DocStoreError::NotFound {
            namespace: namespace.to_owned(),
        }
    }

    /// A store's schema, of whichever kind.
    pub(crate) fn get(&self, namespace: &str) -> Option<StoredSchema> {
        self.entry(namespace).map(|e| e.schema.clone())
    }

    /// A document store's schema. `NotFound` if there is none, `WrongStoreType`
    /// for a KV store.
    pub(crate) fn doc(&self, namespace: &str) -> Result<Arc<DocStoreSchema>, DocStoreError> {
        match self.get(namespace) {
            Some(StoredSchema::Doc(s)) => Ok(s),
            Some(StoredSchema::Kv(_)) => Err(DocStoreError::Schema(SchemaError::WrongStoreType {
                namespace: namespace.to_owned(),
                expected: "doc",
                found: "kv",
            })),
            None => Err(Self::not_found(namespace)),
        }
    }

    /// A KV store's schema. `NotFound` if there is none, `WrongStoreType` for a
    /// document store.
    pub(crate) fn kv(&self, namespace: &str) -> Result<Arc<KvStoreSchema>, DocStoreError> {
        match self.get(namespace) {
            Some(StoredSchema::Kv(s)) => Ok(s),
            Some(StoredSchema::Doc(_)) => Err(DocStoreError::Schema(SchemaError::WrongStoreType {
                namespace: namespace.to_owned(),
                expected: "kv",
                found: "doc",
            })),
            None => Err(Self::not_found(namespace)),
        }
    }

    /// Whether a store of this name exists.
    pub(crate) fn contains(&self, namespace: &str) -> bool {
        self.map.read().contains_key(namespace)
    }

    /// Every schema, sorted by store name.
    pub(crate) fn all(&self) -> Vec<StoredSchema> {
        let mut all: Vec<StoredSchema> = self.map.read().values().map(|e| e.schema.clone()).collect();
        all.sort_by(|a, b| a.namespace().cmp(b.namespace()));
        all
    }

    /// Every document-store schema, sorted by store name.
    pub(crate) fn docs(&self) -> Vec<Arc<DocStoreSchema>> {
        self.all()
            .into_iter()
            .filter_map(|s| match s {
                StoredSchema::Doc(s) => Some(s),
                StoredSchema::Kv(_) => None,
            })
            .collect()
    }

    /// Every KV-store schema, sorted by store name.
    pub(crate) fn kvs(&self) -> Vec<Arc<KvStoreSchema>> {
        self.all()
            .into_iter()
            .filter_map(|s| match s {
                StoredSchema::Kv(s) => Some(s),
                StoredSchema::Doc(_) => None,
            })
            .collect()
    }

    /// A store's resolved vector-index settings and namespace id, whichever kind
    /// of store it is.
    #[cfg(feature = "semantic-search")]
    pub(crate) fn vector_settings(&self, namespace: &str) -> Result<(VectorIndexSettings, u32), DocStoreError> {
        let entry = self.entry(namespace).ok_or_else(|| {
            DocStoreError::Schema(SchemaError::NotFound {
                namespace: namespace.to_owned(),
            })
        })?;
        let settings = match &entry.vector {
            Some(v) => v.clone(),
            None => entry.schema.vector_settings()?,
        };
        let ns_id = entry.schema.ns_id().ok_or_else(|| DocStoreError::MissingNsId {
            namespace: namespace.to_owned(),
        })?;
        Ok((settings, ns_id))
    }

    /// `Err(SemanticSearchNotEnabled)` unless the store (document or KV) has
    /// semantic search on; `Err(NotFound)` for an unknown store.
    #[cfg(feature = "semantic-search")]
    pub(crate) fn require_semantic_search(&self, namespace: &str) -> Result<(), DocStoreError> {
        let schema = self.get(namespace).ok_or_else(|| Self::not_found(namespace))?;
        if schema.is_semantic_search_enabled() {
            Ok(())
        } else {
            Err(DocStoreError::SemanticSearchNotEnabled {
                namespace: namespace.to_owned(),
            })
        }
    }

    /// Persist `schema` under its name and cache it. Caller holds `writer`.
    fn persist(&self, schema: StoredSchema) -> Result<(), DocStoreError> {
        let name = schema.namespace().to_owned();
        let json = schema.to_json()?;
        crate::support::run_blocking_pass(|| self.ns.put_blocking(name.as_bytes(), &json))?;
        self.map.write().insert(name, Entry::new(schema));
        Ok(())
    }

    /// Save a new store's schema. `AlreadyExists` if the name is taken.
    pub(crate) fn insert(&self, schema: StoredSchema) -> Result<(), DocStoreError> {
        let _w = self.writer.lock();
        if self.contains(schema.namespace()) {
            return Err(DocStoreError::AlreadyExists {
                namespace: schema.namespace().to_owned(),
            });
        }
        self.persist(schema)
    }

    /// Change a document store's schema with `f`, atomically with respect to
    /// every other schema write. Nothing is saved if `f` fails.
    pub(crate) fn update_doc<R>(&self, namespace: &str, f: impl FnOnce(&mut DocStoreSchema) -> Result<R, DocStoreError>) -> Result<R, DocStoreError> {
        let _w = self.writer.lock();
        let current = self.doc(namespace)?;
        let mut schema = DocStoreSchema::clone(&current);
        let out = f(&mut schema)?;
        self.persist(StoredSchema::Doc(Arc::new(schema)))?;
        Ok(out)
    }

    /// Change a KV store's schema with `f`; see [`update_doc`](Self::update_doc).
    pub(crate) fn update_kv<R>(&self, namespace: &str, f: impl FnOnce(&mut KvStoreSchema) -> Result<R, DocStoreError>) -> Result<R, DocStoreError> {
        let _w = self.writer.lock();
        let current = self.kv(namespace)?;
        let mut schema = KvStoreSchema::clone(&current);
        let out = f(&mut schema)?;
        self.persist(StoredSchema::Kv(Arc::new(schema)))?;
        Ok(out)
    }

    /// Delete a store's schema (a WAL-backed delete) and drop it from the cache.
    /// Removing a schema that is not there does nothing.
    pub(crate) fn remove(&self, namespace: &str) -> Result<(), DocStoreError> {
        let _w = self.writer.lock();
        crate::support::run_blocking_pass(|| self.ns.delete_blocking(namespace.as_bytes()))?;
        self.map.write().remove(namespace);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doc_store::schema::{KeyType, KvKeyType, KvValueType};

    fn doc(name: &str) -> StoredSchema {
        StoredSchema::Doc(Arc::new(DocStoreSchema {
            store_type: StoreType::Doc,
            namespace: name.to_owned(),
            ns_id: Some(7),
            key_type: KeyType::U64,
            indices: vec![],
            attributes: vec![],
            embedding_fields: vec![],
            semantic_search_enabled: false,
            vector_index: None,
        }))
    }

    fn kv(name: &str) -> StoredSchema {
        StoredSchema::Kv(Arc::new(KvStoreSchema {
            store_type: StoreType::Kv,
            namespace: name.to_owned(),
            ns_id: Some(8),
            key_type: KvKeyType::Str,
            value_type: KvValueType::Str,
            semantic_search_enabled: false,
            vector_index: None,
        }))
    }

    async fn open_db(dir: &std::path::Path) -> AsyncDb {
        AsyncDb::open_with_config(dir.to_path_buf(), crate::doc_store::test_db_config())
            .await
            .unwrap()
    }

    /// A write is durable when it returns: a crash right after an insert, an
    /// update or a removal keeps it.
    #[tokio::test(flavor = "multi_thread")]
    async fn every_write_survives_a_crash_right_after_it() {
        let dir = tempfile::tempdir().unwrap();
        {
            let db = open_db(dir.path()).await;
            let schemas = SchemaStore::open(&db).await.unwrap();
            schemas.insert(doc("docs")).unwrap();
            schemas.insert(kv("tokens")).unwrap();
            schemas.insert(doc("gone")).unwrap();
            schemas
                .update_doc("docs", |s| {
                    s.embedding_fields.push("text".into());
                    Ok(())
                })
                .unwrap();
            schemas.remove("gone").unwrap();
            drop(schemas);
            db.crash().await;
        }
        let db = open_db(dir.path()).await;
        let schemas = SchemaStore::open(&db).await.unwrap();
        assert_eq!(schemas.doc("docs").unwrap().embedding_fields, vec!["text".to_owned()]);
        assert_eq!(schemas.kv("tokens").unwrap().value_type, KvValueType::Str);
        assert!(!schemas.contains("gone"));
        assert_eq!(schemas.all().iter().map(StoredSchema::namespace).collect::<Vec<_>>(), ["docs", "tokens"]);
        db.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn reads_check_the_kind_and_a_failed_update_saves_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let db = open_db(dir.path()).await;
        let schemas = SchemaStore::open(&db).await.unwrap();
        schemas.insert(doc("docs")).unwrap();
        schemas.insert(kv("tokens")).unwrap();
        assert!(matches!(schemas.insert(kv("docs")), Err(DocStoreError::AlreadyExists { .. })));
        assert!(matches!(
            schemas.kv("docs"),
            Err(DocStoreError::Schema(SchemaError::WrongStoreType {
                expected: "kv",
                found: "doc",
                ..
            }))
        ));
        assert!(matches!(
            schemas.doc("tokens"),
            Err(DocStoreError::Schema(SchemaError::WrongStoreType {
                expected: "doc",
                found: "kv",
                ..
            }))
        ));
        assert!(matches!(schemas.doc("nope"), Err(DocStoreError::NotFound { .. })));

        let r: Result<(), _> = schemas.update_doc("docs", |s| {
            s.embedding_fields.push("text".into());
            Err(DocStoreError::NotFound { namespace: "x".into() })
        });
        assert!(r.is_err());
        assert!(schemas.doc("docs").unwrap().embedding_fields.is_empty(), "cache unchanged");
        let fresh = SchemaStore::open(&db).await.unwrap();
        assert!(fresh.doc("docs").unwrap().embedding_fields.is_empty(), "nothing persisted");
        db.shutdown().await.unwrap();
    }

    /// A stored schema that does not parse fails the open, naming the store,
    /// rather than the store silently disappearing.
    #[tokio::test]
    async fn an_unreadable_stored_schema_fails_the_open() {
        let dir = tempfile::tempdir().unwrap();
        let db = open_db(dir.path()).await;
        let ns = db.namespace(SCHEMA_NAMESPACE.to_owned()).await.unwrap();
        ns.put(b"broken".to_vec(), br#"{"store_type":"doc","namespace":"broken","bogus":1}"#.to_vec())
            .await
            .unwrap();
        match SchemaStore::open(&db).await {
            Err(DocStoreError::Schema(SchemaError::Corrupt { namespace, .. })) => assert_eq!(namespace, "broken"),
            Err(e) => panic!("expected Corrupt, got {e:?}"),
            Ok(_) => panic!("expected Corrupt, got Ok"),
        }
        db.shutdown().await.unwrap();
    }
}
