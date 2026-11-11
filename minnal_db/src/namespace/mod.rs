//! Namespaces: the saved list of what exists. `id` is a namespace's identity (a
//! 64-bit id from the catalog's counter); `registry` holds their names, ids, TTL
//! settings and field indexes in memory; `schema` is one namespace's field
//! indexes; `catalog` is how all of it is kept in the `system_catalog`
//! namespace. The `Database` methods that create, look up and remove a
//! namespace, and that open and write the catalog, are in `db::namespace`.

pub(crate) mod catalog;
mod id;
mod registry;
mod schema;

pub use catalog::CATALOG_NAMESPACE;
pub use id::{BadNamespaceId, NamespaceId};
pub use registry::*;
pub use schema::NamespaceSchema;
