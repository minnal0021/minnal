//! Namespaces: the saved list of what exists. `id` is a namespace's identity (a
//! random UUID); `registry` holds their names, ids,
//! schemas and field types. The `Database` methods that create, look up and
//! remove a namespace are in `db::namespace`.

mod id;
mod registry;

pub use id::{BadNamespaceId, NamespaceId};
pub use registry::*;
