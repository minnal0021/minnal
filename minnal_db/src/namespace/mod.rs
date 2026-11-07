//! Namespaces: the saved list of what exists. `registry` holds their names, ids,
//! schemas and field types. The `Database` methods that create, look up and
//! remove a namespace are in `db::namespace`.

mod registry;

pub use registry::*;
