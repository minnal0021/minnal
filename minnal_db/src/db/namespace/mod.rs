//! Namespaces: what exists and how it is created and removed. `registry` is the
//! saved list of namespaces with their schemas and field types, and `ops` holds
//! the `Database` methods that create, look up and remove a namespace.

mod ops;
mod registry;

pub use registry::*;
