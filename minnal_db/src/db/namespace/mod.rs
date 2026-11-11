//! The `Database` methods that create, look up and remove a namespace, and
//! that open and write the system catalog. They need the database's internal
//! fields, so they sit here rather than with the registry in `crate::namespace`.

mod catalog;
mod namespace_ops;
