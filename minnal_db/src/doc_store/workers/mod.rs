//! `DocStore`'s background workers. They need its schemas and in-flight write
//! tracking, and only exist with `semantic-search`. The engine's workers are in
//! `db::workers`, whose module docs list them all.

pub(crate) mod ivf_maintenance;
pub(crate) mod vec_index;
