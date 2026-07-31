//! Hex encoding/decoding helpers.
//!
//! Re-exported from [`crate::support::hex`], which is where they now live — the
//! base engine needs them for the field-index gap record, and `doc_store` is
//! feature-gated. This path is kept because it is part of the published API
//! surface (`minnal_db::doc_store::hex`).

pub use crate::support::hex::{bytes_to_hex, hex_to_bytes};
