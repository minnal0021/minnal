//! The vector index as stores describe it. `vector_settings` holds each store's
//! settings: a store's schema carries them with or without semantic search
//! compiled, so they sit outside `semantic_search`, which runs on them.

pub mod vector_settings;
