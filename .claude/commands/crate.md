Orient on the crate (or folded module) named in $ARGUMENTS before starting any work on it.

Valid crates: minnal_db, minnal_db_api, tools

`minnal_db` is a single feature-gated crate that folds the former `index`,
`semantic_search`, and `minnal_doc_store` crates in as modules under
`minnal_db/src/{index,semantic_search,doc_store}` (plus the top-level
`vector_kv` bridge). To orient on one of those layers, pass e.g.
`minnal_db/src/semantic_search`.

Steps:
1. Read `$ARGUMENTS/CLAUDE.md` if it exists (present at the crate root and in the folded `index`/`semantic_search` module dirs).
2. Read `$ARGUMENTS/src/lib.rs` (crate root), or `$ARGUMENTS/mod.rs` for a folded module dir, or `src/main.rs` for binary-only crates like tools.
3. Read the nearest `Cargo.toml` for the dependency list and, for `minnal_db`, the `[features]` block (`kv-store` default, `doc-store`, `semantic-search`).
4. Confirm you are oriented: summarise in 3-5 bullet points what the crate/module does, its key public types, and which features gate it.

If $ARGUMENTS is empty or not a valid crate name, list the valid crates above and ask which one to orient on.
