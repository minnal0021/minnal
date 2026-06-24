Orient on the crate named in $ARGUMENTS before starting any work on it.

Valid crates: minnal_db, index, semantic_search, minnal_doc_store, minnal_doc_store_api, tools

Steps:
1. Read `$ARGUMENTS/CLAUDE.md` if it exists.
2. Read `$ARGUMENTS/src/lib.rs` (or `src/main.rs` for binary-only crates like tools).
3. Read `$ARGUMENTS/Cargo.toml` for the dependency list.
4. Confirm you are oriented: summarise in 3-5 bullet points what the crate does, its key public types, and which other workspace crates it depends on.

If $ARGUMENTS is empty or not a valid crate name, list the valid crates above and ask which one to orient on.
