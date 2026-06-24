Run clippy across the full workspace, then fix every warning.

Steps:
1. Run: `cargo clippy --all-targets -- -D warnings 2>&1`
2. If there are no warnings, report "No clippy warnings." and stop.
3. For each warning: identify the file and line, understand the lint, apply the fix using Edit.
4. After all fixes, run clippy again to confirm zero warnings.
5. Report a summary: how many warnings were fixed and which files were touched.

Do not suppress warnings with `#[allow(...)]` unless the lint is a known false positive — fix the code instead.
