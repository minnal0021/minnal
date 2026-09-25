//! Document chunking for embedding.
//!
//! Chunking used to live in the external embedding service; it now lives here so
//! minnal sends the service pre-chunked strings (one embedding is returned per
//! string). The service no longer splits text.
//!
//! Only **documents** are chunked: [`chunk_document`] splits the text on
//! **sentence** boundaries ([`split_sentences`]) and groups the sentences into
//! overlapping **sliding windows** ([`sliding_windows`]). Each window becomes one
//! Pass-1 (SingleBit, ColBERT MaxSim) chunk embedding. Each window concatenates
//! `window_size` consecutive sentences, and the window start advances by
//! `sliding_size` sentences between chunks. The final window keeps whatever
//! units remain.
//!
//! ```text
//! units = [a, b, c, d, e],  window_size = 2,  sliding_size = 1
//!   → "a b", "b c", "c d", "d e"
//!
//! units = [a, b, c, d, e],  window_size = 2,  sliding_size = 2
//!   → "a b", "c d", "e"          (last window = remainder)
//! ```
//!
//! **Queries are not chunked.** A query is embedded once, whole, and that one
//! vector serves both passes (see
//! [`embed_query`](crate::semantic_search::service::embed_query)). Earlier
//! versions split queries into 4-word windows; a BEIR evaluation found the
//! whole-query vector never worse, often better, and much cheaper
//! (`semantic_search/query-embedding-report.md`).
//!
//! `window_size` and `sliding_size` are the knobs of
//! [`crate::semantic_search::service::SemanticSearchConfig`] / the TOML
//! `[semantic_search]` section. They are effectively an on-disk decision:
//! changing them requires re-indexing the corpus, since stored chunk vectors
//! keep the old chunking.
//!
//! The sentence splitter is a deterministic, intentionally dependency-free
//! heuristic (a terminator `.`/`!`/`?` followed by whitespace ends a sentence).
//! It mis-handles abbreviations, spaced initials, and list markers; the previous
//! ML-based splitter is not reproducible without a new dependency. See
//! [`split_sentences`] for the enumerated limitations and the characterisation
//! tests that pin each one.

/// Chunk a document: sentence-split, then group into sliding windows.
///
/// Returns an empty vector when `text` contains no sentences. `window_size` and
/// `sliding_size` are clamped to a minimum of 1; a `sliding_size` of 0 would
/// otherwise never advance.
pub fn chunk_document(text: &str, window_size: usize, sliding_size: usize) -> Vec<String> {
    sliding_windows(&split_sentences(text), window_size, sliding_size)
}

/// Split `text` into sentences using a deterministic heuristic.
///
/// A run of one or more terminators (`.`, `!`, `?`) that is immediately followed
/// by whitespace (or the end of the text) ends a sentence; the terminators stay
/// with the sentence and the intervening whitespace is dropped. Each sentence is
/// trimmed and empty results are discarded. This mirrors a `split on (?<=[.!?])\s+`
/// regex.
///
/// # Known limitations (intentional, dependency-free heuristic)
///
/// The rule is purely "terminator + whitespace", with no lexical knowledge, so it
/// mis-handles several common constructs. The behaviour on each is pinned by the
/// `split_sentences_*` characterisation tests; the trade-off is accepting these in
/// exchange for being dependency-free (the previous ML splitter is not reproducible
/// without a new dependency). Because chunking is an on-disk decision, changing the
/// splitter requires a full re-index — see `semantic_search/CLAUDE.md`.
///
/// - **Abbreviations** (`U.S.`, `Inc.`, `etc.`): a dot *inside* the abbreviation is
///   followed by a letter, so it is correctly not a boundary — but the abbreviation's
///   *trailing* dot before a space is mis-read as a sentence end, so `"The U.S. economy"`
///   splits after `"U.S."`.
/// - **Spaced initials** (`J. R. R. Tolkien`): each `X.` is dot+space, so every
///   initial is split into its own "sentence".
/// - **Numbered/bulleted lists** (`1.`, `2.`): a list marker is digit+dot+space and
///   reads as a sentence end, so list items fragment.
/// - **Decimals / versions** (`3.14`, `2.0`): a dot between digits is not followed by
///   whitespace, so these are *correctly* kept intact.
/// - **Ellipses** (`...`): a run of terminators before whitespace is one boundary, so
///   an ellipsis ends a sentence.
/// - **No maximum length**: a long run with no terminator stays a single unit (and
///   therefore a single windowed chunk).
pub fn split_sentences(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut sentences = Vec::new();
    let mut start = 0;
    let mut i = 0;

    let is_terminator = |c: char| c == '.' || c == '!' || c == '?';

    while i < n {
        if is_terminator(chars[i]) {
            // Consume a run of consecutive terminators (e.g. "?!").
            let mut end = i;
            while end < n && is_terminator(chars[end]) {
                end += 1;
            }
            // A boundary only exists if the run is followed by whitespace or EOF.
            if end >= n || chars[end].is_whitespace() {
                push_trimmed(&chars[start..end], &mut sentences);
                // Skip the whitespace separating this sentence from the next.
                let mut next = end;
                while next < n && chars[next].is_whitespace() {
                    next += 1;
                }
                start = next;
                i = next;
                continue;
            }
            // Mid-word terminator (e.g. "3.14") — not a boundary.
            i = end;
            continue;
        }
        i += 1;
    }

    // Trailing text with no terminator forms a final sentence.
    if start < n {
        push_trimmed(&chars[start..n], &mut sentences);
    }

    sentences
}

/// Group `units` into overlapping windows and join each window with a single space.
///
/// Each window starts `sliding_size` units after the previous one and spans up
/// to `window_size` units; the last window holds whatever remains. Returns an
/// empty vector when `units` is empty. `window_size` and `sliding_size` are
/// clamped to a minimum of 1.
pub fn sliding_windows(units: &[String], window_size: usize, sliding_size: usize) -> Vec<String> {
    let window_size = window_size.max(1);
    let sliding_size = sliding_size.max(1);

    let mut windows = Vec::new();
    let mut i = 0;
    while i < units.len() {
        let end = (i + window_size).min(units.len());
        windows.push(units[i..end].join(" "));
        // Stop once a window reaches the end so we don't emit trailing subsets.
        if end >= units.len() {
            break;
        }
        i += sliding_size;
    }
    windows
}

/// Push `chars` as a trimmed, non-empty sentence onto `out`.
fn push_trimmed(chars: &[char], out: &mut Vec<String>) {
    let s: String = chars.iter().collect();
    let trimmed = s.trim();
    if !trimmed.is_empty() {
        out.push(trimmed.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── split_sentences ───────────────────────────────────────────────────────

    #[test]
    fn split_sentences_basic() {
        assert_eq!(
            split_sentences("First sentence. Second one! Third? Done."),
            vec!["First sentence.", "Second one!", "Third?", "Done."],
        );
    }

    #[test]
    fn split_sentences_trailing_without_terminator() {
        assert_eq!(
            split_sentences("One sentence. And a trailing fragment"),
            vec!["One sentence.", "And a trailing fragment"],
        );
    }

    #[test]
    fn split_sentences_consecutive_terminators_are_one_boundary() {
        assert_eq!(split_sentences("Really?! Yes."), vec!["Really?!", "Yes."]);
    }

    #[test]
    fn split_sentences_mid_token_terminator_is_not_a_boundary() {
        // No whitespace after the dot in "3.14", so it stays in one sentence.
        assert_eq!(split_sentences("Pi is 3.14 today."), vec!["Pi is 3.14 today."]);
    }

    #[test]
    fn split_sentences_collapses_whitespace_between_sentences() {
        assert_eq!(split_sentences("A.   B."), vec!["A.", "B."]);
    }

    #[test]
    fn split_sentences_empty() {
        assert!(split_sentences("").is_empty());
        assert!(split_sentences("   \n\t ").is_empty());
    }

    #[test]
    fn split_sentences_single_no_terminator() {
        assert_eq!(split_sentences("just a phrase"), vec!["just a phrase"]);
    }

    // ── split_sentences: documented heuristic limitations ─────────────────────
    //
    // These are *characterisation* tests: they pin the current dependency-free
    // heuristic's behaviour on the hard cases (abbreviations, initials, decimals,
    // ellipses, list markers, long sentences) so the trade-off is explicit and a
    // change is caught. They assert what the splitter *does*, not what an ideal
    // sentence segmenter would do — see the `split_sentences` doc comment.

    #[test]
    fn split_sentences_abbreviation_without_internal_space_ends_sentence() {
        // "U.S." has no whitespace after the first dot (followed by `S`), so that
        // dot is not a boundary — the abbreviation stays intact. But the trailing
        // "S." IS followed by whitespace, so it is (mis)treated as a sentence end:
        // "The U.S. economy grew." splits after the abbreviation.
        assert_eq!(split_sentences("The U.S. economy grew."), vec!["The U.S.", "economy grew."],);
    }

    #[test]
    fn split_sentences_spaced_initials_split_per_initial() {
        // With a space after each dot, every initial looks like a sentence end.
        assert_eq!(split_sentences("J. R. R. Tolkien wrote it."), vec!["J.", "R.", "R.", "Tolkien wrote it."],);
    }

    #[test]
    fn split_sentences_decimal_and_version_stay_intact() {
        // Dots inside numbers are followed by digits, not whitespace — not boundaries.
        assert_eq!(
            split_sentences("Version 2.0 ships at 3.30 today."),
            vec!["Version 2.0 ships at 3.30 today."]
        );
    }

    #[test]
    fn split_sentences_ellipsis_ends_a_sentence() {
        // A run of terminators ("...") followed by whitespace is one boundary.
        assert_eq!(split_sentences("Wait... what happened?"), vec!["Wait...", "what happened?"]);
    }

    #[test]
    fn split_sentences_numbered_list_markers_split() {
        // "1." / "2." are a digit + dot + space — each marker is read as a sentence
        // end, so numbered lists fragment: the marker attaches to the *preceding*
        // text and the item text attaches to the *next* marker.
        assert_eq!(
            split_sentences("Steps: 1. Mix flour 2. Add water 3. Bake"),
            vec!["Steps: 1.", "Mix flour 2.", "Add water 3.", "Bake"],
        );
    }

    #[test]
    fn split_sentences_no_maximum_length() {
        // There is no max-sentence cap: a long run with no terminator is one unit.
        // (Windowing happens over whole sentences, so an unusually long sentence
        // yields one large chunk.)
        let long = "word ".repeat(200);
        let long = long.trim();
        assert_eq!(split_sentences(long), vec![long.to_string()]);
    }

    // ── sliding_windows ───────────────────────────────────────────────────────

    fn units(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn sliding_windows_overlap_step_one() {
        let u = units(&["a", "b", "c", "d", "e"]);
        assert_eq!(sliding_windows(&u, 2, 1), vec!["a b", "b c", "c d", "d e"]);
    }

    #[test]
    fn sliding_windows_step_equals_window_with_remainder() {
        let u = units(&["a", "b", "c", "d", "e"]);
        assert_eq!(sliding_windows(&u, 2, 2), vec!["a b", "c d", "e"]);
    }

    #[test]
    fn sliding_windows_window_larger_than_input_is_single_window() {
        let u = units(&["a", "b", "c"]);
        assert_eq!(sliding_windows(&u, 5, 1), vec!["a b c"]);
    }

    #[test]
    fn sliding_windows_single_unit() {
        let u = units(&["a"]);
        assert_eq!(sliding_windows(&u, 2, 1), vec!["a"]);
    }

    #[test]
    fn sliding_windows_empty_input() {
        assert!(sliding_windows(&[], 2, 1).is_empty());
    }

    #[test]
    fn sliding_windows_clamps_zero_sliding_size() {
        // sliding_size 0 would never advance; clamped to 1.
        let u = units(&["a", "b", "c"]);
        assert_eq!(sliding_windows(&u, 2, 0), vec!["a b", "b c"]);
    }

    #[test]
    fn sliding_windows_clamps_zero_window_size() {
        let u = units(&["a", "b"]);
        assert_eq!(sliding_windows(&u, 0, 1), vec!["a", "b"]);
    }

    #[test]
    fn sliding_windows_last_window_keeps_remainder() {
        // window 3, slide 2 over 4 units: window 0 = [a,b,c], window 1 starts at
        // index 2 and holds the remainder [c,d].
        let u = units(&["a", "b", "c", "d"]);
        assert_eq!(sliding_windows(&u, 3, 2), vec!["a b c", "c d"]);
    }

    #[test]
    fn sliding_windows_remainder_shorter_than_window() {
        // window 2, slide 2 over 5 units: [a,b], [c,d], then remainder [e].
        let u = units(&["a", "b", "c", "d", "e"]);
        assert_eq!(sliding_windows(&u, 2, 2), vec!["a b", "c d", "e"]);
    }

    // ── chunk_document ────────────────────────────────────────────────────────

    #[test]
    fn chunk_document_sentences_with_window() {
        let text = "Alpha one. Beta two. Gamma three.";
        // sentences: ["Alpha one.", "Beta two.", "Gamma three."], W=2 S=1
        assert_eq!(chunk_document(text, 2, 1), vec!["Alpha one. Beta two.", "Beta two. Gamma three."],);
    }

    #[test]
    fn chunk_document_one_sentence_per_window() {
        assert_eq!(chunk_document("one two. three four.", 1, 1), vec!["one two.", "three four."],);
    }

    #[test]
    fn chunk_empty_text_yields_no_chunks() {
        assert!(chunk_document("   ", 2, 1).is_empty());
        assert!(chunk_document("", 2, 1).is_empty());
    }
}
