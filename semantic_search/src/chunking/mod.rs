//! Text chunking for embedding.
//!
//! Chunking used to live in the external embedding service; it now lives here so
//! minnal sends the service pre-chunked strings (one embedding is returned per
//! string). The service no longer splits text.
//!
//! Two boundary strategies, matching how the two kinds of text are embedded:
//!
//! - **Documents** (generally longer) are split on **sentence** boundaries via
//!   [`split_sentences`].
//! - **Queries** (generally shorter) are tokenised on **word** boundaries via
//!   [`split_words`].
//!
//! The resulting base units are then grouped into overlapping **sliding
//! windows** by [`sliding_windows`]: each window concatenates `window_size`
//! consecutive units, and the window start advances by `sliding_size` units
//! between chunks. The final window keeps whatever units remain.
//!
//! ```text
//! units = [a, b, c, d, e],  window_size = 2,  sliding_size = 1
//!   → "a b", "b c", "c d", "d e"
//!
//! units = [a, b, c, d, e],  window_size = 2,  sliding_size = 2
//!   → "a b", "c d", "e"          (last window = remainder)
//! ```
//!
//! `window_size` and `sliding_size` are the same knobs as
//! [`crate::service::SemanticSearchConfig`] / the TOML `[semantic_search]`
//! section. They are effectively an on-disk decision: queries must be chunked
//! with the *same* values used to index documents, or Pass-1 ColBERT MaxSim
//! compares mismatched chunkings (see `semantic_search/CLAUDE.md`).
//!
//! The sentence splitter is a deterministic heuristic (a terminator `.`/`!`/`?`
//! followed by whitespace ends a sentence). It is intentionally dependency-free
//! and will mis-split abbreviations such as `U.S.`; the previous ML-based
//! splitter is not reproducible without a new dependency.

/// Which boundary to split raw text on before windowing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkBoundary {
    /// Split on sentence boundaries — used for documents.
    Sentence,
    /// Split on whitespace word boundaries — used for queries.
    Word,
}

/// Chunk `text` into sliding-window strings ready to send to the embedding service.
///
/// Splits `text` into base units according to `boundary`, then groups them into
/// overlapping windows with [`sliding_windows`]. Returns an empty vector when
/// `text` contains no units.
///
/// `window_size` and `sliding_size` are clamped to a minimum of 1; a
/// `sliding_size` of 0 would otherwise never advance.
pub fn chunk_text(text: &str, boundary: ChunkBoundary, window_size: usize, sliding_size: usize) -> Vec<String> {
    let units = match boundary {
        ChunkBoundary::Sentence => split_sentences(text),
        ChunkBoundary::Word => split_words(text),
    };
    sliding_windows(&units, window_size, sliding_size)
}

/// Chunk a document: sentence-split then sliding-window. See [`chunk_text`].
pub fn chunk_document(text: &str, window_size: usize, sliding_size: usize) -> Vec<String> {
    chunk_text(text, ChunkBoundary::Sentence, window_size, sliding_size)
}

/// Chunk a query: word-tokenise then sliding-window. See [`chunk_text`].
pub fn chunk_query(text: &str, window_size: usize, sliding_size: usize) -> Vec<String> {
    chunk_text(text, ChunkBoundary::Word, window_size, sliding_size)
}

/// Split `text` into whitespace-delimited word tokens.
///
/// Punctuation stays attached to its word (`"when?"` is one token), matching the
/// IR-style whitespace tokenisation the embedding service previously used.
pub fn split_words(text: &str) -> Vec<String> {
    text.split_whitespace().map(str::to_string).collect()
}

/// Split `text` into sentences using a deterministic heuristic.
///
/// A run of one or more terminators (`.`, `!`, `?`) that is immediately followed
/// by whitespace (or the end of the text) ends a sentence; the terminators stay
/// with the sentence and the intervening whitespace is dropped. Each sentence is
/// trimmed and empty results are discarded.
///
/// This mirrors a `split on (?<=[.!?])\s+` regex. It does not understand
/// abbreviations (`U.S.` splits into `U.` / `S.`).
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

    // ── split_words ───────────────────────────────────────────────────────────

    #[test]
    fn split_words_basic() {
        assert_eq!(split_words("who filed the complaint"), vec!["who", "filed", "the", "complaint"]);
    }

    #[test]
    fn split_words_keeps_punctuation_and_collapses_whitespace() {
        assert_eq!(split_words("  What is\tLlama?  and\nwhy "), vec!["What", "is", "Llama?", "and", "why"]);
    }

    #[test]
    fn split_words_empty() {
        assert!(split_words("   ").is_empty());
        assert!(split_words("").is_empty());
    }

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

    // ── chunk_document / chunk_query ──────────────────────────────────────────

    #[test]
    fn chunk_document_sentences_with_window() {
        let text = "Alpha one. Beta two. Gamma three.";
        // sentences: ["Alpha one.", "Beta two.", "Gamma three."], W=2 S=1
        assert_eq!(chunk_document(text, 2, 1), vec!["Alpha one. Beta two.", "Beta two. Gamma three."],);
    }

    #[test]
    fn chunk_query_words_with_window() {
        let text = "who filed the complaint";
        // words: [who, filed, the, complaint], W=2 S=1
        assert_eq!(chunk_query(text, 2, 1), vec!["who filed", "filed the", "the complaint"],);
    }

    #[test]
    fn chunk_text_dispatches_on_boundary() {
        let text = "one two. three four.";
        assert_eq!(chunk_text(text, ChunkBoundary::Word, 4, 4), vec!["one two. three four."],);
        assert_eq!(chunk_text(text, ChunkBoundary::Sentence, 1, 1), vec!["one two.", "three four."],);
    }

    #[test]
    fn chunk_empty_text_yields_no_chunks() {
        assert!(chunk_document("   ", 2, 1).is_empty());
        assert!(chunk_query("", 2, 1).is_empty());
    }
}
