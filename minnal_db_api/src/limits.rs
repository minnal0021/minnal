//! Request-size bounds enforced at the API entry point.
//!
//! Result-count parameters (`limit`, `page_size`, `top_k`) are caller-supplied
//! `usize`s that size a scan, an allocation, or an ANN probe. Left unbounded, a
//! single request asking for `usize::MAX` results forces the engine to build
//! keys and pointers for everything at or past the cursor before the page is
//! truncated — a cheap remote resource-exhaustion lever, and one reachable
//! without authentication today.
//!
//! The bound lives in **one place**: [`Limit`]'s [`Deserialize`] impl. Because
//! axum's `Query` and `Json` extractors deserialize the request, that impl *is*
//! the entry-point check — a value that exceeds the cap cannot reach a handler,
//! and a new endpoint gets the bound by using the type rather than by
//! remembering to call something.
//!
//! Applying it is a type change on each parameter field, which is deliberate:
//! the compiler then flags every use site, so no endpoint can be silently
//! missed the way a scattered `.min(MAX)` would allow.
//!
//! # Why not middleware
//!
//! A layer sitting in front of the router would have to cover both query-string
//! and JSON-body parameters, under several different field names, which means
//! buffering and re-parsing every body and re-encoding the same per-field
//! knowledge anyway. Serde already parses the request once and knows the field
//! shape; putting the bound there is the same check in a place that cannot drift
//! from the types.

use std::fmt;

use serde::{Deserialize, Deserializer};

/// Largest result count any endpoint will honour.
///
/// Chosen to sit far above any legitimate page (defaults are 20) while keeping
/// the worst-case single request bounded. Callers asking for more are clamped,
/// not rejected — see [`Limit`].
pub const MAX_RESULT_LIMIT: usize = 1_000;

/// Default result count when the caller supplies none.
pub const DEFAULT_RESULT_LIMIT: usize = 20;

/// A caller-supplied result count, clamped to [`MAX_RESULT_LIMIT`] on the way in.
///
/// Clamping rather than rejecting is deliberate. The extractors produce
/// inconsistent status codes for a deserialization failure (`Query` rejects with
/// 400, `Json` with 422) and neither routes through this crate's `AppError` JSON
/// shape, so rejecting would need custom rejection plumbing to report
/// consistently. Clamping keeps existing clients working, keeps every response
/// on the normal path, and still bounds the work — an over-large request simply
/// gets the biggest page the server is willing to build.
///
/// A clamped request is logged at `debug` so it can be traced when a client
/// wonders why it received fewer rows than it asked for.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Limit(usize);

/// Forwarded rather than derived so `?limit` in a tracing field logs `5`, not
/// `Limit(5)` — the wrapper is an input-validation detail, not something log
/// consumers should have to learn about.
impl fmt::Debug for Limit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Limit {
    /// The bounded value.
    pub fn get(self) -> usize {
        self.0
    }

    /// Bound `value`, logging when the cap actually bites.
    pub fn clamped(value: usize) -> Self {
        if value > MAX_RESULT_LIMIT {
            tracing::debug!(requested = value, cap = MAX_RESULT_LIMIT, "result limit clamped to cap");
            return Limit(MAX_RESULT_LIMIT);
        }
        Limit(value)
    }
}

impl Default for Limit {
    fn default() -> Self {
        Limit(DEFAULT_RESULT_LIMIT)
    }
}

impl fmt::Display for Limit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl From<Limit> for usize {
    fn from(l: Limit) -> usize {
        l.0
    }
}

impl<'de> Deserialize<'de> for Limit {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Limit::clamped(usize::deserialize(deserializer)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Deserialize)]
    struct Params {
        #[serde(default)]
        limit: Limit,
        top_k: Option<Limit>,
    }

    #[test]
    fn a_value_under_the_cap_passes_through_unchanged() {
        assert_eq!(Limit::clamped(50).get(), 50);
        assert_eq!(Limit::clamped(MAX_RESULT_LIMIT).get(), MAX_RESULT_LIMIT);
        assert_eq!(Limit::clamped(0).get(), 0);
    }

    #[test]
    fn a_value_over_the_cap_is_clamped() {
        assert_eq!(Limit::clamped(MAX_RESULT_LIMIT + 1).get(), MAX_RESULT_LIMIT);
        assert_eq!(Limit::clamped(usize::MAX).get(), MAX_RESULT_LIMIT);
    }

    #[test]
    fn the_bound_is_applied_while_deserializing_not_after() {
        // The point of the newtype: a handler cannot observe an unbounded value,
        // because one never survives extraction. This is the entry-point check.
        let p: Params = serde_json::from_str(r#"{"limit": 999999, "top_k": 500000}"#).unwrap();
        assert_eq!(p.limit.get(), MAX_RESULT_LIMIT);
        assert_eq!(p.top_k.map(Limit::get), Some(MAX_RESULT_LIMIT));
    }

    #[test]
    fn an_absent_field_takes_the_default_not_the_cap() {
        let p: Params = serde_json::from_str("{}").unwrap();
        assert_eq!(p.limit.get(), DEFAULT_RESULT_LIMIT);
        assert_eq!(p.top_k, None);
    }

    #[test]
    fn a_negative_value_is_rejected_rather_than_wrapping() {
        // usize deserialization fails on a negative, which the extractor turns
        // into a 4xx. It must not wrap to a huge positive.
        assert!(serde_json::from_str::<Params>(r#"{"limit": -1}"#).is_err());
    }

    // Note: `Query` (query strings) and `Json` (bodies) run different
    // deserializers, but both drive the single `Deserialize for Limit` impl
    // exercised above — which is the whole reason the bound lives there rather
    // than at each call site. Covering the urlencoded path directly would mean
    // adding `serde_urlencoded` as a dependency, which the project forbids
    // without discussion.
}
