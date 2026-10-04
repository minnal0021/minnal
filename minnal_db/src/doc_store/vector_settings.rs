//! A namespace's vector-index settings: the embedding model and dimension, how
//! documents are chunked, the code widths, and the search defaults.
//!
//! They live in the store's schema (`vector_index` in [`DocStoreSchema`] and
//! [`KvStoreSchema`]) rather than in the engine config, so each namespace
//! describes its own index. Two types:
//!
//! - [`VectorIndexSpec`] is what a schema file or a request carries: every field
//!   optional, unknown keys rejected (a misspelt key fails rather than being
//!   ignored).
//! - [`VectorIndexSettings`] is the resolved, validated form the engine uses.
//!
//! When semantic search is first enabled, omitted fields are filled with the
//! built-in defaults and the filled-in spec is what is saved, so a schema read
//! back always shows the values in force and a later change of default never
//! changes an existing namespace.
//!
//! Mutability, once a namespace has settings:
//!
//! | Fields | Rule |
//! |---|---|
//! | `embedding_model`, `embedding_dim`, `chunking.*` | fixed: they decide what the stored vectors are |
//! | `quantisation.*` | read-only: only the current value is accepted |
//! | `search.*` | changeable at any time; also overridable per request |
//!
//! [`DocStoreSchema`]: crate::doc_store::DocStoreSchema
//! [`KvStoreSchema`]: crate::doc_store::kv_schema::KvStoreSchema

use serde::{Deserialize, Serialize};

use super::error::SchemaError;

/// Model used when a namespace does not name one.
pub const DEFAULT_EMBEDDING_MODEL: &str = "gemma";
/// Longest accepted model name.
pub const MAX_EMBEDDING_MODEL_LEN: usize = 64;
/// Dimension used when a namespace does not give one.
pub const DEFAULT_EMBEDDING_DIM: u32 = 768;
/// Smallest dimension the rotation supports.
pub const MIN_EMBEDDING_DIM: u32 = 8;
/// Largest accepted dimension.
pub const MAX_EMBEDDING_DIM: u32 = 4096;
/// Default sentences per document chunk.
pub const DEFAULT_WINDOW_SIZE: u32 = 4;
/// Largest accepted chunk window, in sentences.
pub const MAX_WINDOW_SIZE: u32 = 64;
/// Default step between document chunks, in sentences.
pub const DEFAULT_SLIDING_SIZE: u32 = 2;
/// Rotation seed a new namespace gets (the same value as
/// `semantic_search::cluster::DEFAULT_ROTATION_SEED`, pinned together by a test).
pub const DEFAULT_ROTATION_SEED: u64 = 0x6d69_6e6e_616c_0001;
/// Code width of Pass-1 (chunk) codes. Read-only.
pub const PASS1_BITS: u8 = 1;
/// Code width of Pass-2 (whole-document) codes. Read-only.
pub const PASS2_BITS: u8 = 8;
/// What Pass-2 codes are encoded against. Read-only: only the zero centre
/// (design doc M2c).
pub const PASS2_CENTRE: &str = "zero";
/// Default number of clusters probed by Pass 1.
pub const DEFAULT_N_PROBES: u32 = 64;
/// Largest accepted `n_probes`. A search probes at most the namespace's
/// posting count, so this is a ceiling, not tied to any centroid file.
pub const MAX_N_PROBES: u32 = 4096;
/// Default number of candidates Pass 1 hands to Pass 2.
pub const DEFAULT_FIRST_PASS_TOP_K: u32 = 1000;
/// Largest accepted Pass-1 cut.
pub const MAX_FIRST_PASS_TOP_K: u32 = 10_000;
/// Default number of results a search returns.
pub const DEFAULT_TOP_K: u32 = 100;
/// Largest accepted result count (the API's result-limit cap).
pub const MAX_TOP_K: u32 = 1000;

// ── Spec: what a schema or a request carries ─────────────────────────────────

/// A namespace's vector-index settings as written in a schema or a request.
/// Every field is optional; see the [module docs](self) for defaults and rules.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VectorIndexSpec {
    /// Embedding model the service is asked for. Lower-cased on save.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_model: Option<String>,
    /// Embedding dimension (the `dimensions` sent to the service).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_dim: Option<u32>,
    /// How documents are split into Pass-1 chunks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunking: Option<ChunkingSpec>,
    /// Code widths (read-only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantisation: Option<QuantisationSpec>,
    /// Search defaults.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search: Option<SearchSpec>,
    /// Where the namespace's centres came from. Written by the server when it
    /// seeds them; a request may not set it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seeded_from: Option<SeededFrom>,
}

/// The centroid file a namespace's centres were seeded from. After seeding the
/// namespace never reads the file again; this records which one it was.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeededFrom {
    /// Path of the centroid file, as the server found it.
    pub file: String,
    /// MurmurHash3 (x64, 128-bit) of the file's bytes, hex.
    pub murmur3_128: String,
    /// Number of centres seeded.
    pub centres: u32,
}

/// Document chunking: sentence windows of `window_size`, advancing by
/// `sliding_size`. Queries are not chunked.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChunkingSpec {
    /// Sentences per chunk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window_size: Option<u32>,
    /// Sentences the window advances by.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sliding_size: Option<u32>,
}

/// Code widths of the two passes. Read-only today.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuantisationSpec {
    /// Bits per dimension of Pass-1 (chunk) codes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pass1_bits: Option<u8>,
    /// Bits per dimension of Pass-2 (whole-document) codes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pass2_bits: Option<u8>,
    /// Seed of the rotation codes are computed in, as a hex string
    /// (`"0x6d696e6e616c0001"`; a JSON number would lose precision above 2^53).
    /// Fixed once set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rotation_seed: Option<String>,
    /// What Pass-2 (whole-document) codes are encoded against: `"zero"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pass2_centre: Option<String>,
}

/// Search settings: the namespace's defaults, or one request's overrides.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchSpec {
    /// Clusters Pass 1 probes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_probes: Option<u32>,
    /// Candidates Pass 1 hands to Pass 2.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_pass_top_k: Option<u32>,
    /// Results returned.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
}

// ── Settings: resolved and validated ─────────────────────────────────────────

/// Resolved, validated vector-index settings of one namespace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorIndexSettings {
    /// Lower-cased embedding model name.
    pub embedding_model: String,
    /// Embedding dimension.
    pub embedding_dim: u32,
    /// Sentences per document chunk.
    pub window_size: u32,
    /// Sentences the chunk window advances by.
    pub sliding_size: u32,
    /// Pass-1 code width (always [`PASS1_BITS`]).
    pub pass1_bits: u8,
    /// Pass-2 code width (always [`PASS2_BITS`]).
    pub pass2_bits: u8,
    /// Seed of the rotation codes are computed in.
    pub rotation_seed: u64,
    /// What Pass-2 codes are encoded against (always [`PASS2_CENTRE`]).
    pub pass2_centre: String,
    /// Where the centres came from, once seeded.
    pub seeded_from: Option<SeededFrom>,
    /// Search defaults.
    pub search: SearchSettings,
}

/// Resolved search settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchSettings {
    /// Clusters Pass 1 probes.
    pub n_probes: u32,
    /// Candidates Pass 1 hands to Pass 2.
    pub first_pass_top_k: u32,
    /// Results returned.
    pub top_k: u32,
}

impl Default for SearchSettings {
    fn default() -> Self {
        Self {
            n_probes: DEFAULT_N_PROBES,
            first_pass_top_k: DEFAULT_FIRST_PASS_TOP_K,
            top_k: DEFAULT_TOP_K,
        }
    }
}

fn invalid(field: &'static str, reason: impl Into<String>) -> SchemaError {
    SchemaError::InvalidVectorSetting {
        field,
        reason: reason.into(),
    }
}

fn check_range(field: &'static str, value: u32, min: u32, max: u32) -> Result<(), SchemaError> {
    if value < min || value > max {
        return Err(invalid(field, format!("must be between {min} and {max}, got {value}")));
    }
    Ok(())
}

/// Lower-case and check a model name.
fn normalise_model(model: &str) -> Result<String, SchemaError> {
    let m = model.trim().to_lowercase();
    if m.is_empty() {
        return Err(invalid("embedding_model", "must not be empty"));
    }
    if m.len() > MAX_EMBEDDING_MODEL_LEN {
        return Err(invalid(
            "embedding_model",
            format!("must be at most {MAX_EMBEDDING_MODEL_LEN} characters, got {}", m.len()),
        ));
    }
    if let Some(c) = m
        .chars()
        .find(|c| !(c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '.' | '_' | '-')))
    {
        return Err(invalid(
            "embedding_model",
            format!("may only contain a-z, 0-9, '.', '_' and '-', got {c:?} in {model:?}"),
        ));
    }
    Ok(m)
}

fn check_dim(dim: u32) -> Result<(), SchemaError> {
    check_range("embedding_dim", dim, MIN_EMBEDDING_DIM, MAX_EMBEDDING_DIM)?;
    if !dim.is_multiple_of(2) {
        return Err(invalid("embedding_dim", format!("must be even (the rotation needs it), got {dim}")));
    }
    Ok(())
}

fn check_chunking(window: u32, sliding: u32) -> Result<(), SchemaError> {
    check_range("chunking.window_size", window, 1, MAX_WINDOW_SIZE)?;
    if sliding < 1 || sliding > window {
        return Err(invalid(
            "chunking.sliding_size",
            format!("must be between 1 and window_size ({window}), got {sliding}; a larger step would skip sentences"),
        ));
    }
    Ok(())
}

/// Parse a `"0x…"` hex seed (1 to 16 hex digits).
fn parse_seed(s: &str) -> Result<u64, SchemaError> {
    let digits = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")).ok_or_else(|| {
        invalid(
            "quantisation.rotation_seed",
            format!("must be a hex string like \"0x6d696e6e616c0001\", got {s:?}"),
        )
    })?;
    if digits.is_empty() || digits.len() > 16 {
        return Err(invalid("quantisation.rotation_seed", format!("must have 1 to 16 hex digits, got {s:?}")));
    }
    u64::from_str_radix(digits, 16).map_err(|_| invalid("quantisation.rotation_seed", format!("is not hex: {s:?}")))
}

/// A seed as the schema writes it.
pub fn format_seed(seed: u64) -> String {
    format!("0x{seed:016x}")
}

fn check_bits(pass1: u8, pass2: u8) -> Result<(), SchemaError> {
    if pass1 != PASS1_BITS {
        return Err(invalid(
            "quantisation.pass1_bits",
            format!("is read-only and must be {PASS1_BITS}, got {pass1}"),
        ));
    }
    if pass2 != PASS2_BITS {
        return Err(invalid(
            "quantisation.pass2_bits",
            format!("is read-only and must be {PASS2_BITS}, got {pass2}"),
        ));
    }
    Ok(())
}

impl SearchSettings {
    /// Check every range and `top_k ≤ first_pass_top_k`.
    pub fn validate(&self) -> Result<(), SchemaError> {
        check_range("search.n_probes", self.n_probes, 1, MAX_N_PROBES)?;
        check_range("search.first_pass_top_k", self.first_pass_top_k, 1, MAX_FIRST_PASS_TOP_K)?;
        check_range("search.top_k", self.top_k, 1, MAX_TOP_K)?;
        if self.top_k > self.first_pass_top_k {
            return Err(invalid(
                "search.top_k",
                format!(
                    "must be at most first_pass_top_k ({}), got {}; Pass 2 can only return what Pass 1 hands it",
                    self.first_pass_top_k, self.top_k
                ),
            ));
        }
        Ok(())
    }
}

impl SearchSpec {
    /// Apply these values over `base` (unset fields keep `base`'s) and validate
    /// the result. Used for `UpdateVectorSearch` and for per-request overrides.
    pub fn apply(&self, base: SearchSettings) -> Result<SearchSettings, SchemaError> {
        let s = SearchSettings {
            n_probes: self.n_probes.unwrap_or(base.n_probes),
            first_pass_top_k: self.first_pass_top_k.unwrap_or(base.first_pass_top_k),
            top_k: self.top_k.unwrap_or(base.top_k),
        };
        s.validate()?;
        Ok(s)
    }

    /// `true` when no field is set.
    pub fn is_empty(&self) -> bool {
        self.n_probes.is_none() && self.first_pass_top_k.is_none() && self.top_k.is_none()
    }
}

impl VectorIndexSpec {
    /// Reject `seeded_from` in a request: only the server writes it, when it
    /// seeds the namespace's centres.
    pub fn reject_seeded_from(&self) -> Result<(), SchemaError> {
        if self.seeded_from.is_some() {
            return Err(invalid("seeded_from", "is written by the server when it seeds the centres; omit it"));
        }
        Ok(())
    }

    /// Fill omitted fields with the built-in defaults and validate.
    pub fn resolve(&self) -> Result<VectorIndexSettings, SchemaError> {
        let chunking = self.chunking.unwrap_or_default();
        let quantisation = self.quantisation.clone().unwrap_or_default();
        let search = self.search.unwrap_or_default();
        // A default never makes a valid request invalid: an omitted slide or
        // result count is capped at its partner field (window 1 alone is
        // window 1 slide 1; a cut of 50 alone returns at most 50). Values the
        // caller gives are checked as given.
        let window_size = chunking.window_size.unwrap_or(DEFAULT_WINDOW_SIZE);
        let first_pass_top_k = search.first_pass_top_k.unwrap_or(DEFAULT_FIRST_PASS_TOP_K);
        let settings = VectorIndexSettings {
            embedding_model: normalise_model(self.embedding_model.as_deref().unwrap_or(DEFAULT_EMBEDDING_MODEL))?,
            embedding_dim: self.embedding_dim.unwrap_or(DEFAULT_EMBEDDING_DIM),
            window_size,
            sliding_size: chunking.sliding_size.unwrap_or(DEFAULT_SLIDING_SIZE.min(window_size)),
            pass1_bits: quantisation.pass1_bits.unwrap_or(PASS1_BITS),
            pass2_bits: quantisation.pass2_bits.unwrap_or(PASS2_BITS),
            rotation_seed: match &quantisation.rotation_seed {
                Some(s) => parse_seed(s)?,
                None => DEFAULT_ROTATION_SEED,
            },
            seeded_from: self.seeded_from.clone(),
            pass2_centre: quantisation.pass2_centre.clone().unwrap_or_else(|| PASS2_CENTRE.to_string()),
            search: SearchSettings {
                n_probes: search.n_probes.unwrap_or(DEFAULT_N_PROBES),
                first_pass_top_k,
                top_k: search.top_k.unwrap_or(DEFAULT_TOP_K.min(first_pass_top_k)),
            },
        };
        settings.validate()?;
        Ok(settings)
    }

    /// Combine this request with settings a namespace already has: a fixed or
    /// read-only field may be omitted or repeated but not changed; search
    /// fields may change. Returns the resulting settings.
    pub fn merge_onto(&self, current: &VectorIndexSettings) -> Result<VectorIndexSettings, SchemaError> {
        fn fixed<T: PartialEq + std::fmt::Display>(field: &'static str, requested: Option<T>, current: T) -> Result<(), SchemaError> {
            match requested {
                Some(r) if r != current => Err(SchemaError::VectorSettingFixed {
                    field,
                    current: current.to_string(),
                    requested: r.to_string(),
                }),
                _ => Ok(()),
            }
        }
        if let Some(m) = &self.embedding_model {
            fixed("embedding_model", Some(normalise_model(m)?), current.embedding_model.clone())?;
        }
        fixed("embedding_dim", self.embedding_dim, current.embedding_dim)?;
        let chunking = self.chunking.unwrap_or_default();
        fixed("chunking.window_size", chunking.window_size, current.window_size)?;
        fixed("chunking.sliding_size", chunking.sliding_size, current.sliding_size)?;
        let quantisation = self.quantisation.clone().unwrap_or_default();
        fixed("quantisation.pass1_bits", quantisation.pass1_bits, current.pass1_bits)?;
        fixed("quantisation.pass2_bits", quantisation.pass2_bits, current.pass2_bits)?;
        fixed(
            "quantisation.pass2_centre",
            quantisation.pass2_centre.clone(),
            current.pass2_centre.clone(),
        )?;
        if let Some(seed) = &quantisation.rotation_seed {
            let seed = parse_seed(seed)?;
            if seed != current.rotation_seed {
                return Err(SchemaError::VectorSettingFixed {
                    field: "quantisation.rotation_seed",
                    current: format_seed(current.rotation_seed),
                    requested: format_seed(seed),
                });
            }
        }
        self.reject_seeded_from()?;
        Ok(VectorIndexSettings {
            search: self.search.unwrap_or_default().apply(current.search)?,
            ..current.clone()
        })
    }
}

impl VectorIndexSettings {
    /// Check every field and the cross-field rules.
    pub fn validate(&self) -> Result<(), SchemaError> {
        normalise_model(&self.embedding_model)?;
        check_dim(self.embedding_dim)?;
        check_chunking(self.window_size, self.sliding_size)?;
        check_bits(self.pass1_bits, self.pass2_bits)?;
        if self.pass2_centre != PASS2_CENTRE {
            return Err(invalid(
                "quantisation.pass2_centre",
                format!("is read-only and must be \"{PASS2_CENTRE}\", got {:?}", self.pass2_centre),
            ));
        }
        self.search.validate()
    }

    /// The fully filled-in spec, as saved in a schema.
    pub fn to_spec(&self) -> VectorIndexSpec {
        VectorIndexSpec {
            embedding_model: Some(self.embedding_model.clone()),
            embedding_dim: Some(self.embedding_dim),
            chunking: Some(ChunkingSpec {
                window_size: Some(self.window_size),
                sliding_size: Some(self.sliding_size),
            }),
            quantisation: Some(QuantisationSpec {
                pass1_bits: Some(self.pass1_bits),
                pass2_bits: Some(self.pass2_bits),
                rotation_seed: Some(format_seed(self.rotation_seed)),
                pass2_centre: Some(self.pass2_centre.clone()),
            }),
            search: Some(SearchSpec {
                n_probes: Some(self.search.n_probes),
                first_pass_top_k: Some(self.search.first_pass_top_k),
                top_k: Some(self.search.top_k),
            }),
            seeded_from: self.seeded_from.clone(),
        }
    }
}

/// The settings a schema's `vector_index` resolves to, or the defaults when it
/// has none.
pub(crate) fn resolve_or_default(spec: Option<&VectorIndexSpec>) -> Result<VectorIndexSettings, SchemaError> {
    spec.cloned().unwrap_or_default().resolve()
}

/// The `vector_index` to save when semantic search is (first) enabled with
/// `requested`: merged onto what the schema already holds, or resolved from
/// the defaults when it holds nothing yet.
pub(crate) fn settle(existing: Option<&VectorIndexSpec>, requested: Option<&VectorIndexSpec>) -> Result<VectorIndexSpec, SchemaError> {
    let requested = requested.cloned().unwrap_or_default();
    requested.reject_seeded_from()?;
    let settings = match existing {
        Some(e) => requested.merge_onto(&e.resolve()?)?,
        None => requested.resolve()?,
    };
    Ok(settings.to_spec())
}

/// The `vector_index` to save after an `UpdateVectorSearch`: `search` applied
/// over the namespace's current search settings, everything else unchanged.
pub(crate) fn update_search(existing: Option<&VectorIndexSpec>, namespace: &str, search: &SearchSpec) -> Result<VectorIndexSpec, SchemaError> {
    let current = existing
        .ok_or_else(|| SchemaError::VectorIndexNotConfigured {
            namespace: namespace.to_owned(),
        })?
        .resolve()?;
    if search.is_empty() {
        return Err(invalid("search", "update must set at least one of n_probes, first_pass_top_k, top_k"));
    }
    Ok(VectorIndexSettings {
        search: search.apply(current.search)?,
        ..current
    }
    .to_spec())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(json: &str) -> VectorIndexSpec {
        serde_json::from_str(json).unwrap()
    }

    fn err_field(r: Result<VectorIndexSettings, SchemaError>) -> &'static str {
        match r.unwrap_err() {
            SchemaError::InvalidVectorSetting { field, .. } | SchemaError::VectorSettingFixed { field, .. } => field,
            other => panic!("unexpected error {other:?}"),
        }
    }

    #[test]
    fn rotation_seed_is_hex_fixed_and_defaults_to_the_pinned_value() {
        let s = VectorIndexSpec::default().resolve().unwrap();
        assert_eq!(s.rotation_seed, DEFAULT_ROTATION_SEED);
        let filled = s.to_spec();
        assert_eq!(filled.quantisation.as_ref().unwrap().rotation_seed.as_deref(), Some("0x6d696e6e616c0001"));
        assert_eq!(filled.resolve().unwrap().rotation_seed, DEFAULT_ROTATION_SEED);
        assert_eq!(spec(r#"{"quantisation":{"rotation_seed":"0X1F"}}"#).resolve().unwrap().rotation_seed, 31);
        for bad in [r#""31""#, r#""0x""#, r#""0xZZ""#, r#""0x11112222333344445""#] {
            assert_eq!(
                err_field(spec(&format!(r#"{{"quantisation":{{"rotation_seed":{bad}}}}}"#)).resolve()),
                "quantisation.rotation_seed",
                "{bad}"
            );
        }
        assert_eq!(
            err_field(spec(r#"{"quantisation":{"rotation_seed":"0x2"}}"#).merge_onto(&s)),
            "quantisation.rotation_seed"
        );
        assert!(spec(r#"{"quantisation":{"rotation_seed":"0x6D696E6E616C0001"}}"#).merge_onto(&s).is_ok());
    }

    #[cfg(feature = "semantic-search")]
    #[test]
    fn default_rotation_seed_matches_the_cluster_index() {
        assert_eq!(DEFAULT_ROTATION_SEED, crate::semantic_search::cluster::DEFAULT_ROTATION_SEED);
    }

    #[test]
    fn pass2_centre_is_read_only_zero() {
        let s = VectorIndexSpec::default().resolve().unwrap();
        assert_eq!(s.pass2_centre, "zero");
        assert_eq!(s.to_spec().quantisation.unwrap().pass2_centre.as_deref(), Some("zero"));
        assert!(spec(r#"{"quantisation":{"pass2_centre":"zero"}}"#).resolve().is_ok());
        assert_eq!(
            err_field(spec(r#"{"quantisation":{"pass2_centre":"nearest"}}"#).resolve()),
            "quantisation.pass2_centre"
        );
        assert_eq!(
            err_field(spec(r#"{"quantisation":{"pass2_centre":"nearest"}}"#).merge_onto(&s)),
            "quantisation.pass2_centre"
        );
        assert!(spec(r#"{"quantisation":{"pass2_centre":"zero"}}"#).merge_onto(&s).is_ok());
    }

    #[test]
    fn seeded_from_is_server_written() {
        let seeded = spec(r#"{"seeded_from":{"file":"f","murmur3_128":"ab","centres":2}}"#);
        assert!(seeded.resolve().is_ok(), "a stored schema with it loads");
        assert_eq!(
            err_field(seeded.merge_onto(&VectorIndexSpec::default().resolve().unwrap())),
            "seeded_from"
        );
        assert!(settle(None, Some(&seeded)).is_err());
        let kept = settle(Some(&seeded), None).unwrap();
        assert_eq!(kept.seeded_from, seeded.seeded_from, "kept across a re-enable");
    }

    #[test]
    fn empty_spec_resolves_to_the_defaults() {
        let s = VectorIndexSpec::default().resolve().unwrap();
        assert_eq!(s.embedding_model, "gemma");
        assert_eq!(s.embedding_dim, 768);
        assert_eq!((s.window_size, s.sliding_size), (4, 2));
        assert_eq!((s.pass1_bits, s.pass2_bits), (1, 8));
        assert_eq!(
            s.search,
            SearchSettings {
                n_probes: 64,
                first_pass_top_k: 1000,
                top_k: 100
            }
        );
    }

    #[test]
    fn to_spec_round_trips_and_fills_every_field() {
        let s = spec(r#"{"embedding_model":"Qwen","chunking":{"window_size":6}}"#).resolve().unwrap();
        let filled = s.to_spec();
        assert_eq!(filled.embedding_model.as_deref(), Some("qwen"));
        assert_eq!(filled.chunking.unwrap().sliding_size, Some(2));
        assert_eq!(filled.search.unwrap().top_k, Some(100));
        assert_eq!(filled.resolve().unwrap(), s);
        let json = serde_json::to_string(&filled).unwrap();
        assert_eq!(serde_json::from_str::<VectorIndexSpec>(&json).unwrap(), filled);
    }

    #[test]
    fn unknown_keys_are_rejected() {
        assert!(serde_json::from_str::<VectorIndexSpec>(r#"{"embeding_model":"gemma"}"#).is_err());
        assert!(serde_json::from_str::<VectorIndexSpec>(r#"{"chunking":{"windowsize":4}}"#).is_err());
        assert!(serde_json::from_str::<VectorIndexSpec>(r#"{"search":{"nprobes":4}}"#).is_err());
        assert!(serde_json::from_str::<VectorIndexSpec>(r#"{"embedding_dim":-8}"#).is_err());
    }

    #[test]
    fn model_rules() {
        assert_eq!(
            spec(r#"{"embedding_model":"  E5-Large.v2_x "}"#).resolve().unwrap().embedding_model,
            "e5-large.v2_x"
        );
        for bad in [r#""""#, r#""   ""#, r#""gem ma""#, r#""gemma/x""#, r#""gémma""#] {
            assert_eq!(
                err_field(spec(&format!(r#"{{"embedding_model":{bad}}}"#)).resolve()),
                "embedding_model",
                "{bad}"
            );
        }
        let long = "a".repeat(MAX_EMBEDDING_MODEL_LEN + 1);
        assert_eq!(
            err_field(spec(&format!(r#"{{"embedding_model":"{long}"}}"#)).resolve()),
            "embedding_model"
        );
        let ok = "a".repeat(MAX_EMBEDDING_MODEL_LEN);
        assert!(spec(&format!(r#"{{"embedding_model":"{ok}"}}"#)).resolve().is_ok());
    }

    #[test]
    fn dimension_rules_at_the_edges() {
        for ok in [8, 10, 768, 4096] {
            assert!(spec(&format!(r#"{{"embedding_dim":{ok}}}"#)).resolve().is_ok(), "{ok}");
        }
        for bad in [0, 6, 7, 9, 767, 4097, 4098] {
            assert_eq!(
                err_field(spec(&format!(r#"{{"embedding_dim":{bad}}}"#)).resolve()),
                "embedding_dim",
                "{bad}"
            );
        }
    }

    #[test]
    fn chunking_rules_at_the_edges() {
        let c = |w: u32, s: u32| spec(&format!(r#"{{"chunking":{{"window_size":{w},"sliding_size":{s}}}}}"#)).resolve();
        assert!(c(1, 1).is_ok());
        assert!(c(64, 64).is_ok());
        assert!(c(4, 4).is_ok());
        assert_eq!(err_field(c(0, 1)), "chunking.window_size");
        assert_eq!(err_field(c(65, 2)), "chunking.window_size");
        assert_eq!(err_field(c(4, 0)), "chunking.sliding_size");
        assert_eq!(err_field(c(4, 5)), "chunking.sliding_size");
        // An omitted slide defaults to at most the window.
        assert_eq!(spec(r#"{"chunking":{"window_size":1}}"#).resolve().unwrap().sliding_size, 1);
        assert_eq!(spec(r#"{"chunking":{"window_size":3}}"#).resolve().unwrap().sliding_size, 2);
    }

    #[test]
    fn bits_are_read_only() {
        assert!(spec(r#"{"quantisation":{"pass1_bits":1,"pass2_bits":8}}"#).resolve().is_ok());
        assert_eq!(
            err_field(spec(r#"{"quantisation":{"pass1_bits":2}}"#).resolve()),
            "quantisation.pass1_bits"
        );
        assert_eq!(
            err_field(spec(r#"{"quantisation":{"pass2_bits":4}}"#).resolve()),
            "quantisation.pass2_bits"
        );
    }

    #[test]
    fn search_rules_at_the_edges() {
        let s = |n: u32, f: u32, k: u32| spec(&format!(r#"{{"search":{{"n_probes":{n},"first_pass_top_k":{f},"top_k":{k}}}}}"#)).resolve();
        assert!(s(1, 1, 1).is_ok());
        assert!(s(MAX_N_PROBES, MAX_FIRST_PASS_TOP_K, MAX_TOP_K).is_ok());
        assert!(s(64, 100, 100).is_ok());
        assert_eq!(err_field(s(0, 1000, 100)), "search.n_probes");
        assert_eq!(err_field(s(MAX_N_PROBES + 1, 1000, 100)), "search.n_probes");
        assert_eq!(err_field(s(64, 1000, 0)), "search.top_k");
        assert_eq!(err_field(s(64, 2000, MAX_TOP_K + 1)), "search.top_k");
        assert_eq!(err_field(s(64, 0, 1)), "search.first_pass_top_k");
        assert_eq!(err_field(s(64, MAX_FIRST_PASS_TOP_K + 1, 100)), "search.first_pass_top_k");
        assert_eq!(err_field(s(64, 50, 100)), "search.top_k");
        // An omitted top_k defaults to at most the cut; an explicit one is checked.
        assert_eq!(spec(r#"{"search":{"first_pass_top_k":50}}"#).resolve().unwrap().search.top_k, 50);
        assert_eq!(
            err_field(spec(r#"{"search":{"first_pass_top_k":0}}"#).resolve()),
            "search.first_pass_top_k"
        );
    }

    #[test]
    fn merge_keeps_fixed_and_read_only_fields() {
        let current = spec(r#"{"embedding_model":"qwen","chunking":{"window_size":6,"sliding_size":3}}"#)
            .resolve()
            .unwrap();
        // Omitted or repeated (any case) is fine.
        assert_eq!(VectorIndexSpec::default().merge_onto(&current).unwrap(), current);
        assert_eq!(
            spec(r#"{"embedding_model":"QWEN","embedding_dim":768}"#).merge_onto(&current).unwrap(),
            current
        );
        // Any change is rejected and names the field.
        for (json, field) in [
            (r#"{"embedding_model":"gemma"}"#, "embedding_model"),
            (r#"{"embedding_dim":1024}"#, "embedding_dim"),
            (r#"{"chunking":{"window_size":4}}"#, "chunking.window_size"),
            (r#"{"chunking":{"sliding_size":2}}"#, "chunking.sliding_size"),
            (r#"{"quantisation":{"pass2_bits":4}}"#, "quantisation.pass2_bits"),
            (r#"{"quantisation":{"pass1_bits":2}}"#, "quantisation.pass1_bits"),
        ] {
            assert_eq!(err_field(spec(json).merge_onto(&current)), field, "{json}");
        }
        // Search settings change, and are still validated.
        let merged = spec(r#"{"search":{"n_probes":8}}"#).merge_onto(&current).unwrap();
        assert_eq!(merged.search.n_probes, 8);
        assert_eq!(merged.window_size, 6);
        assert_eq!(err_field(spec(r#"{"search":{"top_k":2000}}"#).merge_onto(&current)), "search.top_k");
    }

    #[test]
    fn a_fixed_field_error_says_what_and_why() {
        let current = VectorIndexSpec::default().resolve().unwrap();
        let e = spec(r#"{"chunking":{"window_size":8}}"#).merge_onto(&current).unwrap_err().to_string();
        assert!(e.contains("chunking.window_size") && e.contains('4') && e.contains('8'), "{e}");
        let e = spec(r#"{"chunking":{"window_size":4,"sliding_size":6}}"#)
            .resolve()
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("chunking.sliding_size") && e.contains("window_size (4)") && e.contains('6'),
            "{e}"
        );
    }

    #[test]
    fn search_spec_apply_overrides_only_what_it_sets() {
        let base = SearchSettings::default();
        assert_eq!(SearchSpec::default().apply(base).unwrap(), base);
        let s = SearchSpec {
            n_probes: Some(16),
            ..Default::default()
        }
        .apply(base)
        .unwrap();
        assert_eq!(s, SearchSettings { n_probes: 16, ..base });
        // The cross-field rule is checked on the values in effect.
        assert!(
            SearchSpec {
                first_pass_top_k: Some(10),
                ..Default::default()
            }
            .apply(base)
            .is_err()
        );
        assert!(
            SearchSpec {
                first_pass_top_k: Some(10),
                top_k: Some(10),
                ..Default::default()
            }
            .apply(base)
            .is_ok()
        );
    }

    #[test]
    fn update_search_changes_only_search_and_needs_settings() {
        let current = settle(None, Some(&spec(r#"{"embedding_model":"qwen"}"#))).unwrap();
        let updated = update_search(
            Some(&current),
            "ns",
            &SearchSpec {
                top_k: Some(10),
                ..Default::default()
            },
        )
        .unwrap();
        let (a, b) = (current.resolve().unwrap(), updated.resolve().unwrap());
        assert_eq!(b.search, SearchSettings { top_k: 10, ..a.search });
        assert_eq!(b.embedding_model, a.embedding_model);
        assert!(matches!(
            update_search(
                None,
                "ns",
                &SearchSpec {
                    top_k: Some(10),
                    ..Default::default()
                }
            ),
            Err(SchemaError::VectorIndexNotConfigured { .. })
        ));
        assert!(update_search(Some(&current), "ns", &SearchSpec::default()).is_err());
        assert!(
            update_search(
                Some(&current),
                "ns",
                &SearchSpec {
                    n_probes: Some(0),
                    ..Default::default()
                }
            )
            .is_err()
        );
    }

    #[test]
    fn settle_fills_defaults_first_time_and_merges_after() {
        let first = settle(None, Some(&spec(r#"{"embedding_model":"qwen"}"#))).unwrap();
        assert_eq!(first.chunking.unwrap().window_size, Some(4));
        // A later enable may omit everything, or change only search settings.
        assert_eq!(settle(Some(&first), None).unwrap(), first);
        let again = settle(Some(&first), Some(&spec(r#"{"search":{"n_probes":4}}"#))).unwrap();
        assert_eq!(again.search.unwrap().n_probes, Some(4));
        assert!(settle(Some(&first), Some(&spec(r#"{"embedding_model":"gemma"}"#))).is_err());
    }
}
