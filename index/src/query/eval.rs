use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;

use crate::RoaringBitmap;
use crate::field::FieldId;
use crate::field::predicate::Predicate;
use crate::field::value::{DynFieldIndex, DynFieldIndexInner};

use super::error::QueryError;
use super::parser::{Op, RawExpr, RawValue};

// ── Public types ───────────────────────────────────────────────────────────

/// Maps a field name to its `FieldId` within a namespace.
///
/// Build this from [`NamespaceSchema::list_fields`] on the `minnal_db` side:
///
/// ```ignore
/// let schema_map: SchemaMap = registry
///     .schema(ns_id)
///     .map(|s| s.list_fields().into_iter().map(|f| (f.field_name, f.field_id)).collect())
///     .unwrap_or_default();
/// ```
pub type SchemaMap = HashMap<String, FieldId>;

// ── Top-level API ──────────────────────────────────────────────────────────

/// Parse `input` and evaluate it against the live field indices.
///
/// # Arguments
/// * `input`     — the query string, e.g. `"age > 30 AND status = 'active'"`
/// * `schema`    — maps field names to their registered `FieldId`s
/// * `get_index` — returns the live `DynFieldIndex` for a given `FieldId`,
///   or `None` if that field has no active index
///
/// # Returns
/// A [`RoaringBitmap`] of row IDs that satisfy the query.  An empty bitmap
/// means no rows matched (not an error).
pub fn parse_and_evaluate<F>(input: &str, schema: &SchemaMap, get_index: &F) -> Result<RoaringBitmap, QueryError>
where
    F: Fn(FieldId) -> Option<Arc<RwLock<DynFieldIndex>>>,
{
    let expr = super::parser::parse(input)?;
    evaluate(&expr, schema, get_index)
}

/// Evaluate a pre-parsed [`RawExpr`] against the live field indices.
///
/// Use this when you want to parse once and evaluate multiple times, or when
/// you build the expression tree programmatically.
///
/// The **entire** AST is validated (every field known and active, every value
/// type- and op-compatible) *before* any bitmap evaluation runs. This makes
/// query validity independent of the data: an invalid predicate is reported as
/// a [`QueryError`] even if an AND short-circuits on an empty left operand, so a
/// typo / inactive field / type mismatch can never be silently hidden behind an
/// empty result.
pub fn evaluate<F>(expr: &RawExpr, schema: &SchemaMap, get_index: &F) -> Result<RoaringBitmap, QueryError>
where
    F: Fn(FieldId) -> Option<Arc<RwLock<DynFieldIndex>>>,
{
    validate(expr, schema, get_index)?;
    eval_expr(expr, schema, get_index)
}

/// Recursive evaluation worker. Assumes [`validate`] has already accepted the
/// whole AST, so the AND short-circuit below cannot hide an invalid operand.
fn eval_expr<F>(expr: &RawExpr, schema: &SchemaMap, get_index: &F) -> Result<RoaringBitmap, QueryError>
where
    F: Fn(FieldId) -> Option<Arc<RwLock<DynFieldIndex>>>,
{
    match expr {
        RawExpr::And(left, right) => {
            let l = eval_expr(left, schema, get_index)?;
            if l.is_empty() {
                // Short-circuit: AND with empty is always empty. Safe to skip the
                // right operand's *evaluation* because the full AST was already
                // validated up front, so a bad right operand has already errored.
                return Ok(l);
            }
            let r = eval_expr(right, schema, get_index)?;
            Ok(l.and(&r))
        }

        RawExpr::Or(left, right) => {
            let l = eval_expr(left, schema, get_index)?;
            let r = eval_expr(right, schema, get_index)?;
            Ok(l.or(&r))
        }

        RawExpr::Not(inner) => {
            let inner_bm = eval_expr(inner, schema, get_index)?;
            // Universe = OR of bitmaps only for fields referenced by the inner
            // expression.  Rows that have no value for any of those fields are
            // not candidates for NOT, which matches document-store semantics:
            // `NOT status = 'active'` should return rows that *have* a status
            // value that isn't 'active', not every row in the database.
            let mut universe = RoaringBitmap::new();
            let mut referenced_ids = std::collections::HashSet::new();
            collect_field_ids(inner, schema, &mut referenced_ids);
            for field_id in referenced_ids {
                if let Some(idx_lock) = get_index(field_id) {
                    idx_lock.read().for_each_bitmap(|bm| universe.or_inplace(bm));
                }
            }
            Ok(universe.and_not(&inner_bm))
        }

        RawExpr::Predicate { field, op, value } => eval_predicate(field, op, value, schema, get_index),
    }
}

// ── Validation pass ────────────────────────────────────────────────────────

/// Walk the whole AST and check that every predicate is well-formed against the
/// schema and live indices — field known ([`QueryError::UnknownField`]), field
/// active ([`QueryError::InactiveField`]), value type-compatible
/// ([`QueryError::TypeMismatch`]) and op supported
/// ([`QueryError::UnsupportedOp`]) — **without** evaluating any bitmaps. Runs
/// once before [`eval_expr`] so short-circuiting can never hide an invalid leaf.
fn validate<F>(expr: &RawExpr, schema: &SchemaMap, get_index: &F) -> Result<(), QueryError>
where
    F: Fn(FieldId) -> Option<Arc<RwLock<DynFieldIndex>>>,
{
    match expr {
        RawExpr::And(left, right) | RawExpr::Or(left, right) => {
            validate(left, schema, get_index)?;
            validate(right, schema, get_index)
        }
        RawExpr::Not(inner) => validate(inner, schema, get_index),
        RawExpr::Predicate { field, op, value } => validate_predicate(field, op, value, schema, get_index),
    }
}

fn validate_predicate<F>(field: &str, op: &Op, raw_value: &RawValue, schema: &SchemaMap, get_index: &F) -> Result<(), QueryError>
where
    F: Fn(FieldId) -> Option<Arc<RwLock<DynFieldIndex>>>,
{
    let &field_id = schema.get(field).ok_or_else(|| QueryError::UnknownField { name: field.to_string() })?;
    let idx_lock = get_index(field_id).ok_or_else(|| QueryError::InactiveField { field: field.to_string() })?;
    let idx = idx_lock.read();
    // Build the typed predicate to surface any type/op error, then discard it —
    // the same builders `eval_dyn` uses, so validation can never drift from it.
    match &idx.inner {
        DynFieldIndexInner::Bool(_) => build_bool_predicate(op, raw_value, field).map(|_| ()),
        DynFieldIndexInner::Int(_) => build_int_predicate(op, raw_value, field).map(|_| ()),
        DynFieldIndexInner::Str(_) => build_str_predicate(op, raw_value, field).map(|_| ()),
    }
}

// ── Field-reference collection ─────────────────────────────────────────────

/// Walk `expr` and insert the `FieldId` of every `Predicate` leaf into `out`.
///
/// Used by the `Not` arm of [`evaluate`] to build a focused universe that
/// includes only the fields actually queried, not every field in the schema.
fn collect_field_ids(expr: &RawExpr, schema: &SchemaMap, out: &mut std::collections::HashSet<FieldId>) {
    match expr {
        RawExpr::And(l, r) | RawExpr::Or(l, r) => {
            collect_field_ids(l, schema, out);
            collect_field_ids(r, schema, out);
        }
        RawExpr::Not(inner) => collect_field_ids(inner, schema, out),
        RawExpr::Predicate { field, .. } => {
            if let Some(&id) = schema.get(field.as_str()) {
                out.insert(id);
            }
        }
    }
}

// ── Predicate evaluation ───────────────────────────────────────────────────

fn eval_predicate<F>(field: &str, op: &Op, raw_value: &RawValue, schema: &SchemaMap, get_index: &F) -> Result<RoaringBitmap, QueryError>
where
    F: Fn(FieldId) -> Option<Arc<RwLock<DynFieldIndex>>>,
{
    let &field_id = schema.get(field).ok_or_else(|| QueryError::UnknownField { name: field.to_string() })?;

    let idx_lock = get_index(field_id).ok_or_else(|| QueryError::InactiveField { field: field.to_string() })?;

    let idx = idx_lock.read();
    eval_dyn(&idx, field, op, raw_value)
}

fn eval_dyn(idx: &DynFieldIndex, field: &str, op: &Op, raw: &RawValue) -> Result<RoaringBitmap, QueryError> {
    match &idx.inner {
        DynFieldIndexInner::Bool(inner) => Ok(inner.evaluate(&build_bool_predicate(op, raw, field)?)),
        DynFieldIndexInner::Int(inner) => Ok(inner.evaluate(&build_int_predicate(op, raw, field)?)),
        DynFieldIndexInner::Str(inner) => Ok(inner.evaluate(&build_str_predicate(op, raw, field)?)),
    }
}

/// Build a typed `bool` predicate from the raw op/value, dispatching `IN` to a
/// list coercion and everything else to a scalar coercion. Shared by `eval_dyn`
/// (which evaluates the predicate) and `validate_predicate` (which discards it).
fn build_bool_predicate(op: &Op, raw: &RawValue, field: &str) -> Result<Predicate<bool>, QueryError> {
    if *op == Op::In {
        Ok(Predicate::In(coerce_bool_list(raw, field)?))
    } else {
        build_bool_pred(op, coerce_bool(raw, field)?, field)
    }
}

fn build_int_predicate(op: &Op, raw: &RawValue, field: &str) -> Result<Predicate<i64>, QueryError> {
    if *op == Op::In {
        Ok(Predicate::In(coerce_int_list(raw, field)?))
    } else {
        build_int_pred(op, coerce_int(raw, field)?, field)
    }
}

fn build_str_predicate(op: &Op, raw: &RawValue, field: &str) -> Result<Predicate<String>, QueryError> {
    if *op == Op::In {
        Ok(Predicate::In(coerce_str_list(raw, field)?))
    } else {
        build_str_pred(op, coerce_str(raw, field)?, field)
    }
}

// ── Type coercion helpers ──────────────────────────────────────────────────

fn raw_type_name(v: &RawValue) -> &'static str {
    match v {
        RawValue::Bool(_) => "bool",
        RawValue::Int(_) => "integer",
        RawValue::Str(_) => "string",
        RawValue::List(_) => "list",
    }
}

fn coerce_bool(v: &RawValue, field: &str) -> Result<bool, QueryError> {
    match v {
        RawValue::Bool(b) => Ok(*b),
        _ => Err(QueryError::TypeMismatch {
            field: field.to_string(),
            expected: "bool",
            got: raw_type_name(v),
        }),
    }
}

fn coerce_int(v: &RawValue, field: &str) -> Result<i64, QueryError> {
    match v {
        RawValue::Int(n) => Ok(*n),
        _ => Err(QueryError::TypeMismatch {
            field: field.to_string(),
            expected: "integer",
            got: raw_type_name(v),
        }),
    }
}

fn coerce_str(v: &RawValue, field: &str) -> Result<String, QueryError> {
    match v {
        RawValue::Str(s) => Ok(s.clone()),
        _ => Err(QueryError::TypeMismatch {
            field: field.to_string(),
            expected: "string",
            got: raw_type_name(v),
        }),
    }
}

fn coerce_bool_list(v: &RawValue, field: &str) -> Result<Vec<bool>, QueryError> {
    match v {
        RawValue::List(items) => items.iter().map(|i| coerce_bool(i, field)).collect(),
        _ => Err(QueryError::TypeMismatch {
            field: field.to_string(),
            expected: "bool list",
            got: raw_type_name(v),
        }),
    }
}

fn coerce_int_list(v: &RawValue, field: &str) -> Result<Vec<i64>, QueryError> {
    match v {
        RawValue::List(items) => items.iter().map(|i| coerce_int(i, field)).collect(),
        _ => Err(QueryError::TypeMismatch {
            field: field.to_string(),
            expected: "integer list",
            got: raw_type_name(v),
        }),
    }
}

fn coerce_str_list(v: &RawValue, field: &str) -> Result<Vec<String>, QueryError> {
    match v {
        RawValue::List(items) => items.iter().map(|i| coerce_str(i, field)).collect(),
        _ => Err(QueryError::TypeMismatch {
            field: field.to_string(),
            expected: "string list",
            got: raw_type_name(v),
        }),
    }
}

// ── Predicate builders ─────────────────────────────────────────────────────

fn build_bool_pred(op: &Op, v: bool, _field: &str) -> Result<Predicate<bool>, QueryError> {
    match op {
        Op::Eq => Ok(Predicate::Eq(v)),
        Op::Ne => Ok(Predicate::Ne(v)),
        _ => Err(QueryError::UnsupportedOp {
            op: op.to_string(),
            ty: "bool",
        }),
    }
}

fn build_int_pred(op: &Op, v: i64, field: &str) -> Result<Predicate<i64>, QueryError> {
    let _ = field;
    match op {
        Op::Eq => Ok(Predicate::Eq(v)),
        Op::Ne => Ok(Predicate::Ne(v)),
        Op::Lt => Ok(Predicate::Lt(v)),
        Op::Le => Ok(Predicate::Le(v)),
        Op::Gt => Ok(Predicate::Gt(v)),
        Op::Ge => Ok(Predicate::Ge(v)),
        Op::In => unreachable!("IN is handled before build_int_pred"),
    }
}

fn build_str_pred(op: &Op, v: String, field: &str) -> Result<Predicate<String>, QueryError> {
    let _ = field;
    match op {
        Op::Eq => Ok(Predicate::Eq(v)),
        Op::Ne => Ok(Predicate::Ne(v)),
        Op::Lt => Ok(Predicate::Lt(v)),
        Op::Le => Ok(Predicate::Le(v)),
        Op::Gt => Ok(Predicate::Gt(v)),
        Op::Ge => Ok(Predicate::Ge(v)),
        Op::In => unreachable!("IN is handled before build_str_pred"),
    }
}

// ── tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::value::{DynFieldIndex, IndexValue, IndexValueType};

    fn make_int_index(entries: &[(i64, u128)]) -> Arc<RwLock<DynFieldIndex>> {
        let mut idx = DynFieldIndex::new(IndexValueType::Int);
        for &(v, row) in entries {
            idx.insert(&IndexValue::Int(v), row).unwrap();
        }
        Arc::new(RwLock::new(idx))
    }

    fn make_str_index(entries: &[(&str, u128)]) -> Arc<RwLock<DynFieldIndex>> {
        let mut idx = DynFieldIndex::new(IndexValueType::Str);
        for &(v, row) in entries {
            idx.insert(&IndexValue::Str(v.into()), row).unwrap();
        }
        Arc::new(RwLock::new(idx))
    }

    fn make_bool_index(entries: &[(bool, u128)]) -> Arc<RwLock<DynFieldIndex>> {
        let mut idx = DynFieldIndex::new(IndexValueType::Bool);
        for &(v, row) in entries {
            idx.insert(&IndexValue::Bool(v), row).unwrap();
        }
        Arc::new(RwLock::new(idx))
    }

    fn schema(fields: &[(&str, u32)]) -> SchemaMap {
        fields.iter().map(|&(n, id)| (n.to_string(), id)).collect()
    }

    #[test]
    fn test_int_eq() {
        let age_idx = make_int_index(&[(25, 1), (30, 2), (25, 3)]);
        let s = schema(&[("age", 0)]);
        let get_index = |id: u32| if id == 0 { Some(Arc::clone(&age_idx)) } else { None };

        let bm = parse_and_evaluate("age = 25", &s, &get_index).unwrap();
        let rows: Vec<u128> = bm.iter().collect();
        assert_eq!(rows, vec![1, 3]);
    }

    #[test]
    fn test_int_gt() {
        let age_idx = make_int_index(&[(10, 1), (20, 2), (30, 3)]);
        let s = schema(&[("age", 0)]);
        let get_index = |id: u32| if id == 0 { Some(Arc::clone(&age_idx)) } else { None };

        let bm = parse_and_evaluate("age > 15", &s, &get_index).unwrap();
        let rows: Vec<u128> = bm.iter().collect();
        assert_eq!(rows, vec![2, 3]);
    }

    #[test]
    fn test_str_eq() {
        let status_idx = make_str_index(&[("active", 1), ("inactive", 2), ("active", 3)]);
        let s = schema(&[("status", 0)]);
        let get_index = |id: u32| if id == 0 { Some(Arc::clone(&status_idx)) } else { None };

        let bm = parse_and_evaluate("status = 'active'", &s, &get_index).unwrap();
        let rows: Vec<u128> = bm.iter().collect();
        assert_eq!(rows, vec![1, 3]);
    }

    #[test]
    fn test_str_eq_non_ascii_unicode() {
        // End-to-end: a non-ASCII value indexed as stored must match when queried
        // with the same literal — broken when the lexer mangled UTF-8 byte-wise.
        let name_idx = make_str_index(&[("மின்னல்", 1), ("lightning", 2), ("⚡", 3), ("மின்னல்", 4)]);
        let s = schema(&[("name", 0)]);
        let get_index = |id: u32| if id == 0 { Some(Arc::clone(&name_idx)) } else { None };

        let bm = parse_and_evaluate("name = 'மின்னல்'", &s, &get_index).unwrap();
        assert_eq!(bm.iter().collect::<Vec<u128>>(), vec![1, 4]);

        let bm = parse_and_evaluate("name = '⚡'", &s, &get_index).unwrap();
        assert_eq!(bm.iter().collect::<Vec<u128>>(), vec![3]);
    }

    #[test]
    fn test_bool_eq() {
        let active_idx = make_bool_index(&[(true, 1), (false, 2), (true, 3)]);
        let s = schema(&[("active", 0)]);
        let get_index = |id: u32| if id == 0 { Some(Arc::clone(&active_idx)) } else { None };

        let bm = parse_and_evaluate("active = true", &s, &get_index).unwrap();
        let rows: Vec<u128> = bm.iter().collect();
        assert_eq!(rows, vec![1, 3]);
    }

    #[test]
    fn and_intersects_rows_matching_both_predicates() {
        let age_idx = make_int_index(&[(25, 1), (30, 2), (25, 3)]);
        let status_idx = make_str_index(&[("active", 1), ("active", 2), ("inactive", 3)]);
        let s = schema(&[("age", 0), ("status", 1)]);
        let get_index = |id: u32| match id {
            0 => Some(Arc::clone(&age_idx)),
            1 => Some(Arc::clone(&status_idx)),
            _ => None,
        };

        // age = 25 AND status = 'active' → only row 1
        let bm = parse_and_evaluate("age = 25 AND status = 'active'", &s, &get_index).unwrap();
        let rows: Vec<u128> = bm.iter().collect();
        assert_eq!(rows, vec![1]);
    }

    #[test]
    fn or_unions_rows_matching_either_predicate() {
        let age_idx = make_int_index(&[(10, 1), (50, 2), (80, 3)]);
        let s = schema(&[("age", 0)]);
        let get_index = |id: u32| if id == 0 { Some(Arc::clone(&age_idx)) } else { None };

        let bm = parse_and_evaluate("age < 20 OR age > 70", &s, &get_index).unwrap();
        let rows: Vec<u128> = bm.iter().collect();
        assert_eq!(rows, vec![1, 3]);
    }

    #[test]
    fn not_returns_rows_not_matching_predicate() {
        let status_idx = make_str_index(&[("active", 1), ("inactive", 2), ("active", 3)]);
        let s = schema(&[("status", 0)]);
        let get_index = |id: u32| if id == 0 { Some(Arc::clone(&status_idx)) } else { None };

        let bm = parse_and_evaluate("NOT status = 'active'", &s, &get_index).unwrap();
        let rows: Vec<u128> = bm.iter().collect();
        assert_eq!(rows, vec![2]);
    }

    #[test]
    fn test_in_integers() {
        let age_idx = make_int_index(&[(10, 1), (20, 2), (30, 3), (40, 4)]);
        let s = schema(&[("age", 0)]);
        let get_index = |id: u32| if id == 0 { Some(Arc::clone(&age_idx)) } else { None };

        let bm = parse_and_evaluate("age IN (10, 30)", &s, &get_index).unwrap();
        let rows: Vec<u128> = bm.iter().collect();
        assert_eq!(rows, vec![1, 3]);
    }

    #[test]
    fn test_in_strings() {
        let status_idx = make_str_index(&[("a", 1), ("b", 2), ("c", 3)]);
        let s = schema(&[("status", 0)]);
        let get_index = |id: u32| if id == 0 { Some(Arc::clone(&status_idx)) } else { None };

        let bm = parse_and_evaluate("status IN ('a', 'c')", &s, &get_index).unwrap();
        let rows: Vec<u128> = bm.iter().collect();
        assert_eq!(rows, vec![1, 3]);
    }

    #[test]
    fn test_unknown_field_error() {
        let s: SchemaMap = HashMap::new();
        let get_index = |_: u32| None;
        let err = parse_and_evaluate("missing = 1", &s, &get_index).unwrap_err();
        assert!(matches!(err, QueryError::UnknownField { .. }));
    }

    #[test]
    fn test_inactive_field_error() {
        let s = schema(&[("age", 0)]);
        let get_index = |_: u32| None; // no active index
        let err = parse_and_evaluate("age = 1", &s, &get_index).unwrap_err();
        assert!(matches!(err, QueryError::InactiveField { .. }));
    }

    #[test]
    fn test_type_mismatch_error() {
        let age_idx = make_int_index(&[(25, 1)]);
        let s = schema(&[("age", 0)]);
        let get_index = |id: u32| if id == 0 { Some(Arc::clone(&age_idx)) } else { None };
        // Passing a string to an int field
        let err = parse_and_evaluate("age = 'hello'", &s, &get_index).unwrap_err();
        assert!(matches!(err, QueryError::TypeMismatch { .. }));
    }

    #[test]
    fn test_bool_unsupported_op() {
        let active_idx = make_bool_index(&[(true, 1)]);
        let s = schema(&[("active", 0)]);
        let get_index = |id: u32| if id == 0 { Some(Arc::clone(&active_idx)) } else { None };
        let err = parse_and_evaluate("active > true", &s, &get_index).unwrap_err();
        assert!(matches!(err, QueryError::UnsupportedOp { .. }));
    }

    #[test]
    fn test_and_short_circuit_returns_empty_for_valid_query() {
        // A fully-valid query whose left operand matches nothing still returns an
        // empty bitmap (the AND short-circuit), without erroring.
        let age_idx = make_int_index(&[(99, 1)]);
        let active_idx = make_bool_index(&[(true, 1)]);
        let s = schema(&[("age", 0), ("active", 1)]);
        let get_index = |id: u32| match id {
            0 => Some(Arc::clone(&age_idx)),
            1 => Some(Arc::clone(&active_idx)),
            _ => None,
        };
        // age = 5 matches nothing; right operand is valid → Ok(empty).
        let bm = parse_and_evaluate("age = 5 AND active = true", &s, &get_index).unwrap();
        assert!(bm.is_empty());
    }

    // ── Validation runs over the whole AST, regardless of short-circuiting ──────
    //
    // Even when the left operand of an AND matches nothing (so the right operand's
    // bitmap is never computed), an invalid right operand must still error — query
    // validity is data-independent. Regression for the AND short-circuit that used
    // to hide unknown/inactive/type-invalid/unsupported right-hand predicates.

    #[test]
    fn test_and_empty_left_still_errors_on_unknown_right_field() {
        let age_idx = make_int_index(&[(99, 1)]);
        let s = schema(&[("age", 0)]); // `missing` is not in the schema
        let get_index = |id: u32| if id == 0 { Some(Arc::clone(&age_idx)) } else { None };
        // age = 5 is empty, but `missing` is an unknown field → UnknownField.
        let err = parse_and_evaluate("age = 5 AND missing = 1", &s, &get_index).unwrap_err();
        assert!(matches!(err, QueryError::UnknownField { .. }), "got: {err:?}");
    }

    #[test]
    fn test_and_empty_left_still_errors_on_inactive_right_field() {
        let age_idx = make_int_index(&[(99, 1)]);
        // `status` is a known field but has no active index.
        let s = schema(&[("age", 0), ("status", 1)]);
        let get_index = |id: u32| if id == 0 { Some(Arc::clone(&age_idx)) } else { None };
        let err = parse_and_evaluate("age = 5 AND status = 'x'", &s, &get_index).unwrap_err();
        assert!(matches!(err, QueryError::InactiveField { .. }), "got: {err:?}");
    }

    #[test]
    fn test_and_empty_left_still_errors_on_type_mismatch_right() {
        let age_idx = make_int_index(&[(99, 1)]);
        let label_idx = make_str_index(&[("a", 1)]);
        let s = schema(&[("age", 0), ("label", 1)]);
        let get_index = |id: u32| match id {
            0 => Some(Arc::clone(&age_idx)),
            1 => Some(Arc::clone(&label_idx)),
            _ => None,
        };
        // age = 5 is empty; `label` is a string field but the value is an int.
        let err = parse_and_evaluate("age = 5 AND label = 1", &s, &get_index).unwrap_err();
        assert!(matches!(err, QueryError::TypeMismatch { .. }), "got: {err:?}");
    }

    #[test]
    fn test_and_empty_left_still_errors_on_unsupported_op_right() {
        let age_idx = make_int_index(&[(99, 1)]);
        let active_idx = make_bool_index(&[(true, 1)]);
        let s = schema(&[("age", 0), ("active", 1)]);
        let get_index = |id: u32| match id {
            0 => Some(Arc::clone(&age_idx)),
            1 => Some(Arc::clone(&active_idx)),
            _ => None,
        };
        // age = 5 is empty; `active > true` is an unsupported op on a bool field.
        let err = parse_and_evaluate("age = 5 AND active > true", &s, &get_index).unwrap_err();
        assert!(matches!(err, QueryError::UnsupportedOp { .. }), "got: {err:?}");
    }

    #[test]
    fn test_nested_and_empty_left_still_errors_on_deep_invalid_right() {
        // Short-circuit also must not hide an invalid predicate nested deeper in
        // the right subtree: age = 5 AND (active = true AND missing = 1).
        let age_idx = make_int_index(&[(99, 1)]);
        let active_idx = make_bool_index(&[(true, 1)]);
        let s = schema(&[("age", 0), ("active", 1)]);
        let get_index = |id: u32| match id {
            0 => Some(Arc::clone(&age_idx)),
            1 => Some(Arc::clone(&active_idx)),
            _ => None,
        };
        let err = parse_and_evaluate("age = 5 AND (active = true AND missing = 1)", &s, &get_index).unwrap_err();
        assert!(matches!(err, QueryError::UnknownField { .. }), "got: {err:?}");
    }

    #[test]
    fn test_grouped_or_and() {
        // (age < 20 OR age > 80) AND active = true
        let age_idx = make_int_index(&[(10, 1), (50, 2), (90, 3)]);
        let active_idx = make_bool_index(&[(true, 1), (true, 2), (true, 3)]);
        let s = schema(&[("age", 0), ("active", 1)]);
        let get_index = |id: u32| match id {
            0 => Some(Arc::clone(&age_idx)),
            1 => Some(Arc::clone(&active_idx)),
            _ => None,
        };
        let bm = parse_and_evaluate("(age < 20 OR age > 80) AND active = true", &s, &get_index).unwrap();
        let rows: Vec<u128> = bm.iter().collect();
        assert_eq!(rows, vec![1, 3]);
    }

    // ── NOT universe tests ─────────────────────────────────────────────────────

    #[test]
    fn test_not_universe_scoped_to_referenced_field() {
        // Row 4 exists only in age_idx, not in status_idx.
        // NOT status = 'active' should NOT include row 4 — it has no status.
        let status_idx = make_str_index(&[("active", 1), ("inactive", 2), ("active", 3)]);
        let age_idx = make_int_index(&[(25, 1), (30, 2), (25, 3), (40, 4)]);
        let s = schema(&[("status", 0), ("age", 1)]);
        let get_index = |id: u32| match id {
            0 => Some(Arc::clone(&status_idx)),
            1 => Some(Arc::clone(&age_idx)),
            _ => None,
        };
        let bm = parse_and_evaluate("NOT status = 'active'", &s, &get_index).unwrap();
        let rows: Vec<u128> = bm.iter().collect();
        // Only row 2 has a non-active status; row 4 lacks status entirely.
        assert_eq!(rows, vec![2]);
    }

    #[test]
    fn test_not_compound_inner_collects_all_referenced_fields() {
        // NOT (status = 'active' AND age > 18)
        // Universe = rows with any status OR any age value = {1,2,3,4}
        // Inner match = rows where status='active' AND age>18
        let status_idx = make_str_index(&[("active", 1), ("inactive", 2), ("active", 3)]);
        let age_idx = make_int_index(&[(25, 1), (15, 2), (30, 3), (22, 4)]);
        let s = schema(&[("status", 0), ("age", 1)]);
        let get_index = |id: u32| match id {
            0 => Some(Arc::clone(&status_idx)),
            1 => Some(Arc::clone(&age_idx)),
            _ => None,
        };
        // status='active' AND age>18: rows 1 (active,25) and 3 (active,30) match
        // Universe (status OR age) = {1,2,3,4}
        // NOT = {2, 4}
        let bm = parse_and_evaluate("NOT (status = 'active' AND age > 18)", &s, &get_index).unwrap();
        let rows: Vec<u128> = bm.iter().collect();
        assert_eq!(rows, vec![2, 4]);
    }

    #[test]
    fn test_not_nested_not_collects_inner_fields() {
        // NOT (NOT status = 'active') is semantically active rows
        let status_idx = make_str_index(&[("active", 1), ("inactive", 2), ("active", 3)]);
        let s = schema(&[("status", 0)]);
        let get_index = |id: u32| if id == 0 { Some(Arc::clone(&status_idx)) } else { None };
        let bm = parse_and_evaluate("NOT (NOT status = 'active')", &s, &get_index).unwrap();
        let rows: Vec<u128> = bm.iter().collect();
        assert_eq!(rows, vec![1, 3]);
    }

    #[test]
    fn test_not_empty_inner_returns_full_universe() {
        // NOT on something that matches nothing → full universe
        let status_idx = make_str_index(&[("active", 1), ("inactive", 2), ("active", 3)]);
        let s = schema(&[("status", 0)]);
        let get_index = |id: u32| if id == 0 { Some(Arc::clone(&status_idx)) } else { None };
        let bm = parse_and_evaluate("NOT status = 'missing'", &s, &get_index).unwrap();
        let rows: Vec<u128> = bm.iter().collect();
        // "missing" matches no rows, so NOT = all rows with a status value
        assert_eq!(rows, vec![1, 2, 3]);
    }

    // ── Predicate coverage ─────────────────────────────────────────────────────

    #[test]
    fn test_int_lt() {
        let age_idx = make_int_index(&[(10, 1), (20, 2), (30, 3)]);
        let s = schema(&[("age", 0)]);
        let get_index = |id: u32| if id == 0 { Some(Arc::clone(&age_idx)) } else { None };
        let bm = parse_and_evaluate("age < 25", &s, &get_index).unwrap();
        let rows: Vec<u128> = bm.iter().collect();
        assert_eq!(rows, vec![1, 2]);
    }

    #[test]
    fn test_int_le() {
        let age_idx = make_int_index(&[(10, 1), (20, 2), (30, 3)]);
        let s = schema(&[("age", 0)]);
        let get_index = |id: u32| if id == 0 { Some(Arc::clone(&age_idx)) } else { None };
        let bm = parse_and_evaluate("age <= 20", &s, &get_index).unwrap();
        let rows: Vec<u128> = bm.iter().collect();
        assert_eq!(rows, vec![1, 2]);
    }

    #[test]
    fn test_int_ge() {
        let age_idx = make_int_index(&[(10, 1), (20, 2), (30, 3)]);
        let s = schema(&[("age", 0)]);
        let get_index = |id: u32| if id == 0 { Some(Arc::clone(&age_idx)) } else { None };
        let bm = parse_and_evaluate("age >= 20", &s, &get_index).unwrap();
        let rows: Vec<u128> = bm.iter().collect();
        assert_eq!(rows, vec![2, 3]);
    }

    #[test]
    fn test_int_ne() {
        let age_idx = make_int_index(&[(10, 1), (20, 2), (10, 3)]);
        let s = schema(&[("age", 0)]);
        let get_index = |id: u32| if id == 0 { Some(Arc::clone(&age_idx)) } else { None };
        let bm = parse_and_evaluate("age != 10", &s, &get_index).unwrap();
        let rows: Vec<u128> = bm.iter().collect();
        assert_eq!(rows, vec![2]);
    }

    #[test]
    fn test_str_ne() {
        let s_idx = make_str_index(&[("a", 1), ("b", 2), ("a", 3)]);
        let s = schema(&[("label", 0)]);
        let get_index = |id: u32| if id == 0 { Some(Arc::clone(&s_idx)) } else { None };
        let bm = parse_and_evaluate("label != 'a'", &s, &get_index).unwrap();
        let rows: Vec<u128> = bm.iter().collect();
        assert_eq!(rows, vec![2]);
    }

    #[test]
    fn test_bool_ne() {
        let idx = make_bool_index(&[(true, 1), (false, 2), (true, 3)]);
        let s = schema(&[("active", 0)]);
        let get_index = |id: u32| if id == 0 { Some(Arc::clone(&idx)) } else { None };
        let bm = parse_and_evaluate("active != true", &s, &get_index).unwrap();
        let rows: Vec<u128> = bm.iter().collect();
        assert_eq!(rows, vec![2]);
    }
}
