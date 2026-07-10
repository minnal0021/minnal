/// A filter applied to a single indexed field.
///
/// `V` is the field's value type; it must implement [`Ord`] so that range
/// predicates (`Lt`, `Le`, `Gt`, `Ge`, `Between`) can be evaluated efficiently
/// against the underlying `BTreeMap`.
#[derive(Debug, Clone, PartialEq)]
pub enum Predicate<V> {
    /// Rows whose field value equals `V`.
    Eq(V),
    /// Rows whose field value is anything other than `V`.
    Ne(V),
    /// Rows whose field value is strictly less than `V`.
    Lt(V),
    /// Rows whose field value is less than or equal to `V`.
    Le(V),
    /// Rows whose field value is strictly greater than `V`.
    Gt(V),
    /// Rows whose field value is greater than or equal to `V`.
    Ge(V),
    /// Rows whose field value falls in the inclusive range `[lo, hi]`.
    Between {
        /// Lower bound (inclusive).
        lo: V,
        /// Upper bound (inclusive).
        hi: V,
    },
    /// Rows whose field value is any element of `values`.
    In(Vec<V>),
}
