/// Query error type for parsing and evaluation failures.
#[derive(Debug, thiserror::Error)]
pub enum QueryError {
    /// Tokenizer or grammar error.
    #[error("syntax error at position {pos}: {msg}")]
    Syntax { pos: usize, msg: String },

    /// Field name not found in the namespace schema.
    #[error("unknown field '{name}'")]
    UnknownField { name: String },

    /// Field is registered in the schema but has no active in-memory index
    /// (call `activate_field_index` first).
    #[error("field '{field}' has no active index; call activate_field_index first")]
    InactiveField { field: String },

    /// Literal type in the query does not match the index value type.
    #[error("type mismatch on field '{field}': index holds {expected} values, query supplied {got}")]
    TypeMismatch {
        field: String,
        expected: &'static str,
        got: &'static str,
    },

    /// Operator is not valid for the field's value type (e.g. `<` on bool).
    #[error("operator '{op}' is not supported for {ty} fields")]
    UnsupportedOp { op: String, ty: &'static str },

    /// IN list is syntactically present but empty.
    #[error("IN value list must not be empty")]
    EmptyInList,
}

impl QueryError {
    /// Convenience constructor for syntax errors.
    pub(super) fn syntax(pos: usize, msg: impl Into<String>) -> Self {
        Self::Syntax { pos, msg: msg.into() }
    }
}
