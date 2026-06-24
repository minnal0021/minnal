//! Recursive-descent query parser and evaluator.
//!
//! # Grammar
//!
//! ```text
//! query     = expr EOF
//! expr      = term ( ("AND" | "OR") term )*
//! term      = "NOT" term | "(" expr ")" | predicate
//! predicate = FIELD OP VALUE
//! OP        = "=" | "!=" | "<" | "<=" | ">" | ">=" | "IN"
//! VALUE     = string | integer | bool | "(" value_list ")"
//! ```
//!
//! Keywords (`AND`, `OR`, `NOT`, `IN`, `TRUE`, `FALSE`) are case-insensitive.
//! String literals may be single- or double-quoted.
//!
//! # Quick start
//!
//! ```ignore
//! use index::query::{parse_and_evaluate, SchemaMap};
//!
//! // Build SchemaMap from NamespaceSchema::list_fields()
//! let schema: SchemaMap = [("age".to_string(), 0u32)].into();
//!
//! // Provide a closure that looks up the live DynFieldIndex by FieldId
//! let bitmap = parse_and_evaluate("age > 30 AND age < 50", &schema, &|id| {
//!     // return Some(Arc<RwLock<DynFieldIndex>>) or None
//!     todo!()
//! }).unwrap();
//! ```

mod error;
mod eval;
mod lexer;
mod parser;

pub use error::QueryError;
pub use eval::{SchemaMap, evaluate, parse_and_evaluate};
pub use parser::{Op, RawExpr, RawValue, parse};
