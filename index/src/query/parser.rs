use super::error::QueryError;
use super::lexer::{Lexer, Token};

// ── Public AST types ───────────────────────────────────────────────────────

/// Comparison operator extracted from the query string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    In,
}

impl std::fmt::Display for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Op::Eq => "=",
            Op::Ne => "!=",
            Op::Lt => "<",
            Op::Le => "<=",
            Op::Gt => ">",
            Op::Ge => ">=",
            Op::In => "IN",
        };
        f.write_str(s)
    }
}

/// A scalar or list value literal from the query string.
///
/// The type is unresolved at this stage; coercion to the index's concrete type
/// (`bool`, `i64`, `String`) happens in the evaluator.
#[derive(Debug, Clone, PartialEq)]
pub enum RawValue {
    Str(String),
    Int(i64),
    Bool(bool),
    /// Used exclusively by the `IN` operator.
    List(Vec<RawValue>),
}

/// A parsed (but not yet evaluated) query expression.
///
/// Operator precedence is the conventional boolean ordering — `NOT` binds
/// tightest, then `AND`, then `OR` binds loosest — so an unparenthesised
/// `a = 1 OR b = 2 AND c = 3` parses as `a = 1 OR (b = 2 AND c = 3)`, and
/// `NOT a = 1 AND b = 2` as `(NOT a = 1) AND b = 2`. `AND` and `OR` are each
/// left-associative; parentheses override grouping.
///
/// ```text
/// query     = or_expr EOF
/// or_expr   = and_expr ( "OR" and_expr )*      // OR binds loosest
/// and_expr  = term ( "AND" term )*             // AND binds tighter than OR
/// term      = "NOT" term | "(" or_expr ")" | predicate   // NOT binds tightest
/// predicate = FIELD OP VALUE
/// OP        = "=" | "!=" | "<" | "<=" | ">" | ">=" | "IN"
/// VALUE     = string | integer | bool | "(" value_list ")"
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum RawExpr {
    And(Box<RawExpr>, Box<RawExpr>),
    Or(Box<RawExpr>, Box<RawExpr>),
    Not(Box<RawExpr>),
    Predicate { field: String, op: Op, value: RawValue },
}

// ── Parser ─────────────────────────────────────────────────────────────────

/// Parse a query string into a [`RawExpr`].
///
/// Returns a `QueryError::Syntax` if the input does not conform to the grammar.
pub fn parse(input: &str) -> Result<RawExpr, QueryError> {
    let mut p = Parser::new(input)?;
    let expr = p.parse_expr()?;
    if p.current != Token::Eof {
        return Err(QueryError::syntax(
            p.tok_pos,
            format!("unexpected token after expression: {:?}", p.current),
        ));
    }
    Ok(expr)
}

// ── Internal parser state ──────────────────────────────────────────────────

struct Parser<'a> {
    lexer: Lexer<'a>,
    current: Token,
    /// Byte offset of `current` in the source string (for error messages).
    tok_pos: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Result<Self, QueryError> {
        let mut lexer = Lexer::new(input);
        let tok_pos = lexer.pos();
        let current = lexer.next_token()?;
        Ok(Self { lexer, current, tok_pos })
    }

    /// Consume the current token and return it, advancing to the next.
    fn advance(&mut self) -> Result<Token, QueryError> {
        self.tok_pos = self.lexer.pos();
        let next = self.lexer.next_token()?;
        Ok(std::mem::replace(&mut self.current, next))
    }

    // ── Grammar rules ──────────────────────────────────────────────────

    /// expr = or_expr  (entry point; OR is the loosest-binding operator)
    fn parse_expr(&mut self) -> Result<RawExpr, QueryError> {
        self.parse_or()
    }

    /// or_expr = and_expr ( "OR" and_expr )*
    ///
    /// `OR` has the lowest precedence and is left-associative, so each operand
    /// is a full `AND` chain — `a AND b OR c AND d` groups as
    /// `(a AND b) OR (c AND d)`.
    fn parse_or(&mut self) -> Result<RawExpr, QueryError> {
        let mut left = self.parse_and()?;
        while self.current == Token::Or {
            self.advance()?;
            let right = self.parse_and()?;
            left = RawExpr::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    /// and_expr = term ( "AND" term )*
    ///
    /// `AND` binds tighter than `OR` and is left-associative. Parentheses (via
    /// `parse_term`) override grouping.
    fn parse_and(&mut self) -> Result<RawExpr, QueryError> {
        let mut left = self.parse_term()?;
        while self.current == Token::And {
            self.advance()?;
            let right = self.parse_term()?;
            left = RawExpr::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    /// term = "NOT" term | "(" expr ")" | predicate
    fn parse_term(&mut self) -> Result<RawExpr, QueryError> {
        match &self.current.clone() {
            Token::Not => {
                self.advance()?;
                let inner = self.parse_term()?;
                Ok(RawExpr::Not(Box::new(inner)))
            }
            Token::LParen => {
                self.advance()?;
                let expr = self.parse_expr()?;
                if self.current != Token::RParen {
                    return Err(QueryError::syntax(self.tok_pos, "expected ')'"));
                }
                self.advance()?;
                Ok(expr)
            }
            Token::Ident(_) => self.parse_predicate(),
            other => Err(QueryError::syntax(
                self.tok_pos,
                format!("expected field name, NOT, or '(', got {:?}", other),
            )),
        }
    }

    /// predicate = FIELD OP VALUE
    fn parse_predicate(&mut self) -> Result<RawExpr, QueryError> {
        let field = match self.advance()? {
            Token::Ident(name) => name,
            tok => return Err(QueryError::syntax(self.tok_pos, format!("expected field name, got {:?}", tok))),
        };

        let op = match self.advance()? {
            Token::Eq => Op::Eq,
            Token::Ne => Op::Ne,
            Token::Lt => Op::Lt,
            Token::Le => Op::Le,
            Token::Gt => Op::Gt,
            Token::Ge => Op::Ge,
            Token::In => Op::In,
            tok => {
                return Err(QueryError::syntax(
                    self.tok_pos,
                    format!("expected comparison operator (=, !=, <, <=, >, >=, IN), got {:?}", tok),
                ));
            }
        };

        let value = if op == Op::In { self.parse_in_list()? } else { self.parse_scalar()? };

        Ok(RawExpr::Predicate { field, op, value })
    }

    /// IN "(" scalar ("," scalar)* ")"
    fn parse_in_list(&mut self) -> Result<RawValue, QueryError> {
        if self.current != Token::LParen {
            return Err(QueryError::syntax(self.tok_pos, "IN requires a parenthesised list: IN (val1, val2, ...)"));
        }
        self.advance()?;

        let mut items = Vec::new();
        loop {
            items.push(self.parse_scalar()?);
            match &self.current {
                Token::Comma => {
                    self.advance()?;
                }
                Token::RParen => {
                    self.advance()?;
                    break;
                }
                other => {
                    return Err(QueryError::syntax(
                        self.tok_pos,
                        format!("expected ',' or ')' in IN list, got {:?}", other),
                    ));
                }
            }
        }

        if items.is_empty() {
            return Err(QueryError::EmptyInList);
        }
        Ok(RawValue::List(items))
    }

    /// scalar = string | integer | bool
    fn parse_scalar(&mut self) -> Result<RawValue, QueryError> {
        match self.advance()? {
            Token::StrLit(s) => Ok(RawValue::Str(s)),
            Token::IntLit(n) => Ok(RawValue::Int(n)),
            Token::BoolLit(b) => Ok(RawValue::Bool(b)),
            tok => Err(QueryError::syntax(
                self.tok_pos,
                format!("expected a string, integer, or bool literal, got {:?}", tok),
            )),
        }
    }
}

// ── tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_int_eq_predicate_produces_predicate_ast_node() {
        let expr = parse("age = 30").unwrap();
        assert_eq!(
            expr,
            RawExpr::Predicate {
                field: "age".into(),
                op: Op::Eq,
                value: RawValue::Int(30),
            }
        );
    }

    #[test]
    fn test_string_ne() {
        let expr = parse("status != 'active'").unwrap();
        assert_eq!(
            expr,
            RawExpr::Predicate {
                field: "status".into(),
                op: Op::Ne,
                value: RawValue::Str("active".into()),
            }
        );
    }

    #[test]
    fn parse_and_expression_produces_and_node() {
        let expr = parse("age > 18 AND active = true").unwrap();
        assert!(matches!(expr, RawExpr::And(_, _)));
    }

    #[test]
    fn parse_or_expression_produces_or_node() {
        let expr = parse("age < 10 OR age > 90").unwrap();
        assert!(matches!(expr, RawExpr::Or(_, _)));
    }

    #[test]
    fn parse_not_expression_produces_not_node() {
        let expr = parse("NOT active = true").unwrap();
        assert!(matches!(expr, RawExpr::Not(_)));
    }

    #[test]
    fn test_parentheses() {
        // (a = 1 OR b = 2) AND c = 3
        let expr = parse("(a = 1 OR b = 2) AND c = 3").unwrap();
        assert!(matches!(expr, RawExpr::And(_, _)));
        if let RawExpr::And(left, _) = expr {
            assert!(matches!(*left, RawExpr::Or(_, _)));
        }
    }

    #[test]
    fn test_in_list() {
        let expr = parse("status IN ('a', 'b', 'c')").unwrap();
        assert_eq!(
            expr,
            RawExpr::Predicate {
                field: "status".into(),
                op: Op::In,
                value: RawValue::List(vec![RawValue::Str("a".into()), RawValue::Str("b".into()), RawValue::Str("c".into()),]),
            }
        );
    }

    #[test]
    fn test_in_integers() {
        let expr = parse("code IN (1, 2, 3)").unwrap();
        assert!(matches!(expr, RawExpr::Predicate { op: Op::In, .. }));
    }

    #[test]
    fn test_not_not() {
        let expr = parse("NOT NOT active = true").unwrap();
        assert!(matches!(expr, RawExpr::Not(_)));
        if let RawExpr::Not(inner) = expr {
            assert!(matches!(*inner, RawExpr::Not(_)));
        }
    }

    #[test]
    fn test_trailing_token_error() {
        assert!(parse("age = 1 extra").is_err());
    }

    #[test]
    fn test_missing_value_error() {
        assert!(parse("age =").is_err());
    }

    #[test]
    fn test_empty_in_list_error() {
        // Parser can't produce an empty list since it requires at least one scalar
        // before RParen, but the grammar check is there for safety.
        // "IN ()" hits parse_scalar before RParen so it's a syntax error.
        assert!(parse("status IN ()").is_err());
    }

    #[test]
    fn test_missing_rparen() {
        assert!(parse("(age = 1").is_err());
    }

    #[test]
    fn test_bool_predicate() {
        let expr = parse("enabled = true").unwrap();
        assert_eq!(
            expr,
            RawExpr::Predicate {
                field: "enabled".into(),
                op: Op::Eq,
                value: RawValue::Bool(true),
            }
        );
    }

    #[test]
    fn test_negative_int() {
        let expr = parse("balance > -100").unwrap();
        assert_eq!(
            expr,
            RawExpr::Predicate {
                field: "balance".into(),
                op: Op::Gt,
                value: RawValue::Int(-100),
            }
        );
    }

    #[test]
    fn test_complex_and_or() {
        // a = 1 AND b = 2 AND c = 3 — left-associative, so ((a AND b) AND c)
        let expr = parse("a = 1 AND b = 2 AND c = 3").unwrap();
        if let RawExpr::And(left, right) = &expr {
            assert!(matches!(**left, RawExpr::And(_, _)));
            assert!(matches!(**right, RawExpr::Predicate { .. }));
        } else {
            panic!("expected And at top level");
        }
    }

    // ── Operator precedence: NOT > AND > OR ─────────────────────────────────

    fn pred(field: &str, n: i64) -> RawExpr {
        RawExpr::Predicate {
            field: field.into(),
            op: Op::Eq,
            value: RawValue::Int(n),
        }
    }

    #[test]
    fn and_binds_tighter_than_or_on_the_right() {
        // a = 1 OR b = 2 AND c = 3  ==>  a = 1 OR (b = 2 AND c = 3)
        let expr = parse("a = 1 OR b = 2 AND c = 3").unwrap();
        let expected = RawExpr::Or(
            Box::new(pred("a", 1)),
            Box::new(RawExpr::And(Box::new(pred("b", 2)), Box::new(pred("c", 3)))),
        );
        assert_eq!(expr, expected);
    }

    #[test]
    fn and_binds_tighter_than_or_on_the_left() {
        // a = 1 AND b = 2 OR c = 3  ==>  (a = 1 AND b = 2) OR c = 3
        let expr = parse("a = 1 AND b = 2 OR c = 3").unwrap();
        let expected = RawExpr::Or(
            Box::new(RawExpr::And(Box::new(pred("a", 1)), Box::new(pred("b", 2)))),
            Box::new(pred("c", 3)),
        );
        assert_eq!(expr, expected);
    }

    #[test]
    fn and_chains_group_under_each_or_operand() {
        // a=1 AND b=2 OR c=3 AND d=4  ==>  (a AND b) OR (c AND d)
        let expr = parse("a = 1 AND b = 2 OR c = 3 AND d = 4").unwrap();
        let expected = RawExpr::Or(
            Box::new(RawExpr::And(Box::new(pred("a", 1)), Box::new(pred("b", 2)))),
            Box::new(RawExpr::And(Box::new(pred("c", 3)), Box::new(pred("d", 4)))),
        );
        assert_eq!(expr, expected);
    }

    #[test]
    fn or_is_left_associative() {
        // a = 1 OR b = 2 OR c = 3  ==>  ((a OR b) OR c)
        let expr = parse("a = 1 OR b = 2 OR c = 3").unwrap();
        let expected = RawExpr::Or(
            Box::new(RawExpr::Or(Box::new(pred("a", 1)), Box::new(pred("b", 2)))),
            Box::new(pred("c", 3)),
        );
        assert_eq!(expr, expected);
    }

    #[test]
    fn not_binds_tighter_than_and() {
        // NOT a = 1 AND b = 2  ==>  (NOT a = 1) AND b = 2
        let expr = parse("NOT a = 1 AND b = 2").unwrap();
        let expected = RawExpr::And(Box::new(RawExpr::Not(Box::new(pred("a", 1)))), Box::new(pred("b", 2)));
        assert_eq!(expr, expected);
    }

    #[test]
    fn not_binds_tighter_than_or() {
        // NOT a = 1 OR b = 2  ==>  (NOT a = 1) OR b = 2
        let expr = parse("NOT a = 1 OR b = 2").unwrap();
        let expected = RawExpr::Or(Box::new(RawExpr::Not(Box::new(pred("a", 1)))), Box::new(pred("b", 2)));
        assert_eq!(expr, expected);
    }

    #[test]
    fn parens_override_precedence() {
        // (a = 1 OR b = 2) AND c = 3  ==>  (a OR b) AND c  — explicit grouping wins
        let expr = parse("(a = 1 OR b = 2) AND c = 3").unwrap();
        let expected = RawExpr::And(
            Box::new(RawExpr::Or(Box::new(pred("a", 1)), Box::new(pred("b", 2)))),
            Box::new(pred("c", 3)),
        );
        assert_eq!(expr, expected);
    }
}
