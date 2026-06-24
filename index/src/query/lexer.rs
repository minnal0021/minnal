use super::error::QueryError;

/// A single token produced by the lexer.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Literals
    Ident(String),
    StrLit(String),
    IntLit(i64),
    BoolLit(bool),
    // Comparison operators
    Eq, // =
    Ne, // !=
    Lt, // <
    Le, // <=
    Gt, // >
    Ge, // >=
    // Keywords
    And,
    Or,
    Not,
    In,
    // Punctuation
    LParen,
    RParen,
    Comma,
    // Sentinel
    Eof,
}

/// Byte-level tokenizer over a UTF-8 query string.
pub struct Lexer<'a> {
    input: &'a [u8],
    pos: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            input: input.as_bytes(),
            pos: 0,
        }
    }

    /// Current byte offset (useful for error reporting).
    pub fn pos(&self) -> usize {
        self.pos
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        let ch = self.peek();
        if ch.is_some() {
            self.pos += 1;
        }
        ch
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.advance();
        }
    }

    pub fn next_token(&mut self) -> Result<Token, QueryError> {
        self.skip_whitespace();
        let start = self.pos;

        match self.peek() {
            None => Ok(Token::Eof),
            Some(b'(') => {
                self.advance();
                Ok(Token::LParen)
            }
            Some(b')') => {
                self.advance();
                Ok(Token::RParen)
            }
            Some(b',') => {
                self.advance();
                Ok(Token::Comma)
            }
            Some(b'=') => {
                self.advance();
                Ok(Token::Eq)
            }
            Some(b'!') => {
                self.advance();
                if self.peek() == Some(b'=') {
                    self.advance();
                    Ok(Token::Ne)
                } else {
                    Err(QueryError::syntax(start, "expected '=' after '!'"))
                }
            }
            Some(b'<') => {
                self.advance();
                if self.peek() == Some(b'=') {
                    self.advance();
                    Ok(Token::Le)
                } else {
                    Ok(Token::Lt)
                }
            }
            Some(b'>') => {
                self.advance();
                if self.peek() == Some(b'=') {
                    self.advance();
                    Ok(Token::Ge)
                } else {
                    Ok(Token::Gt)
                }
            }
            Some(b'\'' | b'"') => self.lex_string(start),
            Some(b'-') | Some(b'0'..=b'9') => self.lex_number(start),
            Some(b'a'..=b'z') | Some(b'A'..=b'Z') | Some(b'_') => self.lex_ident(),
            Some(ch) => Err(QueryError::syntax(start, format!("unexpected character '{}'", ch as char))),
        }
    }

    fn lex_string(&mut self, start: usize) -> Result<Token, QueryError> {
        let quote = self.advance().unwrap();
        let mut s = String::new();
        loop {
            match self.advance() {
                None => return Err(QueryError::syntax(start, "unterminated string literal")),
                Some(ch) if ch == quote => break,
                Some(b'\\') => match self.advance() {
                    Some(b'\'') => s.push('\''),
                    Some(b'"') => s.push('"'),
                    Some(b'n') => s.push('\n'),
                    Some(b't') => s.push('\t'),
                    Some(b'\\') => s.push('\\'),
                    Some(ch) => {
                        s.push('\\');
                        s.push(ch as char);
                    }
                    None => return Err(QueryError::syntax(start, "unterminated escape sequence")),
                },
                Some(ch) => s.push(ch as char),
            }
        }
        Ok(Token::StrLit(s))
    }

    fn lex_number(&mut self, start: usize) -> Result<Token, QueryError> {
        let neg = self.peek() == Some(b'-');
        if neg {
            self.advance();
        }

        let digit_start = self.pos;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.advance();
        }

        let digits = std::str::from_utf8(&self.input[digit_start..self.pos]).unwrap();
        if digits.is_empty() {
            return Err(QueryError::syntax(start, "expected digit after '-'"));
        }

        let n: i64 = digits
            .parse()
            .map_err(|_| QueryError::syntax(start, "integer literal out of i64 range"))?;
        Ok(Token::IntLit(if neg { -n } else { n }))
    }

    fn lex_ident(&mut self) -> Result<Token, QueryError> {
        let ident_start = self.pos;
        while matches!(self.peek(), Some(b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_')) {
            self.advance();
        }
        let word = std::str::from_utf8(&self.input[ident_start..self.pos]).unwrap();
        let tok = match word.to_ascii_uppercase().as_str() {
            "AND" => Token::And,
            "OR" => Token::Or,
            "NOT" => Token::Not,
            "IN" => Token::In,
            "TRUE" => Token::BoolLit(true),
            "FALSE" => Token::BoolLit(false),
            _ => Token::Ident(word.to_string()),
        };
        Ok(tok)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(input: &str) -> Vec<Token> {
        let mut lexer = Lexer::new(input);
        let mut out = Vec::new();
        loop {
            let tok = lexer.next_token().unwrap();
            let done = tok == Token::Eof;
            out.push(tok);
            if done {
                break;
            }
        }
        out
    }

    #[test]
    fn int_eq_expression_tokenizes_to_ident_eq_int_eof() {
        assert_eq!(
            tokens("age = 30"),
            vec![Token::Ident("age".into()), Token::Eq, Token::IntLit(30), Token::Eof]
        );
    }

    #[test]
    fn test_operators() {
        assert_eq!(tokens("!=")[0], Token::Ne);
        assert_eq!(tokens("<=")[0], Token::Le);
        assert_eq!(tokens(">=")[0], Token::Ge);
        assert_eq!(tokens("<")[0], Token::Lt);
        assert_eq!(tokens(">")[0], Token::Gt);
    }

    #[test]
    fn test_keywords_case_insensitive() {
        assert_eq!(tokens("and")[0], Token::And);
        assert_eq!(tokens("OR")[0], Token::Or);
        assert_eq!(tokens("Not")[0], Token::Not);
        assert_eq!(tokens("in")[0], Token::In);
        assert_eq!(tokens("true")[0], Token::BoolLit(true));
        assert_eq!(tokens("FALSE")[0], Token::BoolLit(false));
    }

    #[test]
    fn test_string_literal() {
        assert_eq!(tokens(r#""hello""#)[0], Token::StrLit("hello".into()));
        assert_eq!(tokens("'world'")[0], Token::StrLit("world".into()));
    }

    #[test]
    fn test_negative_integer() {
        assert_eq!(tokens("-42")[0], Token::IntLit(-42));
    }

    #[test]
    fn test_in_list() {
        assert_eq!(
            tokens("status IN ('a', 'b')"),
            vec![
                Token::Ident("status".into()),
                Token::In,
                Token::LParen,
                Token::StrLit("a".into()),
                Token::Comma,
                Token::StrLit("b".into()),
                Token::RParen,
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_bad_bang() {
        assert!(Lexer::new("a ! b").next_token().is_ok()); // 'a' is fine
        let mut lex = Lexer::new("!b");
        assert!(lex.next_token().is_err());
    }
}
