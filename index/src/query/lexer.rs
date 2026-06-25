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
        // Accumulate raw bytes, then decode the whole literal as UTF-8 once.
        // Scanning byte-by-byte is safe because the only structural bytes — the
        // quote and `\` — are ASCII and so can never occur *inside* a multi-byte
        // UTF-8 sequence; every other byte (including the bytes of a multi-byte
        // character) is copied verbatim and reassembled by `from_utf8`. The
        // previous `ch as char` per byte mangled non-ASCII text (e.g. Tamil,
        // emoji, accented Latin) into unrelated Latin-1 scalar values.
        let mut buf: Vec<u8> = Vec::new();
        loop {
            match self.advance() {
                None => return Err(QueryError::syntax(start, "unterminated string literal")),
                Some(ch) if ch == quote => break,
                Some(b'\\') => match self.advance() {
                    Some(b'\'') => buf.push(b'\''),
                    Some(b'"') => buf.push(b'"'),
                    Some(b'n') => buf.push(b'\n'),
                    Some(b't') => buf.push(b'\t'),
                    Some(b'\\') => buf.push(b'\\'),
                    Some(ch) => {
                        // Unknown escape: keep the backslash and the byte verbatim.
                        // If `ch` is the lead byte of a multi-byte character, its
                        // continuation bytes follow as ordinary content below and
                        // reassemble correctly.
                        buf.push(b'\\');
                        buf.push(ch);
                    }
                    None => return Err(QueryError::syntax(start, "unterminated escape sequence")),
                },
                Some(ch) => buf.push(ch),
            }
        }
        let s = String::from_utf8(buf).map_err(|_| QueryError::syntax(start, "string literal is not valid UTF-8"))?;
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

    // ── Non-ASCII string literals ──────────────────────────────────────────
    //
    // The lexer must preserve multi-byte UTF-8 content exactly, so a queried
    // string equals the stored JSON string. The previous `ch as char` per byte
    // turned each UTF-8 byte into an unrelated Latin-1 scalar.

    /// Helper: lex a single string literal and return its decoded value.
    fn lex_one_string(src: &str) -> String {
        match tokens(src).into_iter().next() {
            Some(Token::StrLit(s)) => s,
            other => panic!("expected StrLit, got {other:?}"),
        }
    }

    #[test]
    fn test_string_literal_tamil() {
        // மின்னல் ("minnal" — lightning), the project's namesake.
        assert_eq!(lex_one_string("'மின்னல்'"), "மின்னல்");
    }

    #[test]
    fn test_string_literal_emoji() {
        assert_eq!(lex_one_string("'⚡🌩️'"), "⚡🌩️");
    }

    #[test]
    fn test_string_literal_accented_latin() {
        assert_eq!(lex_one_string(r#""café Köln naïve""#), "café Köln naïve");
    }

    #[test]
    fn test_string_literal_mixed_ascii_and_unicode() {
        assert_eq!(lex_one_string("'name: மின்னல் ⚡'"), "name: மின்னல் ⚡");
    }

    #[test]
    fn test_string_literal_unicode_byte_for_byte_roundtrip() {
        // The decoded literal must be byte-identical to the source between quotes.
        let content = "Ωμέγα — Tamil:அ Han:漢 emoji:😀";
        let src = format!("'{content}'");
        assert_eq!(lex_one_string(&src), content);
    }

    #[test]
    fn test_string_literal_escape_before_unicode() {
        // Unknown escape `\ ` then a multi-byte char: backslash kept, char intact.
        assert_eq!(lex_one_string(r"'\மி'"), r"\மி");
    }

    #[test]
    fn test_string_literal_known_escapes_still_work() {
        assert_eq!(lex_one_string(r#"'a\nb\tc\'d\\e'"#), "a\nb\tc'd\\e");
    }
}
