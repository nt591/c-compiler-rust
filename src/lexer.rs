use thiserror::Error;

#[derive(Debug, Error)]
pub enum LexerError {
    #[error("Got an unexpected token: {0}")]
    UnexpectedToken(String),
    #[error("Got an unexpected character: {0}")]
    UnexpectedChar(char),
    #[error("Never closed a block comment")]
    UnclosedBlockComment,
}

// todo: let's stop carrying the entire text around
#[derive(Debug)]
pub struct Lexer<'a> {
    tokens: Vec<Token<'a>>,
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Token<'a> {
    Identifier(&'a str),
    Constant(usize),
    SingleLineComment(&'a str),
    BlockComment(&'a str),
    // keywords
    Int,
    Main,
    Void,
    Return,
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    Semicolon,
}

impl<'a> Token<'a> {
    pub fn into_string(&self) -> String {
        use Token::*;
        match self {
            Identifier(s) => format!("Identifier {s}"),
            Constant(c) => format!("Constant {c}"),
            SingleLineComment(c) => format!("SingleLineComment {c}"),
            BlockComment(c) => format!("BlockComment {c}"),
            Int => format!("Int"),
            Main => format!("Main"),
            Void => format!("Void"),
            Return => format!("Return"),
            LeftParen => format!("LeftParen"),
            RightParen => format!("RightParen"),
            LeftBrace => format!("LeftBrace"),
            RightBrace => format!("RightBrace"),
            Semicolon => format!("Semicolon"),
        }
    }
}

impl<'a> Lexer<'a> {
    pub fn tokens(&self) -> std::slice::Iter<'a, Token> {
        self.tokens.iter()
    }

    pub fn as_syntactic_tokens(&self) -> Vec<Token<'a>> {
        self.tokens.iter().filter(|x| !matches!(x, 
            Token::SingleLineComment(_) | Token::BlockComment(_) 
        )).copied().collect::<Vec<_>>()
    }

    pub fn lex(source: &'a str) -> Result<Lexer<'a>, LexerError> {
        let mut tokens = Vec::new();
        let mut idx = 0;
        let len = source.len();
        // todo: I guess emojis will blow up here
        let bytes = source.as_bytes();
        while idx < len {
            let c = bytes[idx];
            match c {
                c if c.is_ascii_whitespace() => (),
                b'{' => tokens.push(Token::LeftBrace),
                b'}' => tokens.push(Token::RightBrace),
                b'(' => tokens.push(Token::LeftParen),
                b')' => tokens.push(Token::RightParen),
                b';' => tokens.push(Token::Semicolon),
                b'/' if idx < len - 1 && bytes[idx + 1] == b'/' => {
                    // single line comment handler
                    // Scoop up until we see a newline or end of index, stuff into token
                    // Start after the slashes
                    let start = idx + 2;
                    let mut end = start;
                    while end < len && bytes[end] != b'\n' {
                        end += 1;
                    }
                    let comment =
                        std::str::from_utf8(&bytes[start..end]).expect("We know this is UTF8");
                    tokens.push(Token::SingleLineComment(comment));
                    idx = end - 1;
                }
                b'/' if idx < len - 1 && bytes[idx + 1] == b'*' => {
                    // block comment. This is tricky.
                    // Let's just scoop up until we see *
                    // and check if the character after is a slash.
                    // Once we see this, we'll capture the inside.
                    // We'll sanity check that we actually closed the comment.
                    let start = idx + 2;
                    let mut end = start;
                    while end < len {
                        // annoying, but we check if we're looking at *, if the next
                        // index is in range AND it's a slash. Gotta tighten this up.
                        if bytes[end] == b'*' && end + 1 < len && bytes[end + 1] == b'/' {
                            break;
                        }
                        end += 1;
                    }
                    if end >= len {
                        // we never closed the comment!
                        return Err(LexerError::UnclosedBlockComment);
                    }
                    let comment =
                        std::str::from_utf8(&bytes[start..end]).expect("We know this is UTF8");
                    tokens.push(Token::BlockComment(comment));
                    idx = end + 1; // move to the closing slash, we'll incr again
                }
                b'a'..=b'z' | b'A'..=b'Z' => {
                    // starts with a letter, just walk until the end.
                    let start = idx;
                    let mut end = idx;
                    while end < len && Self::is_valid_identifier_character(bytes[end]) {
                        end += 1;
                    }

                    // scoop up whatever's in the range,
                    // check for known keywords, else just
                    // return an identifier? Or just ignore.
                    if let Some(keyword_token) = Self::parse_keyword(&source[start..end]) {
                        tokens.push(keyword_token);
                    } else {
                        let content =
                            std::str::from_utf8(&bytes[start..end]).expect("We know this is UTF8");
                        tokens.push(Token::Identifier(content));
                    }
                    idx = end - 1; //since we're incrementing at the end
                }
                c if c.is_ascii_digit() => {
                    // advance while we see digits
                    let start = idx;
                    let mut end = idx;

                    while end < len && bytes[end].is_ascii_digit() {
                        end += 1;
                    }
                    // If end == len, we parsed the last character. In that
                    // case, let's just check that if idx < len, the next character
                    // is not an alphabetic character.
                    // This does not support floats!
                    if end < len && bytes[end].is_ascii_alphabetic() {
                        return Err(Self::error_string(&bytes[start..=end]));
                    }

                    // todo: is this the best I can do?
                    let s = std::str::from_utf8(&bytes[start..end]).expect("We know this is UTF8");
                    let constant: usize = s
                        .parse()
                        .expect("Didn't get a digit after parsing a string");
                    tokens.push(Token::Constant(constant));
                    idx = end - 1;
                }
                _ => return Err(LexerError::UnexpectedChar(bytes[idx].into())),
            }
            idx += 1;
        }

        Ok(Lexer { tokens })
    }

    fn is_valid_identifier_character(c: u8) -> bool {
        matches!(c, b'a'..=b'z' | b'A'..=b'Z'| b'0'..=b'9')
    }

    fn parse_keyword(s: &str) -> Option<Token> {
        match s {
            "int" => Some(Token::Int),
            "main" => Some(Token::Main),
            "return" => Some(Token::Return),
            "void" => Some(Token::Void),
            _ => None,
        }
    }

    fn error_string(v: &[u8]) -> LexerError {
        LexerError::UnexpectedToken(
            String::from_utf8(v.to_vec())
                .expect("got invalid utf-8 when addressing byte range from source string"),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_keywords() {
        let source = "int main 123;";
        let lexer = Lexer::lex(source);
        assert!(lexer.is_ok());
        let lexer = lexer.unwrap();
        let mut tokens = lexer.tokens();
        assert_eq!(Some(&Token::Int), tokens.next());
        assert_eq!(Some(&Token::Main), tokens.next());
        assert_eq!(Some(&Token::Constant(123)), tokens.next());
        assert_eq!(Some(&Token::Semicolon), tokens.next());
        assert_eq!(None, tokens.next());
    }

    #[test]
    fn basic_source() {
        let source = r#"
        int main(void) {
            return 2;
        }"#;
        let lexer = Lexer::lex(source);
        assert!(lexer.is_ok());
        let lexer = lexer.unwrap();
        let mut tokens = lexer.tokens();
        assert_eq!(Some(&Token::Int), tokens.next());
        assert_eq!(Some(&Token::Main), tokens.next());
        assert_eq!(Some(&Token::LeftParen), tokens.next());
        assert_eq!(Some(&Token::Void), tokens.next());
        assert_eq!(Some(&Token::RightParen), tokens.next());
        assert_eq!(Some(&Token::LeftBrace), tokens.next());
        assert_eq!(Some(&Token::Return), tokens.next());
        assert_eq!(Some(&Token::Constant(2)), tokens.next());
        assert_eq!(Some(&Token::Semicolon), tokens.next());
        assert_eq!(Some(&Token::RightBrace), tokens.next());
        assert_eq!(None, tokens.next());
    }

    #[test]
    fn error_backtick() {
        let source = "`";
        let lexer = Lexer::lex(source);
        assert!(lexer.is_err());
    }

    #[test]
    fn error_at() {
        let source = r#"
        int main(void) {
            return 0@1;
        };"#;
        let lexer = Lexer::lex(source);
        assert!(lexer.is_err());
    }

    #[test]
    fn success_multi_digit_constant() {
        let source = r#"
        int main(void) {
            // test case w/ multi-digit constant
            return 100;
        }"#;
        let lexer = Lexer::lex(source);
        assert!(lexer.is_ok());
    }

    #[test]
    fn success_single_line_comment() {
        let source = r#"
        // lorem ipsum
        "#;
        let lexer = Lexer::lex(source);
        assert!(lexer.is_ok());
        let lexer = lexer.unwrap();
        let mut tokens = lexer.tokens();
        assert_eq!(
            Some(&Token::SingleLineComment(" lorem ipsum")),
            tokens.next()
        );
        assert_eq!(None, tokens.next());
    }
    #[test]
    fn success_block_comment() {
        let source = r#"
/* lorem ipsum
bloop blorp */
        "#;
        let lexer = Lexer::lex(source);
        assert!(lexer.is_ok());
        let lexer = lexer.unwrap();
        let mut tokens = lexer.tokens();
        assert_eq!(
            Some(&Token::BlockComment(" lorem ipsum\nbloop blorp ")),
            tokens.next()
        );
        assert_eq!(None, tokens.next());
    }
}
