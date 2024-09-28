use thiserror::Error;

#[derive(Debug, Error)]
pub enum LexerError {
    #[error("Got an unexpected token: {0}")]
    UnexpectedToken(String),
    #[error("Got an unexpected character: {0}")]
    UnexpectedChar(char),
}

// todo: let's stop carrying the entire text around
#[derive(Debug)]
pub struct Lexer<'a> {
    tokens: Vec<Token<'a>>,
}

#[derive(Debug, PartialEq)]
pub enum Token<'a> {
    Identifier(&'a str),
    Constant(usize),
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

impl<'a> Lexer<'a> {
    pub fn tokens(&self) -> std::slice::Iter<'a, Token> {
        self.tokens.iter()
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
                        return Err(Self::error_string(&bytes[start..end]));
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
                _ => return Err(LexerError::UnexpectedChar(bytes[idx].into()))
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
}
