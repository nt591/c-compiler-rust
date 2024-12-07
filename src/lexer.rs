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
    PreprocessorBlock(&'a str), //todo
    // keywords
    Int,
    Main,
    Void,
    Return,
    // special symbols
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    Semicolon,
    Tilde,
    Hyphen,
    DoubleHyphen,
    Plus,
    Star,
    Slash,
    Percent,
    // bitwise
    Ampersand,
    Pipe,
    Caret,
    LessThanLessThan,
    GreaterThanGreaterThan,
    // Chapter 4 logical ops
    Bang,
    AmpersandAmpersand,
    PipePipe,
    EqualEqual,
    BangEqual,
    LessThan,
    GreaterThan,
    LessThanEqual,
    GreaterThanEqual,
    // assignment and variables
    Equal,
    Underscore,
    // chapter 5 compound assignment
    PlusEqual,
    HyphenEqual,
    StarEqual,
    SlashEqual,
    PercentEqual,
    AmpersandEqual,
    PipeEqual,
    CaratEqual,
    LessThanLessThanEqual,
    GreaterThanGreaterThanEqual,
    // increment/decrement
    PlusPlus,
}

impl<'a> Token<'a> {
    pub fn into_string(&self) -> String {
        use Token::*;
        match self {
            Identifier(s) => format!("Identifier {s}"),
            Constant(c) => format!("Constant {c}"),
            SingleLineComment(c) => format!("SingleLineComment {c}"),
            BlockComment(c) => format!("BlockComment {c}"),
            PreprocessorBlock(c) => format!("PreprocessorBlock {c}"),
            Int => format!("Int"),
            Main => format!("Main"),
            Void => format!("Void"),
            Return => format!("Return"),
            LeftParen => format!("LeftParen"),
            RightParen => format!("RightParen"),
            LeftBrace => format!("LeftBrace"),
            RightBrace => format!("RightBrace"),
            Semicolon => format!("Semicolon"),
            Tilde => format!("Tilde"),
            Hyphen => format!("Hyphen"),
            DoubleHyphen => format!("DoubleHyphen"),
            Plus => format!("Plus"),
            Star => format!("Star"),
            Slash => format!("Slash"),
            Percent => format!("Percent"),
            Ampersand => format!("Ampersand"),
            Pipe => format!("Pipe"),
            Caret => format!("Caret"),
            LessThanLessThan => format!("LessThanLessThan"),
            GreaterThanGreaterThan => format!("GreaterThanGreaterThan"),
            Bang => format!("Bang"),
            AmpersandAmpersand => format!("AmpersandAmpersand"),
            PipePipe => format!("PipePipe"),
            Equal => format!("Equal"),
            Underscore => format!("Underscore"),
            EqualEqual => format!("EqualEqual"),
            BangEqual => format!("BangEqual"),
            LessThan => format!("LessEqual"),
            GreaterThan => format!("GreaterThan"),
            LessThanEqual => format!("LessThanEqual"),
            GreaterThanEqual => format!("GreaterThanEqual"),
            PlusEqual => format!("PlusEqual"),
            HyphenEqual => format!("HyphenEqual"),
            StarEqual => format!("StarEqual"),
            SlashEqual => format!("SlashEqual"),
            PercentEqual => format!("PercentEqual"),
            AmpersandEqual => format!("AmpersandEqual"),
            PipeEqual => format!("PipeEqual"),
            CaratEqual => format!("CaratEqual"),
            LessThanLessThanEqual => format!("LessThanLessThanEqual"),
            GreaterThanGreaterThanEqual => format!("GreaterThanGreaterThanEqual"),
            PlusPlus => format!("PlusPlus"),
        }
    }
}

impl<'a> Lexer<'a> {
    #[cfg(test)] // maybe prefer allow dead code
    pub fn tokens(&self) -> std::slice::Iter<'a, Token> {
        self.tokens.iter()
    }

    pub fn as_syntactic_tokens(&self) -> Vec<Token<'a>> {
        self.tokens
            .iter()
            .filter(|x| {
                !matches!(
                    x,
                    Token::SingleLineComment(_)
                        | Token::BlockComment(_)
                        | Token::PreprocessorBlock(_)
                )
            })
            .copied()
            .collect::<Vec<_>>()
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
                b'~' => tokens.push(Token::Tilde),
                b'+' => {
                    if idx < len - 1 && bytes[idx + 1] == b'=' {
                        tokens.push(Token::PlusEqual);
                        idx += 1;
                    } else if idx < len - 1 && bytes[idx + 1] == b'+' {
                        tokens.push(Token::PlusPlus);
                        idx += 1;
                        }
                        tokens.push(Token::Plus);
                }
                b'*' => {
                    if idx < len - 1 && bytes[idx + 1] == b'=' {
                        tokens.push(Token::StarEqual);
                        idx += 1;
                    } else {
                        tokens.push(Token::Star);
                    }
                }
                b'%' => {
                    if idx < len - 1 && bytes[idx + 1] == b'=' {
                        tokens.push(Token::PercentEqual);
                        idx += 1;
                    } else {
                        tokens.push(Token::Percent);
                    }
                }
                b'_' => tokens.push(Token::Underscore),
                b'&' => {
                    if idx < len - 1 && bytes[idx + 1] == b'=' {
                        tokens.push(Token::AmpersandEqual);
                        idx += 1;
                    } else if idx < len - 1 && bytes[idx + 1] == b'&' {
                        tokens.push(Token::AmpersandAmpersand);
                        idx += 1;
                    } else {
                        tokens.push(Token::Ampersand);
                    }
                }
                b'|' => {
                    if idx < len - 1 && bytes[idx + 1] == b'|' {
                        tokens.push(Token::PipePipe);
                        idx += 2;
                        continue;
                    }
                    if idx < len - 1 && bytes[idx + 1] == b'=' {
                        tokens.push(Token::PipeEqual);
                        idx += 2;
                        continue;
                    }
                    tokens.push(Token::Pipe);
                }
                b'^' => {
                    if idx < len - 1 && bytes[idx + 1] == b'=' {
                        tokens.push(Token::CaratEqual);
                        idx += 2;
                        continue;
                    }

                    tokens.push(Token::Caret);
                }
                b'-' => {
                    // first, check if we could be processing a double hyphen
                    if idx < len - 1 && bytes[idx + 1] == b'-' {
                        idx += 1;
                        tokens.push(Token::DoubleHyphen);
                    } else if idx < len - 1 && bytes[idx + 1] == b'=' {
                        idx += 1;
                        tokens.push(Token::HyphenEqual);
                    } else {
                        tokens.push(Token::Hyphen);
                    }
                }
                b'/' => {
                    if idx < len - 1 && bytes[idx + 1] == b'=' {
                        idx += 2;
                        tokens.push(Token::SlashEqual);
                        continue;
                    } else if idx < len - 1 && bytes[idx + 1] == b'/' {
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
                        // we continue the loop and miss the idx increment at the bottom of the
                        // match
                        idx = end;
                        continue;
                    } else if idx < len - 1 && bytes[idx + 1] == b'*' {
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
                        // we continue the loop and miss the idx increment at the bottom of the
                        // match
                        idx = end + 2;
                        continue;
                    }
                    tokens.push(Token::Slash);
                }
                b'#' => {
                    // these are ifdefs, pragmas, etc. Let's for now just ignore
                    // we falsely claim these to be a special Token
                    let start = idx + 1;
                    let mut end = start;
                    while end < len && bytes[end] != b'\n' {
                        end += 1;
                    }
                    let comment =
                        std::str::from_utf8(&bytes[start..end]).expect("We know this is UTF8");
                    tokens.push(Token::PreprocessorBlock(comment));
                    idx = end - 1;
                }
                b'=' => {
                    if idx < len - 1 && bytes[idx + 1] == b'=' {
                        tokens.push(Token::EqualEqual);
                        idx += 2;
                        continue;
                    }
                    tokens.push(Token::Equal);
                }
                b'!' => {
                    if idx < len - 1 && bytes[idx + 1] == b'=' {
                        tokens.push(Token::BangEqual);
                        idx += 2;
                        continue;
                    }
                    tokens.push(Token::Bang);
                }
                b'>' => {
                    if idx < len - 1 && bytes[idx + 1] == b'>' {
                        if idx < len - 2 && bytes[idx + 2] == b'=' {
                            tokens.push(Token::GreaterThanGreaterThanEqual);
                            idx += 3;
                            continue;
                        }
                        // shift right, so increment idx once and capture the token
                        tokens.push(Token::GreaterThanGreaterThan);
                        idx += 2;
                        continue;
                    } else if idx < len - 1 && bytes[idx + 1] == b'=' {
                        tokens.push(Token::GreaterThanEqual);
                        idx += 2;
                        continue;
                    }
                    tokens.push(Token::GreaterThan);
                }
                b'<' => {
                    if idx < len - 2 && bytes[idx + 2] == b'=' {
                        tokens.push(Token::LessThanLessThanEqual);
                        idx += 3;
                        continue;
                    }
                    if idx < len - 1 && bytes[idx + 1] == b'<' {
                        tokens.push(Token::LessThanLessThan);
                        idx += 2;
                        continue;
                    } else if idx < len - 1 && bytes[idx + 1] == b'=' {
                        tokens.push(Token::LessThanEqual);
                        idx += 2;
                        continue;
                    }
                    tokens.push(Token::LessThan);
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
        matches!(c, b'a'..=b'z' | b'A'..=b'Z'| b'0'..=b'9' | b'_')
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

    #[test]
    fn lexes_double_hyphen() {
        let source = r#"
    return --2;
"#;
        let lexer = Lexer::lex(source);
        assert!(lexer.is_ok());
        let lexer = lexer.unwrap();
        let mut tokens = lexer.tokens();
        assert_eq!(Some(&Token::Return), tokens.next());
        assert_eq!(Some(&Token::DoubleHyphen), tokens.next());
        assert_eq!(Some(&Token::Constant(2)), tokens.next());
        assert_eq!(Some(&Token::Semicolon), tokens.next());
        assert_eq!(None, tokens.next());
    }
    #[test]
    fn lexes_tilde_and_hyphen() {
        let source = r#"
    ~(-2);
"#;
        let lexer = Lexer::lex(source);
        assert!(lexer.is_ok());
        let lexer = lexer.unwrap();
        let mut tokens = lexer.tokens();
        assert_eq!(Some(&Token::Tilde), tokens.next());
        assert_eq!(Some(&Token::LeftParen), tokens.next());
        assert_eq!(Some(&Token::Hyphen), tokens.next());
        assert_eq!(Some(&Token::Constant(2)), tokens.next());
        assert_eq!(Some(&Token::RightParen), tokens.next());
        assert_eq!(Some(&Token::Semicolon), tokens.next());
        assert_eq!(None, tokens.next());
    }
    #[test]
    fn test_binary_operators() {
        let source = "+*/% += -= /= *= %= <<= >>=";
        let lexer = Lexer::lex(source);
        assert!(lexer.is_ok());
        let lexer = lexer.unwrap();
        let mut tokens = lexer.tokens();
        assert_eq!(Some(&Token::Plus), tokens.next());
        assert_eq!(Some(&Token::Star), tokens.next());
        assert_eq!(Some(&Token::Slash), tokens.next());
        assert_eq!(Some(&Token::Percent), tokens.next());
        assert_eq!(Some(&Token::PlusEqual), tokens.next());
        assert_eq!(Some(&Token::HyphenEqual), tokens.next());
        assert_eq!(Some(&Token::SlashEqual), tokens.next());
        assert_eq!(Some(&Token::StarEqual), tokens.next());
        assert_eq!(Some(&Token::PercentEqual), tokens.next());
        assert_eq!(Some(&Token::LessThanLessThanEqual), tokens.next());
        assert_eq!(Some(&Token::GreaterThanGreaterThanEqual), tokens.next());
        assert_eq!(None, tokens.next());
    }

    #[test]
    fn test_simple_bitwise_operators() {
        let source = "^ | & << >>";
        let lexer = Lexer::lex(source);
        assert!(lexer.is_ok());
        let lexer = lexer.unwrap();
        let mut tokens = lexer.tokens();
        assert_eq!(Some(&Token::Caret), tokens.next());
        assert_eq!(Some(&Token::Pipe), tokens.next());
        assert_eq!(Some(&Token::Ampersand), tokens.next());
        assert_eq!(Some(&Token::LessThanLessThan), tokens.next());
        assert_eq!(Some(&Token::GreaterThanGreaterThan), tokens.next());
        assert_eq!(None, tokens.next());
    }

    #[test]
    fn test_logical_operators_and_longest_match() {
        let source = "!= ! <= &&| >= == || < >";
        let lexer = Lexer::lex(source);
        assert!(lexer.is_ok());
        let lexer = lexer.unwrap();
        let mut tokens = lexer.tokens();
        assert_eq!(Some(&Token::BangEqual), tokens.next());
        assert_eq!(Some(&Token::Bang), tokens.next());
        assert_eq!(Some(&Token::LessThanEqual), tokens.next());
        assert_eq!(Some(&Token::AmpersandAmpersand), tokens.next());
        assert_eq!(Some(&Token::Pipe), tokens.next());
        assert_eq!(Some(&Token::GreaterThanEqual), tokens.next());
        assert_eq!(Some(&Token::EqualEqual), tokens.next());
        assert_eq!(Some(&Token::PipePipe), tokens.next());
        assert_eq!(Some(&Token::LessThan), tokens.next());
        assert_eq!(Some(&Token::GreaterThan), tokens.next());
        assert_eq!(None, tokens.next());
    }

    #[test]
    fn test_more_logical_lexing() {
        let source = r#"int main(void) {
    return 2 == 2 || 0;
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
        assert_eq!(Some(&Token::EqualEqual), tokens.next());
        assert_eq!(Some(&Token::Constant(2)), tokens.next());
        assert_eq!(Some(&Token::PipePipe), tokens.next());
        assert_eq!(Some(&Token::Constant(0)), tokens.next());
        assert_eq!(Some(&Token::Semicolon), tokens.next());
        assert_eq!(Some(&Token::RightBrace), tokens.next());
        assert_eq!(None, tokens.next());
    }

    #[test]
    fn test_identifier_parsing() {
        let source = "return_val_123";
        let lexer = Lexer::lex(source);
        assert!(lexer.is_ok());
        let lexer = lexer.unwrap();
        let mut tokens = lexer.tokens();
        assert_eq!(Some(&Token::Identifier("return_val_123")), tokens.next());
        assert_eq!(None, tokens.next());
    }
}
