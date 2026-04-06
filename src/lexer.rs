use regex::Regex;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LexerError {
    #[error("Got an unexpected token: {0}")]
    UnexpectedToken(String),
    #[error("Got an unexpected character: {0}")]
    UnexpectedChar(char),
    #[error("Never closed a block comment")]
    UnclosedBlockComment,
    #[error("Unexpected EOF")]
    UnexpectedEOF,
}

// todo: let's stop carrying the entire text around
#[derive(Debug)]
pub struct Lexer<'a> {
    tokens: Vec<Token<'a>>,
}

#[derive(Debug, PartialEq, Copy, Clone, Eq, Hash)]
pub enum Token<'a> {
    Identifier(&'a str),
    Constant(usize),
    LongConstant(usize),
    UnsignedIntConstant(usize),
    UnsignedLongConstant(usize),
    SingleLineComment(&'a str),
    BlockComment(&'a str),
    PreprocessorBlock(&'a str), //todo
    // keywords
    Int,
    Long,
    Main,
    Void,
    Return,
    If,
    Else,
    Goto,
    Do,
    While,
    For,
    Break,
    Continue,
    // special symbols
    Comma,
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
    QuestionMark,
    Colon,
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
    // storage / linkage
    Static,
    Extern,
    // ch 12: signed / unsigned values
    Signed,
    Unsigned,
    // ch 13
    Double,
    FloatingPointConstant(&'a str), // converted to f64 in parser
}

impl<'a> Token<'a> {
    pub fn into_string(&self) -> String {
        use Token::*;
        match self {
            Identifier(s) => format!("Identifier {s}"),
            Constant(c) => format!("Constant {c}"),
            LongConstant(c) => format!("LongConstant {c}"),
            UnsignedIntConstant(c) => format!("UnsignedIntConstant {c}"),
            UnsignedLongConstant(c) => format!("UnsignedLongConstant {c}"),
            SingleLineComment(c) => format!("SingleLineComment {c}"),
            BlockComment(c) => format!("BlockComment {c}"),
            PreprocessorBlock(c) => format!("PreprocessorBlock {c}"),
            Int => format!("Int"),
            Long => format!("Long"),
            Main => format!("Main"),
            Void => format!("Void"),
            Return => format!("Return"),
            If => format!("If"),
            Else => format!("Else"),
            Goto => format!("Goto"),
            Do => format!("Do"),
            While => format!("While"),
            For => format!("For"),
            Break => format!("Break"),
            Continue => format!("Continue"),
            Comma => format!("Comma"),
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
            QuestionMark => format!("QuestionMark"),
            Colon => format!("Colon"),
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
            Static => format!("Static"),
            Extern => format!("Extern"),
            Signed => format!("Signed"),
            Unsigned => format!("Unsigned"),
            Double => format!("Double"),
            FloatingPointConstant(c) => format!("FloatingPointConstant {c}"),
        }
    }
}

// helper for suffix parsing
#[derive(Debug, PartialEq, Copy, Clone)]
enum DigitType {
    Long,
    UnsignedInt,
    UnsignedLongInt,
}

impl<'a> Lexer<'a> {
    #[cfg(test)] // maybe prefer allow dead code
    pub fn tokens(&self) -> std::slice::Iter<'a, Token<'_>> {
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
                b',' => tokens.push(Token::Comma),
                b'{' => tokens.push(Token::LeftBrace),
                b'}' => tokens.push(Token::RightBrace),
                b'(' => tokens.push(Token::LeftParen),
                b')' => tokens.push(Token::RightParen),
                b';' => tokens.push(Token::Semicolon),
                b'~' => tokens.push(Token::Tilde),
                b'?' => tokens.push(Token::QuestionMark),
                b':' => tokens.push(Token::Colon),
                b'+' => {
                    if idx < len - 1 && bytes[idx + 1] == b'=' {
                        tokens.push(Token::PlusEqual);
                        idx += 1;
                    } else if idx < len - 1 && bytes[idx + 1] == b'+' {
                        tokens.push(Token::PlusPlus);
                        idx += 1;
                    } else {
                        tokens.push(Token::Plus);
                    }
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
                b'_' if idx < len - 1 && bytes[idx + 1] == b' ' => tokens.push(Token::Underscore),
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
                    if idx < len - 1 && bytes[idx + 1] == b'<' {
                        if idx < len - 2 && bytes[idx + 2] == b'=' {
                            tokens.push(Token::LessThanLessThanEqual);
                            idx += 3;
                            continue;
                        }
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
                b'_' | b'a'..=b'z' | b'A'..=b'Z' => {
                    // Identifiers can also start with an underscore
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

                    // at this point, either end == len which means
                    // we parsed the entire string, and it ends with a digit, OR
                    // end < len and we are pointing to a non-digit character.
                    // What do we do? If we're at the end, attempt to return a constant.
                    // If we see a period, we'll look for a double precision float.
                    // Else, we'll see if we have a valid suffix and push a different sized
                    // constant. Still does not support floats yet.
                    if end < len && bytes[end] == b'.' {
                        let end = Self::get_end_of_fractional_part_of_decimal(bytes, end)?;
                        let s =
                            std::str::from_utf8(&bytes[start..end]).expect("We know this is UTF8");
                        tokens.push(Token::FloatingPointConstant(s));
                        // end points to space just after parse, so walk backwards once
                        idx = end - 1;
                    } else if end < len && matches!(bytes[end], b'E' | b'e') {
                        // special case: scientific notation
                        if end + 1 == len {
                            // we need an integer for scientific notation form
                            return Err(LexerError::UnexpectedToken(format!(
                                "Need an integer after {} for scientific notation, found end of file",
                                bytes[end] as char
                            )));
                        }
                        let suffix_end =
                            Self::get_valid_suffix_for_exponent_notation(bytes, end + 1)?;
                        let s = std::str::from_utf8(&bytes[start..suffix_end])
                            .expect("We know this is UTF8");
                        tokens.push(Token::FloatingPointConstant(s));
                        // end points to space just after parse, so walk backwards once
                        idx = suffix_end - 1;
                    } else if end < len && bytes[end].is_ascii_alphabetic() {
                        // we need to find a suffix here, if possible. To do that,
                        // move an index forward as long as we see a valid identifier.
                        // Then we hardcode some checks for types.
                        let mut possible_end = end + 1;
                        while possible_end < len
                            && Self::is_valid_identifier_character(bytes[possible_end])
                        {
                            possible_end += 1;
                        }
                        let Some(digit_type) =
                            Self::get_digit_type_from_suffix(&bytes[end..possible_end])
                        else {
                            return Err(LexerError::UnexpectedToken(format!(
                                "Unexpected suffix for digit: {}",
                                std::str::from_utf8(&bytes[start..possible_end])
                                    .expect("Should be UTF-8")
                            )));
                        };
                        let s =
                            std::str::from_utf8(&bytes[start..end]).expect("We know this is UTF8");
                        let constant: usize = s
                            .parse()
                            .expect("Didn't get a digit after parsing a string");
                        let token = match digit_type {
                            DigitType::Long => Token::LongConstant(constant),
                            DigitType::UnsignedInt => Token::UnsignedIntConstant(constant),
                            DigitType::UnsignedLongInt => Token::UnsignedLongConstant(constant),
                        };

                        tokens.push(token);
                        // point to last char of suffix, let increment at the end point after it.
                        idx = possible_end - 1;
                    } else {
                        // at this point, either end == len OR end < len and bytes[end] is a
                        // non-alphabetic, non-period character (whitespace, etc). Just scrape up
                        // the text and move on
                        let s =
                            std::str::from_utf8(&bytes[start..end]).expect("We know this is UTF8");
                        let constant: usize = s
                            .parse()
                            .expect("Didn't get a digit after parsing a string");
                        tokens.push(Token::Constant(constant));
                        idx = end - 1; // increment at the end sets idx == end
                    }
                }
                b'.' => {
                    // floating point number, nothing on the LHS of the period
                    // Here we require a digit after the period, since '.' is not a valid double
                    if idx == len - 1 || !bytes[idx + 1].is_ascii_digit() {
                        return Err(LexerError::UnexpectedChar(bytes[idx].into()));
                    }
                    let end = Self::get_end_of_fractional_part_of_decimal(bytes, idx)?;
                    let s = std::str::from_utf8(&bytes[idx..end]).expect("We know this is UTF8");
                    tokens.push(Token::FloatingPointConstant(s));
                    // end points to space just after parse, so walk backwards once
                    idx = end - 1;
                }
                _ => return Err(LexerError::UnexpectedChar(bytes[idx].into())),
            }
            idx += 1;
        }

        Ok(Lexer { tokens })
    }

    fn get_end_of_fractional_part_of_decimal(
        bytes: &[u8],
        idx: usize,
    ) -> Result<usize, LexerError> {
        // to parse, we'll iterate a pointer forward following some simple rules:
        // 1) scoop up as many digits as we see
        // 2) if the next char is e or E, keep going
        // 3) if we see + or - here ONCE, it's valid
        // 4) keep scooping up digits
        let len = bytes.len();
        let mut end = idx + 1;
        while end < len && bytes[end].is_ascii_digit() {
            end += 1;
        }
        if end == len {
            return Ok(end);
        }
        if !bytes[end].is_ascii_alphanumeric() {
            // just to be safe, make sure we don't grab non-alphanumeric
            // identifier characters. In this case, just underscore.
            if bytes[end] == b'_' {
                return Err(LexerError::UnexpectedToken(format!(
                    "Found {} in the middle of a decimal fractional part",
                    bytes[end] as char
                )));
            }
            return Ok(end);
        }
        // at this point, we have an ascii alphanumeric char and we're within
        // source length.
        if matches!(bytes[end], b'e' | b'E') {
            end += 1;
        } else {
            // we have some weird suffix-looking thing here, error
            return Err(LexerError::UnexpectedToken(format!(
                "Found {} in the middle of a decimal fractional part",
                bytes[end] as char
            )));
        }
        Self::get_valid_suffix_for_exponent_notation(bytes, end)
    }

    fn get_valid_suffix_for_exponent_notation(
        bytes: &[u8],
        idx: usize,
    ) -> Result<usize, LexerError> {
        // the next char MUST be a +, -, or digit or else we're in errortown
        // allow a + or - here
        let mut end = idx;
        let len = bytes.len();
        if end == len {
            return Err(LexerError::UnexpectedEOF);
        }
        let re = Regex::new(r"[0-9]|\+|\-").unwrap();
        if !re.is_match(std::str::from_utf8(&bytes[end..=end]).unwrap()) {
            return Err(LexerError::UnexpectedToken(format!(
                "Expected +, - or digit after E/e, got {}",
                bytes[end] as char
            )));
        }
        if end < len && matches!(bytes[end], b'-' | b'+') {
            end += 1;
        }
        let idx_of_sign_byte = end;
        while end < len && bytes[end].is_ascii_digit() {
            end += 1;
        }
        // If there's source text left, it must not be a valid alphanumeric, or a period which
        // would indicate a nested decimal
        if end < len && (bytes[end].is_ascii_alphanumeric() | (bytes[end] == b'.')) {
            return Err(LexerError::UnexpectedToken(format!(
                "Expected non-alphanumeric, non-period after decimal fractional part, got {}",
                bytes[end] as char
            )));
        }
        if end == idx_of_sign_byte {
            return Err(LexerError::UnexpectedToken(format!(
                "Did not find digits after sign byte in token {:?}",
                std::str::from_utf8(&bytes[idx..end]).unwrap()
            )));
        }
        Ok(end)
    }

    fn is_valid_identifier_character(c: u8) -> bool {
        matches!(c, b'a'..=b'z' | b'A'..=b'Z'| b'0'..=b'9' | b'_')
    }

    fn get_digit_type_from_suffix(bytes: &[u8]) -> Option<DigitType> {
        let bytestring = std::str::from_utf8(bytes).expect("Should be UTF-8");
        if Regex::new(r"^[uU]$").unwrap().is_match(bytestring) {
            return Some(DigitType::UnsignedInt);
        };
        if Regex::new(r"^[lL]$").unwrap().is_match(bytestring) {
            return Some(DigitType::Long);
        };
        if Regex::new(r"^([lL][uU]|[uU][lL])$")
            .unwrap()
            .is_match(bytestring)
        {
            return Some(DigitType::UnsignedLongInt);
        };
        None
    }

    fn parse_keyword(s: &str) -> Option<Token<'_>> {
        match s {
            "int" => Some(Token::Int),
            "long" => Some(Token::Long),
            "main" => Some(Token::Main),
            "return" => Some(Token::Return),
            "void" => Some(Token::Void),
            "if" => Some(Token::If),
            "else" => Some(Token::Else),
            "goto" => Some(Token::Goto),
            "do" => Some(Token::Do),
            "while" => Some(Token::While),
            "for" => Some(Token::For),
            "break" => Some(Token::Break),
            "continue" => Some(Token::Continue),
            "static" => Some(Token::Static),
            "extern" => Some(Token::Extern),
            "signed" => Some(Token::Signed),
            "unsigned" => Some(Token::Unsigned),
            "double" => Some(Token::Double),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keywords() {
        let source = "int main 123 static extern long;";
        let lexer = Lexer::lex(source);
        assert!(lexer.is_ok());
        let lexer = lexer.unwrap();
        let mut tokens = lexer.tokens();
        assert_eq!(Some(&Token::Int), tokens.next());
        assert_eq!(Some(&Token::Main), tokens.next());
        assert_eq!(Some(&Token::Constant(123)), tokens.next());
        assert_eq!(Some(&Token::Static), tokens.next());
        assert_eq!(Some(&Token::Extern), tokens.next());
        assert_eq!(Some(&Token::Long), tokens.next());
        assert_eq!(Some(&Token::Semicolon), tokens.next());
        assert_eq!(None, tokens.next());
    }

    #[test]
    fn basic_source() {
        let source = r#"
        int main(void) {
            long x = 2L;
            long y = 3l;
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
        assert_eq!(Some(&Token::Long), tokens.next());
        assert_eq!(Some(&Token::Identifier("x")), tokens.next());
        assert_eq!(Some(&Token::Equal), tokens.next());
        assert_eq!(Some(&Token::LongConstant(2)), tokens.next());
        assert_eq!(Some(&Token::Semicolon), tokens.next());
        assert_eq!(Some(&Token::Long), tokens.next());
        assert_eq!(Some(&Token::Identifier("y")), tokens.next());
        assert_eq!(Some(&Token::Equal), tokens.next());
        assert_eq!(Some(&Token::LongConstant(3)), tokens.next());
        assert_eq!(Some(&Token::Semicolon), tokens.next());
        assert_eq!(Some(&Token::Return), tokens.next());
        assert_eq!(Some(&Token::Constant(2)), tokens.next());
        assert_eq!(Some(&Token::Semicolon), tokens.next());
        assert_eq!(Some(&Token::RightBrace), tokens.next());
        assert_eq!(None, tokens.next());
    }

    #[test]
    fn invalid_long_constant() {
        // TODO maybe make this work.
        let source = "long x = 2LL";
        let lexer = Lexer::lex(source);
        assert!(lexer.is_err());
    }

    #[test]
    fn invalid_suffix_from_missing_whitespace() {
        let source = "123ullong"; // should error, not be unsigned long + long keyword
        let lexer = Lexer::lex(source);
        let Err(LexerError::UnexpectedToken(s)) = lexer else {
            panic!();
        };
        assert_eq!(s, "Unexpected suffix for digit: 123ullong");
    }

    #[test]
    fn unsigned_and_signed_constants() {
        let source = r"
            unsigned int x = 1u;
            signed int y = 2;
            unsigned long z1 = 3ul;
            unsigned long z2 = 4Lu;
            unsigned long a = 5Ul;
            unsigned long b = 6lU;
            ";
        let lexer = Lexer::lex(source);
        assert!(lexer.is_ok(), "bad lexer {lexer:?}");
        let lexer = lexer.unwrap();
        let mut tokens = lexer.tokens();
        assert_eq!(Some(&Token::Unsigned), tokens.next());
        assert_eq!(Some(&Token::Int), tokens.next());
        assert_eq!(Some(&Token::Identifier("x")), tokens.next());
        assert_eq!(Some(&Token::Equal), tokens.next());
        assert_eq!(Some(&Token::UnsignedIntConstant(1)), tokens.next());
        assert_eq!(Some(&Token::Semicolon), tokens.next());
        assert_eq!(Some(&Token::Signed), tokens.next());
        assert_eq!(Some(&Token::Int), tokens.next());
        assert_eq!(Some(&Token::Identifier("y")), tokens.next());
        assert_eq!(Some(&Token::Equal), tokens.next());
        assert_eq!(Some(&Token::Constant(2)), tokens.next());
        assert_eq!(Some(&Token::Semicolon), tokens.next());
        assert_eq!(Some(&Token::Unsigned), tokens.next());
        assert_eq!(Some(&Token::Long), tokens.next());
        assert_eq!(Some(&Token::Identifier("z1")), tokens.next());
        assert_eq!(Some(&Token::Equal), tokens.next());
        assert_eq!(Some(&Token::UnsignedLongConstant(3)), tokens.next());
        assert_eq!(Some(&Token::Semicolon), tokens.next());
        assert_eq!(Some(&Token::Unsigned), tokens.next());
        assert_eq!(Some(&Token::Long), tokens.next());
        assert_eq!(Some(&Token::Identifier("z2")), tokens.next());
        assert_eq!(Some(&Token::Equal), tokens.next());
        assert_eq!(Some(&Token::UnsignedLongConstant(4)), tokens.next());
        assert_eq!(Some(&Token::Semicolon), tokens.next());
        assert_eq!(Some(&Token::Unsigned), tokens.next());
        assert_eq!(Some(&Token::Long), tokens.next());
        assert_eq!(Some(&Token::Identifier("a")), tokens.next());
        assert_eq!(Some(&Token::Equal), tokens.next());
        assert_eq!(Some(&Token::UnsignedLongConstant(5)), tokens.next());
        assert_eq!(Some(&Token::Semicolon), tokens.next());
        assert_eq!(Some(&Token::Unsigned), tokens.next());
        assert_eq!(Some(&Token::Long), tokens.next());
        assert_eq!(Some(&Token::Identifier("b")), tokens.next());
        assert_eq!(Some(&Token::Equal), tokens.next());
        assert_eq!(Some(&Token::UnsignedLongConstant(6)), tokens.next());
        assert_eq!(Some(&Token::Semicolon), tokens.next());

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
    fn complex_lex() {
        let source = r#"
    if (true) { return 1; } else { return 2; };
    int x = true ? 1 : 5;
    goto banana:
    return --2;
"#;
        let lexer = Lexer::lex(source);
        assert!(lexer.is_ok());
        let lexer = lexer.unwrap();
        let mut tokens = lexer.tokens();
        assert_eq!(Some(&Token::If), tokens.next());
        assert_eq!(Some(&Token::LeftParen), tokens.next());
        assert_eq!(Some(&Token::Identifier("true")), tokens.next());
        assert_eq!(Some(&Token::RightParen), tokens.next());
        assert_eq!(Some(&Token::LeftBrace), tokens.next());
        assert_eq!(Some(&Token::Return), tokens.next());
        assert_eq!(Some(&Token::Constant(1)), tokens.next());
        assert_eq!(Some(&Token::Semicolon), tokens.next());
        assert_eq!(Some(&Token::RightBrace), tokens.next());
        assert_eq!(Some(&Token::Else), tokens.next());
        assert_eq!(Some(&Token::LeftBrace), tokens.next());
        assert_eq!(Some(&Token::Return), tokens.next());
        assert_eq!(Some(&Token::Constant(2)), tokens.next());
        assert_eq!(Some(&Token::Semicolon), tokens.next());
        assert_eq!(Some(&Token::RightBrace), tokens.next());
        assert_eq!(Some(&Token::Semicolon), tokens.next());
        assert_eq!(Some(&Token::Int), tokens.next());
        assert_eq!(Some(&Token::Identifier("x")), tokens.next());
        assert_eq!(Some(&Token::Equal), tokens.next());
        assert_eq!(Some(&Token::Identifier("true")), tokens.next());
        assert_eq!(Some(&Token::QuestionMark), tokens.next());
        assert_eq!(Some(&Token::Constant(1)), tokens.next());
        assert_eq!(Some(&Token::Colon), tokens.next());
        assert_eq!(Some(&Token::Constant(5)), tokens.next());
        assert_eq!(Some(&Token::Semicolon), tokens.next());
        assert_eq!(Some(&Token::Goto), tokens.next());
        assert_eq!(Some(&Token::Identifier("banana")), tokens.next());
        assert_eq!(Some(&Token::Colon), tokens.next());
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
        let source = "!= ! <= &&| >= == || < > < = ";
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
        assert_eq!(Some(&Token::LessThan), tokens.next());
        assert_eq!(Some(&Token::Equal), tokens.next());
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

    #[test]
    fn test_double_precision_floats() {
        // + and - operators are valid in scientific notation
        let source = "1. 1.0 0.5 .5 100e1 1.E+10 1e-10";
        let lexer = Lexer::lex(source);
        assert!(lexer.is_ok(), "got lexer error {lexer:?}");
        let lexer = lexer.unwrap();
        let mut tokens = lexer.tokens();
        assert_eq!(Some(&Token::FloatingPointConstant("1.")), tokens.next());
        assert_eq!(Some(&Token::FloatingPointConstant("1.0")), tokens.next());
        assert_eq!(Some(&Token::FloatingPointConstant("0.5")), tokens.next());
        assert_eq!(Some(&Token::FloatingPointConstant(".5")), tokens.next());
        assert_eq!(Some(&Token::FloatingPointConstant("100e1")), tokens.next());
        assert_eq!(Some(&Token::FloatingPointConstant("1.E+10")), tokens.next());
        assert_eq!(Some(&Token::FloatingPointConstant("1e-10")), tokens.next());
        assert_eq!(None, tokens.next());
    }

    #[test]
    fn invalid_doubles() {
        for (source, err_msg) in [
            ("2._", "Found _ in the middle of a decimal fractional part"),
            // non-sign, non-digit right after e/E
            ("2ex", "Expected +, - or digit after E/e, got x"),
            // outer guard: nothing at all after e (before get_valid_suffix_for_exponent_notation is called)
            ("1e", "Need an integer after e for scientific notation, found end of file"),
            // digits followed by trailing alphanumeric
            ("1e1a", "Expected non-alphanumeric, non-period after decimal fractional part, got a"),
            // digits followed by trailing period
            ("1e1.", "Expected non-alphanumeric, non-period after decimal fractional part, got ."),
            // sign followed by alphanumeric (no digits consumed before the letter)
            ("1e+a", "Expected non-alphanumeric, non-period after decimal fractional part, got a"),
            // sign with no digits — EOF after sign
            ("1e+", "Did not find digits after sign byte in token \"+\""),
            ("1e-", "Did not find digits after sign byte in token \"-\""),
            // sign with no digits — non-alphanumeric (space) after sign
            ("1e+ ", "Did not find digits after sign byte in token \"+\""),
            ("1e- ", "Did not find digits after sign byte in token \"-\""),
        ] {
            let lexer = Lexer::lex(source);
            assert!(lexer.is_err(), "got valid lexer {lexer:?}");
            let Err(LexerError::UnexpectedToken(s)) = lexer else {
                panic!("Expected UnexpectedToken for {source:?}, got {lexer:?}");
            };
            assert_eq!(s, err_msg, "wrong message for {source:?}");
        }

        // UnexpectedEOF only comes from get_valid_suffix_for_exponent_notation via
        // get_end_of_fractional_part_of_decimal when the e/E is the very last byte
        // e.g. "1.e" — period path consumes the e then calls get_valid_suffix with idx==len
        let lexer = Lexer::lex("1.e");
        assert!(
            matches!(lexer, Err(LexerError::UnexpectedEOF)),
            "expected UnexpectedEOF for \"1.e\", got {lexer:?}"
        );
    }
}
