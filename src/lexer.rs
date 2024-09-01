use thiserror::Error;

#[derive(Debug, Error)]
pub enum LexerError {
    #[error("Got an unexpected token: {0}")]
    UnexpectedToken(String),
}

// todo: let's stop carrying the entire text around
pub struct Lexer<'a> {
    tokens: Vec<Token<'a>>,
}

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

pub fn lex<'a>(source: &'a str) -> Result<Lexer<'a>, LexerError> {
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
            c if c.is_ascii_digit() => {
                // advance while we see digits
                let mut end = idx;
                while end < len && bytes[end].is_ascii_digit() {
                    end += 1;
                }
                // at this point, end is either EOF
                // or non-digit character. For now, just error
                // if end points to a..z or A..Z.
                // This does not support floats!
                if end == len {
                    break;
                }
                if bytes[end].is_ascii_alphabetic() {
                    return Err(LexerError::UnexpectedToken(
                        String::from_utf8(bytes[idx..=end].to_vec()).expect(
                            "Got invalid UTF-8 when addressing byte range from source string",
                        ),
                    ));
                }

                // todo: is this the best I can do?
                let s =
                    std::str::from_utf8(&bytes[idx..end]).expect("We know this is UTF8");
                let constant: usize = s
                    .parse()
                    .expect("Didn't get a digit after parsing a string");
                tokens.push(Token::Constant(constant));
            }
            _ => todo!(),
        }
        idx += 1;
    }

    Ok(Lexer { tokens })
}
