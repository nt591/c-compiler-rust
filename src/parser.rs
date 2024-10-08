use std::iter::Peekable;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ParserError {
    #[error("Unexpected token")]
    UnexpectedToken,
}

// super basic AST model
// borrows tokens, bound to lifetime of source text
use crate::lexer::Token;
#[derive(Debug)]
pub enum AST<'a> {
    Program(Box<AST<'a>>),
    Function {
        name: &'a str,
        body: Box<AST<'a>>,
    },
    Return(Box<AST<'a>>),
    Constant(usize),
}

pub struct Parser<'a> {
    tokens: Peekable<std::slice::Iter<'a, Token<'a>>>,
}

impl<'a> Parser<'a> {
    pub fn new(tokens: &'a [Token<'a>]) -> Self {
        // todo
        Self {
            tokens: tokens.iter().peekable()
        }
    }

    pub fn parse_statement(&mut self) -> Result<(), ParserError> {
        self.expect(Token::Return)?;
        Ok(())
    }

    fn expect(&mut self, expected: Token<'a>) -> Result<(), ParserError> {
        match self.tokens.next() {
            Some(token) if std::mem::discriminant(token) == std::mem::discriminant(&expected) => Ok(()),
            _ => Err(ParserError::UnexpectedToken),
        }
    }
}
