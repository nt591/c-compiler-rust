use std::iter::Peekable;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ParserError {
    #[error("Unexpected token: wanted {0} but got {1}")]
    UnexpectedToken(String, String),
    #[error("Unexpected token when parsing expression")]
    UnexpectedExpressionToken,
    #[error("Unexpected function name")]
    UnexpectedName,
    #[error("Got leftover tokens after parsing program")]
    LeftoverTokens,
    #[error("Out of tokens when trying to get next")]
    OutOfTokens,
}

// super basic AST model
// borrows tokens, bound to lifetime of source text
use crate::lexer::Token;
#[derive(Debug)]
pub enum AST<'a> {
    Program(Box<AST<'a>>),
    Function { name: &'a str, body: Box<AST<'a>> },
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
            tokens: tokens.iter().peekable(),
        }
    }

    pub fn into_ast(mut self) -> Result<AST<'a>, ParserError> {
        let program = self.parse_program()?;
        match self.tokens.next() {
            // ensure no leftover tokens
            None => Ok(program),
            Some(_) => Err(ParserError::LeftoverTokens),
        }
    }

    fn parse_program(&mut self) -> Result<AST<'a>, ParserError> {
        let val = self.parse_function()?;
        Ok(AST::Program(Box::new(val)))
    }

    fn parse_function(&mut self) -> Result<AST<'a>, ParserError> {
        // todo: set return type?
        self.expect(Token::Int)?;
        let name = match self.tokens.next() {
            Some(Token::Main) => "main",
            Some(Token::Identifier(ident)) => ident,
            _ => return Err(ParserError::UnexpectedName),
        };
        self.expect(Token::LeftParen)?;
        self.expect(Token::Void)?;
        self.expect(Token::RightParen)?;
        self.expect(Token::LeftBrace)?;
        let body = self.parse_statement()?;
        self.expect(Token::RightBrace)?;

        return Ok(AST::Function {
            name,
            body: Box::new(body),
        });
    }

    fn parse_statement(&mut self) -> Result<AST<'a>, ParserError> {
        self.expect(Token::Return)?;
        let val = self.parse_expression()?;
        self.expect(Token::Semicolon)?;
        Ok(AST::Return(Box::new(val)))
    }

    fn parse_expression(&mut self) -> Result<AST<'a>, ParserError> {
        match self.tokens.next() {
            Some(Token::Constant(c)) => Ok(AST::Constant(*c)),
            _ => Err(ParserError::UnexpectedExpressionToken),
        }
    }

    fn expect(&mut self, expected: Token<'a>) -> Result<(), ParserError> {
        match self.tokens.next() {
            Some(token) if std::mem::discriminant(token) == std::mem::discriminant(&expected) => {
                Ok(())
            }
            Some(token) => Err(ParserError::UnexpectedToken(expected.into_string(), token.into_string())),
            None => Err(ParserError::OutOfTokens),
        }
    }
}
