use std::iter::Peekable;
use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
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
    #[error("Expected binop but got {0}")]
    MissingBinop(String),
    #[error("Tried to calculate precedence for token {0}")]
    NoPrecedenceForToken(String),
}

// super basic AST model
// borrows tokens, bound to lifetime of source text
use crate::lexer::Token;
#[derive(Debug, PartialEq)]
pub enum AST<'a> {
    Program(Box<AST<'a>>),
    Function { name: &'a str, body: Box<AST<'a>> },
    Return(Expression),
}

#[derive(Debug, PartialEq)]
pub enum Expression {
    Constant(usize),
    Unary(UnaryOp, Box<Expression>),
    Binary(BinaryOp, Box<Expression>, Box<Expression>),
}

#[derive(Debug, PartialEq)]
pub enum UnaryOp {
    Negate,
    Complement,
}

#[derive(Debug, PartialEq)]
pub enum BinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Remainder,
    And,
    Or,
    Xor,
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
        let val = self.parse_expression(0)?;
        self.expect(Token::Semicolon)?;
        Ok(AST::Return(val))
    }

    fn parse_binop(&mut self) -> Result<BinaryOp, ParserError> {
        match self.tokens.next() {
            Some(Token::Plus) => Ok(BinaryOp::Add),
            Some(Token::Hyphen) => Ok(BinaryOp::Subtract),
            Some(Token::Star) => Ok(BinaryOp::Multiply),
            Some(Token::Slash) => Ok(BinaryOp::Divide),
            Some(Token::Percent) => Ok(BinaryOp::Remainder),
            Some(Token::Ampersand) => Ok(BinaryOp::And),
            Some(Token::Pipe) => Ok(BinaryOp::Or),
            Some(Token::Caret) => Ok(BinaryOp::Xor),
            Some(t) => Err(ParserError::MissingBinop(t.into_string())),
            None => Err(ParserError::OutOfTokens),
        }
    }

    fn parse_expression(&mut self, min_precedence: u8) -> Result<Expression, ParserError> {
        let mut left = self.parse_factor()?;
        // while next token is a binary operator, and precedence is climbing, keep parsing
        loop {
            match self.tokens.peek() {
                Some(t @ Token::Plus)
                | Some(t @ Token::Hyphen)
                | Some(t @ Token::Star)
                | Some(t @ Token::Slash)
                | Some(t @ Token::Percent)
                | Some(t @ Token::Ampersand)
                | Some(t @ Token::Caret)
                | Some(t @ Token::Pipe) => {
                    let new_prec = Self::precedence(*t)?;
                    if new_prec < min_precedence {
                        break;
                    }
                    let operator = self.parse_binop()?;
                    let right = self.parse_expression(new_prec + 1)?;
                    left = Expression::Binary(operator, Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }

        Ok(left)
    }

    fn parse_factor(&mut self) -> Result<Expression, ParserError> {
        match self.tokens.next() {
            Some(Token::Constant(c)) => Ok(Expression::Constant(*c)),
            Some(Token::Hyphen) => {
                let inner_exp = self.parse_factor()?;
                Ok(Expression::Unary(UnaryOp::Negate, Box::new(inner_exp)))
            }
            Some(Token::Tilde) => {
                let inner_exp = self.parse_factor()?;
                Ok(Expression::Unary(UnaryOp::Complement, Box::new(inner_exp)))
            }
            Some(Token::LeftParen) => {
                // throw away parens and use just inner expression
                let inner_expr = self.parse_expression(0)?;
                self.expect(Token::RightParen)?;
                Ok(inner_expr)
            }
            _ => Err(ParserError::UnexpectedExpressionToken),
        }
    }

    fn expect(&mut self, expected: Token<'a>) -> Result<(), ParserError> {
        match self.tokens.next() {
            Some(token) if std::mem::discriminant(token) == std::mem::discriminant(&expected) => {
                Ok(())
            }
            Some(token) => Err(ParserError::UnexpectedToken(
                expected.into_string(),
                token.into_string(),
            )),
            None => Err(ParserError::OutOfTokens),
        }
    }

    // https://en.cppreference.com/w/c/language/operator_precedence
    // List below must be high-to-low precedence to match link above
    fn precedence(token: &Token) -> Result<u8, ParserError> {
        match token {
            Token::Star | Token::Slash | Token::Percent => Ok(50),
            Token::Plus | Token::Hyphen => Ok(45),
            Token::Ampersand => Ok(32),
            Token::Caret => Ok(31),
            Token::Pipe => Ok(30),
            t => Err(ParserError::NoPrecedenceForToken(t.into_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Token;
    #[test]
    fn basic_parse() {
        /*
         int main(void) {
           return 100;
         }
        */
        let tokens = vec![
            Token::Int,
            Token::Main,
            Token::LeftParen,
            Token::Void,
            Token::RightParen,
            Token::LeftBrace,
            Token::Return,
            Token::Constant(100),
            Token::Semicolon,
            Token::RightBrace,
        ];
        let parse = Parser::new(&tokens);
        let ast = parse.into_ast();
        assert!(ast.is_ok());
        let ast = ast.unwrap();
        let expected = AST::Program(Box::new(AST::Function {
            name: "main",
            body: Box::new(AST::Return(Expression::Constant(100))),
        }));
        assert_eq!(expected, ast);
    }

    #[test]
    fn failed_parse() {
        /* MISSING THE VOID PARAM
        int main() { return 200; }
        */
        let tokens = vec![
            Token::Int,
            Token::Main,
            Token::LeftParen,
            Token::RightParen,
            Token::LeftBrace,
            Token::Return,
            Token::Constant(100),
            Token::Semicolon,
            Token::RightBrace,
        ];
        let parse = Parser::new(&tokens);
        let ast = parse.into_ast();
        assert!(ast.is_err());
        assert_eq!(
            Err(ParserError::UnexpectedToken(
                String::from("Void"),
                String::from("RightParen")
            )),
            ast
        );
    }

    #[test]
    fn parses_tilde() {
        let tokens = vec![
            Token::Int,
            Token::Main,
            Token::LeftParen,
            Token::Void,
            Token::RightParen,
            Token::LeftBrace,
            Token::Return,
            Token::Tilde,
            Token::Constant(100),
            Token::Semicolon,
            Token::RightBrace,
        ];
        let parse = Parser::new(&tokens);
        let ast = parse.into_ast();
        assert!(ast.is_ok());
        let ast = ast.unwrap();
        let expected = AST::Program(Box::new(AST::Function {
            name: "main",
            body: Box::new(AST::Return(Expression::Unary(
                UnaryOp::Complement,
                Box::new(Expression::Constant(100)),
            ))),
        }));
        assert_eq!(expected, ast);
    }

    #[test]
    fn parses_hyphen() {
        let tokens = vec![
            Token::Int,
            Token::Main,
            Token::LeftParen,
            Token::Void,
            Token::RightParen,
            Token::LeftBrace,
            Token::Return,
            Token::Hyphen,
            Token::Constant(100),
            Token::Semicolon,
            Token::RightBrace,
        ];
        let parse = Parser::new(&tokens);
        let ast = parse.into_ast();
        assert!(ast.is_ok());
        let ast = ast.unwrap();
        let expected = AST::Program(Box::new(AST::Function {
            name: "main",
            body: Box::new(AST::Return(Expression::Unary(
                UnaryOp::Negate,
                Box::new(Expression::Constant(100)),
            ))),
        }));
        assert_eq!(expected, ast);
    }

    #[test]
    fn removes_extra_parens_on_expression() {
        let tokens = vec![
            Token::Int,
            Token::Main,
            Token::LeftParen,
            Token::Void,
            Token::RightParen,
            Token::LeftBrace,
            Token::Return,
            Token::LeftParen,
            Token::Hyphen,
            Token::Constant(100),
            Token::RightParen,
            Token::Semicolon,
            Token::RightBrace,
        ];
        let parse = Parser::new(&tokens);
        let ast = parse.into_ast();
        assert!(ast.is_ok());
        let ast = ast.unwrap();
        let expected = AST::Program(Box::new(AST::Function {
            name: "main",
            body: Box::new(AST::Return(Expression::Unary(
                UnaryOp::Negate,
                Box::new(Expression::Constant(100)),
            ))),
        }));
        assert_eq!(expected, ast);
    }

    #[test]
    fn fails_if_missing_right_paren_closing_expression() {
        let tokens = vec![
            Token::Int,
            Token::Main,
            Token::LeftParen,
            Token::Void,
            Token::RightParen,
            Token::LeftBrace,
            Token::Return,
            Token::LeftParen,
            Token::Hyphen,
            Token::Constant(100), // should have right paren after this
            Token::Semicolon,
            Token::RightBrace,
        ];
        let parse = Parser::new(&tokens);
        let ast = parse.into_ast();
        assert!(ast.is_err());
        assert_eq!(
            Err(ParserError::UnexpectedToken(
                String::from("RightParen"),
                String::from("Semicolon")
            )),
            ast
        );
    }

    #[test]
    fn unclosed_braces_fails() {
        /*
        int main(void) { return 200;
        */
        let tokens = vec![
            Token::Int,
            Token::Main,
            Token::LeftParen,
            Token::Void,
            Token::RightParen,
            Token::LeftBrace,
            Token::Return,
            Token::Constant(100),
            Token::Semicolon,
        ];
        let parse = Parser::new(&tokens);
        let ast = parse.into_ast();
        assert!(ast.is_err());
        assert_eq!(Err(ParserError::OutOfTokens), ast);
    }

    #[test]
    fn handles_precedence() {
        /*
         * int main(void) {
         *   return 1 * 2 - 3 * (4 + 5);
         * }
         */
        let tokens = vec![
            Token::Int,
            Token::Main,
            Token::LeftParen,
            Token::Void,
            Token::RightParen,
            Token::LeftBrace,
            Token::Return,
            Token::Constant(1),
            Token::Star,
            Token::Constant(2),
            Token::Hyphen,
            Token::Constant(3),
            Token::Star,
            Token::LeftParen,
            Token::Constant(4),
            Token::Plus,
            Token::Constant(5),
            Token::RightParen,
            Token::Semicolon,
            Token::RightBrace,
        ];
        let parse = Parser::new(&tokens);
        let ast = parse.into_ast();
        assert!(ast.is_ok());
        let actual = ast.unwrap();
        let expected = AST::Program(Box::new(AST::Function {
            name: "main",
            body: Box::new(AST::Return(Expression::Binary(
                BinaryOp::Subtract,
                Box::new(Expression::Binary(
                    BinaryOp::Multiply,
                    Box::new(Expression::Constant(1)),
                    Box::new(Expression::Constant(2)),
                )),
                Box::new(Expression::Binary(
                    BinaryOp::Multiply,
                    Box::new(Expression::Constant(3)),
                    Box::new(Expression::Binary(
                        BinaryOp::Add,
                        Box::new(Expression::Constant(4)),
                        Box::new(Expression::Constant(5)),
                    )),
                )),
            ))),
        }));

        assert_eq!(expected, actual);
    }

    #[test]
    fn handles_precedence2() {
        /*
         * int main(void) {
         *   return 5 * 4 / 2 - 3 % (2 + 1);
         * }
         */
        let tokens = vec![
            Token::Int,
            Token::Main,
            Token::LeftParen,
            Token::Void,
            Token::RightParen,
            Token::LeftBrace,
            Token::Return,
            Token::Constant(5),
            Token::Star,
            Token::Constant(4),
            Token::Slash,
            Token::Constant(2),
            Token::Hyphen,
            Token::Constant(3),
            Token::Percent,
            Token::LeftParen,
            Token::Constant(2),
            Token::Plus,
            Token::Constant(1),
            Token::RightParen,
            Token::Semicolon,
            Token::RightBrace,
        ];
        let parse = Parser::new(&tokens);
        let ast = parse.into_ast();
        assert!(ast.is_ok());
        let actual = ast.unwrap();
        let expected = AST::Program(Box::new(AST::Function {
            name: "main",
            body: Box::new(AST::Return(Expression::Binary(
                BinaryOp::Subtract,
                Box::new(Expression::Binary(
                    BinaryOp::Divide,
                    Box::new(Expression::Binary(
                        BinaryOp::Multiply,
                        Box::new(Expression::Constant(5)),
                        Box::new(Expression::Constant(4)),
                    )),
                    Box::new(Expression::Constant(2)),
                )),
                Box::new(Expression::Binary(
                    BinaryOp::Remainder,
                    Box::new(Expression::Constant(3)),
                    Box::new(Expression::Binary(
                        BinaryOp::Add,
                        Box::new(Expression::Constant(2)),
                        Box::new(Expression::Constant(1)),
                    )),
                )),
            ))),
        }));

        assert_eq!(expected, actual);
    }

    #[test]
    fn handles_bitwise_precedence() {
        /*
         * int main(void) {
         *   return 5 * 4 | 4 - 5 & 6;
         * }
         */
        let tokens = vec![
            Token::Int,
            Token::Main,
            Token::LeftParen,
            Token::Void,
            Token::RightParen,
            Token::LeftBrace,
            Token::Return,
            Token::Constant(5),
            Token::Star,
            Token::Constant(4),
            Token::Pipe,
            Token::Constant(4),
            Token::Hyphen,
            Token::Constant(5),
            Token::Ampersand,
            Token::Constant(6),
            Token::Semicolon,
            Token::RightBrace,
        ];
        let parse = Parser::new(&tokens);
        let ast = parse.into_ast();
        assert!(ast.is_ok());
        let actual = ast.unwrap();
        let expected = AST::Program(Box::new(AST::Function {
            name: "main",
            body: Box::new(AST::Return(Expression::Binary(
                BinaryOp::Or,
                Box::new(Expression::Binary(
                    BinaryOp::Multiply,
                    Box::new(Expression::Constant(5)),
                    Box::new(Expression::Constant(4)),
                )),
                Box::new(Expression::Binary(
                    BinaryOp::And,
                    Box::new(Expression::Binary(
                        BinaryOp::Subtract,
                        Box::new(Expression::Constant(4)),
                        Box::new(Expression::Constant(5)),
                    )),
                    Box::new(Expression::Constant(6)),
                )),
            ))),
        }));

        assert_eq!(expected, actual);
    }
}
