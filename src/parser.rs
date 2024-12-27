use std::iter::Peekable;
use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum ParserError {
    #[error("Unexpected token: wanted {0} but got {1}")]
    UnexpectedToken(String, String),
    #[error("Unexpected token when parsing expression. Got {0}")]
    UnexpectedExpressionToken(String),
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
    #[error("Expected colon after then-expr in conditional")]
    UnexpectedTokenInConditional,
}

// super basic AST model
// borrows tokens, bound to lifetime of source text
use crate::lexer::Token;
#[derive(Debug, PartialEq)]
pub enum AST<'a> {
    Program(Box<AST<'a>>),
    Function { name: &'a str, block: Block },
}

#[derive(Debug, PartialEq)]
pub enum Statement {
    Goto(String),
    Labelled {
        label: String,
        statement: Box<Statement>,
    },
    Return(Expression),
    Expr(Expression),
    If {
        condition: Expression,
        then: Box<Statement>,
        else_: Option<Box<Statement>>,
    },
    Compound(Block),
    Null,
}

// might need to evolve into an enum
// TODO: figure out how to borrow the str without
// two mutable borrows
#[derive(Debug, PartialEq)]
pub struct Declaration {
    pub name: String,
    pub init: Option<Expression>,
}

#[derive(Debug, PartialEq)]
pub enum BlockItem {
    Stmt(Statement),
    Decl(Declaration),
}

#[derive(Debug, PartialEq)]
pub struct Block(pub Vec<BlockItem>);

#[derive(Debug, PartialEq)]
pub enum Expression {
    Constant(usize),
    Var(String), // identifier for variable
    Unary(UnaryOp, Box<Expression>),
    Binary(BinaryOp, Box<Expression>, Box<Expression>),
    Assignment(Box<Expression>, Box<Expression>), // LHS, RHS
    Conditional {
        condition: Box<Expression>,
        then: Box<Expression>,
        else_: Box<Expression>,
    },
}

#[derive(Debug, PartialEq)]
pub enum UnaryOp {
    Negate,
    Complement,
    Not,
}

#[derive(Debug, PartialEq)]
pub enum BinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Remainder,
    BitwiseAnd,
    BitwiseOr,
    Xor,
    ShiftLeft,
    ShiftRight,
    Equal,
    NotEqual,
    LessThan,
    LessOrEqual,
    GreaterThan,
    GreaterOrEqual,
    BinAnd,
    BinOr,
    // Compound assignment operators
    AddAssign,
    MinusAssign,
    MultiplyAssign,
    DivideAssign,
    RemainderAssign,
    BitwiseAndAssign,
    BitwiseOrAssign,
    XorAssign,
    ShiftLeftAssign,
    ShiftRightAssign,
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
        let block = self.parse_block()?;

        return Ok(AST::Function { name, block });
    }

    fn parse_block(&mut self) -> Result<Block, ParserError> {
        self.expect(Token::LeftBrace)?;
        let mut body = vec![];

        loop {
            if self.tokens.peek() == Some(&&Token::RightBrace) {
                break;
            }
            let item = self.parse_block_item()?;
            body.push(item);
        }
        self.expect(Token::RightBrace)?;

        Ok(Block(body))
    }

    fn parse_block_item(&mut self) -> Result<BlockItem, ParserError> {
        let block_item = if let Some(Token::Int) = self.tokens.peek() {
            let decl = self.parse_declaration()?;
            BlockItem::Decl(decl)
        } else {
            let stmt = self.parse_statement()?;
            BlockItem::Stmt(stmt)
        };

        Ok(block_item)
    }

    // "int" 'IDENTIFIER' [ '=' EXPR] ';'
    fn parse_declaration(&mut self) -> Result<Declaration, ParserError> {
        assert_eq!(
            Some(Token::Int),
            self.tokens.next().copied(),
            "Expected to see 'int' token when parsing a declaration"
        );
        let name = match self.tokens.next() {
            Some(Token::Main) => "main",
            Some(Token::Identifier(ident)) => ident,
            _ => return Err(ParserError::UnexpectedName),
        };
        let init = if let Some(Token::Equal) = self.tokens.peek() {
            self.tokens.next();
            let init = self.parse_expression(0)?;
            self.expect(Token::Semicolon)?;
            Some(init)
        } else {
            self.expect(Token::Semicolon)?;
            None
        };

        Ok(Declaration {
            name: name.to_owned(),
            init,
        })
    }

    // if it starts with return, it's a return Statement
    // if it's just a semi colon, it's Null
    // If we start with a left brace, just treat as a block.
    // else, it's an expression
    fn parse_statement(&mut self) -> Result<Statement, ParserError> {
        match self.tokens.peek() {
            Some(Token::LeftBrace) => {
                let block = self.parse_block()?;
                Ok(Statement::Compound(block))
            }
            Some(Token::Return) => {
                self.tokens.next();
                let val = self.parse_expression(0)?;
                self.expect(Token::Semicolon)?;
                Ok(Statement::Return(val))
            }
            Some(Token::Semicolon) => {
                self.tokens.next();
                Ok(Statement::Null)
            }
            Some(Token::If) => {
                self.tokens.next();
                // parse_expression consumes parentheses
                self.expect_peek(Token::LeftParen)?;
                let condition = self.parse_expression(0)?;
                let then = Box::new(self.parse_statement()?);
                // if we see an else token, we have an optional else clause
                let else_ = match self.tokens.peek() {
                    Some(Token::Else) => {
                        self.tokens.next();
                        Some(Box::new(self.parse_statement()?))
                    }
                    _ => None,
                };
                Ok(Statement::If {
                    condition,
                    then,
                    else_,
                })
            }
            Some(Token::Goto) => {
                self.tokens.next();
                // we must have an identifier here, which will be our label.
                let label = match self.tokens.next() {
                    Some(Token::Identifier(ident)) => *ident,
                    Some(Token::Main) => "main",
                    None => return Err(ParserError::OutOfTokens),
                    Some(token) => {
                        return Err(ParserError::UnexpectedToken(
                            Token::Identifier("anything").into_string(),
                            token.into_string(),
                        ))
                    }
                };
                self.expect(Token::Semicolon)?;
                Ok(Statement::Goto(label.into()))
            }
            Some(Token::Identifier(_)) | Some(Token::Main) => {
                // We specially allow `main` to be a label, for some reason.
                // if we have a colon after this, maybe we treat this as a label. Else, parse expr.
                // we've peeked at the identifier, so index = 1 is
                // second element
                // SUPER HACKY CLONE! TODO - fix!
                let mut it = multipeek::multipeek(self.tokens.clone());
                match it.peek_nth(1) {
                    Some(Token::Colon) => {
                        let ident = match self.tokens.next() {
                            Some(Token::Identifier(ident)) => ident,
                            Some(Token::Main) => "main",
                            _ => panic!(),
                        };
                        self.expect(Token::Colon)?;
                        let statement = self.parse_statement()?;
                        Ok(Statement::Labelled {
                            label: ident.to_string(),
                            statement: Box::new(statement),
                        })
                    }
                    _ => {
                        // bail out, we could be doing something like "a = 1 + 2;"
                        let val = self.parse_expression(0)?;
                        self.expect(Token::Semicolon)?;
                        Ok(Statement::Expr(val))
                    }
                }
            }
            Some(_other) => {
                let val = self.parse_expression(0)?;
                self.expect(Token::Semicolon)?;
                Ok(Statement::Expr(val))
            }
            None => Err(ParserError::OutOfTokens),
        }
    }

    fn parse_binop(&mut self) -> Result<BinaryOp, ParserError> {
        match self.tokens.next() {
            Some(Token::Plus) => Ok(BinaryOp::Add),
            Some(Token::Hyphen) => Ok(BinaryOp::Subtract),
            Some(Token::Star) => Ok(BinaryOp::Multiply),
            Some(Token::Slash) => Ok(BinaryOp::Divide),
            Some(Token::Percent) => Ok(BinaryOp::Remainder),
            Some(Token::Ampersand) => Ok(BinaryOp::BitwiseAnd),
            Some(Token::Pipe) => Ok(BinaryOp::BitwiseOr),
            Some(Token::Caret) => Ok(BinaryOp::Xor),
            Some(Token::LessThanLessThan) => Ok(BinaryOp::ShiftLeft),
            Some(Token::GreaterThanGreaterThan) => Ok(BinaryOp::ShiftRight),
            Some(Token::EqualEqual) => Ok(BinaryOp::Equal),
            Some(Token::BangEqual) => Ok(BinaryOp::NotEqual),
            Some(Token::LessThan) => Ok(BinaryOp::LessThan),
            Some(Token::LessThanEqual) => Ok(BinaryOp::LessOrEqual),
            Some(Token::GreaterThan) => Ok(BinaryOp::GreaterThan),
            Some(Token::GreaterThanEqual) => Ok(BinaryOp::GreaterOrEqual),
            Some(Token::AmpersandAmpersand) => Ok(BinaryOp::BinAnd),
            Some(Token::PipePipe) => Ok(BinaryOp::BinOr),
            Some(Token::PlusEqual) => Ok(BinaryOp::AddAssign),
            Some(Token::HyphenEqual) => Ok(BinaryOp::MinusAssign),
            Some(Token::StarEqual) => Ok(BinaryOp::MultiplyAssign),
            Some(Token::SlashEqual) => Ok(BinaryOp::DivideAssign),
            Some(Token::PercentEqual) => Ok(BinaryOp::RemainderAssign),
            Some(Token::AmpersandEqual) => Ok(BinaryOp::BitwiseAndAssign),
            Some(Token::PipeEqual) => Ok(BinaryOp::BitwiseOrAssign),
            Some(Token::CaratEqual) => Ok(BinaryOp::XorAssign),
            Some(Token::LessThanLessThanEqual) => Ok(BinaryOp::ShiftLeftAssign),
            Some(Token::GreaterThanGreaterThanEqual) => Ok(BinaryOp::ShiftRightAssign),
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
                | Some(t @ Token::Pipe)
                | Some(t @ Token::EqualEqual)
                | Some(t @ Token::BangEqual)
                | Some(t @ Token::LessThan)
                | Some(t @ Token::LessThanEqual)
                | Some(t @ Token::GreaterThan)
                | Some(t @ Token::GreaterThanEqual)
                | Some(t @ Token::AmpersandAmpersand)
                | Some(t @ Token::PipePipe)
                | Some(t @ Token::LessThanLessThan)
                | Some(t @ Token::GreaterThanGreaterThan) => {
                    let new_prec = Self::precedence(*t)?;
                    if new_prec < min_precedence {
                        break;
                    }
                    let operator = self.parse_binop()?;
                    let right = self.parse_expression(new_prec + 1)?;
                    left = Expression::Binary(operator, Box::new(left), Box::new(right));
                }
                Some(t @ Token::PlusEqual)
                | Some(t @ Token::HyphenEqual)
                | Some(t @ Token::StarEqual)
                | Some(t @ Token::SlashEqual)
                | Some(t @ Token::PercentEqual)
                | Some(t @ Token::AmpersandEqual)
                | Some(t @ Token::PipeEqual)
                | Some(t @ Token::CaratEqual)
                | Some(t @ Token::LessThanLessThanEqual)
                | Some(t @ Token::GreaterThanGreaterThanEqual) => {
                    let new_prec = Self::precedence(*t)?;
                    if new_prec < min_precedence {
                        break;
                    }
                    let operator = self.parse_binop()?;
                    let right = self.parse_expression(new_prec)?;
                    left = Expression::Binary(operator, Box::new(left), Box::new(right));
                }
                Some(t @ Token::Equal) => {
                    // consume the equality, but since we're left associative
                    // make sure that the precendence is EQUAL to the equal prec
                    let new_prec = Self::precedence(*t)?;
                    if new_prec < min_precedence {
                        break;
                    }
                    self.tokens.next(); // consume the equal token
                    let right = self.parse_expression(new_prec)?;
                    left = Expression::Assignment(Box::new(left), Box::new(right));
                }
                Some(t @ Token::QuestionMark) => {
                    // we're parsing a conditional AKA ternary. We throw away
                    // the question mark, grab an inner expression,
                    // assert we see a colon and throw it away
                    // We must grab another expression.
                    let new_prec = Self::precedence(*t)?;
                    if new_prec < min_precedence {
                        break;
                    }
                    self.tokens.next(); // throw away question mark
                    let then = self.parse_expression(0)?; // reset precedence
                    let Some(Token::Colon) = self.tokens.next() else {
                        return Err(ParserError::UnexpectedTokenInConditional);
                    };
                    let else_ = self.parse_expression(new_prec)?;
                    left = Expression::Conditional {
                        condition: Box::new(left),
                        then: Box::new(then),
                        else_: Box::new(else_),
                    }
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
            Some(Token::Bang) => {
                let inner_exp = self.parse_factor()?;
                Ok(Expression::Unary(UnaryOp::Not, Box::new(inner_exp)))
            }
            Some(Token::LeftParen) => {
                // throw away parens and use just inner expression
                let inner_expr = self.parse_expression(0)?;
                self.expect(Token::RightParen)?;
                Ok(inner_expr)
            }
            Some(Token::Identifier(ident)) => Ok(Expression::Var(ident.to_string())),
            Some(t) => Err(ParserError::UnexpectedExpressionToken(t.into_string())),
            None => Err(ParserError::OutOfTokens),
        }
    }

    fn expect(&mut self, expected: Token<'a>) -> Result<(), ParserError> {
        self.expect_peek(expected)?;
        self.tokens.next();
        Ok(())
    }

    fn expect_peek(&mut self, expected: Token<'a>) -> Result<(), ParserError> {
        match self.tokens.peek() {
            Some(token) if std::mem::discriminant(*token) == std::mem::discriminant(&expected) => {
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
            Token::LessThanLessThan | Token::GreaterThanGreaterThan => Ok(35),
            Token::LessThan
            | Token::LessThanEqual
            | Token::GreaterThan
            | Token::GreaterThanEqual => Ok(34),
            Token::EqualEqual | Token::BangEqual => Ok(33),
            Token::Ampersand => Ok(32),
            Token::Caret => Ok(31),
            Token::Pipe => Ok(30),
            Token::AmpersandAmpersand => Ok(25),
            Token::PipePipe => Ok(24),
            // we may never look at the colon, but copying from the docs
            Token::QuestionMark | Token::Colon => Ok(3),
            Token::Equal
            | Token::PlusEqual
            | Token::HyphenEqual
            | Token::StarEqual
            | Token::SlashEqual
            | Token::PercentEqual
            | Token::AmpersandEqual
            | Token::PipeEqual
            | Token::CaratEqual
            | Token::LessThanLessThanEqual
            | Token::GreaterThanGreaterThanEqual => Ok(2),
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
            block: Block(vec![
                (BlockItem::Stmt(Statement::Return(Expression::Constant(100)))),
            ]),
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
            block: Block(vec![BlockItem::Stmt(Statement::Return(Expression::Unary(
                UnaryOp::Complement,
                Box::new(Expression::Constant(100)),
            )))]),
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
            block: Block(vec![BlockItem::Stmt(Statement::Return(Expression::Unary(
                UnaryOp::Negate,
                Box::new(Expression::Constant(100)),
            )))]),
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
            block: Block(vec![BlockItem::Stmt(Statement::Return(Expression::Unary(
                UnaryOp::Negate,
                Box::new(Expression::Constant(100)),
            )))]),
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
            block: Block(vec![BlockItem::Stmt(Statement::Return(
                Expression::Binary(
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
                ),
            ))]),
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
            block: Block(vec![BlockItem::Stmt(Statement::Return(
                Expression::Binary(
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
                ),
            ))]),
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
            block: Block(vec![BlockItem::Stmt(Statement::Return(
                Expression::Binary(
                    BinaryOp::BitwiseOr,
                    Box::new(Expression::Binary(
                        BinaryOp::Multiply,
                        Box::new(Expression::Constant(5)),
                        Box::new(Expression::Constant(4)),
                    )),
                    Box::new(Expression::Binary(
                        BinaryOp::BitwiseAnd,
                        Box::new(Expression::Binary(
                            BinaryOp::Subtract,
                            Box::new(Expression::Constant(4)),
                            Box::new(Expression::Constant(5)),
                        )),
                        Box::new(Expression::Constant(6)),
                    )),
                ),
            ))]),
        }));

        assert_eq!(expected, actual);
    }

    #[test]
    fn handles_shift_precedence() {
        /*
         * int main(void) {
         *   return 5 * 4 << 2;
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
            Token::LessThanLessThan,
            Token::Constant(2),
            Token::Semicolon,
            Token::RightBrace,
        ];
        let parse = Parser::new(&tokens);
        let ast = parse.into_ast();
        assert!(ast.is_ok());
        let actual = ast.unwrap();
        // multiple is a higher precedence than shift, so we
        // eval 5 * 4 first by pushing it to a lower node, then
        // leftshift the result by 2
        let expected = AST::Program(Box::new(AST::Function {
            name: "main",
            block: Block(vec![BlockItem::Stmt(Statement::Return(
                Expression::Binary(
                    BinaryOp::ShiftLeft,
                    Box::new(Expression::Binary(
                        BinaryOp::Multiply,
                        Box::new(Expression::Constant(5)),
                        Box::new(Expression::Constant(4)),
                    )),
                    Box::new(Expression::Constant(2)),
                ),
            ))]),
        }));

        assert_eq!(expected, actual);
    }

    #[test]
    fn handles_shift_precedence_when_rhs_is_expression() {
        /*
         * int main(void) {
         *   return 5 << (1 + 2);
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
            Token::LessThanLessThan,
            Token::LeftParen,
            Token::Constant(1),
            Token::Plus,
            Token::Constant(2),
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
            block: Block(vec![BlockItem::Stmt(Statement::Return(
                Expression::Binary(
                    BinaryOp::ShiftLeft,
                    Box::new(Expression::Constant(5)),
                    Box::new(Expression::Binary(
                        BinaryOp::Add,
                        Box::new(Expression::Constant(1)),
                        Box::new(Expression::Constant(2)),
                    )),
                ),
            ))]),
        }));

        assert_eq!(expected, actual);
    }

    #[test]
    fn handles_relational_precedence() {
        /*
         * int main(void) {
         *   return (2 <= 3 - 2) != 4 + 5;
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
            Token::LeftParen,
            Token::Constant(2),
            Token::LessThanEqual,
            Token::Constant(3),
            Token::Hyphen,
            Token::Constant(2),
            Token::RightParen,
            Token::BangEqual,
            Token::Constant(4),
            Token::Plus,
            Token::Constant(5),
            Token::Semicolon,
            Token::RightBrace,
        ];
        let parse = Parser::new(&tokens);
        let ast = parse.into_ast();
        assert!(ast.is_ok());
        let actual = ast.unwrap();
        let expected = AST::Program(Box::new(AST::Function {
            name: "main",
            block: Block(vec![BlockItem::Stmt(Statement::Return(
                Expression::Binary(
                    BinaryOp::NotEqual,
                    Box::new(Expression::Binary(
                        BinaryOp::LessOrEqual,
                        Box::new(Expression::Constant(2)),
                        Box::new(Expression::Binary(
                            BinaryOp::Subtract,
                            Box::new(Expression::Constant(3)),
                            Box::new(Expression::Constant(2)),
                        )),
                    )),
                    Box::new(Expression::Binary(
                        BinaryOp::Add,
                        Box::new(Expression::Constant(4)),
                        Box::new(Expression::Constant(5)),
                    )),
                ),
            ))]),
        }));

        assert_eq!(expected, actual);
    }

    #[test]
    fn basic_assignment() {
        /*
         * int main(void) {
         *   int a = 1;
         *   return a;
         * }
         */
        let tokens = vec![
            Token::Int,
            Token::Main,
            Token::LeftParen,
            Token::Void,
            Token::RightParen,
            Token::LeftBrace,
            Token::Int,
            Token::Identifier("a"),
            Token::Equal,
            Token::Constant(1),
            Token::Semicolon,
            Token::Return,
            Token::Identifier("a"),
            Token::Semicolon,
            Token::RightBrace,
        ];
        let parse = Parser::new(&tokens);
        let ast = parse.into_ast();
        assert!(ast.is_ok());
        let actual = ast.unwrap();
        let expected = AST::Program(Box::new(AST::Function {
            name: "main",
            block: Block(vec![
                BlockItem::Decl(Declaration {
                    name: "a".into(),
                    init: Some(Expression::Constant(1)),
                }),
                BlockItem::Stmt(Statement::Return(Expression::Var("a".into()))),
            ]),
        }));

        assert_eq!(expected, actual)
    }

    #[test]
    fn left_assoc_assignment() {
        /*
         * int main(void) {
         *   int a;
         *   return a = b = c;
         * }
         */
        let tokens = vec![
            Token::Int,
            Token::Main,
            Token::LeftParen,
            Token::Void,
            Token::RightParen,
            Token::LeftBrace,
            Token::Int,
            Token::Identifier("a"),
            Token::Semicolon,
            Token::Return,
            Token::Identifier("a"),
            Token::Equal,
            Token::Identifier("b"),
            Token::Equal,
            Token::Identifier("c"),
            Token::Semicolon,
            Token::RightBrace,
        ];
        let parse = Parser::new(&tokens);
        let ast = parse.into_ast();
        assert!(ast.is_ok());
        let actual = ast.unwrap();
        let expected = AST::Program(Box::new(AST::Function {
            name: "main",
            block: Block(vec![
                BlockItem::Decl(Declaration {
                    name: "a".into(),
                    init: None,
                }),
                BlockItem::Stmt(Statement::Return(Expression::Assignment(
                    Box::new(Expression::Var("a".into())),
                    Box::new(Expression::Assignment(
                        Box::new(Expression::Var("b".into())),
                        Box::new(Expression::Var("c".into())),
                    )),
                ))),
            ]),
        }));

        assert_eq!(expected, actual)
    }

    #[test]
    fn expression_statement() {
        /*
         * int main(void) {
         *   2 + 2;
         *   return 0;
         * }
         */
        let tokens = vec![
            Token::Int,
            Token::Main,
            Token::LeftParen,
            Token::Void,
            Token::RightParen,
            Token::LeftBrace,
            Token::Constant(2),
            Token::Plus,
            Token::Constant(2),
            Token::Semicolon,
            Token::Return,
            Token::Constant(0),
            Token::Semicolon,
            Token::RightBrace,
        ];
        let parse = Parser::new(&tokens);
        let ast = parse.into_ast();
        assert!(ast.is_ok());
        let actual = ast.unwrap();
        let expected = AST::Program(Box::new(AST::Function {
            name: "main",
            block: Block(vec![
                BlockItem::Stmt(Statement::Expr(Expression::Binary(
                    BinaryOp::Add,
                    Box::new(Expression::Constant(2)),
                    Box::new(Expression::Constant(2)),
                ))),
                BlockItem::Stmt(Statement::Return(Expression::Constant(0))),
            ]),
        }));

        assert_eq!(expected, actual)
    }

    #[test]
    fn compound_assignment() {
        /*
         * int main(void) {
         *   int a = 1;
         *   a += 2;
         *   a -= 2;
         *   a *= 2;
         *   a /= 2;
         *   a %= 2;
         *   return a;
         * }
         */
        let tokens = vec![
            Token::Int,
            Token::Main,
            Token::LeftParen,
            Token::Void,
            Token::RightParen,
            Token::LeftBrace,
            Token::Int,
            Token::Identifier("a"),
            Token::Equal,
            Token::Constant(1),
            Token::Semicolon,
            Token::Identifier("a"),
            Token::PlusEqual,
            Token::Constant(2),
            Token::Semicolon,
            Token::Identifier("a"),
            Token::HyphenEqual,
            Token::Constant(2),
            Token::Semicolon,
            Token::Identifier("a"),
            Token::StarEqual,
            Token::Constant(2),
            Token::Semicolon,
            Token::Identifier("a"),
            Token::SlashEqual,
            Token::Constant(2),
            Token::Semicolon,
            Token::Identifier("a"),
            Token::PercentEqual,
            Token::Constant(2),
            Token::Semicolon,
            Token::Return,
            Token::Identifier("a"),
            Token::Semicolon,
            Token::RightBrace,
        ];
        let parse = Parser::new(&tokens);
        let ast = parse.into_ast();
        assert!(ast.is_ok());
        let actual = ast.unwrap();
        let expected = AST::Program(Box::new(AST::Function {
            name: "main",
            block: Block(vec![
                BlockItem::Decl(Declaration {
                    name: "a".into(),
                    init: Some(Expression::Constant(1)),
                }),
                BlockItem::Stmt(Statement::Expr(Expression::Binary(
                    BinaryOp::AddAssign,
                    Box::new(Expression::Var("a".into())),
                    Box::new(Expression::Constant(2)),
                ))),
                BlockItem::Stmt(Statement::Expr(Expression::Binary(
                    BinaryOp::MinusAssign,
                    Box::new(Expression::Var("a".into())),
                    Box::new(Expression::Constant(2)),
                ))),
                BlockItem::Stmt(Statement::Expr(Expression::Binary(
                    BinaryOp::MultiplyAssign,
                    Box::new(Expression::Var("a".into())),
                    Box::new(Expression::Constant(2)),
                ))),
                BlockItem::Stmt(Statement::Expr(Expression::Binary(
                    BinaryOp::DivideAssign,
                    Box::new(Expression::Var("a".into())),
                    Box::new(Expression::Constant(2)),
                ))),
                BlockItem::Stmt(Statement::Expr(Expression::Binary(
                    BinaryOp::RemainderAssign,
                    Box::new(Expression::Var("a".into())),
                    Box::new(Expression::Constant(2)),
                ))),
                BlockItem::Stmt(Statement::Return(Expression::Var("a".into()))),
            ]),
        }));

        assert_eq!(expected, actual)
    }

    #[test]
    fn small_compound_assignment() {
        /*
         * int main(void) {
         *   int a = 1;
         *   int b = 0;
         *   int c = 2;
         *   a += b -= c = 5;
         *   return a;
         * }
         */
        let tokens = vec![
            Token::Int,
            Token::Main,
            Token::LeftParen,
            Token::Void,
            Token::RightParen,
            Token::LeftBrace,
            Token::Int,
            Token::Identifier("a"),
            Token::Equal,
            Token::Constant(1),
            Token::Semicolon,
            Token::Int,
            Token::Identifier("b"),
            Token::Equal,
            Token::Constant(0),
            Token::Semicolon,
            Token::Int,
            Token::Identifier("c"),
            Token::Equal,
            Token::Constant(2),
            Token::Semicolon,
            Token::Identifier("a"),
            Token::PlusEqual,
            Token::Identifier("b"),
            Token::HyphenEqual,
            Token::Identifier("c"),
            Token::Equal,
            Token::Constant(5),
            Token::Semicolon,
            Token::Return,
            Token::Identifier("a"),
            Token::Semicolon,
            Token::RightBrace,
        ];
        let parse = Parser::new(&tokens);
        let ast = parse.into_ast();
        assert!(ast.is_ok());
        let actual = ast.unwrap();
        let expected = AST::Program(Box::new(AST::Function {
            name: "main",
            block: Block(vec![
                BlockItem::Decl(Declaration {
                    name: "a".into(),
                    init: Some(Expression::Constant(1)),
                }),
                BlockItem::Decl(Declaration {
                    name: "b".into(),
                    init: Some(Expression::Constant(0)),
                }),
                BlockItem::Decl(Declaration {
                    name: "c".into(),
                    init: Some(Expression::Constant(2)),
                }),
                BlockItem::Stmt(Statement::Expr(Expression::Binary(
                    BinaryOp::AddAssign,
                    Box::new(Expression::Var("a".into())),
                    Box::new(Expression::Binary(
                        BinaryOp::MinusAssign,
                        Box::new(Expression::Var("b".into())),
                        Box::new(Expression::Assignment(
                            Box::new(Expression::Var("c".into())),
                            Box::new(Expression::Constant(5)),
                        )),
                    )),
                ))),
                BlockItem::Stmt(Statement::Return(Expression::Var("a".into()))),
            ]),
        }));

        assert_eq!(expected, actual)
    }

    #[test]
    fn another_small_compound_assignment() {
        /*
         * int main(void) {
         *   int a = 10;
         *   (a += 1) -= 2;
         * }
         */
        let tokens = vec![
            Token::Int,
            Token::Main,
            Token::LeftParen,
            Token::Void,
            Token::RightParen,
            Token::LeftBrace,
            Token::Int,
            Token::Identifier("a"),
            Token::Equal,
            Token::Constant(10),
            Token::Semicolon,
            Token::LeftParen,
            Token::Identifier("a"),
            Token::PlusEqual,
            Token::Constant(1),
            Token::RightParen,
            Token::HyphenEqual,
            Token::Constant(2),
            Token::Semicolon,
            Token::RightBrace,
        ];
        let parse = Parser::new(&tokens);
        let ast = parse.into_ast();
        assert!(ast.is_ok());
        let actual = ast.unwrap();
        let expected = AST::Program(Box::new(AST::Function {
            name: "main",
            block: Block(vec![
                BlockItem::Decl(Declaration {
                    name: "a".into(),
                    init: Some(Expression::Constant(10)),
                }),
                BlockItem::Stmt(Statement::Expr(Expression::Binary(
                    BinaryOp::MinusAssign,
                    Box::new(Expression::Binary(
                        BinaryOp::AddAssign,
                        Box::new(Expression::Var("a".into())),
                        Box::new(Expression::Constant(1)),
                    )),
                    Box::new(Expression::Constant(2)),
                ))),
            ]),
        }));

        assert_eq!(expected, actual)
    }

    #[test]
    fn test_dangling_else() {
        let src = r#"
        int main(void) {
            if (a)
                if (a > 10)
                    return a;
                else
                    return 10 - a;
        }
        "#;
        let lexer = crate::lexer::Lexer::lex(src).unwrap();
        let tokens = lexer.as_syntactic_tokens();
        let parse = Parser::new(&tokens);
        let ast = parse.into_ast();
        assert!(ast.is_ok());

        let expected = AST::Program(Box::new(AST::Function {
            name: "main",
            block: Block(vec![BlockItem::Stmt(Statement::If {
                condition: Expression::Var("a".into()),
                then: Box::new(Statement::If {
                    condition: Expression::Binary(
                        BinaryOp::GreaterThan,
                        Box::new(Expression::Var("a".into())),
                        Box::new(Expression::Constant(10)),
                    ),
                    then: Box::new(Statement::Return(Expression::Var("a".into()))),
                    else_: Some(Box::new(Statement::Return(Expression::Binary(
                        BinaryOp::Subtract,
                        Box::new(Expression::Constant(10)),
                        Box::new(Expression::Var("a".into())),
                    )))),
                }),
                else_: None,
            })]),
        }));

        assert_eq!(ast.unwrap(), expected);
    }

    #[test]
    fn test_conditional_expressions() {
        let src = r#"
        int main(void) {
            a ? b ? 1 : 2 : 3;
            a ? 1 : b ? 2 : 3;
            a || b ? 2 : 3;
            x ? x = 1 : 2;
        }
        "#;
        let lexer = crate::lexer::Lexer::lex(src).unwrap();
        let tokens = lexer.as_syntactic_tokens();
        let parse = Parser::new(&tokens);
        let ast = parse.into_ast();
        assert!(ast.is_ok());

        let expected = AST::Program(Box::new(AST::Function {
            name: "main",
            block: Block(vec![
                // first example:  we nest inner expression
                BlockItem::Stmt(Statement::Expr(Expression::Conditional {
                    condition: Box::new(Expression::Var("a".into())),
                    then: Box::new(Expression::Conditional {
                        condition: Box::new(Expression::Var("b".into())),
                        then: Box::new(Expression::Constant(1)),
                        else_: Box::new(Expression::Constant(2)),
                    }),
                    else_: Box::new(Expression::Constant(3)),
                })),
                // second example: else statement is nested parse
                BlockItem::Stmt(Statement::Expr(Expression::Conditional {
                    condition: Box::new(Expression::Var("a".into())),
                    then: Box::new(Expression::Constant(1)),
                    else_: Box::new(Expression::Conditional {
                        condition: Box::new(Expression::Var("b".into())),
                        then: Box::new(Expression::Constant(2)),
                        else_: Box::new(Expression::Constant(3)),
                    }),
                })),
                // third example: the conditional is a short-circuit expr
                BlockItem::Stmt(Statement::Expr(Expression::Conditional {
                    condition: Box::new(Expression::Binary(
                        BinaryOp::BinOr,
                        Box::new(Expression::Var("a".into())),
                        Box::new(Expression::Var("b".into())),
                    )),
                    then: Box::new(Expression::Constant(2)),
                    else_: Box::new(Expression::Constant(3)),
                })),
                // fourth example: then should be an assignment
                BlockItem::Stmt(Statement::Expr(Expression::Conditional {
                    condition: Box::new(Expression::Var("x".into())),
                    then: Box::new(Expression::Assignment(
                        Box::new(Expression::Var("x".into())),
                        Box::new(Expression::Constant(1)),
                    )),
                    else_: Box::new(Expression::Constant(2)),
                })),
            ]),
        }));

        assert_eq!(ast.unwrap(), expected);
    }

    #[test]
    fn test_goto_and_labelled_statements() {
        let src = r#"
        int main(void) {
            goto foo;
            foo:
                return 1 + 2;
        }
        "#;
        let lexer = crate::lexer::Lexer::lex(src).unwrap();
        let tokens = lexer.as_syntactic_tokens();
        let parse = Parser::new(&tokens);
        let ast = parse.into_ast();
        assert!(ast.is_ok());

        let expected = AST::Program(Box::new(AST::Function {
            name: "main",
            block: Block(vec![
                BlockItem::Stmt(Statement::Goto("foo".into())),
                BlockItem::Stmt(Statement::Labelled {
                    label: "foo".into(),
                    statement: Box::new(Statement::Return(Expression::Binary(
                        BinaryOp::Add,
                        Box::new(Expression::Constant(1)),
                        Box::new(Expression::Constant(2)),
                    ))),
                }),
            ]),
        }));

        assert_eq!(ast.unwrap(), expected);
    }

    #[test]
    fn test_compound_statements_and_blocks() {
        let src = r#"
        int main(void) {
            if (1) {
                int x = 1;
                return x;
            }
            {
                int b = 2;
                b + 1;
            }
        }
        "#;
        let lexer = crate::lexer::Lexer::lex(src).unwrap();
        let tokens = lexer.as_syntactic_tokens();
        let parse = Parser::new(&tokens);
        let ast = parse.into_ast();
        assert!(ast.is_ok());

        let expected = AST::Program(Box::new(AST::Function {
            name: "main",
            block: Block(vec![
                BlockItem::Stmt(Statement::If {
                    condition: Expression::Constant(1),
                    then: Box::new(Statement::Compound(Block(vec![
                        BlockItem::Decl(Declaration {
                            name: "x".into(),
                            init: Some(Expression::Constant(1)),
                        }),
                        BlockItem::Stmt(Statement::Return(Expression::Var("x".into()))),
                    ]))),
                    else_: None,
                }),
                BlockItem::Stmt(Statement::Compound(Block(vec![
                    BlockItem::Decl(Declaration {
                        name: "b".into(),
                        init: Some(Expression::Constant(2)),
                    }),
                    BlockItem::Stmt(Statement::Expr(Expression::Binary(
                        BinaryOp::Add,
                        Box::new(Expression::Var("b".into())),
                        Box::new(Expression::Constant(1)),
                    ))),
                ]))),
            ]),
        }));

        assert_eq!(ast.unwrap(), expected);
    }
}
