pub use crate::ast::{BinaryOp, Const, StorageClass, UnaryOp};
use crate::lexer::Token;
use crate::types::CType;
use std::collections::HashSet;
use std::iter::Peekable;
use thiserror::Error;

pub type AST = crate::ast::AST<()>;
pub type Declaration = crate::ast::Declaration<()>;
pub type Statement = crate::ast::Statement<()>;
pub type ExprKind = crate::ast::ExprKind<()>;
pub type Expression = crate::ast::Expression<()>;
pub type FunctionDeclaration = crate::ast::FunctionDeclaration<()>;
pub type VariableDeclaration = crate::ast::VariableDeclaration<()>;
pub type ForInit = crate::ast::ForInit<()>;
pub type BlockItem = crate::ast::BlockItem<()>;
pub type Block = crate::ast::Block<()>;

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
    #[error("Found a non-variable declaration in a For loop initializer")]
    WrongDeclarationInForInit,
    #[error("Expected an identifier name after a type in a function parameter list")]
    ExpectedIdentifierAfterType,
    #[error("Invalid type specifier")]
    InvalidTypeSpecifier,
    #[error("Invalid storage class")]
    InvalidStorageClass,
    #[error("Integer too large to fit in a C int or long")]
    IntegerTooLargeToFitInConstant,
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

    pub fn into_ast(mut self) -> Result<AST, ParserError> {
        let program = self.parse_program()?;
        match self.tokens.next() {
            // ensure no leftover tokens
            None => Ok(program),
            Some(_) => Err(ParserError::LeftoverTokens),
        }
    }

    fn parse_program(&mut self) -> Result<AST, ParserError> {
        let mut decls = vec![];
        loop {
            if self.tokens.peek().is_none() {
                break;
            }
            decls.push(self.parse_declaration()?);
        }
        Ok(AST::Program(decls))
    }

    fn parse_storage_class(classes: &[StorageClass]) -> Result<Option<StorageClass>, ParserError> {
        if classes.len() > 1 {
            return Err(ParserError::InvalidStorageClass);
        }
        Ok(classes.get(0).copied())
    }

    fn parse_type(specifier_list: &[Token]) -> Result<CType, ParserError> {
        // tokens are allowed here in any order, so here's what we do
        // First, check for empty list. If nothing, we error.
        // Second, look for duplicates: no 'int int' - not implementing 'long long'.
        // Third, look for conflicts - cant have both signed and unsigned.
        if specifier_list.is_empty() {
            return Err(ParserError::InvalidTypeSpecifier);
        }
        let set: HashSet<&Token> = HashSet::from_iter(specifier_list);
        if set.len() != specifier_list.len() {
            // got duplicates
            return Err(ParserError::InvalidTypeSpecifier);
        }
        // can't combine double with any other specifier
        if set.contains(&Token::Double) {
            if set.len() == 1 {
                return Ok(CType::Double);
            }
            return Err(ParserError::InvalidTypeSpecifier);
        }
        if set.contains(&Token::Unsigned) {
            if set.contains(&Token::Signed) {
                return Err(ParserError::InvalidTypeSpecifier);
            }
            if set.contains(&Token::Long) {
                return Ok(CType::ULong);
            }
            return Ok(CType::UInt);
        }
        // at this point, may or may not contain signed Token but MUST be signed type
        if set.contains(&Token::Long) {
            return Ok(CType::Long);
        }
        Ok(CType::Int)
    }

    fn parse_type_and_storage_class(
        &mut self,
    ) -> Result<(CType, Option<StorageClass>), ParserError> {
        let mut types = vec![];
        let mut classes = vec![];

        loop {
            match self.tokens.peek() {
                Some(Token::Int) => types.push(Token::Int),
                Some(Token::Long) => types.push(Token::Long),
                Some(Token::Signed) => types.push(Token::Signed),
                Some(Token::Unsigned) => types.push(Token::Unsigned),
                Some(Token::Double) => types.push(Token::Double),
                Some(Token::Static) => classes.push(StorageClass::Static),
                Some(Token::Extern) => classes.push(StorageClass::Extern),
                _ => break,
            }
            self.tokens.next();
        }
        let type_ = Self::parse_type(&types)?;
        let storage_class = Self::parse_storage_class(&classes)?;
        Ok((type_, storage_class))
    }

    /*
     * <function-declaration> ::= { <specifier> }+ <identifier> "(" <param-list> ")" ( <block> | ";")
     * <param-list> ::= "void"
     *                | { <type-specifier> }+ <identifier> { "," { <type-specifier> }+ <identifier> }
     */
    fn parse_function(
        &mut self,
        name: String,
        storage_class: Option<StorageClass>,
        return_type: CType,
    ) -> Result<FunctionDeclaration, ParserError> {
        self.expect(Token::LeftParen)?;
        // check if we're void, or taking a param list
        // Should we store Void in the identifier list?
        // Emptiness is a way of knowing it's a void function.
        let mut params: Vec<(String, CType)> = vec![];
        if let Some(Token::Void) = self.tokens.peek() {
            self.expect(Token::Void)?;
            self.expect(Token::RightParen)?;
        } else {
            loop {
                // collect token until we see an identifier, then
                // we'll parse a CType
                let (ty_, storage_class) = self.parse_type_and_storage_class()?;
                if storage_class.is_some() {
                    return Err(ParserError::InvalidStorageClass);
                }
                let Some(Token::Identifier(ident)) = self.tokens.next() else {
                    return Err(ParserError::ExpectedIdentifierAfterType);
                };
                params.push((ident.to_string(), ty_));
                if let Some(Token::RightParen) = self.tokens.peek() {
                    // consume the right paren
                    self.tokens.next();
                    break;
                };
                self.expect(Token::Comma)?;
            }
        }
        let block = match self.tokens.peek() {
            Some(Token::Semicolon) => {
                self.tokens.next();
                None
            }
            _ => {
                let block = self.parse_block()?;
                Some(block)
            }
        };

        let decl = FunctionDeclaration::new(name, params, block, storage_class, return_type);
        Ok(decl)
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

        Ok(crate::ast::Block(body))
    }

    fn parse_block_item(&mut self) -> Result<BlockItem, ParserError> {
        let block_item = if token_is_valid_specifier(self.tokens.peek().copied()) {
            let decl = self.parse_declaration()?;
            BlockItem::Decl(decl)
        } else {
            let stmt = self.parse_statement()?;
            BlockItem::Stmt(stmt)
        };

        Ok(block_item)
    }

    fn parse_for_init(&mut self) -> Result<ForInit, ParserError> {
        // technically this matches static and extern, but
        // we'll reject at typecheck time
        if token_is_valid_specifier(self.tokens.peek().copied()) {
            let decl = self.parse_declaration()?;
            let Declaration::VarDecl(var_decl) = decl else {
                return Err(ParserError::WrongDeclarationInForInit);
            };
            Ok(ForInit::InitDecl(var_decl))
        } else {
            let expr = self.parse_optional_expr(Token::Semicolon)?;
            Ok(ForInit::InitExp(expr))
        }
    }

    fn parse_optional_expr(
        &mut self,
        terminator: Token<'a>,
    ) -> Result<Option<Expression>, ParserError> {
        // if the next token is our termination token, just consume and return none.
        // Else, parse an expression, ensure we end with the terminator and return the expr
        match self.tokens.peek() {
            Some(token) if **token == terminator => {
                self.tokens.next();
                Ok(None)
            }
            _ => {
                let expr = self.parse_expression(0)?;
                self.expect(terminator)?;
                Ok(Some(expr))
            }
        }
    }

    fn parse_identifier(&mut self) -> Result<String, ParserError> {
        match self.tokens.next() {
            Some(Token::Main) => Ok("main".into()),
            Some(Token::Identifier(ident)) => Ok((*ident).into()),
            _ => Err(ParserError::UnexpectedName),
        }
    }

    /*
     * var-decl ::= { <specifier> }+ <identifier> [ '=' <expr>] ';'
     * fun-decl::= { <specifier> }+ <identifier> '(' <param-list> ')' ( <block> | ';' )
     * param-list ::= "void" | "int" <identifier> { ',' "int" identifier }
     * <specifier> ::= <type-specifier> | "static" | "extern"
     *
     * since vardecl and fundecl start the same, but differ
     * when we see a paren or not, let's grab all specifiers,
     * then the identifier, then based on what we see we'll parse accordingly but
     * pass in the relevant added values.
     */
    fn parse_declaration(&mut self) -> Result<Declaration, ParserError> {
        let (ctype, storage_class) = self.parse_type_and_storage_class()?;
        let identifier = self.parse_identifier()?;
        if let Some(Token::LeftParen) = self.tokens.peek() {
            let func = self.parse_function(identifier, storage_class, ctype)?;
            Ok(Declaration::FunDecl(func))
        } else {
            let init = if let Some(Token::Equal) = self.tokens.peek() {
                self.tokens.next();
                let init = self.parse_expression(0)?;
                self.expect(Token::Semicolon)?;
                Some(init)
            } else {
                self.expect(Token::Semicolon)?;
                None
            };

            Ok(Declaration::VarDecl(VariableDeclaration {
                name: identifier.into(),
                init,
                storage_class,
                vtype: ctype,
            }))
        }
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
                        ));
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
                // crate::ast HACKY CLONE! TODO - fix!
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
            Some(Token::Break) => {
                self.tokens.next();
                self.expect(Token::Semicolon)?;
                Ok(Statement::Break(self.dummy_label()))
            }
            Some(Token::Continue) => {
                self.tokens.next();
                self.expect(Token::Semicolon)?;
                Ok(Statement::Continue(self.dummy_label()))
            }
            Some(Token::While) => {
                self.tokens.next();
                self.expect(Token::LeftParen)?;
                let exp = self.parse_expression(0)?;
                self.expect(Token::RightParen)?;
                let stmt = self.parse_statement()?;
                Ok(Statement::While {
                    condition: exp,
                    body: Box::new(stmt),
                    label: self.dummy_label(),
                })
            }
            Some(Token::Do) => {
                self.tokens.next();
                let stmt = self.parse_statement()?;
                self.expect(Token::While)?;
                self.expect(Token::LeftParen)?;
                let exp = self.parse_expression(0)?;
                self.expect(Token::RightParen)?;
                self.expect(Token::Semicolon)?;
                Ok(Statement::DoWhile {
                    condition: exp,
                    body: Box::new(stmt),
                    label: self.dummy_label(),
                })
            }
            Some(Token::For) => {
                self.tokens.next();
                self.expect(Token::LeftParen)?;
                let for_init = self.parse_for_init()?;
                let condition = self.parse_optional_expr(Token::Semicolon)?;
                let post = self.parse_optional_expr(Token::RightParen)?;
                let body = self.parse_statement()?;
                Ok(Statement::For {
                    init: for_init,
                    condition,
                    post,
                    body: Box::new(body),
                    label: self.dummy_label(),
                })
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
                    left = Expression::new(ExprKind::Binary(
                        operator,
                        Box::new(left),
                        Box::new(right),
                    ));
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
                    left = Expression::new(ExprKind::Binary(
                        operator,
                        Box::new(left),
                        Box::new(right),
                    ));
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
                    left = Expression::new(ExprKind::Assignment(Box::new(left), Box::new(right)));
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
                    left = Expression::new(ExprKind::Conditional {
                        condition: Box::new(left),
                        then: Box::new(then),
                        else_: Box::new(else_),
                    })
                }
                _ => break,
            }
        }

        Ok(left)
    }

    fn parse_constant(tok: Token) -> Result<Const, ParserError> {
        match tok {
            Token::Constant(c) if c <= i32::MAX as usize => Ok(Const::Int(c.try_into().unwrap())),
            // promote signed int to signed long if c > i32
            Token::Constant(c) | Token::LongConstant(c) => {
                Ok(Const::Long(c.try_into().map_err(|_| {
                    ParserError::IntegerTooLargeToFitInConstant
                })?))
            }
            Token::UnsignedIntConstant(c) if c <= u32::MAX as usize => {
                Ok(Const::UInt(c.try_into().unwrap()))
            }
            // promote unsigned int to unsigned long if c > u32
            Token::UnsignedIntConstant(c) | Token::UnsignedLongConstant(c) => {
                Ok(Const::ULong(c.try_into().map_err(|_| {
                    ParserError::IntegerTooLargeToFitInConstant
                })?))
            }
            Token::FloatingPointConstant(s) => {
                // s is some string that could be scientific notation, or just a normal decimal.
                let c: f64 = s
                    .parse::<f64>()
                    .expect("Should get a float64 from a FloatingPointConstant");
                Ok(Const::Double(c))
            }
            _ => panic!("Somehow tried to parse a token that isn't a constant, got {tok:?}"),
        }
    }

    fn parse_factor(&mut self) -> Result<Expression, ParserError> {
        match self.tokens.next() {
            Some(
                t @ (Token::Constant(_)
                | Token::LongConstant(_)
                | Token::UnsignedLongConstant(_)
                | Token::UnsignedIntConstant(_)
                | Token::FloatingPointConstant(_)),
            ) => {
                let constant = Self::parse_constant(*t)?;
                Ok(Expression::new(ExprKind::Constant(constant)))
            }
            Some(Token::Hyphen) => {
                let inner_exp = self.parse_factor()?;
                Ok(Expression::new(ExprKind::Unary(
                    UnaryOp::Negate,
                    Box::new(inner_exp),
                )))
            }
            Some(Token::Tilde) => {
                let inner_exp = self.parse_factor()?;
                Ok(Expression::new(ExprKind::Unary(
                    UnaryOp::Complement,
                    Box::new(inner_exp),
                )))
            }
            Some(Token::Bang) => {
                let inner_exp = self.parse_factor()?;
                Ok(Expression::new(ExprKind::Unary(
                    UnaryOp::Not,
                    Box::new(inner_exp),
                )))
            }
            Some(Token::LeftParen) => {
                // if the inside is a valid type specifier, we'll scoop up
                // specifiers and turn this into a Cast expression. Else,
                // throw away parens and use just inner expression
                if token_is_valid_specifier(self.tokens.peek().copied()) {
                    let (type_, storage_class) = self.parse_type_and_storage_class()?;
                    if storage_class.is_some() {
                        return Err(ParserError::InvalidStorageClass);
                    }
                    self.expect(Token::RightParen)?;
                    let expr = self.parse_factor()?;
                    Ok(Expression::new(ExprKind::Cast(type_, Box::new(expr))))
                } else {
                    let inner_expr = self.parse_expression(0)?;
                    self.expect(Token::RightParen)?;
                    Ok(inner_expr)
                }
            }
            Some(Token::Identifier(ident)) => {
                if let Some(Token::LeftParen) = self.tokens.peek() {
                    // function call, so get the inner expressions and attach
                    self.tokens.next();
                    // special case here, handle the empty argument list
                    let args = match self.tokens.peek() {
                        Some(Token::RightParen) => {
                            self.tokens.next();
                            vec![]
                        }
                        _ => {
                            let mut args = vec![];
                            loop {
                                args.push(self.parse_expression(0)?);
                                if let Some(Token::RightParen) = self.tokens.peek() {
                                    self.tokens.next();
                                    break;
                                }
                                self.expect(Token::Comma)?;
                            }
                            args
                        }
                    };
                    Ok(Expression::new(ExprKind::FunctionCall {
                        name: ident.to_string(),
                        args,
                    }))
                } else {
                    Ok(Expression::new(ExprKind::Var(ident.to_string())))
                }
            }
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
            Token::LeftParen => Ok(60),
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

    #[inline(always)]
    fn dummy_label(&self) -> String {
        "".into()
    }
}

fn token_is_valid_specifier(tok: Option<&Token>) -> bool {
    token_is_valid_type_specifier(tok) || token_is_valid_storage_class(tok)
}

fn token_is_valid_type_specifier(tok: Option<&Token>) -> bool {
    tok.map(|t| {
        matches!(
            t,
            Token::Int | Token::Long | Token::Signed | Token::Unsigned | Token::Double
        )
    })
    .unwrap_or(false)
}

fn token_is_valid_storage_class(tok: Option<&Token>) -> bool {
    tok.map(|t| matches!(t, Token::Static | Token::Extern))
        .unwrap_or(false)
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
        let expected = AST::Program(vec![Declaration::FunDecl(FunctionDeclaration::new(
            "main".into(),
            vec![],
            Some(crate::ast::Block(vec![BlockItem::Stmt(Statement::Return(
                Expression::new(ExprKind::Constant(Const::Int(100))),
            ))])),
            None,
            CType::Int,
        ))]);
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
        assert_eq!(Err(ParserError::InvalidTypeSpecifier), ast);
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
        let expected = AST::Program(vec![Declaration::FunDecl(FunctionDeclaration::new(
            "main".into(),
            vec![],
            Some(crate::ast::Block(vec![BlockItem::Stmt(Statement::Return(
                Expression::new(ExprKind::Unary(
                    UnaryOp::Complement,
                    Box::new(Expression::new(ExprKind::Constant(Const::Int(100)))),
                )),
            ))])),
            None,
            CType::Int,
        ))]);
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
        let expected = AST::Program(vec![Declaration::FunDecl(FunctionDeclaration::new(
            "main".into(),
            vec![],
            Some(crate::ast::Block(vec![BlockItem::Stmt(Statement::Return(
                Expression::new(ExprKind::Unary(
                    UnaryOp::Negate,
                    Box::new(Expression::new(ExprKind::Constant(Const::Int(100)))),
                )),
            ))])),
            None,
            CType::Int,
        ))]);
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
        let expected = AST::Program(vec![Declaration::FunDecl(FunctionDeclaration::new(
            "main".into(),
            vec![],
            Some(crate::ast::Block(vec![BlockItem::Stmt(Statement::Return(
                Expression::new(ExprKind::Unary(
                    UnaryOp::Negate,
                    Box::new(Expression::new(ExprKind::Constant(Const::Int(100)))),
                )),
            ))])),
            None,
            CType::Int,
        ))]);
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
        let expected = AST::Program(vec![Declaration::FunDecl(FunctionDeclaration::new(
            "main".into(),
            vec![],
            Some(crate::ast::Block(vec![BlockItem::Stmt(Statement::Return(
                Expression::new(ExprKind::Binary(
                    BinaryOp::Subtract,
                    Box::new(Expression::new(ExprKind::Binary(
                        BinaryOp::Multiply,
                        Box::new(Expression::new(ExprKind::Constant(Const::Int(1)))),
                        Box::new(Expression::new(ExprKind::Constant(Const::Int(2)))),
                    ))),
                    Box::new(Expression::new(ExprKind::Binary(
                        BinaryOp::Multiply,
                        Box::new(Expression::new(ExprKind::Constant(Const::Int(3)))),
                        Box::new(Expression::new(ExprKind::Binary(
                            BinaryOp::Add,
                            Box::new(Expression::new(ExprKind::Constant(Const::Int(4)))),
                            Box::new(Expression::new(ExprKind::Constant(Const::Int(5)))),
                        ))),
                    ))),
                )),
            ))])),
            None,
            CType::Int,
        ))]);

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
        let expected = AST::Program(vec![Declaration::FunDecl(FunctionDeclaration::new(
            "main".into(),
            vec![],
            Some(crate::ast::Block(vec![BlockItem::Stmt(Statement::Return(
                Expression::new(ExprKind::Binary(
                    BinaryOp::Subtract,
                    Box::new(Expression::new(ExprKind::Binary(
                        BinaryOp::Divide,
                        Box::new(Expression::new(ExprKind::Binary(
                            BinaryOp::Multiply,
                            Box::new(Expression::new(ExprKind::Constant(Const::Int(5)))),
                            Box::new(Expression::new(ExprKind::Constant(Const::Int(4)))),
                        ))),
                        Box::new(Expression::new(ExprKind::Constant(Const::Int(2)))),
                    ))),
                    Box::new(Expression::new(ExprKind::Binary(
                        BinaryOp::Remainder,
                        Box::new(Expression::new(ExprKind::Constant(Const::Int(3)))),
                        Box::new(Expression::new(ExprKind::Binary(
                            BinaryOp::Add,
                            Box::new(Expression::new(ExprKind::Constant(Const::Int(2)))),
                            Box::new(Expression::new(ExprKind::Constant(Const::Int(1)))),
                        ))),
                    ))),
                )),
            ))])),
            None,
            CType::Int,
        ))]);

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
        let expected = AST::Program(vec![Declaration::FunDecl(FunctionDeclaration::new(
            "main".into(),
            vec![],
            Some(crate::ast::Block(vec![BlockItem::Stmt(Statement::Return(
                Expression::new(ExprKind::Binary(
                    BinaryOp::BitwiseOr,
                    Box::new(Expression::new(ExprKind::Binary(
                        BinaryOp::Multiply,
                        Box::new(Expression::new(ExprKind::Constant(Const::Int(5)))),
                        Box::new(Expression::new(ExprKind::Constant(Const::Int(4)))),
                    ))),
                    Box::new(Expression::new(ExprKind::Binary(
                        BinaryOp::BitwiseAnd,
                        Box::new(Expression::new(ExprKind::Binary(
                            BinaryOp::Subtract,
                            Box::new(Expression::new(ExprKind::Constant(Const::Int(4)))),
                            Box::new(Expression::new(ExprKind::Constant(Const::Int(5)))),
                        ))),
                        Box::new(Expression::new(ExprKind::Constant(Const::Int(6)))),
                    ))),
                )),
            ))])),
            None,
            CType::Int,
        ))]);

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
        let expected = AST::Program(vec![Declaration::FunDecl(FunctionDeclaration::new(
            "main".into(),
            vec![],
            Some(crate::ast::Block(vec![BlockItem::Stmt(Statement::Return(
                Expression::new(ExprKind::Binary(
                    BinaryOp::ShiftLeft,
                    Box::new(Expression::new(ExprKind::Binary(
                        BinaryOp::Multiply,
                        Box::new(Expression::new(ExprKind::Constant(Const::Int(5)))),
                        Box::new(Expression::new(ExprKind::Constant(Const::Int(4)))),
                    ))),
                    Box::new(Expression::new(ExprKind::Constant(Const::Int(2)))),
                )),
            ))])),
            None,
            CType::Int,
        ))]);

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
        let expected = AST::Program(vec![Declaration::FunDecl(FunctionDeclaration::new(
            "main".into(),
            vec![],
            Some(crate::ast::Block(vec![BlockItem::Stmt(Statement::Return(
                Expression::new(ExprKind::Binary(
                    BinaryOp::ShiftLeft,
                    Box::new(Expression::new(ExprKind::Constant(Const::Int(5)))),
                    Box::new(Expression::new(ExprKind::Binary(
                        BinaryOp::Add,
                        Box::new(Expression::new(ExprKind::Constant(Const::Int(1)))),
                        Box::new(Expression::new(ExprKind::Constant(Const::Int(2)))),
                    ))),
                )),
            ))])),
            None,
            CType::Int,
        ))]);

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
        let expected = AST::Program(vec![Declaration::FunDecl(FunctionDeclaration::new(
            "main".into(),
            vec![],
            Some(crate::ast::Block(vec![BlockItem::Stmt(Statement::Return(
                Expression::new(ExprKind::Binary(
                    BinaryOp::NotEqual,
                    Box::new(Expression::new(ExprKind::Binary(
                        BinaryOp::LessOrEqual,
                        Box::new(Expression::new(ExprKind::Constant(Const::Int(2)))),
                        Box::new(Expression::new(ExprKind::Binary(
                            BinaryOp::Subtract,
                            Box::new(Expression::new(ExprKind::Constant(Const::Int(3)))),
                            Box::new(Expression::new(ExprKind::Constant(Const::Int(2)))),
                        ))),
                    ))),
                    Box::new(Expression::new(ExprKind::Binary(
                        BinaryOp::Add,
                        Box::new(Expression::new(ExprKind::Constant(Const::Int(4)))),
                        Box::new(Expression::new(ExprKind::Constant(Const::Int(5)))),
                    ))),
                )),
            ))])),
            None,
            CType::Int,
        ))]);

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
        let expected = AST::Program(vec![Declaration::FunDecl(FunctionDeclaration::new(
            "main".into(),
            vec![],
            Some(crate::ast::Block(vec![
                BlockItem::Decl(Declaration::VarDecl(VariableDeclaration {
                    name: "a".into(),
                    init: Some(Expression::new(ExprKind::Constant(Const::Int(1)))),
                    storage_class: None,
                    vtype: CType::Int,
                })),
                BlockItem::Stmt(Statement::Return(Expression::new(ExprKind::Var(
                    "a".into(),
                )))),
            ])),
            None,
            CType::Int,
        ))]);

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
        let expected = AST::Program(vec![Declaration::FunDecl(FunctionDeclaration::new(
            "main".into(),
            vec![],
            Some(crate::ast::Block(vec![
                BlockItem::Decl(Declaration::VarDecl(VariableDeclaration {
                    name: "a".into(),
                    init: None,
                    storage_class: None,
                    vtype: CType::Int,
                })),
                BlockItem::Stmt(Statement::Return(Expression::new(ExprKind::Assignment(
                    Box::new(Expression::new(ExprKind::Var("a".into()))),
                    Box::new(Expression::new(ExprKind::Assignment(
                        Box::new(Expression::new(ExprKind::Var("b".into()))),
                        Box::new(Expression::new(ExprKind::Var("c".into()))),
                    ))),
                )))),
            ])),
            None,
            CType::Int,
        ))]);

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
        let expected = AST::Program(vec![Declaration::FunDecl(FunctionDeclaration::new(
            "main".into(),
            vec![],
            Some(crate::ast::Block(vec![
                BlockItem::Stmt(Statement::Expr(Expression::new(ExprKind::Binary(
                    BinaryOp::Add,
                    Box::new(Expression::new(ExprKind::Constant(Const::Int(2)))),
                    Box::new(Expression::new(ExprKind::Constant(Const::Int(2)))),
                )))),
                BlockItem::Stmt(Statement::Return(Expression::new(ExprKind::Constant(
                    Const::Int(0),
                )))),
            ])),
            None,
            CType::Int,
        ))]);

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
        let expected = AST::Program(vec![Declaration::FunDecl(FunctionDeclaration::new(
            "main".into(),
            vec![],
            Some(crate::ast::Block(vec![
                BlockItem::Decl(Declaration::VarDecl(VariableDeclaration {
                    name: "a".into(),
                    init: Some(Expression::new(ExprKind::Constant(Const::Int(1)))),
                    storage_class: None,
                    vtype: CType::Int,
                })),
                BlockItem::Stmt(Statement::Expr(Expression::new(ExprKind::Binary(
                    BinaryOp::AddAssign,
                    Box::new(Expression::new(ExprKind::Var("a".into()))),
                    Box::new(Expression::new(ExprKind::Constant(Const::Int(2)))),
                )))),
                BlockItem::Stmt(Statement::Expr(Expression::new(ExprKind::Binary(
                    BinaryOp::MinusAssign,
                    Box::new(Expression::new(ExprKind::Var("a".into()))),
                    Box::new(Expression::new(ExprKind::Constant(Const::Int(2)))),
                )))),
                BlockItem::Stmt(Statement::Expr(Expression::new(ExprKind::Binary(
                    BinaryOp::MultiplyAssign,
                    Box::new(Expression::new(ExprKind::Var("a".into()))),
                    Box::new(Expression::new(ExprKind::Constant(Const::Int(2)))),
                )))),
                BlockItem::Stmt(Statement::Expr(Expression::new(ExprKind::Binary(
                    BinaryOp::DivideAssign,
                    Box::new(Expression::new(ExprKind::Var("a".into()))),
                    Box::new(Expression::new(ExprKind::Constant(Const::Int(2)))),
                )))),
                BlockItem::Stmt(Statement::Expr(Expression::new(ExprKind::Binary(
                    BinaryOp::RemainderAssign,
                    Box::new(Expression::new(ExprKind::Var("a".into()))),
                    Box::new(Expression::new(ExprKind::Constant(Const::Int(2)))),
                )))),
                BlockItem::Stmt(Statement::Return(Expression::new(ExprKind::Var(
                    "a".into(),
                )))),
            ])),
            None,
            CType::Int,
        ))]);

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
        let expected = AST::Program(vec![Declaration::FunDecl(FunctionDeclaration::new(
            "main".into(),
            vec![],
            Some(crate::ast::Block(vec![
                BlockItem::Decl(Declaration::VarDecl(VariableDeclaration {
                    name: "a".into(),
                    init: Some(Expression::new(ExprKind::Constant(Const::Int(1)))),
                    storage_class: None,
                    vtype: CType::Int,
                })),
                BlockItem::Decl(Declaration::VarDecl(VariableDeclaration {
                    name: "b".into(),
                    init: Some(Expression::new(ExprKind::Constant(Const::Int(0)))),
                    storage_class: None,
                    vtype: CType::Int,
                })),
                BlockItem::Decl(Declaration::VarDecl(VariableDeclaration {
                    name: "c".into(),
                    init: Some(Expression::new(ExprKind::Constant(Const::Int(2)))),
                    storage_class: None,
                    vtype: CType::Int,
                })),
                BlockItem::Stmt(Statement::Expr(Expression::new(ExprKind::Binary(
                    BinaryOp::AddAssign,
                    Box::new(Expression::new(ExprKind::Var("a".into()))),
                    Box::new(Expression::new(ExprKind::Binary(
                        BinaryOp::MinusAssign,
                        Box::new(Expression::new(ExprKind::Var("b".into()))),
                        Box::new(Expression::new(ExprKind::Assignment(
                            Box::new(Expression::new(ExprKind::Var("c".into()))),
                            Box::new(Expression::new(ExprKind::Constant(Const::Int(5)))),
                        ))),
                    ))),
                )))),
                BlockItem::Stmt(Statement::Return(Expression::new(ExprKind::Var(
                    "a".into(),
                )))),
            ])),
            None,
            CType::Int,
        ))]);

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
        let expected = AST::Program(vec![Declaration::FunDecl(FunctionDeclaration::new(
            "main".into(),
            vec![],
            Some(crate::ast::Block(vec![
                BlockItem::Decl(Declaration::VarDecl(VariableDeclaration {
                    name: "a".into(),
                    init: Some(Expression::new(ExprKind::Constant(Const::Int(10)))),
                    storage_class: None,
                    vtype: CType::Int,
                })),
                BlockItem::Stmt(Statement::Expr(Expression::new(ExprKind::Binary(
                    BinaryOp::MinusAssign,
                    Box::new(Expression::new(ExprKind::Binary(
                        BinaryOp::AddAssign,
                        Box::new(Expression::new(ExprKind::Var("a".into()))),
                        Box::new(Expression::new(ExprKind::Constant(Const::Int(1)))),
                    ))),
                    Box::new(Expression::new(ExprKind::Constant(Const::Int(2)))),
                )))),
            ])),
            None,
            CType::Int,
        ))]);

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

        let expected = AST::Program(vec![Declaration::FunDecl(FunctionDeclaration::new(
            "main".into(),
            vec![],
            Some(crate::ast::Block(vec![BlockItem::Stmt(Statement::If {
                condition: Expression::new(ExprKind::Var("a".into())),
                then: Box::new(Statement::If {
                    condition: Expression::new(ExprKind::Binary(
                        BinaryOp::GreaterThan,
                        Box::new(Expression::new(ExprKind::Var("a".into()))),
                        Box::new(Expression::new(ExprKind::Constant(Const::Int(10)))),
                    )),
                    then: Box::new(Statement::Return(Expression::new(ExprKind::Var(
                        "a".into(),
                    )))),
                    else_: Some(Box::new(Statement::Return(Expression::new(
                        ExprKind::Binary(
                            BinaryOp::Subtract,
                            Box::new(Expression::new(ExprKind::Constant(Const::Int(10)))),
                            Box::new(Expression::new(ExprKind::Var("a".into()))),
                        ),
                    )))),
                }),
                else_: None,
            })])),
            None,
            CType::Int,
        ))]);

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

        let expected = AST::Program(vec![Declaration::FunDecl(FunctionDeclaration::new(
            "main".into(),
            vec![],
            Some(crate::ast::Block(vec![
                // first example:  we nest inner expression
                BlockItem::Stmt(Statement::Expr(Expression::new(ExprKind::Conditional {
                    condition: Box::new(Expression::new(ExprKind::Var("a".into()))),
                    then: Box::new(Expression::new(ExprKind::Conditional {
                        condition: Box::new(Expression::new(ExprKind::Var("b".into()))),
                        then: Box::new(Expression::new(ExprKind::Constant(Const::Int(1)))),
                        else_: Box::new(Expression::new(ExprKind::Constant(Const::Int(2)))),
                    })),
                    else_: Box::new(Expression::new(ExprKind::Constant(Const::Int(3)))),
                }))),
                // second example: else statement is nested parse
                BlockItem::Stmt(Statement::Expr(Expression::new(ExprKind::Conditional {
                    condition: Box::new(Expression::new(ExprKind::Var("a".into()))),
                    then: Box::new(Expression::new(ExprKind::Constant(Const::Int(1)))),
                    else_: Box::new(Expression::new(ExprKind::Conditional {
                        condition: Box::new(Expression::new(ExprKind::Var("b".into()))),
                        then: Box::new(Expression::new(ExprKind::Constant(Const::Int(2)))),
                        else_: Box::new(Expression::new(ExprKind::Constant(Const::Int(3)))),
                    })),
                }))),
                // third example: the conditional is a short-circuit expr
                BlockItem::Stmt(Statement::Expr(Expression::new(ExprKind::Conditional {
                    condition: Box::new(Expression::new(ExprKind::Binary(
                        BinaryOp::BinOr,
                        Box::new(Expression::new(ExprKind::Var("a".into()))),
                        Box::new(Expression::new(ExprKind::Var("b".into()))),
                    ))),
                    then: Box::new(Expression::new(ExprKind::Constant(Const::Int(2)))),
                    else_: Box::new(Expression::new(ExprKind::Constant(Const::Int(3)))),
                }))),
                // fourth example: then should be an assignment
                BlockItem::Stmt(Statement::Expr(Expression::new(ExprKind::Conditional {
                    condition: Box::new(Expression::new(ExprKind::Var("x".into()))),
                    then: Box::new(Expression::new(ExprKind::Assignment(
                        Box::new(Expression::new(ExprKind::Var("x".into()))),
                        Box::new(Expression::new(ExprKind::Constant(Const::Int(1)))),
                    ))),
                    else_: Box::new(Expression::new(ExprKind::Constant(Const::Int(2)))),
                }))),
            ])),
            None,
            CType::Int,
        ))]);

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

        let expected = AST::Program(vec![Declaration::FunDecl(FunctionDeclaration::new(
            "main".into(),
            vec![],
            Some(crate::ast::Block(vec![
                BlockItem::Stmt(Statement::Goto("foo".into())),
                BlockItem::Stmt(Statement::Labelled {
                    label: "foo".into(),
                    statement: Box::new(Statement::Return(Expression::new(ExprKind::Binary(
                        BinaryOp::Add,
                        Box::new(Expression::new(ExprKind::Constant(Const::Int(1)))),
                        Box::new(Expression::new(ExprKind::Constant(Const::Int(2)))),
                    )))),
                }),
            ])),
            None,
            CType::Int,
        ))]);

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

        let expected = AST::Program(vec![Declaration::FunDecl(FunctionDeclaration::new(
            "main".into(),
            vec![],
            Some(crate::ast::Block(vec![
                BlockItem::Stmt(Statement::If {
                    condition: Expression::new(ExprKind::Constant(Const::Int(1))),
                    then: Box::new(Statement::Compound(crate::ast::Block(vec![
                        BlockItem::Decl(Declaration::VarDecl(VariableDeclaration {
                            name: "x".into(),
                            init: Some(Expression::new(ExprKind::Constant(Const::Int(1)))),
                            storage_class: None,
                            vtype: CType::Int,
                        })),
                        BlockItem::Stmt(Statement::Return(Expression::new(ExprKind::Var(
                            "x".into(),
                        )))),
                    ]))),
                    else_: None,
                }),
                BlockItem::Stmt(Statement::Compound(crate::ast::Block(vec![
                    BlockItem::Decl(Declaration::VarDecl(VariableDeclaration {
                        name: "b".into(),
                        init: Some(Expression::new(ExprKind::Constant(Const::Int(2)))),
                        storage_class: None,
                        vtype: CType::Int,
                    })),
                    BlockItem::Stmt(Statement::Expr(Expression::new(ExprKind::Binary(
                        BinaryOp::Add,
                        Box::new(Expression::new(ExprKind::Var("b".into()))),
                        Box::new(Expression::new(ExprKind::Constant(Const::Int(1)))),
                    )))),
                ]))),
            ])),
            None,
            CType::Int,
        ))]);

        assert_eq!(ast.unwrap(), expected);
    }

    #[test]
    fn test_loop_parsing() {
        let src = r#"
        int main(void) {
            int a = 1;
            for (int b = 1; b < 10; b = b + 1) {
                continue; 
            }
            do {
                continue;
            } while (a < 0);
            while (a > 0) 
                break;
            
        }
        "#;
        let lexer = crate::lexer::Lexer::lex(src).unwrap();
        let tokens = lexer.as_syntactic_tokens();
        let parse = Parser::new(&tokens);
        let ast = parse.into_ast();
        assert!(ast.is_ok());

        let expected = AST::Program(vec![Declaration::FunDecl(FunctionDeclaration::new(
            "main".into(),
            vec![],
            Some(crate::ast::Block(vec![
                BlockItem::Decl(Declaration::VarDecl(VariableDeclaration {
                    name: "a".into(),
                    init: Some(Expression::new(ExprKind::Constant(Const::Int(1)))),
                    storage_class: None,
                    vtype: CType::Int,
                })),
                BlockItem::Stmt(Statement::For {
                    init: ForInit::InitDecl(VariableDeclaration {
                        name: "b".into(),
                        init: Some(Expression::new(ExprKind::Constant(Const::Int(1)))),
                        storage_class: None,
                        vtype: CType::Int,
                    }),
                    condition: Some(Expression::new(ExprKind::Binary(
                        BinaryOp::LessThan,
                        Box::new(Expression::new(ExprKind::Var("b".into()))),
                        Box::new(Expression::new(ExprKind::Constant(Const::Int(10)))),
                    ))),
                    post: Some(Expression::new(ExprKind::Assignment(
                        Box::new(Expression::new(ExprKind::Var("b".into()))),
                        Box::new(Expression::new(ExprKind::Binary(
                            BinaryOp::Add,
                            Box::new(Expression::new(ExprKind::Var("b".into()))),
                            Box::new(Expression::new(ExprKind::Constant(Const::Int(1)))),
                        ))),
                    ))),
                    body: Box::new(Statement::Compound(crate::ast::Block(vec![
                        BlockItem::Stmt(Statement::Continue("".into())),
                    ]))),
                    label: "".into(),
                }),
                BlockItem::Stmt(Statement::DoWhile {
                    body: Box::new(Statement::Compound(crate::ast::Block(vec![
                        BlockItem::Stmt(Statement::Continue("".into())),
                    ]))),
                    condition: Expression::new(ExprKind::Binary(
                        BinaryOp::LessThan,
                        Box::new(Expression::new(ExprKind::Var("a".into()))),
                        Box::new(Expression::new(ExprKind::Constant(Const::Int(0)))),
                    )),
                    label: "".into(),
                }),
                BlockItem::Stmt(Statement::While {
                    condition: Expression::new(ExprKind::Binary(
                        BinaryOp::GreaterThan,
                        Box::new(Expression::new(ExprKind::Var("a".into()))),
                        Box::new(Expression::new(ExprKind::Constant(Const::Int(0)))),
                    )),
                    body: Box::new(Statement::Break("".into())),
                    label: "".into(),
                }),
            ])),
            None,
            CType::Int,
        ))]);

        assert_eq!(ast.unwrap(), expected);
    }

    #[test]
    fn test_functions_and_variables() {
        let src = r#"
            int static bar(int a);
            extern int foo(int x, int y) { 
                extern int y;
                return x + y;
            }
            int main(void) {
                return foo(1, 2) + 3;
            }
            static int a = 3;
        "#;
        let lexer = crate::lexer::Lexer::lex(src).unwrap();
        let tokens = lexer.as_syntactic_tokens();
        let parse = Parser::new(&tokens);
        let ast = parse.into_ast();
        assert!(ast.is_ok());

        let expected = AST::Program(vec![
            Declaration::FunDecl(FunctionDeclaration::new(
                "bar".into(),
                vec![("a".into(), CType::Int)],
                None,
                Some(StorageClass::Static),
                CType::Int,
            )),
            Declaration::FunDecl(FunctionDeclaration::new(
                "foo".into(),
                vec![("x".into(), CType::Int), ("y".into(), CType::Int)],
                Some(crate::ast::Block(vec![
                    BlockItem::Decl(Declaration::VarDecl(VariableDeclaration {
                        name: "y".into(),
                        init: None,
                        storage_class: Some(StorageClass::Extern),
                        vtype: CType::Int,
                    })),
                    BlockItem::Stmt(Statement::Return(Expression::new(ExprKind::Binary(
                        BinaryOp::Add,
                        Box::new(Expression::new(ExprKind::Var("x".into()))),
                        Box::new(Expression::new(ExprKind::Var("y".into()))),
                    )))),
                ])),
                Some(StorageClass::Extern),
                CType::Int,
            )),
            Declaration::FunDecl(FunctionDeclaration::new(
                "main".into(),
                vec![],
                Some(crate::ast::Block(vec![BlockItem::Stmt(Statement::Return(
                    Expression::new(ExprKind::Binary(
                        BinaryOp::Add,
                        Box::new(Expression::new(ExprKind::FunctionCall {
                            name: "foo".into(),
                            args: vec![
                                Expression::new(ExprKind::Constant(Const::Int(1))),
                                Expression::new(ExprKind::Constant(Const::Int(2))),
                            ],
                        })),
                        Box::new(Expression::new(ExprKind::Constant(Const::Int(3)))),
                    )),
                ))])),
                None,
                CType::Int,
            )),
            Declaration::VarDecl(VariableDeclaration {
                name: "a".into(),
                init: Some(Expression::new(ExprKind::Constant(Const::Int(3)))),
                storage_class: Some(StorageClass::Static),
                vtype: CType::Int,
            }),
        ]);

        assert_eq!(ast.unwrap(), expected);
    }

    #[test]
    fn test_nested_function_declarations() {
        let src = r#"
            int main(void) {
                int foo(int x, int y);
                return foo(1, 2) + 3;
            }
        "#;
        let lexer = crate::lexer::Lexer::lex(src).unwrap();
        let tokens = lexer.as_syntactic_tokens();
        let parse = Parser::new(&tokens);
        let ast = parse.into_ast();
        assert!(ast.is_ok());

        let expected = AST::Program(vec![Declaration::FunDecl(FunctionDeclaration::new(
            "main".into(),
            vec![],
            Some(crate::ast::Block(vec![
                BlockItem::Decl(Declaration::FunDecl(FunctionDeclaration::new(
                    "foo".into(),
                    vec![("x".into(), CType::Int), ("y".into(), CType::Int)],
                    None,
                    None,
                    CType::Int,
                ))),
                BlockItem::Stmt(Statement::Return(Expression::new(ExprKind::Binary(
                    BinaryOp::Add,
                    Box::new(Expression::new(ExprKind::FunctionCall {
                        name: "foo".into(),
                        args: vec![
                            Expression::new(ExprKind::Constant(Const::Int(1))),
                            Expression::new(ExprKind::Constant(Const::Int(2))),
                        ],
                    })),
                    Box::new(Expression::new(ExprKind::Constant(Const::Int(3)))),
                )))),
            ])),
            None,
            CType::Int,
        ))]);

        assert_eq!(ast.unwrap(), expected);
    }

    #[test]
    fn parses_long_variable() {
        // long x;  inside a function
        let src = r#"
            int main(void) {
                long x;
                return 0;
            }
        "#;
        let lexer = crate::lexer::Lexer::lex(src).unwrap();
        let tokens = lexer.as_syntactic_tokens();
        let ast = Parser::new(&tokens).into_ast();
        assert!(ast.is_ok());
        let ast = ast.unwrap();
        let AST::Program(decls) = &ast;
        let Declaration::FunDecl(main) = &decls[0] else {
            panic!()
        };
        let crate::ast::Block(items) = main.block.as_ref().unwrap();
        let BlockItem::Decl(Declaration::VarDecl(vd)) = &items[0] else {
            panic!()
        };
        assert_eq!(vd.vtype, CType::Long);
        assert_eq!(vd.name, "x");
    }

    #[test]
    fn parses_long_int_variable() {
        // long int and int long both mean CType::Long
        for src in [
            "int main(void) { long int x; return 0; }",
            "int main(void) { int long x; return 0; }",
        ] {
            let lexer = crate::lexer::Lexer::lex(src).unwrap();
            let tokens = lexer.as_syntactic_tokens();
            let ast = Parser::new(&tokens).into_ast();
            assert!(ast.is_ok(), "failed for: {src}");
            let AST::Program(decls) = ast.unwrap();
            let Declaration::FunDecl(main) = &decls[0] else {
                panic!()
            };
            let crate::ast::Block(items) = main.block.as_ref().unwrap();
            let BlockItem::Decl(Declaration::VarDecl(vd)) = &items[0] else {
                panic!()
            };
            assert_eq!(vd.vtype, CType::Long, "wrong type for: {src}");
        }
    }

    #[test]
    fn parses_long_function() {
        // long foo(long a) { return a; }
        let src = "long foo(long a) { return a; }";
        let lexer = crate::lexer::Lexer::lex(src).unwrap();
        let tokens = lexer.as_syntactic_tokens();
        let ast = Parser::new(&tokens).into_ast();
        assert!(ast.is_ok());
        let AST::Program(decls) = ast.unwrap();
        let Declaration::FunDecl(fun) = &decls[0] else {
            panic!()
        };
        assert_eq!(
            fun.ftype,
            CType::FunType {
                params: vec![CType::Long],
                ret: Box::new(CType::Long),
            }
        );
    }

    #[test]
    fn parses_unsigned_types() {
        // all orderings of unsigned int / unsigned long should parse correctly
        for (src, expected_type) in [
            ("int main(void) { unsigned int x; return 0; }", CType::UInt),
            ("int main(void) { unsigned x; return 0; }", CType::UInt),
            ("int main(void) { int unsigned x; return 0; }", CType::UInt),
            (
                "int main(void) { unsigned long x; return 0; }",
                CType::ULong,
            ),
            (
                "int main(void) { long unsigned x; return 0; }",
                CType::ULong,
            ),
            (
                "int main(void) { unsigned long int x; return 0; }",
                CType::ULong,
            ),
            (
                "int main(void) { signed long int x; return 0; }",
                CType::Long,
            ),
            ("int main(void) { signed int x; return 0; }", CType::Int),
            ("int main(void) { signed x; return 0; }", CType::Int),
        ] {
            let lexer = crate::lexer::Lexer::lex(src).unwrap();
            let tokens = lexer.as_syntactic_tokens();
            let ast = Parser::new(&tokens).into_ast();
            assert!(ast.is_ok(), "failed for: {src}");
            let AST::Program(decls) = ast.unwrap();
            let Declaration::FunDecl(main) = &decls[0] else {
                panic!()
            };
            let crate::ast::Block(items) = main.block.as_ref().unwrap();
            let BlockItem::Decl(Declaration::VarDecl(vd)) = &items[0] else {
                panic!()
            };
            assert_eq!(vd.vtype, expected_type, "wrong type for: {src}");
        }
    }

    #[test]
    fn invalid_unsigned_type_specifiers() {
        for src in [
            // signed and unsigned together
            "int main(void) { signed unsigned x; return 0; }",
            // duplicates
            "int main(void) { unsigned unsigned x; return 0; }",
            "int main(void) { signed signed x; return 0; }",
        ] {
            let lexer = crate::lexer::Lexer::lex(src).unwrap();
            let tokens = lexer.as_syntactic_tokens();
            let ast = Parser::new(&tokens).into_ast();
            assert_eq!(
                ast,
                Err(ParserError::InvalidTypeSpecifier),
                "should fail for: {src}"
            );
        }
    }

    #[test]
    fn invalid_storage_class() {
        for src in [
            // storage class in cast
            "int main(void) { return (static int) 10; }",
            // storage class in function parameter type
            "int foo(static int x) { return x; }",
        ] {
            let lexer = crate::lexer::Lexer::lex(src).unwrap();
            let tokens = lexer.as_syntactic_tokens();
            let ast = Parser::new(&tokens).into_ast();
            assert_eq!(
                ast,
                Err(ParserError::InvalidStorageClass),
                "should fail for: {src}"
            );
        }
    }
    #[test]
    fn parses_unsigned_constants() {
        for (src, expected_const) in [
            ("int main(void) { return 1u; }", Const::UInt(1)),
            ("int main(void) { return 1ul; }", Const::ULong(1)),
        ] {
            let lexer = crate::lexer::Lexer::lex(src).unwrap();
            let tokens = lexer.as_syntactic_tokens();
            let ast = Parser::new(&tokens).into_ast();
            assert!(ast.is_ok(), "failed for: {src}");
            let AST::Program(decls) = ast.unwrap();
            let Declaration::FunDecl(main) = &decls[0] else {
                panic!()
            };
            let crate::ast::Block(items) = main.block.as_ref().unwrap();
            let BlockItem::Stmt(crate::ast::Statement::Return(expr)) = &items[0] else {
                panic!()
            };
            assert_eq!(
                *expr.kind,
                crate::ast::ExprKind::Constant(expected_const),
                "wrong const for: {src}"
            );
        }
    }

    #[test]
    fn invalid_storage_and_type_specifiers() {
        let src = "main(void) { return 0 ; }";
        let lexer = crate::lexer::Lexer::lex(src).unwrap();
        let tokens = lexer.as_syntactic_tokens();
        let parse = Parser::new(&tokens);
        let ast = parse.into_ast();
        assert_eq!(ast, Err(ParserError::InvalidTypeSpecifier));

        let src = r#"
            int main(void) { return 0 ; }
            static extern int a = 2;
        "#;
        let lexer = crate::lexer::Lexer::lex(src).unwrap();
        let tokens = lexer.as_syntactic_tokens();
        let parse = Parser::new(&tokens);
        let ast = parse.into_ast();
        assert_eq!(ast, Err(ParserError::InvalidStorageClass));
    }

    #[test]
    fn parse_constant_should_promote_signed_int_to_signed_long() {
        let t = Token::Constant(i32::MAX as usize + 1);
        let c = Parser::parse_constant(t);
        assert!(c.is_ok());
        let const_ = c.unwrap();
        assert_eq!(const_, Const::Long(i32::MAX as i64 + 1));
    }

    #[test]
    fn parse_constant_should_promote_unsigned_int_to_unsigned_long() {
        let t = Token::UnsignedIntConstant(u32::MAX as usize + 1);
        let c = Parser::parse_constant(t);
        assert!(c.is_ok());
        let const_ = c.unwrap();
        assert_eq!(const_, Const::ULong(u32::MAX as u64 + 1));
    }

    #[test]
    fn invalid_double_type_specifier_combinations() {
        // double cannot be combined with any other type specifier
        for src in [
            "int main(void) { double int x; return 0; }",
            "int main(void) { double long x; return 0; }",
            "int main(void) { double signed x; return 0; }",
            "int main(void) { double unsigned x; return 0; }",
            "int main(void) { double double x; return 0; }",
        ] {
            let lexer = crate::lexer::Lexer::lex(src).unwrap();
            let tokens = lexer.as_syntactic_tokens();
            let ast = Parser::new(&tokens).into_ast();
            assert_eq!(
                ast,
                Err(ParserError::InvalidTypeSpecifier),
                "should fail for: {src}"
            );
        }
    }

    #[test]
    fn parses_function_with_double_return_and_param_type() {
        // double foo(double d) { return d; }
        let src = "double foo(double d) { return d; }";
        let lexer = crate::lexer::Lexer::lex(src).unwrap();
        let tokens = lexer.as_syntactic_tokens();
        let ast = Parser::new(&tokens).into_ast();
        assert!(ast.is_ok(), "failed to parse: {ast:?}");
        let AST::Program(decls) = ast.unwrap();
        let Declaration::FunDecl(f) = &decls[0] else {
            panic!("expected FunDecl");
        };
        let CType::FunType { params, ret } = &f.ftype else {
            panic!("expected FunType, got {:?}", f.ftype);
        };
        assert_eq!(**ret, CType::Double, "wrong return type");
        assert_eq!(params.len(), 1);
        assert_eq!(params[0], CType::Double, "wrong param type");
    }

    #[test]
    fn parses_cast_to_double() {
        // int main(void) { return (double) 5; }
        let src = "int main(void) { return (double) 5; }";
        let lexer = crate::lexer::Lexer::lex(src).unwrap();
        let tokens = lexer.as_syntactic_tokens();
        let ast = Parser::new(&tokens).into_ast();
        assert!(ast.is_ok(), "failed to parse: {ast:?}");
        let AST::Program(decls) = ast.unwrap();
        let Declaration::FunDecl(main) = &decls[0] else {
            panic!("expected FunDecl");
        };
        let crate::ast::Block(items) = main.block.as_ref().unwrap();
        let BlockItem::Stmt(crate::ast::Statement::Return(expr)) = &items[0] else {
            panic!("expected return stmt");
        };
        let ExprKind::Cast(target_type, _) = expr.kind.as_ref() else {
            panic!("expected Cast expr, got {:?}", expr.kind);
        };
        assert_eq!(*target_type, CType::Double);
    }

    #[test]
    fn parses_cast_from_double() {
        // int main(void) { double d = 1.5; return (int) d; }
        let src = "int main(void) { double d = 1.5; return (int) d; }";
        let lexer = crate::lexer::Lexer::lex(src).unwrap();
        let tokens = lexer.as_syntactic_tokens();
        let ast = Parser::new(&tokens).into_ast();
        assert!(ast.is_ok(), "failed to parse: {ast:?}");
        let AST::Program(decls) = ast.unwrap();
        let Declaration::FunDecl(main) = &decls[0] else {
            panic!("expected FunDecl");
        };
        let crate::ast::Block(items) = main.block.as_ref().unwrap();
        let BlockItem::Stmt(crate::ast::Statement::Return(expr)) = &items[1] else {
            panic!("expected return stmt as second block item");
        };
        let ExprKind::Cast(target_type, _) = expr.kind.as_ref() else {
            panic!("expected Cast expr, got {:?}", expr.kind);
        };
        assert_eq!(*target_type, CType::Int);
    }

    #[test]
    fn parses_double_variable_declaration_with_float_initializer() {
        // double x = 1.5; — check the var decl has Double type and a Double constant initializer
        let src = "int main(void) { double x = 1.5; return 0; }";
        let lexer = crate::lexer::Lexer::lex(src).unwrap();
        let tokens = lexer.as_syntactic_tokens();
        let ast = Parser::new(&tokens).into_ast();
        assert!(ast.is_ok(), "failed to parse: {ast:?}");
        let AST::Program(decls) = ast.unwrap();
        let Declaration::FunDecl(main) = &decls[0] else {
            panic!("expected FunDecl");
        };
        let crate::ast::Block(items) = main.block.as_ref().unwrap();
        let BlockItem::Decl(Declaration::VarDecl(var)) = &items[0] else {
            panic!("expected VarDecl as first block item");
        };
        assert_eq!(var.vtype, CType::Double, "wrong var type");
        let Some(init_expr) = &var.init else {
            panic!("expected initializer");
        };
        assert_eq!(
            *init_expr.kind,
            ExprKind::Constant(Const::Double(1.5)),
            "wrong initializer value"
        );
    }

    #[test]
    fn parse_constant_works_with_doubles() {
        let toks = [
            ("1.", 1.0),
            ("1.0", 1.0),
            ("0.5", 0.5),
            (".5", 0.5),
            ("100e1", 1000.0),
            ("1E+1", 10.0),
            ("2e-1", 0.2),
        ];
        for (t, exp) in toks {
            let tok = Token::FloatingPointConstant(t);
            let c = Parser::parse_constant(tok);
            assert!(c.is_ok());
            let const_ = c.unwrap();
            let Const::Double(s) = const_ else {
                panic!();
            };
            assert_eq!(s, exp);
        }
    }
}
