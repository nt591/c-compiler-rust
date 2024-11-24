use crate::parser::BlockItem;
use crate::parser::Declaration;
use crate::parser::Expression;
use crate::parser::Statement;
use crate::parser::AST;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum SemanticAnalysisError {
    #[error("Got duplicate variable declaration {0}")]
    DuplicateDecl(String),
    #[error("Expected top-level program node")]
    UnexpectedNonProgramNode,
    #[error("Expected function node inside program")]
    UnexpectedNonFunctionNode,
    #[error("Found undeclared variable in initializer: {0}")]
    UndeclaredVariableInInitializer(String),
    #[error("Found non-var node on lefthand side of assignment")]
    InvalidLhsAssignmentNode,
}

// maps variable names to a unique identifier
#[derive(Debug)]
struct Resolver {
    renamed_variables: HashMap<String, String>,
    var_counter: usize,
}

impl Resolver {
    fn new() -> Self {
        Self {
            renamed_variables: HashMap::new(),
            var_counter: 0,
        }
    }

    fn seen(&self, name: &str) -> bool {
        self.renamed_variables.contains_key(name)
    }

    fn resolve_variable(&mut self, name: &str) -> String {
        self.renamed_variables
            .entry(name.into())
            .or_insert_with(|| {
                let count = self.var_counter;
                self.var_counter += 1;
                format!("{name}.{count}.decl")
            })
            .clone()
    }
}

pub fn resolve<'a>(ast: &mut AST<'a>) -> Result<(), SemanticAnalysisError> {
    let mut resolver = Resolver::new();
    resolve_ast(ast, &mut resolver)
}

fn resolve_ast<'a>(
    ast: &mut AST<'a>,
    resolver: &mut Resolver,
) -> Result<(), SemanticAnalysisError> {
    let AST::Program(function) = ast else {
        return Err(SemanticAnalysisError::UnexpectedNonProgramNode);
    };

    let AST::Function { body, name: _ } = function.as_mut() else {
        return Err(SemanticAnalysisError::UnexpectedNonFunctionNode);
    };

    for body_item in body.iter_mut() {
        match body_item {
            BlockItem::Decl(declaration) => resolve_decl(declaration, resolver)?,
            BlockItem::Stmt(statement) => resolve_statement(statement, resolver)?,
        }
    }

    Ok(())
}

fn resolve_decl(
    decl: &mut Declaration,
    resolver: &mut Resolver,
) -> Result<(), SemanticAnalysisError> {
    if resolver.seen(&decl.name) {
        return Err(SemanticAnalysisError::DuplicateDecl(decl.name.clone()));
    };
    let renamed_var = resolver.resolve_variable(&decl.name);
    decl.name = renamed_var;
    if let Some(init) = &mut decl.init {
        resolve_expr(init, resolver)?;
    };
    Ok(())
}

fn resolve_statement(
    statement: &mut Statement,
    resolver: &mut Resolver,
) -> Result<(), SemanticAnalysisError> {
    match statement {
        Statement::Null => (),
        Statement::Expr(expr) => resolve_expr(&mut *expr, resolver)?,
        Statement::Return(expr) => resolve_expr(&mut *expr, resolver)?,
    };
    Ok(())
}

fn resolve_expr(
    expr: &mut Expression,
    resolver: &mut Resolver,
) -> Result<(), SemanticAnalysisError> {
    match expr {
        Expression::Constant(_) => (),
        Expression::Var(ident) => {
            if !resolver.seen(&ident) {
                return Err(SemanticAnalysisError::UndeclaredVariableInInitializer(
                    ident.clone(),
                ));
            }

            let renamed = resolver.resolve_variable(&ident);
            *ident = renamed;
        }
        Expression::Unary(_op, expr) => resolve_expr(expr, resolver)?,
        Expression::Binary(_op, lhs, rhs) => {
            resolve_expr(&mut *lhs, resolver)?;
            resolve_expr(&mut *rhs, resolver)?;
        }
        Expression::Assignment(lhs, rhs) => {
            if !matches!(**lhs, Expression::Var(_)) {
                return Err(SemanticAnalysisError::InvalidLhsAssignmentNode);
            }
            resolve_expr(&mut *lhs, resolver)?;
            resolve_expr(&mut *rhs, resolver)?;
        }
    };
    Ok(())
}

mod tests {
    use super::*;
    use crate::parser::BinaryOp;
    use crate::parser::BlockItem;
    use crate::parser::Expression;
    use crate::parser::Statement;
    use crate::parser::AST;

    #[test]
    fn basic_resolve_repeated_variable() {
        let mut before = AST::Program(Box::new(AST::Function {
            name: "main",
            body: vec![
                BlockItem::Decl(Declaration {
                    name: "a".into(),
                    init: Some(Expression::Constant(1)),
                }),
                BlockItem::Decl(Declaration {
                    name: "a".into(),
                    init: Some(Expression::Constant(1)),
                }),
                BlockItem::Stmt(Statement::Return(Expression::Var("a".into()))),
            ],
        }));

        let actual = resolve(&mut before);
        assert!(actual.is_err());
    }

    #[test]
    fn basic_resolve_undeclared_variable() {
        let mut before = AST::Program(Box::new(AST::Function {
            name: "main",
            body: vec![
                BlockItem::Decl(Declaration {
                    name: "a".into(),
                    init: Some(Expression::Var("c".into())),
                }),
                BlockItem::Stmt(Statement::Return(Expression::Var("a".into()))),
            ],
        }));

        let actual = resolve(&mut before);
        assert!(actual.is_err());
    }

    #[test]
    fn basic_resolve_successful() {
        let mut before = AST::Program(Box::new(AST::Function {
            name: "main",
            body: vec![
                BlockItem::Decl(Declaration {
                    name: "a".into(),
                    init: Some(Expression::Constant(1)),
                }),
                BlockItem::Decl(Declaration {
                    name: "b".into(),
                    init: Some(Expression::Var("a".into())),
                }),
                BlockItem::Stmt(Statement::Return(Expression::Var("a".into()))),
            ],
        }));

        let actual = resolve(&mut before);
        assert!(actual.is_ok());
        let expected = AST::Program(Box::new(AST::Function {
            name: "main",
            body: vec![
                BlockItem::Decl(Declaration {
                    name: "a.0.decl".into(),
                    init: Some(Expression::Constant(1)),
                }),
                BlockItem::Decl(Declaration {
                    name: "b.1.decl".into(),
                    init: Some(Expression::Var("a.0.decl".into())),
                }),
                BlockItem::Stmt(Statement::Return(Expression::Var("a.0.decl".into()))),
            ],
        }));

        // mutates taken value
        assert_eq!(before, expected);
    }

    #[test]
    fn complex_resolve_successful() {
        let mut before = AST::Program(Box::new(AST::Function {
            name: "main",
            body: vec![
                BlockItem::Decl(Declaration {
                    name: "a".into(),
                    init: Some(Expression::Constant(1)),
                }),
                BlockItem::Decl(Declaration {
                    name: "b".into(),
                    init: Some(Expression::Var("a".into())),
                }),
                BlockItem::Decl(Declaration {
                    name: "c".into(),
                    init: Some(Expression::Binary(
                        BinaryOp::Add,
                        Box::new(Expression::Var("a".into())),
                        Box::new(Expression::Var("b".into())),
                    )),
                }),
                BlockItem::Stmt(Statement::Return(Expression::Var("c".into()))),
            ],
        }));

        let actual = resolve(&mut before);
        assert!(actual.is_ok());
        let expected = AST::Program(Box::new(AST::Function {
            name: "main",
            body: vec![
                BlockItem::Decl(Declaration {
                    name: "a.0.decl".into(),
                    init: Some(Expression::Constant(1)),
                }),
                BlockItem::Decl(Declaration {
                    name: "b.1.decl".into(),
                    init: Some(Expression::Var("a.0.decl".into())),
                }),
                BlockItem::Decl(Declaration {
                    name: "c.2.decl".into(),
                    init: Some(Expression::Binary(
                        BinaryOp::Add,
                        Box::new(Expression::Var("a.0.decl".into())),
                        Box::new(Expression::Var("b.1.decl".into())),
                    )),
                }),
                BlockItem::Stmt(Statement::Return(Expression::Var("c.2.decl".into()))),
            ],
        }));

        // mutates taken value
        assert_eq!(before, expected);
    }
}
