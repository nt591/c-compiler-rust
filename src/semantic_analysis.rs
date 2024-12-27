use crate::parser::Block;
use crate::parser::BlockItem;
use crate::parser::Declaration;
use crate::parser::Expression;
use crate::parser::Statement;
use crate::parser::AST;
use std::collections::hash_map::Entry;
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
    #[error("Found duplicate label declaration: {0}")]
    DuplicateLabel(String),
    #[error("Undeclared label {0}")]
    UndeclaredLabel(String),
}

// maps variable names to a unique identifier
#[derive(Debug, Clone)]
struct Resolver {
    // stored as <resolved string, declared_in_current_block>
    renamed_variables: HashMap<String, (String, bool)>,
    // stored as (resolved_string, is_declared)
    labels: HashMap<String, (String, bool)>,
    var_counter: usize,
}

impl Resolver {
    fn new() -> Self {
        Self {
            renamed_variables: HashMap::new(),
            labels: HashMap::new(),
            var_counter: 0,
        }
    }

    fn make_temporary_variable(&mut self, name: &str) -> String {
        let count = self.var_counter;
        self.var_counter += 1;
        format!("{name}.{count}.decl")
    }

    fn resolve_variable(&mut self, name: &str, resolved_name: String, in_current_scope: bool) {
        self.renamed_variables
            .insert(name.into(), (resolved_name, in_current_scope));
    }

    fn get_resolved_variable(&self, name: &str) -> Option<String> {
        self.renamed_variables
            .get(name)
            .map(|(name, _scope)| name.into())
    }

    fn variable_declared_in_current_block(&self, name: &str) -> bool {
        self.renamed_variables
            .get(name)
            .map(|(_name, in_current_scope)| *in_current_scope)
            .unwrap_or(false)
    }

    // RULES: If unseen, add an entry. Return name.
    // If trying to declare but already seen and declared, return error .
    // If seen but not yet declared and trying to declare, update in place.
    fn resolve_label(
        &mut self,
        name: &str,
        declaration: bool,
    ) -> Result<String, SemanticAnalysisError> {
        match self.labels.entry(name.into()) {
            Entry::Vacant(v) => {
                let count = self.var_counter;
                self.var_counter += 1;
                let s = format!("{name}.{count}.label");
                v.insert((s.clone(), declaration));
                Ok(s)
            }
            Entry::Occupied(mut o) => {
                let already_declared = o.get().1;
                // Did we already define "foo:"?
                if already_declared && declaration {
                    Err(SemanticAnalysisError::DuplicateLabel(name.to_string()))
                } else {
                    // At this point, we know that we already saw the label in some context.
                    // Just write the declaration to either what's existing in place, or the
                    // incoming.
                    o.get_mut().1 = already_declared || declaration;
                    Ok(o.get().0.clone())
                }
            }
        }
    }

    fn copy_resolver_and_reset_scopes(&self) -> Self {
        let mut new_resolver = self.clone();
        for (_name, (_new_name, in_scope)) in new_resolver.renamed_variables.iter_mut() {
            *in_scope = false;
        }
        new_resolver
    }
}

pub fn resolve<'a>(ast: &mut AST<'a>) -> Result<(), SemanticAnalysisError> {
    let mut resolver = Resolver::new();
    resolve_ast(ast, &mut resolver)?;
    validate_labels(&resolver)
}

fn validate_labels<'a>(resolver: &Resolver) -> Result<(), SemanticAnalysisError> {
    for (label, declaration_status) in resolver.labels.values() {
        if !declaration_status {
            return Err(SemanticAnalysisError::UndeclaredLabel(label.clone()));
        }
    }

    Ok(())
}

fn resolve_ast<'a>(
    ast: &mut AST<'a>,
    resolver: &mut Resolver,
) -> Result<(), SemanticAnalysisError> {
    let AST::Program(function) = ast else {
        return Err(SemanticAnalysisError::UnexpectedNonProgramNode);
    };

    let AST::Function { block, name: _ } = function.as_mut() else {
        return Err(SemanticAnalysisError::UnexpectedNonFunctionNode);
    };

    resolve_block(block, resolver)
}

fn resolve_block<'a>(
    block: &mut Block,
    resolver: &mut Resolver,
) -> Result<(), SemanticAnalysisError> {
    let Block(ref mut body) = block;
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
    if resolver.variable_declared_in_current_block(&decl.name) {
        return Err(SemanticAnalysisError::DuplicateDecl(decl.name.clone()));
    };
    let renamed_var = resolver.make_temporary_variable(&decl.name);
    resolver.resolve_variable(&decl.name, renamed_var.clone(), true);
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
        Statement::If {
            condition,
            then,
            else_,
        } => {
            resolve_expr(&mut *condition, resolver)?;
            resolve_statement(&mut *then, resolver)?;
            if let Some(expr) = else_ {
                resolve_statement(&mut **expr, resolver)?;
            };
        }
        Statement::Goto(lbl) => {
            let renamed = resolver.resolve_label(lbl, false)?;
            *lbl = renamed;
        }
        Statement::Labelled { label, statement } => {
            let renamed = resolver.resolve_label(label, true)?;
            *label = renamed;
            resolve_statement(&mut *statement, resolver)?;
        }
        Statement::Compound(block) => {
            let mut new_resolver = resolver.copy_resolver_and_reset_scopes();
            resolve_block(block, &mut new_resolver)?;
        }
    };
    Ok(())
}

fn resolve_expr(
    expr: &mut Expression,
    resolver: &mut Resolver,
) -> Result<(), SemanticAnalysisError> {
    match expr {
        Expression::Constant(_) => (),
        Expression::Var(ident) => match resolver.get_resolved_variable(&ident) {
            Some(renamed) => *ident = renamed,
            None => {
                return Err(SemanticAnalysisError::UndeclaredVariableInInitializer(
                    ident.clone(),
                ))
            }
        },
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
        Expression::Conditional {
            condition,
            then,
            else_,
        } => {
            resolve_expr(&mut *condition, resolver)?;
            resolve_expr(&mut *then, resolver)?;
            resolve_expr(&mut *else_, resolver)?;
        }
    };
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::BinaryOp;
    use crate::parser::Block;
    use crate::parser::BlockItem;
    use crate::parser::Expression;
    use crate::parser::Statement;
    use crate::parser::AST;

    #[test]
    fn basic_resolve_repeated_variable() {
        let mut before = AST::Program(Box::new(AST::Function {
            name: "main",
            block: Block(vec![
                BlockItem::Decl(Declaration {
                    name: "a".into(),
                    init: Some(Expression::Constant(1)),
                }),
                BlockItem::Decl(Declaration {
                    name: "a".into(),
                    init: Some(Expression::Constant(1)),
                }),
                BlockItem::Stmt(Statement::Return(Expression::Var("a".into()))),
            ]),
        }));

        let actual = resolve(&mut before);
        assert!(actual.is_err());
    }

    #[test]
    fn basic_resolve_undeclared_variable() {
        let mut before = AST::Program(Box::new(AST::Function {
            name: "main",
            block: Block(vec![
                BlockItem::Decl(Declaration {
                    name: "a".into(),
                    init: Some(Expression::Var("c".into())),
                }),
                BlockItem::Stmt(Statement::Return(Expression::Var("a".into()))),
            ]),
        }));

        let actual = resolve(&mut before);
        assert!(actual.is_err());
    }

    #[test]
    fn basic_resolve_successful() {
        let mut before = AST::Program(Box::new(AST::Function {
            name: "main",
            block: Block(vec![
                BlockItem::Decl(Declaration {
                    name: "a".into(),
                    init: Some(Expression::Constant(1)),
                }),
                BlockItem::Decl(Declaration {
                    name: "b".into(),
                    init: Some(Expression::Var("a".into())),
                }),
                BlockItem::Stmt(Statement::Return(Expression::Var("a".into()))),
            ]),
        }));

        let actual = resolve(&mut before);
        assert!(actual.is_ok());
        let expected = AST::Program(Box::new(AST::Function {
            name: "main",
            block: Block(vec![
                BlockItem::Decl(Declaration {
                    name: "a.0.decl".into(),
                    init: Some(Expression::Constant(1)),
                }),
                BlockItem::Decl(Declaration {
                    name: "b.1.decl".into(),
                    init: Some(Expression::Var("a.0.decl".into())),
                }),
                BlockItem::Stmt(Statement::Return(Expression::Var("a.0.decl".into()))),
            ]),
        }));

        // mutates taken value
        assert_eq!(before, expected);
    }

    #[test]
    fn complex_resolve_successful() {
        let mut before = AST::Program(Box::new(AST::Function {
            name: "main",
            block: Block(vec![
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
            ]),
        }));

        let actual = resolve(&mut before);
        assert!(actual.is_ok());
        let expected = AST::Program(Box::new(AST::Function {
            name: "main",
            block: Block(vec![
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
            ]),
        }));

        // mutates taken value
        assert_eq!(before, expected);
    }

    #[test]
    fn test_repeated_labels() {
        let src = r#"
            int main(void) {
                goto foo;
                foo:
                    return 1 + 2;
                foo:
                    return 3 + 4;
            }
        "#;
        let lexer = crate::lexer::Lexer::lex(src).unwrap();
        let tokens = lexer.as_syntactic_tokens();
        let parse = crate::parser::Parser::new(&tokens);
        let mut ast = parse.into_ast().unwrap();
        let analysis = resolve(&mut ast);
        assert!(analysis.is_err());
        assert_eq!(
            analysis,
            Err(SemanticAnalysisError::DuplicateLabel("foo".into()))
        );
    }

    #[test]
    fn test_undeclared_labels() {
        let src = r#"
            int main(void) {
                goto bar;
                goto foo;
                bar:
                    return 0;
            }
        "#;
        let lexer = crate::lexer::Lexer::lex(src).unwrap();
        let tokens = lexer.as_syntactic_tokens();
        let parse = crate::parser::Parser::new(&tokens);
        let mut ast = parse.into_ast().unwrap();
        let analysis = resolve(&mut ast);
        assert!(analysis.is_err());
        assert_eq!(
            analysis,
            Err(SemanticAnalysisError::UndeclaredLabel("foo.1.label".into()))
        );
    }

    #[test]
    fn test_duplicate_variable_decls_in_blocks_fails() {
        let src = r#"
            int main(void) {
                int x = 1;
                { 
                    int b = 1;
                    int b = 2;
                }
            }
        "#;
        let lexer = crate::lexer::Lexer::lex(src).unwrap();
        let tokens = lexer.as_syntactic_tokens();
        let parse = crate::parser::Parser::new(&tokens);
        let mut ast = parse.into_ast().unwrap();
        let analysis = resolve(&mut ast);
        assert!(analysis.is_err());
        assert_eq!(
            analysis,
            Err(SemanticAnalysisError::DuplicateDecl("b".into()))
        );
    }

    #[test]
    fn shadowing_variables_in_blocks_renames_inner_scope() {
        let src = r#"
            int main(void) {
                int x = 1;
                { 
                    int x = 2;
                }
                return x;
            }
        "#;
        let lexer = crate::lexer::Lexer::lex(src).unwrap();
        let tokens = lexer.as_syntactic_tokens();
        let parse = crate::parser::Parser::new(&tokens);
        let mut ast = parse.into_ast().unwrap();
        let analysis = resolve(&mut ast);
        assert!(analysis.is_ok());

        let expected = AST::Program(Box::new(AST::Function {
            name: "main",
            block: Block(vec![
                BlockItem::Decl(Declaration {
                    name: "x.0.decl".into(),
                    init: Some(Expression::Constant(1)),
                }),
                BlockItem::Stmt(Statement::Compound(Block(vec![BlockItem::Decl(
                    // Declaration is the same name, but has a new renamed variable
                    // since we want to ensure any label is pinned to one value
                    Declaration {
                        name: "x.1.decl".into(),
                        init: Some(Expression::Constant(2)),
                    },
                )]))),
                BlockItem::Stmt(Statement::Return(Expression::Var("x.0.decl".into()))),
            ]),
        }));

        assert_eq!(ast, expected);
    }
}
