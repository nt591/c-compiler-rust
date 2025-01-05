use crate::parser::Block;
use crate::parser::BlockItem;
use crate::parser::Declaration;
use crate::parser::Expression;
use crate::parser::ForInit;
use crate::parser::FunctionDeclaration;
use crate::parser::Statement;
use crate::parser::VariableDeclaration;
use crate::parser::AST;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum SemanticAnalysisError {
    #[error("Got duplicate variable declaration {0}")]
    DuplicateDecl(String),
    #[error("Found undeclared variable in initializer: {0}")]
    UndeclaredVariableInInitializer(String),
    #[error("Found non-var node on lefthand side of assignment")]
    InvalidLhsAssignmentNode,
    #[error("Found duplicate label declaration: {0}")]
    DuplicateLabel(String),
    #[error("Undeclared label {0}")]
    UndeclaredLabel(String),
    #[error("Found a Break statement outside of a loop")]
    BreakWithoutLoopConstruct,
    #[error("Found a Continue statement outside of a loop")]
    ContinueWithoutLoopConstruct,
    #[error("Undeclared function")]
    UndeclaredFunction,
    #[error("Duplicate function declaration {0}")]
    DuplicateFunction(String),
    #[error("Nested function definitions are not allowed")]
    NestedFunctionDefinitionsNotAllowed,
    #[error("Expected a function name but found a variable: {0}")]
    VariableUsedAsFunctionName(String),
    #[error("Expected a variable name but found a function: {0}")]
    FunctionUsedAsVariableName(String),
    #[error("Function called with wrong number of arguments")]
    FunctionCalledWithWrongNumOfArgs,
    #[error("Incompatible function declaration")]
    IncompatibleFunctionDeclaration,
}

#[derive(Debug, Clone)]
struct ResolvedIdentifier {
    name: String,
    in_current_scope: bool,
    has_linkage: bool,
}

#[derive(Debug, Clone)]
struct Resolver {
    renamed_variables: HashMap<String, ResolvedIdentifier>,
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

    fn make_loop_label(&mut self, base: &str) -> String {
        let count = self.var_counter;
        self.var_counter += 1;
        format!("{base}.{count}")
    }

    fn resolve_variable(
        &mut self,
        name: &str,
        resolved_name: String,
        in_current_scope: bool,
        has_linkage: bool,
    ) {
        self.renamed_variables.insert(
            name.into(),
            ResolvedIdentifier {
                name: resolved_name,
                in_current_scope,
                has_linkage,
            },
        );
    }

    fn get_resolved_variable(&self, name: &str) -> Option<String> {
        self.renamed_variables.get(name).map(|x| x.name.clone())
    }

    fn get_identifier(&self, name: &str) -> Option<&ResolvedIdentifier> {
        self.renamed_variables.get(name)
    }

    fn variable_declared_in_current_block(&self, name: &str) -> bool {
        self.renamed_variables
            .get(name)
            .map(|x| x.in_current_scope)
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

    // Every time we copy a resolver, we should
    // run all validations of labels just in case we have malformed
    // GOTO / labelled statements
    fn copy_resolver_and_reset_scopes(&self) -> Self {
        let mut new_resolver = self.clone();
        for (_name, x) in new_resolver.renamed_variables.iter_mut() {
            x.in_current_scope = false;
        }
        new_resolver
    }

    fn transfer_count(&mut self, other: &Resolver) {
        self.var_counter = other.var_counter;
    }
}

// BEGIN: typechecking constructs
#[derive(PartialEq)]
enum CType {
    Int,
    FunType(usize), // param_count
}

// TODO: lazy_static this bad boy
// If type is FunType, store if we've seen a definition (func body)
type SymbolTable = HashMap<String, (CType, Option<bool>)>;
// END: typechecking constructs

pub fn resolve(ast: &mut AST) -> Result<(), SemanticAnalysisError> {
    let mut resolver = Resolver::new();
    resolve_ast(ast, &mut resolver)?;
    label_and_validate_loop_constructs(ast, &mut resolver)?;
    validate_labels(&resolver)?;
    let mut table = SymbolTable::new();
    typecheck_ast(ast, &mut table)?;
    Ok(())
}

fn typecheck_ast(ast: &AST, symbol_table: &mut SymbolTable) -> Result<(), SemanticAnalysisError> {
    let AST::Program(functions) = ast;

    for function in functions {
        typecheck_function_declaration(function, symbol_table)?;
    }
    Ok(())
}

fn typecheck_function_declaration(
    decl: &FunctionDeclaration,
    symbol_table: &mut SymbolTable,
) -> Result<(), SemanticAnalysisError> {
    let already_defined = match symbol_table.get(&decl.name) {
        None => Ok(false),
        Some((CType::Int, _)) => Err(SemanticAnalysisError::IncompatibleFunctionDeclaration),
        Some((CType::FunType(count), _)) if *count != decl.params.len() => {
            Err(SemanticAnalysisError::IncompatibleFunctionDeclaration)
        }
        Some((_old_decl, Some(true))) if decl.block.is_some() => {
            Err(SemanticAnalysisError::DuplicateFunction(decl.name.clone()))
        }
        Some((_old_decl, Some(def))) => Ok(*def),
        Some((_, None)) => unreachable!(
            "Somehow found a function stored in the symbol table without an already_defined value"
        ),
    }?;
    symbol_table.insert(
        decl.name.clone(),
        (
            CType::FunType(decl.params.len()),
            Some(already_defined || decl.block.is_some()),
        ),
    );
    if let Some(block) = &decl.block {
        for param in decl.params.iter() {
            symbol_table.insert(param.clone(), (CType::Int, None));
        }
        typecheck_block(&block, symbol_table)?;
    }
    Ok(())
}

fn typecheck_block(
    decl: &Block,
    symbol_table: &mut SymbolTable,
) -> Result<(), SemanticAnalysisError> {
    let Block(body) = decl;
    for body_item in body.iter() {
        match body_item {
            BlockItem::Decl(Declaration::VarDecl(declaration)) => {
                typecheck_decl(declaration, symbol_table)?
            }
            BlockItem::Decl(Declaration::FunDecl(decl)) => {
                typecheck_function_declaration(decl, symbol_table)?;
            }
            BlockItem::Stmt(statement) => typecheck_statement(statement, symbol_table)?,
        }
    }

    Ok(())
}

fn typecheck_decl(
    decl: &VariableDeclaration,
    symbol_table: &mut SymbolTable,
) -> Result<(), SemanticAnalysisError> {
    let VariableDeclaration { init, name } = decl;
    symbol_table.insert(name.clone(), (CType::Int, None));
    if let Some(init) = init {
        typecheck_expr(init, symbol_table)?;
    };
    Ok(())
}

fn typecheck_statement(
    statement: &Statement,
    symbol_table: &mut SymbolTable,
) -> Result<(), SemanticAnalysisError> {
    match statement {
        Statement::Null => (),
        Statement::Expr(expr) => typecheck_expr(expr, symbol_table)?,
        Statement::Return(expr) => typecheck_expr(expr, symbol_table)?,
        Statement::If {
            condition,
            then,
            else_,
        } => {
            typecheck_expr(condition, symbol_table)?;
            typecheck_statement(then, symbol_table)?;
            if let Some(expr) = else_ {
                typecheck_statement(expr, symbol_table)?;
            };
        }
        Statement::Goto(_lbl) => (),
        Statement::Labelled { statement, .. } => {
            typecheck_statement(statement, symbol_table)?;
        }
        Statement::Compound(block) => {
            typecheck_block(block, symbol_table)?;
        }
        Statement::Break(_lbl) | Statement::Continue(_lbl) => (),
        Statement::DoWhile {
            condition, body, ..
        }
        | Statement::While {
            body, condition, ..
        } => {
            typecheck_expr(condition, symbol_table)?;
            typecheck_statement(body, symbol_table)?;
        }
        Statement::For {
            init,
            condition,
            post,
            body,
            ..
        } => {
            typecheck_for_init(init, symbol_table)?;
            typecheck_optional_expr(condition.as_ref(), symbol_table)?;
            typecheck_optional_expr(post.as_ref(), symbol_table)?;
            typecheck_statement(body.as_ref(), symbol_table)?;
        }
    };
    Ok(())
}

fn typecheck_for_init(
    init: &ForInit,
    symbol_table: &mut SymbolTable,
) -> Result<(), SemanticAnalysisError> {
    match init {
        ForInit::InitDecl(decl) => typecheck_decl(decl, symbol_table),
        ForInit::InitExp(expr) => typecheck_optional_expr(expr.as_ref(), symbol_table),
    }
}

fn typecheck_optional_expr(
    expr: Option<&Expression>,
    symbol_table: &mut SymbolTable,
) -> Result<(), SemanticAnalysisError> {
    if let Some(expression) = expr {
        typecheck_expr(expression, symbol_table)?;
    }
    Ok(())
}

fn typecheck_expr(
    expr: &Expression,
    symbol_table: &mut SymbolTable,
) -> Result<(), SemanticAnalysisError> {
    match expr {
        Expression::FunctionCall { name, args } => {
            let stored_type = symbol_table.get(name); // TODO: Can this be None?
            let Some((CType::FunType(count), _)) = stored_type else {
                return Err(SemanticAnalysisError::VariableUsedAsFunctionName(name.clone()));
            };
            if args.len() != *count {
                return Err(SemanticAnalysisError::FunctionCalledWithWrongNumOfArgs);
            }
            for arg in args {
                typecheck_expr(arg, symbol_table)?;
            }
            Ok(())
        }
        Expression::Var(name) => {
            let stored_type = symbol_table.get(name); // TODO: Can this be None?
            let Some((CType::Int, None)) = stored_type else {
                return Err(SemanticAnalysisError::FunctionUsedAsVariableName(name.clone()));
            };
            Ok(())
        }
        Expression::Constant(_) => Ok(()),
        Expression::Binary(_op, lhs, rhs) => {
            typecheck_expr(lhs.as_ref(), symbol_table)?;
            typecheck_expr(rhs.as_ref(), symbol_table)?;
            Ok(())
        }
        Expression::Unary(_op, expr) => {typecheck_expr(expr.as_ref(), symbol_table)?; Ok(()) }
        Expression::Conditional {
            condition,
            then,
            else_
        } => {
            typecheck_expr(condition.as_ref(), symbol_table)?;
            typecheck_expr(then.as_ref(), symbol_table)?;
            typecheck_expr(else_.as_ref(), symbol_table)?;
            Ok(())
        },
        Expression::Assignment(lhs, rhs) => {
            typecheck_expr(lhs.as_ref(), symbol_table)?;
            typecheck_expr(rhs.as_ref(), symbol_table)?;
            Ok(())
        }
    }
}

fn validate_labels(resolver: &Resolver) -> Result<(), SemanticAnalysisError> {
    for (label, declaration_status) in resolver.labels.values() {
        if !declaration_status {
            return Err(SemanticAnalysisError::UndeclaredLabel(label.clone()));
        }
    }

    Ok(())
}

fn resolve_ast(ast: &mut AST, resolver: &mut Resolver) -> Result<(), SemanticAnalysisError> {
    let AST::Program(functions) = ast;

    for function in functions {
        resolve_function_declaration(function, resolver)?;
    }
    Ok(())
}

fn resolve_block(block: &mut Block, resolver: &mut Resolver) -> Result<(), SemanticAnalysisError> {
    let Block(ref mut body) = block;
    for body_item in body.iter_mut() {
        match body_item {
            BlockItem::Decl(Declaration::VarDecl(declaration)) => {
                resolve_decl(declaration, resolver)?
            }
            BlockItem::Decl(Declaration::FunDecl(decl)) => {
                if decl.block.is_some() {
                    // Nested function definitions are not permitted
                    return Err(SemanticAnalysisError::NestedFunctionDefinitionsNotAllowed);
                }
                resolve_function_declaration(decl, resolver)?;
            }
            BlockItem::Stmt(statement) => resolve_statement(statement, resolver)?,
        }
    }

    Ok(())
}

fn resolve_function_declaration(
    decl: &mut FunctionDeclaration,
    resolver: &mut Resolver,
) -> Result<(), SemanticAnalysisError> {
    if let Some(identifier) = resolver.get_identifier(&decl.name) {
        if identifier.in_current_scope && !identifier.has_linkage {
            return Err(SemanticAnalysisError::DuplicateFunction(decl.name.clone()));
        }
    };

    resolver.resolve_variable(&decl.name, decl.name.clone(), true, true);
    let mut new_resolver = resolver.copy_resolver_and_reset_scopes();
    for param in decl.params.iter_mut() {
        resolve_param(param, &mut new_resolver)?;
    }
    if let Some(ref mut block) = decl.block {
        resolve_block(block, &mut new_resolver)?;
    }
    validate_labels(&new_resolver)?;
    resolver.transfer_count(&new_resolver);
    Ok(())
}

fn resolve_param(name: &mut String, resolver: &mut Resolver) -> Result<(), SemanticAnalysisError> {
    if resolver.variable_declared_in_current_block(&name) {
        return Err(SemanticAnalysisError::DuplicateDecl(name.clone()));
    };
    let renamed_var = resolver.make_temporary_variable(&name);
    resolver.resolve_variable(&name, renamed_var.clone(), true, false);
    *name = renamed_var;
    Ok(())
}

fn resolve_decl(
    decl: &mut VariableDeclaration,
    resolver: &mut Resolver,
) -> Result<(), SemanticAnalysisError> {
    let VariableDeclaration { name, init } = decl;
    resolve_param(name, resolver)?;
    if let Some(init) = init {
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
            validate_labels(&new_resolver)?;
            resolver.transfer_count(&new_resolver);
        }
        Statement::Break(_lbl) | Statement::Continue(_lbl) => (),
        Statement::DoWhile {
            condition, body, ..
        }
        | Statement::While {
            body, condition, ..
        } => {
            resolve_expr(&mut *condition, resolver)?;
            resolve_statement(&mut **body, resolver)?;
        }
        Statement::For {
            init,
            condition,
            post,
            body,
            ..
        } => {
            let mut new_resolver = resolver.copy_resolver_and_reset_scopes();
            resolve_for_init(&mut *init, &mut new_resolver)?;
            resolve_optional_expr(condition.as_mut(), &mut new_resolver)?;
            resolve_optional_expr(post.as_mut(), &mut new_resolver)?;
            resolve_statement(&mut **body, &mut new_resolver)?;
            validate_labels(&new_resolver)?;
            resolver.transfer_count(&new_resolver);
        }
    };
    Ok(())
}

fn resolve_for_init(
    init: &mut ForInit,
    resolver: &mut Resolver,
) -> Result<(), SemanticAnalysisError> {
    match init {
        ForInit::InitDecl(ref mut decl) => resolve_decl(decl, resolver),
        ForInit::InitExp(expr) => resolve_optional_expr(expr.as_mut(), resolver),
    }
}

fn resolve_optional_expr(
    mut expr: Option<&mut Expression>,
    resolver: &mut Resolver,
) -> Result<(), SemanticAnalysisError> {
    if let Some(ref mut expression) = expr {
        resolve_expr(expression, resolver)?;
    }
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
        Expression::FunctionCall { name, args } => {
            let ident = resolver
                .get_identifier(name)
                .ok_or_else(|| SemanticAnalysisError::UndeclaredFunction)?;
            // we reset the name here, just for typechecking
            *name = ident.name.clone();
            for arg in args.iter_mut() {
                resolve_expr(arg, resolver)?;
            }
        }
    };
    Ok(())
}

fn label_and_validate_loop_constructs(
    ast: &mut AST,
    resolver: &mut Resolver,
) -> Result<(), SemanticAnalysisError> {
    let AST::Program(functions) = ast;

    for function in functions {
        let FunctionDeclaration { block, name: _, .. } = function;
        if let Some(block) = block.as_mut() {
            label_block(block, None, resolver)?;
        };
    }

    Ok(())
}

fn label_block(
    block: &mut Block,
    current_label: Option<String>,
    resolver: &mut Resolver,
) -> Result<(), SemanticAnalysisError> {
    let Block(ref mut body) = block;
    for body_item in body.iter_mut() {
        if let BlockItem::Stmt(statement) = body_item {
            label_statement(statement, current_label.clone(), resolver)?;
        }
    }

    Ok(())
}

fn label_statement(
    statement: &mut Statement,
    current_label: Option<String>,
    resolver: &mut Resolver,
) -> Result<(), SemanticAnalysisError> {
    use Statement::*;
    match statement {
        Goto(_) | Return(_) | Expr(_) | Null => Ok(()),
        Break(_) if current_label.is_none() => {
            Err(SemanticAnalysisError::BreakWithoutLoopConstruct)
        }
        Break(lbl) => {
            *lbl = current_label.unwrap().clone();
            Ok(())
        }
        Continue(_) if current_label.is_none() => {
            Err(SemanticAnalysisError::ContinueWithoutLoopConstruct)
        }
        Continue(lbl) => {
            *lbl = current_label.unwrap().clone();
            Ok(())
        }
        While { body, label, .. } => {
            let new_label = resolver.make_loop_label("while_label");
            label_statement(&mut *body, Some(new_label.clone()), resolver)?;
            *label = new_label;
            Ok(())
        }
        DoWhile { body, label, .. } => {
            let new_label = resolver.make_loop_label("do_while_label");
            label_statement(&mut *body, Some(new_label.clone()), resolver)?;
            *label = new_label;
            Ok(())
        }
        For { label, body, .. } => {
            let new_label = resolver.make_loop_label("for_label");
            label_statement(&mut *body, Some(new_label.clone()), resolver)?;
            *label = new_label;
            Ok(())
        }
        Labelled { statement, .. } => label_statement(&mut *statement, current_label, resolver),
        Compound(block) => label_block(block, current_label, resolver),
        If { then, else_, .. } => {
            label_statement(then, current_label.clone(), resolver)?;
            if let Some(stmt) = else_ {
                label_statement(stmt, current_label, resolver)?;
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::BinaryOp;
    use crate::parser::Block;
    use crate::parser::BlockItem;
    use crate::parser::Expression;
    use crate::parser::FunctionDeclaration;
    use crate::parser::Statement;
    use crate::parser::AST;

    #[test]
    fn basic_resolve_repeated_variable() {
        let mut before = AST::Program(vec![FunctionDeclaration {
            name: "main".into(),
            block: Some(Block(vec![
                BlockItem::Decl(Declaration::VarDecl(VariableDeclaration {
                    name: "a".into(),
                    init: Some(Expression::Constant(1)),
                })),
                BlockItem::Decl(Declaration::VarDecl(VariableDeclaration {
                    name: "a".into(),
                    init: Some(Expression::Constant(1)),
                })),
                BlockItem::Stmt(Statement::Return(Expression::Var("a".into()))),
            ])),
            params: vec![],
        }]);

        let actual = resolve(&mut before);
        assert!(actual.is_err());
    }

    #[test]
    fn basic_resolve_undeclared_variable() {
        let mut before = AST::Program(vec![FunctionDeclaration {
            name: "main".into(),
            block: Some(Block(vec![
                BlockItem::Decl(Declaration::VarDecl(VariableDeclaration {
                    name: "a".into(),
                    init: Some(Expression::Var("c".into())),
                })),
                BlockItem::Stmt(Statement::Return(Expression::Var("a".into()))),
            ])),
            params: vec![],
        }]);

        let actual = resolve(&mut before);
        assert!(actual.is_err());
    }

    #[test]
    fn basic_resolve_successful() {
        let mut before = AST::Program(vec![FunctionDeclaration {
            name: "main".into(),
            block: Some(Block(vec![
                BlockItem::Decl(Declaration::VarDecl(VariableDeclaration {
                    name: "a".into(),
                    init: Some(Expression::Constant(1)),
                })),
                BlockItem::Decl(Declaration::VarDecl(VariableDeclaration {
                    name: "b".into(),
                    init: Some(Expression::Var("a".into())),
                })),
                BlockItem::Stmt(Statement::Return(Expression::Var("a".into()))),
            ])),
            params: vec![],
        }]);

        let actual = resolve(&mut before);
        assert!(actual.is_ok());
        let expected = AST::Program(vec![FunctionDeclaration {
            name: "main".into(),
            block: Some(Block(vec![
                BlockItem::Decl(Declaration::VarDecl(VariableDeclaration {
                    name: "a.0.decl".into(),
                    init: Some(Expression::Constant(1)),
                })),
                BlockItem::Decl(Declaration::VarDecl(VariableDeclaration {
                    name: "b.1.decl".into(),
                    init: Some(Expression::Var("a.0.decl".into())),
                })),
                BlockItem::Stmt(Statement::Return(Expression::Var("a.0.decl".into()))),
            ])),
            params: vec![],
        }]);

        // mutates taken value
        assert_eq!(before, expected);
    }

    #[test]
    fn complex_resolve_successful() {
        let mut before = AST::Program(vec![FunctionDeclaration {
            name: "main".into(),
            block: Some(Block(vec![
                BlockItem::Decl(Declaration::VarDecl(VariableDeclaration {
                    name: "a".into(),
                    init: Some(Expression::Constant(1)),
                })),
                BlockItem::Decl(Declaration::VarDecl(VariableDeclaration {
                    name: "b".into(),
                    init: Some(Expression::Var("a".into())),
                })),
                BlockItem::Decl(Declaration::VarDecl(VariableDeclaration {
                    name: "c".into(),
                    init: Some(Expression::Binary(
                        BinaryOp::Add,
                        Box::new(Expression::Var("a".into())),
                        Box::new(Expression::Var("b".into())),
                    )),
                })),
                BlockItem::Stmt(Statement::Return(Expression::Var("c".into()))),
            ])),
            params: vec![],
        }]);

        let actual = resolve(&mut before);
        assert!(actual.is_ok());
        let expected = AST::Program(vec![FunctionDeclaration {
            name: "main".into(),
            block: Some(Block(vec![
                BlockItem::Decl(Declaration::VarDecl(VariableDeclaration {
                    name: "a.0.decl".into(),
                    init: Some(Expression::Constant(1)),
                })),
                BlockItem::Decl(Declaration::VarDecl(VariableDeclaration {
                    name: "b.1.decl".into(),
                    init: Some(Expression::Var("a.0.decl".into())),
                })),
                BlockItem::Decl(Declaration::VarDecl(VariableDeclaration {
                    name: "c.2.decl".into(),
                    init: Some(Expression::Binary(
                        BinaryOp::Add,
                        Box::new(Expression::Var("a.0.decl".into())),
                        Box::new(Expression::Var("b.1.decl".into())),
                    )),
                })),
                BlockItem::Stmt(Statement::Return(Expression::Var("c.2.decl".into()))),
            ])),
            params: vec![],
        }]);

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

        let expected = AST::Program(vec![FunctionDeclaration {
            name: "main".into(),
            block: Some(Block(vec![
                BlockItem::Decl(Declaration::VarDecl(VariableDeclaration {
                    name: "x.0.decl".into(),
                    init: Some(Expression::Constant(1)),
                })),
                BlockItem::Stmt(Statement::Compound(Block(vec![BlockItem::Decl(
                    // Declaration is the same name, but has a new renamed variable
                    // since we want to ensure any label is pinned to one value
                    Declaration::VarDecl(VariableDeclaration {
                        name: "x.1.decl".into(),
                        init: Some(Expression::Constant(2)),
                    }),
                )]))),
                BlockItem::Stmt(Statement::Return(Expression::Var("x.0.decl".into()))),
            ])),
            params: vec![],
        }]);

        assert_eq!(ast, expected);
    }

    #[test]
    fn loop_labeling_and_exit_statements() {
        let src = r#"
        int main(void) {
            int a;
            for (a = 1; a < 10; a = a + 1) {
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
        let parse = crate::parser::Parser::new(&tokens);
        let mut ast = parse.into_ast().unwrap();
        let analysis = resolve(&mut ast);
        assert!(analysis.is_ok());

        let expected = AST::Program(vec![FunctionDeclaration {
            name: "main".into(),
            block: Some(Block(vec![
                BlockItem::Decl(Declaration::VarDecl(VariableDeclaration {
                    name: "a.0.decl".into(),
                    init: None,
                })),
                BlockItem::Stmt(Statement::For {
                    init: ForInit::InitExp(Some(Expression::Assignment(
                        Box::new(Expression::Var("a.0.decl".into())),
                        Box::new(Expression::Constant(1)),
                    ))),
                    condition: Some(Expression::Binary(
                        BinaryOp::LessThan,
                        Box::new(Expression::Var("a.0.decl".into())),
                        Box::new(Expression::Constant(10)),
                    )),
                    post: Some(Expression::Assignment(
                        Box::new(Expression::Var("a.0.decl".into())),
                        Box::new(Expression::Binary(
                            BinaryOp::Add,
                            Box::new(Expression::Var("a.0.decl".into())),
                            Box::new(Expression::Constant(1)),
                        )),
                    )),
                    body: Box::new(Statement::Compound(Block(vec![BlockItem::Stmt(
                        Statement::Continue("for_label.1".into()),
                    )]))),
                    label: "for_label.1".into(),
                }),
                BlockItem::Stmt(Statement::DoWhile {
                    body: Box::new(Statement::Compound(Block(vec![BlockItem::Stmt(
                        Statement::Continue("do_while_label.2".into()),
                    )]))),
                    condition: Expression::Binary(
                        BinaryOp::LessThan,
                        Box::new(Expression::Var("a.0.decl".into())),
                        Box::new(Expression::Constant(0)),
                    ),
                    label: "do_while_label.2".into(),
                }),
                BlockItem::Stmt(Statement::While {
                    condition: Expression::Binary(
                        BinaryOp::GreaterThan,
                        Box::new(Expression::Var("a.0.decl".into())),
                        Box::new(Expression::Constant(0)),
                    ),
                    body: Box::new(Statement::Break("while_label.3".into())),
                    label: "while_label.3".into(),
                }),
            ])),
            params: vec![],
        }]);
        assert_eq!(ast, expected);
    }

    #[test]
    fn break_and_continue_must_be_in_loops() {
        let src = r#"
        int main(void) {
            continue; 
        }
        "#;
        let lexer = crate::lexer::Lexer::lex(src).unwrap();
        let tokens = lexer.as_syntactic_tokens();
        let parse = crate::parser::Parser::new(&tokens);
        let mut ast = parse.into_ast().unwrap();
        let analysis = resolve(&mut ast);
        assert!(analysis.is_err());

        let src = r#"
            int main(void) {
                while (1) continue;
                break; 
            }
        "#;
        let lexer = crate::lexer::Lexer::lex(src).unwrap();
        let tokens = lexer.as_syntactic_tokens();
        let parse = crate::parser::Parser::new(&tokens);
        let mut ast = parse.into_ast().unwrap();
        let analysis = resolve(&mut ast);
        assert!(analysis.is_err());
    }
}
