use crate::ast::BinaryOp;
use crate::ast::Const;
use crate::ast::StorageClass;
use crate::ast::UnaryOp;
use crate::const_eval;
use crate::types::CType;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use thiserror::Error; // todo

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

pub type TypedAST = crate::ast::AST<CType>;
pub type TypedDeclaration = crate::ast::Declaration<CType>;
pub type TypedStatement = crate::ast::Statement<CType>;
pub type TypedExprKind = crate::ast::ExprKind<CType>;
pub type TypedExpression = crate::ast::Expression<CType>;
pub type TypedFunctionDeclaration = crate::ast::FunctionDeclaration<CType>;
pub type TypedVariableDeclaration = crate::ast::VariableDeclaration<CType>;
pub type TypedForInit = crate::ast::ForInit<CType>;
pub type TypedBlockItem = crate::ast::BlockItem<CType>;
pub type TypedBlock = crate::ast::Block<CType>;

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
    #[error("Conflicting local declarations")]
    ConflictingLocalDeclarations,
    #[error("Cannot use `static` specifier on block-scoped functions")]
    NoStaticOnBlockScopeFuncs,
    #[error("Static function declaration follows non-static declaration")]
    StaticFuncDeclFollowsNonStatic,
    #[error("Non-constant initializer")]
    NonConstantInitializer,
    #[error("Function redeclared as variable")]
    FunctionRedeclaredAsVariable,
    #[error("Conflicting variable linkage")]
    ConflictingVariableLinkage,
    #[error("Conflicting file-scoped variable declaration")]
    ConflictingFileScopeVarDecl,
    #[error("Initializer on local extern variable declaration")]
    InitializerOnLocalExternVarDecl,
    #[error("Non-constant initializer on local static variable declaration")]
    NonConstInitOnLocalStaticVar,
    #[error("Storage class specifier not allowed in for-loop init declaration")]
    StorageClassInForInit,
    #[error("Redeclared local var from {0:?} to {1:?}")]
    VariableDeclaredWithNewType(CType, CType),
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

/* BEGIN Symbol Table types for typechecking */
#[derive(PartialEq, Debug, Copy, Clone)]
pub enum StaticInit {
    IntInit(i32),
    LongInit(i64),
}

#[derive(PartialEq, Debug)]
pub enum InitialValue {
    Tentative,
    Initial(StaticInit),
    NoInitializer, // extern variable declarations are not tentative
}

#[derive(Debug, PartialEq)]
pub enum IdentifierAttrs {
    FunAttr { defined: bool, global: bool },
    StaticAttr { init: InitialValue, global: bool },
    LocalAttr,
}

// TODO: harden types so CType::FunType ONLY goes with IdentifierAttrs::FunAttr
pub type SymbolTable = HashMap<String, (CType, IdentifierAttrs)>;

struct Ctx {
    symbol_table: SymbolTable,
    enclosing_func_ret_ty: Option<CType>,
}

impl Ctx {
    fn new(symbol_table: SymbolTable) -> Self {
        Self {
            symbol_table,
            enclosing_func_ret_ty: None,
        }
    }

    fn save_enclosing_func_ret_ty(&mut self, ctype: CType) {
        self.enclosing_func_ret_ty = Some(ctype)
    }

    fn reset_enclosing_func_ret_ty(&mut self) {
        self.enclosing_func_ret_ty = None;
    }

    fn take_symbol_table(self) -> SymbolTable {
        self.symbol_table
    }
}
/* END: Symbol Table types for typechecking */

pub fn resolve(ast: &mut AST) -> Result<(SymbolTable, TypedAST), SemanticAnalysisError> {
    let mut resolver = Resolver::new();
    resolve_ast(ast, &mut resolver)?;
    label_and_validate_loop_constructs(ast, &mut resolver)?;
    validate_labels(&resolver)?;
    let table = SymbolTable::new();
    let mut ctx = Ctx::new(table);
    let typed_ast = typecheck_ast(ast, &mut ctx)?;
    let table = ctx.take_symbol_table();
    Ok((table, typed_ast))
}

fn typecheck_ast(ast: &AST, ctx: &mut Ctx) -> Result<TypedAST, SemanticAnalysisError> {
    let AST::Program(decls) = ast;
    let mut typed_decls = Vec::with_capacity(decls.len());
    for decl in decls {
        let d = match decl {
            Declaration::FunDecl(function) => {
                TypedDeclaration::FunDecl(typecheck_function_declaration(function, ctx)?)
            }
            Declaration::VarDecl(var) => {
                TypedDeclaration::VarDecl(typecheck_file_scope_variable_declaration(var, ctx)?)
            }
        };
        typed_decls.push(d)
    }
    Ok(TypedAST::Program(typed_decls))
}

fn typecheck_function_declaration(
    decl: &FunctionDeclaration,
    ctx: &mut Ctx,
) -> Result<TypedFunctionDeclaration, SemanticAnalysisError> {
    let CType::FunType { params, .. } = &decl.ftype else {
        unreachable!();
    };
    let has_body = decl.block.is_some();
    let mut already_defined = false;
    let mut global = decl.storage_class != Some(StorageClass::Static);
    if let Some((old_ctype, attrs)) = ctx.symbol_table.get(&decl.name) {
        let IdentifierAttrs::FunAttr {
            defined,
            global: old_global,
        } = attrs
        else {
            panic!(
                "Somehow fetched attrs that are not FunType when querying symbol for typechecking function declaration"
            );
        };
        let CType::FunType {
            params: old_params, ..
        } = old_ctype
        else {
            return Err(SemanticAnalysisError::IncompatibleFunctionDeclaration);
        };

        // ensure params of new declaration match old declaration (length, param type)
        if params != old_params {
            return Err(SemanticAnalysisError::IncompatibleFunctionDeclaration);
        };
        let defined = match old_ctype {
            _old_decl if *defined && has_body => {
                Err(SemanticAnalysisError::DuplicateFunction(decl.name.clone()))
            }
            _old_decl if *old_global && decl.storage_class == Some(StorageClass::Static) => {
                Err(SemanticAnalysisError::StaticFuncDeclFollowsNonStatic)
            }
            _old_decl => Ok(*defined),
        }?;
        already_defined = defined;
        global = *old_global;
    };
    ctx.symbol_table.insert(
        decl.name.clone(),
        (
            decl.ftype.clone(),
            IdentifierAttrs::FunAttr {
                defined: (already_defined || decl.block.is_some()),
                global,
            },
        ),
    );
    let typed_block = if let Some(block) = &decl.block {
        for (param, ty) in decl.params.iter().zip(params) {
            ctx.symbol_table
                .insert(param.clone(), (ty.clone(), IdentifierAttrs::LocalAttr));
        }
        // store the return type of the function so we can use it for return statements.
        let CType::FunType { ref ret, .. } = decl.ftype else {
            unreachable!("Somehow got a non-funtype when unpacking a function decl ctype")
        };
        ctx.save_enclosing_func_ret_ty(*ret.clone());
        let b = typecheck_block(block, ctx)?;
        ctx.reset_enclosing_func_ret_ty();
        Some(b)
    } else {
        None
    };
    Ok(TypedFunctionDeclaration {
        name: decl.name.clone(),
        params: decl.params.clone(),
        block: typed_block,
        storage_class: decl.storage_class.clone(),
        ftype: decl.ftype.clone(),
    })
}

fn typecheck_file_scope_variable_declaration(
    var: &VariableDeclaration,
    ctx: &mut Ctx,
) -> Result<TypedVariableDeclaration, SemanticAnalysisError> {
    // first, figure out variable initial value if available
    // Can either be a known constant value,
    // Tentative if the storage class is Static or unknown,
    // or explicitly no initializer for an extern variable
    let mut initial_value = match &var.init {
        None if var.storage_class == Some(StorageClass::Extern) => InitialValue::NoInitializer,
        None => InitialValue::Tentative,
        Some(expr) => match *expr.kind {
            ExprKind::Constant(c) => {
                let converted = const_eval::convert_const(c, &var.vtype);
                InitialValue::Initial(converted)
            }
            _ => return Err(SemanticAnalysisError::NonConstantInitializer),
        },
    };

    // is this variable global?
    // Yes IF it isn't explicitly static
    let mut global = var.storage_class != Some(StorageClass::Static);
    // now check previous declarations to make sure it's valid,
    // and to resolve storage classes and linkage.
    if let Some((old_ctype, old_attrs)) = ctx.symbol_table.get(&var.name) {
        // have we previously defined this as a function?
        if *old_ctype != var.vtype {
            return Err(SemanticAnalysisError::FunctionRedeclaredAsVariable);
        }
        // if we've declared this variable as extern,
        // use the old entry's linkage, else make
        // sure they don't disagree.
        let IdentifierAttrs::StaticAttr {
            init: old_init,
            global: old_global,
        } = old_attrs
        else {
            panic!(); // no way to here unless a compiler error
        };
        if let Some(StorageClass::Extern) = var.storage_class {
            global = *old_global;
        } else if global != *old_global {
            return Err(SemanticAnalysisError::ConflictingVariableLinkage);
        }

        // if we've defined the variable with an initial value already,
        // AND we are declaring it again with an initial value, error.
        // Else if we're checking a declaration with no initial value,
        // and we've already called it Tentative, keep it tentative.
        match old_init {
            InitialValue::Initial(c) => {
                if let InitialValue::Initial(_) = initial_value {
                    return Err(SemanticAnalysisError::ConflictingFileScopeVarDecl);
                }
                initial_value = InitialValue::Initial(*c);
            }
            InitialValue::Tentative => {
                if !matches!(initial_value, InitialValue::Initial(_)) {
                    initial_value = InitialValue::Tentative;
                }
            }
            _ => {}
        }
    }
    let typed_init = match initial_value {
        _ if var.init.is_none() => None,
        InitialValue::Initial(StaticInit::IntInit(v)) => Some(TypedExpression {
            ty: CType::Int,
            kind: Box::new(TypedExprKind::Constant(Const::Int(v))),
        }),
        InitialValue::Initial(StaticInit::LongInit(v)) => Some(TypedExpression {
            ty: CType::Long,
            kind: Box::new(TypedExprKind::Constant(Const::Long(v))),
        }),
        _ => unreachable!(),
    };
    ctx.symbol_table.insert(
        var.name.clone(),
        (
            var.vtype.clone(),
            IdentifierAttrs::StaticAttr {
                global,
                init: initial_value,
            },
        ),
    );
    Ok(TypedVariableDeclaration {
        name: var.name.clone(),
        init: typed_init,
        storage_class: var.storage_class,
        vtype: var.vtype.clone(),
    })
}

fn typecheck_block(decl: &Block, ctx: &mut Ctx) -> Result<TypedBlock, SemanticAnalysisError> {
    let crate::ast::Block(body) = decl;
    let mut typed_body = Vec::with_capacity(body.len());
    for body_item in body.iter() {
        let block_item = match body_item {
            BlockItem::Decl(Declaration::VarDecl(declaration)) => TypedBlockItem::Decl(
                TypedDeclaration::VarDecl(typecheck_local_var_decl(declaration, ctx)?),
            ),
            BlockItem::Decl(Declaration::FunDecl(decl)) => TypedBlockItem::Decl(
                TypedDeclaration::FunDecl(typecheck_function_declaration(decl, ctx)?),
            ),
            BlockItem::Stmt(statement) => {
                TypedBlockItem::Stmt(typecheck_statement(statement, ctx)?)
            }
        };
        typed_body.push(block_item);
    }

    Ok(crate::ast::Block(typed_body))
}

fn typecheck_local_var_decl(
    decl: &VariableDeclaration,
    ctx: &mut Ctx,
) -> Result<TypedVariableDeclaration, SemanticAnalysisError> {
    let VariableDeclaration {
        init,
        name,
        storage_class,
        ..
    } = decl;
    let typed_var_decl = match storage_class {
        Some(StorageClass::Extern) => {
            // an extern variable cannot be initialized at block scope, so
            // raise an error if we see an expression in the init position.
            if decl.init.is_some() {
                return Err(SemanticAnalysisError::InitializerOnLocalExternVarDecl);
            }
            // ensure we're not redefining anything
            if let Some((old_ctype, _old_attrs)) = ctx.symbol_table.get(name.as_str()) {
                if *old_ctype != decl.vtype {
                    return Err(SemanticAnalysisError::VariableDeclaredWithNewType(
                        old_ctype.clone(),
                        decl.vtype.clone(),
                    ));
                }
            } else {
                // fair game, add to symbol table.
                ctx.symbol_table.insert(
                    name.clone(),
                    (
                        decl.vtype.clone(),
                        IdentifierAttrs::StaticAttr {
                            init: InitialValue::NoInitializer,
                            global: true,
                        },
                    ),
                );
            }
            TypedVariableDeclaration {
                name: name.clone(),
                init: None,
                storage_class: Some(StorageClass::Extern),
                vtype: decl.vtype.clone(),
            }
        }
        Some(StorageClass::Static) => {
            // for a static block-scope variable, there is no linkage.
            // We either take the initializer
            // or default to an initial value of zero if not present.
            // Any non-constant initializer is a type error.
            let new_init = match init {
                None => {
                    InitialValue::Initial(const_eval::convert_const(Const::Int(0), &decl.vtype))
                }
                Some(expr) => match *expr.kind {
                    ExprKind::Constant(c) => {
                        let converted_const = const_eval::convert_const(c, &decl.vtype);
                        InitialValue::Initial(converted_const)
                    }
                    _ => return Err(SemanticAnalysisError::NonConstInitOnLocalStaticVar),
                },
            };
            // book says to just re-build here instead of typechecking.
            let typed_init = match new_init {
                _ if init.is_none() => None,
                InitialValue::Initial(StaticInit::IntInit(v)) => Some(TypedExpression {
                    ty: CType::Int,
                    kind: Box::new(TypedExprKind::Constant(Const::Int(v))),
                }),
                InitialValue::Initial(StaticInit::LongInit(v)) => Some(TypedExpression {
                    ty: CType::Long,
                    kind: Box::new(TypedExprKind::Constant(Const::Long(v))),
                }),
                _ => unreachable!(),
            };
            ctx.symbol_table.insert(
                name.clone(),
                (
                    decl.vtype.clone(),
                    IdentifierAttrs::StaticAttr {
                        init: new_init,
                        global: false,
                    },
                ),
            );
            TypedVariableDeclaration {
                name: name.clone(),
                init: typed_init,
                storage_class: Some(StorageClass::Static),
                vtype: decl.vtype.clone(),
            }
        }
        None => {
            // automatic variable: block scope, so easy to check
            ctx.symbol_table.insert(
                name.clone(),
                (decl.vtype.clone(), IdentifierAttrs::LocalAttr),
            );
            let typed_init = if let Some(init) = init {
                let e1 = typecheck_expr(init, ctx)?;
                Some(convert_to(e1, decl.vtype.clone()))
            } else {
                None
            };
            TypedVariableDeclaration {
                name: name.clone(),
                init: typed_init,
                storage_class: None,
                vtype: decl.vtype.clone(),
            }
        }
    };
    Ok(typed_var_decl)
}

fn typecheck_statement(
    statement: &Statement,
    ctx: &mut Ctx,
) -> Result<TypedStatement, SemanticAnalysisError> {
    let stmt = match statement {
        Statement::Null => TypedStatement::Null,
        Statement::Expr(expr) => TypedStatement::Expr(typecheck_expr(expr, ctx)?),
        Statement::Return(expr) => {
            let inner_expr = typecheck_expr(expr, ctx)?;
            let Some(ref ret_ty) = ctx.enclosing_func_ret_ty else {
                panic!(
                    "Got a return statement without setting the enclosing function's return type"
                );
            };
            let converted_expr = convert_to(inner_expr, ret_ty.clone());
            TypedStatement::Return(converted_expr)
        }
        Statement::If {
            condition,
            then,
            else_,
        } => {
            let else_stmt = match else_ {
                Some(stmt) => Some(Box::new(typecheck_statement(stmt, ctx)?)),
                None => None,
            };
            TypedStatement::If {
                condition: typecheck_expr(condition, ctx)?,
                then: Box::new(typecheck_statement(then, ctx)?),
                else_: else_stmt,
            }
        }
        Statement::Goto(lbl) => TypedStatement::Goto(lbl.clone()),
        Statement::Labelled { statement, label } => TypedStatement::Labelled {
            label: label.clone(),
            statement: Box::new(typecheck_statement(statement, ctx)?),
        },
        Statement::Compound(block) => TypedStatement::Compound(typecheck_block(block, ctx)?),
        Statement::Break(lbl) => TypedStatement::Break(lbl.clone()),
        Statement::Continue(lbl) => TypedStatement::Continue(lbl.clone()),
        Statement::DoWhile {
            condition,
            body,
            label,
        } => TypedStatement::DoWhile {
            condition: typecheck_expr(condition, ctx)?,
            body: Box::new(typecheck_statement(body, ctx)?),
            label: label.clone(),
        },
        Statement::While {
            condition,
            body,
            label,
        } => TypedStatement::While {
            condition: typecheck_expr(condition, ctx)?,
            body: Box::new(typecheck_statement(body, ctx)?),
            label: label.clone(),
        },
        Statement::For {
            init,
            condition,
            post,
            body,
            label,
        } => TypedStatement::For {
            init: typecheck_for_init(init, ctx)?,
            condition: typecheck_optional_expr(condition.as_ref(), ctx)?,
            post: typecheck_optional_expr(post.as_ref(), ctx)?,
            body: Box::new(typecheck_statement(body, ctx)?),
            label: label.clone(),
        },
    };
    Ok(stmt)
}

fn typecheck_for_init(
    init: &ForInit,
    ctx: &mut Ctx,
) -> Result<TypedForInit, SemanticAnalysisError> {
    let for_init = match init {
        ForInit::InitDecl(decl) => TypedForInit::InitDecl(typecheck_local_var_decl(decl, ctx)?),
        ForInit::InitExp(expr) => {
            TypedForInit::InitExp(typecheck_optional_expr(expr.as_ref(), ctx)?)
        }
    };
    Ok(for_init)
}

fn typecheck_optional_expr(
    expr: Option<&Expression>,
    ctx: &mut Ctx,
) -> Result<Option<TypedExpression>, SemanticAnalysisError> {
    let inner_expr = if let Some(expression) = expr {
        Some(typecheck_expr(expression, ctx)?)
    } else {
        None
    };
    Ok(inner_expr)
}

fn typecheck_expr(
    expr: &Expression,
    ctx: &mut Ctx,
) -> Result<TypedExpression, SemanticAnalysisError> {
    let (expr_kind, expr_type) = match expr.kind.as_ref() {
        ExprKind::Var(name) => {
            let stored_type = ctx
                .symbol_table
                .get(name)
                .expect("Didn't have a type for a variable expression");
            if let (CType::FunType { .. }, IdentifierAttrs::FunAttr { .. }) = stored_type {
                return Err(SemanticAnalysisError::FunctionUsedAsVariableName(
                    name.clone(),
                ));
            };
            (TypedExprKind::Var(name.clone()), stored_type.0.clone())
        }

        ExprKind::Constant(c @ Const::Int(_)) => (TypedExprKind::Constant(c.clone()), CType::Int),
        ExprKind::Constant(c @ Const::Long(_)) => (TypedExprKind::Constant(c.clone()), CType::Long),
        ExprKind::Unary(op, expr) => {
            let inner = typecheck_expr(expr, ctx)?;
            let inner_ty = inner.get_type(); // todo: make sure we only clone when we need
            let expr = TypedExprKind::Unary(*op, Box::new(inner));
            let ty = if *op == UnaryOp::Not {
                CType::Int
            } else {
                inner_ty
            };
            (expr, ty)
        }
        ExprKind::Binary(o @ BinaryOp::BinAnd, lhs, rhs)
        | ExprKind::Binary(o @ BinaryOp::BinOr, lhs, rhs) => {
            let e1 = typecheck_expr(lhs, ctx)?;
            let e2 = typecheck_expr(rhs, ctx)?;
            // AND and OR operations convert to an integer
            let ty = CType::Int;
            (TypedExprKind::Binary(*o, Box::new(e1), Box::new(e2)), ty)
        }
        ExprKind::Binary(op, lhs, rhs) => {
            let e1 = typecheck_expr(lhs, ctx)?;
            let e2 = typecheck_expr(rhs, ctx)?;
            let t1 = e1.get_type();
            let t2 = e2.get_type();
            let common_type = CType::get_common_type(t1, t2);
            // we now need to see if each expression needs to be potentially
            // cast to the common type
            let converted_e1 = convert_to(e1, common_type.clone());
            let converted_e2 = convert_to(e2, common_type.clone());
            let bin_expr =
                TypedExprKind::Binary(*op, Box::new(converted_e1), Box::new(converted_e2));
            // some binops return an int
            let t = match op {
                BinaryOp::Add
                | BinaryOp::Subtract
                | BinaryOp::Multiply
                | BinaryOp::Divide
                | BinaryOp::Remainder => common_type,
                _ => CType::Int,
            };
            (bin_expr, t)
        }
        ExprKind::Assignment(lhs, rhs) => {
            let e1 = typecheck_expr(lhs, ctx)?;
            let e2 = typecheck_expr(rhs, ctx)?;
            let ty = e1.get_type();
            let converted_rhs = convert_to(e2, ty.clone());
            (
                TypedExprKind::Assignment(Box::new(e1), Box::new(converted_rhs)),
                ty,
            )
        }
        ExprKind::Conditional {
            condition,
            then,
            else_,
        } => {
            let cond = typecheck_expr(condition, ctx)?;
            let e1 = typecheck_expr(then, ctx)?;
            let e2 = typecheck_expr(else_, ctx)?;
            let t1 = e1.get_type();
            let t2 = e2.get_type();
            let common_type = CType::get_common_type(t1, t2);
            // we now need to see if each expression needs to be potentially
            // cast to the common type
            let converted_e1 = convert_to(e1, common_type.clone());
            let converted_e2 = convert_to(e2, common_type.clone());
            (
                TypedExprKind::Conditional {
                    condition: Box::new(cond),
                    then: Box::new(converted_e1),
                    else_: Box::new(converted_e2),
                },
                common_type,
            )
        }

        ExprKind::Cast(ty, expr) => {
            let expr = typecheck_expr(expr, ctx)?;
            (TypedExprKind::Cast(ty.clone(), Box::new(expr)), ty.clone())
        }
        ExprKind::FunctionCall { name, args } => {
            // stored type return type is our canonical return type.
            // We'll typecheck each argument and potentially cast to
            // the param types.
            let stored_type = {
                ctx.symbol_table
                    .get(name)
                    .expect("Didn't find type for function call name")
                    .0
                    .clone()
            };
            let CType::FunType { params, ret } = stored_type else {
                return Err(SemanticAnalysisError::VariableUsedAsFunctionName(
                    name.clone(),
                ));
            };
            if args.len() != params.len() {
                return Err(SemanticAnalysisError::FunctionCalledWithWrongNumOfArgs);
            }
            let mut converted_args = vec![];
            for (arg, param_ty) in args.iter().zip(params) {
                let expr = typecheck_expr(arg, ctx)?;
                converted_args.push(convert_to(expr, param_ty.clone()));
            }
            (
                TypedExprKind::FunctionCall {
                    name: name.clone(),
                    args: converted_args,
                },
                ret.as_ref().clone(),
            )
        }
    };
    Ok(TypedExpression {
        kind: Box::new(expr_kind),
        ty: expr_type,
    })
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
    let AST::Program(decls) = ast;

    for decl in decls {
        match decl {
            Declaration::FunDecl(function) => resolve_function_declaration(function, resolver)?,
            Declaration::VarDecl(variable) => {
                resolve_file_scope_variable_declaration(variable, resolver)
            }
        }
    }
    Ok(())
}

fn resolve_block(block: &mut Block, resolver: &mut Resolver) -> Result<(), SemanticAnalysisError> {
    let crate::ast::Block(body) = block;
    for body_item in body.iter_mut() {
        match body_item {
            BlockItem::Decl(Declaration::VarDecl(declaration)) => {
                resolve_local_variable_declaration(declaration, resolver)?
            }
            BlockItem::Decl(Declaration::FunDecl(decl)) => {
                if decl.block.is_some() {
                    // Nested function definitions are not permitted
                    return Err(SemanticAnalysisError::NestedFunctionDefinitionsNotAllowed);
                }
                if let Some(StorageClass::Static) = decl.storage_class {
                    return Err(SemanticAnalysisError::NoStaticOnBlockScopeFuncs);
                }
                resolve_function_declaration(decl, resolver)?;
            }
            BlockItem::Stmt(statement) => resolve_statement(statement, resolver)?,
        }
    }

    Ok(())
}

fn resolve_file_scope_variable_declaration(decl: &VariableDeclaration, resolver: &mut Resolver) {
    resolver.resolve_variable(&decl.name, decl.name.clone(), true, true)
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

fn resolve_local_variable_declaration(
    decl: &mut VariableDeclaration,
    resolver: &mut Resolver,
) -> Result<(), SemanticAnalysisError> {
    let VariableDeclaration {
        name,
        init,
        storage_class,
        ..
    } = decl;
    // first, check for conflicting entries with file-scoped variable
    if let Some(ResolvedIdentifier {
        in_current_scope,
        has_linkage,
        ..
    }) = resolver.get_identifier(&name)
    {
        if *in_current_scope
            && !(*has_linkage && matches!(storage_class, Some(StorageClass::Extern)))
        {
            return Err(SemanticAnalysisError::ConflictingLocalDeclarations);
        }
    }
    // if its an extern variable, we should just add to resolver without
    // using a temporary in the named position: it needs to link across files.
    // C does not allow initializers in this position.
    if let Some(StorageClass::Extern) = decl.storage_class {
        resolver.resolve_variable(&decl.name, decl.name.clone(), true, true);
    } else {
        resolve_param(name, resolver)?;
        if let Some(init) = init {
            resolve_expr(init, resolver)?;
        };
    }
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
        ForInit::InitDecl(decl) => {
            if decl.storage_class.is_some() {
                return Err(SemanticAnalysisError::StorageClassInForInit);
            }
            resolve_local_variable_declaration(decl, resolver)
        }
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
    match &mut *expr.kind {
        ExprKind::Constant(_) => (),
        ExprKind::Var(ident) => match resolver.get_resolved_variable(&ident) {
            Some(renamed) => *ident = renamed,
            None => {
                return Err(SemanticAnalysisError::UndeclaredVariableInInitializer(
                    ident.clone(),
                ));
            }
        },
        ExprKind::Unary(_op, expr) => resolve_expr(expr, resolver)?,
        ExprKind::Binary(_op, lhs, rhs) => {
            resolve_expr(&mut *lhs, resolver)?;
            resolve_expr(&mut *rhs, resolver)?;
        }
        ExprKind::Assignment(lhs, rhs) => {
            if !matches!(*lhs.kind, ExprKind::Var(_)) {
                return Err(SemanticAnalysisError::InvalidLhsAssignmentNode);
            }
            resolve_expr(&mut *lhs, resolver)?;
            resolve_expr(&mut *rhs, resolver)?;
        }
        ExprKind::Conditional {
            condition,
            then,
            else_,
        } => {
            resolve_expr(&mut *condition, resolver)?;
            resolve_expr(&mut *then, resolver)?;
            resolve_expr(&mut *else_, resolver)?;
        }
        ExprKind::FunctionCall { name, args } => {
            let ident = resolver
                .get_identifier(name)
                .ok_or_else(|| SemanticAnalysisError::UndeclaredFunction)?;
            // we reset the name here, just for typechecking
            *name = ident.name.clone();
            for arg in args.iter_mut() {
                resolve_expr(arg, resolver)?;
            }
        }
        ExprKind::Cast(_, expr) => resolve_expr(expr, resolver)?,
    };
    Ok(())
}

fn label_and_validate_loop_constructs(
    ast: &mut AST,
    resolver: &mut Resolver,
) -> Result<(), SemanticAnalysisError> {
    let AST::Program(decls) = ast;

    for decl in decls {
        let Declaration::FunDecl(function) = decl else {
            continue;
        };
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
    let crate::ast::Block(body) = block;
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
    use crate::ast::Statement::*;
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

fn convert_to(e: TypedExpression, t: CType) -> TypedExpression {
    if e.get_type() == t {
        return e;
    }
    TypedExpression {
        ty: t.clone(),
        kind: Box::new(TypedExprKind::Cast(t, Box::new(e))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::AST;
    use crate::parser::BinaryOp;
    use crate::parser::BlockItem;
    use crate::parser::Expression;
    use crate::parser::FunctionDeclaration;
    use crate::parser::Statement;
    use crate::types::CType;

    #[test]
    fn basic_resolve_repeated_variable() {
        let mut before = AST::Program(vec![Declaration::FunDecl(FunctionDeclaration {
            name: "main".into(),
            block: Some(crate::ast::Block(vec![
                BlockItem::Decl(Declaration::VarDecl(VariableDeclaration {
                    name: "a".into(),
                    init: Some(Expression::new(ExprKind::Constant(Const::Int(1)))),
                    storage_class: None,
                    vtype: CType::Int,
                })),
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
            params: vec![],
            storage_class: None,
            ftype: CType::FunType {
                params: vec![],
                ret: Box::new(CType::Int),
            },
        })]);

        let actual = resolve(&mut before).map(|(t, _)| t);
        assert!(actual.is_err());
    }

    #[test]
    fn basic_resolve_undeclared_variable() {
        let mut before = AST::Program(vec![Declaration::FunDecl(FunctionDeclaration {
            name: "main".into(),
            block: Some(crate::ast::Block(vec![
                BlockItem::Decl(Declaration::VarDecl(VariableDeclaration {
                    name: "a".into(),
                    init: Some(Expression::new(ExprKind::Var("c".into()))),
                    storage_class: None,
                    vtype: CType::Int,
                })),
                BlockItem::Stmt(Statement::Return(Expression::new(ExprKind::Var(
                    "a".into(),
                )))),
            ])),
            params: vec![],
            storage_class: None,
            ftype: CType::FunType {
                params: vec![],
                ret: Box::new(CType::Int),
            },
        })]);

        let actual = resolve(&mut before).map(|(t, _)| t);
        assert!(actual.is_err());
    }

    #[test]
    fn basic_resolve_successful() {
        let mut before = AST::Program(vec![Declaration::FunDecl(FunctionDeclaration {
            name: "main".into(),
            block: Some(crate::ast::Block(vec![
                BlockItem::Decl(Declaration::VarDecl(VariableDeclaration {
                    name: "a".into(),
                    init: Some(Expression::new(ExprKind::Constant(Const::Int(1)))),
                    storage_class: None,
                    vtype: CType::Int,
                })),
                BlockItem::Decl(Declaration::VarDecl(VariableDeclaration {
                    name: "b".into(),
                    init: Some(Expression::new(ExprKind::Var("a".into()))),
                    storage_class: None,
                    vtype: CType::Int,
                })),
                BlockItem::Stmt(Statement::Return(Expression::new(ExprKind::Var(
                    "a".into(),
                )))),
            ])),
            params: vec![],
            storage_class: None,
            ftype: CType::FunType {
                params: vec![],
                ret: Box::new(CType::Int),
            },
        })]);

        let actual = resolve(&mut before).map(|(t, _)| t);
        assert!(actual.is_ok());
        let expected = AST::Program(vec![Declaration::FunDecl(FunctionDeclaration {
            name: "main".into(),
            block: Some(crate::ast::Block(vec![
                BlockItem::Decl(Declaration::VarDecl(VariableDeclaration {
                    name: "a.0.decl".into(),
                    init: Some(Expression::new(ExprKind::Constant(Const::Int(1)))),
                    storage_class: None,
                    vtype: CType::Int,
                })),
                BlockItem::Decl(Declaration::VarDecl(VariableDeclaration {
                    name: "b.1.decl".into(),
                    init: Some(Expression::new(ExprKind::Var("a.0.decl".into()))),
                    storage_class: None,
                    vtype: CType::Int,
                })),
                BlockItem::Stmt(Statement::Return(Expression::new(ExprKind::Var(
                    "a.0.decl".into(),
                )))),
            ])),
            params: vec![],
            storage_class: None,
            ftype: CType::FunType {
                params: vec![],
                ret: Box::new(CType::Int),
            },
        })]);

        // mutates taken value
        assert_eq!(before, expected);
    }

    #[test]
    fn complex_resolve_successful() {
        let mut before = AST::Program(vec![Declaration::FunDecl(FunctionDeclaration {
            name: "main".into(),
            block: Some(crate::ast::Block(vec![
                BlockItem::Decl(Declaration::VarDecl(VariableDeclaration {
                    name: "a".into(),
                    init: Some(Expression::new(ExprKind::Constant(Const::Int(1)))),
                    storage_class: None,
                    vtype: CType::Int,
                })),
                BlockItem::Decl(Declaration::VarDecl(VariableDeclaration {
                    name: "b".into(),
                    init: Some(Expression::new(ExprKind::Var("a".into()))),
                    storage_class: None,
                    vtype: CType::Int,
                })),
                BlockItem::Decl(Declaration::VarDecl(VariableDeclaration {
                    name: "c".into(),
                    init: Some(Expression::new(ExprKind::Binary(
                        BinaryOp::Add,
                        Box::new(Expression::new(ExprKind::Var("a".into()))),
                        Box::new(Expression::new(ExprKind::Var("b".into()))),
                    ))),
                    storage_class: None,
                    vtype: CType::Int,
                })),
                BlockItem::Stmt(Statement::Return(Expression::new(ExprKind::Var(
                    "c".into(),
                )))),
            ])),
            params: vec![],
            storage_class: None,
            ftype: CType::FunType {
                params: vec![],
                ret: Box::new(CType::Int),
            },
        })]);

        let actual = resolve(&mut before).map(|(t, _)| t);
        assert!(actual.is_ok());
        let expected = AST::Program(vec![Declaration::FunDecl(FunctionDeclaration {
            name: "main".into(),
            block: Some(crate::ast::Block(vec![
                BlockItem::Decl(Declaration::VarDecl(VariableDeclaration {
                    name: "a.0.decl".into(),
                    init: Some(Expression::new(ExprKind::Constant(Const::Int(1)))),
                    storage_class: None,
                    vtype: CType::Int,
                })),
                BlockItem::Decl(Declaration::VarDecl(VariableDeclaration {
                    name: "b.1.decl".into(),
                    init: Some(Expression::new(ExprKind::Var("a.0.decl".into()))),
                    storage_class: None,
                    vtype: CType::Int,
                })),
                BlockItem::Decl(Declaration::VarDecl(VariableDeclaration {
                    name: "c.2.decl".into(),
                    init: Some(Expression::new(ExprKind::Binary(
                        BinaryOp::Add,
                        Box::new(Expression::new(ExprKind::Var("a.0.decl".into()))),
                        Box::new(Expression::new(ExprKind::Var("b.1.decl".into()))),
                    ))),
                    storage_class: None,
                    vtype: CType::Int,
                })),
                BlockItem::Stmt(Statement::Return(Expression::new(ExprKind::Var(
                    "c.2.decl".into(),
                )))),
            ])),
            params: vec![],
            storage_class: None,
            ftype: CType::FunType {
                params: vec![],
                ret: Box::new(CType::Int),
            },
        })]);

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
        let analysis = resolve(&mut ast).map(|(t, _)| t);
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
        let analysis = resolve(&mut ast).map(|(t, _)| t);
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
        let analysis = resolve(&mut ast).map(|(t, _)| t);
        assert!(analysis.is_err());
        assert_eq!(
            analysis,
            Err(SemanticAnalysisError::ConflictingLocalDeclarations)
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
        let analysis = resolve(&mut ast).map(|(t, _)| t);
        assert!(analysis.is_ok());

        let expected = AST::Program(vec![Declaration::FunDecl(FunctionDeclaration {
            name: "main".into(),
            block: Some(crate::ast::Block(vec![
                BlockItem::Decl(Declaration::VarDecl(VariableDeclaration {
                    name: "x.0.decl".into(),
                    init: Some(Expression::new(ExprKind::Constant(Const::Int(1)))),
                    storage_class: None,
                    vtype: CType::Int,
                })),
                BlockItem::Stmt(Statement::Compound(crate::ast::Block(vec![
                    BlockItem::Decl(
                        // Declaration is the same name, but has a new renamed variable
                        // since we want to ensure any label is pinned to one value
                        Declaration::VarDecl(VariableDeclaration {
                            name: "x.1.decl".into(),
                            init: Some(Expression::new(ExprKind::Constant(Const::Int(2)))),
                            storage_class: None,
                            vtype: CType::Int,
                        }),
                    ),
                ]))),
                BlockItem::Stmt(Statement::Return(Expression::new(ExprKind::Var(
                    "x.0.decl".into(),
                )))),
            ])),
            params: vec![],
            storage_class: None,
            ftype: CType::FunType {
                params: vec![],
                ret: Box::new(CType::Int),
            },
        })]);

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
        let analysis = resolve(&mut ast).map(|(t, _)| t);
        assert!(analysis.is_ok());

        let expected = AST::Program(vec![Declaration::FunDecl(FunctionDeclaration {
            name: "main".into(),
            block: Some(crate::ast::Block(vec![
                BlockItem::Decl(Declaration::VarDecl(VariableDeclaration {
                    name: "a.0.decl".into(),
                    init: None,
                    storage_class: None,
                    vtype: CType::Int,
                })),
                BlockItem::Stmt(Statement::For {
                    init: ForInit::InitExp(Some(Expression::new(ExprKind::Assignment(
                        Box::new(Expression::new(ExprKind::Var("a.0.decl".into()))),
                        Box::new(Expression::new(ExprKind::Constant(Const::Int(1)))),
                    )))),
                    condition: Some(Expression::new(ExprKind::Binary(
                        BinaryOp::LessThan,
                        Box::new(Expression::new(ExprKind::Var("a.0.decl".into()))),
                        Box::new(Expression::new(ExprKind::Constant(Const::Int(10)))),
                    ))),
                    post: Some(Expression::new(ExprKind::Assignment(
                        Box::new(Expression::new(ExprKind::Var("a.0.decl".into()))),
                        Box::new(Expression::new(ExprKind::Binary(
                            BinaryOp::Add,
                            Box::new(Expression::new(ExprKind::Var("a.0.decl".into()))),
                            Box::new(Expression::new(ExprKind::Constant(Const::Int(1)))),
                        ))),
                    ))),
                    body: Box::new(Statement::Compound(crate::ast::Block(vec![
                        BlockItem::Stmt(Statement::Continue("for_label.1".into())),
                    ]))),
                    label: "for_label.1".into(),
                }),
                BlockItem::Stmt(Statement::DoWhile {
                    body: Box::new(Statement::Compound(crate::ast::Block(vec![
                        BlockItem::Stmt(Statement::Continue("do_while_label.2".into())),
                    ]))),
                    condition: Expression::new(ExprKind::Binary(
                        BinaryOp::LessThan,
                        Box::new(Expression::new(ExprKind::Var("a.0.decl".into()))),
                        Box::new(Expression::new(ExprKind::Constant(Const::Int(0)))),
                    )),
                    label: "do_while_label.2".into(),
                }),
                BlockItem::Stmt(Statement::While {
                    condition: Expression::new(ExprKind::Binary(
                        BinaryOp::GreaterThan,
                        Box::new(Expression::new(ExprKind::Var("a.0.decl".into()))),
                        Box::new(Expression::new(ExprKind::Constant(Const::Int(0)))),
                    )),
                    body: Box::new(Statement::Break("while_label.3".into())),
                    label: "while_label.3".into(),
                }),
            ])),
            params: vec![],
            storage_class: None,
            ftype: CType::FunType {
                params: vec![],
                ret: Box::new(CType::Int),
            },
        })]);
        assert_eq!(ast, expected);
    }

    fn resolve_src(src: &str) -> Result<(SymbolTable, TypedAST), SemanticAnalysisError> {
        let lexer = crate::lexer::Lexer::lex(src).unwrap();
        let tokens = lexer.as_syntactic_tokens();
        let parse = crate::parser::Parser::new(&tokens);
        let mut ast = parse.into_ast().unwrap();
        resolve(&mut ast)
    }

    #[test]
    fn incompatible_function_redeclaration_different_param_types() {
        let src = r#"
            int foo(int a);
            int foo(long a);
            int main(void) { return 0; }
        "#;
        assert_eq!(
            resolve_src(src),
            Err(SemanticAnalysisError::IncompatibleFunctionDeclaration)
        );
    }

    #[test]
    fn non_constant_initializer_on_local_static() {
        let src = r#"
            int main(void) {
                int a = 1;
                static int x = a + 1;
                return x;
            }
        "#;
        assert_eq!(
            resolve_src(src),
            Err(SemanticAnalysisError::NonConstInitOnLocalStaticVar)
        );
    }

    #[test]
    fn extern_variable_with_initializer_at_block_scope() {
        let src = r#"
            int main(void) {
                extern int x = 5;
                return x;
            }
        "#;
        assert_eq!(
            resolve_src(src),
            Err(SemanticAnalysisError::InitializerOnLocalExternVarDecl)
        );
    }

    #[test]
    fn variable_used_as_function_name() {
        let src = r#"
            int main(void) {
                int foo;
                return foo();
            }
        "#;
        assert_eq!(
            resolve_src(src),
            Err(SemanticAnalysisError::VariableUsedAsFunctionName(
                "foo.0.decl".into()
            ))
        );
    }

    #[test]
    fn function_used_as_variable_name() {
        let src = r#"
            int foo(int a);
            int main(void) {
                return foo + 1;
            }
        "#;
        assert_eq!(
            resolve_src(src),
            Err(SemanticAnalysisError::FunctionUsedAsVariableName(
                "foo".into()
            ))
        );
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
        let analysis = resolve(&mut ast).map(|(t, _)| t);
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
        let analysis = resolve(&mut ast).map(|(t, _)| t);
        assert!(analysis.is_err());
    }
}
