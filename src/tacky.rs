// implements a AST -> TACKY AST for the IR
// Mostly copied from asm.rs

use crate::parser::BinaryOp as ParserBinaryOp;
use crate::parser::Const;
use crate::parser::UnaryOp as ParserUnaryOp;
use crate::semantic_analysis;
use crate::semantic_analysis::IdentifierAttrs;
use crate::semantic_analysis::SymbolTable;
use crate::semantic_analysis::TypedAST;
use crate::semantic_analysis::TypedBlock;
use crate::semantic_analysis::TypedBlockItem;
use crate::semantic_analysis::TypedDeclaration;
use crate::semantic_analysis::TypedExprKind;
use crate::semantic_analysis::TypedExpression;
use crate::semantic_analysis::TypedForInit;
use crate::semantic_analysis::TypedFunctionDeclaration;
use crate::semantic_analysis::TypedStatement;
use crate::semantic_analysis::TypedVariableDeclaration;
use crate::types::CType;
use crate::types::StaticInit;
use thiserror::Error;

#[derive(Debug, PartialEq, Error)]
pub enum TackyError {
    #[error("Found non-variable on lefthand side of assignment")]
    InvalidLhsOfAssignment,
}

#[derive(Debug, PartialEq)]
pub enum AST {
    Program(Vec<TopLevel>),
}

#[derive(Debug, PartialEq)]
pub enum TopLevel {
    Function {
        name: String,
        params: Vec<String>,
        global: bool,
        instructions: Vec<Instruction>,
    },
    StaticVariable {
        identifier: String,
        global: bool,
        t: CType,
        init: StaticInit,
    },
}

#[derive(Debug, PartialEq)]
pub enum Instruction {
    Ret(Val),
    Unary {
        op: UnaryOp,
        src: Val,
        dst: Val,
    },
    Binary {
        op: BinaryOp,
        src1: Val,
        src2: Val,
        dst: Val,
    },
    Copy {
        src: Val,
        dst: Val,
    },
    Jump(String), // identifier of label
    JumpIfZero {
        cond: Val,
        target: String,
    },
    JumpIfNotZero {
        cond: Val,
        target: String,
    },
    Label(String), // identifier
    FunCall {
        name: String,
        args: Vec<Val>,
        dst: Val,
    },
    // conversions
    SignExtend {
        src: Val,
        dst: Val,
    },
    Truncate {
        src: Val,
        dst: Val,
    },
}

#[derive(Debug, PartialEq, Clone)]
pub enum Val {
    Constant(Const),
    Var(String), // temporary variable name
}

#[derive(Debug, PartialEq)]
pub enum UnaryOp {
    Complement,
    Negate,
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
    // Logical and relational operators
    And,
    Or,
    Equal,
    NotEqual,
    LessThan,
    LessOrEqual,
    GreaterThan,
    GreaterOrEqual,
}

struct TackyCtx<'a> {
    symbol_table: &'a mut SymbolTable,
    instructions: Vec<Instruction>,
}

impl<'a> TackyCtx<'a> {
    fn new(symbol_table: &'a mut SymbolTable) -> Self {
        Self {
            symbol_table,
            instructions: vec![],
        }
    }

    fn push(&mut self, instr: Instruction) {
        self.instructions.push(instr);
    }

    fn take_instructions(&mut self) -> Vec<Instruction> {
        std::mem::take(&mut self.instructions)
    }
}

#[derive(Debug, PartialEq)]
pub struct Tacky {
    parser: TypedAST,
    dst_counter: u16,
    label_counter: u16,
}

impl<'a> Tacky {
    pub fn new(parser: TypedAST) -> Self {
        Self {
            parser,
            dst_counter: 0,
            label_counter: 0,
        }
    }

    pub fn into_ast(
        mut self,
        symbol_table: &mut semantic_analysis::SymbolTable,
    ) -> Result<AST, TackyError> {
        let parser = std::mem::replace(&mut self.parser, TypedAST::Program(vec![]));
        let mut ast = self.parse_program(parser, symbol_table)?;
        let defs = Self::convert_symbols_to_tacky_defs(symbol_table);
        let AST::Program(ref mut ins) = ast;
        ins.extend(defs);
        Ok(ast)
    }

    fn parse_program(
        &mut self,
        parser: TypedAST,
        symbol_table: &mut SymbolTable,
    ) -> Result<AST, TackyError> {
        let TypedAST::Program(decls) = parser;
        let funcs = decls
            .iter()
            .filter(|decl| match decl {
                TypedDeclaration::FunDecl(fun) => fun.block.is_some(),
                TypedDeclaration::VarDecl(_) => false,
            })
            .map(|decl| {
                let TypedDeclaration::FunDecl(fun) = decl else {
                    panic!();
                };
                self.parse_function(fun, symbol_table)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(AST::Program(funcs))
    }

    fn convert_symbols_to_tacky_defs(symbol_table: &SymbolTable) -> Vec<TopLevel> {
        use crate::semantic_analysis::{IdentifierAttrs, InitialValue};
        let mut defs = vec![];
        for (name, entry) in symbol_table {
            if let (ctype, IdentifierAttrs::StaticAttr { init, global }) = entry {
                match init {
                    InitialValue::Initial(i) => defs.push(TopLevel::StaticVariable {
                        identifier: name.clone(),
                        global: *global,
                        init: *i,
                        t: ctype.clone(),
                    }),
                    InitialValue::Tentative => {
                        let init = match ctype {
                            CType::FunType { .. } => {
                                unreachable!("Cannot have a static variable of fun type ctype")
                            }
                            CType::Int => StaticInit::IntInit(0),
                            CType::Long => StaticInit::LongInit(0),
                            CType::ULong | CType::UInt => todo!(),
                        };
                        defs.push(TopLevel::StaticVariable {
                            identifier: name.clone(),
                            global: *global,
                            init,
                            t: ctype.clone(),
                        });
                    }
                    _ => {}
                }
            }
        }
        defs
    }

    fn parse_function(
        &mut self,
        function: &TypedFunctionDeclaration,
        symbol_table: &mut SymbolTable,
    ) -> Result<TopLevel, TackyError> {
        let TypedFunctionDeclaration {
            name,
            block,
            params,
            ..
        } = function;
        let Some(block) = block else {
            panic!("Somehow got a None block in parse_function")
        };
        let mut ctx = TackyCtx::new(symbol_table);
        self.parse_instructions(block, &mut ctx)?;
        let instructions = ctx.take_instructions();
        let (CType::FunType { .. }, semantic_analysis::IdentifierAttrs::FunAttr { global, .. }) =
            ctx.symbol_table
                .get(name)
                .expect("Expected to find a function named {name} in symbol table")
        else {
            panic!(
                "Unexpected types in symbol table, got {:?}",
                ctx.symbol_table.get(name)
            );
        };
        Ok(TopLevel::Function {
            name: name.into(),
            instructions,
            params: params.clone(),
            global: *global,
        })
    }

    fn parse_expression(
        &mut self,
        expr: &TypedExpression,
        ctx: &mut TackyCtx,
    ) -> Result<Val, TackyError> {
        let ty = expr.get_type();
        match expr.kind.as_ref() {
            TypedExprKind::Constant(c) => Ok(Val::Constant(*c)),
            TypedExprKind::Unary(op, exp) => {
                let src = self.parse_expression(exp, ctx)?;
                let dst = self.make_tacky_variable(ctx, exp.get_type())?;
                let unary_op = match op {
                    ParserUnaryOp::Negate => UnaryOp::Negate,
                    ParserUnaryOp::Complement => UnaryOp::Complement,
                    ParserUnaryOp::Not => UnaryOp::Not,
                };
                ctx.push(Instruction::Unary {
                    op: unary_op,
                    src,
                    dst: dst.clone(),
                });
                Ok(dst)
            }
            TypedExprKind::Binary(op @ ParserBinaryOp::BinAnd, left, right)
            | TypedExprKind::Binary(op @ ParserBinaryOp::BinOr, left, right) => {
                self.parse_short_circuit_expression(op, left, right, ctx)
            }
            TypedExprKind::Binary(op @ ParserBinaryOp::AddAssign, left, right)
            | TypedExprKind::Binary(op @ ParserBinaryOp::MinusAssign, left, right)
            | TypedExprKind::Binary(op @ ParserBinaryOp::MultiplyAssign, left, right)
            | TypedExprKind::Binary(op @ ParserBinaryOp::DivideAssign, left, right)
            | TypedExprKind::Binary(op @ ParserBinaryOp::RemainderAssign, left, right)
            | TypedExprKind::Binary(op @ ParserBinaryOp::BitwiseAndAssign, left, right)
            | TypedExprKind::Binary(op @ ParserBinaryOp::BitwiseOrAssign, left, right)
            | TypedExprKind::Binary(op @ ParserBinaryOp::XorAssign, left, right)
            | TypedExprKind::Binary(op @ ParserBinaryOp::ShiftLeftAssign, left, right)
            | TypedExprKind::Binary(op @ ParserBinaryOp::ShiftRightAssign, left, right) => {
                self.parse_eager_compound_binary_expression(op, left, right, ctx)
            }
            TypedExprKind::Binary(op, left, right) => {
                // passing ty in here since the outer ty COULD be Int
                // if we're in relational ops, so its just clumsy
                self.parse_eager_binary_expression(op, left, right, ctx, ty)
            }
            TypedExprKind::Var(ident) => Ok(Val::Var(ident.clone())),
            TypedExprKind::Assignment(lhs, rhs) => {
                let TypedExprKind::Var(ref ident) = *lhs.kind else {
                    return Err(TackyError::InvalidLhsOfAssignment);
                };
                // emit instructions for rhs, then copy into lhs
                let result = self.parse_expression(rhs, ctx)?;
                ctx.push(Instruction::Copy {
                    src: result,
                    dst: Val::Var(ident.clone()),
                });
                Ok(Val::Var(ident.clone()))
            }
            TypedExprKind::Conditional {
                condition,
                then,
                else_,
            } => self.parse_conditional(condition, then, else_, ctx),
            TypedExprKind::FunctionCall { name, args } => {
                let args = args
                    .iter()
                    .map(|arg| self.parse_expression(arg, ctx))
                    .collect::<Result<Vec<_>, _>>()?;
                let dst = self.make_tacky_variable(ctx, ty)?;

                ctx.push(Instruction::FunCall {
                    name: name.clone(),
                    args,
                    dst: dst.clone(),
                });
                Ok(dst)
            }
            TypedExprKind::Cast(t, expr) => {
                let result = self.parse_expression(expr, ctx)?;
                // if we're casting to the same type, no need to emit extra instructions
                if *t == expr.get_type() {
                    return Ok(result);
                };
                let dst = self.make_tacky_variable(ctx, t.clone())?;
                let extension_instruction = match t {
                    CType::Long => Instruction::SignExtend {
                        src: result,
                        dst: dst.clone(),
                    },
                    CType::Int => Instruction::Truncate {
                        src: result,
                        dst: dst.clone(),
                    },
                    _ => unreachable!("Tried to extend a type that isn't int or long"),
                };
                ctx.push(extension_instruction);
                Ok(dst)
            }
        }
    }

    fn parse_short_circuit_expression(
        &mut self,
        op: &ParserBinaryOp,
        left: &TypedExpression,
        right: &TypedExpression,
        ctx: &mut TackyCtx,
    ) -> Result<Val, TackyError> {
        use ParserBinaryOp as PBO;
        match op {
            PBO::BinAnd => self.parse_short_circuit_and_expression(BinaryOp::And, left, right, ctx),
            PBO::BinOr => self.parse_short_circuit_or_expression(BinaryOp::Or, left, right, ctx),
            _ => unreachable!(),
        }
    }

    fn parse_conditional(
        &mut self,
        condition: &TypedExpression,
        then: &TypedExpression,
        else_: &TypedExpression,
        ctx: &mut TackyCtx,
    ) -> Result<Val, TackyError> {
        let else_label = self.make_label("else_label");
        let end_label = self.make_label("end_label");
        let cond = self.parse_expression(condition, ctx)?;
        // move cond into a tmp
        let dst1 = self.make_tacky_variable(ctx, condition.get_type())?;

        ctx.push(Instruction::Copy {
            src: cond,
            dst: dst1.clone(),
        });
        ctx.push(Instruction::JumpIfZero {
            cond: dst1.clone(),
            target: else_label.clone(),
        });
        // temporary for result: We'll copy the then/else logic into it and return
        // result at the end
        let result = self.make_tacky_variable(ctx, then.get_type())?;

        let v1 = self.parse_expression(then, ctx)?;
        ctx.push(Instruction::Copy {
            src: v1,
            dst: result.clone(),
        });

        ctx.push(Instruction::Jump(end_label.clone()));
        ctx.push(Instruction::Label(else_label));
        let v2 = self.parse_expression(else_, ctx)?;
        ctx.push(Instruction::Copy {
            src: v2,
            dst: result.clone(),
        });

        ctx.push(Instruction::Label(end_label));

        Ok(result)
    }

    // short circuiting means that we first
    //   eval left
    //   JumpIfZero to bailout label
    //   eval right
    //   JumpIfZero to bailout label
    //     copy "true" into return variable
    //   Jump to END
    //   BAILOUT label
    //     copy "false" into return variable
    //   END label
    fn parse_short_circuit_and_expression(
        &mut self,
        op: BinaryOp,
        left: &TypedExpression,
        right: &TypedExpression,
        ctx: &mut TackyCtx,
    ) -> Result<Val, TackyError> {
        assert_eq!(
            op,
            BinaryOp::And,
            "Expected BinaryOp::And when parsing short circuit expr"
        );
        let jump_label = self.make_label("and_expr_false");
        let end_label = self.make_label("and_expr_end");
        let src1 = self.parse_expression(left, ctx)?;
        // move src1 into a tmp
        let dst1 = self.make_tacky_variable(ctx, CType::Int)?;
        ctx.push(Instruction::Copy {
            src: src1,
            dst: dst1.clone(),
        });
        ctx.push(Instruction::JumpIfZero {
            cond: dst1,
            target: jump_label.clone(),
        });
        let src2 = self.parse_expression(right, ctx)?;
        // move src2 into a tmp
        let dst2 = self.make_tacky_variable(ctx, CType::Int)?;
        ctx.push(Instruction::Copy {
            src: src2,
            dst: dst2.clone(),
        });
        ctx.push(Instruction::JumpIfZero {
            cond: dst2,
            target: jump_label.clone(),
        });

        // at this point, neither arm are false so we
        // define our destination location, set it to true
        // and jump to the end label
        let result = self.make_tacky_variable(ctx, CType::Int)?;
        ctx.push(Instruction::Copy {
            src: Val::Constant(Const::Int(1)),
            dst: result.clone(),
        });
        ctx.push(Instruction::Jump(end_label.clone()));
        // here we create our labels:
        // Create our jump_label label to reach
        // copy False into our result
        // create our end label
        ctx.push(Instruction::Label(jump_label));
        ctx.push(Instruction::Copy {
            src: Val::Constant(Const::Int(0)),
            dst: result.clone(),
        });
        ctx.push(Instruction::Label(end_label));
        Ok(result)
    }

    // short circuiting means that we first
    //   eval left
    //   JumpIfNotZero to bailout label
    //   eval right
    //   JumpIfNotZero to bailout label
    //     copy "false" into return variable
    //   Jump to END
    //   BAILOUT label
    //     copy "true" into return variable
    //   END label
    fn parse_short_circuit_or_expression(
        &mut self,
        op: BinaryOp,
        left: &TypedExpression,
        right: &TypedExpression,
        ctx: &mut TackyCtx,
    ) -> Result<Val, TackyError> {
        assert_eq!(
            op,
            BinaryOp::Or,
            "Expected BinaryOp::Or when parsing short circuit expr"
        );
        let jump_label = self.make_label("or_expr_true");
        let end_label = self.make_label("or_expr_end");
        let src1 = self.parse_expression(left, ctx)?;
        // move src1 into a tmp
        let dst1 = self.make_tacky_variable(ctx, CType::Int)?;
        ctx.push(Instruction::Copy {
            src: src1,
            dst: dst1.clone(),
        });
        ctx.push(Instruction::JumpIfNotZero {
            cond: dst1,
            target: jump_label.clone(),
        });
        let src2 = self.parse_expression(right, ctx)?;
        // move src2 into a tmp
        let dst2 = self.make_tacky_variable(ctx, CType::Int)?;

        ctx.push(Instruction::Copy {
            src: src2,
            dst: dst2.clone(),
        });
        ctx.push(Instruction::JumpIfNotZero {
            cond: dst2,
            target: jump_label.clone(),
        });

        // at this point, neither arm are true so we
        // define our destination location, set it to false
        // and jump to the end label
        let result = self.make_tacky_variable(ctx, CType::Int)?;
        ctx.push(Instruction::Copy {
            src: Val::Constant(Const::Int(0)),
            dst: result.clone(),
        });
        ctx.push(Instruction::Jump(end_label.clone()));
        // here we create our labels:
        // Create our jump_label label to reach
        // copy true into our result
        // create our end label
        ctx.push(Instruction::Label(jump_label));
        ctx.push(Instruction::Copy {
            src: Val::Constant(Const::Int(1)),
            dst: result.clone(),
        });
        ctx.push(Instruction::Label(end_label));
        Ok(result)
    }

    fn parse_if_then(
        &mut self,
        condition: &TypedExpression,
        then: &TypedStatement,
        ctx: &mut TackyCtx,
    ) -> Result<(), TackyError> {
        let label = self.make_label("end_label");
        let cond = self.parse_expression(condition, ctx)?;
        // move cond into a tmp
        let dst1 = self.make_tacky_variable(ctx, condition.get_type())?;
        ctx.push(Instruction::Copy {
            src: cond,
            dst: dst1.clone(),
        });
        ctx.push(Instruction::JumpIfZero {
            cond: dst1.clone(),
            target: label.clone(),
        });
        self.parse_statement(&then, ctx)?;
        ctx.push(Instruction::Label(label));
        Ok(())
    }

    /*
     * evaluate condition
     * if it's false/zero, jump to the else label
     *   evaluate then:
     *   jump to end
     *   else_label:
     *       evaluate else_
     *   end_label:
     */
    fn parse_if_then_else(
        &mut self,
        condition: &TypedExpression,
        then: &TypedStatement,
        else_: &TypedStatement,
        ctx: &mut TackyCtx,
    ) -> Result<(), TackyError> {
        let else_label = self.make_label("else_label");
        let end_label = self.make_label("end_label");
        let cond = self.parse_expression(condition, ctx)?;
        // move cond into a tmp
        let dst1 = self.make_tacky_variable(ctx, condition.get_type())?;
        ctx.push(Instruction::Copy {
            src: cond,
            dst: dst1.clone(),
        });
        ctx.push(Instruction::JumpIfZero {
            cond: dst1.clone(),
            target: else_label.clone(),
        });
        self.parse_statement(&then, ctx)?;
        ctx.push(Instruction::Jump(end_label.clone()));
        ctx.push(Instruction::Label(else_label));
        self.parse_statement(&else_, ctx)?;
        ctx.push(Instruction::Label(end_label));

        Ok(())
    }

    fn parse_eager_compound_binary_expression(
        &mut self,
        op: &ParserBinaryOp,
        left: &TypedExpression,
        right: &TypedExpression,
        ctx: &mut TackyCtx,
    ) -> Result<Val, TackyError> {
        use ParserBinaryOp as PBO;
        let src1 = self.parse_expression(left, ctx)?;
        // if src1 is a temporary, it's an invalid lvalue.
        // We should only allow variables. To do so, we check if we've emitted
        // the identifier as a temporary.
        let Val::Var(ref ident) = src1 else {
            return Err(TackyError::InvalidLhsOfAssignment);
        };
        if self.created_label(ident) {
            return Err(TackyError::InvalidLhsOfAssignment);
        };

        let src2 = self.parse_expression(right, ctx)?;
        let dst = self.make_tacky_variable(ctx, left.get_type())?;
        let binop = match op {
            PBO::AddAssign => BinaryOp::Add,
            PBO::MinusAssign => BinaryOp::Subtract,
            PBO::MultiplyAssign => BinaryOp::Multiply,
            PBO::DivideAssign => BinaryOp::Divide,
            PBO::RemainderAssign => BinaryOp::Remainder,
            PBO::BitwiseOrAssign => BinaryOp::BitwiseOr,
            PBO::BitwiseAndAssign => BinaryOp::BitwiseAnd,
            PBO::XorAssign => BinaryOp::Xor,
            PBO::ShiftLeftAssign => BinaryOp::ShiftLeft,
            PBO::ShiftRightAssign => BinaryOp::ShiftRight,
            _ => unreachable!("Unexpected compound binary operator"),
        };

        ctx.push(Instruction::Binary {
            op: binop,
            src1: src1.clone(),
            src2,
            dst: dst.clone(),
        });
        ctx.push(Instruction::Copy {
            src: dst.clone(),
            dst: src1.clone(),
        });
        Ok(dst)
    }

    fn parse_eager_binary_expression(
        &mut self,
        op: &ParserBinaryOp,
        left: &TypedExpression,
        right: &TypedExpression,
        ctx: &mut TackyCtx,
        ty: CType,
    ) -> Result<Val, TackyError> {
        use ParserBinaryOp as PBO;
        let src1 = self.parse_expression(left, ctx)?;
        let src2 = self.parse_expression(right, ctx)?;
        let dst = self.make_tacky_variable(ctx, ty)?;
        let binop = match op {
            PBO::Add => BinaryOp::Add,
            PBO::Subtract => BinaryOp::Subtract,
            PBO::Multiply => BinaryOp::Multiply,
            PBO::Divide => BinaryOp::Divide,
            PBO::Remainder => BinaryOp::Remainder,
            PBO::BitwiseAnd => BinaryOp::BitwiseAnd,
            PBO::Xor => BinaryOp::Xor,
            PBO::BitwiseOr => BinaryOp::BitwiseOr,
            PBO::ShiftLeft => BinaryOp::ShiftLeft,
            PBO::ShiftRight => BinaryOp::ShiftRight,
            PBO::LessThan => BinaryOp::LessThan,
            PBO::LessOrEqual => BinaryOp::LessOrEqual,
            PBO::GreaterThan => BinaryOp::GreaterThan,
            PBO::GreaterOrEqual => BinaryOp::GreaterOrEqual,
            PBO::Equal => BinaryOp::Equal,
            PBO::NotEqual => BinaryOp::NotEqual,
            PBO::BinAnd | PBO::BinOr => {
                unreachable!("Cannot eagerly parse And or Or binary expressions")
            }
            PBO::AddAssign
            | PBO::MinusAssign
            | PBO::MultiplyAssign
            | PBO::DivideAssign
            | PBO::RemainderAssign
            | PBO::BitwiseAndAssign
            | PBO::BitwiseOrAssign
            | PBO::XorAssign
            | PBO::ShiftLeftAssign
            | PBO::ShiftRightAssign => {
                unreachable!(
                    "We handle compound assignment in parse_eager_compound_binary_expression"
                )
            }
        };
        ctx.push(Instruction::Binary {
            op: binop,
            src1,
            src2,
            dst: dst.clone(),
        });
        Ok(dst)
    }

    fn parse_statement(
        &mut self,
        statement: &TypedStatement,
        ctx: &mut TackyCtx,
    ) -> Result<(), TackyError> {
        match statement {
            TypedStatement::Return(body) => {
                let val = self.parse_expression(body, ctx)?;
                ctx.push(Instruction::Ret(val.clone()));
                Ok(())
            }
            TypedStatement::Null => Ok(()),
            TypedStatement::Expr(expr) => {
                self.parse_expression(expr, ctx)?;
                Ok(())
            }
            TypedStatement::If {
                condition,
                then,
                else_: None,
            } => self.parse_if_then(condition, then.as_ref(), ctx),
            TypedStatement::If {
                condition,
                then,
                else_: Some(else_),
            } => self.parse_if_then_else(condition, then.as_ref(), else_.as_ref(), ctx),
            TypedStatement::Goto(lbl) => {
                ctx.push(Instruction::Jump(lbl.into()));
                Ok(())
            }
            TypedStatement::Labelled { label, statement } => {
                ctx.push(Instruction::Label(label.into()));
                self.parse_statement(statement, ctx)?;
                Ok(())
            }
            TypedStatement::Compound(block) => self.parse_block(block, ctx),
            TypedStatement::Break(label) => {
                ctx.push(Instruction::Jump(create_break_label(&label)));
                Ok(())
            }
            TypedStatement::Continue(label) => {
                ctx.push(Instruction::Jump(create_continue_label(&label)));
                Ok(())
            }
            TypedStatement::DoWhile {
                label,
                body,
                condition,
            } => {
                // push the label first, then we write the instructions for the body.
                // At this point, we write the label for the continue, which
                // ensures if our body had a Continue, we drop to this label.
                // Then we check our condition, and JumpIfNotZero to the start.
                // Then, we write out the label for the Break, so if our
                // body saw a break, we drop out of the do-while construct
                ctx.push(Instruction::Label(label.into()));
                self.parse_statement(body.as_ref(), ctx)?;
                ctx.push(Instruction::Label(create_continue_label(&label)));
                let v = self.parse_expression(condition, ctx)?;
                ctx.push(Instruction::JumpIfNotZero {
                    cond: v,
                    target: label.into(),
                });
                ctx.push(Instruction::Label(create_break_label(&label)));
                Ok(())
            }
            TypedStatement::While {
                label,
                body,
                condition,
            } => {
                // As with do-while, we start by writing out the label.
                // In this case, we'll emit the continue label at the top.
                // This ensures we re-evaluate the condition each time.
                // We emit the condition out, then
                // we JumpIfZero to the end, our break-label. Afterwards
                // we write out the body, jump to the start (our continue label),
                // then emit our break label at the end.
                let cont_label = create_continue_label(&label);
                let break_label = create_break_label(&label);
                ctx.push(Instruction::Label(cont_label.clone()));
                let v = self.parse_expression(condition, ctx)?;
                ctx.push(Instruction::JumpIfZero {
                    cond: v,
                    target: break_label.clone(),
                });
                self.parse_statement(body, ctx)?;
                ctx.push(Instruction::Jump(cont_label));
                ctx.push(Instruction::Label(break_label));
                Ok(())
            }
            TypedStatement::For {
                init,
                condition,
                post,
                body,
                label,
            } => {
                // We write out instructions for our initializer. Then
                // we construct a label to start our loop. We evaluate
                // the condition, and JumpIfZero to the break label at the end.
                // Then we evaluate the body, then write out our Continue label.
                // Then write out the postcondition expression, so our continue
                // hits this (eg. i++). After this, we jump to the start. Then we
                // write out the Break label at the very end.
                // Since we have so many optionals in our For construct, we
                // will occasionally just omit some instructions in the None case.
                match init {
                    TypedForInit::InitDecl(declaration) => {
                        self.emit_declaration(declaration, ctx)?;
                    }
                    TypedForInit::InitExp(Some(expr)) => {
                        self.parse_expression(expr, ctx)?;
                    }
                    TypedForInit::InitExp(None) => (),
                };
                ctx.push(Instruction::Label(label.clone()));
                if let Some(expr) = condition {
                    let v = self.parse_expression(expr, ctx)?;
                    ctx.push(Instruction::JumpIfZero {
                        cond: v,
                        target: create_break_label(&label),
                    });
                };
                self.parse_statement(body.as_ref(), ctx)?;
                ctx.push(Instruction::Label(create_continue_label(&label)));
                if let Some(expr) = post {
                    self.parse_expression(expr, ctx)?;
                };
                ctx.push(Instruction::Jump(label.clone()));
                ctx.push(Instruction::Label(create_break_label(&label)));
                Ok(())
            }
        }
    }

    fn parse_instructions(
        &mut self,
        block: &TypedBlock,
        ctx: &mut TackyCtx,
    ) -> Result<(), TackyError> {
        self.parse_block(block, ctx)?;
        // temporary hack: always add a Return(Constant(0)) Instruction
        // to handle functions that don't end with a return. If we already
        // have a return, it doesn't run.
        ctx.push(Instruction::Ret(Val::Constant(Const::Int(0))));
        Ok(())
    }

    fn parse_block(&mut self, block: &TypedBlock, ctx: &mut TackyCtx) -> Result<(), TackyError> {
        let crate::ast::Block(body) = block;
        for body_item in body {
            match body_item {
                TypedBlockItem::Stmt(stmt) => {
                    self.parse_statement(&stmt, ctx)?;
                }
                TypedBlockItem::Decl(TypedDeclaration::VarDecl(declaration)) => {
                    self.emit_declaration(&declaration, ctx)?;
                }
                TypedBlockItem::Decl(TypedDeclaration::FunDecl(_decl)) => (),
            }
        }
        Ok(())
    }

    fn emit_declaration(
        &mut self,
        decl: &TypedVariableDeclaration,
        ctx: &mut TackyCtx,
    ) -> Result<(), TackyError> {
        if let TypedVariableDeclaration {
            name,
            init: Some(init),
            // file-scope or local-scope with Static or Extern storage is
            // ignored for .data or .bss reads
            storage_class: None,
            ..
        } = decl
        {
            // emit instructions for rhs, then copy into lhs
            let result = self.parse_expression(init, ctx)?;
            ctx.push(Instruction::Copy {
                src: result,
                dst: Val::Var(name.clone()),
            });
        }
        Ok(())
    }

    fn make_temporary(&mut self) -> String {
        let c = self.dst_counter;
        let s = format!("tmp.{c}");
        self.dst_counter = c + 1;
        s
    }

    fn make_tacky_variable(&mut self, ctx: &mut TackyCtx, ty: CType) -> Result<Val, TackyError> {
        let var_name = self.make_temporary();
        ctx.symbol_table
            .insert(var_name.clone(), (ty, IdentifierAttrs::LocalAttr));
        Ok(Val::Var(var_name))
    }

    fn created_label(&self, label: &str) -> bool {
        // if the first four characters are tmp. we have a temporary.
        label.starts_with("tmp.")
    }

    fn make_label(&mut self, base: &str) -> String {
        let c = self.label_counter;
        let s = format!("{base}.{c}");
        self.label_counter = c + 1;
        s
    }
}

fn create_break_label(lbl: &str) -> String {
    format!("break_{lbl}")
}

fn create_continue_label(lbl: &str) -> String {
    format!("continue_{lbl}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn typed_int(kind: TypedExprKind) -> TypedExpression {
        TypedExpression {
            ty: CType::Int,
            kind: Box::new(kind),
        }
    }

    #[allow(dead_code)]
    fn typed_long(kind: TypedExprKind) -> TypedExpression {
        TypedExpression {
            ty: CType::Long,
            kind: Box::new(kind),
        }
    }

    fn main_symbol_table() -> semantic_analysis::SymbolTable {
        let mut table = semantic_analysis::SymbolTable::new();
        table.insert(
            "main".into(),
            (
                CType::FunType {
                    params: vec![],
                    ret: Box::new(CType::Int),
                },
                semantic_analysis::IdentifierAttrs::FunAttr {
                    defined: true,
                    global: true,
                },
            ),
        );
        table
    }

    #[test]
    fn basic_parse() {
        let ast = TypedAST::Program(vec![TypedDeclaration::FunDecl(TypedFunctionDeclaration {
            name: "main".into(),
            block: Some(crate::ast::Block(vec![TypedBlockItem::Stmt(
                TypedStatement::Return(typed_int(TypedExprKind::Constant(Const::Int(100)))),
            )])),
            params: vec![],
            storage_class: None,
            ftype: CType::FunType {
                params: vec![],
                ret: Box::new(CType::Int),
            },
        })]);

        let expected = AST::Program(vec![TopLevel::Function {
            name: "main".into(),
            params: vec![],
            global: true,
            instructions: vec![
                Instruction::Ret(Val::Constant(Const::Int(100))),
                Instruction::Ret(Val::Constant(Const::Int(0))),
            ],
        }]);

        let tacky = Tacky::new(ast);
        let assembly = tacky.into_ast(&mut &mut main_symbol_table());
        assert!(assembly.is_ok());
        let assembly = assembly.unwrap();
        assert_eq!(assembly, expected);
    }

    #[test]
    fn unary_op_parse() {
        let ast = TypedAST::Program(vec![TypedDeclaration::FunDecl(TypedFunctionDeclaration {
            name: "main".into(),
            block: Some(crate::ast::Block(vec![TypedBlockItem::Stmt(
                TypedStatement::Return(typed_int(TypedExprKind::Unary(
                    ParserUnaryOp::Negate,
                    Box::new(typed_int(TypedExprKind::Constant(Const::Int(100)))),
                ))),
            )])),
            params: vec![],
            storage_class: None,
            ftype: CType::FunType {
                params: vec![],
                ret: Box::new(CType::Int),
            },
        })]);

        let expected = AST::Program(vec![TopLevel::Function {
            name: "main".into(),
            params: vec![],
            global: true,
            instructions: vec![
                Instruction::Unary {
                    op: UnaryOp::Negate,
                    src: Val::Constant(Const::Int(100)),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::Ret(Val::Var("tmp.0".into())),
                Instruction::Ret(Val::Constant(Const::Int(0))),
            ],
        }]);

        let tacky = Tacky::new(ast);
        let assembly = tacky.into_ast(&mut main_symbol_table());
        assert!(assembly.is_ok());
        let assembly = assembly.unwrap();
        assert_eq!(assembly, expected);
    }

    #[test]
    fn complex_unary_parse() {
        let ast = TypedAST::Program(vec![TypedDeclaration::FunDecl(TypedFunctionDeclaration {
            name: "main".into(),
            block: Some(crate::ast::Block(vec![TypedBlockItem::Stmt(
                TypedStatement::Return(typed_int(TypedExprKind::Unary(
                    ParserUnaryOp::Negate,
                    Box::new(typed_int(TypedExprKind::Unary(
                        ParserUnaryOp::Complement,
                        Box::new(typed_int(TypedExprKind::Unary(
                            ParserUnaryOp::Negate,
                            Box::new(typed_int(TypedExprKind::Constant(Const::Int(100)))),
                        ))),
                    ))),
                ))),
            )])),
            params: vec![],
            storage_class: None,
            ftype: CType::FunType {
                params: vec![],
                ret: Box::new(CType::Int),
            },
        })]);

        let expected = AST::Program(vec![TopLevel::Function {
            name: "main".into(),
            params: vec![],
            global: true,
            instructions: vec![
                Instruction::Unary {
                    op: UnaryOp::Negate,
                    src: Val::Constant(Const::Int(100)),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::Unary {
                    op: UnaryOp::Complement,
                    src: Val::Var("tmp.0".into()),
                    dst: Val::Var("tmp.1".into()),
                },
                Instruction::Unary {
                    op: UnaryOp::Negate,
                    src: Val::Var("tmp.1".into()),
                    dst: Val::Var("tmp.2".into()),
                },
                Instruction::Ret(Val::Var("tmp.2".into())),
                Instruction::Ret(Val::Constant(Const::Int(0))),
            ],
        }]);

        let tacky = Tacky::new(ast);
        let assembly = tacky.into_ast(&mut main_symbol_table());
        assert!(assembly.is_ok());
        let assembly = assembly.unwrap();
        assert_eq!(assembly, expected);
    }

    #[test]
    fn complex_binary_parse() {
        let ast = TypedAST::Program(vec![TypedDeclaration::FunDecl(TypedFunctionDeclaration {
            name: "main".into(),
            block: Some(crate::ast::Block(vec![TypedBlockItem::Stmt(
                TypedStatement::Return(typed_int(TypedExprKind::Binary(
                    ParserBinaryOp::Subtract,
                    Box::new(typed_int(TypedExprKind::Binary(
                        ParserBinaryOp::Multiply,
                        Box::new(typed_int(TypedExprKind::Constant(Const::Int(1)))),
                        Box::new(typed_int(TypedExprKind::Constant(Const::Int(2)))),
                    ))),
                    Box::new(typed_int(TypedExprKind::Binary(
                        ParserBinaryOp::Multiply,
                        Box::new(typed_int(TypedExprKind::Constant(Const::Int(3)))),
                        Box::new(typed_int(TypedExprKind::Binary(
                            ParserBinaryOp::Add,
                            Box::new(typed_int(TypedExprKind::Constant(Const::Int(4)))),
                            Box::new(typed_int(TypedExprKind::Constant(Const::Int(5)))),
                        ))),
                    ))),
                ))),
            )])),
            params: vec![],
            storage_class: None,
            ftype: CType::FunType {
                params: vec![],
                ret: Box::new(CType::Int),
            },
        })]);
        let expected = AST::Program(vec![TopLevel::Function {
            name: "main".into(),
            params: vec![],
            global: true,
            instructions: vec![
                Instruction::Binary {
                    op: BinaryOp::Multiply,
                    src1: Val::Constant(Const::Int(1)),
                    src2: Val::Constant(Const::Int(2)),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::Binary {
                    op: BinaryOp::Add,
                    src1: Val::Constant(Const::Int(4)),
                    src2: Val::Constant(Const::Int(5)),
                    dst: Val::Var("tmp.1".into()),
                },
                Instruction::Binary {
                    op: BinaryOp::Multiply,
                    src1: Val::Constant(Const::Int(3)),
                    src2: Val::Var("tmp.1".into()),
                    dst: Val::Var("tmp.2".into()),
                },
                Instruction::Binary {
                    op: BinaryOp::Subtract,
                    src1: Val::Var("tmp.0".into()),
                    src2: Val::Var("tmp.2".into()),
                    dst: Val::Var("tmp.3".into()),
                },
                Instruction::Ret(Val::Var("tmp.3".into())),
                Instruction::Ret(Val::Constant(Const::Int(0))),
            ],
        }]);

        let tacky = Tacky::new(ast);
        let assembly = tacky.into_ast(&mut main_symbol_table());
        assert!(assembly.is_ok());
        let assembly = assembly.unwrap();
        assert_eq!(assembly, expected);
    }

    #[test]
    fn complex_binary_parse2() {
        let ast = TypedAST::Program(vec![TypedDeclaration::FunDecl(TypedFunctionDeclaration {
            name: "main".into(),
            block: Some(crate::ast::Block(vec![TypedBlockItem::Stmt(
                TypedStatement::Return(typed_int(TypedExprKind::Binary(
                    ParserBinaryOp::Subtract,
                    Box::new(typed_int(TypedExprKind::Binary(
                        ParserBinaryOp::Divide,
                        Box::new(typed_int(TypedExprKind::Binary(
                            ParserBinaryOp::Multiply,
                            Box::new(typed_int(TypedExprKind::Constant(Const::Int(5)))),
                            Box::new(typed_int(TypedExprKind::Constant(Const::Int(4)))),
                        ))),
                        Box::new(typed_int(TypedExprKind::Constant(Const::Int(2)))),
                    ))),
                    Box::new(typed_int(TypedExprKind::Binary(
                        ParserBinaryOp::Remainder,
                        Box::new(typed_int(TypedExprKind::Constant(Const::Int(3)))),
                        Box::new(typed_int(TypedExprKind::Binary(
                            ParserBinaryOp::Add,
                            Box::new(typed_int(TypedExprKind::Constant(Const::Int(2)))),
                            Box::new(typed_int(TypedExprKind::Constant(Const::Int(1)))),
                        ))),
                    ))),
                ))),
            )])),
            params: vec![],
            storage_class: None,
            ftype: CType::FunType {
                params: vec![],
                ret: Box::new(CType::Int),
            },
        })]);

        let expected = AST::Program(vec![TopLevel::Function {
            name: "main".into(),
            params: vec![],
            global: true,
            instructions: vec![
                Instruction::Binary {
                    op: BinaryOp::Multiply,
                    src1: Val::Constant(Const::Int(5)),
                    src2: Val::Constant(Const::Int(4)),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::Binary {
                    op: BinaryOp::Divide,
                    src1: Val::Var("tmp.0".into()),
                    src2: Val::Constant(Const::Int(2)),
                    dst: Val::Var("tmp.1".into()),
                },
                Instruction::Binary {
                    op: BinaryOp::Add,
                    src1: Val::Constant(Const::Int(2)),
                    src2: Val::Constant(Const::Int(1)),
                    dst: Val::Var("tmp.2".into()),
                },
                Instruction::Binary {
                    op: BinaryOp::Remainder,
                    src1: Val::Constant(Const::Int(3)),
                    src2: Val::Var("tmp.2".into()),
                    dst: Val::Var("tmp.3".into()),
                },
                Instruction::Binary {
                    op: BinaryOp::Subtract,
                    src1: Val::Var("tmp.1".into()),
                    src2: Val::Var("tmp.3".into()),
                    dst: Val::Var("tmp.4".into()),
                },
                Instruction::Ret(Val::Var("tmp.4".into())),
                Instruction::Ret(Val::Constant(Const::Int(0))),
            ],
        }]);

        let tacky = Tacky::new(ast);
        let assembly = tacky.into_ast(&mut main_symbol_table());
        assert!(assembly.is_ok());
        let assembly = assembly.unwrap();
        assert_eq!(assembly, expected);
    }

    #[test]
    fn simple_bitwise() {
        let ast = TypedAST::Program(vec![TypedDeclaration::FunDecl(TypedFunctionDeclaration {
            name: "main".into(),
            block: Some(crate::ast::Block(vec![TypedBlockItem::Stmt(
                TypedStatement::Return(typed_int(TypedExprKind::Binary(
                    ParserBinaryOp::BitwiseOr,
                    Box::new(typed_int(TypedExprKind::Binary(
                        ParserBinaryOp::Multiply,
                        Box::new(typed_int(TypedExprKind::Constant(Const::Int(5)))),
                        Box::new(typed_int(TypedExprKind::Constant(Const::Int(4)))),
                    ))),
                    Box::new(typed_int(TypedExprKind::Binary(
                        ParserBinaryOp::BitwiseAnd,
                        Box::new(typed_int(TypedExprKind::Binary(
                            ParserBinaryOp::Subtract,
                            Box::new(typed_int(TypedExprKind::Constant(Const::Int(4)))),
                            Box::new(typed_int(TypedExprKind::Constant(Const::Int(5)))),
                        ))),
                        Box::new(typed_int(TypedExprKind::Constant(Const::Int(6)))),
                    ))),
                ))),
            )])),
            params: vec![],
            storage_class: None,
            ftype: CType::FunType {
                params: vec![],
                ret: Box::new(CType::Int),
            },
        })]);

        let expected = AST::Program(vec![TopLevel::Function {
            name: "main".into(),
            params: vec![],
            global: true,
            instructions: vec![
                Instruction::Binary {
                    op: BinaryOp::Multiply,
                    src1: Val::Constant(Const::Int(5)),
                    src2: Val::Constant(Const::Int(4)),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::Binary {
                    op: BinaryOp::Subtract,
                    src1: Val::Constant(Const::Int(4)),
                    src2: Val::Constant(Const::Int(5)),
                    dst: Val::Var("tmp.1".into()),
                },
                Instruction::Binary {
                    op: BinaryOp::BitwiseAnd,
                    src1: Val::Var("tmp.1".into()),
                    src2: Val::Constant(Const::Int(6)),
                    dst: Val::Var("tmp.2".into()),
                },
                Instruction::Binary {
                    op: BinaryOp::BitwiseOr,
                    src1: Val::Var("tmp.0".into()),
                    src2: Val::Var("tmp.2".into()),
                    dst: Val::Var("tmp.3".into()),
                },
                Instruction::Ret(Val::Var("tmp.3".into())),
                Instruction::Ret(Val::Constant(Const::Int(0))),
            ],
        }]);

        let tacky = Tacky::new(ast);
        let assembly = tacky.into_ast(&mut main_symbol_table());
        assert!(assembly.is_ok());
        let assembly = assembly.unwrap();
        assert_eq!(assembly, expected);
    }

    #[test]
    fn shiftleft() {
        let ast = TypedAST::Program(vec![TypedDeclaration::FunDecl(TypedFunctionDeclaration {
            name: "main".into(),
            block: Some(crate::ast::Block(vec![TypedBlockItem::Stmt(
                TypedStatement::Return(typed_int(TypedExprKind::Binary(
                    ParserBinaryOp::ShiftLeft,
                    Box::new(typed_int(TypedExprKind::Binary(
                        ParserBinaryOp::Multiply,
                        Box::new(typed_int(TypedExprKind::Constant(Const::Int(5)))),
                        Box::new(typed_int(TypedExprKind::Constant(Const::Int(4)))),
                    ))),
                    Box::new(typed_int(TypedExprKind::Constant(Const::Int(2)))),
                ))),
            )])),
            params: vec![],
            storage_class: None,
            ftype: CType::FunType {
                params: vec![],
                ret: Box::new(CType::Int),
            },
        })]);
        let expected = AST::Program(vec![TopLevel::Function {
            name: "main".into(),
            params: vec![],
            global: true,
            instructions: vec![
                Instruction::Binary {
                    op: BinaryOp::Multiply,
                    src1: Val::Constant(Const::Int(5)),
                    src2: Val::Constant(Const::Int(4)),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::Binary {
                    op: BinaryOp::ShiftLeft,
                    src1: Val::Var("tmp.0".into()),
                    src2: Val::Constant(Const::Int(2)),
                    dst: Val::Var("tmp.1".into()),
                },
                Instruction::Ret(Val::Var("tmp.1".into())),
                Instruction::Ret(Val::Constant(Const::Int(0))),
            ],
        }]);

        let tacky = Tacky::new(ast);
        let assembly = tacky.into_ast(&mut main_symbol_table());
        assert!(assembly.is_ok());
        let assembly = assembly.unwrap();
        assert_eq!(assembly, expected);
    }

    #[test]
    fn shiftleft_rhs_is_expr() {
        let ast = TypedAST::Program(vec![TypedDeclaration::FunDecl(TypedFunctionDeclaration {
            name: "main".into(),
            block: Some(crate::ast::Block(vec![TypedBlockItem::Stmt(
                TypedStatement::Return(typed_int(TypedExprKind::Binary(
                    ParserBinaryOp::ShiftLeft,
                    Box::new(typed_int(TypedExprKind::Constant(Const::Int(5)))),
                    Box::new(typed_int(TypedExprKind::Binary(
                        ParserBinaryOp::Add,
                        Box::new(typed_int(TypedExprKind::Constant(Const::Int(1)))),
                        Box::new(typed_int(TypedExprKind::Constant(Const::Int(2)))),
                    ))),
                ))),
            )])),
            params: vec![],
            storage_class: None,
            ftype: CType::FunType {
                params: vec![],
                ret: Box::new(CType::Int),
            },
        })]);
        let expected = AST::Program(vec![TopLevel::Function {
            name: "main".into(),
            params: vec![],
            global: true,
            instructions: vec![
                Instruction::Binary {
                    op: BinaryOp::Add,
                    src1: Val::Constant(Const::Int(1)),
                    src2: Val::Constant(Const::Int(2)),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::Binary {
                    op: BinaryOp::ShiftLeft,
                    src1: Val::Constant(Const::Int(5)),
                    src2: Val::Var("tmp.0".into()),
                    dst: Val::Var("tmp.1".into()),
                },
                Instruction::Ret(Val::Var("tmp.1".into())),
                Instruction::Ret(Val::Constant(Const::Int(0))),
            ],
        }]);

        let tacky = Tacky::new(ast);
        let assembly = tacky.into_ast(&mut main_symbol_table());
        assert!(assembly.is_ok());
        let assembly = assembly.unwrap();
        assert_eq!(assembly, expected);
    }

    #[test]
    fn test_short_circuit_and() {
        let ast = TypedAST::Program(vec![TypedDeclaration::FunDecl(TypedFunctionDeclaration {
            name: "main".into(),
            block: Some(crate::ast::Block(vec![TypedBlockItem::Stmt(
                TypedStatement::Return(typed_int(TypedExprKind::Binary(
                    ParserBinaryOp::BinAnd,
                    Box::new(typed_int(TypedExprKind::Constant(Const::Int(5)))),
                    Box::new(typed_int(TypedExprKind::Binary(
                        ParserBinaryOp::Add,
                        Box::new(typed_int(TypedExprKind::Constant(Const::Int(1)))),
                        Box::new(typed_int(TypedExprKind::Constant(Const::Int(2)))),
                    ))),
                ))),
            )])),
            params: vec![],
            storage_class: None,
            ftype: CType::FunType {
                params: vec![],
                ret: Box::new(CType::Int),
            },
        })]);

        let expected = AST::Program(vec![TopLevel::Function {
            name: "main".into(),
            params: vec![],
            global: true,
            instructions: vec![
                Instruction::Copy {
                    src: Val::Constant(Const::Int(5)),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::JumpIfZero {
                    cond: Val::Var("tmp.0".into()),
                    target: "and_expr_false.0".into(),
                },
                Instruction::Binary {
                    op: BinaryOp::Add,
                    src1: Val::Constant(Const::Int(1)),
                    src2: Val::Constant(Const::Int(2)),
                    dst: Val::Var("tmp.1".into()),
                },
                Instruction::Copy {
                    src: Val::Var("tmp.1".into()),
                    dst: Val::Var("tmp.2".into()),
                },
                Instruction::JumpIfZero {
                    cond: Val::Var("tmp.2".into()),
                    target: "and_expr_false.0".into(),
                },
                Instruction::Copy {
                    src: Val::Constant(Const::Int(1)),
                    dst: Val::Var("tmp.3".into()),
                },
                Instruction::Jump("and_expr_end.1".into()),
                Instruction::Label("and_expr_false.0".into()),
                Instruction::Copy {
                    src: Val::Constant(Const::Int(0)),
                    dst: Val::Var("tmp.3".into()),
                },
                Instruction::Label("and_expr_end.1".into()),
                Instruction::Ret(Val::Var("tmp.3".into())),
                Instruction::Ret(Val::Constant(Const::Int(0))),
            ],
        }]);

        let tacky = Tacky::new(ast);
        let assembly = tacky.into_ast(&mut main_symbol_table());
        assert!(assembly.is_ok());
        let assembly = assembly.unwrap();
        assert_eq!(assembly, expected);
    }

    #[test]
    fn test_short_circuit_or() {
        let ast = TypedAST::Program(vec![TypedDeclaration::FunDecl(TypedFunctionDeclaration {
            name: "main".into(),
            block: Some(crate::ast::Block(vec![TypedBlockItem::Stmt(
                TypedStatement::Return(typed_int(TypedExprKind::Binary(
                    ParserBinaryOp::BinOr,
                    Box::new(typed_int(TypedExprKind::Constant(Const::Int(5)))),
                    Box::new(typed_int(TypedExprKind::Binary(
                        ParserBinaryOp::Add,
                        Box::new(typed_int(TypedExprKind::Constant(Const::Int(1)))),
                        Box::new(typed_int(TypedExprKind::Constant(Const::Int(2)))),
                    ))),
                ))),
            )])),
            params: vec![],
            storage_class: None,
            ftype: CType::FunType {
                params: vec![],
                ret: Box::new(CType::Int),
            },
        })]);

        let expected = AST::Program(vec![TopLevel::Function {
            name: "main".into(),
            params: vec![],
            global: true,
            instructions: vec![
                Instruction::Copy {
                    src: Val::Constant(Const::Int(5)),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::JumpIfNotZero {
                    cond: Val::Var("tmp.0".into()),
                    target: "or_expr_true.0".into(),
                },
                Instruction::Binary {
                    op: BinaryOp::Add,
                    src1: Val::Constant(Const::Int(1)),
                    src2: Val::Constant(Const::Int(2)),
                    dst: Val::Var("tmp.1".into()),
                },
                Instruction::Copy {
                    src: Val::Var("tmp.1".into()),
                    dst: Val::Var("tmp.2".into()),
                },
                Instruction::JumpIfNotZero {
                    cond: Val::Var("tmp.2".into()),
                    target: "or_expr_true.0".into(),
                },
                Instruction::Copy {
                    src: Val::Constant(Const::Int(0)),
                    dst: Val::Var("tmp.3".into()),
                },
                Instruction::Jump("or_expr_end.1".into()),
                Instruction::Label("or_expr_true.0".into()),
                Instruction::Copy {
                    src: Val::Constant(Const::Int(1)),
                    dst: Val::Var("tmp.3".into()),
                },
                Instruction::Label("or_expr_end.1".into()),
                Instruction::Ret(Val::Var("tmp.3".into())),
                Instruction::Ret(Val::Constant(Const::Int(0))),
            ],
        }]);

        let tacky = Tacky::new(ast);
        let assembly = tacky.into_ast(&mut main_symbol_table());
        assert!(assembly.is_ok());
        let assembly = assembly.unwrap();
        assert_eq!(assembly, expected);
    }

    #[test]
    fn basic_declarations() {
        let ast = TypedAST::Program(vec![TypedDeclaration::FunDecl(TypedFunctionDeclaration {
            name: "main".into(),
            block: Some(crate::ast::Block(vec![
                TypedBlockItem::Decl(TypedDeclaration::VarDecl(TypedVariableDeclaration {
                    name: "a.0.decl".into(),
                    init: Some(typed_int(TypedExprKind::Constant(Const::Int(1)))),
                    storage_class: None,
                    vtype: CType::Int,
                })),
                TypedBlockItem::Decl(TypedDeclaration::VarDecl(TypedVariableDeclaration {
                    name: "b.1.decl".into(),
                    init: Some(typed_int(TypedExprKind::Var("a.0.decl".into()))),
                    storage_class: None,
                    vtype: CType::Int,
                })),
                TypedBlockItem::Decl(TypedDeclaration::VarDecl(TypedVariableDeclaration {
                    name: "c.2.decl".into(),
                    init: Some(typed_int(TypedExprKind::Binary(
                        ParserBinaryOp::Add,
                        Box::new(typed_int(TypedExprKind::Var("a.0.decl".into()))),
                        Box::new(typed_int(TypedExprKind::Var("b.1.decl".into()))),
                    ))),
                    storage_class: None,
                    vtype: CType::Int,
                })),
                TypedBlockItem::Stmt(TypedStatement::Return(typed_int(TypedExprKind::Var(
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
        let expected = AST::Program(vec![TopLevel::Function {
            name: "main".into(),
            params: vec![],
            global: true,
            instructions: vec![
                Instruction::Copy {
                    src: Val::Constant(Const::Int(1)),
                    dst: Val::Var("a.0.decl".into()),
                },
                Instruction::Copy {
                    src: Val::Var("a.0.decl".into()),
                    dst: Val::Var("b.1.decl".into()),
                },
                Instruction::Binary {
                    op: BinaryOp::Add,
                    src1: Val::Var("a.0.decl".into()),
                    src2: Val::Var("b.1.decl".into()),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::Copy {
                    src: Val::Var("tmp.0".into()),
                    dst: Val::Var("c.2.decl".into()),
                },
                Instruction::Ret(Val::Var("c.2.decl".into())),
                Instruction::Ret(Val::Constant(Const::Int(0))),
            ],
        }]);

        let tacky = Tacky::new(ast);
        let assembly = tacky.into_ast(&mut main_symbol_table());
        assert!(assembly.is_ok());
        let assembly = assembly.unwrap();
        assert_eq!(assembly, expected);
    }

    #[test]
    fn test_compound_assignment() {
        let ast = TypedAST::Program(vec![TypedDeclaration::FunDecl(TypedFunctionDeclaration {
            name: "main".into(),
            block: Some(crate::ast::Block(vec![
                TypedBlockItem::Decl(TypedDeclaration::VarDecl(TypedVariableDeclaration {
                    name: "a".into(),
                    init: Some(typed_int(TypedExprKind::Constant(Const::Int(1)))),
                    storage_class: None,
                    vtype: CType::Int,
                })),
                TypedBlockItem::Stmt(TypedStatement::Expr(typed_int(TypedExprKind::Binary(
                    ParserBinaryOp::AddAssign,
                    Box::new(typed_int(TypedExprKind::Var("a".into()))),
                    Box::new(typed_int(TypedExprKind::Constant(Const::Int(2)))),
                )))),
                TypedBlockItem::Stmt(TypedStatement::Expr(typed_int(TypedExprKind::Binary(
                    ParserBinaryOp::MinusAssign,
                    Box::new(typed_int(TypedExprKind::Var("a".into()))),
                    Box::new(typed_int(TypedExprKind::Constant(Const::Int(2)))),
                )))),
                TypedBlockItem::Stmt(TypedStatement::Expr(typed_int(TypedExprKind::Binary(
                    ParserBinaryOp::MultiplyAssign,
                    Box::new(typed_int(TypedExprKind::Var("a".into()))),
                    Box::new(typed_int(TypedExprKind::Constant(Const::Int(2)))),
                )))),
                TypedBlockItem::Stmt(TypedStatement::Expr(typed_int(TypedExprKind::Binary(
                    ParserBinaryOp::DivideAssign,
                    Box::new(typed_int(TypedExprKind::Var("a".into()))),
                    Box::new(typed_int(TypedExprKind::Constant(Const::Int(2)))),
                )))),
                TypedBlockItem::Stmt(TypedStatement::Expr(typed_int(TypedExprKind::Binary(
                    ParserBinaryOp::RemainderAssign,
                    Box::new(typed_int(TypedExprKind::Var("a".into()))),
                    Box::new(typed_int(TypedExprKind::Constant(Const::Int(2)))),
                )))),
                TypedBlockItem::Stmt(TypedStatement::Return(typed_int(TypedExprKind::Var(
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

        let expected = AST::Program(vec![TopLevel::Function {
            name: "main".into(),
            params: vec![],
            global: true,
            instructions: vec![
                Instruction::Copy {
                    src: Val::Constant(Const::Int(1)),
                    dst: Val::Var("a".into()),
                },
                Instruction::Binary {
                    op: BinaryOp::Add,
                    src1: Val::Var("a".into()),
                    src2: Val::Constant(Const::Int(2)),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::Copy {
                    src: Val::Var("tmp.0".into()),
                    dst: Val::Var("a".into()),
                },
                Instruction::Binary {
                    op: BinaryOp::Subtract,
                    src1: Val::Var("a".into()),
                    src2: Val::Constant(Const::Int(2)),
                    dst: Val::Var("tmp.1".into()),
                },
                Instruction::Copy {
                    src: Val::Var("tmp.1".into()),
                    dst: Val::Var("a".into()),
                },
                Instruction::Binary {
                    op: BinaryOp::Multiply,
                    src1: Val::Var("a".into()),
                    src2: Val::Constant(Const::Int(2)),
                    dst: Val::Var("tmp.2".into()),
                },
                Instruction::Copy {
                    src: Val::Var("tmp.2".into()),
                    dst: Val::Var("a".into()),
                },
                Instruction::Binary {
                    op: BinaryOp::Divide,
                    src1: Val::Var("a".into()),
                    src2: Val::Constant(Const::Int(2)),
                    dst: Val::Var("tmp.3".into()),
                },
                Instruction::Copy {
                    src: Val::Var("tmp.3".into()),
                    dst: Val::Var("a".into()),
                },
                Instruction::Binary {
                    op: BinaryOp::Remainder,
                    src1: Val::Var("a".into()),
                    src2: Val::Constant(Const::Int(2)),
                    dst: Val::Var("tmp.4".into()),
                },
                Instruction::Copy {
                    src: Val::Var("tmp.4".into()),
                    dst: Val::Var("a".into()),
                },
                Instruction::Ret(Val::Var("a".into())),
                Instruction::Ret(Val::Constant(Const::Int(0))),
            ],
        }]);

        let tacky = Tacky::new(ast);
        let assembly = tacky.into_ast(&mut main_symbol_table());
        assert!(assembly.is_ok());
        let assembly = assembly.unwrap();
        assert_eq!(assembly, expected);
    }

    #[test]
    fn test_invalid_lhs_compound_assignment() {
        let ast = TypedAST::Program(vec![TypedDeclaration::FunDecl(TypedFunctionDeclaration {
            name: "main".into(),
            block: Some(crate::ast::Block(vec![
                TypedBlockItem::Decl(TypedDeclaration::VarDecl(TypedVariableDeclaration {
                    name: "a".into(),
                    init: Some(typed_int(TypedExprKind::Constant(Const::Int(10)))),
                    storage_class: None,
                    vtype: CType::Int,
                })),
                TypedBlockItem::Stmt(TypedStatement::Expr(typed_int(TypedExprKind::Binary(
                    ParserBinaryOp::MinusAssign,
                    Box::new(typed_int(TypedExprKind::Binary(
                        ParserBinaryOp::AddAssign,
                        Box::new(typed_int(TypedExprKind::Var("a".into()))),
                        Box::new(typed_int(TypedExprKind::Constant(Const::Int(1)))),
                    ))),
                    Box::new(typed_int(TypedExprKind::Constant(Const::Int(2)))),
                )))),
            ])),
            params: vec![],
            storage_class: None,
            ftype: CType::FunType {
                params: vec![],
                ret: Box::new(CType::Int),
            },
        })]);
        let tacky = Tacky::new(ast);
        let assembly = tacky.into_ast(&mut main_symbol_table());
        assert!(assembly.is_err());
    }

    #[test]
    fn test_if_then_else() {
        let src = r#"
            int main(void) {
                int a = 1;
                int b = 0;
                if (a)
                    return 1;

                if (b) 
                    return 2; 
                else 
                    return 3;
            }
        "#;
        let lexer = crate::lexer::Lexer::lex(src).unwrap();
        let tokens = lexer.as_syntactic_tokens();
        let parse = crate::parser::Parser::new(&tokens);
        let mut ast = parse.into_ast().unwrap();
        let (mut symbol_table, ast) = crate::semantic_analysis::resolve(&mut ast).unwrap();
        let asm = Tacky::new(ast);
        let assembly = asm.into_ast(&mut symbol_table);
        let Ok(actual) = assembly else {
            panic!();
        };
        let expected = AST::Program(vec![TopLevel::Function {
            name: "main".into(),
            params: vec![],
            global: true,
            instructions: vec![
                // first one: if/then
                Instruction::Copy {
                    src: Val::Constant(Const::Int(1)),
                    dst: Val::Var("a.0.decl".into()),
                },
                Instruction::Copy {
                    src: Val::Constant(Const::Int(0)),
                    dst: Val::Var("b.1.decl".into()),
                },
                Instruction::Copy {
                    src: Val::Var("a.0.decl".into()),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::JumpIfZero {
                    cond: Val::Var("tmp.0".into()),
                    target: "end_label.0".into(),
                },
                Instruction::Ret(Val::Constant(Const::Int(1))),
                Instruction::Label("end_label.0".into()),
                // second if/then/else
                Instruction::Copy {
                    src: Val::Var("b.1.decl".into()),
                    dst: Val::Var("tmp.1".into()),
                },
                Instruction::JumpIfZero {
                    cond: Val::Var("tmp.1".into()),
                    target: "else_label.1".into(),
                },
                Instruction::Ret(Val::Constant(Const::Int(2))),
                Instruction::Jump("end_label.2".into()),
                Instruction::Label("else_label.1".into()),
                Instruction::Ret(Val::Constant(Const::Int(3))),
                Instruction::Label("end_label.2".into()),
                Instruction::Ret(Val::Constant(Const::Int(0))),
            ],
        }]);
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_conditional() {
        let src = r#"
            int main(void) {
                int a = 1;
                a ? 1 : 2;
            }
        "#;
        let lexer = crate::lexer::Lexer::lex(src).unwrap();
        let tokens = lexer.as_syntactic_tokens();
        let parse = crate::parser::Parser::new(&tokens);
        let mut ast = parse.into_ast().unwrap();
        let (mut symbol_table, ast) = crate::semantic_analysis::resolve(&mut ast).unwrap();
        let asm = Tacky::new(ast);
        let assembly = asm.into_ast(&mut symbol_table);
        let Ok(actual) = assembly else {
            panic!();
        };
        let expected = AST::Program(vec![TopLevel::Function {
            name: "main".into(),
            params: vec![],
            global: true,
            instructions: vec![
                Instruction::Copy {
                    src: Val::Constant(Const::Int(1)),
                    dst: Val::Var("a.0.decl".into()),
                },
                Instruction::Copy {
                    src: Val::Var("a.0.decl".into()),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::JumpIfZero {
                    cond: Val::Var("tmp.0".into()),
                    target: "else_label.0".into(),
                },
                Instruction::Copy {
                    src: Val::Constant(Const::Int(1)),
                    dst: Val::Var("tmp.1".into()),
                },
                Instruction::Jump("end_label.1".into()),
                Instruction::Label("else_label.0".into()),
                Instruction::Copy {
                    src: Val::Constant(Const::Int(2)),
                    dst: Val::Var("tmp.1".into()),
                },
                Instruction::Label("end_label.1".into()),
                Instruction::Ret(Val::Constant(Const::Int(0))),
            ],
        }]);
        assert_eq!(expected, actual);
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
        let parse = crate::parser::Parser::new(&tokens);
        let mut ast = parse.into_ast().unwrap();
        let (mut symbol_table, ast) = crate::semantic_analysis::resolve(&mut ast).unwrap();
        let asm = Tacky::new(ast);
        let assembly = asm.into_ast(&mut symbol_table);
        let Ok(actual) = assembly else {
            panic!();
        };
        let expected = AST::Program(vec![TopLevel::Function {
            name: "main".into(),
            params: vec![],
            global: true,
            instructions: vec![
                Instruction::Jump("foo.0.label".into()),
                Instruction::Label("foo.0.label".into()),
                Instruction::Binary {
                    op: BinaryOp::Add,
                    src1: Val::Constant(Const::Int(1)),
                    src2: Val::Constant(Const::Int(2)),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::Ret(Val::Var("tmp.0".into())),
                Instruction::Ret(Val::Constant(Const::Int(0))),
            ],
        }]);
        assert_eq!(expected, actual);
    }

    #[test]
    fn compound_statements_and_blocks() {
        let src = r#"
            int main(void) {
                if (1) {
                    return 1;
                } else {
                    return 2;
                }
                {
                    int x = 3;
                    return x;
                }
            }
        "#;
        let lexer = crate::lexer::Lexer::lex(src).unwrap();
        let tokens = lexer.as_syntactic_tokens();
        let parse = crate::parser::Parser::new(&tokens);
        let mut ast = parse.into_ast().unwrap();
        let (mut symbol_table, ast) = crate::semantic_analysis::resolve(&mut ast).unwrap();
        let asm = Tacky::new(ast);
        let assembly = asm.into_ast(&mut symbol_table);
        let Ok(actual) = assembly else {
            panic!();
        };
        let expected = AST::Program(vec![TopLevel::Function {
            name: "main".into(),
            params: vec![],
            global: true,
            instructions: vec![
                Instruction::Copy {
                    src: Val::Constant(Const::Int(1)),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::JumpIfZero {
                    cond: Val::Var("tmp.0".into()),
                    target: "else_label.0".into(),
                },
                Instruction::Ret(Val::Constant(Const::Int(1))),
                Instruction::Jump("end_label.1".into()),
                Instruction::Label("else_label.0".into()),
                Instruction::Ret(Val::Constant(Const::Int(2))),
                Instruction::Label("end_label.1".into()),
                Instruction::Copy {
                    src: Val::Constant(Const::Int(3)),
                    dst: Val::Var("x.0.decl".into()),
                },
                Instruction::Ret(Val::Var("x.0.decl".into())),
                Instruction::Ret(Val::Constant(Const::Int(0))),
            ],
        }]);
        assert_eq!(expected, actual);
    }

    #[test]
    fn loops_and_breaks() {
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
        let (mut symbol_table, ast) = crate::semantic_analysis::resolve(&mut ast).unwrap();
        let asm = Tacky::new(ast);
        let assembly = asm.into_ast(&mut symbol_table);
        let Ok(actual) = assembly else {
            panic!();
        };
        let expected = AST::Program(vec![TopLevel::Function {
            name: "main".into(),
            params: vec![],
            global: true,
            instructions: vec![
                /* handle initializer, write out for loop label,
                 * check condition, JumpIfZero, write out body, write out continue label,
                 * write out post-condition, jump to start, then write break label
                 */
                Instruction::Copy {
                    src: Val::Constant(Const::Int(1)),
                    dst: Val::Var("a.0.decl".into()),
                },
                Instruction::Label("for_label.1".into()),
                Instruction::Binary {
                    op: BinaryOp::LessThan,
                    src1: Val::Var("a.0.decl".into()),
                    src2: Val::Constant(Const::Int(10)),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::JumpIfZero {
                    cond: Val::Var("tmp.0".into()),
                    target: "break_for_label.1".into(),
                },
                Instruction::Jump("continue_for_label.1".into()),
                Instruction::Label("continue_for_label.1".into()),
                Instruction::Binary {
                    op: BinaryOp::Add,
                    src1: Val::Var("a.0.decl".into()),
                    src2: Val::Constant(Const::Int(1)),
                    dst: Val::Var("tmp.1".into()),
                },
                Instruction::Copy {
                    src: Val::Var("tmp.1".into()),
                    dst: Val::Var("a.0.decl".into()),
                },
                Instruction::Jump("for_label.1".into()),
                Instruction::Label("break_for_label.1".into()),
                /*
                 * do-while: write starting label, eval body, write continue label out,
                 * eval condition, JumpIfNotZero to starting label, write break label,
                 */
                Instruction::Label("do_while_label.2".into()),
                Instruction::Jump("continue_do_while_label.2".into()),
                Instruction::Label("continue_do_while_label.2".into()),
                Instruction::Binary {
                    op: BinaryOp::LessThan,
                    src1: Val::Var("a.0.decl".into()),
                    src2: Val::Constant(Const::Int(0)),
                    dst: Val::Var("tmp.2".into()),
                },
                Instruction::JumpIfNotZero {
                    cond: Val::Var("tmp.2".into()),
                    target: "do_while_label.2".into(),
                },
                Instruction::Label("break_do_while_label.2".into()),
                /*
                 * write out the continue label, then check condition. JumpIfZero to
                 * break label. Evaluate body. Write continue label. Write break label.
                 */
                Instruction::Label("continue_while_label.3".into()),
                Instruction::Binary {
                    op: BinaryOp::GreaterThan,
                    src1: Val::Var("a.0.decl".into()),
                    src2: Val::Constant(Const::Int(0)),
                    dst: Val::Var("tmp.3".into()),
                },
                Instruction::JumpIfZero {
                    cond: Val::Var("tmp.3".into()),
                    target: "break_while_label.3".into(),
                },
                Instruction::Jump("break_while_label.3".into()),
                Instruction::Jump("continue_while_label.3".into()),
                Instruction::Label("break_while_label.3".into()),
                // placeholder
                Instruction::Ret(Val::Constant(Const::Int(0))),
            ],
        }]);
        assert_eq!(expected, actual);
    }

    #[test]
    fn missing_condition_in_for_loop() {
        let src = r#"
        int main(void) {
            for (int i = 400; ; i = i - 100)
                if (i == 100)
                    return 0;
        }
        "#;
        let lexer = crate::lexer::Lexer::lex(src).unwrap();
        let tokens = lexer.as_syntactic_tokens();
        let parse = crate::parser::Parser::new(&tokens);
        let mut ast = parse.into_ast().unwrap();
        let (mut symbol_table, ast) = crate::semantic_analysis::resolve(&mut ast).unwrap();
        let asm = Tacky::new(ast);
        let assembly = asm.into_ast(&mut symbol_table);
        let Ok(actual) = assembly else {
            panic!();
        };
        let expected = AST::Program(vec![TopLevel::Function {
            name: "main".into(),
            params: vec![],
            global: true,
            instructions: vec![
                /* handle initializer, write out for loop label,
                 * check condition, JumpIfZero, write out body, write out continue label,
                 * write out post-condition, jump to start, then write break label
                 */
                Instruction::Copy {
                    src: Val::Constant(Const::Int(400)),
                    dst: Val::Var("i.0.decl".into()),
                },
                Instruction::Label("for_label.1".into()),
                Instruction::Binary {
                    op: BinaryOp::Equal,
                    src1: Val::Var("i.0.decl".into()),
                    src2: Val::Constant(Const::Int(100)),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::Copy {
                    src: Val::Var("tmp.0".into()),
                    dst: Val::Var("tmp.1".into()),
                },
                Instruction::JumpIfZero {
                    cond: Val::Var("tmp.1".into()),
                    target: "end_label.0".into(),
                },
                Instruction::Ret(Val::Constant(Const::Int(0))),
                Instruction::Label("end_label.0".into()),
                Instruction::Label("continue_for_label.1".into()),
                Instruction::Binary {
                    op: BinaryOp::Subtract,
                    src1: Val::Var("i.0.decl".into()),
                    src2: Val::Constant(Const::Int(100)),
                    dst: Val::Var("tmp.2".into()),
                },
                Instruction::Copy {
                    src: Val::Var("tmp.2".into()),
                    dst: Val::Var("i.0.decl".into()),
                },
                Instruction::Jump("for_label.1".into()),
                Instruction::Label("break_for_label.1".into()),
                // placeholder
                Instruction::Ret(Val::Constant(Const::Int(0))),
            ],
        }]);
        assert_eq!(expected, actual);
    }

    #[test]
    fn functions() {
        let src = r#"
            int bar(int a);
            int foo(int x, int y) { 
                return x + y;
            }
            int main(void) {
                return foo(1, 2) + 3;
            }
        "#;
        let lexer = crate::lexer::Lexer::lex(src).unwrap();
        let tokens = lexer.as_syntactic_tokens();
        let parse = crate::parser::Parser::new(&tokens);
        let mut ast = parse.into_ast().unwrap();
        let (mut symbol_table, ast) = crate::semantic_analysis::resolve(&mut ast).unwrap();
        let asm = Tacky::new(ast);
        let assembly = asm.into_ast(&mut symbol_table);
        let Ok(actual) = assembly else {
            panic!();
        };
        let expected = AST::Program(vec![
            TopLevel::Function {
                name: "foo".into(),
                params: vec!["x.1.decl".into(), "y.2.decl".into()],
                global: true,
                instructions: vec![
                    Instruction::Binary {
                        op: BinaryOp::Add,
                        src1: Val::Var("x.1.decl".into()),
                        src2: Val::Var("y.2.decl".into()),
                        dst: Val::Var("tmp.0".into()),
                    },
                    Instruction::Ret(Val::Var("tmp.0".into())),
                    Instruction::Ret(Val::Constant(Const::Int(0))),
                ],
            },
            TopLevel::Function {
                name: "main".into(),
                params: vec![],
                global: true,
                instructions: vec![
                    Instruction::FunCall {
                        name: "foo".into(),
                        args: vec![Val::Constant(Const::Int(1)), Val::Constant(Const::Int(2))],
                        dst: Val::Var("tmp.1".into()),
                    },
                    Instruction::Binary {
                        op: BinaryOp::Add,
                        src1: Val::Var("tmp.1".into()),
                        src2: Val::Constant(Const::Int(3)),
                        dst: Val::Var("tmp.2".into()),
                    },
                    Instruction::Ret(Val::Var("tmp.2".into())),
                    Instruction::Ret(Val::Constant(Const::Int(0))),
                ],
            },
        ]);
        assert_eq!(expected, actual);
    }

    #[test]
    fn static_variable() {
        let src = r#"
            static int x = 5;
            int main(void) {
                return x;
            }
        "#;
        let lexer = crate::lexer::Lexer::lex(src).unwrap();
        let tokens = lexer.as_syntactic_tokens();
        let parse = crate::parser::Parser::new(&tokens);
        let mut ast = parse.into_ast().unwrap();
        let (mut symbol_table, ast) = crate::semantic_analysis::resolve(&mut ast).unwrap();
        let asm = Tacky::new(ast);
        let actual = asm.into_ast(&mut symbol_table).unwrap();

        let AST::Program(top_levels) = actual;
        let mut iter = top_levels.into_iter();

        // First: main function with global: true
        let main_fn = iter.next().unwrap();
        assert!(
            matches!(main_fn, TopLevel::Function { ref name, global: true, .. } if name == "main")
        );

        // Second: static variable x with global: false, init: 5
        let static_var = iter.next().unwrap();
        assert!(
            matches!(static_var, TopLevel::StaticVariable { ref identifier, global: false, init: StaticInit::IntInit(5), t: CType::Int } if identifier == "x")
        );
    }

    #[test]
    fn static_function() {
        let src = r#"
            static int foo(void) { return 1; }
            int main(void) { return foo(); }
        "#;
        let lexer = crate::lexer::Lexer::lex(src).unwrap();
        let tokens = lexer.as_syntactic_tokens();
        let parse = crate::parser::Parser::new(&tokens);
        let mut ast = parse.into_ast().unwrap();
        let (mut symbol_table, ast) = crate::semantic_analysis::resolve(&mut ast).unwrap();
        let asm = Tacky::new(ast);
        let actual = asm.into_ast(&mut symbol_table).unwrap();

        let AST::Program(top_levels) = actual;
        let mut iter = top_levels.into_iter();

        // First: foo with global: false (static function)
        let foo_fn = iter.next().unwrap();
        assert!(
            matches!(foo_fn, TopLevel::Function { ref name, global: false, .. } if name == "foo")
        );

        // Second: main with global: true
        let main_fn = iter.next().unwrap();
        assert!(
            matches!(main_fn, TopLevel::Function { ref name, global: true, .. } if name == "main")
        );
    }

    fn tacky_instructions_for(src: &str) -> Vec<Instruction> {
        let lexer = crate::lexer::Lexer::lex(src).unwrap();
        let tokens = lexer.as_syntactic_tokens();
        let parse = crate::parser::Parser::new(&tokens);
        let mut ast = parse.into_ast().unwrap();
        let (mut symbol_table, typed_ast) = crate::semantic_analysis::resolve(&mut ast).unwrap();
        let tacky = Tacky::new(typed_ast);
        let AST::Program(mut top_levels) = tacky.into_ast(&mut symbol_table).unwrap();
        let TopLevel::Function { instructions, .. } = top_levels.remove(0) else {
            panic!("Expected a function");
        };
        instructions
    }

    #[test]
    fn return_int_from_long_function_truncates() {
        // returning int from a long function should emit Truncate
        let src = r#"
            long main(void) {
                int x = 1;
                return x;
            }
        "#;
        let instrs = tacky_instructions_for(src);
        assert!(
            instrs
                .iter()
                .any(|i| matches!(i, Instruction::SignExtend { .. })),
            "Expected a SignExtend instruction, got: {instrs:?}"
        );
    }

    #[test]
    fn return_long_from_int_function_truncates() {
        // returning long from an int function should emit Truncate
        let src = r#"
            int main(void) {
                long x = 1L;
                return x;
            }
        "#;
        let instrs = tacky_instructions_for(src);
        assert!(
            instrs
                .iter()
                .any(|i| matches!(i, Instruction::Truncate { .. })),
            "Expected a Truncate instruction, got: {instrs:?}"
        );
    }

    #[test]
    fn int_plus_long_sign_extends_int() {
        // int + long should sign-extend the int operand
        let src = r#"
            long main(void) {
                int a = 1;
                long b = 2L;
                return a + b;
            }
        "#;
        let instrs = tacky_instructions_for(src);
        assert!(
            instrs
                .iter()
                .any(|i| matches!(i, Instruction::SignExtend { .. })),
            "Expected a SignExtend instruction for int operand, got: {instrs:?}"
        );
    }

    #[test]
    fn function_call_with_mixed_args_emits_cast() {
        // calling foo(long) with an int arg should sign-extend
        let src = r#"
            long foo(long x);
            int main(void) {
                int a = 1;
                return foo(a);
            }
        "#;
        let instrs = tacky_instructions_for(src);
        assert!(
            instrs
                .iter()
                .any(|i| matches!(i, Instruction::SignExtend { .. })),
            "Expected a SignExtend for int->long arg conversion, got: {instrs:?}"
        );
    }
}
