// implements a Parser AST -> TACKY AST for the IR
// Mostly copied from asm.rs

use crate::parser;
use crate::parser::BinaryOp as ParserBinaryOp;
use crate::parser::Block;
use crate::parser::Expression;
use crate::parser::Statement;
use crate::parser::UnaryOp as ParserUnaryOp;
use crate::parser::AST as ParserAST;
use thiserror::Error;

#[derive(Debug, PartialEq, Error)]
pub enum TackyError {
    #[error("Expected program")]
    MissingProgram,
    #[error("Expected function")]
    MissingFunction,
    #[error("Found non-variable on lefthand side of assignment")]
    InvalidLhsOfAssignment,
}

// Lifetime of source test, since we need
// names. TODO: Figure out how to remove this dep.
#[derive(Debug, PartialEq)]
pub enum AST<'a> {
    Program(Function<'a>),
}

#[derive(Debug, PartialEq)]
pub struct Function<'a> {
    pub name: &'a str,
    pub instructions: Vec<Instruction>,
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
}

#[derive(Debug, PartialEq, Clone)]
pub enum Val {
    Constant(usize),
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

#[derive(Debug, PartialEq)]
pub struct Tacky<'a> {
    parser: ParserAST<'a>,
    dst_counter: u16,
    label_counter: u16,
}

impl<'a> Tacky<'a> {
    pub fn new(parser: ParserAST<'a>) -> Self {
        Self {
            parser,
            dst_counter: 0,
            label_counter: 0,
        }
    }

    pub fn into_ast(mut self) -> Result<AST<'a>, TackyError> {
        // Imperfect, but I need to be able to borrow the parser
        let parser = std::mem::replace(
            &mut self.parser,
            ParserAST::Function {
                name: "empty",
                block: crate::parser::Block(vec![]),
            },
        );
        self.parse_program(&parser)
    }

    fn parse_program(&mut self, parser: &ParserAST<'a>) -> Result<AST<'a>, TackyError> {
        match parser {
            ParserAST::Program(func) => {
                let func = self.parse_function(func)?;
                Ok(AST::Program(func))
            }
            _ => Err(TackyError::MissingProgram),
        }
    }

    fn parse_function(&mut self, parser: &ParserAST<'a>) -> Result<Function<'a>, TackyError> {
        match parser {
            ParserAST::Function { name, block } => {
                let instructions = self.parse_instructions(block)?;
                Ok(Function { name, instructions })
            }
            _ => Err(TackyError::MissingFunction),
        }
    }

    fn parse_expression(
        &mut self,
        expr: &Expression,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Val, TackyError> {
        match expr {
            Expression::Constant(imm) => Ok(Val::Constant(*imm)),
            Expression::Unary(op, exp) => {
                let src = self.parse_expression(exp, instructions)?;
                let dst_name = self.make_temporary();
                let dst = Val::Var(dst_name);
                let unary_op = match op {
                    ParserUnaryOp::Negate => UnaryOp::Negate,
                    ParserUnaryOp::Complement => UnaryOp::Complement,
                    ParserUnaryOp::Not => UnaryOp::Not,
                };
                instructions.push(Instruction::Unary {
                    op: unary_op,
                    src,
                    dst: dst.clone(),
                });
                Ok(dst)
            }
            Expression::Binary(op @ ParserBinaryOp::BinAnd, left, right)
            | Expression::Binary(op @ ParserBinaryOp::BinOr, left, right) => {
                self.parse_short_circuit_expression(op, left, right, instructions)
            }
            Expression::Binary(op @ ParserBinaryOp::AddAssign, left, right)
            | Expression::Binary(op @ ParserBinaryOp::MinusAssign, left, right)
            | Expression::Binary(op @ ParserBinaryOp::MultiplyAssign, left, right)
            | Expression::Binary(op @ ParserBinaryOp::DivideAssign, left, right)
            | Expression::Binary(op @ ParserBinaryOp::RemainderAssign, left, right)
            | Expression::Binary(op @ ParserBinaryOp::BitwiseAndAssign, left, right)
            | Expression::Binary(op @ ParserBinaryOp::BitwiseOrAssign, left, right)
            | Expression::Binary(op @ ParserBinaryOp::XorAssign, left, right)
            | Expression::Binary(op @ ParserBinaryOp::ShiftLeftAssign, left, right)
            | Expression::Binary(op @ ParserBinaryOp::ShiftRightAssign, left, right) => {
                self.parse_eager_compound_binary_expression(op, left, right, instructions)
            }
            Expression::Binary(op, left, right) => {
                self.parse_eager_binary_expression(op, left, right, instructions)
            }
            Expression::Var(ident) => Ok(Val::Var(ident.clone())),
            Expression::Assignment(lhs, rhs) => {
                let Expression::Var(ref ident) = **lhs else {
                    return Err(TackyError::InvalidLhsOfAssignment);
                };
                // emit instructions for rhs, then copy into lhs
                let result = self.parse_expression(rhs, instructions)?;
                instructions.push(Instruction::Copy {
                    src: result,
                    dst: Val::Var(ident.clone()),
                });
                Ok(Val::Var(ident.clone()))
            }
            Expression::Conditional {
                condition,
                then,
                else_,
            } => self.parse_conditional(condition, then, else_, instructions),
        }
    }

    fn parse_short_circuit_expression(
        &mut self,
        op: &ParserBinaryOp,
        left: &Expression,
        right: &Expression,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Val, TackyError> {
        use ParserBinaryOp as PBO;
        match op {
            PBO::BinAnd => {
                self.parse_short_circuit_and_expression(BinaryOp::And, left, right, instructions)
            }
            PBO::BinOr => {
                self.parse_short_circuit_or_expression(BinaryOp::Or, left, right, instructions)
            }
            _ => unreachable!(),
        }
    }

    fn parse_conditional(
        &mut self,
        condition: &Expression,
        then: &Expression,
        else_: &Expression,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Val, TackyError> {
        let else_label = self.make_label("else_label");
        let end_label = self.make_label("end_label");
        let cond = self.parse_expression(condition, instructions)?;
        // move cond into a tmp
        let dst1_name = self.make_temporary();
        let dst1 = Val::Var(dst1_name);

        instructions.push(Instruction::Copy {
            src: cond,
            dst: dst1.clone(),
        });
        instructions.push(Instruction::JumpIfZero {
            cond: dst1.clone(),
            target: else_label.clone(),
        });
        // temporary for result: We'll copy the then/else logic into it and return
        // result at the end
        let res_name = self.make_temporary();
        let result = Val::Var(res_name);

        let v1 = self.parse_expression(then, instructions)?;
        instructions.push(Instruction::Copy {
            src: v1,
            dst: result.clone(),
        });

        instructions.push(Instruction::Jump(end_label.clone()));
        instructions.push(Instruction::Label(else_label));
        let v2 = self.parse_expression(else_, instructions)?;
        instructions.push(Instruction::Copy {
            src: v2,
            dst: result.clone(),
        });

        instructions.push(Instruction::Label(end_label));

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
        left: &Expression,
        right: &Expression,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Val, TackyError> {
        assert_eq!(
            op,
            BinaryOp::And,
            "Expected BinaryOp::And when parsing short circuit expr"
        );
        let jump_label = self.make_label("and_expr_false");
        let end_label = self.make_label("and_expr_end");
        let src1 = self.parse_expression(left, instructions)?;
        // move src1 into a tmp
        let dst1_name = self.make_temporary();
        let dst1 = Val::Var(dst1_name);
        instructions.push(Instruction::Copy {
            src: src1,
            dst: dst1.clone(),
        });
        instructions.push(Instruction::JumpIfZero {
            cond: dst1,
            target: jump_label.clone(),
        });
        let src2 = self.parse_expression(right, instructions)?;
        // move src2 into a tmp
        let dst2_name = self.make_temporary();
        let dst2 = Val::Var(dst2_name);
        instructions.push(Instruction::Copy {
            src: src2,
            dst: dst2.clone(),
        });
        instructions.push(Instruction::JumpIfZero {
            cond: dst2,
            target: jump_label.clone(),
        });

        // at this point, neither arm are false so we
        // define our destination location, set it to true
        // and jump to the end label
        let result_name = self.make_temporary();
        let result = Val::Var(result_name);
        instructions.push(Instruction::Copy {
            src: Val::Constant(1),
            dst: result.clone(),
        });
        instructions.push(Instruction::Jump(end_label.clone()));
        // here we create our labels:
        // Create our jump_label label to reach
        // copy False into our result
        // create our end label
        instructions.push(Instruction::Label(jump_label));
        instructions.push(Instruction::Copy {
            src: Val::Constant(0),
            dst: result.clone(),
        });
        instructions.push(Instruction::Label(end_label));
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
        left: &Expression,
        right: &Expression,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Val, TackyError> {
        assert_eq!(
            op,
            BinaryOp::Or,
            "Expected BinaryOp::Or when parsing short circuit expr"
        );
        let jump_label = self.make_label("or_expr_true");
        let end_label = self.make_label("or_expr_end");
        let src1 = self.parse_expression(left, instructions)?;
        // move src1 into a tmp
        let dst1_name = self.make_temporary();
        let dst1 = Val::Var(dst1_name);
        instructions.push(Instruction::Copy {
            src: src1,
            dst: dst1.clone(),
        });
        instructions.push(Instruction::JumpIfNotZero {
            cond: dst1,
            target: jump_label.clone(),
        });
        let src2 = self.parse_expression(right, instructions)?;
        // move src2 into a tmp
        let dst2_name = self.make_temporary();
        let dst2 = Val::Var(dst2_name);
        instructions.push(Instruction::Copy {
            src: src2,
            dst: dst2.clone(),
        });
        instructions.push(Instruction::JumpIfNotZero {
            cond: dst2,
            target: jump_label.clone(),
        });

        // at this point, neither arm are true so we
        // define our destination location, set it to false
        // and jump to the end label
        let result_name = self.make_temporary();
        let result = Val::Var(result_name);
        instructions.push(Instruction::Copy {
            src: Val::Constant(0),
            dst: result.clone(),
        });
        instructions.push(Instruction::Jump(end_label.clone()));
        // here we create our labels:
        // Create our jump_label label to reach
        // copy true into our result
        // create our end label
        instructions.push(Instruction::Label(jump_label));
        instructions.push(Instruction::Copy {
            src: Val::Constant(1),
            dst: result.clone(),
        });
        instructions.push(Instruction::Label(end_label));
        Ok(result)
    }

    fn parse_if_then(
        &mut self,
        condition: &Expression,
        then: &Statement,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), TackyError> {
        let label = self.make_label("end_label");
        let cond = self.parse_expression(condition, instructions)?;
        // move cond into a tmp
        let dst1_name = self.make_temporary();
        let dst1 = Val::Var(dst1_name);
        instructions.push(Instruction::Copy {
            src: cond,
            dst: dst1.clone(),
        });
        instructions.push(Instruction::JumpIfZero {
            cond: dst1.clone(),
            target: label.clone(),
        });
        self.parse_statement(&then, instructions)?;
        instructions.push(Instruction::Label(label));
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
        condition: &Expression,
        then: &Statement,
        else_: &Statement,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), TackyError> {
        let else_label = self.make_label("else_label");
        let end_label = self.make_label("end_label");
        let cond = self.parse_expression(condition, instructions)?;
        // move cond into a tmp
        let dst1_name = self.make_temporary();
        let dst1 = Val::Var(dst1_name);
        instructions.push(Instruction::Copy {
            src: cond,
            dst: dst1.clone(),
        });
        instructions.push(Instruction::JumpIfZero {
            cond: dst1.clone(),
            target: else_label.clone(),
        });
        self.parse_statement(&then, instructions)?;
        instructions.push(Instruction::Jump(end_label.clone()));
        instructions.push(Instruction::Label(else_label));
        self.parse_statement(&else_, instructions)?;
        instructions.push(Instruction::Label(end_label));

        Ok(())
    }

    fn parse_eager_compound_binary_expression(
        &mut self,
        op: &ParserBinaryOp,
        left: &Expression,
        right: &Expression,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Val, TackyError> {
        use ParserBinaryOp as PBO;
        let src1 = self.parse_expression(left, instructions)?;
        // if src1 is a temporary, it's an invalid lvalue.
        // We should only allow variables. To do so, we check if we've emitted
        // the identifier as a temporary.
        let Val::Var(ref ident) = src1 else {
            return Err(TackyError::InvalidLhsOfAssignment);
        };
        if self.created_label(ident) {
            return Err(TackyError::InvalidLhsOfAssignment);
        };

        let src2 = self.parse_expression(right, instructions)?;
        let dst_name = self.make_temporary();
        let dst = Val::Var(dst_name);
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

        instructions.push(Instruction::Binary {
            op: binop,
            src1: src1.clone(),
            src2,
            dst: dst.clone(),
        });
        instructions.push(Instruction::Copy {
            src: dst.clone(),
            dst: src1.clone(),
        });
        Ok(dst)
    }

    fn parse_eager_binary_expression(
        &mut self,
        op: &ParserBinaryOp,
        left: &Expression,
        right: &Expression,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Val, TackyError> {
        use ParserBinaryOp as PBO;
        let src1 = self.parse_expression(left, instructions)?;
        let src2 = self.parse_expression(right, instructions)?;
        let dst_name = self.make_temporary();
        let dst = Val::Var(dst_name);
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
        instructions.push(Instruction::Binary {
            op: binop,
            src1,
            src2,
            dst: dst.clone(),
        });
        Ok(dst)
    }

    fn parse_statement(
        &mut self,
        statement: &Statement,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), TackyError> {
        match statement {
            Statement::Return(body) => {
                let val = self.parse_expression(body, instructions)?;
                instructions.push(Instruction::Ret(val.clone()));
                Ok(())
            }
            Statement::Null => Ok(()),
            Statement::Expr(expr) => {
                self.parse_expression(expr, instructions)?;
                Ok(())
            }
            Statement::If {
                condition,
                then,
                else_: None,
            } => self.parse_if_then(condition, then.as_ref(), instructions),
            Statement::If {
                condition,
                then,
                else_: Some(else_),
            } => self.parse_if_then_else(condition, then.as_ref(), else_.as_ref(), instructions),
            Statement::Goto(lbl) => {
                instructions.push(Instruction::Jump(lbl.into()));
                Ok(())
            }
            Statement::Labelled { label, statement } => {
                instructions.push(Instruction::Label(label.into()));
                self.parse_statement(statement, instructions)?;
                Ok(())
            }
            Statement::Compound(block) => self.parse_block(block, instructions),
        }
    }

    fn parse_instructions(&mut self, block: &Block) -> Result<Vec<Instruction>, TackyError> {
        let mut instructions = vec![];
        self.parse_block(block, &mut instructions)?;
        // temporary hack: always add a Return(Constant(0)) Instruction
        // to handle functions that don't end with a return. If we already
        // have a return, it doesn't run.
        instructions.push(Instruction::Ret(Val::Constant(0)));
        Ok(instructions)
    }

    fn parse_block(
        &mut self,
        block: &Block,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), TackyError> {
        use parser::BlockItem;
        use parser::Declaration;

        let Block(body) = block;
        for body_item in body {
            match body_item {
                BlockItem::Stmt(stmt) => {
                    self.parse_statement(stmt, instructions)?;
                }
                BlockItem::Decl(Declaration {
                    name: _,
                    init: None,
                }) => (),
                BlockItem::Decl(Declaration {
                    name: ident,
                    init: Some(init),
                }) => {
                    // emit instructions for rhs, then copy into lhs
                    let result = self.parse_expression(init, instructions)?;
                    instructions.push(Instruction::Copy {
                        src: result,
                        dst: Val::Var(ident.clone()),
                    });
                }
            }
        }
        Ok(())
    }

    fn make_temporary(&mut self) -> String {
        let c = self.dst_counter;
        let s = format!("tmp.{c}");
        self.dst_counter = c + 1;
        s
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::BinaryOp as ParserBinaryOp;
    use crate::parser::Block;
    use crate::parser::BlockItem;
    use crate::parser::Declaration;
    use crate::parser::Statement;
    use crate::parser::UnaryOp as ParserUnaryOp;
    use crate::parser::AST as ParserAST;

    #[test]
    fn basic_parse() {
        let ast = ParserAST::Program(Box::new(ParserAST::Function {
            name: "main",
            block: Block(vec![BlockItem::Stmt(Statement::Return(
                Expression::Constant(100),
            ))]),
        }));

        let expected = AST::Program(Function {
            name: "main",
            instructions: vec![
                Instruction::Ret(Val::Constant(100)),
                Instruction::Ret(Val::Constant(0)),
            ],
        });

        let tacky = Tacky::new(ast);
        let assembly = tacky.into_ast();
        assert!(assembly.is_ok());
        let assembly = assembly.unwrap();
        assert_eq!(assembly, expected);
    }

    #[test]
    fn unary_op_parse() {
        let ast = ParserAST::Program(Box::new(ParserAST::Function {
            name: "main",
            block: Block(vec![BlockItem::Stmt(Statement::Return(Expression::Unary(
                ParserUnaryOp::Negate,
                Box::new(Expression::Constant(100)),
            )))]),
        }));

        let expected = AST::Program(Function {
            name: "main",
            instructions: vec![
                Instruction::Unary {
                    op: UnaryOp::Negate,
                    src: Val::Constant(100),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::Ret(Val::Var("tmp.0".into())),
                Instruction::Ret(Val::Constant(0)),
            ],
        });

        let tacky = Tacky::new(ast);
        let assembly = tacky.into_ast();
        assert!(assembly.is_ok());
        let assembly = assembly.unwrap();
        assert_eq!(assembly, expected);
    }

    #[test]
    fn complex_unary_parse() {
        let ast = ParserAST::Program(Box::new(ParserAST::Function {
            name: "main",
            block: Block(vec![BlockItem::Stmt(Statement::Return(Expression::Unary(
                ParserUnaryOp::Negate,
                Box::new(Expression::Unary(
                    ParserUnaryOp::Complement,
                    Box::new(Expression::Unary(
                        ParserUnaryOp::Negate,
                        Box::new(Expression::Constant(100)),
                    )),
                )),
            )))]),
        }));

        let expected = AST::Program(Function {
            name: "main",
            instructions: vec![
                Instruction::Unary {
                    op: UnaryOp::Negate,
                    src: Val::Constant(100),
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
                Instruction::Ret(Val::Constant(0)),
            ],
        });

        let tacky = Tacky::new(ast);
        let assembly = tacky.into_ast();
        assert!(assembly.is_ok());
        let assembly = assembly.unwrap();
        assert_eq!(assembly, expected);
    }

    #[test]
    fn complex_binary_parse() {
        let ast = ParserAST::Program(Box::new(ParserAST::Function {
            name: "main",
            block: Block(vec![BlockItem::Stmt(Statement::Return(
                Expression::Binary(
                    ParserBinaryOp::Subtract,
                    Box::new(Expression::Binary(
                        ParserBinaryOp::Multiply,
                        Box::new(Expression::Constant(1)),
                        Box::new(Expression::Constant(2)),
                    )),
                    Box::new(Expression::Binary(
                        ParserBinaryOp::Multiply,
                        Box::new(Expression::Constant(3)),
                        Box::new(Expression::Binary(
                            ParserBinaryOp::Add,
                            Box::new(Expression::Constant(4)),
                            Box::new(Expression::Constant(5)),
                        )),
                    )),
                ),
            ))]),
        }));
        let expected = AST::Program(Function {
            name: "main",
            instructions: vec![
                Instruction::Binary {
                    op: BinaryOp::Multiply,
                    src1: Val::Constant(1),
                    src2: Val::Constant(2),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::Binary {
                    op: BinaryOp::Add,
                    src1: Val::Constant(4),
                    src2: Val::Constant(5),
                    dst: Val::Var("tmp.1".into()),
                },
                Instruction::Binary {
                    op: BinaryOp::Multiply,
                    src1: Val::Constant(3),
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
                Instruction::Ret(Val::Constant(0)),
            ],
        });

        let tacky = Tacky::new(ast);
        let assembly = tacky.into_ast();
        assert!(assembly.is_ok());
        let assembly = assembly.unwrap();
        assert_eq!(assembly, expected);
    }

    #[test]
    fn complex_binary_parse2() {
        let ast = ParserAST::Program(Box::new(ParserAST::Function {
            name: "main",
            block: Block(vec![BlockItem::Stmt(Statement::Return(
                Expression::Binary(
                    ParserBinaryOp::Subtract,
                    Box::new(Expression::Binary(
                        ParserBinaryOp::Divide,
                        Box::new(Expression::Binary(
                            ParserBinaryOp::Multiply,
                            Box::new(Expression::Constant(5)),
                            Box::new(Expression::Constant(4)),
                        )),
                        Box::new(Expression::Constant(2)),
                    )),
                    Box::new(Expression::Binary(
                        ParserBinaryOp::Remainder,
                        Box::new(Expression::Constant(3)),
                        Box::new(Expression::Binary(
                            ParserBinaryOp::Add,
                            Box::new(Expression::Constant(2)),
                            Box::new(Expression::Constant(1)),
                        )),
                    )),
                ),
            ))]),
        }));

        let expected = AST::Program(Function {
            name: "main",
            instructions: vec![
                Instruction::Binary {
                    op: BinaryOp::Multiply,
                    src1: Val::Constant(5),
                    src2: Val::Constant(4),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::Binary {
                    op: BinaryOp::Divide,
                    src1: Val::Var("tmp.0".into()),
                    src2: Val::Constant(2),
                    dst: Val::Var("tmp.1".into()),
                },
                Instruction::Binary {
                    op: BinaryOp::Add,
                    src1: Val::Constant(2),
                    src2: Val::Constant(1),
                    dst: Val::Var("tmp.2".into()),
                },
                Instruction::Binary {
                    op: BinaryOp::Remainder,
                    src1: Val::Constant(3),
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
                Instruction::Ret(Val::Constant(0)),
            ],
        });

        let tacky = Tacky::new(ast);
        let assembly = tacky.into_ast();
        assert!(assembly.is_ok());
        let assembly = assembly.unwrap();
        assert_eq!(assembly, expected);
    }

    #[test]
    fn simple_bitwise() {
        let ast = ParserAST::Program(Box::new(ParserAST::Function {
            name: "main",
            block: Block(vec![BlockItem::Stmt(Statement::Return(
                Expression::Binary(
                    ParserBinaryOp::BitwiseOr,
                    Box::new(Expression::Binary(
                        ParserBinaryOp::Multiply,
                        Box::new(Expression::Constant(5)),
                        Box::new(Expression::Constant(4)),
                    )),
                    Box::new(Expression::Binary(
                        ParserBinaryOp::BitwiseAnd,
                        Box::new(Expression::Binary(
                            ParserBinaryOp::Subtract,
                            Box::new(Expression::Constant(4)),
                            Box::new(Expression::Constant(5)),
                        )),
                        Box::new(Expression::Constant(6)),
                    )),
                ),
            ))]),
        }));

        let expected = AST::Program(Function {
            name: "main",
            instructions: vec![
                Instruction::Binary {
                    op: BinaryOp::Multiply,
                    src1: Val::Constant(5),
                    src2: Val::Constant(4),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::Binary {
                    op: BinaryOp::Subtract,
                    src1: Val::Constant(4),
                    src2: Val::Constant(5),
                    dst: Val::Var("tmp.1".into()),
                },
                Instruction::Binary {
                    op: BinaryOp::BitwiseAnd,
                    src1: Val::Var("tmp.1".into()),
                    src2: Val::Constant(6),
                    dst: Val::Var("tmp.2".into()),
                },
                Instruction::Binary {
                    op: BinaryOp::BitwiseOr,
                    src1: Val::Var("tmp.0".into()),
                    src2: Val::Var("tmp.2".into()),
                    dst: Val::Var("tmp.3".into()),
                },
                Instruction::Ret(Val::Var("tmp.3".into())),
                Instruction::Ret(Val::Constant(0)),
            ],
        });

        let tacky = Tacky::new(ast);
        let assembly = tacky.into_ast();
        assert!(assembly.is_ok());
        let assembly = assembly.unwrap();
        assert_eq!(assembly, expected);
    }

    #[test]
    fn shiftleft() {
        let ast = ParserAST::Program(Box::new(ParserAST::Function {
            name: "main",
            block: Block(vec![BlockItem::Stmt(Statement::Return(
                Expression::Binary(
                    ParserBinaryOp::ShiftLeft,
                    Box::new(Expression::Binary(
                        ParserBinaryOp::Multiply,
                        Box::new(Expression::Constant(5)),
                        Box::new(Expression::Constant(4)),
                    )),
                    Box::new(Expression::Constant(2)),
                ),
            ))]),
        }));
        let expected = AST::Program(Function {
            name: "main",
            instructions: vec![
                Instruction::Binary {
                    op: BinaryOp::Multiply,
                    src1: Val::Constant(5),
                    src2: Val::Constant(4),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::Binary {
                    op: BinaryOp::ShiftLeft,
                    src1: Val::Var("tmp.0".into()),
                    src2: Val::Constant(2),
                    dst: Val::Var("tmp.1".into()),
                },
                Instruction::Ret(Val::Var("tmp.1".into())),
                Instruction::Ret(Val::Constant(0)),
            ],
        });

        let tacky = Tacky::new(ast);
        let assembly = tacky.into_ast();
        assert!(assembly.is_ok());
        let assembly = assembly.unwrap();
        assert_eq!(assembly, expected);
    }

    #[test]
    fn shiftleft_rhs_is_expr() {
        let ast = ParserAST::Program(Box::new(ParserAST::Function {
            name: "main",
            block: Block(vec![BlockItem::Stmt(Statement::Return(
                Expression::Binary(
                    ParserBinaryOp::ShiftLeft,
                    Box::new(Expression::Constant(5)),
                    Box::new(Expression::Binary(
                        ParserBinaryOp::Add,
                        Box::new(Expression::Constant(1)),
                        Box::new(Expression::Constant(2)),
                    )),
                ),
            ))]),
        }));
        let expected = AST::Program(Function {
            name: "main",
            instructions: vec![
                Instruction::Binary {
                    op: BinaryOp::Add,
                    src1: Val::Constant(1),
                    src2: Val::Constant(2),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::Binary {
                    op: BinaryOp::ShiftLeft,
                    src1: Val::Constant(5),
                    src2: Val::Var("tmp.0".into()),
                    dst: Val::Var("tmp.1".into()),
                },
                Instruction::Ret(Val::Var("tmp.1".into())),
                Instruction::Ret(Val::Constant(0)),
            ],
        });

        let tacky = Tacky::new(ast);
        let assembly = tacky.into_ast();
        assert!(assembly.is_ok());
        let assembly = assembly.unwrap();
        assert_eq!(assembly, expected);
    }

    #[test]
    fn test_short_circuit_and() {
        let ast = ParserAST::Program(Box::new(ParserAST::Function {
            name: "main",
            block: Block(vec![BlockItem::Stmt(Statement::Return(
                Expression::Binary(
                    ParserBinaryOp::BinAnd,
                    Box::new(Expression::Constant(5)),
                    Box::new(Expression::Binary(
                        ParserBinaryOp::Add,
                        Box::new(Expression::Constant(1)),
                        Box::new(Expression::Constant(2)),
                    )),
                ),
            ))]),
        }));

        let expected = AST::Program(Function {
            name: "main",
            instructions: vec![
                Instruction::Copy {
                    src: Val::Constant(5),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::JumpIfZero {
                    cond: Val::Var("tmp.0".into()),
                    target: "and_expr_false.0".into(),
                },
                Instruction::Binary {
                    op: BinaryOp::Add,
                    src1: Val::Constant(1),
                    src2: Val::Constant(2),
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
                    src: Val::Constant(1),
                    dst: Val::Var("tmp.3".into()),
                },
                Instruction::Jump("and_expr_end.1".into()),
                Instruction::Label("and_expr_false.0".into()),
                Instruction::Copy {
                    src: Val::Constant(0),
                    dst: Val::Var("tmp.3".into()),
                },
                Instruction::Label("and_expr_end.1".into()),
                Instruction::Ret(Val::Var("tmp.3".into())),
                Instruction::Ret(Val::Constant(0)),
            ],
        });

        let tacky = Tacky::new(ast);
        let assembly = tacky.into_ast();
        assert!(assembly.is_ok());
        let assembly = assembly.unwrap();
        assert_eq!(assembly, expected);
    }

    #[test]
    fn test_short_circuit_or() {
        let ast = ParserAST::Program(Box::new(ParserAST::Function {
            name: "main",
            block: Block(vec![BlockItem::Stmt(Statement::Return(
                Expression::Binary(
                    ParserBinaryOp::BinOr,
                    Box::new(Expression::Constant(5)),
                    Box::new(Expression::Binary(
                        ParserBinaryOp::Add,
                        Box::new(Expression::Constant(1)),
                        Box::new(Expression::Constant(2)),
                    )),
                ),
            ))]),
        }));

        let expected = AST::Program(Function {
            name: "main",
            instructions: vec![
                Instruction::Copy {
                    src: Val::Constant(5),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::JumpIfNotZero {
                    cond: Val::Var("tmp.0".into()),
                    target: "or_expr_true.0".into(),
                },
                Instruction::Binary {
                    op: BinaryOp::Add,
                    src1: Val::Constant(1),
                    src2: Val::Constant(2),
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
                    src: Val::Constant(0),
                    dst: Val::Var("tmp.3".into()),
                },
                Instruction::Jump("or_expr_end.1".into()),
                Instruction::Label("or_expr_true.0".into()),
                Instruction::Copy {
                    src: Val::Constant(1),
                    dst: Val::Var("tmp.3".into()),
                },
                Instruction::Label("or_expr_end.1".into()),
                Instruction::Ret(Val::Var("tmp.3".into())),
                Instruction::Ret(Val::Constant(0)),
            ],
        });

        let tacky = Tacky::new(ast);
        let assembly = tacky.into_ast();
        assert!(assembly.is_ok());
        let assembly = assembly.unwrap();
        assert_eq!(assembly, expected);
    }

    #[test]
    fn basic_declarations() {
        let ast = ParserAST::Program(Box::new(ParserAST::Function {
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
                        ParserBinaryOp::Add,
                        Box::new(Expression::Var("a.0.decl".into())),
                        Box::new(Expression::Var("b.1.decl".into())),
                    )),
                }),
                BlockItem::Stmt(Statement::Return(Expression::Var("c.2.decl".into()))),
            ]),
        }));
        let expected = AST::Program(Function {
            name: "main",
            instructions: vec![
                Instruction::Copy {
                    src: Val::Constant(1),
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
                Instruction::Ret(Val::Constant(0)),
            ],
        });

        let tacky = Tacky::new(ast);
        let assembly = tacky.into_ast();
        assert!(assembly.is_ok());
        let assembly = assembly.unwrap();
        assert_eq!(assembly, expected);
    }

    #[test]
    fn test_compound_assignment() {
        let ast = ParserAST::Program(Box::new(ParserAST::Function {
            name: "main",
            block: Block(vec![
                BlockItem::Decl(Declaration {
                    name: "a".into(),
                    init: Some(Expression::Constant(1)),
                }),
                BlockItem::Stmt(Statement::Expr(Expression::Binary(
                    ParserBinaryOp::AddAssign,
                    Box::new(Expression::Var("a".into())),
                    Box::new(Expression::Constant(2)),
                ))),
                BlockItem::Stmt(Statement::Expr(Expression::Binary(
                    ParserBinaryOp::MinusAssign,
                    Box::new(Expression::Var("a".into())),
                    Box::new(Expression::Constant(2)),
                ))),
                BlockItem::Stmt(Statement::Expr(Expression::Binary(
                    ParserBinaryOp::MultiplyAssign,
                    Box::new(Expression::Var("a".into())),
                    Box::new(Expression::Constant(2)),
                ))),
                BlockItem::Stmt(Statement::Expr(Expression::Binary(
                    ParserBinaryOp::DivideAssign,
                    Box::new(Expression::Var("a".into())),
                    Box::new(Expression::Constant(2)),
                ))),
                BlockItem::Stmt(Statement::Expr(Expression::Binary(
                    ParserBinaryOp::RemainderAssign,
                    Box::new(Expression::Var("a".into())),
                    Box::new(Expression::Constant(2)),
                ))),
                BlockItem::Stmt(Statement::Return(Expression::Var("a".into()))),
            ]),
        }));

        let expected = AST::Program(Function {
            name: "main",
            instructions: vec![
                Instruction::Copy {
                    src: Val::Constant(1),
                    dst: Val::Var("a".into()),
                },
                Instruction::Binary {
                    op: BinaryOp::Add,
                    src1: Val::Var("a".into()),
                    src2: Val::Constant(2),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::Copy {
                    src: Val::Var("tmp.0".into()),
                    dst: Val::Var("a".into()),
                },
                Instruction::Binary {
                    op: BinaryOp::Subtract,
                    src1: Val::Var("a".into()),
                    src2: Val::Constant(2),
                    dst: Val::Var("tmp.1".into()),
                },
                Instruction::Copy {
                    src: Val::Var("tmp.1".into()),
                    dst: Val::Var("a".into()),
                },
                Instruction::Binary {
                    op: BinaryOp::Multiply,
                    src1: Val::Var("a".into()),
                    src2: Val::Constant(2),
                    dst: Val::Var("tmp.2".into()),
                },
                Instruction::Copy {
                    src: Val::Var("tmp.2".into()),
                    dst: Val::Var("a".into()),
                },
                Instruction::Binary {
                    op: BinaryOp::Divide,
                    src1: Val::Var("a".into()),
                    src2: Val::Constant(2),
                    dst: Val::Var("tmp.3".into()),
                },
                Instruction::Copy {
                    src: Val::Var("tmp.3".into()),
                    dst: Val::Var("a".into()),
                },
                Instruction::Binary {
                    op: BinaryOp::Remainder,
                    src1: Val::Var("a".into()),
                    src2: Val::Constant(2),
                    dst: Val::Var("tmp.4".into()),
                },
                Instruction::Copy {
                    src: Val::Var("tmp.4".into()),
                    dst: Val::Var("a".into()),
                },
                Instruction::Ret(Val::Var("a".into())),
                Instruction::Ret(Val::Constant(0)),
            ],
        });

        let tacky = Tacky::new(ast);
        let assembly = tacky.into_ast();
        assert!(assembly.is_ok());
        let assembly = assembly.unwrap();
        assert_eq!(assembly, expected);
    }

    #[test]
    fn test_invalid_lhs_compound_assignment() {
        let ast = ParserAST::Program(Box::new(ParserAST::Function {
            name: "main",
            block: Block(vec![
                BlockItem::Decl(Declaration {
                    name: "a".into(),
                    init: Some(Expression::Constant(10)),
                }),
                BlockItem::Stmt(Statement::Expr(Expression::Binary(
                    ParserBinaryOp::MinusAssign,
                    Box::new(Expression::Binary(
                        ParserBinaryOp::AddAssign,
                        Box::new(Expression::Var("a".into())),
                        Box::new(Expression::Constant(1)),
                    )),
                    Box::new(Expression::Constant(2)),
                ))),
            ]),
        }));
        let tacky = Tacky::new(ast);
        let assembly = tacky.into_ast();
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
        crate::semantic_analysis::resolve(&mut ast).unwrap();
        let asm = Tacky::new(ast);
        let assembly = asm.into_ast();
        let Ok(actual) = assembly else {
            panic!();
        };
        let expected = AST::Program(Function {
            name: "main",
            instructions: vec![
                // first one: if/then
                Instruction::Copy {
                    src: Val::Constant(1),
                    dst: Val::Var("a.0.decl".into()),
                },
                Instruction::Copy {
                    src: Val::Constant(0),
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
                Instruction::Ret(Val::Constant(1)),
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
                Instruction::Ret(Val::Constant(2)),
                Instruction::Jump("end_label.2".into()),
                Instruction::Label("else_label.1".into()),
                Instruction::Ret(Val::Constant(3)),
                Instruction::Label("end_label.2".into()),
                Instruction::Ret(Val::Constant(0)),
            ],
        });
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
        crate::semantic_analysis::resolve(&mut ast).unwrap();
        let asm = Tacky::new(ast);
        let assembly = asm.into_ast();
        let Ok(actual) = assembly else {
            panic!();
        };
        let expected = AST::Program(Function {
            name: "main",
            instructions: vec![
                Instruction::Copy {
                    src: Val::Constant(1),
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
                    src: Val::Constant(1),
                    dst: Val::Var("tmp.1".into()),
                },
                Instruction::Jump("end_label.1".into()),
                Instruction::Label("else_label.0".into()),
                Instruction::Copy {
                    src: Val::Constant(2),
                    dst: Val::Var("tmp.1".into()),
                },
                Instruction::Label("end_label.1".into()),
                Instruction::Ret(Val::Constant(0)),
            ],
        });
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
        crate::semantic_analysis::resolve(&mut ast).unwrap();
        let asm = Tacky::new(ast);
        let assembly = asm.into_ast();
        let Ok(actual) = assembly else {
            panic!();
        };
        let expected = AST::Program(Function {
            name: "main",
            instructions: vec![
                Instruction::Jump("foo.0.label".into()),
                Instruction::Label("foo.0.label".into()),
                Instruction::Binary {
                    op: BinaryOp::Add,
                    src1: Val::Constant(1),
                    src2: Val::Constant(2),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::Ret(Val::Var("tmp.0".into())),
                Instruction::Ret(Val::Constant(0)),
            ],
        });
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
        crate::semantic_analysis::resolve(&mut ast).unwrap();
        let asm = Tacky::new(ast);
        let assembly = asm.into_ast();
        let Ok(actual) = assembly else {
            panic!();
        };
        let expected = AST::Program(Function {
            name: "main",
            instructions: vec![
                Instruction::Copy {
                    src: Val::Constant(1),
                    dst: Val::Var("tmp.0".into()),
                },
                Instruction::JumpIfZero {
                    cond: Val::Var("tmp.0".into()),
                    target: "else_label.0".into(),
                },
                Instruction::Ret(Val::Constant(1)),
                Instruction::Jump("end_label.1".into()),
                Instruction::Label("else_label.0".into()),
                Instruction::Ret(Val::Constant(2)),
                Instruction::Label("end_label.1".into()),
                Instruction::Copy {
                    src: Val::Constant(3),
                    dst: Val::Var("x.0.decl".into()),
                },
                Instruction::Ret(Val::Var("x.0.decl".into())),
                Instruction::Ret(Val::Constant(0)),
            ],
        });
        assert_eq!(expected, actual);
    }
}
