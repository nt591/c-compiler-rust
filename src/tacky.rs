// implements a Parser AST -> TACKY AST for the IR
// Mostly copied from asm.rs
//
use crate::parser::Expression;
use crate::parser::UnaryOp as ParserUnaryOp;
use crate::parser::AST as ParserAST;
use thiserror::Error;

#[derive(Debug, PartialEq, Error)]
pub enum TackyError {
    #[error("Expected program")]
    MissingProgram,
    #[error("Expected function")]
    MissingFunction,
    #[error("Invalid Instruction")]
    InvalidInstruction,
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
    Unary { op: UnaryOp, src: Val, dst: Val },
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
}

#[derive(Debug, PartialEq)]
pub struct Tacky<'a> {
    parser: ParserAST<'a>,
    dst_counter: u16,
}

impl<'a> Tacky<'a> {
    pub fn new(parser: ParserAST<'a>) -> Self {
        Self {
            parser,
            dst_counter: 0,
        }
    }

    pub fn into_ast(mut self) -> Result<AST<'a>, TackyError> {
        // Imperfect, but I need to be able to borrow the parser
        let parser =
            std::mem::replace(&mut self.parser, ParserAST::Return(Expression::Constant(0)));
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
            ParserAST::Function { name, body } => {
                let instructions = self.parse_instructions(&*body)?;
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
                };
                instructions.push(Instruction::Unary {
                    op: unary_op,
                    src,
                    dst: dst.clone(),
                });
                Ok(dst)
            }
        }
    }

    fn parse_instructions(
        &mut self,
        parser: &ParserAST<'a>,
    ) -> Result<Vec<Instruction>, TackyError> {
        match parser {
            ParserAST::Return(body) => {
                let mut instructions = vec![];
                let val = self.parse_expression(body, &mut instructions)?;
                instructions.push(Instruction::Ret(val));
                Ok(instructions)
            }
            _ => Err(TackyError::InvalidInstruction),
        }
    }

    fn make_temporary(&mut self) -> String {
        let c = self.dst_counter;
        let s = format!("tmp.{c}");
        self.dst_counter = c + 1;
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::UnaryOp as ParserUnaryOp;
    use crate::parser::AST as ParserAST;
    #[test]
    fn basic_parse() {
        let ast = ParserAST::Program(Box::new(ParserAST::Function {
            name: "main",
            body: Box::new(ParserAST::Return(Expression::Constant(100))),
        }));

        let expected = AST::Program(Function {
            name: "main",
            instructions: vec![Instruction::Ret(Val::Constant(100))],
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
            body: Box::new(ParserAST::Return(Expression::Unary(
                ParserUnaryOp::Negate,
                Box::new(Expression::Constant(100)),
            ))),
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
            body: Box::new(ParserAST::Return(Expression::Unary(
                ParserUnaryOp::Negate,
                Box::new(Expression::Unary(
                    ParserUnaryOp::Complement,
                    Box::new(Expression::Unary(
                        ParserUnaryOp::Negate,
                        Box::new(Expression::Constant(100)),
                    )),
                )),
            ))),
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
            ],
        });

        let tacky = Tacky::new(ast);
        let assembly = tacky.into_ast();
        assert!(assembly.is_ok());
        let assembly = assembly.unwrap();
        assert_eq!(assembly, expected);
    }
}
