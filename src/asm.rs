// Responsible for taking a TACKY AST
// and converting to an assembly AST
use crate::tacky;
use thiserror::Error;

#[derive(Debug, PartialEq, Error)]
pub enum AsmError {
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
pub enum Asm<'a> {
    Program(Function<'a>),
}

#[derive(Debug, PartialEq)]
pub struct Function<'a> {
    pub name: &'a str,
    pub instructions: Vec<Instruction>,
}

#[derive(Debug, PartialEq)]
pub enum UnaryOp {
    Not,
    Neg,
}

#[derive(Debug, PartialEq)]
pub enum Instruction {
    Mov(Operand, Operand),
    Unary(UnaryOp, Operand),
    AllocateStack(usize),
    Ret,
}

#[derive(Debug, PartialEq)]
pub enum Operand {
    Imm(usize),
    Reg(Register),
    Pseudo(String),
    Stack(i32),
}

#[derive(Debug, PartialEq)]
pub enum Register {
    EAX,
    R10,
}

impl<'a> Asm<'a> {
    pub fn from_tacky(tacky: tacky::AST<'a>) -> Asm<'a> {
        Self::parse_program(&tacky)
    }

    fn parse_program(tacky: &tacky::AST<'a>) -> Asm<'a> {
        match tacky {
            tacky::AST::Program(func) => {
                let func = Self::parse_function(func);
                Asm::Program(func)
            }
        }
    }

    fn parse_function(func: &tacky::Function<'a>) -> Function<'a> {
        let tacky::Function { name, instructions } = func;
        let instructions = Self::parse_instructions(&instructions);
        Function { name, instructions }
    }

    fn parse_instructions(ins: &[tacky::Instruction]) -> Vec<Instruction> {
        ins.iter()
            .flat_map(|instruction| match instruction {
                tacky::Instruction::Ret(val) => vec![
                    Instruction::Mov(val.into(), Operand::Reg(Register::EAX)),
                    Instruction::Ret,
                ],
                tacky::Instruction::Unary { op, src, dst } => vec![
                    Instruction::Mov(src.into(), dst.into()),
                    Instruction::Unary(op.into(), dst.into()),
                ],
            })
            .collect::<Vec<_>>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tacky;
    #[test]
    fn basic_parse() {
        let ast = tacky::AST::Program(tacky::Function {
            name: "main",
            instructions: vec![tacky::Instruction::Ret(tacky::Val::Constant(100))],
        });

        let expected = Asm::Program(Function {
            name: "main",
            instructions: vec![
                Instruction::Mov(Operand::Imm(100), Operand::Reg(Register::EAX)),
                Instruction::Ret,
            ],
        });

        let assembly = Asm::from_tacky(ast);
        assert_eq!(assembly, expected);
    }

    #[test]
    fn parse_with_pseudos() {
        let ast = tacky::AST::Program(tacky::Function {
            name: "main",
            instructions: vec![
                tacky::Instruction::Unary {
                    op: tacky::UnaryOp::Negate,
                    src: tacky::Val::Constant(100),
                    dst: tacky::Val::Var("tmp.0".into()),
                },
                tacky::Instruction::Ret(tacky::Val::Var("tmp.0".into())),
            ],
        });

        let expected = Asm::Program(Function {
            name: "main",
            instructions: vec![
                Instruction::Mov(Operand::Imm(100), Operand::Pseudo("tmp.0".into())),
                Instruction::Unary(UnaryOp::Neg, Operand::Pseudo("tmp.0".into())),
                Instruction::Mov(Operand::Pseudo("tmp.0".into()), Operand::Reg(Register::EAX)),
                Instruction::Ret,
            ],
        });

        let assembly = Asm::from_tacky(ast);
        assert_eq!(assembly, expected);
    }
}

// some niceties. Maybe move to a from.rs
impl From<tacky::UnaryOp> for UnaryOp {
    fn from(op: tacky::UnaryOp) -> Self {
        match op {
            tacky::UnaryOp::Complement => UnaryOp::Not,
            tacky::UnaryOp::Negate => UnaryOp::Neg,
        }
    }
}

impl From<tacky::Val> for Operand {
    fn from(v: tacky::Val) -> Self {
        match v {
            tacky::Val::Constant(imm) => Operand::Imm(imm),
            tacky::Val::Var(ident) => Operand::Pseudo(ident),
        }
    }
}

impl From<&tacky::UnaryOp> for UnaryOp {
    fn from(op: &tacky::UnaryOp) -> Self {
        match op {
            &tacky::UnaryOp::Complement => UnaryOp::Not,
            &tacky::UnaryOp::Negate => UnaryOp::Neg,
        }
    }
}

impl From<&tacky::Val> for Operand {
    fn from(v: &tacky::Val) -> Self {
        match v {
            tacky::Val::Constant(imm) => Operand::Imm(*imm),
            tacky::Val::Var(ident) => Operand::Pseudo(ident.clone()),
        }
    }
}
