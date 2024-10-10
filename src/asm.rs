// Responsible for taking a TACKY AST
// and converting to an assembly AST
use crate::tacky;
use std::collections::HashMap;
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

#[derive(Debug, PartialEq, Clone)]
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

// implement clone so our mapping of Tacky Var
// to Pseudo can always return an owned value
#[derive(Debug, PartialEq, Clone)]
pub enum Operand {
    Imm(usize),
    Reg(Register),
    Pseudo(String),
    Stack(i32),
}

#[derive(Debug, PartialEq, Clone)]
pub enum Register {
    EAX,
    R10,
}

#[derive(Debug, Default)]
struct AsmGenerator {
    stack_offset: i32,     
    identifiers: HashMap<String, Operand>, // Tacky var ident -> Pseudo(string)
}

impl<'a> Asm<'a> {
    pub fn from_tacky(tacky: tacky::AST<'a>) -> Asm<'a> {
        let asm = Self::parse_program(&tacky);
        let mut generator = AsmGenerator::default();
        Self::fixup_pseudos(asm, &mut generator)
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

    fn fixup_pseudos(asm: Asm<'a>, gen: &mut AsmGenerator) -> Asm<'a> {
        match asm {
            Asm::Program(func) => Asm::Program(Self::fixup_pseudos_in_function(&func, gen)),
        }
    }
    fn fixup_pseudos_in_function(
        func: &Function<'a>,
        gen: &mut AsmGenerator,
    ) -> Function<'a> {
        let Function { name, instructions } = func;
        let instructions = Self::fixup_pseudos_in_instructions(instructions, gen);
        Function { name, instructions }
    }

    fn fixup_pseudos_in_instructions(
        ins: &[Instruction],
        gen: &mut AsmGenerator,
    ) -> Vec<Instruction> {
        ins.iter()
            .map(|instruction| match instruction {
                Instruction::Mov(src, dst) => {
                    let src = Self::replace_pseudo_in_op(src, gen);
                    let dst = Self::replace_pseudo_in_op(dst, gen);
                    Instruction::Mov(src, dst)
                }
                Instruction::Unary(op, dst) => {
                    let dst = Self::replace_pseudo_in_op(dst, gen);
                    Instruction::Unary(op.clone(), dst)
                }
                Instruction::AllocateStack(n) => Instruction::AllocateStack(*n),
                Instruction::Ret => Instruction::Ret,
            })
            .collect::<Vec<_>>()
    }

    fn replace_pseudo_in_op(op: &Operand, gen: &mut AsmGenerator) -> Operand {
        match op {
            Operand::Pseudo(var) => gen
                .identifiers
                .entry(var.clone())
                .or_insert_with(|| {
                    let next_offset = gen.stack_offset - 4;
                    gen.stack_offset = next_offset;
                    Operand::Stack(
                        next_offset
                    )
                })
                .clone(),
            o => o.clone(), //no transformation otherwise
        }
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
                Instruction::Mov(Operand::Imm(100), Operand::Stack(-4)),
                Instruction::Unary(UnaryOp::Neg, Operand::Stack(-4)),
                Instruction::Mov(Operand::Stack(-4), Operand::Reg(Register::EAX)),
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
