// Responsible for taking a parser AST
// and converting to an assembly AST
use crate::parser::Expression;
use crate::parser::AST as ParserAST;
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
pub enum Instruction {
    Mov(Operand, Operand),
    Ret,
}

#[derive(Debug, PartialEq)]
pub enum Operand {
    Imm(usize),
    Reg(Register),
}

#[derive(Debug, PartialEq)]
pub enum Register {
    EAX,
}

impl<'a> Asm<'a> {
    pub fn from_parser(parser: ParserAST<'a>) -> Result<Asm<'a>, AsmError> {
        Self::parse_program(&parser)
    }

    fn parse_program(parser: &ParserAST<'a>) -> Result<Asm<'a>, AsmError> {
        match parser {
            ParserAST::Program(func) => {
                let func = Self::parse_function(func)?;
                Ok(Asm::Program(func))
            }
            _ => Err(AsmError::MissingProgram),
        }
    }

    fn parse_function(parser: &ParserAST<'a>) -> Result<Function<'a>, AsmError> {
        match parser {
            ParserAST::Function { name, body } => {
                let instructions = Self::parse_instructions(&*body)?;
                Ok(Function { name, instructions })
            }
            _ => Err(AsmError::MissingFunction),
        }
    }

    fn parse_expression(expr: &Expression) -> Result<Vec<Instruction>, AsmError> {
        match expr {
            Expression::Constant(imm) => {
                let src = Operand::Imm(*imm);
                let dst = Operand::Reg(Register::EAX);
                let instructions = vec![Instruction::Mov(src, dst)];
                Ok(instructions)
            }
            Expression::Unary(_op, _exp) => todo!(),
        }
    }

    fn parse_instructions(parser: &ParserAST<'a>) -> Result<Vec<Instruction>, AsmError> {
        match parser {
            ParserAST::Return(body) => {
                let mut instructions = vec![];
                for instruction in Self::parse_expression(body)? {
                    instructions.push(instruction);
                }
                instructions.push(Instruction::Ret);
                Ok(instructions)
            }
            _ => Err(AsmError::InvalidInstruction),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::AST as ParserAST;
    #[test]
    fn basic_parse() {
        let ast = ParserAST::Program(Box::new(ParserAST::Function {
            name: "main",
            body: Box::new(ParserAST::Return(Expression::Constant(100))),
        }));

        let expected = Asm::Program(Function {
            name: "main",
            instructions: vec![
                Instruction::Mov(Operand::Imm(100), Operand::Reg(Register::EAX)),
                Instruction::Ret,
            ],
        });

        let assembly = Asm::from_parser(ast);
        assert!(assembly.is_ok());
        let assembly = assembly.unwrap();
        assert_eq!(assembly, expected);
    }
}
