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
pub enum AST<'a> {
    Program(Box<AST<'a>>),
    Function {
        name: &'a str,
        instructions: Vec<Instruction>,
    },
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

#[derive(Debug, PartialEq)]
pub struct Asm<'a> {
    parser: ParserAST<'a>,
}

impl<'a> Asm<'a> {
    pub fn new(parser: ParserAST<'a>) -> Self {
        Self { parser }
    }

    pub fn into_ast(self) -> Result<AST<'a>, AsmError> {
        self.parse_program(&self.parser)
    }

    fn parse_program(&self, parser: &ParserAST<'a>) -> Result<AST<'a>, AsmError> {
        match parser {
            ParserAST::Program(func) => {
                let func = self.parse_function(func)?;
                Ok(AST::Program(Box::new(func)))
            }
            _ => Err(AsmError::MissingProgram),
        }
    }

    fn parse_function(&self, parser: &ParserAST<'a>) -> Result<AST<'a>, AsmError> {
        match parser {
            ParserAST::Function { name, body } => {
                let instructions = self.parse_instructions(&*body)?;
                Ok(AST::Function { name, instructions })
            }
            _ => Err(AsmError::MissingFunction),
        }
    }

    fn parse_expression(&self, expr: &Expression) -> Result<Vec<Instruction>, AsmError> {
        match expr {
            Expression::Constant(imm) => {
                let src = Operand::Imm(*imm);
                let dst = Operand::Reg(Register::EAX);
                let instructions = vec![Instruction::Mov(src, dst)];
                Ok(instructions)
            }
        }
    }

    fn parse_instructions(&self, parser: &ParserAST<'a>) -> Result<Vec<Instruction>, AsmError> {
        match parser {
            ParserAST::Return(body) => {
                let mut instructions = vec![];
                for instruction in self.parse_expression(body)? {
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

        let expected = AST::Program(Box::new(AST::Function {
            name: "main",
            instructions: vec![
                Instruction::Mov(Operand::Imm(100), Operand::Reg(Register::EAX)),
                Instruction::Ret,
            ],
        }));

        let asm = Asm::new(ast);
        let assembly = asm.into_ast();
        assert!(assembly.is_ok());
        let assembly = assembly.unwrap();
        assert_eq!(assembly, expected);
    }
}
