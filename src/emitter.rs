// takes the asm parser and emits X64
// only works on my Mac, so do with that what you will.

use crate::asm::Instruction;
use crate::asm::Operand;
use crate::asm::Register;
use crate::asm::AST;

// lifetime bound to source text.
pub struct Emitter<'a>(AST<'a>);

impl<'a> Emitter<'a> {
    pub fn new(ast: AST<'a>) -> Self {
        Self(ast)
    }

    // todo: write to a file
    pub fn emit(self) -> String {
        Self::emit_code(&self.0).join("\n")
    }

    fn emit_code(ast: &AST<'a>) -> Vec<String> {
        match ast {
            AST::Program(func) => Self::emit_code(func),
            AST::Function { name, instructions } => {
                let mut codegen = vec![];
                codegen.push(format!("  .globl {name}"));
                codegen.push(format!("{name}:"));
                for instruction in instructions {
                    let instruction = Self::emit_instruction(instruction);
                    codegen.push(format!("  {instruction}"));
                }
                codegen
            }
        }
    }

    fn emit_instruction(instruction: &Instruction) -> String {
        match instruction {
            Instruction::Ret => "ret".to_string(),
            Instruction::Mov(src, dst) => {
                let src = Self::emit_op(src);
                let dst = Self::emit_op(dst);
                format!("movl {src}, {dst}")
            }
        }
    }

    fn emit_op(op: &Operand) -> String {
        match op {
            Operand::Reg(reg) => Self::emit_register(reg),
            Operand::Imm(imm) => format!("${imm}"),
        }
    }

    fn emit_register(reg: &Register) -> String {
        match reg {
            Register::EAX => "%eax".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::AST as ParserAST;

    #[test]
    fn basic_emit() {
        let ast = AST::Program(Box::new(AST::Function {
            name: "main",
            instructions: vec![
                Instruction::Mov(Operand::Imm(100), Operand::Reg(Register::EAX)),
                Instruction::Ret,
            ],
        }));

        let expected = r#"  .globl main
main:
  movl $100, %eax
  ret"#;
        
    let emitter = Emitter::new(ast);
    let actual = emitter.emit();
    assert_eq!(expected, actual);
    }
}
