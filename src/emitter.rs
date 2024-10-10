// takes the asm parser and emits X64
// only works on my Mac, so do with that what you will.

use crate::asm::Asm;
use crate::asm::Function;
use crate::asm::Instruction;
use crate::asm::Operand;
use crate::asm::Register;

// lifetime bound to source text.
pub struct Emitter<'a>(Asm<'a>);

impl<'a> Emitter<'a> {
    pub fn new(asm: Asm<'a>) -> Self {
        Self(asm)
    }

    // todo: write to a file
    pub fn emit(self) -> String {
        let mut code = Self::emit_code(&self.0).join("\n");
        code.push('\n');
        code
    }

    fn emit_code(asm: &Asm<'a>) -> Vec<String> {
        match asm {
            Asm::Program(func) => Self::emit_function(func),
        }
    }

    fn emit_function(func: &Function<'a>) -> Vec<String> {
        let Function { name, instructions } = func;
        let mut codegen = vec![];
        codegen.push(format!("  .globl _{name}"));
        codegen.push(format!("_{name}:"));
        for instruction in instructions {
            let instruction = Self::emit_instruction(instruction);
            codegen.push(format!("  {instruction}"));
        }
        codegen
    }

    fn emit_instruction(instruction: &Instruction) -> String {
        match instruction {
            Instruction::Ret => "ret".to_string(),
            Instruction::Mov(src, dst) => {
                let src = Self::emit_op(src);
                let dst = Self::emit_op(dst);
                format!("movl {src}, {dst}")
            }
            _ => todo!(),
        }
    }

    fn emit_op(op: &Operand) -> String {
        match op {
            Operand::Reg(reg) => Self::emit_register(reg),
            Operand::Imm(imm) => format!("${imm}"),
            _ => todo!(),
        }
    }

    fn emit_register(reg: &Register) -> String {
        match reg {
            Register::EAX => "%eax".to_string(),
            Register::R10 => "%r10".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asm::Asm;
    use crate::asm::Function;

    #[test]
    fn basic_emit() {
        let ast = Asm::Program(Function {
            name: "main",
            instructions: vec![
                Instruction::Mov(Operand::Imm(100), Operand::Reg(Register::EAX)),
                Instruction::Ret,
            ],
        });

        let expected = r#"  .globl _main
_main:
  movl $100, %eax
  ret
"#;

        let emitter = Emitter::new(ast);
        let actual = emitter.emit();
        assert_eq!(expected, actual);
    }
}
