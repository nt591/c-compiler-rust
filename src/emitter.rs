// takes the asm parser and emits X64
// only works on my Mac, so do with that what you will.

use crate::asm;
use crate::Asm;

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

    fn emit_function(func: &asm::Function<'a>) -> Vec<String> {
        let asm::Function { name, instructions } = func;
        let mut codegen = vec![];
        codegen.push(format!("  .globl _{name}"));
        codegen.push(format!("_{name}:"));
        codegen.push(format!("  pushq  %rbp"));
        codegen.push(format!("  movq   %rsp, %rbp"));
        for instruction in instructions {
            let ins = Self::emit_instructions(instruction);
            for instruction in ins {
                codegen.push(format!("  {instruction}"));
            }
        }
        codegen
    }

    fn emit_instructions(instruction: &asm::Instruction) -> Vec<String> {
        match instruction {
            asm::Instruction::Ret => {
                let mut instructions = vec![];
                instructions.push("movq   %rbp, %rsp".to_string());
                instructions.push("popq   %rbp".to_string());
                instructions.push("ret".to_string());
                instructions
            }
            asm::Instruction::Mov(src, dst) => {
                let src = Self::emit_op(src);
                let dst = Self::emit_op(dst);
                vec![format!("movl   {src}, {dst}")]
            }
            asm::Instruction::Unary(unary, operand) => {
                let uop = Self::emit_unary(unary);
                let op = Self::emit_op(operand);
                vec![format!("{uop}   {op}")]
            }
            asm::Instruction::AllocateStack(n) => {
                vec![format!("subq   ${n}, %rsp")]
            }
        }
    }

    fn emit_op(op: &asm::Operand) -> String {
        match op {
            asm::Operand::Reg(reg) => Self::emit_register(reg),
            asm::Operand::Imm(imm) => format!("${imm}"),
            _ => todo!(),
        }
    }

    fn emit_unary(uop: &asm::UnaryOp) -> String {
        match uop {
            asm::UnaryOp::Not => "notl".into(),
            asm::UnaryOp::Neg => "negl".into(),
        }
    }

    fn emit_register(reg: &asm::Register) -> String {
        match reg {
            asm::Register::EAX => "%eax".to_string(),
            asm::Register::R10 => "%r10".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_emit() {
        let ast = Asm::Program(asm::Function {
            name: "main",
            instructions: vec![
                asm::Instruction::Mov(asm::Operand::Imm(100), asm::Operand::Reg(asm::Register::EAX)),
                asm::Instruction::Ret,
            ],
        });

        let expected = r#"  .globl _main
_main:
  pushq  %rbp
  movq   %rsp, %rbp
  movl   $100, %eax
  movq   %rbp, %rsp
  popq   %rbp
  ret
"#;

        let emitter = Emitter::new(ast);
        let actual = emitter.emit();
        assert_eq!(expected, actual);
    }
}
