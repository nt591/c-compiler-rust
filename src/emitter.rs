// takes the asm parser and emits X64
// only works on my Mac, so do with that what you will.

use crate::asm;
use crate::Asm;
use std::io::Write;

// lifetime bound to source text.
pub struct Emitter<'a>(Asm<'a>);

impl<'a> Emitter<'a> {
    pub fn new(asm: Asm<'a>) -> Self {
        Self(asm)
    }

    // todo: write to a file
    pub fn emit<W: Write>(self, output: &mut W) -> std::io::Result<()> {
        Self::emit_code(&self.0, output)
    }

    fn emit_code<W: Write>(asm: &Asm<'a>, output: &mut W) -> std::io::Result<()> {
        match asm {
            Asm::Program(func) => Self::emit_function(func, output)?,
        }
        Ok(())
    }

    fn emit_comment<W: Write>(comment: &str, output: &mut W) -> std::io::Result<()> {
        writeln!(output, "                       # {}", comment)
    }

    fn emit_function<W: Write>(func: &asm::Function<'a>, output: &mut W) -> std::io::Result<()> {
        let asm::Function { name, instructions } = func;
        writeln!(output, "  .globl _{}", name)?;
        writeln!(output, "_{}:", name)?;
        Self::emit_comment("FUNCTION PROLOGUE", output)?;
        writeln!(output, "{}", "  pushq  %rbp")?;
        writeln!(output, "{}", "  movq   %rsp, %rbp")?;
        for instruction in instructions {
            Self::emit_instructions(instruction, output)?;
        }
        Ok(())
    }

    fn emit_instructions<W: Write>(
        instruction: &asm::Instruction,
        output: &mut W,
    ) -> std::io::Result<()> {
        match instruction {
            asm::Instruction::Ret => {
                Self::emit_comment("RESET REGISTERS", output)?;
                writeln!(output, "  movq   %rbp, %rsp")?;
                writeln!(output, "  popq   %rbp")?;
                writeln!(output, "  ret")?;
            }
            asm::Instruction::Mov(src, dst) => {
                write!(output, "  movl   ")?;
                Self::emit_op(src, output)?;
                write!(output, ", ")?;
                Self::emit_op(dst, output)?;
                write!(output, "\n")?;
            }
            asm::Instruction::Unary(unary, operand) => {
                write!(output, "  ")?;
                Self::emit_unary(unary, output)?;
                write!(output, "   ")?;
                Self::emit_op(operand, output)?;
                write!(output, "\n")?;
            }
            asm::Instruction::Binary(binop, src, dst) => {
                write!(output, "  ")?;
                Self::emit_binary(binop, output)?;
                // assume emit_binary handles proper space formatting!
                Self::emit_op(src, output)?;
                write!(output, ", ")?;
                Self::emit_op(dst, output)?;
                write!(output, "\n")?;
            }
            asm::Instruction::AllocateStack(n) => {
                writeln!(output, "  subq   ${}, %rsp", n)?;
            }
            asm::Instruction::Cdq => {
                writeln!(output, "  cdq")?;
            }
            asm::Instruction::Idiv(operand) => {
                write!(output, "  idivl  ")?;
                Self::emit_op(operand, output)?;
                write!(output, "\n")?;
            }
        }
        Ok(())
    }

    fn emit_op<W: Write>(op: &asm::Operand, output: &mut W) -> std::io::Result<()> {
        match op {
            asm::Operand::Reg(reg) => Self::emit_register(reg, output)?,
            asm::Operand::Imm(imm) => write!(output, "${}", imm)?,
            asm::Operand::Stack(n) => write!(output, "{}(%rbp)", n)?,
            _ => todo!(),
        }

        Ok(())
    }

    fn emit_unary<W: Write>(uop: &asm::UnaryOp, output: &mut W) -> std::io::Result<()> {
        match uop {
            asm::UnaryOp::Not => write!(output, "notl")?,
            asm::UnaryOp::Neg => write!(output, "negl")?,
        }
        Ok(())
    }

    fn emit_binary<W: Write>(binop: &asm::BinaryOp, output: &mut W) -> std::io::Result<()> {
        match binop {
            asm::BinaryOp::Add => write!(output, "addl   ")?,
            asm::BinaryOp::Mult => write!(output, "imull  ")?,
            asm::BinaryOp::Sub => write!(output, "subl   ")?,
            asm::BinaryOp::And => write!(output, "andl   ")?,
            asm::BinaryOp::Or => write!(output, "orl    ")?,
            asm::BinaryOp::Xor => write!(output, "xorl   ")?,
        }
        Ok(())
    }

    fn emit_register<W: Write>(reg: &asm::Register, output: &mut W) -> std::io::Result<()> {
        match reg {
            asm::Register::EAX => write!(output, "{}", "%eax"),
            asm::Register::EDX => write!(output, "{}", "%edx"),
            asm::Register::R10 => write!(output, "{}", "%r10d"),
            asm::Register::R11 => write!(output, "{}", "%r11d"),
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
                asm::Instruction::Mov(
                    asm::Operand::Imm(100),
                    asm::Operand::Reg(asm::Register::EAX),
                ),
                asm::Instruction::Ret,
            ],
        });

        let expected = r#"  .globl _main
_main:
                       # FUNCTION PROLOGUE
  pushq  %rbp
  movq   %rsp, %rbp
  movl   $100, %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
"#;

        let emitter = Emitter::new(ast);
        let mut vec = Vec::new();
        emitter.emit(&mut vec).expect("Could not write to string");
        let actual = String::from_utf8(vec).expect("Got invalid UTF-8");

        // annoying, but gotta
        assert_eq!(expected, actual);
    }

    #[test]
    fn complex_emit() {
        let ast = asm::Asm::Program(asm::Function {
            name: "main",
            instructions: vec![
                asm::Instruction::AllocateStack(-12),
                asm::Instruction::Mov(asm::Operand::Imm(100), asm::Operand::Stack(-4)),
                asm::Instruction::Unary(asm::UnaryOp::Neg, asm::Operand::Stack(-4)),
                asm::Instruction::Mov(
                    asm::Operand::Stack(-4),
                    asm::Operand::Reg(asm::Register::R10),
                ),
                asm::Instruction::Mov(
                    asm::Operand::Reg(asm::Register::R10),
                    asm::Operand::Stack(-8),
                ),
                asm::Instruction::Unary(asm::UnaryOp::Not, asm::Operand::Stack(-8)),
                asm::Instruction::Mov(
                    asm::Operand::Stack(-8),
                    asm::Operand::Reg(asm::Register::R10),
                ),
                asm::Instruction::Mov(
                    asm::Operand::Reg(asm::Register::R10),
                    asm::Operand::Stack(-12),
                ),
                asm::Instruction::Unary(asm::UnaryOp::Neg, asm::Operand::Stack(-12)),
                asm::Instruction::Mov(
                    asm::Operand::Stack(-12),
                    asm::Operand::Reg(asm::Register::EAX),
                ),
                asm::Instruction::Ret,
            ],
        });
        let expected = r#"  .globl _main
_main:
                       # FUNCTION PROLOGUE
  pushq  %rbp
  movq   %rsp, %rbp
  subq   $-12, %rsp
  movl   $100, -4(%rbp)
  negl   -4(%rbp)
  movl   -4(%rbp), %r10d
  movl   %r10d, -8(%rbp)
  notl   -8(%rbp)
  movl   -8(%rbp), %r10d
  movl   %r10d, -12(%rbp)
  negl   -12(%rbp)
  movl   -12(%rbp), %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
"#;
        let emitter = Emitter::new(ast);
        let mut vec = Vec::new();
        emitter.emit(&mut vec).expect("Could not write to string");
        let actual = String::from_utf8(vec).expect("Got invalid UTF-8");

        assert_eq!(expected, actual);
    }

    #[test]
    fn binary_operators() {
        let ast = asm::Asm::Program(asm::Function {
            name: "main",
            instructions: vec![
                asm::Instruction::AllocateStack(-16),
                // tmp0 = 1 * 2
                asm::Instruction::Mov(asm::Operand::Imm(1), asm::Operand::Stack(-4)),
                asm::Instruction::Mov(
                    asm::Operand::Stack(-4),
                    asm::Operand::Reg(asm::Register::R11),
                ),
                asm::Instruction::Binary(
                    asm::BinaryOp::Mult,
                    asm::Operand::Imm(2),
                    asm::Operand::Reg(asm::Register::R11),
                ),
                asm::Instruction::Mov(
                    asm::Operand::Reg(asm::Register::R11),
                    asm::Operand::Stack(-4),
                ),
                // tmp1 = 4 + 5
                asm::Instruction::Mov(asm::Operand::Imm(4), asm::Operand::Stack(-8)),
                asm::Instruction::Binary(
                    asm::BinaryOp::Add,
                    asm::Operand::Imm(5),
                    asm::Operand::Stack(-8),
                ),
                // tmp2 = 3 % tmp1
                asm::Instruction::Mov(asm::Operand::Imm(3), asm::Operand::Reg(asm::Register::EDX)),
                asm::Instruction::Cdq,
                asm::Instruction::Idiv(asm::Operand::Stack(-8)),
                asm::Instruction::Mov(
                    asm::Operand::Reg(asm::Register::EDX),
                    asm::Operand::Stack(-12),
                ),
                // tmp3 = tmp0 / tmp2
                asm::Instruction::Mov(
                    asm::Operand::Stack(-4),
                    asm::Operand::Reg(asm::Register::EAX),
                ),
                asm::Instruction::Cdq,
                asm::Instruction::Idiv(asm::Operand::Stack(-12)),
                asm::Instruction::Mov(
                    asm::Operand::Reg(asm::Register::EAX),
                    asm::Operand::Stack(-16),
                ),
                // return
                asm::Instruction::Mov(
                    asm::Operand::Stack(-16),
                    asm::Operand::Reg(asm::Register::EAX),
                ),
                asm::Instruction::Ret,
            ],
        });

        let expected = r#"  .globl _main
_main:
                       # FUNCTION PROLOGUE
  pushq  %rbp
  movq   %rsp, %rbp
  subq   $-16, %rsp
  movl   $1, -4(%rbp)
  movl   -4(%rbp), %r11d
  imull  $2, %r11d
  movl   %r11d, -4(%rbp)
  movl   $4, -8(%rbp)
  addl   $5, -8(%rbp)
  movl   $3, %edx
  cdq
  idivl  -8(%rbp)
  movl   %edx, -12(%rbp)
  movl   -4(%rbp), %eax
  cdq
  idivl  -12(%rbp)
  movl   %eax, -16(%rbp)
  movl   -16(%rbp), %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
"#;
        let emitter = Emitter::new(ast);
        let mut vec = Vec::new();
        emitter.emit(&mut vec).expect("Could not write to string");
        let actual = String::from_utf8(vec).expect("Got invalid UTF-8");

        assert_eq!(expected, actual);
    }

    #[test]
    fn simple_bitwise() {
        let ast = asm::Asm::Program(asm::Function {
            name: "main",
            instructions: vec![
                asm::Instruction::AllocateStack(-16),
                // tmp0 = 5 * 4
                asm::Instruction::Mov(asm::Operand::Imm(5), asm::Operand::Stack(-4)),
                asm::Instruction::Mov(
                    asm::Operand::Stack(-4),
                    asm::Operand::Reg(asm::Register::R11),
                ),
                asm::Instruction::Binary(
                    asm::BinaryOp::Mult,
                    asm::Operand::Imm(4),
                    asm::Operand::Reg(asm::Register::R11),
                ),
                asm::Instruction::Mov(
                    asm::Operand::Reg(asm::Register::R11),
                    asm::Operand::Stack(-4),
                ),
                // tmp1 = 4 - 5
                asm::Instruction::Mov(asm::Operand::Imm(4), asm::Operand::Stack(-8)),
                asm::Instruction::Binary(
                    asm::BinaryOp::Sub,
                    asm::Operand::Imm(5),
                    asm::Operand::Stack(-8),
                ),
                // tmp2 = tmp1 & 6
                asm::Instruction::Mov(
                    asm::Operand::Stack(-8),
                    asm::Operand::Reg(asm::Register::R10),
                ),
                asm::Instruction::Mov(
                    asm::Operand::Reg(asm::Register::R10),
                    asm::Operand::Stack(-12),
                ),
                asm::Instruction::Binary(
                    asm::BinaryOp::And,
                    asm::Operand::Imm(6),
                    asm::Operand::Stack(-12),
                ),
                // tmp3 = tmp0 | tmp2
                asm::Instruction::Mov(
                    asm::Operand::Stack(-4),
                    asm::Operand::Reg(asm::Register::R10),
                ),
                asm::Instruction::Mov(
                    asm::Operand::Reg(asm::Register::R10),
                    asm::Operand::Stack(-16),
                ),
                asm::Instruction::Mov(
                    asm::Operand::Stack(-12),
                    asm::Operand::Reg(asm::Register::R10),
                ),
                asm::Instruction::Binary(
                    asm::BinaryOp::Or,
                    asm::Operand::Reg(asm::Register::R10),
                    asm::Operand::Stack(-16),
                ),
                // return
                asm::Instruction::Mov(
                    asm::Operand::Stack(-16),
                    asm::Operand::Reg(asm::Register::EAX),
                ),
                asm::Instruction::Ret,
            ],
        });

        let expected = r#"  .globl _main
_main:
                       # FUNCTION PROLOGUE
  pushq  %rbp
  movq   %rsp, %rbp
  subq   $-16, %rsp
  movl   $5, -4(%rbp)
  movl   -4(%rbp), %r11d
  imull  $4, %r11d
  movl   %r11d, -4(%rbp)
  movl   $4, -8(%rbp)
  subl   $5, -8(%rbp)
  movl   -8(%rbp), %r10d
  movl   %r10d, -12(%rbp)
  andl   $6, -12(%rbp)
  movl   -4(%rbp), %r10d
  movl   %r10d, -16(%rbp)
  movl   -12(%rbp), %r10d
  orl    %r10d, -16(%rbp)
  movl   -16(%rbp), %eax
                       # RESET REGISTERS
  movq   %rbp, %rsp
  popq   %rbp
  ret
"#;
        let emitter = Emitter::new(ast);
        let mut vec = Vec::new();
        emitter.emit(&mut vec).expect("Could not write to string");
        let actual = String::from_utf8(vec).expect("Got invalid UTF-8");

        assert_eq!(expected, actual);
    }
}
