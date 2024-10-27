// takes the asm parser and emits X64
// only works on my Mac, so do with that what you will.

use crate::asm;
use crate::Asm;
use std::io::Write;

// lifetime bound to source text.
pub struct Emitter<'a>(Asm<'a>);

#[derive(Debug, PartialEq, Copy, Clone)]
enum RegisterSize {
    FourByte,
    OneByte,
}

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
                Self::emit_op(src, RegisterSize::FourByte, output)?;
                write!(output, ", ")?;
                Self::emit_op(dst, RegisterSize::FourByte, output)?;
                write!(output, "\n")?;
            }
            asm::Instruction::Unary(unary, operand) => {
                write!(output, "  ")?;
                Self::emit_unary(unary, output)?;
                write!(output, "   ")?;
                Self::emit_op(operand, RegisterSize::FourByte, output)?;
                write!(output, "\n")?;
            }
            asm::Instruction::Binary(binop, src, dst) => {
                write!(output, "  ")?;
                Self::emit_binary(binop, output)?;
                // assume emit_binary handles proper space formatting!
                Self::emit_op(src, RegisterSize::FourByte, output)?;
                write!(output, ", ")?;
                // shift left and right use the lower 8 bits of ECX to read
                match binop {
                    asm::BinaryOp::ShiftLeft | asm::BinaryOp::ShiftRight => {
                        Self::emit_op(dst, RegisterSize::OneByte, output)?
                    }
                    _op => Self::emit_op(dst, RegisterSize::FourByte, output)?,
                };
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
                Self::emit_op(operand, RegisterSize::FourByte, output)?;
                write!(output, "\n")?;
            }
            asm::Instruction::Cmp(op1, op2) => {
                write!(output, "  cmpl   ")?;
                Self::emit_op(op1, RegisterSize::FourByte, output)?;
                write!(output, ", ")?;
                Self::emit_op(op2, RegisterSize::FourByte, output)?;
                write!(output, "\n")?;
            }
            asm::Instruction::Jmp(label) => {
                write!(output, "  jmp    ")?;
                Self::emit_label(&label, output)?;
                write!(output, "\n")?;
            }
            asm::Instruction::Label(label) => {
                Self::emit_label(&label, output)?;
                writeln!(output, ":")?
            }
            asm::Instruction::JmpCC(cond, label) => {
                write!(output, "  ")?;
                Self::emit_cond_jmp(*cond, output)?;
                Self::emit_label(&label, output)?;
                write!(output, "\n")?;
            }
            asm::Instruction::SetCC(cond, operand) => {
                write!(output, "  ")?;
                Self::emit_cond_set(*cond, output)?;
                Self::emit_op(operand, RegisterSize::OneByte, output)?;
                write!(output, "\n")?;
            }
        }
        Ok(())
    }

    fn emit_op<W: Write>(
        op: &asm::Operand,
        regsize: RegisterSize,
        output: &mut W,
    ) -> std::io::Result<()> {
        match op {
            asm::Operand::Reg(reg) if regsize == RegisterSize::FourByte => {
                Self::emit_register(reg, output)?
            }
            asm::Operand::Reg(reg) if regsize == RegisterSize::OneByte => {
                Self::emit_register_one_byte(reg, output)?
            }
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
            asm::BinaryOp::BitwiseAnd => write!(output, "andl   ")?,
            asm::BinaryOp::BitwiseOr => write!(output, "orl    ")?,
            asm::BinaryOp::Xor => write!(output, "xorl   ")?,
            asm::BinaryOp::ShiftLeft => write!(output, "shll   ")?,
            asm::BinaryOp::ShiftRight => write!(output, "shrl   ")?,
        }
        Ok(())
    }

    fn emit_register<W: Write>(reg: &asm::Register, output: &mut W) -> std::io::Result<()> {
        match reg {
            asm::Register::AX => write!(output, "{}", "%eax"),
            asm::Register::CX => write!(output, "{}", "%ecx"),
            asm::Register::DX => write!(output, "{}", "%edx"),
            asm::Register::R10 => write!(output, "{}", "%r10d"),
            asm::Register::R11 => write!(output, "{}", "%r11d"),
        }
    }

    fn emit_register_one_byte<W: Write>(
        reg: &asm::Register,
        output: &mut W,
    ) -> std::io::Result<()> {
        match reg {
            asm::Register::AX => write!(output, "{}", "%al"),
            asm::Register::CX => write!(output, "{}", "%cl"),
            asm::Register::DX => write!(output, "{}", "%dl"),
            asm::Register::R10 => write!(output, "{}", "%r10b"),
            asm::Register::R11 => write!(output, "{}", "%r11b"),
        }
    }

    fn emit_label<W: Write>(label: &str, output: &mut W) -> std::io::Result<()> {
        write!(output, "L{}", label)?;
        Ok(())
    }

    fn emit_cond_jmp<W: Write>(cond: asm::CondCode, output: &mut W) -> std::io::Result<()> {
        use asm::CondCode::*;
        match cond {
            E =>  write!(output, "je          ")?,
            NE => write!(output, "jne         ")?,
            G =>  write!(output, "jg          ")?,
            GE => write!(output, "jge         ")?,
            L =>  write!(output, "jl          ")?,
            LE => write!(output, "jle         ")?,
        }
        Ok(())
    }

    fn emit_cond_set<W: Write>(cond: asm::CondCode, output: &mut W) -> std::io::Result<()> {
        use asm::CondCode::*;
        match cond {
            E =>  write!(output, "sete       ")?,
            NE => write!(output, "setne      ")?,
            G =>  write!(output, "setg       ")?,
            GE => write!(output, "setge      ")?,
            L =>  write!(output, "setl       ")?,
            LE => write!(output, "setle      ")?,
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use asm::*;

    #[test]
    fn basic_emit() {
        let ast = Asm::Program(asm::Function {
            name: "main",
            instructions: vec![
                asm::Instruction::Mov(asm::Operand::Imm(100), asm::Operand::Reg(asm::Register::AX)),
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
                    asm::Operand::Reg(asm::Register::AX),
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
                asm::Instruction::Mov(asm::Operand::Imm(3), asm::Operand::Reg(asm::Register::DX)),
                asm::Instruction::Cdq,
                asm::Instruction::Idiv(asm::Operand::Stack(-8)),
                asm::Instruction::Mov(
                    asm::Operand::Reg(asm::Register::DX),
                    asm::Operand::Stack(-12),
                ),
                // tmp3 = tmp0 / tmp2
                asm::Instruction::Mov(
                    asm::Operand::Stack(-4),
                    asm::Operand::Reg(asm::Register::AX),
                ),
                asm::Instruction::Cdq,
                asm::Instruction::Idiv(asm::Operand::Stack(-12)),
                asm::Instruction::Mov(
                    asm::Operand::Reg(asm::Register::AX),
                    asm::Operand::Stack(-16),
                ),
                // return
                asm::Instruction::Mov(
                    asm::Operand::Stack(-16),
                    asm::Operand::Reg(asm::Register::AX),
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
                    asm::BinaryOp::BitwiseAnd,
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
                    asm::BinaryOp::BitwiseOr,
                    asm::Operand::Reg(asm::Register::R10),
                    asm::Operand::Stack(-16),
                ),
                // return
                asm::Instruction::Mov(
                    asm::Operand::Stack(-16),
                    asm::Operand::Reg(asm::Register::AX),
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

    #[test]
    fn shiftleft() {
        let ast = asm::Asm::Program(asm::Function {
            name: "main",
            instructions: vec![
                asm::Instruction::AllocateStack(-8),
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
                // tmp1 = tmp.0 << 2
                // moves tmp.8 into tmp.1 via reg10
                asm::Instruction::Mov(
                    asm::Operand::Stack(-4),
                    asm::Operand::Reg(asm::Register::R10),
                ),
                asm::Instruction::Mov(
                    asm::Operand::Reg(asm::Register::R10),
                    asm::Operand::Stack(-8),
                ),
                asm::Instruction::Binary(
                    asm::BinaryOp::ShiftLeft,
                    asm::Operand::Imm(2),
                    asm::Operand::Stack(-8),
                ),
                // return
                asm::Instruction::Mov(
                    asm::Operand::Stack(-8),
                    asm::Operand::Reg(asm::Register::AX),
                ),
                asm::Instruction::Ret,
            ],
        });
        let expected = r#"  .globl _main
_main:
                       # FUNCTION PROLOGUE
  pushq  %rbp
  movq   %rsp, %rbp
  subq   $-8, %rsp
  movl   $5, -4(%rbp)
  movl   -4(%rbp), %r11d
  imull  $4, %r11d
  movl   %r11d, -4(%rbp)
  movl   -4(%rbp), %r10d
  movl   %r10d, -8(%rbp)
  shll   $2, -8(%rbp)
  movl   -8(%rbp), %eax
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
    fn jump_cond() {
        let ast = Asm::Program(Function {
            name: "main",
            instructions: vec![
                Instruction::AllocateStack(-16),
                Instruction::Mov(Operand::Imm(5), Operand::Stack(-4)),
                Instruction::Cmp(Operand::Imm(0), Operand::Stack(-4)),
                Instruction::JmpCC(CondCode::E, "and_expr_false.0".into()),
                Instruction::Mov(Operand::Imm(1), Operand::Stack(-8)),
                Instruction::Binary(BinaryOp::Add, Operand::Imm(2), Operand::Stack(-8)),
                Instruction::Mov(Operand::Stack(-8), Operand::Reg(Register::R10)),
                Instruction::Mov(Operand::Reg(Register::R10), Operand::Stack(-12)),
                Instruction::Cmp(Operand::Imm(0), Operand::Stack(-12)),
                Instruction::JmpCC(CondCode::E, "and_expr_false.0".into()),
                Instruction::Mov(Operand::Imm(1), Operand::Stack(-16)),
                Instruction::Jmp("and_expr_end.1".into()),
                Instruction::Label("and_expr_false.0".into()),
                Instruction::Mov(Operand::Imm(0), Operand::Stack(-16)),
                Instruction::Label("and_expr_end.1".into()),
                Instruction::Mov(Operand::Stack(-16), Operand::Reg(Register::AX)),
                Instruction::Ret,
            ],
        });
        let expected = r#"  .globl _main
_main:
                       # FUNCTION PROLOGUE
  pushq  %rbp
  movq   %rsp, %rbp
  subq   $-16, %rsp
  movl   $5, -4(%rbp)
  cmpl   $0, -4(%rbp)
  je          Land_expr_false.0
  movl   $1, -8(%rbp)
  addl   $2, -8(%rbp)
  movl   -8(%rbp), %r10d
  movl   %r10d, -12(%rbp)
  cmpl   $0, -12(%rbp)
  je          Land_expr_false.0
  movl   $1, -16(%rbp)
  jmp    Land_expr_end.1
Land_expr_false.0:
  movl   $0, -16(%rbp)
Land_expr_end.1:
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
