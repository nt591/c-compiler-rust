// Responsible for taking a TACKY AST
// and converting to an assembly AST
use crate::tacky;
use std::collections::HashMap;

// Lifetime of source test, since we need
// names. TODO: Figure out how to remove this dep.
#[derive(Debug, PartialEq)]
pub enum Asm {
    Program(Function),
}

#[derive(Debug, PartialEq)]
pub struct Function {
    pub name: String,
    pub instructions: Vec<Instruction>,
}

#[derive(Debug, PartialEq, Clone)]
pub enum UnaryOp {
    Not,
    Neg,
}

#[derive(Debug, PartialEq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mult,
    BitwiseAnd,
    BitwiseOr,
    Xor,
    ShiftLeft,
    ShiftRight,
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum CondCode {
    E,  // equal
    NE, // not equal
    G,  // greater than
    GE, // greater or equal
    L,  // less than
    LE, // less than or equal
}

#[derive(Debug, PartialEq)]
pub enum Instruction {
    Mov(Operand, Operand),
    Unary(UnaryOp, Operand),
    Binary(BinaryOp, Operand, Operand),
    Idiv(Operand),
    Cdq,
    AllocateStack(i32),
    Ret,
    // relational operation instructions
    Cmp(Operand, Operand),
    Jmp(String),              //identifier
    JmpCC(CondCode, String),  //conditional jump, eg jmpne to identifier
    SetCC(CondCode, Operand), //conditional set, eg setl
    Label(String),
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
    AX,
    CX,
    DX,
    R10,
    R11,
}

#[derive(Debug, Default)]
struct AsmGenerator {
    stack_offset: i32,
    identifiers: HashMap<String, Operand>, // Tacky var ident -> Pseudo(string)
}

impl Asm {
    pub fn from_tacky(tacky: tacky::AST) -> Asm {
        let mut asm = Self::parse_program(&tacky);
        let mut generator = AsmGenerator::default();
        Self::fixup(&mut asm, &mut generator);
        asm
    }

    fn parse_program(tacky: &tacky::AST) -> Asm {
        match tacky {
            tacky::AST::Program(func) => {
                let func = Self::parse_function(func);
                Asm::Program(func)
            }
        }
    }

    fn parse_function(func: &tacky::Function) -> Function {
        let tacky::Function { name, instructions } = func;
        let instructions = Self::parse_instructions(&instructions);
        Function {
            name: name.into(),
            instructions,
        }
    }

    fn parse_instructions(ins: &[tacky::Instruction]) -> Vec<Instruction> {
        use tacky::Instruction as TIns;
        use tacky::UnaryOp as TUnaryOp;
        use Instruction::*;
        ins.iter()
            .flat_map(|instruction| match instruction {
                TIns::Ret(val) => vec![Mov(val.into(), Operand::Reg(Register::AX)), Ret],
                TIns::Unary { op, src, dst } => match op {
                    TUnaryOp::Not => {
                        // !x is the same as x==0, so compare
                        // then zero out dest addr and check if cmp returned equal
                        vec![
                            Cmp(Operand::Imm(0), src.into()),
                            Mov(Operand::Imm(0), dst.into()),
                            SetCC(CondCode::E, dst.into()),
                        ]
                    }
                    o => vec![Mov(src.into(), dst.into()), Unary(o.into(), dst.into())],
                },
                TIns::Binary {
                    op: tacky::BinaryOp::Divide,
                    src1,
                    src2,
                    dst,
                } => vec![
                    Mov(src1.into(), Operand::Reg(Register::AX)),
                    Cdq,
                    Idiv(src2.into()),
                    Mov(Operand::Reg(Register::AX), dst.into()),
                ],
                TIns::Binary {
                    op: tacky::BinaryOp::Remainder,
                    src1,
                    src2,
                    dst,
                } => vec![
                    Mov(src1.into(), Operand::Reg(Register::AX)),
                    Cdq,
                    Idiv(src2.into()),
                    Mov(Operand::Reg(Register::DX), dst.into()),
                ],
                // SHIFTRIGHT:
                // since we need to sign extend for arithmetic we can use cdq for this
                // TODO: maybe if src1 isn't a negative we avoid the sign extension
                TIns::Binary {
                    op: tacky::BinaryOp::ShiftRight,
                    src1,
                    src2,
                    dst,
                } => vec![
                    Mov(src1.into(), Operand::Reg(Register::AX)),
                    Binary(
                        BinaryOp::ShiftRight,
                        src2.into(),
                        Operand::Reg(Register::AX),
                    ),
                    Mov(Operand::Reg(Register::AX), dst.into()),
                ],
                TIns::Binary {
                    op,
                    src1,
                    src2,
                    dst,
                } => match op {
                    tacky::BinaryOp::Add
                    | tacky::BinaryOp::Multiply
                    | tacky::BinaryOp::Subtract
                    | tacky::BinaryOp::BitwiseAnd
                    | tacky::BinaryOp::BitwiseOr
                    | tacky::BinaryOp::Xor
                    | tacky::BinaryOp::ShiftLeft => {
                        let o = op.into();
                        vec![
                            Mov(src1.into(), dst.into()),
                            Binary(o, src2.into(), dst.into()),
                        ]
                    }
                    tacky::BinaryOp::Equal
                    | tacky::BinaryOp::NotEqual
                    | tacky::BinaryOp::LessThan
                    | tacky::BinaryOp::LessOrEqual
                    | tacky::BinaryOp::GreaterThan
                    | tacky::BinaryOp::GreaterOrEqual => {
                        Self::parse_binary_relational_ops(op, src1, src2, dst)
                    }
                    tacky::BinaryOp::And | tacky::BinaryOp::Or => todo!(),
                    tacky::BinaryOp::Remainder
                    | tacky::BinaryOp::Divide
                    | tacky::BinaryOp::ShiftRight => unreachable!(),
                },
                TIns::JumpIfZero { cond, target } => {
                    // comp condition to 0, then jump if equal to target
                    vec![
                        Cmp(Operand::Imm(0), cond.into()),
                        JmpCC(CondCode::E, target.clone()),
                    ]
                }
                TIns::JumpIfNotZero { cond, target } => {
                    vec![
                        Cmp(Operand::Imm(0), cond.into()),
                        JmpCC(CondCode::NE, target.clone()),
                    ]
                }
                TIns::Label(ident) => vec![Label(ident.clone())],
                TIns::Copy { src, dst } => vec![Mov(src.into(), dst.into())],
                TIns::Jump(ident) => vec![Jmp(ident.clone())],
            })
            .collect::<Vec<_>>()
    }

    fn parse_binary_relational_ops(
        op: &tacky::BinaryOp,
        src1: &tacky::Val,
        src2: &tacky::Val,
        dst: &tacky::Val,
    ) -> Vec<Instruction> {
        use tacky::BinaryOp as TBO;
        use Instruction::*;
        let cond_code = match op {
            TBO::Equal => CondCode::E,
            TBO::NotEqual => CondCode::NE,
            TBO::GreaterThan => CondCode::G,
            TBO::GreaterOrEqual => CondCode::GE,
            TBO::LessThan => CondCode::L,
            TBO::LessOrEqual => CondCode::LE,
            _ => {
                panic!("Unexpected tacky BinaryOp {op:?} when constructing relational instruction")
            }
        };
        // turns foo = x < y into
        // cmp y, x AKA x - y
        // zeros out foo
        // setl foo
        vec![
            Cmp(src2.into(), src1.into()),
            Mov(Operand::Imm(0), dst.clone().into()),
            SetCC(cond_code, dst.into()),
        ]
    }

    fn fixup(asm: &mut Asm, gen: &mut AsmGenerator) {
        match asm {
            Asm::Program(ref mut func) => Self::fixup_function(func, gen),
        };
    }
    fn fixup_function(func: &mut Function, gen: &mut AsmGenerator) {
        let Function {
            name: _name,
            ref mut instructions,
        } = func;
        Self::fixup_pseudos_in_instructions(instructions, gen);
        Self::insert_alloc_stack_func(func, gen);
        Self::fixup_invalid_memory_accesses(func);
    }

    fn fixup_pseudos_in_instructions(ins: &mut [Instruction], gen: &mut AsmGenerator) {
        ins.iter_mut().for_each(|instruction| match instruction {
            Instruction::Mov(src, dst) => {
                *src = Self::replace_pseudo_in_op(src, gen);
                *dst = Self::replace_pseudo_in_op(dst, gen);
            }
            Instruction::Unary(_op, dst) => {
                *dst = Self::replace_pseudo_in_op(dst, gen);
            }
            Instruction::Binary(_op, src1, src2) => {
                *src1 = Self::replace_pseudo_in_op(src1, gen);
                *src2 = Self::replace_pseudo_in_op(src2, gen);
            }
            Instruction::Idiv(src) => {
                *src = Self::replace_pseudo_in_op(src, gen);
            }
            Instruction::Cmp(src, dst) => {
                *src = Self::replace_pseudo_in_op(src, gen);
                *dst = Self::replace_pseudo_in_op(dst, gen);
            }
            Instruction::SetCC(_cc, dst) => {
                *dst = Self::replace_pseudo_in_op(dst, gen);
            }
            _ => {}
        })
    }

    fn replace_pseudo_in_op(op: &Operand, gen: &mut AsmGenerator) -> Operand {
        match op {
            Operand::Pseudo(var) => gen
                .identifiers
                .entry(var.clone())
                .or_insert_with(|| {
                    let next_offset = gen.stack_offset - 4;
                    gen.stack_offset = next_offset;
                    Operand::Stack(next_offset)
                })
                .clone(),
            o => o.clone(), //no transformation otherwise
        }
    }

    fn insert_alloc_stack_func(func: &mut Function, gen: &AsmGenerator) {
        let old_ins = std::mem::take(&mut func.instructions);
        let mut v = vec![Instruction::AllocateStack(gen.stack_offset)];
        for i in old_ins.into_iter() {
            v.push(i);
        }
        func.instructions = v;
    }

    fn fixup_invalid_memory_accesses(func: &mut Function) {
        let old_ins = std::mem::take(&mut func.instructions);
        let mut v = Vec::with_capacity(old_ins.len());
        for ins in old_ins.into_iter() {
            match ins {
                Instruction::Mov(src @ Operand::Stack(_), dst @ Operand::Stack(_)) => {
                    // movl can't move from two memory addrs, so
                    // use a temporary variable along the way in %r10d
                    v.push(Instruction::Mov(src, Operand::Reg(Register::R10)));
                    v.push(Instruction::Mov(Operand::Reg(Register::R10), dst));
                }
                Instruction::Binary(binop, src @ Operand::Stack(_), dst @ Operand::Stack(_))
                    if matches!(
                        binop,
                        BinaryOp::Add
                            | BinaryOp::Sub
                            | BinaryOp::BitwiseAnd
                            | BinaryOp::Xor
                            | BinaryOp::BitwiseOr
                    ) =>
                {
                    // some instructions can't operate from two memory addrs, so
                    // use a temporary variable along the way in %r10d
                    v.push(Instruction::Mov(src, Operand::Reg(Register::R10)));
                    v.push(Instruction::Binary(binop, Operand::Reg(Register::R10), dst));
                }
                Instruction::Binary(binop, src, dst)
                    if matches!(binop, BinaryOp::ShiftLeft | BinaryOp::ShiftRight) =>
                {
                    // shift left/right cannot use a memory address as a source.
                    // We move the data to a scratch register. We write to ECX,
                    // then read from the lower 8 bits.
                    if let Operand::Stack(n) = src {
                        v.push(Instruction::Mov(
                            Operand::Stack(n),
                            Operand::Reg(Register::CX),
                        ));
                        v.push(Instruction::Binary(binop, Operand::Reg(Register::CX), dst));
                    } else {
                        v.push(Instruction::Binary(binop, src, dst));
                    }
                }
                Instruction::Binary(BinaryOp::Mult, src, dst @ Operand::Stack(_)) => {
                    // imul cannot take an addr as a destination, regardless of src.
                    // Rewrite via register %r11d
                    // Move dst into r11d
                    // Multiply src and r11d, store in r11d
                    // Move r11d into dst
                    v.push(Instruction::Mov(dst.clone(), Operand::Reg(Register::R11)));
                    v.push(Instruction::Binary(
                        BinaryOp::Mult,
                        src,
                        Operand::Reg(Register::R11),
                    ));
                    v.push(Instruction::Mov(Operand::Reg(Register::R11), dst));
                }
                Instruction::Idiv(imm @ Operand::Imm(_)) => {
                    // idiv cannot operate on immediates, so move to a scratch register
                    v.push(Instruction::Mov(imm, Operand::Reg(Register::R10)));
                    v.push(Instruction::Idiv(Operand::Reg(Register::R10)));
                }
                Instruction::Cmp(src @ Operand::Stack(_), dst @ Operand::Stack(_)) => {
                    // cmpl can't move from two memory addrs, so
                    // use a temporary variable along the way in %r10d
                    v.push(Instruction::Mov(src, Operand::Reg(Register::R10)));
                    v.push(Instruction::Cmp(Operand::Reg(Register::R10), dst));
                }
                Instruction::Cmp(src, dst @ Operand::Imm(_)) => {
                    // cmpl can't use a constant as a destination so move into reg
                    // use a temporary variable along the way in %r11d
                    v.push(Instruction::Mov(dst.clone(), Operand::Reg(Register::R11)));
                    v.push(Instruction::Cmp(src, Operand::Reg(Register::R11)));
                }

                i => v.push(i),
            }
        }
        func.instructions = v;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tacky;
    use crate::tacky::UnaryOp as TUOp;
    #[test]
    fn basic_parse() {
        let ast = tacky::AST::Program(tacky::Function {
            name: "main".into(),
            instructions: vec![tacky::Instruction::Ret(tacky::Val::Constant(100))],
        });

        let expected = Asm::Program(Function {
            name: "main".into(),
            instructions: vec![
                Instruction::AllocateStack(0),
                Instruction::Mov(Operand::Imm(100), Operand::Reg(Register::AX)),
                Instruction::Ret,
            ],
        });

        let assembly = Asm::from_tacky(ast);
        assert_eq!(assembly, expected);
    }

    #[test]
    fn parse_with_pseudos() {
        let ast = tacky::AST::Program(tacky::Function {
            name: "main".into(),
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
            name: "main".into(),
            instructions: vec![
                Instruction::AllocateStack(-4),
                Instruction::Mov(Operand::Imm(100), Operand::Stack(-4)),
                Instruction::Unary(UnaryOp::Neg, Operand::Stack(-4)),
                Instruction::Mov(Operand::Stack(-4), Operand::Reg(Register::AX)),
                Instruction::Ret,
            ],
        });

        let assembly = Asm::from_tacky(ast);
        assert_eq!(assembly, expected);
    }

    #[test]
    fn parse_nested_unaries() {
        let ast = tacky::AST::Program(tacky::Function {
            name: "main".into(),
            instructions: vec![
                tacky::Instruction::Unary {
                    op: tacky::UnaryOp::Negate,
                    src: tacky::Val::Constant(100),
                    dst: tacky::Val::Var("tmp.0".into()),
                },
                tacky::Instruction::Unary {
                    op: tacky::UnaryOp::Complement,
                    src: tacky::Val::Var("tmp.0".into()),
                    dst: tacky::Val::Var("tmp.1".into()),
                },
                tacky::Instruction::Unary {
                    op: tacky::UnaryOp::Negate,
                    src: tacky::Val::Var("tmp.1".into()),
                    dst: tacky::Val::Var("tmp.2".into()),
                },
                tacky::Instruction::Ret(tacky::Val::Var("tmp.2".into())),
            ],
        });

        let expected = Asm::Program(Function {
            name: "main".into(),
            instructions: vec![
                Instruction::AllocateStack(-12),
                Instruction::Mov(Operand::Imm(100), Operand::Stack(-4)),
                Instruction::Unary(UnaryOp::Neg, Operand::Stack(-4)),
                Instruction::Mov(Operand::Stack(-4), Operand::Reg(Register::R10)),
                Instruction::Mov(Operand::Reg(Register::R10), Operand::Stack(-8)),
                Instruction::Unary(UnaryOp::Not, Operand::Stack(-8)),
                Instruction::Mov(Operand::Stack(-8), Operand::Reg(Register::R10)),
                Instruction::Mov(Operand::Reg(Register::R10), Operand::Stack(-12)),
                Instruction::Unary(UnaryOp::Neg, Operand::Stack(-12)),
                Instruction::Mov(Operand::Stack(-12), Operand::Reg(Register::AX)),
                Instruction::Ret,
            ],
        });

        let assembly = Asm::from_tacky(ast);
        assert_eq!(assembly, expected);
    }

    #[test]
    fn generate_binary_expressions() {
        let ast = tacky::AST::Program(tacky::Function {
            name: "main".into(),
            instructions: vec![
                tacky::Instruction::Binary {
                    op: tacky::BinaryOp::Multiply,
                    src1: tacky::Val::Constant(1),
                    src2: tacky::Val::Constant(2),
                    dst: tacky::Val::Var("tmp.0".into()),
                },
                tacky::Instruction::Binary {
                    op: tacky::BinaryOp::Add,
                    src1: tacky::Val::Constant(4),
                    src2: tacky::Val::Constant(5),
                    dst: tacky::Val::Var("tmp.1".into()),
                },
                tacky::Instruction::Binary {
                    op: tacky::BinaryOp::Remainder,
                    src1: tacky::Val::Constant(3),
                    src2: tacky::Val::Var("tmp.1".into()),
                    dst: tacky::Val::Var("tmp.2".into()),
                },
                tacky::Instruction::Binary {
                    op: tacky::BinaryOp::Divide,
                    src1: tacky::Val::Var("tmp.0".into()),
                    src2: tacky::Val::Var("tmp.2".into()),
                    dst: tacky::Val::Var("tmp.3".into()),
                },
                tacky::Instruction::Ret(tacky::Val::Var("tmp.3".into())),
            ],
        });
        let expected = Asm::Program(Function {
            name: "main".into(),
            instructions: vec![
                Instruction::AllocateStack(-16),
                // tmp0 = 1 * 2
                Instruction::Mov(Operand::Imm(1), Operand::Stack(-4)),
                Instruction::Mov(Operand::Stack(-4), Operand::Reg(Register::R11)),
                Instruction::Binary(BinaryOp::Mult, Operand::Imm(2), Operand::Reg(Register::R11)),
                Instruction::Mov(Operand::Reg(Register::R11), Operand::Stack(-4)),
                // tmp1 = 4 + 5
                Instruction::Mov(Operand::Imm(4), Operand::Stack(-8)),
                Instruction::Binary(BinaryOp::Add, Operand::Imm(5), Operand::Stack(-8)),
                // tmp2 = 3 % tmp1
                Instruction::Mov(Operand::Imm(3), Operand::Reg(Register::AX)),
                Instruction::Cdq,
                Instruction::Idiv(Operand::Stack(-8)),
                Instruction::Mov(Operand::Reg(Register::DX), Operand::Stack(-12)),
                // tmp3 = tmp0 / tmp2
                Instruction::Mov(Operand::Stack(-4), Operand::Reg(Register::AX)),
                Instruction::Cdq,
                Instruction::Idiv(Operand::Stack(-12)),
                Instruction::Mov(Operand::Reg(Register::AX), Operand::Stack(-16)),
                // return
                Instruction::Mov(Operand::Stack(-16), Operand::Reg(Register::AX)),
                Instruction::Ret,
            ],
        });

        let assembly = Asm::from_tacky(ast);
        assert_eq!(assembly, expected);
    }

    #[test]
    fn complex_binary_expressions() {
        let ast = tacky::AST::Program(tacky::Function {
            name: "main".into(),
            instructions: vec![
                tacky::Instruction::Binary {
                    op: tacky::BinaryOp::Multiply,
                    src1: tacky::Val::Constant(5),
                    src2: tacky::Val::Constant(4),
                    dst: tacky::Val::Var("tmp.0".into()),
                },
                tacky::Instruction::Binary {
                    op: tacky::BinaryOp::Divide,
                    src1: tacky::Val::Var("tmp.0".into()),
                    src2: tacky::Val::Constant(2),
                    dst: tacky::Val::Var("tmp.1".into()),
                },
                tacky::Instruction::Binary {
                    op: tacky::BinaryOp::Add,
                    src1: tacky::Val::Constant(2),
                    src2: tacky::Val::Constant(1),
                    dst: tacky::Val::Var("tmp.2".into()),
                },
                tacky::Instruction::Binary {
                    op: tacky::BinaryOp::Remainder,
                    src1: tacky::Val::Constant(3),
                    src2: tacky::Val::Var("tmp.2".into()),
                    dst: tacky::Val::Var("tmp.3".into()),
                },
                tacky::Instruction::Binary {
                    op: tacky::BinaryOp::Subtract,
                    src1: tacky::Val::Var("tmp.1".into()),
                    src2: tacky::Val::Var("tmp.3".into()),
                    dst: tacky::Val::Var("tmp.4".into()),
                },
                tacky::Instruction::Ret(tacky::Val::Var("tmp.4".into())),
            ],
        });

        let expected = Asm::Program(Function {
            name: "main".into(),
            instructions: vec![
                Instruction::AllocateStack(-20),
                // tmp0 = 5 * 4 = 20
                Instruction::Mov(Operand::Imm(5), Operand::Stack(-4)),
                Instruction::Mov(Operand::Stack(-4), Operand::Reg(Register::R11)),
                Instruction::Binary(BinaryOp::Mult, Operand::Imm(4), Operand::Reg(Register::R11)),
                Instruction::Mov(Operand::Reg(Register::R11), Operand::Stack(-4)),
                // tmp1 = tmp0 / 2 = 10
                Instruction::Mov(Operand::Stack(-4), Operand::Reg(Register::AX)),
                Instruction::Cdq,
                Instruction::Mov(Operand::Imm(2), Operand::Reg(Register::R10)),
                Instruction::Idiv(Operand::Reg(Register::R10)),
                Instruction::Mov(Operand::Reg(Register::AX), Operand::Stack(-8)),
                // tmp2 = 2 + 1  = 3
                Instruction::Mov(Operand::Imm(2), Operand::Stack(-12)),
                Instruction::Binary(BinaryOp::Add, Operand::Imm(1), Operand::Stack(-12)),
                // tmp3 = 3 % tmp2 = 0
                Instruction::Mov(Operand::Imm(3), Operand::Reg(Register::AX)),
                Instruction::Cdq,
                Instruction::Idiv(Operand::Stack(-12)),
                Instruction::Mov(Operand::Reg(Register::DX), Operand::Stack(-16)),
                // tmp3 = tmp1 - tmp3 = 10
                Instruction::Mov(Operand::Stack(-8), Operand::Reg(Register::R10)),
                Instruction::Mov(Operand::Reg(Register::R10), Operand::Stack(-20)),
                Instruction::Mov(Operand::Stack(-16), Operand::Reg(Register::R10)),
                Instruction::Binary(
                    BinaryOp::Sub,
                    Operand::Reg(Register::R10),
                    Operand::Stack(-20),
                ),
                // return
                Instruction::Mov(Operand::Stack(-20), Operand::Reg(Register::AX)),
                Instruction::Ret,
            ],
        });

        let assembly = Asm::from_tacky(ast);
        assert_eq!(assembly, expected);
    }

    #[test]
    fn simple_bitwise() {
        let ast = tacky::AST::Program(tacky::Function {
            name: "main".into(),
            instructions: vec![
                tacky::Instruction::Binary {
                    op: tacky::BinaryOp::Multiply,
                    src1: tacky::Val::Constant(5),
                    src2: tacky::Val::Constant(4),
                    dst: tacky::Val::Var("tmp.0".into()),
                },
                tacky::Instruction::Binary {
                    op: tacky::BinaryOp::Subtract,
                    src1: tacky::Val::Constant(4),
                    src2: tacky::Val::Constant(5),
                    dst: tacky::Val::Var("tmp.1".into()),
                },
                tacky::Instruction::Binary {
                    op: tacky::BinaryOp::BitwiseAnd,
                    src1: tacky::Val::Var("tmp.1".into()),
                    src2: tacky::Val::Constant(6),
                    dst: tacky::Val::Var("tmp.2".into()),
                },
                tacky::Instruction::Binary {
                    op: tacky::BinaryOp::BitwiseOr,
                    src1: tacky::Val::Var("tmp.0".into()),
                    src2: tacky::Val::Var("tmp.2".into()),
                    dst: tacky::Val::Var("tmp.3".into()),
                },
                tacky::Instruction::Ret(tacky::Val::Var("tmp.3".into())),
            ],
        });
        let expected = Asm::Program(Function {
            name: "main".into(),
            instructions: vec![
                Instruction::AllocateStack(-16),
                // tmp0 = 5 * 4
                Instruction::Mov(Operand::Imm(5), Operand::Stack(-4)),
                Instruction::Mov(Operand::Stack(-4), Operand::Reg(Register::R11)),
                Instruction::Binary(BinaryOp::Mult, Operand::Imm(4), Operand::Reg(Register::R11)),
                Instruction::Mov(Operand::Reg(Register::R11), Operand::Stack(-4)),
                // tmp1 = 4 - 5
                Instruction::Mov(Operand::Imm(4), Operand::Stack(-8)),
                Instruction::Binary(BinaryOp::Sub, Operand::Imm(5), Operand::Stack(-8)),
                // tmp2 = tmp1 & 6
                Instruction::Mov(Operand::Stack(-8), Operand::Reg(Register::R10)),
                Instruction::Mov(Operand::Reg(Register::R10), Operand::Stack(-12)),
                Instruction::Binary(BinaryOp::BitwiseAnd, Operand::Imm(6), Operand::Stack(-12)),
                // tmp3 = tmp0 | tmp2
                Instruction::Mov(Operand::Stack(-4), Operand::Reg(Register::R10)),
                Instruction::Mov(Operand::Reg(Register::R10), Operand::Stack(-16)),
                Instruction::Mov(Operand::Stack(-12), Operand::Reg(Register::R10)),
                Instruction::Binary(
                    BinaryOp::BitwiseOr,
                    Operand::Reg(Register::R10),
                    Operand::Stack(-16),
                ),
                // return
                Instruction::Mov(Operand::Stack(-16), Operand::Reg(Register::AX)),
                Instruction::Ret,
            ],
        });
        let assembly = Asm::from_tacky(ast);
        assert_eq!(assembly, expected);
    }

    #[test]
    fn shiftleft() {
        let ast = tacky::AST::Program(tacky::Function {
            name: "main".into(),
            instructions: vec![
                tacky::Instruction::Binary {
                    op: tacky::BinaryOp::Multiply,
                    src1: tacky::Val::Constant(5),
                    src2: tacky::Val::Constant(4),
                    dst: tacky::Val::Var("tmp.0".into()),
                },
                tacky::Instruction::Binary {
                    op: tacky::BinaryOp::ShiftLeft,
                    src1: tacky::Val::Var("tmp.0".into()),
                    src2: tacky::Val::Constant(2),
                    dst: tacky::Val::Var("tmp.1".into()),
                },
                tacky::Instruction::Ret(tacky::Val::Var("tmp.1".into())),
            ],
        });

        let expected = Asm::Program(Function {
            name: "main".into(),
            instructions: vec![
                Instruction::AllocateStack(-8),
                // tmp0 = 5 * 4
                Instruction::Mov(Operand::Imm(5), Operand::Stack(-4)),
                Instruction::Mov(Operand::Stack(-4), Operand::Reg(Register::R11)),
                Instruction::Binary(BinaryOp::Mult, Operand::Imm(4), Operand::Reg(Register::R11)),
                Instruction::Mov(Operand::Reg(Register::R11), Operand::Stack(-4)),
                // tmp1 = tmp.0 << 2
                // moves tmp.8 into tmp.1 via reg10
                Instruction::Mov(Operand::Stack(-4), Operand::Reg(Register::R10)),
                Instruction::Mov(Operand::Reg(Register::R10), Operand::Stack(-8)),
                Instruction::Binary(BinaryOp::ShiftLeft, Operand::Imm(2), Operand::Stack(-8)),
                // return
                Instruction::Mov(Operand::Stack(-8), Operand::Reg(Register::AX)),
                Instruction::Ret,
            ],
        });
        let assembly = Asm::from_tacky(ast);
        assert_eq!(assembly, expected);
    }

    #[test]
    fn shiftright_lhs_is_negative() {
        let ast = tacky::AST::Program(tacky::Function {
            name: "main".into(),
            instructions: vec![
                tacky::Instruction::Unary {
                    op: tacky::UnaryOp::Negate,
                    src: tacky::Val::Constant(5),
                    dst: tacky::Val::Var("tmp.0".into()),
                },
                tacky::Instruction::Binary {
                    op: tacky::BinaryOp::ShiftRight,
                    src1: tacky::Val::Var("tmp.0".into()),
                    src2: tacky::Val::Constant(30),
                    dst: tacky::Val::Var("tmp.1".into()),
                },
                tacky::Instruction::Ret(tacky::Val::Var("tmp.1".into())),
            ],
        });
        let expected = Asm::Program(Function {
            name: "main".into(),
            instructions: vec![
                Instruction::AllocateStack(-8),
                // tmp0 = -5
                Instruction::Mov(Operand::Imm(5), Operand::Stack(-4)),
                Instruction::Unary(UnaryOp::Neg, Operand::Stack(-4)),
                // tmp1 = tmp.0 >> 30
                Instruction::Mov(Operand::Stack(-4), Operand::Reg(Register::AX)),
                Instruction::Binary(
                    BinaryOp::ShiftRight,
                    Operand::Imm(30),
                    Operand::Reg(Register::AX),
                ),
                Instruction::Mov(Operand::Reg(Register::AX), Operand::Stack(-8)),
                // return
                Instruction::Mov(Operand::Stack(-8), Operand::Reg(Register::AX)),
                Instruction::Ret,
            ],
        });
        let assembly = Asm::from_tacky(ast);
        assert_eq!(assembly, expected);
    }

    #[test]
    fn shiftleft_rhs_is_expr() {
        let ast = tacky::AST::Program(tacky::Function {
            name: "main".into(),
            instructions: vec![
                tacky::Instruction::Binary {
                    op: tacky::BinaryOp::Add,
                    src1: tacky::Val::Constant(1),
                    src2: tacky::Val::Constant(2),
                    dst: tacky::Val::Var("tmp.0".into()),
                },
                tacky::Instruction::Binary {
                    op: tacky::BinaryOp::ShiftLeft,
                    src1: tacky::Val::Constant(5),
                    src2: tacky::Val::Var("tmp.0".into()),
                    dst: tacky::Val::Var("tmp.1".into()),
                },
                tacky::Instruction::Ret(tacky::Val::Var("tmp.1".into())),
            ],
        });
        let expected = Asm::Program(Function {
            name: "main".into(),
            instructions: vec![
                Instruction::AllocateStack(-8),
                // tmp0 = 1 + 2
                Instruction::Mov(Operand::Imm(1), Operand::Stack(-4)),
                Instruction::Binary(BinaryOp::Add, Operand::Imm(2), Operand::Stack(-4)),
                // tmp1 = 5 << tmp.0
                Instruction::Mov(Operand::Imm(5), Operand::Stack(-8)),
                Instruction::Mov(Operand::Stack(-4), Operand::Reg(Register::CX)),
                Instruction::Binary(
                    BinaryOp::ShiftLeft,
                    Operand::Reg(Register::CX),
                    Operand::Stack(-8),
                ),
                // return
                Instruction::Mov(Operand::Stack(-8), Operand::Reg(Register::AX)),
                Instruction::Ret,
            ],
        });
        let assembly = Asm::from_tacky(ast);
        assert_eq!(assembly, expected);
    }

    #[test]
    fn unary_not() {
        let ast = tacky::AST::Program(tacky::Function {
            name: "main".into(),
            instructions: vec![
                tacky::Instruction::Unary {
                    op: TUOp::Not,
                    src: tacky::Val::Constant(1),
                    dst: tacky::Val::Var("tmp.0".into()),
                },
                tacky::Instruction::Ret(tacky::Val::Var("tmp.0".into())),
            ],
        });
        let expected = Asm::Program(Function {
            name: "main".into(),
            // move 1 into register 11,
            // then check if 1 == 0
            // clear out the next address, then check if cmp set ZF
            // and write to stack addr -4
            // move stack addr -4 to EAX and return
            instructions: vec![
                Instruction::AllocateStack(-4),
                Instruction::Mov(Operand::Imm(1), Operand::Reg(Register::R11)),
                Instruction::Cmp(Operand::Imm(0), Operand::Reg(Register::R11)),
                Instruction::Mov(Operand::Imm(0), Operand::Stack(-4)),
                Instruction::SetCC(CondCode::E, Operand::Stack(-4)),
                Instruction::Mov(Operand::Stack(-4), Operand::Reg(Register::AX)),
                Instruction::Ret,
            ],
        });
        let assembly = Asm::from_tacky(ast);
        assert_eq!(assembly, expected);
    }

    #[test]
    fn binary_greater_or_equal() {
        let ast = tacky::AST::Program(tacky::Function {
            name: "main".into(),
            instructions: vec![
                tacky::Instruction::Binary {
                    op: tacky::BinaryOp::GreaterOrEqual,
                    src1: tacky::Val::Var("tmp.0".into()),
                    src2: tacky::Val::Constant(2),
                    dst: tacky::Val::Var("tmp.1".into()),
                },
                tacky::Instruction::Ret(tacky::Val::Var("tmp.1".into())),
            ],
        });
        let expected = Asm::Program(Function {
            name: "main".into(),
            instructions: vec![
                Instruction::AllocateStack(-8),
                Instruction::Cmp(Operand::Imm(2), Operand::Stack(-4)),
                Instruction::Mov(Operand::Imm(0), Operand::Stack(-8)),
                Instruction::SetCC(CondCode::GE, Operand::Stack(-8)),
                Instruction::Mov(Operand::Stack(-8), Operand::Reg(Register::AX)),
                Instruction::Ret,
            ],
        });
        let assembly = Asm::from_tacky(ast);
        assert_eq!(assembly, expected);
    }

    #[test]
    fn jump_if_zero() {
        let ast = tacky::AST::Program(tacky::Function {
            name: "main".into(),
            instructions: vec![
                tacky::Instruction::Copy {
                    src: tacky::Val::Constant(5),
                    dst: tacky::Val::Var("tmp.0".into()),
                },
                tacky::Instruction::JumpIfZero {
                    cond: tacky::Val::Var("tmp.0".into()),
                    target: "and_expr_false.0".into(),
                },
                tacky::Instruction::Binary {
                    op: tacky::BinaryOp::Add,
                    src1: tacky::Val::Constant(1),
                    src2: tacky::Val::Constant(2),
                    dst: tacky::Val::Var("tmp.1".into()),
                },
                tacky::Instruction::Copy {
                    src: tacky::Val::Var("tmp.1".into()),
                    dst: tacky::Val::Var("tmp.2".into()),
                },
                tacky::Instruction::JumpIfZero {
                    cond: tacky::Val::Var("tmp.2".into()),
                    target: "and_expr_false.0".into(),
                },
                tacky::Instruction::Copy {
                    src: tacky::Val::Constant(1),
                    dst: tacky::Val::Var("tmp.3".into()),
                },
                tacky::Instruction::Jump("and_expr_end.1".into()),
                tacky::Instruction::Label("and_expr_false.0".into()),
                tacky::Instruction::Copy {
                    src: tacky::Val::Constant(0),
                    dst: tacky::Val::Var("tmp.3".into()),
                },
                tacky::Instruction::Label("and_expr_end.1".into()),
                tacky::Instruction::Ret(tacky::Val::Var("tmp.3".into())),
            ],
        });

        let expected = Asm::Program(Function {
            name: "main".into(),
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
            tacky::UnaryOp::Not => todo!(),
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
            &tacky::UnaryOp::Not => todo!(),
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

impl From<&tacky::BinaryOp> for BinaryOp {
    fn from(o: &tacky::BinaryOp) -> Self {
        use tacky::BinaryOp as TBO;
        match o {
            TBO::Add => BinaryOp::Add,
            TBO::Subtract => BinaryOp::Sub,
            TBO::Multiply => BinaryOp::Mult,
            TBO::BitwiseAnd => BinaryOp::BitwiseAnd,
            TBO::BitwiseOr => BinaryOp::BitwiseOr,
            TBO::Xor => BinaryOp::Xor,
            TBO::ShiftLeft => BinaryOp::ShiftLeft,
            TBO::ShiftRight => BinaryOp::ShiftRight,
            o => panic!("No way to convert tacky binary op {o:?} into asm binary op"),
        }
    }
}
