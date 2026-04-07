// Responsible for taking a TACKY AST
// and converting to an assembly AST
use crate::ast::Const;
use crate::semantic_analysis;
use crate::semantic_analysis::SymbolTable;
use crate::symbol_table;
use crate::symbol_table::BackendSymbolTable;
use crate::tacky;
use crate::tacky::Val;
use crate::types::AssemblyType;
use crate::types::CType;
use crate::types::StaticInit;
use std::collections::HashMap;
use std::i32;

#[derive(Debug, PartialEq)]
pub enum Asm {
    Program(Vec<TopLevel>),
}

#[derive(Debug, PartialEq)]
pub enum TopLevel {
    Func(Function),
    StaticVariable {
        identifier: String,
        global: bool,
        init: StaticInit,
        alignment: usize,
    },
}

#[derive(Debug, PartialEq)]
pub struct Function {
    pub name: String,
    pub instructions: Vec<Instruction>,
    pub global: bool,
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
    E,  // equal, ZF set
    NE, // not equal, ZF not set
    // signed only
    G,  // greater than
    GE, // greater or equal
    L,  // less than
    LE, // less than or equal
    // Unsigned only
    A,  // Above, ZF not set and CF not set (a > b)
    AE, // Above or equal, CF not set (if equal, ZF set)
    B,  // Below, CF set (a < b)
    BE, // Below or equal, CF set or ZF set
}

#[derive(Debug, PartialEq)]
pub enum Instruction {
    Mov(AssemblyType, Operand, Operand),
    Movsx(Operand, Operand),         // src, dst
    MovZeroExtend(Operand, Operand), // src, dst
    Unary(UnaryOp, AssemblyType, Operand),
    Binary(BinaryOp, AssemblyType, Operand, Operand),
    Idiv(AssemblyType, Operand),
    Div(AssemblyType, Operand), // unsigned division instruction
    Cdq(AssemblyType),
    Ret,
    // relational operation instructions
    Cmp(AssemblyType, Operand, Operand),
    Jmp(String),              //identifier
    JmpCC(CondCode, String),  //conditional jump, eg jmpne to identifier
    SetCC(CondCode, Operand), //conditional set, eg setl
    Label(String),
    // function call instructions
    Push(Operand),
    Call(String),
}

// implement clone so our mapping of Tacky Var
// to Pseudo can always return an owned value
#[derive(Debug, PartialEq, Clone)]
pub enum Operand {
    Imm(usize),
    Reg(Register),
    Pseudo(String),
    Stack(i32),
    Data(String), // RIP-relative access to .data and .bss
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Register {
    AX,
    CX,
    DX,
    DI,
    SI,
    R8,
    R9,
    R10,
    R11,
    SP,
}

#[derive(Debug, Default)]
struct AsmGenerator {
    stack_offset: i32,
    identifiers: HashMap<String, Operand>, // Tacky var ident -> Pseudo(string)
}

impl Asm {
    pub fn from_tacky(tacky: tacky::AST, symbol_table: semantic_analysis::SymbolTable) -> Asm {
        let mut asm = Self::parse_program(&tacky, &symbol_table);
        let mut generator = AsmGenerator::default();
        let symbol_table = symbol_table::backend_symbol_table_from_symbol_table(symbol_table);
        Self::fixup(&mut asm, &mut generator, symbol_table);
        asm
    }

    fn parse_program(tacky: &tacky::AST, symbol_table: &semantic_analysis::SymbolTable) -> Asm {
        match tacky {
            tacky::AST::Program(ins) => {
                let ins = ins
                    .into_iter()
                    .map(|instruct| match instruct {
                        func @ tacky::TopLevel::Function { .. } => {
                            Self::parse_function(func, symbol_table)
                        }
                        sv @ tacky::TopLevel::StaticVariable { .. } => {
                            Self::parse_static_variable(sv, symbol_table)
                        }
                    })
                    .collect::<Vec<_>>();
                Asm::Program(ins)
            }
        }
    }

    fn parse_static_variable(sv: &tacky::TopLevel, symbol_table: &SymbolTable) -> TopLevel {
        let tacky::TopLevel::StaticVariable {
            identifier,
            global,
            init,
            ..
        } = sv
        else {
            panic!();
        };
        let assembly_type = get_assembly_type_from_var_name(identifier, symbol_table);
        let alignment = alignment_for_assembly_type(&assembly_type);
        TopLevel::StaticVariable {
            identifier: identifier.clone(),
            global: *global,
            init: *init,
            alignment,
        }
    }

    fn parse_function(
        func: &tacky::TopLevel,
        symbol_table: &semantic_analysis::SymbolTable,
    ) -> TopLevel {
        let tacky::TopLevel::Function {
            name,
            instructions,
            params,
            global,
        } = func
        else {
            panic!("Expected TopLevel::Function in parse_function")
        };
        let mut base_insns = vec![];
        // we move the first 6 params to their respective registers, and
        // the rest to stack offsets starting at -16 and decrementing by 8
        let registers = [
            Register::DI,
            Register::SI,
            Register::DX,
            Register::CX,
            Register::R8,
            Register::R9,
        ];
        let register_args = params.iter().take(6);
        let stack_args = params.iter().skip(6);
        for (idx, p) in register_args.enumerate() {
            debug_assert!(idx < 6);
            let reg = registers[idx];
            let assembly_type = get_assembly_type_from_var_name(p, symbol_table);
            base_insns.push(Instruction::Mov(
                assembly_type,
                Operand::Reg(reg),
                Operand::Pseudo(p.into()),
            ));
        }
        for (idx, p) in stack_args.enumerate() {
            let idx: i32 = idx
                .try_into()
                .expect("Overflow when mapping function params to stack offsets");
            let offset: i32 = 16 + (8 * idx);
            let assembly_type = get_assembly_type_from_var_name(p, symbol_table);
            base_insns.push(Instruction::Mov(
                assembly_type,
                Operand::Stack(offset),
                Operand::Pseudo(p.into()),
            ));
        }

        let instructions = Self::parse_instructions(&instructions, symbol_table);
        base_insns.extend(instructions);
        TopLevel::Func(Function {
            name: name.into(),
            instructions: base_insns,
            global: *global,
        })
    }

    fn parse_instructions(
        ins: &[tacky::Instruction],
        symbol_table: &semantic_analysis::SymbolTable,
    ) -> Vec<Instruction> {
        use Instruction::*;
        use tacky::Instruction as TIns;
        use tacky::UnaryOp as TUnaryOp;
        ins.iter()
            .flat_map(|instruction| match instruction {
                TIns::Ret(val) => {
                    let assembly_type = get_assembly_type_from_val(&val, symbol_table);
                    vec![
                        Mov(assembly_type, val.into(), Operand::Reg(Register::AX)),
                        Ret,
                    ]
                }
                TIns::Unary { op, src, dst } => match op {
                    TUnaryOp::Not => {
                        let a1 = get_assembly_type_from_val(src, symbol_table);
                        let a2 = get_assembly_type_from_val(dst, symbol_table);
                        // !x is the same as x==0, so compare
                        // then zero out dest addr and check if cmp returned equal
                        vec![
                            Cmp(a1, Operand::Imm(0), src.into()),
                            Mov(a2, Operand::Imm(0), dst.into()),
                            SetCC(CondCode::E, dst.into()),
                        ]
                    }
                    o => {
                        let assembly_type = get_assembly_type_from_val(&src, symbol_table);
                        vec![
                            Mov(assembly_type, src.into(), dst.into()),
                            Unary(o.into(), assembly_type, dst.into()),
                        ]
                    }
                },
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
                        let assembly_type = get_assembly_type_from_val(&src1, symbol_table);
                        vec![
                            Mov(assembly_type, src1.into(), dst.into()),
                            Binary(o, assembly_type, src2.into(), dst.into()),
                        ]
                    }
                    tacky::BinaryOp::Equal
                    | tacky::BinaryOp::NotEqual
                    | tacky::BinaryOp::LessThan
                    | tacky::BinaryOp::LessOrEqual
                    | tacky::BinaryOp::GreaterThan
                    | tacky::BinaryOp::GreaterOrEqual => {
                        Self::parse_binary_relational_ops(op, src1, src2, dst, symbol_table)
                    }
                    tacky::BinaryOp::And | tacky::BinaryOp::Or => {
                        unreachable!("BinAnd/BinOr are lowered to jumps")
                    }
                    tacky::BinaryOp::Divide => {
                        let assembly_type = get_assembly_type_from_val(&src1, symbol_table);
                        let ctype = get_ctype_for_val(src1, symbol_table);
                        // if ctype is signed, we sign-extend EAX/RAX (cdq instruction) and issue
                        // idiv.
                        // if ctype is unsigned, we zero out EDX/RDX and issue standard div.
                        // Accumulator stores quotient, so copy EAX/RAX into destination
                        if ctype.signed() {
                            vec![
                                Mov(assembly_type, src1.into(), Operand::Reg(Register::AX)),
                                Cdq(assembly_type),
                                Idiv(assembly_type, src2.into()),
                                Mov(assembly_type, Operand::Reg(Register::AX), dst.into()),
                            ]
                        } else {
                            vec![
                                Mov(assembly_type, src1.into(), Operand::Reg(Register::AX)),
                                Mov(assembly_type, Operand::Imm(0), Operand::Reg(Register::DX)),
                                Div(assembly_type, src2.into()),
                                Mov(assembly_type, Operand::Reg(Register::AX), dst.into()),
                            ]
                        }
                    }
                    tacky::BinaryOp::Remainder => {
                        // if ctype is signed, we sign-extend EAX/RAX (cdq instruction) and issue
                        // idiv.
                        // if ctype is unsigned, we zero out EDX/RDX and issue standard div.
                        // DX stores remainder, so copy EDX/RDX into destination
                        let assembly_type = get_assembly_type_from_val(&src1, symbol_table);
                        let ctype = get_ctype_for_val(src1, symbol_table);
                        if ctype.signed() {
                            vec![
                                Mov(assembly_type, src1.into(), Operand::Reg(Register::AX)),
                                Cdq(assembly_type),
                                Idiv(assembly_type, src2.into()),
                                Mov(assembly_type, Operand::Reg(Register::DX), dst.into()),
                            ]
                        } else {
                            vec![
                                Mov(assembly_type, src1.into(), Operand::Reg(Register::AX)),
                                Mov(assembly_type, Operand::Imm(0), Operand::Reg(Register::DX)),
                                Div(assembly_type, src2.into()),
                                Mov(assembly_type, Operand::Reg(Register::DX), dst.into()),
                            ]
                        }
                    }
                    tacky::BinaryOp::ShiftRight => {
                        // shouldn't expect a shift to change type
                        let assembly_type = get_assembly_type_from_val(src1, symbol_table);
                        vec![
                            Mov(assembly_type, src1.into(), Operand::Reg(Register::AX)),
                            Binary(
                                BinaryOp::ShiftRight,
                                assembly_type,
                                src2.into(),
                                Operand::Reg(Register::AX),
                            ),
                            Mov(assembly_type, Operand::Reg(Register::AX), dst.into()),
                        ]
                    }
                },
                TIns::JumpIfZero { cond, target } => {
                    // comp condition to 0, then jump if equal to target
                    let assembly_type = get_assembly_type_from_val(cond, symbol_table);
                    vec![
                        Cmp(assembly_type, Operand::Imm(0), cond.into()),
                        JmpCC(CondCode::E, target.clone()),
                    ]
                }
                TIns::JumpIfNotZero { cond, target } => {
                    let assembly_type = get_assembly_type_from_val(cond, symbol_table);
                    vec![
                        Cmp(assembly_type, Operand::Imm(0), cond.into()),
                        JmpCC(CondCode::NE, target.clone()),
                    ]
                }
                TIns::Label(ident) => vec![Label(ident.clone())],
                TIns::Copy { src, dst } => {
                    let assembly_type = get_assembly_type_from_val(src, symbol_table);
                    vec![Mov(assembly_type, src.into(), dst.into())]
                }
                TIns::Jump(ident) => vec![Jmp(ident.clone())],
                TIns::FunCall { name, args, dst } => {
                    Self::parse_function_call(name, args, dst, symbol_table)
                }
                TIns::SignExtend { src, dst } => vec![Movsx(src.into(), dst.into())],
                TIns::Truncate { src, dst } => {
                    // truncate by just moving low 4 bytes into destination
                    vec![Mov(AssemblyType::Longword, src.into(), dst.into())]
                }
                TIns::ZeroExtend { src, dst } => {
                    vec![MovZeroExtend(src.into(), dst.into())]
                }
            })
            .collect::<Vec<_>>()
    }

    fn parse_function_call(
        name: &str,
        args: &[tacky::Val],
        dst: &tacky::Val,
        symbol_table: &SymbolTable,
    ) -> Vec<Instruction> {
        let registers = [
            Register::DI,
            Register::SI,
            Register::DX,
            Register::CX,
            Register::R8,
            Register::R9,
        ];
        let register_args = args.iter().take(6);
        let stack_args = args.iter().skip(6);

        // add padding for alignment: we need to be 16-byte aligned,
        // and 8-bytes per arg means that an odd-number of args needs
        // to be padded another 8 bytes
        let stack_padding = if stack_args.len() % 2 == 1 { 8 } else { 0 };

        let mut instructions = vec![];
        if stack_padding != 0 {
            instructions.push(Instruction::Binary(
                BinaryOp::Sub,
                AssemblyType::Quadword,
                Operand::Imm(stack_padding),
                Operand::Reg(Register::SP),
            ));
        }
        // pass register arguments in order
        for (idx, arg) in register_args.enumerate() {
            debug_assert!(idx < registers.len());
            let reg = registers[idx];
            let assembly_type = get_assembly_type_from_val(arg, symbol_table);
            instructions.push(Instruction::Mov(
                assembly_type,
                arg.into(),
                Operand::Reg(reg),
            ));
        }

        // pass stack arguments in reverse order
        for arg in stack_args.rev() {
            let assembly_type = get_assembly_type_from_val(arg, symbol_table);
            let op: Operand = arg.into();
            if assembly_type == AssemblyType::Quadword {
                // we can push 8 byte values onto the stack without issue.
                instructions.push(Instruction::Push(op));
            } else {
                match op {
                    x @ (Operand::Reg(_) | Operand::Imm(_)) => {
                        instructions.push(Instruction::Push(x))
                    }
                    other_op => {
                        instructions.push(Instruction::Mov(
                            AssemblyType::Longword,
                            other_op,
                            Operand::Reg(Register::AX),
                        ));
                        instructions.push(Instruction::Push(Operand::Reg(Register::AX)));
                    }
                }
            }
        }

        // actual function call
        instructions.push(Instruction::Call(name.into()));
        // adjust stack pointer back to where it was before setting up stack.
        // Remove padding + passed arguments
        let stack_args_len = args.iter().skip(6).len();
        let bytes_to_remove = 8 * stack_args_len + stack_padding;
        if bytes_to_remove != 0 {
            instructions.push(Instruction::Binary(
                BinaryOp::Add,
                AssemblyType::Quadword,
                Operand::Imm(bytes_to_remove),
                Operand::Reg(Register::SP),
            ));
        }
        // just move the function return value to the destination register
        let assembly_type = get_assembly_type_from_val(dst, symbol_table);
        instructions.push(Instruction::Mov(
            assembly_type,
            Operand::Reg(Register::AX),
            dst.into(),
        ));
        instructions
    }

    fn parse_binary_relational_ops(
        op: &tacky::BinaryOp,
        src1: &tacky::Val,
        src2: &tacky::Val,
        dst: &tacky::Val,
        symbol_table: &SymbolTable,
    ) -> Vec<Instruction> {
        use Instruction::*;
        use tacky::BinaryOp as TBO;
        let ctype = get_ctype_for_val(src1, symbol_table);
        let cond_code = match op {
            TBO::Equal => CondCode::E,
            TBO::NotEqual => CondCode::NE,
            TBO::GreaterThan if ctype.signed() => CondCode::G,
            TBO::GreaterThan => CondCode::A,
            TBO::GreaterOrEqual if ctype.signed() => CondCode::GE,
            TBO::GreaterOrEqual => CondCode::AE,
            TBO::LessThan if ctype.signed() => CondCode::L,
            TBO::LessThan => CondCode::B,
            TBO::LessOrEqual if ctype.signed() => CondCode::LE,
            TBO::LessOrEqual => CondCode::BE,
            _ => {
                panic!("Unexpected tacky BinaryOp {op:?} when constructing relational instruction")
            }
        };
        let a1 = get_assembly_type_from_val(src1, symbol_table);
        let a2 = get_assembly_type_from_val(dst, symbol_table);
        // turns foo = x < y into
        // cmp y, x AKA x - y
        // zeros out foo
        // setl foo
        vec![
            Cmp(a1, src2.into(), src1.into()),
            Mov(a2, Operand::Imm(0), dst.clone().into()),
            SetCC(cond_code, dst.into()),
        ]
    }

    fn fixup(asm: &mut Asm, generator: &mut AsmGenerator, symbol_table: BackendSymbolTable) {
        match asm {
            Asm::Program(funcs) => {
                for func in funcs {
                    if let TopLevel::Func(func) = func {
                        Self::fixup_function(func, generator, &symbol_table);
                    }
                }
            }
        };
    }
    fn fixup_function(
        func: &mut Function,
        generator: &mut AsmGenerator,
        symbol_table: &BackendSymbolTable,
    ) {
        let Function {
            name: _name,
            instructions,
            ..
        } = func;
        generator.stack_offset = 0; // reset for each function call
        Self::fixup_pseudos_in_instructions(instructions, generator, symbol_table);
        Self::insert_alloc_stack_func(func, generator);
        Self::fixup_invalid_memory_accesses(func);
    }

    fn fixup_pseudos_in_instructions(
        ins: &mut [Instruction],
        generator: &mut AsmGenerator,
        symbol_table: &BackendSymbolTable,
    ) {
        ins.iter_mut().for_each(|instruction| match instruction {
            Instruction::Mov(_, src, dst) => {
                *src = Self::replace_pseudo_in_op(src, generator, symbol_table);
                *dst = Self::replace_pseudo_in_op(dst, generator, symbol_table);
            }
            Instruction::MovZeroExtend(src, dst) => {
                *src = Self::replace_pseudo_in_op(src, generator, symbol_table);
                *dst = Self::replace_pseudo_in_op(dst, generator, symbol_table);
            }
            Instruction::Movsx(src, dst) => {
                *src = Self::replace_pseudo_in_op(src, generator, symbol_table);
                *dst = Self::replace_pseudo_in_op(dst, generator, symbol_table);
            }
            Instruction::Unary(_op, _, dst) => {
                *dst = Self::replace_pseudo_in_op(dst, generator, symbol_table);
            }
            Instruction::Binary(_op, _, src1, src2) => {
                *src1 = Self::replace_pseudo_in_op(src1, generator, symbol_table);
                *src2 = Self::replace_pseudo_in_op(src2, generator, symbol_table);
            }
            Instruction::Idiv(_, src) => {
                *src = Self::replace_pseudo_in_op(src, generator, symbol_table);
            }
            Instruction::Div(_, src) => {
                *src = Self::replace_pseudo_in_op(src, generator, symbol_table);
            }
            Instruction::Cmp(_, src, dst) => {
                *src = Self::replace_pseudo_in_op(src, generator, symbol_table);
                *dst = Self::replace_pseudo_in_op(dst, generator, symbol_table);
            }
            Instruction::SetCC(_cc, dst) => {
                *dst = Self::replace_pseudo_in_op(dst, generator, symbol_table);
            }
            Instruction::Push(op) => *op = Self::replace_pseudo_in_op(op, generator, symbol_table),
            _ => {}
        })
    }

    fn replace_pseudo_in_op(
        op: &Operand,
        generator: &mut AsmGenerator,
        symbol_table: &BackendSymbolTable,
    ) -> Operand {
        match op {
            Operand::Pseudo(var) => {
                // if the pseudo has static linkage, we use a Data operand. Otherwise, fixup with a
                // stack location.
                let entry = symbol_table
                    .get(var)
                    .expect("Should have an entry in backend symbol table for {var:?}");
                let symbol_table::BackendSymTableEntry::ObjEntry { ty, is_static } = entry else {
                    unreachable!(
                        "Got a non-object entry in the backend symbol table when replacing pseudos: {:?}",
                        entry
                    );
                };
                if *is_static {
                    return Operand::Data(var.clone());
                };

                generator
                    .identifiers
                    .entry(var.clone())
                    .or_insert_with(|| {
                        let next_offset = Self::next_aligned_offset(generator.stack_offset, ty);
                        generator.stack_offset = next_offset;
                        Operand::Stack(next_offset)
                    })
                    .clone()
            }
            o => o.clone(), //no transformation otherwise
        }
    }

    fn next_aligned_offset(offset: i32, ty: &AssemblyType) -> i32 {
        // ensures that everything is aligned. Quadwords need to be
        // 8-byte aligned.
        if ty == &AssemblyType::Longword {
            return offset - 4;
        }
        let expected_offset = offset - 8;
        if expected_offset % 8 == 0 {
            return expected_offset;
        }
        (expected_offset as f32 / 8.0).floor() as i32 * 8
    }

    fn insert_alloc_stack_func(func: &mut Function, generator: &AsmGenerator) {
        let old_ins = std::mem::take(&mut func.instructions);
        // round stack offset to next multiple of 16 for easier alignment
        let multiple = generator.stack_offset as f32 / -16.0;
        let rounded = multiple.ceil() as usize;
        let new_offset = rounded * 16;
        let mut v = vec![Instruction::Binary(
            BinaryOp::Sub,
            AssemblyType::Quadword,
            Operand::Imm(new_offset),
            Operand::Reg(Register::SP),
        )];
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
                Instruction::Mov(at, src @ Operand::Stack(_), dst @ Operand::Stack(_))
                | Instruction::Mov(at, src @ Operand::Data(_), dst @ Operand::Data(_))
                | Instruction::Mov(at, src @ Operand::Stack(_), dst @ Operand::Data(_))
                | Instruction::Mov(at, src @ Operand::Data(_), dst @ Operand::Stack(_)) => {
                    // movl can't move from two memory addrs, so
                    // use a temporary variable along the way in %r10d
                    v.push(Instruction::Mov(at, src, Operand::Reg(Register::R10)));
                    v.push(Instruction::Mov(at, Operand::Reg(Register::R10), dst));
                }
                Instruction::Mov(AssemblyType::Longword, src @ Operand::Imm(i), dst)
                    if i > u32::MAX as usize =>
                {
                    // Truncate gets rewritten to a mov, and if i32 doesn't fit then we want to
                    // probably just manually truncate
                    v.push(Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Imm(i & 0xFFFF_FFFF), // keep bottom 32 bits
                        dst,
                    ))
                }
                Instruction::Mov(
                    AssemblyType::Quadword,
                    src @ Operand::Imm(i),
                    dst @ Operand::Data(_),
                )
                | Instruction::Mov(
                    AssemblyType::Quadword,
                    src @ Operand::Imm(i),
                    dst @ Operand::Stack(_),
                ) if i > i32::MAX as usize => {
                    // movq can't move big values into memory but needs an intermediate register
                    v.extend(vec![
                        Instruction::Mov(AssemblyType::Quadword, src, Operand::Reg(Register::R10)),
                        Instruction::Mov(AssemblyType::Quadword, Operand::Reg(Register::R10), dst),
                    ]);
                }
                Instruction::Movsx(_, Operand::Imm(_))
                | Instruction::Movsx(_, Operand::Pseudo(_)) => unreachable!(),
                Instruction::Movsx(src @ Operand::Imm(_), dst @ Operand::Stack(_))
                | Instruction::Movsx(src @ Operand::Imm(_), dst @ Operand::Data(_)) => {
                    // need to use two registers here, R10 and R11
                    // MOV src as a longword into R10, sign extend move into R11,
                    // then MOV as a quadword into dst
                    v.extend(vec![
                        Instruction::Mov(AssemblyType::Longword, src, Operand::Reg(Register::R10)),
                        Instruction::Movsx(
                            Operand::Reg(Register::R10),
                            Operand::Reg(Register::R11),
                        ),
                        Instruction::Mov(AssemblyType::Quadword, Operand::Reg(Register::R11), dst),
                    ])
                }
                Instruction::Movsx(src @ Operand::Imm(_), dst @ Operand::Reg(_)) => {
                    // MOVSX cant use an immediate as a source, so move into R10
                    v.extend(vec![
                        Instruction::Mov(AssemblyType::Longword, src, Operand::Reg(Register::R10)),
                        Instruction::Movsx(Operand::Reg(Register::R10), dst),
                    ])
                }
                Instruction::Movsx(src, dst @ Operand::Stack(_))
                | Instruction::Movsx(src, dst @ Operand::Data(_)) => {
                    // MOVSX can't use a memory address as a destiation, so we movsx into a
                    // register and then MOV into the address.
                    v.extend(vec![
                        Instruction::Movsx(src, Operand::Reg(Register::R10)),
                        Instruction::Mov(AssemblyType::Quadword, Operand::Reg(Register::R10), dst),
                    ])
                }
                Instruction::MovZeroExtend(src, dst @ Operand::Reg(_)) => {
                    // if we're moving and zero extending into a register, just a movl is sufficient
                    v.push(Instruction::Mov(AssemblyType::Longword, src, dst))
                }
                Instruction::MovZeroExtend(src, dst @ (Operand::Data(_) | Operand::Stack(_))) => {
                    // if we're zero extending + moving into a memory address, we first movl into
                    // a register, THEN movq into the actual destination.
                    v.extend(vec![
                        Instruction::Mov(AssemblyType::Longword, src, Operand::Reg(Register::R10)),
                        Instruction::Mov(AssemblyType::Quadword, Operand::Reg(Register::R10), dst),
                    ])
                }
                Instruction::Binary(
                    BinaryOp::Mult,
                    AssemblyType::Quadword,
                    src @ Operand::Imm(i),
                    dst @ Operand::Stack(_),
                )
                | Instruction::Binary(
                    BinaryOp::Mult,
                    AssemblyType::Quadword,
                    src @ Operand::Imm(i),
                    dst @ Operand::Data(_),
                ) if i > i32::MAX as usize => {
                    // special case: multiplying can't use a memory as a destination, so for very
                    // big ints we're going to move into R10, move dest into R11, mutl on registers
                    // and move back into dst at the end.
                    // We catch non-memory-destinations later
                    v.extend(vec![
                        Instruction::Mov(AssemblyType::Quadword, src, Operand::Reg(Register::R10)),
                        Instruction::Mov(
                            AssemblyType::Quadword,
                            dst.clone(),
                            Operand::Reg(Register::R11),
                        ),
                        Instruction::Binary(
                            BinaryOp::Mult,
                            AssemblyType::Quadword,
                            Operand::Reg(Register::R10),
                            Operand::Reg(Register::R11),
                        ),
                        Instruction::Mov(AssemblyType::Quadword, Operand::Reg(Register::R11), dst),
                    ]);
                }
                Instruction::Binary(BinaryOp::Mult, at, src, dst @ Operand::Stack(_))
                | Instruction::Binary(BinaryOp::Mult, at, src, dst @ Operand::Data(_)) => {
                    // imul cannot take an addr as a destination, regardless of src.
                    // Rewrite via register %r11d
                    // Move dst into r11d
                    // Multiply src and r11d, store in r11d
                    // Move r11d into dst
                    v.push(Instruction::Mov(
                        at,
                        dst.clone(),
                        Operand::Reg(Register::R11),
                    ));
                    v.push(Instruction::Binary(
                        BinaryOp::Mult,
                        at,
                        src,
                        Operand::Reg(Register::R11),
                    ));
                    v.push(Instruction::Mov(at, Operand::Reg(Register::R11), dst));
                }
                Instruction::Binary(
                    binop @ BinaryOp::Add,
                    AssemblyType::Quadword,
                    src @ Operand::Imm(i),
                    dst,
                )
                | Instruction::Binary(
                    binop @ BinaryOp::Sub,
                    AssemblyType::Quadword,
                    src @ Operand::Imm(i),
                    dst,
                ) if i > i32::MAX as usize => {
                    // addq, subq require that immediates fit in an int. If not, we use an
                    // intermediary register.
                    v.extend(vec![
                        Instruction::Mov(AssemblyType::Quadword, src, Operand::Reg(Register::R10)),
                        Instruction::Binary(
                            binop,
                            AssemblyType::Quadword,
                            Operand::Reg(Register::R10),
                            dst,
                        ),
                    ])
                }
                Instruction::Binary(
                    binop,
                    at,
                    src @ Operand::Stack(_),
                    dst @ Operand::Stack(_),
                )
                | Instruction::Binary(binop, at, src @ Operand::Data(_), dst @ Operand::Data(_))
                | Instruction::Binary(binop, at, src @ Operand::Stack(_), dst @ Operand::Data(_))
                | Instruction::Binary(binop, at, src @ Operand::Data(_), dst @ Operand::Stack(_))
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
                    v.push(Instruction::Mov(at, src, Operand::Reg(Register::R10)));
                    v.push(Instruction::Binary(
                        binop,
                        at,
                        Operand::Reg(Register::R10),
                        dst,
                    ));
                }
                Instruction::Binary(binop @ BinaryOp::ShiftLeft, at, src, dst)
                | Instruction::Binary(binop @ BinaryOp::ShiftRight, at, src, dst) => {
                    // shift left/right cannot use a memory address as a source.
                    // We move the data to a scratch register. We write to ECX,
                    // then read from the lower 8 bits.
                    if let Operand::Stack(n) = src {
                        v.push(Instruction::Mov(
                            at,
                            Operand::Stack(n),
                            Operand::Reg(Register::CX),
                        ));
                        v.push(Instruction::Binary(
                            binop,
                            at,
                            Operand::Reg(Register::CX),
                            dst,
                        ));
                    } else {
                        v.push(Instruction::Binary(binop, at, src, dst));
                    }
                }
                Instruction::Idiv(at, imm @ Operand::Imm(_)) => {
                    // idiv cannot operate on immediates, so move to a scratch register
                    v.push(Instruction::Mov(at, imm, Operand::Reg(Register::R10)));
                    v.push(Instruction::Idiv(at, Operand::Reg(Register::R10)));
                }
                Instruction::Div(at, imm @ Operand::Imm(_)) => {
                    // div cannot operate on immediates, so move to a scratch register
                    v.push(Instruction::Mov(at, imm, Operand::Reg(Register::R10)));
                    v.push(Instruction::Div(at, Operand::Reg(Register::R10)));
                }
                Instruction::Cmp(at, src @ Operand::Stack(_), dst @ Operand::Stack(_))
                | Instruction::Cmp(at, src @ Operand::Data(_), dst @ Operand::Data(_))
                | Instruction::Cmp(at, src @ Operand::Stack(_), dst @ Operand::Data(_))
                | Instruction::Cmp(at, src @ Operand::Data(_), dst @ Operand::Stack(_)) => {
                    // cmpl can't move from two memory addrs, so
                    // use a temporary variable along the way in %r10d
                    v.push(Instruction::Mov(at, src, Operand::Reg(Register::R10)));
                    v.push(Instruction::Cmp(at, Operand::Reg(Register::R10), dst));
                }
                Instruction::Cmp(at, src, dst @ Operand::Imm(_)) => {
                    // if we're working with quadwords and src is an immediate that doesn't fit in
                    // an int, we'll need to move that into a register as well
                    let mut moved_src_to_reg = false;
                    if at == AssemblyType::Quadword {
                        if let Operand::Imm(i) = src
                            && i > i32::MAX as usize
                        {
                            v.push(Instruction::Mov(
                                at,
                                src.clone(),
                                Operand::Reg(Register::R10),
                            ));
                            moved_src_to_reg = true;
                        }
                    }
                    // cmpl can't use a constant as a destination so move into reg
                    // use a temporary variable along the way in %r11d
                    v.push(Instruction::Mov(
                        at,
                        dst.clone(),
                        Operand::Reg(Register::R11),
                    ));
                    let src = if moved_src_to_reg {
                        Operand::Reg(Register::R10)
                    } else {
                        src
                    };
                    v.push(Instruction::Cmp(at, src, Operand::Reg(Register::R11)));
                }
                Instruction::Cmp(AssemblyType::Quadword, src @ Operand::Imm(i), dst)
                    if i > i32::MAX as usize =>
                {
                    // need to move src into a register
                    v.extend(vec![
                        Instruction::Mov(AssemblyType::Quadword, src, Operand::Reg(Register::R10)),
                        Instruction::Cmp(AssemblyType::Quadword, Operand::Reg(Register::R10), dst),
                    ])
                }
                Instruction::Push(src @ Operand::Imm(i)) if i > i32::MAX as usize => {
                    // need to move operand into register
                    v.extend(vec![
                        Instruction::Mov(AssemblyType::Quadword, src, Operand::Reg(Register::R10)),
                        Instruction::Push(Operand::Reg(Register::R10)),
                    ])
                }
                i => v.push(i),
            }
        }
        func.instructions = v;
    }
}

fn get_assembly_type_from_val(val: &tacky::Val, symbol_table: &SymbolTable) -> AssemblyType {
    match val {
        Val::Constant(Const::Int(_) | Const::UInt(_)) => AssemblyType::Longword,
        Val::Constant(Const::Long(_) | Const::ULong(_)) => AssemblyType::Quadword,
        Val::Constant(Const::Double(_)) => todo!(),
        Val::Var(s) => get_assembly_type_from_var_name(s.as_str(), symbol_table),
    }
}

fn get_assembly_type_from_var_name(var: &str, symbol_table: &SymbolTable) -> AssemblyType {
    let (ctype, _attr) = symbol_table
        .get(var)
        .expect("Expected entry for {s:?} in symbol table");
    match ctype {
        CType::Int | CType::UInt => AssemblyType::Longword,
        CType::Long | CType::ULong => AssemblyType::Quadword,
        CType::Double => todo!(),
        CType::FunType { .. } => unreachable!(),
    }
}

fn alignment_for_assembly_type(at: &AssemblyType) -> usize {
    match at {
        AssemblyType::Longword => 4,
        AssemblyType::Quadword => 8,
    }
}

fn get_ctype_for_val(val: &Val, symbol_table: &SymbolTable) -> CType {
    match val {
        Val::Constant(c) => c.to_ctype(),
        Val::Var(s) => symbol_table
            .get(s)
            .expect("Missing symbol for {s:?} in frontend symbol table")
            .0
            .clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Const;
    use crate::semantic_analysis;
    use crate::symbol_table::IdentifierAttrs;
    use crate::tacky;
    use crate::tacky::UnaryOp as TUOp;

    fn int_sym_table(vars: &[&str]) -> semantic_analysis::SymbolTable {
        let mut table = semantic_analysis::SymbolTable::new();
        for &var in vars {
            table.insert(var.to_string(), (CType::Int, IdentifierAttrs::LocalAttr));
        }
        table
    }

    fn sym_table(vars: &[(&str, CType)]) -> semantic_analysis::SymbolTable {
        let mut table = semantic_analysis::SymbolTable::new();
        for &(var, ref ctype) in vars {
            table.insert(var.to_string(), (ctype.clone(), IdentifierAttrs::LocalAttr));
        }
        table
    }
    #[test]
    fn basic_parse() {
        let ast = tacky::AST::Program(vec![tacky::TopLevel::Function {
            name: "main".into(),
            params: vec![],
            global: true,
            instructions: vec![tacky::Instruction::Ret(tacky::Val::Constant(Const::Int(
                100,
            )))],
        }]);

        let expected = Asm::Program(vec![TopLevel::Func(Function {
            name: "main".into(),
            global: true,
            instructions: vec![
                Instruction::Binary(
                    BinaryOp::Sub,
                    AssemblyType::Quadword,
                    Operand::Imm(0),
                    Operand::Reg(Register::SP),
                ),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Imm(100),
                    Operand::Reg(Register::AX),
                ),
                Instruction::Ret,
            ],
        })]);

        let assembly = Asm::from_tacky(ast, int_sym_table(&[]));
        assert_eq!(assembly, expected);
    }

    #[test]
    fn parse_with_pseudos() {
        let ast = tacky::AST::Program(vec![tacky::TopLevel::Function {
            name: "main".into(),
            params: vec![],
            global: true,
            instructions: vec![
                tacky::Instruction::Unary {
                    op: tacky::UnaryOp::Negate,
                    src: tacky::Val::Constant(Const::Int(100)),
                    dst: tacky::Val::Var("tmp.0".into()),
                },
                tacky::Instruction::Ret(tacky::Val::Var("tmp.0".into())),
            ],
        }]);

        let expected = Asm::Program(vec![TopLevel::Func(Function {
            name: "main".into(),
            global: true,
            instructions: vec![
                Instruction::Binary(
                    BinaryOp::Sub,
                    AssemblyType::Quadword,
                    Operand::Imm(16),
                    Operand::Reg(Register::SP),
                ),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Imm(100),
                    Operand::Stack(-4),
                ),
                Instruction::Unary(UnaryOp::Neg, AssemblyType::Longword, Operand::Stack(-4)),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Stack(-4),
                    Operand::Reg(Register::AX),
                ),
                Instruction::Ret,
            ],
        })]);

        let assembly = Asm::from_tacky(ast, int_sym_table(&["tmp.0"]));
        assert_eq!(assembly, expected);
    }

    #[test]
    fn parse_nested_unaries() {
        let ast = tacky::AST::Program(vec![tacky::TopLevel::Function {
            name: "main".into(),
            params: vec![],
            global: true,
            instructions: vec![
                tacky::Instruction::Unary {
                    op: tacky::UnaryOp::Negate,
                    src: tacky::Val::Constant(Const::Int(100)),
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
        }]);

        let expected = Asm::Program(vec![TopLevel::Func(Function {
            name: "main".into(),
            global: true,
            instructions: vec![
                Instruction::Binary(
                    BinaryOp::Sub,
                    AssemblyType::Quadword,
                    Operand::Imm(16),
                    Operand::Reg(Register::SP),
                ),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Imm(100),
                    Operand::Stack(-4),
                ),
                Instruction::Unary(UnaryOp::Neg, AssemblyType::Longword, Operand::Stack(-4)),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Stack(-4),
                    Operand::Reg(Register::R10),
                ),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Reg(Register::R10),
                    Operand::Stack(-8),
                ),
                Instruction::Unary(UnaryOp::Not, AssemblyType::Longword, Operand::Stack(-8)),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Stack(-8),
                    Operand::Reg(Register::R10),
                ),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Reg(Register::R10),
                    Operand::Stack(-12),
                ),
                Instruction::Unary(UnaryOp::Neg, AssemblyType::Longword, Operand::Stack(-12)),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Stack(-12),
                    Operand::Reg(Register::AX),
                ),
                Instruction::Ret,
            ],
        })]);

        let assembly = Asm::from_tacky(ast, int_sym_table(&["tmp.0", "tmp.1", "tmp.2"]));
        assert_eq!(assembly, expected);
    }

    #[test]
    fn generate_binary_expressions() {
        let ast = tacky::AST::Program(vec![tacky::TopLevel::Function {
            name: "main".into(),
            params: vec![],
            global: true,
            instructions: vec![
                tacky::Instruction::Binary {
                    op: tacky::BinaryOp::Multiply,
                    src1: tacky::Val::Constant(Const::Int(1)),
                    src2: tacky::Val::Constant(Const::Int(2)),
                    dst: tacky::Val::Var("tmp.0".into()),
                },
                tacky::Instruction::Binary {
                    op: tacky::BinaryOp::Add,
                    src1: tacky::Val::Constant(Const::Int(4)),
                    src2: tacky::Val::Constant(Const::Int(5)),
                    dst: tacky::Val::Var("tmp.1".into()),
                },
                tacky::Instruction::Binary {
                    op: tacky::BinaryOp::Remainder,
                    src1: tacky::Val::Constant(Const::Int(3)),
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
        }]);
        let expected = Asm::Program(vec![TopLevel::Func(Function {
            name: "main".into(),
            global: true,
            instructions: vec![
                Instruction::Binary(
                    BinaryOp::Sub,
                    AssemblyType::Quadword,
                    Operand::Imm(16),
                    Operand::Reg(Register::SP),
                ),
                // tmp0 = 1 * 2
                Instruction::Mov(AssemblyType::Longword, Operand::Imm(1), Operand::Stack(-4)),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Stack(-4),
                    Operand::Reg(Register::R11),
                ),
                Instruction::Binary(
                    BinaryOp::Mult,
                    AssemblyType::Longword,
                    Operand::Imm(2),
                    Operand::Reg(Register::R11),
                ),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Reg(Register::R11),
                    Operand::Stack(-4),
                ),
                // tmp1 = 4 + 5
                Instruction::Mov(AssemblyType::Longword, Operand::Imm(4), Operand::Stack(-8)),
                Instruction::Binary(
                    BinaryOp::Add,
                    AssemblyType::Longword,
                    Operand::Imm(5),
                    Operand::Stack(-8),
                ),
                // tmp2 = 3 % tmp1
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Imm(3),
                    Operand::Reg(Register::AX),
                ),
                Instruction::Cdq(AssemblyType::Longword),
                Instruction::Idiv(AssemblyType::Longword, Operand::Stack(-8)),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Reg(Register::DX),
                    Operand::Stack(-12),
                ),
                // tmp3 = tmp0 / tmp2
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Stack(-4),
                    Operand::Reg(Register::AX),
                ),
                Instruction::Cdq(AssemblyType::Longword),
                Instruction::Idiv(AssemblyType::Longword, Operand::Stack(-12)),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Reg(Register::AX),
                    Operand::Stack(-16),
                ),
                // return
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Stack(-16),
                    Operand::Reg(Register::AX),
                ),
                Instruction::Ret,
            ],
        })]);

        let assembly = Asm::from_tacky(ast, int_sym_table(&["tmp.0", "tmp.1", "tmp.2", "tmp.3"]));
        assert_eq!(assembly, expected);
    }

    #[test]
    fn complex_binary_expressions() {
        let ast = tacky::AST::Program(vec![tacky::TopLevel::Function {
            name: "main".into(),
            params: vec![],
            global: true,
            instructions: vec![
                tacky::Instruction::Binary {
                    op: tacky::BinaryOp::Multiply,
                    src1: tacky::Val::Constant(Const::Int(5)),
                    src2: tacky::Val::Constant(Const::Int(4)),
                    dst: tacky::Val::Var("tmp.0".into()),
                },
                tacky::Instruction::Binary {
                    op: tacky::BinaryOp::Divide,
                    src1: tacky::Val::Var("tmp.0".into()),
                    src2: tacky::Val::Constant(Const::Int(2)),
                    dst: tacky::Val::Var("tmp.1".into()),
                },
                tacky::Instruction::Binary {
                    op: tacky::BinaryOp::Add,
                    src1: tacky::Val::Constant(Const::Int(2)),
                    src2: tacky::Val::Constant(Const::Int(1)),
                    dst: tacky::Val::Var("tmp.2".into()),
                },
                tacky::Instruction::Binary {
                    op: tacky::BinaryOp::Remainder,
                    src1: tacky::Val::Constant(Const::Int(3)),
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
        }]);

        let expected = Asm::Program(vec![TopLevel::Func(Function {
            name: "main".into(),
            global: true,
            instructions: vec![
                Instruction::Binary(
                    BinaryOp::Sub,
                    AssemblyType::Quadword,
                    Operand::Imm(32),
                    Operand::Reg(Register::SP),
                ),
                // tmp0 = 5 * 4 = 20
                Instruction::Mov(AssemblyType::Longword, Operand::Imm(5), Operand::Stack(-4)),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Stack(-4),
                    Operand::Reg(Register::R11),
                ),
                Instruction::Binary(
                    BinaryOp::Mult,
                    AssemblyType::Longword,
                    Operand::Imm(4),
                    Operand::Reg(Register::R11),
                ),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Reg(Register::R11),
                    Operand::Stack(-4),
                ),
                // tmp1 = tmp0 / 2 = 10
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Stack(-4),
                    Operand::Reg(Register::AX),
                ),
                Instruction::Cdq(AssemblyType::Longword),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Imm(2),
                    Operand::Reg(Register::R10),
                ),
                Instruction::Idiv(AssemblyType::Longword, Operand::Reg(Register::R10)),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Reg(Register::AX),
                    Operand::Stack(-8),
                ),
                // tmp2 = 2 + 1  = 3
                Instruction::Mov(AssemblyType::Longword, Operand::Imm(2), Operand::Stack(-12)),
                Instruction::Binary(
                    BinaryOp::Add,
                    AssemblyType::Longword,
                    Operand::Imm(1),
                    Operand::Stack(-12),
                ),
                // tmp3 = 3 % tmp2 = 0
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Imm(3),
                    Operand::Reg(Register::AX),
                ),
                Instruction::Cdq(AssemblyType::Longword),
                Instruction::Idiv(AssemblyType::Longword, Operand::Stack(-12)),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Reg(Register::DX),
                    Operand::Stack(-16),
                ),
                // tmp3 = tmp1 - tmp3 = 10
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Stack(-8),
                    Operand::Reg(Register::R10),
                ),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Reg(Register::R10),
                    Operand::Stack(-20),
                ),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Stack(-16),
                    Operand::Reg(Register::R10),
                ),
                Instruction::Binary(
                    BinaryOp::Sub,
                    AssemblyType::Longword,
                    Operand::Reg(Register::R10),
                    Operand::Stack(-20),
                ),
                // return
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Stack(-20),
                    Operand::Reg(Register::AX),
                ),
                Instruction::Ret,
            ],
        })]);

        let assembly = Asm::from_tacky(
            ast,
            int_sym_table(&["tmp.0", "tmp.1", "tmp.2", "tmp.3", "tmp.4"]),
        );
        assert_eq!(assembly, expected);
    }

    #[test]
    fn simple_bitwise() {
        let ast = tacky::AST::Program(vec![tacky::TopLevel::Function {
            name: "main".into(),
            params: vec![],
            global: true,
            instructions: vec![
                tacky::Instruction::Binary {
                    op: tacky::BinaryOp::Multiply,
                    src1: tacky::Val::Constant(Const::Int(5)),
                    src2: tacky::Val::Constant(Const::Int(4)),
                    dst: tacky::Val::Var("tmp.0".into()),
                },
                tacky::Instruction::Binary {
                    op: tacky::BinaryOp::Subtract,
                    src1: tacky::Val::Constant(Const::Int(4)),
                    src2: tacky::Val::Constant(Const::Int(5)),
                    dst: tacky::Val::Var("tmp.1".into()),
                },
                tacky::Instruction::Binary {
                    op: tacky::BinaryOp::BitwiseAnd,
                    src1: tacky::Val::Var("tmp.1".into()),
                    src2: tacky::Val::Constant(Const::Int(6)),
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
        }]);
        let expected = Asm::Program(vec![TopLevel::Func(Function {
            name: "main".into(),
            global: true,
            instructions: vec![
                Instruction::Binary(
                    BinaryOp::Sub,
                    AssemblyType::Quadword,
                    Operand::Imm(16),
                    Operand::Reg(Register::SP),
                ),
                // tmp0 = 5 * 4
                Instruction::Mov(AssemblyType::Longword, Operand::Imm(5), Operand::Stack(-4)),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Stack(-4),
                    Operand::Reg(Register::R11),
                ),
                Instruction::Binary(
                    BinaryOp::Mult,
                    AssemblyType::Longword,
                    Operand::Imm(4),
                    Operand::Reg(Register::R11),
                ),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Reg(Register::R11),
                    Operand::Stack(-4),
                ),
                // tmp1 = 4 - 5
                Instruction::Mov(AssemblyType::Longword, Operand::Imm(4), Operand::Stack(-8)),
                Instruction::Binary(
                    BinaryOp::Sub,
                    AssemblyType::Longword,
                    Operand::Imm(5),
                    Operand::Stack(-8),
                ),
                // tmp2 = tmp1 & 6
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Stack(-8),
                    Operand::Reg(Register::R10),
                ),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Reg(Register::R10),
                    Operand::Stack(-12),
                ),
                Instruction::Binary(
                    BinaryOp::BitwiseAnd,
                    AssemblyType::Longword,
                    Operand::Imm(6),
                    Operand::Stack(-12),
                ),
                // tmp3 = tmp0 | tmp2
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Stack(-4),
                    Operand::Reg(Register::R10),
                ),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Reg(Register::R10),
                    Operand::Stack(-16),
                ),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Stack(-12),
                    Operand::Reg(Register::R10),
                ),
                Instruction::Binary(
                    BinaryOp::BitwiseOr,
                    AssemblyType::Longword,
                    Operand::Reg(Register::R10),
                    Operand::Stack(-16),
                ),
                // return
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Stack(-16),
                    Operand::Reg(Register::AX),
                ),
                Instruction::Ret,
            ],
        })]);
        let assembly = Asm::from_tacky(ast, int_sym_table(&["tmp.0", "tmp.1", "tmp.2", "tmp.3"]));
        assert_eq!(assembly, expected);
    }

    #[test]
    fn shiftleft() {
        let ast = tacky::AST::Program(vec![tacky::TopLevel::Function {
            name: "main".into(),
            params: vec![],
            global: true,
            instructions: vec![
                tacky::Instruction::Binary {
                    op: tacky::BinaryOp::Multiply,
                    src1: tacky::Val::Constant(Const::Int(5)),
                    src2: tacky::Val::Constant(Const::Int(4)),
                    dst: tacky::Val::Var("tmp.0".into()),
                },
                tacky::Instruction::Binary {
                    op: tacky::BinaryOp::ShiftLeft,
                    src1: tacky::Val::Var("tmp.0".into()),
                    src2: tacky::Val::Constant(Const::Int(2)),
                    dst: tacky::Val::Var("tmp.1".into()),
                },
                tacky::Instruction::Ret(tacky::Val::Var("tmp.1".into())),
            ],
        }]);

        let expected = Asm::Program(vec![TopLevel::Func(Function {
            name: "main".into(),
            global: true,
            instructions: vec![
                Instruction::Binary(
                    BinaryOp::Sub,
                    AssemblyType::Quadword,
                    Operand::Imm(16),
                    Operand::Reg(Register::SP),
                ),
                // tmp0 = 5 * 4
                Instruction::Mov(AssemblyType::Longword, Operand::Imm(5), Operand::Stack(-4)),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Stack(-4),
                    Operand::Reg(Register::R11),
                ),
                Instruction::Binary(
                    BinaryOp::Mult,
                    AssemblyType::Longword,
                    Operand::Imm(4),
                    Operand::Reg(Register::R11),
                ),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Reg(Register::R11),
                    Operand::Stack(-4),
                ),
                // tmp1 = tmp.0 << 2
                // moves tmp.8 into tmp.1 via reg10
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Stack(-4),
                    Operand::Reg(Register::R10),
                ),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Reg(Register::R10),
                    Operand::Stack(-8),
                ),
                Instruction::Binary(
                    BinaryOp::ShiftLeft,
                    AssemblyType::Longword,
                    Operand::Imm(2),
                    Operand::Stack(-8),
                ),
                // return
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Stack(-8),
                    Operand::Reg(Register::AX),
                ),
                Instruction::Ret,
            ],
        })]);
        let assembly = Asm::from_tacky(ast, int_sym_table(&["tmp.0", "tmp.1"]));
        assert_eq!(assembly, expected);
    }

    #[test]
    fn shiftright_lhs_is_negative() {
        let ast = tacky::AST::Program(vec![tacky::TopLevel::Function {
            name: "main".into(),
            params: vec![],
            global: true,
            instructions: vec![
                tacky::Instruction::Unary {
                    op: tacky::UnaryOp::Negate,
                    src: tacky::Val::Constant(Const::Int(5)),
                    dst: tacky::Val::Var("tmp.0".into()),
                },
                tacky::Instruction::Binary {
                    op: tacky::BinaryOp::ShiftRight,
                    src1: tacky::Val::Var("tmp.0".into()),
                    src2: tacky::Val::Constant(Const::Int(30)),
                    dst: tacky::Val::Var("tmp.1".into()),
                },
                tacky::Instruction::Ret(tacky::Val::Var("tmp.1".into())),
            ],
        }]);
        let expected = Asm::Program(vec![TopLevel::Func(Function {
            name: "main".into(),
            global: true,
            instructions: vec![
                Instruction::Binary(
                    BinaryOp::Sub,
                    AssemblyType::Quadword,
                    Operand::Imm(16),
                    Operand::Reg(Register::SP),
                ),
                // tmp0 = -5
                Instruction::Mov(AssemblyType::Longword, Operand::Imm(5), Operand::Stack(-4)),
                Instruction::Unary(UnaryOp::Neg, AssemblyType::Longword, Operand::Stack(-4)),
                // tmp1 = tmp.0 >> 30
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Stack(-4),
                    Operand::Reg(Register::AX),
                ),
                Instruction::Binary(
                    BinaryOp::ShiftRight,
                    AssemblyType::Longword,
                    Operand::Imm(30),
                    Operand::Reg(Register::AX),
                ),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Reg(Register::AX),
                    Operand::Stack(-8),
                ),
                // return
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Stack(-8),
                    Operand::Reg(Register::AX),
                ),
                Instruction::Ret,
            ],
        })]);
        let assembly = Asm::from_tacky(ast, int_sym_table(&["tmp.0", "tmp.1"]));
        assert_eq!(assembly, expected);
    }

    #[test]
    fn shiftleft_rhs_is_expr() {
        let ast = tacky::AST::Program(vec![tacky::TopLevel::Function {
            name: "main".into(),
            params: vec![],
            global: true,
            instructions: vec![
                tacky::Instruction::Binary {
                    op: tacky::BinaryOp::Add,
                    src1: tacky::Val::Constant(Const::Int(1)),
                    src2: tacky::Val::Constant(Const::Int(2)),
                    dst: tacky::Val::Var("tmp.0".into()),
                },
                tacky::Instruction::Binary {
                    op: tacky::BinaryOp::ShiftLeft,
                    src1: tacky::Val::Constant(Const::Int(5)),
                    src2: tacky::Val::Var("tmp.0".into()),
                    dst: tacky::Val::Var("tmp.1".into()),
                },
                tacky::Instruction::Ret(tacky::Val::Var("tmp.1".into())),
            ],
        }]);
        let expected = Asm::Program(vec![TopLevel::Func(Function {
            name: "main".into(),
            global: true,
            instructions: vec![
                Instruction::Binary(
                    BinaryOp::Sub,
                    AssemblyType::Quadword,
                    Operand::Imm(16),
                    Operand::Reg(Register::SP),
                ),
                // tmp0 = 1 + 2
                Instruction::Mov(AssemblyType::Longword, Operand::Imm(1), Operand::Stack(-4)),
                Instruction::Binary(
                    BinaryOp::Add,
                    AssemblyType::Longword,
                    Operand::Imm(2),
                    Operand::Stack(-4),
                ),
                // tmp1 = 5 << tmp.0
                Instruction::Mov(AssemblyType::Longword, Operand::Imm(5), Operand::Stack(-8)),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Stack(-4),
                    Operand::Reg(Register::CX),
                ),
                Instruction::Binary(
                    BinaryOp::ShiftLeft,
                    AssemblyType::Longword,
                    Operand::Reg(Register::CX),
                    Operand::Stack(-8),
                ),
                // return
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Stack(-8),
                    Operand::Reg(Register::AX),
                ),
                Instruction::Ret,
            ],
        })]);
        let assembly = Asm::from_tacky(ast, int_sym_table(&["tmp.0", "tmp.1"]));
        assert_eq!(assembly, expected);
    }

    #[test]
    fn unary_not() {
        let ast = tacky::AST::Program(vec![tacky::TopLevel::Function {
            name: "main".into(),
            params: vec![],
            global: true,
            instructions: vec![
                tacky::Instruction::Unary {
                    op: TUOp::Not,
                    src: tacky::Val::Constant(Const::Int(1)),
                    dst: tacky::Val::Var("tmp.0".into()),
                },
                tacky::Instruction::Ret(tacky::Val::Var("tmp.0".into())),
            ],
        }]);
        let expected = Asm::Program(vec![TopLevel::Func(Function {
            name: "main".into(),
            global: true,
            // move 1 into register 11,
            // then check if 1 == 0
            // clear out the next address, then check if cmp set ZF
            // and write to stack addr -4
            // move stack addr -4 to EAX and return
            instructions: vec![
                Instruction::Binary(
                    BinaryOp::Sub,
                    AssemblyType::Quadword,
                    Operand::Imm(16),
                    Operand::Reg(Register::SP),
                ),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Imm(1),
                    Operand::Reg(Register::R11),
                ),
                Instruction::Cmp(
                    AssemblyType::Longword,
                    Operand::Imm(0),
                    Operand::Reg(Register::R11),
                ),
                Instruction::Mov(AssemblyType::Longword, Operand::Imm(0), Operand::Stack(-4)),
                Instruction::SetCC(CondCode::E, Operand::Stack(-4)),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Stack(-4),
                    Operand::Reg(Register::AX),
                ),
                Instruction::Ret,
            ],
        })]);
        let assembly = Asm::from_tacky(ast, int_sym_table(&["tmp.0"]));
        assert_eq!(assembly, expected);
    }

    #[test]
    fn binary_greater_or_equal() {
        let ast = tacky::AST::Program(vec![tacky::TopLevel::Function {
            name: "main".into(),
            params: vec![],
            global: true,
            instructions: vec![
                tacky::Instruction::Binary {
                    op: tacky::BinaryOp::GreaterOrEqual,
                    src1: tacky::Val::Var("tmp.0".into()),
                    src2: tacky::Val::Constant(Const::Int(2)),
                    dst: tacky::Val::Var("tmp.1".into()),
                },
                tacky::Instruction::Ret(tacky::Val::Var("tmp.1".into())),
            ],
        }]);
        let expected = Asm::Program(vec![TopLevel::Func(Function {
            name: "main".into(),
            global: true,
            instructions: vec![
                Instruction::Binary(
                    BinaryOp::Sub,
                    AssemblyType::Quadword,
                    Operand::Imm(16),
                    Operand::Reg(Register::SP),
                ),
                Instruction::Cmp(AssemblyType::Longword, Operand::Imm(2), Operand::Stack(-4)),
                Instruction::Mov(AssemblyType::Longword, Operand::Imm(0), Operand::Stack(-8)),
                Instruction::SetCC(CondCode::GE, Operand::Stack(-8)),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Stack(-8),
                    Operand::Reg(Register::AX),
                ),
                Instruction::Ret,
            ],
        })]);
        let assembly = Asm::from_tacky(ast, int_sym_table(&["tmp.0", "tmp.1"]));
        assert_eq!(assembly, expected);
    }

    #[test]
    fn jump_if_zero() {
        let ast = tacky::AST::Program(vec![tacky::TopLevel::Function {
            name: "main".into(),
            params: vec![],
            global: true,
            instructions: vec![
                tacky::Instruction::Copy {
                    src: tacky::Val::Constant(Const::Int(5)),
                    dst: tacky::Val::Var("tmp.0".into()),
                },
                tacky::Instruction::JumpIfZero {
                    cond: tacky::Val::Var("tmp.0".into()),
                    target: "and_expr_false.0".into(),
                },
                tacky::Instruction::Binary {
                    op: tacky::BinaryOp::Add,
                    src1: tacky::Val::Constant(Const::Int(1)),
                    src2: tacky::Val::Constant(Const::Int(2)),
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
                    src: tacky::Val::Constant(Const::Int(1)),
                    dst: tacky::Val::Var("tmp.3".into()),
                },
                tacky::Instruction::Jump("and_expr_end.1".into()),
                tacky::Instruction::Label("and_expr_false.0".into()),
                tacky::Instruction::Copy {
                    src: tacky::Val::Constant(Const::Int(0)),
                    dst: tacky::Val::Var("tmp.3".into()),
                },
                tacky::Instruction::Label("and_expr_end.1".into()),
                tacky::Instruction::Ret(tacky::Val::Var("tmp.3".into())),
            ],
        }]);

        let expected = Asm::Program(vec![TopLevel::Func(Function {
            name: "main".into(),
            global: true,
            instructions: vec![
                Instruction::Binary(
                    BinaryOp::Sub,
                    AssemblyType::Quadword,
                    Operand::Imm(16),
                    Operand::Reg(Register::SP),
                ),
                Instruction::Mov(AssemblyType::Longword, Operand::Imm(5), Operand::Stack(-4)),
                Instruction::Cmp(AssemblyType::Longword, Operand::Imm(0), Operand::Stack(-4)),
                Instruction::JmpCC(CondCode::E, "and_expr_false.0".into()),
                Instruction::Mov(AssemblyType::Longword, Operand::Imm(1), Operand::Stack(-8)),
                Instruction::Binary(
                    BinaryOp::Add,
                    AssemblyType::Longword,
                    Operand::Imm(2),
                    Operand::Stack(-8),
                ),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Stack(-8),
                    Operand::Reg(Register::R10),
                ),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Reg(Register::R10),
                    Operand::Stack(-12),
                ),
                Instruction::Cmp(AssemblyType::Longword, Operand::Imm(0), Operand::Stack(-12)),
                Instruction::JmpCC(CondCode::E, "and_expr_false.0".into()),
                Instruction::Mov(AssemblyType::Longword, Operand::Imm(1), Operand::Stack(-16)),
                Instruction::Jmp("and_expr_end.1".into()),
                Instruction::Label("and_expr_false.0".into()),
                Instruction::Mov(AssemblyType::Longword, Operand::Imm(0), Operand::Stack(-16)),
                Instruction::Label("and_expr_end.1".into()),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Stack(-16),
                    Operand::Reg(Register::AX),
                ),
                Instruction::Ret,
            ],
        })]);
        let assembly = Asm::from_tacky(ast, int_sym_table(&["tmp.0", "tmp.1", "tmp.2", "tmp.3"]));
        assert_eq!(assembly, expected);
    }

    #[test]
    fn functions() {
        let src = r#"
            int bar(int a);
            int foo(int x, int y) { 
                return x + y;
            }
            int main(void) {
                return foo(1, 2) + 3;
            }
            int many_args(int a, int b, int c, int d, int e, int f, int g, int h) {
                return a;     
            }
        "#;
        let lexer = crate::lexer::Lexer::lex(src).unwrap();
        let tokens = lexer.as_syntactic_tokens();
        let parse = crate::parser::Parser::new(&tokens);
        let mut ast = parse.into_ast().unwrap();
        let (mut symbol_table, ast) = crate::semantic_analysis::resolve(&mut ast).unwrap();
        let tacky = crate::tacky::Tacky::new(ast);
        let tacky = tacky.into_ast(&mut symbol_table);
        let Ok(tacky_ast) = tacky else {
            panic!();
        };
        let actual = Asm::from_tacky(tacky_ast, symbol_table);
        let expected = Asm::Program(vec![
            TopLevel::Func(Function {
                name: "foo".into(),
                global: true,
                instructions: vec![
                    Instruction::Binary(
                        BinaryOp::Sub,
                        AssemblyType::Quadword,
                        Operand::Imm(16),
                        Operand::Reg(Register::SP),
                    ),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Reg(Register::DI),
                        Operand::Stack(-4),
                    ),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Reg(Register::SI),
                        Operand::Stack(-8),
                    ),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Stack(-4),
                        Operand::Reg(Register::R10),
                    ),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Reg(Register::R10),
                        Operand::Stack(-12),
                    ),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Stack(-8),
                        Operand::Reg(Register::R10),
                    ),
                    Instruction::Binary(
                        BinaryOp::Add,
                        AssemblyType::Longword,
                        Operand::Reg(Register::R10),
                        Operand::Stack(-12),
                    ),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Stack(-12),
                        Operand::Reg(Register::AX),
                    ),
                    Instruction::Ret,
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Imm(0),
                        Operand::Reg(Register::AX),
                    ),
                    Instruction::Ret,
                ],
            }),
            TopLevel::Func(Function {
                name: "main".into(),
                global: true,
                instructions: vec![
                    Instruction::Binary(
                        BinaryOp::Sub,
                        AssemblyType::Quadword,
                        Operand::Imm(16),
                        Operand::Reg(Register::SP),
                    ),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Imm(1),
                        Operand::Reg(Register::DI),
                    ),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Imm(2),
                        Operand::Reg(Register::SI),
                    ),
                    Instruction::Call("foo".into()),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Reg(Register::AX),
                        Operand::Stack(-4),
                    ),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Stack(-4),
                        Operand::Reg(Register::R10),
                    ),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Reg(Register::R10),
                        Operand::Stack(-8),
                    ),
                    Instruction::Binary(
                        BinaryOp::Add,
                        AssemblyType::Longword,
                        Operand::Imm(3),
                        Operand::Stack(-8),
                    ),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Stack(-8),
                        Operand::Reg(Register::AX),
                    ),
                    Instruction::Ret,
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Imm(0),
                        Operand::Reg(Register::AX),
                    ),
                    Instruction::Ret,
                ],
            }),
            TopLevel::Func(Function {
                name: "many_args".into(),
                global: true,
                instructions: vec![
                    Instruction::Binary(
                        BinaryOp::Sub,
                        AssemblyType::Quadword,
                        Operand::Imm(32),
                        Operand::Reg(Register::SP),
                    ),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Reg(Register::DI),
                        Operand::Stack(-4),
                    ),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Reg(Register::SI),
                        Operand::Stack(-8),
                    ),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Reg(Register::DX),
                        Operand::Stack(-12),
                    ),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Reg(Register::CX),
                        Operand::Stack(-16),
                    ),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Reg(Register::R8),
                        Operand::Stack(-20),
                    ),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Reg(Register::R9),
                        Operand::Stack(-24),
                    ),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Stack(16),
                        Operand::Reg(Register::R10),
                    ),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Reg(Register::R10),
                        Operand::Stack(-28),
                    ),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Stack(24),
                        Operand::Reg(Register::R10),
                    ),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Reg(Register::R10),
                        Operand::Stack(-32),
                    ),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Stack(-4),
                        Operand::Reg(Register::AX),
                    ),
                    Instruction::Ret,
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Imm(0),
                        Operand::Reg(Register::AX),
                    ),
                    Instruction::Ret,
                ],
            }),
        ]);
        assert_eq!(expected, actual);
    }

    #[test]
    fn unsigned_greater_than_uses_above_cond_code() {
        // unsigned int a > unsigned int b should emit SetCC(A), not SetCC(G)
        let ast = tacky::AST::Program(vec![tacky::TopLevel::Function {
            name: "main".into(),
            params: vec![],
            global: true,
            instructions: vec![
                tacky::Instruction::Binary {
                    op: tacky::BinaryOp::GreaterThan,
                    src1: tacky::Val::Var("a.0".into()),
                    src2: tacky::Val::Var("b.1".into()),
                    dst: tacky::Val::Var("tmp.2".into()),
                },
                tacky::Instruction::Ret(tacky::Val::Var("tmp.2".into())),
            ],
        }]);
        let table = sym_table(&[
            ("a.0", CType::UInt),
            ("b.1", CType::UInt),
            ("tmp.2", CType::Int), // comparison result is always int
        ]);
        let asm = Asm::from_tacky(ast, table);
        let Asm::Program(ref tops) = asm;
        let TopLevel::Func(ref func) = tops[0] else {
            panic!()
        };
        assert!(
            func.instructions
                .iter()
                .any(|i| matches!(i, Instruction::SetCC(CondCode::A, _))),
            "Expected SetCC(A) for unsigned GreaterThan, got: {:?}",
            func.instructions
        );
        assert!(
            !func
                .instructions
                .iter()
                .any(|i| matches!(i, Instruction::SetCC(CondCode::G, _))),
            "Got signed CondCode::G for unsigned comparison"
        );
    }

    #[test]
    fn unsigned_division_emits_div_and_zeroes_dx() {
        // unsigned division should zero DX and use div, not cdq/idiv
        let ast = tacky::AST::Program(vec![tacky::TopLevel::Function {
            name: "main".into(),
            params: vec![],
            global: true,
            instructions: vec![
                tacky::Instruction::Binary {
                    op: tacky::BinaryOp::Divide,
                    src1: tacky::Val::Var("a.0".into()),
                    src2: tacky::Val::Var("b.1".into()),
                    dst: tacky::Val::Var("tmp.2".into()),
                },
                tacky::Instruction::Ret(tacky::Val::Var("tmp.2".into())),
            ],
        }]);
        let table = sym_table(&[
            ("a.0", CType::UInt),
            ("b.1", CType::UInt),
            ("tmp.2", CType::UInt),
        ]);
        let asm = Asm::from_tacky(ast, table);
        let Asm::Program(ref tops) = asm;
        let TopLevel::Func(ref func) = tops[0] else {
            panic!()
        };
        assert!(
            func.instructions
                .iter()
                .any(|i| matches!(i, Instruction::Div(..))),
            "Expected Div for unsigned division, got: {:?}",
            func.instructions
        );
        assert!(
            func.instructions.iter().any(|i| matches!(
                i,
                Instruction::Mov(_, Operand::Imm(0), Operand::Reg(Register::DX))
            )),
            "Expected DX zeroed before unsigned div, got: {:?}",
            func.instructions
        );
        assert!(
            !func
                .instructions
                .iter()
                .any(|i| matches!(i, Instruction::Idiv(..))),
            "Got Idiv for unsigned division"
        );
        assert!(
            !func
                .instructions
                .iter()
                .any(|i| matches!(i, Instruction::Cdq(..))),
            "Got Cdq for unsigned division"
        );
    }
}

// some niceties. Maybe move to a from.rs
impl From<tacky::UnaryOp> for UnaryOp {
    fn from(op: tacky::UnaryOp) -> Self {
        match op {
            tacky::UnaryOp::Complement => UnaryOp::Not,
            tacky::UnaryOp::Negate => UnaryOp::Neg,
            tacky::UnaryOp::Not => unreachable!("tacky Not is lowered to Cmp+SetCC, never goes through From"),
        }
    }
}

impl From<tacky::Val> for Operand {
    fn from(v: tacky::Val) -> Self {
        use crate::parser::Const;
        match v {
            tacky::Val::Constant(imm) => {
                let imm = match imm {
                    Const::Int(i) => i as usize,
                    Const::Long(i) => i as usize,
                    Const::ULong(i) => i as usize,
                    Const::UInt(i) => i as usize,
                    Const::Double(i) => i as usize,
                };
                Operand::Imm(imm)
            }
            tacky::Val::Var(ident) => Operand::Pseudo(ident),
        }
    }
}

impl From<&tacky::UnaryOp> for UnaryOp {
    fn from(op: &tacky::UnaryOp) -> Self {
        match op {
            &tacky::UnaryOp::Complement => UnaryOp::Not,
            &tacky::UnaryOp::Negate => UnaryOp::Neg,
            &tacky::UnaryOp::Not => unreachable!("tacky Not is lowered to Cmp+SetCC, never goes through From"),
        }
    }
}

impl From<&tacky::Val> for Operand {
    fn from(v: &tacky::Val) -> Self {
        use crate::parser::Const;
        match v {
            tacky::Val::Constant(imm) => {
                let imm = match imm {
                    Const::Int(i) => *i as usize,
                    Const::Long(i) => *i as usize,
                    Const::ULong(i) => *i as usize,
                    Const::UInt(i) => *i as usize,
                    Const::Double(i) => *i as usize,
                };
                Operand::Imm(imm)
            }
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
