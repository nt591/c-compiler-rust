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
    StaticConstant {
        identifier: String,
        alignment: usize,
        init: StaticInit,
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
    Shr, // shiftright
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
    DivDouble,
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
    P,  // Parity flag, used for NaN handling
    NP, // opposite of P
}

pub const MAGIC_16_BYTE_ALIGNMENT: usize = 16;

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
    // SSE instructions for floating point operations
    // convert with truncation scalar double to signed int
    Cvttsd2si(AssemblyType, Operand, Operand), // src, dst
    // convert scalar double to signed int
    Cvtsi2sd(AssemblyType, Operand, Operand),
    // load effective address
    Lea(Operand, Operand),
}

// implement clone so our mapping of Tacky Var
// to Pseudo can always return an owned value
#[derive(Debug, PartialEq, Clone)]
pub enum Operand {
    Imm(usize),
    Reg(Register),
    Pseudo(String),
    Memory(Register, i32), // represents an address relative to some base register
    Data(String),          // RIP-relative access to .data and .bss
}

#[allow(unused)]
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
    BP, // base pointer, which we can use for loads/stores
    // SSE registers for floating point
    XMM0,
    XMM1,
    XMM2,
    XMM3,
    XMM4,
    XMM5,
    XMM6,
    XMM7,
    // scratch registers for SSE
    XMM14,
    XMM15,
}

impl Register {
    pub fn is_xmm(&self) -> bool {
        use Register::*;
        matches!(
            self,
            XMM0 | XMM1 | XMM2 | XMM3 | XMM4 | XMM5 | XMM6 | XMM7 | XMM14 | XMM15
        )
    }
}

// used for massaging func params for calling conv
type TypedOperand = (AssemblyType, Operand);

#[derive(Debug, Default)]
struct AsmGenerator {
    stack_offset: i32,
    identifiers: HashMap<String, Operand>, // Tacky var ident -> Pseudo(string)
    const_labels: HashMap<String, TopLevel>, // keep track of floating points and labels
    label_counter: usize,                  // used to handle double <> uint conversions and jmps
}

impl AsmGenerator {
    pub fn get_label_for_16byte_neg_float_0(&mut self) -> String {
        let label = format!("const_label_16byte_neg_0_float.{:?}", f64::to_bits(-0.0));
        self.get_or_create_label(
            label.clone(),
            MAGIC_16_BYTE_ALIGNMENT,
            StaticInit::DoubleInit(-0.0),
        );
        label
    }

    pub fn get_label_for_long_upper_bound(&mut self) -> String {
        let label = "const_label_long_max".to_string();
        self.get_or_create_label(
            label.clone(),
            alignment_for_assembly_type(&AssemblyType::Double),
            StaticInit::DoubleInit(9223372036854775808.0),
        );
        label
    }

    fn get_or_create_label_for_double(&mut self, val: &Val) -> String {
        let Val::Constant(Const::Double(i)) = val else {
            panic!()
        };
        let label = format!("const_label.{:?}", f64::to_bits(*i));
        self.get_or_create_label(
            label.clone(),
            alignment_for_assembly_type(&AssemblyType::Double),
            StaticInit::DoubleInit(*i),
        );
        label
    }

    fn get_or_create_label(&mut self, lbl: String, alignment: usize, init: StaticInit) {
        if !self.const_labels.contains_key(&lbl) {
            self.const_labels.insert(
                lbl.clone(),
                TopLevel::StaticConstant {
                    identifier: lbl,
                    alignment,
                    init,
                },
            );
        };
    }

    pub fn val_to_operand(&mut self, v: &tacky::Val) -> Operand {
        match v {
            tacky::Val::Constant(imm) => match imm {
                Const::Int(i) => Operand::Imm(*i as usize),
                Const::Long(i) => Operand::Imm(*i as usize),
                Const::ULong(i) => Operand::Imm(*i as usize),
                Const::UInt(i) => Operand::Imm(*i as usize),
                Const::Double(_) => {
                    let label = self.get_or_create_label_for_double(v);
                    Operand::Data(label)
                }
            },
            tacky::Val::Var(ident) => Operand::Pseudo(ident.clone()),
        }
    }

    pub fn generate_out_of_range_label(&mut self) -> String {
        self.generate_label("out_of_range_____internal")
    }

    pub fn get_label_for_end_of_uint_double_conv_comp(&mut self) -> String {
        self.generate_label("end_____internal")
    }

    fn generate_label(&mut self, prefix: &str) -> String {
        let label = format!("{prefix}.{}", self.label_counter);
        self.label_counter += 1;
        label
    }
}

impl Asm {
    pub fn from_tacky(
        tacky: tacky::AST,
        symbol_table: semantic_analysis::SymbolTable,
    ) -> (Asm, BackendSymbolTable) {
        let mut generator = AsmGenerator::default();
        let mut asm = Self::parse_program(&tacky, &mut generator, &symbol_table);
        let mut symbol_table = symbol_table::backend_symbol_table_from_symbol_table(symbol_table);
        Self::take_static_constants_into_toplevel(&mut generator, &mut asm, &mut symbol_table);
        Self::fixup(&mut asm, &mut generator, &symbol_table);
        (asm, symbol_table)
    }

    fn parse_program(
        tacky: &tacky::AST,
        generator: &mut AsmGenerator,
        symbol_table: &semantic_analysis::SymbolTable,
    ) -> Asm {
        match tacky {
            tacky::AST::Program(ins) => {
                let ins = ins
                    .into_iter()
                    .map(|instruct| match instruct {
                        func @ tacky::TopLevel::Function { .. } => {
                            Self::parse_function(func, generator, symbol_table)
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
        generator: &mut AsmGenerator,
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
        let vals = params
            .iter()
            .map(|p| tacky::Val::Var(p.clone()))
            .collect::<Vec<_>>();
        let (double_args, int_args, stack_args) = classify_values(&vals, generator, symbol_table);

        let mut base_insns = vec![];
        // same as function calls: take GP register args, doubles in SSE registers,
        // and the rest as stack
        let registers = [
            Register::DI,
            Register::SI,
            Register::DX,
            Register::CX,
            Register::R8,
            Register::R9,
        ];
        debug_assert!(int_args.len() <= registers.len());
        for (idx, (assembly_type, op)) in int_args.into_iter().enumerate() {
            let reg = registers[idx];
            base_insns.push(Instruction::Mov(assembly_type, Operand::Reg(reg), op));
        }

        let registers = [
            Register::XMM0,
            Register::XMM1,
            Register::XMM2,
            Register::XMM3,
            Register::XMM4,
            Register::XMM5,
            Register::XMM6,
            Register::XMM7,
        ];
        debug_assert!(double_args.len() <= registers.len());
        for (idx, op) in double_args.into_iter().enumerate() {
            let reg = registers[idx];
            base_insns.push(Instruction::Mov(
                AssemblyType::Double,
                Operand::Reg(reg),
                op,
            ));
        }
        for (idx, (assembly_type, op)) in stack_args.into_iter().enumerate() {
            let idx: i32 = idx
                .try_into()
                .expect("Overflow when mapping function params to stack offsets");
            let offset: i32 = 16 + (8 * idx);
            base_insns.push(Instruction::Mov(
                assembly_type,
                Operand::Memory(Register::BP, offset),
                op,
            ));
        }

        let instructions = Self::parse_instructions(&instructions, generator, symbol_table);
        base_insns.extend(instructions);
        TopLevel::Func(Function {
            name: name.into(),
            instructions: base_insns,
            global: *global,
        })
    }

    fn parse_instructions(
        ins: &[tacky::Instruction],
        generator: &mut AsmGenerator,
        symbol_table: &semantic_analysis::SymbolTable,
    ) -> Vec<Instruction> {
        use Instruction::*;
        use tacky::Instruction as TIns;
        use tacky::UnaryOp as TUnaryOp;
        ins.iter()
            .flat_map(|instruction| match instruction {
                TIns::Ret(val) => {
                    let assembly_type = get_assembly_type_from_val(&val, symbol_table);
                    let return_register = if assembly_type == AssemblyType::Double {
                        Register::XMM0
                    } else {
                        Register::AX
                    };
                    vec![
                        Mov(
                            assembly_type,
                            generator.val_to_operand(val),
                            Operand::Reg(return_register),
                        ),
                        Ret,
                    ]
                }
                TIns::Unary { op, src, dst } => match op {
                    TUnaryOp::Not => {
                        if get_ctype_for_val(src, symbol_table).is_double() {
                            let a1 = get_assembly_type_from_val(src, symbol_table);
                            let a2 = get_assembly_type_from_val(dst, symbol_table);
                            // sete is true if ZF = 1
                            // setnp is true if PF is 0
                            // therefore, if sete is true AND setnp is true,
                            // then we have a successful NOT comparison. NaN != 0, so
                            // we need that setnp check
                            vec![
                                Binary(
                                    BinaryOp::Xor,
                                    a1,
                                    Operand::Reg(Register::XMM0),
                                    Operand::Reg(Register::XMM0),
                                ),
                                Cmp(
                                    a1,
                                    Operand::Reg(Register::XMM0),
                                    generator.val_to_operand(src),
                                ), // compares src to 0
                                Mov(a2, Operand::Imm(0), Operand::Reg(Register::R10)),
                                Mov(a2, Operand::Imm(0), Operand::Reg(Register::R11)),
                                SetCC(CondCode::E, Operand::Reg(Register::R10)),
                                SetCC(CondCode::NP, Operand::Reg(Register::R11)),
                                Binary(
                                    BinaryOp::BitwiseAnd,
                                    a2,
                                    Operand::Reg(Register::R10),
                                    Operand::Reg(Register::R11),
                                ),
                                Mov(
                                    a2,
                                    Operand::Reg(Register::R11),
                                    generator.val_to_operand(dst),
                                ),
                            ]
                        } else {
                            let a1 = get_assembly_type_from_val(src, symbol_table);
                            let a2 = get_assembly_type_from_val(dst, symbol_table);
                            // !x is the same as x==0, so compare
                            // then zero out dest addr and check if cmp returned equal
                            vec![
                                Cmp(a1, Operand::Imm(0), generator.val_to_operand(src)),
                                Mov(a2, Operand::Imm(0), generator.val_to_operand(dst)),
                                SetCC(CondCode::E, generator.val_to_operand(dst)),
                            ]
                        }
                    }
                    TUnaryOp::Negate if get_ctype_for_val(src, symbol_table).is_double() => {
                        // special case: handle negation of doubles by XOR with -0.0
                        // 16 byte aligned for special SSE xorpd instruction
                        let a1 = get_assembly_type_from_val(src, symbol_table);
                        vec![
                            Mov(
                                a1,
                                generator.val_to_operand(src),
                                generator.val_to_operand(dst),
                            ),
                            Binary(
                                BinaryOp::Xor,
                                a1,
                                Operand::Data(generator.get_label_for_16byte_neg_float_0()),
                                generator.val_to_operand(dst),
                            ),
                        ]
                    }
                    o => {
                        let assembly_type = get_assembly_type_from_val(&src, symbol_table);
                        vec![
                            Mov(
                                assembly_type,
                                generator.val_to_operand(src),
                                generator.val_to_operand(dst),
                            ),
                            Unary(o.into(), assembly_type, generator.val_to_operand(dst)),
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
                            Mov(
                                assembly_type,
                                generator.val_to_operand(src1),
                                generator.val_to_operand(dst),
                            ),
                            Binary(
                                o,
                                assembly_type,
                                generator.val_to_operand(src2),
                                generator.val_to_operand(dst),
                            ),
                        ]
                    }
                    tacky::BinaryOp::Equal
                    | tacky::BinaryOp::NotEqual
                    | tacky::BinaryOp::LessThan
                    | tacky::BinaryOp::LessOrEqual
                    | tacky::BinaryOp::GreaterThan
                    | tacky::BinaryOp::GreaterOrEqual => Self::parse_binary_relational_ops(
                        op,
                        src1,
                        src2,
                        dst,
                        symbol_table,
                        generator,
                    ),
                    tacky::BinaryOp::And | tacky::BinaryOp::Or => {
                        unreachable!("BinAnd/BinOr are lowered to jumps")
                    }
                    tacky::BinaryOp::Divide => {
                        let assembly_type = get_assembly_type_from_val(&src1, symbol_table);
                        let ctype = get_ctype_for_val(src1, symbol_table);

                        // special case: check if we're working with doubles, emit
                        // specialized instructions for SSE
                        // else: if ctype is signed, we sign-extend EAX/RAX (cdq instruction) and issue
                        // idiv.
                        // if ctype is unsigned, we zero out EDX/RDX and issue standard div.
                        // Accumulator stores quotient, so copy EAX/RAX into destination
                        if ctype.is_double() {
                            vec![
                                Mov(
                                    assembly_type,
                                    generator.val_to_operand(src1),
                                    generator.val_to_operand(dst),
                                ),
                                Binary(
                                    BinaryOp::DivDouble,
                                    assembly_type,
                                    generator.val_to_operand(src2),
                                    generator.val_to_operand(dst),
                                ),
                            ]
                        } else if ctype.signed() {
                            vec![
                                Mov(
                                    assembly_type,
                                    generator.val_to_operand(src1),
                                    Operand::Reg(Register::AX),
                                ),
                                Cdq(assembly_type),
                                Idiv(assembly_type, generator.val_to_operand(src2)),
                                Mov(
                                    assembly_type,
                                    Operand::Reg(Register::AX),
                                    generator.val_to_operand(dst),
                                ),
                            ]
                        } else {
                            vec![
                                Mov(
                                    assembly_type,
                                    generator.val_to_operand(src1),
                                    Operand::Reg(Register::AX),
                                ),
                                Mov(assembly_type, Operand::Imm(0), Operand::Reg(Register::DX)),
                                Div(assembly_type, generator.val_to_operand(src2)),
                                Mov(
                                    assembly_type,
                                    Operand::Reg(Register::AX),
                                    generator.val_to_operand(dst),
                                ),
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
                                Mov(
                                    assembly_type,
                                    generator.val_to_operand(src1),
                                    Operand::Reg(Register::AX),
                                ),
                                Cdq(assembly_type),
                                Idiv(assembly_type, generator.val_to_operand(src2)),
                                Mov(
                                    assembly_type,
                                    Operand::Reg(Register::DX),
                                    generator.val_to_operand(dst),
                                ),
                            ]
                        } else {
                            vec![
                                Mov(
                                    assembly_type,
                                    generator.val_to_operand(src1),
                                    Operand::Reg(Register::AX),
                                ),
                                Mov(assembly_type, Operand::Imm(0), Operand::Reg(Register::DX)),
                                Div(assembly_type, generator.val_to_operand(src2)),
                                Mov(
                                    assembly_type,
                                    Operand::Reg(Register::DX),
                                    generator.val_to_operand(dst),
                                ),
                            ]
                        }
                    }
                    tacky::BinaryOp::ShiftRight => {
                        // shouldn't expect a shift to change type
                        let assembly_type = get_assembly_type_from_val(src1, symbol_table);
                        vec![
                            Mov(
                                assembly_type,
                                generator.val_to_operand(src1),
                                Operand::Reg(Register::AX),
                            ),
                            Binary(
                                BinaryOp::ShiftRight,
                                assembly_type,
                                generator.val_to_operand(src2),
                                Operand::Reg(Register::AX),
                            ),
                            Mov(
                                assembly_type,
                                Operand::Reg(Register::AX),
                                generator.val_to_operand(dst),
                            ),
                        ]
                    }
                },
                TIns::JumpIfZero { cond, target } => {
                    let assembly_type = get_assembly_type_from_val(cond, symbol_table);

                    if get_ctype_for_val(cond, symbol_table).is_double() {
                        // we need to work with SSE registers, so zero out XMM0
                        // and compare.
                        // to avoid a jump, we need to do something semi-smart here.
                        // We first write to one register a sete - is the cond equal to zero?
                        // we write to another register a setnp - is the cond NOT NaN?
                        // we then AND those two registers: equal to zero and NOT NaN.
                        // If both are true, then the AND returns 1 which sets the zero flag
                        // to 0 (because zero flag is 1 IFF the setting instruction returned zero).
                        // So then we can setne, which checks "is ZF == 0"
                        vec![
                            Binary(
                                BinaryOp::Xor,
                                assembly_type,
                                Operand::Reg(Register::XMM0),
                                Operand::Reg(Register::XMM0),
                            ),
                            Cmp(
                                assembly_type,
                                generator.val_to_operand(cond),
                                Operand::Reg(Register::XMM0),
                            ),
                            Mov(
                                AssemblyType::Longword,
                                Operand::Imm(0),
                                Operand::Reg(Register::R10),
                            ),
                            Mov(
                                AssemblyType::Longword,
                                Operand::Imm(0),
                                Operand::Reg(Register::R11),
                            ),
                            SetCC(CondCode::E, Operand::Reg(Register::R10)),
                            SetCC(CondCode::NP, Operand::Reg(Register::R11)),
                            Binary(
                                BinaryOp::BitwiseAnd,
                                AssemblyType::Longword,
                                Operand::Reg(Register::R10),
                                Operand::Reg(Register::R11),
                            ),
                            JmpCC(CondCode::NE, target.clone()),
                        ]
                    } else {
                        // comp condition to 0, then jump if equal to target
                        vec![
                            Cmp(
                                assembly_type,
                                Operand::Imm(0),
                                generator.val_to_operand(cond),
                            ),
                            JmpCC(CondCode::E, target.clone()),
                        ]
                    }
                }
                TIns::JumpIfNotZero { cond, target } => {
                    let assembly_type = get_assembly_type_from_val(cond, symbol_table);
                    if get_ctype_for_val(cond, symbol_table).is_double() {
                        // we need to work with SSE registers, so zero out XMM0
                        // and compare.
                        // Quiet NaN: unordered is not zero, so jump
                        // with the same codepath as JNZ
                        vec![
                            Binary(
                                BinaryOp::Xor,
                                assembly_type,
                                Operand::Reg(Register::XMM0),
                                Operand::Reg(Register::XMM0),
                            ),
                            Cmp(
                                assembly_type,
                                generator.val_to_operand(cond),
                                Operand::Reg(Register::XMM0),
                            ),
                            JmpCC(CondCode::P, target.clone()),
                            JmpCC(CondCode::NE, target.clone()),
                        ]
                    } else {
                        vec![
                            Cmp(
                                assembly_type,
                                Operand::Imm(0),
                                generator.val_to_operand(cond),
                            ),
                            JmpCC(CondCode::NE, target.clone()),
                        ]
                    }
                }
                TIns::Label(ident) => vec![Label(ident.clone())],
                TIns::Copy { src, dst } => {
                    let assembly_type = get_assembly_type_from_val(src, symbol_table);
                    vec![Mov(
                        assembly_type,
                        generator.val_to_operand(src),
                        generator.val_to_operand(dst),
                    )]
                }
                TIns::Jump(ident) => vec![Jmp(ident.clone())],
                TIns::FunCall { name, args, dst } => {
                    Self::parse_function_call(name, args, dst, symbol_table, generator)
                }
                TIns::SignExtend { src, dst } => {
                    vec![Movsx(
                        generator.val_to_operand(src),
                        generator.val_to_operand(dst),
                    )]
                }
                TIns::Truncate { src, dst } => {
                    // truncate by just moving low 4 bytes into destination
                    vec![Mov(
                        AssemblyType::Longword,
                        generator.val_to_operand(src),
                        generator.val_to_operand(dst),
                    )]
                }
                TIns::ZeroExtend { src, dst } => {
                    vec![MovZeroExtend(
                        generator.val_to_operand(src),
                        generator.val_to_operand(dst),
                    )]
                }
                TIns::DoubleToInt { src, dst } => {
                    // SSE builtin: converts and truncates, with potential indefinite integer
                    vec![Cvttsd2si(
                        get_assembly_type_from_val(dst, symbol_table),
                        generator.val_to_operand(src),
                        generator.val_to_operand(dst),
                    )]
                }
                TIns::IntToDouble { src, dst } => {
                    // SSE builtin: converts signed int to double
                    vec![Cvtsi2sd(
                        get_assembly_type_from_val(src, symbol_table),
                        generator.val_to_operand(src),
                        generator.val_to_operand(dst),
                    )]
                }
                TIns::DoubleToUInt { src, dst } => {
                    // we need to handle whether dst is a longword or quadword
                    // Rules are to basically see if it fits inside max long,
                    // and if it's too big we'll subtract the upper bound,
                    // convert and truncate, then add upper bound back in.
                    let jump_lbl = generator.generate_out_of_range_label();
                    let end_lbl = generator.get_label_for_end_of_uint_double_conv_comp();
                    let upper_bound_lbl = generator.get_label_for_long_upper_bound();
                    // Compare to longmax, JAE to out of range.
                    // otherwise, convert-and-truncate and jump to end
                    let dst_ty = get_ctype_for_val(dst, symbol_table);
                    let dst_asm_ty = get_assembly_type_from_val(dst, symbol_table);
                    if dst_ty == CType::ULong {
                        vec![
                            Cmp(
                                AssemblyType::Double,
                                Operand::Data(upper_bound_lbl.clone()),
                                generator.val_to_operand(src),
                            ),
                            JmpCC(CondCode::AE, jump_lbl.clone()),
                            Cvttsd2si(
                                AssemblyType::Quadword,
                                generator.val_to_operand(src),
                                generator.val_to_operand(dst),
                            ),
                            Jmp(end_lbl.clone()),
                            Label(jump_lbl),
                            // move src into a scratch register, subtract big number,
                            // convert down, add back big number
                            Mov(
                                AssemblyType::Double,
                                generator.val_to_operand(src),
                                Operand::Reg(Register::XMM1),
                            ),
                            Binary(
                                BinaryOp::Sub,
                                AssemblyType::Double,
                                Operand::Data(upper_bound_lbl.clone()),
                                Operand::Reg(Register::XMM1),
                            ),
                            Cvttsd2si(
                                AssemblyType::Quadword,
                                Operand::Reg(Register::XMM1),
                                Operand::Reg(Register::AX),
                            ),
                            Binary(
                                BinaryOp::Add,
                                dst_asm_ty,
                                Operand::Imm(1usize << 63), // gets rewritten in the fixup
                                Operand::Reg(Register::AX),
                            ),
                            Mov(
                                dst_asm_ty,
                                Operand::Reg(Register::AX),
                                generator.val_to_operand(dst),
                            ),
                            Label(end_lbl),
                        ]
                    } else {
                        // no direct instruction, so we take the src,
                        // convert it to a signed long via SSE instruction
                        // then we truncate by moving (only copying lower 4 bytes)
                        vec![
                            Cvttsd2si(
                                AssemblyType::Quadword,
                                generator.val_to_operand(src),
                                Operand::Reg(Register::AX), // copies 8 bytes into RAX
                            ),
                            Mov(
                                get_assembly_type_from_val(dst, symbol_table),
                                Operand::Reg(Register::AX), // copies 4 bytes from EAX into dst
                                generator.val_to_operand(dst),
                            ),
                        ]
                    }
                }
                TIns::UIntToDouble { src, dst } => {
                    // We special-case unsigned ints to doubles. They'll fit so we zero extend and
                    // use the standard SSE instruction for conversions.
                    let src_ty = get_ctype_for_val(src, symbol_table);
                    let src_asm_ty = get_assembly_type_from_val(src, symbol_table);
                    if src_ty == CType::UInt {
                        vec![
                            MovZeroExtend(
                                generator.val_to_operand(src),
                                Operand::Reg(Register::AX),
                            ),
                            Cvtsi2sd(
                                AssemblyType::Quadword,
                                Operand::Reg(Register::AX),
                                generator.val_to_operand(dst),
                            ),
                        ]
                    } else {
                        // harder here: like above, we first see if it fits inside a signed long.
                        // If yes just use the SSE instruction. Otherwise, we halve the value
                        // (shift-right), convert it, then double (add it to itself)
                        let jump_lbl = generator.generate_out_of_range_label();
                        let end_lbl = generator.get_label_for_end_of_uint_double_conv_comp();
                        vec![
                            // DST - 0 does nothing but it will set the signed flag
                            Cmp(src_asm_ty, Operand::Imm(0), generator.val_to_operand(src)),
                            // if the signed flag is set, then our unsigned long will
                            // reinterpret if we treat it as a signed long, so we can't do a direct
                            // conversion. Handle that case in the jump. If it DOES fit, convert
                            // and go home.
                            JmpCC(CondCode::L, jump_lbl.clone()),
                            Cvtsi2sd(
                                src_asm_ty,
                                generator.val_to_operand(src),
                                generator.val_to_operand(dst),
                            ),
                            Jmp(end_lbl.clone()),
                            Label(jump_lbl),
                            // we halve the value, round to nearest odd, then
                            // convert, and add back to itself
                            Mov(
                                src_asm_ty,
                                generator.val_to_operand(src),
                                Operand::Reg(Register::AX),
                            ),
                            // move into DX so we can shr
                            Mov(
                                src_asm_ty,
                                Operand::Reg(Register::AX),
                                Operand::Reg(Register::DX),
                            ),
                            Unary(UnaryOp::Shr, src_asm_ty, Operand::Reg(Register::DX)),
                            Binary(
                                BinaryOp::BitwiseAnd,
                                src_asm_ty,
                                Operand::Imm(1),
                                Operand::Reg(Register::AX),
                            ), // is RAX odd?
                            Binary(
                                BinaryOp::BitwiseOr,
                                src_asm_ty,
                                Operand::Reg(Register::AX),
                                Operand::Reg(Register::DX),
                            ), // round RDX to nearest odd by maybe toggling low bit IFF
                            // either original or halved (rounded down) are odd
                            Cvtsi2sd(
                                src_asm_ty,
                                Operand::Reg(Register::DX),
                                generator.val_to_operand(dst),
                            ),
                            Binary(
                                BinaryOp::Add,
                                AssemblyType::Double,
                                generator.val_to_operand(dst),
                                generator.val_to_operand(dst),
                            ), // double to undo the shr
                            Label(end_lbl),
                        ]
                    }
                }
                TIns::Load { src_ptr, dst } => {
                    // mov address of src into register, then move from 0(%rax) aka take what's at
                    // that address and move into dst
                    vec![
                        Instruction::Mov(
                            AssemblyType::Quadword,
                            generator.val_to_operand(src_ptr),
                            Operand::Reg(Register::AX),
                        ),
                        Instruction::Mov(
                            get_assembly_type_from_val(dst, symbol_table),
                            Operand::Memory(Register::AX, 0),
                            generator.val_to_operand(dst),
                        ),
                    ]
                }
                TIns::Store { src, dst_ptr } => {
                    // mov the address in dst into tmp register, mov src into the address found at
                    // AX
                    vec![
                        Instruction::Mov(
                            AssemblyType::Quadword,
                            generator.val_to_operand(dst_ptr),
                            Operand::Reg(Register::AX),
                        ),
                        Instruction::Mov(
                            get_assembly_type_from_val(src, symbol_table),
                            generator.val_to_operand(src),
                            Operand::Memory(Register::AX, 0),
                        ),
                    ]
                }
                TIns::GetAddress { src, dst } => vec![Instruction::Lea(
                    generator.val_to_operand(src),
                    generator.val_to_operand(dst),
                )],
            })
            .collect::<Vec<_>>()
    }

    fn parse_function_call(
        name: &str,
        args: &[tacky::Val],
        dst: &tacky::Val,
        symbol_table: &SymbolTable,
        generator: &mut AsmGenerator,
    ) -> Vec<Instruction> {
        let (double_args, int_args, stack_args) = classify_values(&args, generator, symbol_table);

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

        // take our known int values in calling conv registers
        let registers = [
            Register::DI,
            Register::SI,
            Register::DX,
            Register::CX,
            Register::R8,
            Register::R9,
        ];
        debug_assert!(int_args.len() <= registers.len());
        for (idx, (assembly_type, op)) in int_args.into_iter().enumerate() {
            let reg = registers[idx];
            instructions.push(Instruction::Mov(assembly_type, op, Operand::Reg(reg)));
        }

        // same for doubles
        let registers = [
            Register::XMM0,
            Register::XMM1,
            Register::XMM2,
            Register::XMM3,
            Register::XMM4,
            Register::XMM5,
            Register::XMM6,
            Register::XMM7,
        ];
        debug_assert!(double_args.len() <= registers.len());
        for (idx, op) in double_args.into_iter().enumerate() {
            let reg = registers[idx];
            instructions.push(Instruction::Mov(
                AssemblyType::Double,
                op,
                Operand::Reg(reg),
            ));
        }

        let stack_args_len = stack_args.len(); // used later to fixup stack
        // pass stack arguments in reverse order
        for (assembly_type, op) in stack_args.into_iter().rev() {
            if matches!(assembly_type, AssemblyType::Double | AssemblyType::Quadword) {
                // we can push 8 byte values onto the stack without issue.
                instructions.push(Instruction::Push(op));
            } else if matches!(op, Operand::Reg(_) | Operand::Imm(_)) {
                // also allowed to push since its 8 bytesj
                instructions.push(Instruction::Push(op));
            } else {
                instructions.push(Instruction::Mov(
                    assembly_type,
                    op,
                    Operand::Reg(Register::AX),
                ));
                instructions.push(Instruction::Push(Operand::Reg(Register::AX)));
            }
        }

        // actual function call
        instructions.push(Instruction::Call(name.into()));
        // adjust stack pointer back to where it was before setting up stack.
        // Remove padding + passed arguments
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
        let return_type = get_assembly_type_from_val(dst, symbol_table);
        if return_type == AssemblyType::Double {
            instructions.push(Instruction::Mov(
                return_type,
                Operand::Reg(Register::XMM0),
                generator.val_to_operand(dst),
            ));
        } else {
            instructions.push(Instruction::Mov(
                return_type,
                Operand::Reg(Register::AX),
                generator.val_to_operand(dst),
            ));
        }
        instructions
    }

    fn parse_binary_relational_ops(
        op: &tacky::BinaryOp,
        src1: &tacky::Val,
        src2: &tacky::Val,
        dst: &tacky::Val,
        symbol_table: &SymbolTable,
        generator: &mut AsmGenerator,
    ) -> Vec<Instruction> {
        let ctype = get_ctype_for_val(src1, symbol_table);
        if ctype.is_double() {
            Self::parse_binary_relational_ops_for_doubles(
                op,
                ctype,
                src1,
                src2,
                dst,
                symbol_table,
                generator,
            )
        } else {
            Self::parse_binary_relational_ops_for_nondoubles(
                op,
                ctype,
                src1,
                src2,
                dst,
                symbol_table,
                generator,
            )
        }
    }

    fn parse_binary_relational_ops_for_doubles(
        op: &tacky::BinaryOp,
        ctype: CType,
        src1: &tacky::Val,
        src2: &tacky::Val,
        dst: &tacky::Val,
        symbol_table: &SymbolTable,
        generator: &mut AsmGenerator,
    ) -> Vec<Instruction> {
        debug_assert!(ctype.is_double());
        use Instruction::*;
        use tacky::BinaryOp as TBO;
        // treats doubles as unsigned ints for cond codes
        let cond_code = match op {
            TBO::Equal => CondCode::E,
            TBO::NotEqual => CondCode::NE,
            TBO::GreaterThan => CondCode::A,
            TBO::GreaterOrEqual => CondCode::AE,
            TBO::LessThan => CondCode::B,
            TBO::LessOrEqual => CondCode::BE,
            _ => {
                panic!("Unexpected tacky BinaryOp {op:?} when constructing relational instruction")
            }
        };
        let a1 = get_assembly_type_from_val(src1, symbol_table);
        let a2 = get_assembly_type_from_val(dst, symbol_table);
        // we have curious challenges right now. The parity flag is true
        // when handling NaN so we need to inject a jump
        // because NaN is unordered and cannot be <, > or == to a value.
        // The tricky bit is NE - NaN is never equal to anything,
        // including NaN. We'll need to special case just that condcode.
        if cond_code == CondCode::NE {
            vec![
                Cmp(
                    a1,
                    generator.val_to_operand(src2),
                    generator.val_to_operand(src1),
                ),
                // set one scratch register for NE (zero flag = 0)
                // set one scratch register for parity flag
                // if not equal OR parity flag, its truthy so set
                // dst operand to the output of the or
                Mov(a2, Operand::Imm(0), Operand::Reg(Register::R10)),
                Mov(a2, Operand::Imm(0), Operand::Reg(Register::R11)),
                SetCC(cond_code, Operand::Reg(Register::R10)),
                SetCC(CondCode::P, Operand::Reg(Register::R11)),
                Binary(
                    BinaryOp::BitwiseOr,
                    a2,
                    Operand::Reg(Register::R10),
                    Operand::Reg(Register::R11),
                ),
                Mov(
                    a2,
                    Operand::Reg(Register::R11),
                    generator.val_to_operand(dst),
                ),
            ]
        } else {
            // opposite of the NE cond code
            // we'll set one scratch register to the codecode
            // we'll set another not parity flag set
            // Then we AND them. That is,
            // the comparison is true IF the cond code holds
            // AND neither value is a NaN.
            vec![
                Cmp(
                    a1,
                    generator.val_to_operand(src2),
                    generator.val_to_operand(src1),
                ),
                Mov(a2, Operand::Imm(0), Operand::Reg(Register::R10)),
                Mov(a2, Operand::Imm(0), Operand::Reg(Register::R11)),
                SetCC(cond_code, Operand::Reg(Register::R10)),
                SetCC(CondCode::NP, Operand::Reg(Register::R11)), // note: NP
                Binary(
                    BinaryOp::BitwiseAnd,
                    a2,
                    Operand::Reg(Register::R10),
                    Operand::Reg(Register::R11),
                ),
                Mov(
                    a2,
                    Operand::Reg(Register::R11),
                    generator.val_to_operand(dst),
                ),
            ]
        }
    }

    fn parse_binary_relational_ops_for_nondoubles(
        op: &tacky::BinaryOp,
        ctype: CType,
        src1: &tacky::Val,
        src2: &tacky::Val,
        dst: &tacky::Val,
        symbol_table: &SymbolTable,
        generator: &mut AsmGenerator,
    ) -> Vec<Instruction> {
        debug_assert!(!ctype.is_double());
        use Instruction::*;
        use tacky::BinaryOp as TBO;
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
            Cmp(
                a1,
                generator.val_to_operand(src2),
                generator.val_to_operand(src1),
            ),
            Mov(a2, Operand::Imm(0), generator.val_to_operand(dst)),
            SetCC(cond_code, generator.val_to_operand(dst)),
        ]
    }

    fn fixup(asm: &mut Asm, generator: &mut AsmGenerator, symbol_table: &BackendSymbolTable) {
        match asm {
            Asm::Program(funcs) => {
                for func in funcs {
                    if let TopLevel::Func(func) = func {
                        Self::fixup_function(func, generator, &symbol_table);
                    }
                }
            }
        }
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
            Instruction::Cvtsi2sd(_, src, dst) => {
                *src = Self::replace_pseudo_in_op(src, generator, symbol_table);
                *dst = Self::replace_pseudo_in_op(dst, generator, symbol_table);
            }
            Instruction::Cvttsd2si(_, src, dst) => {
                *src = Self::replace_pseudo_in_op(src, generator, symbol_table);
                *dst = Self::replace_pseudo_in_op(dst, generator, symbol_table);
            }
            Instruction::Lea(src, dst) => {
                *src = Self::replace_pseudo_in_op(src, generator, symbol_table);
                *dst = Self::replace_pseudo_in_op(dst, generator, symbol_table);
            }
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
                let symbol_table::BackendSymTableEntry::ObjEntry { ty, is_static, .. } = entry
                else {
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
                        Operand::Memory(Register::BP, next_offset)
                    })
                    .clone()
            }
            o => o.clone(), //no transformation otherwise
        }
    }

    fn next_aligned_offset(offset: i32, ty: &AssemblyType) -> i32 {
        // ensures that everything is aligned.
        // Quadwords and doubles need to be
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
                Instruction::Mov(at, src @ Operand::Memory(_, _), dst @ Operand::Memory(_, _))
                | Instruction::Mov(at, src @ Operand::Data(_), dst @ Operand::Data(_))
                | Instruction::Mov(at, src @ Operand::Memory(_, _), dst @ Operand::Data(_))
                | Instruction::Mov(at, src @ Operand::Data(_), dst @ Operand::Memory(_, _)) => {
                    // movl can't move from two memory addrs, so
                    // use a temporary variable along the way in %r10d or xmm14
                    if at == AssemblyType::Double {
                        v.push(Instruction::Mov(at, src, Operand::Reg(Register::XMM14)));
                        v.push(Instruction::Mov(at, Operand::Reg(Register::XMM14), dst));
                    } else {
                        v.push(Instruction::Mov(at, src, Operand::Reg(Register::R10)));
                        v.push(Instruction::Mov(at, Operand::Reg(Register::R10), dst));
                    }
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
                    dst @ Operand::Memory(_, _),
                ) if i > i32::MAX as usize => {
                    // movq can't move big values into memory but needs an intermediate register
                    v.extend(vec![
                        Instruction::Mov(AssemblyType::Quadword, src, Operand::Reg(Register::R10)),
                        Instruction::Mov(AssemblyType::Quadword, Operand::Reg(Register::R10), dst),
                    ]);
                }
                Instruction::Movsx(_, Operand::Imm(_))
                | Instruction::Movsx(_, Operand::Pseudo(_)) => unreachable!(),
                Instruction::Movsx(src @ Operand::Imm(_), dst @ Operand::Memory(_, _))
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
                Instruction::Movsx(src, dst @ Operand::Memory(_, _))
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
                Instruction::MovZeroExtend(
                    src,
                    dst @ (Operand::Data(_) | Operand::Memory(_, _)),
                ) => {
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
                    dst @ Operand::Memory(_, _),
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
                Instruction::Binary(
                    binop @ (BinaryOp::Add
                    | BinaryOp::Sub
                    | BinaryOp::Mult
                    | BinaryOp::DivDouble
                    | BinaryOp::Xor),
                    ty @ AssemblyType::Double,
                    src,
                    dst @ (Operand::Memory(_, _) | Operand::Data(_)),
                ) => {
                    // addsd, subsd, mulsd, divsd, xorsd require a register in dest. We'll shuffle into,
                    // XMM15
                    v.extend(vec![
                        Instruction::Mov(ty, dst.clone(), Operand::Reg(Register::XMM15)),
                        Instruction::Binary(binop, ty, src, Operand::Reg(Register::XMM15)),
                        Instruction::Mov(ty, Operand::Reg(Register::XMM15), dst),
                    ])
                }
                Instruction::Binary(BinaryOp::Mult, at, src, dst @ Operand::Memory(_, _))
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
                    binop @ (BinaryOp::Add
                    | BinaryOp::Sub
                    | BinaryOp::BitwiseOr
                    | BinaryOp::BitwiseAnd
                    | BinaryOp::Xor),
                    at @ AssemblyType::Quadword,
                    src @ Operand::Imm(i),
                    dst,
                ) if i > i32::MAX as usize => {
                    // addq, subq require that immediates fit in an int. If not, we use an
                    // intermediary register.
                    let reg = Operand::Reg(Register::R10);
                    v.extend(vec![
                        Instruction::Mov(at, src, reg.clone()),
                        Instruction::Binary(binop, at, reg, dst),
                    ])
                }
                Instruction::Binary(
                    binop,
                    at,
                    src @ Operand::Memory(_, _),
                    dst @ Operand::Memory(_, _),
                )
                | Instruction::Binary(binop, at, src @ Operand::Data(_), dst @ Operand::Data(_))
                | Instruction::Binary(
                    binop,
                    at,
                    src @ Operand::Memory(_, _),
                    dst @ Operand::Data(_),
                )
                | Instruction::Binary(
                    binop,
                    at,
                    src @ Operand::Data(_),
                    dst @ Operand::Memory(_, _),
                ) if matches!(
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
                Instruction::Binary(
                    binop @ (BinaryOp::ShiftRight | BinaryOp::ShiftLeft),
                    at,
                    src @ (Operand::Memory(_, _) | Operand::Data(_)),
                    dst,
                ) => {
                    // shift left/right cannot use a memory address as a source.
                    // We move the data to a scratch register. We write to ECX,
                    // then read from the lower 8 bits.
                    v.push(Instruction::Mov(at, src, Operand::Reg(Register::CX)));
                    v.push(Instruction::Binary(
                        binop,
                        at,
                        Operand::Reg(Register::CX),
                        dst,
                    ))
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
                Instruction::Cmp(
                    at @ AssemblyType::Double,
                    src,
                    dst @ (Operand::Memory(_, _) | Operand::Data(_)),
                ) => {
                    // comisd requires dst is in a register, so we shuffle.
                    v.extend(vec![
                        Instruction::Mov(at, dst, Operand::Reg(Register::XMM15)),
                        Instruction::Cmp(at, src, Operand::Reg(Register::XMM15)),
                    ])
                }
                Instruction::Cmp(at, src @ Operand::Memory(_, _), dst @ Operand::Memory(_, _))
                | Instruction::Cmp(at, src @ Operand::Data(_), dst @ Operand::Data(_))
                | Instruction::Cmp(at, src @ Operand::Memory(_, _), dst @ Operand::Data(_))
                | Instruction::Cmp(at, src @ Operand::Data(_), dst @ Operand::Memory(_, _)) => {
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
                Instruction::Cvttsd2si(
                    ty,
                    src,
                    dst @ (Operand::Memory(_, _) | Operand::Data(_)),
                ) => {
                    // destination must be a register
                    v.extend(vec![
                        Instruction::Cvttsd2si(ty, src, Operand::Reg(Register::R11)),
                        Instruction::Mov(ty, Operand::Reg(Register::R11), dst),
                    ])
                }
                Instruction::Cvtsi2sd(ty, src, dst)
                    if matches!(src, Operand::Imm(_)) || !matches!(dst, Operand::Reg(_)) =>
                {
                    // Cvtsi2sd has two constraints: src cannot be a constant,
                    // and dst must be a register. So we gotta do some silly tricks here.
                    // First, if src is a constant we'll juggle into a register.
                    // Then if dest is not a register, we emit Cvtsi2sd into a tmp
                    // register and mov back into dst.
                    // If none of these conditions were true our vec was empty so we'll just fall
                    // into the default of pushing the instruction.
                    let has_constant_src = matches!(src, Operand::Imm(_));
                    let has_register_dst = matches!(dst, Operand::Reg(_));
                    let convert_src = if has_constant_src {
                        Operand::Reg(Register::R10)
                    } else {
                        src.clone()
                    };
                    let convert_dst = if has_register_dst {
                        dst.clone()
                    } else {
                        Operand::Reg(Register::XMM15)
                    };
                    let mut instructions = vec![];
                    // first, if there is a constant src we'll move into the src for conversion
                    if has_constant_src {
                        instructions.push(Instruction::Mov(ty, src, convert_src.clone()))
                    }
                    instructions.push(Instruction::Cvtsi2sd(ty, convert_src, convert_dst.clone()));
                    if !has_register_dst {
                        // mov back from our tmp register into proper dest
                        instructions.push(Instruction::Mov(AssemblyType::Double, convert_dst, dst));
                    }
                    v.extend(instructions)
                }
                Instruction::Lea(src, dst) if !matches!(dst, Operand::Reg(_)) => {
                    // lea dst must be a register, so we'll move to r10
                    v.extend(vec![
                        Instruction::Lea(src, Operand::Reg(Register::R10)),
                        Instruction::Mov(AssemblyType::Quadword, Operand::Reg(Register::R10), dst),
                    ])
                }
                Instruction::Push(r @ Operand::Reg(reg)) if reg.is_xmm() => {
                    // pushing an xmm register is disallowed, so instead subtract space on stack
                    // and mov directly
                    v.extend(vec![
                        Instruction::Binary(
                            BinaryOp::Sub,
                            AssemblyType::Quadword,
                            Operand::Imm(8),
                            Operand::Reg(Register::SP),
                        ),
                        Instruction::Mov(AssemblyType::Double, r, Operand::Memory(Register::SP, 0)),
                    ])
                }
                i => v.push(i),
            }
        }
        func.instructions = v;
    }

    fn take_static_constants_into_toplevel(
        generator: &mut AsmGenerator,
        asm: &mut Asm,
        symtable: &mut BackendSymbolTable,
    ) {
        use crate::symbol_table::BackendSymTableEntry;
        // generator no longer needs the top levels so to avoid refs, we'll just
        // take them directoy and keep owned values in asm
        let mut const_labels: Vec<_> = std::mem::take(&mut generator.const_labels)
            .into_iter()
            .collect();
        // sorting here because hashmap insertion is not guaranteed, and I need
        // reproduceability for tests. Could also use a BTreeMap if I cared.
        const_labels.sort_by(|(a, _), (b, _)| a.cmp(b));
        let Asm::Program(toplevels) = asm;
        toplevels.reserve(const_labels.len());
        symtable.reserve(const_labels.len());
        for (label, constant) in const_labels.into_iter() {
            symtable.insert(
                label,
                BackendSymTableEntry::ObjEntry {
                    ty: AssemblyType::Double,
                    is_static: true, // doubles live as constants in .rodata
                    is_constant: true,
                },
            );
            toplevels.push(constant);
        }
    }
}

fn get_assembly_type_from_val(val: &tacky::Val, symbol_table: &SymbolTable) -> AssemblyType {
    match val {
        Val::Constant(Const::Int(_) | Const::UInt(_)) => AssemblyType::Longword,
        Val::Constant(Const::Long(_) | Const::ULong(_)) => AssemblyType::Quadword,
        Val::Constant(Const::Double(_)) => AssemblyType::Double,
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
        CType::Double => AssemblyType::Double,
        CType::FunType { .. } => unreachable!(),
        CType::Pointer(_) => AssemblyType::Quadword,
        CType::Array(_, _) => todo!(),
    }
}

fn alignment_for_assembly_type(at: &AssemblyType) -> usize {
    match at {
        AssemblyType::Longword => 4,
        AssemblyType::Quadword => 8,
        AssemblyType::Double => 8,
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

// returns doubles, ints, and stack params for function bodies / calls
fn classify_values(
    vals: &[tacky::Val],
    generator: &mut AsmGenerator,
    symbol_table: &SymbolTable,
) -> (Vec<Operand>, Vec<TypedOperand>, Vec<TypedOperand>) {
    let mut int_args = vec![];
    let mut double_args = vec![];
    let mut stack_args = vec![];
    for v in vals {
        let op = generator.val_to_operand(&v);
        let type_ = get_assembly_type_from_val(&v, symbol_table);
        if type_ == AssemblyType::Double {
            // we can take up to 8 via registers (XMM0-7), else stack
            if double_args.len() < 8 {
                double_args.push(op)
            } else {
                let typed_op = (type_, op);
                stack_args.push(typed_op)
            }
        } else {
            let typed_op = (type_, op);
            // we can take up to 6 int args in the GP registers
            if int_args.len() < 6 {
                int_args.push(typed_op)
            } else {
                stack_args.push(typed_op)
            }
        }
    }

    (double_args, int_args, stack_args)
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

    // runs source text through lex -> parse -> semantic analysis -> tacky -> asm,
    // mirroring the `functions()` test's pipeline so callers can just assert on the result.
    fn compile(src: &str) -> Asm {
        let lexer = crate::lexer::Lexer::lex(src).unwrap();
        let tokens = lexer.as_syntactic_tokens();
        let parse = crate::parser::Parser::new(&tokens);
        let mut ast = parse.into_ast().unwrap();
        let (mut symbol_table, ast) = crate::semantic_analysis::resolve(&mut ast).unwrap();
        let tacky = crate::tacky::Tacky::new(ast);
        let tacky = tacky.into_ast(&mut symbol_table);
        let Ok(tacky_ast) = tacky else {
            panic!("tacky generation failed for: {src}");
        };
        let (asm, _) = Asm::from_tacky(tacky_ast, symbol_table);
        asm
    }

    fn func_instructions<'a>(asm: &'a Asm, name: &str) -> &'a [Instruction] {
        let Asm::Program(tops) = asm;
        for top in tops {
            if let TopLevel::Func(f) = top {
                if f.name == name {
                    return &f.instructions;
                }
            }
        }
        panic!("no function named {name} found in {tops:?}");
    }

    #[test]
    fn double_and_int_parameters_fixture_codegens() {
        let src = include_str!("../fixtures/double_and_int_parameters.c");
        insta::assert_debug_snapshot!(compile(src));
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

        let (assembly, _) = Asm::from_tacky(ast, int_sym_table(&[]));
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
                    Operand::Memory(Register::BP, -4),
                ),
                Instruction::Unary(
                    UnaryOp::Neg,
                    AssemblyType::Longword,
                    Operand::Memory(Register::BP, -4),
                ),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Memory(Register::BP, -4),
                    Operand::Reg(Register::AX),
                ),
                Instruction::Ret,
            ],
        })]);

        let (assembly, _) = Asm::from_tacky(ast, int_sym_table(&["tmp.0"]));
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
                    Operand::Memory(Register::BP, -4),
                ),
                Instruction::Unary(
                    UnaryOp::Neg,
                    AssemblyType::Longword,
                    Operand::Memory(Register::BP, -4),
                ),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Memory(Register::BP, -4),
                    Operand::Reg(Register::R10),
                ),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Reg(Register::R10),
                    Operand::Memory(Register::BP, -8),
                ),
                Instruction::Unary(
                    UnaryOp::Not,
                    AssemblyType::Longword,
                    Operand::Memory(Register::BP, -8),
                ),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Memory(Register::BP, -8),
                    Operand::Reg(Register::R10),
                ),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Reg(Register::R10),
                    Operand::Memory(Register::BP, -12),
                ),
                Instruction::Unary(
                    UnaryOp::Neg,
                    AssemblyType::Longword,
                    Operand::Memory(Register::BP, -12),
                ),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Memory(Register::BP, -12),
                    Operand::Reg(Register::AX),
                ),
                Instruction::Ret,
            ],
        })]);

        let (assembly, _) = Asm::from_tacky(ast, int_sym_table(&["tmp.0", "tmp.1", "tmp.2"]));
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
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Imm(1),
                    Operand::Memory(Register::BP, -4),
                ),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Memory(Register::BP, -4),
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
                    Operand::Memory(Register::BP, -4),
                ),
                // tmp1 = 4 + 5
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Imm(4),
                    Operand::Memory(Register::BP, -8),
                ),
                Instruction::Binary(
                    BinaryOp::Add,
                    AssemblyType::Longword,
                    Operand::Imm(5),
                    Operand::Memory(Register::BP, -8),
                ),
                // tmp2 = 3 % tmp1
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Imm(3),
                    Operand::Reg(Register::AX),
                ),
                Instruction::Cdq(AssemblyType::Longword),
                Instruction::Idiv(AssemblyType::Longword, Operand::Memory(Register::BP, -8)),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Reg(Register::DX),
                    Operand::Memory(Register::BP, -12),
                ),
                // tmp3 = tmp0 / tmp2
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Memory(Register::BP, -4),
                    Operand::Reg(Register::AX),
                ),
                Instruction::Cdq(AssemblyType::Longword),
                Instruction::Idiv(AssemblyType::Longword, Operand::Memory(Register::BP, -12)),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Reg(Register::AX),
                    Operand::Memory(Register::BP, -16),
                ),
                // return
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Memory(Register::BP, -16),
                    Operand::Reg(Register::AX),
                ),
                Instruction::Ret,
            ],
        })]);

        let (assembly, _) =
            Asm::from_tacky(ast, int_sym_table(&["tmp.0", "tmp.1", "tmp.2", "tmp.3"]));
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
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Imm(5),
                    Operand::Memory(Register::BP, -4),
                ),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Memory(Register::BP, -4),
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
                    Operand::Memory(Register::BP, -4),
                ),
                // tmp1 = tmp0 / 2 = 10
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Memory(Register::BP, -4),
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
                    Operand::Memory(Register::BP, -8),
                ),
                // tmp2 = 2 + 1  = 3
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Imm(2),
                    Operand::Memory(Register::BP, -12),
                ),
                Instruction::Binary(
                    BinaryOp::Add,
                    AssemblyType::Longword,
                    Operand::Imm(1),
                    Operand::Memory(Register::BP, -12),
                ),
                // tmp3 = 3 % tmp2 = 0
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Imm(3),
                    Operand::Reg(Register::AX),
                ),
                Instruction::Cdq(AssemblyType::Longword),
                Instruction::Idiv(AssemblyType::Longword, Operand::Memory(Register::BP, -12)),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Reg(Register::DX),
                    Operand::Memory(Register::BP, -16),
                ),
                // tmp3 = tmp1 - tmp3 = 10
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Memory(Register::BP, -8),
                    Operand::Reg(Register::R10),
                ),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Reg(Register::R10),
                    Operand::Memory(Register::BP, -20),
                ),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Memory(Register::BP, -16),
                    Operand::Reg(Register::R10),
                ),
                Instruction::Binary(
                    BinaryOp::Sub,
                    AssemblyType::Longword,
                    Operand::Reg(Register::R10),
                    Operand::Memory(Register::BP, -20),
                ),
                // return
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Memory(Register::BP, -20),
                    Operand::Reg(Register::AX),
                ),
                Instruction::Ret,
            ],
        })]);

        let (assembly, _) = Asm::from_tacky(
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
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Imm(5),
                    Operand::Memory(Register::BP, -4),
                ),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Memory(Register::BP, -4),
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
                    Operand::Memory(Register::BP, -4),
                ),
                // tmp1 = 4 - 5
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Imm(4),
                    Operand::Memory(Register::BP, -8),
                ),
                Instruction::Binary(
                    BinaryOp::Sub,
                    AssemblyType::Longword,
                    Operand::Imm(5),
                    Operand::Memory(Register::BP, -8),
                ),
                // tmp2 = tmp1 & 6
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Memory(Register::BP, -8),
                    Operand::Reg(Register::R10),
                ),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Reg(Register::R10),
                    Operand::Memory(Register::BP, -12),
                ),
                Instruction::Binary(
                    BinaryOp::BitwiseAnd,
                    AssemblyType::Longword,
                    Operand::Imm(6),
                    Operand::Memory(Register::BP, -12),
                ),
                // tmp3 = tmp0 | tmp2
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Memory(Register::BP, -4),
                    Operand::Reg(Register::R10),
                ),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Reg(Register::R10),
                    Operand::Memory(Register::BP, -16),
                ),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Memory(Register::BP, -12),
                    Operand::Reg(Register::R10),
                ),
                Instruction::Binary(
                    BinaryOp::BitwiseOr,
                    AssemblyType::Longword,
                    Operand::Reg(Register::R10),
                    Operand::Memory(Register::BP, -16),
                ),
                // return
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Memory(Register::BP, -16),
                    Operand::Reg(Register::AX),
                ),
                Instruction::Ret,
            ],
        })]);
        let (assembly, _) =
            Asm::from_tacky(ast, int_sym_table(&["tmp.0", "tmp.1", "tmp.2", "tmp.3"]));
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
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Imm(5),
                    Operand::Memory(Register::BP, -4),
                ),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Memory(Register::BP, -4),
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
                    Operand::Memory(Register::BP, -4),
                ),
                // tmp1 = tmp.0 << 2
                // moves tmp.8 into tmp.1 via reg10
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Memory(Register::BP, -4),
                    Operand::Reg(Register::R10),
                ),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Reg(Register::R10),
                    Operand::Memory(Register::BP, -8),
                ),
                Instruction::Binary(
                    BinaryOp::ShiftLeft,
                    AssemblyType::Longword,
                    Operand::Imm(2),
                    Operand::Memory(Register::BP, -8),
                ),
                // return
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Memory(Register::BP, -8),
                    Operand::Reg(Register::AX),
                ),
                Instruction::Ret,
            ],
        })]);
        let (assembly, _) = Asm::from_tacky(ast, int_sym_table(&["tmp.0", "tmp.1"]));
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
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Imm(5),
                    Operand::Memory(Register::BP, -4),
                ),
                Instruction::Unary(
                    UnaryOp::Neg,
                    AssemblyType::Longword,
                    Operand::Memory(Register::BP, -4),
                ),
                // tmp1 = tmp.0 >> 30
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Memory(Register::BP, -4),
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
                    Operand::Memory(Register::BP, -8),
                ),
                // return
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Memory(Register::BP, -8),
                    Operand::Reg(Register::AX),
                ),
                Instruction::Ret,
            ],
        })]);
        let (assembly, _) = Asm::from_tacky(ast, int_sym_table(&["tmp.0", "tmp.1"]));
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
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Imm(1),
                    Operand::Memory(Register::BP, -4),
                ),
                Instruction::Binary(
                    BinaryOp::Add,
                    AssemblyType::Longword,
                    Operand::Imm(2),
                    Operand::Memory(Register::BP, -4),
                ),
                // tmp1 = 5 << tmp.0
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Imm(5),
                    Operand::Memory(Register::BP, -8),
                ),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Memory(Register::BP, -4),
                    Operand::Reg(Register::CX),
                ),
                Instruction::Binary(
                    BinaryOp::ShiftLeft,
                    AssemblyType::Longword,
                    Operand::Reg(Register::CX),
                    Operand::Memory(Register::BP, -8),
                ),
                // return
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Memory(Register::BP, -8),
                    Operand::Reg(Register::AX),
                ),
                Instruction::Ret,
            ],
        })]);
        let (assembly, _) = Asm::from_tacky(ast, int_sym_table(&["tmp.0", "tmp.1"]));
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
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Imm(0),
                    Operand::Memory(Register::BP, -4),
                ),
                Instruction::SetCC(CondCode::E, Operand::Memory(Register::BP, -4)),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Memory(Register::BP, -4),
                    Operand::Reg(Register::AX),
                ),
                Instruction::Ret,
            ],
        })]);
        let (assembly, _) = Asm::from_tacky(ast, int_sym_table(&["tmp.0"]));
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
                Instruction::Cmp(
                    AssemblyType::Longword,
                    Operand::Imm(2),
                    Operand::Memory(Register::BP, -4),
                ),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Imm(0),
                    Operand::Memory(Register::BP, -8),
                ),
                Instruction::SetCC(CondCode::GE, Operand::Memory(Register::BP, -8)),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Memory(Register::BP, -8),
                    Operand::Reg(Register::AX),
                ),
                Instruction::Ret,
            ],
        })]);
        let (assembly, _) = Asm::from_tacky(ast, int_sym_table(&["tmp.0", "tmp.1"]));
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
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Imm(5),
                    Operand::Memory(Register::BP, -4),
                ),
                Instruction::Cmp(
                    AssemblyType::Longword,
                    Operand::Imm(0),
                    Operand::Memory(Register::BP, -4),
                ),
                Instruction::JmpCC(CondCode::E, "and_expr_false.0".into()),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Imm(1),
                    Operand::Memory(Register::BP, -8),
                ),
                Instruction::Binary(
                    BinaryOp::Add,
                    AssemblyType::Longword,
                    Operand::Imm(2),
                    Operand::Memory(Register::BP, -8),
                ),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Memory(Register::BP, -8),
                    Operand::Reg(Register::R10),
                ),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Reg(Register::R10),
                    Operand::Memory(Register::BP, -12),
                ),
                Instruction::Cmp(
                    AssemblyType::Longword,
                    Operand::Imm(0),
                    Operand::Memory(Register::BP, -12),
                ),
                Instruction::JmpCC(CondCode::E, "and_expr_false.0".into()),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Imm(1),
                    Operand::Memory(Register::BP, -16),
                ),
                Instruction::Jmp("and_expr_end.1".into()),
                Instruction::Label("and_expr_false.0".into()),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Imm(0),
                    Operand::Memory(Register::BP, -16),
                ),
                Instruction::Label("and_expr_end.1".into()),
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Memory(Register::BP, -16),
                    Operand::Reg(Register::AX),
                ),
                Instruction::Ret,
            ],
        })]);
        let (assembly, _) =
            Asm::from_tacky(ast, int_sym_table(&["tmp.0", "tmp.1", "tmp.2", "tmp.3"]));
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
        let (actual, _) = Asm::from_tacky(tacky_ast, symbol_table);
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
                        Operand::Memory(Register::BP, -4),
                    ),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Reg(Register::SI),
                        Operand::Memory(Register::BP, -8),
                    ),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Memory(Register::BP, -4),
                        Operand::Reg(Register::R10),
                    ),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Reg(Register::R10),
                        Operand::Memory(Register::BP, -12),
                    ),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Memory(Register::BP, -8),
                        Operand::Reg(Register::R10),
                    ),
                    Instruction::Binary(
                        BinaryOp::Add,
                        AssemblyType::Longword,
                        Operand::Reg(Register::R10),
                        Operand::Memory(Register::BP, -12),
                    ),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Memory(Register::BP, -12),
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
                        Operand::Memory(Register::BP, -4),
                    ),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Memory(Register::BP, -4),
                        Operand::Reg(Register::R10),
                    ),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Reg(Register::R10),
                        Operand::Memory(Register::BP, -8),
                    ),
                    Instruction::Binary(
                        BinaryOp::Add,
                        AssemblyType::Longword,
                        Operand::Imm(3),
                        Operand::Memory(Register::BP, -8),
                    ),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Memory(Register::BP, -8),
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
                        Operand::Memory(Register::BP, -4),
                    ),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Reg(Register::SI),
                        Operand::Memory(Register::BP, -8),
                    ),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Reg(Register::DX),
                        Operand::Memory(Register::BP, -12),
                    ),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Reg(Register::CX),
                        Operand::Memory(Register::BP, -16),
                    ),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Reg(Register::R8),
                        Operand::Memory(Register::BP, -20),
                    ),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Reg(Register::R9),
                        Operand::Memory(Register::BP, -24),
                    ),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Memory(Register::BP, 16),
                        Operand::Reg(Register::R10),
                    ),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Reg(Register::R10),
                        Operand::Memory(Register::BP, -28),
                    ),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Memory(Register::BP, 24),
                        Operand::Reg(Register::R10),
                    ),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Reg(Register::R10),
                        Operand::Memory(Register::BP, -32),
                    ),
                    Instruction::Mov(
                        AssemblyType::Longword,
                        Operand::Memory(Register::BP, -4),
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
        let (asm, _) = Asm::from_tacky(ast, table);
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
        let (asm, _) = Asm::from_tacky(ast, table);
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

    #[test]
    fn double_params_use_xmm_registers_and_binary_add_shuffles_through_xmm15() {
        let src = r#"
            double add_doubles(double a, double b) {
                return a + b;
            }
        "#;
        let asm = compile(src);
        let ins = func_instructions(&asm, "add_doubles");

        assert!(
            ins.iter().any(|i| matches!(
                i,
                Instruction::Mov(
                    AssemblyType::Double,
                    Operand::Reg(Register::XMM0),
                    Operand::Memory(Register::BP, _)
                )
            )),
            "expected first double param moved out of XMM0, got: {ins:?}"
        );
        assert!(
            ins.iter().any(|i| matches!(
                i,
                Instruction::Mov(
                    AssemblyType::Double,
                    Operand::Reg(Register::XMM1),
                    Operand::Memory(Register::BP, _)
                )
            )),
            "expected second double param moved out of XMM1, got: {ins:?}"
        );

        // a + b lowers to Binary(Add, Double, .., memory-dst), which is invalid for addsd, so
        // fixup must shuffle the destination through the XMM15 scratch register.
        assert!(
            ins.iter().any(|i| matches!(
                i,
                Instruction::Binary(
                    BinaryOp::Add,
                    AssemblyType::Double,
                    _,
                    Operand::Reg(Register::XMM15)
                )
            )),
            "expected Add destination shuffled into XMM15, got: {ins:?}"
        );
        assert!(
            !ins.iter().any(|i| matches!(
                i,
                Instruction::Binary(
                    BinaryOp::Add,
                    AssemblyType::Double,
                    _,
                    Operand::Memory(Register::BP, _)
                )
            )),
            "addsd cannot take a memory destination, got: {ins:?}"
        );

        // the double return value comes back in XMM0, not the integer accumulator
        assert!(
            ins.iter().any(|i| matches!(
                i,
                Instruction::Mov(AssemblyType::Double, _, Operand::Reg(Register::XMM0))
            )),
            "expected double return value moved into XMM0, got: {ins:?}"
        );
    }

    #[test]
    fn mixed_int_and_double_params_are_classified_independently() {
        let src = r#"
            double mix(int a, double b, int c, double d) {
                return b + d;
            }
        "#;
        let asm = compile(src);
        let ins = func_instructions(&asm, "mix");

        // int params take the GP registers in order, regardless of interleaved doubles
        assert!(ins.iter().any(|i| matches!(
            i,
            Instruction::Mov(AssemblyType::Longword, Operand::Reg(Register::DI), _)
        )));
        assert!(ins.iter().any(|i| matches!(
            i,
            Instruction::Mov(AssemblyType::Longword, Operand::Reg(Register::SI), _)
        )));
        // double params take the XMM registers in order, independent of the int stream
        assert!(ins.iter().any(|i| matches!(
            i,
            Instruction::Mov(AssemblyType::Double, Operand::Reg(Register::XMM0), _)
        )));
        assert!(ins.iter().any(|i| matches!(
            i,
            Instruction::Mov(AssemblyType::Double, Operand::Reg(Register::XMM1), _)
        )));
    }

    #[test]
    fn ninth_double_param_spills_to_the_stack() {
        let src = r#"
            double many_doubles(double a, double b, double c, double d, double e, double f, double g, double h, double i) {
                return i;
            }
        "#;
        let asm = compile(src);
        let ins = func_instructions(&asm, "many_doubles");

        for reg in [
            Register::XMM0,
            Register::XMM1,
            Register::XMM2,
            Register::XMM3,
            Register::XMM4,
            Register::XMM5,
            Register::XMM6,
            Register::XMM7,
        ] {
            assert!(
                ins.iter().any(|i| matches!(
                    i,
                    Instruction::Mov(AssemblyType::Double, Operand::Reg(r), _) if *r == reg
                )),
                "expected a double param moved out of {reg:?}, got: {ins:?}"
            );
        }
        // the ninth double param has no register left, so it's read from the caller's stack frame
        assert!(
            ins.iter().any(|i| matches!(
                i,
                Instruction::Mov(AssemblyType::Double, Operand::Memory(Register::BP, 16), _)
            )),
            "expected the ninth double param read from Memory(Register::BP,16), got: {ins:?}"
        );
    }

    #[test]
    fn interleaved_overflow_params_land_on_the_stack_in_declaration_order() {
        // 6 ints + 8 doubles exhaust every argument register. The remaining params overflow to
        // the stack, and critically alternate double/int/double/int in the signature: if
        // classify_values grouped overflow args by type instead of preserving encounter order,
        // this would read back the wrong stack slot for ov_i0/ov_d1.
        let src = r#"
            double many_mixed(
                int a0, int a1, int a2, int a3, int a4, int a5,
                double d0, double d1, double d2, double d3, double d4, double d5, double d6, double d7,
                double ov_d0, int ov_i0, double ov_d1, int ov_i1
            ) {
                return ov_i1;
            }
        "#;
        let asm = compile(src);
        let ins = func_instructions(&asm, "many_mixed");

        // declaration order is ov_d0, ov_i0, ov_d1, ov_i1, so stack slots must land in that
        // order too: 16, 24, 32, 40 (8 bytes per slot, regardless of type).
        assert!(
            ins.iter().any(|i| matches!(
                i,
                Instruction::Mov(AssemblyType::Double, Operand::Memory(Register::BP, 16), _)
            )),
            "expected ov_d0 read from Memory(Register::BP,16), got: {ins:?}"
        );
        assert!(
            ins.iter().any(|i| matches!(
                i,
                Instruction::Mov(AssemblyType::Longword, Operand::Memory(Register::BP, 24), _)
            )),
            "expected ov_i0 read from Memory(Register::BP,24), got: {ins:?}"
        );
        assert!(
            ins.iter().any(|i| matches!(
                i,
                Instruction::Mov(AssemblyType::Double, Operand::Memory(Register::BP, 32), _)
            )),
            "expected ov_d1 read from Memory(Register::BP,32), got: {ins:?}"
        );
        assert!(
            ins.iter().any(|i| matches!(
                i,
                Instruction::Mov(AssemblyType::Longword, Operand::Memory(Register::BP, 40), _)
            )),
            "expected ov_i1 read from Memory(Register::BP,40), got: {ins:?}"
        );
        // guard against a type-grouped regression, which would instead put both overflow
        // doubles at 16/24 and both overflow ints at 32/40.
        assert!(
            !ins.iter().any(|i| matches!(
                i,
                Instruction::Mov(AssemblyType::Double, Operand::Memory(Register::BP, 24), _)
            )),
            "ov_i0's slot should never hold a double read, got: {ins:?}"
        );
        assert!(
            !ins.iter().any(|i| matches!(
                i,
                Instruction::Mov(AssemblyType::Longword, Operand::Memory(Register::BP, 16), _)
            )),
            "ov_d0's slot should never hold an int read, got: {ins:?}"
        );
    }

    #[test]
    fn interleaved_overflow_call_args_are_pushed_in_reverse_declaration_order() {
        // mirror of the definition-side test above, but for a call site: 6 int args + 8 double
        // args exhaust the registers, and the remaining double/int/double/int args must be
        // pushed in *reverse* declaration order so the callee reads them back left-to-right.
        let src = r#"
            double many_mixed(
                int a0, int a1, int a2, int a3, int a4, int a5,
                double d0, double d1, double d2, double d3, double d4, double d5, double d6, double d7,
                double ov_d0, int ov_i0, double ov_d1, int ov_i1
            );
            double caller(void) {
                return many_mixed(1, 2, 3, 4, 5, 6, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 300.0, 100, 400.0, 200);
            }
        "#;
        let asm = compile(src);
        let ins = func_instructions(&asm, "caller");

        let label_for = |v: f64| format!("const_label.{:?}", f64::to_bits(v));
        let label_300 = label_for(300.0);
        let label_400 = label_for(400.0);

        let idx_push_200 = ins
            .iter()
            .position(|i| matches!(i, Instruction::Push(Operand::Imm(200))))
            .expect("expected ov_i1 (200) to be pushed");
        let idx_push_400 = ins
            .iter()
            .position(|i| matches!(i, Instruction::Push(Operand::Data(l)) if *l == label_400))
            .expect("expected ov_d1 (400.0) to be pushed");
        let idx_push_100 = ins
            .iter()
            .position(|i| matches!(i, Instruction::Push(Operand::Imm(100))))
            .expect("expected ov_i0 (100) to be pushed");
        let idx_push_300 = ins
            .iter()
            .position(|i| matches!(i, Instruction::Push(Operand::Data(l)) if *l == label_300))
            .expect("expected ov_d0 (300.0) to be pushed");

        // declaration order was ov_d0, ov_i0, ov_d1, ov_i1 - pushes happen in reverse, so
        // 200 should be pushed first and 300.0 last.
        assert!(
            idx_push_200 < idx_push_400
                && idx_push_400 < idx_push_100
                && idx_push_100 < idx_push_300,
            "expected push order 200, 400.0, 100, 300.0 (reverse declaration order), got: {ins:?}"
        );
    }

    #[test]
    fn double_division_uses_divdouble_not_integer_division() {
        let src = r#"
            double div_d(double a, double b) {
                return a / b;
            }
        "#;
        let asm = compile(src);
        let ins = func_instructions(&asm, "div_d");

        assert!(
            ins.iter().any(|i| matches!(
                i,
                Instruction::Binary(BinaryOp::DivDouble, AssemblyType::Double, ..)
            )),
            "expected divsd for double division, got: {ins:?}"
        );
        assert!(
            !ins.iter().any(|i| matches!(i, Instruction::Idiv(..))),
            "signed idiv should never be used for double division, got: {ins:?}"
        );
        assert!(
            !ins.iter().any(|i| matches!(i, Instruction::Div(..))),
            "unsigned div should never be used for double division, got: {ins:?}"
        );
        assert!(
            !ins.iter().any(|i| matches!(i, Instruction::Cdq(..))),
            "cdq sign-extension makes no sense for doubles, got: {ins:?}"
        );
    }

    #[test]
    fn double_greater_than_uses_unsigned_style_condcode_and_shuffles_cmp_through_xmm15() {
        let src = r#"
            int cmp_d(double a, double b) {
                return a > b;
            }
        "#;
        let asm = compile(src);
        let ins = func_instructions(&asm, "cmp_d");

        assert!(
            ins.iter()
                .any(|i| matches!(i, Instruction::SetCC(CondCode::A, _))),
            "expected SetCC(A) for double >, got: {ins:?}"
        );
        assert!(
            !ins.iter()
                .any(|i| matches!(i, Instruction::SetCC(CondCode::G, _))),
            "should never use signed CondCode::G for a double comparison, got: {ins:?}"
        );
        // comisd requires a register on the right-hand side, so fixup shuffles a memory operand
        // through XMM15.
        assert!(
            ins.iter().any(|i| matches!(
                i,
                Instruction::Cmp(AssemblyType::Double, _, Operand::Reg(Register::XMM15))
            )),
            "expected comisd operand shuffled through XMM15, got: {ins:?}"
        );
        assert!(
            !ins.iter().any(|i| matches!(
                i,
                Instruction::Cmp(AssemblyType::Double, _, Operand::Memory(Register::BP, _))
            )),
            "comisd cannot compare against two memory operands, got: {ins:?}"
        );
    }

    #[test]
    fn double_negation_emits_xor_with_sign_bit_constant() {
        let src = r#"
            double neg_d(double a) {
                return -a;
            }
        "#;
        let asm = compile(src);
        let ins = func_instructions(&asm, "neg_d");

        assert!(
            ins.iter().any(|i| matches!(
                i,
                Instruction::Binary(BinaryOp::Xor, AssemblyType::Double, Operand::Data(label), _)
                    if label.starts_with("const_label_16byte_neg_0_float")
            )),
            "expected xorpd against the -0.0 sign-bit constant, got: {ins:?}"
        );

        let Asm::Program(tops) = &asm;
        let has_neg_zero_constant = tops.iter().any(|t| {
            matches!(
                t,
                TopLevel::StaticConstant { alignment: 16, init: StaticInit::DoubleInit(z), .. }
                    if z.to_bits() == (-0.0f64).to_bits()
            )
        });
        assert!(
            has_neg_zero_constant,
            "expected a 16-byte aligned static constant holding -0.0, got: {tops:?}"
        );
    }

    #[test]
    fn int_constant_to_double_cast_moves_through_scratch_registers() {
        // the source is an immediate (invalid for cvtsi2sd) and the destination is a stack slot
        // (also invalid), so fixup must juggle both through R10 and XMM15.
        let src = r#"
            double from_literal(void) {
                return (double) 5;
            }
        "#;
        let asm = compile(src);
        let ins = func_instructions(&asm, "from_literal");

        assert!(
            ins.iter().any(|i| matches!(
                i,
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Imm(5),
                    Operand::Reg(Register::R10)
                )
            )),
            "expected the constant moved into R10 before conversion, got: {ins:?}"
        );
        assert!(
            ins.iter().any(|i| matches!(
                i,
                Instruction::Cvtsi2sd(
                    AssemblyType::Longword,
                    Operand::Reg(Register::R10),
                    Operand::Reg(Register::XMM15)
                )
            )),
            "expected cvtsi2sd converting from R10 into scratch XMM15, got: {ins:?}"
        );
        assert!(
            ins.iter().any(|i| matches!(
                i,
                Instruction::Mov(
                    AssemblyType::Double,
                    Operand::Reg(Register::XMM15),
                    Operand::Memory(Register::BP, _)
                )
            )),
            "expected the converted double moved from XMM15 into its stack slot, got: {ins:?}"
        );
    }

    #[test]
    fn double_to_int_cast_requires_register_destination_for_cvttsd2si() {
        let src = r#"
            int to_int(double x) {
                return (int) x;
            }
        "#;
        let asm = compile(src);
        let ins = func_instructions(&asm, "to_int");

        // cvttsd2si can't write directly to a stack slot, so fixup routes it through R11
        assert!(
            ins.iter().any(|i| matches!(
                i,
                Instruction::Cvttsd2si(AssemblyType::Longword, _, Operand::Reg(Register::R11))
            )),
            "expected cvttsd2si writing into scratch R11, got: {ins:?}"
        );
        assert!(
            ins.iter().any(|i| matches!(
                i,
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Reg(Register::R11),
                    Operand::Memory(Register::BP, _)
                )
            )),
            "expected the converted int moved from R11 into its stack slot, got: {ins:?}"
        );
        assert!(
            !ins.iter().any(|i| matches!(
                i,
                Instruction::Cvttsd2si(_, _, Operand::Memory(Register::BP, _))
            )),
            "cvttsd2si cannot write directly to memory, got: {ins:?}"
        );
    }

    #[test]
    fn address_of_local_var_lowers_through_lea_and_scratch_register() {
        let src = r#"
            int *addr_of(void) {
                int a = 5;
                return &a;
            }
        "#;
        let asm = compile(src);
        let ins = func_instructions(&asm, "addr_of");

        // GetAddress lowers to Lea, but lea can't write directly to a stack slot,
        // so fixup must route the destination through R10 first.
        assert!(
            ins.iter().any(|i| matches!(
                i,
                Instruction::Lea(
                    Operand::Memory(Register::BP, _),
                    Operand::Reg(Register::R10)
                )
            )),
            "expected lea computing &a into scratch R10, got: {ins:?}"
        );
        assert!(
            ins.iter().any(|i| matches!(
                i,
                Instruction::Mov(
                    AssemblyType::Quadword,
                    Operand::Reg(Register::R10),
                    Operand::Memory(Register::BP, _)
                )
            )),
            "expected the computed address moved from R10 into its stack slot, got: {ins:?}"
        );
        assert!(
            !ins.iter()
                .any(|i| matches!(i, Instruction::Lea(_, Operand::Memory(_, _)))),
            "lea cannot write directly to memory, got: {ins:?}"
        );
    }

    #[test]
    fn dereference_load_reads_through_ax() {
        let src = r#"
            int deref(int *p) {
                return *p;
            }
        "#;
        let asm = compile(src);
        let ins = func_instructions(&asm, "deref");

        assert!(
            ins.iter().any(|i| matches!(
                i,
                Instruction::Mov(AssemblyType::Quadword, _, Operand::Reg(Register::AX))
            )),
            "expected the pointer value moved into AX before dereferencing, got: {ins:?}"
        );
        assert!(
            ins.iter().any(|i| matches!(
                i,
                Instruction::Mov(AssemblyType::Longword, Operand::Memory(Register::AX, 0), _)
            )),
            "expected the pointee loaded from (%rax), got: {ins:?}"
        );
    }

    #[test]
    fn assignment_through_dereferenced_pointer_stores_through_ax() {
        let src = r#"
            int set_through_ptr(int *p, int v) {
                *p = v;
                return v;
            }
        "#;
        let asm = compile(src);
        let ins = func_instructions(&asm, "set_through_ptr");

        assert!(
            ins.iter().any(|i| matches!(
                i,
                Instruction::Mov(AssemblyType::Quadword, _, Operand::Reg(Register::AX))
            )),
            "expected the pointer value moved into AX before storing, got: {ins:?}"
        );
        assert!(
            ins.iter().any(|i| matches!(
                i,
                Instruction::Mov(AssemblyType::Longword, _, Operand::Memory(Register::AX, 0))
            )),
            "expected v stored through (%rax), got: {ins:?}"
        );
    }

    #[test]
    fn unsigned_int_to_double_cast_zero_extends_before_converting() {
        let src = r#"
            double u2d(unsigned x) {
                return (double) x;
            }
        "#;
        let asm = compile(src);
        let ins = func_instructions(&asm, "u2d");

        // MovZeroExtend into a register collapses to a plain 32-bit mov during fixup, since a
        // movl naturally zeroes the upper 32 bits of the destination 64-bit register.
        assert!(
            ins.iter().any(|i| matches!(
                i,
                Instruction::Mov(AssemblyType::Longword, _, Operand::Reg(Register::AX))
            )),
            "expected the unsigned int zero-extended into RAX before conversion, got: {ins:?}"
        );
        assert!(
            !ins.iter()
                .any(|i| matches!(i, Instruction::MovZeroExtend(..))),
            "MovZeroExtend should have been rewritten to a Mov by fixup, got: {ins:?}"
        );
        assert!(
            ins.iter().any(|i| matches!(
                i,
                Instruction::Cvtsi2sd(AssemblyType::Quadword, Operand::Reg(Register::AX), _)
            )),
            "expected the conversion to read the zero-extended 64-bit value, got: {ins:?}"
        );
    }

    #[test]
    fn double_to_unsigned_long_cast_emits_out_of_range_branch() {
        let src = r#"
            unsigned long d2ul(double x) {
                return (unsigned long) x;
            }
        "#;
        let asm = compile(src);
        let ins = func_instructions(&asm, "d2ul");

        // values that don't fit in a signed long take the slow path: compare against LONG_MAX,
        // jump out of range, then subtract/convert/add-back 2^63 on the overflow branch.
        assert!(
            ins.iter()
                .any(|i| matches!(i, Instruction::JmpCC(CondCode::AE, _))),
            "expected an out-of-range jump for values >= LONG_MAX, got: {ins:?}"
        );
        assert_eq!(
            ins.iter()
                .filter(|i| matches!(i, Instruction::Cvttsd2si(AssemblyType::Quadword, ..)))
                .count(),
            2,
            "expected one conversion on the fast path and one on the out-of-range path, got: {ins:?}"
        );
        // 2^63 doesn't fit in the 32-bit immediate that addq accepts, so fixup shuffles it
        // through R10 before adding it back on the out-of-range path.
        assert!(
            ins.iter().any(|i| matches!(
                i,
                Instruction::Mov(AssemblyType::Quadword, Operand::Imm(i), Operand::Reg(Register::R10))
                    if *i == 1usize << 63
            )),
            "expected 2^63 moved into R10 (too big for an addq immediate), got: {ins:?}"
        );
        assert!(
            ins.iter().any(|i| matches!(
                i,
                Instruction::Binary(
                    BinaryOp::Add,
                    AssemblyType::Quadword,
                    Operand::Reg(Register::R10),
                    _
                )
            )),
            "expected the out-of-range path to add back 2^63 via R10, got: {ins:?}"
        );
    }

    #[test]
    fn double_literal_becomes_a_static_rodata_constant() {
        let src = r#"
            double get_const(void) {
                return 3.5;
            }
        "#;
        let asm = compile(src);
        let Asm::Program(tops) = &asm;
        let has_constant = tops.iter().any(|t| {
            matches!(
                t,
                TopLevel::StaticConstant { alignment: 8, init: StaticInit::DoubleInit(v), .. }
                    if v.to_bits() == 3.5f64.to_bits()
            )
        });
        assert!(
            has_constant,
            "expected a static constant for the double literal 3.5, got: {tops:?}"
        );
    }

    #[test]
    fn function_call_passes_doubles_in_xmm_registers_and_ints_in_gp_registers() {
        let src = r#"
            double callee(int a, double b);
            double caller(void) {
                return callee(1, 2.5);
            }
        "#;
        let asm = compile(src);
        let ins = func_instructions(&asm, "caller");

        assert!(
            ins.iter().any(|i| matches!(
                i,
                Instruction::Mov(
                    AssemblyType::Longword,
                    Operand::Imm(1),
                    Operand::Reg(Register::DI)
                )
            )),
            "expected the int arg passed via DI, got: {ins:?}"
        );
        assert!(
            ins.iter().any(|i| matches!(
                i,
                Instruction::Mov(
                    AssemblyType::Double,
                    Operand::Data(_),
                    Operand::Reg(Register::XMM0)
                )
            )),
            "expected the double arg loaded from its constant and passed via XMM0, got: {ins:?}"
        );
        assert!(
            ins.iter()
                .any(|i| matches!(i, Instruction::Call(name) if name == "callee")),
        );
        // the double return value comes back in XMM0
        assert!(
            ins.iter().any(|i| matches!(
                i,
                Instruction::Mov(AssemblyType::Double, Operand::Reg(Register::XMM0), _)
            )),
            "expected the callee's double return value read out of XMM0, got: {ins:?}"
        );
    }

    #[test]
    fn function_call_pushes_overflow_args_with_correct_padding() {
        let src = r#"
            int many(int a, int b, int c, int d, int e, int f, int g);
            int caller2(void) {
                return many(1, 2, 3, 4, 5, 6, 7);
            }
        "#;
        let asm = compile(src);
        let ins = func_instructions(&asm, "caller2");

        // 7 int args: 6 in registers, 1 pushed to the stack. Since that leaves an odd number of
        // stack args, 8 bytes of padding are pushed first to keep %rsp 16-byte aligned.
        assert!(
            ins.iter().any(|i| matches!(
                i,
                Instruction::Binary(
                    BinaryOp::Sub,
                    AssemblyType::Quadword,
                    Operand::Imm(8),
                    Operand::Reg(Register::SP)
                )
            )),
            "expected 8-byte stack padding before pushing the single overflow arg, got: {ins:?}"
        );
        assert!(
            ins.iter()
                .any(|i| matches!(i, Instruction::Push(Operand::Imm(7)))),
            "expected the seventh argument pushed onto the stack, got: {ins:?}"
        );
    }
}

// some niceties. Maybe move to a from.rs
impl From<tacky::UnaryOp> for UnaryOp {
    fn from(op: tacky::UnaryOp) -> Self {
        match op {
            tacky::UnaryOp::Complement => UnaryOp::Not,
            tacky::UnaryOp::Negate => UnaryOp::Neg,
            tacky::UnaryOp::Not => {
                unreachable!("tacky Not is lowered to Cmp+SetCC, never goes through From")
            }
        }
    }
}

impl From<&tacky::UnaryOp> for UnaryOp {
    fn from(op: &tacky::UnaryOp) -> Self {
        match op {
            &tacky::UnaryOp::Complement => UnaryOp::Not,
            &tacky::UnaryOp::Negate => UnaryOp::Neg,
            &tacky::UnaryOp::Not => {
                unreachable!("tacky Not is lowered to Cmp+SetCC, never goes through From")
            }
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
