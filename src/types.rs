// valid C types (int, unsigned, long, ...)
#[derive(Debug, PartialEq, Clone)]
pub enum CType {
    Int,
    Long,
    UInt,
    ULong,
    FunType {
        params: Vec<CType>,
        ret: Box<CType>, // TODO: intern CType and carry a u32 everywhere.
    },
}

#[derive(PartialEq, Debug, Copy, Clone)]
pub enum StaticInit {
    IntInit(i32),
    LongInit(i64),
    UIntInit(u32),
    ULongInit(u64),
}

impl CType {
    /*
     * Uses C standard ranking logic. Important notes for me:
     * The rank of a signed integer type shall be greater than the rank of any signed integer
     * type with less precision.
     * The rank of any unsigned integer type shall equal the rank of the corresponding
     * signed integer type, if any.
     * TL;DR for me: rank is based on size first.
     */
    pub fn get_common_type(t1: CType, t2: CType) -> CType {
        // C standard specifies in 6.3.1.8. Copying for clarity.
        // (1): If both operands have the same type, then no further conversion is needed.
        if t1 == t2 {
            return t1;
        }

        /*
         * (3): Otherwise, if the operand that has unsigned integer type has rank greater
         *  or equal to the rank of the type of the other operand, then the operand
         *  with signed integer type is converted to the type of the operand with
         *  unsigned integer type.
         * (4): Otherwise, if the type of the operand with signed integer type can represent
         * all of the values of the type of the operand with unsigned integer type, then
         * the operand with unsigned integer type is converted to the type of the
         * operand with signed integer type.
         *  IN OTHER WORDS: two operands of same size, take the unsigned version
         */
        if t1.size() == t2.size() {
            if t1.signed() {
                return t2;
            }
            return t1;
        }
        /*
         * (2): Otherwise, if both operands have signed integer types or both have unsigned
         * integer types, the operand with the type of lesser integer conversion rank is
         * converted to the type of the operand with greater rank.
         * IN OTHER WORDS: pick the bigger sized one since it should fit all values of the smaller.
         */
        if t1.size() > t2.size() {
            return t1;
        } else {
            return t2;
        }
    }

    fn size(&self) -> usize {
        match self {
            CType::Int | CType::UInt => 32,
            CType::Long | CType::ULong => 64,
            CType::FunType { .. } => todo!(), // not sure what goes here.
        }
    }

    fn signed(&self) -> bool {
        match self {
            CType::Int | CType::Long => true,
            CType::UInt | CType::ULong | CType::FunType { .. } => false,
        }
    }
}

pub fn static_init_as_usize(si: &StaticInit) -> usize {
    match si {
        StaticInit::IntInit(i) => *i as usize,
        StaticInit::LongInit(i) => *i as usize,
        StaticInit::UIntInit(i) => *i as usize,
        StaticInit::ULongInit(i) => *i as usize,
    }
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum AssemblyType {
    Longword,
    Quadword,
}
