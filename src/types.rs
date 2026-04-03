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
}

impl CType {
    pub fn get_common_type(t1: CType, t2: CType) -> CType {
        if t1 == t2 { t1 } else { CType::Long }
    }
}

pub fn static_init_as_usize(si: &StaticInit) -> usize {
    match si {
        StaticInit::IntInit(i) => *i as usize,
        StaticInit::LongInit(i) => *i as usize,
    }
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum AssemblyType {
    Longword,
    Quadword,
}
