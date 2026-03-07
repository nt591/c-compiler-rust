// valid C types (int, unsigned, long, ...)
#[derive(Debug, PartialEq, Clone)]
pub enum CType {
    Int,
    Long,
    FunType {
        params: Vec<CType>,
        ret: Box<CType>, // TODO: intern CType and carry a u32 everywhere.
    },
}

impl CType {
    pub fn get_common_type(t1: CType, t2: CType) -> CType {
        if t1 == t2 { t1 } else { CType::Long }
    }
}
