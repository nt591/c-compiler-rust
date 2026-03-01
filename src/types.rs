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
