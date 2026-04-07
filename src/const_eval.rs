// used for constant folding and various type conversions
use crate::ast::Const;
use crate::types::CType;
use crate::types::StaticInit;

pub fn convert_const(c: Const, target: &CType) -> StaticInit {
    // u64 is the widest representation that'll fit, so sign-extend negatives,
    // zero extend unsigned
    let bits: u64 = match c {
        Const::UInt(i) => i as u64,
        Const::Int(i) => i as u64,
        Const::Long(i) => i as u64,
        Const::ULong(i) => i as u64,
        Const::Double(_) => todo!(),
    };
    match target {
        CType::UInt => StaticInit::UIntInit(bits as u32),
        CType::Int => StaticInit::IntInit(bits as i32),
        CType::ULong => StaticInit::ULongInit(bits as u64),
        CType::Long => StaticInit::LongInit(bits as i64),
        CType::Double => todo!(),
        CType::FunType { .. } => {
            unreachable!("Should never try to convert a const node to a function type")
        }
    }
}
