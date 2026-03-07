// used for constant folding and various type conversions
use crate::ast::Const;
use crate::semantic_analysis::StaticInit;
use crate::types::CType;

pub fn convert_const(c: Const, target: &CType) -> StaticInit {
    match (c, target) {
        (Const::Int(i), CType::Long) => StaticInit::LongInit(i as i64),
        (Const::Int(i), CType::Int) => StaticInit::IntInit(i),
        (Const::Long(l), CType::Long) => StaticInit::LongInit(l),
        (Const::Long(l), CType::Int) => StaticInit::IntInit(l as i32),
        (_, CType::FunType { .. }) => {
            unreachable!("Should never try to convert a const node to a function type")
        }
    }
}
