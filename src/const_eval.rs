// used for constant folding and various type conversions
use crate::ast::Const;
use crate::types::CType;
use crate::types::StaticInit;

pub fn convert_const(c: Const, target: &CType) -> StaticInit {
    // u64 is the widest representation that'll fit, so sign-extend negatives,
    // zero extend unsigned
    // Doubles work a little weirdly, we need to truncate towards zero if double -> int.
    // For int -> double we do either exact match, or closest representable.
    if target == &CType::Double {
        return const_to_double(c);
    }
    if let Const::Double(_) = c {
        return double_to_const(c, target);
    }
    let bits: u64 = match c {
        Const::UInt(i) => i as u64,
        Const::Int(i) => i as u64,
        Const::Long(i) => i as u64,
        Const::ULong(i) => i as u64,
        Const::Double(f) => f as u64,
    };
    match target {
        CType::UInt => StaticInit::UIntInit(bits as u32),
        CType::Int => StaticInit::IntInit(bits as i32),
        CType::ULong => StaticInit::ULongInit(bits as u64),
        CType::Long => StaticInit::LongInit(bits as i64),
        CType::Double => StaticInit::DoubleInit(bits as f64),
        CType::FunType { .. } => {
            unreachable!("Should never try to convert a const node to a function type")
        }
    }
}

fn const_to_double(c: Const) -> StaticInit {
    match c {
        Const::UInt(i) => StaticInit::DoubleInit(f64::from(i)),
        Const::Int(i) => StaticInit::DoubleInit(f64::from(i)),
        Const::Long(i) => StaticInit::DoubleInit(i as f64),
        Const::ULong(i) => StaticInit::DoubleInit(i as f64),
        Const::Double(f) => StaticInit::DoubleInit(f),
    }
}

fn double_to_const(c: Const, target: &CType) -> StaticInit {
    let Const::Double(bits) = c else {
        panic!("Only accept doubles");
    };
    match target {
        CType::Int => StaticInit::IntInit(bits as i32),
        CType::Long => StaticInit::LongInit(bits as i64),
        CType::UInt => StaticInit::UIntInit(bits as u32),
        CType::ULong => StaticInit::ULongInit(bits as u64),
        CType::Double => StaticInit::DoubleInit(bits),
        CType::FunType { .. } => panic!("Cannot convert double to funtype"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Const;
    use crate::types::{CType, StaticInit};

    // --- double → integer: must truncate toward zero ---

    #[test]
    fn double_to_int_truncates_toward_zero_positive() {
        assert_eq!(
            convert_const(Const::Double(2.8), &CType::Int),
            StaticInit::IntInit(2)
        );
    }

    #[test]
    fn double_to_int_truncates_toward_zero_negative() {
        // -2.8 truncated toward zero is -2, not -3
        assert_eq!(
            convert_const(Const::Double(-2.8), &CType::Int),
            StaticInit::IntInit(-2)
        );
    }

    #[test]
    fn double_to_long_truncates_toward_zero_negative() {
        assert_eq!(
            convert_const(Const::Double(-9.9), &CType::Long),
            StaticInit::LongInit(-9)
        );
    }

    #[test]
    fn double_to_uint_truncates_positive() {
        assert_eq!(
            convert_const(Const::Double(2.8), &CType::UInt),
            StaticInit::UIntInit(2)
        );
    }

    // --- integer → double: must preserve sign ---

    #[test]
    fn negative_int_to_double_preserves_sign() {
        assert_eq!(
            convert_const(Const::Int(-3), &CType::Double),
            StaticInit::DoubleInit(-3.0)
        );
    }

    #[test]
    fn negative_long_to_double_preserves_sign() {
        assert_eq!(
            convert_const(Const::Long(-100), &CType::Double),
            StaticInit::DoubleInit(-100.0)
        );
    }

    #[test]
    fn positive_int_to_double() {
        assert_eq!(
            convert_const(Const::Int(5), &CType::Double),
            StaticInit::DoubleInit(5.0)
        );
    }

    // --- double → double: must preserve fractional part ---

    #[test]
    fn double_to_double_preserves_fractional_part() {
        assert_eq!(
            convert_const(Const::Double(1.5), &CType::Double),
            StaticInit::DoubleInit(1.5)
        );
    }
}
