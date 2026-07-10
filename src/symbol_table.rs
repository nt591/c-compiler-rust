use crate::types::AssemblyType;
use crate::types::CType;
use crate::types::StaticInit;
use std::collections::HashMap;
#[derive(PartialEq, Debug)]
pub enum InitialValue {
    Tentative,
    Initial(StaticInit),
    NoInitializer, // extern variable declarations are not tentative
}

#[derive(Debug, PartialEq)]
pub enum IdentifierAttrs {
    FunAttr { defined: bool, global: bool },
    StaticAttr { init: InitialValue, global: bool },
    LocalAttr,
}

// TODO: harden types so CType::FunType ONLY goes with IdentifierAttrs::FunAttr
pub type SymbolTable = HashMap<String, (CType, IdentifierAttrs)>;

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum BackendSymTableEntry {
    ObjEntry {
        ty: AssemblyType,
        is_static: bool,
        is_constant: bool,
    },
    FunEntry {
        defined: bool,
    },
}

pub type BackendSymbolTable = HashMap<String, BackendSymTableEntry>;

pub fn backend_symbol_table_from_symbol_table(symtable: SymbolTable) -> BackendSymbolTable {
    let mut new_table = HashMap::with_capacity(symtable.len());
    for (name, (ctype, attrs)) in symtable {
        let entry = match (ctype, attrs) {
            (CType::FunType { .. }, IdentifierAttrs::FunAttr { defined, .. }) => {
                BackendSymTableEntry::FunEntry { defined }
            }
            (CType::Int | CType::UInt, IdentifierAttrs::StaticAttr { .. }) => {
                BackendSymTableEntry::ObjEntry {
                    ty: AssemblyType::Longword,
                    is_static: true,
                    is_constant: false,
                }
            }
            (CType::Int | CType::UInt, _) => BackendSymTableEntry::ObjEntry {
                ty: AssemblyType::Longword,
                is_static: false,
                is_constant: false,
            },
            (CType::Long | CType::ULong, IdentifierAttrs::StaticAttr { .. }) => {
                BackendSymTableEntry::ObjEntry {
                    ty: AssemblyType::Quadword,
                    is_static: true,
                    is_constant: false,
                }
            }
            (CType::Long | CType::ULong, _) => BackendSymTableEntry::ObjEntry {
                ty: AssemblyType::Quadword,
                is_static: false,
                is_constant: false,
            },
            (CType::Double, IdentifierAttrs::StaticAttr { .. }) => BackendSymTableEntry::ObjEntry {
                ty: AssemblyType::Double,
                is_static: true,
                is_constant: false, // tweak this in asm.rs
            },

            (CType::Double, _) => BackendSymTableEntry::ObjEntry {
                ty: AssemblyType::Double,
                is_static: false,
                is_constant: false, // tweak this in asm.rs
            },
            (CType::FunType { .. }, IdentifierAttrs::StaticAttr { .. })
            | (CType::FunType { .. }, IdentifierAttrs::LocalAttr) => unreachable!(),
            (CType::Pointer(_), _) => todo!(),
        };
        new_table.insert(name, entry);
    }
    new_table
}
