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
    ObjEntry { ty: AssemblyType, is_static: bool },
    FunEntry { defined: bool },
}

pub type BackendSymbolTable = HashMap<String, BackendSymTableEntry>;

pub fn backend_symbol_table_from_symbol_table(symtable: SymbolTable) -> BackendSymbolTable {
    let mut new_table = HashMap::with_capacity(symtable.len());
    for (name, (ctype, attrs)) in symtable {
        let entry = match (ctype, attrs) {
            (CType::UInt | CType::ULong, _) => todo!(),
            (CType::FunType { .. }, IdentifierAttrs::FunAttr { defined, .. }) => {
                BackendSymTableEntry::FunEntry { defined }
            }
            (CType::Int, IdentifierAttrs::StaticAttr { .. }) => BackendSymTableEntry::ObjEntry {
                ty: AssemblyType::Longword,
                is_static: true,
            },
            (CType::Int, _) => BackendSymTableEntry::ObjEntry {
                ty: AssemblyType::Longword,
                is_static: false,
            },
            (CType::Long, IdentifierAttrs::StaticAttr { .. }) => BackendSymTableEntry::ObjEntry {
                ty: AssemblyType::Quadword,
                is_static: true,
            },
            (CType::Long, _) => BackendSymTableEntry::ObjEntry {
                ty: AssemblyType::Quadword,
                is_static: false,
            },
            (CType::FunType { .. }, IdentifierAttrs::StaticAttr { .. })
            | (CType::FunType { .. }, IdentifierAttrs::LocalAttr) => unreachable!(),
        };
        new_table.insert(name, entry);
    }
    new_table
}
