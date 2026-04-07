use crate::types::CType;

#[derive(Debug, PartialEq, Clone)]
pub enum AST<T> {
    Program(Vec<Declaration<T>>),
}

#[derive(Debug, PartialEq, Clone)]
pub enum Declaration<T> {
    VarDecl(VariableDeclaration<T>),
    FunDecl(FunctionDeclaration<T>),
}

#[derive(Debug, PartialEq, Clone)]
pub struct FunctionDeclaration<T> {
    pub name: String,
    pub params: Vec<String>, // should this be owned?
    pub block: Option<Block<T>>,
    pub storage_class: Option<StorageClass>,
    pub ftype: CType,
}

impl<T> FunctionDeclaration<T> {
    pub fn new(
        name: String,
        params: Vec<(String, CType)>,
        block: Option<Block<T>>,
        storage_class: Option<StorageClass>,
        return_type: CType,
    ) -> FunctionDeclaration<T> {
        let ftype = CType::FunType {
            ret: Box::new(return_type),
            params: params.iter().map(|(_, t)| t.clone()).collect(),
        };
        let params = params.into_iter().map(|(s, _)| s).collect();
        FunctionDeclaration {
            name,
            params,
            block,
            storage_class,
            ftype,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct VariableDeclaration<T> {
    pub name: String,
    pub init: Option<Expression<T>>,
    pub storage_class: Option<StorageClass>,
    pub vtype: CType,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum StorageClass {
    Static,
    Extern,
}

#[derive(Debug, PartialEq, Clone)]
pub enum ForInit<T> {
    InitDecl(VariableDeclaration<T>),
    InitExp(Option<Expression<T>>),
}

#[derive(Debug, PartialEq, Clone)]
pub enum Statement<T> {
    Goto(String),
    Labelled {
        label: String,
        statement: Box<Statement<T>>,
    },
    Return(Expression<T>),
    Expr(Expression<T>),
    If {
        condition: Expression<T>,
        then: Box<Statement<T>>,
        else_: Option<Box<Statement<T>>>,
    },
    Compound(Block<T>),
    Break(String),
    Continue(String),
    While {
        condition: Expression<T>,
        body: Box<Statement<T>>,
        label: String,
    },
    DoWhile {
        body: Box<Statement<T>>,
        condition: Expression<T>,
        label: String,
    },
    For {
        init: ForInit<T>,
        condition: Option<Expression<T>>,
        post: Option<Expression<T>>,
        body: Box<Statement<T>>,
        label: String,
    },
    Null,
}

#[derive(Debug, PartialEq, Clone)]
pub enum BlockItem<T> {
    Stmt(Statement<T>),
    Decl(Declaration<T>),
}

#[derive(Debug, PartialEq, Clone)]
pub struct Block<T>(pub Vec<BlockItem<T>>);

#[derive(Debug, PartialEq, Clone)]
pub struct Expression<T> {
    pub ty: T,
    pub kind: Box<ExprKind<T>>,
}

impl Expression<()> {
    pub fn new(kind: ExprKind<()>) -> Self {
        Expression {
            kind: Box::new(kind),
            ty: (),
        }
    }
}
impl<T: Clone> Expression<T> {
    pub fn get_type(&self) -> T {
        self.ty.clone()
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum ExprKind<T> {
    Constant(Const),
    Var(String), // identifier for variable
    Unary(UnaryOp, Box<Expression<T>>),
    Binary(BinaryOp, Box<Expression<T>>, Box<Expression<T>>),
    Assignment(Box<Expression<T>>, Box<Expression<T>>), // LHS, RHS
    Conditional {
        condition: Box<Expression<T>>,
        then: Box<Expression<T>>,
        else_: Box<Expression<T>>,
    },
    FunctionCall {
        name: String,
        args: Vec<Expression<T>>,
    },
    Cast(CType, Box<Expression<T>>),
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Const {
    Int(i32),
    Long(i64),
    UInt(u32),
    ULong(u64),
    Double(f64),
}

impl Const {
    pub fn to_ctype(&self) -> CType {
        match self {
            Const::Int(_) => CType::Int,
            Const::Long(_) => CType::Long,
            Const::UInt(_) => CType::UInt,
            Const::ULong(_) => CType::ULong,
            Const::Double(_) => CType::Double,
        }
    }
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum UnaryOp {
    Negate,
    Complement,
    Not,
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum BinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Remainder,
    BitwiseAnd,
    BitwiseOr,
    Xor,
    ShiftLeft,
    ShiftRight,
    Equal,
    NotEqual,
    LessThan,
    LessOrEqual,
    GreaterThan,
    GreaterOrEqual,
    BinAnd,
    BinOr,
    // Compound assignment operators
    AddAssign,
    MinusAssign,
    MultiplyAssign,
    DivideAssign,
    RemainderAssign,
    BitwiseAndAssign,
    BitwiseOrAssign,
    XorAssign,
    ShiftLeftAssign,
    ShiftRightAssign,
}
