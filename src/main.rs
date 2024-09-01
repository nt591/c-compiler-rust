use clap::Parser;
use std::path::PathBuf;

pub mod lexer;

#[derive(Parser, Debug)]
#[clap(name = "C Compiler")]
#[clap(author = "Nikhil Thomas")]
struct Args {
    input: PathBuf,

    /// Run lexical analysis
    #[arg(long, conflicts_with_all = ["parse", "codegen"])]
    lex: bool,

    /// Run parsing
    #[arg(long, conflicts_with_all = ["lex", "codegen"])]
    parse: bool,

    /// Run code generation
    #[arg(long, conflicts_with_all = ["lex", "parse"])]
    codegen: bool,
}

fn main() {
    let args = Args::parse();
    let _input = args.input.canonicalize().expect("Not a valid input file");
    if args.lex {
        println!("lexing");
    } else if args.parse {
        println!("parsing");
    } else if args.codegen {
        println!("codegen");
    } else {
        println!("full run");
    }
    ()
}
