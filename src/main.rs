use clap::Parser as ClapParser;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::path::PathBuf;

pub mod lexer;
pub mod parser;
use lexer::Lexer;
use parser::Parser;

#[derive(ClapParser, Debug)]
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

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let input = args.input.canonicalize().expect("Not a valid input file");
    if args.lex {
        let f = File::open(input)?;
        let reader = BufReader::new(f);
        let contents: String = reader.lines().collect::<Result<Vec<_>, _>>()?.join("\n");
        Lexer::lex(&contents)?;
        println!("lexing");
    } else if args.parse {
        let f = File::open(input)?;
        let reader = BufReader::new(f);
        let contents: String = reader.lines().collect::<Result<Vec<_>, _>>()?.join("\n");
        let lexer = Lexer::lex(&contents)?;
        let parser = Parser::new(lexer.as_tokens());
        let _ast = parser.into_ast()?;
        println!("parsing");
    } else if args.codegen {
        println!("codegen");
    } else {
        println!("full run");
    }
    Ok(())
}