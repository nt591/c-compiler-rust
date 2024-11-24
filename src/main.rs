use clap::Parser as ClapParser;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::io::BufWriter;
use std::path::PathBuf;

mod asm;
mod emitter;
mod lexer;
mod parser;
mod semantic_analysis;
mod tacky;
use asm::Asm;
use emitter::Emitter;
use lexer::Lexer;
use parser::Parser;
use tacky::Tacky;

#[derive(ClapParser, Debug)]
#[clap(name = "C Compiler")]
#[clap(author = "Nikhil Thomas")]
struct Args {
    input: PathBuf,

    /// Run lexical analysis
    #[arg(long, conflicts_with_all = ["parse", "codegen", "tacky", "validate"])]
    lex: bool,

    /// Run parsing
    #[arg(long, conflicts_with_all = ["lex", "codegen", "tacky", "validate"])]
    parse: bool,

    /// Run semantic analysis stages
    #[arg(long, conflicts_with_all = ["lex", "codegen", "parse", "tacky"])]
    validate: bool,

    /// Run code generation
    #[arg(long, conflicts_with_all = ["lex", "parse", "tacky", "validate"])]
    codegen: bool,

    /// runs up to IR stage
    #[arg(long, conflicts_with_all = ["lex", "parse", "codegen", "validate"])]
    tacky: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let input = args.input.canonicalize().expect("Not a valid input file");
    let f = File::open(&input)?;
    let reader = BufReader::new(f);
    let contents: String = reader.lines().collect::<Result<Vec<_>, _>>()?.join("\n");
    let lexer = Lexer::lex(&contents)?;
    if args.lex {
        return Ok(());
    };

    let tokens = lexer.as_syntactic_tokens();
    let parser = Parser::new(&tokens);
    let mut ast = parser.into_ast()?;
    if args.parse {
        return Ok(());
    };

    semantic_analysis::resolve(&mut ast)?;
    if args.validate {
        return Ok(());
    };

    let asm = Tacky::new(ast);
    let ast = asm.into_ast()?;
    if args.tacky {
        return Ok(());
    }
    let asm = Asm::from_tacky(ast);
    if args.codegen {
        return Ok(());
    };

    let emitter = Emitter::new(asm);
    let output_path = input.with_extension("s");
    let output_file = File::create(&output_path)?;
    let mut writer = BufWriter::new(output_file);
    emitter.emit(&mut writer)?;

    let target = output_path.with_extension("");

    // defer to GCC to assemble and link
    let path_to_str = |p: PathBuf| {
        p.into_os_string()
            .into_string()
            .expect("Should be valid string")
    };
    let _cmd = std::process::Command::new("gcc")
        .args([path_to_str(output_path), "-o".into(), path_to_str(target)])
        .spawn()?;
    Ok(())
}
