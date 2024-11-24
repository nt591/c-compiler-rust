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
    #[arg(long, conflicts_with_all = ["parse", "codegen", "tacky"])]
    lex: bool,

    /// Run parsing
    #[arg(long, conflicts_with_all = ["lex", "codegen", "tacky"])]
    parse: bool,

    /// Run code generation
    #[arg(long, conflicts_with_all = ["lex", "parse", "tacky"])]
    codegen: bool,

    /// runs up to IR stage
    #[arg(long, conflicts_with_all = ["lex", "parse", "codegen"])]
    tacky: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let input = args.input.canonicalize().expect("Not a valid input file");
    let f = File::open(&input)?;
    let reader = BufReader::new(f);
    let contents: String = reader.lines().collect::<Result<Vec<_>, _>>()?.join("\n");
    if args.lex {
        Lexer::lex(&contents)?;
    } else if args.parse {
        let lexer = Lexer::lex(&contents)?;
        let tokens = lexer.as_syntactic_tokens();
        let parser = Parser::new(&tokens);
        let _ast = parser.into_ast()?;
    } else if args.codegen {
        let lexer = Lexer::lex(&contents)?;
        let tokens = lexer.as_syntactic_tokens();
        let parser = Parser::new(&tokens);
        let ast = parser.into_ast()?;
        let asm = Tacky::new(ast);
        let ast = asm.into_ast()?;
        let _asm = Asm::from_tacky(ast);
    } else if args.tacky {
        let lexer = Lexer::lex(&contents)?;
        let tokens = lexer.as_syntactic_tokens();
        let parser = Parser::new(&tokens);
        let ast = parser.into_ast()?;
        let asm = Tacky::new(ast);
        let _ast = asm.into_ast()?;
    } else {
        let lexer = Lexer::lex(&contents)?;
        let tokens = lexer.as_syntactic_tokens();
        let parser = Parser::new(&tokens);
        let ast = parser.into_ast()?;
        let asm = Tacky::new(ast);
        let ast = asm.into_ast()?;
        let asm = Asm::from_tacky(ast);
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
    }
    Ok(())
}
