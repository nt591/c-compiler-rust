use clap::Parser as ClapParser;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::io::BufWriter;
use std::io::Write;
use std::path::PathBuf;

mod asm;
mod emitter;
mod lexer;
mod parser;
use asm::Asm;
use emitter::Emitter;
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
    } else if args.parse {
        let f = File::open(input)?;
        let reader = BufReader::new(f);
        let contents: String = reader.lines().collect::<Result<Vec<_>, _>>()?.join("\n");
        let lexer = Lexer::lex(&contents)?;
        let tokens = lexer.as_syntactic_tokens();
        let parser = Parser::new(&tokens);
        let _ast = parser.into_ast()?;
    } else if args.codegen {
        let f = File::open(input)?;
        let reader = BufReader::new(f);
        let contents: String = reader.lines().collect::<Result<Vec<_>, _>>()?.join("\n");
        let lexer = Lexer::lex(&contents)?;
        let tokens = lexer.as_syntactic_tokens();
        let parser = Parser::new(&tokens);
        let ast = parser.into_ast()?;
        let asm = Asm::new(ast);
        let _ast = asm.into_ast()?;
    } else {
        let f = File::open(&input)?;
        let reader = BufReader::new(f);
        let contents: String = reader.lines().collect::<Result<Vec<_>, _>>()?.join("\n");
        let lexer = Lexer::lex(&contents)?;
        let tokens = lexer.as_syntactic_tokens();
        let parser = Parser::new(&tokens);
        let ast = parser.into_ast()?;
        let asm = Asm::new(ast);
        let ast = asm.into_ast()?;
        let emitter = Emitter::new(ast);
        let code = emitter.emit();

        let output_path = input.with_extension("s");
        let output_file = File::create(&output_path)?;

        let mut writer = BufWriter::new(output_file);
        writer.write_all(code.as_bytes())?;

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
