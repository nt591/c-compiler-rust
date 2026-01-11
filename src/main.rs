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
    inputs: Vec<PathBuf>,

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

    /// Compiles to an object file
    #[arg(short = 'c')]
    object: bool,
}

#[derive(PartialEq, Debug, Copy, Clone)]
enum ProcessingStage {
    Lex,
    Parse,
    Validate,
    Codegen,
    Tacky,
    Object,
    Full,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let stage = args_to_processing_stage(&args);
    let inputs: Vec<PathBuf> = args
        .inputs
        .into_iter()
        .map(|input| input.canonicalize().expect("Not a valid input file"))
        .collect();
    let assembly_files = inputs
        .into_iter()
        .map(|file| {
            // object files shouldn't be processed, so we'll skip
            if let Some("o") = file.extension().and_then(|f| f.to_str()) {
                Ok(Some(file))
            } else {
                process_file(file, stage)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;

    let assembly_files = assembly_files.into_iter().flat_map(|x| x).collect();
    match stage {
        ProcessingStage::Object => compile_to_object(assembly_files)?,
        ProcessingStage::Full => compile_to_binary(assembly_files)?,
        _ => (),
    };
    Ok(())
}

fn args_to_processing_stage(args: &Args) -> ProcessingStage {
    if args.lex {
        return ProcessingStage::Lex;
    };
    if args.parse {
        return ProcessingStage::Parse;
    };
    if args.validate {
        return ProcessingStage::Validate;
    };
    if args.codegen {
        return ProcessingStage::Codegen;
    };
    if args.tacky {
        return ProcessingStage::Tacky;
    };
    if args.object {
        return ProcessingStage::Object;
    };
    ProcessingStage::Full
}

// returns Some(PathBuf) if we wrote out assembly files
fn process_file(input: PathBuf, stage: ProcessingStage) -> anyhow::Result<Option<PathBuf>> {
    let f = File::open(&input)?;
    let reader = BufReader::new(f);
    let contents: String = reader.lines().collect::<Result<Vec<_>, _>>()?.join("\n");
    let lexer = Lexer::lex(&contents)?;
    if stage == ProcessingStage::Lex {
        return Ok(None);
    };

    let tokens = lexer.as_syntactic_tokens();
    let parser = Parser::new(&tokens);
    let mut ast = parser.into_ast()?;
    if stage == ProcessingStage::Parse {
        return Ok(None);
    };

    let symbol_table = semantic_analysis::resolve(&mut ast)?;
    if stage == ProcessingStage::Validate {
        return Ok(None);
    };

    let asm = Tacky::new(ast);
    let ast = asm.into_ast(symbol_table)?;
    if stage == ProcessingStage::Tacky {
        return Ok(None);
    }
    let asm = Asm::from_tacky(ast);
    if stage == ProcessingStage::Codegen {
        return Ok(None);
    };

    let emitter = Emitter::new(asm);
    let output_path = input.with_extension("S");
    let output_file = File::create(&output_path)?;
    let mut writer = BufWriter::new(output_file);
    emitter.emit(&mut writer)?;
    Ok(Some(output_path))
}

fn path_to_str(path: PathBuf) -> String {
    path.into_os_string()
        .into_string()
        .expect("Should be valid string")
}

fn compile_to_binary(paths: Vec<PathBuf>) -> anyhow::Result<()> {
    debug_assert!(!paths.is_empty());
    let fst = paths[..].first().unwrap().to_owned();
    let target = fst.with_extension("");
    let mut args = Vec::with_capacity(paths.len() + 2);
    args.extend(paths.into_iter().map(path_to_str));
    args.push("-o".into());
    args.push(path_to_str(target));

    std::process::Command::new("gcc").args(args).spawn()?;
    Ok(())
}

fn compile_to_object(paths: Vec<PathBuf>) -> anyhow::Result<()> {
    debug_assert!(!paths.is_empty());

    // GCC doesn't allow specifying -o flags when compiling multiple files
    let target_file = match paths.len() {
        1 => {
            let path = paths[..].first().unwrap().to_owned();
            Some(path_to_str(path.with_extension("o")))
        }
        _otherwise => None,
    };

    let mut args: Vec<String> = Vec::with_capacity(paths.len() + 1);
    args.push("-c".into());
    args.extend(paths.into_iter().map(path_to_str));
    if let Some(target) = target_file {
        args.push("-o".into());
        args.push(target);
    };
    std::process::Command::new("gcc").args(args).spawn()?;
    Ok(())
}
