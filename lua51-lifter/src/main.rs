#![feature(box_patterns)]
#![feature(let_chains)]
#![feature(mixed_integer_ops)]

use ast::name_locals::name_locals;
use std::io::Write;
use std::{fs::File, io::Read};

use clap::Parser;

use lua51_deserializer::chunk::Chunk;

mod lifter;

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

#[derive(Parser, Debug)]
#[clap(about, version, author)]
struct Args {
    #[clap(short, long)]
    file: String,
}

fn main() -> anyhow::Result<()> {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    let args = Args::parse();
    let mut input = File::open(args.file)?;
    let mut buffer = vec![0; input.metadata()?.len() as usize];
    input.read_exact(&mut buffer)?;

    let chunk = Chunk::parse(&buffer).unwrap().1;
    let (mut main, _, _) = lifter::LifterContext::lift(&chunk.function, Default::default());
    name_locals(&mut main, true);
    write!(File::create("result.lua")?, "{}", main)?;

    Ok(())
}
