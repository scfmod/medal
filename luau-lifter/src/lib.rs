mod deserializer;
mod instruction;
mod lifter;
mod op_code;

use ast::{
    cleanup_for_statements::cleanup_for_statements, inline_temporaries::inline_temporaries,
    local_declarations::LocalDeclarer, name_locals::name_locals, replace_locals::replace_locals,
    Traverse,
};

use by_address::ByAddress;
use cfg::{
    function::Function,
    ssa::{
        self,
        structuring::{structure_conditionals, structure_jumps},
    },
};
use indexmap::IndexMap;

use lifter::Lifter;

//use cfg_ir::{dot, function::Function, ssa};
use clap::Parser;
use parking_lot::Mutex;
use petgraph::algo::dominators::simple_fast;
use rayon::prelude::*;

use anyhow::anyhow;
use rustc_hash::{FxHashMap, FxHashSet};
use triomphe::Arc;
use walkdir::WalkDir;

use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
    time::Instant,
};

use deserializer::bytecode::Bytecode;

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

#[derive(Parser, Debug)]
#[clap(about, version, author)]
struct Args {
    paths: Vec<String>,
    /// Number of threads to use (0 = automatic)
    #[clap(short, long, default_value_t = 0)]
    threads: usize,
    /// op = op * key % 256
    /// For Roblox client bytecode, use 203
    #[clap(short, long, default_value_t = 1)]
    key: u8,
    #[clap(short, long)]
    recursive: bool,
    #[clap(short, long)]
    verbose: bool,
}

/// Dump bytecode instructions for debugging
pub fn dump_bytecode(bytecode: &[u8], encode_key: u8, func_name: Option<&str>) {
    use deserializer::bytecode::Bytecode;
    let chunk = deserializer::deserialize(bytecode, encode_key).unwrap();
    match chunk {
        Bytecode::Error(msg) => eprintln!("Error: {}", msg),
        Bytecode::Chunk(chunk) => {
            for (func_id, func) in chunk.functions.iter().enumerate() {
                // Get function name from string table
                let name = if func.function_name > 0 && func.function_name <= chunk.string_table.len() {
                    String::from_utf8_lossy(&chunk.string_table[func.function_name - 1]).to_string()
                } else {
                    format!("function_{}", func_id)
                };

                // Filter by function name if specified
                if let Some(filter) = func_name {
                    if !name.contains(filter) {
                        continue;
                    }
                }

                println!("\n=== Function {} ({}) ===", func_id, name);
                println!("Parameters: {}, Stack: {}, Upvalues: {}, Vararg: {}",
                    func.num_parameters, func.max_stack_size, func.num_upvalues, func.is_vararg);
                println!("Instructions:");

                for (pc, insn) in func.instructions.iter().enumerate() {
                    println!("  {:4}: {:?}", pc, insn);
                }

                println!("\nConstants:");
                for (i, constant) in func.constants.iter().enumerate() {
                    println!("  K{}: {:?}", i, constant);
                }

                if !func.local_debug_info.is_empty() {
                    println!("\nLocal Debug Info:");
                    for info in &func.local_debug_info {
                        let name = if info.name_index > 0 && info.name_index <= chunk.string_table.len() {
                            String::from_utf8_lossy(&chunk.string_table[info.name_index - 1]).to_string()
                        } else {
                            "(no name)".to_string()
                        };
                        println!("  R{}: \"{}\" (PC {}-{})", info.register, name, info.scope_start, info.scope_end);
                    }
                }
            }
        }
    }
}

pub fn decompile_bytecode(bytecode: &[u8], encode_key: u8) -> String {
    let chunk = deserializer::deserialize(bytecode, encode_key).unwrap();
    match chunk {
        Bytecode::Error(msg) => msg,
        Bytecode::Chunk(chunk) => {
            let mut lifted = Vec::new();
            let mut stack = vec![(Arc::<Mutex<ast::Function>>::default(), chunk.main)];
            while let Some((ast_func, func_id)) = stack.pop() {
                let (function, upvalues, child_functions) =
                    Lifter::lift(&chunk.functions, &chunk.string_table, func_id);
                lifted.push((ast_func, function, upvalues));
                stack.extend(child_functions.into_iter().map(|(a, f)| (a.0, f)));
            }

            let (main, ..) = lifted.first().unwrap().clone();
            let mut upvalues = lifted
                .into_iter()
                .map(|(ast_function, function, upvalues_in)| {
                    use std::{backtrace::Backtrace, cell::RefCell, fmt::Write, panic};

                    thread_local! {
                        static BACKTRACE: RefCell<Option<Backtrace>> = const { RefCell::new(None) };
                    }

                    let function_id = function.id;
                    let mut args = std::panic::AssertUnwindSafe(Some((
                        ast_function.clone(),
                        function,
                        upvalues_in,
                    )));

                    let result = panic::catch_unwind(move || {
                        let (ast_function, function, upvalues_in) = args.take().unwrap();
                        decompile_function(ast_function, function, upvalues_in)
                    });

                    match result {
                        Ok(r) => r,
                        Err(e) => {
                            let panic_information = match e.downcast::<String>() {
                                Ok(v) => *v,
                                Err(e) => match e.downcast::<&str>() {
                                    Ok(v) => v.to_string(),
                                    _ => "Unknown Source of Error".to_owned(),
                                },
                            };

                            let mut message = String::new();
                            writeln!(message, "failed to decompile").unwrap();
                            // writeln!(message, "function {} panicked at '{}'", function_id, panic_information).unwrap();
                            // if let Some(backtrace) = BACKTRACE.with(|b| b.borrow_mut().take()) {
                            //     write!(message, "stack backtrace:\n{}", backtrace).unwrap();
                            // }

                            ast_function.lock().body.extend(
                                message
                                    .trim_end()
                                    .split('\n')
                                    .map(|s| ast::Comment::new(s.to_string()).into()),
                            );
                            (ByAddress(ast_function), Vec::new())
                        }
                    }
                })
                .collect::<FxHashMap<_, _>>();

            let main = ByAddress(main);
            upvalues.remove(&main);
            let mut body = Arc::try_unwrap(main.0).unwrap().into_inner().body;
            link_upvalues(&mut body, &mut upvalues);
            cleanup_for_statements(&mut body);
            name_locals(&mut body, true);
            inline_temporaries(&mut body);
            body.to_string()
        }
    }
}

fn decompile_function(
    ast_function: Arc<Mutex<ast::Function>>,
    mut function: Function,
    upvalues_in: Vec<ast::RcLocal>,
) -> (ByAddress<Arc<Mutex<ast::Function>>>, Vec<ast::RcLocal>) {
    let (local_count, local_groups, upvalue_in_groups, upvalue_passed_groups) =
        cfg::ssa::construct(&mut function, &upvalues_in);
    // Collect upvalues coming IN (from parent) - these will be ignored for declaration
    let upvalue_in_locals: FxHashSet<ast::RcLocal> = upvalue_in_groups
        .iter()
        .flat_map(|(_, g)| g.iter().cloned())
        .collect();

    let upvalue_to_group = upvalue_in_groups
        .into_iter()
        .chain(
            upvalue_passed_groups
                .into_iter()
                .map(|m| (ast::RcLocal::default(), m)),
        )
        .flat_map(|(i, g)| g.into_iter().map(move |u| (u, i.clone())))
        .collect::<IndexMap<_, _>>();
    // TODO: do we even need this?
    let local_to_group = local_groups
        .into_iter()
        .enumerate()
        .flat_map(|(i, g)| g.into_iter().map(move |l| (l, i)))
        .collect::<FxHashMap<_, _>>();
    // TODO: REFACTOR: some way to write a macro that states
    // if cfg::ssa::inline results in change then structure_jumps, structure_compound_conditionals,
    // structure_for_loops and remove_unnecessary_params must run again.
    // if structure_compound_conditionals results in change then dominators and post dominators
    // must be recalculated.
    // etc.
    // the macro could also maybe generate an optimal ordering?
    let mut changed = true;
    let mut iteration = 0;
    // PERF: Cache dominators and only recalculate when CFG structure changes
    let mut dominators = simple_fast(function.graph(), function.entry().unwrap());
    let mut dominators_valid = true;

    while changed {
        iteration += 1;
        if iteration > 20 {
            // Safety limit - most functions converge within 10 iterations
            break;
        }
        changed = false;

        // Only recalculate dominators if CFG structure changed
        if !dominators_valid {
            dominators = simple_fast(function.graph(), function.entry().unwrap());
            dominators_valid = true;
        }

        if structure_jumps(&mut function, &dominators) {
            changed = true;
            dominators_valid = false;  // CFG changed, invalidate dominators
        }

        ssa::inline::inline(&mut function, &local_to_group, &upvalue_to_group);

        if structure_conditionals(&mut function) {
            changed = true;
            dominators_valid = false;  // CFG changed, invalidate dominators
        }

        let mut local_map = FxHashMap::default();
        // TODO: loop until returns false?
        if ssa::construct::remove_unnecessary_params(&mut function, &mut local_map) {
            changed = true;
        }
        ssa::construct::apply_local_map(&mut function, local_map);
    }
    // Build set of locals to ignore for declaration: upvalues_in, params, and all their SSA versions
    // This ensures that when a parameter is reassigned (creating a new SSA version), we don't
    // declare it as a new local - instead it should be assigned to the parameter directly.
    // We need to collect this BEFORE SSA destruction since upvalue_to_group is consumed.
    let mut locals_to_ignore: FxHashSet<ast::RcLocal> = upvalues_in.iter().cloned().collect();
    locals_to_ignore.extend(function.parameters.iter().cloned());

    // Add SSA versions of upvalues coming IN (from parent scope) - these should not be declared.
    // But DO NOT add upvalues being passed OUT to child closures - those are locals defined
    // in THIS function that need to be declared.
    for (ssa_version, _) in &upvalue_to_group {
        // Only ignore if this is an incoming upvalue, not a passed-out one
        if upvalue_in_locals.contains(ssa_version) {
            locals_to_ignore.insert(ssa_version.clone());
        }
    }

    // Also check local_to_group for SSA versions of parameters
    for (ssa_version, _) in &local_to_group {
        // Check if this SSA version maps back to a parameter
        for param in &function.parameters {
            if ssa_version == param {
                continue;  // Already in locals_to_ignore
            }
            // Check if they're in the same local group (meaning ssa_version is an SSA version of param)
            if let (Some(&version_group), Some(&param_group)) = (local_to_group.get(ssa_version), local_to_group.get(param)) {
                if version_group == param_group {
                    locals_to_ignore.insert(ssa_version.clone());
                }
            }
        }
    }

    // cfg::dot::render_to(&function, &mut std::io::stderr()).unwrap();
    ssa::Destructor::new(
        &mut function,
        upvalue_to_group,
        upvalues_in.iter().cloned().collect(),
        local_count,
    )
    .destruct();

    let params = std::mem::take(&mut function.parameters);
    let is_variadic = function.is_variadic;
    let block: Arc<Mutex<ast::Block>> = Arc::new(restructure::lift(function).into());

    LocalDeclarer::default().declare_locals(
        // TODO: why does block.clone() not work?
        Arc::clone(&block),
        &locals_to_ignore,
    );

    {
        let mut ast_function = ast_function.lock();
        ast_function.body = Arc::try_unwrap(block).unwrap().into_inner();
        ast_function.parameters = params;
        ast_function.is_variadic = is_variadic;
    }
    (ByAddress(ast_function), upvalues_in)
}

fn link_upvalues(
    body: &mut ast::Block,
    upvalues: &mut FxHashMap<ByAddress<Arc<Mutex<ast::Function>>>, Vec<ast::RcLocal>>,
) {
    for stat in &mut body.0 {
        stat.traverse_rvalues(&mut |rvalue| {
            if let ast::RValue::Closure(closure) = rvalue {
                let old_upvalues = &upvalues[&closure.function];
                let mut function = closure.function.lock();
                // TODO: inefficient, try constructing a map of all up -> new up first
                // and then call replace_locals on main body
                let mut local_map =
                    FxHashMap::with_capacity_and_hasher(old_upvalues.len(), Default::default());
                for (old, new) in
                    old_upvalues
                        .iter()
                        .zip(closure.upvalues.iter().map(|u| match u {
                            ast::Upvalue::Copy(l) | ast::Upvalue::Ref(l) => l,
                        }))
                {
                    // Propagate debug name from parent local to child upvalue if unnamed
                    if let Some(name) = old.name() {
                        new.set_name_if_unnamed(Some(name));
                    }
                    // println!("{} -> {}", old, new);
                    local_map.insert(old.clone(), new.clone());
                }
                link_upvalues(&mut function.body, upvalues);
                replace_locals(&mut function.body, &local_map);
            }
        });
        match stat {
            ast::Statement::If(r#if) => {
                link_upvalues(&mut r#if.then_block.lock(), upvalues);
                link_upvalues(&mut r#if.else_block.lock(), upvalues);
            }
            ast::Statement::While(r#while) => {
                link_upvalues(&mut r#while.block.lock(), upvalues);
            }
            ast::Statement::Repeat(repeat) => {
                link_upvalues(&mut repeat.block.lock(), upvalues);
            }
            ast::Statement::NumericFor(numeric_for) => {
                link_upvalues(&mut numeric_for.block.lock(), upvalues);
            }
            ast::Statement::GenericFor(generic_for) => {
                link_upvalues(&mut generic_for.block.lock(), upvalues);
            }
            _ => {}
        }
    }
}
