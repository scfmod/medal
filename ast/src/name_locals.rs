use rustc_hash::FxHashSet;
use triomphe::Arc;

use crate::{Block, RValue, RcLocal, Statement, Traverse, Upvalue};

struct Namer {
    rename: bool,
    counter: usize,
    upvalues: FxHashSet<RcLocal>,
}

impl Namer {
    fn name_local(&mut self, prefix: &str, local: &RcLocal) {
        let mut lock = local.0 .0.lock();
        // Only name if no name exists - preserve debug names from bytecode
        if lock.0.is_none() {
            // TODO: hacky and slow
            if Arc::count(&local.0 .0) == 1 {
                lock.0 = Some("_".to_string());
            } else {
                let prefix = prefix.to_string()
                    + if self.upvalues.contains(local) {
                        "_u_"
                    } else {
                        ""
                    };
                lock.0 = Some(format!("{}{}_", prefix, self.counter));
                self.counter += 1;
            }
        } else if self.rename {
            // If rename is requested and local has a name but is an upvalue,
            // we may want to add the _u_ marker. For now, preserve the name.
        }
    }

    fn name_locals(&mut self, block: &mut Block) {
        for statement in &mut block.0 {
            // TODO: traverse_rvalues
            statement.post_traverse_values(&mut |value| -> Option<()> {
                if let itertools::Either::Right(RValue::Closure(closure)) = value {
                    let mut function = closure.function.lock();
                    for param in &function.parameters {
                        self.name_local("p", param);
                    }
                    self.name_locals(&mut function.body);
                };
                None
            });
            match statement {
                Statement::Assign(assign) if assign.prefix => {
                    for lvalue in &assign.left {
                        self.name_local("v", lvalue.as_local().unwrap());
                    }
                }
                Statement::If(r#if) => {
                    self.name_locals(&mut r#if.then_block.lock());
                    self.name_locals(&mut r#if.else_block.lock());
                }
                Statement::While(r#while) => {
                    self.name_locals(&mut r#while.block.lock());
                }
                Statement::Repeat(repeat) => {
                    self.name_locals(&mut repeat.block.lock());
                }
                Statement::NumericFor(numeric_for) => {
                    self.name_local("v", &numeric_for.counter);
                    self.name_locals(&mut numeric_for.block.lock());
                }
                Statement::GenericFor(generic_for) => {
                    for res_local in &generic_for.res_locals {
                        self.name_local("v", res_local);
                    }
                    self.name_locals(&mut generic_for.block.lock());
                }
                _ => {}
            }
        }
    }

    // TODO: does this need to be mut?
    fn find_upvalues(&mut self, block: &mut Block) {
        for statement in &mut block.0 {
            // TODO: traverse_values
            // TODO: doesnt need to be mut
            statement.post_traverse_values(&mut |value| -> Option<()> {
                if let itertools::Either::Right(RValue::Closure(closure)) = value {
                    self.upvalues.extend(
                        closure
                            .upvalues
                            .iter()
                            .map(|u| match u {
                                Upvalue::Copy(l) | Upvalue::Ref(l) => l,
                            })
                            .cloned(),
                    );
                    self.find_upvalues(&mut closure.function.lock().body);
                };
                None
            });
            match statement {
                Statement::If(r#if) => {
                    self.find_upvalues(&mut r#if.then_block.lock());
                    self.find_upvalues(&mut r#if.else_block.lock());
                }
                Statement::While(r#while) => {
                    self.find_upvalues(&mut r#while.block.lock());
                }
                Statement::Repeat(repeat) => {
                    self.find_upvalues(&mut repeat.block.lock());
                }
                Statement::NumericFor(numeric_for) => {
                    self.find_upvalues(&mut numeric_for.block.lock());
                }
                Statement::GenericFor(generic_for) => {
                    self.find_upvalues(&mut generic_for.block.lock());
                }
                _ => {}
            }
        }
    }
}

pub fn name_locals(block: &mut Block, rename: bool) {
    let mut namer = Namer {
        rename,
        counter: 1,
        upvalues: FxHashSet::default(),
    };
    namer.find_upvalues(block);
    namer.name_locals(block);
}
