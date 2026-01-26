use std::collections::BTreeMap;

use array_tool::vec::Intersect;
use by_address::ByAddress;
use indexmap::{IndexMap, IndexSet};
use itertools::Itertools;
use parking_lot::Mutex;
use petgraph::{
    algo::dominators::simple_fast,
    prelude::{DiGraph, NodeIndex},
    Direction,
};
use rustc_hash::{FxHashMap, FxHashSet};
use triomphe::Arc;

use crate::{Assign, Block, LocalRw, RValue, RcLocal, Statement, Traverse, Upvalue};

#[derive(Default)]
pub struct LocalDeclarer {
    block_to_node: FxHashMap<ByAddress<Arc<Mutex<Block>>>, NodeIndex>,
    graph: DiGraph<(Option<Arc<Mutex<Block>>>, usize), ()>,
    local_usages: IndexMap<RcLocal, FxHashMap<NodeIndex, usize>>,
    declarations: FxHashMap<ByAddress<Arc<Mutex<Block>>>, BTreeMap<usize, IndexSet<RcLocal>>>,
}

impl LocalDeclarer {
    /// Recursively collect upvalue locals from closures within an rvalue.
    /// This handles nested closures and closures inside tables, calls, etc.
    fn collect_closure_upvalues(
        rvalue: &RValue,
        local_usages: &mut IndexMap<RcLocal, FxHashMap<NodeIndex, usize>>,
        node: NodeIndex,
        stat_index: usize,
    ) {
        match rvalue {
            RValue::Closure(closure) => {
                for upvalue in &closure.upvalues {
                    let local = match upvalue {
                        Upvalue::Copy(l) | Upvalue::Ref(l) => l,
                    };
                    local_usages
                        .entry(local.clone())
                        .or_default()
                        .entry(node)
                        .or_insert(stat_index);
                }
            }
            _ => {}
        }
        // Recursively check nested rvalues (e.g., closures inside tables)
        for nested in rvalue.rvalues() {
            Self::collect_closure_upvalues(nested, local_usages, node, stat_index);
        }
    }

    fn visit(&mut self, block: Arc<Mutex<Block>>, stat_index: usize) -> NodeIndex {
        let node = self.graph.add_node((Some(block.clone()), stat_index));
        self.block_to_node.insert(block.clone().into(), node);
        for (stat_index, stat) in block.lock().iter().enumerate() {
            // for loops already declare their own locals :)
            if !matches!(stat, Statement::GenericFor(_) | Statement::NumericFor(_)) {
                // Track locals that are written to
                for local in stat.values_written() {
                    self.local_usages
                        .entry(local.clone())
                        .or_default()
                        .entry(node)
                        .or_insert(stat_index);
                }
                // Also track locals that are captured as upvalues in closures.
                // These locals need to be declared even if they're never directly
                // written to in this scope (they may only be initialized in the
                // outer scope and then captured).
                for rvalue in stat.rvalues() {
                    Self::collect_closure_upvalues(rvalue, &mut self.local_usages, node, stat_index);
                }
            }
            match stat {
                Statement::If(r#if) => {
                    let if_node = self.graph.add_node((None, stat_index));
                    self.graph.add_edge(node, if_node, ());
                    let then_node = self.visit(r#if.then_block.clone(), stat_index);
                    self.graph.add_edge(if_node, then_node, ());
                    let else_node = self.visit(r#if.else_block.clone(), stat_index);
                    self.graph.add_edge(if_node, else_node, ());
                }
                Statement::While(r#while) => {
                    let child = self.visit(r#while.block.clone(), stat_index);
                    self.graph.add_edge(node, child, ());
                }
                Statement::Repeat(repeat) => {
                    let child = self.visit(r#repeat.block.clone(), stat_index);
                    self.graph.add_edge(node, child, ());
                }
                Statement::NumericFor(numeric_for) => {
                    let child = self.visit(r#numeric_for.block.clone(), stat_index);
                    self.graph.add_edge(node, child, ());
                }
                Statement::GenericFor(generic_for) => {
                    let child = self.visit(r#generic_for.block.clone(), stat_index);
                    self.graph.add_edge(node, child, ());
                }
                _ => {}
            }
        }
        node
    }

    pub fn declare_locals(
        mut self,
        root_block: Arc<Mutex<Block>>,
        locals_to_ignore: &FxHashSet<RcLocal>,
    ) {
        let root_node = self.visit(root_block, 0);
        let dominators = simple_fast(&self.graph, root_node);
        for (local, usages) in self.local_usages {
            if locals_to_ignore.contains(&local) {
                continue;
            }
            let (mut node, mut first_stat_index) = if usages.len() == 1 {
                usages.into_iter().next().unwrap()
            } else {
                let node_dominators = usages
                    .keys()
                    .map(|&n| dominators.dominators(n).unwrap().collect_vec())
                    .collect_vec();
                let mut dom_iter = node_dominators.iter().cloned();
                let mut common_dominators = dom_iter.next().unwrap();
                for node_dominators in dom_iter {
                    common_dominators = common_dominators.intersect(node_dominators);
                }
                let common_dominator = common_dominators[0];
                if let Some((_, first_stat_index)) =
                    usages.into_iter().find(|&(n, _)| n == common_dominator)
                {
                    (common_dominator, first_stat_index)
                } else {
                    // find the left-most dominated node
                    let mut first_stat_index = None;
                    for child in self
                        .graph
                        .neighbors_directed(common_dominator, Direction::Outgoing)
                    {
                        for node_dominators in &node_dominators {
                            if node_dominators.contains(&child) {
                                first_stat_index = Some(self.graph.node_weight(child).unwrap().1);
                            }
                        }
                    }
                    (common_dominator, first_stat_index.unwrap())
                }
            };
            while let (block, parent_stat_index) = self.graph.node_weight(node).unwrap()
                && block.is_none()
            {
                let parent = self
                    .graph
                    .neighbors_directed(node, Direction::Incoming)
                    .exactly_one()
                    .unwrap();
                (node, first_stat_index) = (parent, *parent_stat_index);
            }
            let block = self
                .graph
                .node_weight(node)
                .unwrap()
                .0
                .as_ref()
                .unwrap()
                .clone();
            self.declarations
                .entry(block.into())
                .or_default()
                .entry(first_stat_index)
                .or_default()
                .insert(local);
        }

        for (ByAddress(block), declarations) in self.declarations {
            let mut block = block.lock();
            for (stat_index, mut locals) in declarations.into_iter().rev() {
                match &mut block[stat_index] {
                    Statement::Assign(assign)
                        if assign
                            .left
                            .iter()
                            .all(|l| l.as_local().is_some_and(|l| locals.contains(l))) =>
                    {
                        locals.retain(|l| {
                            !assign
                                .left
                                .iter()
                                .map(|l| l.as_local().unwrap())
                                .contains(l)
                        });
                        assign.prefix = true;
                    }
                    _ => {}
                }
                if !locals.is_empty() {
                    let mut declaration =
                        Assign::new(locals.into_iter().map(|l| l.into()).collect_vec(), vec![]);
                    declaration.prefix = true;
                    block.insert(stat_index, declaration.into());
                }
            }
        }
    }
}
