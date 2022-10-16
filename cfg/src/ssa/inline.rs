use crate::function::Function;
use ast::{LocalRw, SideEffects, Traverse};
use fxhash::{FxHashMap, FxHashSet};
use indexmap::IndexMap;
use itertools::Either;

fn inline_rvalue(
    statement: &mut ast::Statement,
    read: &ast::RcLocal,
    new_rvalue: &mut Option<ast::RValue>,
    new_rvalue_has_side_effects: bool,
) -> bool {
    statement
        .post_traverse_values(&mut |v| {
            if let Either::Right(rvalue) = v {
                if let ast::RValue::Local(rvalue_local) = rvalue && *rvalue_local == *read {
                    *rvalue = new_rvalue.take().unwrap();
                    // success!
                    return Some(true)
                }
                if new_rvalue_has_side_effects && rvalue.has_side_effects() {
                    // failure :(
                    return Some(false);
                }
            }

            // keep searching
            None
        })
        .unwrap_or(false)
}

// TODO: dont clone rvalues
// TODO: REFACTOR: move to ssa module?
// TODO: inline into block arguments
fn inline_rvalues(
    function: &mut Function,
    upvalue_to_group: &IndexMap<ast::RcLocal, usize>,
    local_usages: &mut FxHashMap<ast::RcLocal, usize>,
) {
    let node_indices = function.graph().node_indices().collect::<Vec<_>>();
    for node in node_indices {
        let mut locals_out = FxHashSet::default();
        locals_out.extend(
            function
                .edges(node)
                .into_iter()
                .flat_map(|e| e.weight().arguments.iter().map(|(_, a)| a.as_local().unwrap()))
                .cloned(),
        );
        let block = function.block_mut(node).unwrap();

        // TODO: rename values_read to locals_read
        let mut stat_to_values_read = Vec::with_capacity(block.len());
        for stat in &block.0 {
            stat_to_values_read.push(
                stat.values_read()
                    .into_iter()
                    .filter(|&l| {
                        local_usages[l] == 1
                            && !upvalue_to_group.contains_key(l)
                            // sadly we cant inline into block params
                            // TODO: but maybe we should be able to?
                            && !locals_out.contains(l)
                    })
                    .cloned()
                    .map(Some)
                    .collect::<Vec<_>>(),
            );
        }

        let mut index = 0;
        'w: while index < block.len() {
            let mut allow_side_effects = true;
            for stat_index in (0..index).rev() {
                if let ast::Statement::Assign(assign) = &block[stat_index]
                    && assign.left.len() == 1
                    && assign.right.len() == 1
                    && let ast::LValue::Local(local) = &assign.left[0]
                {
                    let new_rvalue = &assign.right[0];
                    let new_rvalue_has_side_effects = new_rvalue.has_side_effects();
                    let local = local.clone();
                    for read_opt in stat_to_values_read[index]
                        .iter_mut()
                        .filter(|l| l.is_some())
                    {
                        // TODO: filter?
                        let read = read_opt.as_ref().unwrap();
                        if read != &local {
                            continue;
                        };
                        if !new_rvalue_has_side_effects || allow_side_effects {
                            let mut new_rvalue = Some(
                                block[stat_index]
                                    .as_assign_mut()
                                    .unwrap()
                                    .right
                                    .pop()
                                    .unwrap(),
                            );
                            if inline_rvalue(
                                &mut block[index],
                                read,
                                &mut new_rvalue,
                                new_rvalue_has_side_effects,
                            ) {
                                assert!(new_rvalue.is_none());
                                // TODO: PERF: remove local_usages[l] == 1 in stat_to_values_read and use that
                                for local in block[stat_index].values_read() {
                                    let local_usage_count = local_usages.get_mut(local).unwrap();
                                    *local_usage_count = local_usage_count.saturating_sub(1);
                                }
                                // we dont need to update local usages because tracking usages for a local
                                // with no declarations serves no purpose
                                block[stat_index] = ast::Empty {}.into();
                                *read_opt = None;
                                continue 'w;
                            } else {
                                block[stat_index]
                                    .as_assign_mut()
                                    .unwrap()
                                    .right
                                    .push(new_rvalue.unwrap());
                            }
                        }
                    }
                }
                allow_side_effects &= !block[stat_index].has_side_effects();
            }
            index += 1;
        }
    }
}

pub fn inline(function: &mut Function, upvalue_to_group: &IndexMap<ast::RcLocal, usize>) {
    let mut local_usages = FxHashMap::default();
    for node in function.graph().node_indices() {
        let block = function.block(node).unwrap();
        for read in function.values_read(node) {
            *local_usages.entry(read.clone()).or_insert(0usize) += 1;
        }
    }

    let mut changed = true;
    while changed {
        changed = false;
        inline_rvalues(function, upvalue_to_group, &mut local_usages);

        // remove unused locals
        for (_, block) in function.blocks_mut() {
            for stat_index in 0..block.len() {
                if let ast::Statement::Assign(assign) = &block[stat_index]
                    && assign.left.len() == 1
                    && assign.right.len() == 1
                    && let ast::LValue::Local(local) = &assign.left[0]
                {
                    let rvalue = &assign.right[0];
                    let has_side_effects = rvalue.has_side_effects();
                    if !upvalue_to_group.contains_key(local) && local_usages.get(local).map_or(true, |&u| u == 0) {
                        if has_side_effects {
                            // TODO: PERF: dont clone
                            let new_stat = match rvalue {
                                ast::RValue::Call(call)
                                | ast::RValue::Select(ast::Select::Call(call)) => {
                                    Some(call.clone().into())
                                }
                                ast::RValue::MethodCall(method_call)
                                | ast::RValue::Select(ast::Select::MethodCall(method_call)) => {
                                    Some(method_call.clone().into())
                                }
                                _ => None,
                            };
                            if let Some(new_stat) = new_stat {
                                block[stat_index] = new_stat;
                                changed = true;
                            }
                        } else {
                            block[stat_index] = ast::Empty {}.into();
                            changed = true;
                        }
                    }
                }
            }
        }

        for (_, block) in function.blocks_mut() {
            // we check block.ast.len() elsewhere and do `i - ` here and elsewhere so we need to get rid of empty statements
            // TODO: fix ^
            block.retain(|s| s.as_empty().is_none());

            let mut i = 0;
            while i < block.len() {
                if let ast::Statement::Assign(assign) = &block[i]
                    && assign.left.len() == 1
                    && assign.right.len() == 1
                    && assign.right[0].as_table().is_some()
                    && let ast::LValue::Local(table_local) = &assign.left[0]
                {
                    let table_index = i;
                    let table_local = table_local.clone();
                    i += 1;
                    while i < block.len()
                        && let ast::Statement::Assign(field_assign) = &block[i]
                        && field_assign.left.len() == 1
                        && field_assign.right.len() == 1
                        && let ast::LValue::Index(ast::Index {
                            left: box ast::RValue::Local(local),
                            ..
                        }) = &field_assign.left[0]
                        && local == &table_local
                    {
                        let field_assign = std::mem::replace(&mut block[i], ast::Empty {}.into()).into_assign().unwrap();
                        block[table_index].as_assign_mut().unwrap().right[0].as_table_mut().unwrap().0.push((Some(Box::into_inner(field_assign.left.into_iter().next().unwrap().into_index().unwrap().right)), field_assign.right.into_iter().next().unwrap()));
                        changed = true;
                        i += 1;
                    }
                } else {
                    i += 1;
                }
            }

            // if the first statement is a setlist, we cant inline it anyway
            for i in 1..block.len() {
                if let ast::Statement::SetList(setlist) = &block[i] {
                    let table_local = setlist.table.clone();
                    if let Some(assign) = block[i - 1].as_assign_mut()
                        && assign.left == [table_local.into()]
                    {
                        let setlist =
                            std::mem::replace(block.get_mut(i).unwrap(), ast::Empty {}.into())
                                .into_set_list()
                                .unwrap();
                        *local_usages.get_mut(&setlist.table).unwrap() -= 1;
                        let assign = block.get_mut(i - 1).unwrap().as_assign_mut().unwrap();
                        let table = assign.right[0].as_table_mut().unwrap();
                        assert!(
                            table.0.iter().filter(|(k, _)| k.is_none()).count()
                                == setlist.index - 1
                        );
                        for value in setlist.values {
                            table.0.push((None, value));
                        }
                        // table already has tail?
                        assert!(!table.0.last().map_or(false, |(k, v)| k.is_none()
                            && matches!(
                                v,
                                ast::RValue::VarArg(_)
                                    | ast::RValue::Call(_)
                                    | ast::RValue::MethodCall(_)
                            )));
                        if let Some(tail) = setlist.tail {
                            table.0.push((None, tail));
                        }
                        changed = true;
                    }
                    // todo: only inline in changed blocks
                    //cfg::dot::render_to(function, &mut std::io::stdout());
                    //break 'outer;
                }
            }
        }
    }
    // we check block.ast.len() elsewhere and do `i - ` here and elsewhere so we need to get rid of empty statements
    // TODO: fix ^
    for (_, block) in function.blocks_mut() {
        block.retain(|s| s.as_empty().is_none());
    }
}
