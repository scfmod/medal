use rustc_hash::{FxHashMap, FxHashSet};

use crate::{Block, LValue, LocalRw, RValue, RcLocal, Select, SideEffects, Statement, Traverse};

/// Maximum recursion depth to prevent stack overflow
const MAX_DEPTH: usize = 100;

/// Inlines unnamed single-use temporaries back into expressions.
///
/// This pass identifies locals that:
/// 1. Have no debug name (were compiler-generated temporaries)
/// 2. Are assigned exactly once
/// 3. Are used exactly once
///
/// For side-effect-free RValues (like field access), inlining can happen
/// across multiple statements. For calls, inlining only happens into the
/// immediately following statement to preserve execution order.
///
/// Additionally, this pass inlines iterator expressions into generic for-loop
/// headers. For example:
///   local v3_, v4_, v5_ = pairs(x)
///   for k, v in v3_, v4_, v5_ do
/// Becomes:
///   for k, v in pairs(x) do
pub fn inline_temporaries(block: &mut Block) {
    let mut inliner = Inliner::default();
    inliner.inline_block(block, 0, &FxHashSet::default());
    // Remove dead stores (self-assignments, round-trip assignments) after inlining
    remove_dead_stores(block);
    // Remove dead locals (assigned but never read)
    remove_dead_locals(block);
}

#[derive(Default)]
struct Inliner {
    // Map from local to RValue for deferred inlining (side-effect-free only)
    deferred_inlines: FxHashMap<RcLocal, RValue>,
    // Locals that should be inlined into the immediate next statement only
    immediate_inlines: FxHashMap<RcLocal, RValue>,
}

impl Inliner {
    /// `protected_locals` contains locals from outer scopes that should never be inlined
    /// (e.g., variables used in outer while/repeat conditions)
    fn inline_block(&mut self, block: &mut Block, depth: usize, protected_locals: &FxHashSet<RcLocal>) {
        if depth > MAX_DEPTH {
            return;
        }

        // First: inline iterator expressions into generic for-loop headers
        // This transforms: `local v1, v2, v3 = pairs(x); for k, v in v1, v2, v3 do`
        // Into: `for k, v in pairs(x) do`
        self.inline_iterator_into_for_loops(block);

        // Simplify boolean ternary patterns like:
        // local v3_; if x == nil then v3_ = false else v3_ = true end
        // Into: local v3_ = x ~= nil
        simplify_boolean_ternaries(block);

        // Simplify and-chain patterns like:
        // local v = expr1; if v then v = expr2 end
        // Into: local v = expr1 and expr2
        simplify_and_chains(block);

        // Collapse multi-return patterns like:
        // local v1, v2 = call(); target1 = v1; target2 = v2
        // Into: target1, target2 = call()
        collapse_multi_return_assignments(block);

        // Second pass: collect all single-use unnamed locals and their definitions
        let inline_candidates = self.find_inline_candidates(block, protected_locals);

        // Third pass: apply inlines and remove definition statements
        let mut i = 0;
        while i < block.len() {
            // Apply any pending immediate inlines
            self.apply_immediate_inlines(&mut block[i]);

            // Apply deferred inlines (side-effect-free)
            self.apply_deferred_inlines(&mut block[i], &inline_candidates);

            // Check if this statement defines an inline candidate
            if let Some((local, rvalue)) = self.extract_inline_candidate(&block[i]) {
                // Use name-based comparison for SSA versions
                let is_candidate = inline_candidates.contains(&local)
                    || inline_candidates.iter().any(|c| c.name().is_some() && c.name() == local.name());

                if is_candidate {
                    if has_side_effects(&rvalue) {
                        // Must inline carefully for side-effect RValues.
                        // We can inline if:
                        // 1. The next statement(s) don't have side effects (they're just local assignments)
                        // 2. We find a statement that uses this local exactly once
                        // 3. No intervening statement writes to any local we read
                        if let Some(_use_idx) = self.find_safe_inline_target(&local, &rvalue, block, i) {
                            // Store for later inlining when we reach the target statement
                            self.immediate_inlines.insert(local.clone(), rvalue);
                            block.0.remove(i);
                            continue;
                        }
                    } else {
                        // Can defer inlining - add to deferred map
                        self.deferred_inlines.insert(local.clone(), rvalue);
                        block.0.remove(i);
                        continue;
                    }
                }
            }

            // Note: we DON'T clear immediate_inlines or deferred_inlines here anymore -
            // they persist until applied to a later statement in this block.

            // Recurse into nested blocks and closures, passing down protected locals
            // Save and restore both maps so nested processing doesn't affect outer scope
            let saved_immediate = std::mem::take(&mut self.immediate_inlines);
            let saved_deferred = std::mem::take(&mut self.deferred_inlines);
            self.recurse_into_statement(&mut block[i], depth, protected_locals);
            self.immediate_inlines = saved_immediate;
            self.deferred_inlines = saved_deferred;

            i += 1;
        }

        self.immediate_inlines.clear();
        self.deferred_inlines.clear();

        // Simplify __set_list patterns like:
        // local v35_ = {}; __set_list(v35_, 1, {a, b, c}); return v35_
        // Into: return {a, b, c}
        simplify_set_list_patterns(block);

        // Coalesce reassignment patterns like:
        // local year = match(...); local v12_ = tonumber(year)
        // Into: local year = match(...); year = tonumber(year)
        // (where year is only used once, in the tonumber call)
        coalesce_reassignments(block);

        // Simplify write-only locals to underscores:
        // local v28_, v29_; v28_, rotY, v29_ = getWorldRotation(...)
        // Into: _, rotY, _ = getWorldRotation(...)
        simplify_write_only_locals(block);

        // Fix broken swap patterns like:
        // local temp = a; a = b; temp = b; b = a
        // Into: a, b = b, a
        // This runs LAST to avoid interference from inlining passes
        fix_broken_swap_patterns(block);
    }

    /// Inline expressions into for-loop headers.
    ///
    /// For generic for loops, transforms patterns like:
    ///   local v1_, v2_, v3_ = pairs(x)
    ///   for k, v in v1_, v2_, v3_ do
    /// Into:
    ///   for k, v in pairs(x) do
    ///
    /// For numeric for loops, transforms patterns like:
    ///   local v2_ = select("#", ...)
    ///   for i = 1, v2_ do
    /// Into:
    ///   for i = 1, select("#", ...) do
    ///
    /// This handles both unnamed temporaries and cases where names have leaked
    /// to the iterator state variables (which is a separate decompilation artifact).
    fn inline_iterator_into_for_loops(&self, block: &mut Block) {
        // First handle numeric for loops
        self.inline_into_numeric_for_loops(block);
        // Then handle generic for loops
        // Look for patterns like:
        //   local v1, v2, v3 = pairs(x)
        //   [intervening safe statements]
        //   for k, v in v1, v2, v3 do
        let mut i = 0;
        while i < block.len() {
            // Check if statement i is an assignment with iterator call
            let (is_iterator_assign, assign_locals) = if let Statement::Assign(assign) = &block[i] {
                let is_iterator_call = assign.right.len() == 1
                    && matches!(&assign.right[0],
                        RValue::Call(_) | RValue::Select(Select::Call(_)) |
                        RValue::MethodCall(_) | RValue::Select(Select::MethodCall(_)));
                if is_iterator_call && assign.left.len() >= 1 {
                    // Collect the local names being assigned
                    let locals: Vec<_> = assign.left.iter()
                        .filter_map(|lv| lv.as_local())
                        .filter_map(|l| l.name())
                        .collect();
                    if locals.len() == assign.left.len() {
                        (true, locals)
                    } else {
                        (false, vec![])
                    }
                } else {
                    (false, vec![])
                }
            } else {
                (false, vec![])
            };

            if !is_iterator_assign {
                i += 1;
                continue;
            }

            // Look ahead for a GenericFor that uses these locals
            let mut for_idx = None;
            for j in (i + 1)..block.len() {
                if let Statement::GenericFor(generic_for) = &block[j] {
                    // Check if this for-loop uses exactly our assigned locals
                    if generic_for.right.len() == assign_locals.len() {
                        let matches = generic_for.right.iter().zip(&assign_locals).all(|(rv, expected_name)| {
                            if let RValue::Local(for_local) = rv {
                                for_local.name().as_ref() == Some(expected_name)
                            } else {
                                false
                            }
                        });
                        if matches {
                            for_idx = Some(j);
                            break;
                        }
                    }
                }
                // Check if intervening statement uses our locals (would make reordering unsafe)
                // Only allow simple local assignments that don't reference our iterator locals
                if let Statement::Assign(intervening) = &block[j] {
                    let reads_our_locals = intervening.right.iter().any(|rv| {
                        rv.values_read().iter().any(|l| {
                            assign_locals.iter().any(|n| l.name().as_ref() == Some(n))
                        })
                    });
                    if reads_our_locals {
                        break; // Can't move past this
                    }
                    // Continue looking
                } else {
                    // Non-assignment statement - not safe to look past
                    break;
                }
            }

            if let Some(for_idx) = for_idx {
                // Found the for-loop! Move the iterator call to just before it
                if for_idx > i + 1 {
                    // Need to move the statement - remove and reinsert
                    let stmt = block.0.remove(i);
                    block.0.insert(for_idx - 1, stmt);
                    // Don't increment i since we removed at i
                    continue;
                } else {
                    // Already adjacent, do the inline
                    let Statement::Assign(assign) = block.0.remove(i) else {
                        unreachable!()
                    };
                    if let Statement::GenericFor(generic_for) = &mut block[i] {
                        generic_for.right = assign.right;
                    }
                    continue;
                }
            }

            i += 1;
        }

        // Second pass: do the actual inlining for adjacent pairs
        let mut i = 0;
        while i + 1 < block.len() {
            // Check if statement i is an assignment and i+1 is a GenericFor
            let can_inline = if let (
                Statement::Assign(assign),
                Statement::GenericFor(generic_for),
            ) = (&block[i], &block[i + 1])
            {
                // The assignment must have a call as its RHS (like pairs() or ipairs())
                let is_iterator_call = assign.right.len() == 1
                    && matches!(&assign.right[0],
                        RValue::Call(_) | RValue::Select(Select::Call(_)) |
                        RValue::MethodCall(_) | RValue::Select(Select::MethodCall(_)));

                if is_iterator_call && assign.left.len() >= 1 && generic_for.right.len() == assign.left.len() {
                    // Check that all for-loop iterators are locals matching the assignment
                    let mut matches = true;
                    for (j, rv) in generic_for.right.iter().enumerate() {
                        if let RValue::Local(for_local) = rv {
                            if let LValue::Local(assign_local) = &assign.left[j] {
                                // Compare by identity OR by name for SSA versions
                                let same = for_local == assign_local
                                    || (for_local.name().is_some() && for_local.name() == assign_local.name());
                                if !same {
                                    matches = false;
                                    break;
                                }
                            } else {
                                matches = false;
                                break;
                            }
                        } else {
                            matches = false;
                            break;
                        }
                    }
                    matches
                } else {
                    false
                }
            } else {
                false
            };

            if can_inline {
                // Extract the RHS from the assignment
                let Statement::Assign(assign) = block.0.remove(i) else {
                    unreachable!()
                };

                // Update the GenericFor to use the assignment's RHS directly
                if let Statement::GenericFor(generic_for) = &mut block[i] {
                    generic_for.right = assign.right;
                }
                // Don't increment i since we removed a statement
            } else {
                i += 1;
            }
        }
    }

    /// Inline expressions into numeric for-loop headers (initial, limit, step).
    ///
    /// Transforms patterns like:
    ///   local v2_ = select("#", ...)
    ///   for i = 1, v2_ do
    /// Into:
    ///   for i = 1, select("#", ...) do
    ///
    /// Also handles the special case where the loop counter is initialized separately:
    ///   local i = 1
    ///   for i = i, limit do
    /// Into:
    ///   for i = 1, limit do
    ///
    /// This searches backwards from the for-loop to find assignments to locals
    /// used in the for-loop header.
    fn inline_into_numeric_for_loops(&self, block: &mut Block) {
        let mut for_idx = 0;
        while for_idx < block.len() {
            let Statement::NumericFor(numeric_for) = &block[for_idx] else {
                for_idx += 1;
                continue;
            };

            // Collect locals used in the for-loop header
            let initial_local = if let RValue::Local(l) = &numeric_for.initial { Some(l.clone()) } else { None };
            let limit_local = if let RValue::Local(l) = &numeric_for.limit { Some(l.clone()) } else { None };
            let step_local = if let RValue::Local(l) = &numeric_for.step { Some(l.clone()) } else { None };

            // Search backwards for assignments that can be inlined
            let mut assign_idx = for_idx;
            while assign_idx > 0 {
                assign_idx -= 1;

                let Statement::Assign(assign) = &block[assign_idx] else {
                    continue;
                };

                if assign.left.len() != 1 || assign.right.len() != 1 {
                    continue;
                }

                let LValue::Local(assign_local) = &assign.left[0] else {
                    continue;
                };

                // Check if this assignment is to a local used in the for-loop header
                let same_local_or_name = |l: &RcLocal| {
                    l == assign_local ||
                    (assign_local.name().is_some() && l.name() == assign_local.name())
                };

                let matches_initial = initial_local.as_ref().map_or(false, |l| same_local_or_name(l));
                let matches_limit = limit_local.as_ref().map_or(false, |l| same_local_or_name(l));
                let matches_step = step_local.as_ref().map_or(false, |l| same_local_or_name(l));

                // Count how many places this local is used in the for header
                let use_count = [matches_initial, matches_limit, matches_step]
                    .iter()
                    .filter(|&&b| b)
                    .count();

                if use_count != 1 {
                    continue;
                }

                // For unnamed temporaries, always inline
                // For named locals, only inline if used in initial position
                let is_unnamed = assign_local.name()
                    .map_or(true, |n| is_unnamed_temporary(&n));

                let should_inline = if is_unnamed {
                    true
                } else {
                    // Named local: only inline into initial position
                    matches_initial
                };

                if !should_inline {
                    continue;
                }

                // Extract the assignment
                let Statement::Assign(assign) = block.0.remove(assign_idx) else {
                    unreachable!()
                };
                let rvalue = assign.right.into_iter().next().unwrap();

                // Update for_idx since we removed a statement before it
                for_idx -= 1;

                // Update the NumericFor to use the assignment's RHS directly
                if let Statement::NumericFor(numeric_for) = &mut block[for_idx] {
                    if matches_initial {
                        numeric_for.initial = rvalue;
                    } else if matches_limit {
                        numeric_for.limit = rvalue;
                    } else if matches_step {
                        numeric_for.step = rvalue;
                    }
                }

                // Continue searching for more assignments (may need to inline multiple)
                // Note: assign_idx is already decremented by the loop
            }

            for_idx += 1;
        }
    }

    /// Find all locals that are assigned once and used exactly once
    ///
    /// We consider both unnamed temporaries (like v1_) and named locals that are
    /// assigned side-effect-free values (like field access). Named locals assigned
    /// to calls are NOT inlined since the original code likely wanted that variable.
    ///
    /// `protected_locals` contains locals from outer scopes that should never be inlined
    fn find_inline_candidates(&self, block: &Block, protected_locals: &FxHashSet<RcLocal>) -> FxHashSet<RcLocal> {
        // Track (local, rvalue) for definitions
        let mut definitions: FxHashMap<RcLocal, (usize, Option<&RValue>)> = FxHashMap::default();
        // Track uses at the current block level (not inside nested blocks)
        let mut current_level_uses: FxHashMap<RcLocal, usize> = FxHashMap::default();
        // Track locals used in while/repeat conditions - these should never be inlined
        // because the variable may be modified inside the loop body (which we don't track)
        let mut loop_condition_locals: FxHashSet<RcLocal> = FxHashSet::default();
        // Track locals WRITTEN in nested blocks - these must never be inlined
        // because the value changes inside the nested block
        let mut nested_block_writes: FxHashSet<RcLocal> = FxHashSet::default();
        // Track locals READ in nested blocks - these are uses but can't be inlined via deferred_inlines
        let mut nested_block_reads: FxHashSet<RcLocal> = FxHashSet::default();

        for statement in &block.0 {
            // Count definitions and track what they're assigned to
            if let Statement::Assign(assign) = statement {
                if assign.left.len() == 1 && assign.right.len() == 1 {
                    if let LValue::Local(local) = &assign.left[0] {
                        let entry = definitions.entry(local.clone()).or_insert((0, None));
                        entry.0 += 1;
                        entry.1 = Some(&assign.right[0]);
                    }
                }
            }

            // Track locals used in while/repeat conditions
            match statement {
                Statement::While(w) => {
                    for local in w.condition.values_read() {
                        loop_condition_locals.insert(local.clone());
                    }
                    // Track reads and writes separately
                    Self::collect_reads_in_block_set(&w.block.lock(), &mut nested_block_reads);
                    Self::collect_writes_in_block(&w.block.lock(), &mut nested_block_writes);
                }
                Statement::Repeat(r) => {
                    for local in r.condition.values_read() {
                        loop_condition_locals.insert(local.clone());
                    }
                    Self::collect_reads_in_block_set(&r.block.lock(), &mut nested_block_reads);
                    Self::collect_writes_in_block(&r.block.lock(), &mut nested_block_writes);
                }
                Statement::If(i) => {
                    // Track reads and writes in if/else bodies
                    Self::collect_reads_in_block_set(&i.then_block.lock(), &mut nested_block_reads);
                    Self::collect_reads_in_block_set(&i.else_block.lock(), &mut nested_block_reads);
                    Self::collect_writes_in_block(&i.then_block.lock(), &mut nested_block_writes);
                    Self::collect_writes_in_block(&i.else_block.lock(), &mut nested_block_writes);
                }
                Statement::NumericFor(nf) => {
                    Self::collect_reads_in_block_set(&nf.block.lock(), &mut nested_block_reads);
                    Self::collect_writes_in_block(&nf.block.lock(), &mut nested_block_writes);
                }
                Statement::GenericFor(gf) => {
                    Self::collect_reads_in_block_set(&gf.block.lock(), &mut nested_block_reads);
                    Self::collect_writes_in_block(&gf.block.lock(), &mut nested_block_writes);
                }
                _ => {}
            }

            // Count uses at the current block level
            for local in statement.values_read() {
                *current_level_uses.entry(local.clone()).or_default() += 1;
            }
        }

        // Return locals that are defined once and used once, with appropriate conditions:
        // - Unnamed temporaries: always inline
        // - Named locals: only inline if assigned side-effect-free value (like field access)
        // - Never inline locals used in while/repeat conditions
        // - Never inline locals that are ONLY used in nested blocks (deferred inlines won't work)
        definitions
            .into_iter()
            .filter(|(local, (def_count, rvalue))| {
                // Use name-based lookup for current_level_uses to handle SSA versions
                let current_uses = current_level_uses.get(local).copied()
                    .or_else(|| {
                        local.name().and_then(|name| {
                            current_level_uses.iter()
                                .find(|(l, _)| l.name().as_ref() == Some(&name))
                                .map(|(_, &count)| count)
                        })
                    })
                    .unwrap_or(0);
                // Check if local is read in nested blocks (by identity or by name for SSA versions)
                let nested_uses = if nested_block_reads.contains(local) ||
                    nested_block_reads.iter().any(|l| l.name().is_some() && l.name() == local.name()) {
                    1
                } else {
                    0
                };
                let total_uses = current_uses + nested_uses;

                if *def_count != 1 || total_uses != 1 {
                    return false;
                }

                // Never inline locals used in loop conditions (current block)
                if loop_condition_locals.contains(local) {
                    return false;
                }

                // Never inline locals protected from outer scopes (e.g., outer loop conditions)
                if protected_locals.contains(local) {
                    return false;
                }

                // Never inline locals that are WRITTEN in nested blocks (if/while/for bodies)
                // because the value changes inside the nested block
                // Check by identity or by name for SSA versions
                if nested_block_writes.contains(local) ||
                    nested_block_writes.iter().any(|l| l.name().is_some() && l.name() == local.name()) {
                    return false;
                }

                // Never inline locals that are ONLY read in nested blocks
                // because deferred_inlines gets cleared when we recurse into nested blocks
                if current_uses == 0 && nested_uses > 0 {
                    return false;
                }

                // Unnamed temporaries: no name, or name matches v\d+_ pattern
                let is_unnamed = local.name().map_or(true, |n| is_unnamed_temporary(&n));
                if is_unnamed {
                    return true;
                }

                // For named locals, only inline side-effect-free values
                // This handles cases like `local i = string.format` followed by `i(...)`
                if let Some(rv) = rvalue {
                    !has_side_effects(rv)
                } else {
                    false
                }
            })
            .map(|(local, _)| local)
            .collect()
    }

    /// Collect all locals READ in a block (recursively) into a set.
    fn collect_reads_in_block_set(block: &Block, reads: &mut FxHashSet<RcLocal>) {
        for statement in &block.0 {
            for local in statement.values_read() {
                reads.insert(local.clone());
            }
            // Recurse into nested blocks
            match statement {
                Statement::If(i) => {
                    Self::collect_reads_in_block_set(&i.then_block.lock(), reads);
                    Self::collect_reads_in_block_set(&i.else_block.lock(), reads);
                }
                Statement::While(w) => {
                    Self::collect_reads_in_block_set(&w.block.lock(), reads);
                }
                Statement::Repeat(r) => {
                    Self::collect_reads_in_block_set(&r.block.lock(), reads);
                }
                Statement::NumericFor(nf) => {
                    Self::collect_reads_in_block_set(&nf.block.lock(), reads);
                }
                Statement::GenericFor(gf) => {
                    Self::collect_reads_in_block_set(&gf.block.lock(), reads);
                }
                _ => {}
            }
        }
    }

    /// Collect all locals WRITTEN in a block (recursively).
    fn collect_writes_in_block(block: &Block, writes: &mut FxHashSet<RcLocal>) {
        for statement in &block.0 {
            for local in statement.values_written() {
                writes.insert(local.clone());
            }
            // Recurse into nested blocks
            match statement {
                Statement::If(i) => {
                    Self::collect_writes_in_block(&i.then_block.lock(), writes);
                    Self::collect_writes_in_block(&i.else_block.lock(), writes);
                }
                Statement::While(w) => {
                    Self::collect_writes_in_block(&w.block.lock(), writes);
                }
                Statement::Repeat(r) => {
                    Self::collect_writes_in_block(&r.block.lock(), writes);
                }
                Statement::NumericFor(nf) => {
                    Self::collect_writes_in_block(&nf.block.lock(), writes);
                }
                Statement::GenericFor(gf) => {
                    Self::collect_writes_in_block(&gf.block.lock(), writes);
                }
                _ => {}
            }
        }
    }

    /// Collect all locals that are read OR written in a block (recursively).
    ///
    /// This includes both `values_read()` and `values_written()` to catch cases
    /// where a local is modified inside a nested block (like if-statements).
    #[allow(dead_code)]
    fn collect_locals_in_block(block: &Block, locals: &mut FxHashSet<RcLocal>) {
        for statement in &block.0 {
            // Collect reads
            for local in statement.values_read() {
                locals.insert(local.clone());
            }
            // Collect writes - important for catching reassignments in nested blocks
            for local in statement.values_written() {
                locals.insert(local.clone());
            }
            // Recurse into nested blocks
            match statement {
                Statement::If(i) => {
                    Self::collect_locals_in_block(&i.then_block.lock(), locals);
                    Self::collect_locals_in_block(&i.else_block.lock(), locals);
                }
                Statement::While(w) => {
                    Self::collect_locals_in_block(&w.block.lock(), locals);
                }
                Statement::Repeat(r) => {
                    Self::collect_locals_in_block(&r.block.lock(), locals);
                }
                Statement::NumericFor(nf) => {
                    Self::collect_locals_in_block(&nf.block.lock(), locals);
                }
                Statement::GenericFor(gf) => {
                    Self::collect_locals_in_block(&gf.block.lock(), locals);
                }
                _ => {}
            }
        }
    }

    /// Check if a statement is a candidate for inlining
    /// Returns the local and rvalue if it's a simple single assignment to a local
    fn extract_inline_candidate(&self, statement: &Statement) -> Option<(RcLocal, RValue)> {
        let Statement::Assign(assign) = statement else {
            return None;
        };

        if assign.left.len() != 1 || assign.right.len() != 1 {
            return None;
        }

        let LValue::Local(local) = &assign.left[0] else {
            return None;
        };

        // We only check that a name exists, the filtering based on whether
        // it's unnamed or side-effect-free is done in find_inline_candidates
        local.name()?;

        Some((local.clone(), assign.right[0].clone()))
    }

    /// Find a safe target for inlining a side-effect-having RValue.
    ///
    /// For side-effect RValues (like calls), we can only inline if:
    /// 1. Intervening statements don't have side effects (they're safe assignments)
    /// 2. We find a statement that uses this local exactly once
    /// 3. No intervening statement could observe or modify state affected by the RValue
    ///
    /// Returns the index of the safe target statement, if one exists.
    fn find_safe_inline_target(
        &self,
        local: &RcLocal,
        rvalue: &RValue,
        block: &Block,
        start_idx: usize,
    ) -> Option<usize> {
        // Collect locals that our RValue reads - we need to ensure no intervening
        // statement writes to these
        let rvalue_reads: FxHashSet<_> = rvalue.values_read().into_iter().cloned().collect();

        // Look for the first statement that uses this local
        for j in (start_idx + 1)..block.len() {
            let statement = &block[j];

            // Check if this statement uses our local
            if self.is_single_use(local, statement) {
                return Some(j);
            }

            // Check if this intervening statement conflicts with our RValue
            // A statement conflicts if it writes to a local that our RValue reads
            let writes = statement.values_written();
            let conflicts = writes.iter().any(|w| {
                rvalue_reads.contains(*w) ||
                rvalue_reads.iter().any(|r| r.name().is_some() && r.name() == w.name())
            });

            if conflicts {
                // Can't safely reorder past a statement that modifies our inputs
                return None;
            }

            // For assignments, we allow continuing even if they have side effects,
            // as long as they don't conflict with our RValue's reads
            if let Statement::Assign(_) = statement {
                // Continue looking
            } else {
                // Non-assignment statement (if, while, call, etc.) - not safe to reorder past
                return None;
            }
        }

        // Didn't find a use
        None
    }

    /// Check if the local is used exactly once in the given statement
    fn is_single_use(&self, local: &RcLocal, statement: &Statement) -> bool {
        let reads = statement.values_read();
        // Compare by identity OR by name for SSA versions
        let count = reads.iter().filter(|l| {
            **l == local || (l.name().is_some() && l.name() == local.name())
        }).count();
        count == 1
    }

    /// Apply immediate inlines (for side-effect RValues)
    ///
    /// Only applies inlines for locals that are actually used in this statement.
    fn apply_immediate_inlines(&mut self, statement: &mut Statement) {
        if self.immediate_inlines.is_empty() {
            return;
        }

        // Find which locals are used in this statement
        let reads = statement.values_read();
        let mut to_apply = Vec::new();

        for read_local in reads {
            // Check if this local (by identity or name) is in our immediate_inlines
            let mut found_key = None;
            for (local, _) in &self.immediate_inlines {
                let matches = local == read_local
                    || (local.name().is_some() && local.name() == read_local.name());
                if matches {
                    found_key = Some(local.clone());
                    break;
                }
            }
            if let Some(key) = found_key {
                if let Some(rvalue) = self.immediate_inlines.remove(&key) {
                    to_apply.push((key, rvalue));
                }
            }
        }

        for (local, rvalue) in to_apply {
            self.replace_local_with_rvalue(statement, &local, rvalue);
        }
    }

    /// Apply deferred inlines (for side-effect-free RValues)
    fn apply_deferred_inlines(
        &mut self,
        statement: &mut Statement,
        candidates: &FxHashSet<RcLocal>,
    ) {
        if self.deferred_inlines.is_empty() {
            return;
        }

        // Never inline into while/repeat conditions - the variable may be modified
        // inside the loop body which we don't track when counting uses.
        // This prevents incorrectly inlining loop counters like:
        //   local i = 1
        //   while i <= n do  -- would incorrectly become: while 1 <= n do
        //     i = i + 1
        //   end
        if matches!(statement, Statement::While(_) | Statement::Repeat(_)) {
            return;
        }

        // Find which deferred locals are used in this statement
        let reads = statement.values_read();
        let mut to_inline = Vec::new();

        for local in reads {
            // Check if this local is a candidate - use name-based comparison for SSA versions
            let is_candidate = candidates.contains(local)
                || candidates.iter().any(|c| c.name().is_some() && c.name() == local.name());

            if is_candidate {
                // Look up in deferred_inlines using name-based matching for SSA versions
                let mut found_key = None;
                for (key, _) in self.deferred_inlines.iter() {
                    let matches = key == local
                        || (key.name().is_some() && key.name() == local.name());
                    if matches {
                        found_key = Some(key.clone());
                        break;
                    }
                }
                if let Some(key) = found_key {
                    if let Some(rvalue) = self.deferred_inlines.remove(&key) {
                        to_inline.push((local.clone(), rvalue));
                    }
                }
            }
        }

        for (local, rvalue) in to_inline {
            self.replace_local_with_rvalue(statement, &local, rvalue);
        }
    }

    /// Replace occurrences of a local with an RValue in a statement
    fn replace_local_with_rvalue(
        &self,
        statement: &mut Statement,
        local: &RcLocal,
        replacement: RValue,
    ) {
        // Collect RValues from both the right side AND from LValues on the left side.
        // For example, in `v2_[1] = x`, the `v2_` is inside an LValue::Index and needs
        // to be traversed for replacement.
        let mut work: Vec<*mut RValue> = statement
            .rvalues_mut()
            .into_iter()
            .map(|r| r as *mut RValue)
            .collect();

        // Also collect RValues from LValues (e.g., the table in `table[idx] = val`)
        for lvalue in statement.lvalues_mut() {
            if let LValue::Index(index) = lvalue {
                work.extend(index.rvalues_mut().into_iter().map(|r| r as *mut RValue));
            }
        }

        while let Some(ptr) = work.pop() {
            let rvalue = unsafe { &mut *ptr };

            if let RValue::Local(l) = rvalue {
                // Use identity-based comparison only - name-based matching is too aggressive
                // and can incorrectly replace unrelated locals that happen to have the same name.
                // SSA versions should already be unified by SSA destruction, so identity comparison
                // should be sufficient.
                if l == local {
                    *rvalue = replacement.clone();
                    continue;
                }
            }

            work.extend(
                rvalue
                    .rvalues_mut()
                    .into_iter()
                    .map(|r| r as *mut RValue),
            );
        }
    }

    /// Recurse into nested blocks and closures within a statement
    /// `protected_locals` contains locals from outer scopes that should never be inlined
    fn recurse_into_statement(&mut self, statement: &mut Statement, depth: usize, protected_locals: &FxHashSet<RcLocal>) {
        match statement {
            Statement::If(r#if) => {
                self.inline_block(&mut r#if.then_block.lock(), depth + 1, protected_locals);
                self.inline_block(&mut r#if.else_block.lock(), depth + 1, protected_locals);
            }
            Statement::While(r#while) => {
                // Add while condition locals to protected set for nested processing
                let mut new_protected = protected_locals.clone();
                for local in r#while.condition.values_read() {
                    new_protected.insert(local.clone());
                }
                self.inline_block(&mut r#while.block.lock(), depth + 1, &new_protected);
            }
            Statement::Repeat(repeat) => {
                // Add repeat condition locals to protected set for nested processing
                let mut new_protected = protected_locals.clone();
                for local in repeat.condition.values_read() {
                    new_protected.insert(local.clone());
                }
                self.inline_block(&mut repeat.block.lock(), depth + 1, &new_protected);
            }
            Statement::NumericFor(numeric_for) => {
                self.inline_block(&mut numeric_for.block.lock(), depth + 1, protected_locals);
            }
            Statement::GenericFor(generic_for) => {
                self.inline_block(&mut generic_for.block.lock(), depth + 1, protected_locals);
            }
            _ => {}
        }

        self.find_and_process_closures(statement, depth, protected_locals);
    }

    /// Find closures in rvalues using iterative traversal
    fn find_and_process_closures(&mut self, statement: &mut Statement, depth: usize, protected_locals: &FxHashSet<RcLocal>) {
        let mut closures = Vec::new();
        let mut work: Vec<*mut RValue> = statement
            .rvalues_mut()
            .into_iter()
            .map(|r| r as *mut RValue)
            .collect();

        while let Some(ptr) = work.pop() {
            let rvalue = unsafe { &mut *ptr };

            if let RValue::Closure(closure) = rvalue {
                closures.push(closure.function.clone());
            } else {
                work.extend(
                    rvalue
                        .rvalues_mut()
                        .into_iter()
                        .map(|r| r as *mut RValue),
                );
            }
        }

        for func in closures {
            // Closures start with a fresh protected set (they're separate scopes)
            self.inline_block(&mut func.lock().body, depth + 1, &FxHashSet::default());
        }
    }
}

/// Check if a name represents an unnamed temporary (e.g., "v1_", "v2_")
fn is_unnamed_temporary(name: &str) -> bool {
    if name == "_" {
        return true;
    }

    if let Some(rest) = name.strip_prefix('v') {
        if let Some(digits) = rest.strip_suffix('_') {
            return digits.chars().all(|c| c.is_ascii_digit());
        }
    }

    false
}

/// Check if an RValue has side effects (calls can have side effects)
///
/// Note: We intentionally don't treat Index as having side effects even though
/// it could theoretically call __index metamethod. In practice, decompiled code
/// rarely uses metamethods and treating Index as side-effect-free produces much
/// cleaner output by allowing field accesses like `DebugUtil.renderTextLine` to
/// be inlined into call expressions.
fn has_side_effects(rvalue: &RValue) -> bool {
    match rvalue {
        RValue::Call(_) | RValue::MethodCall(_) => true,
        RValue::Select(_) => true, // Select wraps calls
        RValue::Unary(unary) => has_side_effects(&unary.value),
        RValue::Binary(binary) => has_side_effects(&binary.left) || has_side_effects(&binary.right),
        RValue::Table(table) => table.0.iter().any(|(k, v)| {
            k.as_ref().map_or(false, has_side_effects) || has_side_effects(v)
        }),
        // Index could theoretically call __index, but we treat it as side-effect-free
        // for practical decompilation purposes
        RValue::Index(index) => has_side_effects(&index.left) || has_side_effects(&index.right),
        _ => false,
    }
}

use crate::{Assign, BinaryOperation, If, Literal, Unary, UnaryOperation};

/// Simplify boolean ternary patterns.
///
/// Transforms patterns like:
///   local v3_
///   if condition then
///       v3_ = false
///   else
///       v3_ = true
///   end
///
/// Into:
///   local v3_ = not condition
///
/// Also handles the inverted case:
///   if condition then v3_ = true else v3_ = false end
/// Into:
///   local v3_ = condition  (or v3_ = condition ~= nil for nil checks)
fn simplify_boolean_ternaries(block: &mut Block) {
    let mut i = 0;
    while i < block.len() {
        // First, recurse into nested blocks
        match &mut block[i] {
            Statement::If(if_stat) => {
                simplify_boolean_ternaries(&mut if_stat.then_block.lock());
                simplify_boolean_ternaries(&mut if_stat.else_block.lock());
            }
            Statement::While(while_stat) => {
                simplify_boolean_ternaries(&mut while_stat.block.lock());
            }
            Statement::Repeat(repeat_stat) => {
                simplify_boolean_ternaries(&mut repeat_stat.block.lock());
            }
            Statement::NumericFor(for_stat) => {
                simplify_boolean_ternaries(&mut for_stat.block.lock());
            }
            Statement::GenericFor(for_stat) => {
                simplify_boolean_ternaries(&mut for_stat.block.lock());
            }
            _ => {}
        }

        // Check for closures in RValues
        for rvalue in block[i].rvalues_mut() {
            simplify_ternaries_in_rvalue(rvalue);
        }

        // Now check for the ternary pattern: if cond then x = true/false else x = false/true end
        if let Statement::If(if_stat) = &block[i] {
            if let Some((local, value)) = extract_boolean_ternary(if_stat) {
                // Check if there's a preceding declaration for this local that we can merge
                let has_preceding_decl = i > 0 && is_empty_declaration_for(&block[i - 1], &local);

                // Replace the if statement with an assignment
                let mut assign = Assign::new(vec![LValue::Local(local)], vec![value]);

                if has_preceding_decl {
                    // Remove the preceding declaration and make this a prefix assignment
                    block.0.remove(i - 1);
                    i -= 1;
                    assign.prefix = true;
                }

                block.0[i] = assign.into();
            }
        }

        i += 1;
    }
}

/// Check if a statement is an empty declaration (prefix assignment with no RHS) for a specific local
fn is_empty_declaration_for(statement: &Statement, target_local: &RcLocal) -> bool {
    if let Statement::Assign(assign) = statement {
        if assign.prefix && assign.right.is_empty() && assign.left.len() == 1 {
            if let Some(local) = assign.left[0].as_local() {
                // Compare by identity or by name for SSA versions
                return local == target_local ||
                    (local.name().is_some() && local.name() == target_local.name());
            }
        }
    }
    false
}

fn simplify_ternaries_in_rvalue(rvalue: &mut RValue) {
    if let RValue::Closure(closure) = rvalue {
        simplify_boolean_ternaries(&mut closure.function.lock().body);
    }

    for rv in rvalue.rvalues_mut() {
        simplify_ternaries_in_rvalue(rv);
    }
}

/// Extract the boolean ternary pattern from an if statement.
///
/// Returns Some((local, simplified_value)) if the pattern matches:
///   if condition then local = true/false else local = false/true end
///
/// The simplified_value will be either the condition itself (possibly with `not`)
/// or for nil checks like `x == nil`, it will be `x ~= nil` or `x == nil`.
fn extract_boolean_ternary(if_stat: &If) -> Option<(RcLocal, RValue)> {
    let then_block = if_stat.then_block.lock();
    let else_block = if_stat.else_block.lock();

    // Both branches must have exactly one statement
    if then_block.len() != 1 || else_block.len() != 1 {
        return None;
    }

    // Both statements must be assignments
    let then_assign = then_block[0].as_assign()?;
    let else_assign = else_block[0].as_assign()?;

    // Both must assign to a single local
    if then_assign.left.len() != 1 || else_assign.left.len() != 1 {
        return None;
    }
    if then_assign.right.len() != 1 || else_assign.right.len() != 1 {
        return None;
    }

    let then_local = then_assign.left[0].as_local()?;
    let else_local = else_assign.left[0].as_local()?;

    // Must be assigning to the same local (compare by identity or by name for SSA versions)
    let same_local = then_local == else_local ||
        (then_local.name().is_some() && then_local.name() == else_local.name());
    if !same_local {
        return None;
    }

    let then_value = &then_assign.right[0];
    let else_value = &else_assign.right[0];

    // First try: boolean literals (true/false)
    if let Some(result) = try_extract_boolean_ternary(if_stat, then_local, then_value, else_value) {
        return Some(result);
    }

    // Second try: "or-pattern" - when then-value is the same as condition
    // Pattern: if cond then x = cond else x = val end
    // Becomes: x = cond or val
    // This is a common Lua idiom for default values
    if rvalues_equal(&if_stat.condition, then_value) {
        let or_expr = RValue::Binary(crate::Binary::new(
            if_stat.condition.clone(),
            else_value.clone(),
            BinaryOperation::Or,
        ));
        return Some((then_local.clone(), or_expr));
    }

    // Third try: "and-pattern" - when else-value is the same as condition
    // Pattern: if cond then x = val else x = cond end
    // Becomes: x = cond and val
    // This works because:
    // - If cond is truthy: cond and val = val (correct)
    // - If cond is falsy: cond and val = cond (short-circuits to the falsy value)
    if rvalues_equal(&if_stat.condition, else_value) {
        let and_expr = RValue::Binary(crate::Binary::new(
            if_stat.condition.clone(),
            then_value.clone(),
            BinaryOperation::And,
        ));
        return Some((then_local.clone(), and_expr));
    }

    // Fourth try: general ternary with truthy then-value
    // Pattern: if cond then x = val1 else x = val2 end
    // Becomes: x = cond and val1 or val2
    // This is only safe when val1 is guaranteed truthy (non-nil, non-false)
    if is_guaranteed_truthy(then_value) {
        let ternary = RValue::Binary(crate::Binary::new(
            RValue::Binary(crate::Binary::new(
                if_stat.condition.clone(),
                then_value.clone(),
                BinaryOperation::And,
            )),
            else_value.clone(),
            BinaryOperation::Or,
        ));
        return Some((then_local.clone(), ternary));
    }

    None
}

/// Try to extract the special case of boolean true/false ternary
fn try_extract_boolean_ternary(
    if_stat: &If,
    then_local: &RcLocal,
    then_value: &RValue,
    else_value: &RValue,
) -> Option<(RcLocal, RValue)> {
    let then_bool = match then_value {
        RValue::Literal(Literal::Boolean(b)) => Some(*b),
        _ => None,
    }?;

    let else_bool = match else_value {
        RValue::Literal(Literal::Boolean(b)) => Some(*b),
        _ => None,
    }?;

    // Must be opposite booleans
    if then_bool == else_bool {
        return None;
    }

    // Now simplify the condition based on which branch is true
    let simplified = if then_bool {
        // if cond then x = true else x = false => x = cond (or simplified form)
        simplify_to_boolean(&if_stat.condition)
    } else {
        // if cond then x = false else x = true => x = not cond (or simplified form)
        negate_condition(&if_stat.condition)
    };

    Some((then_local.clone(), simplified))
}

/// Check if a value is guaranteed to be truthy (not nil, not false)
/// This is needed for the `cond and val1 or val2` transformation to be safe
fn is_guaranteed_truthy(value: &RValue) -> bool {
    match value {
        // Numbers are always truthy (including 0)
        RValue::Literal(Literal::Number(_)) => true,
        // Non-empty strings are truthy
        RValue::Literal(Literal::String(_)) => true,
        // Tables are truthy
        RValue::Table(_) => true,
        // Closures are truthy
        RValue::Closure(_) => true,
        // true is truthy, false and nil are not
        RValue::Literal(Literal::Boolean(b)) => *b,
        RValue::Literal(Literal::Nil) => false,
        // For other expressions (locals, calls, etc.) we can't be sure
        _ => false,
    }
}

/// Convert a condition to a boolean expression.
///
/// For comparisons like `x == nil`, returns `x ~= nil` (which is a boolean).
/// For other expressions, returns the expression as-is (truthy/falsy behavior).
fn simplify_to_boolean(condition: &RValue) -> RValue {
    // For `x == nil`, we can use `x ~= nil` which is already boolean
    if let RValue::Binary(binary) = condition {
        if binary.operation == BinaryOperation::Equal {
            if matches!(&*binary.right, RValue::Literal(Literal::Nil)) {
                // x == nil => x ~= nil (inverted)
                return RValue::Binary(crate::Binary::new(
                    (*binary.left).clone(),
                    (*binary.right).clone(),
                    BinaryOperation::NotEqual,
                ));
            }
            if matches!(&*binary.left, RValue::Literal(Literal::Nil)) {
                // nil == x => x ~= nil (inverted)
                return RValue::Binary(crate::Binary::new(
                    (*binary.right).clone(),
                    (*binary.left).clone(),
                    BinaryOperation::NotEqual,
                ));
            }
        }
        if binary.operation == BinaryOperation::NotEqual {
            if matches!(&*binary.right, RValue::Literal(Literal::Nil)) {
                // x ~= nil => already boolean, just return as-is
                return condition.clone();
            }
            if matches!(&*binary.left, RValue::Literal(Literal::Nil)) {
                // nil ~= x => x ~= nil
                return RValue::Binary(crate::Binary::new(
                    (*binary.right).clone(),
                    (*binary.left).clone(),
                    BinaryOperation::NotEqual,
                ));
            }
        }
    }

    // For other conditions, return as-is
    condition.clone()
}

/// Negate a condition, simplifying where possible.
///
/// For `x == nil`, returns `x ~= nil`.
/// For `x ~= nil`, returns `x == nil`.
/// For other expressions, returns `not condition`.
fn negate_condition(condition: &RValue) -> RValue {
    // For comparisons, we can flip the operator
    if let RValue::Binary(binary) = condition {
        let flipped_op = match binary.operation {
            BinaryOperation::Equal => Some(BinaryOperation::NotEqual),
            BinaryOperation::NotEqual => Some(BinaryOperation::Equal),
            BinaryOperation::LessThan => Some(BinaryOperation::GreaterThanOrEqual),
            BinaryOperation::LessThanOrEqual => Some(BinaryOperation::GreaterThan),
            BinaryOperation::GreaterThan => Some(BinaryOperation::LessThanOrEqual),
            BinaryOperation::GreaterThanOrEqual => Some(BinaryOperation::LessThan),
            _ => None,
        };

        if let Some(op) = flipped_op {
            return RValue::Binary(crate::Binary::new(
                (*binary.left).clone(),
                (*binary.right).clone(),
                op,
            ));
        }
    }

    // For `not x`, return `x`
    if let RValue::Unary(unary) = condition {
        if unary.operation == UnaryOperation::Not {
            return (*unary.value).clone();
        }
    }

    // Default: wrap in `not`
    RValue::Unary(Unary::new(condition.clone(), UnaryOperation::Not))
}

/// Check if two RValues are structurally equal.
///
/// For locals, compares by identity or by name (for SSA versions).
/// For other types, uses standard equality.
fn rvalues_equal(a: &RValue, b: &RValue) -> bool {
    match (a, b) {
        (RValue::Local(la), RValue::Local(lb)) => {
            la == lb || (la.name().is_some() && la.name() == lb.name())
        }
        (RValue::Literal(la), RValue::Literal(lb)) => la == lb,
        (RValue::Binary(ba), RValue::Binary(bb)) => {
            ba.operation == bb.operation
                && rvalues_equal(&ba.left, &bb.left)
                && rvalues_equal(&ba.right, &bb.right)
        }
        (RValue::Unary(ua), RValue::Unary(ub)) => {
            ua.operation == ub.operation && rvalues_equal(&ua.value, &ub.value)
        }
        (RValue::Index(ia), RValue::Index(ib)) => {
            rvalues_equal(&ia.left, &ib.left) && rvalues_equal(&ia.right, &ib.right)
        }
        // For other complex types (Call, MethodCall, etc.), don't consider them equal
        // This is conservative and avoids false positives
        _ => false,
    }
}

/// Simplify and-chain patterns.
///
/// Transforms patterns like:
///   local v = expr1
///   if v then
///       v = expr2
///   end
///
/// Into:
///   local v = expr1 and expr2
///
/// This handles the common Lua idiom for short-circuit evaluation chains.
/// The pattern can repeat multiple times:
///   local v = expr1
///   if v then v = expr2 end
///   if v then v = expr3 end
/// Into:
///   local v = expr1 and expr2 and expr3
fn simplify_and_chains(block: &mut Block) {
    let mut i = 0;
    while i < block.len() {
        // First, recurse into nested blocks
        match &mut block[i] {
            Statement::If(if_stat) => {
                simplify_and_chains(&mut if_stat.then_block.lock());
                simplify_and_chains(&mut if_stat.else_block.lock());
            }
            Statement::While(while_stat) => {
                simplify_and_chains(&mut while_stat.block.lock());
            }
            Statement::Repeat(repeat_stat) => {
                simplify_and_chains(&mut repeat_stat.block.lock());
            }
            Statement::NumericFor(for_stat) => {
                simplify_and_chains(&mut for_stat.block.lock());
            }
            Statement::GenericFor(for_stat) => {
                simplify_and_chains(&mut for_stat.block.lock());
            }
            _ => {}
        }

        // Check for closures in RValues
        for rvalue in block[i].rvalues_mut() {
            simplify_and_chains_in_rvalue(rvalue);
        }

        // Look for pattern: assign to local, then if-reassign chain
        // Start by checking if current statement is an assignment to a single local
        let Some(assign) = block[i].as_assign() else {
            i += 1;
            continue;
        };

        if assign.left.len() != 1 || assign.right.len() != 1 {
            i += 1;
            continue;
        }

        let Some(local) = assign.left[0].as_local() else {
            i += 1;
            continue;
        };

        let local = local.clone();
        let initial_value = assign.right[0].clone();

        // Check for nil-check field assignment pattern FIRST (works for all locals, not just unnamed):
        // local v = expr
        // if v ~= nil then
        //     field = v
        // end
        // Becomes: field = expr or field
        // This eliminates the local entirely, so it's safe even for named locals
        if i + 1 < block.len() {
            if let Some((field_lvalue, fallback)) =
                extract_nil_check_field_assignment(&block[i + 1], &local)
            {
                // Transform to: field = expr or fallback
                let combined = RValue::Binary(crate::Binary::new(
                    initial_value.clone(),
                    fallback,
                    BinaryOperation::Or,
                ));

                // Replace the local assignment with field assignment
                let Statement::Assign(assign) = &mut block[i] else {
                    unreachable!()
                };
                assign.left[0] = field_lvalue;
                assign.right[0] = combined;
                assign.prefix = false; // Field assignments don't have local prefix

                // Remove the if statement
                block.0.remove(i + 1);

                i += 1;
                continue;
            }
        }

        // Only simplify other patterns for unnamed temporaries
        let is_unnamed = local.name().map_or(true, |n| is_unnamed_temporary(&n));
        if !is_unnamed {
            i += 1;
            continue;
        }

        let is_prefix = assign.prefix;

        // Now look ahead for if-reassign patterns
        let mut j = i + 1;
        let mut chain_count = 0;

        while j < block.len() {
            if is_and_chain_pattern(&block[j], &local) {
                chain_count += 1;
                j += 1;
            } else {
                break;
            }
        }

        if chain_count > 0 {
            // Now fold all the if statements into the assignment
            // Get the initial RValue
            let Statement::Assign(assign) = &mut block[i] else {
                unreachable!()
            };
            let mut combined = assign.right[0].clone();

            // Fold each if statement
            for k in (i + 1)..=(i + chain_count) {
                let new_value = extract_and_chain_value(&block[k]);

                // Create: combined and new_value
                combined = RValue::Binary(crate::Binary::new(
                    combined,
                    new_value,
                    BinaryOperation::And,
                ));
            }

            // Update the original assignment
            let Statement::Assign(assign) = &mut block[i] else {
                unreachable!()
            };
            assign.right[0] = combined;
            assign.prefix = is_prefix;

            // Remove the if statements
            for _ in 0..chain_count {
                block.0.remove(i + 1);
            }

            i += 1;
            continue;
        }

        // Check for or-chain pattern: local v = cond1; if not cond1 then v = cond2 end
        // This becomes: v = cond1 or cond2
        if i + 1 < block.len() {
            if let Some(or_value) = is_or_chain_pattern(&block[i + 1], &local, &initial_value) {
                // Transform to: v = initial_value or or_value
                let combined = RValue::Binary(crate::Binary::new(
                    initial_value,
                    or_value,
                    BinaryOperation::Or,
                ));

                let Statement::Assign(assign) = &mut block[i] else {
                    unreachable!()
                };
                assign.right[0] = combined;
                assign.prefix = is_prefix;

                // Remove the if statement
                block.0.remove(i + 1);

                i += 1;
                continue;
            }
        }

        // Check for nested if-chain pattern with false initialization:
        // local v = false
        // if cond1 then
        //     v = false
        //     if cond2 then
        //         v = finalExpr
        //     end
        // end
        // Becomes: v = cond1 and cond2 and finalExpr
        if matches!(initial_value, RValue::Literal(Literal::Boolean(false))) {
            if i + 1 < block.len() {
                if let Some((conditions, final_expr)) = extract_nested_and_chain(&block[i + 1], &local) {
                    if !conditions.is_empty() {
                        // Build: cond1 and cond2 and ... and finalExpr
                        let mut combined = conditions[0].clone();
                        for cond in &conditions[1..] {
                            combined = RValue::Binary(crate::Binary::new(
                                combined,
                                cond.clone(),
                                BinaryOperation::And,
                            ));
                        }
                        combined = RValue::Binary(crate::Binary::new(
                            combined,
                            final_expr,
                            BinaryOperation::And,
                        ));

                        let Statement::Assign(assign) = &mut block[i] else {
                            unreachable!()
                        };
                        assign.right[0] = combined;
                        assign.prefix = is_prefix;

                        // Remove the if statement
                        block.0.remove(i + 1);

                        i += 1;
                        continue;
                    }
                }
            }
        }

        // Check for expression-initialized and-chain pattern:
        // local v = expr1
        // if v then
        //     v = false
        //     if cond2 then
        //         v = finalExpr
        //     end
        // end
        // Becomes: v = expr1 and cond2 and finalExpr
        // This handles patterns where the initial value is not just `false` but an expression
        if !matches!(initial_value, RValue::Literal(Literal::Boolean(false))) {
            if i + 1 < block.len() {
                if let Some((conditions, final_expr)) = extract_expr_and_chain(&block[i + 1], &local) {
                    if !conditions.is_empty() {
                        // Build: expr1 and cond1 and cond2 and ... and finalExpr
                        let mut combined = initial_value.clone();
                        for cond in &conditions {
                            combined = RValue::Binary(crate::Binary::new(
                                combined,
                                cond.clone(),
                                BinaryOperation::And,
                            ));
                        }
                        combined = RValue::Binary(crate::Binary::new(
                            combined,
                            final_expr,
                            BinaryOperation::And,
                        ));

                        let Statement::Assign(assign) = &mut block[i] else {
                            unreachable!()
                        };
                        assign.right[0] = combined;
                        assign.prefix = is_prefix;

                        // Remove the if statement
                        block.0.remove(i + 1);

                        i += 1;
                        continue;
                    }
                }
            }
        }

        // Check for true initialization with conditional reassignment:
        // local v = true
        // if cond then
        //     v = expr
        // end
        // Becomes: v = not cond or expr
        if matches!(initial_value, RValue::Literal(Literal::Boolean(true))) {
            if i + 1 < block.len() {
                if let Some((condition, new_value)) = extract_true_or_chain(&block[i + 1], &local) {
                    // Transform to: v = not cond or new_value
                    let negated_cond = negate_condition(&condition);
                    let combined = RValue::Binary(crate::Binary::new(
                        negated_cond,
                        new_value,
                        BinaryOperation::Or,
                    ));

                    let Statement::Assign(assign) = &mut block[i] else {
                        unreachable!()
                    };
                    assign.right[0] = combined;
                    assign.prefix = is_prefix;

                    // Remove the if statement
                    block.0.remove(i + 1);

                    i += 1;
                    continue;
                }
            }
        }

        i += 1;
    }
}

fn simplify_and_chains_in_rvalue(rvalue: &mut RValue) {
    if let RValue::Closure(closure) = rvalue {
        simplify_and_chains(&mut closure.function.lock().body);
    }

    for rv in rvalue.rvalues_mut() {
        simplify_and_chains_in_rvalue(rv);
    }
}

/// Check if statement matches the and-chain pattern: if v then v = expr end
fn is_and_chain_pattern(statement: &Statement, target_local: &RcLocal) -> bool {
    let Some(if_stat) = statement.as_if() else {
        return false;
    };

    // Check condition is the local itself
    let Some(cond_local) = if_stat.condition.as_local() else {
        return false;
    };

    // Must be same local (by identity or name)
    let same = cond_local == target_local
        || (cond_local.name().is_some() && cond_local.name() == target_local.name());
    if !same {
        return false;
    }

    // Must have empty else branch
    if !if_stat.else_block.lock().is_empty() {
        return false;
    }

    // Then branch must have exactly one assignment
    let then_block = if_stat.then_block.lock();
    if then_block.len() != 1 {
        return false;
    }

    let Some(then_assign) = then_block[0].as_assign() else {
        return false;
    };

    // Must assign to single local
    if then_assign.left.len() != 1 || then_assign.right.len() != 1 {
        return false;
    }

    let Some(assign_local) = then_assign.left[0].as_local() else {
        return false;
    };

    // Must assign to the same local
    assign_local == target_local
        || (assign_local.name().is_some() && assign_local.name() == target_local.name())
}

/// Check if statement matches the or-chain pattern: if not cond1 then v = cond2 end
/// where cond1 is the initial value assigned to v.
///
/// Pattern: local v = cond1; if not cond1 then v = cond2 end
/// Becomes: local v = cond1 or cond2
///
/// The condition can be:
/// 1. Negated initial value: `if not (x == y) then` or `if x ~= y then`
/// 2. The initial value must be a comparison that can be negated
///
/// Returns Some(or_value) if pattern matches, where or_value is the value to OR with.
fn is_or_chain_pattern(
    statement: &Statement,
    target_local: &RcLocal,
    initial_value: &RValue,
) -> Option<RValue> {
    let if_stat = statement.as_if()?;

    // Must have empty else branch
    if !if_stat.else_block.lock().is_empty() {
        return None;
    }

    // Then branch must have exactly one assignment
    let then_block = if_stat.then_block.lock();
    if then_block.len() != 1 {
        return None;
    }

    let then_assign = then_block[0].as_assign()?;

    // Must assign to single local
    if then_assign.left.len() != 1 || then_assign.right.len() != 1 {
        return None;
    }

    let assign_local = then_assign.left[0].as_local()?;

    // Must assign to the same local
    let same_local = assign_local == target_local
        || (assign_local.name().is_some() && assign_local.name() == target_local.name());
    if !same_local {
        return None;
    }

    // Check if condition is the negation of initial_value OR negation of the local
    // Case 1: initial_value is `x == y` and condition is `x ~= y`
    // Case 2: initial_value is `x ~= y` and condition is `x == y`
    // Case 3: condition is `not initial_value`
    // Case 4: condition is `not local` (the common case for field accesses)
    if is_negation_of(&if_stat.condition, initial_value) {
        return Some(then_assign.right[0].clone());
    }

    // Case 4: condition is `not local` where local is the target
    // This handles: local v = x.y; if not v then v = z end
    // => v = x.y or z
    if let RValue::Unary(unary) = &if_stat.condition {
        if unary.operation == UnaryOperation::Not {
            if let RValue::Local(cond_local) = &*unary.value {
                let same = cond_local == target_local
                    || (cond_local.name().is_some() && cond_local.name() == target_local.name());
                if same {
                    return Some(then_assign.right[0].clone());
                }
            }
        }
    }

    None
}

/// Extract nil-check field assignment pattern.
///
/// Pattern:
/// ```lua
/// local v = expr
/// if v ~= nil then
///     field = v
/// end
/// ```
///
/// Returns Some((field_lvalue, field_rvalue)) if pattern matches.
/// The caller should transform this to: `field = expr or field`
fn extract_nil_check_field_assignment(
    statement: &Statement,
    target_local: &RcLocal,
) -> Option<(LValue, RValue)> {
    let if_stat = statement.as_if()?;

    // Must have empty else branch
    if !if_stat.else_block.lock().is_empty() {
        return None;
    }

    // Then branch must have exactly one assignment
    let then_block = if_stat.then_block.lock();
    if then_block.len() != 1 {
        return None;
    }

    let then_assign = then_block[0].as_assign()?;

    // Must be single assignment
    if then_assign.left.len() != 1 || then_assign.right.len() != 1 {
        return None;
    }

    // Left side must be a field (index), not a local
    // e.g., self.colorDisabled
    let field_lvalue = &then_assign.left[0];
    if field_lvalue.as_local().is_some() {
        // If assigning to a local, this is handled by other patterns
        return None;
    }

    // Right side must be the target local
    let rhs_local = then_assign.right[0].as_local()?;
    let same = rhs_local == target_local
        || (rhs_local.name().is_some() && rhs_local.name() == target_local.name());
    if !same {
        return None;
    }

    // Condition must be `v ~= nil` or `v` (truthy check)
    // Check for `v ~= nil`
    if let RValue::Binary(binary) = &if_stat.condition {
        if binary.operation == BinaryOperation::NotEqual {
            // Check if it's `local ~= nil` or `nil ~= local`
            let is_nil_check = if let RValue::Local(left_local) = &*binary.left {
                let same = left_local == target_local
                    || (left_local.name().is_some() && left_local.name() == target_local.name());
                same && matches!(&*binary.right, RValue::Literal(Literal::Nil))
            } else if let RValue::Local(right_local) = &*binary.right {
                let same = right_local == target_local
                    || (right_local.name().is_some() && right_local.name() == target_local.name());
                same && matches!(&*binary.left, RValue::Literal(Literal::Nil))
            } else {
                false
            };

            if is_nil_check {
                // The fallback is reading the same field
                let fallback = lvalue_to_rvalue(field_lvalue);
                return Some((field_lvalue.clone(), fallback));
            }
        }
    }

    // Also check for simple truthy check: `if v then`
    if let RValue::Local(cond_local) = &if_stat.condition {
        let same = cond_local == target_local
            || (cond_local.name().is_some() && cond_local.name() == target_local.name());
        if same {
            let fallback = lvalue_to_rvalue(field_lvalue);
            return Some((field_lvalue.clone(), fallback));
        }
    }

    None
}

/// Convert an LValue to an equivalent RValue for reading.
fn lvalue_to_rvalue(lvalue: &LValue) -> RValue {
    match lvalue {
        LValue::Local(local) => RValue::Local(local.clone()),
        LValue::Index(index) => RValue::Index(index.clone()),
        LValue::Global(global) => RValue::Global(global.clone()),
    }
}

/// Check if `condition` is the negation of `original`.
///
/// Returns true if:
/// - `condition` is `not original`
/// - `original` is `a == b` and `condition` is `a ~= b`
/// - `original` is `a ~= b` and `condition` is `a == b`
/// - Similar for <, <=, >, >=
fn is_negation_of(condition: &RValue, original: &RValue) -> bool {
    // Case 1: condition is `not original`
    if let RValue::Unary(unary) = condition {
        if unary.operation == UnaryOperation::Not {
            return rvalues_equal(&unary.value, original);
        }
    }

    // Case 2: Both are binary comparisons with opposite operators
    if let (RValue::Binary(cond_bin), RValue::Binary(orig_bin)) = (condition, original) {
        // Check if operands are the same
        let same_operands = rvalues_equal(&cond_bin.left, &orig_bin.left)
            && rvalues_equal(&cond_bin.right, &orig_bin.right);

        if same_operands {
            // Check if operations are negations of each other
            let is_negated = matches!(
                (cond_bin.operation, orig_bin.operation),
                (BinaryOperation::Equal, BinaryOperation::NotEqual)
                    | (BinaryOperation::NotEqual, BinaryOperation::Equal)
                    | (BinaryOperation::LessThan, BinaryOperation::GreaterThanOrEqual)
                    | (BinaryOperation::GreaterThanOrEqual, BinaryOperation::LessThan)
                    | (BinaryOperation::LessThanOrEqual, BinaryOperation::GreaterThan)
                    | (BinaryOperation::GreaterThan, BinaryOperation::LessThanOrEqual)
            );

            if is_negated {
                return true;
            }
        }
    }

    false
}

/// Extract the new value from an and-chain if statement
fn extract_and_chain_value(statement: &Statement) -> RValue {
    let if_stat = statement.as_if().unwrap();
    let then_block = if_stat.then_block.lock();
    let then_assign = then_block[0].as_assign().unwrap();
    then_assign.right[0].clone()
}

/// Extract conditions from a nested if-chain pattern.
///
/// Pattern:
/// ```lua
/// if cond1 then
///     v = false  -- optional
///     if cond2 then
///         v = false  -- optional
///         if cond3 then
///             v = finalExpr
///         end
///     end
/// end
/// ```
///
/// Returns (conditions, final_expr) where conditions is [cond1, cond2, cond3, ...]
/// and final_expr is the value assigned in the innermost if.
fn extract_nested_and_chain(
    statement: &Statement,
    target_local: &RcLocal,
) -> Option<(Vec<RValue>, RValue)> {
    let if_stat = statement.as_if()?;

    // Must have empty else branch
    if !if_stat.else_block.lock().is_empty() {
        return None;
    }

    let mut conditions = vec![if_stat.condition.clone()];
    let then_block = if_stat.then_block.lock();

    // The then-block can have:
    // 1. Just another if statement (nested chain)
    // 2. An assignment to false followed by another if statement
    // 3. Just an assignment (the final value)

    let mut statements = then_block.0.iter().peekable();

    // Skip any assignments to false
    while let Some(stmt) = statements.peek() {
        if let Some(assign) = stmt.as_assign() {
            if assign.left.len() == 1 && assign.right.len() == 1 {
                if let Some(local) = assign.left[0].as_local() {
                    let same_local = local == target_local
                        || (local.name().is_some() && local.name() == target_local.name());
                    if same_local {
                        if matches!(&assign.right[0], RValue::Literal(Literal::Boolean(false))) {
                            statements.next();
                            continue;
                        }
                    }
                }
            }
        }
        break;
    }

    // Now we should have either an if statement or a final assignment
    let remaining: Vec<_> = statements.collect();

    if remaining.len() == 1 {
        // Could be a nested if or a final assignment
        if let Some(nested_if) = remaining[0].as_if() {
            // Recursively extract from nested if
            let nested_stmt = Statement::If(nested_if.clone());
            if let Some((nested_conditions, final_expr)) =
                extract_nested_and_chain(&nested_stmt, target_local)
            {
                conditions.extend(nested_conditions);
                return Some((conditions, final_expr));
            }
        } else if let Some(assign) = remaining[0].as_assign() {
            // This is the final assignment
            if assign.left.len() == 1 && assign.right.len() == 1 {
                if let Some(local) = assign.left[0].as_local() {
                    let same_local = local == target_local
                        || (local.name().is_some() && local.name() == target_local.name());
                    if same_local {
                        // Don't match if final value is also false (that would be pointless)
                        if !matches!(&assign.right[0], RValue::Literal(Literal::Boolean(false))) {
                            return Some((conditions, assign.right[0].clone()));
                        }
                    }
                }
            }
        }
    }

    None
}

/// Extract the pattern for expression-initialized and-chain.
///
/// Pattern:
/// ```lua
/// local v = expr1    -- some expression (not just false)
/// if v then          -- condition is the local itself
///     v = false      -- reset to false
///     if cond2 then
///         v = finalExpr
///     end
/// end
/// ```
///
/// Returns (conditions, final_expr) where conditions is [cond2, cond3, ...]
/// (the outer "if v" is implicit since v = expr1 is used directly)
fn extract_expr_and_chain(
    statement: &Statement,
    target_local: &RcLocal,
) -> Option<(Vec<RValue>, RValue)> {
    let if_stat = statement.as_if()?;

    // Must have empty else branch
    if !if_stat.else_block.lock().is_empty() {
        return None;
    }

    // Condition must be the local itself: `if v then`
    let cond_local = if_stat.condition.as_local()?;
    let same = cond_local == target_local
        || (cond_local.name().is_some() && cond_local.name() == target_local.name());
    if !same {
        return None;
    }

    let then_block = if_stat.then_block.lock();

    // The then-block should have exactly 2 statements:
    // 1. v = false (reset)
    // 2. if cond2 then ... end (nested chain)
    // We require exactly 2 to avoid issues with intermediate temporaries
    if then_block.len() != 2 {
        return None;
    }

    let mut statements = then_block.0.iter();

    // First statement should be v = false
    let first = statements.next()?;
    if let Some(assign) = first.as_assign() {
        if assign.left.len() == 1 && assign.right.len() == 1 {
            if let Some(local) = assign.left[0].as_local() {
                let same_local = local == target_local
                    || (local.name().is_some() && local.name() == target_local.name());
                if !same_local || !matches!(&assign.right[0], RValue::Literal(Literal::Boolean(false))) {
                    return None;
                }
            } else {
                return None;
            }
        } else {
            return None;
        }
    } else {
        return None;
    }

    // Second statement should be a nested if
    let second = statements.next()?;

    // Extract from the nested if using the existing function
    extract_nested_and_chain(second, target_local)
}

/// Extract the pattern for true initialization with conditional reassignment.
///
/// Pattern:
/// ```lua
/// local v = true
/// if cond then
///     v = expr
/// end
/// ```
///
/// Returns Some((condition, new_value)) if pattern matches.
fn extract_true_or_chain(
    statement: &Statement,
    target_local: &RcLocal,
) -> Option<(RValue, RValue)> {
    let if_stat = statement.as_if()?;

    // Must have empty else branch
    if !if_stat.else_block.lock().is_empty() {
        return None;
    }

    // Then branch must have exactly one assignment
    let then_block = if_stat.then_block.lock();
    if then_block.len() != 1 {
        return None;
    }

    let then_assign = then_block[0].as_assign()?;

    // Must assign to single local
    if then_assign.left.len() != 1 || then_assign.right.len() != 1 {
        return None;
    }

    let assign_local = then_assign.left[0].as_local()?;

    // Must assign to the same local
    let same_local = assign_local == target_local
        || (assign_local.name().is_some() && assign_local.name() == target_local.name());
    if !same_local {
        return None;
    }

    // Don't match if the new value is also true (that would be pointless)
    if matches!(&then_assign.right[0], RValue::Literal(Literal::Boolean(true))) {
        return None;
    }

    Some((if_stat.condition.clone(), then_assign.right[0].clone()))
}

/// Collapse multi-return assignment patterns.
///
/// Transforms patterns like:
///   local v1, v2 = call()
///   target1 = v1
///   target2 = v2
///
/// Into:
///   target1, target2 = call()
///
/// This handles the common case where multiple return values from a call
/// are assigned to temporary locals and then immediately assigned to their
/// final destinations (fields, table indices, etc.)
fn collapse_multi_return_assignments(block: &mut Block) {
    let mut i = 0;
    while i < block.len() {
        // First, recurse into nested blocks
        match &mut block[i] {
            Statement::If(if_stat) => {
                collapse_multi_return_assignments(&mut if_stat.then_block.lock());
                collapse_multi_return_assignments(&mut if_stat.else_block.lock());
            }
            Statement::While(while_stat) => {
                collapse_multi_return_assignments(&mut while_stat.block.lock());
            }
            Statement::Repeat(repeat_stat) => {
                collapse_multi_return_assignments(&mut repeat_stat.block.lock());
            }
            Statement::NumericFor(for_stat) => {
                collapse_multi_return_assignments(&mut for_stat.block.lock());
            }
            Statement::GenericFor(for_stat) => {
                collapse_multi_return_assignments(&mut for_stat.block.lock());
            }
            _ => {}
        }

        // Check for closures in RValues
        for rvalue in block[i].rvalues_mut() {
            collapse_multi_return_in_rvalue(rvalue);
        }

        // Look for pattern: local v1, v2, ... = call()
        let Some(assign) = block[i].as_assign() else {
            i += 1;
            continue;
        };

        // Must have multiple locals on the left and a single call on the right
        if assign.left.len() < 2 || assign.right.len() != 1 {
            i += 1;
            continue;
        }

        // Get all left-hand side locals
        let locals: Vec<_> = assign.left.iter()
            .filter_map(|lv| lv.as_local())
            .cloned()
            .collect();

        if locals.len() != assign.left.len() {
            // Not all LHS are locals (could be globals/indexes)
            i += 1;
            continue;
        }

        // Right side must be a call (or method call), possibly wrapped in Select for multi-return
        let is_call = matches!(&assign.right[0],
            RValue::Call(_) |
            RValue::MethodCall(_) |
            RValue::Select(Select::Call(_)) |
            RValue::Select(Select::MethodCall(_))
        );
        if !is_call {
            i += 1;
            continue;
        }

        // Check if there's at least one unnamed temporary that could be collapsed
        let has_unnamed = locals.iter().any(|l| {
            l.name().map_or(true, |n| is_unnamed_temporary(&n)) && l.name().as_deref() != Some("_")
        });
        if !has_unnamed {
            i += 1;
            continue;
        }

        // Now look ahead for assignments that use each local.
        // For "_" locals, keep them as "_".
        // For ALL other locals (named or unnamed), search for a subsequent simple assignment
        // that uses them. If found AND the local is only used in that assignment, collapse.
        let num_locals = locals.len();
        let mut targets: Vec<LValue> = Vec::with_capacity(num_locals);
        let mut stmts_to_remove: FxHashSet<usize> = FxHashSet::default();
        let mut any_collapsed = false;
        let search_window = num_locals + 2; // Search a bit beyond the number of locals

        // Collect all reads of locals in subsequent statements to check for single-use
        let mut local_read_counts: FxHashMap<String, usize> = FxHashMap::default();
        for offset in 0..search_window {
            let search_idx = i + 1 + offset;
            if search_idx >= block.len() {
                break;
            }
            let Some(search_assign) = block[search_idx].as_assign() else {
                break;
            };
            // Count how many times each local is read in the RHS
            for rv in &search_assign.right {
                for read_local in rv.values_read() {
                    if let Some(name) = read_local.name() {
                        *local_read_counts.entry(name).or_insert(0) += 1;
                    }
                }
            }
        }

        for local in locals.iter() {
            let is_discard = local.name().as_deref() == Some("_");

            if is_discard {
                // Keep "_" as-is
                targets.push(LValue::Local(local.clone()));
                continue;
            }

            // Search for a subsequent assignment to collapse
            let mut found_target: Option<LValue> = None;
            let mut found_idx: Option<usize> = None;

            for offset in 0..search_window {
                let search_idx = i + 1 + offset;
                if search_idx >= block.len() {
                    break;
                }
                if stmts_to_remove.contains(&search_idx) {
                    // Already marked for removal by another local
                    continue;
                }

                let Some(search_assign) = block[search_idx].as_assign() else {
                    // Non-assignment statement - stop searching
                    break;
                };

                // Must be single assignment: target = local
                if search_assign.left.len() != 1 || search_assign.right.len() != 1 {
                    continue;
                }

                // Right side must be this local
                let Some(rhs_local) = search_assign.right[0].as_local() else {
                    continue;
                };

                let same = rhs_local == local
                    || (rhs_local.name().is_some() && rhs_local.name() == local.name());
                if same {
                    // For named locals, only collapse if it's only used once (in this assignment)
                    let is_unnamed_temp = local.name().map_or(true, |n| is_unnamed_temporary(&n));
                    if !is_unnamed_temp {
                        // Named local - check if it's only used once
                        let read_count = local.name()
                            .and_then(|n| local_read_counts.get(&n))
                            .copied()
                            .unwrap_or(0);
                        if read_count != 1 {
                            // Named local used multiple times - don't collapse
                            break;
                        }
                    }
                    found_target = Some(search_assign.left[0].clone());
                    found_idx = Some(search_idx);
                    break;
                }
            }

            if let (Some(target), Some(idx)) = (found_target, found_idx) {
                targets.push(target);
                stmts_to_remove.insert(idx);
                any_collapsed = true;
            } else {
                // Not found - keep the local as-is
                targets.push(LValue::Local(local.clone()));
            }
        }

        // Only proceed if we actually collapsed something
        if !any_collapsed {
            i += 1;
            continue;
        }

        // Extract the call RValue
        let Statement::Assign(assign) = &mut block[i] else {
            unreachable!()
        };
        let call_rvalue = assign.right[0].clone();

        // Create the collapsed assignment: target1, target2, ... = call()
        let collapsed = Assign::new(targets, vec![call_rvalue]);

        // Replace the original multi-local assignment with the collapsed one
        block.0[i] = collapsed.into();

        // Remove the matched assignment statements in reverse order
        let mut indices: Vec<_> = stmts_to_remove.into_iter().collect();
        indices.sort_by(|a, b| b.cmp(a)); // Sort descending
        for idx in indices {
            block.0.remove(idx);
        }

        i += 1;
    }
}

fn collapse_multi_return_in_rvalue(rvalue: &mut RValue) {
    if let RValue::Closure(closure) = rvalue {
        collapse_multi_return_assignments(&mut closure.function.lock().body);
    }

    for rv in rvalue.rvalues_mut() {
        collapse_multi_return_in_rvalue(rv);
    }
}

/// Remove dead store patterns (no-op assignments).
///
/// Transforms patterns like:
///   local v5_ = maxXi
///   maxXi = v5_           -- dead store: assigns maxXi to itself via temp
///
/// Into:
///   local v5_ = maxXi     -- kept (may be used elsewhere)
///
/// Also removes completely dead round-trip patterns:
///   local v5_ = x
///   x = v5_
///   (where v5_ is only used in the assignment back to x)
///
/// And removes self-assignments:
///   x = x
pub fn remove_dead_stores(block: &mut Block) {
    let mut remover = DeadStoreRemover::default();
    remover.process_block(block);
}

#[derive(Default)]
struct DeadStoreRemover {
    // Track: temp -> (source_local, definition_index)
    // When we see `temp = source`, we record it here
    temp_sources: FxHashMap<RcLocal, (RcLocal, usize)>,
}

impl DeadStoreRemover {
    fn process_block(&mut self, block: &mut Block) {
        // First pass: collect temp assignments and identify dead stores
        let mut to_remove: Vec<usize> = Vec::new();

        // Clear tracking at block start
        self.temp_sources.clear();

        let mut i = 0;
        while i < block.len() {
            // Recurse into nested blocks first
            match &mut block[i] {
                Statement::If(if_stat) => {
                    // Save state, process nested, restore
                    let saved = std::mem::take(&mut self.temp_sources);
                    self.process_block(&mut if_stat.then_block.lock());
                    self.temp_sources.clear();
                    self.process_block(&mut if_stat.else_block.lock());
                    self.temp_sources = saved;
                }
                Statement::While(while_stat) => {
                    let saved = std::mem::take(&mut self.temp_sources);
                    self.process_block(&mut while_stat.block.lock());
                    self.temp_sources = saved;
                }
                Statement::Repeat(repeat_stat) => {
                    let saved = std::mem::take(&mut self.temp_sources);
                    self.process_block(&mut repeat_stat.block.lock());
                    self.temp_sources = saved;
                }
                Statement::NumericFor(for_stat) => {
                    let saved = std::mem::take(&mut self.temp_sources);
                    self.process_block(&mut for_stat.block.lock());
                    self.temp_sources = saved;
                }
                Statement::GenericFor(for_stat) => {
                    let saved = std::mem::take(&mut self.temp_sources);
                    self.process_block(&mut for_stat.block.lock());
                    self.temp_sources = saved;
                }
                _ => {}
            }

            // Process closures in RValues
            for rvalue in block[i].rvalues_mut() {
                self.process_closures_in_rvalue(rvalue);
            }

            // Check for self-assignment: x = x
            if let Some(assign) = block[i].as_assign() {
                if assign.left.len() == 1 && assign.right.len() == 1 {
                    if let LValue::Local(lhs) = &assign.left[0] {
                        if let RValue::Local(rhs) = &assign.right[0] {
                            // Self-assignment check
                            let same = lhs == rhs
                                || (lhs.name().is_some() && lhs.name() == rhs.name());
                            if same {
                                to_remove.push(i);
                                i += 1;
                                continue;
                            }
                        }
                    }
                }
            }

            // Check for round-trip dead store: temp = source; ... ; source = temp
            if let Some(assign) = block[i].as_assign() {
                if assign.left.len() == 1 && assign.right.len() == 1 {
                    if let LValue::Local(lhs) = &assign.left[0] {
                        if let RValue::Local(rhs) = &assign.right[0] {
                            // Check if rhs is a temp that was assigned from lhs
                            let found_source = self.find_temp_source(rhs);
                            if let Some((source, _def_idx)) = found_source {
                                // Check if lhs is the original source
                                let same = &source == lhs
                                    || (source.name().is_some() && source.name() == lhs.name());
                                if same {
                                    // This is a dead store: source = temp where temp came from source
                                    to_remove.push(i);
                                    i += 1;
                                    continue;
                                }
                            }
                        }
                    }
                }
            }

            // Track new temp assignments: temp = source
            if let Some(assign) = block[i].as_assign() {
                if assign.left.len() == 1 && assign.right.len() == 1 {
                    if let LValue::Local(lhs) = &assign.left[0] {
                        // Only track unnamed temporaries
                        let is_unnamed = lhs.name().map_or(true, |n| is_unnamed_temporary(&n));
                        if is_unnamed {
                            if let RValue::Local(rhs) = &assign.right[0] {
                                // Record: temp -> source
                                self.temp_sources.insert(lhs.clone(), (rhs.clone(), i));
                            }
                        }
                    }
                }
            }

            // Invalidate temps that are reassigned
            if let Some(assign) = block[i].as_assign() {
                for lv in &assign.left {
                    if let LValue::Local(local) = lv {
                        // Remove any temp whose source is being reassigned
                        self.temp_sources.retain(|_, (src, _)| {
                            src != local
                                && !(src.name().is_some() && src.name() == local.name())
                        });
                    }
                }
            }

            i += 1;
        }

        // Remove dead stores in reverse order to preserve indices
        for idx in to_remove.into_iter().rev() {
            block.0.remove(idx);
        }
    }

    fn find_temp_source(&self, temp: &RcLocal) -> Option<(RcLocal, usize)> {
        // Look up by identity first
        if let Some((src, idx)) = self.temp_sources.get(temp) {
            return Some((src.clone(), *idx));
        }
        // Then by name for SSA versions
        if let Some(name) = temp.name() {
            for (t, (src, idx)) in &self.temp_sources {
                if t.name().as_ref() == Some(&name) {
                    return Some((src.clone(), *idx));
                }
            }
        }
        None
    }

    fn process_closures_in_rvalue(&mut self, rvalue: &mut RValue) {
        if let RValue::Closure(closure) = rvalue {
            // Closures have their own scope
            let mut nested = DeadStoreRemover::default();
            nested.process_block(&mut closure.function.lock().body);
        }

        for rv in rvalue.rvalues_mut() {
            self.process_closures_in_rvalue(rv);
        }
    }
}

/// Remove dead locals - locals that are assigned but never read.
///
/// This handles patterns like:
/// ```lua
/// local v8_ = numKeys   -- assigned but never read
/// for i = 2, numKeys do -- uses numKeys directly, not v8_
/// ```
///
/// The local `v8_` is completely dead and can be removed.
pub fn remove_dead_locals(block: &mut Block) {
    // First, collect all reads and writes across the entire block (including nested)
    let mut reads: FxHashSet<RcLocal> = FxHashSet::default();
    let mut writes: FxHashMap<RcLocal, Vec<usize>> = FxHashMap::default();

    collect_local_usage(block, &mut reads, &mut writes, 0);

    // Find locals that are written but never read
    let dead_locals: FxHashSet<_> = writes.keys()
        .filter(|local| {
            // Only remove unnamed temporaries
            local.name().map_or(true, |n| is_unnamed_temporary(&n))
                && !reads.contains(*local)
                // Also check by name for SSA versions
                && !reads.iter().any(|r| r.name().is_some() && r.name() == local.name())
        })
        .cloned()
        .collect();

    if dead_locals.is_empty() {
        return;
    }

    // Remove statements that only write to dead locals
    remove_dead_local_assignments(block, &dead_locals);
}

fn collect_local_usage(
    block: &Block,
    reads: &mut FxHashSet<RcLocal>,
    writes: &mut FxHashMap<RcLocal, Vec<usize>>,
    base_index: usize,
) {
    for (i, stmt) in block.iter().enumerate() {
        // Collect reads from this statement
        for local in stmt.values_read() {
            reads.insert(local.clone());
        }

        // Collect writes from this statement
        for local in stmt.values_written() {
            writes.entry(local.clone()).or_default().push(base_index + i);
        }

        // Recurse into nested blocks
        match stmt {
            Statement::If(if_stat) => {
                collect_local_usage(&if_stat.then_block.lock(), reads, writes, 0);
                collect_local_usage(&if_stat.else_block.lock(), reads, writes, 0);
            }
            Statement::While(while_stat) => {
                // Condition is also a read
                for local in while_stat.condition.values_read() {
                    reads.insert(local.clone());
                }
                collect_local_usage(&while_stat.block.lock(), reads, writes, 0);
            }
            Statement::Repeat(repeat_stat) => {
                for local in repeat_stat.condition.values_read() {
                    reads.insert(local.clone());
                }
                collect_local_usage(&repeat_stat.block.lock(), reads, writes, 0);
            }
            Statement::NumericFor(for_stat) => {
                collect_local_usage(&for_stat.block.lock(), reads, writes, 0);
            }
            Statement::GenericFor(for_stat) => {
                collect_local_usage(&for_stat.block.lock(), reads, writes, 0);
            }
            _ => {}
        }

        // Recurse into closures
        for rvalue in stmt.rvalues() {
            collect_local_usage_in_rvalue(rvalue, reads, writes);
        }
    }
}

fn collect_local_usage_in_rvalue(
    rvalue: &RValue,
    reads: &mut FxHashSet<RcLocal>,
    writes: &mut FxHashMap<RcLocal, Vec<usize>>,
) {
    if let RValue::Closure(closure) = rvalue {
        collect_local_usage(&closure.function.lock().body, reads, writes, 0);
    }

    for rv in rvalue.rvalues() {
        collect_local_usage_in_rvalue(rv, reads, writes);
    }
}

fn remove_dead_local_assignments(block: &mut Block, dead_locals: &FxHashSet<RcLocal>) {
    // Remove assignments to dead locals
    block.retain(|stmt| {
        if let Some(assign) = stmt.as_assign() {
            // Check if this assignment ONLY writes to dead locals
            // and the RHS has no side effects
            if assign.left.len() == 1
                && !assign.right.iter().any(|r| r.has_side_effects())
            {
                if let Some(local) = assign.left[0].as_local() {
                    let is_dead = dead_locals.contains(local)
                        || dead_locals.iter().any(|d| d.name().is_some() && d.name() == local.name());
                    if is_dead {
                        return false; // Remove this statement
                    }
                }
            }
        }
        true
    });

    // Recurse into nested blocks
    for stmt in block.iter_mut() {
        match stmt {
            Statement::If(if_stat) => {
                remove_dead_local_assignments(&mut if_stat.then_block.lock(), dead_locals);
                remove_dead_local_assignments(&mut if_stat.else_block.lock(), dead_locals);
            }
            Statement::While(while_stat) => {
                remove_dead_local_assignments(&mut while_stat.block.lock(), dead_locals);
            }
            Statement::Repeat(repeat_stat) => {
                remove_dead_local_assignments(&mut repeat_stat.block.lock(), dead_locals);
            }
            Statement::NumericFor(for_stat) => {
                remove_dead_local_assignments(&mut for_stat.block.lock(), dead_locals);
            }
            Statement::GenericFor(for_stat) => {
                remove_dead_local_assignments(&mut for_stat.block.lock(), dead_locals);
            }
            _ => {}
        }

        // Process closures
        for rvalue in stmt.rvalues_mut() {
            remove_dead_locals_in_closures(rvalue, dead_locals);
        }
    }
}

fn remove_dead_locals_in_closures(rvalue: &mut RValue, dead_locals: &FxHashSet<RcLocal>) {
    if let RValue::Closure(closure) = rvalue {
        remove_dead_local_assignments(&mut closure.function.lock().body, dead_locals);
    }

    for rv in rvalue.rvalues_mut() {
        remove_dead_locals_in_closures(rv, dead_locals);
    }
}

/// Fix broken swap patterns that the decompiler generates incorrectly.
///
/// Pattern (broken):
/// ```lua
/// local temp = a     -- (0) temp gets a's value
/// a = b              -- (1) a gets b's value
/// temp = b           -- (2) temp gets b's value (BUG - overwrites saved a!)
/// b = a              -- (3) b gets a (which is b now)
/// ```
///
/// This is transformed into:
/// ```lua
/// a, b = b, a
/// ```
fn fix_broken_swap_patterns(block: &mut Block) {
    let mut i = 0;
    while i + 3 < block.len() {
        // Check for the 4-statement broken swap pattern
        if let Some((var_a, var_b)) = detect_broken_swap(&block[i..i + 4]) {
            // Remove statements 1, 2, 3 (keeping index i for the swap)
            block.0.remove(i + 3);
            block.0.remove(i + 2);
            block.0.remove(i + 1);

            // Replace statement 0 with the multi-assignment swap
            // For: temp = a; a = b; temp = b; b = a
            // We want: a, b = b, a
            let swap_assign = Assign::new(
                vec![var_a.clone().into(), var_b.clone().into()],
                vec![RValue::Local(var_b.clone()), RValue::Local(var_a.clone())],
            );
            block[i] = swap_assign.into();

            // Don't increment - check from same position in case there are more
            continue;
        }

        i += 1;
    }

    // Recurse into nested blocks
    for stmt in block.iter_mut() {
        match stmt {
            Statement::If(if_stat) => {
                fix_broken_swap_patterns(&mut if_stat.then_block.lock());
                fix_broken_swap_patterns(&mut if_stat.else_block.lock());
            }
            Statement::While(while_stat) => {
                fix_broken_swap_patterns(&mut while_stat.block.lock());
            }
            Statement::Repeat(repeat_stat) => {
                fix_broken_swap_patterns(&mut repeat_stat.block.lock());
            }
            Statement::NumericFor(for_stat) => {
                fix_broken_swap_patterns(&mut for_stat.block.lock());
            }
            Statement::GenericFor(for_stat) => {
                fix_broken_swap_patterns(&mut for_stat.block.lock());
            }
            _ => {}
        }

        // Process closures
        for rvalue in stmt.rvalues_mut() {
            fix_broken_swap_in_closures(rvalue);
        }
    }
}

fn fix_broken_swap_in_closures(rvalue: &mut RValue) {
    if let RValue::Closure(closure) = rvalue {
        fix_broken_swap_patterns(&mut closure.function.lock().body);
    }

    for rv in rvalue.rvalues_mut() {
        fix_broken_swap_in_closures(rv);
    }
}

/// Detect the broken swap pattern in 4 consecutive statements.
///
/// Pattern:
/// ```lua
/// local temp = a     -- stmt[0]: temp = a
/// a = b              -- stmt[1]: a = b
/// temp = b           -- stmt[2]: temp = b (the bug - should not exist)
/// b = a              -- stmt[3]: b = a
/// ```
///
/// Returns Some((a, b)) if the pattern matches, where a and b are the variables being swapped.
fn detect_broken_swap(stmts: &[Statement]) -> Option<(RcLocal, RcLocal)> {
    if stmts.len() < 4 {
        return None;
    }

    // stmt[0]: local temp = a
    let stmt0 = stmts[0].as_assign()?;
    if stmt0.left.len() != 1 || stmt0.right.len() != 1 {
        return None;
    }
    let temp = stmt0.left[0].as_local()?;
    // Only match unnamed temporaries
    if !temp.name().map_or(true, |n| is_unnamed_temporary(&n)) {
        return None;
    }
    let var_a = stmt0.right[0].as_local()?;

    // stmt[1]: a = b
    let stmt1 = stmts[1].as_assign()?;
    if stmt1.left.len() != 1 || stmt1.right.len() != 1 {
        return None;
    }
    let stmt1_left = stmt1.left[0].as_local()?;
    let var_b = stmt1.right[0].as_local()?;
    // stmt1_left should be var_a
    if !locals_equal(stmt1_left, var_a) {
        return None;
    }

    // stmt[2]: temp = b (the broken reassignment)
    let stmt2 = stmts[2].as_assign()?;
    if stmt2.left.len() != 1 || stmt2.right.len() != 1 {
        return None;
    }
    let stmt2_left = stmt2.left[0].as_local()?;
    let stmt2_right = stmt2.right[0].as_local()?;
    // stmt2_left should be temp, stmt2_right should be var_b (or var_a after stmt1)
    if !locals_equal(stmt2_left, temp) {
        return None;
    }
    // stmt2_right could be var_b (before swap) or the new value of var_a (which is var_b)
    // In the broken pattern, it's typically var_b or the same as stmt1's right
    if !locals_equal(stmt2_right, var_b) && !locals_equal(stmt2_right, var_a) {
        return None;
    }

    // stmt[3]: b = a (but a is now b, so this makes b = b effectively)
    let stmt3 = stmts[3].as_assign()?;
    if stmt3.left.len() != 1 || stmt3.right.len() != 1 {
        return None;
    }
    let stmt3_left = stmt3.left[0].as_local()?;
    let stmt3_right = stmt3.right[0].as_local()?;
    // stmt3_left should be var_b, stmt3_right should be var_a (or temp in a correct swap)
    if !locals_equal(stmt3_left, var_b) {
        return None;
    }
    // stmt3_right should be var_a (the now-corrupted value) or temp
    if !locals_equal(stmt3_right, var_a) && !locals_equal(stmt3_right, temp) {
        return None;
    }

    Some((var_a.clone(), var_b.clone()))
}

/// Check if two locals are equal (by identity or by name for SSA versions)
fn locals_equal(a: &RcLocal, b: &RcLocal) -> bool {
    a == b || (a.name().is_some() && a.name() == b.name())
}

/// Coalesce reassignment patterns where an unnamed temporary is assigned the result
/// of a call that takes a named variable as its only/first argument.
///
/// Pattern:
/// ```lua
/// local named = expr           -- (i) named variable with a real name
/// local v12_ = call(named)     -- (j) unnamed temp assigned call result using named
/// ... uses of v12_ ...
/// ```
///
/// Transforms to:
/// ```lua
/// local named = expr
/// named = call(named)          -- reassign to named instead of creating temp
/// ... uses of named ...
/// ```
///
/// This is useful for patterns like:
/// ```lua
/// local year = match(...)
/// local v12_ = tonumber(year)
/// ```
/// Becoming:
/// ```lua
/// local year = match(...)
/// year = tonumber(year)
/// ```
fn coalesce_reassignments(block: &mut Block) {
    // First pass: identify candidates
    // A candidate is: local temp = call(named) where:
    // - temp is unnamed (starts with v followed by digits and _)
    // - named has a real name
    // - named is only used in this one place (the call argument)
    // - the call has exactly one argument that is `named`

    let mut replacements: Vec<(RcLocal, RcLocal)> = Vec::new(); // (temp, named) pairs
    let mut statements_to_modify: Vec<usize> = Vec::new();

    // Count uses of each named local in the block
    let mut local_uses: FxHashMap<String, usize> = FxHashMap::default();
    for stmt in block.iter() {
        for local in stmt.values_read() {
            if let Some(name) = local.name() {
                if !is_unnamed_temporary(&name) {
                    *local_uses.entry(name).or_insert(0) += 1;
                }
            }
        }
    }

    for (i, stmt) in block.iter().enumerate() {
        if let Some(assign) = stmt.as_assign() {
            // Check: single assignment to an unnamed local
            if assign.left.len() != 1 || assign.right.len() != 1 || !assign.prefix {
                continue;
            }

            let temp = match assign.left[0].as_local() {
                Some(l) => l,
                None => continue,
            };

            // temp must be unnamed
            let _temp_name = match temp.name() {
                Some(n) if is_unnamed_temporary(&n) => n,
                _ => continue,
            };

            // Right side must be a call or method call (may be wrapped in Select for multi-return)
            let call_args = match &assign.right[0] {
                RValue::Call(call) => &call.arguments,
                RValue::MethodCall(mc) => &mc.arguments,
                RValue::Select(Select::Call(call)) => &call.arguments,
                RValue::Select(Select::MethodCall(mc)) => &mc.arguments,
                _ => continue,
            };

            // Call must have at least one argument
            if call_args.is_empty() {
                continue;
            }

            // First argument must be a named local
            let named = match call_args[0].as_local() {
                Some(l) => l,
                None => continue,
            };

            let named_name = match named.name() {
                Some(n) if !is_unnamed_temporary(&n) => n,
                _ => continue,
            };

            // Check if named is only used once in this block (in this call)
            let uses = local_uses.get(&named_name).copied().unwrap_or(0);
            if uses != 1 {
                continue;
            }

            // This is a candidate for coalescing
            replacements.push((temp.clone(), named.clone()));
            statements_to_modify.push(i);
        }
    }

    // Second pass: apply replacements
    for (temp, named) in &replacements {
        // Replace all uses of temp with named throughout the block
        for stmt in block.iter_mut() {
            replace_local_in_statement(stmt, temp, named);
        }
    }

    // Third pass: change the assignment from "local temp = call(named)" to "named = call(named)"
    for &i in &statements_to_modify {
        if let Some(assign) = block[i].as_assign_mut() {
            // Change left side from temp to named
            if let Some((_, named)) = replacements.iter().find(|(t, _)| {
                assign.left[0].as_local().map(|l| l == t).unwrap_or(false)
            }) {
                assign.left[0] = named.clone().into();
                assign.prefix = false; // No longer a local declaration
            }
        }
    }

    // Recurse into nested blocks
    for stmt in block.iter_mut() {
        match stmt {
            Statement::If(if_stat) => {
                coalesce_reassignments(&mut if_stat.then_block.lock());
                coalesce_reassignments(&mut if_stat.else_block.lock());
            }
            Statement::While(while_stat) => {
                coalesce_reassignments(&mut while_stat.block.lock());
            }
            Statement::Repeat(repeat_stat) => {
                coalesce_reassignments(&mut repeat_stat.block.lock());
            }
            Statement::NumericFor(for_stat) => {
                coalesce_reassignments(&mut for_stat.block.lock());
            }
            Statement::GenericFor(for_stat) => {
                coalesce_reassignments(&mut for_stat.block.lock());
            }
            _ => {}
        }

        // Process closures
        for rvalue in stmt.rvalues_mut() {
            coalesce_reassignments_in_closures(rvalue);
        }
    }
}

fn coalesce_reassignments_in_closures(rvalue: &mut RValue) {
    if let RValue::Closure(closure) = rvalue {
        coalesce_reassignments(&mut closure.function.lock().body);
    }

    for rv in rvalue.rvalues_mut() {
        coalesce_reassignments_in_closures(rv);
    }
}

/// Replace all occurrences of `old` local with `new` local in a statement
fn replace_local_in_statement(stmt: &mut Statement, old: &RcLocal, new: &RcLocal) {
    // Replace in values read
    for local in stmt.values_read_mut() {
        if local == old || (local.name().is_some() && local.name() == old.name()) {
            *local = new.clone();
        }
    }

    // Replace in nested rvalues
    for rvalue in stmt.rvalues_mut() {
        replace_local_in_rvalue(rvalue, old, new);
    }

    // Recurse into nested statements
    match stmt {
        Statement::If(if_stat) => {
            replace_local_in_rvalue(&mut if_stat.condition, old, new);
            for s in if_stat.then_block.lock().iter_mut() {
                replace_local_in_statement(s, old, new);
            }
            for s in if_stat.else_block.lock().iter_mut() {
                replace_local_in_statement(s, old, new);
            }
        }
        Statement::While(while_stat) => {
            replace_local_in_rvalue(&mut while_stat.condition, old, new);
            for s in while_stat.block.lock().iter_mut() {
                replace_local_in_statement(s, old, new);
            }
        }
        Statement::Repeat(repeat_stat) => {
            replace_local_in_rvalue(&mut repeat_stat.condition, old, new);
            for s in repeat_stat.block.lock().iter_mut() {
                replace_local_in_statement(s, old, new);
            }
        }
        Statement::NumericFor(for_stat) => {
            for s in for_stat.block.lock().iter_mut() {
                replace_local_in_statement(s, old, new);
            }
        }
        Statement::GenericFor(for_stat) => {
            for s in for_stat.block.lock().iter_mut() {
                replace_local_in_statement(s, old, new);
            }
        }
        _ => {}
    }
}

fn replace_local_in_rvalue(rvalue: &mut RValue, old: &RcLocal, new: &RcLocal) {
    if let RValue::Local(local) = rvalue {
        if local == old || (local.name().is_some() && local.name() == old.name()) {
            *local = new.clone();
        }
    }

    for rv in rvalue.rvalues_mut() {
        replace_local_in_rvalue(rv, old, new);
    }
}

/// Simplify write-only locals to underscore placeholders.
///
/// Transforms patterns like:
/// ```lua
/// local v28_, v29_
/// v28_, rotY, v29_ = getWorldRotation(self.spot.node)
/// Simplifies __set_list patterns where an empty table is created, populated via __set_list,
/// and then immediately used (returned or assigned).
///
/// Pattern 1 - Return:
/// ```lua
/// local v35_ = {}
/// __set_list(v35_, 1, {a, b, c})
/// return v35_
/// ```
/// Becomes:
/// ```lua
/// return {a, b, c}
/// ```
///
/// Pattern 2 - Assignment:
/// ```lua
/// local v1_ = {}
/// __set_list(v1_, 1, {unpack(src.items)})
/// self.items = v1_
/// ```
/// Becomes:
/// ```lua
/// self.items = {unpack(src.items)}
/// ```
fn simplify_set_list_patterns(block: &mut Block) {
    // First recurse into nested blocks
    for stmt in block.0.iter_mut() {
        match stmt {
            Statement::If(if_stat) => {
                simplify_set_list_patterns(&mut if_stat.then_block.lock());
                simplify_set_list_patterns(&mut if_stat.else_block.lock());
            }
            Statement::While(while_stat) => {
                simplify_set_list_patterns(&mut while_stat.block.lock());
            }
            Statement::Repeat(repeat_stat) => {
                simplify_set_list_patterns(&mut repeat_stat.block.lock());
            }
            Statement::NumericFor(for_stat) => {
                simplify_set_list_patterns(&mut for_stat.block.lock());
            }
            Statement::GenericFor(for_stat) => {
                simplify_set_list_patterns(&mut for_stat.block.lock());
            }
            _ => {}
        }

        // Recurse into closures
        for rvalue in stmt.rvalues_mut() {
            simplify_set_list_in_closures(rvalue);
        }
    }

    let mut i = 0;
    while i + 2 < block.len() {
        // Look for: local v = {}
        let Some(assign1) = block[i].as_assign() else {
            i += 1;
            continue;
        };

        // Must be single local = empty table
        if assign1.left.len() != 1 || assign1.right.len() != 1 {
            i += 1;
            continue;
        }

        let Some(local) = assign1.left[0].as_local() else {
            i += 1;
            continue;
        };

        // Right side must be empty table literal
        let is_empty_table = matches!(&assign1.right[0], RValue::Table(tbl) if tbl.0.is_empty());
        if !is_empty_table {
            i += 1;
            continue;
        }

        // Check if it's an unnamed temp
        let is_unnamed = local.name().map_or(true, |n| is_unnamed_temporary(&n));
        if !is_unnamed {
            i += 1;
            continue;
        }

        // Look for: SetList statement (not a Call)
        let Some(set_list) = block[i + 1].as_set_list() else {
            i += 1;
            continue;
        };

        // Check if the SetList target is our local
        let same_local = &set_list.object_local == local
            || (set_list.object_local.name().is_some() && set_list.object_local.name() == local.name());
        if !same_local {
            i += 1;
            continue;
        }

        // Index should be 1
        if set_list.index != 1 {
            i += 1;
            continue;
        }

        // Build the table from SetList values
        // Table entries are (Option<RValue>, RValue) where None means array-style
        let mut table_entries: Vec<(Option<RValue>, RValue)> = Vec::new();
        for val in set_list.values.iter() {
            table_entries.push((None, val.clone()));
        }
        // Handle tail (vararg-like multi-return) - also array-style
        if let Some(tail) = &set_list.tail {
            table_entries.push((None, tail.clone()));
        }
        let values_rvalue = RValue::Table(crate::Table(table_entries));

        // Now check the third statement - either return v or target = v
        // Pattern 1: return v
        if let Some(ret) = block[i + 2].as_return() {
            if ret.values.len() == 1 {
                if let Some(ret_local) = ret.values[0].as_local() {
                    let same = ret_local == local
                        || (ret_local.name().is_some() && ret_local.name() == local.name());
                    if same {
                        // Transform: remove first two statements, change return to return {...}
                        block.0.remove(i); // Remove local v = {}
                        block.0.remove(i); // Remove SetList
                        // Modify return
                        if let Statement::Return(ret) = &mut block.0[i] {
                            ret.values = vec![values_rvalue];
                        }
                        continue; // Don't increment i, check the new statement at this position
                    }
                }
            }
        }

        // Pattern 2: target = v
        if let Some(assign3) = block[i + 2].as_assign() {
            if assign3.left.len() == 1 && assign3.right.len() == 1 {
                if let Some(rhs_local) = assign3.right[0].as_local() {
                    let same = rhs_local == local
                        || (rhs_local.name().is_some() && rhs_local.name() == local.name());
                    if same {
                        // Transform: remove first two statements, change assignment RHS to {...}
                        let target = assign3.left[0].clone();
                        block.0.remove(i); // Remove local v = {}
                        block.0.remove(i); // Remove SetList
                        // Modify assignment
                        if let Statement::Assign(assign) = &mut block.0[i] {
                            assign.left = vec![target];
                            assign.right = vec![values_rvalue];
                        }
                        continue;
                    }
                }
            }
        }

        i += 1;
    }
}

fn simplify_set_list_in_closures(rvalue: &mut RValue) {
    if let RValue::Closure(closure) = rvalue {
        simplify_set_list_patterns(&mut closure.function.lock().body);
    }

    for rv in rvalue.rvalues_mut() {
        simplify_set_list_in_closures(rv);
    }
}

/// ```
/// Into:
/// ```lua
/// _, rotY, _ = getWorldRotation(self.spot.node)
/// ```
///
/// This works by:
/// 1. Finding empty local declarations (prefix assignment with no RHS)
/// 2. Looking for the next assignment that writes to those locals
/// 3. Checking if those locals are never read in the entire block
/// 4. If so, removing the declaration and replacing the locals with underscore
fn simplify_write_only_locals(block: &mut Block) {
    // First, collect all locals that are read anywhere in the block
    let mut locals_read: FxHashSet<RcLocal> = FxHashSet::default();
    collect_locals_read(block, &mut locals_read);

    let mut i = 0;
    while i < block.len() {
        // Look for empty local declarations: `local v28_, v29_`
        let Some(assign) = block[i].as_assign() else {
            i += 1;
            continue;
        };

        if !assign.prefix || !assign.right.is_empty() {
            i += 1;
            continue;
        }

        // Collect the declared locals
        let declared_locals: Vec<RcLocal> = assign.left.iter()
            .filter_map(|lv| lv.as_local().cloned())
            .collect();

        if declared_locals.is_empty() {
            i += 1;
            continue;
        }

        // Check if ALL declared locals are unnamed temporaries and never read
        let all_write_only = declared_locals.iter().all(|local| {
            let is_unnamed = local.name()
                .map(|n| is_unnamed_temporary(&n))
                .unwrap_or(true);
            let never_read = !locals_read.contains(local);
            is_unnamed && never_read
        });

        if !all_write_only {
            i += 1;
            continue;
        }

        // Look for the next statement that assigns to these locals
        if i + 1 >= block.len() {
            i += 1;
            continue;
        }

        let Some(next_assign) = block[i + 1].as_assign() else {
            i += 1;
            continue;
        };

        // The next assignment should not be a prefix (not a local declaration)
        if next_assign.prefix {
            i += 1;
            continue;
        }

        // Check if this assignment writes to any of our declared locals
        let writes_to_declared: Vec<usize> = next_assign.left.iter()
            .enumerate()
            .filter_map(|(idx, lv)| {
                lv.as_local().and_then(|l| {
                    if declared_locals.iter().any(|dl| dl == l) {
                        Some(idx)
                    } else {
                        None
                    }
                })
            })
            .collect();

        if writes_to_declared.is_empty() {
            i += 1;
            continue;
        }

        // Create the underscore local
        let underscore = RcLocal::new(crate::Local::new(Some("_".to_string())));

        // Replace the write-only locals with underscore in the assignment
        if let Some(next_assign) = block[i + 1].as_assign_mut() {
            for &idx in &writes_to_declared {
                if let Some(local) = next_assign.left[idx].as_local() {
                    if declared_locals.iter().any(|dl| dl == local) {
                        next_assign.left[idx] = underscore.clone().into();
                    }
                }
            }
        }

        // Remove the empty declaration if all its locals are now replaced
        let all_replaced = declared_locals.iter().all(|dl| {
            writes_to_declared.iter().any(|&idx| {
                block[i + 1].as_assign()
                    .and_then(|a| a.left.get(idx))
                    .and_then(|lv| lv.as_local())
                    .map(|l| l.name() == Some("_".to_string()))
                    .unwrap_or(false)
            }) || !block[i + 1].as_assign()
                .map(|a| a.left.iter().any(|lv| lv.as_local().map(|l| l == dl).unwrap_or(false)))
                .unwrap_or(false)
        });

        if all_replaced {
            block.0.remove(i);
            // Don't increment i, we need to check the same position again
        } else {
            i += 1;
        }
    }

    // Recurse into nested blocks
    for stmt in block.iter_mut() {
        match stmt {
            Statement::If(if_stat) => {
                simplify_write_only_locals(&mut if_stat.then_block.lock());
                simplify_write_only_locals(&mut if_stat.else_block.lock());
            }
            Statement::While(while_stat) => {
                simplify_write_only_locals(&mut while_stat.block.lock());
            }
            Statement::Repeat(repeat_stat) => {
                simplify_write_only_locals(&mut repeat_stat.block.lock());
            }
            Statement::NumericFor(for_stat) => {
                simplify_write_only_locals(&mut for_stat.block.lock());
            }
            Statement::GenericFor(for_stat) => {
                simplify_write_only_locals(&mut for_stat.block.lock());
            }
            _ => {}
        }

        // Process closures
        for rvalue in stmt.rvalues_mut() {
            simplify_write_only_locals_in_closures(rvalue);
        }
    }
}

fn simplify_write_only_locals_in_closures(rvalue: &mut RValue) {
    if let RValue::Closure(closure) = rvalue {
        simplify_write_only_locals(&mut closure.function.lock().body);
    }

    for rv in rvalue.rvalues_mut() {
        simplify_write_only_locals_in_closures(rv);
    }
}

/// Collect all locals that are read in a block (recursively)
fn collect_locals_read(block: &Block, locals_read: &mut FxHashSet<RcLocal>) {
    for stmt in block.iter() {
        // Collect locals that are read (not written)
        for local in stmt.values_read() {
            locals_read.insert(local.clone());
        }

        // Recurse into nested blocks
        match stmt {
            Statement::If(if_stat) => {
                // The condition reads values
                for local in if_stat.condition.values_read() {
                    locals_read.insert(local.clone());
                }
                collect_locals_read(&if_stat.then_block.lock(), locals_read);
                collect_locals_read(&if_stat.else_block.lock(), locals_read);
            }
            Statement::While(while_stat) => {
                for local in while_stat.condition.values_read() {
                    locals_read.insert(local.clone());
                }
                collect_locals_read(&while_stat.block.lock(), locals_read);
            }
            Statement::Repeat(repeat_stat) => {
                for local in repeat_stat.condition.values_read() {
                    locals_read.insert(local.clone());
                }
                collect_locals_read(&repeat_stat.block.lock(), locals_read);
            }
            Statement::NumericFor(for_stat) => {
                collect_locals_read(&for_stat.block.lock(), locals_read);
            }
            Statement::GenericFor(for_stat) => {
                collect_locals_read(&for_stat.block.lock(), locals_read);
            }
            _ => {}
        }

        // Process closures
        for rvalue in stmt.rvalues() {
            collect_locals_read_in_rvalue(rvalue, locals_read);
        }
    }
}

fn collect_locals_read_in_rvalue(rvalue: &RValue, locals_read: &mut FxHashSet<RcLocal>) {
    if let RValue::Local(local) = rvalue {
        locals_read.insert(local.clone());
    }

    if let RValue::Closure(closure) = rvalue {
        collect_locals_read(&closure.function.lock().body, locals_read);
    }

    for rv in rvalue.rvalues() {
        collect_locals_read_in_rvalue(rv, locals_read);
    }
}
