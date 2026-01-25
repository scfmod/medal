use rustc_hash::{FxHashMap, FxHashSet};

use crate::{Block, LValue, LocalRw, RValue, RcLocal, Select, Statement, Traverse};

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
                if inline_candidates.contains(&local) {
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

            // Note: we DON'T clear immediate_inlines here anymore - they persist
            // until applied to a later statement in this block.
            // Only clear deferred_inlines when recursing (different scope).

            // Recurse into nested blocks and closures, passing down protected locals
            let saved_immediate = std::mem::take(&mut self.immediate_inlines);
            self.deferred_inlines.clear();
            self.recurse_into_statement(&mut block[i], depth, protected_locals);
            self.immediate_inlines = saved_immediate;

            i += 1;
        }

        self.immediate_inlines.clear();
        self.deferred_inlines.clear();
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
        // Then handle generic for loops (existing logic below)
        let mut i = 0;
        while i + 1 < block.len() {
            // Check if statement i is an assignment and i+1 is a GenericFor
            let can_inline = if let (
                Statement::Assign(assign),
                Statement::GenericFor(generic_for),
            ) = (&block[i], &block[i + 1])
            {
                // The assignment must have a call as its RHS (like pairs() or ipairs())
                // and the for-loop must use exactly the assigned locals
                // Note: calls that return multiple values are wrapped in Select::Call
                // Also include MethodCall for things like xmlFile:iterator()
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
        let mut uses: FxHashMap<RcLocal, usize> = FxHashMap::default();
        // Track locals used in while/repeat conditions - these should never be inlined
        // because the variable may be modified inside the loop body (which we don't track)
        let mut loop_condition_locals: FxHashSet<RcLocal> = FxHashSet::default();
        // Track locals used in nested blocks (if/while/for bodies) - we need to count
        // these uses to avoid incorrectly inlining variables that are used multiple times
        let mut nested_block_locals: FxHashSet<RcLocal> = FxHashSet::default();

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
                    // Also track locals used inside the while body
                    Self::collect_locals_in_block(&w.block.lock(), &mut nested_block_locals);
                }
                Statement::Repeat(r) => {
                    for local in r.condition.values_read() {
                        loop_condition_locals.insert(local.clone());
                    }
                    Self::collect_locals_in_block(&r.block.lock(), &mut nested_block_locals);
                }
                Statement::If(i) => {
                    // Track locals used inside if/else bodies
                    Self::collect_locals_in_block(&i.then_block.lock(), &mut nested_block_locals);
                    Self::collect_locals_in_block(&i.else_block.lock(), &mut nested_block_locals);
                }
                Statement::NumericFor(nf) => {
                    Self::collect_locals_in_block(&nf.block.lock(), &mut nested_block_locals);
                }
                Statement::GenericFor(gf) => {
                    Self::collect_locals_in_block(&gf.block.lock(), &mut nested_block_locals);
                }
                _ => {}
            }

            // Count uses (only in the same block, don't look into nested blocks)
            for local in statement.values_read() {
                *uses.entry(local.clone()).or_default() += 1;
            }
        }

        // Return locals that are defined once and used once, with appropriate conditions:
        // - Unnamed temporaries: always inline
        // - Named locals: only inline if assigned side-effect-free value (like field access)
        // - Never inline locals used in while/repeat conditions
        // - Never inline locals that are also used in nested blocks
        definitions
            .into_iter()
            .filter(|(local, (def_count, rvalue))| {
                let use_count = uses.get(local).copied().unwrap_or(0);
                if *def_count != 1 || use_count != 1 {
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

                // Never inline locals that are also used in nested blocks (if/while/for bodies)
                // because the apparent "single use" in the outer block may be accompanied by
                // additional uses inside the nested blocks
                if nested_block_locals.contains(local) {
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

    /// Collect all locals read in a block (including nested blocks) into the given set
    /// Collect all locals that are read OR written in a block (recursively).
    ///
    /// This includes both `values_read()` and `values_written()` to catch cases
    /// where a local is modified inside a nested block (like if-statements).
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
        _rvalue: &RValue,
        block: &Block,
        start_idx: usize,
    ) -> Option<usize> {
        // Look for the first statement that uses this local
        for j in (start_idx + 1)..block.len() {
            let statement = &block[j];

            // Check if this statement uses our local
            if self.is_single_use(local, statement) {
                return Some(j);
            }

            // Check if this is a safe intervening statement (side-effect-free assignment)
            // Safe: local = <side-effect-free expression>
            // Unsafe: anything with side effects (calls, etc.)
            if let Statement::Assign(assign) = statement {
                // Check if the RHS has side effects
                let rhs_has_side_effects = assign.right.iter().any(|r| has_side_effects(r));
                if rhs_has_side_effects {
                    // Can't safely reorder past a side-effect statement
                    return None;
                }
                // This is a safe intervening statement, continue looking
            } else {
                // Non-assignment statement - not safe to reorder past
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
            if candidates.contains(local) {
                if let Some(rvalue) = self.deferred_inlines.remove(local) {
                    to_inline.push((local.clone(), rvalue));
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
        let mut work: Vec<*mut RValue> = statement
            .rvalues_mut()
            .into_iter()
            .map(|r| r as *mut RValue)
            .collect();

        while let Some(ptr) = work.pop() {
            let rvalue = unsafe { &mut *ptr };

            if let RValue::Local(l) = rvalue {
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

    // Third try: general ternary with truthy then-value
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

        // Only simplify unnamed temporaries
        let is_unnamed = local.name().map_or(true, |n| is_unnamed_temporary(&n));
        if !is_unnamed {
            i += 1;
            continue;
        }

        let local = local.clone();
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
        let Statement::Assign(assign) = &block[i] else {
            unreachable!()
        };
        let initial_value = assign.right[0].clone();

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

    // Check if condition is the negation of initial_value
    // Case 1: initial_value is `x == y` and condition is `x ~= y`
    // Case 2: initial_value is `x ~= y` and condition is `x == y`
    // Case 3: condition is `not initial_value`
    if is_negation_of(&if_stat.condition, initial_value) {
        return Some(then_assign.right[0].clone());
    }

    None
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

        // All left-hand sides must be unnamed local temporaries
        let locals: Vec<_> = assign.left.iter()
            .filter_map(|lv| lv.as_local())
            .filter(|l| l.name().map_or(true, |n| is_unnamed_temporary(&n)))
            .cloned()
            .collect();

        if locals.len() != assign.left.len() {
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

        // Now look ahead for consecutive single assignments that use each local exactly once
        // in order: target1 = v1, target2 = v2, ...
        let num_locals = locals.len();
        let mut targets: Vec<LValue> = Vec::with_capacity(num_locals);
        let mut matched_count = 0;

        for (idx, local) in locals.iter().enumerate() {
            let next_idx = i + 1 + idx;
            if next_idx >= block.len() {
                break;
            }

            let Some(next_assign) = block[next_idx].as_assign() else {
                break;
            };

            // Must be single assignment: target = local
            if next_assign.left.len() != 1 || next_assign.right.len() != 1 {
                break;
            }

            // Right side must be this local
            let Some(rhs_local) = next_assign.right[0].as_local() else {
                break;
            };

            let same = rhs_local == local
                || (rhs_local.name().is_some() && rhs_local.name() == local.name());
            if !same {
                break;
            }

            targets.push(next_assign.left[0].clone());
            matched_count += 1;
        }

        // Only collapse if we matched ALL the locals
        if matched_count != num_locals {
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

        // Remove the individual assignments
        for _ in 0..matched_count {
            block.0.remove(i + 1);
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
