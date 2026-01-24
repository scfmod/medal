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
                        // Must inline into next statement only
                        if i + 1 < block.len() && self.is_single_use(&local, &block[i + 1]) {
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

            // Recurse into nested blocks and closures, passing down protected locals
            self.immediate_inlines.clear();
            self.deferred_inlines.clear();
            self.recurse_into_statement(&mut block[i], depth, protected_locals);

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
                let is_iterator_call = assign.right.len() == 1
                    && matches!(&assign.right[0], RValue::Call(_) | RValue::Select(Select::Call(_)));

                if is_iterator_call && assign.left.len() >= 1 && generic_for.right.len() == assign.left.len() {
                    // Check that all for-loop iterators are locals matching the assignment
                    let mut matches = true;
                    for (j, rv) in generic_for.right.iter().enumerate() {
                        if let RValue::Local(for_local) = rv {
                            if let LValue::Local(assign_local) = &assign.left[j] {
                                if for_local != assign_local {
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
                if *def_count != 1 || uses.get(local).copied().unwrap_or(0) != 1 {
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

                let is_unnamed = local.name().map_or(false, |n| is_unnamed_temporary(&n));
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
    fn collect_locals_in_block(block: &Block, locals: &mut FxHashSet<RcLocal>) {
        for statement in &block.0 {
            for local in statement.values_read() {
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

    /// Check if the local is used exactly once in the given statement
    fn is_single_use(&self, local: &RcLocal, statement: &Statement) -> bool {
        let reads = statement.values_read();
        reads.iter().filter(|l| **l == local).count() == 1
    }

    /// Apply immediate inlines (for side-effect RValues)
    fn apply_immediate_inlines(&mut self, statement: &mut Statement) {
        if self.immediate_inlines.is_empty() {
            return;
        }

        let inlines = std::mem::take(&mut self.immediate_inlines);
        for (local, rvalue) in inlines {
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

    // Check if the values are boolean literals
    let then_value = &then_assign.right[0];
    let else_value = &else_assign.right[0];

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
