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

    /// Inline iterator expressions into generic for-loop headers.
    ///
    /// Transforms patterns like:
    ///   local v1_, v2_, v3_ = pairs(x)
    ///   for k, v in v1_, v2_, v3_ do
    /// Into:
    ///   for k, v in pairs(x) do
    ///
    /// This handles both unnamed temporaries and cases where names have leaked
    /// to the iterator state variables (which is a separate decompilation artifact).
    fn inline_iterator_into_for_loops(&self, block: &mut Block) {
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
