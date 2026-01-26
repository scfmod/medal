# Medal - Luau Bytecode Decompiler

## Build Commands

```bash
# Build (debug mode - use this for development)
cargo +nightly build

# Run decompiler on a single file
./target/debug/luau-lifter /path/to/file.l64

# Dump bytecode for debugging
./target/debug/luau-lifter /path/to/file.l64 -d [optional_function_name]
```

## Test Scripts Location

FS25 scripts are in `/Users/kim/dev/fs25/dataS/scripts/`

Files are `.l64` format (encrypted Luau bytecode). Decompiled `.lua` files are also there.

## Counting Unnamed Temporaries

```bash
# Count unnamed temporaries in decompiled output
./target/debug/luau-lifter /path/to/file.l64 2>&1 | grep -oE ' v[0-9]+_' | wc -l

# Count across all decompiled lua files
grep -rn ' v[0-9]*_' /Users/kim/dev/fs25/dataS/scripts/**/*.lua 2>/dev/null | wc -l

# Classify remaining temporaries
grep -rhn ' v[0-9]*_' /Users/kim/dev/fs25/dataS/scripts/**/*.lua 2>/dev/null | while read line; do
  content=$(echo "$line" | sed 's/^[0-9]*://')
  # ... classification logic
done | sort | uniq -c | sort -rn
```

## Bulk Decompile All Scripts

```bash
# Copy built binary and run bulk decompile
cp /Users/kim/dev/fs25/medal/target/debug/luau-lifter /Users/kim/dev/fs25/fs-utils/bin/luau-lifter && \
LUAU_LIFTER_PATH=/Users/kim/dev/fs25/fs-utils/bin/luau-lifter /Users/kim/dev/fs25/fs-utils/bin/fs-luau-decompile -r /Users/kim/dev/fs25/dataS/scripts/
```

## Key Files

- `ast/src/inline_temporaries.rs` - Main inlining/cleanup pass
- `cfg/src/ssa/construct.rs` - SSA construction (copy propagation fix at line ~425)
- `cfg/src/ssa/destruct.rs` - SSA destruction (phi sequentialization)
- `luau-lifter/src/lifter.rs` - Bytecode to CFG lifting

## Current Status

Unnamed temporaries reduced from ~10,000 to ~152 (98.5% reduction).

Remaining categories (difficult to optimize):

- Control flow result variables (used across branches)
- For-loop bound preservation (semantically necessary)
- Variable shadowing preservation
- Complex boolean expressions used as conditions

## PR

PR #9 on scfmod/medal - pushed to Paint-a-Farm/medal fork, branch `inline-side-effect-lookahead`
