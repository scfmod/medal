use nom::{
    number::complete::{le_u32, le_u8},
    IResult,
};
use nom_leb128::leb128_usize;

use super::{
    constant::Constant,
    list::{parse_list, parse_list_len},
};

use crate::{instruction::*, op_code::OpCode};

/// Debug info for a local variable in Luau bytecode
#[derive(Debug, Clone)]
pub struct LocalDebugInfo {
    pub name_index: usize,  // Index into string table (1-based, 0 = no name)
    pub scope_start: usize, // PC where variable scope starts
    pub scope_end: usize,   // PC where variable scope ends
    pub register: u8,       // Register number this variable uses
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct Function {
    pub max_stack_size: u8,
    pub num_parameters: u8,
    pub num_upvalues: u8,
    pub is_vararg: bool,
    //pub instructions: Vec<u32>,
    pub instructions: Vec<Instruction>,
    pub constants: Vec<Constant>,
    pub functions: Vec<usize>,
    pub line_info: (usize, usize), // line defined, last line
    pub function_name: usize,
    pub line_gap_log2: Option<u8>,
    pub line_info_delta: Option<Vec<u8>>,
    pub abs_line_info_delta: Option<Vec<u32>>,
    pub local_debug_info: Vec<LocalDebugInfo>,
    pub upvalue_debug_names: Vec<usize>, // name indices for upvalues (1-based, 0 = no name)
}

impl Function {
    fn parse_instructions(vec: &Vec<u32>, encode_key: u8) -> Vec<Instruction> {
        let mut v: Vec<Instruction> = Vec::new();
        let mut pc = 0;

        loop {
            let ins = Instruction::parse(vec[pc], encode_key).unwrap();
            let op = match ins {
                Instruction::BC { op_code, .. } => op_code,
                Instruction::AD { op_code, .. } => op_code,
                Instruction::E { op_code, .. } => op_code,
            };

            // handle ops with aux values
            match op {
                OpCode::LOP_GETGLOBAL
                | OpCode::LOP_SETGLOBAL
                | OpCode::LOP_GETIMPORT
                | OpCode::LOP_GETTABLEKS
                | OpCode::LOP_SETTABLEKS
                | OpCode::LOP_NAMECALL
                | OpCode::LOP_JUMPIFEQ
                | OpCode::LOP_JUMPIFLE
                | OpCode::LOP_JUMPIFLT
                | OpCode::LOP_JUMPIFNOTEQ
                | OpCode::LOP_JUMPIFNOTLE
                | OpCode::LOP_JUMPIFNOTLT
                | OpCode::LOP_NEWTABLE
                | OpCode::LOP_SETLIST
                | OpCode::LOP_FORGLOOP
                | OpCode::LOP_LOADKX
                | OpCode::LOP_FASTCALL2
                | OpCode::LOP_FASTCALL2K
                | OpCode::LOP_FASTCALL3
                | OpCode::LOP_JUMPXEQKNIL
                | OpCode::LOP_JUMPXEQKB
                | OpCode::LOP_JUMPXEQKN
                | OpCode::LOP_JUMPXEQKS => {
                    let aux = vec[pc + 1];
                    pc += 2;
                    match ins {
                        Instruction::BC {
                            op_code, a, b, c, ..
                        } => {
                            v.push(Instruction::BC {
                                op_code,
                                a,
                                b,
                                c,
                                aux,
                            });
                        }
                        Instruction::AD { op_code, a, d, .. } => {
                            v.push(Instruction::AD { op_code, a, d, aux });
                        }
                        _ => unreachable!(),
                    }
                    v.push(Instruction::BC {
                        op_code: OpCode::LOP_NOP,
                        a: 0,
                        b: 0,
                        c: 0,
                        aux: 0,
                    });
                }
                _ => {
                    v.push(ins);
                    pc += 1;
                }
            }

            if pc == vec.len() {
                break;
            }
        }

        v
    }

    pub(crate) fn parse(input: &[u8], encode_key: u8, version: u8) -> IResult<&[u8], Self> {
        let (input, max_stack_size) = le_u8(input)?;
        let (input, num_parameters) = le_u8(input)?;
        let (input, num_upvalues) = le_u8(input)?;
        let (input, is_vararg) = le_u8(input)?;

        let input = if version >= 4 {
            let (t_input, _flags) = le_u8(input)?;
            let (t_input, _) = parse_list(t_input, le_u8)?;
            t_input
        } else {
            input
        };

        let (input, u32_instructions) = parse_list(input, le_u32)?;
        //let (input, instructions) = parse_list(input, Function::parse_instrution)?;
        let instructions = Self::parse_instructions(&u32_instructions, encode_key);
        let (input, constants) = parse_list(input, Constant::parse)?;
        let (input, functions) = parse_list(input, leb128_usize)?;
        let (input, line_defined) = leb128_usize(input)?;
        let (input, function_name) = leb128_usize(input)?;
        let (input, has_line_info) = le_u8(input)?;
        let (input, line_gap_log2) = match has_line_info {
            0 => (input, None),
            _ => {
                let (input, line_gap_log2) = le_u8(input)?;
                (input, Some(line_gap_log2))
            }
        };
        let (input, line_info_delta) = match has_line_info {
            0 => (input, None),
            _ => {
                let (input, line_info_delta) =
                    parse_list_len(input, le_u8, u32_instructions.len())?;
                (input, Some(line_info_delta))
            }
        };

        let (input, abs_line_info_delta, last_line) = match &line_info_delta {
            None => (input, None, 0),
            Some(line_info_delta) => {
                let (input, abs_line_info_delta) = parse_list_len(
                    input,
                    le_u32,
                    ((u32_instructions.len() - 1) >> line_gap_log2.unwrap()) + 1,
                )?;

                let last_line = {
                    let line_delta: u8 =
                        line_info_delta.iter().copied().fold(0u8, u8::wrapping_add);

                    let abs_line_delta: usize = abs_line_info_delta
                        .iter()
                        .copied()
                        .map(|v| v as usize)
                        .sum();

                    line_delta as usize + abs_line_delta
                };

                (input, Some(abs_line_info_delta), last_line)
            }
        };
        let (input, (local_debug_info, upvalue_debug_names)) = match le_u8(input)? {
            (input, 0) => (input, (Vec::new(), Vec::new())),
            (input, _) => {
                let (mut input, num_locvars) = leb128_usize(input)?;
                let mut debug_info = Vec::with_capacity(num_locvars);
                for _ in 0..num_locvars {
                    let (new_input, name_index) = leb128_usize(input)?;
                    let (new_input, scope_start) = leb128_usize(new_input)?;
                    let (new_input, scope_end) = leb128_usize(new_input)?;
                    let (new_input, register) = le_u8(new_input)?;
                    input = new_input;
                    debug_info.push(LocalDebugInfo {
                        name_index,
                        scope_start,
                        scope_end,
                        register,
                    });
                }
                let (mut input, num_upvalue_names) = leb128_usize(input)?;
                let mut upvalue_names = Vec::with_capacity(num_upvalue_names);
                for _ in 0..num_upvalue_names {
                    let name_index;
                    (input, name_index) = leb128_usize(input)?;
                    upvalue_names.push(name_index);
                }
                (input, (debug_info, upvalue_names))
            }
        };
        Ok((
            input,
            Self {
                max_stack_size,
                num_parameters,
                num_upvalues,
                is_vararg: is_vararg != 0u8,
                instructions,
                constants,
                functions,
                line_info: (line_defined, last_line),
                function_name,
                line_gap_log2,
                line_info_delta,
                abs_line_info_delta,
                local_debug_info,
                upvalue_debug_names,
            },
        ))
    }

    /// Get the debug name index for a register at a specific PC, if available.
    /// Returns the name_index from debug info (1-based into string table).
    /// The PC is used to find the correct scope when a register is reused
    /// for different variables in different scopes.
    pub fn get_debug_name_for_register(&self, register: u8, pc: usize) -> Option<usize> {
        // Find the debug info entry for this register that contains the current PC in its scope.
        // If multiple entries match (nested scopes), prefer the one with the narrowest scope
        // (largest scope_start that still contains pc).
        //
        // Note: Luau debug info has scope_start pointing to the instruction AFTER the assignment.
        // We extend the range by a few instructions to catch the assignment itself.
        // This handles cases like: reg 3 written at PC 11, scope_start = 14 (for xmlFileContent)
        const SCOPE_EXTENSION: usize = 5;
        self.local_debug_info
            .iter()
            .filter(|info| {
                info.register == register
                    && info.scope_start <= pc + SCOPE_EXTENSION
                    && pc < info.scope_end
            })
            .max_by_key(|info| info.scope_start)
            .map(|info| info.name_index)
    }
}
