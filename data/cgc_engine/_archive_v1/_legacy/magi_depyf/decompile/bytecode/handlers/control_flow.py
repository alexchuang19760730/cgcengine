# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Handlers for control-flow opcodes: jumps, if/else, for, return, yield, raise."""

from __future__ import annotations

from typing import Optional

from ..decompile_context import DecompileContext
from ..handler_registry import registry
from ..instruction import Instruction
from ..source_emitter import LoopContext, SourceEmitter

_reg = registry.register


# ── Simple returns / yield / raise ────────────────────────────────────────


@_reg("RETURN_VALUE")
def _return_value(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    em.emit(f"return {em.peek()}")
    em.pop()


@_reg("RETURN_CONST")
def _return_const(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    em.emit(f"return {repr(inst.argval)}")


@_reg("YIELD_VALUE")
def _yield_value(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    import sys

    if sys.version_info >= (3, 12):
        raise NotImplementedError("YIELD_VALUE is not supported in Python 3.12+")
    em.emit(f"yield {em.peek()}")


@_reg("RETURN_GENERATOR")
def _return_generator(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    """Python 3.11+ generator function prologue. Each generator has its own stack frame;
    RETURN_GENERATOR creates the generator object and returns it to the caller,
    subsequent next(gen) resumes from RESUME. Push None as a placeholder during decompilation."""
    em.push(None)


@_reg("GEN_START")
def _gen_start(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    """Python 3.11 marks generator start (replaced by RESUME in 3.12)."""
    assert inst.argval == 0, "Only generator expression is supported"


@_reg("RAISE_VARARGS")
def _raise_varargs(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    if inst.argval == 0:
        em.emit("raise")
    elif inst.argval == 1:
        em.emit(f"raise {em.pop()}")
    elif inst.argval == 2:
        tos = em.pop()
        tos1 = em.pop()
        em.emit(f"raise {tos1} from {tos}")


@_reg("BREAK_LOOP")
def _break_loop(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    em.emit("break")


# ── Unconditional jumps ───────────────────────────────────────────────────


@_reg("JUMP_ABSOLUTE")
@_reg("JUMP_FORWARD")
@_reg("JUMP_BACKWARD")
@_reg("JUMP_BACKWARD_NO_INTERRUPT")
def _abs_jump(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> Optional[int]:
    """Unconditional jump. Returns len(instructions) to make decompile_range stop immediately."""
    target = inst.jump_target_offset()
    idx = ctx.index_of(target)
    loop = em.loop
    if loop is not None:
        if idx >= loop.end_index:
            em.emit("break")
            return len(ctx.instructions)
        if idx == loop.start_index:
            em.emit("continue")
            return len(ctx.instructions)
    return idx


# ── Conditional jumps (if / else) ─────────────────────────────────────────


@_reg("POP_JUMP_IF_TRUE", "POP_JUMP_IF_FALSE")
@_reg("POP_JUMP_FORWARD_IF_TRUE", "POP_JUMP_FORWARD_IF_FALSE")
@_reg("POP_JUMP_BACKWARD_IF_TRUE", "POP_JUMP_BACKWARD_IF_FALSE")
@_reg("POP_JUMP_FORWARD_IF_NONE", "POP_JUMP_FORWARD_IF_NOT_NONE")
@_reg("POP_JUMP_BACKWARD_IF_NONE", "POP_JUMP_BACKWARD_IF_NOT_NONE")
@_reg("JUMP_IF_TRUE_OR_POP", "JUMP_IF_FALSE_OR_POP")
@_reg("POP_JUMP_IF_NOT_NONE", "POP_JUMP_IF_NONE")
def _jump_if(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> Optional[int]:
    """Decompile if/else structure.

    Standard if/else bytecode:
      POP_JUMP_IF_FALSE else_start   ← this_idx
      (if-body)
      JUMP_FORWARD after_else        ← last instruction of if-body
      >> else_start:                 ← jump_idx
      (else-body)
      >> after_else:                 ← merge point (end)
    """

    jump_offset = inst.jump_target_offset()
    jump_idx = ctx.index_of(jump_offset)
    this_idx = ctx.index_of(inst.offset)

    # ── Step 1: condition expression and branch stack state ──
    cond = em.peek()
    fall_stack = list(em.stack)
    jump_stack = list(em.stack)

    if "IF_NOT_NONE" in inst.opname:
        cond = f"({cond} is None)"
    elif "IF_NONE" in inst.opname:
        cond = f"({cond} is not None)"
    elif "IF_TRUE" in inst.opname:
        cond = f"(not {cond})"
    else:
        cond = f"{cond}"

    if "POP_JUMP" in inst.opname:
        jump_stack.pop()
        fall_stack.pop()
    elif "OR_POP" in inst.opname:
        fall_stack.pop()

    # ── Step 2: merge point candidate upper bounds ──
    merge_upper_bounds = [len(ctx.instructions)]
    if em.loop is not None:
        merge_upper_bounds.append(em.loop.end_index)

    # ── Step 3: find "skip else" JUMPs in the if-body ──
    def _is_forward_past_else(i: Instruction) -> bool:
        return i.is_jump and i.jump_target_offset() >= jump_offset

    forward_targets = [i.jump_target_offset() for i in ctx.instructions[this_idx:jump_idx] if _is_forward_past_else(i)]

    # ── Step 4: compute merge point by case ──
    if not forward_targets:
        if jump_idx <= this_idx:
            # Case C: backward jump (inside loop), emit if cond: continue
            rev_cond = em.peek()
            if "IF_NOT_NONE" in inst.opname:
                rev_cond = f"({rev_cond} is not None)"
            elif "IF_NONE" in inst.opname:
                rev_cond = f"({rev_cond} is None)"
            elif "IF_TRUE" in inst.opname:
                rev_cond = f"{rev_cond}"
            elif "IF_FALSE" in inst.opname:
                rev_cond = f"(not {rev_cond})"
            em.emit(f"if {rev_cond}:")
            em.emit(em.indent("continue\n").rstrip("\n"))
            return None
        # Case B: both branches terminate with RETURN/RAISE
        end = jump_idx
    else:
        # Case A: standard if/else, infer merge point from forward_targets
        max_jump = max(forward_targets)
        max_idx = ctx.index_of(max_jump)
        all_targets = [i.jump_target_offset() for i in ctx.instructions[this_idx:max_idx] if _is_forward_past_else(i)]
        max_idx = ctx.index_of(max(all_targets))

        last = ctx.instructions[max_idx - 1]
        if not ("RAISE" in last.opname or "RETURN" in last.opname or "STORE" in last.opname):
            old = max_idx
            while max_idx < len(ctx.instructions):
                op = ctx.instructions[max_idx].opname
                if "STORE" in op or "RETURN" in op:
                    max_idx += 1
                    break
                if ("JUMP" in op and max_idx > old) or "FOR_ITER" in op:
                    break
                max_idx += 1

        merge_upper_bounds.append(max_idx)
        end = min(merge_upper_bounds)

    # ── Step 5: else-body end position (PR#91 fix) ──
    else_end = end
    if end == jump_idx and jump_idx < len(ctx.instructions):
        last_if = ctx.instructions[jump_idx - 1]
        if "RETURN" in last_if.opname or "RAISE" in last_if.opname:
            else_end = len(ctx.instructions)
            if em.loop is not None:
                else_end = min(else_end, em.loop.end_index)

    # ── Step 6: decompile both branches ──
    with em.fork(stack=fall_stack) as if_em:
        ctx.decompile_range(this_idx + 1, jump_idx, if_em)
    if_body = em.indent(if_em.get_source())
    if_end_stack = list(if_em.stack)
    em.emit_raw(f"if {cond}:\n{if_body}")

    with em.fork(stack=jump_stack) as else_em:
        ctx.decompile_range(jump_idx, else_end, else_em)
    else_body = else_em.get_source()
    if else_body:
        em.emit_raw(f"else:\n{em.indent(else_body)}")

    em.stack[:] = if_end_stack
    return else_end


# ── FOR_ITER ──────────────────────────────────────────────────────────────


@_reg("FOR_ITER")
def _for_iter(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> Optional[int]:
    """Decompile for loop.

    Bytecode layout (3.12):
      FOR_ITER target       ← get next value; jump to target when exhausted (END_FOR)
      (loop body)
      JUMP_BACKWARD for_iter ← normal back-jump (not continue)
      >> target: END_FOR

    Loop body range excludes the trailing JUMP_BACKWARD to avoid emitting a spurious continue.
    """
    start_idx = ctx.index_of(inst.offset)
    end_idx = ctx.index_of(inst.jump_target_offset())

    temp = em.make_temp()
    iterator = em.pop()
    em.push(temp)

    # Determine the actual end position of the loop body:
    # if the instruction at end_idx is a back-jump to FOR_ITER, extend end_idx so
    # the LoopContext boundary is correct (break needs to jump past end_idx)
    if end_idx < len(ctx.instructions):
        at_end = ctx.instructions[end_idx]
        if at_end.is_jump and at_end.jump_target_offset() == inst.offset:
            end_idx += 1

    # Exclude the trailing JUMP_BACKWARD: it is the normal loop back-jump mechanism, not continue.
    # Only JUMP_BACKWARDs in the middle of the loop body are continue (handled by _abs_jump).
    body_end = end_idx
    if body_end > start_idx + 1:
        back_jump = ctx.instructions[body_end - 1]
        if back_jump.is_jump and back_jump.jump_target_offset() == inst.offset:
            body_end -= 1

    loop = LoopContext(start_index=start_idx, end_index=end_idx)
    with em.fork(stack=list(em.stack), loop=loop) as body_em:
        ctx.decompile_range(start_idx + 1, body_end, body_em)

    body_src = em.indent(body_em.get_source())
    em.emit_raw(f"for {temp} in {iterator}:\n{body_src}")
    em.stack[:] = body_em.stack
    return end_idx
