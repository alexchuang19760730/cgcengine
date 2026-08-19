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

"""Handlers for stack-manipulation opcodes: ROT, SWAP, COPY, POP, DUP.

See bytecode_explained.py §16 for details.
"""

from __future__ import annotations

from ..decompile_context import DecompileContext
from ..handler_registry import registry
from ..instruction import Instruction
from ..source_emitter import SourceEmitter

_reg = registry.register


# ── ROT_N family (Python ≤3.10, replaced by SWAP/COPY in 3.11+) ───────────


@_reg("ROT_N")
@_reg("ROT_TWO")
@_reg("ROT_THREE")
@_reg("ROT_FOUR")
def _rot_n(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    """Top-n stack rotation: [a, b, c] → [c, a, b] (n=3).

    ROT_TWO:   a, b → b, a             (swap, used for a, b = b, a)
    ROT_THREE: a, b, c → c, a, b       (3-element rotation)
    ROT_FOUR:  a, b, c, d → d, a, b, c (4-element rotation)
    ROT_N:     generic n-element rotation (argval = n)
    """
    n = inst.argval if inst.opname == "ROT_N" else {"ROT_TWO": 2, "ROT_THREE": 3, "ROT_FOUR": 4}[inst.opname]
    vals = em.stack[-n:]
    em.stack[-n:] = [vals[-1]] + vals[:-1]


@_reg("SWAP")
def _swap(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    """Python 3.11+: swap stack[-1] and stack[-n]."""
    n = inst.argval
    em.stack[-1], em.stack[-n] = em.stack[-n], em.stack[-1]


@_reg("COPY")
def _copy(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    """Python 3.11+: copy stack[-n] to top of stack (COPY 1 = DUP_TOP)."""
    n = inst.argval
    if n == 0:
        return
    em.push(em.stack[-1 - (n - 1)])


@_reg("POP_TOP")
def _pop_top(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    if em.stack_size > 0:
        em.pop()


@_reg("DUP_TOP")
def _dup_top(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    """Python ≤3.10: duplicate top of stack. Replaced by COPY 1 in 3.11+."""
    em.push(em.peek())


@_reg("DUP_TOP_TWO")
def _dup_top_two(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    """Python ≤3.10: duplicate top two stack items. Replaced by two COPYs in 3.11+."""
    tos = em.peek(0)
    tos1 = em.peek(1)
    em.push(tos1)
    em.push(tos)
