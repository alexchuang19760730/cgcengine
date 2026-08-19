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

"""Handlers for unary, binary, inplace, and comparison operations."""

from __future__ import annotations

from ..decompile_context import DecompileContext
from ..handler_registry import registry
from ..instruction import Instruction
from ..source_emitter import SourceEmitter

_reg = registry.register

# ── Unary ─────────────────────────────────────────────────────────────────

_UNARY_SYMBOLS = {"UNARY_NEGATIVE": "-", "UNARY_POSITIVE": "+", "UNARY_INVERT": "~", "UNARY_NOT": "not"}


@_reg(*_UNARY_SYMBOLS)
def _unary(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    em.push(f"({_UNARY_SYMBOLS[inst.opname]} {em.pop()})")


@_reg("GET_LEN")
def _get_len(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    em.push(f"len({em.peek()})")


# ── Binary ────────────────────────────────────────────────────────────────

_BINARY_SYMBOLS = {
    "BINARY_MULTIPLY": "*",
    "BINARY_ADD": "+",
    "BINARY_SUBTRACT": "-",
    "BINARY_TRUE_DIVIDE": "/",
    "BINARY_FLOOR_DIVIDE": "//",
    "BINARY_MODULO": "%",
    "BINARY_POWER": "**",
    "BINARY_AND": "&",
    "BINARY_OR": "|",
    "BINARY_XOR": "^",
    "BINARY_LSHIFT": "<<",
    "BINARY_RSHIFT": ">>",
    "BINARY_MATRIX_MULTIPLY": "@",
}


@_reg(*_BINARY_SYMBOLS)
def _binary(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    rhs = em.pop()
    lhs = em.pop()
    em.push(f"({lhs} {_BINARY_SYMBOLS[inst.opname]} {rhs})")


@_reg("BINARY_SUBSCR")
def _subscr(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    rhs = em.pop()
    lhs = em.pop()
    em.push(f"{lhs}[{rhs}]")


@_reg("BINARY_SLICE")
def _slice(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    end = em.pop()
    start = em.pop()
    container = em.pop()
    em.push(f"{container}[{start}:{end}]")


# ── Inplace ───────────────────────────────────────────────────────────────

_INPLACE_SYMBOLS = {
    "INPLACE_MULTIPLY": "*",
    "INPLACE_ADD": "+",
    "INPLACE_SUBTRACT": "-",
    "INPLACE_TRUE_DIVIDE": "/",
    "INPLACE_FLOOR_DIVIDE": "//",
    "INPLACE_MODULO": "%",
    "INPLACE_POWER": "**",
    "INPLACE_AND": "&",
    "INPLACE_OR": "|",
    "INPLACE_XOR": "^",
    "INPLACE_LSHIFT": "<<",
    "INPLACE_RSHIFT": ">>",
    "INPLACE_MATRIX_MULTIPLY": "@",
}


@_reg(*_INPLACE_SYMBOLS)
def _inplace(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    rhs = em.pop()
    lhs = em.pop()
    em.emit(f"{lhs} {_INPLACE_SYMBOLS[inst.opname]}= {rhs}")
    em.push(lhs)


@_reg("BINARY_OP")
def _binary_op(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    """Python 3.12+ unified BINARY_OP."""
    rhs = em.pop()
    lhs = em.pop()
    if "=" in inst.argrepr:
        em.emit(f"{lhs} {inst.argrepr} {rhs}")
        em.push(lhs)
    else:
        em.push(f"({lhs} {inst.argrepr} {rhs})")


# ── Comparison ────────────────────────────────────────────────────────────


@_reg("COMPARE_OP")
def _compare(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    rhs = em.pop()
    lhs = em.pop()
    em.push(f"({lhs} {inst.argval} {rhs})")


@_reg("IS_OP")
def _is_op(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    rhs = em.pop()
    lhs = em.pop()
    op = "is" if inst.argval == 0 else "is not"
    em.push(f"({lhs} {op} {rhs})")


@_reg("CONTAINS_OP")
def _contains(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    rhs = em.pop()
    lhs = em.pop()
    op = "in" if inst.argval == 0 else "not in"
    em.push(f"({lhs} {op} {rhs})")
