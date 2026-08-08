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

"""Handlers for BUILD_*, UNPACK_*, LIST_EXTEND/APPEND, SET_ADD, MAP_ADD,
FORMAT_VALUE, and BUILD_SLICE / BUILD_STRING."""

from __future__ import annotations

import sys

from ..decompile_context import DecompileContext
from ..handler_registry import registry
from ..instruction import Instruction
from ..source_emitter import SourceEmitter

_reg = registry.register


# ── BUILD tuple / list / set ──────────────────────────────────────────────


def _safe_str(val) -> str:
    """Convert a stack value to string, handling None sentinels from PUSH_NULL."""
    return "None" if val is None else str(val)


@_reg("BUILD_TUPLE", "BUILD_TUPLE_UNPACK", "BUILD_TUPLE_UNPACK_WITH_CALL")
def _build_tuple(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    args = [_safe_str(em.pop()) for _ in range(inst.argval)][::-1]
    if "UNPACK" in inst.opname:
        args = [f"*{a}" for a in args]
    em.push(f"({args[0]},)" if inst.argval == 1 else f"({', '.join(args)})")


@_reg("BUILD_LIST", "BUILD_LIST_UNPACK")
def _build_list(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    args = [_safe_str(em.pop()) for _ in range(inst.argval)][::-1]
    if "UNPACK" in inst.opname:
        args = [f"*{a}" for a in args]
    em.push(f"[{', '.join(args)}]")
    em.replace_tos_with_temp()


@_reg("BUILD_SET", "BUILD_SET_UNPACK")
def _build_set(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    if inst.argval == 0:
        em.push("set()")
    else:
        args = [em.pop() for _ in range(inst.argval)][::-1]
        if "UNPACK" in inst.opname:
            args = [f"*{a}" for a in args]
        em.push(f"{{{', '.join(args)}}}")
    em.replace_tos_with_temp()


# ── BUILD map ─────────────────────────────────────────────────────────────


@_reg("BUILD_MAP")
def _build_map(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    items = [em.pop() for _ in range(inst.argval * 2)][::-1]
    keys, vals = items[::2], items[1::2]
    em.push(f"{{{', '.join(f'{k}: {v}' for k, v in zip(keys, vals))}}}")
    em.replace_tos_with_temp()


@_reg("BUILD_MAP_UNPACK", "BUILD_MAP_UNPACK_WITH_CALL")
def _build_map_unpack(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    if inst.argval == 0:
        em.push("dict()")
    else:
        args = [em.pop() for _ in range(inst.argval)][::-1]
        em.push(f"{{{', '.join(f'**{a}' for a in args)}}}")
    em.replace_tos_with_temp()


@_reg("BUILD_CONST_KEY_MAP")
def _const_key_map(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    keys = eval(em.pop())
    vals = [em.pop() for _ in range(inst.argval)][::-1]
    em.push(f"{{{', '.join(f'{k!r}: {v}' for k, v in zip(keys, vals))}}}")
    em.replace_tos_with_temp()


@_reg("BUILD_STRING")
def _build_string(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    args = [em.pop() for _ in range(inst.argval)][::-1]
    em.push(" + ".join(args))


@_reg("BUILD_SLICE")
def _build_slice(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    tos = em.pop()
    tos1 = em.pop()
    if inst.argval == 2:
        em.push(f"slice({tos1}, {tos})")
    elif inst.argval == 3:
        tos2 = em.pop()
        em.push(f"slice({tos2}, {tos1}, {tos})")


@_reg("LIST_TO_TUPLE")
def _list_to_tuple(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    em.push(f"tuple({em.pop()})")


# ── Mutating container ops ────────────────────────────────────────────────


@_reg("LIST_EXTEND")
def _list_extend(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    values = em.pop()
    temp = em.replace_tos_with_temp(depth=inst.argval)
    em.emit(f"{temp}.extend({values})")


@_reg("LIST_APPEND")
def _list_append(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    argval = inst.argval if inst.argval != 1 else 2
    container = em.stack[-argval]
    value = em.pop()
    em.emit(f"{container}.append({value})")


@_reg("SET_UPDATE", "DICT_UPDATE", "DICT_MERGE")
def _generic_update(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    assert inst.argval == 1, "Only tested for argval==1"
    values = em.pop()
    temp = em.replace_tos_with_temp()
    em.emit(f"{temp}.update({values})")


@_reg("SET_ADD")
def _set_add(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    argval = inst.argval if inst.argval != 1 else 2
    container = em.stack[-argval]
    value = em.pop()
    em.emit(f"{container}.add({value})")


@_reg("MAP_ADD")
def _map_add(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    container = em.stack[-inst.argval - 1]
    if sys.version_info >= (3, 8):
        value = em.pop()
        key = em.pop()
    else:
        key = em.pop()
        value = em.pop()
    em.emit(f"{container}.__setitem__({key}, {value})")


# ── Unpack ────────────────────────────────────────────────────────────────


@_reg("UNPACK_SEQUENCE")
def _unpack_seq(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    varname = em.pop()
    tmps = [em.make_temp() for _ in range(inst.argval)]
    em.emit("".join(f"{t}, " for t in tmps) + f"= {varname}")
    for t in reversed(tmps):
        em.push(t)


@_reg("UNPACK_EX")
def _unpack_ex(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    varname = em.pop()
    tmps = [em.make_temp() for _ in range(inst.argval)]
    star = em.make_temp()
    em.emit(f"{', '.join(tmps)}, *{star} = {varname}")
    em.push(star)
    for t in reversed(tmps):
        em.push(t)


# ── Format ────────────────────────────────────────────────────────────────


@_reg("FORMAT_VALUE")
def _format_value(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    func, spec = inst.argval
    if spec:
        form_spec = em.pop()
        value = em.pop()
        em.push(f"format({value}, {form_spec})")
    else:
        value = em.pop()
        fn = str if func is None else func
        em.push(f"{fn.__name__}({value})")
