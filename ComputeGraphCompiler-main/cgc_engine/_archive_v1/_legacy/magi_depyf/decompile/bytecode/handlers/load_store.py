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

"""Handlers for LOAD_*, STORE_*, DELETE_*, IMPORT_*, PUSH_NULL, GET_ITER."""

from __future__ import annotations

from types import CodeType

from ..decompile_context import DecompileContext
from ..handler_registry import registry
from ..instruction import Instruction
from ..source_emitter import SourceEmitter

_reg = registry.register


# ── NOP / unsupported sentinels ──────────────────────────────────────────


@_reg("NOP", "RESUME", "EXTENDED_ARG", "SETUP_LOOP", "POP_BLOCK")
@_reg("PRECALL", "BEGIN_FINALLY", "END_FINALLY", "MAKE_CELL")
@_reg("RERAISE", "END_FOR", "COPY_FREE_VARS")
def _nop(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    pass


@_reg("GET_YIELD_FROM_ITER")
@_reg("POP_EXCEPT", "WITH_EXCEPT_START", "JUMP_IF_NOT_EXC_MATCH")
@_reg("CHECK_EG_MATCH", "PUSH_EXC_INFO", "PREP_RERAISE_STAR")
@_reg("WITH_CLEANUP_FINISH", "CALL_FINALLY", "POP_FINALLY")
@_reg("WITH_CLEANUP_START", "SETUP_EXCEPT", "CHECK_EXC_MATCH")
@_reg("CLEANUP_THROW")
@_reg("GET_AWAITABLE", "GET_AITER", "GET_ANEXT", "END_ASYNC_FOR")
@_reg("BEFORE_ASYNC_WITH", "SETUP_ASYNC_WITH", "SEND", "ASYNC_GEN_WRAP")
@_reg("CACHE")
@_reg("PRINT_EXPR", "COPY_DICT_WITHOUT_KEYS")
@_reg("IMPORT_STAR")
@_reg("YIELD_FROM", "SETUP_ANNOTATIONS", "LOAD_BUILD_CLASS")
@_reg("MATCH_MAPPING", "MATCH_SEQUENCE", "MATCH_KEYS", "MATCH_CLASS")
@_reg("CALL_INTRINSIC_2")
@_reg("SETUP_FINALLY", "SETUP_WITH", "BEFORE_WITH")
def _unsupported(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    from ...decompiler import DecompilationError

    raise DecompilationError(f"Unsupported opcode: {inst.opname}", instruction=inst)


# ── LOAD instructions ────────────────────────────────────────────────────


@_reg("LOAD_CONST")
def _load_const(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    """Load a constant. Branches: can_repr → direct repr / type → importlib /
    torch prefix → import torch / CodeType → push as-is for MAKE_FUNCTION."""
    can_repr = False
    try:
        can_repr = eval(repr(inst.argval)) == inst.argval
    except BaseException:
        pass
    if can_repr:
        em.push(repr(inst.argval))
    elif isinstance(inst.argval, type):
        module = inst.argval.__module__
        name = inst.argval.__name__
        em.emit("import importlib")
        tmp = em.make_temp()
        em.emit(f'{tmp} = importlib.import_module("{module}").{name}')
        em.push(tmp)
    elif inst.argrepr.startswith("torch."):
        em.emit("import torch")
        tmp = em.make_temp()
        em.emit(f"{tmp} = {inst.argval}")
        em.push(tmp)
    elif isinstance(inst.argval, CodeType):
        em.push(inst.argval)
    else:
        from ...decompiler import DecompilationError

        raise DecompilationError(
            f"LOAD_CONST: cannot represent co_consts[{inst.arg}] = {repr(inst.argval)!r} "
            f"(type {type(inst.argval).__name__}) as source code",
            instruction=inst,
        )


@_reg("LOAD_FAST", "LOAD_FAST_CHECK")
@_reg("LOAD_GLOBAL", "LOAD_DEREF", "LOAD_NAME")
@_reg("LOAD_CLASSDEREF", "LOAD_CLOSURE")
def _generic_load(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    """Generic load. 3.11+ LOAD_GLOBAL argrepr "NULL + name" pushes a NULL sentinel first.
    Python <3.12 comprehension parameter name ".0" is replaced with "comp_arg_0"."""
    if "NULL + " in inst.argrepr:
        em.push(None)
    if inst.argrepr.startswith("."):
        em.push(inst.argval.replace(".", "comp_arg_"))
    else:
        em.push(inst.argval)


# Python 3.12 comprehension variable protection: LOAD_FAST_AND_CLEAR saves old value + STORE_FAST restores.
# During decompilation, temp variables used for loops don't need save/restore; push a sentinel so STORE_FAST skips.
_CLEAR_SENTINEL = object()


@_reg("LOAD_FAST_AND_CLEAR")
def _load_fast_and_clear(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    em.push(_CLEAR_SENTINEL)


@_reg("LOAD_LOCALS")
def _load_locals(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    """3.12 class body: locals() returns a new dict snapshot, cached in a temp to avoid repeated calls."""
    em.push("locals()")
    em.replace_tos_with_temp()


@_reg("LOAD_FROM_DICT_OR_GLOBALS", "LOAD_FROM_DICT_OR_DEREF")
def _load_from_dict(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    """3.12 class body: look up in locals dict first, fall back to globals if not found."""
    tos = em.pop()
    em.push(f"{tos}[{inst.argval}] if '{inst.argval}' in {tos} else {inst.argval}")
    em.replace_tos_with_temp()


@_reg("LOAD_ATTR")
def _load_attr(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    """Attribute access. isidentifier() checks if the attr name is valid; if not, use getattr()."""
    lhs = str(em.pop())
    rhs = inst.argval
    em.push(f"{lhs}.{rhs}" if rhs.isidentifier() else f"getattr({lhs}, {rhs!r})")


@_reg("LOAD_SUPER_ATTR")
def _load_super_attr(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    self_obj = em.pop()
    cls_obj = em.pop()
    super_obj = em.pop()
    em.push(f"{super_obj}({cls_obj}, {self_obj}).{inst.argval}")
    em.replace_tos_with_temp()


@_reg("LOAD_METHOD")
def _load_method(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    em.push(f"{em.pop()}.{inst.argval}")


@_reg("LOAD_ASSERTION_ERROR")
def _load_assertion_error(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    em.push("AssertionError")


@_reg("PUSH_NULL")
def _push_null(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    """3.11+ pushes a NULL sentinel before function calls; the CALL handler will clear it."""
    em.push(None)


@_reg("GET_ITER")
def _get_iter(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    em.push(f"iter({em.pop()})")


# ── STORE instructions ───────────────────────────────────────────────────


@_reg("STORE_FAST", "STORE_GLOBAL", "STORE_DEREF", "STORE_NAME")
def _generic_store(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    """Generic store. Skips _CLEAR_SENTINEL and self-assignment, protects variable names on the stack that are about to be overwritten."""
    left = inst.argval
    right = em.pop()
    if right is _CLEAR_SENTINEL:
        return
    if left != right:
        if isinstance(left, str) and left in em.stack:
            tmp = em.make_temp()
            em.emit(f"{tmp} = {left}")
            em.stack[:] = [tmp if x == left else x for x in em.stack]
        em.emit(f"{left} = {right}")


@_reg("STORE_SUBSCR")
def _store_subscr(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    index = em.pop()
    obj = em.pop()
    value = em.pop()
    em.emit(f"{obj}[{index}] = {value}")


@_reg("STORE_SLICE")
def _store_slice(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    end = em.pop()
    start = em.pop()
    container = em.pop()
    value = em.pop()
    em.emit(f"{container}[{start}:{end}] = {value}")


@_reg("STORE_ATTR")
def _store_attr(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    obj = em.pop()
    value = em.pop()
    em.emit(f"{obj}.{inst.argval} = {value}")


# ── DELETE instructions ──────────────────────────────────────────────────


@_reg("DELETE_SUBSCR")
def _delete_subscr(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    index = em.pop()
    obj = em.pop()
    if f"{obj}[{index}]" not in em.stack:
        em.emit(f"del {obj}[{index}]")


@_reg("DELETE_NAME", "DELETE_GLOBAL", "DELETE_DEREF")
def _generic_delete(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    em.emit(f"del {inst.argval}")


@_reg("DELETE_FAST")
def _delete_fast(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    """Dynamo cleans up temp variables; no explicit del needed after decompilation."""
    pass


@_reg("DELETE_ATTR")
def _delete_attr(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    em.emit(f"del {em.pop()}.{inst.argval}")


# ── IMPORT instructions ──────────────────────────────────────────────────


@_reg("IMPORT_NAME")
def _import_name(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    """import os.path → binds 'os' (top-level module), accesses submodules via os.path.sep."""
    name = inst.argval.split(".")[0]
    fromlist = em.pop()
    level = em.pop()
    em.emit(f"{name} = __import__({inst.argval!r}, fromlist={fromlist}, level={level})")
    em.push(name)


@_reg("IMPORT_FROM")
def _import_from(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    name = inst.argval
    module = em.peek()
    em.emit(f"{name} = {module}.{name}")
    em.push(name)
