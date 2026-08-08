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

"""Handlers for function-call and function-creation opcodes."""

from __future__ import annotations

import sys
from typing import Optional

from ..decompile_context import DecompileContext
from ..handler_registry import registry
from ..instruction import Instruction
from ..source_emitter import SourceEmitter

_reg = registry.register


@_reg("KW_NAMES")
def _kw_names(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    # Python 3.11+ instruction that passes keyword argument names to the subsequent CALL.
    # inst.arg indexes into co_consts for the key-name tuple, e.g. ('y', 'z').
    # Push repr so it becomes the string "('y', 'z')"; the CALL handler later eval()s it back to a tuple.
    names = ctx.code.co_consts[inst.arg]
    em.push(repr(names))


@_reg("CALL")
def _call(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    """Python 3.11+ unified CALL.

    3.12 stack layout: [NULL, callable, arg0, ..., argN-1]  (KW_NAMES precedes)
    3.11 stack layout: [NULL, callable, arg0, ..., argN-1]  (KW_NAMES → PRECALL → CALL)
    """
    # Check whether KW_NAMES precedes CALL (indicating keyword arguments exist).
    # 3.12: KW_NAMES → CALL;  3.11: KW_NAMES → PRECALL → CALL
    preceding = [x for x in ctx.instructions if x.offset < inst.offset]
    has_kw = False
    if preceding:
        if preceding[-1].opname == "KW_NAMES" or (
            len(preceding) > 1
            and preceding[-2].opname == "KW_NAMES"
            and preceding[-1].opname == "PRECALL"  # 3.11 transitional opcode, removed in 3.12
        ):
            has_kw = True

    kw_names: tuple = ()
    if has_kw:
        kw_names = eval(em.pop())  # retrieve the tuple stored by KW_NAMES from the stack
    args = [em.pop() for _ in range(inst.argval)][::-1]
    pos_args = args[: len(args) - len(kw_names)]
    kw_args = args[len(args) - len(kw_names) :]
    kwcalls = [f"{n}={v}" for n, v in zip(kw_names, kw_args)]
    func = em.pop()
    # 3.11+ PUSH_NULL / LOAD_GLOBAL(NULL+name) pushes a NULL sentinel before the call.
    # After popping the callable, the top of stack may be NULL (represented as None); clear it.
    if em.stack_size and em.peek() is None:
        em.pop()
    # GET_ITER produces "iter(x)"; if func happens to be "iter(x)" it is actually an argument
    # (e.g. in the next(iter(x)) pattern), and the real callable is further down the stack.
    if "iter(" in str(func):
        pos_args = [func]
        func = em.pop()
    em.push(f"{func}({', '.join(pos_args + kwcalls)})")
    # replace_tos_with_temp: the call result may be referenced multiple times (assignment, passing,
    # method call), so store it in a temp to avoid repeated evaluation and side effects.
    em.replace_tos_with_temp()


@_reg("CALL_FUNCTION", "CALL_METHOD")
def _call_legacy(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    """CALL_FUNCTION / CALL_METHOD (Python ≤3.10)."""
    args = [em.pop() for _ in range(inst.argval)][::-1]
    func = em.pop()
    em.push(f"{func}({', '.join(args)})")
    em.replace_tos_with_temp()


@_reg("CALL_FUNCTION_KW")
def _call_function_kw(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    kw_args = eval(em.pop())
    kw_vals = [em.pop() for _ in range(len(kw_args))]
    kw_vals.reverse()
    kwcalls = [f"{n}={v}" for n, v in zip(kw_args, kw_vals)]
    pos_args = [em.pop() for _ in range(inst.argval - len(kw_args))][::-1]
    func = em.pop()
    em.push(f"{func}({', '.join(pos_args + kwcalls)})")
    em.replace_tos_with_temp()


@_reg("CALL_FUNCTION_EX")
def _call_function_ex(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    # 3.11+ stack: [NULL, func, args (, kwargs)]
    # After popping func, clear the NULL sentinel before pushing the result
    if inst.argval == 0:
        a = em.pop()
        f = em.pop()
        if em.stack_size and em.peek() is None:
            em.pop()
        em.push(f"{f}(*{a})")
    elif inst.argval == 1:
        kw = em.pop()
        a = em.pop()
        f = em.pop()
        if em.stack_size and em.peek() is None:
            em.pop()
        em.push(f"{f}(*{a}, **{kw})")
    em.replace_tos_with_temp()


@_reg("CALL_INTRINSIC_1")
def _intrinsic_1(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> None:
    """Python 3.12 instruction replacing some internal C-level calls.
    argrepr identifies the specific operation, e.g. INTRINSIC_PRINT, INTRINSIC_UNARY_POSITIVE.
    Most are compiler-internal operations (import *, typealias) rarely triggered by user code."""
    _SKIP = {
        "INTRINSIC_1_INVALID",
        "INTRINSIC_IMPORT_STAR",
        "INTRINSIC_STOPITERATION_ERROR",
        "INTRINSIC_ASYNC_GEN_WRAP",
        "INTRINSIC_TYPEVAR",
        "INTRINSIC_PARAMSPEC",
        "INTRINSIC_TYPEVARTUPLE",
        "INTRINSIC_SUBSCRIPT_GENERIC",
        "INTRINSIC_TYPEALIAS",
    }
    if inst.argrepr in _SKIP:
        return
    if inst.argrepr == "INTRINSIC_PRINT":
        em.emit(f"print({em.pop()})")
        em.push("None")
    elif inst.argrepr == "INTRINSIC_UNARY_POSITIVE":
        em.set_at(0, f"+{em.peek()}")
    elif inst.argrepr == "INTRINSIC_LIST_TO_TUPLE":
        em.push(f"tuple({em.pop()})")


# ── MAKE_FUNCTION ─────────────────────────────────────────────────────────


@_reg("MAKE_FUNCTION")
def _make_function(em: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> Optional[int]:
    """Handle bytecode for def inner(...) and lambda.

    Bytecode: LOAD_CONST <code_object> → MAKE_FUNCTION → STORE_FAST name
    The handler recursively decompiles the inner code object and emits the full def statement.
    """
    if sys.version_info < (3, 11):
        # 3.10: qualified_name string is still on the stack
        qual_name = em.pop()
        try:
            qual_name = eval(qual_name)
        except Exception:
            pass
        func_name = qual_name.split(".")[-1]
        if "<" in func_name:  # <lambda>, <listcomp>, etc. — invalid identifiers
            em.emit(f'"original function name {func_name} is illegal, use a temp name."')
            func_name = em.make_temp()
    else:
        func_name = em.make_temp()

    code = em.pop()  # inner CodeType object pushed by LOAD_CONST
    # argval bit flags indicate whether extra function components remain on the stack
    if inst.argval & 0x08:
        em.pop()  # closure tuple (cell references for freevars)
    if inst.argval & 0x04:
        em.pop()  # annotations dict
    if inst.argval & 0x02:
        em.pop()  # keyword-only defaults tuple
    if inst.argval & 0x01:
        em.pop()  # positional defaults tuple

    # If the next instruction is STORE_FAST, use the target variable name as the function name
    this_idx = ctx.index_of(inst.offset)
    immediately_used = False
    if ctx.instructions[this_idx + 1].opname == "STORE_FAST":
        func_name = ctx.instructions[this_idx + 1].argval
        immediately_used = True

    # Recurse: create a new Decompiler instance for the inner code object
    from ...decompiler import Decompiler

    inner = Decompiler(code).decompile(overwrite_fn_name=func_name)
    em.emit_raw(inner)

    if immediately_used:
        return this_idx + 2  # skip the MAKE_FUNCTION + STORE_FAST pair
    em.push(func_name)  # not immediately assigned — push onto stack for later use (e.g. as an argument)
    return None
