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

"""Decompiler — the orchestrator that ties everything together.

This module is the only place that coordinates ``SourceEmitter``,
``HandlerRegistry``, and ``DecompileContext``.
Individual handler functions never import from here (except for
``DecompilationError`` and recursive ``Decompiler`` usage in
``MAKE_FUNCTION``).
"""

from __future__ import annotations

import dis
import inspect
import os
from types import CodeType
from typing import Callable, List, Optional, Union

# Force handler registration by importing the package.
import cgc_engine.magi_depyf.decompile.bytecode.handlers  # noqa: F401

from .bytecode.decompile_context import DecompileContext
from .bytecode.handler_registry import registry
from .bytecode.instruction import Instruction
from .bytecode.source_emitter import SourceEmitter

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DecompilationError(Exception):
    """Raised when decompilation fails.

    Carries optional ``instruction`` context so callers can produce
    actionable error messages.
    """

    def __init__(self, message: str = "", *, instruction: Optional[Instruction] = None):
        self.message = message
        self.instruction = instruction
        super().__init__(message)

    def __str__(self) -> str:
        loc = ""
        if self.instruction is not None:
            loc = f" at {self.instruction}"
        return f"DecompilationError: {self.message}{loc}"


# ---------------------------------------------------------------------------
# Signature builder (lives here — it's a Decompiler concern, not a util)
# ---------------------------------------------------------------------------


class SignatureBuilder:
    """Build the ``def fn(args):`` header from a ``CodeType``."""

    @staticmethod
    def build(code: CodeType, overwrite_name: Optional[str] = None) -> str:
        n = code.co_argcount + code.co_kwonlyargcount
        names = [x.replace(".", "comp_arg_") if x.startswith(".") else x for x in code.co_varnames[:n]]
        if code.co_flags & inspect.CO_VARARGS:
            names.append("*" + code.co_varnames[n])
            n += 1
        if code.co_flags & inspect.CO_VARKEYWORDS:
            names.append("**" + code.co_varnames[n])
            n += 1
        fn_name = overwrite_name or code.co_name
        return f"def {fn_name}({', '.join(names)}):\n"


# ---------------------------------------------------------------------------
# Decompiler
# ---------------------------------------------------------------------------


class Decompiler:
    """Decompile a ``CodeType`` into Python source code.

    Design differences from depyf's ``Decompiler``:

    * Handlers live in separate modules and receive
      ``(emitter, inst, ctx)`` — they never reference this class.
    * All mutable state is on ``SourceEmitter`` (instance-scoped counter).
    * ``decompile_range`` is delegated *through* ``DecompileContext``
      so handlers can recurse without importing this class (except
      ``MAKE_FUNCTION`` which needs a fresh ``Decompiler`` instance).
    """

    _TERMINATORS = frozenset({"RETURN_VALUE", "RETURN_CONST", "RAISE_VARARGS"})

    def __init__(self, code: Union[CodeType, Callable]) -> None:
        if callable(code) and not isinstance(code, CodeType):
            code = _get_code_owner(code).__code__
        self.code: CodeType = code
        self.instructions = [Instruction.from_dis(i) for i in dis.get_instructions(code)]
        self._cleanup()

    # -- bytecode cleanup ---------------------------------------------------

    def _cleanup(self) -> None:
        """Propagate line numbers and NOP dead code after unconditional exits."""
        cur: Optional[int] = None
        for inst in self.instructions:
            if inst.starts_line is not None:
                cur = inst.starts_line
            inst.starts_line = cur

        in_dead = False
        for inst in self.instructions:
            if in_dead:
                if inst.is_jump_target:
                    in_dead = False
                else:
                    inst.nop_()
            elif inst.opname in self._TERMINATORS:
                in_dead = True

    # -- core loop ----------------------------------------------------------

    def decompile_range(self, start: int, end: int, emitter: SourceEmitter) -> None:
        """Execute instruction handlers from *start* to *end* (exclusive)."""
        idx = start
        try:
            while idx < end:
                inst = self.instructions[idx]
                handler = registry.get(inst.opname)
                if handler is None:
                    raise DecompilationError(f"No handler for opcode {inst.opname}", instruction=inst)
                ctx = self._make_context(emitter)
                result = handler(emitter, inst, ctx)
                idx = result if result is not None else idx + 1
        except DecompilationError:
            raise
        except Exception as e:
            raise DecompilationError(f"Failed at {inst!r} in {self.code.co_name}", instruction=inst) from e

    def _make_context(self, emitter: SourceEmitter) -> DecompileContext:
        return DecompileContext(
            code=self.code,
            instructions=tuple(self.instructions),
            indentation=emitter._indent_size,
            decompile_range=lambda start, end, em: self.decompile_range(start, end, em),
            offset_to_index={inst.offset: idx for idx, inst in enumerate(self.instructions)},
        )

    # -- public API ---------------------------------------------------------

    def decompile(self, indentation: int = 4, temp_prefix: str = "__temp_", overwrite_fn_name: Optional[str] = None) -> str:
        """Return decompiled Python source code."""
        try:
            emitter = SourceEmitter(indent_size=indentation, temp_prefix=temp_prefix)
            self.decompile_range(0, len(self.instructions), emitter)
            body = emitter.get_source()

            if os.environ.get("DEPYF_REMOVE_TEMP", "1") == "1":
                from .postprocess import run_all as _postprocess

                body = _postprocess(body, temp_prefix, indentation)

            header = SignatureBuilder.build(self.code, overwrite_fn_name)

            global_names = {i.argval for i in dis.get_instructions(self.code) if i.opname == "STORE_GLOBAL"}
            preamble = ""
            if global_names:
                preamble += "global " + ", ".join(global_names) + "\n"
            if self.code.co_freevars:
                preamble += "nonlocal " + ", ".join(self.code.co_freevars) + "\n"

            body = preamble + body
            return header + emitter.indent(body)
        except DecompilationError:
            raise
        except Exception as e:
            raise DecompilationError(f"Failed to decompile {self.code.co_name}") from e

    @staticmethod
    def supported_opnames() -> List[str]:
        return registry.supported_opnames()


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def decompile(code: Union[CodeType, Callable]) -> str:
    """One-liner: decompile a code object or callable to source."""
    return Decompiler(code).decompile()


def safe_decompile(code: CodeType) -> str:
    """Decompile *code* without raising; fall back to depyf then placeholder."""
    try:
        return Decompiler(code).decompile()
    except Exception:
        try:
            from depyf import decompile as _depyf_decompile

            return _depyf_decompile(code)
        except Exception:
            return f"# Failed to decompile {code.co_name}\n"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_code_owner(fn):
    """Walk through wrappers to find the object that owns ``__code__``."""
    if hasattr(fn, "__func__"):
        return fn.__func__
    if hasattr(fn, "__wrapped__"):
        return _get_code_owner(fn.__wrapped__)
    return fn
