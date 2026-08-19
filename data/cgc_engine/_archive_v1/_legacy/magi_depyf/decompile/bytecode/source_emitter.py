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

"""SourceEmitter: manages the evaluation stack and source-code emission.

This replaces the bare ``DecompilerState`` (just ``source_code: str``
and ``stack: list``) with a proper class that owns *all* mutable state
touched during decompilation, including the temp-variable counter
(instance-level, not class-level — thread-safe by design).
"""

from __future__ import annotations

import contextlib
import dataclasses
from typing import Any, Iterator, List, Optional


@dataclasses.dataclass
class LoopContext:
    """Current loop boundaries, used for break/continue determination.

    Range semantics similar to range(start, end):
      start_index: index of FOR_ITER itself (inclusive — part of the loop)
      end_index:   index of the first instruction outside the loop (exclusive — not part of the loop)

    break/continue determination (used in _abs_jump):
      jump target >= end_index   → break  (jump out of the loop)
      jump target == start_index → continue (jump back to loop head)
    """

    start_index: int  # index of FOR_ITER, inclusive (part of the loop)
    end_index: int  # first instruction outside the loop, exclusive (not part of the loop)


class SourceEmitter:
    """Stateful accumulator for the decompiler's output.

    Improvements over depyf's ``DecompilerState``:
    * ``_temp_counter`` is **instance-level** (no thread-safety issues).
    * Stack operations (``push / pop / peek``) are proper methods.
    * ``emit()`` appends with a trailing newline automatically.
    * ``fork()`` context-manager creates a child emitter for sub-blocks
      (if-else branches, loop bodies, etc.) and returns it so the caller
      can inspect the generated source and final stack.
    """

    def __init__(self, indent_size: int = 4, temp_prefix: str = "__temp_", *, _parent_counter: Optional[list] = None) -> None:
        self._lines: List[str] = []
        self._stack: List[Any] = []
        self._indent_size = indent_size
        self._temp_prefix = temp_prefix
        # Share counter across forks so names are globally unique within
        # one Decompiler invocation, but still instance-scoped.
        self._counter: list = _parent_counter if _parent_counter is not None else [0]
        self.loop: Optional[LoopContext] = None

    # -- source emission ----------------------------------------------------

    def emit(self, line: str) -> None:
        """Append *line* (with auto newline) to accumulated source."""
        self._lines.append(line + "\n")

    def emit_raw(self, text: str) -> None:
        """Append pre-formatted *text* verbatim (e.g. nested function defs)."""
        self._lines.append(text)

    def get_source(self) -> str:
        return "".join(self._lines)

    # -- stack operations ---------------------------------------------------

    def push(self, value: Any) -> None:
        self._stack.append(value)

    def pop(self) -> Any:
        return self._stack.pop()

    def peek(self, depth: int = 0) -> Any:
        """Return item at ``stack[-(depth+1)]`` without popping."""
        return self._stack[-(depth + 1)]

    def set_at(self, depth: int, value: Any) -> None:
        """Set ``stack[-(depth+1)]`` to *value*."""
        self._stack[-(depth + 1)] = value

    @property
    def stack(self) -> List[Any]:
        """Direct access (for complex multi-item operations)."""
        return self._stack

    @property
    def stack_size(self) -> int:
        return len(self._stack)

    # -- temp variables (instance-scoped counter) ---------------------------

    def make_temp(self) -> str:
        """Return a unique temporary variable name."""
        self._counter[0] += 1
        return f"{self._temp_prefix}{self._counter[0]}"

    def replace_tos_with_temp(self, depth: int = 1) -> str:
        """Replace ``stack[-depth]`` with a fresh temp, emitting the
        assignment ``__temp_N = <old_value>``.  Returns the temp name."""
        old = self._stack[-depth]
        name = self.make_temp()
        self.emit(f"{name} = {old}")
        self._stack[-depth] = name
        return name

    # -- sub-block forking --------------------------------------------------

    @contextlib.contextmanager
    def fork(self, stack: Optional[List[Any]] = None, loop: Optional[LoopContext] = None) -> Iterator["SourceEmitter"]:
        """Create a child emitter for a sub-block (if-branch, loop body …).

        The child shares the temp counter but has its own ``_lines`` and
        ``_stack``.  If *loop* is ``None`` the parent's loop context is
        inherited (matching depyf's ``new_state`` semantics).

        Usage::

            with emitter.fork(stack=my_stack) as child:
                decompile_range(start, end, child)
            child_source = child.get_source()
            child_final_stack = child.stack
        """
        child = SourceEmitter(indent_size=self._indent_size, temp_prefix=self._temp_prefix, _parent_counter=self._counter)
        child._stack = list(stack) if stack is not None else list(self._stack)
        if loop is not None:
            child.loop = loop
        elif self.loop is not None:
            child.loop = self.loop
        yield child

    # -- indentation helpers ------------------------------------------------

    def indent(self, text: str) -> str:
        """Add one level of indentation to every line in *text*."""
        prefix = " " * self._indent_size
        return "".join(prefix + line + "\n" for line in text.splitlines())
