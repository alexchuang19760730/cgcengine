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

"""HandlerRegistry — opcode-to-handler dispatch.

A *handler* is a plain function with signature::

    (emitter: SourceEmitter, inst: Instruction, ctx: DecompileContext) -> Optional[int]

Returning ``None`` advances to the next instruction.
Returning an ``int`` jumps to that instruction index.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Optional

if TYPE_CHECKING:
    pass

HandlerFn = Callable[..., Optional[int]]


class HandlerRegistry:
    """Maps opcode names -> handler functions."""

    def __init__(self) -> None:
        self._handlers: dict[str, HandlerFn] = {}

    def register(self, *opnames: str) -> Callable[[HandlerFn], HandlerFn]:
        """Decorator that registers *fn* for one or more opcode names."""

        def decorator(fn: HandlerFn) -> HandlerFn:
            for name in opnames:
                self._handlers[name] = fn
            return fn

        return decorator

    def get(self, opname: str) -> Optional[HandlerFn]:
        return self._handlers.get(opname)

    def __contains__(self, opname: str) -> bool:
        return opname in self._handlers

    def supported_opnames(self) -> List[str]:
        return sorted(self._handlers.keys())


# Singleton registry — handlers register against this at import time.
registry = HandlerRegistry()
