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

"""DecompileContext — read-only bag passed to every handler.

Handlers receive ``(emitter, inst, ctx)`` — they mutate *emitter*
and call *ctx* methods but never touch the ``Decompiler`` directly.
"""

from __future__ import annotations

from types import CodeType
from typing import TYPE_CHECKING, Callable, Dict, Tuple

if TYPE_CHECKING:
    from .instruction import Instruction


class DecompileContext:
    """Read-only context providing handlers with instructions, code object,
    and the ``decompile_range`` callback for recursive sub-block decompilation."""

    def __init__(
        self,
        code: CodeType,
        instructions: Tuple["Instruction", ...],
        indentation: int,
        decompile_range: Callable,
        offset_to_index: Dict[int, int],
    ) -> None:
        self.code = code
        self.instructions = instructions
        self.indentation = indentation
        self.decompile_range = decompile_range
        self._offset_to_index = offset_to_index

    def index_of(self, offset: int) -> int:
        """Return the index of the instruction at *offset* (O(1) lookup)."""
        try:
            return self._offset_to_index[offset]
        except KeyError:
            raise ValueError(f"No instruction at offset {offset}") from None
