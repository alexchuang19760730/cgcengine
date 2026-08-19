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

"""Enhanced Instruction dataclass with rich querying properties."""

from __future__ import annotations

import dataclasses
import dis
import sys
from typing import Any, Optional

_ALL_JUMP_OPCODES = frozenset(dis.hasjabs) | frozenset(dis.hasjrel)
_PY311 = sys.version_info >= (3, 11)

_LOAD_OPCODES = frozenset(n for n in dis.opname if n.startswith("LOAD_") or n in ("PUSH_NULL", "GET_ITER"))
_STORE_OPCODES = frozenset(n for n in dis.opname if n.startswith("STORE_"))
_DELETE_OPCODES = frozenset(n for n in dis.opname if n.startswith("DELETE_"))


@dataclasses.dataclass
class Instruction:
    """Mutable mirror of ``dis.Instruction`` with convenience queries.

    Unlike the stdlib version this is mutable so cleanup passes can
    modify instructions in-place (e.g. NOP-ing unreachable bytecode).
    """

    opcode: int
    opname: str
    # arg:     raw integer argument (the number in the bytecode), may be an index into co_consts/co_varnames
    # argval:  Python object resolved by the dis module (value of co_consts[arg], or a variable name string)
    # argrepr: human-readable string of argval (e.g. "NULL + print", "to 20")
    # See bytecode_explained.py §1 for details
    arg: Optional[int]
    argval: Any
    argrepr: str
    offset: Optional[int] = None
    starts_line: Optional[int] = None
    is_jump_target: bool = False

    # -- identity / hashing (by object id, not value) ----------------------

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __repr__(self) -> str:
        return f"Instruction({self.opname}, offset={self.offset}, argval={self.argrepr!r})"

    # -- category queries ---------------------------------------------------

    @property
    def is_load(self) -> bool:
        return self.opname in _LOAD_OPCODES

    @property
    def is_store(self) -> bool:
        return self.opname in _STORE_OPCODES

    @property
    def is_delete(self) -> bool:
        return self.opname in _DELETE_OPCODES

    @property
    def is_jump(self) -> bool:
        return self.opcode in _ALL_JUMP_OPCODES

    @property
    def is_conditional_jump(self) -> bool:
        return self.is_jump and ("IF" in self.opname or "FOR_ITER" in self.opname)

    @property
    def is_unconditional_jump(self) -> bool:
        return self.is_jump and not self.is_conditional_jump

    @property
    def is_return(self) -> bool:
        return self.opname in ("RETURN_VALUE", "RETURN_CONST")

    @property
    def is_nop(self) -> bool:
        return self.opname == "NOP"

    # -- jump target --------------------------------------------------------

    def jump_target_offset(self) -> Optional[int]:
        """Return the absolute bytecode offset this instruction jumps to,
        or ``None`` if it is not a jump instruction."""
        if not self.is_jump:
            return None
        if "to " in self.argrepr:
            return int(self.argrepr.replace("to ", "").strip())
        if self.opcode in dis.hasjabs:
            return self.argval
        if self.opcode in dis.hasjrel:
            return self.argval if _PY311 else self.offset + self.argval
        return None

    # -- mutation helpers (for cleanup passes) ------------------------------

    def nop_(self) -> None:
        """In-place convert this instruction to a NOP."""
        self.opname = "NOP"
        self.opcode = dis.opmap["NOP"]
        self.arg = 0
        self.argval = 0
        self.argrepr = ""
        self.is_jump_target = False

    # -- factory ------------------------------------------------------------

    @staticmethod
    def from_dis(i: dis.Instruction) -> "Instruction":
        """Create from a stdlib ``dis.Instruction``."""
        return Instruction(i.opcode, i.opname, i.arg, i.argval, i.argrepr, i.offset, i.starts_line, i.is_jump_target)
