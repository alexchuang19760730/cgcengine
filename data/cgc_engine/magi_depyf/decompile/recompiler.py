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

"""CodeRecompiler: round-trip decompile -> compile -> extract target CodeType.

Pipeline: CodeType -> decompile -> compile -> find target.
"""

from __future__ import annotations

from types import CodeType
from typing import List

from .decompiler import Decompiler


class CodeRecompiler:
    """Decompile *code*, recompile, and produce a compatible ``CodeType``."""

    @staticmethod
    def recompile(
        code_to_decompile: CodeType, reference_code: CodeType, indentation: int = 4, temp_prefix: str = "__temp_"
    ) -> CodeType:
        """Full round-trip: decompile -> compile -> find target."""
        fn_name = reference_code.co_name

        src = Decompiler(code_to_decompile).decompile(
            indentation=indentation, temp_prefix=temp_prefix, overwrite_fn_name=fn_name
        )

        compiled = compile(src, "noname", "exec")
        all_codes = CodeRecompiler.collect_code_objects(compiled)
        return [c for c in all_codes if c.co_name == fn_name][0]

    @staticmethod
    def collect_code_objects(code: CodeType) -> List[CodeType]:
        """Recursively collect all ``CodeType`` objects from *code*."""
        result = [code]
        for c in code.co_consts:
            if isinstance(c, CodeType):
                result.extend(CodeRecompiler.collect_code_objects(c))
        return result
