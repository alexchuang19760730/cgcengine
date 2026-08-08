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

"""Source-level post-processing pipeline for decompiled code.

Each pass is a function ``(source, ...) -> source`` that performs one
semantics-preserving transformation.  ``run_all`` applies them in order.
All passes are best-effort: on any exception they return the input unchanged.
"""

from .branch_dedup import dedup_branch_tails
from .for_temps import eliminate_for_temps
from .inline_temps import eliminate_inline_temps


def run_all(source: str, temp_prefix: str = "__temp_", indent: int = 4) -> str:
    """Apply all post-processing passes in sequence."""
    source = eliminate_for_temps(source, temp_prefix, indent)
    source = eliminate_inline_temps(source, temp_prefix, indent)
    source = dedup_branch_tails(source, indent)
    return source


__all__ = ["run_all", "eliminate_for_temps", "eliminate_inline_temps", "dedup_branch_tails"]
