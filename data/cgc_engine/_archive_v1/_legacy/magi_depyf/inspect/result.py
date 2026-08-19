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

"""CaptureResult — structured data model for one compilation event."""

from __future__ import annotations

import dataclasses
import time
from types import CodeType
from typing import List, Optional


@dataclasses.dataclass
class CaptureResult:
    """Data captured from a single ``torch.compile`` bytecode event.

    - original_code: the user's original function code
    - dynamo_code: the code after Dynamo transformation (with __compiled_fn / __resume calls)
    - decompiled_source: dynamo_code decompiled back to Python source
    - fn_globals: the function's global namespace (for post-hoc introspection)
    """

    function_name: str
    original_code: CodeType
    dynamo_code: CodeType
    decompiled_source: str
    fn_globals: Optional[dict] = None
    guards: List[str] = dataclasses.field(default_factory=list)
    graph_source: Optional[str] = None
    timestamp: float = dataclasses.field(default_factory=time.time)

    def summary(self) -> str:
        n_guards = len(self.guards)
        return (
            f"[{self.function_name}] "
            f"original={self.original_code.co_name}, "
            f"guards={n_guards}, "
            f"graph={'yes' if self.graph_source else 'no'}"
        )
