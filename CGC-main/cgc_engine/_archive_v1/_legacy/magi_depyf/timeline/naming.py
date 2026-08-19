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

import re
from typing import Any


def sanitize_event_fragment(fragment: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(fragment).strip()).strip("_").lower()
    return normalized or "unknown"


def event_name(base: str, subgraph_index: int | None = None) -> str:
    if subgraph_index is None:
        return f"fullgraph_{base}"
    return f"subgraph_{subgraph_index}_{base}"


def lifecycle_event_name(phase: str, lifecycle_name: str, subgraph_index: int | None = None) -> str:
    lifecycle_fragment = sanitize_event_fragment(lifecycle_name)
    suffix = f"{phase}_{lifecycle_fragment}"
    return event_name(suffix, subgraph_index)


def scope_attributes(subgraph_index: int | None = None) -> dict[str, Any]:
    if subgraph_index is None:
        return {"scope": "fullgraph"}
    return {"scope": "subgraph", "subgraph": f"submod_{subgraph_index}", "subgraph_index": subgraph_index}
