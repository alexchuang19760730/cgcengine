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

from typing import Any

import torch.fx as fx


def fmt_gm(gm: fx.GraphModule) -> str:
    if hasattr(gm, "print_readable"):
        return "from __future__ import annotations\nimport torch\n" + gm.print_readable(print_output=False).replace(
            "<lambda>", "GraphModule"
        )
    return str(gm)


def fmt_compiled_graph_output(compiled_graph: Any) -> str:
    source = getattr(compiled_graph, "source_code", None)
    if isinstance(source, str) and source.strip():
        return source

    for attr in ("code", "generated_code", "python_code"):
        value = getattr(compiled_graph, attr, None)
        if isinstance(value, str) and value.strip():
            return value
        if callable(value):
            try:
                generated = value()
            except Exception:
                generated = None
            if isinstance(generated, str) and generated.strip():
                return generated

    return repr(compiled_graph)


def graph_files(prefix: str, graph: Any | None) -> dict[str, str]:
    if graph is None:
        return {}
    if isinstance(graph, fx.GraphModule):
        return {f"{prefix}_graph_module.py": fmt_gm(graph)}
    if isinstance(graph, fx.Graph):
        return {f"{prefix}_graph.txt": str(graph)}
    return {f"{prefix}_graph.txt": str(graph)}
