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

"""Pass 1: for-loop temp elimination.

``for __temp in iter: var = __temp; ...`` → ``for var in iter: ...``
"""

from __future__ import annotations

import ast

import astor


def eliminate_for_temps(source: str, temp_prefix: str = "__temp_", indent: int = 4) -> str:
    """Only applies when the first body statement is a plain assignment
    from the temp to a real variable."""
    try:
        tree = ast.parse(source)
        tree = _ForTempEliminator(temp_prefix).visit(tree)
        ast.fix_missing_locations(tree)
        return astor.to_source(tree, indent_with=" " * indent)
    except Exception:
        return source


class _ForTempEliminator(ast.NodeTransformer):
    def __init__(self, prefix: str):
        self._prefix = prefix

    def visit_For(self, node: ast.For) -> ast.For:
        self.generic_visit(node)
        if not (
            isinstance(node.target, ast.Name)
            and node.target.id.startswith(self._prefix)
            and node.body
            and isinstance(node.body[0], ast.Assign)
            and len(node.body[0].targets) == 1
            and isinstance(node.body[0].value, ast.Name)
            and node.body[0].value.id == node.target.id
        ):
            return node
        node.target = node.body[0].targets[0]
        node.body = node.body[1:] or [ast.Pass()]
        return node
