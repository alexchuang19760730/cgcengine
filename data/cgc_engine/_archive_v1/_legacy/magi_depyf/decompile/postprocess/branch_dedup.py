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

"""Pass 3: if/else branch tail deduplication.

Move identical trailing statements from if/else branches to after the block.

Example::

    if cond:            if cond:
        x = 1               x = 1
        return x    →   else:
    else:                   x = 2
        x = 2           return x
        return x
"""

from __future__ import annotations

import ast
from typing import List, Tuple

import astor


def dedup_branch_tails(source: str, indent: int = 4) -> str:
    """Move identical trailing statements from if/else branches to after the block."""
    try:
        tree = ast.parse(source)
        changed = False
        for node in ast.walk(tree):
            if hasattr(node, "body") and isinstance(node.body, list):
                new_body, c = _dedup_stmts(node.body)
                if c:
                    node.body = new_body
                    changed = True
        if not changed:
            return source
        ast.fix_missing_locations(tree)
        return astor.to_source(tree, indent_with=" " * indent)
    except Exception:
        return source


def _dedup_stmts(stmts: List[ast.stmt]) -> Tuple[List[ast.stmt], bool]:
    """Process a statement list, extracting common if/else tails."""
    result: List[ast.stmt] = []
    changed = False

    for stmt in stmts:
        for attr in ("body", "orelse", "handlers", "finalbody"):
            sub = getattr(stmt, attr, None)
            if isinstance(sub, list) and sub:
                new_sub, c = _dedup_stmts(sub)
                if c:
                    setattr(stmt, attr, new_sub)
                    changed = True

        if isinstance(stmt, ast.If) and stmt.orelse:
            n = _common_tail_length(stmt.body, stmt.orelse)
            if n > 0:
                common = stmt.body[-n:]
                stmt.body = stmt.body[:-n] or [ast.Pass()]
                stmt.orelse = stmt.orelse[:-n] or []
                result.append(stmt)
                result.extend(common)
                changed = True
                continue

        result.append(stmt)

    return result, changed


def _common_tail_length(body: List[ast.stmt], orelse: List[ast.stmt]) -> int:
    """Count identical trailing statements (by AST dump equality)."""
    count = 0
    i, j = len(body) - 1, len(orelse) - 1
    while i >= 0 and j >= 0:
        if ast.dump(body[i]) == ast.dump(orelse[j]):
            count += 1
            i -= 1
            j -= 1
        else:
            break
    if count >= len(body) or count >= len(orelse):
        count = min(len(body), len(orelse)) - 1
    return max(count, 0)
