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

"""Pass 2: single-use temp inlining.

``__temp = expr; use(__temp)`` → ``use(expr)`` for single-use temps.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from typing import List, Optional

import astor


def eliminate_inline_temps(source: str, temp_prefix: str = "__temp_", indent: int = 4) -> str:
    """Inline single-use temporaries into their use site."""
    try:
        tree = ast.parse(source)
        _set_parents(tree)

        occurrences: dict[str, list] = defaultdict(list)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id.startswith(temp_prefix):
                occurrences[node.id].append(node)

        _INDENT_NODES = (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.For,
            ast.AsyncFor,
            ast.While,
            ast.If,
            ast.Try,
            ast.With,
            ast.AsyncWith,
            ast.ClassDef,
        )

        for name in occurrences:
            occ = occurrences[name]
            if len(occ) == 2:
                n1, n2 = occ
                _, p1, p2 = _lowest_common_parent(n1, n2)
                ap = p1 if isinstance(getattr(n1, "parent", None), ast.Assign) else p2
                can = not isinstance(ap, _INDENT_NODES)
                if can:
                    can = _safe_to_inline(tree, n1, n2)
                occ.append(can)
            tree = _RemoveAssign(name, occurrences).visit(tree)
            tree = _InlineTemp(name, occurrences).visit(tree)

        return astor.to_source(tree, indent_with=" " * indent)
    except Exception:
        return source


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _set_parents(node: ast.AST, parent: Optional[ast.AST] = None) -> None:
    for child in ast.iter_child_nodes(node):
        child.parent = parent  # type: ignore[attr-defined]
        _set_parents(child, child)


def _get_parents(node: ast.AST) -> List[ast.AST]:
    out = []
    while node:
        out.append(node)
        node = getattr(node, "parent", None)
    return out


def _lowest_common_parent(n1: ast.AST, n2: ast.AST):
    p1 = _get_parents(n1)
    p2 = _get_parents(n2)
    p1.reverse()
    p2.reverse()
    last = c1 = c2 = None
    for a, b in zip(p1, p2):
        if a is b:
            last = a
        else:
            c1, c2 = a, b
            break
    return last, c1, c2


def _safe_to_inline(tree: ast.AST, def_node: ast.AST, use_node: ast.AST) -> bool:
    """Verify the RHS variable is not reassigned between definition and use."""
    assign_parent = getattr(def_node, "parent", None)
    if not isinstance(assign_parent, ast.Assign):
        return True
    rhs = assign_parent.value
    if not isinstance(rhs, ast.Name):
        return True

    rhs_name = rhs.id
    stmts: List[ast.stmt] = []
    for node in ast.walk(tree):
        if hasattr(node, "body") and isinstance(node.body, list):
            stmts = node.body
            break
    try:
        def_idx = next(i for i, s in enumerate(stmts) if s is assign_parent)
        use_stmt = getattr(use_node, "parent", None)
        while use_stmt and use_stmt not in stmts:
            use_stmt = getattr(use_stmt, "parent", None)
        use_idx = next(i for i, s in enumerate(stmts) if s is use_stmt)
    except StopIteration:
        return True

    for stmt in stmts[def_idx + 1 : use_idx]:
        if isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name) and t.id == rhs_name:
                    return False
    return True


class _RemoveAssign(ast.NodeTransformer):
    def __init__(self, name: str, occ: dict):
        self._name = name
        self._occ = occ

    def visit_Assign(self, node: ast.Assign):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            n = node.targets[0].id
            if n == self._name:
                o = self._occ[n]
                if len(o) == 1:
                    return ast.Expr(value=node.value)
                if len(o) == 3 and isinstance(o[-1], bool):
                    o.append(node.value)
                    if o[-2]:
                        return None
        return node


class _InlineTemp(ast.NodeTransformer):
    def __init__(self, name: str, occ: dict):
        self._name = name
        self._occ = occ

    def visit_Name(self, node: ast.Name):
        o = self._occ.get(node.id, [])
        if node.id == self._name and len(o) == 4 and isinstance(o[-2], bool) and o[-2]:
            return o[-1]
        return node
