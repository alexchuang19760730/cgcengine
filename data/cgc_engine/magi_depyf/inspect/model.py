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

"""Data model for structured compilation output.

These dataclasses represent the full compilation state that
torch.compile produces, organized to reflect the actual runtime
structure: CacheEntry linked list, fn/resume recursion,
compiled_fn → backend mapping, and guard trees.
"""

from __future__ import annotations

import dataclasses
import dis
import inspect
import io
from types import CodeType
from typing import Dict, List, Optional


def format_code_info(code: CodeType) -> str:
    """Format key attributes of a CodeType for debugging."""
    lines: List[str] = []
    lines.append(f"co_name:          {code.co_name}")
    if hasattr(code, "co_qualname"):
        lines.append(f"co_qualname:      {code.co_qualname}")
    lines.append(f"co_filename:      {code.co_filename}")
    lines.append(f"co_firstlineno:   {code.co_firstlineno}")
    lines.append(f"co_argcount:      {code.co_argcount}")
    lines.append(f"co_kwonlyargcount:{code.co_kwonlyargcount}")
    lines.append(f"co_varnames:      {code.co_varnames}")
    lines.append(f"co_freevars:      {code.co_freevars}")
    lines.append(f"co_cellvars:      {code.co_cellvars}")
    lines.append(f"co_names:         {code.co_names}")
    flags = code.co_flags
    flag_strs = [name for name, val in _CODE_FLAGS.items() if flags & val]
    lines.append(f"co_flags:         0x{flags:04x} ({' | '.join(flag_strs) if flag_strs else 'none'})")
    lines.append(f"co_stacksize:     {code.co_stacksize}")
    lines.append("")
    lines.append("co_consts:")
    for i, c in enumerate(code.co_consts):
        lines.append(f"  [{i:3d}] {type(c).__name__:12s} {_safe_repr(c)}")
    lines.append("")
    lines.append("dis:")
    buf = io.StringIO()
    dis.dis(code, file=buf)
    lines.append(buf.getvalue())
    return "\n".join(lines)


_CODE_FLAGS = {
    "CO_OPTIMIZED": inspect.CO_OPTIMIZED,
    "CO_NEWLOCALS": inspect.CO_NEWLOCALS,
    "CO_VARARGS": inspect.CO_VARARGS,
    "CO_VARKEYWORDS": inspect.CO_VARKEYWORDS,
    "CO_NESTED": inspect.CO_NESTED,
    "CO_GENERATOR": inspect.CO_GENERATOR,
    "CO_COROUTINE": inspect.CO_COROUTINE,
    "CO_ASYNC_GENERATOR": inspect.CO_ASYNC_GENERATOR,
}


def _safe_repr(obj, max_len: int = 120) -> str:
    try:
        r = repr(obj)
    except Exception:
        r = f"<repr failed: {type(obj).__name__}>"
    if len(r) > max_len:
        r = r[: max_len - 3] + "..."
    return r


@dataclasses.dataclass
class GuardNode:
    """One node in the guard tree (mirrors RootGuardManager / GuardManager)."""

    type_name: str
    leaf_guards: List[str]
    children: List["GuardNode"] = dataclasses.field(default_factory=list)

    def format(self, depth: int = 0, max_depth: int = 32) -> str:
        prefix = "  " * depth
        lines = [f"{prefix}[{self.type_name}] " f"({len(self.leaf_guards)} leaf guards, {len(self.children)} children)"]
        for g in self.leaf_guards:
            lines.append(f"{prefix}  LEAF: {g}")
        if depth < max_depth:
            for i, child in enumerate(self.children):
                lines.append(f"{prefix}  child[{i}]:")
                lines.append(child.format(depth + 2, max_depth))
        elif self.children:
            lines.append(f"{prefix}  ... ({len(self.children)} children omitted)")
        return "\n".join(lines)


@dataclasses.dataclass
class SubgraphInfo:
    """One piecewise subgraph in the magi split pipeline."""

    name: str
    is_splitting_graph: bool = False
    readable_code: Optional[str] = None
    graph_module_code: Optional[str] = None
    fx_graph_tabular: Optional[str] = None
    inductor_code: Optional[str] = None

    def format(self) -> str:
        if self.inductor_code:
            return self.inductor_code
        if self.readable_code:
            return self.readable_code
        if self.graph_module_code:
            return self.graph_module_code
        tag = "splitting_op" if self.is_splitting_graph else "compiled"
        return f"# {self.name} ({tag})\n"


@dataclasses.dataclass
class CompiledFnInfo:
    """What __compiled_fn_xxx actually points to in the backend."""

    name: str
    backend: str  # "eager", "inductor", or "magi_compile"
    cudagraph_mode: Optional[str] = None  # "NONE", "PIECEWISE", "FULL" (magi_compile only)
    readable_code: Optional[str] = None
    graph_module_code: Optional[str] = None
    fx_graph_tabular: Optional[str] = None
    source_code: Optional[str] = None
    inductor_post_grad_graph: Optional[str] = None
    runnable_graph_str: Optional[str] = None
    cache_key: Optional[str] = None
    split_graph_readable: Optional[str] = None
    subgraph_infos: List["SubgraphInfo"] = dataclasses.field(default_factory=list)

    def format(self) -> str:
        """Full content for writing to file (compiled output)."""
        if self.source_code:
            return self.source_code
        if self.readable_code:
            return self.readable_code
        if self.graph_module_code:
            return self.graph_module_code
        return f"# {self.name} (backend={self.backend})\n"

    def format_summary(self) -> str:
        """Short summary for overview / full_code."""
        header = f"{self.name} (backend={self.backend}"
        if self.cudagraph_mode:
            header += f", cudagraph={self.cudagraph_mode}"
        header += ")"
        lines = [header]
        if self.cache_key:
            lines.append(f"  cache_key: {self.cache_key}")
        if self.graph_module_code:
            lines.append("  GraphModule.code:")
            for l in self.graph_module_code.strip().splitlines():
                lines.append(f"    {l}")
        if self.subgraph_infos:
            lines.append(f"  piecewise subgraphs: {len(self.subgraph_infos)}")
            for sg in self.subgraph_infos:
                tag = "splitting_op" if sg.is_splitting_graph else "compiled"
                lines.append(f"    {sg.name} ({tag})")
        return "\n".join(lines)


@dataclasses.dataclass
class GuardInfo:
    """Guard information for a CacheEntry."""

    tree: Optional[GuardNode] = None
    closure_vars: Optional[Dict[str, str]] = None

    def format(self) -> str:
        lines = []
        if self.tree:
            lines.append(self.tree.format())
        if self.closure_vars:
            lines.append("  closure_vars:")
            for k, v in list(self.closure_vars.items())[:8]:
                lines.append(f"    {k} = {v}")
        return "\n".join(lines)


@dataclasses.dataclass
class EntryInfo:
    """One CacheEntry in the linked list."""

    index: int
    dynamo_code: Optional[CodeType] = None
    decompiled_source: str = ""
    guard: Optional[GuardInfo] = None
    compiled_fns: List[CompiledFnInfo] = dataclasses.field(default_factory=list)
    resume_fns: List["FunctionInfo"] = dataclasses.field(default_factory=list)

    def format(self, indent: int = 0) -> str:
        pfx = "  " * indent
        lines = [f"{pfx}entry[{self.index}]:"]
        if self.decompiled_source:
            lines.append(f"{pfx}  dynamo_code (decompiled):")
            for l in self.decompiled_source.splitlines():
                lines.append(f"{pfx}    {l}")
        if self.compiled_fns:
            lines.append(f"{pfx}  compiled functions:")
            for cf in self.compiled_fns:
                lines.append(cf.format_summary())
        if self.guard:
            lines.append(f"{pfx}  guards:")
            lines.append(self.guard.format())
        if self.resume_fns:
            lines.append(f"{pfx}  resume functions:")
            for rf in self.resume_fns:
                lines.append(rf.format(indent + 2))
        return "\n".join(lines)


@dataclasses.dataclass
class FunctionInfo:
    """A compiled function and its CacheEntry chain."""

    name: str
    original_code: Optional[CodeType] = None
    original_source: str = ""
    entries: List[EntryInfo] = dataclasses.field(default_factory=list)

    def format(self, indent: int = 0) -> str:
        pfx = "  " * indent
        lines = [f"{pfx}{self.name}: {len(self.entries)} cache entries"]
        for entry in self.entries:
            lines.append(entry.format(indent + 1))
        return "\n".join(lines)
