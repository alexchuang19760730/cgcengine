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

"""File writer: serialize FunctionInfo tree to organized output files.

Output structure:
  output_dir/
    {fn_name}/
      decompiled_code.py          (original bytecode decompiled)
      overview.md
      entry_0/
        decompiled_code.py        (dynamo-transformed bytecode decompiled)
        guards.txt
        compiled_fns/
          __compiled_fn_xxx.py
        resume_fns/
          {resume_name}/
            decompiled_code.py
            overview.md
            entry_0/
              ...
      entry_1/
        ...
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from .model import CompiledFnInfo, EntryInfo, FunctionInfo, GuardNode, SubgraphInfo, format_code_info

_DECOMPILED = "decompiled_code.py"
_BYTECODE_INFO = "bytecode_info.txt"
_OVERVIEW = "overview.md"

# ── Convenience function (preserves existing public API) ─────────────


def write_function(fn_info: FunctionInfo, output_dir: str | Path) -> Path:
    """Write *fn_info* to a directory tree under *output_dir*.
    Returns the root directory created."""
    return FunctionWriter(fn_info, output_dir).write()


# ── Main class ───────────────────────────────────────────────────────


class FunctionWriter:
    """Serialize a :class:`FunctionInfo` tree into an organized directory."""

    def __init__(self, fn_info: FunctionInfo, output_dir: str | Path) -> None:
        self.fn_info = fn_info
        self.root = (Path(output_dir) / self._sanitize(fn_info.name)).resolve()

    def write(self) -> Path:
        """Write all output files.  Returns the root directory."""
        self.root.mkdir(parents=True, exist_ok=True)
        if self.fn_info.original_source:
            self._write_text(self.root / _DECOMPILED, self.fn_info.original_source)
        if self.fn_info.original_code is not None:
            self._write_text(self.root / _BYTECODE_INFO, format_code_info(self.fn_info.original_code))
        self._write_text(self.root / _OVERVIEW, self._format_overview(self.fn_info, self.root))
        for entry in self.fn_info.entries:
            self._write_entry(entry, self.root / f"entry_{entry.index}")
        return self.root

    # ── Entry writing ────────────────────────────────────────────────

    def _write_entry(self, entry: EntryInfo, entry_dir: Path) -> None:
        entry_dir.mkdir(parents=True, exist_ok=True)
        self._write_text(entry_dir / _DECOMPILED, entry.decompiled_source)
        if entry.dynamo_code is not None:
            self._write_text(entry_dir / _BYTECODE_INFO, format_code_info(entry.dynamo_code))

        if entry.guard:
            self._write_text(entry_dir / "guards.txt", entry.guard.format())

        if entry.compiled_fns:
            cfns_dir = entry_dir / "compiled_fns"
            cfns_dir.mkdir(parents=True, exist_ok=True)
            for cf in entry.compiled_fns:
                base = self._sanitize(cf.name)
                self._write_text(cfns_dir / f"{base}.py", cf.format())
                if cf.inductor_post_grad_graph:
                    self._write_text(cfns_dir / f"{base}_post_grad.py", cf.inductor_post_grad_graph)
                if cf.runnable_graph_str:
                    self._write_text(cfns_dir / f"{base}_runnable.py", cf.runnable_graph_str)
                if cf.split_graph_readable:
                    self._write_text(cfns_dir / f"{base}_split_graph.py", cf.split_graph_readable)
                for sg in cf.subgraph_infos:
                    self._write_subgraph(cfns_dir / base, sg)

        if entry.resume_fns:
            rfns_dir = entry_dir / "resume_fns"
            for rf in entry.resume_fns:
                sub_writer = FunctionWriter(rf, rfns_dir)
                sub_writer.write()

    # ── Subgraph writing (magi piecewise) ───────────────────────────

    def _write_subgraph(self, parent_dir: Path, sg: SubgraphInfo) -> None:
        sg_dir = parent_dir / sg.name
        sg_dir.mkdir(parents=True, exist_ok=True)
        tag = "splitting_op" if sg.is_splitting_graph else "compiled"
        if sg.readable_code:
            self._write_text(sg_dir / "graph_module.py", sg.readable_code)
        if sg.graph_module_code:
            self._write_text(sg_dir / "graph_module_code.py", sg.graph_module_code)
        if sg.fx_graph_tabular:
            self._write_text(sg_dir / "fx_graph_tabular.txt", sg.fx_graph_tabular)
        if sg.inductor_code:
            self._write_text(sg_dir / "inductor_output.py", sg.inductor_code)
        summary = f"# {sg.name} ({tag})\n"
        if sg.readable_code:
            summary += f"# graph_module.py: GraphModule.print_readable()\n"
        if sg.inductor_code:
            summary += f"# inductor_output.py: inductor kernel source\n"
        self._write_text(sg_dir / "README.txt", summary)

    # ── Overview (markdown) ───────────────────────────────────────────

    def _format_overview(self, fn_info: FunctionInfo, root: Path) -> str:
        lines: List[str] = []
        lines.append(f"# {fn_info.name}")
        lines.append("")
        lines.append(f"**Root:** `{self.root}`  ")
        lines.append(f"**Cache entries:** {len(fn_info.entries)}")
        lines.append("")

        if fn_info.original_source:
            lines.append(f"[decompiled code (before dynamo)](./{_DECOMPILED})")
        if fn_info.original_code is not None:
            lines.append(f"[bytecode info](./{_BYTECODE_INFO})")
        lines.append("")

        for entry in fn_info.entries:
            lines.extend(self._format_entry_md(entry, root))

        return "\n".join(lines) + "\n"

    def _format_entry_md(self, entry: EntryInfo, root: Path) -> List[str]:
        entry_dir = root / f"entry_{entry.index}"
        lines: List[str] = []
        lines.append(f"## entry\\[{entry.index}\\]")
        lines.append("")

        items = self._build_entry_items(entry, entry_dir, root)
        lines.extend(self._render_tree_md(items, depth=0))
        lines.append("")
        return lines

    def _build_entry_items(self, entry: EntryInfo, entry_dir: Path, root: Path) -> "List[_TreeItem]":
        items: List[_TreeItem] = []

        items.append(_TreeItem("decompiled code", self._rel(entry_dir / _DECOMPILED, root)))
        if entry.dynamo_code is not None:
            items.append(_TreeItem("bytecode info", self._rel(entry_dir / _BYTECODE_INFO, root)))

        if entry.guard:
            n = len(self._collect_leaf_guards(entry.guard.tree)) if entry.guard.tree else 0
            label = f"guards ({n} leaf)" if n else "guards"
            items.append(_TreeItem(label, self._rel(entry_dir / "guards.txt", root)))

        if entry.compiled_fns:
            cfns_dir = entry_dir / "compiled_fns"
            cf_children = self._build_compiled_fn_items(entry.compiled_fns, cfns_dir, root)
            items.append(_TreeItem("compiled_fns/", "", cf_children))

        if entry.resume_fns:
            rfns_dir = entry_dir / "resume_fns"
            rf_children: List[_TreeItem] = []
            for rf in entry.resume_fns:
                resume_dir = rfns_dir / self._sanitize(rf.name)
                fn_children: List[_TreeItem] = []
                if rf.original_source:
                    fn_children.append(_TreeItem("decompiled code", self._rel(resume_dir / _DECOMPILED, root)))
                for re in rf.entries:
                    sub_entry_dir = resume_dir / f"entry_{re.index}"
                    entry_children = self._build_entry_items(re, sub_entry_dir, root)
                    fn_children.append(
                        _TreeItem(f"entry\\[{re.index}\\]", self._rel(sub_entry_dir / _DECOMPILED, root), entry_children)
                    )
                rf_children.append(_TreeItem(f"`{rf.name}`/", self._rel(resume_dir / _OVERVIEW, root), fn_children))
            items.append(_TreeItem("resume_fns/", "", rf_children))

        return items

    def _build_compiled_fn_items(self, compiled_fns: List[CompiledFnInfo], cfns_dir: Path, root: Path) -> "List[_TreeItem]":
        items: List[_TreeItem] = []
        for cf in compiled_fns:
            base = self._sanitize(cf.name)
            label = f"`{cf.name}` ({cf.backend}"
            if cf.cudagraph_mode and cf.cudagraph_mode != "NONE":
                label += f", cudagraph={cf.cudagraph_mode}"
            label += ")"
            items.append(_TreeItem(label, self._rel(cfns_dir / f"{base}.py", root)))
            if cf.inductor_post_grad_graph:
                items.append(
                    _TreeItem(
                        f"`{cf.name}` (inductor_post_grad_graph_str)", self._rel(cfns_dir / f"{base}_post_grad.py", root)
                    )
                )
            if cf.runnable_graph_str:
                items.append(_TreeItem(f"`{cf.name}` (runnable_graph_str)", self._rel(cfns_dir / f"{base}_runnable.py", root)))
            if cf.split_graph_readable:
                items.append(_TreeItem(f"`{cf.name}` (split_graph)", self._rel(cfns_dir / f"{base}_split_graph.py", root)))
            if cf.subgraph_infos:
                sg_children: List[_TreeItem] = []
                for sg in cf.subgraph_infos:
                    sg_dir = cfns_dir / base / sg.name
                    tag = "splitting_op" if sg.is_splitting_graph else "compiled"
                    sg_sub: List[_TreeItem] = []
                    if sg.readable_code:
                        sg_sub.append(_TreeItem("graph_module", self._rel(sg_dir / "graph_module.py", root)))
                    if sg.inductor_code:
                        sg_sub.append(_TreeItem("inductor_output", self._rel(sg_dir / "inductor_output.py", root)))
                    if sg.fx_graph_tabular:
                        sg_sub.append(_TreeItem("fx_graph_tabular", self._rel(sg_dir / "fx_graph_tabular.txt", root)))
                    sg_children.append(_TreeItem(f"`{sg.name}` ({tag})", self._rel(sg_dir / "README.txt", root), sg_sub))
                items.append(_TreeItem("piecewise_subgraphs/", "", sg_children))
        return items

    @staticmethod
    def _render_tree_md(items: "List[_TreeItem]", depth: int = 0) -> List[str]:
        lines: List[str] = []
        indent = "  " * depth
        for item in items:
            if item.rel_path:
                lines.append(f"{indent}- [{item.label}](./{item.rel_path})")
            else:
                lines.append(f"{indent}- {item.label}")
            if item.children:
                lines.extend(FunctionWriter._render_tree_md(item.children, depth + 1))
        return lines

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _collect_leaf_guards(node: GuardNode) -> List[str]:
        guards = list(node.leaf_guards)
        for child in node.children:
            guards.extend(FunctionWriter._collect_leaf_guards(child))
        return guards

    @staticmethod
    def _rel(path: Path, base: Path) -> str:
        try:
            return str(path.relative_to(base))
        except ValueError:
            return str(path)

    @staticmethod
    def _write_text(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    @staticmethod
    def _sanitize(name: str) -> str:
        return name.replace("<", "_").replace(">", "_").replace(" ", "_").replace(".", "_")


class _TreeItem:
    __slots__ = ("label", "rel_path", "children")

    def __init__(self, label: str, rel_path: str, children: "List[_TreeItem] | None" = None):
        self.label = label
        self.rel_path = rel_path
        self.children = children or []
