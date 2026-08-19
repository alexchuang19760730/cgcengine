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

"""Runtime introspection of torch.compile artifacts.

Walk actual runtime state (CacheEntry chain, guard trees, __compiled_fn
objects) to build the structured model.  All torch imports are lazy so
this module can be imported without torch.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Optional

from ..decompile import safe_decompile
from .model import CompiledFnInfo, EntryInfo, FunctionInfo, GuardInfo, GuardNode, SubgraphInfo


class Introspector:
    """Namespace for runtime introspection helpers (all static methods)."""

    @staticmethod
    def get_cache_entries(fn) -> list:
        """Return CacheEntry list for *fn* (function or code object)."""
        from torch._dynamo.eval_frame import _debug_get_cache_entry_list

        code = fn.__code__ if hasattr(fn, "__code__") else fn
        return _debug_get_cache_entry_list(code)

    @staticmethod
    def build_guard_tree(node, max_depth: int = 32, _depth: int = 0) -> GuardNode:
        """Recursively build a GuardNode from a GuardManager C++ object."""
        type_name = type(node).__name__
        leaf_guards = []
        for lg in node.get_leaf_guards():
            for part in lg.verbose_code_parts():
                leaf_guards.append(part.strip()[:120])
        children = []
        if _depth < max_depth:
            for child in node.get_child_managers():
                children.append(Introspector.build_guard_tree(child, max_depth, _depth + 1))
        return GuardNode(type_name=type_name, leaf_guards=leaf_guards, children=children)

    @staticmethod
    def extract_guard_info(entry) -> Optional[GuardInfo]:
        """Extract structured guard info from a CacheEntry (post-hoc introspection).

        Operates on the persisted CacheEntry and builds a full GuardNode tree.
        """
        try:
            gm = entry.guard_manager
            tree = Introspector.build_guard_tree(gm.root)
            closure_vars: Dict[str, str] = {}
            if hasattr(gm, "closure_vars") and gm.closure_vars:
                for k, v in list(gm.closure_vars.items())[:10]:
                    closure_vars[k] = repr(v)[:100]
            return GuardInfo(tree=tree, closure_vars=closure_vars or None)
        except Exception:
            return None

    @staticmethod
    def extract_compiled_fn_info(name: str, fn_globals: dict) -> Optional[CompiledFnInfo]:
        """Inspect a __compiled_fn_xxx from fn.__globals__.

        Handles three backend types:
          eager:        wrapper -> closure[0]=GraphModule.forward (bound method)
          inductor:     wrapper -> closure[0]=aot_forward -> ... -> CompiledFxGraph
          magi_compile: MagiSerializableFunction -> split_gm -> PiecewiseBackend(s)
        """
        obj = fn_globals.get(name)
        if obj is None:
            return None

        magi_info = Introspector._try_extract_magi_info(name, obj)
        if magi_info is not None:
            return magi_info

        info = CompiledFnInfo(name=name, backend="eager")

        gm = Introspector._find_graph_module(obj)
        if gm is not None:
            Introspector._fill_graph_module_info(info, gm)

        cfx = Introspector._find_compiled_fx_graph(obj)
        if cfx is not None:
            info.backend = "inductor"
            Introspector._fill_compiled_fx_graph_info(info, cfx)

        return info

    @staticmethod
    def _fill_graph_module_info(info: CompiledFnInfo, gm) -> None:
        try:
            info.readable_code = gm.print_readable(print_output=False)
        except Exception:
            pass
        try:
            info.graph_module_code = str(gm.code) if hasattr(gm, "code") else None
        except Exception:
            pass
        try:
            buf = io.StringIO()
            gm.graph.print_tabular(file=buf)
            info.fx_graph_tabular = buf.getvalue()
        except Exception:
            pass

    @staticmethod
    def _fill_compiled_fx_graph_info(info: CompiledFnInfo, cfx) -> None:
        try:
            info.source_code = cfx.source_code
        except Exception:
            pass
        try:
            info.inductor_post_grad_graph = cfx.inductor_post_grad_graph_str
        except Exception:
            pass
        try:
            info.cache_key = cfx.cache_key
        except Exception:
            pass
        try:
            info.runnable_graph_str = cfx.runnable_graph_str
        except Exception:
            pass

    # -- Magi backend introspection ----------------------------------------

    @staticmethod
    def _try_extract_magi_info(name: str, obj) -> Optional[CompiledFnInfo]:
        """Detect MagiSerializableFunction and walk its hierarchy.

        MagiSerializableFunction hierarchy:
          .graph_module   → fx.GraphModule (full graph before splitting)
          .optimized_call → split_gm (fx.GraphModule with PiecewiseBackend submodules)
            .submod_N → PiecewiseBackend
              .graph                           → fx.GraphModule (the subgraph)
              .compiled_graph_for_general_shape → inductor compiled output

        Dynamo wraps the backend result in a DisableContext closure, so the
        MagiSerializableFunction may live one level deep in the closure chain.
        """
        msf = obj if (hasattr(obj, "graph_module") and hasattr(obj, "optimized_call")) else None
        if msf is None and callable(obj) and getattr(obj, "__closure__", None):
            for cell in obj.__closure__:
                try:
                    val = cell.cell_contents
                except ValueError:
                    continue
                if hasattr(val, "graph_module") and hasattr(val, "optimized_call"):
                    msf = val
                    break
        if msf is None:
            return None
        obj = msf

        import torch.fx

        info = CompiledFnInfo(name=name, backend="magi_compile")

        full_gm = getattr(obj, "graph_module", None)
        if isinstance(full_gm, torch.fx.GraphModule):
            Introspector._fill_graph_module_info(info, full_gm)

        split_gm = getattr(obj, "optimized_call", None)

        # In FULL cudagraph mode, optimized_call is a wrapper function whose
        # __dict__ carries the GraphModule's attributes (via __dict__.update).
        # Unwrap to find the actual GraphModule for print_readable / named_children.
        actual_gm = split_gm if isinstance(split_gm, torch.fx.GraphModule) else None
        if actual_gm is None and split_gm is not None:
            actual_gm = Introspector._find_graph_module_deep(split_gm)

        info.cudagraph_mode = Introspector._detect_cudagraph_mode(split_gm, actual_gm)

        if actual_gm is not None:
            try:
                info.split_graph_readable = actual_gm.print_readable(print_output=False)
            except Exception:
                pass

            # PiecewiseCompileInterpreter replaces submodules via __dict__,
            # so named_children() still sees the original GraphModules while
            # __dict__ contains the PiecewiseBackend (or cudagraph wrapper).
            # In FULL cudagraph mode, those __dict__ entries are copied onto
            # the wrapper function, so we look up runtime objects from
            # split_gm (the wrapper) rather than actual_gm.
            runtime_source = split_gm if split_gm is not None else actual_gm
            for sub_name, original_gm in actual_gm.named_children():
                runtime_obj = runtime_source.__dict__.get(sub_name, original_gm)
                sg_info = Introspector._extract_subgraph_info(sub_name, runtime_obj, original_gm)
                if sg_info is not None:
                    info.subgraph_infos.append(sg_info)

            info.subgraph_infos.sort(key=lambda s: s.name)

        return info

    @staticmethod
    def _extract_subgraph_info(sub_name: str, runtime_obj, original_gm=None) -> Optional[SubgraphInfo]:
        """Extract info from one submodule of the split graph.

        Args:
            sub_name: The submodule name (e.g. "submod_0").
            runtime_obj: The actual runtime object — PiecewiseBackend,
                         cudagraph wrapper, or the original GraphModule.
            original_gm: The original GraphModule before replacement (from _modules).
        """
        import torch.fx

        piecewise = Introspector._unwrap_piecewise_backend(runtime_obj)

        if piecewise is not None:
            sg = SubgraphInfo(name=sub_name, is_splitting_graph=False)
            inner_gm = getattr(piecewise, "graph", None)
            if isinstance(inner_gm, torch.fx.GraphModule):
                Introspector._fill_subgraph_gm_info(sg, inner_gm)

            compiled = getattr(piecewise, "compiled_graph_for_general_shape", None)
            if compiled is not None:
                sg.inductor_code = Introspector._try_extract_inductor_source(compiled)

            if sg.inductor_code is None:
                sg.inductor_code = Introspector._read_artifact_source_from_piecewise(piecewise)

            return sg

        gm = original_gm if isinstance(original_gm, torch.fx.GraphModule) else None
        if gm is None and isinstance(runtime_obj, torch.fx.GraphModule):
            gm = runtime_obj

        if gm is not None:
            sg = SubgraphInfo(name=sub_name, is_splitting_graph=True)
            Introspector._fill_subgraph_gm_info(sg, gm)
            return sg

        return None

    @staticmethod
    def _unwrap_piecewise_backend(obj):
        """Find a PiecewiseBackend from obj, unwrapping closures/wrappers if needed."""
        if hasattr(obj, "graph") and hasattr(obj, "compiled_graph_for_general_shape"):
            return obj

        if callable(obj) and hasattr(obj, "__closure__") and obj.__closure__:
            for cell in obj.__closure__:
                try:
                    val = cell.cell_contents
                except ValueError:
                    continue
                if hasattr(val, "graph") and hasattr(val, "compiled_graph_for_general_shape"):
                    return val
        return None

    @staticmethod
    def _fill_subgraph_gm_info(sg: SubgraphInfo, gm) -> None:
        try:
            sg.readable_code = gm.print_readable(print_output=False)
        except Exception:
            pass
        try:
            sg.graph_module_code = str(gm.code) if hasattr(gm, "code") else None
        except Exception:
            pass
        try:
            buf = io.StringIO()
            gm.graph.print_tabular(file=buf)
            sg.fx_graph_tabular = buf.getvalue()
        except Exception:
            pass

    @staticmethod
    def _try_extract_inductor_source(compiled) -> Optional[str]:
        """Try to extract inductor kernel source from a compiled graph object.

        Handles CompiledFxGraph, CompiledArtifact, and closure-wrapped variants.
        """
        for attr in ("source_code", "_source_code"):
            val = getattr(compiled, attr, None)
            if isinstance(val, str) and val:
                return val

        cfx = Introspector._find_compiled_fx_graph(compiled)
        if cfx is not None:
            try:
                return cfx.source_code
            except Exception:
                pass

        if hasattr(compiled, "print_readable"):
            try:
                return compiled.print_readable(print_output=False)
            except Exception:
                pass

        return None

    @staticmethod
    def _read_artifact_source_from_piecewise(piecewise) -> Optional[str]:
        """Read Inductor-generated source from the saved artifact directory.

        PiecewiseBackend stores a compiler_manager whose cache maps
        CacheEntry(runtime_shape, graph_index, backend_name) → CacheHandle(key, path).
        The artifact at CacheHandle.path is an unpacked directory containing
        ``py/*.py`` — the full Inductor output code.
        """
        try:
            compiler_manager = getattr(piecewise, "compiler_manager", None)
            if compiler_manager is None:
                return None
            cache = getattr(compiler_manager, "cache", None)
            if not cache:
                return None
            index = getattr(piecewise, "piecewise_compile_index", None)
            if index is None:
                return None

            for cache_entry, cache_handle in cache.items():
                if cache_entry.graph_index == index and cache_entry.runtime_shape is None:
                    artifact_path = getattr(cache_handle, "path", None)
                    if artifact_path:
                        return Introspector._read_py_from_artifact(artifact_path)
            return None
        except Exception:
            return None

    @staticmethod
    def _read_py_from_artifact(artifact_path: str) -> Optional[str]:
        """Read the Inductor-generated Python wrapper from an artifact directory.

        The unpacked artifact layout varies across PyTorch versions; the
        wrapper ``.py`` file has been observed under ``yb/`` and ``py/``.
        We try known directories first, then fall back to scanning all
        immediate subdirectories.
        """
        root = Path(artifact_path)
        if not root.is_dir():
            return None

        for candidate_dir in ("yb", "py"):
            d = root / candidate_dir
            if d.is_dir():
                py_files = sorted(d.glob("*.py"))
                if py_files:
                    try:
                        return py_files[0].read_text(encoding="utf-8")
                    except Exception:
                        pass

        for d in sorted(root.iterdir()):
            if d.is_dir():
                py_files = sorted(d.glob("*.py"))
                if py_files:
                    try:
                        return py_files[0].read_text(encoding="utf-8")
                    except Exception:
                        pass
        return None

    @staticmethod
    def _detect_cudagraph_mode(split_gm, actual_gm) -> str:
        """Detect cudagraph wrapping mode from the split graph structure.

        - FULL:      split_gm itself is a cudagraph wrapper (not a GraphModule),
                     with __qualname__ containing 'Athena_CUDAGraph_full'.
        - PIECEWISE: split_gm is a GraphModule, but its __dict__ submodules are
                     cudagraph wrappers with __qualname__ 'Athena_CUDAGraph_piecewise'.
        - NONE:      no cudagraph wrapping detected.
        """
        _CG_PREFIX = "Athena_CUDAGraph_"

        qualname = getattr(split_gm, "__qualname__", "") or ""
        if qualname.startswith(f"{_CG_PREFIX}full"):
            return "FULL"

        if actual_gm is not None:
            for key, val in actual_gm.__dict__.items():
                if not key.startswith("submod_"):
                    continue
                sub_qualname = getattr(val, "__qualname__", "") or ""
                if sub_qualname.startswith(f"{_CG_PREFIX}piecewise"):
                    return "PIECEWISE"

        return "NONE"

    @staticmethod
    def _find_graph_module_deep(obj, _depth: int = 0, _max_depth: int = 4) -> Optional[Any]:
        """Recursively walk closure chain to find a ``torch.fx.GraphModule``.

        This is needed for FULL cudagraph mode where the split GraphModule is
        wrapped by ``gen_wrap_func_for_cudagraph`` (+ ``@instrument_nvtx``),
        placing the GraphModule 2-3 levels deep in the closure chain.
        """
        import torch.fx

        if isinstance(obj, torch.fx.GraphModule):
            return obj
        if _depth >= _max_depth:
            return None
        if not callable(obj) or not getattr(obj, "__closure__", None):
            return None
        for cell in obj.__closure__:
            try:
                val = cell.cell_contents
            except ValueError:
                continue
            if isinstance(val, torch.fx.GraphModule):
                return val
            if callable(val):
                found = Introspector._find_graph_module_deep(val, _depth + 1, _max_depth)
                if found is not None:
                    return found
        return None

    @staticmethod
    def _find_graph_module(obj) -> Optional[Any]:
        """Walk closure chain to find a torch.fx.GraphModule."""
        import torch.fx

        if isinstance(obj, torch.fx.GraphModule):
            return obj
        if hasattr(obj, "__self__") and isinstance(obj.__self__, torch.fx.GraphModule):
            return obj.__self__
        if not callable(obj) or not hasattr(obj, "__closure__") or not obj.__closure__:
            return None
        for cell in obj.__closure__:
            try:
                val = cell.cell_contents
            except ValueError:
                continue
            if isinstance(val, torch.fx.GraphModule):
                return val
            if hasattr(val, "__self__") and isinstance(val.__self__, torch.fx.GraphModule):
                return val.__self__
        return None

    @staticmethod
    def _find_compiled_fx_graph(obj, _depth: int = 0) -> Optional[Any]:
        """Walk closure chain (up to 4 levels) to find a CompiledFxGraph."""
        if _depth > 4:
            return None
        try:
            from torch._inductor.codecache import CompiledFxGraph
        except ImportError:
            return None
        if isinstance(obj, CompiledFxGraph):
            return obj
        if not callable(obj) or not hasattr(obj, "__closure__") or not obj.__closure__:
            return None
        for cell in obj.__closure__:
            try:
                val = cell.cell_contents
            except ValueError:
                continue
            if isinstance(val, CompiledFxGraph):
                return val
            if callable(val):
                found = Introspector._find_compiled_fx_graph(val, _depth + 1)
                if found is not None:
                    return found
        return None

    @staticmethod
    def build_entry_info(entry, index: int, fn_globals: dict) -> EntryInfo:
        """Build an EntryInfo from a CacheEntry."""
        tc = entry.code
        decompiled = safe_decompile(tc)

        compiled_names = [n for n in tc.co_names if n.startswith("__compiled")]
        compiled_fns = []
        for cn in compiled_names:
            cf = Introspector.extract_compiled_fn_info(cn, fn_globals)
            if cf:
                compiled_fns.append(cf)

        resume_names = [n for n in tc.co_names if n.startswith("__resume")]
        resume_fns = []
        for rn in resume_names:
            rfn = fn_globals.get(rn)
            if rfn is not None and hasattr(rfn, "__code__"):
                resume_info = Introspector.build_function_info(rfn, fn_globals=fn_globals)
                resume_info.name = rn
                resume_fns.append(resume_info)

        guard = Introspector.extract_guard_info(entry)

        return EntryInfo(
            index=index,
            dynamo_code=tc,
            decompiled_source=decompiled,
            guard=guard,
            compiled_fns=compiled_fns,
            resume_fns=resume_fns,
        )

    @staticmethod
    def build_function_info(fn, fn_globals: Optional[dict] = None) -> FunctionInfo:
        """Build full FunctionInfo by walking CacheEntry chain."""
        if fn_globals is None:
            fn_globals = fn.__globals__ if hasattr(fn, "__globals__") else {}

        code = fn.__code__ if hasattr(fn, "__code__") else fn
        name = code.co_name
        original_source = safe_decompile(code)

        entries_raw = Introspector.get_cache_entries(fn)
        entries = []
        for i, raw_entry in enumerate(entries_raw):
            entries.append(Introspector.build_entry_info(raw_entry, i, fn_globals))

        return FunctionInfo(name=name, original_code=code, original_source=original_source, entries=entries)
