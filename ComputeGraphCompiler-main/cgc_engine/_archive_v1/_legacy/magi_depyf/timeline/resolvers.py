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

import os
import re
from typing import Any

import torch.fx as fx

from .formatters import fmt_compiled_graph_output, fmt_gm
from .registry import register_attrs_resolver


@register_attrs_resolver("graph_split")
def resolve_attrs_for_graph_split(
    phase: str, _call_args: tuple[Any, ...], _call_kwargs: dict[str, Any], result: Any | None, _error: Exception | None
) -> dict[str, Any] | None:
    if phase == "after" and isinstance(result, tuple) and len(result) == 2:
        _split_gm, piecewise_graphs = result
        return {"num_subgraphs": len(piecewise_graphs)}
    return None


@register_attrs_resolver("compiler_manager_load")
def resolve_attrs_for_cache_load(
    phase: str, call_args: tuple[Any, ...], call_kwargs: dict[str, Any], result: Any | None, _error: Exception | None
) -> dict[str, Any] | None:
    attrs: dict[str, Any] = {}
    cache_entry = None
    if len(call_args) >= 4 and hasattr(call_args[3], "graph_index") and hasattr(call_args[3], "runtime_shape"):
        cache_entry = call_args[3]
    elif call_kwargs.get("cache_entry") is not None:
        candidate = call_kwargs["cache_entry"]
        if hasattr(candidate, "graph_index") and hasattr(candidate, "runtime_shape"):
            cache_entry = candidate

    if cache_entry is not None:
        attrs.update(
            {
                "graph_index": cache_entry.graph_index,
                "runtime_shape": cache_entry.runtime_shape,
                "backend_name": getattr(cache_entry, "backend_name", None),
            }
        )

    manager = call_args[0] if call_args else None
    cache_handle = None
    if manager is not None and cache_entry is not None and hasattr(manager, "cache"):
        cache_handle = manager.cache.get(cache_entry)

    if phase == "before":
        attrs["cache_has_entry"] = cache_handle is not None
        return attrs or None

    if phase != "after":
        return attrs or None

    if cache_handle is None:
        attrs.update({"load_result": "miss", "reason": "entry_not_found"})
        return attrs

    attrs["cache_key"] = cache_handle.key or ""

    if len(call_args) >= 1 and hasattr(call_args[0], "_remaining_restart_skips"):
        remaining = call_args[0]._remaining_restart_skips.get(cache_entry.graph_index)
        if remaining is not None and remaining >= 0:
            attrs["remaining_restart_skip_count"] = remaining

    if result is None:
        attrs.update({"load_result": "miss", "reason": "cache_entry_load_miss"})
        return attrs

    attrs.update({"load_result": "hit", "reason": "cache_hit"})
    return attrs


@register_attrs_resolver("compiler_manager_compile")
def resolve_attrs_for_compiler_manager_compile(
    phase: str, call_args: tuple[Any, ...], _call_kwargs: dict[str, Any], result: Any | None, _error: Exception | None
) -> dict[str, Any] | None:
    attrs: dict[str, Any] = {}
    graph_index = call_args[4] if len(call_args) > 4 else None
    runtime_shape = call_args[6] if len(call_args) > 6 else None

    if isinstance(graph_index, int):
        attrs["graph_index"] = graph_index
    if runtime_shape is not None:
        attrs["runtime_shape"] = runtime_shape

    if phase == "after":
        attrs["compile_result"] = "hit" if result is not None else "miss"

    return attrs or None


@register_attrs_resolver("compiler_compile")
def resolve_attrs_for_compiler_compile(
    phase: str, call_args: tuple[Any, ...], _call_kwargs: dict[str, Any], result: Any | None, error: Exception | None
) -> dict[str, Any] | None:
    attrs: dict[str, Any] = {}
    runtime_shape = call_args[4] if len(call_args) > 4 else None
    key = call_args[5] if len(call_args) > 5 else None
    graph_index = None

    if runtime_shape is not None:
        attrs["runtime_shape"] = runtime_shape
    if key is not None:
        attrs["key"] = key

    if key is not None:
        match = re.search(r"_subgraph_(\d+)$", str(key))
        if match:
            graph_index = int(match.group(1))

    if graph_index is None:
        try:
            from ...passes.pass_base import get_pass_context

            graph_index = get_pass_context().subgraph_index
        except Exception:
            graph_index = None

    if isinstance(graph_index, int):
        attrs["graph_index"] = graph_index

    if phase == "before":
        return attrs or None

    if phase == "failed":
        attrs["compile_result"] = "failed"
        if isinstance(error, Exception):
            attrs["error_type"] = type(error).__name__
            attrs["error_message"] = str(error)
            restart_reason = getattr(error, "restart_reason", None)
            if restart_reason is not None:
                attrs["restart_reason"] = str(restart_reason)
            if key is not None and hasattr(call_args[0], "_restart_analysis_counts"):
                restart_count = call_args[0]._restart_analysis_counts.get(key, 0)
                attrs["restart_count"] = restart_count
        return attrs

    if phase == "after":
        compiled_graph = None
        cache_handle = None
        if isinstance(result, tuple) and len(result) == 2:
            compiled_graph, cache_handle = result
        attrs["compile_result"] = "hit" if compiled_graph is not None else "miss"
        attrs["cache_handle_available"] = cache_handle is not None
        if compiled_graph is not None:
            attrs["compiled_graph_available"] = True
            graph = call_args[1] if len(call_args) > 1 else None
            if isinstance(graph, fx.GraphModule):
                attrs["__files__"] = {
                    "graph_module.py": fmt_gm(graph),
                    "inductor_output.py": fmt_compiled_graph_output(compiled_graph),
                }
        return attrs

    return attrs or None


@register_attrs_resolver("compiler_manager_cache_store")
def resolve_attrs_for_cache_store(
    phase: str, call_args: tuple[Any, ...], _call_kwargs: dict[str, Any], result: Any | None, _error: Exception | None
) -> dict[str, Any] | None:
    attrs: dict[str, Any] = {}
    cache_entry = call_args[1] if len(call_args) > 1 else None
    cache_handle = call_args[2] if len(call_args) > 2 else None
    runtime_shape = call_args[3] if len(call_args) > 3 else None
    key = call_args[4] if len(call_args) > 4 else None

    if cache_entry is not None and hasattr(cache_entry, "graph_index"):
        attrs["graph_index"] = cache_entry.graph_index
    if runtime_shape is not None:
        attrs["runtime_shape"] = runtime_shape
    if key is not None:
        attrs["key"] = key

    if phase == "before":
        manager = call_args[0] if call_args else None
        attrs["cache_disabled"] = bool(getattr(manager, "disable_cache", False)) if manager is not None else False
        attrs["cache_handle_available"] = cache_handle is not None
        return attrs or None

    if phase != "after":
        return attrs or None

    stored = bool(result)
    if stored:
        attrs["store_result"] = "stored"
    else:
        manager = call_args[0] if call_args else None
        cache_disabled = bool(getattr(manager, "disable_cache", False)) if manager is not None else False
        attrs["store_result"] = "skipped"
        if cache_disabled:
            attrs["reason"] = "cache_disabled"
        elif cache_handle is None:
            attrs["reason"] = "compiler_returned_no_cache_handle"
        else:
            attrs["reason"] = "store_condition_not_met"
    return attrs


@register_attrs_resolver("piecewise_compile")
def resolve_attrs_for_piecewise_compile(
    phase: str, call_args: tuple[Any, ...], _call_kwargs: dict[str, Any], _result: Any | None, _error: Exception | None
) -> dict[str, Any] | None:
    if phase not in {"before", "after", "failed"}:
        return None
    if not call_args:
        return None

    interpreter = call_args[0]
    submod_names_to_compile = getattr(interpreter, "compile_submod_names", None)
    if not isinstance(submod_names_to_compile, list):
        return None

    return {"num_submods": len(submod_names_to_compile)}


@register_attrs_resolver("aot_cache_load")
def resolve_attrs_for_aot_cache_load(
    phase: str, call_args: tuple[Any, ...], _call_kwargs: dict[str, Any], result: Any | None, _error: Exception | None
) -> dict[str, Any] | None:
    if phase not in {"before", "after"}:
        return None
    if not call_args:
        return None

    state = call_args[0]
    aot_path = getattr(state, "aot_compilation_path", None)
    attrs: dict[str, Any] = {}
    if isinstance(aot_path, str):
        attrs["aot_path"] = aot_path

    if phase == "before":
        return attrs or None

    load_result = "hit" if result else "miss"
    attrs["load_result"] = load_result
    if load_result == "miss" and isinstance(aot_path, str):
        if not os.path.exists(aot_path):
            attrs["reason"] = "file_not_found"
        elif not os.path.exists(os.path.join(os.path.dirname(aot_path), "source_meta.json")):
            attrs["reason"] = "source_meta_missing"
        else:
            attrs["reason"] = "cache_entry_load_miss"

    return attrs


@register_attrs_resolver("aot_compile")
def resolve_attrs_for_aot_compile(
    phase: str, call_args: tuple[Any, ...], _call_kwargs: dict[str, Any], _result: Any | None, _error: Exception | None
) -> dict[str, Any] | None:
    if not call_args:
        return None

    state = call_args[0]
    max_retries = getattr(state, "_AOT_MAX_RETRIES", None)
    retry_count = getattr(state, "_aot_retry_count", 0)

    attrs: dict[str, Any] = {}
    if isinstance(max_retries, int):
        attrs["max_retries"] = max_retries
    if isinstance(retry_count, int):
        attrs["retry_count"] = retry_count
        attrs["retried"] = retry_count > 0

    if phase == "before":
        return attrs or None

    if phase == "after":
        attrs["compile_result"] = "hit"
        return attrs

    if phase == "failed":
        attrs["compile_result"] = "failed"
        return attrs

    return None


@register_attrs_resolver("aot_artifact_save")
def resolve_attrs_for_aot_artifact_save(
    phase: str, call_args: tuple[Any, ...], _call_kwargs: dict[str, Any], _result: Any | None, _error: Exception | None
) -> dict[str, Any] | None:
    if phase not in {"before", "after", "failed"}:
        return None
    if not call_args:
        return None

    state = call_args[0]
    aot_path = getattr(state, "aot_compilation_path", None)

    attrs: dict[str, Any] = {}
    if isinstance(aot_path, str):
        attrs["aot_path"] = aot_path

    if phase == "after":
        attrs["save_result"] = "saved"
    elif phase == "failed":
        attrs["save_result"] = "failed"

    return attrs or None
