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

from .core import emit_event
from .formatters import fmt_gm
from .naming import event_name, scope_attributes


def emit_before_magi_compile(original_source_file: str):
    emit_event(
        name=event_name("before_magi_compile"),
        files={"original_source.py": original_source_file},
        attributes=scope_attributes(),
    )


def emit_after_dynamo_capture(graph: fx.GraphModule):
    emit_event(
        name=event_name("after_dynamo_capture"), files={"captured_graph.py": fmt_gm(graph)}, attributes=scope_attributes()
    )


def emit_after_magi_backend_done(compile_config: Any, split_gm: fx.GraphModule):
    emit_event(
        name=event_name("after_magi_backend_done"),
        attributes={
            **scope_attributes(),
            "cudagraph_mode": compile_config.cudagraph_mode.name if compile_config.cudagraph_mode else "NONE",
            "offload": compile_config.offload_config.model_cpu_offload,
        },
        files={"split_gm.py": fmt_gm(split_gm)},
    )


def emit_after_dynamo_bytecode_transform():
    emit_event(name=event_name("after_dynamo_bytecode_transform"), attributes=scope_attributes())


def emit_aot_cache_miss(aot_path: str, reason: str):
    attributes = scope_attributes()
    attributes["aot_path"] = aot_path
    attributes["reason"] = reason
    emit_event(name=event_name("aot_cache_miss"), attributes=attributes)


def emit_aot_cache_hit(aot_path: str):
    attributes = scope_attributes()
    attributes["aot_path"] = aot_path
    emit_event(name=event_name("aot_cache_hit"), attributes=attributes)


def emit_before_aot_compile(max_retries: int):
    attributes = scope_attributes()
    attributes["max_retries"] = max_retries
    emit_event(name=event_name("before_aot_compile"), attributes=attributes)


def emit_after_aot_compile(attempt: int):
    attributes = scope_attributes()
    attributes["attempt"] = attempt
    emit_event(name=event_name("after_aot_compile"), attributes=attributes)


def emit_aot_retry(attempt: int, max_retries: int):
    attributes = scope_attributes()
    attributes["attempt"] = attempt
    attributes["max_retries"] = max_retries
    emit_event(name=event_name("aot_retry"), attributes=attributes)


def emit_after_aot_artifact_save(aot_path: str):
    attributes = scope_attributes()
    attributes["aot_path"] = aot_path
    emit_event(name=event_name("after_aot_artifact_save"), attributes=attributes)


def emit_after_inductor_schedule(graph_lowering: Any, files: dict[str, str]):
    emit_event(
        name=event_name("after_inductor_schedule"),
        files=files,
        attributes={
            **scope_attributes(),
            "scheduler_node_count": len(graph_lowering.scheduler.nodes)
            if hasattr(graph_lowering, "scheduler") and hasattr(graph_lowering.scheduler, "nodes")
            else 0,
        },
    )
