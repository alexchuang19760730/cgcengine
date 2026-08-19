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

from .core import end_timeline, start_timeline
from .events import (
    emit_after_aot_artifact_save,
    emit_after_aot_compile,
    emit_after_dynamo_bytecode_transform,
    emit_after_dynamo_capture,
    emit_after_inductor_schedule,
    emit_after_magi_backend_done,
    emit_aot_cache_hit,
    emit_aot_cache_miss,
    emit_aot_retry,
    emit_before_aot_compile,
    emit_before_magi_compile,
)
from .lifecycle import (
    emit_after_lifecycle_run,
    emit_before_lifecycle_run,
    emit_lifecycle_run_failed,
    emit_pass_lifecycle,
    emit_skip_lifecycle_run,
    observe_lifecycle,
    observe_lifecycle_context,
)
from .registry import clear_attrs_resolvers, get_attrs_resolver, register_attrs_resolver
from .resolvers import (
    resolve_attrs_for_cache_load,
    resolve_attrs_for_cache_store,
    resolve_attrs_for_compiler_compile,
    resolve_attrs_for_compiler_manager_compile,
    resolve_attrs_for_graph_split,
    resolve_attrs_for_piecewise_compile,
)

__all__ = [
    "start_timeline",
    "end_timeline",
    "observe_lifecycle",
    "observe_lifecycle_context",
    "emit_pass_lifecycle",
    "emit_before_lifecycle_run",
    "emit_after_lifecycle_run",
    "emit_skip_lifecycle_run",
    "emit_lifecycle_run_failed",
    "register_attrs_resolver",
    "get_attrs_resolver",
    "clear_attrs_resolvers",
    "resolve_attrs_for_graph_split",
    "resolve_attrs_for_piecewise_compile",
    "resolve_attrs_for_cache_load",
    "resolve_attrs_for_compiler_manager_compile",
    "resolve_attrs_for_compiler_compile",
    "resolve_attrs_for_cache_store",
    "emit_before_magi_compile",
    "emit_after_dynamo_capture",
    "emit_after_magi_backend_done",
    "emit_after_dynamo_bytecode_transform",
    "emit_aot_cache_miss",
    "emit_aot_cache_hit",
    "emit_before_aot_compile",
    "emit_after_aot_compile",
    "emit_aot_retry",
    "emit_after_aot_artifact_save",
    "emit_after_inductor_schedule",
]
