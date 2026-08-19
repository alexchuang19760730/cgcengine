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

"""explain_compilation -- unified entry-point for magi_depyf output.

Usage::

    from cgc_engine.magi_depyf import explain_compilation

    with explain_compilation("./debug_output"):
        model(inputs)

Captures both JIT bytecode events and backend-specific compilation events
(graph splitting, inductor output, cache hits) and writes them to disk.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

from cgc_engine.utils import magi_logger

from ..timeline import end_timeline, start_timeline
from .introspect import Introspector
from .session import CaptureSession
from .writer import write_function


@contextlib.contextmanager
def explain_compilation(output_dir: str):
    """Context manager that captures compilation events and writes output.

    Args:
        output_dir: Directory to write the human-readable output into.
    """
    dump_dir = Path(output_dir).resolve()
    dump_dir.mkdir(parents=True, exist_ok=True)

    timeline_dir = dump_dir / "timeline_events"
    start_timeline(timeline_dir)

    from .graph_capture import capture_graph_lowerings

    try:
        with CaptureSession() as jit_session, capture_graph_lowerings():
            yield
    finally:
        try:
            end_timeline()

            if "jit_session" in locals():
                seen = set()
                jit_out_dir = dump_dir / "compiled_functions"
                for r in jit_session.results:
                    name = r.original_code.co_name
                    if name in seen or name.startswith("torch_dynamo_resume_in_"):
                        continue
                    seen.add(name)
                    try:
                        info = Introspector.build_function_info(r.original_code, fn_globals=r.fn_globals)
                        write_function(info, jit_out_dir)
                    except Exception as e:
                        magi_logger.warning("[magi_depyf] JIT info failed for '%s': %s", name, e)
        except Exception as e:
            magi_logger.error(f"[magi_depyf] Error generating compilation explanation: {e}", exc_info=True)
