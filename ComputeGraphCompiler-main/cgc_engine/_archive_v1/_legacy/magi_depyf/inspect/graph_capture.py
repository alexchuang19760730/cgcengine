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

"""Capture ``torch._inductor.graph.GraphLowering`` objects during compilation.

Install the hook via :func:`install_graph_lowering_hook` before model
compilation.  Each ``GraphLowering.compile_to_module()`` call appends the
object (which holds the scheduler fusion decisions, IR buffers, and the
post-grad FX graph) to a global list.

After compilation, :func:`pop_captured_graph_lowering` returns them in
FIFO order so that the introspector can attach one to each compiled submod.
"""

from __future__ import annotations

import contextlib
from collections import deque
from typing import Any

_captured: deque[Any] = deque()
_original_method: Any = None


def install_graph_lowering_hook() -> None:
    """Monkey-patch ``GraphLowering.compile_to_module`` to capture instances."""
    global _original_method
    if _original_method is not None:
        return  # already installed

    from torch._inductor.graph import GraphLowering

    _original_method = GraphLowering.compile_to_module

    def _patched(self):
        compiled_module = _original_method(self)
        # Read back the generated source from the compiled module's file
        files = {}
        try:
            src_path = getattr(compiled_module, "__file__", None)
            if src_path:
                with open(src_path) as f:
                    self._captured_inductor_source = f.read()
                    files["inductor_output.py"] = self._captured_inductor_source
            else:
                self._captured_inductor_source = None
        except Exception as e:
            from cgc_engine.utils import magi_logger

            magi_logger.warning(f"[magi_depyf] Error capturing inductor lowering: {e}")
        finally:
            _captured.append(self)

        # Publish this event through timeline event sink.
        from ..timeline import emit_after_inductor_schedule

        emit_after_inductor_schedule(self, files)

        return compiled_module

    GraphLowering.compile_to_module = _patched


def uninstall_graph_lowering_hook() -> None:
    """Restore original ``GraphLowering.compile_to_module``."""
    global _original_method
    if _original_method is None:
        return
    from torch._inductor.graph import GraphLowering

    GraphLowering.compile_to_module = _original_method
    _original_method = None


def pop_captured_graph_lowering() -> Any | None:
    """Return the next captured ``GraphLowering`` (FIFO), or ``None``."""
    return _captured.popleft() if _captured else None


def clear_captured() -> None:
    _captured.clear()


@contextlib.contextmanager
def capture_graph_lowerings():
    """Context manager: install hook, yield, then uninstall and clear."""
    clear_captured()
    install_graph_lowering_hook()
    try:
        yield
    finally:
        uninstall_graph_lowering_hook()
