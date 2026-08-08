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

"""CaptureSession -- lifecycle-managed torch.compile interception.

All state is instance-scoped (no global dictionaries).  Only the
official ``register_bytecode_hook`` API is used -- no monkey-patching.
"""

from __future__ import annotations

import os
import sys
from types import CodeType, FrameType
from typing import List, Optional

from ..decompile import safe_decompile
from .result import CaptureResult

_MAX_FRAME_WALK = 64


class CaptureSession:
    """Context-manager that intercepts ``torch.compile`` bytecode events.

    Usage::

        with CaptureSession() as session:
            compiled_fn(input_tensor)
        for r in session.results:
            print(r.summary())
            print(r.decompiled_source)
    """

    def __init__(self) -> None:
        self._results: List[CaptureResult] = []
        self._hook_handle = None

    @property
    def results(self) -> List[CaptureResult]:
        return list(self._results)

    def __enter__(self) -> "CaptureSession":
        try:
            import torch._dynamo.convert_frame as cf
        except ImportError as e:
            raise ImportError("CaptureSession requires PyTorch (torch._dynamo). " "Install with: pip install torch") from e

        hook = self._make_hook(self._results)
        self._hook_handle = cf.register_bytecode_hook(hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def clear(self) -> None:
        """Discard all captured results."""
        self._results.clear()

    # ------------------------------------------------------------------
    # Hook internals
    # ------------------------------------------------------------------

    @staticmethod
    def _find_compile_frame(start: Optional[FrameType] = None) -> Optional[FrameType]:
        """Walk the call stack to find the ``_compile`` frame inside
        ``convert_frame.py``.  Returns ``None`` if not found within
        ``_MAX_FRAME_WALK`` steps.
        """
        frame = start or sys._getframe(1)
        for _ in range(_MAX_FRAME_WALK):
            if frame is None:
                return None
            name = frame.f_code.co_name
            filename = os.path.basename(frame.f_code.co_filename)
            if name == "_compile" and filename == "convert_frame.py":
                return frame
            frame = frame.f_back
        return None

    @staticmethod
    def _make_hook(results: List[CaptureResult]):
        """Return a bytecode-hook callable that appends ``CaptureResult``
        objects to *results*.
        """

        def hook(old_code: CodeType, new_code: CodeType) -> CodeType:
            compile_frame = CaptureSession._find_compile_frame()
            fn_globals = None
            if compile_frame is not None:
                inner_frame = compile_frame.f_locals.get("frame")
                if inner_frame is not None:
                    fn_globals = inner_frame.f_globals

            result = CaptureResult(
                function_name=old_code.co_name,
                original_code=old_code,
                dynamo_code=new_code,
                decompiled_source=safe_decompile(new_code),
                fn_globals=fn_globals,
            )
            results.append(result)
            return new_code

        return hook
