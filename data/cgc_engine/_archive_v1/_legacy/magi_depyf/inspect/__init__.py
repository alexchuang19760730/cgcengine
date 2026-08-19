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

"""Inspection layer: capture torch.compile events, introspect artifacts, write structured output."""

from typing import Optional

from .dump_src import explain_compilation
from .introspect import Introspector
from .model import CompiledFnInfo, EntryInfo, FunctionInfo, GuardInfo, GuardNode, SubgraphInfo
from .result import CaptureResult
from .session import CaptureSession
from .writer import FunctionWriter, write_function


def debug_compiled(fn, output_dir: Optional[str] = None) -> FunctionInfo:
    """Introspect a compiled function and optionally write debug output.

    Args:
        fn: The original (uncompiled) function.
        output_dir: If provided, write organized files to this directory.

    Returns:
        FunctionInfo with full compilation state.
    """
    info = Introspector.build_function_info(fn)
    if output_dir is not None:
        write_function(info, output_dir)
    return info


__all__ = [
    "CompiledFnInfo",
    "EntryInfo",
    "FunctionInfo",
    "GuardInfo",
    "GuardNode",
    "SubgraphInfo",
    "Introspector",
    "FunctionWriter",
    "write_function",
    "explain_compilation",
    "debug_compiled",
    "CaptureSession",
    "CaptureResult",
]
