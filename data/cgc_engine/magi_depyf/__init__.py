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

"""magi_depyf — a modern bytecode decompiler and torch.compile inspector."""

from .decompile import DecompilationError, Decompiler, decompile, safe_decompile
from .inspect import explain_compilation

__version__ = "0.1.0"

__all__ = ["Decompiler", "decompile", "safe_decompile", "DecompilationError", "__version__"]
