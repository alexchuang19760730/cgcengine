# Copyright (c) 2025 SandAI. All Rights Reserved.
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

"""
UnifiedIOController - 跨平臺 I/O 統一調度

支援:
- GDS + SPDK (Linux NVIDIA)
- MPS + mmap (macOS Metal)
- CPU Fallback (PyTorch)

調度關係:
- Scheduler → UnifiedIOController (調度層)
- UnifiedIOController → 各後端 (存儲層)
"""

from .unified_io_controller import (
    UnifiedIOController,
    get_unified_io_controller,
    Platform,
    UnifiedIOConfig,
)

from .io_backend import IOBackend, IOStats

from .metal_backend import MetalBackend

from .pytorch_backend import PyTorchBackend

__version__ = "0.1.0"
__all__ = [
    "UnifiedIOController",
    "get_unified_io_controller",
    "Platform",
    "UnifiedIOConfig",
    "IOBackend",
    "IOStats",
    "MetalBackend",
    "PyTorchBackend",
]