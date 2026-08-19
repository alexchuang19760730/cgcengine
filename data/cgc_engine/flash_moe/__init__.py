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
FlashMoE - 跨平台 MoE 引擎
支持 Metal (Apple Silicon)、CUDA (NVIDIA)、CPU 多線程
"""

from .client import FlashMoEClient, BackendType
from .metal_infer import MetalMLPInfer
from .cuda_infer import CudaMLPInfer
from .cpu_infer import CPUMLPInfer
from .utils import ExpertCacheManager, load_expert_weights, save_expert_weights

__version__ = "0.2.0"
__all__ = [
    "FlashMoEClient",
    "BackendType",
    "MetalMLPInfer",
    "CudaMLPInfer",
    "CPUMLPInfer",
    "ExpertCacheManager",
    "load_expert_weights",
    "save_expert_weights",
]