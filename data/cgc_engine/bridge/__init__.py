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
Megatrain ↔ vLLM Bridge Module

训推一体闭环:
    Megatrain 训练 → Bridge 转换 → vLLM 推理 → 导出模型
"""

from .megatrain_vllm_bridge import (
    MegatrainVLLMBridge,
    BridgeConfig,
    create_bridge,
)

from .lora_vllm_bridge import (
    LoRAtoVLLMBridge,
    LoRAWeightPaths,
    load_and_export,
)

__all__ = [
    "MegatrainVLLMBridge",
    "BridgeConfig",
    "create_bridge",
    "LoRAtoVLLMBridge",
    "LoRAWeightPaths",
    "load_and_export",
]
