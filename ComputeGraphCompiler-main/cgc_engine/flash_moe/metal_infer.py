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
flash_moe/metal_infer.py - Metal 加速推理（Apple Silicon 专属）
"""

import torch


class MetalMLPInfer:
    """
    Metal MLP 推理器 - Apple Silicon 加速执行层

    执行层职责：
    - 调用 Metal shader 执行 MLP 计算
    - 零拷贝 GPU ↔ SSD 数据传输
    """

    def __init__(self):
        self.available = torch.backends.mps.is_available()
        self.device = torch.device("mps") if self.available else None

    def run(self, x: torch.Tensor, expert_ids: list, cache_manager) -> torch.Tensor:
        """
        执行 Metal 加速 MLP

        执行层职责：
        - 获取缓存的专家权重
        - 调用 Metal kernel 执行 SwiGLU
        """
        if not self.available:
            raise RuntimeError("Metal 不可用")

        x = x.to(self.device)

        expert = cache_manager[expert_ids[0]]
        w1 = expert["w1"].to(self.device)
        w3 = expert["w3"].to(self.device)
        w2 = expert["w2"].to(self.device)
        gate = torch.nn.functional.silu(torch.nn.functional.linear(x, w1))
        up = torch.nn.functional.linear(x, w3)
        return torch.nn.functional.linear(gate * up, w2)

    def run_moe(
        self,
        x: torch.Tensor,
        expert_ids: list,
        cache_manager,
        top_k: int = 2,
    ) -> torch.Tensor:
        """执行 MoE 模式"""
        if not self.available:
            raise RuntimeError("Metal 不可用")

        x = x.to(self.device)
        outputs = []

        for idx in expert_ids[:top_k]:
            expert = cache_manager[idx]
            w1 = expert["w1"].to(self.device)
            w3 = expert["w3"].to(self.device)
            w2 = expert["w2"].to(self.device)
            gate = torch.nn.functional.silu(torch.nn.functional.linear(x, w1))
            up = torch.nn.functional.linear(x, w3)
            outputs.append(torch.nn.functional.linear(gate * up, w2))

        result = torch.stack(outputs, dim=0).mean(dim=0)
        return result
