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
omlx/predict_exec.py - 专家激活预测器
"""

import torch
import torch.nn as nn
from typing import List


class ExpertPredictor(nn.Module):
    """
    专家激活预测器

    调度层职责：
    - 基于输入特征预测将被激活的专家
    - 减少不必要的专家加载开销
    - 与 FlashMoE 协同实现智能预热
    """

    def __init__(self, input_dim: int = 4096, num_experts: int = 8):
        super().__init__()
        self.predictor = nn.Linear(input_dim, num_experts, bias=False)
        self._device = self._get_device()
        self.to(self._device)

    def predict(
        self,
        x: torch.Tensor,
        num_experts: int,
        top_k: int
    ) -> torch.Tensor:
        """
        预测 top_k 个将被激活的专家

        Args:
            x: 输入张量 [batch, seq_len, hidden_dim] 或 [batch, hidden_dim]
            num_experts: 总专家数
            top_k: 选择 top_k 个专家

        Returns:
            predicted_experts: [batch, top_k] 专家索引
        """
        if x.dim() == 3:
            x_pooled = x.mean(dim=1)
        elif x.dim() == 2:
            x_pooled = x
        else:
            x_pooled = x

        x_pooled = x_pooled.to(device=self._device, dtype=self.predictor.weight.dtype)
        logits = self.predictor(x_pooled)
        top_k_experts = torch.topk(logits, k=min(top_k, num_experts), dim=-1).indices

        return top_k_experts

    def _get_device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
