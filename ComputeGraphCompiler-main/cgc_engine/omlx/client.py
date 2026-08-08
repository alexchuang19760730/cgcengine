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
omlx/client.py - oMLX 核心客户端（对接 CGC 调度层）
"""

import torch
import os
from pathlib import Path
from typing import List, Optional
from .flash_cache import OMLXFlashCache
from .predict_exec import ExpertPredictor


class OMLXClient:
    """
    oMLX 客户端 - 专家预测与缓存调度入口

    调度层职责：
    - 专家激活预测（减少不必要的专家加载）
    - 两级缓存调度（显存热缓存 + SSD 冷缓存）
    - 淘汰策略执行（LRU/FIFO）
    """

    def __init__(
        self,
        model_dir: str = "/tmp/omlx_model",
        *,
        num_experts: int = 8,
        expert_dim: int = 4096,
        intermediate_dim: Optional[int] = None,
        gpu_cache_size: int = 2,
        ssd_cache_dir: str = "/tmp/omlx_ssd_cache",
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.num_experts = int(num_experts)
        self.expert_dim = int(expert_dim)
        self.intermediate_dim = int(intermediate_dim) if intermediate_dim is not None else int(self.expert_dim * 4)

        self.predictor = ExpertPredictor(input_dim=self.expert_dim, num_experts=self.num_experts)
        self.flash_cache = OMLXFlashCache(
            gpu_cache_size=int(gpu_cache_size),
            ssd_cache_dir=str(ssd_cache_dir),
            expert_dim=self.expert_dim,
            intermediate_dim=self.intermediate_dim,
        )

        self._device = self._get_device()

    def predict_experts(self, x: torch.Tensor = None, top_k: int = 2, **params) -> torch.Tensor:
        """
        CGC 0xD2 指令：oMLX 专家激活预测

        调度层职责：
        - 使用预测模型预测将被激活的专家
        - 触发缓存预热（提前加载预测到的专家）
        - 返回 top_k 个预测专家 ID
        """
        if x is None:
            x = torch.randn(1, self.expert_dim, device=self._device)

        x = x.to(self._device)
        predicted_experts = self.predictor.predict(x, self.num_experts, top_k)

        expert_ids = predicted_experts.flatten().tolist()
        self.flash_cache.preload_experts(expert_ids)

        return predicted_experts

    def evict_cache(self, strategy: str = "lru", **params) -> bool:
        """
        CGC 0xD3 指令：oMLX 专家缓存管理

        调度层职责：
        - 执行缓存淘汰策略
        - 将淘汰的专家权重写回 SSD
        """
        self.flash_cache.evict(strategy=strategy)
        return True

    def update_cache(self, expert_id: int, **params) -> bool:
        self.flash_cache.preload_experts([int(expert_id)])
        return True

    def evict(self, expert_id: int, **params) -> bool:
        expert_id = int(expert_id)
        if expert_id in self.flash_cache.gpu_cache:
            self.flash_cache._evict_single_to_ssd(expert_id)
            return True
        return False

    def get_cached_experts(self) -> List[int]:
        """获取当前缓存的专家列表"""
        return list(self.flash_cache.gpu_cache.keys())

    def _get_device(self):
        """自动适配设备"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def info(self):
        return {
            "device": str(self._device),
            "num_experts": self.num_experts,
            "cached_experts": self.get_cached_experts(),
            "gpu_cache_size": len(self.flash_cache.gpu_cache),
        }
