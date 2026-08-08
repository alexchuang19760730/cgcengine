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
flash_moe/client.py - FlashMoE 核心客户端（跨平台调度入口）
"""

import torch
from pathlib import Path
from typing import List, Optional, Literal, Dict, Any
from enum import Enum

from .utils import load_expert_weights, ExpertCacheManager
from .metal_infer import MetalMLPInfer
from .cuda_infer import CudaMLPInfer
from .cpu_infer import CPUMLPInfer


class BackendType(Enum):
    """支持的後端類型"""
    METAL = "metal"
    CUDA = "cuda"
    CPU = "cpu"
    AUTO = "auto"


class FlashMoEClient:
    """
    FlashMoE 客户端 - 跨平台 MoE 引擎调度入口

    支持平台：
    - Apple Silicon (Metal)
    - NVIDIA GPU (CUDA)
    - CPU (多线程)

    调度层职责：
    - 自动选择最优后端
    - 专家权重按需加载（通过 GDS/SPDK 读取 SSD）
    - 专家缓存管理（LRU 策略）
    - 计算调度到对应平台执行
    """

    def __init__(
        self,
        expert_dir: str = "/tmp/flash_moe_experts",
        backend: Literal["metal", "cuda", "cpu", "auto"] = "auto",
        num_threads: Optional[int] = None,
    ):
        self.expert_dir = Path(expert_dir)
        self.expert_dir.mkdir(parents=True, exist_ok=True)

        self.cache_manager = ExpertCacheManager(max_cache_size=4)

        self.num_experts = 8
        self.expert_dim = 4096
        self.intermediate_dim = self.expert_dim * 4

        self._backend_type = BackendType(backend) if backend != "auto" else BackendType.AUTO
        self._backend = None
        self._backend_name = None
        self._device = None

        self._init_backend(num_threads)

    def _init_backend(self, num_threads: Optional[int] = None):
        """初始化后端"""
        if self._backend_type == BackendType.AUTO:
            self._select_auto_backend()
        else:
            self._backend_name = self._backend_type.value
            self._init_specific_backend(self._backend_type, num_threads)

    def _select_auto_backend(self):
        """自动选择最优后端"""
        if torch.cuda.is_available():
            self._backend_type = BackendType.CUDA
            self._init_specific_backend(BackendType.CUDA)
        elif torch.backends.mps.is_available():
            self._backend_type = BackendType.METAL
            self._init_specific_backend(BackendType.METAL)
        else:
            self._backend_type = BackendType.CPU
            self._init_specific_backend(BackendType.CPU)

    def _init_specific_backend(self, backend_type: BackendType, num_threads: Optional[int] = None):
        """初始化指定后端"""
        if backend_type == BackendType.METAL:
            self._backend = MetalMLPInfer()
            self._backend_name = "metal"
            self._device = torch.device("mps")
        elif backend_type == BackendType.CUDA:
            self._backend = CudaMLPInfer()
            self._backend_name = "cuda"
            self._device = torch.device("cuda")
        else:
            self._backend = CPUMLPInfer(num_threads=num_threads)
            self._backend_name = "cpu"
            self._device = torch.device("cpu")

    @property
    def backend_info(self):
        """獲取後端信息"""
        return {
            "backend": self._backend_name,
            "device": str(self._device),
            "available": self._backend.available if hasattr(self._backend, 'available') else True,
        }

    def load_experts(self, expert_ids: List[int] = None, **params) -> torch.Tensor:
        """
        CGC 0xD0 指令：FlashMoE 专家权重按需加载

        调度层职责：
        - 检查专家缓存
        - 缓存未命中时从 SSD 加载（调用 GDS/SPDK）
        - 返回加载的专家权重张量
        """
        expert_ids = expert_ids or [0, 1, 2, 3]
        experts = []

        for idx in expert_ids:
            if idx in self.cache_manager:
                experts.append(self.cache_manager[idx]["w1"])
            else:
                weight_path = self.expert_dir / f"expert_{idx}.bin"
                shape = [self.intermediate_dim, self.expert_dim]
                expert_weight = load_expert_weights(weight_path, shape, device=self._device)
                self.cache_manager[idx] = expert_weight
                experts.append(expert_weight["w1"])

        return torch.stack(experts, dim=0)

    def load_expert(
        self,
        expert_id: int,
        expert_path: Optional[str] = None,
        *,
        expert_dim: Optional[int] = None,
        intermediate_dim: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        if expert_id in self.cache_manager:
            return self.cache_manager[expert_id]

        expert_dim = int(expert_dim if expert_dim is not None else self.expert_dim)
        intermediate_dim = int(intermediate_dim if intermediate_dim is not None else self.intermediate_dim)

        if expert_path:
            weight_path = Path(expert_path)
        else:
            weight_path = self.expert_dir / f"expert_{expert_id}.bin"

        expert_weight = load_expert_weights(weight_path, [intermediate_dim, expert_dim], device=self._device)
        self.cache_manager[expert_id] = expert_weight
        return expert_weight

    def mlp_forward(
        self,
        x: torch.Tensor,
        expert_ids: List[int] = None,
        **params
    ) -> torch.Tensor:
        """
        CGC 0xD1 指令：FlashMoE 专家 MLP 执行

        执行层职责：
        - 调度到对应平台后端执行
        - 执行 SwiGLU MLP 计算
        """
        expert_ids = expert_ids or [0]
        x = x.to(self._device)

        if hasattr(self._backend, 'run'):
            return self._backend.run(x, expert_ids, self.cache_manager)
        else:
            return self._pytorch_mlp_fallback(x, expert_ids)

    def mlp_forward_moe(
        self,
        x: torch.Tensor,
        expert_ids: List[int] = None,
        top_k: int = 2,
        **params,
    ) -> torch.Tensor:
        """
        CGC 0xD1 指令：FlashMoE MoE 执行

        执行层职责：
        - 调度到对应平台后端执行
        - 计算多个专家的加权和
        """
        expert_ids = expert_ids or list(range(top_k))
        x = x.to(self._device)

        if hasattr(self._backend, 'run_moe'):
            return self._backend.run_moe(x, expert_ids, self.cache_manager, top_k=top_k)
        else:
            return self._pytorch_mlp_fallback(x, expert_ids[:top_k])

    moe_forward = mlp_forward_moe

    def _pytorch_mlp_fallback(self, x: torch.Tensor, expert_ids: List[int]) -> torch.Tensor:
        """PyTorch 降级执行（无 GPU 时）"""
        outputs = []
        for idx in expert_ids:
            expert_weight: Dict[str, Any] = self.cache_manager[idx]
            w1 = expert_weight["w1"].to(self._device)
            w3 = expert_weight["w3"].to(self._device)
            w2 = expert_weight["w2"].to(self._device)
            gate = torch.nn.functional.silu(torch.nn.functional.linear(x, w1))
            up = torch.nn.functional.linear(x, w3)
            output = torch.nn.functional.linear(gate * up, w2)
            outputs.append(output)

        return torch.stack(outputs, dim=0).mean(dim=0)

    def evict_cache(self, strategy: str = "lru") -> bool:
        """
        缓存淘汰调度

        调度层职责：
        - 执行 LRU/FIFO 淘汰策略
        - 将淘汰的专家权重写回 SSD（通过 GDS/SPDK）
        """
        self.cache_manager.evict(strategy)
        return True

    def set_backend(self, backend: Literal["metal", "cuda", "cpu"]):
        """
        动态切换后端

        Args:
            backend: 后端类型 ("metal", "cuda", "cpu")
        """
        self._backend_type = BackendType(backend)
        self._init_specific_backend(self._backend_type)

    def info(self):
        return {
            "device": str(self._device),
            "backend": self._backend_name,
            "num_experts": self.num_experts,
            "expert_dim": self.expert_dim,
            "cached_experts": len(self.cache_manager),
            **self.backend_info,
        }

    def forward_with_auto_load(self, x, top_k=2, expert_ids=None):
        """
        MoE forward with automatic expert loading.
        
        If expert_ids is None, uses all cached experts.
        If expert_ids are provided but not in cache, automatically loads them.
        
        Args:
            x: Input tensor
            top_k: Number of experts per token
            expert_ids: Optional list of expert IDs to use
        
        Returns:
            Output tensor
        """
        if expert_ids is None:
            # Use all cached experts
            expert_ids = list(self.cache_manager.keys())
            if not expert_ids:
                raise ValueError("No experts loaded in cache")
        else:
            # Ensure all requested experts are loaded
            for eid in expert_ids:
                if eid not in self.cache_manager:
                    self.load_experts([eid])
        
        return self.moe_forward(x, expert_ids=expert_ids, top_k=top_k)
    
