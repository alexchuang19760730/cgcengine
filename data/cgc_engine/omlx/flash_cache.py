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
omlx/flash_cache.py - oMLX 两级缓存（显存 + SSD）

通过 UnifiedIOController 统一调度 SSD 存储操作
"""

import torch
import os
from pathlib import Path
from typing import Dict, List, Optional, Any


class OMLXFlashCache:
    """
    oMLX 两级缓存管理器

    存储层职责：
    - 显存缓存（热专家）- GPU/Metal 高带宽访问
    - SSD 缓存（冷专家）- UnifiedIOController 统一调度（GDS/SPDK）
    - LRU/FIFO 淘汰策略

    改进：通过 UnifiedIOController 实现统一的 SSD 存储管理
    """

    def __init__(
        self,
        gpu_cache_size: int = 2,
        ssd_cache_dir: str = "/tmp/omlx_ssd_cache",
        use_unified_io: bool = True,
        expert_dim: int = 4096,
        intermediate_dim: int = 14336,
    ):
        self.gpu_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self.gpu_cache_size = gpu_cache_size
        self.access_order: List[int] = []

        self.ssd_cache_dir = Path(ssd_cache_dir)
        self.ssd_cache_dir.mkdir(parents=True, exist_ok=True)

        self._device = self._get_device()
        self._expert_dim = int(expert_dim)
        self._intermediate_dim = int(intermediate_dim)
        self._dtype = torch.float16 if self._device.type in ("cuda", "mps") else torch.float32
        self._use_unified_io = use_unified_io
        self._unified_io = None

        if self._use_unified_io:
            self._init_unified_io()

    def _init_unified_io(self):
        """初始化 UnifiedIOController"""
        try:
            from cgc_engine.io_unified.unified_io_controller import UnifiedIOController
            self._unified_io = UnifiedIOController.get_instance()
        except ImportError:
            self._unified_io = None
            self._use_unified_io = False

    def _expert_base_path(self, expert_id: int) -> str:
        return str(self.ssd_cache_dir / f"expert_{expert_id}.pt")

    def _load_from_ssd(self, expert_id: int) -> Optional[Dict[str, torch.Tensor]]:
        """从 SSD 加载专家权重（通过 UnifiedIOController）"""
        if self._unified_io:
            try:
                return self._unified_io.load_expert_mlp(
                    expert_id=expert_id,
                    base_path=self._expert_base_path(expert_id),
                    expert_dim=self._expert_dim,
                    intermediate_dim=self._intermediate_dim,
                    dtype=self._dtype,
                )
            except Exception:
                pass

        ssd_path = self.ssd_cache_dir / f"expert_{expert_id}.pt"
        if ssd_path.exists():
            try:
                obj: Any = torch.load(ssd_path, map_location=self._device)
                if isinstance(obj, dict) and "w1" in obj and "w2" in obj:
                    if "w3" not in obj:
                        obj["w3"] = obj["w1"].clone()
                    return {
                        "w1": obj["w1"].to(self._device),
                        "w3": obj["w3"].to(self._device),
                        "w2": obj["w2"].to(self._device),
                    }
                if torch.is_tensor(obj):
                    if obj.ndim == 2 and obj.shape[0] == 3 * self._intermediate_dim and obj.shape[1] == self._expert_dim:
                        w1 = obj[: self._intermediate_dim, :]
                        w3 = obj[self._intermediate_dim : 2 * self._intermediate_dim, :]
                        w2 = obj[2 * self._intermediate_dim :, :].t()
                        return {"w1": w1.to(self._device), "w3": w3.to(self._device), "w2": w2.to(self._device)}
            except Exception:
                pass
        return None

    def _save_to_ssd(self, expert_id: int, tensor: Dict[str, torch.Tensor]):
        """保存专家权重到 SSD（通过 UnifiedIOController）"""
        if self._unified_io:
            try:
                self._unified_io.save_expert_mlp(expert_id, self._expert_base_path(expert_id), tensor)
                return
            except Exception:
                pass

        ssd_path = self.ssd_cache_dir / f"expert_{expert_id}.pt"
        obj = {k: v.detach().cpu() for k, v in tensor.items()}
        torch.save(obj, ssd_path)

    def preload_experts(self, expert_ids: List[int]):
        """
        预热缓存：将预测到的专家加载到显存缓存

        调度层职责：
        - 检查专家是否已在显存缓存
        - 未命中时从 SSD 加载（通过 UnifiedIOController）
        - 缓存满时执行淘汰
        """
        for idx in expert_ids:
            if idx in self.gpu_cache:
                self.access_order.remove(idx)
                self.access_order.append(idx)
                continue

            expert_weight = self._load_from_ssd(idx)
            if expert_weight is None:
                w1 = torch.randn((self._intermediate_dim, self._expert_dim), dtype=self._dtype, device=self._device)
                w3 = torch.randn((self._intermediate_dim, self._expert_dim), dtype=self._dtype, device=self._device)
                w2 = torch.randn((self._expert_dim, self._intermediate_dim), dtype=self._dtype, device=self._device)
                expert_weight = {"w1": w1, "w3": w3, "w2": w2}

            if len(self.gpu_cache) >= self.gpu_cache_size:
                self._evict_to_ssd()

            self.gpu_cache[idx] = expert_weight
            self.access_order.append(idx)

    def evict(self, strategy: str = "lru"):
        """
        缓存淘汰调度

        调度层职责：
        - 执行指定的淘汰策略
        - 将淘汰的专家写入 SSD（通过 UnifiedIOController）
        """
        if strategy == "lru" and self.access_order:
            self._evict_to_ssd()
        elif strategy == "fifo" and self.gpu_cache:
            evict_idx = next(iter(self.gpu_cache.keys()))
            self._evict_single_to_ssd(evict_idx)

    def _evict_to_ssd(self):
        """淘汰最久未访问的专家到 SSD"""
        if not self.access_order:
            return
        evict_idx = self.access_order.pop(0)
        self._evict_single_to_ssd(evict_idx)

    def _evict_single_to_ssd(self, evict_idx: int):
        """淘汰单个专家到 SSD（通过 UnifiedIOController）"""
        if evict_idx in self.gpu_cache:
            evict_weight = self.gpu_cache.pop(evict_idx)
            self._save_to_ssd(evict_idx, evict_weight)
            if evict_idx in self.access_order:
                self.access_order.remove(evict_idx)

    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        stats = {
            "gpu_cache_size": len(self.gpu_cache),
            "gpu_cache_max": self.gpu_cache_size,
            "gpu_cached_experts": list(self.gpu_cache.keys()),
            "access_order": self.access_order,
            "use_unified_io": self._use_unified_io,
        }

        if self._unified_io:
            try:
                stats["unified_io_stats"] = self._unified_io.get_stats()
            except Exception:
                pass

        return stats

    def _get_device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
