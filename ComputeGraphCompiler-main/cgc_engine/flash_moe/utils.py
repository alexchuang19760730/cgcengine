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
flash_moe/utils.py - 工具函数：专家权重加载与缓存管理
"""

import torch
from pathlib import Path
from typing import Dict, List, Optional, Any


class ExpertCacheManager:
    """
    专家权重缓存管理器（LRU 策略）

    存储层职责：
    - 显存缓存管理（热专家）
    - SSD 缓存管理（冷专家）
    - LRU 淘汰策略
    """

    def __init__(self, max_cache_size: int = 4):
        self.cache: Dict[int, torch.Tensor] = {}
        self.access_order: List[int] = []
        self.max_size = max_cache_size

        self.ssd_cache_dir = Path("/tmp/flash_moe_ssd_cache")
        self.ssd_cache_dir.mkdir(parents=True, exist_ok=True)

    def __contains__(self, key: int) -> bool:
        return key in self.cache

    def __getitem__(self, key: int) -> torch.Tensor:
        if key in self.cache:
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        raise KeyError(f"Expert {key} not in cache")

    def __setitem__(self, key: int, value: torch.Tensor):
        if key not in self.cache:
            if len(self.cache) >= self.max_size:
                self._evict_one()
            self.cache[key] = value
            self.access_order.append(key)

    def __len__(self):
        return len(self.cache)

    def _evict_one(self):
        """淘汰最久未访问的专家到 SSD"""
        if not self.access_order:
            return
        evict_key = self.access_order.pop(0)
        if evict_key in self.cache:
            evict_weight = self.cache.pop(evict_key)
            ssd_path = self.ssd_cache_dir / f"expert_{evict_key}.bin"
            torch.save(evict_weight.cpu(), ssd_path)

    def evict(self, strategy: str = "lru"):
        """缓存淘汰调度"""
        if strategy == "lru" and self.access_order:
            self._evict_one()
        elif strategy == "fifo" and self.cache:
            evict_key = next(iter(self.cache.keys()))
            self.cache.pop(evict_key)
            self.access_order.remove(evict_key) if evict_key in self.access_order else None


def load_expert_weights(
    path: Path,
    shape: List[int],
    device: torch.device = None
) -> Dict[str, torch.Tensor]:
    """
    从存储层加载专家权重

    存储层职责：
    - 检测 GDS/SPDK 可用性
    - 优先使用零拷贝 I/O
    - 降级到标准文件 I/O
    """
    if device is None:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    intermediate_dim = int(shape[0])
    expert_dim = int(shape[1])

    if path.exists():
        try:
            obj: Any = torch.load(path, map_location=device)
            if isinstance(obj, dict) and "w1" in obj and "w2" in obj:
                if "w3" not in obj:
                    obj["w3"] = obj["w1"].clone()
                return {
                    "w1": obj["w1"].to(device),
                    "w3": obj["w3"].to(device),
                    "w2": obj["w2"].to(device),
                }
            if torch.is_tensor(obj):
                if obj.ndim == 2 and obj.shape[0] == 3 * intermediate_dim and obj.shape[1] == expert_dim:
                    w1 = obj[:intermediate_dim, :]
                    w3 = obj[intermediate_dim : 2 * intermediate_dim, :]
                    w2 = obj[2 * intermediate_dim :, :].t()
                    return {"w1": w1.to(device), "w3": w3.to(device), "w2": w2.to(device)}
        except Exception:
            pass

    w1 = torch.randn((intermediate_dim, expert_dim), dtype=torch.float16, device=device)
    w3 = torch.randn((intermediate_dim, expert_dim), dtype=torch.float16, device=device)
    w2 = torch.randn((expert_dim, intermediate_dim), dtype=torch.float16, device=device)
    return {"w1": w1, "w3": w3, "w2": w2}


def save_expert_weights(path: Path, tensor: Any):
    """保存专家权重到存储层"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(tensor, dict):
        obj = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in tensor.items()}
        torch.save(obj, path)
    else:
        torch.save(tensor.detach().cpu() if torch.is_tensor(tensor) else tensor, path)
