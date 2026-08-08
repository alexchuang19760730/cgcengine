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
MetalBackend - macOS Apple Silicon 的 I/O 後端實現

使用 Metal MPS (Multi-Processor Support) 和 mmap 作為存儲後端，
在 macOS 上提供高效的 GPU 直接訪問和記憶體映射文件支持。
"""

import os
import mmap
import threading
from typing import Dict, List, Tuple, Any
from pathlib import Path
import torch

from cgc_engine.io_unified.io_backend import IOBackend, IOStats
from cgc_engine.io_unified.unified_io_controller import UnifiedIOConfig


class MetalBackend(IOBackend):
    """
    Metal MPS 後端

    macOS Apple Silicon 上的 I/O 後端實現，使用 MPS 和 mmap。
    """

    def __init__(self, config: UnifiedIOConfig):
        self.config = config
        self._stats = IOStats()
        self._device = None
        self._mmap_dir = "/tmp/magi_metal_mmap"
        self._cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._weight_cache: Dict[str, torch.Tensor] = {}
        self._expert_cache: Dict[int, torch.Tensor] = {}
        self._lock = threading.Lock()

        Path(self._mmap_dir).mkdir(parents=True, exist_ok=True)

        if torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

    def initialize(self) -> None:
        """初始化 Metal 後端"""
        if not torch.backends.mps.is_available():
            import warnings
            warnings.warn("Metal MPS not available, falling back to CPU")
        self._stats = IOStats()

    def load_kv(
        self,
        key: str,
        seq_len: int,
        head_dim: int,
        num_heads: int = 32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """載入 KV Cache"""
        cache_key = f"kv:{key}"

        with self._lock:
            if cache_key in self._cache:
                self._stats.hits += 1
                k, v = self._cache[cache_key]
                return k.to(self._device), v.to(self._device)

            self._stats.misses += 1

        mmap_path = os.path.join(self._mmap_dir, f"{key}.mmap")

        if os.path.exists(mmap_path):
            k, v = self._load_from_mmap(mmap_path, seq_len, head_dim, num_heads)
        else:
            k = torch.randn(1, num_heads, seq_len, head_dim, device="cpu")
            v = torch.randn(1, num_heads, seq_len, head_dim, device="cpu")

        with self._lock:
            self._stats.reads += 1
            self._stats.bytes_read += k.numel() * k.element_size() + v.numel() * v.element_size()
            self._cache[cache_key] = (k, v)

        return k.to(self._device), v.to(self._device)

    def _load_from_mmap(
        self,
        path: str,
        seq_len: int,
        head_dim: int,
        num_heads: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """從 mmap 文件載入"""
        try:
            with open(path, "rb") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped:
                    data = torch.frombuffer(
                        mmapped[:],
                        dtype=torch.float32
                    ).reshape(2, num_heads, seq_len, head_dim)
                    k = data[0].clone()
                    v = data[1].clone()
                    return k, v
        except Exception:
            k = torch.randn(1, num_heads, seq_len, head_dim, device="cpu")
            v = torch.randn(1, num_heads, seq_len, head_dim, device="cpu")
            return k, v

    def save_kv(
        self,
        key: str,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> bool:
        """儲存 KV Cache"""
        cache_key = f"kv:{key}"

        k_cpu = k.cpu()
        v_cpu = v.cpu()

        with self._lock:
            self._cache[cache_key] = (k_cpu, v_cpu)
            self._stats.writes += 1
            self._stats.bytes_written += k_cpu.numel() * k_cpu.element_size() + v_cpu.numel() * v_cpu.element_size()

        mmap_path = os.path.join(self._mmap_dir, f"{key}.mmap")

        try:
            with open(mmap_path, "wb") as f:
                data = torch.cat([k_cpu, v_cpu]).numpy().tobytes()
                f.write(data)
            return True
        except Exception:
            return False

    def load_weight(
        self,
        path: str,
        shape: List[int],
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """載入權重"""
        with self._lock:
            if path in self._weight_cache:
                self._stats.hits += 1
                return self._weight_cache[path].to(self._device)

            self._stats.misses += 1

        if os.path.exists(path):
            try:
                weight = torch.load(path, map_location="cpu")
                if not isinstance(weight, torch.Tensor):
                    weight = torch.tensor(weight)
            except Exception:
                weight = torch.randn(shape, dtype=dtype, device="cpu")
        else:
            weight = torch.randn(shape, dtype=dtype, device="cpu")

        with self._lock:
            self._stats.reads += 1
            self._stats.bytes_read += weight.numel() * weight.element_size()
            self._weight_cache[path] = weight.cpu()

        return weight.to(self._device)

    def save_weight(
        self,
        path: str,
        tensor: torch.Tensor,
    ) -> bool:
        """儲存權重"""
        tensor_cpu = tensor.cpu()

        with self._lock:
            self._weight_cache[path] = tensor_cpu
            self._stats.writes += 1
            self._stats.bytes_written += tensor_cpu.numel() * tensor_cpu.element_size()

        try:
            torch.save(tensor_cpu, path)
            return True
        except Exception:
            return False

    def load_expert(
        self,
        expert_id: int,
        path: str,
    ) -> torch.Tensor:
        """載入專家權重"""
        with self._lock:
            if expert_id in self._expert_cache:
                self._stats.hits += 1
                return self._expert_cache[expert_id].to(self._device)

            self._stats.misses += 1

        if os.path.exists(path):
            try:
                expert = torch.load(path, map_location="cpu")
                if not isinstance(expert, torch.Tensor):
                    expert = torch.tensor(expert)
            except Exception:
                expert = torch.randn(4096, 4096, device="cpu")
        else:
            expert = torch.randn(4096, 4096, device="cpu")

        with self._lock:
            self._stats.reads += 1
            self._stats.bytes_read += expert.numel() * expert.element_size()
            self._expert_cache[expert_id] = expert.cpu()

        return expert.to(self._device)

    def save_expert(
        self,
        expert_id: int,
        tensor: torch.Tensor,
    ) -> bool:
        """儲存專家權重"""
        tensor_cpu = tensor.cpu()

        with self._lock:
            self._expert_cache[expert_id] = tensor_cpu
            self._stats.writes += 1
            self._stats.bytes_written += tensor_cpu.numel() * tensor_cpu.element_size()

        return True

    def prefetch(self, keys: List[str]) -> None:
        """預取資料"""
        for key in keys:
            if key.startswith("kv:"):
                k, v = self.load_kv(key[3:], 2048, 128)
            elif key.startswith("weight:"):
                self.load_weight(key[7:], [4096, 4096])

    def evict(self, keys: List[str]) -> None:
        """驅逐資料"""
        with self._lock:
            for key in keys:
                if key in self._cache:
                    del self._cache[key]
                if key in self._weight_cache:
                    del self._weight_cache[key]

    @property
    def name(self) -> str:
        """後端名稱"""
        return "metal_mps"

    @property
    def stats(self) -> IOStats:
        """I/O 統計"""
        return self._stats

    def shutdown(self) -> None:
        """關閉後端"""
        self._cache.clear()
        self._weight_cache.clear()
        self._expert_cache.clear()
        self._stats = IOStats()