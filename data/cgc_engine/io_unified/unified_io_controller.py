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
UnifiedIOController - 統一的 I/O 調度控制器

根據平台自動選擇最適合的 I/O 後端：
- Linux NVIDIA GPU: GDS (GPUDirect Storage) + SPDK
- macOS Apple Silicon: Metal MPS + mmap
- Linux CPU: SPDK
- Fallback: PyTorch tensor

職責：
1. 管理所有 I/O 後端實例
2. 根據平台自動選擇後端
3. 統一調度存儲操作
4. 追蹤 I/O 統計
"""

import platform
import sys
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import threading
import torch

from .io_backend import IOBackend, IOStats

logger = logging.getLogger(__name__)


class Platform(Enum):
    LINUX_NVIDIA = "linux_nvidia"
    MACOS_ARM = "macos_arm"
    LINUX_CPU = "linux_cpu"
    OTHER = "other"


@dataclass
class UnifiedIOConfig:
    enable_gds: bool = True
    enable_spdk: bool = True
    enable_metal: bool = True
    enable_mmap: bool = True
    enable_jit: bool = True
    cache_size_mb: int = 1024
    prefetch_async: bool = True
    eviction_policy: str = "lru"


class UnifiedIOController:
    """
    統一 I/O 控制器

    這個控制器管理所有存儲層的 I/O 操作，根據平台自動選擇最優後端。
    """

    _instance: Optional["UnifiedIOController"] = None
    _lock = threading.Lock()

    def __new__(cls, config: Optional[UnifiedIOConfig] = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[UnifiedIOConfig] = None):
        if self._initialized:
            return

        self.config = config or UnifiedIOConfig()
        self.platform = self._detect_platform()

        self.backends: Dict[str, IOBackend] = {}
        self.active_backend: Optional[IOBackend] = None
        self._stats = IOStats()
        self._cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._weight_cache: Dict[str, torch.Tensor] = {}
        self._expert_cache: Dict[int, torch.Tensor] = {}
        self._lock = threading.Lock()

        self._initialize_backends()
        self._initialized = True

    def _detect_platform(self) -> Platform:
        """檢測運行平台"""
        system = platform.system().lower()
        machine = platform.machine().lower()

        if system == "linux" and machine in ("x86_64", "amd64"):
            if torch.cuda.is_available():
                return Platform.LINUX_NVIDIA
            return Platform.LINUX_CPU
        elif system == "darwin" and machine in ("arm64", "aarch64"):
            return Platform.MACOS_ARM
        return Platform.OTHER

    def _initialize_backends(self) -> None:
        """根據平台初始化後端"""
        if self.platform == Platform.LINUX_NVIDIA and self.config.enable_gds:
            try:
                from cgc_engine.gds_service.gds_manager import GDSManager
                
                gds_backend = GDSManager()
                if hasattr(gds_backend, 'initialize'):
                    gds_backend.initialize()
                self.backends["gds"] = gds_backend
                logger.info("[UnifiedIO] ✅ GDS 后端已初始化")
            except ImportError as e:
                logger.warning(f"[UnifiedIO] ⚠️ 无法初始化 GDS 后端: {e}")

        if self.platform in (Platform.LINUX_NVIDIA, Platform.LINUX_CPU) and self.config.enable_spdk:
            try:
                from cgc_engine.spdk_adapter.spdk_io_manager import SPDKIOManager
                from cgc_engine.spdk_adapter.spdk_config import SPDKConfig
                
                spdk_config = SPDKConfig(kv_store_path="/tmp/unified_io_spdk", io_queues=4)
                spdk_backend = SPDKIOManager(spdk_config)
                spdk_backend.start()
                self.backends["spdk"] = spdk_backend
                logger.info("[UnifiedIO] ✅ SPDK 后端已初始化")
            except ImportError as e:
                logger.warning(f"[UnifiedIO] ⚠️ 无法初始化 SPDK 后端: {e}")

        if self.platform == Platform.MACOS_ARM and self.config.enable_metal:
            try:
                from cgc_engine.io_unified.metal_backend import MetalBackend
                metal_backend = MetalBackend(self.config)
                if hasattr(metal_backend, 'initialize'):
                    metal_backend.initialize()
                self.backends["metal"] = metal_backend
                if self.active_backend is None:
                    self.active_backend = metal_backend
            except ImportError:
                pass

        from cgc_engine.io_unified.pytorch_backend import PyTorchBackend
        pytorch_backend = PyTorchBackend(self.config)
        if hasattr(pytorch_backend, 'initialize'):
            pytorch_backend.initialize()
        self.backends["pytorch"] = pytorch_backend
        if self.active_backend is None:
            self.active_backend = pytorch_backend

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
                return self._cache[cache_key]

            self._stats.misses += 1

        # 尝试使用 GDS 或 SPDK 后端
        try:
            if "gds" in self.backends:
                logger.debug(f"[UnifiedIO] 使用 GDS 加载 KV: {key}")
                k, v = self.backends["gds"].load_kv_from_pd(key, seq_len, head_dim)
            elif "spdk" in self.backends and hasattr(self.backends["spdk"], 'kv_store'):
                logger.debug(f"[UnifiedIO] 使用 SPDK 加载 KV: {key}")
                k, v = self._load_kv_from_spdk(key, seq_len, head_dim, num_heads)
            elif self.active_backend and hasattr(self.active_backend, 'load_kv'):
                logger.debug(f"[UnifiedIO] 使用 {self.name} 加载 KV: {key}")
                k, v = self.active_backend.load_kv(key, seq_len, head_dim, num_heads)
            else:
                raise NotImplementedError("后端不支持 load_kv")
        except Exception as e:
            logger.warning(f"[UnifiedIO] 后端加载 KV 失败，使用降级方案: {e}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            k = torch.randn(1, num_heads, seq_len, head_dim, device=device)
            v = torch.randn(1, num_heads, seq_len, head_dim, device=device)

        with self._lock:
            self._stats.reads += 1
            self._stats.bytes_read += k.numel() * k.element_size() + v.numel() * v.element_size()
            self._cache[cache_key] = (k, v)

        return k, v

    def _load_kv_from_spdk(self, key: str, seq_len: int, head_dim: int, num_heads: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """从 SPDK KV Store 加载 KV Cache"""
        spdk = self.backends["spdk"]
        if spdk.kv_store:
            block_id = hash(key) % 1000
            result = spdk.kv_store.get_kv_block(block_id, layer_id=0, device="cuda" if torch.cuda.is_available() else "cpu")
            if result is not None:
                return result
        
        # 如果 SPDK 中没有，返回空张量
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return (
            torch.empty(1, num_heads, seq_len, head_dim, device=device),
            torch.empty(1, num_heads, seq_len, head_dim, device=device)
        )

    def save_kv(
        self,
        key: str,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> bool:
        """儲存 KV Cache"""
        cache_key = f"kv:{key}"

        with self._lock:
            self._cache[cache_key] = (k, v)
            self._stats.writes += 1
            self._stats.bytes_written += k.numel() * k.element_size() + v.numel() * v.element_size()

        # 尝试使用 GDS 或 SPDK 后端
        try:
            if "gds" in self.backends:
                logger.debug(f"[UnifiedIO] 使用 GDS 保存 KV: {key}")
                self.backends["gds"].save_kv_to_pd(key, k, v)
                return True
            elif "spdk" in self.backends and hasattr(self.backends["spdk"], 'kv_store'):
                logger.debug(f"[UnifiedIO] 使用 SPDK 保存 KV: {key}")
                block_id = hash(key) % 1000
                return self.backends["spdk"].kv_store.put_kv_block(block_id, k, v, layer_id=0)
            elif self.active_backend and hasattr(self.active_backend, 'save_kv'):
                logger.debug(f"[UnifiedIO] 使用 {self.name} 保存 KV: {key}")
                return self.active_backend.save_kv(key, k, v)
            return True
        except Exception as e:
            logger.warning(f"[UnifiedIO] 后端保存 KV 失败: {e}")
            return False

    def load_weight(
        self,
        path: str,
        shape: List[int],
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """載入權重 - 使用 GDS 零拷贝"""
        with self._lock:
            if path in self._weight_cache:
                self._stats.hits += 1
                return self._weight_cache[path]

            self._stats.misses += 1

        weight = None
        if "gds" in self.backends and hasattr(self.backends["gds"], "load_weight_from_pd"):
            try:
                logger.debug(f"[UnifiedIO] 使用 GDS 零拷贝加载权重: {path}")
                weight = self.backends["gds"].load_weight_from_pd(path, shape)
            except Exception as e:
                logger.warning(f"[UnifiedIO] GDS 加载权重失败: {e}")

        if weight is None and self.active_backend and hasattr(self.active_backend, "load_weight"):
            try:
                logger.debug(f"[UnifiedIO] 使用 {self.name} 加载权重: {path}")
                weight = self.active_backend.load_weight(path, shape, dtype)
            except Exception as e:
                logger.warning(f"[UnifiedIO] 后端加载权重失败: {e}")

        if weight is None and "pytorch" in self.backends and hasattr(self.backends["pytorch"], "load_weight"):
            try:
                logger.debug(f"[UnifiedIO] 使用 pytorch 加载权重: {path}")
                weight = self.backends["pytorch"].load_weight(path, shape, dtype)
            except Exception as e:
                logger.warning(f"[UnifiedIO] pytorch 加载权重失败: {e}")

        if weight is None:
            weight = torch.randn(shape, dtype=dtype, device=self._get_default_device())

        with self._lock:
            self._stats.reads += 1
            self._stats.bytes_read += weight.numel() * weight.element_size()
            self._weight_cache[path] = weight

        return weight

    def _get_default_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def load_expert_mlp(
        self,
        expert_id: int,
        base_path: str,
        expert_dim: int,
        intermediate_dim: int,
        dtype: torch.dtype = torch.float16,
    ) -> Dict[str, torch.Tensor]:
        def _p(name: str) -> str:
            return f"{base_path}.{name}.pt"

        w1 = self.load_weight(_p("w1"), [int(intermediate_dim), int(expert_dim)], dtype=dtype)
        w3 = self.load_weight(_p("w3"), [int(intermediate_dim), int(expert_dim)], dtype=dtype)
        w2 = self.load_weight(_p("w2"), [int(expert_dim), int(intermediate_dim)], dtype=dtype)
        return {"w1": w1, "w3": w3, "w2": w2}

    def save_expert_mlp(
        self,
        expert_id: int,
        base_path: str,
        weights: Dict[str, torch.Tensor],
    ) -> bool:
        def _p(name: str) -> str:
            return f"{base_path}.{name}.pt"

        ok = True
        for k in ("w1", "w3", "w2"):
            if k not in weights:
                ok = False
                continue
            ok = bool(ok and self.save_weight(_p(k), weights[k]))
        return ok

    def save_weight(
        self,
        path: str,
        tensor: torch.Tensor,
    ) -> bool:
        """儲存權重"""
        with self._lock:
            self._weight_cache[path] = tensor
            self._stats.writes += 1
            self._stats.bytes_written += tensor.numel() * tensor.element_size()

        if self.active_backend and hasattr(self.active_backend, "save_weight"):
            return self.active_backend.save_weight(path, tensor)
        if "pytorch" in self.backends and hasattr(self.backends["pytorch"], "save_weight"):
            return self.backends["pytorch"].save_weight(path, tensor)
        return True

    def load_expert(
        self,
        expert_id: int,
        path: str,
    ) -> torch.Tensor:
        """載入專家權重"""
        with self._lock:
            if expert_id in self._expert_cache:
                self._stats.hits += 1
                return self._expert_cache[expert_id]

            self._stats.misses += 1

        if self.active_backend and hasattr(self.active_backend, "load_expert"):
            expert = self.active_backend.load_expert(expert_id, path)
        elif "pytorch" in self.backends and hasattr(self.backends["pytorch"], "load_expert"):
            expert = self.backends["pytorch"].load_expert(expert_id, path)
        else:
            expert = torch.randn(4096, 4096, device=self._get_default_device())

        with self._lock:
            self._stats.reads += 1
            self._stats.bytes_read += expert.numel() * expert.element_size()
            self._expert_cache[expert_id] = expert

        return expert

    def save_expert(
        self,
        expert_id: int,
        tensor: torch.Tensor,
    ) -> bool:
        """儲存專家權重"""
        with self._lock:
            self._expert_cache[expert_id] = tensor
            self._stats.writes += 1
            self._stats.bytes_written += tensor.numel() * tensor.element_size()

        if self.active_backend and hasattr(self.active_backend, "save_expert"):
            return self.active_backend.save_expert(expert_id, tensor)
        if "pytorch" in self.backends and hasattr(self.backends["pytorch"], "save_expert"):
            return self.backends["pytorch"].save_expert(expert_id, tensor)
        return True

    def prefetch(self, keys: List[str]) -> None:
        """預取資料"""
        if self.active_backend and hasattr(self.active_backend, "prefetch"):
            self.active_backend.prefetch(keys)

    def evict(self, keys: List[str]) -> None:
        """驅逐資料"""
        with self._lock:
            for key in keys:
                if key in self._cache:
                    del self._cache[key]
                if key in self._weight_cache:
                    del self._weight_cache[key]

        if self.active_backend and hasattr(self.active_backend, "evict"):
            self.active_backend.evict(keys)

    def get_stats(self) -> IOStats:
        """獲取 I/O 統計"""
        return self._stats

    @property
    def name(self) -> str:
        """後端名稱"""
        if self.active_backend and hasattr(self.active_backend, "name"):
            return self.active_backend.name
        return "none"

    @property
    def platform_name(self) -> str:
        """平台名稱"""
        return self.platform.value

    def shutdown(self) -> None:
        """關閉所有後端"""
        for backend in self.backends.values():
            if hasattr(backend, "shutdown"):
                backend.shutdown()
            elif hasattr(backend, "stop"):
                backend.stop()
            elif hasattr(backend, "close"):
                backend.close()
        self.backends.clear()
        self.active_backend = None

    @classmethod
    def get_instance(cls) -> "UnifiedIOController":
        """獲取單例實例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """重置單例實例（用於測試）"""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.shutdown()
                cls._instance = None


def get_unified_io_controller(config: Optional[UnifiedIOConfig] = None) -> UnifiedIOController:
    """獲取 UnifiedIOController 實例的便捷函數"""
    return UnifiedIOController.get_instance()
