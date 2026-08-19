
# Copyright (c) 2026 SandAI. All Rights Reserved.
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
全球唯一：双分层统一管理器
KV Cache + MoE专家权重 → RAM / SSD / PD 三级存储

核心思想：
1. Flash-MoE: "Trust the OS" - 利用 OS 页缓存，不需要复杂自定义缓存
2. oMLX: 块级 KV 缓存，前缀共享，Copy-on-Write
3. 双分层统一管理：KV Cache 和 专家权重 同时支持三级存储
4. LRU 淘汰，自动写回 SSD
5. 支持 PD 分布式同步
"""

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import threading

import torch
from dataclasses import dataclass
from enum import Enum, auto

# 尝试导入 PD 客户端
try:
    from .pd import PDClient, PDClientConfig
    PD_AVAILABLE = True
except ImportError:
    PD_AVAILABLE = False

# 尝试导入 MLX
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

logger = __import__('logging').getLogger(__name__)


class StorageTier(Enum):
    """存储层级"""
    RAM = auto()
    SSD = auto()
    PD = auto()


@dataclass
class CacheEntry:
    """缓存条目"""
    key: Any
    value: Any
    last_access: float
    tier: StorageTier
    size: int


@dataclass
class DualLayerConfig:
    """双分层配置"""
    # KV Cache 配置
    max_ram_kv_blocks: int = 32
    max_ram_kv_size_mb: int = 512

    # 专家权重配置
    max_ram_experts: int = 12
    max_ram_experts_size_mb: int = 4096

    # SSD 配置
    ssd_root: str = "./dual_storage"
    ssd_kv_dir: str = "kv_blocks"
    ssd_expert_dir: str = "experts"
    
    # SPDK 配置
    enable_spdk: bool = False
    
    # GDS 配置
    enable_gds: bool = False

    # PD 配置
    pd_endpoint: Optional[str] = None
    enable_pd_sync: bool = True

    # LRU 配置
    trust_os_page_cache: bool = True  # Flash-MoE 的核心思想
    lru_enabled: bool = True


class DualLayerManager:
    """全球唯一：双分层统一管理器"""

    def __init__(self, config: Optional[DualLayerConfig] = None):
        self.config = config or DualLayerConfig()

        # 创建目录
        self.ssd_root = Path(self.config.ssd_root)
        self.kv_dir = self.ssd_root / self.config.ssd_kv_dir
        self.expert_dir = self.ssd_root / self.config.ssd_expert_dir
        self.kv_dir.mkdir(parents=True, exist_ok=True)
        self.expert_dir.mkdir(parents=True, exist_ok=True)

        # RAM 缓存
        self.kv_cache: Dict[int, CacheEntry] = {}
        self.kv_lru: List[int] = []
        self.kv_lock = threading.RLock()

        self.expert_cache: Dict[int, CacheEntry] = {}
        self.expert_lru: List[int] = []
        self.expert_lock = threading.RLock()

        # 统计信息
        self.stats = {
            "kv_hits": 0,
            "kv_misses": 0,
            "kv_evictions": 0,
            "kv_ram_reads": 0,
            "kv_ssd_reads": 0,
            "kv_spdk_reads": 0,
            "kv_gds_reads": 0,
            "kv_pd_reads": 0,
            "expert_hits": 0,
            "expert_misses": 0,
            "expert_evictions": 0,
            "expert_ram_reads": 0,
            "expert_ssd_reads": 0,
            "expert_spdk_reads": 0,
            "expert_gds_reads": 0,
            "expert_pd_reads": 0,
        }

        # PD 客户端
        self.pd_client = None
        if PD_AVAILABLE and self.config.pd_endpoint and self.config.enable_pd_sync:
            try:
                pd_config = PDClientConfig(address=self.config.pd_endpoint)
                self.pd_client = PDClient(address=self.config.pd_endpoint, config=pd_config)
                logger.info(f"✅ DualLayerManager 连接 PD: {self.config.pd_endpoint}")
            except Exception as e:
                logger.warning(f"⚠️  PD 连接失败，使用本地模式: {e}")
        
        # SPDK 组件
        self.spdk_kv_store = None
        self.spdk_expert_store = None
        if self.config.enable_spdk:
            try:
                from .spdk_adapter import SPDKConfig, SPDKKVStore, SPDKExpertStore, SPDKBlockDevice
                spdk_config = SPDKConfig(enable_spdk=True)
                block_device = SPDKBlockDevice(spdk_config)
                block_device.initialize()
                self.spdk_kv_store = SPDKKVStore(spdk_config, block_device)
                self.spdk_kv_store.initialize()
                self.spdk_expert_store = SPDKExpertStore(spdk_config, block_device)
                self.spdk_expert_store.initialize()
                logger.info(f"✅ DualLayerManager 启用 SPDK")
            except Exception as e:
                logger.warning(f"⚠️  SPDK 初始化失败: {e}")
        
        # GDS 组件
        self.gds_integration = None
        if self.config.enable_gds:
            try:
                from .gds_service import GDSConfig, GDSIntegration
                gds_config = GDSConfig(enable_gds=True)
                self.gds_integration = GDSIntegration(gds_config)
                self.gds_integration.initialize()
                logger.info(f"✅ DualLayerManager 启用 GDS")
            except Exception as e:
                logger.warning(f"⚠️  GDS 初始化失败: {e}")

        # 全局单例标记
        self._is_global = False
        logger.info(f"🚀 DualLayerManager 初始化完成")
        logger.info(f"  - KV RAM: {self.config.max_ram_kv_blocks} blocks / {self.config.max_ram_kv_size_mb} MB")
        logger.info(f"  - Expert RAM: {self.config.max_ram_experts} experts / {self.config.max_ram_experts_size_mb} MB")
        logger.info(f"  - SSD: {self.ssd_root}")
        logger.info(f"  - SPDK: {self.config.enable_spdk}")
        logger.info(f"  - GDS: {self.config.enable_gds}")
        logger.info(f"  - Trust OS Page Cache: {self.config.trust_os_page_cache}")

    # ==============================================
    # KV Cache 管理（三级存储 + SPDK + GDS）
    # ==============================================
    def get_kv_block(self, block_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取 KV 块（自动三级调度）"""
        with self.kv_lock:
            # 1. RAM 检查
            if block_id in self.kv_cache:
                self._touch_kv(block_id)
                self.stats["kv_hits"] += 1
                self.stats["kv_ram_reads"] += 1
                return self.kv_cache[block_id].value

            # RAM miss
            self.stats["kv_misses"] += 1

            # 2. SPDK 检查
            if self.spdk_kv_store and self.spdk_kv_store.kv_block_exists(block_id):
                try:
                    kv_data = self.spdk_kv_store.get_kv_block(block_id)
                    if kv_data:
                        self._insert_kv(block_id, kv_data)
                        self.stats["kv_spdk_reads"] += 1
                        return kv_data
                except Exception as e:
                    logger.warning(f"⚠️  加载 KV {block_id} 从 SPDK 失败: {e}")

            # 3. GDS 检查
            if self.gds_integration:
                try:
                    kv_tensor = self.gds_integration.load_kv_to_gpu(block_id)
                    if kv_tensor is not None:
                        # 这里简化处理，实际应该是 k 和 v 两个 tensor
                        kv_data = (kv_tensor, kv_tensor)
                        self._insert_kv(block_id, kv_data)
                        self.stats["kv_gds_reads"] += 1
                        return kv_data
                except Exception as e:
                    logger.warning(f"⚠️  加载 KV {block_id} 从 GDS 失败: {e}")

            # 4. SSD 检查
            kv_path = self.kv_dir / f"block_{block_id}.safetensors"
            if kv_path.exists():
                try:
                    kv_data = self._load_kv_from_ssd(kv_path)
                    self._insert_kv(block_id, kv_data)
                    self.stats["kv_ssd_reads"] += 1
                    return kv_data
                except Exception as e:
                    logger.warning(f"⚠️  加载 KV {block_id} 从 SSD 失败: {e}")

            # 5. PD 远程拉取
            if self.pd_client:
                try:
                    pd_key = f"kv_block_{block_id}"
                    kv_bytes, hit = self.pd_client.get_prefix(pd_key)
                    if hit and kv_bytes:
                        kv_data = pickle.loads(kv_bytes)
                        self._insert_kv(block_id, kv_data)
                        self._save_kv_to_ssd(block_id, kv_data)
                        # 同时保存到 SPDK
                        if self.spdk_kv_store:
                            self.spdk_kv_store.put_kv_block(block_id, kv_data[0], kv_data[1])
                        self.stats["kv_pd_reads"] += 1
                        return kv_data
                except Exception as e:
                    logger.warning(f"⚠️  加载 KV {block_id} 从 PD 失败: {e}")

            raise ValueError(f"KV Block {block_id} 在所有层级都不存在")

    def put_kv_block(self, block_id: int, k: torch.Tensor, v: torch.Tensor):
        """存储 KV 块（自动三级调度）"""
        with self.kv_lock:
            kv_data = (k, v)
            size = self._calculate_size(k) + self._calculate_size(v)

            # 插入到 RAM 缓存
            self._insert_kv(block_id, kv_data, size)

            # 异步写入 SPDK
            if self.spdk_kv_store:
                try:
                    self.spdk_kv_store.put_kv_block(block_id, k, v)
                except Exception as e:
                    logger.warning(f"⚠️  保存 KV {block_id} 到 SPDK 失败: {e}")
            
            # 异步写入 GDS
            if self.gds_integration and k.is_cuda:
                try:
                    self.gds_integration.save_kv_from_gpu(block_id, k)
                except Exception as e:
                    logger.warning(f"⚠️  保存 KV {block_id} 到 GDS 失败: {e}")

            # 异步写入 SSD
            try:
                self._save_kv_to_ssd(block_id, kv_data)
            except Exception as e:
                logger.warning(f"⚠️  保存 KV {block_id} 到 SSD 失败: {e}")

            # 同步到 PD
            if self.pd_client and self.config.enable_pd_sync:
                try:
                    pd_key = f"kv_block_{block_id}"
                    kv_bytes = pickle.dumps(kv_data)
                    self.pd_client.store_prefix(pd_key, kv_bytes, ttl_seconds=86400)
                except Exception as e:
                    logger.warning(f"⚠️  同步 KV {block_id} 到 PD 失败: {e}")

    def release_kv_block(self, block_id: int):
        """释放 KV 块（仅从 RAM 移除，保留 SSD/PD）"""
        with self.kv_lock:
            if block_id in self.kv_cache:
                del self.kv_cache[block_id]
                if block_id in self.kv_lru:
                    self.kv_lru.remove(block_id)

    def _insert_kv(self, block_id: int, kv_data: Tuple[torch.Tensor, torch.Tensor], size: Optional[int] = None):
        """插入 KV 到 RAM 缓存"""
        if size is None:
            size = self._calculate_size(kv_data[0]) + self._calculate_size(kv_data[1])

        # 检查是否需要淘汰
        while len(self.kv_cache) >= self.config.max_ram_kv_blocks and self.config.lru_enabled:
            if not self.kv_lru:
                break
            evict_id = self.kv_lru.pop(0)
            del self.kv_cache[evict_id]
            self.stats["kv_evictions"] += 1

        entry = CacheEntry(
            key=block_id,
            value=kv_data,
            last_access=self._get_current_time(),
            tier=StorageTier.RAM,
            size=size
        )
        self.kv_cache[block_id] = entry
        self.kv_lru.append(block_id)

    def _touch_kv(self, block_id: int):
        """更新 LRU"""
        if block_id in self.kv_lru:
            self.kv_lru.remove(block_id)
        self.kv_lru.append(block_id)
        self.kv_cache[block_id].last_access = self._get_current_time()

    def _save_kv_to_ssd(self, block_id: int, kv_data: Tuple[torch.Tensor, torch.Tensor]):
        """保存 KV 到 SSD"""
        kv_path = self.kv_dir / f"block_{block_id}.safetensors"
        if MLX_AVAILABLE:
            # 使用 MLX 保存
            mx_k = mx.array(kv_data[0].cpu().numpy()) if isinstance(kv_data[0], torch.Tensor) else kv_data[0]
            mx_v = mx.array(kv_data[1].cpu().numpy()) if isinstance(kv_data[1], torch.Tensor) else kv_data[1]
            mx.save_safetensors(str(kv_path), {"k": mx_k, "v": mx_v})
        else:
            # 使用 pickle 保存
            with open(kv_path, "wb") as f:
                pickle.dump({"k": kv_data[0], "v": kv_data[1]}, f)

    def _load_kv_from_ssd(self, kv_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        """从 SSD 加载 KV"""
        if MLX_AVAILABLE:
            # 使用 MLX 加载
            data = mx.load(str(kv_path))
            k = torch.tensor(data["k"]) if torch.is_tensor(data["k"]) else data["k"]
            v = torch.tensor(data["v"]) if torch.is_tensor(data["v"]) else data["v"]
        else:
            # 使用 pickle 加载
            with open(kv_path, "rb") as f:
                data = pickle.load(f)
            k = data["k"]
            v = data["v"]
        return (k, v)

    # ==============================================
    # MoE 专家权重管理（三级存储）
    # ==============================================
    def get_expert(self, expert_id: int) -> Dict[str, torch.Tensor]:
        """获取专家权重（自动三级调度）"""
        with self.expert_lock:
            # 1. RAM 检查
            if expert_id in self.expert_cache:
                self._touch_expert(expert_id)
                self.stats["expert_hits"] += 1
                self.stats["expert_ram_reads"] += 1
                return self.expert_cache[expert_id].value

            # RAM miss
            self.stats["expert_misses"] += 1

            # 2. SSD 检查
            expert_path = self.expert_dir / f"expert_{expert_id}.safetensors"
            if expert_path.exists():
                try:
                    expert_data = self._load_expert_from_ssd(expert_path)
                    self._insert_expert(expert_id, expert_data)
                    self.stats["expert_ssd_reads"] += 1
                    return expert_data
                except Exception as e:
                    logger.warning(f"⚠️  加载 Expert {expert_id} 从 SSD 失败: {e}")

            # 3. PD 远程拉取
            if self.pd_client:
                try:
                    pd_key = f"expert_{expert_id}"
                    expert_bytes, hit = self.pd_client.get_prefix(pd_key)
                    if hit and expert_bytes:
                        expert_data = pickle.loads(expert_bytes)
                        self._insert_expert(expert_id, expert_data)
                        self._save_expert_to_ssd(expert_id, expert_data)
                        self.stats["expert_pd_reads"] += 1
                        return expert_data
                except Exception as e:
                    logger.warning(f"⚠️  加载 Expert {expert_id} 从 PD 失败: {e}")

            raise ValueError(f"Expert {expert_id} 在所有层级都不存在")

    def put_expert(self, expert_id: int, expert_weights: Dict[str, torch.Tensor]):
        """存储专家权重（自动三级调度）"""
        with self.expert_lock:
            size = sum(self._calculate_size(v) for v in expert_weights.values())

            # 插入到 RAM 缓存
            self._insert_expert(expert_id, expert_weights, size)

            # 异步写入 SSD
            try:
                self._save_expert_to_ssd(expert_id, expert_weights)
            except Exception as e:
                logger.warning(f"⚠️  保存 Expert {expert_id} 到 SSD 失败: {e}")

            # 同步到 PD
            if self.pd_client and self.config.enable_pd_sync:
                try:
                    pd_key = f"expert_{expert_id}"
                    expert_bytes = pickle.dumps(expert_weights)
                    self.pd_client.store_prefix(pd_key, expert_bytes, ttl_seconds=86400)
                except Exception as e:
                    logger.warning(f"⚠️  同步 Expert {expert_id} 到 PD 失败: {e}")

    def release_expert(self, expert_id: int):
        """释放专家权重（仅从 RAM 移除，保留 SSD/PD）"""
        with self.expert_lock:
            if expert_id in self.expert_cache:
                del self.expert_cache[expert_id]
                if expert_id in self.expert_lru:
                    self.expert_lru.remove(expert_id)

    def _insert_expert(self, expert_id: int, expert_weights: Dict[str, torch.Tensor], size: Optional[int] = None):
        """插入专家到 RAM 缓存"""
        if size is None:
            size = sum(self._calculate_size(v) for v in expert_weights.values())

        # 检查是否需要淘汰
        while len(self.expert_cache) >= self.config.max_ram_experts and self.config.lru_enabled:
            if not self.expert_lru:
                break
            evict_id = self.expert_lru.pop(0)
            del self.expert_cache[evict_id]
            self.stats["expert_evictions"] += 1

        entry = CacheEntry(
            key=expert_id,
            value=expert_weights,
            last_access=self._get_current_time(),
            tier=StorageTier.RAM,
            size=size
        )
        self.expert_cache[expert_id] = entry
        self.expert_lru.append(expert_id)

    def _touch_expert(self, expert_id: int):
        """更新 LRU"""
        if expert_id in self.expert_lru:
            self.expert_lru.remove(expert_id)
        self.expert_lru.append(expert_id)
        self.expert_cache[expert_id].last_access = self._get_current_time()

    def _save_expert_to_ssd(self, expert_id: int, expert_weights: Dict[str, torch.Tensor]):
        """保存专家到 SSD"""
        expert_path = self.expert_dir / f"expert_{expert_id}.safetensors"
        if MLX_AVAILABLE:
            mx_weights = {}
            for name, weight in expert_weights.items():
                if isinstance(weight, torch.Tensor):
                    mx_weights[name] = mx.array(weight.cpu().numpy())
                else:
                    mx_weights[name] = weight
            mx.save_safetensors(str(expert_path), mx_weights)
        else:
            with open(expert_path, "wb") as f:
                pickle.dump(expert_weights, f)

    def _load_expert_from_ssd(self, expert_path: Path) -> Dict[str, torch.Tensor]:
        """从 SSD 加载专家"""
        if MLX_AVAILABLE:
            data = mx.load(str(expert_path))
            weights = {}
            for name, weight in data.items():
                weights[name] = torch.tensor(weight) if torch.is_tensor(weight) else weight
            return weights
        else:
            with open(expert_path, "rb") as f:
                return pickle.load(f)

    # ==============================================
    # 工具函数
    # ==============================================
    def _calculate_size(self, tensor: torch.Tensor) -> int:
        """计算 Tensor 大小（字节）"""
        if isinstance(tensor, torch.Tensor):
            return tensor.numel() * tensor.element_size()
        elif hasattr(tensor, "nbytes"):
            return tensor.nbytes
        else:
            return 1024  # 默认值

    def _get_current_time(self) -> float:
        """获取当前时间"""
        import time
        return time.time()

    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        with self.kv_lock, self.expert_lock:
            stats = self.stats.copy()
            stats["kv_ram_blocks"] = len(self.kv_cache)
            stats["expert_ram_count"] = len(self.expert_cache)
            return stats

    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()
        print("\n" + "=" * 80)
        print("📊 DualLayerManager 统计")
        print("=" * 80)
        print(f"KV Cache:")
        print(f"  - RAM Hits: {stats['kv_hits']}")
        print(f"  - RAM Misses: {stats['kv_misses']}")
        print(f"  - RAM Evictions: {stats['kv_evictions']}")
        print(f"  - RAM Blocks: {stats['kv_ram_blocks']}")
        print(f"  - RAM Reads: {stats['kv_ram_reads']}")
        print(f"  - SSD Reads: {stats['kv_ssd_reads']}")
        print(f"  - PD Reads: {stats['kv_pd_reads']}")
        print()
        print(f"Experts:")
        print(f"  - RAM Hits: {stats['expert_hits']}")
        print(f"  - RAM Misses: {stats['expert_misses']}")
        print(f"  - RAM Evictions: {stats['expert_evictions']}")
        print(f"  - RAM Count: {stats['expert_ram_count']}")
        print(f"  - RAM Reads: {stats['expert_ram_reads']}")
        print(f"  - SSD Reads: {stats['expert_ssd_reads']}")
        print(f"  - PD Reads: {stats['expert_pd_reads']}")
        print("=" * 80 + "\n")

    def clear_ram_cache(self):
        """清空 RAM 缓存"""
        with self.kv_lock, self.expert_lock:
            self.kv_cache.clear()
            self.kv_lru.clear()
            self.expert_cache.clear()
            self.expert_lru.clear()
            logger.info("✅ 已清空 RAM 缓存")

    def prefetch_experts(self, expert_ids: List[int], blocking: bool = False):
        """预加载专家（异步或同步）"""
        def _prefetch():
            for expert_id in expert_ids:
                try:
                    self.get_expert(expert_id)
                except Exception:
                    pass

        if blocking:
            _prefetch()
        else:
            threading.Thread(target=_prefetch, daemon=True).start()

    def prefetch_kv_blocks(self, block_ids: List[int], blocking: bool = False):
        """预加载 KV 块（异步或同步）"""
        def _prefetch():
            for block_id in block_ids:
                try:
                    self.get_kv_block(block_id)
                except Exception:
                    pass

        if blocking:
            _prefetch()
        else:
            threading.Thread(target=_prefetch, daemon=True).start()


# 全局单例
_global_instance: Optional[DualLayerManager] = None
_global_lock = threading.Lock()


def get_dual_layer_manager(config: Optional[DualLayerConfig] = None) -> DualLayerManager:
    """获取全局双分层管理器单例"""
    global _global_instance
    if _global_instance is None:
        with _global_lock:
            if _global_instance is None:
                _global_instance = DualLayerManager(config)
                _global_instance._is_global = True
    return _global_instance


def init_dual_layer_manager(config: Optional[DualLayerConfig] = None) -> DualLayerManager:
    """初始化全局双分层管理器（可选）"""
    global _global_instance
    with _global_lock:
        _global_instance = DualLayerManager(config)
        _global_instance._is_global = True
        return _global_instance

