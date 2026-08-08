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
SPDK KV Cache for PD Service

功能：
- 使用 SPDK 异步 IO 存储和读取 KV Cache
- 支持批量操作和预取优化
- 高吞吐量、低延迟的 KV 存储
"""

import torch
import pickle
import logging
from typing import Dict, Optional, Tuple, Any
import threading
import time

logger = logging.getLogger(__name__)

# 尝试导入 SPDK
try:
    from spdk_adapter.spdk_io_manager import SPDKIOManager, SPDK_AVAILABLE
    from spdk_adapter.spdk_kv_store import SPDKKVStore
    from spdk_adapter.spdk_config import SPDKConfig
except ImportError:
    SPDK_AVAILABLE = False
    logger.warning("[SPDK KV Cache] SPDK 不可用，将使用内存缓存")


class SPDKKVCache:
    """
    SPDK KV Cache - 使用 SPDK 异步 IO 存储 KV Cache
    
    特性：
    - 异步读写操作
    - 支持预取优化
    - 自动降级到内存缓存
    """
    
    def __init__(self, kv_store_path: str = "/data/pd_kv_cache", io_queues: int = 8):
        """
        初始化 SPDK KV Cache
        
        Args:
            kv_store_path: KV 存储路径
            io_queues: IO 队列数量
        """
        self._spdk_io_manager = None
        self._spdk_kv_store = None
        self._memory_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._lock = threading.Lock()
        
        if SPDK_AVAILABLE:
            try:
                config = SPDKConfig(kv_store_path=kv_store_path, io_queues=io_queues)
                self._spdk_kv_store = SPDKKVStore(config)
                self._spdk_kv_store.initialize()
                
                self._spdk_io_manager = SPDKIOManager(config)
                self._spdk_io_manager.start()
                
                logger.info("[SPDK KV Cache] ✅ SPDK KV Cache 初始化成功")
            except Exception as e:
                logger.warning(f"[SPDK KV Cache] ⚠️ SPDK 初始化失败: {e}")
        
        # 统计信息
        self._stats = {
            "reads": 0,
            "writes": 0,
            "hits": 0,
            "misses": 0,
            "prefetch_hits": 0,
            "spdk_operations": 0,
            "memory_operations": 0
        }
        
        # 预取队列
        self._prefetch_keys: set = set()
    
    @property
    def spdk_enabled(self) -> bool:
        """检查 SPDK 是否可用"""
        return self._spdk_kv_store is not None
    
    def get_kv(self, key: str, device: str = "cpu") -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        获取 KV Cache
        
        Args:
            key: KV 键
            device: 目标设备
        
        Returns:
            (k, v) 张量对，如果不存在返回 None
        """
        self._stats["reads"] += 1
        
        # 检查内存缓存
        with self._lock:
            if key in self._memory_cache:
                self._stats["hits"] += 1
                k, v = self._memory_cache[key]
                return (k.to(device), v.to(device))
        
        self._stats["misses"] += 1
        
        # 尝试从 SPDK 读取
        if self.spdk_enabled:
            try:
                block_id = hash(key) % 10000
                result = self._spdk_kv_store.get_kv_block(block_id, layer_id=0, device=device)
                if result is not None:
                    k, v = result
                    self._stats["spdk_operations"] += 1
                    
                    # 缓存到内存
                    with self._lock:
                        self._memory_cache[key] = (k.cpu(), v.cpu())
                    
                    return (k, v)
            except Exception as e:
                logger.warning(f"[SPDK KV Cache] SPDK 读取失败 {key}: {e}")
        
        return None
    
    def set_kv(self, key: str, k: torch.Tensor, v: torch.Tensor) -> bool:
        """
        设置 KV Cache
        
        Args:
            key: KV 键
            k: Key 张量
            v: Value 张量
        
        Returns:
            是否成功
        """
        self._stats["writes"] += 1
        
        # 缓存到内存
        with self._lock:
            self._memory_cache[key] = (k.cpu(), v.cpu())
        
        # 异步写入 SPDK
        if self.spdk_enabled:
            try:
                block_id = hash(key) % 10000
                self._spdk_kv_store.put_kv_block(block_id, k, v, layer_id=0)
                self._stats["spdk_operations"] += 1
                logger.debug(f"[SPDK KV Cache] 已写入 SPDK: {key}")
                return True
            except Exception as e:
                logger.warning(f"[SPDK KV Cache] SPDK 写入失败 {key}: {e}")
        
        self._stats["memory_operations"] += 1
        return True
    
    def delete_kv(self, key: str) -> bool:
        """
        删除 KV Cache
        
        Args:
            key: KV 键
        
        Returns:
            是否成功
        """
        with self._lock:
            if key in self._memory_cache:
                del self._memory_cache[key]
        
        if self.spdk_enabled:
            try:
                block_id = hash(key) % 10000
                self._spdk_kv_store.delete_kv_block(block_id, layer_id=0)
                return True
            except Exception as e:
                logger.warning(f"[SPDK KV Cache] SPDK 删除失败 {key}: {e}")
        
        return True
    
    def add_prefetch_key(self, key: str):
        """
        添加预取键
        
        Args:
            key: 要预取的键
        """
        self._prefetch_keys.add(key)
        
        # 如果 SPDK 可用，提交预取任务
        if self._spdk_io_manager:
            self._spdk_io_manager.add_prefetch_hint([key])
    
    def prefetch_all(self):
        """预取所有标记的键"""
        if self._spdk_io_manager and self._prefetch_keys:
            keys = list(self._prefetch_keys)
            logger.debug(f"[SPDK KV Cache] 预取 {len(keys)} 个键")
            self._spdk_io_manager.submit_batch_read(keys)
            self._prefetch_keys.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self._stats.copy()
        if self._spdk_kv_store:
            stats["spdk_stats"] = self._spdk_kv_store.get_stats()
        stats["memory_cache_size"] = len(self._memory_cache)
        return stats
    
    def clear(self):
        """清空所有缓存"""
        with self._lock:
            self._memory_cache.clear()
        self._prefetch_keys.clear()
        logger.debug("[SPDK KV Cache] 缓存已清空")
    
    def shutdown(self):
        """关闭 KV Cache"""
        if self._spdk_io_manager:
            self._spdk_io_manager.stop()
        self.clear()
        logger.info("[SPDK KV Cache] 已关闭")


# 使用示例
def main():
    logging.basicConfig(level=logging.INFO)
    
    kv_cache = SPDKKVCache()
    logger.info(f"SPDK 可用: {kv_cache.spdk_enabled}")
    
    # 创建测试张量
    k = torch.randn(1, 32, 128, 64)
    v = torch.randn(1, 32, 128, 64)
    
    # 设置 KV
    success = kv_cache.set_kv("test_session", k, v)
    logger.info(f"KV 设置成功: {success}")
    
    # 获取 KV
    result = kv_cache.get_kv("test_session")
    if result:
        k_out, v_out = result
        logger.info(f"KV 获取成功: k.shape={k_out.shape}, v.shape={v_out.shape}")
    
    # 统计信息
    stats = kv_cache.get_stats()
    logger.info(f"统计信息: {stats}")
    
    kv_cache.shutdown()


if __name__ == "__main__":
    main()
