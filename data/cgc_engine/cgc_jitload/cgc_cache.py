
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
CGC 指令缓存
缓存已编译的 CGC 指令序列
"""

import pickle
import hashlib
from typing import Optional, Dict, Any, List
from pathlib import Path
import threading
import logging

from .jitload_config import JITLoadConfig

logger = logging.getLogger(__name__)


class CacheEntry:
    """缓存条目"""
    
    def __init__(self, key: str, value: Any, size: int = 0):
        self.key = key
        self.value = value
        self.size = size
        self.access_count = 0
        self.last_access = 0.0


class CGCCache:
    """CGC 指令缓存"""
    
    def __init__(self, config: JITLoadConfig):
        self.config = config
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._total_size = 0
        self._stats: Dict[str, int] = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "puts": 0
        }
    
    def _compute_key(self, data: Any) -> str:
        """计算缓存键"""
        serialized = pickle.dumps(data)
        return hashlib.sha256(serialized).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                entry.access_count += 1
                import time
                entry.last_access = time.time()
                self._stats["hits"] += 1
                logger.debug(f"CGC 缓存命中: {key[:16]}...")
                return entry.value
            self._stats["misses"] += 1
            return None
    
    def put(self, key: str, value: Any, size: int = 1):
        """存入缓存"""
        with self._lock:
            # 检查是否需要驱逐
            if key not in self._cache:
                while self._total_size + size > self.config.cgc_cache_size:
                    self._evict()
            
            # 存入
            entry = CacheEntry(key, value, size)
            import time
            entry.last_access = time.time()
            self._cache[key] = entry
            self._total_size += size
            self._stats["puts"] += 1
            logger.debug(f"CGC 缓存存入: {key[:16]}...")
    
    def _evict(self):
        """驱逐条目"""
        if not self._cache:
            return
        
        # LRU 策略
        sorted_entries = sorted(self._cache.values(), key=lambda e: e.last_access)
        victim = sorted_entries[0]
        
        del self._cache[victim.key]
        self._total_size -= victim.size
        self._stats["evictions"] += 1
        logger.debug(f"CGC 缓存驱逐: {victim.key[:16]}...")
    
    def save_to_disk(self, path: Optional[Path] = None):
        """保存到磁盘"""
        path = path or self.config.cache_dir / "cgc_cache.pkl"
        try:
            with self._lock:
                with open(path, "wb") as f:
                    pickle.dump(self._cache, f)
            logger.info(f"CGC 缓存已保存到磁盘: {path}")
        except Exception as e:
            logger.error(f"保存 CGC 缓存失败: {e}")
    
    def load_from_disk(self, path: Optional[Path] = None):
        """从磁盘加载"""
        path = path or self.config.cache_dir / "cgc_cache.pkl"
        if not path.exists():
            return
        
        try:
            with open(path, "rb") as f:
                cache = pickle.load(f)
            
            with self._lock:
                self._cache = cache
                self._total_size = sum(e.size for e in cache.values())
            
            logger.info(f"CGC 缓存已从磁盘加载: {len(cache)} entries")
        except Exception as e:
            logger.error(f"加载 CGC 缓存失败: {e}")
    
    def clear(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._total_size = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        with self._lock:
            stats = self._stats.copy()
            stats["size"] = len(self._cache)
            stats["total_size"] = self._total_size
            stats["max_size"] = self.config.cgc_cache_size
            if stats["hits"] + stats["misses"] > 0:
                stats["hit_rate"] = stats["hits"] / (stats["hits"] + stats["misses"])
            return stats

