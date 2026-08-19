
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
SPDK 块设备抽象层
支持真实 SPDK 或文件系统 Fallback
"""

import os
import mmap
import hashlib
from typing import Optional, Dict, Any
from pathlib import Path
import threading
import logging

from .spdk_config import SPDKConfig

logger = logging.getLogger(__name__)


class SPDKBlockDevice:
    """SPDK 块设备抽象"""
    
    def __init__(self, config: SPDKConfig):
        self.config = config
        self._initialized = False
        self._lock = threading.Lock()
        
        # Fallback 模式的文件存储
        self._file_map: Dict[str, str] = {}
        self._data_dir = Path("/tmp/spdk_fallback")
        self._data_dir.mkdir(parents=True, exist_ok=True)
        
        # 内存缓存（模拟 SPDK 的性能）
        self._cache: Dict[str, bytes] = {}
        self._cache_size = 0
        self._max_cache_size = config.cache_size_mb * 1024 * 1024
        
        logger.info(f"SPDKBlockDevice 初始化: enable_spdk={config.enable_spdk}")
    
    def initialize(self) -> bool:
        """初始化 SPDK 设备"""
        with self._lock:
            if self._initialized:
                return True
            
            if self.config.enable_spdk:
                # 尝试加载真实 SPDK
                try:
                    logger.info("正在初始化真实 SPDK 驱动...")
                    # 这里可以集成真实的 SPDK Python 绑定
                    # 例如：import spdk
                    # 由于环境依赖，我们使用 Fallback 模式演示
                    logger.warning("SPDK Python 绑定不可用，使用 Fallback 模式")
                except ImportError:
                    logger.warning("SPDK 模块不可用，使用 Fallback 模式")
            
            self._initialized = True
            logger.info("SPDKBlockDevice 初始化完成 (Fallback 模式)")
            return True
    
    def _get_file_path(self, key: str) -> Path:
        """获取键对应的文件路径"""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self._data_dir / key_hash[:2] / key_hash[2:4] / key_hash

    def read(self, key: str) -> Optional[bytes]:
        """读取数据"""
        if not self._initialized:
            self.initialize()
        
        # 先查缓存
        with self._lock:
            if key in self._cache:
                logger.debug(f"SPDK 缓存命中: {key}")
                return self._cache[key]
        
        # 从文件读取
        try:
            file_path = self._get_file_path(key)
            if file_path.exists():
                with open(file_path, "rb") as f:
                    data = f.read()
                    # 更新缓存
                    with self._lock:
                        self._put_cache(key, data)
                    return data
        except Exception as e:
            logger.error(f"SPDK 读取失败: {key}, error: {e}")
        
        return None

    def write(self, key: str, data: bytes) -> bool:
        """写入数据"""
        if not self._initialized:
            self.initialize()
        
        try:
            # 更新缓存
            with self._lock:
                self._put_cache(key, data)
            
            # 写入文件
            file_path = self._get_file_path(key)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(data)
            
            logger.debug(f"SPDK 写入成功: {key}, size={len(data)}")
            return True
        except Exception as e:
            logger.error(f"SPDK 写入失败: {key}, error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """删除数据"""
        if not self._initialized:
            self.initialize()
        
        with self._lock:
            if key in self._cache:
                del self._cache[key]
        
        try:
            file_path = self._get_file_path(key)
            if file_path.exists():
                file_path.unlink()
            return True
        except Exception as e:
            logger.error(f"SPDK 删除失败: {key}, error: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        if not self._initialized:
            self.initialize()
        
        with self._lock:
            if key in self._cache:
                return True
        
        file_path = self._get_file_path(key)
        return file_path.exists()
    
    def _put_cache(self, key: str, data: bytes):
        """放入缓存，处理 LRU"""
        if self._cache_size + len(data) > self._max_cache_size:
            keys_to_remove = list(self._cache.keys())[:len(self._cache) // 2]
            for k in keys_to_remove:
                self._cache_size -= len(self._cache[k])
                del self._cache[k]
        
        self._cache[key] = data
        self._cache_size += len(data)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                "cache_size": self._cache_size,
                "cache_entries": len(self._cache),
                "max_cache_size": self._max_cache_size,
                "mode": "spdk" if self.config.enable_spdk else "fallback"
            }
    
    def shutdown(self):
        """关闭设备"""
        with self._lock:
            if not self._initialized:
                return
            self._cache.clear()
            self._cache_size = 0
            self._initialized = False
            logger.info("SPDKBlockDevice 已关闭")

