
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
SPDK KV 存储
用于存储 KV Cache 块
"""

import struct
import pickle
from typing import Optional, Tuple, Dict, Any
import torch
import logging

from .spdk_config import SPDKConfig
from .spdk_block_device import SPDKBlockDevice

logger = logging.getLogger(__name__)


class SPDKKVStore:
    """SPDK KV Cache 存储"""
    
    def __init__(self, config: SPDKConfig, block_device: Optional[SPDKBlockDevice] = None):
        self.config = config
        self.block_device = block_device or SPDKBlockDevice(config)
        self._stats: Dict[str, int] = {
            "reads": 0,
            "writes": 0,
            "hits": 0,
            "misses": 0
        }
    
    def initialize(self):
        """初始化存储"""
        self.block_device.initialize()
        logger.info("SPDKKVStore 初始化完成")
    
    def _get_kv_key(self, block_id: int, layer_id: int = 0) -> str:
        """生成 KV 块的键"""
        return f"kv:layer_{layer_id}:block_{block_id}"
    
    def put_kv_block(self, block_id: int, k: torch.Tensor, v: torch.Tensor, layer_id: int = 0) -> bool:
        """存储 KV 块"""
        key = self._get_kv_key(block_id, layer_id)
        
        try:
            # 序列化张量
            data = {
                "k": k.cpu().numpy() if isinstance(k, torch.Tensor) else k,
                "v": v.cpu().numpy() if isinstance(v, torch.Tensor) else v,
                "shape_k": k.shape,
                "shape_v": v.shape,
                "dtype_k": str(k.dtype),
                "dtype_v": str(v.dtype)
            }
            serialized = pickle.dumps(data)
            
            # 写入 SPDK
            success = self.block_device.write(key, serialized)
            if success:
                self._stats["writes"] += 1
                logger.debug(f"KV 块已存储: block_id={block_id}, layer={layer_id}")
            return success
        except Exception as e:
            logger.error(f"存储 KV 块失败: {e}")
            return False
    
    def get_kv_block(self, block_id: int, layer_id: int = 0, device: str = "cpu") -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """获取 KV 块"""
        key = self._get_kv_key(block_id, layer_id)
        self._stats["reads"] += 1
        
        try:
            # 从 SPDK 读取
            serialized = self.block_device.read(key)
            if serialized is None:
                self._stats["misses"] += 1
                logger.debug(f"KV 块未找到: block_id={block_id}, layer={layer_id}")
                return None
            
            # 反序列化
            data = pickle.loads(serialized)
            
            # 重建张量
            import numpy as np
            k_np = np.asarray(data["k"])
            v_np = np.asarray(data["v"])
            
            k = torch.from_numpy(k_np).to(device)
            v = torch.from_numpy(v_np).to(device)
            
            self._stats["hits"] += 1
            logger.debug(f"KV 块已加载: block_id={block_id}, layer={layer_id}")
            return (k, v)
        except Exception as e:
            logger.error(f"加载 KV 块失败: {e}")
            self._stats["misses"] += 1
            return None
    
    def delete_kv_block(self, block_id: int, layer_id: int = 0) -> bool:
        """删除 KV 块"""
        key = self._get_kv_key(block_id, layer_id)
        success = self.block_device.delete(key)
        if success:
            logger.debug(f"KV 块已删除: block_id={block_id}, layer={layer_id}")
        return success
    
    def kv_block_exists(self, block_id: int, layer_id: int = 0) -> bool:
        """检查 KV 块是否存在"""
        key = self._get_kv_key(block_id, layer_id)
        return self.block_device.exists(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self._stats.copy()
        stats["device_stats"] = self.block_device.get_stats()
        return stats

