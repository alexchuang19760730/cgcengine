
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
SPDK 专家权重存储
用于存储 MoE 专家权重
"""

import pickle
from typing import Optional, Dict, Any
import torch
import logging

from .spdk_config import SPDKConfig
from .spdk_block_device import SPDKBlockDevice

logger = logging.getLogger(__name__)


class SPDKExpertStore:
    """SPDK 专家权重存储"""
    
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
        logger.info("SPDKExpertStore 初始化完成")
    
    def _get_expert_key(self, expert_id: int, layer_id: int = 0) -> str:
        """生成专家权重的键"""
        return f"expert:layer_{layer_id}:expert_{expert_id}"
    
    def put_expert(self, expert_id: int, weights: Dict[str, torch.Tensor], layer_id: int = 0) -> bool:
        """存储专家权重"""
        key = self._get_expert_key(expert_id, layer_id)
        
        try:
            # 序列化权重
            data = {}
            for name, tensor in weights.items():
                data[name] = {
                    "value": tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor,
                    "shape": tensor.shape,
                    "dtype": str(tensor.dtype)
                }
            
            serialized = pickle.dumps(data)
            
            # 写入 SPDK
            success = self.block_device.write(key, serialized)
            if success:
                self._stats["writes"] += 1
                logger.debug(f"专家权重已存储: expert_id={expert_id}, layer={layer_id}")
            return success
        except Exception as e:
            logger.error(f"存储专家权重失败: {e}")
            return False
    
    def get_expert(self, expert_id: int, layer_id: int = 0, device: str = "cpu") -> Optional[Dict[str, torch.Tensor]]:
        """获取专家权重"""
        key = self._get_expert_key(expert_id, layer_id)
        self._stats["reads"] += 1
        
        try:
            # 从 SPDK 读取
            serialized = self.block_device.read(key)
            if serialized is None:
                self._stats["misses"] += 1
                logger.debug(f"专家权重未找到: expert_id={expert_id}, layer={layer_id}")
                return None
            
            # 反序列化
            data = pickle.loads(serialized)
            
            # 重建张量
            import numpy as np
            weights = {}
            for name, tensor_data in data.items():
                np_arr = np.asarray(tensor_data["value"])
                weights[name] = torch.from_numpy(np_arr).to(device)
            
            self._stats["hits"] += 1
            logger.debug(f"专家权重已加载: expert_id={expert_id}, layer={layer_id}")
            return weights
        except Exception as e:
            logger.error(f"加载专家权重失败: {e}")
            self._stats["misses"] += 1
            return None
    
    def delete_expert(self, expert_id: int, layer_id: int = 0) -> bool:
        """删除专家权重"""
        key = self._get_expert_key(expert_id, layer_id)
        success = self.block_device.delete(key)
        if success:
            logger.debug(f"专家权重已删除: expert_id={expert_id}, layer={layer_id}")
        return success
    
    def expert_exists(self, expert_id: int, layer_id: int = 0) -> bool:
        """检查专家权重是否存在"""
        key = self._get_expert_key(expert_id, layer_id)
        return self.block_device.exists(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self._stats.copy()
        stats["device_stats"] = self.block_device.get_stats()
        return stats

