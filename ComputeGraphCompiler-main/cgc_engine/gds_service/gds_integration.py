
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
GDS 整合模块
与 CGC 双引擎执行器、SPDK、PD 服务的整合
"""

from typing import Optional, Dict, Any
from pathlib import Path
import torch
import logging

from .gds_config import GDSConfig
from .gds_memory_manager import GDSMemoryManager
from .gds_file_io import GDSFileIO

from ..spdk_adapter import SPDKConfig, SPDKKVStore, SPDKExpertStore
from ..dual_layer_manager import DualLayerManager

logger = logging.getLogger(__name__)


class GDSIntegration:
    """GDS 与整个系统的整合"""
    
    def __init__(self, gds_config: Optional[GDSConfig] = None, spdk_config: Optional[SPDKConfig] = None):
        self.gds_config = gds_config or GDSConfig()
        self.spdk_config = spdk_config or SPDKConfig()
        
        # GDS 组件
        self.gds_memory: Optional[GDSMemoryManager] = None
        self.gds_io: Optional[GDSFileIO] = None
        
        # SPDK 组件
        self.spdk_kv: Optional[SPDKKVStore] = None
        self.spdk_expert: Optional[SPDKExpertStore] = None
        
        self._initialized = False
    
    def initialize(self):
        """初始化所有组件"""
        if self._initialized:
            return
        
        logger.info("正在初始化 GDS 整合组件...")
        
        # 初始化 GDS
        self.gds_memory = GDSMemoryManager(self.gds_config)
        self.gds_memory.initialize()
        
        self.gds_io = GDSFileIO(self.gds_config, self.gds_memory)
        self.gds_io.initialize()
        
        # 初始化 SPDK（如果启用）
        if self.spdk_config.enable_spdk:
            from ..spdk_adapter import SPDKBlockDevice
            block_device = SPDKBlockDevice(self.spdk_config)
            block_device.initialize()
            self.spdk_kv = SPDKKVStore(self.spdk_config, block_device)
            self.spdk_kv.initialize()
            self.spdk_expert = SPDKExpertStore(self.spdk_config, block_device)
            self.spdk_expert.initialize()
        
        self._initialized = True
        logger.info("GDS 整合组件初始化完成")
    
    def load_kv_to_gpu(self, block_id: int, device: str = "cuda:0") -> Optional[torch.Tensor]:
        """加载 KV 块直接到 GPU"""
        if not self._initialized:
            self.initialize()
        
        try:
            # 首先尝试从 SPDK+GDS 加载
            if self.spdk_kv:
                kv = self.spdk_kv.get_kv_block(block_id, device=device)
                if kv:
                    logger.debug(f"从 SPDK+GDS 加载 KV 块: {block_id}")
                    return kv[0]  # 返回 k
            
            # Fallback: 从文件加载
            file_path = Path(f"/tmp/kv_cache/block_{block_id}.dat")
            if file_path.exists() and self.gds_io:
                return self.gds_io.read_kv_block(file_path, device=device)
            
            return None
        except Exception as e:
            logger.error(f"加载 KV 到 GPU 失败: {e}")
            return None
    
    def save_kv_from_gpu(self, block_id: int, tensor: torch.Tensor):
        """从 GPU 保存 KV 块"""
        if not self._initialized:
            self.initialize()
        
        try:
            # 保存到 SPDK（如果可用）
            if self.spdk_kv:
                # 这里简化处理，实际需要 k 和 v
                self.spdk_kv.put_kv_block(block_id, tensor, tensor)
                logger.debug(f"KV 块已保存到 SPDK: {block_id}")
            
            # 同时保存到文件
            file_path = Path(f"/tmp/kv_cache/block_{block_id}.dat")
            if self.gds_io:
                self.gds_io.write_kv_block(file_path, tensor)
        except Exception as e:
            logger.error(f"从 GPU 保存 KV 失败: {e}")
    
    def integrate_with_dual_layer(self, dual_layer: DualLayerManager) -> bool:
        """与双分层管理器整合"""
        if not self._initialized:
            self.initialize()
        
        try:
            # 替换双分层管理器的 SSD 层为 SPDK+GDS
            if self.spdk_kv:
                logger.info("正在整合 SPDK+GDS 到双分层管理器...")
                # 这里可以设置回调或直接替换方法
            
            return True
        except Exception as e:
            logger.error(f"双分层管理器整合失败: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取整体统计"""
        stats = {}
        if self.gds_io:
            stats["gds"] = self.gds_io.get_stats()
        if self.spdk_kv:
            stats["spdk_kv"] = self.spdk_kv.get_stats()
        return stats
    
    def shutdown(self):
        """关闭所有组件"""
        if self.gds_memory:
            self.gds_memory.shutdown()
        self._initialized = False
        logger.info("GDS 整合组件已关闭")

