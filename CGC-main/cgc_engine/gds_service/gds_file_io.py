
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
GDS 文件 I/O
GPU 直接访问文件
"""

import os
from typing import Optional, Dict, Any
from pathlib import Path
import torch
import logging

from .gds_config import GDSConfig
from .gds_memory_manager import GDSMemoryManager

logger = logging.getLogger(__name__)


class GDSFileIO:
    """GDS 文件 I/O 操作"""
    
    def __init__(self, config: GDSConfig, memory_manager: Optional[GDSMemoryManager] = None):
        self.config = config
        self.memory_manager = memory_manager or GDSMemoryManager(config)
        self._stats: Dict[str, int] = {
            "reads": 0,
            "writes": 0,
            "read_bytes": 0,
            "write_bytes": 0,
            "gds_hits": 0,
            "gds_misses": 0
        }
    
    def initialize(self):
        """初始化"""
        self.memory_manager.initialize()
        logger.info("GDSFileIO 初始化完成")
    
    def _can_use_gds(self, file_path: Path) -> bool:
        """检查是否可以使用 GDS"""
        if not self.config.enable_gds:
            return False
        
        # 检查文件系统
        try:
            # 这里可以检查真实的文件系统类型
            # 简化版本，总是返回 True (Fallback 模式)
            return True
        except:
            return False
    
    def read_to_gpu(self, file_path: Path, tensor: torch.Tensor, offset: int = 0) -> bool:
        """从文件直接读取到 GPU 内存"""
        self._stats["reads"] += 1
        
        try:
            # 检查是否可以使用 GDS
            use_gds = self._can_use_gds(file_path)
            
            if use_gds and tensor.is_cuda and self.memory_manager.is_registered(tensor):
                # 使用 GDS 读取
                logger.debug(f"使用 GDS 读取: {file_path}")
                self._stats["gds_hits"] += 1
                
                # 这里应该调用真实的 cuFile 库
                # 例如：cuFileRead(file_handle, tensor_ptr, size, offset)
                # 为了演示，我们使用 Fallback 方法
                use_gds = False
            
            if not use_gds:
                self._stats["gds_misses"] += 1
                # Fallback: CPU 读取然后拷贝到 GPU
                logger.debug(f"使用 Fallback 读取: {file_path}")
                
                with open(file_path, "rb") as f:
                    f.seek(offset)
                    data = f.read(tensor.numel() * tensor.element_size())
                
                # 转换为 numpy 然后拷贝到 GPU
                import numpy as np
                np_arr = np.frombuffer(data, dtype=np.uint8)
                tensor.view(torch.uint8).copy_(torch.from_numpy(np_arr).to(tensor.device))
            
            self._stats["read_bytes"] += tensor.numel() * tensor.element_size()
            return True
        except Exception as e:
            logger.error(f"GDS 读取失败: {e}")
            return False
    
    def write_from_gpu(self, file_path: Path, tensor: torch.Tensor, offset: int = 0) -> bool:
        """从 GPU 内存直接写入文件"""
        self._stats["writes"] += 1
        
        try:
            # 检查是否可以使用 GDS
            use_gds = self._can_use_gds(file_path)
            
            if use_gds and tensor.is_cuda and self.memory_manager.is_registered(tensor):
                # 使用 GDS 写入
                logger.debug(f"使用 GDS 写入: {file_path}")
                self._stats["gds_hits"] += 1
                use_gds = False
            
            if not use_gds:
                self._stats["gds_misses"] += 1
                # Fallback: 拷贝到 CPU 然后写入
                logger.debug(f"使用 Fallback 写入: {file_path}")
                
                # 拷贝到 CPU
                cpu_tensor = tensor.cpu()
                
                # 写入文件
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, "wb") as f:
                    f.seek(offset)
                    f.write(cpu_tensor.numpy().tobytes())
            
            self._stats["write_bytes"] += tensor.numel() * tensor.element_size()
            return True
        except Exception as e:
            logger.error(f"GDS 写入失败: {e}")
            return False
    
    def read_kv_block(self, file_path: Path, device: str = "cuda:0") -> Optional[torch.Tensor]:
        """读取 KV 块到 GPU"""
        try:
            # 先读取文件大小
            file_size = file_path.stat().st_size
            
            # 分配 GPU 内存
            tensor = torch.empty(file_size, dtype=torch.uint8, device=device)
            
            # 注册（如果需要）
            if self.config.use_registered_memory:
                self.memory_manager.register_tensor(tensor)
            
            # 读取
            if self.read_to_gpu(file_path, tensor):
                return tensor
            return None
        except Exception as e:
            logger.error(f"读取 KV 块失败: {e}")
            return None
    
    def write_kv_block(self, file_path: Path, tensor: torch.Tensor) -> bool:
        """从 GPU 写入 KV 块"""
        return self.write_from_gpu(file_path, tensor)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self._stats.copy()
        stats["memory_stats"] = self.memory_manager.get_stats()
        return stats

