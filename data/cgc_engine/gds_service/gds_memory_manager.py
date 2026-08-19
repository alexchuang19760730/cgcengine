
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
GDS 内存管理器
管理 GPU 内存注册和分配
"""

import threading
from typing import Optional, Dict, Any, Tuple
import torch
import logging

from .gds_config import GDSConfig

logger = logging.getLogger(__name__)


class GDSMemoryPool:
    """GDS 内存池"""
    
    def __init__(self, size_bytes: int):
        self.size_bytes = size_bytes
        self._used_bytes = 0
        self._lock = threading.Lock()
        self._allocations: Dict[int, Tuple[int, int]] = {}  # ptr: (offset, size)
    
    def allocate(self, size: int, alignment: int = 4096) -> Optional[int]:
        """分配内存（模拟）"""
        with self._lock:
            if self._used_bytes + size > self.size_bytes:
                return None
            # 简单的 bump allocator
            offset = self._used_bytes
            # 对齐
            aligned_offset = ((offset + alignment - 1) // alignment) * alignment
            self._used_bytes = aligned_offset + size
            return aligned_offset
    
    def free(self, ptr: int):
        """释放内存（模拟）"""
        with self._lock:
            if ptr in self._allocations:
                offset, size = self._allocations.pop(ptr)
                # 这里可以实现更复杂的内存回收
                pass


class GDSMemoryManager:
    """GDS 内存管理器"""
    
    def __init__(self, config: GDSConfig):
        self.config = config
        self._initialized = False
        self._lock = threading.Lock()
        
        # 内存池
        self._memory_pool: Optional[GDSMemoryPool] = None
        
        # 注册的张量
        self._registered_tensors: Dict[int, torch.Tensor] = {}
        
        # 统计
        self._stats: Dict[str, int] = {
            "registered_bytes": 0,
            "allocation_count": 0,
            "free_count": 0
        }
    
    def initialize(self) -> bool:
        """初始化 GDS 内存管理器"""
        with self._lock:
            if self._initialized:
                return True
            
            if self.config.enable_gds:
                # 尝试加载真实的 GDS 库
                try:
                    logger.info("正在初始化真实 GDS 内存管理器...")
                    # 这里可以集成真实的 cuFile 库
                    # 例如：import cupy
                    # 或者使用 NVIDIA 的 cufile Python 绑定
                    logger.warning("GDS Python 绑定不可用，使用 Fallback 模式")
                except ImportError:
                    logger.warning("GDS 模块不可用，使用 Fallback 模式")
            
            # 初始化内存池
            if self.config.use_registered_memory:
                pool_size = self.config.registered_memory_size_mb * 1024 * 1024
                self._memory_pool = GDSMemoryPool(pool_size)
                logger.info(f"GDS 内存池初始化: {pool_size / (1024*1024)} MB")
            
            self._initialized = True
            logger.info("GDSMemoryManager 初始化完成 (Fallback 模式)")
            return True
    
    def register_tensor(self, tensor: torch.Tensor) -> bool:
        """注册张量用于 GDS 传输"""
        if not self._initialized:
            self.initialize()
        
        try:
            # 只有 GPU 张量才能注册
            if not tensor.is_cuda:
                logger.warning("只能注册 CUDA 张量用于 GDS")
                return False
            
            ptr = tensor.data_ptr()
            
            if ptr in self._registered_tensors:
                return True
            
            # 模拟注册
            self._registered_tensors[ptr] = tensor
            self._stats["registered_bytes"] += tensor.numel() * tensor.element_size()
            self._stats["allocation_count"] += 1
            
            logger.debug(f"GDS 张量已注册: ptr={ptr}, size={tensor.shape}")
            return True
        except Exception as e:
            logger.error(f"GDS 张量注册失败: {e}")
            return False
    
    def unregister_tensor(self, tensor: torch.Tensor) -> bool:
        """注销张量"""
        if not self._initialized:
            return True
        
        try:
            ptr = tensor.data_ptr()
            if ptr in self._registered_tensors:
                del self._registered_tensors[ptr]
                self._stats["registered_bytes"] -= tensor.numel() * tensor.element_size()
                self._stats["free_count"] += 1
                logger.debug(f"GDS 张量已注销: ptr={ptr}")
            return True
        except Exception as e:
            logger.error(f"GDS 张量注销失败: {e}")
            return False
    
    def is_registered(self, tensor: torch.Tensor) -> bool:
        """检查张量是否已注册"""
        ptr = tensor.data_ptr()
        return ptr in self._registered_tensors
    
    def allocate_gds_buffer(self, size: int, device: str = "cuda:0") -> Optional[torch.Tensor]:
        """分配 GDS 缓冲区"""
        if not self._initialized:
            self.initialize()
        
        try:
            # 分配对齐的 GPU 张量
            tensor = torch.empty(size, dtype=torch.uint8, device=device)
            
            # 尝试注册
            if self.register_tensor(tensor):
                return tensor
            
            return tensor
        except Exception as e:
            logger.error(f"GDS 缓冲区分配失败: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self._stats.copy()
        stats["registered_tensors"] = len(self._registered_tensors)
        return stats
    
    def shutdown(self):
        """关闭内存管理器"""
        with self._lock:
            if not self._initialized:
                return
            self._registered_tensors.clear()
            self._memory_pool = None
            self._initialized = False
            logger.info("GDSMemoryManager 已关闭")

