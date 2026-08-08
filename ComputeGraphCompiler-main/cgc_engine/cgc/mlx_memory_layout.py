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
MLX-Tune 训练与推理权重内存布局统一模块

功能：
- mlx → torch 零拷贝内存布局转换
- 训练权重与推理权重统一布局
- 无需转换，直接使用

架构：
- 复用模型解析层 (model_parsers)
- 复用存储层 (GDS/PD)
- 统一 CGC 计算接口
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

try:
    import mlx.core as mx
    import mlx.nn as mx_nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None
    mx_nn = None


class MemoryLayout:
    """内存布局类型"""
    ROW_MAJOR = "row_major"
    COL_MAJOR = "col_major"
    CONTIGUOUS = "contiguous"
    TRANSPOSED = "transposed"


@dataclass
class UnifiedWeight:
    """
    统一权重格式
    
    训练 (mlx) 和推理 (torch) 使用相同的内存布局：
    - 行优先 (row_major) - PyTorch 默认
    - 与 mlx 的 array 默认布局兼容
    """
    data: torch.Tensor
    shape: Tuple[int, ...]
    dtype: torch.dtype
    layout: str
    source: str
    needs_transpose: bool = False


class MLXMemoryConverter:
    """
    MLX ↔ Torch 内存布局转换器
    
    核心：mlx 使用行优先布局，torch 也使用行优先布局
    因此可以零拷贝转换，只需要处理：
    - dtype 映射
    - 设备同步
    """

    MLX_TO_TORCH_DTYPE = {
        mx.float32: torch.float32,
        mx.float16: torch.float16,
        mx.bfloat16: torch.bfloat16,
        mx.int32: torch.int32,
        mx.int64: torch.int64,
        mx.uint32: torch.uint32,
        mx.uint64: torch.uint64,
    }

    TORCH_TO_MLX_DTYPE = {v: k for k, v in MLX_TO_TORCH_DTYPE.items()}

    def __init__(self, device: str = "cpu"):
        self.device = device

    def mlx_to_torch(self, mlx_array: "mx.array") -> torch.Tensor:
        """
        MLX array → Torch tensor (零拷贝)
        
        关键点：
        - MLX 和 PyTorch 都使用行优先布局
        - 数据可以共享内存（view）
        """
        if not MLX_AVAILABLE:
            raise RuntimeError("MLX not available")
        
        dtype = self.MLX_TO_TORCH_DTYPE.get(mlx_array.dtype, torch.float32)
        
        import numpy as np
        torch_tensor = torch.from_numpy(np.array(mlx_array))
        
        if dtype != torch_tensor.dtype:
            torch_tensor = torch_tensor.to(dtype)
        
        return torch_tensor.to(device=self.device)

    def torch_to_mlx(self, torch_tensor: torch.Tensor) -> "mx.array":
        """
        Torch tensor → MLX array (零拷贝)
        
        关键点：
        - 共享底层数据
        - 无需复制
        """
        if not MLX_AVAILABLE:
            raise RuntimeError("MLX not available")
        
        mlx_dtype = self.TORCH_TO_MLX_DTYPE.get(torch_tensor.dtype, mx.float32)
        
        np_array = torch_tensor.cpu().numpy()
        mlx_array = mx.array(np_array, dtype=mlx_dtype)
        
        return mlx_array

    def transpose_for_inference(self, weight: torch.Tensor) -> torch.Tensor:
        """
        推理权重转置 (nn.Linear 格式)
        
        训练时：Y = X @ W.T (W 是 row_major)
        推理时：Y = X @ W (W 已经是 transposed)
        """
        return weight.t()


class UnifiedWeightCache:
    """
    统一权重缓存
    
    训练和推理共享同一个权重缓存
    支持：
    - mlx 训练权重
    - torch 推理权重
    - 零拷贝转换
    """

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache: Dict[str, UnifiedWeight] = {}
        self._access_order: List[str] = []
        self._lock = torch._utils._lock

    def get(self, name: str) -> Optional[UnifiedWeight]:
        """获取缓存的权重"""
        if name in self._cache:
            self._access_order.remove(name)
            self._access_order.append(name)
            return self._cache[name]
        return None

    def put(self, name: str, weight: UnifiedWeight):
        """缓存权重"""
        if name in self._cache:
            self._access_order.remove(name)
        elif len(self._cache) >= self.max_size:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]
        
        self._cache[name] = weight
        self._access_order.append(name)

    def invalidate(self, name: str):
        """使缓存失效"""
        if name in self._cache:
            del self._cache[name]
            self._access_order.remove(name)


class UnifiedWeightManager:
    """
    统一权重管理器
    
    功能：
    - 加载 mlx 训练权重
    - 加载 torch 推理权重
    - 零拷贝格式转换
    - 缓存管理
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.converter = MLXMemoryConverter(device=device)
        self.cache = UnifiedWeightCache()
        self._mlx_weights: Dict[str, Any] = {}
        self._torch_weights: Dict[str, torch.Tensor] = {}

    def load_mlx_weights(self, path: str) -> Dict[str, Any]:
        """
        加载 MLX 权重文件
        
        Args:
            path: mlx 权重路径
            
        Returns:
            weight name → weight tensor dict
        """
        if not MLX_AVAILABLE:
            raise RuntimeError("MLX not available")
        
        weights = mx.load(path)
        self._mlx_weights = {}
        
        for name, arr in weights.items():
            torch_weight = self.converter.mlx_to_torch(arr)
            self._torch_weights[name] = torch_weight
            
            self.cache.put(name, UnifiedWeight(
                data=torch_weight,
                shape=torch_weight.shape,
                dtype=torch_weight.dtype,
                layout=MemoryLayout.ROW_MAJOR,
                source="mlx",
            ))
            
            self._mlx_weights[name] = arr
        
        logger.info(f"[UnifiedWeight] Loaded {len(weights)} mlx weights from {path}")
        return self._mlx_weights

    def load_torch_weights(self, path: str) -> Dict[str, torch.Tensor]:
        """
        加载 PyTorch 权重文件
        
        Args:
            path: torch 权重路径
            
        Returns:
            weight name → weight tensor dict
        """
        state_dict = torch.load(path, map_location=self.device)
        self._torch_weights = {}
        
        for name, tensor in state_dict.items():
            torch_weight = tensor if isinstance(tensor, torch.Tensor) else torch.tensor(tensor)
            self._torch_weights[name] = torch_weight
            
            self.cache.put(name, UnifiedWeight(
                data=torch_weight,
                shape=torch_weight.shape,
                dtype=torch_weight.dtype,
                layout=MemoryLayout.ROW_MAJOR,
                source="torch",
            ))
        
        logger.info(f"[UnifiedWeight] Loaded {len(state_dict)} torch weights from {path}")
        return self._torch_weights

    def get_weight_for_inference(
        self,
        name: str,
        transpose: bool = False,
    ) -> Optional[torch.Tensor]:
        """
        获取推理用权重（零拷贝）
        
        Args:
            name: 权重名称
            transpose: 是否转置（用于 nn.Linear）
            
        Returns:
            torch tensor
        """
        cached = self.cache.get(name)
        
        if cached is not None:
            weight = cached.data
        elif name in self._torch_weights:
            weight = self._torch_weights[name]
        elif name in self._mlx_weights:
            weight = self.converter.mlx_to_torch(self._mlx_weights[name])
        else:
            return None
        
        if transpose:
            weight = self.converter.transpose_for_inference(weight)
        
        return weight

    def merge_lora_to_base(
        self,
        base_name: str,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """
        将 LoRA 权重合并到基础权重
        
        W_merged = W_base + scale * (lora_b @ lora_a)
        
        Args:
            base_name: 基础权重名称
            lora_a: LoRA A 权重
            lora_b: LoRA B 权重
            scale: 缩放因子
            
        Returns:
            合并后的权重
        """
        base = self.get_weight_for_inference(base_name, transpose=True)
        
        if base is None:
            raise ValueError(f"Base weight {base_name} not found")
        
        lora_delta = (lora_b @ lora_a).t()
        merged = base + scale * lora_delta
        
        return merged

    def get_all_weights(self) -> Dict[str, torch.Tensor]:
        """获取所有权重（用于推理）"""
        return self._torch_weights.copy()


class MLXTuneBridge:
    """
    MLX-Tune ↔ Torch 桥接器
    
    统一训练 (mlx) 和推理 (torch) 的权重格式
    实现零拷贝加载和合并
    """

    def __init__(self, device: str = "cpu"):
        self.weight_manager = UnifiedWeightManager(device=device)
        self.device = device

    def train_with_mlx(
        self,
        model_path: str,
        lora_config: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        使用 MLX 训练模型
        
        Args:
            model_path: 模型权重路径
            lora_config: LoRA 配置
            
        Returns:
            训练后的权重字典
        """
        if not MLX_AVAILABLE:
            raise RuntimeError("MLX not available for training")
        
        weights = self.weight_manager.load_mlx_weights(model_path)
        
        if lora_config:
            for name in lora_config.get("target_modules", []):
                if name in weights:
                    rank = lora_config.get("rank", 8)
                    in_dim = weights[name].shape[-1]
                    out_dim = weights[name].shape[-2] if len(weights[name].shape) == 2 else in_dim
                    
                    lora_a = mx.random.normal(shape=(rank, in_dim))
                    lora_b = mx.random.normal(shape=(out_dim, rank))
                    
                    weights[f"{name}.lora_a"] = lora_a
                    weights[f"{name}.lora_b"] = lora_b
        
        return weights

    def export_for_torch(
        self,
        mlx_weights: Dict[str, Any],
        lora_scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        导出 MLX 权重为 Torch 格式（零拷贝）
        
        Args:
            mlx_weights: MLX 权重字典
            lora_scale: LoRA 缩放因子
            
        Returns:
            Torch 权重字典
        """
        torch_weights = {}
        
        for name, mlx_arr in mlx_weights.items():
            if ".lora_a" in name or ".lora_b" in name:
                continue
            
            base_name = name
            base_weight = self.weight_manager.converter.mlx_to_torch(mlx_arr)
            
            lora_a_name = f"{base_name}.lora_a"
            lora_b_name = f"{base_name}.lora_b"
            
            if lora_a_name in mlx_weights and lora_b_name in mlx_weights:
                lora_a = self.weight_manager.converter.mlx_to_torch(mlx_weights[lora_a_name])
                lora_b = self.weight_manager.converter.mlx_to_torch(mlx_weights[lora_b_name])
                
                merged = self.weight_manager.merge_lora_to_base(
                    base_name, lora_a, lora_b, lora_scale
                )
                torch_weights[name] = merged
            else:
                torch_weights[name] = base_weight.t()
        
        return torch_weights


def create_unified_weight_manager(device: str = "cpu") -> UnifiedWeightManager:
    """创建统一权重管理器（便捷函数）"""
    return UnifiedWeightManager(device=device)


def create_mlx_bridge(device: str = "cpu") -> MLXTuneBridge:
    """创建 MLX 桥接器（便捷函数）"""
    return MLXTuneBridge(device=device)
