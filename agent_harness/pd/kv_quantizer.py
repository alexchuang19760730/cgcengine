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
KV Cache Quantizer - 4-bit / 8-bit KV Cache 量化支持

支持:
- INT8 量化 (降低 50% 显存)
- INT4 量化 (降低 75% 显存)
- Symmetric / Asymmetric quantization
- Group-wise quantization (per 128 tokens)
"""

import torch
import pickle
from typing import Dict, Any, Tuple, Optional, Literal
from dataclasses import dataclass


@dataclass
class QuantizedKV:
    """量化後的 KV 數據"""
    k_quantized: torch.Tensor
    v_quantized: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    k_zero: Optional[torch.Tensor] = None
    v_zero: Optional[torch.Tensor] = None
    bits: int = 8
    group_size: int = 128


class KVQuantizer:
    """KV Cache 量化器"""

    def __init__(self, bits: int = 8, group_size: int = 128, symmetric: bool = True):
        """
        Args:
            bits: 量化位數 (4 或 8)
            group_size: 每組量化的 token 數
            symmetric: 是否使用對稱量化
        """
        self.bits = bits
        self.group_size = group_size
        self.symmetric = symmetric
        self._validate_config()

    def _validate_config(self):
        """驗證配置"""
        if self.bits not in [4, 8]:
            raise ValueError(f"不支持的位數: {self.bits}，只支持 4 或 8")
        if self.group_size <= 0:
            raise ValueError(f"group_size 必須大於 0: {self.group_size}")

    def quantize(
        self, 
        k: torch.Tensor, 
        v: torch.Tensor
    ) -> QuantizedKV:
        """
        量化 KV Cache

        Args:
            k: Key tensor [batch, heads, seq_len, dim]
            v: Value tensor [batch, heads, seq_len, dim]

        Returns:
            QuantizedKV: 量化後的 KV
        """
        # 按組量化
        batch_size, num_heads, seq_len, dim = k.shape
        
        # Reshape 以便按組量化
        k_reshaped = k.reshape(batch_size, num_heads, -1, self.group_size, dim)
        v_reshaped = v.reshape(batch_size, num_heads, -1, self.group_size, dim)
        
        # 計算 scale
        if self.symmetric:
            k_scale = k_reshaped.abs().max(dim=-2, keepdim=True)[0]
            v_scale = v_reshaped.abs().max(dim=-2, keepdim=True)[0]
            k_zero = None
            v_zero = None
        else:
            k_min = k_reshaped.min(dim=-2, keepdim=True)[0]
            k_max = k_reshaped.max(dim=-2, keepdim=True)[0]
            k_scale = (k_max - k_min) / 2
            k_zero = -(k_min + k_max) / 2
            
            v_min = v_reshaped.min(dim=-2, keepdim=True)[0]
            v_max = v_reshaped.max(dim=-2, keepdim=True)[0]
            v_scale = (v_max - v_min) / 2
            v_zero = -(v_min + v_max) / 2
        
        # 量化
        q_min = -(2 ** (self.bits - 1))
        q_max = 2 ** (self.bits - 1) - 1
        
        k_normalized = k_reshaped / (k_scale + 1e-8)
        v_normalized = v_reshaped / (v_scale + 1e-8)
        
        if not self.symmetric:
            k_normalized = k_normalized - k_zero / (k_scale + 1e-8)
            v_normalized = v_normalized - v_zero / (v_scale + 1e-8)
        
        k_quantized = k_normalized.round().clamp(q_min, q_max).to(torch.int8 if self.bits == 8 else torch.int32)
        v_quantized = v_normalized.round().clamp(q_min, q_max).to(torch.int8 if self.bits == 8 else torch.int32)
        
        # Reshape 回原始形狀
        k_quantized = k_quantized.reshape(k.shape)
        v_quantized = v_quantized.reshape(v.shape)
        
        return QuantizedKV(
            k_quantized=k_quantized,
            v_quantized=v_quantized,
            k_scale=k_scale.reshape(batch_size, num_heads, -1, dim),
            v_scale=v_scale.reshape(batch_size, num_heads, -1, dim),
            k_zero=k_zero.reshape(batch_size, num_heads, -1, dim) if k_zero is not None else None,
            v_zero=v_zero.reshape(batch_size, num_heads, -1, dim) if v_zero is not None else None,
            bits=self.bits,
            group_size=self.group_size,
        )

    def dequantize(
        self, 
        quantized: QuantizedKV
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        反量化 KV Cache

        Args:
            quantized: QuantizedKV 對象

        Returns:
            (k_dequantized, v_dequantized): 反量化後的 KV
        """
        k_quantized = quantized.k_quantized.to(torch.float32)
        v_quantized = quantized.v_quantized.to(torch.float32)
        
        # Reshape 以便按組反量化
        batch_size, num_heads, seq_len, dim = k_quantized.shape
        num_groups = (seq_len + self.group_size - 1) // self.group_size
        
        k_reshaped = k_quantized.reshape(batch_size, num_heads, num_groups, self.group_size, dim)
        v_reshaped = v_quantized.reshape(batch_size, num_heads, num_groups, self.group_size, dim)
        
        k_scale_reshaped = quantized.k_scale.reshape(batch_size, num_heads, num_groups, 1, dim)
        v_scale_reshaped = quantized.v_scale.reshape(batch_size, num_heads, num_groups, 1, dim)
        
        # 反量化
        k_dequantized = k_reshaped * k_scale_reshaped
        v_dequantized = v_reshaped * v_scale_reshaped
        
        if not self.symmetric and quantized.k_zero is not None and quantized.v_zero is not None:
            k_zero_reshaped = quantized.k_zero.reshape(batch_size, num_heads, num_groups, 1, dim)
            v_zero_reshaped = quantized.v_zero.reshape(batch_size, num_heads, num_groups, 1, dim)
            k_dequantized = k_dequantized + k_zero_reshaped
            v_dequantized = v_dequantized + v_zero_reshaped
        
        return (
            k_dequantized.reshape(k_quantized.shape),
            v_dequantized.reshape(v_quantized.shape)
        )

    def serialize(self, quantized: QuantizedKV) -> bytes:
        """序列化 QuantizedKV 為 bytes"""
        # 先移到 CPU 以便存儲
        cpu_data = {
            'k_quantized': quantized.k_quantized.cpu(),
            'v_quantized': quantized.v_quantized.cpu(),
            'k_scale': quantized.k_scale.cpu(),
            'v_scale': quantized.v_scale.cpu(),
            'k_zero': quantized.k_zero.cpu() if quantized.k_zero is not None else None,
            'v_zero': quantized.v_zero.cpu() if quantized.v_zero is not None else None,
            'bits': quantized.bits,
            'group_size': quantized.group_size,
        }
        return pickle.dumps(cpu_data)

    def deserialize(self, data: bytes, device: str = 'cpu') -> QuantizedKV:
        """反序列化 bytes 為 QuantizedKV"""
        cpu_data = pickle.loads(data)
        return QuantizedKV(
            k_quantized=cpu_data['k_quantized'].to(device),
            v_quantized=cpu_data['v_quantized'].to(device),
            k_scale=cpu_data['k_scale'].to(device),
            v_scale=cpu_data['v_scale'].to(device),
            k_zero=cpu_data['k_zero'].to(device) if cpu_data['k_zero'] is not None else None,
            v_zero=cpu_data['v_zero'].to(device) if cpu_data['v_zero'] is not None else None,
            bits=cpu_data['bits'],
            group_size=cpu_data['group_size'],
        )
