# Copyright (c) 2025 SandAI. All Rights Reserved.
"""
Base Parser - 基类和工具函数
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import torch


@dataclass
class ParsedModel:
    """解析后的模型结构"""
    model_type: str
    vocab_size: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    head_dim: int
    num_kv_heads: Optional[int] = None
    max_seq_len: Optional[int] = None
    layers: Dict[str, Dict] = None
    metadata: Optional[Dict] = None


@dataclass
class ParsedWeight:
    """解析后的权重"""
    name: str
    tensor: torch.Tensor
    shape: List[int]
    dtype: str
    layer_id: Optional[int] = None
    weight_type: Optional[str] = None


class BaseModelParser(ABC):
    """模型解析器基类"""

    def __init__(self, model_path: str, **kwargs):
        self.model_path = model_path
        self.kwargs = kwargs

    @abstractmethod
    def parse_model(self) -> ParsedModel:
        """解析模型结构"""
        pass

    @abstractmethod
    def load_weights(self, layer_id: Optional[int] = None) -> List[ParsedWeight]:
        """加载权重"""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict:
        """获取元数据"""
        pass

    @abstractmethod
    def close(self):
        """关闭解析器"""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
