# Copyright (c) 2025 SandAI. All Rights Reserved.
"""
策略基类 - 定义标准化接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseStrategy(ABC):
    """策略基类，定义标准化接口"""

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BaseStrategy":
        """从字典反序列化"""
        pass

    @abstractmethod
    def validate(self) -> bool:
        """验证策略配置是否合法"""
        pass

    def merge(self, other: "BaseStrategy") -> None:
        """合并另一个策略的配置"""
        pass