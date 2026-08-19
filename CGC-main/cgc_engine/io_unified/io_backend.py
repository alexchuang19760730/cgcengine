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
IOBackend - I/O 後端抽象介面

所有 I/O 後端必須實現此介面：
- load_kv() / save_kv()
- load_weight() / save_weight()
- load_expert() / save_expert()
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
from dataclasses import dataclass
import torch


@dataclass
class IOStats:
    """I/O 統計"""
    reads: int = 0
    writes: int = 0
    hits: int = 0
    misses: int = 0
    bytes_read: int = 0
    bytes_written: int = 0


class IOBackend(ABC):
    """
    I/O 後端抽象介面

    所有後端 (GDS/SPDK/MPS) 必須實現此介面
    """

    @abstractmethod
    def load_kv(
        self,
        key: str,
        seq_len: int,
        head_dim: int,
        num_heads: int = 32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        載入 KV Cache

        Args:
            key: KV Cache 鍵
            seq_len: 序列長度
            head_dim: 頭維度
            num_heads: KV 頭數

        Returns:
            (k, v) - Key 和 Value tensor
        """
        pass

    @abstractmethod
    def save_kv(
        self,
        key: str,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> bool:
        """
        儲存 KV Cache

        Args:
            key: KV Cache 鍵
            k: Key tensor
            v: Value tensor

        Returns:
            success
        """
        pass

    @abstractmethod
    def load_weight(
        self,
        path: str,
        shape: List[int],
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """
        載入權重

        Args:
            path: 權重路徑
            shape: 權重形狀
            dtype: 權重資料類型

        Returns:
            weight tensor
        """
        pass

    @abstractmethod
    def save_weight(
        self,
        path: str,
        tensor: torch.Tensor,
    ) -> bool:
        """
        儲存權重

        Args:
            path: 權重路徑
            tensor: 權重張量

        Returns:
            success
        """
        pass

    @abstractmethod
    def load_expert(
        self,
        expert_id: int,
        path: str,
    ) -> torch.Tensor:
        """
        載入專家權重 (MoE)

        Args:
            expert_id: 專家 ID
            path: 專家權重路徑

        Returns:
            expert weight tensor
        """
        pass

    @abstractmethod
    def save_expert(
        self,
        expert_id: int,
        tensor: torch.Tensor,
    ) -> bool:
        """
        儲存專家權重 (MoE)

        Args:
            expert_id: 專家 ID
            tensor: 專家權重張量

        Returns:
            success
        """
        pass

    @abstractmethod
    def prefetch(self, keys: List[str]) -> None:
        """
        預取資料

        Args:
            keys: 要預取的鍵列表
        """
        pass

    @abstractmethod
    def evict(self, keys: List[str]) -> None:
        """
        驅逐資料

        Args:
            keys: 要驅逐的鍵列表
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """後端名稱"""
        pass

    @property
    @abstractmethod
    def stats(self) -> IOStats:
        """I/O 統計"""
        pass

    @abstractmethod
    def initialize(self) -> None:
        """初始化後端"""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """關閉後端"""
        pass
