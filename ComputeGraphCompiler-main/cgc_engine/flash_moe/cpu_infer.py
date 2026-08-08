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
flash_moe/cpu_infer.py - CPU 多線程加速推理

利用 CPU 多核心和 Intel MKL/OpenBLAS 進行加速
"""

import torch
import torch.nn.functional as F
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional


class CPUMLPInfer:
    """
    CPU MLP 推理器 - 多線程加速執行層

    執行層職責：
    - 利用多核心 CPU 並行計算
    - 支持 Intel MKL / OpenBLAS 加速
    - 異步加載專家權重
    """

    def __init__(
        self,
        num_threads: Optional[int] = None,
        use_openmp: bool = True,
    ):
        self.available = True
        self.device = torch.device("cpu")

        self.num_threads = num_threads or os.cpu_count() or 4
        self.use_openmp = use_openmp and "OMP_NUM_THREADS" in os.environ

        self._executor: Optional[ThreadPoolExecutor] = None
        self._init_thread_pool()

        torch.set_num_threads(self.num_threads)

    def _init_thread_pool(self):
        """初始化線程池用於異步操作"""
        self._executor = ThreadPoolExecutor(max_workers=self.num_threads)

    def run(
        self,
        x: torch.Tensor,
        expert_ids: list,
        cache_manager,
        parallel: bool = True,
    ) -> torch.Tensor:
        """
        執行 CPU 加速 MLP

        執行層職責：
        - 獲取緩存的專家權重
        - 執行 SwiGLU 操作
        - 可選並行計算多個專家
        """
        x = x.to(self.device)

        if parallel and len(expert_ids) > 1:
            return self._run_parallel(x, expert_ids, cache_manager)

        expert = cache_manager[expert_ids[0]]
        w1 = expert["w1"].to(self.device)
        w3 = expert["w3"].to(self.device)
        w2 = expert["w2"].to(self.device)
        gate = F.silu(F.linear(x, w1))
        up = F.linear(x, w3)
        return F.linear(gate * up, w2)

    def _run_parallel(
        self,
        x: torch.Tensor,
        expert_ids: List[int],
        cache_manager,
    ) -> torch.Tensor:
        """並行計算多個專家"""
        if self._executor is None:
            self._init_thread_pool()

        def compute_expert(idx: int) -> torch.Tensor:
            expert = cache_manager[idx]
            w1 = expert["w1"].to(self.device)
            w3 = expert["w3"].to(self.device)
            w2 = expert["w2"].to(self.device)
            gate = F.silu(F.linear(x, w1))
            up = F.linear(x, w3)
            return F.linear(gate * up, w2)

        futures = [
            self._executor.submit(compute_expert, idx)
            for idx in expert_ids
        ]

        outputs = [f.result() for f in futures]
        return torch.stack(outputs, dim=0).mean(dim=0)

    def run_moe(
        self,
        x: torch.Tensor,
        expert_ids: List[int],
        cache_manager,
        top_k: int = 2,
    ) -> torch.Tensor:
        """
        執行 CPU 加速 MoE

        執行層職責：
        - 加載多個專家權重
        - 並行計算提升吞吐量
        - 加權合併輸出
        """
        x = x.to(self.device)

        if self._executor is None:
            self._init_thread_pool()

        def compute_expert(idx: int) -> torch.Tensor:
            expert = cache_manager[idx]
            w1 = expert["w1"].to(self.device)
            w3 = expert["w3"].to(self.device)
            w2 = expert["w2"].to(self.device)
            gate = F.silu(F.linear(x, w1))
            up = F.linear(x, w3)
            return F.linear(gate * up, w2)

        futures = [
            self._executor.submit(compute_expert, idx)
            for idx in expert_ids[:top_k]
        ]

        outputs = [f.result() for f in futures]
        result = torch.stack(outputs, dim=0).mean(dim=0)

        return result

    def run_swiglu(
        self,
        x: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        執行 SwiGLU 激活函數

        SwiGLU = SiLU(W1 @ x) * (W3 @ x) @ W2
        """
        gate = F.linear(x, w1)
        if w3 is not None:
            up = F.linear(x, w3)
        elif hasattr(self, "_w3") and getattr(self, "_w3") is not None:
            up = F.linear(x, self._w3)
        else:
            up = gate
        return F.linear(F.silu(gate) * up, w2)

    def info(self):
        return {
            "available": self.available,
            "device": str(self.device),
            "num_threads": self.num_threads,
            "openmp_enabled": self.use_openmp,
            "mkl_available": torch.backends.mkldnn.is_available(),
        }

    def __del__(self):
        if self._executor is not None:
            self._executor.shutdown(wait=False)
