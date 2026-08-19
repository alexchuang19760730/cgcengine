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
flash_moe/cuda_infer.py - CUDA 加速推理（NVIDIA GPU）
"""

import torch
import torch.nn.functional as F


class CudaMLPInfer:
    """
    CUDA MLP 推理器 - NVIDIA GPU 加速執行層

    執行層職責：
    - 調用 CUDA kernel 執行 MLP 計算
    - 利用 cuBLAS 加速 GEMM 操作
    - 支持 FP16/BF16 混合精度
    """

    def __init__(self):
        self.available = torch.cuda.is_available()
        self.device = torch.device("cuda") if self.available else None
        self._stream = None
        if self.available:
            self._stream = torch.cuda.Stream()

    def run(
        self,
        x: torch.Tensor,
        expert_ids: list,
        cache_manager,
        async_run: bool = True,
    ) -> torch.Tensor:
        """
        執行 CUDA 加速 MLP

        執行層職責：
        - 獲取緩存的專家權重
        - 異步執行 SwiGLU 操作
        - 自動選擇最優精度
        """
        if not self.available:
            raise RuntimeError("CUDA 不可用")

        x = x.to(self.device)

        with torch.cuda.stream(self._stream):
            expert = cache_manager[expert_ids[0]]
            w1 = expert["w1"].to(self.device)
            w3 = expert["w3"].to(self.device)
            w2 = expert["w2"].to(self.device)
            x = x.to(w1.dtype)
            gate = F.silu(F.linear(x, w1))
            up = F.linear(x, w3)
            out = F.linear(gate * up, w2)

        if not async_run:
            torch.cuda.current_stream().synchronize()

        return out

    def run_moe(
        self,
        x: torch.Tensor,
        expert_ids: list,
        cache_manager,
        top_k: int = 2,
        async_run: bool = True,
    ) -> torch.Tensor:
        """
        執行 CUDA 加速 MoE

        執行層職責：
        - 加載多個專家權重
        - 計算門控權重
        - 加權合併輸出
        """
        if not self.available:
            raise RuntimeError("CUDA 不可用")

        x = x.to(self.device)
        outputs = []

        with torch.cuda.stream(self._stream):
            for idx in expert_ids[:top_k]:
                expert = cache_manager[idx]
                w1 = expert["w1"].to(self.device)
                w3 = expert["w3"].to(self.device)
                w2 = expert["w2"].to(self.device)
                x = x.to(w1.dtype)
                gate = F.silu(F.linear(x, w1))
                up = F.linear(x, w3)
                outputs.append(F.linear(gate * up, w2))

            result = torch.stack(outputs, dim=0).mean(dim=0)

        if not async_run:
            torch.cuda.current_stream().synchronize()
        return result

    def info(self):
        return {
            "available": self.available,
            "device": str(self.device) if self.device else None,
            "cuda_device_name": torch.cuda.get_device_name(0) if self.available else None,
            "cuda_capability": f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}" if self.available else None,
        }
