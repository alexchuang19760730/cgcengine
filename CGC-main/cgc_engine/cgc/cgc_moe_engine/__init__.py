"""CGC MoE Engine - Python Wrapper

訓推共用算子的 Python 介面。

設計：
1. 優先載入編譯好的 C++ extension（cgc_moe_engine.so）
2. 若未編譯，fallback 到純 PyTorch 實作（功能相同，速度較慢）
3. 推理路徑：直接呼叫 forward（torch.no_grad()）
4. 訓練路徑：透過 autograd::Function 自動獲得 backward

使用範例：

    # 推理
    with torch.no_grad():
        out = cgc_moe_engine.grouped_gemm_bf16_forward(tokens, weights, indices)

    # 訓練（自動 backward）
    out = cgc_moe_engine.grouped_gemm_bf16(tokens, weights, indices)
    loss = out.sum()
    loss.backward()  # 自動呼叫 C++ backward
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ============================================================================
# 載入 C++ extension（若已編譯）
# ============================================================================

_cpp_module = None
_cpp_load_attempted = False


def _load_cpp_module():
    """載入編譯好的 C++ extension"""
    global _cpp_module, _cpp_load_attempted
    if _cpp_load_attempted:
        return _cpp_module
    _cpp_load_attempted = True

    # 嘗試載入已編譯的 extension
    try:
        from torch.utils.cpp_extension import load_inline
        # 方法 1: 從 .so 載入
        so_path = os.environ.get("CGC_MOE_ENGINE_SO")
        if so_path and os.path.exists(so_path):
            torch.ops.load_library(so_path)
            _cpp_module = torch.ops.cgc_moe_engine
            logger.info(f"[cgc_moe_engine] Loaded from {so_path}")
            return _cpp_module
    except Exception as e:
        logger.debug(f"[cgc_moe_engine] .so load failed: {e}")

    # 方法 2: 嘗試 import（若已 pip install）
    try:
        import cgc_moe_engine as _mod  # type: ignore
        _cpp_module = _mod
        logger.info("[cgc_moe_engine] Loaded via import")
        return _cpp_module
    except ImportError:
        pass

    # 方法 3: JIT 編譯（開發用）
    if os.environ.get("CGC_MOE_ENGINE_JIT", "0") == "1":
        try:
            from torch.utils.cpp_extension import load
            cpp_dir = os.path.dirname(os.path.abspath(__file__))
            _cpp_module = load(
                name="cgc_moe_engine",
                sources=[
                    os.path.join(cpp_dir, "cgc_moe_engine.cpp"),
                    os.path.join(cpp_dir, "bindings.cpp"),
                ],
                extra_cflags=["-O2", "-std=c++17"],
                verbose=False,
            )
            logger.info("[cgc_moe_engine] JIT compiled")
            return _cpp_module
        except Exception as e:
            logger.warning(f"[cgc_moe_engine] JIT compile failed: {e}")

    logger.info("[cgc_moe_engine] C++ module not available, using PyTorch fallback")
    return None


def get_engine():
    """取得 engine 模組（C++ 或 fallback）"""
    global _cpp_module
    if _cpp_module is None:
        _cpp_module = _load_cpp_module()
    return _cpp_module


# ============================================================================
# PyTorch Fallback 實作（當 C++ 未編譯時）
# ============================================================================

def _pytorch_deepep_dispatch_forward(
    tokens: torch.Tensor,
    gating_logits: torch.Tensor,
    num_experts: int,
    num_experts_per_token: int,
    ep_size: int = 1,
    ep_rank: int = 0,
    mode: str = "normal",
) -> dict:
    """PyTorch 原生 fallback"""
    topk = gating_logits.topk(num_experts_per_token, dim=-1)
    weights = topk.values.softmax(-1)
    indices = topk.indices
    dispatched = tokens.repeat_interleave(num_experts_per_token, dim=0)
    return {
        "dispatched_tokens": dispatched,
        "dispatch_indices": indices,
        "weights": weights,
        "handle": torch.stack([indices.float(), weights], 0),
    }


def _pytorch_grouped_gemm_bf16_forward(
    tokens: torch.Tensor,
    expert_weights: torch.Tensor,
    indices: torch.Tensor,
    transposed: bool = False,
) -> torch.Tensor:
    """PyTorch 原生 grouped GEMM fallback（autograd-friendly）

    用 index_select + 批次 matmul 避免 in-place 操作，確保梯度能流回 expert_weights。
    訓練路徑需要 autograd，因此不可用 .item() 或 in-place 賦值。
    """
    num_tokens = tokens.size(0)
    k = indices.size(1)

    # 用 list 收集結果，最後 stack（避免 in-place 打斷 autograd）
    outputs = []
    for j in range(k):
        # indices[:, j]: [num_tokens] 每個 token 的第 j 個專家
        expert_idx_j = indices[:, j]  # [num_tokens]
        # 選出對應的專家權重: [num_tokens, hidden_in, hidden_out]
        w_j = expert_weights.index_select(0, expert_idx_j)
        if transposed:
            w_j = w_j.transpose(-1, -2)
        # tokens: [num_tokens, hidden_in] -> [num_tokens, 1, hidden_in]
        t = tokens.unsqueeze(1)
        # batch matmul: [num_tokens, 1, hidden_in] @ [num_tokens, hidden_in, hidden_out]
        #            -> [num_tokens, 1, hidden_out] -> [num_tokens, hidden_out]
        out_j = torch.bmm(t, w_j).squeeze(1)
        outputs.append(out_j)
    # stack: [num_tokens, k, hidden_out]
    return torch.stack(outputs, dim=1)


# ============================================================================
# 統一 API（自動選擇 C++ 或 fallback）
# ============================================================================

class CGCMoEEngine:
    """CGC MoE 訓推共用引擎

    自動選擇 C++ extension 或 PyTorch fallback。
    """

    def __init__(self):
        self._cpp = get_engine()
        self._backend = "cpp" if self._cpp is not None else "pytorch"
        logger.info(f"[CGCMoEEngine] Backend: {self._backend}")

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def version(self) -> str:
        if self._cpp is not None:
            return self._cpp.version()
        return "cgc_moe_engine-1.0.0-pytorch-fallback"

    def deepep_available(self) -> bool:
        if self._cpp is not None:
            return self._cpp.deepep_available()
        return False

    def deep_gemm_available(self) -> bool:
        if self._cpp is not None:
            return self._cpp.deep_gemm_available()
        return False

    # === 推理 API（forward only，無 autograd）===

    def deepep_dispatch_forward(
        self,
        tokens: torch.Tensor,
        gating_logits: torch.Tensor,
        num_experts: int,
        num_experts_per_token: int,
        ep_size: int = 1,
        ep_rank: int = 0,
        mode: str = "normal",
    ) -> dict:
        if self._cpp is not None:
            result = self._cpp.deepep_dispatch_forward(
                tokens, gating_logits, num_experts, num_experts_per_token,
                ep_size, ep_rank, mode
            )
            return {
                "dispatched_tokens": result.dispatched_tokens,
                "dispatch_indices": result.dispatch_indices,
                "weights": result.weights,
                "handle": result.handle,
            }
        return _pytorch_deepep_dispatch_forward(
            tokens, gating_logits, num_experts, num_experts_per_token, ep_size, ep_rank, mode
        )

    def deepep_combine_forward(
        self,
        expert_output: torch.Tensor,
        handle: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        if self._cpp is not None:
            return self._cpp.deepep_combine_forward(expert_output, handle, weights)
        # PyTorch fallback
        num_tokens = weights.size(0)
        k = weights.size(1)
        hidden_dim = expert_output.size(-1)
        reshaped = expert_output.view(num_tokens, k, hidden_dim)
        return (reshaped * weights.unsqueeze(-1)).sum(1)

    def grouped_gemm_bf16_forward(
        self,
        tokens: torch.Tensor,
        expert_weights: torch.Tensor,
        indices: torch.Tensor,
        transposed: bool = False,
    ) -> torch.Tensor:
        if self._cpp is not None:
            return self._cpp.grouped_gemm_bf16_forward(tokens, expert_weights, indices, transposed)
        return _pytorch_grouped_gemm_bf16_forward(tokens, expert_weights, indices, transposed)

    # === 訓練 API（透過 autograd，自動 backward）===

    def deepep_dispatch(
        self,
        tokens: torch.Tensor,
        gating_logits: torch.Tensor,
        num_experts: int,
        num_experts_per_token: int,
        ep_size: int = 1,
        ep_rank: int = 0,
        mode: str = "normal",
    ) -> torch.Tensor:
        """訓練用 dispatch（自動 backward）

        在 C++ 可用時，使用 autograd::Function（高效）
        否則用 PyTorch 原生（autograd 自動）
        """
        if self._cpp is not None:
            return self._cpp.deepep_dispatch(
                tokens, gating_logits, num_experts, num_experts_per_token,
                ep_size, ep_rank, mode
            )
        # PyTorch fallback（autograd 原生支持）
        result = _pytorch_deepep_dispatch_forward(
            tokens, gating_logits, num_experts, num_experts_per_token, ep_size, ep_rank, mode
        )
        return result["dispatched_tokens"]

    def grouped_gemm_bf16(
        self,
        tokens: torch.Tensor,
        expert_weights: torch.Tensor,
        indices: torch.Tensor,
        transposed: bool = False,
    ) -> torch.Tensor:
        """訓練用 GEMM（自動 backward）"""
        if self._cpp is not None:
            return self._cpp.grouped_gemm_bf16(tokens, expert_weights, indices, transposed)
        # PyTorch fallback（autograd 原生支持）
        return _pytorch_grouped_gemm_bf16_forward(tokens, expert_weights, indices, transposed)


# ============================================================================
# 單例
# ============================================================================

_engine_instance: Optional[CGCMoEEngine] = None


def get_moe_engine() -> CGCMoEEngine:
    """取得全域 MoE engine 單例"""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = CGCMoEEngine()
    return _engine_instance


# ============================================================================
# 訓推共用 MoE Layer（示範訓推算子共用）
# ============================================================================

class CGCMoELayer(nn.Module):
    """CGC 訓推共用 MoE Layer

    同一個 layer，推理和訓練都透過 CGCMoEEngine 呼叫同一份算子。
    - 推理：torch.no_grad() 下直接 forward
    - 訓練：autograd 自動計算 backward（透過 C++ autograd::Function）
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int = 2,
        expert_inter_dim: int = 1408,
        ep_size: int = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.ep_size = ep_size
        self.expert_inter_dim = expert_inter_dim

        # Gate
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

        # Expert weights (3 個 GEMM: gate_proj, up_proj, down_proj)
        self.expert_gate_weight = nn.Parameter(
            torch.randn(num_experts, hidden_dim, expert_inter_dim) * 0.02
        )
        self.expert_up_weight = nn.Parameter(
            torch.randn(num_experts, hidden_dim, expert_inter_dim) * 0.02
        )
        self.expert_down_weight = nn.Parameter(
            torch.randn(num_experts, expert_inter_dim, hidden_dim) * 0.02
        )

        self.engine = get_moe_engine()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """前向傳播

        訓練和推理都呼叫這個方法，差異只在是否有 gradient
        """
        batch_size, seq_len, hidden = tokens.shape
        tokens_flat = tokens.view(-1, hidden)  # [num_tokens, hidden]

        # Gate
        gating_logits = self.gate(tokens_flat)  # [num_tokens, num_experts]

        # Dispatch
        dispatch_result = self.engine.deepep_dispatch(
            tokens_flat,
            gating_logits,
            self.num_experts,
            self.num_experts_per_token,
            ep_size=self.ep_size,
            ep_rank=int(os.environ.get("RANK", "0")),
        )

        # Grouped GEMM: gate_proj + silu + up_proj
        gate_out = self.engine.grouped_gemm_bf16(
            dispatch_result, self.expert_gate_weight, self._get_indices(gating_logits)
        )
        up_out = self.engine.grouped_gemm_bf16(
            dispatch_result, self.expert_up_weight, self._get_indices(gating_logits)
        )
        act = F.silu(gate_out) * up_out

        # down_proj
        expert_output = self.engine.grouped_gemm_bf16(
            act, self.expert_down_weight, self._get_indices(gating_logits)
        )

        # Combine
        weights = gating_logits.topk(self.num_experts_per_token, dim=-1).values.softmax(-1)
        output = self.engine.deepep_combine_forward(
            expert_output, self._get_handle(gating_logits), weights
        )

        return output.view(batch_size, seq_len, hidden)

    def _get_indices(self, gating_logits: torch.Tensor) -> torch.Tensor:
        return gating_logits.topk(self.num_experts_per_token, dim=-1).indices

    def _get_handle(self, gating_logits: torch.Tensor) -> torch.Tensor:
        indices = self._get_indices(gating_logits)
        weights = gating_logits.topk(self.num_experts_per_token, dim=-1).values.softmax(-1)
        return torch.stack([indices.float(), weights], 0)
