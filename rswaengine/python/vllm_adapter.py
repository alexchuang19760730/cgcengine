# =============================================================================
# vllm_adapter.py  —  rswaengine 统一注入的 vLLM monkey-patch 消费者
# -----------------------------------------------------------------------------
# 与 sglang_adapter 对称：消费 inject_unified_ir_for_role 产出的统一 IR，把
# 「R-SWA + 窗口 OrthoKDA」安装进 vLLM 的 attention backend。
#
# 两种模式：
#   1) 真实 patch（vllm + torch 已安装）：替换注意力 backend（如
#      vllm.attention.backends 中的实现），注入 RSWAManager 可见性掩码。
#   2) 独立 demo（无 vllm/torch）：numpy 跑通同一条注意力路径。
#
# 部署时请以目标 vLLM 版本的 attention backend API 为准微调 patch 位置。
# =============================================================================
from __future__ import annotations

import numpy as np

from cgc_unified_injection import (
    RSWAManager,
    inject_unified_ir_for_role,
    CGC_INJECT_ROLE_DECODE,
    CGC_BACKEND_ADAPTER_VLLM,
)

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

try:
    import vllm  # noqa: F401
    HAS_VLLM = True
except Exception:
    HAS_VLLM = False


def _make_rswa_ortho_backend_class():
    import torch

    class RSWAOrthoBackend(torch.nn.Module):
        """替代 vLLM 的注意力 backend：每个序列维护 RSWAManager。"""

        def __init__(self, num_heads, head_dim, reference_len, window_size,
                     ortho_base_dim, decay_rate=0.01, hybrid_every=0,
                     enable_window_ortho=True):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = head_dim
            self.reference_len = reference_len
            self.window_size = window_size
            self.ortho_base_dim = ortho_base_dim
            self.decay_rate = decay_rate
            self.hybrid_every = hybrid_every
            self.enable_window_ortho = enable_window_ortho
            self._managers = {}

        def _get_mgr(self, seq_id):
            mgr = self._managers.get(seq_id)
            if mgr is None:
                mgr = RSWAManager(
                    self.num_heads, self.head_dim, self.reference_len,
                    self.window_size, self.ortho_base_dim, self.decay_rate,
                    self.hybrid_every, self.enable_window_ortho,
                )
                self._managers[seq_id] = mgr
            return mgr

        def prime_reference(self, seq_id, ref_k, ref_v):
            mgr = self._get_mgr(seq_id)
            for i in range(self.reference_len):
                mgr.set_reference(
                    np.ascontiguousarray(ref_k[i], dtype=np.float32),
                    np.ascontiguousarray(ref_v[i], dtype=np.float32),
                )

        def step(self, seq_id, k, v):
            return self._get_mgr(seq_id).feed_token(
                np.ascontiguousarray(k, dtype=np.float32),
                np.ascontiguousarray(v, dtype=np.float32),
            )

        def forward(self, q, seq_id=0):
            q_np = q.detach().cpu().numpy().astype(np.float32) if HAS_TORCH else np.asarray(q, dtype=np.float32)
            out = self._get_mgr(seq_id).combined_attention(q_np)
            return torch.from_numpy(out.copy()) if HAS_TORCH else out

        def get_state(self, seq_id):
            return self._get_mgr(seq_id).get_state()

        def set_state(self, seq_id, state):
            self._get_mgr(seq_id).set_state(state)

    return RSWAOrthoBackend


def patch_vllm(model, config: dict):
    """注册统一 IR 并替换 model 的 vLLM attention backend。

    config 同 sglang_adapter.patch_sglang。
    """
    if not (HAS_TORCH and HAS_VLLM):
        raise RuntimeError(
            "patch_vllm 需要 vllm + torch。当前未安装，请运行 run_demo() 验证路径，"
            "或在已装 vllm/torch 的环境调用本函数。"
        )

    inject_unified_ir_for_role(
        CGC_INJECT_ROLE_DECODE, CGC_BACKEND_ADAPTER_VLLM,
        num_layers=config["num_layers"], reference_len=config["reference_len"],
        window_size=config["window_size"], ortho_base_dim=config["ortho_base_dim"],
        num_heads=config["num_heads"], head_dim=config["head_dim"],
        decay_rate=config.get("decay_rate", 0.01),
        hybrid_every=config.get("hybrid_every", 0),
        enable_window_ortho=config.get("enable_window_ortho", True),
    )

    RSWAOrthoBackend = _make_rswa_ortho_backend_class()
    new_backend = RSWAOrthoBackend(
        config["num_heads"], config["head_dim"], config["reference_len"],
        config["window_size"], config["ortho_base_dim"],
        config.get("decay_rate", 0.01), config.get("hybrid_every", 0),
        config.get("enable_window_ortho", True),
    )
    # 模板：实际部署需按 vLLM 版本定位每层 attention backend 并原地替换。
    if hasattr(model, "attn_backend"):
        model.attn_backend = new_backend
    else:
        setattr(model, "rswa_ortho_backend", new_backend)
    print("[vllm_adapter] patched model with RSWAOrthoBackend "
          f"(num_layers={config['num_layers']}, ref_std forced, "
          f"window_ortho={config.get('enable_window_ortho', True)})")
    return new_backend


def run_demo():
    print("==============================================================")
    print(" vllm_adapter demo (standalone, numpy path)")
    print("==============================================================")
    NH, HD, REF, WIN, ORTHO = 2, 8, 4, 8, 8
    rng = np.random.default_rng(7)

    rc = inject_unified_ir_for_role(
        CGC_INJECT_ROLE_DECODE, CGC_BACKEND_ADAPTER_VLLM,
        num_layers=32, reference_len=REF, window_size=WIN,
        ortho_base_dim=ORTHO, num_heads=NH, head_dim=HD,
    )
    print(f"[demo] inject_unified_ir_for_role -> cgc_error_t={rc}")

    mgr = RSWAManager(NH, HD, REF, WIN, ORTHO)
    ref_k = rng.standard_normal((REF, NH, HD)).astype(np.float32)
    ref_v = rng.standard_normal((REF, NH, HD)).astype(np.float32)
    for i in range(REF):
        mgr.set_reference(ref_k[i], ref_v[i])

    evicted = []
    for t in range(20):
        k = rng.standard_normal((NH, HD)).astype(np.float32)
        v = rng.standard_normal((NH, HD)).astype(np.float32)
        ev = mgr.feed_token(k, v)
        if ev >= 0:
            evicted.append(ev)
    print(f"[demo] window evictions: {evicted[:6]}... total={len(evicted)}")
    print(f"[demo] visible count = {mgr.visible_count()} (expect ref+win={REF + WIN})")

    Q = rng.standard_normal((NH, HD)).astype(np.float32)
    out = mgr.combined_attention(Q)
    assert np.all(np.isfinite(out))
    st = mgr.get_state()
    print(f"[demo] combined_attention norm={float(np.linalg.norm(out)):.4f}; "
          f"wokdc current_dim={st['current_dim']}")
    mgr.destroy()
    print("[demo] PASS ✅  (vllm 真实 patch 见 patch_vllm()，需 vllm+torch)")


if __name__ == "__main__":
    run_demo()
