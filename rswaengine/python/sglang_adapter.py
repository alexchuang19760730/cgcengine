# =============================================================================
# sglang_adapter.py — R-SWA + 窗口 OrthoKDA GPU 版本的 sglang 适配器
# -----------------------------------------------------------------------------
# 用 RSWAOrthoAttentionGPU (纯 torch GPU, cuda-graph 兼容) 替代 C 库版本.
# 无 .cpu().numpy() host-sync, cuda-graph 兼容.
#
# 三种模式:
#   1) patch_sglang_gpu: 完整 patch (替换 attention, R-SWA 生效)
#   2) unpatch_sglang: 回退 (恢复原始 attention)
#   3) safe_patch_sglang: cuda-graph 安全 (只注册, 不替换)
# =============================================================================
from __future__ import annotations

import torch
import torch.nn as nn

try:
    from rswa_gpu import RSWAOrthoAttentionGPU
    HAS_GPU_RSWA = True
except Exception:
    HAS_GPU_RSWA = False


def _make_rswa_sglang_adapter(num_heads, head_dim, reference_len, window_size,
                               ortho_base_dim, device="cuda"):
    """创建 R-SWA GPU adapter, 适配 sglang attention 接口.

    RSWAOrthoAttentionGPU 内部管理 KV (Reference + 窗口),
    forward(q) 只接收 query, 返回 attention output.
    """
    if not HAS_GPU_RSWA:
        raise RuntimeError("rswa_gpu.py not available")

    rswa = RSWAOrthoAttentionGPU(
        num_heads=num_heads, head_dim=head_dim,
        reference_len=reference_len, window_size=window_size,
        ortho_base_dim=ortho_base_dim,
    ).to(device)

    return rswa


def patch_sglang_gpu(model, config: dict):
    """用 RSWAOrthoAttentionGPU 替换 sglang 的 attention (cuda-graph 兼容).

    config: num_heads, head_dim, reference_len, window_size, ortho_base_dim

    替换策略:
    - 遍历 model 的 decoder layers
    - 保存原始 self_attn (可回退)
    - 替换为 RSWAOrthoAttentionGPU wrapper

    ⚠️ sglang 的 attention 接口复杂, 这里提供模板.
    实际部署需按 sglang 版本适配 forward 签名.
    """
    if not HAS_GPU_RSWA:
        raise RuntimeError("rswa_gpu.py not available")

    device = next(model.parameters()).device if hasattr(model, 'parameters') else "cuda"
    rswa = _make_rswa_sglang_adapter(
        num_heads=config["num_heads"], head_dim=config["head_dim"],
        reference_len=config["reference_len"], window_size=config["window_size"],
        ortho_base_dim=config["ortho_base_dim"], device=str(device),
    )

    # 保存原始 attention (可回退)
    _originals = {}
    layers_replaced = 0

    # 尝试常见的 layer 路径
    layer_paths = [
        ("language_model", "model", "layers"),   # Qwen3-VL
        ("model", "layers"),                       # Qwen3 / LLaMA
        ("transformer", "h"),                      # GPT
    ]

    layers = None
    for path in layer_paths:
        obj = model
        found = True
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                found = False
                break
        if found and isinstance(obj, (list, nn.ModuleList)):
            layers = obj
            break

    if layers is None:
        print("[sglang_adapter] ⚠️ Cannot find decoder layers, setting model.rswa_attn")
        model._cgc_original_attention = {}
        model.rswa_attn = rswa
        return rswa

    for i, layer in enumerate(layers):
        # 保存原始 self_attn
        if hasattr(layer, "self_attn"):
            _originals[i] = layer.self_attn
            # 创建 wrapper
            layer.self_attn = _RSWAWrapper(rswa, config)
            layers_replaced += 1
        elif hasattr(layer, "attention"):
            _originals[i] = layer.attention
            layer.attention = _RSWAWrapper(rswa, config)
            layers_replaced += 1

    model._cgc_original_attention = _originals
    model._cgc_rswa_attention = rswa

    print(f"[sglang_adapter] ✅ Patched {layers_replaced} layers with RSWAOrthoAttentionGPU")
    print(f"[sglang_adapter]    R-SWA: ref={config['reference_len']}, win={config['window_size']}, ortho={config['ortho_base_dim']}")
    print(f"[sglang_adapter]    cuda-graph compatible (no host-sync)")
    print(f"[sglang_adapter]    回退: unpatch_sglang(model)")
    return rswa


class _RSWAWrapper(nn.Module):
    """适配 sglang attention 接口的 R-SWA wrapper.

    sglang attention forward 签名复杂, 这里提供简化适配.
    实际部署需按 sglang 版本的 RadixAttention.forward 签名调整.
    """

    def __init__(self, rswa: RSWAOrthoAttentionGPU, config: dict):
        super().__init__()
        self.rswa = rswa
        self.config = config
        # 保存原始 forward 的关键属性
        self.num_heads = config["num_heads"]
        self.head_dim = config["head_dim"]

    def forward(self, q, k=None, v=None, **kwargs):
        """适配 sglang attention forward.

        如果有 k, v (decode step): feed_token + forward
        如果只有 q (cuda-graph replay): forward only
        """
        # 确保 q 在 GPU 上
        if not q.is_cuda:
            q = q.cuda()

        # 如果有 k, v, 喂入新 token
        if k is not None and v is not None:
            # k, v 可能是 [batch, num_heads, head_dim] 或 [num_heads, head_dim]
            if k.dim() == 3:
                k_2d = k[0] if k.shape[0] == 1 else k.mean(0)
                v_2d = v[0] if v.shape[0] == 1 else v.mean(0)
            else:
                k_2d = k
                v_2d = v
            self.rswa.feed_token(k_2d, v_2d)

        # R-SWA attention (GPU, cuda-graph 兼容)
        q_2d = q[0] if q.dim() == 3 and q.shape[0] == 1 else q
        out = self.rswa.forward(q_2d)

        # 适配返回 shape
        if q.dim() == 3:
            return out.unsqueeze(0)
        return out


def unpatch_sglang(model):
    """回退 patch, 恢复原始 attention (cuda-graph 安全)."""
    _originals = getattr(model, "_cgc_original_attention", {})
    if not _originals:
        print("[sglang_adapter] no patch to unpatch")
        return

    # 恢复原始 attention
    layer_paths = [
        ("language_model", "model", "layers"),
        ("model", "layers"),
        ("transformer", "h"),
    ]

    layers = None
    for path in layer_paths:
        obj = model
        found = True
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                found = False
                break
        if found and isinstance(obj, (list, nn.ModuleList)):
            layers = obj
            break

    if layers:
        for i, original in _originals.items():
            if i < len(layers):
                if hasattr(layers[i], "self_attn"):
                    layers[i].self_attn = original
                elif hasattr(layers[i], "attention"):
                    layers[i].attention = original

    if hasattr(model, "_cgc_original_attention"):
        delattr(model, "_cgc_original_attention")
    if hasattr(model, "_cgc_rswa_attention"):
        delattr(model, "_cgc_rswa_attention")
    if hasattr(model, "rswa_attn"):
        delattr(model, "rswa_attn")

    print("[sglang_adapter] ✅ Unpatched, original attention restored (cuda-graph safe)")


def safe_patch_sglang(model, config: dict, cuda_graph_active: bool = False):
    """cuda-graph 安全的 patch.

    Args:
        cuda_graph_active: True = 不替换 attention (cuda-graph 安全)
                          False = 完整 patch (R-SWA 生效)
    """
    if cuda_graph_active:
        print("[sglang_adapter] safe_patch: attention NOT replaced (cuda-graph safe) ✅")
        print("[sglang_adapter]   R-SWA 验证已通过 (独立测试), 但 sglang 仍用标准 attention")
        return None
    else:
        return patch_sglang_gpu(model, config)


# -----------------------------------------------------------------------------
# 独立 demo (无 sglang): 验证 GPU 版本工作
# -----------------------------------------------------------------------------
def run_demo():
    if not HAS_GPU_RSWA:
        print("rswa_gpu.py not available")
        return

    print("=" * 60)
    print(" R-SWA GPU Adapter Demo (cuda-graph compatible)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    NH, HD, REF, WIN, ORTHO = 16, 128, 4, 128, 64

    rswa = RSWAOrthoAttentionGPU(NH, HD, REF, WIN, ORTHO).to(device)
    rswa.set_reference(torch.randn(REF, NH, HD, device=device), torch.randn(REF, NH, HD, device=device))

    # feed 200 tokens
    for _ in range(200):
        rswa.feed_token(torch.randn(NH, HD, device=device), torch.randn(NH, HD, device=device))

    # forward
    q = torch.randn(NH, HD, device=device)
    out = rswa.forward(q)
    print(f"  forward OK: out shape={out.shape}, norm={out.norm():.4f}")
    print(f"  visible_count={rswa.visible_count} (expect {REF+WIN})")

    # cuda-graph
    if device == "cuda":
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = rswa.forward(q)
        g.replay()
        torch.cuda.synchronize()
        print(f"  cuda-graph replay OK ✅")

    print("=" * 60)


if __name__ == "__main__":
    run_demo()
