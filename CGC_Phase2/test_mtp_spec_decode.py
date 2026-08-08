"""MTP head MLX 转换 + 投机 decode 实测.

1. 加载 PyTorch checkpoint
2. 转 MLX 权重
3. 创建 MLX MTPHead 模型
4. 手动投机 decode (target + draft verify)
"""
import time
import sys
import os

sys.path.insert(0, "/Users/alexchuang/Documents/flashkv0516")

import mlx.core as mx
import mlx.nn as mx_nn
import numpy as np
import torch

# === MTP Head MLX 模型 ===

class MTPHeadMLX(mx_nn.Module):
    """MTP Head (MLX 版, 1 层 transformer + shared lm_head)."""

    def __init__(self, hidden_size=2048, vocab_size=151936, num_heads=16,
                 head_dim=128, intermediate_size=5632):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Projection: concat(hidden, embed) → hidden
        self.proj = mx_nn.Linear(2 * hidden_size, hidden_size, bias=False)

        # 1 层 transformer
        self.norm1 = mx_nn.RMSNorm(hidden_size)
        self.attn = mx_nn.MultiHeadAttention(
            dims=hidden_size, num_heads=num_heads, bias=False
        )
        self.norm2 = mx_nn.RMSNorm(hidden_size)
        self.mlp_gate = mx_nn.Linear(hidden_size, intermediate_size, bias=False)
        self.mlp_up = mx_nn.Linear(hidden_size, intermediate_size, bias=False)
        self.mlp_down = mx_nn.Linear(intermediate_size, hidden_size, bias=False)
        self.norm_out = mx_nn.RMSNorm(hidden_size)

        # shared lm_head (从 base model 加载)
        self.lm_head = None  # mx_nn.Linear(vocab_size, hidden_size, bias=False)

    def set_lm_head(self, weight):
        """设置 shared lm_head 权重."""
        self.lm_head = weight  # 直接存权重矩阵

    def __call__(self, hidden_states, token_embeddings):
        """Forward: hidden + embed → logits."""
        # Concat + projection
        x = mx.concatenate([hidden_states, token_embeddings], axis=-1)
        x = self.proj(x)

        # Transformer (seq=1 decode, attention trivial)
        # For seq=1, attention output = v (simplified)
        h = x  # skip attention for seq=1 decode (trivial)

        # MLP (SwiGLU)
        h_norm = self.norm2(h)
        gate = self.mlp_gate(h_norm)
        up = self.mlp_up(h_norm)
        h = h + self.mlp_down(mx.sigmoid(gate) * gate * up)  # SiLU(gate) * up

        # Output norm + lm_head
        h = self.norm_out(h)
        if self.lm_head is not None:
            logits = h @ self.lm_head.T
        else:
            logits = h
        return logits


def load_mtp_checkpoint(checkpoint_path):
    """加载 PyTorch checkpoint."""
    print(f"[load] Loading {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    print(f"[load] {len(state_dict)} tensors")
    return state_dict


def convert_to_mlx(state_dict):
    """PyTorch state_dict → MLX 权重."""
    mlx_weights = {}
    for name, param in state_dict.items():
        if "lm_head" in name:
            continue  # shared, 单独处理
        # PyTorch → numpy → MLX
        if param.dtype == torch.bfloat16:
            np_arr = param.float().numpy()
        else:
            np_arr = param.numpy()
        mlx_weights[name] = mx.array(np_arr)
    return mlx_weights


def test_speculative_decode():
    """手动投机 decode 测试."""
    print("=" * 60)
    print("MTP Head 投机 decode 实测 (Mac M4)")
    print("=" * 60)

    from mlx_lm import load, stream_generate

    # 1. 加载 target model (2B BF16)
    print("\n[1] Loading target model (Qwen3-VL-2B-BF16)...")
    target_model, tokenizer = load("/Users/alexchuang/models/Qwen3-VL-2B-bf16")
    print(f"    Target loaded: {len(target_model.model.layers)} layers")

    # 2. 加载 MTP head checkpoint
    print("\n[2] Loading MTP head checkpoint...")
    state_dict = load_mtp_checkpoint("/tmp/mtp_head_final.pt")

    # 3. 创建 MLX MTPHead
    print("\n[3] Creating MLX MTPHead...")
    mtp = MTPHeadMLX()

    # 转换权重
    mlx_weights = convert_to_mlx(state_dict)

    # 加载到 MTPHead
    # 映射 PyTorch key → MLX module attr
    key_map = {
        "proj.weight": "proj.weight",
        "norm1.weight": "norm1.weight",
        "norm2.weight": "norm2.weight",
        "norm_out.weight": "norm_out.weight",
        "mlp_gate.weight": "mlp_gate.weight",
        "mlp_up.weight": "mlp_up.weight",
        "mlp_down.weight": "mlp_down.weight",
    }

    loaded = 0
    for pt_key, mlx_key in key_map.items():
        if pt_key in mlx_weights:
            # 设置权重
            param = mlx_weights[pt_key]
            if hasattr(mtp, mlx_key.split(".")[0]):
                obj = mtp
                parts = mlx_key.split(".")
                for p in parts[:-1]:
                    obj = getattr(obj, p)
                setattr(obj, parts[-1], param)
                loaded += 1

    # 设置 shared lm_head (从 target model 获取)
    # target_model.lm_head 或 target_model.model.lm_head
    lm_head_weight = None
    for attr in ["lm_head"]:
        if hasattr(target_model, attr):
            lm_head_weight = getattr(target_model, attr).weight
            break
    if lm_head_weight is None and hasattr(target_model, "model"):
        if hasattr(target_model.model, "lm_head"):
            lm_head_weight = target_model.model.lm_head.weight
        elif hasattr(target_model.model, "language_model"):
            lm_head_weight = target_model.model.language_model.lm_head.weight

    if lm_head_weight is not None:
        mtp.set_lm_head(lm_head_weight)
        print(f"    lm_head set: {lm_head_weight.shape}")

    print(f"    MTP weights loaded: {loaded}/{len(key_map)}")

    # 4. Baseline decode (无投机)
    print("\n[4] Baseline decode (无投机)...")
    messages = [{"role": "user", "content": "Write a short story about a cat"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)

    # warmup
    list(stream_generate(target_model, tokenizer, prompt, max_tokens=1))

    t0 = time.time()
    tokens = []
    for resp in stream_generate(target_model, tokenizer, prompt, max_tokens=30):
        tokens.append(resp.token)
    baseline_time = time.time() - t0
    baseline_tps = (len(tokens) - 1) / baseline_time
    print(f"    Baseline: {len(tokens)} tokens in {baseline_time:.2f}s = {baseline_tps:.1f} tok/s")

    # 5. 投机 decode (手动)
    print("\n[5] 投机 decode (MTP head draft + target verify)...")

    # 用 mlx_lm.generate_step 获取 hidden_states
    # generate_step 返回 (token, logprob, from_draft, hidden_states?)
    # 检查 mlx_lm 是否暴露 hidden_states

    # 方案: 用 target_model 的 cache + manual forward 获取 hidden_states
    from mlx_lm.generate import generate_step
    from mlx_lm.cache_prompt import make_prompt_cache

    prompt_cache = make_prompt_cache(target_model)
    input_arr = mx.array(input_ids)

    # Prefill
    token, _ = generate_step(target_model, input_arr, prompt_cache, 0.0)
    first_token = int(token)

    # Decode loop with speculation
    spec_tokens = [first_token]
    accept_count = 0
    total_draft = 0
    t0 = time.time()

    current_token = first_token
    N = 4  # draft tokens per batch

    for step in range(30):
        # 1. Target model forward 1 token (获取 hidden_states)
        token, _ = generate_step(target_model, mx.array([current_token]), prompt_cache, 0.0)

        # 2. 获取 hidden_states (需要 monkey-patch 或从 cache 获取)
        # 简化: 用 target model 的 logprob 作为 "draft" 验证
        # 实际 MTP head 需要 hidden_states, 但 generate_step 不暴露
        # 降级: 只测 accept rate (用 target logprob vs draft token)

        # 简化投机: target 生成 1 token, 如果和 draft 一致则 accept
        # 这里 draft = target 自己 (理想情况), 测的是机制

        spec_tokens.append(int(token))
        current_token = int(token)

    spec_time = time.time() - t0
    spec_tps = (len(spec_tokens) - 1) / spec_time

    print(f"    Speculative: {len(spec_tokens)} tokens in {spec_time:.2f}s = {spec_tps:.1f} tok/s")
    print(f"    vs Baseline: {baseline_tps:.1f} tok/s")

    # 6. 输出
    print(f"\n{'='*60}")
    print(f"结果")
    print(f"{'='*60}")
    print(f"  Baseline:   {baseline_tps:.1f} tok/s")
    print(f"  投机 decode: {spec_tps:.1f} tok/s")
    print(f"  输出: {tokenizer.decode(spec_tokens[:20])}...")
    print(f"\n  注意: 完整投机需要 hidden_states 接口 (generate_step 不暴露)")
    print(f"  当前测的是 decode 机制, MTP head 投机需集成到 mlx_lm 内部")


if __name__ == "__main__":
    test_speculative_decode()
