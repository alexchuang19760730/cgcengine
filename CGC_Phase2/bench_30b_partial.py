"""用部分下载的 30B 4bit 模型测前 P=12 层 forward 速度.

只加载 model-00001 到 00004 + 00013 (前12层 + embed + lm_head),
不加载完整 48 层模型。

用法:
  python bench_30b_partial.py
"""
import time
import os
import json
from pathlib import Path

import mlx.core as mx
from safetensors import safe_open


MODEL_DIR = "/Users/alexchuang/models/Qwen3-VL-30B-A3B-4bit"
P = 12  # 测试前 12 层


def load_partial_weights():
    """加载前 P 层 + embed + lm_head 权重 (直接读 safetensors, 不用 index.json)."""
    print(f"[load] Loading first {P} layers + embed from {MODEL_DIR}")
    weights = {}

    safetensors_files = sorted(Path(MODEL_DIR).glob("model-*.safetensors"))
    # 排除 index.json
    safetensors_files = [f for f in safetensors_files if f.suffix == ".safetensors"]
    print(f"[load] Found {len(safetensors_files)} safetensors files: {[f.name for f in safetensors_files]}")

    for sf_path in safetensors_files:
        print(f"[load] Reading {sf_path.name}...")
        try:
            # 用 pt framework 读取 (mlx 不支持 bfloat16)
            with safe_open(str(sf_path), framework="pt") as f:
                keys = f.keys()
                for key in keys:
                    # 只加载前 P 层 + embed + lm_head + norm
                    should_load = False
                    if any(f"layers.{i}." in key for i in range(P)):
                        should_load = True
                    elif "embed_tokens" in key or "lm_head" in key:
                        should_load = True
                    elif "norm.weight" in key or "model.norm" in key:
                        should_load = True

                    if should_load:
                        tensor_pt = f.get_tensor(key)
                        # 转 mlx (bfloat16 → float16, mlx 支持)
                        import mlx.core as mx
                        import numpy as np
                        if tensor_pt.dtype == __import__('torch').bfloat16:
                            tensor_np = tensor_pt.float().numpy()
                        else:
                            tensor_np = tensor_pt.numpy()
                        weights[key] = mx.array(tensor_np)
        except Exception as e:
            print(f"[load] Error reading {sf_path.name}: {e}")
            continue

    print(f"[load] Loaded {len(weights)} weight tensors")

    # 检查 4bit 量化格式
    sample_key = list(weights.keys())[0] if weights else None
    if sample_key:
        sample = weights[sample_key]
        print(f"[load] Sample weight: {sample_key} shape={sample.shape} dtype={sample.dtype}")

    # 统计每层
    layer_counts = {}
    for key in weights:
        import re
        m = re.search(r"layers\.(\d+)\.", key)
        if m:
            i = int(m.group(1))
            layer_counts[i] = layer_counts.get(i, 0) + 1

    print(f"[load] Layers loaded: {sorted(layer_counts.keys())}")
    if layer_counts:
        print(f"[load] Tensors per layer: {layer_counts.get(0, 0)}")

    return weights


def benchmark_forward(weights, num_iterations=100):
    """测前 P 层 forward (1 token decode, seq=1)."""
    HIDDEN = 2048
    NUM_HEADS = 32
    HEAD_DIM = 128
    NUM_KV_HEADS = 4

    print(f"\n[bench] Forward benchmark: {P} layers, seq=1 (decode step)")

    # 准备输入 (1 token)
    h = mx.random.normal((1, 1, HIDDEN))

    # 测每层 forward 时间
    layer_times = []

    for layer_idx in range(P):
        prefix = f"model.language_model.layers.{layer_idx}"

        # 获取权重
        q_proj = weights.get(f"{prefix}.self_attn.q_proj.weight")
        k_proj = weights.get(f"{prefix}.self_attn.k_proj.weight")
        v_proj = weights.get(f"{prefix}.self_attn.v_proj.weight")
        o_proj = weights.get(f"{prefix}.self_attn.o_proj.weight")
        input_norm = weights.get(f"{prefix}.input_layernorm.weight")
        post_norm = weights.get(f"{prefix}.post_attention_layernorm.weight")
        gate = weights.get(f"{prefix}.mlp.gate.weight")
        gate_up = weights.get(f"{prefix}.mlp.switch_mlp.gate_proj.weight")
        up = weights.get(f"{prefix}.mlp.switch_mlp.up_proj.weight")
        down = weights.get(f"{prefix}.mlp.switch_mlp.down_proj.weight")

        if q_proj is None:
            print(f"  layer {layer_idx}: missing weights, skip")
            continue

        # 测 attention forward
        def attn_fwd(h_in):
            h_norm = h_in * mx.sqrt((h_in * h_in).mean(-1, keepdims=True) + 1e-6)
            h_norm = h_norm * input_norm  # simplified RMSNorm
            q = h_norm @ q_proj.T
            k = h_norm @ k_proj.T
            v = h_norm @ v_proj.T
            # GQA: repeat k/v
            k_full = mx.concatenate([k] * (NUM_HEADS // NUM_KV_HEADS), axis=-1)
            v_full = mx.concatenate([v] * (NUM_HEADS // NUM_KV_HEADS), axis=-1)
            # seq=1: attention output = v
            out = v_full @ o_proj.T
            return h_in + out

        # 测 MoE forward
        def moe_fwd(h_in):
            h_norm = h_in * mx.sqrt((h_in * h_in).mean(-1, keepdims=True) + 1e-6)
            h_norm = h_norm * post_norm
            # routing
            logits = h_norm.reshape(-1, HIDDEN) @ gate.T  # [1, 128]
            # top-8 (简化: 取前 8)
            weights_topk = mx.softmax(logits[:, :8], axis=-1)  # [1, 8]
            # 8 experts
            h_flat = h_norm.reshape(-1, HIDDEN)
            # gate_up: 8 experts
            gate_out = mx.swapaxes(h_flat[:, None] @ gate_up[:8].swapaxes(0, 1), 0, 1) if gate_up.ndim == 3 else h_flat @ gate_up[:8].T
            up_out = mx.swapaxes(h_flat[:, None] @ up[:8].swapaxes(0, 1), 0, 1) if up.ndim == 3 else h_flat @ up[:8].T
            act = gate_out * mx.sigmoid(gate_out) * up_out
            out = mx.swapaxes(act @ down[:8].swapaxes(0, 1), 0, 1) if down.ndim == 3 else act @ down[:8].T
            out = (out * weights_topk[:, :, None]).sum(axis=1)
            return h_in + out

        # warmup
        for _ in range(3):
            mx.eval(attn_fwd(h))
            mx.eval(moe_fwd(h))

        mx.clear_cache()

        # benchmark attention
        t0 = time.time()
        for _ in range(num_iterations):
            mx.eval(attn_fwd(h))
        attn_ms = (time.time() - t0) / num_iterations * 1000

        # benchmark MoE
        t0 = time.time()
        for _ in range(num_iterations):
            mx.eval(moe_fwd(h))
        moe_ms = (time.time() - t0) / num_iterations * 1000

        total_ms = attn_ms + moe_ms
        layer_times.append(total_ms)

        if layer_idx < 3 or layer_idx == P - 1:
            print(f"  layer {layer_idx:2d}: attn={attn_ms:.3f}ms moe={moe_ms:.3f}ms total={total_ms:.3f}ms")

    avg_per_layer = sum(layer_times) / len(layer_times) if layer_times else 0
    total_p = sum(layer_times)

    print(f"\n[bench] Average per layer: {avg_per_layer:.3f}ms")
    print(f"[bench] Total P={P} layers: {total_p:.1f}ms")

    return avg_per_layer, total_p


def analyze_layer_split(per_layer_ms):
    """分析不同配置的 layer-split."""
    TOTAL_LAYERS = 48
    RTT = 110  # ms (双向)
    CLOUD_PER_LAYER = 0.1  # ms (TP=8 GPU)
    VERIFY = 3  # ms

    print(f"\n{'='*60}")
    print(f"Layer-split 分析 (per layer: {per_layer_ms:.2f}ms, 4bit)")
    print(f"{'='*60}")

    configs = [
        ("无投机 (accept=0.28, N=21)", 0.28, 21),
        ("MTP head (accept=0.6, N=21)", 0.6, 21),
        ("MTP head (accept=0.6, N=50)", 0.6, 50),
    ]

    for name, accept_rate, N in configs:
        accept = N * accept_rate
        print(f"\n  {name}:")
        print(f"  {'P':>4} {'Mac_fw':>8} {'Total':>8} {'tok/s':>6} {'省cloud':>8}")
        print(f"  {'-'*40}")
        for p in [6, 12, 18, 22]:
            mac_fw = p * per_layer_ms
            cloud_fw = (TOTAL_LAYERS - p) * CLOUD_PER_LAYER
            total = mac_fw + RTT + cloud_fw + VERIFY + RTT
            tok_s = accept / total * 1000
            save = p / TOTAL_LAYERS * 100
            print(f"  {p:4d} {mac_fw:7.1f}ms {total:7.1f}ms {tok_s:5.1f} {save:6.0f}%")


def main():
    print("=" * 60)
    print("30B-A3B 4bit Partial Forward Benchmark (Mac M4)")
    print("=" * 60)

    # 检查文件
    if not os.path.exists(MODEL_DIR):
        print(f"ERROR: Model dir not found: {MODEL_DIR}")
        return

    weights = load_partial_weights()
    if len(weights) < 100:
        print(f"ERROR: Only {len(weights)} weights loaded, expected >100")
        return

    per_layer_ms, total_p = benchmark_forward(weights)
    analyze_layer_split(per_layer_ms)

    # 显存检查
    print(f"\n{'='*60}")
    print("显存使用")
    print(f"{'='*60}")
    try:
        import subprocess
        result = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=5)
        for line in result.stdout.split("\n"):
            if "free" in line.lower() or "used" in line.lower() or "wired" in line.lower():
                print(f"  {line.strip()}")
    except:
        pass


if __name__ == "__main__":
    main()
