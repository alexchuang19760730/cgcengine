"""30B 4bit 真实权重 forward benchmark (使用 MLX quantized_matmul)."""
import time
import os
from pathlib import Path

import mlx.core as mx
from safetensors import safe_open
import torch
import numpy as np

MODEL_DIR = "/Users/alexchuang/models/Qwen3-VL-30B-A3B-4bit"
P = 12
HIDDEN = 2048
NUM_HEADS = 32
HEAD_DIM = 128
NUM_KV_HEADS = 4
GROUP_SIZE = 64
BITS = 4


def load_4bit_weights():
    """加载前 P 层 4bit 权重."""
    print(f"[load] Loading first {P} layers (4bit)")
    weights = {}

    sf_path = Path(MODEL_DIR) / "model-00001-of-00004.safetensors"
    with safe_open(str(sf_path), framework="pt") as f:
        for key in f.keys():
            if any(f"layers.{i}." in key for i in range(P)):
                tensor_pt = f.get_tensor(key)
                if tensor_pt.dtype == torch.bfloat16:
                    arr = mx.array(tensor_pt.float().numpy())
                else:
                    arr = mx.array(tensor_pt.numpy())
                weights[key] = arr

    print(f"[load] Loaded {len(weights)} tensors")
    return weights


def to_mx(tensor_pt):
    """torch tensor → mlx array."""
    if tensor_pt.dtype == torch.bfloat16:
        return mx.array(tensor_pt.float().numpy())
    return mx.array(tensor_pt.numpy())


def benchmark_layer_4bit(weights, layer_idx, num_iter=50):
    """测单层 4bit forward."""
    prefix = f"language_model.model.layers.{layer_idx}"

    # 获取 4bit 权重
    q_w = weights[f"{prefix}.self_attn.q_proj.weight"]
    q_s = weights[f"{prefix}.self_attn.q_proj.scales"]
    q_b = weights[f"{prefix}.self_attn.q_proj.biases"]

    k_w = weights[f"{prefix}.self_attn.k_proj.weight"]
    k_s = weights[f"{prefix}.self_attn.k_proj.scales"]
    k_b = weights[f"{prefix}.self_attn.k_proj.biases"]

    v_w = weights[f"{prefix}.self_attn.v_proj.weight"]
    v_s = weights[f"{prefix}.self_attn.v_proj.scales"]
    v_b = weights[f"{prefix}.self_attn.v_proj.biases"]

    o_w = weights[f"{prefix}.self_attn.o_proj.weight"]
    o_s = weights[f"{prefix}.self_attn.o_proj.scales"]
    o_b = weights[f"{prefix}.self_attn.o_proj.biases"]

    # MoE 权重
    gate_w = weights[f"{prefix}.mlp.gate.weight"]
    gate_s = weights[f"{prefix}.mlp.gate.scales"]
    gate_b = weights[f"{prefix}.mlp.gate.biases"]

    gu_w = weights[f"{prefix}.mlp.switch_mlp.gate_proj.weight"]
    gu_s = weights[f"{prefix}.mlp.switch_mlp.gate_proj.scales"]
    gu_b = weights[f"{prefix}.mlp.switch_mlp.gate_proj.biases"]

    up_w = weights[f"{prefix}.mlp.switch_mlp.up_proj.weight"]
    up_s = weights[f"{prefix}.mlp.switch_mlp.up_proj.scales"]
    up_b = weights[f"{prefix}.mlp.switch_mlp.up_proj.biases"]

    down_w = weights[f"{prefix}.mlp.switch_mlp.down_proj.weight"]
    down_s = weights[f"{prefix}.mlp.switch_mlp.down_proj.scales"]
    down_b = weights[f"{prefix}.mlp.switch_mlp.down_proj.biases"]

    # 输入 (1 token decode)
    h = mx.random.normal((1, 1, HIDDEN))

    def attn_fwd(h_in):
        h_flat = h_in.reshape(-1, HIDDEN)  # [1, 2048]
        # quantized matmul
        q = mx.quantized_matmul(h_flat, q_w, scales=q_s, biases=q_b, group_size=GROUP_SIZE, bits=BITS)
        k = mx.quantized_matmul(h_flat, k_w, scales=k_s, biases=k_b, group_size=GROUP_SIZE, bits=BITS)
        v = mx.quantized_matmul(h_flat, v_w, scales=v_s, biases=v_b, group_size=GROUP_SIZE, bits=BITS)
        # GQA repeat
        v_full = mx.concatenate([v] * (NUM_HEADS // NUM_KV_HEADS), axis=-1)  # [1, 4096]
        out = mx.quantized_matmul(v_full, o_w, scales=o_s, biases=o_b, group_size=GROUP_SIZE, bits=BITS)
        return out.reshape(1, 1, HIDDEN)

    def moe_fwd(h_in):
        h_flat = h_in.reshape(-1, HIDDEN)
        # routing
        logits = mx.quantized_matmul(h_flat, gate_w, scales=gate_s, biases=gate_b, group_size=GROUP_SIZE, bits=BITS)
        weights_topk = mx.softmax(logits[:, :8], axis=-1)  # [1, 8]

        # 8 experts (只测前 8 个)
        # gate_proj + up_proj (每个 expert)
        # weight shape: (128, 768, 256) uint32 → 每 expert (768, 256)
        # 取前 8 experts
        gu_w8 = gu_w[:8]  # [8, 768, 256]
        gu_s8 = gu_s[:8]  # [8, 768, 32]
        gu_b8 = gu_b[:8]

        up_w8 = up_w[:8]
        up_s8 = up_s[:8]
        up_b8 = up_b[:8]

        down_w8 = down_w[:8]
        down_s8 = down_s[:8]
        down_b8 = down_b[:8]

        # batch quantized matmul: 8 experts
        # h_flat [1, 2048] → expand [8, 1, 2048] (repeat for 8 experts)
        h_batch = mx.broadcast_to(h_flat, (8, 1, HIDDEN))  # [8, 1, 2048]

        # per-expert forward (loop, MLX 不支持 batch quantized_matmul)
        expert_outs = []
        for i in range(8):
            gp = mx.quantized_matmul(h_flat, gu_w8[i], scales=gu_s8[i], biases=gu_b8[i], group_size=GROUP_SIZE, bits=BITS)
            up = mx.quantized_matmul(h_flat, up_w8[i], scales=up_s8[i], biases=up_b8[i], group_size=GROUP_SIZE, bits=BITS)
            act = mx.sigmoid(gp) * up  # SiLU
            out = mx.quantized_matmul(act, down_w8[i], scales=down_s8[i], biases=down_b8[i], group_size=GROUP_SIZE, bits=BITS)
            expert_outs.append(out)

        # stack + weighted sum
        out = mx.stack(expert_outs, axis=0)  # [8, 1, 2048]
        out = (out * weights_topk[:, :, None]).sum(axis=0)  # [1, 2048]
        return out.reshape(1, 1, HIDDEN)

    # warmup
    for _ in range(3):
        mx.eval(attn_fwd(h))
        mx.eval(moe_fwd(h))
    mx.clear_cache()

    # benchmark attention
    t0 = time.time()
    for _ in range(num_iter):
        mx.eval(attn_fwd(h))
    attn_ms = (time.time() - t0) / num_iter * 1000

    # benchmark MoE
    t0 = time.time()
    for _ in range(num_iter):
        mx.eval(moe_fwd(h))
    moe_ms = (time.time() - t0) / num_iter * 1000

    total = attn_ms + moe_ms
    return attn_ms, moe_ms, total


def main():
    print("=" * 60)
    print("30B-A3B 4bit Real Weights Forward Benchmark (Mac M4)")
    print("=" * 60)

    weights = load_4bit_weights()
    if len(weights) < 100:
        print(f"ERROR: Only {len(weights)} weights")
        return

    print(f"\n[bench] Testing {P} layers (4bit, quantized_matmul)")
    print(f"{'Layer':>4} {'Attn':>8} {'MoE':>8} {'Total':>8}")
    print("-" * 35)

    layer_times = []
    for i in range(min(P, 12)):
        try:
            attn, moe, total = benchmark_layer_4bit(weights, i, num_iter=30)
            layer_times.append(total)
            print(f"{i:4d} {attn:7.2f}ms {moe:7.2f}ms {total:7.2f}ms")
        except Exception as e:
            print(f"{i:4d} ERROR: {e}")

    if layer_times:
        avg = sum(layer_times) / len(layer_times)
        print(f"\n{'='*60}")
        print(f"Average per layer (4bit): {avg:.2f}ms")
        print(f"Total P={P}: {avg * P:.1f}ms")
        print(f"{'='*60}")

        # 对比 bf16
        print(f"\n对比:")
        print(f"  bf16 (micro-benchmark): 10.2ms/层")
        print(f"  4bit (实测):            {avg:.2f}ms/层")
        print(f"  加速比:                 {10.2/avg:.2f}x")

        # Layer-split 分析
        RTT = 110; cloud_per_layer = 0.1; verify = 3
        print(f"\n{'='*60}")
        print(f"Layer-split 分析 (4bit, {avg:.2f}ms/层)")
        print(f"{'='*60}")

        for name, acc_rate, N in [("无投机 0.28", 0.28, 21), ("MTP 0.6", 0.6, 21), ("MTP 0.6 N=50", 0.6, 50)]:
            accept = N * acc_rate
            print(f"\n  {name}:")
            for p in [6, 12, 18, 22]:
                mac_fw = p * avg
                cloud = (48 - p) * cloud_per_layer
                batch = mac_fw + RTT + cloud + verify + RTT
                tok_s = accept / batch * 1000
                save = p / 48 * 100
                print(f"    P={p:2d}: {mac_fw:6.1f}ms → {tok_s:5.1f} tok/s, 省{save:.0f}%")


if __name__ == "__main__":
    main()
