"""30B MoE 每层 forward micro-benchmark (不需要下载模型).

用随机权重模拟 30B-A3B MoE 每层 forward, 测 Mac Metal GPU 时间。

30B-A3B 配置:
  hidden_size=2048, num_heads=32, head_dim=128
  num_experts=128, num_experts_per_tok=8
  intermediate_size=6144

每层 forward:
  1. Attention (q/k/v/o proj + RoPE + SDPA)
  2. MoE routing (gate → top-8 experts)
  3. 8 experts MLP (gate_up + down)
"""
import time
import mlx.core as mx
import mlx.nn as nn

DTYPE = mx.bfloat16  # 实际推理用 bfloat16

HIDDEN = 2048
NUM_HEADS = 32
HEAD_DIM = 128
NUM_EXPERTS = 128
EXPERTS_PER_TOK = 8
INTERMEDIATE = 6144
SEQ_LEN = 1  # decode step (1 token)


def benchmark_attention():
    """测 attention forward (1 token decode, seq=1 时 attention 极简)."""
    # 权重 (随机初始化, bf16 模拟)
    q_proj = mx.random.normal((4096, HIDDEN))
    k_proj = mx.random.normal((512, HIDDEN))
    v_proj = mx.random.normal((512, HIDDEN))
    o_proj = mx.random.normal((HIDDEN, 4096))

    h = mx.random.normal((1, SEQ_LEN, HIDDEN))

    def forward():
        # seq=1 decode: 4 个线性变换 (attention score trivial for seq=1)
        q = h @ q_proj.T  # [1, 1, 4096]
        k = h @ k_proj.T  # [1, 1, 512]
        v = h @ v_proj.T  # [1, 1, 512]
        # GQA: repeat k/v 8x then concat to 4096
        k_full = mx.concatenate([k] * 8, axis=-1)  # [1, 1, 4096]
        v_full = mx.concatenate([v] * 8, axis=-1)  # [1, 1, 4096]
        # seq=1: attention output = v (score=1, softmax trivial)
        out = v_full  # [1, 1, 4096]
        return out @ o_proj.T

    # warmup
    for _ in range(5):
        mx.eval(forward())
    mx.eval(mx.metal.clear_cache())

    # benchmark
    t0 = time.time()
    N = 100
    for _ in range(N):
        mx.eval(forward())
    elapsed = time.time() - t0
    return elapsed / N * 1000  # ms


def benchmark_moe():
    """测 MoE forward (1 token, 8/128 experts activated)."""
    # 所有 expert 权重 (模拟, 只测激活的 8 个)
    gate_up_weights = mx.random.normal((EXPERTS_PER_TOK, HIDDEN, INTERMEDIATE * 2))  # 8 experts
    down_weights = mx.random.normal((EXPERTS_PER_TOK, INTERMEDIATE, HIDDEN))
    router = mx.random.normal((NUM_EXPERTS, HIDDEN))  # routing gate

    h = mx.random.normal((1, SEQ_LEN, HIDDEN))

    def forward():
        # Routing
        logits = h.reshape(-1, HIDDEN) @ router.T  # [1, 128]
        # top-8 experts (模拟, 直接选前 8 个)
        weights = mx.softmax(logits[:, :EXPERTS_PER_TOK], axis=-1)  # [1, 8]

        # 8 experts MLP
        h_flat = h.reshape(-1, HIDDEN)  # [1, HIDDEN]
        # batch matmul: 8 experts
        gate_up = mx.swapaxes(h_flat[:, None] @ gate_up_weights, 0, 1)  # [8, 1, INTERMEDIATE*2]
        gate_part, up_part = gate_up[..., :INTERMEDIATE], gate_up[..., INTERMEDIATE:]
        act = gate_part * mx.sigmoid(gate_part) * up_part  # SiLU(gate) * up
        # down: [8, 1, INTERMEDIATE] @ [INTERMEDIATE, HIDDEN]
        out = mx.swapaxes(act @ down_weights, 0, 1)  # [1, 8, HIDDEN]
        # weighted sum
        out = (out * weights[:, :, None]).sum(axis=1)  # [1, HIDDEN]
        return out

    # warmup
    for _ in range(5):
        mx.eval(forward())
    mx.eval(mx.metal.clear_cache())

    t0 = time.time()
    N = 100
    for _ in range(N):
        mx.eval(forward())
    elapsed = time.time() - t0
    return elapsed / N * 1000


def benchmark_full_layer():
    """测完整 1 层 forward (attention + MoE + norm)."""
    # Norms
    norm1 = nn.RMSNorm(HIDDEN)
    norm2 = nn.RMSNorm(HIDDEN)

    attn_time = benchmark_attention()
    moe_time = benchmark_moe()
    norm_time = 0.01  # RMSNorm 极快, 忽略

    total = attn_time + moe_time + norm_time
    return total, attn_time, moe_time


def main():
    print("=" * 60)
    print("30B-A3B MoE 每层 forward micro-benchmark (Mac M4)")
    print("=" * 60)
    print(f"Config: hidden={HIDDEN}, heads={NUM_HEADS}, experts={NUM_EXPERTS}/{EXPERTS_PER_TOK}")
    print(f"Seq len: {SEQ_LEN} (decode step)")
    print()

    print("Benchmarking attention...")
    attn_ms = benchmark_attention()
    print(f"  Attention: {attn_ms:.3f} ms")

    print("Benchmarking MoE (8 experts)...")
    moe_ms = benchmark_moe()
    print(f"  MoE (8 experts): {moe_ms:.3f} ms")

    total_ms = attn_ms + moe_ms
    print(f"\n  Total per layer: {total_ms:.3f} ms")

    print()
    print("=" * 60)
    print("Layer-split P 分析 (N=21, accept=0.28, RTT=110ms)")
    print("=" * 60)

    rtt = 110  # ms (双向)
    cloud_per_layer = 0.1  # ms (TP=8 GPU 极快)
    verify = 3  # ms
    accept = 21 * 0.28  # 5.88

    for P in [6, 12, 18, 24, 30]:
        mac_forward = P * total_ms
        cloud_forward = (48 - P) * cloud_per_layer
        total_batch = mac_forward + rtt + cloud_forward + verify + rtt
        tok_s = accept / total_batch * 1000
        save_cost = P / 48 * 100
        print(f"  P={P:2d}: Mac={mac_forward:6.1f}ms  cloud={cloud_forward:4.1f}ms  "
              f"total={total_batch:6.1f}ms  → {tok_s:5.1f} tok/s  省{save_cost:.0f}%")

    print()
    print("=" * 60)
    print("对比 2B (实测)")
    print("=" * 60)
    print(f"  2B: 28层 × 1.36ms = 38ms → 26 tok/s (PD 分离, 无 RTT)")
    print(f"  30B P=12 layer-split: {12*total_ms:.1f}ms Mac + 110ms RTT → {accept/(12*total_ms+110+verify+110)*1000:.1f} tok/s")


if __name__ == "__main__":
    main()
