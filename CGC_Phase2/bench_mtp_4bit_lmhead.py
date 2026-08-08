"""MTP head 4bit 量化 + Mac Metal forward benchmark.

lm_head 是瓶颈 (622MB bf16 → 15ms)。
4bit 量化: 622MB → 156MB, 预期 ~1.3ms。
"""
import time
import torch
import mlx.core as mx
import numpy as np


def quantize_4bit(weight_bf16: mx.array, group_size: int = 64) -> dict:
    """4bit 量化 weight [out, in] → {w, scales, biases}."""
    # weight: [out, in]
    out_dim, in_dim = weight_bf16.shape
    assert in_dim % group_size == 0, f"in_dim {in_dim} not divisible by group_size {group_size}"
    n_groups = in_dim // group_size

    # reshape [out, n_groups, group_size]
    w = weight_bf16.astype(mx.float32).reshape(out_dim, n_groups, group_size)

    # 每组找 max
    w_max = mx.max(mx.abs(w), axis=-1, keepdims=True)  # [out, n_groups, 1]
    w_max = mx.maximum(w_max, 1e-5)

    # 量化到 [-8, 7] (4bit signed)
    w_q = mx.round(w / w_max * 7).astype(mx.int32)  # [out, n_groups, group_size]
    # 转 uint32 packed (8 个 4bit 值 per uint32)
    # 重排: [out, n_groups, group_size/8, 8] → pack
    assert group_size % 8 == 0
    w_q = w_q.reshape(out_dim, n_groups, group_size // 8, 8)
    # pack 8 个 int4 → 1 uint32
    # 用 2^shifts (mlx 不支持 1 << array)
    shifts = mx.array([1, 16, 256, 4096, 65536, 1048576, 16777216, 268435456], dtype=mx.int32)
    w_packed = mx.sum(w_q * shifts, axis=-1)  # [out, n_groups, group_size/8]
    # flatten 到 [out, in/8]
    w_packed = w_packed.reshape(out_dim, -1).astype(mx.uint32)

    # scales = w_max / 7, shape [out, n_groups]
    scales = (w_max.squeeze(-1) / 7.0).astype(mx.float16)  # [out, n_groups]

    # biases = 0
    biases = mx.zeros_like(scales)

    return {"w": w_packed, "scales": scales, "biases": biases, "group_size": group_size}


def dequantize_4bit(q: dict) -> mx.array:
    """解量化回 bf16 (用于验证)."""
    w_packed = q["w"]  # [out, in/8] uint32
    scales = q["scales"]  # [out, n_groups] float16
    group_size = q["group_size"]

    out_dim = w_packed.shape[0]
    in_dim = w_packed.shape[1] * 8
    n_groups = in_dim // group_size

    # unpack: [out, n_groups, group_size/8, 8]
    shifts = mx.array([0, 4, 8, 12, 16, 20, 24, 28], dtype=mx.uint32)
    w_q = (w_packed[:, :, None] >> shifts[None, None, :]) & 0xF  # [out, in/8, 8]
    # 转 signed: >7 的减 16
    w_q = mx.where(w_q > 7, w_q - 16, w_q).astype(mx.float32)
    # reshape [out, n_groups, group_size]
    w_q = w_q.reshape(out_dim, n_groups, group_size)

    # dequantize: w = w_q * scales
    w = w_q * scales[:, :, None].astype(mx.float32)
    # reshape [out, in]
    return w.reshape(out_dim, in_dim).astype(mx.bfloat16)


def benchmark():
    print("[1] Loading MTP head checkpoint...")
    ckpt = torch.load("/tmp/mtp_head_final.pt", weights_only=False, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt)

    # 转换权重到 MLX
    mlx_w = {}
    for name, p in sd.items():
        if p.dtype == torch.bfloat16:
            mlx_w[name] = mx.array(p.float().numpy())
        else:
            mlx_w[name] = mx.array(p.numpy())

    lm_head_w = mlx_w["lm_head.weight"]  # [151936, 2048]
    print(f"    lm_head: {lm_head_w.shape}, {lm_head_w.dtype}, {lm_head_w.size * 2 / 1e6:.0f}MB")

    # 原始 (bf16) forward
    print("\n[2] Benchmark bf16 lm_head (baseline)...")
    hidden = mx.random.normal((1, 1, 2048)).astype(mx.bfloat16)

    def forward_bf16(h):
        return h @ lm_head_w.T  # [1, 1, 151936]

    for _ in range(5):
        mx.eval(forward_bf16(hidden))
    mx.clear_cache()
    t0 = time.time()
    for _ in range(100):
        mx.eval(forward_bf16(hidden))
    bf16_ms = (time.time() - t0) / 100 * 1000
    print(f"    bf16 lm_head: {bf16_ms:.2f}ms")

    # 4bit 量化
    print("\n[3] Quantizing lm_head to 4bit...")
    q = quantize_4bit(lm_head_w, group_size=64)
    print(f"    packed: {q['w'].shape}, scales: {q['scales'].shape}")
    print(f"    size: {q['w'].size * 4 / 1e6:.0f}MB + {q['scales'].size * 2 / 1e6:.0f}MB = {(q['w'].size * 4 + q['scales'].size * 2) / 1e6:.0f}MB")

    # 验证量化精度
    print("\n[4] Verify quantization accuracy...")
    w_deq = dequantize_4bit(q)
    # 采样几个值对比
    diff = mx.abs(lm_head_w.astype(mx.float32) - w_deq.astype(mx.float32))
    max_diff = float(mx.max(diff))
    mean_diff = float(mx.mean(diff))
    print(f"    max diff: {max_diff:.4f}, mean diff: {mean_diff:.4f}")

    # 4bit forward (用 quantized_matmul)
    print("\n[5] Benchmark 4bit lm_head...")
    q_w = q["w"]
    q_s = q["scales"].astype(mx.bfloat16)
    q_b = q["biases"].astype(mx.bfloat16)
    group_size = q["group_size"]

    def forward_4bit(h):
        # h: [1, 1, 2048] → [1, 2048] for quantized_matmul
        h_flat = h.reshape(1, -1).astype(mx.bfloat16)
        # quantized_matmul(x, w, scales, biases, group_size, bits)
        logits = mx.quantized_matmul(h_flat, q_w, scales=q_s, biases=q_b, group_size=group_size, bits=4)
        return logits.reshape(1, 1, -1)

    for _ in range(5):
        mx.eval(forward_4bit(hidden))
    mx.clear_cache()
    t0 = time.time()
    for _ in range(100):
        mx.eval(forward_4bit(hidden))
    q4_ms = (time.time() - t0) / 100 * 1000
    print(f"    4bit lm_head: {q4_ms:.2f}ms (vs bf16 {bf16_ms:.2f}ms, {bf16_ms/q4_ms:.1f}x faster)")

    # 完整 MTP forward (proj + MLP + 4bit lm_head)
    print("\n[6] Full MTP forward with 4bit lm_head...")
    proj_w = mlx_w["proj.weight"]
    gate_w = mlx_w["mlp.gate_proj.weight"]
    up_w = mlx_w["mlp.up_proj.weight"]
    down_w = mlx_w["mlp.down_proj.weight"]
    embed = mx.random.normal((1, 1, 2048)).astype(mx.bfloat16)

    def mtp_forward_4bit(h, e):
        x = mx.concatenate([h, e], axis=-1)
        x = x @ proj_w.T
        gate = x @ gate_w.T
        up = x @ up_w.T
        act = mx.sigmoid(gate) * gate * up
        out = act @ down_w.T
        # 4bit lm_head
        out_flat = out.reshape(1, -1).astype(mx.bfloat16)
        logits = mx.quantized_matmul(out_flat, q_w, scales=q_s, biases=q_b, group_size=group_size, bits=4)
        return logits.reshape(1, 1, -1)

    for _ in range(5):
        mx.eval(mtp_forward_4bit(hidden, embed))
    mx.clear_cache()
    t0 = time.time()
    for _ in range(200):
        mx.eval(mtp_forward_4bit(hidden, embed))
    full_q4_ms = (time.time() - t0) / 200 * 1000
    print(f"    Full MTP (4bit lm_head): {full_q4_ms:.2f}ms")

    # 对比 bf16 full
    def mtp_forward_bf16(h, e):
        x = mx.concatenate([h, e], axis=-1)
        x = x @ proj_w.T
        gate = x @ gate_w.T
        up = x @ up_w.T
        act = mx.sigmoid(gate) * gate * up
        out = act @ down_w.T
        logits = out @ lm_head_w.T
        return logits

    for _ in range(5):
        mx.eval(mtp_forward_bf16(hidden, embed))
    mx.clear_cache()
    t0 = time.time()
    for _ in range(200):
        mx.eval(mtp_forward_bf16(hidden, embed))
    full_bf16_ms = (time.time() - t0) / 200 * 1000
    print(f"    Full MTP (bf16 lm_head): {full_bf16_ms:.2f}ms")

    # 投机 decode 估算
    print("\n[7] Spec decode 估算 (accept=60.8%, baseline=26 tok/s, verify=38.5ms):")
    verify_ms = 38.5
    accept_rate = 0.608

    for label, draft_ms in [("bf16 (当前)", full_bf16_ms), ("4bit lm_head", full_q4_ms)]:
        print(f"\n  {label} (draft={draft_ms:.2f}ms):")
        for N in [2, 4, 10, 21]:
            total = N * draft_ms + verify_ms
            accept = N * accept_rate
            tps = accept / total * 1000
            boost = tps / 26
            status = "✓" if tps > 26 else "✗"
            print(f"    N={N:2d}: {tps:.0f} tok/s ({boost:.1f}x) {status}")

    print(f"\n[8] 优化效果:")
    print(f"  bf16 → 4bit: {full_bf16_ms:.2f}ms → {full_q4_ms:.2f}ms ({full_bf16_ms/full_q4_ms:.1f}x faster)")
    print(f"  N=4: {4*accept_rate/(4*full_bf16_ms+verify_ms)*1000:.0f} → {4*accept_rate/(4*full_q4_ms+verify_ms)*1000:.0f} tok/s")


if __name__ == "__main__":
    benchmark()
