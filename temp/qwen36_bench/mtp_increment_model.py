#!/usr/bin/env python3
"""Qwen3.6 MTP-enabled decode 预估 (v3 final).

关键确认 (官方 modeling_qwen3_5_moe.py):
  - DeltaNet 用 chunk_gated_delta_rule, chunk_size=64
  - chunk 内 = 矩阵化并行 (attn @ v_beta, 无逐位置循环)
  - chunk 间 = 串行 (for i in range(num_chunks))
  -> verify batch (1+d <= 4 位置) 全部落在单个 chunk 内 -> DeltaNet 可 batch!
  -> 乐观场景成立: verify 的 DeltaNet 成本 ≈ 单次 (不随 draft 数线性增长)

输入 (实测):
  - Qwen36 kernel: r3 step 86.6ms / r4 step 88.7ms @ P=1024
  - MoE amortized (batch-50): 0.59ms/tok vs current 1.145ms/tok
  - 接受率: code 88% / prose 54% (前会话实测)
"""
print("=" * 70)
print("Qwen3.6 MTP-enabled decode 预估 (final: DeltaNet chunk 并行已确认)")
print("=" * 70)

q36 = {
    "r3": {"step": 86.6, "moe": 45.7, "delta": 31.0, "gated": 9.9},
    "r4": {"step": 88.7, "moe": 45.8, "delta": 32.9, "gated": 10.0},
}
accept_code, accept_prose, accept_mix = 0.88, 0.54, 0.70

def expected_tokens(a, d):
    s = 1.0
    for i in range(1, d + 1):
        s += a ** i
    return s

print("\n--- 期望产出/步 (draft=3) ---")
for label, a in [("code", 0.88), ("prose", 0.54), ("mix", 0.70)]:
    print(f"  {label} (a={a}): E[token/步] = {expected_tokens(a, 3):.2f}")

print("\n--- verify 成本结构 (每步, 处理 1+d 位置, DeltaNet chunk 并行) ---")
print("  MoE(batch 摊薄 0.65x) + GatedAttn(1.1x) + DeltaNet(1 次, chunk 内并行) + draft head(1ms)")

def verify_cost(cfg, a, d):
    e = expected_tokens(a, d)
    moe_v = cfg["moe"] * 0.65
    gat_v = cfg["gated"] * 1.1
    delta_v = cfg["delta"]  # chunk 并行: 1 次
    draft = 1.0
    v = moe_v + gat_v + delta_v + draft
    return v, e, v / e

print("\n--- decode 预估 (端到端 = GPU x 1.30 overhead) ---")
for bit, cfg in q36.items():
    base_ms = cfg["step"] * 1.30
    base_tok = 1000 / base_ms
    print(f"[{bit}] MTP-off: {base_tok:.1f} tok/s")
    for label, a in [("code", 0.88), ("prose", 0.54), ("mix", 0.70)]:
        v, e, per_tok = verify_cost(cfg, a, 3)
        e2e = per_tok * 1.30
        tok = 1000 / e2e
        print(f"  {label:5s} MTP-on: verify {v:.0f}ms/{e:.2f}tok -> {tok:.1f} tok/s ({(tok/base_tok-1)*100:+.0f}%)")

print("\n--- draft 数敏感性 (mix 0.70, r4) ---")
cfg = q36["r4"]
for d in [1, 2, 3, 4, 5]:
    a = accept_mix
    v, e, per_tok = verify_cost(cfg, a, d)
    e2e = per_tok * 1.30
    print(f"  draft={d}: E[token]={e:.2f}/步, verify={v:.0f}ms -> {1000/e2e:.1f} tok/s")

print("\n--- 最终结论 ---")
print("Qwen36 fused MTP (DeltaNet chunk 并行确认):")
print("  - mix 负载: MTP-off 8.7-8.9 -> MTP-on 25-27 tok/s (+200%!)")
print("  - code: +300%, prose: +135%")
print("  - 对比 Gemma4 MTP (负收益) 的根本差异: fused 无二次 MoE + DeltaNet chunk 并行")
print("  - 注意: 这是理论上限, 实际受 CPU 调度/内存带宽/实现完整度折损")
print("  - 实际落地预估: r3 15-20 tok/s / r4 14-19 tok/s (含实现折损 40-50%)")
