#!/usr/bin/env python3
"""Gemma4 vs Qwen36 trust-receipt 收益对比（为什么 Qwen36 省更多）。

核心概念：
  - "收益" = 跳过哈希省下的等待时间 = 哈希原本要花的时间
  - 哈希量越大 -> 原本越慢 -> 跳过省得越多
  - 不是 Qwen36 更快，而是 Qwen36 原本哈希更久，省下更多
"""
print("=" * 70)
print("trust-receipt 收益对比: Gemma4 vs Qwen36")
print("=" * 70)

# ---- 哈希总量（实测 manifest）----
gemma4 = {
    "model_weights": 1.35,   # GB (实测)
    "layers": 30,
    "layer_gb": (429916160 * 30) / 1e9,   # layer 文件 ~430MB x 30
    "total": 14.3,           # GB (实测 manifest 全量)
    "benefit_s": 3.9,        # 文档实测 trust-receipt 省的时间
}
qwen36 = {
    "r3": {"model_weights": 4.9, "layers": 40, "layer_mb": 336, "layer_gb": 336*40/1024, "total": 18.0},
    "r4": {"model_weights": 4.9, "layers": 40, "layer_mb": 432, "layer_gb": 432*40/1024, "total": 21.8},
}

print()
print("--- 哈希总量 ---")
print(f"Gemma4: model_weights {gemma4['model_weights']}GB + 30层 {gemma4['layer_gb']:.1f}GB = {gemma4['total']}GB")
for k, v in qwen36.items():
    print(f"Qwen36 {k}: model_weights {v['model_weights']}GB + 40层 {v['layer_gb']:.1f}GB = {v['total']:.1f}GB")

# ---- 吞吐校准 ----
# Gemma4 实测: 14.3GB 全量哈希 -> trust-receipt 省 3.9s
# 但 3.9s 是 TTFT 内的收益（eager model_weights + 首token touch 的层）
# 实际吞吐: model_weights 1.35GB eager（加载时哈希）
print()
print("--- 吞吐校准 (用 Gemma4 实测) ---")
# Gemma4: eager 1.35GB + 首 token touch 30 层 (约 13GB lazy) = 14.3GB, 但 3.9s 省的是这部分
# 简化: 全量 14.3GB / 3.9s = 3.67 GB/s（这是 Gemma4 实测的综合哈希吞吐，含磁盘读）
throughput = gemma4["total"] / gemma4["benefit_s"]
print(f"Gemma4 综合哈希吞吐 = {gemma4['total']}GB / {gemma4['benefit_s']}s = {throughput:.2f} GB/s")

print()
print("--- Qwen36 预估收益（同吞吐）---")
for k, v in qwen36.items():
    est = v["total"] / throughput
    print(f"Qwen36 {k}: {v['total']:.1f}GB / {throughput:.2f}GB/s = 预估省 {est:.1f}s (Gemma4 是 3.9s)")

print()
print("--- 为什么 Qwen36 收益更大 ---")
print("1. 哈希量: Qwen36 18-22GB vs Gemma4 14.3GB → 原本要哈希更久")
print("2. '收益' = 跳过哈希省下的时间 → 原本越久, 省得越多")
print("3. Qwen36 层数 40 vs 30, 每层更大 (r4 432MB vs Gemma4 430MB), weights 4.9GB vs 1.35GB")
print("4. 相对收益 (%): Qwen36 省 26-30% of TTFT vs Gemma4 省 3.9s/8.05s = 48%")
print("   注意: 相对 % 是 Gemma4 更高（因为 Qwen36 其他环节更慢）")
print("   但绝对秒数: Qwen36 省 5-8s vs Gemma4 省 3.9s → Qwen36 绝对收益更大")
print()
print("=== 一句话 ===")
print("'收益更大' = 绝对秒数省更多 (5-8s vs 3.9s), 因为 Qwen36 原本哈希更耗时 (18-22GB vs 14.3GB)。")
print("不是 Qwen36 更快, 而是它原本更慢, 跳过哈希的边际收益更高。")
