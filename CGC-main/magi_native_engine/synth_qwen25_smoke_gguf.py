#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
synth_qwen25_smoke_gguf.py
合成「最小 Qwen2.5 结构 GGUF」用于验证 magi_infer 真实加载路径
（GGUF 打开 → tensor 扫描 → 反量化 → Metal 上传），补齐 charter 验收第 5 项。

结构对齐 main.cpp 的 ModelConfig：n_layer=28, dim=3584。
tensor 命名对齐 metal_load_gguf 扫描清单：
  token_embd.weight, blk.{l}.{attn_norm,attn_q,attn_k,attn_v,attn_output,
  ffn_norm,ffn_up,ffn_gate,ffn_down}.weight, final_norm.weight, output.weight
全部 F16（metal_load_gguf 原生支持分支），shape 缩小以控制文件体积
（加载路径验证不跑推理，shape 与 cfg 不一致不影响路径覆盖）。

用法: /opt/homebrew/bin/python3.13 synth_qwen25_smoke_gguf.py <out.gguf> [layers]
"""
import sys
import numpy as np
from gguf import GGUFWriter

def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "synthetic-qwen2.5-28L-smoke.gguf"
    n_layer = int(sys.argv[2]) if len(sys.argv) > 2 else 28
    dim = 3584          # 对齐 main.cpp ModelConfig.dim
    small = 64          # 缩小 shape（不跑推理，仅覆盖加载路径）

    w = GGUFWriter(out, "qwen2.5")
    rng = np.random.default_rng(42)

    def add(name, shape):
        a = rng.standard_normal(shape).astype(np.float16)
        w.add_tensor(name, a, raw_shape=shape)

    add("token_embd.weight", (small, dim))
    for l in range(n_layer):
        add(f"blk.{l}.attn_norm.weight", (dim,))
        add(f"blk.{l}.attn_q.weight", (small, dim))
        add(f"blk.{l}.attn_k.weight", (small, dim))
        add(f"blk.{l}.attn_v.weight", (small, dim))
        add(f"blk.{l}.attn_output.weight", (dim, small))
        add(f"blk.{l}.ffn_norm.weight", (dim,))
        add(f"blk.{l}.ffn_up.weight", (small, dim))
        add(f"blk.{l}.ffn_gate.weight", (small, dim))
        add(f"blk.{l}.ffn_down.weight", (dim, small))
    add("final_norm.weight", (dim,))
    add("output.weight", (small, dim))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"[synth] OK -> {out} (layers={n_layer}, dim={dim})")

if __name__ == "__main__":
    main()
