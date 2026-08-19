#!/usr/bin/env python3
import os
os.environ["VLLM_USE_CGC_KDA"] = "1"
import sys
sys.path.insert(0, "/home/gs01")

import torch
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)

try:
    import flash_kda
    print("FlashKDA imported successfully!")
    # Qwen2.5-7B: hidden_size=3584, num_heads=28, head_size=128, num_kv_heads=4 (GQA)
    B, T, H_q, H_kv, D = 1, 10, 28, 4, 128
    q = torch.randn(B, T, H_q, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, T, H_kv, D, dtype=torch.bfloat16, device="cuda")  # GQA: 4 kv heads
    v = torch.randn(B, T, H_kv, D, dtype=torch.bfloat16, device="cuda")  # GQA: 4 kv heads
    print("Q shape:", q.shape, "K shape:", k.shape, "V shape:", v.shape)

    # Expand k/v to match q for GQA
    num_kv_groups = H_q // H_kv  # 7
    k = k.repeat_interleave(num_kv_groups, dim=2)  # [B, T, 28, D]
    v = v.repeat_interleave(num_kv_groups, dim=2)  # [B, T, 28, D]
    print("Expanded K shape:", k.shape, "Expanded V shape:", v.shape)

    g = torch.ones_like(q)
    beta = torch.ones(B, T, H_q, device="cuda", dtype=torch.bfloat16)
    A_log = torch.zeros(H_q, dtype=torch.float32, device="cuda")
    dt_bias = torch.zeros(H_q, D, dtype=torch.float32, device="cuda")
    out = torch.empty(B, T, H_q, D, dtype=torch.bfloat16, device="cuda")
    print("Calling flash_kda.fwd...")
    flash_kda.fwd(q=q, k=k, v=v, g=g, beta=beta, scale=1.0/11.0, out=out, A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0, initial_state=None, final_state=None, cu_seqlens=None)
    print("FlashKDA forward successful!")
except Exception as e:
    print(f"FlashKDA error: {e}")
    import traceback
    traceback.print_exc()
