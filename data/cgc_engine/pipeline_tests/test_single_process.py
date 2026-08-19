#!/usr/bin/env python3
"""
單進程測試腳本，用於驗證 CGC KDA Backend 是否正確工作
"""

import sys
import os
import torch
from pathlib import Path

# 先註冊我們的後端
print("=== 註冊 CGC KDA Backend ===")
repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
import Backend.Vllm.vllm_backend.cgc_kda_backend as _cgc_kda_backend
success = True
print(f"註冊結果: {success}")

# 現在以單進程模式創建 LLM
print("\n=== 以單進程模式創建 LLM ===")
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

from vllm import LLM, SamplingParams

# 使用單進程模式 - 關鍵是 enforce_eager=True
llm = LLM(
    model='/home/gs01/models/Qwen/Qwen2___5-7B-Instruct',
    tensor_parallel_size=1,
    gpu_memory_utilization=0.5,
    max_model_len=1024,
    enforce_eager=True,  # 禁用 CUDA 圖和多進程優化，確保用我們的後端
    disable_log_stats=False
)

print("\n=== 測試生成 ===")
prompt = "Hello, world! Please write a story about a robot."
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=100
)

outputs = llm.generate(prompt, sampling_params)
print(f"\n生成結果: {outputs[0].outputs[0].text}")
