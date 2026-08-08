#!/usr/bin/env python3
"""Simple test for KDA Backend"""
import os
import sys
import json
import glob
import time
from pathlib import Path

# 设置路径，确保能找到 Backend/Vllm/vllm_backend
repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# 设置环境变量
os.environ['VLLM_USE_CGC_KDA'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

print("=" * 80)
print("KDA Backend Test")
print("=" * 80)
print(f"[1] PYTHONPATH: {sys.path[:5]}")
print(f"[1] VLLM_USE_CGC_KDA: {os.environ.get('VLLM_USE_CGC_KDA')}")

# 清理旧统计
print("\n[2] Cleaning old stats...")
for f in glob.glob('/tmp/kda_stats_pid*.json'):
    try:
        os.remove(f)
        print(f"  - Deleted: {f}")
    except Exception as e:
        print(f"  - Failed to delete: {e}")

# 导入 vLLM 和我们的后端
print("\n[3] Importing vLLM...")
try:
    # 确保我们的后端被导入和注册
    import Backend.Vllm.vllm_backend.cgc_kda_backend
    print("✅ Backend.Vllm.vllm_backend imported")
    
    from vllm import LLM, SamplingParams
    print("✅ vLLM imported")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 模型路径
MODEL_PATH = "/home/gs01/models/Qwen/Qwen2___5-7B-Instruct"
print(f"\n[4] Using model: {MODEL_PATH}")

print("\n" + "=" * 80)
print("[5] Loading model...")
print("=" * 80)
llm = None
try:
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.70,
        max_model_len=4096,
        enforce_eager=True,
        disable_log_stats=True,
    )
    print("✅ Model loaded!")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("[6] Running inference...")
print("=" * 80)
if llm:
    try:
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=64,
        )
        prompts = [
            "Hello, world! What is 2+2?",
            "Explain quantum computing in simple terms.",
        ]
        print(f"Prompts: {prompts}")
        
        outputs = llm.generate(prompts, sampling_params)
        
        for i, output in enumerate(outputs):
            print(f"\n✅ Output {i+1}:")
            print(f"  Prompt: {output.prompt[:80]}...")
            print(f"  Response: {output.outputs[0].text}")
            
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("[7] Reading KDA stats...")
print("=" * 80)
stats_files = glob.glob('/tmp/kda_stats_pid*.json')
print(f"Found {len(stats_files)} stats files:")

all_stats = {}
for f in sorted(stats_files):
    try:
        with open(f, 'r') as fp:
            s = json.load(fp)
            pid = s.get('pid', 'unknown')
            all_stats[pid] = s
            print(f"\n  [{pid}]")
            for k, v in s.items():
                if k in ['timestamp', 'pid']:
                    continue
                print(f"    {k}: {v}")
    except Exception as e:
        print(f"\n  Failed to read {f}: {e}")

print("\n" + "=" * 80)
print("Test complete!")
print("=" * 80)
