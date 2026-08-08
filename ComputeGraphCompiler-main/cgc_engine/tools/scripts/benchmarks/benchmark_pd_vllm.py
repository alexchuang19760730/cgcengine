#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/gs01')

import time
import json
import subprocess
import threading
import os

MODEL_PATH = '/home/gs01/models/Qwen/Qwen2___5-7B-Instruct'

WARMUP_TOKENS = 20
MAX_TOKENS = 100

def start_pd_server(port=50051):
    """启动 PD 服务"""
    cmd = f"python3 -m cgc_engine.pd.pd_server {port}"
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"[PD Server] Started on port {port}")
    return process

def benchmark_cgc_vllm():
    """CGC Engine vLLM 直接模式"""
    from cgc_engine import CGCEngine, CGCEngineConfig
    
    config = CGCEngineConfig(
        model_name_or_path=MODEL_PATH,
        enable_vllm=True,
        tensor_parallel_size=2,  # 双卡模式
        gpu_memory_utilization=0.65,
    )

    print(f"Creating CGCEngine with vLLM (TP=2)...")
    engine = CGCEngine(config=config)
    print(f"Engine created! Mode: {engine._get_mode()}")

    test_cases = [("Short (128)", 128), ("Medium (512)", 512), ("Long (1024)", 1024)]
    results = {}

    for name, target_tokens in test_cases:
        prompt = ("The quick brown fox jumps over the lazy dog. " * 10)[:target_tokens]
        
        _ = engine.generate(prompt[:50], max_tokens=WARMUP_TOKENS)
        
        start = time.time()
        result = engine.generate(prompt, max_tokens=MAX_TOKENS)
        elapsed = time.time() - start

        gen_tokens = len(result.get("text", "")) if isinstance(result, dict) else len(result)
        tps = gen_tokens / elapsed if elapsed > 0 else 0

        print(f"  {name}: {elapsed*1000:.1f}ms, {gen_tokens} tokens, {tps:.1f} TPS")
        results[name] = {"time_ms": elapsed*1000, "tokens": gen_tokens, "tps": tps}

    return results

def benchmark_pd_vllm():
    """PD 分离 + vLLM 模式"""
    from cgc_engine.pd import PDClient
    from cgc_engine import CGCEngine, CGCEngineConfig
    
    pd_client = PDClient("localhost:50051")
    healthy, stats = pd_client.health_check()
    print(f"[PD Client] Connected: {healthy}, Stats: {stats}")

    config = CGCEngineConfig(
        model_name_or_path=MODEL_PATH,
        enable_vllm=True,
        enable_pd_service=True,  # 启用 PD 服务
        pd_address="localhost:50051",
        tensor_parallel_size=2,  # 双卡模式
        gpu_memory_utilization=0.65,
    )

    print(f"Creating CGCEngine with PD + vLLM (TP=2)...")
    engine = CGCEngine(config=config)
    print(f"Engine created! Mode: {engine._get_mode()}")

    test_cases = [("Short (128)", 128), ("Medium (512)", 512), ("Long (1024)", 1024)]
    results = {}

    for name, target_tokens in test_cases:
        prompt = ("The quick brown fox jumps over the lazy dog. " * 10)[:target_tokens]
        
        _ = engine.generate(prompt[:50], max_tokens=WARMUP_TOKENS)
        
        start = time.time()
        result = engine.generate(prompt, max_tokens=MAX_TOKENS)
        elapsed = time.time() - start

        gen_tokens = len(result.get("text", "")) if isinstance(result, dict) else len(result)
        tps = gen_tokens / elapsed if elapsed > 0 else 0

        print(f"  {name}: {elapsed*1000:.1f}ms, {gen_tokens} tokens, {tps:.1f} TPS")
        results[name] = {"time_ms": elapsed*1000, "tokens": gen_tokens, "tps": tps}

    pd_client.close()
    return results

def main():
    print("=" * 80)
    print("PD 分离 vs CGC Engine vLLM 双卡模式对比测试")
    print("=" * 80)
    print(f"Model: {MODEL_PATH}")
    print(f"GPU: NVIDIA RTX 5090 x 2")
    print("=" * 80)

    # 测试 1: CGC Engine vLLM 直接模式
    print("\n【1】CGC Engine vLLM 直接模式 (TP=2)")
    print("-" * 60)
    cgc_results = benchmark_cgc_vllm()

    # 启动 PD 服务
    print("\n【2】启动 PD 服务...")
    pd_process = start_pd_server(50051)
    time.sleep(3)  # 等待服务启动

    # 测试 2: PD 分离 + vLLM 模式
    print("\n【3】PD 分离 + vLLM 模式 (TP=2)")
    print("-" * 60)
    pd_results = benchmark_pd_vllm()

    # 停止 PD 服务
    pd_process.terminate()
    pd_process.wait()

    # 输出对比结果
    print("\n" + "=" * 80)
    print("📊 性能对比结果")
    print("=" * 80)
    print(f"{'Test Case':<15} {'CGC vLLM':<20} {'PD + vLLM':<20} {'Improvement':<10}")
    print("-" * 80)
    
    for case in ["Short (128)", "Medium (512)", "Long (1024)"]:
        cgc_tps = cgc_results[case]["tps"]
        pd_tps = pd_results[case]["tps"]
        improvement = ((pd_tps - cgc_tps) / cgc_tps * 100) if cgc_tps > 0 else 0
        
        print(f"{case:<15} {cgc_tps:<20.1f} {pd_tps:<20.1f} {improvement:<10.1f}%")

    with open('/home/gs01/pd_vs_cgc_results.json', 'w') as f:
        json.dump({
            "cgc_vllm": cgc_results,
            "pd_vllm": pd_results
        }, f, indent=2)

    print("\n📝 结果已保存到 /home/gs01/pd_vs_cgc_results.json")

if __name__ == "__main__":
    main()
