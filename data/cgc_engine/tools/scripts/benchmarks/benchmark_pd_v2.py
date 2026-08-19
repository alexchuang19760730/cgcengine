#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/gs01')

import time
import json
import subprocess
import time

MODEL_PATH = '/home/gs01/models/Qwen/Qwen2___5-7B-Instruct'

WARMUP_TOKENS = 20
MAX_TOKENS = 100

def start_pd_server(port=50051):
    """启动 PD 服务"""
    cmd = f"python3 -m cgc_engine.pd.pd_server {port}"
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"[PD Server] Started on port {port}")
    return process

def test_pd_service():
    """测试 PD 服务连接"""
    from cgc_engine.pd import PDClient
    
    client = PDClient("localhost:50051")
    healthy, stats = client.health_check()
    print(f"[PD Client] Health Check: {healthy}, Stats: {stats}")
    
    # 测试块分配
    block_ids, success = client.allocate_blocks(sequence_ids=[1], num_blocks=4)
    print(f"[PD Client] Allocated blocks: {block_ids}, Success: {success}")
    
    return healthy

def benchmark_vllm(tp_size=2, description=""):
    """Benchmark vLLM"""
    from cgc_engine import CGCEngine, CGCEngineConfig
    
    config = CGCEngineConfig(
        model_name_or_path=MODEL_PATH,
        enable_vllm=True,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=0.65,
    )

    print(f"Creating CGCEngine with vLLM (TP={tp_size})...")
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

def main():
    print("=" * 80)
    print("PD 分离 vs CGC Engine vLLM 双卡模式对比测试")
    print("=" * 80)
    print(f"Model: {MODEL_PATH}")
    print(f"GPU: NVIDIA RTX 5090 x 2")
    print("=" * 80)

    # 测试 1: 单卡模式
    print("\n【1】CGC Engine vLLM (TP=1)")
    print("-" * 60)
    tp1_results = benchmark_vllm(tp_size=1, description="Single GPU")

    # 测试 2: 双卡模式
    print("\n【2】CGC Engine vLLM (TP=2)")
    print("-" * 60)
    tp2_results = benchmark_vllm(tp_size=2, description="Dual GPU")

    # 启动 PD 服务
    print("\n【3】启动 PD 服务...")
    pd_process = start_pd_server(50051)
    time.sleep(3)
    
    # 测试 PD 服务
    print("\n【4】测试 PD 服务连接")
    print("-" * 60)
    pd_available = test_pd_service()

    # 测试 3: PD + vLLM 双卡模式
    if pd_available:
        print("\n【5】PD + vLLM (TP=2)")
        print("-" * 60)
        pd_vllm_results = benchmark_vllm(tp_size=2, description="PD + Dual GPU")
    else:
        pd_vllm_results = None
        print("\n【5】PD 服务不可用，跳过 PD + vLLM 测试")

    # 停止 PD 服务
    pd_process.terminate()
    pd_process.wait()

    # 输出对比结果
    print("\n" + "=" * 80)
    print("📊 性能对比结果")
    print("=" * 80)
    print(f"{'Test Case':<15} {'TP=1':<15} {'TP=2':<15} {'Improvement':<10}")
    print("-" * 80)
    
    for case in ["Short (128)", "Medium (512)", "Long (1024)"]:
        tp1_tps = tp1_results[case]["tps"]
        tp2_tps = tp2_results[case]["tps"]
        improvement = ((tp2_tps - tp1_tps) / tp1_tps * 100) if tp1_tps > 0 else 0
        
        print(f"{case:<15} {tp1_tps:<15.1f} {tp2_tps:<15.1f} {improvement:<10.1f}%")

    with open('/home/gs01/pd_vllm_results.json', 'w') as f:
        json.dump({
            "tp1": tp1_results,
            "tp2": tp2_results,
            "pd_vllm": pd_vllm_results
        }, f, indent=2)

    print("\n📝 结果已保存到 /home/gs01/pd_vllm_results.json")

if __name__ == "__main__":
    main()
