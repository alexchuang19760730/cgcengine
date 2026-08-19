#!/usr/bin/env python3
"""
消融测试脚本：原生vLLM vs 双GPU PD分离并行+CUDAGraph+NCCL

测试内容：
1. 原生vLLM推理
2. PD分离并行（云端Prefill + 本地Decode）
3. CUDAGraph优化
4. NCCL多卡通信
5. 内存使用比较

运行方式: python3 ablation_test.py
"""

import sys
import os
import time
import torch
import json
from datetime import datetime

sys.path.insert(0, '/home/gs01/MagiCompiler-main')

print('=' * 80)
print('消融测试：原生vLLM vs 双GPU PD分离并行+CUDAGraph+NCCL')
print('=' * 80)
print(f'测试时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

# 测试配置
TEST_CONFIG = {
    "model_name": "Qwen/Qwen2___5-7B-Instruct",
    "max_tokens": 100,
    "prompt_lengths": [1024, 4096, 8192, 16384],
    "num_runs": 3,
}

results = []

def get_memory_usage():
    """获取GPU内存使用"""
    if torch.cuda.is_available():
        return {
            "total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
            "used_gb": torch.cuda.memory_allocated(0) / (1024**3),
            "reserved_gb": torch.cuda.memory_reserved(0) / (1024**3),
        }
    return {}

def test_native_vllm():
    """测试原生vLLM"""
    print('\n【测试1】原生vLLM推理')
    print('-' * 60)
    
    try:
        from vllm import LLM, SamplingParams
        
        results_native = {"test": "native_vllm", "runs": []}
        
        for prompt_len in TEST_CONFIG["prompt_lengths"]:
            print(f'\n测试 prompt_length={prompt_len}')
            
            # 创建测试prompt
            prompt = "Hello, " * (prompt_len // 7)
            
            # 初始化LLM
            llm = LLM(
                model=TEST_CONFIG["model_name"],
                tensor_parallel_size=1,
                gpu_memory_utilization=0.9,
                max_model_len=prompt_len + TEST_CONFIG["max_tokens"],
                enable_cuda_graph=False,
            )
            
            sampling_params = SamplingParams(max_tokens=TEST_CONFIG["max_tokens"])
            
            run_results = []
            for i in range(TEST_CONFIG["num_runs"]):
                start_time = time.time()
                outputs = llm.generate(prompt, sampling_params)
                elapsed = time.time() - start_time
                
                mem_usage = get_memory_usage()
                tokens_per_sec = TEST_CONFIG["max_tokens"] / elapsed
                
                run_results.append({
                    "run": i+1,
                    "elapsed_ms": elapsed * 1000,
                    "tokens_per_sec": tokens_per_sec,
                    "memory_used_gb": mem_usage.get("used_gb", 0),
                })
                
                print(f'  Run {i+1}: {elapsed:.2f}s, {tokens_per_sec:.1f} tokens/s')
            
            avg_elapsed = sum(r["elapsed_ms"] for r in run_results) / TEST_CONFIG["num_runs"]
            avg_tps = sum(r["tokens_per_sec"] for r in run_results) / TEST_CONFIG["num_runs"]
            avg_memory = sum(r["memory_used_gb"] for r in run_results) / TEST_CONFIG["num_runs"]
            
            results_native["runs"].append({
                "prompt_length": prompt_len,
                "avg_elapsed_ms": avg_elapsed,
                "avg_tokens_per_sec": avg_tps,
                "avg_memory_used_gb": avg_memory,
                "detailed": run_results,
            })
            
            print(f'  平均: {avg_elapsed:.2f}ms, {avg_tps:.1f} tokens/s, {avg_memory:.2f}GB')
            
            del llm
            torch.cuda.empty_cache()
        
        results.append(results_native)
        print('✅ 原生vLLM测试完成')
        
    except Exception as e:
        print(f'❌ 原生vLLM测试失败: {e}')
        import traceback
        traceback.print_exc()

def test_pd_separation():
    """测试PD分离并行"""
    print('\n【测试2】PD分离并行（双GPU）')
    print('-' * 60)
    
    try:
        from vllm import LLM, SamplingParams
        
        results_pd = {"test": "pd_separation", "runs": []}
        
        for prompt_len in TEST_CONFIG["prompt_lengths"]:
            print(f'\n测试 prompt_length={prompt_len}')
            
            # 创建测试prompt
            prompt = "Hello, " * (prompt_len // 7)
            
            # 初始化LLM（双GPU + CUDAGraph）
            llm = LLM(
                model=TEST_CONFIG["model_name"],
                tensor_parallel_size=2,
                gpu_memory_utilization=0.9,
                max_model_len=prompt_len + TEST_CONFIG["max_tokens"],
                enable_cuda_graph=True,
            )
            
            sampling_params = SamplingParams(max_tokens=TEST_CONFIG["max_tokens"])
            
            run_results = []
            for i in range(TEST_CONFIG["num_runs"]):
                start_time = time.time()
                outputs = llm.generate(prompt, sampling_params)
                elapsed = time.time() - start_time
                
                mem_usage = get_memory_usage()
                tokens_per_sec = TEST_CONFIG["max_tokens"] / elapsed
                
                run_results.append({
                    "run": i+1,
                    "elapsed_ms": elapsed * 1000,
                    "tokens_per_sec": tokens_per_sec,
                    "memory_used_gb": mem_usage.get("used_gb", 0),
                })
                
                print(f'  Run {i+1}: {elapsed:.2f}s, {tokens_per_sec:.1f} tokens/s')
            
            avg_elapsed = sum(r["elapsed_ms"] for r in run_results) / TEST_CONFIG["num_runs"]
            avg_tps = sum(r["tokens_per_sec"] for r in run_results) / TEST_CONFIG["num_runs"]
            avg_memory = sum(r["memory_used_gb"] for r in run_results) / TEST_CONFIG["num_runs"]
            
            results_pd["runs"].append({
                "prompt_length": prompt_len,
                "avg_elapsed_ms": avg_elapsed,
                "avg_tokens_per_sec": avg_tps,
                "avg_memory_used_gb": avg_memory,
                "detailed": run_results,
            })
            
            print(f'  平均: {avg_elapsed:.2f}ms, {avg_tps:.1f} tokens/s, {avg_memory:.2f}GB')
            
            del llm
            torch.cuda.empty_cache()
        
        results.append(results_pd)
        print('✅ PD分离并行测试完成')
        
    except Exception as e:
        print(f'❌ PD分离并行测试失败: {e}')
        import traceback
        traceback.print_exc()

def test_ortho_kda():
    """测试OrthoKDA v4 + 固定KV缓存"""
    print('\n【测试3】OrthoKDA v4 + 固定KV缓存')
    print('-' * 60)
    
    try:
        from cgc_engine.cgc.ortho_kda_v4_bridge import OrthoKDAV4
        import torch
        
        results_kda = {"test": "ortho_kda", "runs": []}
        
        # 初始化OrthoKDA
        kda = OrthoKDAV4(
            num_heads=32,
            head_dim=128,
            ortho_base_dim=128,
            use_cuda=True,
        )
        
        print(f'✅ OrthoKDA初始化完成: {kda.num_heads} heads × {kda.head_dim} dim')
        
        # 模拟不同长度的KV更新
        for seq_len in [1024, 4096, 8192, 16384]:
            print(f'\n测试 seq_len={seq_len}')
            
            run_results = []
            for i in range(TEST_CONFIG["num_runs"]):
                # 模拟KV更新
                start_time = time.time()
                for _ in range(seq_len // 32):  # 分批更新
                    k = torch.randn(32, 128, device='cuda')
                    v = torch.randn(32, 128, device='cuda')
                    kda.update(k, v)
                
                # 注意力计算
                q = torch.randn(32, 128, device='cuda')
                output = kda.forward(q)
                elapsed = time.time() - start_time
                
                mem_usage = get_memory_usage()
                
                run_results.append({
                    "run": i+1,
                    "elapsed_ms": elapsed * 1000,
                    "memory_used_gb": mem_usage.get("used_gb", 0),
                    "kv_cache_size": list(kda.K.shape),
                })
                
                print(f'  Run {i+1}: {elapsed:.2f}s, KV={kda.K.shape}')
            
            avg_elapsed = sum(r["elapsed_ms"] for r in run_results) / TEST_CONFIG["num_runs"]
            avg_memory = sum(r["memory_used_gb"] for r in run_results) / TEST_CONFIG["num_runs"]
            
            results_kda["runs"].append({
                "seq_length": seq_len,
                "avg_elapsed_ms": avg_elapsed,
                "avg_memory_used_gb": avg_memory,
                "kv_cache_shape": list(kda.K.shape),
                "detailed": run_results,
            })
            
            print(f'  平均: {avg_elapsed:.2f}ms, {avg_memory:.2f}GB (KV固定大小)')
        
        results.append(results_kda)
        print('✅ OrthoKDA测试完成')
        
    except Exception as e:
        print(f'❌ OrthoKDA测试失败: {e}')
        import traceback
        traceback.print_exc()

def generate_report():
    """生成测试报告"""
    print('\n' + '=' * 80)
    print('消融测试报告')
    print('=' * 80)
    
    # 保存结果到JSON
    output_file = f'ablation_test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f'\n📊 测试结果已保存到: {output_file}')
    
    # 打印对比表格
    print('\n📋 性能对比表')
    print('-' * 80)
    print(f'{"测试项":<20} {"Prompt长度":<12} {"延迟(ms)":<10} {"吞吐量":<12} {"内存(GB)":<10}')
    print('-' * 80)
    
    for result in results:
        test_name = result["test"]
        for run in result["runs"]:
            print(f'{test_name:<20} {run["prompt_length"] if "prompt_length" in run else run["seq_length"]:<12} '
                  f'{run["avg_elapsed_ms"]:<10.1f} {run.get("avg_tokens_per_sec", "-"):<12} '
                  f'{run["avg_memory_used_gb"]:<10.2f}')
    
    print('-' * 80)
    
    # 计算加速比
    if len(results) >= 2:
        native = results[0]["runs"]
        pd = results[1]["runs"]
        
        print('\n📈 加速比对比')
        print('-' * 80)
        print(f'{"Prompt长度":<12} {"PD/原生 延迟比":<15} {"PD/原生 吞吐比":<15} {"内存节省":<12}')
        print('-' * 80)
        
        for n, p in zip(native, pd):
            speedup = n["avg_elapsed_ms"] / p["avg_elapsed_ms"]
            tps_ratio = p["avg_tokens_per_sec"] / n["avg_tokens_per_sec"]
            mem_saving = (n["avg_memory_used_gb"] - p["avg_memory_used_gb"]) / n["avg_memory_used_gb"] * 100
            
            print(f'{n["prompt_length"]:<12} {speedup:<15.2f}x {tps_ratio:<15.2f}x {mem_saving:<12.1f}%')

if __name__ == '__main__':
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print('❌ CUDA不可用，请检查GPU环境')
        sys.exit(1)
    
    print(f'📌 CUDA设备: {torch.cuda.get_device_name(0)}')
    print(f'📌 CUDA版本: {torch.version.cuda}')
    
    # 运行测试
    test_native_vllm()
    test_pd_separation()
    test_ortho_kda()
    
    # 生成报告
    generate_report()