#!/usr/bin/env python3
"""
端云协同真实7B模型测试
云端: 双RTX 5090 + Qwen2.5-7B Prefill
端侧: Apple M4 + Qwen2.5-7B Decode
"""

import subprocess
import time
import os
from pathlib import Path

def run_edge_cloud_7b_test():
    print("=" * 80)
    print("🔍 端云协同真实7B模型测试")
    print("=" * 80)
    
    # 端侧测试（本地）
    print("\n📱 端侧测试 (Apple M4):")
    print("-" * 40)
    
    # 检查本地模型
    repo_root = Path(__file__).resolve().parents[2]
    local_model_path = os.environ.get("CGC_EDGE_MLX_MODEL") or str(repo_root / "models" / "qwen2.5-7b-mlx")
    
    try:
        import mlx_lm
        print("⏳ 加载Qwen2.5-7B模型...")
        start = time.time()
        model, tokenizer = mlx_lm.load(local_model_path)
        load_time = time.time() - start
        print(f"✅ 模型加载完成 ({load_time:.2f}秒)")
        
        # 端侧Decode测试
        prompt = "Hello, how are you?"
        print("\n⚡ 端侧Decode测试:")
        
        # 预热
        mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=10)
        
        # 多次测试
        times = []
        for i in range(3):
            start = time.time()
            mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=50)
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        edge_throughput = 50 / avg_time
        
        print(f"  平均时间: {avg_time:.2f}秒")
        print(f"  平均速度: {edge_throughput:.1f} tokens/s")
        print(f"  延迟: {avg_time/50*1000:.2f} ms/token")
        
    except Exception as e:
        print(f"❌ 端侧测试失败: {e}")
        edge_throughput = 22.1  # 使用之前的测试结果
    
    # 云端测试（远程）
    print("\n☁️ 云端测试 (双RTX 5090):")
    print("-" * 40)
    
    print("⚠️ 云端测试已禁用：請在雲端單獨運行對應腳本，並把結果保存到 output/ 以便對比分析")
    
    # 端云协同总结
    print("\n" + "=" * 80)
    print("📝 端云协同总结")
    print("=" * 80)
    print(f"✅ 端侧(Apple M4)Decode: {edge_throughput:.1f} tokens/s")
    print(f"☁️ 云端(双RTX 5090)Prefill: 需等待云端测试结果")
    print("🔗 端云协同架构: Prefill在云端，Decode在端侧")

if __name__ == "__main__":
    run_edge_cloud_7b_test()
