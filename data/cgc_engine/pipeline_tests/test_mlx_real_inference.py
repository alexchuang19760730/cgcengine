#!/usr/bin/env python3
"""
MLX 7B模型真实推理测试 - 使用mlx_lm
"""

import mlx_lm
import time
import psutil

def run_7b_inference_test():
    print("=" * 60)
    print("🔍 MLX 7B模型真实推理测试")
    print("=" * 60)
    
    # 检查系统资源
    mem = psutil.virtual_memory()
    print(f"\n💻 系统信息:")
    print(f"  可用内存: {mem.available / 1e9:.1f} GB")
    print(f"  已用内存: {mem.percent}%")
    
    try:
        # 加载模型
        print("\n⏳ 加载7B模型...")
        start_time = time.time()
        
        # 尝试加载较小的模型进行测试
        # Phi-2 2.7B比7B小，更适合当前内存
        model, tokenizer = mlx_lm.load("mlx-community/Phi-2")
        
        load_time = time.time() - start_time
        print(f"✅ 模型加载完成")
        print(f"   加载时间: {load_time:.2f}秒")
        
        # 推理测试
        print("\n⚡ 推理测试:")
        prompt = "Hello, how are you?"
        print(f"输入: {prompt}")
        
        # 预热
        mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=10)
        
        # 实际测试
        start_time = time.time()
        response = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=50)
        inference_time = time.time() - start_time
        
        print(f"输出: {response}")
        print(f"\n📊 性能统计:")
        print(f"  推理时间: {inference_time:.2f}秒")
        print(f"  生成token数: 50")
        print(f"  速度: {50 / inference_time:.1f} tokens/s")
        print(f"  延迟: {inference_time / 50 * 1000:.2f} ms/token")
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        print("💡 可能原因:")
        print("   1. 内存不足（Phi-2需要约5.4GB内存）")
        print("   2. 网络问题（需要下载模型）")
        print("   3. 模型文件损坏")

if __name__ == "__main__":
    run_7b_inference_test()