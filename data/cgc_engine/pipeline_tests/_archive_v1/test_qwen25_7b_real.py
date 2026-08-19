#!/usr/bin/env python3
"""
MLX Qwen2.5-7B真实推理测试
使用本地模型文件
"""

import mlx_lm
import time
import psutil
import os

def run_real_7b_inference():
    print("=" * 70)
    print("🔍 MLX Qwen2.5-7B真实推理测试")
    print("=" * 70)
    
    # 模型路径
    model_path = "/Users/alexchuang/Documents/flashkv0430/models/qwen2.5-7b-mlx"
    
    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        return
    
    print(f"\n📦 模型路径: {model_path}")
    
    # 检查系统资源
    mem = psutil.virtual_memory()
    print(f"\n💻 系统信息:")
    print(f"  CPU核心: {psutil.cpu_count()}")
    print(f"  内存总量: {mem.total / 1e9:.1f} GB")
    print(f"  可用内存: {mem.available / 1e9:.1f} GB")
    print(f"  已用内存: {mem.percent}%")
    
    try:
        # 加载模型
        print("\n⏳ 加载Qwen2.5-7B模型...")
        start_time = time.time()
        
        model, tokenizer = mlx_lm.load(model_path)
        
        load_time = time.time() - start_time
        print(f"✅ 模型加载完成")
        print(f"   加载时间: {load_time:.2f}秒")
        
        # 检查加载后的内存使用
        mem_after = psutil.virtual_memory()
        print(f"   加载后可用内存: {mem_after.available / 1e9:.1f} GB")
        print(f"   模型占用内存: {(mem.available - mem_after.available) / 1e9:.1f} GB")
        
        # 推理测试
        print("\n⚡ 推理测试:")
        prompt = "Hello, how are you?"
        print(f"输入: {prompt}")
        
        # 预热
        mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=10)
        
        # 实际测试 - 生成50个token
        start_time = time.time()
        response = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=50)
        inference_time = time.time() - start_time
        
        print(f"输出: {response}")
        print(f"\n📊 性能统计:")
        print(f"  推理时间: {inference_time:.2f}秒")
        print(f"  生成token数: 50")
        print(f"  速度: {50 / inference_time:.1f} tokens/s")
        print(f"  延迟: {inference_time / 50 * 1000:.2f} ms/token")
        
        # 多轮推理测试
        print("\n🔄 多轮推理测试 (5次):")
        total_time = 0
        for i in range(5):
            start_time = time.time()
            mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=30)
            elapsed = time.time() - start_time
            total_time += elapsed
            print(f"  第{i+1}轮: {elapsed:.2f}秒 ({30/elapsed:.1f} tokens/s)")
        
        avg_time = total_time / 5
        print(f"\n  平均时间: {avg_time:.2f}秒")
        print(f"  平均速度: {30 / avg_time:.1f} tokens/s")
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_real_7b_inference()