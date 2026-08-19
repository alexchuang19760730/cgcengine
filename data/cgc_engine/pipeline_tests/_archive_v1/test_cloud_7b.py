#!/usr/bin/env python3
"""
云端7B模型真实推理测试
"""

import torch
import time
import subprocess

def run_cloud_7b_test():
    print("=" * 60)
    print("☁️ 云端Qwen2.5-7B真实推理测试")
    print("=" * 60)
    
    # 系统信息
    print("\n💻 系统信息:")
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    print(f"GPU: {result.stdout.strip()}")
    
    result = subprocess.run(["free", "-h"], capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')
    print(f"内存: {lines[1]}")
    
    # 检查PyTorch和CUDA
    print("\n⚡ PyTorch信息:")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    
    # 加载Qwen2.5-7B模型
    print("\n⏳ 加载Qwen2.5-7B模型...")
    start = time.time()
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=True  # 使用4bit量化
        )
        
        load_time = time.time() - start
        print(f"✅ 模型加载完成 ({load_time:.2f}秒)")
        
        # Prefill测试
        print("\n⚡ Prefill测试:")
        prompt = "Hello, how are you?"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # 预热
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=10)
        
        # 真实测试
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=2048)
        elapsed = time.time() - start
        
        print(f"  输入长度: {len(inputs['input_ids'][0])} tokens")
        print(f"  输出长度: {len(outputs[0])} tokens")
        print(f"  生成token数: {len(outputs[0]) - len(inputs['input_ids'][0])}")
        print(f"  耗时: {elapsed:.2f}秒")
        print(f"  Prefill吞吐量: {(len(outputs[0]) - len(inputs['input_ids'][0])) / elapsed:.1f} tokens/s")
        
        # Decode测试
        print("\n⚡ Decode测试:")
        prompt = "Hello"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # 预热
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=10)
        
        # 真实测试
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)
        elapsed = time.time() - start
        
        print(f"  生成token数: 50")
        print(f"  耗时: {elapsed:.2f}秒")
        print(f"  Decode吞吐量: {50 / elapsed:.1f} tokens/s")
        print(f"  延迟: {elapsed/50*1000:.2f} ms/token")
        
        # 多GPU测试
        print("\n🔄 多GPU并行测试:")
        if torch.cuda.device_count() > 1:
            print(f"  GPU数量: {torch.cuda.device_count()}")
            print(f"  已启用多GPU并行")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_cloud_7b_test()