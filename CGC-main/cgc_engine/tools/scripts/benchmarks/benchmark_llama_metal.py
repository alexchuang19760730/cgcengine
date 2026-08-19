import sys
sys.path.insert(0, '/Users/alexchuang/Documents/flashkv0430/MagiCompiler-main')

import time
import psutil

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 2)  # MB

def benchmark_llama_cpp_metal():
    """测试 llama.cpp Metal 加速性能"""
    print("\n" + "="*60)
    print("🔥 llama.cpp Metal 加速测试")
    print("="*60)
    
    import llama_cpp
    
    GGUF_FILE = "/Users/alexchuang/Documents/flashkv0430/qwen2.5-7b-q4_k_m.gguf"
    PROMPT = "Hello, my name is"
    MAX_TOKENS = 128
    
    start_mem = get_memory_usage()
    
    # 加载模型（使用 Metal 加速）
    print("\n🔹 加载模型...")
    t0 = time.time()
    llm = llama_cpp.Llama(
        model_path=GGUF_FILE,
        n_ctx=2048,
        n_threads=8,
        n_gpu_layers=-1,  # 全部加载到 GPU
        verbose=False,
        logits_all=False
    )
    load_time = time.time() - t0
    load_mem = get_memory_usage() - start_mem
    
    print(f"✅ 模型加载完成")
    print(f"   加载时间: {load_time:.2f}s")
    print(f"   内存使用: {load_mem:.2f} MB")
    
    # 预热
    print("\n🔹 预热阶段...")
    for _ in range(3):
        llm(PROMPT, max_tokens=1, echo=False)
    
    # Prefill 测试
    print("\n🔹 Prefill 测试")
    t0 = time.time()
    for _ in range(10):
        output = llm(PROMPT, max_tokens=1, echo=True)
    prefill_time = (time.time() - t0) / 10
    prefill_tokens = len(output['choices'][0]['text'].split())
    
    print(f"   平均时间: {prefill_time:.4f}s")
    print(f"   速度: {prefill_tokens/prefill_time:.2f} tokens/s")
    
    # Decode 测试
    print("\n🔹 Decode 测试")
    t0 = time.time()
    for _ in range(5):
        output = llm(PROMPT, max_tokens=MAX_TOKENS, echo=False)
    decode_time = (time.time() - t0) / 5
    
    print(f"   生成 {MAX_TOKENS} tokens")
    print(f"   平均时间: {decode_time:.4f}s")
    print(f"   速度: {MAX_TOKENS/decode_time:.2f} tokens/s")
    
    # 清理
    del llm
    
    return {
        "load_time": load_time,
        "load_mem": load_mem,
        "prefill_speed": prefill_tokens/prefill_time,
        "decode_speed": MAX_TOKENS/decode_time
    }

def main():
    print("\n" + "="*70)
    print("🎯 llama.cpp Metal 真实性能测试")
    print("="*70)
    
    results = benchmark_llama_cpp_metal()
    
    print("\n" + "="*70)
    print("📊 测试结果")
    print("="*70)
    print(f"""
┌─────────────────────────────────────────────────────────┐
│           llama.cpp Metal 加速测试结果                 │
├─────────────────────────────────────────────────────────┤
│  模型: Qwen2.5-7B-Q4_K_M                              │
│  设备: Apple M4 (Metal GPU)                           │
├─────────────────────────────────────────────────────────┤
│  模型加载时间: {results['load_time']:>8.2f}s           │
│  内存使用:     {results['load_mem']:>8.2f} MB         │
├─────────────────────────────────────────────────────────┤
│  Prefill 速度: {results['prefill_speed']:>8.2f} tok/s │
│  Decode 速度:  {results['decode_speed']:>8.2f} tok/s  │
└─────────────────────────────────────────────────────────┘
    """)

if __name__ == "__main__":
    main()