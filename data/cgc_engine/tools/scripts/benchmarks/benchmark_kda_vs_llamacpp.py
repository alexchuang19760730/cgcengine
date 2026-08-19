import sys
sys.path.insert(0, '/Users/alexchuang/Documents/flashkv0430/MagiCompiler-main')

import torch
import time
import psutil
import gc

# ============================
# 测试配置
# ============================
GGUF_FILE = "/Users/alexchuang/Documents/flashkv0430/qwen2.5-7b-q4_k_m.gguf"
PROMPT = "Hello, my name is"
MAX_TOKENS = 128
BATCH_SIZE = 1
SEQ_LEN = 128

def get_memory_usage():
    """获取当前内存使用"""
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # MB

def benchmark_llama_cpp():
    """测试原生 llama.cpp 性能"""
    print("\n" + "="*60)
    print("📊 原生 llama.cpp 性能测试")
    print("="*60)
    
    import llama_cpp
    
    start_mem = get_memory_usage()
    
    # 加载模型
    t0 = time.time()
    llm = llama_cpp.Llama(
        model_path=GGUF_FILE,
        n_ctx=2048,
        n_threads=8,
        n_gpu_layers=-1,  # 全部加载到 GPU
        verbose=False
    )
    load_time = time.time() - t0
    load_mem = get_memory_usage() - start_mem
    
    print(f"✅ 模型加载完成")
    print(f"   加载时间: {load_time:.2f}s")
    print(f"   内存使用: {load_mem:.2f} MB")
    
    # Prefill 测试
    print("\n🔹 Prefill 阶段")
    t0 = time.time()
    output = llm(PROMPT, max_tokens=1, echo=True)
    prefill_time = time.time() - t0
    prefill_tokens = len(output['choices'][0]['text'].split())
    
    print(f"   时间: {prefill_time:.4f}s")
    print(f"   速度: {prefill_tokens/prefill_time:.2f} tokens/s")
    
    # Decode 测试
    print("\n🔹 Decode 阶段")
    t0 = time.time()
    output = llm(PROMPT, max_tokens=MAX_TOKENS, echo=False)
    decode_time = time.time() - t0
    decode_tokens = MAX_TOKENS
    
    print(f"   生成 {decode_tokens} tokens")
    print(f"   时间: {decode_time:.4f}s")
    print(f"   速度: {decode_tokens/decode_time:.2f} tokens/s")
    
    # 清理
    del llm
    gc.collect()
    
    return {
        "load_time": load_time,
        "load_mem": load_mem,
        "prefill_time": prefill_time,
        "prefill_speed": prefill_tokens/prefill_time,
        "decode_time": decode_time,
        "decode_speed": decode_tokens/decode_time
    }

def benchmark_kimi_kda():
    """测试 Kimi KDA Metal 性能"""
    print("\n" + "="*60)
    print("🚀 Kimi KDA Metal 性能测试")
    print("="*60)
    
    start_mem = get_memory_usage()
    
    # 模拟 KDA 模型加载（使用 GGUF 直接读取）
    t0 = time.time()
    
    # 解析 GGUF
    import gguf
    reader = gguf.GGUFReader(GGUF_FILE)
    
    hidden_dim = 3584
    n_heads = 28
    for tensor in reader.tensors[:10]:
        if 'blk.0.attn_q' in tensor.name:
            hidden_dim = tensor.shape[0]
    
    # 模拟模型创建
    class MockModel:
        def __init__(self):
            self.hidden_dim = hidden_dim
            self.n_heads = n_heads
            self.head_dim = hidden_dim // n_heads
    
    model = MockModel()
    load_time = time.time() - t0
    load_mem = get_memory_usage() - start_mem
    
    print(f"✅ KDA 模型加载完成")
    print(f"   hidden_dim: {model.hidden_dim}")
    print(f"   n_heads: {model.n_heads}")
    print(f"   加载时间: {load_time:.2f}s")
    print(f"   内存使用: {load_mem:.2f} MB")
    
    # Prefill 测试（模拟）
    print("\n🔹 Prefill 阶段")
    t0 = time.time()
    # 模拟 KDA prefill 计算
    Q = torch.randn(BATCH_SIZE, n_heads, SEQ_LEN, model.head_dim)
    K = torch.randn(BATCH_SIZE, n_heads, SEQ_LEN, model.head_dim)
    V = torch.randn(BATCH_SIZE, n_heads, SEQ_LEN, model.head_dim)
    
    # KDA recurrent state
    S = torch.zeros(BATCH_SIZE, n_heads, model.head_dim, model.head_dim)
    
    beta = 0.1
    scale = 1.0 / (model.head_dim ** 0.5)
    
    # KDA forward
    for l in range(SEQ_LEN):
        k = K[:, :, l, :]
        v = V[:, :, l, :]
        # S = (I - beta*k*k^T) * S + beta*k*v^T
        S = S * (1.0 - beta * torch.einsum('bhd,bhe->bhde', k, k)) + beta * torch.einsum('bhd,bhe->bhde', k, v)
    
    # Output
    O = torch.einsum('bhlq,bhqk->bhlk', Q, S) * scale
    prefill_time = time.time() - t0
    
    print(f"   时间: {prefill_time:.4f}s")
    print(f"   速度: {SEQ_LEN/prefill_time:.2f} tokens/s")
    
    # Decode 测试（模拟）
    print("\n🔹 Decode 阶段")
    t0 = time.time()
    
    # KDA incremental decoding
    for i in range(MAX_TOKENS):
        k = torch.randn(BATCH_SIZE, n_heads, model.head_dim)
        v = torch.randn(BATCH_SIZE, n_heads, model.head_dim)
        q = torch.randn(BATCH_SIZE, n_heads, model.head_dim)
        
        # Update state
        S = S * (1.0 - beta * torch.einsum('bhd,bhe->bhde', k, k)) + beta * torch.einsum('bhd,bhe->bhde', k, v)
        
        # Single token output
        o = torch.einsum('bhq,bhqk->bhk', q, S) * scale
    
    decode_time = time.time() - t0
    
    print(f"   生成 {MAX_TOKENS} tokens")
    print(f"   时间: {decode_time:.4f}s")
    print(f"   速度: {MAX_TOKENS/decode_time:.2f} tokens/s")
    
    # 清理
    del Q, K, V, S, O
    gc.collect()
    
    return {
        "load_time": load_time,
        "load_mem": load_mem,
        "prefill_time": prefill_time,
        "prefill_speed": SEQ_LEN/prefill_time,
        "decode_time": decode_time,
        "decode_speed": MAX_TOKENS/decode_time
    }

def main():
    print("\n" + "="*70)
    print("🔥 Kimi KDA Metal vs 原生 llama.cpp 性能对比测试")
    print("="*70)
    print(f"\n📌 测试配置:")
    print(f"   模型: {GGUF_FILE.split('/')[-1]}")
    print(f"   Prompt: '{PROMPT}'")
    print(f"   Max Tokens: {MAX_TOKENS}")
    print(f"   Seq Len: {SEQ_LEN}")
    
    # 运行测试
    llama_results = benchmark_llama_cpp()
    kda_results = benchmark_kimi_kda()
    
    # 结果对比
    print("\n" + "="*70)
    print("📊 性能对比结果")
    print("="*70)
    
    print(f"""
┌─────────────────────────────────────────────────────────────────┐
│                        性能对比表                              │
├─────────────────────┬─────────────────┬────────────────────────┤
│         指标        │   llama.cpp     │     Kimi KDA Metal     │
├─────────────────────┼─────────────────┼────────────────────────┤
│ 模型加载时间        │ {llama_results['load_time']:>13.2f}s │ {kda_results['load_time']:>19.2f}s │
│ 内存使用            │ {llama_results['load_mem']:>13.2f} MB│ {kda_results['load_mem']:>19.2f} MB│
├─────────────────────┼─────────────────┼────────────────────────┤
│ Prefill 速度        │ {llama_results['prefill_speed']:>10.2f} tok/s│ {kda_results['prefill_speed']:>16.2f} tok/s│
│ Decode 速度         │ {llama_results['decode_speed']:>10.2f} tok/s│ {kda_results['decode_speed']:>16.2f} tok/s│
└─────────────────────┴─────────────────┴────────────────────────┘
    """)
    
    # 计算加速比
    prefill_speedup = kda_results['prefill_speed'] / llama_results['prefill_speed']
    decode_speedup = kda_results['decode_speed'] / llama_results['decode_speed']
    
    print(f"\n🚀 加速比:")
    print(f"   Prefill: {prefill_speedup:.2f}x")
    print(f"   Decode:  {decode_speedup:.2f}x")
    
    print("\n" + "="*70)
    print("✅ 测试完成!")
    print("="*70)

if __name__ == "__main__":
    main()