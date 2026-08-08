#!/usr/bin/env python3
"""
🔥 llama.cpp 原生 vs KDA 替换 Attention 完整推理对比
目标：证明 KDA 替换后完整推理的效能提升
"""

import sys
import time
import psutil
import gc

sys.path.insert(0, '/Users/alexchuang/Documents/flashkv0430/MagiCompiler-main/cgc_engine/cgc/cgc_cpp/build')

print("="*70)
print("🔥 llama.cpp 原生 vs KDA 替换 Attention 完整推理对比")
print("="*70)

GGUF_FILE = "/Users/alexchuang/Documents/flashkv0430/qwen2.5-7b-q4_k_m.gguf"

def get_memory():
    return psutil.Process().memory_info().rss / (1024 ** 2)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

print(f"\n📊 初始内存: {get_memory():.2f} MB")
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ MPS: {torch.backends.mps.is_available()}")

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"📊 设备: {device}")

print("\n" + "="*70)
print("【第一步】加载 GGUF 权重")
print("="*70)

import gguf
reader = gguf.GGUFReader(GGUF_FILE)

weights = {}
for tensor in reader.tensors:
    try:
        qtype = gguf.GGMLQuantizationType(tensor.tensor_type)
        arr = gguf.dequantize(tensor.data, qtype)
        if arr is not None:
            if hasattr(arr, 'numpy'):
                arr = arr.numpy()
            weights[tensor.name] = arr.astype(np.float32)
    except:
        continue

print(f"✅ GGUF 加载: {len(weights)} tensors")

print(f"📊 当前内存: {get_memory():.2f} MB")

print("\n" + "="*70)
print("【第二步】构建 PyTorch 模型 (KDA Attention 替换版)")
print("="*70)

class QwenAttention(nn.Module):
    """标准 Attention"""
    def __init__(self, hidden_dim, num_heads, num_kv_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v)

        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn)

class KDAAttention(nn.Module):
    """KDA Attention - 替换标准 Attention"""
    def __init__(self, hidden_dim, num_heads, num_kv_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.kda = None
        try:
            import kda_cpp
            self.kda = kda_cpp.KDA()
            self.kda.init(1, num_heads, self.head_dim)
            print("   ✅ C++ KDA 已加载")
        except:
            print("   ⚠️ C++ KDA 不可用")

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        if self.kda is not None:
            q_np = np.ascontiguousarray(q.cpu().numpy().astype(np.float32))
            k_np = np.ascontiguousarray(k.cpu().numpy().astype(np.float32))
            v_np = np.ascontiguousarray(v.cpu().numpy().astype(np.float32))

            O = self.kda.forward(q_np, k_np, v_np, beta=0.1)
            attn = torch.from_numpy(np.array(O)).to(q.device)
            attn = attn.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            attn = F.scaled_dot_product_attention(q, k, v)
            attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            return self.o_proj(attn)

        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn)

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_kv_heads, use_kda=True):
        super().__init__()
        self.attn = KDAAttention(hidden_dim, num_heads, num_kv_heads) if use_kda else QwenAttention(hidden_dim, num_heads, num_kv_heads)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 18944, bias=False),
            nn.SiLU(),
            nn.Linear(18944, hidden_dim, bias=False)
        )
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class MiniLLM(nn.Module):
    """简化版 LLM 用于测试"""
    def __init__(self, weights_dict, num_layers=4, use_kda=True):
        super().__init__()
        self.num_layers = num_layers
        vocab_size = 152064
        hidden_dim = 3584
        num_heads = 28
        num_kv_heads = 4

        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, num_kv_heads, use_kda)
            for _ in range(num_layers)
        ])
        self.norm = nn.RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        self._load_weights(weights_dict)

    def _load_weights(self, w):
        try:
            if 'token_embd.weight' in w:
                self.embed.weight.data = torch.from_numpy(w['token_embd.weight']).float()
            if 'output_norm.weight' in w:
                self.norm.weight.data = torch.from_numpy(w['output_norm.weight']).float()

            for i in range(self.num_layers):
                prefix = f'blk.{i}'

                for name, proj in [('q', 'q_proj'), ('k', 'k_proj'), ('v', 'v_proj')]:
                    w_name = f'{prefix}.attn_{name}.weight'
                    if w_name in w:
                        data = w[w_name]
                        if data.shape[0] == 3584:
                            data = data.T
                        getattr(self.layers[i].attn, proj).weight.data = torch.from_numpy(data).float()

                o_name = f'{prefix}.attn_output.weight'
                if o_name in w:
                    data = w[o_name]
                    if data.shape[0] == 3584:
                        data = data.T
                    self.layers[i].attn.o_proj.weight.data = torch.from_numpy(data).float()

                for name, idx in [('gate', 0), ('up', 2)]:
                    w_name = f'{prefix}.ffn_{name}.weight'
                    if w_name in w:
                        data = w[w_name]
                        if data.shape[0] == 3584:
                            data = data.T
                        self.layers[i].mlp[idx].weight.data = torch.from_numpy(data).float()

                down_name = f'{prefix}.ffn_down.weight'
                if down_name in w:
                    data = w[down_name]
                    if data.shape[1] == 3584:
                        data = data.T
                    self.layers[i].mlp[2].weight.data = torch.from_numpy(data).float()

            print("   ✅ 权重加载完成")
        except Exception as e:
            print(f"   ⚠️ 权重加载错误: {e}")

    def forward(self, input_ids):
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.norm(hidden)
        return self.lm_head(hidden)

NUM_LAYERS = 4

print(f"\n🔨 构建 {NUM_LAYERS} 层标准模型...")
model_std = MiniLLM(weights, num_layers=NUM_LAYERS, use_kda=False).to(device)
model_std.eval()
print(f"📊 内存: {get_memory():.2f} MB")

gc.collect()

print(f"\n🔨 构建 {NUM_LAYERS} 层 KDA 模型...")
model_kda = MiniLLM(weights, num_layers=NUM_LAYERS, use_kda=True).to(device)
model_kda.eval()
print(f"📊 内存: {get_memory():.2f} MB")

print("\n" + "="*70)
print("【第三步】完整推理测试")
print("="*70)

@torch.no_grad()
def run_full_inference(model, prompt, max_tokens=8):
    """完整端到端推理"""
    input_ids = [ord(c) % 152064 for c in prompt]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    t0 = time.time()
    for _ in range(max_tokens):
        logits = model(input_tensor)
        next_token = torch.argmax(logits[0, -1]).item()
        input_ids.append(next_token)
        input_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
    elapsed = time.time() - t0

    return {
        "time": elapsed,
        "tps": max_tokens / elapsed if elapsed > 0 else 0,
        "tokens": max_tokens
    }

prompt = "Hello"
max_tokens = 8

print(f"\n📝 提示: {repr(prompt)}, 生成: {max_tokens} tokens")

print("\n" + "-"*50)
print("🔹 llama.cpp 原生推理")
print("-"*50)

try:
    from llama_cpp import Llama

    llm = Llama(
        model_path=GGUF_FILE,
        n_ctx=512,
        n_gpu_layers=32 if torch.backends.mps.is_available() else 0,
        verbose=False
    )

    t0 = time.time()
    output = llm(prompt, max_tokens=max_tokens)
    llama_time = time.time() - t0

    result_llama = {
        "time": llama_time,
        "tps": max_tokens / llama_time
    }

    print(f"   ✅ llama.cpp 完成")
    print(f"   • 时间: {llama_time*1000:.2f} ms")
    print(f"   • 速度: {max_tokens/llama_time:.2f} tok/s")

    del llm
    gc.collect()

except Exception as e:
    print(f"   ❌ llama.cpp 失败: {e}")
    result_llama = None

print(f"\n📊 内存: {get_memory():.2f} MB")

print("\n" + "-"*50)
print(f"🔹 PyTorch 标准模型 (4层, {NUM_LAYERS})")
print("-"*50)

try:
    result_std = run_full_inference(model_std, prompt, max_tokens)
    print(f"   ✅ 标准模型完成")
    print(f"   • 时间: {result_std['time']*1000:.2f} ms")
    print(f"   • 速度: {result_std['tps']:.2f} tok/s")
except Exception as e:
    print(f"   ❌ 标准模型失败: {e}")
    result_std = None

gc.collect()
print(f"📊 内存: {get_memory():.2f} MB")

print("\n" + "-"*50)
print(f"🔹 PyTorch KDA 模型 (4层, {NUM_LAYERS})")
print("-"*50)

try:
    result_kda = run_full_inference(model_kda, prompt, max_tokens)
    print(f"   ✅ KDA 模型完成")
    print(f"   • 时间: {result_kda['time']*1000:.2f} ms")
    print(f"   • 速度: {result_kda['tps']:.2f} tok/s")
except Exception as e:
    print(f"   ❌ KDA 模型失败: {e}")
    result_kda = None

print(f"\n📊 内存: {get_memory():.2f} MB")

print("\n" + "="*70)
print("📊 结果对比")
print("="*70)

print(f"\n{'方案':<30} {'速度 (tok/s)':<15} {'时间 (ms)':<15}")
print("-"*60)

if result_llama:
    print(f"{'llama.cpp 原生':<30} {result_llama['tps']:<15.2f} {result_llama['time']*1000:<15.2f}")
if result_std:
    print(f"{'PyTorch 标准 (4层)':<30} {result_std['tps']:<15.2f} {result_std['time']*1000:<15.2f}")
if result_kda:
    print(f"{'PyTorch KDA (4层)':<30} {result_kda['tps']:<15.2f} {result_kda['time']*1000:<15.2f}")

if result_std and result_kda:
    speedup = result_kda['tps'] / result_std['tps']
    print(f"\n🔥 KDA vs 标准: {speedup:.2f}x")

if result_llama and result_kda:
    speedup = result_kda['tps'] / result_llama['tps']
    print(f"🔥 KDA vs llama.cpp: {speedup:.2f}x")

print("\n" + "="*70)
print("✅ 测试完成!")
print("="*70)

print("""
📋 测试说明:
   • llama.cpp: 原生完整推理 (28层)
   • PyTorch 标准: 4层标准 Attention
   • PyTorch KDA: 4层 KDA Attention
   
🔑 关键比较:
   • KDA 替换 Attention 后是否有加速
   • 端到端推理性能对比
""")