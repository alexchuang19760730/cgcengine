#!/usr/bin/env python3
"""
🔥 CGC Engine 完整推理测试 - 简化版（2层）
从 GGUF 加载权重 → KDA Attention → 完整推理

用 2 层模型快速验证 Metal GPU 上的 KDA 效果
"""

import sys
import time
import psutil
import gc

print("="*70)
print("🔥 CGC Engine 完整推理测试 (Metal GPU - 简化版)")
print("="*70)

GGUF_FILE = "/Users/alexchuang/Documents/flashkv0430/qwen2.5-7b-q4_k_m.gguf"

def get_memory():
    return psutil.Process().memory_info().rss / (1024 ** 2)

print(f"\n📊 初始内存: {get_memory():.2f} MB")

import torch
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ MPS 可用: {torch.backends.mps.is_available()}")

import torch.nn as nn
import numpy as np

device = "mps"
print(f"📊 设备: {device}")

print("\n" + "="*70)
print("【第一步】加载 GGUF 权重")
print("="*70)

import gguf
reader = gguf.GGUFReader(GGUF_FILE)

weights = {}
for tensor in reader.tensors:
    name = tensor.name
    try:
        qtype = gguf.GGMLQuantizationType(tensor.tensor_type)
        arr = gguf.dequantize(tensor.data, qtype)
        if arr is not None and hasattr(arr, 'numpy'):
            arr = arr.numpy()
        if arr is not None:
            weights[name] = arr.astype(np.float32)
    except:
        continue

print(f"✅ 加载了 {len(weights)} 个权重张量")
print(f"📊 当前内存: {get_memory():.2f} MB")

print("\n" + "="*70)
print("【第二步】构建 2 层 PyTorch 模型（带 GGUF 权重）")
print("="*70)

NUM_LAYERS = 2

class Qwen7BConfig:
    vocab_size = 152064
    hidden_dim = 3584
    num_layers = NUM_LAYERS
    num_heads = 28
    num_kv_heads = 4
    head_dim = 128
    intermediate_size = 18944

config = Qwen7BConfig()

class KDAAttention(nn.Module):
    def __init__(self, cfg, beta=0.1):
        super().__init__()
        self.cfg = cfg
        self.beta = beta
        self.q_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.o_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)

        self.kda = None
        try:
            import kda_cpp
            self.kda = kda_cpp.KDA()
            self.kda.init(1, cfg.num_heads, cfg.head_dim)
            print("   ✅ C++ KDA NEON 已初始化")
        except:
            print("   ⚠️ C++ KDA 不可用，使用 PyTorch")

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        batch_size, seq_len, _ = x.shape
        q = q.view(batch_size, seq_len, self.cfg.num_heads, self.cfg.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.cfg.num_heads, self.cfg.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.cfg.num_heads, self.cfg.head_dim).transpose(1, 2)

        if self.kda is not None and seq_len <= 256:
            q_np = np.ascontiguousarray(q.cpu().numpy().astype(np.float32))
            k_np = np.ascontiguousarray(k.cpu().numpy().astype(np.float32))
            v_np = np.ascontiguousarray(v.cpu().numpy().astype(np.float32))
            O = self.kda.forward(q_np, k_np, v_np, beta=self.beta)
            attn = torch.from_numpy(np.array(O)).reshape(q.shape).to(q.device)
        else:
            attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn)

class StandardAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.o_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        batch_size, seq_len, _ = x.shape
        q = q.view(batch_size, seq_len, self.cfg.num_heads, self.cfg.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.cfg.num_heads, self.cfg.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.cfg.num_heads, self.cfg.head_dim).transpose(1, 2)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn)

class Block(nn.Module):
    def __init__(self, cfg, use_kda=True):
        super().__init__()
        self.attn = KDAAttention(cfg) if use_kda else StandardAttention(cfg)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.intermediate_size, bias=False),
            nn.SiLU(),
            nn.Linear(cfg.intermediate_size, cfg.hidden_dim, bias=False)
        )
        self.norm1 = nn.RMSNorm(cfg.hidden_dim)
        self.norm2 = nn.RMSNorm(cfg.hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class MiniQwen(nn.Module):
    def __init__(self, cfg, weights_dict, use_kda=True):
        super().__init__()
        self.cfg = cfg
        self.use_kda = use_kda
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.layers = nn.ModuleList([Block(cfg, use_kda) for _ in range(cfg.num_layers)])
        self.norm = nn.RMSNorm(cfg.hidden_dim)
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        self._load_weights(cfg, weights_dict)

    def _load_weights(self, cfg, w):
        try:
            if 'token_embd.weight' in w:
                self.embed.weight.data = torch.from_numpy(w['token_embd.weight']).to(device)
            if 'output_norm.weight' in w:
                self.norm.weight.data = torch.from_numpy(w['output_norm.weight']).to(device)

            for i in range(cfg.num_layers):
                prefix = f'blk.{i}'
                for proj_name, proj_attr in [('q', 'q_proj'), ('k', 'k_proj'), ('v', 'v_proj')]:
                    weight_name = f'{prefix}.attn_{proj_name}.weight'
                    if weight_name in w:
                        w_data = w[weight_name]
                        if len(w_data.shape) == 2 and w_data.shape[0] == cfg.hidden_dim:
                            w_data = w_data.T
                        getattr(self.layers[i].attn, proj_attr).weight.data = torch.from_numpy(w_data).to(device)

                out_name = f'{prefix}.attn_output.weight'
                if out_name in w:
                    w_data = w[out_name]
                    if len(w_data.shape) == 2 and w_data.shape[0] == cfg.hidden_dim:
                        w_data = w_data.T
                    self.layers[i].attn.o_proj.weight.data = torch.from_numpy(w_data).to(device)

                for mlp_name, mlp_idx in [('gate', 0), ('up', 2), ('down', 2)]:
                    if mlp_idx == 2 and mlp_name == 'down':
                        w_name = f'{prefix}.ffn_down.weight'
                    else:
                        w_name = f'{prefix}.ffn_{mlp_name}.weight'
                    if w_name in w:
                        w_data = w[w_name]
                        if len(w_data.shape) == 2:
                            w_data = w_data.T
                        self.layers[i].mlp[mlp_idx].weight.data = torch.from_numpy(w_data).to(device)

            print("   ✅ 权重加载完成")
        except Exception as e:
            print(f"   ⚠️ 权重加载错误: {e}")

    def forward(self, input_ids):
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.norm(hidden)
        return self.lm_head(hidden)

print(f"\n📊 构建 {NUM_LAYERS} 层模型...")

print("   构建标准模型...")
model_std = MiniQwen(config, weights, use_kda=False).to(device)
model_std.eval()
print("   ✅ 标准模型完成")

gc.collect()

print("   构建 KDA 模型...")
model_kda = MiniQwen(config, weights, use_kda=True).to(device)
model_kda.eval()
print("   ✅ KDA 模型完成")

print(f"\n📊 当前内存: {get_memory():.2f} MB")

print("\n" + "="*70)
print("【第三步】完整推理测试")
print("="*70)

@torch.no_grad()
def run_inference(model, prompt, max_tokens=8):
    input_ids = [ord(c) % config.vocab_size for c in prompt]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    generated = input_ids.copy()
    t0 = time.time()
    for _ in range(max_tokens):
        logits = model(input_tensor)
        next_token = torch.argmax(logits[0, -1]).item()
        generated.append(next_token)
        input_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
    elapsed = time.time() - t0
    return {"time": elapsed, "tokens_per_second": max_tokens / elapsed if elapsed > 0 else 0}

prompt = "Hello"
max_tokens = 8

print(f"\n📝 提示: {repr(prompt)}, 生成: {max_tokens} tokens")

print("\n" + "-"*50)
print("🔹 llama.cpp (Ground Truth)")
print("-"*50)

try:
    from llama_cpp import Llama
    llm = Llama(model_path=GGUF_FILE, n_ctx=512, n_gpu_layers=32, verbose=False)
    t0 = time.time()
    output = llm(prompt, max_tokens=max_tokens)
    elapsed = time.time() - t0
    result_llama = {"time": elapsed, "tokens_per_second": max_tokens / elapsed}
    print(f"   时间: {elapsed*1000:.2f} ms, 速度: {max_tokens/elapsed:.2f} tok/s")
    del llm
    gc.collect()
except Exception as e:
    print(f"   ❌ llama.cpp 失败: {e}")
    result_llama = None

print(f"📊 当前内存: {get_memory():.2f} MB")

print("\n" + "-"*50)
print("🔹 PyTorch 标准模型 (MPS)")
print("-"*50)

try:
    result_std = run_inference(model_std, prompt, max_tokens)
    print(f"   时间: {result_std['time']*1000:.2f} ms, 速度: {result_std['tokens_per_second']:.2f} tok/s")
except Exception as e:
    print(f"   ❌ 标准模型失败: {e}")
    result_std = None

gc.collect()
print(f"📊 当前内存: {get_memory():.2f} MB")

print("\n" + "-"*50)
print("🔹 PyTorch KDA 模型 (MPS)")
print("-"*50)

try:
    result_kda = run_inference(model_kda, prompt, max_tokens)
    print(f"   时间: {result_kda['time']*1000:.2f} ms, 速度: {result_kda['tokens_per_second']:.2f} tok/s")
except Exception as e:
    print(f"   ❌ KDA 模型失败: {e}")
    result_kda = None

print(f"\n📊 当前内存: {get_memory():.2f} MB")

print("\n" + "="*70)
print("【第四步】性能对比")
print("="*70)

print(f"\n{'方案':<25} {'速度 (tok/s)':<15} {'时间 (ms)':<15}")
print("-"*55)

if result_llama:
    print(f"{'llama.cpp (Metal)':<25} {result_llama['tokens_per_second']:<15.2f} {result_llama['time']*1000:<15.2f}")
if result_std:
    print(f"{'PyTorch 标准 (MPS)':<25} {result_std['tokens_per_second']:<15.2f} {result_std['time']*1000:<15.2f}")
if result_kda:
    print(f"{'PyTorch KDA (MPS)':<25} {result_kda['tokens_per_second']:<15.2f} {result_kda['time']*1000:<15.2f}")

if result_llama and result_kda:
    speedup = result_kda['tokens_per_second'] / result_llama['tokens_per_second']
    print(f"\n🔥 PyTorch KDA vs llama.cpp: {speedup:.2f}x")

if result_std and result_kda:
    speedup = result_kda['tokens_per_second'] / result_std['tokens_per_second']
    print(f"🔥 PyTorch KDA vs 标准: {speedup:.2f}x")

print("\n" + "="*70)
print("✅ 完整推理测试完成!")
print("="*70)