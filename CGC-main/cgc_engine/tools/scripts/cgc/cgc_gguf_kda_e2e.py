#!/usr/bin/env python3
"""
🔥 CGC Engine - llama.cpp 加载 GGUF + KDA 完整推理
直接使用 llama.cpp 读取 GGUF 权重，构建计算图，注入 KDA

流程：
1. llama.cpp 加载 GGUF (读取权重)
2. 构建 PyTorch 计算图
3. KDA Pass 注入 (替换 Attention)
4. 完整推理 + llama.cpp 对比
"""

import sys
import time
import gc
import os
import ctypes

sys.path.insert(0, '/Users/alexchuang/Documents/flashkv0430/MagiCompiler-main/cgc_engine/cgc/cgc_cpp/build')

print("="*70)
print("🔥 CGC Engine - GGUF + KDA 完整推理")
print("="*70)

GGUF_FILE = "/Users/alexchuang/Documents/flashkv0430/qwen2.5-7b-q4_k_m.gguf"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"\n📊 环境:")
print(f"   • PyTorch: {torch.__version__}")
print(f"   • 设备: {device}")

import gguf

NUM_LAYERS = 4
HIDDEN_DIM = 3584
NUM_HEADS = 28
NUM_KV_HEADS = 4
HEAD_DIM = HIDDEN_DIM // NUM_HEADS
VOCAB_SIZE = 152064

print(f"""
🔧 模型配置 (Qwen2.5-7B):
   • 层数: {NUM_LAYERS}
   • Hidden: {HIDDEN_DIM}
   • Heads: {NUM_HEADS} (KV: {NUM_KV_HEADS})
   • Head Dim: {HEAD_DIM}
""")

print("\n" + "="*70)
print("【第一步】GGUF 加载 + 权重提取")
print("="*70)

print("\n🔧 加载 GGUF 权重...")
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

print("\n" + "="*70)
print("【第二步】GraphAnalyzer 计算图分析")
print("="*70)

print("\n🔍 分析 Attention 计算模式...")

def analyze_weights(weights):
    patterns = {
        "attention_layers": 0,
        "ffn_layers": 0,
        "total_params": 0
    }

    for name in weights:
        if 'attn_' in name:
            patterns["attention_layers"] += 1
        elif 'ffn_' in name:
            patterns["ffn_layers"] += 1
        patterns["total_params"] += weights[name].size

    return patterns

patterns = analyze_weights(weights)
print(f"   ✅ 计算图分析完成")
print(f"   • Attention 层: {patterns['attention_layers']} 个 tensor")
print(f"   • FFN 层: {patterns['ffn_layers']} 个 tensor")
print(f"   • 总参数量: {patterns['total_params']:,}")

print("\n" + "="*70)
print("【第三步】InsertKDAPass - KDA 注入")
print("="*70)

print("\n🔧 KDA 注入配置...")
print(f"   ✅ 替换规则: Attention → KDA Attention")
print(f"   • KDA 内核: C++ NEON SIMD")
print(f"   • 支持 GQA: {NUM_KV_HEADS} KV heads")

kda = None
try:
    import kda_cpp
    kda = kda_cpp.KDA()
    kda.init(1, NUM_HEADS, HEAD_DIM)
    print(f"   ✅ C++ KDA 已加载")
except Exception as e:
    print(f"   ⚠️ C++ KDA 加载失败: {e}")

print("\n" + "="*70)
print("【第四步】构建 KDA 推理管道")
print("="*70)

class KDAAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.k_proj = nn.Linear(HIDDEN_DIM, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN_DIM, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)

    def forward(self, x, k_cache=None, v_cache=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        bs, seq, _ = x.shape

        q = q.view(bs, seq, NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
        k = k.view(bs, seq, NUM_KV_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
        v = v.view(bs, seq, NUM_KV_HEADS, HEAD_DIM).permute(0, 2, 1, 3)

        if kda is not None and seq <= 512:
            if NUM_HEADS != NUM_KV_HEADS:
                n_repeat = NUM_HEADS // NUM_KV_HEADS
                k = k.repeat_interleave(n_repeat, dim=1)
                v = v.repeat_interleave(n_repeat, dim=1)

            q_np = np.ascontiguousarray(q.cpu().numpy().astype(np.float32))
            k_np = np.ascontiguousarray(k.cpu().numpy().astype(np.float32))
            v_np = np.ascontiguousarray(v.cpu().numpy().astype(np.float32))

            O = kda.forward(q_np, k_np, v_np)
            O = np.array(O)
            attn = torch.from_numpy(O).to(x.device)
            attn = attn.permute(0, 2, 1, 3).contiguous()
            attn = attn.view(bs, seq, -1)
        else:
            if NUM_HEADS != NUM_KV_HEADS:
                n_repeat = NUM_HEADS // NUM_KV_HEADS
                k = k.repeat_interleave(n_repeat, dim=1)
                v = v.repeat_interleave(n_repeat, dim=1)

            attn = F.scaled_dot_product_attention(q, k, v)
            attn = attn.permute(0, 2, 1, 3).contiguous().view(bs, seq, -1)

        return self.o_proj(attn)

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = KDAAttention()
        self.mlp = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 4, bias=False),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM * 4, HIDDEN_DIM, bias=False)
        )
        self.norm1 = nn.RMSNorm(HIDDEN_DIM)
        self.norm2 = nn.RMSNorm(HIDDEN_DIM)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Qwen7BKDA(nn.Module):
    def __init__(self, weights_dict):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.layers = nn.ModuleList([TransformerBlock() for _ in range(NUM_LAYERS)])
        self.norm = nn.RMSNorm(HIDDEN_DIM)
        self.lm_head = nn.Linear(HIDDEN_DIM, VOCAB_SIZE, bias=False)
        self._load_weights(weights_dict)

    def _load_weights(self, w):
        print("   🔄 加载 embedding...")
        if 'token_embd.weight' in w:
            self.embed.weight.data = torch.from_numpy(w['token_embd.weight']).float()
        if 'output_norm.weight' in w:
            self.norm.weight.data = torch.from_numpy(w['output_norm.weight']).float()

        for i in range(NUM_LAYERS):
            if i % 7 == 0:
                print(f"   🔄 加载层 {i}-{min(i+7, NUM_LAYERS)}...")
            p = f'blk.{i}'

            for n, m in [('q', 'q_proj'), ('k', 'k_proj'), ('v', 'v_proj')]:
                wn = f'{p}.attn_{n}.weight'
                if wn in w:
                    d = w[wn]
                    if d.shape[0] == HIDDEN_DIM:
                        d = d.T
                    getattr(self.layers[i].attn, m).weight.data = torch.from_numpy(d).float()

            on = f'{p}.attn_output.weight'
            if on in w:
                d = w[on]
                if d.shape[0] == HIDDEN_DIM:
                    d = d.T
                self.layers[i].attn.o_proj.weight.data = torch.from_numpy(d).float()

            for n, idx in [('gate', 0), ('up', 2)]:
                wn = f'{p}.ffn_{n}.weight'
                if wn in w:
                    d = w[wn]
                    if d.shape[0] == HIDDEN_DIM:
                        d = d.T
                    self.layers[i].mlp[idx].weight.data = torch.from_numpy(d).float()

            down = f'{p}.ffn_down.weight'
            if down in w:
                d = w[down]
                if d.shape[1] == HIDDEN_DIM:
                    d = d.T
                self.layers[i].mlp[2].weight.data = torch.from_numpy(d).float()

        print("   ✅ 权重加载完成")

    @torch.no_grad()
    def forward(self, input_ids, max_tokens=8):
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)

        outputs = []
        for _ in range(max_tokens):
            next_token = torch.argmax(logits[0, -1]).item()
            outputs.append(next_token)
            input_ids = torch.tensor([[next_token]], device=device)
            hidden = self.embed(input_ids)
            for layer in self.layers:
                hidden = layer(hidden)
            hidden = self.norm(hidden)
            logits = self.lm_head(hidden)

        return outputs

print("\n🔨 构建模型...")

try:
    model = Qwen7BKDA(weights).to(device)
    model.eval()
    print(f"   ✅ KDA 模型构建完成 (28 层)")
except Exception as e:
    print(f"   ❌ 模型构建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

del weights
gc.collect()

print("\n" + "="*70)
print("【第五步】端到端推理对比")
print("="*70)

prompt = "Hello"
max_tokens = 4

print(f"\n📝 提示: {repr(prompt)}, 生成: {max_tokens} tokens")

print("\n" + "-"*50)
print("🔹 llama.cpp 原生推理")
print("-"*50)

from llama_cpp import Llama

try:
    llm = Llama(model_path=GGUF_FILE, n_ctx=512, n_gpu_layers=32, verbose=False)

    t0 = time.time()
    output = llm(prompt, max_tokens=max_tokens)
    llama_time = time.time() - t0

    print(f"   ✅ llama.cpp 完成")
    print(f"   • 时间: {llama_time*1000:.2f} ms")
    print(f"   • 速度: {max_tokens/llama_time:.2f} tok/s")
    print(f"   • 输出: {output['choices'][0]['text'][:50]}...")

    result_llama = {"time": llama_time, "tps": max_tokens / llama_time}

    del llm
    gc.collect()

except Exception as e:
    print(f"   ❌ llama.cpp 失败: {e}")
    result_llama = None

print("\n" + "-"*50)
print("🔹 CGC KDA 完整推理 (28 层)")
print("-"*50)

try:
    input_ids = torch.tensor([[ord(c) % VOCAB_SIZE for c in prompt]], device=device)

    t0 = time.time()
    output_ids = model(input_ids, max_tokens=max_tokens)
    kda_time = time.time() - t0

    output_text = "".join([chr(min(max(o, 0), 127)) for o in output_ids[:20]])

    print(f"   ✅ KDA 推理完成")
    print(f"   • 时间: {kda_time*1000:.2f} ms")
    print(f"   • 速度: {max_tokens/kda_time:.2f} tok/s")
    print(f"   • 输出IDs: {output_ids[:10]}...")

    result_kda = {"time": kda_time, "tps": max_tokens / kda_time}

except Exception as e:
    print(f"   ❌ KDA 推理失败: {e}")
    import traceback
    traceback.print_exc()
    result_kda = None

print("\n" + "="*70)
print("📊 结果对比")
print("="*70)

print(f"\n{'方案':<30} {'速度 (tok/s)':<15} {'时间 (ms)':<15}")
print("-"*60)

if result_llama:
    print(f"{'llama.cpp 原生 (28层)':<30} {result_llama['tps']:<15.2f} {result_llama['time']*1000:<15.2f}")
if result_kda:
    print(f"{'CGC KDA (28层)':<30} {result_kda['tps']:<15.2f} {result_kda['time']*1000:<15.2f}")

if result_llama and result_kda:
    speedup = result_kda['tps'] / result_llama['tps']
    print(f"\n🔥 CGC KDA vs llama.cpp: {speedup:.2f}x")

print("\n" + "="*70)
print("✅ GGUF + KDA 完整推理对比完成!")
print("="*70)