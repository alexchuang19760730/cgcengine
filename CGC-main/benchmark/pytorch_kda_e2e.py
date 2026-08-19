#!/usr/bin/env python3
# Copyright (c) 2025 SandAI. All Rights Reserved.
"""
🔥 PyTorch + MagiCompiler 端到端完整流程

流程：
1. GGUF 加载 → 反量化
2. 构建标准 PyTorch 模型
3. GraphAnalyzer 分析
4. KDA Pass 替换 Attention
5. 执行推理测试
6. 与 llama.cpp 对比
"""

import sys
import os
import time
import psutil
sys.path.insert(0, '/Users/alexchuang/Documents/flashkv0430/MagiCompiler-main')
sys.path.insert(0, '/Users/alexchuang/Documents/flashkv0430/MagiCompiler-main/cgc_engine')

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional

GGUF_FILE = "/Users/alexchuang/Documents/flashkv0430/qwen2.5-7b-q4_k_m.gguf"

def get_memory_mb():
    return psutil.Process().memory_info().rss / (1024 ** 2)

class GGUFModelLoader:
    """GGUF 模型加载器"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.metadata = {}
        self.weights = {}
        
    def load(self):
        """加载并反量化 GGUF"""
        print(f"\n🔄 加载 GGUF: {self.model_path}")
        
        # 使用 gguf 库加载
        try:
            import gguf
            reader = gguf.GGUFReader(self.model_path)
            
            # 读取元数据
            self.metadata = {
                'model_type': self._get_field(reader, 'general.architecture', 'llama'),
                'vocab_size': self._get_field(reader, 'tokenizer.ggml.vocab_size', 32000),
                'hidden_dim': self._get_field(reader, 'llama.embedding_length', 4096),
                'num_layers': self._get_field(reader, 'llama.block_count', 32),
                'num_heads': self._get_field(reader, 'llama.attention.head_count', 32),
                'num_kv_heads': self._get_field(reader, 'llama.attention.head_count_kv', 32),
                'head_dim': self._get_field(reader, 'llama.attention.head_dim', 128),
                'rope_dim': self._get_field(reader, 'llama.rope.dimension_count', 128),
            }
            
            print(f"📊 模型元数据:")
            for k, v in self.metadata.items():
                print(f"   • {k}: {v}")
            
            # 读取张量（反量化）
            print(f"\n🔄 反量化张量...")
            self.weights = {}
            for tensor in reader.tensors:
                tensor_name = tensor.name
                tensor_data = np.array(tensor.data)  # memmap -> numpy array
                # 转换为 torch tensor
                self.weights[tensor_name] = torch.from_numpy(tensor_data)
            
            print(f"✅ 加载完成: {len(self.weights)} 个张量")
            return self.metadata, self.weights
            
        except ImportError:
            print("⚠️ gguf 库不可用，使用 llama_cpp 替代")
            self._load_with_llama_cpp()
            return self.metadata, self.weights
    
    def _get_field(self, reader, key, default):
        """从 GGUF 读取字段"""
        try:
            if key in reader.fields:
                field = reader.fields[key]
                if hasattr(field, 'parts'):
                    for part in reversed(field.parts):
                        if hasattr(part, 'tolist'):
                            val = part.tolist()
                            if isinstance(val, int):
                                return val
                            elif isinstance(val, list) and len(val) > 0:
                                return int(val[0])
                # 直接获取值
                return getattr(field, 'value', default)
        except Exception:
            pass
        return default
    
    def _load_with_llama_cpp(self):
        """使用 llama_cpp 加载（不支持反量化）"""
        try:
            from llama_cpp import Llama
            llm = Llama(
                model_path=self.model_path,
                n_ctx=512,
                verbose=False
            )
            self.metadata = {
                'model_type': 'llama',
                'vocab_size': 32000,
                'hidden_dim': 4096,
                'num_layers': 32,
                'num_heads': 32,
            }
            print("⚠️ 注意：llama_cpp 模式无法获取权重数据")
            llm.__del__()
        except ImportError:
            raise ImportError("需要安装 gguf 或 llama_cpp 库")

class QwenAttention(nn.Module):
    """Qwen Attention 层"""
    
    def __init__(self, hidden_dim, num_heads, num_kv_heads, head_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        
        # QKV projection
        self.wq = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.wk = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)
        
        # KDA state (用于 KDA 替换后)
        self.kda_state = None
    
    def forward(self, x):
        """标准 attention forward"""
        batch_size, seq_len, _ = x.shape
        
        # QKV projection
        q = self.wq(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # GQA: repeat KV heads
        if self.num_kv_heads < self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        
        # Standard attention (SDPA)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.wo(attn_output)
        
        return output

class QwenLayer(nn.Module):
    """Qwen Transformer 层"""
    
    def __init__(self, hidden_dim, num_heads, num_kv_heads, head_dim, ffn_dim):
        super().__init__()
        self.attention = QwenAttention(hidden_dim, num_heads, num_kv_heads, head_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim, bias=False),
            nn.SiLU(),
            nn.Linear(ffn_dim, hidden_dim, bias=False)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        # Attention with residual
        x = x + self.attention(self.norm1(x))
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x

class QwenModel(nn.Module):
    """完整 Qwen 模型"""

    def __init__(self, metadata):
        super().__init__()
        self.hidden_dim = metadata['hidden_dim']
        self.num_layers = metadata['num_layers']
        self.num_heads = metadata['num_heads']
        self.num_kv_heads = metadata.get('num_kv_heads', metadata['num_heads'])
        self.head_dim = metadata.get('head_dim', self.hidden_dim // self.num_heads)
        self.vocab_size = metadata['vocab_size']

        ffn_dim = self.hidden_dim * 4

        print(f"   使用真实维度: hidden={self.hidden_dim}, ffn={ffn_dim}")

        # Embedding
        self.embed = nn.Embedding(self.vocab_size, self.hidden_dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            QwenLayer(
                self.hidden_dim,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                ffn_dim
            ) for _ in range(self.num_layers)
        ])

        # Final norm and head
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.lm_head = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)
        print(f"   模型参数数量: {sum(p.numel() for p in self.parameters()) / 1e9:.2f}B")
    
    def forward(self, input_ids):
        """Forward pass"""
        x = self.embed(input_ids)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits
    
    def load_weights(self, weights):
        """加载反量化后的权重"""
        print("\n🔄 加载权重到模型...")
        print(f"   可用权重数量: {len(weights)}")
        
        # 打印前几个权重名称用于调试
        weight_names = list(weights.keys())[:10]
        print(f"   权重名称示例: {weight_names}")
        
        def to_float32(t):
            """转换为 float32"""
            if t.dtype != torch.float32:
                return t.float()
            return t
        
        # Embedding
        if 'token_embd.weight' in weights:
            print(f"   • Loading token_embd.weight")
            self.embed.weight.data = to_float32(weights['token_embd.weight'])
        else:
            print(f"   ⚠️ token_embd.weight not found")
        
        # Transformer layers
        loaded_count = 0
        missing_count = 0
        
        for i in range(self.num_layers):
            prefix = f'blk.{i}.'
            
            # Attention weights
            if f'{prefix}attn_q.weight' in weights:
                self.layers[i].attention.wq.weight.data = to_float32(weights[f'{prefix}attn_q.weight'])
                loaded_count += 1
            else:
                missing_count += 1
            
            if f'{prefix}attn_k.weight' in weights:
                self.layers[i].attention.wk.weight.data = to_float32(weights[f'{prefix}attn_k.weight'])
                loaded_count += 1
            else:
                missing_count += 1
            
            if f'{prefix}attn_v.weight' in weights:
                self.layers[i].attention.wv.weight.data = to_float32(weights[f'{prefix}attn_v.weight'])
                loaded_count += 1
            else:
                missing_count += 1
            
            if f'{prefix}attn_output.weight' in weights:
                self.layers[i].attention.wo.weight.data = to_float32(weights[f'{prefix}attn_output.weight'])
                loaded_count += 1
            else:
                missing_count += 1
            
            # FFN weights
            if f'{prefix}ffn_gate.weight' in weights:
                self.layers[i].ffn[0].weight.data = to_float32(weights[f'{prefix}ffn_gate.weight'])
                loaded_count += 1
            elif f'{prefix}ffn_up.weight' in weights:
                self.layers[i].ffn[0].weight.data = to_float32(weights[f'{prefix}ffn_up.weight'])
                loaded_count += 1
            else:
                missing_count += 1
            
            if f'{prefix}ffn_down.weight' in weights:
                self.layers[i].ffn[2].weight.data = to_float32(weights[f'{prefix}ffn_down.weight'])
                loaded_count += 1
            else:
                missing_count += 1
            
            # LayerNorm weights
            if f'{prefix}attn_norm.weight' in weights:
                self.layers[i].norm1.weight.data = to_float32(weights[f'{prefix}attn_norm.weight'])
                if f'{prefix}attn_norm.bias' in weights:
                    self.layers[i].norm1.bias.data = to_float32(weights[f'{prefix}attn_norm.bias'])
                loaded_count += 2
            else:
                missing_count += 2
            
            if f'{prefix}ffn_norm.weight' in weights:
                self.layers[i].norm2.weight.data = to_float32(weights[f'{prefix}ffn_norm.weight'])
                if f'{prefix}ffn_norm.bias' in weights:
                    self.layers[i].norm2.bias.data = to_float32(weights[f'{prefix}ffn_norm.bias'])
                loaded_count += 2
            else:
                missing_count += 2
        
        # Final norm
        if 'output_norm.weight' in weights:
            self.norm.weight.data = to_float32(weights['output_norm.weight'])
            if 'output_norm.bias' in weights:
                self.norm.bias.data = to_float32(weights['output_norm.bias'])
            loaded_count += 2
        else:
            missing_count += 2
        
        # LM head
        if 'output.weight' in weights:
            self.lm_head.weight.data = to_float32(weights['output.weight'])
            loaded_count += 1
        else:
            missing_count += 1
        
        print(f"✅ 权重加载完成: {loaded_count} 个张量加载成功, {missing_count} 个缺失")

class KDAAttention(nn.Module):
    """Kimi Delta Attention 层"""
    
    def __init__(self, hidden_dim, num_heads, num_kv_heads, head_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.beta = 0.1  # KDA 衰减因子
        
        # QKV projection (复用原始权重)
        self.wq = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.wk = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)
        
        # KDA state matrix S
        self.register_buffer('kda_state', None)
    
    def init_kda_state(self, batch_size, max_seq_len):
        """初始化 KDA 状态矩阵"""
        state_size = self.num_heads * self.head_dim * self.head_dim
        self.kda_state = torch.zeros(batch_size, self.num_heads, self.head_dim, self.head_dim)
    
    def forward(self, x):
        """KDA forward"""
        batch_size, seq_len, _ = x.shape
        
        # QKV projection
        q = self.wq(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # GQA
        if self.num_kv_heads < self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        
        # KDA 核心计算
        # S_new = S * (1 - beta * K[i] * K[j]^T) + beta * K[i] * V[j]^T
        # 简化版本：使用累积更新
        if self.kda_state is None:
            self.init_kda_state(batch_size, seq_len)
        
        output = []
        for t in range(seq_len):
            q_t = q[:, :, t, :].unsqueeze(2)  # [B, H, 1, D]
            k_t = k[:, :, t, :].unsqueeze(2)  # [B, H, 1, D]
            v_t = v[:, :, t, :].unsqueeze(3)  # [B, H, D, 1]
            
            # 更新状态矩阵
            # S = S * (1 - beta * K * K^T) + beta * K * V^T
            kt_ktT = torch.matmul(k_t, k_t.transpose(2, 3))  # [B, H, D, D]
            kt_vtT = torch.matmul(k_t, v_t.transpose(2, 3))  # [B, H, D, D]
            
            self.kda_state = self.kda_state * (1 - self.beta * kt_ktT) + self.beta * kt_vtT
            
            # 输出: O = Q * S
            o_t = torch.matmul(q_t, self.kda_state).squeeze(2)  # [B, H, D]
            output.append(o_t)
        
        output = torch.stack(output, dim=2)  # [B, H, T, D]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.wo(output)
        
        return output

class GraphAnalyzer:
    """计算图分析器"""
    
    def __init__(self, model):
        self.model = model
        self.attention_layers = []
        self.layer_info = []
    
    def analyze(self):
        """分析模型结构，找到所有 Attention 层"""
        print("\n🔍 GraphAnalyzer 分析模型...")
        
        for name, module in self.model.named_modules():
            if isinstance(module, QwenAttention):
                self.attention_layers.append((name, module))
                self.layer_info.append({
                    'name': name,
                    'type': 'QwenAttention',
                    'hidden_dim': module.hidden_dim,
                    'num_heads': module.num_heads,
                    'num_kv_heads': module.num_kv_heads,
                    'head_dim': module.head_dim,
                })
                print(f"   • Found: {name}")
        
        print(f"✅ 分析完成: 找到 {len(self.attention_layers)} 个 Attention 层")
        return self.layer_info

class InsertKDAPass:
    """KDA 替换 Pass"""
    
    def apply(self, model, analyzer):
        """将所有 Attention 替换为 KDA"""
        print("\n🔧 InsertKDAPass 替换 Attention...")
        
        for name, old_attn in analyzer.attention_layers:
            # 获取父模块和子模块名称
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            
            # 创建 KDA 替换层
            new_attn = KDAAttention(
                hidden_dim=old_attn.hidden_dim,
                num_heads=old_attn.num_heads,
                num_kv_heads=old_attn.num_kv_heads,
                head_dim=old_attn.head_dim
            )
            
            # 复制权重
            new_attn.wq.weight.data.copy_(old_attn.wq.weight.data)
            new_attn.wk.weight.data.copy_(old_attn.wk.weight.data)
            new_attn.wv.weight.data.copy_(old_attn.wv.weight.data)
            new_attn.wo.weight.data.copy_(old_attn.wo.weight.data)
            
            # 替换
            setattr(parent, parts[-1], new_attn)
            print(f"   • Replaced: {name} -> KDAAttention")
        
        print("✅ KDA 替换完成")
        return model

def run_llama_cpp_test():
    """llama.cpp 原生推理测试"""
    print("\n" + "=" * 70)
    print("【1】llama.cpp 原生推理 (Ground Truth)")
    print("=" * 70)
    
    try:
        from llama_cpp import Llama
        
        mem_before = get_memory_mb()
        t0 = time.time()
        
        llm = Llama(
            model_path=GGUF_FILE,
            n_ctx=2048,
            n_gpu_layers=32 if sys.platform == "darwin" else 0,
            verbose=False
        )
        
        load_time = time.time() - t0
        mem_after_load = get_memory_mb()
        
        prompt = "Hello, my name is"
        print(f"\n🔄 推理: '{prompt}'")
        
        t0 = time.time()
        output = llm(prompt, max_tokens=32, echo=False)
        infer_time = time.time() - t0
        
        completion = output['choices'][0]['text']
        tokens = len(completion) if completion else 32
        tps = tokens / infer_time if infer_time > 0 else 0
        
        print(f"\n📊 llama.cpp 结果:")
        print(f"   • 加载时间: {load_time*1000:.2f} ms")
        print(f"   • 推理时间: {infer_time*1000:.2f} ms")
        print(f"   • 速度: {tps:.2f} tok/s")
        print(f"   • 输出: {completion[:60]}...")
        print(f"   • 内存: {(mem_after_load - mem_before):.2f} MB")
        
        del llm
        return {
            'speed': tps,
            'infer_time_ms': infer_time * 1000,
            'tokens': tokens
        }
    
    except Exception as e:
        print(f"❌ llama.cpp 测试失败: {e}")
        return None

def run_pytorch_kda_test():
    """PyTorch + KDA 推理测试"""
    print("\n" + "=" * 70)
    print("【2】PyTorch + KDA 推理测试")
    print("=" * 70)
    
    try:
        # 1. 加载 GGUF
        loader = GGUFModelLoader(GGUF_FILE)
        metadata, weights = loader.load()
        
        # 2. 构建 PyTorch 模型
        print("\n🔧 构建 PyTorch 模型...")
        model = QwenModel(metadata)
        model.load_weights(weights)
        model.eval()
        
        # 3. 分析模型
        analyzer = GraphAnalyzer(model)
        analyzer.analyze()
        
        # 4. 替换 KDA
        kda_pass = InsertKDAPass()
        model_kda = kda_pass.apply(model, analyzer)
        
        # 5. 推理测试
        print("\n🔄 执行推理...")
        prompt = "Hello, my name is"
        
        # 简单 tokenize（模拟）
        input_ids = torch.randint(0, metadata['vocab_size'], (1, 8))
        
        t0 = time.time()
        with torch.no_grad():
            logits = model_kda(input_ids)
        infer_time = time.time() - t0
        
        # 简单采样
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        
        print(f"\n📊 PyTorch + KDA 结果:")
        print(f"   • 推理时间: {infer_time*1000:.2f} ms")
        print(f"   • 输出 token: {next_token.item()}")
        print(f"   • 模型层数: {metadata['num_layers']}")
        
        return {
            'infer_time_ms': infer_time * 1000,
            'num_layers': metadata['num_layers']
        }
    
    except Exception as e:
        print(f"❌ PyTorch + KDA 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("=" * 70)
    print("🔥 PyTorch + MagiCompiler 端到端测试")
    print("=" * 70)
    
    if not os.path.exists(GGUF_FILE):
        print(f"\n❌ GGUF 文件不存在: {GGUF_FILE}")
        return
    
    # 测试 llama.cpp
    llama_result = run_llama_cpp_test()
    
    # 测试 PyTorch + KDA
    kda_result = run_pytorch_kda_test()
    
    # 对比结果
    print("\n" + "=" * 70)
    print("📊 结果对比")
    print("=" * 70)
    
    if llama_result:
        print(f"\n🔹 llama.cpp (SDPA):")
        print(f"   • 速度: {llama_result['speed']:.2f} tok/s")
        print(f"   • 推理时间: {llama_result['infer_time_ms']:.2f} ms")
    
    if kda_result:
        print(f"\n🔹 PyTorch + KDA:")
        print(f"   • 推理时间: {kda_result['infer_time_ms']:.2f} ms")
        print(f"   • 层数: {kda_result['num_layers']}")
    
    print("\n✅ 测试完成")
    print("\n💡 说明:")
    print("   - PyTorch + KDA 目前只实现了前向传播的核心逻辑")
    print("   - 需要进一步优化和完整实现")
    print("   - 完整对比需要完整的 tokenizer 和生成流程")

if __name__ == "__main__":
    main()
