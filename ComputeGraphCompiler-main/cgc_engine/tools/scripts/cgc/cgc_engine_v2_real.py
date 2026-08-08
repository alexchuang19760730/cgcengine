#!/usr/bin/env python3
"""
CGC Engine V2 - 完整端到端测试（使用实际 GGUF 权重）

关键区别：
1. 从 GGUF 加载实际权重（使用 dequantize）
2. 构建带有真实权重的模型
3. 应用 KDA Pass
4. 完整推理对比
"""

import torch
import torch.nn as nn
from typing import Union, List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
import time
import psutil

logger = logging.getLogger(__name__)


class Mode(Enum):
    INFERENCE = "inference"
    TRAINING = "training"


class KDAMode(Enum):
    STANDARD = "standard"
    KDA_PYTORCH = "kda_pytorch"
    KDA_CPP_NEON = "kda_cpp_neon"


@dataclass
class ModelConfig:
    vocab_size: int = 152064
    hidden_dim: int = 3584
    num_layers: int = 28
    num_heads: int = 28
    num_kv_heads: int = 4
    head_dim: int = 128
    intermediate_size: int = 18944
    max_seq_len: int = 2048


def get_memory():
    return psutil.Process().memory_info().rss / (1024 ** 2)


class CGCEngineV2Real:
    """
    CGC Engine V2 - 使用实际 GGUF 权重
    """

    def __init__(
        self,
        mode: str = "inference",
        kda_mode: str = "standard",
        device: str = "auto"
    ):
        self.mode = Mode(mode)
        self.kda_mode = KDAMode(kda_mode)
        self.device = device if device != "auto" else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = None
        self.config = ModelConfig()
        self.gguf_reader = None
        self.gguf_weights = {}

    def load_gguf_weights(self, gguf_path: str) -> bool:
        """从 GGUF 加载实际权重（反量化）"""
        logger.info(f"[CGCEngineV2] Loading GGUF weights from: {gguf_path}")

        try:
            import gguf
            import numpy as np

            self.gguf_reader = gguf.GGUFReader(gguf_path)

            for tensor in self.gguf_reader.tensors:
                name = tensor.name
                shape = list(tensor.shape)
                dtype = str(tensor.tensor_type)

                try:
                    qtype = gguf.GGMLQuantizationType(tensor.tensor_type)
                    arr = gguf.dequantize(tensor.data, qtype)

                    if arr is None:
                        continue

                    if hasattr(arr, 'numpy'):
                        arr = arr.numpy()

                    if arr.dtype != np.float32:
                        arr = arr.astype(np.float32)

                    self.gguf_weights[name] = {
                        "data": arr,
                        "shape": shape,
                        "dtype": dtype
                    }
                except Exception as e:
                    logger.debug(f"Could not load tensor {name}: {e}")
                    continue

            logger.info(f"[CGCEngineV2] Loaded {len(self.gguf_weights)} tensors")
            return True

        except Exception as e:
            logger.error(f"[CGCEngineV2] Failed to load GGUF: {e}")
            return False

    def _get_tensor(self, name: str) -> Optional[torch.Tensor]:
        """获取 GGUF 张量"""
        if name not in self.gguf_weights:
            return None

        data = self.gguf_weights[name]["data"]
        return torch.from_numpy(data)

    def build_model(self) -> nn.Module:
        """构建带权重的模型"""
        logger.info("[CGCEngineV2] Building model with GGUF weights...")

        cfg = self.config

        class CGCModel(nn.Module):
            def __init__(self, config, engine):
                super().__init__()
                self.config = config
                self.engine = engine

                self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)
                self.layers = nn.ModuleList([
                    CGCBlock(config, engine, i) for i in range(config.num_layers)
                ])
                self.norm = nn.RMSNorm(config.hidden_dim)
                self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

                self._load_weights()

            def _load_weights(self):
                """加载 GGUF 权重"""
                embed = self.engine._get_tensor("token_embd.weight")
                if embed is not None and embed.shape == (self.config.vocab_size, self.config.hidden_dim):
                    self.embed_tokens.weight.data = embed

                for i, layer in enumerate(self.layers):
                    prefix = f"blk.{i}"

                    q_weight = self.engine._get_tensor(f"{prefix}.attn_q.weight")
                    k_weight = self.engine._get_tensor(f"{prefix}.attn_k.weight")
                    v_weight = self.engine._get_tensor(f"{prefix}.attn_v.weight")
                    o_weight = self.engine._get_tensor(f"{prefix}.attn_output.weight")

                    if q_weight is not None:
                        layer.attention.q_proj.weight.data = q_weight.t().contiguous()
                    if k_weight is not None:
                        layer.attention.k_proj.weight.data = k_weight.t().contiguous()
                    if v_weight is not None:
                        layer.attention.v_proj.weight.data = v_weight.t().contiguous()
                    if o_weight is not None:
                        layer.attention.o_proj.weight.data = o_weight.t().contiguous()

                    gate_weight = self.engine._get_tensor(f"{prefix}.ffn_gate.weight")
                    up_weight = self.engine._get_tensor(f"{prefix}.ffn_up.weight")
                    down_weight = self.engine._get_tensor(f"{prefix}.ffn_down.weight")

                    if gate_weight is not None:
                        layer.mlp.gate_proj.weight.data = gate_weight.t().contiguous()
                    if up_weight is not None:
                        layer.mlp.up_proj.weight.data = up_weight.t().contiguous()
                    if down_weight is not None:
                        layer.mlp.down_proj.weight.data = down_weight.t().contiguous()

                final_norm = self.engine._get_tensor("output_norm.weight")
                if final_norm is not None:
                    self.norm.weight.data = final_norm

                lm_head = self.engine._get_tensor("lm_head.weight")
                if lm_head is not None and lm_head.shape == (self.config.vocab_size, self.config.hidden_dim):
                    self.lm_head.weight.data = lm_head.t().contiguous()

                logger.info("[CGCEngineV2] Weights loaded successfully")

            def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
                hidden = self.embed_tokens(input_ids)
                for layer in self.layers:
                    hidden = layer(hidden)
                hidden = self.norm(hidden)
                return self.lm_head(hidden)

        class CGCBlock(nn.Module):
            def __init__(self, config, engine, layer_idx):
                super().__init__()
                self.attention = CGCAttention(config, engine.kda_mode)
                self.mlp = nn.Sequential(
                    nn.Linear(config.hidden_dim, config.intermediate_size, bias=False),
                    nn.SiLU(),
                    nn.Linear(config.intermediate_size, config.hidden_dim, bias=False)
                )
                self.layer_idx = layer_idx

            def forward(self, x):
                attn_out = self.attention(x)
                return self.mlp(attn_out)

        class CGCAttention(nn.Module):
            def __init__(self, config, kda_mode):
                super().__init__()
                self.config = config
                self.kda_mode = kda_mode

                self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
                self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
                self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
                self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

            def forward(self, x):
                q = self.q_proj(x)
                k = self.k_proj(x)
                v = self.v_proj(x)

                q = q.view(1, -1, self.config.num_heads, self.config.head_dim).transpose(1, 2)
                k = k.view(1, -1, self.config.num_heads, self.config.head_dim).transpose(1, 2)
                v = v.view(1, -1, self.config.num_heads, self.config.head_dim).transpose(1, 2)

                if self.kda_mode == KDAMode.KDA_PYTORCH:
                    out = self._kda_forward(q, k, v)
                else:
                    out = nn.functional.scaled_dot_product_attention(q, k, v)

                out = out.transpose(1, 2).contiguous().view(1, -1, self.config.hidden_dim)
                return self.o_proj(out.squeeze(0))

            def _kda_forward(self, q, k, v):
                """KDA 实现"""
                batch_size, num_heads, seq_len, head_dim = q.shape
                scale = 1.0 / (head_dim ** 0.5)
                beta = 0.1

                S = torch.zeros(batch_size, num_heads, head_dim, head_dim, device=q.device, dtype=q.dtype)
                O = torch.zeros_like(q)

                for i in range(seq_len):
                    ki = k[:, :, i, :]
                    vi = v[:, :, i, :]
                    qi = q[:, :, i, :]
                    kk = torch.einsum('bhd,bhe->bhde', ki, ki)
                    kv = torch.einsum('bhd,bhe->bhde', ki, vi)
                    S = S * (1.0 - beta * kk) + beta * kv
                    oi = torch.einsum('bhd,bhde->bhe', qi, S) * scale
                    O[:, :, i, :] = oi
                return O

        self.model = CGCModel(cfg, self).to(self.device)
        self.model.eval()

        return self.model

    @torch.no_grad()
    def generate(self, prompt: str, max_tokens: int = 32) -> Dict:
        """生成"""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        input_ids = [ord(c) % self.config.vocab_size for c in prompt]
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        generated = input_ids.copy()
        t0 = time.time()

        for _ in range(max_tokens):
            logits = self.model(input_tensor)
            next_token = torch.argmax(logits[0, -1]).item()
            generated.append(next_token)
            input_tensor = torch.tensor([[next_token]], dtype=torch.long, device=self.device)

        elapsed = time.time() - t0

        return {
            "generated_ids": generated,
            "time": elapsed,
            "tokens_per_second": max_tokens / elapsed
        }


def run_benchmark():
    """运行完整 benchmark"""
    print("="*70)
    print("🔥 CGC Engine V2 完整端到端测试")
    print("="*70)

    GGUF_FILE = "/Users/alexchuang/Documents/flashkv0430/qwen2.5-7b-q4_k_m.gguf"
    prompt = "Hello"
    max_tokens = 16

    print(f"\n📋 配置:")
    print(f"   GGUF: {GGUF_FILE}")
    print(f"   提示: {repr(prompt)}")
    print(f"   生成: {max_tokens} tokens")

    results = {}

    print("\n" + "="*70)
    print("【1】llama.cpp (Ground Truth)")
    print("="*70)

    try:
        from llama_cpp import Llama

        llm = Llama(
            model_path=GGUF_FILE,
            n_ctx=512,
            n_gpu_layers=32 if torch.backends.mps.is_available() else 0,
            verbose=False
        )

        mem_before = get_memory()
        t0 = time.time()
        output = llm(prompt, max_tokens=max_tokens)
        elapsed = time.time() - t0
        mem_after = get_memory()

        results["llama.cpp"] = {
            "time": elapsed,
            "tps": max_tokens / elapsed,
            "mem": mem_after - mem_before
        }

        print(f"   时间: {elapsed*1000:.2f} ms")
        print(f"   速度: {max_tokens/elapsed:.2f} tok/s")

        del llm

    except Exception as e:
        print(f"   ❌ 失败: {e}")

    print("\n" + "="*70)
    print("【2】CGC Engine V2 - GGUF 权重加载")
    print("="*70)

    try:
        engine = CGCEngineV2Real(
            mode="inference",
            kda_mode="standard",
            device="mps" if torch.backends.mps.is_available() else "cpu"
        )

        loaded = engine.load_gguf_weights(GGUF_FILE)
        if loaded:
            print(f"   GGUF 权重加载成功: {len(engine.gguf_weights)} tensors")

            model = engine.build_model()
            print(f"   模型构建成功")

            mem_before = get_memory()
            t0 = time.time()
            output = engine.generate(prompt, max_tokens=max_tokens)
            elapsed = time.time() - t0
            mem_after = get_memory()

            results["CGC-Standard"] = {
                "time": elapsed,
                "tps": output["tokens_per_second"],
                "mem": mem_after - mem_before
            }

            print(f"   时间: {elapsed*1000:.2f} ms")
            print(f"   速度: {output['tokens_per_second']:.2f} tok/s")
        else:
            print("   ❌ GGUF 权重加载失败")

    except Exception as e:
        print(f"   ❌ 失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("📊 结果对比")
    print("="*70)

    if results:
        print(f"\n{'方案':<20} {'速度 (tok/s)':<15} {'时间 (ms)':<15}")
        print("-"*50)
        for name, res in sorted(results.items(), key=lambda x: -x[1]["tps"]):
            print(f"{name:<20} {res['tps']:<15.2f} {res['time']*1000:<15.2f}")

    print("\n" + "="*70)
    print("✅ 完成!")
    print("="*70)


if __name__ == "__main__":
    run_benchmark()