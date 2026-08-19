# Copyright (c) 2025 SandAI. All Rights Reserved.
"""
ParsedModel → PyTorch 模块适配器

功能：
- 将 ParsedModel 转换为标准 PyTorch nn.Module
- 支持 MagiCompiler / Harness Agent 使用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from dataclasses import dataclass
from .base_parser import ParsedModel, ParsedWeight
import logging

logger = logging.getLogger(__name__)


class AdapterTransformerBlock(nn.Module):
    """适配器 Transformer Block"""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int = 128,
        num_kv_heads: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads

        # Attention
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim)
        self.k_proj = nn.Linear(hidden_dim, self.num_kv_heads * head_dim)
        self.v_proj = nn.Linear(hidden_dim, self.num_kv_heads * head_dim)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_dim)

        # FFN (Llama style)
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim * 4)
        self.up_proj = nn.Linear(hidden_dim, hidden_dim * 4)
        self.down_proj = nn.Linear(hidden_dim * 4, hidden_dim)

        # Norms
        self.input_layernorm = nn.LayerNorm(hidden_dim)
        self.post_attention_layernorm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # Attention
        residual = x
        x = self.input_layernorm(x)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        batch_size, seq_len, _ = x.shape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # SDPA with GQA support
        if self.num_kv_heads != self.num_heads:
            # GQA: repeat K/V heads
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        attn_output = F.scaled_dot_product_attention(q, k, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        x = residual + attn_output

        # FFN
        residual = x
        x = self.post_attention_layernorm(x)

        ffn_output = self.gate_proj(x)
        ffn_output = F.silu(ffn_output)
        ffn_output = ffn_output * self.up_proj(x)
        ffn_output = self.down_proj(ffn_output)

        x = residual + ffn_output
        return x


class AdapterLLM(nn.Module):
    """适配器：ParsedModel → PyTorch Module"""

    def __init__(self, parsed_model: ParsedModel):
        super().__init__()
        self.parsed_model = parsed_model
        self.vocab_size = parsed_model.vocab_size
        self.hidden_dim = parsed_model.hidden_dim
        self.num_layers = parsed_model.num_layers
        self.num_heads = parsed_model.num_heads
        self.head_dim = parsed_model.head_dim
        
        # 从元数据获取 KV heads（GQA 支持）
        self.num_kv_heads = parsed_model.metadata.get('num_kv_heads', self.num_heads)

        # Embedding - 使用解析的 hidden_dim
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_dim)

        # Layers
        self.layers = nn.ModuleList([
            AdapterTransformerBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                num_kv_heads=self.num_kv_heads,
            )
            for _ in range(self.num_layers)
        ])

        # Final norm and LM head
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.lm_head = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)

        # 添加 config 属性用于 benchmark
        self.config = type('Config', (), {'vocab_size': self.vocab_size})

        logger.info(
            f"[AdapterLLM] Created from ParsedModel: "
            f"vocab={self.vocab_size}, hidden={self.hidden_dim}, "
            f"layers={self.num_layers}, heads={self.num_heads}, kv_heads={self.num_kv_heads}"
        )

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


def parsed_model_to_pytorch(
    parsed_model: ParsedModel,
    weights: Optional[List[ParsedWeight]] = None,
) -> nn.Module:
    """
    工厂函数：将 ParsedModel 转换为 PyTorch Module

    Args:
        parsed_model: 解析后的模型
        weights: 可选，要加载的权重列表

    Returns:
        PyTorch nn.Module
    """
    model = AdapterLLM(parsed_model)

    if weights:
        logger.info(f"[Adapter] Loading {len(weights)} weights...")
        state_dict = {}
        
        for w in weights:
            # 转换 GGUF 权重名称到 PyTorch 格式
            pt_name = _gguf_to_pytorch_name(w.name)
            if pt_name:
                # 确保形状匹配
                pt_tensor = _adjust_tensor_shape(w.tensor, pt_name, model)
                state_dict[pt_name] = pt_tensor
        
        try:
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"[Adapter] Successfully loaded {len(state_dict)} weights")
        except Exception as e:
            logger.warning(f"[Adapter] Failed to load weights: {e}")

    return model


def _gguf_to_pytorch_name(gguf_name: str) -> str:
    """将 GGUF 权重名称转换为 PyTorch 模型名称"""
    # 处理 embedding
    if gguf_name.startswith("token_embd.weight"):
        return "embed_tokens.weight"
    
    # 处理 final norm
    if gguf_name.startswith("output_norm.weight"):
        return "norm.weight"
    
    # 处理 lm_head
    if gguf_name.startswith("output.weight"):
        return "lm_head.weight"
    
    # 处理 transformer blocks
    if gguf_name.startswith("blk."):
        # blk.0.attn_q.weight -> layers.0.q_proj.weight
        parts = gguf_name.split(".")
        if len(parts) >= 4:
            layer_idx = parts[1]
            attn_type = parts[2]
            param_type = parts[3]
            
            # 映射 attention 类型
            attn_map = {
                "attn_q": "q_proj",
                "attn_k": "k_proj",
                "attn_v": "v_proj",
                "attn_output": "o_proj",
                "attn_norm": "input_layernorm",
            }
            
            # 映射 FFN 类型
            ffn_map = {
                "ffn_gate": "gate_proj",
                "ffn_up": "up_proj",
                "ffn_down": "down_proj",
                "ffn_norm": "post_attention_layernorm",
            }
            
            if attn_type in attn_map:
                return f"layers.{layer_idx}.{attn_map[attn_type]}.{param_type}"
            elif attn_type in ffn_map:
                return f"layers.{layer_idx}.{ffn_map[attn_type]}.{param_type}"
    
    # 保留原始名称（可能不匹配）
    return None


def _adjust_tensor_shape(tensor: torch.Tensor, pt_name: str, model: nn.Module) -> torch.Tensor:
    """调整张量形状以匹配 PyTorch 模型"""
    if pt_name in model.state_dict():
        target_shape = model.state_dict()[pt_name].shape
        
        # 如果形状不匹配，尝试转置（常见于 GGUF 权重）
        if tensor.shape != target_shape:
            if tensor.T.shape == target_shape:
                tensor = tensor.T
                logger.debug(f"[Adapter] Transposed {pt_name}: {tensor.shape}")
            else:
                logger.debug(f"[Adapter] Shape mismatch for {pt_name}: {tensor.shape} vs {target_shape}")
    
    return tensor
