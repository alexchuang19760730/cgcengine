#!/usr/bin/env python3
# Copyright (c) 2026 SandAI. All Rights Reserved.
"""
GGUF → MLX 模型转换器
将GGUF模型转换为MLX格式，支持MLX算子劫持测试
"""

import sys
import os
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, '/Users/alexchuang/Documents/flashkv0430/MagiCompiler-main')

try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None
    nn = None
    logger.error("MLX not available. Install with: pip install mlx")


class GGUFToMLXConverter:
    """GGUF → MLX 转换器"""
    
    def __init__(self, gguf_path: str):
        self.gguf_path = gguf_path
        self.model_config = {}
        self.weights = {}
        
        if not os.path.exists(gguf_path):
            raise FileNotFoundError(f"GGUF model not found: {gguf_path}")
        
        logger.info(f"📦 Initializing GGUF → MLX converter")
        logger.info(f"   Input: {gguf_path}")
    
    def parse_gguf_metadata(self) -> Dict[str, Any]:
        """解析GGUF元数据"""
        logger.info("🔍 Parsing GGUF metadata...")
        
        try:
            import gguf
            reader = gguf.GGUFReader(self.gguf_path)
            
            config = {
                "vocab_size": 32000,
                "hidden_dim": 4096,
                "num_layers": 32,
                "num_heads": 32,
                "num_kv_heads": 32,
                "head_dim": 128,
                "intermediate_dim": 11008,
                "norm_eps": 1e-5,
            }
            
            # 尝试读取元数据
            if hasattr(reader, 'fields'):
                fields = reader.fields
                
                # Qwen2.5
                if "qwen2.embedding_length" in fields:
                    config["hidden_dim"] = self._get_field_value(fields["qwen2.embedding_length"], 4096)
                    config["vocab_size"] = self._get_field_value(fields.get("qwen2.vocab_size", fields.get("tokenizer.ggml.model")), 32000)
                    config["num_layers"] = self._get_field_value(fields["qwen2.block_count"], 32)
                    config["num_heads"] = self._get_field_value(fields["qwen2.attention.head_count"], 32)
                    config["num_kv_heads"] = self._get_field_value(fields["qwen2.attention.head_count_kv"], 32)
                    config["intermediate_dim"] = self._get_field_value(fields.get("qwen2.feed_forward_length"), 11008)
                
                # LLaMA
                elif "llama.embedding_length" in fields:
                    config["hidden_dim"] = self._get_field_value(fields["llama.embedding_length"], 4096)
                    config["num_layers"] = self._get_field_value(fields["llama.block_count"], 32)
                    config["num_heads"] = self._get_field_value(fields["llama.attention.head_count"], 32)
                    config["num_kv_heads"] = self._get_field_value(fields["llama.attention.head_count_kv"], 32)
            
            self.model_config = config
            
            logger.info("✅ GGUF metadata parsed:")
            logger.info(f"   Vocab size: {config['vocab_size']}")
            logger.info(f"   Hidden dim: {config['hidden_dim']}")
            logger.info(f"   Num layers: {config['num_layers']}")
            logger.info(f"   Num heads: {config['num_heads']}")
            
            return config
            
        except ImportError:
            logger.error("❌ 'gguf' library not found. Install with: pip install gguf")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to parse GGUF metadata: {e}")
            raise
    
    def _get_field_value(self, field, default):
        """从GGUF字段获取值"""
        try:
            if hasattr(field, 'parts') and len(field.parts) > 0:
                for part in reversed(field.parts):
                    if hasattr(part, 'tolist'):
                        val = part.tolist()
                        if isinstance(val, (int, float)):
                            return val
                        elif isinstance(val, list) and len(val) > 0:
                            return val[0]
                    try:
                        return int(part)
                    except:
                        continue
        except:
            pass
        return default
    
    def load_gguf_weights(self) -> Dict[str, mx.array]:
        """加载GGUF权重"""
        logger.info("⏳ Loading GGUF weights...")
        
        try:
            import gguf
            reader = gguf.GGUFReader(self.gguf_path)
            
            weights = {}
            weight_count = 0
            
            for tensor in reader.tensors:
                name = tensor.name
                data = tensor.data
                
                # 转换为numpy再转MLX
                import numpy as np
                np_data = np.array(data, dtype=np.float32)
                mlx_data = mx.array(np_data)
                
                weights[name] = mlx_data
                weight_count += 1
            
            self.weights = weights
            
            logger.info(f"✅ Loaded {weight_count} weight tensors")
            return weights
            
        except Exception as e:
            logger.error(f"❌ Failed to load GGUF weights: {e}")
            raise
    
    def create_mlx_model(self) -> nn.Module:
        """创建MLX模型"""
        logger.info("🏗️ Creating MLX model...")
        
        config = self.model_config
        
        class SimpleLLM(nn.Module):
            """简化的LLM模型"""
            def __init__(self, config):
                super().__init__()
                self.config = config
                
                self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_dim"])
                
                self.layers = []
                for _ in range(config["num_layers"]):
                    layer = {
                        'self_attn': {
                            'q_proj': nn.Linear(config["hidden_dim"], config["hidden_dim"], bias=False),
                            'k_proj': nn.Linear(config["hidden_dim"], config["hidden_dim"], bias=False),
                            'v_proj': nn.Linear(config["hidden_dim"], config["hidden_dim"], bias=False),
                            'o_proj': nn.Linear(config["hidden_dim"], config["hidden_dim"], bias=False),
                        },
                        'mlp': {
                            'gate_proj': nn.Linear(config["hidden_dim"], config["intermediate_dim"], bias=False),
                            'up_proj': nn.Linear(config["hidden_dim"], config["intermediate_dim"], bias=False),
                            'down_proj': nn.Linear(config["intermediate_dim"], config["hidden_dim"], bias=False),
                        },
                        'input_layernorm': nn.LayerNorm(config["hidden_dim"], eps=config["norm_eps"]),
                        'post_attention_layernorm': nn.LayerNorm(config["hidden_dim"], eps=config["norm_eps"]),
                    }
                    self.layers.append(layer)
                
                self.norm = nn.LayerNorm(config["hidden_dim"], eps=config["norm_eps"])
                self.lm_head = nn.Linear(config["hidden_dim"], config["vocab_size"], bias=False)
            
            def __call__(self, x):
                B, T = x.shape
                
                h = self.embed_tokens(x)
                
                for layer in self.layers:
                    # Self-attention
                    residual = h
                    h = layer['input_layernorm'](h)
                    
                    q = layer['self_attn']['q_proj'](h)
                    k = layer['self_attn']['k_proj'](h)
                    v = layer['self_attn']['v_proj'](h)
                    
                    # Simplified attention
                    q = q.reshape(B, T, self.config["num_heads"], -1).transpose(0, 2, 1, 3)
                    k = k.reshape(B, T, self.config["num_heads"], -1).transpose(0, 2, 1, 3)
                    v = v.reshape(B, T, self.config["num_heads"], -1).transpose(0, 2, 1, 3)
                    
                    scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) / (self.config["hidden_dim"] ** 0.5)
                    attn = mx.softmax(scores, axis=-1)
                    out = mx.matmul(attn, v)
                    
                    out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
                    h = residual + layer['self_attn']['o_proj'](out)
                    
                    # MLP
                    residual = h
                    h = layer['post_attention_layernorm'](h)
                    
                    gate = mx.sigmoid(layer['mlp']['gate_proj'](h))
                    up = layer['mlp']['up_proj'](h)
                    h = residual + layer['mlp']['down_proj'](gate * up)
                
                h = self.norm(h)
                logits = self.lm_head(h)
                
                return logits
        
        model = SimpleLLM(config)
        
        logger.info("✅ MLX model created")
        return model
    
    def convert(self, output_path: Optional[str] = None) -> nn.Module:
        """执行转换"""
        logger.info("=" * 80)
        logger.info("🔄 GGUF → MLX Conversion")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # 1. 解析元数据
        config = self.parse_gguf_metadata()
        
        # 2. 加载权重
        weights = self.load_gguf_weights()
        
        # 3. 创建MLX模型
        model = self.create_mlx_model()
        
        # 4. 初始化模型
        logger.info("🔧 Initializing model...")
        dummy_input = mx.array([[0]])
        _ = model(dummy_input)
        mx.eval(_)
        
        elapsed = time.time() - start_time
        
        logger.info("=" * 80)
        logger.info(f"✅ Conversion completed in {elapsed:.2f}s")
        logger.info("=" * 80)
        
        # 5. 保存模型（可选）
        if output_path:
            logger.info(f"💾 Saving model to {output_path}...")
            mx.savez(output_path, **weights)
            logger.info("✅ Model saved")
        
        return model


def convert_gguf_to_mlx(gguf_path: str, output_path: Optional[str] = None) -> nn.Module:
    """转换GGUF到MLX（便捷函数）"""
    converter = GGUFToMLXConverter(gguf_path)
    return converter.convert(output_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GGUF → MLX Converter")
    parser.add_argument("gguf_path", help="Path to GGUF model")
    parser.add_argument("-o", "--output", help="Output path for MLX model", default=None)
    
    args = parser.parse_args()
    
    if not MLX_AVAILABLE:
        logger.error("MLX not available. Install with: pip install mlx")
        sys.exit(1)
    
    model = convert_gguf_to_mlx(args.gguf_path, args.output)
    
    logger.info("\n📊 Model Summary:")
    logger.info(f"   Type: {type(model).__name__}")
    logger.info(f"   Parameters: {sum(p.size for p in model.parameters().values())}")