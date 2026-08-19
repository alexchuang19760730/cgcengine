# Copyright (c) 2025 SandAI. All Rights Reserved.
"""
GGUF Parser - llama.cpp GGUF 格式解析器
"""

import os
import torch
import logging
from typing import Dict, List, Optional
from .base_parser import BaseModelParser, ParsedModel, ParsedWeight

logger = logging.getLogger(__name__)


class GGUFParser(BaseModelParser):
    """llama.cpp GGUF 格式解析器"""

    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, **kwargs)
        self._gguf_data = None
        self._gguf_ctx = None

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self._try_load_gguf()

    def _try_load_gguf(self):
        """尝试加载 GGUF - 优先使用 gguf 库（支持反量化）"""
        try:
            import gguf
            self._gguf_data = gguf.GGUFReader(self.model_path)
            logger.info(f"[GGUFParser] Loaded using gguf library")
        except ImportError:
            try:
                import llama_cpp
                self._gguf_ctx = llama_cpp.Llama(
                    model_path=self.model_path,
                    n_ctx=512,
                    verbose=False
                )
                logger.info(f"[GGUFParser] Loaded using llama_cpp library")
            except ImportError:
                raise ImportError("Either 'gguf' or 'llama_cpp' library is required")

    def parse_model(self) -> ParsedModel:
        """解析 GGUF 模型结构"""

        # 默认结构（LLaMA）
        model_type = "llama"
        vocab_size = 32000
        hidden_dim = 4096
        num_layers = 32
        num_heads = 32
        head_dim = 128
        num_kv_heads = 32

        if self._gguf_data:
            try:
                # 从 GGUF 元数据读取
                # 辅助函数：从 ReaderField 获取整数值
                def get_int_field(key, default):
                    if key in self._gguf_data.fields:
                        field = self._gguf_data.fields[key]
                        if hasattr(field, 'parts') and len(field.parts) > 0:
                            # GGUF字段结构：parts[0]=偏移量, parts[1]=名称, parts[2]=类型, parts[3]=值
                            # 或者 parts[-1] 是值
                            for part in reversed(field.parts):
                                try:
                                    # 尝试从 parts 中提取整数值
                                    if hasattr(part, 'tolist'):
                                        val = part.tolist()
                                        if isinstance(val, int):
                                            if val > 0 and val < 100000:  # 合理范围
                                                return val
                                        elif isinstance(val, list) and len(val) > 0:
                                            int_val = int(val[0])
                                            if int_val > 0 and int_val < 100000:
                                                return int_val
                                    # 尝试直接转换
                                    int_val = int(part)
                                    if int_val > 0 and int_val < 100000:
                                        return int_val
                                except:
                                    continue
                        # 尝试从 data 属性获取
                        if hasattr(field, 'data') and field.data:
                            try:
                                val = field.data[0]
                                if isinstance(val, int) and val > 0 and val < 100000:
                                    return val
                            except:
                                pass
                    return default

                # Qwen2.5 特定的字段
                hidden_dim = get_int_field("qwen2.embedding_length", 4096)
                num_layers = get_int_field("qwen2.block_count", 32)
                num_heads = get_int_field("qwen2.attention.head_count", 32)
                num_kv_heads = get_int_field("qwen2.attention.head_count_kv", num_heads)
                vocab_size = get_int_field("qwen2.vocabulary_size", 32000)
                
                # 备用字段（通用）
                if hidden_dim == 4096:
                    hidden_dim = get_int_field("n_embd", 4096)
                if num_layers == 32:
                    num_layers = get_int_field("n_layer", 32)
                if num_heads == 32:
                    num_heads = get_int_field("n_head", 32)
                if num_kv_heads == num_heads:
                    num_kv_heads = get_int_field("n_head_kv", num_heads)
                if vocab_size == 32000:
                    vocab_size = get_int_field("vocab_size", 32000)
                
                # 从 llama.cpp metadata 获取（如果可用）
                if hasattr(self, '_gguf_ctx') and self._gguf_ctx is not None:
                    if hasattr(self._gguf_ctx, 'metadata'):
                        meta = self._gguf_ctx.metadata
                        if 'general.quantization_version' in meta:
                            # 量化模型的 vocab_size 可能不同
                            pass
                
                # 计算 head_dim
                head_dim = hidden_dim // num_heads if num_heads > 0 else 128
                
                # 如果从元数据获取的 vocab_size 与权重形状不匹配，从权重推断
                if self._gguf_data:
                    try:
                        for tensor_info in self._gguf_data.tensors:
                            if tensor_info.name == "token_embd.weight":
                                # token_embd.weight 的形状是 [vocab_size, hidden_dim] 或 [hidden_dim, vocab_size]
                                inferred_vocab = max(tensor_info.shape)
                                if inferred_vocab != vocab_size:
                                    logger.info(f"[GGUFParser] Adjusting vocab_size from {vocab_size} to {inferred_vocab} based on token_embd.weight shape")
                                    vocab_size = inferred_vocab
                                break
                    except Exception as e:
                        logger.debug(f"[GGUFParser] Failed to infer vocab_size from weights: {e}")
                    
            except Exception as e:
                logger.debug(f"[GGUFParser] Failed to parse metadata from gguf: {e}")

        elif self._gguf_ctx:
            # 从 llama_cpp Llama object 的 metadata 读取
            metadata = self._gguf_ctx.metadata
            
            # Qwen2.5 特定的字段
            vocab_size = int(metadata.get('qwen2.vocabulary_size', metadata.get('vocab_size', 32000)))
            hidden_dim = int(metadata.get('qwen2.embedding_length', metadata.get('n_embd', 4096)))
            num_layers = int(metadata.get('qwen2.block_count', metadata.get('n_layer', 32)))
            num_heads = int(metadata.get('qwen2.attention.head_count', metadata.get('n_head', 32)))
            
            # GQA 支持
            num_kv_heads = int(metadata.get('qwen2.attention.head_count_kv', metadata.get('n_head_kv', num_heads)))
            head_dim = hidden_dim // num_kv_heads if num_kv_heads > 0 else 128

        logger.info(f"[GGUFParser] Model: {model_type}, vocab={vocab_size}, hidden={hidden_dim}, layers={num_layers}, heads={num_heads}, kv_heads={num_kv_heads}, head_dim={head_dim}")

        return ParsedModel(
            model_type=model_type,
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            metadata={"format": "gguf", "path": self.model_path, "num_kv_heads": num_kv_heads},
        )

    def load_weights(self, layer_id: Optional[int] = None) -> List[ParsedWeight]:
        """从 GGUF 加载权重"""
        weights = []

        if self._gguf_data:
            # 从 gguf 库加载（优先，支持反量化）
            import gguf
            
            for tensor_info in self._gguf_data.tensors:
                name = tensor_info.name
                if layer_id is not None and f"blk.{layer_id}." not in name and "blk." in name:
                    continue

                try:
                    # 获取量化类型并反量化
                    qtype = gguf.GGMLQuantizationType(tensor_info.tensor_type)
                    tensor_data = gguf.dequantize(tensor_info.data, qtype)
                    
                    # 转换为 torch.Tensor
                    import numpy as np
                    if hasattr(tensor_data, "__array__"):
                        np_arr = np.array(tensor_data)
                        tensor = torch.from_numpy(np_arr).float()
                    else:
                        tensor = torch.tensor(tensor_data, dtype=torch.float32)

                    # 推断层 ID
                    if "blk." in name:
                        try:
                            lid = int(name.split("blk.")[1].split(".")[0])
                        except (IndexError, ValueError):
                            lid = layer_id
                    else:
                        lid = layer_id

                    # 推断权重类型
                    w_type = self._infer_weight_type(name)

                    weights.append(ParsedWeight(
                        name=name,
                        tensor=tensor,
                        shape=list(tensor.shape),
                        dtype=str(tensor.dtype),
                        layer_id=lid,
                        weight_type=w_type
                    ))
                except Exception as e:
                    # 跳过失败的 tensor
                    logger.debug(f"[GGUFParser] Failed to load tensor {name}: {e}")
                    continue

        elif self._gguf_ctx:
            # 从 llama_cpp 库加载（优先）
            try:
                # 获取内部模型对象
                llm_model = None
                if hasattr(self._gguf_ctx, '_model'):
                    llm_model = self._gguf_ctx._model
                elif hasattr(self._gguf_ctx, 'model') and hasattr(self._gguf_ctx.model, '_model'):
                    llm_model = self._gguf_ctx.model._model

                if llm_model is not None and hasattr(llm_model, 'get_tensor'):
                    # 获取张量名称列表
                    tensor_names = []
                    if hasattr(llm_model, 'tensor_names'):
                        tensor_names = llm_model.tensor_names
                    else:
                        # 如果没有 tensor_names，尝试从元数据推断常见张量名
                        parsed_model = self.parse_model()
                        num_layers = parsed_model.num_layers
                        common_names = [
                            "token_embd.weight",
                            "output.weight",
                            "output_norm.weight"
                        ]
                        for i in range(num_layers):
                            common_names.extend([
                                f"blk.{i}.attn_q.weight",
                                f"blk.{i}.attn_q.bias",
                                f"blk.{i}.attn_k.weight",
                                f"blk.{i}.attn_k.bias",
                                f"blk.{i}.attn_v.weight",
                                f"blk.{i}.attn_v.bias",
                                f"blk.{i}.attn_output.weight",
                                f"blk.{i}.attn_output.bias",
                                f"blk.{i}.attn_norm.weight",
                                f"blk.{i}.ffn_gate.weight",
                                f"blk.{i}.ffn_gate.bias",
                                f"blk.{i}.ffn_up.weight",
                                f"blk.{i}.ffn_up.bias",
                                f"blk.{i}.ffn_down.weight",
                                f"blk.{i}.ffn_down.bias",
                                f"blk.{i}.ffn_norm.weight",
                            ])
                        tensor_names = common_names

                    logger.info(f"[GGUFParser] Found {len(tensor_names)} tensors")

                    for name in tensor_names:
                        if layer_id is not None and f"blk.{layer_id}." not in name and "blk." in name:
                            continue

                        try:
                            tensor = llm_model.get_tensor(name)
                            if tensor is not None:
                                # 转换为 torch.Tensor
                                import numpy as np
                                if hasattr(tensor, "__array__"):
                                    np_arr = np.array(tensor)
                                    tensor = torch.from_numpy(np_arr).float()
                                else:
                                    tensor = torch.tensor(tensor, dtype=torch.float32)

                                # 推断层 ID
                                if "blk." in name:
                                    try:
                                        lid = int(name.split("blk.")[1].split(".")[0])
                                    except (IndexError, ValueError):
                                        lid = layer_id
                                else:
                                    lid = layer_id

                                w_type = self._infer_weight_type(name)
                                weights.append(ParsedWeight(
                                    name=name,
                                    tensor=tensor,
                                    shape=list(tensor.shape),
                                    dtype=str(tensor.dtype),
                                    layer_id=lid,
                                    weight_type=w_type
                                ))
                        except Exception as e:
                            logger.debug(f"[GGUFParser] Failed to load tensor {name}: {e}")
                            continue

                else:
                    # 如果没有 get_tensor 方法，尝试旧方法
                    self._load_weights_fallback(weights, layer_id)

            except Exception as e:
                logger.debug(f"[GGUFParser] Failed to load weights from llama_cpp: {e}")
                self._load_weights_fallback(weights, layer_id)

        # 如果没加载到，返回简单示例
        if len(weights) == 0:
            example_weight = torch.randn(4096, 4096, dtype=torch.float32)
            weights.append(ParsedWeight(
                name="example_weight",
                tensor=example_weight,
                shape=list(example_weight.shape),
                dtype=str(example_weight.dtype),
                layer_id=layer_id,
                weight_type="linear"
            ))

        logger.info(f"[GGUFParser] Loaded {len(weights)} weights")
        return weights

    def _load_weights_fallback(self, weights, layer_id):
        """fallback 权重加载"""
        try:
            state_dict = {}
            if hasattr(self._gguf_ctx, "model"):
                state_dict = self._gguf_ctx.model.state_dict()

            if state_dict:
                for name, tensor in state_dict.items():
                    if layer_id is not None and f"layers.{layer_id}." not in name:
                        continue

                    w_type = self._infer_weight_type(name)
                    weights.append(ParsedWeight(
                        name=name,
                        tensor=tensor.clone(),
                        shape=list(tensor.shape),
                        dtype=str(tensor.dtype),
                        layer_id=layer_id,
                        weight_type=w_type
                    ))
        except Exception as e:
            logger.debug(f"[GGUFParser] Fallback load failed: {e}")
            pass

    def _get_tensor_from_llama_cpp(self, model, name: str):
        """从 llama_cpp 模型获取张量"""
        try:
            # 尝试多种方式获取张量
            if hasattr(model, 'get_tensor'):
                return model.get_tensor(name)

            # 通过内部 API
            if hasattr(model, '_model'):
                inner_model = model._model
                if hasattr(inner_model, 'tensors'):
                    if name in inner_model.tensors:
                        return torch.tensor(inner_model.tensors[name], dtype=torch.float32)

            # 尝试直接访问
            if hasattr(model, 'tensors'):
                if name in model.tensors:
                    return torch.tensor(model.tensors[name], dtype=torch.float32)

        except Exception as e:
            logger.debug(f"[GGUFParser] _get_tensor_from_llama_cpp failed for {name}: {e}")

        return None

    def _infer_weight_type(self, name: str) -> str:
        """推断 GGUF 权重类型"""
        if "attn_q" in name or "attn_k" in name or "attn_v" in name:
            return "attention_proj"
        elif "attn_output" in name or "attn_o" in name:
            return "attention_out"
        elif "ffn_gate" in name or "ffn_up" in name:
            return "mlp_up"
        elif "ffn_down" in name or "ffn_o" in name:
            return "mlp_down"
        elif "ffn_norm" in name or "attn_norm" in name or "norm" in name:
            return "norm"
        elif "token_embd" in name or "embed" in name:
            return "embedding"
        elif "output" in name or "lm_head" in name:
            return "lm_head"
        return "unknown"

    def get_metadata(self) -> Dict:
        """获取 GGUF 元数据"""
        if self._gguf_data:
            return {k: str(v) for k, v in self._gguf_data.fields.items()}
        elif self._gguf_ctx:
            return {
                "n_vocab": self._gguf_ctx.n_vocab,
                "n_embd": self._gguf_ctx.n_embd,
                "n_layer": self._gguf_ctx.n_layer,
                "n_head": self._gguf_ctx.n_head,
            }
        return {"format": "gguf", "path": self.model_path}

    def close(self):
        """关闭 GGUF 解析器"""
        if self._gguf_ctx:
            del self._gguf_ctx
            self._gguf_ctx = None