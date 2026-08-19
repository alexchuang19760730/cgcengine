# Copyright (c) 2025 SandAI. All Rights Reserved.
"""
vLLM Parser - vLLM 格式解析器
"""

import os
import torch
from typing import Dict, List, Optional
from .base_parser import BaseModelParser, ParsedModel, ParsedWeight


class VLLMParser(BaseModelParser):
    """vLLM 格式解析器（支持 HuggingFace/vLLM checkpoint）"""

    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, **kwargs)
        self._hf_model = None
        self._vllm_model = None

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")

        self._try_load_model()

    def _try_load_model(self):
        """尝试加载模型"""
        try:
            from transformers import AutoModelForCausalLM
            self._hf_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="cpu",
                trust_remote_code=True
            )
        except Exception:
            try:
                from vllm import LLM
                self._vllm_model = LLM(
                    model=self.model_path,
                    quantization=None
                )
            except Exception:
                pass

    def parse_model(self) -> ParsedModel:
        """解析模型结构"""

        # 默认结构
        model_type = "llama"
        vocab_size = 32000
        hidden_dim = 4096
        num_layers = 32
        num_heads = 32
        head_dim = 128

        if self._hf_model:
            # 从 HuggingFace 读取
            config = self._hf_model.config
            vocab_size = getattr(config, "vocab_size", vocab_size)
            hidden_dim = getattr(config, "hidden_size", hidden_dim)
            num_layers = getattr(config, "num_hidden_layers", num_layers)
            num_heads = getattr(config, "num_attention_heads", num_heads)
            if hasattr(config, "model_type"):
                model_type = config.model_type
            head_dim = hidden_dim // num_heads

        return ParsedModel(
            model_type=model_type,
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            metadata={"format": "vllm/hf", "path": self.model_path},
        )

    def load_weights(self, layer_id: Optional[int] = None) -> List[ParsedWeight]:
        """加载权重"""
        weights = []

        if self._hf_model:
            state_dict = self._hf_model.state_dict()
            for name, tensor in state_dict.items():
                if layer_id is not None and f"layers.{layer_id}." not in name:
                    continue

                weight = ParsedWeight(
                    name=name,
                    tensor=tensor.clone(),
                    shape=list(tensor.shape),
                    dtype=str(tensor.dtype),
                    layer_id=layer_id,
                    weight_type=self._infer_weight_type(name)
                )
                weights.append(weight)
        else:
            # 尝试直接读取 safetensors
            weights = self._load_safetensors_direct(layer_id)

        return weights

    def _load_safetensors_direct(self, layer_id: Optional[int] = None) -> List[ParsedWeight]:
        """直接从 safetensors 加载（不加载完整模型）"""
        weights = []
        try:
            import safetensors
            import safetensors.torch

            # 查找 safetensors 文件
            safetensors_files = []
            if os.path.isdir(self.model_path):
                safetensors_files = [
                    os.path.join(self.model_path, f)
                    for f in os.listdir(self.model_path)
                    if f.endswith(".safetensors")
                ]
            elif self.model_path.endswith(".safetensors"):
                safetensors_files = [self.model_path]

            for sf_file in safetensors_files:
                tensors = safetensors.torch.load_file(sf_file, device="cpu")
                for name, tensor in tensors.items():
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
        except Exception:
            # 失败返回空
            pass

        return weights

    def _infer_weight_type(self, name: str) -> str:
        """推断权重类型"""
        if "q_proj" in name or "k_proj" in name or "v_proj" in name:
            return "attention_proj"
        elif "o_proj" in name:
            return "attention_out"
        elif "gate_proj" in name or "up_proj" in name:
            return "mlp_up"
        elif "down_proj" in name:
            return "mlp_down"
        elif "norm" in name or "ln" in name:
            return "norm"
        elif "embed" in name or "wte" in name:
            return "embedding"
        elif "lm_head" in name:
            return "lm_head"
        return "unknown"

    def get_metadata(self) -> Dict:
        """获取元数据"""
        if self._hf_model:
            config = self._hf_model.config
            return {
                "model_type": getattr(config, "model_type", "unknown"),
                "vocab_size": getattr(config, "vocab_size", -1),
                "hidden_size": getattr(config, "hidden_size", -1),
                "num_hidden_layers": getattr(config, "num_hidden_layers", -1),
                "num_attention_heads": getattr(config, "num_attention_heads", -1),
            }
        return {"format": "vllm/hf", "path": self.model_path}

    def close(self):
        """关闭解析器"""
        if self._hf_model:
            del self._hf_model
            self._hf_model = None
        if self._vllm_model:
            del self._vllm_model
            self._vllm_model = None
