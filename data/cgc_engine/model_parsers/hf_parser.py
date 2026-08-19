# Copyright (c) 2025 SandAI. All Rights Reserved.
"""
HF Parser - HuggingFace 标准格式解析器
"""

import os
import torch
from typing import Dict, List, Optional
from .base_parser import BaseModelParser, ParsedModel, ParsedWeight


class HuggingFaceParser(BaseModelParser):
    """HuggingFace 标准格式解析器"""

    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, **kwargs)
        self._hf_model = None
        self._use_safetensors = False

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")

        self._try_load_model()

    def _try_load_model(self):
        """尝试加载模型"""
        try:
            import safetensors
            self._use_safetensors = True
        except ImportError:
            pass

        try:
            from transformers import AutoModelForCausalLM
            self._hf_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="cpu",
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
        except Exception:
            pass

    def parse_model(self) -> ParsedModel:
        """解析模型结构"""

        vocab_size = 32000
        hidden_dim = 4096
        num_layers = 32
        num_heads = 32
        head_dim = 128
        model_type = "llama"

        if self._hf_model:
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
            metadata={"format": "huggingface", "path": self.model_path},
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
                    layer_id=layer_id
                )
                weights.append(weight)

        return weights

    def get_metadata(self) -> Dict:
        """获取元数据"""
        if self._hf_model:
            return dict(self._hf_model.config.to_dict())
        return {"format": "huggingface", "path": self.model_path}

    def close(self):
        """关闭解析器"""
        if self._hf_model:
            del self._hf_model
            self._hf_model = None
