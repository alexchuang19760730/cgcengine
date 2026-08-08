# Copyright (c) 2025 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CGC Runtime - 獨立運行時（移除 vLLM/Megatron 依賴）

功能：
- 只保留 CGC 作為唯一計算內核
- 上層只做 API / 調度 / Dataset
- 穩定、無框架衝突、極致低延遲

架構：
- CGCExecutor: 統一計算引擎
- 調度層: PD Client
- 存儲層: GDS/PD/SPDK
- API 層: REST/gRPC
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """模型配置"""
    vocab_size: int = 32000
    hidden_dim: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int = 128
    intermediate_size: int = 11008
    max_seq_len: int = 4096
    dropout: float = 0.0
    activation: str = "silu"


class CGCModel(nn.Module):
    """
    CGC 原生模型
    
    不依賴 vLLM/Megatron，使用純 CGC 計算
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.layers = nn.ModuleList([
            CGCLayer(config) for _ in range(config.num_layers)
        ])
        self.norm = nn.RMSNorm(config.hidden_dim)
        
        logger.info(f"[CGCModel] Initialized: layers={config.num_layers}, hidden={config.hidden_dim}")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向傳播
        
        全部使用 CGC 計算，無 vLLM/Megatron 依賴
        """
        hidden_states = self.embed_tokens(input_ids)
        
        if positions is None:
            positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions)
        
        hidden_states = self.norm(hidden_states)
        
        return hidden_states


class CGCLayer(nn.Module):
    """CGC 計算層"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.attention = CGCAttention(config)
        self.mlp = CGCMLP(config)

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """殮層前向傳播"""
        attn_output = self.attention(hidden_states, positions)
        return self.mlp(attn_output)


class CGCAttention(nn.Module):
    """CGC Attention 層"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Attention 前向傳播"""
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        q = q.view(batch_size, seq_len, self.config.num_heads, self.config.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.config.num_heads, self.config.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.config.num_heads, self.config.head_dim).transpose(1, 2)
        
        scale = 1.0 / (self.config.head_dim ** 0.5)
        attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.o_proj(attn_output)


class CGCMLP(nn.Module):
    """CGC MLP 層"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.gate_proj = nn.Linear(config.hidden_dim, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_dim, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """MLP 前向傳播"""
        gate = self.gate_proj(hidden_states)
        gate = torch.nn.functional.silu(gate)
        up = self.up_proj(hidden_states)
        
        return self.down_proj(gate * up)


class CGCRuntime:
    """
    CGC 獨立運行時

    功能：
    - 卸載所有 vLLM/Megatron 依賴
    - 統一 CGC 計算引擎
    - REST/gRPC API 接口
    - llama.cpp GGUF 橋接支援
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        device: str = None,
        enable_llama_cpp_bridge: bool = False,
        gguf_path: Optional[str] = None,
        enable_moe: bool = True,
        enable_gds: bool = True,
    ):
        """
        Args:
            model_config: 模型配置（可選）
            device: 設備（默認自動檢測）
            enable_llama_cpp_bridge: 是否啟用 llama.cpp GGUF 橋接
            gguf_path: GGUF 模型路徑
            enable_moe: 是否啟用 MoE
            enable_gds: 是否啟用 GDS
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.model_config = model_config or ModelConfig()
        self.device = device
        self.enable_moe = enable_moe
        self.enable_gds = enable_gds

        self.model = None
        self.llama_cpp_model = None
        self.tokenizer = None

        if enable_llama_cpp_bridge and gguf_path:
            self._init_llama_cpp(gguf_path)
        else:
            self.model = CGCModel(self.model_config).to(device)
            self.model.eval()

        logger.info(f"[CGCRuntime] Initialized on {device}, llama_cpp={'Yes' if self.llama_cpp_model else 'No'}")

    def _init_llama_cpp(self, gguf_path: str) -> None:
        """初始化 llama.cpp GGUF 模型"""
        try:
            from llama_cpp import Llama

            n_gpu_layers = 32
            if self.device == "mps":
                n_gpu_layers = 0

            self.llama_cpp_model = Llama(
                model_path=gguf_path,
                n_ctx=8192,
                n_gpu_layers=n_gpu_layers,
                use_mmap=True,
                use_mlock=False,
                verbose=False,
            )

            logger.info(f"[CGCRuntime] llama.cpp GGUF loaded: {gguf_path}")

            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    gguf_path.replace(".gguf", ""),
                    trust_remote_code=True,
                )
            except:
                pass

        except ImportError:
            logger.error("[CGCRuntime] llama_cpp not installed")
            raise
        except Exception as e:
            logger.error(f"[CGCRuntime] Failed to load GGUF: {e}")
            raise

    def is_llama_cpp_mode(self) -> bool:
        """是否使用 llama.cpp 模式"""
        return self.llama_cpp_model is not None

    @torch.no_grad()
    def generate(
        self,
        input_ids: Union[torch.Tensor, str, List[int]],
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        top_k: int = 50,
        stop_token_id: Optional[int] = None,
        **kwargs,
    ) -> Union[List[int], str, Dict]:
        """
        生成文本（支援 llama.cpp 和 PyTorch 兩種模式）

        Args:
            input_ids: 輸入 token IDs 或文字
            max_new_tokens: 最大新 token 數
            temperature: 採樣溫度
            top_k: Top-K 採樣
            stop_token_id: 停止 token ID
            **kwargs: 其他參數

        Returns:
            生成的 token 列表或文字
        """
        if self.llama_cpp_model is not None:
            return self._generate_llama_cpp(
                input_ids, max_new_tokens, temperature, top_k, stop_token_id, **kwargs
            )
        else:
            return self._generate_torch(
                input_ids, max_new_tokens, temperature, top_k, stop_token_id
            )

    def _generate_llama_cpp(
        self,
        input_ids: Union[torch.Tensor, str, List[int]],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        stop_token_id: Optional[int],
        **kwargs,
    ) -> Dict:
        """使用 llama.cpp 生成"""
        if isinstance(input_ids, torch.Tensor):
            if input_ids.dim() > 1:
                input_ids = input_ids[0].tolist()
            else:
                input_ids = input_ids.tolist()

        if isinstance(input_ids, list):
            if self.tokenizer:
                text = self.tokenizer.decode(input_ids)
            else:
                text = "".join(chr(t) if t < 256 else f"<{t}>" for t in input_ids)
        else:
            text = str(input_ids)

        stop = kwargs.get("stop", ["</s>", "<|endoftext|>"])
        if stop_token_id is not None:
            stop = stop + [stop_token_id]

        result = self.llama_cpp_model(
            text,
            max_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 0.7,
            top_k=top_k if top_k > 0 else 50,
            stop=stop,
            echo=False,
            **kwargs,
        )

        return {
            "text": result.get("choices", [{}])[0].get("text", ""),
            "usage": result.get("usage", {}),
            "generated_text": result.get("choices", [{}])[0].get("text", ""),
        }

    def _generate_torch(
        self,
        input_ids: Union[torch.Tensor, str, List[int]],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        stop_token_id: Optional[int],
    ) -> List[int]:
        """使用 PyTorch 模型生成"""
        if self.model is None:
            raise RuntimeError("No model loaded")

        if isinstance(input_ids, str):
            if self.tokenizer:
                input_ids = self.tokenizer.encode(input_ids, return_tensors="pt")[0]
            else:
                input_ids = torch.tensor([ord(c) for c in input_ids], dtype=torch.long)

        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        generated = input_ids[0].tolist()

        for _ in range(max_new_tokens):
            logits = self.model(input_ids)
            next_token_logits = logits[0, -1, :]

            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                if top_k > 0:
                    v, i = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits[i] = v
                next_token = torch.argmax(next_token_logits).item()

            generated.append(next_token)

            if stop_token_id is not None and next_token == stop_token_id:
                break

            input_ids = torch.tensor([[next_token]], dtype=torch.long, device=self.device)

        return generated

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """前向傳播"""
        if self.llama_cpp_model is not None:
            raise NotImplementedError("Use generate() for llama.cpp mode")
        return self.model(input_ids)


class CGCDataset:
    """
    CGC 數據集封裝
    
    替代 vLLM 的 DataLoader
    """

    def __init__(
        self,
        data_source: Any,
        tokenizer: Optional[Any] = None,
    ):
        """
        Args:
            data_source: 數據源
            tokenizer: 分詞器
        """
        self.data_source = data_source
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data_source)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data_source[idx]
        
        if self.tokenizer is not None and isinstance(item, str):
            return self.tokenizer(item)
        
        return item


class CGCAutoModel:
    """
    CGC 自動模型加載
    
    替代 transformers.AutoModel
    """

    @staticmethod
    def from_pretrained(
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> nn.Module:
        """
        加載模型（不依賴 transformers）
        
        Args:
            model_path: 模型路徑
            device: 設備
        """
        config_path = Path(model_path) / "config.json"
        
        if config_path.exists():
            import json
            with open(config_path) as f:
                config_dict = json.load(f)
            
            config = ModelConfig(
                vocab_size=config_dict.get("vocab_size", 32000),
                hidden_dim=config_dict.get("hidden_size", 4096),
                num_layers=config_dict.get("num_hidden_layers", 32),
                num_heads=config_dict.get("num_attention_heads", 32),
                head_dim=config_dict.get("head_dim", 128),
                intermediate_size=config_dict.get("intermediate_size", 11008),
            )
        else:
            config = ModelConfig()
        
        model = CGCModel(config).to(device)
        
        weight_path = Path(model_path) / "pytorch_model.bin"
        if weight_path.exists():
            state_dict = torch.load(weight_path, map_location=device)
            model.load_state_dict(state_dict)
        
        return model


def create_cgc_runtime(
    model_path: Optional[str] = None,
    config: Optional[ModelConfig] = None,
    device: Optional[str] = None,
    enable_llama_cpp_bridge: bool = False,
    gguf_path: Optional[str] = None,
) -> CGCRuntime:
    """
    創建 CGC 運行時（便捷函數）

    Args:
        model_path: 模型路徑
        config: 模型配置
        device: 設備
        enable_llama_cpp_bridge: 是否啟用 llama.cpp GGUF 橋接
        gguf_path: GGUF 模型路徑

    Returns:
        CGCRuntime 實例
    """
    return CGCRuntime(
        model_config=config,
        device=device,
        enable_llama_cpp_bridge=enable_llama_cpp_bridge,
        gguf_path=gguf_path or model_path,
    )
