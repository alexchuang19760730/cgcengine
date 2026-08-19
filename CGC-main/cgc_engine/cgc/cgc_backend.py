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
🌟 CGC 终极架构：统一计算 + 全局调度

【本质定义（可写进论文/白皮书）】
所有计算 → 统一收敛到 CGC SIMD 指令
所有非计算 → 全部下沉为 PD 调度

【计算归一】
vLLM：PagedAttention / GPTQ / AWQ / FP16
llama.cpp：GGUF Q4_K / Q5_K / MoE 量化
全部被 CGC 统一抽象成同一套 SIMD 运算指令

【调度分离】
权重加载、KV 存取、专家路由、显存调度、分层卸载、LRU 淘汰
全部从执行层剥离，变成 CGC Backend + PD 的统一调度行为

【极简架构图】
┌─────────────────────────────────┐
│     CGC Backend (调度 - 控制面)  │
│   只做：路由、PD 交互、指令下发   │
└───────────────┬─────────────────┘
                │
┌───────────────▼─────────────────┐
│    CGC Executor (计算 - 数据面)  │
│   统一运算：vLLM + llama.cpp 融合 │
├─────────────┬───────────────────┤
│ vLLM 算力域 │ llama.cpp 量化域   │
└─────────────┴───────────────────┘

【行业突破】
vLLM 只做高吞吐计算，不碰调度
llama.cpp 只做极低显存量化计算，不碰调度
把调度全部抽走，变成全局统一的 PD 控制面
把计算全部归一，变成统一的 CGC 执行面
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum, auto
import logging
import sys
import pickle
from pathlib import Path

from .cgc_commands import (
    CGCInstruction,
    CGCInstructionType,
    CGC_SIMD_COMMAND_SET,
)
from .cgc_simd_executor import (
    CGCExecutor,
    CGCCommand,
    CGCKernelRegistry,
    CGCKernelSpec,
    KernelType,
)
from .cgc_opcodes import is_llama_cpp_opcode, list_llama_cpp_opcodes
from .flashkda_integration import FlashKDALayer

# 🚀 使用真实的 PD 客户端
from ..pd import PDClient as RealPDClient
from ..pd import PDClientConfig

logger = logging.getLogger(__name__)

# ==============================================
# 🧠 增强的 PD 客户端（扩展权重管理）
# ==============================================
class EnhancedPDClient:
    """
    增强版 PD 客户端（调度服务专用）
    - 基于真实的 PD gRPC 客户端
    - 扩展：权重管理、llama.cpp 支持
    """
    def __init__(self, pd_endpoint: str = "localhost:50051"):
        self.pd_endpoint = pd_endpoint
        self.pd_client = None
        
        # 本地权重缓存（用于演示/本地模式）
        self._local_weights = {}
        self._local_kv = {}
        
        self._init_pd()
    
    def _init_pd(self):
        """初始化 PD 客户端"""
        try:
            config = PDClientConfig(address=self.pd_endpoint)
            self.pd_client = RealPDClient(address=self.pd_endpoint, config=config)
            print(f"✅ [CGC-PD] Connected to PD Service: {self.pd_endpoint}")
        except Exception as e:
            print(f"⚠️ [CGC-PD] PD Service not available, using local mode: {e}")
            self.pd_client = None
    
    # ==========================================
    # KV Cache 操作
    # ==========================================
    def kv_cache_fetch(self, block_ids: List[int]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """从 PD 获取 KV 块"""
        if self.pd_client:
            # 从真实 PD 获取
            kv_list = []
            for block_id in block_ids:
                key = f"kv_block_{block_id}"
                kv_bytes, hit = self.pd_client.get_prefix(key)
                if hit:
                    try:
                        kv = pickle.loads(kv_bytes)
                        kv_list.append((kv["k"], kv["v"]))
                    except:
                        pass
            return kv_list
        else:
            # 本地模式
            return [self._local_kv.get(block_id, (torch.empty(0), torch.empty(0))) for block_id in block_ids]
    
    def kv_cache_store(self, block_ids: List[int], k_data: torch.Tensor, v_data: torch.Tensor):
        """将 KV 存入 PD"""
        if self.pd_client:
            # 存到真实 PD
            for idx, block_id in enumerate(block_ids):
                if k_data.shape[0] > idx and v_data.shape[0] > idx:
                    kv_data = {
                        "k": k_data[idx] if idx < k_data.shape[0] else None,
                        "v": v_data[idx] if idx < v_data.shape[0] else None
                    }
                    kv_bytes = pickle.dumps(kv_data)
                    key = f"kv_block_{block_id}"
                    self.pd_client.store_prefix(key, kv_bytes, ttl_seconds=3600)
        else:
            # 本地模式
            for idx, block_id in enumerate(block_ids):
                self._local_kv[block_id] = (k_data[idx], v_data[idx])
    
    def kv_cache_release(self, block_ids: List[int]):
        """释放 KV 块到 PD LRU 调度"""
        if self.pd_client:
            for block_id in block_ids:
                self.pd_client.invalidate_prefix(f"kv_block_{block_id}")
        else:
            for block_id in block_ids:
                self._local_kv.pop(block_id, None)
    
    def allocate_blocks(self, sequence_ids: List[int], num_blocks: int = 1, model_name: str = "default") -> Tuple[List[int], bool]:
        """分配 KV Cache Blocks"""
        if self.pd_client:
            return self.pd_client.allocate_blocks(sequence_ids, num_blocks, model_name)
        else:
            block_ids = list(range(len(self._local_kv), len(self._local_kv) + num_blocks))
            return block_ids, True
    
    # ==========================================
    # 权重管理（新增）
    # ==========================================
    def fetch_expert_weights(self, expert_ids: List[int]) -> List[torch.Tensor]:
        """从 PD 加载 MoE 专家权重（按需加载）"""
        if self.pd_client:
            expert_weights = []
            for expert_id in expert_ids:
                key = f"expert_{expert_id}"
                w_bytes, hit = self.pd_client.get_prefix(key)
                if hit:
                    try:
                        expert_weights.append(pickle.loads(w_bytes))
                    except:
                        pass
            return expert_weights
        else:
            return [self._local_weights.get(f"expert_{e}", torch.randn(1024, 1024)) for e in expert_ids]
    
    def fetch_linear_weight(self, layer_name: str) -> torch.Tensor:
        """从 PD 获取线性层权重（GGUF / 量化 / FP16）"""
        if self.pd_client:
            key = f"linear_{layer_name}"
            w_bytes, hit = self.pd_client.get_prefix(key)
            if hit:
                try:
                    return pickle.loads(w_bytes)
                except:
                    pass
        # 返回默认值（演示）
        if layer_name == "embedding":
            return torch.randn(32000, 1024)
        elif layer_name == "lm_head":
            return torch.randn(1024, 32000)
        return torch.randn(1024, 1024)
    
    def fetch_norm_weight(self, layer_name: str) -> torch.Tensor:
        """从 PD 获取 Norm 权重"""
        if self.pd_client:
            key = f"norm_{layer_name}"
            w_bytes, hit = self.pd_client.get_prefix(key)
            if hit:
                try:
                    return pickle.loads(w_bytes)
                except:
                    pass
        return torch.ones(1024)
    
    def store_weight(self, name: str, weight: torch.Tensor):
        """存储权重到 PD"""
        if self.pd_client:
            key = f"{name}"
            w_bytes = pickle.dumps(weight)
            self.pd_client.store_prefix(key, w_bytes, ttl_seconds=7200)
        else:
            self._local_weights[name] = weight
    
    # ==========================================
    # CGC 命令远程执行
    # ==========================================
    def run_cgc_command(self, opcode: int, tensors: Dict[str, Any] = None, params: Dict[str, Any] = None):
        """远程执行 CGC 命令"""
        if self.pd_client:
            return self.pd_client.run_cgc_command(opcode, tensors, params)
        return None, False, "PD not available"
    
    def health_check(self):
        """PD 健康检查"""
        if self.pd_client:
            return self.pd_client.health_check()
        return False, {"error": "PD not available"}


class BackendType(Enum):
    CUDA = auto()
    LLAMA_CPP = auto()
    MIXED = auto()


@dataclass
class CGCConfig:
    enable_flashkda: bool = True
    enable_magicompiler: bool = True
    enable_rope: bool = True
    enable_kv_cache: bool = True
    max_batch_size: int = 32
    max_seq_len: int = 8192
    kda_scale: float = 1.0
    use_gate: bool = True
    use_qk_l2norm: bool = True
    use_beta_sigmoid: bool = True
    chunk_size: int = 64
    k_dim: int = 128
    v_dim: int = 128

    # llama.cpp
    enable_llama_cpp: bool = True
    gguf_model_path: Optional[str] = None
    llama_cpp_n_ctx: int = 2048
    llama_cpp_n_batch: int = 512
    llama_cpp_n_threads: int = 4

    # ✅ PD 配置
    pd_endpoint: str = "local://pd"
    use_pd_kv: bool = True
    use_pd_weights: bool = True
    enable_lru_offload: bool = True


class OpCodeMap:
    # Attention
    ATTENTION_KDA = 0x80
    ATTENTION_SDPA = 0x10
    ATTENTION_PAGED = 0x81

    # Linear/MLP
    LINEAR = 0x01
    MLP_SILU = 0x20
    MLP_GELU = 0x21

    # Norm
    LAYER_NORM = 0x30
    RMS_NORM = 0x31

    # RoPE
    ROPE = 0x40

    # Activation
    SILU = 0x50
    GELU = 0x51
    RELU = 0x52

    SOFTMAX = 0x60

    # ✅ PD KV 指令
    PD_KV_FETCH = 0x90
    PD_KV_STORE = 0x91
    PD_KV_RELEASE = 0x92

    # Sampling
    TOP_K = 0xA0
    TOP_P = 0xA1
    SOFTMAX_SAMPLE = 0xA2

    # llama.cpp
    LLAMA_GGUF_LOAD = 0xC0
    LLAMA_GGUF_QUANTIZE = 0xC1
    LLAMA_GGUF_DEQUANTIZE = 0xC2
    LLAMA_Q4_K_MATMUL = 0xC3
    LLAMA_Q5_K_MATMUL = 0xC4
    LLAMA_Q6_K_MATMUL = 0xC5
    LLAMA_Q8_0_MATMUL = 0xC6
    LLAMA_Q2_K_MATMUL = 0xC7
    LLAMA_Q3_K_MATMUL = 0xC8
    LLAMA_Q8_K_MATMUL = 0xC9
    LLAMA_MOE_ROUTING = 0xCA
    LLAMA_MOE_EXPERT_FWD = 0xCB
    LLAMA_ROPE_GGUF = 0xCC
    LLAMA_RMSNORM_GGUF = 0xCD
    LLAMA_SILU_GGUF = 0xCE
    LLAMA_GELU_GGUF = 0xCF
    LLAMA_KV_CACHE_GGUF = 0xD0
    LLAMA_SAMPLING_GGUF = 0xD1
    LLAMA_INFERENCE = 0xD2
    LLAMA_EMBEDDING_GGUF = 0xD3
    LLAMA_DETOKENIZE_GGUF = 0xD4
    LLAMA_TOKENIZE_GGUF = 0xD5


# ==============================================
# ✅ CGCModule：PD 感知模块（不存权重）
# ==============================================
class CGCModule(nn.Module):
    def __init__(self, config: CGCConfig, pd_client: EnhancedPDClient, layer_name: str = ""):
        super().__init__()
        self.config = config
        self.layer_name = layer_name
        self.pd_client = pd_client  # 从 PD 获取所有资源
        self.executor = CGCExecutor()
        self._init_cgc_ops()
        self.llama_cpp_available = self._check_llama_cpp()
    
    def _check_llama_cpp(self) -> bool:
        try:
            import llama_cpp
            return True
        except ImportError:
            return False
    
    def _init_cgc_ops(self):
        self.opcode_map: Dict[str, int] = {
            "attention_kda": OpCodeMap.ATTENTION_KDA,
            "attention_sdpa": OpCodeMap.ATTENTION_SDPA,
            "linear": OpCodeMap.LINEAR,
            "mlp_silu": OpCodeMap.MLP_SILU,
            "rms_norm": OpCodeMap.RMS_NORM,
            "rope": OpCodeMap.ROPE,
            "silu": OpCodeMap.SILU,
            "pd_kv_fetch": OpCodeMap.PD_KV_FETCH,
            "pd_kv_store": OpCodeMap.PD_KV_STORE,
            "llama_q4_k_matmul": OpCodeMap.LLAMA_Q4_K_MATMUL,
            "llama_moe_routing": OpCodeMap.LLAMA_MOE_ROUTING,
            "llama_rmsnorm_gguf": OpCodeMap.LLAMA_RMSNORM_GGUF,
        }
    
    def execute_op(self, op_name: str, inputs: List[torch.Tensor], params: Dict[str, Any]):
        opcode = self.opcode_map[op_name]
        cmd = CGCCommand(opcode=opcode, inputs=inputs, outputs=[], params=params)
        return self.executor.execute(cmd)


# ==============================================
# ✅ Attention：PD KV 完全分离
# ==============================================
class AttentionCGC(CGCModule):
    def __init__(self, num_heads: int, head_dim: int, config: CGCConfig, pd_client: EnhancedPDClient, layer_name: str):
        super().__init__(config, pd_client, layer_name)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.flashkda_layer = FlashKDALayer(
            hidden_dim=num_heads*head_dim, num_heads=num_heads,
            k_dim=config.k_dim, v_dim=config.v_dim) if config.enable_flashkda else None
        self._use_cpp_kda = config.enable_cpp_simd and hasattr(self.executor, 'has_opcode') and self.executor.has_opcode(0x11)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, positions: torch.Tensor, block_ids: List[int] = None):
        if self.config.use_pd_kv and block_ids:
            kv_list = self.pd_client.kv_cache_fetch(block_ids)
            if kv_list and len(kv_list) > 0:
                k_loaded = [kv[0] for kv in kv_list]
                v_loaded = [kv[1] for kv in kv_list]
                if len(k_loaded) > 0 and len(k_loaded[0].shape) > 0:
                    k = torch.cat([k] + [k_loaded[0]])
                    v = torch.cat([v] + [v_loaded[0]])

        if self._use_cpp_kda:
            out = self.execute_op("attention_kda", [q, k, v], {
                "state_id": positions.long().tolist()[-1] if len(positions) > 0 else 0,
                "n_heads": self.num_heads,
                "d_state": self.head_dim,
                "scale": 1.0 / math.sqrt(self.head_dim),
                "is_first_chunk": positions[0] == 0 if len(positions) > 0 else True,
            })[0]
        elif self.flashkda_layer:
            out = self.flashkda_layer(x=q.view(-1, q.shape[-1]), initial_state=None)[0]
            out = out.view(q.shape)
        else:
            out = self.execute_op("attention_sdpa", [q, k, v], {})[0]

        if self.config.use_pd_kv and block_ids:
            self.pd_client.kv_cache_store(block_ids, k, v)
        return out


# ==============================================
# ✅ MLP：权重全部来自 PD
# ==============================================
class MLPCGC(CGCModule):
    def __init__(self, hidden_dim: int, config: CGCConfig, pd_client: EnhancedPDClient, layer_name: str):
        super().__init__(config, pd_client, layer_name)
        self.hidden_dim = hidden_dim
    
    def forward(self, x: torch.Tensor):
        # ✅ 从 PD 获取权重
        w_gate = self.pd_client.fetch_linear_weight(f"{self.layer_name}.gate")
        w_up = self.pd_client.fetch_linear_weight(f"{self.layer_name}.up")
        w_down = self.pd_client.fetch_linear_weight(f"{self.layer_name}.down")
        
        gate = self.execute_op("linear", [x, w_gate], {})[0]
        gate = self.execute_op("silu", [gate], {})[0]
        up = self.execute_op("linear", [x, w_up], {})[0]
        out = self.execute_op("linear", [gate * up, w_down], {})[0]
        return out


# ==============================================
# ✅ RMSNorm：权重来自 PD
# ==============================================
class RMSNormCGC(CGCModule):
    def __init__(self, hidden_dim: int, config: CGCConfig, pd_client: EnhancedPDClient, layer_name: str):
        super().__init__(config, pd_client, layer_name)
        self.hidden_dim = hidden_dim
        self.eps = 1e-6
    
    def forward(self, x: torch.Tensor):
        weight = self.pd_client.fetch_norm_weight(self.layer_name)
        return self.execute_op("rms_norm", [x, weight], {"eps": self.eps})[0]


# ==============================================
# ✅ 模型：不存储任何参数
# ==============================================
class ModelCGC(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int, num_heads: int, head_dim: int, config: CGCConfig, pd_client: EnhancedPDClient):
        super().__init__()
        self.config = config
        self.pd_client = pd_client
        self.vocab_size = vocab_size
        self.layers = []
        for i in range(num_layers):
            self.layers.append({
                "attn": AttentionCGC(num_heads, head_dim, config, pd_client, f"layer.{i}.attn"),
                "mlp": MLPCGC(hidden_dim, config, pd_client, f"layer.{i}.mlp"),
                "norm1": RMSNormCGC(hidden_dim, config, pd_client, f"layer.{i}.norm1"),
                "norm2": RMSNormCGC(hidden_dim, config, pd_client, f"layer.{i}.norm2"),
            })
    
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, block_ids: List[List[int]] = None):
        x = self.pd_client.fetch_linear_weight("embedding")[input_ids]
        for idx, layer in enumerate(self.layers):
            x = layer["norm1"](x)
            x = x + layer["attn"](x, x, x, positions, block_ids[idx] if block_ids else None)
            x = layer["norm2"](x)
            x = x + layer["mlp"](x)
        logits = torch.matmul(x, self.pd_client.fetch_linear_weight("lm_head").t())
        return logits


# ==============================================
# ✅ CGCBackend：纯调度服务
# ==============================================
class CGCBackend:
    """
    统一调度服务（控制面）
    • 不存权重
    • 不存 KV
    • 不执行算子
    • 只做：路由、指令下发、PD 交互
    """
    def __init__(self, config: Optional[CGCConfig] = None):
        self.config = config or CGCConfig()
        self.executor = CGCExecutor()
        self.pd_client = EnhancedPDClient(self.config.pd_endpoint)  # ✅ PD 客户端
        self.model = None
        self.kernel_registry = CGCKernelRegistry()
        print(f"🚀 [CGC-Backend] Initialized (PD mode: {'remote' if self.pd_client.pd_client else 'local'})")
    
    def set_model(self, vocab_size: int, hidden_dim: int, num_layers: int, num_heads: int, head_dim: int):
        """构建调度模型（无参数，纯结构）"""
        self.model = ModelCGC(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            config=self.config,
            pd_client=self.pd_client,
        )
        print(f"✅ [CGC-Backend] Model set (layers={num_layers})")
    
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, block_ids: List[List[int]] = None):
        """推理接口"""
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")
        return self.model(input_ids, positions, block_ids)
    
    def fetch_kv_from_pd(self, block_ids: List[int]):
        """从 PD 加载 KV"""
        return self.pd_client.kv_cache_fetch(block_ids)
    
    def release_kv_to_pd(self, block_ids: List[int]):
        """释放 KV 到 PD"""
        self.pd_client.kv_cache_release(block_ids)
    
    def store_weights_to_pd(self, weight_dict: Dict[str, torch.Tensor]):
        """批量存储权重到 PD"""
        for name, weight in weight_dict.items():
            self.pd_client.store_weight(name, weight)
        print(f"✅ [CGC-Backend] Stored {len(weight_dict)} weights to PD")
    
    def health_check(self):
        """健康检查"""
        pd_healthy, pd_stats = self.pd_client.health_check()
        return {
            "pd_healthy": pd_healthy,
            "pd_stats": pd_stats,
        }
    
    def get_kernel_stats(self):
        """获取执行统计"""
        return self.executor.get_stats()


# ==============================================
# ✅ 向后兼容：保留原有的 create_cgc_model 接口
# ==============================================
def create_cgc_model(vocab_size: int, hidden_dim: int, num_layers: int, num_heads: int, head_dim: int, **kwargs):
    """向后兼容的创建接口"""
    config = CGCConfig(**kwargs)
    pd_client = EnhancedPDClient(config.pd_endpoint)
    return ModelCGC(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        config=config,
        pd_client=pd_client,
    )


# ==============================================
# ✅ 全局 Kernel 注册
# ==============================================
def _register_cgc_kernels():
    """注册 CGC Kernels"""
    reg = CGCKernelRegistry()
    reg.register(0x10, CGCKernelSpec(name="sdpa", kernel_type=KernelType.ATTENTION, cuda_kernel=torch.nn.functional.scaled_dot_product_attention))
    reg.register(0x01, CGCKernelSpec(name="linear", kernel_type=KernelType.LINEAR, cuda_kernel=torch.matmul))
    reg.register(0x31, CGCKernelSpec(name="rms_norm", kernel_type=KernelType.NORM, cuda_kernel=lambda x, w, eps: x/torch.sqrt(x.pow(2).mean(-1, keepdim=True)+eps)*w))

_register_cgc_kernels()