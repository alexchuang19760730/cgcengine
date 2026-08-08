"""统一 MTP IR — 一份算法描述, 自动生成 PyTorch / MLX / sglang 代码.

定义 MTP head 的计算图一次, 通过 IR Compiler 自动生成:
  - PyTorch model (训练)
  - MLX model (Mac 推理)
  - sglang EAGLE draft config (cloud 投机)

改 IR 一处 → 三后端自动同步。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ============================================================================
# IR 定义
# ============================================================================

@dataclass
class MTPLayerSpec:
    """单层 IR 规格 (对齐 CGC LayerSpec)."""
    name: str                    # "proj" | "norm1" | "attn" | "mlp" | "norm_out" | "lm_head"
    layer_type: str              # "linear" | "rms_norm" | "attention" | "mlp_silu"
    input_shape: List[int]       # 输入 shape
    output_shape: List[int]      # 输出 shape
    params: Dict[str, Any] = field(default_factory=dict)
    # linear: {"in_features": int, "out_features": int, "bias": bool}
    # rms_norm: {"dim": int, "eps": float}
    # attention: {"hidden_size": int, "num_heads": int, "head_dim": int}
    # mlp_silu: {"hidden_size": int, "intermediate_size": int}


@dataclass
class MTPHeadIR:
    """MTP head 统一 IR — 模型级计算图."""
    name: str = "mtp_head"
    version: str = "1.0"

    # 模型配置
    hidden_size: int = 2048
    vocab_size: int = 151936
    num_heads: int = 16
    head_dim: int = 128
    intermediate_size: int = 5632
    rms_norm_eps: float = 1e-6

    # 训练配置
    training: Dict[str, Any] = field(default_factory=lambda: {
        "lr": 1e-4,
        "epochs": 3,
        "batch_size": 32,
        "optimizer": "adamw",
        "loss": "cross_entropy",
        "chained": False,  # 是否链式训练 (方案 1)
    })

    # 推理配置
    inference: Dict[str, Any] = field(default_factory=lambda: {
        "quantization": None,  # None | "4bit"
        "lm_head_shared": True,  # 是否共享 target lm_head
        "cache_mode": None,  # None | "kv_cache"
    })

    # 计算图 (层列表)
    layers: List[MTPLayerSpec] = field(default_factory=list)

    # 权重映射 (IR name → checkpoint key)
    weight_map: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """自动构建计算图."""
        if not self.layers:
            self._build_graph()

    def _build_graph(self):
        """构建 MTP head 计算图."""
        H = self.hidden_size
        self.layers = [
            MTPLayerSpec(
                name="proj", layer_type="linear",
                input_shape=[1, 1, H * 2], output_shape=[1, 1, H],
                params={"in_features": H * 2, "out_features": H, "bias": False},
            ),
            MTPLayerSpec(
                name="norm1", layer_type="rms_norm",
                input_shape=[1, 1, H], output_shape=[1, 1, H],
                params={"dim": H, "eps": self.rms_norm_eps},
            ),
            MTPLayerSpec(
                name="attn", layer_type="attention",
                input_shape=[1, 1, H], output_shape=[1, 1, H],
                params={"hidden_size": H, "num_heads": self.num_heads, "head_dim": self.head_dim},
            ),
            MTPLayerSpec(
                name="norm2", layer_type="rms_norm",
                input_shape=[1, 1, H], output_shape=[1, 1, H],
                params={"dim": H, "eps": self.rms_norm_eps},
            ),
            MTPLayerSpec(
                name="mlp", layer_type="mlp_silu",
                input_shape=[1, 1, H], output_shape=[1, 1, H],
                params={"hidden_size": H, "intermediate_size": self.intermediate_size},
            ),
            MTPLayerSpec(
                name="norm_out", layer_type="rms_norm",
                input_shape=[1, 1, H], output_shape=[1, 1, H],
                params={"dim": H, "eps": self.rms_norm_eps},
            ),
        ]

        # 权重映射
        self.weight_map = {
            "proj": "proj.weight",
            "norm1": "norm1.weight",
            "attn.q_proj": "attn.q_proj.weight",
            "attn.k_proj": "attn.k_proj.weight",
            "attn.v_proj": "attn.v_proj.weight",
            "attn.o_proj": "attn.o_proj.weight",
            "norm2": "norm2.weight",
            "mlp.gate_proj": "mlp.gate_proj.weight",
            "mlp.up_proj": "mlp.up_proj.weight",
            "mlp.down_proj": "mlp.down_proj.weight",
            "norm_out": "norm_out.weight",
        }

    def summary(self) -> Dict:
        return {
            "name": self.name,
            "version": self.version,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "num_layers": len(self.layers),
            "layer_types": [l.layer_type for l in self.layers],
            "training": self.training,
            "inference": self.inference,
        }

    def to_json(self) -> str:
        return json.dumps({
            "name": self.name,
            "version": self.version,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "rms_norm_eps": self.rms_norm_eps,
            "training": self.training,
            "inference": self.inference,
            "layers": [asdict(l) for l in self.layers],
            "weight_map": self.weight_map,
        }, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "MTPHeadIR":
        d = json.loads(json_str)
        ir = cls(
            name=d["name"], version=d["version"],
            hidden_size=d["hidden_size"], vocab_size=d["vocab_size"],
            num_heads=d["num_heads"], head_dim=d["head_dim"],
            intermediate_size=d["intermediate_size"],
            rms_norm_eps=d["rms_norm_eps"],
            training=d.get("training", {}),
            inference=d.get("inference", {}),
        )
        ir.layers = [MTPLayerSpec(**l) for l in d.get("layers", [])]
        ir.weight_map = d.get("weight_map", {})
        return ir


# ============================================================================
# IR Compiler — 生成各后端代码
# ============================================================================

class MTPHeadIRCompiler:
    """从 IR 生成各后端的 MTP head 实现."""

    def __init__(self, ir: MTPHeadIR):
        self.ir = ir

    def generate_pytorch(self) -> str:
        """生成 PyTorch model.py 代码."""
        H = self.ir.hidden_size
        NH = self.ir.num_heads
        HD = self.ir.head_dim
        INT = self.ir.intermediate_size
        EPS = self.ir.rms_norm_eps

        return f'''"""Auto-generated by MTPHeadIRCompiler from IR v{self.ir.version}."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class MTPHeadConfig:
    hidden_size: int = {H}
    vocab_size: int = {self.ir.vocab_size}
    num_heads: int = {NH}
    head_dim: int = {HD}
    intermediate_size: int = {INT}
    rms_norm_eps: float = {EPS}

class MTPRMSNorm(nn.Module):
    def __init__(self, dim, eps={EPS}):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        x32 = x.float()
        var = x32.pow(2).mean(-1, keepdim=True)
        return (self.weight * (x32 * torch.rsqrt(var + self.eps))).to(x.dtype)

class MTPAttention(nn.Module):
    def __init__(self, hidden_size={H}, num_heads={NH}, head_dim={HD}):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        inner = num_heads * head_dim
        self.q_proj = nn.Linear(hidden_size, inner, bias=False)
        self.k_proj = nn.Linear(hidden_size, inner, bias=False)
        self.v_proj = nn.Linear(hidden_size, inner, bias=False)
        self.o_proj = nn.Linear(inner, hidden_size, bias=False)
    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out)

class MTPMLP(nn.Module):
    def __init__(self, hidden_size={H}, intermediate_size={INT}):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class MTPHead(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        config = config or MTPHeadConfig()
        self.proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.norm1 = MTPRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = MTPAttention(config.hidden_size, config.num_heads, config.head_dim)
        self.norm2 = MTPRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MTPMLP(config.hidden_size, config.intermediate_size)
        self.norm_out = MTPRMSNorm(config.hidden_size, config.rms_norm_eps)
        self._shared_lm_head = None
    def set_shared_lm_head(self, w):
        self._shared_lm_head = w
    def forward(self, hidden_states, token_embedding):
        x = torch.cat([hidden_states, token_embedding], dim=-1)
        x = self.proj(x)
        x = self.norm1(x)
        x = x + self.attn(x)
        x = self.norm2(x)
        x = x + self.mlp(x)
        x = self.norm_out(x)
        if self._shared_lm_head is not None:
            return F.linear(x, self._shared_lm_head)
        return x
'''

    def generate_mlx(self) -> str:
        """生成 MLX model 代码."""
        H = self.ir.hidden_size
        NH = self.ir.num_heads
        HD = self.ir.head_dim
        INT = self.ir.intermediate_size
        EPS = self.ir.rms_norm_eps

        return f'''"""Auto-generated by MTPHeadIRCompiler from IR v{self.ir.version}."""
import mlx.core as mx
import mlx.nn as nn

class MTPRMSNorm(nn.Module):
    def __init__(self, dim, eps={EPS}):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps
    def __call__(self, x):
        x32 = x.astype(mx.float32)
        var = mx.mean(x32 * x32, axis=-1, keepdims=True)
        return (self.weight * (x32 * mx.rsqrt(var + self.eps))).astype(x.dtype)

class MTPAttention(nn.Module):
    def __init__(self, hidden_size={H}, num_heads={NH}, head_dim={HD}):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        inner = num_heads * head_dim
        self.q_proj = nn.Linear(hidden_size, inner, bias=False)
        self.k_proj = nn.Linear(hidden_size, inner, bias=False)
        self.v_proj = nn.Linear(hidden_size, inner, bias=False)
        self.o_proj = nn.Linear(inner, hidden_size, bias=False)
    def __call__(self, x, cache=None):
        B, T, _ = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if cache is not None and k.shape[2] > T:
            attn = attn[:, :, -1:, :]
        else:
            mask = mx.triu(mx.full((T, T), -1e9), k=1)
            attn = attn + mask
        attn = mx.softmax(attn, axis=-1)
        out = attn @ v
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, T, -1))

class MTPMLP(nn.Module):
    def __init__(self, hidden_size={H}, intermediate_size={INT}):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

class MTPHead(nn.Module):
    def __init__(self, hidden_size={H}, num_heads={NH}, head_dim={HD}, intermediate_size={INT}):
        super().__init__()
        self.proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.norm1 = MTPRMSNorm(hidden_size)
        self.attn = MTPAttention(hidden_size, num_heads, head_dim)
        self.norm2 = MTPRMSNorm(hidden_size)
        self.mlp = MTPMLP(hidden_size, intermediate_size)
        self.norm_out = MTPRMSNorm(hidden_size)
        self.layers = [self.attn]
    def __call__(self, hidden, embed, cache=None):
        x = mx.concatenate([hidden, embed], axis=-1)
        x = self.proj(x)
        x = self.norm1(x)
        attn_cache = cache[0] if isinstance(cache, list) else cache
        x = x + self.attn(x, attn_cache)
        x = self.norm2(x)
        x = x + self.mlp(x)
        x = self.norm_out(x)
        return x
'''

    def generate_sglang_config(self) -> str:
        """生成 sglang EAGLE draft model 配置."""
        return json.dumps({
            "model_type": "mtp_head",
            "architectures": ["MTPHeadForSpeculativeDecoding"],
            "hidden_size": self.ir.hidden_size,
            "vocab_size": self.ir.vocab_size,
            "num_heads": self.ir.num_heads,
            "head_dim": self.ir.head_dim,
            "intermediate_size": self.ir.intermediate_size,
            "rms_norm_eps": self.ir.rms_norm_eps,
            "speculative_algorithm": "EAGLE",
            "speculative_num_steps": 4,
            "speculative_num_draft_tokens": 4,
            "weight_map": self.ir.weight_map,
            "ir_version": self.ir.version,
        }, indent=2)

    def generate_launch_command(self, draft_model_path: str, target_model_path: str) -> str:
        """生成 sglang 启动命令 (含投机 decode)."""
        return f"""python -m sglang.launch_server \\
    --model-path {target_model_path} \\
    --speculative-algorithm EAGLE \\
    --speculative-draft-model-path {draft_model_path} \\
    --speculative-num-steps 4 \\
    --speculative-num-draft-tokens 4 \\
    --speculative-eagle-topk 1 \\
    --host 0.0.0.0 --port 30001 \\
    --tp-size 1 --trust-remote-code"""


# ============================================================================
# 验证
# ============================================================================

if __name__ == "__main__":
    # 1. 创建 IR
    ir = MTPHeadIR()
    print("=== MTP Head IR ===")
    print(ir.to_json()[:500])
    print("...\n")

    # 2. 编译到各后端
    compiler = MTPHeadIRCompiler(ir)

    print("=== PyTorch 代码 (前 200 字符) ===")
    pytorch_code = compiler.generate_pytorch()
    print(pytorch_code[:200])
    print(f"... ({len(pytorch_code)} chars total)\n")

    print("=== MLX 代码 (前 200 字符) ===")
    mlx_code = compiler.generate_mlx()
    print(mlx_code[:200])
    print(f"... ({len(mlx_code)} chars total)\n")

    print("=== sglang 配置 ===")
    sglang_config = compiler.generate_sglang_config()
    print(sglang_config[:300])
    print("...\n")

    print("=== sglang 启动命令 ===")
    print(compiler.generate_launch_command(
        "/data/mtp_head_output/mtp_head_sglang",
        "/data2/models/Qwen3-VL-2B-Instruct"
    ))

    # 3. 验证 IR 一致性
    ir_json = ir.to_json()
    ir_restored = MTPHeadIR.from_json(ir_json)
    assert ir_restored.hidden_size == ir.hidden_size
    assert len(ir_restored.layers) == len(ir.layers)
    print("\n=== IR 一致性验证 ===")
    print(f"✓ IR 序列化/反序列化正确")
    print(f"✓ 层数: {len(ir.layers)}")
    print(f"✓ 权重映射: {len(ir.weight_map)} keys")
    print(f"\n三后端代码生成完成:")
    print(f"  PyTorch: {len(pytorch_code)} chars")
    print(f"  MLX: {len(mlx_code)} chars")
    print(f"  sglang config: {len(sglang_config)} chars")
