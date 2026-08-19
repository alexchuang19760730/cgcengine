#!/usr/bin/env python3
"""
MLX-Tune Integration - 使用真正的MLX Custom Backend
已迁移到 mlx_custom_backend.py
"""

import sys
import torch
from typing import Dict, Optional, List

# ================================================
# 导入新的MLX Custom Backend
# ================================================
try:
    from .mlx_custom_backend import (
        mlx_custom_backend,
        mlx_lora_forward,
        mlx_flash_kda,
        mlx_rope,
        KVCache,
        CGCOpcodeEngine,
    )
    MLX_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ MLX Custom Backend导入失败: {e}")
    MLX_AVAILABLE = False

# ================================================
# CGC Opcode定义（保持兼容）
# ================================================
class CGC_OP_CODES:
    # 原有定义保持不变
    LORA_A_MATMUL = 0xA0
    LORA_B_MATMUL = 0xA1
    LORA_MERGE = 0xA2
    QLORA_DEQUANT = 0xA3
    LORA_SCATTER = 0xA4
    LORA_GRAD = 0xA5
    MLX_TUNE_FWD = 0xB0
    MLX_TUNE_BWD = 0xB1
    KDA_LORA_FUSE = 0xB2
    LORA_ROPE_FUSE = 0xB3
    LORA_GELU_FUSE = 0xB4
    LORA_ATTN_FUSE = 0xB5
    MLX_ROPE_FUSE = 0xB6
    MLX_KDA_FUSE = 0xB7
    MLX_FLASH_ATTN = 0xB8
    MLX_KVCACHE = 0xB9
    MLX_GELU_FUSE = 0xBA
    MLX_RMS_NORM = 0xBB
    MLX_SAMPLING_TOPK = 0xBC
    MLX_KV_CACHE = 0xBD
    MLX_QUANTIZE = 0xBE
    MLX_DEQUANTIZE = 0xBF
    
    # 新增加的MLX原生Opcode
    MLX_LORA_FWD = 0xC0
    MLX_FLASH_KDA = 0xC1
    MLX_ROPE = 0xC2

# ================================================
# CGC MLX Tune Backend（使用新的Custom Backend）
# ================================================

class CGCMLXTuneBackend:
    """
    CGC MLX Tune后端 - 使用真正的MLX Custom Backend
    
    功能：
    - LoRA/QLoRA支持
    - FlashKDA融合
    - CGC Opcode执行
    - KV Cache管理
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance
    
    def _init(self):
        """初始化后端"""
        self.backend = None
        self.flash_kda = None
        self._mx_cache = {}
        self._compiled = {}
        
        if not MLX_AVAILABLE:
            print("⚠️ MLX不可用，MLX-Tune后端已禁用")
            self.flash_kda_available = False
            return
        
        self.backend = mlx_custom_backend
        
        # 初始化FlashKDA
        try:
            self.backend.init_flash_kda(head_dim=64, lora_rank=8)
            self.flash_kda_available = True
        except Exception as e:
            print(f"⚠️ FlashKDA初始化失败: {e}")
            self.flash_kda_available = False
            self.flash_kda_available = False
        
        print("✅ CGCMLXTuneBackend (MLX Custom Backend) 初始化完成")

    def _to_mx_cached(self, t):
        import mlx.core as mx

        key = (id(t), getattr(t, "_version", None), str(t.dtype))
        v = self._mx_cache.get(key)
        if v is not None:
            return v
        t_cpu = t.detach().cpu()
        if hasattr(torch.utils, "dlpack") and hasattr(torch.utils.dlpack, "to_dlpack"):
            try:
                if not t_cpu.is_contiguous():
                    t_cpu = t_cpu.contiguous()
                arr = mx.array(torch.utils.dlpack.to_dlpack(t_cpu))
            except Exception:
                arr = mx.array(t_cpu)
        else:
            arr = mx.array(t_cpu)
        self._mx_cache[key] = arr
        return arr

    def _mx_to_torch(self, x, *, device, dtype):
        if hasattr(torch.utils, "dlpack") and hasattr(torch.utils.dlpack, "from_dlpack") and hasattr(x, "__dlpack__"):
            try:
                out = torch.utils.dlpack.from_dlpack(x)
            except Exception:
                out = torch.tensor(x)
        else:
            out = torch.tensor(x)
        return out.to(device=device, dtype=dtype)
    
    def run_cgc_command(self, opcode: int, tensors: dict, **kwargs):
        """执行CGC命令（路由到新的Custom Backend）"""
        if not MLX_AVAILABLE:
            return self._fallback_to_torch(opcode, tensors, **kwargs)
        
        # 路由到新的MLX Custom Backend
        return self.backend.run_opcode(opcode, tensors, **kwargs)
    
    def _fallback_to_torch(self, opcode: int, tensors: dict, **kwargs):
        """PyTorch降级实现（保持兼容性）"""
        import torch
        
        if opcode == CGC_OP_CODES.MLX_GELU_FUSE:
            x = tensors["x"]
            return torch.nn.functional.gelu(x)
        
        elif opcode == CGC_OP_CODES.MLX_RMS_NORM:
            x = tensors["x"]
            weight = tensors.get("weight")
            if weight is not None:
                return torch.nn.functional.rms_norm(x, weight.shape, weight)
            return torch.nn.functional.rms_norm(x, x.shape[-1])
        
        elif opcode == CGC_OP_CODES.MLX_SAMPLING_TOPK:
            logits = tensors["logits"]
            k = tensors.get("k", 1)
            values, indices = torch.topk(logits, k=k, dim=-1)
            return indices
        
        elif opcode == CGC_OP_CODES.MLX_KV_CACHE:
            k = tensors["k"]
            v = tensors["v"]
            k_cache = tensors.get("k_cache")
            v_cache = tensors.get("v_cache")
            if k_cache is not None and v_cache is not None:
                k_cache = torch.cat([k_cache, k], dim=1)
                v_cache = torch.cat([v_cache, v], dim=1)
                return k_cache, v_cache
            return k, v
        
        else:
            raise NotImplementedError(f"Opcode {opcode:#x} 不支持")
    
    # ============================================
    # 便捷方法（保持兼容性）
    # ============================================
    
    def lora_forward(self, x, w, a, b, scale=1.0):
        """LoRA前向"""
        if MLX_AVAILABLE:
            import mlx.core as mx
            x_mlx = mx.array(x.detach().cpu())
            w_mlx = self._to_mx_cached(w)
            a_mlx = self._to_mx_cached(a)
            b_mlx = self._to_mx_cached(b)
            out = mlx_lora_forward(x_mlx, w_mlx, a_mlx, b_mlx, scale)
            return self._mx_to_torch(out, device=x.device, dtype=x.dtype)
        else:
            base_out = torch.matmul(x, w.t())
            lora_out = torch.matmul(torch.matmul(x, a.t()), b.t())
            return base_out + scale * lora_out
    
    def flash_kda_forward(self, q, k, v):
        """FlashKDA前向"""
        if MLX_AVAILABLE:
            import mlx.core as mx
            q_mlx = mx.array(q.detach().cpu())
            k_mlx = mx.array(k.detach().cpu())
            v_mlx = mx.array(v.detach().cpu())
            out = mlx_flash_kda(q_mlx, k_mlx, v_mlx)
            return self._mx_to_torch(out, device=q.device, dtype=q.dtype)
        else:
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)
    
    def rope_forward(self, x, cos, sin):
        """RoPE前向"""
        if MLX_AVAILABLE:
            import mlx.core as mx
            x_mlx = mx.array(x.detach().cpu())
            cos_mlx = mx.array(cos.detach().cpu())
            sin_mlx = mx.array(sin.detach().cpu())
            out = mlx_rope(x_mlx, cos_mlx, sin_mlx)
            return self._mx_to_torch(out, device=x.device, dtype=x.dtype)
        else:
            rotary = x * cos + self._rotate_half(x) * sin
            return rotary

    def lora_rope_forward(self, x, w, a, b, scale=1.0):
        if MLX_AVAILABLE:
            import mlx.core as mx

            x_mlx = mx.array(x.detach().cpu())
            w_mlx = self._to_mx_cached(w)
            a_mlx = self._to_mx_cached(a)
            b_mlx = self._to_mx_cached(b)
            key = (int(x_mlx.shape[-1]), str(x_mlx.dtype), float(scale))
            compiled = self._compiled.get(key)
            if compiled is None:
                scale_const = float(scale)

                def _f(xi, wi, ai, bi):
                    yi = mlx_lora_forward(xi, wi, ai, bi, scale_const)
                    cosi = mx.cos(yi)
                    sini = mx.sin(yi)
                    return mlx_rope(yi, cosi, sini)

                compiled = mx.compile(_f)
                self._compiled[key] = compiled

            z = compiled(x_mlx, w_mlx, a_mlx, b_mlx)
            return self._mx_to_torch(z, device=x.device, dtype=x.dtype)

        base_out = torch.matmul(x, w.t())
        lora_out = torch.matmul(torch.matmul(x, a.t()), b.t())
        y = base_out + scale * lora_out
        cos = torch.cos(y)
        sin = torch.sin(y)
        return y * cos + self._rotate_half(y) * sin
    
    def _rotate_half(self, x):
        """RoPE旋转"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

# ================================================
# 全局实例
# ================================================

cgc_mlx_tune = CGCMLXTuneBackend()

# ================================================
# 便捷函数（保持兼容性）
# ================================================

def get_mlx_tune_info():
    """获取MLX-Tune信息"""
    return {
        "mlx_available": MLX_AVAILABLE,
        "flashkda_available": cgc_mlx_tune.flash_kda_available,
        "backend": "MLX Custom Backend",
        "finetuning_opcodes": {
            "LORA_A_MATMUL": CGC_OP_CODES.LORA_A_MATMUL,
            "LORA_B_MATMUL": CGC_OP_CODES.LORA_B_MATMUL,
            "LORA_MERGE": CGC_OP_CODES.LORA_MERGE,
            "QLORA_DEQUANT": CGC_OP_CODES.QLORA_DEQUANT,
            "LORA_SCATTER": CGC_OP_CODES.LORA_SCATTER,
            "LORA_GRAD": CGC_OP_CODES.LORA_GRAD,
            "MLX_TUNE_FWD": CGC_OP_CODES.MLX_TUNE_FWD,
            "MLX_TUNE_BWD": CGC_OP_CODES.MLX_TUNE_BWD,
            "KDA_LORA_FUSE": CGC_OP_CODES.KDA_LORA_FUSE,
            "LORA_ROPE_FUSE": CGC_OP_CODES.LORA_ROPE_FUSE,
            "LORA_GELU_FUSE": CGC_OP_CODES.LORA_GELU_FUSE,
            "LORA_ATTN_FUSE": CGC_OP_CODES.LORA_ATTN_FUSE,
            "MLX_ROPE_FUSE": CGC_OP_CODES.MLX_ROPE_FUSE,
            "MLX_KDA_FUSE": CGC_OP_CODES.MLX_KDA_FUSE,
            "MLX_FLASH_ATTN": CGC_OP_CODES.MLX_FLASH_ATTN,
            "MLX_KVCACHE": CGC_OP_CODES.MLX_KVCACHE,
            "MLX_GELU_FUSE": CGC_OP_CODES.MLX_GELU_FUSE,
            "MLX_RMS_NORM": CGC_OP_CODES.MLX_RMS_NORM,
            "MLX_SAMPLING_TOPK": CGC_OP_CODES.MLX_SAMPLING_TOPK,
            "MLX_KV_CACHE": CGC_OP_CODES.MLX_KV_CACHE,
            "MLX_QUANTIZE": CGC_OP_CODES.MLX_QUANTIZE,
            "MLX_DEQUANTIZE": CGC_OP_CODES.MLX_DEQUANTIZE,
            "MLX_LORA_FWD": CGC_OP_CODES.MLX_LORA_FWD,
            "MLX_FLASH_KDA": CGC_OP_CODES.MLX_FLASH_KDA,
            "MLX_ROPE": CGC_OP_CODES.MLX_ROPE,
        },
    }

def mlx_lora_fwd(x, w, a, b, scale=1.0):
    """MLX LoRA前向便捷函数"""
    return cgc_mlx_tune.lora_forward(x, w, a, b, scale)

def mlx_flash_kda_fwd(q, k, v):
    """MLX FlashKDA前向便捷函数"""
    return cgc_mlx_tune.flash_kda_forward(q, k, v)

def mlx_rope_fwd(x, cos, sin):
    """MLX RoPE前向便捷函数"""
    return cgc_mlx_tune.rope_forward(x, cos, sin)


def mlx_lora_rope_fwd(x, w, a, b, scale=1.0):
    return cgc_mlx_tune.lora_rope_forward(x, w, a, b, scale)

# ================================================
# 兼容性类（保持与旧版API兼容）
# ================================================

class LoRAManager:
    """LoRA 权重管理器"""

    def __init__(self):
        self.lora_weights: Dict[str, Dict[str, torch.Tensor]] = {}

    def register_lora(self, name: str, lora_a: torch.Tensor, lora_b: torch.Tensor, scale: float = 1.0):
        self.lora_weights[name] = {
            "lora_a": lora_a,
            "lora_b": lora_b,
            "scale": scale,
        }

    def get_lora(self, name: str) -> Optional[Dict[str, torch.Tensor]]:
        return self.lora_weights.get(name)


class LoRAConfig:
    """LoRA 配置"""
    def __init__(self, rank: int = 8, alpha: float = 16.0, target_modules: list = None):
        self.rank = rank
        self.alpha = alpha
        self.target_modules = target_modules or []


class QLoRAConfig:
    """QLoRA 配置"""
    def __init__(self, rank: int = 8, alpha: float = 16.0, bits: int = 4, target_modules: list = None):
        self.rank = rank
        self.alpha = alpha
        self.bits = bits
        self.target_modules = target_modules or []


class FineTuningMode:
    """微调模式枚举"""
    LORA = "lora"
    QLORA = "qlora"
    LISA = "lisa"
    FULL = "full"


class LISAFineTuner:
    """LISA (Low-Rank Subspace Adaptation) 微调器"""
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def fine_tune(self, data):
        pass


def is_finetuning_opcode(opcode: int) -> bool:
    """检查 opcode 是否为微调相关"""
    return 0xB0 <= opcode <= 0xBF


def is_lora_opcode(opcode: int) -> bool:
    """检查 opcode 是否为 LoRA 相关"""
    return 0xB0 <= opcode <= 0xB5
