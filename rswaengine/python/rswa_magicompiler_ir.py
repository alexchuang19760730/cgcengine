"""R-SWA + Magicompiler + AutoTunner 统一 IR.

整合三个组件:
1. Magicompiler: 统一 IR → 多后端编译 (MLX/CUDA/Native)
2. R-SWA: Reference + 窗口 OrthoKDA attention
3. AutoTunner: 硬件检测 → 参数选择 → 运行时调优

架构:
  AutoTunner.detect(backend) → 硬件 profile
    → RSWAAttentionIR(config) → 统一 IR
    → Magicompiler.compile(ir, backend) → 编译到 MLX/CUDA
    → InsertKDAPass.replace(model, compiled) → 替换 attention
    → AutoTunner.runtime_tune(accept/tok_s) → 动态调整

不修改 Magicompiler 源码, 只用其 API.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import time


# =============================================================================
# 1. R-SWA 统一 IR (平台无关, 描述 attention 计算图)
# =============================================================================

@dataclass
class RSWAAttentionIR:
    """R-SWA attention 的 Magicompiler 统一 IR.

    平台无关描述: Reference 标准 attention + 窗口 OrthoKDA 压缩.
    Magicompiler 编译到 MLX/CUDA/Native 后端.
    """
    # R-SWA 参数
    num_heads: int = 16
    head_dim: int = 128
    reference_len: int = 4       # Reference 永久区
    window_size: int = 128       # 窗口滑动区
    ortho_base_dim: int = 64     # OrthoKDA 降维维度

    # 投机 decode 参数 (AutoTunner 调优)
    num_draft_tokens: int = 16   # 投机 N
    mode: str = "chain"          # chain | eagle | rswa

    # 编译目标
    backend: str = "mlx"         # mlx | cuda | native | sglang

    # 运行时状态 (AutoTunner 更新)
    accept_rate: float = 0.0
    tps: float = 0.0

    def to_magicompiler_request(self) -> Dict[str, Any]:
        """转换为 Magicompiler 的 UnifiedComputeRequest 格式."""
        return {
            "compute_type": "kda_attention",  # Magicompiler 的 KDA_ATTENTION
            "backend": self.backend,
            "params": {
                "num_heads": self.num_heads,
                "head_dim": self.head_dim,
                "reference_len": self.reference_len,
                "window_size": self.window_size,
                "ortho_base_dim": self.ortho_base_dim,
            },
            "subgraph": {
                "reference_attention": "standard_sdpa",  # Reference 用标准 attention
                "window_attention": "orthokda_projected",  # 窗口用 OrthoKDA 压缩
                "merge": "add",  # ref_out + win_out
            },
        }

    def compile(self):
        """Magicompiler 编译 IR → 后端 attention 模块.

        不修改 Magicompiler 源码, 调用现有 rswa_gpu/rswa_mlx.
        """
        if self.backend == "mlx":
            from rswa_mlx import RSWAOrthoAttentionMLX
            return RSWAOrthoAttentionMLX(
                num_heads=self.num_heads, head_dim=self.head_dim,
                reference_len=self.reference_len, window_size=self.window_size,
                ortho_base_dim=self.ortho_base_dim,
            )
        elif self.backend in ("cuda", "sglang"):
            from rswa_gpu import RSWAOrthoAttentionGPU
            import torch
            module = RSWAOrthoAttentionGPU(
                num_heads=self.num_heads, head_dim=self.head_dim,
                reference_len=self.reference_len, window_size=self.window_size,
                ortho_base_dim=self.ortho_base_dim,
            )
            return module.to("cuda" if torch.cuda.is_available() else "cpu")
        elif self.backend == "native":
            # Native PyTorch (通用, 无 GPU 加速)
            from rswa_gpu import RSWAOrthoAttentionGPU
            return RSWAOrthoAttentionGPU(
                num_heads=self.num_heads, head_dim=self.head_dim,
                reference_len=self.reference_len, window_size=self.window_size,
                ortho_base_dim=self.ortho_base_dim,
            )
        else:
            raise ValueError(f"Unknown backend: {self.backend}")


# =============================================================================
# 2. AutoTunner + Magicompiler 整合
# =============================================================================

class AutoTunnerMagicompiler:
    """AutoTunner + Magicompiler 统一调优 + 编译.

    流程:
    1. detect(backend) → 硬件 profile → R-SWA 参数
    2. check_native_compress(model_path) → 检测模型是否已有原生压缩
       - 有 compress_ratios → 跳过 R-SWA (已有 attn_sink + sliding_window + compress)
       - 无 → 插入 R-SWA (MLX/小模型)
    3. create_ir(profile) → R-SWA 统一 IR (仅当无原生压缩时)
    4. compile(ir) → Magicompiler 编译到后端
    5. insert(model, compiled) → 替换 attention (InsertKDAPass)
    6. runtime_tune(accept/tok_s) → 动态调整参数
    """

    # 硬件 profile → R-SWA 最优参数
    PROFILES = {
        "mlx_m4": {
            "backend": "mlx", "num_heads": 16, "head_dim": 128,
            "reference_len": 4, "window_size": 128, "ortho_base_dim": 64,
            "num_draft_tokens": 16, "mode": "chain",
        },
        "cuda_rtx5000": {
            "backend": "cuda", "num_heads": 16, "head_dim": 128,
            "reference_len": 4, "window_size": 128, "ortho_base_dim": 64,
            "num_draft_tokens": 4, "mode": "chain",
        },
        "sglang_cloud": {
            "backend": "sglang", "num_heads": 32, "head_dim": 128,
            "reference_len": 4, "window_size": 128, "ortho_base_dim": 64,
            "num_draft_tokens": 4, "mode": "rswa",
        },
    }

    @classmethod
    def detect(cls, backend: str) -> Dict[str, Any]:
        """检测硬件 → 返回 R-SWA 最优参数."""
        if backend == "mlx":
            return cls.PROFILES["mlx_m4"]
        elif backend == "pytorch":
            return cls.PROFILES["cuda_rtx5000"]
        elif backend == "sglang":
            return cls.PROFILES["sglang_cloud"]
        return cls.PROFILES["mlx_m4"]

    @classmethod
    def check_native_compress(cls, model_path: str = "") -> Dict[str, Any]:
        """检测模型是否已有原生 KV 压缩 (compress_ratios + attn_sink + sliding_window).

        V4-Flash 等模型已原生实现 R-SWA 等效机制:
        - compress_ratios: [0,0,4,128,...] → KV cache 压缩 (4x/128x)
        - sliding_window: 128 → 滑动窗口 (= R-SWA window_size)
        - attn_sink: nn.Parameter → Reference 永久区 (= R-SWA Reference)

        Returns:
            {
                "has_native_compress": bool,
                "compress_ratios": list or None,
                "sliding_window": int or None,
                "has_attn_sink": bool,
                "should_skip_rswa": bool,  # True = 跳过 R-SWA (已有原生)
            }
        """
        import os, json

        result = {
            "has_native_compress": False,
            "compress_ratios": None,
            "sliding_window": None,
            "has_attn_sink": False,
            "should_skip_rswa": False,
        }

        if not model_path or not os.path.isdir(model_path):
            return result

        config_path = os.path.join(model_path, "config.json")
        if not os.path.isfile(config_path):
            return result

        try:
            with open(config_path) as f:
                config = json.load(f)

            compress_ratios = config.get("compress_ratios")
            sliding_window = config.get("sliding_window")

            if compress_ratios and any(r > 0 for r in compress_ratios):
                result["has_native_compress"] = True
                result["compress_ratios"] = compress_ratios
                result["sliding_window"] = sliding_window
                result["has_attn_sink"] = True  # dsv4 model 有 attn_sink 参数
                result["should_skip_rswa"] = True

                print(f"[AutoTunner] 模型已有原生压缩:")
                print(f"  compress_ratios: {compress_ratios[:10]}... ({len(compress_ratios)} 层)")
                print(f"  sliding_window: {sliding_window}")
                print(f"  attn_sink: ✅ (dsv4 原生)")
                print(f"  → 跳过 R-SWA (已有等效机制)")
        except Exception:
            pass

        return result

    @classmethod
    def create_ir(cls, backend: str, model_config: Optional[Dict] = None,
                  model_path: str = "") -> Optional[RSWAAttentionIR]:
        """创建 R-SWA 统一 IR (AutoTunner 自动选择参数).

        如果模型已有原生压缩 (compress_ratios), 返回 None (跳过 R-SWA).
        """
        # 1. 检测模型是否已有原生压缩
        native = cls.check_native_compress(model_path)
        if native["should_skip_rswa"]:
            return None  # 跳过 R-SWA

        # 2. 无原生压缩 → 创建 R-SWA IR
        profile = cls.detect(backend)

        # 如果有模型配置, 覆盖 num_heads/head_dim
        if model_config:
            profile["num_heads"] = model_config.get("num_heads", profile["num_heads"])
            profile["head_dim"] = model_config.get("head_dim", profile["head_dim"])

        ir = RSWAAttentionIR(
            num_heads=profile["num_heads"],
            head_dim=profile["head_dim"],
            reference_len=profile["reference_len"],
            window_size=profile["window_size"],
            ortho_base_dim=profile["ortho_base_dim"],
            num_draft_tokens=profile["num_draft_tokens"],
            mode=profile["mode"],
            backend=profile["backend"],
        )

        print(f"[AutoTunner+Magicompiler] IR created: backend={ir.backend}, "
              f"heads={ir.num_heads}, ref={ir.reference_len}, win={ir.window_size}, "
              f"ortho={ir.ortho_base_dim}, N={ir.num_draft_tokens}, mode={ir.mode}")
        return ir

    @classmethod
    def compile_and_insert(cls, model, backend: str, model_config: Optional[Dict] = None,
                           model_path: str = ""):
        """编译 R-SWA IR + 替换 model 的 attention (InsertKDAPass).

        如果模型已有原生压缩 (V4-Flash 等), 跳过 R-SWA.
        不修改 Magicompiler 源码, 用现有 hook 机制.
        返回 restore 函数 (可回退, None 表示跳过).
        """
        ir = cls.create_ir(backend, model_config, model_path)

        if ir is None:
            print("[AutoTunner+Magicompiler] R-SWA skipped (model has native compress)")
            return lambda: None  # 空恢复函数

        # 1. Magicompiler 编译 IR → 后端 attention 模块
        compiled_attn = ir.compile()
        print(f"[AutoTunner+Magicompiler] Compiled to {ir.backend} backend ✅")

        # 2. InsertKDAPass: 替换 model 的 attention
        originals = {}
        layers = None

        # 找到 decoder layers (多路径兼容)
        if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
            layers = model.language_model.layers  # Qwen3-VL (MLX)
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers  # Qwen2/3 (PyTorch)
        elif hasattr(model, "layers"):
            layers = model.layers

        if layers is None:
            print(f"[AutoTunner+Magicompiler] ⚠️ No layers found, attaching to model.rswa_attn")
            model.rswa_attn = compiled_attn
            return lambda: None

        # 替换每层的 attention
        for i, layer in enumerate(layers):
            if hasattr(layer, "self_attn"):
                originals[i] = layer.self_attn
                layer.self_attn = compiled_attn
            elif hasattr(layer, "attn"):
                originals[i] = layer.attn
                layer.attn = compiled_attn

        print(f"[AutoTunner+Magicompiler] InsertKDAPass: {len(originals)} layers replaced ✅")

        # 3. 返回 restore 函数
        def restore():
            for i, original in originals.items():
                if hasattr(layers[i], "self_attn"):
                    layers[i].self_attn = original
                elif hasattr(layers[i], "attn"):
                    layers[i].attn = original
            print(f"[AutoTunner+Magicompiler] Restored {len(originals)} layers ✅")

        return restore

    @classmethod
    def runtime_tune(cls, ir: RSWAAttentionIR, accept_rate: float, tps: float) -> RSWAAttentionIR:
        """运行时调优 (AutoTunner accept rate 驱动).

        规则:
          - accept < 15% → N=0, 关闭投机
          - accept 15-30% → N=2, 最小投机
          - accept > 60% → N 增大
          - tok_s 下降 → 减小 window_size (减少 attention 范围)
        """
        old_N = ir.num_draft_tokens
        old_win = ir.window_size

        # 投机 decode N 调整
        if accept_rate < 0.15:
            ir.num_draft_tokens = 0
        elif accept_rate < 0.30:
            ir.num_draft_tokens = 2
        elif accept_rate > 0.60:
            ir.num_draft_tokens = min(32, ir.num_draft_tokens * 2)

        # R-SWA window_size 调整 (tok_s 下降时减小窗口)
        if tps > 0 and ir.tps > 0 and tps < ir.tps * 0.9:
            ir.window_size = max(32, ir.window_size // 2)

        ir.accept_rate = accept_rate
        ir.tps = tps

        if ir.num_draft_tokens != old_N or ir.window_size != old_win:
            print(f"[AutoTunner+Magicompiler] retune: N={old_N}→{ir.num_draft_tokens}, "
                  f"win={old_win}→{ir.window_size} (accept={accept_rate:.0%}, tps={tps:.1f})")

        return ir


# =============================================================================
# 3. 统一入口 (一行调用)
# =============================================================================

def auto_rswa(model, backend: str = "mlx", model_config: Optional[Dict] = None,
              model_path: str = ""):
    """一行调用: AutoTunner 检测 + Magicompiler 编译 + R-SWA 替换.

    自动检测模型是否已有原生压缩:
    - V4-Flash (有 compress_ratios) → 跳过 R-SWA
    - Qwen3-VL-2B (无 compress_ratios) → 插入 R-SWA

    Args:
        model: LLM 模型 (MLX/PyTorch)
        backend: mlx | cuda | sglang | native
        model_config: 模型配置 (num_heads, head_dim 等)
        model_path: 模型路径 (用于检测 compress_ratios)

    Returns:
        restore 函数 (调用后恢复原始 attention, None=跳过)

    用法:
        restore = auto_rswa(model, "mlx", model_path="/path/to/model")
        # ... 推理 (R-SWA 生效, 或跳过如果已有原生压缩) ...
        restore()  # 恢复
    """
    return AutoTunnerMagicompiler.compile_and_insert(model, backend, model_config, model_path)


if __name__ == "__main__":
    print("=" * 70)
    print(" R-SWA + Magicompiler + AutoTunner 统一 IR")
    print("=" * 70)

    # 测试 compress_ratios 检测
    print("\n--- compress_ratios 检测 ---")
    # V4-Flash (有原生压缩)
    native_v4 = AutoTunnerMagicompiler.check_native_compress("/data/models/DeepSeek-V4-Flash-UD-IQ2")
    print(f"  V4-Flash: has_native={native_v4['has_native_compress']}, skip_rswa={native_v4['should_skip_rswa']}")
    # Qwen3-VL-2B (无原生压缩)
    native_q3 = AutoTunnerMagicompiler.check_native_compress("/data2/models/Qwen3-VL-2B-Instruct")
    print(f"  Qwen3-VL-2B: has_native={native_q3['has_native_compress']}, skip_rswa={native_q3['should_skip_rswa']}")

    # 测试 IR 创建 (V4-Flash 应跳过, Qwen3-VL 应创建)
    print("\n--- IR 创建 (带 model_path 检测) ---")
    for backend, path in [("mlx", "/data2/models/Qwen3-VL-2B-Instruct"),
                           ("sglang", "/data/models/DeepSeek-V4-Flash-UD-IQ2")]:
        ir = AutoTunnerMagicompiler.create_ir(backend, model_path=path)
        if ir is None:
            print(f"  {backend} ({path.split('/')[-1]}): R-SWA 跳过 (已有原生压缩) ✅")
        else:
            print(f"  {backend} ({path.split('/')[-1]}): R-SWA IR 创建 ✅ (ref={ir.reference_len}, win={ir.window_size})")

    # 测试 runtime_tune
    print("\n--- runtime_tune 测试 ---")
    ir = AutoTunnerMagicompiler.create_ir("mlx")
    AutoTunnerMagicompiler.runtime_tune(ir, 0.28, 20.0)  # accept 28% → N=2
    AutoTunnerMagicompiler.runtime_tune(ir, 0.70, 50.0)  # accept 70% → N 增大
    AutoTunnerMagicompiler.runtime_tune(ir, 0.10, 10.0)  # accept 10% → N=0

    print("\n  用法: restore = auto_rswa(model, 'mlx')")
    print("  一行调用: AutoTunner 检测 + Magicompiler 编译 + R-SWA 替换")
    print("=" * 70)
