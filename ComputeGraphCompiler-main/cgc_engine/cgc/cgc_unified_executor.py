
# 统一执行层 (Unified Execution Layer)
# 让用户不用关心 vLLM/llama.cpp 差别！

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
import torch

from .cgc_opcodes import CGC_OP_CODES


# ============================================================
# 统一 Command 枚举 (UnifiedCommand
# ============================================================


class UnifiedCommandType:
    """统一的 Command 类型（用户不用关心 vLLM/llama.cpp）"""
    # ==================================================
    # 通用计算
    # ==================================================
    LINEAR_GEMM = "UNIFIED_LINEAR_GEMM"
    LINEAR_BIAS = "UNIFIED_LINEAR_BIAS"
    RMS_NORM = "UNIFIED_RMS_NORM"
    ROPE = "UNIFIED_ROPE"
    SILU = "UNIFIED_SILU"
    GELU = "UNIFIED_GELU"
    SOFTMAX = "UNIFIED_SOFTMAX"
    ATTENTION_SDPA = "UNIFIED_ATTENTION_SDPA"
    ATTENTION_FLASH = "UNIFIED_ATTENTION_FLASH"
    ATTENTION_KDA = "UNIFIED_ATTENTION_KDA"

    # ==================================================
    # 量化/反量化
    # ==================================================
    QUANTIZE_W8A16 = "UNIFIED_QUANTIZE_W8A16"
    QUANTIZE_W4A16 = "UNIFIED_QUANTIZE_W4A16"
    DEQUANTIZE = "UNIFIED_DEQUANTIZE"
    GPTQ_KERNEL = "UNIFIED_GPTQ_KERNEL"
    AWQ_KERNEL = "UNIFIED_AWQ_KERNEL"

    # ==================================================
    # llama.cpp 专用
    # ==================================================
    GGUF_LOAD = "UNIFIED_GGUF_LOAD"
    GGUF_QUANTIZE = "UNIFIED_GGUF_QUANTIZE"
    GGUF_DEQUANTIZE = "UNIFIED_GGUF_DEQUANTIZE"
    MOE_ROUTING = "UNIFIED_MOE_ROUTING"
    MOE_EXPERT_FWD = "UNIFIED_MOE_EXPERT_FWD"
    LLAMA_INFERENCE = "UNIFIED_LLAMA_INFERENCE"


UnifiedCommand = UnifiedCommandType


# ============================================================
# 后端选择策略
# ============================================================


class BackendStrategy:
    """后端选择策略"""
    @staticmethod
    def select_backend(unified_cmd: str, hints: Dict[str, Any]) -> str:
        """选择使用哪个后端"""
        device_type = hints.get("device", "cuda")
        memory_available = hints.get("memory_available_gb", 16.0)

        if device_type == "mps" or device_type == "apple":
            if memory_available < 8.0:
                return "llama_cpp"
            else:
                return "llama_cpp"
        elif device_type == "cuda":
            if memory_available < 8.0:
                return "llama_cpp"
            else:
                return "vllm"
        else:
            return "llama_cpp"


# ============================================================
# 统一 Command 转具体 Opcode 映射
# ============================================================


class UnifiedCommandToOpcode:
    """统一 Command -> 具体 Opcode 映射"""

    @staticmethod
    def vllm(unified_cmd: str) -> int:
        """vLLM 域的 opcode"""
        _map = {
            # 通用计算
            UnifiedCommand.LINEAR_GEMM: CGC_OP_CODES.LINEAR_GEMM,
            UnifiedCommand.LINEAR_BIAS: CGC_OP_CODES.LINEAR_BIAS,
            UnifiedCommand.RMS_NORM: CGC_OP_CODES.RMS_NORM,
            UnifiedCommand.ROPE: CGC_OP_CODES.ROPE,
            UnifiedCommand.SILU: CGC_OP_CODES.SILU,
            UnifiedCommand.GELU: CGC_OP_CODES.GELU,
            UnifiedCommand.SOFTMAX: CGC_OP_CODES.SOFTMAX,
            UnifiedCommand.ATTENTION_SDPA: CGC_OP_CODES.ATTENTION_SDPA,
            UnifiedCommand.ATTENTION_FLASH: CGC_OP_CODES.ATTENTION_FLASH,
            UnifiedCommand.ATTENTION_KDA: CGC_OP_CODES.ATTENTION_KDA,
            
            # 量化/反量化
            UnifiedCommand.QUANTIZE_W8A16: CGC_OP_CODES.QUANTIZE_W8A16,
            UnifiedCommand.QUANTIZE_W4A16: CGC_OP_CODES.QUANTIZE_W4A16,
            UnifiedCommand.DEQUANTIZE: CGC_OP_CODES.DEQUANTIZE,
            UnifiedCommand.GPTQ_KERNEL: CGC_OP_CODES.GPTQ_KERNEL,
            UnifiedCommand.AWQ_KERNEL: CGC_OP_CODES.AWQ_KERNEL,
        }
        return _map.get(unified_cmd)

    @staticmethod
    def llama_cpp(unified_cmd: str, quant_type: str = "q4_k") -> int:
        """llama.cpp 域的 opcode"""
        _quant_map = {
            "q2_k": CGC_OP_CODES.LLAMA_Q2_K_MATMUL,
            "q3_k": CGC_OP_CODES.LLAMA_Q3_K_MATMUL,
            "q4_k": CGC_OP_CODES.LLAMA_Q4_K_MATMUL,
            "q5_k": CGC_OP_CODES.LLAMA_Q5_K_MATMUL,
            "q6_k": CGC_OP_CODES.LLAMA_Q6_K_MATMUL,
            "q8_0": CGC_OP_CODES.LLAMA_Q8_0_MATMUL,
            "q8_k": CGC_OP_CODES.LLAMA_Q8_K_MATMUL,
        }
        _map = {
            # 通用计算
            UnifiedCommand.LINEAR_GEMM: _quant_map.get(quant_type, CGC_OP_CODES.LLAMA_Q4_K_MATMUL),
            UnifiedCommand.LINEAR_BIAS: CGC_OP_CODES.LLAMA_Q4_K_MATMUL,
            UnifiedCommand.RMS_NORM: CGC_OP_CODES.LLAMA_RMSNORM_GGUF,
            UnifiedCommand.ROPE: CGC_OP_CODES.LLAMA_ROPE_GGUF,
            UnifiedCommand.SILU: CGC_OP_CODES.LLAMA_SILU_GGUF,
            UnifiedCommand.GELU: CGC_OP_CODES.LLAMA_GELU_GGUF,
            UnifiedCommand.SOFTMAX: CGC_OP_CODES.LLAMA_SAMPLING_GGUF,
            
            # 量化/反量化 (llama.cpp 专属)
            UnifiedCommand.GGUF_LOAD: CGC_OP_CODES.LLAMA_GGUF_LOAD,
            UnifiedCommand.GGUF_QUANTIZE: CGC_OP_CODES.LLAMA_GGUF_QUANTIZE,
            UnifiedCommand.GGUF_DEQUANTIZE: CGC_OP_CODES.LLAMA_GGUF_DEQUANTIZE,
            
            # MoE
            UnifiedCommand.MOE_ROUTING: CGC_OP_CODES.LLAMA_MOE_ROUTING,
            UnifiedCommand.MOE_EXPERT_FWD: CGC_OP_CODES.LLAMA_MOE_EXPERT_FWD,
            
            # 通用
            UnifiedCommand.LLAMA_INFERENCE: CGC_OP_CODES.LLAMA_INFERENCE,
        }
        return _map.get(unified_cmd)


# ============================================================
# 统一执行接口
# ============================================================


def execute_unified(executor, unified_cmd: str,
                inputs: List[torch.Tensor],
                params: Optional[Dict[str, Any]] = None,
                hints: Optional[Dict[str, Any]] = None) -> List[torch.Tensor]:
    """执行统一 Command（自动选择后端）"""
    from .cgc_simd_executor import CGCCommand

    params = params or {}
    hints = hints or {}

    # 默认使用 llama.cpp（覆盖范围广）
    backend = hints.get("backend", "auto")
    
    if backend == "auto":
        backend = BackendStrategy.select_backend(unified_cmd, hints)
    
    quant_type = hints.get("quant_type", "q4_k")

    # 获取 opcode
    if backend == "vllm":
        opcode = UnifiedCommandToOpcode.vllm(unified_cmd)
    else:
        opcode = UnifiedCommandToOpcode.llama_cpp(unified_cmd, quant_type)

    if opcode is None:
        raise ValueError(f"Unified Command {unified_cmd} not supported by backend {backend}")

    command = CGCCommand(
        opcode=opcode,
        inputs=inputs,
        outputs=[],
        params=params
    )

    return executor.execute(command)
