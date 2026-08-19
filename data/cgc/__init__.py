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

import os as _os
import platform as _platform

_CGC_LIGHT_IMPORT = (_platform.system() != "Linux") and (_os.environ.get("CGC_ENGINE_FORCE_FULL_IMPORT", "0") != "1")

from .cgc_commands import (
    CGCInstruction,
    CGCInstructionType,
    KDAInstruction,
    CGC_SIMD_COMMAND_SET,
    get_cgc_command,
    get_cgc_command_by_opcode,
    get_all_cgc_commands,
    get_commands_by_category,
    get_commands_by_module,
    create_kda_instruction,
    get_command_summary,
    WEIGHT_STAY_CMD,
    LAYER_STREAM_LOAD_CMD,
    LAYER_FORWARD_CMD,
    ORTHO_BASIS_UPDATE_CMD,
    KDA_CHUNK_CMD,
    KDA_PROJECT_CMD,
    KDA_ORTHO_UPDATE_CMD,
    ATTENTION_SDPA_CMD,
    ATTENTION_KDA_CMD,
    ATTENTION_PAGED_CMD,
    ATTENTION_FLASH_CMD,
    LINEAR_GEMM_CMD,
    LINEAR_BIAS_CMD,
    GEMM_BATCHED_CMD,
    LAYER_NORM_CMD,
    RMS_NORM_CMD,
    GROUP_NORM_CMD,
    ROPE_CMD,
    ROPE_FUSED_CMD,
    YARN_ROPE_CMD,
    SILU_CMD,
    GELU_CMD,
    GELU_TANH_CMD,
    RELU_CMD,
    SIGMOID_CMD,
    SOFTMAX_CMD,
    LOG_SOFTMAX_CMD,
    TOP_K_CMD,
    TOP_P_CMD,
    TEMPERATURE_CMD,
    KV_CACHE_LOAD_CMD,
    KV_CACHE_STORE_CMD,
    KV_CACHE_UPDATE_CMD,
    EMBEDDING_LOOKUP_CMD,
    ALL_REDUCE_CMD,
    ALL_GATHER_CMD,
    REDUCE_SCATTER_CMD,
    QUANTIZE_W8A16_CMD,
    QUANTIZE_W4A16_CMD,
    DEQUANTIZE_CMD,
    GPTQ_KERNEL_CMD,
    AWQ_KERNEL_CMD,
)

from .cgc_opcodes import (
    CGC_OP_CODES,
    get_opcode_name,
    get_opcode_value,
    list_attention_opcodes,
    list_kda_opcodes,
    is_attention_opcode,
    is_kda_opcode,
    get_category,
)

try:
    from .kda_pass import InsertKDAPass, CGCKDAVisitor
except Exception:
    InsertKDAPass = None
    CGCKDAVisitor = None

try:
    if _CGC_LIGHT_IMPORT:
        raise ImportError("light import")
    from .flashkda_integration import (
        FlashKDALayer,
        create_flashkda_attention,
        CGCKDAKernelRegistry,
        register_cgc_kda_ops,
        get_cgc_kda_metadata,
        FLASHKDA_AVAILABLE,
    )
except Exception:
    FlashKDALayer = None
    create_flashkda_attention = None
    CGCKDAKernelRegistry = None
    register_cgc_kda_ops = None
    get_cgc_kda_metadata = None
    FLASHKDA_AVAILABLE = False

try:
    if _CGC_LIGHT_IMPORT:
        raise ImportError("light import")
    from .vllm_integration import (
        VLLMKDAConfig,
        VLLMKDABackend,
        MagiCompilerVLLMWrapper,
        create_vllm_kda_backend,
        get_vllm_kda_attention_layer,
        VLLM_AVAILABLE,
        INTEGRATION_GUIDE,
    )
except Exception:
    VLLMKDAConfig = None
    VLLMKDABackend = None
    MagiCompilerVLLMWrapper = None
    create_vllm_kda_backend = None
    get_vllm_kda_attention_layer = None
    VLLM_AVAILABLE = False
    INTEGRATION_GUIDE = ""

if _CGC_LIGHT_IMPORT:
    CGCExecutor = None
    CGCKernelRegistry = None
    CGCCommand = None
    KernelType = None
    register_cuda_kernel = None
    execute_cgc_command = None
    get_kernel_registry = None
    list_available_kernels = None
else:
    from .cgc_simd_executor import (
        CGCExecutor,
        CGCKernelRegistry,
        CGCCommand,
        KernelType,
        register_cuda_kernel,
        execute_cgc_command,
        get_kernel_registry,
        list_available_kernels,
    )

try:
    from .cgc_unified_executor import (
        UnifiedCommandType,
        execute_unified,
    )
except Exception:
    UnifiedCommandType = None
    execute_unified = None

try:
    from .cgc_backend import (
        CGCBackend,
        CGCConfig,
    )
except Exception:
    CGCBackend = None
    CGCConfig = None

try:
    if _CGC_LIGHT_IMPORT:
        raise ImportError("light import")
    from .vllm_cgc_backend import (
        VLLMCGCBackend,
        VLLMCGCConfig,
        VLLMModelCGC,
        VLLMAttentionCGC,
        VLLMMLPCGC,
        VLLMRMSNormCGC,
        VLLMRoPECGC,
        create_vllm_cgc_model,
        VLLMOpCode,
    )
except Exception:
    VLLMCGCBackend = None
    VLLMCGCConfig = None
    VLLMModelCGC = None
    VLLMAttentionCGC = None
    VLLMMLPCGC = None
    VLLMRMSNormCGC = None
    VLLMRoPECGC = None
    create_vllm_cgc_model = None
    VLLMOpCode = None

try:
    if _CGC_LIGHT_IMPORT:
        raise ImportError("light import")
    from .vllm_kda_ops import (
        flash_kda_cgc_forward,
        flash_kda_cgc_forward_native,
        sdpa_cgc_forward,
        rms_norm_cgc_forward,
        rope_cgc_forward,
        silu_cgc_forward,
        softmax_cgc_forward,
        get_cgc_backend_info,
        CGC_AVAILABLE,
    )
except Exception:
    flash_kda_cgc_forward = None
    flash_kda_cgc_forward_native = None
    sdpa_cgc_forward = None
    rms_norm_cgc_forward = None
    rope_cgc_forward = None
    silu_cgc_forward = None
    softmax_cgc_forward = None
    get_cgc_backend_info = None
    CGC_AVAILABLE = False

try:
    if _CGC_LIGHT_IMPORT:
        raise ImportError("light import")
    from .vllm_kda_attention import (
        CGCKDABackend,
        CGCKDABackendConfig,
        create_cgc_kda_backend,
    )
except Exception:
    CGCKDABackend = None
    CGCKDABackendConfig = None
    create_cgc_kda_backend = None

try:
    if _CGC_LIGHT_IMPORT:
        raise ImportError("light import")
    from .vllm_full_cgc import get_full_stack_info
except Exception:
    get_full_stack_info = None

try:
    if _CGC_LIGHT_IMPORT:
        raise ImportError("light import")
    from .megatrain_integration import (
        MegatrainCGCAttention,
        MegatrainCGCLayerNorm,
        MegatrainCGCRMSNorm,
        MegatrainCGCRoPE,
        MegatrainCGCMLP,
        MegatrainCGCLinear,
        MegatrainCGCEmbedding,
        MegatrainCGCSoftmax,
        MegatrainCGCKDAOrthoUpdate,
        MegatrainCGCTransformerLayer,
        MegatrainCGCModel,
        MegatrainCGCEngine,
        replace_megatrain_attention,
        create_megatrain_cgc_model,
        get_megatrain_cgc_info,
    )
except Exception:
    MegatrainCGCAttention = None
    MegatrainCGCLayerNorm = None
    MegatrainCGCRMSNorm = None
    MegatrainCGCRoPE = None
    MegatrainCGCMLP = None
    MegatrainCGCLinear = None
    MegatrainCGCEmbedding = None
    MegatrainCGCSoftmax = None
    MegatrainCGCKDAOrthoUpdate = None
    MegatrainCGCTransformerLayer = None
    MegatrainCGCModel = None
    MegatrainCGCEngine = None
    replace_megatrain_attention = None
    create_megatrain_cgc_model = None
    get_megatrain_cgc_info = None

try:
    from .cgc_dashboard import (
        CGCDashboardServer,
        CGCTracer,
        CGCStats,
        CGCTraceEvent,
        OpcodeCategory,
        start_dashboard,
        get_dashboard_info,
    )
except Exception:
    CGCDashboardServer = None
    CGCTracer = None
    CGCStats = None
    CGCTraceEvent = None
    OpcodeCategory = None
    start_dashboard = None
    get_dashboard_info = None

try:
    from .benchmark import (
        CGCBenchmark,
        BenchmarkConfig,
        BenchmarkResult,
        get_memory_usage,
        get_peak_memory,
    )
except Exception:
    CGCBenchmark = None
    BenchmarkConfig = None
    BenchmarkResult = None
    get_memory_usage = None
    get_peak_memory = None

try:
    from .benchmark_production import (
        ProductionBenchmark,
        run_production_benchmark,
        run_prefill_decode_benchmark,
        ThroughputMetrics,
        LatencyMetrics,
        MemoryMetrics,
        ArchitectureMetrics,
        CorrectnessMetrics,
        StabilityMetrics,
        BenchmarkMode,
        BackendType,
        CGCMemoryProfiler,
        LatencyTracker,
    )
except Exception:
    ProductionBenchmark = None
    run_production_benchmark = None
    run_prefill_decode_benchmark = None
    ThroughputMetrics = None
    LatencyMetrics = None
    MemoryMetrics = None
    ArchitectureMetrics = None
    CorrectnessMetrics = None
    StabilityMetrics = None
    BenchmarkMode = None
    BackendType = None
    CGCMemoryProfiler = None
    LatencyTracker = None

try:
    from .mlx_tune_integration import (
        CGCMLXTuneBackend as CGCMlxTune,
        LoRAManager as LoRALayer,
        LISAFineTuner,
        LoRAConfig,
        QLoRAConfig,
        FineTuningMode,
        is_finetuning_opcode,
        is_lora_opcode,
        get_mlx_tune_info,
    )
except Exception:
    CGCMlxTune = None
    LoRALayer = None
    LISAFineTuner = None
    LoRAConfig = None
    QLoRAConfig = None
    FineTuningMode = None
    is_finetuning_opcode = None
    is_lora_opcode = None
    get_mlx_tune_info = None

__all__ = [
    "CGCInstruction",
    "CGCInstructionType",
    "KDAInstruction",
    "CGC_SIMD_COMMAND_SET",
    "get_cgc_command",
    "get_cgc_command_by_opcode",
    "get_all_cgc_commands",
    "get_commands_by_category",
    "get_commands_by_module",
    "create_kda_instruction",
    "get_command_summary",
    "CGC_OP_CODES",
    "get_opcode_name",
    "get_opcode_value",
    "list_attention_opcodes",
    "list_kda_opcodes",
    "is_attention_opcode",
    "is_kda_opcode",
    "get_category",
    "WEIGHT_STAY_CMD",
    "LAYER_STREAM_LOAD_CMD",
    "LAYER_FORWARD_CMD",
    "ORTHO_BASIS_UPDATE_CMD",
    "KDA_CHUNK_CMD",
    "KDA_PROJECT_CMD",
    "KDA_ORTHO_UPDATE_CMD",
    "ATTENTION_SDPA_CMD",
    "ATTENTION_KDA_CMD",
    "ATTENTION_PAGED_CMD",
    "ATTENTION_FLASH_CMD",
    "LINEAR_GEMM_CMD",
    "LINEAR_BIAS_CMD",
    "GEMM_BATCHED_CMD",
    "LAYER_NORM_CMD",
    "RMS_NORM_CMD",
    "GROUP_NORM_CMD",
    "ROPE_CMD",
    "ROPE_FUSED_CMD",
    "YARN_ROPE_CMD",
    "SILU_CMD",
    "GELU_CMD",
    "GELU_TANH_CMD",
    "RELU_CMD",
    "SIGMOID_CMD",
    "SOFTMAX_CMD",
    "LOG_SOFTMAX_CMD",
    "TOP_K_CMD",
    "TOP_P_CMD",
    "TEMPERATURE_CMD",
    "KV_CACHE_LOAD_CMD",
    "KV_CACHE_STORE_CMD",
    "KV_CACHE_UPDATE_CMD",
    "EMBEDDING_LOOKUP_CMD",
    "ALL_REDUCE_CMD",
    "ALL_GATHER_CMD",
    "REDUCE_SCATTER_CMD",
    "QUANTIZE_W8A16_CMD",
    "QUANTIZE_W4A16_CMD",
    "DEQUANTIZE_CMD",
    "GPTQ_KERNEL_CMD",
    "AWQ_KERNEL_CMD",
    "InsertKDAPass",
    "CGCKDAVisitor",
    "FlashKDALayer",
    "create_flashkda_attention",
    "CGCKDAKernelRegistry",
    "register_cgc_kda_ops",
    "get_cgc_kda_metadata",
    "FLASHKDA_AVAILABLE",
    "VLLMKDAConfig",
    "VLLMKDABackend",
    "MagiCompilerVLLMWrapper",
    "create_vllm_kda_backend",
    "get_vllm_kda_attention_layer",
    "VLLM_AVAILABLE",
    "INTEGRATION_GUIDE",
    "CGCExecutor",
    "CGCKernelRegistry",
    "CGCCommand",
    "KernelType",
    "register_cuda_kernel",
    "execute_cgc_command",
    "get_kernel_registry",
    "list_available_kernels",
    "UnifiedCommandType",
    "execute_unified",
    "CGCBackend",
    "CGCConfig",
    "VLLMCGCBackend",
    "VLLMCGCConfig",
    "VLLMModelCGC",
    "VLLMAttentionCGC",
    "VLLMMLPCGC",
    "VLLMRMSNormCGC",
    "VLLMRoPECGC",
    "create_vllm_cgc_model",
    "VLLMOpCode",
    "flash_kda_cgc_forward",
    "flash_kda_cgc_forward_native",
    "sdpa_cgc_forward",
    "rms_norm_cgc_forward",
    "rope_cgc_forward",
    "silu_cgc_forward",
    "softmax_cgc_forward",
    "get_cgc_backend_info",
    "CGCKDABackend",
    "CGCKDABackendConfig",
    "create_cgc_kda_backend",
    "get_full_stack_info",
    "MegatrainCGCAttention",
    "MegatrainCGCLayerNorm",
    "MegatrainCGCRMSNorm",
    "MegatrainCGCRoPE",
    "MegatrainCGCMLP",
    "MegatrainCGCLinear",
    "MegatrainCGCEmbedding",
    "MegatrainCGCSoftmax",
    "MegatrainCGCKDAOrthoUpdate",
    "MegatrainCGCTransformerLayer",
    "MegatrainCGCModel",
    "MegatrainCGCEngine",
    "replace_megatrain_attention",
    "create_megatrain_cgc_model",
    "get_megatrain_cgc_info",
    "CGCDashboardServer",
    "CGCTracer",
    "CGCStats",
    "CGCTraceEvent",
    "OpcodeCategory",
    "start_dashboard",
    "get_dashboard_info",
    "CGCBenchmark",
    "BenchmarkConfig",
    "BenchmarkResult",
    "get_memory_usage",
    "get_peak_memory",
    "CGCMlxTune",
    "LoRALayer",
    "LISAFineTuner",
    "LoRAConfig",
    "QLoRAConfig",
    "FineTuningMode",
    "is_finetuning_opcode",
    "is_lora_opcode",
    "get_mlx_tune_info",
    "ProductionBenchmark",
    "run_production_benchmark",
    "run_prefill_decode_benchmark",
    "ThroughputMetrics",
    "LatencyMetrics",
    "MemoryMetrics",
    "ArchitectureMetrics",
    "CorrectnessMetrics",
    "StabilityMetrics",
    "BenchmarkMode",
    "BackendType",
    "CGCMemoryProfiler",
    "LatencyTracker",
]
