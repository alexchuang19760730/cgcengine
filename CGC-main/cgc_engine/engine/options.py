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
CGC Engine 編譯器配置選項

CGCEngineOptions 定義了 CGC Engine 的所有配置選項，
連接所有功能模組，提供統一的配置接口。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Literal
from enum import Enum


class QuantizationMode(Enum):
    FP16 = "fp16"
    BF16 = "bf16"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    INT8 = "int8"
    INT4 = "int4"
    AUTO = "auto"


class AttentionBackend(Enum):
    FLASH_ATTENTION = "flash"
    KDA = "kda"
    PAGED = "paged"
    SDPA = "sdpa"
    AUTO = "auto"


class MemoryBackend(Enum):
    PD = "pd"
    GDS = "gds"
    SPDK = "spdk"
    METAL_MPS = "metal"
    PYTORCH = "pytorch"


class ExportFormat(Enum):
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    METAL = "metal"


class ScheduleStrategy(Enum):
    HEURISTIC = "heuristic"
    CONTINOUS_BATCHING = "continuous_batching"
    SPECULATIVE = "speculative"


@dataclass
class AttentionOptions:
    """注意力機制配置"""
    backend: AttentionBackend = AttentionBackend.AUTO
    enable_flash_kda: bool = True
    enable_kv_cache_static_layout: bool = True
    enable_paged_attention: bool = True
    page_size: int = 16
    kv_cache_quantize: QuantizationMode = QuantizationMode.INT8
    enable_kv_async_prefetch: bool = True


@dataclass
class QuantizationOptions:
    """量化配置"""
    mode: QuantizationMode = QuantizationMode.FP16
    weight_quantize_bits: int = 16
    activation_quantize_bits: int = 16
    group_size: int = 128
    enable_gptq_kernel: bool = False
    enable_awq_kernel: bool = False
    enable_fp8_kernels: bool = False


@dataclass
class MemoryOptions:
    """記憶體/存儲配置"""
    backend: MemoryBackend = MemoryBackend.PYTORCH
    enable_pd分离: bool = True
    pd_server_address: str = "localhost:5555"
    enable_gds: bool = False
    enable_spdk: bool = False
    metal_mps_enabled: bool = False
    offload_threshold_gb: float = 16.0
    max_offload_size_gb: float = 32.0
    enable_memory_tracker: bool = False


@dataclass
class MoEOptions:
    """MoE 配置"""
    enable_flashmoe: bool = False
    enable_omlx: bool = False
    expert_cache_size: int = 8
    top_k_experts: int = 2
    expert_prediction_threshold: float = 0.8


@dataclass
class DistributedOptions:
    """分布式訓練配置"""
    enable_nccl: bool = True
    enable_fsdp: bool = False
    world_size: int = 1
    rank: int = 0
    master_addr: str = "localhost"
    master_port: int = 29500


@dataclass
class PerformanceOptions:
    """性能分析配置"""
    enable_profiler: bool = False
    enable_memory_tracker: bool = False
    enable_tensor_timeline: bool = False
    enable_dynamic_precision: bool = False
    profile_output_dir: str = "./cgc_profiles"
    generate_heatmap: bool = True
    generate_chrome_trace: bool = True


@dataclass
class ExportOptions:
    """導出配置"""
    formats: List[ExportFormat] = field(default_factory=lambda: [ExportFormat.ONNX])
    target_device: str = "cuda"
    optimization_level: int = 3
    enable_cgc_commands: bool = True
    opset_version: int = 17


@dataclass
class CompilerOptions:
    """CGC Engine 編譯器主配置"""
    model_name: str = "cgc_model"
    hidden_dim: int = 4096
    num_layers: int = 32
    num_attention_heads: int = 32
    vocab_size: int = 32000

    attention: AttentionOptions = field(default_factory=AttentionOptions)
    quantization: QuantizationOptions = field(default_factory=QuantizationOptions)
    memory: MemoryOptions = field(default_factory=MemoryOptions)
    moe: MoEOptions = field(default_factory=MoEOptions)
    distributed: DistributedOptions = field(default_factory=DistributedOptions)
    performance: PerformanceOptions = field(default_factory=PerformanceOptions)
    export: ExportOptions = field(default_factory=ExportOptions)

    schedule_strategy: ScheduleStrategy = ScheduleStrategy.HEURISTIC
    enable_jit_compilation: bool = True
    enable_cuda_graph: bool = True
    enable_dynamic_batching: bool = True
    max_batch_size: int = 32

    device: str = "cuda"
    dtype: Literal["float32", "float16", "bfloat16"] = "float16"

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        result = {}
        for key, value in self.__dict__.items():
            if hasattr(value, '__dataclass_fields__'):
                result[key] = {k: v for k, v in value.__dict__.items()}
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "CompilerOptions":
        """從字典創建"""
        attention = AttentionOptions(**config.get("attention", {}))
        quantization = QuantizationOptions(**config.get("quantization", {}))
        memory = MemoryOptions(**config.get("memory", {}))
        moe = MoEOptions(**config.get("moe", {}))
        distributed = DistributedOptions(**config.get("distributed", {}))
        performance = PerformanceOptions(**config.get("performance", {}))
        export = ExportOptions(**config.get("export", {}))

        return cls(
            **{k: v for k, v in config.items()
               if k not in ("attention", "quantization", "memory", "moe", "distributed", "performance", "export")},
            attention=attention,
            quantization=quantization,
            memory=memory,
            moe=moe,
            distributed=distributed,
            performance=performance,
            export=export,
        )


class CGCEngineOptions:
    """
    CGC Engine 統一配置接口

    使用範例:
        from cgc_engine.engine import CGCEngineOptions, CompilerOptions

        # 創建配置
        options = CompilerOptions(
            model_name="my_model",
            hidden_dim=4096,
            quantization=QuantizationOptions(mode=QuantizationMode.FP8),
            attention=AttentionOptions(backend=AttentionBackend.KDA),
        )

        # 或使用預設配置
        options = CGCEngineOptions.create_preset("optimized")
        options = CGCEngineOptions.create_preset("edge_deployment")
        options = CGCEngineOptions.create_preset("training")
    """

    _presets: Dict[str, CompilerOptions] = {}

    @classmethod
    def create_preset(cls, name: Literal["optimized", "edge_deployment", "training", "inference", "minimal"]) -> CompilerOptions:
        """創建預設配置"""
        presets = {
            "optimized": CompilerOptions(
                model_name="optimized_model",
                hidden_dim=4096,
                num_layers=32,
                attention=AttentionOptions(
                    backend=AttentionBackend.FLASH_ATTENTION,
                    enable_flash_kda=True,
                    enable_kv_cache_static_layout=True,
                    kv_cache_quantize=QuantizationMode.INT8,
                ),
                quantization=QuantizationOptions(
                    mode=QuantizationMode.INT8,
                    enable_gptq_kernel=True,
                ),
                memory=MemoryOptions(
                    backend=MemoryBackend.PD,
                    enable_pd分离=True,
                ),
                performance=PerformanceOptions(
                    enable_profiler=True,
                ),
            ),
            "edge_deployment": CompilerOptions(
                model_name="edge_model",
                hidden_dim=2048,
                num_layers=24,
                attention=AttentionOptions(
                    backend=AttentionBackend.KDA,
                    enable_paged_attention=False,
                    kv_cache_quantize=QuantizationMode.INT4,
                ),
                quantization=QuantizationOptions(
                    mode=QuantizationMode.INT4,
                    group_size=64,
                ),
                memory=MemoryOptions(
                    backend=MemoryBackend.METAL_MPS,
                    metal_mps_enabled=True,
                ),
                export=ExportOptions(
                    formats=[ExportFormat.METAL, ExportFormat.ONNX],
                    target_device="apple_silicon",
                ),
            ),
            "training": CompilerOptions(
                model_name="training_model",
                hidden_dim=4096,
                num_layers=32,
                attention=AttentionOptions(
                    backend=AttentionBackend.KDA,
                    enable_flash_kda=True,
                ),
                quantization=QuantizationOptions(
                    mode=QuantizationMode.FP16,
                    enable_fp8_kernels=True,
                ),
                distributed=DistributedOptions(
                    enable_nccl=True,
                    enable_fsdp=True,
                ),
                performance=PerformanceOptions(
                    enable_profiler=True,
                    enable_dynamic_precision=True,
                ),
            ),
            "inference": CompilerOptions(
                model_name="inference_model",
                hidden_dim=4096,
                num_layers=32,
                attention=AttentionOptions(
                    backend=AttentionBackend.PAGED,
                    enable_kv_async_prefetch=True,
                    kv_cache_quantize=QuantizationMode.INT8,
                ),
                quantization=QuantizationOptions(
                    mode=QuantizationMode.INT8,
                    enable_gptq_kernel=True,
                    enable_awq_kernel=True,
                ),
                schedule_strategy=ScheduleStrategy.CONTINOUS_BATCHING,
                enable_dynamic_batching=True,
            ),
            "minimal": CompilerOptions(
                model_name="minimal_model",
                hidden_dim=1024,
                num_layers=12,
                attention=AttentionOptions(
                    backend=AttentionBackend.SDPA,
                    enable_flash_kda=False,
                    enable_kv_cache_static_layout=False,
                ),
                quantization=QuantizationOptions(
                    mode=QuantizationMode.FP16,
                ),
                memory=MemoryOptions(
                    backend=MemoryBackend.PYTORCH,
                    enable_pd分离=False,
                ),
            ),
        }

        return presets.get(name, presets["optimized"])

    @classmethod
    def get_all_options(cls) -> List[str]:
        """獲取所有配置選項的說明"""
        return [
            "model_name: 模型名稱",
            "hidden_dim: 隱藏層維度",
            "num_layers: 層數",
            "num_attention_heads: 注意力頭數",
            "vocab_size: 詞彙表大小",
            "attention.backend: 注意力後端 (flash/kda/paged/sdpa/auto)",
            "attention.enable_flash_kda: 啟用 FlashKDA",
            "attention.kv_cache_quantize: KV Cache 量化模式",
            "quantization.mode: 量化模式 (fp16/bf16/fp8/int8/int4/auto)",
            "quantization.group_size: 量化組大小",
            "memory.backend: 記憶體後端 (pd/gds/spdk/metal/pytorch)",
            "memory.enable_pd分离: 啟用 PD 分離",
            "moe.enable_flashmoe: 啟用 FlashMoE",
            "moe.enable_omlx: 啟用 oMLX 專家預測",
            "distributed.enable_nccl: 啟用 NCCL 分布式",
            "distributed.enable_fsdp: 啟用 FSDP",
            "performance.enable_profiler: 啟用性能分析",
            "performance.enable_memory_tracker: 啟用記憶體追蹤",
            "export.formats: 導出格式列表",
            "schedule_strategy: 調度策略",
            "enable_jit_compilation: 啟用 JIT 編譯",
            "enable_cuda_graph: 啟用 CUDA Graph",
            "device: 目標設備",
            "dtype: 數據類型",
        ]
