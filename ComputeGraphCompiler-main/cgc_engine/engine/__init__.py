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
CGC Engine - 統一的深度學習編譯與執行引擎

CGC Engine 整合了以下功能:
- CGC SIMD 執行器 (統一算子執行)
- PD 存儲層 (權重/KV Cache 管理)
- UnifiedIO 控制器 (跨平台 I/O)
- FlashMoE / oMLX (MoE 專家調度)
- JITLoad (即時編譯)
- 性能分析與導出

使用範例:
    from cgc_engine.engine import CGCEngine, CGCEngineOptions

    # 方式1: 使用預設配置
    engine = CGCEngine.from_preset("optimized")

    # 方式2: 自定義配置
    options = CGCEngineOptions.create_preset("training")
    options.quantization.mode = QuantizationMode.FP8
    engine = CGCEngine(options)

    # 編譯模型
    compiled_model = engine.compile(model, sample_inputs)

    # 執行推理
    output = engine.forward(compiled_model, inputs)

    # 導出模型
    engine.export(compiled_model, format=ExportFormat.TENSORRT)
"""

import os
import json
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime

import torch
import torch.nn as nn

from .options import (
    CompilerOptions,
    CGCEngineOptions,
    AttentionOptions,
    QuantizationOptions,
    MemoryOptions,
    MoEOptions,
    DistributedOptions,
    PerformanceOptions,
    ExportOptions,
    QuantizationMode,
    AttentionBackend,
    MemoryBackend,
    ExportFormat,
    ScheduleStrategy,
)


class CGCEngine:
    """
    CGC Engine 主類

    整合所有 CGC Engine 組件，提供統一的編譯和執行接口。
    """

    def __init__(self, options: Optional[CompilerOptions] = None):
        """
        初始化 CGC Engine

        Args:
            options: 編譯器配置，如果為 None 則使用默認配置
        """
        self.options = options or CompilerOptions()
        self._initialized = False
        self._compiled_model: Optional[nn.Module] = None
        self._executor = None
        self._profiler = None
        self._io_controller = None
        self._pd_client = None

    @classmethod
    def from_preset(cls, preset: str) -> "CGCEngine":
        """從預設配置創建 CGC Engine"""
        options = CGCEngineOptions.create_preset(preset)
        return cls(options)

    def initialize(self):
        """初始化所有組件"""
        if self._initialized:
            return

        self._init_executor()
        self._init_io_controller()
        self._init_profiler()
        self._init_pd_client()

        self._initialized = True

    def _init_executor(self):
        """初始化 CGC 執行器"""
        try:
            from ..cgc.cgc_simd_executor import CGCExecutor
            self._executor = CGCExecutor(device=self.options.device)
        except ImportError:
            from ..cgc.cgc_unified_executor import CGCUnifiedExecutor
            self._executor = CGCUnifiedExecutor(device=self.options.device)

    def _init_io_controller(self):
        """初始化 I/O 控制器"""
        try:
            from ..io_unified.unified_io_controller import UnifiedIOController
            from ..io_unified.pytorch_backend import PyTorchBackend
            from ..io_unified.metal_backend import MetalBackend

            backend_map = {
                "pytorch": PyTorchBackend(),
            }

            if self.options.memory.metal_mps_enabled:
                backend_map["metal"] = MetalBackend()

            self._io_controller = UnifiedIOController(backend_map=backend_map)
        except ImportError:
            self._io_controller = None

    def _init_profiler(self):
        """初始化性能分析器"""
        if self.options.performance.enable_profiler:
            try:
                from ..profiler import get_profiler
                self._profiler = get_profiler()
            except ImportError:
                self._profiler = None

    def _init_pd_client(self):
        """初始化 PD 客戶端"""
        if self.options.memory.enable_pd分离:
            try:
                from ..pd.pd_client import PDClient
                self._pd_client = PDClient(
                    server_address=self.options.memory.pd_server_address
                )
            except ImportError:
                self._pd_client = None

    def compile(
        self,
        model: nn.Module,
        sample_inputs: List[torch.Tensor],
    ) -> nn.Module:
        """
        編譯模型

        Args:
            model: 要編譯的模型
            sample_inputs: 樣本輸入張量列表

        Returns:
            編譯後的模型
        """
        self.initialize()

        if self.options.quantization.mode != QuantizationMode.FP16:
            model = self._apply_quantization(model)

        if self.options.attention.enable_flash_kda:
            model = self._apply_flash_kda(model)

        if self.options.moe.enable_flashmoe or self.options.moe.enable_omlx:
            model = self._apply_moe(model)

        if self.options.schedule_strategy == ScheduleStrategy.HEURISTIC:
            model = self._apply_scheduling(model)

        self._compiled_model = model
        return model

    def _apply_quantization(self, model: nn.Module) -> nn.Module:
        """應用量化"""
        mode = self.options.quantization.mode

        if mode == QuantizationMode.FP8:
            try:
                from ..cgc.fp8_support import FP8Quantizer
                quantizer = FP8Quantizer()
                return quantizer.quantize_model(model)
            except ImportError:
                pass

        elif mode in (QuantizationMode.INT8, QuantizationMode.INT4):
            try:
                from ..pd.kv_quantizer import KVCacheQuantizer
                quantizer = KVCacheQuantizer(bits=8 if mode == QuantizationMode.INT8 else 4)
                return quantizer
            except ImportError:
                pass

        return model

    def _apply_flash_kda(self, model: nn.Module) -> nn.Module:
        """應用 FlashKDA"""
        try:
            from ..cgc.flashkda_integration import apply_flashkda_to_model
            return apply_flashkda_to_model(model)
        except ImportError:
            return model

    def _apply_moe(self, model: nn.Module) -> nn.Module:
        """應用 MoE 優化"""
        if self.options.moe.enable_flashmoe:
            try:
                from ..flash_moe.client import FlashMoEClient
                return model
            except ImportError:
                pass

        if self.options.moe.enable_omlx:
            try:
                from ..omlx.client import OMLXClient
                return model
            except ImportError:
                pass

        return model

    def _apply_scheduling(self, model: nn.Module) -> nn.Module:
        """應用調度策略"""
        try:
            from ..offload.scheduler import OffloadScheduler
            scheduler = OffloadScheduler(
                options=self.options
            )
            return scheduler
        except ImportError:
            return model

    def forward(
        self,
        model: nn.Module,
        inputs: Union[torch.Tensor, List[torch.Tensor]],
    ) -> torch.Tensor:
        """
        前向推導

        Args:
            model: 編譯後的模型
            inputs: 輸入張量

        Returns:
            模型輸出
        """
        if not self._initialized:
            self.initialize()

        if isinstance(inputs, torch.Tensor):
            inputs = [inputs]

        with torch.no_grad():
            if self._profiler and self.options.performance.enable_profiler:
                result = self._profiler.profile_cgc_command(
                    None,
                    lambda: model(*inputs)
                )
            else:
                result = model(*inputs)

        return result

    def export(
        self,
        model: nn.Module,
        format: ExportFormat,
        output_dir: str = "./exports",
    ) -> str:
        """
        導出模型

        Args:
            model: 編譯後的模型
            format: 導出格式
            output_dir: 輸出目錄

        Returns:
            導出的模型路徑
        """
        from ..export import UnifiedExporter, ExportConfig

        config = ExportConfig(
            format=format.value,
            quantize=self.options.quantization.mode.value,
            model_name=self.options.model_name,
            output_dir=output_dir,
            enable_cgc_commands=self.options.export.enable_cgc_commands,
        )

        exporter = UnifiedExporter.get_exporter(format.value, config)
        result = exporter.export(model, [], output_dir)

        return result.model_path

    def profile(
        self,
        model: Optional[nn.Module] = None,
        inputs: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """
        性能分析

        Args:
            model: 要分析的模型（如果為 None，則分析上次編譯的模型）
            inputs: 輸入張量（如果為 None，則使用樣本輸入）

        Returns:
            性能分析報告
        """
        if self._profiler is None:
            return {"error": "Profiler not enabled"}

        if model is None:
            model = self._compiled_model

        if inputs is None:
            return self._profiler.generate_full_report(
                self.options.performance.profile_output_dir
            )

        for _ in range(10):
            model(*inputs)

        return self._profiler.generate_full_report(
            self.options.performance.profile_output_dir
        )

    def get_memory_stats(self) -> Dict[str, Any]:
        """獲取記憶體統計"""
        if self._profiler is None:
            return {"error": "Profiler not enabled"}

        return self._profiler.memory_tracker.get_waste_analysis()

    def get_config(self) -> Dict[str, Any]:
        """獲取當前配置"""
        return self.options.to_dict()

    def save_config(self, path: str):
        """保存配置到文件"""
        config = self.get_config()
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load_config(cls, path: str) -> "CGCEngine":
        """從文件加載配置"""
        with open(path, 'r') as f:
            config = json.load(f)
        options = CompilerOptions.from_dict(config)
        return cls(options)


class CGCEngineLite:
    """
    CGC Engine 輕量版本

    適用於邊緣設備和資源受限環境。
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._executor = None

    def compile_and_run(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """編譯並運行"""
        model = model.to(self.device)
        model.eval()

        with torch.no_grad():
            return model(inputs)


@dataclass
class CGCBuildInfo:
    """CGC Engine 構建信息"""
    version: str = "1.0.0"
    build_date: str = field(default_factory=lambda: datetime.now().isoformat())
    cuda_available: bool = field(default_factory=lambda: torch.cuda.is_available())
    metal_available: bool = False
    cgc_opcode_count: int = 0
    supported_backends: List[str] = field(default_factory=list)

    @classmethod
    def collect(cls) -> "CGCBuildInfo":
        """收集當前環境信息"""
        info = cls()

        if torch.cuda.is_available():
            info.supported_backends.append("cuda")
            info.supported_backends.append("pytorch")

        try:
            import torch.backends.mps as mps
            if mps.is_available():
                info.metal_available = True
                info.supported_backends.append("metal")
        except ImportError:
            pass

        try:
            from ..cgc.cgc_opcodes import CGC_OP_CODES
            info.cgc_opcode_count = len(CGC_OP_CODES)
        except ImportError:
            pass

        return info

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "build_date": self.build_date,
            "cuda_available": self.cuda_available,
            "metal_available": self.metal_available,
            "cgc_opcode_count": self.cgc_opcode_count,
            "supported_backends": self.supported_backends,
        }


def get_build_info() -> CGCBuildInfo:
    """獲取 CGC Engine 構建信息"""
    return CGCBuildInfo.collect()
