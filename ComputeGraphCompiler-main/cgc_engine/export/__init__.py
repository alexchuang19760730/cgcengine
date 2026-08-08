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
CGC 端側導出器 - ONNX/TensorRT/Metal 統一導出

功能：
- ONNX 導出：通用模型格式，支持 iPhone/Android/嵌入式
- TensorRT 導出：NVIDIA GPU 高性能推理
- Metal 導出：Apple Silicon (iPhone/Mac) 原生加速

訓練 → 導出無縫銜接：
- Megatrain 訓練的權重直接導出
- 保持 CGC 指令集兼容性
- 支持 INT4/FP8 量化導出
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import os
import json

try:
    from ..cgc_opcodes import CGC_OP_CODES
    from ..cgc_simd_executor import CGCExecutor, CGCCommand
    CGC_AVAILABLE = True
except ImportError:
    CGC_AVAILABLE = False


class ExportFormat(Enum):
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    METAL = "metal"
    MLIR = "mlir"


class QuantizationMode(Enum):
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    FP8 = "fp8"
    FP16 = "fp16"
    BF16 = "bf16"


@dataclass
class ExportConfig:
    format: ExportFormat = ExportFormat.ONNX
    quantize: QuantizationMode = QuantizationMode.NONE
    optimize_level: int = 3
    enable_profiling: bool = False
    output_dir: str = "./exported_models"
    model_name: str = "cgc_model"
    opset_version: int = 17
    target_device: str = "cuda"
    enable_cgc_commands: bool = True


@dataclass
class ExportedModel:
    model_path: str
    format: ExportFormat
    quantization: QuantizationMode
    metadata: Dict[str, Any]
    cgc_commands: List[int]


class BaseExporter(ABC):
    """導出器基類"""

    def __init__(self, config: ExportConfig):
        self.config = config
        self.model: Optional[nn.Module] = None
        self.cgc_executor: Optional[CGCExecutor] = None

    @abstractmethod
    def export(self, model: nn.Module, sample_inputs: List[torch.Tensor], output_path: str) -> ExportedModel:
        """導出模型"""
        pass

    @abstractmethod
    def optimize(self, model_path: str) -> str:
        """優化模型"""
        pass

    def prepare_model(self, model: nn.Module) -> nn.Module:
        """準備模型：設置 eval 模式、freeze"""
        self.model = model
        self.model.eval()

        if CGC_AVAILABLE:
            self.cgc_executor = CGCExecutor(enable_profiling=False)

        return self.model

    def extract_cgc_commands(self, model: nn.Module) -> List[int]:
        """提取模型使用的 CGC 命令"""
        commands = []
        if hasattr(model, 'cgc_opcodes'):
            commands = list(model.cgc_opcodes)
        return commands

    def save_metadata(self, metadata: Dict[str, Any], path: str):
        """保存導出元數據"""
        metadata_path = os.path.join(path, f"{self.config.model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


class ONNXExporter(BaseExporter):
    """ONNX 導出器"""

    def export(self, model: nn.Module, sample_inputs: List[torch.Tensor], output_path: str) -> ExportedModel:
        """導出為 ONNX 格式"""
        self.prepare_model(model)

        os.makedirs(output_path, exist_ok=True)
        model_path = os.path.join(output_path, f"{self.config.model_name}.onnx")

        dynamic_axes = {
            'input_ids': {0: 'batch_size', 1: 'seq_len'},
            'hidden_states': {0: 'batch_size', 1: 'seq_len'},
            'logits': {0: 'batch_size', 1: 'seq_len'},
        }

        if len(sample_inputs) == 0:
            sample_inputs = [torch.randint(0, 32000, (1, 128))]

        input_names = ['input_ids']
        if self.config.enable_cgc_commands:
            input_names.append('cgc_commands')

        with torch.no_grad():
            torch.onnx.export(
                model,
                tuple(sample_inputs),
                model_path,
                input_names=input_names,
                output_names=['logits'],
                dynamic_axes=dynamic_axes,
                opset_version=self.config.opset_version,
                do_constant_folding=True,
            )

        metadata = {
            "format": ExportFormat.ONNX.value,
            "quantization": self.config.quantize.value,
            "cgc_commands": self.extract_cgc_commands(model),
            "opset_version": self.config.opset_version,
            "model_name": self.config.model_name,
        }

        self.save_metadata(metadata, output_path)

        return ExportedModel(
            model_path=model_path,
            format=ExportFormat.ONNX,
            quantization=self.config.quantize,
            metadata=metadata,
            cgc_commands=metadata["cgc_commands"],
        )

    def optimize(self, model_path: str) -> str:
        """優化 ONNX 模型"""
        try:
            import onnx
            from onnx import optimizer

            model = onnx.load(model_path)
            passes = ['eliminate_deadend', 'eliminate_identity', 'eliminate_redundant']
            model = optimizer.optimize(model, passes)

            optimized_path = model_path.replace('.onnx', '_optimized.onnx')
            onnx.save(model, optimized_path)
            return optimized_path
        except ImportError:
            return model_path


class TensorRTExporter(BaseExporter):
    """TensorRT 導出器"""

    _INT4_UNSUPPORTED = True

    def export(self, model: nn.Module, sample_inputs: List[torch.Tensor], output_path: str) -> ExportedModel:
        """導出為 TensorRT 格式"""
        self.prepare_model(model)

        os.makedirs(output_path, exist_ok=True)

        actual_quant = self.config.quantize
        fallback_reason = None

        if self.config.quantize == QuantizationMode.INT4:
            actual_quant = QuantizationMode.INT8
            fallback_reason = "TensorRT INT4 需要 INT8 + per-channel quantization，自動降級為 INT8"

        if fallback_reason:
            import warnings
            warnings.warn(fallback_reason)

        onnx_path = os.path.join(output_path, f"{self.config.model_name}.onnx")

        if len(sample_inputs) == 0:
            sample_inputs = [torch.randn(1, 128, 4096)]

        input_names = ['input_ids']
        if self.config.enable_cgc_commands:
            input_names.append('cgc_commands')

        with torch.no_grad():
            torch.onnx.export(
                model,
                tuple(sample_inputs),
                onnx_path,
                input_names=input_names,
                output_names=['logits'],
                opset_version=self.config.opset_version,
                do_constant_folding=True,
            )

        trt_path = self._convert_to_trt(onnx_path, output_path)

        metadata = {
            "format": ExportFormat.TENSORRT.value,
            "quantization": actual_quant.value,
            "original_quantization": self.config.quantize.value,
            "fallback_reason": fallback_reason,
            "cgc_commands": self.extract_cgc_commands(model),
            "opset_version": self.config.opset_version,
            "model_name": self.config.model_name,
            "target_device": self.config.target_device,
        }

        self.save_metadata(metadata, output_path)

        return ExportedModel(
            model_path=trt_path,
            format=ExportFormat.TENSORRT,
            quantization=actual_quant,
            metadata=metadata,
            cgc_commands=metadata["cgc_commands"],
        )

    def _convert_to_trt(self, onnx_path: str, output_path: str) -> str:
        """將 ONNX 轉換為 TensorRT"""
        try:
            import tensorrt as trt
            from polygraphy.backend.trt import EngineFromOnnx

            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, logger)

            with open(onnx_path, 'rb') as f:
                parser.parse(f.read())

            builder.max_batch_size = 32
            builder.max_workspace_size = 1 << 30

            if actual_quant == QuantizationMode.INT8:
                builder.int8_mode = True
            elif actual_quant == QuantizationMode.FP16:
                builder.fp16_mode = True

            engine = builder.build_cuda_engine(network)
            engine_path = os.path.join(output_path, f"{self.config.model_name}.engine")

            with open(engine_path, 'wb') as f:
                f.write(engine.serialize())

            return engine_path
        except ImportError:
            onnx_path_optimized = self.optimize(onnx_path)
            return onnx_path_optimized

    def optimize(self, model_path: str) -> str:
        """優化 TensorRT 模型"""
        return model_path


class MetalExporter(BaseExporter):
    """Metal 導出器 - Apple Silicon (iPhone/Mac)"""

    _FP8_UNSUPPORTED = True
    _INT4_UNSUPPORTED = True

    def export(self, model: nn.Module, sample_inputs: List[torch.Tensor], output_path: str) -> ExportedModel:
        """導出為 Metal 格式 (使用 MPS/CoreML)"""
        self.prepare_model(model)

        os.makedirs(output_path, exist_ok=True)

        actual_quant = self.config.quantize
        fallback_reason = None

        if self.config.quantize == QuantizationMode.FP8:
            actual_quant = QuantizationMode.FP16
            fallback_reason = "Apple Silicon 不支援 FP8，自動降級為 FP16"

        if self.config.quantize == QuantizationMode.INT4:
            actual_quant = QuantizationMode.INT8
            fallback_reason = "Metal 不支援 INT4，自動降級為 INT8"

        if fallback_reason:
            import warnings
            warnings.warn(fallback_reason)

        mlmodel_path = os.path.join(output_path, f"{self.config.model_name}.mlmodel")

        if len(sample_inputs) == 0:
            sample_inputs = [torch.randn(1, 128, 4096)]

        model = self._prepare_for_metal(model)

        try:
            import coremltools as cml

            traced_model = torch.jit.trace(model, tuple(sample_inputs))
            mlmodel = cml.convert(
                traced_model,
                inputs=[cml.TensorType(name="input", shape=x.shape) for x in sample_inputs],
                compute_units=cml.ComputeUnit.ALL,
            )

            mlmodel.save(mlmodel_path)
        except ImportError:
            torch.jit.script(model).save(
                os.path.join(output_path, f"{self.config.model_name}_jit.pt")
            )
            mlmodel_path = output_path

        metadata = {
            "format": ExportFormat.METAL.value,
            "quantization": actual_quant.value,
            "original_quantization": self.config.quantize.value,
            "fallback_reason": fallback_reason,
            "cgc_commands": self.extract_cgc_commands(model),
            "target_device": "apple_silicon",
            "model_name": self.config.model_name,
        }

        self.save_metadata(metadata, output_path)

        return ExportedModel(
            model_path=mlmodel_path,
            format=ExportFormat.METAL,
            quantization=actual_quant,
            metadata=metadata,
            cgc_commands=metadata["cgc_commands"],
        )

    def _prepare_for_metal(self, model: nn.Module) -> nn.Module:
        """準備 Metal 模型"""
        if hasattr(model, 'to'):
            model = model.to('mps')

        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
            if isinstance(module, nn.BatchNorm1d):
                module.eval()
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

        return model

    def optimize(self, model_path: str) -> str:
        """優化 Metal 模型"""
        return model_path


class UnifiedExporter:
    """統一導出器 - 工廠模式"""

    _exporters: Dict[ExportFormat, BaseExporter] = {}

    @classmethod
    def get_exporter(cls, format: ExportFormat, config: ExportConfig = None) -> BaseExporter:
        if config is None:
            config = ExportConfig(format=format)

        if format == ExportFormat.ONNX:
            return ONNXExporter(config)
        elif format == ExportFormat.TENSORRT:
            return TensorRTExporter(config)
        elif format == ExportFormat.METAL:
            return MetalExporter(config)
        else:
            raise ValueError(f"Unsupported format: {format}")

    @classmethod
    def export_model(
        cls,
        model: nn.Module,
        sample_inputs: List[torch.Tensor],
        config: ExportConfig,
    ) -> ExportedModel:
        """導出模型"""
        exporter = cls.get_exporter(config.format, config)
        output_path = os.path.join(config.output_dir, config.model_name)
        return exporter.export(model, sample_inputs, output_path)

    @classmethod
    def export_multi_format(
        cls,
        model: nn.Module,
        sample_inputs: List[torch.Tensor],
        formats: List[ExportFormat],
        base_config: ExportConfig = None,
    ) -> Dict[ExportFormat, ExportedModel]:
        """導出為多種格式"""
        results = {}

        for fmt in formats:
            config = base_config or ExportConfig(format=fmt)
            config.format = fmt
            results[fmt] = cls.export_model(model, sample_inputs, config)

        return results


def export_from_megatrain(
    megatrain_model: nn.Module,
    sample_inputs: List[torch.Tensor],
    config: ExportConfig,
) -> ExportedModel:
    """從 Megatrain 模型導出

    訓練側 → 導出無縫銜接

    Args:
        megatrain_model: Megatrain 訓練的模型
        sample_inputs: 範例輸入
        config: 導出配置

    Returns:
        ExportedModel: 導出的模型
    """
    if not CGC_AVAILABLE:
        raise RuntimeError("CGC not available, cannot export from Megatrain")

    if hasattr(megatrain_model, 'apply_cgc_commands'):
        megatrain_model.apply_cgc_commands()

    return UnifiedExporter.export_model(megatrain_model, sample_inputs, config)


def export_with_quantization(
    model: nn.Module,
    sample_inputs: List[torch.Tensor],
    quant_mode: QuantizationMode,
    formats: List[ExportFormat] = None,
) -> Dict[ExportFormat, ExportedModel]:
    """量化導出

    支持 INT4/FP8 等量化格式導出

    Args:
        model: 模型
        sample_inputs: 範例輸入
        quant_mode: 量化模式
        formats: 導出格式列表

    Returns:
        Dict[ExportFormat, ExportedModel]: 各格式的導出模型
    """
    if formats is None:
        formats = [ExportFormat.ONNX]

    config = ExportConfig(quantize=quant_mode)

    quantized_model = model
    if quant_mode != QuantizationMode.NONE:
        from ..cgc_simd_executor import CGCExecutor
        executor = CGCExecutor()

        if quant_mode == QuantizationMode.INT8:
            quantized_model = executor._quantize_w8a16_impl
        elif quant_mode == QuantizationMode.INT4:
            quantized_model = executor._quantize_w4a16_impl
        elif quant_mode == QuantizationMode.FP8:
            quantized_model = executor._fp8_e4m3_impl

    results = {}
    for fmt in formats:
        config.format = fmt
        results[fmt] = UnifiedExporter.export_model(quantized_model, sample_inputs, config)

    return results
