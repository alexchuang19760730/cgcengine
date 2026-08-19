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
vLLM Metal Integration - Apple Silicon GPU 加速

使用 vllm-metal 插件在 Apple Silicon 上運行 vLLM
使用 MLX 作為計算後端，充分發揮統一記憶體架構

官網: https://github.com/vllm-project/vllm-metal
文檔: https://docs.vllm.ai/projects/vllm-metal/
"""

import os
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

import torch

logger = logging.getLogger(__name__)


VLLM_METAL_AVAILABLE = False
VLLM_METAL_VERSION = None

try:
    import vllm
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
    logger.info(f"vLLM version: {vllm.__version__}")
except ImportError:
    VLLM_AVAILABLE = False
    vllm = None
    LLM = None
    SamplingParams = None
    logger.warning("vLLM not available. Install with: pip install vllm")


@dataclass
class MetalBackendConfig:
    """Metal 後端配置"""
    model_path: str = ""
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    num_kv_heads: Optional[int] = None
    enable_chunked_prefill: bool = True
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 256
    enable_prefix_caching: bool = True
    distributed_executor_backend: str = "mp"


class MetalBackendType(Enum):
    """Metal 後端類型"""
    MLX = "mlx"
    MPS = "mps"
    CPU = "cpu"
    AUTO = "auto"


class VLLMetalBackend:
    """
    vLLM Metal 後端 - Apple Silicon 統一調度入口

    支援:
    - MLX 加速 (需安裝 vllm-metal)
    - MPS 回退 (PyTorch Metal)
    - CPU 回退
    """

    def __init__(
        self,
        config: Optional[MetalBackendConfig] = None,
        backend_type: str = "auto",
    ):
        self.config = config or MetalBackendConfig()
        self.backend_type = MetalBackendType(backend_type)
        self.llm = None
        self.backend_info: Dict[str, Any] = {}

        self._detect_backend()
        self._init_backend()

    def _detect_backend(self) -> str:
        """自動檢測最優後端"""
        if torch.backends.mps.is_available():
            self.backend_type = MetalBackendType.MPS
            self.backend_info["mps_available"] = True
            self.backend_info["mps_device"] = torch.backends.mps.is_available()
        else:
            self.backend_type = MetalBackendType.CPU
            self.backend_info["mps_available"] = False

        self.backend_info["cuda_available"] = torch.cuda.is_available()
        self.backend_info["detected_backend"] = self.backend_type.value
        self.backend_info["vllm_available"] = VLLM_AVAILABLE

        logger.info(f"[VLLMetal] Detected backend: {self.backend_type.value}")
        return self.backend_type.value

    def _init_backend(self):
        """初始化後端"""
        if not VLLM_AVAILABLE:
            logger.warning("[VLLMetal] vLLM not available, using PyTorch backend")
            self._init_pytorch_backend()
            return

        try:
            if self.backend_type == MetalBackendType.MPS:
                self._init_mlx_backend()
            else:
                self._init_cpu_backend()
        except Exception as e:
            logger.warning(f"[VLLMetal] Failed to initialize {self.backend_type.value}: {e}")
            logger.info("[VLLMetal] Falling back to PyTorch backend")
            self._init_pytorch_backend()

    def _init_mlx_backend(self):
        """初始化 MLX 後端 (vllm-metal)"""
        logger.info("[VLLMetal] Initializing MLX backend (vllm-metal)...")

        env_vars = {
            "VLLM_METAL_DEVICE": "mlx",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        }
        for key, value in env_vars.items():
            os.environ[key] = value

        try:
            if self.config.model_path:
                self.llm = LLM(
                    model=self.config.model_path,
                    max_model_len=self.config.max_model_len,
                    gpu_memory_utilization=self.config.gpu_memory_utilization,
                    tensor_parallel_size=self.config.tensor_parallel_size,
                    enable_chunked_prefill=self.config.enable_chunked_prefill,
                    max_num_batched_tokens=self.config.max_num_batched_tokens,
                    max_num_seqs=self.config.max_num_seqs,
                    enable_prefix_caching=self.config.enable_prefix_caching,
                )
                self.backend_info["backend"] = "mlx"
                self.backend_info["model_loaded"] = True
                logger.info("[VLLMetal] MLX backend initialized successfully")
        except Exception as e:
            logger.warning(f"[VLLMetal] MLX backend init failed: {e}")
            raise

    def _init_cpu_backend(self):
        """初始化 CPU 後端"""
        logger.info("[VLLMetal] Initializing CPU backend...")

        try:
            if self.config.model_path:
                self.llm = LLM(
                    model=self.config.model_path,
                    max_model_len=self.config.max_model_len,
                    gpu_memory_utilization=0.0,
                    tensor_parallel_size=1,
                    enable_chunked_prefill=False,
                )
                self.backend_info["backend"] = "cpu"
                self.backend_info["model_loaded"] = True
                logger.info("[VLLMetal] CPU backend initialized successfully")
        except Exception as e:
            logger.warning(f"[VLLMetal] CPU backend init failed: {e}")
            raise

    def _init_pytorch_backend(self):
        """初始化 PyTorch 後端 (MPS/CPU)"""
        logger.info(f"[VLLMetal] Initializing PyTorch {self.backend_type.value.upper()} backend...")

        self.backend_info["backend"] = "pytorch"
        self.backend_info["pytorch_device"] = "mps" if self.backend_type == MetalBackendType.MPS else "cpu"
        self.backend_info["model_loaded"] = False

    def generate(
        self,
        prompts: List[str],
        sampling_params: Optional[SamplingParams] = None,
    ) -> List[str]:
        """生成文字"""
        if self.llm is None:
            raise RuntimeError("[VLLMetal] Backend not initialized")

        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=256,
            )

        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]

    def benchmark_prefill(
        self,
        prompt: str,
        seq_len: Optional[int] = None,
    ) -> Dict[str, float]:
        """Benchmark prefill 階段"""
        import time

        if seq_len is None:
            seq_len = len(prompt.split()) * 2

        start_time = time.time()
        outputs = self.generate([prompt])
        elapsed = time.time() - start_time

        return {
            "elapsed_ms": elapsed * 1000,
            "tokens_per_sec": len(outputs[0].split()) / elapsed if elapsed > 0 else 0,
            "seq_len": seq_len,
        }

    def benchmark_decode(
        self,
        prompt: str,
        max_tokens: int = 128,
    ) -> Dict[str, float]:
        """Benchmark decode 階段"""
        import time

        start_time = time.time()
        outputs = self.generate([prompt], SamplingParams(max_tokens=max_tokens))
        elapsed = time.time() - start_time

        return {
            "elapsed_ms": elapsed * 1000,
            "tokens_per_sec": max_tokens / elapsed if elapsed > 0 else 0,
            "generated_tokens": max_tokens,
        }

    def info(self) -> Dict[str, Any]:
        """獲取後端資訊"""
        return {
            "backend_type": self.backend_type.value,
            "vllm_available": VLLM_AVAILABLE,
            "vllm_metal_available": VLLM_METAL_AVAILABLE,
            "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(),
            "device": self.backend_info.get("backend", "unknown"),
            "model_loaded": self.backend_info.get("model_loaded", False),
        }


class CGCVLLMetalBridge:
    """
    CGC ↔ vLLM Metal 橋接器

    整合 CGC Engine 調度層與 vLLM Metal 執行層
    """

    def __init__(self, metal_backend: Optional[VLLMetalBackend] = None):
        self.metal_backend = metal_backend or VLLMetalBackend()
        self.cgc_executor = None

        try:
            from .cgc_simd_executor import CGCExecutor
            self.cgc_executor = CGCExecutor(enable_profiling=False)
            logger.info("[CGC-VLLMetal] CGC Executor initialized")
        except Exception as e:
            logger.warning(f"[CGC-VLLMetal] CGC Executor init failed: {e}")

    def run_with_cgc_schedule(
        self,
        prompt: str,
        use_cgc_kernels: bool = True,
    ) -> Dict[str, Any]:
        """使用 CGC 調度執行"""
        result = {
            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "metal_backend": self.metal_backend.backend_info.get("backend", "unknown"),
            "cgc_enabled": use_cgc_kernels and self.cgc_executor is not None,
        }

        if use_cgc_kernels and self.cgc_executor:
            result["schedule_method"] = "cgc"
        else:
            result["schedule_method"] = "native"

        return result

    def info(self) -> Dict[str, Any]:
        """獲取橋接器資訊"""
        return {
            "metal_backend": self.metal_backend.info(),
            "cgc_executor_available": self.cgc_executor is not None,
        }


def create_vllm_metal_backend(
    model_path: str = "",
    backend: str = "auto",
    **kwargs,
) -> VLLMetalBackend:
    """工廠函數：創建 vLLM Metal 後端"""
    config = MetalBackendConfig(model_path=model_path, **kwargs)
    return VLLMetalBackend(config=config, backend_type=backend)


if __name__ == "__main__":
    print("=" * 60)
    print("vLLM Metal Backend Test")
    print("=" * 60)

    backend = VLLMetalBackend()

    print("\nBackend Information:")
    for key, value in backend.info().items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("vLLM Metal Backend Test Complete")
    print("=" * 60)