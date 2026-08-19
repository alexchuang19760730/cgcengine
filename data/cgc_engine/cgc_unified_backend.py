#!/usr/bin/env python3
"""
CGC Engine 统一后端调度器

支持动态切换 vLLM / llama.cpp / FlashMoE+oMLX+KDA 后端
"""

import torch
import logging
from typing import Union, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """支持的推理后端类型"""
    VLLM = "vllm"
    LLAMA_CPP = "llama_cpp"
    FLASHMOE = "flashmoe"
    PYTHON = "python"
    AUTO = "auto"


@dataclass
class BackendConfig:
    """后端配置"""
    backend_type: BackendType = BackendType.AUTO
    model_path: Optional[str] = None
    gguf_path: Optional[str] = None
    vllm_path: Optional[str] = None
    expert_dir: Optional[str] = None
    device: str = "auto"
    max_tokens: int = 100
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9


class UnifiedBackendAdapter:
    """
    统一后端适配器

    提供统一的推理接口，支持动态切换不同后端：
    - vLLM: CUDA 高性能推理
    - llama.cpp: GGUF 模型推理
    - FlashMoE: oMLX + FlashMoE + KDA 混合架构
    - Python: PyTorch 原生推理
    """

    def __init__(self, config: Optional[BackendConfig] = None):
        self.config = config or BackendConfig()
        self._backends = {}
        self._active_backend: Optional[BackendType] = None
        self._device = self._detect_device()

        if self.config.backend_type == BackendType.AUTO:
            self._auto_select_backend()
        else:
            self._init_backend(self.config.backend_type)

    def _detect_device(self) -> str:
        """自动检测设备"""
        if self.config.device != "auto":
            return self.config.device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _auto_select_backend(self) -> None:
        """根据环境和模型自动选择最佳后端"""
        if torch.cuda.is_available() and self.config.vllm_path:
            logger.info("[UnifiedBackend] CUDA available, selecting vLLM backend")
            self._init_backend(BackendType.VLLM)
        elif self.config.gguf_path:
            logger.info("[UnifiedBackend] GGUF model, selecting llama.cpp backend")
            self._init_backend(BackendType.LLAMA_CPP)
        elif self.config.expert_dir:
            logger.info("[UnifiedBackend] Expert weights found, selecting FlashMoE backend")
            self._init_backend(BackendType.FLASHMOE)
        else:
            logger.info("[UnifiedBackend] No specific backend selected, using Python backend")
            self._init_backend(BackendType.PYTHON)

    def _init_backend(self, backend_type: BackendType) -> None:
        """初始化指定后端"""
        if backend_type == BackendType.VLLM:
            self._init_vllm_backend()
        elif backend_type == BackendType.LLAMA_CPP:
            self._init_llama_cpp_backend()
        elif backend_type == BackendType.FLASHMOE:
            self._init_flashmoe_backend()
        elif backend_type == BackendType.PYTHON:
            self._init_python_backend()

        self._active_backend = backend_type
        logger.info(f"[UnifiedBackend] Active backend: {backend_type.value}")

    def _init_vllm_backend(self) -> None:
        """初始化 vLLM 后端"""
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer

            model_path = self.config.vllm_path or self.config.model_path
            if not model_path:
                raise ValueError("vllm_path or model_path is required for vLLM backend")

            if not torch.cuda.is_available():
                raise RuntimeError("vLLM backend requires CUDA GPU")

            logger.info(f"[UnifiedBackend] Loading vLLM model: {model_path}")

            self._backends[BackendType.VLLM] = {
                "llm": LLM(
                    model=model_path,
                    trust_remote_code=True,
                    tensor_parallel_size=self.config.tensor_parallel_size,
                    gpu_memory_utilization=self.config.gpu_memory_utilization,
                    dtype="float16",
                ),
                "tokenizer": AutoTokenizer.from_pretrained(model_path, trust_remote_code=True),
                "sampling_params": SamplingParams(
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_tokens=self.config.max_tokens,
                ),
            }
            logger.info("[UnifiedBackend] vLLM backend initialized")

        except ImportError:
            logger.error("[UnifiedBackend] vLLM not installed. Run: pip install vllm")
            raise
        except Exception as e:
            logger.error(f"[UnifiedBackend] Failed to initialize vLLM: {e}")
            raise

    def _init_llama_cpp_backend(self) -> None:
        """初始化 llama.cpp 后端"""
        try:
            from .cgc.cgc_runtime import CGCRuntime

            if not self.config.gguf_path:
                raise ValueError("gguf_path is required for llama.cpp backend")

            logger.info(f"[UnifiedBackend] Loading GGUF model: {self.config.gguf_path}")

            self._backends[BackendType.LLAMA_CPP] = CGCRuntime(
                model_config=None,
                device=self._device,
                enable_llama_cpp_bridge=True,
                gguf_path=self.config.gguf_path,
            )
            logger.info("[UnifiedBackend] llama.cpp backend initialized")

        except Exception as e:
            logger.error(f"[UnifiedBackend] Failed to initialize llama.cpp: {e}")
            raise

    def _init_flashmoe_backend(self) -> None:
        """初始化 FlashMoE + oMLX + KDA 后端"""
        try:
            import sys
            from pathlib import Path

            cgc_path = Path(__file__).parent.parent
            sys.path.insert(0, str(cgc_path))

            from cgc_engine.flash_moe import FlashMoEClient
            from cgc_engine.omlx import OMLXClient

            expert_dir = self.config.expert_dir
            if not expert_dir:
                expert_dir = str(cgc_path / "expert_weights")

            logger.info(f"[UnifiedBackend] Initializing FlashMoE backend from: {expert_dir}")

            self._backends[BackendType.FLASHMOE] = {
                "flashmoe": FlashMoEClient(expert_dir=expert_dir, backend="auto"),
                "omlx": OMLXClient(model_dir="/tmp/omlx_model"),
            }
            logger.info("[UnifiedBackend] FlashMoE backend initialized")

        except Exception as e:
            logger.error(f"[UnifiedBackend] Failed to initialize FlashMoE: {e}")
            raise

    def _init_python_backend(self) -> None:
        """初始化 PyTorch 后端"""
        logger.info("[UnifiedBackend] PyTorch backend initialized")
        self._backends[BackendType.PYTHON] = {
            "model": None,
            "tokenizer": None,
        }

    def load_pytorch_model(self, model_path: str, tokenizer_path: Optional[str] = None) -> None:
        """
        加载 PyTorch 模型到 Python 后端

        Args:
            model_path: 模型路径 (HuggingFace format)
            tokenizer_path: Tokenizer 路径 (可选)
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"[UnifiedBackend] Loading PyTorch model: {model_path}")

            self._backends[BackendType.PYTHON] = {
                "model": AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map=self._device,
                    trust_remote_code=True,
                ),
                "tokenizer": AutoTokenizer.from_pretrained(
                    tokenizer_path or model_path,
                    trust_remote_code=True,
                ),
            }
            logger.info("[UnifiedBackend] PyTorch model loaded")

        except Exception as e:
            logger.error(f"[UnifiedBackend] Failed to load PyTorch model: {e}")
            raise

    def generate(
        self,
        prompt: Union[str, List[int]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        统一生成接口

        Args:
            prompt: 输入文本或 token IDs
            max_tokens: 最大生成长度
            temperature: 温度
            **kwargs: 其他参数

        Returns:
            生成结果字典
        """
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature if temperature is not None else self.config.temperature

        if self._active_backend == BackendType.VLLM:
            return self._generate_vllm(prompt, max_tokens, temperature, **kwargs)
        elif self._active_backend == BackendType.LLAMA_CPP:
            return self._generate_llama_cpp(prompt, max_tokens, temperature, **kwargs)
        elif self._active_backend == BackendType.FLASHMOE:
            return self._generate_flashmoe(prompt, max_tokens, temperature, **kwargs)
        else:
            return self._generate_python(prompt, max_tokens, temperature, **kwargs)

    def _generate_vllm(
        self,
        prompt: Union[str, List[int]],
        max_tokens: int,
        temperature: float,
        **kwargs,
    ) -> Dict[str, Any]:
        """vLLM 生成"""
        backend = self._backends[BackendType.VLLM]
        llm = backend["llm"]
        tokenizer = backend["tokenizer"]

        if isinstance(prompt, str):
            input_ids = tokenizer.encode(prompt)
        else:
            input_ids = prompt

        sampling_params = backend["sampling_params"]
        sampling_params.max_tokens = max_tokens
        sampling_params.temperature = temperature

        outputs = llm.generate([input_ids] if isinstance(input_ids, list) else [prompt], sampling_params)

        return {
            "backend": "vLLM",
            "prompt": prompt,
            "output": outputs[0].outputs[0].text,
            "num_tokens": outputs[0].outputs[0].token_ids,
        }

    def _generate_llama_cpp(
        self,
        prompt: Union[str, List[int]],
        max_tokens: int,
        temperature: float,
        **kwargs,
    ) -> Dict[str, Any]:
        """llama.cpp 生成"""
        backend = self._backends[BackendType.LLAMA_CPP]

        if isinstance(prompt, str):
            result = backend.generate(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
        else:
            result = backend.generate(
                prompt,
                max_new_tokens=max_tokens,
                **kwargs,
            )

        return {
            "backend": "llama.cpp",
            "prompt": prompt,
            "output": result,
        }

    def _generate_flashmoe(
        self,
        prompt: Union[str, List[int]],
        max_tokens: int,
        temperature: float,
        **kwargs,
    ) -> Dict[str, Any]:
        """FlashMoE + oMLX + KDA 生成"""
        backend = self._backends[BackendType.FLASHMOE]
        flashmoe = backend["flashmoe"]
        omlx = backend["omlx"]

        if isinstance(prompt, str):
            prompt_tokens = list(range(len(prompt)))
        else:
            prompt_tokens = prompt

        num_tokens = len(prompt_tokens)

        return {
            "backend": "FlashMoE+oMLX+KDA",
            "prompt": prompt,
            "output": f"[FlashMoE模拟输出] 收到 {num_tokens} tokens",
            "num_tokens": num_tokens,
        }

    def _generate_python(
        self,
        prompt: Union[str, List[int]],
        max_tokens: int,
        temperature: float,
        **kwargs,
    ) -> Dict[str, Any]:
        """PyTorch 后端生成"""
        backend = self._backends.get(BackendType.PYTHON, {})
        model = backend.get("model")
        tokenizer = backend.get("tokenizer")

        if model is not None and tokenizer is not None:
            inputs = tokenizer(prompt, return_tensors="pt").to(self._device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return {
                "backend": "PyTorch",
                "prompt": prompt,
                "output": generated_text,
                "num_tokens": len(outputs[0]),
            }

        if isinstance(prompt, str):
            return {
                "backend": "PyTorch",
                "prompt": prompt,
                "output": f"[PyTorch模拟] 收到: {prompt}",
            }
        else:
            return {
                "backend": "PyTorch",
                "prompt": prompt,
                "output": f"[PyTorch模拟] 收到 {len(prompt)} tokens",
            }

    def switch_backend(self, backend_type: BackendType) -> None:
        """
        切换推理后端

        Args:
            backend_type: 目标后端类型
        """
        if backend_type == self._active_backend:
            logger.info(f"[UnifiedBackend] Already using {backend_type.value} backend")
            return

        if backend_type not in self._backends:
            self._init_backend(backend_type)

        self._active_backend = backend_type
        logger.info(f"[UnifiedBackend] Switched to {backend_type.value} backend")

    def get_active_backend(self) -> str:
        """获取当前活跃后端"""
        return self._active_backend.value if self._active_backend else "none"

    def get_available_backends(self) -> List[str]:
        """获取可用后端列表"""
        return [b.value for b in self._backends.keys()]


def create_unified_engine(
    model_path: Optional[str] = None,
    gguf_path: Optional[str] = None,
    vllm_path: Optional[str] = None,
    expert_dir: Optional[str] = None,
    backend: str = "auto",
    **kwargs,
) -> UnifiedBackendAdapter:
    """
    创建统一后端引擎的工厂函数

    Args:
        model_path: 模型路径
        gguf_path: GGUF 模型路径
        vllm_path: vLLM 模型路径
        expert_dir: 专家权重目录
        backend: 后端类型 ("vllm", "llama_cpp", "flashmoe", "auto")
        **kwargs: 其他配置参数

    Returns:
        UnifiedBackendAdapter 实例
    """
    config = BackendConfig(
        model_path=model_path,
        gguf_path=gguf_path,
        vllm_path=vllm_path,
        expert_dir=expert_dir,
        backend_type=BackendType(backend) if backend != "auto" else BackendType.AUTO,
        **kwargs,
    )
    return UnifiedBackendAdapter(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("  CGC Engine 统一后端调度器")
    print("=" * 70)

    config = BackendConfig(
        expert_dir="/Users/alexchuang/Documents/cgcjitload/flashkv0430/MagiCompiler-main/expert_weights",
        backend_type=BackendType.FLASHMOE,
    )

    engine = UnifiedBackendAdapter(config)

    print(f"\n活跃后端: {engine.get_active_backend()}")
    print(f"可用后端: {engine.get_available_backends()}")

    print("\n测试生成...")
    result = engine.generate("Hello")
    print(f"结果: {result}")
