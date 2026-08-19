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
CGC Engine - MagiCompiler 統一入口

🔥 唯一對外接口：一行編譯 PyTorch 模型
推理/訓練通用，支援 llama.cpp GGUF

使用方式：
    from cgc_engine import CGCEngine

    # 方案 1: llama.cpp GGUF 模式
    engine = CGCEngine(gguf_path="/path/to/model.gguf")
    result = engine.generate("Hello world")

    # 方案 2: PyTorch 模型模式
    engine = CGCEngine(model=my_model)
    result = engine.generate(input_ids)

    # 方案 3: MagiCompiler 編譯模式
    engine = CGCEngine.compile(my_model, device="cuda")
    result = engine(x)
"""

import torch
import torch.nn as nn
from typing import Union, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

from .cgc.cgc_runtime import (
    CGCRuntime,
    ModelConfig,
    CGCModel,
    create_cgc_runtime,
)
from .cgc.torch_cpp_bridge import torch_to_cgc, cgc_execute_command, CGCOp, CGCCommand
from .agent import HarnessAgent, HarnessCompileStrategy, StrategyExecutor

__version__ = "2.1.0"
__all__ = ["CGCEngine", "CGCEngineConfig", "compile", "run_cgc_with_kda"]


@dataclass
class CGCEngineConfig:
    """CGC Engine 配置"""
    model_name_or_path: Optional[str] = None
    gguf_path: Optional[str] = None
    vllm_path: Optional[str] = None
    model_config: Optional[ModelConfig] = None
    device: Optional[str] = None
    enable_llama_cpp: bool = False
    enable_vllm: bool = False
    enable_moe: bool = True
    enable_gds: bool = True
    max_tokens: int = 100
    temperature: float = 0.0
    top_k: int = 50
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9


class CGCEngine:
    """
    CGC Engine - MagiCompiler 統一入口

    支援：
    - llama.cpp GGUF 模型
    - vLLM (CUDA) 模型
    - PyTorch 原生模型
    - CGC SIMD 命令執行
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        config: Optional[CGCEngineConfig] = None,
        model_name_or_path: Optional[str] = None,
        gguf_path: Optional[str] = None,
        vllm_path: Optional[str] = None,
        device: Optional[str] = None,
        enable_vllm: bool = False,
        enable_llama_cpp: bool = False,
        **kwargs,
    ):
        """
        初始化 CGC Engine

        Args:
            model: PyTorch 模型（可選）
            config: CGCEngineConfig 配置
            model_name_or_path: HuggingFace 模型路徑或名稱
            gguf_path: GGUF 模型路徑
            vllm_path: vLLM 模型路徑或名稱
            device: 設備
            enable_vllm: 是否啟用 vLLM (需要 CUDA)
            enable_llama_cpp: 是否啟用 llama.cpp
            **kwargs: 其他參數
        """
        self.config = config or CGCEngineConfig()
        if model_name_or_path:
            self.config.model_name_or_path = model_name_or_path
        if gguf_path:
            self.config.gguf_path = gguf_path
        if vllm_path:
            self.config.vllm_path = vllm_path
        if device:
            self.config.device = device

        self._device = self._detect_device()
        self._vllm_model = None
        self._vllm_tokenizer = None

        if enable_vllm or self.config.enable_vllm:
            self._runtime = self._init_vllm_mode()
        elif self.config.gguf_path or enable_llama_cpp:
            self._runtime = self._init_llama_cpp_mode()
        elif model is not None:
            self._model = model
            self._runtime = None
            self._use_cgc_commands = True
        else:
            self._model = None
            self._runtime = CGCRuntime(
                model_config=self.config.model_config,
                device=self._device,
            )
            self._use_cgc_commands = False

        logger.info(f"[CGCEngine] Initialized: device={self._device}, mode={self._get_mode()}")

    def _get_mode(self) -> str:
        """獲取當前模式"""
        if self._vllm_model is not None:
            return "vLLM"
        if self._runtime is not None and self._runtime.is_llama_cpp_mode():
            return "llama.cpp"
        if self._model is not None:
            return "PyTorch"
        return "CGC"

    def _detect_device(self) -> str:
        """自動檢測設備"""
        if self.config.device:
            return self.config.device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _init_llama_cpp_mode(self) -> CGCRuntime:
        """初始化 llama.cpp GGUF 模式"""
        return CGCRuntime(
            model_config=None,
            device=self._device,
            enable_llama_cpp_bridge=True,
            gguf_path=self.config.gguf_path,
        )

    def _init_vllm_mode(self) -> None:
        """初始化 vLLM 模式 (CUDA)"""
        if not torch.cuda.is_available():
            raise RuntimeError("vLLM requires CUDA GPU")

        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer

            model_path = self.config.vllm_path or self.config.model_name_or_path
            if not model_path:
                raise ValueError("vllm_path or model_name_or_path is required for vLLM mode")

            logger.info(f"[CGCEngine] Loading vLLM model: {model_path}")

            tensor_parallel = self.config.tensor_parallel_size
            gpu_memory_utilization = self.config.gpu_memory_utilization

            self._vllm_model = LLM(
                model=model_path,
                trust_remote_code=True,
                tensor_parallel_size=tensor_parallel,
                gpu_memory_utilization=gpu_memory_utilization,
                dtype="float16",
            )

            try:
                self._vllm_tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                )
            except:
                self._vllm_tokenizer = None

            logger.info(f"[CGCEngine] vLLM model loaded successfully")

        except ImportError:
            logger.error("[CGCEngine] vLLM not installed. Install with: pip install vllm")
            raise
        except Exception as e:
            logger.error(f"[CGCEngine] Failed to load vLLM model: {e}")
            raise

    @classmethod
    def from_gguf(
        cls,
        gguf_path: str,
        device: Optional[str] = None,
        **kwargs,
    ) -> "CGCEngine":
        """
        從 GGUF 文件創建 Engine

        Args:
            gguf_path: GGUF 模型路徑
            device: 設備
            **kwargs: 其他參數

        Returns:
            CGCEngine 實例
        """
        return cls(gguf_path=gguf_path, device=device, **kwargs)

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        device: Optional[str] = None,
        **kwargs,
    ) -> "CGCEngine":
        """
        從 PyTorch 模型創建 Engine

        Args:
            model: PyTorch 模型
            device: 設備
            **kwargs: 其他參數

        Returns:
            CGCEngine 實例
        """
        return cls(model=model, device=device, **kwargs)

    @classmethod
    def from_vllm(
        cls,
        model_name_or_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        **kwargs,
    ) -> "CGCEngine":
        """
        從 HuggingFace 模型創建 vLLM Engine (CUDA)

        Args:
            model_name_or_path: 模型名稱或路徑
            tensor_parallel_size: Tensor parallel 大小
            gpu_memory_utilization: GPU 記憶體利用率
            **kwargs: 其他參數

        Returns:
            CGCEngine 實例 (vLLM 模式)
        """
        config = CGCEngineConfig(
            model_name_or_path=model_name_or_path,
            enable_vllm=True,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        return cls(config=config, **kwargs)

    @classmethod
    def compile(
        cls,
        model: nn.Module,
        device: str = "cuda",
        enable_moe: bool = True,
        enable_gds: bool = True,
        enable_agent: bool = True,
        agent_config: Optional[Dict[str, Any]] = None,
    ) -> "CGCEngine":
        """
        MagiCompiler 編譯模式：將 PyTorch 模型編譯為 CGC SIMD

        Args:
            model: PyTorch 模型
            device: 設備
            enable_moe: 是否啟用 MoE 優化
            enable_gds: 是否啟用 GDS
            enable_agent: 是否啟用 Harness Agent 策略優化
            agent_config: Harness Agent 配置（可選）

        Returns:
            CGCEngine 實例（使用 CGC 命令執行）
        """
        engine = cls(model=model, device=device)
        engine._use_cgc_commands = True

        if enable_agent:
            logger.info("[CGCEngine] Initializing Harness Agent...")
            agent_config = agent_config or {}
            engine._harness_agent = HarnessAgent(device=device, **agent_config)
            engine._strategy_executor = StrategyExecutor()
            engine._compile_strategy = None

            if hasattr(model, "__call__"):
                try:
                    # 尝试根据默认输入形状生成策略
                    input_shape = (1, 64)
                    engine._compile_strategy = engine._harness_agent.decide(
                        model=model,
                        input_shape=input_shape,
                    )
                    logger.info("[CGCEngine] Harness Agent strategy decided successfully")
                except Exception as e:
                    logger.warning(f"[CGCEngine] Failed to decide strategy: {e}")
        else:
            engine._harness_agent = None
            engine._strategy_executor = None

        return engine

    @torch.no_grad()
    def generate(
        self,
        input_ids: Union[torch.Tensor, str, List[int]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> Union[List[int], str, Dict]:
        """
        生成文字/tokens

        Args:
            input_ids: 輸入
            max_tokens: 最大生成長度
            temperature: 溫度
            top_k: Top-K
            **kwargs: 其他參數

        Returns:
            生成結果
        """
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k

        if self._vllm_model is not None:
            return self._generate_vllm(
                input_ids,
                max_tokens,
                temperature,
                top_k,
                **kwargs,
            )

        if self._runtime is not None:
            return self._runtime.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                **kwargs,
            )

        if self._model is not None:
            if self._use_cgc_commands:
                return self._generate_with_cgc_commands(
                    input_ids, max_tokens, temperature, top_k
                )
            else:
                return self._generate_direct(
                    input_ids, max_tokens, temperature, top_k
                )

        raise RuntimeError("No model or runtime available")

    def _generate_vllm(
        self,
        input_ids: Union[torch.Tensor, str, List[int]],
        max_tokens: int,
        temperature: float,
        top_k: int,
        **kwargs,
    ) -> Dict:
        """使用 vLLM 生成"""
        from vllm import SamplingParams

        if isinstance(input_ids, torch.Tensor):
            if input_ids.dim() > 1:
                input_ids = input_ids[0].tolist()

        if isinstance(input_ids, list):
            if self._vllm_tokenizer:
                text = self._vllm_tokenizer.decode(input_ids)
            else:
                text = "".join(chr(t) if t < 256 else f"<{t}>" for t in input_ids)
        else:
            text = str(input_ids)

        stop = kwargs.get("stop", None)

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 0.7,
            top_k=top_k if top_k > 0 else -1,
            stop=stop,
            **kwargs,
        )

        outputs = self._vllm_model.generate([text], sampling_params)

        result_text = outputs[0].outputs[0].text

        return {
            "text": result_text,
            "generated_text": result_text,
            "usage": {
                "prompt_eval_count": outputs[0].outputs[0].cumulative_logprob,
                "eval_count": len(result_text),
            },
        }

    def _generate_with_cgc_commands(
        self,
        input_ids: Union[torch.Tensor, str, List[int]],
        max_tokens: int,
        temperature: float,
        top_k: int,
    ) -> List[int]:
        """使用 CGC 命令生成"""
        if isinstance(input_ids, str):
            input_ids = self._tokenize(input_ids)

        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

        generated = input_ids[0].tolist()

        for _ in range(max_tokens):
            inputs = (input_ids,)
            commands = torch_to_cgc(self._model, inputs)

            x = input_ids
            for cmd in commands:
                x = cgc_execute_command(cmd)

            if isinstance(x, tuple):
                x = x[0]

            logits = x[0, -1, :]

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = torch.argmax(logits).item()

            generated.append(next_token)
            input_ids = torch.tensor([[next_token]], dtype=torch.long)

        return generated

    def _generate_direct(
        self,
        input_ids: Union[torch.Tensor, str, List[int]],
        max_tokens: int,
        temperature: float,
        top_k: int,
    ) -> List[int]:
        """直接 PyTorch 生成"""
        if isinstance(input_ids, str):
            input_ids = self._tokenize(input_ids)

        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

        generated = input_ids[0].tolist()

        for _ in range(max_tokens):
            logits = self._model(input_ids)
            next_logits = logits[0, -1, :]

            if temperature > 0:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                if top_k > 0:
                    v, i = torch.topk(next_logits, top_k)
                    next_logits = torch.full_like(next_logits, float('-inf'))
                    next_logits[i] = v
                next_token = torch.argmax(next_logits).item()

            generated.append(next_token)
            input_ids = torch.tensor([[next_token]], dtype=torch.long)

        return generated

    def _tokenize(self, text: str) -> List[int]:
        """簡單分詞"""
        return [ord(c) for c in text]

    def __call__(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        前向傳播（訓練/推理）

        Args:
            input_ids: 輸入 token IDs
            **kwargs: 其他參數

        Returns:
            Logits
        """
        if self._use_cgc_commands and self._model is not None:
            commands = torch_to_cgc(self._model, (input_ids,))
            outputs = []
            x = input_ids
            for cmd in commands:
                x = cgc_execute_command(cmd)
                outputs.append(x)
            return x

        if self._model is not None:
            return self._model(input_ids, **kwargs)

        if self._runtime is not None:
            return self._runtime.forward(input_ids)

        raise RuntimeError("No model available")


def compile(
    model: nn.Module,
    device: str = "cuda",
    enable_moe: bool = True,
    enable_gds: bool = True,
) -> CGCEngine:
    """
    MagiCompiler compile - 便捷函數

    Args:
        model: PyTorch 模型
        device: 設備
        enable_moe: 是否啟用 MoE
        enable_gds: 是否啟用 GDS

    Returns:
        CGCEngine 實例
    """
    return CGCEngine.compile(model, device, enable_moe, enable_gds)


def run_cgc_with_kda(model: nn.Module, device: str = "metal") -> torch.Tensor:
    """
    🔥 CGC Engine + Agent + Metal + KDA 完整運行

    最終調用鏈（架構靈魂）：
        Harness Agent 決策策略
                ↓
        CompileStrategy (fusion/tiling/memory/scheduling)
                ↓
        CGCStrategyInjector 注入 C++
                ↓
        C++ SIMD Engine 接收策略
                ↓
        Metal Backend 執行 KDA SIMD Shader

    支持的硬件後端（Harness Agent 自動選擇）：
        ✅ CPU
        ✅ CUDA
        ✅ Metal (Apple GPU)

    支持的算子（SIMD 引擎 + C++ + Metal 加速）：
        ✅ KDA (Kernel Delta Attention)
        ✅ GEMM
        ✅ FlashAttention
        ✅ ROPE
        ✅ MLP fusion

    Args:
        model: PyTorch 模型
        device: 設備

    Returns:
        模型輸出
    """
    from .magicompiler_integration import MagiCompiler
    from .agent.harness_agent import HarnessAgent
    from .agent.strategy_executor import StrategyExecutor
    from .hardware.hardware_constraints import HardwareConstraints
    from .cgc.cgc_strategy_injection import CGCStrategyInjector

    logger.info("=" * 60)
    logger.info("🔥 CGC Engine + Agent + Metal + KDA 完整運行")
    logger.info("=" * 60)

    mgc = MagiCompiler(model)
    graph = mgc.capture_full_graph()
    logger.info(f"✅ Step 1: 計算圖捕獲完成，設備: {device}")

    hw = HardwareConstraints(device=device)
    logger.info(f"✅ Step 2: 硬件約束加載完成: {hw}")

    agent = HarnessAgent(device=device)
    strategy = agent.decide(model, (1, 128), graph_features=graph)
    logger.info("✅ Step 3: Harness Agent 策略決策完成")
    logger.info(f"   - fusion_boundary: {strategy.fusion_regions}")
    logger.info(f"   - tiling_config: {strategy.tile_sizes}")
    logger.info(f"   - enable_op_fusion: {strategy.enable_op_fusion}")

    injector = CGCStrategyInjector()
    injector.inject_strategy(strategy)
    logger.info("✅ Step 4: 策略注入 C++ SIMD Engine 完成")

    mgc.set_fusion_boundary(strategy.fusion_regions)
    mgc.set_tiling_config(strategy.tile_sizes)
    mgc.set_memory_hierarchy(strategy.memory_layouts)
    mgc.set_scheduling_plan(strategy.schedules)
    mgc.set_backend(strategy.backend)
    for hint in strategy.op_hints:
        mgc.apply_op_hint(hint)
    logger.info("✅ Step 5: 策略注入 MagiCompiler 完成")

    compiled_model = mgc.compile()
    logger.info("✅ Step 6: 模型編譯完成")

    batch_size = 1
    seq_len = 128
    hidden_dim = 128

    if hasattr(model, 'd_model'):
        hidden_dim = model.d_model
    elif hasattr(model, 'hidden_dim'):
        hidden_dim = model.hidden_dim

    x = torch.randn(batch_size, seq_len, hidden_dim)
    logger.info(f"▶️  Step 7: 執行編譯後模型，輸入 shape: {x.shape}")

    out = compiled_model(x)
    logger.info(f"✅ Step 7: 執行完成，輸出 shape: {out.shape}")

    logger.info("=" * 60)
    logger.info("🎉 CGC Engine + Agent + Metal + KDA 完整運行成功！")
    logger.info("=" * 60)

    return out
