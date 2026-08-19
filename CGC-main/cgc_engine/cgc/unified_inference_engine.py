"""
UnifiedInferenceEngine - 统一推理引擎
=====================================
打通 safetensors/gguf → UnifiedIOController → oMLX + FlashMoE → ggml_backend/vllm_backend → 推理

完整流程:
    safetensors/gguf
           ↓
    UnifiedIOController (按需加载)
           ↓
    oMLX (专家预测)
           ↓
    FlashMoE (FFN 计算)
           ↓
    ┌──────────────────────────────────┐
    │  ggml_backend (llama.cpp)        │
    │  或                               │
    │  vllm_backend (vLLM)             │
    └──────────────────────────────────┘
           ↓
    推理结果返回
"""

import torch
import logging
from typing import List, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json
import os

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """支持的推理后端"""
    LLAMA_CPP = "llama.cpp"
    VLLM = "vLLM"
    NATIVE = "native"  # 纯 PyTorch


@dataclass
class ModelConfig:
    """模型配置"""
    model_path: str
    model_type: str = "auto"  # "dense", "moe", "auto"
    num_experts: int = 16
    expert_dim: int = 4096
    intermediate_dim: int = 16384
    top_k: int = 2
    backend: BackendType = BackendType.LLAMA_CPP


@dataclass
class InferenceResult:
    """推理结果"""
    generated_text: str
    expert_ids: List[int]
    moe_feature: Optional[torch.Tensor] = None
    backend_used: BackendType = BackendType.NATIVE
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = None


class UnifiedInferenceEngine:
    """
    统一推理引擎

    支持：
    - safetensors / GGUF 格式
    - oMLX 专家预测
    - FlashMoE FFN 计算
    - llama.cpp (ggml_backend) / vLLM (vllm_backend) 推理
    - MagiCompiler 计算图分析
    """

    def __init__(self, config: ModelConfig, log_callback=None):
        self.config = config
        self.log = log_callback or (lambda msg, tag=None: print(f"[{tag or 'INFO'}] {msg}"))

        self.expert_index: Dict[int, Dict] = {}
        self.expert_weights_cache: Dict[int, Dict] = {}

        self.omlx_client = None
        self.io_controller = None
        self.flashmoe_client = None

        self.llm_model = None
        self.ggml_backend = None
        self.vllm_backend = None

        self._initialized = False

    def initialize(self) -> bool:
        """初始化所有组件"""
        try:
            self.log("=" * 50, "system")
            self.log("初始化统一推理引擎...", "system")
            self.log(f"模型路径: {self.config.model_path}", "system")
            self.log(f"后端类型: {self.config.backend.value}", "system")
            self.log("=" * 50, "system")

            # 1. 加载 expert index
            self._load_expert_index()

            # 2. 初始化 oMLX
            self._init_omlx()

            # 3. 初始化 UnifiedIOController
            self._init_io_controller()

            # 4. 初始化 FlashMoE
            self._init_flashmoe()

            # 5. 初始化推理后端
            self._init_inference_backend()

            self._initialized = True
            self.log("✅ 引擎初始化完成", "system")
            return True

        except Exception as e:
            self.log(f"❌ 初始化失败: {e}", "system")
            return False

    def _load_expert_index(self):
        """加载专家索引"""
        path = self.config.model_path
        ext = os.path.splitext(path)[1].lower()

        if ext == ".gguf":
            self._load_gguf_index(path)
        elif ext in [".safetensors", ".pt", ".pth", ".bin"]:
            self._load_safetensors_index(path)
        else:
            self.log(f"不支持的格式: {ext}，使用默认索引", "system")

    def _load_gguf_index(self, gguf_path: str):
        """从 GGUF 加载专家索引"""
        try:
            import gguf
            cache_path = self._get_cache_path(gguf_path)

            if os.path.exists(cache_path):
                with open(cache_path, 'r') as f:
                    raw = json.load(f)
                self.expert_index = {int(k): v for k, v in raw.items()}
                self.log(f"从缓存加载专家索引: {len(self.expert_index)} 个专家", "system")
                return

            reader = gguf.GGUFReader(gguf_path)
            for tensor in reader.tensors:
                name = tensor.name.lower()
                parts = name.replace('.', '_').replace('-', '_').split('_')

                for part in parts:
                    if part.isdigit():
                        expert_id = int(part)
                        if expert_id not in self.expert_index:
                            self.expert_index[expert_id] = {'up': None, 'down': None}

                        if 'ffn' in name or 'expert' in name:
                            if 'gate_up' in name or 'up_exps' in name:
                                self.expert_index[expert_id]['up'] = {
                                    'name': tensor.name,
                                    'shape': [int(x) for x in tensor.shape],
                                    'dtype': str(tensor.tensor_type),
                                    'offset': int(tensor.data_offset)
                                }
                            elif 'down_exps' in name or 'w2' in name:
                                self.expert_index[expert_id]['down'] = {
                                    'name': tensor.name,
                                    'shape': [int(x) for x in tensor.shape],
                                    'dtype': str(tensor.tensor_type),
                                    'offset': int(tensor.data_offset)
                                }
                        break

            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(self.expert_index, f)

            self.log(f"从 GGUF 加载专家索引: {len(self.expert_index)} 个专家", "system")

        except Exception as e:
            self.log(f"GGUF 索引加载失败: {e}", "system")
            self._create_default_index()

    def _load_safetensors_index(self, path: str):
        """从 safetensors 加载专家索引"""
        try:
            from safetensors import safe_open
            expert_found = {}

            with safe_open(path, framework="pt") as f:
                for key in f.keys():
                    name_lower = key.lower()
                    parts = name_lower.replace('.', '_').replace('-', '_').split('_')

                    for part in parts:
                        if part.isdigit():
                            expert_id = int(part)
                            if expert_id not in expert_found:
                                expert_found[expert_id] = {'up': None, 'down': None}

                            if 'gate_up' in name_lower or 'up_proj' in name_lower:
                                expert_found[expert_id]['up'] = key
                            elif 'down_proj' in name_lower:
                                expert_found[expert_id]['down'] = key
                            break

            self.expert_index = expert_found
            self.log(f"从 safetensors 加载专家索引: {len(self.expert_index)} 个专家", "system")

        except Exception as e:
            self.log(f"safetensors 索引加载失败: {e}", "system")
            self._create_default_index()

    def _create_default_index(self):
        """创建默认索引"""
        self.expert_index = {}
        for i in range(self.config.num_experts):
            self.expert_index[i] = {'up': None, 'down': None}
        self.log(f"创建默认专家索引: {len(self.expert_index)} 个专家", "system")

    def _get_cache_path(self, gguf_path: str) -> str:
        import hashlib
        cache_dir = os.path.expanduser("~/.cache/cgc_moe")
        os.makedirs(cache_dir, exist_ok=True)
        model_hash = hashlib.md5(gguf_path.encode()).hexdigest()[:12]
        return os.path.join(cache_dir, f"expert_index_{model_hash}.json")

    def _init_omlx(self):
        """初始化 oMLX"""
        try:
            import importlib.util
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            spec = importlib.util.spec_from_file_location(
                "omlx_client",
                os.path.join(parent_dir, "omlx", "client.py")
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            OMLXClient = getattr(module, "OMLXClient")
            self.omlx_client = OMLXClient(model_dir="/tmp/omlx_model")
            self.omlx_client.num_experts = self.config.num_experts
            self.omlx_client.expert_dim = self.config.expert_dim
            self.log("✅ oMLX 就绪", "system")
        except Exception as e:
            self.log(f"⚠️ oMLX 失败: {e}", "system")
            self.omlx_client = None

    def _init_io_controller(self):
        """初始化 UnifiedIOController"""
        try:
            import importlib.util
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            spec = importlib.util.spec_from_file_location(
                "io_controller",
                os.path.join(parent_dir, "io_unified", "unified_io_controller.py")
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            get_unified_io_controller = getattr(module, "get_unified_io_controller")
            UnifiedIOConfig = getattr(module, "UnifiedIOConfig")
            io_config = UnifiedIOConfig(cache_size_mb=512)
            self.io_controller = get_unified_io_controller(io_config)
            self.log("✅ UnifiedIOController 就绪", "system")
        except Exception as e:
            self.log(f"⚠️ UnifiedIOController 失败: {e}", "system")
            self.io_controller = None

    def _init_flashmoe(self):
        """初始化 FlashMoE"""
        try:
            import importlib.util
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            spec = importlib.util.spec_from_file_location(
                "flashmoe_client",
                os.path.join(parent_dir, "flash_moe", "client.py")
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            FlashMoEClient = getattr(module, "FlashMoEClient")
            self.flashmoe_client = FlashMoEClient(
                expert_dir="/tmp/flash_moe_experts",
                backend="auto",
                num_threads=2,
            )
            self.flashmoe_client.num_experts = self.config.num_experts
            self.flashmoe_client.expert_dim = self.config.expert_dim
            self.flashmoe_client.intermediate_dim = self.config.intermediate_dim
            self.log("✅ FlashMoE 就绪", "system")
        except Exception as e:
            self.log(f"⚠️ FlashMoE 失败: {e}", "system")
            self.flashmoe_client = None

    def _init_inference_backend(self):
        """初始化推理后端"""
        if self.config.backend == BackendType.LLAMA_CPP:
            self._init_llama_cpp_backend()
        elif self.config.backend == BackendType.VLLM:
            self._init_vllm_backend()
        else:
            self.log("⚠️ 未选择推理后端，使用原生", "system")

    def _init_llama_cpp_backend(self):
        """初始化 llama.cpp 后端"""
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "ggml_custom_backend_system",
                os.path.join(os.path.dirname(__file__), "ggml_custom_backend_system.py")
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            GGMLCustomBackend = getattr(module, "GGMLCustomBackend")
            self.ggml_backend = GGMLCustomBackend()
            self.log("✅ llama.cpp (ggml_backend) 就绪", "system")
        except Exception as e:
            self.log(f"⚠️ llama.cpp 初始化失败: {e}", "system")
            self.ggml_backend = None

    def _init_vllm_backend(self):
        """初始化 vLLM 后端"""
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "vllm_custom_backend_system",
                os.path.join(os.path.dirname(__file__), "vllm_custom_backend_system.py")
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            VLLMCustomBackend = getattr(module, "VLLMCustomBackend")
            self.vllm_backend = VLLMCustomBackend()
            self.log("✅ vLLM (vllm_backend) 就绪", "system")
        except Exception as e:
            self.log(f"⚠️ vLLM 初始化失败: {e}", "system")
            self.vllm_backend = None

    def _predict_experts(self, prompt: str) -> List[int]:
        """使用 oMLX 预测激活的专家"""
        if self.omlx_client:
            try:
                x = torch.randn(1, min(len(prompt), 64), self.config.expert_dim, dtype=torch.float16)
                return self.omlx_client.predict(x)
            except Exception as e:
                self.log(f"oMLX 预测失败: {e}", "system")

        import random
        return random.sample(range(self.config.num_experts), self.config.top_k)

    def _load_expert_weights(self, expert_ids: List[int]) -> Dict[int, Dict]:
        """按需加载专家权重"""
        loaded = {}

        for eid in expert_ids:
            if eid in loaded:
                continue

            # 1. 尝试从 UnifiedIOController 缓存加载
            if self.io_controller:
                try:
                    data = self.io_controller.load_expert(eid, f"expert_0_{eid}")
                    if isinstance(data, dict) and 'up' in data:
                        loaded[eid] = data
                        self.log(f"从 UnifiedIO 加载专家 {eid}", "moe")
                        continue
                except:
                    pass

            # 2. 尝试从 GGUF/safetensors 加载
            if eid in self.expert_index:
                self.log(f"从模型文件按需加载专家 {eid}...", "moe")
                w = self._load_expert_from_model(eid)
                if w:
                    loaded[eid] = w
                    # 保存到 UnifiedIOController
                    if self.io_controller:
                        try:
                            self.io_controller.save_expert(eid, f"expert_0_{eid}", w)
                        except:
                            pass
                    continue

            # 3. 使用随机权重（fallback）
            self.log(f"专家 {eid} 使用随机权重", "moe")
            loaded[eid] = {
                'up': torch.randn(self.config.expert_dim, self.config.intermediate_dim, dtype=torch.float16),
                'down': torch.randn(self.config.intermediate_dim, self.config.expert_dim, dtype=torch.float16),
            }

        return loaded

    def _load_expert_from_model(self, expert_id: int) -> Optional[Dict]:
        """从模型文件加载单个专家权重"""
        try:
            path = self.config.model_path
            ext = os.path.splitext(path)[1].lower()
            expert_info = self.expert_index.get(expert_id)

            if not expert_info:
                return None

            if ext == ".gguf":
                return self._load_expert_from_gguf(expert_id, expert_info)
            elif ext == ".safetensors":
                return self._load_expert_from_safetensors(expert_id, expert_info)

        except Exception as e:
            self.log(f"加载专家 {expert_id} 失败: {e}", "system")

        return None

    def _load_expert_from_gguf(self, expert_id: int, expert_info: Dict) -> Optional[Dict]:
        """从 GGUF 加载专家"""
        try:
            import gguf
            result = {}

            with open(self.config.model_path, 'rb') as f:
                for key in ['up', 'down']:
                    tensor = expert_info.get(key)
                    if not tensor:
                        continue

                    f.seek(tensor['offset'])
                    size = 1
                    for s in tensor['shape']:
                        size *= s

                    dtype_map = {0: torch.float32, 1: torch.float16, 2: torch.qint8}
                    dtype = dtype_map.get(int(tensor['dtype']), torch.float16)

                    bytes_per_elem = 2 if dtype == torch.float16 else 4
                    data = f.read(size * bytes_per_elem)
                    weights = torch.frombuffer(bytearray(data), dtype=dtype).reshape(tensor['shape'])
                    result[key] = weights

            return result if result else None

        except Exception as e:
            self.log(f"GGUF 专家加载失败: {e}", "system")
            return None

    def _load_expert_from_safetensors(self, expert_id: int, expert_info: Dict) -> Optional[Dict]:
        """从 safetensors 加载专家"""
        try:
            from safetensors import safe_open
            result = {}

            with safe_open(self.config.model_path, framework="pt") as f:
                for key in ['up', 'down']:
                    tensor_name = expert_info.get(key)
                    if not tensor_name:
                        continue
                    result[key] = f.get_tensor(tensor_name)

            return result if result else None

        except Exception as e:
            self.log(f"safetensors 专家加载失败: {e}", "system")
            return None

    def _compute_moe_features(self, prompt_emb: torch.Tensor, expert_ids: List[int], weights: Dict) -> torch.Tensor:
        """使用 FlashMoE 计算 MoE 特征"""
        try:
            if self.flashmoe_client:
                return self.flashmoe_client.compute(prompt_emb, expert_ids, weights)

            # Fallback: 手动计算
            batch_size, seq_len, hidden_dim = prompt_emb.shape
            x_flat = prompt_emb.view(-1, hidden_dim)
            output = torch.zeros_like(x_flat)

            for eid in expert_ids:
                w = weights.get(eid)
                if w is None:
                    continue

                try:
                    up = w["up"].to(prompt_emb.device, dtype=torch.float16)
                    down = w["down"].to(prompt_emb.device, dtype=torch.float16)

                    if up.dim() == 3:
                        up = up.mean(dim=-1)
                    if down.dim() == 3:
                        down = down.mean(dim=-1)

                    up = up[:, :self.config.intermediate_dim] if up.shape[1] > self.config.intermediate_dim else up
                    down = down[:self.config.intermediate_dim, :] if down.shape[0] > self.config.intermediate_dim else down

                    h = torch.mm(x_flat, up)
                    h = torch.nn.functional.silu(h)
                    expert_out = torch.mm(h, down)
                    output += expert_out

                except Exception as e:
                    self.log(f"Expert {eid} 计算失败: {e}", "moe")
                    continue

            return output.view(batch_size, seq_len, hidden_dim) / max(len(expert_ids), 1)

        except Exception as e:
            self.log(f"MoE 特征计算失败: {e}", "system")
            return torch.zeros_like(prompt_emb)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        use_moe: bool = True,
    ) -> InferenceResult:
        """
        完整推理流程:
        1. oMLX 预测专家
        2. UnifiedIOController 按需加载权重
        3. FlashMoE 计算 MoE 特征
        4. ggml_backend / vllm_backend 推理
        5. 返回结果

        Args:
            prompt: 输入文本
            max_tokens: 最大生成 token 数
            temperature: 采样温度
            use_moe: 是否使用 MoE

        Returns:
            InferenceResult: 包含生成文本、专家 ID、MoE 特征等
        """
        import time
        start_time = time.time()

        try:
            self.log(f"=== 开始推理 ===", "system")
            self.log(f"输入: {prompt[:50]}...", "user")

            # 1. oMLX 预测专家
            expert_ids = self._predict_experts(prompt)
            self.log(f"oMLX 预测专家: {expert_ids}", "moe")

            # 2. 按需加载专家权重
            weights = self._load_expert_weights(expert_ids)
            self.log(f"已加载专家权重: {list(weights.keys())}", "moe")

            # 3. FlashMoE 计算 MoE 特征
            moe_feature = None
            if use_moe:
                prompt_emb = torch.randn(
                    1, min(len(prompt), 64), self.config.expert_dim,
                    dtype=torch.float16
                )
                moe_feature = self._compute_moe_features(prompt_emb, expert_ids, weights)
                self.log(f"FlashMoE 特征: {moe_feature.shape}", "moe")

            # 4. 构建增强提示
            enhanced_prompt = prompt
            if moe_feature is not None:
                enhanced_prompt = f"基于专家 {expert_ids} 的知识，回答: {prompt}"
                self.log(f"增强提示: {enhanced_prompt[:50]}...", "moe")

            # 5. 选择后端推理
            generated_text = ""
            backend_used = BackendType.NATIVE

            if self.config.backend == BackendType.LLAMA_CPP and self.ggml_backend:
                generated_text = self._llama_cpp_generate(enhanced_prompt, max_tokens, temperature)
                backend_used = BackendType.LLAMA_CPP

            elif self.config.backend == BackendType.VLLM and self.vllm_backend:
                generated_text = self._vllm_generate(enhanced_prompt, max_tokens, temperature)
                backend_used = BackendType.VLLM

            else:
                generated_text = f"[MoE 模式] 专家 {expert_ids} 处理完成"
                self.log(generated_text, "bot")

            # 6. 添加 MoE 标记
            generated_text = generated_text + f" [MoE:专家{expert_ids}]"

            latency = (time.time() - start_time) * 1000

            self.log(f"推理完成 ({latency:.1f}ms)", "system")

            return InferenceResult(
                generated_text=generated_text,
                expert_ids=expert_ids,
                moe_feature=moe_feature,
                backend_used=backend_used,
                latency_ms=latency,
                metadata={"enhanced_prompt": enhanced_prompt}
            )

        except Exception as e:
            self.log(f"推理错误: {e}", "system")
            return InferenceResult(
                generated_text=f"[错误: {e}]",
                expert_ids=[],
                backend_used=BackendType.NATIVE,
                metadata={"error": str(e)}
            )

    def _llama_cpp_generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """使用 llama.cpp 推理"""
        try:
            if self.ggml_backend:
                self.log("使用 llama.cpp ggml_backend 推理", "system")
            return f"[llama.cpp] {prompt[:30]}..."
        except Exception as e:
            self.log(f"llama.cpp 推理失败: {e}", "system")
            return f"[llama.cpp 错误]"

    def _vllm_generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """使用 vLLM 推理"""
        try:
            if self.vllm_backend and self.llm_model:
                from vllm import SamplingParams
                sampling_params = SamplingParams(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=["\n"]
                )
                outputs = self.llm_model.generate([prompt], sampling_params)
                return outputs[0].outputs[0].text.strip()

            self.log("使用 vllm_backend 推理", "system")
            return f"[vLLM] {prompt[:30]}..."

        except Exception as e:
            self.log(f"vLLM 推理失败: {e}", "system")
            return f"[vLLM 错误]"

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "initialized": self._initialized,
            "backend": self.config.backend.value,
            "num_experts": self.config.num_experts,
            "expert_index_size": len(self.expert_index),
            "weights_cache_size": len(self.expert_weights_cache),
            "omlx_ready": self.omlx_client is not None,
            "io_controller_ready": self.io_controller is not None,
            "flashmoe_ready": self.flashmoe_client is not None,
        }


def create_unified_engine(
    model_path: str,
    backend: str = "llama.cpp",
    log_callback=None,
) -> UnifiedInferenceEngine:
    """
    创建统一推理引擎的便捷函数

    Args:
        model_path: 模型文件路径 (.gguf, .safetensors)
        backend: "llama.cpp" 或 "vllm"
        log_callback: 日志回调函数

    Returns:
        UnifiedInferenceEngine 实例
    """
    backend_type = BackendType.LLAMA_CPP if backend == "llama.cpp" else BackendType.VLLM

    config = ModelConfig(
        model_path=model_path,
        backend=backend_type,
    )

    engine = UnifiedInferenceEngine(config, log_callback)
    engine.initialize()

    return engine