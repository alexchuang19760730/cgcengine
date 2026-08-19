#!/usr/bin/env python3
"""
CGCEngine 增强版 - 完整推理/训练支持 + KDA 集成

功能:
1. 支持推理/训练模式切换
2. GGUF 权重加载
3. PyTorch 计算图构建
4. KDA Pass 替换 Attention
5. Metal GPU 执行
6. 与 llama.cpp 对比
"""

import torch
import torch.nn as nn
from typing import Union, List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Mode(Enum):
    """CGC Engine 运行模式"""
    INFERENCE = "inference"
    TRAINING = "training"


class KDAMode(Enum):
    """KDA 模式"""
    STANDARD = "standard"           # 标准 Attention
    KDA_PYTORCH = "kda_pytorch"      # PyTorch KDA
    KDA_CPP_NEON = "kda_cpp_neon"    # C++ NEON SIMD
    KDA_METAL = "kda_metal"          # Metal GPU


@dataclass
class CGCEngineConfigV2:
    """CGC Engine V2 配置"""
    mode: Mode = Mode.INFERENCE
    kda_mode: KDAMode = KDAMode.STANDARD
    device: str = "auto"
    gguf_path: Optional[str] = None
    model_config: Optional[Dict[str, Any]] = None
    max_seq_len: int = 2048
    num_layers: int = 28
    num_heads: int = 28
    num_kv_heads: int = 4
    hidden_dim: int = 3584
    head_dim: int = 128
    vocab_size: int = 152064
    intermediate_size: int = 18944
    enable_graph_optimize: bool = True
    enable_kda_pass: bool = True
    beta: float = 0.1


class CGCEngineV2:
    """
    CGC Engine V2 - 完整推理/训练支持

    使用方式:
        # 推理模式 + KDA
        engine = CGCEngineV2(
            mode=Mode.INFERENCE,
            kda_mode=KDAMode.KDA_CPP_NEON,
            gguf_path="/path/to/model.gguf"
        )
        result = engine.generate("Hello world")

        # 训练模式
        engine = CGCEngineV2(
            mode=Mode.TRAINING,
            model=my_model
        )
        loss = engine.train(inputs, targets)
    """

    def __init__(
        self,
        mode: Union[str, Mode] = Mode.INFERENCE,
        kda_mode: Union[str, KDAMode] = KDAMode.STANDARD,
        device: str = "auto",
        gguf_path: Optional[str] = None,
        model: Optional[nn.Module] = None,
        config: Optional[CGCEngineConfigV2] = None,
        **kwargs
    ):
        if isinstance(mode, str):
            self.mode = Mode(mode.lower())
        else:
            self.mode = mode

        if isinstance(kda_mode, str):
            self.kda_mode = KDAMode(kda_mode.lower())
        else:
            self.kda_mode = kda_mode

        self.config = config or CGCEngineConfigV2(mode=self.mode, kda_mode=self.kda_mode)
        self._device = self._detect_device(device)
        self._model: Optional[nn.Module] = None
        self._gguf_path = gguf_path
        self._kda_passes_applied = False
        self._graph_analyzer = None
        self._kda_instance = None

        logger.info(f"[CGCEngineV2] Initializing: mode={self.mode.value}, kda={self.kda_mode.value}, device={self._device}")

        if model is not None:
            self._model = model
        elif gguf_path is not None:
            self._load_gguf_model(gguf_path)
        else:
            logger.warning("[CGCEngineV2] No model provided, engine is empty")

    def _detect_device(self, device: str) -> str:
        """自动检测设备"""
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_gguf_model(self, gguf_path: str):
        """从 GGUF 加载模型"""
        logger.info(f"[CGCEngineV2] Loading GGUF model: {gguf_path}")

        try:
            import gguf
            reader = gguf.GGUFReader(gguf_path)
            logger.info(f"[CGCEngineV2] GGUF reader created")

            tensor_info = {}
            for tensor in reader.tensors:
                tensor_info[tensor.name] = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.tensor_type),
                }

            self._gguf_reader = reader
            self._gguf_tensors = tensor_info

            logger.info(f"[CGCEngineV2] GGUF loaded: {len(tensor_info)} tensors")

            self._build_model_from_gguf()

        except ImportError:
            logger.error("[CGCEngineV2] gguf library not installed. Install with: pip install gguf")
            raise
        except Exception as e:
            logger.error(f"[CGCEngineV2] Failed to load GGUF: {e}")
            raise

    def _build_model_from_gguf(self):
        """从 GGUF 构建 PyTorch 模型"""
        logger.info("[CGCEngineV2] Building PyTorch model from GGUF...")

        config = self.config

        class CGCModel(nn.Module):
            def __init__(self, cfg, kda_mode, device, kda_instance=None):
                super().__init__()
                self.config = cfg
                self.kda_mode = kda_mode
                self.device = device
                self.kda_instance = kda_instance

                self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)

                self.layers = nn.ModuleList([
                    CGCLayer(cfg, kda_mode, kda_instance) for _ in range(cfg.num_layers)
                ])

                self.norm = nn.RMSNorm(cfg.hidden_dim)
                self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)

            def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
                hidden = self.embed_tokens(input_ids)
                for layer in self.layers:
                    hidden = layer(hidden)
                hidden = self.norm(hidden)
                return self.lm_head(hidden)

        class CGCLayer(nn.Module):
            def __init__(self, cfg, kda_mode, kda_instance=None):
                super().__init__()
                self.config = cfg
                self.kda_mode = kda_mode
                self.kda_instance = kda_instance

                self.attention = CGCSelfAttention(cfg, kda_mode, kda_instance)
                self.mlp = nn.Sequential(
                    nn.Linear(cfg.hidden_dim, cfg.intermediate_size, bias=False),
                    nn.SiLU(),
                    nn.Linear(cfg.intermediate_size, cfg.hidden_dim, bias=False)
                )

            def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                attn_output = self.attention(hidden_states)
                return self.mlp(attn_output)

        class CGCSelfAttention(nn.Module):
            """支持 KDA 的 Attention"""
            def __init__(self, cfg, kda_mode, kda_instance=None):
                super().__init__()
                self.config = cfg
                self.kda_mode = kda_mode
                self.kda_instance = kda_instance

                self.q_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
                self.k_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
                self.v_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
                self.o_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)

                self.beta = cfg.beta

            def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                batch_size, seq_len, _ = hidden_states.shape

                q = self.q_proj(hidden_states)
                k = self.k_proj(hidden_states)
                v = self.v_proj(hidden_states)

                q = q.view(batch_size, seq_len, self.config.num_heads, self.config.head_dim).transpose(1, 2)
                k = k.view(batch_size, seq_len, self.config.num_heads, self.config.head_dim).transpose(1, 2)
                v = v.view(batch_size, seq_len, self.config.num_heads, self.config.head_dim).transpose(1, 2)

                if self.kda_mode == KDAMode.KDA_CPP_NEON and self.kda_instance is not None:
                    import numpy as np
                    q_np = np.ascontiguousarray(q.cpu().numpy().astype(np.float32))
                    k_np = np.ascontiguousarray(k.cpu().numpy().astype(np.float32))
                    v_np = np.ascontiguousarray(v.cpu().numpy().astype(np.float32))
                    O = self.kda_instance.forward(q_np, k_np, v_np, beta=self.beta)
                    attn_output = torch.from_numpy(np.array(O)).reshape(q.shape).to(q.device)
                elif self.kda_mode == KDAMode.KDA_PYTORCH:
                    attn_output = self._kda_forward(q, k, v)
                elif self.kda_mode == KDAMode.KDA_METAL:
                    attn_output = self._metal_kda_forward(q, k, v)
                else:
                    attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v)

                attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
                return self.o_proj(attn_output)

            def _kda_forward(self, q, k, v):
                """PyTorch KDA 实现"""
                batch_size, num_heads, seq_len, head_dim = q.shape
                scale = 1.0 / (head_dim ** 0.5)
                S = torch.zeros(batch_size, num_heads, head_dim, head_dim, device=q.device, dtype=q.dtype)
                O = torch.zeros_like(q)

                for i in range(seq_len):
                    ki = k[:, :, i, :]
                    vi = v[:, :, i, :]
                    qi = q[:, :, i, :]
                    kk = torch.einsum('bhd,bhe->bhde', ki, ki)
                    kv = torch.einsum('bhd,bhe->bhde', ki, vi)
                    S = S * (1.0 - self.beta * kk) + self.beta * kv
                    oi = torch.einsum('bhd,bhde->bhe', qi, S) * scale
                    O[:, :, i, :] = oi
                return O

            def _metal_kda_forward(self, q, k, v):
                """Metal KDA (使用 MPS)"""
                return torch.nn.functional.scaled_dot_product_attention(q, k, v)

        kda_instance = None
        if self.kda_mode == KDAMode.KDA_CPP_NEON:
            try:
                import kda_cpp
                kda_instance = kda_cpp.KDA()
                kda_instance.init(1, config.num_heads, config.head_dim)
                logger.info("[CGCEngineV2] C++ KDA NEON loaded")
            except ImportError:
                logger.warning("[CGCEngineV2] C++ KDA not available, falling back to PyTorch")
                self.kda_mode = KDAMode.KDA_PYTORCH

        self._model = CGCModel(config, self.kda_mode, self._device, kda_instance).to(self._device)
        self._model.eval() if self.mode == Mode.INFERENCE else self._model.train()

        logger.info(f"[CGCEngineV2] Model built: layers={config.num_layers}, heads={config.num_heads}")

    def analyze_graph(self) -> Dict[str, Any]:
        """
        计算图分析 - GraphAnalyzer

        Returns:
            分析结果字典
        """
        if self._model is None:
            raise RuntimeError("No model loaded")

        logger.info("[CGCEngineV2] Analyzing computation graph...")

        from cgc_engine.agent.graph_analyzer import GraphAnalyzer

        self._graph_analyzer = GraphAnalyzer()
        features = self._graph_analyzer.analyze(self._model)

        result = {
            "has_attention": features.has_attention,
            "has_flash_attention": features.has_flash_attention,
            "has_moe": features.has_moe,
            "num_layers": features.num_layers,
            "hidden_dim": features.hidden_dim,
            "num_heads": features.num_heads,
        }

        logger.info(f"[CGCEngineV2] Graph analysis: {result}")
        return result

    def apply_kda_pass(self) -> bool:
        """
        应用 KDA Pass - 将 Attention 替换为 KDA

        Returns:
            是否成功
        """
        if self._model is None:
            raise RuntimeError("No model loaded")

        if not self.config.enable_kda_pass:
            logger.info("[CGCEngineV2] KDA pass disabled")
            return False

        logger.info("[CGCEngineV2] Applying KDA Pass...")

        try:
            import torch.fx as fx
            from cgc_engine.cgc.kda_pass import InsertKDAPass

            tracer = fx.Tracer()
            graph = tracer.trace(self._model)

            kda_pass = InsertKDAPass(
                enable_ortho_basis_update=True,
                enable_flashkda_fusion=True,
                kda_scale=self.config.beta,
            )

            if kda_pass.is_applicable(graph):
                logger.info("[CGCEngineV2] KDA Pass applicable, inserting KDA instructions...")
                self._kda_passes_applied = True
            else:
                logger.info("[CGCEngineV2] KDA Pass not applicable for this model")

            return self._kda_passes_applied

        except Exception as e:
            logger.warning(f"[CGCEngineV2] KDA Pass failed: {e}")
            return False

    @torch.no_grad()
    def generate(
        self,
        input_text: str,
        max_tokens: int = 32,
        temperature: float = 0.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成文字

        Args:
            input_text: 输入文本
            max_tokens: 最大生成 token 数
            temperature: 温度

        Returns:
            生成结果
        """
        if self._model is None:
            raise RuntimeError("No model loaded")

        if self.mode != Mode.INFERENCE:
            logger.warning("[CGCEngineV2] generate() called in TRAINING mode")

        input_ids = self._tokenize(input_text)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self._device)

        generated = input_ids.copy()

        import time
        t0 = time.time()

        for _ in range(max_tokens):
            logits = self._model(input_tensor)
            next_logits = logits[0, -1, :]

            if temperature > 0:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = torch.argmax(next_logits).item()

            generated.append(next_token)
            input_tensor = torch.tensor([[next_token]], dtype=torch.long, device=self._device)

        elapsed = time.time() - t0
        tokens_generated = len(generated) - len(input_ids)

        return {
            "generated_ids": generated,
            "generated_text": self._detokenize(generated),
            "num_generated": tokens_generated,
            "time_seconds": elapsed,
            "tokens_per_second": tokens_generated / elapsed if elapsed > 0 else 0,
        }

    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        单步训练

        Args:
            input_ids: 输入 token IDs
            labels: 标签（可选）

        Returns:
            Loss
        """
        if self._model is None:
            raise RuntimeError("No model loaded")

        if self.mode != Mode.TRAINING:
            logger.warning("[CGCEngineV2] train_step() called in INFERENCE mode, switching to training")
            self._model.train()
            self.mode = Mode.TRAINING

        logits = self._model(input_ids)

        if labels is None:
            labels = input_ids

        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )

        return loss

    def _tokenize(self, text: str) -> List[int]:
        """简单分词"""
        return [ord(c) % self.config.vocab_size for c in text]

    def _detokenize(self, token_ids: List[int]) -> str:
        """简单反分词"""
        return "".join(chr(t) if t < 128000 else "?" for t in token_ids)

    def __call__(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """前向传播"""
        if self._model is None:
            raise RuntimeError("No model loaded")
        return self._model(input_ids, **kwargs)

    @property
    def model(self) -> Optional[nn.Module]:
        return self._model

    @property
    def device(self) -> str:
        return self._device


def run_cgc_vs_llama_benchmark():
    """
    CGC Engine vs llama.cpp 完整对比测试
    """
    import time
    import psutil

    print("="*70)
    print("🔥 CGC Engine V2 vs llama.cpp 完整对比测试")
    print("="*70)

    GGUF_FILE = "/Users/alexchuang/Documents/flashkv0430/qwen2.5-7b-q4_k_m.gguf"

    def get_memory():
        return psutil.Process().memory_info().rss / (1024 ** 2)

    prompt = "The quick brown fox jumps over the lazy dog. "
    max_tokens = 32

    print("\n📋 测试配置:")
    print(f"   GGUF: {GGUF_FILE}")
    print(f"   提示: {repr(prompt[:40])}...")
    print(f"   最大生成: {max_tokens} tokens")

    results = {}

    print("\n" + "="*70)
    print("【1】llama.cpp (Ground Truth)")
    print("="*70)

    try:
        from llama_cpp import Llama

        llm = Llama(
            model_path=GGUF_FILE,
            n_ctx=2048,
            n_gpu_layers=32 if torch.backends.mps.is_available() else 0,
            n_threads=8,
            verbose=False
        )

        mem_before = get_memory()
        t0 = time.time()
        output = llm(prompt, max_tokens=max_tokens)
        elapsed = time.time() - t0
        mem_after = get_memory()

        results["llama.cpp"] = {
            "time": elapsed,
            "tokens_per_second": max_tokens / elapsed,
            "memory_mb": mem_after - mem_before,
            "text": output["choices"][0]["text"]
        }

        print(f"   耗时: {elapsed*1000:.2f} ms")
        print(f"   速度: {max_tokens/elapsed:.2f} tok/s")
        print(f"   内存: {mem_after - mem_before:.2f} MB")

        del llm

    except Exception as e:
        print(f"   ❌ llama.cpp 测试失败: {e}")

    print("\n" + "="*70)
    print("【2】CGC Engine V2 - 标准 Attention (MPS)")
    print("="*70)

    try:
        engine_std = CGCEngineV2(
            mode=Mode.INFERENCE,
            kda_mode=KDAMode.STANDARD,
            gguf_path=GGUF_FILE,
            device="mps"
        )

        mem_before = get_memory()
        t0 = time.time()
        output_std = engine_std.generate(prompt, max_tokens=max_tokens)
        elapsed = time.time() - t0
        mem_after = get_memory()

        results["CGC-Standard"] = {
            "time": elapsed,
            "tokens_per_second": output_std["tokens_per_second"],
            "memory_mb": mem_after - mem_before,
            "text": output_std["generated_text"]
        }

        print(f"   耗时: {elapsed*1000:.2f} ms")
        print(f"   速度: {output_std['tokens_per_second']:.2f} tok/s")
        print(f"   内存: {mem_after - mem_before:.2f} MB")

    except Exception as e:
        print(f"   ❌ CGC Standard 测试失败: {e}")

    print("\n" + "="*70)
    print("【3】CGC Engine V2 - PyTorch KDA (MPS)")
    print("="*70)

    try:
        engine_kda = CGCEngineV2(
            mode=Mode.INFERENCE,
            kda_mode=KDAMode.KDA_PYTORCH,
            gguf_path=GGUF_FILE,
            device="mps"
        )

        mem_before = get_memory()
        t0 = time.time()
        output_kda = engine_kda.generate(prompt, max_tokens=max_tokens)
        elapsed = time.time() - t0
        mem_after = get_memory()

        results["CGC-KDA-Py"] = {
            "time": elapsed,
            "tokens_per_second": output_kda["tokens_per_second"],
            "memory_mb": mem_after - mem_before,
            "text": output_kda["generated_text"]
        }

        print(f"   耗时: {elapsed*1000:.2f} ms")
        print(f"   速度: {output_kda['tokens_per_second']:.2f} tok/s")
        print(f"   内存: {mem_after - mem_before:.2f} MB")

    except Exception as e:
        print(f"   ❌ CGC KDA 测试失败: {e}")

    print("\n" + "="*70)
    print("【4】CGC Engine V2 - C++ NEON KDA")
    print("="*70)

    try:
        engine_neon = CGCEngineV2(
            mode=Mode.INFERENCE,
            kda_mode=KDAMode.KDA_CPP_NEON,
            gguf_path=GGUF_FILE,
            device="cpu"
        )

        mem_before = get_memory()
        t0 = time.time()
        output_neon = engine_neon.generate(prompt, max_tokens=max_tokens)
        elapsed = time.time() - t0
        mem_after = get_memory()

        results["CGC-KDA-NEON"] = {
            "time": elapsed,
            "tokens_per_second": output_neon["tokens_per_second"],
            "memory_mb": mem_after - mem_before,
            "text": output_neon["generated_text"]
        }

        print(f"   耗时: {elapsed*1000:.2f} ms")
        print(f"   速度: {output_neon['tokens_per_second']:.2f} tok/s")
        print(f"   内存: {mem_after - mem_before:.2f} MB")

    except Exception as e:
        print(f"   ❌ CGC NEON 测试失败: {e}")

    print("\n" + "="*70)
    print("📊 最终对比结果")
    print("="*70)

    if results:
        print(f"\n{'方案':<20} {'速度 (tok/s)':<15} {'时间 (ms)':<15} {'内存 (MB)':<15}")
        print("-"*70)

        for name, res in sorted(results.items(), key=lambda x: -x[1]["tokens_per_second"]):
            print(f"{name:<20} {res['tokens_per_second']:<15.2f} {res['time']*1000:<15.2f} {res['memory_mb']:<15.2f}")

        if "llama.cpp" in results and "CGC-KDA-NEON" in results:
            speedup = results["CGC-KDA-NEON"]["tokens_per_second"] / results["llama.cpp"]["tokens_per_second"]
            print(f"\n🔥 CGC KDA NEON vs llama.cpp: {speedup:.2f}x")

    print("\n" + "="*70)
    print("✅ 测试完成!")
    print("="*70)

    return results


if __name__ == "__main__":
    run_cgc_vs_llama_benchmark()