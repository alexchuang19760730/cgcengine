"""端侧 MTP Verify Loop — llama.cpp hidden states → MTP head → accept/reject.

架构:
  1. llama.cpp (Metal GPU) 作为 target model, 生成 + 获取 hidden states
  2. PyTorch MTP head (CPU) 接收 hidden states, 链式生成 draft tokens
  3. llama.cpp 逐 token verify draft tokens, argmax 比对 accept/reject
  4. KV cache rewind: 删除被拒绝 token 的 KV cache

关键 API (已验证可用):
  - llama_get_embeddings_ith(ctx, 0) → 单 token decode 后获取 hidden state (norm=284, finite=True)
  - llama_get_logits(ctx) → 单 token decode 后获取 logits (LP_c_float 指针)
  - llama_kv_cache_seq_rm(ctx, seq_id, p0, p1) → 删除 KV cache [p0, p1) 区间
  - llama_get_kv_cache_token_count(ctx) → KV cache 中的 token 数
  - llama_batch_get_one(tokens, n_tokens, pos_0, seq_id) → batch

重要发现:
  - embedding=True + logits_all=True 时, prefill 的 embeddings 只有第一个 token 正确
  - 单 token decode 后, llama_get_embeddings_ith(ctx, 0) 返回正确的 hidden state
  - 因此 verify loop 中每次只 decode 单个 token, 获取其 embedding

性能预估 (M4 16GB):
  - llama.cpp Qwen2.5-0.5B Q4_K_M: ~150 tok/s (Metal)
  - MTP head forward (CPU): ~1-2ms/token
  - 单 token verify: ~6ms (Metal)
  - 预期加速: 1.5-2.5x (accept rate 60-80%)

用法:
  from mtp_verify_loop import MTPVerifyLoop

  loop = MTPVerifyLoop(
      model_path="qwen2.5-0.5b-instruct-q4_k_m.gguf",
      mtp_checkpoint="mtp_head.pt",  # 可选, None 则用 n-gram fallback
      hidden_size=896,
      vocab_size=151936,
  )
  loop.prefill("Hello, how are you?")
  for token_id, from_draft in loop.generate(max_tokens=50, num_draft=4):
      print(token_id, from_draft)
"""
from __future__ import annotations

import ctypes
import json
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Generator, List, Optional, Tuple

import numpy as np
import torch

# llama-cpp-python (系统 Python 3.13)
import llama_cpp


@dataclass
class VerifyStats:
    """Verify loop 统计."""
    total_tokens: int = 0
    draft_tokens: int = 0
    accepted_tokens: int = 0
    rejected_rounds: int = 0
    total_rounds: int = 0
    prefill_ms: float = 0.0
    draft_ms_total: float = 0.0
    verify_ms_total: float = 0.0
    runtime_plan_mode: str = "bypass"
    runtime_plan_enabled: bool = False
    runtime_prefetch_calls: int = 0
    runtime_prefetch_units: int = 0
    runtime_prefetch_noop: bool = True
    runtime_begin_request_ms: float = 0.0

    @property
    def accept_rate(self) -> float:
        return self.accepted_tokens / max(self.draft_tokens, 1)

    @property
    def avg_accept_len(self) -> float:
        return self.accepted_tokens / max(self.total_rounds, 1)

    @property
    def tps(self) -> float:
        total_time = (self.prefill_ms + self.draft_ms_total + self.verify_ms_total) / 1000
        return self.total_tokens / max(total_time, 0.001)

    def summary(self) -> str:
        return (
            f"tokens={self.total_tokens}, "
            f"accept={self.accept_rate:.1%} ({self.accepted_tokens}/{self.draft_tokens}), "
            f"avg_accept_len={self.avg_accept_len:.2f}, "
            f"rounds={self.total_rounds} (rej={self.rejected_rounds}), "
            f"prefill={self.prefill_ms:.0f}ms, "
            f"draft={self.draft_ms_total:.0f}ms, "
            f"verify={self.verify_ms_total:.0f}ms, "
            f"tps={self.tps:.1f}, "
            f"runtime={self.runtime_plan_mode}, "
            f"prefetch_calls={self.runtime_prefetch_calls}, "
            f"prefetch_units={self.runtime_prefetch_units}"
        )


class MTPVerifyLoop:
    """端侧 MTP verify loop.

    流程:
      1. Prefill: llama.cpp 处理 prompt → 获取最后一个 token 的 hidden state + logits
      2. Draft: MTP head 链式生成 N 个 draft tokens (CPU, PyTorch)
      3. Verify: 逐 token forward, 对比 argmax
         - forward draft_tokens[0] → get logits → argmax == draft_tokens[1]?
         - 如果 match, continue; 如果 mismatch, 用 argmax 替代
      4. KV cache rewind: 删除被拒绝 token 的 KV cache
      5. Repeat from step 2

    注意:
      - 单 token decode 后 llama_get_embeddings_ith(ctx, 0) 返回正确的 hidden state
      - hidden_states = result_norm 后, lm_head 前的输出
      - MTP head 的 forward 输入就是这个 hidden state + token embedding
    """

    def __init__(
        self,
        model_path: str,
        mtp_checkpoint: Optional[str] = None,
        hidden_size: int = 896,
        vocab_size: int = 151936,
        num_heads: int = 14,
        head_dim: int = 64,
        intermediate_size: int = 4864,
        n_gpu_layers: int = -1,
        n_ctx: int = 2048,
        n_batch: int = 512,
        n_ubatch: int = 512,
        n_threads: int = 4,
        n_threads_batch: int = 4,
        flash_attn: bool = True,
        offload_kqv: bool = True,
        use_mmap: bool = True,
        use_mlock: bool = False,
        verbose: bool = False,
        use_ngram_fallback: bool = True,
        embed_head_path: Optional[str] = None,
        assistant_model_path: Optional[str] = None,
        use_mlx: bool = False,
        use_cgc_ir: bool = False,
        use_ggml: bool = False,
    ):
        """初始化 verify loop.

        Args:
            model_path: GGUF 模型路径
            mtp_checkpoint: MTP head 权重 (.pt), None 则用 n-gram fallback
            hidden_size: base model hidden size (Qwen2.5-0.5B=896, Qwen3-VL-2B=2048)
            vocab_size: 词表大小
            num_heads: attention heads
            head_dim: head dimension
            intermediate_size: MLP intermediate size
            n_gpu_layers: GPU 层数 (-1 = 全部)
            n_ctx: context size
            n_batch: batch size
            verbose: llama.cpp verbose 输出
            use_ngram_fallback: 无 MTP checkpoint 时用 n-gram 做 draft
            embed_head_path: embed_head.pt 路径 (包含 embed_weight + lm_head_weight)
                             优先从 GGUF 提取, 失败则从此文件加载
            use_mlx: 使用 MLX Metal GPU 加速 MTP forward (2x faster than PyTorch CPU)
            use_cgc_ir: 使用 CGC IR Dispatcher (IR-driven Metal execution, 统一架構)
            use_ggml: 使用 ggml Metal backend 原生执行 (最快, 无 Python per-op 开销)
        """
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.use_ngram_fallback = use_ngram_fallback
        self.embed_head_path = embed_head_path
        self.assistant_model_path = assistant_model_path
        self._use_mlx = use_mlx
        self._mlx_mtp = None
        self._use_cgc_ir = use_cgc_ir
        self._cgc_dispatcher = None
        self._use_ggml = use_ggml
        self._ggml_mtp = None
        self._assistant_proxy = None

        # === 1. 加载 llama.cpp 模型 ===
        print(f"[MTP] Loading llama.cpp model: {model_path}")
        self.llm = llama_cpp.Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_ubatch=n_ubatch,
            n_threads=n_threads,
            n_threads_batch=n_threads_batch,
            flash_attn=flash_attn,
            offload_kqv=offload_kqv,
            use_mmap=use_mmap,
            use_mlock=use_mlock,
            embedding=True,   # 启用 embeddings (hidden states)
            logits_all=False,  # 只需要最后 token logits (节省内存)
            verbose=verbose,
        )

        self.ctx = self.llm.ctx
        self.model = self.llm.model
        self.n_embd = self.llm.n_embd()
        self.n_vocab = self.llm.n_vocab()

        print(f"  n_embd={self.n_embd}, n_vocab={self.n_vocab}")

        if self.n_embd != hidden_size:
            print(f"  [WARN] n_embd={self.n_embd} != hidden_size={hidden_size}, using n_embd")
            self.hidden_size = self.n_embd

        # === 2. 加载 MTP head (PyTorch, CPU) ===
        self._mtp_available = False
        if assistant_model_path and os.path.exists(assistant_model_path):
            print(f"[MTP] Loading official Gemma4 assistant proxy...")
            self._init_assistant_proxy(assistant_model_path)
        elif mtp_checkpoint:
            print(f"[MTP] Loading MTP head...")
            self._init_mtp_head(
                mtp_checkpoint,
                hidden_size=self.hidden_size,
                vocab_size=self.n_vocab,
                num_heads=num_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
            )
        elif use_ngram_fallback:
            print(f"[MTP] No checkpoint, using N-gram draft fallback")
            self._ngram_table: dict = {}
            self._ngram_max = 4
        else:
            print(f"[MTP] No MTP, will do greedy decode only")

        # === 3. 状态 ===
        self.seq_id = 0
        self.n_past = 0
        self.last_hidden: Optional[np.ndarray] = None
        self._last_hidden_ptr: int = 0  # raw ptr for zero-copy ggml native
        self.last_token_id: Optional[int] = None
        self.stats = VerifyStats()
        self._request_sequence = 0
        self._request_runtime_unit_plan: dict[str, Any] = {}
        self._last_unified_runtime_ir: dict[str, Any] = {}
        self._last_backend_lowering: dict[str, Any] = {}
        self._runtime_prefetch_summary: dict[str, Any] = {}

    # ==================== MTP Head ====================

    def _init_mtp_head(
        self,
        checkpoint: str,
        hidden_size: int,
        vocab_size: int,
        num_heads: int,
        head_dim: int,
        intermediate_size: int,
    ):
        """初始化 MTP head 模型."""
        import sys
        import os
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        try:
            from mtp_head.model import MTPHead, MTPHeadConfig

            config = MTPHeadConfig(
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_heads=num_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
            )
            self.mtp_head = MTPHead(config)

            # 加载 checkpoint
            if os.path.exists(checkpoint):
                print(f"  Loading checkpoint: {checkpoint}")
                ckpt = torch.load(checkpoint, weights_only=False, map_location="cpu")
                sd = ckpt.get("model_state_dict", ckpt)
                filtered = {}
                for k, v in sd.items():
                    if "lm_head" in k or "embed" in k:
                        continue
                    filtered[k] = v
                self.mtp_head.load_state_dict(filtered, strict=False)
                print(f"  Loaded {len(filtered)} tensors")
            else:
                print(f"  [WARN] Checkpoint not found: {checkpoint}")
                return

            # 获取 lm_head 和 embedding 权重
            # 优先从 embed_head.pt 加载 (可靠, 避免 GGUF struct 布局兼容问题)
            # GGUF tensor 提取 (_get_model_tensor_torch) 在不同 ggml 版本下
            # struct 布局不同, 容易 segfault; embed_head.pt 是 FP32 numpy, 无兼容问题
            self._lm_head_weight = None
            self._embed_weight = None

            if self.embed_head_path:
                print(f"  Loading weights from {self.embed_head_path}...")
                eh = torch.load(self.embed_head_path, map_location="cpu", weights_only=True)
                embed_weight = eh.get("embed_weight")
                lm_head_weight = eh.get("lm_head_weight")
                tied = lm_head_weight is None and bool(eh.get("lm_head_tied_to_embed"))

                if embed_weight is not None:
                    embed_weight = embed_weight.to(torch.float32)
                if tied:
                    lm_head_weight = embed_weight
                elif lm_head_weight is not None:
                    lm_head_weight = lm_head_weight.to(torch.float32)

                self._embed_weight = embed_weight
                self._lm_head_weight = lm_head_weight

                if self._lm_head_weight is not None:
                    print(f"  lm_head: {self._lm_head_weight.shape}")
                if self._embed_weight is not None:
                    print(f"  embed: {self._embed_weight.shape}")

            if self._lm_head_weight is not None:
                self.mtp_head.set_shared_lm_head(self._lm_head_weight)
            if self._embed_weight is None:
                print(f"  [WARN] Could not extract embedding weight")

            self.mtp_head.eval()
            self._mtp_available = True
            print(f"  MTP head ready")

            # Initialize MLX Metal MTP forward if requested
            if self._use_mlx and self.embed_head_path:
                try:
                    from mtp_mlx_forward import MTPMLXForward
                    self._mlx_mtp = MTPMLXForward(
                        checkpoint=checkpoint,
                        embed_head_path=self.embed_head_path,
                        hidden_size=self.hidden_size,
                        vocab_size=self.n_vocab,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        intermediate_size=intermediate_size,
                    )
                    print(f"  MLX Metal MTP enabled (2x faster draft)")
                except Exception as e:
                    print(f"  [WARN] MLX MTP init failed, using PyTorch CPU: {e}")
                    self._mlx_mtp = None

            # Initialize CGC IR Dispatcher if requested (unified IR-driven execution)
            if self._use_cgc_ir and self.embed_head_path:
                try:
                    from cgc_ir_dispatcher import CGCIRDispatcher, CGCIRConfig
                    cgc_config = CGCIRConfig(
                        hidden_size=self.hidden_size,
                        vocab_size=self.n_vocab,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        intermediate_size=intermediate_size,
                    )
                    self._cgc_dispatcher = CGCIRDispatcher(
                        checkpoint=checkpoint,
                        embed_head_path=self.embed_head_path,
                        config=cgc_config,
                    )
                    print(f"  CGC IR Dispatcher enabled (IR-driven Metal execution)")
                except Exception as e:
                    print(f"  [WARN] CGC IR Dispatcher init failed: {e}")
                    self._cgc_dispatcher = None

            # Initialize ggml native Metal backend if requested (fastest option)
            if self._use_ggml and self.embed_head_path:
                try:
                    from mtp_ggml_native import MTPGgmlNative
                    self._ggml_mtp = MTPGgmlNative(
                        checkpoint_path=checkpoint,
                        embed_head_path=self.embed_head_path,
                        hidden_size=self.hidden_size,
                        vocab_size=self.n_vocab,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        intermediate_size=intermediate_size,
                        use_f16=True,
                    )
                    print(f"  ggml Metal native enabled (F16 weights, ~4ms/step)")
                except Exception as e:
                    print(f"  [WARN] ggml native init failed: {e}")
                    self._ggml_mtp = None

        except Exception as e:
            print(f"  [WARN] MTP head init failed: {e}")
            import traceback
            traceback.print_exc()
            if self.use_ngram_fallback:
                print(f"  Falling back to N-gram draft")
                self._ngram_table = {}
                self._ngram_max = 4

    def _init_assistant_proxy(self, assistant_model_path: str):
        """Load trained Gemma4 assistant weights into a lightweight proxy runtime."""
        import sys

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cgc_phase2 = os.path.join(repo_root, "CGC_Phase2")
        for p in [repo_root, cgc_phase2]:
            if p not in sys.path:
                sys.path.insert(0, p)

        try:
            from gemma4_assistant_proxy import Gemma4AssistantProxy

            self._assistant_proxy = Gemma4AssistantProxy(assistant_model_path, device="cpu")
            self._mtp_available = True
            print(f"  Assistant proxy ready")
        except Exception as e:
            print(f"  [WARN] Assistant proxy init failed: {e}")
            if self.use_ngram_fallback:
                print(f"  Falling back to N-gram draft")
                self._ngram_table = {}
                self._ngram_max = 4

    def _get_model_tensor_torch(self, name: str) -> Optional[torch.Tensor]:
        """从 llama.cpp model 获取内部 tensor 权重 (PyTorch tensor).

        使用 llama_get_model_tensor(model, name) -> ggml_tensor*.
        注意: Q4 量化模型的 tensor 是量化的, 无法直接读取为 float.
        此方法仅对 FP16/FP32 tensor 有效.

        在 llama-cpp-python >= 0.3.x 中, llama_get_model_tensor 可能不存在,
        此时                                                                                                                                                                            返回 None, 由调用方从 embed_head.pt 加载权重.

        Args:
            name: tensor 名 (如 "token_embd.weight", "output.weight")

        Returns:
            torch.Tensor [vocab_size, hidden_size] 或 None
        """
        # 检查 llama_get_model_tensor 是否可用
        if not hasattr(llama_cpp, 'llama_get_model_tensor'):
            return None

        # model 是一个 int (地址)
        model_ptr = ctypes.c_void_p(self.model)
        name_bytes = name.encode("utf-8")

        tensor_ptr = llama_cpp.llama_get_model_tensor(model_ptr, name_bytes)
        if not tensor_ptr:
            return None

        # ggml_tensor struct 布局
        class GGMLTensor(ctypes.Structure):
            _fields_ = [
                ("ne", ctypes.c_int64 * 4),
                ("nb", ctypes.c_size_t * 4),
                ("type", ctypes.c_int32),
                ("_pad", ctypes.c_int32),
                ("buffer", ctypes.c_void_p),
                ("data", ctypes.c_void_p),
            ]

        # 尝试不同的 struct 布局
        # 实际 ggml_tensor 布局 (llama.cpp b5000+):
        # enum ggml_type type;          // 4 bytes
        # struct ggml_backend_buffer * buffer; // 8 bytes (pointer)
        # void * data;                  // 8 bytes (pointer)
        # int64_t ne[GGML_MAX_DIMS];    // 4 * 8 = 32 bytes
        # size_t nb[GGML_MAX_DIMS];     // 4 * 8 = 32 bytes
        class GGMLTensorV2(ctypes.Structure):
            _fields_ = [
                ("type", ctypes.c_int32),
                ("_pad", ctypes.c_int32),
                ("buffer", ctypes.c_void_p),
                ("data", ctypes.c_void_p),
                ("ne", ctypes.c_int64 * 4),
                ("nb", ctypes.c_size_t * 4),
            ]

        # 用 V2 布局
        tensor = GGMLTensorV2.from_address(tensor_ptr)

        n0 = tensor.ne[0]  # hidden_size
        n1 = tensor.ne[1]  # vocab_size
        elem_size = tensor.nb[0]
        tensor_type = tensor.type
        data_ptr = tensor.data

        print(f"    tensor {name}: type={tensor_type}, ne=[{n0},{n1}], nb[0]={elem_size}, data={data_ptr}")

        if n0 == 0 or n1 == 0 or data_ptr == 0:
            return None

        # 检查 tensor type: 0=f32, 1=f16, 2=bf16, 8=q4_0, 12=q8_0, 14=q4_k
        # 量化类型无法直接读取, 需要反量化
        if tensor_type in (0,):  # f32
            total = n0 * n1
            arr_type = ctypes.c_float * total
            arr = arr_type.from_address(data_ptr)
            np_arr = np.frombuffer(arr, dtype=np.float32).reshape(n1, n0).copy()
            return torch.from_numpy(np_arr)
        elif tensor_type in (1,):  # f16
            total = n0 * n1
            arr_type = ctypes.c_uint16 * total
            arr = arr_type.from_address(data_ptr)
            np_arr = np.frombuffer(arr, dtype=np.float16).reshape(n1, n0).copy()
            return torch.from_numpy(np_arr.astype(np.float32))
        else:
            # 量化 tensor (Q4, Q8, Q_K etc.) - 无法直接读取
            # 需要 llama.cpp 的 dequantize API, 或用另一种方式
            print(f"    [WARN] Tensor {name} is quantized (type={tensor_type}), cannot read directly")
            print(f"    Will use logits-based verify (no MTP draft)")
            return None

    # ==================== llama.cpp API Wrappers ====================

    def _get_last_logits(self) -> np.ndarray:
        """获取最后一个 decode 的 logits. shape: [n_vocab]"""
        logits_ptr = llama_cpp.llama_get_logits(self.ctx)
        if not logits_ptr:
            raise RuntimeError("Failed to get logits")
        base = ctypes.addressof(logits_ptr.contents)
        arr_type = ctypes.c_float * self.n_vocab
        return np.array(arr_type.from_address(base), dtype=np.float32)

    def _get_last_hidden(self) -> np.ndarray:
        """获取最后一个 decode 的 hidden state (embedding). shape: [n_embd]

        单 token decode 后, llama_get_embeddings_ith(ctx, 0) 返回正确值.
        """
        emb_ptr = llama_cpp.llama_get_embeddings_ith(self.ctx, 0)
        if not emb_ptr:
            raise RuntimeError("Failed to get embeddings")
        self._last_hidden_ptr = ctypes.addressof(emb_ptr.contents)  # store raw ptr for zero-copy
        base = self._last_hidden_ptr
        arr_type = ctypes.c_float * self.n_embd
        return np.array(arr_type.from_address(base), dtype=np.float32)

    def _get_logits_ith(self, i: int) -> np.ndarray:
        """獲取 batch 中第 i 個 token 的 logits. shape: [n_vocab]"""
        logits_ptr = llama_cpp.llama_get_logits_ith(self.ctx, i)
        if not logits_ptr:
            raise RuntimeError(f"Failed to get logits_ith({i})")
        base = ctypes.addressof(logits_ptr.contents)
        arr_type = ctypes.c_float * self.n_vocab
        return np.array(arr_type.from_address(base), dtype=np.float32)

    def _get_hidden_ith(self, i: int) -> np.ndarray:
        """獲取 batch 中第 i 個 token 的 hidden state. shape: [n_embd]"""
        emb_ptr = llama_cpp.llama_get_embeddings_ith(self.ctx, i)
        if not emb_ptr:
            raise RuntimeError(f"Failed to get embeddings_ith({i})")
        base = ctypes.addressof(emb_ptr.contents)
        arr_type = ctypes.c_float * self.n_embd
        return np.array(arr_type.from_address(base), dtype=np.float32)

    def _make_explicit_batch(
        self,
        tokens: List[int],
        pos: int,
        *,
        logits_mode: str = "last",
    ):
        """顯式構造 llama_batch，適配新版 batch API 對 pos/seq_id 的要求."""
        n = len(tokens)
        batch = llama_cpp.llama_batch_init(n, 0, 1)
        for i, tok in enumerate(tokens):
            batch.token[i] = int(tok)
            batch.pos[i] = pos + i
            batch.n_seq_id[i] = 1
            batch.seq_id[i][0] = self.seq_id
            batch.logits[i] = 1 if logits_mode == "all" or (logits_mode == "last" and i == n - 1) else 0
        batch.n_tokens = n
        return batch

    def _decode_batch_verify(
        self, draft_tokens: List[int], pos: int
    ) -> List[np.ndarray]:
        """Batch verify: 1 次 forward 驗證所有 draft tokens.

        將 [draft_0, draft_1, ..., draft_{N-1}] 一次性送入 llama_decode，
        為每個 token 設置 logits flag，然後用 llama_get_logits_ith 獲取每個位置的 logits。

        Args:
            draft_tokens: draft token IDs
            pos: KV cache 起始位置 (current_token 已在 cache 中)

        Returns:
            logits_list: [N][n_vocab] - 每個 draft token 的 logits
        """
        n = len(draft_tokens)
        batch = self._make_explicit_batch(draft_tokens, pos, logits_mode="all")
        try:
            ret = llama_cpp.llama_decode(self.ctx, batch)
            if ret != 0:
                raise RuntimeError(f"llama_decode batch failed with code {ret}")

            logits_list = [self._get_logits_ith(i) for i in range(n)]
        finally:
            llama_cpp.llama_batch_free(batch)

        return logits_list

    def _decode_single(self, token_id: int, pos: int) -> Tuple[np.ndarray, np.ndarray]:
        """单 token decode, 返回 (hidden_state, logits).

        Args:
            token_id: 要 decode 的 token
            pos: KV cache 位置

        Returns:
            hidden: [n_embd]
            logits: [n_vocab]
        """
        # 交給高階 eval 管理單序列 decode 的位置與 KV 狀態，避免新版 batch ABI/Memory API 失配。
        self.llm.n_tokens = int(pos)
        self.llm.eval([int(token_id)])

        hidden = self._get_last_hidden()
        logits = self._get_last_logits()
        return hidden, logits

    def _decode_batch_prefill(self, tokens: List[int], pos: int) -> Tuple[np.ndarray, np.ndarray]:
        """Batch decode for prefill. 返回最后一个 token 的 (hidden, logits).

        注意: embedding=True + logits_all=False 时:
        - llama_get_logits 返回最后一个 token 的 logits
        - llama_get_embeddings_ith(ctx, 0) 返回第一个 token 的 embedding (不完整)
        - 因此 prefill 后需要做一次单 token decode 获取正确的 hidden

        Args:
            tokens: 要 decode 的 token IDs
            pos: KV cache 起始位置

        Returns:
            hidden: [n_embd] 最后一个 token 的 hidden state
            logits: [n_vocab] 最后一个 token 的 logits
        """
        n_tokens = len(tokens)
        self.llm.n_tokens = int(pos)
        self.llm.eval([int(tok) for tok in tokens])

        # 获取 logits (最后一个 token 的)
        logits = self._get_last_logits()

        # 获取 hidden: prefill 后 embeddings_ith 不可靠
        # 需要重新 decode 最后一个 token 来获取正确的 hidden
        # 但这会重复 forward... 用另一个方法:
        # 做 1 次 single decode 获取 hidden
        last_token = tokens[-1]
        # 不需要重新 decode, 直接用 embedding 的最后一个位置
        emb_ptr = llama_cpp.llama_get_embeddings_ith(self.ctx, n_tokens - 1)
        if emb_ptr:
            base = ctypes.addressof(emb_ptr.contents)
            arr_type = ctypes.c_float * self.n_embd
            hidden = np.array(arr_type.from_address(base), dtype=np.float32)
            # 检查是否有效
            if np.isfinite(hidden).all() and np.linalg.norm(hidden) < 1e6:
                return hidden, logits

        # 如果 embedding_ith 不可靠, 做一次 single decode
        # 回退 KV cache 1 位然后重新 decode
        # 实际上不需要回退, 因为我们要获取的就是最后一个 token 的 hidden
        # 直接 decode 最后一个 token (它会追加到 KV cache)
        # 但这样会导致最后一个 token 被 decode 两次...
        # 更好的方案: prefill 后, 最后一个 token 的 logits 是正确的
        # hidden 可以通过 logits 反推, 或者在第一次 generate 时获取

        # 最简单: 记录 logits, 在 verify 第一个 token 时获取 hidden
        hidden = np.zeros(self.n_embd, dtype=np.float32)  # placeholder
        return hidden, logits

    def _kv_cache_rewind(self, n_keep: int):
        """删除 KV cache 中 n_keep 之后的位置.

        在 llama-cpp-python >= 0.3.x 中, 使用 llama_memory_seq_rm.
        在旧版本中, 使用 llama_kv_cache_seq_rm.

        Args:
            n_keep: 保留前 n_keep 个 token 的 KV cache
        """
        if self.n_past <= n_keep:
            return

        # 新 API (v0.3+): llama_memory_seq_rm(memory, seq, p0, p1)
        if hasattr(llama_cpp, 'llama_memory_seq_rm'):
            mem = llama_cpp.llama_get_memory(self.ctx)
            llama_cpp.llama_memory_seq_rm(
                mem,
                self.seq_id,
                n_keep,
                self.n_past,
            )
        # 旧 API: llama_kv_cache_seq_rm(ctx, seq, p0, p1)
        elif hasattr(llama_cpp, 'llama_kv_cache_seq_rm'):
            llama_cpp.llama_kv_cache_seq_rm(
                self.ctx,
                self.seq_id,
                n_keep,
                self.n_past,
            )
            if hasattr(llama_cpp, 'llama_kv_cache_update'):
                llama_cpp.llama_kv_cache_update(self.ctx)

        self.n_past = n_keep
        try:
            self.llm.n_tokens = n_keep
        except Exception:
            pass

    def _kv_seq_rm(self, p0: int, p1: int = -1):
        """刪除 KV cache 中 [p0, p1) 區間.

        統一封裝新舊 API:
          - v0.3+: llama_memory_seq_rm(memory, seq, p0, p1)
          - 舊版:   llama_kv_cache_seq_rm(ctx, seq, p0, p1) + llama_kv_cache_update

        Args:
            p0: 起始位置 (含)
            p1: 結束位置 (不含), -1 表示到末尾
        """
        if hasattr(llama_cpp, 'llama_memory_seq_rm'):
            mem = llama_cpp.llama_get_memory(self.ctx)
            llama_cpp.llama_memory_seq_rm(mem, self.seq_id, p0, p1)
        elif hasattr(llama_cpp, 'llama_kv_cache_seq_rm'):
            llama_cpp.llama_kv_cache_seq_rm(self.ctx, self.seq_id, p0, p1)
            if hasattr(llama_cpp, 'llama_kv_cache_update'):
                llama_cpp.llama_kv_cache_update(self.ctx)
        try:
            if p1 == -1:
                self.llm.n_tokens = min(int(getattr(self.llm, "n_tokens", 0) or 0), p0)
            else:
                self.llm.n_tokens = min(int(getattr(self.llm, "n_tokens", 0) or 0), p0)
        except Exception:
            pass

    # ==================== Draft Generation ====================

    def _mtp_draft_chain(self, hidden: np.ndarray, token_id: int, num_draft: int) -> List[int]:
        """MTP head 链式生成 draft tokens.

        流程:
          1. current_hidden + embed(current_token) → MTP forward → mtp_hidden
          2. mtp_hidden → lm_head → logits → argmax → draft_token
          3. mtp_hidden 变成新的 current_hidden, draft_token 变成新的 current_token
          4. 重复 1-3

        Args:
            hidden: base model 最后一个 token 的 hidden state [n_embd]
            token_id: 最后一个 token ID
            num_draft: 生成 draft token 数量

        Returns:
            draft_tokens: 生成的 token IDs
        """
        if not self._mtp_available:
            return self._ngram_draft(token_id, num_draft)

        if self._assistant_proxy is not None:
            draft_tokens = []
            current_hidden = torch.from_numpy(hidden).float()
            current_token = token_id
            with torch.no_grad():
                for _ in range(num_draft):
                    next_hidden, logits = self._assistant_proxy.step(current_hidden, current_token)
                    draft_token = int(logits.argmax(dim=-1).item())
                    draft_tokens.append(draft_token)
                    current_hidden = next_hidden[0]
                    current_token = draft_token
            return draft_tokens

        # Use ggml Metal native if available (fastest — F16 weights, zero-copy ptr)
        if self._ggml_mtp is not None:
            # True zero-copy: pass raw pointer from llama_get_embeddings_ith
            if hasattr(self, "_last_hidden_ptr") and self._last_hidden_ptr:
                draft_tokens, _ = self._ggml_mtp.draft_chain_from_ptr(
                    self._last_hidden_ptr, token_id, num_draft
                )
            else:
                draft_tokens, _ = self._ggml_mtp.draft_chain(hidden, token_id, num_draft)
            return draft_tokens

        # Use CGC IR Dispatcher if available (IR-driven Metal execution, unified architecture)
        if self._cgc_dispatcher is not None:
            draft_tokens, _ = self._cgc_dispatcher.draft_chain(hidden, token_id, num_draft)
            return draft_tokens

        # Use MLX Metal GPU if available (2x faster than PyTorch CPU)
        if self._mlx_mtp is not None:
            draft_tokens, _ = self._mlx_mtp.draft_chain(hidden, token_id, num_draft)
            return draft_tokens

        draft_tokens = []
        current_hidden = torch.from_numpy(hidden).float().unsqueeze(0).unsqueeze(0)  # [1, 1, hidden]
        current_token = token_id

        with torch.no_grad():
            for i in range(num_draft):
                # 获取 token embedding
                if self._embed_weight is not None:
                    token_embed = self._embed_weight[current_token].unsqueeze(0).unsqueeze(0)
                else:
                    token_embed = current_hidden  # fallback

                # MTP forward (single pass): compute hidden, then lm_head
                # Fix: previously forward was done twice (once via self.mtp_head() for logits,
                # once manually for hidden). Now compute hidden first, apply lm_head once.
                x = torch.cat([current_hidden, token_embed], dim=-1)
                x = self.mtp_head.proj(x)
                h = x + self.mtp_head.attn(self.mtp_head.norm1(x))
                h = h + self.mtp_head.mlp(self.mtp_head.norm2(h))
                mtp_hidden = self.mtp_head.norm_out(h)

                # lm_head only (shared with base model)
                mtp_logits = self.mtp_head.lm_head(mtp_hidden)  # [1, 1, vocab]
                draft_token = int(mtp_logits.argmax(dim=-1).item())

                draft_tokens.append(draft_token)
                current_hidden = mtp_hidden
                current_token = draft_token

        return draft_tokens

    def _ngram_draft(self, token_id: int, num_draft: int) -> List[int]:
        """N-gram draft: 从历史 token 序列中查找匹配的后续 tokens."""
        if not hasattr(self, "_token_history"):
            self._token_history = []

        self._token_history.append(token_id)
        if len(self._token_history) < 3:
            return []

        # 查找最近 3-gram 在历史中的匹配
        ngram = tuple(self._token_history[-3:])
        draft = []

        # 在历史中搜索
        for i in range(len(self._token_history) - 3):
            if tuple(self._token_history[i:i+3]) == ngram:
                # 取匹配后的 tokens 作为 draft
                for j in range(min(num_draft, len(self._token_history) - i - 3)):
                    draft.append(self._token_history[i + 3 + j])
                    if len(draft) >= num_draft:
                        break
                if draft:
                    break

        return draft[:num_draft]

    # ==================== Public API ====================

    def _normalize_runtime_unit_plan(
        self,
        runtime_unit_plan: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        plan = dict(runtime_unit_plan or {})
        current = list(plan.get("current") or [])
        next_units = list(plan.get("next") or [])
        next_next = list(plan.get("next_next") or [])
        far = list(plan.get("far") or [])
        summary = dict(plan.get("summary") or {})
        return {
            "control_plane": str(plan.get("control_plane") or "expert_data_plane"),
            "enabled": bool(plan.get("enabled")),
            "mode": str(plan.get("mode") or "bypass"),
            "reason": str(plan.get("reason") or ""),
            "model": str(plan.get("model") or ""),
            "family": str(plan.get("family") or ""),
            "route_mode": str(plan.get("route_mode") or ""),
            "frontier_key": str(plan.get("frontier_key") or ""),
            "current": current,
            "next": next_units,
            "next_next": next_next,
            "far": far,
            "summary": summary,
        }

    def prefetch_units(
        self,
        runtime_unit_plan: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Telemetry-only runtime prefetch hook.

        这版先不做真实 residency，只记录 runtime 看见的 unit plan。
        """
        plan = self._normalize_runtime_unit_plan(
            runtime_unit_plan if runtime_unit_plan is not None else self._request_runtime_unit_plan
        )
        candidate_units = [
            *list(plan.get("current") or []),
            *list(plan.get("next") or []),
            *list(plan.get("next_next") or []),
        ]
        sample_keys: list[str] = []
        for unit in candidate_units[:8]:
            if isinstance(unit, dict):
                sample_keys.append(str(unit.get("key") or ""))
            else:
                sample_keys.append(str(unit))
        summary = {
            "status": "noop",
            "residency_action": "telemetry_only",
            "candidate_unit_count": len(candidate_units),
            "current_unit_count": len(plan.get("current") or []),
            "next_unit_count": len(plan.get("next") or []),
            "next_next_unit_count": len(plan.get("next_next") or []),
            "sample_keys": [key for key in sample_keys if key],
        }
        self._runtime_prefetch_summary = summary
        self.stats.runtime_prefetch_calls += 1
        self.stats.runtime_prefetch_units = int(summary["candidate_unit_count"])
        self.stats.runtime_prefetch_noop = True
        # #region debug-point C:prefetch-units
        try:
            _p = os.path.join(os.path.dirname(__file__), "..", ".dbg", "dense-streaming-measure.env")
            _u, _s = "http://127.0.0.1:7777/event", "dense-streaming-measure"
            try:
                with open(_p, "r", encoding="utf-8") as _f:
                    _c = _f.read()
                _u = next((l.split("=", 1)[1] for l in _c.splitlines() if l.startswith("DEBUG_SERVER_URL=")), _u)
                _s = next((l.split("=", 1)[1] for l in _c.splitlines() if l.startswith("DEBUG_SESSION_ID=")), _s)
            except Exception:
                pass
            urllib.request.urlopen(urllib.request.Request(_u, data=json.dumps({
                "sessionId": _s,
                "runId": os.environ.get("EDGE_DEBUG_RUN_ID", "pre-fix"),
                "hypothesisId": "C",
                "location": "CGC_Phase2/mtp_verify_loop.py:prefetch_units",
                "msg": "[DEBUG] evaluated runtime prefetch units",
                "data": {
                    "runtime_mode": str(plan.get("mode") or ""),
                    "runtime_enabled": bool(plan.get("enabled")),
                    "runtime_reason": str(plan.get("reason") or ""),
                    "residency_action": str(summary.get("residency_action") or ""),
                    "candidate_unit_count": int(summary.get("candidate_unit_count") or 0),
                    "current_unit_count": int(summary.get("current_unit_count") or 0),
                    "next_unit_count": int(summary.get("next_unit_count") or 0),
                    "next_next_unit_count": int(summary.get("next_next_unit_count") or 0),
                    "sample_keys": list(summary.get("sample_keys") or []),
                },
            }).encode(), headers={"Content-Type": "application/json"}), timeout=0.35).read()
        except Exception:
            pass
        # #endregion
        return summary

    def begin_request(
        self,
        runtime_unit_plan: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        t0 = time.time()
        self._last_unified_runtime_ir = {}
        self._last_backend_lowering = {}
        payload = runtime_unit_plan
        try:
            from app.shared.colibri_backend import is_unified_runtime_ir_v0, lower_unified_runtime_ir_v0
            if is_unified_runtime_ir_v0(runtime_unit_plan):
                lowered = lower_unified_runtime_ir_v0(runtime_unit_plan)
                self._last_unified_runtime_ir = dict(lowered.get("ir") or {})
                self._last_backend_lowering = dict(lowered.get("backend_lowering") or {})
                payload = lowered.get("runtime_unit_plan")
        except Exception:
            payload = runtime_unit_plan
        plan = self._normalize_runtime_unit_plan(payload)
        self._request_sequence += 1
        self._request_runtime_unit_plan = plan
        self.stats.runtime_plan_mode = str(plan.get("mode") or "bypass")
        self.stats.runtime_plan_enabled = bool(plan.get("enabled"))
        self._runtime_prefetch_summary = self.prefetch_units(plan)
        self.stats.runtime_begin_request_ms = (time.time() - t0) * 1000
        # #region debug-point C:begin-request
        try:
            _p = os.path.join(os.path.dirname(__file__), "..", ".dbg", "dense-streaming-measure.env")
            _u, _s = "http://127.0.0.1:7777/event", "dense-streaming-measure"
            try:
                with open(_p, "r", encoding="utf-8") as _f:
                    _c = _f.read()
                _u = next((l.split("=", 1)[1] for l in _c.splitlines() if l.startswith("DEBUG_SERVER_URL=")), _u)
                _s = next((l.split("=", 1)[1] for l in _c.splitlines() if l.startswith("DEBUG_SESSION_ID=")), _s)
            except Exception:
                pass
            urllib.request.urlopen(urllib.request.Request(_u, data=json.dumps({
                "sessionId": _s,
                "runId": os.environ.get("EDGE_DEBUG_RUN_ID", "pre-fix"),
                "hypothesisId": "C",
                "location": "CGC_Phase2/mtp_verify_loop.py:begin_request",
                "msg": "[DEBUG] began local_mtp runtime request",
                "data": {
                    "request_seq": int(self._request_sequence),
                    "mode": str(plan.get("mode") or ""),
                    "enabled": bool(plan.get("enabled")),
                    "reason": str(plan.get("reason") or ""),
                    "frontier_key": str(plan.get("frontier_key") or ""),
                    "prefetch_status": str((self._runtime_prefetch_summary or {}).get("status") or ""),
                    "prefetch_residency_action": str((self._runtime_prefetch_summary or {}).get("residency_action") or ""),
                    "begin_request_ms": float(self.stats.runtime_begin_request_ms or 0.0),
                },
            }).encode(), headers={"Content-Type": "application/json"}), timeout=0.35).read()
        except Exception:
            pass
        # #endregion
        return self.runtime_request_snapshot()

    def runtime_request_snapshot(self) -> dict[str, Any]:
        plan = dict(self._request_runtime_unit_plan or {})
        return {
            "request_seq": int(self._request_sequence),
            "control_plane": str(plan.get("control_plane") or "expert_data_plane"),
            "mode": str(plan.get("mode") or "bypass"),
            "enabled": bool(plan.get("enabled")),
            "reason": str(plan.get("reason") or ""),
            "frontier_key": str(plan.get("frontier_key") or ""),
            "summary": dict(plan.get("summary") or {}),
            "prefetch": dict(self._runtime_prefetch_summary or {}),
            "unified_runtime_ir": dict(self._last_unified_runtime_ir or {}),
            "backend_lowering": dict(self._last_backend_lowering or {}),
            "begin_request_ms": float(self.stats.runtime_begin_request_ms or 0.0),
        }

    def prefill(self, prompt: str) -> int:
        """Prefill: 处理 prompt, 返回第一个生成 token.

        Args:
            prompt: 输入文本

        Returns:
            first_token_id: 第一个生成的 token ID
        """
        t0 = time.time()

        # Always fully rewind KV cache before prefill
        # (create_completion or prior generate may have left state)
        self.llm.reset()
        self._kv_seq_rm(0, -1)
        self.n_past = 0

        # Tokenize
        tokens = self.llm.tokenize(
            prompt.encode("utf-8"),
            add_bos=True,
            special=True,
        )
        tokens = list(tokens)
        n_prompt = len(tokens)

        self.llm.eval(tokens)
        self.n_past = int(self.llm.n_tokens)

        # 获取最后一个 token 的 logits
        logits = self._get_last_logits()
        self.last_token_id = int(logits.argmax())
        self.last_hidden = None  # prefill 后的 embedding 不可靠

        # 初始化 n-gram 历史
        self._token_history = list(tokens) + [self.last_token_id]

        self.stats.prefill_ms = (time.time() - t0) * 1000
        print(f"  [prefill] {n_prompt} tokens, {self.stats.prefill_ms:.0f}ms, first_token={self.last_token_id}")

        return self.last_token_id

    def generate(
        self,
        max_tokens: int = 50,
        num_draft: int = 4,
    ) -> Generator[Tuple[int, bool], None, None]:
        """MTP speculative decoding generate.

        Correct flow:
          1. Decode current_token → hidden + logits (only ONCE per round start)
          2. MTP draft from hidden
          3. Verify: logits.argmax() vs draft[0]
             - accept → decode draft[0] → new hidden + logits → verify draft[1]
             - reject → yield target, decode target → new hidden + logits
          4. Next round uses saved hidden + logits

        Yields:
            (token_id, from_draft): token ID 和是否来自 draft
        """
        if self.last_token_id is None:
            raise RuntimeError("Call prefill() first")

        current_token = self.last_token_id
        current_hidden = self.last_hidden  # None on first round
        current_logits = None
        n_generated = 0

        while n_generated < max_tokens:
            self.stats.total_rounds += 1

            # === 1. Ensure we have hidden + logits for current_token ===
            if current_hidden is None or current_logits is None:
                t0 = time.time()
                current_hidden, current_logits = self._decode_single(
                    current_token, self.n_past
                )
                self.n_past += 1
                self.stats.verify_ms_total += (time.time() - t0) * 1000

            # === 2. MTP draft ===
            t0 = time.time()
            draft_tokens = self._mtp_draft_chain(
                current_hidden, current_token, num_draft
            )
            self.stats.draft_ms_total += (time.time() - t0) * 1000
            self.stats.draft_tokens += len(draft_tokens)

            if not draft_tokens:
                # No draft, greedy decode
                next_token = int(current_logits.argmax())
                yield next_token, False
                n_generated += 1
                current_token = next_token
                current_hidden = None  # will decode next round
                current_logits = None
                continue

            # === 3. Batch verify: 1 次 forward 驗證所有 draft ===
            t0 = time.time()
            batch_pos = self.n_past
            batch_error: Optional[Exception] = None
            batch_ok = False
            # 单 draft 时 batch verify 没有收益，且当前 llama.cpp 绑定在 fatal error 后会污染上下文；
            # 这里优先走逐 token verify，避免把真正错误掩盖成后续 single decode 的连锁失败。
            if len(draft_tokens) > 1:
                try:
                    batch_logits = self._decode_batch_verify(
                        draft_tokens, self.n_past
                    )
                    self.n_past += len(draft_tokens)
                    batch_ok = True
                except Exception as e:
                    batch_error = e
                    self._kv_seq_rm(batch_pos, -1)
                    self.n_past = batch_pos
                    print(f"  [verify] batch verify failed, fallback to single-step verify: {e}")

            n_accept = 0
            rejected = False

            if batch_ok:
                # current_logits 驗證 draft[0]，batch_logits[i] 驗證 draft[i+1]
                verify_logits = current_logits
                for i, draft_token in enumerate(draft_tokens):
                    target_token = int(verify_logits.argmax())

                    if target_token == draft_token:
                        # Accept!
                        n_accept += 1
                        yield draft_token, True
                        self.stats.accepted_tokens += 1
                        n_generated += 1
                        if n_generated >= max_tokens:
                            break
                        # 下一個驗證用 batch_logits[i]
                        if i < len(draft_tokens) - 1:
                            verify_logits = batch_logits[i]
                        current_token = draft_token
                    else:
                        # Reject! KV cache 回退 + decode target
                        yield target_token, False
                        n_generated += 1
                        self.stats.rejected_rounds += 1

                        # 刪除 reject 位置之後的 KV cache
                        rm_start = batch_pos + i
                        self._kv_seq_rm(rm_start, -1)
                        self.n_past = rm_start

                        # decode target → new hidden + logits
                        t_v = time.time()
                        current_hidden, current_logits = self._decode_single(
                            target_token, self.n_past
                        )
                        self.n_past += 1
                        self.stats.verify_ms_total += (time.time() - t_v) * 1000
                        current_token = target_token
                        rejected = True
                        break

                self.stats.verify_ms_total += (time.time() - t0) * 1000
            else:
                # Fallback: 逐個 decode 驗證 (舊邏輯)
                for i, draft_token in enumerate(draft_tokens):
                    target_token = int(current_logits.argmax())

                    if target_token == draft_token:
                        n_accept += 1
                        yield draft_token, True
                        self.stats.accepted_tokens += 1
                        n_generated += 1
                        if n_generated >= max_tokens:
                            break
                        t_v = time.time()
                        current_hidden, current_logits = self._decode_single(
                            draft_token, self.n_past
                        )
                        self.n_past += 1
                        self.stats.verify_ms_total += (time.time() - t_v) * 1000
                        current_token = draft_token
                    else:
                        yield target_token, False
                        n_generated += 1
                        self.stats.rejected_rounds += 1
                        t_v = time.time()
                        current_hidden, current_logits = self._decode_single(
                            target_token, self.n_past
                        )
                        self.n_past += 1
                        self.stats.verify_ms_total += (time.time() - t_v) * 1000
                        current_token = target_token
                        rejected = True
                        break

                self.stats.verify_ms_total += (time.time() - t0) * 1000

            if not rejected and n_accept == len(draft_tokens) and n_generated < max_tokens:
                # All drafts accepted! Bonus token from last logits
                # batch_ok 時用 batch_logits[-1]（最後一個 draft 的 logits）
                if batch_ok:
                    bonus_token = int(batch_logits[-1].argmax())
                else:
                    bonus_token = int(current_logits.argmax())
                yield bonus_token, False
                n_generated += 1
                current_token = bonus_token
                # Need to decode bonus_token next round
                current_hidden = None
                current_logits = None
            elif not rejected:
                # Ran out of max_tokens during accept loop
                break
            # If rejected, current_token and current_hidden/logits already set

            self.stats.total_tokens = n_generated

            # If all draft accepted, need to forward bonus_token next round
            if n_accept == len(draft_tokens) and n_generated < max_tokens:
                # current_token = bonus_token (already set)
                # Need to get hidden for bonus_token
                # But we have logits from last forward
                # hidden from last forward is current_hidden
                # That's the hidden of the last draft token, not bonus_token
                # We need to forward bonus_token to get its hidden
                # But that would be the first thing in next round
                # Set current_hidden = None to trigger forward in next round
                # Actually, we can use current_hidden (from last draft token forward)
                # because MTP uses the hidden of the last token, not the bonus token
                pass  # current_hidden is already set

        self.stats.total_tokens = n_generated

    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 50,
        num_draft: int = 4,
    ) -> Tuple[str, VerifyStats]:
        """完整的文本生成接口."""
        first_token = self.prefill(prompt)

        tokens = [first_token]
        for token_id, from_draft in self.generate(max_tokens=max_tokens - 1, num_draft=num_draft):
            tokens.append(token_id)

        generated = self.llm.detokenize(tokens)
        return generated, self.stats

    def bench(
        self,
        prompt: str,
        max_tokens: int = 50,
        num_draft: int = 4,
        label: str = "",
    ) -> Optional[dict]:
        """Benchmark MTP verify loop.

        Returns:
            {"tps": float, "accept_rate": float, "stats": VerifyStats}
        """
        print(f"\n{'='*60}")
        print(f"  {label}" if label else f"  MTP Verify Loop (N={num_draft})")
        print(f"{'='*60}")

        # Reset stats
        self.stats = VerifyStats()

        # Actual run
        t0 = time.time()
        first_token = self.prefill(prompt)
        t_first = time.time()

        tokens = [first_token]
        try:
            for token_id, from_draft in self.generate(max_tokens=max_tokens - 1, num_draft=num_draft):
                tokens.append(token_id)
        except Exception as e:
            print(f"  Generate error: {e}")
            import traceback
            traceback.print_exc()
            return None

        t_end = time.time()
        dt = t_end - t_first
        nd = len(tokens) - 1

        if dt <= 0 or nd <= 0:
            print("  No tokens generated")
            return None

        tps = nd / dt
        text = self.llm.detokenize(tokens)

        print(f"  TTFT: {1000*(t_first-t0):.0f}ms")
        print(f"  Decode: {tps:.1f} tok/s")
        print(f"  Accept: {self.stats.accept_rate:.1%} ({self.stats.accepted_tokens}/{self.stats.draft_tokens})")
        print(f"  Avg accept len: {self.stats.avg_accept_len:.2f}")
        print(f"  Rounds: {self.stats.total_rounds} (rejected: {self.stats.rejected_rounds})")
        print(f"  Draft time: {self.stats.draft_ms_total:.0f}ms total ({self.stats.draft_ms_total/max(self.stats.total_rounds,1):.1f}ms/round)")
        print(f"  Verify time: {self.stats.verify_ms_total:.0f}ms total ({self.stats.verify_ms_total/max(self.stats.total_rounds,1):.1f}ms/round)")
        print(f"  Output: {text[:200]}")
        print(f"  Stats: {self.stats.summary()}")

        return {"tps": tps, "accept_rate": self.stats.accept_rate, "stats": self.stats}


def main():
    """主入口: 测试 verify loop."""
    import sys
    import os

    # 默认模型路径
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/alexchuang/models/gguf/qwen2.5-0.5b-instruct-fp16.gguf"
    mtp_ckpt = sys.argv[2] if len(sys.argv) > 2 else None
    embed_head = sys.argv[3] if len(sys.argv) > 3 else None

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Usage: python mtp_verify_loop.py <model.gguf> [mtp_checkpoint.pt]")
        sys.exit(1)

    # Qwen2.5-0.5B config
    loop = MTPVerifyLoop(
        model_path=model_path,
        mtp_checkpoint=mtp_ckpt,
        hidden_size=896,
        vocab_size=151936,
        num_heads=14,
        head_dim=64,
        intermediate_size=4864,
        n_gpu_layers=-1,
        n_ctx=2048,
        verbose=False,
        use_ngram_fallback=True,
        embed_head_path=embed_head,
        use_cgc_ir=False,
        use_ggml=True,
    )

    prompt = "Write a Python function to check if a number is prime:"

    # Baseline (no MTP, just llama.cpp generate)
    print("\n" + "="*60)
    print("  Baseline (llama.cpp, no MTP)")
    print("="*60)
    t0 = time.time()
    output = loop.llm.create_completion(
        prompt,
        max_tokens=50,
        temperature=0,
        top_p=1,
    )
    dt = time.time() - t0
    baseline_tps = 50 / dt
    print(f"  Time: {dt:.2f}s, TPS: {baseline_tps:.1f}")
    print(f"  Output: {output['choices'][0]['text'][:200]}")

    # MTP verify loop
    num_draft = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    max_tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 50
    result = loop.bench(prompt, max_tokens=max_tokens, num_draft=num_draft,
                        label=f"MTP Verify Loop (N={num_draft}, ggml native C/C++)")

    if result:
        print(f"\n  Speedup: {result['tps']/baseline_tps:.2f}x vs baseline")
        print(f"  Accept rate: {result['accept_rate']:.1%}")


if __name__ == "__main__":
    main()
