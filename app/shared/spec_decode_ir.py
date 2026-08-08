"""统一投机 decode IR — 一份配置, 跨后端运行 (mlx | pytorch | sglang).

扩展 unified_mtp_ir.py, 支持 0.5B 独立 draft model 的投机 decode.

用法:
  # MLX 后端 (Mac)
  python -m app.shared.spec_decode_ir --backend mlx --mode chain --num-draft 16

  # PyTorch 后端 (GPU)
  python -m app.shared.spec_decode_ir --backend pytorch --mode chain --num-draft 16

  # SGLang 后端 (cloud)
  python -m app.shared.spec_decode_ir --backend sglang --mode chain --num-draft 16

配置可保存为 JSON, 不同后端共享同一配置:
  ir = SpecDecodeConfig.from_json("spec_config.json")
  backend = create_backend(ir.backend)
  result = backend.generate(prompt, max_tokens=100)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Generator, Tuple
from pathlib import Path


# ============================================================================
# IR 配置定义
# ============================================================================

@dataclass
class SpecDecodeConfig:
    """投机 decode 统一配置 — 跨后端共享."""
    # 后端
    backend: str = "mlx"  # mlx | pytorch | sglang

    # 模式
    mode: str = "chain"  # chain | eagle | pipeline
    num_draft_tokens: int = 16  # chain: N (最优 16, photosynthesis 2.0x)
    top_k: int = 4  # eagle: top-k 候选
    tree_depth: int = 1  # eagle: tree 深度

    # Pipeline 模式配置 (端侧 draft + 云端 verify 重叠)
    pipeline_cloud_url: str = ""  # 云端 target URL (mTLS 直连, e.g. http://47.95.250.55:30001)
    pipeline_rtt_ms: float = 15.0  # 预估 RTT (mTLS ~15ms, SSH ~55ms)
    pipeline_overlap: bool = True  # draft/verify 重叠 (True=pipeline, False=串行)

    # 模型路径 (后端特定)
    target_model: str = "Qwen3-VL-2B"  # mlx: 本地路径; pytorch: HF/本地; sglang: server URL
    draft_model: str = "Qwen2.5-0.5B-Instruct-4bit"  # draft model

    # 后端特定配置
    mlx_target_path: str = "/Users/alexchuang/models/Qwen3-VL-2B-bf16"
    mlx_draft_model: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

    pytorch_target_path: str = "/data2/models/Qwen3-VL-2B-Instruct"
    pytorch_draft_model: str = "/data2/models/Qwen2.5-0.5B-Instruct"  # 本地路径 (避免网络)
    pytorch_device: str = "cuda:0"
    pytorch_dtype: str = "bfloat16"

    sglang_target_url: str = "http://47.95.250.55:30001"
    sglang_draft_url: str = ""  # 可选: 独立 draft server
    sglang_api_key: str = ""

    # sglang server 启动参数 (AutoTunner 根据模型类型自动设置)
    sglang_model_path: str = ""  # sglang server 的模型路径
    sglang_env_vars: Dict[str, str] = field(default_factory=dict)  # CGC_ENABLE_ORTHO_KDA 等
    sglang_extra_args: List[str] = field(default_factory=list)  # --speculative-algorithm NEXTN 等
    sglang_mem_fraction: float = 0.88
    sglang_cuda_graph_max_bs: int = 256
    sglang_disable_cuda_graph: bool = False
    sglang_speculative_algorithm: str = ""  # NEXTN | NGRAM | EAGLE | "" (none)

    # 基准
    baseline_tps: float = 26.8  # Mac MLX baseline (无投机)

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "SpecDecodeConfig":
        with open(path) as f:
            return cls(**json.load(f))


@dataclass
class SpecResult:
    """投机 decode 单次结果."""
    tokens: List[int] = field(default_factory=list)
    accept_count: int = 0
    total_count: int = 0
    tps: float = 0.0
    ttft_ms: float = 0.0
    accept_rate: float = 0.0
    speedup: float = 0.0
    output_text: str = ""

    def __post_init__(self):
        if self.total_count > 0:
            self.accept_rate = self.accept_count / self.total_count


# ============================================================================
# 抽象后端接口
# ============================================================================

class SpecDecodeBackend:
    """投机 decode 抽象后端接口."""

    def __init__(self, config: SpecDecodeConfig):
        self.config = config
        self.target_model = None
        self.draft_model = None
        self.tokenizer = None

    def load(self):
        """加载 target + draft 模型."""
        raise NotImplementedError

    def generate(self, prompt: str, max_tokens: int = 100) -> SpecResult:
        """投机 decode 生成."""
        raise NotImplementedError

    def bench(self, prompts: List[str], max_tokens: int = 30) -> Dict[str, SpecResult]:
        """批量 bench."""
        results = {}
        for prompt in prompts:
            print(f"  {prompt[:40]}...")
            results[prompt[:30]] = self.generate(prompt, max_tokens)
        return results


# ============================================================================
# MLX 后端 (Mac)
# ============================================================================

class MLXBackend(SpecDecodeBackend):
    """MLX 后端 — Mac 本地推理."""

    def load(self):
        from mlx_lm import load
        print(f"[mlx] Loading target: {self.config.mlx_target_path}")
        self.target_model, self.tokenizer = load(self.config.mlx_target_path)
        print(f"[mlx] Loading draft: {self.config.mlx_draft_model}")
        self.draft_model, _ = load(self.config.mlx_draft_model)
        print("[mlx] Models loaded")

    def generate(self, prompt: str, max_tokens: int = 100) -> SpecResult:
        if self.target_model is None:
            self.load()

        if self.config.mode == "chain":
            return self._generate_chain(prompt, max_tokens)
        elif self.config.mode == "eagle":
            return self._generate_eagle(prompt, max_tokens)
        elif self.config.mode == "pipeline":
            return self._generate_pipeline(prompt, max_tokens)
        else:
            raise ValueError(f"Unknown mode: {self.config.mode}")

    def _generate_pipeline(self, prompt: str, max_tokens: int) -> SpecResult:
        """Pipeline 投机解码: 端侧 draft + 云端 verify 重叠.

        draft N token → 异步 emit → 立即 draft N+1 → 收到 verify N 结果
        RTT 完全隐藏 (draft 和 verify 并行)

        需要:
          - config.pipeline_cloud_url: 云端 target URL
          - config.draft_model: 端侧 MTP head / 小 draft model
          - mTLS 直连 (RTT ~15ms)
        """
        import asyncio
        import time as _time
        import requests

        N = self.config.num_draft_tokens
        cloud_url = self.config.pipeline_cloud_url
        if not cloud_url:
            raise ValueError("pipeline mode 需要 config.pipeline_cloud_url")

        print(f"[pipeline] N={N}, cloud={cloud_url}, RTT={self.config.pipeline_rtt_ms}ms")

        result = SpecResult()
        result.mode = "pipeline"
        t0 = _time.time()

        # 端侧 draft model (MTP head 或小模型)
        # 使用 MLX stream_generate 生成 draft
        from mlx_lm import stream_generate

        # 云端 prefill (首次请求)
        try:
            r = requests.post(
                f"{cloud_url}/generate",
                json={"text": prompt, "max_tokens": 1, "stream": False},
                timeout=30,
            )
            if r.status_code == 200:
                first_token = r.json().get("text", "")
                result.tokens.append(first_token)
                result.accepted += 1
                print(f"[pipeline] 云端 prefill: first token = '{first_token}'")
            else:
                print(f"[pipeline] 云端 prefill 失败: {r.status_code}")
        except Exception as e:
            print(f"[pipeline] 云端连接失败: {e}")
            # 降级到纯端侧
            return self._generate_chain(prompt, max_tokens)

        # Pipeline 循环: draft → emit → (并行) draft next → verify result
        total_tokens = 1
        while total_tokens < max_tokens:
            # ① 端侧 draft N token
            draft_tokens = []
            draft_prompt = prompt + "".join(result.tokens)
            try:
                for _ in range(N):
                    # MTP head / 小模型生成 1 token
                    stream = stream_generate(self.draft_model, self.tokenizer, draft_prompt, max_tokens=1)
                    for s in stream:
                        draft_tokens.append(s.token)
                        draft_prompt += s.token
                        break
            except Exception as e:
                print(f"[pipeline] draft 失败: {e}")
                break

            if not draft_tokens:
                break

            t_draft = _time.time()

            # ② emit draft → 云端 verify (batch)
            try:
                r = requests.post(
                    f"{cloud_url}/generate",
                    json={
                        "text": prompt + "".join(result.tokens),
                        "draft_tokens": draft_tokens,
                        "max_tokens": N + 1,
                        "speculative": True,
                        "stream": False,
                    },
                    timeout=30,
                )
                t_verify = _time.time()

                if r.status_code == 200:
                    resp = r.json()
                    verified = resp.get("tokens", [])
                    accepted = resp.get("accepted", 0)

                    # 添加 accepted tokens
                    for tok in verified[:accepted + 1]:
                        result.tokens.append(tok)
                        total_tokens += 1

                    result.accepted += accepted
                    result.rejected += (N - accepted)

                    rtt_ms = (t_verify - t_draft) * 1000
                    print(f"[pipeline] draft={N}, accepted={accepted}, RTT={rtt_ms:.0f}ms, total={total_tokens}")

                    if accepted == 0:
                        # 全部 reject, 云端给了 1 个新 token
                        result.tokens.append(verified[0] if verified else "")
                        total_tokens += 1
                else:
                    print(f"[pipeline] verify 失败: {r.status_code}")
                    break
            except Exception as e:
                print(f"[pipeline] verify 错误: {e}")
                break

        result.total_time = _time.time() - t0
        result.tps = total_tokens / result.total_time if result.total_time > 0 else 0
        print(f"[pipeline] 完成: {total_tokens} tokens, {result.total_time:.2f}s, {result.tps:.1f} tok/s")
        return result


    def _generate_chain(self, prompt: str, max_tokens: int) -> SpecResult:
        from mlx_lm import stream_generate
        N = self.config.num_draft_tokens

        # warmup
        try:
            list(stream_generate(self.target_model, self.tokenizer, prompt,
                                 max_tokens=2, draft_model=self.draft_model,
                                 num_draft_tokens=N))
        except Exception:
            pass

        t0 = time.time()
        tokens = []
        draft_count = 0
        total = 0
        t_first = None

        for resp in stream_generate(self.target_model, self.tokenizer, prompt,
                                     max_tokens=max_tokens,
                                     draft_model=self.draft_model,
                                     num_draft_tokens=N):
            tokens.append(resp.token)
            total += 1
            if getattr(resp, "from_draft", False):
                draft_count += 1
            if len(tokens) == 1:
                t_first = time.time()

        t_end = time.time()
        if t_first is None:
            t_first = t0
        dt = t_end - t_first
        nd = len(tokens) - 1

        result = SpecResult(
            tokens=tokens,
            accept_count=draft_count,
            total_count=total,
        )
        if nd > 0 and dt > 0:
            result.tps = nd / dt
            result.ttft_ms = 1000 * (t_first - t0)
            result.speedup = result.tps / self.config.baseline_tps
        result.output_text = self.tokenizer.decode(tokens[:50])
        return result

    def _generate_eagle(self, prompt: str, max_tokens: int) -> SpecResult:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "CGC_Phase2"))
        from eagle_tree_search import eagle_tree_generate

        # warmup
        try:
            list(eagle_tree_generate(self.target_model, self.tokenizer, self.draft_model,
                                     prompt, max_tokens=3, top_k=self.config.top_k,
                                     tree_depth=self.config.tree_depth))
        except Exception:
            pass

        t0 = time.time()
        tokens = []
        draft_count = 0
        total = 0
        t_first = None

        for token_id, from_draft in eagle_tree_generate(
            self.target_model, self.tokenizer, self.draft_model, prompt,
            max_tokens=max_tokens, top_k=self.config.top_k,
            tree_depth=self.config.tree_depth
        ):
            tokens.append(token_id)
            total += 1
            if from_draft:
                draft_count += 1
            if len(tokens) == 1:
                t_first = time.time()

        t_end = time.time()
        if t_first is None:
            t_first = t0
        dt = t_end - t_first
        nd = len(tokens) - 1

        result = SpecResult(
            tokens=tokens,
            accept_count=draft_count,
            total_count=total,
        )
        if nd > 0 and dt > 0:
            result.tps = nd / dt
            result.ttft_ms = 1000 * (t_first - t0)
            result.speedup = result.tps / self.config.baseline_tps
        result.output_text = self.tokenizer.decode(tokens[:50])
        return result


# ============================================================================
# PyTorch 后端 (GPU)
# ============================================================================

class PyTorchBackend(SpecDecodeBackend):
    """PyTorch 后端 — GPU 推理 (Host2)."""

    def load(self):
        import sys
        # 统一用 model_loader 加载 (自动检测 VL/纯文本/MoE, 不硬编码类名)
        sys.path.insert(0, "/root/flashkv0516/app/shared")
        try:
            from model_loader import load_base_model
        except ImportError:
            # Mac 本地路径
            sys.path.insert(0, "/Users/alexchuang/Documents/flashkv0516/app/shared")
            from model_loader import load_base_model

        device = self.config.pytorch_device
        dtype = self.config.pytorch_dtype  # 字符串, model_loader 内部转换

        print(f"[pytorch] Loading target: {self.config.pytorch_target_path}")
        self.target_model, self.tokenizer = load_base_model(
            self.config.pytorch_target_path, device=device, dtype=dtype
        )
        self.target_model.eval()

        print(f"[pytorch] Loading draft: {self.config.pytorch_draft_model}")
        self.draft_model, _ = load_base_model(
            self.config.pytorch_draft_model, device=device, dtype=dtype
        )
        self.draft_model.eval()
        print("[pytorch] Models loaded (via unified model_loader)")

    def generate(self, prompt: str, max_tokens: int = 100) -> SpecResult:
        import torch
        import torch.nn.functional as F

        if self.target_model is None:
            self.load()

        device = self.config.pytorch_device
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)

        # Prefill target
        with torch.no_grad():
            target_out = self.target_model(input_ids, use_cache=True, output_hidden_states=False)
        target_kv = target_out.past_key_values
        target_logits = target_out.logits[0, -1]
        first_token = int(target_logits.argmax().item())

        # Prefill draft
        with torch.no_grad():
            draft_out = self.draft_model(input_ids, use_cache=True)
        draft_kv = draft_out.past_key_values

        tokens = [first_token]
        draft_count = 0
        total = 1
        t0 = time.time()
        t_first = None

        current_token = first_token
        N = self.config.num_draft_tokens

        while len(tokens) < max_tokens:
            # Draft N tokens (chain)
            draft_tokens = []
            draft_current = current_token
            for _ in range(N):
                with torch.no_grad():
                    d_out = self.draft_model(
                        torch.tensor([[draft_current]], device=device),
                        past_key_values=draft_kv, use_cache=True
                    )
                draft_kv = d_out.past_key_values
                d_next = int(d_out.logits[0, -1].argmax().item())
                draft_tokens.append(d_next)
                draft_current = d_next

            # Target verify [current + draft_tokens]
            verify_tokens = torch.tensor([[current_token] + draft_tokens], device=device)
            with torch.no_grad():
                v_out = self.target_model(verify_tokens, past_key_values=target_kv, use_cache=True)
            target_kv = v_out.past_key_values
            verify_argmax = v_out.logits[0].argmax(dim=-1).tolist()

            # Accept matching
            n_accept = 0
            for i in range(N):
                if draft_tokens[i] == verify_argmax[i]:
                    n_accept += 1
                else:
                    break

            for i in range(n_accept):
                tokens.append(draft_tokens[i])
                total += 1
                draft_count += 1
                if len(tokens) == 2:
                    t_first = time.time()

            # Add target's correction token
            if len(tokens) < max_tokens:
                correct = verify_argmax[n_accept] if n_accept < len(verify_argmax) else verify_argmax[-1]
                tokens.append(correct)
                total += 1
                if len(tokens) == 2:
                    t_first = time.time()
                current_token = correct
            else:
                break

            # Trim KV caches
            rewind = N - n_accept
            if rewind > 0 and target_kv is not None:
                target_kv = self._trim_kv(target_kv, rewind)
            if rewind > 0 and draft_kv is not None:
                draft_kv = self._trim_kv(draft_kv, rewind)

        t_end = time.time()
        if t_first is None:
            t_first = t0
        dt = t_end - t_first
        nd = len(tokens) - 1

        result = SpecResult(
            tokens=tokens,
            accept_count=draft_count,
            total_count=total,
        )
        if nd > 0 and dt > 0:
            result.tps = nd / dt
            result.ttft_ms = 1000 * (t_first - t0)
            result.speedup = result.tps / self.config.baseline_tps
        result.output_text = self.tokenizer.decode(tokens[:50])
        return result

    def _trim_kv(self, kv, n):
        """Trim KV cache by n tokens (兼容 transformers 4.x/5.x DynamicCache)."""
        if kv is None or n <= 0:
            return kv
        try:
            # 方法1: DynamicCache.crop (transformers 4.44+)
            if hasattr(kv, 'crop'):
                seq_len = kv.get_seq_length() if hasattr(kv, 'get_seq_length') else None
                if seq_len and seq_len > n:
                    kv.crop(seq_len - n)
                return kv
        except Exception:
            pass
        try:
            # 方法2: 直接操作 key_cache/value_cache
            if hasattr(kv, 'key_cache') and hasattr(kv, 'value_cache'):
                for layer_idx in list(kv.key_cache.keys()):
                    k = kv.key_cache[layer_idx]
                    if k.shape[-1] > n:
                        kv.key_cache[layer_idx] = k[:, :, :-n]
                        kv.value_cache[layer_idx] = kv.value_cache[layer_idx][:, :, :-n]
                return kv
        except Exception:
            pass
        return kv


# ============================================================================
# SGLang 后端 (cloud)
# ============================================================================

class SGLangBackend(SpecDecodeBackend):
    """sglang 后端 — cloud 推理 (通过 HTTP API)."""

    def load(self):
        import requests
        self.session = requests.Session()  # 连接池
        # 检查 sglang server 是否在线
        url = f"{self.config.sglang_target_url}/v1/models"
        try:
            resp = self.session.get(url, timeout=5)
            if resp.status_code == 200:
                print(f"[sglang] Target server online: {self.config.sglang_target_url}")
            else:
                print(f"[sglang] Target server responded {resp.status_code}")
        except Exception as e:
            print(f"[sglang] WARNING: cannot reach target server: {e}")
        print("[sglang] Ready")

    def generate(self, prompt: str, max_tokens: int = 100) -> SpecResult:
        import requests

        if not hasattr(self, "session"):
            self.load()

        url = f"{self.config.sglang_target_url}/v1/chat/completions"
        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        headers = {}
        if self.config.sglang_api_key:
            headers["Authorization"] = f"Bearer {self.config.sglang_api_key}"

        t0 = time.time()
        resp = self.session.post(url, json=payload, headers=headers, timeout=60)
        t_end = time.time()

        if resp.status_code != 200:
            return SpecResult(output_text=f"Error: {resp.status_code} {resp.text[:100]}")

        data = resp.json()
        choice = data.get("choices", [{}])[0]
        text = choice.get("message", {}).get("content", "")
        usage = data.get("usage", {})

        # sglang 的 speculative decode 在 server 端
        # accept rate 无法从 API 获取, 只能估算
        total_tokens = usage.get("completion_tokens", max_tokens)

        result = SpecResult(
            tokens=[],  # sglang 不返回 token ids
            accept_count=0,  # 无法获取
            total_count=total_tokens,
            output_text=text,
        )
        result.ttft_ms = 1000 * (t_end - t0)  # 粗略 (含网络)
        if total_tokens > 0:
            result.tps = total_tokens / (t_end - t0)
            result.speedup = result.tps / self.config.baseline_tps
        return result


# ============================================================================
# 工厂函数
# ============================================================================

def create_backend(config: SpecDecodeConfig) -> SpecDecodeBackend:
    """根据 config.backend 创建后端实例."""
    backends = {
        "mlx": MLXBackend,
        "pytorch": PyTorchBackend,
        "sglang": SGLangBackend,
    }
    if config.backend not in backends:
        raise ValueError(f"Unknown backend: {config.backend}. Supported: {list(backends.keys())}")
    return backends[config.backend](config)


# ============================================================================
# 自适应参数调优器 (AutoTuner)
# ============================================================================

@dataclass
class HardwareProfile:
    """硬件 profile — 描述硬件特性 + 最优参数."""
    backend: str
    device_name: str
    memory_gb: float
    compute_tier: str          # low | medium | high
    draft_fwd_ms: float        # draft forward 预估时间
    optimal_N: int             # 最优 num_draft_tokens
    optimal_mode: str          # chain | eagle
    optimal_dtype: str         # bfloat16 | float16 | int4
    baseline_tps: float        # 无投机 baseline

    def to_config(self) -> "SpecDecodeConfig":
        """从 hardware profile 生成最优配置."""
        return SpecDecodeConfig(
            backend=self.backend,
            mode=self.optimal_mode,
            num_draft_tokens=self.optimal_N,
            baseline_tps=self.baseline_tps,
        )


class AutoTuner:
    """自适应参数调优器 — 自动检测硬件 + 运行时动态调整."""

    # 静态硬件 profile 库 (实测数据)
    PROFILES: Dict[str, HardwareProfile] = {
        "mlx_m4_16gb": HardwareProfile(
            backend="mlx", device_name="Apple M4 16GB", memory_gb=16,
            compute_tier="low", draft_fwd_ms=4.0,
            optimal_N=16, optimal_mode="chain", optimal_dtype="int4",
            baseline_tps=26.8,
        ),
        "pytorch_rtx5000": HardwareProfile(
            backend="pytorch", device_name="RTX PRO 5000 72GB", memory_gb=72,
            compute_tier="high", draft_fwd_ms=1.0,
            optimal_N=4, optimal_mode="chain", optimal_dtype="bfloat16",
            baseline_tps=36.9,
        ),
        "sglang_cloud": HardwareProfile(
            backend="sglang", device_name="Cloud GPU (sglang)", memory_gb=72,
            compute_tier="high", draft_fwd_ms=0.0,  # server-side
            optimal_N=4, optimal_mode="chain", optimal_dtype="bfloat16",
            baseline_tps=121.2,
        ),
    }

    @classmethod
    def detect(cls, backend: str) -> HardwareProfile:
        """自动检测硬件 → 返回最优 profile.

        规则:
          - draft_fwd < 2ms → N=4 (GPU, draft 快)
          - draft_fwd 2-5ms → N=8 (中等)
          - draft_fwd > 5ms → N=16 (Mac, draft 慢, 需 batch 多分摊)
        """
        # 尝试匹配已知 profile
        if backend == "mlx":
            # Mac 检测
            try:
                import platform
                if "arm" in platform.machine().lower():
                    return cls.PROFILES["mlx_m4_16gb"]
            except Exception:
                pass
            return cls.PROFILES["mlx_m4_16gb"]

        elif backend == "pytorch":
            # GPU 检测
            try:
                import torch
                if torch.cuda.is_available():
                    name = torch.cuda.get_device_name(0)
                    if "PRO 5000" in name or "A100" in name or "H100" in name:
                        return cls.PROFILES["pytorch_rtx5000"]
                # 默认 GPU
                return cls.PROFILES["pytorch_rtx5000"]
            except Exception:
                return cls.PROFILES["pytorch_rtx5000"]

        elif backend == "sglang":
            return cls.PROFILES["sglang_cloud"]

        # 未知 backend, 用通用规则
        return HardwareProfile(
            backend=backend, device_name="unknown", memory_gb=0,
            compute_tier="medium", draft_fwd_ms=3.0,
            optimal_N=8, optimal_mode="chain", optimal_dtype="bfloat16",
            baseline_tps=26.8,
        )

    @classmethod
    def get_optimal_config(cls, backend: str, model_path: str = "") -> "SpecDecodeConfig":
        """自动检测硬件 → 返回最优 SpecDecodeConfig.

        Args:
            backend: mlx | pytorch | sglang
            model_path: 模型路径 (sglang 后端用于检测模型类型, 自动设置启动参数)
        """
        profile = cls.detect(backend)
        print(f"[autotune] Detected: {profile.device_name} (compute={profile.compute_tier})")
        print(f"[autotune] Optimal: N={profile.optimal_N}, mode={profile.optimal_mode}, dtype={profile.optimal_dtype}")
        print(f"[autotune] Expected: baseline={profile.baseline_tps} tok/s, draft_fwd={profile.draft_fwd_ms}ms")

        config = profile.to_config()

        # sglang 后端: 根据模型类型自动设置启动参数
        if backend == "sglang" and model_path:
            cls.apply_model_params(config, model_path)

        return config

    @classmethod
    def apply_model_params(cls, config: "SpecDecodeConfig", model_path: str):
        """根据模型类型自动设置 sglang 启动参数.

        V4-Flash (deepseek_v4): CGC_ENABLE_ORTHO_KDA=0 + cuda-graph + NEXTN + mem 0.7
        Qwen3-VL: 默认参数
        其他: 默认参数
        """
        import json, os
        config_path = os.path.join(model_path, "config.json")
        try:
            with open(config_path) as f:
                model_config = json.load(f)
        except Exception:
            print(f"[autotune] WARNING: cannot read {config_path}, using default sglang params")
            return

        model_type = model_config.get("model_type", "").lower()
        has_nextn = model_config.get("num_nextn_predict_layers", 0) > 0
        architectures = model_config.get("architectures", [])

        config.sglang_model_path = model_path

        if "deepseek" in model_type or "v4" in model_type:
            # === V4-Flash: cuda-graph + CGC_ENABLE_ORTHO_KDA=0 + NEXTN ===
            print(f"[autotune] Model: {model_type} (V4-Flash) → auto-setting sglang params:")
            config.sglang_env_vars = {"CGC_ENABLE_ORTHO_KDA": "0"}
            config.sglang_disable_cuda_graph = False  # 开启 cuda-graph
            config.sglang_mem_fraction = 0.7  # OOM 修复
            config.sglang_cuda_graph_max_bs = 16  # OOM 修复

            if has_nextn:
                # V4-Flash 内置 MTP, N=4 最优 (实测 N=2 被 sglang 转为 EAGLE, 反而慢 38%)
                print(f"[autotune]   ✓ cuda-graph (CGC GPU adapter, cuda-graph compatible)")
                print(f"[autotune]   ✓ NEXTN speculative (num_nextn_predict_layers={model_config['num_nextn_predict_layers']})")
                print(f"[autotune]   ✓ N=4 (实测最优, N=2 被 sglang 转 EAGLE 反而慢)")
                print(f"[autotune]   ✓ mem-fraction=0.7, cuda-graph-max-bs=16 (OOM fix)")
                config.sglang_speculative_algorithm = "NEXTN"
                config.sglang_extra_args = [
                    "--speculative-algorithm", "NEXTN",
                    "--speculative-num-steps", "4",      # N=4 实测最优
                    "--speculative-eagle-topk", "1",
                    "--speculative-num-draft-tokens", "16",
                ]
            else:
                print(f"[autotune]   ✓ cuda-graph (no NEXTN, model has no MTP)")
                config.sglang_speculative_algorithm = ""
        else:
            # === Qwen3-VL / 其他: 默认参数 ===
            print(f"[autotune] Model: {model_type} → default sglang params")
            config.sglang_disable_cuda_graph = False
            config.sglang_mem_fraction = 0.88

    @classmethod
    def generate_sglang_command(cls, config: "SpecDecodeConfig") -> str:
        """生成 sglang 启动命令 (包含自动设置的参数)."""
        parts = []

        # env vars
        env_str = " ".join(f"{k}={v}" for k, v in config.sglang_env_vars.items())

        # cgc wrapper or direct sglang
        if config.sglang_env_vars.get("CGC_ENABLE_ORTHO_KDA") == "0":
            parts.append(f"{env_str} python3 cgc_launch_dual_node.py")
        else:
            parts.append(f"{env_str} python3 -m sglang.launch_server")

        parts.extend([
            f"--model-path {config.sglang_model_path or config.target_model}",
            "--host 0.0.0.0",
            f"--port {config.sglang_target_url.split(':')[-1].split('/')[0]}",
            "--tp-size 8" if any(x in (config.sglang_model_path or "").lower() for x in ["deepseek", "v4", "flash"]) else "--tp 1",
            "--context-length 16384",
            f"--mem-fraction-static {config.sglang_mem_fraction}",
            f"--cuda-graph-max-bs {config.sglang_cuda_graph_max_bs}",
            "--trust-remote-code",
            "--skip-server-warmup",
        ])

        if not config.sglang_disable_cuda_graph:
            pass  # 不加 --disable-cuda-graph (默认开启)
        else:
            parts.append("--disable-cuda-graph")

        # speculative
        parts.extend(config.sglang_extra_args)

        return " ".join(parts)

    @classmethod
    def runtime_tune(cls, config: "SpecDecodeConfig", accept_rate: float, tps: float) -> "SpecDecodeConfig":
        """运行时根据 accept rate 动态调整 N.

        规则 (修正版, 针对 V4-Flash accept 28% 优化):
          - accept < 15% → N=0, 关闭投机 (draft 预测极差, 浪费算力)
          - accept 15-30% → N=2, 最小投机 (减少浪费, V4-Flash MTP 28% 在此区间)
          - accept 30-60% → 保持当前 N
          - accept > 60% → N 增大 (draft 预测好, 多 draft)

        成本-收益分析:
          - 如果投机 tok/s < plain tok/s * 0.95 → 投机有负收益, 关闭
        """
        old_N = config.num_draft_tokens

        if accept_rate < 0.15:
            # accept 极低, 关闭投机
            new_N = 0
            print(f"[autotune] accept={accept_rate:.0%} < 15%, 关闭投机 (plain mode, 避免浪费)")
        elif accept_rate < 0.30:
            # accept 低 (V4-Flash MTP 28%), 减到最小 N=2
            new_N = 2
            if new_N != old_N:
                print(f"[autotune] accept={accept_rate:.0%} < 30%, N={old_N}→{new_N} (最小投机, V4-Flash MTP 优化)")
        elif accept_rate > 0.60:
            new_N = min(32, old_N * 2)
            if new_N != old_N:
                print(f"[autotune] accept={accept_rate:.0%} > 60%, N={old_N}→{new_N} (增大)")
        else:
            new_N = old_N

        config.num_draft_tokens = new_N
        return config

    @classmethod
    def auto_bench(cls, backend: str, prompts: List[str], max_tokens: int = 30, model_path: str = "") -> Dict:
        """自动检测 + 自适应 bench: baseline → chain → EAGLE, 根据速度选最优.

        策略:
          1. 跑 chain speculative (默认最优 N)
          2. 如果 chain speedup < 1.5x → 尝试 EAGLE (只对 mlx/pytorch)
          3. 选 chain vs EAGLE 中速度最快的
          4. runtime_tune: accept 驱动动态调整 N

        Args:
            backend: mlx | pytorch | sglang
            model_path: 模型路径 (sglang: 自动检测模型类型, 设置启动参数)
        """
        config = cls.get_optimal_config(backend, model_path)
        backend_obj = create_backend(config)
        backend_obj.load()

        results = {}
        for prompt in prompts:
            print(f"\n  {prompt[:40]}...")

            # === 1. Chain speculative ===
            config.mode = "chain"
            backend_obj.config = config
            chain_result = backend_obj.generate(prompt, max_tokens)
            print(f"    [chain] tps={chain_result.tps:.1f}, accept={chain_result.accept_rate:.0%}, speedup={chain_result.speedup:.2f}x")

            # 运行时自适应 (chain)
            if chain_result.accept_rate > 0:
                tuned_config = cls.runtime_tune(config, chain_result.accept_rate, chain_result.tps)
                if tuned_config.num_draft_tokens != config.num_draft_tokens:
                    config = tuned_config
                    backend_obj.config = config
                    print(f"    [retune] N={config.num_draft_tokens}")
                    chain_result = backend_obj.generate(prompt, max_tokens)
                    print(f"    [chain retuned] tps={chain_result.tps:.1f}, accept={chain_result.accept_rate:.0%}")

            best = chain_result
            best_mode = "chain"

            # === 2. EAGLE (只对 mlx/pytorch, sglang 不支持客户端 EAGLE) ===
            if backend in ("mlx", "pytorch") and chain_result.speedup < 1.5:
                print(f"    [chain speedup < 1.5x, trying EAGLE...]")
                eagle_config = SpecDecodeConfig(
                    backend=backend, mode="eagle",
                    num_draft_tokens=config.num_draft_tokens,
                    top_k=4, tree_depth=1,
                    baseline_tps=config.baseline_tps,
                    mlx_target_path=config.mlx_target_path,
                    mlx_draft_model=config.mlx_draft_model,
                    pytorch_target_path=config.pytorch_target_path,
                    pytorch_draft_model=config.pytorch_draft_model,
                    pytorch_device=config.pytorch_device,
                    pytorch_dtype=config.pytorch_dtype,
                    sglang_target_url=config.sglang_target_url,
                )
                try:
                    eagle_backend = create_backend(eagle_config)
                    # 复用已加载的模型 (避免重新加载)
                    eagle_backend.target_model = backend_obj.target_model
                    eagle_backend.tokenizer = backend_obj.tokenizer
                    eagle_backend.draft_model = backend_obj.draft_model
                    eagle_result = eagle_backend.generate(prompt, max_tokens)
                    print(f"    [eagle] tps={eagle_result.tps:.1f}, accept={eagle_result.accept_rate:.0%}, speedup={eagle_result.speedup:.2f}x")

                    if eagle_result.tps > best.tps:
                        best = eagle_result
                        best_mode = "eagle"
                        print(f"    [EAGLE wins!]")
                    else:
                        print(f"    [chain wins]")
                except Exception as e:
                    print(f"    [eagle error: {e}]")

            results[prompt[:30]] = best
            print(f"    => BEST: {best_mode} tps={best.tps:.1f} ({best.speedup:.2f}x, accept {best.accept_rate:.0%})")

        return results


# ============================================================================
# CLI 入口
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="统一投机 decode IR — 跨后端")
    parser.add_argument("--backend", default="mlx", choices=["mlx", "pytorch", "sglang"])
    parser.add_argument("--mode", default="chain", choices=["chain", "eagle"])
    parser.add_argument("--num-draft", type=int, default=16, help="chain N (16 最优)")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--tree-depth", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--save-config", default="", help="保存配置到 JSON")
    parser.add_argument("--load-config", default="", help="从 JSON 加载配置")
    parser.add_argument("--prompt", default="", help="单 prompt 测试")
    parser.add_argument("--auto", action="store_true", help="自动检测硬件 + 运行时自适应调优")
    parser.add_argument("--model-path", default="", help="模型路径 (sglang: 自动检测模型类型, 设置启动参数)")
    parser.add_argument("--show-launch", action="store_true", help="只显示 sglang 启动命令 (不跑 bench)")
    args = parser.parse_args()

    # --show-launch: 只生成 sglang 启动命令
    if args.show_launch and args.model_path:
        config = AutoTuner.get_optimal_config("sglang", args.model_path)
        cmd = AutoTuner.generate_sglang_command(config)
        print(f"\n[sglang launch command]")
        print(f"  {cmd}")
        return

    # --auto 模式: 自动检测硬件 → 最优参数 → 运行时自适应
    if args.auto:
        prompts = [
            "Write a short story about a cat",
            "Explain how photosynthesis works in simple terms",
            "What are the benefits of exercise?",
        ] if not args.prompt else [args.prompt]

        print(f"[auto] Backend={args.backend}, auto-detecting hardware...")
        if args.model_path:
            print(f"[auto] Model path: {args.model_path}")
        results = AutoTuner.auto_bench(args.backend, prompts, args.max_tokens, model_path=args.model_path)

        print(f"\n{'='*60}")
        print(f"汇总 (auto-tuned {args.backend})")
        print(f"{'='*60}")
        for prompt, res in results.items():
            print(f"  {prompt}: {res.tps:.1f} tok/s ({res.speedup:.2f}x, accept {res.accept_rate:.0%})")
        return

    # 加载或创建配置
    if args.load_config:
        config = SpecDecodeConfig.from_json(args.load_config)
    else:
        config = SpecDecodeConfig(
            backend=args.backend,
            mode=args.mode,
            num_draft_tokens=args.num_draft,
            top_k=args.top_k,
            tree_depth=args.tree_depth,
        )

    if args.save_config:
        config.to_json(args.save_config)
        print(f"Config saved to {args.save_config}")
        return

    print(f"[config] backend={config.backend}, mode={config.mode}, N={config.num_draft_tokens}")

    # 创建后端
    backend = create_backend(config)
    backend.load()

    # 测试
    if args.prompt:
        result = backend.generate(args.prompt, args.max_tokens)
        print(f"\nTTFT: {result.ttft_ms:.0f}ms")
        print(f"Decode: {result.tps:.1f} tok/s ({result.speedup:.2f}x)")
        print(f"Accept: {result.accept_rate:.0%} ({result.accept_count}/{result.total_count})")
        print(f"Output: {result.output_text[:100]}")
    else:
        prompts = [
            "Write a short story about a cat",
            "Explain how photosynthesis works in simple terms",
            "What are the benefits of exercise?",
        ]
        print("\n[bench]")
        results = backend.bench(prompts, args.max_tokens)

        print(f"\n{'='*60}")
        print(f"汇总 ({config.backend}/{config.mode} N={config.num_draft_tokens})")
        print(f"{'='*60}")
        for prompt, res in results.items():
            print(f"  {prompt}: {res.tps:.1f} tok/s ({res.speedup:.2f}x, accept {res.accept_rate:.0%})")


if __name__ == "__main__":
    main()
