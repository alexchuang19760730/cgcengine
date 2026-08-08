#!/usr/bin/env python3
"""oMLX+FlashMoE: mlx_lm wrapper with expert streaming injection.

方案 A: 包装 mlx_lm 的真实 forward，在 MoE 层用 StreamingSwitchGLU 做 expert swap.

核心组件:
1. StreamingExpertManager: 管理 expert 的 hot/warm/cold 缓存, 统计命中率
2. StreamingSwitchGLU: 包装 SwitchGLU, 在 __call__ 前触发 ensure_experts
3. OMLXMLXEngine: 推理引擎, 加载模型, 替换 SwitchGLU, 跑 generate
"""
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx_lm


# ─── Streaming Expert Manager ────────────────────────────────────────

@dataclass
class StreamingStats:
    """Expert streaming 统计."""
    total_calls: int = 0
    total_expert_hits: int = 0
    total_expert_misses: int = 0
    total_swaps: int = 0
    total_swap_time_ms: float = 0.0
    # per-layer 统计
    per_layer_calls: Dict[int, int] = field(default_factory=dict)
    per_layer_misses: Dict[int, int] = field(default_factory=dict)

    @property
    def hit_rate(self) -> float:
        total = self.total_expert_hits + self.total_expert_misses
        return self.total_expert_hits / total if total > 0 else 0.0

    def summary(self) -> str:
        return (
            f"calls={self.total_calls} "
            f"hits={self.total_expert_hits} misses={self.total_expert_misses} "
            f"hit_rate={self.hit_rate:.2%} "
            f"swaps={self.total_swaps} "
            f"swap_time={self.total_swap_time_ms:.1f}ms"
        )


class StreamingExpertManager:
    """管理 MoE expert 的 hot/warm/cold 缓存.

    模拟真实 streaming 场景:
    - hot_cache: 常驻内存的 expert (max_experts_in_memory 个)
    - 当需要的 expert 不在 hot_cache 时, 触发 swap (LRU 淘汰)
    - swap_time_per_expert: 模拟单个 expert 的 I/O 加载延迟 (ms)

    优化模式 (lazy_stats=True):
    - 不在每次 __call__ 触发 MLX 同步 (避免 tolist 开销)
    - 累积 indices 到 pending, 在 step 边界批量 flush
    - LRU 语义: 用 flush 时刻的 expert 集合更新, 仍保留命中率统计
    """

    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        max_experts_in_memory: int = 2,  # 每层最多常驻几个 expert
        swap_time_per_expert_ms: float = 0.0,  # I/O 延迟模拟 (0 = 不模拟)
        enable_io_simulation: bool = False,
        lazy_stats: bool = True,  # 延迟统计模式 (避免 tolist 同步点)
        stats_mode: str = "sample",  # off | sample | full
        sample_interval: int = 10,  # sample 模式下每 N token 采样一次
        true_swap: bool = False,  # True Expert Swap: 每层后 eval+clear_cache 释放 GPU 内存
    ):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.max_experts_in_memory = max_experts_in_memory
        self.swap_time_per_expert_ms = swap_time_per_expert_ms
        self.enable_io_simulation = enable_io_simulation
        self.lazy_stats = lazy_stats
        # stats_mode: off=零开销(生产), sample=采样统计, full=每token精确统计
        self.stats_mode = stats_mode
        self.sample_interval = max(1, sample_interval)
        self._step_counter = 0  # 用于 sample 模式计数
        # True Expert Swap: 每层 MoE 计算后 mx.eval + clear_cache
        # 用于大模型 (>GPU 内存) 场景, 牺牲性能换取不 OOM
        self.true_swap = true_swap

        # per-layer LRU cache: layer_idx -> OrderedDict[expert_id -> True]
        self._caches: Dict[int, OrderedDict] = {
            l: OrderedDict() for l in range(num_layers)
        }
        self.stats = StreamingStats()

        # lazy 模式: 累积 pending indices (layer_idx -> list of mx.array)
        self._pending: Dict[int, List[mx.array]] = {
            l: [] for l in range(num_layers)
        }

    def ensure_experts(self, layer_idx: int, expert_ids: List[int]) -> int:
        """确保指定 expert 已加载到缓存, 返回 miss 数. (eager 模式)"""
        cache = self._caches[layer_idx]
        misses = 0

        for eid in expert_ids:
            if eid in cache:
                # hit: move to end (LRU)
                cache.move_to_end(eid)
                self.stats.total_expert_hits += 1
            else:
                # miss: 需要加载
                misses += 1
                self.stats.total_expert_misses += 1

                # LRU 淘汰
                while len(cache) >= self.max_experts_in_memory:
                    cache.popitem(last=False)  # evict oldest
                    self.stats.total_swaps += 1

                # 加载
                cache[eid] = True
                self.stats.total_swaps += 1

                # 模拟 I/O 延迟
                if self.enable_io_simulation and self.swap_time_per_expert_ms > 0:
                    time.sleep(self.swap_time_per_expert_ms / 1000.0)
                    self.stats.total_swap_time_ms += self.swap_time_per_expert_ms

        self.stats.total_calls += 1
        self.stats.per_layer_calls[layer_idx] = (
            self.stats.per_layer_calls.get(layer_idx, 0) + 1
        )
        self.stats.per_layer_misses[layer_idx] = (
            self.stats.per_layer_misses.get(layer_idx, 0) + misses
        )

        return misses

    def record_indices(self, layer_idx: int, indices: mx.array) -> None:
        """延迟模式: 记录 indices, 不触发 tolist 同步.

        stats_mode:
        - off: 完全不记录 (零开销, 生产用)
        - sample: 只在采样窗口内记录
        - full: 每次都记录
        """
        if self.stats_mode == "off":
            return
        # sample/full 都 append (flush 时按 interval 控制频率)
        self._pending[layer_idx].append(indices)

    def maybe_flush(self) -> None:
        """在 generate_step 边界调用, 按 stats_mode 决定是否 flush.

        - off: 无操作
        - sample: 每 sample_interval 步 flush 一次
        - full: 每步都 flush
        """
        if self.stats_mode == "off":
            return
        self._step_counter += 1
        if self.stats_mode == "full":
            self.flush_pending()
        elif self.stats_mode == "sample" and self._step_counter % self.sample_interval == 0:
            self.flush_pending()

    def flush_pending(self) -> None:
        """批量 flush 累积的 indices, 更新 LRU + 统计.

        在 generate_step 边界调用, 一次性 eval 所有 pending indices.
        相比每次 __call__ 调 tolist, 只触发一次 MLX 同步.
        """
        for layer_idx, pending_list in self._pending.items():
            if not pending_list:
                continue
            # 合并所有 pending indices, 一次性 tolist
            combined = mx.concatenate([p.flatten() for p in pending_list])
            expert_ids = list(set(combined.tolist()))
            expert_ids = [int(e) for e in expert_ids]

            # 更新 LRU + 统计 (批量)
            cache = self._caches[layer_idx]
            misses = 0
            for eid in expert_ids:
                if eid in cache:
                    cache.move_to_end(eid)
                    self.stats.total_expert_hits += 1
                else:
                    misses += 1
                    self.stats.total_expert_misses += 1
                    while len(cache) >= self.max_experts_in_memory:
                        cache.popitem(last=False)
                        self.stats.total_swaps += 1
                    cache[eid] = True
                    self.stats.total_swaps += 1
                    if self.enable_io_simulation and self.swap_time_per_expert_ms > 0:
                        time.sleep(self.swap_time_per_expert_ms / 1000.0)
                        self.stats.total_swap_time_ms += self.swap_time_per_expert_ms

            self.stats.total_calls += len(pending_list)
            self.stats.per_layer_calls[layer_idx] = (
                self.stats.per_layer_calls.get(layer_idx, 0) + len(pending_list)
            )
            self.stats.per_layer_misses[layer_idx] = (
                self.stats.per_layer_misses.get(layer_idx, 0) + misses
            )
            pending_list.clear()

    def get_stats(self) -> StreamingStats:
        return self.stats


# ─── Streaming SwitchGLU ─────────────────────────────────────────────

class StreamingSwitchGLU(nn.Module):
    """包装 SwitchGLU, 在 forward 前触发 expert streaming.

    继承自 nn.Module 而非 SwitchGLU, 因为我们需要包装一个已有的 SwitchGLU 实例.
    保持 SwitchGLU 的所有权重和计算逻辑不变, 只在 __call__ 前注入 streaming 逻辑.
    """

    def __init__(
        self,
        original_switch_glu: Any,
        layer_idx: int,
        manager: StreamingExpertManager,
    ):
        super().__init__()
        # 复制原 SwitchGLU 的所有子模块和参数
        self.gate_proj = original_switch_glu.gate_proj
        self.up_proj = original_switch_glu.up_proj
        self.down_proj = original_switch_glu.down_proj
        self.activation = original_switch_glu.activation

        self.layer_idx = layer_idx
        self.manager = manager

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        """重写 __call__, 注入 expert streaming 逻辑.

        lazy 模式: 只记录 indices (不触发 tolist 同步), forward 逻辑完全复用 SwitchGLU
        eager 模式: 立即 tolist + ensure_experts (会触发 MLX 同步, 开销大)

        True Expert Swap: 每层计算后 mx.eval + clear_cache, 释放 GPU 内存.
        避免 30 层权重同时 wire 到 GPU (解决 OOM).
        """
        # 1. streaming 统计
        if self.manager.lazy_stats:
            self.manager.record_indices(self.layer_idx, indices)
        else:
            unique_experts = list(set(indices.flatten().tolist()))
            unique_experts = [int(e) for e in unique_experts]
            self.manager.ensure_experts(self.layer_idx, unique_experts)

        # 2. 调用原 SwitchGLU forward 逻辑 (完全复用)
        from mlx_lm.models.switch_layers import (
            SwitchGLU,
            _gather_sort,
            _scatter_unsort,
        )

        x = mx.expand_dims(x, (-2, -3))

        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)

        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        x = self.down_proj(
            self.activation(x_up, x_gate),
            idx,
            sorted_indices=do_sort,
        )

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        out = x.squeeze(-2)

        # True Expert Swap 的 eval+clear_cache 由 SwapDecoderLayer 在 decoder layer
        # 级别统一控制 (swap_interval), 避免每层 MoE 都同步 (双重 eval 开销大).

        return out


# ─── OMLX MLX Engine ─────────────────────────────────────────────────

class OMLXMLXEngine:
    """oMLX+FlashMoE 推理引擎.

    包装 mlx_lm 的 load + generate, 注入 StreamingSwitchGLU 实现 expert streaming.

    用法:
        engine = OMLXMLXEngine(model_path, streaming_config={...})
        engine.load()
        text = engine.generate("Hello", max_tokens=100)
    """

    def __init__(
        self,
        model_path: str,
        streaming_config: Optional[Dict[str, Any]] = None,
        enable_streaming: bool = True,
    ):
        self.model_path = model_path
        self.enable_streaming = enable_streaming
        self.streaming_config = streaming_config or {}

        self.model: Any = None
        self.tokenizer: Any = None
        self.manager: Optional[StreamingExpertManager] = None
        self._is_moe: bool = False
        self._num_layers: int = 0
        self._num_experts: int = 0

    def load(self) -> "OMLXMLXEngine":
        """加载模型并注入 streaming."""
        # 1. 用 mlx_lm 加载模型
        self.model, self.tokenizer = mlx_lm.load(self.model_path, lazy=True)

        # 2. 检查是否是 MoE 模型
        self._detect_moe()

        # 3. 如果是 MoE 且启用 streaming, 注入 StreamingSwitchGLU
        if self._is_moe and self.enable_streaming:
            self._inject_streaming()

        return self

    def _get_layers(self) -> List[Any]:
        """获取模型 layers, 兼容多种结构.

        - 纯文本: model.model.layers (Qwen3MoE 等)
        - Gemma4 多模态: model.layers (顶层直接暴露)
        - 其他多模态: model.language_model.layers
        """
        # Gemma4: 顶层 Model 直接有 layers
        if hasattr(self.model, "layers") and isinstance(self.model.layers, list):
            return self.model.layers
        # 标准: model.model.layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        # 多模态: model.language_model.layers
        if hasattr(self.model, "language_model") and hasattr(self.model.language_model, "layers"):
            return self.model.language_model.layers
        raise AttributeError("Cannot find layers in model")

    def _detect_moe(self) -> None:
        """检测模型是否包含 MoE 层.

        支持两种格式:
        - Qwen3MoE: layer.mlp.switch_mlp.switch_glu
        - Gemma4:   layer.experts.switch_glu (且 layer.enable_moe=True)
        """
        layers = self._get_layers()
        self._num_layers = len(layers)

        # 检查每层是否有 MoE 结构
        moe_layer_count = 0
        for layer in layers:
            mlp = getattr(layer, "mlp", None)
            experts = getattr(layer, "experts", None)

            # Qwen3MoE 格式: mlp.switch_mlp
            if mlp is not None and hasattr(mlp, "switch_mlp"):
                self._is_moe = True
                if self._num_experts == 0:
                    self._num_experts = mlp.num_experts
                moe_layer_count += 1
            # Gemma4 格式: experts.switch_glu (且 enable_moe)
            elif experts is not None and hasattr(experts, "switch_glu"):
                self._is_moe = True
                if self._num_experts == 0:
                    sglu = experts.switch_glu
                    # num_experts 不一定直接暴露, 从权重 shape 推断
                    if hasattr(sglu, "num_experts"):
                        self._num_experts = sglu.num_experts
                    elif hasattr(sglu, "gate_proj") and hasattr(sglu.gate_proj, "weight"):
                        self._num_experts = sglu.gate_proj.weight.shape[0]
                moe_layer_count += 1

        self._moe_layer_count = moe_layer_count

    def _inject_streaming(self) -> None:
        """将所有 MoE 层的 SwitchGLU 替换为 StreamingSwitchGLU.

        支持:
        - Qwen3MoE: mlp.switch_mlp -> StreamingSwitchGLU
        - Gemma4:   experts.switch_glu -> StreamingSwitchGLU
        """
        # 创建 streaming manager
        self.manager = StreamingExpertManager(
            num_layers=self._num_layers,
            num_experts=self._num_experts,
            max_experts_in_memory=self.streaming_config.get(
                "max_experts_in_memory", 2
            ),
            swap_time_per_expert_ms=self.streaming_config.get(
                "swap_time_per_expert_ms", 0.0
            ),
            enable_io_simulation=self.streaming_config.get(
                "enable_io_simulation", False
            ),
            lazy_stats=self.streaming_config.get("lazy_stats", True),
            stats_mode=self.streaming_config.get("stats_mode", "sample"),
            sample_interval=self.streaming_config.get("sample_interval", 10),
            true_swap=self.streaming_config.get("true_swap", False),
        )

        # 替换每一层的 SwitchGLU
        layers = self._get_layers()
        injected = 0
        for l_idx, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None)
            experts = getattr(layer, "experts", None)

            # Qwen3MoE 格式
            if mlp is not None and hasattr(mlp, "switch_mlp"):
                original = mlp.switch_mlp
                mlp.switch_mlp = StreamingSwitchGLU(
                    original_switch_glu=original,
                    layer_idx=l_idx,
                    manager=self.manager,
                )
                injected += 1
            # Gemma4 格式
            elif experts is not None and hasattr(experts, "switch_glu"):
                original = experts.switch_glu
                experts.switch_glu = StreamingSwitchGLU(
                    original_switch_glu=original,
                    layer_idx=l_idx,
                    manager=self.manager,
                )
                injected += 1

        print(f"[oMLX] Injected StreamingSwitchGLU into {injected}/{self._num_layers} layers")

        # True Swap: 包装每层 __call__, 每层后 eval + clear_cache
        # 解决大模型 (>GPU wired_limit) OOM 问题
        if self.manager.true_swap:
            swap_interval = self.streaming_config.get("swap_interval", 10)
            self._wrap_layers_for_swap(layers, swap_interval)
            print(f"[oMLX] True Expert Swap enabled (interval={swap_interval} layers)")

    def _wrap_layers_for_swap(self, layers: List[Any], swap_interval: int = 10) -> None:
        """创建 SwapDecoderLayer 类, 每 N 层后 mx.eval + clear_cache.

        目的: 拆分 MLX lazy 计算图, 让 GPU 每 N 层后释放权重 wire.
        N 层权重约 (N/30 * 13.8GB), 需 < wired_limit (8GB).
        - N=1: 最安全但最慢 (每层同步)
        - N=10: 10层~4.6GB, 平衡安全与性能
        - N=30: 不拆分 (会 OOM)

        注意: Python __call__ 是类方法, 必须替换 __class__ 而非实例属性.
        """
        orig_class = type(layers[0])
        orig_call = orig_class.__call__
        counter = [0]

        def swap_call(self, x, *args, **kwargs):
            out = orig_call(self, x, *args, **kwargs)
            counter[0] += 1
            if counter[0] % swap_interval == 0:
                # Gemma4 DecoderLayer 回傳 (h, shared_kv, offset)
                # shared_kv = (keys, values) 是 KV cache, 必須一起 eval
                # 否則下層 (KV-shared layer) 會讀到 stale/lazy KV -> garbage 輸出
                if isinstance(out, tuple) and len(out) == 3:
                    h, shared_kv, offset = out
                    if shared_kv is not None:
                        keys, values = shared_kv
                        mx.eval(h, keys, values)
                    else:
                        mx.eval(h)
                else:
                    mx.eval(out)
            return out

        # 创建新类, 避免修改原始 mlx_lm 类
        SwapDecoderLayer = type(
            "SwapDecoderLayer",
            (orig_class,),
            {"__call__": swap_call},
        )
        for layer in layers:
            layer.__class__ = SwapDecoderLayer

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        verbose: bool = False,
    ) -> str:
        """生成文本."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # true_swap 模式: 用 generate_step 绕过 mlx_lm 的 wired_limit 上下文管理器
        # (mlx_lm 会把 wired_limit 设为 max_recommended 11.5GB, 覆盖我们的 8GB)
        if self.manager and self.manager.true_swap:
            return self._generate_with_swap(prompt, max_tokens)

        # 正常模式: 用 mlx_lm 的 stream_generate + maybe_flush
        from mlx_lm.generate import stream_generate
        final_text = ""
        for resp in stream_generate(
            self.model, self.tokenizer, prompt, max_tokens=max_tokens
        ):
            final_text += str(getattr(resp, "text", "") or "")
            if self.manager and self.manager.lazy_stats:
                self.manager.maybe_flush()
        if self.manager and self.manager.lazy_stats and self.manager.stats_mode != "off":
            self.manager.flush_pending()
        return final_text

    def _generate_with_swap(self, prompt: str, max_tokens: int) -> str:
        """true_swap 模式下的生成, 绕过 mlx_lm 的 wired_limit 上下文管理器.

        直接用 generate_step, 它不修改 wired_limit.
        配合 SwapDecoderLayer 的每层 eval+clear_cache, 控制内存.
        """
        from mlx_lm.generate import generate_step
        import mlx.nn as nn

        # tokenize (用 chat template, Gemma4 需要 <bos> 和 turn 格式)
        if hasattr(self.tokenizer, 'chat_template'):
            msgs = [{'role': 'user', 'content': prompt}]
            formatted = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            tokens = self.tokenizer.encode(formatted)
        elif hasattr(self.tokenizer, 'encode'):
            tokens = self.tokenizer.encode(prompt)
        else:
            tokens = self.tokenizer(prompt)

        prompt_arr = mx.array(tokens)

        final_text = ""
        n = 0
        for token, logprobs in generate_step(
            prompt_arr, self.model, max_tokens=max_tokens
        ):
            # token 是 mx.array 標量, 需轉 int
            token_id = int(token) if isinstance(token, mx.array) else int(token)
            final_text += self.tokenizer.decode([token_id])
            n += 1
            if self.manager and self.manager.lazy_stats:
                self.manager.maybe_flush()
        if self.manager and self.manager.lazy_stats and self.manager.stats_mode != "off":
            self.manager.flush_pending()
        return final_text

    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 100,
    ):
        """流式生成 (yield token)."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        from mlx_lm.generate import stream_generate
        for resp in stream_generate(
            self.model, self.tokenizer, prompt, max_tokens=max_tokens
        ):
            # 按 stats_mode 决定是否 flush
            if self.manager and self.manager.lazy_stats:
                self.manager.maybe_flush()
            yield resp
        # 最后 flush 一次
        if self.manager and self.manager.lazy_stats and self.manager.stats_mode != "off":
            self.manager.flush_pending()

    def get_stats(self) -> Optional[StreamingStats]:
        """获取 streaming 统计."""
        if self.manager is None:
            return None
        return self.manager.get_stats()

    def benchmark(
        self,
        prompt: str,
        max_tokens: int = 50,
        warmup: int = 2,
    ) -> Dict[str, Any]:
        """基准测试, 返回 TTFT, tps, streaming stats."""
        import time

        # warmup
        for _ in range(warmup):
            self.generate(prompt, max_tokens=10)

        # 重置 stats
        if self.manager:
            self.manager.stats = StreamingStats()

        # 测量
        t0 = time.perf_counter()
        text = self.generate(prompt, max_tokens=max_tokens)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        tps = max_tokens / elapsed if elapsed > 0 else 0
        ttft = elapsed  # 简化: TTFT ≈ total time / (1 + max_tokens) 近似

        result = {
            "text": text[:100],
            "max_tokens": max_tokens,
            "elapsed_s": elapsed,
            "tps": tps,
            "ttft_s": ttft / (max_tokens + 1),  # 近似
            "streaming_stats": (
                self.manager.get_stats().summary() if self.manager else "N/A"
            ),
        }
        return result


# ─── CLI ─────────────────────────────────────────────────────────────

def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="oMLX+FlashMoE engine")
    parser.add_argument(
        "--model", default="models/test_moe_small", help="Model path"
    )
    parser.add_argument("--prompt", default="Hello, world!", help="Prompt")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument(
        "--mode",
        choices=["bypass", "streaming"],
        default="streaming",
        help="bypass = 全量加载; streaming = expert swap",
    )
    parser.add_argument(
        "--max-experts-in-memory",
        type=int,
        default=2,
        help="每层最多常驻 expert 数 (streaming mode)",
    )
    parser.add_argument(
        "--simulate-io-ms",
        type=float,
        default=0.0,
        help="模拟每个 expert swap 的 I/O 延迟 (ms)",
    )
    parser.add_argument(
        "--enable-io-simulation",
        action="store_true",
        help="启用 I/O 延迟模拟",
    )
    parser.add_argument(
        "--lazy-stats",
        choices=["on", "off"],
        default="on",
        help="延迟统计模式 (on=step边界flush, off=每次call tolist)",
    )
    parser.add_argument(
        "--stats-mode",
        choices=["off", "sample", "full"],
        default="sample",
        help="off=零开销(生产), sample=采样统计, full=每token精确统计",
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=10,
        help="sample 模式下每 N token 采样一次",
    )
    args = parser.parse_args()

    enable_streaming = args.mode == "streaming"
    lazy_stats = args.lazy_stats == "on"

    print(f"Mode: {args.mode}")
    print(f"Model: {args.model}")
    print(f"Streaming: {enable_streaming}, lazy: {lazy_stats}, stats_mode: {args.stats_mode}, sample_interval: {args.sample_interval}")

    engine = OMLXMLXEngine(
        model_path=args.model,
        enable_streaming=enable_streaming,
        streaming_config={
            "max_experts_in_memory": args.max_experts_in_memory,
            "swap_time_per_expert_ms": args.simulate_io_ms,
            "enable_io_simulation": args.enable_io_simulation,
            "lazy_stats": lazy_stats,
            "stats_mode": args.stats_mode,
            "sample_interval": args.sample_interval,
        },
    )

    engine.load()

    print(f"MoE: {engine._is_moe}")
    if engine._is_moe:
        print(f"Layers: {engine._num_layers} (MoE: {engine._moe_layer_count}), Experts: {engine._num_experts}")

    result = engine.benchmark(args.prompt, max_tokens=args.max_tokens)

    print("\n--- Benchmark Result ---")
    print(f"Mode: {args.mode}")
    print(f"Elapsed: {result['elapsed_s']:.3f}s")
    print(f"TPS: {result['tps']:.1f}")
    print(f"TTFT (approx): {result['ttft_s']*1000:.1f}ms")
    print(f"Streaming stats: {result['streaming_stats']}")
    print(f"Output: {result['text']}")


if __name__ == "__main__":
    main()
