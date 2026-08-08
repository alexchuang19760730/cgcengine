#!/usr/bin/env python3
"""OMLX Runtime -- 端侧 by-layer 流式推理引擎.

oMLX = optimized MLX, 在 Apple MLX 之上增加:
1. 层预取: forward 第 N 层时, 异步加载第 N+2/N+3 层
2. MoE 切层: 对 MoE 模型只保留 router + top-k 专家
3. 内存池: 权重 (LayerSwapPool) / KV cache / 激活 分池管理
4. 与 Hermes 路由接口: route.use_flashmoe=True 时启用 FlashMoE

关键接口:
  forward_layer_streaming(layer_idx, hidden, kv) → (hidden, kv)
  forward_draft(model_name, prompt_ids, draft_n, pivot_layer) → DraftResult
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional, AsyncIterator

from app.edge_engine.layer_swap_pool import (
    LayerSwapPool,
    KVCachePool,
    ActivationPool,
    ExpertCache,
)

logger = logging.getLogger(__name__)


@dataclass
class DraftResult:
    """Draft 生成结果."""
    tokens: list[int] = field(default_factory=list)
    hidden_states: list[Any] = field(default_factory=list)  # 每步的 hidden
    pivot_token: Optional[int] = None                       # pivot 层抢出的首 token
    pivot_layer: int = -1                                   # pivot 层索引
    latency_ms: float = 0.0
    prefill_ms: float = 0.0
    draft_ms: float = 0.0
    success: bool = False
    error: str = ""


@dataclass
class DraftEvent:
    """流式 draft 事件 (for async iterator)."""
    type: str                   # "first_token" / "draft_token" / "draft_sequence" / "error"
    token_id: int = -1
    sequence: list[int] = field(default_factory=list)
    pivot_layer: int = -1
    latency_ms: float = 0.0
    error: str = ""


class OMLXRuntime:
    """oMLX 端侧推理运行时 -- by-layer MoE 切层.

    用法:
        runtime = OMLXRuntime(model_path, layer_swap_config={...})
        runtime.load_model_metadata()
        result = runtime.forward_draft("gemma4", prompt_ids, draft_n=8, pivot_layer=6)
    """

    def __init__(
        self,
        model_path: str = "",
        layer_swap_config: Optional[dict] = None,
        dtype: str = "bf16",
    ):
        """初始化 oMLX Runtime.

        Args:
            model_path: 模型权重路径 (safetensors / MLX 格式)
            layer_swap_config: 层交换池配置
            dtype: 权重精度 ("bf16" / "int4" / "int8")
        """
        self.model_path = model_path
        self.dtype = dtype

        config = layer_swap_config or {}
        self.swap_pool = LayerSwapPool(
            hot_slots=config.get("hot_slots", 4),
            warm_slots=config.get("warm_slots", 8),
            cold_swap_ms=config.get("cold_swap_ms", 20.0),
            prefetch_depth=config.get("prefetch_depth", 2),
        )
        self.kv_cache = KVCachePool(
            max_gb=config.get("kv_cache_gb", 4.0),
        )
        self.activation_pool = ActivationPool(
            max_gb=config.get("activation_gb", 2.0),
        )
        self.expert_cache = ExpertCache(
            keep_top_k=config.get("expert_top_k", 2),
            eviction="lru",
        )

        # 模型元数据
        self.num_layers: int = 0
        self.hidden_size: int = 0
        self.vocab_size: int = 0
        self.is_moe: bool = False
        self.num_experts: int = 0
        self.experts_per_tok: int = 0
        self.embed_weight: Any = None
        self.lm_head_weight: Any = None

        # 后端: MLX (Apple Silicon)
        self._backend = "mlx"
        self._mlx_model: Any = None
        self._flashmoe: Any = None  # FlashMoEByLayer (if MoE)

        # 加载函数注入
        self.swap_pool.set_load_function(
            self._load_layer_weight,
            self._get_layer_size_mb,
        )

        self._initialized = False

    def load_model_metadata(self, model_info: Optional[Any] = None):
        """加载模型元数据 (不加载权重).

        Args:
            model_info: ModelInfo 或类似对象 (num_layers, hidden_size, etc.)
        """
        if model_info is not None:
            self.num_layers = getattr(model_info, "num_layers", 0)
            self.hidden_size = getattr(model_info, "hidden_size", 0)
            self.vocab_size = getattr(model_info, "vocab_size", 0)
            self.is_moe = getattr(model_info, "is_moe", False)
            self.num_experts = getattr(model_info, "num_experts", 0)
            self.experts_per_tok = getattr(model_info, "experts_per_tok", 0)

        # 尝试从 config.json 读取
        if self.model_path and os.path.exists(self.model_path):
            import json
            config_path = os.path.join(self.model_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    cfg = json.load(f)
                self.num_layers = self.num_layers or cfg.get("num_hidden_layers", 0)
                self.hidden_size = self.hidden_size or cfg.get("hidden_size", 0)
                self.vocab_size = self.vocab_size or cfg.get("vocab_size", 0)
                self.is_moe = self.is_moe or cfg.get("is_moe", False)
                self.num_experts = self.num_experts or cfg.get("num_experts", 0)
                self.experts_per_tok = self.experts_per_tok or cfg.get("num_experts_per_tok", 0)

        # 分配 KV cache
        self.kv_cache.allocate(self.num_layers, self.hidden_size)

        # 固定关键层到 hot pool (embedding + 早期层)
        critical_layers = list(range(min(4, self.num_layers)))
        self.swap_pool.pin_hot(critical_layers)

        self._initialized = True
        logger.info(
            f"[omlx] Model metadata loaded: {self.num_layers} layers, "
            f"hidden={self.hidden_size}, vocab={self.vocab_size}, "
            f"moe={self.is_moe} ({self.num_experts} experts, top-{self.experts_per_tok})"
        )

    def _load_layer_weight(self, layer_idx: int) -> Optional[dict]:
        """加载单层权重 (由 LayerSwapPool 调用).

        生产环境: 从 safetensors / mmap 加载
        Mock 模式: 无 model_path 时返回 mock 权重 (用于测试)
        """
        if not self.model_path or not os.path.exists(self.model_path):
            # Mock 模式: 返回 dummy 权重
            return {"layer": layer_idx, "weight": f"mock_layer_{layer_idx}",
                    "hidden_size": self.hidden_size}

        try:
            # 尝试 MLX 权重加载
            if self._backend == "mlx":
                return self._load_mlx_layer(layer_idx)
        except Exception as e:
            logger.warning(f"[omlx] Failed to load layer {layer_idx}: {e}")

        return None

    def _load_mlx_layer(self, layer_idx: int) -> Optional[dict]:
        """从 MLX 格式加载单层权重."""
        try:
            import mlx.core as mx
            from mlx.utils import tree_flatten

            # 如果已加载完整模型, 直接返回对应层
            if self._mlx_model is not None:
                # 从已加载模型中提取层权重
                layers = self._mlx_model.get("model", {}).get("layers", [])
                if layer_idx < len(layers):
                    return dict(layers[layer_idx])

            # 从 safetensors 文件按需加载
            weight_files = [
                f for f in os.listdir(self.model_path)
                if f.endswith(".safetensors")
            ]
            if not weight_files:
                return None

            # 简化: 加载包含目标层的文件
            # 生产环境用 index.json 做精确定位
            import json
            index_path = os.path.join(self.model_path, "model.safetensors.index.json")
            if os.path.exists(index_path):
                with open(index_path) as f:
                    index = json.load(f)
                weight_map = index.get("weight_map", {})
                layer_prefix = f"model.layers.{layer_idx}."
                layer_keys = {k: v for k, v in weight_map.items() if k.startswith(layer_prefix)}
                if not layer_keys:
                    return None
                # 加载对应的 safetensors 文件
                files_needed = set(layer_keys.values())
                weights = {}
                for fname in files_needed:
                    from safetensors.mlx import load_file
                    fpath = os.path.join(self.model_path, fname)
                    file_weights = load_file(fpath)
                    for k, v in file_weights.items():
                        if k.startswith(layer_prefix):
                            weights[k] = v
                return weights if weights else None

        except ImportError:
            logger.debug("[omlx] MLX not available, using mock weights")
            return None
        except Exception as e:
            logger.warning(f"[omlx] MLX layer load error: {e}")
            return None

    def _get_layer_size_mb(self, layer_idx: int) -> float:
        """估算单层权重大小 (MB)."""
        if self.is_moe:
            # MoE: router + top-k experts
            expert_size = self.hidden_size ** 2 * 3 * 2  # gate, up, down × bf16
            return (expert_size * self.experts_per_tok + self.hidden_size ** 2 * 4 * 2) / 1e6
        else:
            # Dense: attention + MLP
            return (self.hidden_size ** 2 * 12 * 2) / 1e6  # bf16

    def load_mlx_model(self, model_path: str):
        """加载完整 MLX 模型 (用于小模型 / 测试).

        生产环境用 by-layer, 此方法用于 fallback.
        """
        try:
            from mlx_lm import load
            self._mlx_model, self._tokenizer = load(model_path)
            self._backend = "mlx"
            self.model_path = model_path
            logger.info(f"[omlx] Full MLX model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"[omlx] Failed to load MLX model: {e}")
            return False

    def embed(self, token_ids: list[int]) -> Any:
        """Token embedding."""
        if self._mlx_model is not None:
            import mlx.core as mx
            # MLX embed
            return mx.array(token_ids)[None, :]  # 简化: 用 token id 作为 index
        return token_ids  # fallback: 返回 raw ids

    def lm_head(self, hidden: Any) -> Any:
        """LM head: hidden → logits."""
        if self._mlx_model is not None:
            import mlx.core as mx
            # MLX lm_head
            if hasattr(hidden, "argmax"):
                return hidden
            return mx.array(hidden)
        return hidden

    def forward_layer_streaming(
        self,
        layer_idx: int,
        hidden_states: Any,
        kv_cache: Any = None,
    ) -> tuple[Any, Any]:
        """流式 forward 单层, 触发层预取.

        Args:
            layer_idx: 层索引
            hidden_states: 输入 hidden states
            kv_cache: 该层 KV cache (可选)

        Returns:
            (output_hidden, new_kv_cache)
        """
        t0 = time.time()

        # 确保层已加载
        layer_weight = self.swap_pool.ensure_layer(layer_idx)

        # 异步预取下一层
        for depth in range(1, self.swap_pool.prefetch_depth + 1):
            next_idx = layer_idx + depth
            if next_idx < self.num_layers:
                self.swap_pool.prefetch(next_idx)

        # 执行 forward
        if self.is_moe and self._flashmoe is not None:
            # FlashMoE by-layer forward
            output = self._flashmoe.forward_layer(layer_idx, hidden_states, layer_weight)
        elif layer_weight is not None:
            # 标准层 forward (MLX 或 mock)
            output = self._forward_single_layer(layer_idx, hidden_states, layer_weight)
        else:
            # Fallback: 不做实际 forward (测试模式)
            output = hidden_states

        elapsed_ms = (time.time() - t0) * 1000
        logger.debug(f"[omlx] Layer {layer_idx} forward: {elapsed_ms:.1f}ms")

        return output, kv_cache

    def _forward_single_layer(self, layer_idx: int, hidden: Any, weight: dict) -> Any:
        """标准单层 forward."""
        if self._mlx_model is not None:
            # MLX forward (简化)
            try:
                import mlx.core as mx
                # 如果 hidden 是 mx.array, 执行实际 forward
                if isinstance(hidden, mx.array):
                    layers = self._mlx_model.get("model", {}).get("layers", [])
                    if layer_idx < len(layers):
                        # 实际层 forward (依赖模型结构)
                        return hidden  # 简化: 实际需调用 layer(hidden)
            except Exception:
                pass

        # Mock forward: 返回输入 (测试模式)
        return hidden

    def prefill(self, prompt_ids: list[int]) -> tuple[Any, Any]:
        """Prefill: 处理 prompt, 返回最后一个位置的 hidden states + KV cache.

        Args:
            prompt_ids: prompt token ids

        Returns:
            (last_hidden, kv_cache)
        """
        t0 = time.time()

        # Embedding
        hidden = self.embed(prompt_ids)

        # 逐层 forward (prefill)
        kv_cache = {}
        for layer_idx in range(self.num_layers):
            hidden, layer_kv = self.forward_layer_streaming(layer_idx, hidden, kv_cache)
            if layer_kv is not None:
                kv_cache[layer_idx] = layer_kv

        # 取最后一个位置
        last_hidden = hidden
        if hasattr(hidden, "__getitem__"):
            try:
                last_hidden = hidden[:, -1:, :]
            except Exception:
                pass

        elapsed_ms = (time.time() - t0) * 1000
        logger.info(f"[omlx] Prefill {len(prompt_ids)} tokens in {elapsed_ms:.1f}ms")

        return last_hidden, kv_cache

    def forward_draft(
        self,
        model_name: str,
        prompt_ids: list[int],
        draft_n: int = 8,
        pivot_layer: int = -1,
    ) -> DraftResult:
        """生成 draft 序列, 支持 pivot_layer 抢首包.

        Args:
            model_name: 模型名 (从 DraftRegistry 查找)
            prompt_ids: prompt token ids
            draft_n: 要生成的 draft token 数
            pivot_layer: pivot 层索引 (-1 = 不抢首包)

        Returns:
            DraftResult
        """
        t0 = time.time()
        result = DraftResult(pivot_layer=pivot_layer)

        try:
            # 1. Prefill
            t_prefill = time.time()
            hidden, kv_cache = self.prefill(prompt_ids)
            result.prefill_ms = (time.time() - t_prefill) * 1000

            # 2. 生成 draft tokens
            draft_tokens = []
            current_hidden = hidden

            for step in range(draft_n):
                t_step = time.time()

                # 分层 forward
                for layer_idx in range(self.num_layers):
                    current_hidden, kv_cache = self.forward_layer_streaming(
                        layer_idx, current_hidden, kv_cache
                    )

                    # Pivot: 在指定层抢首 token
                    if pivot_layer >= 0 and step == 0 and layer_idx == pivot_layer:
                        pivot_token = self._argmax_token(current_hidden)
                        result.pivot_token = pivot_token
                        draft_tokens.append(pivot_token)
                        logger.info(
                            f"[omlx] Pivot token at layer {pivot_layer}: "
                            f"token_id={pivot_token} ({(time.time()-t0)*1000:.1f}ms)"
                        )

                # 正常 token (非 pivot 或 step > 0)
                if pivot_layer < 0 or step > 0:
                    token = self._argmax_token(current_hidden)
                    draft_tokens.append(token)

                result.hidden_states.append(current_hidden)
                elapsed_step = (time.time() - t_step) * 1000
                logger.debug(f"[omlx] Draft step {step}: {elapsed_step:.1f}ms, token={draft_tokens[-1]}")

            result.tokens = draft_tokens
            result.draft_ms = (time.time() - t_prefill) * 1000 - result.prefill_ms
            result.latency_ms = (time.time() - t0) * 1000
            result.success = True

            logger.info(
                f"[omlx] Draft complete: {draft_n} tokens in {result.latency_ms:.1f}ms "
                f"(prefill={result.prefill_ms:.1f}ms, draft={result.draft_ms:.1f}ms)"
            )

        except Exception as e:
            result.error = str(e)
            result.latency_ms = (time.time() - t0) * 1000
            logger.error(f"[omlx] Draft failed: {e}")

        return result

    async def forward_draft_async(
        self,
        model_name: str,
        prompt_ids: list[int],
        draft_n: int = 8,
        pivot_layer: int = -1,
    ) -> AsyncIterator[DraftEvent]:
        """异步流式生成 draft, 逐步 yield 事件.

        用于 DraftPivotEngine 的流式输出.
        """
        t0 = time.time()

        try:
            # Prefill
            t_prefill = time.time()
            hidden, kv_cache = await asyncio.get_event_loop().run_in_executor(
                None, self.prefill, prompt_ids
            )
            prefill_ms = (time.time() - t_prefill) * 1000

            draft_tokens = []
            current_hidden = hidden

            for step in range(draft_n):
                # 分层 forward
                for layer_idx in range(self.num_layers):
                    current_hidden, kv_cache = self.forward_layer_streaming(
                        layer_idx, current_hidden, kv_cache
                    )

                    # Pivot 抢首包
                    if pivot_layer >= 0 and step == 0 and layer_idx == pivot_layer:
                        pivot_token = self._argmax_token(current_hidden)
                        draft_tokens.append(pivot_token)
                        yield DraftEvent(
                            type="first_token",
                            token_id=pivot_token,
                            pivot_layer=pivot_layer,
                            latency_ms=(time.time() - t0) * 1000,
                        )

                # 正常 token
                if pivot_layer < 0 or step > 0:
                    token = self._argmax_token(current_hidden)
                    draft_tokens.append(token)
                    yield DraftEvent(
                        type="draft_token",
                        token_id=token,
                        latency_ms=(time.time() - t0) * 1000,
                    )

            yield DraftEvent(
                type="draft_sequence",
                sequence=draft_tokens,
                latency_ms=(time.time() - t0) * 1000,
            )

        except Exception as e:
            yield DraftEvent(
                type="error",
                error=str(e),
                latency_ms=(time.time() - t0) * 1000,
            )

    def _argmax_token(self, hidden: Any) -> int:
        """从 hidden states 取 argmax token id."""
        try:
            if hasattr(hidden, "argmax"):
                # MLX / PyTorch tensor
                result = hidden.argmax(dim=-1)
                if hasattr(result, "item"):
                    return int(result.item())
                return int(result.flatten()[-1])
            elif isinstance(hidden, (list, tuple)):
                return int(hidden[-1]) if hidden else 0
        except Exception:
            pass
        return 0  # fallback

    def set_flashmoe(self, flashmoe: Any):
        """注入 FlashMoE by-layer 引擎 (MoE 模型用)."""
        self._flashmoe = flashmoe
        self.is_moe = True
        logger.info("[omlx] FlashMoE by-layer engine attached")

    def get_stats(self) -> dict:
        """获取运行时统计."""
        return {
            "backend": self._backend,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "is_moe": self.is_moe,
            "initialized": self._initialized,
            "layer_swap": self.swap_pool.get_stats(),
            "kv_cache": self.kv_cache.get_stats(),
            "activation": self.activation_pool.get_stats(),
            "expert_cache": self.expert_cache.get_stats(),
        }


if __name__ == "__main__":
    # 自测 (无模型权重, 测试 mock 路径)
    runtime = OMLXRuntime(
        model_path="",  # 空路径 → mock 模式
        layer_swap_config={"hot_slots": 4, "warm_slots": 8},
    )

    # 模拟模型元数据
    class MockModelInfo:
        num_layers = 12
        hidden_size = 512
        vocab_size = 32000
        is_moe = False
        num_experts = 0
        experts_per_tok = 0

    runtime.load_model_metadata(MockModelInfo())

    # Mock prefill + draft
    prompt_ids = [1, 2, 3, 4, 5]
    result = runtime.forward_draft("test", prompt_ids, draft_n=4, pivot_layer=6)

    print(f"Draft result: success={result.success}")
    print(f"  tokens={result.tokens}")
    print(f"  pivot_token={result.pivot_token}")
    print(f"  latency={result.latency_ms:.1f}ms")
    print(f"  prefill={result.prefill_ms:.1f}ms")
    print(f"  draft={result.draft_ms:.1f}ms")

    print(f"\nStats: {runtime.get_stats()}")
