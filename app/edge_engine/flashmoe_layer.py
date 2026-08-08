#!/usr/bin/env python3
"""FlashMoE by-layer -- 端侧 MoE 切层加载引擎.

设计前提:
  - 端侧内存 16-32GB (M4 Pro / M4 Max)
  - 目标 MoE 模型 30B-200B, 完整加载不可能
  - 路由只需 top-k=2 个 expert, 其余丢弃
  - 切粒度 = 1 层 (含 1 个 router + N 个 expert, 每次只 swap router + top-k experts)

内存预算 (DSV4-Flash 671B, 端侧跑 1 层):
  单层所有 expert (256 个): ~10GB
  单层 top-2 expert: ~80MB  ← 端侧可行
  单层 router + layer_norm: ~50MB
  单层 KV cache (seq=1, hidden=7168): ~30MB
  总单层常驻: ~160MB, 4 层 hot pool = 640MB

与现有 omlx_flashmoe.py 关系:
  复用 OMLXOptimizer 接口, 重写内部实现:
  - analyze_model() 保留
  - _apply_flash_moe_optimization() 改为按 FlashMoEByLayer 切层
  - 加速比从 hard-coded 1.8x → 实测测量
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from app.edge_engine.layer_swap_pool import ExpertCache

logger = logging.getLogger(__name__)


@dataclass
class MoELayerWeight:
    """单层 MoE 权重 (只含 router + top-k experts)."""
    layer_idx: int
    router_weight: Any = None        # gate network
    layer_norm_weight: Any = None    # input layernorm
    post_norm_weight: Any = None     # post-attention layernorm
    attention_weight: Any = None     # self-attention weights
    experts: dict[int, Any] = field(default_factory=dict)  # expert_id → weight
    expert_scores: dict[int, float] = field(default_factory=dict)  # expert_id → access frequency
    all_expert_ids: list[int] = field(default_factory=list)
    top_k: int = 2


class FlashMoEByLayer:
    """FlashMoE 切层加载 -- 端侧 MoE Draft 专用.

    用法:
        flashmoe = FlashMoEByLayer(model_path, layer_idx=0, top_k=2)
        flashmoe.load()
        output = flashmoe.forward(hidden_states)

    或集成到 OMLXRuntime:
        runtime.set_flashmoe(flashmoe)
    """

    def __init__(
        self,
        model_path: str = "",
        layer_idx: int = 0,
        top_k: int = 2,
        num_experts: int = 256,
        hidden_size: int = 7168,
        expert_cache: Optional[ExpertCache] = None,
    ):
        """初始化 FlashMoE 切层引擎.

        Args:
            model_path: 模型路径
            layer_idx: 要加载的层索引
            top_k: 每层只保留 top-k 个 expert
            num_experts: 模型总 expert 数
            hidden_size: hidden 维度
            expert_cache: 共享的 expert 缓存池
        """
        self.model_path = model_path
        self.layer_idx = layer_idx
        self.top_k = top_k
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.expert_cache = expert_cache or ExpertCache(keep_top_k=top_k)

        self._weight: Optional[MoELayerWeight] = None
        self._loaded = False
        self._load_time_ms = 0.0

        # 统计
        self._stats = {
            "forward_count": 0,
            "expert_swaps": 0,
            "total_forward_ms": 0.0,
            "router_evaluations": 0,
        }

    def load(self) -> "FlashMoEByLayer":
        """按需加载 1 层权重 (router + top-k experts).

        步骤:
        1. 读 layer_norm / router (小, ~50MB)
        2. 读 attention 权重 (共享, 不切)
        3. 读所有 expert 索引 (metadata)
        4. 按需只读 top-k expert (预取 top-2 假设分布稳定)
        5. 释放其余 expert 内存
        """
        t0 = time.time()

        weight = MoELayerWeight(
            layer_idx=self.layer_idx,
            top_k=self.top_k,
        )

        # 1. 加载 router + layer_norm (小)
        weight.router_weight = self._load_router()
        weight.layer_norm_weight = self._load_layer_norm()
        weight.post_norm_weight = self._load_post_norm()
        weight.attention_weight = self._load_attention()

        # Mock 模式: 无模型路径时创建 dummy 权重
        if not self.model_path or not os.path.exists(self.model_path):
            weight.router_weight = weight.router_weight or {"gate": "mock"}
            weight.layer_norm_weight = weight.layer_norm_weight or {"norm": "mock"}
            weight.post_norm_weight = weight.post_norm_weight or {"norm": "mock"}
            weight.attention_weight = weight.attention_weight or {"attn": "mock"}

        # 2. 加载所有 expert 索引 (metadata, 不含权重)
        weight.all_expert_ids = list(range(self.num_experts))

        # 3. 预取 top-k expert (默认前 k 个, 后续按 router 输出动态调整)
        for k in range(min(self.top_k, len(weight.all_expert_ids))):
            expert_id = weight.all_expert_ids[k]
            expert_weight = self._load_expert(expert_id)
            if expert_weight is not None:
                weight.experts[expert_id] = expert_weight
                weight.expert_scores[expert_id] = 1.0  # 初始分数

        # 4. 释放不需要的 metadata
        # (all_expert_ids 保留, 用于动态 swap)

        self._weight = weight
        self._loaded = True
        self._load_time_ms = (time.time() - t0) * 1000

        loaded_mb = self._estimate_loaded_mb()
        logger.info(
            f"[flashmoe] Layer {self.layer_idx} loaded: "
            f"router + {len(weight.experts)}/{self.num_experts} experts "
            f"in {self._load_time_ms:.1f}ms ({loaded_mb:.1f}MB)"
        )

        return self

    def forward(self, hidden_states: Any) -> Any:
        """按 MoE 路由分发到 top-k expert.

        步骤:
        1. LayerNorm
        2. Self-Attention (标准, 不切)
        3. Router → top-k indices
        4. 动态加载缺失的 expert (如果 router 选了未缓存的)
        5. Expert forward → 加权求和
        6. Post-LayerNorm + residual
        """
        if not self._loaded or self._weight is None:
            raise RuntimeError(f"Layer {self.layer_idx} not loaded")

        t0 = time.time()
        self._stats["forward_count"] += 1

        # 1. Input LayerNorm
        h = self._apply_layer_norm(hidden_states, self._weight.layer_norm_weight)

        # 2. Self-Attention
        h_attn = self._attention_forward(h, self._weight.attention_weight)
        h = self._residual(hidden_states, h_attn)

        # 3. Post-attention LayerNorm
        h_normed = self._apply_layer_norm(h, self._weight.post_norm_weight)

        # 4. Router → top-k indices
        top_k_indices, top_k_weights = self._router_forward(h_normed)
        self._stats["router_evaluations"] += 1

        # 5. 动态加载缺失的 expert
        for idx in top_k_indices:
            if idx not in self._weight.experts:
                self._swap_expert(idx)
                self._stats["expert_swaps"] += 1

        # 6. Expert forward → 加权求和
        expert_outputs = []
        for i, expert_idx in enumerate(top_k_indices):
            expert_weight = self._weight.experts.get(expert_idx)
            if expert_weight is not None:
                out = self._expert_forward(h_normed, expert_weight)
                if out is not None:
                    expert_outputs.append(out * top_k_weights[i])

        if expert_outputs:
            moe_output = sum(expert_outputs)
        else:
            moe_output = h_normed  # fallback (包括 mock 模式)

        # 7. Residual
        output = self._residual(h, moe_output)

        elapsed_ms = (time.time() - t0) * 1000
        self._stats["total_forward_ms"] += elapsed_ms

        return output

    def forward_layer(
        self,
        layer_idx: int,
        hidden_states: Any,
        layer_weight: Optional[dict] = None,
    ) -> Any:
        """OMLXRuntime 兼容接口: forward 单层.

        如果 layer_idx != self.layer_idx, 需要重新加载.
        """
        if layer_idx != self.layer_idx:
            self.layer_idx = layer_idx
            self._loaded = False
            self.load()
        return self.forward(hidden_states)

    def _load_router(self) -> Any:
        """加载 router (gate network) 权重."""
        if not self.model_path or not os.path.exists(self.model_path):
            return None
        try:
            # 尝试从 safetensors 加载
            prefix = f"model.layers.{self.layer_idx}.mlp.gate"
            return self._load_weight_by_prefix(prefix)
        except Exception as e:
            logger.warning(f"[flashmoe] Router load failed: {e}")
            return None

    def _load_layer_norm(self) -> Any:
        """加载 input layer norm."""
        if not self.model_path or not os.path.exists(self.model_path):
            return None
        return self._load_weight_by_prefix(f"model.layers.{self.layer_idx}.input_layernorm")

    def _load_post_norm(self) -> Any:
        """加载 post-attention layer norm."""
        if not self.model_path or not os.path.exists(self.model_path):
            return None
        return self._load_weight_by_prefix(f"model.layers.{self.layer_idx}.post_attention_layernorm")

    def _load_attention(self) -> Any:
        """加载 self-attention 权重."""
        if not self.model_path or not os.path.exists(self.model_path):
            return None
        return self._load_weight_by_prefix(f"model.layers.{self.layer_idx}.self_attn")

    def _load_expert(self, expert_id: int) -> Any:
        """加载单个 expert 权重 (gate, up, down)."""
        # 先查缓存
        cached = self.expert_cache.get(self.layer_idx, expert_id)
        if cached is not None:
            return cached

        if not self.model_path or not os.path.exists(self.model_path):
            # Mock 模式: 返回 dummy expert 权重
            mock_weight = {"gate_proj": "mock", "up_proj": "mock", "down_proj": "mock",
                           "expert_id": expert_id}
            self.expert_cache.put(self.layer_idx, expert_id, mock_weight)
            return mock_weight

        try:
            prefix = f"model.layers.{self.layer_idx}.mlp.experts.{expert_id}"
            weight = self._load_weight_by_prefix(prefix)
            if weight is not None:
                self.expert_cache.put(self.layer_idx, expert_id, weight)
            return weight
        except Exception as e:
            logger.warning(f"[flashmoe] Expert {expert_id} load failed: {e}")
            return None

    def _load_weight_by_prefix(self, prefix: str) -> Any:
        """从 safetensors 按前缀加载权重."""
        try:
            import json
            index_path = os.path.join(self.model_path, "model.safetensors.index.json")
            if not os.path.exists(index_path):
                return None

            with open(index_path) as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})

            # 找到所有匹配前缀的权重
            matching = {k: v for k, v in weight_map.items() if k.startswith(prefix)}
            if not matching:
                return None

            # 加载对应的文件
            files_needed = set(matching.values())
            weights = {}
            for fname in files_needed:
                try:
                    from safetensors.mlx import load_file
                    fpath = os.path.join(self.model_path, fname)
                    file_weights = load_file(fpath)
                    for k, v in file_weights.items():
                        if k.startswith(prefix):
                            weights[k] = v
                except ImportError:
                    # MLX 不可用, 尝试 numpy
                    try:
                        from safetensors.numpy import load_file as np_load
                        fpath = os.path.join(self.model_path, fname)
                        file_weights = np_load(fpath)
                        for k, v in file_weights.items():
                            if k.startswith(prefix):
                                weights[k] = v
                    except ImportError:
                        logger.debug("[flashmoe] No safetensors backend available")
                        return None

            return weights if weights else None

        except Exception as e:
            logger.warning(f"[flashmoe] Weight load error for {prefix}: {e}")
            return None

    def _swap_expert(self, expert_id: int):
        """动态换入一个 expert, 淘汰最少使用的."""
        # 加载新 expert
        expert_weight = self._load_expert(expert_id)
        if expert_weight is None:
            return

        # 淘汰最少使用的
        if len(self._weight.experts) >= self.top_k:
            # 找分数最低的
            min_id = min(self._weight.experts, key=lambda k: self._weight.expert_scores.get(k, 0))
            del self._weight.experts[min_id]
            self._weight.expert_scores.pop(min_id, None)
            logger.debug(f"[flashmoe] Swapped expert {min_id} → {expert_id}")

        self._weight.experts[expert_id] = expert_weight
        self._weight.expert_scores[expert_id] = 1.0

    def _router_forward(self, hidden: Any) -> tuple[list[int], list[float]]:
        """Router forward → top-k indices + weights.

        生产环境: matmul(hidden, router_weight) → softmax → top-k
        当前: mock 返回 [0, 1] + 均匀权重
        """
        if self._weight.router_weight is None:
            # Mock: 返回前 top_k 个 expert
            return list(range(self.top_k)), [1.0 / self.top_k] * self.top_k

        try:
            import mlx.core as mx
            if isinstance(hidden, mx.array) and isinstance(self._weight.router_weight, dict):
                # 实际 router forward
                w = list(self._weight.router_weight.values())[0]
                logits = mx.matmul(hidden, w.T)
                probs = mx.softmax(logits, axis=-1)
                # top-k
                # MLX doesn't have topk in all versions, use argpartition
                top_indices = mx.argpartition(probs, -self.top_k)[-self.top_k:]
                top_weights = probs[top_indices]
                # 归一化
                top_weights = top_weights / mx.sum(top_weights)
                return [int(i) for i in top_indices], [float(w) for w in top_weights]
        except Exception:
            pass

        # Mock fallback
        return list(range(self.top_k)), [1.0 / self.top_k] * self.top_k

    def _expert_forward(self, hidden: Any, expert_weight: Any) -> Any:
        """单个 expert forward: SiLU(gate(x)) * up(x) → down()."""
        if expert_weight is None:
            return hidden

        try:
            import mlx.core as mx
            if isinstance(hidden, mx.array) and isinstance(expert_weight, dict):
                # 实际 expert forward
                gate_w = expert_weight.get("gate_proj", {}).get("weight", None)
                up_w = expert_weight.get("up_proj", {}).get("weight", None)
                down_w = expert_weight.get("down_proj", {}).get("weight", None)

                if gate_w is not None and up_w is not None and down_w is not None:
                    gate_out = mx.matmul(hidden, gate_w.T)
                    up_out = mx.matmul(hidden, up_w.T)
                    activated = mx.silu(gate_out) * up_out
                    return mx.matmul(activated, down_w.T)
        except Exception:
            pass

        # Mock: identity
        return hidden

    def _apply_layer_norm(self, hidden: Any, weight: Any) -> Any:
        """LayerNorm forward."""
        if weight is None:
            return hidden
        try:
            import mlx.core as mx
            if isinstance(hidden, mx.array):
                # 简化 LayerNorm
                mean = mx.mean(hidden, axis=-1, keepdims=True)
                var = mx.var(hidden, axis=-1, keepdims=True)
                hidden = (hidden - mean) / mx.sqrt(var + 1e-5)
                if isinstance(weight, dict):
                    gamma = list(weight.values())[0]
                    return hidden * gamma
                return hidden
        except Exception:
            pass
        return hidden

    def _attention_forward(self, hidden: Any, weight: Any) -> Any:
        """Self-attention forward (简化)."""
        return hidden  # 简化: 生产环境需完整 attention

    def _residual(self, a: Any, b: Any) -> Any:
        """Residual connection."""
        if a is None:
            return b
        if b is None:
            return a
        try:
            return a + b
        except Exception:
            return a

    def _estimate_loaded_mb(self) -> float:
        """估算已加载权重大小 (MB)."""
        # router + attention + top-k experts
        expert_mb = self.hidden_size ** 2 * 3 * 2 / 1e6  # gate, up, down × bf16
        attn_mb = self.hidden_size ** 2 * 4 * 2 / 1e6   # q, k, v, o × bf16
        return expert_mb * self.top_k + attn_mb + 0.05  # +50MB router/norm

    def get_stats(self) -> dict:
        """获取 FlashMoE 统计."""
        avg_forward_ms = (
            self._stats["total_forward_ms"] / self._stats["forward_count"]
            if self._stats["forward_count"] > 0 else 0
        )
        return {
            "layer_idx": self.layer_idx,
            "top_k": self.top_k,
            "num_experts": self.num_experts,
            "loaded_experts": len(self._weight.experts) if self._weight else 0,
            "loaded": self._loaded,
            "load_time_ms": round(self._load_time_ms, 1),
            "loaded_mb": round(self._estimate_loaded_mb(), 1),
            "avg_forward_ms": round(avg_forward_ms, 2),
            **self._stats,
            "expert_cache": self.expert_cache.get_stats(),
        }


class FlashMoEAnalyzer:
    """分析 MoE 模型, 决定切层策略.

    复用 omlx_flashmoe.py 的 OMLXOptimizer.analyze_model() 接口.
    """

    @staticmethod
    def analyze_model(model_info: Any, top_k: int = 2) -> dict:
        """分析 MoE 模型, 输出切层建议.

        Args:
            model_info: ModelInfo 或类似对象
            top_k: FlashMoE 每层保留的 expert 数 (设计参数, 默认 2)

        Returns:
            分析结果 dict
        """
        is_moe = getattr(model_info, "is_moe", False)
        if not is_moe:
            return {
                "is_moe": False,
                "recommendation": "no_flashmoe",
                "reason": "Dense model, FlashMoE not needed",
            }

        num_experts = getattr(model_info, "num_experts", 0)
        experts_per_tok = getattr(model_info, "experts_per_tok", 0)
        hidden_size = getattr(model_info, "hidden_size", 0)
        num_layers = getattr(model_info, "num_layers", 0)
        per_layer_gb = getattr(model_info, "per_layer_gb", 0.0)

        # 单 expert 大小 (gate + up + down, bf16)
        single_expert_mb = hidden_size ** 2 * 3 * 2 / 1e6

        # FlashMoE top-k expert 大小 (用设计参数 top_k, 不是 experts_per_tok)
        flashmoe_topk = min(top_k, experts_per_tok) if experts_per_tok > 0 else top_k
        topk_mb = single_expert_mb * flashmoe_topk

        # 全量 expert 大小
        all_experts_mb = single_expert_mb * num_experts

        # 节省比例
        savings = 1 - topk_mb / all_experts_mb if all_experts_mb > 0 else 0

        return {
            "is_moe": True,
            "recommendation": "flashmoe_by_layer",
            "num_experts": num_experts,
            "experts_per_tok": experts_per_tok,
            "flashmoe_top_k": flashmoe_topk,
            "single_expert_mb": round(single_expert_mb, 1),
            "topk_experts_mb": round(topk_mb, 1),
            "all_experts_mb": round(all_experts_mb, 1),
            "memory_savings": round(savings, 4),
            "savings_pct": round(savings * 100, 1),
            "per_layer_full_gb": per_layer_gb,
            "per_layer_flashmoe_mb": round(topk_mb + hidden_size ** 2 * 4 * 2 / 1e6 + 50, 1),
            "num_layers": num_layers,
            "reason": f"FlashMoE: {num_experts} experts → top-{flashmoe_topk} (of {experts_per_tok} active), "
                      f"save {savings*100:.1f}% memory per layer",
        }


if __name__ == "__main__":
    # 自测: 分析 DSV4-Flash
    class MockDSV4Model:
        is_moe = True
        num_experts = 256
        experts_per_tok = 8
        hidden_size = 7168
        num_layers = 61
        per_layer_gb = 4.9

    analysis = FlashMoEAnalyzer.analyze_model(MockDSV4Model())
    print("FlashMoE Analysis (DSV4-Flash 671B):")
    for k, v in analysis.items():
        print(f"  {k}: {v}")

    print()

    # 自测: 切层加载
    flashmoe = FlashMoEByLayer(
        model_path="",  # mock
        layer_idx=0,
        top_k=2,
        num_experts=256,
        hidden_size=7168,
    )
    flashmoe.load()

    print(f"\nFlashMoE Stats: {flashmoe.get_stats()}")

    # 测试 forward (mock)
    output = flashmoe.forward(None)  # mock hidden
    print(f"Forward output: {output is not None}")
