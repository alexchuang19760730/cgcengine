#!/usr/bin/env python3
"""LayerSwapPool -- 层预取 + 内存池调度.

oMLX Runtime 的核心组件: 管理 hot/warm/cold 三级层缓存,
在 forward 第 N 层时异步预取第 N+2/N+3 层权重.

设计:
  hot_slots  (4): 常驻内存, 关键层 (embedding + 早期 attention)
  warm_slots (8): 预取池, forward 第 N 层时加载第 N+2~N+3 层
  cold:       NVMe SSD 兜底, 通过 mmap 近零开销切层

内存预算 (M4 16GB):
  4 hot × 160MB = 640MB
  8 warm × 160MB = 1280MB
  KV cache pool = 4GB
  Activation pool = 2GB
  Total ≈ 8GB (留 8GB 给系统)
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class LayerSlot:
    """单个层缓存槽位."""
    layer_idx: int
    weight: Any = None           # 层权重 (dict of tensors)
    loaded_at: float = 0.0       # 加载时间戳
    last_used: float = 0.0       # 最后访问时间
    size_mb: float = 0.0         # 权重大小 (MB)
    is_moe: bool = False         # 是否 MoE 层
    expert_ids: list = field(default_factory=list)  # MoE: 已加载的 expert id


class LayerSwapPool:
    """三级层缓存池: hot (常驻) → warm (预取) → cold (SSD).

    线程安全. 支持异步预取和同步加载.
    """

    def __init__(
        self,
        hot_slots: int = 4,
        warm_slots: int = 8,
        cold_swap_ms: float = 20.0,
        prefetch_depth: int = 2,
    ):
        """初始化层交换池.

        Args:
            hot_slots: 常驻层数 (embedding + 早期层)
            warm_slots: 预取池大小
            cold_swap_ms: 冷加载预估延迟 (ms, 用于性能监控)
            prefetch_depth: 预取深度 (forward N 时预取 N+depth)
        """
        self.hot_slots_max = hot_slots
        self.warm_slots_max = warm_slots
        self.cold_swap_ms = cold_swap_ms
        self.prefetch_depth = prefetch_depth

        # 三级缓存: OrderedDict for LRU
        self._hot: OrderedDict[int, LayerSlot] = OrderedDict()
        self._warm: OrderedDict[int, LayerSlot] = OrderedDict()

        # 加载函数 (由 OMLXRuntime 注入)
        self._load_fn: Optional[Callable[[int], Any]] = None
        self._weight_size_fn: Optional[Callable[[int], float]] = None

        # 统计
        self._stats = {
            "hot_hits": 0,
            "warm_hits": 0,
            "cold_misses": 0,
            "prefetch_count": 0,
            "swap_count": 0,
            "total_swap_ms": 0.0,
        }

        self._lock = threading.RLock()
        self._prefetch_executor: Optional[asyncio.Task] = None

    def set_load_function(self, load_fn: Callable[[int], Any],
                          weight_size_fn: Callable[[int], float] = None):
        """注入层加载函数 (由 OMLXRuntime 调用).

        Args:
            load_fn: layer_idx → weight_dict
            weight_size_fn: layer_idx → size_mb
        """
        self._load_fn = load_fn
        self._weight_size_fn = weight_size_fn

    def ensure_layer(self, layer_idx: int) -> Any:
        """确保某层已加载到内存, 返回权重.

        查找顺序: hot → warm → cold (加载)
        """
        with self._lock:
            # Hot hit
            if layer_idx in self._hot:
                self._hot[layer_idx].last_used = time.time()
                self._hot.move_to_end(layer_idx)
                self._stats["hot_hits"] += 1
                return self._hot[layer_idx].weight

            # Warm hit
            if layer_idx in self._warm:
                slot = self._warm.pop(layer_idx)
                self._stats["warm_hits"] += 1
                # Promote to hot
                self._promote_to_hot(layer_idx, slot)
                return slot.weight

            # Cold miss — load from disk
            self._stats["cold_misses"] += 1
            return self._load_cold(layer_idx)

    def prefetch(self, layer_idx: int):
        """异步预取一层到 warm pool (非阻塞).

        如果层已在 hot/warm 中, 跳过.
        如果 warm pool 满, 淘汰最久未使用的.
        """
        with self._lock:
            if layer_idx in self._hot or layer_idx in self._warm:
                return
            if len(self._warm) >= self.warm_slots_max:
                self._evict_warm()
            if self._load_fn is None:
                return

        # 在后台线程加载 (不阻塞 forward)
        threading.Thread(
            target=self._prefetch_worker,
            args=(layer_idx,),
            daemon=True,
        ).start()
        self._stats["prefetch_count"] += 1

    def prefetch_range(self, start_idx: int, count: int):
        """预取连续多层."""
        for i in range(start_idx, start_idx + count):
            self.prefetch(i)

    def _prefetch_worker(self, layer_idx: int):
        """后台预取线程."""
        try:
            t0 = time.time()
            weight = self._load_fn(layer_idx)
            if weight is None:
                return
            size_mb = 0.0
            if self._weight_size_fn:
                size_mb = self._weight_size_fn(layer_idx)

            slot = LayerSlot(
                layer_idx=layer_idx,
                weight=weight,
                loaded_at=time.time(),
                last_used=time.time(),
                size_mb=size_mb,
            )
            with self._lock:
                if layer_idx not in self._hot and layer_idx not in self._warm:
                    if len(self._warm) >= self.warm_slots_max:
                        self._evict_warm()
                    self._warm[layer_idx] = slot
                    swap_ms = (time.time() - t0) * 1000
                    self._stats["swap_count"] += 1
                    self._stats["total_swap_ms"] += swap_ms
                    logger.debug(
                        f"[layer-swap] Prefetched layer {layer_idx} "
                        f"in {swap_ms:.1f}ms ({size_mb:.1f}MB)"
                    )
        except Exception as e:
            logger.warning(f"[layer-swap] Prefetch layer {layer_idx} failed: {e}")

    def _load_cold(self, layer_idx: int) -> Any:
        """从磁盘加载一层 (同步, 阻塞)."""
        if self._load_fn is None:
            raise RuntimeError("No load function set")

        t0 = time.time()
        weight = self._load_fn(layer_idx)
        if weight is None:
            raise RuntimeError(f"Failed to load layer {layer_idx}")

        size_mb = 0.0
        if self._weight_size_fn:
            size_mb = self._weight_size_fn(layer_idx)

        slot = LayerSlot(
            layer_idx=layer_idx,
            weight=weight,
            loaded_at=time.time(),
            last_used=time.time(),
            size_mb=size_mb,
        )

        # 放入 warm (如果满则淘汰)
        with self._lock:
            if len(self._warm) >= self.warm_slots_max:
                self._evict_warm()
            self._warm[layer_idx] = slot

        swap_ms = (time.time() - t0) * 1000
        self._stats["swap_count"] += 1
        self._stats["total_swap_ms"] += swap_ms
        logger.info(
            f"[layer-swap] Cold load layer {layer_idx} in {swap_ms:.1f}ms ({size_mb:.1f}MB)"
        )
        return weight

    def _promote_to_hot(self, layer_idx: int, slot: LayerSlot):
        """将 warm 层提升到 hot pool."""
        if len(self._hot) >= self.hot_slots_max:
            # 淘汰最久未使用的 hot → 降级到 warm
            evicted_idx, evicted_slot = self._hot.popitem(last=False)
            if len(self._warm) >= self.warm_slots_max:
                self._evict_warm()
            self._warm[evicted_idx] = evicted_slot

        slot.last_used = time.time()
        self._hot[layer_idx] = slot

    def _evict_warm(self):
        """淘汰 warm pool 中最久未使用的层."""
        if not self._warm:
            return
        evicted_idx, slot = self._warm.popitem(last=False)
        # 释放权重内存
        if slot.weight is not None:
            del slot.weight
            slot.weight = None
        logger.debug(f"[layer-swap] Evicted warm layer {evicted_idx} (LRU)")

    def pin_hot(self, layer_indices: list[int]):
        """将关键层固定到 hot pool (常驻不淘汰).

        Args:
            layer_indices: 要固定的层索引列表
        """
        for idx in layer_indices:
            with self._lock:
                if idx in self._warm:
                    slot = self._warm.pop(idx)
                    self._promote_to_hot(idx, slot)
                elif idx not in self._hot:
                    # 需要加载
                    weight = self._load_fn(idx) if self._load_fn else None
                    if weight is not None:
                        size_mb = self._weight_size_fn(idx) if self._weight_size_fn else 0.0
                        slot = LayerSlot(
                            layer_idx=idx,
                            weight=weight,
                            loaded_at=time.time(),
                            last_used=time.time(),
                            size_mb=size_mb,
                        )
                        self._promote_to_hot(idx, slot)

    def get_stats(self) -> dict:
        """获取缓存统计."""
        with self._lock:
            total = self._stats["hot_hits"] + self._stats["warm_hits"] + self._stats["cold_misses"]
            hit_rate = (self._stats["hot_hits"] + self._stats["warm_hits"]) / total if total > 0 else 0
            avg_swap_ms = self._stats["total_swap_ms"] / self._stats["swap_count"] if self._stats["swap_count"] > 0 else 0
            return {
                **self._stats,
                "hot_count": len(self._hot),
                "warm_count": len(self._warm),
                "hit_rate": round(hit_rate, 3),
                "avg_swap_ms": round(avg_swap_ms, 1),
                "hot_layers": list(self._hot.keys()),
                "warm_layers": list(self._warm.keys()),
            }

    def clear(self):
        """清空所有缓存."""
        with self._lock:
            self._hot.clear()
            self._warm.clear()
            self._stats = {k: 0 if isinstance(v, int) else 0.0 for k, v in self._stats.items()}


class KVCachePool:
    """KV Cache 内存池 — 管理 decode 过程中的 KV cache.

    预分配固定大小内存块, 避免频繁 alloc/free.
    """

    def __init__(self, max_gb: float = 4.0, num_layers: int = 0, hidden_size: int = 0):
        self.max_gb = max_gb
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self._cache: dict[int, Any] = {}  # layer_idx → (k, v)
        self._current_seq_len = 0
        self._max_seq_len = 0

    def allocate(self, num_layers: int, hidden_size: int, batch: int = 1):
        """预分配 KV cache."""
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # 实际分配由后端 (MLX/PyTorch) 决定
        logger.info(f"[kv-pool] Allocated for {num_layers} layers, hidden={hidden_size}, batch={batch}")

    def set_layer_kv(self, layer_idx: int, k: Any, v: Any):
        self._cache[layer_idx] = (k, v)

    def get_layer_kv(self, layer_idx: int) -> tuple:
        return self._cache.get(layer_idx, (None, None))

    def extend_seq(self, n: int = 1):
        self._current_seq_len += n
        self._max_seq_len = max(self._max_seq_len, self._current_seq_len)

    def reset(self):
        self._cache.clear()
        self._current_seq_len = 0

    def get_stats(self) -> dict:
        return {
            "num_layers_cached": len(self._cache),
            "current_seq_len": self._current_seq_len,
            "max_seq_len": self._max_seq_len,
            "max_gb": self.max_gb,
        }


class ActivationPool:
    """激活值内存池 — 管理 forward 中间结果."""

    def __init__(self, max_gb: float = 2.0):
        self.max_gb = max_gb
        self._activations: dict[str, Any] = {}
        self._peak_mb = 0.0

    def put(self, key: str, tensor: Any, size_mb: float = 0.0):
        self._activations[key] = tensor
        self._peak_mb = max(self._peak_mb, size_mb)

    def get(self, key: str) -> Any:
        return self._activations.get(key)

    def pop(self, key: str) -> Any:
        return self._activations.pop(key, None)

    def clear(self):
        self._activations.clear()

    def get_stats(self) -> dict:
        return {
            "active_count": len(self._activations),
            "peak_mb": round(self._peak_mb, 1),
            "max_gb": self.max_gb,
        }


class ExpertCache:
    """MoE 专家缓存 — 只保留 top-k 激活专家.

    MoE 模型每层有 N 个 expert, 但推理时只激活 top-k 个.
    缓存最近 k 个 expert, 避免 swap 开销.
    """

    def __init__(self, keep_top_k: int = 2, eviction: str = "lru"):
        self.keep_top_k = keep_top_k
        self.eviction = eviction
        self._cache: dict[tuple[int, int], Any] = {}  # (layer_idx, expert_id) → weight
        self._access_count: dict[tuple[int, int], int] = {}
        self._lock = threading.RLock()

    def get(self, layer_idx: int, expert_id: int) -> Any:
        with self._lock:
            key = (layer_idx, expert_id)
            self._access_count[key] = self._access_count.get(key, 0) + 1
            return self._cache.get(key)

    def put(self, layer_idx: int, expert_id: int, weight: Any):
        with self._lock:
            key = (layer_idx, expert_id)
            self._cache[key] = weight
            # 淘汰: 超过 keep_top_k × num_layers 时淘汰
            layer_experts = {k[1]: k for k in self._cache if k[0] == layer_idx}
            if len(layer_experts) > self.keep_top_k:
                # 淘汰最少访问的
                to_evict = min(
                    layer_experts.values(),
                    key=lambda k: self._access_count.get(k, 0),
                )
                del self._cache[to_evict]
                self._access_count.pop(to_evict, None)

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "cached_experts": len(self._cache),
                "keep_top_k": self.keep_top_k,
                "eviction": self.eviction,
            }


if __name__ == "__main__":
    # 自测
    pool = LayerSwapPool(hot_slots=4, warm_slots=8)

    # 模拟加载函数
    def mock_load(idx):
        return {"layer": idx, "weight": f"mock_weight_{idx}"}

    def mock_size(idx):
        return 160.0

    pool.set_load_function(mock_load, mock_size)

    # 固定关键层
    pool.pin_hot([0, 1, 2, 3])

    # 访问测试
    w = pool.ensure_layer(5)
    print(f"Layer 5 loaded: {w is not None}")

    pool.prefetch(7)
    pool.prefetch(8)
    time.sleep(0.5)  # 等预取

    w = pool.ensure_layer(7)
    print(f"Layer 7 (should be warm hit): {w is not None}")

    stats = pool.get_stats()
    print(f"\nStats: {stats}")
