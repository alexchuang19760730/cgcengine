"""
MagiCompiler Phase 2: 動態 Shape 的 Graph 緩存機制
Dynamic Shape CUDA Graph Cache for vLLM
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class GraphConfig:
    """CUDA Graph 配置"""

    seq_len: int
    batch_size: int = 1
    num_heads_q: int = 0
    num_heads_kv: int = 0
    head_dim: int = 128


class DynamicGraphCache:
    """
    動態 Shape 的 CUDA Graph 緩存管理器

    核心功能：
    1. 根據不同序列長度緩存對應的 CUDA Graph
    2. 支持批量大小和模型結構參數
    3. LRU 緩存淘汰策略
    4. 圖形預熱和自動重建
    """

    def __init__(self, max_cache_size: int = 10, warmup_iterations: int = 3):
        """
        Args:
            max_cache_size: 最大緩存數量（不同 seq_len 的 Graph 數量）
            warmup_iterations: 預熱迭代次數
        """
        self.max_cache_size = max_cache_size
        self.warmup_iterations = warmup_iterations

        self.graph_cache: Dict[int, Dict] = {}
        self.access_order: List[int] = []
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "graphs_created": 0,
            "graphs_evicted": 0,
        }

    def _get_cache_key(self, seq_len: int, batch_size: int = 1) -> int:
        return seq_len

    def _evict_lru(self):
        if len(self.graph_cache) >= self.max_cache_size:
            oldest_seq_len = self.access_order.pop(0)
            if oldest_seq_len in self.graph_cache:
                del self.graph_cache[oldest_seq_len]
                self.stats["graphs_evicted"] += 1
                print(f"[DynamicGraphCache] 🗑️ 淘汰 seq_len={oldest_seq_len} 的 Graph")

    def _update_access(self, seq_len: int):
        if seq_len in self.access_order:
            self.access_order.remove(seq_len)
        self.access_order.append(seq_len)

    def register_graph(
        self,
        seq_len: int,
        graph: torch.cuda.CUDAGraph,
        input_placeholder: torch.Tensor,
        output_placeholder: torch.Tensor,
        batch_size: int = 1,
    ):
        if len(self.graph_cache) >= self.max_cache_size and seq_len not in self.graph_cache:
            self._evict_lru()

        cache_key = self._get_cache_key(seq_len, batch_size)
        self.graph_cache[cache_key] = {
            "graph": graph,
            "input_placeholder": input_placeholder,
            "output_placeholder": output_placeholder,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "created_at": time.time(),
        }

        self._update_access(cache_key)
        self.stats["graphs_created"] += 1

        print(f"[DynamicGraphCache] ✅ 註冊 Graph: seq_len={seq_len}, batch_size={batch_size}")

    def get_graph(self, seq_len: int, batch_size: int = 1) -> Optional[Dict]:
        cache_key = self._get_cache_key(seq_len, batch_size)

        if cache_key in self.graph_cache:
            self.stats["cache_hits"] += 1
            self._update_access(cache_key)
            return self.graph_cache[cache_key]

        self.stats["cache_misses"] += 1
        return None

    def has_graph(self, seq_len: int, batch_size: int = 1) -> bool:
        return self._get_cache_key(seq_len, batch_size) in self.graph_cache

    def replay(self, seq_len: int, input_tensor: torch.Tensor, batch_size: int = 1) -> torch.Tensor:
        graph_info = self.get_graph(seq_len, batch_size)
        if graph_info is None:
            raise RuntimeError(f"未找到 seq_len={seq_len} 的 Graph，請先調用 create_graph()")

        graph_info["input_placeholder"].copy_(input_tensor)
        graph_info["graph"].replay()
        return graph_info["output_placeholder"].clone()

    def clear(self):
        self.graph_cache.clear()
        self.access_order.clear()
        print("[DynamicGraphCache] 🧹 緩存已清除")

    def get_stats(self) -> Dict:
        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        hit_rate = self.stats["cache_hits"] / total_requests if total_requests > 0 else 0
        return {
            **self.stats,
            "cache_size": len(self.graph_cache),
            "hit_rate": hit_rate,
        }


class PrefillDecodeGraphManager:
    """
    Prefill 和 Decode 階段的 Graph 管理器

    設計考慮：
    - Prefill: 序列長度固定，捕獲一次即可
    - Decode: 序列長度動態增長，需要多個 Graph
    """

    def __init__(self, max_decode_cache_size: int = 20):
        self.prefill_graph: Optional[torch.cuda.CUDAGraph] = None
        self.prefill_input: Optional[torch.Tensor] = None
        self.prefill_output: Optional[Any] = None

        self.decode_cache = DynamicGraphCache(max_cache_size=max_decode_cache_size)
        self._is_prefill_ready = False
        self._current_decode_seq_len = 0

    def capture_prefill(
        self, model: torch.nn.Module, sample_input: torch.Tensor, **forward_kwargs
    ) -> torch.cuda.CUDAGraph:
        if self._is_prefill_ready:
            print("[PrefillDecodeManager] ⚠️ Prefill Graph 已存在，將重新捕獲")
            self.prefill_graph = None

        self.prefill_input = sample_input.clone().detach().requires_grad_(False).cuda()

        with torch.no_grad():
            warmup_output = model(self.prefill_input, **forward_kwargs)

        if isinstance(warmup_output, torch.Tensor):
            self.prefill_output = warmup_output.clone().detach().requires_grad_(False).cuda()
        else:
            self.prefill_output = tuple(
                o.clone().detach().requires_grad_(False).cuda() for o in warmup_output
            )

        self.prefill_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.prefill_graph):
            output = model(self.prefill_input, **forward_kwargs)
            if isinstance(output, torch.Tensor):
                self.prefill_output.copy_(output)
            else:
                for out, placeholder in zip(output, self.prefill_output):
                    placeholder.copy_(out)

        self._is_prefill_ready = True
        print(f"[PrefillDecodeManager] ✅ Prefill Graph 已捕獲 (input_shape: {sample_input.shape})")

        return self.prefill_graph

    def capture_decode_for_seq_len(
        self, model: torch.nn.Module, seq_len: int, sample_input: torch.Tensor, **forward_kwargs
    ) -> torch.cuda.CUDAGraph:
        input_placeholder = sample_input.clone().detach().requires_grad_(False).cuda()

        with torch.no_grad():
            warmup_output = model(input_placeholder, **forward_kwargs)

        output_placeholder = warmup_output.clone().detach().requires_grad_(False).cuda()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = model(input_placeholder, **forward_kwargs)
            output_placeholder.copy_(output)

        self.decode_cache.register_graph(
            seq_len=seq_len,
            graph=graph,
            input_placeholder=input_placeholder,
            output_placeholder=output_placeholder,
        )

        self._current_decode_seq_len = max(self._current_decode_seq_len, seq_len)
        return graph

    def replay_prefill(self, input_tensor: torch.Tensor) -> Any:
        if not self._is_prefill_ready or self.prefill_graph is None:
            raise RuntimeError("Prefill Graph 未捕獲")

        self.prefill_input.copy_(input_tensor)
        self.prefill_graph.replay()

        if isinstance(self.prefill_output, torch.Tensor):
            return self.prefill_output.clone()
        return tuple(o.clone() for o in self.prefill_output)

    def replay_decode(self, seq_len: int, input_tensor: torch.Tensor, batch_size: int = 1) -> torch.Tensor:
        graph_info = self.decode_cache.get_graph(seq_len, batch_size)

        if graph_info is None:
            raise RuntimeError(
                f"未找到 seq_len={seq_len} 的 Decode Graph，請先調用 capture_decode_for_seq_len()"
            )

        graph_info["input_placeholder"].copy_(input_tensor)
        graph_info["graph"].replay()
        return graph_info["output_placeholder"].clone()

    def get_or_create_decode_graph(
        self, model: torch.nn.Module, seq_len: int, sample_input: torch.Tensor, **forward_kwargs
    ) -> torch.cuda.CUDAGraph:
        if self.decode_cache.has_graph(seq_len):
            return self.decode_cache.get_graph(seq_len)["graph"]

        return self.capture_decode_for_seq_len(model, seq_len, sample_input, **forward_kwargs)


def benchmark_dynamic_graph(
    model: torch.nn.Module, seq_lens: List[int], num_iterations: int = 100
) -> Dict[int, Dict[str, float]]:
    results = {}
    manager = PrefillDecodeGraphManager(max_decode_cache_size=len(seq_lens) + 5)

    for seq_len in seq_lens:
        print(f"\n[ Benchmark ] 測試 seq_len={seq_len}")

        input_tensor = torch.randn(1, seq_len, 512).cuda()

        print("  - 捕獲 Graph...")
        start = time.time()
        manager.capture_decode_for_seq_len(model, seq_len, input_tensor)
        capture_time = (time.time() - start) * 1000
        print(f"  - 捕獲耗時: {capture_time:.2f} ms")

        print(f"  - 性能測試 ({num_iterations} iterations)...")

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iterations):
            _ = manager.replay_decode(seq_len, input_tensor)
        torch.cuda.synchronize()

        graph_time = (time.time() - start) * 1000 / num_iterations

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(input_tensor)
        torch.cuda.synchronize()

        eager_time = (time.time() - start) * 1000 / num_iterations
        speedup = eager_time / graph_time if graph_time > 0 else 0

        results[seq_len] = {
            "eager_ms": eager_time,
            "graph_ms": graph_time,
            "speedup": speedup,
            "capture_ms": capture_time,
        }

        print(
            f"  - Eager: {eager_time:.3f} ms | Graph: {graph_time:.3f} ms | Speedup: {speedup:.2f}x"
        )

    return results
