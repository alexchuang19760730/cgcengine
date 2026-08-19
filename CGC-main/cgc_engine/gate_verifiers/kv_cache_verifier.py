"""kv_cache_verifier.py — Gate 2.2 KV cache source-contract verifiers.

这 4 个能力都不适合依赖本地环境直接 import/启动完整 vendored SGLang。
因此这里采用与其他 Gate 2.0 foundation verifier 一致的 source-contract 方式，
直接锚定到仓库中的真实实现文件，而不是返回无条件 PASS。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .base import BaseVerifier, VerificationStatus


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_source(rel_path: str) -> str:
    path = _repo_root() / rel_path
    return path.read_text(encoding="utf-8")


def _contains_all(source: str, markers: list[str]) -> bool:
    return all(marker in source for marker in markers)


class KVCacheManagementVerifier(BaseVerifier):
    """KV 缓存管理（分配与回收）验证器"""
    capability = "kv_cache_management"

    def verify(self) -> Dict[str, Any]:
        start = self._start()
        try:
            builder_source = _read_source(
                "Backend/CGC/cloud_sglang/python/sglang/srt/mem_cache/kv_cache_builder.py"
            )
            init_params_source = _read_source(
                "Backend/CGC/cloud_sglang/python/sglang/srt/mem_cache/cache_init_params.py"
            )
            markers = [
                "def build_kv_cache(",
                "req_to_token_pool, token_to_kv_pool_allocator = tp_worker.get_memory_pool()",
                "token_to_kv_pool_allocator=token_to_kv_pool_allocator",
                "eviction_policy=server_args.radix_eviction_policy",
                "tree_cache = create_tree_cache(",
            ]
            has_markers = _contains_all(builder_source, markers) and _contains_all(
                init_params_source,
                [
                    "class CacheInitParams:",
                    "token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator",
                    'eviction_policy: str = "lru"',
                ],
            )
            self._add_metric("source_contract_builder", has_markers)
            self._add_metric("allocator_binding", "token_to_kv_pool_allocator")
            self._add_metric("eviction_policy", "lru")
            if not has_markers:
                return self._finish(
                    start,
                    VerificationStatus.FAIL,
                    "kv cache management source contract missing",
                )
            self._add_evidence(
                "[kv_cache_management] build_kv_cache binds req_to_token_pool + token_to_kv_pool_allocator and passes radix eviction policy into CacheInitParams/create_tree_cache"
            )
            return self._finish(start, VerificationStatus.PASS)
        except Exception as e:
            return self._finish(start, VerificationStatus.FAIL, str(e))


class KVCacheReuseVerifier(BaseVerifier):
    """缓存复用优化（多轮对话）验证器"""
    capability = "kv_cache_reuse"

    def verify(self) -> Dict[str, Any]:
        start = self._start()
        try:
            radix_source = _read_source(
                "Backend/CGC/cloud_sglang/python/sglang/srt/mem_cache/radix_cache_cpp.py"
            )
            markers = [
                "new_indices_vec, _, new_last_node, _ = self.tree.match_prefix(",
                "reused_indices = new_indices[old_prefix_len:new_prefix_len]",
                "self.req_to_token_pool.req_to_token[",
                "req.prefix_indices = torch.cat(",
                "req.prefix_indices = new_indices",
            ]
            has_markers = _contains_all(radix_source, markers)
            self._add_metric("reuse_strategy", "prefix_match")
            self._add_metric("multi_turn_supported", has_markers)
            if not has_markers:
                return self._finish(
                    start,
                    VerificationStatus.FAIL,
                    "kv cache reuse source contract missing",
                )
            self._add_evidence(
                "[kv_cache_reuse] radix_cache_cpp caches unfinished requests via match_prefix, reuses prefix indices already in pool, and writes reused_indices back into req_to_token_pool"
            )
            return self._finish(start, VerificationStatus.PASS)
        except Exception as e:
            return self._finish(start, VerificationStatus.FAIL, str(e))


class KVDynamicSizingVerifier(BaseVerifier):
    """动态缓存大小验证器"""
    capability = "kv_dynamic_sizing"

    def verify(self) -> Dict[str, Any]:
        start = self._start()
        try:
            builder_source = _read_source(
                "Backend/CGC/cloud_sglang/python/sglang/srt/mem_cache/kv_cache_builder.py"
            )
            host_pool_source = _read_source(
                "Backend/CGC/cloud_sglang/python/sglang/srt/mem_cache/memory_pool_host.py"
            )
            init_params_source = _read_source(
                "Backend/CGC/cloud_sglang/python/sglang/srt/mem_cache/cache_init_params.py"
            )
            has_markers = _contains_all(
                builder_source,
                [
                    "effective_chunked_prefill_size = server_args.chunked_prefill_size",
                    "sliding_window_size = tp_worker.sliding_window_size",
                    "chunked_prefill_size=effective_chunked_prefill_size",
                    "sliding_window_size=sliding_window_size",
                ],
            ) and _contains_all(
                init_params_source,
                [
                    "chunked_prefill_size: Optional[int] = None",
                    "sliding_window_size: Optional[int] = None",
                ],
            ) and _contains_all(
                host_pool_source,
                [
                    "def _round_up_to_page_size(self, size: int) -> int:",
                    "page_end = self._round_up_to_page_size(end_pos)",
                    "num_new_pages = (page_end - allocated_len) // self.page_size",
                ],
            )
            self._add_metric("sizing_policy", "adaptive_page_and_prefill")
            self._add_metric("source_contract_dynamic_sizing", has_markers)
            if not has_markers:
                return self._finish(
                    start,
                    VerificationStatus.FAIL,
                    "kv dynamic sizing source contract missing",
                )
            self._add_evidence(
                "[kv_dynamic_sizing] kv_cache_builder threads chunked_prefill_size/sliding_window_size into CacheInitParams, and memory_pool_host grows host slots by rounded page allocations"
            )
            return self._finish(start, VerificationStatus.PASS)
        except Exception as e:
            return self._finish(start, VerificationStatus.FAIL, str(e))


class KVCachePrefetchingVerifier(BaseVerifier):
    """缓存预取优化验证器"""
    capability = "kv_cache_prefetching"

    def verify(self) -> Dict[str, Any]:
        start = self._start()
        try:
            mooncake_source = _read_source(
                "Backend/CGC/cloud_sglang/python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py"
            )
            async_prefetch_source = _read_source("cgc_engine/pd/kv_async_prefetch.py")
            has_markers = _contains_all(
                mooncake_source,
                [
                    'class MooncakeTransferEngine:',
                    "def transfer_sync(",
                    "def batch_transfer_sync(",
                    "def init_mooncake_transfer_engine(",
                ],
            ) and _contains_all(
                async_prefetch_source,
                [
                    "self._prefetch_queue: deque = deque()",
                    "self._prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)",
                    "def submit_prefetch(self, request: PrefetchRequest) -> asyncio.Future:",
                    "def _do_prefetch(self, request: PrefetchRequest) -> PrefetchResult:",
                    "loaded = self.pd_client.load_kv(block_id)",
                ],
            )
            self._add_metric("prefetch_strategy", "async_queue_plus_mooncake_transfer")
            self._add_metric("source_contract_prefetch", has_markers)
            if not has_markers:
                return self._finish(
                    start,
                    VerificationStatus.FAIL,
                    "kv cache prefetching source contract missing",
                )
            self._add_evidence(
                "[kv_cache_prefetching] async prefetch worker queues and fills KV blocks via PD client, while vendored Mooncake transfer engine provides transfer_sync/batch_transfer_sync primitives"
            )
            return self._finish(start, VerificationStatus.PASS)
        except Exception as e:
            return self._finish(start, VerificationStatus.FAIL, str(e))
