"""
MagiCompiler Phase 3: Prefill-Decode 流水線優化
Optimized Prefill-Decode Pipeline for vLLM Integration
"""

import torch
import time
from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass
from queue import Queue
import threading

from magi_attention_backend import MagiKVAttentionBackend, AttentionConfig


@dataclass
class InferenceRequest:
    """推理請求"""
    prompt: str
    prompt_tokens: List[int]
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0


@dataclass
class InferenceState:
    """推理狀態"""
    request_id: int
    prompt_tokens: List[int]
    output_tokens: List[int] = None
    kv_cache_key: Optional[torch.Tensor] = None
    kv_cache_value: Optional[torch.Tensor] = None
    current_pos: int = 0
    is_prefilled: bool = False
    is_finished: bool = False

    def __post_init__(self):
        if self.output_tokens is None:
            self.output_tokens = []


class MagiPipeline:
    """
    MagiCompiler Prefill-Decode 流水線

    核心特性：
    1. Prefill-Decode 流水線優化
    2. 批量請求處理
    3. 動態 KV Cache 管理
    4. CUDA Graph 加速支持
    """

    def __init__(
        self,
        attention_backend: Optional[MagiKVAttentionBackend] = None,
        max_batch_size: int = 32,
        enable_cudagraph: bool = True,
    ):
        """
        初始化流水線

        Args:
            attention_backend: 注意力後端
            max_batch_size: 最大批量大小
            enable_cudagraph: 是否啟用 CUDA Graph
        """
        self.attention_backend = attention_backend or MagiKVAttentionBackend()
        self.max_batch_size = max_batch_size
        self.enable_cudagraph = enable_cudagraph

        # 請求隊列
        self.request_queue = Queue()
        self.processing_queue = []

        # 推理狀態管理
        self.inference_states: Dict[int, InferenceState] = {}
        self.next_request_id = 0

        # 流水線狀態
        self._prefill_busy = False
        self._decode_busy = False

        # 統計信息
        self.stats = {
            "total_requests": 0,
            "prefill_time_ms": 0,
            "decode_time_ms": 0,
            "total_tokens_generated": 0,
            "throughput": 0,
        }

    def submit_request(self, request: InferenceRequest) -> int:
        """
        提交推理請求

        Args:
            request: 推理請求

        Returns:
            請求 ID
        """
        request_id = self.next_request_id
        self.next_request_id += 1

        # 創建推理狀態
        self.inference_states[request_id] = InferenceState(
            request_id=request_id,
            prompt_tokens=request.prompt_tokens,
            output_tokens=[],
        )

        # 添加到隊列
        self.request_queue.put((request_id, request))
        self.stats["total_requests"] += 1

        return request_id

    def _batch_prefill(self, batch: List[Tuple[int, InferenceRequest]]) -> None:
        """
        批量 Prefill 處理

        Args:
            batch: 批量請求
        """
        if not batch:
            return

        start_time = time.time()

        # 提取數據
        request_ids = [req_id for req_id, _ in batch]
        requests = [req for _, req in batch]

        # 找到最長序列
        max_seq_len = max(len(req.prompt_tokens) for req in requests)

        # 構建 batch 輸入（假設已處理為張量）
        # 這裡需要與 vLLM 的 KV Cache 格式對齊
        batch_query = torch.randn(len(batch), max_seq_len, 32, 128).cuda()
        batch_key = torch.randn(len(batch), max_seq_len, 8, 128).cuda()
        batch_value = torch.randn(len(batch), max_seq_len, 8, 128).cuda()

        # 執行 Prefill
        output = self.attention_backend.prefill(
            batch_query,
            batch_key,
            batch_value,
            use_graph=self.enable_cudagraph,
            causal=True,
        )

        # 更新狀態
        for i, req_id in enumerate(request_ids):
            self.inference_states[req_id].is_prefilled = True
            self.inference_states[req_id].current_pos = len(requests[i].prompt_tokens)
            self.inference_states[req_id].kv_cache_key = batch_key[i:i+1]
            self.inference_states[req_id].kv_cache_value = batch_value[i:i+1]

        prefill_time = (time.time() - start_time) * 1000
        self.stats["prefill_time_ms"] += prefill_time

        print(f"[MagiPipeline] Prefill 完成: {len(batch)} requests, {prefill_time:.2f} ms")

    def _batch_decode(self, batch: List[Tuple[int, InferenceState]]) -> bool:
        """
        批量 Decode 處理

        Args:
            batch: 批量推理狀態

        Returns:
            是否還有未完成的請求
        """
        if not batch:
            return False

        start_time = time.time()

        # 找到最長序列
        max_seq_len = max(state.current_pos for _, state in batch)

        # 構建 batch 輸入
        batch_query = torch.randn(len(batch), 1, 32, 128).cuda()

        # 執行 Decode（使用統一的 KV Cache 大小）
        output = self.attention_backend.decode(
            batch_query,
            use_graph=self.enable_cudagraph,
            causal=True,
        )

        # 更新狀態
        active_count = 0
        for i, (req_id, state) in enumerate(batch):
            # 模擬 token 生成
            new_token = 1  # 實際應該是模型輸出

            state.output_tokens.append(new_token)
            state.current_pos += 1

            # 檢查是否完成
            if len(state.output_tokens) >= state.max_tokens:
                state.is_finished = True
            else:
                active_count += 1

        decode_time = (time.time() - start_time) * 1000
        self.stats["decode_time_ms"] += decode_time
        self.stats["total_tokens_generated"] += len(batch)

        print(f"[MagiPipeline] Decode 完成: {len(batch)} requests, {decode_time:.2f} ms")

        return active_count > 0

    def run(self, max_iterations: int = 1000) -> Dict[int, InferenceState]:
        """
        運行流水線

        Args:
            max_iterations: 最大迭代次數

        Returns:
            所有請求的最終狀態
        """
        start_time = time.time()

        for iteration in range(max_iterations):
            # 階段 1: 收集待處理的請求
            while not self.request_queue.empty() and len(self.processing_queue) < self.max_batch_size:
                req_id, req = self.request_queue.get()
                self.processing_queue.append((req_id, req))

            # 如果沒有待處理的請求，檢查是否完成
            if not self.processing_queue:
                # 檢查是否還有未完成的推理
                active_states = [s for s in self.inference_states.values() if not s.is_finished]
                if not active_states:
                    break
                continue

            # 階段 2: Prefill 未處理的請求
            prefill_batch = [
                (req_id, req) for req_id, req in self.processing_queue
                if not self.inference_states[req_id].is_prefilled
            ]

            if prefill_batch:
                self._batch_prefill(prefill_batch)

            # 階段 3: Decode 已 Prefill 的請求
            decode_batch = [
                (req_id, self.inference_states[req_id]) for req_id, _ in self.processing_queue
                if self.inference_states[req_id].is_prefilled and not self.inference_states[req_id].is_finished
            ]

            if decode_batch:
                has_active = self._batch_decode(decode_batch)

                # 移除完成的請求
                self.processing_queue = [
                    (req_id, req) for req_id, req in self.processing_queue
                    if not self.inference_states[req_id].is_finished
                ]

                if not has_active and not self.processing_queue:
                    break

        # 計算吞吐量
        total_time = time.time() - start_time
        self.stats["throughput"] = self.stats["total_tokens_generated"] / total_time if total_time > 0 else 0

        print(f"\n[MagiPipeline] 推理完成")
        print(f"  總請求數: {self.stats['total_requests']}")
        print(f"  總生成 tokens: {self.stats['total_tokens_generated']}")
        print(f"  總時間: {total_time:.2f} s")
        print(f"  吞吐量: {self.stats['throughput']:.2f} tokens/s")

        return self.inference_states


class AsyncMagiPipeline(MagiPipeline):
    """
    異步版本的 MagiPipeline
    支持 Prefill 和 Decode 的並行執行
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prefill_thread = None
        self._decode_thread = None
        self._shutdown_event = threading.Event()

    def _prefill_worker(self):
        """Prefill 工作線程"""
        while not self._shutdown_event.is_set():
            # 收集待 Prefill 的請求
            prefill_batch = []
            while not self.request_queue.empty() and len(prefill_batch) < self.max_batch_size:
                req_id, req = self.request_queue.get()
                if not self.inference_states[req_id].is_prefilled:
                    prefill_batch.append((req_id, req))

            if prefill_batch:
                self._batch_prefill(prefill_batch)

            time.sleep(0.001)

    def _decode_worker(self):
        """Decode 工作線程"""
        while not self._shutdown_event.is_set():
            # 收集待 Decode 的請求
            decode_batch = [
                (req_id, self.inference_states[req_id])
                for req_id in self.inference_states
                if self.inference_states[req_id].is_prefilled and not self.inference_states[req_id].is_finished
            ]

            if decode_batch:
                self._batch_decode(decode_batch)

            time.sleep(0.001)

    def start(self):
        """啟動異步線程"""
        self._prefill_thread = threading.Thread(target=self._prefill_worker, daemon=True)
        self._decode_thread = threading.Thread(target=self._decode_worker, daemon=True)

        self._prefill_thread.start()
        self._decode_thread.start()

        print("[AsyncMagiPipeline] 異步流水線已啟動")

    def stop(self):
        """停止異步線程"""
        self._shutdown_event.set()

        if self._prefill_thread:
            self._prefill_thread.join(timeout=1)
        if self._decode_thread:
            self._decode_thread.join(timeout=1)

        print("[AsyncMagiPipeline] 異步流水線已停止")


def create_pipeline(
    async_mode: bool = False,
    **kwargs
) -> Union[MagiPipeline, AsyncMagiPipeline]:
    """
    工廠函數：創建流水線

    Args:
        async_mode: 是否使用異步模式
        kwargs: 流水線參數

    Returns:
        MagiPipeline 或 AsyncMagiPipeline 實例
    """
    if async_mode:
        return AsyncMagiPipeline(**kwargs)
    else:
        return MagiPipeline(**kwargs)
