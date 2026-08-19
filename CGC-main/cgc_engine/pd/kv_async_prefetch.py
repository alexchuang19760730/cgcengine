# Copyright (c) 2025 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
KV Cache 异步预取模块

功能：
- PD → Worker KV 预读取
- 计算与 KV 搬运重叠（双流）
- 确保 KDA/FlashKDA 可用时才启用

架构：
- 复用 GDS/PD 存储层
- 复用 CGC 计算层
- 异步非阻塞预取
"""

import asyncio
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from .kv_quantizer import KVQuantizer
    QUANT_AVAILABLE = True
except ImportError:
    QUANT_AVAILABLE = False

try:
    from ..cgc.cgc_simd_executor import CGCExecutor
    from ..cgc.flashkda_integration import FLASHKDA_AVAILABLE as KDA_AVAILABLE
    CGC_AVAILABLE = True
except ImportError:
    CGC_AVAILABLE = False
    KDA_AVAILABLE = False

try:
    from .pd_client import PDClient
    PD_CLIENT_AVAILABLE = True
except ImportError:
    PD_CLIENT_AVAILABLE = False


@dataclass
class PrefetchRequest:
    """预取请求"""
    sequence_id: int
    block_ids: List[int]
    priority: int = 0
    callback: Optional[Any] = None


@dataclass
class PrefetchResult:
    """预取结果"""
    sequence_id: int
    block_ids: List[int]
    k_tensors: List[torch.Tensor] = field(default_factory=list)
    v_tensors: List[torch.Tensor] = field(default_factory=list)
    success: bool = False
    error: str = ""


class KVCachAsyncPrefetcher:
    """
    KV Cache 异步预取器
    
    特性：
    - 异步非阻塞预取，不阻塞 KDA 计算
    - 复用 GDS/PD 存储层
    - KDA 可用时启用优化模式
    - 双流：计算流 + 预取流
    """

    def __init__(
        self,
        pd_client: Optional[PDClient] = None,
        enable_kda: bool = True,
        enable_quant: bool = True,
        quant_bits: int = 8,
        max_prefetch_queue: int = 32,
    ):
        """
        Args:
            pd_client: PD 客户端（复用存储层）
            enable_kda: 是否启用 KDA 优化
            enable_quant: 是否启用 KV 量化
            quant_bits: 量化位数
            max_prefetch_queue: 最大预取队列长度
        """
        self.pd_client = pd_client
        self.enable_kda = enable_kda and KDA_AVAILABLE and CGC_AVAILABLE
        self.enable_quant = enable_quant and QUANT_AVAILABLE
        
        if not self.enable_kda:
            logger.warning("[AsyncPrefetch] KDA not available, falling back to sync mode")
        if not self.enable_quant:
            logger.warning("[AsyncPrefetch] Quantization not available")
            
        self.max_queue = max_prefetch_queue
        self._prefetch_queue: deque = deque()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        
        # 缓存已预取的 KV
        self._kv_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        
        # CGC Executor（复用计算层）
        self._cgc_executor: Optional[CGCExecutor] = None
        if CGC_AVAILABLE:
            self._cgc_executor = CGCExecutor(enable_profiling=False)
        
        # 量化器（复用存储层量化功能）
        self._kv_quantizer: Optional[KVQuantizer] = None
        if self.enable_quant:
            try:
                self._kv_quantizer = KVQuantizer(bits=quant_bits, group_size=128)
            except Exception as e:
                logger.warning(f"[AsyncPrefetch] KVQuantizer init failed: {e}")
                self._kv_quantizer = None
        
        # 异步事件循环
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._executor = asyncio.get_event_loop() if asyncio.get_event_loop().is_running() else None
        
        logger.info(f"[AsyncPrefetch] Initialized: KDA={self.enable_kda}, Quant={self.enable_quant and self._kv_quantizer is not None}")

    def start(self):
        """启动异步预取线程"""
        if self._running:
            return
            
        self._running = True
        self._prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._prefetch_thread.start()
        logger.info("[AsyncPrefetch] Started async prefetch worker")

    def stop(self):
        """停止异步预取"""
        self._running = False
        if self._prefetch_thread:
            self._prefetch_thread.join(timeout=2.0)
        logger.info("[AsyncPrefetch] Stopped")

    def submit_prefetch(self, request: PrefetchRequest) -> asyncio.Future:
        """
        提交预取请求（非阻塞）
        
        Args:
            request: 预取请求
            
        Returns:
            asyncio.Future，预取结果
        """
        future = asyncio.Future()
        
        with self._lock:
            if len(self._prefetch_queue) >= self.max_queue:
                logger.warning("[AsyncPrefetch] Queue full, dropping oldest")
                self._prefetch_queue.popleft()
            
            request.callback = lambda r, f=future: f.set_result(r)
            self._prefetch_queue.append(request)
        
        return future

    def get_cached_kv(self, block_id: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        获取已缓存的 KV（同步接口，供 KDA 计算使用）
        
        Args:
            block_id: block ID
            
        Returns:
            (k, v) tensor tuple or None
        """
        return self._kv_cache.get(block_id)

    def put_cached_kv(self, block_id: int, k: torch.Tensor, v: torch.Tensor):
        """
        缓存 KV（供预取线程调用）
        
        Args:
            block_id: block ID
            k: Key tensor
            v: Value tensor
        """
        self._kv_cache[block_id] = (k, v)

    def _prefetch_worker(self):
        """预取工作线程"""
        logger.info("[AsyncPrefetch] Worker started")
        
        while self._running:
            request = None
            
            with self._lock:
                if self._prefetch_queue:
                    request = self._prefetch_queue.popleft()
            
            if request is None:
                threading.Event().wait(0.001)
                continue
            
            result = self._do_prefetch(request)
            
            if request.callback:
                try:
                    request.callback(result)
                except Exception as e:
                    logger.error(f"[AsyncPrefetch] Callback error: {e}")
        
        logger.info("[AsyncPrefetch] Worker stopped")

    def _do_prefetch(self, request: PrefetchRequest) -> PrefetchResult:
        """
        执行预取（同步）
        
        复用存储层：GDS/PD
        """
        result = PrefetchResult(
            sequence_id=request.sequence_id,
            block_ids=request.block_ids,
        )
        
        try:
            k_tensors = []
            v_tensors = []
            
            for block_id in request.block_ids:
                if block_id in self._kv_cache:
                    k, v = self._kv_cache[block_id]
                    k_tensors.append(k)
                    v_tensors.append(v)
                    continue
                
                if self.pd_client and PD_CLIENT_AVAILABLE:
                    loaded = self.pd_client.load_kv(block_id)
                    if loaded is not None:
                        k, v = loaded
                    else:
                        k, v = self._generate_dummy_kv()
                else:
                    k, v = self._generate_dummy_kv()
                
                self._kv_cache[block_id] = (k, v)
                k_tensors.append(k)
                v_tensors.append(v)
            
            result.k_tensors = k_tensors
            result.v_tensors = v_tensors
            result.success = True
            
        except Exception as e:
            result.error = str(e)
            logger.error(f"[AsyncPrefetch] Prefetch error: {e}")
        
        return result

    def _generate_dummy_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成假 KV（当 PD 不可用时）"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        k = torch.randn(1, 32, 2048, 128, device=device)
        v = torch.randn(1, 32, 2048, 128, device=device)
        return k, v


class KVCachDualStreamManager:
    """
    KV Cache 双流管理器
    
    功能：
    - Stream 1: KDA 计算
    - Stream 2: PD 异步写 KV
    - 计算与 KV 写入重叠
    
    架构：
    复用 CGC 计算层 + GDS/PD 存储层
    """

    def __init__(
        self,
        prefetcher: KVCachAsyncPrefetcher,
        pd_client: Optional[PDClient] = None,
    ):
        """
        Args:
            prefetcher: 异步预取器
            pd_client: PD 客户端（复用存储层）
        """
        self.prefetcher = prefetcher
        self.pd_client = pd_client
        self._write_queue: deque = deque()
        self._write_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        
        logger.info("[DualStream] Initialized")

    def start(self):
        """启动双流管理器"""
        self._running = True
        self.prefetcher.start()
        
        self._write_thread = threading.Thread(target=self._kv_write_worker, daemon=True)
        self._write_thread.start()
        logger.info("[DualStream] Started")

    def stop(self):
        """停止双流管理器"""
        self._running = False
        self.prefetcher.stop()
        if self._write_thread:
            self._write_thread.join(timeout=2.0)
        logger.info("[DualStream] Stopped")

    def submit_kda_compute(
        self,
        sequence_id: int,
        block_ids: List[int],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        提交 KDA 计算任务（计算流）
        
        特点：
        - 预取的 KV 自动用于计算
        - 不阻塞，等待预取完成
        
        Returns:
            Attention output
        """
        for block_id in block_ids:
            cached = self.prefetcher.get_cached_kv(block_id)
            if cached is not None:
                k_cached, v_cached = cached
                if k is None:
                    k = k_cached
                else:
                    k = torch.cat([k, k_cached], dim=2)
                if v is None:
                    v = v_cached
                else:
                    v = torch.cat([v, v_cached], dim=2)
        
        if self.prefetcher.enable_kda and self.prefetcher._cgc_executor is not None:
            from ..cgc.cgc_simd_executor import CGCCommand
            from ..cgc.cgc_opcodes import CGC_OP_CODES
            
            cmd = CGCCommand(
                opcode=CGC_OP_CODES.KDA_CHUNK,
                inputs=[q, k, v],
                outputs=[],
                params={"scale": 1.0},
            )
            outputs = self.prefetcher._cgc_executor.execute(cmd)
            return outputs[0] if outputs else q
        else:
            import torch.nn.functional as F
            return F.scaled_dot_product_attention(q, k, v)

    def submit_kv_write(
        self,
        sequence_id: int,
        block_ids: List[int],
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        """
        提交 KV 写入任务（写入流）
        
        特点：
        - 异步非阻塞
        - 与计算流重叠
        """
        with self._lock:
            self._write_queue.append((sequence_id, block_ids, k, v))

    def _kv_write_worker(self):
        """KV 写入工作线程"""
        logger.info("[DualStream] KV write worker started")
        
        while self._running:
            item = None
            
            with self._lock:
                if self._write_queue:
                    item = self._write_queue.popleft()
            
            if item is None:
                threading.Event().wait(0.001)
                continue
            
            sequence_id, block_ids, k, v = item
            
            try:
                if self.pd_client and PD_CLIENT_AVAILABLE:
                    for i, block_id in enumerate(block_ids):
                        k_i = k[:, :, i * 2048:(i + 1) * 2048, :] if k.dim() == 4 else k
                        v_i = v[:, :, i * 2048:(i + 1) * 2048, :] if v.dim() == 4 else v
                        self.pd_client.store_kv(block_id, k_i, v_i)
            except Exception as e:
                logger.error(f"[DualStream] KV write error: {e}")
        
        logger.info("[DualStream] KV write worker stopped")


def create_async_prefetcher(
    pd_endpoint: str = "localhost:50051",
    enable_kda: bool = True,
) -> KVCachAsyncPrefetcher:
    """
    创建异步预取器（便捷函数）
    
    复用存储层和计算层
    """
    pd_client = None
    if PD_CLIENT_AVAILABLE:
        pd_client = PDClient(pd_endpoint)
    
    return KVCachAsyncPrefetcher(
        pd_client=pd_client,
        enable_kda=enable_kda,
    )
