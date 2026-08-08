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
PD (Prefill/Decode) Scheduler - vLLM 整合

此模块实现了 PD 分离架构的调度器，与 vLLM 整合：

架构：
    [vLLM Scheduler Layer] ← 处理请求队列
    ↓
    [PD Service Layer] ← KV Cache 管理 + CGC 命令执行
    ↓
    [vLLM Worker Layer] ← 模型计算

优势：
1. Prefill/Decode 分离调度，优化不同阶段的资源使用
2. 分布式 KV Cache，支持多 GPU 和模型并行
3. CGC 命令执行器，支持 KDA/FlashKDA 加速
4. Prefix Cache，复用相同前缀的 KV
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum, auto
import logging
import time

logger = logging.getLogger(__name__)

try:
    from ..pd.pd_client import PDClient, PDClientConfig
    from ..pd.pd_client import get_pd_client
    from ..cgc.cgc_opcodes import CGC_OP_CODES
    PD_AVAILABLE = True
except ImportError:
    PD_AVAILABLE = False
    logger.warning("PD service not available, using local mode")


class Phase(Enum):
    """执行阶段"""
    PREFILL = auto()
    DECODE = auto()
    HYBRID = auto()


@dataclass
class PDSchedulerConfig:
    """PD 调度器配置"""
    pd_endpoint: str = "localhost:50051"
    enable_pd_mode: bool = True
    enable_prefix_cache: bool = True
    enable_kv_quantization: bool = True
    kv_quant_bits: int = 8
    max_prefill_batch: int = 32
    max_decode_batch: int = 128
    phase_separation_enabled: bool = True
    hybrid_phase_threshold: int = 4096  # 超过此长度使用 hybrid phase


class PDScheduler:
    """
    PD (Prefill/Decode) Scheduler - vLLM 整合

    功能：
    1. 分离 Prefill 和 Decode 阶段的调度
    2. 与 PD Service 通信管理 KV Cache
    3. 执行 CGC 命令 (KDA 等)
    4. Prefix Cache 复用
    """

    def __init__(self, config: Optional[PDSchedulerConfig] = None):
        self.config = config or PDSchedulerConfig()
        self.pd_client: Optional[PDClient] = None
        self._init_pd_client()
        
        self.active_sequences: Dict[int, Any] = {}
        self.prefix_cache_hits: int = 0
        self.total_requests: int = 0
        
        logger.info(f"[PDScheduler] Initialized: PD={PD_AVAILABLE}, endpoint={self.config.pd_endpoint}")

    def _init_pd_client(self):
        """初始化 PD 客户端"""
        if not PD_AVAILABLE:
            logger.warning("[PDScheduler] PD not available, using local mode")
            return
        
        try:
            self.pd_client = get_pd_client(self.config.pd_endpoint)
            healthy, stats = self.pd_client.health_check()
            
            if healthy:
                logger.info(f"[PDScheduler] PD service connected: {stats}")
            else:
                logger.warning(f"[PDScheduler] PD service not healthy: {stats}")
                
        except Exception as e:
            logger.error(f"[PDScheduler] Failed to connect to PD service: {e}")
            self.pd_client = None

    def determine_phase(self, input_length: int, output_length: int = 0) -> Phase:
        """
        确定执行阶段

        Args:
            input_length: 输入 token 数
            output_length: 已输出 token 数

        Returns:
            Phase: 执行阶段
        """
        if output_length == 0:
            return Phase.PREFILL
        elif not self.config.phase_separation_enabled:
            return Phase.HYBRID
        elif input_length > self.config.hybrid_phase_threshold:
            return Phase.HYBRID
        else:
            return Phase.DECODE

    def allocate_kv_blocks(self, sequence_ids: List[int], num_blocks: int = 1) -> Tuple[List[int], bool]:
        """
        分配 KV Cache 块

        Args:
            sequence_ids: 序列 ID 列表
            num_blocks: 每序列块数

        Returns:
            (block_ids, success)
        """
        if self.pd_client is None:
            return [], False
        
        return self.pd_client.allocate_blocks(sequence_ids, num_blocks)

    def store_prefix_kv(self, key: str, kv_data: bytes, ttl_seconds: int = 3600) -> bool:
        """
        存储 Prefix KV

        Args:
            key: prefix key (hash of input tokens)
            kv_data: KV 数据
            ttl_seconds: TTL

        Returns:
            success
        """
        if self.pd_client is None or not self.config.enable_prefix_cache:
            return False
        
        return self.pd_client.store_prefix(key, kv_data, ttl_seconds)

    def get_prefix_kv(self, key: str) -> Tuple[bytes, bool]:
        """
        获取 Prefix KV

        Args:
            key: prefix key

        Returns:
            (kv_data, cache_hit)
        """
        if self.pd_client is None or not self.config.enable_prefix_cache:
            return b"", False
        
        kv_data, cache_hit = self.pd_client.get_prefix(key)
        if cache_hit:
            self.prefix_cache_hits += 1
        return kv_data, cache_hit

    def execute_cgc_command(
        self,
        opcode: int,
        tensors: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, bool, str]:
        """
        执行 CGC 命令

        Args:
            opcode: 操作码
            tensors: 输入张量
            params: 参数

        Returns:
            (output, success, error)
        """
        if self.pd_client is None:
            return None, False, "PD client not available"
        
        return self.pd_client.run_cgc_command(opcode, tensors, params)

    def execute_kda_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float = 1.0,
    ) -> Tuple[torch.Tensor, bool, str]:
        """
        执行 KDA Forward

        Args:
            q, k, v: 查询/键/值张量
            scale: 缩放因子

        Returns:
            (output, success, error)
        """
        if self.pd_client is None:
            return None, False, "PD client not available"
        
        return self.pd_client.kda_forward(q, k, v, scale)

    def schedule_prefill(self, sequences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        调度 Prefill 阶段

        Args:
            sequences: 序列列表

        Returns:
            调度结果
        """
        self.total_requests += len(sequences)
        
        results = []
        for seq in sequences:
            seq_id = seq.get("sequence_id", 0)
            input_ids = seq.get("input_ids", [])
            
            # 尝试获取 prefix cache
            prefix_key = str(hash(tuple(input_ids)))
            cached_kv, cache_hit = self.get_prefix_kv(prefix_key)
            
            if cache_hit:
                results.append({
                    "sequence_id": seq_id,
                    "phase": Phase.PREFILL,
                    "cache_hit": True,
                    "cached_kv": cached_kv,
                })
            else:
                results.append({
                    "sequence_id": seq_id,
                    "phase": Phase.PREFILL,
                    "cache_hit": False,
                })
        
        return results

    def schedule_decode(self, sequences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        调度 Decode 阶段

        Args:
            sequences: 序列列表

        Returns:
            调度结果
        """
        results = []
        for seq in sequences:
            seq_id = seq.get("sequence_id", 0)
            
            results.append({
                "sequence_id": seq_id,
                "phase": Phase.DECODE,
            })
        
        return results

    def get_stats(self) -> Dict[str, Any]:
        """获取调度器统计信息"""
        stats = {
            "active_sequences": len(self.active_sequences),
            "total_requests": self.total_requests,
            "prefix_cache_hits": self.prefix_cache_hits,
            "prefix_cache_hit_rate": self.prefix_cache_hits / max(self.total_requests, 1),
        }
        
        if self.pd_client:
            healthy, pd_stats = self.pd_client.health_check()
            stats["pd_healthy"] = healthy
            stats["pd_stats"] = pd_stats
        
        return stats


class PDKVCacheManager:
    """
    PD KV Cache 管理器

    管理 vLLM 与 PD 服务之间的 KV Cache 传输
    """

    def __init__(
        self,
        pd_endpoint: str = "localhost:50051",
        enable_quantization: bool = True,
        quant_bits: int = 8,
    ):
        self.pd_endpoint = pd_endpoint
        self.enable_quantization = enable_quantization
        self.quant_bits = quant_bits
        
        self.pd_client: Optional[PDClient] = None
        self._init_pd_client()
        
        self.local_kv_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        
        logger.info(f"[PDKVCacheManager] Initialized: quantization={enable_quantization}")

    def _init_pd_client(self):
        """初始化 PD 客户端"""
        if not PD_AVAILABLE:
            logger.warning("[PDKVCacheManager] PD not available")
            return
        
        try:
            self.pd_client = get_pd_client(self.pd_endpoint)
            logger.info("[PDKVCacheManager] PD client connected")
        except Exception as e:
            logger.error(f"[PDKVCacheManager] Failed to connect: {e}")
            self.pd_client = None

    def store_kv(self, block_id: int, k: torch.Tensor, v: torch.Tensor) -> bool:
        """
        存储 KV Cache

        Args:
            block_id: 块 ID
            k, v: KV 张量

        Returns:
            success
        """
        if self.pd_client is None:
            self.local_kv_cache[block_id] = (k, v)
            return True
        
        import pickle
        try:
            k_bytes = pickle.dumps(k)
            v_bytes = pickle.dumps(v)
            self.pd_client.store_prefix(f"kv_{block_id}", k_bytes + b"|||" + v_bytes)
            return True
        except Exception as e:
            logger.error(f"[PDKVCacheManager] Store error: {e}")
            self.local_kv_cache[block_id] = (k, v)
            return True

    def load_kv(self, block_id: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        加载 KV Cache

        Args:
            block_id: 块 ID

        Returns:
            (k, v) or None
        """
        if block_id in self.local_kv_cache:
            return self.local_kv_cache[block_id]
        
        if self.pd_client is None:
            return None
        
        import pickle
        try:
            kv_data, _ = self.pd_client.get_prefix(f"kv_{block_id}")
            if kv_data:
                k_bytes, v_bytes = kv_data.split(b"|||")
                k = pickle.loads(k_bytes)
                v = pickle.loads(v_bytes)
                return k, v
        except Exception as e:
            logger.error(f"[PDKVCacheManager] Load error: {e}")
        
        return None

    def release_kv(self, block_ids: List[int]) -> int:
        """释放 KV Cache"""
        released = 0
        for block_id in block_ids:
            if block_id in self.local_kv_cache:
                del self.local_kv_cache[block_id]
                released += 1
        return released


class PDCommandExecutor:
    """
    PD 命令执行器

    执行 CGC 命令，包括 KDA/FlashKDA
    """

    def __init__(self, pd_endpoint: str = "localhost:50051"):
        self.pd_endpoint = pd_endpoint
        self.pd_client: Optional[PDClient] = None
        self._init_pd_client()
        
        logger.info("[PDCommandExecutor] Initialized")

    def _init_pd_client(self):
        if not PD_AVAILABLE:
            return
        
        try:
            self.pd_client = get_pd_client(self.pd_endpoint)
            logger.info("[PDCommandExecutor] PD client connected")
        except Exception as e:
            logger.error(f"[PDCommandExecutor] Failed to connect: {e}")

    def execute_kda_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float = 1.0,
    ) -> Tuple[torch.Tensor, bool, str]:
        """执行 KDA Forward"""
        if self.pd_client is not None:
            output, success, err = self.pd_client.kda_forward(q, k, v, scale)
            if success:
                return output, True, err
        
        # Fallback to local SDPA
        logger.warning("[PDCommandExecutor] Falling back to SDPA")
        return self._fallback_sdpa(q, k, v, scale)

    def _fallback_sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
    ) -> Tuple[torch.Tensor, bool, str]:
        """Fallback to SDPA"""
        import torch.nn.functional as F
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        return output, True, ""


def create_pd_scheduler(
    pd_endpoint: str = "localhost:50051",
    enable_prefix_cache: bool = True,
    enable_kv_quantization: bool = True,
) -> PDScheduler:
    """
    创建 PD 调度器

    Args:
        pd_endpoint: PD 服务端点
        enable_prefix_cache: 是否启用 prefix cache
        enable_kv_quantization: 是否启用 KV 量化

    Returns:
        PDScheduler 实例
    """
    config = PDSchedulerConfig(
        pd_endpoint=pd_endpoint,
        enable_prefix_cache=enable_prefix_cache,
        enable_kv_quantization=enable_kv_quantization,
    )
    return PDScheduler(config)


def create_pd_kv_manager(
    pd_endpoint: str = "localhost:50051",
    enable_quantization: bool = True,
) -> PDKVCacheManager:
    """创建 KV 管理器"""
    return PDKVCacheManager(pd_endpoint, enable_quantization)


def create_pd_command_executor(pd_endpoint: str = "localhost:50051") -> PDCommandExecutor:
    """创建命令执行器"""
    return PDCommandExecutor(pd_endpoint)


# Convenience wrapper for vLLM integration
class VLLMPDIntegration:
    """
    vLLM + PD 集成包装器

    一站式使用 PD 调度、KV 管理和 CGC 命令执行
    """

    def __init__(self, pd_endpoint: str = "localhost:50051"):
        self.scheduler = create_pd_scheduler(pd_endpoint)
        self.kv_manager = create_pd_kv_manager(pd_endpoint)
        self.command_executor = create_pd_command_executor(pd_endpoint)
        
        logger.info("[VLLMPDIntegration] All components initialized")

    def health_check(self) -> Tuple[bool, Dict[str, Any]]:
        """健康检查"""
        if self.scheduler.pd_client is None:
            return False, {"error": "PD client not available"}
        
        return self.scheduler.pd_client.health_check()
