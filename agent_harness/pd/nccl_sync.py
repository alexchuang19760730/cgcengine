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
PD NCCL Multi-Node Synchronization

功能:
- 多机 KV Cache 同步
- Prefix Cache 广播
- KDA 正交基同步
- 集合通信 (AllReduce, AllGather, ReduceScatter)
"""

import torch
import torch.distributed as dist
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import threading
import pickle

try:
    import NCCL
    NCCL_AVAILABLE = True
except ImportError:
    NCCL_AVAILABLE = False


@dataclass
class NCCLConfig:
    """NCCL 配置"""
    backend: str = "nccl"
    init_method: str = "tcp://localhost:29500"
    world_size: int = 1
    rank: int = 0
    master_addr: str = "localhost"
    master_port: int = 29500

    def to_dist_init(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "init_method": f"tcp://{self.master_addr}:{self.master_port}",
            "world_size": self.world_size,
            "rank": self.rank,
        }


class NCCLSynchronizer:
    """
    NCCL 多机同步器

    功能:
    - KV Cache 同步
    - Prefix Cache 广播
    - KDA 正交基 AllReduce
    - 集合通信原语
    """

    def __init__(self, config: Optional[NCCLConfig] = None):
        self.config = config or NCCLConfig()
        self.initialized = False
        self._lock = threading.RLock()

    def initialize(self) -> bool:
        """
        初始化 NCCL

        Returns:
            success
        """
        if self.initialized:
            return True

        if not torch.cuda.is_available():
            print("[NCCL] CUDA not available, NCCL disabled")
            return False

        try:
            if dist.is_initialized():
                print("[NCCL] Already initialized")
                self.initialized = True
                return True

            dist.init_process_group(
                backend=self.config.backend,
                init_method=f"tcp://{self.config.master_addr}:{self.config.master_port}",
                world_size=self.config.world_size,
                rank=self.config.rank,
            )

            torch.cuda.set_device(self.config.rank % torch.cuda.device_count())

            self.initialized = True
            print(f"[NCCL] Initialized: rank={self.config.rank}, world_size={self.config.world_size}")
            return True

        except Exception as e:
            print(f"[NCCL] Failed to initialize: {e}")
            return False

    def finalize(self):
        """关闭 NCCL"""
        if self.initialized:
            dist.destroy_process_group()
            self.initialized = False
            print("[NCCL] Finalized")

    # =========================================================================
    # KV Cache Synchronization
    # =========================================================================

    def sync_kv_cache(self, kv_tensor: torch.Tensor, src_rank: int = 0) -> torch.Tensor:
        """
        同步 KV Cache (Broadcast)

        Args:
            kv_tensor: KV tensor
            src_rank: 源节点 rank

        Returns:
            同步后的 tensor
        """
        if not self.initialized:
            return kv_tensor

        with self._lock:
            try:
                if self.config.rank == src_rank:
                    dist.broadcast(kv_tensor, src=src_rank)
                else:
                    dist.broadcast(kv_tensor, src=src_rank)

                return kv_tensor

            except Exception as e:
                print(f"[NCCL] sync_kv_cache failed: {e}")
                return kv_tensor

    def allgather_kv_caches(
        self,
        local_kv: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Gather 所有节点的 KV Cache

        Args:
            local_kv: 本地 KV tensor

        Returns:
            所有节点的 KV tensor 列表
        """
        if not self.initialized:
            return [local_kv]

        with self._lock:
            try:
                world_size = dist.get_world_size()
                gathered = [torch.zeros_like(local_kv) for _ in range(world_size)]

                dist.all_gather(gathered, local_kv)

                return gathered

            except Exception as e:
                print(f"[NCCL] allgather_kv_caches failed: {e}")
                return [local_kv]

    def reduce_scatter_kv(
        self,
        kv_tensors: List[torch.Tensor],
        op: str = "sum",
    ) -> torch.Tensor:
        """
        Reduce Scatter KV Cache

        Args:
            kv_tensors: 所有节点的 KV tensors
            op: 操作类型 (sum, avg, max, min)

        Returns:
            聚合后的 tensor
        """
        if not self.initialized:
            return kv_tensors[self.config.rank]

        with self._lock:
            try:
                world_size = dist.get_world_size()
                local_kv = kv_tensors[self.config.rank]

                if op == "sum":
                    dist.reduce_scatter_tensor(
                        torch.zeros_like(local_kv),
                        [kv.clone() for kv in kv_tensors],
                        op=dist.ReduceOp.SUM,
                    )
                elif op == "avg":
                    dist.reduce_scatter_tensor(
                        torch.zeros_like(local_kv),
                        [kv.clone() for kv in kv_tensors],
                        op=dist.ReduceOp.SUM,
                    )
                    local_kv = local_kv / world_size

                return local_kv

            except Exception as e:
                print(f"[NCCL] reduce_scatter_kv failed: {e}")
                return kv_tensors[self.config.rank]

    # =========================================================================
    # KDA Ortho Basis Synchronization
    # =========================================================================

    def allreduce_kda_basis(
        self,
        basis_tensor: torch.Tensor,
        op: str = "avg",
    ) -> torch.Tensor:
        """
        AllReduce KDA 正交基

        Args:
            basis_tensor: 正交基 tensor
            op: 操作 (sum, avg)

        Returns:
            聚合后的 tensor
        """
        if not self.initialized:
            return basis_tensor

        with self._lock:
            try:
                if op == "sum":
                    dist.all_reduce(basis_tensor, op=dist.ReduceOp.SUM)
                elif op == "avg":
                    dist.all_reduce(basis_tensor, op=dist.ReduceOp.SUM)
                    basis_tensor = basis_tensor / self.config.world_size

                return basis_tensor

            except Exception as e:
                print(f"[NCCL] allreduce_kda_basis failed: {e}")
                return basis_tensor

    def broadcast_kda_basis(
        self,
        basis_tensor: torch.Tensor,
        src_rank: int = 0,
    ) -> torch.Tensor:
        """
        Broadcast KDA 正交基

        Args:
            basis_tensor: 正交基 tensor
            src_rank: 源节点

        Returns:
            广播后的 tensor
        """
        if not self.initialized:
            return basis_tensor

        with self._lock:
            try:
                dist.broadcast(basis_tensor, src=src_rank)
                return basis_tensor

            except Exception as e:
                print(f"[NCCL] broadcast_kda_basis failed: {e}")
                return basis_tensor

    # =========================================================================
    # Prefix Cache Synchronization
    # =========================================================================

    def sync_prefix_cache(
        self,
        prefix_key: str,
        prefix_data: bytes,
    ) -> Optional[bytes]:
        """
        同步 Prefix Cache (Broadcast)

        Args:
            prefix_key: Prefix key
            prefix_data: Prefix data

        Returns:
            同步后的 data (所有节点返回相同值)
        """
        if not self.initialized:
            return prefix_data

        with self._lock:
            try:
                device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

                if self.config.rank == 0:
                    payload = bytes(prefix_data) if isinstance(prefix_data, (bytes, bytearray)) else pickle.dumps(prefix_data)
                    n_bytes = torch.tensor([len(payload)], dtype=torch.int64, device=device)
                    dist.broadcast(n_bytes, src=0)
                    if int(n_bytes.item()) <= 0:
                        return b""
                    data_tensor = torch.tensor(list(payload), dtype=torch.uint8, device=device)
                    dist.broadcast(data_tensor, src=0)
                    return payload

                n_bytes = torch.tensor([0], dtype=torch.int64, device=device)
                dist.broadcast(n_bytes, src=0)
                n = int(n_bytes.item())
                if n <= 0:
                    return b""
                data_tensor = torch.empty(n, dtype=torch.uint8, device=device)
                dist.broadcast(data_tensor, src=0)
                return bytes(data_tensor.detach().cpu().tolist())

            except Exception as e:
                print(f"[NCCL] sync_prefix_cache failed: {e}")
                return prefix_data

    def allgather_prefix_keys(self, local_keys: List[str]) -> List[str]:
        """
        Gather 所有节点的 prefix keys

        Args:
            local_keys: 本地 keys

        Returns:
            所有节点的 keys
        """
        if not self.initialized:
            return local_keys

        with self._lock:
            try:
                payload = "|".join([k for k in local_keys if k]).encode("utf-8", errors="replace")
                device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                keys_len = torch.tensor([len(payload)], dtype=torch.int64, device=device)
                gathered_lens = [torch.zeros(1, dtype=torch.int64, device=device) for _ in range(self.config.world_size)]

                dist.all_gather(gathered_lens, keys_len)

                max_len = int(max(int(l.item()) for l in gathered_lens))
                padded = torch.zeros(max_len, dtype=torch.uint8, device=device)
                if len(payload) > 0:
                    padded[: len(payload)] = torch.tensor(list(payload), dtype=torch.uint8, device=device)

                gathered = [torch.zeros(max_len, dtype=torch.uint8, device=device) for _ in range(self.config.world_size)]
                dist.all_gather(gathered, padded)

                all_keys: list[str] = []
                for t, l in zip(gathered, gathered_lens):
                    n = int(l.item())
                    if n <= 0:
                        continue
                    b = bytes(t[:n].detach().cpu().tolist())
                    keys_str = b.decode("utf-8", errors="replace")
                    all_keys.extend([x for x in keys_str.split("|") if x])

                return list(set(all_keys))

            except Exception as e:
                print(f"[NCCL] allgather_prefix_keys failed: {e}")
                return local_keys

    # =========================================================================
    # Generic Collective Operations
    # =========================================================================

    def barrier(self):
        """屏障同步"""
        if self.initialized:
            dist.barrier()

    def get_rank(self) -> int:
        """获取当前 rank"""
        return self.config.rank if self.initialized else 0

    def get_world_size(self) -> int:
        """获取世界大小"""
        return self.config.world_size if self.initialized else 1

    def is_main_rank(self) -> bool:
        """是否是主节点"""
        return self.get_rank() == 0


class PDNCCLManager:
    """
    PD NCCL 管理器

    管理多机 PD 节点的 NCCL 通信
    """

    def __init__(self, config: Optional[NCCLConfig] = None):
        self.config = config or NCCLConfig()
        self.synchronizer = NCCLSynchronizer(config)

        self.local_kv_cache: Dict[str, torch.Tensor] = {}
        self.local_prefix_cache: Dict[str, bytes] = {}
        self.kda_basis: Optional[torch.Tensor] = None

    def start(self) -> bool:
        """启动 NCCL"""
        return self.synchronizer.initialize()

    def stop(self):
        """停止 NCCL"""
        self.synchronizer.finalize()

    def sync_kv_to_all(self, key: str, kv_tensor: torch.Tensor) -> torch.Tensor:
        """
        同步 KV Cache 到所有节点

        Args:
            key: KV key
            kv_tensor: KV tensor

        Returns:
            同步后的 tensor
        """
        self.local_kv_cache[key] = kv_tensor
        synced = self.synchronizer.sync_kv_cache(kv_tensor, src_rank=0)
        self.local_kv_cache[key] = synced
        return synced

    def get_global_kv(self, key: str) -> Optional[torch.Tensor]:
        """获取全局 KV Cache"""
        return self.local_kv_cache.get(key)

    def broadcast_kda_basis(self, basis_tensor: torch.Tensor, src_rank: int = 0) -> torch.Tensor:
        """
        广播 KDA 正交基

        Args:
            basis_tensor: 正交基
            src_rank: 源节点

        Returns:
            广播后的正交基
        """
        self.kda_basis = self.synchronizer.broadcast_kda_basis(basis_tensor, src_rank)
        return self.kda_basis

    def sync_prefix(self, key: str, data: bytes) -> Optional[bytes]:
        """同步 Prefix Cache"""
        self.local_prefix_cache[key] = data
        return self.synchronizer.sync_prefix_cache(key, data)
