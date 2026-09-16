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
PD Server - Prefetch Distribution Service

功能:
- KV Cache 块管理
- Prefix KV 缓存
- CGC 命令绑定 (PD ↔ Worker)
- FlashKDA 融合
- 多机分布式支持

Architecture:
    [Client: vLLM]
         ↓ gRPC
    [PD Service Cluster]
         ↓ gRPC + CGC
    [Model Worker]
"""

import asyncio
import pickle
import time
import hashlib
import os
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from concurrent import futures
import threading
import queue

from .dopd_runtime import DOPDSessionRuntime

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import grpc
    from google.protobuf import json_format
except ImportError:
    grpc = None

try:
    from ..cgc import CGCExecutor, CGC_OP_CODES, FlashKDALayer
    from ..flashkda_integration import FLASHKDA_AVAILABLE
    CGC_AVAILABLE = True
except ImportError:
    CGC_AVAILABLE = False
    FLASHKDA_AVAILABLE = False


@dataclass
class BlockInfo:
    """KV Cache Block"""
    block_id: int
    sequence_id: int
    size: int
    allocated: bool = False
    last_access: float = 0.0


@dataclass
class PrefixKVEntry:
    """Prefix KV Cache Entry"""
    key: str
    kv_data: bytes
    ttl_seconds: int
    created_at: float = field(default_factory=time.time)
    access_count: int = 0


class DistributedKVCache:
    """
    分布式 KV Cache (内存版，可替换为 Redis)
    
    支持 KV Cache 量化:
    - INT8: 50% 显存節省
    - INT4: 75% 显存節省
    """

    def __init__(self, max_blocks: int = 10000, enable_quant: bool = True, quant_bits: int = 8):
        self.max_blocks = max_blocks
        self.blocks: Dict[int, BlockInfo] = {}
        self.prefix_cache: Dict[str, PrefixKVEntry] = {}
        self.sequence_to_blocks: Dict[int, list] = {}
        self._lock = threading.RLock()
        
        # 量化支持
        self.enable_quant = enable_quant
        if enable_quant:
            try:
                from .kv_quantizer import KVQuantizer
                self.kv_quantizer = KVQuantizer(bits=quant_bits, group_size=128)
                print(f"[PD-KV] 量化已啟用: {quant_bits}bit")
            except ImportError:
                self.kv_quantizer = None
                print(f"[PD-KV] 量化模塊導入失敗，禁用量化")
        else:
            self.kv_quantizer = None
        
        # 存儲量化後的 KV
        self.quantized_kv: Dict[int, Any] = {}
        self.raw_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def allocate_blocks(self, sequence_id: int, num_blocks: int) -> list:
        """分配 block IDs"""
        with self._lock:
            allocated = []
            for block_id in range(self.max_blocks):
                if block_id not in self.blocks or not self.blocks[block_id].allocated:
                    self.blocks[block_id] = BlockInfo(
                        block_id=block_id,
                        sequence_id=sequence_id,
                        size=0,
                        allocated=True,
                        last_access=time.time(),
                    )
                    allocated.append(block_id)
                    if len(allocated) >= num_blocks:
                        break

            if sequence_id not in self.sequence_to_blocks:
                self.sequence_to_blocks[sequence_id] = []
            self.sequence_to_blocks[sequence_id].extend(allocated)

            return allocated

    def store_prefix(self, key: str, kv_data: bytes, ttl_seconds: int = 3600):
        """存储 Prefix KV"""
        with self._lock:
            self.prefix_cache[key] = PrefixKVEntry(
                key=key,
                kv_data=kv_data,
                ttl_seconds=ttl_seconds,
            )

    def get_prefix(self, key: str) -> Optional[bytes]:
        """获取 Prefix KV"""
        with self._lock:
            if key in self.prefix_cache:
                entry = self.prefix_cache[key]
                entry.access_count += 1
                age = time.time() - entry.created_at
                if age < entry.ttl_seconds:
                    return entry.kv_data
                else:
                    del self.prefix_cache[key]
            return None

    def get_stats(self) -> Dict[str, int]:
        """获取缓存统计"""
        with self._lock:
            return {
                "total_blocks": len(self.blocks),
                "allocated_blocks": sum(1 for b in self.blocks.values() if b.allocated),
                "prefix_cache_size": len(self.prefix_cache),
                "active_sequences": len(self.sequence_to_blocks),
                "quantized_kv_count": len(self.quantized_kv),
                "raw_kv_count": len(self.raw_kv),
            }

    def store_kv(self, block_id: int, k: torch.Tensor, v: torch.Tensor) -> bool:
        """存储 KV Cache (支持量化)"""
        with self._lock:
            if block_id not in self.blocks or not self.blocks[block_id].allocated:
                return False

            try:
                if self.enable_quant and self.kv_quantizer is not None:
                    quantized = self.kv_quantizer.quantize(k, v)
                    self.quantized_kv[block_id] = quantized
                    if block_id in self.raw_kv:
                        del self.raw_kv[block_id]
                else:
                    self.raw_kv[block_id] = (k, v)
                    if block_id in self.quantized_kv:
                        del self.quantized_kv[block_id]

                self.blocks[block_id].last_access = time.time()
                return True
            except Exception as e:
                print(f"[PD-KV] Store error: {e}")
                return False

    def load_kv(self, block_id: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """加载 KV Cache (支持反量化)"""
        with self._lock:
            if block_id not in self.blocks or not self.blocks[block_id].allocated:
                return None

            try:
                if block_id in self.quantized_kv and self.kv_quantizer is not None:
                    quantized = self.quantized_kv[block_id]
                    k, v = self.kv_quantizer.dequantize(quantized)
                    self.blocks[block_id].last_access = time.time()
                    return k, v
                elif block_id in self.raw_kv:
                    k, v = self.raw_kv[block_id]
                    self.blocks[block_id].last_access = time.time()
                    return k, v
                else:
                    return None
            except Exception as e:
                print(f"[PD-KV] Load error: {e}")
                return None

    def release_blocks(self, block_ids: list) -> int:
        """释放 KV Cache Blocks"""
        released = 0
        with self._lock:
            for block_id in block_ids:
                if block_id in self.blocks and self.blocks[block_id].allocated:
                    self.blocks[block_id].allocated = False
                    if block_id in self.quantized_kv:
                        del self.quantized_kv[block_id]
                    if block_id in self.raw_kv:
                        del self.raw_kv[block_id]
                    released += 1
        return released


class CGCCommandExecutor:
    """
    CGC 命令执行器 (PD 层)

    支持:
    - KDA Forward/Backward
    - KV Cache 操作
    - Prefix 处理
    """

    def __init__(self):
        self.cgc_exec: Optional[CGCExecutor] = None
        self.flashkda: Optional[FlashKDALayer] = None
        self._init_cgc()

    def _init_cgc(self):
        if CGC_AVAILABLE:
            self.cgc_exec = CGCExecutor(enable_profiling=False)
            if FLASHKDA_AVAILABLE:
                self.flashkda = FlashKDALayer()
            print(f"[PD-CGC] Initialized: CGC={CGC_AVAILABLE}, FlashKDA={FLASHKDA_AVAILABLE}")

    def execute(self, opcode: int, tensors: Dict[str, Any], params: Dict[str, Any]) -> Tuple[Any, bool, str]:
        """
        执行 CGC 命令

        Returns:
            (output, success, error_message)
        """
        if not CGC_AVAILABLE:
            return None, False, "CGC not available"

        try:
            if opcode in [CGC_OP_CODES.KDA_CHUNK, CGC_OP_CODES.KDA_FORWARD, 0x80, 0x07]:
                return self._execute_kda_forward(tensors, params)
            elif opcode == CGC_OP_CODES.KDA_ORTHO_UPDATE:
                return self._execute_kda_ortho_update(tensors, params)
            elif opcode == CGC_OP_CODES.KV_CACHE_LOAD:
                return self._execute_kv_cache_load(tensors, params)
            elif opcode == CGC_OP_CODES.KV_CACHE_STORE:
                return self._execute_kv_cache_store(tensors, params)
            elif opcode == CGC_OP_CODES.ATTENTION_SDPA:
                return self._execute_sdpa(tensors, params)
            else:
                return self._execute_generic(opcode, tensors, params)

        except Exception as e:
            return None, False, str(e)

    def _execute_kda_forward(self, tensors: Dict[str, Any], params: Dict[str, Any]) -> Tuple[Any, bool, str]:
        """KDA Forward via FlashKDA"""
        if self.flashkda is None:
            return None, False, "FlashKDA not available"

        q = tensors.get("q")
        k = tensors.get("k")
        v = tensors.get("v")
        scale = params.get("scale", 1.0)

        if q is None or k is None or v is None:
            return None, False, "Missing q, k, v tensors"

        try:
            output, _ = self.flashkda(q, k, v, scale=scale)
            return output, True, ""
        except Exception as e:
            return None, False, str(e)

    def _execute_kda_ortho_update(self, tensors: Dict[str, Any], params: Dict[str, Any]) -> Tuple[Any, bool, str]:
        """KDA Orthogonal Basis Update"""
        proj_kv = tensors.get("proj_kv")
        global_basis = tensors.get("global_basis")
        decay = params.get("decay", 0.99)

        if proj_kv is None or global_basis is None:
            return None, False, "Missing proj_kv or global_basis"

        updated_basis = decay * global_basis + (1 - decay) * proj_kv
        return updated_basis, True, ""

    def _execute_kv_cache_load(self, tensors: Dict[str, Any], params: Dict[str, Any]) -> Tuple[Any, bool, str]:
        """KV Cache Load"""
        block_ids = params.get("block_ids", [])
        return {"block_ids": block_ids, "data": None}, True, ""

    def _execute_kv_cache_store(self, tensors: Dict[str, Any], params: Dict[str, Any]) -> Tuple[Any, bool, str]:
        """KV Cache Store"""
        return {"success": True}, True, ""

    def _execute_sdpa(self, tensors: Dict[str, Any], params: Dict[str, Any]) -> Tuple[Any, bool, str]:
        """SDPA via CGC"""
        if self.cgc_exec is None:
            return None, False, "CGC executor not available"

        q = tensors.get("q")
        k = tensors.get("k")
        v = tensors.get("v")
        scale = params.get("scale", 1.0)

        from ..cgc.cgc_simd_executor import CGCCommand
        cmd = CGCCommand(
            opcode=CGC_OP_CODES.ATTENTION_SDPA,
            inputs=[q, k, v],
            outputs=[],
            params={"scale": scale},
        )

        outputs = self.cgc_exec.execute(cmd)
        return outputs[0] if outputs else None, True, ""

    def _execute_generic(self, opcode: int, tensors: Dict[str, Any], params: Dict[str, Any]) -> Tuple[Any, bool, str]:
        """Generic CGC command"""
        if self.cgc_exec is None:
            return None, False, "CGC executor not available"

        inputs = list(tensors.values())
        from ..cgc.cgc_simd_executor import CGCCommand
        cmd = CGCCommand(
            opcode=opcode,
            inputs=inputs,
            outputs=[],
            params=params,
        )

        outputs = self.cgc_exec.execute(cmd)
        return outputs[0] if outputs else None, True, ""


class PDServerServicer:
    """
    PD Service gRPC Servicer

    实现所有 PD RPC 方法
    """

    def __init__(self, max_blocks: int = 10000):
        self.kv_cache = DistributedKVCache(max_blocks=max_blocks)
        self.cgc_executor = CGCCommandExecutor()
        self.dopd_runtime = DOPDSessionRuntime()
        self.active_connections = 0
        self._lock = threading.Lock()
        self._dist = _PDDistributedSync(self.kv_cache)

        print(f"[PD Server] Initialized: max_blocks={max_blocks}")

    def AllocateBlocks(self, request, context):
        """分配 KV Cache Blocks"""
        num_blocks = request.num_blocks if request.num_blocks > 0 else 1
        sequence_id = request.sequence_ids[0] if request.sequence_ids else 0

        block_ids = self.kv_cache.allocate_blocks(sequence_id, num_blocks)

        from . import pd_service_pb2
        return pd_service_pb2.BlockResponse(
            block_ids=block_ids,
            success=True,
            error_message="",
        )

    def StorePrefixKV(self, request, context):
        """存储 Prefix KV"""
        self.kv_cache.store_prefix(
            key=request.key,
            kv_data=request.kv_data,
            ttl_seconds=request.ttl_seconds,
        )
        self._dist.enqueue_prefix_update(key=request.key, kv_data=bytes(request.kv_data), ttl_seconds=int(request.ttl_seconds))

        from . import pd_service_pb2
        return pd_service_pb2.PrefixKVResponse(success=True)

    def GetPrefixKV(self, request, context):
        """获取 Prefix KV"""
        kv_data = self.kv_cache.get_prefix(request.key)

        from . import pd_service_pb2
        return pd_service_pb2.PrefixResponse(
            kv_data=kv_data or b"",
            cache_hit=kv_data is not None,
        )

    def InvalidatePrefix(self, request, context):
        """失效 Prefix"""
        with self.kv_cache._lock:
            if request.key in self.kv_cache.prefix_cache:
                del self.kv_cache.prefix_cache[request.key]
        self._dist.enqueue_prefix_invalidate(key=request.key)

        from . import pd_service_pb2
        return pd_service_pb2.PrefixInvalidateResponse(success=True)

    def StoreKV(self, request, context):
        """存储 KV Cache"""
        try:
            k = pickle.loads(request.k_data) if request.k_data else None
            v = pickle.loads(request.v_data) if request.v_data else None
            if k is not None and v is not None:
                success = self.kv_cache.store_kv(request.block_id, k, v)
            else:
                success = False
            from . import pd_service_pb2
            return pd_service_pb2.StoreKVResponse(success=success)
        except Exception as e:
            from . import pd_service_pb2
            return pd_service_pb2.StoreKVResponse(success=False)

    def LoadKV(self, request, context):
        """加载 KV Cache"""
        try:
            result = self.kv_cache.load_kv(request.block_id)
            from . import pd_service_pb2
            if result:
                k, v = result
                return pd_service_pb2.LoadKVResponse(
                    k_data=pickle.dumps(k),
                    v_data=pickle.dumps(v),
                    success=True,
                )
            else:
                return pd_service_pb2.LoadKVResponse(
                    k_data=b"",
                    v_data=b"",
                    success=False,
                )
        except Exception as e:
            from . import pd_service_pb2
            return pd_service_pb2.LoadKVResponse(
                k_data=b"",
                v_data=b"",
                success=False,
            )

    def ReleaseKV(self, request, context):
        """释放 KV Cache"""
        try:
            released = self.kv_cache.release_blocks(list(request.block_ids))
            from . import pd_service_pb2
            return pd_service_pb2.ReleaseKVResponse(
                released_count=released,
                success=True,
            )
        except Exception as e:
            from . import pd_service_pb2
            return pd_service_pb2.ReleaseKVResponse(
                released_count=0,
                success=False,
            )

    def ExecuteCGCCommand(self, request, context):
        """执行 CGC 命令"""
        start_time = time.time()

        tensors = {}
        for k, v in request.tensors.items():
            tensors[k] = pickle.loads(v)

        params = dict(request.params) if request.params else {}

        output, success, error_msg = self.cgc_executor.execute(
            opcode=request.opcode,
            tensors=tensors,
            params=params,
        )

        output_bytes = pickle.dumps(output) if output is not None else b""
        execution_time = int((time.time() - start_time) * 1e6)

        from . import pd_service_pb2
        return pd_service_pb2.CGCCommandResponse(
            output=output_bytes,
            success=success,
            error_message=error_msg,
            execution_time_us=execution_time,
        )

    def ExecuteKDAForward(self, request, context):
        """KDA Forward (PD 层预处理)"""
        start_time = time.time()

        tensors = {}
        if request.q_data:
            tensors["q"] = pickle.loads(request.q_data)
        if request.k_data:
            tensors["k"] = pickle.loads(request.k_data)
        if request.v_data:
            tensors["v"] = pickle.loads(request.v_data)
        if request.g_data:
            tensors["g"] = pickle.loads(request.g_data)
        if request.beta_data:
            tensors["beta"] = pickle.loads(request.beta_data)

        params = {
            "scale": request.scale,
            "lower_bound": request.lower_bound,
        }

        output, success, error_msg = self.cgc_executor._execute_kda_forward(tensors, params)

        output_bytes = pickle.dumps(output) if output is not None else b""
        kernel_time = int((time.time() - start_time) * 1e6)

        from . import pd_service_pb2
        return pd_service_pb2.KDAForwardResponse(
            output_data=output_bytes,
            final_state_data=b"",
            success=success,
            kernel_time_us=kernel_time,
        )

    def UpdateKDABasis(self, request, context):
        """更新 KDA 正交基"""
        tensors = {}
        if request.proj_kv_data:
            tensors["proj_kv"] = pickle.loads(request.proj_kv_data)
        if request.global_basis_data:
            tensors["global_basis"] = pickle.loads(request.global_basis_data)

        params = {
            "decay": request.decay,
            "gram_schmidt_iter": request.gram_schmidt_iter,
        }

        updated_basis, success, error_msg = self.cgc_executor._execute_kda_ortho_update(tensors, params)

        updated_bytes = pickle.dumps(updated_basis) if updated_basis is not None else b""

        from . import pd_service_pb2
        return pd_service_pb2.KDABasisResponse(
            updated_basis_data=updated_bytes,
            success=success,
        )

    def PrepareHandoff(self, request, context):
        from . import pd_service_pb2

        try:
            record = self.dopd_runtime.prepare_handoff(
                session_id=str(request.session_id or ""),
                handoff_id=str(request.handoff_id or ""),
                source_role=str(request.source_role or ""),
                target_role=str(request.target_role or ""),
                phase_role=str(request.phase_role or ""),
                model_name=str(request.model_name or ""),
                cache_schema=str(request.cache_schema or ""),
                kv_variant=str(request.kv_variant or ""),
                transport_codec=str(request.transport_codec or ""),
                compression_codec=str(request.compression_codec or ""),
                zero_copy_vram=bool(request.zero_copy_vram),
                resume_payload=bytes(request.resume_payload or b""),
                metadata=dict(request.metadata) if request.metadata else {},
            )
            return pd_service_pb2.HandoffPrepareResponse(
                success=True,
                handoff_id=record.handoff_id,
                session_id=record.session_id,
                ack_status="prepared",
                error_message="",
            )
        except Exception as exc:
            return pd_service_pb2.HandoffPrepareResponse(
                success=False,
                handoff_id=str(request.handoff_id or ""),
                session_id=str(request.session_id or ""),
                ack_status="prepare_failed",
                error_message=str(exc),
            )

    def CommitHandoff(self, request, context):
        from . import pd_service_pb2

        try:
            record = self.dopd_runtime.commit_handoff(
                session_id=str(request.session_id or ""),
                handoff_id=str(request.handoff_id or ""),
                target_worker=str(request.target_worker or ""),
                resume_position=int(request.resume_position or 0),
                resume_payload=bytes(request.resume_payload or b""),
                metadata=dict(request.metadata) if request.metadata else {},
            )
            return pd_service_pb2.HandoffCommitResponse(
                success=True,
                ack_status="committed",
                resume_token=str(record.resume_token or ""),
                error_message="",
            )
        except Exception as exc:
            return pd_service_pb2.HandoffCommitResponse(
                success=False,
                ack_status="commit_failed",
                resume_token="",
                error_message=str(exc),
            )

    def ResumeDecode(self, request, context):
        from . import pd_service_pb2

        try:
            record = self.dopd_runtime.resume_decode(
                session_id=str(request.session_id or ""),
                handoff_id=str(request.handoff_id or ""),
                resume_token=str(request.resume_token or ""),
                worker_id=str(request.worker_id or ""),
                max_new_tokens=int(request.max_new_tokens or 0),
                resume_payload=bytes(request.resume_payload or b""),
                metadata=dict(request.metadata) if request.metadata else {},
            )
            return pd_service_pb2.ResumeDecodeResponse(
                success=record.status != "failed",
                ack_status=str(record.status or "resume_accepted"),
                worker_id=str(record.active_worker or record.target_worker or ""),
                error_message=str(record.error_message or ""),
                accepted_at_us=int(record.accepted_at_us or 0),
            )
        except Exception as exc:
            return pd_service_pb2.ResumeDecodeResponse(
                success=False,
                ack_status="resume_failed",
                worker_id=str(request.worker_id or ""),
                error_message=str(exc),
                accepted_at_us=0,
            )

    def AbortHandoff(self, request, context):
        from . import pd_service_pb2

        try:
            record = self.dopd_runtime.abort_handoff(
                session_id=str(request.session_id or ""),
                handoff_id=str(request.handoff_id or ""),
                rollback_reason=str(request.rollback_reason or ""),
            )
            return pd_service_pb2.HandoffAbortResponse(
                success=True,
                ack_status=str(record.status or "aborted"),
                error_message="",
            )
        except Exception as exc:
            return pd_service_pb2.HandoffAbortResponse(
                success=False,
                ack_status="abort_failed",
                error_message=str(exc),
            )

    def QuerySessionState(self, request, context):
        from . import pd_service_pb2

        try:
            record = self.dopd_runtime.query_session(
                session_id=str(request.session_id or ""),
                handoff_id=str(request.handoff_id or ""),
            )
            return pd_service_pb2.SessionQueryResponse(
                success=True,
                session_id=str(record.session_id or ""),
                handoff_id=str(record.handoff_id or ""),
                phase_role=str(record.phase_role or ""),
                session_status=str(record.status or ""),
                active_worker=str(record.active_worker or record.target_worker or ""),
                metadata=record.to_metadata(),
            )
        except Exception as exc:
            return pd_service_pb2.SessionQueryResponse(
                success=False,
                session_id=str(request.session_id or ""),
                handoff_id=str(request.handoff_id or ""),
                phase_role="",
                session_status="query_failed",
                active_worker="",
                metadata={"error_message": str(exc)},
            )

    def HealthCheck(self, request, context):
        """健康检查"""
        stats = self.kv_cache.get_stats()
        stats.update(self._dist.get_dist_stats())
        stats.update(self.dopd_runtime.get_stats())

        from . import pd_service_pb2
        return pd_service_pb2.HealthCheckResponse(
            healthy=True,
            status="ok",
            active_connections=self.active_connections,
            cache_stats=stats,
        )


class _PDDistributedSync:
    def __init__(self, kv_cache: DistributedKVCache):
        self.kv_cache = kv_cache
        self.enabled = bool(int(os.environ.get("CGC_PD_ENABLE_NCCL_SYNC", "0") or "0"))
        self.rank = 0
        self.world_size = 1
        self.device = None
        self._q: "queue.Queue[tuple]" = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        if not self.enabled:
            return
        if not TORCH_AVAILABLE:
            self.enabled = False
            return
        try:
            world = int(os.environ.get("WORLD_SIZE", "1") or "1")
        except Exception:
            world = 1
        if world <= 1:
            self.enabled = False
            return

        try:
            import torch.distributed as dist

            if not dist.is_available():
                self.enabled = False
                return
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")

            self.rank = int(dist.get_rank())
            self.world_size = int(dist.get_world_size())
            local_rank = int(os.environ.get("LOCAL_RANK", str(self.rank)) or self.rank)
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
                self.device = torch.device(f"cuda:{local_rank}")
            else:
                self.device = torch.device("cpu")

            self._thread = threading.Thread(target=self._loop, name="pd_nccl_sync", daemon=True)
            self._thread.start()
            print(f"[PD-NCCL] Enabled: rank={self.rank} world={self.world_size} device={self.device}")
        except Exception as e:
            print(f"[PD-NCCL] Disabled (init failed): {e}")
            self.enabled = False

    def get_dist_stats(self) -> Dict[str, int]:
        if not self.enabled:
            return {"dist_world_size": 1, "dist_rank": 0, "dist_enabled": 0}
        return {"dist_world_size": int(self.world_size), "dist_rank": int(self.rank), "dist_enabled": 1}

    def is_main(self) -> bool:
        return (not self.enabled) or int(self.rank) == 0

    def enqueue_prefix_update(self, *, key: str, kv_data: bytes, ttl_seconds: int) -> None:
        if not self.enabled:
            return
        if not self.is_main():
            return
        self._q.put(("prefix_set", str(key), bytes(kv_data), int(ttl_seconds)))

    def enqueue_prefix_invalidate(self, *, key: str) -> None:
        if not self.enabled:
            return
        if not self.is_main():
            return
        self._q.put(("prefix_del", str(key)))

    def _broadcast_int(self, v: int) -> int:
        import torch.distributed as dist
        import torch

        t = torch.tensor([int(v)], dtype=torch.int64, device=self.device)
        dist.broadcast(t, src=0)
        return int(t.item())

    def _broadcast_bytes(self, b: bytes) -> bytes:
        import torch.distributed as dist
        import torch

        if self.is_main():
            payload = bytes(b)
            n = self._broadcast_int(len(payload))
            if n <= 0:
                return b""
            t = torch.tensor(list(payload), dtype=torch.uint8, device=self.device)
            dist.broadcast(t, src=0)
            return payload

        n = self._broadcast_int(0)
        if n <= 0:
            return b""
        t = torch.empty(n, dtype=torch.uint8, device=self.device)
        dist.broadcast(t, src=0)
        return bytes(t.detach().cpu().tolist())

    def _loop(self) -> None:
        import torch.distributed as dist

        tick_s = float(os.environ.get("CGC_PD_NCCL_TICK_S", "0.05") or "0.05")
        while not self._stop.is_set():
            msg = None
            if self.is_main():
                try:
                    msg = self._q.get_nowait()
                except Exception:
                    msg = None

            kind = 0
            if msg is not None:
                kind = 1 if msg[0] == "prefix_set" else 2
            kind = self._broadcast_int(kind)

            if kind == 0:
                dist.barrier()
                time.sleep(tick_s)
                continue

            if kind == 1:
                if self.is_main():
                    _, key, kv_data, ttl_seconds = msg
                else:
                    key, kv_data, ttl_seconds = "", b"", 0
                key_b = self._broadcast_bytes(key.encode("utf-8", errors="replace") if self.is_main() else b"")
                data_b = self._broadcast_bytes(kv_data if self.is_main() else b"")
                ttl = self._broadcast_int(int(ttl_seconds) if self.is_main() else 0)
                key_s = key_b.decode("utf-8", errors="replace")
                with self.kv_cache._lock:
                    self.kv_cache.store_prefix(key_s, data_b, ttl_seconds=int(ttl))
                if not self.is_main():
                    print(f"[PD-NCCL] rank={self.rank} prefix_set key={key_s} bytes={len(data_b)} ttl={int(ttl)}")
                dist.barrier()
                continue

            if kind == 2:
                if self.is_main():
                    _, key = msg
                else:
                    key = ""
                key_b = self._broadcast_bytes(key.encode("utf-8", errors="replace") if self.is_main() else b"")
                key_s = key_b.decode("utf-8", errors="replace")
                with self.kv_cache._lock:
                    if key_s in self.kv_cache.prefix_cache:
                        del self.kv_cache.prefix_cache[key_s]
                if not self.is_main():
                    print(f"[PD-NCCL] rank={self.rank} prefix_del key={key_s}")
                dist.barrier()
                continue

            dist.barrier()
            time.sleep(tick_s)


async def serve(
    port: int = 50051,
    max_blocks: int = 10000,
    max_workers: int = 100,
):
    """
    启动 PD gRPC 服务

    Args:
        port: 服务端口
        max_blocks: 最大 KV Cache blocks
        max_workers: 最大 worker 线程数
    """
    if grpc is None:
        raise RuntimeError("grpcio not installed. Run: pip install grpcio grpcio-tools")

    from . import pd_service_pb2
    from . import pd_service_pb2_grpc

    servicer = PDServerServicer(max_blocks=max_blocks)
    rank_raw = str(os.environ.get("RANK") or "").strip()
    world_raw = str(os.environ.get("WORLD_SIZE") or "").strip()
    if str(os.environ.get("CGC_PD_ENABLE_NCCL_SYNC", "0") or "0") == "1" and world_raw.isdigit() and int(world_raw) > 1:
        if rank_raw.isdigit() and int(rank_raw) != 0:
            print(f"[PD Server] NCCL worker-only rank={int(rank_raw)} (no gRPC bind)")
            while True:
                await asyncio.sleep(3600)

    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
        ],
    )

    pd_service_pb2_grpc.add_PDServiceServicer_to_server(servicer, server)

    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)

    print(f"[PD Server] Starting on {listen_addr}")
    print(f"[PD Server] Max blocks: {max_blocks}")
    print(f"[PD Server] CGC available: {CGC_AVAILABLE}")
    print(f"[PD Server] FlashKDA available: {FLASHKDA_AVAILABLE}")

    await server.start()
    print(f"[PD Server] Started successfully!")

    await server.wait_for_termination()


def serve_sync(port: int = 50051, max_blocks: int = 10000):
    """同步版本 (用于非 async 环境)"""
    if grpc is None:
        raise RuntimeError("grpcio not installed. Run: pip install grpcio grpcio-tools")

    from . import pd_service_pb2
    from . import pd_service_pb2_grpc

    servicer = PDServerServicer(max_blocks=max_blocks)
    rank_raw = str(os.environ.get("RANK") or "").strip()
    world_raw = str(os.environ.get("WORLD_SIZE") or "").strip()
    if str(os.environ.get("CGC_PD_ENABLE_NCCL_SYNC", "0") or "0") == "1" and world_raw.isdigit() and int(world_raw) > 1:
        if rank_raw.isdigit() and int(rank_raw) != 0:
            print(f"[PD Server] NCCL worker-only rank={int(rank_raw)} (no gRPC bind)")
            while True:
                time.sleep(3600)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=100),
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
        ]
    )
    pd_service_pb2_grpc.add_PDServiceServicer_to_server(servicer, server)

    server.add_insecure_port(f"[::]:{port}")
    server.start()

    print(f"[PD Server] Started on port {port}")
    server.wait_for_termination()


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 50051
    asyncio.run(serve(port=port))
