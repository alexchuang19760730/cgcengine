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
PD Client - vLLM gRPC Client for PD Service

功能:
- 连接 PD 服务
- 执行 CGC 命令
- KV Cache 操作
- Prefix Cache 操作
"""

import pickle
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from .dopd_schema import (
    DOPDResumePayloadV2,
    decode_dopd_resume_payload_v2,
    encode_dopd_resume_payload_v2,
)

try:
    import grpc
except ImportError:
    grpc = None

try:
    from ..cgc import CGC_OP_CODES
    CGC_AVAILABLE = True
except ImportError:
    CGC_AVAILABLE = False


_PD_KV_BLOCKS_MAGIC_V1 = b"CGC_PD_KV_BLOCKS_V1\x00"


def encode_pd_kv_blocks_v1(payload: Dict[str, Any]) -> bytes:
    return _PD_KV_BLOCKS_MAGIC_V1 + pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)


def decode_pd_kv_blocks_v1(data: bytes) -> Optional[Dict[str, Any]]:
    if not isinstance(data, (bytes, bytearray)):
        return None
    b = bytes(data)
    if not b.startswith(_PD_KV_BLOCKS_MAGIC_V1):
        return None
    raw = b[len(_PD_KV_BLOCKS_MAGIC_V1) :]
    try:
        obj = pickle.loads(raw) if raw else None
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def decode_pd_resume_payload(data: bytes) -> Optional[Dict[str, Any]]:
    payload = decode_dopd_resume_payload_v2(data)
    if payload is not None:
        return payload
    return decode_pd_kv_blocks_v1(data)


@dataclass
class PDClientConfig:
    """PD Client 配置"""
    address: str = "localhost:50051"
    timeout_seconds: int = 30
    max_retries: int = 3
    enable_cgc_binding: bool = True


class PDClient:
    """
    PD Service gRPC Client

    使用方式:
        from cgc_engine.pd import PDClient

        client = PDClient("localhost:50051")
        block_ids = client.allocate_blocks(sequence_ids=[1], num_blocks=4)
        client.store_prefix("prompt_key", kv_data)
        result = client.run_cgc_command(opcode=CGC_OP_CODES.KDA_CHUNK, q=q, k=k, v=v)
    """

    _instance = None

    def __init__(self, address: str = "localhost:50051", config: Optional[PDClientConfig] = None):
        if grpc is None:
            raise RuntimeError("grpcio not installed. Run: pip install grpcio")

        self.address = address
        self.config = config or PDClientConfig(address=address)

        self.channel = grpc.insecure_channel(
            address,
            options=[
                ("grpc.max_send_message_length", -1),
                ("grpc.max_receive_message_length", -1),
            ],
        )

        try:
            from . import pd_service_pb2
            from . import pd_service_pb2_grpc
            self._pd = pd_service_pb2
            self._stub = pd_service_pb2_grpc.PDServiceStub(self.channel)
            self._available = True
        except ImportError as e:
            print(f"[PD Client] Warning: Could not load proto generated code: {e}")
            self._available = False
            self._stub = None

        print(f"[PD Client] Connected to PD Service: {address}")

    @classmethod
    def get(
        cls,
        address: str = "localhost:50051",
        config: Optional[PDClientConfig] = None,
    ) -> "PDClient":
        """获取单例 PD Client"""
        if cls._instance is None:
            cls._instance = cls(address, config)
        return cls._instance

    @classmethod
    def reset(cls):
        """重置单例 (用于测试或重新连接)"""
        if cls._instance is not None:
            try:
                cls._instance.channel.close()
            except:
                pass
            cls._instance = None

    # =========================================================================
    # Block Allocation
    # =========================================================================

    def allocate_blocks(
        self,
        sequence_ids: List[int],
        num_blocks: int = 1,
        model_name: str = "default",
    ) -> Tuple[List[int], bool]:
        """
        分配 KV Cache Blocks

        Args:
            sequence_ids: 序列 IDs
            num_blocks: 分配的 block 数量
            model_name: 模型名称

        Returns:
            (block_ids, success)
        """
        if not self._available:
            return [], False

        request = self._pd.BlockRequest(
            sequence_ids=sequence_ids,
            num_blocks=num_blocks,
            model_name=model_name,
        )

        try:
            response = self._stub.AllocateBlocks(
                request,
                timeout=self.config.timeout_seconds,
            )
            return list(response.block_ids), response.success
        except grpc.RpcError as e:
            print(f"[PD Client] AllocateBlocks error: {e}")
            return [], False

    # =========================================================================
    # Prefix KV Cache
    # =========================================================================

    def store_prefix(
        self,
        key: str,
        kv_data: bytes,
        ttl_seconds: int = 3600,
        metadata: Optional[Dict[str, str]] = None,
    ) -> bool:
        """
        存储 Prefix KV

        Args:
            key: Prefix key
            kv_data: KV 数据
            ttl_seconds: TTL
            metadata: 元数据

        Returns:
            success
        """
        if not self._available:
            return False

        request = self._pd.PrefixKVRequest(
            key=key,
            kv_data=kv_data,
            ttl_seconds=ttl_seconds,
            metadata=metadata or {},
        )

        try:
            response = self._stub.StorePrefixKV(
                request,
                timeout=self.config.timeout_seconds,
            )
            return response.success
        except grpc.RpcError as e:
            print(f"[PD Client] StorePrefixKV error: {e}")
            return False

    def get_prefix(self, key: str, use_cache: bool = True) -> Tuple[bytes, bool]:
        """
        获取 Prefix KV

        Args:
            key: Prefix key
            use_cache: 是否使用缓存

        Returns:
            (kv_data, cache_hit)
        """
        if not self._available:
            return b"", False

        request = self._pd.PrefixRequest(
            key=key,
            use_cache=use_cache,
        )

        try:
            response = self._stub.GetPrefixKV(
                request,
                timeout=self.config.timeout_seconds,
            )
            return response.kv_data, response.cache_hit
        except grpc.RpcError as e:
            print(f"[PD Client] GetPrefixKV error: {e}")
            return b"", False

    def get_prefix_kv_blocks_v1(self, key: str, use_cache: bool = True) -> Tuple[Optional[Dict[str, Any]], bool]:
        kv_data, cache_hit = self.get_prefix(key, use_cache=use_cache)
        payload = decode_pd_kv_blocks_v1(kv_data) if cache_hit else None
        return payload, cache_hit

    def store_prefix_kv_blocks_v1(
        self,
        key: str,
        payload: Dict[str, Any],
        ttl_seconds: int = 3600,
        metadata: Optional[Dict[str, str]] = None,
    ) -> bool:
        return self.store_prefix(
            key=key,
            kv_data=encode_pd_kv_blocks_v1(payload),
            ttl_seconds=ttl_seconds,
            metadata=metadata,
        )

    def get_prefix_resume_payload(self, key: str, use_cache: bool = True) -> Tuple[Optional[Dict[str, Any]], bool]:
        kv_data, cache_hit = self.get_prefix(key, use_cache=use_cache)
        payload = decode_pd_resume_payload(kv_data) if cache_hit else None
        return payload, cache_hit

    def get_prefix_resume_payload_v2(
        self,
        key: str,
        use_cache: bool = True,
    ) -> Tuple[Optional[Dict[str, Any]], bool]:
        kv_data, cache_hit = self.get_prefix(key, use_cache=use_cache)
        payload = decode_dopd_resume_payload_v2(kv_data) if cache_hit else None
        return payload, cache_hit

    def store_prefix_resume_payload_v2(
        self,
        key: str,
        payload: Dict[str, Any] | DOPDResumePayloadV2,
        ttl_seconds: int = 3600,
        metadata: Optional[Dict[str, str]] = None,
    ) -> bool:
        return self.store_prefix(
            key=key,
            kv_data=encode_dopd_resume_payload_v2(payload),
            ttl_seconds=ttl_seconds,
            metadata=metadata,
        )

    def invalidate_prefix(self, key: str) -> bool:
        """失效 Prefix"""
        if not self._available:
            return False

        request = self._pd.PrefixInvalidateRequest(key=key)

        try:
            response = self._stub.InvalidatePrefix(
                request,
                timeout=self.config.timeout_seconds,
            )
            return response.success
        except grpc.RpcError as e:
            print(f"[PD Client] InvalidatePrefix error: {e}")
            return False

    # =========================================================================
    # DOPD Session / Resume Contract
    # =========================================================================

    def prepare_handoff(
        self,
        *,
        session_id: str,
        handoff_id: str,
        source_role: str,
        target_role: str,
        phase_role: str,
        model_name: str,
        cache_schema: str,
        kv_variant: str,
        transport_codec: str,
        compression_codec: str,
        zero_copy_vram: bool,
        resume_payload: bytes,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        if not self._available:
            return False, {"error": "PD Client not available"}
        request = self._pd.HandoffPrepareRequest(
            session_id=session_id,
            handoff_id=handoff_id,
            source_role=source_role,
            target_role=target_role,
            phase_role=phase_role,
            model_name=model_name,
            cache_schema=cache_schema,
            kv_variant=kv_variant,
            transport_codec=transport_codec,
            compression_codec=compression_codec,
            zero_copy_vram=zero_copy_vram,
            resume_payload=resume_payload,
            metadata=metadata or {},
        )
        try:
            response = self._stub.PrepareHandoff(request, timeout=self.config.timeout_seconds)
            return bool(response.success), {
                "session_id": response.session_id,
                "handoff_id": response.handoff_id,
                "ack_status": response.ack_status,
                "error_message": response.error_message,
            }
        except grpc.RpcError as e:
            return False, {"error": str(e)}

    def commit_handoff(
        self,
        *,
        session_id: str,
        handoff_id: str,
        target_worker: str,
        resume_position: int,
        resume_payload: bytes,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        if not self._available:
            return False, {"error": "PD Client not available"}
        request = self._pd.HandoffCommitRequest(
            session_id=session_id,
            handoff_id=handoff_id,
            target_worker=target_worker,
            resume_position=resume_position,
            resume_payload=resume_payload,
            metadata=metadata or {},
        )
        try:
            response = self._stub.CommitHandoff(request, timeout=self.config.timeout_seconds)
            return bool(response.success), {
                "ack_status": response.ack_status,
                "resume_token": response.resume_token,
                "error_message": response.error_message,
            }
        except grpc.RpcError as e:
            return False, {"error": str(e)}

    def resume_decode(
        self,
        *,
        session_id: str,
        handoff_id: str,
        resume_token: str,
        worker_id: str,
        max_new_tokens: int,
        resume_payload: bytes = b"",
        metadata: Optional[Dict[str, str]] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        if not self._available:
            return False, {"error": "PD Client not available"}
        request = self._pd.ResumeDecodeRequest(
            session_id=session_id,
            handoff_id=handoff_id,
            resume_token=resume_token,
            worker_id=worker_id,
            max_new_tokens=max_new_tokens,
            resume_payload=resume_payload,
            metadata=metadata or {},
        )
        try:
            response = self._stub.ResumeDecode(request, timeout=self.config.timeout_seconds)
            return bool(response.success), {
                "ack_status": response.ack_status,
                "worker_id": response.worker_id,
                "accepted_at_us": int(response.accepted_at_us),
                "error_message": response.error_message,
            }
        except grpc.RpcError as e:
            return False, {"error": str(e)}

    def abort_handoff(self, *, session_id: str, handoff_id: str, rollback_reason: str) -> Tuple[bool, Dict[str, Any]]:
        if not self._available:
            return False, {"error": "PD Client not available"}
        request = self._pd.HandoffAbortRequest(
            session_id=session_id,
            handoff_id=handoff_id,
            rollback_reason=rollback_reason,
        )
        try:
            response = self._stub.AbortHandoff(request, timeout=self.config.timeout_seconds)
            return bool(response.success), {
                "ack_status": response.ack_status,
                "error_message": response.error_message,
            }
        except grpc.RpcError as e:
            return False, {"error": str(e)}

    def query_session_state(
        self,
        *,
        session_id: str,
        handoff_id: str = "",
    ) -> Tuple[bool, Dict[str, Any]]:
        if not self._available:
            return False, {"error": "PD Client not available"}
        request = self._pd.SessionQueryRequest(session_id=session_id, handoff_id=handoff_id)
        try:
            response = self._stub.QuerySessionState(request, timeout=self.config.timeout_seconds)
            return bool(response.success), {
                "session_id": response.session_id,
                "handoff_id": response.handoff_id,
                "phase_role": response.phase_role,
                "session_status": response.session_status,
                "active_worker": response.active_worker,
                "metadata": dict(response.metadata),
            }
        except grpc.RpcError as e:
            return False, {"error": str(e)}

    # =========================================================================
    # CGC Command Binding
    # =========================================================================

    def run_cgc_command(
        self,
        opcode: int,
        tensors: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        command_name: str = "",
    ) -> Tuple[Any, bool, str]:
        """
        执行 CGC 命令 (远程 PD)

        Args:
            opcode: CGC 操作码
            tensors: 输入张量
            params: 参数
            command_name: 命令名称

        Returns:
            (output, success, error_message)
        """
        if not self._available:
            return None, False, "PD Client not available"

        tensors_bytes = {}
        if tensors:
            for k, v in tensors.items():
                try:
                    tensors_bytes[k] = pickle.dumps(v)
                except Exception as e:
                    print(f"[PD Client] Failed to pickle tensor {k}: {e}")

        request = self._pd.CGCCommandRequest(
            opcode=opcode,
            tensors=tensors_bytes,
            params=params or {},
            command_name=command_name or str(opcode),
        )

        try:
            response = self._stub.ExecuteCGCCommand(
                request,
                timeout=self.config.timeout_seconds,
            )

            output = pickle.loads(response.output) if response.output else None
            return output, response.success, response.error_message

        except grpc.RpcError as e:
            return None, False, str(e)

    # =========================================================================
    # KDA Operations
    # =========================================================================

    def kda_forward(
        self,
        q: Any,
        k: Any,
        v: Any,
        scale: float = 1.0,
        g: Any = None,
        beta: Any = None,
        A_log: Any = None,
        dt_bias: Any = None,
        lower_bound: float = -5.0,
        use_flashkda: bool = True,
    ) -> Tuple[Any, bool, str]:
        """
        KDA Forward (PD 层预处理)

        Returns:
            (output, success, error_message)
        """
        if not self._available:
            return None, False, "PD Client not available"

        def pickle_safe(t):
            return pickle.dumps(t) if t is not None else b""

        request = self._pd.KDAForwardRequest(
            q_data=pickle_safe(q),
            k_data=pickle_safe(k),
            v_data=pickle_safe(v),
            scale=scale,
            g_data=pickle_safe(g),
            beta_data=pickle_safe(beta),
            A_log_data=pickle_safe(A_log),
            dt_bias_data=pickle_safe(dt_bias),
            lower_bound=lower_bound,
            use_flashkda=use_flashkda,
        )

        try:
            response = self._stub.ExecuteKDAForward(
                request,
                timeout=self.config.timeout_seconds,
            )

            output = pickle.loads(response.output_data) if response.output_data else None
            return output, response.success, ""

        except grpc.RpcError as e:
            return None, False, str(e)

    def update_kda_basis(
        self,
        proj_kv: Any,
        global_basis: Any,
        decay: float = 0.99,
        gram_schmidt_iter: int = 1,
    ) -> Tuple[Any, bool, str]:
        """
        更新 KDA 正交基

        Returns:
            (updated_basis, success, error_message)
        """
        if not self._available:
            return None, False, "PD Client not available"

        request = self._pd.KDABasisRequest(
            proj_kv_data=pickle.dumps(proj_kv),
            global_basis_data=pickle.dumps(global_basis),
            decay=decay,
            gram_schmidt_iter=gram_schmidt_iter,
        )

        try:
            response = self._stub.UpdateKDABasis(
                request,
                timeout=self.config.timeout_seconds,
            )

            updated_basis = pickle.loads(response.updated_basis_data) if response.updated_basis_data else None
            return updated_basis, response.success, ""

        except grpc.RpcError as e:
            return None, False, str(e)

    # =========================================================================
    # Health Check
    # =========================================================================

    def health_check(self) -> Tuple[bool, Dict[str, Any]]:
        """
        健康检查

        Returns:
            (healthy, stats)
        """
        if not self._available:
            return False, {"error": "PD Client not available"}

        request = self._pd.HealthCheckRequest(client_id=str(id(self)))

        try:
            response = self._stub.HealthCheck(
                request,
                timeout=5,
            )
            cache_stats = dict(response.cache_stats)
            return response.healthy, {
                "status": response.status,
                "active_connections": response.active_connections,
                "cache_stats": cache_stats,
                "dopd_sessions": cache_stats.get("dopd_sessions"),
                "dopd_handoffs": cache_stats.get("dopd_handoffs"),
                "dopd_active_handoffs": cache_stats.get("dopd_active_handoffs"),
            }
        except grpc.RpcError as e:
            return False, {"error": str(e)}

    def close(self):
        """关闭连接"""
        if self.channel:
            self.channel.close()
        PDClient._instance = None


# =============================================================================
# Convenience Functions
# =============================================================================

def get_pd_client(address: str = "localhost:50051") -> PDClient:
    """获取 PD Client 单例"""
    return PDClient.get(address)


def create_pd_client(address: str = "localhost:50051") -> PDClient:
    """创建新的 PD Client"""
    return PDClient(address)
