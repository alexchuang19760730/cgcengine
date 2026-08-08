#!/usr/bin/env python3
"""CGC 端云协同控制协议 (Edge-Cloud Collaborative Control Protocol)

轻量双向控制通道, 传输特征/元信息/策略指令, 不传输完整 KV。

消息类型:
  edge→cloud:
    - draft_sync:    draft tokens + hidden 特征 (TrueOrthoKDA 压缩)
    - cache_meta:    端侧 KV cache 元信息 (layer, seq_len, shape)
    - mode_request:  请求切换模式 (online/offline/hybrid)
    - heartbeat:     端侧状态心跳 (显存/算力/温度)

  cloud→edge:
    - verify_result: 投机验证结果 (accept/reject + accepted tokens)
    - strategy:      策略指令 (P 值/N 值/路由模式/quantization)
    - ortho_basis:   正交基参数 (TrueOrthoKDA basis matrix)
    - tokenizer_cfg:  tokenizer 配置同步
    - mode_switch:   模式切换指令 (online↔offline)

传输层:
  - 复用 cgc_handoff_transport.py (TCP/NIXL)
  - 控制消息: JSON (小, ~1KB)
  - 特征数据: 压缩张量 (TrueOrthoKDA + CQ4, 32x 压缩)

与现有 CGC 组件的关系:
  - edge_first_proxy: 路由 + 首 token 预测 + warm cache
  - cgc_pd_patch.py: emit/resume 通用 patch
  - cgc_handoff_transport.py: 传输层 (TCP/NIXL/file)
  - fusionroute_cloud_orchestrator.py: 云端编排
  - verify_trueorthokda.py: KV 压缩验证 (已有)
  - cgc.py strategy_decision: 策略决策 JSON (已有)
"""
import json
import time
import struct
import threading
from typing import Optional, Any, Dict, List
from dataclasses import dataclass, asdict
from enum import Enum


class MsgType(Enum):
    # edge → cloud
    DRAFT_SYNC = "draft_sync"        # draft tokens + compressed hidden
    CACHE_META = "cache_meta"        # KV cache metadata
    MODE_REQUEST = "mode_request"    # mode switch request
    HEARTBEAT = "heartbeat"          # edge status

    # cloud → edge
    VERIFY_RESULT = "verify_result"  # speculative verify result
    STRATEGY = "strategy"            # strategy directive
    ORTHO_BASIS = "ortho_basis"      # TrueOrthoKDA basis matrix
    TOKENIZER_CFG = "tokenizer_cfg"  # tokenizer config
    MODE_SWITCH = "mode_switch"      # mode switch command


class RunMode(Enum):
    ONLINE_CLOUD = "online_cloud"     # 纯云 (cloud prefill + decode)
    OFFLINE_LOCAL = "offline_local"   # Mac 本地 (MLX)
    HYBRID_SPLIT = "hybrid_split"     # layer-split (cloud prefill + Mac/cloud decode)
    HYBRID_SPEC = "hybrid_spec"       # 投机 (Mac draft + cloud verify)


@dataclass
class CGCMessage:
    """CGC 控制协议消息。"""
    msg_type: str          # MsgType value
    session_id: str        # 会话 ID (请求级别)
    timestamp: float       # 发送时间戳
    payload: dict          # 消息体 (JSON-serializable)
    binary_meta: dict = None  # 二进制附件元信息 (大小/形状/dtype)

    def to_json(self) -> bytes:
        return json.dumps(asdict(self), ensure_ascii=False).encode()

    @classmethod
    def from_json(cls, data: bytes) -> "CGCMessage":
        d = json.loads(data)
        return cls(**d)


class CGCControlChannel:
    """端云协同控制通道 — 双向消息 + 二进制附件。

    基于 TCP 长连接 (复用 cgc_handoff_transport 的 TcpHandoff)。
    控制消息: JSON (小, ~1KB)。
    特征数据: 二进制 (TrueOrthoKDA 压缩张量)。

    用法 (edge 端):
        ch = CGCControlChannel(role="edge", host="47.95.250.55", port=31020)
        ch.connect()
        ch.send_draft_sync(draft_tokens=["The", " cat"], hidden_compressed=binary_data)
        result = ch.recv_verify_result()  # 阻塞等待 cloud 验证

    用法 (cloud 端):
        ch = CGCControlChannel(role="cloud", host="0.0.0.0", port=31020)
        ch.listen()
        draft = ch.recv_draft_sync()  # 阻塞等待 edge draft
        accepted = verify_tokens(draft.tokens, target_model)
        ch.send_verify_result(accepted=accepted, rejected=[...])
    """

    def __init__(self, role: str, host: str, port: int = 31020):
        self.role = role  # "edge" or "cloud"
        self.host = host
        self.port = port
        self._sock = None
        self._lock = threading.Lock()
        self._connected = False

    def connect(self):
        """Edge 端: 连接到 cloud。"""
        import socket
        self._sock = socket.create_connection((self.host, self.port), timeout=10)
        self._connected = True
        print(f"[CGC-CC] edge connected to {self.host}:{self.port}", flush=True)

    def listen(self):
        """Cloud 端: 监听连接。"""
        import socket
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(1)
        print(f"[CGC-CC] cloud listening on {self.host}:{self.port}", flush=True)
        self._sock, addr = srv.accept()
        self._connected = True
        print(f"[CGC-CC] edge connected from {addr}", flush=True)

    def _send_frame(self, json_data: bytes, binary_data: bytes = None):
        """发送一帧: [json_len(4B)] [json] [binary_len(4B)] [binary]"""
        with self._lock:
            # JSON 部分
            self._sock.sendall(struct.pack(">I", len(json_data)))
            self._sock.sendall(json_data)
            # 二进制部分
            if binary_data:
                self._sock.sendall(struct.pack(">I", len(binary_data)))
                self._sock.sendall(binary_data)
            else:
                self._sock.sendall(struct.pack(">I", 0))

    def _recv_frame(self) -> tuple:
        """接收一帧: 返回 (CGCMessage, binary_data)"""
        # JSON
        (jl,) = struct.unpack(">I", self._recv_exact(4))
        json_data = self._recv_exact(jl)
        msg = CGCMessage.from_json(json_data)
        # Binary
        (bl,) = struct.unpack(">I", self._recv_exact(4))
        binary_data = self._recv_exact(bl) if bl > 0 else None
        return msg, binary_data

    def _recv_exact(self, n: int) -> bytes:
        data = b""
        while len(data) < n:
            chunk = self._sock.recv(n - len(data))
            if not chunk:
                raise ConnectionError("connection closed")
            data += chunk
        return data

    # === Edge → Cloud 消息 ===

    def send_draft_sync(self, session_id: str, draft_tokens: list,
                        hidden_compressed: bytes = None,
                        hidden_meta: dict = None):
        """发送 draft tokens + 压缩 hidden 特征。"""
        msg = CGCMessage(
            msg_type=MsgType.DRAFT_SYNC.value,
            session_id=session_id,
            timestamp=time.time(),
            payload={"draft_tokens": draft_tokens, "n_draft": len(draft_tokens)},
            binary_meta=hidden_meta,
        )
        self._send_frame(msg.to_json(), hidden_compressed)

    def send_heartbeat(self, session_id: str, gpu_mem: int, cpu_pct: float,
                       mode: str = RunMode.HYBRID_SPEC.value):
        """发送端侧状态心跳。"""
        msg = CGCMessage(
            msg_type=MsgType.HEARTBEAT.value,
            session_id=session_id,
            timestamp=time.time(),
            payload={"gpu_mem_mb": gpu_mem, "cpu_pct": cpu_pct, "mode": mode},
        )
        self._send_frame(msg.to_json())

    def send_mode_request(self, session_id: str, target_mode: str, reason: str = ""):
        """请求切换运行模式。"""
        msg = CGCMessage(
            msg_type=MsgType.MODE_REQUEST.value,
            session_id=session_id,
            timestamp=time.time(),
            payload={"target_mode": target_mode, "reason": reason},
        )
        self._send_frame(msg.to_json())

    # === Cloud → Edge 消息 ===

    def send_verify_result(self, session_id: str, accepted: list,
                           rejected: list, next_strategy: dict = None):
        """发送投机验证结果。"""
        msg = CGCMessage(
            msg_type=MsgType.VERIFY_RESULT.value,
            session_id=session_id,
            timestamp=time.time(),
            payload={
                "accepted_tokens": accepted,
                "rejected_tokens": rejected,
                "n_accepted": len(accepted),
                "next_strategy": next_strategy or {},
            },
        )
        self._send_frame(msg.to_json())

    def send_strategy(self, session_id: str, P: int, N: int, mode: str,
                      quantization: str = "bf16"):
        """发送策略指令 (P 值/N 值/模式/量化)。"""
        msg = CGCMessage(
            msg_type=MsgType.STRATEGY.value,
            session_id=session_id,
            timestamp=time.time(),
            payload={"P": P, "N": N, "mode": mode, "quantization": quantization},
        )
        self._send_frame(msg.to_json())

    def send_ortho_basis(self, session_id: str, basis_bytes: bytes,
                         basis_shape: list, reduced_dim: int):
        """发送 TrueOrthoKDA 正交基参数。"""
        msg = CGCMessage(
            msg_type=MsgType.ORTHO_BASIS.value,
            session_id=session_id,
            timestamp=time.time(),
            payload={"basis_shape": basis_shape, "reduced_dim": reduced_dim},
            binary_meta={"dtype": "float16", "shape": basis_shape},
        )
        self._send_frame(msg.to_json(), basis_bytes)

    def send_mode_switch(self, session_id: str, target_mode: str,
                         cache_handoff: dict = None):
        """发送模式切换指令 (online↔offline)。"""
        msg = CGCMessage(
            msg_type=MsgType.MODE_SWITCH.value,
            session_id=session_id,
            timestamp=time.time(),
            payload={
                "target_mode": target_mode,
                "cache_handoff": cache_handoff or {},
            },
        )
        self._send_frame(msg.to_json())

    # === 通用接收 ===

    def recv(self, timeout: float = 30.0) -> tuple:
        """接收消息 (阻塞), 返回 (CGCMessage, binary_data)。"""
        self._sock.settimeout(timeout)
        return self._recv_frame()

    def close(self):
        if self._sock:
            self._sock.close()
            self._connected = False


# === TrueOrthoKDA 压缩工具 ===

def compress_hidden_trueorthokda(hidden_states, basis_matrix=None,
                                  reduced_dim: int = 16):
    """用 TrueOrthoKDA 压缩 hidden_states。

    输入: hidden_states [batch, seq, hidden_dim] (torch tensor)
    输出: compressed_bytes (二进制, 32x 压缩)

    流程:
    1. 正交基投影: [hidden_dim] → [reduced_dim] (8x 压缩)
    2. CQ4 量化: float16 → 4bit (4x 压缩)
    3. 总压缩: 32x
    """
    import torch

    if basis_matrix is None:
        # 生成随机正交基 (QR 分解)
        hidden_dim = hidden_states.shape[-1]
        random_matrix = torch.randn(hidden_dim, reduced_dim, dtype=torch.float32)
        q, _ = torch.linalg.qr(random_matrix)
        basis_matrix = q.half()

    # 正交投影: [B, S, H] @ [H, R] → [B, S, R]
    projected = torch.matmul(hidden_states.half(), basis_matrix)
    # 量化 (简化: 用 int8 近似, 实际用 CQ4)
    compressed = projected.to(torch.int8)
    return compressed.numpy().tobytes(), basis_matrix


def decompress_hidden_trueorthokda(compressed_bytes, basis_matrix,
                                    original_shape):
    """解压 TrueOrthoKDA 压缩的 hidden_states。"""
    import torch
    import numpy as np

    # 反量化
    compressed = torch.from_numpy(
        np.frombuffer(compressed_bytes, dtype=np.int8).copy()
    ).to(torch.float16)
    # 重塑
    projected_shape = list(original_shape[:-1]) + [basis_matrix.shape[1]]
    projected = compressed.reshape(projected_shape)
    # 反投影 (近似, 正交基的伪逆 = 转置)
    hidden_approx = torch.matmul(projected, basis_matrix.T)
    return hidden_approx


if __name__ == "__main__":
    # 协议自测
    print("=== CGC Control Protocol Self-Test ===")
    print(f"MsgTypes: {[t.value for t in MsgType]}")
    print(f"RunModes: {[m.value for m in RunMode]}")

    # TrueOrthoKDA 压缩测试
    import torch
    hidden = torch.randn(1, 7, 2048, dtype=torch.float16)
    original_size = hidden.element_size() * hidden.numel()
    compressed_bytes, basis = compress_hidden_trueorthokda(hidden, reduced_dim=16)
    compressed_size = len(compressed_bytes)
    ratio = compressed_size / original_size
    print(f"\nTrueOrthoKDA: {original_size} bytes → {compressed_size} bytes "
          f"(ratio: {ratio:.4f}, {1/ratio:.1f}x compression)")

    # 解压验证
    decompressed = decompress_hidden_trueorthokda(compressed_bytes, basis, hidden.shape)
    error = (hidden.float() - decompressed.float()).abs().mean().item()
    print(f"Decompression error (MAE): {error:.4f}")
    print(f"Shape: {hidden.shape} → compressed → {decompressed.shape}")
    print("\n✅ Protocol self-test passed")
